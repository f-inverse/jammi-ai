//! The `model_artifacts` catalog entity: one row per bundle of bytes under
//! `models/`, the peer of a `result_tables` row.
//!
//! The row owns three facts about the bytes and nothing else does:
//!
//! - **Existence.** A row is staged ([`Catalog::stage_model_artifact`])
//!   before the bundle's first byte is written, so every byte under `models/`
//!   a live writer produced is named by a row.
//! - **Reference.** A `models` row names an artifact through the
//!   `models.artifact_prefix` foreign key. "Referenced" is `EXISTS (SELECT 1
//!   FROM models WHERE artifact_prefix = $1)` — evaluated across every tenant,
//!   disclosed as a boolean only.
//! - **Deletability.** The bytes may be deleted only by the holder of a
//!   [`ReclaimLicence`], and the ONLY constructor of that type is the
//!   compare-and-set in this module that moves the row to
//!   [`ArtifactState::Reclaiming`] while no reference exists. The check and
//!   the commitment to delete are one statement in one `Serializable`
//!   transaction, so a reference can never attach to an artifact whose delete
//!   is already licensed, and a licence can never be minted for an artifact a
//!   reference is attaching to.
//!
//! A `staged` artifact is additionally protected while its stager is live: an
//! attempt-scoped bundle while its job is `running` that exact attempt, a
//! job-scoped bundle while its job is non-terminal. The stager itself holds a
//! [`StagedArtifact`] — the typed proof of being that writer — and may reclaim
//! its own staged bundle at any time.

use object_store::path::Path as ObjectPath;
use serde::{Deserialize, Serialize};

use crate::catalog::backend::{BackendError, SqlNullType, SqlValue, TxOptions};
use crate::catalog::lease::{canonical_stamp_now, stale_before_clause, CanonicalStampColumn};
use crate::catalog::status::{ArtifactState, JobStatus};
use crate::error::{JammiError, Result};
use crate::storage::StorageUrl;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

use super::Catalog;

/// The identity of one model artifact: the prefix its bundle lives under.
///
/// A reference names bytes, never a `models` row — several rows may name one
/// artifact. It grants nothing: holding one cannot delete a byte (only a
/// [`ReclaimLicence`] can), so it is freely parseable from its string form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArtifactRef(StorageUrl);

impl ArtifactRef {
    /// Parse an artifact reference from its string form (a storage URL).
    pub fn parse(input: &str) -> Result<Self> {
        Ok(Self(StorageUrl::parse(input)?))
    }

    /// The prefix the artifact's bundle lives under.
    pub fn url(&self) -> &StorageUrl {
        &self.0
    }

    pub(crate) fn from_url(url: StorageUrl) -> Self {
        Self(url)
    }
}

impl std::fmt::Display for ArtifactRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Who is writing a `staged` artifact — the identity its liveness is read
/// from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagingScope {
    /// A bundle one attempt of a job writes (the served artifact, an epoch
    /// checkpoint). Live while the job is `running` exactly this attempt.
    Attempt {
        /// The staging job.
        job_id: String,
        /// The staging attempt (`jobs.attempts` at claim time).
        attempt: u32,
    },
    /// A bundle every attempt of a job shares (the durable resume
    /// checkpoint). Live while the job is non-terminal.
    Job {
        /// The staging job.
        job_id: String,
    },
}

impl StagingScope {
    /// The staging job's id.
    pub fn job_id(&self) -> &str {
        match self {
            Self::Attempt { job_id, .. } | Self::Job { job_id } => job_id,
        }
    }

    /// The staging attempt, for an attempt-scoped bundle.
    pub fn attempt(&self) -> Option<u32> {
        match self {
            Self::Attempt { attempt, .. } => Some(*attempt),
            Self::Job { .. } => None,
        }
    }

    fn attempt_value(&self) -> SqlValue<'static> {
        match self.attempt() {
            Some(attempt) => SqlValue::Int(i64::from(attempt)),
            None => SqlValue::Null(SqlNullType::Int),
        }
    }
}

/// The stager's handle on an artifact it staged: proof of being that writer.
///
/// Returned by [`Catalog::stage_model_artifact`] and recovered for a whole
/// attempt by [`Catalog::staged_artifacts_of_attempt`]. It is what a finalize
/// publishes and what [`Catalog::reclaim_own_staged_artifact`] accepts; it is
/// deliberately not `Clone`, so publishing or reclaiming consumes the claim.
#[derive(Debug, PartialEq, Eq)]
pub struct StagedArtifact {
    artifact: ArtifactRef,
    scope: StagingScope,
}

impl StagedArtifact {
    /// The staged artifact's identity.
    pub fn artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    /// Who staged it.
    pub fn scope(&self) -> &StagingScope {
        &self.scope
    }
}

/// The licence to delete one artifact's bytes.
///
/// There is no public constructor and no `Clone`: the only way to obtain one
/// is to win the reclaim compare-and-set ([`Catalog::begin_artifact_reclaim`]
/// / [`Catalog::reclaim_own_staged_artifact`]), which has already committed
/// the artifact to [`ArtifactState::Reclaiming`] with no `models` row
/// referencing it.
///
/// ```compile_fail
/// use jammi_db::catalog::artifact_repo::{ArtifactRef, ReclaimLicence};
/// let artifact = ArtifactRef::parse("file:///store/models/_global/job/w/1").unwrap();
/// // The fields are private to the module that runs the compare-and-set.
/// let forged = ReclaimLicence { artifact, prefix_path: Default::default() };
/// ```
#[derive(Debug)]
pub struct ReclaimLicence {
    artifact: ArtifactRef,
    prefix_path: ObjectPath,
}

impl ReclaimLicence {
    fn mint(artifact: ArtifactRef) -> Result<Self> {
        let prefix_path = artifact.url().object_key(artifact.url().path())?;
        Ok(Self {
            artifact,
            prefix_path,
        })
    }

    /// The artifact this licence reclaims.
    pub fn artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    /// Whether `path` is one of this artifact's own objects: a key DIRECTLY
    /// inside the artifact's prefix. A bundle is a flat directory, and a
    /// bundle nested beneath it (an epoch checkpoint under its attempt's
    /// prefix) is a separate artifact with its own row — so a key any deeper
    /// is never covered, whatever this artifact's own state.
    pub fn covers(&self, path: &ObjectPath) -> bool {
        path.prefix_match(&self.prefix_path)
            .is_some_and(|mut rest| rest.next().is_some() && rest.next().is_none())
    }
}

/// What a reclaim compare-and-set decided. A refusal names its reason and
/// nothing else — in particular never which row, or which tenant's row,
/// references the artifact.
#[derive(Debug)]
pub enum ReclaimDecision {
    /// The artifact is now [`ArtifactState::Reclaiming`]; the licence deletes
    /// its bytes.
    Licensed(ReclaimLicence),
    /// At least one `models` row, in some tenant, references the artifact.
    Referenced,
    /// The artifact is `staged` and its stager is still live.
    Live,
    /// No artifact row matches in the caller's tenant scope.
    Absent,
}

/// A `model_artifacts` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactRecord {
    /// The artifact's identity.
    pub artifact: ArtifactRef,
    /// The owning tenant (`None` for a global artifact).
    pub tenant_id: Option<TenantId>,
    /// Lifecycle state.
    pub state: ArtifactState,
    /// The materialization-contract definition hash of the bytes, once
    /// published with one.
    pub definition_hash: Option<String>,
    /// The materialization-contract input anchors, as canonical JSON.
    pub input_anchors_json: Option<String>,
    /// The stager's identity (`None` for an artifact adopted from a listing
    /// or carried over from a catalog that predates staging).
    pub staging: Option<StagingScope>,
    /// Canonical-stamp creation time — the grace clock.
    pub created_at: String,
}

const SELECT_COLS: &str = "prefix, tenant_id, state, definition_hash, input_anchors_json, \
                           staging_job_id, staging_attempt, created_at";

/// The one reference predicate, correlated to the `model_artifacts` row a
/// statement is visiting. Evaluated with no tenant filter: a reference in any
/// tenant keeps the bytes.
const REFERENCED: &str =
    "EXISTS (SELECT 1 FROM models WHERE models.artifact_prefix = model_artifacts.prefix)";

/// Whether the stager of the visited `staged` row is still live: an
/// attempt-scoped bundle while its job runs that attempt, a job-scoped one
/// while its job is non-terminal.
fn stager_live_clause() -> String {
    let running = JobStatus::Running;
    let non_terminal = JobStatus::non_terminal_sql_list();
    format!(
        "EXISTS (SELECT 1 FROM jobs \
                 WHERE jobs.job_id = model_artifacts.staging_job_id \
                   AND ((model_artifacts.staging_attempt IS NULL \
                         AND jobs.status IN ({non_terminal})) \
                     OR (model_artifacts.staging_attempt IS NOT NULL \
                         AND jobs.status = '{running}' \
                         AND jobs.attempts = model_artifacts.staging_attempt)))"
    )
}

/// The strict tenant predicate over `model_artifacts`, binding the caller's
/// tenant as the next parameter — or nothing at all under admin scope, which
/// spans every tenant. Reclaim is a write: a tenant-bound caller reclaims
/// only its own artifacts, never a global or a peer's.
fn tenant_clause(tenant: Option<TenantId>, params: &mut Vec<SqlValue<'static>>) -> String {
    if TenantBinding::is_admin_scope() {
        return "1 = 1".to_string();
    }
    params.push(SqlValue::from(tenant.map(|t| t.to_string())));
    let n = params.len();
    format!("(tenant_id = ${n} OR (tenant_id IS NULL AND ${n} IS NULL))")
}

impl Catalog {
    /// Write the `staged` row for the bundle about to be written under
    /// `prefix`, owned by the catalog's bound tenant — BEFORE its first byte.
    ///
    /// Re-staging the same prefix by the same stager is idempotent (a
    /// job-scoped bundle is overwritten in place, epoch after epoch). A
    /// prefix already held in any other state, or staged by another writer,
    /// is refused: bytes must never be written into an artifact that is
    /// published or being reclaimed.
    pub(crate) async fn stage_model_artifact(
        &self,
        prefix: &StorageUrl,
        scope: StagingScope,
    ) -> Result<StagedArtifact> {
        let tenant = self.current_tenant();
        let artifact = ArtifactRef::from_url(prefix.clone());
        let staged = ArtifactState::Staged.as_db_str();
        let prefix_value = prefix.as_str().to_string();
        let job_id = scope.job_id().to_string();
        let attempt = scope.attempt_value();
        let held_by_stager = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "model_artifacts")?;
                    tx.execute(
                        "INSERT INTO model_artifacts \
                             (prefix, tenant_id, state, staging_job_id, staging_attempt, created_at) \
                         VALUES ($1, $2, $3, $4, $5, $6) \
                         ON CONFLICT(prefix) DO NOTHING",
                        &[
                            SqlValue::TextOwned(prefix_value.clone()),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                            SqlValue::Text(staged),
                            SqlValue::TextOwned(job_id.clone()),
                            attempt.clone(),
                            SqlValue::TextOwned(canonical_stamp_now()),
                        ],
                    )
                    .await?;
                    let held = tx
                        .query_opt(
                            "SELECT 1 AS one FROM model_artifacts \
                             WHERE prefix = $1 AND state = $2 AND staging_job_id = $3 \
                               AND (staging_attempt = $4 \
                                    OR (staging_attempt IS NULL AND $4 IS NULL))",
                            &[
                                SqlValue::TextOwned(prefix_value),
                                SqlValue::Text(staged),
                                SqlValue::TextOwned(job_id),
                                attempt,
                            ],
                            |row| row.get::<i32>("one"),
                        )
                        .await?;
                    Ok(held.is_some())
                })
            })
            .await?;
        if !held_by_stager {
            return Err(JammiError::Catalog(format!(
                "cannot stage model artifact '{artifact}': the prefix is already held by \
                 another writer or is no longer staged"
            )));
        }
        Ok(StagedArtifact { artifact, scope })
    }

    /// Every artifact `attempt` of `job_id` staged and has neither published
    /// nor reclaimed — the handles an attempt that is giving up reclaims its
    /// own bytes through, recovered from the catalog rather than from
    /// whatever the attempt still holds in memory.
    pub async fn staged_artifacts_of_attempt(
        &self,
        job_id: &str,
        attempt: u32,
    ) -> Result<Vec<StagedArtifact>> {
        let job_id = job_id.to_string();
        let scope_job = job_id.clone();
        let prefixes = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT prefix FROM model_artifacts \
                             WHERE staging_job_id = $1 AND staging_attempt = $2 AND state = $3 \
                             ORDER BY prefix",
                            &[
                                SqlValue::TextOwned(job_id),
                                SqlValue::Int(i64::from(attempt)),
                                SqlValue::Text(ArtifactState::Staged.as_db_str()),
                            ],
                            |row| row.get::<String>("prefix"),
                        )
                        .await
                    })
                },
            )
            .await?;
        prefixes
            .iter()
            .map(|prefix| {
                Ok(StagedArtifact {
                    artifact: ArtifactRef::parse(prefix)?,
                    scope: StagingScope::Attempt {
                        job_id: scope_job.clone(),
                        attempt,
                    },
                })
            })
            .collect()
    }

    /// The reclaim compare-and-set for an artifact the caller does not own:
    /// `→ reclaiming` while no `models` row references it and, if it is still
    /// `staged`, its stager is no longer live. An artifact already
    /// `reclaiming` is licensed again, so an interrupted reclaim resumes.
    ///
    /// Tenant-strict unless the caller runs under admin scope; the reference
    /// check itself is always cross-tenant.
    pub async fn begin_artifact_reclaim(&self, artifact: &ArtifactRef) -> Result<ReclaimDecision> {
        self.reclaim_cas(artifact, None).await
    }

    /// The reclaim compare-and-set for the stager's own bundle: the
    /// [`StagedArtifact`] is the proof of being the writer, so the liveness
    /// protection — which exists to keep everyone ELSE off a live writer's
    /// bytes — does not apply. The reference check still does.
    pub async fn reclaim_own_staged_artifact(
        &self,
        staged: StagedArtifact,
    ) -> Result<ReclaimDecision> {
        self.reclaim_cas(&staged.artifact, Some(&staged.scope))
            .await
    }

    async fn reclaim_cas(
        &self,
        artifact: &ArtifactRef,
        owner: Option<&StagingScope>,
    ) -> Result<ReclaimDecision> {
        let tenant = self.current_tenant();
        let staged = ArtifactState::Staged.as_db_str();
        let published = ArtifactState::Published.as_db_str();
        let reclaiming = ArtifactState::Reclaiming.as_db_str();
        let live = stager_live_clause();
        // `$2`/`$3` are the owner's staging identity; with no owner they bind
        // NULLs, so the owner arm is simply never true.
        let (owner_job, owner_attempt) = match owner {
            Some(scope) => (
                SqlValue::TextOwned(scope.job_id().to_string()),
                scope.attempt_value(),
            ),
            None => (
                SqlValue::Null(SqlNullType::Text),
                SqlValue::Null(SqlNullType::Int),
            ),
        };
        let mut cas_params = vec![
            SqlValue::TextOwned(artifact.url().as_str().to_string()),
            owner_job,
            owner_attempt,
        ];
        let scoped = tenant_clause(tenant, &mut cas_params);
        let cas = format!(
            "UPDATE model_artifacts SET state = '{reclaiming}' \
             WHERE prefix = $1 AND {scoped} AND NOT {REFERENCED} \
               AND (state IN ('{published}', '{reclaiming}') \
                    OR (state = '{staged}' \
                        AND (NOT {live} \
                             OR (staging_job_id = $2 \
                                 AND (staging_attempt = $3 \
                                      OR (staging_attempt IS NULL AND $3 IS NULL))))))"
        );
        let mut diagnose_params = vec![SqlValue::TextOwned(artifact.url().as_str().to_string())];
        let scoped = tenant_clause(tenant, &mut diagnose_params);
        let diagnose = format!(
            "SELECT {REFERENCED} AS referenced FROM model_artifacts \
             WHERE prefix = $1 AND {scoped}"
        );

        let outcome = self
            .backend()
            .serializable(|tx| {
                let cas = cas.clone();
                let diagnose = diagnose.clone();
                let cas_params = cas_params.clone();
                let diagnose_params = diagnose_params.clone();
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    if tx.execute(&cas, &cas_params).await? == 1 {
                        return Ok(CasOutcome::Won);
                    }
                    let referenced = tx
                        .query_opt(&diagnose, &diagnose_params, |row| {
                            row.get::<bool>("referenced")
                        })
                        .await?;
                    Ok(match referenced {
                        None => CasOutcome::Absent,
                        Some(true) => CasOutcome::Referenced,
                        Some(false) => CasOutcome::Live,
                    })
                })
            })
            .await?;
        Ok(match outcome {
            CasOutcome::Won => ReclaimDecision::Licensed(ReclaimLicence::mint(artifact.clone())?),
            CasOutcome::Referenced => ReclaimDecision::Referenced,
            CasOutcome::Live => ReclaimDecision::Live,
            CasOutcome::Absent => ReclaimDecision::Absent,
        })
    }

    /// Retire a reclaimed artifact's row once its bytes are gone, consuming
    /// the licence. The foreign key is the backstop: a row a `models` row
    /// still named could not be deleted — and none can, because nothing
    /// attaches to a `reclaiming` artifact.
    pub(crate) async fn retire_reclaimed_artifact(&self, licence: ReclaimLicence) -> Result<()> {
        let prefix = licence.artifact.url().as_str().to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM model_artifacts WHERE prefix = $1 AND state = $2",
                        &[
                            SqlValue::TextOwned(prefix),
                            SqlValue::Text(ArtifactState::Reclaiming.as_db_str()),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// One artifact's row, or `None`. Reads across every tenant: an artifact
    /// is named by its full prefix, which already carries its tenant segment.
    pub async fn get_model_artifact(
        &self,
        artifact: &ArtifactRef,
    ) -> Result<Option<ModelArtifactRecord>> {
        let prefix = artifact.url().as_str().to_string();
        let sql = format!("SELECT {SELECT_COLS} FROM model_artifacts WHERE prefix = $1");
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(&sql, &[SqlValue::TextOwned(prefix)], parse_artifact_row)
                            .await
                    })
                },
            )
            .await?)
    }

    /// Whether any `models` row, in any tenant, references `artifact` — a
    /// boolean and nothing more.
    pub async fn model_artifact_is_referenced(&self, artifact: &ArtifactRef) -> Result<bool> {
        let prefix = artifact.url().as_str().to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT 1 AS one FROM models WHERE artifact_prefix = $1 LIMIT 1",
                            &[SqlValue::TextOwned(prefix)],
                            |row| row.get::<i32>("one"),
                        )
                        .await
                    })
                },
            )
            .await?
            .is_some())
    }

    /// Every artifact a reconcile pass should consider, in the caller's tenant
    /// scope (strict; every tenant under admin scope), each tagged with
    /// whether it is reclaimable right now and whether it has aged past
    /// `grace` on the backend's own clock.
    pub async fn list_model_artifacts_for_reconcile(
        &self,
        grace: std::time::Duration,
    ) -> Result<Vec<ReconcileArtifact>> {
        let tenant = self.current_tenant();
        let kind = self.backend().backend_kind();
        let mut params = Vec::new();
        let scoped = tenant_clause(tenant, &mut params);
        let aged = stale_before_clause(
            CanonicalStampColumn::ModelArtifactsCreatedAt,
            None,
            kind,
            grace,
            &mut params,
        );
        let live = stager_live_clause();
        let staged = ArtifactState::Staged.as_db_str();
        let sql = format!(
            "SELECT {SELECT_COLS}, \
                    {REFERENCED} AS referenced, \
                    (state = '{staged}' AND {live}) AS live, \
                    ({aged}) AS aged \
             FROM model_artifacts WHERE {scoped} ORDER BY prefix"
        );
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(&sql, &params, |row| {
                            Ok(ReconcileArtifact {
                                record: parse_artifact_row(row)?,
                                referenced: row.get::<bool>("referenced")?,
                                live: row.get::<bool>("live")?,
                                aged: row.get::<bool>("aged")?,
                            })
                        })
                        .await
                    })
                },
            )
            .await?)
    }
}

/// One artifact as a reconcile pass sees it: the row plus the three facts the
/// pass classifies on, all evaluated by the catalog in one statement.
#[derive(Debug, Clone)]
pub struct ReconcileArtifact {
    /// The artifact's row.
    pub record: ModelArtifactRecord,
    /// Some `models` row, in some tenant, references it.
    pub referenced: bool,
    /// It is `staged` and its stager is still live.
    pub live: bool,
    /// It is older than the pass's grace.
    pub aged: bool,
}

enum CasOutcome {
    Won,
    Referenced,
    Live,
    Absent,
}

fn parse_artifact_row(
    row: &crate::catalog::backend::Row<'_>,
) -> std::result::Result<ModelArtifactRecord, BackendError> {
    let conversion = |column: &str, detail: String| BackendError::TypeConversion {
        column: column.to_string(),
        detail,
    };
    let prefix: String = row.get("prefix")?;
    let artifact = StorageUrl::parse(&prefix)
        .map(ArtifactRef::from_url)
        .map_err(|e| conversion("prefix", e.to_string()))?;
    let tenant_id = row
        .try_get::<String>("tenant_id")?
        .map(|t| t.parse::<TenantId>())
        .transpose()
        .map_err(|e| conversion("tenant_id", e.to_string()))?;
    let state = row
        .get::<String>("state")?
        .parse::<ArtifactState>()
        .map_err(|e| conversion("state", e.to_string()))?;
    let staging_attempt = row
        .try_get::<i64>("staging_attempt")?
        .map(u32::try_from)
        .transpose()
        .map_err(|e| conversion("staging_attempt", e.to_string()))?;
    let staging = row
        .try_get::<String>("staging_job_id")?
        .map(|job_id| match staging_attempt {
            Some(attempt) => StagingScope::Attempt { job_id, attempt },
            None => StagingScope::Job { job_id },
        });
    Ok(ModelArtifactRecord {
        artifact,
        tenant_id,
        state,
        definition_hash: row.try_get("definition_hash")?,
        input_anchors_json: row.try_get("input_anchors_json")?,
        staging,
        created_at: row.get("created_at")?,
    })
}
