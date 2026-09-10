use std::str::FromStr;
use std::time::Duration;

use crate::catalog::backend::{BackendError, BackendKind, Row, SqlValue, Transaction, TxOptions};
#[cfg(feature = "test-hooks")]
use crate::catalog::lease::LEASE_TS_FORMAT;
use crate::catalog::lease::{lease_deadline_expr, lease_expired_clause};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::config::StoragePrecision;
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// Whether a result table is a direct model output or a derivation of another
/// result table.
///
/// The discriminator is orthogonal to [`ModelTask`]: a [`Model`](Self::Model)
/// row's `task` names a genuine model task and resolves as an embedding or
/// inference output; a derivation row's `task` still names the source
/// embedding's task (so it round-trips through the same column) but the kind
/// excludes it from embedding-table resolution. Keeping the distinction here
/// rather than in `ModelTask` leaves that enum a pristine catalogue of model
/// tasks (S9 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ResultTableKind {
    /// An embedding or inference table produced by running a model.
    Model,
    /// A k-nearest-neighbour edge relation derived from an embedding table.
    NeighborGraph,
    /// A relational table produced by an as-of temporal join of two relations.
    /// Like [`NeighborGraph`](Self::NeighborGraph) it is a derivation that
    /// carries no ANN sidecar and is excluded from embedding-table resolution —
    /// it is data of record, not a search structure.
    AsofJoin,
}

impl ResultTableKind {
    /// Canonical string stored in the `result_tables.kind` column. The single
    /// source of truth — [`try_from_db_str`](Self::try_from_db_str) decodes it.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::NeighborGraph => "neighbor_graph",
            Self::AsofJoin => "asof_join",
        }
    }

    /// Decode the canonical string back into a [`ResultTableKind`]. Unknown
    /// spellings raise [`JammiError::Catalog`] naming the offending value.
    pub fn try_from_db_str(s: &str) -> Result<Self> {
        match s {
            "model" => Ok(Self::Model),
            "neighbor_graph" => Ok(Self::NeighborGraph),
            "asof_join" => Ok(Self::AsofJoin),
            other => Err(JammiError::Catalog(format!(
                "Unknown result-table kind '{other}'. Expected: model, neighbor_graph, asof_join"
            ))),
        }
    }
}

/// Parameters for creating a new result table entry.
/// `status` defaults to `'building'` via SQL DEFAULT — not passed here.
#[derive(Debug)]
pub struct CreateResultTableParams<'a> {
    pub table_name: &'a str,
    pub source_id: &'a str,
    pub model_id: &'a str,
    pub task: ModelTask,
    pub kind: ResultTableKind,
    pub derived_from: Option<&'a str>,
    pub parquet_path: &'a str,
    pub dimensions: Option<i32>,
    pub key_column: Option<&'a str>,
    pub text_columns: Option<&'a str>,
    /// The precision this table's sidecar index is built at — the deployment's
    /// [`crate::config::AnnIndexConfig::storage_precision`] default at the
    /// moment of creation, stamped once and read back verbatim by every later
    /// build/load of this table's index. Recorded even for a non-embedding row
    /// (one that never grows an index segment), so the column stays a total
    /// function of "when was this row created", not a value that only sometimes
    /// exists.
    pub storage_precision: StoragePrecision,
    /// The per-table default retrieve→rescore oversample multiplier — the
    /// deployment's [`crate::config::AnnIndexConfig::effective_oversample`] at
    /// creation time. A search request may still override it for one call.
    pub oversample: usize,
    /// The row's creation timestamp, stamped by the caller via
    /// [`crate::catalog::backend::now_sortable`] rather than left to a SQL
    /// `DEFAULT` — a backend-computed default would give SQLite and Postgres
    /// different resolutions and shapes for the same column, which is exactly
    /// the ordering-key parity bug [`Catalog::resolve_embedding_table`] used
    /// to hit. Threading it through here means every INSERT of this table
    /// binds one app-supplied, backend-identical value.
    pub created_at: String,
    /// The writer creating the row — the `ResultStore` instance's
    /// `writer-{uuid}` — stamped into `writer_id` so every later transition on
    /// the `building` row is a compare-and-set naming it. `None` seeds a row
    /// with no writer (a test fixture, or a caller registering a table it does
    /// not build): recovery reads the absent lease as "no live writer".
    pub writer_id: Option<&'a str>,
    /// The writer's initial lease WINDOW (renewed by its heartbeat to the
    /// same window every time) — a duration, not a timestamp: the deadline
    /// this stamps is `now() + lease` evaluated by the catalog backend's OWN
    /// clock on Postgres ([`lease_deadline_expr`]), never this process's
    /// clock. `None` when `writer_id` is `None`.
    pub lease: Option<Duration>,
    /// The job attempt this table is being materialised for (N11), or `None`
    /// for a table created outside the job machinery (a test fixture, a
    /// direct `register_table` path). When `Some`, [`Catalog::create_result_table`]
    /// performs the `jobs.partial_result` compare-and-set — `UPDATE jobs SET
    /// partial_result = table_name WHERE job_id = $1 AND claimed_by = $2 AND
    /// status = 'running' AND attempts = $3 AND partial_result IS NULL` —
    /// inside the SAME transaction as this row's own INSERT, so the two
    /// either land together or neither lands at all. A zero-row CAS (the
    /// caller's attempt is no longer the current lease holder, or a peer
    /// already recorded a `partial_result`) rolls the whole transaction back
    /// and returns [`JammiError::JobAttemptSuperseded`] — no `result_tables`
    /// row and no bytes are ever committed for a superseded attempt.
    ///
    /// Bundled as one [`JobAttempt`] rather than three parallel `Option`
    /// fields (esc-107): a `job_id`-only guard is unsound on its own — see
    /// [`Catalog::create_result_table`]'s doc for the exact race a
    /// `job_id`-only CAS admits. Reshaping the type so `job_id` cannot be
    /// supplied without the attempt identity the CAS needs makes that
    /// half-supplied state unrepresentable, rather than trusting every call
    /// site to remember to pass all three together.
    pub job_attempt: Option<JobAttempt<'a>>,
}

/// The exact job-attempt identity [`CreateResultTableParams::job_attempt`]'s
/// CAS matches against `jobs` — the same `(job_id, claimed_by, attempts)`
/// triple every other lease-guarded write in `crate::catalog::jobs_repo`
/// requires, so a superseded attempt's call can never win the CAS just
/// because the job happens to still be `running` under a LATER attempt.
#[derive(Debug, Clone, Copy)]
pub struct JobAttempt<'a> {
    pub job_id: &'a str,
    /// The instance the CAS matches against `jobs.claimed_by`.
    pub instance_id: &'a str,
    /// The attempt the CAS matches against `jobs.attempts`.
    pub attempts: u32,
}

/// A row from the `result_tables` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResultTableRecord {
    pub table_name: String,
    pub source_id: String,
    pub model_id: String,
    pub task: ModelTask,
    pub kind: ResultTableKind,
    pub derived_from: Option<String>,
    pub parquet_path: String,
    pub dimensions: Option<i32>,
    pub distance_metric: String,
    pub row_count: usize,
    pub status: String,
    pub key_column: Option<String>,
    pub text_columns: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
    /// Owning tenant, or `None` for a GLOBAL (shared) table. Recovery reads
    /// this to re-scope a cross-tenant reconciliation back to the row's own
    /// tenant when it enumerates orphans under an admin scan; tenant-scoped
    /// callers see only their own rows, so for them this is always either
    /// their tenant or `None`.
    pub tenant_id: Option<String>,
    /// The materialization-contract definition hash — the indexable summary of
    /// the `.materialization.json` sidecar's `definition_hash`. `None` for a
    /// pre-contract table (created before migration 021), which verifies as
    /// [`crate::store::manifest::MatchVerdict::MissingManifest`].
    pub definition_hash: Option<String>,
    /// The materialization-contract input anchors as canonical JSON — the
    /// indexable summary of the sidecar's `input_anchors`. `None` for a
    /// pre-contract table.
    pub input_anchors_json: Option<String>,
    /// The precision this table's sidecar index was built at, stamped once at
    /// creation from the deployment default then carried unchanged across
    /// every later index build/load of this table (including a
    /// crash-recovery rebuild) — reading it here rather than the *current*
    /// deployment config is what keeps a recovery rebuild from silently
    /// landing at a different precision than the table's existing catalog
    /// promise. `None` for a row created before migration 023, read as `F32`
    /// by every consumer (the precision every pre-migration index was built
    /// at, since quantization did not exist yet).
    pub storage_precision: Option<StoragePrecision>,
    /// The per-table default retrieve→rescore oversample multiplier, stamped
    /// once at creation from the deployment default then carried unchanged
    /// across every later rescore of this table — reading it here rather
    /// than the *current* deployment config is what keeps a later
    /// deployment-config change from silently rescoring an existing table at
    /// a different candidate breadth than the table's own catalog promise.
    /// `None` for a row created before migration 023, read as the
    /// deployment's current oversample default by every consumer (the
    /// oversample every pre-migration table implicitly used, since no column
    /// existed yet to stamp it).
    pub oversample: Option<usize>,
    /// The writer that created this row (`writer-{uuid}`), or `None` for a
    /// row created before migration 027 or seeded without a writer. Survives
    /// promote/fail as history — only the lease is cleared.
    pub writer_id: Option<String>,
    /// The writer's lease deadline while the row is `building`; `None` once
    /// the row is terminal, and `None` on a pre-027 `building` row (recovery
    /// reads that as an absent lease and reconciles it as before).
    pub lease_expires_at: Option<String>,
}

fn parse_row(row: &Row<'_>) -> std::result::Result<ResultTableRecord, BackendError> {
    let task_raw: String = row.get("task")?;
    let task = ModelTask::try_from_db_str(&task_raw).map_err(|e| BackendError::TypeConversion {
        column: "task".into(),
        detail: e.to_string(),
    })?;
    let kind_raw: String = row.get("kind")?;
    let kind =
        ResultTableKind::try_from_db_str(&kind_raw).map_err(|e| BackendError::TypeConversion {
            column: "kind".into(),
            detail: e.to_string(),
        })?;
    let storage_precision_raw: Option<String> = row.try_get("storage_precision")?;
    let storage_precision = storage_precision_raw
        .map(|s| StoragePrecision::from_str(&s))
        .transpose()
        .map_err(|e| BackendError::TypeConversion {
            column: "storage_precision".into(),
            detail: e.to_string(),
        })?;
    let oversample = row.try_get::<i32>("oversample")?.map(|v| v.max(0) as usize);
    Ok(ResultTableRecord {
        table_name: row.get("table_name")?,
        source_id: row.get("source_id")?,
        model_id: row.get("model_id")?,
        task,
        kind,
        derived_from: row.try_get("derived_from")?,
        parquet_path: row.get("parquet_path")?,
        dimensions: row.try_get("dimensions")?,
        distance_metric: row.get("distance_metric")?,
        row_count: row.get::<i32>("row_count")? as usize,
        status: row.get("status")?,
        key_column: row.try_get("key_column")?,
        text_columns: row.try_get("text_columns")?,
        created_at: row.get("created_at")?,
        completed_at: row.try_get("completed_at")?,
        tenant_id: row.try_get("tenant_id")?,
        definition_hash: row.try_get("definition_hash")?,
        input_anchors_json: row.try_get("input_anchors_json")?,
        storage_precision,
        oversample,
        writer_id: row.try_get("writer_id")?,
        lease_expires_at: row.try_get("lease_expires_at")?,
    })
}

/// Which tenant predicate a building-row compare-and-set carries — chosen
/// from the binding in force at the call, never from the row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantArm {
    /// No tenant predicate: the call runs inside
    /// [`crate::session::JammiSession::with_admin_scope`] (startup recovery is
    /// the one named implicit-admin pass).
    Admin,
    /// STRICT: `tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)` — the
    /// row must belong to exactly this tenant (`None` = GLOBAL). A
    /// tenant-scoped writer can never transition a GLOBAL row, and an
    /// unscoped one never a tenant's.
    Strict(Option<TenantId>),
}

impl TenantArm {
    /// The arm the binding in force selects: [`TenantArm::Admin`] inside an
    /// admin scope, else STRICT on `tenant` — the tenant the caller captured
    /// when it created the row, so a heartbeat running on a task without the
    /// session's task-local scope still names the row's own tenant.
    pub fn in_force(tenant: Option<TenantId>) -> Self {
        if TenantBinding::is_admin_scope() {
            Self::Admin
        } else {
            Self::Strict(tenant)
        }
    }
}

/// Who a building-row compare-and-set requires the current owner to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Owner {
    /// The row's `writer_id` must equal this writer's id — the live writer's
    /// own transitions (renew, checkpoint, segment insert, promote, fail).
    Writer(String),
    /// The row's lease must be absent or expired AT THE INSTANT THE
    /// STATEMENT RUNS — recovery's arms, which may only touch a row whose
    /// writer is dead. Carries no timestamp: the predicate
    /// ([`lease_expired_clause`]) compares against the catalog backend's OWN
    /// clock (`now()` on Postgres), never a value this process computed
    /// ahead of time and bound in — that value could go stale by the time
    /// the CAS actually runs, and on Postgres it would be this process's
    /// clock rather than the database's regardless of freshness.
    ExpiredLease,
}

/// The one predicate builder every transition on a `building` result table
/// renders its `WHERE` through: `table_name = $t AND status = 'building' AND
/// <owner> AND <tenant>`. A transition that matches zero rows is classified
/// by re-reading the row (`Catalog::classify_cas_miss`) into exactly one of
/// [`JammiError::RowGone`], [`JammiError::TenantMismatch`],
/// [`JammiError::CasFailed`], [`JammiError::LeaseLost`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultTableCas {
    /// The `result_tables` primary key.
    pub table: String,
    /// The tenant predicate — see [`TenantArm::in_force`].
    pub tenant_arm: TenantArm,
    /// The ownership predicate.
    pub owner: Owner,
}

impl ResultTableCas {
    /// A live writer's CAS on its own row under the binding in force.
    pub fn writer(table: &str, writer_id: &str, tenant: Option<TenantId>) -> Self {
        Self {
            table: table.to_string(),
            tenant_arm: TenantArm::in_force(tenant),
            owner: Owner::Writer(writer_id.to_string()),
        }
    }

    /// The lease keeper's CAS on a live writer's row (N3,
    /// `crate::catalog::lease_keeper`): [`Owner::Writer`] with an ADMIN
    /// tenant arm, bypassing the STRICT per-tenant predicate
    /// [`Self::writer`] bakes in. The keeper thread renews every registration
    /// this PROCESS holds — possibly spanning several tenants — from a
    /// dedicated thread with no task-local session scope of its own to
    /// resolve a single tenant from; the row's `writer_id` match is the
    /// entire ownership check, exactly as strict as [`Self::writer`]'s own
    /// `Owner::Writer` arm, just without the additional tenant filter a
    /// single-tenant session-scoped caller can supply and a cross-tenant
    /// infrastructure thread cannot.
    pub fn writer_any_tenant(table: &str, writer_id: &str) -> Self {
        Self {
            table: table.to_string(),
            tenant_arm: TenantArm::Admin,
            owner: Owner::Writer(writer_id.to_string()),
        }
    }

    /// Recovery's CAS on a row whose lease is absent or expired AT THE
    /// INSTANT THE STATEMENT RUNS (the catalog backend's own clock — see
    /// [`Owner::ExpiredLease`]), under the binding in force (admin inside
    /// [`crate::store::ResultStore::recover`]).
    pub fn expired(table: &str, tenant: Option<TenantId>) -> Self {
        Self {
            table: table.to_string(),
            tenant_arm: TenantArm::in_force(tenant),
            owner: Owner::ExpiredLease,
        }
    }

    /// Render the owner + tenant arms (no table / status arm) against column
    /// prefix `col` (`""` or `"r."`), appending binds to `params`. `kind`
    /// selects the backend-appropriate lease-expiry SQL for
    /// [`Owner::ExpiredLease`] — see `catalog::lease`'s module docs.
    fn render_owner_and_tenant(
        &self,
        col: &str,
        kind: BackendKind,
        params: &mut Vec<SqlValue<'static>>,
    ) -> String {
        let mut sql = String::new();
        match &self.owner {
            Owner::Writer(w) => {
                params.push(SqlValue::TextOwned(w.clone()));
                sql.push_str(&format!("{col}writer_id = ${}", params.len()));
            }
            Owner::ExpiredLease => {
                sql.push_str(&lease_expired_clause(
                    &format!("{col}lease_expires_at"),
                    kind,
                    params,
                ));
            }
        }
        match &self.tenant_arm {
            TenantArm::Admin => {}
            TenantArm::Strict(t) => {
                params.push(SqlValue::from(t.map(|t| t.to_string())));
                let n = params.len();
                sql.push_str(&format!(
                    " AND ({col}tenant_id = ${n} OR ({col}tenant_id IS NULL AND ${n} IS NULL))"
                ));
            }
        }
        sql
    }

    /// Render the full building-row predicate, appending its binds to
    /// `params` (whose existing entries are the statement's earlier binds).
    fn render(&self, kind: BackendKind, params: &mut Vec<SqlValue<'static>>) -> String {
        params.push(SqlValue::TextOwned(self.table.clone()));
        let mut sql = format!(
            "table_name = ${} AND status = 'building' AND ",
            params.len()
        );
        sql.push_str(&self.render_owner_and_tenant("", kind, params));
        sql
    }

    /// Render an `EXISTS (SELECT 1 FROM result_tables r WHERE …)` ownership
    /// test WITHOUT the status arm — the predicate a segment purge uses, since
    /// a purge follows a transition that already left `building` (the writer's
    /// own `failed`, or recovery's claim) and `writer_id` survives that
    /// transition as history.
    pub(crate) fn render_owner_exists(
        &self,
        kind: BackendKind,
        params: &mut Vec<SqlValue<'static>>,
    ) -> String {
        params.push(SqlValue::TextOwned(self.table.clone()));
        let mut sql = format!(
            "EXISTS (SELECT 1 FROM result_tables r WHERE r.table_name = ${} AND ",
            params.len()
        );
        sql.push_str(&self.render_owner_and_tenant("r.", kind, params));
        sql.push(')');
        sql
    }
}

/// The row a zero-row CAS re-reads by primary key (no tenant predicate) to
/// classify the miss.
#[derive(Debug, Clone)]
pub(crate) struct CasTarget {
    tenant_id: Option<String>,
    status: String,
    writer_id: Option<String>,
}

/// Re-read the CAS target by primary key inside the same transaction.
pub(crate) async fn read_cas_target(
    tx: &mut Transaction<'_>,
    table: &str,
) -> std::result::Result<Option<CasTarget>, BackendError> {
    tx.query_opt(
        "SELECT tenant_id, status, writer_id FROM result_tables WHERE table_name = $1",
        &[SqlValue::TextOwned(table.to_string())],
        |row| {
            Ok(CasTarget {
                tenant_id: row.try_get("tenant_id")?,
                status: row.get("status")?,
                writer_id: row.try_get("writer_id")?,
            })
        },
    )
    .await
}

/// Outcome of a CAS statement: applied to exactly one row, or missed with the
/// row's current shape (or `None` when the row is gone).
pub(crate) enum CasOutcome<T> {
    Applied(T),
    Missed(Option<CasTarget>),
}

/// Whether [`Catalog::building_tables_by_lease_liveness`]'s non-admin arm
/// drops a GLOBAL (`tenant_id IS NULL`) row entirely — an enumeration that
/// feeds a MUTATING pass must never hand a tenant-bound caller a `_global/`
/// row (esc-094) — or reads it the same way every other read on this table
/// does (`tenant_id = $t OR tenant_id IS NULL` — safe for a READ-ONLY
/// protective set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExcludeGlobalUnderTenantScope {
    Yes,
    No,
}

impl Catalog {
    /// Insert a new result table record with status = 'building'. Binds
    /// the session's tenant to the row (SPEC-03 §7).
    ///
    /// When `p.job_attempt` is `Some`, the SAME transaction also performs the
    /// `jobs.partial_result` compare-and-set this table's job depends on
    /// (N11): `UPDATE jobs SET partial_result = table_name WHERE job_id =
    /// $job_id AND claimed_by = $instance_id AND status = 'running' AND
    /// attempts = $attempts AND partial_result IS NULL`. Landing it in the
    /// same transaction as the row's own INSERT means the two either commit
    /// together or neither does — there is no window where a `result_tables`
    /// row exists with no job pointing at it, or a job's `partial_result`
    /// names a table whose INSERT never landed. A zero-row CAS means the
    /// caller's `(instance_id, attempts)` is no longer the job's current
    /// lease holder (a peer reclaimed and re-claimed it: this attempt is
    /// superseded) or a peer's attempt already recorded a `partial_result`
    /// first (also superseded — first writer of record wins) — either way
    /// the whole transaction is rolled back (no `result_tables` row, no
    /// INSERT) and this call returns [`JammiError::JobAttemptSuperseded`],
    /// touching no bytes.
    ///
    /// This CAS carries the full `(job_id, claimed_by, attempts)` guard every
    /// other lease-guarded write on `jobs` carries (the same one
    /// [`Catalog::record_partial_result`], `crate::catalog::jobs_repo`,
    /// uses) — esc-107: an earlier `job_id`-only predicate (`status =
    /// 'running' AND partial_result IS NULL`, no `claimed_by`/`attempts`
    /// check) was UNSOUND, not merely narrower. A zombie of a REQUEUED and
    /// RE-CLAIMED attempt — its own lease expired, the job went
    /// `queued -> running` again under a later attempt, all while the
    /// zombie never learned its lease was gone — still observes the job as
    /// `running` with `partial_result IS NULL` and would win that weaker
    /// CAS, recording a DEAD attempt's table as the job's `partial_result`
    /// out from under the live, current attempt. Pinning `claimed_by` and
    /// `attempts` closes exactly this window: a zombie's stale identity
    /// matches zero rows the instant a later attempt has claimed the row,
    /// the same guarantee every other write in `jobs_repo` already carries.
    ///
    /// Rollback is forced, not merely reported: [`BackendError::Busy`] is the
    /// transaction-internal refusal sentinel every CAS-inside-a-multi-write
    /// transaction on this catalog uses to unwind writes already issued in
    /// the SAME closure (the same mechanism
    /// [`Catalog::delete_result_tables_for_source`]'s `SourceBusy` uses) —
    /// returning `Ok` here instead would COMMIT the INSERT that already ran.
    pub async fn create_result_table(&self, p: CreateResultTableParams<'_>) -> Result<()> {
        let table_name = p.table_name.to_string();
        let source_id = p.source_id.to_string();
        let model_id = p.model_id.to_string();
        let task = p.task.as_db_str();
        let kind = p.kind.as_db_str();
        let derived_from = p.derived_from.map(str::to_string);
        let parquet_path = p.parquet_path.to_string();
        let dimensions = p.dimensions;
        let key_column = p.key_column.map(str::to_string);
        let text_columns = p.text_columns.map(str::to_string);
        let storage_precision = p.storage_precision.to_string();
        let oversample = p.oversample;
        let created_at = p.created_at;
        let writer_id = p.writer_id.map(str::to_string);
        let lease = p.lease;
        let job_attempt = p.job_attempt.map(|ja| {
            (
                ja.job_id.to_string(),
                ja.instance_id.to_string(),
                ja.attempts,
            )
        });
        let kind_backend = self.backend().backend_kind();
        let tenant = self.current_tenant();

        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "result_tables")?;
                    let table_name_for_cas = table_name.clone();
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::TextOwned(table_name),
                        SqlValue::TextOwned(source_id),
                        SqlValue::TextOwned(model_id),
                        SqlValue::Text(task),
                        SqlValue::Text(kind),
                        SqlValue::from(derived_from),
                        SqlValue::TextOwned(parquet_path),
                        SqlValue::from(dimensions.map(|d| d as i64)),
                        SqlValue::from(key_column),
                        SqlValue::from(text_columns),
                        SqlValue::from(tenant.map(|t| t.to_string())),
                        SqlValue::TextOwned(storage_precision),
                        SqlValue::from(oversample as i64),
                        SqlValue::TextOwned(created_at),
                        SqlValue::from(writer_id),
                    ];
                    // The lease deadline is computed by the DB clock on
                    // Postgres (never this process's clock — see
                    // `catalog::lease`'s module docs); `None` binds a typed
                    // NULL rather than embedding an expression at all.
                    let lease_expr = match lease {
                        Some(window) => lease_deadline_expr(kind_backend, window, &mut params),
                        None => {
                            params.push(SqlValue::Null(crate::catalog::backend::SqlNullType::Text));
                            format!("${}", params.len())
                        }
                    };
                    tx.execute(
                        &format!(
                            "INSERT INTO result_tables (table_name, source_id, model_id, task, kind, \
                             derived_from, parquet_path, dimensions, key_column, \
                             text_columns, tenant_id, storage_precision, oversample, created_at, \
                             writer_id, lease_expires_at) \
                             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, \
                             $15, {lease_expr})"
                        ),
                        &params,
                    )
                    .await?;

                    let Some((job_id, instance_id, attempts)) = job_attempt else {
                        return Ok(());
                    };
                    let running = crate::catalog::status::JobStatus::Running.to_string();
                    let cas_updated = tx
                        .execute(
                            "UPDATE jobs SET partial_result = $1 \
                             WHERE job_id = $2 AND claimed_by = $3 AND status = $4 \
                               AND attempts = $5 AND partial_result IS NULL",
                            &[
                                SqlValue::TextOwned(table_name_for_cas),
                                SqlValue::TextOwned(job_id.clone()),
                                SqlValue::TextOwned(instance_id),
                                SqlValue::TextOwned(running),
                                SqlValue::Int(attempts as i64),
                            ],
                        )
                        .await?;
                    if cas_updated == 1 {
                        Ok(())
                    } else {
                        // `Busy` forces a ROLLBACK of the INSERT that already
                        // ran in this same transaction — see the doc above.
                        Err(BackendError::Busy(job_id))
                    }
                })
            })
            .await;
        match outcome {
            Ok(()) => Ok(()),
            Err(BackendError::Busy(job_id)) => Err(JammiError::JobAttemptSuperseded { job_id }),
            Err(e) => Err(e.into()),
        }
    }

    /// Update a result table's status and row count. Sets `completed_at` when
    /// transitioning to a terminal state (Ready/Failed).
    ///
    /// This is NOT a building-row transition: every `building -> ready` /
    /// `building -> failed` flip goes through the lease-owned compare-and-set
    /// helpers ([`Self::promote_result_table_with_manifest`],
    /// [`Self::fail_building_table`]), which name the writer. This remains for
    /// its non-building callers (a test forging a pre-contract `ready` row;
    /// the `ready -> failed` arm is [`Self::fail_ready_result_table`]).
    ///
    /// Inside a [`crate::session::JammiSession::with_admin_scope`] closure the
    /// tenant predicate is dropped and the row is addressed by its
    /// `table_name` primary key alone. Outside admin scope the update is
    /// STRICT-scoped — `tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`
    /// — so a tenant can never flip another tenant's row, nor a GLOBAL one.
    pub async fn update_result_table_status(
        &self,
        name: &str,
        status: ResultTableStatus,
        rows: usize,
    ) -> Result<()> {
        let completed_at = if matches!(status, ResultTableStatus::Ready | ResultTableStatus::Failed)
        {
            Some(
                chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            )
        } else {
            None
        };
        let status_str = status.to_string();
        let name = name.to_string();
        let rows_i64 = rows as i64;
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    if admin {
                        tx.execute(
                            "UPDATE result_tables SET status = $1, row_count = $2, \
                             completed_at = $3 WHERE table_name = $4",
                            &[
                                SqlValue::TextOwned(status_str),
                                SqlValue::Int(rows_i64),
                                SqlValue::from(completed_at),
                                SqlValue::TextOwned(name),
                            ],
                        )
                        .await?;
                    } else {
                        tx.execute(
                            "UPDATE result_tables SET status = $1, row_count = $2, \
                             completed_at = $3 \
                             WHERE table_name = $4 \
                               AND (tenant_id = $5 OR (tenant_id IS NULL AND $5 IS NULL))",
                            &[
                                SqlValue::TextOwned(status_str),
                                SqlValue::Int(rows_i64),
                                SqlValue::from(completed_at),
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Flip a `ready` result table to `failed` — the arm
    /// [`crate::store::ResultStore::recover`]'s ready-manifest reconciliation
    /// takes when a post-contract `ready` row has lost its attestation
    /// sidecar. A compare-and-set guarded on `status = 'ready'` under the
    /// tenant arm in force (admin bypass / STRICT), returning whether it
    /// affected exactly one row; the caller deletes the row's objects only
    /// after `Ok(true)`.
    pub async fn fail_ready_result_table(&self, name: &str) -> Result<bool> {
        let completed_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        let name = name.to_string();
        let arm = TenantArm::in_force(self.current_tenant());
        let tenant = self.current_tenant();
        let affected = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(completed_at), SqlValue::TextOwned(name)];
                    let mut sql = String::from(
                        "UPDATE result_tables SET status = 'failed', row_count = 0, \
                         completed_at = $1, lease_expires_at = NULL \
                         WHERE table_name = $2 AND status = 'ready'",
                    );
                    if let TenantArm::Strict(t) = arm {
                        params.push(SqlValue::from(t.map(|t| t.to_string())));
                        sql.push_str(" AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL))");
                    }
                    tx.execute(&sql, &params).await
                })
            })
            .await?;
        Ok(affected == 1)
    }

    /// Classify a building-row CAS that matched zero rows into exactly one
    /// typed error, status-first (esc-094):
    ///
    /// 1. no row → [`JammiError::RowGone`] (nothing to delete);
    /// 2. a non-admin binding and the row's tenant differs →
    ///    [`JammiError::TenantMismatch`] (never deletes);
    /// 3. `status != 'building'` → [`JammiError::CasFailed`] naming the status
    ///    (e.g. `ready` because recovery promoted an expired-lease row whose
    ///    sidecar had landed; the caller never re-promotes);
    /// 4. `status == 'building'` and the owner arm no longer holds (the row's
    ///    `writer_id` is not ours; or, for an expired-lease arm, the lease was
    ///    renewed) → [`JammiError::LeaseLost`]. The claimant now owns the row
    ///    and its bytes: the loser deletes nothing.
    pub(crate) fn classify_cas_miss(cas: &ResultTableCas, target: Option<CasTarget>) -> JammiError {
        let table = cas.table.clone();
        let Some(row) = target else {
            return JammiError::RowGone { table };
        };
        if let TenantArm::Strict(t) = &cas.tenant_arm {
            let bound = t.map(|t| t.to_string());
            if row.tenant_id != bound {
                return JammiError::TenantMismatch { table };
            }
        }
        if row.status != ResultTableStatus::Building.to_string() {
            return JammiError::CasFailed {
                table,
                status: row.status,
            };
        }
        match &cas.owner {
            Owner::Writer(w) if row.writer_id.as_deref() == Some(w.as_str()) => {
                // Every arm matched on re-read: the statement and the re-read
                // straddled a concurrent transition. Report the row as it is.
                JammiError::CasFailed {
                    table,
                    status: row.status,
                }
            }
            _ => JammiError::LeaseLost { table },
        }
    }

    /// Run `set_clause` as a compare-and-set on the building row `cas` names,
    /// returning `Ok(())` when exactly one row changed and the classified
    /// typed error otherwise. `set_params` are the `SET` binds `$1..$n`;
    /// `set_clause` may reference them.
    async fn building_row_cas(
        &self,
        cas: &ResultTableCas,
        set_clause: &str,
        set_params: Vec<SqlValue<'static>>,
    ) -> Result<()> {
        let cas_in_tx = cas.clone();
        let set_clause = set_clause.to_string();
        let tenant = self.current_tenant();
        let kind = self.backend().backend_kind();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let cas = cas_in_tx;
                    tx.set_tenant(tenant);
                    let mut params = set_params;
                    let predicate = cas.render(kind, &mut params);
                    let sql = format!("UPDATE result_tables SET {set_clause} WHERE {predicate}");
                    let affected = tx.execute(&sql, &params).await?;
                    if affected == 1 {
                        return Ok(CasOutcome::Applied(()));
                    }
                    Ok(CasOutcome::Missed(read_cas_target(tx, &cas.table).await?))
                })
            })
            .await?;
        match outcome {
            CasOutcome::Applied(()) => Ok(()),
            CasOutcome::Missed(target) => Err(Self::classify_cas_miss(cas, target)),
        }
    }

    /// Renew the lease on the building row `cas` names to `lease` from NOW —
    /// the catalog backend's OWN clock on Postgres, never this process's
    /// (`catalog::lease`'s module docs); `lease` is the WINDOW, always the
    /// deployment's configured duration, not a computed deadline. The
    /// writer's heartbeat.
    pub async fn renew_lease(&self, cas: &ResultTableCas, lease: Duration) -> Result<()> {
        let kind = self.backend().backend_kind();
        let mut params = Vec::new();
        let expr = lease_deadline_expr(kind, lease, &mut params);
        self.building_row_cas(cas, &format!("lease_expires_at = {expr}"), params)
            .await
    }

    /// Persist a checkpoint (batch number) on the building row `cas` names.
    pub async fn set_checkpoint(&self, cas: &ResultTableCas, batch: usize) -> Result<()> {
        self.building_row_cas(cas, "checkpoint = $1", vec![SqlValue::Int(batch as i64)])
            .await
    }

    /// The only `building -> failed` transition: a compare-and-set on the row
    /// `cas` names that sets `failed`, `row_count = 0`, `completed_at`, and
    /// clears the lease (`writer_id` stays as history). The caller may delete
    /// the row's objects only after this returns `Ok(())` — a writer aborting
    /// its own table, or recovery reaping an expired-lease row.
    pub async fn fail_building_table(&self, cas: &ResultTableCas) -> Result<()> {
        let completed_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        self.building_row_cas(
            cas,
            "status = 'failed', row_count = 0, completed_at = $1, lease_expires_at = NULL",
            vec![SqlValue::TextOwned(completed_at)],
        )
        .await
    }

    /// Recovery's claim on an expired-lease building row: a compare-and-set
    /// (`cas.owner` must be [`Owner::ExpiredLease`]) that stamps
    /// `writer_id = new_writer_id` and a fresh lease `lease` FROM THE
    /// CATALOG'S OWN CLOCK on Postgres, so the recoverer becomes the owner
    /// BEFORE it rebuilds, promotes, or deletes anything. A returning
    /// writer's next heartbeat then misses its own CAS and reports
    /// [`JammiError::LeaseLost`]. Returns `Ok(false)` — never an error — when
    /// the claim matched zero rows (another recoverer or the writer got there
    /// first): the loser skips the row.
    pub async fn claim_expired_building_table(
        &self,
        cas: &ResultTableCas,
        new_writer_id: &str,
        lease: Duration,
    ) -> Result<bool> {
        let kind = self.backend().backend_kind();
        let mut params = vec![SqlValue::TextOwned(new_writer_id.to_string())];
        let expr = lease_deadline_expr(kind, lease, &mut params);
        match self
            .building_row_cas(
                cas,
                &format!("writer_id = $1, lease_expires_at = {expr}"),
                params,
            )
            .await
        {
            Ok(()) => Ok(true),
            Err(
                JammiError::RowGone { .. }
                | JammiError::TenantMismatch { .. }
                | JammiError::CasFailed { .. }
                | JammiError::LeaseLost { .. },
            ) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Test-only: force the building row `cas` names into an already-expired
    /// lease, using the catalog backend's OWN clock — a value strictly in the
    /// past by the time any later statement reads `now()` — so a recovery
    /// sweep run immediately afterwards treats the row as a dead writer's.
    /// Runs as `cas`'s own CAS (typically [`Owner::Writer`]), so it fails
    /// loudly rather than silently leaving a live lease if the row is not the
    /// caller's own `building` row.
    ///
    /// The SQLite arm binds a [`LEASE_TS_FORMAT`]-shaped
    /// timestamp (through the application clock, matching every other SQLite
    /// lease stamp) rather than SQLite's OWN `datetime()` rendering
    /// (space-separated, no fractional seconds): comparing the two shapes
    /// would be "expired" by construction regardless of the actual instants
    /// involved (`' ' < 'T'` lexicographically, always), never by the real
    /// contract [`lease_expired_clause`] evaluates elsewhere.
    #[cfg(feature = "test-hooks")]
    pub async fn expire_lease_for_test(&self, cas: &ResultTableCas) -> Result<()> {
        let kind = self.backend().backend_kind();
        let (set_clause, set_params): (String, Vec<SqlValue<'static>>) = match kind {
            BackendKind::Postgres => (
                "lease_expires_at = (now() - interval '1 second')::text".to_string(),
                Vec::new(),
            ),
            BackendKind::Sqlite => {
                let past = (chrono::Utc::now() - chrono::Duration::seconds(1))
                    .format(LEASE_TS_FORMAT)
                    .to_string();
                (
                    "lease_expires_at = $1".to_string(),
                    vec![SqlValue::TextOwned(past)],
                )
            }
        };
        self.building_row_cas(cas, &set_clause, set_params).await
    }

    /// Flip the building row `cas` names `building -> ready` **and** persist
    /// its materialization-contract summary columns (`definition_hash`,
    /// `input_anchors_json`) in a single compare-and-set — the indexable
    /// summary of the `.materialization.json` sidecar, so verification and
    /// provenance queries need not open every sidecar. This is the catalog
    /// half of the single `building -> ready` boundary
    /// ([`crate::store::BuildingTable::finish`]); the sidecar is written
    /// before this commits, so a crash never leaves a `ready` row whose
    /// manifest never landed. Clears the lease; `writer_id` stays as history.
    ///
    /// Returns the promoted row's `tenant_id` (the owner stamped at
    /// `create_table`, or `None` for a GLOBAL row) so the caller can register
    /// the table's DataFusion provider under its catalog owner by construction.
    /// A zero-row match is the classified typed error — a writer that gets
    /// [`JammiError::CasFailed`] with `status = ready` was superseded by
    /// recovery's promotion of its own bytes and must not re-promote.
    pub async fn promote_result_table_with_manifest(
        &self,
        cas: &ResultTableCas,
        rows: usize,
        definition_hash: &str,
        input_anchors_json: &str,
    ) -> Result<Option<TenantId>> {
        let completed_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        let cas_in_tx = cas.clone();
        let rows_i64 = rows as i64;
        let definition_hash = definition_hash.to_string();
        let input_anchors_json = input_anchors_json.to_string();
        let tenant = self.current_tenant();
        let kind = self.backend().backend_kind();

        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let cas = cas_in_tx;
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::Int(rows_i64),
                        SqlValue::TextOwned(completed_at),
                        SqlValue::TextOwned(definition_hash),
                        SqlValue::TextOwned(input_anchors_json),
                    ];
                    let predicate = cas.render(kind, &mut params);
                    let sql = format!(
                        "UPDATE result_tables SET status = 'ready', row_count = $1, \
                         completed_at = $2, definition_hash = $3, input_anchors_json = $4, \
                         lease_expires_at = NULL WHERE {predicate}"
                    );
                    let affected = tx.execute(&sql, &params).await?;
                    // Read the row's own owner back inside the same
                    // transaction — by primary key, so it is exact and
                    // scope-independent.
                    let target = read_cas_target(tx, &cas.table).await?;
                    if affected == 1 {
                        return Ok(CasOutcome::Applied(target.and_then(|t| t.tenant_id)));
                    }
                    Ok(CasOutcome::Missed(target))
                })
            })
            .await?;
        match outcome {
            CasOutcome::Applied(owner) => owner.map(|s| TenantId::from_str(&s)).transpose(),
            CasOutcome::Missed(target) => Err(Self::classify_cas_miss(cas, target)),
        }
    }

    /// Every `building` row whose lease is absent or expired AT THE INSTANT
    /// THE STATEMENT RUNS — the backend's OWN clock
    /// ([`crate::catalog::lease::lease_expired_clause`]; never a bound
    /// application timestamp on Postgres) — the enumeration BOTH
    /// `recover()`'s admin-scoped sweep and a tenant-scoped
    /// [`crate::store::ResultStore::reconcile`]'s expired-building pre-pass
    /// (esc-094) claim/fail/delete against. A row under a live lease
    /// belongs to a live writer and is never listed here.
    ///
    /// **Never includes a GLOBAL (`tenant_id IS NULL`) row under a
    /// non-admin binding** — unlike every other read on this
    /// table, which treats GLOBAL as "visible to every tenant" because
    /// reading a shared row leaks nothing. THIS enumeration feeds a MUTATING
    /// pass (claim, then promote-or-fail, then delete): a tenant-bound
    /// caller must never be able to reach a `_global/` row's bytes, and
    /// [`ResultTableCas::expired`]'s `Strict` tenant arm renders against the
    /// ROW's OWN tenant (so it cannot itself refuse a GLOBAL row for a
    /// tenant-bound caller — the arm's whole point is recovery's admin bypass,
    /// not a caller-identity check), so the exclusion has to happen HERE, at
    /// the enumeration that decides which rows a tenant-scoped pass ever
    /// touches. Inside an admin scope every tenant's (and GLOBAL's) rows are
    /// still returned.
    pub async fn list_expired_building_tables(&self) -> Result<Vec<ResultTableRecord>> {
        self.building_tables_by_lease_liveness(false, ExcludeGlobalUnderTenantScope::Yes)
            .await
    }

    /// Every `building` row under a LIVE (unexpired) lease at the instant the
    /// query runs — the rows a reconcile pass must never treat as an orphan
    /// candidate (an expired-lease `building` row is recovery's to
    /// claim-then-reap, esc-094; only a row a live writer still owns is
    /// protected here). Inside an admin scope every tenant's rows are returned; outside
    /// it the enumeration is tenant-scoped like every other READ on this
    /// table (GLOBAL included) — this is a READ-ONLY protective set (it only
    /// ever widens what a listed object is considered "referenced" by,
    /// never licenses a mutation), so including a GLOBAL row here is
    /// harmless: a tenant-scoped reconcile's own `in_scope` filter already
    /// excludes every `_global/` object from that pass regardless.
    pub async fn list_live_building_tables(&self) -> Result<Vec<ResultTableRecord>> {
        self.building_tables_by_lease_liveness(true, ExcludeGlobalUnderTenantScope::No)
            .await
    }

    /// Shared enumeration behind [`Self::list_expired_building_tables`] and
    /// [`Self::list_live_building_tables`]: every `status = 'building'` row
    /// additionally matching `lease_predicate` (a full boolean SQL
    /// expression, built entirely from the backend's own clock on Postgres
    /// (a bound [`crate::catalog::lease::lease_now`] on SQLite, per
    /// `catalog::lease`'s module docs) — its own bind, if any, is threaded
    /// through so the tenant bind that follows numbers correctly regardless
    /// of backend). `live` selects expired (`false`) vs. live (`true`);
    /// `exclude_global` selects whether a non-admin binding's query drops
    /// `tenant_id IS NULL` rows entirely or reads them the same
    /// way every other table read does.
    async fn building_tables_by_lease_liveness(
        &self,
        live: bool,
        exclude_global: ExcludeGlobalUnderTenantScope,
    ) -> Result<Vec<ResultTableRecord>> {
        let kind = self.backend().backend_kind();
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let mut params: Vec<SqlValue<'static>> = Vec::new();
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        let lease_predicate =
                            if live { format!("NOT {expired}") } else { expired };
                        if admin {
                            tx.query(
                                &format!(
                                    "SELECT * FROM result_tables WHERE status = 'building' \
                                     AND {lease_predicate} ORDER BY created_at"
                                ),
                                &params,
                                parse_row,
                            )
                            .await
                        } else if matches!(exclude_global, ExcludeGlobalUnderTenantScope::Yes) {
                            // Exact-match, never "OR tenant_id IS NULL": a
                            // GLOBAL caller (`tenant = None`) still sees its
                            // OWN GLOBAL rows (`tenant_id IS NULL AND $n IS
                            // NULL`), but a tenant-bound caller (`tenant =
                            // Some(t)`) never matches a row whose
                            // `tenant_id` differs, INCLUDING a GLOBAL one.
                            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                            let n = params.len();
                            tx.query(
                                &format!(
                                    "SELECT * FROM result_tables WHERE status = 'building' \
                                     AND {lease_predicate} \
                                     AND (tenant_id = ${n} OR (tenant_id IS NULL AND ${n} IS NULL)) \
                                     ORDER BY created_at"
                                ),
                                &params,
                                parse_row,
                            )
                            .await
                        } else {
                            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                            let n = params.len();
                            tx.query(
                                &format!(
                                    "SELECT * FROM result_tables WHERE status = 'building' \
                                     AND {lease_predicate} \
                                     AND (tenant_id = ${n} OR tenant_id IS NULL) \
                                     ORDER BY created_at"
                                ),
                                &params,
                                parse_row,
                            )
                            .await
                        }
                    })
                },
            )
            .await?)
    }

    /// Fetch a single result table by name. Tenant-filtered; inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the tenant
    /// predicate is dropped and the row resolves by its primary key alone
    /// (the read recovery and its tests use to observe a row of any tenant).
    pub async fn get_result_table(&self, name: &str) -> Result<Option<ResultTableRecord>> {
        let name = name.to_string();
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        if admin {
                            tx.query_opt(
                                "SELECT * FROM result_tables WHERE table_name = $1",
                                &[SqlValue::TextOwned(name)],
                                parse_row,
                            )
                            .await
                        } else {
                            tx.query_opt(
                                "SELECT * FROM result_tables WHERE table_name = $1 \
                                   AND (tenant_id = $2 OR tenant_id IS NULL)",
                                &[
                                    SqlValue::TextOwned(name),
                                    SqlValue::from(tenant.map(|t| t.to_string())),
                                ],
                                parse_row,
                            )
                            .await
                        }
                    })
                },
            )
            .await?)
    }

    /// List result tables with a given status, scoped to the session tenant.
    ///
    /// Inside a [`crate::session::JammiSession::with_admin_scope`] closure the
    /// per-row tenant filter is dropped and rows from **every** tenant are
    /// returned — the cross-tenant enumeration startup recovery needs so a
    /// `building` orphan owned by any tenant is reconciled, not only the
    /// session's own and GLOBAL (`tenant_id IS NULL`) rows. Outside admin scope
    /// the query is tenant-scoped exactly as every other read on this table.
    pub async fn list_result_tables_by_status(
        &self,
        status: ResultTableStatus,
    ) -> Result<Vec<ResultTableRecord>> {
        let status_str = status.to_string();
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        if admin {
                            tx.query(
                                "SELECT * FROM result_tables WHERE status = $1 \
                                 ORDER BY created_at",
                                &[SqlValue::TextOwned(status_str)],
                                parse_row,
                            )
                            .await
                        } else {
                            tx.query(
                                "SELECT * FROM result_tables WHERE status = $1 \
                                   AND (tenant_id = $2 OR tenant_id IS NULL) \
                                 ORDER BY created_at",
                                &[
                                    SqlValue::TextOwned(status_str),
                                    SqlValue::from(tenant.map(|t| t.to_string())),
                                ],
                                parse_row,
                            )
                            .await
                        }
                    })
                },
            )
            .await?)
    }

    /// Find result tables matching source, optional task, optional model;
    /// scoped to the session tenant.
    pub async fn find_result_tables(
        &self,
        source_id: &str,
        task: Option<ModelTask>,
        model_id: Option<&str>,
    ) -> Result<Vec<ResultTableRecord>> {
        let mut sql = "SELECT * FROM result_tables WHERE source_id = $1".to_string();
        let mut params: Vec<SqlValue<'static>> = vec![SqlValue::TextOwned(source_id.to_string())];

        if let Some(t) = task {
            sql.push_str(&format!(" AND task = ${}", params.len() + 1));
            params.push(SqlValue::Text(t.as_db_str()));
        }
        if let Some(m) = model_id {
            sql.push_str(&format!(" AND model_id = ${}", params.len() + 1));
            params.push(SqlValue::TextOwned(m.to_string()));
        }
        let tenant = self.current_tenant();
        sql.push_str(&format!(
            " AND (tenant_id = ${} OR tenant_id IS NULL)",
            params.len() + 1
        ));
        params.push(SqlValue::from(tenant.map(|t| t.to_string())));
        sql.push_str(" ORDER BY created_at");

        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query(&sql, &params, parse_row).await }),
            )
            .await?)
    }

    /// Find the `ready` result tables produced by a given `definition_hash`,
    /// scoped to the session tenant — the indexed candidate set the sensing
    /// layer's cache lookup ([`crate::store::ResultStore::lookup_cached`])
    /// narrows before its Rust exact-anchor post-filter.
    ///
    /// The predicate is `definition_hash = $1 AND status = 'ready'` (migration
    /// 022's index covers the equality arm). Many rows can share a definition
    /// hash (the same definition over different anchors, or re-emissions —
    /// including a producer that legitimately re-materialises the *same*
    /// `(definition, inputs)` key, e.g. an idempotent recompute or a race), so
    /// this returns every candidate, **newest first** (`ORDER BY created_at
    /// DESC, table_name DESC`). Every same-key `ready` row is a semantically
    /// equivalent reuse; the caller prefers the newest, and iterates the rest as
    /// a fallback when the newest's artifact turns out to have been reaped (see
    /// [`crate::store::ResultStore::probe_cache_record`]). The *exact*
    /// `(definition_hash, input_anchors)` match is the caller's anchor-set
    /// comparison.
    ///
    /// `created_at` is app-supplied (`backend::now_sortable`) at nanosecond
    /// resolution, identical in shape on both backends; `table_name DESC` is a
    /// deterministic final tiebreak for a same-nanosecond collision (every
    /// table name carries a uuid suffix, so the pick is at least
    /// deterministic), the same idiom [`Self::resolve_embedding_table`] uses.
    pub async fn find_ready_result_tables_by_definition(
        &self,
        definition_hash: &str,
    ) -> Result<Vec<ResultTableRecord>> {
        let hash = definition_hash.to_string();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT * FROM result_tables \
                               WHERE definition_hash = $1 AND status = 'ready' \
                                 AND (tenant_id = $2 OR tenant_id IS NULL) \
                             ORDER BY created_at DESC, table_name DESC",
                            &[
                                SqlValue::TextOwned(hash),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Find the `ready` result tables whose recorded `input_anchors_json` names
    /// `source` as an input — the candidate set for the sensing layer's
    /// one-hop reverse-dependency lineage
    /// ([`crate::store::ResultStore::derives_from`]). Scoped to the session
    /// tenant.
    ///
    /// The predicate is a `LIKE` over the JSON-encoded source key — a safe
    /// *over-approximation* (the substring could in principle appear in another
    /// field) the caller refines with an exact decode-and-match. The pattern is
    /// a bound parameter, and `source` is escaped for LIKE so a name containing
    /// `%`, `_`, or the escape character cannot widen the match.
    pub async fn find_ready_result_tables_anchored_on(
        &self,
        source: &str,
    ) -> Result<Vec<ResultTableRecord>> {
        // The anchor's source is serialised as `"source":"<value>"` in
        // `input_anchors_json`; escape LIKE metacharacters in `<value>` so a
        // crafted source name cannot turn the pre-filter into a wildcard. `\` is
        // the escape character declared in the ESCAPE clause.
        let escaped = source
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%\"source\":\"{escaped}\"%");
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT * FROM result_tables \
                               WHERE status = 'ready' \
                                 AND input_anchors_json LIKE $1 ESCAPE '\\' \
                                 AND (tenant_id = $2 OR tenant_id IS NULL) \
                             ORDER BY created_at",
                            &[
                                SqlValue::TextOwned(pattern),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Delete all result tables for a source. Returns the deleted records
    /// so callers can clean up associated disk files. Scoped strictly to the
    /// session's tenant — a tenant deletes only its own result tables, never a
    /// shared GLOBAL (`tenant_id IS NULL`) table it did not create; only an
    /// unscoped session manages GLOBAL rows.
    ///
    /// ONE atomic statement, not a SELECT-then-DELETE: under READ COMMITTED
    /// a SELECT and a later DELETE are two independent snapshots, so a
    /// `building` row a concurrent writer commits between them would be
    /// missed by the liveness check yet still deleted by the DELETE's own
    /// (unconditional) `WHERE source_id = …`. Here the DELETE's `WHERE`
    /// itself carries the liveness guard (`RETURNING` gives the disk-cleanup
    /// set in the same round trip, so it is always exactly the deleted set),
    /// and a second statement in the SAME transaction re-checks, under a
    /// fresh READ COMMITTED snapshot, whether a live `building` row is now
    /// present — either because it was excluded by the DELETE's guard, or
    /// because a writer's `create_result_table` committed one in the window
    /// between the two statements. Either way, finding one rolls back the
    /// WHOLE transaction (nothing the DELETE already removed survives past
    /// the rollback) and the caller sees [`JammiError::SourceBusy`] naming
    /// it: a writer is still materialising over this source, and deleting
    /// other rows underneath it while leaving that race unresolved is not
    /// atomic enough to trust the returned cleanup set.
    pub async fn delete_result_tables_for_source(
        &self,
        source_id: &str,
    ) -> Result<Vec<ResultTableRecord>> {
        let sid = source_id.to_string();
        let tenant = self.current_tenant();
        let kind = self.backend().backend_kind();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let tenant_param = SqlValue::from(tenant.map(|t| t.to_string()));
                    // Delete every row for this source that is NOT a
                    // live-lease `building` row (a terminal row, or a
                    // `building` row whose lease is absent/expired — a dead
                    // writer's, safely reaped with the rest). `expired`'s
                    // bind (if any, on SQLite) is appended AFTER sid/tenant,
                    // so it renders at whatever position it actually falls
                    // in THIS statement's own numbering.
                    let mut delete_params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(sid.clone()), tenant_param.clone()];
                    let expired =
                        lease_expired_clause("lease_expires_at", kind, &mut delete_params);
                    let records = tx
                        .query(
                            &format!(
                                "DELETE FROM result_tables WHERE source_id = $1 \
                                   AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL)) \
                                   AND (status <> 'building' OR {expired}) \
                                 RETURNING *"
                            ),
                            &delete_params,
                            parse_row,
                        )
                        .await?;
                    // Fresh statement-level snapshot: any live `building`
                    // row for this source still (or now) present means the
                    // delete above is not safe to keep. Returning a plain
                    // `Ok` here (as the pre-atomic SELECT-then-DELETE shape
                    // did, when it ran this check BEFORE any DELETE) would
                    // COMMIT the DELETE that already ran above — `Busy` is a
                    // real transaction error precisely so the backend's
                    // `transaction()` rolls the whole thing back. A SEPARATE
                    // statement needs its OWN `expired` (its own bind
                    // position, if any — a fresh `Vec` numbered from this
                    // statement's own $1).
                    let mut busy_params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(sid), tenant_param];
                    let expired = lease_expired_clause("lease_expires_at", kind, &mut busy_params);
                    let busy = tx
                        .query_opt(
                            &format!(
                                "SELECT table_name FROM result_tables WHERE source_id = $1 \
                                   AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL)) \
                                   AND status = 'building' AND NOT {expired} LIMIT 1"
                            ),
                            &busy_params,
                            |row| row.get::<String>("table_name"),
                        )
                        .await?;
                    if let Some(table) = busy {
                        return Err(BackendError::Busy(table));
                    }
                    Ok(records)
                })
            })
            .await;
        match outcome {
            Ok(records) => Ok(records),
            Err(BackendError::Busy(table)) => Err(JammiError::SourceBusy {
                source_id: source_id.to_string(),
                table,
            }),
            Err(e) => Err(e.into()),
        }
    }

    /// Resolve which embedding table to use for a source. Tenant-filtered.
    pub async fn resolve_embedding_table(
        &self,
        source_id: &str,
        table_name: Option<&str>,
    ) -> Result<ResultTableRecord> {
        if let Some(name) = table_name {
            return self
                .get_result_table(name)
                .await?
                .ok_or_else(|| JammiError::Catalog(format!("Result table '{name}' not found")));
        }

        // Derive the embedding-task list from `ModelTask::ALL` so that
        // adding a future embedding variant automatically extends this
        // resolver — the enum is the single source of truth, not a
        // hardcoded `task IN ('text_embedding', 'image_embedding')`
        // literal. Mirrors the dynamic-placeholder idiom that
        // `find_result_tables` above uses for its conditional binds.
        let embedding_tasks: Vec<&'static str> = ModelTask::ALL
            .iter()
            .filter(|t| t.is_embedding())
            .map(|t| t.as_db_str())
            .collect();
        if embedding_tasks.is_empty() {
            return Err(JammiError::Catalog(
                "ModelTask defines no embedding variants — resolver cannot run".into(),
            ));
        }

        let sid = source_id.to_string();
        let tenant = self.current_tenant();

        // $1 = source_id; $2..$(1+N) = embedding tasks; $(2+N) = tenant.
        let mut params: Vec<SqlValue<'static>> = Vec::with_capacity(embedding_tasks.len() + 2);
        params.push(SqlValue::TextOwned(sid));
        let task_placeholders: Vec<String> = (0..embedding_tasks.len())
            .map(|i| format!("${}", i + 2))
            .collect();
        for t in &embedding_tasks {
            params.push(SqlValue::Text(t));
        }
        let tenant_placeholder = format!("${}", params.len() + 1);
        params.push(SqlValue::from(tenant.map(|t| t.to_string())));

        // `kind = 'model'` excludes derived tables (e.g. a neighbor-graph edge
        // relation) whose `task` column still names the source embedding's
        // task — only genuine model outputs resolve as an embedding source.
        //
        // `created_at` is app-supplied (`backend::now_sortable`) at
        // nanosecond resolution and identical in shape on both backends, so
        // it is the correct primary ordering key — no `rowid` (SQLite has
        // one, Postgres does not; this query used to hard-error on Postgres
        // reaching for it). `table_name DESC` is a deterministic final
        // tiebreak, not a correctness guarantee: `now_sortable` is wall-clock
        // (`chrono::Utc::now`), which is not monotonic, so a coarse or
        // backward clock step could in principle collide two genuinely
        // distinct creation instants. The tiebreak resolves a true
        // same-nanosecond collision correctly (every table name carries a
        // uuid suffix, so the pick is at least deterministic); it does not
        // repair a clock-caused false collision between otherwise-ordered
        // rows.
        let sql = format!(
            "SELECT * FROM result_tables \
             WHERE source_id = $1 AND task IN ({tasks}) \
               AND kind = 'model' \
               AND status = 'ready' \
               AND (tenant_id = {tenant} OR tenant_id IS NULL) \
             ORDER BY created_at DESC, table_name DESC LIMIT 1",
            tasks = task_placeholders.join(", "),
            tenant = tenant_placeholder,
        );

        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query_opt(&sql, &params, parse_row).await }),
            )
            .await?;
        found.ok_or_else(|| {
            JammiError::Catalog(format!("No ready embedding table for source '{source_id}'"))
        })
    }

    /// Retrieve the last checkpoint for a result table.
    pub async fn get_checkpoint(&self, name: &str) -> Result<Option<usize>> {
        let name = name.to_string();
        let tenant = self.current_tenant();
        let found: Option<Option<i32>> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT checkpoint FROM result_tables WHERE table_name = $1 \
                               AND (tenant_id = $2 OR tenant_id IS NULL)",
                            &[
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            |row| row.try_get::<i32>("checkpoint"),
                        )
                        .await
                    })
                },
            )
            .await?;
        Ok(found.flatten().map(|c| c as usize))
    }
}
