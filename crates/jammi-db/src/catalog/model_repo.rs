use crate::catalog::backend::{
    BackendError, BackendKind, IsolationLevel, Row, SqlValue, Transaction, TxOptions,
};
use crate::catalog::lease::{canonical_stamp_now, stale_before_clause};
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

use super::status::JobStatus;
use super::Catalog;

/// Construct the catalog primary key for a model — the single source of truth
/// for model identity in `models.model_id`.
///
/// The key is tenant-qualified so two tenants registering the same
/// `name`/`version` occupy distinct rows instead of colliding on a global PK:
///
/// - global model (`tenant = None`): `"{name}::{version}"`. Left unqualified so
///   a tenant's training job can carry a single-column `base_model_id` FK to a
///   global base model, and so re-registering a global base model stays
///   idempotent.
/// - tenant-scoped model (`tenant = Some(t)`): `"{t}::{name}::{version}"`.
///
/// This is the *only* place a model PK is built; every reference site uses the
/// PK off the resolved [`ModelRecord`] rather than reconstructing it.
pub(crate) fn model_pk(tenant: Option<TenantId>, name: &str, version: i64) -> String {
    match tenant {
        Some(t) => format!("{t}::{name}::{version}"),
        None => format!("{name}::{version}"),
    }
}

/// Materialized row from the `models` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelRecord {
    /// Model name (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`). Tenants
    /// may each own a row under the same name; the row identity is
    /// [`Self::catalog_pk`], not this name.
    pub model_id: String,
    /// Catalog primary key for this exact row (`models.model_id`). Reference
    /// sites (a training job's `base_model_id`, an eval run's `model_id`) use
    /// this PK so they bind to the resolved row — a global base model, or the
    /// caller's own tenant-scoped row — rather than reconstructing
    /// `name::version`.
    pub catalog_pk: String,
    /// Monotonically increasing version number for this model name.
    pub version: i32,
    /// Model category (e.g., `"embedding"`, `"llm"`, `"lora"`).
    pub model_type: String,
    /// Parent model this was derived from (fine-tuned or adapted).
    pub base_model_id: Option<String>,
    /// Inference backend (e.g., `"candle"`, `"vllm"`, `"http"`).
    pub backend: String,
    /// Task this model performs.
    pub task: ModelTask,
    /// Filesystem path to model weights or adapter files.
    pub artifact_path: Option<String>,
    /// Serialized JSON blob with backend-specific configuration.
    pub config_json: Option<String>,
    /// Lifecycle status (e.g., `"registered"`, `"loaded"`, `"failed"`).
    pub status: String,
    /// ISO-8601 timestamp of initial registration.
    pub created_at: String,
    /// The materialization-contract definition hash (migration
    /// `model_materialization`) — the indexable summary of this model's
    /// `materialization.json` sidecar (no leading dot — a fixed name under
    /// the model's artifact prefix, `ArtifactStore::MATERIALIZATION_NAME`,
    /// unlike a result table's `{table}.materialization.json`), mirroring
    /// `result_tables.definition_hash`. `None` for a model with no
    /// materialization (a directly-registered base model, a
    /// `ContextPredictor`) or a pre-migration row.
    pub definition_hash: Option<String>,
    /// The materialization-contract input anchors as canonical JSON — the
    /// indexable summary [`Catalog::probe_model_by_definition`] matches
    /// against, mirroring `result_tables.input_anchors_json`. `None`
    /// alongside [`Self::definition_hash`].
    pub input_anchors_json: Option<String>,
}

/// Registry introspection for one registered model — the client-facing
/// projection of a [`ModelRecord`].
///
/// This is the model peer of [`SourceDescriptor`](super::source_repo::SourceDescriptor):
/// it carries only the fields a client keys off (the model's id, inference
/// backend, task, and lifecycle status), so every transport — the embedded
/// session, the gRPC `Model` projection, and the remote client — reads the same
/// shape. The record's server-internal bookkeeping (version counter,
/// derived-from lineage, artifact path, config blob, registration timestamp)
/// stays in [`ModelRecord`] and never reaches a client.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelDescriptor {
    /// The model's name (an HF repo id or a fine-tuned id).
    pub model_id: String,
    /// Inference backend (e.g. `"candle"`, `"vllm"`, `"http"`).
    pub backend: String,
    /// Task this model performs.
    pub task: ModelTask,
    /// Lifecycle status (e.g. `"registered"`, `"loaded"`, `"failed"`).
    pub status: String,
}

impl From<&ModelRecord> for ModelDescriptor {
    fn from(record: &ModelRecord) -> Self {
        Self {
            model_id: record.model_id.clone(),
            backend: record.backend.clone(),
            task: record.task,
            status: record.status.clone(),
        }
    }
}

/// Input parameters for [`Catalog::register_model`].
#[derive(Debug)]
pub struct RegisterModelParams<'a> {
    /// Unique model name.
    pub model_id: &'a str,
    /// Version number for this registration.
    pub version: i32,
    /// Model category (e.g., `"embedding"`, `"llm"`).
    pub model_type: &'a str,
    /// Inference backend identifier.
    pub backend: &'a str,
    /// Task this model performs.
    pub task: ModelTask,
    /// Optional parent model ID (for fine-tuned variants).
    pub base_model_id: Option<&'a str>,
    /// Optional filesystem path to model weights.
    pub artifact_path: Option<&'a str>,
    /// Optional JSON blob with backend-specific settings.
    pub config_json: Option<&'a str>,
}

const SELECT_COLS: &str =
    "model_id, name, model_type, task, backend, version, status, metadata, artifact_path, \
     created_at, definition_hash, input_anchors_json";

impl Catalog {
    /// Register or refresh a model in the catalog. The session's bound
    /// tenant is written to `tenant_id` and asserted before INSERT.
    ///
    /// `artifact_path` is the served commit pointer a reload resolves the
    /// model's bytes from. On a re-registration (`ON CONFLICT`) it is updated
    /// with `COALESCE(excluded, existing)`: a `Some` path sets it, a `None`
    /// leaves whatever is already committed in place. So this call can *set*
    /// the path (a directly-registered base model) but can never *clear* nor
    /// overwrite a committed path to `NULL` — the path a finalized training
    /// job serves is meant to be written by the lease-guarded finalize
    /// write's own CAS (its caller's own transaction, composed on top of the
    /// `jobs`/`models` primitives this crate exposes), never by a worker's
    /// pre-finalize or a zombie's late `register_model`.
    pub async fn register_model(&self, params: RegisterModelParams<'_>) -> Result<()> {
        let tenant = self.current_tenant();
        let pk = model_pk(tenant, params.model_id, params.version as i64);
        // The served path is a dedicated column (a single-writer commit
        // pointer), not a `metadata` field; the blob carries only the
        // descriptive bits.
        let metadata = serde_json::json!({
            "base_model_id": params.base_model_id,
            "config_json": params.config_json,
        })
        .to_string();
        let model_id = params.model_id.to_string();
        let model_type = params.model_type.to_string();
        let task = params.task.as_db_str();
        let backend = params.backend.to_string();
        let version = params.version as i64;
        let artifact_path = params.artifact_path.map(str::to_string);
        // `models.created_at` is compared (`list_models`'s `ORDER BY
        // created_at`) and `models.updated_at` joins the schema-edge
        // enforced domain unconditionally (`catalog::lease`'s universe gate,
        // R4) — both are bound explicitly, in `CANONICAL_STAMP` shape,
        // rather than left to the schema's `DEFAULT (CAST(CURRENT_TIMESTAMP
        // AS TEXT))` / a literal `CAST(CURRENT_TIMESTAMP AS TEXT)` in the SET
        // clause, either of which renders a THIRD, backend-native shape.
        let now = canonical_stamp_now();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    tx.execute(
                        "INSERT INTO models (model_id, name, model_type, task, backend, version, status, metadata, artifact_path, tenant_id, created_at, updated_at) \
                         VALUES ($1, $2, $3, $4, $5, $6, 'registered', $7, $8, $9, $10, $11) \
                         ON CONFLICT(model_id) DO UPDATE SET \
                             metadata = excluded.metadata, \
                             backend = excluded.backend, \
                             task = excluded.task, \
                             model_type = excluded.model_type, \
                             artifact_path = COALESCE(excluded.artifact_path, models.artifact_path), \
                             updated_at = excluded.updated_at",
                        &[
                            SqlValue::TextOwned(pk),
                            SqlValue::TextOwned(model_id),
                            SqlValue::TextOwned(model_type),
                            SqlValue::Text(task),
                            SqlValue::TextOwned(backend),
                            SqlValue::Int(version),
                            SqlValue::TextOwned(metadata),
                            SqlValue::from(artifact_path),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                            SqlValue::TextOwned(now.clone()),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get the latest version of a model by name. Tenant-filtered.
    ///
    /// This is the reference-resolution path: a training job's base model, an
    /// eval run's model, and the serve/load resolver all bind through it. It
    /// resolves the model regardless of lifecycle status so a job or eval that
    /// references it always binds. The list-facing sense lives in
    /// [`Self::list_models`].
    pub async fn get_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE name = $1 AND (tenant_id = $2 OR tenant_id IS NULL) \
             ORDER BY (tenant_id IS NOT NULL) DESC, version DESC LIMIT 1"
        );
        let mid = model_id.to_string();
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
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(mid),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_model_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Get a specific version of a model.
    pub async fn get_model_version(
        &self,
        model_id: &str,
        version: i32,
    ) -> Result<Option<ModelRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE name = $1 AND version = $2 \
               AND (tenant_id = $3 OR tenant_id IS NULL) \
             ORDER BY (tenant_id IS NOT NULL) DESC LIMIT 1"
        );
        let mid = model_id.to_string();
        let v = version as i64;
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
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(mid),
                                SqlValue::Int(v),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_model_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Hard-delete a model row, removing it entirely — so it is refused while
    /// any reference still points at the model, to avoid orphaning those edges.
    ///
    /// The referential scan covers all five edges that target a model, each
    /// matched on the key that edge actually stores:
    ///
    /// - `result_tables.model_id` — the model NAME (no FK).
    /// - `jobs.output_model_id` — the model NAME (no FK).
    /// - `jobs.model_source` — the model NAME (no FK): the `ModelSource`
    ///   string a compute kind (`embedding`/`infer`) resolves its model
    ///   against, the exact vocabulary `result_tables.model_id` carries, so a
    ///   compute job still reading a model blocks that model's delete the
    ///   same way the table it will produce does once written.
    /// - `jobs.model_ref` — the catalog PK (FK to `models`).
    /// - `eval_runs.model_id` — the catalog PK (FK to `models`).
    ///
    /// The two FK-backed edges are scanned in the engine and surface the typed
    /// [`JammiError::ModelReferenced`] just like the no-FK edges — the database
    /// FK is never the thing that rejects the DELETE, because a raw constraint
    /// violation would leak as an opaque backend error.
    ///
    /// **The three `jobs` edges are age-gated (N9): a row counts as a blocking
    /// reference only while it is non-terminal OR younger than
    /// `retention_days`** — an age PREDICATE evaluated fresh on every scan,
    /// never a sweep-dependent flag. A terminal `jobs` row (`completed` /
    /// `failed`) past the window does not block; a non-terminal row blocks
    /// indefinitely regardless of age; a young terminal row still blocks. The
    /// predicate is evaluated fresh on every scan, exactly like every other
    /// tenant/admin-scope decision on this table — no separate reaper needs to
    /// run first for a delete to succeed.
    ///
    /// Tenant scope is strict — a session deletes only a row whose owner equals
    /// its OWN tenant (`tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`),
    /// so a tenant cannot delete a GLOBAL or a peer's model. An absent row is
    /// `NotFound` unless `if_exists` is set, in which case it is a success no-op.
    ///
    /// The scan and the DELETE run in a single `Serializable` transaction: the
    /// three no-FK edges (`result_tables.model_id`, `jobs.output_model_id`,
    /// `jobs.model_source`) have no constraint backstop, so a weaker isolation
    /// level would admit a concurrent insert between the scan and the delete.
    pub async fn delete_model(
        &self,
        model_id: &str,
        version: Option<i32>,
        if_exists: bool,
        retention_days: i64,
    ) -> Result<()> {
        let record = match version {
            Some(v) => self.get_model_version(model_id, v).await?,
            None => self.get_model(model_id).await?,
        };
        let record = match record {
            Some(r) => r,
            None if if_exists => return Ok(()),
            None => {
                return Err(JammiError::ModelNotFound {
                    model_id: model_id.to_string(),
                })
            }
        };
        let pk = record.catalog_pk;
        let name = record.model_id;
        let tenant = self.current_tenant();
        let backend_kind = self.backend().backend_kind();

        // The scan and the DELETE share one `Serializable` transaction. A
        // discovered reference is carried OUT through the success value (not a
        // `BackendError`), so the typed `ModelReferenced` is raised here, where
        // its `model_id`/`referenced_by` are in scope, rather than round-tripped
        // through the backend-error channel.
        let outcome = self
            .backend()
            .transaction(
                TxOptions {
                    isolation: IsolationLevel::Serializable,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.set_tenant(tenant);
                        tx.assert_tenant_matches(tenant, "models")?;
                        let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));

                        // Referential scan — each edge keyed by what it stores
                        // (NAME for the three no-FK edges, PK for the two
                        // FK-backed ones), tenant-scoped with the same strict
                        // predicate as the delete below.
                        let referenced_by = scan_model_references(
                            tx,
                            &name,
                            &pk,
                            &tenant_val,
                            backend_kind,
                            retention_days,
                        )
                        .await?;
                        if !referenced_by.is_empty() {
                            return Ok(DeleteOutcome::Referenced(referenced_by));
                        }

                        let affected = tx
                            .execute(
                                "DELETE FROM models \
                                 WHERE model_id = $1 \
                                   AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                                &[SqlValue::TextOwned(pk), tenant_val],
                            )
                            .await?;
                        Ok(DeleteOutcome::Deleted(affected))
                    })
                },
            )
            .await?;
        match outcome {
            DeleteOutcome::Referenced(referenced_by) => Err(JammiError::ModelReferenced {
                model_id: model_id.to_string(),
                referenced_by,
            }),
            DeleteOutcome::Deleted(0) => Err(JammiError::ModelNotFound {
                model_id: model_id.to_string(),
            }),
            DeleteOutcome::Deleted(_) => Ok(()),
        }
    }

    /// List the models visible to the session's tenant — the peer of
    /// `list_sources`. A reference resolver that binds a single model by name
    /// (provenance, a base-model FK) uses [`Self::get_model`] instead.
    ///
    /// Inside a [`crate::session::JammiSession::with_admin_scope`] closure the
    /// per-row tenant filter is dropped and every tenant's models are returned
    /// — the same admin arm every other catalog enumeration carries.
    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let admin = TenantBinding::is_admin_scope();
        let sql = if admin {
            format!("SELECT {SELECT_COLS} FROM models ORDER BY created_at")
        } else {
            format!(
                "SELECT {SELECT_COLS} FROM models \
                 WHERE (tenant_id = $1 OR tenant_id IS NULL) \
                 ORDER BY created_at"
            )
        };
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
                        let params: Vec<SqlValue<'static>> = if admin {
                            Vec::new()
                        } else {
                            vec![SqlValue::from(tenant.map(|t| t.to_string()))]
                        };
                        tx.query(&sql, &params, parse_model_row).await
                    })
                },
            )
            .await?)
    }

    /// Whole-catalog scan, admin-scoped BY CONSTRUCTION: every `models`
    /// row's (non-`NULL`) `artifact_path`, across EVERY tenant and
    /// untenanted rows, regardless of the calling task's own tenant binding
    /// — this method issues NO tenant predicate at all (never
    /// [`TenantBinding::current_tenant`], never
    /// [`TenantBinding::is_admin_scope`]).
    ///
    /// [`crate::store::reconcile`]'s attribution set is built from THIS
    /// scan, never from [`Self::list_models`] (tenant-scoped): a
    /// tenant-scoped or unbound-but-not-admin-scoped listing can only ever
    /// see its own tenant's (and untenanted) rows, so a tenant-A row
    /// reusing a GLOBAL prefix — the cache-hit fan-out shape, where
    /// [`Self::find_models_by_definition`] returns a `NULL`-tenant row and
    /// the winning job registers a SECOND, tenant-A row pointing at the
    /// same prefix — is invisible to an UNBOUND reconcile pass reading
    /// through [`Self::list_models`]: exactly the gap this method exists to
    /// close. The per-prefix count predicate over this same admin scan is
    /// [`Self::count_models_naming_prefix_all_tenants`].
    ///
    /// Returns raw `artifact_path` strings — the exact value each row's
    /// finalize CAS committed (a full [`crate::storage::StorageUrl`]
    /// string) — never a [`ModelRecord`]: this is a bytes-reachability
    /// scan, not a row read, so it never discloses a model's name, id, or
    /// tenant to whatever tenant scope the caller is running under.
    pub async fn list_model_artifact_paths_all_tenants(&self) -> Result<Vec<String>> {
        let sql = "SELECT artifact_path FROM models WHERE artifact_path IS NOT NULL";
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(sql, &[], |row| row.get::<String>("artifact_path"))
                            .await
                    })
                },
            )
            .await?)
    }

    /// The count of `models` rows — across EVERY tenant and untenanted
    /// rows, admin-scoped BY CONSTRUCTION exactly like
    /// [`Self::list_model_artifact_paths_all_tenants`] (see that method's
    /// doc for why no tenant predicate is issued here) — whose
    /// `artifact_path` names `key_or_prefix` as ITSELF or as its
    /// IMMEDIATE containing directory.
    ///
    /// A row's `artifact_path` is a FLAT directory of files (every
    /// `put_artifact` bundle this store ever writes — served, resume, or
    /// epoch-checkpoint — is a flat file list, never a bundle nested inside
    /// its own subdirectories), so a byte a row's own publish is
    /// responsible for is either the row's `artifact_path` itself (an
    /// epoch checkpoint published UNDER a served attempt's prefix is
    /// itself registered as its OWN row whose `artifact_path` EQUALS that
    /// exact checkpoint prefix — the winning finalize CAS inserts one such
    /// row per RETAINED checkpoint) or a plain file directly inside it
    /// (e.g. `{attempt}/adapter.safetensors`, whose CONTAINING directory
    /// equals the row's `artifact_path`). The predicate checks both shapes
    /// in one query — `artifact_path = $1 OR artifact_path = $2`, where
    /// `$2` is `key_or_prefix`'s own immediate parent directory (computed
    /// in Rust, `rsplit_once('/')`, never a SQL `LIKE`/`SUBSTR` walk) —
    /// and DELIBERATELY stops at one level: an ANCESTOR further up (e.g.
    /// the served attempt's row, relative to an UNRETAINED epoch
    /// checkpoint nested two segments deeper under
    /// `checkpoints/epoch_N/`) is NEVER treated as referencing it. Each
    /// independently-registered nested artifact carries its OWN row when
    /// retained; an ancestor's row is never inherited protection for a
    /// SEPARATE artifact merely because it happens to sit somewhere below
    /// it in the physical layout — the whole reason an unretained epoch
    /// checkpoint must stay reclaimable even while its enclosing served
    /// attempt is very much alive and referenced. Indexed by
    /// `idx_models_artifact_path` (migration 033) — this predicate issues
    /// exactly ONE indexed lookup (matching either of two exact values)
    /// per candidate object or prefix a caller is about to delete, an
    /// acceptable cost for a byte-deleter that already performs its own
    /// I/O per candidate.
    ///
    /// `_resume/` is the one namespace this predicate is never expected to
    /// match: it is a SIBLING of a job's attempt-level artifact paths
    /// (`{job}/_resume` vs. `{job}/{worker}/{attempt}`), so neither
    /// `key_or_prefix` nor its immediate parent can ever equal a served or
    /// checkpoint row's `artifact_path` — proven by an executed test, see
    /// `a_resume_checkpoint_prefix_is_never_referenced_even_under_the_containment_aware_predicate`
    /// in `tests/it/reconcile.rs`.
    ///
    /// Discloses a COUNT ONLY — never row ids, model names, or tenant ids
    /// — so a tenant-bound caller consulting this predicate (through
    /// [`crate::store::ResultStore::prefix_is_referenced`]) learns only
    /// "referenced" vs. "not", never by whom.
    pub async fn count_models_naming_prefix_all_tenants(&self, key_or_prefix: &str) -> Result<i64> {
        let parent = key_or_prefix
            .rsplit_once('/')
            .map(|(parent, _leaf)| parent.to_string());
        let sql = "SELECT COUNT(*) AS n FROM models WHERE artifact_path = $1 OR artifact_path = $2";
        let key_or_prefix = key_or_prefix.to_string();
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
                            sql,
                            &[SqlValue::TextOwned(key_or_prefix), SqlValue::from(parent)],
                            |row| row.get::<i64>("n"),
                        )
                        .await
                        .map(|opt| opt.unwrap_or(0))
                    })
                },
            )
            .await?)
    }

    /// Every SERVABLE `models` row carrying exactly `definition_hash`, newest
    /// first — the raw candidate set [`Self::probe_model_by_definition`]
    /// narrows with the exact anchor match. Mirrors
    /// [`Catalog::find_ready_result_tables_by_definition`]'s shape for
    /// `result_tables`. Tenant-scoped like every other catalog read.
    ///
    /// The predicate is `definition_hash = $1 AND artifact_path IS NOT NULL
    /// AND (tenant_id = $2 OR tenant_id IS NULL)` — ONE predicate, both
    /// halves load-bearing:
    ///
    /// - `definition_hash = $1` can never match a `NULL` column (SQL's
    ///   three-valued equality), so a pre-migration or non-materialized row
    ///   (`ContextPredictor`, a directly-registered base model) is excluded
    ///   without a separate guard.
    /// - `artifact_path IS NOT NULL` restricts the candidate set to rows the
    ///   finalize CAS ([`Catalog::finish_job_with_model`]) has already
    ///   committed: a row a losing or still-running attempt registered
    ///   with `definition_hash` set but no committed artifact — which would
    ///   otherwise poison every future `cache=Use` probe for that definition
    ///   forever, since the row never becomes servable on its own — is
    ///   excluded from the candidate set rather than merely filtered
    ///   downstream. A miss caused by this predicate is exactly that: a
    ///   miss, never an error, so the caller trains.
    ///
    /// **Tenant fan-out convention.** `(tenant_id = $2 OR tenant_id IS
    /// NULL)` is the same relaxed READ convention every other nullable-
    /// tenant catalog probe uses ([`Self::get_model`]/[`Self::get_model_version`]'s
    /// global-base-model resolution, [`Catalog::find_ready_result_tables_by_definition`]):
    /// a `NULL`-tenant model row is a cache-hit CANDIDATE FOR EVERY TENANT —
    /// a global fine-tune output is reusable by any caller, exactly like a
    /// global base model is loadable by any caller — while a tenant-owned
    /// row is visible only to that exact tenant, never a peer's. When a
    /// caller's own row exists but is unservable, the caller falls
    /// through to a matching global row rather than missing outright; when
    /// a caller has no own row at all, the global row is the only candidate.
    /// This is a READ-side relaxation only: the WRITE side
    /// ([`Self::record_model_materialization`]) stays STRICT
    /// (`tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`), so a tenant
    /// session can never populate another tenant's row, only its own or —
    /// from an explicitly unscoped session — the global one.
    pub async fn find_models_by_definition(
        &self,
        definition_hash: &str,
    ) -> Result<Vec<ModelRecord>> {
        let hash = definition_hash.to_string();
        let tenant = self.current_tenant();
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE definition_hash = $1 AND artifact_path IS NOT NULL \
               AND (tenant_id = $2 OR tenant_id IS NULL) \
             ORDER BY created_at DESC"
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
                        tx.query(
                            &sql,
                            &[
                                SqlValue::TextOwned(hash),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_model_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Find a model row already materialised by the EXACT same definition
    /// over the EXACT same input anchors — the model peer of
    /// [`crate::store::ResultStore::probe_cache_record`]'s `result_tables`
    /// probe, restated over `models` because a fine-tuned model is not a
    /// `result_tables` row (the reuse rule: definition hash AND pinned equal
    /// anchors; a plain/unpinned source is never reused).
    ///
    /// `NULL` never matches: [`Self::find_models_by_definition`]'s own
    /// predicate already excludes every row with no recorded
    /// `definition_hash`, so a model with no materialization can never be a
    /// cache-hit candidate. The candidate set is additionally restricted to
    /// the SERVABLE set (`artifact_path IS NOT NULL`) by that same
    /// predicate, so a hash-bearing row a losing/zombie attempt left behind
    /// is never a hit either — this function issues no SQL of its own and so
    /// inherits both halves automatically. An anchor set containing an
    /// [`crate::store::manifest::AnchorKind::UnpinnedAtInstant`] anchor is
    /// likewise never a hit — an unpinned input's current instant proves
    /// nothing about what the training set actually was, so no recorded
    /// model can be a sound reuse of it (the same rule
    /// `ResultStore::exact_match_candidates` applies).
    ///
    /// When several rows share the exact key (a reuse chain: job C reused
    /// job B's prefix, which reused job A's), the newest one wins, by a
    /// deterministic TOTAL order this function imposes in Rust rather than
    /// trusting the catalog's `ORDER BY` (r32: `RETURNING`/`ORDER BY` order
    /// is never trusted as the tie-break of record) — `created_at` alone can
    /// tie at whatever timestamp resolution a backend renders, so ties break
    /// on `catalog_pk` DESCENDING, the same shape
    /// `ResultStore::probe_ready_training_set` uses for `table_name`. This is
    /// a pure sensor over the catalog's `definition_hash` index; it does not
    /// check whether the row's artifact prefix still exists on disk (that
    /// check, if the caller needs it, composes on top through
    /// [`crate::store::ArtifactStore::read_model_materialization`]).
    pub async fn probe_model_by_definition(
        &self,
        definition_hash: &str,
        anchors: &[crate::store::manifest::InputAnchor],
    ) -> Result<Option<ModelRecord>> {
        use crate::store::manifest::AnchorKind;

        if anchors
            .iter()
            .any(|a| a.kind == AnchorKind::UnpinnedAtInstant)
        {
            return Ok(None);
        }
        let mut exact = Vec::new();
        for candidate in self.find_models_by_definition(definition_hash).await? {
            let Some(ref anchors_json) = candidate.input_anchors_json else {
                continue;
            };
            let recorded: Vec<crate::store::manifest::InputAnchor> =
                serde_json::from_str(anchors_json)?;
            if anchor_sets_equal(&recorded, anchors) {
                exact.push(candidate);
            }
        }
        exact.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| b.catalog_pk.cmp(&a.catalog_pk))
        });
        Ok(exact.into_iter().next())
    }

    /// Record a fine-tuned model's materialization-contract summary (the
    /// `model_materialization` migration's two columns) — called after the
    /// model's artifact prefix's `materialization.json` sidecar has been
    /// written ([`crate::store::ArtifactStore::write_model_materialization`],
    /// itself written LAST, after the bundle's own `manifest.json`), and
    /// only AFTER the finalize CAS ([`Catalog::finish_job_with_model`]) has
    /// already committed this exact row's `artifact_path`.
    /// Tenant-scoped with the same STRICT predicate [`Self::delete_model`]
    /// uses (`tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`).
    ///
    /// **The ordering guard.** `models` carries no per-attempt identity of
    /// its own (unlike `jobs.claimed_by`/`jobs.attempts`) to bind a lease
    /// directly against, so the predicate binds to the ONE fact only the
    /// winning attempt's finalize CAS can have produced on this row:
    /// `artifact_path IS NOT NULL`. `artifact_path` is written exactly once,
    /// unconditionally, by `finish_job_with_model`'s attempt-guarded
    /// transaction — a losing or still-running attempt's row never carries
    /// it — so requiring it here means a losing/zombie attempt's call can
    /// never win this `UPDATE` and leave a hash-bearing row for
    /// [`Self::find_models_by_definition`] to have to filter out: the
    /// row simply never becomes probe-eligible in the first place.
    ///
    /// Unconditional `SET` (never a `COALESCE`) for the two columns it does
    /// write: a model's materialization summary is written exactly once, by
    /// the same finalize sequence that already committed its `artifact_path`,
    /// so there is no "leave unchanged on re-registration" case to protect
    /// here.
    ///
    /// Refuses distinctly depending on why zero rows matched:
    /// [`JammiError::ModelNotFound`] when no row exists at all for
    /// `model_id` (this name), `version`, and the caller's tenant; a typed
    /// [`JammiError::Model`] precondition failure when the row exists but
    /// `artifact_path` is still `NULL` — the finalize CAS has not won for
    /// this attempt yet, so recording a definition hash now would create
    /// exactly the poisoned, hash-bearing-but-unservable row
    /// [`Self::find_models_by_definition`]'s predicate keeps unreachable.
    pub async fn record_model_materialization(
        &self,
        model_id: &str,
        version: i32,
        definition_hash: &str,
        input_anchors_json: &str,
    ) -> Result<()> {
        let tenant = self.current_tenant();
        let model_id_for_tx = model_id.to_string();
        let version_i64 = version as i64;
        let definition_hash = definition_hash.to_string();
        let input_anchors_json = input_anchors_json.to_string();

        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));
                    let affected = tx
                        .execute(
                            "UPDATE models SET definition_hash = $1, input_anchors_json = $2, \
                             updated_at = $6 \
                             WHERE name = $3 AND version = $4 \
                               AND (tenant_id = $5 OR (tenant_id IS NULL AND $5 IS NULL)) \
                               AND artifact_path IS NOT NULL",
                            &[
                                SqlValue::TextOwned(definition_hash),
                                SqlValue::TextOwned(input_anchors_json),
                                SqlValue::TextOwned(model_id_for_tx.clone()),
                                SqlValue::Int(version_i64),
                                tenant_val.clone(),
                                SqlValue::TextOwned(canonical_stamp_now()),
                            ],
                        )
                        .await?;
                    if affected == 1 {
                        return Ok(RecordMaterializationOutcome::Recorded);
                    }
                    // Disambiguate a missing row from an unfinalized one so
                    // the caller gets a precise typed refusal rather than a
                    // misleading `ModelNotFound` for a row that DOES exist.
                    let exists = tx
                        .query_opt(
                            "SELECT 1 AS one FROM models \
                             WHERE name = $1 AND version = $2 \
                               AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL))",
                            &[
                                SqlValue::TextOwned(model_id_for_tx),
                                SqlValue::Int(version_i64),
                                tenant_val,
                            ],
                            |row| row.get::<i32>("one"),
                        )
                        .await?
                        .is_some();
                    Ok(if exists {
                        RecordMaterializationOutcome::RowNotYetFinalized
                    } else {
                        RecordMaterializationOutcome::RowAbsent
                    })
                })
            })
            .await?;

        match outcome {
            RecordMaterializationOutcome::Recorded => Ok(()),
            RecordMaterializationOutcome::RowAbsent => Err(JammiError::ModelNotFound {
                model_id: model_id.to_string(),
            }),
            RecordMaterializationOutcome::RowNotYetFinalized => Err(JammiError::Model {
                model_id: model_id.to_string(),
                message: "cannot record a materialization summary before the finalize CAS \
                          has committed artifact_path for this attempt's row"
                    .to_string(),
            }),
        }
    }

    /// Delete a `models` row still in its pre-finalize state — the failure-
    /// arm cleanup for a fine-tune attempt that registered a row
    /// ([`Self::register_model`], `artifact_path: None`) and then lost its
    /// lease, errored, or was superseded before the finalize CAS
    /// ([`Catalog::finish_job_with_model`]) ever ran. Named beside
    /// [`Self::delete_model`] (the referenced-checked hard delete for a
    /// SERVED model) as the unfinalized-row peer: this one carries no
    /// referential scan because an unfinalized row can carry no outbound
    /// reference yet.
    ///
    /// The guard is `artifact_path IS NULL` — the same fact
    /// [`Self::record_model_materialization`]'s ordering guard requires the
    /// OPPOSITE of. A row the finalize CAS already committed (`artifact_path`
    /// set) is never matched, so this can never delete a servable model out
    /// from under a concurrent reader, even called against the wrong
    /// attempt or a job whose finalize CAS actually won the race the caller
    /// believed it lost. Tenant-scoped with the same STRICT predicate
    /// [`Self::delete_model`] uses.
    ///
    /// Returns `true` when a row was deleted, `false` when no row matched —
    /// already deleted, already finalized, or never registered. A caller
    /// treats `false` as a no-op, never an error: every one of those states
    /// is already the state this call exists to converge on.
    pub async fn delete_registered_model_if_unfinalized(
        &self,
        model_id: &str,
        version: i32,
    ) -> Result<bool> {
        let tenant = self.current_tenant();
        let model_id = model_id.to_string();
        let version_i64 = version as i64;
        let affected = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));
                    tx.execute(
                        "DELETE FROM models WHERE name = $1 AND version = $2 \
                           AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL)) \
                           AND artifact_path IS NULL",
                        &[
                            SqlValue::TextOwned(model_id),
                            SqlValue::Int(version_i64),
                            tenant_val,
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(affected == 1)
    }
}

/// In-transaction outcome of [`Catalog::record_model_materialization`]'s
/// guarded `UPDATE`, distinguishing "no such row" from "row exists but the
/// finalize CAS has not committed `artifact_path` yet" so the caller gets a
/// precise typed refusal ([`JammiError::ModelNotFound`] vs
/// [`JammiError::Model`]) rather than one error shape standing in for two
/// distinct preconditions.
enum RecordMaterializationOutcome {
    Recorded,
    RowAbsent,
    RowNotYetFinalized,
}

/// In-transaction outcome of [`Catalog::delete_model`]'s scan-then-delete: the
/// model is either still referenced (carry the blocking edges out so the typed
/// [`JammiError::ModelReferenced`] is raised by the caller) or deleted (carry
/// the affected-row count to distinguish a hit from a vanished/cross-tenant
/// row).
enum DeleteOutcome {
    Referenced(Vec<String>),
    Deleted(u64),
}

/// One scanned reference edge: the generic edge name surfaced in
/// [`JammiError::ModelReferenced`], the table/column to count, and which key the
/// edge stores — the model NAME for the no-FK edges, the catalog PK for the
/// FK-backed ones.
struct ReferenceEdge {
    /// Generic edge name (e.g. `result_tables`, `jobs.output_model_id`).
    name: &'static str,
    /// The `COUNT(*)` query, tenant-scoped with the strict predicate.
    sql: &'static str,
    /// Whether the edge's value column holds the model PK (`true`) or the model
    /// NAME (`false`).
    keyed_by_pk: bool,
}

/// The two non-job edges that reference a model, tenant-scoped with the same
/// strict predicate the DELETE uses. `result_tables.model_id` holds the model
/// NAME and has no FK; `eval_runs.model_id` holds the catalog PK and is
/// FK-backed. The three `jobs` edges (`model_ref`, `output_model_id`,
/// `model_source`) are scanned separately by [`scan_model_references`] — they
/// need one more bind (`retention_days`) and a backend-specific age clause
/// (N9) neither of these static templates carries.
const REFERENCE_EDGES: [ReferenceEdge; 2] = [
    ReferenceEdge {
        name: "result_tables",
        sql: "SELECT COUNT(*) AS n FROM result_tables \
              WHERE model_id = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
        keyed_by_pk: false,
    },
    ReferenceEdge {
        name: "eval_runs",
        sql: "SELECT COUNT(*) AS n FROM eval_runs \
              WHERE model_id = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
        keyed_by_pk: true,
    },
];

/// Count every reference edge that still points at the model and return the
/// generic names of the non-empty ones — the two static [`REFERENCE_EDGES`]
/// plus the three `jobs` edges (`model_ref`, PK-keyed; `output_model_id` and
/// `model_source`, NAME-keyed), each age-gated (N9): a `jobs` row counts as
/// blocking only while its status is non-terminal
/// ([`JobStatus::terminal_sql_list`] renders the terminal set) OR it is younger than
/// `retention_days` — evaluated with [`stale_before_clause`] on
/// `jobs.updated_at`, negated (a row counts when it is NOT stale-and-terminal).
/// The FK-backed edges are scanned here (rather than left to the database FK)
/// so they raise the same typed [`JammiError::ModelReferenced`] as the no-FK
/// edges instead of leaking a raw constraint violation. `tenant_val` is bound
/// to every count.
async fn scan_model_references(
    tx: &mut Transaction<'_>,
    name: &str,
    pk: &str,
    tenant_val: &SqlValue<'static>,
    kind: BackendKind,
    retention_days: i64,
) -> std::result::Result<Vec<String>, BackendError> {
    let mut referenced_by = Vec::new();
    for edge in &REFERENCE_EDGES {
        let key = if edge.keyed_by_pk { pk } else { name };
        let count = tx
            .query_opt(
                edge.sql,
                &[SqlValue::TextOwned(key.to_string()), tenant_val.clone()],
                |row| row.get::<i64>("n"),
            )
            .await?
            .unwrap_or(0);
        if count > 0 {
            referenced_by.push(edge.name.to_string());
        }
    }

    let retention = std::time::Duration::from_secs((retention_days.max(0) as u64) * 86_400);
    let terminal = JobStatus::terminal_sql_list();
    for (edge_name, column, key) in [
        ("jobs.model_ref", "model_ref", pk),
        ("jobs.output_model_id", "output_model_id", name),
        ("jobs.model_source", "model_source", name),
    ] {
        // `$1`/`$2` (key, tenant) bind first; the retention clause appends its
        // own bind(s) after, so both fragments share one params vector built
        // in one pass — never two separately-numbered queries.
        let mut params = vec![SqlValue::TextOwned(key.to_string()), tenant_val.clone()];
        let stale = stale_before_clause("updated_at", kind, retention, &mut params);
        let sql = format!(
            "SELECT COUNT(*) AS n FROM jobs \
             WHERE {column} = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL)) \
               AND NOT (status IN ({terminal}) AND {stale})"
        );
        let count = tx
            .query_opt(&sql, &params, |row| row.get::<i64>("n"))
            .await?
            .unwrap_or(0);
        if count > 0 {
            referenced_by.push(edge_name.to_string());
        }
    }
    Ok(referenced_by)
}

/// Parse: model_id, name, model_type, task, backend, version, status, metadata,
/// artifact_path, created_at, definition_hash, input_anchors_json
fn parse_model_row(row: &Row<'_>) -> std::result::Result<ModelRecord, BackendError> {
    let catalog_pk: String = row.get("model_id")?;
    let name: String = row.get("name")?;
    let model_type: String = row.get("model_type")?;
    let task_raw: String = row.get("task")?;
    let task = ModelTask::try_from_db_str(&task_raw).map_err(|e| BackendError::TypeConversion {
        column: "task".into(),
        detail: e.to_string(),
    })?;
    let backend: String = row.try_get("backend")?.unwrap_or_default();
    let version: i32 = row.try_get("version")?.unwrap_or(1);
    let status: String = row.get("status")?;
    let metadata: Option<String> = row.try_get("metadata")?;
    let created_at: String = row.get("created_at")?;

    // The served path is its own column (the single-writer commit pointer); the
    // `metadata` blob carries only the descriptive `base_model_id`/`config_json`.
    let artifact_path: Option<String> = row.try_get("artifact_path")?;
    let (base_model_id, config_json) = metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .map(|v| {
            (
                v["base_model_id"].as_str().map(String::from),
                v["config_json"].as_str().map(String::from),
            )
        })
        .unwrap_or((None, None));

    // Migration `model_materialization` (033): absent on a pre-migration row
    // or a model with no materialization at all — `try_get` reads a genuinely
    // absent column the same way as a present `NULL`, so a query that omits
    // these columns entirely (an older wire projection) still parses.
    let definition_hash: Option<String> = row.try_get("definition_hash")?;
    let input_anchors_json: Option<String> = row.try_get("input_anchors_json")?;

    Ok(ModelRecord {
        model_id: name,
        catalog_pk,
        version,
        model_type,
        base_model_id,
        backend,
        task,
        artifact_path,
        config_json,
        status,
        created_at,
        definition_hash,
        input_anchors_json,
    })
}

/// Two `InputAnchor` sets are the SAME reuse key iff they are equal as SETS
/// (order-independent — a producer's `Vec<InputAnchor>` is built in producer
/// order, which is not itself a determinant of the reuse key). Duplicated
/// from `crate::store::freshness`'s private helper of the same shape rather
/// than exposed across the `catalog`/`store` boundary this file does not
/// otherwise cross: [`Catalog::probe_model_by_definition`] is the ONE
/// catalog-layer caller that needs it, and the module boundary between
/// catalog primitives and the store's sensing layer is worth keeping even at
/// the cost of this five-line duplication.
fn anchor_sets_equal(
    a: &[crate::store::manifest::InputAnchor],
    b: &[crate::store::manifest::InputAnchor],
) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x)) && b.iter().all(|y| a.contains(y))
}
