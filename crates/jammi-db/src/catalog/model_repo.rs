use crate::catalog::artifact_repo::ArtifactRef;
use crate::catalog::backend::{
    BackendError, BackendKind, IsolationLevel, Row, SqlValue, Transaction, TxOptions,
};
use crate::catalog::lease::{canonical_stamp_now, stale_before_clause, CanonicalStampColumn};
use crate::error::{JammiError, Result};
use crate::storage::{StorageError, StorageUrl};
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;
use crate::ModelTask;

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

/// Where a model's bytes live. A row names them exactly one way.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLocation {
    /// An engine-produced bundle under `models/`: the `model_artifacts` row
    /// the model references (`models.artifact_prefix`). Written only by the
    /// finalize transaction of the job that produced it
    /// ([`Catalog::finish_job_with_model`]).
    Artifact(ArtifactRef),
    /// Where a directly-registered model's bytes live
    /// (`models.external_location`): a base model's local weights directory,
    /// or a bundle written elsewhere — bytes this catalog's engine did not
    /// produce and never deletes.
    External(String),
}

impl ModelLocation {
    /// The storage URL of the bundle a reload fetches
    /// ([`crate::store::ArtifactStore::fetch_artifact`]): the referenced
    /// artifact's prefix, or an external location parsed as one. The two
    /// differ in who owns the bytes, not in how they are read.
    pub fn bundle_url(&self) -> std::result::Result<StorageUrl, StorageError> {
        match self {
            Self::Artifact(artifact) => Ok(artifact.url().clone()),
            Self::External(location) => StorageUrl::parse(location),
        }
    }
}

impl std::fmt::Display for ModelLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Artifact(artifact) => artifact.fmt(f),
            Self::External(location) => f.write_str(location),
        }
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
    /// Where the model's bytes live, or `None` for a row registered before
    /// its weights were ever resolved.
    pub location: Option<ModelLocation>,
    /// Serialized JSON blob with backend-specific configuration.
    pub config_json: Option<String>,
    /// Lifecycle status (e.g., `"registered"`, `"loaded"`, `"failed"`).
    pub status: String,
    /// ISO-8601 timestamp of initial registration.
    pub created_at: String,
}

/// Registry introspection for one registered model — the client-facing
/// projection of a [`ModelRecord`].
///
/// This is the model peer of [`SourceDescriptor`](super::source_repo::SourceDescriptor):
/// it carries only the fields a client keys off (the model's id, inference
/// backend, task, and lifecycle status), so every transport — the embedded
/// session, the gRPC `Model` projection, and the remote client — reads the same
/// shape. The record's server-internal bookkeeping (version counter,
/// derived-from lineage, location, config blob, registration timestamp)
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
    /// Where a directly-registered model's bytes live
    /// ([`ModelLocation::External`]) — the only location a registration can
    /// write.
    pub external_location: Option<&'a str>,
    /// Optional JSON blob with backend-specific settings.
    pub config_json: Option<&'a str>,
}

const SELECT_COLS: &str =
    "model_id, name, model_type, task, backend, version, status, metadata, artifact_prefix, \
     external_location, created_at";

impl Catalog {
    /// Register or refresh a directly-registered model. The session's bound
    /// tenant is written to `tenant_id` and asserted before INSERT.
    ///
    /// The only location this call writes is
    /// [`ModelLocation::External`]. On a re-registration it is updated with
    /// `COALESCE(excluded, existing)`: a `Some` directory sets it, a `None`
    /// leaves whatever is already recorded in place, so the call can *set*
    /// the location but never *clear* it.
    ///
    /// A row that references an artifact ([`ModelLocation::Artifact`]) was
    /// written by the job that produced it
    /// ([`Catalog::finish_job_with_model`]) and is refused here, typed,
    /// untouched: a registration can neither rewrite its type and lineage nor
    /// give it a second location.
    pub async fn register_model(&self, params: RegisterModelParams<'_>) -> Result<()> {
        let tenant = self.current_tenant();
        let pk = model_pk(tenant, params.model_id, params.version as i64);
        let metadata = model_metadata(params.base_model_id, params.config_json);
        let model_id = params.model_id.to_string();
        let model_type = params.model_type.to_string();
        let task = params.task.as_str();
        let backend = params.backend.to_string();
        let version = params.version as i64;
        let external_location = params.external_location.map(str::to_string);
        // `models.created_at` is compared (`list_models`'s `ORDER BY
        // created_at`) and `models.updated_at` joins the schema-edge
        // enforced domain unconditionally (`catalog::lease`'s universe gate)
        // — both are bound explicitly, in `CANONICAL_STAMP` shape, rather
        // than left to a schema default that renders a backend-native shape.
        let now = canonical_stamp_now();

        let written = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    tx.execute(
                        "INSERT INTO models (model_id, name, model_type, task, backend, version, status, metadata, external_location, tenant_id, created_at, updated_at) \
                         VALUES ($1, $2, $3, $4, $5, $6, 'registered', $7, $8, $9, $10, $11) \
                         ON CONFLICT(model_id) DO UPDATE SET \
                             metadata = excluded.metadata, \
                             backend = excluded.backend, \
                             task = excluded.task, \
                             model_type = excluded.model_type, \
                             external_location = COALESCE(excluded.external_location, models.external_location), \
                             updated_at = excluded.updated_at \
                         WHERE models.artifact_prefix IS NULL",
                        &[
                            SqlValue::TextOwned(pk),
                            SqlValue::TextOwned(model_id),
                            SqlValue::TextOwned(model_type),
                            SqlValue::Text(task),
                            SqlValue::TextOwned(backend),
                            SqlValue::Int(version),
                            SqlValue::TextOwned(metadata),
                            SqlValue::from(external_location),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                            SqlValue::TextOwned(now.clone()),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await
                })
            })
            .await?;
        if written == 1 {
            Ok(())
        } else {
            Err(JammiError::Model {
                model_id: params.model_id.to_string(),
                message: "is the output of a training job; its catalog row is written by the \
                          job that produced it and cannot be re-registered"
                    .to_string(),
            })
        }
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
    /// **The three `jobs` edges are age-gated: a row counts as a blocking
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
/// neither of these static templates carries.
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
/// `model_source`, NAME-keyed), each age-gated: a `jobs` row counts as
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
        let stale = stale_before_clause(
            CanonicalStampColumn::JobsUpdatedAt,
            None,
            kind,
            retention,
            &mut params,
        );
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

/// The descriptive `models.metadata` blob: lineage and backend configuration.
/// Where the bytes live is never part of it.
pub(super) fn model_metadata(base_model_id: Option<&str>, config_json: Option<&str>) -> String {
    serde_json::json!({
        "base_model_id": base_model_id,
        "config_json": config_json,
    })
    .to_string()
}

fn parse_model_row(row: &Row<'_>) -> std::result::Result<ModelRecord, BackendError> {
    let catalog_pk: String = row.get("model_id")?;
    let name: String = row.get("name")?;
    let model_type: String = row.get("model_type")?;
    let task_raw: String = row.get("task")?;
    let task = ModelTask::parse(&task_raw).map_err(|e| BackendError::TypeConversion {
        column: "task".into(),
        detail: e.to_string(),
    })?;
    let backend: String = row.try_get("backend")?.unwrap_or_default();
    let version: i32 = row.try_get("version")?.unwrap_or(1);
    let status: String = row.get("status")?;
    let metadata: Option<String> = row.try_get("metadata")?;
    let created_at: String = row.get("created_at")?;

    let location = match (
        row.try_get::<String>("artifact_prefix")?,
        row.try_get::<String>("external_location")?,
    ) {
        (None, None) => None,
        (Some(prefix), None) => Some(ModelLocation::Artifact(
            ArtifactRef::parse(&prefix).map_err(|e| BackendError::TypeConversion {
                column: "artifact_prefix".into(),
                detail: e.to_string(),
            })?,
        )),
        (None, Some(directory)) => Some(ModelLocation::External(directory)),
        (Some(_), Some(_)) => {
            return Err(BackendError::TypeConversion {
                column: "artifact_prefix".into(),
                detail: format!(
                    "model row '{catalog_pk}' names both an artifact and an external location"
                ),
            })
        }
    };
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

    Ok(ModelRecord {
        model_id: name,
        catalog_pk,
        version,
        model_type,
        base_model_id,
        backend,
        task,
        location,
        config_json,
        status,
        created_at,
    })
}
