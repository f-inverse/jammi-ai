use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::tenant::TenantId;

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
     created_at";

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
    /// job serves is written solely by the lease-guarded
    /// [`Self::finalize_training_job`] CAS, never by a worker's pre-finalize
    /// (or a zombie's late) `register_model`.
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

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    tx.execute(
                        "INSERT INTO models (model_id, name, model_type, task, backend, version, status, metadata, artifact_path, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, $6, 'registered', $7, $8, $9) \
                         ON CONFLICT(model_id) DO UPDATE SET \
                             metadata = excluded.metadata, \
                             backend = excluded.backend, \
                             task = excluded.task, \
                             model_type = excluded.model_type, \
                             artifact_path = COALESCE(excluded.artifact_path, models.artifact_path), \
                             updated_at = CAST(CURRENT_TIMESTAMP AS TEXT)",
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
    /// returns RETIRED models too — a job or eval that references a since-retired
    /// model must still resolve it, so excluding retired here would break
    /// historical provenance. The active-listing sense (hiding retired) lives in
    /// [`Self::list_models`]; the serve/load path refuses a retired model itself.
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

    /// Soft-retire a model. Resolves the catalog PK off the read path
    /// ([`Self::get_model`] for the latest version, or
    /// [`Self::get_model_version`] when `version` is given), then flips that
    /// row's `status` to [`ModelStatus::Retired`](super::status::ModelStatus).
    ///
    /// A tenant may retire ONLY a row it owns: the UPDATE is filtered on the
    /// resolved PK AND `tenant_id = $current` strictly — not the read path's
    /// `OR tenant_id IS NULL`. So a tenant session asking to retire a GLOBAL
    /// (`tenant_id IS NULL`) model, or a model that does not exist for it, gets
    /// [`JammiError::Model`] NotFound: retiring a model outside the caller's
    /// own scope is forbidden, not silently applied to a shared row.
    ///
    /// Idempotent: retiring an already-retired model is a success no-op (the
    /// row stays `'retired'`).
    pub async fn retire_model(&self, model_id: &str, version: Option<i32>) -> Result<()> {
        let record = match version {
            Some(v) => self.get_model_version(model_id, v).await?,
            None => self.get_model(model_id).await?,
        };
        let record = record.ok_or_else(|| JammiError::Model {
            model_id: model_id.to_string(),
            message: "no such model to retire for this tenant".into(),
        })?;
        let pk = record.catalog_pk;
        let tenant = self.current_tenant();
        let retired = super::status::ModelStatus::Retired.to_string();
        let affected = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    // Strict tenant predicate: a session retires only a row
                    // whose owner equals its OWN tenant — never the read path's
                    // `OR tenant_id IS NULL` leak. The `IS NULL AND $3 IS NULL`
                    // arm matches `None == None` (a NULL `=` comparison is never
                    // true in SQL), so an unscoped session retires its own
                    // GLOBAL row while a tenant session (with a non-NULL `$3`)
                    // matches only its own rows — a GLOBAL row's NULL tenant
                    // fails both arms, so a tenant cannot retire a shared model.
                    // A non-matching PK affects zero rows and surfaces NotFound
                    // below.
                    tx.execute(
                        "UPDATE models \
                         SET status = $1, updated_at = CAST(CURRENT_TIMESTAMP AS TEXT) \
                         WHERE model_id = $2 \
                           AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL))",
                        &[
                            SqlValue::TextOwned(retired),
                            SqlValue::TextOwned(pk),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await
                })
            })
            .await?;
        if affected == 0 {
            return Err(JammiError::Model {
                model_id: model_id.to_string(),
                message: "no such model to retire for this tenant".into(),
            });
        }
        Ok(())
    }

    /// List the *active* models visible to the session's tenant. Retired models
    /// are hidden — this is the active-listing sense, the peer of
    /// `list_sources`. A reference resolver that must still see a retired model
    /// (provenance, a base-model FK) uses [`Self::get_model`], which returns
    /// retired rows.
    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE (tenant_id = $1 OR tenant_id IS NULL) AND status != 'retired' \
             ORDER BY created_at"
        );
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
                            &sql,
                            &[SqlValue::from(tenant.map(|t| t.to_string()))],
                            parse_model_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}

/// Parse: model_id, name, model_type, task, backend, version, status, metadata,
/// artifact_path, created_at
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
    let status: String = row.try_get("status")?.unwrap_or_default();
    let metadata: Option<String> = row.try_get("metadata")?;
    let created_at: String = row.try_get("created_at")?.unwrap_or_default();

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
    })
}
