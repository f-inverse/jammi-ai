use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
use crate::error::Result;

use super::Catalog;

/// Materialized row from the `models` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelRecord {
    /// Unique model name (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).
    pub model_id: String,
    /// Monotonically increasing version number for this model name.
    pub version: i32,
    /// Model category (e.g., `"embedding"`, `"llm"`, `"lora"`).
    pub model_type: String,
    /// Parent model this was derived from (fine-tuned or adapted).
    pub base_model_id: Option<String>,
    /// Inference backend (e.g., `"candle"`, `"vllm"`, `"http"`).
    pub backend: String,
    /// Task this model performs (e.g., `"text-generation"`, `"embedding"`).
    pub task: String,
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
#[derive(Debug, Default)]
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
    pub task: &'a str,
    /// Optional parent model ID (for fine-tuned variants).
    pub base_model_id: Option<&'a str>,
    /// Optional filesystem path to model weights.
    pub artifact_path: Option<&'a str>,
    /// Optional JSON blob with backend-specific settings.
    pub config_json: Option<&'a str>,
}

const SELECT_COLS: &str =
    "model_id, name, model_type, task, backend, version, status, metadata, created_at";

impl Catalog {
    /// Register or refresh a model in the catalog. The session's bound
    /// tenant is written to `tenant_id` and asserted before INSERT.
    pub async fn register_model(&self, params: RegisterModelParams<'_>) -> Result<()> {
        let pk = format!("{}::{}", params.model_id, params.version);
        let metadata = serde_json::json!({
            "base_model_id": params.base_model_id,
            "artifact_path": params.artifact_path,
            "config_json": params.config_json,
        })
        .to_string();
        let model_id = params.model_id.to_string();
        let model_type = params.model_type.to_string();
        let task = params.task.to_string();
        let backend = params.backend.to_string();
        let version = params.version as i64;
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "models")?;
                    tx.execute(
                        "INSERT INTO models (model_id, name, model_type, task, backend, version, status, metadata, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, $6, 'registered', $7, $8) \
                         ON CONFLICT(model_id) DO UPDATE SET \
                             metadata = excluded.metadata, \
                             backend = excluded.backend, \
                             task = excluded.task, \
                             model_type = excluded.model_type, \
                             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                        &[
                            SqlValue::TextOwned(pk),
                            SqlValue::TextOwned(model_id),
                            SqlValue::TextOwned(model_type),
                            SqlValue::TextOwned(task),
                            SqlValue::TextOwned(backend),
                            SqlValue::Int(version),
                            SqlValue::TextOwned(metadata),
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
    pub async fn get_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE name = $1 AND (tenant_id = $2 OR tenant_id IS NULL) \
             ORDER BY version DESC LIMIT 1"
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
               AND (tenant_id = $3 OR tenant_id IS NULL)"
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

    /// Update model status. Scoped to the session's tenant.
    pub async fn update_model_status(
        &self,
        model_id: &str,
        status: super::status::ModelStatus,
    ) -> Result<()> {
        let status_str = status.to_string();
        let mid = model_id.to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE models SET status = $1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                         WHERE name = $2 AND (tenant_id = $3 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(status_str),
                            SqlValue::TextOwned(mid),
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

    /// List models visible to the session's tenant.
    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models \
             WHERE tenant_id = $1 OR tenant_id IS NULL \
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

/// Parse: model_id, name, model_type, task, backend, version, status, metadata, created_at
fn parse_model_row(row: &Row<'_>) -> std::result::Result<ModelRecord, BackendError> {
    let _pk: String = row.get("model_id")?;
    let name: String = row.get("name")?;
    let model_type: String = row.get("model_type")?;
    let task: String = row.get("task")?;
    let backend: String = row.try_get("backend")?.unwrap_or_default();
    let version: i64 = row.try_get("version")?.unwrap_or(1);
    let status: String = row.try_get("status")?.unwrap_or_default();
    let metadata: Option<String> = row.try_get("metadata")?;
    let created_at: String = row.try_get("created_at")?.unwrap_or_default();

    let (base_model_id, artifact_path, config_json) = metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .map(|v| {
            (
                v["base_model_id"].as_str().map(String::from),
                v["artifact_path"].as_str().map(String::from),
                v["config_json"].as_str().map(String::from),
            )
        })
        .unwrap_or((None, None, None));

    Ok(ModelRecord {
        model_id: name,
        version: version as i32,
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
