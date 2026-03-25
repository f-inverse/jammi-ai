use crate::error::Result;

use super::Catalog;

/// Materialized row from the `models` catalog table.
#[derive(Debug, Clone)]
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
    /// Register a model in the catalog.
    pub fn register_model(&self, params: RegisterModelParams<'_>) -> Result<()> {
        let conn = self.conn()?;
        let pk = format!("{}::{}", params.model_id, params.version);
        let metadata = serde_json::json!({
            "base_model_id": params.base_model_id,
            "artifact_path": params.artifact_path,
            "config_json": params.config_json,
        })
        .to_string();

        conn.execute(
            "INSERT INTO models (model_id, name, model_type, task, backend, version, status, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'registered', ?7)",
            rusqlite::params![
                pk,
                params.model_id,
                params.model_type,
                params.task,
                params.backend,
                params.version,
                metadata
            ],
        )?;
        Ok(())
    }

    /// Get the latest version of a model by name.
    pub fn get_model(&self, model_id: &str) -> Result<Option<ModelRecord>> {
        let conn = self.conn()?;
        let sql = format!(
            "SELECT {SELECT_COLS} FROM models WHERE name = ?1 ORDER BY version DESC LIMIT 1"
        );
        let mut stmt = conn.prepare(&sql)?;
        Self::query_one_model(&mut stmt, rusqlite::params![model_id])
    }

    /// Get a specific version of a model.
    pub fn get_model_version(&self, model_id: &str, version: i32) -> Result<Option<ModelRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM models WHERE name = ?1 AND version = ?2");
        let mut stmt = conn.prepare(&sql)?;
        Self::query_one_model(&mut stmt, rusqlite::params![model_id, version])
    }

    /// Update model status (e.g., Registered → Loaded).
    pub fn update_model_status(
        &self,
        model_id: &str,
        status: super::status::ModelStatus,
    ) -> Result<()> {
        let conn = self.conn()?;
        let status_str = status.to_string();
        conn.execute(
            "UPDATE models SET status = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE name = ?2",
            rusqlite::params![status_str, model_id],
        )?;
        Ok(())
    }

    /// List all registered models.
    pub fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM models ORDER BY created_at");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], parse_model_row)?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    fn query_one_model(
        stmt: &mut rusqlite::Statement<'_>,
        params: &[&dyn rusqlite::types::ToSql],
    ) -> Result<Option<ModelRecord>> {
        let mut rows = stmt.query_map(params, parse_model_row)?;
        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

/// Parse: model_id, name, model_type, task, backend, version, status, metadata, created_at
fn parse_model_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRecord> {
    let _pk: String = row.get(0)?;
    let name: String = row.get(1)?;
    let model_type: String = row.get(2)?;
    let task: String = row.get(3)?;
    let backend: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
    let version: i32 = row.get::<_, Option<i32>>(5)?.unwrap_or(1);
    let status: String = row.get::<_, Option<String>>(6)?.unwrap_or_default();
    let metadata: Option<String> = row.get(7)?;
    let created_at: String = row.get::<_, Option<String>>(8)?.unwrap_or_default();

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
