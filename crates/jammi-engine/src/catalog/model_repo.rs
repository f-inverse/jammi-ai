use crate::error::Result;

use super::Catalog;

/// A row from the `models` table.
#[derive(Debug, Clone)]
pub struct ModelRecord {
    pub model_id: String,
    pub version: i32,
    pub model_type: String,
    pub base_model_id: Option<String>,
    pub backend: String,
    pub task: String,
    pub artifact_path: Option<String>,
    pub config_json: Option<String>,
    pub status: String,
    pub created_at: String,
}

/// Parameters for registering a model.
#[derive(Debug, Default)]
pub struct RegisterModelParams<'a> {
    pub model_id: &'a str,
    pub version: i32,
    pub model_type: &'a str,
    pub backend: &'a str,
    pub task: &'a str,
    pub base_model_id: Option<&'a str>,
    pub artifact_path: Option<&'a str>,
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

    /// Update model status (e.g., "registered" → "loaded").
    pub fn update_model_status(&self, model_id: &str, status: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE models SET status = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE name = ?2",
            rusqlite::params![status, model_id],
        )?;
        Ok(())
    }

    /// List all registered models.
    pub fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let conn = self.conn()?;
        let sql = format!("SELECT {SELECT_COLS} FROM models ORDER BY created_at");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| Ok(parse_model_row(row)))?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    fn query_one_model(
        stmt: &mut rusqlite::Statement<'_>,
        params: &[&dyn rusqlite::types::ToSql],
    ) -> Result<Option<ModelRecord>> {
        let mut rows = stmt.query_map(params, |row| Ok(parse_model_row(row)))?;
        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

/// Parse: model_id, name, model_type, task, backend, version, status, metadata, created_at
fn parse_model_row(row: &rusqlite::Row<'_>) -> ModelRecord {
    let _pk: String = row.get(0).unwrap_or_default();
    let name: String = row.get(1).unwrap_or_default();
    let model_type: String = row.get(2).unwrap_or_default();
    let task: String = row.get(3).unwrap_or_default();
    let backend: String = row.get(4).unwrap_or_default();
    let version: i32 = row.get(5).unwrap_or(1);
    let status: String = row.get(6).unwrap_or_default();
    let metadata: Option<String> = row.get(7).unwrap_or(None);
    let created_at: String = row.get(8).unwrap_or_default();

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

    ModelRecord {
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
    }
}
