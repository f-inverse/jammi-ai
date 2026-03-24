use rusqlite::params;

use crate::catalog::Catalog;
use crate::error::{JammiError, Result};

/// Parameters for creating a new result table entry.
/// `status` defaults to `'building'` via SQL DEFAULT — not passed here.
#[derive(Debug)]
pub struct CreateResultTableParams<'a> {
    pub table_name: &'a str,
    pub source_id: &'a str,
    pub model_id: &'a str,
    pub task: &'a str,
    pub parquet_path: &'a str,
    pub index_path: Option<&'a str>,
    pub dimensions: Option<i32>,
    pub key_column: Option<&'a str>,
    pub text_columns: Option<&'a str>,
}

/// A row from the `result_tables` catalog table.
#[derive(Debug, Clone)]
pub struct ResultTableRecord {
    pub table_name: String,
    pub source_id: String,
    pub model_id: String,
    pub task: String,
    pub parquet_path: String,
    pub index_path: Option<String>,
    pub dimensions: Option<i32>,
    pub distance_metric: String,
    pub row_count: usize,
    pub status: String,
    pub key_column: Option<String>,
    pub text_columns: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

fn parse_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ResultTableRecord> {
    Ok(ResultTableRecord {
        table_name: row.get("table_name")?,
        source_id: row.get("source_id")?,
        model_id: row.get("model_id")?,
        task: row.get("task")?,
        parquet_path: row.get("parquet_path")?,
        index_path: row.get("index_path")?,
        dimensions: row.get("dimensions")?,
        distance_metric: row.get("distance_metric")?,
        row_count: row.get::<_, i64>("row_count")? as usize,
        status: row.get("status")?,
        key_column: row.get("key_column")?,
        text_columns: row.get("text_columns")?,
        created_at: row.get("created_at")?,
        completed_at: row.get("completed_at")?,
    })
}

impl Catalog {
    /// Insert a new result table record with status = 'building'.
    pub fn create_result_table(&self, p: CreateResultTableParams<'_>) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO result_tables (table_name, source_id, model_id, task, parquet_path,
             index_path, dimensions, key_column, text_columns)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                p.table_name,
                p.source_id,
                p.model_id,
                p.task,
                p.parquet_path,
                p.index_path,
                p.dimensions,
                p.key_column,
                p.text_columns,
            ],
        )?;
        Ok(())
    }

    /// Update a result table's status and row count. Sets `completed_at` when
    /// transitioning to a terminal state (ready/failed).
    pub fn update_result_table_status(&self, name: &str, status: &str, rows: usize) -> Result<()> {
        let conn = self.conn()?;
        let completed_at = if status == "ready" || status == "failed" {
            Some(
                chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            )
        } else {
            None
        };
        conn.execute(
            "UPDATE result_tables SET status = ?1, row_count = ?2, completed_at = ?3
             WHERE table_name = ?4",
            params![status, rows as i64, completed_at, name],
        )?;
        Ok(())
    }

    /// Fetch a single result table by name.
    pub fn get_result_table(&self, name: &str) -> Result<Option<ResultTableRecord>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT * FROM result_tables WHERE table_name = ?1")?;
        let mut rows = stmt.query_map(params![name], parse_row)?;
        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List all result tables with a given status.
    pub fn list_result_tables_by_status(&self, status: &str) -> Result<Vec<ResultTableRecord>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT * FROM result_tables WHERE status = ?1 ORDER BY created_at")?;
        let rows = stmt.query_map(params![status], parse_row)?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    /// Find result tables matching source, optional task, optional model.
    pub fn find_result_tables(
        &self,
        source_id: &str,
        task: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<ResultTableRecord>> {
        let conn = self.conn()?;
        let mut sql = "SELECT * FROM result_tables WHERE source_id = ?1".to_string();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(source_id.to_string())];

        if let Some(t) = task {
            sql.push_str(&format!(" AND task = ?{}", param_values.len() + 1));
            param_values.push(Box::new(t.to_string()));
        }
        if let Some(m) = model_id {
            sql.push_str(&format!(" AND model_id = ?{}", param_values.len() + 1));
            param_values.push(Box::new(m.to_string()));
        }
        sql.push_str(" ORDER BY created_at");

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_refs.as_slice(), parse_row)?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    /// Resolve which embedding table to use for a source.
    ///
    /// - Explicit name → return that table.
    /// - None → find ready embedding tables for source, return latest by `created_at`.
    ///   Zero → error. One or more → latest.
    pub fn resolve_embedding_table(
        &self,
        source_id: &str,
        table_name: Option<&str>,
    ) -> Result<ResultTableRecord> {
        if let Some(name) = table_name {
            return self
                .get_result_table(name)?
                .ok_or_else(|| JammiError::Catalog(format!("Result table '{name}' not found")));
        }

        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM result_tables
             WHERE source_id = ?1 AND task = 'embedding' AND status = 'ready'
             ORDER BY created_at DESC, rowid DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![source_id], parse_row)?;
        match rows.next() {
            Some(Ok(record)) => Ok(record),
            Some(Err(e)) => Err(e.into()),
            None => Err(JammiError::Catalog(format!(
                "No ready embedding table for source '{source_id}'"
            ))),
        }
    }

    /// Persist a checkpoint (batch number) for a result table.
    pub fn set_checkpoint(&self, name: &str, batch: usize) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE result_tables SET checkpoint = ?1 WHERE table_name = ?2",
            params![batch as i64, name],
        )?;
        Ok(())
    }

    /// Retrieve the last checkpoint for a result table.
    pub fn get_checkpoint(&self, name: &str) -> Result<Option<usize>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT checkpoint FROM result_tables WHERE table_name = ?1")?;
        let checkpoint: Option<i64> = stmt.query_row(params![name], |row| row.get(0))?;
        Ok(checkpoint.map(|c| c as usize))
    }
}
