use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
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

fn parse_row(row: &Row<'_>) -> std::result::Result<ResultTableRecord, BackendError> {
    Ok(ResultTableRecord {
        table_name: row.get("table_name")?,
        source_id: row.get("source_id")?,
        model_id: row.get("model_id")?,
        task: row.get("task")?,
        parquet_path: row.get("parquet_path")?,
        index_path: row.try_get("index_path")?,
        dimensions: row.try_get("dimensions")?,
        distance_metric: row.get("distance_metric")?,
        row_count: row.get::<i64>("row_count")? as usize,
        status: row.get("status")?,
        key_column: row.try_get("key_column")?,
        text_columns: row.try_get("text_columns")?,
        created_at: row.get("created_at")?,
        completed_at: row.try_get("completed_at")?,
    })
}

impl Catalog {
    /// Insert a new result table record with status = 'building'.
    pub async fn create_result_table(&self, p: CreateResultTableParams<'_>) -> Result<()> {
        let table_name = p.table_name.to_string();
        let source_id = p.source_id.to_string();
        let model_id = p.model_id.to_string();
        let task = p.task.to_string();
        let parquet_path = p.parquet_path.to_string();
        let index_path = p.index_path.map(str::to_string);
        let dimensions = p.dimensions;
        let key_column = p.key_column.map(str::to_string);
        let text_columns = p.text_columns.map(str::to_string);

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO result_tables (table_name, source_id, model_id, task, parquet_path, \
                         index_path, dimensions, key_column, text_columns) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::TextOwned(source_id),
                            SqlValue::TextOwned(model_id),
                            SqlValue::TextOwned(task),
                            SqlValue::TextOwned(parquet_path),
                            SqlValue::from(index_path),
                            SqlValue::from(dimensions.map(|d| d as i64)),
                            SqlValue::from(key_column),
                            SqlValue::from(text_columns),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Update a result table's status and row count. Sets `completed_at` when
    /// transitioning to a terminal state (Ready/Failed).
    pub async fn update_result_table_status(
        &self,
        name: &str,
        status: super::status::ResultTableStatus,
        rows: usize,
    ) -> Result<()> {
        let completed_at = if matches!(
            status,
            super::status::ResultTableStatus::Ready | super::status::ResultTableStatus::Failed
        ) {
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

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE result_tables SET status = $1, row_count = $2, completed_at = $3 \
                         WHERE table_name = $4",
                        &[
                            SqlValue::TextOwned(status_str),
                            SqlValue::Int(rows_i64),
                            SqlValue::from(completed_at),
                            SqlValue::TextOwned(name),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Fetch a single result table by name.
    pub async fn get_result_table(&self, name: &str) -> Result<Option<ResultTableRecord>> {
        let name = name.to_string();
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
                            "SELECT * FROM result_tables WHERE table_name = $1",
                            &[SqlValue::TextOwned(name)],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// List all result tables with a given status.
    pub async fn list_result_tables_by_status(
        &self,
        status: super::status::ResultTableStatus,
    ) -> Result<Vec<ResultTableRecord>> {
        let status_str = status.to_string();
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
                            "SELECT * FROM result_tables WHERE status = $1 ORDER BY created_at",
                            &[SqlValue::TextOwned(status_str)],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Find result tables matching source, optional task, optional model.
    pub async fn find_result_tables(
        &self,
        source_id: &str,
        task: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<ResultTableRecord>> {
        let mut sql = "SELECT * FROM result_tables WHERE source_id = $1".to_string();
        let mut params: Vec<SqlValue<'static>> = vec![SqlValue::TextOwned(source_id.to_string())];

        if let Some(t) = task {
            sql.push_str(&format!(" AND task = ${}", params.len() + 1));
            params.push(SqlValue::TextOwned(t.to_string()));
        }
        if let Some(m) = model_id {
            sql.push_str(&format!(" AND model_id = ${}", params.len() + 1));
            params.push(SqlValue::TextOwned(m.to_string()));
        }
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

    /// Delete all result tables for a source. Returns the deleted records
    /// so callers can clean up associated disk files.
    pub async fn delete_result_tables_for_source(
        &self,
        source_id: &str,
    ) -> Result<Vec<ResultTableRecord>> {
        let sid = source_id.to_string();
        // Use a single transaction so the listing and delete are atomic.
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let records = tx
                        .query(
                            "SELECT * FROM result_tables WHERE source_id = $1",
                            &[SqlValue::TextOwned(sid.clone())],
                            parse_row,
                        )
                        .await?;
                    tx.execute(
                        "DELETE FROM result_tables WHERE source_id = $1",
                        &[SqlValue::TextOwned(sid)],
                    )
                    .await?;
                    Ok(records)
                })
            })
            .await?)
    }

    /// Resolve which embedding table to use for a source.
    ///
    /// - Explicit name → return that table.
    /// - None → find ready embedding tables for source, return latest by `created_at`.
    ///   Zero → error. One or more → latest.
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

        let sid = source_id.to_string();
        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT * FROM result_tables \
                         WHERE source_id = $1 AND task IN ('text_embedding', 'image_embedding') \
                           AND status = 'ready' \
                         ORDER BY created_at DESC, rowid DESC LIMIT 1",
                            &[SqlValue::TextOwned(sid)],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| {
            JammiError::Catalog(format!("No ready embedding table for source '{source_id}'"))
        })
    }

    /// Persist a checkpoint (batch number) for a result table.
    pub async fn set_checkpoint(&self, name: &str, batch: usize) -> Result<()> {
        let name = name.to_string();
        let batch_i64 = batch as i64;
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE result_tables SET checkpoint = $1 WHERE table_name = $2",
                        &[SqlValue::Int(batch_i64), SqlValue::TextOwned(name)],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Retrieve the last checkpoint for a result table.
    pub async fn get_checkpoint(&self, name: &str) -> Result<Option<usize>> {
        let name = name.to_string();
        let found: Option<Option<i64>> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT checkpoint FROM result_tables WHERE table_name = $1",
                            &[SqlValue::TextOwned(name)],
                            |row| row.try_get::<i64>("checkpoint"),
                        )
                        .await
                    })
                },
            )
            .await?;
        Ok(found.flatten().map(|c| c as usize))
    }
}
