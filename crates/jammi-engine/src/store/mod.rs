pub mod reader;
pub mod schema;
pub mod writer;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::Array;
use datafusion::prelude::SessionContext;
use tracing::warn;

use crate::catalog::result_repo::{CreateResultTableParams, ResultTableRecord};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::error::{JammiError, Result};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;

/// Returned by `ResultStore::create_table()` — the generated paths and name
/// for a new result table, before any data has been written.
pub struct ResultTableInfo {
    pub table_name: String,
    pub parquet_path: PathBuf,
    pub index_path: Option<PathBuf>,
}

/// Coordinates Parquet storage, ANN indexes, DataFusion registration,
/// catalog metadata, and crash recovery for result tables.
pub struct ResultStore {
    jammi_db_dir: PathBuf,
    catalog: Arc<Catalog>,
}

/// Sanitize a model ID for use in file names: `/`, `:`, spaces → `-`.
fn sanitize_model_id(model_id: &str) -> String {
    model_id
        .chars()
        .map(|c| {
            if c == '/' || c == ':' || c == ' ' {
                '_'
            } else {
                c
            }
        })
        .take(64)
        .collect()
}

impl ResultStore {
    /// Create a new ResultStore rooted at `{artifact_dir}/jammi_db/`.
    pub fn new(artifact_dir: &Path, catalog: Arc<Catalog>) -> Result<Self> {
        let jammi_db_dir = artifact_dir.join("jammi_db");
        std::fs::create_dir_all(&jammi_db_dir)?;
        Ok(Self {
            jammi_db_dir,
            catalog,
        })
    }

    /// Generate paths and register a new result table in the catalog with
    /// status = 'building'.
    pub fn create_table(
        &self,
        source_id: &str,
        task: &str,
        model_id: &str,
        dimensions: Option<i32>,
        key_column: Option<&str>,
        text_columns: Option<&str>,
    ) -> Result<ResultTableInfo> {
        let sanitized = sanitize_model_id(model_id);
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%3f");
        let table_name = format!("{source_id}__{task}__{sanitized}__{timestamp}");

        let parquet_path = self.jammi_db_dir.join(format!("{table_name}.parquet"));
        let index_path = if task == "text_embedding" || task == "image_embedding" {
            Some(self.jammi_db_dir.join(&table_name))
        } else {
            None
        };

        self.catalog.create_result_table(CreateResultTableParams {
            table_name: &table_name,
            source_id,
            model_id,
            task,
            parquet_path: parquet_path.to_str().unwrap_or_default(),
            index_path: index_path.as_ref().and_then(|p| p.to_str()),
            dimensions,
            key_column,
            text_columns,
        })?;

        Ok(ResultTableInfo {
            table_name,
            parquet_path,
            index_path,
        })
    }

    /// Register an existing Parquet file as a DataFusion table.
    pub async fn register_table(
        &self,
        ctx: &SessionContext,
        name: &str,
        path: &Path,
    ) -> Result<()> {
        reader::register_parquet_table(ctx, name, path).await
    }

    /// Finalize a result table: register in DataFusion and update catalog
    /// status to 'ready'.
    pub async fn finalize(
        &self,
        ctx: &SessionContext,
        name: &str,
        path: &Path,
        rows: usize,
    ) -> Result<()> {
        self.register_table(ctx, name, path).await?;
        self.catalog
            .update_result_table_status(name, ResultTableStatus::Ready, rows)?;
        Ok(())
    }

    /// Recover tables stuck in 'building' status after a crash.
    pub async fn recover(&self) -> Result<()> {
        let building = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Building)?;
        for table in building {
            let parquet_exists = Path::new(&table.parquet_path).exists();
            let parquet_valid = parquet_exists && reader::is_valid_parquet(&table.parquet_path);

            if !parquet_exists {
                warn!(
                    table = table.table_name,
                    "Recovery: Parquet missing, marking failed"
                );
                self.catalog.update_result_table_status(
                    &table.table_name,
                    ResultTableStatus::Failed,
                    0,
                )?;
            } else if !parquet_valid {
                warn!(
                    table = table.table_name,
                    "Recovery: invalid Parquet, deleting and marking failed"
                );
                std::fs::remove_file(&table.parquet_path).ok();
                if let Some(ref idx) = table.index_path {
                    let base = Path::new(idx);
                    std::fs::remove_file(base.with_extension("usearch")).ok();
                    std::fs::remove_file(base.with_extension("rowmap")).ok();
                    std::fs::remove_file(base.with_extension("manifest.json")).ok();
                }
                self.catalog.update_result_table_status(
                    &table.table_name,
                    ResultTableStatus::Failed,
                    0,
                )?;
            } else {
                let row_count = reader::count_parquet_rows(&table.parquet_path)?;
                // Rebuild ANN index if this is an embedding table
                if table.task == "embedding" {
                    if let Some(ref idx_path) = table.index_path {
                        if let Err(e) = rebuild_index_from_parquet(
                            &table.parquet_path,
                            idx_path,
                            table.dimensions.unwrap_or(0) as usize,
                        ) {
                            warn!(
                                table = table.table_name,
                                error = %e,
                                "Recovery: failed to rebuild index, proceeding without"
                            );
                        }
                    }
                }
                self.catalog.update_result_table_status(
                    &table.table_name,
                    ResultTableStatus::Ready,
                    row_count,
                )?;
            }
        }
        Ok(())
    }

    /// Load all 'ready' result tables into DataFusion.
    pub async fn load_existing_tables(&self, ctx: &SessionContext) -> Result<()> {
        let ready = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)?;
        for table in ready {
            let path = Path::new(&table.parquet_path);
            if path.exists() {
                if let Err(e) = self.register_table(ctx, &table.table_name, path).await {
                    warn!(
                        table = table.table_name,
                        error = %e,
                        "Failed to register existing table"
                    );
                }
            }
        }
        Ok(())
    }

    /// Search an embedding table for the nearest neighbors of a query vector.
    /// Uses SidecarIndex (ANN) when available, falls back to exact brute-force search.
    pub async fn search_vectors(
        &self,
        ctx: &SessionContext,
        table: &ResultTableRecord,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let index = self.resolve_search_mode(table)?;
        match index {
            Some(idx) => idx.search(query, k),
            None => {
                crate::index::exact::exact_vector_search(ctx, &table.table_name, query, k).await
            }
        }
    }

    /// Resolve whether to use ANN (sidecar index) or exact search for a table.
    /// Returns `Some(SidecarIndex)` for ANN, `None` for exact fallback.
    pub fn resolve_search_mode(&self, table: &ResultTableRecord) -> Result<Option<SidecarIndex>> {
        let Some(ref idx_path) = table.index_path else {
            return Ok(None);
        };
        match SidecarIndex::load(Path::new(idx_path)) {
            Ok(index) => Ok(Some(index)),
            Err(e) => {
                warn!(
                    table = table.table_name,
                    error = %e,
                    "Sidecar index unavailable, falling back to exact search"
                );
                Ok(None)
            }
        }
    }
}

/// Rebuild a sidecar ANN index from a Parquet embedding file.
fn rebuild_index_from_parquet(
    parquet_path: &str,
    _index_base_path: &str,
    dimensions: usize,
) -> Result<()> {
    if dimensions == 0 {
        return Ok(());
    }

    let file = std::fs::File::open(parquet_path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| JammiError::Other(format!("Parquet read for index rebuild: {e}")))?;
    let reader = builder
        .build()
        .map_err(|e| JammiError::Other(format!("Parquet reader build for index rebuild: {e}")))?;

    let mut index = SidecarIndex::new(dimensions)?;
    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| JammiError::Other(format!("Parquet batch read: {e}")))?;

        let row_ids = batch
            .column_by_name("_row_id")
            .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>());
        let vectors = batch.column_by_name("vector").and_then(|c| {
            c.as_any()
                .downcast_ref::<arrow::array::FixedSizeListArray>()
        });

        if let (Some(ids), Some(vecs)) = (row_ids, vectors) {
            for i in 0..ids.len() {
                let row_id = ids.value(i);
                let v = vecs.value(i);
                let float_arr = v
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .ok_or_else(|| JammiError::Other("Vector not Float32".into()))?;
                let vec: Vec<f32> = (0..float_arr.len()).map(|j| float_arr.value(j)).collect();
                index.add(row_id, &vec)?;
            }
        }
    }

    if index.len() > 0 {
        index.build()?;
        index.save(Path::new(_index_base_path))?;
    }
    Ok(())
}
