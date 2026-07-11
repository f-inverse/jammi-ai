use std::any::Any;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties,
};
use futures::stream;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::Result;
use jammi_db::index::exact::exact_vector_search;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::store::ResultStore;

/// ANN vector search over an embedding table.
///
/// Delegates to `ResultStore::resolve_search_mode()` — ANN via SidecarIndex
/// when available, brute-force via `exact_vector_search()` otherwise. When the
/// resolved index is quantized ([`jammi_db::config::StoragePrecision::needs_rescore`])
/// this is a two-stage retrieve→rescore: the quantized graph proposes `k *
/// oversample` candidates, then each candidate's *exact* `f32` vector is read
/// off the sidecar's mmap'd rescore companion ([`SidecarIndex::get_exact`])
/// and cosine similarity is recomputed against it,
/// then the top `k` of the re-ranked set are returned. An `F32` table's index
/// already holds exact vectors, so it stays the original single-stage path —
/// no oversampling, no companion read.
pub struct AnnSearchExec {
    table: ResultTableRecord,
    query_vector: Vec<f32>,
    k: usize,
    /// Per-request oversample override (`SearchRequest::oversample`). `None`
    /// defers to the table's own stamped default
    /// (`ResultTableRecord::oversample`), which itself falls back to the
    /// deployment's current
    /// [`jammi_db::config::AnnIndexConfig::effective_oversample`] only for a
    /// pre-migration-023 table with no stamped column.
    oversample_override: Option<usize>,
    result_store: Arc<ResultStore>,
    session_ctx: datafusion::prelude::SessionContext,
    properties: PlanProperties,
}

impl AnnSearchExec {
    pub fn new(
        table: ResultTableRecord,
        query_vector: Vec<f32>,
        k: usize,
        oversample_override: Option<usize>,
        result_store: Arc<ResultStore>,
        session_ctx: datafusion::prelude::SessionContext,
    ) -> Result<Self> {
        let schema = Self::output_schema();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Ok(Self {
            table,
            query_vector,
            k,
            oversample_override,
            result_store,
            session_ctx,
            properties,
        })
    }

    fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new("_source_id", DataType::Utf8, false),
            Field::new("similarity", DataType::Float32, false),
        ]))
    }
}

impl std::fmt::Debug for AnnSearchExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnSearchExec")
            .field("table", &self.table.table_name)
            .field("k", &self.k)
            .finish()
    }
}

impl DisplayAs for AnnSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "AnnSearchExec: table={}, k={}",
            self.table.table_name, self.k
        )
    }
}

impl ExecutionPlan for AnnSearchExec {
    fn name(&self) -> &str {
        "AnnSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![] // leaf node
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self) // no children to replace
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = self.schema();
        let schema_for_stream = Arc::clone(&schema);
        let result_store = Arc::clone(&self.result_store);
        let table = self.table.clone();
        let query = self.query_vector.clone();
        let k = self.k;
        let oversample_override = self.oversample_override;
        let ctx = self.session_ctx.clone();

        let result_stream = stream::once(async move {
            let search_results = match result_store
                .resolve_search_mode(&table)
                .await
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            {
                Some(index) if index.storage_precision().needs_rescore() => {
                    let oversample = resolve_oversample(
                        oversample_override,
                        table.oversample,
                        result_store.ann_config(),
                    );
                    retrieve_then_rescore(&index, &query, k, oversample)
                        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
                }
                Some(index) => index
                    .search(&query, k)
                    .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
                None => exact_vector_search(&ctx, &table.table_name, &query, k)
                    .await
                    .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
            };

            // Convert Vec<(row_id, cosine_distance)> to RecordBatch
            let row_ids: Vec<&str> = search_results.iter().map(|(id, _)| id.as_str()).collect();
            let similarities: Vec<f32> =
                search_results.iter().map(|(_, dist)| 1.0 - dist).collect();
            let source_ids: Vec<&str> = vec![table.source_id.as_str(); search_results.len()];

            RecordBatch::try_new(
                schema_for_stream.clone(),
                vec![
                    Arc::new(StringArray::from(row_ids)),
                    Arc::new(StringArray::from(source_ids)),
                    Arc::new(Float32Array::from(similarities)),
                ],
            )
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            result_stream,
        )))
    }
}

/// Retrieve→rescore over a quantized [`SidecarIndex`]: `search` an oversampled
/// `k * oversample` candidate set off the quantized graph, look up each
/// candidate's *exact* `f32` vector via [`SidecarIndex::get_exact`] (the mmap'd
/// rescore companion — never the quantized graph's own lossy `get`), recompute
/// cosine distance against the query, then re-sort and truncate to `k`.
///
/// A ties on distance breaks on `row_id` ascending — the same stable tie-break
/// the exact neighbor-graph driver uses — so the re-ranked order is
/// deterministic given identical inputs. A candidate whose exact vector is
/// missing (an id present in the graph but absent from the companion — a
/// corrupted or torn bundle) is a hard error, never a silent drop: a search
/// result set that quietly shrank below `k` would look like "fewer matches
/// exist" rather than "the index is broken".
fn retrieve_then_rescore(
    index: &SidecarIndex,
    query: &[f32],
    k: usize,
    oversample: usize,
) -> Result<Vec<(String, f32)>> {
    let candidate_k = k.saturating_mul(oversample).max(k);
    let candidates = index.search(query, candidate_k)?;

    let mut rescored: Vec<(String, f32)> = Vec::with_capacity(candidates.len());
    for (row_id, _quantized_distance) in candidates {
        let exact = index.get_exact(&row_id)?.ok_or_else(|| {
            jammi_db::error::JammiError::Other(format!(
                "rescore: candidate '{row_id}' has no exact vector in the rescore companion \
                 (corrupted or torn sidecar bundle)"
            ))
        })?;
        let distance = jammi_numerics::distance::cosine_distance(query, &exact);
        rescored.push((row_id, distance));
    }

    rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    rescored.truncate(k);
    Ok(rescored)
}

/// Resolve the oversample multiplier a rescore actually uses: a per-request
/// `SearchRequest::oversample` override wins when present, otherwise the
/// table's own stamped default (`ResultTableRecord::oversample`) — never
/// today's deployment config, mirroring how `storage_precision` is resolved
/// — falling back to the deployment's current
/// [`jammi_db::config::AnnIndexConfig::effective_oversample`] only for a
/// pre-migration-023 table with no stamped column. Either way clamped to at
/// least `1` (a `0` override or a misconfigured `0` default must not shrink
/// the candidate set below `k`).
fn resolve_oversample(
    request_override: Option<usize>,
    table_default: Option<usize>,
    ann: &jammi_db::config::AnnIndexConfig,
) -> usize {
    request_override
        .or(table_default)
        .unwrap_or_else(|| ann.effective_oversample())
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jammi_db::config::AnnIndexConfig;

    #[test]
    fn request_override_wins_over_table_default_and_config_default() {
        let ann = AnnIndexConfig {
            oversample: 4,
            ..AnnIndexConfig::default()
        };
        assert_eq!(resolve_oversample(Some(9), Some(6), &ann), 9);
        assert_eq!(resolve_oversample(None, Some(6), &ann), 6);
        assert_eq!(resolve_oversample(None, None, &ann), 4);
    }

    #[test]
    fn table_default_wins_over_deployment_config_default() {
        let ann = AnnIndexConfig {
            oversample: 4,
            ..AnnIndexConfig::default()
        };
        assert_eq!(
            resolve_oversample(None, Some(8), &ann),
            8,
            "the table's own stamped oversample must drive the rescore, not \
             today's deployment config default"
        );
    }

    #[test]
    fn a_zero_override_is_clamped_to_one_not_zero() {
        let ann = AnnIndexConfig {
            oversample: 4,
            ..AnnIndexConfig::default()
        };
        assert_eq!(
            resolve_oversample(Some(0), Some(6), &ann),
            1,
            "a 0 override must never shrink the candidate set below k"
        );
    }

    #[test]
    fn a_misconfigured_zero_default_is_also_clamped() {
        let ann = AnnIndexConfig {
            oversample: 0,
            ..AnnIndexConfig::default()
        };
        assert_eq!(resolve_oversample(None, None, &ann), 1);
    }
}
