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
use jammi_db::index::{Admission, SearchMethod, ValidatedQuery};
use jammi_db::session::QueryContext;
use jammi_db::store::ResultStore;

/// Vector search over an embedding table, by the request's [`SearchMethod`],
/// among the rows its [`Admission`] admits: the `k` nearest admitted rows,
/// through [`ResultStore::search_vectors`] — the table's placed segment set
/// when it has one (exact-rescored to a cross-segment comparable top-`k`),
/// an exact scan otherwise or when the method asks for one.
pub struct VectorSearchExec {
    table: ResultTableRecord,
    query_vector: ValidatedQuery,
    k: usize,
    /// Exact, or approximate with an optional per-request oversample
    /// override. An override of `None` defers to the table's own stamped
    /// default (`ResultTableRecord::oversample`), which itself falls back to
    /// the deployment's current
    /// [`jammi_db::config::AnnIndexConfig::effective_oversample`] only for a
    /// pre-migration-023 table with no stamped column.
    method: SearchMethod,
    /// The rows the search may return.
    admission: Admission,
    result_store: Arc<ResultStore>,
    session_ctx: QueryContext,
    properties: Arc<PlanProperties>,
}

impl VectorSearchExec {
    pub fn new(
        table: ResultTableRecord,
        query_vector: ValidatedQuery,
        k: usize,
        method: SearchMethod,
        admission: Admission,
        result_store: Arc<ResultStore>,
        session_ctx: QueryContext,
    ) -> Result<Self> {
        let schema = Self::output_schema();
        // `UnknownPartitioning(1)`, unconditionally: this node has no input
        // plan at all (it drives the ANN sidecar directly through
        // `result_store`/`session_ctx`, not a child `ExecutionPlan`), so there
        // is nothing to partition.
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
            method,
            admission,
            result_store,
            session_ctx,
            properties: Arc::new(properties),
        })
    }

    /// The catalog record for the table this node searches.
    pub fn table(&self) -> &ResultTableRecord {
        &self.table
    }

    /// The validated query vector.
    pub fn query_vector(&self) -> &ValidatedQuery {
        &self.query_vector
    }

    /// The number of results requested.
    pub fn k(&self) -> usize {
        self.k
    }

    /// How this node ranks the table's vectors.
    pub fn method(&self) -> SearchMethod {
        self.method
    }

    /// The rows this node may return.
    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    fn output_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new("_source_id", DataType::Utf8, false),
            Field::new("similarity", DataType::Float32, false),
        ]))
    }
}

impl std::fmt::Debug for VectorSearchExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorSearchExec")
            .field("table", &self.table.table_name)
            .field("k", &self.k)
            .finish()
    }
}

impl DisplayAs for VectorSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "VectorSearchExec: table={}, k={}",
            self.table.table_name, self.k
        )
    }
}

impl ExecutionPlan for VectorSearchExec {
    fn name(&self) -> &str {
        "VectorSearchExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
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
        let method = self.method;
        let ctx = self.session_ctx.clone();

        let admission = self.admission.clone();
        let result_stream = stream::once(async move {
            let search_results = result_store
                .search_vectors(&ctx, &table, &query, k, method, &admission)
                .await
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

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
