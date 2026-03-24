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

use jammi_engine::catalog::result_repo::ResultTableRecord;
use jammi_engine::error::Result;
use jammi_engine::index::exact::exact_vector_search;
use jammi_engine::index::VectorIndex;
use jammi_engine::store::ResultStore;

/// ANN vector search over an embedding table.
/// Delegates to `ResultStore::resolve_search_mode()` — ANN via SidecarIndex
/// when available, brute-force via `exact_vector_search()` otherwise.
pub struct AnnSearchExec {
    table: ResultTableRecord,
    query_vector: Vec<f32>,
    k: usize,
    result_store: Arc<ResultStore>,
    session_ctx: datafusion::prelude::SessionContext,
    properties: PlanProperties,
}

impl AnnSearchExec {
    pub fn new(
        table: ResultTableRecord,
        query_vector: Vec<f32>,
        k: usize,
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
        let ctx = self.session_ctx.clone();

        let result_stream = stream::once(async move {
            let search_results = match result_store
                .resolve_search_mode(&table)
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
            {
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
