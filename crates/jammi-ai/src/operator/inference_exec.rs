use std::any::Any;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties,
};

use crate::inference::observer::InferenceObserver;
use crate::inference::runner::InferenceRunner;
use crate::inference::schema::build_output_schema;
use crate::model::cache::ModelCache;
use crate::model::{BackendType, ModelTask};

/// InferenceExec — the core intelligence operator.
/// Reads input RecordBatches, runs model inference, and outputs
/// RecordBatches with common prefix + task-specific columns.
pub struct InferenceExec {
    input: Arc<dyn ExecutionPlan>,
    model_id: String,
    task: ModelTask,
    content_columns: Vec<String>,
    key_column: String,
    source_id: String,
    backend: Option<BackendType>,
    batch_size: usize,
    model_cache: Arc<ModelCache>,
    observer: Option<Arc<dyn InferenceObserver>>,
    embedding_dim: Option<usize>,
    properties: PlanProperties,
}

impl std::fmt::Debug for InferenceExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceExec")
            .field("model_id", &self.model_id)
            .field("task", &self.task)
            .field("content_columns", &self.content_columns)
            .finish()
    }
}

impl InferenceExec {
    /// Create a new inference operator wrapping the given input plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        model_id: String,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        source_id: String,
        backend: Option<BackendType>,
        batch_size: usize,
        model_cache: Arc<ModelCache>,
        observer: Option<Arc<dyn InferenceObserver>>,
        embedding_dim: Option<usize>,
    ) -> jammi_engine::error::Result<Self> {
        let output_schema =
            build_output_schema(&task, &input.schema(), &key_column, embedding_dim)?;
        let properties = Self::compute_properties(output_schema);
        Ok(Self {
            input,
            model_id,
            task,
            content_columns,
            key_column,
            source_id,
            backend,
            batch_size,
            model_cache,
            observer,
            embedding_dim,
            properties,
        })
    }

    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        )
    }
}

impl DisplayAs for InferenceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "InferenceExec: model={}, task={:?}, columns={:?}",
            self.model_id, self.task, self.content_columns
        )
    }
}

impl ExecutionPlan for InferenceExec {
    fn name(&self) -> &str {
        "InferenceExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(
            Self::new(
                Arc::clone(&children[0]),
                self.model_id.clone(),
                self.task,
                self.content_columns.clone(),
                self.key_column.clone(),
                self.source_id.clone(),
                self.backend,
                self.batch_size,
                Arc::clone(&self.model_cache),
                self.observer.clone(),
                self.embedding_dim,
            )
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let output_schema = self.schema();

        // Bounded channel for backpressure (capacity = 2 batches)
        let mut builder = RecordBatchReceiverStreamBuilder::new(output_schema.clone(), 2);
        let tx = builder.tx();

        // Build the runner with everything it needs
        let runner = InferenceRunner::new(
            Arc::clone(&self.model_cache),
            self.model_id.clone(),
            self.task,
            self.content_columns.clone(),
            self.key_column.clone(),
            self.source_id.clone(),
            self.backend,
            self.batch_size,
            self.observer.clone(),
        );

        builder.spawn(async move { runner.run(input_stream, tx, output_schema).await });

        Ok(builder.build())
    }
}
