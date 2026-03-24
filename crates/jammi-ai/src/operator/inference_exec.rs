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
use crate::model::{BackendType, ModelSource, ModelTask};

/// InferenceExec — the core intelligence operator.
/// Reads input RecordBatches, runs model inference, and outputs
/// RecordBatches with common prefix + task-specific columns.
pub struct InferenceExec {
    input: Arc<dyn ExecutionPlan>,
    source: ModelSource,
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
            .field("source", &self.source)
            .field("task", &self.task)
            .field("content_columns", &self.content_columns)
            .finish()
    }
}

/// Builder for constructing an `InferenceExec` operator.
pub struct InferenceExecBuilder {
    input: Arc<dyn ExecutionPlan>,
    source: ModelSource,
    task: ModelTask,
    content_columns: Vec<String>,
    key_column: String,
    source_id: String,
    model_cache: Arc<ModelCache>,
    backend: Option<BackendType>,
    batch_size: usize,
    observer: Option<Arc<dyn InferenceObserver>>,
    embedding_dim: Option<usize>,
}

impl InferenceExecBuilder {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        source: ModelSource,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        source_id: String,
        model_cache: Arc<ModelCache>,
    ) -> Self {
        Self {
            input,
            source,
            task,
            content_columns,
            key_column,
            source_id,
            model_cache,
            backend: None,
            batch_size: 32,
            observer: None,
            embedding_dim: None,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn observer(mut self, observer: Option<Arc<dyn InferenceObserver>>) -> Self {
        self.observer = observer;
        self
    }

    pub fn embedding_dim(mut self, dim: Option<usize>) -> Self {
        self.embedding_dim = dim;
        self
    }

    pub fn build(self) -> jammi_engine::error::Result<InferenceExec> {
        let output_schema = build_output_schema(
            &self.task,
            &self.input.schema(),
            &self.key_column,
            self.embedding_dim,
        )?;
        let properties = InferenceExec::compute_properties(output_schema);
        Ok(InferenceExec {
            input: self.input,
            source: self.source,
            task: self.task,
            content_columns: self.content_columns,
            key_column: self.key_column,
            source_id: self.source_id,
            backend: self.backend,
            batch_size: self.batch_size,
            model_cache: self.model_cache,
            observer: self.observer,
            embedding_dim: self.embedding_dim,
            properties,
        })
    }
}

impl InferenceExec {
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
            self.source, self.task, self.content_columns
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
            InferenceExecBuilder::new(
                Arc::clone(&children[0]),
                self.source.clone(),
                self.task,
                self.content_columns.clone(),
                self.key_column.clone(),
                self.source_id.clone(),
                Arc::clone(&self.model_cache),
            )
            .batch_size(self.batch_size)
            .observer(self.observer.clone())
            .embedding_dim(self.embedding_dim)
            .build()
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
            self.source.clone(),
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
