use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties,
};

use crate::inference::adapter::DistributionForm;
use crate::inference::observer::InferenceObserver;
use crate::inference::runner::InferenceRunner;
use crate::inference::schema::build_output_schema;
use crate::model::cache::ModelCache;
use crate::model::{BackendType, ModelSource, ModelTask};
use jammi_db::store::manifest::ComputeDeviceKind;

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
    /// Served regression head's persisted distribution form, for schema
    /// construction. `None` for non-regression tasks.
    regression_form: Option<DistributionForm>,
    /// Input columns copied verbatim to the end of every output batch.
    passthrough: Vec<String>,
    /// The device KIND (contract `feat_500-wave4` §9 B3) this descriptor
    /// declares it must run on. `None` when unset at construction — every
    /// in-process pipeline call site today (unmodified: out of this file's
    /// grant) — in which case a decoding executor treats the descriptor as
    /// "run on whatever this executor's own session runs on" (`jammi-
    /// ballista`'s codec fills this from the SUBMITTING session's
    /// `compute_device().kind()` at encode time when `None`, which is the
    /// same effective default this field's doc names, applied at the wire
    /// boundary rather than at every construction call site).
    device_kind: Option<ComputeDeviceKind>,
    properties: Arc<PlanProperties>,
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
    regression_form: Option<DistributionForm>,
    passthrough: Vec<String>,
    device_kind: Option<ComputeDeviceKind>,
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
            regression_form: None,
            passthrough: Vec::new(),
            device_kind: None,
        }
    }

    /// Copy the named input columns verbatim to the end of every output
    /// batch (after the task columns), keeping their input fields. The
    /// embedding pipeline passes `["_content_hash"]`.
    pub fn passthrough(mut self, columns: Vec<String>) -> Self {
        self.passthrough = columns;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Explicit backend override (`None` defers to the model cache's own
    /// resolution). Needed to round-trip a decoded node's exact backend
    /// choice (`jammi-ballista`'s codec); `with_new_children` below now
    /// threads it through too (a pre-existing gap this fixes as a side
    /// effect of adding the setter).
    pub fn backend(mut self, backend: Option<BackendType>) -> Self {
        self.backend = backend;
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

    pub fn regression_form(mut self, form: Option<DistributionForm>) -> Self {
        self.regression_form = form;
        self
    }

    /// Explicit device-kind override — a submitter placing this plan onto a
    /// different device kind than its own sets this. `None` (the default)
    /// means "run on whatever executes this plan" (see the field's doc on
    /// [`InferenceExec`]).
    pub fn device_kind(mut self, kind: Option<ComputeDeviceKind>) -> Self {
        self.device_kind = kind;
        self
    }

    pub fn build(self) -> jammi_db::error::Result<InferenceExec> {
        let output_schema = build_output_schema(
            &self.task,
            &self.input.schema(),
            &self.key_column,
            self.embedding_dim,
            self.regression_form.as_ref(),
            &self.passthrough,
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
            regression_form: self.regression_form,
            passthrough: self.passthrough,
            device_kind: self.device_kind,
            properties: Arc::new(properties),
        })
    }
}

impl InferenceExec {
    /// The model source this node runs inference against.
    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    /// The inference task this node performs.
    pub fn task(&self) -> ModelTask {
        self.task
    }

    /// The input columns whose content this node reads.
    pub fn content_columns(&self) -> &[String] {
        &self.content_columns
    }

    /// The row-identity column threaded through to the output.
    pub fn key_column(&self) -> &str {
        &self.key_column
    }

    /// The catalog source id this node's output is attributed to.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// The explicit backend override, if any (`None` defers to the model
    /// cache's own resolution).
    pub fn backend(&self) -> Option<BackendType> {
        self.backend
    }

    /// The inference batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The embedding output width, for tasks that produce one.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.embedding_dim
    }

    /// The served regression head's persisted distribution form, if any.
    pub fn regression_form(&self) -> Option<&DistributionForm> {
        self.regression_form.as_ref()
    }

    /// Input columns copied verbatim to the end of every output batch.
    pub fn passthrough(&self) -> &[String] {
        &self.passthrough
    }

    /// The child plan this node reads from.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// The device-kind override this descriptor declares, if any (contract
    /// `feat_500-wave4` §9 B3). See the field's doc for the `None` case.
    pub fn device_kind(&self) -> Option<ComputeDeviceKind> {
        self.device_kind
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
            self.source, self.task, self.content_columns
        )
    }
}

impl ExecutionPlan for InferenceExec {
    fn name(&self) -> &str {
        "InferenceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
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
            .backend(self.backend)
            .observer(self.observer.clone())
            .embedding_dim(self.embedding_dim)
            .regression_form(self.regression_form.clone())
            .passthrough(self.passthrough.clone())
            .device_kind(self.device_kind)
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
        )
        .with_passthrough(self.passthrough.clone());

        builder.spawn(async move { runner.run(input_stream, tx, output_schema).await });

        Ok(builder.build())
    }
}
