use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, Distribution,
    ExecutionPlan, ExecutionPlanProperties, Partitioning, PlanProperties,
};

use crate::inference::adapter::DistributionForm;
use crate::inference::observer::InferenceObserver;
use crate::inference::runner::InferenceRunner;
use crate::inference::schema::{build_output_schema, ORDINAL_COLUMN};
use crate::model::cache::ModelCache;
use crate::model::{BackendType, ModelSource, ModelTask};
use jammi_db::store::manifest::ComputeDeviceKind;

/// The default forward admission, scoped to ONE `InferenceExec` INSTANCE
/// (never a whole device — see `forward_permits`' field doc for what that
/// means for two concurrent instances sharing a GPU): `available_
/// parallelism()` concurrent forwards on the CPU (bounded parallel CPU
/// inference actually helps — see `InferenceRunner`'s CPU speedup
/// measurement), 1 on a CUDA or Metal device (a conservative default:
/// DEVICE-WIDE admission across every concurrent instance — e.g. multiple
/// small models, or a scheduler-approved batch split — is
/// `concurrency::GpuScheduler`'s seam, not this one).
fn default_forward_permits(device_kind: ComputeDeviceKind) -> usize {
    match device_kind {
        ComputeDeviceKind::Cpu => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ComputeDeviceKind::Cuda | ComputeDeviceKind::Metal => 1,
    }
}

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
    /// The device KIND this descriptor
    /// declares it must run on: a REQUIRED constructor argument, stamped at
    /// every call site from `session.compute_device().kind()` (or, for a
    /// submitter placing this plan onto another kind, that kind directly —
    /// there is no separate override setter). The wire carries exactly this
    /// value; a decoding codec never invents or rewrites it (`jammi-
    /// ballista`'s codec, `codec.rs`).
    device_kind: ComputeDeviceKind,
    properties: Arc<PlanProperties>,
    /// The forward admission shared by every partition of THIS
    /// `InferenceExec` instance (cloned into each partition's
    /// `InferenceRunner` at `execute()` time). This is per-INSTANCE, never
    /// per-device: two concurrent `InferenceExec` instances targeting the
    /// SAME GPU each get their own `forward_permits` and will each run one
    /// forward concurrently (up to two total on that device) — bounding a
    /// whole device's admission across every instance is the named,
    /// unbuilt seam (`concurrency::GpuScheduler`; `default_forward_
    /// permits`'s doc). Sized from `device_kind` at build time.
    forward_permits: Arc<tokio::sync::Semaphore>,
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
    device_kind: ComputeDeviceKind,
}

impl InferenceExecBuilder {
    /// `device_kind` is a REQUIRED constructor argument — a submitter placing this plan onto a
    /// device kind other than its own session's builds with that kind directly;
    /// there is no separate optional override setter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        source: ModelSource,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        source_id: String,
        model_cache: Arc<ModelCache>,
        device_kind: ComputeDeviceKind,
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
            device_kind,
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
    /// choice (`jammi-ballista`'s codec); `with_new_children` below threads
    /// it through too.
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

    pub fn build(self) -> jammi_db::error::Result<InferenceExec> {
        let output_schema = build_output_schema(
            &self.task,
            &self.input.schema(),
            &self.key_column,
            self.embedding_dim,
            self.regression_form.as_ref(),
            &self.passthrough,
        )?;
        let properties = InferenceExec::compute_properties(output_schema, &self.input);
        let forward_permits = Arc::new(tokio::sync::Semaphore::new(
            default_forward_permits(self.device_kind).max(1),
        ));
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
            forward_permits,
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

    /// The device kind this descriptor declares it must run on.
    pub fn device_kind(&self) -> ComputeDeviceKind {
        self.device_kind
    }

    /// `output_partitioning` propagates the CHILD's own partition count
    /// (never a hardcoded 1): with a declared `UnknownPartitioning(1)`, a
    /// multi-partition child (the UDTF/`annotate` scan) would have its extra
    /// partitions silently unreachable, since a call site calling
    /// `.execute(0, ..)` gives the optimizer no reason to ever coalesce
    /// first. The equivalence
    /// properties publish `[_ordinal ASC]` — true BY CONSTRUCTION regardless
    /// of whether an `OrdinalSplitExec` sits below (see
    /// `inference::schema::extract_or_generate_ordinals`'s doc: either arm
    /// hands this operator, and this operator alone emits, a strictly
    /// increasing per-partition `_ordinal` subsequence).
    fn compute_properties(schema: SchemaRef, input: &Arc<dyn ExecutionPlan>) -> PlanProperties {
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        if let Ok(ordinal) = col(ORDINAL_COLUMN, schema.as_ref()) {
            let sort_opts = arrow::compute::SortOptions {
                descending: false,
                nulls_first: false,
            };
            if let Some(ordering) = LexOrdering::new([PhysicalSortExpr::new(ordinal, sort_opts)]) {
                eq.add_ordering(ordering);
            }
        }
        PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
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

    /// `UnspecifiedDistribution` (never `SinglePartition`) — this node
    /// does not require its child to be coalesced; `OrdinalSplitExec`'s own
    /// `SinglePartition` requirement (or a multi-partition scan directly
    /// below, on the `n == 1` no-split shape) is what `EnforceDistribution`
    /// actually satisfies.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    /// `false`, not the DataFusion default (`true` for a node whose
    /// `required_input_distribution` is `Unspecified`). The default would let
    /// `EnforceDistribution` insert a round-robin `RepartitionExec` between
    /// `OrdinalSplitExec` and this node whenever it judges that repartitioning
    /// "benefits" — defeating the point of the split. Without this override
    /// (`vec![true]`), 60 of the 120-cell grid's cells fail
    /// (`tests/it/rangesplit.rs`'s
    /// `nothing_between_split_and_inference_across_the_grid`); the exact count shifts with which
    /// optimizer passes fire.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    /// Each partition forwards its input rows in arrival order and
    /// stamps `_ordinal` (or reads the split's own) without ever reordering
    /// them — the per-partition ordering this node's own `[_ordinal ASC]`
    /// equivalence claims is genuinely maintained, never merely declared.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
                self.device_kind,
            )
            .batch_size(self.batch_size)
            .backend(self.backend)
            .observer(self.observer.clone())
            .embedding_dim(self.embedding_dim)
            .regression_form(self.regression_form.clone())
            .passthrough(self.passthrough.clone())
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
        .with_passthrough(self.passthrough.clone())
        .with_forward_permits(Arc::clone(&self.forward_permits));

        builder.spawn(async move { runner.run(input_stream, tx, output_schema).await });

        Ok(builder.build())
    }
}

/// Wire `InferenceConfig::partitions` into one of this
/// crate's four `InferenceExec`-building call sites: `partitions <= 1`
/// coalesces `input` to a single partition when it is not already one
/// (`build_inference` always sees exactly ONE partition — see the
/// implementation below for why this coalesce is load-bearing, not
/// cosmetic) and builds `build_inference` on it with no split and no
/// merge. `partitions > 1` inserts
/// [`OrdinalSplitExec`](crate::operator::ordinal_split_exec::OrdinalSplitExec)
/// below `input` (which coalesces a multi-partition `input` itself, the
/// same way) and wraps the built `InferenceExec` in a
/// `SortPreservingMergeExec([_ordinal ASC])` (the merge key — see that
/// module's doc for why `_ordinal` alone, never `[_row_id, _ordinal]`).
///
/// `session::annotate_plan`, `session::infer_materialize` (`infer`'s actual
/// materializer), `pipeline::embedding::build_embedding_plan`, and
/// `pipeline::embedding_refresh::infer_delta` all call this rather than
/// each repeating the wrap/merge logic — the "four roots" invariant holds
/// because all four share this one function, and is checked live by
/// `tests/it/rangesplit.rs`'s `split_merge_source_oracle` (a `syn`-based scan of
/// every `InferenceExecBuilder::new` call site in this crate).
pub fn wrap_with_split_and_merge(
    input: Arc<dyn ExecutionPlan>,
    partitions: usize,
    build_inference: impl FnOnce(Arc<dyn ExecutionPlan>) -> jammi_db::error::Result<InferenceExec>,
) -> jammi_db::error::Result<Arc<dyn ExecutionPlan>> {
    if partitions <= 1 {
        // `InferenceExec::compute_properties` (below) propagates the
        // CHILD's own partition count rather than a hardcoded 1, so an
        // uncoalesced multi-partition `input` here would make it declare
        // more than one partition while every one of those partitions'
        // `InferenceRunner`s independently self-generates `_ordinal`
        // starting at 0 (no split providing a shared global sequence), and
        // a caller that always calls `.execute(0, ..)` (every one in this
        // crate) would see only a fraction of the rows. Coalesce
        // defensively, exactly as `OrdinalSplitExec::new` does at
        // `partitions > 1` for the identical reason.
        let input: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
            Arc::new(
                datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec::new(input),
            )
        } else {
            input
        };
        return Ok(Arc::new(build_inference(input)?));
    }
    let split = Arc::new(
        crate::operator::ordinal_split_exec::OrdinalSplitExec::new(input, partitions).map_err(
            |e| jammi_db::error::JammiError::Inference(format!("OrdinalSplitExec: {e}")),
        )?,
    );
    let inference: Arc<dyn ExecutionPlan> = Arc::new(build_inference(split)?);
    let schema = inference.schema();
    let sort_opts = arrow::compute::SortOptions {
        descending: false,
        nulls_first: false,
    };
    let ordinal_expr = col(ORDINAL_COLUMN, schema.as_ref())
        .map_err(|e| jammi_db::error::JammiError::Inference(format!("merge ordering: {e}")))?;
    let ordering = LexOrdering::new([PhysicalSortExpr::new(ordinal_expr, sort_opts)])
        .ok_or_else(|| jammi_db::error::JammiError::Inference("merge ordering: empty".into()))?;
    Ok(Arc::new(
        datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec::new(
            ordering, inference,
        ),
    ))
}
