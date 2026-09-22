//! `InferenceExec` and the one plan every model forward runs in.
//!
//! [`plan_inference`] builds the same shape at every scale — one partition in
//! one process, `N` partitions in one process, `N` tasks on a cluster:
//!
//! ```text
//! SortPreservingMergeExec [_ordinal ASC]        restores the row sequence
//!   InferenceExec                                N partitions
//!     RepartitionExec Hash([_chunk], N)          the fan-out
//!       NumberedInputExec                        costs, orders, numbers and chunks the rows
//!         CoalescePartitionsExec
//!           input
//! ```
//!
//! The exchange, the merge and the coalesce are stock DataFusion operators,
//! so an optimizer re-derives them from this module's declared requirements
//! and a distributed planner cuts its stages at them. Each is the identity
//! over one partition and is left out there: at `N == 1` the plan is
//! `InferenceExec(NumberedInputExec(input))`.
//!
//! The fan-out `N` has one source: [`InferenceSpec::partitions`], which the
//! node carries and declares. DataFusion has no vocabulary for a declared
//! partition COUNT — `EnforceDistribution` strips the exchange and re-adds
//! it from the declared hash requirement at the session's
//! `target_partitions`, or not at all when that is 1 — so a session that
//! plans through the optimizer registers [`InferenceFanOut`], which puts
//! every `InferenceExec` back over an exchange of the node's own width.
//!
//! The rows a model forwards together are decided once, in the numbered
//! input, and carried as `_chunk` ([`crate::inference::chunk`]). The
//! exchange hashes on that chunk id, so a chunk is never divided between
//! partitions, and every partition count — and every re-batching an exchange
//! or a shuffle performs on the way — forwards identical chunks and writes
//! identical bytes.

use std::fmt::{self, Formatter};
use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, OrderingRequirements, PhysicalExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, Distribution,
    ExecutionPlan, ExecutionPlanProperties, Partitioning, PlanProperties,
};

use crate::inference::adapter::DistributionForm;
use crate::inference::chunk::chunk_expr;
use crate::inference::observer::InferenceObserver;
use crate::inference::runner::InferenceRunner;
use crate::inference::schema::build_output_schema;
use crate::model::cache::ModelCache;
use crate::model::{BackendType, ModelSource, ModelTask};
use crate::operator::numbered_input_exec::{ordinal_ordering, NumberedInputExec, RowOrder};
use crate::operator::single_partition;
use jammi_db::error::Result;
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_numerics::ChunkBudget;

/// What an [`InferenceExec`] computes: plain data, and everything about the
/// node that crosses a process boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSpec {
    /// The model to run.
    pub source: ModelSource,
    /// The task it performs.
    pub task: ModelTask,
    /// The input columns whose content the model reads.
    pub content_columns: Vec<String>,
    /// The row-identity column, carried to the output as `_row_id`.
    pub key_column: String,
    /// The catalog source id the output is attributed to.
    pub source_id: String,
    /// An explicit backend; `None` defers to the model cache's resolution.
    pub backend: Option<BackendType>,
    /// What bounds one forward chunk: its rows and its padded tokens.
    pub chunk: ChunkBudget,
    /// The embedding output width, for a task that produces one.
    pub embedding_dim: Option<usize>,
    /// The served regression head's persisted distribution form.
    pub regression_form: Option<DistributionForm>,
    /// Input columns copied verbatim to the end of every output batch.
    pub passthrough: Vec<String>,
    /// The device kind this node must run on. A submitter placing the plan
    /// onto a kind other than its own session's names that kind here; nothing
    /// downstream rewrites it.
    pub device_kind: ComputeDeviceKind,
    /// The fan-out: how many partitions forward chunks concurrently.
    pub partitions: NonZeroUsize,
}

/// The process-local handles an [`InferenceExec`] runs against. Never
/// serialized: a node rebuilt in another process binds to that process's own.
#[derive(Clone)]
pub struct InferenceRuntime {
    /// Where the node's model is loaded from and kept.
    pub model_cache: Arc<ModelCache>,
    /// Observes every output batch.
    pub observer: Option<Arc<dyn InferenceObserver>>,
}

/// Runs a model over a numbered input, one forward per chunk, emitting the
/// common prefix columns followed by the task's own.
///
/// The node holds no forward admission of its own: each forward is admitted
/// by the device the model is resident on
/// ([`GpuScheduler::admit_forward`](crate::concurrency::GpuScheduler::admit_forward)),
/// so the bound holds across this node's partitions and across every other
/// node on that device — including one decoded, task by task, into an
/// executor that is running others.
pub struct InferenceExec {
    input: Arc<dyn ExecutionPlan>,
    spec: InferenceSpec,
    runtime: InferenceRuntime,
    /// `_chunk` over the input schema: the key this node requires its input
    /// hash-partitioned on.
    chunk_id: Arc<dyn PhysicalExpr>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for InferenceExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceExec")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl InferenceExec {
    /// Bind `spec` to `input` in this process. The one constructor: the
    /// planner, `with_new_children` and a wire decode all build the node here.
    ///
    /// Refuses an `input` that is not numbered (`_ordinal` and `_chunk`,
    /// `UInt64 NOT NULL`), and one of several partitions that is not
    /// hash-partitioned on the chunk id — either would let a chunk's rows be
    /// forwarded apart.
    pub fn bind(
        input: Arc<dyn ExecutionPlan>,
        spec: InferenceSpec,
        runtime: InferenceRuntime,
    ) -> DfResult<Self> {
        let input_schema = input.schema();
        let chunk_id = chunk_expr(input_schema.as_ref())?;
        let by_chunk = Distribution::HashPartitioned(vec![Arc::clone(&chunk_id)]);
        if !input
            .output_partitioning()
            .satisfaction(&by_chunk, input.equivalence_properties(), false)
            .is_satisfied()
        {
            return Err(DataFusionError::Plan(format!(
                "InferenceExec: an input of {} partitions must be hash-partitioned on the \
                 forward chunk, found {}",
                input.output_partitioning().partition_count(),
                input.output_partitioning()
            )));
        }
        let schema = build_output_schema(
            &spec.task,
            &input_schema,
            &spec.key_column,
            spec.embedding_dim,
            spec.regression_form.as_ref(),
            &spec.passthrough,
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        eq.add_ordering(ordinal_ordering(schema.as_ref())?);
        let properties = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Ok(Self {
            input,
            spec,
            runtime,
            chunk_id,
            properties: Arc::new(properties),
        })
    }

    /// What this node computes.
    pub fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    /// The child plan this node reads from.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl DisplayAs for InferenceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "InferenceExec: model={}, task={:?}, columns={:?}, batch_size={}, batch_tokens={}, \
             partitions={}",
            self.spec.source,
            self.spec.task,
            self.spec.content_columns,
            self.spec.chunk.rows,
            self.spec.chunk.tokens,
            self.spec.partitions
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

    /// A fan-out of one reads one partition. A wider one reads partitions
    /// hashed on the chunk id, so an optimizer that strips the exchange
    /// re-adds it from this requirement rather than serialising the node.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![if self.spec.partitions.get() == 1 {
            Distribution::SinglePartition
        } else {
            Distribution::HashPartitioned(vec![Arc::clone(&self.chunk_id)])
        }]
    }

    /// `true` exactly when the node fans out. `EnforceDistribution` adds a
    /// hash exchange over a single-partition child only for a node that
    /// benefits from partitioning; without this the numbered input's one
    /// partition would satisfy the hash requirement as it stands.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![self.spec.partitions.get() > 1]
    }

    /// Chunks are gathered from consecutive rows, so each partition must
    /// arrive in `_ordinal` order.
    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![ordinal_ordering(self.input.schema().as_ref())
            .ok()
            .map(OrderingRequirements::from)]
    }

    /// Each partition emits its chunks in the order it read them, which is
    /// what makes the `[_ordinal ASC]` output ordering true.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::bind(
            Arc::clone(&children[0]),
            self.spec.clone(),
            self.runtime.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let output_schema = self.schema();

        // A bounded channel (two batches) is the backpressure on the model.
        let mut builder = RecordBatchReceiverStreamBuilder::new(Arc::clone(&output_schema), 2);
        let tx = builder.tx();
        let runner = InferenceRunner::new(self.spec.clone(), self.runtime.clone());
        builder.spawn(async move { runner.run(input_stream, tx, output_schema).await });
        Ok(builder.build())
    }
}

/// `numbered` hashed on the chunk id `spec.partitions` ways: a stock
/// exchange, left out where it would be the identity.
fn exchanged(
    numbered: Arc<dyn ExecutionPlan>,
    spec: &InferenceSpec,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let partitions = spec.partitions.get();
    if partitions == 1 {
        return Ok(numbered);
    }
    let chunk_id = chunk_expr(numbered.schema().as_ref())?;
    Ok(Arc::new(RepartitionExec::try_new(
        numbered,
        Partitioning::Hash(vec![chunk_id], partitions),
    )?))
}

/// `inference`'s partitions merged back into the one `_ordinal` sequence: a
/// stock merge, left out where it would be the identity.
fn merged(inference: Arc<dyn ExecutionPlan>) -> DfResult<Arc<dyn ExecutionPlan>> {
    if inference.output_partitioning().partition_count() == 1 {
        return Ok(inference);
    }
    let ordering = ordinal_ordering(inference.schema().as_ref())?;
    Ok(Arc::new(SortPreservingMergeExec::new(ordering, inference)))
}

/// The plan that runs `spec` over `input`: the one shape in the module doc,
/// which every caller that runs a model builds through here.
///
/// `order` is how the rows are ordered before they are numbered; a keyed
/// order must name the column the rows are identified by.
pub fn plan_inference(
    input: Arc<dyn ExecutionPlan>,
    order: RowOrder,
    spec: InferenceSpec,
    runtime: InferenceRuntime,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let RowOrder::Keyed { key_column } = &order {
        if key_column != &spec.key_column {
            return Err(DataFusionError::Plan(format!(
                "plan_inference: the input is ordered by '{key_column}' but its rows are \
                 identified by '{}'",
                spec.key_column
            ))
            .into());
        }
    }
    let numbered: Arc<dyn ExecutionPlan> = Arc::new(NumberedInputExec::try_new(
        single_partition(input),
        order,
        spec.clone(),
        runtime.clone(),
    )?);
    let inference = InferenceExec::bind(exchanged(numbered, &spec)?, spec, runtime)?;
    Ok(merged(Arc::new(inference))?)
}

/// Restores the fan-out an optimized plan lost: every `InferenceExec` whose
/// input does not have the node's own `partitions` is put back over an
/// exchange of that width. See the module doc for why a declaration alone
/// cannot carry the count.
///
/// Registered after DataFusion's own rules. A node the optimizer left serial
/// gains the exchange below and the merge above, so it still presents one
/// `_ordinal`-ordered partition; one it fanned out at another width has the
/// exchange re-sized in place.
#[derive(Debug, Default)]
pub struct InferenceFanOut;

impl InferenceFanOut {
    fn restore(node: Arc<dyn ExecutionPlan>) -> DfResult<Transformed<Arc<dyn ExecutionPlan>>> {
        let Some(exec) = node.downcast_ref::<InferenceExec>() else {
            return Ok(Transformed::no(node));
        };
        let input = exec.input();
        let width = input.output_partitioning().partition_count();
        if width == exec.spec.partitions.get() {
            return Ok(Transformed::no(node));
        }
        let (numbered, presents_one) = match input.downcast_ref::<RepartitionExec>() {
            Some(exchange) => (Arc::clone(exchange.input()), false),
            None if width == 1 => (Arc::clone(input), true),
            None => return Ok(Transformed::no(node)),
        };
        let inference: Arc<dyn ExecutionPlan> = Arc::new(InferenceExec::bind(
            exchanged(numbered, &exec.spec)?,
            exec.spec.clone(),
            exec.runtime.clone(),
        )?);
        Ok(Transformed::yes(if presents_one {
            merged(inference)?
        } else {
            inference
        }))
    }
}

impl PhysicalOptimizerRule for InferenceFanOut {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(Self::restore).data()
    }

    fn name(&self) -> &str {
        "InferenceFanOut"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
