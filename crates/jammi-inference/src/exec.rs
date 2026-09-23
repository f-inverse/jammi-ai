//! `InferenceExec` and the one plan every model forward runs in.
//!
//! [`plan_inference`] builds the same shape at every scale — one partition in
//! one process, `N` partitions in one process, `N` tasks on a cluster:
//!
//! ```text
//! SortExec [_ordinal ASC]                       restores the row order
//!   CoalescePartitionsExec                       one stream again
//!     InferenceExec                              N partitions
//!       RepartitionExec Hash([_chunk], N)        the fan-out
//!         NumberedInputExec                      orders, numbers, costs and chunks the rows
//!           CoalescePartitionsExec
//!             input
//! ```
//!
//! The sort, the exchange and the two coalesces are stock DataFusion
//! operators, so an optimizer re-derives them from this module's declared
//! requirements and a distributed planner cuts its stages at them. The
//! exchange and the coalesces are each the identity over one partition and
//! are left out there; the sort is a merge where the rows already leave the
//! model in `_ordinal` order (an arrival-order input) and is left out over
//! one such partition: at `N == 1` over a keyed input the plan is
//! `SortExec(InferenceExec(NumberedInputExec(input)))`.
//!
//! The rows leave the model in CHUNK order — the cost order the forward
//! wants — and the plan's output is `_ordinal` order, the order the rows
//! belong in ([`crate::numbered`]). The one sort that
//! restores it runs under the session's memory pool and spills to its disk
//! manager past the pool's limit, so its memory is bounded by `[engine]
//! memory_limit` at every input size and every fan-out: one sort reserves
//! `sort_spill_reservation_bytes` (10 MiB) for its merge whatever `N` is,
//! where a sort per partition would reserve it `N` times.
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
//! input, and carried as `_chunk` ([`crate::chunk`]). The
//! exchange hashes on that chunk id, so a chunk is never divided between
//! partitions, and every partition count — and every re-batching an exchange
//! or a shuffle performs on the way — forwards identical chunks and writes
//! identical bytes.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, OrderingRequirements, PhysicalExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    stream::RecordBatchReceiverStreamBuilder, DisplayAs, DisplayFormatType, Distribution,
    ExecutionPlan, ExecutionPlanProperties, Partitioning, PlanProperties,
};

use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

use crate::chunk::{chunk_expr, chunk_ordering};
use crate::numbered::{ordinal_ordering, NumberedInputExec};
use crate::runner::InferenceRunner;
use crate::runtime::InferenceRuntime;
use crate::schema::build_output_schema;
use crate::spec::{InferenceSpec, RowOrder};

/// `plan` over one partition: a stock coalesce, left out where it would be
/// the identity.
pub fn single_partition(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(plan))
    } else {
        plan
    }
}

/// Every [`InferenceExec`]'s spec in `plan`, in pre-order — walked with an
/// explicit stack, so a deep plan cannot exhaust the thread's. What a
/// consumer reads to know which models a plan runs.
pub fn inference_specs(plan: &Arc<dyn ExecutionPlan>) -> Vec<InferenceSpec> {
    let mut stack = vec![Arc::clone(plan)];
    let mut specs = Vec::new();
    while let Some(node) = stack.pop() {
        if let Some(inference) = node.downcast_ref::<InferenceExec>() {
            specs.push(inference.spec().clone());
        }
        stack.extend(node.children().into_iter().rev().cloned());
    }
    specs
}

/// Runs a model over a numbered input, one forward per chunk, emitting the
/// common prefix columns followed by the task's own.
///
/// The node holds no forward admission of its own: each forward is admitted
/// by the device the model is resident on
/// ([`BoundModel::admit_forward`](crate::runtime::BoundModel::admit_forward)),
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
    /// forwarded apart. The output keeps the input's `_ordinal` order where
    /// the input has one (an arrival-order input), and declares none
    /// otherwise.
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
        if input
            .equivalence_properties()
            .ordering_satisfy(ordinal_ordering(input_schema.as_ref())?)?
        {
            eq.add_ordering(ordinal_ordering(schema.as_ref())?);
        }
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
    /// arrive in chunk order.
    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![chunk_ordering(self.input.schema().as_ref())
            .ok()
            .map(OrderingRequirements::from)]
    }

    /// Each partition emits its chunks in the order it read them, and each
    /// chunk's rows in their input order.
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

/// `inference`'s output as the one `_ordinal` sequence: the partitions
/// coalesced — a stock coalesce, left out where it would be the identity —
/// and sorted on `_ordinal` by one stock sort. Partitions that already
/// leave the model in that order are merged instead, and one such partition
/// is returned as it is.
fn restored(inference: Arc<dyn ExecutionPlan>) -> DfResult<Arc<dyn ExecutionPlan>> {
    let ordering = ordinal_ordering(inference.schema().as_ref())?;
    let ordered = inference
        .equivalence_properties()
        .ordering_satisfy(ordering.clone())?;
    let partitions = inference.output_partitioning().partition_count();
    Ok(match (ordered, partitions) {
        (true, 1) => inference,
        (true, _) => Arc::new(SortPreservingMergeExec::new(ordering, inference)),
        (false, _) => Arc::new(SortExec::new(ordering, single_partition(inference))),
    })
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
) -> DfResult<Arc<dyn ExecutionPlan>> {
    if let RowOrder::Keyed { key_column, .. } = &order {
        if key_column != &spec.key_column {
            return Err(DataFusionError::Plan(format!(
                "plan_inference: the input is ordered by '{key_column}' but its rows are \
                 identified by '{}'",
                spec.key_column
            )));
        }
    }
    let numbered: Arc<dyn ExecutionPlan> = Arc::new(NumberedInputExec::try_new(
        single_partition(input),
        order,
        spec.clone(),
        runtime.clone(),
    )?);
    let inference = InferenceExec::bind(exchanged(numbered, &spec)?, spec, runtime)?;
    restored(Arc::new(inference))
}

/// Restores the fan-out an optimized plan lost: every `InferenceExec` whose
/// input does not have the node's own `partitions` is put back over an
/// exchange of that width. See the module doc for why a declaration alone
/// cannot carry the count.
///
/// Registered after DataFusion's own rules. A node the optimizer left serial
/// gains the exchange below and the sort and merge above, so it still
/// presents one `_ordinal`-ordered partition; one it fanned out at another
/// width has the exchange re-sized in place.
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
            restored(inference)?
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

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::physical_plan::common::collect;
    use datafusion::prelude::{SessionConfig, SessionContext};

    const WIDTH: i32 = 256;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(crate::schema::ORDINAL_COLUMN, DataType::UInt64, false),
            Field::new_fixed_size_list(
                "vector",
                Field::new("item", DataType::Float32, false),
                WIDTH,
                false,
            ),
        ]))
    }

    /// Rows `ordinals`, each with a 1 KiB vector derived from its ordinal.
    fn rows(ordinals: &[u64]) -> RecordBatch {
        let values: Vec<f32> = ordinals
            .iter()
            .flat_map(|&o| (0..WIDTH).map(move |i| (o as f32) + i as f32 * 1e-3))
            .collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(UInt64Array::from(ordinals.to_vec())),
                Arc::new(FixedSizeListArray::new(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    WIDTH,
                    Arc::new(Float32Array::from(values)),
                    None,
                )),
            ],
        )
        .unwrap()
    }

    /// The restore stage's memory is the session pool's, not the input's
    /// size: 160 MiB of rows across four partitions, each in the reverse of
    /// `_ordinal` order, come out as the one ascending sequence under the
    /// 64 MiB pool `[engine] memory_limit` refuses below — the sort spills
    /// past the pool's limit instead of exceeding it.
    #[tokio::test]
    async fn the_restored_order_is_bounded_by_the_memory_pool_and_spills() {
        let partitions: Vec<Vec<RecordBatch>> = (0..4u64)
            .map(|p| {
                // Partition p holds the ordinals congruent to p mod 4,
                // descending, 512 per batch.
                let mut ordinals: Vec<u64> = (0..40_960u64).map(|i| i * 4 + p).collect();
                ordinals.reverse();
                ordinals.chunks(512).map(rows).collect()
            })
            .collect();
        let total: usize = partitions.iter().flatten().map(RecordBatch::num_rows).sum();
        assert_eq!(total, 163_840);
        let input = MemorySourceConfig::try_new_exec(&partitions, schema(), None).unwrap();
        let plan = restored(input).unwrap();
        assert_eq!(plan.name(), "SortExec");
        assert_eq!(plan.children()[0].name(), "CoalescePartitionsExec");
        let sort = Arc::clone(&plan);

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(64 << 20)))
            .build_arc()
            .unwrap();
        let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), runtime);
        let out = collect(plan.execute(0, ctx.task_ctx()).unwrap())
            .await
            .unwrap();
        let ordinals: Vec<u64> = out
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ordinals, (0..163_840u64).collect::<Vec<_>>());
        let spills = sort.metrics().unwrap().spill_count().unwrap_or(0);
        assert!(spills > 0, "160 MiB sorted under a 64 MiB pool must spill");
    }

    /// An input that already leaves the model in `_ordinal` order (an
    /// arrival-order input) is restored by nothing at one partition and by
    /// the merge alone at several.
    #[tokio::test]
    async fn an_ordered_output_needs_no_sort() {
        let ordered = |partitions: &[Vec<RecordBatch>]| -> Arc<dyn ExecutionPlan> {
            let source = MemorySourceConfig::try_new(partitions, schema(), None)
                .unwrap()
                .try_with_sort_information(vec![ordinal_ordering(schema().as_ref()).unwrap()])
                .unwrap();
            DataSourceExec::from_data_source(source)
        };
        let one = ordered(&[vec![rows(&[0, 1, 2])]]);
        assert_eq!(restored(Arc::clone(&one)).unwrap().name(), one.name());
        let two = ordered(&[vec![rows(&[0, 2])], vec![rows(&[1, 3])]]);
        let plan = restored(two).unwrap();
        assert_eq!(plan.name(), "SortPreservingMergeExec");
        assert_ne!(plan.children()[0].name(), "SortExec");
    }
}
