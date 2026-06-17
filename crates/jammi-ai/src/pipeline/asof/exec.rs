//! [`AsofJoinExec`] — the hand-built physical operator for the as-of join.
//!
//! Matches the engine's existing operator idiom ([`crate::operator`]'s
//! `InferenceExec`/`AnnSearchExec`): a concrete [`ExecutionPlan`] the verb wraps
//! its inputs in and drives directly, not a logical node behind an
//! `ExtensionPlanner` (the engine plans no `LogicalPlan` for its compute verbs).
//!
//! The operator declares its semantic requirements truthfully — both children
//! hash-partitioned on the equality keys and sorted by (`by`..., `time`) — so a
//! planner that ever drives it inserts the right shuffle/sort, and a reader sees
//! the operator's contract in `EXPLAIN`. The verb satisfies those requirements
//! by wrapping each input in a [`SortExec`](datafusion::physical_plan::sorts)
//! before construction, then executes the single output partition: each side is
//! collected and concatenated into one sorted run, and [`merge_partition`]
//! advances a single pointer per `by`-group. "At most one match" is what lets
//! the merge never backtrack.

use std::any::Any;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::compute::concat_batches;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{
    EquivalenceProperties, LexRequirement, OrderingRequirements, PhysicalSortRequirement,
};
use datafusion::physical_plan::expressions::col;
use datafusion::physical_plan::{
    stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, PlanProperties,
};
use futures::stream;

use super::merge::{merge_partition, output_schema, SortedPartition};
use super::spec::{AsofError, AsofJoinSpec, TieBreak};
use arrow::datatypes::SchemaRef;

/// Sort-merge as-of join. Holds both child plans, the validated spec, the
/// resolved right-projection indices, and the cached plan properties (the
/// left-outer output schema and the single output partition).
pub struct AsofJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    spec: AsofJoinSpec,
    /// Right-column indices to attach, resolved once at construction (an empty
    /// `spec.project` expands to every non-`by`, non-`time` right column).
    project: Vec<usize>,
    out_schema: SchemaRef,
    properties: PlanProperties,
}

impl AsofJoinExec {
    /// Build the operator over two child plans and a validated spec. The spec is
    /// validated against the children's schemas (key presence, temporal
    /// orderability, type match, the `Nearest`-needs-numeric rule), the
    /// projection is resolved, and the output schema is derived. The children
    /// must already satisfy the operator's ordering requirement; the verb wraps
    /// them in `SortExec` to guarantee it.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        spec: AsofJoinSpec,
    ) -> Result<Self, AsofError> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        spec.validate_against(&left_schema, &right_schema)?;

        let project = resolve_projection(&spec, &right_schema)?;
        let out_schema = output_schema(&left_schema, &right_schema, &project);
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&out_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Ok(Self {
            left,
            right,
            spec,
            project,
            out_schema,
            properties,
        })
    }

    /// The lexicographic ordering requirement for one side: ascending by
    /// (`by`..., `time`), and — under a `ByColumnDesc` tie-break — by that
    /// secondary column last so the maximal value of a coincident-time run is the
    /// run's final row (the merge then reads it as the winner). Ascending on the
    /// secondary column is the correct shape: the merge takes the LAST eligible
    /// fact, so ascending puts the maximal tie-break value last.
    fn ordering_requirement(
        &self,
        schema: &SchemaRef,
        by: &[String],
        time: &str,
        include_tie_break: bool,
    ) -> datafusion::error::Result<LexRequirement> {
        let mut reqs = Vec::with_capacity(by.len() + 2);
        for key in by {
            reqs.push(PhysicalSortRequirement::new(col(key, schema)?, None));
        }
        reqs.push(PhysicalSortRequirement::new(col(time, schema)?, None));
        // The tie-break column lives only on the facts side — it disambiguates
        // coincident-time facts, so only the right ordering carries it.
        if include_tie_break {
            if let TieBreak::ByColumnDesc(secondary) = &self.spec.tie_break {
                reqs.push(PhysicalSortRequirement::new(col(secondary, schema)?, None));
            }
        }
        // `reqs` always carries the temporal key, so it is non-empty and
        // `LexRequirement::new` is `Some`; a `None` here would mean the time key
        // was dropped, which is a planner-contract violation worth surfacing.
        LexRequirement::new(reqs).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "asof_join ordering requirement is unexpectedly empty".into(),
            )
        })
    }
}

/// Resolve `spec.project` (right-column names) to indices; an empty projection
/// expands to every right column that is neither a `by` key nor the temporal
/// key (the "all non-key columns" default).
fn resolve_projection(
    spec: &AsofJoinSpec,
    right_schema: &SchemaRef,
) -> Result<Vec<usize>, AsofError> {
    if spec.project.is_empty() {
        let excluded: Vec<&String> = spec.right.by.iter().chain([&spec.right.time]).collect();
        Ok((0..right_schema.fields().len())
            .filter(|&i| !excluded.iter().any(|e| *e == right_schema.field(i).name()))
            .collect())
    } else {
        spec.project
            .iter()
            .map(|name| {
                right_schema
                    .index_of(name)
                    .map_err(|_| AsofError::MissingByKey {
                        column: name.clone(),
                        side: "right",
                    })
            })
            .collect()
    }
}

impl std::fmt::Debug for AsofJoinExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsofJoinExec")
            .field("direction", &self.spec.direction)
            .field("boundary", &self.spec.boundary)
            .field("by", &self.spec.left.by)
            .finish()
    }
}

impl DisplayAs for AsofJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "AsofJoinExec: by={:?}, time=({}, {}), direction={:?}, boundary={:?}",
            self.spec.left.by,
            self.spec.left.time,
            self.spec.right.time,
            self.spec.direction,
            self.spec.boundary,
        )
    }
}

impl ExecutionPlan for AsofJoinExec {
    fn name(&self) -> &str {
        "AsofJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    /// Both children must arrive hash-partitioned on their equality keys so each
    /// partition holds whole `by`-groups — the precondition the single-pointer
    /// merge keys on. An empty `by` (one global group) requires a single
    /// partition on each side.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        let dist = |schema: &SchemaRef, by: &[String]| {
            if by.is_empty() {
                Distribution::SinglePartition
            } else {
                let exprs = by
                    .iter()
                    .filter_map(|k| col(k, schema).ok())
                    .collect::<Vec<_>>();
                Distribution::HashPartitioned(exprs)
            }
        };
        vec![
            dist(&self.left.schema(), &self.spec.left.by),
            dist(&self.right.schema(), &self.spec.right.by),
        ]
    }

    /// Each child sorted by (`by`..., `time`[, tie-break]).
    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        let left = self
            .ordering_requirement(
                &self.left.schema(),
                &self.spec.left.by,
                &self.spec.left.time,
                false,
            )
            .ok()
            .map(OrderingRequirements::new);
        let right = self
            .ordering_requirement(
                &self.right.schema(),
                &self.spec.right.by,
                &self.spec.right.time,
                true,
            )
            .ok()
            .map(OrderingRequirements::new);
        vec![left, right]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(
            AsofJoinExec::try_new(
                Arc::clone(&children[0]),
                Arc::clone(&children[1]),
                self.spec.clone(),
            )
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
        ))
    }

    /// Execute the single output partition: collect and concatenate each side
    /// into one sorted run, then merge. Both sides are bounded result/source
    /// scans, so the full collect is the operator's natural shape — the merge is
    /// a single linear pass over the sorted runs.
    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let left = Arc::clone(&self.left);
        let right = Arc::clone(&self.right);
        let spec = self.spec.clone();
        let project = self.project.clone();
        let out_schema = Arc::clone(&self.out_schema);
        let left_schema = left.schema();
        let right_schema = right.schema();

        let result = stream::once(async move {
            let to_df = |e: AsofError| match e {
                AsofError::DataFusion(d) => d,
                other => datafusion::error::DataFusionError::External(Box::new(other)),
            };

            let left_batch = collect_concat(&left, Arc::clone(&context), &left_schema).await?;
            let right_batch = collect_concat(&right, Arc::clone(&context), &right_schema).await?;

            let tie = match &spec.tie_break {
                TieBreak::ByColumnDesc(c) => Some(c.as_str()),
                TieBreak::Error => None,
            };
            // The tie-break column is a facts-side column only — the spine has
            // no secondary disambiguator, so the left side resolves none.
            let left_part =
                SortedPartition::resolve(left_batch, &spec.left.by, &spec.left.time, None, "left")
                    .map_err(to_df)?;
            let right_part = SortedPartition::resolve(
                right_batch,
                &spec.right.by,
                &spec.right.time,
                tie,
                "right",
            )
            .map_err(to_df)?;

            merge_partition(&left_part, &right_part, &spec, &project, &out_schema).map_err(to_df)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.out_schema),
            result,
        )))
    }
}

/// Execute every partition of a child and concatenate into one batch in the
/// child's declared (sorted) order. The merge needs each `by`-group contiguous
/// and ordered; a single partition delivered by the verb's `SortExec` already
/// is, and concatenating preserves that order.
async fn collect_concat(
    plan: &Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
    schema: &SchemaRef,
) -> datafusion::error::Result<arrow::array::RecordBatch> {
    let partitions = plan.properties().output_partitioning().partition_count();
    let mut batches = Vec::new();
    for p in 0..partitions {
        let stream = plan.execute(p, Arc::clone(&context))?;
        let mut collected = datafusion::physical_plan::common::collect(stream).await?;
        batches.append(&mut collected);
    }
    concat_batches(schema, &batches)
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
}
