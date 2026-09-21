//! [`HopFoldExec`] — one propagation hop, as a plan operator.
//!
//! A hop is a sparse matrix–vector product `X⁽ᵏ⁾ = Â·X⁽ᵏ⁻¹⁾`, relationally: the
//! previous hop's state joined to the self-loop-augmented adjacency
//! `(g, n, w)` on `n`, so each row is one neighbour's vector for one group —
//! the node's own among them; then each group's rows folded into one vector.
//! The join, the shuffle and the sort are stock operators, so a hop
//! partitions, spills and is placed like any other plan. This operator is
//! the fold.
//!
//! # Why the fold is an operator and not an aggregate
//!
//! `f64` addition is not associative, so a byte-identical sum needs a fixed
//! fold order per group. Sorting an aggregate's input fixes the order rows
//! reach each accumulator, but a grouped aggregate runs in two phases — a
//! partial accumulator per input partition, merged afterwards — and the
//! partial sums it merges are cut wherever the partitioning cut the group. The
//! association, and so the last bit, would follow `target_partitions`. A fold
//! is reproducible only when one accumulator sees a whole group, in order, in
//! one pass; SQL cannot ask for that mode, an operator can **require** it:
//! hash-partitioned on `group` (a group lives in exactly one partition) and
//! sorted by `(group, neighbour)`. The fold is then a single left-to-right
//! pass holding one group's `O(d)` accumulator, and its result depends on the
//! rows alone — never on how many partitions carried them.
//!
//! The order is verified as the rows arrive, not assumed: a row that falls
//! before its predecessor in `(group, neighbour)` fails the hop. A row equal
//! to its predecessor is the same edge declared twice — or once in each
//! direction of an undirected read — and collapses into it, its weight the
//! larger (an order-free choice), so `Ã` is a set of edges however the
//! relation spelled them.
//!
//! # One hop
//!
//! Per group `g`, over its collapsed rows in neighbour order (its own row —
//! `n = g`, weight `1`, the one carrying `X⁽⁰⁾`, the running readout and
//! `d̃_g` — among them):
//!
//! ```text
//! Uniform           agg = (Σₙ xₙ) / count
//! DegreeNormalized  agg = (Σₙ xₙ · (1/√d̃ₙ)) · (1/√d̃_g)
//! EdgeSimilarity    agg = (Σₙ wₙ·xₙ) / (Σₙ wₙ)
//! X⁽ᵏ⁾ = agg                 (α = 0)    or    α·X⁽⁰⁾ + (1−α)·agg
//! ```
//!
//! each lane folded in `f64` through the engine's one lane operator
//! (`VectorReduce::fold_lanes`): a term is scaled by its own factor, added
//! in neighbour order, and the group's factor is applied once to the sum. The
//! hop absorbs block `k−1` into the readout (`BlockReadout::absorb`) and
//! hands it, `X⁽⁰⁾` and `d̃_g` forward.
//!
//! A group with no own row is a key the adjacency names that the node set
//! does not hold: an edge with an endpoint the embedding table lacks is no
//! edge, so the group is dropped. Every node has a self-loop and a state
//! row, so its own row is present by construction, and a drop is never a
//! node.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Array, Float64Array, RecordBatch};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{
    EquivalenceProperties, LexRequirement, OrderingRequirements, PhysicalSortRequirement,
};
use datafusion::physical_plan::expressions::col;
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::StreamExt;
use tokio::sync::mpsc::Sender;

use super::readout::BlockReadout;
use super::state::{
    augmented_degree, int64_column, list_vector, state_schema, utf8_column, StateBatchBuilder,
    ACC_COLUMN, DEGREE_COLUMN, KEY_COLUMN, X0_COLUMN, X_COLUMN,
};
use super::PropagationWeighting;
use crate::query::vector_agg_udaf::VectorReduce;

/// The group key column of a hop's input: the node whose vector is folded.
pub(crate) const GROUP_COLUMN: &str = "g";
/// The neighbour key column of a hop's input: the node contributing a term.
pub(crate) const NEIGHBOUR_COLUMN: &str = "n";
/// The declared weight column of a hop's input.
pub(crate) const WEIGHT_COLUMN: &str = "w";

/// What [`HopFoldExec`] is built from, and what crosses a process boundary
/// for it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HopSpec {
    pub(crate) weighting: PropagationWeighting,
    /// The teleport probability `α`; `0` is no restart.
    pub(crate) alpha: f64,
    /// The block this hop produces, `k ≥ 1`.
    pub(crate) block: usize,
    pub(crate) readout: BlockReadout,
}

/// One propagation hop over its sorted, hash-partitioned input. See the
/// module doc.
#[derive(Debug)]
pub struct HopFoldExec {
    input: Arc<dyn ExecutionPlan>,
    spec: HopSpec,
    properties: Arc<PlanProperties>,
}

impl HopFoldExec {
    /// Bind `spec` over `input`.
    ///
    /// The partitioning and order the fold needs are *declared*
    /// ([`ExecutionPlan::required_input_distribution`],
    /// [`ExecutionPlan::required_input_ordering`]), never checked here: a
    /// later hop plans over this node, and the optimizer that plans it
    /// rebuilds this node around children it is still rewriting toward those
    /// declarations. A constructor that refused the intermediate shapes would
    /// refuse the rewrite that satisfies it.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, spec: HopSpec) -> DfResult<Self> {
        let group = col(GROUP_COLUMN, input.schema().as_ref())?;
        col(NEIGHBOUR_COLUMN, input.schema().as_ref())?;
        let schema = state_schema();
        // A group is one output row under the same key, so an input hashed on
        // the group key is an output hashed on the state key.
        let partitioning = match input.output_partitioning() {
            Partitioning::Hash(exprs, partitions) if exprs.len() == 1 && exprs[0].eq(&group) => {
                Partitioning::Hash(vec![col(KEY_COLUMN, schema.as_ref())?], *partitions)
            }
            other => Partitioning::UnknownPartitioning(other.partition_count()),
        };
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            partitioning,
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            input.boundedness(),
        );
        Ok(Self {
            input,
            spec,
            properties: Arc::new(properties),
        })
    }

    /// The spec this node was built from.
    pub fn spec(&self) -> &HopSpec {
        &self.spec
    }
}

impl DisplayAs for HopFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "HopFoldExec: block={}, weighting={:?}, alpha={}",
            self.spec.block, self.spec.weighting, self.spec.alpha
        )
    }
}

impl ExecutionPlan for HopFoldExec {
    fn name(&self) -> &str {
        "HopFoldExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        let group = col(GROUP_COLUMN, self.input.schema().as_ref()).ok();
        vec![group.map_or(Distribution::SinglePartition, |g| {
            Distribution::HashPartitioned(vec![g])
        })]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        let schema = self.input.schema();
        let requirement = [GROUP_COLUMN, NEIGHBOUR_COLUMN]
            .iter()
            .map(|name| col(name, schema.as_ref()).map(|c| PhysicalSortRequirement::new(c, None)))
            .collect::<DfResult<Vec<_>>>()
            .ok()
            .and_then(LexRequirement::new);
        vec![requirement.map(OrderingRequirements::new)]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            self.spec.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        // Two batches of backpressure between the fold and its consumer.
        let mut builder = RecordBatchReceiverStreamBuilder::new(state_schema(), 2);
        let tx = builder.tx();
        let fold = HopFold {
            spec: self.spec.clone(),
            rows: StateBatchBuilder::new(context.session_config().batch_size()),
            group: None,
            previous: None,
        };
        builder.spawn(fold.run(input, tx));
        Ok(builder.build())
    }
}

/// The open term of a group: one neighbour's rows collapsed — the larger
/// declared weight of them — with its vector and degree.
struct Term {
    weight: f64,
    x: Vec<f64>,
    degree: u64,
}

/// The node's own row of a group: what only it carries.
struct OwnRow {
    x: Vec<f64>,
    x0: Vec<f64>,
    acc: Vec<f64>,
    degree: u64,
}

/// The group being folded. A term folds when the next neighbour opens, so
/// only the open term and the accumulator are held.
struct Group {
    key: String,
    lanes: Vec<f64>,
    count: u64,
    weight_sum: f64,
    own: Option<OwnRow>,
    term: Option<Term>,
}

impl Group {
    /// Fold the open term, if any, into the lanes under `weighting`: the
    /// term's own factor applied to it, then added in order.
    fn fold_open_term(&mut self, weighting: PropagationWeighting) {
        let Some(term) = self.term.take() else {
            return;
        };
        let factor = match weighting {
            PropagationWeighting::Uniform => 1.0,
            PropagationWeighting::EdgeSimilarity => term.weight,
            PropagationWeighting::DegreeNormalized => 1.0 / (term.degree as f64).sqrt(),
        };
        VectorReduce::Sum.fold_lanes(&mut self.lanes, |lane| term.x[lane] * factor);
        self.count += 1;
        self.weight_sum += term.weight;
    }
}

/// One partition's fold: the open group and the rows finished so far.
struct HopFold {
    spec: HopSpec,
    rows: StateBatchBuilder,
    group: Option<Group>,
    /// The last `(group, neighbour)` seen — the order check's left operand.
    previous: Option<(String, String)>,
}

impl HopFold {
    async fn run(
        mut self,
        mut input: SendableRecordBatchStream,
        tx: Sender<DfResult<RecordBatch>>,
    ) -> DfResult<()> {
        while let Some(batch) = input.next().await {
            for finished in self.fold_batch(&batch?)? {
                if tx.send(Ok(finished)).await.is_err() {
                    return Ok(());
                }
            }
        }
        let mut tail = Vec::new();
        if let Some(group) = self.group.take() {
            tail.extend(self.close(group)?);
        }
        tail.extend(self.rows.flush()?);
        for finished in tail {
            if tx.send(Ok(finished)).await.is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Fold one input batch, returning the state batches it completed.
    fn fold_batch(&mut self, batch: &RecordBatch) -> DfResult<Vec<RecordBatch>> {
        let groups = utf8_column(batch, GROUP_COLUMN)?;
        let neighbours = utf8_column(batch, NEIGHBOUR_COLUMN)?;
        let degrees = int64_column(batch, DEGREE_COLUMN)?;
        let weights = batch
            .column_by_name(WEIGHT_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "HopFoldExec: `{WEIGHT_COLUMN}` is not a Float64 column"
                ))
            })?;
        let weighting = self.spec.weighting;
        let mut finished = Vec::new();
        for row in 0..batch.num_rows() {
            let (g, n) = (groups.value(row), neighbours.value(row));
            let repeated = self.check_order(g, n)?;
            if self.group.as_ref().is_some_and(|open| open.key != g) {
                if let Some(group) = self.group.take() {
                    finished.extend(self.close(group)?);
                }
            }
            let x = list_vector(batch, X_COLUMN, row)?.ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "HopFoldExec: neighbour '{n}' of '{g}' carries no vector"
                ))
            })?;
            let weight = weights.value(row);
            let degree = augmented_degree(n, degrees.value(row))?;
            let group = self.group.get_or_insert_with(|| Group {
                key: g.to_string(),
                lanes: vec![0.0; x.len()],
                count: 0,
                weight_sum: 0.0,
                own: None,
                term: None,
            });
            if x.len() != group.lanes.len() {
                return Err(DataFusionError::Execution(format!(
                    "HopFoldExec: neighbour '{n}' of '{g}' is {} wide, the group is {}",
                    x.len(),
                    group.lanes.len()
                )));
            }
            match &mut group.term {
                // The same edge again: one term, the larger weight.
                Some(term) if repeated => term.weight = term.weight.max(weight),
                _ => {
                    group.fold_open_term(weighting);
                    group.term = Some(Term {
                        weight,
                        x: x.to_vec(),
                        degree,
                    });
                }
            }
            if g == n {
                group.own = Some(OwnRow {
                    x: x.to_vec(),
                    x0: list_vector(batch, X0_COLUMN, row)?
                        .unwrap_or_default()
                        .to_vec(),
                    acc: list_vector(batch, ACC_COLUMN, row)?
                        .unwrap_or_default()
                        .to_vec(),
                    degree,
                });
            }
        }
        Ok(finished)
    }

    /// `(g, n)` must not fall before the previous pair; `true` when it
    /// repeats it.
    fn check_order(&mut self, g: &str, n: &str) -> DfResult<bool> {
        let repeated = match &self.previous {
            Some((pg, pn)) => match (pg.as_str(), pn.as_str()).cmp(&(g, n)) {
                Ordering::Greater => {
                    return Err(DataFusionError::Execution(format!(
                        "HopFoldExec: ('{g}', '{n}') falls before ('{pg}', '{pn}') — the input \
                         is not sorted by (`{GROUP_COLUMN}`, `{NEIGHBOUR_COLUMN}`)"
                    )));
                }
                Ordering::Equal => true,
                Ordering::Less => false,
            },
            None => false,
        };
        match &mut self.previous {
            Some((pg, pn)) => {
                pg.clear();
                pg.push_str(g);
                pn.clear();
                pn.push_str(n);
            }
            None => self.previous = Some((g.to_string(), n.to_string())),
        }
        Ok(repeated)
    }

    /// Finish `group`: fold its open term, apply the group's factor,
    /// teleport, absorb the previous block into the readout, and append its
    /// state row — or drop it when it is no node.
    fn close(&mut self, mut group: Group) -> DfResult<Option<RecordBatch>> {
        let Some(own) = group.own.take() else {
            return Ok(None);
        };
        group.fold_open_term(self.spec.weighting);
        let Group {
            key,
            mut lanes,
            count,
            weight_sum,
            ..
        } = group;
        let factor = match self.spec.weighting {
            PropagationWeighting::Uniform => 1.0 / count as f64,
            // The own row weighs one, so the sum is positive.
            PropagationWeighting::EdgeSimilarity => 1.0 / weight_sum,
            PropagationWeighting::DegreeNormalized => 1.0 / (own.degree as f64).sqrt(),
        };
        lanes.iter_mut().for_each(|lane| *lane *= factor);
        let alpha = self.spec.alpha;
        if alpha != 0.0 {
            if own.x0.len() != lanes.len() {
                return Err(DataFusionError::Execution(format!(
                    "HopFoldExec: group '{key}' carries no X⁽⁰⁾ to teleport to"
                )));
            }
            for (lane, x0) in lanes.iter_mut().zip(&own.x0) {
                *lane = alpha * x0 + (1.0 - alpha) * *lane;
            }
        }
        // The own row's `x` is this node's block `k−1` — what the readout
        // absorbs now.
        let mut acc = own.acc;
        self.spec
            .readout
            .absorb(&mut acc, self.spec.block - 1, &own.x);
        self.rows.push(
            &key,
            &lanes,
            Some(own.x0.as_slice()),
            Some(acc.as_slice()),
            own.degree,
        )
    }
}
