//! The block readout — how the per-hop blocks `[X⁽⁰⁾, …, X⁽ᴷ⁾]` become one
//! output row — and [`ReadoutExec`], the operator that emits it.
//!
//! Every output a propagation offers is one linear readout of the hop history:
//!
//! ```text
//! out = ⊕ₖ wₖ · ν(X⁽ᵏ⁾)        over the blocks k the readout reads
//! ```
//!
//! where `⊕` is a lane-wise sum or a concatenation, `ν` is the identity or the
//! row-wise L2 normalisation, and an unread block contributes nothing. The
//! public [`PropagationOutput`] names three points of that family; each lowers
//! to one [`BlockReadout`], and one operator — [`BlockReadout::absorb`] — folds
//! a block into the running readout whichever point was asked for:
//!
//! | output | reads | `ν` | `⊕` | width |
//! |---|---|---|---|---|
//! | `Final` | block `K`, weight 1 | identity | sum | `d` |
//! | `JumpingKnowledge` | every block, weight 1 | L2 | concat | `(K+1)·d` |
//! | `WeightedSum { weights }` | each block with `wₖ ≠ 0` | L2 | sum | `d` |
//!
//! The readout is *running*: a hop state row carries the readout of the blocks
//! before it, so no block but the current one is ever held — the history is
//! never materialised, whatever the depth.
//!
//! # Arithmetic
//!
//! All of it in `f64`, in block order `k = 0, 1, …, K`. A sum readout starts
//! from the `+0.0` vector; `ν(x) = x / √(Σᵢ xᵢ²)` with the squares folded in
//! lane order, and a zero vector stays zero. The single `f32` cast
//! (round-to-nearest) happens once, at emit.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::StreamExt;

use jammi_db::error::{JammiError, Result};

use super::state::{HopState, ACC_COLUMN, KEY_COLUMN, X_COLUMN};
use super::PropagationOutput;

/// How the blocks a readout reads are joined into the output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum BlockJoin {
    /// Lane-wise sum — the output keeps the block width `d`.
    Sum,
    /// Concatenation in block order — the output is `d` per block read.
    Concat,
}

/// A lowered [`PropagationOutput`]: per block, the weight it is read with
/// (`None`: not read), whether a read block is L2-normalised first, and how
/// the read blocks join. See the module doc.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BlockReadout {
    weights: Vec<Option<f64>>,
    normalize: bool,
    join: BlockJoin,
}

impl BlockReadout {
    /// Lower `output` for a propagation of `hops` hops (blocks `0..=hops`).
    ///
    /// Refuses ([`JammiError::Config`]) a weighted sum whose weights do not
    /// number exactly `hops + 1`, carry a non-finite value, or are all zero —
    /// a readout that reads no block has no output.
    pub fn lower(output: &PropagationOutput, hops: usize) -> Result<Self> {
        let blocks = hops + 1;
        match output {
            PropagationOutput::Final => Ok(Self {
                weights: (0..blocks).map(|k| (k == hops).then_some(1.0)).collect(),
                normalize: false,
                join: BlockJoin::Sum,
            }),
            PropagationOutput::JumpingKnowledge => Ok(Self {
                weights: vec![Some(1.0); blocks],
                normalize: true,
                join: BlockJoin::Concat,
            }),
            PropagationOutput::WeightedSum { weights } => {
                if weights.len() != blocks {
                    return Err(JammiError::Config(format!(
                        "a weighted-sum readout over {hops} hops reads blocks 0..={hops} and \
                         needs {blocks} weights, got {}",
                        weights.len()
                    )));
                }
                if let Some(bad) = weights.iter().find(|w| !w.is_finite()) {
                    return Err(JammiError::Config(format!(
                        "a weighted-sum readout weight must be finite, got {bad}"
                    )));
                }
                if weights.iter().all(|w| *w == 0.0) {
                    return Err(JammiError::Config(
                        "a weighted-sum readout whose weights are all zero reads no block".into(),
                    ));
                }
                Ok(Self {
                    weights: weights.iter().map(|w| (*w != 0.0).then_some(*w)).collect(),
                    normalize: true,
                    join: BlockJoin::Sum,
                })
            }
        }
    }

    /// The output width for blocks of width `dimensions`.
    pub(crate) fn out_dim(&self, dimensions: usize) -> usize {
        match self.join {
            BlockJoin::Sum => dimensions,
            BlockJoin::Concat => dimensions * self.weights.iter().flatten().count(),
        }
    }

    /// The bytes of one hop state row over blocks of width `dimensions`: the
    /// current block, `X⁽⁰⁾`, and the running readout — what an out-of-core
    /// context sizes its batches by.
    pub(crate) fn state_row_bytes(&self, dimensions: usize) -> usize {
        (2 * dimensions + self.out_dim(dimensions)) * std::mem::size_of::<f64>()
    }

    /// The index of the last block, `K`.
    pub(crate) fn last_block(&self) -> usize {
        self.weights.len() - 1
    }

    /// Fold block `block` (the vector `x`) into the running readout `acc`. An
    /// empty `acc` is the readout of no block yet. A block the readout does
    /// not read leaves `acc` untouched.
    pub(crate) fn absorb(&self, acc: &mut Vec<f64>, block: usize, x: &[f64]) {
        let Some(weight) = self.weights.get(block).copied().flatten() else {
            return;
        };
        // The divisor of `ν`: absent when the readout does not normalise, and
        // for a zero block, which stays zero.
        let norm = self
            .normalize
            .then(|| x.iter().map(|v| v * v).sum::<f64>().sqrt())
            .filter(|norm| *norm != 0.0);
        // `w · (x / ‖x‖)` — normalise, then weigh: two roundings in that
        // order, never one folded `w / ‖x‖` factor.
        let term = |v: f64| match norm {
            Some(norm) => weight * (v / norm),
            None => weight * v,
        };
        match self.join {
            BlockJoin::Sum => {
                if acc.is_empty() {
                    acc.resize(x.len(), 0.0);
                }
                for (a, v) in acc.iter_mut().zip(x) {
                    *a += term(*v);
                }
            }
            BlockJoin::Concat => acc.extend(x.iter().map(|v| term(*v))),
        }
    }
}

/// The columns [`ReadoutExec`] emits — the inference-output columns the
/// embedding sink reads (`_status = "ok"` on every row: a propagation has no
/// per-row failure mode).
fn readout_schema(out_dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source", DataType::Utf8, false),
        Field::new("_model", DataType::Utf8, false),
        Field::new("_status", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                out_dim as i32,
            ),
            false,
        ),
    ]))
}

/// What [`ReadoutExec`] is built from, and what crosses a process boundary
/// for it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReadoutSpec {
    pub(crate) readout: BlockReadout,
    /// The block width `d`.
    pub(crate) dimensions: usize,
    /// Stamped on every row's `_source`.
    pub(crate) source_id: String,
    /// Stamped on every row's `_model`.
    pub(crate) model_id: String,
}

/// Emits a finished propagation: folds the last block `X⁽ᴷ⁾` into each state
/// row's running readout and casts it to the `f32` row the embedding sink
/// writes. A row-wise map — it keeps its input's partitioning and order.
#[derive(Debug)]
pub struct ReadoutExec {
    input: Arc<dyn ExecutionPlan>,
    spec: ReadoutSpec,
    properties: Arc<PlanProperties>,
}

impl ReadoutExec {
    /// Bind `spec` over an `input` of hop state rows.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, spec: ReadoutSpec) -> DfResult<Self> {
        HopState::check(input.schema().as_ref())?;
        let schema = readout_schema(spec.readout.out_dim(spec.dimensions));
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(
                input.output_partitioning().partition_count(),
            ),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            spec,
            properties: Arc::new(properties),
        })
    }

    /// The spec this node was built from.
    pub fn spec(&self) -> &ReadoutSpec {
        &self.spec
    }
}

impl DisplayAs for ReadoutExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "ReadoutExec: out_dim={}",
            self.spec.readout.out_dim(self.spec.dimensions)
        )
    }
}

impl ExecutionPlan for ReadoutExec {
    fn name(&self) -> &str {
        "ReadoutExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
        let schema = self.schema();
        let spec = self.spec.clone();
        let out_schema = Arc::clone(&schema);
        let stream = self
            .input
            .execute(partition, context)?
            .map(move |batch| emit(&spec, &out_schema, &batch?));
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// One state batch → one sink batch.
fn emit(spec: &ReadoutSpec, schema: &SchemaRef, batch: &RecordBatch) -> DfResult<RecordBatch> {
    let state = HopState::read(batch)?;
    let out_dim = spec.readout.out_dim(spec.dimensions);
    let last = spec.readout.last_block();
    let rows = batch.num_rows();
    let mut values: Vec<f32> = Vec::with_capacity(rows * out_dim);
    for row in 0..rows {
        let mut acc = state.vector(ACC_COLUMN, row)?.unwrap_or_default().to_vec();
        let x = state.vector(X_COLUMN, row)?.ok_or_else(|| {
            DataFusionError::Execution(format!("ReadoutExec: row {row} carries no `{X_COLUMN}`"))
        })?;
        spec.readout.absorb(&mut acc, last, x);
        if acc.len() != out_dim {
            return Err(DataFusionError::Execution(format!(
                "ReadoutExec: key '{}' read out {} lanes, the table is {out_dim} wide",
                state.key(row),
                acc.len()
            )));
        }
        values.extend(acc.iter().map(|v| *v as f32));
    }
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors = FixedSizeListArray::try_new(
        item,
        out_dim as i32,
        Arc::new(Float32Array::from(values)),
        None,
    )?;
    let keys: ArrayRef = Arc::clone(batch.column_by_name(KEY_COLUMN).ok_or_else(|| {
        DataFusionError::Execution(format!("ReadoutExec: no `{KEY_COLUMN}` column"))
    })?);
    let stamp = |value: &str| -> ArrayRef { Arc::new(StringArray::from(vec![value; rows])) };
    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            keys,
            stamp(&spec.source_id),
            stamp(&spec.model_id),
            stamp("ok"),
            Arc::new(vectors),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lowered(output: PropagationOutput, hops: usize) -> BlockReadout {
        BlockReadout::lower(&output, hops).unwrap()
    }

    /// Run a whole history through the one operator.
    fn read_out(readout: &BlockReadout, history: &[Vec<f64>]) -> Vec<f64> {
        let mut acc = Vec::new();
        for (k, block) in history.iter().enumerate() {
            readout.absorb(&mut acc, k, block);
        }
        acc
    }

    #[test]
    fn final_reads_only_the_last_block_unnormalised() {
        let readout = lowered(PropagationOutput::Final, 2);
        let out = read_out(&readout, &[vec![1.0, 2.0], vec![9.0, 9.0], vec![5.0, 6.0]]);
        assert_eq!(out, vec![5.0, 6.0]);
        assert_eq!(readout.out_dim(2), 2);
    }

    #[test]
    fn jumping_knowledge_concats_every_block_unit_normalised() {
        let readout = lowered(PropagationOutput::JumpingKnowledge, 2);
        let out = read_out(&readout, &[vec![3.0, 4.0], vec![0.0, 5.0], vec![6.0, 8.0]]);
        assert_eq!(readout.out_dim(2), 6);
        assert_eq!(out.len(), 6);
        for block in out.chunks(2) {
            let norm = (block[0] * block[0] + block[1] * block[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn weighted_sum_skips_zero_weight_blocks_and_sums_normalised_ones() {
        let readout = lowered(
            PropagationOutput::WeightedSum {
                weights: vec![0.0, 1.0, 2.0],
            },
            2,
        );
        // Block 0 is unread, so its content (even a non-finite one) cannot
        // reach the output.
        let out = read_out(
            &readout,
            &[vec![f64::NAN, f64::NAN], vec![3.0, 4.0], vec![0.0, 5.0]],
        );
        assert_eq!(
            out,
            vec![0.0 + 1.0 * (3.0 / 5.0), 1.0 * (4.0 / 5.0) + 2.0 * 1.0]
        );
        assert_eq!(readout.out_dim(2), 2);
    }

    #[test]
    fn a_zero_block_stays_zero_under_normalisation() {
        let readout = lowered(PropagationOutput::JumpingKnowledge, 0);
        assert_eq!(read_out(&readout, &[vec![0.0, 0.0]]), vec![0.0, 0.0]);
    }

    #[test]
    fn weighted_sum_refuses_a_shape_that_reads_nothing_or_mismatches_the_hops() {
        let refuse = |weights: Vec<f64>, hops| {
            BlockReadout::lower(&PropagationOutput::WeightedSum { weights }, hops).unwrap_err()
        };
        assert!(matches!(refuse(vec![], 0), JammiError::Config(_)));
        assert!(matches!(refuse(vec![1.0, 1.0], 2), JammiError::Config(_)));
        assert!(matches!(refuse(vec![0.0, 0.0], 1), JammiError::Config(_)));
        assert!(matches!(
            refuse(vec![1.0, f64::NAN], 1),
            JammiError::Config(_)
        ));
    }
}
