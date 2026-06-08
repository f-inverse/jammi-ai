//! Element-wise vector-aggregation UDAFs — the engine's first DataFusion
//! aggregate functions.
//!
//! Each function reduces a group of fixed-width float vectors to a single
//! vector of the same width by applying one element-wise reduction across the
//! group:
//!
//! ```sql
//! SELECT vector_mean(v) FROM embeddings GROUP BY cluster
//! SELECT vector_sum(v)  FROM embeddings GROUP BY cluster
//! SELECT vector_max(v)  FROM embeddings GROUP BY cluster
//! ```
//!
//! The argument is a `FixedSizeList<Float32>` column; the result is a
//! `FixedSizeList<Float32>` of the same width. Each reduction folds every lane
//! with a mathematically commutative operator (`+` for sum/mean, `max` for max)
//! in `f64`, so the **value** is permutation-invariant: it does not depend on
//! the order rows arrive in.
//!
//! It is **not** byte-identical across arbitrary parallel partitionings. `f64`
//! addition is *not* associative — `(a + b) + c` can differ in the last bit from
//! `a + (b + c)` — and a parallel plan is free to split a group across
//! accumulators and merge their partial sums in an order the plan, not the
//! caller, decides. So `vector_sum`/`vector_mean` over the same multiset can
//! land on bit-for-bit different results under different `target_partitions`.
//! (`vector_max` *is* exact and so byte-stable, but the additive arms are the
//! ones callers reach for.) A caller that needs a reproducible, byte-identical
//! reduction must fold over a *fixed* order it imposes itself — which the
//! streaming SQL aggregate cannot guarantee. `fold_vectors_in_order` is that
//! fixed-order reduction, exposed (crate-internal) for exactly such callers; the
//! streaming accumulator below applies the identical per-lane operator without
//! the order guarantee.
//!
//! One reduction operator, three SQL names: the three functions share a single
//! `VectorAggAccumulator` parameterised by [`VectorReduce`], and that
//! accumulator folds through the same `fold_vectors_in_order` primitive a
//! fixed-order caller uses. Adding a fourth reduction is a new enum arm and a
//! new registration, not a new accumulator.

use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion::error::Result;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;

/// The element-wise reduction a vector-aggregation UDAF applies across a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorReduce {
    /// Element-wise arithmetic mean (sum of each lane divided by the row count).
    Mean,
    /// Element-wise sum of each lane.
    Sum,
    /// Element-wise maximum of each lane.
    Max,
}

impl VectorReduce {
    /// The SQL name this reduction registers under.
    const fn udaf_name(self) -> &'static str {
        match self {
            VectorReduce::Mean => "vector_mean",
            VectorReduce::Sum => "vector_sum",
            VectorReduce::Max => "vector_max",
        }
    }

    /// The identity each lane's accumulator starts from before the first row.
    const fn identity(self) -> f64 {
        match self {
            // Sum and mean accumulate additively from zero.
            VectorReduce::Mean | VectorReduce::Sum => 0.0,
            // Max folds from negative infinity so the first real value wins.
            VectorReduce::Max => f64::NEG_INFINITY,
        }
    }

    /// Fold one lane value into the running per-lane accumulator. Mathematically
    /// commutative in every arm, which makes the aggregate's *value*
    /// permutation-invariant (the additive arms are not bit-associative, so the
    /// result is byte-stable only over a fixed fold order — see the module doc).
    fn fold(self, acc: f64, value: f64) -> f64 {
        match self {
            VectorReduce::Mean | VectorReduce::Sum => acc + value,
            VectorReduce::Max => acc.max(value),
        }
    }

    /// Fold one `width`-wide vector, read lane-by-lane from `value`, into the
    /// running `lanes`. The single element-wise reduction loop both
    /// [`fold_vectors_in_order`] and the streaming accumulator descend through,
    /// so there is one lane operator across the fixed-order and streaming paths.
    fn fold_lanes(self, lanes: &mut [f64], value: impl Fn(usize) -> f64) {
        for (offset, acc) in lanes.iter_mut().enumerate() {
            *acc = self.fold(*acc, value(offset));
        }
    }

    /// Finalise a lane: mean divides the accumulated sum by the row count;
    /// sum and max return the accumulator unchanged.
    fn finalize(self, acc: f64, count: u64) -> f64 {
        match self {
            VectorReduce::Mean => acc / count as f64,
            VectorReduce::Sum | VectorReduce::Max => acc,
        }
    }
}

/// Fold an **ordered** sequence of equal-width vectors into one `dimensions`-wide
/// `f64` result under `reduce`, folding in iteration order. Lanes fold in `f64`
/// regardless of the input lane type (`f32` for the SQL vector columns, `f64` for
/// a caller that already works in `f64`), so an `f64`-precision caller stays
/// lossless.
///
/// This is the single element-wise reduction in the engine: the streaming
/// [`VectorAggAccumulator`] folds through the same lane operator (so the SQL UDAF
/// and a direct caller apply the identical per-lane reduction), and a caller that
/// needs a byte-identical reduction the streaming aggregate cannot guarantee
/// (because a parallel plan fixes neither the fold nor the merge order, and `f64`
/// `+` is non-associative) calls this directly over a canonical order it imposes
/// itself.
///
/// `Mean` divides each lane by the number of vectors folded; an empty sequence
/// yields the reduction's identity in every lane (and `Mean` of nothing is left
/// at identity rather than dividing by zero). Every input vector must be exactly
/// `dimensions` wide — equal width is the contract, the same one the SQL UDAF
/// enforces at plan time.
pub(crate) fn fold_vectors_in_order<V, T>(
    vectors: impl IntoIterator<Item = V>,
    reduce: VectorReduce,
    dimensions: usize,
) -> Vec<f64>
where
    V: AsRef<[T]>,
    T: Copy + Into<f64>,
{
    let mut lanes = vec![reduce.identity(); dimensions];
    let mut count: u64 = 0;
    for vector in vectors {
        let vector = vector.as_ref();
        reduce.fold_lanes(&mut lanes, |offset| vector[offset].into());
        count += 1;
    }
    if count == 0 {
        return lanes;
    }
    lanes
        .iter()
        .map(|&acc| reduce.finalize(acc, count))
        .collect()
}

/// Build the three vector-aggregation [`AggregateUDF`]s and register them on
/// `ctx`. Idempotent per session: re-registering replaces the prior binding.
pub fn register_vector_agg_udafs(ctx: &SessionContext) {
    for reduce in [VectorReduce::Mean, VectorReduce::Sum, VectorReduce::Max] {
        ctx.register_udaf(AggregateUDF::from(VectorAggUdaf::new(reduce)));
    }
}

/// One vector-aggregation aggregate function. Carries the reduction it applies
/// and a `user_defined` signature, since the accepted argument is any
/// `FixedSizeList<Float32>` width and the validation happens against the actual
/// input field at plan time.
#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorAggUdaf {
    reduce: VectorReduce,
    signature: Signature,
}

impl VectorAggUdaf {
    fn new(reduce: VectorReduce) -> Self {
        Self {
            reduce,
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

/// Read the lane width out of a `FixedSizeList<Float32>` data type, or a typed
/// planning error naming what was actually supplied. This is the single place
/// the engine asserts the argument shape these UDAFs accept.
fn fixed_width_float32(data_type: &DataType) -> Result<i32> {
    match data_type {
        DataType::FixedSizeList(field, width) if field.data_type() == &DataType::Float32 => {
            Ok(*width)
        }
        other => {
            exec_err!("vector aggregate requires a FixedSizeList<Float32> argument, got {other}")
        }
    }
}

/// The `FixedSizeList<Float32>` field of the given width, shaped exactly like
/// the engine's vector columns (child field named `item`, non-nullable child,
/// nullable list).
fn vector_field(name: &str, width: i32) -> FieldRef {
    Arc::new(Field::new_fixed_size_list(
        name,
        Field::new("item", DataType::Float32, false),
        width,
        true,
    ))
}

impl AggregateUDFImpl for VectorAggUdaf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.reduce.udaf_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [arg] = arg_types else {
            return exec_err!(
                "{} takes exactly one argument, got {}",
                self.name(),
                arg_types.len()
            );
        };
        // The output vector has the same width as the input vector.
        let width = fixed_width_float32(arg)?;
        Ok(vector_field("item", width).data_type().clone())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        // No coercion: the argument must already be FixedSizeList<Float32>.
        // Validate here so a wrong type is a planning error, not a downcast
        // failure at execution time.
        let [arg] = arg_types else {
            return exec_err!(
                "{} takes exactly one argument, got {}",
                self.name(),
                arg_types.len()
            );
        };
        fixed_width_float32(arg)?;
        Ok(arg_types.to_vec())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let width = fixed_width_float32(acc_args.return_type())?;
        Ok(Box::new(VectorAggAccumulator::new(
            self.reduce,
            width as usize,
        )))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let width = fixed_width_float32(args.return_type())?;
        // Partial state is the running per-lane fold (in f64) plus the row
        // count. Mean needs the count to divide at finalisation; sum and max
        // carry it too so all three share one state schema and one
        // `merge_batch`.
        let acc_field = Arc::new(Field::new_fixed_size_list(
            format!("{}_acc", args.name),
            Field::new("item", DataType::Float64, false),
            width,
            false,
        ));
        let count_field = Arc::new(Field::new(
            format!("{}_count", args.name),
            DataType::UInt64,
            false,
        ));
        Ok(vec![acc_field, count_field])
    }
}

/// Folds a group of fixed-width float vectors into one running per-lane
/// accumulator. Holds `width` lanes plus the count of rows folded in.
#[derive(Debug)]
struct VectorAggAccumulator {
    reduce: VectorReduce,
    /// Per-lane running fold, length `width`.
    lanes: Vec<f64>,
    /// Rows folded so far (drives the mean divisor; null rows are skipped).
    count: u64,
}

impl VectorAggAccumulator {
    fn new(reduce: VectorReduce, width: usize) -> Self {
        Self {
            reduce,
            lanes: vec![reduce.identity(); width],
            count: 0,
        }
    }

    /// Fold every non-null vector in `list` into the running lanes, advancing
    /// the row count by each row's contribution. Each row is a contiguous
    /// `width`-long slice of the backing values array of type `T`; `lane` reads
    /// one element as `f64`, and `row_count` is the count this row carries (one
    /// for an input vector in `update_batch`, the merged partition's count in
    /// `merge_batch`). A null row contributes nothing.
    ///
    /// This is the single per-row fold both `update_batch` and `merge_batch`
    /// descend through, so the input path and the partial-state merge path apply
    /// the identical lane reduction.
    fn fold_vectors<T>(
        &mut self,
        list: &FixedSizeListArray,
        lane: impl Fn(&T, usize) -> f64,
        row_count: impl Fn(usize) -> u64,
    ) -> Result<()>
    where
        T: Array + 'static,
    {
        let width = self.lanes.len();
        let values =
            list.values().as_any().downcast_ref::<T>().ok_or_else(|| {
                exec_datafusion_err!("vector aggregate: unexpected child array type")
            })?;
        for row in 0..list.len() {
            if list.is_null(row) {
                continue;
            }
            let base = row * width;
            self.reduce
                .fold_lanes(&mut self.lanes, |offset| lane(values, base + offset));
            self.count += row_count(row);
        }
        Ok(())
    }
}

/// Downcast `array` to a `FixedSizeListArray` whose width matches `expected`,
/// or a typed error. Shared by `update_batch` (input vectors) and `merge_batch`
/// (partial-state vectors).
fn fixed_size_list_of_width(array: &ArrayRef, expected: usize) -> Result<&FixedSizeListArray> {
    let list = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| exec_datafusion_err!("vector aggregate: argument is not a FixedSizeList"))?;
    if list.value_length() as usize != expected {
        return exec_err!(
            "vector aggregate: expected vector width {expected}, got {}",
            list.value_length()
        );
    }
    Ok(list)
}

impl Accumulator for VectorAggAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let [input] = values else {
            return exec_err!("vector aggregate takes exactly one argument");
        };
        let list = fixed_size_list_of_width(input, self.lanes.len())?;
        // Every input vector counts as one row.
        self.fold_vectors::<Float32Array>(list, |a, i| a.value(i) as f64, |_| 1)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let width = self.lanes.len() as i32;
        // An empty group (no rows folded) yields a null vector — there is no
        // mean of nothing, and sum/max over nothing has no defined value. The
        // null scalar is one list row whose validity bit is unset.
        if self.count == 0 {
            return ScalarValue::try_from(vector_field("item", width).data_type());
        }
        let out: Float32Array = self
            .lanes
            .iter()
            .map(|&acc| self.reduce.finalize(acc, self.count) as f32)
            .collect();
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let list = FixedSizeListArray::new(field, width, Arc::new(out), None);
        Ok(ScalarValue::FixedSizeList(Arc::new(list)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.lanes.capacity() * std::mem::size_of::<f64>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let width = self.lanes.len() as i32;
        let acc = Float64Array::from(self.lanes.clone());
        let field = Arc::new(Field::new("item", DataType::Float64, false));
        let list = FixedSizeListArray::new(field, width, Arc::new(acc), None);
        Ok(vec![
            ScalarValue::FixedSizeList(Arc::new(list)),
            ScalarValue::UInt64(Some(self.count)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let [acc_states, count_states] = states else {
            return exec_err!("vector aggregate merge expects [acc, count] state");
        };
        let lists = fixed_size_list_of_width(acc_states, self.lanes.len())?;
        let counts = count_states
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| exec_datafusion_err!("vector aggregate: state count is not UInt64"))?;
        // Merge each peer accumulator's partial fold (an f64 vector) and carry
        // over the rows it had folded, through the same per-row primitive the
        // input path uses.
        self.fold_vectors::<Float64Array>(
            lists,
            |a, i| a.value(i),
            |partition| counts.value(partition),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::RecordBatch;
    use arrow::datatypes::Schema;
    use datafusion::prelude::SessionContext;

    /// A `FixedSizeList<Float32>` column shaped like the engine's vector columns,
    /// one row per inner `Vec<f32>`.
    fn vector_column(rows: &[Vec<f32>], width: i32) -> ArrayRef {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        Arc::new(FixedSizeListArray::new(
            field,
            width,
            Arc::new(Float32Array::from(flat)),
            None,
        ))
    }

    /// Run `vector_<reduce>(v)` over a single-group table built from `rows`,
    /// returning the one output vector.
    async fn run_reduce(name: &str, rows: &[Vec<f32>], width: i32) -> Vec<f32> {
        let ctx = SessionContext::new();
        register_vector_agg_udafs(&ctx);
        let schema = Arc::new(Schema::new(vec![Field::new_fixed_size_list(
            "v",
            Field::new("item", DataType::Float32, false),
            width,
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![vector_column(rows, width)]).unwrap();
        ctx.register_batch("t", batch).unwrap();
        let df = ctx
            .sql(&format!("SELECT {name}(v) AS r FROM t"))
            .await
            .unwrap();
        let out = df.collect().await.unwrap();
        let col = out[0].column(0);
        let list = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let values = list
            .value(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        values
    }

    /// Reference reduction over `rows`, computed independently of the UDAF.
    fn reference(reduce: VectorReduce, rows: &[Vec<f32>], width: usize) -> Vec<f32> {
        let mut lanes = vec![reduce.identity(); width];
        for row in rows {
            for (lane, &v) in lanes.iter_mut().zip(row) {
                *lane = reduce.fold(*lane, v as f64);
            }
        }
        lanes
            .iter()
            .map(|&acc| reduce.finalize(acc, rows.len() as u64) as f32)
            .collect()
    }

    fn sample_rows() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, -6.0, 7.0, 0.5],
            vec![-1.0, 2.5, 0.0, 8.0],
            vec![3.0, 3.0, 3.0, 3.0],
            vec![0.25, 0.5, 0.75, 1.0],
        ]
    }

    #[tokio::test]
    async fn matches_reference_reduce() {
        let rows = sample_rows();
        for (name, reduce) in [
            ("vector_sum", VectorReduce::Sum),
            ("vector_mean", VectorReduce::Mean),
            ("vector_max", VectorReduce::Max),
        ] {
            let got = run_reduce(name, &rows, 4).await;
            let want = reference(reduce, &rows, 4);
            assert_eq!(got, want, "{name} disagreed with reference reduce");
        }
    }

    /// A single sequential accumulator folds rows in arrival order. Shuffling the
    /// rows of one batch under the *same* (single-partition) plan reorders that
    /// one fold, and the result is byte-identical — `f64` `+` over a fixed fold
    /// order is reproducible. This is the guarantee that actually holds; it does
    /// NOT extend to splitting the group across partitions, where the plan
    /// chooses the merge order and `f64` non-associativity can flip the last bit
    /// (see `multi_partition_merge_is_approximately_equal`).
    #[tokio::test]
    async fn single_partition_row_shuffle_is_byte_identical() {
        let rows = sample_rows();
        let mut shuffled = rows.clone();
        shuffled.reverse();
        shuffled.swap(0, 2);
        shuffled.swap(1, 4);

        for name in ["vector_sum", "vector_mean", "vector_max"] {
            let a = run_reduce(name, &rows, 4).await;
            let b = run_reduce(name, &shuffled, 4).await;
            assert_eq!(
                a.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "{name} is not byte-identical under a single-partition row shuffle"
            );
        }
    }

    /// Split a group across two accumulators and merge them in both orders. This
    /// is what a parallel plan does, and it is the case the streaming aggregate
    /// does NOT make byte-identical: `f64` `+` is non-associative, so the two
    /// merge orders can disagree in the last bit. The guarantee is *value*
    /// equality up to `f64` rounding (a tiny tolerance), not byte-identity — the
    /// honest semantics the module doc states, and the reason a caller needing
    /// reproducibility folds over a fixed order via [`fold_vectors_in_order`].
    #[test]
    fn multi_partition_merge_is_approximately_equal() {
        // Magnitudes spread wide enough that summation order is observable in
        // f64's last bits.
        let left = vec![1e8_f32, -3.0, 2.5, 7.0];
        let mid = vec![1.0_f32, 1e-3, -1e8, 0.25];
        let right = vec![-7.5_f32, 5.0, 1e8, -2.0];

        for reduce in [VectorReduce::Sum, VectorReduce::Mean] {
            // One accumulator over all three (a fixed reference order).
            let mut whole = VectorAggAccumulator::new(reduce, 4);
            for v in [&left, &mid, &right] {
                fold_one(&mut whole, v);
            }
            let reference = finalize(&mut whole);

            // Two partitions merged in each order, simulating a parallel plan.
            let forward = merge_partitions(reduce, &[&left], &[&mid, &right]);
            let reverse = merge_partitions(reduce, &[&mid, &right], &[&left]);

            for (label, got) in [("forward", &forward), ("reverse", &reverse)] {
                for (r, g) in reference.iter().zip(got) {
                    assert!(
                        (r - g).abs() <= 1e-3 * r.abs().max(1.0),
                        "{reduce:?} {label} merge drifted past tolerance: {r} vs {g}"
                    );
                }
            }
        }
    }

    /// Fold one f32 vector into `acc` as a one-row input batch.
    fn fold_one(acc: &mut VectorAggAccumulator, vector: &[f32]) {
        let list = vector_column(&[vector.to_vec()], vector.len() as i32);
        acc.update_batch(&[list]).unwrap();
    }

    /// Finalise an accumulator to its `f32` output lanes.
    fn finalize(acc: &mut VectorAggAccumulator) -> Vec<f32> {
        match acc.evaluate().unwrap() {
            ScalarValue::FixedSizeList(list) => list
                .value(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec(),
            other => panic!("expected FixedSizeList output, got {other:?}"),
        }
    }

    /// Fold each side into its own accumulator, then merge the second into the
    /// first through the partial-state `merge_batch` path — the parallel plan's
    /// two-partition reduction.
    fn merge_partitions(reduce: VectorReduce, a: &[&Vec<f32>], b: &[&Vec<f32>]) -> Vec<f32> {
        let width = a.first().or_else(|| b.first()).map_or(0, |v| v.len());
        let mut left = VectorAggAccumulator::new(reduce, width);
        for v in a {
            fold_one(&mut left, v);
        }
        let mut right = VectorAggAccumulator::new(reduce, width);
        for v in b {
            fold_one(&mut right, v);
        }
        let state = right.state().unwrap();
        let acc_state = state[0].to_array().unwrap();
        let count_state = state[1].to_array().unwrap();
        left.merge_batch(&[acc_state, count_state]).unwrap();
        finalize(&mut left)
    }

    /// The exposed fixed-order primitive folds an ordered sequence to the same
    /// value the streaming SQL aggregate produces over the same rows (they share
    /// one lane operator) — and folding the SAME order twice is byte-identical,
    /// which is the reproducibility a fixed-order caller relies on.
    #[tokio::test]
    async fn fold_vectors_in_order_matches_udaf_and_is_fixed_order_stable() {
        let rows = sample_rows();
        for (name, reduce) in [
            ("vector_sum", VectorReduce::Sum),
            ("vector_mean", VectorReduce::Mean),
            ("vector_max", VectorReduce::Max),
        ] {
            let via_udaf = run_reduce(name, &rows, 4).await;
            let via_primitive: Vec<f32> =
                fold_vectors_in_order(rows.iter().map(Vec::as_slice), reduce, 4)
                    .iter()
                    .map(|&x| x as f32)
                    .collect();
            assert_eq!(
                via_udaf, via_primitive,
                "{name}: the shared primitive disagrees with the SQL aggregate"
            );
            // Folding the identical order again is bit-for-bit reproducible.
            let again: Vec<f32> = fold_vectors_in_order(rows.iter().map(Vec::as_slice), reduce, 4)
                .iter()
                .map(|&x| x as f32)
                .collect();
            assert_eq!(
                via_primitive
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                again.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "{name}: fixed-order fold is not byte-identical"
            );
        }
    }

    #[tokio::test]
    async fn deterministic_across_runs() {
        let rows = sample_rows();
        for name in ["vector_sum", "vector_mean", "vector_max"] {
            let first = run_reduce(name, &rows, 4).await;
            for _ in 0..4 {
                let again = run_reduce(name, &rows, 4).await;
                assert_eq!(
                    first.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    again.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "{name} is not deterministic across runs"
                );
            }
        }
    }

    #[tokio::test]
    async fn grouped_reduction_per_group() {
        let ctx = SessionContext::new();
        register_vector_agg_udafs(&ctx);
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new_fixed_size_list("v", Field::new("item", DataType::Float32, false), 2, true),
        ]));
        let groups = Arc::new(arrow::array::StringArray::from(vec!["a", "b", "a", "b"]));
        let vectors = vector_column(
            &[
                vec![1.0, 1.0],
                vec![10.0, 10.0],
                vec![3.0, 3.0],
                vec![20.0, 20.0],
            ],
            2,
        );
        let batch = RecordBatch::try_new(schema, vec![groups, vectors]).unwrap();
        ctx.register_batch("t", batch).unwrap();
        let out = ctx
            .sql("SELECT g, vector_sum(v) AS r FROM t GROUP BY g ORDER BY g")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let list = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let group_a = list
            .value(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        let group_b = list
            .value(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(group_a, vec![4.0, 4.0]);
        assert_eq!(group_b, vec![30.0, 30.0]);
    }

    #[tokio::test]
    async fn wrong_argument_type_is_planning_error() {
        let ctx = SessionContext::new();
        register_vector_agg_udafs(&ctx);
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        ctx.register_batch("t", batch).unwrap();
        let err = ctx.sql("SELECT vector_sum(x) FROM t").await.err();
        // Either planning or execution must reject a non-vector argument.
        let err = match err {
            Some(e) => e,
            None => ctx
                .sql("SELECT vector_sum(x) FROM t")
                .await
                .unwrap()
                .collect()
                .await
                .expect_err("vector_sum over Int64 must fail"),
        };
        assert!(
            err.to_string().contains("FixedSizeList"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn empty_group_is_null_vector() {
        let ctx = SessionContext::new();
        register_vector_agg_udafs(&ctx);
        let schema = Arc::new(Schema::new(vec![Field::new_fixed_size_list(
            "v",
            Field::new("item", DataType::Float32, false),
            2,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![vector_column(&[vec![1.0, 2.0]], 2)]).unwrap();
        ctx.register_batch("t", batch).unwrap();
        // The WHERE eliminates every row, so the aggregate folds an empty group.
        let out = ctx
            .sql("SELECT vector_mean(v) AS r FROM t WHERE false")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let list = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(list.len(), 1, "one output row for the single empty group");
        assert!(list.is_null(0), "an empty group reduces to a null vector");
    }
}
