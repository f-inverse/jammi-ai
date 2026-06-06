//! Differentiable segment-aggregate: reduce a `[N, D]` value tensor into a
//! `[num_segments, D]` tensor by folding each row into the segment its id names,
//! with full autograd support.
//!
//! This is the train-time twin of the data-plane vector-aggregation UDAF
//! (`vector_mean` / `vector_sum` / `vector_max`): one pooling semantics, two
//! execution contexts. The UDAF folds f64 accumulators over arbitrary SQL group
//! keys and serves a result table; this folds an f32 candle [`Tensor`] over
//! dense `0..num_segments` ids inside an autograd graph so the same pooling can
//! be a differentiable layer.
//!
//! # Parity with the UDAF, and the one deliberate divergence
//!
//! The two paths agree row-by-row within f32-vs-f64 rounding **except** on the
//! empty-segment convention, which cannot agree by construction: a candle
//! tensor has no per-row null, so where the UDAF emits a *null* vector for a
//! group with no rows, this emits a **zero** row. Zero is NaN-free,
//! deterministic, and differentiable (an all-empty input still backprops),
//! which a null cannot be. The divergence is intentional and is the only point
//! where the two pooling implementations differ; everything else (the lane fold,
//! permutation-invariance, mean's divide-at-finalise, max's lane-wise reduction)
//! matches.
//!
//! # Index domain
//!
//! `segment_ids` must be dense (`0..num_segments`) — mapping arbitrary keys to
//! dense ids is the caller's job and is the parity surface between this and the
//! UDAF's SQL `GROUP BY`. Ids out of `0..num_segments` are a caller error
//! surfaced by the underlying gather/scatter, not silently dropped.

use candle_core::{DType, Tensor, D};

use crate::error::EncoderError;

/// The element-wise reduction [`segment_aggregate`] applies across a segment.
///
/// Mirrors the data-plane UDAF's `VectorReduce`: one operator, three names. A
/// fourth reduction is a new arm here, not a new function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentReduce {
    /// Element-wise sum of each lane over the segment's rows.
    Sum,
    /// Element-wise arithmetic mean (sum of each lane over the count of rows).
    Mean,
    /// Element-wise maximum of each lane over the segment's rows.
    Max,
}

/// Reduce `values` (`[N, D]`, f32) into `[num_segments, D]` by folding row `n`
/// into segment `segment_ids[n]`, differentiably.
///
/// `segment_ids` is a 1-D index tensor of length `N` with dense ids in
/// `0..num_segments`. The reduction is permutation-invariant: permuting the rows
/// of `values` and `segment_ids` together leaves the output unchanged (each
/// reduction operator is commutative and associative).
///
/// An empty segment (no row carries its id) reduces to a **zero** row — the
/// documented, NaN-free divergence from the UDAF's null (see module docs). Mean
/// guards `count == 0` before dividing so an empty segment is never a `0/0`,
/// neither in the forward value nor in the backward gradient.
pub fn segment_aggregate(
    values: &Tensor,
    segment_ids: &Tensor,
    num_segments: usize,
    reduce: SegmentReduce,
) -> Result<Tensor, EncoderError> {
    let (n, d) = values.dims2()?;
    let ids_len = segment_ids.dims1()?;
    if ids_len != n {
        return Err(EncoderError::Config(format!(
            "segment_aggregate: segment_ids length {ids_len} must equal values rows {n}"
        )));
    }
    // No rows means every segment is empty, which by the documented convention
    // is all zeros. Handle it here so no reduction touches a zero-length axis
    // (Max's per-segment max over zero members has no value to fold).
    if n == 0 {
        return Ok(Tensor::zeros(
            (num_segments, d),
            values.dtype(),
            values.device(),
        )?);
    }
    // index_add / index_select want an integer index of the same length as the
    // scattered/​gathered dimension. Normalise to U32 once so the caller may pass
    // any integer dtype.
    let ids = if segment_ids.dtype() == DType::U32 {
        segment_ids.clone()
    } else {
        segment_ids.to_dtype(DType::U32)?
    };

    match reduce {
        SegmentReduce::Sum => segment_sum(values, &ids, num_segments, d),
        SegmentReduce::Mean => {
            let sum = segment_sum(values, &ids, num_segments, d)?;
            let counts = segment_counts(&ids, num_segments, values.device())?;
            // Divide each segment's lane-sum by its row count. Empty segments
            // have count 0; dividing by a guarded count of 1 keeps the row at
            // its summed value of exactly 0 (no rows folded in), so an empty
            // mean is the zero convention with no `0/0` in forward or backward.
            let safe_counts = counts.clamp(1f32, f32::MAX as f64)?;
            Ok(sum.broadcast_div(&safe_counts.reshape((num_segments, 1))?)?)
        }
        SegmentReduce::Max => segment_max(values, &ids, num_segments, d),
    }
}

/// Segment sum: scatter-add each row into its segment in a single differentiable
/// `index_add`. The gradient is the exact scatter-back (`grad.index_select`),
/// free from the op.
fn segment_sum(
    values: &Tensor,
    ids: &Tensor,
    num_segments: usize,
    d: usize,
) -> Result<Tensor, EncoderError> {
    let zeros = Tensor::zeros((num_segments, d), values.dtype(), values.device())?;
    Ok(zeros.index_add(ids, values, 0)?)
}

/// Per-segment row count as a `[num_segments]` f32 tensor, via `index_add` of a
/// column of ones. Drives the mean divisor; an empty segment counts 0.
fn segment_counts(
    ids: &Tensor,
    num_segments: usize,
    device: &candle_core::Device,
) -> Result<Tensor, EncoderError> {
    let n = ids.dims1()?;
    let zeros = Tensor::zeros((num_segments, 1), DType::F32, device)?;
    let ones = Tensor::ones((n, 1), DType::F32, device)?;
    Ok(zeros.index_add(ids, &ones, 0)?.reshape((num_segments,))?)
}

/// Segment max: tie-safe, NaN-free, differentiable lane-wise maximum per
/// segment.
///
/// The `-inf`-mask approach (push non-members to `-inf`, reduce-max) NaNs the
/// backward on an empty segment — candle's max-grad is `node.eq(arg) * grad`, so
/// an all-`-inf` column makes `node` itself `-inf` and the `eq` mask degenerate.
/// Instead we shift the values into a strictly-positive band and zero out
/// non-members, so a member always out-ranks the `0` of a non-member and the
/// max-grad routes only to real members:
///
/// 1. `shifted = values - global_min + 1` — every entry is `>= 1`, finite.
/// 2. `masked[s, n, :] = shifted[n, :]` where row `n` is in segment `s`, else
///    `0`. A non-member is exactly `0 < 1 <= shifted`, so it can never win a
///    lane's max and never ties a real member (ties stay among members).
/// 3. `seg_shifted = max over the N axis` — a real member's value for any
///    non-empty segment; `0` for an empty one.
/// 4. Shift back `+ global_min - 1`, then overwrite empty segments with `0`
///    (the documented empty convention) using the per-segment count.
///
/// The membership mask carries no gradient (it is built from ids), so the
/// backward flows only to the winning member row, exactly as a per-segment
/// reduce-max should.
fn segment_max(
    values: &Tensor,
    ids: &Tensor,
    num_segments: usize,
    d: usize,
) -> Result<Tensor, EncoderError> {
    let (n, _) = values.dims2()?;
    let device = values.device();

    // A finite global floor strictly below every value, so the shifted band is
    // >= 1 and the non-member sentinel 0 can never win a max.
    let global_min = values.min(0)?.min(0)?.to_scalar::<f32>()? as f64;
    let shifted = values.affine(1.0, 1.0 - global_min)?; // values - global_min + 1

    // Membership mask `[num_segments, N]`: row n contributes to segment s iff
    // segment_ids[n] == s. Built by scattering a column of ones, so it carries
    // no gradient.
    let membership = {
        let zeros = Tensor::zeros((num_segments, n), DType::F32, device)?;
        // Scatter ones into column-per-row: place a 1 at (segment_ids[n], n).
        // index_add along dim 0 with a one-hot-per-row source achieves this.
        let eye_rows = Tensor::eye(n, DType::F32, device)?; // [N, N], row n is e_n
        zeros.index_add(ids, &eye_rows, 0)?
    };

    // `masked[s, n, :] = membership[s, n] * shifted[n, :]`.
    let masked = membership
        .reshape((num_segments, n, 1))?
        .broadcast_mul(&shifted.reshape((1, n, d))?)?;
    let seg_shifted = masked.max(1)?; // [num_segments, D]

    // Undo the shift, then zero empty segments per the documented convention.
    let seg_max = seg_shifted.affine(1.0, global_min - 1.0)?;
    let counts = segment_counts(ids, num_segments, device)?; // [num_segments]
    let non_empty = counts
        .gt(0f32)?
        .to_dtype(DType::F32)?
        .reshape((num_segments, 1))?;
    Ok(seg_max.broadcast_mul(&non_empty)?)
}

/// L2-normalise each row of a `[num_segments, D]` (or any `[.., D]`) tensor along
/// the last axis, matching the encoders' pooled-embedding normalisation. Exposed
/// because a segment-pooled embedding is normalised the same way a token-pooled
/// one is.
pub fn l2_normalize_rows(t: &Tensor) -> Result<Tensor, EncoderError> {
    let norm = t
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX as f64)?;
    Ok(t.broadcast_div(&norm)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn dev() -> Device {
        Device::Cpu
    }

    /// `[[1,2],[3,4],[5,6]]` with ids `[0,0,1]`: segment 0 = rows 0,1; segment 1
    /// = row 2; one segment of two and one singleton.
    fn sample() -> (Tensor, Tensor) {
        let v = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev()).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 0, 1], (3,), &dev()).unwrap();
        (v, ids)
    }

    #[test]
    fn sum_folds_rows_into_segments() {
        let (v, ids) = sample();
        let out = segment_aggregate(&v, &ids, 2, SegmentReduce::Sum)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(out, vec![vec![4.0, 6.0], vec![5.0, 6.0]]);
    }

    #[test]
    fn mean_divides_by_count() {
        let (v, ids) = sample();
        let out = segment_aggregate(&v, &ids, 2, SegmentReduce::Mean)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(out, vec![vec![2.0, 3.0], vec![5.0, 6.0]]);
    }

    #[test]
    fn max_reduces_lanewise() {
        let (v, ids) = sample();
        let out = segment_aggregate(&v, &ids, 2, SegmentReduce::Max)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(out, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
    }

    /// An id range wider than any present id leaves the trailing segments empty;
    /// every reduction returns a zero row for them (the documented divergence
    /// from the UDAF's null), and never a NaN.
    #[test]
    fn empty_segment_is_zero_not_nan() {
        let (v, ids) = sample();
        for reduce in [SegmentReduce::Sum, SegmentReduce::Mean, SegmentReduce::Max] {
            let out = segment_aggregate(&v, &ids, 4, reduce)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            // Segments 2 and 3 have no members.
            assert_eq!(out[2], vec![0.0, 0.0], "{reduce:?} empty segment 2");
            assert_eq!(out[3], vec![0.0, 0.0], "{reduce:?} empty segment 3");
            assert!(
                out.iter().flatten().all(|x| !x.is_nan()),
                "{reduce:?} produced a NaN on an empty segment"
            );
        }
    }

    /// Zero input rows means every segment is empty — all reductions return the
    /// all-zero `[num_segments, D]` (the documented convention), with no
    /// reduction touching a zero-length axis.
    #[test]
    fn empty_input_is_all_zeros() {
        let v = Tensor::from_vec(Vec::<f32>::new(), (0, 2), &dev()).unwrap();
        let ids = Tensor::from_vec(Vec::<u32>::new(), (0,), &dev()).unwrap();
        for reduce in [SegmentReduce::Sum, SegmentReduce::Mean, SegmentReduce::Max] {
            let out = segment_aggregate(&v, &ids, 3, reduce)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_eq!(
                out,
                vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
                "{reduce:?} on empty input"
            );
        }
    }

    /// Permuting rows of `values` and `segment_ids` together leaves the output
    /// unchanged — the determinism the UDAF guarantees.
    #[test]
    fn permutation_invariant() {
        let (v, ids) = sample();
        // Reverse row order: rows [2,1,0], ids become [1,0,0].
        let v_perm = Tensor::from_vec(vec![5f32, 6.0, 3.0, 4.0, 1.0, 2.0], (3, 2), &dev()).unwrap();
        let ids_perm = Tensor::from_vec(vec![1u32, 0, 0], (3,), &dev()).unwrap();
        for reduce in [SegmentReduce::Sum, SegmentReduce::Mean, SegmentReduce::Max] {
            let a = segment_aggregate(&v, &ids, 2, reduce)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let b = segment_aggregate(&v_perm, &ids_perm, 2, reduce)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_eq!(a, b, "{reduce:?} not permutation-invariant");
        }
    }

    /// Max must not leak gradient to a non-winning member: with all-negative
    /// values (so the positive-shift band matters), the per-segment max routes
    /// gradient to exactly the winning row.
    #[test]
    fn max_handles_all_negative_values() {
        let v = Tensor::from_vec(vec![-5f32, -1.0, -3.0, -2.0], (2, 2), &dev()).unwrap();
        let ids = Tensor::from_vec(vec![0u32, 0], (2,), &dev()).unwrap();
        let out = segment_aggregate(&v, &ids, 1, SegmentReduce::Max)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // lane 0: max(-5, -3) = -3; lane 1: max(-1, -2) = -1.
        assert_eq!(out, vec![vec![-3.0, -1.0]]);
    }
}
