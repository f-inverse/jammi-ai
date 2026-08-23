//! A minimal iterator over per-element storage offsets for an arbitrarily
//! strided N-d `candle_core::Layout`.
//!
//! ADAPTED FROM `candle_core::strided_index::StridedIndex`
//! (candle-core 0.11.0, <https://github.com/huggingface/candle>, licensed
//! MIT OR Apache-2.0). That type's struct and its `Iterator` /
//! `ExactSizeIterator` impls are public, but both of its constructors
//! (`StridedIndex::new`, `StridedIndex::from_layout`) are `pub(crate)` in
//! candle-core, so — even though the trait docs on `CustomOp2::cpu_fwd`
//! promise "the storage can use arbitrary strides, offsets etc" — a
//! downstream crate's `CustomOp` cannot actually construct one. This file
//! reproduces that type's iteration algorithm (the same row-major
//! unravel-against-`dims` walk, offset arithmetic included) under a public
//! constructor (`StridedOffsets::from_layout`) so `Axpy`'s CPU forward can
//! honor that documented contract instead of silently requiring
//! contiguity. Per Apache License, Version 2.0 §4(b)/(c): this file is a
//! modified copy of candle-core source, its origin is stated here, and
//! candle-core's own license (MIT OR Apache-2.0) is unaffected by this
//! reuse — candle-core carries no separate per-file copyright header on
//! the original to reproduce.
//!
//! Determinism (family J): iteration order is fixed by `dims`/`stride`
//! alone — the same layout always yields the same offset sequence, which is
//! what makes the CPU fold order in `ops::axpy` reproducible.

use candle_core::Layout;

pub(crate) struct StridedOffsets<'a> {
    dims: &'a [usize],
    stride: &'a [usize],
    multi_index: Vec<usize>,
    next: Option<usize>,
    remaining: usize,
}

impl<'a> StridedOffsets<'a> {
    pub(crate) fn from_layout(layout: &'a Layout) -> Self {
        let dims = layout.dims();
        let stride = layout.stride();
        let elem_count: usize = dims.iter().product();
        // Degenerate/boundary case (family D): a zero-length dimension
        // means zero elements, regardless of start_offset — the iterator
        // yields nothing rather than one spurious offset.
        let next = if elem_count == 0 {
            None
        } else {
            Some(layout.start_offset())
        };
        Self {
            dims,
            stride,
            multi_index: vec![0; dims.len()],
            next,
            remaining: elem_count,
        }
    }
}

impl Iterator for StridedOffsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let storage_index = self.next?;
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * stride_i;
                *multi_i = 0;
            }
        }
        self.remaining -= 1;
        self.next = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedOffsets<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Layout;

    #[test]
    fn contiguous_2d_walks_row_major() {
        let layout = Layout::contiguous((2usize, 3usize));
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn transposed_view_walks_the_permuted_strides() {
        // A [2,3] contiguous buffer viewed as its [3,2] transpose: stride
        // becomes [1, 3] instead of [3, 1] — the walk must follow the
        // (permuted) strides, not assume row-major contiguity.
        let layout = Layout::new(vec![3usize, 2usize].into(), vec![1, 3], 0);
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn nonzero_start_offset_shifts_every_element() {
        let layout = Layout::contiguous_with_offset(4usize, 10);
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![10, 11, 12, 13]);
    }

    #[test]
    fn empty_dimension_yields_no_offsets() {
        let layout = Layout::contiguous((0usize, 5usize));
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert!(offsets.is_empty());
    }

    #[test]
    fn scalar_rank_zero_yields_exactly_one_offset() {
        let layout = Layout::contiguous_with_offset((), 7);
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![7]);
    }

    #[test]
    fn nonzero_offset_combined_with_non_unit_strides() {
        // dims=[3], stride=[2], start_offset=5: elements at storage
        // indices 5, 7, 9. Neither `nonzero_start_offset_shifts_every_
        // element` (offset, but stride 1) nor `transposed_view_walks_the_
        // permuted_strides` (non-unit strides, but offset 0) alone
        // exercises BOTH at once — e.g. a narrow'd-then-strided view.
        let layout = Layout::new(vec![3usize].into(), vec![2], 5);
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![5, 7, 9]);
    }

    #[test]
    fn stride_zero_broadcast_repeats_the_same_offset() {
        // candle represents a broadcast dimension with stride 0: every
        // logical position reads the SAME storage element. A walk that
        // assumed strictly-increasing offsets (or treated stride 0 as
        // "done") would either skip elements or misread this case.
        let layout = Layout::new(vec![4usize].into(), vec![0], 2);
        let offsets: Vec<usize> = StridedOffsets::from_layout(&layout).collect();
        assert_eq!(offsets, vec![2, 2, 2, 2]);
    }
}
