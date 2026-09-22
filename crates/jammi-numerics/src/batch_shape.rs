//! The shape of a forward: which rows share one, and how wide it is padded.
//!
//! A row has a COST — its length along the axis a forward pads: tokens for
//! text, one for a fixed-shape input. A forward over `n` rows runs at
//! `n × width`, where `width` is the longest row's cost rounded up a
//! [`ShapeLadder`]. Two pure decisions follow, and both live here, below
//! every crate that forwards a tensor, so serving and training make the
//! identical decision:
//!
//! * [`ShapeLadder::width`] — the padded width of a set of rows. Padding
//!   past the natural width is output-invariant (every padded position is
//!   fully masked) and bounds the COUNT of distinct shapes a run presents to
//!   the device. `cudarc` has no caching allocator: every tensor is a raw
//!   `cuMemAlloc`/`cuMemFree`, and a run whose shapes are not drawn from a
//!   small fixed set fragments the device's reserved footprint with the
//!   number of distinct sizes it has ever been asked for
//!   (`crates/jammi-encoders/tests/eager_training_memory.rs` measures it).
//! * [`ChunkCutter`] — the rows that share one forward, cut by one pass over
//!   rows in some order under a [`ChunkBudget`]: at most `rows` of them, and
//!   `n × width ≤ tokens`. A budget in padded tokens is what bounds a
//!   forward's activation memory; a fixed row count under-fills the device
//!   on short rows and over-fills it on long ones.
//!
//! The waste a forward carries is `n × width − Σ cost`. Rows ordered by cost
//! before the cut make every chunk's rows nearly equal in length, so the
//! waste is the ladder's rounding alone.
//!
//! # References
//!
//! * fairseq's `--max-tokens` (a batch is bounded by its padded token
//!   count, with `--max-sentences` as the row cap) and its length-sorted
//!   batching: Ott et al., "fairseq: A Fast, Extensible Toolkit for Sequence
//!   Modeling", NAACL 2019 demo, §3; `fairseq/data/data_utils.py`
//!   (`batch_by_size`).
//! * sentence-transformers' `encode()` sorts sentences by length before
//!   batching so each batch pads to near-equal lengths
//!   (`SentenceTransformer.encode`, `length_sorted_idx`).
//! * NVIDIA, "Matrix Multiplication Background User's Guide" and "Tips for
//!   Optimizing GPU Performance Using Tensor Cores": tensor-core GEMMs run
//!   at full rate when the dimensions are multiples of 8 (FP16) — the
//!   ladder's alignment, and the convention Hugging Face's
//!   `pad_to_multiple_of` encodes
//!   (<https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>).
//! * PyTorch, "CUDA semantics — Memory management", the caching allocator's
//!   `roundup_power2_divisions`: a size is rounded up to the nearest of `N`
//!   divisions of its power-of-two interval ("1200 with 4 divisions … rounds
//!   to 1280"), bounding the number of distinct sizes without paying a 2×
//!   round-up (<https://docs.pytorch.org/docs/main/notes/cuda.html>) — the
//!   ladder's rungs.
//! * Continuous batching (Yu et al., "Orca", OSDI 2022; vLLM) re-forms the
//!   batch at every decode step from a live request queue. It answers a
//!   different question — latency under arrival — and does not apply to an
//!   offline encode over a known input set, where the whole set is available
//!   to order before the first forward.

use std::num::NonZeroUsize;

/// The alignment every rung is a multiple of, and the ladder's first rung.
/// Eight is the dimension multiple tensor-core GEMMs run at full rate on.
pub const RUNG_ALIGNMENT: usize = 8;

/// The ladder's rungs per octave, `N`: between `2^k` and `2^(k+1)` the rungs
/// are `2^k · (1 + j/N)`. A width is therefore padded by less than `1/N` of
/// itself (12.5%) once an octave's divisions are [`RUNG_ALIGNMENT`] apart —
/// from 64 up — and by under eight tokens below that, where the rungs are
/// the multiples of eight. The ladder has 16 rungs up to a limit of 128, 32
/// up to 512 and 64 up to 8192.
pub const RUNGS_PER_OCTAVE: usize = 8;

/// The set of padded widths a forward may run at, capped at the model's own
/// sequence limit: one ladder for every forward — serving, a training step
/// and an evaluation pass alike. See the module doc.
///
/// The ladder has two jobs and is sized by both: the COUNT of its rungs
/// bounds the distinct shapes a run presents to a non-caching allocator,
/// and the RATIO between neighbouring rungs bounds the padding a batch pays
/// for that. Powers of two alone hold the count lowest and pad a batch by
/// up to 2×; every multiple of eight pads by at most seven tokens and
/// presents `limit / 8` shapes. Power-of-two DIVISIONS hold both: relative
/// waste under `1 / RUNGS_PER_OCTAVE`, a rung count logarithmic in the
/// limit, in exact integer arithmetic — the rounding PyTorch's CUDA caching
/// allocator applies to allocation sizes for the same reason
/// (`roundup_power2_divisions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeLadder {
    max: usize,
}

impl ShapeLadder {
    /// The ladder up to `max`, the model's sequence limit.
    pub fn new(max: usize) -> Self {
        Self { max }
    }

    /// The one width of a fixed-shape input — an image resized to the
    /// tower's size, a clip cut to the front end's window — where every row
    /// costs one and nothing is padded.
    pub fn fixed() -> Self {
        Self::new(1)
    }

    /// The width rows of natural width `natural` are padded to: the smallest
    /// rung at or above it, never above `max`. A `natural` of zero (no rows)
    /// or a `max` of zero (no sequence axis) passes through unchanged.
    pub fn width(&self, natural: usize) -> usize {
        if natural == 0 || self.max == 0 {
            return natural;
        }
        let natural = natural.clamp(RUNG_ALIGNMENT, self.max.max(RUNG_ALIGNMENT));
        // The octave `[2^k, 2^(k+1))` holding `natural`, cut into
        // `RUNGS_PER_OCTAVE` divisions; the division at or above `natural`,
        // brought up to the alignment where an octave's divisions are finer
        // than it.
        let octave = 1usize << natural.ilog2();
        let division = (octave / RUNGS_PER_OCTAVE).max(1);
        let rung = octave + (natural - octave).div_ceil(division) * division;
        rung.div_ceil(RUNG_ALIGNMENT)
            .saturating_mul(RUNG_ALIGNMENT)
            .min(self.max)
    }
}

/// What bounds one forward: at most `rows` rows, at most `tokens` padded
/// tokens (`rows in the chunk × the chunk's padded width`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkBudget {
    /// The row cap.
    pub rows: NonZeroUsize,
    /// The padded-token cap.
    pub tokens: NonZeroUsize,
}

/// The chunk being filled.
#[derive(Debug, Clone, Copy)]
struct OpenChunk {
    id: u64,
    rows: usize,
    longest: u32,
}

/// Cuts a sequence of row costs into forward chunks under a [`ChunkBudget`],
/// padded on a [`ShapeLadder`]: one sequential pass, each row joining the
/// open chunk unless that would exceed the budget, in which case it opens
/// the next. A row alone always forms a chunk, however long: the budget
/// bounds a chunk, never refuses a row.
///
/// The chunk ids are a function of the cost sequence alone, so the same rows
/// in the same order cut into the same chunks however they are batched.
#[derive(Debug, Clone)]
pub struct ChunkCutter {
    budget: ChunkBudget,
    ladder: ShapeLadder,
    open: Option<OpenChunk>,
}

impl ChunkCutter {
    /// A cutter under `budget`, padding on `ladder`.
    pub fn new(budget: ChunkBudget, ladder: ShapeLadder) -> Self {
        Self {
            budget,
            ladder,
            open: None,
        }
    }

    /// The chunk id of the next row, whose cost is `cost`.
    pub fn push(&mut self, cost: u32) -> u64 {
        let next = match self.open {
            Some(open) if self.fits(open, cost) => OpenChunk {
                id: open.id,
                rows: open.rows + 1,
                longest: open.longest.max(cost),
            },
            Some(open) => OpenChunk {
                id: open.id + 1,
                rows: 1,
                longest: cost,
            },
            None => OpenChunk {
                id: 0,
                rows: 1,
                longest: cost,
            },
        };
        self.open = Some(next);
        next.id
    }

    /// Whether `open` can take one more row of cost `cost` within the budget.
    fn fits(&self, open: OpenChunk, cost: u32) -> bool {
        let rows = open.rows + 1;
        let width = self.ladder.width(open.longest.max(cost) as usize);
        rows <= self.budget.rows.get()
            && rows
                .checked_mul(width)
                .is_some_and(|padded| padded <= self.budget.tokens.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn rungs(max: usize) -> Vec<usize> {
        let ladder = ShapeLadder::new(max);
        (1..=max)
            .map(|natural| ladder.width(natural))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The ladder itself, to a limit of 512: every multiple of eight up to
    /// 128, then eight divisions of each octave, ending on the limit.
    #[test]
    fn the_ladder_is_power_of_two_divisions_aligned_to_eight() {
        let expected: Vec<usize> = (1..=16)
            .map(|j| j * 8)
            .chain((1..=8).map(|j| 128 + j * 16))
            .chain((1..=8).map(|j| 256 + j * 32))
            .collect();
        assert_eq!(rungs(512), expected);
        assert_eq!(rungs(128).len(), 16);
        assert_eq!(rungs(512).len(), 32);
        assert_eq!(rungs(8192).len(), 64);
        let ladder = ShapeLadder::new(8192);
        assert_eq!(ladder.width(1200), 1280);
        assert_eq!(ladder.width(289), 320);
        assert_eq!(ladder.width(1), 8);
        assert_eq!(ladder.width(9), 16);
    }

    /// Both bounds at once, over every natural width to 8192: every width is
    /// a multiple of eight, never truncates, and pads by under eight tokens
    /// or an eighth of the width, whichever is larger.
    #[test]
    fn a_width_is_aligned_and_within_the_padding_tolerance() {
        let ladder = ShapeLadder::new(8192);
        for natural in 1..=8192usize {
            let width = ladder.width(natural);
            assert!(width.is_multiple_of(RUNG_ALIGNMENT), "{natural} -> {width}");
            assert!(width >= natural, "{natural} -> {width}");
            assert!(
                width - natural < (natural / RUNGS_PER_OCTAVE).max(RUNG_ALIGNMENT),
                "{natural} -> {width}"
            );
        }
        assert_eq!(ladder.width(9000), 8192);
    }

    #[test]
    fn a_ladder_caps_at_the_model_limit_and_never_truncates() {
        let ladder = ShapeLadder::new(100);
        assert_eq!(ladder.width(100), 100);
        assert_eq!(ladder.width(97), 100);
        assert_eq!(ladder.width(64), 64);
        assert_eq!(rungs(100).last(), Some(&100));
        for natural in 1..=4 {
            assert_eq!(ShapeLadder::new(4).width(natural), 4);
        }
    }

    #[test]
    fn the_fixed_ladder_has_one_width() {
        let ladder = ShapeLadder::fixed();
        assert_eq!(rungs(1), vec![1]);
        assert_eq!(ladder.width(1), 1);
        assert_eq!(ladder.width(7), 1);
        assert_eq!(ladder.width(0), 0);
    }

    #[test]
    fn degenerate_zero_inputs_pass_through() {
        assert_eq!(ShapeLadder::new(128).width(0), 0);
        assert_eq!(ShapeLadder::new(0).width(5), 5);
    }

    fn cut(costs: &[u32], rows: usize, tokens: usize) -> Vec<u64> {
        let mut cutter = ChunkCutter::new(
            ChunkBudget {
                rows: nz(rows),
                tokens: nz(tokens),
            },
            ShapeLadder::new(512),
        );
        costs.iter().map(|&c| cutter.push(c)).collect()
    }

    /// The token budget closes a chunk when one more row would push
    /// `rows × padded width` over it; the row cap closes it regardless of
    /// width.
    #[test]
    fn the_budget_closes_a_chunk_by_tokens_or_by_rows() {
        // Width 8 throughout: 64 / 8 = 8 rows per chunk.
        assert_eq!(cut(&[3; 20], 100, 64), {
            let mut ids = vec![0; 8];
            ids.extend([1; 8]);
            ids.extend([2; 4]);
            ids
        });
        // The row cap of 3 binds first.
        assert_eq!(cut(&[3; 7], 3, 64), vec![0, 0, 0, 1, 1, 1, 2]);
        // Rising costs: the padded width is the longest row's, on the
        // ladder. 8 → 8, then 20 pads to 24: 2 × 24 = 48 ≤ 64, 3 × 24 = 72 > 64.
        assert_eq!(cut(&[8, 20, 20, 20], 100, 64), vec![0, 0, 1, 1]);
    }

    /// A row longer than the whole budget still forms a chunk of its own;
    /// the budget never refuses a row.
    #[test]
    fn a_row_over_the_budget_is_a_chunk_of_one() {
        assert_eq!(cut(&[500, 500, 3, 3], 100, 64), vec![0, 1, 2, 2]);
    }

    /// The chunk sequence is a function of the cost sequence alone.
    #[test]
    fn the_cut_is_the_same_however_the_rows_are_pushed() {
        let costs: Vec<u32> = (0..500).map(|i| (i * 37 % 120) as u32 + 3).collect();
        let whole = cut(&costs, 32, 4096);
        let mut cutter = ChunkCutter::new(
            ChunkBudget {
                rows: nz(32),
                tokens: nz(4096),
            },
            ShapeLadder::new(512),
        );
        let mut resumed = Vec::new();
        for part in costs.chunks(7) {
            resumed.extend(part.iter().map(|&c| cutter.push(c)));
        }
        assert_eq!(resumed, whole);
        assert!(whole.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1));
    }

    /// Cost-ordered rows waste only the ladder's rounding: the padded token
    /// total over a spread of lengths sits within the ladder's tolerance of
    /// the real total, where the same rows in a scrambled order pad to far
    /// more.
    #[test]
    fn cost_ordered_rows_pad_to_near_their_real_length() {
        let mut costs: Vec<u32> = (0..4096).map(|i| (i * 7919 % 118) as u32 + 5).collect();
        let real: usize = costs.iter().map(|&c| c as usize).sum();
        let padded = |costs: &[u32]| -> usize {
            let ladder = ShapeLadder::new(128);
            let ids = cut(costs, 32, 4096);
            let mut total = 0;
            let mut start = 0;
            while start < ids.len() {
                let end = start
                    + ids[start..]
                        .iter()
                        .take_while(|&&id| id == ids[start])
                        .count();
                let longest = *costs[start..end].iter().max().unwrap() as usize;
                total += (end - start) * ladder.width(longest);
                start = end;
            }
            total
        };
        let scrambled = padded(&costs) as f64 / real as f64;
        costs.sort_unstable();
        let ordered = padded(&costs) as f64 / real as f64;
        assert!(scrambled > 1.5, "scrambled order pads {scrambled:.2}x");
        assert!(ordered < 1.15, "cost order pads {ordered:.2}x");
    }
}
