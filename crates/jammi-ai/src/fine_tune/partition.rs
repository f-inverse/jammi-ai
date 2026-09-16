//! Partition rule v1 ("block-by-global-batch") and the global-batch step
//! formula every step quantity indexes by (DESIGN.md §2; PRESSURE round-2
//! design findings 5, 6).
//!
//! With per-rank batch `B` and world `W`, global batch `t` is rows
//! `[t·W·B, (t+1)·W·B)` of the train prefix; rank `r` reads
//! `[t·W·B + r·B, t·W·B + (r+1)·B)`. The union over ranks at step `t` is
//! exactly the `W=1` batch of size `W·B` at step `t` — [`PartitionSpec::
//! rows_for_step`] is the one place that arithmetic is spelled, so a caller
//! that hand-derives a rank's slice can never drift from the rule the
//! multiset oracle (`data.rs`'s `partition_rule_*` tests) checks.
//!
//! The trailing global batch is **kept**: ranks then hold unequal counts,
//! and when `train_count mod (W·B) <= r·B` rank `r` holds **zero** rows for
//! that step — a valid state (K2), never a division by zero or an
//! out-of-bounds slice. [`PartitionSpec::rows_for_step`] returns an empty
//! range rather than panicking or wrapping.

use std::ops::Range;

/// The row-partitioning rule a training set's rows are read under. `V1` is
/// the only rule this plan defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionRule {
    /// "block-by-global-batch" — see the module doc.
    BlockByGlobalBatch,
}

/// `(rank, world, batch, rule)` — the per-run partition assignment.
///
/// `batch` is the PER-RANK batch size (`FineTuneConfig::batch_size`), never
/// the global `W·B` batch.
///
/// Fields are PRIVATE: [`Self::single_rank`] (rank 0 of world 1, W=1's only
/// value) and [`Self::for_gang`] (U4b's validated arbitrary-rank
/// constructor, for a real `Local`/`Peer`/`Nccl` gang) are the only
/// constructors reachable outside this module in a release build, so every
/// `PartitionSpec` a running trainer ever holds has already cleared
/// `for_gang`'s bounds (or is the trivially-valid `single_rank`). Tests that
/// need an arbitrary `(rank, world)` assignment to exercise the partition
/// RULE itself (never a real trainer/gang) use the `#[cfg(test)]`-only
/// [`Self::for_test`] instead, which does not exist in a release build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionSpec {
    rank: usize,
    world: usize,
    batch: usize,
    rule: PartitionRule,
}

impl PartitionSpec {
    /// The single-rank, single-process assignment: rank 0 of world 1. The
    /// ONLY way to build a [`PartitionSpec`] outside this module in a
    /// release build — see the struct's own doc.
    pub fn single_rank(batch: usize, rule: PartitionRule) -> Self {
        Self {
            rank: 0,
            world: 1,
            batch,
            rule,
        }
    }

    /// Test-only constructor for an arbitrary `(rank, world)` assignment —
    /// used to exercise the partition rule itself (the multiset oracle, the
    /// K3 scaler oracle) at worlds a release build never reaches. Absent
    /// from a release build, so it can never become a second production
    /// route to a `rank != 0` / `world != 1` spec.
    #[cfg(test)]
    pub(crate) fn for_test(rank: usize, world: usize, batch: usize, rule: PartitionRule) -> Self {
        Self {
            rank,
            world,
            batch,
            rule,
        }
    }

    /// A VALIDATED arbitrary-rank constructor (#500 U2c §9 advisory): `world
    /// >= 1`, `rank < world`, `batch >= 1`, refused by name otherwise. Gated
    /// `#[cfg(any(test, feature = "test-hooks"))]`, reachable from
    /// `tests/it` only through this crate's own `test-hooks` dev-dependency
    /// (`Cargo.toml`), never from a release build. Thin wrapper over
    /// [`Self::for_gang`] — see that constructor's own doc for the bounds and
    /// for why a release build's own multi-rank route is that one, not this
    /// one. Used by the per-rank stream's `P5` oracle (`world = 2`), which
    /// needs a real, out-of-this-module `PartitionSpec` value rather than
    /// `for_test`'s `pub(crate)`-only visibility.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn for_rank(
        rank: usize,
        world: usize,
        batch: usize,
        rule: PartitionRule,
    ) -> jammi_db::error::Result<Self> {
        Self::for_gang(rank, world, batch, rule)
    }

    /// U4b's validated, PRODUCTION arbitrary-rank constructor: `world >= 1`,
    /// `rank < world`, `batch >= 1`, refused by name otherwise — a real
    /// `Local`/`Peer`/`Nccl` gang's rank builds its own
    /// [`super::trainer::RankContext`] through this, one call per rank, every
    /// rank agreeing on `world`/`batch`/`rule` and differing only in `rank`.
    ///
    /// This is the ONE place outside [`Self::single_rank`] a release build
    /// can construct a [`PartitionSpec`] with `rank != 0` or `world != 1` —
    /// every value is bounds-checked here, once, so nothing downstream (the
    /// gather-count derivation in [`Self::counts_for_step`], `rows_for_step`)
    /// ever sees an inconsistent `(rank, world)` pair.
    pub fn for_gang(
        rank: usize,
        world: usize,
        batch: usize,
        rule: PartitionRule,
    ) -> jammi_db::error::Result<Self> {
        if world == 0 {
            return Err(jammi_db::error::JammiError::FineTune(format!(
                "PartitionSpec::for_gang: world must be >= 1, got {world}"
            )));
        }
        if rank >= world {
            return Err(jammi_db::error::JammiError::FineTune(format!(
                "PartitionSpec::for_gang: rank {rank} must be < world {world}"
            )));
        }
        if batch == 0 {
            return Err(jammi_db::error::JammiError::FineTune(
                "PartitionSpec::for_gang: batch must be >= 1, got 0".into(),
            ));
        }
        Ok(Self {
            rank,
            world,
            batch,
            rule,
        })
    }

    /// This rank's index in `0..world` (`rank()`) — the trainer's own
    /// [`super::trainer::RankContext::rank`] delegates here so the two
    /// numbers can never independently drift apart.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// How many ranks this partition spans.
    pub fn world(&self) -> usize {
        self.world
    }

    /// The PER-RANK batch size this spec was built with (never the global
    /// `world * batch` batch) — `stream.rs::run_pump` reads this to derive
    /// its own `step_bound` (`batches_per_epoch(train_count, world, batch)`)
    /// identically to the trainer's `train_batches_per_epoch`, so the two
    /// can never drift apart (U4b tail).
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The row range THIS rank holds for global step `step`, over a train
    /// prefix of `train_count` rows.
    ///
    /// `world == 0` or `batch == 0` names an unrepresentable assignment (no
    /// rank could read anything); it returns an empty range rather than
    /// dividing by zero — the caller that built such a spec is the one that
    /// should have refused it (K2 names the boundary, not this leaf).
    ///
    /// Every arm is `.min(train_count)` before subtraction, so `start <=
    /// end` always holds and the range is never inverted: a rank whose slice
    /// starts at or past `train_count` gets `start == end == train_count`,
    /// i.e. an explicit ZERO-ROW state, not a panic and not a wrapped
    /// negative width.
    pub fn rows_for_step(&self, train_count: usize, step: usize) -> Range<usize> {
        if self.world == 0 || self.batch == 0 {
            return 0..0;
        }
        let global_batch = self.world * self.batch;
        let global_start = step.saturating_mul(global_batch);
        let start = global_start
            .saturating_add(self.rank * self.batch)
            .min(train_count);
        let end = global_start
            .saturating_add((self.rank + 1) * self.batch)
            .min(train_count);
        start..end.max(start)
    }

    /// The gather-count vector for global step `step` over a train prefix of
    /// `train_count` rows: `counts[r]` is the row count [`Self::rows_for_step`]
    /// would compute for rank `r` — EVERY rank's, not just `self.rank`'s.
    ///
    /// DESIGN.md §4's "Counts are derived, never exchanged": every rank knows
    /// `train_count` (the shared training set) and its own `(world, batch,
    /// rule)` (the shared spec), so every rank computes the IDENTICAL
    /// `counts` vector locally, with no round-trip — this is the pure
    /// function every [`super::collective::Collective::all_gather`] caller in
    /// the trainer derives its `counts` argument from. Reuses the exact
    /// per-rank width arithmetic [`Self::rows_for_step`] uses (for rank `r`
    /// instead of `self.rank`), so the two can never drift apart: the sum of
    /// this vector for a given `(train_count, step)` always equals the total
    /// row count [`Self::rows_for_step`]'s union over ranks would produce.
    pub fn counts_for_step(&self, train_count: usize, step: usize) -> Vec<usize> {
        (0..self.world)
            .map(|r| {
                if self.world == 0 || self.batch == 0 {
                    return 0;
                }
                let global_batch = self.world * self.batch;
                let global_start = step.saturating_mul(global_batch);
                let start = global_start.saturating_add(r * self.batch).min(train_count);
                let end = global_start
                    .saturating_add((r + 1) * self.batch)
                    .min(train_count);
                end.saturating_sub(start)
            })
            .collect()
    }
}

/// `ceil(train_count / (world * batch))` — the number of global batches (and
/// therefore of trainer-relevant steps) a train prefix takes at this
/// world/batch (DESIGN.md §2, `batches_per_epoch = ceil(train_count /
/// (W·B))`).
///
/// `0` when there is nothing to divide by (`world == 0` or `batch == 0`) or
/// nothing to divide (`train_count == 0`) — the same edge cases
/// [`super::data::TrainingDataLoader::num_batches`] already special-cases, so
/// at `world == 1` this function's value equals that method's for every
/// `batch`: the **W=1 parity oracle** this unit's acceptance (f) pins.
pub fn batches_per_epoch(train_count: usize, world: usize, batch: usize) -> usize {
    if world == 0 || batch == 0 || train_count == 0 {
        return 0;
    }
    train_count.div_ceil(world * batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (f), part 2: the formula at `world = 2` is `ceil(train_count / (2B))`
    /// — RED at base (this function does not exist there).
    #[test]
    fn batches_per_epoch_at_world_two_halves_the_w1_count() {
        for &train_count in &[0usize, 1, 2, 3, 4, 5, 7, 8, 10, 11, 100] {
            for &batch in &[1usize, 2, 3, 4] {
                let expected = if train_count == 0 {
                    0
                } else {
                    train_count.div_ceil(2 * batch)
                };
                assert_eq!(
                    batches_per_epoch(train_count, 2, batch),
                    expected,
                    "train_count={train_count} batch={batch}"
                );
            }
        }
    }

    /// (f), part 1 (the W=1 parity oracle): `batches_per_epoch(n, 1, b)` must
    /// equal `n.div_ceil(b)` for every `n`, `b` — the exact value
    /// `TrainingDataLoader::num_batches` already computes today, so wiring
    /// the trainer's step formula through this function at `world = 1`
    /// changes not one byte of any existing run.
    #[test]
    fn batches_per_epoch_at_world_one_matches_div_ceil() {
        for train_count in 0..40usize {
            for batch in 1..6usize {
                assert_eq!(
                    batches_per_epoch(train_count, 1, batch),
                    train_count.div_ceil(batch),
                    "train_count={train_count} batch={batch}"
                );
            }
        }
        // batch == 0 is the other edge `num_batches` special-cases: zero
        // batches, never a division by zero.
        assert_eq!(batches_per_epoch(10, 1, 0), 0);
        assert_eq!(batches_per_epoch(0, 1, 4), 0);
    }

    /// K2: a world or batch of zero is an unrepresentable assignment, not a
    /// panic — `rows_for_step` returns the empty range.
    #[test]
    fn rows_for_step_never_panics_on_a_degenerate_spec() {
        let degenerate_world = PartitionSpec {
            rank: 0,
            world: 0,
            batch: 4,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        assert_eq!(degenerate_world.rows_for_step(100, 0), 0..0);
        let degenerate_batch = PartitionSpec {
            rank: 0,
            world: 2,
            batch: 0,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        assert_eq!(degenerate_batch.rows_for_step(100, 3), 0..0);
    }

    /// The single-rank constructor is exactly rank 0 of world 1 under the
    /// v1 rule — what `run_spec` always builds at this commit.
    #[test]
    fn single_rank_is_rank_zero_of_world_one() {
        let spec = PartitionSpec::single_rank(8, PartitionRule::BlockByGlobalBatch);
        assert_eq!(spec.rank, 0);
        assert_eq!(spec.world, 1);
        assert_eq!(spec.batch, 8);
        assert_eq!(spec.rule, PartitionRule::BlockByGlobalBatch);
        // At world 1 every row belongs to rank 0: `rows_for_step` recovers
        // exactly the plain `[t*B, (t+1)*B)` batch, clamped to `train_count`.
        assert_eq!(spec.rows_for_step(20, 0), 0..8);
        assert_eq!(spec.rows_for_step(20, 2), 16..20);
        assert_eq!(spec.rows_for_step(20, 3), 20..20);
    }

    /// A rank whose slice starts exactly at `train_count` (not past it) is
    /// the boundary K2 names: still zero rows, still no panic.
    #[test]
    fn a_rank_starting_exactly_at_train_count_is_zero_rows_not_an_overshoot() {
        // world=2, batch=3: global batch = 6. train_count=6 -> the NEXT step
        // (t=1) starts exactly at row 6 for rank 0.
        let rank0 = PartitionSpec {
            rank: 0,
            world: 2,
            batch: 3,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        assert_eq!(rank0.rows_for_step(6, 1), 6..6);
    }

    /// U4b: `for_gang` clears its bounds, `for_rank` (its test/`test-hooks`
    /// wrapper) agrees with it byte-for-byte, and every out-of-bounds input
    /// is refused by name rather than panicking downstream.
    #[test]
    fn for_gang_validates_and_for_rank_agrees_with_it() {
        let gang = PartitionSpec::for_gang(1, 3, 4, PartitionRule::BlockByGlobalBatch).unwrap();
        let rank = PartitionSpec::for_rank(1, 3, 4, PartitionRule::BlockByGlobalBatch).unwrap();
        assert_eq!(gang, rank);
        assert_eq!(gang.rank(), 1);
        assert_eq!(gang.world(), 3);

        assert!(PartitionSpec::for_gang(0, 0, 4, PartitionRule::BlockByGlobalBatch).is_err());
        assert!(PartitionSpec::for_gang(3, 3, 4, PartitionRule::BlockByGlobalBatch).is_err());
        assert!(PartitionSpec::for_gang(0, 3, 0, PartitionRule::BlockByGlobalBatch).is_err());
    }

    /// U4b acceptance (b)/(d)'s own primitive: `counts_for_step` computes
    /// EVERY rank's row count from one rank's spec, agreeing with
    /// `rows_for_step` for each rank it stands in for, with no exchange.
    /// `train_count = 8`, `W = 2`, `B = 3`: global batch 6; step 0 is full
    /// (3, 3); step 1 is the zero-row-rank case DESIGN.md §2 names (train
    /// count 8 mod 6 = 2, so rank 1's slice `[9, 12)` clamped to `[8, 8)` is
    /// empty while rank 0's `[6, 9)` clamped to `[6, 8)` still holds 2 rows).
    #[test]
    fn counts_for_step_matches_rows_for_step_for_every_rank_including_a_zero_row_rank() {
        let rule = PartitionRule::BlockByGlobalBatch;
        let train_count = 8usize;
        for step in 0..3usize {
            let counts = PartitionSpec::for_gang(0, 2, 3, rule)
                .unwrap()
                .counts_for_step(train_count, step);
            assert_eq!(counts.len(), 2);
            for (r, &count) in counts.iter().enumerate() {
                let spec = PartitionSpec::for_gang(r, 2, 3, rule).unwrap();
                let range = spec.rows_for_step(train_count, step);
                assert_eq!(
                    count,
                    range.end - range.start,
                    "rank {r} step {step}: counts_for_step disagrees with rows_for_step"
                );
            }
        }
        // step 1 is the zero-row-rank case named above.
        let counts = PartitionSpec::for_gang(0, 2, 3, rule)
            .unwrap()
            .counts_for_step(train_count, 1);
        assert_eq!(counts, vec![2, 0]);
    }
}
