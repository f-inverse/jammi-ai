//! esc-076 (`.jammi/escapes.jsonl`, `esc-076-f16-eager-finetune-oom-nonmonotone-memory`)
//! fix: sequence-length BUCKETING — the pure, candle-free arithmetic half of
//! the allocator-shape-bounding mechanism.
//!
//! ## Why this lives here, not in `jammi-ai` or `jammi-encoders`
//!
//! [`bucket_seq_len`] is a total function over two `usize`s with no tensor,
//! device, or dtype dependency at all — it decides WHICH shape a batch's own
//! natural width rounds up to, before any `Vec<u32>` row or `candle_core::Tensor`
//! exists. Both `jammi-ai` (the trainer's own batch-construction seam,
//! `fine_tune::batch_bucket::pad_rows_to_bucket` — the tensor/row-padding
//! half, which stays there since it is not candle-free-independent the same
//! way: it mutates the caller's own `Vec<u32>` rows) and `jammi-encoders`
//! (this crate's own `tests/esc076_comparable_eager_control.rs` D3 leg,
//! which needs the IDENTICAL bucket decision to prove the fix bounds memory
//! at the library seam, never re-deriving its own copy) need the SAME
//! decision — `jammi-numerics` is the one crate both already depend on
//! (`jammi-ai`/`jammi-encoders`'s own `Cargo.toml`) that is candle-free, so
//! this is the natural, dependency-direction-respecting home: campaign #443
//! W2c's own boundary doctrine forbids `jammi-encoders` depending "upstream"
//! on `jammi-ai`, and `jammi-numerics` sits below both.
//!
//! ## The mechanism this closes
//!
//! `crates/jammi-encoders/tests/esc076_comparable_eager_control.rs`'s D3
//! ATTRIBUTION (campaign #443 W2c) pins the root cause precisely: `cudarc`
//! (and candle-core's own CUDA backend, which allocates through the same
//! `CudaDevice::alloc`/`alloc_zeros` primitives) has NO caching allocator —
//! every tensor is a raw `cuMemAlloc`/`cuMemFree` pair. A raw, non-pooling
//! allocator fed a training loop whose per-step tensor shapes are NOT drawn
//! from a small, fixed set fragments/grows its reserved footprint with the
//! COUNT of DISTINCT shapes it has ever been asked to satisfy, independent
//! of dtype — the ledger's own "duplicated-batch legs plateau; variable-shape
//! legs OOM" finding is exactly this. Neither `jammi-encoders`' own per-op
//! eager fallbacks nor this crate can fix the allocator itself (extend-
//! seams-not-upstream: no candle/cudarc patch); the D3 doc names the sound
//! fix point as "the trainer's own batch-construction step (padding/
//! bucketing sequence lengths to a small, fixed set of buckets)" — this
//! function, consumed by `jammi-ai::fine_tune::batch_bucket` at
//! `TrainingLoop::encode_texts`, and directly by `jammi-encoders`' own D3
//! GPU oracle to prove the bound holds at the library seam too.
//!
//! ## Design: power-of-two buckets, capped at `max_seq_length`
//!
//! [`bucket_seq_len`] rounds a batch's own natural (e.g. `tokenizers::
//! PaddingStrategy::BatchLongest`) padded width UP to the smallest bucket in
//! the ladder `{MIN_BUCKET_LEN, 2*MIN_BUCKET_LEN, 4*MIN_BUCKET_LEN, ...,
//! max_seq_length}`. Power-of-two doubling gives a BOUNDED bucket count —
//! `ceil(log2(max_seq_length / MIN_BUCKET_LEN)) + 1` buckets regardless of
//! how many distinct natural lengths a run's texts produce (for
//! `max_seq_length = 128`, `MIN_BUCKET_LEN = 8`: `{8, 16, 32, 64, 128}`, 5
//! buckets total) — matching the ledger's own "duplicated batches (a
//! SINGLE repeated shape) plateau" finding by generalizing "one shape" to "a
//! handful of shapes", which is enough to bound the allocator's distinct-size
//! count for any run length.
//!
//! `MIN_BUCKET_LEN` floors the ladder so a run of very short texts does not
//! still cycle through a `{1, 2, 4, ...}`-shaped ladder for no benefit (the
//! per-shape allocator cost this fixes is roughly constant regardless of how
//! small the shape is, so a coarser floor loses nothing while keeping the
//! total bucket count — and therefore the worst-case wasted compute over
//! padding — small).

/// The smallest bucket length in the ladder [`bucket_seq_len`] rounds up to
/// — see this module's doc for why a floor (not `{1, 2, 4, ...}`) is the
/// right ladder.
pub const MIN_BUCKET_LEN: usize = 8;

/// Rounds `natural_len` (a batch's own tokenizer-padded width, already
/// truncated to `max_seq_length` by the caller) UP to the smallest
/// power-of-two bucket `>= natural_len`, floored at [`MIN_BUCKET_LEN`] and
/// capped at `max_seq_length` — see this module's doc for the ladder and why
/// it bounds the count of DISTINCT shapes a training run ever presents to
/// the encoder.
///
/// `natural_len == 0` (an all-empty batch — never reachable from a real
/// tokenizer call, since `[CLS]`/`[SEP]`-style special tokens make every row
/// nonempty, but a defensive identity here rather than a bucket) and
/// `max_seq_length == 0` both pass through unchanged: there is no
/// well-formed bucket ladder to build over a zero-width sequence axis.
///
/// # Panics
/// Never — this is a total function over `usize`.
pub fn bucket_seq_len(natural_len: usize, max_seq_length: usize) -> usize {
    if natural_len == 0 || max_seq_length == 0 {
        return natural_len;
    }
    debug_assert!(
        natural_len <= max_seq_length,
        "bucket_seq_len's caller must already have truncated to max_seq_length \
         (natural_len={natural_len} > max_seq_length={max_seq_length})"
    );
    let mut bucket = MIN_BUCKET_LEN.min(max_seq_length);
    while bucket < natural_len && bucket < max_seq_length {
        // Saturating: `max_seq_length` is a real, bounded config value (never
        // anywhere near `usize::MAX`), so overflow here would itself be a
        // caller error this function has no better response to than capping.
        bucket = bucket.saturating_mul(2).min(max_seq_length);
    }
    // `natural_len` may still exceed the doubled ladder's own top rung only
    // when `max_seq_length` sits strictly between two powers of two AND
    // `natural_len` sits in that same gap above the ladder's last rung below
    // `max_seq_length` — capping at `max_seq_length` (already applied above)
    // is exactly the closing rung the ladder always ends on, so this is
    // unreachable, kept as a debug-only invariant rather than trusted away.
    debug_assert!(bucket >= natural_len.min(max_seq_length));
    bucket
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_seq_len_rounds_up_to_the_next_power_of_two() {
        // max_seq_length large enough that the ladder never caps early.
        assert_eq!(bucket_seq_len(1, 1024), MIN_BUCKET_LEN);
        assert_eq!(bucket_seq_len(MIN_BUCKET_LEN, 1024), MIN_BUCKET_LEN);
        assert_eq!(bucket_seq_len(MIN_BUCKET_LEN + 1, 1024), MIN_BUCKET_LEN * 2);
        assert_eq!(bucket_seq_len(9, 1024), 16);
        assert_eq!(bucket_seq_len(16, 1024), 16);
        assert_eq!(bucket_seq_len(17, 1024), 32);
        assert_eq!(bucket_seq_len(63, 1024), 64);
        assert_eq!(bucket_seq_len(64, 1024), 64);
        assert_eq!(bucket_seq_len(65, 1024), 128);
        assert_eq!(bucket_seq_len(127, 1024), 128);
        assert_eq!(bucket_seq_len(128, 1024), 128);
        assert_eq!(bucket_seq_len(129, 1024), 256);
    }

    #[test]
    fn bucket_seq_len_never_exceeds_max_seq_length() {
        // esc-076's own reporter shape: max_seq_length = 128.
        for natural in 0..=128 {
            let bucketed = bucket_seq_len(natural, 128);
            assert!(
                bucketed <= 128,
                "bucket({natural}, 128) = {bucketed} must never exceed max_seq_length"
            );
            assert!(
                bucketed >= natural,
                "bucket({natural}, 128) = {bucketed} must never truncate content"
            );
        }
    }

    #[test]
    fn bucket_seq_len_the_full_ladder_for_esc076_reporter_max_seq_length() {
        // The exact bucket SET a `max_seq_length = 128` run ever presents to
        // the encoder, over every possible natural width — this is the
        // "small, fixed set of buckets" the D3 attribution names as the fix:
        // 5 distinct shapes, never 128.
        let mut seen = std::collections::BTreeSet::new();
        for natural in 1..=128 {
            seen.insert(bucket_seq_len(natural, 128));
        }
        assert_eq!(
            seen,
            std::collections::BTreeSet::from([8, 16, 32, 64, 128]),
            "the reporter shape's bucket ladder must be exactly this bounded set"
        );
    }

    #[test]
    fn bucket_seq_len_caps_at_max_seq_length_below_a_power_of_two() {
        // max_seq_length itself need not be a power of two (e.g. a
        // model's own positional-embedding cap).
        assert_eq!(bucket_seq_len(100, 100), 100);
        assert_eq!(bucket_seq_len(65, 100), 100);
        assert_eq!(bucket_seq_len(64, 100), 64);
    }

    #[test]
    fn bucket_seq_len_degenerate_zero_inputs_pass_through() {
        assert_eq!(bucket_seq_len(0, 128), 0);
        assert_eq!(bucket_seq_len(5, 0), 5);
    }

    #[test]
    fn bucket_seq_len_small_max_seq_length_below_min_bucket_len_is_one_bucket() {
        // max_seq_length smaller than MIN_BUCKET_LEN: every natural length
        // collapses to the single bucket max_seq_length itself.
        for natural in 1..=4 {
            assert_eq!(bucket_seq_len(natural, 4), 4);
        }
    }
}
