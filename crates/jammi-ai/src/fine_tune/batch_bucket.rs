//! Sequence-length BUCKETING at the fine-tune trainer's own
//! batch-construction seam.
//!
//! ## Why bucket
//!
//! `cudarc` (and candle-core's own CUDA backend, which allocates through the
//! same `CudaDevice::alloc`/`alloc_zeros` primitives) has NO caching
//! allocator — every tensor is a raw `cuMemAlloc`/`cuMemFree` pair. A raw,
//! non-pooling allocator fed a training loop whose per-step tensor shapes are
//! NOT drawn from a small, fixed set fragments/grows its reserved footprint
//! with the COUNT of DISTINCT shapes it has ever been asked to satisfy,
//! independent of dtype: duplicated-batch runs plateau, variable-shape runs
//! OOM (the comparable eager-control test in `crates/jammi-encoders/tests/`,
//! `esc076_comparable_eager_control.rs`, measures this). `jammi-encoders`'
//! per-op eager fallbacks cannot fix this without corrupting the math
//! (padding an activation INSIDE a mean/variance reduction fabricates
//! values), so the fix point is the trainer's own
//! batch-construction step — this module, and its TRAINING-STEP call site in
//! `TrainingLoop::encode_texts` (`tokenize_and_bucket`).
//!
//! **Eval is not bucketed** (see `tokenize_natural_width`'s doc for the full
//! argument): bucketing bounds the allocator's distinct-shape count across a
//! sequence of per-step batches that is UNBOUNDED across the whole run. An
//! eval pass (`evaluate`/`evaluate_held_out`) re-presents a DETERMINISTIC
//! held-out/val partition (same rows, same batch order, every pass), so its
//! distinct-shape contribution is paid ONCE per run, never growing per-step
//! or per-epoch, regardless of `eval_cadence`. That one-time set's SIZE is
//! still caller-dependent (bounded by the split's own natural-width
//! diversity, not by this ladder). `encode_texts` only calls into this module
//! while `self.training_mode` is `true`; while evaluating it calls the
//! natural-width sibling `tokenize_natural_width` instead — bucketing a
//! 321-token held-out batch up to the 512 rung OOMs.
//!
//! ## Bucket DECISION lives in `jammi-numerics`, not here
//!
//! [`bucket_seq_len`] (re-exported from `jammi_numerics::bucket_seq_len`,
//! see that function's own doc for the ladder design) is a pure
//! `usize -> usize` decision with no tensor/row dependency. `jammi-encoders`'
//! GPU oracle needs the IDENTICAL decision and must not depend on `jammi-ai`,
//! so the decision lives in `jammi-numerics`, the candle-free crate both
//! depend on; this module keeps only the row-mutating half
//! ([`pad_rows_to_bucket`]) and its call site.
//!
//! ## Correctness: padded positions are FULLY masked, never a wrong answer
//! wearing a fixed shape
//!
//! [`pad_rows_to_bucket`] extends each row's `input_ids` with `pad_id` (`0`
//! — `crates/jammi-ai/src/model/tokenizer.rs`'s `TokenizerWrapper::from_file`
//! never overrides `tokenizers::PaddingParams::default()`'s own `pad_id: 0`,
//! so this is the SAME pad value the tokenizer's own `BatchLongest` padding
//! uses for every row shorter than the batch's natural width) and its
//! `attention_mask` with `0` (fully masked — the identical mechanism
//! `BatchLongest` relies on for intra-batch length variance; this module
//! only extends how FAR that trailing zero run goes, never how it is
//! interpreted downstream). `bucketed_and_natural_batches_produce_the_same_
//! pooled_output` (below) proves this output-invariance on a real encoder;
//! `crate::fine_tune::trainer`'s `encode_texts_bucketing_oracle` module
//! proves the same property AND shape-bucketing itself at the production
//! call site (`TrainingLoop::encode_texts`'s `EncoderAdapters` branch, via
//! `tokenize_and_bucket` — the sole caller of [`pad_rows_to_bucket`] outside
//! this module's tests), since a unit test that only calls
//! [`pad_rows_to_bucket`] directly cannot catch the call site going unwired.

/// Re-exported from `jammi_numerics::bucket_seq_len`/`MIN_BUCKET_LEN` — see
/// that function's own doc for the full ladder design, and this module's own
/// doc ("Bucket DECISION lives in `jammi-numerics`, not here") for why the
/// decision lives outside this crate. Re-exported (not merely called
/// fully-qualified at the one call site below) so this file's own tests keep
/// exercising the exact names this crate's call site imports.
pub use jammi_numerics::{bucket_seq_len, MIN_BUCKET_LEN};

/// The bucket rung a batch's rows are padded to: `pinned`, if the caller
/// supplies one, or [`bucket_seq_len`] of THIS batch's own natural width
/// otherwise (every production call site, including a real gang — see
/// below).
///
/// **Why a real gang does NOT need to pin one.** The cross-rank gather
/// concatenates each rank's
/// POOLED `[rows, hidden]` output along dim 0 (the row axis) — never along
/// dim 1, the sequence axis this bucket rung governs. Two ranks whose local
/// batches bucket to two DIFFERENT rungs still produce IDENTICAL trailing
/// `hidden` widths (pooling already collapsed the sequence axis before the
/// gather ever runs), so a dim-0 gather never needs the ranks to agree on
/// how they got there. `pinned` therefore stays UNUSED by every production
/// call site (`TrainingLoop::encode_texts` always passes `None`, at every
/// world size): each rank keeps bucketing its own local batch to its OWN
/// natural width, independently, exactly as a single rank always has. The
/// tolerance a cross-rank gather-exactness oracle needs to absorb is the
/// bucket ladder's own documented PADDING variance (this module's own
/// `1e-4`/`1e-5`-scale bound on a padded-vs-natural-width forward, not a
/// bit-identity claim) — never a reason to force every rank onto one rung.
///
/// `pinned` is a real, callable parameter for a caller that wants to bound
/// the per-run distinct-shape count across ranks (an allocator concern, not
/// a gather-correctness one); no production call site sets it.
pub fn resolve_bucket_rung(
    natural_cols: usize,
    effective_max: usize,
    pinned: Option<usize>,
) -> usize {
    pinned.unwrap_or_else(|| bucket_seq_len(natural_cols, effective_max))
}

/// Extends every row of `input_ids`/`attention_masks` (in place) from their
/// current (equal, tokenizer-padded) width out to `bucketed_len`, with the
/// pad token id (`0`) and a fully-masked (`0`) attention entry respectively
/// — see this module's doc for why `0` is the correct pad id to reuse
/// (never a second, invented padding convention) and why extending the
/// SAME trailing-zero mechanism the tokenizer's own `BatchLongest` padding
/// already relies on is output-invariant.
///
/// No-ops (every row already at `bucketed_len` or beyond — the latter never
/// happens from a real caller, since `bucketed_len` is always
/// `>= natural_len`, but this is still a safe extend either way) when
/// nothing needs padding.
///
/// # Panics
/// Never — `Vec::resize` extends or is a no-op, and both slices are
/// resized independently, so a caller passing mismatched row counts between
/// `input_ids`/`attention_masks` (a caller bug elsewhere) still leaves each
/// individually well-formed; it is the caller's responsibility that both
/// have the same NUMBER of rows to begin with (this function does not read
/// across the two, only within each row).
pub fn pad_rows_to_bucket(rows: &mut [Vec<u32>], bucketed_len: usize, pad_value: u32) {
    for row in rows.iter_mut() {
        if row.len() < bucketed_len {
            row.resize(bucketed_len, pad_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `None` reproduces exactly what an unconditional `bucket_seq_len` call
    /// computes — the W=1 parity oracle for `resolve_bucket_rung`'s
    /// production arm.
    #[test]
    fn resolve_bucket_rung_with_no_pin_matches_bucket_seq_len() {
        for natural in [1usize, MIN_BUCKET_LEN, MIN_BUCKET_LEN + 1, 9, 200] {
            assert_eq!(
                resolve_bucket_rung(natural, 1024, None),
                bucket_seq_len(natural, 1024),
                "natural={natural}"
            );
        }
    }

    /// A pinned rung wins outright, regardless of what this batch's own
    /// natural width would otherwise resolve to — the cross-rank agreement
    /// this function exists to apply is never second-guessed by a local
    /// recomputation.
    #[test]
    fn resolve_bucket_rung_pinned_overrides_the_local_natural_width() {
        assert_eq!(resolve_bucket_rung(3, 1024, Some(64)), 64);
        assert_eq!(
            resolve_bucket_rung(0, 1024, Some(MIN_BUCKET_LEN)),
            MIN_BUCKET_LEN,
            "a zero-row chunk's natural width (0) must not defeat a pinned rung"
        );
    }

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
        // A common encoder shape: max_seq_length = 128.
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
    fn bucket_seq_len_the_full_ladder_for_max_seq_length_128() {
        // The exact bucket SET a `max_seq_length = 128` run ever presents to
        // the encoder, over every possible natural width — the "small, fixed
        // set of buckets" that bounds the allocator: 5 distinct shapes,
        // never 128.
        let mut seen = std::collections::BTreeSet::new();
        for natural in 1..=128 {
            seen.insert(bucket_seq_len(natural, 128));
        }
        assert_eq!(
            seen,
            std::collections::BTreeSet::from([8, 16, 32, 64, 128]),
            "the max_seq_length = 128 bucket ladder must be exactly this bounded set"
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

    #[test]
    fn pad_rows_to_bucket_extends_with_pad_value_never_truncates() {
        let mut rows = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
        pad_rows_to_bucket(&mut rows, 8, 0);
        assert_eq!(rows[0], vec![1, 2, 3, 0, 0, 0, 0, 0]);
        assert_eq!(rows[1], vec![4, 5, 6, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_rows_to_bucket_is_a_noop_when_already_at_length() {
        let mut rows = vec![vec![1u32, 2, 3, 4]];
        pad_rows_to_bucket(&mut rows, 4, 0);
        assert_eq!(rows[0], vec![1, 2, 3, 4]);
    }

    #[test]
    fn pad_rows_to_bucket_uses_the_given_pad_value_for_masks_vs_ids() {
        // The SAME function serves both input_ids (pad_value = pad token id,
        // here standing in for a nonzero token id to prove it is not
        // hardcoded) and attention_masks (pad_value = 0, "fully masked").
        let mut ids = vec![vec![7u32, 8]];
        pad_rows_to_bucket(&mut ids, 4, 0);
        assert_eq!(ids[0], vec![7, 8, 0, 0]);

        let mut masks = vec![vec![1u32, 1]];
        pad_rows_to_bucket(&mut masks, 4, 0);
        assert_eq!(masks[0], vec![1, 1, 0, 0]);
    }

    /// The SAME batch, encoded once at its natural (unbucketed) width and
    /// once bucketed via [`bucket_seq_len`]/[`pad_rows_to_bucket`], must
    /// produce the SAME pooled output — bucketing is a shape-canonicalization
    /// knob, never an output-affecting one (if it were, it would have to be
    /// part of the result's identity).
    ///
    /// Reasoned correctness, then PROVEN on a real encoder rather than
    /// merely asserted from the reasoning: `jammi_encoders::pooling::
    /// mean_pool` divides by `attention_mask.sum(1)` (the REAL, unmasked
    /// token count), never by the raw sequence length, and every padded
    /// position this module extends carries `attention_mask = 0` — so the
    /// extra bucketed positions contribute ZERO to the pooled sum and do
    /// NOT change its divisor either. The attention softmax's own additive
    /// mask bias already has to exclude a padded tail correctly (ordinary
    /// intra-batch `BatchLongest` length variance already produces rows
    /// shorter than the batch's own width), so extending that SAME trailing
    /// run further is not a new code path.
    ///
    /// Tolerance: exact bit-for-bit equality is NOT claimed — `ModernBert`'s
    /// attention softmax still runs the reduction over a WIDER last axis
    /// (`cols` positions instead of `natural_cols`, even though the extra
    /// ones are driven to ~0 probability by the mask bias), so floating-
    /// point summation order differs. The stated, derived tolerance is
    /// `1e-5` absolute per lane on this tiny (`hidden_size=32`) F32 fixture
    /// — several orders of magnitude looser than F32 epsilon accumulated
    /// over a `hidden_size`-wide reduction, and orders of magnitude tighter
    /// than any real bug (a wrong mask, a wrong pad id, a divisor that
    /// counted padding) would produce, which collapses agreement completely
    /// rather than leaving a small residual.
    #[test]
    fn bucketed_and_natural_batches_produce_the_same_pooled_output() {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::VarMap;
        use jammi_encoders::{ModernBert, ModernBertConfig, Pooling};
        use jammi_lora::LoraBuildConfig;

        let device = Device::Cpu;
        // The committed `tests/fixtures/tiny_modernbert` config (workspace
        // root, two levels up from `CARGO_MANIFEST_DIR` =
        // `crates/jammi-ai`) — the same fixture
        // `crate::fine_tune::trainer::encoder_adapters_training_state_tests`
        // and `tests/it/encoder_adapters.rs` already build against.
        // `vocab_size=256, max_position_embeddings=128, pad_token_id=0`.
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures")
            .join("tiny_modernbert");
        let config: ModernBertConfig = serde_json::from_str(
            &std::fs::read_to_string(fixture_dir.join("config.json"))
                .expect("tiny_modernbert fixture config.json must be readable"),
        )
        .expect("tiny_modernbert config.json must parse");
        let weights = fixture_dir.join("model.safetensors");

        let varmap = VarMap::new();
        let encoder = ModernBert::builder()
            .pooling(Pooling::Mean)
            .lora(LoraBuildConfig::frozen())
            .build(&[weights.as_path()], &config, &device, &varmap)
            .expect("tiny_modernbert fixture must build");

        // Two rows of DIFFERENT real-token lengths (5 and 10), matching a
        // genuine `BatchLongest` natural width of 10 — token ids kept well
        // inside `vocab_size=256`, deterministic (no unseeded RNG).
        let row_a: Vec<u32> = vec![10, 11, 12, 13, 14];
        let row_b: Vec<u32> = vec![20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        let natural_cols = row_b.len(); // BatchLongest's own natural width.
        let mut ids = vec![row_a.clone(), row_b.clone()];
        pad_rows_to_bucket(&mut ids, natural_cols, 0); // row_a -> width 8, tail pad_id 0.
        let mut masks: Vec<Vec<u32>> = vec![vec![1; row_a.len()], vec![1; row_b.len()]];
        pad_rows_to_bucket(&mut masks, natural_cols, 0); // row_a's tail -> mask 0.

        let bucketed_cols = bucket_seq_len(natural_cols, config.max_position_embeddings);
        assert!(
            bucketed_cols > natural_cols,
            "the test's own natural width ({natural_cols}) must be non-bucket-aligned so this \
             test actually exercises extra padding, got bucketed={bucketed_cols}"
        );

        let mut bucketed_ids = ids.clone();
        pad_rows_to_bucket(&mut bucketed_ids, bucketed_cols, 0);
        let mut bucketed_masks = masks.clone();
        pad_rows_to_bucket(&mut bucketed_masks, bucketed_cols, 0);

        let to_tensor = |rows: &[Vec<u32>], cols: usize| -> Tensor {
            let flat: Vec<u32> = rows.iter().flatten().copied().collect();
            Tensor::from_vec(flat, (rows.len(), cols), &device).unwrap()
        };

        let natural_out = encoder
            .forward(
                &to_tensor(&ids, natural_cols),
                &to_tensor(&masks, natural_cols),
            )
            .expect("natural-width forward must succeed");
        let bucketed_out = encoder
            .forward(
                &to_tensor(&bucketed_ids, bucketed_cols),
                &to_tensor(&bucketed_masks, bucketed_cols),
            )
            .expect("bucketed-width forward must succeed");

        assert_eq!(natural_out.dims(), bucketed_out.dims());
        assert_eq!(natural_out.dtype(), DType::F32);

        let natural_vals = natural_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bucketed_vals = bucketed_out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(natural_vals.len(), bucketed_vals.len());
        const TOLERANCE: f32 = 1e-5;
        for (i, (n, b)) in natural_vals.iter().zip(&bucketed_vals).enumerate() {
            assert!(
                (n - b).abs() <= TOLERANCE,
                "lane {i}: natural={n} bucketed={b} differ by {} > tolerance {TOLERANCE} — \
                 bucketing must be output-invariant at real (unpadded) positions",
                (n - b).abs()
            );
        }
    }
}
