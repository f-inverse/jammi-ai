//! Pooled-embedding batch-composition invariance oracle (unit 62, contract
//! `docs/plans/62-embedding-surface/CONTRACT.md` §E4 / plan v2 reshape 5;
//! G4 in `PLAN.md`'s five-gap census).
//!
//! **The property.** The same row, encoded ALONE (`batch=1`, no padding)
//! vs encoded as one row of a PADDED BATCH (other rows of different
//! lengths sharing the call), must produce the same pooled embedding on
//! the real (non-pad) positions -- the whole reason a serving path is
//! allowed to batch requests of different lengths together at all.
//!
//! **Why "near-exact", not exact.** `ModernBertAttention`'s masked-out
//! (pad) key positions are biased by `MASKED_LOGIT` (a large negative
//! finite constant) before softmax; at `f32` accumulation that pad weight
//! underflows to EXACTLY `0.0`, so a pad column's contribution to the
//! value-weighted sum is `0.0 * finite = 0.0` exactly, not merely small.
//! Mathematically, therefore, batching changes nothing. At the FLOAT
//! level it still can: `x + 0.0 == x` exactly for any one addition, but
//! the underlying GEMMs that compute these sums pick their
//! blocking/accumulation order from the OPERAND SHAPE (total batch,
//! total sequence length), not from which entries happen to be exact
//! zeros -- so a batch-of-16 GEMM is not guaranteed to fold its
//! nonzero terms in the SAME order as a batch-of-1 GEMM, even though
//! both compute the same real-valued sum (family J: float addition is
//! non-associative). This is the SAME finding
//! `tests/it/modernbert.rs`'s `padded_training_loss_and_lora_grads_match_unpadded_rows_run_individually_f32_cpu`
//! documents for hidden states/gradients; this file is its POOLED-OUTPUT,
//! EVAL-PATH (not training-loss) analogue, scoped to `crates/jammi-encoders/tests/`
//! per the contract's `files_in_scope` (this file does not touch
//! `src/modernbert.rs`, so it re-derives its own small helpers rather than
//! reusing that file's private `fold_order_bound`/`FOLD_ORDER_*_ULP`).
//!
//! **Per reachable arm, not per fused kernel.** `attention_block_fused`/
//! `attention_block_flash` dispatch ONLY under `self.training == true`
//! (contract v4 §2, `modernbert.rs:2543`) -- the encode/serving surface
//! this oracle exercises never reaches them, and the unit's own Frame
//! forbids a forced-arm encode A/B (`ForcedFlash` stays private, B1) and
//! any dispatch-counter assertion here (fused arms are training-only by
//! design). "Per reachable arm" therefore means the two device/dtype
//! paths encode/serving actually uses: eager `f32` on CPU (this file's
//! CPU-hermetic legs) and eager `bf16` on CUDA (the `cuda`-gated legs,
//! capability-gated the same way the M1b family gates --
//! `Device::new_cuda(0)`, `JAMMI_REQUIRE_CUDA` opt-in-panic, silent skip
//! otherwise). No new `gpu_capability` module is added by this file (all
//! CUDA-gated tests live directly in this file, per-item `#[cfg(feature =
//! "cuda")]`, mirroring `src/modernbert.rs`'s own convention), so there is
//! no `gpu-parity-cell` marker to carry here -- stated no-marker reason.
//!
//! **Anchored to f32 truth.** The CPU legs run the encoder entirely in
//! `f32` (candle's CPU backend has no `bf16` GEMM arm at all -- see
//! `esc-046`'s resolution note) -- there is no `bf16` rounding noise to
//! separate from reduction-order noise, so the CPU comparison IS the
//! f32-truth-anchored comparison, with nothing further to subtract. The
//! CUDA legs additionally compute an `f32`-CPU truth for the SAME
//! composition and require each `bf16` arm to track it, so a real
//! dtype-rounding regression cannot hide inside an alone-vs-batch
//! agreement that happens to cancel.
//!
//! **Floor discipline (guide checklist rule 8 / esc-045's own null-band
//! control (a)): a bound must never be invented, only measured.** The
//! CPU bound below is a REAL measurement on this development box (macOS,
//! this crate's own `tiny_modernbert_head64` fixture, batch=16/seq=64,
//! printed by the measurement this doc cites) -- not a guess -- but it
//! is landed PROVISIONAL: `tests/it/modernbert.rs`'s own finding is that
//! CPU BLAS microkernel/blocking selection by total-M is
//! architecture-dependent (macOS's `gemm` happened not to depend on `M`
//! at ITS tiny shape; a Linux pod's did), so THIS oracle's own
//! same-composition floor at ITS shape must still be re-measured on the
//! CI-representative Linux runner/pod train (contract E4, Step 5) before
//! this constant can be treated as final -- it may need tightening or
//! (if Linux shows a materially larger floor) loosening, but never by
//! guesswork; only by a fresh measurement replacing this one. The `bf16`
//! CUDA bound is UNMEASURED by this agent (no CUDA device in this
//! environment) and is marked PROVISIONAL with an explicit placeholder;
//! the pod train (Step 5) MUST replace it with a measured value before
//! this leg gates anything, exactly as `FLASH_ORACLE_PADDED_BOUND`
//! (`src/modernbert.rs`) was derived from a real 8-seed pod harvest
//! rather than invented.
//!
//! **Conjunctive red controls.** Two independent mutants -- a
//! `row_lengths` off-by-one (a batch construction bug: one row's real
//! length disagrees between the alone and batch legs) and a sliding-
//! window radius off-by-one (`local_attention` shifted by 2, i.e.
//! `half_window` shifted by 1) -- must EACH separate above the measured
//! floor for this oracle to be admissible; per the contract, if either
//! cannot separate the oracle is reshaped, never tuned to pass. The
//! window control's fixture asserts `segment_len >= half_window + 2`
//! IN-TEST for both the original and mutant config (a windowed-attention
//! control the contract requires be non-vacuous by construction, not by
//! doc claim), matching `half_window()`'s own "a local layer's query at
//! position `i` attends keys `j` with `|i-j| <= half_window`" contract:
//! a segment shorter than `half_window + 2` can never have its window
//! actually truncate anything (every position sees the whole segment
//! regardless of the radius), so a shorter fixture would pass this
//! control VACUOUSLY.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_encoders::modernbert::ModernBertConfig;
use jammi_encoders::{ModernBert, Pooling};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_modernbert_head64")
}

fn load_config() -> ModernBertConfig {
    let raw = std::fs::read_to_string(fixture_dir().join("config.json"))
        .expect("read tiny_modernbert_head64 config");
    serde_json::from_str(&raw).expect("parse ModernBertConfig")
}

fn build_encoder(device: &Device, dtype: DType, config: &ModernBertConfig) -> ModernBert {
    let varmap = VarMap::new();
    let weights = fixture_dir().join("model.safetensors");
    ModernBert::builder()
        .pooling(Pooling::Mean)
        .backbone_dtype(dtype)
        .build(&[weights.as_path()], config, device, &varmap)
        .expect("build ModernBert from tiny_modernbert_head64")
}

/// The shared padded-batch fixture: 16 rows of `tiny_modernbert_head64`
/// (`vocab_size=32`) with lengths sweeping `3..=63` (max
/// `max_position_embeddings=64`), deterministic token content (no RNG --
/// this is an eager, non-random encode path; the model has fixed weights
/// and no LoRA, so a seed sweep the way the M1b flash-arm oracle needs
/// (its LoRA init varies by seed) has no analogue here). Row 0 (length 3)
/// stays BELOW the window-binding threshold on purpose (a degenerate,
/// fully-visible case); most other rows sit above it.
struct Fixture {
    batch: usize,
    seq: usize,
    rows: Vec<Vec<u32>>,
    lengths: Vec<usize>,
    ids_padded: Tensor,
    mask_padded: Tensor,
}

fn build_fixture(device: &Device) -> Fixture {
    let batch = 16usize;
    let seq = 64usize;
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(batch);
    let mut lengths: Vec<usize> = Vec::with_capacity(batch);
    for b in 0..batch {
        let len = 3 + (b * 4) % 62; // 3, 7, 11, ..., 63
        let row: Vec<u32> = (0..len)
            .map(|i| ((b * 7 + i * 3 + 1) % 30 + 1) as u32)
            .collect();
        lengths.push(row.len());
        rows.push(row);
    }

    let mut ids_padded = vec![0u32; batch * seq];
    let mut mask_padded = vec![0u32; batch * seq];
    for (b, row) in rows.iter().enumerate() {
        for (i, &t) in row.iter().enumerate() {
            ids_padded[b * seq + i] = t;
            mask_padded[b * seq + i] = 1;
        }
    }
    let ids_padded = Tensor::from_vec(ids_padded, (batch, seq), device).unwrap();
    let mask_padded = Tensor::from_vec(mask_padded, (batch, seq), device).unwrap();

    Fixture {
        batch,
        seq,
        rows,
        lengths,
        ids_padded,
        mask_padded,
    }
}

fn pooled_alone(encoder: &ModernBert, device: &Device, row: &[u32]) -> Vec<f32> {
    let len = row.len();
    let ids = Tensor::from_vec(row.to_vec(), (1, len), device).unwrap();
    let mask = Tensor::from_vec(vec![1u32; len], (1, len), device).unwrap();
    let pooled = encoder.forward(&ids, &mask).expect("alone forward");
    pooled.flatten_all().unwrap().to_vec1().unwrap()
}

fn assert_all_finite(v: &[f32], what: &str) {
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "{what}: {bad} non-finite elements");
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// PROVISIONAL, MEASURED (macOS dev box, this file's `build_fixture`,
/// `f32` CPU): the observed per-row max-abs pooled-embedding delta
/// between the alone and padded-batch legs peaked at `4.4703484e-8`
/// (`0.375 * f32::EPSILON`) across all 16 rows (row 12's per-element
/// RELATIVE error spiked to `1.54e-4` only because that specific element
/// is itself near-cancelling; the ABSOLUTE delta there was `1.86e-8`,
/// still tiny -- this bound is stated in absolute terms because the
/// compared quantity is a pooled-and-L2-normalised embedding, bounded to
/// `[-1, 1]` per element by construction, so an absolute bound is
/// meaningful and a relative one is not, near cancellation). `8.0 *
/// f32::EPSILON` (`9.5367432e-7`) gives ~21x headroom over the measured
/// worst case -- enough to absorb this file's own CPU-vs-CPU
/// architecture caveat (`tests/it/modernbert.rs`'s documented finding
/// that Linux CPU BLAS blocking-by-total-M can differ from macOS's) --
/// while still separating comfortably below both red controls' measured
/// margins (row-length off-by-one: `1.6e-2`, ~17,000x above this bound;
/// window radius off-by-one: `9.5e-6`, ~10x above this bound). NOT YET
/// measured on the CI-representative Linux runner or the pod train
/// (contract E4, Step 5) -- that measurement must replace this one, not
/// merely confirm it, before this bound is treated as final.
const PROVISIONAL_CPU_FLOOR: f32 = 8.0 * f32::EPSILON;

/// Separation multiple a red control must clear over
/// [`PROVISIONAL_CPU_FLOOR`] to count as having discriminating power.
/// `5.0` is chosen BELOW the smaller of the two controls' own measured
/// margins (window radius, ~10x) so a modest re-measurement on a
/// different CPU architecture cannot flip an admissible control to an
/// inadmissible one by a hair -- see [`PROVISIONAL_CPU_FLOOR`]'s doc for
/// both measured margins.
const RED_CONTROL_SEPARATION_MULTIPLE: f32 = 5.0;

/// The oracle: every row's pooled embedding, computed alone, must match
/// that SAME row's slot in the padded-batch forward within
/// [`PROVISIONAL_CPU_FLOOR`] -- `f32`, CPU, no forced arm (the plain
/// eager encode path, matching what `ModernBert::forward` actually does
/// in eval mode).
#[test]
fn pooled_embedding_alone_matches_padded_batch_real_row_f32_cpu() {
    let device = Device::Cpu;
    let config = load_config();
    let fixture = build_fixture(&device);
    // Checked premise: at least one row must bind the sliding window, or
    // this oracle never actually exercises `ModernBertAttention`'s local
    // layer's masking band -- see this file's own module doc.
    let window_threshold = config.half_window() + 2;
    assert!(
        fixture.lengths.iter().any(|&l| l >= window_threshold),
        "checked premise: the fixture must include a row >= half_window + 2 ({window_threshold}) \
         for the sliding window to bind anything"
    );

    let encoder = build_encoder(&device, DType::F32, &config);
    let pooled_batch = encoder
        .forward(&fixture.ids_padded, &fixture.mask_padded)
        .expect("padded batch forward");
    let pooled_batch: Vec<Vec<f32>> = pooled_batch.to_vec2().unwrap();
    assert_eq!(pooled_batch.len(), fixture.batch);

    for (b, row) in fixture.rows.iter().enumerate() {
        let alone = pooled_alone(&encoder, &device, row);
        assert_all_finite(&alone, &format!("row {b} alone"));
        assert_all_finite(&pooled_batch[b], &format!("row {b} in padded batch"));
        let diff = max_abs_diff(&alone, &pooled_batch[b]);
        assert!(
            diff.is_finite() && diff <= PROVISIONAL_CPU_FLOOR,
            "row {b} (len={}): alone vs padded-batch pooled embedding differs by {diff:e}, \
             exceeding the PROVISIONAL floor {PROVISIONAL_CPU_FLOOR:e} -- see that constant's \
             own doc for its derivation and pod-train follow-up",
            fixture.lengths[b],
        );
    }
}

/// RED control 1 (row_lengths off-by-one): the padded batch's row 0
/// mask is truncated by one real token relative to the alone reference
/// (which keeps the full, correct length) -- a batch-construction bug
/// class (the alone and batch legs disagree about how many tokens of
/// row 0 are real). This is NOT reduction-order noise: it changes which
/// tokens the mean-pool actually averages, so the divergence must sit
/// far above [`PROVISIONAL_CPU_FLOOR`] or the oracle's comparison has no
/// power to catch this class of defect (family F: a non-vacuous negative
/// control).
#[test]
fn pooled_embedding_red_control_row_length_off_by_one_f32_cpu() {
    let device = Device::Cpu;
    let config = load_config();
    let encoder = build_encoder(&device, DType::F32, &config);
    let fixture = build_fixture(&device);

    let mut mask_mut = vec![0u32; fixture.batch * fixture.seq];
    for (b, row) in fixture.rows.iter().enumerate() {
        let take = if b == 0 { row.len() - 1 } else { row.len() };
        for i in 0..take {
            mask_mut[b * fixture.seq + i] = 1;
        }
    }
    let mask_mut = Tensor::from_vec(mask_mut, (fixture.batch, fixture.seq), &device).unwrap();
    let pooled_batch_mut = encoder
        .forward(&fixture.ids_padded, &mask_mut)
        .expect("mutated-length padded batch forward");
    let pooled_batch_mut: Vec<Vec<f32>> = pooled_batch_mut.to_vec2().unwrap();

    let alone0 = pooled_alone(&encoder, &device, &fixture.rows[0]);
    let diff = max_abs_diff(&alone0, &pooled_batch_mut[0]);
    assert!(
        diff.is_finite() && diff > PROVISIONAL_CPU_FLOOR * RED_CONTROL_SEPARATION_MULTIPLE,
        "row_lengths off-by-one control failed to separate above the measured floor \
         (diff={diff:e}, required > {:e}) -- per contract E4, an oracle whose red control \
         cannot separate is inadmissible and must be reshaped, never tuned",
        PROVISIONAL_CPU_FLOOR * RED_CONTROL_SEPARATION_MULTIPLE,
    );
}

/// RED control 2 (window radius off-by-one): the padded-batch leg is
/// computed with a MUTANT config whose `local_attention` is shifted by
/// `2` (`half_window` shifted by `1`) relative to the config the alone
/// leg uses -- simulating a defect where the batch path and the alone
/// path disagree about the sliding-window radius. Uses the fixture's
/// LONGEST row so the window actually binds under BOTH the original and
/// the mutant `half_window` (checked premise, asserted in-test, not
/// merely claimed in a doc comment -- a shorter row would pass this
/// control vacuously per this file's own module doc).
#[test]
fn pooled_embedding_red_control_window_radius_off_by_one_f32_cpu() {
    let device = Device::Cpu;
    let config = load_config();
    let fixture = build_fixture(&device);

    let mut mutant_config = config.clone();
    mutant_config.local_attention = config.local_attention + 2;
    assert_eq!(
        mutant_config.half_window(),
        config.half_window() + 1,
        "checked premise: the mutant config must actually change half_window by exactly 1"
    );

    let long_idx = fixture
        .lengths
        .iter()
        .enumerate()
        .max_by_key(|&(_, &len)| len)
        .map(|(i, _)| i)
        .expect("fixture is non-empty");
    let long_len = fixture.lengths[long_idx];
    assert!(
        long_len >= mutant_config.half_window() + 2,
        "checked premise: segment_len ({long_len}) must be >= half_window + 2 \
         ({}) for BOTH the original and mutant config, or the window control is vacuous",
        mutant_config.half_window() + 2,
    );
    assert!(
        long_len >= config.half_window() + 2,
        "checked premise (original config side)"
    );

    let encoder = build_encoder(&device, DType::F32, &config);
    let encoder_mut = build_encoder(&device, DType::F32, &mutant_config);

    let alone_long = pooled_alone(&encoder, &device, &fixture.rows[long_idx]);
    let pooled_batch_mut = encoder_mut
        .forward(&fixture.ids_padded, &fixture.mask_padded)
        .expect("mutant-window padded batch forward");
    let pooled_batch_mut: Vec<Vec<f32>> = pooled_batch_mut.to_vec2().unwrap();

    let diff = max_abs_diff(&alone_long, &pooled_batch_mut[long_idx]);
    assert!(
        diff.is_finite() && diff > PROVISIONAL_CPU_FLOOR * RED_CONTROL_SEPARATION_MULTIPLE,
        "window radius off-by-one control failed to separate above the measured floor \
         (diff={diff:e}, required > {:e}) -- per contract E4, an oracle whose red control \
         cannot separate is inadmissible and must be reshaped, never tuned",
        PROVISIONAL_CPU_FLOOR * RED_CONTROL_SEPARATION_MULTIPLE,
    );
}

// ─────────────────────────────────────────────────────────────────────────
// GPU-gated legs (`bf16`, CUDA eager arm) -- capability-gated the same way
// the M1b family gates (`Device::new_cuda(0)`, `JAMMI_REQUIRE_CUDA`
// opt-in-panic, silent skip otherwise). PROVISIONAL bounds below are
// PLACEHOLDERS this agent could not measure (no CUDA device in this
// environment) -- the pod train (contract E4, Step 5) MUST replace them
// with a measured same-composition floor before this leg gates anything,
// the same way `FLASH_ORACLE_PADDED_BOUND` (`src/modernbert.rs`) was
// replaced by a real 8-seed pod harvest before it gated anything.
// ─────────────────────────────────────────────────────────────────────────

/// Mirrors `src/modernbert.rs`'s own `growth_oracle_cuda_device` /
/// `tests/cuda_parity.rs`'s `cuda_device`: a machine built with the
/// `cuda` feature but with no physical GPU reads as "skip", not "fail",
/// UNLESS `JAMMI_REQUIRE_CUDA` is set, in which case device-acquisition
/// failure panics rather than silently reading as a skip.
#[cfg(feature = "cuda")]
fn cuda_device_or_skip(test_name: &str) -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!("{test_name}: JAMMI_REQUIRE_CUDA is set but no CUDA device: {e}");
            }
            eprintln!("{test_name}: skipping -- no CUDA device available ({e})");
            None
        }
    }
}

/// `Σ|a_i - b_i| / max(Σ|a_i|, f32::EPSILON)` -- a bare relative-L1
/// ratio whose noise-free value is `0.0` (matching this file's CPU
/// metric's floor), used for the `bf16` legs where an absolute
/// `f32::EPSILON`-unit bound has no principled meaning (`bf16` rounding
/// noise is a relative, not an absolute, quantity).
#[cfg(feature = "cuda")]
fn relative_l1_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum();
    let den: f64 = a
        .iter()
        .map(|x| (*x as f64).abs())
        .sum::<f64>()
        .max(f64::EPSILON);
    num / den
}

/// PROVISIONAL PLACEHOLDER -- UNMEASURED. Bounds the `bf16`-CUDA
/// alone-vs-padded-batch [`relative_l1_error`] on real rows. This value
/// is NOT derived from any measurement (no CUDA device was available to
/// this agent); it exists only so the CPU-hermetic gate can typecheck
/// this file's structure while feature-gated out. The pod train
/// (contract E4, Step 5) MUST measure this oracle's real `relative_l1_error`
/// on hardware (the same discipline `FLASH_ORACLE_PADDED_BOUND` used: an
/// 8-seed-class harvest, mean asserted, per-seed ratios printed) and
/// replace this constant before this leg is allowed to gate anything.
#[cfg(feature = "cuda")]
const PROVISIONAL_GPU_FLOOR_PLACEHOLDER: f64 = f64::NAN;

/// Refuses to let [`PROVISIONAL_GPU_FLOOR_PLACEHOLDER`] silently gate a
/// real assertion: `NaN` fails every ordered comparison (`esc-005`'s own
/// "`NaN > c` is `false`" trap, applied deliberately here in the OTHER
/// direction), so any CUDA-gated test that reaches this placeholder
/// panics loudly identifying itself, instead of a `NaN` bound vacuously
/// admitting an unmeasured leg.
#[cfg(feature = "cuda")]
fn require_pod_measured_gpu_floor(test_name: &str) -> ! {
    panic!(
        "{test_name}: PROVISIONAL_GPU_FLOOR_PLACEHOLDER is unmeasured (NaN) -- the pod train \
         (contract docs/plans/62-embedding-surface/CONTRACT.md E4, Step 5) must land a real \
         measured same-composition bf16 floor here before this CUDA leg can assert anything; \
         see that constant's own doc"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_alone_matches_padded_batch_real_row_bf16_cuda() {
    let Some(device) =
        cuda_device_or_skip("pooled_embedding_alone_matches_padded_batch_real_row_bf16_cuda")
    else {
        return;
    };
    if PROVISIONAL_GPU_FLOOR_PLACEHOLDER.is_nan() {
        require_pod_measured_gpu_floor(
            "pooled_embedding_alone_matches_padded_batch_real_row_bf16_cuda",
        );
    }
    let config = load_config();
    let encoder = build_encoder(&device, DType::BF16, &config);
    let fixture = build_fixture(&device);
    let pooled_batch = encoder
        .forward(&fixture.ids_padded, &fixture.mask_padded)
        .expect("padded batch forward (bf16 cuda)");
    let pooled_batch: Vec<Vec<f32>> = pooled_batch
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2()
        .unwrap();

    // f32-truth anchor: the CPU f32 forward of each row's alone
    // composition, independent of the CUDA bf16 arm under test.
    let truth_encoder = build_encoder(&Device::Cpu, DType::F32, &config);

    for (b, row) in fixture.rows.iter().enumerate() {
        let alone_bf16 = pooled_alone(&encoder, &device, row);
        let truth = pooled_alone(&truth_encoder, &Device::Cpu, row);
        let ratio_alone_vs_truth = relative_l1_error(&alone_bf16, &truth);
        let ratio_batch_vs_truth = relative_l1_error(&pooled_batch[b], &truth);
        let ratio_alone_vs_batch = relative_l1_error(&alone_bf16, &pooled_batch[b]);
        eprintln!(
            "row={b} alone_vs_truth={ratio_alone_vs_truth:e} batch_vs_truth={ratio_batch_vs_truth:e} \
             alone_vs_batch={ratio_alone_vs_batch:e} (bound {PROVISIONAL_GPU_FLOOR_PLACEHOLDER:e})"
        );
        assert!(
            ratio_alone_vs_batch.is_finite()
                && ratio_alone_vs_batch < PROVISIONAL_GPU_FLOOR_PLACEHOLDER,
            "row {b}: bf16 alone-vs-batch relative_l1_error {ratio_alone_vs_batch:e} exceeds the \
             PROVISIONAL bound {PROVISIONAL_GPU_FLOOR_PLACEHOLDER:e}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_red_control_row_length_off_by_one_bf16_cuda() {
    let Some(device) =
        cuda_device_or_skip("pooled_embedding_red_control_row_length_off_by_one_bf16_cuda")
    else {
        return;
    };
    if PROVISIONAL_GPU_FLOOR_PLACEHOLDER.is_nan() {
        require_pod_measured_gpu_floor(
            "pooled_embedding_red_control_row_length_off_by_one_bf16_cuda",
        );
    }
    let config = load_config();
    let encoder = build_encoder(&device, DType::BF16, &config);
    let fixture = build_fixture(&device);

    let mut mask_mut = vec![0u32; fixture.batch * fixture.seq];
    for (b, row) in fixture.rows.iter().enumerate() {
        let take = if b == 0 { row.len() - 1 } else { row.len() };
        for i in 0..take {
            mask_mut[b * fixture.seq + i] = 1;
        }
    }
    let mask_mut = Tensor::from_vec(mask_mut, (fixture.batch, fixture.seq), &device).unwrap();
    let pooled_batch_mut = encoder
        .forward(&fixture.ids_padded, &mask_mut)
        .expect("mutated-length padded batch forward (bf16 cuda)");
    let pooled_batch_mut: Vec<Vec<f32>> = pooled_batch_mut
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2()
        .unwrap();

    let alone0 = pooled_alone(&encoder, &device, &fixture.rows[0]);
    let ratio = relative_l1_error(&alone0, &pooled_batch_mut[0]);
    assert!(
        ratio.is_finite() && ratio > PROVISIONAL_GPU_FLOOR_PLACEHOLDER * 5.0,
        "row_lengths off-by-one control failed to separate above the PROVISIONAL bf16 floor \
         (ratio={ratio:e}, required > {:e})",
        PROVISIONAL_GPU_FLOOR_PLACEHOLDER * 5.0,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda() {
    let Some(device) =
        cuda_device_or_skip("pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda")
    else {
        return;
    };
    if PROVISIONAL_GPU_FLOOR_PLACEHOLDER.is_nan() {
        require_pod_measured_gpu_floor(
            "pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda",
        );
    }
    let config = load_config();
    let fixture = build_fixture(&device);

    let mut mutant_config = config.clone();
    mutant_config.local_attention = config.local_attention + 2;
    assert_eq!(mutant_config.half_window(), config.half_window() + 1);

    let long_idx = fixture
        .lengths
        .iter()
        .enumerate()
        .max_by_key(|&(_, &len)| len)
        .map(|(i, _)| i)
        .expect("fixture is non-empty");
    let long_len = fixture.lengths[long_idx];
    assert!(long_len >= mutant_config.half_window() + 2);
    assert!(long_len >= config.half_window() + 2);

    let encoder = build_encoder(&device, DType::BF16, &config);
    let encoder_mut = build_encoder(&device, DType::BF16, &mutant_config);

    let alone_long = pooled_alone(&encoder, &device, &fixture.rows[long_idx]);
    let pooled_batch_mut = encoder_mut
        .forward(&fixture.ids_padded, &fixture.mask_padded)
        .expect("mutant-window padded batch forward (bf16 cuda)");
    let pooled_batch_mut: Vec<Vec<f32>> = pooled_batch_mut
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2()
        .unwrap();

    let ratio = relative_l1_error(&alone_long, &pooled_batch_mut[long_idx]);
    assert!(
        ratio.is_finite() && ratio > PROVISIONAL_GPU_FLOOR_PLACEHOLDER * 5.0,
        "window radius off-by-one control failed to separate above the PROVISIONAL bf16 floor \
         (ratio={ratio:e}, required > {:e})",
        PROVISIONAL_GPU_FLOOR_PLACEHOLDER * 5.0,
    );
}
