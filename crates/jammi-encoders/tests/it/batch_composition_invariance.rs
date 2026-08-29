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
//! (contract v4 §2, `modernbert.rs:2801-2808`) -- the encode/serving surface
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
//! composition and ASSERT each `bf16` arm -- alone and batch,
//! independently -- tracks it within [`GPU_TRUTH_DRIFT_BOUND`], so a real
//! dtype-rounding regression cannot hide inside an alone-vs-batch
//! agreement that happens to cancel (both arms drifting IDENTICALLY away
//! from truth would still fail these two asserts even though the
//! alone-vs-batch [`gpu_composition_floor`] check would pass) -- the
//! truth ratios are not merely printed, they gate the test.
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
//! CUDA bounds ([`gpu_composition_floor`], [`GPU_TRUTH_DRIFT_BOUND`])
//! were MEASURED on the unit 62 landing round (`measure_gpu_floors_print_only`,
//! tree `67ba2394`, `JAMMI_REQUIRE_CUDA=1`, 8 compositions x 88
//! row-measurements per arch, one pod run per arch -- a100
//! `cjjh6oaqehvpwi`, h100 `gufh54wmqox1rw`, a40 `qlc5z76zh98v6c`, l40s
//! `kccwbawx92pou1`) and folded into per-arch bounds with documented
//! margin arithmetic -- see [`gpu_composition_floor`]'s and
//! [`GPU_TRUTH_DRIFT_BOUND`]'s own docs for the full measured values,
//! margins, and admissibility scoping. Exactly as `FLASH_ORACLE_PADDED_BOUND`
//! (`src/modernbert.rs`) was derived from a real 8-seed pod harvest rather
//! than invented, these two are derived from real pod measurements, never
//! guessed -- reshaped (arch-conditional / control admissibility scoped),
//! never tuned to pass.
//!
//! **Conjunctive red controls (scoped per-arch).** Two independent
//! mutants -- a `row_lengths` off-by-one (a batch construction bug: one
//! row's real length disagrees between the alone and batch legs) and a
//! sliding-window radius off-by-one (`local_attention` shifted by 2, i.e.
//! `half_window` shifted by 1) -- must EACH separate above the measured
//! floor for this oracle to be admissible; per the contract, if either
//! cannot separate the oracle is reshaped, never tuned to pass. This
//! conjunctive requirement holds UNSCOPED on the exact-arches class
//! (sm80/sm86/sm90): both controls separate above
//! `EXACT_ARCH_COMPOSITION_FLOOR` on every composition measured there. It
//! does NOT hold unscoped on sm89 (L40S): the window-radius control's own
//! measured minimum separation there is smaller than
//! `SM89_COMPOSITION_FLOOR`, so that control is INADMISSIBLE on sm89 and
//! is SKIPPED with a loud documented reason instead of asserted
//! (`pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda`);
//! the row-length control stays admissible on sm89 but only
//! COMPOSITION-SCOPED there (admissible for the fixture composition the
//! gating test actually exercises, not for every composition measured on
//! that arch) -- see `gpu_composition_floor`'s own doc for the full
//! per-arch, per-composition numbers and margin arithmetic this scoping
//! is derived from. The window control's fixture asserts `segment_len >= half_window + 2`
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

/// Extracts the pooled row as `Vec<f32>` for the alone-encode leg, dtype-robust
/// to the encoder's backbone dtype: on CPU the encoder runs (and emits) `F32`,
/// so `to_dtype(F32)` is a no-op; on CUDA the encoder emits `BF16` (pod
/// evidence, unit 62: `measure_gpu_floors_print_only` panicked here with
/// `unexpected dtype, expected: F32, got: BF16` before this fix), so the
/// explicit cast converts the OUTPUT for extraction only -- the computation
/// itself stays at the encoder's own dtype (`bf16` on CUDA); this function
/// never widens the arithmetic, only the value it hands back for comparison.
/// Mirrors `src/modernbert.rs`'s own CUDA test idiom (e.g.
/// `forward_hidden_dispatches_attention_block_flash_fused_on_a_dense_cuda_bf16_checkpoint`):
/// `.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()`.
fn pooled_alone(encoder: &ModernBert, device: &Device, row: &[u32]) -> Vec<f32> {
    let len = row.len();
    let ids = Tensor::from_vec(row.to_vec(), (1, len), device).unwrap();
    let mask = Tensor::from_vec(vec![1u32; len], (1, len), device).unwrap();
    let pooled = encoder.forward(&ids, &mask).expect("alone forward");
    pooled
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
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

/// MEASURED (macOS dev box, this file's `build_fixture`, `f32` CPU): the
/// observed per-row max-abs pooled-embedding delta between the alone and
/// padded-batch legs peaked at `4.4703484e-8` (`0.375 * f32::EPSILON`)
/// across all 16 rows (row 12's per-element RELATIVE error spiked to
/// `1.54e-4` only because that specific element is itself
/// near-cancelling; the ABSOLUTE delta there was `1.86e-8`, still tiny --
/// this bound is stated in absolute terms because the compared quantity
/// is a pooled-and-L2-normalised embedding, bounded to `[-1, 1]` per
/// element by construction, so an absolute bound is meaningful and a
/// relative one is not, near cancellation). `8.0 * f32::EPSILON`
/// (`9.5367432e-7`) gives ~21x headroom over the measured worst case --
/// enough to absorb this file's own CPU-vs-CPU architecture caveat
/// (`tests/it/modernbert.rs`'s documented finding that Linux CPU BLAS
/// blocking-by-total-M can differ from macOS's) -- while still
/// separating comfortably below both red controls' measured margins
/// (row-length off-by-one: `1.6e-2`, ~17,000x above this bound; window
/// radius off-by-one: `9.5e-6`, ~10x above this bound).
///
/// CONFIRMED on Linux (contract E4, Step 5): this exact bound was
/// re-measured on the CI-representative pod train across all four
/// Linux pod architectures (a100, h100, l40s, a40) -- the f32 invariance
/// leg, both red controls, and the padded-training-parity sibling
/// (`modernbert::padded_training_loss_and_lora_grads_match_unpadded_rows_run_individually_f32_cpu`)
/// each reported `4 passed; 0 failed` on every pod, with no admission
/// flips. Evidence:
/// `docs/plans/62-embedding-surface/measurements/pod-runs/cpu-floor-legs-4pods.txt`.
/// The value is retained as-is, not replaced -- the Linux re-measurement
/// confirmed the macOS-derived bound rather than requiring a tighter or
/// looser one. The constant's `PROVISIONAL` name prefix is now
/// historical (it predates the Linux confirmation); it is kept unchanged
/// here to avoid a repo-wide rename of every call site for zero semantic
/// gain.
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
// opt-in-panic, silent skip otherwise). Bounds below (`gpu_composition_floor`,
// `GPU_TRUTH_DRIFT_BOUND`) are MEASURED (unit 62 landing round, tree
// `67ba2394`, four-arch pod run) and folded in with documented margin
// arithmetic -- see each constant/function's own doc -- the same way
// `FLASH_ORACLE_PADDED_BOUND` (`src/modernbert.rs`) was replaced by a real
// 8-seed pod harvest before it gated anything. An arch outside the
// measured set still fails LOUD (`f64::NAN` -> `require_pod_measured_floor`
// panic), never silently guesses a floor -- see `gpu_composition_floor`'s
// own doc.
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
/// noise is a relative, not an absolute, quantity). NOT `cuda`-gated
/// (pure numeric logic, no device dependency) so the CPU-hermetic mutant
/// demonstration below (`identical_bf16_drift_...`) can exercise the same
/// function the CUDA legs use, rather than a re-implementation that could
/// drift from it.
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

/// PER-ARCH, MEASURED composition-invariance floor for the `bf16`-CUDA
/// alone-vs-padded-batch [`relative_l1_error`] on real rows (does the SAME
/// dtype/device arm agree with itself across the alone vs padded-batch
/// composition). Measured by `measure_gpu_floors_print_only` (unit 62
/// landing round, tree `67ba2394`, `JAMMI_REQUIRE_CUDA=1`, 8 compositions
/// x 88 row-measurements per arch, one pod run per arch, push-stamps
/// verified):
///
/// | arch (compute cap) | pod              | alone_vs_batch mean      | alone_vs_batch max        |
/// |---------------------|-------------------|---------------------------|-----------------------------|
/// | a100 sm80 (8,0)      | `cjjh6oaqehvpwi`   | `0e0`                     | `0e0` (EXACT)               |
/// | h100 sm90 (9,0)      | `gufh54wmqox1rw`   | `0e0`                     | `0e0` (EXACT)               |
/// | a40  sm86 (8,6)      | `qlc5z76zh98v6c`   | `0e0`                     | `0e0` (EXACT)               |
/// | l40s sm89 (8,9)      | `kccwbawx92pou1`   | `1.1805235731113235e-3`   | `4.118649354617619e-3`      |
///
/// **Exact-arches class (sm80/sm86/sm90, [`EXACT_ARCH_COMPOSITION_FLOOR`]).**
/// 264 row-measurements (3 arches x 88 each) came back EXACTLY `0.0`,
/// zero flakiness -- the same bit-exact-zero this file's own module doc
/// predicts from `MASKED_LOGIT` underflow (a pad weight's contribution to
/// the value-weighted sum is `0.0 * finite == 0.0` exactly). The floor is
/// set to `1e-5`: the weakest red control separation measured on this
/// class (the window-radius control's own min, `7.508231757090548e-4`)
/// clears the ASSERTED THRESHOLD (`floor * RED_CONTROL_SEPARATION_MULTIPLE`
/// = `1e-5 * 5.0 = 5e-5`), never the bare floor, by `~15.0x`
/// (`7.508231757090548e-4 / 5e-5 ~= 15.0x`), and the floor is infinitely
/// above the measured `0.0` -- a `0.0` floor itself would admit no float
/// slop at all, including benign FMA/ordering differences on a future,
/// architecturally-identical but as-yet-unmeasured exact-arch SKU.
///
/// **sm89 (L40S, [`SM89_COMPOSITION_FLOOR`]) genuinely diverges** --
/// consistent with the M1b campaign's own A40-pass/L40S-fail finding for
/// this exact "Ada-class is not one behavior" pattern
/// (`lora_linear_dx_abs_floor`, `crates/jammi-kernels/tests/cuda_parity.rs`):
/// a different cuBLAS/cuDNN kernel selection on this SKU, not flakiness.
/// Derivation (M1b-style measure-then-margin-then-round-clean discipline):
/// ```text
/// measured max (l40s, kccwbawx92pou1, tree 67ba2394):
///   4.118649354617619e-3
/// margin (~2% headroom over the measured max -- kept modest rather than
/// `lora_linear_dx_abs_floor`'s 1.5x precedent, because a larger margin
/// here would further erode the row-length red control's own
/// composition-scoped separation documented in the admissibility note
/// below):
///   4.118649354617619e-3 * 1.02 = 4.2010263...e-3
/// NOTE (numeric-advisory correction): `4.2e-3` is actually BELOW this
/// 1.02x product (`4.118649354617619e-3 * 1.02 = 4.2010263...e-3 >
/// 4.2e-3`), so this is NOT a genuine round-UP past the margin target --
/// the realized margin against the measured max is
/// `4.2e-3 / 4.118649354617619e-3 ~= 1.0197x` (~1.97% headroom, a hair
/// under the intended 2%). The constant is kept as `4.2e-3` regardless
/// (not tightened by this correction): it is still finite, positive
/// headroom over the measured max and a "clean" decimal value, which is
/// what this derivation actually needs -- only the prose claiming a
/// round-up was wrong, not the constant itself:
///   4.2e-3
/// ```
///
/// **Admissibility scoping (reshape, never tune).** The window-radius red
/// control's own measured minimum separation on sm89,
/// `1.139471768897301e-3`, is SMALLER than [`SM89_COMPOSITION_FLOOR`]
/// (`4.2e-3`) -- the control cannot separate above the floor on this arch
/// at all, so it is INADMISSIBLE there and SKIPPED WITH A LOUD DOCUMENTED
/// REASON in
/// [`pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda`]
/// (never silently green).
///
/// The row-length control's admissibility on sm89 is stated the same way
/// the window control's already is above: the GATING STATISTIC vs the
/// ASSERTED THRESHOLD (`floor * RED_CONTROL_SEPARATION_MULTIPLE`), never
/// a raw max-over-floor ratio (a prior draft of this doc divided the
/// cross-composition MAX by the bare floor -- `6.881763611768685e-2 /
/// 4.2e-3 ~= 16.4x` -- which is the wrong pair of numbers; corrected
/// below). The gating test
/// ([`pooled_embedding_red_control_row_length_off_by_one_bf16_cuda`])
/// exercises composition 0; its measured ratio, `6.881763611768685e-2`,
/// clears the asserted threshold
/// (`SM89_COMPOSITION_FLOOR * RED_CONTROL_SEPARATION_MULTIPLE` =
/// `4.2e-3 * 5.0 = 2.1e-2`) by `~3.28x`
/// (`6.881763611768685e-2 / 2.1e-2 ~= 3.28x`).
///
/// This clearance is COMPOSITION-SCOPED on sm89, not universal, and this
/// doc says so plainly rather than implying every composition clears:
/// the full 8-composition row-length ratio set measured this landing
/// round (l40s `kccwbawx92pou1`, tree `67ba2394`, cited verbatim from
/// `docs/plans/62-embedding-surface/measurements/gpu-floors-l40s.txt`),
/// compositions 0..7 in order, is `6.881763611768685e-2,
/// 6.881763611768685e-2, 6.9996589149257556e-3, 6.881763611768685e-2,
/// 4.2952616000474945e-2, 5.418507501917013e-3, 6.881763611768685e-2,
/// 1.0134661986953957e-2`. Compositions 2 (`6.9996589149257556e-3`), 5
/// (`5.418507501917013e-3`), AND 7 (`1.0134661986953957e-2`) all measure
/// BELOW the `2.1e-2` threshold on sm89 -- the row-length control is
/// therefore admissible on sm89 ONLY for the fixture composition the
/// gating test actually exercises (composition 0), not for every
/// composition this arch was measured at; a future change to the gating
/// test's fixture composition would need to re-check this scoping, never
/// assume it carries over unchanged.
///
/// On the exact-arches class the row-length control IS universal
/// (unlike sm89): even its weakest measured composition there
/// (composition 5, `5.483950988976972e-3` -- a100/h100/a40, same
/// artifact directory) clears
/// [`EXACT_ARCH_COMPOSITION_FLOOR`]'s asserted threshold
/// (`1e-5 * 5.0 = 5e-5`) by `~109.7x`
/// (`5.483950988976972e-3 / 5e-5 ~= 109.7x`); the gating composition
/// (composition 0, `6.881763611768685e-2`) clears it by `~1376x`.
///
/// Detected at runtime via
/// `jammi_kernels::admission::probe_cuda_compute_capability` /
/// `ComputeCapability`, the SAME per-arch idiom
/// `crates/jammi-kernels/tests/cuda_parity.rs`'s `lora_linear_dx_abs_floor`
/// uses (`ComputeCapability::new(major, minor)` equality match). An arch
/// this table has not measured (including a probe failure or a non-CUDA
/// device) returns `f64::NAN` deliberately -- [`require_pod_measured_floor`]
/// turns that into a loud, named panic rather than a silent guess (family
/// D: an untested arch must fail loud, never silently borrow a floor it
/// was never shown to need).
#[cfg(feature = "cuda")]
const EXACT_ARCH_COMPOSITION_FLOOR: f64 = 1e-5;

/// See [`gpu_composition_floor`]'s doc for the full sm89 derivation
/// (measured max `4.118649354617619e-3`, `1.02x` margin, rounded to a
/// clean `4.2e-3`).
#[cfg(feature = "cuda")]
const SM89_COMPOSITION_FLOOR: f64 = 4.2e-3;

/// Arch-conditional lookup for [`EXACT_ARCH_COMPOSITION_FLOOR`] /
/// [`SM89_COMPOSITION_FLOOR`] -- see [`EXACT_ARCH_COMPOSITION_FLOOR`]'s own
/// doc for the full measured values, margin arithmetic, and admissibility
/// scoping this landed from. Mirrors
/// `crates/jammi-kernels/tests/cuda_parity.rs`'s `lora_linear_dx_abs_floor`
/// arch-detection idiom exactly (`probe_cuda_compute_capability` +
/// `ComputeCapability::new(major, minor)` equality match), except an
/// UNRECOGNISED capability here returns `f64::NAN` rather than a
/// tight-but-untested default: this oracle already has a dedicated
/// NaN-panic guard ([`require_pod_measured_floor`]) built for exactly this
/// "arch outside the measured set" case, so the lookup fails loud through
/// that mechanism instead of silently reusing a bound derived from
/// different hardware.
///
/// **Recorded residual: capability-only key, not `(arch, build)`.** This
/// lookup keys SOLELY on driver-probed compute capability
/// (`ComputeCapability::new(major, minor)`), a coarser key than the
/// repository's own per-`(arch, build)` determinism rule (family J) used
/// elsewhere. Only ONE SKU per capability class was actually measured
/// this landing round -- a100 sm80 `(8,0)`, h100 sm90 `(9,0)`, a40 sm86
/// `(8,6)`, l40s sm89 `(8,9)`, per
/// `docs/plans/62-embedding-surface/measurements/README.md` -- so an
/// unmeasured SKU that merely REPORTS the same capability (e.g. a
/// different sm89 card) inherits its whole class's floor without ever
/// having been measured itself. This is a stated residual, not a silent
/// gap: it belongs to `esc-062`'s arch-axis family (capability is a
/// coarser key than the repo's `(arch, build)` determinism unit), and
/// tightening it -- per-SKU measurement, or a documented argument that
/// capability alone suffices -- is deferred, not resolved, by this
/// lookup.
#[cfg(feature = "cuda")]
fn gpu_composition_floor(device: &Device) -> f64 {
    use jammi_kernels::admission::{probe_cuda_compute_capability, ComputeCapability};
    match probe_cuda_compute_capability(device) {
        Some(cap) if cap == ComputeCapability::new(8, 9) => SM89_COMPOSITION_FLOOR,
        Some(cap)
            if cap == ComputeCapability::new(8, 0)
                || cap == ComputeCapability::new(8, 6)
                || cap == ComputeCapability::new(9, 0) =>
        {
            EXACT_ARCH_COMPOSITION_FLOOR
        }
        // Every other probed capability (including a probe failure or
        // non-CUDA device) is outside the measured set: fail loud via
        // `require_pod_measured_floor`, never guess (family D).
        _ => f64::NAN,
    }
}

/// Arch-CONSISTENT, MEASURED dtype-rounding drift bound for each
/// `bf16`-CUDA arm's [`relative_l1_error`] against the independently
/// computed `f32`-CPU truth (`ratio_alone_vs_truth`, `ratio_batch_vs_truth`)
/// -- a DIFFERENT quantity from [`gpu_composition_floor`]'s
/// composition-invariance floor: "does bf16 agree with itself across
/// compositions" and "does bf16 agree with f32 truth" can diverge
/// independently -- two bf16 arms can drift IDENTICALLY away from truth (a
/// real dtype-rounding regression) while still agreeing with EACH OTHER,
/// which would pass a composition-invariance-only check vacuously
/// (finding F-6). Measured (unit 62 landing round,
/// `measure_gpu_floors_print_only`, tree `67ba2394`, `JAMMI_REQUIRE_CUDA=1`):
/// a100 (`cjjh6oaqehvpwi`), h100 (`gufh54wmqox1rw`), a40 (`qlc5z76zh98v6c`)
/// all reported IDENTICAL values, `alone_vs_truth mean=3.5208168455960124e-3`
/// `max=4.222114077112533e-3` (`batch_vs_truth` identical to
/// `alone_vs_truth`); l40s (`kccwbawx92pou1`) measured
/// `alone_vs_truth mean=3.597633555417087e-3` `max=4.233727028564512e-3` --
/// the cross-arch worst max across all four pods.
///
/// ONE bound serves every arch here (unlike [`gpu_composition_floor`],
/// which splits arch-conditionally): the dtype-rounding noise this bound
/// governs (`bf16` round-to-nearest storage error) is a property of the
/// dtype/kernel choice, not the GEMM reduction-order divergence that made
/// the composition floor arch-split -- measured evidence bears this out.
/// Reproducibly, from the numbers above: the MAX spread between l40s and
/// the three identical arches is
/// `(4.233727028564512e-3 - 4.222114077112533e-3) /
/// 4.222114077112533e-3 ~= 0.275%`, and the MEAN spread is
/// `(3.597633555417087e-3 - 3.5208168455960124e-3) /
/// 3.5208168455960124e-3 ~= 2.18%` -- both well under the composition
/// floor's `0` vs `4.1e-3` split (an effectively-infinite relative
/// spread, since the exact-arches side is exactly `0.0`).
///
/// Derivation (M1b margin convention -- `crates/jammi-kernels/tests/cuda_parity.rs`'s
/// `lora_linear_dx_abs_floor` sibling comment: "`dx_bound_margin`'s sibling
/// comment used `2.0x` against a measured `1.34x` need"; `2.0x` chosen over
/// this file's own tighter `1.02x` composition-floor margin because this
/// bound has no red-control separation ceiling pushing back against a
/// generous margin):
/// ```text
/// cross-arch max (l40s, kccwbawx92pou1, tree 67ba2394):
///   4.233727028564512e-3
/// margin (2.0x):
///   4.233727028564512e-3 * 2.0 = 8.467454057129024e-3
/// rounded UP to a clean value (never round down past the margin):
///   1e-2
/// ```
/// giving `~2.36x` headroom over the measured cross-arch worst case
/// (`1e-2 / 4.233727028564512e-3 ~= 2.362x`).
#[cfg(feature = "cuda")]
const GPU_TRUTH_DRIFT_BOUND: f64 = 1e-2;

/// Refuses to let an unmeasured floor value (identified by `floor_name`)
/// silently gate a real assertion: `NaN` fails every ordered comparison
/// (`esc-005`'s own "`NaN > c` is `false`" trap, applied deliberately here
/// in the OTHER direction), so any CUDA-gated test that reaches an
/// unmeasured floor panics loudly identifying itself and the specific
/// constant, instead of a `NaN` bound vacuously admitting an unmeasured
/// leg. [`gpu_composition_floor`]'s arch-conditional lookup is the value
/// this guard actually protects: an arch outside its measured set
/// (`EXACT_ARCH_COMPOSITION_FLOOR`'s sm80/sm86/sm90 class,
/// `SM89_COMPOSITION_FLOOR`'s sm89) returns `f64::NAN`, and this call
/// turns that into a named panic rather than a silent pass.
/// [`GPU_TRUTH_DRIFT_BOUND`] is now a fixed, arch-consistent, always-real
/// constant (never `NaN` for any arch), so passing it through this guard
/// is defensive-only here (never expected to fire) -- kept for uniformity
/// with the composition-floor call rather than special-cased away. NOT
/// `cuda`-gated (pure control-flow logic, no device dependency) so
/// [`require_pod_measured_floor_panics_on_unmeasured_nan`] below can
/// exercise this exact function on CPU.
fn require_pod_measured_floor(test_name: &str, floor_name: &str, floor: f64) {
    if floor.is_nan() {
        panic!(
            "{test_name}: {floor_name} is unmeasured (NaN) -- the pod train \
             (contract docs/plans/62-embedding-surface/CONTRACT.md E4, Step 5) must land a real \
             measured value here before this CUDA leg can assert anything; see that constant's \
             own doc"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_alone_matches_padded_batch_real_row_bf16_cuda() {
    let test_name = "pooled_embedding_alone_matches_padded_batch_real_row_bf16_cuda";
    let Some(device) = cuda_device_or_skip(test_name) else {
        return;
    };
    let composition_floor = gpu_composition_floor(&device);
    require_pod_measured_floor(test_name, "gpu_composition_floor", composition_floor);
    require_pod_measured_floor(test_name, "GPU_TRUTH_DRIFT_BOUND", GPU_TRUTH_DRIFT_BOUND);
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
             alone_vs_batch={ratio_alone_vs_batch:e} (composition floor \
             {composition_floor:e}, drift bound {GPU_TRUTH_DRIFT_BOUND:e})"
        );
        // Truth-tracking control (finding F-6): each bf16 arm -- alone AND
        // batch, independently -- must itself stay within
        // `GPU_TRUTH_DRIFT_BOUND` of the f32-CPU truth. Without these two
        // asserts, two bf16 arms that drift IDENTICALLY away from truth
        // would still agree with each other and pass the alone-vs-batch
        // assertion below, hiding a real dtype-rounding regression inside a
        // cancelling agreement -- exactly the gap this file's module doc
        // claims does not exist.
        assert!(
            ratio_alone_vs_truth.is_finite() && ratio_alone_vs_truth < GPU_TRUTH_DRIFT_BOUND,
            "row {b}: bf16 alone-vs-f32-truth relative_l1_error {ratio_alone_vs_truth:e} exceeds \
             the measured dtype-rounding drift bound {GPU_TRUTH_DRIFT_BOUND:e}"
        );
        assert!(
            ratio_batch_vs_truth.is_finite() && ratio_batch_vs_truth < GPU_TRUTH_DRIFT_BOUND,
            "row {b}: bf16 batch-vs-f32-truth relative_l1_error {ratio_batch_vs_truth:e} exceeds \
             the measured dtype-rounding drift bound {GPU_TRUTH_DRIFT_BOUND:e}"
        );
        assert!(
            ratio_alone_vs_batch.is_finite() && ratio_alone_vs_batch < composition_floor,
            "row {b}: bf16 alone-vs-batch relative_l1_error {ratio_alone_vs_batch:e} exceeds the \
             measured per-arch composition floor {composition_floor:e}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_red_control_row_length_off_by_one_bf16_cuda() {
    let test_name = "pooled_embedding_red_control_row_length_off_by_one_bf16_cuda";
    let Some(device) = cuda_device_or_skip(test_name) else {
        return;
    };
    let composition_floor = gpu_composition_floor(&device);
    require_pod_measured_floor(test_name, "gpu_composition_floor", composition_floor);
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
    // Row-length control is conjunctive with the window-radius control on
    // every arch (never skipped), but its admissibility is
    // COMPOSITION-SCOPED on sm89: this composition-0 fixture's measured
    // ratio (`6.881763611768685e-2`) clears the asserted threshold
    // `SM89_COMPOSITION_FLOOR * 5.0` (`2.1e-2`) by `~3.28x`, but
    // compositions 2, 5, and 7 measured BELOW that same threshold on sm89
    // this landing round and would not pass this exact assert if this
    // test built one of THEM instead -- see `gpu_composition_floor`'s own
    // doc for the full per-composition measurements and the honest
    // scoping statement.
    assert!(
        ratio.is_finite() && ratio > composition_floor * 5.0,
        "row_lengths off-by-one control failed to separate above the measured bf16 floor \
         (ratio={ratio:e}, required > {:e})",
        composition_floor * 5.0,
    );
}

#[test]
#[cfg(feature = "cuda")]
fn pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda() {
    let test_name = "pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda";
    let Some(device) = cuda_device_or_skip(test_name) else {
        return;
    };
    // Admissibility scoping (reshape, never tune -- see
    // `gpu_composition_floor`'s own doc for the full measured numbers this
    // scoping cites): on sm89 (L40S) the window-radius control's own
    // measured minimum separation (`1.139471768897301e-3`, unit 62 landing
    // round, pod `kccwbawx92pou1`, tree `67ba2394`) is SMALLER than
    // `SM89_COMPOSITION_FLOOR` (`4.2e-3`) -- the control cannot separate
    // above the floor on this arch at all, so it is INADMISSIBLE there and
    // SKIPPED, loudly, rather than asserted (which would either be vacuous
    // or reject a passing arch on control noise, not a real defect). The
    // row-length control above stays universal and is NOT skipped on any
    // arch.
    use jammi_kernels::admission::{probe_cuda_compute_capability, ComputeCapability};
    if probe_cuda_compute_capability(&device) == Some(ComputeCapability::new(8, 9)) {
        eprintln!(
            "{test_name}: SKIPPED on sm89 (L40S) -- the window-radius red control is \
             INADMISSIBLE on this arch (measured min separation 1.139471768897301e-3 < \
             SM89_COMPOSITION_FLOOR 4.2e-3; unit 62 landing round, pod kccwbawx92pou1, \
             tree 67ba2394). Per contract E4, an inadmissible control is scoped out with a \
             loud documented reason, never silently tuned to pass -- see \
             gpu_composition_floor's own doc for the full derivation and the row-length \
             control's universal admissibility on this same arch."
        );
        return;
    }
    let composition_floor = gpu_composition_floor(&device);
    require_pod_measured_floor(test_name, "gpu_composition_floor", composition_floor);
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
    // Only reachable on the exact-arches class (sm89 returns above): this
    // control's own measured min separation there (`7.508231757090548e-4`)
    // clears `EXACT_ARCH_COMPOSITION_FLOOR * 5.0` (`5e-5`) by `~15x` -- see
    // `gpu_composition_floor`'s own doc.
    assert!(
        ratio.is_finite() && ratio > composition_floor * 5.0,
        "window radius off-by-one control failed to separate above the measured bf16 floor \
         (ratio={ratio:e}, required > {:e})",
        composition_floor * 5.0,
    );
}

// ─────────────────────────────────────────────────────────────────────────
// CPU-hermetic mutant demonstration for finding F-6. No CUDA device is
// available in this environment, so the CUDA legs above cannot be run
// here; these two tests instead exercise the EXACT SAME helper functions
// (`relative_l1_error`, `require_pod_measured_floor`) the CUDA legs call,
// on synthetic data, to prove by construction that the new truth-tracking
// asserts (added by this fix) catch a mutant the old alone-vs-batch-only
// assertion structure would have missed.
// ─────────────────────────────────────────────────────────────────────────

/// F-6's exact failure mode, reconstructed on CPU: an "identical drift"
/// bf16 mutant where the alone and batch arms compute EXACTLY the same
/// (wrong) values -- so `relative_l1_error(alone, batch) == 0.0`, passing
/// any positive composition-invariance floor -- while BOTH arms have
/// drifted `1%` away from the independently-known truth (simulating a
/// dtype-rounding regression that a real GEMM/cast bug could introduce
/// identically on both the alone and batch code paths). Before this fix,
/// this file's sole bf16-CUDA assertion was on `ratio_alone_vs_batch`
/// (see this file's history / finding F-6): that check alone would have
/// PASSED this mutant, exactly as the auditor found. This test proves the
/// NEW truth-tracking checks -- `ratio_alone_vs_truth` /
/// `ratio_batch_vs_truth` each `< ILLUSTRATIVE_DRIFT_BOUND` -- correctly
/// FAIL it, using the identical [`relative_l1_error`] function the real
/// CUDA legs call (`ILLUSTRATIVE_DRIFT_BOUND` is a demonstration-only
/// value, deliberately kept separate from [`GPU_TRUTH_DRIFT_BOUND`] itself
/// -- this test's job is to prove the MECHANISM catches this mutant class
/// on synthetic data, independent of whatever the production constant's
/// own measured value happens to be).
#[test]
fn identical_bf16_drift_from_truth_passes_composition_check_but_fails_truth_tracking() {
    const ILLUSTRATIVE_COMPOSITION_FLOOR: f64 = 1e-6;
    const ILLUSTRATIVE_DRIFT_BOUND: f64 = 1e-3;

    let truth: Vec<f32> = vec![1.0, 2.0, 3.0, -4.0, 5.0];
    // Both bf16 arms round IDENTICALLY to the same (1% high) mutant
    // values -- simulating a dtype-rounding regression that hits the
    // alone and batch code paths the same way.
    let alone_bf16: Vec<f32> = truth.iter().map(|x| x * 1.01).collect();
    let batch_bf16 = alone_bf16.clone();

    let ratio_alone_vs_batch = relative_l1_error(&alone_bf16, &batch_bf16);
    let ratio_alone_vs_truth = relative_l1_error(&alone_bf16, &truth);
    let ratio_batch_vs_truth = relative_l1_error(&batch_bf16, &truth);

    // The OLD (pre-fix) check: alone-vs-batch agreement. Identical drift
    // means the two arms agree PERFECTLY with each other, so this check
    // is exactly the vacuous pass the auditor found.
    assert_eq!(
        ratio_alone_vs_batch, 0.0,
        "the mutant must be constructed so alone and batch agree exactly \
         (ratio_alone_vs_batch == 0.0), reproducing F-6's cancelling-agreement case"
    );
    assert!(
        ratio_alone_vs_batch.is_finite() && ratio_alone_vs_batch < ILLUSTRATIVE_COMPOSITION_FLOOR,
        "checked premise: the old alone-vs-batch-only check must PASS this mutant \
         (ratio={ratio_alone_vs_batch:e}), or this is not reproducing F-6's failure mode"
    );

    // The NEW checks this fix adds: each arm vs f32 truth, independently.
    // Both must be FAR enough above the illustrative bound to prove a
    // real dtype-rounding regression cannot hide here anymore.
    assert!(
        ratio_alone_vs_truth.is_finite() && ratio_alone_vs_truth >= ILLUSTRATIVE_DRIFT_BOUND,
        "the new alone-vs-truth check must CATCH this mutant \
         (ratio={ratio_alone_vs_truth:e}, bound={ILLUSTRATIVE_DRIFT_BOUND:e}) -- if it does not, \
         the truth-tracking fix has no discriminating power over this mutant class"
    );
    assert!(
        ratio_batch_vs_truth.is_finite() && ratio_batch_vs_truth >= ILLUSTRATIVE_DRIFT_BOUND,
        "the new batch-vs-truth check must CATCH this mutant \
         (ratio={ratio_batch_vs_truth:e}, bound={ILLUSTRATIVE_DRIFT_BOUND:e}) -- if it does not, \
         the truth-tracking fix has no discriminating power over this mutant class"
    );
}

/// [`require_pod_measured_floor`] must panic BEFORE any numeric compare
/// runs whenever its floor argument is unmeasured (`NaN`) -- the same
/// panic-before-compare discipline the pre-existing
/// [`gpu_composition_floor`] unknown-arch case relies on, now shared by
/// the [`GPU_TRUTH_DRIFT_BOUND`] truth-tracking checks too (finding F-6's
/// fix requirement: "the NaN placeholder must make these
/// panic-before-compare exactly like the main assertion"). This test
/// calls the exact function the CUDA legs call, with a bare `f64::NAN`,
/// so it needs no CUDA device to prove the mechanism.
#[test]
#[should_panic(expected = "is unmeasured (NaN)")]
fn require_pod_measured_floor_panics_on_unmeasured_nan() {
    require_pod_measured_floor("demo_test", "DEMO_BOUND", f64::NAN);
}

// ─────────────────────────────────────────────────────────────────────────
// Unit 62 gap fix: the MEASUREMENT path (PRINT ONLY -- gates nothing).
//
// The gap this section originally closed: every CUDA leg above called
// `require_pod_measured_floor` FIRST -- before the row-loop `eprintln!`
// that prints the very ratios the floor constants needed -- so a pod run
// of THOSE tests panicked on the then-unmeasured (`f64::NAN`) placeholders
// before a single ratio was ever computed: there was no path to obtain the
// numbers those constants needed. This section added that path: an
// `#[ignore]`d, print-only test that computes and prints the SAME
// quantities the gating tests assert (`ratio_alone_vs_truth`,
// `ratio_batch_vs_truth`, `ratio_alone_vs_batch`) plus the two red
// controls' own separations, over an 8-way sweep, asserting ONLY
// finiteness -- NEVER a numeric bound.
//
// Unit 62 bound-derivation round: this section's four-arch pod run (tree
// `67ba2394`, pods `cjjh6oaqehvpwi`/`gufh54wmqox1rw`/`qlc5z76zh98v6c`/
// `kccwbawx92pou1`) produced the measurements folded into
// [`EXACT_ARCH_COMPOSITION_FLOOR`], [`SM89_COMPOSITION_FLOOR`], and
// [`GPU_TRUTH_DRIFT_BOUND`] above (see each constant's own doc for the
// margin arithmetic). This test remains a measurement tool, not a gate --
// it stays in the tree, `#[ignore]`d, for the NEXT re-derivation (a new
// unmeasured arch, a fixture change, or a build change that could move the
// worst element the way `lora_linear_dx_abs_floor`'s own precedent
// documents) rather than being deleted now that one round has landed.
//
// **Composition sweep, not a seed sweep -- this fixture is
// content-deterministic.** `src/modernbert.rs`'s own multi-seed convention
// (`FLASH_ORACLE_SWEEP_SEEDS`, 8 seeds `201..=208`, re-drawing token
// content per seed via a SplitMix64 stream in `flash_oracle_synthetic_ids`)
// varies RANDOM token content per seed because that oracle's fixture has no
// other axis of variation. THIS file's fixture (`build_fixture`, see this
// file's own module doc) is explicitly content-deterministic -- "no RNG --
// this is an eager, non-random encode path" -- so there is no token content
// to re-seed. Per this unit's own plan (`docs/plans/62-embedding-surface/PLAN.md`,
// PR-C: "truth-relative mean ratio over the 8-seed convention"), the axis
// this oracle actually varies is BATCH COMPOSITION: the SAME 16-row pool
// `build_fixture` already builds (same content, same lengths), composed
// into 8 DIFFERENT padded batches (different subsets, different orders,
// different total batch sizes) -- directly exercising the mechanism this
// file's own module doc names as the source of any real divergence ("the
// underlying GEMMs...pick their blocking/accumulation order from the
// OPERAND SHAPE (total batch, total sequence length)"). `composition_id`
// (`0..8`) plays the structural role `seed` plays in the M1b family's
// sweep: an index this print-only test iterates and reports per-index,
// mean/max included -- the same reduction discipline (`total_cmp`-folded
// max, explicit mean) `mean_max` in `src/modernbert.rs` uses.
// ─────────────────────────────────────────────────────────────────────────

/// The 8 batch compositions this measurement sweeps, all built from the
/// SAME 16-row pool [`build_fixture`] returns (same token content, same
/// per-row lengths) -- only which rows participate and in what order/total
/// batch size varies, so any divergence measured here traces to composition
/// (operand shape), never to content drift between compositions.
#[cfg(feature = "cuda")]
const MEASUREMENT_COMPOSITION_SWEEP: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Maps a `composition_id` (`0..8`, [`MEASUREMENT_COMPOSITION_SWEEP`]) to
/// the row indices (into [`build_fixture`]'s 16-row pool) that compose that
/// batch: full set, both halves, even/odd interleave, full-reversed-order,
/// and two overlapping three-quarter windows -- eight structurally distinct
/// `(batch, seq)` operand shapes and orderings from one fixed content pool.
#[cfg(feature = "cuda")]
fn composition_row_indices(composition_id: usize, n: usize) -> Vec<usize> {
    match composition_id {
        0 => (0..n).collect(),
        1 => (0..n / 2).collect(),
        2 => (n / 2..n).collect(),
        3 => (0..n).step_by(2).collect(),
        4 => (1..n).step_by(2).collect(),
        5 => (0..n).rev().collect(),
        6 => (0..(n * 3 / 4)).collect(),
        7 => (n / 4..n).collect(),
        other => panic!("composition_row_indices: unknown composition_id {other} (expected 0..8)"),
    }
}

/// Builds one composed padded batch (rows/lengths + device tensors) from a
/// subset of `base`'s row pool, in the given order -- reuses
/// [`build_fixture`]'s row content/lengths verbatim (no re-derivation) so
/// every composition draws from the identical content this file's gating
/// tests already exercise.
#[cfg(feature = "cuda")]
fn build_composition(
    device: &Device,
    base: &Fixture,
    row_indices: &[usize],
) -> (Vec<Vec<u32>>, Vec<usize>, Tensor, Tensor) {
    let seq = base.seq;
    let batch = row_indices.len();
    assert!(batch > 0, "build_composition: empty composition");
    let mut ids_padded = vec![0u32; batch * seq];
    let mut mask_padded = vec![0u32; batch * seq];
    let mut rows = Vec::with_capacity(batch);
    let mut lengths = Vec::with_capacity(batch);
    for (slot, &idx) in row_indices.iter().enumerate() {
        let row = &base.rows[idx];
        for (i, &t) in row.iter().enumerate() {
            ids_padded[slot * seq + i] = t;
            mask_padded[slot * seq + i] = 1;
        }
        rows.push(row.clone());
        lengths.push(base.lengths[idx]);
    }
    let ids = Tensor::from_vec(ids_padded, (batch, seq), device).unwrap();
    let mask = Tensor::from_vec(mask_padded, (batch, seq), device).unwrap();
    (rows, lengths, ids, mask)
}

/// `total_cmp`-folded max over `values` -- the same fixed fold-order/tie-break
/// discipline (family J) `mean_max` in `src/modernbert.rs` uses, re-derived
/// here since this file does not import that module's private helper (see
/// this file's own module doc on why it re-derives its small helpers).
#[cfg(feature = "cuda")]
fn measurement_max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
        if b.total_cmp(&a).is_gt() {
            b
        } else {
            a
        }
    })
}

/// `(mean, total_cmp-folded max)` over `values`. Panics on an empty slice --
/// a measurement sweep that produced zero data points is itself a defect in
/// this test, not a `0.0`/`NaN` result to silently propagate.
#[cfg(feature = "cuda")]
fn measurement_mean_max(values: &[f64]) -> (f64, f64) {
    assert!(!values.is_empty(), "measurement_mean_max: empty slice");
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    (mean, measurement_max(values))
}

/// MEASUREMENT-ONLY, print-only, `#[ignore]`d by default (see this test's
/// own attribute for the exact rationale) -- computes and prints, for every
/// row of every composition in [`MEASUREMENT_COMPOSITION_SWEEP`], the exact
/// three ratios the gating `bf16`-CUDA tests above assert
/// (`ratio_alone_vs_truth`, `ratio_batch_vs_truth`, `ratio_alone_vs_batch`,
/// via the identical [`relative_l1_error`] function those tests call), plus
/// each composition's own red-control separations
/// (row_lengths off-by-one, window radius off-by-one), so the
/// conjunctive-control admissibility check the module doc requires can be
/// derived from THIS SAME run rather than a second pod invocation. Asserts
/// ONLY finiteness on every printed ratio: a `NaN`/`inf` measurement is
/// itself a RED finding here (this test's job is to report reality, not
/// pass), never silently dropped or averaged away.
///
/// Two independent, structural reasons this cannot run inside the
/// CI-hermetic gate: (1) `#[cfg(feature = "cuda")]` means the function does
/// not even exist in a plain `cargo test -p jammi-encoders` build (CI's
/// default, no `--features cuda`); (2) even a `--features cuda` build
/// compiles it but `cargo test`'s documented default behavior skips
/// `#[ignore]`-annotated tests unless `--ignored` (or `--include-ignored`)
/// is passed explicitly -- this test relies on neither a hand-rolled skip
/// check nor a meta-test to enforce that; it is cargo's own, standard
/// semantics.
///
/// Invocation: `cargo test -p jammi-encoders --features cuda --test it \
/// measure_gpu_floors -- --ignored --nocapture` (matches this unit's
/// `cuda_device_or_skip` idiom: silent skip with no device unless
/// `JAMMI_REQUIRE_CUDA` is set, in which case device-acquisition failure
/// panics).
///
/// **Producer self-identification.** Before any per-row line, this test
/// prints ONE `HEADER` line carrying `compute_capability` (via
/// `jammi_kernels::admission::probe_cuda_compute_capability`, the same
/// probe [`gpu_composition_floor`] gates dispatch on), `device_name` (via
/// the sibling `probe_cuda_device_name`, the CUDA driver's own device-name
/// string when this build/arch can query it), and
/// `jammi_encoders_version` (`env!("CARGO_PKG_VERSION")`) -- so a captured
/// log substantiates, from the file alone, which arch/build produced it,
/// rather than relying on an out-of-band pod label. Every per-row/
/// per-composition line after it keeps the pre-existing byte-stable
/// format unchanged (downstream tooling greps those lines).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "measurement-only: prints ratios for pod floor derivation, asserts nothing beyond finiteness"]
fn measure_gpu_floors_print_only() {
    let test_name = "measure_gpu_floors_print_only";
    let Some(device) = cuda_device_or_skip(test_name) else {
        return;
    };

    // Producer self-identification (unit 62 final audit, BLOCK 1): printed
    // ONCE, before any per-row line, so a captured log substantiates which
    // arch/build produced it rather than relying on the invoking human to
    // remember which pod they ran on. Uses the SAME probes
    // `gpu_composition_floor` gates dispatch on
    // (`probe_cuda_compute_capability`) plus its new sibling
    // (`probe_cuda_device_name`) -- see both functions' docs in
    // `jammi_kernels::admission` for the "reads the CONTEXT candle already
    // holds, never binds a fresh one" rationale shared by both probes, and
    // for why a probe failure collapses to `"unknown"` here rather than a
    // panic (this line is identification metadata for a measurement run,
    // not an admission predicate -- family D: report reality, never guess
    // a wrong device). Every subsequent per-row/per-composition line below
    // keeps its existing byte-stable format unchanged.
    {
        use jammi_kernels::admission::{probe_cuda_compute_capability, probe_cuda_device_name};
        let compute_capability = probe_cuda_compute_capability(&device)
            .map(|cap| format!("{}.{}", cap.major, cap.minor))
            .unwrap_or_else(|| "unknown".to_string());
        let device_name = probe_cuda_device_name(&device).unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "{test_name}: HEADER compute_capability={compute_capability} \
             device_name={device_name} jammi_encoders_version={}",
            env!("CARGO_PKG_VERSION"),
        );
    }

    let config = load_config();
    let encoder = build_encoder(&device, DType::BF16, &config);
    let truth_encoder = build_encoder(&Device::Cpu, DType::F32, &config);
    let base = build_fixture(&device);

    let mut all_alone_vs_truth: Vec<f64> = Vec::new();
    let mut all_batch_vs_truth: Vec<f64> = Vec::new();
    let mut all_alone_vs_batch: Vec<f64> = Vec::new();
    let mut all_row_len_control: Vec<f64> = Vec::new();
    let mut all_window_control: Vec<f64> = Vec::new();

    for &composition_id in MEASUREMENT_COMPOSITION_SWEEP.iter() {
        let indices = composition_row_indices(composition_id, base.rows.len());
        let (rows, lengths, ids, mask) = build_composition(&device, &base, &indices);
        let pooled_batch = encoder
            .forward(&ids, &mask)
            .expect("measurement composition forward (bf16 cuda)");
        let pooled_batch: Vec<Vec<f32>> = pooled_batch
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2()
            .unwrap();

        let mut comp_alone_vs_truth = Vec::with_capacity(rows.len());
        let mut comp_batch_vs_truth = Vec::with_capacity(rows.len());
        let mut comp_alone_vs_batch = Vec::with_capacity(rows.len());

        for (slot, row) in rows.iter().enumerate() {
            let alone_bf16 = pooled_alone(&encoder, &device, row);
            let truth = pooled_alone(&truth_encoder, &Device::Cpu, row);
            let ratio_alone_vs_truth = relative_l1_error(&alone_bf16, &truth);
            let ratio_batch_vs_truth = relative_l1_error(&pooled_batch[slot], &truth);
            let ratio_alone_vs_batch = relative_l1_error(&alone_bf16, &pooled_batch[slot]);
            eprintln!(
                "{test_name}: composition={composition_id} row_slot={slot} len={} \
                 alone_vs_truth={ratio_alone_vs_truth:e} batch_vs_truth={ratio_batch_vs_truth:e} \
                 alone_vs_batch={ratio_alone_vs_batch:e}",
                lengths[slot],
            );
            assert!(
                ratio_alone_vs_truth.is_finite(),
                "{test_name}: composition={composition_id} row_slot={slot} ratio_alone_vs_truth \
                 is non-finite ({ratio_alone_vs_truth}) -- a NaN/inf measurement is a RED \
                 finding, not usable data"
            );
            assert!(
                ratio_batch_vs_truth.is_finite(),
                "{test_name}: composition={composition_id} row_slot={slot} ratio_batch_vs_truth \
                 is non-finite ({ratio_batch_vs_truth}) -- a NaN/inf measurement is a RED \
                 finding, not usable data"
            );
            assert!(
                ratio_alone_vs_batch.is_finite(),
                "{test_name}: composition={composition_id} row_slot={slot} ratio_alone_vs_batch \
                 is non-finite ({ratio_alone_vs_batch}) -- a NaN/inf measurement is a RED \
                 finding, not usable data"
            );
            comp_alone_vs_truth.push(ratio_alone_vs_truth);
            comp_batch_vs_truth.push(ratio_batch_vs_truth);
            comp_alone_vs_batch.push(ratio_alone_vs_batch);
        }

        let comp_max_alone_vs_truth = measurement_max(&comp_alone_vs_truth);
        let comp_max_batch_vs_truth = measurement_max(&comp_batch_vs_truth);
        let comp_max_alone_vs_batch = measurement_max(&comp_alone_vs_batch);
        eprintln!(
            "{test_name}: composition={composition_id} MAX OVER {} ROWS: \
             max_alone_vs_truth={comp_max_alone_vs_truth:e} \
             max_batch_vs_truth={comp_max_batch_vs_truth:e} \
             max_alone_vs_batch={comp_max_alone_vs_batch:e}",
            rows.len(),
        );

        all_alone_vs_truth.extend_from_slice(&comp_alone_vs_truth);
        all_batch_vs_truth.extend_from_slice(&comp_batch_vs_truth);
        all_alone_vs_batch.extend_from_slice(&comp_alone_vs_batch);

        // Red control 1 (row_lengths off-by-one), THIS composition's own
        // slot 0: same construction as
        // `pooled_embedding_red_control_row_length_off_by_one_bf16_cuda`
        // above, scoped to this composition's batch.
        let mut mask_mut = vec![0u32; lengths.len() * base.seq];
        let mut ids_mut = vec![0u32; lengths.len() * base.seq];
        for (slot, (row, &len)) in rows.iter().zip(lengths.iter()).enumerate() {
            let take = if slot == 0 {
                len.saturating_sub(1)
            } else {
                len
            };
            for (i, &t) in row.iter().enumerate() {
                ids_mut[slot * base.seq + i] = t;
            }
            for i in 0..take {
                mask_mut[slot * base.seq + i] = 1;
            }
        }
        let ids_mut_t = Tensor::from_vec(ids_mut, (lengths.len(), base.seq), &device).unwrap();
        let mask_mut_t = Tensor::from_vec(mask_mut, (lengths.len(), base.seq), &device).unwrap();
        let pooled_batch_mut = encoder
            .forward(&ids_mut_t, &mask_mut_t)
            .expect("row-length control forward (bf16 cuda)");
        let pooled_batch_mut: Vec<Vec<f32>> = pooled_batch_mut
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2()
            .unwrap();
        let alone0 = pooled_alone(&encoder, &device, &rows[0]);
        let row_len_ratio = relative_l1_error(&alone0, &pooled_batch_mut[0]);
        eprintln!(
            "{test_name}: composition={composition_id} RED_CONTROL row_length_off_by_one \
             ratio={row_len_ratio:e}"
        );
        assert!(
            row_len_ratio.is_finite(),
            "{test_name}: composition={composition_id} row_length_off_by_one control ratio is \
             non-finite ({row_len_ratio}) -- a RED finding, not usable data"
        );
        all_row_len_control.push(row_len_ratio);

        // Red control 2 (window radius off-by-one), THIS composition's
        // longest row -- checked premise mirrors the gating control's own
        // (a composition whose longest row cannot bind the window under
        // BOTH configs is SKIPPED here, printed as such, rather than
        // silently counted as a pass).
        let mut mutant_config = config.clone();
        mutant_config.local_attention = config.local_attention + 2;
        assert_eq!(
            mutant_config.half_window(),
            config.half_window() + 1,
            "checked premise: the mutant config must actually change half_window by exactly 1"
        );
        let (long_slot, &long_len) = lengths
            .iter()
            .enumerate()
            .max_by_key(|&(_, &len)| len)
            .expect("composition is non-empty");
        if long_len >= mutant_config.half_window() + 2 && long_len >= config.half_window() + 2 {
            let encoder_mut = build_encoder(&device, DType::BF16, &mutant_config);
            let alone_long = pooled_alone(&encoder, &device, &rows[long_slot]);
            let pooled_batch_mut_window = encoder_mut
                .forward(&ids, &mask)
                .expect("window control forward (bf16 cuda)");
            let pooled_batch_mut_window: Vec<Vec<f32>> = pooled_batch_mut_window
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2()
                .unwrap();
            let window_ratio = relative_l1_error(&alone_long, &pooled_batch_mut_window[long_slot]);
            eprintln!(
                "{test_name}: composition={composition_id} RED_CONTROL window_radius_off_by_one \
                 row_slot={long_slot} len={long_len} ratio={window_ratio:e}"
            );
            assert!(
                window_ratio.is_finite(),
                "{test_name}: composition={composition_id} window_radius_off_by_one control \
                 ratio is non-finite ({window_ratio}) -- a RED finding, not usable data"
            );
            all_window_control.push(window_ratio);
        } else {
            eprintln!(
                "{test_name}: composition={composition_id} RED_CONTROL window_radius_off_by_one \
                 SKIPPED -- longest row (len={long_len}) does not clear half_window+2 for both \
                 configs in this composition (window_threshold original={}, mutant={})",
                config.half_window() + 2,
                mutant_config.half_window() + 2,
            );
        }
    }

    let (mean_a2t, max_a2t) = measurement_mean_max(&all_alone_vs_truth);
    let (mean_b2t, max_b2t) = measurement_mean_max(&all_batch_vs_truth);
    let (mean_a2b, max_a2b) = measurement_mean_max(&all_alone_vs_batch);
    eprintln!(
        "{test_name}: OVERALL OVER {} COMPOSITIONS / {} ROW-MEASUREMENTS: \
         alone_vs_truth mean={mean_a2t:e} max={max_a2t:e} | batch_vs_truth mean={mean_b2t:e} \
         max={max_b2t:e} | alone_vs_batch mean={mean_a2b:e} max={max_a2b:e} \
         (these bound GPU_TRUTH_DRIFT_BOUND [alone_vs_truth/batch_vs_truth] and \
         gpu_composition_floor [alone_vs_batch] respectively)",
        MEASUREMENT_COMPOSITION_SWEEP.len(),
        all_alone_vs_truth.len(),
    );
    if all_row_len_control.is_empty() {
        eprintln!("{test_name}: row_length_off_by_one control produced ZERO measurements");
    } else {
        let (mean_rl, max_rl) = measurement_mean_max(&all_row_len_control);
        eprintln!(
            "{test_name}: OVERALL row_length_off_by_one control OVER {} COMPOSITIONS: \
             mean={mean_rl:e} max={max_rl:e}",
            all_row_len_control.len(),
        );
    }
    if all_window_control.is_empty() {
        eprintln!(
            "{test_name}: window_radius_off_by_one control had ZERO admissible compositions -- \
             see per-composition SKIPPED lines above"
        );
    } else {
        let (mean_w, max_w) = measurement_mean_max(&all_window_control);
        eprintln!(
            "{test_name}: OVERALL window_radius_off_by_one control OVER {} COMPOSITIONS: \
             mean={mean_w:e} max={max_w:e}",
            all_window_control.len(),
        );
    }

    eprintln!(
        "{test_name}: measurement complete -- these numbers are the pod-derivation input for \
         gpu_composition_floor's EXACT_ARCH_COMPOSITION_FLOOR / SM89_COMPOSITION_FLOOR and \
         GPU_TRUTH_DRIFT_BOUND; folding them into those constants (with safety-margin \
         arithmetic documented there) is a SEPARATE derivation commit \
         (contract docs/plans/62-embedding-surface/CONTRACT.md E4, Step 5), not this test -- \
         this test asserts finiteness only and gates nothing"
    );
}
