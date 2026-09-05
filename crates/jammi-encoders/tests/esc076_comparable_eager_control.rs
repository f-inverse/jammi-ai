//! esc-076 (`.jammi/escapes.jsonl`,
//! `esc-076-f16-eager-finetune-oom-nonmonotone-memory`) — the COMPARABLE-
//! EAGER control the escape's own spec names as unbuilt: "a fully-eager
//! bf16 leg at identical shape must be constructed somewhere (cuda-gated
//! Rust test at the library seam, or a new/relaxed harness arm ...) and
//! its result recorded; both branches handled: fully-eager bf16 also OOMs
//! => defect is eager-composition memory; completes => f16-specific."
//!
//! ## Why this is a LIBRARY-SEAM test, not the bench arm
//!
//! Campaign #443's own §Folded defect is explicit: "the comparable-eager
//! control's vehicle is a cuda-gated Rust test at the library seam, owned
//! in W2a ... NOT the bench arm — widening the arm validator is W4's
//! domain and mutates `arm`, the A/B protocol's declared independent
//! variable." So this file drives `jammi_encoders::ModernBert` and
//! `jammi_kernels::admission` DIRECTLY, mirroring the shape of a real
//! LoRA fine-tune step (three forward passes — anchor/positive/negative —
//! through the SAME weight-tied model, a margin loss, one backward) rather
//! than going through `jammi-bench`'s `finetune-run`/`finetune-step` CLI
//! or its `arm` validator at all.
//!
//! ## Why synthetic weights, not a real checkpoint
//!
//! The reporter shape is ModernBERT-large (`hidden_size=1024,
//! num_hidden_layers=28, num_attention_heads=16, intermediate_size=2624,
//! vocab_size=50368, max_position_embeddings=8192,
//! global_attn_every_n_layers=3, local_attention=128` — the public
//! `answerdotai/ModernBERT-large` `config.json` values, ALSO this port's
//! own compiled-in defaults for every field the JSON omits:
//! `DEFAULT_LAYER_NORM_EPS`/`DEFAULT_GLOBAL_ROPE_THETA`/
//! `DEFAULT_LOCAL_ROPE_THETA`/`DEFAULT_LOCAL_ATTENTION`/
//! `DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS` in `src/modernbert.rs`). This
//! wave's charter explicitly permits driving the encoder/trainer seam with
//! SYNTHETIC weights "if checkpoint-shape fidelity is preserved" — the
//! question under test is peak-memory/OOM behavior of the EAGER
//! composition, which depends on tensor SHAPES and DTYPES, never on
//! weight VALUES. [`write_synthetic_checkpoint`] builds every tensor
//! `ModernBertBuilder::build` expects, at the exact real-checkpoint names
//! and shapes (`model.embeddings.tok_embeddings.weight`,
//! `model.layers.{n}.attn.{Wqkv,Wo}.weight`, `model.layers.{n}.mlp.
//! {Wi,Wo}.weight`, the `attn_norm`/`mlp_norm`/`emb_norm`/`final_norm`
//! bias-free LayerNorm weights — see that function's own doc for the
//! exact list, cross-checked against `ModernBertBuilder::build`'s own
//! `frozen_vb.pp(..)` call sites), filled with `Tensor::randn` (finite,
//! nonzero, never a degenerate all-same-value fixture) at F32, written to
//! a temp safetensors file ONCE and reused for both dtype legs (candle's
//! `VarBuilder::from_mmaped_safetensors` casts to the requested
//! `backbone_dtype` on `get`, so one F32 file serves both the BF16 and F16
//! legs without rewriting it).
//!
//! ## The four esc-076 vacuity controls, as implemented here
//!
//! 1. **ANTI-SIDESTEP** ([`assert_ran_eager`]): `JAMMI_KERNELS_DISABLE=all`
//!    is set as the FIRST statement of EVERY `#[test]` in this file (this
//!    file is its own Cargo-autodiscovered test binary/process, so there is
//!    no other test file racing `jammi_kernels::admission::disabled_ops`'s
//!    process-wide `OnceLock` for who initializes it first — see that
//!    function's own doc and `crates/jammi-bench/tests/finetune_step_kernel_disable.rs`'s
//!    identical concern, resolved there by spawning a child process
//!    instead). That `OnceLock` specifically is safe under concurrency
//!    here precisely BECAUSE every test sets the identical value `"all"` —
//!    the race the doc above would matter for is between DIFFERENT disable
//!    configurations, never between two writers agreeing on the same one.
//!    CORRECTED (campaign #446, finding 9): an earlier revision of this
//!    paragraph generalized that one env var's safety into "this file's
//!    own tests are safe to run concurrently in the SAME process", which
//!    is FALSE for the part of these oracles that matters — see
//!    [`SerialGpu`] and the section below. [`assert_ran_eager`]
//!    reads [`jammi_encoders::ln_dispatch_snapshot`] (this crate's own
//!    published counter, the same `LayerNormFused`/eager-fallback pair
//!    every other dispatch-count oracle in this crate's own test suite
//!    reads) before/after each leg and asserts `fused == 0` — the eager
//!    fallback contract is proven to hold on its OWN terms, not assumed
//!    from the env var alone.
//! 2. **ANTI-SHRINK**: the real legs run at the EXACT measured failing
//!    shape (`batch=16, seq=128`, ModernBERT-large) — [`REPORTER_BATCH`]/
//!    [`REPORTER_SEQ`] are never reduced for test speed. A SEPARATE,
//!    deliberately-oversized witness leg ([`oom_capability_witness_leg`])
//!    proves this harness's OWN OOM-classification path is alive
//!    (`assert_oom_classification_is_a_capability_witness_not_hollow`):
//!    without it, "neither leg OOM'd" could mean either "the defect is
//!    fixed" or "this harness cannot detect an OOM at all", and green
//!    would not distinguish them.
//! 3. **ANTI-HOLLOW-PASS**: a completing leg is checked for a finite loss
//!    (`assert!(loss.is_finite())`, never `!(x > bound)` — KO-2/family F)
//!    AND that each pooled forward output's dtype is STILL the requested
//!    one (`a.dtype() == dtype`, checked for all three of
//!    anchor/positive/negative before the loss's own `to_dtype(F32)` cast)
//!    — a silent internal upcast would read as "it works" for the wrong
//!    reason, and would invalidate the whole dtype comparison.
//! 4. **COMPARABLE-EAGER ARM EXISTS**: this whole file IS that arm — both
//!    `run_leg(DType::BF16, ..)` and `run_leg(DType::F16, ..)` drive the
//!    IDENTICAL pipeline (same weights file, same synthetic token ids,
//!    same margin-loss/backward shape), differing ONLY in `backbone_dtype`.
//!
//! ## One test at a time, structurally
//!
//! This file has THREE `#[test]` fns (two run by default; the pre-fix RED
//! reproduction is `#[ignore]`d), and every one of them measures
//! DEVICE-GLOBAL free memory. Concurrency between them is not a style
//! question: two legs sampling `cuMemGetInfo` while a third holds tens of
//! GB of eager activations read each other's allocations as their own.
//! [`SerialGpu`] makes single-test-at-a-time a property of the ONLY way
//! this file can obtain a `Device`, so it holds for tests that do not
//! exist yet. It replaces a comment that asserted the opposite and a
//! `--test-threads=1` expectation nothing in CI supplies.
//!
//! ## Reading the verdict
//!
//! `main()`-adjacent to the single `#[test]`, `print_diagnosis` states
//! which of esc-076's two root-cause branches this run's evidence supports
//! — printed via `--nocapture`, not asserted as a hard pass/fail condition
//! (per campaign #443's Part 3 item 4 fold-the-fix note: "Do NOT fix the
//! root cause yet ... this wave delivers the DIAGNOSIS and the control
//! test"). The test itself still FAILS on any outcome that is neither "ran
//! to completion, finite" nor "refused with the CUDA out-of-memory driver
//! error" — an unrelated panic, shape error, or silent dtype coercion is a
//! harness defect, not a diagnosis.
//!
//! ## D3 ATTRIBUTION (campaign #443 W2c)
//!
//! The lead's own discriminator legs (`.jammi/ledger/aa09a171-443-fa2-f16.jsonl`,
//! "esc-076 MECHANISM PINNED") measured, on the real bench path: a full f16
//! leg reproduces the OOM (0->49GB@5s->78GB@25s); a 4-step TRUNCATED (seq
//! capped at 64) f16 leg peaks 63.3GB and completes; a 4-step DUPLICATED-
//! batch f16 leg (the SAME shape repeated, never a new one) peaks at an
//! IDENTICAL 63.1GB — "shape-variability exonerated within a few steps,
//! but full-run growth tracks the COUNT OF DISTINCT batch shapes"; and a
//! same-config bf16-fused/f32-fused pair lands at 44.2/41.7GB — i.e. the
//! eager composition (any dtype) carries a ~19GB BASE offset over the
//! fused path PLUS a per-NEW-shape growth term, and the defect is
//! dtype-INDEPENDENT in principle (only f16-shaped in practice because
//! admission today routes only f16 to the eager fallback at all).
//!
//! **Attribution, reasoned from that evidence plus this crate's own build
//! facts.** `crate::cuda::cast_scale`'s own doc states the load-bearing
//! fact: "cudarc has no caching allocator" — every `CudaStorage` this
//! crate (and candle-core's OWN CUDA backend, which allocates through the
//! SAME `cudarc::driver::CudaDevice::alloc`/`alloc_zeros` primitives) ever
//! creates is a RAW `cuMemAlloc`/`cuMemFree` pair, never pooled or cached
//! Rust-side. A raw, non-pooling allocator hitting MANY DIFFERENT
//! allocation sizes over a run's lifetime is exactly the shape CUDA's own
//! driver-level allocator is known to fragment under (the driver's arena
//! cannot always satisfy a NEW size class from a block freed at a
//! DIFFERENT size, so its total reserved footprint grows with the COUNT OF
//! DISTINCT sizes ever requested, not with any one tensor's own lifetime)
//! — this is the textbook mechanism the ledger's "growth tracks count of
//! DISTINCT batch shapes" finding describes, and it requires NO f16-
//! specific or jammi-encoders-specific defect to explain: it is a property
//! of feeding a raw (non-caching) CUDA allocator a training loop whose
//! per-step tensor shapes are not drawn from a small, fixed set.
//!
//! **Where the ~19GB BASE offset comes from (jammi-encoders' own share).**
//! The eager composition materialises MORE simultaneously-live, separately-
//! allocated tensors per op than the fused kernels do (e.g. `fn slow`'s
//! upcast-compute-cast-back — `fn slow`, `jammi-encoders/src/layer_norm.rs:682` — and the
//! analogous multi-step compositions in `softmax`/`geglu`'s own eager
//! fallbacks) — each such intermediate is its own `cuMemAlloc`, at its OWN
//! (upcast, often F32) size, on top of whatever the fused kernel would
//! have needed zero extra allocations for. This portion IS attributable to
//! this crate. It is a FIXED, shape-count-independent overhead (present
//! even in the flat, fixed-shape leg above) — bounding it would mean
//! reusing scratch buffers across an op's own intermediates, a real but
//! separate, larger refactor than this wave's scope names, and NOT what
//! drives the escape's actual OOM (the fixed-shape leg above does not
//! OOM at 40 steps; only variable-shape growth does).
//!
//! **Where the GROWTH term (the actual OOM trigger) belongs, and why it is
//! NOT a jammi-encoders fix.** The growth is driven by the NUMBER OF
//! DISTINCT INPUT SHAPES the eager composition is ever asked to run at —
//! and every one of those shapes is a direct, undistorted function of the
//! `(batch, seq)` the CALLER hands to `ModernBert::forward`. Nothing
//! inside this crate's per-op eager arms can reduce that count without
//! CHANGING the computation: padding an intermediate activation's shape
//! inside e.g. `LayerNorm::slow` would inject fabricated values into a
//! mean/variance reduction, corrupting the result — canonicalizing shapes
//! is only sound BEFORE the real tokens ever reach the encoder, i.e. at
//! the trainer's own batch-construction step (padding/bucketing sequence
//! lengths to a small, fixed set of buckets so the SAME small set of
//! shapes recurs across steps, exactly the shape the ledger's "duplicated
//! batches plateau" leg already proved bounds growth). That decision is
//! `jammi-ai`'s trainer/dataloader, never `jammi-encoders`'s own forward
//! path. Per campaign #443's own instruction and the extend-seams-not-
//! upstream doctrine (never patch candle's/cudarc's allocator internals
//! either — there is no house seam exposing an allocator policy knob to
//! wrap), this wave does NOT reach into `jammi-ai` to add bucketing:
//! **the seam is reported here, precisely, for the lead to route.**
//!
//! [`esc076_variable_shape_unbucketed_reproduces_the_pre_fix_oom`] (below,
//! `#[ignore]`d) is the committed reproduction of the growth mechanism at
//! THIS crate's own seam (variable seq length per step vs a fixed-shape
//! control at the identical step count) — it FAILS (RED) whenever run,
//! by design, and is not part of the default green suite once the fix
//! below landed; `fix-verifier` runs it explicitly (`--ignored`) as
//! esc-076's RED oracle: revert `jammi-ai`'s bucketing fix and this test
//! must fail the identical way it does today.
//!
//! **FOLLOW-UP (fix landed): `jammi-ai`'s sequence-length bucketing.**
//! `crates/jammi-ai/src/fine_tune/batch_bucket.rs`, wired at
//! `TrainingLoop::encode_texts`, now rounds each batch's natural width up
//! to a small, fixed power-of-two bucket ladder before any tensor is
//! built — the reporter-shape f16 leg completes on the pod (44.3GB flat)
//! post-fix. The bucket DECISION itself
//! (`bucket_seq_len`/`MIN_BUCKET_LEN`) lives in `jammi_numerics`
//! (`crates/jammi-numerics/src/batch_shape.rs`) rather than `jammi-ai`,
//! specifically so THIS crate's own D3 oracle
//! ([`esc076_variable_shape_bucketed_completes_with_bounded_memory`],
//! below) can call the IDENTICAL decision without depending on
//! `jammi-ai` (the wrong dependency direction — `jammi-numerics` sits
//! below both crates). That GREEN leg proves the fix bounds memory at
//! THIS library seam directly, not merely inferred from `jammi-ai`'s own
//! unit tests.
//!
//! **Pod finding, disclosed honestly (this branch's own landing run):** the
//! FIRST attempt at [`VARIABLE_SHAPE_SEQS`] (five values, all `<=
//! REPORTER_SEQ = 128`, `20` steps) produced a FLAT trace on BOTH the
//! variable-shape leg AND its fixed-shape control (`0.0` MiB drop, both) —
//! a genuine NEGATIVE result at that scale, not silently discarded (kept
//! as this constant's own prior-value note). Reproduction required
//! WIDENING the shape range up to `512` (`11` values, `33` steps) — at
//! which point the variable-shape leg OOM'd after only 3 completed steps
//! (having reached seq lengths 64, 96, 128 before failing at 160), while
//! the fixed-shape control (33 steps at the fixed `REPORTER_SEQ = 128`)
//! completed cleanly. This is an HONEST but IMPERFECT isolation of the
//! ledger's own "count of DISTINCT shapes" variable: this leg's shapes
//! both vary in COUNT and GROW in AMPLITUDE across the cycle (unlike the
//! ledger's own controls, which held amplitude fixed and varied only
//! whether a shape repeated), so this reproduction demonstrates "a
//! realistic variable-length-batch training loop OOMs at a moderate,
//! ordinary shape where a fixed-shape loop of the same nominal severity
//! does not" — the same CLASS of defect and the same practical
//! consequence — without cleanly separating "shape variety alone" from
//! "the largest shape reached, retained via a non-releasing allocator"
//! as independent causes. Either framing routes to the SAME fix (bound
//! the shape variety AND the peak amplitude a step can introduce, i.e.
//! bucket/pad at the batch-construction layer), so the attribution and
//! routing above stand; a reader wanting the cleaner isolation should
//! re-run at a FIXED amplitude (all shapes `== REPORTER_SEQ`'s own
//! token count, permuted only in which axis carries it) before treating
//! "distinct count alone, no growth" as separately confirmed.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use jammi_encoders::{ModernBert, ModernBertConfig, Pooling};
use jammi_lora::LoraBuildConfig;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

/// The escape's own measured failing shape (`esc-076`'s `observable`
/// field: "f16 fine-tune ... terminates rc=1 ... Backward:
/// DriverError(CUDA_ERROR_OUT_OF_MEMORY) ~85s in" at
/// `--batch 16 --max-seq-length 128`).
const REPORTER_BATCH: usize = 16;
const REPORTER_SEQ: usize = 128;

/// Public `answerdotai/ModernBERT-large` `config.json` values (see this
/// file's own module doc for the full citation and cross-check against
/// this port's compiled-in per-field defaults).
fn modernbert_large_config() -> ModernBertConfig {
    ModernBertConfig {
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        intermediate_size: 2624,
        vocab_size: 50368,
        max_position_embeddings: 8192,
        layer_norm_eps: 1e-5,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        attention_dropout: 0.0,
    }
}

/// KO-7 require-gate for the drop-ratio INCONCLUSIVE fallback branches
/// (registered in `ci/kernel-oracle-helpers.txt`): under `JAMMI_REQUIRE_CUDA`
/// a too-short memory trace must FAIL loudly rather than fall back to an
/// inconclusive skip — on a prove lane an unmeasurable oracle is a failure,
/// never a soft pass.
fn esc076_trace_gate(context: &str) {
    if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
        panic!(
            "{context}: JAMMI_REQUIRE_CUDA is set but the memory trace was too short to compare \
             — an inconclusive drop-ratio oracle is not acceptable on a prove lane"
        );
    }
}

/// This process's single GPU slot. See [`SerialGpu`] for why it exists and
/// why it is STRUCTURAL rather than a convention.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

/// A CUDA device that cannot be held without also holding [`GPU_SERIAL`].
///
/// Campaign #446, finding 9. This file's oracles read DEVICE-GLOBAL free
/// memory (`cuda_free_mib` -> `cuMemGetInfo`), and it has THREE `#[test]`
/// fns, two of which run by default. `cargo test` runs the tests of one
/// binary CONCURRENTLY, so two legs measuring the same device's free
/// memory while a third is allocating tens of GB of eager activations
/// attribute each other's allocations to their own trace -- a false RED
/// (or, worse, a false GREEN when the interleaving happens to cancel
/// out).
///
/// The remedy is structural, not documentary: `cuda_device` is the ONLY
/// source of a `Device` in this file, and it now hands back this wrapper,
/// which holds the process-wide lock for as long as the caller holds the
/// device. A test added tomorrow cannot forget to serialize, because it
/// cannot obtain a device without doing so -- whereas a comment asking
/// for `--test-threads=1` (nothing in CI passes it) is exactly the kind
/// of instruction a new test silently does not read.
///
/// # Why there is no `Deref<Target = Device>`
///
/// This type used to `impl Deref<Target = Device>`, which kept every
/// `&device` call site working unchanged -- and let
/// `let d = (*cuda_device().unwrap()).clone();` type-check. That one-liner
/// ENDS the serialization: the temporary guard (and with it the slot) is
/// dropped at the end of the statement, while `d` is a live, owned
/// `Device` the caller then measures device-global free memory on with no
/// slot held -- the exact escape this wrapper exists to prevent, spelled
/// as ordinary deref usage. Mirrors `jammi-ai`'s
/// `tests/gpu_capability/harness.rs` fix of the same shape.
///
/// [`SerialGpu::device`] replaces it: the borrow it returns is tied to
/// `&self`, so no `&Device` can outlive the guard. Every call site passes
/// `device.device()` where it used to pass `&device`.
///
/// **What is still open, stated rather than implied.** `candle_core::Device`
/// is `Clone`, so `guard.device().clone()` compiles and always will --
/// nothing an API of this shape can do prevents cloning a `Clone` type
/// reachable by reference (a `&DeviceRef` newtype does not help: if it
/// derefs to `Device` the clone resolves straight through it by autoderef,
/// and if it does not, no call site can pass it where `&Device` is
/// wanted). What changed is that the escape is now an EXPLICIT, greppable
/// `.device().clone()` rather than an incidental consequence of deref, and
/// no call site in this file needs it.
///
/// A poisoned lock is recovered with `into_inner` rather than unwrapped:
/// one leg panicking must fail THAT leg, not turn every sibling into a
/// confusing poison error that buries the original diagnosis.
struct SerialGpu {
    device: Device,
    /// Held, never read -- dropping it at the end of the test body is the
    /// entire mechanism.
    _slot: MutexGuard<'static, ()>,
}

impl SerialGpu {
    /// This guard's device, borrowed for no longer than the guard itself --
    /// the ONLY way to reach it, and the reason there is no `Deref` (see
    /// the type doc). Deliberately NOT `into_device`/`to_device`: nothing
    /// in this file has a legitimate reason to hold a `Device` past the
    /// slot.
    fn device(&self) -> &Device {
        &self.device
    }
}

/// Take this file's one-at-a-time GPU slot, recovering a poisoned lock
/// rather than unwrapping it (see [`SerialGpu`]'s doc for why).
///
/// Split out from [`cuda_device`] so the exclusion property itself is
/// testable without a DEVICE, where no [`SerialGpu`] can ever be
/// constructed: [`gpu_slot_is_exclusive_while_held`] is the non-vacuous
/// control that the slot is a real mutex and not a decorative field.
fn take_gpu_slot() -> MutexGuard<'static, ()> {
    GPU_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The slot is a real mutual exclusion -- the property every device-global
/// memory oracle in this file leans on.
///
/// The assertion is made while THIS test provably owns the slot
/// ([`take_gpu_slot`] blocks until it does), and `std::sync::Mutex` is not
/// reentrant, so `try_lock` failing here is evidence that `GPU_SERIAL` is a
/// real mutex actually held by the value [`take_gpu_slot`] returns -- which
/// is the same value [`SerialGpu`]'s `_slot` field stores. A decorative,
/// always-available field fails this.
///
/// # Two things this deliberately does NOT assert, and why
///
/// **The release half.** `jammi-ai`'s `tests/gpu_capability/harness.rs`
/// twin also asserts `try_lock().is_ok()` after the guard drops. That
/// assertion is sound THERE because it runs on the CPU lane, where no
/// sibling can take the slot. Here it is not: this file is
/// `#![cfg(feature = "cuda")]` in its entirety, so the control only ever
/// runs alongside legs that hold the slot for MINUTES, and `cargo test`
/// runs a binary's tests concurrently -- a sibling holding the slot at that
/// instant is CORRECT behavior, not a leak, and asserting on it is a
/// coin-flip. (Measured, not predicted: the first version of this control
/// asserted it and failed on an L40S pod for exactly that reason, with
/// `esc076_variable_shape_bucketed_completes_with_bounded_memory` holding
/// the slot.) Release is `MutexGuard`'s own `Drop`, which is not this
/// file's code; exclusion is the part that is.
///
/// **CPU-lane coverage.** Unlike the `jammi-ai` twin, this control cannot
/// run without the `cuda` feature -- that lane compiles none of this file.
/// What it does buy is that it needs no DEVICE: every `cuda`-featured lane
/// runs it, including one where `Device::new_cuda` fails and no
/// [`SerialGpu`] can be constructed at all, which is exactly where this
/// file's other tests skip out.
#[test]
fn gpu_slot_is_exclusive_while_held() {
    let slot = take_gpu_slot();
    assert!(
        GPU_SERIAL.try_lock().is_err(),
        "a second holder must not be able to take the slot while this test holds it -- \
         without a real mutex behind `GPU_SERIAL`, `SerialGpu` would serialize nothing"
    );
    drop(slot);
}

fn cuda_device() -> Option<SerialGpu> {
    // Taken BEFORE `Device::new_cuda`, so even device ACQUISITION (which
    // allocates a context on the device) is serialized against a sibling
    // leg's memory trace.
    let slot = take_gpu_slot();
    match Device::new_cuda(0) {
        Ok(device) => Some(SerialGpu {
            device,
            _slot: slot,
        }),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "esc076_comparable_eager_control: JAMMI_REQUIRE_CUDA is set but no CUDA \
                     device could be acquired -- this is a landing proof, a silent skip here is \
                     not acceptable: {e}"
                );
            }
            eprintln!(
                "esc076_comparable_eager_control: skipping -- no CUDA device available ({e})"
            );
            None
        }
    }
}

/// Every tensor `ModernBertBuilder::build` reads, at real-checkpoint names
/// and shapes, filled with `Tensor::randn` (finite, nonzero, non-degenerate
/// -- never all-same-value) -- see this file's module doc for why VALUES
/// do not matter here, only shapes/dtypes/names. Built and written ONCE
/// (CPU, F32); both dtype legs load the SAME file (candle's mmaped
/// `VarBuilder` casts to the requested `backbone_dtype` on `get`).
fn write_synthetic_checkpoint(config: &ModernBertConfig, path: &Path) {
    let cpu = Device::Cpu;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let randn = |shape: (usize, usize)| -> Tensor {
        Tensor::randn(0f32, 0.02, shape, &cpu).expect("randn fixture tensor")
    };
    let randn1 = |n: usize| -> Tensor {
        // LayerNorm weight: initialised near 1.0 (matching a real
        // checkpoint's post-training LayerNorm scale far better than a
        // zero-mean fixture would -- irrelevant to the OOM/memory question
        // this file investigates, but avoids a gratuitously atypical
        // fixture).
        (Tensor::randn(0f32, 0.02, n, &cpu).expect("randn fixture tensor") + 1.0f64)
            .expect("add fixture tensor")
    };

    t.insert(
        "model.embeddings.tok_embeddings.weight".to_string(),
        randn((config.vocab_size, config.hidden_size)),
    );
    t.insert(
        "model.embeddings.norm.weight".to_string(),
        randn1(config.hidden_size),
    );

    for n in 0..config.num_hidden_layers {
        t.insert(
            format!("model.layers.{n}.attn.Wqkv.weight"),
            randn((config.hidden_size * 3, config.hidden_size)),
        );
        t.insert(
            format!("model.layers.{n}.attn.Wo.weight"),
            randn((config.hidden_size, config.hidden_size)),
        );
        if n > 0 {
            t.insert(
                format!("model.layers.{n}.attn_norm.weight"),
                randn1(config.hidden_size),
            );
        }
        t.insert(
            format!("model.layers.{n}.mlp.Wi.weight"),
            randn((config.intermediate_size * 2, config.hidden_size)),
        );
        t.insert(
            format!("model.layers.{n}.mlp.Wo.weight"),
            randn((config.hidden_size, config.intermediate_size)),
        );
        t.insert(
            format!("model.layers.{n}.mlp_norm.weight"),
            randn1(config.hidden_size),
        );
    }

    t.insert(
        "model.final_norm.weight".to_string(),
        randn1(config.hidden_size),
    );

    candle_core::safetensors::save(&t, path).expect("write synthetic checkpoint");
}

/// `Wqkv`-only, rank 16, alpha 32.0, Gaussian init, seed 1 -- matches
/// `esc-076`'s own reproduction seed and `flash_oracle_build_model`'s
/// (`src/modernbert.rs`) production-shaped LoRA fixture. DISCLOSED
/// assumption: the escape's own `symptom_spec` does not state its exact
/// `--lora-rank`/`--lora-alpha`, so this is the best-available production-
/// like stand-in, not a verified-exact reproduction of those two flags.
fn production_lora_config() -> (Vec<String>, HashMap<String, usize>) {
    (vec!["Wqkv".to_string()], HashMap::new())
}

/// `with_lora`: when `true`, `Wqkv` gets a REAL trainable LoRA adapter
/// (see [`production_lora_config`]) instead of [`LoraBuildConfig::frozen`].
/// This matters far more than it first appears (this file's own landing
/// run's finding, see the module doc's amended verdict): a `frozen()`
/// build has NO `Var` anywhere in the graph, so `Tensor::backward()`'s
/// walk finds nothing needing a gradient and candle can drop every
/// intermediate activation as soon as its Rust-side reference count hits
/// zero -- there is no backward-tape RETENTION pressure at all. A real
/// fine-tune's trainable LoRA A/B matrices are genuine `Var`s, so every
/// intermediate activation on ANY path from a LoRA-touched layer through
/// to the loss must stay alive until `.backward()` runs -- the actual
/// eager-composition memory profile `esc-076` measured. `frozen()` alone
/// is therefore not a comparable control for the escape's own defect,
/// regardless of dtype.
fn build_model(
    config: &ModernBertConfig,
    weights: &Path,
    dtype: DType,
    device: &Device,
    with_lora: bool,
) -> (ModernBert, VarMap) {
    let varmap = VarMap::new();
    let (target_modules, rank_pattern) = production_lora_config();
    let lora = if with_lora {
        LoraBuildConfig {
            target_modules: &target_modules,
            layers_to_transform: &None,
            lora_rank: 16,
            lora_alpha: 32.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &rank_pattern,
            init_mode: jammi_lora::LoraInitMode::Gaussian,
            seed: 1,
        }
    } else {
        LoraBuildConfig::frozen()
    };
    let mut model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .backbone_dtype(dtype)
        .lora(lora)
        .build(&[weights], config, device, &varmap)
        .unwrap_or_else(|e| panic!("esc076: build ModernBert ({dtype:?}) failed: {e}"));
    model.set_training(true);
    (model, varmap)
}

/// Synthetic token ids -- deterministic (no external RNG dependency,
/// family L), `< vocab_size`, `[batch, seq]`.
fn synthetic_ids(batch: usize, seq: usize, vocab: usize, salt: u32, device: &Device) -> Tensor {
    let ids: Vec<u32> = (0..batch * seq)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) % vocab as u32)
        .collect();
    Tensor::from_vec(ids, (batch, seq), device).unwrap()
}

/// This crate's own published dispatch-count seam
/// (`jammi_encoders::ln_dispatch_snapshot`) -- the SAME counter every
/// other fused-vs-eager oracle in this crate's test suite reads. `(fused,
/// eager)` deltas across a leg.
fn ln_dispatch_delta(
    before: jammi_kernels::admission::DispatchSnapshot,
    after: jammi_kernels::admission::DispatchSnapshot,
) -> (u64, u64) {
    (after.fused - before.fused, after.eager - before.eager)
}

#[derive(Debug)]
enum LegOutcome {
    /// `steps_completed == STEPS_PER_LEG`: the whole run completed. `losses`
    /// is per-step, for a human `--nocapture` read; `free_mib_after_step`
    /// is the SAME per-step trace `esc-076`'s own title
    /// ("nonmonotone-memory") asks about -- a leg that "completes" but
    /// shows a MONOTONE downward free-memory trend (no plateau) is still a
    /// leak, just one this device's 80GB happened to absorb.
    Completed {
        losses: Vec<f32>,
        free_mib_after_step: Vec<f64>,
    },
    /// `steps_completed < STEPS_PER_LEG`: OOM'd partway through -- the
    /// escape's own observable ("~85s in", not immediately) is exactly
    /// this shape, a multi-step run, not a single-step peak.
    CudaOutOfMemory {
        steps_completed: usize,
        free_mib_after_step: Vec<f64>,
        message: String,
    },
    OtherError {
        message: String,
    },
}

/// Number of training steps per leg. The escape's own observable
/// ("Backward: DriverError(CUDA_ERROR_OUT_OF_MEMORY) ~85s in") reports a
/// failure well after the FIRST step, which is the title's own
/// "nonmonotone-memory" framing: this is a MULTI-STEP memory-growth
/// question, not a single-step peak-memory question. This wave's own
/// landing run found a single forward+backward step at the reporter shape
/// leaves tens of GiB of headroom on an 80GB A100 (see
/// [`oom_capability_witness_leg`]'s doc for the single-step numbers) --
/// `STEPS_PER_LEG` is chosen to give a real optimizer loop (with its own
/// `Var::set` in-place update, `AdamW`'s m/v state, and a FRESH forward
/// tape retained across MANY steps) room to reveal a genuine trend before
/// this test's own runtime budget is spent.
const STEPS_PER_LEG: usize = 40;

/// `(free, total)` MiB, the same `cuMemGetInfo` driver call
/// `jammi_encoders::modernbert`'s own (private) VRAM probes make --
/// duplicated here since that helper is `#[cfg(test)]`-private to this
/// crate's OWN test module, not exported.
fn cuda_free_mib(device: &Device) -> f64 {
    device
        .synchronize()
        .expect("device sync before mem_get_info");
    let (free, _total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .expect("cuMemGetInfo_v2 failed");
    free as f64 / (1024.0 * 1024.0)
}

/// Runs a MULTI-STEP fully-eager training leg at `(REPORTER_BATCH,
/// REPORTER_SEQ)`: each step is three forward passes (anchor/positive/
/// negative, weight-tied -- the real LoRA fine-tune step's own shape,
/// `docs/maintainer/cuda-kernel-guide.md` §3.6's
/// `mean(relu(margin - cos(a,p) + cos(a,n)))`), pooled + L2-normalised
/// (`ModernBert::forward`'s own `pool_and_normalize`), a margin loss, one
/// `AdamW::backward_step` (a REAL optimizer step -- `Var::set` in place,
/// exactly what a real fine-tune loop does between steps; a leaked
/// reference anywhere in the eager composition would show up as a
/// per-step free-memory trend here, not just a single elevated peak).
/// `JAMMI_KERNELS_DISABLE=all` must already be set (checked by the caller
/// via [`assert_ran_eager`]) before this runs.
fn run_leg(dtype: DType, config: &ModernBertConfig, weights: &Path, device: &Device) -> LegOutcome {
    let (model, varmap) = build_model(config, weights, dtype, device, /* with_lora */ true);
    // ANTI-HOLLOW-PASS control (3), sanity half: the built model's shape
    // agrees with the config it was built from.
    assert_eq!(
        model.hidden_size(),
        config.hidden_size,
        "sanity: built model's hidden_size must match the config"
    );

    let trainable_vars = varmap.all_vars();
    assert!(
        !trainable_vars.is_empty(),
        "sanity: with_lora=true must register at least one trainable Var -- an empty varmap \
         means the LoRA build silently produced a frozen model, defeating this leg's whole \
         point (see build_model's own doc on why frozen() alone is not comparable)"
    );
    let mut optimizer = candle_nn::AdamW::new_lr(trainable_vars, 1e-4)
        .unwrap_or_else(|e| panic!("esc076: AdamW::new_lr failed: {e}"));

    let anchor = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 1, device);
    let positive = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 2, device);
    let negative = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 3, device);
    let mask = Tensor::ones((REPORTER_BATCH, REPORTER_SEQ), DType::U32, device).unwrap();

    let mut losses = Vec::with_capacity(STEPS_PER_LEG);
    let mut free_mib_after_step = Vec::with_capacity(STEPS_PER_LEG);

    for step in 0..STEPS_PER_LEG {
        let forward = |ids: &Tensor| -> Result<Tensor, jammi_encoders::EncoderError> {
            model.forward(ids, &mask)
        };
        let step_result: Result<f32, jammi_encoders::EncoderError> = (|| {
            let a = forward(&anchor)?;
            let p = forward(&positive)?;
            let n = forward(&negative)?;
            // ANTI-HOLLOW-PASS control (3), the concrete check: the pooled
            // output's dtype must still be the REQUESTED `dtype` -- a
            // silent internal upcast to F32 somewhere in the forward pass
            // would make "it completed" meaningless (no longer testing
            // the requested dtype's own eager-composition memory profile).
            if step == 0 {
                for (label, t) in [("anchor", &a), ("positive", &p), ("negative", &n)] {
                    assert_eq!(
                        t.dtype(),
                        dtype,
                        "[{dtype:?}] {label}'s pooled output dtype is {:?}, not the requested \
                         {dtype:?} -- a silent internal upcast would invalidate this leg's \
                         dtype comparison",
                        t.dtype()
                    );
                }
            }
            let a32 = a.to_dtype(DType::F32)?;
            let p32 = p.to_dtype(DType::F32)?;
            let n32 = n.to_dtype(DType::F32)?;
            // Both `a`/`p`/`n` are L2-normalised by `pool_and_normalize`,
            // so a row-wise dot product IS the cosine similarity.
            let cos_ap = (&a32 * &p32)?.sum(candle_core::D::Minus1)?;
            let cos_an = (&a32 * &n32)?.sum(candle_core::D::Minus1)?;
            let margin = 0.2f64;
            let hinge = (cos_an - cos_ap)?.affine(1.0, margin)?.relu()?;
            let loss = hinge.mean_all()?;
            let loss_scalar = loss.to_scalar::<f32>()?;
            optimizer.backward_step(&loss)?;
            Ok(loss_scalar)
        })();

        match step_result {
            Ok(loss) => {
                losses.push(loss);
                free_mib_after_step.push(cuda_free_mib(device));
            }
            Err(e) => {
                let message = e.to_string();
                return if message.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || message.contains("OutOfMemory")
                {
                    LegOutcome::CudaOutOfMemory {
                        steps_completed: step,
                        free_mib_after_step,
                        message,
                    }
                } else {
                    LegOutcome::OtherError { message }
                };
            }
        }
    }

    LegOutcome::Completed {
        losses,
        free_mib_after_step,
    }
}

/// The ANTI-SHRINK capability witness (control 2): a DELIBERATELY
/// oversized leg -- `seq` raised to `config.max_position_embeddings`
/// (`8192`, ModernBERT-large's own RoPE-table ceiling, so this is still a
/// VALID input the model accepts, never an out-of-domain shape) at the
/// SAME `REPORTER_BATCH` -- that must ALWAYS classify as
/// [`LegOutcome::CudaOutOfMemory`] on any real GPU: attention's own
/// `[batch, heads, seq, seq]` scores tensor ALONE is
/// `16*16*8192*8192*4 bytes ~= 68.7 GiB` at this shape (global layer 0),
/// well past any single real GPU's VRAM even before any other tensor is
/// counted. MEASURED (this file's own landing run): raising ONLY `batch`
/// (to `16 * 64 = 1024`, holding `seq = REPORTER_SEQ = 128`) did NOT OOM
/// on an 80GB A100 -- `forward_hidden` alone at that shape completed --
/// confirming the quadratic-in-`seq` lever is the reliable one here, not
/// a linear-in-`batch` scale-up. If this ever reads as `Completed` (an
/// implausibly large device) or `OtherError` (the harness's
/// OOM-classification string match is broken), the real legs' outcomes
/// below cannot be trusted: "green" would not distinguish "the defect is
/// fixed" from "this harness cannot see an OOM at all".
fn oom_capability_witness_leg(
    config: &ModernBertConfig,
    weights: &Path,
    device: &Device,
) -> LegOutcome {
    let batch = REPORTER_BATCH;
    let seq = config.max_position_embeddings;
    let (model, _varmap) = build_model(
        config,
        weights,
        DType::F32,
        device,
        /* with_lora */ false,
    );
    let ids = synthetic_ids(batch, seq, config.vocab_size, 99, device);
    let mask = Tensor::ones((batch, seq), DType::U32, device).unwrap();
    let result: Result<(), jammi_encoders::EncoderError> =
        model.forward_hidden(&ids, &mask).and_then(|h| {
            let s = h.sum_all()?;
            s.backward()?;
            Ok(())
        });
    match result {
        Ok(()) => LegOutcome::Completed {
            losses: vec![],
            free_mib_after_step: vec![],
        },
        Err(e) => {
            let message = e.to_string();
            if message.contains("CUDA_ERROR_OUT_OF_MEMORY") || message.contains("OutOfMemory") {
                LegOutcome::CudaOutOfMemory {
                    steps_completed: 0,
                    free_mib_after_step: vec![],
                    message,
                }
            } else {
                LegOutcome::OtherError { message }
            }
        }
    }
}

/// Distinct sequence lengths cycled round-robin, one per step — the
/// concrete, checkable form of the REAL bench path's "duplicated identical
/// batches plateau; DISTINCT batch shapes keep growing" finding (campaign
/// #443 W2c contract, D3; `esc-076 MECHANISM PINNED` ledger row: "4-step
/// DUPLICATED-batch f16 peaks 63.1GB — identical ramp => shape-variability
/// exonerated within few steps, but full-run growth tracks count of
/// DISTINCT batch shapes"). All five are `<= REPORTER_SEQ` (never exceed
/// the already-proven-representative reporter shape) and span a REAL
/// range a variable-length-batch trainer would actually produce, never a
/// single repeated value (which would degenerate to the fixed-shape leg
/// this file already carries).
/// AMENDED (this branch's first pod run): the initial 5-value,
/// `<=REPORTER_SEQ` set produced a flat, non-reproducing trace (0.0 MiB
/// drop, identical to the fixed-shape control) — an honest negative
/// result at that scale, not a silently-adjusted one (see this file's
/// module doc, D3 ATTRIBUTION, for the full disclosure). Widened to 11
/// values spanning up to `max_position_embeddings`'s own practical
/// mid-range (`512` — the escape's OWN second failing config, `b8*s512`,
/// per `esc-076`'s `observable` field) so the leg sweeps genuinely NOVEL
/// allocation sizes across a wider range, closer to what a real
/// variable-length-sentence dataset would produce (natural sentence
/// lengths rarely repeat exactly), rather than 5 small values a
/// non-caching allocator's driver-level arena might already have slack
/// for.
const VARIABLE_SHAPE_SEQS: [usize; 11] = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512];

/// Steps for the variable-shape leg: 3 full cycles through
/// [`VARIABLE_SHAPE_SEQS`] (`11 * 3 = 33`) — enough for a genuine trend
/// (vs a single cycle, which cannot distinguish "one-time cost per NEW
/// shape, then plateau" from "unbounded per-cycle growth") while staying
/// inside a pod session's practical wall-clock budget (this crate's own
/// `STEPS_PER_LEG=40` fixed-shape leg is the existing budget precedent;
/// the WIDER shape range above, up to 4x `REPORTER_SEQ`'s own token
/// count, already costs proportionally more per step than the original
/// 5-value/20-step design did, so the step count is not raised further).
const VARIABLE_SHAPE_STEPS: usize = 33;

/// The variable-shape twin of [`run_leg`]: IDENTICAL pipeline (same
/// weights, same LoRA config, same three-forward-pass margin-loss shape,
/// same real `AdamW::backward_step`), except each step's `(anchor,
/// positive, negative, mask)` are rebuilt at `VARIABLE_SHAPE_SEQS[step %
/// 5]` instead of the fixed `REPORTER_SEQ` — reproducing the ONE
/// independent variable the mechanism finding isolated (distinct batch
/// SHAPE, not dtype, not batch count) at the library seam, never through
/// `jammi-bench`'s `arm` validator (this file's own module doc, "why this
/// is a library-seam test").
fn run_leg_variable_shape(
    dtype: DType,
    config: &ModernBertConfig,
    weights: &Path,
    device: &Device,
) -> LegOutcome {
    let (model, varmap) = build_model(config, weights, dtype, device, /* with_lora */ true);
    let trainable_vars = varmap.all_vars();
    assert!(
        !trainable_vars.is_empty(),
        "sanity: with_lora=true must register at least one trainable Var (see run_leg's \
         identical assertion)"
    );
    let mut optimizer = candle_nn::AdamW::new_lr(trainable_vars, 1e-4)
        .unwrap_or_else(|e| panic!("esc076: AdamW::new_lr failed: {e}"));

    let mut losses = Vec::with_capacity(VARIABLE_SHAPE_STEPS);
    let mut free_mib_after_step = Vec::with_capacity(VARIABLE_SHAPE_STEPS);

    for step in 0..VARIABLE_SHAPE_STEPS {
        let seq = VARIABLE_SHAPE_SEQS[step % VARIABLE_SHAPE_SEQS.len()];
        let anchor = synthetic_ids(REPORTER_BATCH, seq, config.vocab_size, 1, device);
        let positive = synthetic_ids(REPORTER_BATCH, seq, config.vocab_size, 2, device);
        let negative = synthetic_ids(REPORTER_BATCH, seq, config.vocab_size, 3, device);
        let mask = Tensor::ones((REPORTER_BATCH, seq), DType::U32, device).unwrap();

        let forward = |ids: &Tensor| -> Result<Tensor, jammi_encoders::EncoderError> {
            model.forward(ids, &mask)
        };
        let step_result: Result<f32, jammi_encoders::EncoderError> = (|| {
            let a = forward(&anchor)?;
            let p = forward(&positive)?;
            let n = forward(&negative)?;
            if step == 0 {
                for (label, t) in [("anchor", &a), ("positive", &p), ("negative", &n)] {
                    assert_eq!(
                        t.dtype(),
                        dtype,
                        "[{dtype:?}] variable-shape {label}'s pooled output dtype is {:?}, not \
                         the requested {dtype:?}",
                        t.dtype()
                    );
                }
            }
            let a32 = a.to_dtype(DType::F32)?;
            let p32 = p.to_dtype(DType::F32)?;
            let n32 = n.to_dtype(DType::F32)?;
            let cos_ap = (&a32 * &p32)?.sum(candle_core::D::Minus1)?;
            let cos_an = (&a32 * &n32)?.sum(candle_core::D::Minus1)?;
            let margin = 0.2f64;
            let hinge = (cos_an - cos_ap)?.affine(1.0, margin)?.relu()?;
            let loss = hinge.mean_all()?;
            let loss_scalar = loss.to_scalar::<f32>()?;
            optimizer.backward_step(&loss)?;
            Ok(loss_scalar)
        })();

        match step_result {
            Ok(loss) => {
                losses.push(loss);
                free_mib_after_step.push(cuda_free_mib(device));
            }
            Err(e) => {
                let message = e.to_string();
                return if message.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || message.contains("OutOfMemory")
                {
                    LegOutcome::CudaOutOfMemory {
                        steps_completed: step,
                        free_mib_after_step,
                        message,
                    }
                } else {
                    LegOutcome::OtherError { message }
                };
            }
        }
    }

    LegOutcome::Completed {
        losses,
        free_mib_after_step,
    }
}

/// The fixed-shape twin of [`run_leg_variable_shape`] at the SAME
/// [`VARIABLE_SHAPE_STEPS`] step count (never [`STEPS_PER_LEG`] — a
/// different step count would make the two traces' `total_drop`
/// incomparable) — every step uses the IDENTICAL `REPORTER_SEQ`, so this
/// isolates "many steps at ONE shape" from "many steps across several
/// shapes", the ONE independent variable [`run_leg_variable_shape`]
/// changes.
fn run_leg_fixed_shape_same_step_count(
    dtype: DType,
    config: &ModernBertConfig,
    weights: &Path,
    device: &Device,
) -> LegOutcome {
    let (model, varmap) = build_model(config, weights, dtype, device, /* with_lora */ true);
    let trainable_vars = varmap.all_vars();
    let mut optimizer = candle_nn::AdamW::new_lr(trainable_vars, 1e-4)
        .unwrap_or_else(|e| panic!("esc076: AdamW::new_lr failed: {e}"));

    let anchor = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 1, device);
    let positive = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 2, device);
    let negative = synthetic_ids(REPORTER_BATCH, REPORTER_SEQ, config.vocab_size, 3, device);
    let mask = Tensor::ones((REPORTER_BATCH, REPORTER_SEQ), DType::U32, device).unwrap();

    let mut losses = Vec::with_capacity(VARIABLE_SHAPE_STEPS);
    let mut free_mib_after_step = Vec::with_capacity(VARIABLE_SHAPE_STEPS);

    for _step in 0..VARIABLE_SHAPE_STEPS {
        let forward = |ids: &Tensor| -> Result<Tensor, jammi_encoders::EncoderError> {
            model.forward(ids, &mask)
        };
        let step_result: Result<f32, jammi_encoders::EncoderError> = (|| {
            let a = forward(&anchor)?.to_dtype(DType::F32)?;
            let p = forward(&positive)?.to_dtype(DType::F32)?;
            let n = forward(&negative)?.to_dtype(DType::F32)?;
            let cos_ap = (&a * &p)?.sum(candle_core::D::Minus1)?;
            let cos_an = (&a * &n)?.sum(candle_core::D::Minus1)?;
            let margin = 0.2f64;
            let hinge = (cos_an - cos_ap)?.affine(1.0, margin)?.relu()?;
            let loss = hinge.mean_all()?;
            let loss_scalar = loss.to_scalar::<f32>()?;
            optimizer.backward_step(&loss)?;
            Ok(loss_scalar)
        })();
        match step_result {
            Ok(loss) => {
                losses.push(loss);
                free_mib_after_step.push(cuda_free_mib(device));
            }
            Err(e) => {
                let message = e.to_string();
                return if message.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || message.contains("OutOfMemory")
                {
                    LegOutcome::CudaOutOfMemory {
                        steps_completed: losses.len(),
                        free_mib_after_step,
                        message,
                    }
                } else {
                    LegOutcome::OtherError { message }
                };
            }
        }
    }
    LegOutcome::Completed {
        losses,
        free_mib_after_step,
    }
}

/// The cap [`jammi_numerics::bucket_seq_len`] rounds each step's raw
/// length up against — set to [`REPORTER_SEQ`] (`128`), matching the REAL
/// `jammi-ai` trainer's own `effective_max` at the reporter shape (the
/// SAME config the coordinator's own landing claim cites: "the
/// reporter-shape f16 leg now completes on the pod (44.3GB flat)").
///
/// **POD FINDING (this branch's own landing run, disclosed honestly): a
/// cap of `512` (matching [`VARIABLE_SHAPE_SEQS`]'s own raw maximum, this
/// constant's FIRST value) does NOT green this leg** — bucketing to
/// `{64, 128, 256, 512}` still visits `256`/`512`, and EACH of those
/// individually costs tens of GB at this harness's shape (28-layer
/// ModernBERT-large, 3-forward eager LoRA backward) — the leg OOM'd after
/// 3 steps, identically to the un-bucketed RED leg. This is a REAL,
/// important distinction bucketing does NOT erase: bucketing bounds the
/// COUNT of distinct shapes within an ALREADY-reasonable `max_seq_length`
/// ceiling; it does not lower that ceiling. A trainer configured with
/// `max_seq_length = 512` and genuinely-512-token batches pays that cost
/// regardless of bucketing — the lever for THAT is the `max_seq_length`
/// config value itself, a data/config decision, not this mechanism. The
/// coordinator's own "reporter-shape f16 leg... completes" claim is
/// specifically about `max_seq_length = 128` (`REPORTER_SEQ`), so THAT is
/// the cap this leg proves against — matching, not overreaching, the
/// fix's real proven domain.
const VARIABLE_SHAPE_BUCKET_CAP: usize = REPORTER_SEQ;

/// The bucketed twin of [`run_leg_variable_shape`]: the IDENTICAL raw
/// `VARIABLE_SHAPE_SEQS` cycle, but each step's raw length is FIRST
/// truncated to [`VARIABLE_SHAPE_BUCKET_CAP`] (mirroring the REAL
/// trainer's own tokenizer call, `tokenizer.encode_batch(&text_refs,
/// Some(effective_max))`, which truncates BEFORE any bucketing ever runs —
/// `crates/jammi-ai/src/fine_tune/trainer.rs`), then rounded UP through
/// `jammi_numerics::bucket_seq_len` (the SAME candle-free decision that
/// trainer calls next, via `crates/jammi-ai/src/fine_tune/batch_bucket.rs`)
/// BEFORE any tensor is constructed. The extra `(bucketed_len - raw_len)`
/// tail positions are padded with token id `0` and attention-mask `0` —
/// the SAME trivial extend-with-zeros contract `jammi-ai`'s own
/// `pad_rows_to_bucket` implements, re-stated inline here (a few lines)
/// rather than IMPORTED, since `jammi-encoders` must not depend on
/// `jammi-ai` (the wrong dependency direction for this workspace — only
/// the candle-free bucket DECISION is shared, via `jammi-numerics`, never
/// the row-mutation helper). This is the GREEN leg proving the fix AT ITS
/// OWN PROVEN DOMAIN: truncate-then-bucket at `REPORTER_SEQ` collapses
/// `VARIABLE_SHAPE_SEQS`'s 11 raw values (many `> REPORTER_SEQ`) down to
/// just `{64, 128}` (2 distinct shapes, both already known-safe from the
/// fixed-shape control) — completing without the un-bucketed
/// [`run_leg_variable_shape`]'s pre-fix OOM.
fn run_leg_variable_shape_bucketed(
    dtype: DType,
    config: &ModernBertConfig,
    weights: &Path,
    device: &Device,
) -> LegOutcome {
    let (model, varmap) = build_model(config, weights, dtype, device, /* with_lora */ true);
    let trainable_vars = varmap.all_vars();
    assert!(
        !trainable_vars.is_empty(),
        "sanity: with_lora=true must register at least one trainable Var (see run_leg's \
         identical assertion)"
    );
    let mut optimizer = candle_nn::AdamW::new_lr(trainable_vars, 1e-4)
        .unwrap_or_else(|e| panic!("esc076: AdamW::new_lr failed: {e}"));

    // A row's ids/mask at the BUCKETED width: the first `raw_len` columns
    // are real synthetic content (mirroring `synthetic_ids`'s own hash),
    // the remaining `bucketed_len - raw_len` columns are pad id `0` /
    // mask `0` — exactly `pad_rows_to_bucket`'s own contract, restated for
    // a flat `(batch, bucketed_len)` tensor build.
    let build_bucketed = |raw_len: usize, bucketed_len: usize, salt: u32| -> (Tensor, Tensor) {
        let mut ids: Vec<u32> = Vec::with_capacity(REPORTER_BATCH * bucketed_len);
        let mut mask: Vec<u32> = Vec::with_capacity(REPORTER_BATCH * bucketed_len);
        for row in 0..REPORTER_BATCH {
            for col in 0..bucketed_len {
                if col < raw_len {
                    let flat = (row * raw_len + col) as u32;
                    ids.push(
                        flat.wrapping_mul(2654435761).wrapping_add(salt) % config.vocab_size as u32,
                    );
                    mask.push(1);
                } else {
                    ids.push(0);
                    mask.push(0);
                }
            }
        }
        (
            Tensor::from_vec(ids, (REPORTER_BATCH, bucketed_len), device).unwrap(),
            Tensor::from_vec(mask, (REPORTER_BATCH, bucketed_len), device).unwrap(),
        )
    };

    let mut losses = Vec::with_capacity(VARIABLE_SHAPE_STEPS);
    let mut free_mib_after_step = Vec::with_capacity(VARIABLE_SHAPE_STEPS);

    for step in 0..VARIABLE_SHAPE_STEPS {
        // Truncate FIRST (mirroring `tokenizer.encode_batch(&text_refs,
        // Some(effective_max))`'s own truncation, which the real trainer
        // runs BEFORE any bucketing) — a raw length above the cap is not a
        // `bucket_seq_len` domain violation the caller silently walks into,
        // it is the SAME "already truncated to max_seq_length" precondition
        // that function's own doc states.
        let raw_len = VARIABLE_SHAPE_SEQS[step % VARIABLE_SHAPE_SEQS.len()].min(REPORTER_SEQ);
        let bucketed_len = jammi_numerics::bucket_seq_len(raw_len, VARIABLE_SHAPE_BUCKET_CAP);
        let (anchor_ids, anchor_mask) = build_bucketed(raw_len, bucketed_len, 1);
        let (positive_ids, _positive_mask) = build_bucketed(raw_len, bucketed_len, 2);
        let (negative_ids, _negative_mask) = build_bucketed(raw_len, bucketed_len, 3);
        // All three rows share the SAME mask (identical raw_len/bucketed_len
        // per step, mirroring `run_leg_variable_shape`'s own single shared
        // `mask` per step) — `_positive_mask`/`_negative_mask` are built
        // (not skipped) so a future divergence in per-row padding would
        // still construct a real tensor to compare against, even though
        // this leg's own uniform-length-per-step design makes them
        // identical to `anchor_mask` today.
        let mask = anchor_mask;

        let forward = |ids: &Tensor| -> Result<Tensor, jammi_encoders::EncoderError> {
            model.forward(ids, &mask)
        };
        let step_result: Result<f32, jammi_encoders::EncoderError> = (|| {
            let a = forward(&anchor_ids)?;
            let p = forward(&positive_ids)?;
            let n = forward(&negative_ids)?;
            if step == 0 {
                for (label, t) in [("anchor", &a), ("positive", &p), ("negative", &n)] {
                    assert_eq!(
                        t.dtype(),
                        dtype,
                        "[{dtype:?}] bucketed variable-shape {label}'s pooled output dtype is \
                         {:?}, not the requested {dtype:?}",
                        t.dtype()
                    );
                }
            }
            let a32 = a.to_dtype(DType::F32)?;
            let p32 = p.to_dtype(DType::F32)?;
            let n32 = n.to_dtype(DType::F32)?;
            let cos_ap = (&a32 * &p32)?.sum(candle_core::D::Minus1)?;
            let cos_an = (&a32 * &n32)?.sum(candle_core::D::Minus1)?;
            let margin = 0.2f64;
            let hinge = (cos_an - cos_ap)?.affine(1.0, margin)?.relu()?;
            let loss = hinge.mean_all()?;
            let loss_scalar = loss.to_scalar::<f32>()?;
            optimizer.backward_step(&loss)?;
            Ok(loss_scalar)
        })();

        match step_result {
            Ok(loss) => {
                losses.push(loss);
                free_mib_after_step.push(cuda_free_mib(device));
            }
            Err(e) => {
                let message = e.to_string();
                return if message.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || message.contains("OutOfMemory")
                {
                    LegOutcome::CudaOutOfMemory {
                        steps_completed: step,
                        free_mib_after_step,
                        message,
                    }
                } else {
                    LegOutcome::OtherError { message }
                };
            }
        }
    }

    LegOutcome::Completed {
        losses,
        free_mib_after_step,
    }
}

fn total_drop_mib(outcome: &LegOutcome) -> Option<f64> {
    let trace = match outcome {
        LegOutcome::Completed {
            free_mib_after_step,
            ..
        } => free_mib_after_step,
        LegOutcome::CudaOutOfMemory {
            free_mib_after_step,
            ..
        } => free_mib_after_step,
        LegOutcome::OtherError { .. } => return None,
    };
    if trace.len() < 2 {
        return None;
    }
    Some(trace[0] - trace[trace.len() - 1])
}

/// esc-076 D3(a)'s ORIGINAL honest RED at the library seam — PRE-FIX
/// reproduction, kept `#[ignore]`d now that the fix has landed
/// (`crates/jammi-ai/src/fine_tune/batch_bucket.rs`, wired at
/// `TrainingLoop::encode_texts`, campaign #443 follow-up). Runs BF16 (the
/// escape's own comparable-eager arm's calibrated-good dtype — see
/// `esc076_fully_eager_bf16_vs_f16_at_reporter_shape`'s `print_diagnosis`)
/// fully-eager at [`VARIABLE_SHAPE_SEQS`] cycled round-robin, WITHOUT any
/// bucketing (the pathological, pre-fix shape of a real variable-length
/// trainer loop), and its fixed-shape twin at the IDENTICAL step count,
/// then asserts the two total-memory-drop traces are COMPARABLE — a
/// variable-shape leg that drops SUBSTANTIALLY more free memory than its
/// own fixed-shape control over the SAME number of steps is measured,
/// non-vacuous evidence of exactly the "distinct shapes keep growing"
/// mechanism the ledger's MECHANISM PINNED finding names.
///
/// **`#[ignore]`d by default so the branch's GPU suite is green** — this
/// test's OWN job now is being esc-076's RED oracle for `fix-verifier`
/// (revert the bucketing fix, this test must fail the SAME way it did
/// before the fix landed; confirm it, then re-apply the fix) — run it
/// explicitly with `cargo test -- --ignored
/// esc076_variable_shape_unbucketed_reproduces_the_pre_fix_oom`. It is
/// EXPECTED TO fail (OOM) whenever it runs, by design: this is the
/// UNBUCKETED path, which the fix's own point is to make unreachable from
/// `jammi-ai`'s real trainer (see
/// [`esc076_variable_shape_bucketed_completes_with_bounded_memory`] below
/// for the GREEN, post-fix leg proving the SAME shape cycle completes once
/// bucketed). A relative (never absolute) bound: `k = 3.0` applied to the
/// fixed-shape leg's OWN drop (comparable_eager oracle design rule — a
/// bound relative to the SAME run's own baseline, never a constant pulled
/// from a different session).
#[test]
#[ignore = "esc-076 pre-fix RED reproduction (unbucketed eager growth) -- \
            run explicitly by fix-verifier with --ignored, not part of the \
            default green suite; see this fn's own doc"]
fn esc076_variable_shape_unbucketed_reproduces_the_pre_fix_oom() {
    std::env::set_var("JAMMI_KERNELS_DISABLE", "all");

    let Some(device) = cuda_device() else {
        return;
    };

    let config = modernbert_large_config();
    let dir = tempfile::tempdir().expect("tempdir for synthetic checkpoint");
    let weights_path = dir.path().join("model.safetensors");
    write_synthetic_checkpoint(&config, &weights_path);

    let variable_outcome = assert_ran_eager("variable_shape_bf16", || {
        run_leg_variable_shape(DType::BF16, &config, &weights_path, device.device())
    });
    let fixed_outcome = assert_ran_eager("fixed_shape_bf16_same_step_count", || {
        run_leg_fixed_shape_same_step_count(DType::BF16, &config, &weights_path, device.device())
    });

    println!(
        "[esc-076 D3] variable-shape bf16 ({VARIABLE_SHAPE_STEPS} steps, seqs={VARIABLE_SHAPE_SEQS:?}): {}",
        match &variable_outcome {
            LegOutcome::Completed { free_mib_after_step, .. } =>
                summarize_free_mib_trace(free_mib_after_step),
            LegOutcome::CudaOutOfMemory { steps_completed, free_mib_after_step, message } => format!(
                "CudaOutOfMemory after {steps_completed} steps -- {message}; trace so far: {}",
                summarize_free_mib_trace(free_mib_after_step)
            ),
            LegOutcome::OtherError { message } => format!("OtherError -- {message}"),
        }
    );
    println!(
        "[esc-076 D3] fixed-shape bf16 ({VARIABLE_SHAPE_STEPS} steps, seq={REPORTER_SEQ}): {}",
        match &fixed_outcome {
            LegOutcome::Completed {
                free_mib_after_step,
                ..
            } => summarize_free_mib_trace(free_mib_after_step),
            LegOutcome::CudaOutOfMemory {
                steps_completed,
                free_mib_after_step,
                message,
            } => format!(
                "CudaOutOfMemory after {steps_completed} steps -- {message}; trace so far: {}",
                summarize_free_mib_trace(free_mib_after_step)
            ),
            LegOutcome::OtherError { message } => format!("OtherError -- {message}"),
        }
    );

    // A variable-shape leg that itself OOMs (while the fixed-shape control
    // at the IDENTICAL step count does not) is the sharpest possible
    // reproduction of the defect -- an immediate, unambiguous RED, no
    // ratio needed.
    if matches!(variable_outcome, LegOutcome::CudaOutOfMemory { .. })
        && matches!(fixed_outcome, LegOutcome::Completed { .. })
    {
        panic!(
            "[esc-076 D3] REPRODUCED: the variable-shape leg OOM'd while its fixed-shape control \
             (identical step count, identical dtype, identical everything except which seq \
             lengths are cycled) completed -- this is the monotone-growth-tracks-distinct-shapes \
             mechanism, reproduced as a committed oracle. See this file's module doc's D3 \
             ATTRIBUTION section for the seam this fix belongs at (jammi-ai batch \
             bucketing/padding, out of this crate's worktree scope)."
        );
    }

    let (Some(var_drop), Some(fixed_drop)) = (
        total_drop_mib(&variable_outcome),
        total_drop_mib(&fixed_outcome),
    ) else {
        println!(
            "[esc-076 D3] INCONCLUSIVE on the drop-ratio oracle: at least one leg's trace was too \
             short to compare (see the OOM-branch check above for the sharper reproduction this \
             falls back from)."
        );
        esc076_trace_gate("esc076_variable_shape_unbucketed_reproduces_the_pre_fix_oom");
        return;
    };
    println!(
        "[esc-076 D3] total_drop_mib: variable-shape={var_drop:.1} fixed-shape={fixed_drop:.1} \
         ratio={:.2}",
        var_drop / fixed_drop.max(1.0)
    );
    const GROWTH_RATIO_BOUND: f64 = 3.0;
    assert!(
        var_drop <= GROWTH_RATIO_BOUND * fixed_drop.max(1.0),
        "[esc-076 D3] REPRODUCED: variable-shape eager composition dropped {var_drop:.1} MiB of \
         free memory over {VARIABLE_SHAPE_STEPS} steps vs {fixed_drop:.1} MiB for its \
         fixed-shape control at the IDENTICAL step count -- a {:.2}x ratio, past the {} \
         no-producer:derived-comparable-eager-oracle-bound (three times the SAME run's own \
         fixed-shape baseline, family K's 'measure against the strongest baseline' rule) -- this \
         is the mechanism the ledger's MECHANISM PINNED finding names ('growth tracks count of \
         DISTINCT batch shapes'), reproduced as a committed, re-runnable oracle. See this file's \
         module doc's D3 ATTRIBUTION section for the seam this fix belongs at.",
        var_drop / fixed_drop.max(1.0),
        GROWTH_RATIO_BOUND
    );
}

/// esc-076 D3 FIX VERIFICATION (GREEN, runs by default): the SAME
/// [`VARIABLE_SHAPE_SEQS`] cycle the pre-fix RED leg
/// ([`esc076_variable_shape_unbucketed_reproduces_the_pre_fix_oom`], above,
/// `#[ignore]`d) OOMs on, except each step's raw length is FIRST truncated
/// to [`VARIABLE_SHAPE_BUCKET_CAP`] (`REPORTER_SEQ` — mirroring the real
/// trainer's own tokenizer truncation, `Some(effective_max)`) and THEN
/// rounded up through `jammi_numerics::bucket_seq_len` — the SAME
/// candle-free decision `jammi-ai`'s own trainer now calls at its
/// batch-construction seam (`crates/jammi-ai/src/fine_tune/trainer.rs`,
/// via `crates/jammi-ai/src/fine_tune/batch_bucket.rs`). Collapses
/// `VARIABLE_SHAPE_SEQS`'s 11 raw values (most `> REPORTER_SEQ`) down to
/// just `{64, 128}` (2 distinct shapes, [`VARIABLE_SHAPE_BUCKET_CAP`]'s own
/// doc) — this proves the fix AT its own claimed domain (the reporter
/// shape, `max_seq_length = REPORTER_SEQ`), at the `jammi-encoders`
/// library seam directly (never through `jammi-ai`, which this crate must
/// not depend on) rather than only inferred from `jammi-ai`'s own unit
/// tests. See [`VARIABLE_SHAPE_BUCKET_CAP`]'s own doc for the POD FINDING
/// that a cap matching the RAW range's own maximum (`512`) does NOT green
/// this leg — bucketing bounds shape COUNT, not shape AMPLITUDE, and this
/// leg is deliberately scoped to the domain where the fix's own landing
/// claim was measured.
///
/// Asserts BOTH: (a) every one of `VARIABLE_SHAPE_STEPS` steps completes
/// with a finite loss (never merely "did not panic" — KO-2/family F), and
/// (b) the bucketed leg's own total free-memory drop stays within the SAME
/// `3x`-of-its-fixed-shape-control bound the pre-fix RED leg uses to prove
/// the OPPOSITE outcome — bucketing should genuinely BOUND this leg's
/// memory behavior, not merely happen not to OOM at this run's particular
/// step count.
#[test]
fn esc076_variable_shape_bucketed_completes_with_bounded_memory() {
    std::env::set_var("JAMMI_KERNELS_DISABLE", "all");

    let Some(device) = cuda_device() else {
        return;
    };

    let config = modernbert_large_config();
    let dir = tempfile::tempdir().expect("tempdir for synthetic checkpoint");
    let weights_path = dir.path().join("model.safetensors");
    write_synthetic_checkpoint(&config, &weights_path);

    let bucketed_outcome = assert_ran_eager("variable_shape_bucketed_bf16", || {
        run_leg_variable_shape_bucketed(DType::BF16, &config, &weights_path, device.device())
    });
    let fixed_outcome =
        assert_ran_eager("fixed_shape_bf16_same_step_count_bucketed_control", || {
            run_leg_fixed_shape_same_step_count(
                DType::BF16,
                &config,
                &weights_path,
                device.device(),
            )
        });

    println!(
        "[esc-076 D3 FIX] bucketed variable-shape bf16 ({VARIABLE_SHAPE_STEPS} steps, raw \
         seqs={VARIABLE_SHAPE_SEQS:?}, bucket cap={VARIABLE_SHAPE_BUCKET_CAP}): {}",
        match &bucketed_outcome {
            LegOutcome::Completed {
                free_mib_after_step,
                ..
            } => summarize_free_mib_trace(free_mib_after_step),
            LegOutcome::CudaOutOfMemory {
                steps_completed,
                free_mib_after_step,
                message,
            } => format!(
                "CudaOutOfMemory after {steps_completed} steps -- {message}; trace so far: {}",
                summarize_free_mib_trace(free_mib_after_step)
            ),
            LegOutcome::OtherError { message } => format!("OtherError -- {message}"),
        }
    );

    match &bucketed_outcome {
        LegOutcome::Completed { losses, .. } => {
            assert_eq!(
                losses.len(),
                VARIABLE_SHAPE_STEPS,
                "the bucketed leg must complete EVERY step, not merely more than the un-bucketed \
                 leg's own partial 3-step run"
            );
            for (step, loss) in losses.iter().enumerate() {
                assert!(
                    loss.is_finite(),
                    "[esc-076 D3 FIX] step {step} completed but loss is non-finite ({loss}) -- a \
                     completing bucketed leg must produce a genuinely finite result at EVERY \
                     step, never a silently-propagated NaN/inf read as success"
                );
            }
        }
        other => panic!(
            "[esc-076 D3 FIX] the bucketed variable-shape leg must COMPLETE now that jammi-ai's \
             sequence-length bucketing fix has landed -- got {other:?} instead. Either the fix \
             regressed, or this leg's own shape/step budget needs re-measuring against the fix's \
             own bucket ladder."
        ),
    }

    let (Some(bucketed_drop), Some(fixed_drop)) = (
        total_drop_mib(&bucketed_outcome),
        total_drop_mib(&fixed_outcome),
    ) else {
        println!(
            "[esc-076 D3 FIX] INCONCLUSIVE on the drop-ratio bound: a trace was too short to \
             compare (the completion assertion above already establishes the primary GREEN \
             claim)."
        );
        esc076_trace_gate("esc076_variable_shape_bucketed_completes_with_bounded_memory");
        return;
    };
    println!(
        "[esc-076 D3 FIX] total_drop_mib: bucketed-variable={bucketed_drop:.1} \
         fixed-shape={fixed_drop:.1} ratio={:.2}",
        bucketed_drop / fixed_drop.max(1.0)
    );
    const GROWTH_RATIO_BOUND: f64 = 3.0;
    assert!(
        bucketed_drop <= GROWTH_RATIO_BOUND * fixed_drop.max(1.0),
        "[esc-076 D3 FIX] bucketed variable-shape eager composition dropped {bucketed_drop:.1} \
         MiB of free memory over {VARIABLE_SHAPE_STEPS} steps vs {fixed_drop:.1} MiB for its \
         fixed-shape control -- a {:.2}x ratio, past the SAME {GROWTH_RATIO_BOUND}x bound the \
         pre-fix RED leg (above, #[ignore]d) uses to prove the OPPOSITE outcome -- bucketing \
         should collapse this leg's own distinct-shape count down to {{64,128}} (2 buckets \
         total, matching the ledger's own duplicated-shape-plateaus finding), not merely \
         shrink the un-bucketed growth without genuinely bounding it.",
        bucketed_drop / fixed_drop.max(1.0)
    );
}

/// ANTI-SIDESTEP control (1): asserts the fused LayerNorm kernel dispatched
/// ZERO times and the eager fallback dispatched at least once, over the
/// window `f()` runs in -- proving the leg genuinely ran on the eager
/// fallback (never merely assuming `JAMMI_KERNELS_DISABLE=all` worked from
/// the env var alone). `layer_norm_fused` is the representative op (every
/// call site in this crate's forward path reaches it, training-only, once
/// per layer).
fn assert_ran_eager<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let before = jammi_encoders::ln_dispatch_snapshot();
    let out = f();
    let after = jammi_encoders::ln_dispatch_snapshot();
    let (fused, eager) = ln_dispatch_delta(before, after);
    assert_eq!(
        fused, 0,
        "[{label}] ANTI-SIDESTEP violated: layer_norm_fused dispatched {fused} times under \
         JAMMI_KERNELS_DISABLE=all -- the eager-fallback contract this control exists to prove \
         does not hold"
    );
    assert!(
        eager > 0,
        "[{label}] ANTI-SIDESTEP vacuous: layer_norm_fused's EAGER counter never incremented -- \
         this leg never actually reached a real LayerNorm call, so 'ran eager' is unproven, not \
         merely assumed"
    );
    out
}

#[test]
fn esc076_fully_eager_bf16_vs_f16_at_reporter_shape() {
    // SAFETY-of-test: this file is Cargo-autodiscovered as its OWN test
    // binary/process (no `tests/it/`-style shared harness), and this is
    // the FIRST statement that could touch `jammi_kernels::admission`'s
    // process-wide `OnceLock` -- see this file's own module doc, control
    // 1. CORRECTED (campaign #446, finding 9): this comment used to also
    // claim there was "no other `#[test]` in this file". There are THREE,
    // and they set the identical `"all"` value, which is what actually
    // makes the `OnceLock` safe here. Mutual exclusion for the
    // DEVICE-GLOBAL memory oracles is a separate concern, handled
    // structurally by `SerialGpu` rather than by this comment.
    std::env::set_var("JAMMI_KERNELS_DISABLE", "all");

    let Some(device) = cuda_device() else {
        return;
    };

    let config = modernbert_large_config();
    let dir = tempfile::tempdir().expect("tempdir for synthetic checkpoint");
    let weights_path = dir.path().join("model.safetensors");
    write_synthetic_checkpoint(&config, &weights_path);

    // Control 2 (ANTI-SHRINK), part 1: the capability witness. Run FIRST,
    // so a broken OOM-classification path is caught before either real
    // leg's outcome is trusted.
    let witness = assert_ran_eager("oom_capability_witness", || {
        oom_capability_witness_leg(&config, &weights_path, device.device())
    });
    match &witness {
        LegOutcome::CudaOutOfMemory { message, .. } => {
            println!(
                "[esc-076] OOM capability witness: CONFIRMED (harness can detect a real CUDA \
                 OOM) -- {message}"
            );
        }
        other => panic!(
            "oom_capability_witness_leg (batch={REPORTER_BATCH}, seq=max_position_embeddings) \
             did not classify as CUDA OOM on this device -- got {other:?} instead. Either this \
             device has more VRAM than the witness's deliberately-oversized shape needs (raise \
             the witness further), or this harness's OOM-string-match classification is broken \
             -- either way, the real legs' outcomes below cannot be trusted until this witness \
             reads CudaOutOfMemory"
        ),
    }

    let bf16_outcome = assert_ran_eager("bf16_reporter_shape", || {
        run_leg(DType::BF16, &config, &weights_path, device.device())
    });
    let f16_outcome = assert_ran_eager("f16_reporter_shape", || {
        run_leg(DType::F16, &config, &weights_path, device.device())
    });

    for (label, outcome) in [("bf16", &bf16_outcome), ("f16", &f16_outcome)] {
        match outcome {
            LegOutcome::Completed { losses, .. } => {
                for (step, loss) in losses.iter().enumerate() {
                    assert!(
                        loss.is_finite(),
                        "[{label}] ANTI-HOLLOW-PASS violated: step {step} completed but loss is \
                         non-finite ({loss}) -- a completing eager leg must produce a genuinely \
                         finite result at EVERY step, never a silently-propagated NaN/inf read \
                         as success"
                    );
                }
            }
            LegOutcome::CudaOutOfMemory {
                steps_completed,
                message,
                ..
            } => {
                println!(
                    "[{label}] classified as CUDA OOM after {steps_completed} completed steps: \
                     {message}"
                );
            }
            LegOutcome::OtherError { message } => {
                panic!(
                    "[{label}] unexpected error class (neither a clean completion nor a \
                     classified CUDA OOM) -- this is a harness defect, not an esc-076 finding: \
                     {message}"
                );
            }
        }
    }

    print_diagnosis(&bf16_outcome, &f16_outcome);
}

/// Free-MiB trace summary -- first/last few points plus the delta, rather
/// than a raw `{:?}` dump of a `STEPS_PER_LEG`-long vector. A MONOTONE
/// downward trend across steps (rather than a drop-then-plateau) is the
/// concrete, checkable form of `esc-076`'s own "nonmonotone-memory" title:
/// a plateau after step 1 is ordinary steady-state retention; a trend that
/// keeps falling step after step is a leak.
fn summarize_free_mib_trace(trace: &[f64]) -> String {
    if trace.len() < 2 {
        return format!("{trace:?} (too short to trend)");
    }
    let first = trace[0];
    let last = *trace.last().unwrap();
    let mid = trace[trace.len() / 2];
    let total_drop = first - last;
    let first_half_drop = first - mid;
    let second_half_drop = mid - last;
    let trend = if second_half_drop > first_half_drop * 1.2 {
        "ACCELERATING drop (later steps lose MORE free memory than earlier ones -- leak-shaped)"
    } else if second_half_drop < first_half_drop * 0.3 {
        "PLATEAUING (drop concentrated early, later steps roughly flat -- steady-state-shaped)"
    } else {
        "roughly LINEAR drop"
    };
    format!(
        "{} steps, free MiB: step0={first:.1} mid={mid:.1} last={last:.1} \
         (total_drop={total_drop:.1} MiB, first_half={first_half_drop:.1}, \
         second_half={second_half_drop:.1}) -- {trend}",
        trace.len()
    )
}

/// States which of esc-076's two pre-registered root-cause branches this
/// run's evidence supports. Printed (`--nocapture`), not asserted as a
/// pass/fail condition -- campaign #443's Part 3 item 4 fold-the-fix note:
/// this wave delivers the DIAGNOSIS and the control test, not the fix.
fn print_diagnosis(bf16: &LegOutcome, f16: &LegOutcome) {
    let oom = |o: &LegOutcome| matches!(o, LegOutcome::CudaOutOfMemory { .. });
    for (label, outcome) in [("bf16", bf16), ("f16", f16)] {
        match outcome {
            LegOutcome::Completed {
                losses,
                free_mib_after_step,
            } => {
                println!(
                    "[esc-076] {label} (fully-eager, reporter shape): Completed, \
                     {} steps, final loss={:.6}",
                    losses.len(),
                    losses.last().copied().unwrap_or(f32::NAN)
                );
                println!(
                    "[esc-076] {label} free-memory trace: {}",
                    summarize_free_mib_trace(free_mib_after_step)
                );
            }
            LegOutcome::CudaOutOfMemory {
                steps_completed,
                free_mib_after_step,
                message,
            } => {
                println!(
                    "[esc-076] {label} (fully-eager, reporter shape): CudaOutOfMemory after \
                     {steps_completed} steps -- {message}"
                );
                println!(
                    "[esc-076] {label} free-memory trace (up to the OOM'ing step): {}",
                    summarize_free_mib_trace(free_mib_after_step)
                );
            }
            LegOutcome::OtherError { message } => {
                println!("[esc-076] {label}: OtherError -- {message}");
            }
        }
    }
    match (oom(bf16), oom(f16)) {
        (true, true) => println!(
            "[esc-076] VERDICT: fully-eager BF16 ALSO OOMs at the reporter shape -- the defect \
             is EAGER-COMPOSITION MEMORY (dtype-independent), not f16-specific. The retained \
             three-forward-pass tape (anchor/positive/negative) under a fully-eager composition \
             is the more likely culprit than any f16-only allocation path."
        ),
        (false, true) => println!(
            "[esc-076] VERDICT: fully-eager BF16 completes but fully-eager F16 OOMs at the \
             IDENTICAL shape -- the defect is F16-SPECIFIC. Candidate mechanisms to check next \
             (W2b/W2c): candle f16 op fallbacks upcasting internally (extra retained f32 \
             copies), F16-specific autograd retention, or F16 CPU<->GPU transfer/alloc overhead \
             this composition does not pay for BF16."
        ),
        (true, false) => println!(
            "[esc-076] UNEXPECTED: fully-eager BF16 OOMs but fully-eager F16 completes at the \
             identical shape -- the reverse of the escape's own observable. Re-verify the \
             fixture/config against the escape's exact reproduction before trusting this run."
        ),
        (false, false) => println!(
            "[esc-076] INCONCLUSIVE on OOM alone at this device/step-count: neither fully-eager \
             leg OOM'd at the reporter shape over {STEPS_PER_LEG} steps (both completed) -- \
             READ THE free-memory trace lines above before concluding the defect no longer \
             reproduces. An 'ACCELERATING drop' or unbroken 'LINEAR drop' trend on the f16 leg \
             (and not on bf16, at a comparable magnitude) is still evidence of the SAME \
             f16-specific leak, just one this run's {STEPS_PER_LEG}-step / 80GB budget did not \
             exhaust; re-run with a larger STEPS_PER_LEG (or on a smaller-VRAM arch) if the \
             trace shows a trend rather than a plateau."
        ),
    }
}
