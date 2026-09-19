//! Eager-composition training memory on CUDA: a fully-eager bf16 leg and a fully-eager f16
//! leg at identical shape, driven at the library seam, tell an eager-composition memory
//! defect (both OOM) from an f16-specific one (only f16 OOMs); a variable-shape leg shows
//! sequence-length bucketing keeps a variable-shape run within the fixed-shape run's memory.
//!
//! ## Why the library seam
//!
//! This file drives `jammi_encoders::ModernBert` and `jammi_kernels::admission` directly,
//! mirroring one LoRA fine-tune step (three weight-tied forward passes —
//! anchor/positive/negative — a margin loss, one backward), rather than going through
//! `jammi-bench`'s CLI, whose `arm` is the A/B protocol's independent variable and must not
//! be widened to carry this control.
//!
//! ## Why synthetic weights
//!
//! The shape is ModernBERT-large (`hidden_size=1024, num_hidden_layers=28,
//! num_attention_heads=16, intermediate_size=2624, vocab_size=50368,
//! max_position_embeddings=8192, global_attn_every_n_layers=3, local_attention=128` — the
//! public `answerdotai/ModernBERT-large` `config.json`, and this port's compiled-in
//! defaults for every field the JSON omits: `DEFAULT_LAYER_NORM_EPS`/
//! `DEFAULT_GLOBAL_ROPE_THETA`/`DEFAULT_LOCAL_ROPE_THETA`/`DEFAULT_LOCAL_ATTENTION`/
//! `DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS` in `src/modernbert.rs`). Peak memory depends on
//! tensor shapes and dtypes, never on weight values, so [`write_synthetic_checkpoint`]
//! builds every tensor `ModernBertBuilder::build` reads, at the real checkpoint's names
//! and shapes, filled with `Tensor::randn` at F32 and written once; candle's
//! `VarBuilder::from_mmaped_safetensors` casts to the requested `backbone_dtype` on `get`,
//! so one file serves both the BF16 and F16 legs.
//!
//! ## The four vacuity controls
//!
//! 1. **Anti-sidestep** ([`assert_ran_eager`]): every `#[test]` sets
//!    `JAMMI_KERNELS_DISABLE=all` as its first statement. This file is its own test binary,
//!    and every test sets the identical value, so the process-wide `OnceLock` in
//!    `jammi_kernels::admission::disabled_ops` cannot be initialised to a different
//!    configuration by a sibling (`crates/jammi-bench/tests/finetune_step_kernel_disable.rs`
//!    has the same concern and spawns a child instead). That makes the env var safe; it does
//!    NOT make the device-memory oracles safe to run concurrently — see [`SerialGpu`].
//!    [`assert_ran_eager`] then reads [`jammi_encoders::ln_dispatch_snapshot`] around each
//!    leg and asserts `fused == 0`, so eager execution is proven, not assumed from the env
//!    var.
//! 2. **Anti-shrink**: the legs run at the full reference shape (`batch=16, seq=128`,
//!    ModernBERT-large); [`REFERENCE_BATCH`]/[`REFERENCE_SEQ`] are never reduced for test
//!    speed. A deliberately oversized witness leg ([`oom_capability_witness_leg`]) proves the
//!    harness's own OOM classification works; without it, "neither leg OOM'd" could not be
//!    told apart from "this harness cannot detect an OOM".
//! 3. **Anti-hollow-pass**: a completing leg must have a finite loss
//!    (`assert!(loss.is_finite())`, never `!(x > bound)`, which a NaN passes) AND every
//!    pooled output must still be in the requested dtype (checked for anchor, positive and
//!    negative before the loss's own `to_dtype(F32)`): a silent internal upcast would make
//!    the dtype comparison meaningless.
//! 4. **Comparable eager arm**: `run_leg(DType::BF16, ..)` and `run_leg(DType::F16, ..)`
//!    drive the identical pipeline (same weights file, same token ids, same margin loss and
//!    backward), differing only in `backbone_dtype`.
//!
//! ## The device this file needs
//!
//! The reference shapes are calibrated against an 80 GB device: a verdict here is about how
//! eager training memory GROWS across steps, which is only readable where a single
//! reference-shape step fits with room to spare. On a smaller device every leg runs out of
//! memory inside its first steps, which says nothing about eager composition.
//! [`cuda_device`] therefore refuses, by name, a device below
//! [`REFERENCE_DEVICE_MIN_TOTAL_MIB`]; a lane selects this target only where device 0 is
//! that large.
//!
//! ## One test at a time, structurally
//!
//! Every test here measures DEVICE-GLOBAL free memory. Two legs sampling `cuMemGetInfo`
//! while a third holds tens of GB of eager activations read each other's allocations as
//! their own. [`SerialGpu`] makes single-test-at-a-time a property of the only way this
//! file can obtain a `Device`, so it holds for tests that do not exist yet, without relying
//! on `--test-threads=1`.
//!
//! ## Reading the verdict
//!
//! [`print_diagnosis`] states which of the two root-cause branches (eager composition vs
//! f16-specific) a run's evidence supports, printed under `--nocapture` rather than
//! asserted. The test still FAILS on any outcome that is neither "completed, finite" nor
//! "refused with the CUDA out-of-memory driver error": an unrelated panic, a shape error
//! or a silent dtype coercion is a harness defect, not a diagnosis.
//!
//! ## Where eager training memory goes
//!
//! Measured on the bench path: a full f16 eager run at the reference shape runs out of
//! memory (0 → 49 GB at 5 s → 78 GB at 25 s); a 4-step run with seq capped at 64 peaks at
//! 63.3 GB and completes; a 4-step run repeating ONE batch shape peaks at an identical
//! 63.1 GB; a bf16-fused/f32-fused pair of the same config lands at 44.2/41.7 GB. So the
//! eager composition (any dtype) carries a ~19 GB base offset over the fused path plus a
//! growth term that tracks the number of DISTINCT batch shapes. The defect is
//! dtype-independent in principle, and f16-shaped in practice only because admission
//! routes f16 to the eager fallback.
//!
//! **The growth term.** `cudarc` has no caching allocator: every `CudaStorage` this crate
//! (and candle-core's own CUDA backend, through the same `CudaDevice::alloc`/`alloc_zeros`)
//! creates is a raw `cuMemAlloc`/`cuMemFree` pair. A raw, non-pooling allocator fed many
//! distinct allocation sizes fragments at the driver level — a new size class cannot
//! always be served from a block freed at a different size — so the reserved footprint
//! grows with the count of distinct sizes ever requested. Every one of those sizes is a
//! direct function of the `(batch, seq)` the caller hands `ModernBert::forward`; nothing
//! inside this crate's per-op eager arms can reduce the count without changing the
//! computation (padding an intermediate inside `LayerNorm::slow` would inject values into a
//! mean/variance reduction). Canonicalising shapes is only sound before tokens reach the
//! encoder, at the trainer's batch construction: `jammi-ai`'s
//! `fine_tune::batch_bucket`, wired at `TrainingLoop::encode_texts`, rounds each batch's
//! natural width up to a small, fixed power-of-two ladder, and the f16 run at the
//! reference shape completes at a flat 44.3 GB with it. The bucket decision itself
//! (`bucket_seq_len`/`MIN_BUCKET_LEN`) lives in `jammi_numerics`, below both crates, so
//! [`variable_shape_bucketed_steps_complete_with_bounded_memory`] calls the identical
//! decision at this seam without depending on `jammi-ai`.
//!
//! **The base offset.** The eager composition materialises more simultaneously-live,
//! separately-allocated tensors per op than the fused kernels (e.g. `LayerNorm::slow`'s
//! upcast-compute-cast-back in `src/layer_norm.rs`, and the analogous compositions in the
//! `softmax`/`geglu` eager fallbacks), each its own `cuMemAlloc` at its own (often F32)
//! size. This is a fixed, shape-count-independent overhead that does not by itself run out
//! of memory at 40 fixed-shape steps; bounding it means reusing scratch buffers across an
//! op's intermediates.
//!
//! **What the variable-shape legs show, and what they do not.** With five sequence lengths
//! all `<= 128` over 20 steps, the variable-shape leg and its fixed-shape control both
//! stayed flat (0.0 MiB drop). With [`VARIABLE_SHAPE_SEQS`] (11 lengths up to 512, 33
//! steps), the unbucketed variable-shape leg ran out of memory after 3 completed steps
//! (64, 96, 128, failing at 160) while the fixed-shape control completed all 33. Those
//! shapes vary both in count and in amplitude, so this shows that a realistic
//! variable-length training loop runs out of memory where a fixed-shape loop of the same
//! nominal size does not, without separating "shape variety alone" from "the largest
//! shape reached, retained by a non-releasing allocator". Both route to the same remedy
//! (bound shape variety and peak amplitude at batch construction); isolating shape count
//! alone needs a run at fixed amplitude.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use jammi_encoders::{ModernBert, ModernBertConfig, Pooling};
use jammi_lora::LoraBuildConfig;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

/// The reference training shape, `--batch 16 --max-seq-length 128`: the f16 eager
/// fine-tune at this shape fails in backward with `CUDA_ERROR_OUT_OF_MEMORY` about 85 s in.
const REFERENCE_BATCH: usize = 16;
const REFERENCE_SEQ: usize = 128;

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

/// This process's single GPU slot. See [`SerialGpu`] for why it exists and
/// why it is STRUCTURAL rather than a convention.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

/// A CUDA device that cannot be held without also holding [`GPU_SERIAL`].
///
/// This file's oracles read DEVICE-GLOBAL free memory (`cuda_free_mib` -> `cuMemGetInfo`),
/// and `cargo test` runs one binary's tests concurrently, so two legs measuring the same
/// device while a third allocates tens of GB of eager activations attribute each other's
/// allocations to their own trace -- a false failure, or a false pass when the interleaving
/// cancels out.
///
/// `cuda_device` is the only source of a `Device` in this file and returns this wrapper,
/// which holds the process-wide lock for as long as the caller holds the device, so a new
/// test cannot forget to serialize.
///
/// # Why there is no `Deref<Target = Device>`
///
/// A `Deref` would let `let d = (*cuda_device()).clone();` type-check, which ends the
/// serialization: the temporary guard (and the slot) drops at the end of the statement
/// while `d` is a live `Device` measured with no slot held. [`SerialGpu::device`] ties the
/// borrow to `&self`, so no `&Device` outlives the guard. `jammi-ai`'s
/// `tests/gpu_capability/harness.rs` has the same shape.
///
/// `candle_core::Device` is `Clone`, so `guard.device().clone()` still compiles; no API of
/// this shape can prevent cloning a `Clone` type reachable by reference (a `&DeviceRef`
/// newtype either derefs to `Device`, and the clone resolves through it, or cannot be
/// passed where `&Device` is wanted). That escape hatch is an explicit, greppable
/// `.device().clone()`, and no call site in this file needs it.
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
/// sibling can take the slot. Here it is not: this whole test target
/// requires `live-gpu-tests`, so the control only ever runs alongside legs
/// that hold the slot for MINUTES, and `cargo test` runs a binary's tests
/// concurrently -- a sibling holding the slot at that instant is CORRECT
/// behavior, not a leak, and asserting on it fails intermittently
/// (observed on an L40S, with
/// `variable_shape_bucketed_steps_complete_with_bounded_memory` holding
/// the slot). Release is `MutexGuard`'s own `Drop`, which is not this
/// file's code; exclusion is the part that is.
///
/// **No CPU-lane coverage.** Unlike the `jammi-ai` twin, this control is
/// compiled only with `live-gpu-tests`. It needs no DEVICE itself.
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

/// The least total device memory, in MiB, this file's reference shapes are calibrated for.
///
/// 64 GiB separates the two device classes the shapes have been measured on: a 48 GB
/// device cannot hold one reference-shape step (the bucketed variable-shape leg runs out
/// of memory within its first two steps there), and an 80 GB device holds every leg with
/// tens of GiB to spare. No device between the two has been measured, so this is the
/// bracket's midpoint, not a measured peak.
const REFERENCE_DEVICE_MIN_TOTAL_MIB: f64 = 64.0 * 1024.0;

/// Refuses, by name, a device whose total memory is below
/// [`REFERENCE_DEVICE_MIN_TOTAL_MIB`] -- instead of the out-of-memory or cuBLAS execution
/// failure such a device produces deep inside a leg, which reads as a finding or a harness
/// defect and is neither.
fn require_reference_device_memory(total_mib: f64) {
    assert!(
        total_mib >= REFERENCE_DEVICE_MIN_TOTAL_MIB,
        "this test's reference shapes are calibrated for a device with at least \
         {REFERENCE_DEVICE_MIN_TOTAL_MIB:.0} MiB; CUDA device 0 has {total_mib:.0} MiB"
    );
}

/// The refusal names both the memory this file needs and the memory the device has. Needs
/// no DEVICE: the measured total is the function's input.
#[test]
#[should_panic(
    expected = "calibrated for a device with at least 65536 MiB; CUDA device 0 has 46068 MiB"
)]
fn a_device_below_the_reference_memory_is_refused_by_name() {
    require_reference_device_memory(46068.0);
}

fn cuda_device() -> SerialGpu {
    // Taken BEFORE the device opens, so even device acquisition (which
    // allocates a context on the device) is serialized against a sibling
    // leg's memory trace.
    let slot = take_gpu_slot();
    let device = jammi_test_resources::cuda_device(0);
    require_reference_device_memory(cuda_memory_mib(&device).total);
    SerialGpu {
        device,
        _slot: slot,
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
/// `flash_oracle_build_model`'s (`src/modernbert.rs`) production-shaped LoRA
/// fixture. The reference failure does not fix `--lora-rank`/`--lora-alpha`,
/// so this is a production-like stand-in for those two flags.
fn production_lora_config() -> (Vec<String>, HashMap<String, usize>) {
    (vec!["Wqkv".to_string()], HashMap::new())
}

/// `with_lora`: when `true`, `Wqkv` gets a REAL trainable LoRA adapter
/// (see [`production_lora_config`]) instead of [`LoraBuildConfig::frozen`].
/// This matters: a `frozen()`
/// build has NO `Var` anywhere in the graph, so `Tensor::backward()`'s
/// walk finds nothing needing a gradient and candle can drop every
/// intermediate activation as soon as its Rust-side reference count hits
/// zero -- there is no backward-tape RETENTION pressure at all. A real
/// fine-tune's trainable LoRA A/B matrices are genuine `Var`s, so every
/// intermediate activation on ANY path from a LoRA-touched layer through
/// to the loss must stay alive until `.backward()` runs -- the actual
/// eager-composition memory profile of a real fine-tune. `frozen()` alone
/// is therefore not a comparable control, regardless of dtype.
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
            dropout_seed: 1,
        }
    } else {
        LoraBuildConfig::frozen()
    };
    let mut model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .backbone_dtype(dtype)
        .lora(lora)
        .build(&[weights], config, device, &varmap)
        .unwrap_or_else(|e| panic!("build ModernBert ({dtype:?}) failed: {e}"));
    model.set_training(true);
    (model, varmap)
}

/// Synthetic token ids -- deterministic (no external RNG dependency), `< vocab_size`, `[batch, seq]`.
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
    /// is the per-step free-memory trace -- a leg that "completes" but
    /// shows a MONOTONE downward free-memory trend (no plateau) is still a
    /// leak, just one this device's 80GB happened to absorb.
    Completed {
        losses: Vec<f32>,
        free_mib_after_step: Vec<f64>,
    },
    /// `steps_completed < STEPS_PER_LEG`: OOM'd partway through -- the
    /// reference failure (about 85 s in, not immediately) has this shape, a
    /// multi-step run, not a single-step peak.
    CudaOutOfMemory {
        steps_completed: usize,
        free_mib_after_step: Vec<f64>,
        message: String,
    },
    OtherError {
        message: String,
    },
}

/// Number of training steps per leg. The reference failure
/// ("Backward: DriverError(CUDA_ERROR_OUT_OF_MEMORY)" about 85 s in) comes
/// well after the FIRST step: this is a MULTI-STEP memory-growth
/// question, not a single-step peak-memory question. A single
/// forward+backward step at the reference shape
/// leaves tens of GiB of headroom on an 80GB A100 (see
/// [`oom_capability_witness_leg`]'s doc for the single-step numbers) --
/// `STEPS_PER_LEG` is chosen to give a real optimizer loop (with its own
/// `Var::set` in-place update, `AdamW`'s m/v state, and a FRESH forward
/// tape retained across MANY steps) room to reveal a genuine trend before
/// this test's own runtime budget is spent.
const STEPS_PER_LEG: usize = 40;

/// One `cuMemGetInfo` reading of a device, in MiB: `free` moves with every allocation on
/// the device, `total` is the installed memory and never moves.
struct CudaMemoryMib {
    free: f64,
    total: f64,
}

/// `cuMemGetInfo` after a device sync, the same driver call
/// `jammi_encoders::modernbert`'s own (private) VRAM probes make --
/// duplicated here since that helper is `#[cfg(test)]`-private to this
/// crate's OWN test module, not exported.
fn cuda_memory_mib(device: &Device) -> CudaMemoryMib {
    const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;
    device
        .synchronize()
        .expect("device sync before mem_get_info");
    let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .expect("cuMemGetInfo_v2 failed");
    CudaMemoryMib {
        free: free as f64 / BYTES_PER_MIB,
        total: total as f64 / BYTES_PER_MIB,
    }
}

fn cuda_free_mib(device: &Device) -> f64 {
    cuda_memory_mib(device).free
}

/// Runs a MULTI-STEP fully-eager training leg at `(REFERENCE_BATCH,
/// REFERENCE_SEQ)`: each step is three forward passes (anchor/positive/
/// negative, weight-tied -- the real LoRA fine-tune step's own shape,
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
        .unwrap_or_else(|e| panic!("AdamW::new_lr failed: {e}"));

    let anchor = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 1, device);
    let positive = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 2, device);
    let negative = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 3, device);
    let mask = Tensor::ones((REFERENCE_BATCH, REFERENCE_SEQ), DType::U32, device).unwrap();

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
/// SAME `REFERENCE_BATCH` -- that must ALWAYS classify as
/// [`LegOutcome::CudaOutOfMemory`] on any real GPU: attention's own
/// `[batch, heads, seq, seq]` scores tensor ALONE is
/// `16*16*8192*8192*4 bytes ~= 68.7 GiB` at this shape (global layer 0),
/// well past any single real GPU's VRAM even before any other tensor is
/// counted. Raising ONLY `batch`
/// (to `16 * 64 = 1024`, holding `seq = REFERENCE_SEQ = 128`) did NOT OOM
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
    let batch = REFERENCE_BATCH;
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

/// Distinct sequence lengths cycled round-robin, one per step: the
/// checkable form of "repeated batch shapes plateau; distinct batch shapes
/// keep growing" (module doc, "Where eager training memory goes"). The
/// range runs up to `512` (the second failing configuration, `b8*s512`) so
/// the leg sweeps genuinely new allocation sizes, as a variable-length
/// sentence dataset does; five lengths all `<= REFERENCE_SEQ` leave the
/// trace flat, indistinguishable from the fixed-shape control.
const VARIABLE_SHAPE_SEQS: [usize; 11] = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512];

/// Steps for the variable-shape leg: 3 full cycles through
/// [`VARIABLE_SHAPE_SEQS`] (`11 * 3 = 33`) — enough for a genuine trend
/// (vs a single cycle, which cannot distinguish "one-time cost per NEW
/// shape, then plateau" from "unbounded per-cycle growth") within a
/// practical wall-clock budget (shapes up to 4x `REFERENCE_SEQ`'s token
/// count cost proportionally more per step than the fixed-shape leg's
/// `STEPS_PER_LEG=40`).
const VARIABLE_SHAPE_STEPS: usize = 33;

/// The fixed-shape twin of an un-bucketed variable-shape leg at the SAME
/// [`VARIABLE_SHAPE_STEPS`] step count (never [`STEPS_PER_LEG`] — a
/// different step count would make the two traces' `total_drop`
/// incomparable) — every step uses the IDENTICAL `REFERENCE_SEQ`, so this
/// isolates "many steps at ONE shape" from "many steps across several
/// shapes", the ONE independent variable an un-bucketed variable-shape leg
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
        .unwrap_or_else(|e| panic!("AdamW::new_lr failed: {e}"));

    let anchor = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 1, device);
    let positive = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 2, device);
    let negative = synthetic_ids(REFERENCE_BATCH, REFERENCE_SEQ, config.vocab_size, 3, device);
    let mask = Tensor::ones((REFERENCE_BATCH, REFERENCE_SEQ), DType::U32, device).unwrap();

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
/// length up against — [`REFERENCE_SEQ`] (`128`), the `jammi-ai`
/// trainer's `effective_max` at the reference shape, where the f16 run
/// completes at a flat 44.3 GB.
///
/// **A cap of `512` ([`VARIABLE_SHAPE_SEQS`]'s raw maximum) does not pass
/// this leg**: bucketing to `{64, 128, 256, 512}` still visits `256`/`512`,
/// each of which costs tens of GB at this harness's shape (28-layer
/// ModernBERT-large, 3-forward eager LoRA backward), and the leg runs out
/// of memory after 3 steps. Bucketing bounds the COUNT of distinct shapes
/// under a reasonable `max_seq_length` ceiling; it does not lower that
/// ceiling. A trainer configured with `max_seq_length = 512` and
/// genuinely 512-token batches pays that cost regardless; the lever for
/// that is the `max_seq_length` config value itself.
const VARIABLE_SHAPE_BUCKET_CAP: usize = REFERENCE_SEQ;

/// The bucketed twin of an un-bucketed variable-shape leg: the IDENTICAL raw
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
/// the row-mutation helper). Truncate-then-bucket at `REFERENCE_SEQ` collapses
/// `VARIABLE_SHAPE_SEQS`'s 11 raw values (many `> REFERENCE_SEQ`) down to
/// just `{64, 128}` (2 distinct shapes, both already known-safe from the
/// fixed-shape control), and the leg completes where the unbucketed cycle
/// runs out of memory.
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
        .unwrap_or_else(|e| panic!("AdamW::new_lr failed: {e}"));

    // A row's ids/mask at the BUCKETED width: the first `raw_len` columns
    // are real synthetic content (mirroring `synthetic_ids`'s own hash),
    // the remaining `bucketed_len - raw_len` columns are pad id `0` /
    // mask `0` — exactly `pad_rows_to_bucket`'s own contract, restated for
    // a flat `(batch, bucketed_len)` tensor build.
    let build_bucketed = |raw_len: usize, bucketed_len: usize, salt: u32| -> (Tensor, Tensor) {
        let mut ids: Vec<u32> = Vec::with_capacity(REFERENCE_BATCH * bucketed_len);
        let mut mask: Vec<u32> = Vec::with_capacity(REFERENCE_BATCH * bucketed_len);
        for row in 0..REFERENCE_BATCH {
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
            Tensor::from_vec(ids, (REFERENCE_BATCH, bucketed_len), device).unwrap(),
            Tensor::from_vec(mask, (REFERENCE_BATCH, bucketed_len), device).unwrap(),
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
        let raw_len = VARIABLE_SHAPE_SEQS[step % VARIABLE_SHAPE_SEQS.len()].min(REFERENCE_SEQ);
        let bucketed_len = jammi_numerics::bucket_seq_len(raw_len, VARIABLE_SHAPE_BUCKET_CAP);
        let (anchor_ids, anchor_mask) = build_bucketed(raw_len, bucketed_len, 1);
        let (positive_ids, _positive_mask) = build_bucketed(raw_len, bucketed_len, 2);
        let (negative_ids, _negative_mask) = build_bucketed(raw_len, bucketed_len, 3);
        // All three rows share the SAME mask (identical raw_len/bucketed_len
        // per step, mirroring an un-bucketed variable-shape leg's own single shared
        // `mask` per step) — `_positive_mask`/`_negative_mask` are built
        // anyway so a divergence in per-row padding would still construct
        // a real tensor to compare against; this leg's uniform length per
        // step makes them identical to `anchor_mask`.
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

/// The [`VARIABLE_SHAPE_SEQS`] cycle that runs out of memory unbucketed,
/// with each step's raw length first truncated to
/// [`VARIABLE_SHAPE_BUCKET_CAP`] (`REFERENCE_SEQ`, as the trainer's
/// tokenizer truncates with `Some(effective_max)`) and then rounded up
/// through `jammi_numerics::bucket_seq_len`, the same candle-free decision
/// `jammi-ai`'s trainer calls at batch construction
/// (`fine_tune::batch_bucket`). The 11 raw lengths (most
/// `> REFERENCE_SEQ`) collapse to `{64, 128}`. This proves the bound at the
/// `jammi-encoders` library seam directly, at `max_seq_length =
/// REFERENCE_SEQ`; bucketing bounds shape COUNT, not shape AMPLITUDE (see
/// [`VARIABLE_SHAPE_BUCKET_CAP`]).
///
/// Asserts BOTH: (a) every one of `VARIABLE_SHAPE_STEPS` steps completes
/// with a finite loss (never merely "did not panic"), and (b) the bucketed
/// leg's total free-memory drop stays within `3x` its fixed-shape control's
/// drop over the same step count -- a bound relative to the same run's own
/// baseline, never a constant from a different session -- so bucketing
/// genuinely BOUNDS memory rather than merely not running out at this step
/// count.
#[test]
fn variable_shape_bucketed_steps_complete_with_bounded_memory() {
    std::env::set_var("JAMMI_KERNELS_DISABLE", "all");

    let device = cuda_device();

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
        "[bucketed variable-shape] bucketed variable-shape bf16 ({VARIABLE_SHAPE_STEPS} steps, raw \
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
                    "[bucketed variable-shape] step {step} completed but loss is non-finite ({loss}) -- a \
                     completing bucketed leg must produce a genuinely finite result at EVERY \
                     step, never a silently-propagated NaN/inf read as success"
                );
            }
        }
        other => panic!(
            "[bucketed variable-shape] the bucketed variable-shape leg must COMPLETE -- got \
             {other:?} instead. Either sequence-length bucketing no longer bounds memory, or this \
             leg's shape/step budget needs re-measuring against the bucket ladder."
        ),
    }

    let bucketed_drop = total_drop_mib(&bucketed_outcome)
        .expect("the completed bucketed leg records free memory after every step");
    let fixed_drop = total_drop_mib(&fixed_outcome)
        .expect("the fixed-shape reference leg records free memory after every step");
    println!(
        "[bucketed variable-shape] total_drop_mib: bucketed-variable={bucketed_drop:.1} \
         fixed-shape={fixed_drop:.1} ratio={:.2}",
        bucketed_drop / fixed_drop.max(1.0)
    );
    const GROWTH_RATIO_BOUND: f64 = 3.0;
    assert!(
        bucketed_drop <= GROWTH_RATIO_BOUND * fixed_drop.max(1.0),
        "[bucketed variable-shape] bucketed variable-shape eager composition dropped {bucketed_drop:.1} \
         MiB of free memory over {VARIABLE_SHAPE_STEPS} steps vs {fixed_drop:.1} MiB for its \
         fixed-shape control -- a {:.2}x ratio, past the {GROWTH_RATIO_BOUND}x bound -- \
         bucketing should collapse this leg's distinct-shape count to {{64,128}} (2 buckets, \
         where repeated shapes plateau), not merely shrink the unbucketed growth without \
         bounding it.",
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
fn fully_eager_bf16_vs_f16_at_reference_shape() {
    // This file is its own test binary, and this is the FIRST statement
    // that could touch `jammi_kernels::admission`'s process-wide
    // `OnceLock` (module doc, control 1). Every test here sets the
    // identical `"all"` value, which is what makes the `OnceLock` safe;
    // mutual exclusion for the DEVICE-GLOBAL memory oracles is
    // `SerialGpu`'s job.
    std::env::set_var("JAMMI_KERNELS_DISABLE", "all");

    let device = cuda_device();

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
                "[eager-memory] OOM capability witness: CONFIRMED (harness can detect a real CUDA \
                 OOM) -- {message}"
            );
        }
        other => panic!(
            "oom_capability_witness_leg (batch={REFERENCE_BATCH}, seq=max_position_embeddings) \
             did not classify as CUDA OOM on this device -- got {other:?} instead. Either this \
             device has more VRAM than the witness's deliberately-oversized shape needs (raise \
             the witness further), or this harness's OOM-string-match classification is broken \
             -- either way, the real legs' outcomes below cannot be trusted until this witness \
             reads CudaOutOfMemory"
        ),
    }

    let bf16_outcome = assert_ran_eager("bf16_reference_shape", || {
        run_leg(DType::BF16, &config, &weights_path, device.device())
    });
    let f16_outcome = assert_ran_eager("f16_reference_shape", || {
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
                     classified CUDA OOM) -- this is a harness defect, not a memory finding: \
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
/// checkable form of memory growth: a plateau after step 1 is ordinary steady-state retention; a trend that
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

/// States which of the two root-cause branches (eager composition vs
/// f16-specific) this run's evidence supports. Printed (`--nocapture`), not
/// asserted: the verdict is a diagnosis, not a pass/fail condition.
fn print_diagnosis(bf16: &LegOutcome, f16: &LegOutcome) {
    let oom = |o: &LegOutcome| matches!(o, LegOutcome::CudaOutOfMemory { .. });
    for (label, outcome) in [("bf16", bf16), ("f16", f16)] {
        match outcome {
            LegOutcome::Completed {
                losses,
                free_mib_after_step,
            } => {
                println!(
                    "[eager-memory] {label} (fully-eager, reference shape): Completed, \
                     {} steps, final loss={:.6}",
                    losses.len(),
                    losses.last().copied().unwrap_or(f32::NAN)
                );
                println!(
                    "[eager-memory] {label} free-memory trace: {}",
                    summarize_free_mib_trace(free_mib_after_step)
                );
            }
            LegOutcome::CudaOutOfMemory {
                steps_completed,
                free_mib_after_step,
                message,
            } => {
                println!(
                    "[eager-memory] {label} (fully-eager, reference shape): CudaOutOfMemory after \
                     {steps_completed} steps -- {message}"
                );
                println!(
                    "[eager-memory] {label} free-memory trace (up to the OOM'ing step): {}",
                    summarize_free_mib_trace(free_mib_after_step)
                );
            }
            LegOutcome::OtherError { message } => {
                println!("[eager-memory] {label}: OtherError -- {message}");
            }
        }
    }
    match (oom(bf16), oom(f16)) {
        (true, true) => println!(
            "[eager-memory] VERDICT: fully-eager BF16 ALSO OOMs at the reference shape -- the defect \
             is EAGER-COMPOSITION MEMORY (dtype-independent), not f16-specific. The retained \
             three-forward-pass tape (anchor/positive/negative) under a fully-eager composition \
             is the more likely culprit than any f16-only allocation path."
        ),
        (false, true) => println!(
            "[eager-memory] VERDICT: fully-eager BF16 completes but fully-eager F16 OOMs at the \
             IDENTICAL shape -- the defect is F16-SPECIFIC. Candidate mechanisms to check next: \
             candle f16 op fallbacks upcasting internally (extra retained f32 \
             copies), F16-specific autograd retention, or F16 CPU<->GPU transfer/alloc overhead \
             this composition does not pay for BF16."
        ),
        (true, false) => println!(
            "[eager-memory] UNEXPECTED: fully-eager BF16 OOMs but fully-eager F16 completes at the \
             identical shape -- the reverse of the reference failure. Re-verify the \
             fixture/config against the reference shape before trusting this run."
        ),
        (false, false) => println!(
            "[eager-memory] INCONCLUSIVE on OOM alone at this device/step-count: neither fully-eager \
             leg OOM'd at the reference shape over {STEPS_PER_LEG} steps (both completed) -- \
             READ THE free-memory trace lines above before concluding the defect no longer \
             reproduces. An 'ACCELERATING drop' or unbroken 'LINEAR drop' trend on the f16 leg \
             (and not on bf16, at a comparable magnitude) is still evidence of the SAME \
             f16-specific leak, just one this run's {STEPS_PER_LEG}-step / 80GB budget did not \
             exhaust; re-run with a larger STEPS_PER_LEG (or on a smaller-VRAM arch) if the \
             trace shows a trend rather than a plateau."
        ),
    }
}
