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
//!    is set as the FIRST statement of the one `#[test]` in this file
//!    (this file is its own Cargo-autodiscovered test binary/process, so
//!    there is no other test racing `jammi_kernels::admission::disabled_ops`'s
//!    process-wide `OnceLock` for who initializes it first — see that
//!    function's own doc and `crates/jammi-bench/tests/finetune_step_kernel_disable.rs`'s
//!    identical concern, resolved there by spawning a child process
//!    instead; a single-test-per-binary file resolves it just as
//!    completely without the subprocess indirection). [`assert_ran_eager`]
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

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use jammi_encoders::{ModernBert, ModernBertConfig, Pooling};
use jammi_lora::LoraBuildConfig;
use std::collections::HashMap;
use std::path::Path;

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

fn cuda_device() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
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
    // binary/process (no `tests/it/`-style shared harness, no other
    // `#[test]` in this file) and this is the FIRST statement that could
    // touch `jammi_kernels::admission`'s process-wide `OnceLock` --
    // see this file's own module doc, control 1.
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
        oom_capability_witness_leg(&config, &weights_path, &device)
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
        run_leg(DType::BF16, &config, &weights_path, &device)
    });
    let f16_outcome = assert_ran_eager("f16_reporter_shape", || {
        run_leg(DType::F16, &config, &weights_path, &device)
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
