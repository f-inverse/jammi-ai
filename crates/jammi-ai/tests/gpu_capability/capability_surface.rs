//! `capability_surface` — campaign #443's data-driven runtime capability
//! proof, data-driven from `ci/release-feature-manifest.json`'s
//! `cu12-tarball` lane (the same lane `runpod_gpu_prove.sh`'s
//! capability-surface build derives its `jammi-ai`-applicable feature subset
//! from).
//!
//! Asserts:
//! - `CUDA_COMPILED` matches this build's `cuda` feature.
//! - `FLASH_COMPILED` matches this build's `flash-attn` feature — CFG-GATED
//!   arms (not a runtime branch), so the assertion is honest under either
//!   build: a build compiled WITHOUT `flash-attn` cannot even express "assert
//!   FLASH_COMPILED == true".
//! - Every op the manifest's `fused_op_admission` list declares admits
//!   (`Holds`) on THIS device, for the dtypes the current tree admits fused
//!   kernels for today (f32, bf16 — f16 kernel authoring is a separate,
//!   in-flight wave; campaign #443 plan v3 §Part 3), under
//!   `JAMMI_KERNELS_STRICT=1` (any admission Miss anywhere would hard-error
//!   the forward pass below — see `jammi_kernels::admission::admit`'s
//!   Strict-mode contract), by reading the per-op dispatch-registry deltas.
//! - `flash_dtypes` admits (`Holds`) the flash cascade when this build
//!   compiled `flash-attn` — cfg-gated off entirely otherwise.
//!
//! ## Why a synthetic, directly-built ModernBERT (not `session.fine_tune`)
//!
//! `attention_block_fused`'s domain requires the EXACT fixed head dimension
//! the kernel was built for (`jammi_kernels::ops::attention_block::HEAD_DIM
//! == 64`, `crates/jammi-encoders/src/modernbert.rs`'s
//! `attention_block_admission_predicate`) — independent of dtype, and
//! `admit`'s Strict mode has no domain-vs-capability exemption for a
//! two-arm op (unlike a cascade's `PredicateOutcome::DomainMiss`), so
//! **any** shape whose `hidden_size / num_attention_heads != 64` hard-errors
//! under Strict regardless of dtype. The `it` suite's `tiny_modernbert`
//! fixture (`hidden_size=64, num_attention_heads=... `→ head_dim≠64) cannot
//! exercise this op under Strict at all. This test instead builds a REAL,
//! purpose-shaped ModernBERT (`hidden_size=64, num_attention_heads=1` →
//! `head_dim=64`) directly via `jammi_encoders::ModernBert::builder`, with
//! synthetic (`Tensor::randn`) weights written once to a temp safetensors
//! file — mirroring `crates/jammi-encoders/tests/esc076_comparable_eager_control.rs`'s
//! already-landed pattern (that file's own doc: driving the encoder/trainer
//! seam with synthetic weights "if checkpoint-shape fidelity is preserved" is
//! this wave's own established idiom for a controlled-shape admission
//! probe). `global_attn_every_n_layers = 1` makes every layer global (no
//! sliding-window local-mask path), keeping the domain surface to exactly
//! the dtype/head-dim/seq checks this test cares about.
//!
//! `ci/scripts/runpod_gpu_prove.sh`'s capability-surface group invokes this
//! test by EXACT name (`--test gpu_capability capability_surface`) and reads
//! a "running 0 tests" match as a hard failure — this file's presence and
//! this function's name are load-bearing, not cosmetic.
//!
//! Gated like the rest of the suite: `live-gpu-tests` + a meaningful run also
//! needs `cuda` + a visible GPU; without them it skips loudly via
//! `skip_without_gpu!` (never `#[ignore]`).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_ai::fine_tune::ComputePrecision;
use jammi_encoders::{ModernBert, ModernBertConfig, Pooling};
use jammi_kernels::admission::{AdmissionMode, CascadeDispatchSnapshot, DispatchSnapshot};
use jammi_lora::{LoraBuildConfig, LoraInitMode};
use tempfile::TempDir;

use crate::skip_without_gpu;

/// The manifest lane this test reads — the same lane `runpod_gpu_prove.sh`'s
/// capability-surface build derives its feature list from.
const MANIFEST_LANE: &str = "cu12-tarball";

/// `ci/release-feature-manifest.json`, located relative to this crate's
/// manifest dir (`crates/jammi-ai` → workspace root → `ci/`).
const MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../ci/release-feature-manifest.json"
);

/// Manifest-declared op → the process-wide TWO-ARM dispatch-registry `op` key
/// the kernel's own `jammi_kernels::admission::admit` call site registers
/// under — reused VERBATIM (never re-derived) from reading
/// `crates/jammi-encoders/src/layer_norm.rs:100`,
/// `crates/jammi-encoders/src/modernbert.rs:190,671,1827`, and
/// `crates/jammi-lora/src/lora_linear.rs:36,65,205`.
///
/// `"dropout"` and `"low_rank_residual_linear"` BOTH map to `lora_linear_fused`
/// — `lora_linear.rs:36`'s own doc states its `lora_dropout` counter is
/// "Permanently `{fused: 0, eager: 0}` today": dropout is now reserved via
/// `DropoutMasks::next_key` and consumed DIRECTLY inside
/// `LowRankResidualLinear`'s own fused-or-eager arm, folded into the SAME
/// `lora_linear_fused` dispatch decision this file already reads for
/// `"low_rank_residual_linear"` — there is no longer a SEPARATE admission
/// signal for dropout to read; asserting `lora_linear_fused` Holds proves
/// both. Both fire during a plain forward pass (LoRA's forward composes
/// `base(x) + lora_delta(x)`), so no backward/optimizer step is needed.
///
/// `"softmax"` and `"rope"` are deliberately ABSENT from this list — see
/// [`STRICT_UNREACHABLE_OPS`]'s doc for why asserting either `Holds` in a
/// Strict-mode run through ModernBERT's attention path is not merely
/// untested but STRUCTURALLY IMPOSSIBLE.
const TWO_ARM_OPS: &[(&str, &str)] = &[
    ("layer_norm", "layer_norm_fused"),
    ("geglu", "geglu_fused"),
    ("attention_block", "attention_block_fused"),
    ("dropout", "lora_linear_fused"),
    ("low_rank_residual_linear", "lora_linear_fused"),
];

/// Manifest-declared op → the process-wide CASCADE dispatch-registry `op`
/// key (`jammi_kernels::admission::cascade_counters_for`) —
/// `crates/jammi-encoders/src/modernbert.rs:1242` et al.
///
/// Logged informationally rather than asserted `Holds`: `admit_cascade` has
/// no `fallback_warnings`-shaped reason channel, and `mem_efficient_attention`
/// additionally has its OWN shape/capability domain (independent of dtype) —
/// declining on this test's tiny fixture shape is a legitimate `DomainMiss`,
/// not evidence against f32/bf16 dtype admission.
const CASCADE_OPS: &[(&str, &str)] = &[("mem_efficient_attention", "mem_efficient_attention")];

/// Manifest-declared ops that are structurally UNREACHABLE in a
/// `JAMMI_KERNELS_STRICT=1` run through ModernBERT's attention path, proven
/// by reading `crates/jammi-encoders/src/modernbert.rs`'s
/// `forward_training_attention` (`:1176`): when `attention_block_fused`'s own
/// `admit()` (`:1273`) HOLDS, its `Fused` arm applies RoPE INTERNALLY off a
/// precomputed `rope_pack` (`:1283`, `RotaryEmbedding::cached_rope_pack`) and
/// runs the whole attention (QK^T, mask, softmax, PV) in ONE kernel — neither
/// `RotaryEmbedding::apply_training` (the `rope_fused` `admit()` site) nor
/// `softmax_apply_training` (the `softmax_last_dim_fused` `admit()` site,
/// `:1794`) is ever consulted. Both are called ONLY from
/// `forward_eager_training_attention_composition` (`:1577`), reached ONLY
/// through `attention_block_fused`'s `DispatchOutcome::Eager` arm. `admit`'s
/// Strict-mode contract makes `Eager` a provably unreachable outcome for a
/// two-arm op (a predicate miss becomes a hard `Err`, never `Eager`, in
/// Strict mode — `crates/jammi-kernels/src/admission.rs`'s `admit_inner`):
/// whenever `attention_block_fused` HOLDS (required for a Strict run to
/// complete at all through this path), `rope_fused`/`softmax_last_dim_fused`
/// are never even reached; whenever it does not hold, Strict aborts before
/// ever reaching that `Eager` arm. These ops are alternate composition TIERS
/// for the same computation, not independently provable in the same Strict
/// run — logged informationally, never asserted `Holds`, and never silently
/// dropped either (a human reading `--nocapture` output sees exactly why).
const STRICT_UNREACHABLE_OPS: &[&str] = &["softmax", "rope"];

/// The purpose-built tiny ModernBERT shape this test proves admission
/// against: `hidden_size / num_attention_heads == 64` —
/// [`jammi_kernels::ops::attention_block`]'s fixed `HEAD_DIM` — is the ONE
/// load-bearing property; everything else is sized only for speed.
fn probe_config() -> ModernBertConfig {
    ModernBertConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 1,
        intermediate_size: 128,
        vocab_size: 64,
        max_position_embeddings: 32,
        layer_norm_eps: 1e-5,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 32,
        // Every layer global (no sliding-window local-mask path) — see this
        // file's module doc.
        global_attn_every_n_layers: 1,
        attention_dropout: 0.0,
    }
}

/// Every tensor `ModernBertBuilder::build` expects, at real-checkpoint names
/// and shapes, filled with `Tensor::randn` — mirrors
/// `crates/jammi-encoders/tests/esc076_comparable_eager_control.rs`'s
/// `write_synthetic_checkpoint` (values never matter for an admission probe,
/// only shapes/dtypes/names do).
fn write_synthetic_checkpoint(config: &ModernBertConfig, path: &Path) {
    let cpu = Device::Cpu;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let randn =
        |shape: (usize, usize)| -> Tensor { Tensor::randn(0f32, 0.02, shape, &cpu).unwrap() };
    let randn1 =
        |n: usize| -> Tensor { (Tensor::randn(0f32, 0.02, n, &cpu).unwrap() + 1.0f64).unwrap() };

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

/// Deterministic (no unseeded RNG — family L/J), `< vocab_size`,
/// `[batch, seq]` synthetic token ids.
fn synthetic_ids(batch: usize, seq: usize, vocab: usize, device: &Device) -> Tensor {
    let ids: Vec<u32> = (0..batch * seq)
        .map(|i| (i as u32).wrapping_mul(2654435761) % vocab as u32)
        .collect();
    Tensor::from_vec(ids, (batch, seq), device).unwrap()
}

fn load_manifest() -> serde_json::Value {
    let raw = std::fs::read_to_string(MANIFEST_PATH)
        .unwrap_or_else(|e| panic!("read {MANIFEST_PATH}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{MANIFEST_PATH} must be valid JSON: {e}"))
}

fn manifest_string_list(manifest: &serde_json::Value, capability: &str) -> Vec<String> {
    manifest["lanes"][MANIFEST_LANE]["capabilities"][capability]
        .as_array()
        .unwrap_or_else(|| {
            panic!("manifest lane {MANIFEST_LANE:?} is missing capabilities.{capability}")
        })
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("capabilities.{capability} entry {v:?} is not a string"))
                .to_string()
        })
        .collect()
}

fn compute_precision_to_candle_dtype(p: ComputePrecision) -> DType {
    match p {
        ComputePrecision::F32 => DType::F32,
        ComputePrecision::F16 => DType::F16,
        ComputePrecision::BF16 => DType::BF16,
    }
}

/// Builds the probe ModernBERT (LoRA on `Wqkv`/`Wo`, real trainable adapter —
/// a `frozen()` build has no `Var` anywhere and candle's autodiff machinery
/// can take structurally different paths with no gradient to carry; a
/// trainable adapter is the comparable, real shape) at `dtype` on `device`,
/// sets it to training mode (the ONLY mode any of these fused arms is
/// reachable from — see `crates/jammi-ai/src/fine_tune/worker.rs`'s esc-075
/// module doc), and runs ONE forward pass over a synthetic batch. Returns the
/// forward `Result` unchanged — the caller decides what a failure means.
fn build_and_probe_forward(
    config: &ModernBertConfig,
    weights: &Path,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, jammi_encoders::EncoderError> {
    let varmap = VarMap::new();
    let target_modules = vec!["Wqkv".to_string(), "Wo".to_string()];
    let lora = LoraBuildConfig {
        target_modules: &target_modules,
        layers_to_transform: &None,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        // Nonzero: exercises the `lora_dropout` admission site too (a `0.0`
        // dropout takes a structurally different, dropout-free path).
        lora_dropout: Some(0.1),
        rank_pattern: &HashMap::new(),
        init_mode: LoraInitMode::Gaussian,
        seed: 1,
    };
    let mut model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .backbone_dtype(dtype)
        .lora(lora)
        .build(&[weights], config, device, &varmap)
        .unwrap_or_else(|e| panic!("capability_surface: build ModernBert ({dtype:?}): {e}"));
    model.set_training(true);

    let input_ids = synthetic_ids(2, 8, config.vocab_size, device);
    let mask = Tensor::ones((2, 8), DType::U32, device).unwrap();
    model.forward(&input_ids, &mask)
}

#[tokio::test(flavor = "multi_thread")]
async fn capability_surface() {
    skip_without_gpu!();
    assert_eq!(
        jammi_kernels::admission::admission_mode(),
        AdmissionMode::Strict,
        "capability_surface must run under JAMMI_KERNELS_STRICT=1 (see \
         ci/scripts/runpod_gpu_prove.sh's capability-surface group) — without it a fallen-back \
         run would read as a false pass"
    );

    // CUDA_COMPILED / FLASH_COMPILED, cfg-gated so the assertion is honest
    // under either build (never a runtime branch that could silently assert
    // the wrong side).
    assert!(
        jammi_kernels::admission::CUDA_COMPILED,
        "this suite only runs meaningfully with the cuda feature on (skip_without_gpu already \
         gates a GPU-less/non-cuda build out above)"
    );
    #[cfg(feature = "flash-attn")]
    assert!(
        jammi_kernels::admission::FLASH_COMPILED,
        "built with flash-attn: FLASH_COMPILED must be true"
    );
    #[cfg(not(feature = "flash-attn"))]
    assert!(
        !jammi_kernels::admission::FLASH_COMPILED,
        "built WITHOUT flash-attn: FLASH_COMPILED must be false — the flash arm below is \
         cfg-off (honest), never silently skipped at runtime"
    );

    let manifest = load_manifest();
    let declared_ops = manifest_string_list(&manifest, "fused_op_admission");
    assert!(
        !declared_ops.is_empty(),
        "ci/release-feature-manifest.json's {MANIFEST_LANE:?} lane must declare a non-empty \
         fused_op_admission list"
    );

    let device = Device::new_cuda(0).expect("skip_without_gpu already proved a CUDA device opens");
    let config = probe_config();
    let dir = TempDir::new().unwrap();
    let weights_path = dir.path().join("model.safetensors");
    write_synthetic_checkpoint(&config, &weights_path);

    let mapped: HashSet<&str> = TWO_ARM_OPS
        .iter()
        .chain(CASCADE_OPS.iter())
        .map(|&(op, _)| op)
        .chain(STRICT_UNREACHABLE_OPS.iter().copied())
        .collect();

    for dtype in [ComputePrecision::F32, ComputePrecision::BF16] {
        let candle_dtype = compute_precision_to_candle_dtype(dtype);
        let two_arm_before: Vec<(&str, DispatchSnapshot)> = TWO_ARM_OPS
            .iter()
            .map(|&(_, key)| (key, jammi_kernels::admission::counters_for(key).snapshot()))
            .collect();
        let cascade_before: Vec<(&str, CascadeDispatchSnapshot)> = CASCADE_OPS
            .iter()
            .map(|&(_, key)| {
                (
                    key,
                    jammi_kernels::admission::cascade_counters_for(key).snapshot(),
                )
            })
            .collect();

        build_and_probe_forward(&config, &weights_path, candle_dtype, &device).unwrap_or_else(
            |e| {
                panic!(
                    "capability_surface: the probe forward at backbone_dtype={dtype} must \
                     complete under JAMMI_KERNELS_STRICT=1 — a failure here means a \
                     manifest-declared fused op did NOT admit for this dtype on this device: {e}"
                )
            },
        );

        for &(report_op, registry_key) in TWO_ARM_OPS {
            if !declared_ops.iter().any(|d| d == report_op) {
                continue;
            }
            let before = two_arm_before
                .iter()
                .find(|(k, _)| *k == registry_key)
                .expect("registry_key present in the before snapshot map")
                .1;
            let after = jammi_kernels::admission::counters_for(registry_key).snapshot();
            assert!(
                after.fused > before.fused,
                "manifest-declared op {report_op:?} (registry key {registry_key:?}) must have \
                 dispatched FUSED at least once for backbone_dtype={dtype}, got before={before:?} \
                 after={after:?}"
            );
            assert_eq!(
                after.eager, before.eager,
                "manifest-declared op {report_op:?} must not have fallen back to eager at \
                 backbone_dtype={dtype} (a real eager fallback here would already have \
                 hard-errored the probe forward under Strict) — before={before:?} after={after:?}"
            );
        }

        for &(report_op, registry_key) in CASCADE_OPS {
            if !declared_ops.iter().any(|d| d == report_op) {
                continue;
            }
            let before = cascade_before
                .iter()
                .find(|(k, _)| *k == registry_key)
                .expect("registry_key present in the before snapshot map")
                .1;
            let after = jammi_kernels::admission::cascade_counters_for(registry_key).snapshot();
            tracing::info!(
                op = report_op,
                dtype = %dtype,
                ?before,
                ?after,
                "capability_surface: cascade op observed (informational only — see this \
                 file's CASCADE_OPS doc for why it is not asserted Holds)"
            );
        }

        for &op in STRICT_UNREACHABLE_OPS {
            if !declared_ops.iter().any(|d| d == op) {
                continue;
            }
            tracing::info!(
                op,
                dtype = %dtype,
                "capability_surface: structurally unreachable under Strict mode through this \
                 path — see STRICT_UNREACHABLE_OPS's doc; not asserted Holds, not silently \
                 dropped"
            );
        }

        for op in &declared_ops {
            if !mapped.contains(op.as_str()) {
                tracing::info!(
                    op,
                    dtype = %dtype,
                    "capability_surface: no dedicated dispatch-registry mapping for this \
                     manifest-declared op — covered only by the coarser \"the probe forward \
                     completed under Strict mode\" proof above (any admission Miss anywhere in \
                     this op's dispatch would already have hard-errored that forward)"
                );
            }
        }
    }

    // Flash: only meaningful (and only compiled at all) when this build
    // turned on `flash-attn`. Cfg-off entirely otherwise — "flash arm cfg-off
    // honest" per the campaign #443 W3 contract.
    #[cfg(feature = "flash-attn")]
    {
        let flash_dtypes = manifest_string_list(&manifest, "flash_dtypes");
        assert!(
            !flash_dtypes.is_empty(),
            "{MANIFEST_LANE:?}'s flash_dtypes must be non-empty when flash_compiled is true"
        );
        for dtype_name in &flash_dtypes {
            let dtype: ComputePrecision = dtype_name.parse().unwrap_or_else(|e| {
                panic!("manifest flash_dtypes entry {dtype_name:?} is not a known dtype: {e:?}")
            });
            let candle_dtype = compute_precision_to_candle_dtype(dtype);
            let before =
                jammi_kernels::admission::cascade_counters_for("attention_block_flash").snapshot();
            build_and_probe_forward(&config, &weights_path, candle_dtype, &device).unwrap_or_else(
                |e| {
                    panic!(
                        "capability_surface: the flash-dtype probe forward at \
                         backbone_dtype={dtype} must complete under JAMMI_KERNELS_STRICT=1: {e}"
                    )
                },
            );
            let after =
                jammi_kernels::admission::cascade_counters_for("attention_block_flash").snapshot();
            assert!(
                after.fused > before.fused,
                "flash_dtypes declares {dtype_name:?} — the flash cascade must have dispatched \
                 FUSED at least once for it, got before={before:?} after={after:?}"
            );
        }
    }
}
