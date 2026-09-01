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
//! - Every op the manifest's `fused_op_admission` list declares that this
//!   file can attribute to a real dispatch-registry key admits (`Holds`) on
//!   THIS device, for every dtype the current tree admits fused kernels for
//!   (f32, bf16, f16 — f16 kernel authoring landed in campaign #443's FA2
//!   fp16 wave, closing the gap plan v3 §Part 3 tracked as in-flight),
//!   under `JAMMI_KERNELS_STRICT=1` (a two-arm op's admission Miss
//!   hard-errors under Strict — see `jammi_kernels::admission::admit`'s
//!   Strict-mode contract), by reading the per-op dispatch-registry deltas
//!   around a real forward + backward + one optimizer step over a synthetic
//!   batch.
//! - `flash_dtypes` admits (`Holds`) the flash cascade when this build
//!   compiled `flash-attn` AND the dtype is one flash preempts attention_block
//!   for — see the `attention_block` TIER PREEMPTION note below. The
//!   manifest's `flash_dtypes` is `["bf16", "f16"]` (widened alongside FA2
//!   fp16 dispatch, campaign #443) — never bf16-only.
//!
//! ## TIER PREEMPTION: `attention_block` vs the flash cascade
//! (adversarial-audit finding 1, campaign #443 Phase 4)
//!
//! `ModernBertAttention::forward_training_attention`
//! (`crates/jammi-encoders/src/modernbert.rs:~1192`) consults the flash
//! cascade FIRST and returns immediately on `CascadeOutcome::Fused` —
//! `attention_block_fused`'s own `admit()` call is never even reached in
//! that case. So on a `flash-attn`-compiled build, for a dtype the manifest's
//! `flash_dtypes` declares (today: `bf16`), the flash cascade counter moves
//! and `attention_block_fused`'s counter does NOT — asserting BOTH moved (the
//! pre-fix bug) is mutually exclusive and fails on whichever build the OTHER
//! half assumes. This file computes, per dtype, whether flash is expected to
//! preempt (`FLASH_COMPILED && flash_dtypes.contains(dtype)`) and asserts the
//! CORRESPONDING real pair: preemption asserts flash moved AND
//! `attention_block_fused` did NOT; no preemption asserts `attention_block_fused`
//! moved AND flash did NOT (both are real assertions — neither arm is
//! logging-only).
//!
//! ## VACUOUS-COVERAGE fix: a real backward + optimizer step
//! (adversarial-audit finding 2, campaign #443 Phase 4)
//!
//! A forward-only probe cannot claim "a Strict-mode failure anywhere in this
//! op's dispatch would already have hard-errored the forward" for an op that
//! never dispatches during a plain forward — that claim is FALSE, not merely
//! unproven, for a backward/optimizer-time op. [`probe_dtype`] now runs a
//! real `loss.backward()` plus one [`jammi_ai::fine_tune::adamw::AdamW`] step
//! over the trainable LoRA vars, so any backward/optimizer-time
//! admission-gated dispatch genuinely fires and its registry delta is
//! readable.
//!
//! ## The probed-op table is ONE static fact (campaign #446 finding 2)
//!
//! Every "op → dispatch-registry key" fact below is DERIVED from
//! `jammi_kernels::admission::PROBED_OPS`, shared with
//! `crates/jammi-ai/src/fine_tune/worker.rs`'s esc-075 acceleration report.
//! An earlier revision of this file hand-encoded its own copy and stated, as
//! settled fact, that `rope_positions`, `axpy`, `cast_scale` and
//! `scaled_cast_add` all had no `admit`/`admit_cascade` call site anywhere.
//! That was WRONG for `cast_scale` (and its sibling `cast_add`), which admit
//! under DTYPE-RESOLVED registry keys — `cast_scale_bf16_f32` /
//! `cast_scale_f16_f32`, `cast_add_bf16` / `cast_add_f16` — not under a bare
//! report-key literal. `rope_positions`/`scaled_cast_add` genuinely have no
//! admission gate but are provable through their PARENT's fused dispatch
//! (`ProbedOpKind::InternalSubkernel`), and `axpy` alone is left with neither
//! a gate nor a parent. See [`known_no_dispatch_site_ops`] and
//! [`internal_subkernel_ops`].
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
//! the dtype/head-dim/seq checks this test cares about. The whole model
//! (fresh `VarMap`, fresh weights file) is discarded after each dtype's
//! probe, so the backward+optimizer step's weight mutation and RNG draw have
//! no effect beyond this test process — unlike the worker's own in-place
//! production probe (`crates/jammi-ai/src/fine_tune/worker.rs`'s esc-075
//! module doc), which snapshots and restores the real trainable weights for
//! exactly this reason.
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
use jammi_ai::fine_tune::adamw::{AdamW, ParamsAdamW};
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

/// This test's ONE source of probed-op facts:
/// [`jammi_kernels::admission::PROBED_OPS`] (campaign #446 finding 2). Before
/// it, this file hand-encoded its own copy of "op → registry key", the worker
/// hand-encoded a second and third (its snapshot struct's fields and its
/// `two_arm` match), and the two drifted — the f16 cast-epilogue keys were in
/// none of them. Nothing below re-types a registry key.
use jammi_kernels::admission::{DtypeClass, ProbedOpKind, PROBED_OPS};

/// The manifest capability list every [`ProbedOpKind::TwoArm`] /
/// [`ProbedOpKind::Cascade`] row belongs to.
const MANIFEST_FUSED_OP_ADMISSION: &str = "fused_op_admission";

/// The manifest capability list every [`ProbedOpKind::InternalSubkernel`] row
/// belongs to — the category the campaign lead's manifest reclassification
/// introduces for kernels that are launched unconditionally from inside an
/// already-admitted parent's fused arm and therefore have NO admission gate
/// of their own to assert `Holds` against.
const MANIFEST_INTERNAL_SUBKERNELS: &str = "internal_subkernels";

/// The manifest capability list for kernels this build COMPILES but that
/// dispatch through no admission gate and have no probed parent either —
/// after the reclassification, `axpy` alone (pending its own W2 disposition).
const MANIFEST_FUSED_KERNELS_COMPILED: &str = "fused_kernels_compiled";

/// This test's `ComputePrecision` as the dtype class [`PROBED_OPS`] resolves
/// registry keys against — mirrors `crates/jammi-ai/src/fine_tune/worker.rs`'s
/// own `dtype_class_of`.
fn dtype_class(p: ComputePrecision) -> DtypeClass {
    match p {
        ComputePrecision::F32 => DtypeClass::F32,
        ComputePrecision::BF16 => DtypeClass::Bf16,
        ComputePrecision::F16 => DtypeClass::F16,
    }
}

/// `(manifest-declared op, TWO-ARM dispatch-registry key)` for `dtype`,
/// DERIVED from [`PROBED_OPS`] — never re-typed here.
///
/// Two report keys are filtered out, each for a reason this file proves
/// elsewhere rather than by omission:
///
/// - `"attention_block"` — its own admission can be PREEMPTED by the flash
///   cascade (see this file's module doc's "TIER PREEMPTION" section), so it
///   is asserted separately, per-dtype, in [`capability_surface`] itself with
///   the preemption-aware PAIR of assertions.
/// - `"softmax"`/`"rope"` ([`STRICT_UNREACHABLE_OPS`]) — asserting either
///   `Holds` in a Strict-mode run through ModernBERT's attention path is not
///   merely untested but STRUCTURALLY IMPOSSIBLE; see that const's doc.
///
/// `"dropout"` and `"low_rank_residual_linear"` both resolve to
/// `lora_linear_fused` — that is a fact of the table now, not of this file
/// (`crates/jammi-lora/src/lora_linear.rs:37`'s own doc: the separate
/// `lora_dropout` counter is "Permanently `{fused: 0, eager: 0}` today";
/// dropout is consumed directly inside `LowRankResidualLinear`'s own
/// fused-or-eager arm, folded into the SAME dispatch decision).
///
/// `"cast_scale"`/`"cast_add"` appear ONLY for `bf16`/`f16`: on an `f32`
/// backbone `LowRankResidualLinear::bwd` takes a structurally different,
/// `admit()`-free "nothing to fuse" branch, so there is no key to assert and
/// the op is absent rather than claimed a miss.
///
/// `"adamw_step"` appears at EVERY dtype (a `DtypeClass::Any` row): the fused
/// multi-tensor AdamW step's own domain is `F32`, but that is a fact about
/// the TRAINABLE VARS — which are `F32` on every backbone — not about the
/// job's backbone dtype class. [`probe_dtype`] already runs a real
/// `AdamW::step`, so this row is asserted `Holds` under Strict on the
/// optimizer step that probe already performs; nothing new has to run for it.
/// See `jammi_kernels::admission::PROBED_OPS`'s "the optimizer's dtype DOMAIN
/// is not a dtype CLASS" note for why encoding it as `DtypeClass::F32` would
/// have silently omitted it from every bf16/f16 report.
fn two_arm_ops_for(dtype: DtypeClass) -> Vec<(&'static str, &'static str)> {
    PROBED_OPS
        .iter()
        .filter(|op| op.kind == ProbedOpKind::TwoArm)
        .filter(|op| op.report_key != "attention_block")
        .filter(|op| !STRICT_UNREACHABLE_OPS.contains(&op.report_key))
        .filter_map(|op| {
            op.registry_keys_for(dtype)
                .next()
                .map(|key| (op.report_key, key))
        })
        .collect()
}

/// `(manifest-declared op, CASCADE dispatch-registry key)`, DERIVED from
/// [`PROBED_OPS`]'s [`ProbedOpKind::Cascade`] rows.
///
/// Logged informationally rather than asserted `Holds`: `admit_cascade` has
/// no `fallback_warnings`-shaped reason channel, `mem_efficient_attention`
/// additionally has its OWN shape/capability domain (independent of dtype) —
/// declining on this test's tiny fixture shape is a legitimate `DomainMiss`,
/// not evidence against f32/bf16 dtype admission — and, per Phase-4 finding
/// number one, `mem_efficient_attention` is ALSO consulted only after the
/// flash cascade declines (`forward_training_attention`'s own ordering), so
/// it can be preempted by flash exactly like `attention_block_fused` can.
fn cascade_ops() -> Vec<(&'static str, &'static str)> {
    PROBED_OPS
        .iter()
        .filter(|op| op.kind == ProbedOpKind::Cascade)
        .filter_map(|op| {
            op.registry_keys_for(DtypeClass::Any)
                .next()
                .map(|key| (op.report_key, key))
        })
        .collect()
}

/// `(op, the parent whose fused dispatch proves it ran)`, DERIVED from
/// [`PROBED_OPS`]'s [`ProbedOpKind::InternalSubkernel`] rows — the manifest's
/// new `internal_subkernels` category.
fn internal_subkernel_ops() -> Vec<(&'static str, &'static str)> {
    PROBED_OPS
        .iter()
        .filter_map(|op| match op.kind {
            ProbedOpKind::InternalSubkernel { parent } => Some((op.report_key, parent)),
            _ => None,
        })
        .collect()
}

/// Every [`PROBED_OPS`] report key, regardless of kind.
fn all_probed_report_keys() -> HashSet<&'static str> {
    PROBED_OPS.iter().map(|op| op.report_key).collect()
}

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

/// Manifest-declared ops that [`PROBED_OPS`] names in NO capacity at all:
/// no `admit`/`admit_cascade` registry key AND no probed parent whose fused
/// dispatch would imply they ran. Derived from the manifest's
/// `fused_kernels_compiled` list minus every [`PROBED_OPS`] report key —
/// never re-typed.
///
/// **The prior revision of this doc was WRONG and is corrected here**
/// (campaign #446 finding 2). It claimed `"rope_positions"`, `"axpy"`,
/// `"cast_scale"` and `"scaled_cast_add"` ALL had "NO registered
/// `admit`/`admit_cascade` call site ANYWHERE in this workspace". Two of the
/// four do:
///
/// - `cast_scale` DOES admit, under dtype-resolved keys
///   `cast_scale_bf16_f32` / `cast_scale_f16_f32`
///   (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:800`,
///   `:814`); `cast_add` likewise under `cast_add_bf16` / `cast_add_f16`
///   (`:899`, `:911`). The old claim held only because the old text looked
///   for a bare `"cast_scale"` literal, which is a REPORT key, not a registry
///   key — the exact confusion that left the f16 acceleration report unable
///   to name its own cast epilogue.
/// - `rope_positions` (`crates/jammi-kernels/src/ops/flash_attention.rs:645`)
///   and `scaled_cast_add`
///   (`crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:693`,
///   `crates/jammi-kernels/src/cuda/low_rank_residual_linear.rs:131`) really
///   do have no admission gate — but they are not unobservable either: each
///   is a bare launcher call from inside an already-admitted PARENT's fused
///   arm, so the parent dispatching fused proves the subkernel ran. They are
///   [`ProbedOpKind::InternalSubkernel`] rows, reported through
///   [`internal_subkernel_ops`].
///
/// What is left in this bucket is `axpy` alone: compiled, but with no
/// dispatch decision and no parent to prove it through. It is logged, never
/// claimed covered by the completed-run argument (which is FALSE for an op
/// that never dispatches through `admit` at all).
fn known_no_dispatch_site_ops(manifest: &serde_json::Value) -> Vec<String> {
    let probed = all_probed_report_keys();
    manifest_string_list(manifest, MANIFEST_FUSED_KERNELS_COMPILED)
        .into_iter()
        .filter(|op| !probed.contains(op.as_str()))
        .collect()
}

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

/// `capabilities.internal_subkernels`, which — unlike its sibling capability
/// lists — is an OBJECT keyed by op, not a string array:
/// `{"<op>": {"parent": "<op>", "launch_site": "<path>"}, ..}`
/// (`ci/scripts/check_release_manifest.py` validates that every `parent`
/// resolves and every `launch_site` exists).
///
/// The shape is the point, not an inconsistency to paper over: an internal
/// subkernel has no admission gate of its own and is PROVABLE only through its
/// parent's dispatch, so the manifest records the proof RELATION alongside the
/// name. A plain string list could name the op but not say what proves it —
/// the same "claimed but unprovable" shape campaign #446 finding 2 was about.
/// Returns `(op, parent)` pairs, sorted by op.
fn manifest_internal_subkernels(manifest: &serde_json::Value) -> Vec<(String, String)> {
    let obj = manifest["lanes"][MANIFEST_LANE]["capabilities"][MANIFEST_INTERNAL_SUBKERNELS]
        .as_object()
        .unwrap_or_else(|| {
            panic!(
                "manifest lane {MANIFEST_LANE:?}'s capabilities.{MANIFEST_INTERNAL_SUBKERNELS} \
                 must be an OBJECT keyed by op (each value carrying `parent` + `launch_site`), \
                 not an array — see ci/scripts/check_release_manifest.py"
            )
        });
    let mut out: Vec<(String, String)> = obj
        .iter()
        .map(|(op, entry)| {
            let parent = entry["parent"].as_str().unwrap_or_else(|| {
                panic!(
                    "capabilities.{MANIFEST_INTERNAL_SUBKERNELS}[{op:?}] must carry a string \
                     `parent` — the op whose fused dispatch is what proves this subkernel ran"
                )
            });
            (op.clone(), parent.to_string())
        })
        .collect();
    out.sort();
    out
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
/// module doc), and drives ONE forward + backward + optimizer step over a
/// synthetic batch — closing the vacuous-coverage gap a forward-only probe
/// left (Phase-4 audit finding 2): backward/optimizer-time admission-gated
/// ops (e.g. `low_rank_residual_linear`'s own backward-time `cast_add_bf16`
/// epilogue) now genuinely dispatch. The whole model is discarded after this
/// call returns (fresh `VarMap` per call), so the step's weight mutation and
/// dropout RNG draw have no effect beyond this test process.
fn probe_dtype(config: &ModernBertConfig, weights: &Path, dtype: DType, device: &Device) {
    let varmap = VarMap::new();
    let target_modules = vec!["Wqkv".to_string(), "Wo".to_string()];
    let lora = LoraBuildConfig {
        target_modules: &target_modules,
        layers_to_transform: &None,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        // Nonzero: exercises the `lora_dropout`/`lora_linear_fused` dropout
        // admission site too (a `0.0` dropout takes a structurally
        // different, dropout-free path).
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
    let output = model.forward(&input_ids, &mask).unwrap_or_else(|e| {
        panic!(
            "capability_surface: the probe forward at backbone_dtype={dtype:?} must complete \
             under JAMMI_KERNELS_STRICT=1 — a failure here means a manifest-declared fused op \
             did NOT admit for this dtype on this device: {e}"
        )
    });

    // Backward + one optimizer step (Phase-4 audit finding 2): a plain scalar
    // loss over the pooled output is enough to drive a real
    // `Tensor::backward()` walk through every LoRA-touched layer.
    let loss = output
        .to_dtype(DType::F32)
        .and_then(|t| t.sqr())
        .and_then(|t| t.mean_all())
        .unwrap_or_else(|e| panic!("capability_surface: loss at backbone_dtype={dtype:?}: {e}"));
    let grads = loss.backward().unwrap_or_else(|e| {
        panic!("capability_surface: backward at backbone_dtype={dtype:?}: {e}")
    });
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default()).unwrap_or_else(|e| {
        panic!("capability_surface: AdamW::new at backbone_dtype={dtype:?}: {e}")
    });
    opt.step(&grads).unwrap_or_else(|e| {
        panic!(
            "capability_surface: the optimizer step at backbone_dtype={dtype:?} must complete \
             under JAMMI_KERNELS_STRICT=1 — a failure here means a manifest-declared \
             backward/optimizer-time fused op did NOT admit for this dtype on this device: {e}"
        )
    });
}

// The CUDA_COMPILED/FLASH_COMPILED asserts below are deliberately over a
// compile-time constant: the point is pinning "this build's `cfg!()`-derived
// constant matches the feature this test was actually compiled with", a
// regression guard against the two ever silently drifting from `cfg!()`'s
// own value — not a removable tautology.
#[allow(clippy::assertions_on_constants)]
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
         driven by this same runtime constant, never a `#[cfg]` branch, so it degrades \
         correctly rather than silently skipping"
    );

    let manifest = load_manifest();
    let declared_ops = manifest_string_list(&manifest, MANIFEST_FUSED_OP_ADMISSION);
    assert!(
        !declared_ops.is_empty(),
        "ci/release-feature-manifest.json's {MANIFEST_LANE:?} lane must declare a non-empty \
         fused_op_admission list"
    );
    // Read once: which dtypes (if any) this build's flash cascade preempts
    // `attention_block`/`mem_efficient_attention` for — empty on a
    // non-`flash-attn` build (never even reads the manifest key in that
    // case, matching "flash arm cfg-off honest").
    let flash_dtypes: Vec<String> = if jammi_kernels::admission::FLASH_COMPILED {
        manifest_string_list(&manifest, "flash_dtypes")
    } else {
        Vec::new()
    };
    if jammi_kernels::admission::FLASH_COMPILED {
        assert!(
            !flash_dtypes.is_empty(),
            "{MANIFEST_LANE:?}'s flash_dtypes must be non-empty when flash_compiled is true"
        );
    }

    let device = Device::new_cuda(0).expect("skip_without_gpu already proved a CUDA device opens");
    let config = probe_config();
    let dir = TempDir::new().unwrap();
    let weights_path = dir.path().join("model.safetensors");
    write_synthetic_checkpoint(&config, &weights_path);

    let no_dispatch_site_ops = known_no_dispatch_site_ops(&manifest);
    let subkernel_ops = internal_subkernel_ops();
    let cascade_op_list = cascade_ops();
    let mapped: HashSet<String> = all_probed_report_keys()
        .into_iter()
        .map(str::to_string)
        .chain(STRICT_UNREACHABLE_OPS.iter().map(|s| (*s).to_string()))
        .chain(no_dispatch_site_ops.iter().cloned())
        .collect();

    for dtype in [
        ComputePrecision::F32,
        ComputePrecision::BF16,
        ComputePrecision::F16,
    ] {
        let candle_dtype = compute_precision_to_candle_dtype(dtype);
        // The two-arm rows this dtype class actually has a registry key for —
        // `cast_scale`/`cast_add` appear for bf16/f16 and are absent for f32
        // (the `admit()`-free "nothing to fuse" branch), which is exactly the
        // dtype-resolved fact the pre-#446 hand-encoded table could not
        // express.
        let two_arm_ops = two_arm_ops_for(dtype_class(dtype));
        // TIER PREEMPTION (Phase-4 audit finding 1): whether THIS dtype is
        // one the flash cascade is expected to preempt attention_block /
        // mem_efficient_attention for, on THIS build.
        let flash_should_preempt = flash_dtypes.iter().any(|d| d == &dtype.to_string());

        let two_arm_before: Vec<(&str, DispatchSnapshot)> = two_arm_ops
            .iter()
            .map(|&(_, key)| (key, jammi_kernels::admission::counters_for(key).snapshot()))
            .collect();
        let cascade_before: Vec<(&str, CascadeDispatchSnapshot)> = cascade_op_list
            .iter()
            .map(|&(_, key)| {
                (
                    key,
                    jammi_kernels::admission::cascade_counters_for(key).snapshot(),
                )
            })
            .collect();
        let attention_block_before =
            jammi_kernels::admission::counters_for("attention_block_fused").snapshot();
        let flash_cascade_before =
            jammi_kernels::admission::cascade_counters_for("attention_block_flash").snapshot();

        probe_dtype(&config, &weights_path, candle_dtype, &device);

        for &(report_op, registry_key) in &two_arm_ops {
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
                 hard-errored the probe under Strict) — before={before:?} after={after:?}"
            );
        }

        // `attention_block`: the TIER-PREEMPTION-AWARE pair of assertions —
        // both arms real, never logging-only (Phase-4 audit finding 1).
        if declared_ops.iter().any(|d| d == "attention_block") {
            let attention_block_after =
                jammi_kernels::admission::counters_for("attention_block_fused").snapshot();
            let flash_cascade_after =
                jammi_kernels::admission::cascade_counters_for("attention_block_flash").snapshot();
            if flash_should_preempt {
                assert!(
                    flash_cascade_after.fused > flash_cascade_before.fused,
                    "backbone_dtype={dtype} is a declared flash_dtype on a flash-attn build — \
                     the flash cascade must have dispatched FUSED at least once, got \
                     before={flash_cascade_before:?} after={flash_cascade_after:?}"
                );
                assert_eq!(
                    attention_block_after.fused, attention_block_before.fused,
                    "backbone_dtype={dtype}: attention_block_fused must NOT have dispatched — \
                     the flash cascade preempts it (forward_training_attention consults flash \
                     FIRST and returns on Fused) — before={attention_block_before:?} \
                     after={attention_block_after:?}"
                );
                assert_eq!(
                    attention_block_after.eager, attention_block_before.eager,
                    "backbone_dtype={dtype}: attention_block_fused must not have been reached \
                     AT ALL (fused or eager) when flash preempts it — \
                     before={attention_block_before:?} after={attention_block_after:?}"
                );
            } else {
                assert!(
                    attention_block_after.fused > attention_block_before.fused,
                    "backbone_dtype={dtype} (flash absent or not a declared flash_dtype): \
                     attention_block_fused must have dispatched FUSED at least once, got \
                     before={attention_block_before:?} after={attention_block_after:?}"
                );
                assert_eq!(
                    attention_block_after.eager, attention_block_before.eager,
                    "attention_block_fused must not have fallen back to eager at \
                     backbone_dtype={dtype} — before={attention_block_before:?} \
                     after={attention_block_after:?}"
                );
                assert_eq!(
                    flash_cascade_after.fused, flash_cascade_before.fused,
                    "backbone_dtype={dtype} does not preempt attention_block via flash on this \
                     build/dtype — the flash cascade must not have dispatched FUSED, got \
                     before={flash_cascade_before:?} after={flash_cascade_after:?}"
                );
            }
        }

        for &(report_op, registry_key) in &cascade_op_list {
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
                flash_should_preempt,
                ?before,
                ?after,
                "capability_surface: cascade op observed (informational only — see this \
                 file's `cascade_ops` doc for why it is not asserted Holds, including its own \
                 flash-preemption exposure)"
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

        // `internal_subkernels`: no admission gate of their own, but PROVEN
        // by the parent's fused dispatch — which the assertions above already
        // established for `low_rank_residual_linear` (and, for
        // `attention_block_flash`, the preemption arm). Logged with the
        // parent named, so a human reading `--nocapture` sees the actual
        // evidence chain rather than a bare "unverified".
        for &(op, parent) in &subkernel_ops {
            tracing::info!(
                op,
                parent,
                dtype = %dtype,
                "capability_surface: internal subkernel — launched unconditionally from inside \
                 its parent's already-admitted fused arm (a bare launcher call, not a tracked \
                 Tensor op), so it has NO registry key of its own; its execution is implied by \
                 the parent dispatching fused, never claimed as an independent admission"
            );
        }

        // Whatever the manifest's `fused_kernels_compiled` still lists that
        // PROBED_OPS names in no capacity at all (post-reclassification:
        // `axpy`). Not filtered by `declared_ops` — these live in a DIFFERENT
        // manifest list than `fused_op_admission`, and filtering them by it
        // would silently log nothing at all.
        for op in &no_dispatch_site_ops {
            tracing::info!(
                op,
                dtype = %dtype,
                "capability_surface: NO admission-gated dispatch site AND no probed parent \
                 exists anywhere in this workspace for this manifest-declared compiled kernel \
                 — it never dispatches through ANY probe, forward or backward, and nothing \
                 else's dispatch implies it ran; logged, never claimed covered by the \
                 completed-run argument"
            );
        }

        for op in &declared_ops {
            if !mapped.contains(op.as_str()) {
                tracing::info!(
                    op,
                    dtype = %dtype,
                    "capability_surface: no dedicated dispatch-registry mapping wired for this \
                     manifest-declared op in this file today — UNVERIFIED by this probe (never \
                     claimed as \"the completed run would have hard-errored\": that argument is \
                     only sound for an op this file has confirmed genuinely dispatches)"
                );
            }
        }
    }
}

/// Set-EQUALITY between `ci/release-feature-manifest.json`'s capability
/// categories and [`PROBED_OPS`] grouped by [`ProbedOpKind`] — the structural
/// guard that stops the manifest and the probed-op table from drifting apart
/// again (campaign #446 finding 2's root cause was five unsynced copies of
/// this same fact; the manifest was a sixth).
///
/// Deliberately NOT gated on a GPU (`skip_without_gpu!` is absent): this is a
/// pure data cross-check between a JSON file and a `const`, and gating it on
/// hardware would make the one assertion that catches manifest drift run only
/// on the pod. It still needs the `live-gpu-tests` feature to compile into
/// this suite at all, so `cargo test -p jammi-ai --features live-gpu-tests
/// --test gpu_capability` runs it anywhere.
///
/// The manifest reclassification this used to wait on has LANDED
/// (`ci/release-feature-manifest.json`, `969376cd`): `fused_op_admission`
/// gained `cast_scale`/`cast_add` (both DO admit, under dtype-resolved
/// registry keys — `low_rank_residual_linear.rs:800,814,899,911`) and
/// `adamw_step`; `internal_subkernels` was introduced; `fused_kernels_compiled`
/// keeps only `axpy`. This test is GREEN and is now a live drift guard rather
/// than a pending-work marker.
///
/// The three categories are read in the SHAPE each actually has —
/// `fused_op_admission`/`fused_kernels_compiled` as string lists,
/// `internal_subkernels` as an OBJECT keyed by op (see
/// [`manifest_internal_subkernels`], and `ci/scripts/check_release_manifest.py`
/// which validates each entry's `parent`/`launch_site`). That asymmetry is
/// deliberate on the manifest's side and is asserted THROUGH here, not
/// flattened away: a subkernel's `parent` is the only evidence it ran at all.
///
/// A RED here names exactly which side is missing which key (or which parent
/// the two disagree on); it is never a reason to weaken the assertion to a
/// subset check, because a subset check is precisely what let the f16 keys go
/// missing.
#[test]
fn manifest_capability_categories_match_probed_ops_by_kind() {
    let manifest = load_manifest();

    let mut expected_admission: Vec<&str> = PROBED_OPS
        .iter()
        .filter(|op| matches!(op.kind, ProbedOpKind::TwoArm | ProbedOpKind::Cascade))
        .map(|op| op.report_key)
        .collect();
    expected_admission.sort_unstable();
    let mut declared_admission = manifest_string_list(&manifest, MANIFEST_FUSED_OP_ADMISSION);
    declared_admission.sort();
    assert_eq!(
        declared_admission,
        expected_admission
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<String>>(),
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_FUSED_OP_ADMISSION} must name EXACTLY the \
         PROBED_OPS rows that dispatch through admit()/admit_cascade(). A key only the manifest \
         names is a capability nothing can prove; a key only PROBED_OPS names is a real \
         admission decision the release manifest does not claim (which is how the f16 \
         cast-epilogue keys went missing). See this test's doc for the pending manifest edit."
    );

    // `internal_subkernels` is checked on BOTH the op set AND the proof
    // relation. Set-equality alone would let the manifest name the right op
    // while attributing it to the wrong parent — and the parent IS the whole
    // evidence chain for these rows (they have no admission gate; "it ran" is
    // inferred entirely from the parent dispatching fused). A manifest that
    // said `scaled_cast_add`'s parent were, say, `attention_block_flash` would
    // be claiming a proof that does not exist, which set-equality could not
    // see. Comparing `(op, parent)` pairs makes the table and the manifest
    // agree on the relation, not just the name.
    let mut expected_subkernels: Vec<(String, String)> = internal_subkernel_ops()
        .into_iter()
        .map(|(op, parent)| (op.to_string(), parent.to_string()))
        .collect();
    expected_subkernels.sort();
    let declared_subkernels = manifest_internal_subkernels(&manifest);
    assert_eq!(
        declared_subkernels, expected_subkernels,
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_INTERNAL_SUBKERNELS} must name EXACTLY the \
         PROBED_OPS rows with no admission gate of their own, AND agree with the table on each \
         one's `parent` — the op whose fused dispatch is the only thing that proves the \
         subkernel ran. A name-only match with a wrong parent is a claimed proof that does not \
         exist."
    );

    // The compiled-but-unprovable bucket must be DISJOINT from everything
    // PROBED_OPS names: an op that is both "probed" and "no dispatch site" is
    // a contradiction, and the pre-#446 manifest had exactly that shape
    // (`cast_scale`/`scaled_cast_add`/`rope_positions` sat in
    // `fused_kernels_compiled` while two of the three were provable).
    let probed = all_probed_report_keys();
    let compiled_only = manifest_string_list(&manifest, MANIFEST_FUSED_KERNELS_COMPILED);
    let overlap: Vec<&String> = compiled_only
        .iter()
        .filter(|op| probed.contains(op.as_str()))
        .collect();
    assert!(
        overlap.is_empty(),
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_FUSED_KERNELS_COMPILED} must not name any \
         op PROBED_OPS can attribute — {overlap:?} are provable (as an admit() site or via a \
         parent's dispatch) and belong in {MANIFEST_FUSED_OP_ADMISSION} / \
         {MANIFEST_INTERNAL_SUBKERNELS} instead"
    );
    assert!(
        !compiled_only.is_empty(),
        "a {MANIFEST_FUSED_KERNELS_COMPILED} that is EMPTY would make the disjointness \
         assertion above vacuous — it must still name the compiled kernels with neither a \
         dispatch gate nor a probed parent (today: axpy)"
    );
}
