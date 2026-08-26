//! The deterministic jammi-vs-torch LEARNING oracle — one forward+backward
//! at IDENTICAL weights, compared by GRADIENT DIRECTION, never by loss
//! trajectory.
//!
//! ## Why this tier exists next to `finetune-step`
//!
//! [`crate::finetune_step`] (and `ci/scripts/perf/finetune_ab.sh`'s A/B
//! sweep over it) proves fused-vs-eager equivalence: same jammi build, one
//! kernel path forced on or off, elementwise-identical losses. That is
//! value-neutral evidence about FUSION, not about LEARNING — if jammi's
//! EAGER path itself computed a wrong gradient, that oracle stays green on
//! both arms, because both arms are wrong the same way.
//!
//! A jammi-vs-torch LOSS TRAJECTORY comparison (`finetune_ab.sh`'s printed
//! `loss_final_ratio`, `torch_finetune_step.py`'s own module doc) is NOT a
//! substitute, for two reasons that are structural, not incidental:
//!
//! 1. **The optimizer-update placement was off by one** (B1, now fixed —
//!    see [`crate::finetune_step::run`]'s own doc) — but even fixed, the
//!    trajectories diverge from step 0 for reason 2.
//! 2. **`--lora-init jammi` is DISTRIBUTION-matched, never BIT-matched**
//!    (`torch_finetune_step.py`'s "LoRA INIT IS NOT A MATCH BY DEFAULT"
//!    section): jammi draws `A` from a SplitMix64 stream keyed by
//!    `(seed, parameter name)`; torch draws from its own sequential
//!    generator. Same bound, different bits.
//!
//! Through a bf16 triplet hinge, those two facts alone separate ANY
//! multi-step trajectory permanently — a loss-trajectory comparison can
//! only ever catch a GROSS failure (jammi flat while torch learns), never
//! certify parity.
//!
//! ## The oracle this tier IS
//!
//! Compare gradients, not losses, at IDENTICAL weights:
//!
//! 1. Load the SAME base checkpoint on both stacks.
//! 2. Load the SAME LoRA `A`/`B` matrices from a shared file — this is the
//!    crux: it removes the init mismatch ENTIRELY, on both the value and
//!    the bit level, rather than only matching a distribution. See
//!    "Weight interchange format" below.
//! 3. Run ONE forward + backward on ONE identical batch, LoRA dropout
//!    forced to `0.0` (no per-framework RNG divergence inside the
//!    forward). NO optimizer step — a gradient-direction comparison does
//!    not need one, and skipping it keeps this tier a pure read of "what
//!    direction would this step move the weights", not a second-step
//!    trajectory question reason 2 above already rules out.
//! 4. Dump, per trainable tensor by name: the loss (a scalar, shared
//!    across every tensor since one forward produces one loss) and that
//!    tensor's gradient, as `f32`.
//!
//! A SEPARATE comparator (`ci/scripts/perf/compare_grad_oracle.py` —
//! deliberately Python/numpy, not Rust: family F's "numpy-first oracle"
//! convention, and this comparator's whole job is comparing two
//! INDEPENDENT dumps, so it must not share a code path with either
//! producer) matches tensors by name and reports max\|Δ\|, max\|Δ\|/max
//! \|signal\|, and cosine similarity — per tensor and overall. Cosine is
//! the LEARNING-DIRECTION metric: two stacks can differ by bf16 rounding
//! and still train identically if the gradient direction agrees; a
//! max\|Δ\| bound alone cannot distinguish "rounding noise" from "wrong
//! sign on a whole tensor" the way cosine does.
//!
//! ## Weight interchange format — the crux
//!
//! The shared file is a plain `safetensors` file, written and read via
//! `candle_nn::VarMap::save`/`VarMap::load` UNCHANGED (no new candle/jammi
//! API — see those methods' own doc) — so on the JAMMI side, "load the same
//! weights" is a straight `VarMap::load(path)` call: it matches tensors to
//! the ALREADY-REGISTERED `Var`s by NAME and overwrites their storage in
//! place, preserving `Var` identity (so `backward()` still tracks them).
//! The names it matches on are jammi's OWN internal `VarBuilder` path
//! naming — e.g. `layer.3.Wqkv.lora_a` — not any PEFT-style name. This
//! tier does zero name translation: the FIRST invocation (no
//! `--lora-weights-in`) uses its own seeded init and can `--lora-weights-out`
//! that exact file for a SECOND jammi invocation to `--lora-weights-in`
//! load — a jammi-vs-jammi round trip proves the interchange mechanism
//! itself is lossless (see `grad_oracle_self_consistency_round_trip`
//! below) independent of ever bringing torch into the picture.
//!
//! For a jammi-vs-TORCH comparison, the torch-side reference script is
//! responsible for translating between PEFT's own `named_parameters()`
//! naming (`base_model.model.layers.{n}.{attn|mlp}.{Wqkv|Wo|Wi}.lora_A.default.weight`,
//! shape `[rank, in_features]`, matching jammi's `lora_a` orientation
//! exactly) and jammi's naming above, in BOTH directions — see
//! `crates/jammi-bench/reference/torch_grad_oracle.py`'s own module doc
//! for the exact table. **That script is UNTESTED in this round** — no
//! `torch` install was available in this environment; see this crate's
//! test-suite doc / the dispatch verdict for what WAS exercised
//! (jammi-side unit tests only, CPU/F32, the tiny fixture).
//!
//! ## Structural limitation: a single fresh-init call tests ONLY `dL/dB`
//!
//! At [`jammi_lora::LoraInitMode::ZerosB`] (this tier's only mode — see
//! [`GradOracleParams::lora_weights_in`]'s doc), `B` starts at the exact
//! zero matrix. The LoRA forward is `base(x) + scaling *
//! dropout(x @ A^T @ B^T)`; the chain rule routes `dL/dA` through `B^T @
//! dL/d(output)`, which is the ZERO matrix whenever `B == 0`, for ANY value
//! of `A`, on BOTH stacks, REGARDLESS of whether either stack's backward
//! arithmetic is actually correct there. Confirmed empirically on a live
//! A100 run (ModernBERT-large, tip `e62c8a8`): every `lora_a` tensor's
//! gradient measured EXACTLY `0.0` on both the jammi and the torch dump
//! (112 of 224 matched tensors that run). A single forward+backward at a
//! fresh init therefore provides ZERO evidence about whether jammi's and
//! torch's `dL/dA` computations agree — a real defect specific to that
//! path (a transposed axis, a dropped scale factor) could NOT be caught
//! this way; it would read as the same uninformative, structurally
//! guaranteed cosine of `0.0` whether the two stacks agree or not.
//! `compare_grad_oracle.py`'s `is_vacuous_pair`/`vacuous_tensor_count`
//! classify and surface exactly this case rather than let it masquerade as
//! either a pass or a fail signal. Catching a real `dL/dA` defect needs AT
//! LEAST one optimizer step first (moving `B` away from zero) — see "What
//! this tier does NOT do" below for the N-step extension that would close
//! this gap; not implemented this round.
//!
//! ## What this tier does NOT do (design, not shipped, this round)
//!
//! Extending to N steps in TEACHER-FORCED form — after each step,
//! overwrite one side's weights with the other's, so both always take the
//! next step from identical state, measuring PER-STEP divergence without
//! chaotic accumulation — is a real, useful extension, and the CLI/report
//! shape here is deliberately structured (one `GradOracleReport` per call,
//! `--lora-weights-out` writing the POST-this-forward's weights are NOT
//! written here since no optimizer step ran — a future step would need an
//! `AdamW::new`+`.step()` call added and the updated `VarMap` re-dumped)
//! so that extension is a thin wrapper around repeated single-step calls,
//! not a rewrite. Not implemented this round.
//!
//! ## Determinant table — every field either producer emits, classified
//!
//! `ci/scripts/perf/compare_grad_oracle.py`'s `_premise_violations`
//! certifies that two dumps were produced under IDENTICAL premises. That
//! certification is only as complete as the field list it actually checks.
//! This table enumerates EVERY output-affecting determinant either producer
//! emits, classified as:
//!
//! - **identity** — must match across the two dumps for the comparison to
//!   mean anything; a mismatch is a hard premise violation
//!   (`compare_grad_oracle.RUN_IDENTITY_FIELDS` — the single source of
//!   truth both `_premise_violations`'s per-field loop and
//!   `test_compare_grad_oracle.py::RunIdentityFieldCanonicalizationLattice`
//!   iterate, never a second, hand-maintained field list).
//! - **provenance** — recorded, reported, NEVER compared: legitimately
//!   differs across two independent producers/boxes (e.g. device model,
//!   library versions), and comparing it would either always fail (two
//!   different stacks never share a torch version) or be meaningless.
//! - **measurement** — this run's OWN output (loss, gradients, dispatch
//!   counters); the thing the oracle exists to compare or report, not a
//!   premise the comparison depends on.
//!
//! | field | class | jammi emit site | torch emit site |
//! |---|---|---|---|
//! | `seed` | identity | `grad_oracle.rs:GradOracleReport::seed` field, `run()`'s report literal | `"seed": args.seed` (`torch_grad_oracle.py:538`) |
//! | `batch` | identity | `run()`'s report literal | `"batch": args.batch` (`torch_grad_oracle.py:532`) |
//! | `seq` | identity | `run()`'s report literal | `"seq": args.seq` (`torch_grad_oracle.py:533`) |
//! | `lora_rank` | identity | `run()`'s report literal | `"lora_rank": args.lora_rank` (`torch_grad_oracle.py:534`) |
//! | `lora_alpha` | identity | `run()`'s report literal | `"lora_alpha": args.lora_alpha` (`torch_grad_oracle.py:535`) |
//! | `target_modules` | identity | `run()`'s report literal | `"target_modules": [t.strip()` (`torch_grad_oracle.py:536`) |
//! | `batched_forward` | identity | `run()`'s report literal | `"batched_forward": args.batched_forward` (`torch_grad_oracle.py:537`) |
//! | `backbone_dtype` | identity | `run()`'s report literal (`format!("{:?}", ..).to_lowercase()`) | `translate_dtype_flag_to_jammi_spelling(args.dtype)` (`torch_grad_oracle.py:531`) |
//! | `checkpoint_config_sha256` | identity | `sha256_and_len(&model_dir.join("config.json"))` — called in `run()` before the forward, via the SAME shared streaming implementation `finetune_step.rs` also uses: `pub(crate) fn sha256_and_len` (`finetune_step.rs:700`) | `checkpoint_identity_fields = checkpoint_identity(args.model_dir)` (`torch_grad_oracle.py:413`) — `checkpoint_identity` is a bare alias for the real, streaming implementation torch_finetune_step.py's own `checkpoint_identity` function provides (see the two field citations directly below) |
//! | `checkpoint_weights_sha256` | identity | `sha256_and_len(&weights)` | `"checkpoint_weights_sha256": weights_sha256` (`torch_finetune_step.py:551`) |
//! | `checkpoint_weights_size_bytes` | identity | `sha256_and_len`'s byte-length return | `"checkpoint_weights_size_bytes": weights_len` (`torch_finetune_step.py:552`) |
//! | `lora_weights_in` (presence, not value) | identity (checked separately — `_premise_violations`'s `lora_weights_in` loop, not `RUN_IDENTITY_FIELDS`) | `run()`'s report literal | `torch_grad_oracle.py`'s report literal |
//! | `batch_token_id_sums` | identity (checked separately, `or`-gated presence) | `run()`'s report literal | `torch_grad_oracle.py`'s report literal |
//! | `model_dir` | provenance (human debugging only — a path string is not comparable across two boxes; superseded by the two checksum fields above) | `run()`'s report literal | `torch_grad_oracle.py`'s report literal |
//! | `device` / `device_name` | provenance | `run()`'s report literal (`device_name` reuses `finetune_step::device_name`) | `"provenance": tfs.provenance(device, fast_path_globals)` (`torch_grad_oracle.py:507`) |
//! | `git_rev` (jammi) / `provenance.git_rev` (torch) | provenance | `tip_sha()`, called in `run()`'s report literal | `torch_finetune_step.py`'s `git_rev()`, via `provenance()` |
//! | torch/transformers/peft versions | provenance (jammi has no equivalent — no torch/transformers/peft dependency) | n/a | same call site as the `device` row directly above (`torch_grad_oracle.py`'s `provenance` field) |
//! | `attn_requested` / `attn_implementation` | provenance (jammi has no `--attn` lever; its own analog is the MEASUREMENT dispatch counters below) | n/a | `"attn_requested": args.attn` (`torch_grad_oracle.py:516`), `"attn_implementation": resolved_attn_implementation` (`torch_grad_oracle.py:517`), resolved in `run()` mirroring the identical pattern `torch_finetune_step.py`'s own `run()` already established (see `ab_merge.py`'s determinant table for that file's own citations of this exact pair) |
//! | `lora_dropout` | identity, but UNCONDITIONALLY forced to `0.0` by both producers so it can never legitimately differ — excluded from `RUN_IDENTITY_FIELDS` on that basis, not compared | `run()`'s report literal (hardcoded `0.0`) | `torch_grad_oracle.py`'s report literal (hardcoded `0.0`) |
//! | `trainable_tensor_count` | measurement (redundant with the tensor NAME SET, which `compare_reports`'s `only_in_a`/`only_in_b` already checks structurally) | `run()`'s report literal | `torch_grad_oracle.py`'s report literal |
//! | `loss` / `gradients` / per-tensor `weight` | measurement — the oracle's actual output | `run()`'s report literal | `torch_grad_oracle.py`'s report literal |
//! | `ln`/`rope`/`softmax`/`geglu`/`lora_epilogue`/`lora_linear`/`attention_block` `_fused_dispatches`/`_eager_dispatches` (14 fields) | measurement (jammi-only; no torch equivalent — torch's analog is the `attn_requested`/`attn_implementation` provenance pair above) | `run()`'s dispatch-counter delta, mirroring `finetune_step.rs`'s own `*_dispatch_before`/`*_dispatch_after` snapshot pattern | n/a |
//! | `kernels_disabled_requested`/`kernels_disabled_fired` | provenance — K-aux (`feat/kernels-admission-disable`) landed on `main` at `c0f0e98`; this tier now records the resolved `JAMMI_KERNELS_DISABLE` state unconditionally, mirroring `FinetuneStepTier`'s own pair exactly, but does NOT gate on `unmatched_disables()` the way `finetune_step.rs`'s `run()` does (that INVALID-run check is scoped to the forced-eager A/B use case this oracle's own CLI has no equivalent flag for) | `run()`'s report literal, via `jammi_kernels::admission::disabled_ops_requested`/`disabled_ops_fired` | n/a (torch has no equivalent env var) |
//! | `tool` | identity, but only for SAME-vs-DIFFERENT-producer detection, not compared as a normal identity field — `compare_grad_oracle.py`'s `_same_producer_violation` refuses when both dumps carry the SAME `tool` string (`compare a.json a.json`, or a jammi-vs-jammi mix-up), overridable via `--allow-same-producer` for a deliberate self-consistency check | `run()`'s report literal (`"jammi_grad_oracle"`) | `torch_grad_oracle.py`'s report literal (`"torch_grad_oracle"`) |
//!
//! `RUN_IDENTITY_FIELDS` in `compare_grad_oracle.py` is the tuple that
//! actually encodes the **identity** rows above (the `lora_weights_in`
//! presence check and `batch_token_id_sums` equality check are separate,
//! purpose-built checks in `_premise_violations`, not members of that
//! tuple) — `test_grad_oracle_cross_producer_parity.py`'s
//! `test_run_identity_key_set_present_on_both_real_dumps` asserts every
//! entry is PRESENT on a REAL dump from EACH producer, and
//! `test_compare_grad_oracle.py::RunIdentityFieldCanonicalizationLattice::test_every_run_identity_field_has_a_lattice_cell`
//! fails loudly if a field is added to the tuple without per-field test
//! coverage.

use std::path::PathBuf;

use crate::finetune_step::{device_name, sha256_and_len, synthetic_ids, triplet_loss};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Optimizer, VarMap};
use serde::Serialize;

/// How this tier seeds LoRA `A`/`B` on a FRESH (no `--lora-weights-in`)
/// call — a jammi-bench-local enum, NOT `jammi_lora::LoraInitMode`
/// (2 variants: `ZerosB`/`Gaussian`), because `PeftStep1` is a PROCEDURE
/// (an init plus one real optimizer step), not a distribution jammi-lora
/// itself knows how to draw — see [`GradOracleParams::lora_init`]'s doc for
/// the full rationale of each variant, and [`peft_step1_weights`] for the
/// procedure `PeftStep1` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradOracleLoraInit {
    ZerosB,
    Gaussian,
    PeftStep1,
}

impl GradOracleLoraInit {
    /// Parses the `--lora-init` CLI flag's value. Shared by `main.rs` (both
    /// the plain `grad-oracle` path and `--ablate-each-op`) so there is
    /// exactly one spelling table, never two independently-drifting
    /// `match` arms.
    pub fn from_flag(s: &str) -> Result<Self, String> {
        match s {
            "zeros-b" => Ok(Self::ZerosB),
            "gaussian" => Ok(Self::Gaussian),
            "peft-step1" => Ok(Self::PeftStep1),
            other => Err(format!(
                "unknown lora_init {other:?}; expected gaussian, zeros-b, or peft-step1"
            )),
        }
    }

    pub fn as_flag(self) -> &'static str {
        match self {
            Self::ZerosB => "zeros-b",
            Self::Gaussian => "gaussian",
            Self::PeftStep1 => "peft-step1",
        }
    }

    /// The `jammi_lora::LoraInitMode` this variant's INITIAL draw uses —
    /// `PeftStep1` maps to `ZerosB` (PEFT's own default: `A` Kaiming-
    /// uniform, `B` zeros) because the "PEFT" part of `PeftStep1` IS
    /// `LoraInitMode::ZerosB`; what makes it different from plain `ZerosB`
    /// is the one optimizer step [`peft_step1_weights`] takes afterward,
    /// not a different initial distribution.
    fn initial_jammi_mode(self) -> jammi_lora::LoraInitMode {
        match self {
            Self::ZerosB | Self::PeftStep1 => jammi_lora::LoraInitMode::ZerosB,
            Self::Gaussian => jammi_lora::LoraInitMode::Gaussian,
        }
    }
}

/// Parameters the oracle drives its single forward+backward off of.
#[derive(Debug, Clone)]
pub struct GradOracleParams {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    pub cuda_device: Option<usize>,
    /// Drives the synthetic batch (`synthetic_ids(seed + i, ..)`, `i` in
    /// `0..3`) AND, when `lora_weights_in` is `None`, the fresh LoRA `A`
    /// draw (`LoraInitMode::ZerosB`, jammi's own SplitMix64 stream) — same
    /// role `FinetuneStepParams::seed` plays in `finetune_step.rs`.
    pub seed: u64,
    pub batched_forward: bool,
    /// A safetensors file (jammi's OWN internal `VarBuilder`-path naming —
    /// see this module's doc) to `VarMap::load` BEFORE the forward,
    /// overwriting the freshly-seeded LoRA `A`/`B` values in place. `None`
    /// keeps the fresh seeded draw (useful for the FIRST invocation that
    /// produces the shared file via `lora_weights_out`).
    pub lora_weights_in: Option<PathBuf>,
    /// Where to `VarMap::save` the LoRA weights ACTUALLY used for this
    /// forward (post-`lora_weights_in` load, if any) — lets a first
    /// invocation seed the shared file a second, independent invocation
    /// then loads via `lora_weights_in`.
    pub lora_weights_out: Option<PathBuf>,
    /// How the LoRA `A`/`B` matrices are seeded on a FRESH (no
    /// `lora_weights_in`) call. See this module's doc's "Structural
    /// limitation: a single fresh-init call tests ONLY dL/dA" section:
    /// under `ZerosB`, `B` starts at the exact zero matrix, so `dL/dA` is
    /// IDENTICALLY zero on every `lora_a` tensor regardless of whether
    /// either stack's backward arithmetic is correct there — a single
    /// forward+backward at that init is STRUCTURALLY BLIND to a `dL/dA`
    /// defect. `Gaussian` (both `A` and `B` drawn from `Normal(0, 0.02)`)
    /// starts both factors nonzero, so `dL/dA` and the LoRA epilogue's own
    /// gradient term are live from a single call — a DIAGNOSTIC operating
    /// point (round-7 audit finding on PR #383: `alpha/rank = 2` with both
    /// factors at `N(0, 0.02)` is a loss-surface region real training never
    /// occupies). `PeftStep1` is the REALISTIC operating point instead
    /// (`--ablate-each-op`'s own default as of this round): PEFT's own
    /// init (`ZerosB`), one real `AdamW` step at the reference lr
    /// (`peft_step1_weights`'s own doc) on this call's own synthetic batch,
    /// THEN the measured forward+backward — real training's first LIVE-
    /// gradient point, not a synthetic distribution nothing ever trains
    /// from. See [`GradOracleReport::vacuous_tensor_count`] for the
    /// non-vacuous certification every mode still carries (only `ZerosB`
    /// is expected to fail it).
    pub lora_init: GradOracleLoraInit,
}

/// Below this L2 norm, a gradient vector is treated as (numerically) the
/// zero vector — mirrors `compare_grad_oracle.py`'s own `NORM_FLOOR`
/// constant (`1e-12`) EXACTLY, so a tensor this crate calls "vacuous" and a
/// tensor the Python comparator calls "vacuous" are the SAME classification,
/// never two independently-drifting floors that could disagree on a
/// borderline tensor.
const VACUOUS_NORM_FLOOR: f64 = 1e-12;

/// `true` iff `grad`'s L2 norm is below [`VACUOUS_NORM_FLOOR`] — computed in
/// `f64` (never `f32`) so a long `f32` gradient vector's own summation
/// rounding cannot itself manufacture a false "vacuous" reading at this
/// floor.
fn grad_is_vacuous(grad: &[f32]) -> bool {
    let sum_sq: f64 = grad.iter().map(|&g| (g as f64) * (g as f64)).sum();
    sum_sq.sqrt() < VACUOUS_NORM_FLOOR
}

/// A `(fused, eager)` dispatch-count pair for one `jammi_kernels::admission`
/// op key — see [`GradOracleReport::admit_key_dispatches`]'s own doc.
#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize)]
pub struct DispatchPair {
    pub fused: u64,
    pub eager: u64,
}

/// One trainable tensor's dumped gradient (and, for the non-vacuous check
/// described in this module's doc, the exact weight value the forward
/// actually used) — `f32` regardless of `backbone_dtype`: the D2H read
/// widens STORAGE, it does not add mantissa bits the compute dtype lacked
/// (same convention `finetune_step.rs`'s `losses` field doc states).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct GradOracleTensor {
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
    pub weight: Vec<f32>,
}

/// The oracle's full dump. See this module's doc for the placement/format
/// contract every field below carries.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct GradOracleReport {
    // `String`, not `&'static str`: a round-tripped `Deserialize` (see
    // `grad_oracle_ablation::spawn_arm`, which reads a child arm's dump
    // back into an owned `GradOracleReport`) cannot borrow a `&'static
    // str` out of a local, non-`'static` JSON text buffer — `serde_json::
    // from_str`'s zero-copy borrowed-deserialize path for a `&'a str`
    // field ties that borrow to the INPUT buffer's own lifetime, which a
    // temporary `String` read from disk never satisfies. The wire format
    // is unaffected (`serde_json` serializes `String` and `&str`
    // identically); only the in-memory representation changed.
    pub tool: String,
    /// Human-readable, for debugging only — NOT a comparator identity field
    /// (a path string is not comparable across two boxes/producers; see this
    /// module's doc's determinant table). `checkpoint_config_sha256`/
    /// `checkpoint_weights_sha256` below are the field the comparator's
    /// premise actually depends on.
    pub model_dir: String,
    pub device: String,
    /// The concrete device sub-class (`finetune_step.rs`'s own
    /// `device_name`, reused unchanged — see this module's doc's
    /// determinant table). PROVENANCE, never compared: two producers
    /// legitimately run on different device models.
    pub device_name: String,
    /// Best-effort `git rev-parse HEAD` against this crate's own directory —
    /// `None` if `git` is unavailable or this binary is running outside a
    /// git worktree. PROVENANCE, never compared (mirrors
    /// `torch_finetune_step.py`'s `git_rev`).
    pub git_rev: Option<String>,
    /// sha256 of `model_dir/config.json`'s raw bytes — half of the base
    /// checkpoint's CONTENT identity (see this module's doc's determinant
    /// table). IDENTITY: both producers must have loaded the byte-identical
    /// checkpoint for a gradient comparison to mean anything.
    pub checkpoint_config_sha256: String,
    /// sha256 of `model_dir/model.safetensors`'s raw bytes — the other half
    /// of the base checkpoint's CONTENT identity. IDENTITY.
    pub checkpoint_weights_sha256: String,
    /// `model_dir/model.safetensors`'s byte length — a cheap, redundant
    /// cross-check alongside the sha256 above (a size mismatch is a coarser,
    /// faster-to-eyeball signal of "not the same file" than a hex digest).
    /// IDENTITY.
    pub checkpoint_weights_size_bytes: u64,
    pub backbone_dtype: String,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    pub batched_forward: bool,
    pub seed: u64,
    pub lora_dropout: f64,
    /// `"zeros-b"` | `"gaussian"` | `"peft-step1"` — [`GradOracleLoraInit::as_flag`].
    /// IDENTITY within a same-producer comparison (the ablation
    /// orchestrator's whole premise depends on every arm having used the
    /// SAME init procedure); no torch-side equivalent.
    pub lora_init: String,
    /// `true` iff this run synthesized its OWN LoRA weights via
    /// [`peft_step1_weights`] (i.e. `lora_init == "peft-step1"` AND no
    /// `--lora-weights-in` was supplied) rather than loading a caller-
    /// supplied file. `lora_weights_in` below stays `None` in that case —
    /// the synthesized file lives in a `TempDir` that is deleted before
    /// this function returns, so reporting its path would name a file that
    /// no longer exists.
    pub peft_step1_applied: bool,
    pub lora_weights_in: Option<String>,
    pub lora_weights_out: Option<String>,
    pub trainable_tensor_count: usize,
    /// `[sum(anchor ids), sum(positive ids), sum(negative ids)]` — a cheap,
    /// deterministic digest of the THREE synthetic batches this call
    /// actually fed the encoder (`synthetic_ids(.., seed + i, ..)` for `i`
    /// in `0..3`), exposed so a caller (or a test — see
    /// `grad_oracle_batch_group_offsets_match_synthetic_ids_seed_plus_i`
    /// below) can verify the group-selection arithmetic independently of
    /// this report's `loss`/`gradients`, which do not otherwise reveal
    /// which tokens produced them.
    pub batch_token_id_sums: [u64; 3],
    /// This call's ONE loss value — shared across every tensor in
    /// `gradients` (one forward, one loss, `gradients.len()` backward
    /// destinations), never per-tensor.
    pub loss: f32,
    // The 14 process-wide dispatch-counter fields (7 op families x
    // fused/eager), a snapshot DELTA taken around this call's ONE
    // forward+backward — the SAME shape `finetune_step.rs`'s own
    // `*_fused_dispatches`/`*_eager_dispatches` fields carry (see this
    // module's doc's determinant table), so a jammi-side dump also records
    // WHICH kernel composition actually ran (fused whole-attention-block /
    // fused LoRA site where eligible), not just that a forward+backward
    // happened. MEASUREMENT, never compared cross-producer: torch has no
    // equivalent counter (its own analog is `attn_implementation`, a
    // torch-only PROVENANCE field on that side).
    pub ln_fused_dispatches: u64,
    pub ln_eager_dispatches: u64,
    pub rope_fused_dispatches: u64,
    pub rope_eager_dispatches: u64,
    pub softmax_fused_dispatches: u64,
    pub softmax_eager_dispatches: u64,
    pub geglu_fused_dispatches: u64,
    pub geglu_eager_dispatches: u64,
    pub lora_epilogue_fused_dispatches: u64,
    pub lora_epilogue_eager_dispatches: u64,
    pub lora_linear_fused_dispatches: u64,
    pub lora_linear_eager_dispatches: u64,
    pub attention_block_fused_dispatches: u64,
    pub attention_block_eager_dispatches: u64,
    /// The `JAMMI_KERNELS_DISABLE` op keys this process REQUESTED (sorted,
    /// empty when the env var was unset or empty) — K-aux lands on `main`
    /// this round; mirrors `FinetuneStepTier::kernels_disabled_requested`
    /// exactly (`jammi_kernels::admission::disabled_ops_requested`).
    /// PROVENANCE (recorded, never compared cross-producer — torch has no
    /// equivalent env var).
    pub kernels_disabled_requested: Vec<String>,
    /// The `JAMMI_KERNELS_DISABLE` op keys that actually FIRED (disabled at
    /// least one live dispatch) this run (sorted) — mirrors
    /// `FinetuneStepTier::kernels_disabled_fired` exactly
    /// (`jammi_kernels::admission::disabled_ops_fired`). PROVENANCE. This
    /// tier does NOT gate on `jammi_kernels::admission::unmatched_disables`
    /// the way `finetune_step.rs`'s `run()` does (contract K-aux's INVALID-run
    /// check is scoped to that tier's forced-eager A/B use case, which this
    /// oracle's own CLI has no equivalent flag for) — recorded unconditionally,
    /// same posture as the 14 dispatch counters above.
    pub kernels_disabled_fired: Vec<String>,
    /// Every `jammi_kernels::admission` op key that dispatched FUSED at
    /// least once during this call's forward+backward (a snapshot-delta
    /// read of `jammi_kernels::admission::snapshot_all()`, taken around the
    /// same forward+backward the 14 explicit dispatch-counter fields above
    /// bracket), sorted. This is the "registered admit key" enumeration
    /// [`crate::grad_oracle_ablation`]'s `--ablate-each-op` orchestrator
    /// drives its per-op ablation loop off of, in place of a hand-maintained
    /// literal key list: `jammi_kernels::admission`'s own module doc
    /// documents that not every op key in the crate's call graph reaches
    /// `admit` on every run (a `"lora_epilogue"`/`"lora_dropout"`-style
    /// registered-but-permanently-dead name, or a SUBSUMED key like
    /// `"rope_fused"`/`"softmax_last_dim_fused"` that never fires while
    /// `"attention_block_fused"` admits) — reading the keys that ACTUALLY
    /// fired FUSED on this exact checkpoint/config, rather than assuming a
    /// fixed set, is what keeps the ablation loop from wasting a run on an
    /// op key `unmatched_disables()` would reject as an invalid leg.
    /// PROVENANCE (a fact about which kernels this run reached, not a value
    /// either producer's gradient math depends on).
    pub live_admit_keys: Vec<String>,
    /// The RAW `(fused, eager)` snapshot-delta for EVERY op key
    /// `jammi_kernels::admission::snapshot_all()` observed this run (a
    /// superset of [`Self::live_admit_keys`]: an op key can appear here
    /// with `fused == 0` — e.g. disabled-and-therefore-always-eager, or a
    /// SUBSUMED key that only reaches `admit` once its subsuming parent is
    /// ALSO disabled — without qualifying as "live"). Round-7 audit finding
    /// (PR #383): `grad_oracle_ablation`'s per-op table needs each arm's
    /// OWN dispatch counters for the key it claims to have ablated, not
    /// just a boolean "did this key fire fused ANYWHERE this run" — an
    /// `ablate:<key>` arm whose OWN `admit_key_dispatches[key].fused > 0`
    /// is a hard contradiction (the disable did not actually take effect
    /// for every call) and [`crate::grad_oracle_ablation::run`] refuses on
    /// exactly that condition.
    pub admit_key_dispatches: std::collections::BTreeMap<String, DispatchPair>,
    /// How many of [`Self::gradients`]' tensors are [`grad_is_vacuous`] —
    /// the fixture's own non-vacuous-control self-check (family F): under
    /// `LoraInitMode::Gaussian` this must be `0` (both `dL/dA` and `dL/dB`
    /// are live from a single fresh forward+backward — see
    /// [`GradOracleParams::lora_init`]'s doc); under the legacy `ZerosB`
    /// mode every `lora_a` tensor is STRUCTURALLY vacuous by construction,
    /// so this is expected to be nonzero there. RECORDED, never gated by
    /// `run()` itself (the same "records, does not gate" posture every
    /// field on this report takes) — a caller with a non-vacuous premise to
    /// certify (`grad_oracle_ablation::run`) is the one that asserts `== 0`
    /// and refuses to proceed otherwise.
    pub vacuous_tensor_count: usize,
    /// The tensor names counted in [`Self::vacuous_tensor_count`], sorted —
    /// so a caller's refusal message can name exactly which tensor(s) carry
    /// no gradient signal, rather than only the count.
    pub vacuous_tensor_names: Vec<String>,
    /// Keyed by jammi's internal `VarBuilder`-path tensor name (e.g.
    /// `layer.3.Wqkv.lora_a`), sorted for determinism (a `BTreeMap`
    /// serializes in key order; a `HashMap` would not).
    pub gradients: std::collections::BTreeMap<String, GradOracleTensor>,
}

/// Best-effort `git rev-parse HEAD` run against THIS crate's own directory
/// (`CARGO_MANIFEST_DIR`) — mirrors `torch_finetune_step.py`'s `git_rev`
/// exactly (PROVENANCE, never gates anything, never a hard error): `None` if
/// `git` is not on `PATH`, this binary is running outside a git worktree
/// (e.g. a packaged/vendored copy), or the command otherwise fails.
fn tip_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Real training's FIRST live-gradient point: PEFT's own init (`ZerosB`),
/// one real `AdamW` step on THIS call's own synthetic batch
/// (`synthetic_ids(.., params.seed + i, ..)`, `i` in `0..3` — the SAME
/// batch `run()`'s own measured forward will reuse, since `params.seed` is
/// unchanged between this call and the one that loads its output — see
/// `GradOracleLoraInit::PeftStep1`'s own doc: "on the SAME data, then dump
/// gradients on the second step"), then writes the POST-step LoRA weights
/// to `out_path`. Deliberately duplicates (never shares) the model-
/// construction prologue `run()` itself uses below — the same
/// `finetune_step.rs::build_fixture`-style duplication this crate already
/// takes for its own per-tier fixtures, rather than a shared helper with a
/// growing parameter list neither call site fully needs.
///
/// `lr = 2e-4`, the SAME reference lr `finetune_step.rs::build_fixture`
/// uses (`crates/jammi-bench/src/finetune_step.rs`) — not a new number
/// invented for this tier.
const PEFT_STEP1_REFERENCE_LR: f64 = 2e-4;

fn peft_step1_weights(
    params: &GradOracleParams,
    out_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
    };
    let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))?;
    let config: jammi_encoders::ModernBertConfig = serde_json::from_str(&config_raw)?;
    let weights = params.model_dir.join("model.safetensors");

    let varmap = VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &params.target_modules,
        layers_to_transform: &None,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &empty_ranks,
        init_mode: jammi_lora::LoraInitMode::ZerosB,
        seed: params.seed,
    };
    let mut encoder = jammi_encoders::ModernBert::builder()
        .pooling(jammi_encoders::Pooling::Mean)
        .backbone_dtype(jammi_encoders::compute_precision_to_dtype(
            params.backbone_dtype,
        ))
        .lora(lora)
        .build(&[weights.as_path()], &config, &device, &varmap)?;
    encoder.set_training(true);

    let trainable = varmap.all_vars();
    if trainable.is_empty() {
        return Err(
            "peft_step1_weights: no trainable LoRA tensors -- target_modules matched nothing"
                .into(),
        );
    }
    let mut opt = candle_nn::AdamW::new(
        trainable,
        candle_nn::ParamsAdamW {
            lr: PEFT_STEP1_REFERENCE_LR,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    let mask = Tensor::ones((params.batch, params.seq), DType::U32, &device)?;
    let blocks: Vec<Tensor> = (0..3)
        .map(|i| {
            synthetic_ids(
                params.batch,
                params.seq,
                config.vocab_size,
                params.seed + i,
                &device,
            )
        })
        .collect();

    let (a, p, n) = if params.batched_forward {
        let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0)?;
        let joined_mask = Tensor::cat(&[&mask, &mask, &mask], 0)?;
        let all = encoder.forward(&joined, &joined_mask)?;
        let b = params.batch;
        (
            all.narrow(0, 0, b)?,
            all.narrow(0, b, b)?,
            all.narrow(0, 2 * b, b)?,
        )
    } else {
        (
            encoder.forward(&blocks[0], &mask)?,
            encoder.forward(&blocks[1], &mask)?,
            encoder.forward(&blocks[2], &mask)?,
        )
    };
    let loss = triplet_loss(&a, &p, &n, 0.3)?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    varmap.save(out_path)?;
    Ok(())
}

/// Run the oracle and return its report. The MEASURED step itself takes NO
/// optimizer step (`PeftStep1`'s own step happens in [`peft_step1_weights`]
/// BEFORE this function's own forward — see this module's doc for why a
/// gradient-direction comparison at the MEASURED step does not need one of
/// its own).
pub fn run(params: &GradOracleParams) -> Result<GradOracleReport, Box<dyn std::error::Error>> {
    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
    };
    let device_label = match params.cuda_device {
        Some(o) => format!("cuda:{o}"),
        None => "cpu".to_string(),
    };

    let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))?;
    let config: jammi_encoders::ModernBertConfig = serde_json::from_str(&config_raw)?;
    let weights = params.model_dir.join("model.safetensors");

    // Base-checkpoint CONTENT identity — computed BEFORE the forward, off
    // the exact bytes this run loads, so it can never drift from what the
    // model actually built from (see this module's doc's determinant
    // table).
    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&params.model_dir.join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) = sha256_and_len(&weights)?;

    let mut varmap = VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &params.target_modules,
        layers_to_transform: &None,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        use_rslora: false,
        // Forced 0.0, unconditionally — a gradient-direction comparison
        // that let dropout differ between the two stacks would compare
        // gradients of DIFFERENT computations, not the same one through
        // two arithmetic paths. This tier has no CLI knob for this on
        // purpose.
        lora_dropout: None,
        rank_pattern: &empty_ranks,
        init_mode: params.lora_init.initial_jammi_mode(),
        seed: params.seed,
    };

    let mut encoder = jammi_encoders::ModernBert::builder()
        .pooling(jammi_encoders::Pooling::Mean)
        .backbone_dtype(jammi_encoders::compute_precision_to_dtype(
            params.backbone_dtype,
        ))
        .lora(lora)
        .build(&[weights.as_path()], &config, &device, &varmap)?;
    // Training mode: dropout is forced off above (`lora_dropout: None`), so
    // this only selects the training-arm KERNEL COMPOSITION (fused
    // whole-attention-block / fused LoRA site where eligible), not any
    // additional randomness — the point is comparing the SAME arithmetic
    // shape jammi actually trains with, not the (dropout-free either way)
    // eval-mode composition.
    encoder.set_training(true);

    // `PeftStep1` with no explicit `--lora-weights-in`: compute the
    // POST-one-optimizer-step weights ourselves (`peft_step1_weights`'s own
    // doc), then load them exactly as if they had been an
    // externally-supplied `--lora-weights-in` file — `_peft_step1_scratch`
    // keeps the backing `TempDir` alive through the `varmap.load` call
    // below (a `TempDir` deletes its contents on drop).
    let _peft_step1_scratch;
    let effective_lora_weights_in: Option<PathBuf> =
        if params.lora_init == GradOracleLoraInit::PeftStep1 && params.lora_weights_in.is_none() {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("peft_step1_weights.safetensors");
            peft_step1_weights(params, &path)?;
            _peft_step1_scratch = Some(dir);
            Some(path)
        } else {
            _peft_step1_scratch = None;
            params.lora_weights_in.clone()
        };

    if let Some(path) = &effective_lora_weights_in {
        varmap
            .load(path)
            .map_err(|e| -> Box<dyn std::error::Error> {
                format!(
                    "loading --lora-weights-in {path:?} into the VarMap failed: {e} — the file's \
                 tensor names must match jammi's OWN internal VarBuilder-path naming for this \
                 exact config (model dir / target-modules / lora-rank), see grad_oracle.rs's \
                 module doc's 'Weight interchange format' section"
                )
                .into()
            })?;
    }

    let trainable: Vec<Var> = {
        let data = varmap.data().lock().map_err(|_| "VarMap mutex poisoned")?;
        let mut named: Vec<(String, Var)> =
            data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named.into_iter().map(|(_, v)| v).collect()
    };
    if trainable.is_empty() {
        return Err("no trainable LoRA tensors — target_modules matched nothing".into());
    }
    // Names, same sorted order as `trainable`, kept alongside for the
    // gradient-dump loop below (a `Var` alone does not carry its own name).
    let trainable_names: Vec<String> = {
        let data = varmap.data().lock().map_err(|_| "VarMap mutex poisoned")?;
        let mut names: Vec<String> = data.keys().cloned().collect();
        names.sort();
        names
    };

    if let Some(path) = &params.lora_weights_out {
        varmap.save(path)?;
    }

    let mask = Tensor::ones((params.batch, params.seq), DType::U32, &device)?;
    let blocks: Vec<Tensor> = (0..3)
        .map(|i| {
            synthetic_ids(
                params.batch,
                params.seq,
                config.vocab_size,
                params.seed + i,
                &device,
            )
        })
        .collect();
    // See `GradOracleReport::batch_token_id_sums`'s own doc: a cheap digest
    // of which tokens each of the three groups actually got, computed
    // BEFORE the batched/non-batched split below (both arms consume the
    // SAME `blocks`), so a test can independently recompute
    // `synthetic_ids(.., seed + i, ..)` and compare sums without this
    // report otherwise revealing token content.
    let mut batch_token_id_sums = [0u64; 3];
    for (i, block) in blocks.iter().enumerate() {
        let ids = block.flatten_all()?.to_vec1::<u32>()?;
        batch_token_id_sums[i] = ids.iter().map(|&x| x as u64).sum();
    }

    // Dispatch-counter "before" snapshots, taken immediately around this
    // call's ONE forward+backward — same mechanism `finetune_step.rs` uses
    // (see this module's doc's determinant table), so a jammi-side dump
    // also records WHICH kernel composition actually ran, isolated from
    // anything an earlier tier in the same process invocation did.
    let ln_dispatch_before = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_before = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_before = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_before = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_before = jammi_lora::lora_epilogue_dispatch_snapshot();
    let lora_linear_fused_dispatch_before = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_before = jammi_encoders::attention_block_dispatch_snapshot();
    // Every op-keyed counter currently registered (see
    // `GradOracleReport::live_admit_keys`'s own doc) — a snapshot delta
    // taken around the SAME forward+backward the 14 explicit fields above
    // bracket, so `--ablate-each-op` can discover the real
    // `JAMMI_KERNELS_DISABLE` key set this exact run reached, instead of a
    // hand-maintained literal list.
    let admit_snapshot_before = jammi_kernels::admission::snapshot_all();

    let (a, p, n) = if params.batched_forward {
        let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0)?;
        let joined_mask = Tensor::cat(&[&mask, &mask, &mask], 0)?;
        let all = encoder.forward(&joined, &joined_mask)?;
        let b = params.batch;
        (
            all.narrow(0, 0, b)?,
            all.narrow(0, b, b)?,
            all.narrow(0, 2 * b, b)?,
        )
    } else {
        (
            encoder.forward(&blocks[0], &mask)?,
            encoder.forward(&blocks[1], &mask)?,
            encoder.forward(&blocks[2], &mask)?,
        )
    };
    let loss = triplet_loss(&a, &p, &n, 0.3)?;
    let grads = loss.backward()?;
    let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

    let ln_dispatch_after = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_after = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_after = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_after = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_after = jammi_lora::lora_epilogue_dispatch_snapshot();
    let lora_linear_fused_dispatch_after = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_after = jammi_encoders::attention_block_dispatch_snapshot();
    let admit_snapshot_after = jammi_kernels::admission::snapshot_all();
    let admit_key_dispatches: std::collections::BTreeMap<String, DispatchPair> =
        admit_snapshot_after
            .iter()
            .map(|(op, after)| {
                let before = admit_snapshot_before.get(*op).copied().unwrap_or_default();
                (
                    (*op).to_string(),
                    DispatchPair {
                        fused: after.fused.saturating_sub(before.fused),
                        eager: after.eager.saturating_sub(before.eager),
                    },
                )
            })
            .collect();
    let mut live_admit_keys: Vec<String> = admit_key_dispatches
        .iter()
        .filter(|(_, pair)| pair.fused > 0)
        .map(|(op, _)| op.clone())
        .collect();
    live_admit_keys.sort();

    // The RESOLVED `JAMMI_KERNELS_DISABLE` state (contract K-aux, now on
    // `main`) — see `GradOracleReport::kernels_disabled_requested`'s own
    // doc for why this tier records it unconditionally but does not gate
    // on `unmatched_disables()` the way `finetune_step.rs`'s `run()` does.
    let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
    let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();

    let mut gradients = std::collections::BTreeMap::new();
    for (name, var) in trainable_names.into_iter().zip(trainable.into_iter()) {
        let tensor = var.as_tensor();
        let shape = tensor.dims().to_vec();
        let weight = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let grad_tensor = grads.get(tensor).ok_or_else(|| {
            format!("no gradient recorded for trainable tensor {name:?} — did backward() reach it?")
        })?;
        let grad = grad_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        gradients.insert(
            name,
            GradOracleTensor {
                shape,
                grad,
                weight,
            },
        );
    }

    let vacuous_tensor_names: Vec<String> = gradients
        .iter()
        .filter(|(_, t)| grad_is_vacuous(&t.grad))
        .map(|(name, _)| name.clone())
        .collect();
    let vacuous_tensor_count = vacuous_tensor_names.len();

    Ok(GradOracleReport {
        tool: "jammi_grad_oracle".to_string(),
        model_dir: params.model_dir.display().to_string(),
        device: device_label,
        device_name: device_name(params.cuda_device),
        git_rev: tip_sha(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        target_modules: params.target_modules.clone(),
        batched_forward: params.batched_forward,
        seed: params.seed,
        lora_dropout: 0.0,
        lora_init: params.lora_init.as_flag().to_string(),
        peft_step1_applied: params.lora_init == GradOracleLoraInit::PeftStep1
            && params.lora_weights_in.is_none(),
        lora_weights_in: path_display(&params.lora_weights_in),
        lora_weights_out: path_display(&params.lora_weights_out),
        trainable_tensor_count: gradients.len(),
        batch_token_id_sums,
        loss: loss_val,
        ln_fused_dispatches: ln_dispatch_after
            .fused
            .saturating_sub(ln_dispatch_before.fused),
        ln_eager_dispatches: ln_dispatch_after
            .eager
            .saturating_sub(ln_dispatch_before.eager),
        rope_fused_dispatches: rope_dispatch_after
            .fused
            .saturating_sub(rope_dispatch_before.fused),
        rope_eager_dispatches: rope_dispatch_after
            .eager
            .saturating_sub(rope_dispatch_before.eager),
        softmax_fused_dispatches: softmax_dispatch_after
            .fused
            .saturating_sub(softmax_dispatch_before.fused),
        softmax_eager_dispatches: softmax_dispatch_after
            .eager
            .saturating_sub(softmax_dispatch_before.eager),
        geglu_fused_dispatches: geglu_dispatch_after
            .fused
            .saturating_sub(geglu_dispatch_before.fused),
        geglu_eager_dispatches: geglu_dispatch_after
            .eager
            .saturating_sub(geglu_dispatch_before.eager),
        lora_epilogue_fused_dispatches: lora_epilogue_dispatch_after
            .fused
            .saturating_sub(lora_epilogue_dispatch_before.fused),
        lora_epilogue_eager_dispatches: lora_epilogue_dispatch_after
            .eager
            .saturating_sub(lora_epilogue_dispatch_before.eager),
        lora_linear_fused_dispatches: lora_linear_fused_dispatch_after
            .fused
            .saturating_sub(lora_linear_fused_dispatch_before.fused),
        lora_linear_eager_dispatches: lora_linear_fused_dispatch_after
            .eager
            .saturating_sub(lora_linear_fused_dispatch_before.eager),
        attention_block_fused_dispatches: attention_block_dispatch_after
            .fused
            .saturating_sub(attention_block_dispatch_before.fused),
        attention_block_eager_dispatches: attention_block_dispatch_after
            .eager
            .saturating_sub(attention_block_dispatch_before.eager),
        kernels_disabled_requested,
        kernels_disabled_fired,
        live_admit_keys,
        admit_key_dispatches,
        vacuous_tensor_count,
        vacuous_tensor_names,
        gradients,
    })
}

fn path_display(p: &Option<PathBuf>) -> Option<String> {
    p.as_ref().map(|p| p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn tiny_model_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_modernbert_classifier")
    }

    fn tiny_params() -> GradOracleParams {
        GradOracleParams {
            model_dir: tiny_model_dir(),
            // 3, deliberately NOT 2: `run()`'s batched arm computes the
            // negative group's row offset as `2 * b`. At `b == 2`,
            // `2 * b == 2 + b == 4` — a MUTATION of that `*` to `+` is
            // undetectable by ANY test using `batch: 2` (cargo-mutants
            // caught exactly this: `replace * with + in run` survived
            // until this fixture moved off `b == 2`). `b == 3` makes
            // `2 * b = 6` and `2 + b = 5` diverge.
            batch: 3,
            seq: 8,
            lora_rank: 2,
            lora_alpha: 4.0,
            target_modules: vec!["Wqkv".to_string()],
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            cuda_device: None,
            seed: 7,
            batched_forward: true,
            lora_weights_in: None,
            lora_weights_out: None,
            // Existing tests below assert exact ZerosB semantics (`dL/dA
            // IDENTICALLY zero`, etc.) — keep this fixture's DEFAULT on the
            // legacy mode so every pre-existing test's behaviour is
            // unchanged by this field's addition; the new
            // `LoraInitMode::Gaussian`-specific tests below override it
            // explicitly.
            lora_init: GradOracleLoraInit::ZerosB,
        }
    }

    /// Drives the REAL entry point, `run()`. A fresh (no `lora_weights_in`)
    /// call must produce a finite, non-degenerate loss and at least one
    /// nonzero gradient entry — the cheap sanity floor every measurement
    /// tier in this crate applies before trusting a more specific claim.
    #[test]
    fn grad_oracle_run_produces_finite_loss_and_nonzero_gradients() {
        let report = run(&tiny_params()).expect("grad-oracle run");
        assert!(report.loss.is_finite());
        assert!(!report.gradients.is_empty());
        for (name, t) in &report.gradients {
            assert_eq!(
                t.grad.len(),
                t.weight.len(),
                "{name}: grad/weight length mismatch"
            );
            assert_eq!(
                t.shape.iter().product::<usize>(),
                t.grad.len(),
                "{name}: shape does not match flattened grad length"
            );
            assert!(
                t.grad.iter().all(|g| g.is_finite()),
                "{name}: non-finite gradient entry"
            );
        }
        assert!(
            report
                .gradients
                .values()
                .any(|t| t.grad.iter().any(|&g| g != 0.0)),
            "every gradient entry is exactly zero — looks like backward() never reached the \
             trainable tensors, or the dump read the wrong store"
        );
    }

    /// THE SELF-CONSISTENCY ROUND TRIP this module's doc promises: a
    /// SECOND, INDEPENDENT `run()` call that (a) loads the FIRST call's
    /// `lora_weights_out` dump via `lora_weights_in`, on the SAME
    /// synthetic batch (same `seed`), must reproduce the FIRST call's loss
    /// and every gradient entry BIT-FOR-BIT (CPU/F32 is deterministic —
    /// same weights in, same tokens in, same arithmetic). This is the
    /// mechanism proof the cross-framework comparison depends on: if the
    /// weight-interchange file did not carry the values that actually ran,
    /// this round trip would diverge even before torch enters the
    /// picture. Drives `run()` twice — the real entry point — never
    /// `VarMap::load`/`save` tested in isolation.
    #[test]
    fn grad_oracle_self_consistency_round_trip() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut first_params = tiny_params();
        first_params.lora_weights_out = Some(weights_path.clone());
        let first = run(&first_params).expect("first grad-oracle run");

        let mut second_params = tiny_params();
        second_params.lora_weights_in = Some(weights_path.clone());
        let second = run(&second_params).expect("second grad-oracle run");

        // MUTATION-TRIAGE (cargo-mutants caught `path_display` surviving
        // as `None`/`Some(String::new())`/`Some("xyzzy".into())`): pin the
        // reported provenance strings against the ACTUAL paths passed in,
        // not just "some Option came back".
        let weights_path_str = weights_path.display().to_string();
        assert_eq!(first.lora_weights_in, None);
        assert_eq!(first.lora_weights_out, Some(weights_path_str.clone()));
        assert_eq!(second.lora_weights_in, Some(weights_path_str));
        assert_eq!(second.lora_weights_out, None);

        assert_eq!(first.loss, second.loss, "loss must round-trip bit-for-bit");
        assert_eq!(
            first.gradients.keys().collect::<Vec<_>>(),
            second.gradients.keys().collect::<Vec<_>>(),
            "tensor name sets must match"
        );
        for (name, t1) in &first.gradients {
            let t2 = &second.gradients[name];
            assert_eq!(
                t1.weight, t2.weight,
                "{name}: weight did not round-trip through the file"
            );
            assert_eq!(
                t1.grad, t2.grad,
                "{name}: gradient diverged despite identical weights+batch"
            );
        }

        let _ = std::fs::remove_file(&weights_path);
    }

    /// NEGATIVE CONTROL (family F: non-vacuous): loading a DIFFERENT LoRA
    /// weights file must actually change the dumped weight values (proving
    /// `--lora-weights-in` is not silently ignored) AND must actually
    /// change the gradient of the `lora_b` tensor too (proving the
    /// forward+backward actually ran against the loaded values, not the
    /// pre-load seeded draw). Two DIFFERENT seeds give two different fresh
    /// `A` draws (`B` starts at zero either way under
    /// `LoraInitMode::ZerosB`).
    ///
    /// This deliberately does NOT compare the `lora_a` tensor's gradient:
    /// under `LoraInitMode::ZerosB`, `dL/dA` is IDENTICALLY zero at a fresh
    /// (pre-optimizer-step) init regardless of `A`'s own value — the LoRA
    /// forward's `B @ (A @ x)` has `B == 0`, so the chain rule's `B^T @
    /// dL/d(output)` factor that backprops into `dL/dA` is the zero
    /// matrix. Asserting `lora_a`'s gradient differs would therefore be
    /// VACUOUS (0.0 != 0.0 never holds; the assertion would trivially pass
    /// for the wrong reason, or trivially fail always, neither of which
    /// tests the load path). `dL/dB`, in contrast, IS `A`-dependent even
    /// though `B == 0` — `dL/dB` is proportional to `(A @ x)`, which
    /// changes with `A` — so it is both meaningful (catches a silently
    /// skipped/wrong-path `varmap.load`) and cheap (no optimizer step
    /// needed to make it informative, unlike an earlier draft of this
    /// test's docstring claimed).
    #[test]
    fn grad_oracle_lora_weights_in_actually_overrides_the_fresh_init() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut seed_params = tiny_params();
        seed_params.seed = 123;
        seed_params.lora_weights_out = Some(weights_path.clone());
        let seeded = run(&seed_params).expect("seed run");

        let mut baseline_params = tiny_params();
        baseline_params.seed = 999; // a different fresh draw, no file loaded
        let baseline = run(&baseline_params).expect("baseline run (different fresh seed)");

        let mut loaded_params = tiny_params();
        loaded_params.seed = 999; // SAME batch/seed as baseline...
        loaded_params.lora_weights_in = Some(weights_path.clone()); // ...but weights overridden
        let loaded = run(&loaded_params).expect("loaded run");

        let any_name = baseline
            .gradients
            .keys()
            .next()
            .expect("at least one tensor")
            .clone();
        assert_ne!(
            baseline.gradients[&any_name].weight, loaded.gradients[&any_name].weight,
            "lora_weights_in did not change the weight actually used -- looks like the load call \
             is being silently skipped or its error swallowed"
        );
        assert_eq!(
            loaded.gradients[&any_name].weight, seeded.gradients[&any_name].weight,
            "the loaded weight does not match the file's own recorded value"
        );

        // `any_name` is a `lora_a`-suffixed key (`.lora_a` < `.lora_b`
        // lexically, so `BTreeMap`'s first key is always `lora_a` here —
        // see this test's doc for why `lora_b`'s gradient, not `lora_a`'s,
        // is the informative one to assert on).
        assert!(
            any_name.ends_with("lora_a"),
            "fixture assumption broken: expected the sorted-first gradient key to be a \
             lora_a-suffixed name, got {any_name:?} -- update the lora_b lookup below to match"
        );
        // NOTE: `loaded` and `seeded` do NOT share a batch (`loaded_params`
        // keeps `seed = 999`, only its WEIGHTS come from the `seed = 123`
        // file), so their `lora_b` gradients are not expected to match —
        // only `baseline` (also `seed = 999`, no file loaded) is the right
        // same-batch comparator for the load-actually-took-effect check.
        let lora_b_name = format!("{}lora_b", any_name.strip_suffix("lora_a").unwrap());
        assert_ne!(
            baseline.gradients[&lora_b_name].grad, loaded.gradients[&lora_b_name].grad,
            "lora_weights_in changed the weight (asserted above) but NOT the lora_b gradient -- \
             looks like the forward+backward ran against the PRE-load seeded draw instead of the \
             loaded values (dL/dB is A-dependent even under LoraInitMode::ZerosB, see this test's \
             doc)"
        );

        let _ = std::fs::remove_file(&weights_path);
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `run()`'s `params.seed +
    /// i` block-offset arithmetic surviving as `seed - i`/`seed * i`): the
    /// three earlier tests above only assert "finite", "nonzero", and
    /// "round-trips against ITSELF" -- none of them pin `seed + i`
    /// SPECIFICALLY, since a self-consistent-but-wrong formula still
    /// passes all of them. This test recomputes `synthetic_ids(.., seed +
    /// i, ..)` INDEPENDENTLY (never by calling `run()` a second time) for
    /// `i` in `0..3` and compares against `batch_token_id_sums` --
    /// `GradOracleReport`'s one field whose whole purpose is making this
    /// arithmetic externally checkable.
    #[test]
    fn grad_oracle_batch_group_offsets_match_synthetic_ids_seed_plus_i() {
        let params = tiny_params();
        let report = run(&params).expect("grad-oracle run");

        let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))
            .expect("read config.json");
        let config: jammi_encoders::ModernBertConfig =
            serde_json::from_str(&config_raw).expect("parse config.json");
        let device = Device::Cpu;

        for i in 0u64..3 {
            let ids = synthetic_ids(
                params.batch,
                params.seq,
                config.vocab_size,
                params.seed + i,
                &device,
            );
            let expected_sum: u64 = ids
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()
                .iter()
                .map(|&x| x as u64)
                .sum();
            assert_eq!(
                report.batch_token_id_sums[i as usize], expected_sum,
                "group {i}'s token-id sum does not match synthetic_ids(.., seed + {i}, ..) -- \
                 run()'s block-construction offset arithmetic must be exactly `seed + i`, not \
                 `seed - i`/`seed * i`/anything else"
            );
        }
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught the batched arm's `all
    /// .narrow(0, 2 * b, b)` — the negative group's row offset — surviving
    /// as `2 + b`/`2 / b`): batched (one joined forward, split by
    /// `narrow`) and per-group (three separate forwards) MUST produce the
    /// identical loss/gradients for the SAME weights and the SAME
    /// synthetic batch — ModernBERT's per-row attention mask means no row
    /// can see any other row's tokens, so joining three groups into one
    /// forward is a pure reshape of the computation, not a different one.
    /// A miscomputed row offset in the batched arm picks the WRONG rows as
    /// one of anchor/positive/negative there, while the per-group arm
    /// (which uses no arithmetic at all — `blocks[0]`/`blocks[1]`/`blocks[2]`
    /// directly) stays correct, so the two arms diverge exactly when this
    /// arithmetic is wrong.
    #[test]
    fn grad_oracle_batched_and_unbatched_forward_agree() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut seed_params = tiny_params();
        seed_params.lora_weights_out = Some(weights_path.clone());
        let _seeded = run(&seed_params).expect("seed run");

        let mut batched_params = tiny_params();
        batched_params.lora_weights_in = Some(weights_path.clone());
        batched_params.batched_forward = true;
        let batched = run(&batched_params).expect("batched run");

        let mut unbatched_params = tiny_params();
        unbatched_params.lora_weights_in = Some(weights_path.clone());
        unbatched_params.batched_forward = false;
        let unbatched = run(&unbatched_params).expect("unbatched run");

        assert_eq!(
            batched.batch_token_id_sums, unbatched.batch_token_id_sums,
            "the SAME seed must produce the SAME three synthetic batches regardless of \
             batched_forward -- this rules out 'the batches themselves differed' as the \
             explanation for any loss/gradient difference below"
        );
        // NOT bit-exact: candle's batched (3b-row) matmul kernel is free to
        // reduce in a different order than three separate b-row matmuls
        // (mathematically equivalent, not bitwise so — f32 addition is not
        // associative). MEASURED on this tiny fixture: relative differences
        // around 1e-6, four to five orders of magnitude below what a
        // genuine group-selection bug produces (a wrong `narrow` offset
        // picks ENTIRELY DIFFERENT rows, not a slightly-differently-rounded
        // version of the right ones) -- `TOL_REL`/`TOL_ABS` are generous
        // relative to the measured noise floor while staying far tighter
        // than a real defect would clear.
        const TOL_REL: f32 = 1e-3;
        const TOL_ABS: f32 = 1e-6;
        assert!(
            (batched.loss - unbatched.loss).abs()
                <= TOL_ABS + TOL_REL * batched.loss.abs().max(unbatched.loss.abs()),
            "batched vs per-group forward loss differs beyond floating-point reduction-order \
             noise: {} vs {} -- a group-selection offset bug (e.g. narrow(0, 2*b, b) \
             miscomputed) would silently pick the WRONG rows in the batched arm only",
            batched.loss,
            unbatched.loss
        );
        assert_eq!(
            batched.gradients.keys().collect::<Vec<_>>(),
            unbatched.gradients.keys().collect::<Vec<_>>()
        );
        for (name, t1) in &batched.gradients {
            let t2 = &unbatched.gradients[name];
            assert_eq!(
                t1.weight, t2.weight,
                "{name}: weight differs between batched/unbatched runs (both loaded the SAME file)"
            );
            for (i, (&x, &y)) in t1.grad.iter().zip(t2.grad.iter()).enumerate() {
                let diff = (x - y).abs();
                let scale = x.abs().max(y.abs());
                assert!(
                    diff <= TOL_ABS + TOL_REL * scale,
                    "{name}[{i}]: batched={x} vs unbatched={y} (|diff|={diff}) exceeds the \
                     floating-point reduction-order noise tolerance (abs {TOL_ABS} + rel \
                     {TOL_REL}*{scale}) -- this looks like a real divergence (e.g. the WRONG \
                     rows selected), not rounding noise"
                );
            }
        }

        let _ = std::fs::remove_file(&weights_path);
    }

    /// Unit-level pin on [`grad_is_vacuous`]'s own floor, independent of any
    /// full `run()` call: exactly at [`VACUOUS_NORM_FLOOR`] classifies as
    /// vacuous (`<`, not `<=` — a vector whose norm lands EXACTLY on the
    /// floor is still "below or at" in this classifier's own intent, and a
    /// mutation flipping `<` to `<=` at this one boundary value would
    /// otherwise be undetectable), while a norm one order of magnitude
    /// above it does not.
    #[test]
    fn grad_is_vacuous_floor_boundary() {
        assert!(grad_is_vacuous(&[0.0, 0.0, 0.0]));
        // sqrt(3) * 1e-13 ~= 1.7e-13, well below VACUOUS_NORM_FLOOR (1e-12).
        assert!(grad_is_vacuous(&[1e-13, 1e-13, 1e-13]));
        // A single element at 1e-11 alone has norm 1e-11, one order of
        // magnitude ABOVE the floor -- must NOT classify as vacuous.
        assert!(!grad_is_vacuous(&[1e-11]));
        assert!(!grad_is_vacuous(&[1.0, 0.0, 0.0]));
    }

    /// Cross-checks [`GradOracleReport::vacuous_tensor_count`]/
    /// `vacuous_tensor_names` against `run()`'s own `gradients` map,
    /// recomputed INDEPENDENTLY here (never by trusting the report's own
    /// count) -- and pins the STRUCTURAL note this module's doc and
    /// `LoraInitMode::ZerosB`'s field doc both make: under the legacy
    /// `ZerosB` init, every vacuous tensor is a `lora_a`-suffixed one (`B ==
    /// 0` makes `dL/dA` identically zero for ANY `A`), never a `lora_b`.
    #[test]
    fn grad_oracle_zeros_b_init_leaves_lora_a_structurally_vacuous() {
        let report = run(&tiny_params()).expect("grad-oracle run (ZerosB, tiny_params default)");

        let expected_names: Vec<String> = report
            .gradients
            .iter()
            .filter(|(_, t)| grad_is_vacuous(&t.grad))
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(
            report.vacuous_tensor_names, expected_names,
            "report.vacuous_tensor_names must match an independent recomputation off \
             report.gradients"
        );
        assert_eq!(report.vacuous_tensor_count, expected_names.len());
        assert!(
            report.vacuous_tensor_count > 0,
            "ZerosB's own structural limitation (this module's doc) means at least the lora_a \
             tensors must be vacuous at a fresh init -- a count of 0 here would mean either the \
             fixture stopped using ZerosB or grad_is_vacuous stopped detecting the known-zero case"
        );
        for name in &report.vacuous_tensor_names {
            assert!(
                name.ends_with("lora_a"),
                "{name}: a vacuous tensor under ZerosB must be a lora_a tensor -- dL/dB is \
                 A-dependent even at B == 0 (see grad_oracle_lora_weights_in_actually_overrides_the_fresh_init's \
                 own doc), so a vacuous lora_b entry here would be a genuine backward-path defect"
            );
        }
    }

    /// THE oracle this feature (ledger row 240) exists to make possible:
    /// `LoraInitMode::Gaussian` starts BOTH `A` and `B` nonzero, so a
    /// SINGLE fresh forward+backward must make EVERY trainable tensor's
    /// gradient live -- `vacuous_tensor_count == 0`. This is the exact
    /// assertion `grad_oracle_ablation::run` gates its own refusal on;
    /// pinned here directly against `run()`, independent of that
    /// orchestrator, so a regression in `Gaussian`'s own liveness
    /// property is caught at this layer even if the ablation orchestrator
    /// were never invoked.
    #[test]
    fn grad_oracle_gaussian_init_makes_every_gradient_live() {
        let mut params = tiny_params();
        params.lora_init = GradOracleLoraInit::Gaussian;
        let report = run(&params).expect("grad-oracle run (Gaussian)");

        assert_eq!(
            report.vacuous_tensor_count, 0,
            "Gaussian init must make every trainable tensor's gradient live -- a nonzero count \
             here means dL/dA (or, less expectedly, dL/dB) collapsed to zero despite a nonzero B, \
             which should not happen for this fixture's shapes/seed"
        );
        assert!(report.vacuous_tensor_names.is_empty());
        // Independent recomputation, same discipline as the ZerosB test
        // above -- never trust the report's own count without rechecking
        // it against the raw gradients.
        for (name, t) in &report.gradients {
            assert!(
                !grad_is_vacuous(&t.grad),
                "{name}: report claims vacuous_tensor_count == 0 but this tensor's own gradient \
                 is (numerically) the zero vector"
            );
        }
    }

    /// The REALISTIC operating point (round-7 audit fix, PR #383):
    /// `GradOracleLoraInit::PeftStep1` must ALSO make every gradient live
    /// (the one real `AdamW` step moves `B` away from zero on this
    /// fixture) and must set `peft_step1_applied` on the report.
    #[test]
    fn grad_oracle_peft_step1_makes_every_gradient_live() {
        let mut params = tiny_params();
        params.lora_init = GradOracleLoraInit::PeftStep1;
        let report = run(&params).expect("grad-oracle run (PeftStep1)");

        assert!(report.peft_step1_applied);
        assert_eq!(report.lora_init, "peft-step1");
        assert_eq!(
            report.vacuous_tensor_count, 0,
            "PeftStep1's one real AdamW step must move B away from zero, making dL/dA live too"
        );
        for (name, t) in &report.gradients {
            assert!(
                !grad_is_vacuous(&t.grad),
                "{name}: report claims vacuous_tensor_count == 0 but this tensor's own gradient \
                 is (numerically) the zero vector"
            );
        }
    }

    /// `PeftStep1`'s own weight-file round trip: TWO independent `run()`
    /// calls at the SAME seed must produce BIT-IDENTICAL `peft_step1`
    /// weights (the step-1 AdamW update is deterministic given the seed),
    /// which in turn makes the SECOND step's loss/gradients bit-identical
    /// too (CPU/F32 is deterministic) -- proving the internally-synthesized
    /// weights file is not itself a source of run-to-run noise.
    #[test]
    fn grad_oracle_peft_step1_is_deterministic_given_the_same_seed() {
        let mut params = tiny_params();
        params.lora_init = GradOracleLoraInit::PeftStep1;
        let first = run(&params).expect("first PeftStep1 run");
        let second = run(&params).expect("second PeftStep1 run");
        assert_eq!(first.loss, second.loss);
        assert_eq!(
            first.gradients.keys().collect::<Vec<_>>(),
            second.gradients.keys().collect::<Vec<_>>()
        );
        for (name, t1) in &first.gradients {
            let t2 = &second.gradients[name];
            assert_eq!(
                t1.grad, t2.grad,
                "{name}: PeftStep1 gradients diverged across two runs at the SAME seed"
            );
        }
    }

    /// [`GradOracleReport::live_admit_keys`] is a SET rendered as a sorted
    /// `Vec` (family J: no duplicate op key, ascending order) -- the
    /// invariant `grad_oracle_ablation`'s per-op loop depends on (iterating
    /// it directly as the ablation key list, never re-sorting/re-deduping
    /// itself).
    #[test]
    fn grad_oracle_live_admit_keys_is_sorted_and_deduplicated() {
        let report = run(&tiny_params()).expect("grad-oracle run");
        let mut sorted_deduped = report.live_admit_keys.clone();
        sorted_deduped.sort();
        sorted_deduped.dedup();
        assert_eq!(
            report.live_admit_keys, sorted_deduped,
            "live_admit_keys must already be sorted with no duplicate entries"
        );
    }

    /// A process-unique temp directory (never `/tmp` directly, and never
    /// shared with a parallel test in this same binary) for the two tests
    /// above that write real files.
    fn tempdir() -> PathBuf {
        // `cargo test`'s default thread pool runs this crate's `#[test]`
        // fns in PARALLEL — `std::process::id()` is shared by every test in
        // this binary, and `SystemTime::now()`'s actual clock resolution on
        // some platforms is coarser than true nanoseconds, so two
        // concurrently-running tests calling this within the same tick CAN
        // collide on path alone (MEASURED: this collision produced a
        // transient failure in `grad_oracle_lora_weights_in_actually_overrides_the_fresh_init`
        // while `grad_oracle_batched_and_unbatched_forward_agree` was
        // concurrently writing/reading the SAME nominally-"unique" path).
        // A process-wide atomic counter is monotonically unique per call
        // regardless of clock resolution, closing that race.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "jammi-bench-grad-oracle-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
