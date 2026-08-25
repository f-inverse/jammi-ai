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
//! | `checkpoint_config_sha256` | identity | `sha256_and_len(&model_dir.join("config.json"))` — called in `run()` before the forward, via the SAME shared streaming implementation `finetune_step.rs` also uses: `pub(crate) fn sha256_and_len` (`finetune_step.rs:574`) | `checkpoint_identity_fields = checkpoint_identity(args.model_dir)` (`torch_grad_oracle.py:413`) — `checkpoint_identity` is a bare alias for the real, streaming implementation torch_finetune_step.py's own `checkpoint_identity` function provides (see the two field citations directly below) |
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
//! | `kernels_disabled_requested`/`kernels_disabled_fired` | NOT YET APPLICABLE — the underlying instrumentation (`feat/kernels-admission-disable`, "K-aux") has not landed on `main` as of this round (`grep -rn kernels_disabled crates/` finds zero hits at this branch's base) | n/a (add alongside the dispatch counters above once K-aux lands) | n/a |
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
use candle_nn::VarMap;
use serde::Serialize;

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
}

/// One trainable tensor's dumped gradient (and, for the non-vacuous check
/// described in this module's doc, the exact weight value the forward
/// actually used) — `f32` regardless of `backbone_dtype`: the D2H read
/// widens STORAGE, it does not add mantissa bits the compute dtype lacked
/// (same convention `finetune_step.rs`'s `losses` field doc states).
#[derive(Debug, Clone, Serialize)]
pub struct GradOracleTensor {
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
    pub weight: Vec<f32>,
}

/// The oracle's full dump. See this module's doc for the placement/format
/// contract every field below carries.
#[derive(Debug, Serialize)]
pub struct GradOracleReport {
    pub tool: &'static str,
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

/// Run the oracle and return its report. NO optimizer step — see this
/// module's doc for why a gradient-direction comparison does not need one.
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
    // Training mode: dropout is forced off above (`lora_dropout: None`), so
    // this only selects the training-arm KERNEL COMPOSITION (fused
    // whole-attention-block / fused LoRA site where eligible), not any
    // additional randomness — the point is comparing the SAME arithmetic
    // shape jammi actually trains with, not the (dropout-free either way)
    // eval-mode composition.
    encoder.set_training(true);

    if let Some(path) = &params.lora_weights_in {
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

    Ok(GradOracleReport {
        tool: "jammi_grad_oracle",
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
