//! The `--ablate-each-op` orchestrator: ONE build, N `jammi-bench grad-oracle`
//! child invocations (`--ablate-each-op` itself is never passed to a child —
//! see [`spawn_arm`]'s doc for why recursion cannot happen), each set at
//! IDENTICAL LoRA weights (the [`super::grad_oracle`] weight-interchange
//! mechanism, unchanged), differing only in `backbone_dtype` or which
//! `jammi_kernels::admission` op key `JAMMI_KERNELS_DISABLE` names for that
//! one child.
//!
//! ## Why this tool did not exist before (ledger row 240)
//!
//! Every earlier fused-vs-eager oracle in this crate compared arms of the
//! SAME backward composition against EACH OTHER (`finetune_step.rs`'s A/B),
//! or compared jammi's gradient against torch's at a `LoraInitMode::ZerosB`
//! init (`grad_oracle.rs`'s own tier) — the latter is STRUCTURALLY BLIND to
//! any defect reachable only through `dL/dA` or the LoRA epilogue's own
//! gradient term, because `B == 0` makes that whole path's gradient
//! identically zero regardless of whether the arithmetic feeding it is
//! correct (see `grad_oracle.rs`'s module doc). Round 6 of the dispatch that
//! commissioned this module found exactly that blind spot hiding a real
//! defect: at `LoraInitMode::Gaussian` (both `A` and `B` nonzero, so every
//! gradient is LIVE), the all-fused arm's cosine against an f32-precision
//! truth run measured `0.610` — BELOW torch's own bf16 ceiling (`0.796` on
//! the same architecture) and well below the all-eager arm's `0.810`. A
//! `ZerosB`-only oracle would have read this defect as a healthy `0.0`
//! (vacuous, not wrong) on every affected tensor.
//!
//! ## Design: which side computes the cosine
//!
//! This orchestrator computes each arm's cosine similarity against the
//! f32-truth arm IN RUST (never delegated to `ci/scripts/perf/
//! compare_grad_oracle.py`), and the committed report carries only the
//! computed scalar per arm/per tensor — NOT the raw per-element gradient
//! arrays (`--keep-arm-dumps`, below, is the escape hatch for a caller that
//! wants those). This is a deliberate departure from `compare_grad_oracle.py`'s
//! own "deliberately Python, not Rust" convention (that module's doc,
//! `identity_fields.py`'s reuse discipline): THAT comparator's whole job is
//! reconciling TWO INDEPENDENT PRODUCERS (jammi vs torch — different
//! languages, different numeric libraries, a genuine risk of a shared bug
//! between the producer and the comparator if both were the same codebase).
//! This orchestrator instead compares MULTIPLE ARMS OF THE SAME PRODUCER
//! (jammi vs jammi, differing only in dtype/kernel composition) — the same
//! "same-build forced-arm A/B" class `finetune_step.rs`'s own Rust-computed
//! `r(L)` growth ratios and esc-044's Rust-asserted GEMM-operand-form checks
//! already use directly (`docs/maintainer/cuda-kernel-guide.md` §3.2/§3.3),
//! never routed through the Python comparator either. Family F's numeric
//! guarantee ("measured-and-asserted, never transcribed... a numpy-first
//! oracle") is satisfied at the GATING layer instead:
//! `ci/scripts/perf/check_fused_op_gradient_parity.py` is the independent,
//! Python-side consumer that applies a THRESHOLD to these Rust-computed
//! numbers and is itself self-tested against a synthetic report (never
//! trusting a Rust number it cannot independently re-derive the PASS/FAIL
//! verdict from); `--keep-arm-dumps` (below) additionally lets a caller who
//! wants to re-derive the cosine itself with numpy do exactly that, off the
//! SAME raw per-tensor gradients this orchestrator's own arithmetic reads.
//!
//! ## Weights: identical across every arm
//!
//! The reference (`all_fused`) arm is run FIRST with `--lora-weights-out`
//! pointed at a scratch file; every other arm (`f32_truth`, each per-op
//! ablation, `all_off`) then `--lora-weights-in`s that SAME file. LoRA `A`/
//! `B` are always stored at `F32` regardless of `backbone_dtype`
//! (`jammi_lora::lora_linear`'s `lora_ab_dtype_f32` admission gate — the
//! LoRA arm never runs the fused kernel at any OTHER dtype), so the
//! interchange file is dtype-invariant: the SAME file loads correctly
//! whether the loading arm's backbone is `bf16` or `f32`.
//!
//! ## `JAMMI_KERNELS_STRICT` per arm
//!
//! `all_fused` and every per-op-disabled/`all_off` arm run with
//! `JAMMI_KERNELS_STRICT=1` — `jammi_kernels::admission`'s own module doc:
//! "disable wins over Strict... `<op>` is forced eager while every OTHER op
//! passing through this same function is still strictly proven fused". This
//! is what makes each ablation arm a genuine ONE-OP isolation rather than a
//! "some unrelated predicate silently fell back too" ambiguity. `f32_truth`
//! runs WITHOUT `JAMMI_KERNELS_STRICT` (`Fallback` mode): every fused op's
//! own domain predicate requires `bf16` (or a matched `f32`/`bf16` pairing —
//! see `jammi-lora/src/lora_linear.rs`'s `base_dtype_f32_or_bf16_matched`),
//! so an f32 backbone run legitimately falls back everywhere; that is not a
//! predicate DEFECT and must not error under `Strict`.
//!
//! ## Which op keys get ablated
//!
//! [`super::grad_oracle::GradOracleReport::live_admit_keys`] from the
//! `all_fused` arm's OWN dump — never a hand-maintained literal list (see
//! that field's own doc for why `jammi_kernels::admission`'s standalone-vs-
//! subsumed op-key lattice makes a fixed list actively wrong on some
//! checkpoints).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::grad_oracle::GradOracleReport;

/// This orchestrator's own CLI-facing parameters — a superset of
/// `grad_oracle::GradOracleParams` minus the two fields
/// (`lora_weights_in`/`lora_weights_out`) this orchestrator drives itself,
/// per arm. Plain backtick code spans, not intra-doc links: `grad_oracle`
/// is a private `mod` of this crate, so `GradOracleParams` is not
/// resolvable from this module's own doc scope (7fd457e's convention —
/// convert to a backtick span, never a doc-hidden bypass).
#[derive(Debug, Clone)]
pub struct AblationParams {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    /// The backbone precision every arm EXCEPT `f32_truth` runs at
    /// (`f32_truth` always hardcodes `F32` regardless of this value — see
    /// this module's doc).
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    pub cuda_device: Option<usize>,
    pub seed: u64,
    pub batched_forward: bool,
    /// Passed to the `all_fused` (seed) arm's `--lora-init`. Almost always
    /// `Gaussian` — see `grad_oracle::GradOracleParams::lora_init`'s doc for
    /// why `ZerosB` would make this whole tool structurally blind to a
    /// `dL/dA` defect; kept configurable (not hardcoded) only so a caller
    /// can deliberately reproduce that blind spot for a regression test.
    pub lora_init: jammi_lora::LoraInitMode,
    /// If set, every arm's RAW `GradOracleReport` dump (full per-element
    /// `f32` gradients — see this module's doc's "which side computes the
    /// cosine" section) is copied into this directory as
    /// `<arm-slug>.json`, alongside the compact, cosine-only report this
    /// orchestrator writes to `--out`. `None` keeps only the compact report
    /// (the default — the arm dumps are large and are not committed
    /// artifacts).
    pub keep_arm_dumps: Option<PathBuf>,
}

/// One arm's result: which kernel-disable request produced it, its own
/// dispatch/vacuous provenance (copied straight off that arm's
/// [`GradOracleReport`]), its overall cosine similarity against the
/// `f32_truth` arm, and the per-tensor breakdown.
#[derive(Debug, Serialize)]
pub struct AblationArm {
    /// `"all_fused"` | `"f32_truth"` | `"ablate:<op_key>"` | `"all_off"`.
    pub arm: String,
    /// `Some(<op_key>)` for a per-op ablation arm, `None` for the other
    /// three (`all_fused`, `f32_truth`, `all_off` disable either nothing or
    /// everything, never exactly one named key).
    pub op_key: Option<String>,
    pub backbone_dtype: String,
    pub kernels_disabled_requested: Vec<String>,
    pub kernels_disabled_fired: Vec<String>,
    /// `kernels_disabled_requested` entries absent from
    /// `kernels_disabled_fired` — computed the SAME way
    /// `jammi_kernels::admission::unmatched_disables()` would (set
    /// difference), but from this arm's OWN recorded pair rather than a
    /// live process-wide `OnceLock` (this arm ran in a separate child
    /// process). Empty iff this arm's provenance is self-describing;
    /// [`run`] refuses (returns `Err`, never writes `--out`) if ANY arm's
    /// list here is non-empty.
    pub unmatched_disables: Vec<String>,
    pub vacuous_tensor_count: usize,
    pub loss: f32,
    pub matched_tensor_count: usize,
    pub overall_cosine_vs_f32_truth: f64,
    pub per_tensor: Vec<AblationTensorRow>,
}

/// One matched tensor's cosine against the `f32_truth` arm, plus the
/// (matrix, module, layer) decomposition of its jammi-internal
/// `VarBuilder`-path name — grouping key for a caller building the
/// (lora_a|lora_b) × module × layer table this module's own doc/the
/// dispatch that commissioned it asks for.
#[derive(Debug, Serialize)]
pub struct AblationTensorRow {
    pub name: String,
    /// `"lora_a"` | `"lora_b"` | `"unknown"` (a name not shaped like
    /// `layer.<n>.<module>.<lora_a|lora_b>` — never panics on an
    /// unexpected name, see [`parse_tensor_name`]).
    pub matrix: String,
    pub module: String,
    /// `-1` for a name [`parse_tensor_name`] could not parse a layer index
    /// out of (never silently `0`, which would collide with a real layer
    /// 0 row).
    pub layer: i64,
    pub cosine_vs_f32_truth: f64,
    pub n: usize,
}

/// The full ablation report [`run`] writes to `--out`.
#[derive(Debug, Serialize)]
pub struct AblationReport {
    pub tool: &'static str,
    pub git_rev: Option<String>,
    pub model_dir: String,
    pub checkpoint_config_sha256: String,
    pub checkpoint_weights_sha256: String,
    pub checkpoint_weights_size_bytes: u64,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    pub batched_forward: bool,
    pub seed: u64,
    pub lora_init: String,
    /// Every op key ablated (`AblationArm::op_key` restricted to `Some`,
    /// same order) — copied from the `all_fused` arm's own
    /// `live_admit_keys`, never a literal list (see this module's doc).
    pub ablated_op_keys: Vec<String>,
    pub arms: Vec<AblationArm>,
}

/// jammi's internal `VarBuilder`-path tensor-name shape is
/// `layer.<n>.<module>.<lora_a|lora_b>` (`grad_oracle.rs`'s module doc,
/// "Weight interchange format" section, e.g. `layer.3.Wqkv.lora_a`) — parsed
/// here defensively (never panics; a name outside this exact 4-segment
/// shape reports `("unknown", name.to_string(), -1)` rather than aborting
/// the whole ablation run over one oddly-named tensor).
fn parse_tensor_name(name: &str) -> (String, String, i64) {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() == 4 && parts[0] == "layer" && (parts[3] == "lora_a" || parts[3] == "lora_b") {
        if let Ok(layer) = parts[1].parse::<i64>() {
            return (parts[3].to_string(), parts[2].to_string(), layer);
        }
    }
    ("unknown".to_string(), name.to_string(), -1)
}

const VACUOUS_NORM_FLOOR: f64 = 1e-12;

/// `f64` dot product / norm — deliberately independent of
/// `grad_oracle::grad_is_vacuous` (a different private fn in a sibling
/// module; not reused) so this orchestrator's own arithmetic does not share
/// a bug with that module's self-check, even though both apply the SAME
/// documented floor.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < VACUOUS_NORM_FLOOR {
        return 0.0;
    }
    dot / denom
}

fn has_nonfinite(v: &[f32]) -> bool {
    v.iter().any(|x| !x.is_finite())
}

fn precision_flag(p: jammi_numerics::ComputePrecision) -> &'static str {
    match p {
        jammi_numerics::ComputePrecision::F32 => "f32",
        jammi_numerics::ComputePrecision::F16 => "f16",
        jammi_numerics::ComputePrecision::BF16 => "bf16",
    }
}

fn lora_init_flag(m: jammi_lora::LoraInitMode) -> &'static str {
    match m {
        jammi_lora::LoraInitMode::Gaussian => "gaussian",
        jammi_lora::LoraInitMode::ZerosB => "zeros-b",
    }
}

/// One child invocation of `jammi-bench grad-oracle` (never `--ablate-each-op`
/// — that flag is simply never in this argument list, so a child cannot
/// recurse into a grandchild ablation run regardless of how this function is
/// called). Spawns `std::env::current_exe()` (ledger row 197: the
/// `JAMMI_KERNELS_DISABLE`/`JAMMI_KERNELS_STRICT` switches are process-wide
/// `OnceLock`s — the only way to change them within one `cargo run` is a
/// fresh child process, same mechanism `train_scale.rs::spawn_measure` and
/// `crates/jammi-bench/tests/finetune_step_kernel_disable.rs` already use).
fn spawn_arm(
    params: &AblationParams,
    backbone_dtype: jammi_numerics::ComputePrecision,
    lora_weights_in: Option<&Path>,
    lora_weights_out: Option<&Path>,
    out_path: &Path,
    strict: bool,
    disable: Option<&str>,
) -> Result<GradOracleReport, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.arg("grad-oracle")
        .arg("--model-dir")
        .arg(&params.model_dir)
        .arg("--batch")
        .arg(params.batch.to_string())
        .arg("--seq")
        .arg(params.seq.to_string())
        .arg("--lora-rank")
        .arg(params.lora_rank.to_string())
        .arg("--lora-alpha")
        .arg(params.lora_alpha.to_string())
        .arg("--target-modules")
        .arg(params.target_modules.join(","))
        .arg("--backbone-dtype")
        .arg(precision_flag(backbone_dtype))
        .arg("--seed")
        .arg(params.seed.to_string())
        .arg("--batched-forward")
        .arg(params.batched_forward.to_string())
        .arg("--lora-init")
        .arg(lora_init_flag(params.lora_init))
        .arg("--out")
        .arg(out_path);
    if let Some(ordinal) = params.cuda_device {
        cmd.arg("--cuda").arg(ordinal.to_string());
    }
    if let Some(p) = lora_weights_in {
        cmd.arg("--lora-weights-in").arg(p);
    }
    if let Some(p) = lora_weights_out {
        cmd.arg("--lora-weights-out").arg(p);
    }
    if strict {
        cmd.env("JAMMI_KERNELS_STRICT", "1");
    } else {
        cmd.env_remove("JAMMI_KERNELS_STRICT");
    }
    if let Some(key) = disable {
        cmd.env("JAMMI_KERNELS_DISABLE", key);
    } else {
        cmd.env_remove("JAMMI_KERNELS_DISABLE");
    }
    cmd.stderr(std::process::Stdio::inherit());
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "grad-oracle child arm failed (disable={disable:?} strict={strict}): exit {}",
            output.status
        )
        .into());
    }
    let text = std::fs::read_to_string(out_path)?;
    Ok(serde_json::from_str(&text)?)
}

/// `requested - fired`, sorted — the same set-difference
/// `jammi_kernels::admission::unmatched_disables()` computes, applied here
/// to one arm's OWN recorded `kernels_disabled_requested`/`kernels_disabled_fired`
/// pair (that arm ran in a separate child process; there is no live
/// process-wide state to read here).
fn unmatched(requested: &[String], fired: &[String]) -> Vec<String> {
    let fired_set: std::collections::HashSet<&String> = fired.iter().collect();
    let mut out: Vec<String> = requested
        .iter()
        .filter(|r| !fired_set.contains(r))
        .cloned()
        .collect();
    out.sort();
    out
}

/// Builds one [`AblationArm`] from a raw [`GradOracleReport`], comparing
/// every matched tensor (and the concatenated whole) against `truth`.
/// Returns `Err` if any matched tensor's gradient carries a non-finite
/// element on EITHER side (§3.7 of the cuda-kernel-guide: "write
/// comparisons affirmatively" — refuse before computing a cosine off NaN
/// data, never let `NaN`-poisoned arithmetic silently produce a number that
/// happens to read as low-but-plausible).
fn build_arm(
    label: &str,
    op_key: Option<String>,
    backbone_dtype: &str,
    report: &GradOracleReport,
    truth: &GradOracleReport,
) -> Result<AblationArm, Box<dyn std::error::Error>> {
    let mut per_tensor = Vec::with_capacity(report.gradients.len());
    let mut all_a: Vec<f32> = Vec::new();
    let mut all_b: Vec<f32> = Vec::new();
    let mut names: Vec<&String> = report.gradients.keys().collect();
    names.sort();
    for name in names {
        let a = &report.gradients[name];
        let b = truth.gradients.get(name).ok_or_else(|| {
            format!("arm {label:?}: tensor {name:?} is absent from the f32_truth dump")
        })?;
        if has_nonfinite(&a.grad) || has_nonfinite(&b.grad) {
            return Err(format!(
                "arm {label:?}, tensor {name:?}: non-finite gradient element on at least one \
                 side -- refusing to compute a cosine off NaN/Inf data"
            )
            .into());
        }
        let (matrix, module, layer) = parse_tensor_name(name);
        per_tensor.push(AblationTensorRow {
            name: name.clone(),
            matrix,
            module,
            layer,
            cosine_vs_f32_truth: cosine(&a.grad, &b.grad),
            n: a.grad.len(),
        });
        all_a.extend_from_slice(&a.grad);
        all_b.extend_from_slice(&b.grad);
    }
    let overall = cosine(&all_a, &all_b);
    Ok(AblationArm {
        arm: label.to_string(),
        op_key,
        backbone_dtype: backbone_dtype.to_string(),
        kernels_disabled_requested: report.kernels_disabled_requested.clone(),
        kernels_disabled_fired: report.kernels_disabled_fired.clone(),
        unmatched_disables: unmatched(
            &report.kernels_disabled_requested,
            &report.kernels_disabled_fired,
        ),
        vacuous_tensor_count: report.vacuous_tensor_count,
        loss: report.loss,
        matched_tensor_count: per_tensor.len(),
        overall_cosine_vs_f32_truth: overall,
        per_tensor,
    })
}

/// Runs the full ablation: `all_fused` (seed) → `f32_truth` →
/// `ablate:<key>` for each of `all_fused`'s own `live_admit_keys` →
/// `all_off`, then writes the compact [`AblationReport`] to `params_out`.
/// Refuses (returns `Err`, never writes `params_out`) if the `all_fused`
/// arm's `vacuous_tensor_count != 0` (this tool's whole premise — see this
/// module's doc) or if ANY arm's `unmatched_disables` is non-empty (a
/// provenance that is not self-describing certifies nothing).
pub fn run(
    params: &AblationParams,
    out: &Path,
) -> Result<AblationReport, Box<dyn std::error::Error>> {
    let scratch = tempfile::tempdir()?;
    let weights_path = scratch.path().join("lora_weights.safetensors");
    let dump_dir = params
        .keep_arm_dumps
        .clone()
        .unwrap_or_else(|| scratch.path().to_path_buf());
    std::fs::create_dir_all(&dump_dir)?;

    let ref_dump_path = dump_dir.join("all_fused.json");
    let reference = spawn_arm(
        params,
        params.backbone_dtype,
        None,
        Some(&weights_path),
        &ref_dump_path,
        true,
        None,
    )?;
    if reference.vacuous_tensor_count != 0 {
        return Err(format!(
            "all_fused arm has vacuous_tensor_count = {} (names: {:?}) -- this tool's whole \
             premise is that --lora-init makes EVERY gradient live; refusing to emit an \
             ablation comparison off a fixture that is not actually non-vacuous",
            reference.vacuous_tensor_count, reference.vacuous_tensor_names
        )
        .into());
    }

    let truth_dump_path = dump_dir.join("f32_truth.json");
    let truth = spawn_arm(
        params,
        jammi_numerics::ComputePrecision::F32,
        Some(&weights_path),
        None,
        &truth_dump_path,
        false,
        None,
    )?;

    let mut arms = Vec::new();
    arms.push(build_arm(
        "all_fused",
        None,
        precision_flag(params.backbone_dtype),
        &reference,
        &truth,
    )?);
    arms.push(build_arm("f32_truth", None, "f32", &truth, &truth)?);

    let ablated_op_keys = reference.live_admit_keys.clone();
    for key in &ablated_op_keys {
        let dump_path = dump_dir.join(format!("ablate_{key}.json"));
        let arm_report = spawn_arm(
            params,
            params.backbone_dtype,
            Some(&weights_path),
            None,
            &dump_path,
            true,
            Some(key),
        )?;
        arms.push(build_arm(
            &format!("ablate:{key}"),
            Some(key.clone()),
            precision_flag(params.backbone_dtype),
            &arm_report,
            &truth,
        )?);
    }

    let off_dump_path = dump_dir.join("all_off.json");
    let all_off = spawn_arm(
        params,
        params.backbone_dtype,
        Some(&weights_path),
        None,
        &off_dump_path,
        true,
        Some("all"),
    )?;
    arms.push(build_arm(
        "all_off",
        None,
        precision_flag(params.backbone_dtype),
        &all_off,
        &truth,
    )?);

    for arm in &arms {
        if !arm.unmatched_disables.is_empty() {
            return Err(format!(
                "arm {:?} has unmatched_disables = {:?} -- this arm's JAMMI_KERNELS_DISABLE \
                 request did not fully fire; its provenance is not self-describing, refusing to \
                 emit the ablation report",
                arm.arm, arm.unmatched_disables
            )
            .into());
        }
        if arm.vacuous_tensor_count != 0 {
            return Err(format!(
                "arm {:?} has vacuous_tensor_count = {} -- expected every gradient to stay live \
                 across every kernel-composition arm at a Gaussian init",
                arm.arm, arm.vacuous_tensor_count
            )
            .into());
        }
    }

    let (checkpoint_config_sha256, checkpoint_weights_sha256, checkpoint_weights_size_bytes) = (
        reference.checkpoint_config_sha256.clone(),
        reference.checkpoint_weights_sha256.clone(),
        reference.checkpoint_weights_size_bytes,
    );

    let report = AblationReport {
        tool: "jammi_grad_oracle_ablation",
        git_rev: reference.git_rev.clone(),
        model_dir: params.model_dir.display().to_string(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        target_modules: params.target_modules.clone(),
        batched_forward: params.batched_forward,
        seed: params.seed,
        lora_init: lora_init_flag(params.lora_init).to_string(),
        ablated_op_keys,
        arms,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(out, json)?;
    Ok(report)
}

/// Never used directly by [`run`] (kept for a future summary consumer /
/// tests) — a convenience alias so a caller can look up one arm by its
/// `arm` label without re-implementing the linear scan.
#[allow(dead_code)]
pub fn find_arm<'a>(report: &'a AblationReport, label: &str) -> Option<&'a AblationArm> {
    report.arms.iter().find(|a| a.arm == label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tensor_name_matches_the_documented_shape() {
        assert_eq!(
            parse_tensor_name("layer.3.Wqkv.lora_a"),
            ("lora_a".to_string(), "Wqkv".to_string(), 3)
        );
        assert_eq!(
            parse_tensor_name("layer.0.Wo.lora_b"),
            ("lora_b".to_string(), "Wo".to_string(), 0)
        );
    }

    /// Defensive-parse cases: none of these panic, and none is silently
    /// misread as a real `(matrix, module, layer)` triple.
    #[test]
    fn parse_tensor_name_refuses_unshaped_names_without_panicking() {
        assert_eq!(
            parse_tensor_name("not_layer_shaped"),
            ("unknown".to_string(), "not_layer_shaped".to_string(), -1)
        );
        assert_eq!(
            parse_tensor_name("layer.notanumber.Wqkv.lora_a"),
            (
                "unknown".to_string(),
                "layer.notanumber.Wqkv.lora_a".to_string(),
                -1
            )
        );
        assert_eq!(
            parse_tensor_name("layer.3.Wqkv.lora_c"),
            ("unknown".to_string(), "layer.3.Wqkv.lora_c".to_string(), -1)
        );
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = vec![1.0f32, 2.0, -3.0, 4.5];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_of_opposite_vectors_is_minus_one() {
        let a = vec![1.0f32, 2.0, -3.0];
        let b = vec![-1.0f32, -2.0, 3.0];
        assert!((cosine(&a, &b) - (-1.0)).abs() < 1e-9);
    }

    /// Non-vacuous control (family F): a zero vector on EITHER side must
    /// return a well-defined `0.0`, never `NaN`/panic (division by a
    /// near-zero denominator).
    #[test]
    fn cosine_of_a_zero_vector_is_zero_not_nan() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let nonzero = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine(&zero, &nonzero), 0.0);
        assert_eq!(cosine(&zero, &zero), 0.0);
    }

    #[test]
    fn unmatched_is_the_set_difference_sorted() {
        let requested = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        let fired = vec!["a".to_string()];
        assert_eq!(
            unmatched(&requested, &fired),
            vec!["b".to_string(), "c".to_string()]
        );
        assert!(unmatched(&requested, &requested).is_empty());
    }

    #[test]
    fn has_nonfinite_detects_nan_and_inf_affirmatively() {
        assert!(has_nonfinite(&[1.0, f32::NAN, 2.0]));
        assert!(has_nonfinite(&[f32::INFINITY]));
        assert!(has_nonfinite(&[f32::NEG_INFINITY]));
        assert!(!has_nonfinite(&[1.0, -2.0, 0.0]));
        assert!(!has_nonfinite(&[]));
    }
}
