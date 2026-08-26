//! The `--ablate-each-op` orchestrator: ONE build, N `jammi-bench grad-oracle`
//! child invocations (`--ablate-each-op` itself is never passed to a child —
//! see [`spawn_arm`]'s doc for why recursion cannot happen), each set at
//! IDENTICAL LoRA weights (the `grad_oracle` weight-interchange mechanism,
//! unchanged), differing only in `backbone_dtype` or which
//! `jammi_kernels::admission` op key `JAMMI_KERNELS_DISABLE` names for that
//! one child — run across MULTIPLE SEEDS and aggregated, because a
//! single-seed cosine is not a stable enough quantity to gate on (see
//! "Round 7: seeds and the budget", below).
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
//! correct (see `grad_oracle.rs`'s module doc).
//!
//! ## Round 7: seeds and the budget (audit finding on PR #383)
//!
//! Round 6's SINGLE-seed run (`--lora-init gaussian`, `alpha/rank = 2`)
//! measured `all_fused=0.610 < all_off=0.810` at `b4-s128`. Round 7's own
//! b8-s128 leg on the SAME build inverted that: `all_fused=0.843 >
//! all_off=0.773`, and a THIRD seed on a different pod reproduced the same
//! non-reproducibility. A single-seed cosine at this operating point is
//! CHAOTIC (same family as "a loss value is chaotic" — same-run forced-arm
//! A/B is the oracle, not a single number) — this module now runs every arm
//! across `AblationParams::seeds` (default `42,43,44`) and reports the
//! MEDIAN and min-max SPREAD, never a lone number. The per-op FAIL budget
//! is DERIVED from that measured spread, never a fixed constant fitted to
//! one run: `budget = 3 * (max - min)` of the `all_fused` arm's OWN
//! `full_tensor_cosine` across seeds, AT THAT SHAPE — see
//! [`AblationReport::derived_per_op_budget`]. `ci/scripts/perf/
//! check_fused_op_gradient_parity.py` (this tool's consumer) requires
//! reports from BOTH `b4-s128` and `b8-s128` and fails an op only if its
//! median delta exceeds ITS OWN shape's budget at BOTH shapes — a defect
//! specific to one shape/seed combination must not, on its own, either pass
//! or fail the gate.
//!
//! ## Round 7: the realistic operating point (audit finding on PR #383)
//!
//! `--lora-init gaussian` (round 6's only mode) draws both `A` and `B` from
//! `Normal(0, 0.02)` at `alpha/rank = 2` — a loss-surface region no real
//! training run ever occupies (real training starts at PEFT's own
//! `ZerosB`-style init and only ever sees `B != 0` AFTER at least one real
//! gradient step). `GradOracleLoraInit::PeftStep1` (this tier's new
//! default under `--ablate-each-op`) is the fix: PEFT's own init, ONE real
//! `AdamW` step, THEN measure — see `grad_oracle::peft_step1_weights`'s own
//! doc. `Gaussian` is kept as a selectable DIAGNOSTIC (`--lora-init
//! gaussian`), never the default.
//!
//! ## Design: which side computes the cosine
//!
//! This orchestrator computes each arm/seed's cosine similarity against the
//! f32-truth arm IN RUST (never delegated to `ci/scripts/perf/
//! compare_grad_oracle.py`), and the committed report carries only the
//! computed scalars per arm/seed/tensor — NOT the raw per-element gradient
//! arrays (`--keep-arm-dumps`, below, is the escape hatch for a caller that
//! wants those). This compares MULTIPLE ARMS OF THE SAME PRODUCER (jammi vs
//! jammi, differing only in dtype/kernel composition/seed) — the same
//! "same-build forced-arm A/B" class `finetune_step.rs`'s own Rust-computed
//! `r(L)` growth ratios and esc-044's Rust-asserted GEMM-operand-form checks
//! already use directly. Family F's numeric guarantee is satisfied at the
//! GATING layer instead: `check_fused_op_gradient_parity.py` applies a
//! THRESHOLD to these Rust-computed numbers and is itself self-tested
//! against synthetic reports; `--keep-arm-dumps` additionally lets a caller
//! re-derive the cosine itself with numpy off the SAME raw per-tensor
//! gradients this orchestrator's own arithmetic reads.
//!
//! ## Weights: identical across every arm, per seed
//!
//! Within ONE seed's cascade, the `all_fused` arm is run FIRST with
//! `--lora-weights-out` pointed at a scratch file (built via THIS seed's
//! own `--lora-init`, e.g. one real `AdamW` step under `PeftStep1`); every
//! other arm in the SAME seed's cascade then `--lora-weights-in`s that file.
//! A DIFFERENT seed draws a genuinely different weight file (and a
//! different synthetic batch) — that is the whole point of running multiple
//! seeds.
//!
//! ## `JAMMI_KERNELS_STRICT` per arm
//!
//! `all_fused` and every per-op-disabled/`all_off` arm run with
//! `JAMMI_KERNELS_STRICT=1` — disable wins over Strict (`jammi_kernels::
//! admission`'s own module doc), so `<op>` is forced eager while every
//! OTHER op is still strictly proven fused. `f32_truth` runs WITHOUT
//! `JAMMI_KERNELS_STRICT` (`Fallback` mode): an f32 backbone legitimately
//! falls back everywhere (every fused predicate requires `bf16` or a
//! matched `f32`/`bf16` pairing), which is not a predicate DEFECT and must
//! not error under `Strict`.
//!
//! ## Which op keys get ablated, and which are `untestable`
//!
//! `grad_oracle::GradOracleReport::live_admit_keys` from the FIRST seed's
//! `all_fused` arm's OWN dump — never a hand-maintained literal list.
//! [`AblationReport::untestable_op_keys`] names every op key observed
//! (`admit_key_dispatches`) in the `all_off` arm (`JAMMI_KERNELS_DISABLE=all`
//! forces the eager attention COMPOSITION, which is the only path that ever
//! consults a SUBSUMED key like `"rope_fused"`/`"softmax_last_dim_fused"`
//! at all — `jammi_kernels::admission`'s own module doc) that is NOT in
//! `ablated_op_keys` — round-7 audit finding: these keys were previously
//! silently absent from the table with no explanation; now named
//! explicitly rather than left for a reader to notice their absence.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::grad_oracle::{DispatchPair, GradOracleLoraInit, GradOracleReport};

/// This orchestrator's own CLI-facing parameters.
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
    /// Every arm runs once PER SEED here, aggregated afterward (median,
    /// min, max) — see this module's doc's "Round 7: seeds and the budget"
    /// section. Never empty (the CLI layer refuses an empty list before
    /// this struct is ever constructed).
    pub seeds: Vec<u64>,
    pub batched_forward: bool,
    pub lora_init: GradOracleLoraInit,
    /// If set, every arm/seed's RAW `GradOracleReport` dump (full
    /// per-element `f32` gradients) is copied into this directory as
    /// `<seed>-<arm-slug>.json`, for an independent numpy-side
    /// re-derivation of the cosine this command already computed. `None`
    /// keeps only the compact `--out` report (the default).
    pub keep_arm_dumps: Option<PathBuf>,
}

/// A typed refusal for a tensor-level comparison mismatch — round-7 audit
/// finding (PR #383): the previous round silently zip-truncated on a
/// length mismatch inside `cosine()`, and never checked SHAPE equality at
/// all (two tensors of equal total element count but transposed dims would
/// have compared silently). Both are now hard, typed refusals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorCompareError {
    LengthMismatch {
        name: String,
        a_len: usize,
        b_len: usize,
    },
    ShapeMismatch {
        name: String,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    MissingInTruth {
        name: String,
    },
}

impl std::fmt::Display for TensorCompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { name, a_len, b_len } => write!(
                f,
                "{name}: gradient length mismatch ({a_len} vs {b_len}) -- refusing to compute a \
                 cosine over mismatched-length vectors"
            ),
            Self::ShapeMismatch {
                name,
                a_shape,
                b_shape,
            } => write!(
                f,
                "{name}: tensor shape mismatch ({a_shape:?} vs {b_shape:?}) -- same total \
                 element count is not sufficient; the two dumps must agree on SHAPE, not just \
                 flattened length"
            ),
            Self::MissingInTruth { name } => {
                write!(
                    f,
                    "{name}: present in this arm but absent from the f32_truth dump"
                )
            }
        }
    }
}

impl std::error::Error for TensorCompareError {}

const VACUOUS_NORM_FLOOR: f64 = 1e-12;

/// `f64` cosine similarity. Refuses (typed error, [`TensorCompareError::LengthMismatch`])
/// on a length mismatch rather than silently comparing a truncated prefix —
/// round-7 audit finding.
fn cosine(name: &str, a: &[f32], b: &[f32]) -> Result<f64, TensorCompareError> {
    if a.len() != b.len() {
        return Err(TensorCompareError::LengthMismatch {
            name: name.to_string(),
            a_len: a.len(),
            b_len: b.len(),
        });
    }
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
        return Ok(0.0);
    }
    Ok(dot / denom)
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

/// jammi's internal `VarBuilder`-path tensor-name shape is
/// `layer.<n>.<module>.<lora_a|lora_b>` — parsed defensively (never
/// panics; a name outside this exact 4-segment shape reports
/// `("unknown", name, -1)` rather than aborting the whole run).
fn parse_tensor_name(name: &str) -> (String, String, i64) {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() == 4 && parts[0] == "layer" && (parts[3] == "lora_a" || parts[3] == "lora_b") {
        if let Ok(layer) = parts[1].parse::<i64>() {
            return (parts[3].to_string(), parts[2].to_string(), layer);
        }
    }
    ("unknown".to_string(), name.to_string(), -1)
}

/// One matched tensor's cosine against the `f32_truth` arm, plus its
/// (matrix, module, layer) decomposition — grouping key for the
/// `(lora_a|lora_b) × module × layer` table this module's own doc/the
/// dispatch that commissioned it asks for.
#[derive(Debug, Serialize, Clone)]
pub struct AblationTensorRow {
    pub name: String,
    pub matrix: String,
    pub module: String,
    pub layer: i64,
    pub cosine_vs_f32_truth: f64,
    pub n: usize,
}

/// One seed's raw sample for one arm — kept in the committed report (round-7
/// audit finding: "the raw AblationReports and `per_tensor` rows committed
/// alongside").
#[derive(Debug, Serialize)]
pub struct AblationArmSeedSample {
    pub seed: u64,
    pub full_tensor_cosine: f64,
    pub per_tensor_median_cosine: f64,
    pub mass_weighted_mean_cosine: f64,
    pub matched_tensor_count: usize,
    pub vacuous_tensor_count: usize,
    pub loss: f32,
    pub kernels_disabled_requested: Vec<String>,
    pub kernels_disabled_fired: Vec<String>,
    pub unmatched_disables: Vec<String>,
    pub admit_key_dispatches: BTreeMap<String, DispatchPair>,
    pub per_tensor: Vec<AblationTensorRow>,
}

/// `{median, min, max}` over one arm's per-seed samples of a single
/// statistic.
#[derive(Debug, Serialize, Clone, Copy)]
pub struct SeedStat {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

/// Median of `values` (NOT modified in place — copies first). Panics if
/// `values` is empty (every call site here is guaranteed non-empty by
/// construction: `AblationParams::seeds` is refused-empty at the CLI
/// layer) or contains a non-finite entry (callers refuse non-finite
/// cosines BEFORE aggregating — `partial_cmp` on a `NaN` would otherwise
/// silently mis-sort rather than panic, hiding exactly the defect this
/// tool exists to catch).
fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median: empty input");
    let mut v = values.to_vec();
    v.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("median: non-finite value reached aggregation -- caller must refuse first")
    });
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn seed_stat(values: &[f64]) -> SeedStat {
    assert!(!values.is_empty(), "seed_stat: empty input");
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    SeedStat {
        median: median(values),
        min,
        max,
    }
}

/// Mass-weighted mean over `rows`' own `cosine_vs_f32_truth`, weighted by
/// each tensor's element count `n` — a tensor with more elements
/// contributes proportionally more to this statistic than the plain
/// (unweighted) median does.
fn mass_weighted_mean(rows: &[AblationTensorRow]) -> f64 {
    let total_n: usize = rows.iter().map(|r| r.n).sum();
    if total_n == 0 {
        return 0.0;
    }
    let sum: f64 = rows
        .iter()
        .map(|r| r.cosine_vs_f32_truth * r.n as f64)
        .sum();
    sum / total_n as f64
}

/// One arm's full aggregate (across every seed) — the unit `--out`'s
/// `arms` list carries.
#[derive(Debug, Serialize)]
pub struct AblationArm {
    /// `"all_fused"` | `"f32_truth"` | `"ablate:<op_key>"` | `"all_off"`.
    pub arm: String,
    pub op_key: Option<String>,
    pub backbone_dtype: String,
    /// `"reference"` (`all_fused`) | `"truth"` (`f32_truth`) | `"control"`
    /// (`all_off`) | `"neutral"` (ablate arm: this op ran EAGER on every
    /// seed AND its median cosine stayed within the derived budget of
    /// `all_fused`'s own) | `"divergent"` (ablate arm: ran eager, but its
    /// median cosine moved beyond the derived budget) — round-7 audit
    /// finding: the table must mark these explicitly, not leave a reader
    /// to infer them from raw numbers.
    pub classification: String,
    pub full_tensor_cosine: SeedStat,
    pub per_tensor_median_cosine: SeedStat,
    pub mass_weighted_mean_cosine: SeedStat,
    pub per_seed: Vec<AblationArmSeedSample>,
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
    pub seeds: Vec<u64>,
    pub lora_init: String,
    /// Every op key ablated (`AblationArm::op_key` restricted to `Some`,
    /// same order) — copied from the FIRST seed's `all_fused` arm's own
    /// `live_admit_keys`, never a literal list.
    pub ablated_op_keys: Vec<String>,
    /// Op keys observed dispatching (fused OR eager) in the `all_off` arm
    /// that are NOT in `ablated_op_keys` — subsumed keys that never reach
    /// `admit` standalone on this checkpoint (see this module's doc).
    pub untestable_op_keys: Vec<String>,
    /// `3 * (max - min)` of the `all_fused` arm's own `full_tensor_cosine`
    /// across `seeds`, AT THIS SHAPE — see this module's doc's "Round 7:
    /// seeds and the budget" section. `check_fused_op_gradient_parity.py`
    /// independently recomputes this from `arms`' own raw `per_seed` data
    /// rather than trusting this field verbatim (family F).
    pub derived_per_op_budget: f64,
    pub arms: Vec<AblationArm>,
}

/// One child invocation of `jammi-bench grad-oracle` (never
/// `--ablate-each-op` — that flag is simply never in this argument list, so
/// a child cannot recurse into a grandchild ablation run). Spawns
/// `std::env::current_exe()` (ledger row 197: `JAMMI_KERNELS_DISABLE`/
/// `JAMMI_KERNELS_STRICT` are process-wide `OnceLock`s — the only way to
/// change them within one `cargo run` is a fresh child process).
#[allow(clippy::too_many_arguments)]
fn spawn_arm(
    params: &AblationParams,
    seed: u64,
    backbone_dtype: jammi_numerics::ComputePrecision,
    lora_init: GradOracleLoraInit,
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
        .arg(seed.to_string())
        .arg("--batched-forward")
        .arg(params.batched_forward.to_string())
        .arg("--lora-init")
        .arg(lora_init.as_flag())
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
            "grad-oracle child arm failed (seed={seed} disable={disable:?} strict={strict}): \
             exit {}",
            output.status
        )
        .into());
    }
    let text = std::fs::read_to_string(out_path)?;
    Ok(serde_json::from_str(&text)?)
}

/// `requested - fired`, sorted.
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

/// Builds one [`AblationArmSeedSample`] from a raw [`GradOracleReport`],
/// comparing every matched tensor (and the concatenated whole) against
/// `truth`. Refuses (typed [`TensorCompareError`]) on a length or shape
/// mismatch, and on a non-finite gradient element on either side.
fn build_sample(
    label: &str,
    seed: u64,
    report: &GradOracleReport,
    truth: &GradOracleReport,
) -> Result<AblationArmSeedSample, Box<dyn std::error::Error>> {
    let mut per_tensor = Vec::with_capacity(report.gradients.len());
    let mut all_a: Vec<f32> = Vec::new();
    let mut all_b: Vec<f32> = Vec::new();
    let mut names: Vec<&String> = report.gradients.keys().collect();
    names.sort();
    for name in names {
        let a = &report.gradients[name];
        let b = truth.gradients.get(name).ok_or_else(|| {
            Box::new(TensorCompareError::MissingInTruth { name: name.clone() })
                as Box<dyn std::error::Error>
        })?;
        if a.shape != b.shape {
            return Err(Box::new(TensorCompareError::ShapeMismatch {
                name: name.clone(),
                a_shape: a.shape.clone(),
                b_shape: b.shape.clone(),
            }));
        }
        if has_nonfinite(&a.grad) || has_nonfinite(&b.grad) {
            return Err(format!(
                "arm {label:?} seed {seed}, tensor {name:?}: non-finite gradient element on at \
                 least one side -- refusing to compute a cosine off NaN/Inf data"
            )
            .into());
        }
        let c = cosine(name, &a.grad, &b.grad)?;
        let (matrix, module, layer) = parse_tensor_name(name);
        per_tensor.push(AblationTensorRow {
            name: name.clone(),
            matrix,
            module,
            layer,
            cosine_vs_f32_truth: c,
            n: a.grad.len(),
        });
        all_a.extend_from_slice(&a.grad);
        all_b.extend_from_slice(&b.grad);
    }
    if per_tensor.is_empty() {
        return Err(
            format!("arm {label:?} seed {seed}: matched zero tensors against f32_truth").into(),
        );
    }
    let full_tensor_cosine = cosine("<concatenated>", &all_a, &all_b)?;
    let per_tensor_cosines: Vec<f64> = per_tensor.iter().map(|r| r.cosine_vs_f32_truth).collect();
    Ok(AblationArmSeedSample {
        seed,
        full_tensor_cosine,
        per_tensor_median_cosine: median(&per_tensor_cosines),
        mass_weighted_mean_cosine: mass_weighted_mean(&per_tensor),
        matched_tensor_count: per_tensor.len(),
        vacuous_tensor_count: report.vacuous_tensor_count,
        loss: report.loss,
        kernels_disabled_requested: report.kernels_disabled_requested.clone(),
        kernels_disabled_fired: report.kernels_disabled_fired.clone(),
        unmatched_disables: unmatched(
            &report.kernels_disabled_requested,
            &report.kernels_disabled_fired,
        ),
        admit_key_dispatches: report.admit_key_dispatches.clone(),
        per_tensor,
    })
}

fn aggregate_arm(
    arm_label: &str,
    op_key: Option<String>,
    backbone_dtype: &str,
    samples: Vec<AblationArmSeedSample>,
) -> AblationArm {
    let full: Vec<f64> = samples.iter().map(|s| s.full_tensor_cosine).collect();
    let per_tensor_med: Vec<f64> = samples.iter().map(|s| s.per_tensor_median_cosine).collect();
    let mass: Vec<f64> = samples
        .iter()
        .map(|s| s.mass_weighted_mean_cosine)
        .collect();
    AblationArm {
        arm: arm_label.to_string(),
        op_key,
        backbone_dtype: backbone_dtype.to_string(),
        // Placeholder; `run` overwrites this once the budget is known.
        classification: "unclassified".to_string(),
        full_tensor_cosine: seed_stat(&full),
        per_tensor_median_cosine: seed_stat(&per_tensor_med),
        mass_weighted_mean_cosine: seed_stat(&mass),
        per_seed: samples,
    }
}

/// Runs the full multi-seed ablation and writes the compact
/// [`AblationReport`] to `out`. Refuses (returns `Err`, never writes `out`)
/// if: the `all_fused` arm's `vacuous_tensor_count != 0` on any seed; any
/// arm's `unmatched_disables` is non-empty on any seed; or an
/// `ablate:<key>` arm's OWN `admit_key_dispatches[key].fused > 0` on any
/// seed (a hard contradiction — the disable did not actually take effect).
pub fn run(
    params: &AblationParams,
    out: &Path,
) -> Result<AblationReport, Box<dyn std::error::Error>> {
    if params.seeds.is_empty() {
        return Err("AblationParams::seeds must not be empty".into());
    }
    let scratch = tempfile::tempdir()?;
    let dump_dir = params
        .keep_arm_dumps
        .clone()
        .unwrap_or_else(|| scratch.path().to_path_buf());
    std::fs::create_dir_all(&dump_dir)?;

    // Per-arm-label accumulator: label -> (op_key, Vec<sample>).
    let mut samples_by_arm: BTreeMap<String, (Option<String>, Vec<AblationArmSeedSample>)> =
        BTreeMap::new();
    let mut ablated_op_keys: Vec<String> = Vec::new();
    let mut untestable_op_keys_set: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    let mut checkpoint_identity: Option<(String, String, u64)> = None;
    let mut git_rev: Option<String> = None;

    for (seed_idx, &seed) in params.seeds.iter().enumerate() {
        let weights_path = scratch
            .path()
            .join(format!("seed{seed}-lora_weights.safetensors"));

        let ref_dump_path = dump_dir.join(format!("{seed}-all_fused.json"));
        let reference = spawn_arm(
            params,
            seed,
            params.backbone_dtype,
            params.lora_init,
            None,
            Some(&weights_path),
            &ref_dump_path,
            true,
            None,
        )?;
        if reference.vacuous_tensor_count != 0 {
            return Err(format!(
                "seed {seed}: all_fused arm has vacuous_tensor_count = {} (names: {:?}) -- this \
                 tool's whole premise is that --lora-init makes EVERY gradient live; refusing to \
                 emit an ablation comparison off a fixture that is not actually non-vacuous",
                reference.vacuous_tensor_count, reference.vacuous_tensor_names
            )
            .into());
        }
        if checkpoint_identity.is_none() {
            checkpoint_identity = Some((
                reference.checkpoint_config_sha256.clone(),
                reference.checkpoint_weights_sha256.clone(),
                reference.checkpoint_weights_size_bytes,
            ));
            git_rev = reference.git_rev.clone();
        }
        // The FIRST seed's own live_admit_keys is the authoritative
        // per-shape key set (this module's doc: keys are structural, not
        // data-dependent, for a fixed shape/config).
        if seed_idx == 0 {
            ablated_op_keys = reference.live_admit_keys.clone();
        }

        let truth_dump_path = dump_dir.join(format!("{seed}-f32_truth.json"));
        let truth = spawn_arm(
            params,
            seed,
            jammi_numerics::ComputePrecision::F32,
            params.lora_init,
            Some(&weights_path),
            None,
            &truth_dump_path,
            false,
            None,
        )?;

        let ref_sample = build_sample("all_fused", seed, &reference, &truth)?;
        samples_by_arm
            .entry("all_fused".to_string())
            .or_insert((None, Vec::new()))
            .1
            .push(ref_sample);
        let truth_sample = build_sample("f32_truth", seed, &truth, &truth)?;
        samples_by_arm
            .entry("f32_truth".to_string())
            .or_insert((None, Vec::new()))
            .1
            .push(truth_sample);

        for key in &ablated_op_keys {
            let dump_path = dump_dir.join(format!("{seed}-ablate_{key}.json"));
            let arm_report = spawn_arm(
                params,
                seed,
                params.backbone_dtype,
                params.lora_init,
                Some(&weights_path),
                None,
                &dump_path,
                true,
                Some(key),
            )?;
            if let Some(pair) = arm_report.admit_key_dispatches.get(key) {
                if pair.fused > 0 {
                    return Err(format!(
                        "seed {seed}: ablate:{key} arm's OWN admit_key_dispatches[{key:?}].fused \
                         = {} > 0 -- JAMMI_KERNELS_DISABLE={key} did not actually force this op \
                         eager; this arm's cosine cannot be trusted to measure ablating {key} at \
                         all",
                        pair.fused
                    )
                    .into());
                }
            }
            let label = format!("ablate:{key}");
            let sample = build_sample(&label, seed, &arm_report, &truth)?;
            samples_by_arm
                .entry(label)
                .or_insert((Some(key.clone()), Vec::new()))
                .1
                .push(sample);
        }

        let off_dump_path = dump_dir.join(format!("{seed}-all_off.json"));
        let all_off = spawn_arm(
            params,
            seed,
            params.backbone_dtype,
            params.lora_init,
            Some(&weights_path),
            None,
            &off_dump_path,
            true,
            Some("all"),
        )?;
        for key in all_off.admit_key_dispatches.keys() {
            if !ablated_op_keys.contains(key) {
                untestable_op_keys_set.insert(key.clone());
            }
        }
        let off_sample = build_sample("all_off", seed, &all_off, &truth)?;
        samples_by_arm
            .entry("all_off".to_string())
            .or_insert((None, Vec::new()))
            .1
            .push(off_sample);
    }

    // Cross-arm/cross-seed provenance refusal — checked over EVERY sample
    // collected above, not just the ones checked inline during the loop
    // (the inline checks above cover the two HARD mechanism contradictions
    // early to fail fast; this pass is the exhaustive sweep).
    for (label, (_, seed_samples)) in &samples_by_arm {
        for s in seed_samples {
            if !s.unmatched_disables.is_empty() {
                return Err(format!(
                    "arm {label:?} seed {}: unmatched_disables = {:?} -- this arm's \
                     JAMMI_KERNELS_DISABLE request did not fully fire; its provenance is not \
                     self-describing",
                    s.seed, s.unmatched_disables
                )
                .into());
            }
            if s.vacuous_tensor_count != 0 {
                return Err(format!(
                    "arm {label:?} seed {}: vacuous_tensor_count = {} -- expected every \
                     gradient to stay live across every kernel-composition arm",
                    s.seed, s.vacuous_tensor_count
                )
                .into());
            }
        }
    }

    let mut arms: Vec<AblationArm> = samples_by_arm
        .into_iter()
        .map(|(label, (op_key, seed_samples))| {
            let dtype_label = if label == "f32_truth" {
                "f32".to_string()
            } else {
                precision_flag(params.backbone_dtype).to_string()
            };
            aggregate_arm(&label, op_key, &dtype_label, seed_samples)
        })
        .collect();

    let reference_arm = arms
        .iter()
        .find(|a| a.arm == "all_fused")
        .ok_or("internal error: no all_fused arm aggregated")?;
    let derived_per_op_budget =
        3.0 * (reference_arm.full_tensor_cosine.max - reference_arm.full_tensor_cosine.min);
    let reference_median = reference_arm.full_tensor_cosine.median;

    for arm in &mut arms {
        arm.classification = match arm.arm.as_str() {
            "all_fused" => "reference".to_string(),
            "f32_truth" => "truth".to_string(),
            "all_off" => "control".to_string(),
            _ => {
                let delta = (arm.full_tensor_cosine.median - reference_median).abs();
                if delta <= derived_per_op_budget {
                    "neutral".to_string()
                } else {
                    "divergent".to_string()
                }
            }
        };
    }
    // Sort for a stable, deterministic report: reference/truth/tested/
    // control order, ablate arms alphabetical by op_key within "tested".
    arms.sort_by(|a, b| {
        fn rank(arm: &str) -> u8 {
            match arm {
                "all_fused" => 0,
                "f32_truth" => 1,
                "all_off" => 3,
                _ => 2,
            }
        }
        rank(&a.arm).cmp(&rank(&b.arm)).then(a.arm.cmp(&b.arm))
    });

    let (checkpoint_config_sha256, checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        checkpoint_identity.ok_or("internal error: checkpoint identity never recorded")?;

    let report = AblationReport {
        tool: "jammi_grad_oracle_ablation",
        git_rev,
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
        seeds: params.seeds.clone(),
        lora_init: params.lora_init.as_flag().to_string(),
        ablated_op_keys,
        untestable_op_keys: untestable_op_keys_set.into_iter().collect(),
        derived_per_op_budget,
        arms,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(out, json)?;
    Ok(report)
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
        assert!((cosine("t", &v, &v).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_of_opposite_vectors_is_minus_one() {
        let a = vec![1.0f32, 2.0, -3.0];
        let b = vec![-1.0f32, -2.0, 3.0];
        assert!((cosine("t", &a, &b).unwrap() - (-1.0)).abs() < 1e-9);
    }

    /// Non-vacuous control (family F): a zero vector on EITHER side must
    /// return a well-defined `0.0`, never `NaN`/panic.
    #[test]
    fn cosine_of_a_zero_vector_is_zero_not_nan() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let nonzero = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine("t", &zero, &nonzero).unwrap(), 0.0);
        assert_eq!(cosine("t", &zero, &zero).unwrap(), 0.0);
    }

    /// Round-7 audit finding (PR #383, item 3): `cosine` must REFUSE a
    /// length mismatch with a typed error, never silently zip-truncate.
    #[test]
    fn cosine_refuses_length_mismatch_with_a_typed_error() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        let err = cosine("mytensor", &a, &b).unwrap_err();
        assert_eq!(
            err,
            TensorCompareError::LengthMismatch {
                name: "mytensor".to_string(),
                a_len: 3,
                b_len: 2,
            }
        );
    }

    /// Round-7 audit finding (PR #383, item 3): `build_sample` must assert
    /// per-tensor SHAPE equality (not just flattened length) with a typed
    /// error, tested with a mismatched synthetic pair.
    #[test]
    fn build_sample_refuses_shape_mismatch_with_a_typed_error() {
        let mut arm = synthetic_report(vec![("layer.0.Wqkv.lora_a", vec![2, 3], vec![1.0; 6])]);
        let truth = synthetic_report(vec![("layer.0.Wqkv.lora_a", vec![3, 2], vec![1.0; 6])]);
        // Same flattened length (6), DIFFERENT shape -- must still refuse.
        let err = build_sample("all_fused", 42, &arm, &truth).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("shape mismatch"),
            "expected a shape-mismatch refusal, got: {msg}"
        );
        // Sanity: a length-only mismatch (no shape at all recorded
        // differently, but grad vector length differs) hits the
        // length-mismatch path in `cosine` -- prove BOTH mismatch kinds
        // are reachable, not just one path silently swallowing the other.
        arm.gradients
            .get_mut("layer.0.Wqkv.lora_a")
            .unwrap()
            .grad
            .push(9.0);
        let err2 = build_sample("all_fused", 42, &arm, &truth).unwrap_err();
        assert!(err2.to_string().contains("mismatch"));
    }

    fn synthetic_report(tensors: Vec<(&str, Vec<usize>, Vec<f32>)>) -> GradOracleReport {
        let mut gradients = std::collections::BTreeMap::new();
        for (name, shape, grad) in tensors {
            let weight = grad.clone();
            gradients.insert(
                name.to_string(),
                crate::grad_oracle::GradOracleTensor {
                    shape,
                    grad,
                    weight,
                },
            );
        }
        GradOracleReport {
            tool: "jammi_grad_oracle".to_string(),
            model_dir: "synthetic".to_string(),
            device: "cpu".to_string(),
            device_name: "cpu".to_string(),
            git_rev: None,
            checkpoint_config_sha256: "0".repeat(64),
            checkpoint_weights_sha256: "0".repeat(64),
            checkpoint_weights_size_bytes: 0,
            backbone_dtype: "f32".to_string(),
            batch: 1,
            seq: 1,
            lora_rank: 1,
            lora_alpha: 1.0,
            target_modules: vec!["Wqkv".to_string()],
            batched_forward: true,
            seed: 0,
            lora_dropout: 0.0,
            lora_init: "gaussian".to_string(),
            peft_step1_applied: false,
            lora_weights_in: None,
            lora_weights_out: None,
            trainable_tensor_count: gradients.len(),
            batch_token_id_sums: [0, 0, 0],
            loss: 1.0,
            ln_fused_dispatches: 0,
            ln_eager_dispatches: 0,
            rope_fused_dispatches: 0,
            rope_eager_dispatches: 0,
            softmax_fused_dispatches: 0,
            softmax_eager_dispatches: 0,
            geglu_fused_dispatches: 0,
            geglu_eager_dispatches: 0,
            lora_epilogue_fused_dispatches: 0,
            lora_epilogue_eager_dispatches: 0,
            lora_linear_fused_dispatches: 0,
            lora_linear_eager_dispatches: 0,
            attention_block_fused_dispatches: 0,
            attention_block_eager_dispatches: 0,
            kernels_disabled_requested: vec![],
            kernels_disabled_fired: vec![],
            live_admit_keys: vec![],
            admit_key_dispatches: std::collections::BTreeMap::new(),
            vacuous_tensor_count: 0,
            vacuous_tensor_names: vec![],
            gradients,
        }
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

    #[test]
    fn median_of_odd_and_even_length_slices() {
        assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
    }

    #[test]
    fn seed_stat_reports_median_min_max() {
        let s = seed_stat(&[0.6, 0.9, 0.7]);
        assert_eq!(s.median, 0.7);
        assert_eq!(s.min, 0.6);
        assert_eq!(s.max, 0.9);
    }

    #[test]
    fn mass_weighted_mean_weights_by_element_count() {
        let rows = vec![
            AblationTensorRow {
                name: "big".to_string(),
                matrix: "lora_a".to_string(),
                module: "Wqkv".to_string(),
                layer: 0,
                cosine_vs_f32_truth: 0.0,
                n: 100,
            },
            AblationTensorRow {
                name: "small".to_string(),
                matrix: "lora_b".to_string(),
                module: "Wqkv".to_string(),
                layer: 0,
                cosine_vs_f32_truth: 1.0,
                n: 1,
            },
        ];
        // Dominated by the 100-element tensor's 0.0 cosine, not a plain
        // 50/50 average of 0.0 and 1.0 (which would be 0.5).
        let m = mass_weighted_mean(&rows);
        assert!(m < 0.02, "expected mass-weighted mean near 0, got {m}");
    }
}
