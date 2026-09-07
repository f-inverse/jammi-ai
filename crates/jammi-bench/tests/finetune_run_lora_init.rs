//! `finetune-run --lora-init {zeros_b|gaussian}` (issue #421 P1-b(ii)),
//! driven through the REAL compiled CLI.
//!
//! The flag exists for the #421 BF16 pre-flight (contract P2): under
//! `zeros_b` every LoRA `B` is zero, so the adapter contributes exactly
//! nothing at step 0 and `dL/dA == 0` there — a "every LoRA `Var` received a
//! non-zero gradient" bf16 check is VACUOUS in that mode and would pass on a
//! dtype path that never worked. `gaussian` makes the adapter non-identity
//! at construction, which is what turns that check into a real one.
//!
//! What this file proves, and why it is not a flag-plumbing tautology: the
//! mode must reach the WEIGHTS, not merely the report. The mechanism
//! assertion is the run's own INIT PROBE — `finetune_run::run_impl`
//! evaluates the held-out probe batch BEFORE the first optimizer step
//! (`train_probe_series[0]`). Under `zeros_b` that probe is, by
//! construction, the BASE model's loss (the adapter is identity); under
//! `gaussian` it cannot be, because `B != 0` perturbs every wrapped linear's
//! output. Two runs identical in every other flag (same seed, same
//! selectors, same fixture) must therefore disagree on that first probe
//! value — and agree on nothing having gone non-finite.
//!
//! A fresh child PROCESS per case (`env!("CARGO_BIN_EXE_jammi-bench")`),
//! matching every other integration test in this directory; the fixture
//! helpers are re-derived locally for the same reason
//! `finetune_run_kernel_disable.rs` gives (this crate is `[[bin]]`-only, so
//! there is no library target a test could import them from).

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

fn write_triplets_jsonl(dir: &Path, name: &str, n: usize, offset: usize) -> PathBuf {
    let path = dir.join(name);
    let mut body = String::new();
    for i in 0..n {
        let k = offset + i;
        body.push_str(&format!(
            "{{\"anchor_id\":\"a{k}\",\"anchor_text\":\"synthetic anchor sentence number {k} \
             about widgets\",\"positive_id\":\"p{k}\",\"positive_text\":\"synthetic positive \
             sentence number {k} about widgets too\",\"negative_id\":\"n{k}\",\"negative_text\":\
             \"synthetic negative sentence number {k} about gadgets instead\"}}\n"
        ));
    }
    std::fs::write(&path, body).expect("write triplets jsonl");
    path
}

fn write_heldout_ids(dir: &Path, n: usize, offset: usize) -> PathBuf {
    let path = dir.join("heldout_ids.txt");
    let mut body = String::new();
    for i in 0..n {
        let k = offset + i;
        body.push_str(&format!("a{k}\tp{k}\tn{k}\n"));
    }
    std::fs::write(&path, body).expect("write heldout ids");
    path
}

/// One `finetune-run`, everything pinned except `--lora-init` (which the
/// caller appends, or omits entirely to exercise the default).
fn base_command(work_dir: &Path, fixtures_dir: &Path) -> Command {
    let train_jsonl = write_triplets_jsonl(fixtures_dir, "train.jsonl", 4, 0);
    let heldout_jsonl = write_triplets_jsonl(fixtures_dir, "heldout.jsonl", 2, 100);
    let heldout_ids = write_heldout_ids(fixtures_dir, 2, 100);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args(["finetune-run", "--model-dir"])
        .arg(model_dir())
        .args(["--arm", "fused"])
        .arg("--train-jsonl")
        .arg(&train_jsonl)
        .arg("--heldout-ids")
        .arg(&heldout_ids)
        .arg("--heldout-jsonl")
        .arg(&heldout_jsonl)
        .args([
            "--seed",
            "7",
            "--epochs",
            "1",
            "--eval-cadence",
            "1",
            "--batch",
            "2",
            "--lr",
            "0.001",
            "--schedule",
            "constant",
            "--validation-fraction",
            "0.0",
            "--early-stopping-patience",
            "10000",
            "--early-stopping-metric",
            "train_loss",
            "--max-grad-norm",
            "0.0",
            // MNRL, not Triplet, for ONE measured reason: on this tiny
            // fixture the triplet loss is SATURATED at its own margin (the
            // untrained probe reads exactly `--margin` for both init modes,
            // since `max(0, margin + d_pos - d_neg)` clips whenever the
            // three embeddings are near-identical), which makes it blind to
            // any perturbation and would render the mechanism assertion
            // below vacuously equal. MNRL's in-batch-negative cross-entropy
            // has no such ceiling, so the probe actually responds to the
            // adapter. Nothing here depends on which objective is used —
            // the flag under test is `--lora-init`.
            "--objective",
            "mnrl",
            "--temperature",
            "20.0",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0.0",
            "--target-modules",
            "query,value",
            "--backbone-dtype",
            "f32",
            "--max-seq-length",
            "16",
        ])
        .arg("--work-dir")
        .arg(work_dir);
    cmd
}

fn run_and_read_tier(cmd: &mut Command) -> serde_json::Value {
    let output = cmd.output().expect("spawn jammi-bench finetune-run");
    assert!(
        output.status.success(),
        "finetune-run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    report["tiers"]["finetune_run"].clone()
}

/// The untrained init probe — `train_probe_series[0]`, the value
/// `finetune_run::run_impl` measures BEFORE this run's first optimizer step.
/// Checked for FINITENESS on the way out: `NaN != NaN` is true, so a
/// diverged run would otherwise satisfy the "the two modes differ"
/// assertion below for entirely the wrong reason.
fn init_probe(tier: &serde_json::Value) -> f64 {
    let series = tier["train_probe_series"]
        .as_array()
        .unwrap_or_else(|| panic!("train_probe_series must be an array: tier={tier}"));
    let v = series
        .first()
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| panic!("train_probe_series[0] must be a number: tier={tier}"));
    assert!(
        v.is_finite(),
        "the init probe must be finite; a non-finite probe is a diverged run, not a datum ({v})"
    );
    v
}

/// Default and explicit `zeros_b` are the SAME run: the flag's default is
/// byte-identical to what every invocation written before it existed did.
/// Compared on the emitted field AND on the measured probe, so this is a
/// statement about the run, not only about the report.
#[test]
fn the_default_is_zeros_b_and_is_identical_to_stating_it_explicitly() {
    let work_a = tempfile::tempdir().expect("tempdir");
    let fixtures_a = tempfile::tempdir().expect("fixtures tempdir");
    let default_tier = run_and_read_tier(&mut base_command(work_a.path(), fixtures_a.path()));

    let work_b = tempfile::tempdir().expect("tempdir");
    let fixtures_b = tempfile::tempdir().expect("fixtures tempdir");
    let explicit_tier = run_and_read_tier(
        base_command(work_b.path(), fixtures_b.path()).args(["--lora-init", "zeros_b"]),
    );

    assert_eq!(
        default_tier["lora_init"],
        serde_json::json!("zeros_b"),
        "the omitted flag must record the default it actually ran under: tier={default_tier}"
    );
    assert_eq!(explicit_tier["lora_init"], serde_json::json!("zeros_b"));
    assert_eq!(
        init_probe(&default_tier),
        init_probe(&explicit_tier),
        "stating the default explicitly must change nothing about the run"
    );
}

/// The mechanism assertion: `gaussian` reaches the WEIGHTS. A `zeros_b`
/// adapter is identity at construction, so its init probe is the base
/// model's; a `gaussian` one is not. Same seed, same selectors, same
/// fixture — the ONLY difference is the flag, so a producer that recorded
/// `"gaussian"` while still building `ZerosB` weights (the exact failure of
/// threading the flag into one of the two `LoraBuildConfig` sites and not
/// the other) reads an IDENTICAL probe here and this test goes red.
#[test]
fn gaussian_actually_changes_the_initialized_adapter_not_just_the_report() {
    let work_zeros = tempfile::tempdir().expect("tempdir");
    let fixtures_zeros = tempfile::tempdir().expect("fixtures tempdir");
    let zeros_tier = run_and_read_tier(
        base_command(work_zeros.path(), fixtures_zeros.path()).args(["--lora-init", "zeros_b"]),
    );

    let work_gauss = tempfile::tempdir().expect("tempdir");
    let fixtures_gauss = tempfile::tempdir().expect("fixtures tempdir");
    let gauss_tier = run_and_read_tier(
        base_command(work_gauss.path(), fixtures_gauss.path()).args(["--lora-init", "gaussian"]),
    );

    assert_eq!(
        gauss_tier["lora_init"],
        serde_json::json!("gaussian"),
        "tier={gauss_tier}"
    );
    let zeros_probe = init_probe(&zeros_tier);
    let gauss_probe = init_probe(&gauss_tier);
    assert_ne!(
        zeros_probe, gauss_probe,
        "a gaussian-initialized adapter is NON-identity at construction, so the untrained probe \
         cannot equal the zeros_b one ({zeros_probe} vs {gauss_probe}) — an equal pair means the \
         flag never reached the weights"
    );
}

/// An unrecognized value is a typed refusal naming both accepted tokens —
/// the same posture `--objective`/`--arm`/`--task` already take — never a
/// silent fall-back to the default (which would run the WRONG
/// initialization under a caller's explicit instruction).
#[test]
fn an_unknown_lora_init_is_refused_before_anything_runs() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path())
        .args(["--lora-init", "kaiming"])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "an unknown --lora-init must refuse, never silently default — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("kaiming") && stderr.contains("zeros_b") && stderr.contains("gaussian"),
        "the refusal must name the bad value and both accepted tokens: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "a refused invocation must emit no report at all"
    );
}
