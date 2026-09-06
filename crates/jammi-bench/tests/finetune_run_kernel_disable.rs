//! `--arm alloff` kernel-disable control for `finetune-run` (unit 63 H4a's
//! own flagged gap, closed here — a coverage gap, not a parity gap:
//! `finetune_step_kernel_disable.rs`'s cell 10 (the safety property) proves
//! `finetune-step` refuses to emit a JSON tier when its declared
//! `JAMMI_KERNELS_DISABLE` intent was dropped/mistyped/partial; H4a shipped
//! the SAME check inside `finetune_run::run` for `Arm::Alloff` (`--arm
//! alloff requires JAMMI_KERNELS_DISABLE to resolve to exactly
//! {ALLOFF_KEYS}` — see that function's doc) but never added an integration
//! test driving it through the real compiled `jammi-bench finetune-run` CLI
//! entry point the way `finetune_step_kernel_disable.rs` does for
//! `finetune-step`. This file closes that gap.
//!
//! Each case spawns the compiled `jammi-bench` binary as a fresh child
//! PROCESS (`env!("CARGO_BIN_EXE_jammi-bench")`), never
//! `finetune_run::run` in-process — the SAME reason
//! `finetune_step_kernel_disable.rs`'s own doc gives:
//! `jammi_kernels::admission::disabled_ops`/`disabled_ops_requested` memoize
//! into a process-wide `OnceLock` read once per process, so an in-process
//! `std::env::set_var` test would race every other test in this crate's
//! shared test binary for who reads that `OnceLock` first. A fresh child
//! process side-steps it entirely.
//!
//! `ALLOFF_KEYS` (`finetune_run.rs`) is `attention_block_flash,adamw_step_fused`
//! verbatim (CONTRACT Frame) — spelled out literally here rather than
//! imported, because this crate is `[[bin]]`-only (no `[lib]` target an
//! integration test could `use jammi_bench::finetune_run::ALLOFF_KEYS`
//! from), mirroring every other test file in this directory's own
//! convention of re-deriving fixture/constant values locally.

use std::path::{Path, PathBuf};
use std::process::Command;

// `tiny_modernbert_classifier`, not `tiny_bert` (C-ATTN unit, campaign
// #462/#463): `fused_dispatch_proof_gate` (`finetune_run.rs`) now refuses
// ANY architecture's leg that took a real optimizer step with all four
// training-mode attention/flash dispatch counters at `0`, and classic
// BERT's attention forward does not dispatch through
// `jammi_kernels::admission::admit` at all until `jammi_encoders`'
// companion C-ATTN seam lands — so `tiny_bert` alone cannot satisfy the
// widened gate yet. This file's own claim (the `--arm alloff`
// two-op-set check) is architecture-agnostic, so it moved to the SAME
// already-wired ModernBert fixture `crate::finetune_run`'s own
// `non_perturbation_test_params_modernbert` test helper uses, with the
// matching `--target-modules Wqkv` selector below.
fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../cookbook/fixtures/tiny_modernbert_classifier")
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

/// A minimal `finetune-run` invocation, `--arm` left for the caller to
/// append (positional last so `alloff`-vs-`fused` cases share everything
/// else). Small enough (2 train batches, 1 held-out batch, 1 epoch) to run
/// fast as a `cargo test` case.
fn base_command(work_dir: &Path, fixtures_dir: &Path, arm: &str) -> Command {
    let train_jsonl = write_triplets_jsonl(fixtures_dir, "train.jsonl", 4, 0);
    let heldout_jsonl = write_triplets_jsonl(fixtures_dir, "heldout.jsonl", 2, 100);
    let heldout_ids = write_heldout_ids(fixtures_dir, 2, 100);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args(["finetune-run", "--model-dir"])
        .arg(model_dir())
        .args(["--arm", arm])
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
            "--warmup-steps",
            "0",
            "--weight-decay",
            "0.0",
            "--grad-accum",
            "1",
            "--validation-fraction",
            "0.0",
            "--early-stopping-patience",
            "10000",
            "--early-stopping-metric",
            "train_loss",
            "--max-grad-norm",
            "0.0",
            "--objective",
            "triplet",
            "--margin",
            "0.3",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0.0",
            "--target-modules",
            "Wqkv",
            "--backbone-dtype",
            "f32",
            "--max-seq-length",
            "16",
        ])
        .arg("--work-dir")
        .arg(work_dir);
    cmd
}

/// The safety property (mirrors `finetune_step_kernel_disable.rs`'s cell
/// 10): `--arm alloff` with NO `JAMMI_KERNELS_DISABLE` set at all must fail
/// the run — never emit a JSON tier as if the forced-eager arm had worked.
/// This is the "dropped var" failure mode: an operator declares `--arm
/// alloff` on the command line but the env var never reached this process
/// (an unforwarded ssh/`docker -e` environment looks identical).
#[test]
fn alloff_without_the_env_var_set_invalidates_the_run() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "alloff")
        .env_remove("JAMMI_KERNELS_DISABLE")
        .output()
        .expect("spawn jammi-bench finetune-run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "--arm alloff without JAMMI_KERNELS_DISABLE must fail the run, not emit a JSON tier — \
         stdout={stdout} stderr={stderr}"
    );
    assert!(
        stderr.contains("attention_block_flash") && stderr.contains("adamw_step_fused"),
        "the failure must name the required ALLOFF set so a caller can distinguish a dropped \
         env var from every other failure mode — stderr={stderr}"
    );
    // Not a datum: an INVALID run must never print the report shape at all.
    assert!(
        !stdout.contains("finetune_run"),
        "an INVALID run printed a JSON tier on stdout — stdout={stdout}"
    );
}

/// The PARTIAL-disable variant of the safety property: only ONE of the two
/// required ALLOFF op keys is named. This is the real-world failure mode a
/// pure "was anything disabled at all" check would miss — `alloff` is a
/// SPECIFIC two-op set (CONTRACT Frame), not "at least one kernel disabled".
#[test]
fn alloff_with_only_one_of_the_two_required_ops_invalidates_the_run() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "alloff")
        .env("JAMMI_KERNELS_DISABLE", "attention_block_flash")
        .output()
        .expect("spawn jammi-bench finetune-run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "--arm alloff with only one of the two required ops disabled must fail — \
         stdout={stdout} stderr={stderr}"
    );
    assert!(!stdout.contains("finetune_run"), "stdout={stdout}");
}

/// The OVER-disable variant: BOTH required ops plus an extra, unrelated
/// real op. `alloff` names an EXACT set (CONTRACT Frame's
/// `ALLOFF=attention_block_flash,adamw_step_fused`), so a superset must
/// also be refused — a merger pairing this leg against a genuine `alloff`
/// leg elsewhere would otherwise silently compare runs under different
/// forced-eager conditions.
#[test]
fn alloff_with_an_extra_op_beyond_the_required_two_invalidates_the_run() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "alloff")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "attention_block_flash,adamw_step_fused,layer_norm_fused",
        )
        .output()
        .expect("spawn jammi-bench finetune-run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "--arm alloff with a THIRD op disabled beyond the exact required set must fail — \
         stdout={stdout} stderr={stderr}"
    );
    assert!(!stdout.contains("finetune_run"), "stdout={stdout}");
}

/// The positive control: `--arm alloff` with `JAMMI_KERNELS_DISABLE`
/// resolving to EXACTLY the required two-op set (reordered, to prove the
/// check is set-equality, not string-equality) must succeed, and the
/// report's `arm`/`kernels_disabled_requested` must reflect it. Without
/// this control, the three failing tests above would not by themselves
/// prove the check ever lets a GENUINE alloff leg through — only that it
/// rejects bad ones.
#[test]
fn alloff_with_exactly_the_required_two_ops_reordered_succeeds() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "alloff")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "adamw_step_fused,attention_block_flash",
        )
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        output.status.success(),
        "--arm alloff with exactly the required two ops (reordered) must succeed — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_run"];
    assert_eq!(tier["arm"], serde_json::json!("alloff"), "tier={tier}");
    assert_eq!(
        tier["kernels_disabled_requested"],
        serde_json::json!(["adamw_step_fused", "attention_block_flash"]),
        "tier={tier}"
    );
}

/// `--arm fused` makes no ALLOFF claim at all (this check is
/// `Arm::Alloff`-only — see `finetune_run::run`'s own doc: "the fused arm
/// makes no such claim"), so it must succeed regardless of whether
/// `JAMMI_KERNELS_DISABLE` happens to be unset. The negative-control half of
/// this file's whole coverage: proves the check above is actually gated on
/// `--arm alloff`, not firing unconditionally on every run.
#[test]
fn fused_arm_never_hard_errors_on_a_missing_kernels_disable_env_var() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env_remove("JAMMI_KERNELS_DISABLE")
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        output.status.success(),
        "--arm fused must never require JAMMI_KERNELS_DISABLE — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_run"];
    assert_eq!(tier["arm"], serde_json::json!("fused"), "tier={tier}");
}
