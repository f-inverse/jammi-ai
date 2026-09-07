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

/// `--arm fused` with `JAMMI_KERNELS_DISABLE` genuinely UNSET (never merely
/// empty-string) and no `--expect-kernels-disabled` claim must succeed — the
/// ordinary, unlabeled fused leg with nothing ambient to contaminate it. The
/// negative-control half of this file's ALLOFF coverage: proves the
/// `Arm::Alloff`-only check above is gated on `--arm alloff`, not firing
/// unconditionally on every run. See
/// `fused_arm_with_no_flag_refuses_an_ambient_kernels_disable_env_var` below
/// for the companion case this control makes non-vacuous: unlike THAT case,
/// here there is genuinely nothing disabled to name.
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
    assert_eq!(
        tier["kernels_disabled_expected"],
        serde_json::json!([]),
        "an unclaimed leg records the empty claim, never omits the field: tier={tier}"
    );
}

/// Unit-467 adversarial audit finding F1: an unlabeled `--arm fused` leg
/// (no `--expect-kernels-disabled`) with a NON-empty ambient
/// `JAMMI_KERNELS_DISABLE` — e.g. a hand-run `alloff` leg's env var left
/// exported in the operator's shell — must refuse at START, naming the
/// offending key, rather than silently running as a contaminated "fused"
/// datum. Before this check existed, this exact invocation succeeded and
/// emitted a report claiming `arm: "fused"` while a real op was disabled the
/// whole time.
#[test]
fn fused_arm_with_no_flag_refuses_an_ambient_kernels_disable_env_var() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env("JAMMI_KERNELS_DISABLE", "gelu_erf_fused")
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "--arm fused with no --expect-kernels-disabled and a non-empty ambient \
         JAMMI_KERNELS_DISABLE must refuse — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("gelu_erf_fused") && stderr.contains("INVALID run"),
        "the refusal must name the ambient key and say the leg is invalid: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "an INVALID leg must emit no report at all"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// `--expect-kernels-disabled` (issue #421 P1-b(i)) — the eager-twin proof
// a forced-eager profile leg needs. Ported from `finetune-step`'s own flag
// (`finetune_step_kernel_disable.rs`), with THREE checks instead of one:
// (1) at START this field must equal the process's real
// `JAMMI_KERNELS_DISABLE` EXACTLY (finding F1: an earlier revision here was
// a subset check; the "legitimate combined leg on top of `--arm alloff`"
// premise it rested on cannot occur, since `--arm alloff`'s own arm-level
// check forces an exact set), (2) at the END `unmatched_disables()` must be
// empty, (3) at the END every named key's `fused` dispatch DELTA over the
// measured epoch loop must be 0. Each case below drives exactly one of the
// three, plus the equality semantics and the unchanged-by-default control.
//
// `lora_linear_fused` is the key used throughout: `jammi-lora`'s LoRA
// linear calls `admit` once per training forward on EVERY architecture
// including this CPU tiny-BERT fixture (`lora_linear.rs`'s own admit call),
// so it is guaranteed to reach `admit` on this leg — which is what makes
// check (2) non-vacuous here (a key that never fires would trip it).
// ─────────────────────────────────────────────────────────────────────────

/// Reads the emitted report's `finetune_run` tier, failing loudly (never
/// silently returning an empty object) when the run did not emit one.
fn tier_of(stdout: &str) -> serde_json::Value {
    let report: serde_json::Value = serde_json::from_str(stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    report["tiers"]["finetune_run"].clone()
}

/// The happy path, with the POSITIVE PROOF attached: the named key is in
/// the env, it actually fired, and the run's own `lora_linear` counters
/// show the eager arm really ran (`fused == 0`, `eager > 0`). Asserting
/// `fused == 0` alone would pass on a leg where NOTHING dispatched at all;
/// requiring `eager > 0` in the same breath is what makes this a proof the
/// forced-eager arm executed rather than a proof it was merely not fused.
#[test]
fn expect_kernels_disabled_succeeds_and_proves_the_eager_arm_actually_ran() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env("JAMMI_KERNELS_DISABLE", "lora_linear_fused")
        .args(["--expect-kernels-disabled", "lora_linear_fused"])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        output.status.success(),
        "a satisfied --expect-kernels-disabled must not refuse — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tier = tier_of(&String::from_utf8_lossy(&output.stdout));
    assert_eq!(
        tier["kernels_disabled_expected"],
        serde_json::json!(["lora_linear_fused"]),
        "the caller's claim must be RECORDED as provenance: tier={tier}"
    );
    assert_eq!(
        tier["lora_linear_fused_dispatches"],
        serde_json::json!(0),
        "a disabled key must not have dispatched fused: tier={tier}"
    );
    let eager = tier["lora_linear_eager_dispatches"]
        .as_u64()
        .expect("lora_linear_eager_dispatches must be a number");
    assert!(
        eager > 0,
        "the forced-eager arm must have actually EXECUTED (eager > 0), not merely failed to \
         dispatch fused: tier={tier}"
    );
}

/// Check (1): the "env var was dropped" failure mode — the flag names a key
/// and `JAMMI_KERNELS_DISABLE` is not set at all. Must refuse, naming the
/// key, and emit NO report (an INVALID leg is never a datum).
#[test]
fn expect_kernels_disabled_refuses_when_the_env_var_was_dropped() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env_remove("JAMMI_KERNELS_DISABLE")
        .args(["--expect-kernels-disabled", "lora_linear_fused"])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "a dropped JAMMI_KERNELS_DISABLE must hard-fail when --expect-kernels-disabled named a \
         key — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("lora_linear_fused") && stderr.contains("INVALID run"),
        "the refusal must name the missing key and say the leg is invalid: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "an INVALID leg must emit no report at all"
    );
}

/// Check (1) again, on the MISTYPED-key half: the env var IS set, but to a
/// different key than the one claimed. Distinguishes "dropped" from "wrong"
/// — a check that only looked at emptiness would pass here.
#[test]
fn expect_kernels_disabled_refuses_when_the_env_names_a_different_key() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env("JAMMI_KERNELS_DISABLE", "layer_norm_fused")
        .args(["--expect-kernels-disabled", "lora_linear_fused"])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "a JAMMI_KERNELS_DISABLE naming a DIFFERENT key must hard-fail — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("lora_linear_fused"),
        "the refusal must name the key that was claimed but absent: {stderr}"
    );
}

/// Finding F1's equality fix: `--expect-kernels-disabled` naming ONE key
/// while `JAMMI_KERNELS_DISABLE` resolves to that key PLUS an extra,
/// unrelated one must now REFUSE — this used to be accepted (a SUBSET
/// check, on the premise that an `--arm alloff` leg legitimately combines
/// its two pinned keys with a claimed chain key), but that premise is false
/// (`--arm alloff`'s own arm-level check forces an EXACT set, so the
/// "combined leg" can never exist) and the weaker check let an ambient
/// extra key through undetected — exactly finding F1's contamination shape,
/// just with the flag present instead of absent.
#[test]
fn expect_kernels_disabled_refuses_an_extra_env_key_beyond_the_claim() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "lora_linear_fused,layer_norm_fused",
        )
        .args(["--expect-kernels-disabled", "lora_linear_fused"])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "a SUPERSET JAMMI_KERNELS_DISABLE must now be refused (set-equality semantics) — \
         stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("lora_linear_fused") && stderr.contains("layer_norm_fused"),
        "the refusal must name both the claimed key and the extra ambient one: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "an INVALID leg must emit no report at all"
    );
}

/// Check (2): a `JAMMI_KERNELS_DISABLE` entry that never disables a live
/// dispatch is a TYPO, not evidence the eager arm ran. Both entries are
/// named on `--expect-kernels-disabled` (so check (1)'s set EQUALITY passes
/// cleanly — this case is about check (2), not a repeat of the equality
/// tests above), and `lora_linear_fused` genuinely fires, so checks (1) and
/// (3) both pass — only the unmatched sibling entry can fail this run,
/// which is exactly the hole this check exists to close.
#[test]
fn expect_kernels_disabled_refuses_an_unmatched_disable_entry() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "lora_linear_fused,not_a_real_op_key_at_all",
        )
        .args([
            "--expect-kernels-disabled",
            "lora_linear_fused,not_a_real_op_key_at_all",
        ])
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "an op key that never disabled a live dispatch must invalidate the run — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not_a_real_op_key_at_all") && stderr.contains("never disabled"),
        "the refusal must name the entry that never fired: {stderr}"
    );
}

/// Finding F1's closed gap, in its most literal shape: WITHOUT
/// `--expect-kernels-disabled`, an ambient `JAMMI_KERNELS_DISABLE` naming
/// even a BOGUS, non-existent op key (`disabled_ops_requested()` is a raw
/// parse of the env var — it does not validate keys against any known op
/// list) must now be REFUSED at START, never silently accepted. Before
/// finding F1's fix this exact invocation succeeded (this tier never read
/// `disabled_ops_requested()` at all for an unlabeled `--arm fused` leg,
/// only `unmatched_disables()`, and only when `--expect-kernels-disabled`
/// was `Some`) — the pressure-tester's whole point: an ambient env var
/// contaminates a leg regardless of whether the disabled key names a real
/// kernel or a typo, and the check must not depend on the key being
/// "real" to catch it.
#[test]
fn without_the_flag_an_ambient_disable_entry_is_now_refused_even_when_bogus() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "fused")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "lora_linear_fused,not_a_real_op_key_at_all",
        )
        .output()
        .expect("spawn jammi-bench finetune-run");

    assert!(
        !output.status.success(),
        "a run making no --expect-kernels-disabled claim must now refuse a non-empty ambient \
         JAMMI_KERNELS_DISABLE (finding F1) — stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("lora_linear_fused")
            && stderr.contains("not_a_real_op_key_at_all")
            && stderr.contains("INVALID run"),
        "the refusal must name both ambient keys and say the leg is invalid: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "an INVALID leg must emit no report at all"
    );
}
