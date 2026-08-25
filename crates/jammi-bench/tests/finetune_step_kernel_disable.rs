//! Contract K-aux, cell 10 (the safety property) and cell 6 (the
//! load-bearing cell), driven through the REAL `jammi-bench finetune-step`
//! CLI entry point — not `jammi_kernels::admission`'s functions called
//! directly with literals.
//!
//! Each case spawns the compiled `jammi-bench` binary as a fresh child
//! PROCESS (`env!("CARGO_BIN_EXE_jammi-bench")`), rather than calling
//! `finetune_step::run` in-process: `jammi_kernels::admission::disabled_ops`
//! (and `admission_mode`, for `JAMMI_KERNELS_STRICT`) memoize into a
//! process-wide `OnceLock` read once per process — see that module's doc.
//! An in-process `std::env::set_var` test would race every OTHER test in
//! this crate's shared test binary for who reads that `OnceLock` first
//! (exactly the hazard
//! `jammi_kernels::admission::tests::admission_mode_defaults_to_fallback_without_the_env_var`'s
//! own doc names). A fresh child process side-steps it entirely: a fresh
//! `OnceLock`, guaranteed, every time.
//!
//! The fixture is `jammi-encoders`' own committed `head_dim == 64`
//! checkpoint (`crates/jammi-encoders/tests/fixtures/tiny_modernbert_head64`:
//! hidden 64, 1 head, 2 layers, layer 0 global / layer 1 local) — reused
//! rather than duplicated so this crate's fixture stays a single committed
//! source of truth. `jammi_encoders::modernbert`'s own
//! `forward_hidden_reaches_the_fused_attention_block_on_a_head_dim_64_checkpoint`
//! test already proves `layer_norm_fused`'s domain holds trivially on it
//! (CPU, F32, contiguous, `hidden = 64`) in training mode, which is what
//! makes it usable here as the "a real op WILL dispatch" fixture for cell 6.

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

/// A minimal, fast `finetune-step` invocation against `model_dir` — small
/// enough (`batch=2, seq=6, steps=1, warmup=0`) to run as a `cargo test`
/// case rather than a bench, while still driving one full LoRA training
/// step end to end (encoder forward × 3, triplet loss, backward, one
/// `AdamW` step) through the real production call graph.
fn base_command(model_dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args([
        "finetune-step",
        "--model-dir",
        &model_dir.to_string_lossy(),
        "--batch",
        "2",
        "--seq",
        "6",
        "--steps",
        "1",
        "--warmup",
        "0",
        "--lora-rank",
        "2",
        "--target-modules",
        "Wqkv,Wo",
    ]);
    cmd
}

/// Cell 10 — the safety property. A `JAMMI_KERNELS_DISABLE` entry that
/// never disabled a single live dispatch this run (a typo, or a real-but-
/// dead registry name — see `jammi_kernels::admission`'s module doc for
/// why `"lora_dropout"`/`"lora_epilogue"` qualify too) must fail the run,
/// never emit a JSON report as if the forced-eager arm had worked.
#[test]
fn kernel_disable_typo_invalidates_the_run() {
    let dir = model_dir();
    assert!(
        dir.join("config.json").exists(),
        "fixture missing: {}",
        dir.display()
    );

    let output = base_command(&dir)
        .env("JAMMI_KERNELS_DISABLE", "not_a_real_kernel_op")
        .env_remove("JAMMI_KERNELS_STRICT")
        .output()
        .expect("spawn jammi-bench finetune-step");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "an unmatched JAMMI_KERNELS_DISABLE entry must fail the run, not emit a JSON tier — \
         stdout={stdout} stderr={stderr}"
    );
    assert!(
        stderr.contains("not_a_real_kernel_op"),
        "the failure must name the unmatched entry so a caller can distinguish a typo from \
         every other failure mode — stderr={stderr}"
    );
    // Not a datum: an INVALID run must never print the report shape at all.
    assert!(
        !stdout.contains("finetune_step"),
        "an INVALID run printed a JSON tier on stdout — stdout={stdout}"
    );
}

/// The same safety property against a REAL, registered-but-permanently-
/// dead op name (`jammi_kernels::admission`'s module doc: neither
/// `"lora_epilogue"` nor `"lora_dropout"` is ever passed to `admit` in
/// today's call graph — both stand-alone call sites were superseded by
/// the fused LoRA site's single `CustomOp3`). This is the stronger
/// version of the safety-property test: not a synthetic misspelling, but
/// a name that legitimately exists in the process-wide registry
/// (`counters_for` has seen it) and STILL must count as unmatched,
/// because "registered" and "ever reaches `admit`" are different claims.
#[test]
fn kernel_disable_of_a_registered_but_dead_op_name_invalidates_the_run() {
    let dir = model_dir();
    let output = base_command(&dir)
        .env("JAMMI_KERNELS_DISABLE", "lora_dropout")
        .env_remove("JAMMI_KERNELS_STRICT")
        .output()
        .expect("spawn jammi-bench finetune-step");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "lora_dropout never reaches admit today — disabling it must invalidate the run, not \
         emit a JSON tier — stdout={stdout} stderr={stderr}"
    );
    assert!(stderr.contains("lora_dropout"), "stderr={stderr}");
}

/// Cell 6, the load-bearing cell, through the real entry point:
/// `JAMMI_KERNELS_STRICT=1` together with `JAMMI_KERNELS_DISABLE` naming a
/// REAL, dispatching op (`layer_norm_fused`) must still SUCCEED — disable
/// wins over Strict — with the reported dispatch counters showing that op
/// ran eager-only. This is the actual one-build A/B shape the contract
/// exists to enable (`JAMMI_KERNELS_STRICT=1
/// JAMMI_KERNELS_DISABLE=<op-under-test>`, every OTHER op still strictly
/// proven fused).
#[test]
fn strict_mode_disable_forces_layer_norm_eager_and_the_run_still_succeeds() {
    let dir = model_dir();
    let output = base_command(&dir)
        .env("JAMMI_KERNELS_STRICT", "1")
        .env("JAMMI_KERNELS_DISABLE", "layer_norm_fused")
        .output()
        .expect("spawn jammi-bench finetune-step");

    assert!(
        output.status.success(),
        "JAMMI_KERNELS_STRICT=1 with a MATCHED JAMMI_KERNELS_DISABLE entry must still succeed \
         (disable wins over Strict) — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_step"];
    assert_eq!(
        tier["ln_fused_dispatches"].as_u64(),
        Some(0),
        "layer_norm_fused was disabled — it must never dispatch fused; tier={tier}"
    );
    assert!(
        tier["ln_eager_dispatches"].as_u64().unwrap_or(0) > 0,
        "layer_norm_fused was disabled — every call must have taken the eager arm; tier={tier}"
    );
}

/// RED control for the cell-6 test above: WITHOUT `JAMMI_KERNELS_DISABLE`,
/// `JAMMI_KERNELS_STRICT=1` alone against this same fixture/params must
/// leave `layer_norm_fused` dispatching FUSED — its domain holds trivially
/// here (CPU, F32, contiguous, `hidden = 64`; see
/// `jammi_encoders::modernbert`'s
/// `forward_hidden_reaches_the_fused_attention_block_on_a_head_dim_64_checkpoint`
/// doc for the same fixture proven the same way for the attention-block
/// op). This is what makes the eager count in the test above evidence of
/// the DISABLE, not of some unrelated admission failure this fixture
/// always hits — without this control, a bug that broke `layer_norm_fused`
/// admission entirely would make the disable test above pass for the
/// wrong reason.
#[test]
fn strict_mode_without_disable_dispatches_layer_norm_fused_on_this_fixture() {
    let dir = model_dir();
    let output = base_command(&dir)
        .env("JAMMI_KERNELS_STRICT", "1")
        .env_remove("JAMMI_KERNELS_DISABLE")
        .output()
        .expect("spawn jammi-bench finetune-step");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_step"];
    assert!(
        tier["ln_fused_dispatches"].as_u64().unwrap_or(0) > 0,
        "control: layer_norm_fused must dispatch FUSED on this fixture when not disabled — \
         tier={tier}"
    );
    assert_eq!(tier["ln_eager_dispatches"].as_u64(), Some(0));
}

/// Cell 9 — env unset must be byte-identical to today's (pre-K-aux)
/// behaviour: the same command with neither env var set must succeed and
/// still dispatch `layer_norm_fused` fused, exactly like the Strict-only
/// control above (this is the ONE real end-to-end assertion that the
/// disable mechanism is inert by default, on the real CLI, in a fresh
/// process — not just the parser-level unit tests in
/// `jammi_kernels::admission`'s own test module).
#[test]
fn no_env_vars_set_dispatches_layer_norm_fused_unchanged() {
    let dir = model_dir();
    let output = base_command(&dir)
        .env_remove("JAMMI_KERNELS_STRICT")
        .env_remove("JAMMI_KERNELS_DISABLE")
        .output()
        .expect("spawn jammi-bench finetune-step");

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_step"];
    assert!(tier["ln_fused_dispatches"].as_u64().unwrap_or(0) > 0);
}

/// B2 (fix round): proves the CORRECTED nesting
/// `jammi_kernels::admission`'s module doc's "correct one-build A/B for
/// `softmax_last_dim_fused`" section describes, through the real CLI.
/// `attention_block_fused` SUBSUMES both RoPE and softmax on this
/// fixture's training path (`ModernBertAttention::forward_training_attention`),
/// so isolating `softmax_last_dim_fused` alone needs `attention_block_fused`
/// disabled on BOTH legs — differing ONLY in whether
/// `softmax_last_dim_fused` is ALSO named, the isolated variable. A
/// prior, since-corrected version of that doc advertised
/// `JAMMI_KERNELS_DISABLE=softmax_last_dim_fused` alone as a working A/B;
/// on this exact fixture that entry would never fire (`attention_block_fused`
/// admits fused, so `softmax_apply_training` is never reached at all) —
/// `kernel_disable_of_a_registered_but_dead_op_name_invalidates_the_run`'s
/// sibling test class covers that failure mode; this test proves the
/// REPLACEMENT nesting actually produces the fused/eager split it claims.
#[test]
fn softmax_last_dim_fused_nesting_isolates_the_softmax_kernel_through_the_real_cli() {
    let dir = model_dir();

    // Eager leg: `softmax_last_dim_fused` runs eager — both ops disabled.
    let eager_output = base_command(&dir)
        .env("JAMMI_KERNELS_STRICT", "1")
        .env(
            "JAMMI_KERNELS_DISABLE",
            "attention_block_fused,softmax_last_dim_fused",
        )
        .output()
        .expect("spawn jammi-bench finetune-step (eager leg)");
    assert!(
        eager_output.status.success(),
        "eager leg must succeed — stderr={}",
        String::from_utf8_lossy(&eager_output.stderr)
    );
    let eager_stdout = String::from_utf8_lossy(&eager_output.stdout);
    let eager_report: serde_json::Value = serde_json::from_str(&eager_stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{eager_stdout}"));
    let eager_tier = &eager_report["tiers"]["finetune_step"];
    assert_eq!(
        eager_tier["softmax_fused_dispatches"].as_u64(),
        Some(0),
        "softmax_last_dim_fused was disabled — it must never dispatch fused; tier={eager_tier}"
    );
    assert!(
        eager_tier["softmax_eager_dispatches"].as_u64().unwrap_or(0) > 0,
        "softmax_last_dim_fused was disabled — every call must have taken the eager arm; \
         tier={eager_tier}"
    );

    // Fused leg: `softmax_last_dim_fused` runs fused, INSIDE the eager
    // attention composition — only `attention_block_fused` is disabled.
    let fused_output = base_command(&dir)
        .env("JAMMI_KERNELS_STRICT", "1")
        .env("JAMMI_KERNELS_DISABLE", "attention_block_fused")
        .output()
        .expect("spawn jammi-bench finetune-step (fused leg)");
    assert!(
        fused_output.status.success(),
        "fused leg must succeed — stderr={}",
        String::from_utf8_lossy(&fused_output.stderr)
    );
    let fused_stdout = String::from_utf8_lossy(&fused_output.stdout);
    let fused_report: serde_json::Value = serde_json::from_str(&fused_stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{fused_stdout}"));
    let fused_tier = &fused_report["tiers"]["finetune_step"];
    assert!(
        fused_tier["softmax_fused_dispatches"].as_u64().unwrap_or(0) > 0,
        "softmax_last_dim_fused was NOT disabled and its domain holds on this fixture — it must \
         dispatch fused once attention_block_fused alone is disabled; tier={fused_tier}"
    );
    assert_eq!(
        fused_tier["softmax_eager_dispatches"].as_u64(),
        Some(0),
        "tier={fused_tier}"
    );

    // Both legs disabled `attention_block_fused` itself — proof the ONLY
    // variable between the two legs is `softmax_last_dim_fused`.
    assert_eq!(
        eager_tier["attention_block_fused_dispatches"].as_u64(),
        Some(0)
    );
    assert_eq!(
        fused_tier["attention_block_fused_dispatches"].as_u64(),
        Some(0)
    );
}

/// B2's negative control: the PRIOR (since-corrected) flagship doc example
/// advertised `JAMMI_KERNELS_DISABLE=softmax_last_dim_fused` ALONE as a
/// working one-build A/B. On this crate's own committed test fixture — the
/// same one every other test in this file drives — that entry never
/// disables a live dispatch at all (`attention_block_fused` admits fused
/// here too, subsuming it, exactly as on the real ModernBERT-large
/// checkpoint the module doc's committed A100 artifact cites): the run
/// must be reported INVALID, not a datum with a plausible-looking JSON
/// tier. Without this control, `softmax_last_dim_fused_nesting_isolates_
/// the_softmax_kernel_through_the_real_cli` above would not by itself
/// prove the OLD single-entry command was ever broken — it only proves
/// the NEW two-entry nesting works, which the underlying disable
/// mechanism was already capable of before this fix round (B2 was a
/// documentation defect, not a functional one).
#[test]
fn old_doc_softmax_only_disable_is_unmatched_on_this_fixture() {
    let dir = model_dir();
    let output = base_command(&dir)
        .env("JAMMI_KERNELS_DISABLE", "softmax_last_dim_fused")
        .env_remove("JAMMI_KERNELS_STRICT")
        .output()
        .expect("spawn jammi-bench finetune-step");

    assert!(
        !output.status.success(),
        "softmax_last_dim_fused alone must be UNMATCHED on this fixture — attention_block_fused \
         subsumes it — stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("softmax_last_dim_fused"), "stderr={stderr}");
}

/// B3 (execution-provenance hole on the instrument itself): a run whose
/// `JAMMI_KERNELS_DISABLE` was DROPPED — simulated here via a var-NAME
/// typo (`JAMMI_KERNEL_DISABLE`, missing the trailing `S`; an unforwarded
/// ssh/`docker -e` environment looks IDENTICAL to this process) — must
/// still succeed (a var-name typo is not itself a run failure this binary
/// can detect), but the report's `kernels_disabled_requested` /
/// `kernels_disabled_fired` must read `[]`/`[]`, distinguishable from a
/// GENUINE forced-eager run naming the exact same intended op, whose
/// report carries both non-empty and equal. `unmatched_disables()` alone
/// cannot catch this: nothing was requested from THIS process's point of
/// view, so there is nothing to report unmatched — the pair this test
/// checks is what lets a downstream A/B harness compare a report against
/// its OWN recorded intent and refuse a mismatched "eager" leg that
/// silently measured the fused arm instead.
#[test]
fn dropped_disable_var_is_distinguishable_from_a_genuine_forced_eager_run() {
    let dir = model_dir();

    // Simulates the dropped var: a NAME typo means THIS process's real
    // `JAMMI_KERNELS_DISABLE` is unset, even though the caller intended to
    // force `attention_block_fused` eager.
    let dropped_output = base_command(&dir)
        .env_remove("JAMMI_KERNELS_DISABLE")
        .env("JAMMI_KERNEL_DISABLE", "attention_block_fused") // NOTE: missing the `S`
        .output()
        .expect("spawn jammi-bench finetune-step (dropped var)");
    assert!(
        dropped_output.status.success(),
        "a var-NAME typo must not itself fail the run — it looks, from this process, exactly \
         like an ordinary undisabled run; stderr={}",
        String::from_utf8_lossy(&dropped_output.stderr)
    );
    let dropped_stdout = String::from_utf8_lossy(&dropped_output.stdout);
    let dropped_report: serde_json::Value = serde_json::from_str(&dropped_stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{dropped_stdout}"));
    let dropped_tier = &dropped_report["tiers"]["finetune_step"];
    assert_eq!(
        dropped_tier["kernels_disabled_requested"],
        serde_json::json!([]),
        "the dropped var must read as nothing requested — tier={dropped_tier}"
    );
    assert_eq!(
        dropped_tier["kernels_disabled_fired"],
        serde_json::json!([]),
        "tier={dropped_tier}"
    );
    assert!(
        dropped_tier["attention_block_fused_dispatches"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "the caller INTENDED attention_block_fused eager but the var never reached this \
         process — it must dispatch FUSED, the exact false-green this pair exists to catch; \
         tier={dropped_tier}"
    );

    // The genuine forced-eager run, naming the SAME op correctly.
    let genuine_output = base_command(&dir)
        .env("JAMMI_KERNELS_DISABLE", "attention_block_fused")
        .output()
        .expect("spawn jammi-bench finetune-step (genuine)");
    assert!(
        genuine_output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&genuine_output.stderr)
    );
    let genuine_stdout = String::from_utf8_lossy(&genuine_output.stdout);
    let genuine_report: serde_json::Value = serde_json::from_str(&genuine_stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{genuine_stdout}"));
    let genuine_tier = &genuine_report["tiers"]["finetune_step"];
    assert_eq!(
        genuine_tier["kernels_disabled_requested"],
        serde_json::json!(["attention_block_fused"]),
        "tier={genuine_tier}"
    );
    assert_eq!(
        genuine_tier["kernels_disabled_fired"],
        serde_json::json!(["attention_block_fused"]),
        "tier={genuine_tier}"
    );
    assert_eq!(
        genuine_tier["attention_block_fused_dispatches"].as_u64(),
        Some(0),
        "the genuine run must show attention_block_fused NEVER dispatching fused — this pair \
         proves the arm actually ran eager, distinguishing it from the dropped-var run above; \
         tier={genuine_tier}"
    );

    // The two runs' `kernels_disabled_requested`/`_fired` pairs differ —
    // that difference IS the provenance signal a caller uses to refuse a
    // mismatched A/B pair (the dropped-var run's pair can never equal a
    // genuine forced-eager run's for the same intended op).
    assert_ne!(
        dropped_tier["kernels_disabled_requested"],
        genuine_tier["kernels_disabled_requested"]
    );
}
