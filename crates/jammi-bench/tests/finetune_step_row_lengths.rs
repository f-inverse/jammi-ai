//! `--row-lengths` (contract v4 §1 item 1) through the REAL `jammi-bench
//! finetune-step` CLI entry point — mirroring
//! `finetune_step_kernel_disable.rs`'s own process-isolation discipline (a
//! fresh child PROCESS per case, never `finetune_step::run` in-process,
//! because `jammi_kernels::admission`'s registries are process-wide
//! `OnceLock`s — see that file's own module doc for the full reasoning).
//!
//! CPU-runnable cases: CLI-level parsing and validation (a malformed or
//! out-of-domain `--row-lengths` must fail loud, never silently build a
//! mask that does not mean what the caller intended) and one genuinely
//! padded end-to-end run.
//!
//! CUDA-gated cases (item 2, the A5 padded-shape block-arm VRAM baseline
//! leg, and item 3, the A3 padded loss-sequence flash-vs-block A/B) live in
//! `finetune_step_padded_cuda.rs`, next to this file, gated on
//! [`cuda_available`] — the SAME `#[cfg(feature = "cuda")]` +
//! `Device::new_cuda(0).is_ok()` shape `crates/jammi-ai/tests/gpu_capability/
//! harness.rs`'s own `gpu_available()` uses, so a GPU-less or
//! cuda-feature-off host skips them HONESTLY (a stated per-skip reason,
//! never a silent pass) rather than reading green for having run nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

/// Same shape as `finetune_step_kernel_disable.rs`'s own `base_command` —
/// `batch=2, seq=6, steps=1, warmup=0` — small enough to run as a `cargo
/// test` case while still driving one full LoRA training step end to end.
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

/// A non-numeric token must be refused at CLI-parse time (`main.rs`'s own
/// `t.parse::<usize>()` guard), never silently coerced to `0` or dropped.
#[test]
fn row_lengths_rejects_a_non_numeric_token() {
    let dir = model_dir();
    let output = base_command(&dir)
        .args(["--row-lengths", "3,not-a-number"])
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        !output.status.success(),
        "a non-numeric --row-lengths token must fail the run — stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--row-lengths"),
        "the failure must name the flag — stderr={stderr}"
    );
}

/// A count mismatch (`--batch 2` here) must be refused by
/// `finetune_step::validate_row_lengths`, reached AFTER CLI parsing
/// succeeds — proves the count check fires through the real binary, not
/// just in the crate's own unit tests.
#[test]
fn row_lengths_rejects_a_count_that_does_not_match_batch() {
    let dir = model_dir();
    let output = base_command(&dir)
        .args(["--row-lengths", "3,3,3"]) // 3 entries, --batch is 2
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        !output.status.success(),
        "a row_lengths count mismatch must fail the run — stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("row-lengths") || stderr.contains("row_lengths"),
        "stderr={stderr}"
    );
}

/// A zero-length row must be refused (the B3-padded arm's own guard
/// inventory: `total == 0` is a REFUSAL, never a silently-accepted empty
/// row).
#[test]
fn row_lengths_rejects_a_zero_length_row() {
    let dir = model_dir();
    let output = base_command(&dir)
        .args(["--row-lengths", "0,6"])
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        !output.status.success(),
        "a zero-length row must fail the run — stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A length above `--seq` cannot describe a real row of a `[batch, seq]`
/// mask and must be refused.
#[test]
fn row_lengths_rejects_a_length_above_seq() {
    let dir = model_dir();
    let output = base_command(&dir)
        .args(["--row-lengths", "7,6"]) // --seq is 6
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        !output.status.success(),
        "a length above --seq must fail the run — stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A genuinely padded, valid `--row-lengths` succeeds end to end on CPU
/// (the flash cascade is CUDA-only and simply DECLINES by capability here
/// — the block/eager attention arm still runs the padded mask correctly)
/// and reports the requested lengths back exactly, honestly (never a
/// re-derived or rounded echo).
#[test]
fn row_lengths_padded_batch_runs_end_to_end_on_cpu_and_reports_itself_honestly() {
    let dir = model_dir();
    let output = base_command(&dir)
        .args(["--row-lengths", "3,6"]) // row 0 padded (3 of 6 real), row 1 dense
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"));
    let tier = &report["tiers"]["finetune_step"];
    assert_eq!(
        tier["row_lengths"],
        serde_json::json!([3, 6]),
        "tier={tier}"
    );
    assert_eq!(tier["losses"].as_array().map(|v| v.len()), Some(1));
    assert!(
        tier["losses"][0]
            .as_f64()
            .expect("loss is a number")
            .is_finite(),
        "tier={tier}"
    );
}

/// A4 (dense invariance), through the real CLI: omitting `--row-lengths`
/// entirely reports the dense-leg IDENTITY value `[seq; batch]` — here
/// `[6, 6]` (`--batch 2 --seq 6`).
#[test]
fn row_lengths_absent_reports_the_dense_seq_vector_through_the_real_cli() {
    let dir = model_dir();
    let output = base_command(&dir)
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
    assert_eq!(
        tier["row_lengths"],
        serde_json::json!([6, 6]),
        "tier={tier}"
    );
}
