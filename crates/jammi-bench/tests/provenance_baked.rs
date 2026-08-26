//! Unification contract C1.6 — the mechanical proof that this binary's
//! build-time identity is genuinely baked at COMPILE time, never read at
//! run time. Driven through the REAL `jammi-bench` CLI entry point (the same
//! `env!("CARGO_BIN_EXE_jammi-bench")` pattern
//! `finetune_step_kernel_disable.rs:42` already establishes), not
//! `report::Provenance::baked()` called in-process — a Rust-level call
//! would prove nothing about the compiled BINARY's own baked env vars.
//!
//! Deliberately does NOT assert `build_sha` is 40-hex, and does NOT assert
//! it is not `"unknown"` — either shape (a genuine git worktree sha, a
//! `-dirty` suffix, or `"unknown"` on a git-less checkout) is a legitimate
//! outcome for a dev tree; this suite proves the SHAPE and the run-time
//! inertness, never a specific value (contract C1.2's own pin: "never
//! asserts 40-hex, never asserts not-unknown").

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

fn provenance_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.arg("provenance");
    cmd
}

fn finetune_step_command(model_dir: &Path) -> Command {
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

/// The four fields `report::Provenance` serializes, and the shape check
/// each one must pass — never a specific value.
fn assert_provenance_shape(provenance: &serde_json::Value) {
    let build_sha = provenance["build_sha"]
        .as_str()
        .expect("build_sha present and a string");
    let shape_ok = build_sha == "unknown"
        || (build_sha.len() == 40 && build_sha.bytes().all(|b| b.is_ascii_hexdigit()))
        || (build_sha.len() == 46
            && build_sha.ends_with("-dirty")
            && build_sha[..40].bytes().all(|b| b.is_ascii_hexdigit()));
    assert!(
        shape_ok,
        "build_sha {build_sha:?} matches none of unknown / 40-hex / 40-hex-dirty"
    );
    assert!(
        provenance["target"].as_str().is_some_and(|s| !s.is_empty()),
        "target must be a non-empty string; provenance={provenance}"
    );
    assert!(
        provenance["profile"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "profile must be a non-empty string; provenance={provenance}"
    );
    let build_features = provenance["build_features"]
        .as_array()
        .expect("build_features present and an array");
    assert!(
        build_features.iter().all(|v| v.is_string()),
        "every build_features entry must be a string; provenance={provenance}"
    );
    assert_eq!(
        provenance["report_schema_version"].as_u64(),
        Some(2),
        "report_schema_version must be 2; provenance={provenance}"
    );
}

/// C1.6(a): `jammi-bench provenance` prints a standalone JSON object with
/// the shape above.
#[test]
fn provenance_subcommand_prints_the_baked_identity() {
    let output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance");
    assert!(
        output.status.success(),
        "provenance subcommand must never fail — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let provenance: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON from `jammi-bench provenance`: {e}\n{stdout}"));
    assert_provenance_shape(&provenance);
}

/// C1.6(b): the SAME object (byte-for-byte, field for field) appears
/// verbatim under `report.provenance` of a real CPU `finetune-step` run —
/// one baked identity, shared by every emit site (`Report::new`), not two
/// independently-computed copies that could drift.
#[test]
fn report_carries_provenance() {
    let dir = model_dir();
    assert!(
        dir.join("config.json").exists(),
        "fixture missing: {}",
        dir.display()
    );

    let standalone_output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance");
    assert!(standalone_output.status.success());
    let standalone: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&standalone_output.stdout))
            .expect("valid JSON from `jammi-bench provenance`");

    let report_output = finetune_step_command(&dir)
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        report_output.status.success(),
        "finetune-step failed — stderr={}",
        String::from_utf8_lossy(&report_output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&report_output.stdout))
            .expect("valid JSON from `jammi-bench finetune-step`");
    let embedded = report
        .get("provenance")
        .expect("report.provenance present — RED at base (key absent)");
    assert_provenance_shape(embedded);
    assert_eq!(
        embedded, &standalone,
        "report.provenance must be byte-for-byte the SAME object `jammi-bench provenance` \
         prints standalone — a mismatch here means two independently-computed copies exist"
    );
}

/// C1.6(c): run-time inertness. Re-running the ALREADY-COMPILED binary with
/// a DIFFERENT `JAMMI_BUILD_SHA` in the process environment, from a FRESH
/// `cwd` (a brand-new `git init` temp repo, never this workspace's own
/// `.git`), yields a byte-identical `provenance` object. Proves neither the
/// environment nor the filesystem is consulted at run time — every field
/// baked by `build.rs` is a compile-time `env!()` literal (the `tip_sha()`
/// defect this replaces DID re-read `git` at run time; this test is what a
/// regression back to that shape would fail).
#[test]
fn runtime_env_and_cwd_are_inert() {
    let baseline_output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance (baseline)");
    assert!(baseline_output.status.success());
    let baseline: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&baseline_output.stdout))
            .expect("valid JSON from the baseline run");

    let tmp = tempfile::tempdir().expect("create fresh temp dir");
    let init = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(tmp.path())
        .output()
        .expect("spawn git init");
    assert!(
        init.status.success(),
        "git init in the fresh temp dir failed — stderr={}",
        String::from_utf8_lossy(&init.stderr)
    );

    // A 40-hex sha that is NOT this workspace's own HEAD (all `f`s can
    // never collide with a real sha256-derived... well, sha1 hex digest by
    // construction being astronomically unlikely, and is trivially
    // distinguishable from any plausible real value by eye in a failure
    // message).
    let injected_output = provenance_command()
        .env("JAMMI_BUILD_SHA", "f".repeat(40))
        .current_dir(tmp.path())
        .output()
        .expect("spawn jammi-bench provenance (env+cwd injected)");
    assert!(
        injected_output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&injected_output.stderr)
    );
    let injected: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&injected_output.stdout))
            .expect("valid JSON from the env+cwd-injected run");

    assert_eq!(
        baseline, injected,
        "a run-time JAMMI_BUILD_SHA and a fresh cwd must change NOTHING about this ALREADY-\
         COMPILED binary's reported provenance — every field is a compile-time env!() literal, \
         never a run-time std::env::var/git read"
    );
}
