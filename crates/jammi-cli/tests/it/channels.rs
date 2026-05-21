//! CLI integration tests for `jammi channels`.
//!
//! Each test spawns the `jammi-cli` binary as a subprocess via `assert_cmd`,
//! redirecting catalog and artifact state to a per-test `TempDir` via
//! `JAMMI_ARTIFACT_DIR`. Channels are global (per SPEC-01 §11) so these
//! tests do not bind a tenant.

use std::path::Path;

use assert_cmd::Command;
use tempfile::TempDir;

fn jammi_cmd(artifact_dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("jammi-cli").expect("jammi-cli binary built");
    cmd.env("JAMMI_ARTIFACT_DIR", artifact_dir)
        .env_remove("JAMMI_CONFIG");
    cmd
}

#[test]
fn cli_channels_register_and_list() {
    let dir = TempDir::new().expect("tempdir");

    jammi_cmd(dir.path())
        .args([
            "channels",
            "register",
            "--name",
            "scored_by",
            "--priority",
            "3",
            "--column",
            "ranker:Utf8",
            "--column",
            "rank_score:Float32",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("registered"));

    let out = jammi_cmd(dir.path())
        .args(["channels", "list"])
        .output()
        .expect("run channels list");
    assert!(out.status.success(), "list failed: {out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("scored_by"),
        "list did not surface registered channel:\n{stdout}"
    );
    assert!(
        stdout.contains("ranker:Utf8") && stdout.contains("rank_score:Float32"),
        "list did not render declared columns:\n{stdout}"
    );
}

#[test]
fn cli_channels_register_rejects_unknown_type() {
    let dir = TempDir::new().expect("tempdir");
    let out = jammi_cmd(dir.path())
        .args([
            "channels",
            "register",
            "--name",
            "bad",
            "--priority",
            "1",
            "--column",
            "x:Decimal",
        ])
        .output()
        .expect("run register");
    assert!(!out.status.success(), "register must reject Decimal");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Decimal"),
        "stderr should mention the unknown type:\n{stderr}"
    );
}

#[test]
fn cli_channels_register_rejects_missing_column() {
    let dir = TempDir::new().expect("tempdir");
    let out = jammi_cmd(dir.path())
        .args(["channels", "register", "--name", "empty", "--priority", "1"])
        .output()
        .expect("run register without column");
    assert!(
        !out.status.success(),
        "register must reject no --column flags"
    );
}

#[test]
fn cli_channels_add_column_rejects_retype() {
    let dir = TempDir::new().expect("tempdir");

    jammi_cmd(dir.path())
        .args([
            "channels",
            "register",
            "--name",
            "scored_by",
            "--priority",
            "3",
            "--column",
            "ranker:Utf8",
        ])
        .assert()
        .success();

    // Append-only: re-declaring `ranker` with a different type must fail.
    let out = jammi_cmd(dir.path())
        .args([
            "channels",
            "add-column",
            "scored_by",
            "--column",
            "ranker:Int32",
        ])
        .output()
        .expect("run add-column");
    assert!(!out.status.success(), "retype must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot redeclare") && stderr.contains("Utf8") && stderr.contains("Int32"),
        "stderr must explain the redeclaration conflict:\n{stderr}"
    );
}

#[test]
fn cli_channels_add_column_extends_existing() {
    let dir = TempDir::new().expect("tempdir");

    jammi_cmd(dir.path())
        .args([
            "channels",
            "register",
            "--name",
            "scored_by",
            "--priority",
            "3",
            "--column",
            "ranker:Utf8",
        ])
        .assert()
        .success();

    jammi_cmd(dir.path())
        .args([
            "channels",
            "add-column",
            "scored_by",
            "--column",
            "rank_score:Float32",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("extended"));

    let out = jammi_cmd(dir.path())
        .args(["channels", "list"])
        .output()
        .expect("run channels list");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ranker:Utf8") && stdout.contains("rank_score:Float32"),
        "list must show both declared columns:\n{stdout}"
    );
}
