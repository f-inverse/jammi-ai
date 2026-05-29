//! CLI integration tests for `jammi trigger`.
//!
//! Drive the rewritten surface end-to-end through the binary boundary —
//! `--topic` rename, `--json-file` / `--row` mutual exclusion, and
//! `--no-follow` replay. Catalog state lives in a per-test `TempDir` via
//! `JAMMI_ARTIFACT_DIR`; the in-memory broker means everything fits in one
//! process with zero ports.

use std::path::Path;

use assert_cmd::Command;
use tempfile::TempDir;

fn jammi_cmd(artifact_dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("jammi").expect("jammi-cli binary built");
    cmd.env("JAMMI_ARTIFACT_DIR", artifact_dir)
        .env_remove("JAMMI_CONFIG");
    cmd
}

fn register_topic(dir: &Path, name: &str) {
    jammi_cmd(dir)
        .args([
            "trigger",
            "register",
            "--name",
            name,
            "--schema",
            "op:string,ts_ms:int,key:string,after:string:nullable",
        ])
        .assert()
        .success();
}

#[test]
fn cli_trigger_publish_uses_topic_flag() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--row",
            r#"{"op":"c","ts_ms":1,"key":"a"}"#,
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("Published offset"));
}

#[test]
fn cli_trigger_publish_rejects_topic_legacy_alias() {
    // `--name` was the pre-cp9 flag; the rename is intentionally clean.
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--name",
            "events.changes",
            "--row",
            r#"{"op":"c","ts_ms":1,"key":"a"}"#,
        ])
        .output()
        .expect("run publish");
    assert!(
        !out.status.success(),
        "publish must not accept legacy --name flag"
    );
}

#[test]
fn cli_trigger_publish_accepts_json_file() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");
    let payload = dir.path().join("rows.json");
    std::fs::write(
        &payload,
        r#"[
            {"op":"c","ts_ms":1,"key":"a"},
            {"op":"u","ts_ms":2,"key":"a"}
        ]"#,
    )
    .unwrap();

    jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--json-file",
            payload.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("Published offset"));
}

#[test]
fn cli_trigger_publish_rejects_row_and_json_file_together() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");
    let payload = dir.path().join("rows.json");
    std::fs::write(&payload, r#"[{"op":"c","ts_ms":1,"key":"a"}]"#).unwrap();

    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--row",
            r#"{"op":"c","ts_ms":1,"key":"a"}"#,
            "--json-file",
            payload.to_str().unwrap(),
        ])
        .output()
        .expect("run publish");
    assert!(
        !out.status.success(),
        "publish must reject --row and --json-file simultaneously"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("cannot be used")
            || stderr.to_lowercase().contains("conflict")
            || stderr.contains("--row")
            || stderr.contains("--json-file"),
        "stderr must explain the conflict:\n{stderr}"
    );
}

#[test]
fn cli_trigger_publish_rejects_neither_input() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    let out = jammi_cmd(dir.path())
        .args(["trigger", "publish", "--topic", "events.changes"])
        .output()
        .expect("run publish");
    assert!(
        !out.status.success(),
        "publish must require one of --row / --json-file"
    );
}

#[test]
fn cli_trigger_publish_rejects_malformed_json_file() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");
    let payload = dir.path().join("rows.json");
    std::fs::write(&payload, "not json").unwrap();

    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--json-file",
            payload.to_str().unwrap(),
        ])
        .output()
        .expect("run publish");
    assert!(!out.status.success(), "publish must reject malformed json");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("parse json file"),
        "stderr must mention parse failure:\n{stderr}"
    );
}

#[test]
fn cli_trigger_subscribe_uses_topic_flag() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    // Legacy --name flag must be rejected (clean rename).
    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "subscribe",
            "--name",
            "events.changes",
            "--no-follow",
        ])
        .output()
        .expect("run subscribe");
    assert!(
        !out.status.success(),
        "subscribe must not accept legacy --name flag"
    );
}

#[test]
fn cli_trigger_subscribe_no_follow_drains_replay() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    // Publish two batches so the replay window is non-empty.
    jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--row",
            r#"{"op":"c","ts_ms":1,"key":"row-1"}"#,
        ])
        .assert()
        .success();
    jammi_cmd(dir.path())
        .args([
            "trigger",
            "publish",
            "--topic",
            "events.changes",
            "--row",
            r#"{"op":"u","ts_ms":2,"key":"row-1"}"#,
        ])
        .assert()
        .success();

    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "subscribe",
            "--topic",
            "events.changes",
            "--from-offset",
            "0",
            "--no-follow",
        ])
        .output()
        .expect("run subscribe --no-follow");
    assert!(
        out.status.success(),
        "subscribe --no-follow should exit cleanly after draining replay; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"op\":\"c\"") && stdout.contains("\"op\":\"u\""),
        "replay should emit both rows:\n{stdout}"
    );
    // Exactly two delivered rows, exactly two offsets.
    assert_eq!(
        stdout.matches("\"offset\":0").count(),
        1,
        "offset 0 delivered exactly once:\n{stdout}"
    );
    assert_eq!(
        stdout.matches("\"offset\":1").count(),
        1,
        "offset 1 delivered exactly once:\n{stdout}"
    );
}

#[test]
fn cli_trigger_subscribe_no_follow_empty_topic() {
    let dir = TempDir::new().expect("tempdir");
    register_topic(dir.path(), "events.changes");

    let out = jammi_cmd(dir.path())
        .args([
            "trigger",
            "subscribe",
            "--topic",
            "events.changes",
            "--from-offset",
            "0",
            "--no-follow",
        ])
        .output()
        .expect("run subscribe --no-follow on empty topic");
    assert!(
        out.status.success(),
        "subscribe --no-follow on empty topic should exit cleanly; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.is_empty(),
        "empty replay window must print zero rows; got:\n{stdout}"
    );
}
