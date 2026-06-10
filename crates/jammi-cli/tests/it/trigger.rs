//! CLI integration tests for `jammi trigger`.
//!
//! The strict client CLI exposes only the control-plane topic-admin verbs —
//! `register` / `drop` / `list`. The data-plane `publish` / `subscribe` compute
//! verbs are intentionally not on the CLI. These tests drive the admin surface
//! end-to-end through the binary boundary against a hermetic `jammi-server`
//! (default SQLite catalog + in-memory broker) reached over `--target`, and pin
//! the absence of the data-plane subcommands.

use crate::server_harness::TestServer;

fn register_topic(server: &TestServer, name: &str) {
    server
        .cli()
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
fn cli_trigger_register_then_list_then_drop() {
    let server = TestServer::spawn();
    register_topic(&server, "events.changes");

    // `list` shows the registered topic with its columns.
    server
        .cli()
        .args(["trigger", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("events.changes"));

    // `drop` removes it; a subsequent `list` no longer shows it.
    server
        .cli()
        .args(["trigger", "drop", "--name", "events.changes"])
        .assert()
        .success()
        .stdout(predicates::str::contains("dropped"));

    let out = server
        .cli()
        .args(["trigger", "list"])
        .output()
        .expect("run list");
    assert!(out.status.success(), "list after drop must succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("events.changes"),
        "dropped topic must not appear in `trigger list`:\n{stdout}"
    );
}

#[test]
fn cli_trigger_drop_missing_errors_without_if_exists() {
    let server = TestServer::spawn();

    let out = server
        .cli()
        .args(["trigger", "drop", "--name", "no.such.topic"])
        .output()
        .expect("run drop");
    assert!(
        !out.status.success(),
        "dropping a missing topic without --if-exists must fail"
    );
}

#[test]
fn cli_trigger_drop_missing_with_if_exists_is_noop() {
    let server = TestServer::spawn();

    server
        .cli()
        .args(["trigger", "drop", "--name", "no.such.topic", "--if-exists"])
        .assert()
        .success();
}

#[test]
fn cli_trigger_has_no_data_plane_subcommands() {
    let server = TestServer::spawn();
    register_topic(&server, "events.changes");

    for verb in ["publish", "subscribe"] {
        let out = server
            .cli()
            .args(["trigger", verb, "--topic", "events.changes"])
            .output()
            .expect("run removed trigger subcommand");
        assert!(
            !out.status.success(),
            "`trigger {verb}` must not exist on the strict client CLI"
        );
    }
}
