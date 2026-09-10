//! CLI integration tests for `jammi reconcile` over the wire.
//!
//! Spawns a hermetic `jammi-server` subprocess (the real OSS binary, which
//! wires the shipped `admin_authorizer: None` default) and drives the CLI
//! against it with `--target`, exactly like the other CLI integration
//! suites in this crate.

use crate::server_harness::TestServer;

/// The default `jammi reconcile` invocation (no `--apply`, default
/// `--grace-secs`, no `--all`) prints the label-shaped report — the same
/// rendering shape `jammi status` uses — and, on a freshly-created empty
/// catalog, an unscoped (`_global`) dry run that changed nothing.
#[test]
fn cli_reconcile_prints_the_label_shaped_report() {
    let server = TestServer::spawn();

    server
        .cli()
        .args(["reconcile"])
        .assert()
        .success()
        .stdout(predicates::str::contains("scope:"))
        .stdout(predicates::str::contains("applied:         false"))
        .stdout(predicates::str::contains("rows_failed:"))
        .stdout(predicates::str::contains("orphans:"))
        .stdout(predicates::str::contains("pending:"))
        .stdout(predicates::str::contains("unattributed:"))
        .stdout(predicates::str::contains("bytes_reclaimed:"))
        .stdout(predicates::str::contains("_global"));
}

/// `--all` against a server that never wired an admin authorizer (the OSS
/// binary's shipped default) is refused with `PERMISSION_DENIED`, and this
/// CLI surfaces the server's message verbatim — no special-casing on top of
/// the ordinary `Err(e) => eprintln!("Error: {e}")` path every other verb's
/// failure takes.
#[test]
fn cli_reconcile_all_is_denied_by_default_without_a_wired_authorizer() {
    let server = TestServer::spawn();

    server
        .cli()
        .args(["reconcile", "--all"])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "reconcile --all requires an admin authorizer",
        ));
}

/// `--apply` with a `--grace-secs` shorter than the server's configured lease
/// duration is refused `INVALID_ARGUMENT`, naming both values — proven at the
/// CLI boundary, not just the engine/wire layers underneath it.
#[test]
fn cli_reconcile_apply_with_too_short_grace_secs_is_refused() {
    let server = TestServer::spawn();

    server
        .cli()
        .args(["reconcile", "--apply", "--grace-secs", "0"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("grace"));
}
