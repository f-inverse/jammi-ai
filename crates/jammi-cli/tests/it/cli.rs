//! CLI integration test for tenant-scoped `sources list` over the wire.
//!
//! The `jammi` CLI is a strict gRPC client, so this spawns a hermetic
//! `jammi-server` (default SQLite catalog + in-memory broker) and drives the
//! CLI against it with `--target`. Bootstrap (registering sources) goes through
//! the CLI's own `sources add` subcommand, keeping the test purely at the CLI
//! boundary. The intent is unchanged from the in-process version: confirm that
//! a `--tenant` binding receives a disjoint view of the `sources` catalog —
//! now end-to-end over the gRPC wire, where the CLI binds its tenant and the
//! server scopes the read to it.

use std::path::{Path, PathBuf};

use crate::server_harness::TestServer;

const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b";

fn workspace_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("tests/fixtures")
        .join(name)
}

fn add_source(server: &TestServer, tenant: &str, name: &str, fixture: &Path) {
    let path = fixture.to_str().expect("fixture path is utf-8");
    server
        .cli()
        .args(["--tenant", tenant])
        .args(["sources", "add", name, "--url", path, "--format", "parquet"])
        .assert()
        .success();
}

/// Every tenant binding receives a disjoint view of the `sources` catalog at
/// the CLI surface. Two sources are registered, one per tenant, then
/// `sources list` is invoked with each tenant binding (and once unscoped) to
/// confirm the server's tenant predicate-injection is observable end-to-end
/// through the CLI binary talking to a real server.
#[test]
fn cli_sources_list_filters_by_tenant_binding() {
    let fixture = workspace_fixture("patents.parquet");
    assert!(
        fixture.exists(),
        "expected fixture {fixture:?} to exist (run from workspace)"
    );

    let server = TestServer::spawn();

    add_source(&server, TENANT_A, "src_a", &fixture);
    add_source(&server, TENANT_B, "src_b", &fixture);

    let out_a = server
        .cli()
        .args(["--tenant", TENANT_A, "sources", "list"])
        .output()
        .expect("run sources list for tenant A");
    assert!(
        out_a.status.success(),
        "tenant A `sources list` failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out_a.stdout),
        String::from_utf8_lossy(&out_a.stderr),
    );
    let stdout_a = String::from_utf8_lossy(&out_a.stdout);
    assert!(
        stdout_a.contains("src_a"),
        "tenant A must see its own source 'src_a'; got:\n{stdout_a}"
    );
    assert!(
        !stdout_a.contains("src_b"),
        "tenant A must NOT see tenant B's source 'src_b'; got:\n{stdout_a}"
    );

    let out_b = server
        .cli()
        .args(["--tenant", TENANT_B, "sources", "list"])
        .output()
        .expect("run sources list for tenant B");
    assert!(
        out_b.status.success(),
        "tenant B `sources list` failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out_b.stdout),
        String::from_utf8_lossy(&out_b.stderr),
    );
    let stdout_b = String::from_utf8_lossy(&out_b.stdout);
    assert!(
        stdout_b.contains("src_b"),
        "tenant B must see its own source 'src_b'; got:\n{stdout_b}"
    );
    assert!(
        !stdout_b.contains("src_a"),
        "tenant B must NOT see tenant A's source 'src_a'; got:\n{stdout_b}"
    );

    // The unscoped binding sees only rows with `tenant_id IS NULL`. Both seeded
    // sources are tenant-bound, so the listing must be empty.
    let out_u = server
        .cli()
        .args(["sources", "list"])
        .output()
        .expect("run sources list unscoped");
    assert!(
        out_u.status.success(),
        "unscoped `sources list` failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out_u.stdout),
        String::from_utf8_lossy(&out_u.stderr),
    );
    let stdout_u = String::from_utf8_lossy(&out_u.stdout);
    assert!(
        stdout_u.contains("No sources registered."),
        "unscoped binding must see no tenant-bound rows; got:\n{stdout_u}"
    );
}
