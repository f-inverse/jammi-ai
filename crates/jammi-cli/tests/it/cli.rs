//! SPEC-03 §12 #7 — `cargo test -p jammi-cli` exercise that runs
//! `jammi --tenant <uuid> sources list` and confirms tenant scoping
//! is observable at the CLI surface.
//!
//! The CLI binary is spawned as a subprocess via `assert_cmd`. The
//! catalog is redirected to a per-test tempdir via the
//! `JAMMI_ARTIFACT_DIR` env var so the test never touches the
//! developer's real Jammi data directory. Bootstrap (registering
//! sources) is done through the CLI's own `sources add` subcommand,
//! keeping the test purely at the CLI boundary — it never reaches
//! into the engine or AI crate's Rust API.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use tempfile::TempDir;

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

fn jammi_cmd(artifact_dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("jammi-cli").expect("jammi-cli binary built");
    cmd.env("JAMMI_ARTIFACT_DIR", artifact_dir)
        .env_remove("JAMMI_CONFIG");
    cmd
}

fn add_source(artifact_dir: &Path, tenant: Option<&str>, name: &str, fixture: &Path) {
    let path = fixture.to_str().expect("fixture path is utf-8");
    let mut cmd = jammi_cmd(artifact_dir);
    if let Some(t) = tenant {
        cmd.args(["--tenant", t]);
    }
    cmd.args([
        "sources", "add", name, "--url", path, "--format", "parquet",
    ])
    .assert()
    .success();
}

/// SPEC-03 §12 #7 — every tenant binding receives a disjoint view of
/// the `sources` catalog table at the CLI surface. Two sources are
/// registered, one per tenant, then `sources list` is invoked with
/// each tenant binding (and once unscoped) to confirm the engine's
/// tenant predicate-injection is observable end-to-end through the
/// CLI binary.
#[test]
fn cli_sources_list_filters_by_tenant_binding() {
    let dir = TempDir::new().expect("tempdir");
    let fixture = workspace_fixture("patents.parquet");
    assert!(
        fixture.exists(),
        "expected fixture {fixture:?} to exist (run from workspace)"
    );

    add_source(dir.path(), Some(TENANT_A), "src_a", &fixture);
    add_source(dir.path(), Some(TENANT_B), "src_b", &fixture);

    let out_a = jammi_cmd(dir.path())
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

    let out_b = jammi_cmd(dir.path())
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

    // The unscoped binding sees only rows with `tenant_id IS NULL`
    // per the analyzer rule. Both seeded sources are tenant-bound,
    // so the listing must be empty.
    let out_u = jammi_cmd(dir.path())
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
