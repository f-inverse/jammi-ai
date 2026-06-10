//! CLI integration tests for `jammi mutable`.
//!
//! Each test spawns a hermetic `jammi-server` (default SQLite catalog +
//! in-memory broker) and drives the `jammi` CLI against it with `--target`. The
//! shipped fixtures under `tests/fixtures/cp9/` provide tenant-neutral schemas;
//! tests reach for `feature_schema.json` directly so a regression in either the
//! fixture or the CLI surfaces immediately.

use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::server_harness::TestServer;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("tests/fixtures/cp9")
        .join(name)
}

fn write_schema(tmp: &TempDir, body: &str) -> PathBuf {
    let p = tmp.path().join("schema.json");
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn cli_mutable_create_list_drop_happy_path() {
    let server = TestServer::spawn();
    let schema = fixture("feature_schema.json");
    assert!(
        schema.exists(),
        "expected fixture {schema:?} to exist (run from workspace)"
    );

    server
        .cli()
        .args([
            "mutable",
            "create",
            "--name",
            "feature_store_dimensions",
            "--schema",
            schema.to_str().unwrap(),
            "--primary-key",
            "feature_id,effective_from",
            "--index",
            "name=idx_active,columns=feature_id+effective_to,unique=false",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("registered"))
        .stdout(predicates::str::contains("idx_active"));

    let out = server
        .cli()
        .args(["mutable", "list"])
        .output()
        .expect("run mutable list");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("feature_store_dimensions"),
        "list missing table:\n{stdout}"
    );
    assert!(
        stdout.contains("feature_id,effective_from"),
        "list missing primary key:\n{stdout}"
    );

    server
        .cli()
        .args(["mutable", "drop", "feature_store_dimensions"])
        .assert()
        .success()
        .stdout(predicates::str::contains("dropped"));

    let out = server
        .cli()
        .args(["mutable", "list"])
        .output()
        .expect("run mutable list after drop");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("feature_store_dimensions"),
        "table still present after drop:\n{stdout}"
    );
}

#[test]
fn cli_mutable_create_rejects_reserved_tenant_id_column() {
    let server = TestServer::spawn();
    let schema_dir = TempDir::new().expect("schema tempdir");
    let schema = write_schema(
        &schema_dir,
        r#"[
            {"name":"feature_id","type":"Int64","nullable":false},
            {"name":"tenant_id","type":"Utf8","nullable":false}
        ]"#,
    );
    let out = server
        .cli()
        .args([
            "mutable",
            "create",
            "--name",
            "tries_to_smuggle_tenant",
            "--schema",
            schema.to_str().unwrap(),
            "--primary-key",
            "feature_id",
        ])
        .output()
        .expect("run create");
    assert!(!out.status.success(), "must reject reserved column name");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("reserved") && stderr.contains("tenant_id"),
        "stderr must explain reserved column:\n{stderr}"
    );
}

#[test]
fn cli_mutable_create_rejects_missing_primary_key() {
    let server = TestServer::spawn();
    let schema = fixture("feature_schema.json");
    let out = server
        .cli()
        .args([
            "mutable",
            "create",
            "--name",
            "missing_pk",
            "--schema",
            schema.to_str().unwrap(),
            "--primary-key",
            "",
        ])
        .output()
        .expect("run create");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("at least one column"),
        "stderr must require pk column:\n{stderr}"
    );
}

#[test]
fn cli_mutable_drop_rejects_unknown_table() {
    let server = TestServer::spawn();
    let out = server
        .cli()
        .args(["mutable", "drop", "never_registered"])
        .output()
        .expect("run drop");
    assert!(!out.status.success(), "drop of unknown table must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("never_registered") || stderr.to_lowercase().contains("not found"),
        "stderr must explain missing table:\n{stderr}"
    );
}

#[test]
fn cli_mutable_create_rejects_unknown_schema_type() {
    let server = TestServer::spawn();
    let schema_dir = TempDir::new().expect("schema tempdir");
    let schema = write_schema(
        &schema_dir,
        r#"[{"name":"x","type":"Decimal","nullable":false}]"#,
    );
    let out = server
        .cli()
        .args([
            "mutable",
            "create",
            "--name",
            "bad_schema",
            "--schema",
            schema.to_str().unwrap(),
            "--primary-key",
            "x",
        ])
        .output()
        .expect("run create");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Decimal"),
        "stderr must mention unsupported type:\n{stderr}"
    );
}
