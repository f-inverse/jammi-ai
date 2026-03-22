mod common;

use jammi_engine::catalog::Catalog;
use jammi_engine::config::JammiConfig;
use jammi_engine::error::JammiError;
use std::path::Path;
use tempfile::tempdir;

// --- Configuration ---

#[test]
fn config_loads_from_toml_file() {
    let config = JammiConfig::load(Some(Path::new(&common::fixture("config_test.toml")))).unwrap();
    assert_eq!(config.engine.batch_size, 4096);
    assert_eq!(config.engine.execution_threads, 2);
    assert_eq!(config.gpu.device, -1);
    assert_eq!(config.inference.batch_size, 8);
    assert_eq!(config.logging.level, "debug");
}

#[test]
fn config_defaults_without_file() {
    let config = JammiConfig::load(None).unwrap();
    assert_eq!(config.engine.batch_size, 8192);
    assert_eq!(config.gpu.device, 0);
    assert_eq!(config.inference.batch_size, 32);
    assert_eq!(config.logging.level, "info");
}

#[test]
fn config_env_override_gpu_device() {
    std::env::set_var("JAMMI_GPU__DEVICE", "2");
    let config = JammiConfig::load(None).unwrap();
    assert_eq!(config.gpu.device, 2);
    std::env::remove_var("JAMMI_GPU__DEVICE");
}

#[test]
fn config_env_override_inference_batch_size() {
    std::env::set_var("JAMMI_INFERENCE__BATCH_SIZE", "64");
    let config = JammiConfig::load(None).unwrap();
    assert_eq!(config.inference.batch_size, 64);
    std::env::remove_var("JAMMI_INFERENCE__BATCH_SIZE");
}

#[test]
fn config_artifact_dir_has_platform_default() {
    let config = JammiConfig::load(None).unwrap();
    let path_str = config.artifact_dir.to_str().unwrap();
    assert!(
        path_str.contains("jammi"),
        "artifact_dir should contain 'jammi': {path_str}"
    );
}

// --- Catalog ---

#[test]
fn catalog_opens_and_creates_tables() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let conn = catalog.conn().unwrap();

    let tables = [
        "sources",
        "embedding_sets",
        "models",
        "fine_tune_jobs",
        "eval_runs",
        "evidence_channels",
    ];
    for table in tables {
        let count: i64 = conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap_or_else(|e| panic!("Table {table} should exist: {e}"));
        assert!(count >= 0, "Table {table} should be queryable");
    }
}

#[test]
fn catalog_seeds_evidence_channels() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let conn = catalog.conn().unwrap();

    let count: i64 = conn
        .query_row("SELECT count(*) FROM evidence_channels", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2, "Should have vector and inference channels");

    let names: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT channel_name FROM evidence_channels ORDER BY priority")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(names, vec!["vector", "inference"]);
}

#[test]
fn catalog_migrations_idempotent() {
    let dir = tempdir().unwrap();
    let _catalog1 = Catalog::open(dir.path()).unwrap();
    drop(_catalog1);
    let catalog2 = Catalog::open(dir.path());
    assert!(catalog2.is_ok(), "Opening catalog twice should not fail");
}

#[test]
fn catalog_creates_artifact_directory() {
    let dir = tempdir().unwrap();
    let sub = dir.path().join("nested").join("deep");
    let _catalog = Catalog::open(&sub).unwrap();
    assert!(sub.join("catalog.db").exists());
}

// --- Error types ---

#[test]
fn error_from_io() {
    let err: JammiError = std::io::Error::new(std::io::ErrorKind::NotFound, "test").into();
    assert!(err.to_string().contains("test"));
}

#[test]
fn error_from_rusqlite() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let conn = catalog.conn().unwrap();
    let result: Result<i64, _> = conn.query_row("SELECT * FROM nonexistent", [], |r| r.get(0));
    let err: JammiError = result.unwrap_err().into();
    assert!(err.to_string().to_lowercase().contains("no such table"));
}
