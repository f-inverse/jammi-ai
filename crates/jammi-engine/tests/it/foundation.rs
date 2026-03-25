use crate::common;

use jammi_engine::catalog::status::ResultTableStatus;
use jammi_engine::catalog::Catalog;
use jammi_engine::config::JammiConfig;
use std::path::Path;
use std::sync::Mutex;
use tempfile::tempdir;

// Env-var-mutating config tests must not run in parallel.
static ENV_LOCK: Mutex<()> = Mutex::new(());

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
    let _lock = ENV_LOCK.lock().unwrap();
    std::env::remove_var("JAMMI_GPU__DEVICE");
    std::env::remove_var("JAMMI_INFERENCE__BATCH_SIZE");
    let config = JammiConfig::load(None).unwrap();
    assert_eq!(config.engine.batch_size, 8192);
    assert_eq!(config.gpu.device, 0);
    assert_eq!(config.inference.batch_size, 32);
    assert_eq!(config.logging.level, "info");
}

#[test]
fn config_env_overrides() {
    let _lock = ENV_LOCK.lock().unwrap();
    std::env::set_var("JAMMI_GPU__DEVICE", "2");
    std::env::set_var("JAMMI_INFERENCE__BATCH_SIZE", "64");
    let config = JammiConfig::load(None).unwrap();
    assert_eq!(config.gpu.device, 2);
    assert_eq!(config.inference.batch_size, 64);
    std::env::remove_var("JAMMI_GPU__DEVICE");
    std::env::remove_var("JAMMI_INFERENCE__BATCH_SIZE");
}

// --- Catalog ---

#[test]
fn catalog_opens_and_creates_tables() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // Exercise public APIs that transitively verify each table exists
    assert!(catalog.list_sources().unwrap().is_empty());
    assert!(catalog.list_models().unwrap().is_empty());
    assert!(catalog
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .unwrap()
        .is_empty());
    assert!(!catalog.evidence_channel_names().unwrap().is_empty());
}

#[test]
fn catalog_seeds_evidence_channels() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    let names = catalog.evidence_channel_names().unwrap();
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
