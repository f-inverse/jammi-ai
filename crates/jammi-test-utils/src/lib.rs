//! Shared test helpers for jammi-engine and jammi-ai integration tests.

use std::path::{Path, PathBuf};

/// Workspace root — two levels up from any crate in `crates/<name>/`.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Root of the test fixtures directory (at workspace root).
pub fn fixtures_dir() -> PathBuf {
    workspace_root().join("tests").join("fixtures")
}

/// Path to a specific fixture file.
pub fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

/// URL for a fixture file suitable for DataFusion's ListingTable.
pub fn fixture_url(name: &str) -> String {
    format!("file://{}", fixture(name).display())
}

/// Create a JammiConfig pointing at a temporary artifact directory.
pub fn test_config(artifact_dir: &Path) -> jammi_engine::config::JammiConfig {
    jammi_engine::config::JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_engine::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_engine::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: jammi_engine::config::LoggingConfig {
            level: "debug".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
