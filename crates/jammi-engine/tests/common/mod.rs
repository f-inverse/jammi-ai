use std::path::{Path, PathBuf};

/// Workspace root — two levels up from any crate in `crates/<name>/`.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
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
