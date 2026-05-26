//! Shared test helpers for jammi-db and jammi-ai integration tests.

use std::path::{Path, PathBuf};

use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::session::JammiSession;

/// Env var inspected by [`pg_url_for_tests`] and [`make_test_session`] to
/// reach a live Postgres instance. CI sets this for the `test-pg` job; local
/// runs can leave it unset.
pub const PG_URL_ENV: &str = "JAMMI_TEST_PG_URL";

/// Return the configured Postgres URL when both `JAMMI_TEST_PG_URL` is set
/// and the value is non-empty. Tests that need a live Postgres backend call
/// this to decide whether to skip (without `#[ignore]`, which CLAUDE.md
/// forbids — instead they early-return with a `tracing::warn` so CI logs
/// surface the skip).
pub fn pg_url_for_tests() -> Option<String> {
    std::env::var(PG_URL_ENV).ok().filter(|s| !s.is_empty())
}

/// Build a [`JammiSession`] backed by `kind` for parameterized integration
/// tests. The caller passes an artifact dir (used by SQLite for the catalog
/// file and by both backends for result-table parquet); the Postgres variant
/// connects to `JAMMI_TEST_PG_URL` and runs migrations.
///
/// Returns `None` when `kind = Postgres` and the env var is unset, so tests
/// can `let session = match make_test_session(kind, dir).await { Some(s) => s,
/// None => return };` to skip Postgres parameterizations on the hermetic
/// `cargo test` lane without per-test `#[cfg(feature = …)]` decoration.
pub async fn make_test_session(kind: BackendKind, artifact_dir: &Path) -> Option<JammiSession> {
    let config = test_config(artifact_dir);
    match kind {
        BackendKind::Sqlite => Some(
            JammiSession::new(config)
                .await
                .expect("sqlite-backed session"),
        ),
        BackendKind::Postgres => {
            let url = pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .expect("open postgres backend");
            let backend = BackendImpl::Postgres(pg);
            Some(
                JammiSession::with_backend(config, backend)
                    .await
                    .expect("postgres-backed session"),
            )
        }
    }
}

/// Workspace root — two levels up from any crate in `crates/<name>/`.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Root of the test fixtures directory (at workspace root). Houses the
/// generic test-only fixtures (`patents.parquet`, `assignees.csv`,
/// `golden_relevance.csv`, the tiny encoder fixtures that are not part of
/// the public cookbook surface, etc.).
pub fn fixtures_dir() -> PathBuf {
    workspace_root().join("tests").join("fixtures")
}

/// Root of the cookbook fixtures directory (at workspace root). Houses
/// the fixtures consumed by the OSS cookbook recipes — currently
/// `tiny_bert/`, `tiny_modernbert_classifier/`, and the synthetic data
/// files (`tiny_corpus.parquet`, `tiny_golden.json`, `tiny_labels.csv`,
/// `tiny_pairs.csv`). Integration tests that exercise the same model
/// fixtures the cookbook ships read from here so the recipe and the test
/// share one source of truth.
pub fn cookbook_fixtures_dir() -> PathBuf {
    workspace_root().join("cookbook").join("fixtures")
}

/// Path to a specific fixture file under `tests/fixtures/`.
pub fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

/// Path to a specific fixture file under `cookbook/fixtures/`.
pub fn cookbook_fixture(name: &str) -> PathBuf {
    cookbook_fixtures_dir().join(name)
}

/// URL for a `tests/fixtures/` fixture suitable for DataFusion's ListingTable.
pub fn fixture_url(name: &str) -> String {
    format!("file://{}", fixture(name).display())
}

/// URL for a `cookbook/fixtures/` fixture suitable for DataFusion's ListingTable.
pub fn cookbook_fixture_url(name: &str) -> String {
    format!("file://{}", cookbook_fixture(name).display())
}

/// Convert a `file://...` URL back into a filesystem `PathBuf` for tests
/// that need to exercise on-disk file existence checks (e.g. asserting a
/// sidecar bundle was written, or peeking at the raw bytes a result-table
/// row references). Returns the input unchanged when no `file://` prefix
/// is present so the helper composes with callers that already strip it.
pub fn url_to_path(url: &str) -> PathBuf {
    PathBuf::from(url.strip_prefix("file://").unwrap_or(url))
}

/// Create a JammiConfig pointing at a temporary artifact directory.
pub fn test_config(artifact_dir: &Path) -> jammi_db::config::JammiConfig {
    jammi_db::config::JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_db::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: jammi_db::config::LoggingConfig {
            level: "debug".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Register a custom evidence channel with the catalog. Used by tests
/// that exercise the data-driven provenance machinery beyond the seeded
/// `vector` and `inference` channels.
pub async fn register_test_channel(
    catalog: &jammi_db::catalog::Catalog,
    id: &str,
    priority: i32,
    columns: &[(&str, jammi_db::catalog::channel_repo::ChannelColumnType)],
) -> jammi_db::error::Result<()> {
    let spec = jammi_db::catalog::channel_repo::ChannelSpec {
        id: jammi_db::ChannelId::new(id)?,
        priority,
        columns: columns
            .iter()
            .map(
                |(name, dtype)| jammi_db::catalog::channel_repo::ChannelColumn {
                    name: (*name).into(),
                    data_type: *dtype,
                },
            )
            .collect(),
    };
    catalog.channels().register(&spec).await
}
