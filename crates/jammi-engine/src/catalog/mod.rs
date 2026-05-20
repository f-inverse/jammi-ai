pub mod backend;
pub mod backend_postgres;
pub mod backend_sqlite;
pub mod channel_repo;
pub mod eval_repo;
pub mod fine_tune_repo;
pub mod migrations;
pub mod model_repo;
pub mod result_repo;
pub mod schema;
pub mod source_repo;
pub mod status;

use std::path::Path;

use crate::error::Result;

use backend::BackendImpl;
use backend_sqlite::SqliteBackend;
use channel_repo::ChannelRepo;

/// Artifact catalog for models, sources, and experiment metadata.
///
/// Holds a [`BackendImpl`] enum dispatching to the concrete backend
/// implementation. Default deployment uses SQLite (WAL mode, single-process
/// embedded). Multi-process server deployments construct a Postgres backend
/// via [`Catalog::from_backend`].
pub struct Catalog {
    backend: BackendImpl,
}

impl Catalog {
    /// Open (or create) the catalog database in `artifact_dir`, running any
    /// pending migrations. Default backend: SQLite WAL at
    /// `<artifact_dir>/catalog.db`.
    pub async fn open(artifact_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(artifact_dir)?;
        let db_path = artifact_dir.join("catalog.db");
        let backend = SqliteBackend::open(&db_path).await?;
        let cat = Self {
            backend: BackendImpl::Sqlite(backend),
        };
        cat.backend.migrate().await?;
        Ok(cat)
    }

    /// Open a catalog from a pre-built backend. Used by tests and by the
    /// server deployment shape that wires a Postgres backend.
    pub fn from_backend(backend: BackendImpl) -> Self {
        Self { backend }
    }

    /// Repository view over the evidence-channel tables.
    pub fn channels(&self) -> ChannelRepo<'_> {
        ChannelRepo::new(self)
    }

    /// Crate-internal access to the underlying backend for repo
    /// implementations.
    pub(crate) fn backend(&self) -> &BackendImpl {
        &self.backend
    }
}
