pub mod backend;
pub mod backend_postgres;
pub mod backend_sqlite;
pub mod channel_repo;
pub mod eval_repo;
pub mod fine_tune_repo;
pub mod migrations;
pub mod model_repo;
pub mod mutable_repo;
pub mod result_repo;
pub mod schema;
pub mod source_repo;
pub mod status;

use std::path::Path;
use std::sync::Arc;

use crate::error::Result;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

use backend::BackendImpl;
use backend_sqlite::SqliteBackend;
use channel_repo::ChannelRepo;

/// Artifact catalog for models, sources, and experiment metadata.
///
/// Holds a [`BackendImpl`] enum dispatching to the concrete backend
/// implementation. Default deployment uses SQLite (WAL mode, single-process
/// embedded). Multi-process server deployments construct a Postgres backend
/// via [`Catalog::from_backend`].
///
/// Tenant binding: optional. When set, every catalog write reads the bound
/// tenant on each call, writes `tenant_id = <bound>` (NULL when Unscoped),
/// and asserts via [`backend::Transaction::assert_tenant_matches`] before the
/// underlying INSERT to honour SPEC-03 §7 defence-in-depth. Reads filter to
/// `tenant_id = <bound> OR tenant_id IS NULL`. When unbound (default), every
/// row is written with NULL `tenant_id` and reads return every row — the
/// no-op identity for single-tenant deployments.
pub struct Catalog {
    backend: Arc<BackendImpl>,
    tenant: Option<TenantBinding>,
}

impl Catalog {
    /// Open (or create) the catalog database in `artifact_dir`, running any
    /// pending migrations. Default backend: SQLite WAL at
    /// `<artifact_dir>/catalog.db`. No tenant binding.
    pub async fn open(artifact_dir: &Path) -> Result<Self> {
        Self::open_with_tenant(artifact_dir, None).await
    }

    /// Like [`Catalog::open`], but bind the catalog to a shared session
    /// tenant. The binding is consulted on every read and write; mutations
    /// to the shared `TenantBinding` (e.g. via `JammiSession::with_tenant`)
    /// are observed by subsequent catalog operations.
    pub async fn open_with_tenant(
        artifact_dir: &Path,
        tenant: Option<TenantBinding>,
    ) -> Result<Self> {
        std::fs::create_dir_all(artifact_dir)?;
        let db_path = artifact_dir.join("catalog.db");
        let backend = SqliteBackend::open(&db_path).await?;
        let cat = Self {
            backend: Arc::new(BackendImpl::Sqlite(backend)),
            tenant,
        };
        cat.backend.migrate().await?;
        Ok(cat)
    }

    /// Open a catalog from a pre-built backend. Used by tests and by the
    /// server deployment shape that wires a Postgres backend.
    pub fn from_backend(backend: BackendImpl) -> Self {
        Self {
            backend: Arc::new(backend),
            tenant: None,
        }
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

    /// Shared handle to the underlying backend. Used by the mutable-table
    /// registry to issue DDL/DML on the same database the catalog rows live
    /// in.
    pub fn backend_arc(&self) -> Arc<BackendImpl> {
        Arc::clone(&self.backend)
    }

    /// The tenant currently bound to this catalog, or `None` if unscoped.
    /// Resolved through the shared [`TenantBinding`] (so a `with_tenant`
    /// call on the owning session is observable here immediately).
    pub fn current_tenant(&self) -> Option<TenantId> {
        self.tenant
            .as_ref()
            .and_then(|b| b.read().ok().and_then(|c| c.tenant()))
    }
}
