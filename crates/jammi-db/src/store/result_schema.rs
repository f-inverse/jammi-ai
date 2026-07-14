//! Tenant-scoped DataFusion [`SchemaProvider`] for the result-table lane.
//!
//! Result tables are wholly owned by one tenant (or GLOBAL, `tenant_id IS
//! NULL`) — the owner lives on the catalog row, not on a per-data-row column,
//! so a result Parquet carries no `tenant_id` the predicate-injection analyzer
//! ([`crate::tenant_scope::TenantScopeAnalyzerRule`]) could filter on. This
//! provider closes that gap: it gates a result table's **resolution
//! visibility** on the catalog owner, applying the same
//! `(tenant_id = $current OR tenant_id IS NULL)` + admin-scope bypass the
//! catalog read API ([`crate::catalog::Catalog::get_result_table`]) and the
//! mutable-table read lane already enforce. A correctly-bound tenant resolves
//! only its own and GLOBAL result tables over every lane that names
//! `jammi.{table}` through the session context (Flight `db.sql`, gRPC `sql`,
//! the exact-search fallback, vector-by-key); a peer's private table resolves
//! not-found, and does not appear in the schema's table enumeration.
//!
//! This is the *organizational* half of the mechanism, matching the two lanes
//! that already scope on the catalog owner. It is not a hostile-principal
//! boundary — the trusted-network + BYO-auth posture is unchanged.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::prelude::SessionContext;

use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// One registered result table: its DataFusion provider and the catalog-row
/// owner that gates whether the current scope may resolve it.
struct ResultTableEntry {
    provider: Arc<dyn TableProvider>,
    /// Owning tenant, or `None` for a GLOBAL (`tenant_id IS NULL`) table.
    owner: Option<TenantId>,
}

/// DataFusion [`SchemaProvider`] holding the session's result tables under
/// their bare `jammi.{name}` identifiers, gating each on its catalog owner.
///
/// Installed as the session context's default schema (`datafusion.public`) —
/// the schema bare result-table names resolve through — so every read lane
/// that resolves `jammi.{table}` observes the tenant gate uniformly.
pub struct ResultTableSchemaProvider {
    tables: RwLock<HashMap<String, ResultTableEntry>>,
    /// Shared with the analyzer, catalog, and mutable lane, so every surface
    /// reads the same effective tenant (sticky binding or task-local scope).
    binding: TenantBinding,
}

impl std::fmt::Debug for ResultTableSchemaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultTableSchemaProvider")
            .field("tables", &self.table_names())
            .finish()
    }
}

impl ResultTableSchemaProvider {
    /// Construct an empty provider sharing `binding` with the rest of the
    /// session's tenant-aware surfaces.
    pub fn new(binding: TenantBinding) -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
            binding,
        }
    }

    /// Register (or replace) a result table under `name` with its catalog
    /// owner — the single owner-aware registration path the [`crate::store::ResultStore`]
    /// routes through, distinct from the ownerless [`SchemaProvider::register_table`]
    /// trait entry point.
    pub fn add_result_table(
        &self,
        name: String,
        provider: Arc<dyn TableProvider>,
        owner: Option<TenantId>,
    ) {
        self.tables
            .write()
            .expect("result-table schema lock poisoned")
            .insert(name, ResultTableEntry { provider, owner });
    }

    /// Remove one registration by name, returning its provider if present.
    /// Used by source removal so post-removal queries resolve not-found.
    pub fn remove(&self, name: &str) -> Option<Arc<dyn TableProvider>> {
        self.tables
            .write()
            .expect("result-table schema lock poisoned")
            .remove(name)
            .map(|e| e.provider)
    }

    /// Drop every registration.
    pub fn clear(&self) {
        self.tables
            .write()
            .expect("result-table schema lock poisoned")
            .clear();
    }

    /// Whether a table owned by `owner` is visible to the current scope — the
    /// same `(tenant_id = $current OR tenant_id IS NULL)` + admin-scope bypass
    /// [`crate::catalog::Catalog::get_result_table`] applies: an admin scan
    /// sees everything, a GLOBAL (`owner = None`) table is visible to all, and
    /// a tenant-owned table is visible only to that tenant.
    fn visible(&self, owner: Option<TenantId>) -> bool {
        TenantBinding::is_admin_scope() || owner.is_none() || owner == self.binding.current_tenant()
    }
}

#[async_trait]
impl SchemaProvider for ResultTableSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        match self.tables.read() {
            Ok(guard) => guard
                .iter()
                .filter(|(_, e)| self.visible(e.owner))
                .map(|(name, _)| name.clone())
                .collect(),
            Err(e) => {
                tracing::error!("result-table schema lock poisoned in table_names: {e}");
                Vec::new()
            }
        }
    }

    async fn table(&self, name: &str) -> DfResult<Option<Arc<dyn TableProvider>>> {
        let guard = self
            .tables
            .read()
            .map_err(|e| DataFusionError::Internal(format!("result-table schema lock: {e}")))?;
        match guard.get(name) {
            Some(entry) if self.visible(entry.owner) => Ok(Some(Arc::clone(&entry.provider))),
            // Present-but-invisible resolves the same not-found as absent, so a
            // peer's private result table is indistinguishable from one that
            // was never created.
            _ => Ok(None),
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        match self.tables.read() {
            Ok(guard) => guard
                .get(name)
                .map(|e| self.visible(e.owner))
                .unwrap_or(false),
            Err(e) => {
                tracing::error!("result-table schema lock poisoned in table_exist: {e}");
                false
            }
        }
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> DfResult<Option<Arc<dyn TableProvider>>> {
        // The engine's owner-aware path is `add_result_table`; this bare
        // trait entry point carries no owner, so a table registered through it
        // (a `CREATE TABLE` over the SQL surface) is owned by the tenant
        // currently in scope.
        let owner = self.binding.current_tenant();
        let prev = self
            .tables
            .write()
            .map_err(|e| DataFusionError::Internal(format!("result-table schema lock: {e}")))?
            .insert(
                name,
                ResultTableEntry {
                    provider: table,
                    owner,
                },
            );
        Ok(prev.map(|e| e.provider))
    }

    fn deregister_table(&self, name: &str) -> DfResult<Option<Arc<dyn TableProvider>>> {
        Ok(self.remove(name))
    }
}

/// Deregister the named result tables from the [`ResultTableSchemaProvider`]
/// installed as `ctx`'s default schema, if one is installed. Best-effort: a
/// context whose default schema is not a result-table provider (a bare test
/// context that never registered one) is a no-op. Called by source removal so a
/// removed source's result tables resolve not-found afterwards.
pub(crate) fn deregister_result_tables<'a, I>(ctx: &SessionContext, table_names: I)
where
    I: IntoIterator<Item = &'a str>,
{
    let config = ctx.copied_config();
    let catalog_opts = &config.options().catalog;
    let Some(catalog) = ctx.catalog(&catalog_opts.default_catalog) else {
        return;
    };
    let Some(schema) = catalog.schema(&catalog_opts.default_schema) else {
        return;
    };
    let Some(provider) = schema.as_any().downcast_ref::<ResultTableSchemaProvider>() else {
        return;
    };
    for name in table_names {
        provider.remove(&format!("jammi.{name}"));
    }
}
