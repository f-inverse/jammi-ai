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
//!
//! A result table is catalogued state every replica sees, and the catalog
//! row is what a name means: every resolution reads the row and serves the
//! binding it holds only while the row still names the artifact the binding
//! was taken from (`BoundArtifact`). A name this provider holds no binding
//! for is bound on the spot, so a table another replica created after this
//! one started reads here through the same binding it would have taken at
//! startup; a name whose row moved on — replaced by `CREATE OR REPLACE`,
//! published at a new version — is rebound, so a reader on any replica
//! resolves the table the catalog names and never the storage of one it no
//! longer does; a name whose row is gone resolves nothing.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock, Weak};

use async_trait::async_trait;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionState;
use datafusion::prelude::SessionContext;

use crate::catalog::result_repo::ResultTableRecord;
use crate::catalog::status::ResultTableStatus;
use crate::session::QueryContext;
use crate::store::{result_table_relation, RelationKey, ResultStore};
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// The artifact a binding was taken from: the row's `parquet_path` and
/// `current_version` at binding time. A row whose artifact differs names
/// another table under the same name, and the binding is stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundArtifact {
    parquet_path: String,
    current_version: Option<i64>,
}

impl BoundArtifact {
    /// The artifact `record` names now.
    pub(crate) fn of(record: &ResultTableRecord) -> Self {
        Self {
            parquet_path: record.parquet_path.clone(),
            current_version: record.current_version,
        }
    }
}

/// A binding this provider holds: the DataFusion provider and the artifact
/// it was bound from — `None` for a binding registered without a row (the
/// ownerless trait entry point).
#[derive(Clone)]
struct Binding {
    provider: Arc<dyn TableProvider>,
    bound: Option<BoundArtifact>,
}

/// One registered result table: its binding and the catalog-row owner that
/// gates whether the current scope may resolve it.
struct ResultTableEntry {
    binding: Binding,
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
    /// How a name this provider does not hold is bound from the catalog —
    /// installed once by the store that owns this provider.
    resolver: OnceLock<Resolver>,
}

/// The store and session a miss is resolved through, held weakly: the
/// session's config owns the store (its extension) and the store owns this
/// provider, so a strong reference here would be a cycle.
struct Resolver {
    store: Weak<ResultStore>,
    state: Weak<parking_lot::RwLock<SessionState>>,
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
            resolver: OnceLock::new(),
        }
    }

    /// Install the store and session a miss is resolved through — once;
    /// `false` when one is already installed (the first stays).
    pub(crate) fn install_resolver(&self, store: &Arc<ResultStore>, ctx: &SessionContext) -> bool {
        self.resolver
            .set(Resolver {
                store: Arc::downgrade(store),
                state: ctx.state_weak_ref(),
            })
            .is_ok()
    }

    /// The binding held under `name` and visible to the current scope.
    fn bound(&self, name: &str) -> DfResult<Option<Binding>> {
        let guard = self
            .tables
            .read()
            .map_err(|e| DataFusionError::Internal(format!("result-table schema lock: {e}")))?;
        Ok(match guard.get(name) {
            Some(entry) if self.visible(entry.owner) => Some(entry.binding.clone()),
            // Present-but-invisible resolves the same not-found as absent, so a
            // peer's private result table is indistinguishable from one that
            // was never created.
            _ => None,
        })
    }

    /// Resolve `name` against the catalog: the binding held for it when the
    /// row still names the artifact it was bound from; a fresh binding —
    /// the one this replica would have taken at startup had the table
    /// existed then — when the row names another artifact or nothing is
    /// bound; nothing when the row is gone or not `ready` (the stale
    /// binding, if any, dropped). A name that is no result table's
    /// relation, or a provider with no resolver installed, serves whatever
    /// is bound: there is no row to read.
    async fn resolve(
        &self,
        name: &str,
        held: Option<Binding>,
    ) -> DfResult<Option<Arc<dyn TableProvider>>> {
        let (Some(resolver), Some(table)) = (self.resolver.get(), RelationKey::table_name_of(name))
        else {
            return Ok(held.map(|binding| binding.provider));
        };
        let (Some(store), Some(state)) = (resolver.store.upgrade(), resolver.state.upgrade())
        else {
            return Ok(held.map(|binding| binding.provider));
        };
        let df = |e: crate::error::JammiError| DataFusionError::External(Box::new(e));
        let record = store.catalog().get_result_table(table).await.map_err(df)?;
        let Some(record) = record.filter(|r| r.status == ResultTableStatus::Ready.to_string())
        else {
            if held.is_some() {
                self.remove(&result_table_relation(table));
            }
            return Ok(None);
        };
        if let Some(Binding {
            provider,
            bound: Some(bound),
        }) = held
        {
            if bound == BoundArtifact::of(&record) {
                return Ok(Some(provider));
            }
        }
        let ctx = QueryContext::from(SessionContext::new_with_state(state.read().clone()));
        store.bind_result_table(&ctx, &record).await.map_err(df)?;
        Ok(self.bound(name)?.map(|binding| binding.provider))
    }

    /// Register (or replace) a result table under its session relation with
    /// its catalog owner and the artifact it is bound from — the single
    /// owner-aware registration path the [`crate::store::ResultStore`]
    /// routes through, distinct from the ownerless
    /// [`SchemaProvider::register_table`] trait entry point. The map key is
    /// the relation's registered name, so a table is bound under exactly
    /// the identifier [`result_table_relation`] spells and no other.
    pub(crate) fn add_result_table(
        &self,
        relation: &RelationKey,
        provider: Arc<dyn TableProvider>,
        owner: Option<TenantId>,
        bound: BoundArtifact,
    ) {
        self.tables
            .write()
            .expect("result-table schema lock poisoned")
            .insert(
                relation.bound_name().to_string(),
                ResultTableEntry {
                    binding: Binding {
                        provider,
                        bound: Some(bound),
                    },
                    owner,
                },
            );
    }

    /// Remove one registration, returning its provider if present. Used by
    /// source removal so post-removal queries resolve not-found.
    pub fn remove(&self, relation: &RelationKey) -> Option<Arc<dyn TableProvider>> {
        self.tables
            .write()
            .expect("result-table schema lock poisoned")
            .remove(relation.bound_name())
            .map(|e| e.binding.provider)
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
        let held = self.bound(name)?;
        self.resolve(name, held).await
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
                    binding: Binding {
                        provider: table,
                        bound: None,
                    },
                    owner,
                },
            );
        Ok(prev.map(|e| e.binding.provider))
    }

    fn deregister_table(&self, name: &str) -> DfResult<Option<Arc<dyn TableProvider>>> {
        // The ownerless trait entry point's inverse: a name a `DROP TABLE`
        // over the SQL surface spells is removed as spelled, never re-derived
        // from a result-table relation.
        Ok(self
            .tables
            .write()
            .map_err(|e| DataFusionError::Internal(format!("result-table schema lock: {e}")))?
            .remove(name)
            .map(|e| e.binding.provider))
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
    let Some(provider) = schema.downcast_ref::<ResultTableSchemaProvider>() else {
        return;
    };
    for name in table_names {
        provider.remove(&result_table_relation(name));
    }
}
