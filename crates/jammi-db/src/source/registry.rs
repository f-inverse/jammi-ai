//! Source resolution: the catalog's `sources` table is the truth, and this
//! process's DataFusion catalogs are a cache of it that revalidates on every
//! resolution.
//!
//! A session's DataFusion context resolves a three-part name
//! `<source>.public.<table>` through the [`JammiCatalogList`] installed as
//! its catalog list. Every name that is not one of the session's fixed
//! catalogs (DataFusion's default catalog, where result tables live, and
//! `mutable`) is a source id, and the catalog handed back for it is a lazy
//! view onto the [`SourceRegistry`]: asking that view for a table reads the
//! source's row from the catalog, builds (or rebuilds) the providers when
//! the row's [`SourceDefinition`] is not the one the cached providers were
//! built from, evicts them when the row is gone, and serves the table from
//! the result. A source registered, redefined or removed on any replica
//! sharing the catalog is therefore observed by every other replica at its
//! next resolution — nothing needs a restart, and nothing is revalidated
//! verb by verb.
//!
//! Three entry points build providers, and all three descend through ONE
//! function, [`SourceRegistry::build`]: [`crate::session::JammiSession::add_source`]
//! (build, persist the row, admit), the startup preload (adopt every
//! persisted row), and a resolution miss or change (adopt the row just
//! read). A build is single-flighted: two concurrent resolutions of the
//! same unbuilt source wait on one build rather than each opening a driver.
//!
//! # Where the read-through sits, and why it is async
//!
//! DataFusion resolves a table reference in two steps: a synchronous
//! [`CatalogProviderList::catalog`] → [`CatalogProvider::schema`] walk, then
//! an asynchronous [`SchemaProvider::table`]. The catalog read is async, so
//! it lives in the async step — the synchronous step hands out a view bound
//! to the source id and touches nothing. This is the honest shape: no
//! `block_in_place` bridging a synchronous method onto an async catalog, so
//! resolution behaves identically on a multi-thread runtime, a
//! current-thread runtime (`#[tokio::test]`), and a runtime with one worker,
//! and can never deadlock a pool connection the surrounding task already
//! holds. The two synchronous listings DataFusion also exposes —
//! [`CatalogProviderList::catalog_names`] and
//! [`SchemaProvider::table_names`] — report what this process has cached,
//! never the catalog: they serve `information_schema` and diagnostics, not
//! name resolution.
//!
//! Tenant scope does not enter here: provider resolution is tenant-agnostic
//! (a source's row is read across tenants, exactly as the startup preload
//! lists them), and tenant isolation is enforced on the rows a scan returns
//! by the analyzer, not by which providers resolve.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, CatalogProviderList, SchemaProvider, TableProvider};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::session_state::SessionState;
use parking_lot::RwLock;

use super::file_format;
use super::schema_provider::PublicSchemaCatalog;
use super::{table_name_from_url, SourceDefinition, SourceType};
use crate::catalog::Catalog;
use crate::error::{JammiError, Result};
use crate::storage::{StorageRegistry, StorageUrl};
use crate::tenant_scope::SourceTenantColumns;

/// A source's DataFusion table providers keyed by table name, in the order
/// the driver discovered them.
pub(crate) type SourceTables = Vec<(String, Arc<dyn TableProvider>)>;

/// What one build of a source yields: its tables, plus the directory-listing
/// extension [`file_format::create_listing_table`] resolved adaptively
/// (`Some` only for a `JsonLines` file source with no explicit override —
/// the value `add_source` pins into the persisted connection).
pub(crate) struct BuiltSource {
    pub(crate) tables: SourceTables,
    pub(crate) resolved_extension: Option<String>,
}

/// The providers this process holds for one source, with the definition
/// they were built from — the cache entry a resolution revalidates against.
pub(crate) struct ResolvedSource {
    definition: SourceDefinition,
    tables: SourceTables,
}

impl ResolvedSource {
    /// The source's table names, in discovery order.
    pub(crate) fn table_names(&self) -> Vec<String> {
        self.tables.iter().map(|(name, _)| name.clone()).collect()
    }

    fn table(&self, name: &str) -> Option<Arc<dyn TableProvider>> {
        self.tables
            .iter()
            .find(|(table, _)| table == name)
            .map(|(_, provider)| Arc::clone(provider))
    }
}

/// The session's cache of source providers, read through to the catalog.
/// See the module docs.
pub(crate) struct SourceRegistry {
    catalog: Arc<Catalog>,
    storage: StorageRegistry,
    /// Replayed from each admitted definition's `tenant_column`, so the
    /// analyzer scopes a source the moment its providers exist — on this
    /// process's own `add_source` and on a resolution from another replica's
    /// row alike.
    tenant_columns: Arc<SourceTenantColumns>,
    /// The session state providers are built against: its config options
    /// drive listing and schema inference, and its runtime env — shared with
    /// the live context — is where a source's object store is registered.
    /// A clone of the session's own state at construction; nothing in it
    /// that a build reads changes afterwards.
    session: SessionState,
    cached: RwLock<HashMap<String, Arc<ResolvedSource>>>,
    /// Single-flights builds: held across one build, never across a cache
    /// hit.
    build: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for SourceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceRegistry")
            .field("cached", &self.cached_ids())
            .finish_non_exhaustive()
    }
}

impl SourceRegistry {
    pub(crate) fn new(
        catalog: Arc<Catalog>,
        storage: StorageRegistry,
        tenant_columns: Arc<SourceTenantColumns>,
        session: SessionState,
    ) -> Self {
        Self {
            catalog,
            storage,
            tenant_columns,
            session,
            cached: RwLock::new(HashMap::new()),
            build: tokio::sync::Mutex::new(()),
        }
    }

    /// Resolve `source_id` from its catalog row: the cached providers when
    /// the row still carries the definition they were built from, freshly
    /// built ones when it does not, and [`JammiError::SourceNotFound`] —
    /// after evicting whatever was cached — when there is no row.
    pub(crate) async fn resolve(&self, source_id: &str) -> Result<Arc<ResolvedSource>> {
        match self.catalog.get_source_across_tenants(source_id).await? {
            Some(record) => self.adopt(source_id, record.into_definition()).await,
            None => {
                self.evict(source_id);
                Err(JammiError::SourceNotFound {
                    source_id: source_id.into(),
                })
            }
        }
    }

    /// Hold providers for `definition`: the cached ones when they were built
    /// from an equal definition, otherwise a new build admitted in their
    /// place. The row-driven path — the startup preload and every
    /// resolution descend through here.
    pub(crate) async fn adopt(
        &self,
        source_id: &str,
        definition: SourceDefinition,
    ) -> Result<Arc<ResolvedSource>> {
        if let Some(hit) = self.cached_for(source_id, &definition) {
            return Ok(hit);
        }
        let _building = self.build.lock().await;
        if let Some(hit) = self.cached_for(source_id, &definition) {
            return Ok(hit);
        }
        let built = self.build(source_id, &definition).await?;
        Ok(self.admit(source_id, definition, built.tables))
    }

    /// Build the DataFusion table providers for `definition` — the ONE
    /// place a source's providers are constructed, whichever entry point
    /// asked for them. Registers the source's object store on the session's
    /// runtime env as a side effect of a file source's build.
    pub(crate) async fn build(
        &self,
        source_id: &str,
        definition: &SourceDefinition,
    ) -> Result<BuiltSource> {
        let connection = &definition.connection;
        match &definition.source_type {
            SourceType::File => {
                let format = connection.format.as_ref().ok_or_else(|| {
                    JammiError::Config(format!("File source '{source_id}' requires a format"))
                })?;
                let raw_url = connection.url.as_deref().ok_or_else(|| {
                    JammiError::Config(format!("File source '{source_id}' requires a URL"))
                })?;
                let url = StorageUrl::parse(raw_url)?;
                let (table, resolved_extension) = file_format::create_listing_table(
                    &self.storage,
                    &url,
                    format,
                    connection.file_extension.as_deref(),
                    connection.cloud.as_ref(),
                    &self.session,
                )
                .await?;
                Ok(BuiltSource {
                    tables: vec![(table_name_from_url(url.as_str()), table)],
                    resolved_extension,
                })
            }
            #[cfg(feature = "postgres")]
            SourceType::Postgres => Ok(BuiltSource {
                tables: crate::source::postgres::create_postgres_tables(source_id, connection)
                    .await?,
                resolved_extension: None,
            }),
            #[cfg(not(feature = "postgres"))]
            SourceType::Postgres => Err(JammiError::Config(
                "Postgres support requires the 'postgres' feature flag".into(),
            )),
            #[cfg(feature = "mysql")]
            SourceType::Mysql => Ok(BuiltSource {
                tables: crate::source::mysql::create_mysql_tables(source_id, connection).await?,
                resolved_extension: None,
            }),
            #[cfg(not(feature = "mysql"))]
            SourceType::Mysql => Err(JammiError::Config(
                "MySQL support requires the 'mysql' feature flag".into(),
            )),
        }
    }

    /// Make `tables` the providers this process serves for `source_id`,
    /// keyed on `definition`, and replay the definition's tenant column into
    /// the analyzer's lookup. Replaces whatever was cached.
    pub(crate) fn admit(
        &self,
        source_id: &str,
        definition: SourceDefinition,
        tables: SourceTables,
    ) -> Arc<ResolvedSource> {
        self.tenant_columns
            .set(source_id, definition.connection.tenant_column.clone());
        let resolved = Arc::new(ResolvedSource { definition, tables });
        self.cached
            .write()
            .insert(source_id.to_string(), Arc::clone(&resolved));
        resolved
    }

    /// Drop the providers cached for `source_id`, if any, and its tenant
    /// column. A later resolution starts from the catalog row again.
    pub(crate) fn evict(&self, source_id: &str) {
        self.tenant_columns.set(source_id, None);
        self.cached.write().remove(source_id);
    }

    /// The source ids this process currently holds providers for.
    pub(crate) fn cached_ids(&self) -> Vec<String> {
        self.cached.read().keys().cloned().collect()
    }

    fn cached(&self, source_id: &str) -> Option<Arc<ResolvedSource>> {
        self.cached.read().get(source_id).cloned()
    }

    fn cached_for(
        &self,
        source_id: &str,
        definition: &SourceDefinition,
    ) -> Option<Arc<ResolvedSource>> {
        self.cached(source_id)
            .filter(|resolved| resolved.definition == *definition)
    }
}

/// The `public` schema of one source, resolved through the registry on
/// every table lookup. See the module docs for which methods read through
/// and which report the cache.
struct SourceSchema {
    source_id: String,
    registry: Arc<SourceRegistry>,
}

impl std::fmt::Debug for SourceSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceSchema")
            .field("source_id", &self.source_id)
            .field("tables", &self.table_names())
            .finish()
    }
}

#[async_trait]
impl SchemaProvider for SourceSchema {
    fn table_names(&self) -> Vec<String> {
        self.registry
            .cached(&self.source_id)
            .map(|resolved| resolved.table_names())
            .unwrap_or_default()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        let resolved = self
            .registry
            .resolve(&self.source_id)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(resolved.table(name))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.registry
            .cached(&self.source_id)
            .is_some_and(|resolved| resolved.table(name).is_some())
    }
}

/// The session's [`CatalogProviderList`]: the fixed catalogs registered on
/// it by name, and every other name resolved as a source through the
/// [`SourceRegistry`]. See the module docs.
#[derive(Debug)]
pub(crate) struct JammiCatalogList {
    /// DataFusion's default catalog (result tables) and `mutable` — the
    /// catalogs `register_catalog` binds, none of them a source.
    fixed: Arc<dyn CatalogProviderList>,
    sources: Arc<SourceRegistry>,
}

impl JammiCatalogList {
    pub(crate) fn new(fixed: Arc<dyn CatalogProviderList>, sources: Arc<SourceRegistry>) -> Self {
        Self { fixed, sources }
    }
}

impl CatalogProviderList for JammiCatalogList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.fixed.register_catalog(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        let mut names = self.fixed.catalog_names();
        names.extend(self.sources.cached_ids());
        names
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.fixed.catalog(name).or_else(|| {
            let schema: Arc<dyn SchemaProvider> = Arc::new(SourceSchema {
                source_id: name.to_string(),
                registry: Arc::clone(&self.sources),
            });
            Some(Arc::new(PublicSchemaCatalog::new(schema)))
        })
    }
}
