use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_federation::{FederatedQueryPlanner, FederationOptimizerRule};

use std::sync::RwLock;

use crate::catalog::topic_repo::TopicRepo;
use crate::catalog::Catalog;
use crate::config::JammiConfig;
use crate::error::{JammiError, Result};
use crate::source::mutable::MutableTableRegistry;
use crate::source::registry::SourceCatalog;
use crate::source::schema_provider::JammiSchemaProvider;
use crate::source::{local, table_name_from_url, SourceConnection, SourceType};
use crate::sql::topic_ddl::{self, TopicDdl};
use crate::store::mutable::definition::{MutableTableDefinition, MutableTableId};
use crate::store::mutable::sqlite::SqliteMutableBackend;
use crate::store::mutable::MutableBackend;
use crate::tenant::{TenantContext, TenantId};
use crate::tenant_scope::{SourceTenantColumns, TenantBinding, TenantScopeAnalyzerRule};
use crate::trigger::{
    InMemoryBroker, Publisher, Subscriber, TopicDefinition, TopicId, TriggerBroker,
};

/// Primary entry point for the Jammi query engine.
///
/// Wraps a DataFusion `SessionContext` with source registration,
/// an artifact catalog, and platform configuration.
pub struct JammiSession {
    ctx: SessionContext,
    catalog: Arc<Catalog>,
    config: Arc<JammiConfig>,
    tenant: TenantBinding,
    /// Shared with the `TenantScopeAnalyzerRule` so callers can register
    /// per-source tenant columns at `add_source` time (for federated sources
    /// whose tenant discriminator has a non-`tenant_id` name).
    source_tenant_columns: Arc<SourceTenantColumns>,
    mutable: Arc<MutableTableRegistry>,
    mutable_schema: Arc<JammiSchemaProvider>,
    /// Default broker used by the in-process trigger surface — replaced at
    /// deployment time by callers that wire a clustered broker (e.g.
    /// JetStream) into the session via `with_trigger_broker`.
    trigger_broker: Arc<dyn TriggerBroker>,
    topic_repo: Arc<TopicRepo>,
    publisher: Arc<Publisher>,
    subscriber: Arc<Subscriber>,
}

impl JammiSession {
    /// Create a new session, opening the catalog and configuring DataFusion.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        let config = Arc::new(config);

        // Construct the shared tenant binding first so the catalog can be
        // opened with it; downstream writes / reads in the catalog repos
        // consult the binding to bind / filter `tenant_id` on every row.
        let tenant_binding: TenantBinding = Arc::new(RwLock::new(TenantContext::Unscoped));
        let catalog = Arc::new(
            Catalog::open_with_tenant(&config.artifact_dir, Some(Arc::clone(&tenant_binding)))
                .await?,
        );
        Self::build(config, catalog, tenant_binding).await
    }

    /// Build a session around a caller-supplied catalog backend. Migrations
    /// are applied here so the caller hands in a connected-but-unmigrated
    /// [`crate::catalog::backend::BackendImpl`]; the session takes it from
    /// there.
    ///
    /// Used by:
    /// - tests that need to parameterize over backend (SQLite tempfile vs
    ///   live Postgres URL) without going through the path-based
    ///   [`Catalog::open_with_tenant`] surface
    /// - server deployments that compose their own backend (e.g. a shared
    ///   Postgres pool across multiple `jammi-server` replicas)
    pub async fn with_backend(
        config: JammiConfig,
        backend: crate::catalog::backend::BackendImpl,
    ) -> Result<Self> {
        let config = Arc::new(config);
        let tenant_binding: TenantBinding = Arc::new(RwLock::new(TenantContext::Unscoped));
        backend.migrate().await?;
        let catalog = Arc::new(Catalog::from_backend_with_tenant(
            backend,
            Some(Arc::clone(&tenant_binding)),
        ));
        Self::build(config, catalog, tenant_binding).await
    }

    async fn build(
        config: Arc<JammiConfig>,
        catalog: Arc<Catalog>,
        tenant_binding: TenantBinding,
    ) -> Result<Self> {
        let session_config = SessionConfig::new()
            .with_target_partitions(config.engine.execution_threads)
            .with_batch_size(config.engine.batch_size);

        // Build a base context to get the default state, then layer in
        // federation support (optimizer rule + query planner).
        let base_ctx = SessionContext::new_with_config(session_config);
        let base_state = base_ctx.state();

        let mut rules = base_state.optimizer().rules.clone();
        let insert_pos = rules
            .iter()
            .position(|r| r.name() == "scalar_subquery_to_join")
            .map(|pos| pos + 1)
            .unwrap_or(rules.len());
        rules.insert(insert_pos, Arc::new(FederationOptimizerRule::new()));

        // Tenant-scope analyzer rule injected before existing analyzer rules
        // so the predicate is applied before federation, projection pruning,
        // etc. Shares the binding with the catalog and the mutable-table
        // registry so every read/write surface observes the same value.
        let source_tenant_columns = Arc::new(SourceTenantColumns::new());
        let analyzer_rule = Arc::new(TenantScopeAnalyzerRule::new(
            Arc::clone(&tenant_binding),
            Arc::clone(&source_tenant_columns),
        ));

        let mut analyzer_rules = base_state.analyzer().rules.clone();
        analyzer_rules.push(analyzer_rule);

        let federated_state = SessionStateBuilder::new_from_existing(base_state)
            .with_optimizer_rules(rules)
            .with_analyzer_rules(analyzer_rules)
            .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
            .build();

        let ctx = SessionContext::new_with_state(federated_state);

        // Construct the mutable-table registry backed by the same backend
        // the catalog runs on. Phase 2 ships the SQLite renderer; a Postgres
        // deployment would swap the renderer when `BackendKind::Postgres`.
        let mutable_backend: Arc<dyn MutableBackend> = match catalog.backend_arc().backend_kind() {
            crate::catalog::backend::BackendKind::Sqlite => {
                Arc::new(SqliteMutableBackend::new(catalog.backend_arc()))
            }
            crate::catalog::backend::BackendKind::Postgres => Arc::new(
                crate::store::mutable::postgres::PostgresMutableBackend::new(catalog.backend_arc()),
            ),
        };
        let mutable = Arc::new(MutableTableRegistry::new(
            Arc::clone(&catalog),
            mutable_backend,
            Arc::clone(&tenant_binding),
        ));

        // The "mutable" catalog hosts every registered mutable companion
        // table under a single "public" schema. Register the empty catalog
        // up-front so three-part names resolve before any table is added.
        let mutable_schema = Arc::new(JammiSchemaProvider::new());
        let mutable_catalog = Arc::new(SourceCatalog::new(Arc::clone(&mutable_schema)));
        ctx.register_catalog("mutable", mutable_catalog);

        let trigger_broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
        let topic_repo = Arc::new(TopicRepo::new(Arc::clone(&catalog), Arc::clone(&mutable)));
        let publisher = Arc::new(Publisher::new(
            Arc::clone(&trigger_broker),
            catalog.backend_arc(),
            Arc::clone(&mutable),
        ));
        let subscriber = Arc::new(Subscriber::new(
            Arc::clone(&trigger_broker),
            Arc::clone(&mutable),
        ));

        let session = Self {
            ctx,
            catalog,
            config,
            tenant: tenant_binding,
            source_tenant_columns,
            mutable,
            mutable_schema,
            trigger_broker,
            topic_repo,
            publisher,
            subscriber,
        };
        session.reload_sources().await?;
        session.reload_mutable_tables().await?;
        Ok(session)
    }

    /// Bind a tenant scope to this session.
    ///
    /// The engine stores the identifier; auth/identity systems above the
    /// engine are responsible for verifying the caller is permitted to use
    /// `t`. The engine never mints a `TenantId` itself.
    ///
    /// After this returns, every subsequent SQL query, mutable-table read,
    /// and write through this session observes tenant-scoped data: rows
    /// whose `tenant_id = t` plus globally-scoped rows whose `tenant_id IS
    /// NULL`. Writes that target a different tenant are rejected by the
    /// write-side guard in [`crate::store::mutable::sink::MutableTableSink`].
    pub fn with_tenant(self, t: TenantId) -> Self {
        *self.tenant.write().expect("tenant binding lock poisoned") = TenantContext::Scoped(t);
        self
    }

    /// Return the tenant scope bound to this session, if any.
    pub fn tenant(&self) -> Option<TenantId> {
        self.tenant
            .read()
            .expect("tenant binding lock poisoned")
            .tenant()
    }

    /// Shared handle to the underlying tenant binding. Used by server-side
    /// composition (Flight SQL + gRPC) to plumb the per-session tenant value
    /// from a `TenantBoundProvider` / interceptor (in the `jammi-server`
    /// crate) into the engine without re-creating the binding for every
    /// request.
    pub fn tenant_binding_arc(&self) -> crate::tenant_scope::TenantBinding {
        std::sync::Arc::clone(&self.tenant)
    }

    /// Mutate the bound tenant in place. Equivalent to `with_tenant` but does
    /// not consume `self` — needed for callers that hold the session behind a
    /// shared reference (`Arc<JammiSession>` in PyO3 / CLI wrappers, the
    /// per-request gRPC handler reading `SessionTenant` from the request
    /// interceptor).
    pub fn bind_tenant(&self, t: TenantId) {
        *self.tenant.write().expect("tenant binding lock poisoned") = TenantContext::Scoped(t);
    }

    /// Clear the bound tenant in place.
    pub fn unbind_tenant(&self) {
        *self.tenant.write().expect("tenant binding lock poisoned") = TenantContext::Unscoped;
    }

    /// Register a tenant-discriminator column for a federated source. The
    /// `TenantScopeAnalyzerRule` consults this lookup when a `TableScan`'s
    /// schema does *not* itself declare a `tenant_id` column — i.e., when
    /// the user's source carries the discriminator under a different name.
    pub fn set_source_tenant_column(&self, source: &str, column: Option<String>) {
        self.source_tenant_columns.set(source, column);
    }

    /// Re-register all sources persisted in the catalog into DataFusion.
    ///
    /// Called on startup so that sources added by previous sessions (e.g. a
    /// prior CLI invocation) are available for queries immediately.
    async fn reload_sources(&self) -> Result<()> {
        let sources = self.catalog.list_sources().await?;
        for record in sources {
            if let Err(e) = self
                .register_source_tables(&record.source_id, &record.source_type, &record.connection)
                .await
            {
                tracing::warn!(
                    source_id = %record.source_id,
                    "Failed to reload source: {e}"
                );
            }
        }
        Ok(())
    }

    /// Register a data source, persisting it to the catalog and exposing it to SQL queries.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        // Check for duplicate registration.
        if self.catalog.get_source(source_id).await?.is_some() {
            return Err(JammiError::Source {
                source_id: source_id.into(),
                message: "Source already registered".into(),
            });
        }

        // Register tables in DataFusion.
        self.register_source_tables(source_id, &source_type, &connection)
            .await?;

        // Persist to catalog.
        self.catalog
            .register_source(source_id, source_type, &connection)
            .await?;

        Ok(())
    }

    /// Create DataFusion table providers for a source and register them.
    async fn register_source_tables(
        &self,
        source_id: &str,
        source_type: &SourceType,
        connection: &SourceConnection,
    ) -> Result<()> {
        let tables: Vec<(String, Arc<dyn datafusion::catalog::TableProvider>)> = match source_type {
            SourceType::Local => {
                let format = connection
                    .format
                    .as_ref()
                    .ok_or_else(|| JammiError::Config("Local source requires a format".into()))?;
                let url = connection
                    .url
                    .as_deref()
                    .ok_or_else(|| JammiError::Config("Local source requires a URL".into()))?;
                let state = self.ctx.state();
                let table = local::create_listing_table(
                    url,
                    format,
                    connection.file_extension.as_deref(),
                    &state,
                )
                .await?;
                let name = table_name_from_url(url);
                vec![(name, table)]
            }
            #[cfg(feature = "postgres")]
            SourceType::Postgres => {
                crate::source::postgres::create_postgres_tables(source_id, connection).await?
            }
            #[cfg(not(feature = "postgres"))]
            SourceType::Postgres => {
                return Err(JammiError::Config(
                    "Postgres support requires the 'postgres' feature flag".into(),
                ));
            }
            #[cfg(feature = "mysql")]
            SourceType::Mysql => {
                crate::source::mysql::create_mysql_tables(source_id, connection).await?
            }
            #[cfg(not(feature = "mysql"))]
            SourceType::Mysql => {
                return Err(JammiError::Config(
                    "MySQL support requires the 'mysql' feature flag".into(),
                ));
            }
        };

        let schema_provider = Arc::new(JammiSchemaProvider::new());
        for (table_name, table) in tables {
            schema_provider
                .add_table(table_name, table)
                .map_err(|e| JammiError::Source {
                    source_id: source_id.into(),
                    message: format!("Failed to register table: {e}"),
                })?;
        }

        let source_catalog = Arc::new(SourceCatalog::new(schema_provider));
        self.ctx.register_catalog(source_id, source_catalog);

        Ok(())
    }

    /// Remove a source and all associated state: catalog entries, result tables,
    /// disk files (Parquet + index), ANN cache, and DataFusion registration.
    ///
    /// Eval runs are preserved as immutable historical records.
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        // 1. Fetch result tables so we know which files to delete.
        let result_tables = self
            .catalog
            .delete_result_tables_for_source(source_id)
            .await?;

        // 2. Delete Parquet and index files from disk (best-effort).
        for rt in &result_tables {
            if let Err(e) = std::fs::remove_file(&rt.parquet_path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("Failed to delete parquet file '{}': {e}", rt.parquet_path);
                }
            }
            if let Some(idx) = &rt.index_path {
                for ext in ["", ".rowmap", ".manifest"] {
                    let path = format!("{idx}{ext}");
                    if let Err(e) = std::fs::remove_file(&path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!("Failed to delete index file '{path}': {e}");
                        }
                    }
                }
            }
        }

        // 3. Delete the source row from the catalog.
        self.catalog.remove_source(source_id).await?;

        // 4. Clear the DataFusion schema provider so queries return "not found".
        if let Some(catalog) = self.ctx.catalog(source_id) {
            if let Some(schema) = catalog.schema("public") {
                if let Some(provider) = schema.as_any().downcast_ref::<JammiSchemaProvider>() {
                    if let Err(e) = provider.clear() {
                        tracing::warn!("Failed to clear schema provider for '{source_id}': {e}");
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a SQL query and collect results as Arrow `RecordBatch`es.
    ///
    /// Inspects the input for trigger-stream DDL (`CREATE TOPIC` /
    /// `DROP TOPIC`) before handing it to DataFusion: those statements are
    /// routed to the engine's [`TopicRepo`] and return an empty result set.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        match topic_ddl::maybe_parse(query).map_err(|e| JammiError::Catalog(e.to_string()))? {
            Some(TopicDdl::Create(create)) => {
                self.exec_create_topic(create).await?;
                Ok(Vec::new())
            }
            Some(TopicDdl::Drop(drop_)) => {
                self.exec_drop_topic(drop_).await?;
                Ok(Vec::new())
            }
            None => {
                let df = self.ctx.sql(query).await?;
                Ok(df.collect().await?)
            }
        }
    }

    async fn exec_create_topic(&self, create: crate::sql::topic_ddl::CreateTopic) -> Result<()> {
        let topic = TopicDefinition {
            id: TopicId::new(),
            name: create.name,
            schema: Arc::new(create.schema),
            tenant: self.tenant(),
            broker_metadata: create.broker_metadata,
        };
        self.trigger_broker.register_topic(&topic).await?;
        self.topic_repo.register_topic(&topic).await?;
        Ok(())
    }

    async fn exec_drop_topic(&self, drop_: crate::sql::topic_ddl::DropTopic) -> Result<()> {
        let tenant = self.tenant();
        let topic = self.topic_repo.lookup_by_name(&drop_.name, tenant).await?;
        match topic {
            Some(t) => {
                self.topic_repo.drop_topic(t.id, tenant).await?;
                // The broker driver's view of the topic is best-effort: an
                // in-memory broker has no durable state, and the catalog row
                // (the system of record) is already gone. A driver-side
                // failure here would leak driver resources, not catalog
                // state, so surface it via tracing rather than reverting
                // the successful catalog drop.
                if let Err(e) = self.trigger_broker.drop_topic(t.id).await {
                    tracing::warn!(
                        topic_id = %t.id,
                        error = %e,
                        "trigger broker driver failed to drop topic after catalog row removal",
                    );
                }
                Ok(())
            }
            None if drop_.if_exists => Ok(()),
            None => Err(JammiError::Trigger(
                crate::trigger::TriggerError::TopicNotFound(drop_.name),
            )),
        }
    }

    /// Shared handle to the trigger broker the session was constructed with.
    pub fn trigger_broker(&self) -> Arc<dyn TriggerBroker> {
        Arc::clone(&self.trigger_broker)
    }

    /// Shared handle to the topic-catalog repo.
    pub fn topic_repo(&self) -> Arc<TopicRepo> {
        Arc::clone(&self.topic_repo)
    }

    /// Shared handle to the trigger-stream publisher.
    pub fn publisher(&self) -> Arc<Publisher> {
        Arc::clone(&self.publisher)
    }

    /// Shared handle to the trigger-stream subscriber.
    pub fn subscriber(&self) -> Arc<Subscriber> {
        Arc::clone(&self.subscriber)
    }

    /// Register a mutable companion table. After this returns, the table is
    /// queryable as `mutable.public.<id>` in the same SQL surface that
    /// federates Parquet result tables and external sources.
    pub async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        let id = def.id.clone();
        let provider = self.mutable.register(def).await?;
        self.mutable_schema
            .add_table(id.as_str().to_string(), provider)?;
        Ok(id)
    }

    /// Drop a mutable companion table.
    pub async fn drop_mutable_table(&self, id: &MutableTableId) -> Result<()> {
        self.mutable.drop_table(id).await?;
        self.mutable_schema.remove_table(id.as_str())?;
        Ok(())
    }

    /// Re-register mutable tables persisted in the catalog from a previous
    /// process. Called on startup. Registers a `TableProvider` for every
    /// table across every tenant — DataFusion name resolution does not
    /// honour session scope, so the table must be discoverable in
    /// `mutable.public.<id>` regardless of which tenant the session later
    /// binds to. Per-row tenant filtering is applied by the analyzer at
    /// query time.
    async fn reload_mutable_tables(&self) -> Result<()> {
        let defs = self.mutable.list_all().await?;
        for def in defs {
            let id = def.id.clone();
            let provider = self.mutable.provider_for(def);
            self.mutable_schema
                .add_table(id.as_str().to_string(), provider)?;
        }
        Ok(())
    }

    /// Reference to the mutable-table registry.
    pub fn mutable_tables(&self) -> &MutableTableRegistry {
        &self.mutable
    }

    /// Shared handle to the mutable-table registry. Use when a caller needs
    /// to move the registry into an `async move` closure (e.g. opening a
    /// transaction and calling `insert_batch` inside it).
    pub fn mutable_tables_arc(&self) -> Arc<MutableTableRegistry> {
        Arc::clone(&self.mutable)
    }

    /// Return a reference to the underlying DataFusion `SessionContext`.
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Return a reference to the artifact catalog.
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// Return a reference to the active configuration.
    pub fn config(&self) -> &JammiConfig {
        &self.config
    }
}
