use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_federation::{FederatedQueryPlanner, FederationOptimizerRule};

use crate::audit::{EnvSigningKeyStore, SigningKeyStore};
use crate::catalog::backend::BackendImpl;
use crate::catalog::topic_repo::TopicRepo;
use crate::catalog::Catalog;
use crate::config::{BrokerConfig, CatalogConfig, JammiConfig, SigningKeyConfig};
use crate::error::{JammiError, Result};
use crate::source::mutable::MutableTableRegistry;
use crate::source::registry::SourceCatalog;
use crate::source::schema_provider::JammiSchemaProvider;
use crate::source::{file_format, table_name_from_url, SourceConnection, SourceType};
use crate::storage::{StorageRegistry, StorageUrl};
use crate::store::mutable::definition::{MutableTableDefinition, MutableTableId};
use crate::store::mutable::sqlite::SqliteMutableBackend;
use crate::store::mutable::MutableBackend;
use crate::tenant::{TenantContext, TenantId};
use crate::tenant_scope::{SourceTenantColumns, TenantBinding, TenantScopeAnalyzerRule};
use crate::trigger::{InMemoryBroker, Publisher, Subscriber, TriggerBroker};

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
    /// Process-wide cache of `object_store` drivers keyed by (scheme, bucket).
    /// Used to register `s3://` / `gs://` / `azure://` URLs with both the
    /// engine's own writers and DataFusion's `ListingTable` reader.
    storage_registry: StorageRegistry,
    mutable: Arc<MutableTableRegistry>,
    mutable_schema: Arc<JammiSchemaProvider>,
    /// Broker for the trigger-stream surface. Defaults to [`InMemoryBroker`];
    /// callers that wire a clustered broker (e.g. JetStream) pass it through
    /// [`JammiSession::with_broker`] or [`JammiSession::with_backend_and_broker`].
    trigger_broker: Arc<dyn TriggerBroker>,
    topic_repo: Arc<TopicRepo>,
    publisher: Arc<Publisher>,
    subscriber: Arc<Subscriber>,
    /// Source of the audit HMAC master key. Read via
    /// [`JammiSession::signing_key_store`] by the audit write path. Either
    /// derived from `config.signing_key` (defaulting to [`EnvSigningKeyStore`])
    /// or supplied by the caller at construction via
    /// [`JammiSession::with_backend_broker_and_signing_key`] /
    /// [`JammiSession::with_signing_key_store`], so the sign path routes through
    /// a caller-chosen store.
    signing_key_store: Arc<dyn SigningKeyStore>,
}

impl JammiSession {
    /// Create a new session. The catalog backend and trigger broker are
    /// constructed from `config.catalog` and `config.broker` respectively —
    /// SQLite + in-process is the dev-laptop default, Postgres + JetStream
    /// is the SaaS-deployment pairing.
    ///
    /// Selecting `BrokerConfig::JetStream` without the `jetstream-broker`
    /// cargo feature on `jammi-db` returns [`JammiError::Config`] rather
    /// than panicking — the broker variant is gone from the build, not
    /// merely unreachable.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        let backend = build_backend_from_config(&config).await?;
        let broker = build_broker_from_config(&config).await?;
        Self::with_backend_and_broker(config, backend, broker).await
    }

    /// Create a new session with a caller-supplied trigger broker. The
    /// catalog backend is still resolved from `config.catalog`, so callers
    /// that just want to override the broker (test harnesses arming an
    /// [`InMemoryBroker`] with
    /// [`InMemoryBroker::trigger_failure_for_next_publish`] to exercise
    /// publisher-failure paths) do not have to construct a backend
    /// themselves.
    pub async fn with_broker(
        config: JammiConfig,
        trigger_broker: Arc<dyn TriggerBroker>,
    ) -> Result<Self> {
        let backend = build_backend_from_config(&config).await?;
        Self::with_backend_and_broker(config, backend, trigger_broker).await
    }

    /// Build a session around a caller-supplied catalog backend. Migrations
    /// are applied here so the caller hands in a connected-but-unmigrated
    /// [`crate::catalog::backend::BackendImpl`]; the session takes it from
    /// there. The trigger broker is resolved from `config.broker` — pairs
    /// with [`Self::with_broker`] for tests that want to override one
    /// dimension and keep the other config-driven.
    ///
    /// Used by:
    /// - tests that need to parameterize over backend (SQLite tempfile vs
    ///   live Postgres URL) without going through the path-based
    ///   [`Catalog::open_with_tenant`] surface
    /// - server deployments that compose their own backend (e.g. a shared
    ///   Postgres pool across multiple `jammi-server` replicas)
    pub async fn with_backend(config: JammiConfig, backend: BackendImpl) -> Result<Self> {
        let broker = build_broker_from_config(&config).await?;
        Self::with_backend_and_broker(config, backend, broker).await
    }

    /// Build a session around a caller-supplied catalog backend AND a
    /// caller-supplied trigger broker. The signing-key store is derived from
    /// `config.signing_key` (defaulting to [`EnvSigningKeyStore`]). Server
    /// deployments combining a shared Postgres pool with a JetStream broker
    /// reach for this one.
    pub async fn with_backend_and_broker(
        config: JammiConfig,
        backend: BackendImpl,
        trigger_broker: Arc<dyn TriggerBroker>,
    ) -> Result<Self> {
        let signing_key_store = signing_key_store_from_config(&config);
        Self::with_backend_broker_and_signing_key(
            config,
            backend,
            trigger_broker,
            signing_key_store,
        )
        .await
    }

    /// The most-composable constructor: caller-supplied catalog backend,
    /// trigger broker, AND signing-key store. Every other backend constructor
    /// delegates here, deriving the store from `config.signing_key`; callers
    /// that need the audit sign/verify path to route through a custom store
    /// (e.g. a deployment whose master key lives behind a secrets adapter)
    /// inject it directly through this seam.
    pub async fn with_backend_broker_and_signing_key(
        config: JammiConfig,
        backend: BackendImpl,
        trigger_broker: Arc<dyn TriggerBroker>,
        signing_key_store: Arc<dyn SigningKeyStore>,
    ) -> Result<Self> {
        let config = Arc::new(config);
        let tenant_binding = TenantBinding::unscoped();
        backend.migrate().await?;
        let catalog = Arc::new(Catalog::from_backend_with_tenant(
            backend,
            Some(tenant_binding.clone()),
        ));
        Self::build(
            config,
            catalog,
            tenant_binding,
            trigger_broker,
            signing_key_store,
        )
        .await
    }

    /// Build a session with a caller-supplied signing-key store, resolving the
    /// catalog backend and trigger broker from `config` (the config-driven
    /// counterpart to [`Self::with_broker`]). Deployments that keep the
    /// SQLite/Postgres + broker selection config-driven but need the audit
    /// sign/verify path to route through a custom store reach for this one.
    pub async fn with_signing_key_store(
        config: JammiConfig,
        signing_key_store: Arc<dyn SigningKeyStore>,
    ) -> Result<Self> {
        let backend = build_backend_from_config(&config).await?;
        let broker = build_broker_from_config(&config).await?;
        Self::with_backend_broker_and_signing_key(config, backend, broker, signing_key_store).await
    }

    async fn build(
        config: Arc<JammiConfig>,
        catalog: Arc<Catalog>,
        tenant_binding: TenantBinding,
        trigger_broker: Arc<dyn TriggerBroker>,
        signing_key_store: Arc<dyn SigningKeyStore>,
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
            tenant_binding.clone(),
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
            tenant_binding.clone(),
        ));

        // The "mutable" catalog hosts every registered mutable companion
        // table under a single "public" schema. Register the empty catalog
        // up-front so three-part names resolve before any table is added.
        let mutable_schema = Arc::new(JammiSchemaProvider::new());
        let mutable_catalog = Arc::new(SourceCatalog::new(Arc::clone(&mutable_schema)));
        ctx.register_catalog("mutable", mutable_catalog);

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

        // The session's driver cache carries the deploy-wide `[storage.cloud]`
        // credentials as its default. Every `driver_for(url, None)` — source
        // reads, result-table reads/writes, cleanup — falls back to it, so a
        // wire `AddSource("r2://…")` (no inline creds) and the result root
        // both resolve with the configured credentials.
        let storage_registry = StorageRegistry::with_default_cloud(config.storage.cloud.clone());

        let session = Self {
            ctx,
            catalog,
            config,
            tenant: tenant_binding,
            source_tenant_columns,
            storage_registry,
            mutable,
            mutable_schema,
            trigger_broker,
            topic_repo,
            publisher,
            subscriber,
            signing_key_store,
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
    ///
    /// This is the **sticky** binding form: it mutates session-shared state
    /// and stays in effect until [`Self::unbind_tenant`] or another bind
    /// call replaces it. For concurrent multi-tenant request handlers on
    /// the same session, prefer [`Self::with_tenant_scoped`].
    pub fn with_tenant(self, t: TenantId) -> Self {
        self.tenant.set_shared(TenantContext::Scoped(t));
        self
    }

    /// Return the tenant scope effective on this session for the current
    /// task: the task-local override installed by [`Self::with_tenant_scoped`]
    /// when called from inside such a closure, otherwise the sticky shared
    /// value set by [`Self::with_tenant`] / [`Self::bind_tenant`].
    pub fn tenant(&self) -> Option<TenantId> {
        self.tenant.current_tenant()
    }

    /// Shared handle to the underlying tenant binding. Used by server-side
    /// composition (Flight SQL + gRPC) to plumb the per-session tenant value
    /// from a `TenantBoundProvider` / interceptor (in the `jammi-server`
    /// crate) into the engine without re-creating the binding for every
    /// request.
    pub fn tenant_binding_arc(&self) -> crate::tenant_scope::TenantBinding {
        self.tenant.clone()
    }

    /// Mutate the sticky bound tenant in place. Equivalent to
    /// [`Self::with_tenant`] but does not consume `self` — needed for callers
    /// that hold the session behind a shared reference (`Arc<JammiSession>`
    /// in PyO3 / CLI wrappers, the per-request gRPC handler reading
    /// `SessionTenant` from the request interceptor).
    ///
    /// Like [`Self::with_tenant`], this writes session-shared state and is
    /// the wrong primitive for concurrent multi-tenant request handlers on
    /// one shared session — see [`Self::with_tenant_scoped`].
    pub fn bind_tenant(&self, t: TenantId) {
        self.tenant.set_shared(TenantContext::Scoped(t));
    }

    /// Clear the sticky bound tenant.
    pub fn unbind_tenant(&self) {
        self.tenant.set_shared(TenantContext::Unscoped);
    }

    /// Run `f` with `tenant` bound for the duration of the closure's future.
    ///
    /// The binding lives in a Tokio task-local installed for the executing
    /// task; concurrent invocations from different tasks each see their own
    /// `tenant` with no shared-state mutation between them. Reads inside the
    /// closure — analyzer-rule predicate injection, catalog
    /// `tenant_id`-aware queries, mutable-table sink / provider — observe
    /// `tenant` regardless of what any other concurrent caller has set on
    /// the sticky shared binding.
    ///
    /// Use this from concurrent gRPC handlers that share one
    /// `Arc<JammiSession>` across requests. The sticky
    /// [`Self::bind_tenant`] / [`Self::with_tenant`] surface remains for
    /// single-flight callers (CLI, single-tenant embedding), but does not
    /// fix the race when two handlers from different tenants overlap on a
    /// shared session.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use jammi_db::session::JammiSession;
    /// use jammi_db::TenantId;
    ///
    /// let session: Arc<JammiSession> = unimplemented!();
    /// let tenant: TenantId = unimplemented!();
    /// let rows = session
    ///     .with_tenant_scoped(tenant, |scope| async move {
    ///         scope.sql("SELECT COUNT(*) FROM mutable.public.widgets").await
    ///     })
    ///     .await?;
    /// # let _ = rows;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Implementation note
    ///
    /// This is **Option β** from the design: rather than threading an
    /// explicit `tenant` argument through every internal call (catalog,
    /// mutable-table registry, analyzer rule), the binding's read path
    /// consults a [`tokio::task_local`] override that
    /// [`crate::tenant_scope::TenantBinding::scope`] installs for the
    /// duration of `f`. The shared `Arc<RwLock<TenantContext>>` continues
    /// to serve the sticky-binding path; the task-local shadows it for
    /// concurrent scoped callers.
    pub async fn with_tenant_scoped<'a, F, Fut, T>(&'a self, tenant: TenantId, f: F) -> T
    where
        F: FnOnce(TenantScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        let scope = TenantScope { session: self };
        self.tenant
            .scope(TenantContext::Scoped(tenant), f(scope))
            .await
    }

    /// Run `f` with the tenant analyzer rule disabled for the duration of
    /// the closure's future.
    ///
    /// Intended for cross-tenant administrative scans — e.g. server-startup
    /// recovery enumerating work-in-progress rows in a mutable companion
    /// table across every tenant so each can be re-bound and resumed.
    /// Inside the closure, SQL queries against tenant-aware tables return
    /// rows from every tenant (and from globally-scoped rows whose
    /// `tenant_id IS NULL`); the mutable-table provider's backend SQL also
    /// drops its tenant filter. The caller is responsible for re-binding
    /// the session to the row's `tenant_id` (via [`Self::with_tenant_scoped`])
    /// before issuing any follow-up tenant-scoped work.
    ///
    /// This is **not** a general-purpose escape hatch. It is auditable, it
    /// must not be exposed on the gRPC wire surface, and the closure must
    /// fully consume any data it intends to use across tenants — the
    /// bypass marker is task-local and clears the instant `f`'s future
    /// resolves.
    ///
    /// Concurrent tasks each carry their own admin-scope marker; an
    /// admin-scope on one task never bleeds into a tenant-scoped request
    /// running on another task that shares the same `Arc<JammiSession>`.
    pub async fn with_admin_scope<'a, F, Fut, T>(&'a self, f: F) -> T
    where
        F: FnOnce(AdminScope<'a>) -> Fut,
        Fut: std::future::Future<Output = T> + 'a,
    {
        let scope = AdminScope { session: self };
        TenantBinding::admin_scope(f(scope)).await
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
        let sources = self.catalog.list_all_sources().await?;
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
            SourceType::File => {
                let format = connection
                    .format
                    .as_ref()
                    .ok_or_else(|| JammiError::Config("File source requires a format".into()))?;
                let raw_url = connection
                    .url
                    .as_deref()
                    .ok_or_else(|| JammiError::Config("File source requires a URL".into()))?;
                let url = StorageUrl::parse(raw_url)?;
                let state = self.ctx.state();
                let table = file_format::create_listing_table(
                    &self.ctx,
                    &self.storage_registry,
                    &url,
                    format,
                    connection.file_extension.as_deref(),
                    connection.cloud.as_ref(),
                    &state,
                )
                .await?;
                let name = table_name_from_url(url.as_str());
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

        // 2. Delete Parquet objects and sidecar bundles via the storage
        //    layer. 404 is not an error — same best-effort semantics as
        //    before, now portable across `file://` / `s3://` / `gs://`.
        for rt in &result_tables {
            let parquet_url = match StorageUrl::parse(&rt.parquet_path) {
                Ok(u) => u,
                Err(e) => {
                    tracing::warn!(
                        path = %rt.parquet_path,
                        error = %e,
                        "Invalid parquet path on result-table row, skipping delete"
                    );
                    continue;
                }
            };
            let driver = match self.storage_registry.driver_for(&parquet_url, None) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(path = %parquet_url, error = %e, "Driver init failed during cleanup");
                    continue;
                }
            };
            let handle = crate::storage::JammiObjectStore::new(driver, parquet_url.clone());
            if let Ok(path) = handle.data_path() {
                if let Err(e) = handle.delete_if_exists(&path).await {
                    tracing::warn!(path = %parquet_url, error = %e, "Failed to delete parquet object");
                }
            }
            if let Some(idx) = &rt.index_path {
                let idx_url = match StorageUrl::parse(idx) {
                    Ok(u) => u,
                    Err(e) => {
                        tracing::warn!(path = %idx, error = %e, "Invalid index path on result-table row, skipping delete");
                        continue;
                    }
                };
                let idx_driver = match self.storage_registry.driver_for(&idx_url, None) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(path = %idx_url, error = %e, "Index driver init failed during cleanup");
                        continue;
                    }
                };
                let idx_handle = crate::storage::JammiObjectStore::new(idx_driver, idx_url.clone());
                if let Err(e) = crate::storage::sidecar_layout::delete_sidecar(
                    &idx_handle,
                    crate::storage::sidecar_layout::SidecarKind::Ann,
                )
                .await
                {
                    tracing::warn!(path = %idx_url, error = %e, "Failed to delete sidecar bundle");
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
    /// The Flight SQL surface carries query + data-DML only (per ADR-01 §3.2);
    /// topic lifecycle is the typed `register_topic` / `drop_topic` surface, not
    /// a SQL statement.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    /// Shared handle to the trigger broker the session was constructed with.
    pub fn trigger_broker(&self) -> Arc<dyn TriggerBroker> {
        Arc::clone(&self.trigger_broker)
    }

    /// Shared handle to the audit signing-key store the session was constructed
    /// with. The audit write path reads this to obtain the HMAC master key.
    pub fn signing_key_store(&self) -> Arc<dyn SigningKeyStore> {
        Arc::clone(&self.signing_key_store)
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
    ///
    /// Reserved system table names (the `_jammi_` prefix) cannot be created
    /// through this user-facing path. The substrate owns those tables and
    /// creates them implicitly (e.g. the per-query audit table
    /// `_jammi_search_audit` is created by the audit module). Allowing a user
    /// to pre-create them would let writes bypass substrate-enforced invariants
    /// such as audit signing.
    pub async fn create_mutable_table(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        if crate::audit::is_reserved_table_name(def.id.as_str()) {
            return Err(JammiError::MutableTable(
                crate::store::mutable::MutableTableError::InvalidId(format!(
                    "table name '{}' is reserved for the Jammi substrate; names \
                     beginning with '_jammi_' are created implicitly and must not \
                     be created by users",
                    def.id.as_str()
                )),
            ));
        }
        self.create_mutable_table_unchecked(def).await
    }

    /// Register a mutable companion table without the reserved-name guard.
    ///
    /// Internal to the substrate: used by system components (e.g. the audit
    /// primitive) that legitimately own a reserved `_jammi_*` table. Not part
    /// of the user-facing surface.
    pub async fn register_mutable_table_unchecked(
        &self,
        def: MutableTableDefinition,
    ) -> Result<MutableTableId> {
        self.create_mutable_table_unchecked(def).await
    }

    async fn create_mutable_table_unchecked(
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

    /// Typed handle to the per-query audit primitive, scoped to this session's
    /// tenant. See [`crate::audit`].
    pub fn audit(&self) -> crate::audit::AuditHandle<'_> {
        crate::audit::AuditHandle::new(self)
    }

    /// Return a reference to the underlying DataFusion `SessionContext`.
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Return a reference to the artifact catalog.
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// Shared handle to the storage registry. Components that write
    /// Jammi-owned artifacts (result Parquet, sidecar indexes) consult
    /// this to resolve a [`crate::storage::JammiObjectStore`] for any
    /// URL the session has already seen.
    pub fn storage_registry(&self) -> StorageRegistry {
        self.storage_registry.clone()
    }

    /// Return a reference to the active configuration.
    pub fn config(&self) -> &JammiConfig {
        &self.config
    }

    /// Read the `vector` column of an embedding result table into one
    /// `Vec<f32>` per row.
    ///
    /// Resolves the table's `parquet_path` through the session's
    /// [`StorageRegistry`] so cloud credentials registered with the session
    /// are inherited; opens the underlying object via
    /// [`crate::storage::JammiObjectStore`] and streams the column through
    /// the engine's typed-vector reader.
    ///
    /// Surfaces [`JammiError::Schema`] when the table's parquet does not
    /// carry a `vector` column shaped `FixedSizeList<Float32>`, so callers
    /// see a typed signal instead of a panic on the downcast.
    pub async fn read_vectors(
        &self,
        table: &crate::catalog::result_repo::ResultTableRecord,
    ) -> Result<Vec<Vec<f32>>> {
        let url = StorageUrl::parse(&table.parquet_path)?;
        let driver = self.storage_registry.driver_for(&url, None)?;
        let handle = crate::storage::JammiObjectStore::new(driver, url);
        crate::store::vectors::read_fixed_size_list_f32_column(&handle, &table.table_name, "vector")
            .await
    }

    /// Read a single row's stored `vector` from an embedding result table by
    /// its `_row_id` (the key-column value set at embedding time).
    ///
    /// Scans the registered `jammi.{table_name}` table through DataFusion with
    /// a typed equality filter (no SQL string interpolation of the key, so an
    /// arbitrary key is not an injection vector) and extracts the one
    /// `FixedSizeList<Float32>` cell. Returns [`JammiError::Catalog`] when no
    /// row matches the key, and [`JammiError::Schema`] when the `vector`
    /// column is not shaped `FixedSizeList<Float32>` — the same typed signal
    /// [`Self::read_vectors`] gives. The vector stays inside the engine; this
    /// is the resolver behind `search_by_id`'s query-by-example path.
    pub async fn read_vector_by_key(
        &self,
        table: &crate::catalog::result_repo::ResultTableRecord,
        row_key: &str,
    ) -> Result<Vec<f32>> {
        use datafusion::prelude::{col, lit};
        use datafusion::sql::TableReference;

        // Result tables register under the single bare identifier
        // `jammi.{name}`; reach this one the same way rather than as a string
        // DataFusion would re-parse into a `jammi` schema reference (see
        // `register_parquet_table`).
        let table_ref = TableReference::bare(format!("jammi.{}", table.table_name));
        let batches = self
            .ctx
            .table(table_ref.clone())
            .await
            .map_err(|e| JammiError::Other(format!("Resolve embedding table '{table_ref}': {e}")))?
            .filter(col("_row_id").eq(lit(row_key)))
            .map_err(|e| JammiError::Other(format!("Vector-by-key filter: {e}")))?
            .select_columns(&["vector"])
            .map_err(|e| JammiError::Other(format!("Vector-by-key projection: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::Other(format!("Vector-by-key scan: {e}")))?;

        let mut out: Vec<Vec<f32>> = Vec::new();
        for batch in &batches {
            crate::store::vectors::extend_with_fixed_size_list_f32(
                batch,
                &table.table_name,
                "vector",
                &mut out,
            )?;
        }
        out.into_iter().next().ok_or_else(|| {
            JammiError::Catalog(format!(
                "no row with key '{row_key}' in embedding table '{}'",
                table.table_name
            ))
        })
    }
}

/// Resolve the signing-key store selected by `config.signing_key`. The single
/// variant today resolves to [`EnvSigningKeyStore`]; callers that need a
/// different store inject it through
/// [`JammiSession::with_backend_broker_and_signing_key`] rather than extending
/// this match.
fn signing_key_store_from_config(config: &JammiConfig) -> Arc<dyn SigningKeyStore> {
    match config.signing_key {
        SigningKeyConfig::Env => Arc::new(EnvSigningKeyStore),
    }
}

/// Build a [`BackendImpl`] from `config.catalog`, honouring the SQLite path
/// default and the Postgres pool options.
async fn build_backend_from_config(config: &JammiConfig) -> Result<BackendImpl> {
    match &config.catalog {
        CatalogConfig::Sqlite { path } => {
            let db_path = match path {
                Some(p) => p.clone(),
                None => {
                    std::fs::create_dir_all(&config.artifact_dir)?;
                    config.artifact_dir.join("catalog.db")
                }
            };
            Ok(BackendImpl::sqlite_from_path(&db_path).await?)
        }
        CatalogConfig::Postgres {
            url,
            pool_size,
            max_lifetime_secs,
        } => Ok(BackendImpl::postgres_from_url(url, *pool_size, *max_lifetime_secs).await?),
    }
}

/// Build a trigger broker from `config.broker`. JetStream requires the
/// `jetstream-broker` cargo feature; selecting it without the feature
/// returns a typed [`JammiError::Config`] rather than panicking.
async fn build_broker_from_config(config: &JammiConfig) -> Result<Arc<dyn TriggerBroker>> {
    match &config.broker {
        BrokerConfig::InMemory => Ok(Arc::new(InMemoryBroker::new())),
        BrokerConfig::JetStream {
            url,
            retention_seconds,
            credentials_path,
        } => build_jetstream_broker(url, *retention_seconds, credentials_path.as_deref()).await,
    }
}

#[cfg(feature = "jetstream-broker")]
async fn build_jetstream_broker(
    url: &str,
    retention_seconds: u64,
    credentials_path: Option<&std::path::Path>,
) -> Result<Arc<dyn TriggerBroker>> {
    let js = match credentials_path {
        Some(p) => {
            crate::trigger::jetstream::JetStreamBroker::connect_with_credentials(
                url,
                retention_seconds,
                p,
            )
            .await?
        }
        None => crate::trigger::jetstream::JetStreamBroker::connect(url, retention_seconds).await?,
    };
    Ok(Arc::new(js))
}

#[cfg(not(feature = "jetstream-broker"))]
async fn build_jetstream_broker(
    _url: &str,
    _retention_seconds: u64,
    _credentials_path: Option<&std::path::Path>,
) -> Result<Arc<dyn TriggerBroker>> {
    Err(JammiError::Config(
        "broker.kind = \"jet_stream\" requires the `jetstream-broker` cargo feature on jammi-db"
            .into(),
    ))
}

/// Handle passed to the closure inside [`JammiSession::with_tenant_scoped`].
///
/// `Deref`s to the underlying [`JammiSession`] so the closure can call any
/// session method (`sql`, `mutable_tables`, `catalog`, …) and observe the
/// scoped tenant. Holding a `TenantScope` does not by itself bind anything —
/// the binding is installed by the surrounding `with_tenant_scoped` call's
/// task-local. The scope's lifetime is tied to the closure's stack frame so
/// it cannot leak past the scoped region.
pub struct TenantScope<'a> {
    session: &'a JammiSession,
}

impl<'a> TenantScope<'a> {
    /// Borrow the underlying session.
    pub fn session(&self) -> &'a JammiSession {
        self.session
    }
}

impl<'a> std::ops::Deref for TenantScope<'a> {
    type Target = JammiSession;

    fn deref(&self) -> &Self::Target {
        self.session
    }
}

/// Handle passed to the closure inside [`JammiSession::with_admin_scope`].
///
/// The presence of this handle signals that the surrounding closure is
/// running inside the admin-bypass task-local: every SQL query issued
/// through [`Self::sql`] yields rows across every tenant, and the
/// mutable-table provider drops its backend-SQL tenant filter for the
/// duration of the scope.
///
/// Construction is private to this module so the type cannot be forged: an
/// `AdminScope` only exists for the lifetime of one `with_admin_scope`
/// closure. The closure must finish consuming any cross-tenant data before
/// it returns — once the scope drops, subsequent reads on the same session
/// are tenant-filtered again, even if a stream from inside leaks past the
/// closure boundary.
pub struct AdminScope<'a> {
    session: &'a JammiSession,
}

impl<'a> AdminScope<'a> {
    /// Execute a SQL query with the tenant analyzer rule disabled.
    ///
    /// Returns the fully materialised `Vec<RecordBatch>` so the cross-tenant
    /// data is collected inside the admin scope; the caller cannot
    /// accidentally hold an unresolved stream past the closure boundary.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.session.sql(query).await
    }

    /// Borrow the underlying session. Use sparingly: any method called
    /// through this borrow that consults the tenant binding will observe
    /// the bypass.
    pub fn session(&self) -> &'a JammiSession {
        self.session
    }
}
