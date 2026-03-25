use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_federation::{FederatedQueryPlanner, FederationOptimizerRule};

use crate::catalog::Catalog;
use crate::config::JammiConfig;
use crate::error::{JammiError, Result};
use crate::source::registry::SourceCatalog;
use crate::source::schema_provider::JammiSchemaProvider;
use crate::source::{local, table_name_from_url, SourceConnection, SourceType};

/// Primary entry point for the Jammi query engine.
///
/// Wraps a DataFusion `SessionContext` with source registration,
/// an artifact catalog, and platform configuration.
pub struct JammiSession {
    ctx: SessionContext,
    catalog: Arc<Catalog>,
    config: Arc<JammiConfig>,
}

impl JammiSession {
    /// Create a new session, opening the catalog and configuring DataFusion.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        let config = Arc::new(config);
        let catalog = Arc::new(Catalog::open(&config.artifact_dir)?);

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

        let federated_state = SessionStateBuilder::new_from_existing(base_state)
            .with_optimizer_rules(rules)
            .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
            .build();

        let ctx = SessionContext::new_with_state(federated_state);

        Ok(Self {
            ctx,
            catalog,
            config,
        })
    }

    /// Register a data source, persisting it to the catalog and exposing it to SQL queries.
    pub async fn add_source(
        &self,
        source_id: &str,
        source_type: SourceType,
        connection: SourceConnection,
    ) -> Result<()> {
        // Check for duplicate registration.
        if self.catalog.get_source(source_id)?.is_some() {
            return Err(JammiError::Source {
                source_id: source_id.into(),
                message: "Source already registered".into(),
            });
        }

        // Create (table_name, TableProvider) pairs depending on source type.
        let tables: Vec<(String, Arc<dyn datafusion::catalog::TableProvider>)> =
            match &source_type {
                SourceType::Local => {
                    let format = connection.format.as_ref().ok_or_else(|| {
                        JammiError::Config("Local source requires a format".into())
                    })?;
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
                    crate::source::postgres::create_postgres_tables(source_id, &connection).await?
                }
                #[cfg(not(feature = "postgres"))]
                SourceType::Postgres => {
                    return Err(JammiError::Config(
                        "Postgres support requires the 'postgres' feature flag".into(),
                    ));
                }
                #[cfg(feature = "mysql")]
                SourceType::Mysql => {
                    crate::source::mysql::create_mysql_tables(source_id, &connection).await?
                }
                #[cfg(not(feature = "mysql"))]
                SourceType::Mysql => {
                    return Err(JammiError::Config(
                        "MySQL support requires the 'mysql' feature flag".into(),
                    ));
                }
            };

        // Persist to catalog.
        self.catalog
            .register_source(source_id, source_type, &connection)?;

        // Register all discovered tables under one schema provider.
        let schema_provider = Arc::new(JammiSchemaProvider::new());
        for (table_name, table) in tables {
            schema_provider
                .add_table(table_name, table)
                .map_err(|e| JammiError::Source {
                    source_id: source_id.into(),
                    message: format!("Failed to register table: {e}"),
                })?;
        }

        // Register as a named catalog so queries use `source_id.public.table_name`.
        let source_catalog = Arc::new(SourceCatalog::new(schema_provider));
        self.ctx.register_catalog(source_id, source_catalog);

        Ok(())
    }

    /// Remove a source and all associated state: catalog entries, result tables,
    /// disk files (Parquet + index), ANN cache, and DataFusion registration.
    ///
    /// Eval runs are preserved as immutable historical records.
    pub fn remove_source(&self, source_id: &str) -> Result<()> {
        // 1. Fetch result tables so we know which files to delete.
        let result_tables = self.catalog.delete_result_tables_for_source(source_id)?;

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
        self.catalog.remove_source(source_id)?;

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
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(query).await?;
        Ok(df.collect().await?)
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
