use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::catalog::Catalog;
use crate::config::JammiConfig;
use crate::error::{JammiError, Result};
use crate::source::registry::SourceCatalog;
use crate::source::schema_provider::JammiSchemaProvider;
use crate::source::{local, table_name_from_url, SourceConnection, SourceType};

/// The main entry point. Wraps a DataFusion SessionContext with Jammi's
/// source registry, catalog, and configuration.
pub struct JammiSession {
    ctx: SessionContext,
    catalog: Arc<Catalog>,
    config: Arc<JammiConfig>,
}

impl JammiSession {
    /// Create a new session.
    pub async fn new(config: JammiConfig) -> Result<Self> {
        let config = Arc::new(config);
        let catalog = Arc::new(Catalog::open(&config.artifact_dir)?);

        let session_config = SessionConfig::new()
            .with_target_partitions(config.engine.execution_threads)
            .with_batch_size(config.engine.batch_size);
        let ctx = SessionContext::new_with_config(session_config);

        Ok(Self {
            ctx,
            catalog,
            config,
        })
    }

    /// Register a data source.
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

        // Create the table provider for local sources.
        let table = match &source_type {
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
                local::create_listing_table(
                    url,
                    format,
                    connection.file_extension.as_deref(),
                    &state,
                )
                .await?
            }
            other => {
                return Err(JammiError::Config(format!(
                    "Source type {other:?} not yet supported"
                )));
            }
        };

        // Persist to catalog.
        self.catalog
            .register_source(source_id, source_type, &connection)?;

        // Derive table name from URL and build schema provider.
        let table_name = table_name_from_url(connection.url.as_deref().unwrap_or("data"));
        let schema_provider = Arc::new(JammiSchemaProvider::new());
        schema_provider.add_table(table_name, table);

        // Register as a named catalog so queries use `source_id.public.table_name`.
        let source_catalog = Arc::new(SourceCatalog::new(schema_provider));
        self.ctx.register_catalog(source_id, source_catalog);

        Ok(())
    }

    /// Execute a SQL query, return results as RecordBatches.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    /// Access the underlying DataFusion SessionContext.
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Access the catalog.
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Access the config.
    pub fn config(&self) -> &JammiConfig {
        &self.config
    }
}
