use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, SchemaProvider};
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result};

/// DataFusion [`SchemaProvider`] holding an in-process set of table
/// providers: the `mutable` catalog's companion tables, added and removed as
/// they are created and dropped.
pub struct JammiSchemaProvider {
    tables: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
}

impl std::fmt::Debug for JammiSchemaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JammiSchemaProvider")
            .field("tables", &self.table_names())
            .finish()
    }
}

impl Default for JammiSchemaProvider {
    fn default() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
        }
    }
}

impl JammiSchemaProvider {
    /// Create an empty schema provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a table provider under the given name.
    pub fn add_table(&self, name: String, table: Arc<dyn TableProvider>) -> Result<()> {
        self.tables
            .write()
            .map_err(|e| DataFusionError::Internal(format!("Lock poisoned: {e}")))?
            .insert(name, table);
        Ok(())
    }

    /// Remove one table by name. Returns the dropped provider if present.
    pub fn remove_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(self
            .tables
            .write()
            .map_err(|e| DataFusionError::Internal(format!("Lock poisoned: {e}")))?
            .remove(name))
    }
}

#[async_trait]
impl SchemaProvider for JammiSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables
            .read()
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_else(|e| {
                tracing::error!("Lock poisoned in table_names: {e}");
                Vec::new()
            })
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        let guard = self
            .tables
            .read()
            .map_err(|e| DataFusionError::Internal(format!("Lock poisoned: {e}")))?;
        Ok(guard.get(name).cloned())
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables
            .read()
            .map(|t| t.contains_key(name))
            .unwrap_or_else(|e| {
                tracing::error!("Lock poisoned in table_exist: {e}");
                false
            })
    }
}

/// DataFusion [`CatalogProvider`] that exposes exactly one schema, `public`.
///
/// Every data source and the `mutable` catalog are catalogs of this shape,
/// so a table is addressable as `<catalog>.public.<table>`.
pub(crate) struct PublicSchemaCatalog {
    schema: Arc<dyn SchemaProvider>,
}

impl std::fmt::Debug for PublicSchemaCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicSchemaCatalog")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PublicSchemaCatalog {
    /// Wrap a schema provider as a catalog with a single `public` schema.
    pub(crate) fn new(schema: Arc<dyn SchemaProvider>) -> Self {
        Self { schema }
    }
}

impl CatalogProvider for PublicSchemaCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec!["public".to_string()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        (name == "public").then(|| Arc::clone(&self.schema))
    }
}
