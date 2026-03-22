use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::TableProvider;
use datafusion::error::Result;

/// DataFusion [`SchemaProvider`] that holds the table providers for a single data source.
///
/// Tables are added at registration time and looked up by name during query planning.
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
    pub fn add_table(&self, name: String, table: Arc<dyn TableProvider>) {
        self.tables.write().unwrap().insert(name, table);
    }
}

#[async_trait]
impl SchemaProvider for JammiSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.tables.read().unwrap().keys().cloned().collect()
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.read().unwrap().get(name).cloned())
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.read().unwrap().contains_key(name)
    }
}
