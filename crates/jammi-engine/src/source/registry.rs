use std::any::Any;
use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, SchemaProvider};

use super::schema_provider::JammiSchemaProvider;

/// A thin CatalogProvider wrapping a single JammiSchemaProvider.
/// Each registered source gets its own SourceCatalog, exposed as the "public" schema.
pub struct SourceCatalog {
    schema: Arc<JammiSchemaProvider>,
}

impl std::fmt::Debug for SourceCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCatalog")
            .field("schema", &self.schema)
            .finish()
    }
}

impl SourceCatalog {
    pub fn new(schema: Arc<JammiSchemaProvider>) -> Self {
        Self { schema }
    }
}

impl CatalogProvider for SourceCatalog {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema_names(&self) -> Vec<String> {
        vec!["public".to_string()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        if name == "public" {
            Some(Arc::clone(&self.schema) as Arc<dyn SchemaProvider>)
        } else {
            None
        }
    }
}
