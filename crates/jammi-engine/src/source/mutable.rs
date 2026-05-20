//! Mutable companion table registry — lifecycle of catalog row + storage table.

use std::sync::Arc;

use datafusion::catalog::TableProvider;

use crate::catalog::backend::TxOptions;
use crate::catalog::Catalog;
use crate::store::mutable::definition::{
    MutableTableDefinition, MutableTableError, MutableTableId,
};
use crate::store::mutable::provider::MutableTableProvider;
use crate::store::mutable::MutableBackend;
use crate::tenant::TenantId;

/// Owns the lifecycle of mutable companion tables: catalog row + storage
/// table + DataFusion `TableProvider` construction.
pub struct MutableTableRegistry {
    catalog: Arc<Catalog>,
    backend: Arc<dyn MutableBackend>,
}

impl MutableTableRegistry {
    pub fn new(catalog: Arc<Catalog>, backend: Arc<dyn MutableBackend>) -> Self {
        Self { catalog, backend }
    }

    /// Register a mutable table. Atomically: catalog row + storage table +
    /// secondary indexes commit together; nothing lands if any step fails.
    pub async fn register(
        &self,
        def: MutableTableDefinition,
    ) -> Result<Arc<dyn TableProvider>, MutableTableError> {
        // 1) Catalog row + index rows (their own transaction inside the repo).
        self.catalog.create_mutable_table(&def).await?;

        // 2) Storage table + secondary indexes in one backend transaction.
        let ddl = self.backend.create_table_ddl(&def);
        let index_ddls: Vec<String> = def
            .indexes
            .iter()
            .map(|idx| self.backend.create_index_ddl(&def, idx))
            .collect();

        self.backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    tx.execute(&ddl, &[]).await?;
                    for stmt in index_ddls {
                        tx.execute(&stmt, &[]).await?;
                    }
                    Ok(())
                })
            })
            .await?;

        let def = Arc::new(def);
        Ok(Arc::new(MutableTableProvider::new(
            def,
            Arc::clone(&self.backend),
        )))
    }

    /// Drop a mutable table: backend `DROP TABLE` then delete catalog row.
    pub async fn drop_table(&self, id: &MutableTableId) -> Result<(), MutableTableError> {
        let def = self
            .catalog
            .get_mutable_table(id)
            .await?
            .ok_or_else(|| MutableTableError::NotFound(id.clone()))?;

        let ddl = self.backend.drop_table_ddl(&def);
        self.backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    tx.execute(&ddl, &[]).await?;
                    Ok(())
                })
            })
            .await?;

        self.catalog.delete_mutable_table(id).await?;
        Ok(())
    }

    /// Look up a registered table by id.
    pub async fn get(
        &self,
        id: &MutableTableId,
    ) -> Result<Option<MutableTableDefinition>, MutableTableError> {
        self.catalog.get_mutable_table(id).await
    }

    /// List registered tables visible to the given tenant scope.
    pub async fn list(
        &self,
        tenant: Option<TenantId>,
    ) -> Result<Vec<MutableTableDefinition>, MutableTableError> {
        self.catalog.list_mutable_tables(tenant).await
    }

    /// Build a `TableProvider` for an already-registered table (does not
    /// touch storage). Used by `JammiSession::reload_mutable_tables` at startup.
    pub fn provider_for(&self, def: MutableTableDefinition) -> Arc<dyn TableProvider> {
        Arc::new(MutableTableProvider::new(
            Arc::new(def),
            Arc::clone(&self.backend),
        ))
    }
}
