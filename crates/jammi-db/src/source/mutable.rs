//! Mutable companion table registry — lifecycle of catalog row + storage table.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::catalog::TableProvider;
use futures::Stream;

use crate::catalog::backend::{BackendError, Transaction, TxOptions};
use crate::catalog::mutable_repo::{delete_mutable_table_rows, MutableRowPayload};
use crate::catalog::Catalog;
use crate::store::mutable::definition::{
    MutableTableDefinition, MutableTableError, MutableTableId,
};
use crate::store::mutable::provider::MutableTableProvider;
use crate::store::mutable::sink::batch_to_params;
use crate::store::mutable::MutableBackend;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// Owns the lifecycle of mutable companion tables: catalog row + storage
/// table + DataFusion `TableProvider` construction.
pub struct MutableTableRegistry {
    catalog: Arc<Catalog>,
    backend: Arc<dyn MutableBackend>,
    tenant: TenantBinding,
}

impl MutableTableRegistry {
    pub fn new(
        catalog: Arc<Catalog>,
        backend: Arc<dyn MutableBackend>,
        tenant: TenantBinding,
    ) -> Self {
        Self {
            catalog,
            backend,
            tenant,
        }
    }

    /// Shared handle to the DDL/DML-rendering backend. Lets the trigger-stream
    /// [`TopicRepo`](crate::catalog::topic_repo::TopicRepo) provision a backing
    /// table inside its own topic-row transaction via `register_in_tx` /
    /// `drop_in_tx` without going through [`Self::register`] (which would open
    /// a second transaction).
    pub fn backend_arc(&self) -> Arc<dyn MutableBackend> {
        Arc::clone(&self.backend)
    }

    /// Register a mutable table. The catalog registration row, its index rows,
    /// the storage `CREATE TABLE`, and the secondary-index DDL all run in ONE
    /// backend transaction via `register_in_tx`. The mutable storage tables
    /// live in the catalog's own database, so a single transaction spans both
    /// the catalog row and the backing table: either every step commits or
    /// nothing lands. A crash mid-register can never leave a catalog row without
    /// its backing table, or a backing table without its catalog row.
    pub async fn register(
        &self,
        def: MutableTableDefinition,
    ) -> Result<Arc<dyn TableProvider>, MutableTableError> {
        let def = Arc::new(def);
        let def_for_tx = Arc::clone(&def);
        let backend = Arc::clone(&self.backend);
        self.backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    register_in_tx(backend.as_ref(), tx, &def_for_tx)
                        .await
                        .map_err(map_mutable_err)?;
                    #[cfg(feature = "test-hooks")]
                    crate::store::mutable::test_hook::maybe_signal_lifecycle("register").await;
                    Ok(())
                })
            })
            .await?;

        Ok(Arc::new(MutableTableProvider::new(
            def,
            Arc::clone(&self.backend),
            self.tenant.clone(),
        )))
    }

    /// Drop a mutable table: storage `DROP TABLE` + catalog-row delete in ONE
    /// backend transaction via `drop_in_tx`. The storage table and the
    /// catalog/index rows share the catalog's database, so both vanish together
    /// or neither does — a crash mid-drop never strands a catalog row pointing
    /// at a missing table.
    pub async fn drop_table(&self, id: &MutableTableId) -> Result<(), MutableTableError> {
        let def = self
            .catalog
            .get_mutable_table(id)
            .await?
            .ok_or_else(|| MutableTableError::NotFound(id.clone()))?;
        // Resolve tenant once, outside the closure, so the strict delete
        // predicate carries the session's scope into the transaction.
        let tenant = self.catalog.current_tenant();
        let backend = Arc::clone(&self.backend);
        self.backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    drop_in_tx(backend.as_ref(), tx, &def, tenant)
                        .await
                        .map_err(map_mutable_err)?;
                    #[cfg(feature = "test-hooks")]
                    crate::store::mutable::test_hook::maybe_signal_lifecycle("drop_table").await;
                    Ok(())
                })
            })
            .await?;
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

    /// List every registered table across all tenants. Used at session
    /// startup to register a `TableProvider` for each persisted mutable
    /// table so DataFusion can resolve `mutable.public.<id>` regardless of
    /// which tenant the session later binds to; per-row tenant filtering
    /// is then applied by the tenant-scope analyzer at query time.
    pub async fn list_all(&self) -> Result<Vec<MutableTableDefinition>, MutableTableError> {
        self.catalog.list_all_mutable_tables().await
    }

    /// Build a `TableProvider` for an already-registered table (does not
    /// touch storage). Used by `JammiSession::reload_mutable_tables` at startup.
    pub fn provider_for(&self, def: MutableTableDefinition) -> Arc<dyn TableProvider> {
        Arc::new(MutableTableProvider::new(
            Arc::new(def),
            Arc::clone(&self.backend),
            self.tenant.clone(),
        ))
    }

    /// Borrow the registry's tenant binding. Used by the subscriber's
    /// non-scoped entry points to resolve the current tenant from the
    /// session at subscribe time, before threading it explicitly through
    /// the replay path.
    pub fn binding(&self) -> &TenantBinding {
        &self.tenant
    }

    /// Append a `RecordBatch` to a mutable table without going through
    /// DataFusion's planner.
    ///
    /// The caller owns the [`Transaction`]; this lets a single unit of work
    /// (e.g. Phase 4's trigger-stream publish path) insert into a backing
    /// table and update related catalog state in one atomic step. Schema
    /// must match the registered definition exactly. The tenant bound on
    /// `tx` is asserted via [`Transaction::assert_tenant_matches`] and
    /// stored on every row's `tenant_id` slot — caller is responsible for
    /// having bound the session tenant before invoking.
    pub async fn insert_batch(
        &self,
        tx: &mut Transaction<'_>,
        table: &MutableTableId,
        batch: &RecordBatch,
    ) -> Result<u64, MutableTableError> {
        // Resolve the table definition under the transaction's bound tenant,
        // not the session binding. The caller (e.g. the trigger publish path)
        // has already authorized the write and bound the owning tenant on the
        // transaction; the backing table is stamped with that tenant, so the
        // lookup must scope to it rather than whatever the request-time session
        // binding happens to be.
        let def = self
            .catalog
            .get_mutable_table_for_tenant(table, tx.tenant())
            .await?
            .ok_or_else(|| MutableTableError::NotFound(table.clone()))?;

        if batch.schema().as_ref() != def.schema.as_ref() {
            return Err(MutableTableError::Schema(format!(
                "batch schema mismatch for {}: expected {} columns, got {}",
                table,
                def.schema.fields().len(),
                batch.schema().fields().len()
            )));
        }

        let session_tenant = tx.tenant();
        tx.assert_tenant_matches(session_tenant, table.as_str())?;

        let col_names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let cols: Vec<&str> = col_names.iter().map(String::as_str).collect();
        let dml = self.backend.insert_dml(&def, &cols, batch.num_rows());
        let params = batch_to_params(batch, session_tenant)
            .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.into())))?;
        let rows = tx.execute(&dml, &params).await?;
        #[cfg(feature = "test-hooks")]
        crate::store::mutable::test_hook::maybe_signal(rows).await;
        Ok(rows)
    }

    /// Stream rows from a mutable table whose declared `order_column` value
    /// is strictly greater than `after`, in ascending `order_column` order.
    /// Errors with [`MutableTableError::NoOrderColumn`] if the table was
    /// registered without an `order_column`.
    ///
    /// Resolves tenant from the registry's binding (session sticky value or
    /// `with_tenant_scoped` task-local override). For paths that must bind
    /// tenant explicitly — e.g. a subscriber stream whose lifetime extends
    /// past the binding's task-local — use [`Self::scan_after_for_tenant`].
    ///
    /// Implementation note: the closure-passing
    /// [`crate::catalog::backend::CatalogBackend::transaction`] API closes
    /// the transaction when the closure returns, so the stream cannot
    /// lazily fetch rows across `poll_next` calls without leaking the
    /// transaction. Instead, this materialises a single `RecordBatch` from
    /// the entire qualifying row set inside one read-only transaction, then
    /// yields it via [`futures::stream::iter`]. Memory is bounded by the
    /// total matching row count; Phase 4's broker is responsible for
    /// configuring topic retention so the backing tables stay bounded.
    pub async fn scan_after(
        &self,
        table: &MutableTableId,
        after: i64,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<RecordBatch, MutableTableError>> + Send>>,
        MutableTableError,
    > {
        let tenant = self.catalog.current_tenant();
        self.scan_after_for_tenant(table, after, tenant).await
    }

    /// Variant of [`Self::scan_after`] that takes an explicit `tenant`
    /// rather than reading the binding.
    ///
    /// Used by the subscriber's `subscribe_scoped` path: the gRPC handler
    /// resolves tenant once at request entry, then threads it down so the
    /// backing-table replay query bakes the right filter into its SQL
    /// regardless of whether any task-local override is still in effect
    /// when this method is awaited. The returned stream is fully
    /// materialised inside this call, so subsequent polls do not consult
    /// any tenant state.
    pub async fn scan_after_for_tenant(
        &self,
        table: &MutableTableId,
        after: i64,
        tenant: Option<TenantId>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<RecordBatch, MutableTableError>> + Send>>,
        MutableTableError,
    > {
        // Resolve the definition under the same explicit `tenant` the replay
        // SQL filters by, not the session binding: the subscribe/replay path
        // threads the topic's tenant down here, and the backing table is owned
        // by that tenant. Reading the binding instead would miss the table when
        // the request scope differs from the topic's tenant.
        let def = self
            .catalog
            .get_mutable_table_for_tenant(table, tenant)
            .await?
            .ok_or_else(|| MutableTableError::NotFound(table.clone()))?;
        let order_col = def
            .order_column
            .clone()
            .ok_or(MutableTableError::NoOrderColumn)?;

        let batch =
            fetch_scan_after_batch(Arc::clone(&self.backend), def, &order_col, after, tenant)
                .await?;
        let batches = if batch.num_rows() == 0 {
            Vec::new()
        } else {
            vec![batch]
        };
        Ok(Box::pin(futures::stream::iter(batches.into_iter().map(Ok))))
    }
}

/// Provision a mutable table inside an already-open transaction: write the
/// catalog registration row + index rows, then the storage `CREATE TABLE` and
/// its secondary-index DDL. Does NOT commit — the caller's transaction owns
/// commit/rollback.
///
/// Shared primitive behind [`MutableTableRegistry::register`] (which wraps it
/// in its own transaction) and
/// [`TopicRepo::register_topic`](crate::catalog::topic_repo::TopicRepo::register_topic)
/// (which shares the topic-row transaction). Because the mutable storage tables
/// live in the catalog's own database, one transaction spans the catalog row
/// and the backing table — they commit or roll back together.
pub(crate) async fn register_in_tx(
    backend: &dyn MutableBackend,
    tx: &mut Transaction<'_>,
    def: &MutableTableDefinition,
) -> Result<(), MutableTableError> {
    let payload = MutableRowPayload::encode(def, backend.catalog_backend().backend_kind())?;
    // Catalog registration row + index rows first.
    payload.write(tx).await?;
    // Then the backing storage table + its secondary indexes.
    tx.execute(&backend.create_table_ddl(def), &[]).await?;
    for idx in &def.indexes {
        tx.execute(&backend.create_index_ddl(def, idx), &[]).await?;
    }
    Ok(())
}

/// Drop a mutable table inside an already-open transaction: storage
/// `DROP TABLE` then catalog-row + index-row delete (strict tenant scope).
/// Does NOT commit — the caller's transaction owns commit/rollback.
///
/// Shared primitive behind [`MutableTableRegistry::drop_table`] and
/// [`TopicRepo::drop_topic`](crate::catalog::topic_repo::TopicRepo::drop_topic).
pub(crate) async fn drop_in_tx(
    backend: &dyn MutableBackend,
    tx: &mut Transaction<'_>,
    def: &MutableTableDefinition,
    tenant: Option<TenantId>,
) -> Result<(), MutableTableError> {
    tx.execute(&backend.drop_table_ddl(def), &[]).await?;
    delete_mutable_table_rows(tx, def.id.as_str(), tenant).await?;
    Ok(())
}

/// Flatten a [`MutableTableError`] into a [`BackendError`] so an `*_in_tx`
/// helper's error can propagate out of a `CatalogBackend::transaction` closure
/// (whose error type is `BackendError`) and trigger rollback. A `Backend`
/// variant passes through unwrapped; any other variant (schema/JSON encode
/// failure) is surfaced as [`BackendError::Execution`].
fn map_mutable_err(e: MutableTableError) -> BackendError {
    match e {
        MutableTableError::Backend(b) => b,
        other => BackendError::Execution(other.to_string()),
    }
}

/// Issue `scan_dml` with `order_column > $after AND (tenant_id = $t OR tenant_id IS NULL)`
/// in a single read-only transaction; materialise rows into one `RecordBatch`.
async fn fetch_scan_after_batch(
    backend: Arc<dyn MutableBackend>,
    def: MutableTableDefinition,
    order_col: &str,
    after: i64,
    tenant: Option<TenantId>,
) -> Result<RecordBatch, MutableTableError> {
    use arrow::array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, StringArray,
    };
    use arrow_schema::DataType;
    use std::sync::Arc as StdArc;

    // Build predicate: order_col > $after AND (tenant filter).
    let tenant_pred = match tenant {
        Some(t) => format!("(\"tenant_id\" = '{t}' OR \"tenant_id\" IS NULL)"),
        None => "\"tenant_id\" IS NULL".to_string(),
    };
    let predicate = format!(
        "\"{}\" > {} AND {}",
        order_col.replace('"', "\"\""),
        after,
        tenant_pred
    );

    let col_names: Vec<&str> = def
        .schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let base_sql = backend.scan_dml(&def, &col_names, Some(predicate.as_str()), None);
    // `scan_dml` does not emit ORDER BY; without it Postgres is free to return
    // rows in any sequence. `scan_after`'s ascending-order contract requires
    // the sort, so wrap the rendered statement here.
    let sql = format!(
        "{base_sql} ORDER BY \"{}\" ASC",
        order_col.replace('"', "\"\"")
    );

    let columns: Vec<(String, DataType)> = def
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let columns_for_closure = columns.clone();
    let owned_sql = sql;

    #[derive(Clone)]
    enum Decoded {
        Null,
        Bool(bool),
        Int(i64),
        Float(f64),
        Text(String),
    }

    fn decode_row(
        row: &crate::catalog::backend::Row<'_>,
        columns: &[(String, DataType)],
    ) -> Result<Vec<Decoded>, BackendError> {
        columns
            .iter()
            .map(|(name, ty)| match ty {
                DataType::Boolean => row
                    .try_get::<bool>(name)?
                    .map(Decoded::Bool)
                    .map(Ok)
                    .unwrap_or(Ok(Decoded::Null)),
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => row
                    .try_get::<i64>(name)?
                    .map(Decoded::Int)
                    .map(Ok)
                    .unwrap_or(Ok(Decoded::Null)),
                DataType::Float16 | DataType::Float32 | DataType::Float64 => row
                    .try_get::<f64>(name)?
                    .map(Decoded::Float)
                    .map(Ok)
                    .unwrap_or(Ok(Decoded::Null)),
                _ => row
                    .try_get::<String>(name)?
                    .map(Decoded::Text)
                    .map(Ok)
                    .unwrap_or(Ok(Decoded::Null)),
            })
            .collect()
    }

    let rows_per_col: Vec<Vec<Decoded>> = backend
        .catalog_backend()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            move |tx| {
                Box::pin(async move {
                    let raw = tx
                        .query(&owned_sql, &[], |row| decode_row(row, &columns_for_closure))
                        .await?;
                    let mut transposed: Vec<Vec<Decoded>> = (0..columns_for_closure.len())
                        .map(|_| Vec::with_capacity(raw.len()))
                        .collect();
                    for r in raw {
                        for (i, v) in r.into_iter().enumerate() {
                            transposed[i].push(v);
                        }
                    }
                    Ok(transposed)
                })
            },
        )
        .await?;

    let arrays: Vec<ArrayRef> = columns
        .iter()
        .zip(rows_per_col.into_iter())
        .map(|((_, ty), values)| -> ArrayRef {
            match ty {
                DataType::Boolean => {
                    let arr: BooleanArray = values
                        .into_iter()
                        .map(|v| match v {
                            Decoded::Bool(b) => Some(b),
                            _ => None,
                        })
                        .collect();
                    StdArc::new(arr) as ArrayRef
                }
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => {
                    let arr: Int64Array = values
                        .into_iter()
                        .map(|v| match v {
                            Decoded::Int(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    StdArc::new(arr) as ArrayRef
                }
                DataType::Float32 => {
                    let arr: Float32Array = values
                        .into_iter()
                        .map(|v| match v {
                            Decoded::Float(f) => Some(f as f32),
                            _ => None,
                        })
                        .collect();
                    StdArc::new(arr) as ArrayRef
                }
                DataType::Float64 => {
                    let arr: Float64Array = values
                        .into_iter()
                        .map(|v| match v {
                            Decoded::Float(f) => Some(f),
                            _ => None,
                        })
                        .collect();
                    StdArc::new(arr) as ArrayRef
                }
                _ => {
                    let arr: StringArray = values
                        .into_iter()
                        .map(|v| match v {
                            Decoded::Text(s) => Some(s),
                            _ => None,
                        })
                        .collect();
                    StdArc::new(arr) as ArrayRef
                }
            }
        })
        .collect();

    RecordBatch::try_new(Arc::clone(&def.schema), arrays)
        .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))
}
