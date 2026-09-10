//! Mutable companion table registry — lifecycle of catalog row + storage table.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::catalog::TableProvider;
use futures::Stream;
use tokio::sync::Semaphore;

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
    /// Bounds concurrent trigger-stream replays against this backend's
    /// connection pool — sized `pool_size - 2` (min 1), leaving headroom for
    /// the write path's own connection use. Acquired inside [`Self::tail_replay`]
    /// (never at each call site), so every replay caller against this pool —
    /// the trigger tail's own driver-triggered replay
    /// (`crate::trigger::tail::replay_and_fan_out`), a lagging subscriber's
    /// own replay (`crate::trigger::tail::lag_replay`), and `Subscriber`'s
    /// subscribe-time backing-table drain — is bounded through the SAME
    /// permit pool and none of them can start an unbounded number of
    /// concurrent replays that would starve publishers of pool connections.
    replay_permits: Arc<Semaphore>,
}

impl MutableTableRegistry {
    pub fn new(
        catalog: Arc<Catalog>,
        backend: Arc<dyn MutableBackend>,
        tenant: TenantBinding,
    ) -> Self {
        let pool_size = backend.catalog_backend().pool_size();
        let permits = pool_size.saturating_sub(2).max(1) as usize;
        Self {
            catalog,
            backend,
            tenant,
            replay_permits: Arc::new(Semaphore::new(permits)),
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
    /// `with_tenant_scoped` task-local override). For a caller that must bind
    /// tenant explicitly instead of consulting that binding, use
    /// [`Self::scan_after_for_tenant`] — NOT the trigger-stream replay path
    /// any more, which is `Self::tail_replay` (one permit-bounded step per
    /// call, used by [`crate::trigger::Subscriber`] and the trigger tail).
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

    /// Resolve `table`'s definition scoped to `tenant`, without opening any
    /// transaction. Used by the trigger-stream `TopicTail` actor to cache a
    /// topic's backing-table definition ONCE, before its replay transaction
    /// opens: a tail must never hold two connections at once.
    pub async fn definition_for_tenant(
        &self,
        table: &MutableTableId,
        tenant: Option<TenantId>,
    ) -> Result<MutableTableDefinition, MutableTableError> {
        self.catalog
            .get_mutable_table_for_tenant(table, tenant)
            .await?
            .ok_or_else(|| MutableTableError::NotFound(table.clone()))
    }

    /// One tail-replay STEP: read the tenant-blind head `MAX(order_col)`
    /// FIRST, then fetch ONE `chunk_size`-row-bounded group of tenant-scoped
    /// rows in `order_col > cursor_before AND order_col <= head` order — a
    /// group that straddles the `chunk_size` boundary is fetched whole rather
    /// than split — all inside ONE read-only transaction. Returns that one
    /// step's rows (already reassembled into whole-group batches), the new
    /// cursor after this step, and whether the step reached `head` (`true`) or
    /// more remains (`false`, in which case the caller loops, passing the
    /// returned cursor back in as the next call's `cursor_before`).
    ///
    /// This is a SINGLE STEP, never a whole-backlog drain. Two of its three
    /// callers — `replay_and_fan_out`'s own retry loop and a lagging
    /// subscriber's own catch-up loop — hand each step's batches off
    /// (fan-out send, or `yield`) before calling again, so their resident
    /// memory across a full catch-up from a low `cursor_before` to a far
    /// `head` is one step's rows (`chunk_size`, or one group's width if
    /// wider) at a time, never the total backlog. The third,
    /// `Subscriber::drain_replay`, deliberately accumulates every step into
    /// one `Vec` — its contract is a finite, fully materialised window — so
    /// its residency is that window's total rows, bounded by the head as it
    /// MOVES (each step re-reads `MAX(order_col)`, so the window is not
    /// pinned at call time), plus at most one step ahead of what it has
    /// already accumulated.
    ///
    /// `def` and `order_col` are resolved by the caller ONCE via
    /// [`Self::definition_for_tenant`], before this call, so a tail never
    /// holds two connections.
    ///
    /// Acquires a permit from [`Self::replay_permits`] BEFORE opening the
    /// transaction below and holds it for this one step's duration — this is
    /// the ONE place every trigger-stream replay path (the tail's own
    /// driver-triggered replay, a lagging subscriber's own replay, and a
    /// fresh subscriber's subscribe-time backing-table drain) is bounded, so
    /// no caller of this method can bypass the bound by calling some other,
    /// unbounded entry point. Releasing the permit between steps (rather than
    /// holding it for an entire multi-step catch-up) also means a long catch-up
    /// no longer monopolises one pool connection for its full duration.
    ///
    /// If [`Self::replay_permits`] is ever closed (never done in this
    /// registry's own lifetime today — see the field doc), this degrades
    /// rather than panics: it logs a warning and returns `(vec![],
    /// cursor_before, true)`, i.e. "no progress, nothing more to try" — never
    /// a panic inside a long-lived tail task.
    pub(crate) async fn tail_replay(
        &self,
        def: &MutableTableDefinition,
        order_col: &str,
        tenant: Option<TenantId>,
        cursor_before: i64,
        chunk_size: usize,
    ) -> Result<(Vec<RecordBatch>, i64, bool), MutableTableError> {
        let _permit = match self.replay_permits.acquire().await {
            Ok(permit) => permit,
            Err(_) => {
                tracing::warn!(
                    "trigger tail: replay semaphore closed unexpectedly; cursor left unchanged"
                );
                return Ok((Vec::new(), cursor_before, true));
            }
        };
        let backend = Arc::clone(&self.backend);
        let def = def.clone();
        let order_col = order_col.to_string();
        self.backend
            .catalog_backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    let backend = Arc::clone(&backend);
                    let def = def.clone();
                    let order_col = order_col.clone();
                    Box::pin(async move {
                        tail_replay_in_tx(
                            tx,
                            backend.as_ref(),
                            &def,
                            &order_col,
                            tenant,
                            cursor_before,
                            chunk_size,
                        )
                        .await
                        .map_err(map_mutable_err)
                    })
                },
            )
            .await
            .map_err(MutableTableError::Backend)
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
///
/// `ORDER BY` carries `order_col` plus every `def.primary_key` column that is
/// not itself `order_col`, as a tiebreak — the topic backing table's PK is
/// `(_offset, _row_idx)` with `order_col = _offset`, so this emits
/// `ORDER BY _offset, _row_idx` and makes intra-batch row order exact rather
/// than whatever order Postgres happens to return same-`_offset` rows in.
/// The builder rejects an empty primary key
/// ([`crate::store::mutable::definition::MutableTableDefinitionBuilder::build`]),
/// so this clause is always well-formed when non-empty; a table whose sole PK
/// column equals `order_col` simply emits no extra tiebreak column.
async fn fetch_scan_after_batch(
    backend: Arc<dyn MutableBackend>,
    def: MutableTableDefinition,
    order_col: &str,
    after: i64,
    tenant: Option<TenantId>,
) -> Result<RecordBatch, MutableTableError> {
    use crate::store::mutable::provider::{build_arrays, decode_row, DecodedValue};
    use arrow_schema::DataType;

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
    // the sort, so wrap the rendered statement here — order_col first, then
    // every other PK column as an intra-group tiebreak.
    let tiebreak_cols: Vec<&String> = def
        .primary_key
        .iter()
        .filter(|c| c.as_str() != order_col)
        .collect();
    let mut order_by = format!("\"{}\" ASC", order_col.replace('"', "\"\""));
    for c in &tiebreak_cols {
        order_by.push_str(&format!(", \"{}\" ASC", c.replace('"', "\"\"")));
    }
    let sql = format!("{base_sql} ORDER BY {order_by}");

    let columns: Vec<(String, DataType)> = def
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let columns_for_closure = columns.clone();
    let owned_sql = sql;

    let rows_per_col: Vec<Vec<DecodedValue>> = backend
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
                    let mut transposed: Vec<Vec<DecodedValue>> = (0..columns_for_closure.len())
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

    let arrays = build_arrays(&columns, rows_per_col)
        .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;

    RecordBatch::try_new(Arc::clone(&def.schema), arrays)
        .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))
}

/// Render `(tenant_id = '$t' OR tenant_id IS NULL)` for a tenant-scoped
/// caller, or `tenant_id IS NULL` for a globally-scoped one — the same shape
/// [`fetch_scan_after_batch`] inlines, factored out so
/// [`tail_replay_in_tx`] shares it.
fn tenant_predicate_clause(tenant: Option<TenantId>) -> String {
    match tenant {
        Some(t) => format!("(\"tenant_id\" = '{t}' OR \"tenant_id\" IS NULL)"),
        None => "\"tenant_id\" IS NULL".to_string(),
    }
}

/// Transpose a `Vec` of row-major decoded values into column-major order.
fn transpose_rows(
    rows: Vec<Vec<crate::store::mutable::provider::DecodedValue>>,
    n_cols: usize,
) -> Vec<Vec<crate::store::mutable::provider::DecodedValue>> {
    let mut transposed: Vec<Vec<crate::store::mutable::provider::DecodedValue>> = (0..n_cols)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for r in rows {
        for (i, v) in r.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }
    transposed
}

/// Read the `order_col` value out of one decoded row (row-major, as returned
/// by `decode_row`) as an `i64`. Topic backing tables always declare
/// `_offset`/`_row_idx` as `Int64` (`augment_schema_for_backing`), so the
/// decode arm is `Int64` in practice; the narrower integer arms are accepted
/// too so this helper stays correct if it is ever pointed at a differently
/// typed order column.
fn order_value_of(
    row: &[crate::store::mutable::provider::DecodedValue],
    columns: &[(String, arrow_schema::DataType)],
    order_col: &str,
) -> Result<i64, MutableTableError> {
    use crate::store::mutable::provider::DecodedValue;
    let idx = columns
        .iter()
        .position(|(name, _)| name == order_col)
        .ok_or_else(|| {
            MutableTableError::Backend(BackendError::Execution(format!(
                "tail replay: order column '{order_col}' missing from column list"
            )))
        })?;
    match &row[idx] {
        DecodedValue::Int64(v) => Ok(*v),
        DecodedValue::Int32(v) => Ok(*v as i64),
        DecodedValue::Int16(v) => Ok(*v as i64),
        other => Err(MutableTableError::Backend(BackendError::Execution(
            format!(
                "tail replay: order column '{order_col}' decoded as non-integer value {other:?}"
            ),
        ))),
    }
}

/// The body of [`MutableTableRegistry::tail_replay`] — see its docs. A free
/// function (rather than a method) so it can run entirely inside the
/// `catalog_backend().transaction` closure without borrowing `&self` across
/// the closure's `'tx` lifetime.
///
/// A SINGLE STEP (never an internal loop to `head`): fetches at most one
/// `chunk_size`-row-bounded group of rows (wider only if a single `order_col`
/// group itself is wider than `chunk_size` — a group is never split), returns
/// them plus the new cursor and whether `head` was reached. Bounding this to
/// one step is what keeps `MutableTableRegistry::tail_replay`'s resident
/// memory to one step's rows regardless of how far `cursor_before` trails
/// `head` — see that method's doc for the caller-side contract this depends
/// on.
async fn tail_replay_in_tx(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    order_col: &str,
    tenant: Option<TenantId>,
    cursor_before: i64,
    chunk_size: usize,
) -> Result<(Vec<RecordBatch>, i64, bool), MutableTableError> {
    use crate::store::mutable::provider::{build_arrays, decode_row};
    use arrow_schema::DataType;

    // Head FIRST, tenant-blind: everything committed at or below
    // this value is visible to every replica under READ COMMITTED (the
    // offset-assigning UPDATE holds its row lock until commit), so bounding
    // this step by it is non-lossy; anything committing after this read
    // is handled by the next `Wake`/`Batch`.
    let head_sql = format!(
        "SELECT MAX(\"{}\") AS m FROM \"{}\"",
        order_col.replace('"', "\"\""),
        def.id.as_str().replace('"', "\"\"")
    );
    let head: Option<i64> = tx
        .query(&head_sql, &[], |row| row.try_get::<i64>("m"))
        .await?
        .into_iter()
        .next()
        .flatten();
    let Some(head) = head else {
        // Table has no rows at all yet.
        return Ok((Vec::new(), cursor_before, true));
    };
    if head <= cursor_before {
        // Nothing committed above the cursor (a driver `Wake` that raced an
        // already-observed commit, or a regression — never negative work).
        return Ok((Vec::new(), cursor_before, true));
    }

    let col_names: Vec<&str> = def
        .schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let columns: Vec<(String, DataType)> = def
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let tiebreak_cols: Vec<&String> = def
        .primary_key
        .iter()
        .filter(|c| c.as_str() != order_col)
        .collect();
    let mut order_by = format!("\"{}\" ASC", order_col.replace('"', "\"\""));
    for c in &tiebreak_cols {
        order_by.push_str(&format!(", \"{}\" ASC", c.replace('"', "\"\"")));
    }
    let tenant_pred = tenant_predicate_clause(tenant);

    let predicate = format!(
        "\"{oc}\" > {from} AND \"{oc}\" <= {head} AND {tp}",
        oc = order_col.replace('"', "\"\""),
        from = cursor_before,
        tp = tenant_pred,
    );
    let base_sql = backend.scan_dml(def, &col_names, Some(predicate.as_str()), None);
    // `scan_dml`'s own `limit` param renders LIMIT immediately after the
    // WHERE clause, before any ORDER BY this call appends — so the probe
    // limit is appended by hand, after ORDER BY, rather than threaded
    // through `scan_dml`.
    let sql = format!("{base_sql} ORDER BY {order_by} LIMIT {}", chunk_size + 1);
    let raw = tx.query(&sql, &[], |row| decode_row(row, &columns)).await?;
    if raw.is_empty() {
        // No tenant-visible rows remain between `cursor_before` and `head`:
        // fully drained even though the head-to-head gap is nonzero.
        return Ok((Vec::new(), head, true));
    }
    let got = raw.len();
    if got <= chunk_size {
        // Bounded by `head` and within one step's LIMIT, so this is provably
        // the last step: every row up to `head` is present, whole groups
        // included.
        let arrays = build_arrays(&columns, transpose_rows(raw, columns.len()))
            .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
        let batch = RecordBatch::try_new(Arc::clone(&def.schema), arrays)
            .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
        return Ok((vec![batch], head, true));
    }
    // `got == chunk_size + 1`: the last row is a probe. If it shares
    // `order_col` with the row we intend to keep as the tail of this
    // step, that group spans the step boundary — fetch the WHOLE
    // group by exact match instead of taking a truncated slice, WITHOUT
    // discarding the earlier, already-complete groups this same fetch
    // returned (every row with a smaller `order_col` value sorts before
    // every row of the boundary group, so if this fetch reached the
    // boundary group at all, every row of every smaller group is
    // already present here).
    let last_kept = order_value_of(&raw[chunk_size - 1], &columns, order_col)?;
    let probe = order_value_of(&raw[chunk_size], &columns, order_col)?;
    if probe == last_kept {
        let mut raw = raw;
        let mut boundary_start = chunk_size - 1;
        while boundary_start > 0
            && order_value_of(&raw[boundary_start - 1], &columns, order_col)? == last_kept
        {
            boundary_start -= 1;
        }
        // Keep only the prior, definitely-complete groups; the boundary
        // group itself (and the discarded probe row) is re-fetched whole
        // below rather than trusted from this LIMIT-bounded slice.
        raw.truncate(boundary_start);
        let mut batches: Vec<RecordBatch> = Vec::new();
        if !raw.is_empty() {
            let arrays = build_arrays(&columns, transpose_rows(raw, columns.len()))
                .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
            let batch = RecordBatch::try_new(Arc::clone(&def.schema), arrays)
                .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
            batches.push(batch);
        }

        let exact_predicate = format!(
            "\"{oc}\" = {v} AND {tp}",
            oc = order_col.replace('"', "\"\""),
            v = last_kept,
            tp = tenant_pred,
        );
        let exact_sql = format!(
            "{} ORDER BY {}",
            backend.scan_dml(def, &col_names, Some(exact_predicate.as_str()), None),
            order_by
        );
        let exact_raw = tx
            .query(&exact_sql, &[], |row| decode_row(row, &columns))
            .await?;
        let arrays = build_arrays(&columns, transpose_rows(exact_raw, columns.len()))
            .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
        let batch = RecordBatch::try_new(Arc::clone(&def.schema), arrays)
            .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
        batches.push(batch);
        // Conservatively `false` (more work may remain): `last_kept` can, in
        // the rare case where the boundary group's own value is `head`
        // itself, already equal `head` — the next call's own `head <=
        // cursor_before` check (above) catches that and returns `(vec![],
        // head, true)` in one cheap extra step, never mis-skipping rows.
        return Ok((batches, last_kept, false));
    }
    let kept: Vec<_> = raw.into_iter().take(chunk_size).collect();
    let arrays = build_arrays(&columns, transpose_rows(kept, columns.len()))
        .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
    let batch = RecordBatch::try_new(Arc::clone(&def.schema), arrays)
        .map_err(|e| MutableTableError::Backend(BackendError::Execution(e.to_string())))?;
    Ok((vec![batch], last_kept, false))
}
