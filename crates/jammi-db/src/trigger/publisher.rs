//! Transactional-outbox publisher for trigger-stream topics.
//!
//! Per SPEC-04 §7.2, every successful publish writes the augmented batch to
//! the topic's Phase-2 backing table inside one `CatalogBackend::transaction`
//! closure (the authoritative log) and then fans out to the broker (a best-
//! effort delivery accelerator). A broker fan-out failure after commit is
//! recorded and the RPC still returns `Ok` — subscribers replay from the
//! backing table on next reconnect.
//!
//! ## Transactional offset assignment
//!
//! The offset is assigned by ONE locking statement — an `UPDATE … RETURNING`
//! against `topics.next_offset` — inside the SAME transaction that inserts
//! the augmented batch (see [`Publisher::publish_scoped`]). This closes the
//! multi-replica defect a per-process `AtomicU64` counter could not: two
//! engine replicas publishing against the same Postgres database each read
//! `MAX(_offset)` independently and could compute the SAME next offset,
//! colliding on the backing table's `(_offset, _row_idx)` composite key (or,
//! worse, partially colliding on a multi-row batch). `next_offset` makes the
//! counter a durable, row-locked catalog value: the `UPDATE`'s row lock is
//! held until commit, so **per topic, offset order == commit order** — the
//! invariant every watermark-bounded replay (`_offset > watermark`) depends
//! on for completeness (see `crate::trigger::subscriber`).
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use chrono::Utc;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::catalog::backend::{BackendImpl, SqlValue, TxOptions};
use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::MutableTableId;
use crate::tenant::TenantId;
use crate::trigger::broker::TriggerBroker;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::TopicId;
use crate::trigger::offset::Offset;
use crate::trigger::topic::{augment_schema_for_backing, TopicDefinition};

/// Publishes batches to topics using the transactional-outbox pattern.
///
/// Offset assignment lives in the catalog (`topics.next_offset`), not in
/// process memory — see the module docs. An [`AsyncMutex`] per topic is
/// RETAINED as the in-process fast-path arm: it serialises concurrent
/// publishers on the SAME topic within one process so they don't all race
/// into the same row-locked `UPDATE` simultaneously (harmless on Postgres,
/// which simply serialises them at the row lock, but avoids piling up
/// waiters against SQLite's single-writer `BEGIN IMMEDIATE` unnecessarily).
pub struct Publisher {
    broker: Arc<dyn TriggerBroker>,
    backend: Arc<BackendImpl>,
    mutable: Arc<MutableTableRegistry>,
    write_locks: Mutex<HashMap<TopicId, Arc<AsyncMutex<()>>>>,
}

impl Publisher {
    pub fn new(
        broker: Arc<dyn TriggerBroker>,
        backend: Arc<BackendImpl>,
        mutable: Arc<MutableTableRegistry>,
    ) -> Self {
        Self {
            broker,
            backend,
            mutable,
            write_locks: Mutex::new(HashMap::new()),
        }
    }

    /// Publish one batch to `topic` under the given `tenant` scope.
    ///
    /// `tenant` is the tenant whose rows are being published. It is bound on
    /// the backing-table transaction via [`crate::catalog::backend::Transaction::set_tenant`]
    /// so every row's `tenant_id` column is stamped with the same value the
    /// mutable-table write-side guard
    /// ([`crate::catalog::backend::Transaction::assert_tenant_matches`])
    /// asserts. The resulting rows are visible to a tenant-scoped subscriber
    /// (the `tenant_id = $current OR tenant_id IS NULL` predicate) only when
    /// the subscriber's tenant equals `tenant`.
    ///
    /// `None` publishes a globally-scoped row (`tenant_id IS NULL`). This is
    /// the right value for engine-default topics that are read by every
    /// tenant; pass a `Some(_)` value for any topic whose readers are
    /// tenant-scoped.
    ///
    /// # Tenant contract
    ///
    /// * If [`TopicDefinition::tenant`] is `Some(t)`, `tenant` must equal
    ///   `Some(t)` — a mismatch returns [`TriggerError::PublishTenantMismatch`]
    ///   before any transaction is opened. The topic itself is the source of
    ///   truth: tenant-pinned topics never accept cross-tenant publishes.
    /// * If [`TopicDefinition::tenant`] is `None`, `tenant` may be either
    ///   `None` (global row) or `Some(_)` (tenant-tagged row on a globally-
    ///   declared topic). Both shapes are well-defined; readers see them
    ///   according to the standard `tenant_id = $current OR tenant_id IS NULL`
    ///   predicate.
    ///
    /// Validates the batch schema against the topic's schema, mints an
    /// offset, commits to the backing table inside a single transaction
    /// with `tenant` bound, and best-effort fans out to the broker. Returns
    /// the assigned offset.
    pub async fn publish_scoped(
        &self,
        topic: &TopicDefinition,
        tenant: Option<TenantId>,
        user_batch: RecordBatch,
    ) -> Result<Offset, TriggerError> {
        // Offsets must be gap-free by construction: an empty batch would
        // still mint and burn an offset for zero rows, which is pointless
        // and (worse) makes "every offset has at least one row" stop being
        // an invariant callers can rely on.
        if user_batch.num_rows() == 0 {
            return Err(TriggerError::BatchSchemaMismatch(
                "publish_scoped rejects an empty batch".to_string(),
            ));
        }
        if user_batch.schema().as_ref() != topic.schema.as_ref() {
            return Err(TriggerError::BatchSchemaMismatch(format!(
                "topic '{}' expected {} columns, got {}",
                topic.name,
                topic.schema.fields().len(),
                user_batch.schema().fields().len()
            )));
        }

        // Tenant-pinned topics reject cross-tenant publishes up front rather
        // than relying on the backing-table write-side guard to surface a
        // generic `TenantMismatch` from inside the transaction. The pre-check
        // keeps the failure attributable to the caller's `tenant` argument.
        if let Some(topic_tenant) = topic.tenant {
            if tenant != Some(topic_tenant) {
                return Err(TriggerError::PublishTenantMismatch {
                    topic: topic.name.clone(),
                    topic_tenant: Some(topic_tenant),
                    publish_tenant: tenant,
                });
            }
        }

        let write_lock = self.write_lock_for(topic.id);
        let _guard = write_lock.lock().await;

        let backing_table_id = MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
        let topic_id_str = topic.id.to_string();
        let backing_name = backing_table_id.as_str().to_string();
        let produced_at = Utc::now();
        let produced_at_micros = produced_at.timestamp_micros();
        let topic_schema = Arc::clone(&topic.schema);

        let registry = Arc::clone(&self.mutable);
        let id_for_closure = backing_table_id.clone();
        let user_batch_for_tx = user_batch.clone();
        // Seed-and-bump `topics.next_offset` with ONE locking statement, then
        // insert the augmented batch, all inside the SAME transaction — the
        // offset does not exist before the transaction opens (it is minted by
        // the `UPDATE … RETURNING` itself), so `augment_batch_for_backing`
        // moves inside the closure too. On Postgres the `UPDATE`'s row lock
        // on this topic's `topics` row is what makes two concurrent
        // publishers (in-process or cross-replica) serialize on THIS
        // statement rather than racing an unlocked read-then-write window;
        // on SQLite `BEGIN IMMEDIATE` already holds the single writer lock.
        // Same statement text on both backends (sqlx runtime queries).
        let assigned: Option<u64> = self
            .backend
            .transaction(TxOptions::default(), move |tx| {
                let registry = Arc::clone(&registry);
                let id = id_for_closure.clone();
                let topic_schema = Arc::clone(&topic_schema);
                let user_batch = user_batch_for_tx.clone();
                let topic_id_str = topic_id_str.clone();
                let backing_name = backing_name.clone();
                Box::pin(async move {
                    tx.set_tenant(tenant);

                    let sql = format!(
                        "UPDATE topics SET next_offset = COALESCE(next_offset, \
                         (SELECT COALESCE(MAX(\"_offset\"), -1) + 1 FROM \"{backing}\")) + 1 \
                         WHERE topic_id = $1 RETURNING next_offset - 1 AS assigned",
                        backing = backing_name.replace('"', "\"\"")
                    );
                    let assigned_rows: Option<i64> = tx
                        .query_opt(&sql, &[SqlValue::TextOwned(topic_id_str)], |row| {
                            row.get::<i64>("assigned")
                        })
                        .await?;
                    // Zero rows means the topic row does not exist (never a
                    // defaulted offset) — return `None` without inserting;
                    // the empty transaction commits as a no-op.
                    let Some(offset_value) = assigned_rows else {
                        return Ok::<Option<u64>, crate::catalog::backend::BackendError>(None);
                    };
                    let offset_value = offset_value as u64;

                    let augmented = augment_batch_for_backing(
                        &topic_schema,
                        &user_batch,
                        offset_value,
                        produced_at_micros,
                    )
                    .map_err(|e| crate::catalog::backend::BackendError::Execution(e.to_string()))?;
                    // Bind the publish-scoped tenant on the transaction so
                    // `MutableTableRegistry::insert_batch` stamps every row's
                    // `tenant_id` slot with `tenant` and its write-side
                    // `assert_tenant_matches` guard agrees.
                    registry
                        .insert_batch(tx, &id, &augmented)
                        .await
                        .map_err(|e| {
                            crate::catalog::backend::BackendError::Execution(e.to_string())
                        })?;
                    Ok(Some(offset_value))
                })
            })
            .await?;

        let offset_value =
            assigned.ok_or_else(|| TriggerError::TopicNotFound(topic.id.to_string()))?;

        // Best-effort fan-out — a broker failure leaves the backing table as
        // the authoritative log and subscribers replay on reconnect.
        match self
            .broker
            .publish(topic.id, user_batch, produced_at, offset_value, tenant)
            .await
        {
            Ok(off) => Ok(off),
            Err(err) => {
                tracing::warn!(
                    topic = %topic.name,
                    offset = offset_value,
                    error = %err,
                    "broker fan-out failed; backing table is authoritative"
                );
                Ok(Offset::new(offset_value, produced_at))
            }
        }
    }

    /// The in-process fast-path arm — see the struct docs.
    fn write_lock_for(&self, topic_id: TopicId) -> Arc<AsyncMutex<()>> {
        let mut guard = self.write_locks.lock();
        if let Some(existing) = guard.get(&topic_id) {
            return Arc::clone(existing);
        }
        let new = Arc::new(AsyncMutex::new(()));
        guard.insert(topic_id, Arc::clone(&new));
        new
    }
}

/// Prepend the three engine-controlled columns to a user batch:
/// `_offset` repeats for every row (so the subscribe path can group rows
/// back into the originally-published batch), `_row_idx` is the per-row
/// position within the batch (used in the composite PK and for intra-batch
/// ordering), and `_produced_at` is the publish-time microsecond instant.
fn augment_batch_for_backing(
    user_schema: &SchemaRef,
    user_batch: &RecordBatch,
    offset_value: u64,
    produced_at_micros: i64,
) -> Result<RecordBatch, TriggerError> {
    let n = user_batch.num_rows();
    let augmented_schema = Arc::new(augment_schema_for_backing(user_schema));
    let row_indices: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(user_batch.num_columns() + 3);
    columns.push(Arc::new(Int64Array::from(vec![offset_value as i64; n])));
    columns.push(Arc::new(Int64Array::from(row_indices)));
    columns.push(Arc::new(Int64Array::from(vec![produced_at_micros; n])));
    for c in user_batch.columns() {
        columns.push(c.clone());
    }
    RecordBatch::try_new(augmented_schema, columns)
        .map_err(|e| TriggerError::BatchSchemaMismatch(e.to_string()))
}
