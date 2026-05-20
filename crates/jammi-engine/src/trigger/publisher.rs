//! Transactional-outbox publisher for trigger-stream topics.
//!
//! Per SPEC-04 §7.2, every successful publish writes the augmented batch to
//! the topic's Phase-2 backing table inside one `CatalogBackend::transaction`
//! closure (the authoritative log) and then fans out to the broker (a best-
//! effort delivery accelerator). A broker fan-out failure after commit is
//! recorded and the RPC still returns `Ok` — subscribers replay from the
//! backing table on next reconnect.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use chrono::Utc;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::catalog::backend::{BackendImpl, TxOptions};
use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::MutableTableId;
use crate::trigger::broker::TriggerBroker;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::TopicId;
use crate::trigger::offset::Offset;
use crate::trigger::topic::{augment_schema_for_backing, TopicDefinition};

/// Publishes batches to topics using the transactional-outbox pattern.
///
/// Each topic owns one [`AtomicU64`] offset counter, lazily seeded from
/// `MAX(_offset)` on the backing table the first time the topic is
/// published. An [`AsyncMutex`] per topic serialises the counter-seed +
/// insert critical section so concurrent publishes assign contiguous
/// offsets without leaving gaps on rollback.
pub struct Publisher {
    broker: Arc<dyn TriggerBroker>,
    backend: Arc<BackendImpl>,
    mutable: Arc<MutableTableRegistry>,
    counters: Mutex<HashMap<TopicId, Arc<TopicCounter>>>,
}

struct TopicCounter {
    /// Serialises the read-MAX + insert critical section.
    write_lock: AsyncMutex<()>,
    /// Next offset to assign. `u64::MAX` means "not yet seeded".
    next: AtomicU64,
}

impl TopicCounter {
    fn new() -> Self {
        Self {
            write_lock: AsyncMutex::new(()),
            next: AtomicU64::new(u64::MAX),
        }
    }
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
            counters: Mutex::new(HashMap::new()),
        }
    }

    /// Publish one batch to `topic`. Validates the batch schema against
    /// the topic's schema, mints an offset, commits to the backing table,
    /// and fans out to the broker. Returns the assigned offset.
    pub async fn publish(
        &self,
        topic: &TopicDefinition,
        user_batch: RecordBatch,
    ) -> Result<Offset, TriggerError> {
        if user_batch.schema().as_ref() != topic.schema.as_ref() {
            return Err(TriggerError::BatchSchemaMismatch(format!(
                "topic '{}' expected {} columns, got {}",
                topic.name,
                topic.schema.fields().len(),
                user_batch.schema().fields().len()
            )));
        }

        let counter = self.counter_for(topic.id);
        let _guard = counter.write_lock.lock().await;

        let backing_table_id = MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;

        // Seed the offset counter on first use by reading MAX(_offset) from
        // the backing table. Acquiring `write_lock` already serialises
        // concurrent publishers on this topic, so a single fetch is enough.
        if counter.next.load(Ordering::Acquire) == u64::MAX {
            let next = self.read_next_offset(backing_table_id.as_str()).await?;
            counter.next.store(next, Ordering::Release);
        }

        let offset_value = counter.next.load(Ordering::Acquire);
        let produced_at = Utc::now();
        let produced_at_micros = produced_at.timestamp_micros();
        let augmented = augment_batch_for_backing(
            &topic.schema,
            &user_batch,
            offset_value,
            produced_at_micros,
        )?;

        let registry = Arc::clone(&self.mutable);
        let id_for_closure = backing_table_id.clone();
        let augmented_for_closure = augmented;
        self.backend
            .transaction(TxOptions::default(), move |tx| {
                let registry = Arc::clone(&registry);
                let id = id_for_closure.clone();
                let augmented = augmented_for_closure.clone();
                Box::pin(async move {
                    registry
                        .insert_batch(tx, &id, &augmented)
                        .await
                        .map_err(|e| {
                            crate::catalog::backend::BackendError::Execution(e.to_string())
                        })?;
                    Ok::<(), crate::catalog::backend::BackendError>(())
                })
            })
            .await?;

        // Only advance the counter after the transaction commits — a rollback
        // preserves the offset for the next attempt and avoids gaps in the
        // backing table.
        counter.next.store(offset_value + 1, Ordering::Release);

        // Best-effort fan-out — a broker failure leaves the backing table as
        // the authoritative log and subscribers replay on reconnect.
        match self
            .broker
            .publish(topic.id, user_batch, produced_at, offset_value)
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

    fn counter_for(&self, topic_id: TopicId) -> Arc<TopicCounter> {
        let mut guard = self.counters.lock();
        if let Some(existing) = guard.get(&topic_id) {
            return Arc::clone(existing);
        }
        let new = Arc::new(TopicCounter::new());
        guard.insert(topic_id, Arc::clone(&new));
        new
    }

    /// Read `COALESCE(MAX(_offset), -1) + 1` from the backing table; used to
    /// seed the in-memory offset counter on first publish (or after a
    /// process restart).
    async fn read_next_offset(&self, backing_table: &str) -> Result<u64, TriggerError> {
        let sql = format!(
            "SELECT COALESCE(MAX(\"_offset\"), -1) + 1 AS next FROM \"{}\"",
            backing_table.replace('"', "\"\"")
        );
        let result = self
            .backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    let sql = sql.clone();
                    Box::pin(async move {
                        let rows: Vec<i64> =
                            tx.query(&sql, &[], |row| row.get::<i64>("next")).await?;
                        Ok::<i64, crate::catalog::backend::BackendError>(
                            rows.into_iter().next().unwrap_or(0),
                        )
                    })
                },
            )
            .await?;
        Ok(result.max(0) as u64)
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
