//! Replay+live join for trigger-stream subscriptions.
//!
//! The publisher commits every batch to the topic's backing table inside one
//! `CatalogBackend::transaction` and best-effort fans out to the broker. The
//! subscriber stitches a contiguous stream by routing the historical prefix
//! through `MutableTableRegistry::scan_after` and the live tail through the
//! broker's `subscribe`, filtering by offset so the two halves do not
//! overlap.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use async_stream::try_stream;
use chrono::DateTime;
use futures::StreamExt;

use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::MutableTableId;
use crate::trigger::broker::TriggerBroker;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::SubscriptionId;
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{DeliveredBatch, Subscription};
use crate::trigger::topic::{TopicDefinition, OFFSET_COLUMN, PRODUCED_AT_COLUMN, ROW_INDEX_COLUMN};

pub struct Subscriber {
    broker: Arc<dyn TriggerBroker>,
    mutable: Arc<MutableTableRegistry>,
}

impl Subscriber {
    pub fn new(broker: Arc<dyn TriggerBroker>, mutable: Arc<MutableTableRegistry>) -> Self {
        Self { broker, mutable }
    }

    /// Open a subscription that yields every batch matching `predicate` for
    /// `topic`, starting at `from_offset` if set. The returned stream is the
    /// concatenation of the backing-table replay (offsets `>= from_offset`,
    /// strictly less than the broker's first live offset) followed by the
    /// live broker stream.
    pub async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError> {
        let replay_delivered = self.drain_replay(topic, &predicate, from_offset).await?;
        let last_replayed = replay_delivered.iter().map(|d| d.offset.value()).max();

        // Live tail starts strictly above the replayed prefix so the two
        // halves do not deliver the same offset twice.
        let live_from = match (from_offset, last_replayed) {
            (_, Some(max)) => Some(Offset::new(max + 1, chrono::Utc::now())),
            (Some(off), None) => Some(off),
            (None, None) => None,
        };
        let mut live = self
            .broker
            .subscribe(topic.id, predicate, live_from)
            .await?;

        let stream = try_stream! {
            for delivered in replay_delivered {
                yield delivered;
            }
            while let Some(item) = live.next().await {
                let delivered = item?;
                yield delivered;
            }
        };
        Ok(Subscription::new(SubscriptionId::new(), Box::pin(stream)))
    }

    /// Drain the backing-table replay window without attaching to the live
    /// broker tail. Returns every event with offset `>= from_offset` that the
    /// predicate accepts, in ascending `_offset` order.
    ///
    /// This is the engine-level primitive used by CLI-shaped callers (`jammi
    /// trigger subscribe --no-follow`) that want a finite drain rather than
    /// the infinite tail. Producing a `Vec<DeliveredBatch>` is acceptable
    /// because the caller exits after consuming it; long-running subscribers
    /// should keep using [`Subscriber::subscribe`].
    pub async fn replay_only(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Vec<DeliveredBatch>, TriggerError> {
        self.drain_replay(topic, &predicate, from_offset).await
    }

    /// Shared helper: collect the replay prefix matching `predicate` from the
    /// backing table starting at `from_offset` (defaulting to the live tail
    /// when `None`, in which case the replay is empty).
    async fn drain_replay(
        &self,
        topic: &TopicDefinition,
        predicate: &Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Vec<DeliveredBatch>, TriggerError> {
        let backing_id = MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
        let user_schema = Arc::clone(&topic.schema);

        let replay_batches = match from_offset {
            Some(off) => {
                // `scan_after` is strictly greater than, so subtract one to
                // include `off` itself in the replay window. Using `i64`
                // arithmetic so `Offset(0)` produces `-1` (return every row).
                let scan_after_value = (off.value() as i64).saturating_sub(1);
                let mut stream = self
                    .mutable
                    .scan_after(&backing_id, scan_after_value)
                    .await
                    .map_err(TriggerError::BackingTable)?;
                let mut batches: Vec<RecordBatch> = Vec::new();
                while let Some(b) = stream.next().await {
                    batches.push(b.map_err(TriggerError::BackingTable)?);
                }
                batches
            }
            None => Vec::new(),
        };

        let replay_events = group_replay_batches(&replay_batches, &user_schema)?;
        let mut delivered: Vec<DeliveredBatch> = Vec::with_capacity(replay_events.len());
        for event in replay_events {
            if let Some(filtered) = predicate.evaluate(&event.batch)? {
                delivered.push(DeliveredBatch {
                    offset: event.offset,
                    produced_at: event.produced_at,
                    batch: filtered,
                });
            }
        }
        Ok(delivered)
    }
}

/// One reassembled publish from the backing-table replay path.
struct ReplayEvent {
    offset: Offset,
    produced_at: chrono::DateTime<chrono::Utc>,
    batch: RecordBatch,
}

/// Walk the scan_after results — already in ascending `_offset` order — and
/// reassemble each publish into one `RecordBatch` matching the topic schema.
fn group_replay_batches(
    batches: &[RecordBatch],
    user_schema: &SchemaRef,
) -> Result<Vec<ReplayEvent>, TriggerError> {
    let mut events: Vec<ReplayEvent> = Vec::new();
    let user_field_count = user_schema.fields().len();

    for batch in batches {
        let offset_idx = batch
            .schema()
            .index_of(OFFSET_COLUMN)
            .map_err(|_| TriggerError::Catalog("backing table missing _offset".into()))?;
        let row_idx_idx = batch
            .schema()
            .index_of(ROW_INDEX_COLUMN)
            .map_err(|_| TriggerError::Catalog("backing table missing _row_idx".into()))?;
        let produced_idx = batch
            .schema()
            .index_of(PRODUCED_AT_COLUMN)
            .map_err(|_| TriggerError::Catalog("backing table missing _produced_at".into()))?;

        let offsets = batch
            .column(offset_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TriggerError::Catalog("_offset column must be Int64".into()))?;
        let _row_indices = batch
            .column(row_idx_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TriggerError::Catalog("_row_idx column must be Int64".into()))?;
        let produced = batch
            .column(produced_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TriggerError::Catalog("_produced_at column must be Int64".into()))?;

        // Determine which non-control columns belong to the user payload.
        let mut user_indices: Vec<usize> = Vec::with_capacity(user_field_count);
        for f in user_schema.fields() {
            let i = batch.schema().index_of(f.name()).map_err(|_| {
                TriggerError::Catalog(format!("backing table missing topic column '{}'", f.name()))
            })?;
            user_indices.push(i);
        }

        // Group runs of equal `_offset` (already ascending after Phase 2
        // scan_after's ORDER BY) into one ReplayEvent.
        let mut start = 0usize;
        while start < batch.num_rows() {
            let off = offsets.value(start);
            let mut end = start + 1;
            while end < batch.num_rows() && offsets.value(end) == off {
                end += 1;
            }
            let slice_len = end - start;
            let produced_at_micros = produced.value(start);
            let produced_at =
                DateTime::from_timestamp_micros(produced_at_micros).ok_or_else(|| {
                    TriggerError::Catalog(format!(
                        "_produced_at out of range: {produced_at_micros}"
                    ))
                })?;
            let columns: Vec<ArrayRef> = user_indices
                .iter()
                .map(|&i| batch.column(i).slice(start, slice_len))
                .collect();
            let event_batch = RecordBatch::try_new(Arc::clone(user_schema), columns)
                .map_err(|e| TriggerError::Catalog(e.to_string()))?;
            events.push(ReplayEvent {
                offset: Offset::new(off as u64, produced_at),
                produced_at,
                batch: event_batch,
            });
            start = end;
        }
    }
    Ok(events)
}
