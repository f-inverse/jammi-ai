//! Replay+live join for trigger-stream subscriptions.
//!
//! The publisher commits every batch to the topic's backing table inside one
//! `CatalogBackend::transaction` and best-effort fans out to the broker. The
//! subscriber stitches a contiguous stream by routing the historical prefix
//! through `drain_replay` and the live portion through a shared
//! `crate::trigger::tail::TopicTail` — one per `(topic, tenant)` per
//! process, owned by [`Subscriber`]'s tail registry — rather than a private
//! driver subscription per call to [`Subscriber::subscribe_scoped`].
//!
//! ## Keying the replay/live seam by engine `_offset`
//!
//! The engine `_offset` (a per-topic monotone counter assigned transactionally
//! by [`crate::trigger::Publisher`]) is the *only* sequence the seam keys on.
//! A broker's own native sequence (JetStream's stream sequence) is an
//! independent counter: after any post-commit fan-out failure — the
//! best-effort path in [`crate::trigger::Publisher`] — the engine offset and
//! the native sequence skew permanently. The tail never hands an engine
//! offset to a driver as if it were a native start-sequence; instead its own
//! driver subscription is always `(Predicate::match_all(), from_offset =
//! None)` (H3), and every gap or `Wake` self-heals through a replay of the
//! backing table (`crate::trigger::tail`'s module docs).
//!
//! `subscribe_scoped` yields the replay prefix (covering `[from_offset ..=
//! last_replayed]`), then attaches to the tail's broadcast and yields live
//! events, deduping by engine `_offset` (only advancing past what replay
//! already covered) so the replay/live overlap never re-delivers what this
//! subscriber has already seen. Predicate filtering happens here, in-process,
//! per subscriber (H3) — the tail itself is predicate-blind.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use async_stream::try_stream;
use chrono::DateTime;
use futures::StreamExt;
use tokio::sync::broadcast;

use crate::catalog::backend::TxOptions;
use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::MutableTableId;
use crate::tenant::TenantId;
use crate::trigger::broker::TriggerBroker;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::SubscriptionId;
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{DeliveredBatch, Subscription};
use crate::trigger::tail::TailRegistry;
use crate::trigger::topic::{TopicDefinition, OFFSET_COLUMN, PRODUCED_AT_COLUMN, ROW_INDEX_COLUMN};

pub struct Subscriber {
    broker: Arc<dyn TriggerBroker>,
    mutable: Arc<MutableTableRegistry>,
    /// One live tail per `(topic, tenant)` this process has ever served a
    /// subscriber for (PLAN-F H1/H6/K1) — see `crate::trigger::tail`.
    tails: TailRegistry,
}

impl Subscriber {
    pub fn new(broker: Arc<dyn TriggerBroker>, mutable: Arc<MutableTableRegistry>) -> Self {
        let pool_size = mutable.backend_arc().catalog_backend().pool_size();
        Self {
            broker,
            mutable,
            tails: TailRegistry::new(pool_size),
        }
    }

    /// Open a subscription that yields every batch matching `predicate` for
    /// `topic`, starting at `from_offset` if set. The returned stream is the
    /// backing-table replay (offsets `>= from_offset`) followed by the live
    /// broker stream, which overlaps the replayed prefix and is deduped by
    /// engine `_offset` (see the module docs). No engine `_offset` is skipped.
    ///
    /// Resolves the tenant binding once, here, to filter both the
    /// backing-table replay and the live broker tail. The resulting stream
    /// contains data only from that tenant (plus globally-scoped rows), even
    /// if the caller polls it after the surrounding
    /// [`crate::session::JammiSession::with_tenant_scoped`] closure has
    /// returned and the task-local binding has cleared.
    ///
    /// For server-streaming gRPC handlers that return the stream past the
    /// closure boundary, prefer [`Self::subscribe_scoped`]: it takes the
    /// tenant explicitly so the binding does not have to be inferred from
    /// whatever task-local is in effect at this method's await point.
    pub async fn subscribe(
        &self,
        topic: &TopicDefinition,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError> {
        let tenant = self.mutable.binding().current_tenant();
        self.subscribe_scoped(topic, tenant, predicate, from_offset)
            .await
    }

    /// Open a subscription with an explicit `tenant` binding for the
    /// backing-table replay query.
    ///
    /// The replay's tenant predicate is computed from `tenant` at subscribe
    /// time and the resulting rows are materialised inside this call — no
    /// subsequent `poll_next` consults `current_tenant()` for replay. The
    /// live broker tail is filtered the same way, per delivered event: each
    /// live [`DeliveredBatch`] carries the publish-scoped tenant tag the
    /// broker stamped on it opaquely (see
    /// [`crate::trigger::broker::TriggerBroker::publish`]), and this seam
    /// yields it only when that tag equals `tenant` or is absent (a
    /// globally-scoped publish) — mirroring the replay's own
    /// `tenant_id = $current OR tenant_id IS NULL` predicate. `topic.id` is
    /// NOT itself a tenant partition: a globally-registered topic
    /// (`TopicDefinition::tenant == None`) shares one `topic.id` across every
    /// tenant, so without this per-event filter the live tail would deliver
    /// every tenant's events to every subscriber. Together the replay filter
    /// and this live filter guarantee the returned stream stays inside
    /// `tenant`'s data even when polled outside any surrounding
    /// `with_tenant_scoped` block.
    ///
    /// This is the safe primitive for gRPC server-streaming handlers that
    /// return the stream to tonic past the request closure boundary:
    ///
    /// ```ignore
    /// async fn watch(&self, req) -> Result<Response<Stream>, Status> {
    ///     let tenant = extract_tenant(&req)?;
    ///     let topic  = self.lookup_topic(req, tenant).await?;
    ///     let stream = self
    ///         .subscriber
    ///         .subscribe_scoped(&topic, Some(tenant), predicate, from_offset)
    ///         .await?;
    ///     Ok(Response::new(Box::pin(stream)))
    /// }
    /// ```
    pub async fn subscribe_scoped(
        &self,
        topic: &TopicDefinition,
        tenant: Option<TenantId>,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError> {
        let replay_delivered = self
            .drain_replay(topic, tenant, &predicate, from_offset)
            .await?;
        let last_replayed = replay_delivered.iter().map(|d| d.offset.value()).max();

        // A live-only subscriber (`from_offset = None`) did no replay above,
        // so seed the cursor with the current tenant-blind watermark (the
        // highest committed `_offset` at subscribe time) rather than leaving
        // it unset — otherwise the first `LiveEvent::Wake` this subscriber
        // sees would replay the ENTIRE backing-table history instead of only
        // what committed after this point. A publish racing this read is
        // still delivered, either by this read observing it or by the very
        // next live `Batch`/`Wake`.
        let watermark = if from_offset.is_none() {
            current_watermark(&self.mutable, topic).await?
        } else {
            None
        };

        // G3: attach to the tail's broadcast BEFORE reading any watermark —
        // rows fanned out before this attach have offset <= the watermark
        // (read just above) and are suppressed by dedup below; rows
        // committing after the attach are fanned out after it, so nothing
        // is missed. `_tail_guard` keeps the tail's `Arc` (and therefore its
        // task and driver subscription) alive for exactly as long as this
        // subscription is polled (H6).
        let (tail_guard, mut rx) = self
            .tails
            .attach_or_create(&self.broker, &self.mutable, topic, tenant)
            .await?;

        let mutable = Arc::clone(&self.mutable);
        let topic_owned = topic.clone();

        let stream = try_stream! {
            let _tail_guard = tail_guard;
            // Highest engine `_offset` already yielded. Seeded with the
            // replay high-water mark (or, for a live-only subscribe, the
            // watermark read above) so live events inside the overlap window
            // — or this subscriber's own lag replay — never re-deliver what
            // it has already seen or was never asked for.
            let mut last_yielded = last_replayed.or(watermark);
            for delivered in replay_delivered {
                yield delivered;
            }
            loop {
                match rx.recv().await {
                    Ok(delivered) => {
                        // The tail is keyed on `(topic, tenant)` and already
                        // scopes its fan-out to exactly this tenant (H1) —
                        // no further tenant filter needed here. Dedup by
                        // engine `_offset` ALWAYS advances on a higher
                        // offset, independent of the predicate: the tail's
                        // own driver subscription is `Predicate::match_all`
                        // (H3), so predicate filtering is this subscriber's
                        // job, applied in-process, and must not be confused
                        // with the dedup cursor (a row this predicate
                        // rejects has still been "seen").
                        if last_yielded.is_none_or(|seen| delivered.offset.value() > seen) {
                            last_yielded = Some(delivered.offset.value());
                            if let Some(filtered) = predicate.evaluate(&delivered.batch)? {
                                yield DeliveredBatch { batch: filtered, ..delivered };
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_n)) => {
                        // G4/K2: this subscriber's OWN chunked, group-
                        // completing replay from its own `last_yielded` —
                        // never `drain_replay`'s whole-suffix materialisation
                        // and never the tail's shared cursor (a lag here is
                        // this receiver's own backlog, not the tail's).
                        let from = last_yielded.map(|o| o as i64).unwrap_or(-1);
                        let more = crate::trigger::tail::lag_replay(&mutable, &topic_owned, tenant, from)
                            .await?;
                        for delivered in more {
                            if last_yielded.is_none_or(|seen| delivered.offset.value() > seen) {
                                last_yielded = Some(delivered.offset.value());
                                if let Some(filtered) = predicate.evaluate(&delivered.batch)? {
                                    yield DeliveredBatch { batch: filtered, ..delivered };
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
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
        let tenant = self.mutable.binding().current_tenant();
        self.drain_replay(topic, tenant, &predicate, from_offset)
            .await
    }

    /// Explicit-tenant variant of [`Self::replay_only`]. The replay query
    /// uses `tenant` directly rather than consulting the session binding,
    /// matching the safety contract of [`Self::subscribe_scoped`].
    pub async fn replay_only_scoped(
        &self,
        topic: &TopicDefinition,
        tenant: Option<TenantId>,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Vec<DeliveredBatch>, TriggerError> {
        self.drain_replay(topic, tenant, &predicate, from_offset)
            .await
    }

    /// Shared helper: collect the replay prefix matching `predicate` from the
    /// backing table starting at `from_offset` (defaulting to the live tail
    /// when `None`, in which case the replay is empty). The `tenant`
    /// argument is baked into the backend SQL — no task-local lookup
    /// happens inside the underlying scan.
    ///
    /// Delegates to the free [`drain_replay`] function, which takes the
    /// [`MutableTableRegistry`] handle by `Arc` reference rather than `&self`,
    /// matching the shape [`Self::subscribe_scoped`]'s `'static` live-tail
    /// stream body needs for its own calls into the backing table (a
    /// subscriber's lag replay, `crate::trigger::tail::lag_replay`) without
    /// borrowing `self`.
    async fn drain_replay(
        &self,
        topic: &TopicDefinition,
        tenant: Option<TenantId>,
        predicate: &Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Vec<DeliveredBatch>, TriggerError> {
        drain_replay(&self.mutable, topic, tenant, predicate, from_offset).await
    }
}

/// Read the current tenant-blind watermark (`MAX("_offset")`) on `topic`'s
/// backing table, or `None` if the table has no rows yet. Used to seed a
/// live-only subscriber's dedup cursor (see [`Subscriber::subscribe_scoped`])
/// so a subsequent [`crate::trigger::LiveEvent::Wake`] replays only what
/// committed after subscribe time, not the entire history.
async fn current_watermark(
    mutable: &Arc<MutableTableRegistry>,
    topic: &TopicDefinition,
) -> Result<Option<u64>, TriggerError> {
    let backing = topic.backing_table_name();
    let sql = format!(
        "SELECT MAX(\"{OFFSET_COLUMN}\") AS m FROM \"{}\"",
        backing.replace('"', "\"\"")
    );
    let backend = mutable.backend_arc();
    let rows: Vec<Option<i64>> = backend
        .catalog_backend()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            move |tx| {
                let sql = sql.clone();
                Box::pin(async move { tx.query(&sql, &[], |row| row.try_get::<i64>("m")).await })
            },
        )
        .await
        .map_err(TriggerError::Backend)?;
    Ok(rows.into_iter().next().flatten().map(|v| v as u64))
}

/// Free-function form of [`Subscriber::drain_replay`] — see its docs. Exists
/// so [`Subscriber::subscribe_scoped`]'s `'static` live-tail stream body can
/// call it for the initial replay prefix without holding a `&self` borrow
/// across the stream. The live portion no longer calls back into this
/// function on a lag/gap/wake — that self-healing replay is
/// `crate::trigger::tail`'s job (the tail's own driver-triggered replay, or
/// a subscriber's own `crate::trigger::tail::lag_replay`), both of which
/// reuse `MutableTableRegistry::tail_replay`'s chunked, group-completing
/// query instead of this function's whole-suffix materialisation.
async fn drain_replay(
    mutable: &Arc<MutableTableRegistry>,
    topic: &TopicDefinition,
    tenant: Option<TenantId>,
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
            let mut stream = mutable
                .scan_after_for_tenant(&backing_id, scan_after_value, tenant)
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
                // The replay path is already tenant-filtered by the
                // `scan_after_for_tenant` predicate above, so this tag
                // carries no further meaning here and is not authoritative
                // — unlike the live tail, which relies on it (see
                // `subscribe_scoped`'s live-branch filter).
                tenant: None,
            });
        }
    }
    Ok(delivered)
}

/// One reassembled publish from the backing-table replay path.
pub(crate) struct ReplayEvent {
    pub(crate) offset: Offset,
    pub(crate) produced_at: chrono::DateTime<chrono::Utc>,
    pub(crate) batch: RecordBatch,
}

/// Walk the scan_after results — already in ascending `_offset` order — and
/// reassemble each publish into one `RecordBatch` matching the topic schema.
///
/// `pub(crate)`: also used by [`crate::trigger::tail::TopicTail`]'s chunked
/// replay, which reassembles rows fetched via
/// [`crate::source::mutable::MutableTableRegistry::tail_replay`] the same
/// way this subscribe-time replay does.
pub(crate) fn group_replay_batches(
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
