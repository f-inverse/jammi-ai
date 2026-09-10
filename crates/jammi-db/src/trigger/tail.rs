//! Per-`(topic, tenant)` live-tail actor.
//!
//! `Subscriber` owns one `TopicTail` per `(topic, tenant)` pair it has ever
//! served a subscriber for. A tail holds the single driver-level
//! [`crate::trigger::subscription::LiveStream`] subscription for that scope
//! and fans out a tenant-blind [`DeliveredBatch`] stream to every subscriber
//! of that `(topic, tenant)` through a `tokio::sync::broadcast` channel —
//! `N` subscribers of the same topic/tenant cost ONE driver subscription and
//! ONE replay per wake, not `N`.
//!
//! ## Contiguity-checked fan-out
//!
//! The tail keeps a cursor: the last **engine `_offset`** it has delivered or
//! replayed, in *global* offset space (tenant-blind). A driver
//! [`LiveEvent::Batch`] is fanned out directly ONLY when its offset equals
//! `cursor + 1` (then the cursor advances by one); a gap (`offset` more than
//! one past `cursor`) or regression (`offset` at or below `cursor`) drops the
//! batch instead, and for a gap, triggers a replay from the cursor. A
//! [`LiveEvent::Wake`] always replays. Post-commit fan-out across
//! replicas/drivers is unordered, so this contiguity check — not the
//! driver's own delivery order — is what makes "multi-replica publish is
//! correct" true for every driver.
//!
//! One task owns the cursor and processes driver events and replays
//! SERIALLY: a `Batch` arriving mid-replay simply waits in the driver's
//! own broadcast (overflow there self-heals via `Lagged` → `Wake` →
//! replay). The cursor can therefore never regress.
//!
//! ## Replay
//!
//! A replay is one or more STEPS of
//! [`crate::source::mutable::MutableTableRegistry::tail_replay`], each its
//! own read-only transaction: a step reads the tenant-blind head `MAX(_offset)`
//! = `H` FIRST, then fetches ONE `chunk_size`-row-bounded group of
//! tenant-scoped rows `_offset > cursor AND _offset <= H` (a group that
//! straddles the chunk boundary is fetched whole, never split), and reports
//! whether `H` was reached. [`replay_and_fan_out`] loops calling `tail_replay`
//! — fanning out each step's rows immediately — until a step reports `H`
//! reached, then advances the cursor to `H`; a lagging subscriber's own catch
//! up ([`lag_replay`]) is the SAME one-step primitive, looped the same way by
//! its caller (`crate::trigger::subscriber`'s `subscribe_scoped`). Two
//! things are bounded, not one: CONCURRENCY (every replay call — a tail's own
//! driver-triggered replay, a lagging subscriber's own replay, and
//! `Subscriber`'s subscribe-time backing-table drain — acquires a permit from
//! the semaphore owned by [`MutableTableRegistry`], sized `pool_size - 2`
//! (min 1), before opening its transaction, so replays collectively can never
//! starve publishers of pool connections) and PER-STEP RESIDENCY (each
//! `tail_replay` call materialises at most one step's rows, never the whole
//! gap between `cursor` and `H` — a catch-up from a far-behind cursor to a
//! busy topic's head resides in memory one step at a time, never the entire
//! backlog at once).
//!
//! ## Tenant scope
//!
//! A tail is keyed on `(topic_id, tenant)`, not on topic alone: there is no
//! all-tenants replay query (`tail_replay(tenant = None)` renders
//! `tenant_id IS NULL`, i.e. *global rows only*), so each tail's OWN replay
//! is scoped to its own tenant (tenant rows + global rows) — exactly
//! `Subscriber::subscribe_scoped`'s existing semantics, now shared across
//! every subscriber of that tenant instead of computed per subscriber. A
//! `tenant = None` tail serves globally-scoped subscribers. The driver
//! subscription underneath every tail is identical —
//! `(Predicate::match_all(), from_offset = None)` — and tenant-blind;
//! the tenant filter is applied when a `Batch` is fanned out directly, and
//! is baked into the SQL for a replay.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use futures::StreamExt;
use tokio::sync::{broadcast, Mutex as AsyncMutex};
use tokio::task::JoinHandle;

use crate::source::mutable::MutableTableRegistry;
use crate::store::mutable::definition::MutableTableDefinition;
use crate::store::mutable::MutableTableError;
use crate::tenant::TenantId;
use crate::trigger::broker::TriggerBroker;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::TopicId;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscriber::group_replay_batches;
use crate::trigger::subscription::{DeliveredBatch, LiveEvent};
use crate::trigger::topic::TopicDefinition;

/// Broadcast capacity for a tail's fan-out channel. A subscriber that lags
/// behind this self-heals via its OWN chunked replay from its last-yielded
/// offset rather than erroring, so this only needs to absorb short
/// bursts between a subscriber's `poll_next` calls.
const TAIL_BROADCAST_CAPACITY: usize = 256;

/// Rows fetched per replay chunk before a group-completion probe. Kept
/// small enough that one tail replay never holds its pool connection for an
/// unbounded time under a very wide backlog.
pub(crate) const REPLAY_CHUNK_SIZE: usize = 500;

/// One `(topic, tenant)` live tail. See the module docs.
pub(crate) struct TopicTail {
    sender: broadcast::Sender<DeliveredBatch>,
    task: JoinHandle<()>,
}

impl TopicTail {
    /// Subscribe to this tail's fan-out. Callers attach BEFORE reading any
    /// watermark so nothing committed after the attach is missed.
    pub(crate) fn attach(&self) -> broadcast::Receiver<DeliveredBatch> {
        self.sender.subscribe()
    }
}

impl Drop for TopicTail {
    fn drop(&mut self) {
        // Dropping the tail only aborts its task (which owns the driver
        // `LiveStream` and releases it). It never removes the registry's key
        // — a successor tail may already occupy it by the time this runs;
        // stale `Weak`s are pruned on the registry's next insert.
        self.task.abort();
    }
}

/// Key a tail is registered under: the topic and the tenant scope its own
/// replay query is bound to (`None` = the globally-scoped tail).
type TailKey = (TopicId, Option<TenantId>);

/// Registry of live tails keyed by `(topic_id, tenant)`, owned by
/// [`crate::trigger::Subscriber`].
pub(crate) struct TailRegistry {
    tails: AsyncMutex<HashMap<TailKey, Weak<TopicTail>>>,
}

impl TailRegistry {
    pub(crate) fn new() -> Self {
        Self {
            tails: AsyncMutex::new(HashMap::new()),
        }
    }

    /// Get or create the tail for `(topic.id, tenant)` and attach to it,
    /// returning the strong handle (kept alive by the caller for the
    /// lifetime of its subscription) and a fresh broadcast receiver.
    ///
    /// The whole upgrade-or-create-and-attach sequence runs under ONE lock
    /// acquisition ("atomic against teardown") — a concurrent
    /// `TopicTail::drop` running its `Drop` impl cannot race a fresh
    /// `attach()` into observing a half-torn-down tail, because the
    /// registry's `Weak` entry is only replaced here, under this same lock.
    pub(crate) async fn attach_or_create(
        &self,
        broker: &Arc<dyn TriggerBroker>,
        mutable: &Arc<MutableTableRegistry>,
        topic: &TopicDefinition,
        tenant: Option<TenantId>,
    ) -> Result<(Arc<TopicTail>, broadcast::Receiver<DeliveredBatch>), TriggerError> {
        let key = (topic.id, tenant);
        let mut tails = self.tails.lock().await;
        // Prune stale `Weak`s (their `TopicTail` already dropped) on
        // every insert path, not just this key, so the map does not grow
        // without bound across the lifetime of a long-lived process.
        tails.retain(|_, w| w.strong_count() > 0);
        if let Some(existing) = tails.get(&key).and_then(Weak::upgrade) {
            let rx = existing.attach();
            return Ok((existing, rx));
        }
        let created = spawn_tail(broker, mutable, topic, tenant).await?;
        let rx = created.attach();
        tails.insert(key, Arc::downgrade(&created));
        Ok((created, rx))
    }
}

/// Build a fresh [`TopicTail`]: register its driver-level subscription
/// FIRST (`(Predicate::match_all(), from_offset = None)`, tenant-blind),
/// resolve the backing-table definition ONCE (a tail must never hold two
/// connections at once), read the tenant-blind initial cursor (never `0`,
/// or the first `Wake` would replay the entire history), then spawn the
/// task that owns the cursor for the rest of the tail's life.
async fn spawn_tail(
    broker: &Arc<dyn TriggerBroker>,
    mutable: &Arc<MutableTableRegistry>,
    topic: &TopicDefinition,
    tenant: Option<TenantId>,
) -> Result<Arc<TopicTail>, TriggerError> {
    let driver = broker
        .subscribe(topic.id, Predicate::match_all(), None)
        .await?;

    let backing_id =
        crate::store::mutable::definition::MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
    let def = mutable.definition_for_tenant(&backing_id, tenant).await?;
    let order_col = def.order_column.clone().ok_or_else(|| {
        TriggerError::Catalog(format!(
            "topic '{}' backing table has no order_column",
            topic.name
        ))
    })?;

    let initial_cursor = read_tenant_blind_head(mutable, &def, &order_col)
        .await?
        .unwrap_or(-1);

    let (sender, _initial_rx) = broadcast::channel(TAIL_BROADCAST_CAPACITY);
    let sender_for_task = sender.clone();
    let mutable_for_task = Arc::clone(mutable);
    let user_schema = Arc::clone(&topic.schema);

    let task = tokio::spawn(run_tail_loop(
        driver,
        sender_for_task,
        mutable_for_task,
        def,
        order_col,
        tenant,
        user_schema,
        initial_cursor,
    ));

    Ok(Arc::new(TopicTail { sender, task }))
}

/// Read the tenant-blind `MAX(order_col)` on `def`'s backing table, or
/// `None` if it has no rows yet. Runs its own tiny transaction — this is a
/// one-shot call at tail creation, not part of the per-wake replay loop.
async fn read_tenant_blind_head(
    mutable: &Arc<MutableTableRegistry>,
    def: &MutableTableDefinition,
    order_col: &str,
) -> Result<Option<i64>, TriggerError> {
    // A dedicated head-only query rather than routing through
    // `MutableTableRegistry::tail_replay`: that call always fetches at least
    // one chunk of rows once `cursor_before < head`, which would fetch (and
    // discard) the entire backing-table history just to read a watermark.
    let backend = mutable.backend_arc();
    let sql = format!(
        "SELECT MAX(\"{}\") AS m FROM \"{}\"",
        order_col.replace('"', "\"\""),
        def.id.as_str().replace('"', "\"\"")
    );
    let rows: Vec<Option<i64>> = backend
        .catalog_backend()
        .transaction(
            crate::catalog::backend::TxOptions {
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
    Ok(rows.into_iter().next().flatten())
}

/// Whether a fanned-out row belongs to `tail_tenant`'s scope: a globally
/// published row (`row_tenant.is_none()`) is visible to every tail, and a
/// `None` (global) tail only ever fans out globally published rows —
/// mirroring `Subscriber`'s `tenant_id = $current OR tenant_id IS NULL`
/// replay predicate.
fn tenant_visible(tail_tenant: Option<TenantId>, row_tenant: Option<TenantId>) -> bool {
    row_tenant.is_none() || row_tenant == tail_tenant
}

/// The tail task body. Owns the cursor for the tail's entire lifetime;
/// processes driver events and replays serially, so the cursor can
/// never regress.
#[allow(clippy::too_many_arguments)]
async fn run_tail_loop(
    mut driver: crate::trigger::subscription::LiveStream,
    sender: broadcast::Sender<DeliveredBatch>,
    mutable: Arc<MutableTableRegistry>,
    def: MutableTableDefinition,
    order_col: String,
    tenant: Option<TenantId>,
    user_schema: SchemaRef,
    initial_cursor: i64,
) {
    let mut cursor = initial_cursor;
    loop {
        match driver.next().await {
            None => {
                // The driver's own stream ended (broker shutdown). Nothing
                // further to deliver; exit rather than spin.
                return;
            }
            Some(Ok(LiveEvent::Batch(delivered))) => {
                let off = delivered.offset.value() as i64;
                if off == cursor + 1 {
                    cursor = off;
                    if tenant_visible(tenant, delivered.tenant) {
                        // A closed channel (zero receivers) is not a fan-out
                        // failure — the same rule `InMemoryBroker::publish`
                        // documents.
                        let _ = sender.send(delivered);
                    }
                } else if off > cursor + 1 {
                    cursor = replay_and_fan_out(
                        &mutable,
                        &def,
                        &order_col,
                        tenant,
                        cursor,
                        &user_schema,
                        &sender,
                    )
                    .await;
                }
                // `off <= cursor`: regression — drop without replay.
            }
            Some(Ok(LiveEvent::Wake)) => {
                cursor = replay_and_fan_out(
                    &mutable,
                    &def,
                    &order_col,
                    tenant,
                    cursor,
                    &user_schema,
                    &sender,
                )
                .await;
            }
            Some(Err(err)) => {
                // No driver is documented to emit a bare stream error today
                // (every driver routes a lag/gap through `Wake` instead —
                // `broker.rs`'s trait doc). Treat it the same way regardless:
                // self-heal by replay rather than tearing the tail down, so
                // a future driver that DOES surface a transient error keeps
                // the tail alive.
                tracing::warn!(error = %err, "trigger tail: driver stream error; replaying");
                cursor = replay_and_fan_out(
                    &mutable,
                    &def,
                    &order_col,
                    tenant,
                    cursor,
                    &user_schema,
                    &sender,
                )
                .await;
            }
        }
    }
}

/// Drive [`MutableTableRegistry::tail_replay`] to completion — looping over
/// its one-step-at-a-time contract, fanning out each step's rows BEFORE
/// fetching the next one — and return the final cursor. Bounds resident
/// memory to one step's rows at a time (`tail_replay`'s own doc), never the
/// whole gap between `cursor_before` and the topic's head, however wide a
/// backlog a `Wake` or gap is catching up from. On any replay error the
/// cursor is left at wherever the last successful step advanced it (never
/// past rows that were never actually delivered) — the next `Wake` or gap
/// retries from there.
#[allow(clippy::too_many_arguments)]
async fn replay_and_fan_out(
    mutable: &Arc<MutableTableRegistry>,
    def: &MutableTableDefinition,
    order_col: &str,
    tenant: Option<TenantId>,
    cursor_before: i64,
    user_schema: &SchemaRef,
    sender: &broadcast::Sender<DeliveredBatch>,
) -> i64 {
    let mut cursor = cursor_before;
    loop {
        match mutable
            .tail_replay(def, order_col, tenant, cursor, REPLAY_CHUNK_SIZE)
            .await
        {
            Ok((raw_batches, new_cursor, drained)) => {
                match group_replay_batches(&raw_batches, user_schema) {
                    Ok(events) => {
                        for event in events {
                            let delivered = DeliveredBatch {
                                offset: event.offset,
                                produced_at: event.produced_at,
                                batch: event.batch,
                                // The replay query is already tenant-scoped —
                                // this tag is informational only, matching
                                // `Subscriber::drain_replay`'s own replay path.
                                tenant: None,
                            };
                            let _ = sender.send(delivered);
                        }
                        cursor = new_cursor;
                        if drained {
                            return cursor;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "trigger tail: replay group reassembly failed");
                        return cursor;
                    }
                }
            }
            Err(err) => {
                let err: TriggerError = match err {
                    MutableTableError::Backend(b) => TriggerError::Backend(b),
                    other => TriggerError::BackingTable(other),
                };
                tracing::warn!(error = %err, "trigger tail: replay failed; cursor unchanged");
                return cursor;
            }
        }
    }
}

/// A single subscriber's own lag recovery, ONE STEP at a time: when a
/// subscriber's broadcast receiver observes `RecvError::Lagged`, it replays
/// from its OWN `last_yielded`/floor rather than erroring or falling back to
/// `Subscriber::drain_replay`'s whole-suffix materialisation. This reuses the
/// SAME one-step, group-completing replay primitive the tail itself loops
/// over for a driver-level `Wake` (`replay_and_fan_out`) — never a second,
/// less-bounded code path.
///
/// Returns ONE step's tenant-scoped (already-filtered by the backing query)
/// but NOT predicate-filtered events, the new cursor
/// (`from_offset_exclusive`'s replacement for the caller's next call), and
/// whether the topic's head was reached. The caller
/// (`crate::trigger::subscriber`'s `subscribe_scoped`) loops calling this
/// again with the returned cursor until `drained`, consuming (dedup-checking,
/// predicate-filtering, and `yield`ing) each step's events before the next
/// call — so a catch-up from a far-behind cursor never resides in memory
/// as more than one step's events at a time, and applies the predicate
/// uniformly across every source (live relay and lag replay alike), matching
/// the rule that predicate and dedup are per subscriber, in-process.
pub(crate) async fn lag_replay(
    mutable: &Arc<MutableTableRegistry>,
    topic: &TopicDefinition,
    tenant: Option<TenantId>,
    from_offset_exclusive: i64,
) -> Result<(Vec<DeliveredBatch>, i64, bool), TriggerError> {
    let backing_id =
        crate::store::mutable::definition::MutableTableId::new(topic.backing_table_name())
            .map_err(|e| TriggerError::Catalog(e.to_string()))?;
    let def = mutable.definition_for_tenant(&backing_id, tenant).await?;
    let order_col = def.order_column.clone().ok_or_else(|| {
        TriggerError::Catalog(format!(
            "topic '{}' backing table has no order_column",
            topic.name
        ))
    })?;
    let (raw_batches, new_cursor, drained) = mutable
        .tail_replay(
            &def,
            &order_col,
            tenant,
            from_offset_exclusive,
            REPLAY_CHUNK_SIZE,
        )
        .await
        .map_err(|e| match e {
            MutableTableError::Backend(b) => TriggerError::Backend(b),
            other => TriggerError::BackingTable(other),
        })?;
    let events = group_replay_batches(&raw_batches, &topic.schema)?;
    let delivered = events
        .into_iter()
        .map(|event| DeliveredBatch {
            offset: event.offset,
            produced_at: event.produced_at,
            batch: event.batch,
            tenant: None,
        })
        .collect();
    Ok((delivered, new_cursor, drained))
}
