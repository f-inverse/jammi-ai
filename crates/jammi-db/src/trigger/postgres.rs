//! Postgres wake-up `TriggerBroker`: LISTEN/NOTIFY as a transport ONLY.
//!
//! The trait forbids a driver from persisting (`crate::trigger::broker`'s
//! module docs): the topic's mutable backing table IS the durable,
//! authoritative log, and the engine already replays it
//! (`Subscriber`/`TopicTail`) before attaching to a live tail. So this driver
//! carries no bytes and no offset payload past the wire message itself — it
//! only tells a subscriber "topic T may have advanced; go check" via
//! [`crate::trigger::LiveEvent::Wake`], the same signal every driver's lag
//! path already emits. `register_topic`/`drop_topic` are no-ops (F14): there
//! is nothing for this driver to own per topic beyond the in-process wake
//! fan-out `subscribe` lazily creates.
//!
//! # Connections (I6)
//!
//! Up to THREE Postgres connections per process, all dedicated to this
//! broker and never shared with the catalog backend's own pool (the catalog
//! backend is constructed first and is a private pool, `backend_postgres.rs`;
//! F9):
//!
//! - **one** dedicated connection for [`sqlx::postgres::PgListener`], which
//!   `LISTEN`s on the single channel `jammi_trigger` for the process's
//!   lifetime;
//! - **up to two** in a small pool used only to `SELECT pg_notify(...)` from
//!   the coalescing wake task (G9) — never from the publish path itself,
//!   so `publish` enqueues and returns without waiting on a NOTIFY round-trip.
//!
//! # Same-database rule (G12)
//!
//! Every replica's `[broker.postgres] url` MUST point at the SAME database:
//! `NOTIFY` is scoped to one Postgres instance, and a replica listening on a
//! different database silently degrades to `idle_poll`-only delivery (never
//! silently loses data — the tail still replays the authoritative backing
//! table — but live delivery latency degrades to the poll interval with no
//! error surfaced, because there is nothing to detect from either side).
//!
//! # Listener mechanics (F5/I5)
//!
//! The listener task never exits. It drives [`PgListener::try_recv`]
//! concurrently with an idle-tick timer (`tokio::select!`):
//! `Ok(Some(notification))` parses `"{topic_id}:{offset}"` and wakes only
//! that topic; `Ok(None)` means the connection was lost and has been
//! transparently re-established (sqlx's own contract) — notifications during
//! the gap are gone, so every topic is woken; `Err(_)` warns, wakes every
//! topic, and backs off before retrying; a malformed payload warns and wakes
//! every topic (never drops the notification silently); the idle timer wakes
//! every topic every `idle_poll` regardless, bounding how long a lost NOTIFY
//! (a notify-queue overflow, a missed reconnect window) can go undetected.
//!
//! # Pool budget (J8)
//!
//! Each `TopicTail` replay (`crate::trigger::tail`) holds one CATALOG-pool
//! connection for its chunked replay — never one of this broker's own
//! connections. With one tail per `(topic, tenant)` per process, the
//! concurrent-replay bound is the catalog pool size (see
//! `crate::trigger::tail::TailRegistry::new`'s semaphore, K1); size
//! `[catalog.postgres].pool_size` for the expected number of concurrent
//! tenant tails plus writers, independent of this broker's own three-
//! connection budget.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use arrow::record_batch::RecordBatch;
use async_stream::try_stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use sqlx::postgres::{PgConnectOptions, PgListener, PgPool, PgPoolOptions};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::tenant::TenantId;
use crate::trigger::broker::{BrokerKind, TriggerBroker};
use crate::trigger::consumer::ConsumerOffsetSnapshot;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::{SubscriptionId, TopicId};
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{LiveEvent, LiveStream};
use crate::trigger::topic::TopicDefinition;

/// The single NOTIFY channel every replica LISTENs on and NOTIFYs into.
const NOTIFY_CHANNEL: &str = "jammi_trigger";
/// Depth of the coalescing wake queue (G9): bounded so a publish storm can
/// never make `publish` block on the NOTIFY path. An overflow simply drops —
/// the idle tick (I5) is the net that catches it.
const WAKE_QUEUE_CAPACITY: usize = 1024;
/// Per-topic Wake fan-out capacity. `Wake` carries no state to lose (F7), so
/// a lagged receiver here just means "check again" — this only needs to
/// absorb bursts between a subscriber's own `poll_next` calls.
const WAKE_BROADCAST_CAPACITY: usize = 64;
/// Backoff between listener retries after a non-reconnect error, so a
/// persistent failure does not spin the task.
const LISTENER_ERROR_BACKOFF: Duration = Duration::from_millis(500);
/// `application_name` tags so an operator (or a test) can find this broker's
/// own connections in `pg_stat_activity` without guessing pids.
const LISTENER_APPLICATION_NAME: &str = "jammi-trigger-listener";
const NOTIFY_APPLICATION_NAME: &str = "jammi-trigger-notify";

/// Per-subscription bookkeeping mirroring [`crate::trigger::in_memory::InMemoryBroker`]'s
/// tracker: the stream's async body owns the `Arc`, [`Topics`]'s state holds
/// only a `Weak`, so dropping the subscription automatically prunes it from
/// [`TriggerBroker::list_consumers`].
struct ConsumerTracker {
    consumer_name: String,
    topic_id: TopicId,
    /// Best-effort, INFORMATIONAL only: the highest offset this consumer has
    /// observed in a NOTIFY payload. This driver never delivers a batch
    /// itself (only `Wake`), so — unlike `InMemoryBroker`/`JetStreamBroker`,
    /// whose `last_delivered`/`last_acked` are authoritative because they
    /// carry the batch — this value can lag or skip (a coalesced or dropped
    /// NOTIFY, an idle-tick wake) and must never be read as "the last engine
    /// offset this consumer has replayed".
    last_seen_offset: AtomicU64,
}

/// One topic's in-process Wake fan-out plus the consumers currently attached
/// to it.
struct TopicWake {
    sender: broadcast::Sender<Option<u64>>,
    consumers: Vec<Weak<ConsumerTracker>>,
}

type Topics = Arc<RwLock<HashMap<TopicId, TopicWake>>>;

/// Production [`TriggerBroker`] backed by Postgres `LISTEN`/`NOTIFY`. See the
/// module docs.
pub struct PostgresBroker {
    wake_tx: mpsc::Sender<(TopicId, u64)>,
    topics: Topics,
    /// Test-only one-shot hook — see [`Self::suppress_next_notify_for_testing`].
    suppress_next_notify: AtomicBool,
    listener_task: JoinHandle<()>,
    coalesce_task: JoinHandle<()>,
}

// Manual, minimal `Debug` (rather than `#[derive(Debug)]`, which would force
// every field's type transitively into the bound) so this broker is usable
// behind `Result::unwrap_err`/`expect` in tests without printing internal
// channel/task state nobody needs.
impl std::fmt::Debug for PostgresBroker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresBroker").finish_non_exhaustive()
    }
}

impl PostgresBroker {
    /// Open a broker bound to `url` (must be a `postgres://`/`postgresql://`
    /// connection string) with the given idle-tick interval. `idle_poll` must
    /// be non-zero — the bound that caps how long a lost NOTIFY can go
    /// undetected; a zero interval would busy-loop. The config layer's
    /// `[broker.postgres] idle_poll_secs: u64` (K2: `>= 1`) is the only
    /// production path into this constructor and always produces a
    /// whole-second, non-zero `Duration` once validated; this constructor
    /// itself accepts any non-zero `Duration` (including sub-second) for
    /// callers below the config layer (e.g. a test wanting a fast idle tick).
    pub async fn connect(url: &str, idle_poll: Duration) -> Result<Self, TriggerError> {
        if idle_poll.is_zero() {
            return Err(TriggerError::Driver(
                "[broker.postgres] idle_poll_secs must be >= 1".to_string(),
            ));
        }
        if !(url.starts_with("postgres://") || url.starts_with("postgresql://")) {
            return Err(TriggerError::Driver(
                "[broker.postgres] url must be a postgres:// (or postgresql://) URL".to_string(),
            ));
        }

        let listener_opts: PgConnectOptions = url
            .parse()
            .map_err(|e| TriggerError::Driver(format!("postgres broker: parse url: {e}")))?;
        let listener_opts = listener_opts.application_name(LISTENER_APPLICATION_NAME);
        // A dedicated 1-connection pool, exactly what `PgListener::connect`
        // builds internally — constructed by hand here only so the
        // connection carries a distinguishing `application_name` (used by
        // the parity suite's "kill the listener" oracle, and by an operator
        // narrowing `pg_stat_activity`).
        let listener_pool = PgPoolOptions::new()
            .max_connections(1)
            .max_lifetime(None)
            .idle_timeout(None)
            .connect_with(listener_opts)
            .await
            .map_err(|e| TriggerError::Driver(format!("postgres broker: listener connect: {e}")))?;
        let mut listener = PgListener::connect_with(&listener_pool)
            .await
            .map_err(|e| TriggerError::Driver(format!("postgres broker: listener connect: {e}")))?;
        listener.listen(NOTIFY_CHANNEL).await.map_err(|e| {
            TriggerError::Driver(format!("postgres broker: LISTEN {NOTIFY_CHANNEL}: {e}"))
        })?;

        let notify_opts: PgConnectOptions = url
            .parse()
            .map_err(|e| TriggerError::Driver(format!("postgres broker: parse url: {e}")))?;
        let notify_opts = notify_opts.application_name(NOTIFY_APPLICATION_NAME);
        let notify_pool = PgPoolOptions::new()
            .max_connections(2)
            .connect_with(notify_opts)
            .await
            .map_err(|e| {
                TriggerError::Driver(format!("postgres broker: notify pool connect: {e}"))
            })?;

        let topics: Topics = Arc::new(RwLock::new(HashMap::new()));
        let (wake_tx, wake_rx) = mpsc::channel::<(TopicId, u64)>(WAKE_QUEUE_CAPACITY);

        let listener_task = tokio::spawn(listener_loop(listener, Arc::clone(&topics), idle_poll));
        let coalesce_task = tokio::spawn(coalesce_loop(notify_pool, wake_rx));

        Ok(Self {
            wake_tx,
            topics,
            suppress_next_notify: AtomicBool::new(false),
            listener_task,
            coalesce_task,
        })
    }

    /// Test-only hook, mirroring [`crate::trigger::InMemoryBroker::trigger_failure_for_next_publish`]:
    /// arms the broker so the very next [`TriggerBroker::publish`] call skips
    /// enqueueing its wake — simulating a NOTIFY Postgres itself silently
    /// dropped (its own notify queue overflowed; a listener mid-reconnect) —
    /// so the idle-tick fallback (PLAN-F §5(g)) is exercised deterministically
    /// rather than depending on actually losing a real NOTIFY.
    pub fn suppress_next_notify_for_testing(&self) {
        self.suppress_next_notify.store(true, Ordering::SeqCst);
    }
}

impl Drop for PostgresBroker {
    fn drop(&mut self) {
        // Background tasks hold this broker's own connections; abort them so
        // a process (or a test harness constructing many brokers) does not
        // leak live Postgres connections past this broker's lifetime.
        self.listener_task.abort();
        self.coalesce_task.abort();
    }
}

#[async_trait]
impl TriggerBroker for PostgresBroker {
    async fn register_topic(&self, _topic: &TopicDefinition) -> Result<(), TriggerError> {
        // No-op (F14): this driver owns no per-topic state until `subscribe`
        // lazily creates the topic's wake fan-out. Schema conflict detection
        // is the engine's job (`TopicRepo::register_topic`).
        Ok(())
    }

    async fn drop_topic(&self, topic_id: TopicId) -> Result<(), TriggerError> {
        // Idempotent: removing an entry that was never created (this topic
        // had no subscriber on this process) is a no-op.
        self.topics.write().remove(&topic_id);
        Ok(())
    }

    async fn publish(
        &self,
        topic_id: TopicId,
        _batch: RecordBatch,
        produced_at: DateTime<Utc>,
        offset: u64,
        _publish_tenant: Option<TenantId>,
    ) -> Result<Offset, TriggerError> {
        // This driver carries no bytes and no tenant (B5): the batch and the
        // publish-scoped tenant tag are never inspected — the engine's own
        // replay is what any subscriber actually reads once woken.
        if self.suppress_next_notify.swap(false, Ordering::SeqCst) {
            return Ok(Offset::new(offset, produced_at));
        }
        // Best-effort enqueue (G9): a full queue means many topics already
        // have a pending NOTIFY in flight; the idle tick covers the drop.
        let _ = self.wake_tx.try_send((topic_id, offset));
        Ok(Offset::new(offset, produced_at))
    }

    async fn subscribe(
        &self,
        topic_id: TopicId,
        _predicate: Predicate,
        _from_offset: Option<Offset>,
    ) -> Result<LiveStream, TriggerError> {
        // `predicate`/`from_offset` are accepted for trait parity but never
        // consulted (broker.rs's trait doc): filtering happens in the
        // engine's replay, and this driver never refuses a start point — it
        // always yields `Wake` first and lets the engine's replay honour
        // `from_offset` (F2/G3).
        let subscription_id = SubscriptionId::new();
        let tracker = Arc::new(ConsumerTracker {
            consumer_name: subscription_id.to_string(),
            topic_id,
            last_seen_offset: AtomicU64::new(0),
        });

        let mut rx = {
            let mut topics = self.topics.write();
            let state = topics.entry(topic_id).or_insert_with(|| TopicWake {
                sender: broadcast::channel(WAKE_BROADCAST_CAPACITY).0,
                consumers: Vec::new(),
            });
            // Prune weaks whose subscription already dropped so this vector
            // does not grow without bound across a long-lived process.
            state.consumers.retain(|w| w.strong_count() > 0);
            state.consumers.push(Arc::downgrade(&tracker));
            state.sender.subscribe()
        };

        let stream = try_stream! {
            let _tracker_guard = Arc::clone(&tracker);
            // The first item is ALWAYS Wake: this driver never refuses a
            // requested start point, it just says "go check" and lets the
            // engine's replay (which honours `from_offset`) do the rest.
            yield LiveEvent::Wake;
            loop {
                match rx.recv().await {
                    Ok(offset) => {
                        if let Some(o) = offset {
                            tracker.last_seen_offset.fetch_max(o, Ordering::Relaxed);
                        }
                        yield LiveEvent::Wake;
                    }
                    // Wake carries no state to lose (F7) -- a lagged
                    // receiver here still just means "check again".
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        yield LiveEvent::Wake;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        };
        Ok(LiveStream::new(subscription_id, Box::pin(stream)))
    }

    async fn list_consumers(
        &self,
        topic_id: TopicId,
    ) -> Result<Vec<ConsumerOffsetSnapshot>, TriggerError> {
        // `register_topic`/`drop_topic` are no-ops, so this driver keeps no
        // registry of "known" topics distinct from "topics with an active
        // wake fan-out". A topic no subscriber has ever attached to on this
        // process is therefore indistinguishable from one that was never
        // registered — reporting an empty list here (never `TopicNotFound`)
        // is the honest answer for a wake-up transport rather than asserting
        // a distinction this driver cannot make.
        let mut topics = self.topics.write();
        let Some(state) = topics.get_mut(&topic_id) else {
            return Ok(Vec::new());
        };
        let mut snapshots = Vec::with_capacity(state.consumers.len());
        state.consumers.retain(|w| {
            if let Some(tracker) = w.upgrade() {
                let last = tracker.last_seen_offset.load(Ordering::Relaxed);
                snapshots.push(ConsumerOffsetSnapshot {
                    consumer_name: tracker.consumer_name.clone(),
                    topic_id: tracker.topic_id,
                    last_delivered_offset: Some(last),
                    last_acked_offset: Some(last),
                });
                true
            } else {
                false
            }
        });
        Ok(snapshots)
    }

    fn driver_kind(&self) -> BrokerKind {
        BrokerKind::Postgres
    }
}

/// Drains the wake queue, coalescing repeated wakes for the same topic
/// (last-offset-wins) so a publish burst issues one `pg_notify` per topic per
/// drain rather than one per publish (G9). Never exits — a `pg_notify`
/// failure is logged and the idle tick covers it (the next drain retries
/// naturally on the next enqueued wake).
async fn coalesce_loop(notify_pool: PgPool, mut rx: mpsc::Receiver<(TopicId, u64)>) {
    while let Some((first_topic, first_offset)) = rx.recv().await {
        let mut pending: HashMap<TopicId, u64> = HashMap::new();
        pending.insert(first_topic, first_offset);
        while let Ok((topic_id, offset)) = rx.try_recv() {
            pending.insert(topic_id, offset);
        }
        for (topic_id, offset) in pending {
            let payload = format!("{topic_id}:{offset}");
            if let Err(err) = sqlx::query("SELECT pg_notify($1, $2)")
                .bind(NOTIFY_CHANNEL)
                .bind(&payload)
                .execute(&notify_pool)
                .await
            {
                tracing::warn!(
                    error = %err,
                    topic = %topic_id,
                    "postgres broker: pg_notify failed; idle tick will cover it"
                );
            }
        }
    }
}

/// Drives `PgListener::try_recv` concurrently with an idle-tick timer. Never
/// exits (I5) — see the module docs for the full state table.
async fn listener_loop(mut listener: PgListener, topics: Topics, idle_poll: Duration) {
    let mut idle_ticker = tokio::time::interval(idle_poll);
    idle_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // `interval` fires immediately on its first tick; consume it so the very
    // first wake this process observes is the one `subscribe` itself already
    // yields, not a redundant, immediate second one.
    idle_ticker.tick().await;
    loop {
        tokio::select! {
            recv = listener.try_recv() => {
                match recv {
                    Ok(Some(notification)) => match parse_payload(notification.payload()) {
                        Some((topic_id, offset)) => wake_topic(&topics, topic_id, Some(offset)),
                        None => {
                            tracing::warn!(
                                payload = notification.payload(),
                                "postgres broker: malformed NOTIFY payload; waking every topic"
                            );
                            wake_all(&topics);
                        }
                    },
                    Ok(None) => {
                        // Connection lost and transparently re-established
                        // (sqlx's own reconnect contract); notifications
                        // during the gap are gone, so every topic must be
                        // woken to self-heal via replay.
                        wake_all(&topics);
                    }
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "postgres broker: listener error; waking every topic and retrying"
                        );
                        wake_all(&topics);
                        tokio::time::sleep(LISTENER_ERROR_BACKOFF).await;
                    }
                }
            }
            _ = idle_ticker.tick() => {
                wake_all(&topics);
            }
        }
    }
}

/// Parse a NOTIFY payload of the form `"{topic_id}:{offset}"`. `None` on any
/// parse failure — the caller warns and wakes every topic rather than
/// silently dropping the notification.
fn parse_payload(payload: &str) -> Option<(TopicId, u64)> {
    let (topic_str, offset_str) = payload.split_once(':')?;
    let topic_id = TopicId::from_str(topic_str).ok()?;
    let offset = offset_str.parse::<u64>().ok()?;
    Some((topic_id, offset))
}

fn wake_topic(topics: &Topics, topic_id: TopicId, offset: Option<u64>) {
    if let Some(state) = topics.read().get(&topic_id) {
        let _ = state.sender.send(offset);
    }
}

fn wake_all(topics: &Topics) {
    for state in topics.read().values() {
        let _ = state.sender.send(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_payload_accepts_the_documented_shape() {
        let topic_id = TopicId::new();
        let payload = format!("{topic_id}:42");
        let (parsed_id, parsed_offset) = parse_payload(&payload).expect("payload parses");
        assert_eq!(parsed_id, topic_id);
        assert_eq!(parsed_offset, 42);
    }

    #[test]
    fn parse_payload_rejects_malformed_input() {
        assert!(parse_payload("not-a-payload").is_none());
        assert!(parse_payload("").is_none());
        assert!(parse_payload(":42").is_none());
        assert!(parse_payload("not-a-uuid:42").is_none());
        let topic_id = TopicId::new();
        assert!(parse_payload(&format!("{topic_id}:not-a-number")).is_none());
        assert!(parse_payload(&format!("{topic_id}")).is_none());
    }

    /// K2 edge: `idle_poll_secs` must be `>= 1`. Fails before any network
    /// call, so this needs no live Postgres.
    #[tokio::test]
    async fn connect_rejects_zero_idle_poll() {
        let err = PostgresBroker::connect("postgres://u:p@h/db", Duration::from_secs(0))
            .await
            .unwrap_err();
        match err {
            TriggerError::Driver(msg) => assert!(
                msg.contains("idle_poll_secs") && msg.contains(">= 1"),
                "{msg}"
            ),
            other => panic!("expected TriggerError::Driver, got {other:?}"),
        }
    }

    /// K2 edge: `url` must be a `postgres://`/`postgresql://` URL. Fails
    /// before any network call, so this needs no live Postgres.
    #[tokio::test]
    async fn connect_rejects_non_postgres_url() {
        let err = PostgresBroker::connect("nats://nats.svc:4222", Duration::from_secs(5))
            .await
            .unwrap_err();
        match err {
            TriggerError::Driver(msg) => assert!(msg.contains("postgres://"), "{msg}"),
            other => panic!("expected TriggerError::Driver, got {other:?}"),
        }
    }
}
