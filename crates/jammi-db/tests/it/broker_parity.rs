//! Cross-driver parity suite (contract item 2).
//!
//! Parameterised over [`Arm::InMemory`] (always), [`Arm::Postgres`]
//! (`JAMMI_TEST_PG_URL`; runs in the `test-pg` job, no cargo feature),
//! and [`Arm::JetStream`] (`live-broker-tests`, requiring
//! `JAMMI_TEST_NATS_URL`; `JAMMI_REQUIRE_NATS` turns an unset URL into a hard
//! failure rather than a silent skip — the same require-gate shape
//! `recovery.rs`'s `require_live_pg` uses). Every arm exercises the SAME
//! engine-facing surface (`Subscriber`/`Publisher`/`TopicRepo`) over a fresh
//! SQLite catalog, so a passing suite proves the `Subscriber`/`TopicTail`
//! seam behaves identically regardless of which driver sits underneath it —
//! the Postgres arm's backing table stays on SQLite; only the broker (a
//! wake-up transport, never the log) is real Postgres `LISTEN`/`NOTIFY`. The
//! offset-order == commit-order oracle runs against a real Postgres
//! catalog instead (two sessions sharing one database), gated by
//! `live-postgres-tests`.
//!
//! Two Postgres-only oracles below are not test-case-parameterised because
//! they need the CONCRETE `PostgresBroker` type (a test-only hook /
//! `pg_terminate_backend`), not the `Arc<dyn TriggerBroker>` trait object
//! `broker_for` returns: `postgres_listener_killed_recovers_via_replay`
//! and `postgres_suppressed_notify_recovers_via_idle_tick`.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use jammi_db::catalog::backend::BackendImpl;
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::catalog::Catalog;
use jammi_db::source::mutable::MutableTableRegistry;
use jammi_db::store::mutable::postgres::PostgresMutableBackend;
use jammi_db::store::mutable::sqlite::SqliteMutableBackend;
use jammi_db::store::mutable::MutableBackend;
use jammi_db::tenant::TenantId;
use jammi_db::tenant_scope::TenantBinding;
#[cfg(feature = "jetstream-broker")]
use jammi_db::trigger::JetStreamBroker;
use jammi_db::trigger::{
    InMemoryBroker, LiveEvent, Offset, PostgresBroker, Predicate, Publisher, Subscriber,
    TopicDefinition, TopicId, TriggerBroker,
};
use test_case::test_case;

/// Require-gate (same shape as `recovery.rs`'s `require_live_pg`): a lane
/// that wants to REQUIRE the real JetStream arm sets `JAMMI_REQUIRE_NATS`,
/// turning an unset `JAMMI_TEST_NATS_URL` into a panic instead of a skip.
#[cfg(feature = "jetstream-broker")]
fn require_live_nats(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_NATS").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_NATS is set but JAMMI_TEST_NATS_URL is unset -- this \
             lane must run the real JetStream arm, not skip it"
        );
    }
}

fn require_live_pg(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_PG").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_PG is set but JAMMI_TEST_PG_URL is unset -- this lane \
             must run the real Postgres arm, not skip it"
        );
    }
}

/// Driver arm selector.
#[derive(Clone, Copy)]
enum Arm {
    InMemory,
    Postgres,
    #[cfg(feature = "jetstream-broker")]
    JetStream,
}

/// `idle_poll` for every `Arm::Postgres` broker built in this suite: short
/// enough that the idle-tick fallback (used by every replay-triggered
/// assertion below, not only the dedicated idle-tick oracle) never makes a
/// test wait long, without being so short it flakes under CI scheduling
/// jitter.
const TEST_IDLE_POLL: Duration = Duration::from_secs(1);

/// Build the driver for `arm`, or `None` for a live-broker arm whose env var
/// is unset (never `#[ignore]`).
#[cfg_attr(not(feature = "jetstream-broker"), allow(unused_variables))]
async fn broker_for(arm: Arm, test_name: &str) -> Option<Arc<dyn TriggerBroker>> {
    match arm {
        Arm::InMemory => Some(Arc::new(InMemoryBroker::new()) as Arc<dyn TriggerBroker>),
        Arm::Postgres => {
            let url = match jammi_test_utils::pg_url_for_tests() {
                Some(u) => u,
                None => {
                    eprintln!("skipping {test_name}: JAMMI_TEST_PG_URL unset");
                    require_live_pg(test_name);
                    return None;
                }
            };
            let broker = PostgresBroker::connect(&url, TEST_IDLE_POLL)
                .await
                .expect("connect to Postgres broker");
            Some(Arc::new(broker) as Arc<dyn TriggerBroker>)
        }
        #[cfg(feature = "jetstream-broker")]
        Arm::JetStream => {
            let url = match std::env::var("JAMMI_TEST_NATS_URL") {
                Ok(u) => u,
                Err(_) => {
                    eprintln!("skipping {test_name}: JAMMI_TEST_NATS_URL unset");
                    require_live_nats(test_name);
                    return None;
                }
            };
            let broker = JetStreamBroker::connect(&url, 60)
                .await
                .expect("connect to JetStream");
            Some(Arc::new(broker) as Arc<dyn TriggerBroker>)
        }
    }
}

/// Skip (never `#[ignore]`) when `broker_for` returned `None`.
macro_rules! broker_or_skip {
    ($arm:expr, $name:expr) => {
        match broker_for($arm, $name).await {
            Some(b) => b,
            None => return,
        }
    };
}

struct Harness {
    _dir: tempfile::TempDir,
    registry: Arc<MutableTableRegistry>,
    topic_repo: TopicRepo,
    backend_arc: Arc<BackendImpl>,
    broker: Arc<dyn TriggerBroker>,
    publisher: Publisher,
    subscriber: Subscriber,
}

async fn build_harness(broker: Arc<dyn TriggerBroker>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
        .await
        .unwrap();
    let backend_impl = BackendImpl::Sqlite(sqlite);
    backend_impl.migrate().await.unwrap();
    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    Harness {
        _dir: dir,
        registry,
        topic_repo,
        backend_arc,
        broker,
        publisher,
        subscriber,
    }
}

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]))
}

fn topic_def(name: &str) -> TopicDefinition {
    TopicDefinition {
        id: TopicId::new(),
        name: name.to_string(),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    }
}

fn batch_of(seq: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(seq.to_vec()))],
    )
    .unwrap()
}

fn seq_column(batch: &RecordBatch) -> Vec<i64> {
    let col = batch
        .column_by_name("seq")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    (0..col.len()).map(|i| col.value(i)).collect()
}

async fn next_with_timeout<S>(stream: &mut S) -> Option<S::Item>
where
    S: futures::Stream + Unpin,
{
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("stream item timed out")
}

/// Live tail after subscribe delivers every published offset exactly
/// once, in ascending order.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn live_tail_delivers_every_offset_exactly_once_in_order(arm: Arm) {
    let broker = broker_or_skip!(arm, "live_tail_delivers_every_offset_exactly_once_in_order");
    let h = build_harness(broker).await;
    let topic = topic_def("parity.live_tail_order");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    const N: i64 = 40;
    for i in 0..N {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < N as usize {
        let delivered = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    let expected: Vec<u64> = (0..N as u64).collect();
    assert_eq!(
        seen, expected,
        "every published offset must arrive exactly once, in order"
    );
}

/// Two independent `Publisher`s over ONE broker/topic in one process
/// publish interleaved; post-commit fan-out
/// across two writers is unordered, so it is the tail's contiguity check
/// (never delivery order) that must still deliver a gap-free, in-order,
/// exactly-once stream.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn two_publishers_over_one_broker_deliver_gap_free_in_order(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "two_publishers_over_one_broker_deliver_gap_free_in_order"
    );
    let h = build_harness(Arc::clone(&broker)).await;
    let topic = topic_def("parity.two_publishers");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    let publisher_b = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&h.backend_arc),
        Arc::clone(&h.registry),
    );

    const N: i64 = 30;
    let (a_res, b_res) = tokio::join!(
        async {
            for i in 0..N {
                h.publisher
                    .publish_scoped(&topic, None, batch_of(&[i]))
                    .await
                    .unwrap();
            }
        },
        async {
            for i in 0..N {
                publisher_b
                    .publish_scoped(&topic, None, batch_of(&[1000 + i]))
                    .await
                    .unwrap();
            }
        },
    );
    let _ = (a_res, b_res);

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < (2 * N) as usize {
        let delivered = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    let expected: Vec<u64> = (0..(2 * N) as u64).collect();
    assert_eq!(
        sorted, expected,
        "two publishers over one broker must produce a gap-free, collision-free offset set"
    );
    assert_eq!(
        seen, sorted,
        "the tail must deliver offsets to the subscriber in strictly ascending order even \
         though the two publishers' post-commit fan-out races each other"
    );
}

/// `from_offset` in the past: the replay prefix and the live tail
/// overlap, and the seam must dedup that overlap so no offset is delivered
/// twice.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn from_offset_in_past_replay_and_live_overlap_no_duplicates(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "from_offset_in_past_replay_and_live_overlap_no_duplicates"
    );
    let h = build_harness(broker).await;
    let topic = topic_def("parity.replay_overlap");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..10i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    // Subscribe from offset 0 -- replay covers 0..=9, and the live tail
    // overlaps starting at or before 0 too (per-driver contract).
    let mut sub = h
        .subscriber
        .subscribe(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    for i in 10..20i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < 20 {
        let delivered = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    let expected: Vec<u64> = (0..20u64).collect();
    assert_eq!(
        seen, expected,
        "replay prefix + live overlap must yield every offset exactly once, no duplicates, no gaps"
    );
}

/// Adversarial regression: a publish committing WHILE `subscribe` is
/// between its replay/watermark reads and its tail attach must still reach
/// this subscriber live, never be silently dropped. Distinct from
/// [`from_offset_in_past_replay_and_live_overlap_no_duplicates`] above,
/// whose writer publishes strictly BEFORE `subscribe` returns: here the
/// writer races the `subscribe` call itself, concurrently, for its entire
/// duration. A wide pre-existing history (`HISTORY`) widens the window
/// `drain_replay`'s materialise/decode/group work takes, giving the
/// concurrent writer a real chance to land a commit inside it.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn publish_racing_subscribe_is_never_lost(arm: Arm) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let broker = broker_or_skip!(arm, "publish_racing_subscribe_is_never_lost");
    let h = Arc::new(build_harness(broker).await);
    let topic = topic_def("parity.subscribe_race");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    const HISTORY: i64 = 1500;
    for i in 0..HISTORY {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let stop_w = Arc::clone(&stop);
    let h_w = Arc::clone(&h);
    let topic_w = topic.clone();
    let writer = tokio::spawn(async move {
        let mut i = 10_000i64;
        while !stop_w.load(Ordering::Relaxed) {
            h_w.publisher
                .publish_scoped(&topic_w, None, batch_of(&[i]))
                .await
                .unwrap();
            i += 1;
            tokio::task::yield_now().await;
        }
    });

    // Races the concurrent writer above: `subscribe` itself, not merely a
    // publish before or after it.
    let mut sub = h
        .subscriber
        .subscribe(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    const WANT: usize = HISTORY as usize + 40;
    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < WANT {
        let delivered = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    stop.store(true, Ordering::Relaxed);
    let _ = writer.await;

    let expected: Vec<u64> = (0..WANT as u64).collect();
    let gaps: Vec<u64> = expected
        .iter()
        .copied()
        .filter(|o| !seen.contains(o))
        .collect();
    assert!(
        gaps.is_empty(),
        "subscribe(from_offset = 0) must skip no engine offset; missing {gaps:?}; \
         first delivered = {:?}, delivered count = {}",
        seen.first(),
        seen.len()
    );
}

/// A tenant-scoped subscriber never sees another tenant's rows, even
/// though the driver is tenant-blind and the topic is globally registered
/// (`TopicDefinition::tenant == None`), so one `topic.id` is shared across
/// tenants.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn tenant_scoped_subscriber_never_sees_another_tenants_rows(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "tenant_scoped_subscriber_never_sees_another_tenants_rows"
    );
    let h = build_harness(broker).await;
    let topic = topic_def("parity.tenant_scope");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c84-aaaa-7e10-9c4f-bbbbcccc8e9a").unwrap();

    let mut sub_a = h
        .subscriber
        .subscribe_scoped(&topic, Some(tenant_a), Predicate::match_all(), None)
        .await
        .unwrap();

    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[1]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, Some(tenant_b), batch_of(&[2]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[3]))
        .await
        .unwrap();

    // Drain exactly two deliveries (tenant A's two rows); anything else
    // arriving on this stream within the timeout window would be a leak.
    let d1 = next_with_timeout(&mut sub_a).await.unwrap().unwrap();
    let d2 = next_with_timeout(&mut sub_a).await.unwrap().unwrap();
    let seen: Vec<i64> = seq_column(&d1.batch)
        .into_iter()
        .chain(seq_column(&d2.batch))
        .collect();
    assert_eq!(
        seen,
        vec![1, 3],
        "tenant A's live tail must see only tenant A's (and global) rows, in order"
    );
    let third = tokio::time::timeout(Duration::from_millis(300), sub_a.next()).await;
    assert!(
        third.is_err(),
        "no third delivery should arrive on tenant A's stream -- tenant B's row must not leak"
    );
}

/// A subscriber whose OWN broadcast receiver lags (never the
/// tail's shared cursor) self-heals via its own chunked, group-completing
/// replay -- no error, no loss -- and that replay correctly reassembles a
/// publish wider than one replay chunk (via the group-completion probe) back
/// into ONE `DeliveredBatch` in original row order.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn subscriber_lag_self_heals_via_chunked_group_completing_replay(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "subscriber_lag_self_heals_via_chunked_group_completing_replay"
    );
    let h = build_harness(broker).await;
    let topic = topic_def("parity.lag_self_heal");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    // Publish far more single-row batches than the tail's broadcast
    // capacity (256) WITHOUT draining `sub` -- its own receiver overflows
    // and the next `recv()` observes `RecvError::Lagged`, forcing this
    // subscriber's own chunked lag-replay (never an error, never the whole-
    // suffix `drain_replay` path). One publish in the middle is a single
    // WIDE batch (1200 rows, one `_offset`) -- comfortably wider than the
    // replay chunk size, so the lag-replay's group-completion probe must
    // fire to keep it as one `DeliveredBatch`.
    const BEFORE: i64 = 150;
    const WIDE: usize = 1200;
    const AFTER: i64 = 150;

    for i in 0..BEFORE {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }
    let wide_seq: Vec<i64> = (0..WIDE as i64).map(|i| 1_000_000 + i).collect();
    let wide_batch = RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(wide_seq.clone()))],
    )
    .unwrap();
    h.publisher
        .publish_scoped(&topic, None, wide_batch)
        .await
        .unwrap();
    for i in 0..AFTER {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[2_000_000 + i]))
            .await
            .unwrap();
    }

    let total_publishes = (BEFORE + 1 + AFTER) as usize;
    let mut deliveries = Vec::with_capacity(total_publishes);
    for _ in 0..total_publishes {
        let d = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        deliveries.push(d);
    }

    // Offsets must be gap-free, exactly-once, in order across the whole run
    // (the lag-replay reassembles by `_offset` groups, same contract as the
    // live path).
    let offsets: Vec<u64> = deliveries.iter().map(|d| d.offset.value()).collect();
    let expected_offsets: Vec<u64> = (0..total_publishes as u64).collect();
    assert_eq!(
        offsets, expected_offsets,
        "lag recovery must deliver every offset exactly once, in order, with no gaps"
    );

    // The wide publish (index BEFORE) must reassemble as ONE DeliveredBatch
    // containing all 1200 rows, in original order.
    let wide_delivery = &deliveries[BEFORE as usize];
    assert_eq!(
        seq_column(&wide_delivery.batch),
        wide_seq,
        "a publish wider than one replay chunk must reassemble as one whole DeliveredBatch, \
         in original row order, even when discovered through a subscriber's own lag replay"
    );
}

/// Two subscribers with DIFFERENT predicates, on one `Subscriber`,
/// on the SAME `(topic, tenant)` share exactly one driver-level
/// subscription -- proven via `list_consumers`, which enumerates driver
/// consumers directly, never the number of `Subscriber`-level subscriptions
/// attached to a tail. Each subscriber must still see exactly its own
/// predicate's matches.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn two_predicate_subscribers_share_one_driver_subscription(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "two_predicate_subscribers_share_one_driver_subscription"
    );
    let h = build_harness(broker).await;
    let topic = topic_def("parity.shared_tail");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let session = datafusion::execution::context::SessionContext::new();
    let even = Predicate::from_sql(&session, Arc::clone(&topic.schema), "seq % 2 = 0").unwrap();
    let odd = Predicate::from_sql(&session, Arc::clone(&topic.schema), "seq % 2 = 1").unwrap();

    let mut sub_even = h.subscriber.subscribe(&topic, even, None).await.unwrap();
    let mut sub_odd = h.subscriber.subscribe(&topic, odd, None).await.unwrap();

    for i in 0..10i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    let mut evens: Vec<i64> = Vec::new();
    while evens.len() < 5 {
        let d = next_with_timeout(&mut sub_even).await.unwrap().unwrap();
        evens.extend(seq_column(&d.batch));
    }
    let mut odds: Vec<i64> = Vec::new();
    while odds.len() < 5 {
        let d = next_with_timeout(&mut sub_odd).await.unwrap().unwrap();
        odds.extend(seq_column(&d.batch));
    }
    assert_eq!(
        evens,
        vec![0, 2, 4, 6, 8],
        "the even-predicate subscriber sees only even seqs"
    );
    assert_eq!(
        odds,
        vec![1, 3, 5, 7, 9],
        "the odd-predicate subscriber sees only odd seqs"
    );

    // Two `Subscriber`-level subscriptions on one `(topic, tenant)` cost
    // exactly ONE driver-level consumer -- `list_consumers` enumerates the
    // driver's own subscriptions, and there is only ever one: the tail's.
    let consumers = h.broker.list_consumers(topic.id).await.unwrap();
    assert_eq!(
        consumers.len(),
        1,
        "two predicate-scoped subscribers sharing one tail must cost exactly one driver-level \
         consumer, not two -- got {consumers:?}"
    );
}

/// Every accepted topic column type round-trips through a LIVE
/// subscribe (not just replay), registered through `TopicRepo::register_topic`
/// so the oracle pins the persisted type-name table.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn every_accepted_type_round_trips_through_live_subscribe(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "every_accepted_type_round_trips_through_live_subscribe"
    );
    let h = build_harness(broker).await;

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("i32_col", DataType::Int32, false),
        Field::new("u16_col", DataType::UInt16, false),
        Field::new("u64_col", DataType::UInt64, false),
        Field::new("f32_col", DataType::Float32, false),
        Field::new("bytes_col", DataType::Binary, false),
    ]));
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: "parity.every_type".to_string(),
        schema: Arc::clone(&schema),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();
    let loaded = h
        .topic_repo
        .lookup_by_name(&topic.name, None)
        .await
        .unwrap()
        .expect("topic persisted");

    let mut sub = h
        .subscriber
        .subscribe(&loaded, Predicate::match_all(), None)
        .await
        .unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&loaded.schema),
        vec![
            Arc::new(Int32Array::from(vec![-7i32, 42])),
            Arc::new(UInt16Array::from(vec![1u16, 65535])),
            Arc::new(UInt64Array::from(vec![0u64, i64::MAX as u64])),
            Arc::new(Float32Array::from(vec![1.5f32, -2.25])),
            Arc::new(BinaryArray::from(vec![
                b"hello".as_ref(),
                b"\x00\x01\xff".as_ref(),
            ])),
        ],
    )
    .unwrap();
    h.publisher
        .publish_scoped(&loaded, None, batch)
        .await
        .expect("publish accepts every declared column type");

    let delivered = next_with_timeout(&mut sub).await.unwrap().unwrap();
    assert_eq!(
        delivered.batch.schema().as_ref(),
        loaded.schema.as_ref(),
        "the live-delivered batch schema must equal the declared topic schema exactly"
    );
    let i32_col = delivered
        .batch
        .column_by_name("i32_col")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("i32_col must decode as Int32Array");
    assert_eq!(i32_col.values(), &[-7, 42]);
    let bytes_col = delivered
        .batch
        .column_by_name("bytes_col")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("bytes_col must decode as BinaryArray");
    assert_eq!(bytes_col.value(0), b"hello");
    assert_eq!(bytes_col.value(1), b"\x00\x01\xff");
}

/// Offset order == commit order. Two independent `Publisher`
/// "sessions" share ONE Postgres database; interleaved publishes still
/// yield a replay whose `_offset` order equals the true commit order,
/// because the offset-assigning `UPDATE`'s row lock holds until commit.
#[tokio::test]
async fn offset_order_equals_commit_order_two_sessions_one_postgres() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!("skipping offset_order_equals_commit_order_two_sessions_one_postgres: JAMMI_TEST_PG_URL unset");
        require_live_pg("offset_order_equals_commit_order_two_sessions_one_postgres");
        return;
    };
    let pg = PostgresBackend::open_with_options(&url, 8, None)
        .await
        .unwrap();
    let backend_impl = BackendImpl::Postgres(pg);
    backend_impl.migrate().await.unwrap();
    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    let topic = topic_def(&format!(
        "parity.pg_commit_order.{}",
        jammi_test_utils::unique_suffix()
    ));
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    // Two "sessions": independent `Publisher`s, each with its own in-process
    // write lock, sharing the same Postgres-backed catalog/backing table --
    // the shape two OS processes on one Postgres database would take.
    let session_a = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let session_b = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );

    const N: i64 = 100;
    // A commit-order log: each successful publish appends `(offset, marker)`
    // the instant it commits, under a shared async lock, so the log's
    // insertion order IS the true commit order regardless of which session
    // committed it.
    let commit_log: Arc<tokio::sync::Mutex<Vec<(u64, i64)>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(2 * N as usize)));

    async fn publish_and_log(
        publisher: &Publisher,
        topic: &TopicDefinition,
        marker: i64,
        n: i64,
        log: &Arc<tokio::sync::Mutex<Vec<(u64, i64)>>>,
    ) {
        for i in 0..n {
            let off = publisher
                .publish_scoped(topic, None, batch_of(&[marker * 1_000_000 + i]))
                .await
                .unwrap();
            log.lock().await.push((off.value(), marker * 1_000_000 + i));
        }
    }

    tokio::join!(
        publish_and_log(&session_a, &topic, 1, N, &commit_log),
        publish_and_log(&session_b, &topic, 2, N, &commit_log),
    );

    // `commit_order` is the log in TRUE commit order (append order, under
    // the shared lock in `publish_and_log`) — kept UNSORTED and untouched
    // from here on, so the replay-order assertion below actually compares
    // against real commit order rather than against itself re-derived from
    // a sorted copy. `sorted_by_offset` is a SEPARATE clone (the pattern
    // `two_publishers_over_one_broker_deliver_gap_free_in_order` above uses
    // for `seen`/`sorted`): sorting it in place would otherwise silently
    // launder a reversed or reordered log into "already sorted", since
    // `expected_seq` below would then be re-derived from the sorted copy
    // instead of the true commit order.
    let commit_order = commit_log.lock().await.clone();
    let mut sorted_by_offset = commit_order.clone();
    sorted_by_offset.sort_by_key(|(off, _)| *off);
    let offsets: Vec<u64> = sorted_by_offset.iter().map(|(off, _)| *off).collect();
    let expected: Vec<u64> = (0..(2 * N) as u64).collect();
    assert_eq!(
        offsets, expected,
        "offsets assigned across two sessions on one Postgres must be gap-free and unique"
    );

    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));
    let drained = subscriber
        .replay_only(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();
    let replay_seq: Vec<i64> = drained.iter().flat_map(|d| seq_column(&d.batch)).collect();
    // Derived from the UNSORTED `commit_order` — the true, as-recorded
    // commit sequence — not `sorted_by_offset`: a replay that merely
    // reproduced offset order (even if that order diverged from real
    // commit order) would pass against a sorted expectation but must fail
    // here.
    let expected_seq: Vec<i64> = commit_order.iter().map(|(_, marker)| *marker).collect();
    assert_eq!(
        replay_seq, expected_seq,
        "replay order (by `_offset`) must equal the true commit order recorded by the shared \
         commit log -- this is the invariant the offset-assigning UPDATE's row lock provides"
    );
}

/// Every accepted topic column type round-trips through publish/replay with
/// the BACKING TABLE ITSELF on Postgres (`PostgresBackend` +
/// `PostgresMutableBackend`) -- unlike every other Postgres-arm test in this
/// suite, whose backing table stays on SQLite (module docs) and whose broker
/// is the only real-Postgres component. A Postgres-specific decode bug in
/// `store/mutable/postgres.rs` (column encode/decode) or
/// `catalog/backend_postgres.rs` (DDL/DML rendering) would pass every other
/// test in this file; this oracle was missing.
#[tokio::test]
async fn every_accepted_type_round_trips_through_postgres_backing_table() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!(
            "skipping every_accepted_type_round_trips_through_postgres_backing_table: \
             JAMMI_TEST_PG_URL unset"
        );
        require_live_pg("every_accepted_type_round_trips_through_postgres_backing_table");
        return;
    };
    let pg = PostgresBackend::open_with_options(&url, 4, None)
        .await
        .unwrap();
    let backend_impl = BackendImpl::Postgres(pg);
    backend_impl.migrate().await.unwrap();
    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));

    // Every type `register_topic` accepts (`topic_repo.rs`'s type-name
    // table): Boolean, every signed/unsigned integer width, both float
    // widths, Utf8, Binary.
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("bool_col", DataType::Boolean, false),
        Field::new("i8_col", DataType::Int8, false),
        Field::new("i16_col", DataType::Int16, false),
        Field::new("i32_col", DataType::Int32, false),
        Field::new("i64_col", DataType::Int64, false),
        Field::new("u8_col", DataType::UInt8, false),
        Field::new("u16_col", DataType::UInt16, false),
        Field::new("u32_col", DataType::UInt32, false),
        Field::new("u64_col", DataType::UInt64, false),
        Field::new("f32_col", DataType::Float32, false),
        Field::new("f64_col", DataType::Float64, false),
        Field::new("utf8_col", DataType::Utf8, false),
        Field::new("bytes_col", DataType::Binary, false),
    ]));
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: format!(
            "parity.pg_backing_every_type.{}",
            jammi_test_utils::unique_suffix()
        ),
        schema: Arc::clone(&schema),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();
    let loaded = topic_repo
        .lookup_by_name(&topic.name, None)
        .await
        .unwrap()
        .expect("topic persisted");

    let batch = RecordBatch::try_new(
        Arc::clone(&loaded.schema),
        vec![
            Arc::new(BooleanArray::from(vec![true, false])),
            Arc::new(Int8Array::from(vec![-7i8, 42])),
            Arc::new(Int16Array::from(vec![-700i16, 4200])),
            Arc::new(Int32Array::from(vec![-70000i32, 420000])),
            Arc::new(Int64Array::from(vec![-7_000_000_000i64, 42])),
            Arc::new(UInt8Array::from(vec![1u8, 255])),
            Arc::new(UInt16Array::from(vec![1u16, 65535])),
            Arc::new(UInt32Array::from(vec![1u32, u32::MAX])),
            Arc::new(UInt64Array::from(vec![0u64, i64::MAX as u64])),
            Arc::new(Float32Array::from(vec![1.5f32, -2.25])),
            Arc::new(Float64Array::from(vec![1.5f64, -2.25])),
            Arc::new(StringArray::from(vec!["hello", ""])),
            Arc::new(BinaryArray::from(vec![
                b"hello".as_ref(),
                b"\x00\x01\xff".as_ref(),
            ])),
        ],
    )
    .unwrap();

    publisher
        .publish_scoped(&loaded, None, batch)
        .await
        .expect("publish accepts every declared column type over the Postgres backing table");

    let from = Offset::new(0, chrono::Utc::now());
    let drained = subscriber
        .replay_only(&loaded, Predicate::match_all(), Some(from))
        .await
        .expect("replay must reconstruct the batch against the declared schema over Postgres");
    assert_eq!(drained.len(), 1);
    let replayed = &drained[0].batch;
    assert_eq!(
        replayed.schema().as_ref(),
        loaded.schema.as_ref(),
        "replayed batch schema must equal the declared topic schema exactly over Postgres"
    );

    macro_rules! assert_col {
        ($name:expr, $arr_ty:ty, $expected:expr) => {
            let col = replayed
                .column_by_name($name)
                .unwrap()
                .as_any()
                .downcast_ref::<$arr_ty>()
                .unwrap_or_else(|| panic!("{} must decode as {}", $name, stringify!($arr_ty)));
            assert_eq!(col.values(), $expected, "{} round-trip mismatch", $name);
        };
    }
    let bool_col = replayed
        .column_by_name("bool_col")
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("bool_col must decode as BooleanArray");
    assert!(bool_col.value(0), "bool_col[0] round-trip mismatch");
    assert!(!bool_col.value(1), "bool_col[1] round-trip mismatch");
    assert_col!("i8_col", Int8Array, &[-7i8, 42]);
    assert_col!("i16_col", Int16Array, &[-700i16, 4200]);
    assert_col!("i32_col", Int32Array, &[-70000i32, 420000]);
    assert_col!("i64_col", Int64Array, &[-7_000_000_000i64, 42]);
    assert_col!("u8_col", UInt8Array, &[1u8, 255]);
    assert_col!("u16_col", UInt16Array, &[1u16, 65535]);
    assert_col!("u32_col", UInt32Array, &[1u32, u32::MAX]);
    assert_col!("u64_col", UInt64Array, &[0u64, i64::MAX as u64]);
    assert_col!("f32_col", Float32Array, &[1.5f32, -2.25]);
    assert_col!("f64_col", Float64Array, &[1.5f64, -2.25]);

    let utf8_col = replayed
        .column_by_name("utf8_col")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("utf8_col must decode as StringArray");
    assert_eq!(utf8_col.value(0), "hello");
    assert_eq!(utf8_col.value(1), "");

    let bytes_col = replayed
        .column_by_name("bytes_col")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("bytes_col must decode as BinaryArray");
    assert_eq!(bytes_col.value(0), b"hello");
    assert_eq!(bytes_col.value(1), b"\x00\x01\xff");
}

/// `application_name` `PostgresBroker`'s dedicated listener connection tags
/// itself with — must match `crate::trigger::postgres::LISTENER_APPLICATION_NAME`
/// (a private constant; duplicated here rather than exposed only for a test).
const LISTENER_APPLICATION_NAME: &str = "jammi-trigger-listener";

/// Terminate every Postgres backend tagged with [`LISTENER_APPLICATION_NAME`]
/// (never this connection's own backend) — simulates a `PostgresBroker`'s
/// dedicated listener connection dying out from under it, so the next
/// `try_recv()` observes a connection-reset error and self-heals via sqlx's
/// own eager-reconnect contract.
async fn kill_broker_listener_backends(url: &str) {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await
        .expect("connect to kill the listener backend");
    sqlx::query(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
         WHERE application_name = $1 AND pid <> pg_backend_pid()",
    )
    .bind(LISTENER_APPLICATION_NAME)
    .fetch_all(&pool)
    .await
    .expect("pg_terminate_backend query");
}

/// Killing the broker's dedicated listener backend loses whatever
/// NOTIFYs arrive during the reconnect window, but every offset still arrives
/// -- via `Ok(None)`'s wake-every-topic fallback (and, as a second net, the
/// idle tick) driving the engine's own replay.
#[tokio::test]
async fn postgres_listener_killed_recovers_via_replay() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!("skipping postgres_listener_killed_recovers_via_replay: JAMMI_TEST_PG_URL unset");
        require_live_pg("postgres_listener_killed_recovers_via_replay");
        return;
    };
    let broker: Arc<dyn TriggerBroker> = Arc::new(
        PostgresBroker::connect(&url, TEST_IDLE_POLL)
            .await
            .expect("connect to Postgres broker"),
    );
    let h = build_harness(Arc::clone(&broker)).await;
    let topic = topic_def(&format!(
        "parity.pg_listener_killed.{}",
        jammi_test_utils::unique_suffix()
    ));
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    h.publisher
        .publish_scoped(&topic, None, batch_of(&[1]))
        .await
        .unwrap();
    let d1 = next_with_timeout(&mut sub).await.unwrap().unwrap();
    assert_eq!(seq_column(&d1.batch), vec![1]);

    kill_broker_listener_backends(&url).await;

    // Publish more rows while the listener may still be mid-reconnect; any
    // NOTIFY lost in that window is covered by `Ok(None)`'s wake-every-topic
    // fallback once reconnected, and by the idle tick as a second net.
    for i in 2..=5i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    let mut seen: Vec<i64> = vec![1];
    while seen.len() < 5 {
        let d = next_with_timeout(&mut sub).await.unwrap().unwrap();
        seen.extend(seq_column(&d.batch));
    }
    assert_eq!(
        seen,
        vec![1, 2, 3, 4, 5],
        "every offset must arrive via replay after the listener's own backend is killed"
    );
}

/// `ConsumerOffsetSnapshot` reports `None` for a consumer that has never
/// observed a NOTIFY payload carrying an offset -- never a fabricated `0`,
/// which would be indistinguishable from a real `_offset = 0` (the first
/// published row). Observable immediately after subscribing: the driver's
/// own subscribe stream always yields an unconditional first `Wake` that
/// carries no offset (`TriggerBroker::subscribe`'s own doc contract), so a
/// consumer can be enumerated before any publish has happened at all. Uses
/// the CONCRETE `PostgresBroker` type directly (its own `TriggerBroker`
/// methods, not `Publisher`/`Subscriber`) because only this driver's
/// `ConsumerOffsetSnapshot` is best-effort/informational in this way — the
/// in-memory and JetStream drivers carry the delivered batch itself, so
/// their consumer's last-delivered offset is authoritative from the first
/// event.
#[tokio::test]
async fn postgres_list_consumers_reports_none_until_a_notify_is_seen() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!(
            "skipping postgres_list_consumers_reports_none_until_a_notify_is_seen: \
             JAMMI_TEST_PG_URL unset"
        );
        require_live_pg("postgres_list_consumers_reports_none_until_a_notify_is_seen");
        return;
    };
    let broker = PostgresBroker::connect(&url, TEST_IDLE_POLL)
        .await
        .expect("connect to Postgres broker");
    let topic = topic_def(&format!(
        "parity.pg_never_woken.{}",
        jammi_test_utils::unique_suffix()
    ));
    broker.register_topic(&topic).await.unwrap();

    let mut driver_stream = broker
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();
    // Drain the unconditional first `Wake` -- it carries no offset.
    let first = next_with_timeout(&mut driver_stream)
        .await
        .expect("stream ended early")
        .unwrap();
    assert!(
        matches!(first, LiveEvent::Wake),
        "the driver-level subscribe always yields Wake first"
    );

    let before = broker.list_consumers(topic.id).await.unwrap();
    assert_eq!(before.len(), 1, "exactly one consumer is attached");
    assert_eq!(
        before[0].last_delivered_offset, None,
        "a consumer that has never seen a NOTIFY payload carrying an offset must report None, \
         never a fabricated 0"
    );
    assert_eq!(before[0].last_acked_offset, None);

    // Publish -- carries a real engine offset through the NOTIFY payload --
    // and wait for the resulting Wake before re-checking.
    broker
        .publish(topic.id, batch_of(&[1]), chrono::Utc::now(), 7, None)
        .await
        .unwrap();
    let _woken = next_with_timeout(&mut driver_stream)
        .await
        .expect("stream ended early")
        .unwrap();

    let after = broker.list_consumers(topic.id).await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(
        after[0].last_delivered_offset,
        Some(7),
        "once a NOTIFY payload carrying an offset has been seen, it must be reported"
    );
    assert_eq!(after[0].last_acked_offset, Some(7));
}

/// A NOTIFY the broker itself never sends (simulating one Postgres
/// silently drops) is still covered within `idle_poll` by the idle tick — no
/// error, no permanent loss.
#[tokio::test]
async fn postgres_suppressed_notify_recovers_via_idle_tick() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!(
            "skipping postgres_suppressed_notify_recovers_via_idle_tick: JAMMI_TEST_PG_URL unset"
        );
        require_live_pg("postgres_suppressed_notify_recovers_via_idle_tick");
        return;
    };
    let concrete = Arc::new(
        PostgresBroker::connect(&url, Duration::from_millis(300))
            .await
            .expect("connect to Postgres broker"),
    );
    let broker: Arc<dyn TriggerBroker> = Arc::clone(&concrete) as Arc<dyn TriggerBroker>;
    let h = build_harness(broker).await;
    let topic = topic_def(&format!(
        "parity.pg_suppressed_notify.{}",
        jammi_test_utils::unique_suffix()
    ));
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    concrete.suppress_next_notify_for_testing();
    h.publisher
        .publish_scoped(&topic, None, batch_of(&[7]))
        .await
        .unwrap();

    // No NOTIFY was ever sent for this publish; only the idle tick (300ms)
    // can surface it. `next_with_timeout`'s 5s budget comfortably covers it.
    let d = next_with_timeout(&mut sub).await.unwrap().unwrap();
    assert_eq!(
        seq_column(&d.batch),
        vec![7],
        "a suppressed NOTIFY must still be delivered within idle_poll via the idle tick"
    );
}

/// `from_offset` is a LOWER BOUND, never a mere replay-window cursor: a
/// `subscribe(from_offset = Some(N))` whose replay window is EMPTY at
/// subscribe time (this client has already consumed every row below `N`)
/// must still never deliver an engine offset below `N`, even after this
/// subscriber's own broadcast receiver lags and self-heals via its own
/// chunked replay. Regression for #490: `last_yielded` used to seed as
/// `None` whenever the replay window was empty (`from_offset` was only
/// consulted to compute the replay window itself, never carried forward as
/// a floor), so a lag taken before this subscriber ever yielded anything
/// reseeded its own lag-replay cursor from `-1` — the ENTIRE backing table
/// — and the live-recv admission check (`last_yielded.is_none_or(...)`)
/// admitted any offset at all once that reseed replayed past `N`.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn from_offset_lower_bound_holds_after_lag_replay(arm: Arm) {
    let broker = broker_or_skip!(arm, "from_offset_lower_bound_holds_after_lag_replay");
    let h = build_harness(broker).await;
    let topic = topic_def("parity.from_offset_lower_bound_lag");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    // History this client has ALREADY consumed: offsets 0..=9.
    for i in 0..10i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    // Resume strictly after it: "deliver me offset >= 10". The replay
    // window for this call is empty (nothing at or above offset 10 exists
    // yet), so `last_yielded` has nothing to seed from except the floor
    // this fix introduces.
    let mut sub = h
        .subscriber
        .subscribe(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(10, chrono::Utc::now())),
        )
        .await
        .unwrap();

    // Fan out more than the tail's broadcast capacity (256) WITHOUT polling
    // `sub`, so its own receiver overflows and its next `recv()` observes
    // `RecvError::Lagged`, forcing this subscriber's own lag-replay path —
    // the SAME path that used to reseed from `-1` when `last_yielded` was
    // `None`.
    for i in 100..500i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut first: Vec<u64> = Vec::new();
    for _ in 0..5 {
        let d = next_with_timeout(&mut sub)
            .await
            .expect("stream ended early")
            .unwrap();
        first.push(d.offset.value());
    }
    assert!(
        first.iter().all(|o| *o >= 10),
        "subscribe(from_offset = 10) must never deliver an offset below 10 even after this \
         subscriber's own lag replay; first delivered offsets = {first:?}"
    );
}

/// Second member of the same class, with NO lag involved: when the
/// subscribe-time replay window is empty, the FIRST live event delivered
/// must still respect `from_offset` as a lower bound — every live publish
/// below it (e.g. a resume point read from a checkpoint written by a replica
/// ahead of this one) is silently swallowed, never delivered, and the first
/// delivery is exactly `from_offset` itself, never anything above OR below
/// it.
#[test_case(Arm::InMemory ; "in_memory")]
#[test_case(Arm::Postgres ; "postgres")]
#[cfg_attr(feature = "jetstream-broker", test_case(Arm::JetStream ; "jetstream"))]
#[tokio::test]
async fn from_offset_lower_bound_holds_on_first_live_event_with_empty_replay_window(arm: Arm) {
    let broker = broker_or_skip!(
        arm,
        "from_offset_lower_bound_holds_on_first_live_event_with_empty_replay_window"
    );
    let h = build_harness(broker).await;
    let topic = topic_def("parity.from_offset_lower_bound_live");
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..10i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }

    // A resume point the client has not reached yet: the next five live
    // publishes will land engine offsets 10..=14, all strictly below it.
    let mut sub = h
        .subscriber
        .subscribe(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(15, chrono::Utc::now())),
        )
        .await
        .unwrap();

    for i in 0..5i64 {
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[900 + i]))
            .await
            .unwrap();
        let premature = tokio::time::timeout(Duration::from_millis(200), sub.next()).await;
        assert!(
            premature.is_err(),
            "engine offset {} is below from_offset = 15 (an empty-replay-window subscribe) and \
             must never be delivered, even live; got {premature:?}",
            10 + i
        );
    }

    // This publish lands engine offset 15 -- exactly `from_offset`, the
    // first one `subscribe` actually asked for.
    h.publisher
        .publish_scoped(&topic, None, batch_of(&[777]))
        .await
        .unwrap();
    let d = next_with_timeout(&mut sub)
        .await
        .expect("stream ended early")
        .unwrap();
    assert_eq!(
        d.offset.value(),
        15,
        "the first delivered offset must be exactly `from_offset`, never anything below it"
    );
}

/// Cross-surface parity for the one-step `tail_replay` over a REAL Postgres
/// BACKING table. Every other multi-step / boundary-group oracle in this
/// suite runs over a SQLite backing table (`build_harness` is SQLite-only;
/// the `Arm::Postgres` arm swaps only the broker), so the Postgres-dialect
/// half of the step query (`ORDER BY … LIMIT chunk_size + 1`, the exact
/// boundary-group re-fetch) had no oracle of its own. Shape: 150 single-row
/// publishes, ONE 1200-row publish (a single `_offset` group far wider than
/// `REPLAY_CHUNK_SIZE = 500`, so it straddles a step boundary and must be
/// fetched whole rather than split), then 150 more — and two drains:
/// `replay_only(Some(0))` must yield all 301 offsets exactly once, in order,
/// with the wide group intact and in original row order; and
/// `replay_only(Some(150))` — a mid-window inclusive lower bound, exercising
/// `cursor_before > 0` — must yield exactly the 151 offsets from the wide
/// group onward.
///
/// Skips (or, under `JAMMI_REQUIRE_PG`, panics) without `JAMMI_TEST_PG_URL`,
/// the `require_live_pg` shape.
#[tokio::test]
async fn postgres_backing_multistep_replay_keeps_boundary_group_whole() {
    let Some(url) = jammi_test_utils::pg_url_for_tests() else {
        eprintln!(
            "skipping postgres_backing_multistep_replay_keeps_boundary_group_whole: \
             JAMMI_TEST_PG_URL unset"
        );
        require_live_pg("postgres_backing_multistep_replay_keeps_boundary_group_whole");
        return;
    };
    let pg = PostgresBackend::open_with_options(&url, 8, None)
        .await
        .unwrap();
    let backend_impl = BackendImpl::Postgres(pg);
    backend_impl.migrate().await.unwrap();
    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));
    let topic = topic_def(&format!(
        "parity.pg_backing_multistep.{}",
        jammi_test_utils::unique_suffix()
    ));
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    const BEFORE: i64 = 150;
    const WIDE: usize = 1200;
    const AFTER: i64 = 150;
    for i in 0..BEFORE {
        publisher
            .publish_scoped(&topic, None, batch_of(&[i]))
            .await
            .unwrap();
    }
    let wide_seq: Vec<i64> = (0..WIDE as i64).map(|i| 1_000_000 + i).collect();
    let wide_batch = RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(wide_seq.clone()))],
    )
    .unwrap();
    publisher
        .publish_scoped(&topic, None, wide_batch)
        .await
        .unwrap();
    for i in 0..AFTER {
        publisher
            .publish_scoped(&topic, None, batch_of(&[2_000_000 + i]))
            .await
            .unwrap();
    }

    let total = (BEFORE + 1 + AFTER) as u64;
    let out = subscriber
        .replay_only(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();
    let offsets: Vec<u64> = out.iter().map(|d| d.offset.value()).collect();
    assert_eq!(
        offsets,
        (0..total).collect::<Vec<_>>(),
        "a multi-step drain over a Postgres backing table must deliver every offset exactly \
         once, in order"
    );
    assert_eq!(
        seq_column(&out[BEFORE as usize].batch),
        wide_seq,
        "a 1200-row group must survive the 500-row step boundary whole, in original row order, \
         over a Postgres backing table"
    );
    for (i, d) in out.iter().enumerate().take(BEFORE as usize) {
        assert_eq!(seq_column(&d.batch), vec![i as i64], "pre-group offset {i}");
    }
    for (j, d) in out.iter().skip(BEFORE as usize + 1).enumerate() {
        assert_eq!(
            seq_column(&d.batch),
            vec![2_000_000 + j as i64],
            "post-group offset {}",
            BEFORE as usize + 1 + j
        );
    }

    // The same over a mid-window inclusive lower bound (`cursor_before > 0`):
    // the wide group is the FIRST event of this window, so the boundary
    // re-fetch is exercised from a non-zero cursor too.
    let out2 = subscriber
        .replay_only(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(BEFORE as u64, chrono::Utc::now())),
        )
        .await
        .unwrap();
    let offsets2: Vec<u64> = out2.iter().map(|d| d.offset.value()).collect();
    assert_eq!(
        offsets2,
        (BEFORE as u64..total).collect::<Vec<_>>(),
        "`from_offset` must be an inclusive lower bound on the Postgres-backing drain too: \
         exactly {} offsets from {BEFORE} onward",
        total - BEFORE as u64
    );
    assert_eq!(
        seq_column(&out2[0].batch),
        wide_seq,
        "the wide group must be whole when it is the first group of the window"
    );
}
