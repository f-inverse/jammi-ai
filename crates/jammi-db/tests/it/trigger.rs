//! Integration tests for Phase 4 — trigger-stream primitive.
//!
//! Exercises the in-memory broker through the publisher/subscriber surface
//! plus the topic-catalog repo, covering SPEC-04 §15 exit criteria
//! #1 (register-publish-subscribe-filter), #2 (replay correctness),
//! #3 (broadcast fan-out), #4 (tenant-scope isolation), #9 (schema
//! validation), and #10 (backpressure smoke test). The live-broker
//! variant (#5) and the gRPC surface tests land with Phases 4b/4c.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::context::SessionContext;
use futures::StreamExt;
use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::catalog::Catalog;
use jammi_db::source::mutable::MutableTableRegistry;
use jammi_db::store::mutable::postgres::PostgresMutableBackend;
use jammi_db::store::mutable::sqlite::SqliteMutableBackend;
use jammi_db::store::mutable::MutableBackend;
use jammi_db::tenant::{TenantContext, TenantId};
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::trigger::{
    InMemoryBroker, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TopicId,
    TriggerBroker, TriggerError,
};
use jammi_test_utils::unique_suffix;
use std::str::FromStr;
use tempfile::TempDir;
use test_case::test_case;
use uuid::Uuid;

struct Harness {
    _dir: TempDir,
    registry: Arc<MutableTableRegistry>,
    topic_repo: TopicRepo,
    broker: Arc<dyn TriggerBroker>,
    publisher: Publisher,
    subscriber: Subscriber,
    session: SessionContext,
}

async fn build_harness(backend: BackendKind) -> Option<Harness> {
    build_harness_with_tenant(backend, None).await
}

/// Like [`build_harness`] but also hands back the concrete [`InMemoryBroker`]
/// so a test can arm `trigger_failure_for_next_publish`. The same broker is
/// wired into the harness's publisher/subscriber as the `dyn TriggerBroker`,
/// so arming it affects the live fan-out the harness exercises.
async fn build_harness_with_in_memory_broker(
    backend: BackendKind,
) -> Option<(Harness, Arc<InMemoryBroker>)> {
    let in_mem = Arc::new(InMemoryBroker::new());
    let h = build_harness_with_broker(backend, None, Arc::clone(&in_mem) as Arc<dyn TriggerBroker>)
        .await?;
    Some((h, in_mem))
}

async fn build_harness_with_tenant(
    backend: BackendKind,
    tenant: Option<TenantId>,
) -> Option<Harness> {
    build_harness_with_broker(backend, tenant, Arc::new(InMemoryBroker::new())).await
}

/// Build the harness on `backend`, returning `None` for the Postgres arm when
/// `JAMMI_TEST_PG_URL` is unset (so callers skip, never `#[ignore]`).
async fn build_harness_with_broker(
    backend: BackendKind,
    tenant: Option<TenantId>,
    broker: Arc<dyn TriggerBroker>,
) -> Option<Harness> {
    let dir = tempfile::tempdir().unwrap();
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let db_path = dir.path().join("catalog.db");
            let sqlite = SqliteBackend::open(&db_path).await.unwrap();
            BackendImpl::Sqlite(sqlite)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            BackendImpl::Postgres(pg)
        }
    };
    backend_impl.migrate().await.unwrap();

    // The catalog and the mutable-table registry must share ONE tenant binding
    // — exactly as `JammiSession::build` wires them — so a catalog-row lookup
    // (`get_mutable_table`) and a row scan resolve the same tenant. A divergent
    // binding would let the registry believe it is tenant-scoped while the
    // catalog reads unscoped, silently missing the tenant's backing tables.
    let tenant_binding = TenantBinding::unscoped();
    tenant_binding.set_shared(match tenant {
        Some(t) => TenantContext::Scoped(t),
        None => TenantContext::Unscoped,
    });
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();

    let mutable_backend: Arc<dyn MutableBackend> = match backend {
        BackendKind::Sqlite => Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc))),
        BackendKind::Postgres => Arc::new(PostgresMutableBackend::new(Arc::clone(&backend_arc))),
    };
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

    Some(Harness {
        _dir: dir,
        registry,
        topic_repo,
        broker,
        publisher,
        subscriber,
        session: SessionContext::new(),
    })
}

/// Fetch a backend-parameterized harness, skipping the test (with a warning,
/// never `#[ignore]`) when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! harness_or_skip {
    ($build:expr) => {
        match $build.await {
            Some(h) => h,
            None => {
                eprintln!("skipping: JAMMI_TEST_PG_URL unset");
                return;
            }
        }
    };
}

/// A per-test-unique topic name — the Postgres lane shares one catalog
/// database across the whole run, and `topics` enforces `UNIQUE(name,
/// tenant_id)`, so a fixed literal reused across sibling tests (or repeated
/// runs) under the same tenant scope would collide.
fn unique_topic(base: &str) -> String {
    format!("{base}.{}", unique_suffix())
}

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]))
}

fn topic_def(name: &str, tenant: Option<TenantId>) -> TopicDefinition {
    TopicDefinition {
        id: TopicId::new(),
        name: name.to_string(),
        schema: topic_schema(),
        tenant,
        broker_metadata: BTreeMap::new(),
    }
}

fn batch_of(ids: &[i64], kinds: &[&str], values: &[f64]) -> RecordBatch {
    assert_eq!(ids.len(), kinds.len());
    assert_eq!(ids.len(), values.len());
    RecordBatch::try_new(
        topic_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(kinds.to_vec())),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn register_publish_subscribe_filter_end_to_end(backend: BackendKind) {
    // SPEC-04 §15 #1 — register a topic, publish 100 batches of 10 rows,
    // subscribe with `kind = 'X'`, verify only matching batches arrive.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let predicate =
        Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "kind = 'X'").unwrap();

    let mut stream = h
        .subscriber
        .subscribe(&topic, predicate, None)
        .await
        .unwrap();

    // Publish 100 batches; even-indexed batches are kind='X', odd are 'Y'.
    let publisher = h.publisher;
    let topic_clone = topic.clone();
    let publisher_handle = tokio::spawn(async move {
        for i in 0..100 {
            let kind = if i % 2 == 0 { "X" } else { "Y" };
            let kinds = vec![kind; 10];
            let ids: Vec<i64> = (i * 10..i * 10 + 10).collect();
            let values: Vec<f64> = (0..10).map(|j| (i * 10 + j) as f64).collect();
            let batch = batch_of(&ids, &kinds, &values);
            publisher
                .publish_scoped(&topic_clone, None, batch)
                .await
                .unwrap();
        }
    });

    // Receive every matching batch — 50 expected.
    let mut matched_offsets: Vec<u64> = Vec::new();
    while matched_offsets.len() < 50 {
        let delivered = stream.next().await.expect("stream ended early").unwrap();
        matched_offsets.push(delivered.offset.value());
        let kinds = delivered
            .batch
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..kinds.len() {
            assert_eq!(kinds.value(i), "X");
        }
    }
    publisher_handle.await.unwrap();
    assert_eq!(matched_offsets.len(), 50);
    // Offsets must be the even ones in publish order.
    let expected_offsets: Vec<u64> = (0..100).filter(|i| i % 2 == 0).collect();
    assert_eq!(matched_offsets, expected_offsets);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn replay_correctness_after_broker_restart(backend: BackendKind) {
    // SPEC-04 §15 #2 — publish 100 batches, drop the broker (and the
    // subscriber), construct a fresh broker, subscribe with from_offset=0,
    // expect all 100 batches replayed from the backing table.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();
    for i in 0..100i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher
            .publish_scoped(&topic, None, batch)
            .await
            .unwrap();
    }

    // Restart: build a fresh broker (empty in-memory state) bound to the
    // same backing table via a new Subscriber.
    let fresh_broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    fresh_broker.register_topic(&topic).await.unwrap();
    let fresh_subscriber = Subscriber::new(Arc::clone(&fresh_broker), Arc::clone(&h.registry));

    let from = Offset::new(0, chrono::Utc::now());
    let mut stream = fresh_subscriber
        .subscribe(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();

    let mut replayed = 0usize;
    while replayed < 100 {
        let delivered = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("subscribe stream timed out")
            .expect("stream ended early")
            .unwrap();
        let ids = delivered
            .batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids.value(0), replayed as i64);
        replayed += 1;
    }
    assert_eq!(replayed, 100);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn broadcast_fan_out_to_two_subscribers(backend: BackendKind) {
    // SPEC-04 §15 #3 — one topic, two subscriptions with different
    // predicates, mixed publishes; each subscriber sees its matching subset.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let pred_x = Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "kind = 'X'").unwrap();
    let pred_y = Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "kind = 'Y'").unwrap();
    let mut stream_x = h.subscriber.subscribe(&topic, pred_x, None).await.unwrap();
    let subscriber_y = Subscriber::new(Arc::clone(&h.broker), Arc::clone(&h.registry));
    let mut stream_y = subscriber_y.subscribe(&topic, pred_y, None).await.unwrap();

    let publisher = h.publisher;
    let topic_clone = topic.clone();
    let publisher_handle = tokio::spawn(async move {
        for i in 0..100 {
            let kind = if i % 2 == 0 { "X" } else { "Y" };
            let batch = batch_of(&[i], &[kind], &[i as f64]);
            publisher
                .publish_scoped(&topic_clone, None, batch)
                .await
                .unwrap();
        }
    });

    let mut count_x = 0;
    let mut count_y = 0;
    let target = 50usize;
    while count_x < target || count_y < target {
        tokio::select! {
            biased;
            Some(b) = stream_x.next() => {
                let _ = b.unwrap();
                count_x += 1;
            }
            Some(b) = stream_y.next() => {
                let _ = b.unwrap();
                count_y += 1;
            }
        }
    }
    publisher_handle.await.unwrap();
    assert_eq!(count_x, 50);
    assert_eq!(count_y, 50);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn tenant_scope_isolates_topics(backend: BackendKind) {
    // SPEC-04 §15 #4 — tenant A registers t1, tenant B registers t2,
    // neither sees the other's topic via lookup_by_name; the global None
    // tenant sees both.
    let tenant_a = TenantId::from_uuid(Uuid::new_v4()).unwrap();
    let tenant_b = TenantId::from_uuid(Uuid::new_v4()).unwrap();
    let topic_a_name = unique_topic("tenant_a.events");
    let topic_b_name = unique_topic("tenant_b.events");

    let h_a = harness_or_skip!(build_harness_with_tenant(backend, Some(tenant_a)));
    let topic_a = topic_def(&topic_a_name, Some(tenant_a));
    h_a.broker.register_topic(&topic_a).await.unwrap();
    h_a.topic_repo.register_topic(&topic_a).await.unwrap();

    // A second harness for tenant B: on SQLite a different on-disk DB, on
    // Postgres the same shared database — either way `lookup_by_name`'s
    // tenant-scoped WHERE clause is the isolation boundary under test, not
    // physical DB separation, so this proves the same property on both.
    let h_b = harness_or_skip!(build_harness_with_tenant(backend, Some(tenant_b)));
    let topic_b = topic_def(&topic_b_name, Some(tenant_b));
    h_b.broker.register_topic(&topic_b).await.unwrap();
    h_b.topic_repo.register_topic(&topic_b).await.unwrap();

    // Tenant A cannot find tenant B's topic in its own catalog.
    let cross = h_a
        .topic_repo
        .lookup_by_name(&topic_b_name, Some(tenant_a))
        .await
        .unwrap();
    assert!(cross.is_none());

    // Tenant A finds its own.
    let own = h_a
        .topic_repo
        .lookup_by_name(&topic_a_name, Some(tenant_a))
        .await
        .unwrap();
    assert!(own.is_some());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn publish_rejects_schema_mismatch(backend: BackendKind) {
    // SPEC-04 §15 #9 — a batch whose schema does not match the topic's
    // returns BatchSchemaMismatch and writes nothing to the backing table.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let wrong_batch =
        RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();
    match h.publisher.publish_scoped(&topic, None, wrong_batch).await {
        Err(TriggerError::BatchSchemaMismatch(_)) => {}
        other => panic!("expected BatchSchemaMismatch, got {other:?}"),
    }

    // No row in the backing table after a rejected publish.
    let from = Offset::new(0, chrono::Utc::now());
    let mut stream = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let next = tokio::time::timeout(Duration::from_millis(50), stream.next()).await;
    assert!(
        next.is_err(),
        "stream must not yield after a rejected publish"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn backpressure_slows_publisher_without_dropping(backend: BackendKind) {
    // SPEC-04 §15 #10 — a slow subscriber slows the broker tail but does
    // not drop events; offsets must be contiguous and complete after
    // catch-up.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut stream = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), None)
        .await
        .unwrap();

    let publisher = h.publisher;
    let topic_clone = topic.clone();
    // The in-memory channel capacity (1024) is comfortably above the
    // test publish budget; the backpressure assertion here is the
    // offset-contiguity check after catch-up.
    let publisher_handle = tokio::spawn(async move {
        for i in 0..200i64 {
            let batch = batch_of(&[i], &["X"], &[i as f64]);
            publisher
                .publish_scoped(&topic_clone, None, batch)
                .await
                .unwrap();
        }
    });

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < 200 {
        let delivered = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("subscribe stream timed out")
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    publisher_handle.await.unwrap();
    assert_eq!(seen.len(), 200);
    for (i, offset) in seen.iter().enumerate() {
        assert_eq!(*offset, i as u64, "offsets must be contiguous");
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn empty_predicate_matches_every_batch(backend: BackendKind) {
    // Predicate-dialect smoke test: empty string ≡ match_all per SPEC-04 §3.5.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.changes"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let pred = Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "").unwrap();
    let mut stream = h.subscriber.subscribe(&topic, pred, None).await.unwrap();

    let publisher = h.publisher;
    let topic_clone = topic.clone();
    tokio::spawn(async move {
        for i in 0..5i64 {
            let batch = batch_of(&[i], &["X"], &[i as f64]);
            publisher
                .publish_scoped(&topic_clone, None, batch)
                .await
                .unwrap();
        }
    });

    let mut count = 0;
    while count < 5 {
        let delivered = tokio::time::timeout(Duration::from_secs(3), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(delivered.batch.num_rows(), 1);
        count += 1;
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn session_topic_register_drop_round_trip(backend: BackendKind) {
    let dir = tempfile::tempdir().unwrap();
    let session = match jammi_test_utils::make_test_session(backend, dir.path()).await {
        Some(s) => s,
        None => {
            eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
            return;
        }
    };

    // Registering a topic dual-registers the broker driver (so a later publish
    // resolves it) and the catalog (the system of record a lookup reads). The
    // session always carries a broker (defaulting to the in-memory broker), so
    // both registrations succeed.
    let mut broker_metadata = BTreeMap::new();
    broker_metadata.insert("retention_seconds".to_string(), "3600".to_string());
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: unique_topic("orders.changes"),
        schema: Arc::new(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("ts_ms", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, true),
        ])),
        tenant: None,
        broker_metadata,
    };
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .unwrap();
    session.topic_repo().register_topic(&topic).await.unwrap();

    // `list_topics(None)` returns every globally-scoped (`tenant_id IS NULL`)
    // topic — on the shared Postgres lane that accumulates every OTHER
    // sibling test's unscoped topic too, so filter to this test's own topic
    // id before asserting.
    let topics = session.topic_repo().list_topics(None).await.unwrap();
    let mine: Vec<_> = topics.iter().filter(|t| t.id == topic.id).collect();
    assert_eq!(mine.len(), 1);
    assert_eq!(mine[0].name, topic.name);
    assert_eq!(mine[0].schema.fields().len(), 3);
    assert_eq!(
        mine[0].broker_metadata.get("retention_seconds"),
        Some(&"3600".to_string())
    );

    // Dropping removes the catalog row; the broker drop is best-effort.
    session
        .topic_repo()
        .drop_topic(topic.id, None)
        .await
        .unwrap();
    session.trigger_broker().drop_topic(topic.id).await.unwrap();
    let topics = session.topic_repo().list_topics(None).await.unwrap();
    assert!(!topics.iter().any(|t| t.id == topic.id));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn session_drop_missing_topic_is_not_found(backend: BackendKind) {
    let dir = tempfile::tempdir().unwrap();
    let session = match jammi_test_utils::make_test_session(backend, dir.path()).await {
        Some(s) => s,
        None => {
            eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
    let err = session
        .topic_repo()
        .drop_topic(TopicId::new(), None)
        .await
        .unwrap_err();
    assert!(matches!(err, TriggerError::TopicNotFound(_)));
}

#[tokio::test]
async fn predicate_rejects_unsupported_constructs() {
    // SPEC-04 §8.2 — subqueries, aggregates, and other forms are rejected
    // at parse time with `PredicateUnsupported`.
    let session = SessionContext::new();
    let schema = topic_schema();
    let err = match Predicate::from_sql(&session, Arc::clone(&schema), "SUM(id) > 0") {
        Ok(_) => panic!("aggregate predicate must be rejected"),
        Err(e) => e,
    };
    assert!(
        matches!(err, TriggerError::PredicateUnsupported(_)),
        "expected PredicateUnsupported, got {err:?}"
    );

    // SQL parse failures are returned as `PredicateParse`.
    let err = match Predicate::from_sql(&session, Arc::clone(&schema), "(((") {
        Ok(_) => panic!("unparseable predicate must be rejected"),
        Err(e) => e,
    };
    assert!(
        matches!(err, TriggerError::PredicateParse(_)),
        "expected PredicateParse, got {err:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn replay_only_drains_backing_table_without_live_tail(backend: BackendKind) {
    // `Subscriber::replay_only` is the CLI-shaped variant of `subscribe`
    // that returns the replay prefix as a Vec and exits without attaching
    // to the live broker tail. Publishing two batches and replaying from
    // offset 0 must return exactly those two batches, in order.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.replay_only"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..2i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher
            .publish_scoped(&topic, None, batch)
            .await
            .unwrap();
    }

    let from = Offset::new(0, chrono::Utc::now());
    let drained = h
        .subscriber
        .replay_only(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].offset.value(), 0);
    assert_eq!(drained[1].offset.value(), 1);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn replay_only_returns_empty_when_from_offset_none(backend: BackendKind) {
    // Without a `from_offset` the live-tail flow has nothing to replay,
    // so the engine returns an empty Vec rather than blocking on the
    // broker tail.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.replay_only_empty"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..3i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher
            .publish_scoped(&topic, None, batch)
            .await
            .unwrap();
    }

    let drained = h
        .subscriber
        .replay_only(&topic, Predicate::match_all(), None)
        .await
        .unwrap();
    assert!(drained.is_empty());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn replay_only_applies_predicate_to_replay_window(backend: BackendKind) {
    // Predicate filter on the replay path: publish two batches with
    // kind='X' / 'Y'; replay with `kind = 'X'` returns only the X batch.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.replay_only_pred"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    h.publisher
        .publish_scoped(&topic, None, batch_of(&[0], &["X"], &[0.0]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, None, batch_of(&[1], &["Y"], &[1.0]))
        .await
        .unwrap();

    let predicate =
        Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "kind = 'X'").unwrap();
    let from = Offset::new(0, chrono::Utc::now());
    let drained = h
        .subscriber
        .replay_only(&topic, predicate, Some(from))
        .await
        .unwrap();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].offset.value(), 0);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn publish_tags_rows_with_supplied_tenant(backend: BackendKind) {
    // `Publisher::publish_scoped` stamps every persisted row's `tenant_id`
    // with the `tenant` argument by binding it on the backing-table
    // transaction. We verify the stamp by replaying through
    // `Subscriber::replay_only_scoped`, whose tenant-scoped backing-table
    // query (`tenant_id = $current OR tenant_id IS NULL`) surfaces a row
    // iff that row's tenant matches. The tenant literals below are fixed
    // (not per-test-fresh): every assertion is scoped to THIS test's own
    // uniquely-named topic's backing table, so reusing the same tenant ids
    // across sibling tests is safe — the topic name is the isolation axis.
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c84-aaaa-7e10-9c4f-bbbbcccc8e9a").unwrap();

    // The topic is declared unscoped so the publish-time tenant is the
    // only thing that distinguishes the stored rows.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.tenant_stamp"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    // Publish three rows under three different tenant scopes.
    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[1], &["X"], &[1.0]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, Some(tenant_b), batch_of(&[2], &["X"], &[2.0]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, None, batch_of(&[3], &["X"], &[3.0]))
        .await
        .unwrap();

    let from = Offset::new(0, chrono::Utc::now());

    // Tenant A's scope: A's row + the globally-scoped (`None`) row.
    let drained_a = h
        .subscriber
        .replay_only_scoped(&topic, Some(tenant_a), Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let ids_a: Vec<i64> = drained_a
        .iter()
        .flat_map(|d| {
            let col = d
                .batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        ids_a,
        vec![1, 3],
        "tenant A must see its row (id=1) plus the globally-scoped row (id=3)"
    );

    // Tenant B's scope: B's row + the globally-scoped row.
    let drained_b = h
        .subscriber
        .replay_only_scoped(&topic, Some(tenant_b), Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let ids_b: Vec<i64> = drained_b
        .iter()
        .flat_map(|d| {
            let col = d
                .batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        ids_b,
        vec![2, 3],
        "tenant B must see its row (id=2) plus the globally-scoped row (id=3)"
    );

    // Unscoped (None): only globally-scoped rows.
    let drained_none = h
        .subscriber
        .replay_only_scoped(&topic, None, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let ids_none: Vec<i64> = drained_none
        .iter()
        .flat_map(|d| {
            let col = d
                .batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        ids_none,
        vec![3],
        "an unscoped subscriber sees only the row that was published with None"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn subscribe_scoped_filters_published_rows_by_tenant(backend: BackendKind) {
    // End-to-end check that `Publisher::publish_scoped(..., Some(t), ...)`
    // segregates rows across tenants from the subscriber's vantage point.
    // Three rows under {A, B, A}; tenant A's scope must see two, B's one,
    // an unscoped scope zero. Fixed tenant literals are safe here — every
    // assertion is scoped to this test's own uniquely-named topic.
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c84-aaaa-7e10-9c4f-bbbbcccc8e9a").unwrap();

    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.scoped_filter"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[10], &["X"], &[10.0]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, Some(tenant_b), batch_of(&[20], &["X"], &[20.0]))
        .await
        .unwrap();
    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[11], &["X"], &[11.0]))
        .await
        .unwrap();

    let from = Offset::new(0, chrono::Utc::now());

    let drained_a = h
        .subscriber
        .replay_only_scoped(&topic, Some(tenant_a), Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let ids_a: Vec<i64> = drained_a
        .iter()
        .flat_map(|d| {
            let col = d
                .batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        ids_a,
        vec![10, 11],
        "tenant A's scope must see exactly the two A-published rows in publish order"
    );

    let drained_b = h
        .subscriber
        .replay_only_scoped(&topic, Some(tenant_b), Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let ids_b: Vec<i64> = drained_b
        .iter()
        .flat_map(|d| {
            let col = d
                .batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        ids_b,
        vec![20],
        "tenant B's scope must see only the one B-published row"
    );

    // None-scoped subscribers see only `tenant_id IS NULL` rows; none were
    // published under None here, so the replay is empty.
    let drained_none = h
        .subscriber
        .replay_only_scoped(&topic, None, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert!(
        drained_none.is_empty(),
        "an unscoped subscriber must see zero rows when every publish was tenant-tagged"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn publish_returns_error_on_tenant_mismatch_when_topic_is_tenant_pinned(
    backend: BackendKind,
) {
    // When `TopicDefinition::tenant` is `Some(A)`, the topic is pinned and
    // only an `A`-scoped publish is permitted. `Publisher::publish_scoped`
    // rejects anything else with `PublishTenantMismatch` before opening a
    // transaction; the backing table is never touched.
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c84-aaaa-7e10-9c4f-bbbbcccc8e9a").unwrap();

    let h = harness_or_skip!(build_harness_with_tenant(backend, Some(tenant_a)));
    let topic_name = unique_topic("events.pinned");
    let topic = topic_def(&topic_name, Some(tenant_a));
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    // Wrong tenant — typed error, attributable to the publish argument.
    let err = h
        .publisher
        .publish_scoped(&topic, Some(tenant_b), batch_of(&[1], &["X"], &[1.0]))
        .await
        .expect_err("publish under tenant B against an A-pinned topic must fail");
    match err {
        TriggerError::PublishTenantMismatch {
            topic: ref name,
            topic_tenant,
            publish_tenant,
        } => {
            assert_eq!(name, &topic_name);
            assert_eq!(topic_tenant, Some(tenant_a));
            assert_eq!(publish_tenant, Some(tenant_b));
        }
        other => panic!("expected PublishTenantMismatch, got {other:?}"),
    }

    // Unscoped publish against a pinned topic — also rejected.
    let err = h
        .publisher
        .publish_scoped(&topic, None, batch_of(&[2], &["X"], &[2.0]))
        .await
        .expect_err("publish under None against an A-pinned topic must fail");
    assert!(
        matches!(
            err,
            TriggerError::PublishTenantMismatch {
                publish_tenant: None,
                ..
            }
        ),
        "expected PublishTenantMismatch with publish_tenant=None, got {err:?}"
    );

    // Nothing landed in the backing table.
    let from = Offset::new(0, chrono::Utc::now());
    let drained = h
        .subscriber
        .replay_only_scoped(&topic, Some(tenant_a), Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert!(
        drained.is_empty(),
        "rejected publishes must not write any row to the backing table"
    );

    // The matching-tenant publish still works.
    h.publisher
        .publish_scoped(&topic, Some(tenant_a), batch_of(&[3], &["X"], &[3.0]))
        .await
        .expect("publish under the topic's own tenant must succeed");
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn list_consumers_returns_each_subscribers_last_delivered_offset(backend: BackendKind) {
    // SPEC-04 backup/restore hook: `TriggerBroker::list_consumers` returns one
    // `ConsumerOffsetSnapshot` per live subscription, carrying the broker's
    // last-delivered stream sequence. The capture is what a downstream
    // consumer's backup path will dump into the manifest so a restored
    // deployment can resume subscriptions at the right point.
    //
    // Test: register a topic, attach two subscribers, publish three batches,
    // drive both subscriber streams until they observe every batch, then call
    // `list_consumers` and verify both names plus a matching last-delivered
    // offset come back. The in-memory broker has no ack model, so
    // `last_ack_stream_sequence == last_delivered_stream_sequence` by design.
    let h = harness_or_skip!(build_harness(backend));
    let topic = topic_def(&unique_topic("events.list_consumers"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let mut sub_a = h
        .broker
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();
    let consumer_a = sub_a.id.to_string();
    let mut sub_b = h
        .broker
        .subscribe(topic.id, Predicate::match_all(), None)
        .await
        .unwrap();
    let consumer_b = sub_b.id.to_string();

    for i in 0..3i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher
            .publish_scoped(&topic, None, batch)
            .await
            .unwrap();
    }

    // Drain three batches per subscriber so each tracker observes
    // offset = 2 (the last published value).
    for _ in 0..3 {
        let _ = tokio::time::timeout(Duration::from_secs(2), sub_a.next())
            .await
            .expect("sub_a timed out")
            .expect("sub_a stream ended early")
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(2), sub_b.next())
            .await
            .expect("sub_b timed out")
            .expect("sub_b stream ended early")
            .unwrap();
    }

    let mut snapshots = h.broker.list_consumers(topic.id).await.unwrap();
    snapshots.sort_by(|x, y| x.consumer_name.cmp(&y.consumer_name));
    assert_eq!(
        snapshots.len(),
        2,
        "expected one snapshot per live subscription, got {snapshots:?}"
    );
    let names: std::collections::BTreeSet<&str> =
        snapshots.iter().map(|s| s.consumer_name.as_str()).collect();
    assert!(
        names.contains(consumer_a.as_str()) && names.contains(consumer_b.as_str()),
        "list_consumers must surface both subscription ids; got {names:?}, expected {consumer_a} and {consumer_b}"
    );
    for snap in &snapshots {
        assert_eq!(snap.topic_id, topic.id, "snapshot topic_id mismatch");
        assert_eq!(
            snap.last_delivered_stream_sequence, 2,
            "subscriber {} should be at offset 2 after draining three batches",
            snap.consumer_name
        );
        assert_eq!(
            snap.last_ack_stream_sequence, snap.last_delivered_stream_sequence,
            "in-memory broker has no ack model; ack floor must equal delivered"
        );
    }

    // Dropping one subscription removes it from the listing on the next call.
    drop(sub_a);
    let after_drop = h.broker.list_consumers(topic.id).await.unwrap();
    assert_eq!(
        after_drop.len(),
        1,
        "list_consumers must prune the dropped subscription"
    );
    assert_eq!(after_drop[0].consumer_name, consumer_b);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn session_with_broker_swallows_fan_out_failure(backend: BackendKind) {
    // Verifies the session-level broker injection point + the publisher's
    // transactional-outbox contract: a caller-built `InMemoryBroker` armed
    // with `trigger_failure_for_next_publish` is wired into the session via
    // `JammiSession::with_backend_and_broker`. When the publisher fans out,
    // the broker returns its configured driver error; the publisher logs at
    // WARN and still returns Ok because the backing table commit is the
    // authoritative log. Subscribers see the row on replay. The next publish
    // succeeds because the failure was one-shot.
    //
    // Underwrites a downstream consumer's "publish failure does not fail the
    // check" invariant by giving downstream tests a deterministic failure
    // injection point.

    let dir = tempfile::tempdir().unwrap();
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
                .await
                .unwrap();
            BackendImpl::Sqlite(sqlite)
        }
        BackendKind::Postgres => {
            let Some(url) = jammi_test_utils::pg_url_for_tests() else {
                eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
                return;
            };
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            BackendImpl::Postgres(pg)
        }
    };
    let config = jammi_test_utils::test_config(dir.path());
    let broker = Arc::new(InMemoryBroker::new());
    let session = jammi_db::session::JammiSession::with_backend_and_broker(
        config,
        backend_impl,
        Arc::clone(&broker) as Arc<dyn TriggerBroker>,
    )
    .await
    .expect("session with broker");

    let topic = topic_def(&unique_topic("test.session_inject_failure"), None);
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .unwrap();
    session.topic_repo().register_topic(&topic).await.unwrap();

    broker.trigger_failure_for_next_publish("simulated broker outage");

    // Publisher returns Ok despite the armed driver error — the backing
    // table commit is the authoritative log; broker fan-out is best-effort.
    session
        .publisher()
        .publish_scoped(&topic, None, batch_of(&[1], &["X"], &[1.0]))
        .await
        .expect("publish returns Ok despite armed broker failure");

    // Subscriber replay sees the row that landed in the backing table.
    let from = Offset::new(0, chrono::Utc::now());
    let drained = session
        .subscriber()
        .replay_only_scoped(&topic, None, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    assert_eq!(
        drained.iter().map(|d| d.batch.num_rows()).sum::<usize>(),
        1,
        "publisher commits the backing table before broker fan-out per the outbox contract"
    );

    // The next publish reaches the broker normally — the failure was one-shot.
    session
        .publisher()
        .publish_scoped(&topic, None, batch_of(&[2], &["Y"], &[2.0]))
        .await
        .expect("subsequent publish succeeds after the one-shot failure clears");
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn crash_mid_publish_replays_committed_offsets_with_no_loss(backend: BackendKind) {
    // Track T1 — crash-mid-publish replay (hermetic, in-memory).
    //
    // Publish N batches against a real backing table, injecting a
    // post-commit broker fan-out failure on one of them via
    // `InMemoryBroker::trigger_failure_for_next_publish`. The publisher's
    // transactional-outbox contract commits the augmented event BEFORE the
    // best-effort fan-out, so the failed offset is still durably logged.
    //
    // Simulate a crash by dropping the publisher and the broker, stand up a
    // fresh empty broker + subscriber, and replay from offset 0. The full
    // multiset `{0..N-1}` must come back, contiguous — no loss, even at the
    // offset whose live fan-out failed.
    let (h, in_mem) = harness_or_skip!(build_harness_with_in_memory_broker(backend));
    let topic = topic_def(&unique_topic("events.crash_mid_publish"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    const N: i64 = 64;
    // Inject the post-commit fan-out failure on the publish at offset 7 — the
    // commit lands, the broker `publish` returns Err, the publisher logs WARN
    // and returns Ok. The backing table is authoritative.
    const FAIL_AT: i64 = 7;
    for i in 0..N {
        if i == FAIL_AT {
            in_mem.trigger_failure_for_next_publish("simulated post-commit fan-out failure");
        }
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i], &["X"], &[i as f64]))
            .await
            .expect("publish returns Ok even when the best-effort fan-out fails");
    }

    // Crash: drop the publisher and the original broker entirely.
    drop(h.publisher);
    drop(in_mem);
    drop(h.broker);
    // (the harness still owns `registry`, the durable backing-table handle)

    // Fresh broker (empty live state) + subscriber over the same backing table.
    let fresh_broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    fresh_broker.register_topic(&topic).await.unwrap();
    let fresh_subscriber = Subscriber::new(Arc::clone(&fresh_broker), Arc::clone(&h.registry));

    let from = Offset::new(0, chrono::Utc::now());
    let mut stream = fresh_subscriber
        .subscribe(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < N as usize {
        let delivered = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("subscribe stream timed out")
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    assert_eq!(seen.len(), N as usize, "every committed offset must replay");
    for (i, off) in seen.iter().enumerate() {
        assert_eq!(
            *off, i as u64,
            "offsets must replay contiguous {{0..N-1}} — the fan-out failure at offset {FAIL_AT} must not create a gap"
        );
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn live_tail_resumes_with_no_loss_after_post_commit_fan_out_failure(backend: BackendKind) {
    // Track T1 — the in-memory analogue of the JetStream consumer-recreate
    // resume test. A late subscriber attaches at `from_offset` AFTER a
    // post-commit fan-out failure has skewed the broker's view from the
    // engine `_offset`, then keeps consuming as new publishes arrive live.
    // Every committed offset in `[from..max]` must be delivered with no skip,
    // proving the replay/live seam is keyed on the engine `_offset` and not on
    // any broker-native sequence.
    let (h, in_mem) = harness_or_skip!(build_harness_with_in_memory_broker(backend));
    let topic = topic_def(&unique_topic("events.resume_after_failure"), None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    // Phase 1: publish 0..K with a post-commit fan-out failure in the middle.
    const K: i64 = 10;
    const FAIL_AT: i64 = 4;
    for i in 0..K {
        if i == FAIL_AT {
            in_mem.trigger_failure_for_next_publish("simulated post-commit fan-out failure");
        }
        h.publisher
            .publish_scoped(&topic, None, batch_of(&[i], &["X"], &[i as f64]))
            .await
            .unwrap();
    }

    // A late subscriber attaches at offset 0: replay covers [0..K-1] from the
    // backing table, and the live tail overlaps it (deduped by `_offset`).
    let from = Offset::new(0, chrono::Utc::now());
    let mut stream = h
        .subscriber
        .subscribe(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();

    // Phase 2: keep publishing past the attach point — these arrive live.
    const TOTAL: i64 = 20;
    let publisher = h.publisher;
    let topic_clone = topic.clone();
    let pub_handle = tokio::spawn(async move {
        for i in K..TOTAL {
            publisher
                .publish_scoped(&topic_clone, None, batch_of(&[i], &["X"], &[i as f64]))
                .await
                .unwrap();
        }
    });

    let mut seen: Vec<u64> = Vec::new();
    while seen.len() < TOTAL as usize {
        let delivered = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("subscribe stream timed out")
            .expect("stream ended early")
            .unwrap();
        seen.push(delivered.offset.value());
    }
    pub_handle.await.unwrap();

    // No skip anywhere in [0..TOTAL); the seam (replay→live) must not drop the
    // boundary offset even though the broker's live view skewed at FAIL_AT.
    let expected: Vec<u64> = (0..TOTAL as u64).collect();
    let mut deduped = seen.clone();
    deduped.dedup();
    assert_eq!(
        deduped, expected,
        "every committed offset [0..TOTAL) must be delivered in order with no skip"
    );
    // Duplicates, if any, occur only at the replay/live seam — never a skip.
    assert!(
        seen.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1),
        "offsets must be non-decreasing with unit steps (seam duplicates allowed, skips not): {seen:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn at_least_once_no_skip_property_over_randomized_states(backend: BackendKind) {
    // Track T1 — at-least-once / no-skip property test.
    //
    // Over randomized publish counts, subscriber attach points, and an
    // injected post-commit broker fan-out failure offset, assert the
    // delivery contract for every case:
    //   * every committed `_offset` in `[from..max]` is delivered at least
    //     once (no gap / no skip);
    //   * deliveries are non-decreasing and step by at most 1, so any
    //     duplicate is a replay/live SEAM duplicate, never a reorder or skip.
    //
    // Non-vacuity: each case constructs the post-commit-failure state by
    // arming `trigger_failure_for_next_publish` at a deterministically-chosen
    // offset BEFORE that publish, so the engine `_offset` and the broker's
    // live view are genuinely skewed when the subscriber attaches at
    // `from < fail_at` and must cross the seam.

    // Small deterministic LCG (Numerical Recipes constants) — keeps the test
    // hermetic and reproducible without pulling in a PRNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_in(&mut self, modulus: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % modulus
        }
    }

    let mut rng = Lcg(0x1234_5678_9abc_def0);
    // A run-unique prefix: the LCG seed is fixed, so `case` alone repeats the
    // same 24 topic names every run — on the shared Postgres lane, a repeated
    // suite run would collide with the prior run's still-registered topics.
    let run_suffix = unique_suffix();
    // Enough randomized cases to cover the interesting attach/fail orderings
    // without making the hermetic suite slow.
    for case in 0..24u64 {
        let total = 4 + rng.next_in(28); // 4..=31 publishes
                                         // Fail-at lands somewhere inside the published range so the live view
                                         // skews from the engine offset.
        let fail_at = rng.next_in(total);
        // Attach at or before the failure so the subscriber crosses the seam
        // covering the skewed offset.
        let from = rng.next_in(fail_at + 1);

        let (h, in_mem) = match build_harness_with_in_memory_broker(backend).await {
            Some(h) => h,
            None => {
                eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
                return;
            }
        };
        let topic = topic_def(&format!("events.prop_case_{case}.{run_suffix}"), None);
        h.broker.register_topic(&topic).await.unwrap();
        h.topic_repo.register_topic(&topic).await.unwrap();

        for i in 0..total as i64 {
            if i as u64 == fail_at {
                in_mem.trigger_failure_for_next_publish("simulated post-commit fan-out failure");
            }
            h.publisher
                .publish_scoped(&topic, None, batch_of(&[i], &["X"], &[i as f64]))
                .await
                .unwrap();
        }

        let from_off = Offset::new(from, chrono::Utc::now());
        let mut stream = h
            .subscriber
            .subscribe(&topic, Predicate::match_all(), Some(from_off))
            .await
            .unwrap();

        let expected_count = (total - from) as usize;
        let mut seen: Vec<u64> = Vec::new();
        while seen.len() < expected_count {
            let delivered = tokio::time::timeout(Duration::from_secs(5), stream.next())
                .await
                .unwrap_or_else(|_| panic!("case {case}: stream timed out (from={from} fail_at={fail_at} total={total})"))
                .expect("stream ended early")
                .unwrap();
            seen.push(delivered.offset.value());
        }

        // Every offset in [from..total) delivered at least once, no skip.
        let mut deduped = seen.clone();
        deduped.dedup();
        let expected: Vec<u64> = (from..total).collect();
        assert_eq!(
            deduped, expected,
            "case {case}: from={from} fail_at={fail_at} total={total} — every committed offset must be delivered at least once with no skip; saw {seen:?}"
        );
        // Non-decreasing, unit step — duplicates only at the seam.
        assert!(
            seen.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1),
            "case {case}: deliveries must be ordered with unit steps (seam dup ok, skip not): {seen:?}"
        );
        // The seam may duplicate the boundary offset; it must never duplicate
        // more than one offset's worth (a runaway dup would signal the live
        // tail is re-delivering far behind the replay high-water mark).
        let dup_count = seen.len() - deduped.len();
        assert!(
            dup_count <= 1,
            "case {case}: at most one seam duplicate expected, saw {dup_count}: {seen:?}"
        );
    }
}
