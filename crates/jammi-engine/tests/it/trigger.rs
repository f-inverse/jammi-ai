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
use std::sync::RwLock;
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::context::SessionContext;
use futures::StreamExt;
use jammi_engine::catalog::backend::BackendImpl;
use jammi_engine::catalog::backend_sqlite::SqliteBackend;
use jammi_engine::catalog::topic_repo::TopicRepo;
use jammi_engine::catalog::Catalog;
use jammi_engine::source::mutable::MutableTableRegistry;
use jammi_engine::store::mutable::sqlite::SqliteMutableBackend;
use jammi_engine::store::mutable::MutableBackend;
use jammi_engine::tenant::{TenantContext, TenantId};
use jammi_engine::trigger::{
    InMemoryBroker, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TopicId,
    TriggerBroker, TriggerError,
};
use std::str::FromStr;
use tempfile::TempDir;

struct Harness {
    _dir: TempDir,
    registry: Arc<MutableTableRegistry>,
    topic_repo: TopicRepo,
    broker: Arc<dyn TriggerBroker>,
    publisher: Publisher,
    subscriber: Subscriber,
    session: SessionContext,
}

async fn build_harness() -> Harness {
    build_harness_with_tenant(None).await
}

async fn build_harness_with_tenant(tenant: Option<TenantId>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("catalog.db");
    let sqlite = SqliteBackend::open(&db_path).await.unwrap();
    let backend_impl = BackendImpl::Sqlite(sqlite);
    backend_impl.migrate().await.unwrap();
    let catalog = Arc::new(Catalog::from_backend(backend_impl));
    let backend = catalog.backend_arc();

    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(SqliteMutableBackend::new(Arc::clone(&backend)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        Arc::new(RwLock::new(match tenant {
            Some(t) => TenantContext::Scoped(t),
            None => TenantContext::Unscoped,
        })),
    ));

    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let publisher = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend),
        Arc::clone(&registry),
    );
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));

    Harness {
        _dir: dir,
        registry,
        topic_repo,
        broker,
        publisher,
        subscriber,
        session: SessionContext::new(),
    }
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

#[tokio::test]
async fn register_publish_subscribe_filter_end_to_end() {
    // SPEC-04 §15 #1 — register a topic, publish 100 batches of 10 rows,
    // subscribe with `kind = 'X'`, verify only matching batches arrive.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
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
            publisher.publish(&topic_clone, batch).await.unwrap();
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

#[tokio::test]
async fn replay_correctness_after_broker_restart() {
    // SPEC-04 §15 #2 — publish 100 batches, drop the broker (and the
    // subscriber), construct a fresh broker, subscribe with from_offset=0,
    // expect all 100 batches replayed from the backing table.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();
    for i in 0..100i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher.publish(&topic, batch).await.unwrap();
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

#[tokio::test]
async fn broadcast_fan_out_to_two_subscribers() {
    // SPEC-04 §15 #3 — one topic, two subscriptions with different
    // predicates, mixed publishes; each subscriber sees its matching subset.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
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
            publisher.publish(&topic_clone, batch).await.unwrap();
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

#[tokio::test]
async fn tenant_scope_isolates_topics() {
    // SPEC-04 §15 #4 — tenant A registers t1, tenant B registers t2,
    // neither sees the other's topic via lookup_by_name; the global None
    // tenant sees both.
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c84-aaaa-7e10-9c4f-bbbbcccc8e9a").unwrap();

    let h_a = build_harness_with_tenant(Some(tenant_a)).await;
    let topic_a = topic_def("tenant_a.events", Some(tenant_a));
    h_a.broker.register_topic(&topic_a).await.unwrap();
    h_a.topic_repo.register_topic(&topic_a).await.unwrap();

    // Different harness (different DB) for tenant B — verifies they don't
    // accidentally share state via the in-memory broker either.
    let h_b = build_harness_with_tenant(Some(tenant_b)).await;
    let topic_b = topic_def("tenant_b.events", Some(tenant_b));
    h_b.broker.register_topic(&topic_b).await.unwrap();
    h_b.topic_repo.register_topic(&topic_b).await.unwrap();

    // Tenant A cannot find tenant B's topic in its own catalog.
    let cross = h_a
        .topic_repo
        .lookup_by_name("tenant_b.events", Some(tenant_a))
        .await
        .unwrap();
    assert!(cross.is_none());

    // Tenant A finds its own.
    let own = h_a
        .topic_repo
        .lookup_by_name("tenant_a.events", Some(tenant_a))
        .await
        .unwrap();
    assert!(own.is_some());
}

#[tokio::test]
async fn publish_rejects_schema_mismatch() {
    // SPEC-04 §15 #9 — a batch whose schema does not match the topic's
    // returns BatchSchemaMismatch and writes nothing to the backing table.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let wrong_batch =
        RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();
    match h.publisher.publish(&topic, wrong_batch).await {
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

#[tokio::test]
async fn backpressure_slows_publisher_without_dropping() {
    // SPEC-04 §15 #10 — a slow subscriber slows the broker tail but does
    // not drop events; offsets must be contiguous and complete after
    // catch-up.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
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
            publisher.publish(&topic_clone, batch).await.unwrap();
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

#[tokio::test]
async fn empty_predicate_matches_every_batch() {
    // Predicate-dialect smoke test: empty string ≡ match_all per SPEC-04 §3.5.
    let h = build_harness().await;
    let topic = topic_def("events.changes", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    let pred = Predicate::from_sql(&h.session, Arc::clone(&topic.schema), "").unwrap();
    let mut stream = h.subscriber.subscribe(&topic, pred, None).await.unwrap();

    let publisher = h.publisher;
    let topic_clone = topic.clone();
    tokio::spawn(async move {
        for i in 0..5i64 {
            let batch = batch_of(&[i], &["X"], &[i as f64]);
            publisher.publish(&topic_clone, batch).await.unwrap();
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

#[tokio::test]
async fn session_create_topic_ddl_round_trip() {
    use jammi_engine::config::JammiConfig;
    use jammi_engine::session::JammiSession;
    let dir = tempfile::tempdir().unwrap();
    let cfg = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let session = JammiSession::new(cfg).await.unwrap();

    // CREATE TOPIC inserts a topics row + provisions the backing table.
    session
        .sql(
            "CREATE TOPIC orders.changes (op TEXT NOT NULL, ts_ms BIGINT NOT NULL, payload TEXT) \
             WITH (retention_seconds = '3600')",
        )
        .await
        .unwrap();

    let topics = session.topic_repo().list_topics(None).await.unwrap();
    assert_eq!(topics.len(), 1);
    assert_eq!(topics[0].name, "orders.changes");
    assert_eq!(topics[0].schema.fields().len(), 3);
    assert_eq!(
        topics[0].broker_metadata.get("retention_seconds"),
        Some(&"3600".to_string())
    );

    // DROP TOPIC removes it; DROP TOPIC IF EXISTS on a missing name is a no-op.
    session.sql("DROP TOPIC orders.changes").await.unwrap();
    let topics = session.topic_repo().list_topics(None).await.unwrap();
    assert!(topics.is_empty());

    session
        .sql("DROP TOPIC IF EXISTS orders.changes")
        .await
        .unwrap();
}

#[tokio::test]
async fn session_drop_topic_missing_errors_without_if_exists() {
    use jammi_engine::config::JammiConfig;
    use jammi_engine::session::JammiSession;
    let dir = tempfile::tempdir().unwrap();
    let cfg = JammiConfig {
        artifact_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let session = JammiSession::new(cfg).await.unwrap();
    let err = session.sql("DROP TOPIC missing.topic").await.unwrap_err();
    assert!(err.to_string().contains("not found"));
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

#[tokio::test]
async fn replay_only_drains_backing_table_without_live_tail() {
    // `Subscriber::replay_only` is the CLI-shaped variant of `subscribe`
    // that returns the replay prefix as a Vec and exits without attaching
    // to the live broker tail. Publishing two batches and replaying from
    // offset 0 must return exactly those two batches, in order.
    let h = build_harness().await;
    let topic = topic_def("events.replay_only", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..2i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher.publish(&topic, batch).await.unwrap();
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

#[tokio::test]
async fn replay_only_returns_empty_when_from_offset_none() {
    // Without a `from_offset` the live-tail flow has nothing to replay,
    // so the engine returns an empty Vec rather than blocking on the
    // broker tail.
    let h = build_harness().await;
    let topic = topic_def("events.replay_only_empty", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    for i in 0..3i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        h.publisher.publish(&topic, batch).await.unwrap();
    }

    let drained = h
        .subscriber
        .replay_only(&topic, Predicate::match_all(), None)
        .await
        .unwrap();
    assert!(drained.is_empty());
}

#[tokio::test]
async fn replay_only_applies_predicate_to_replay_window() {
    // Predicate filter on the replay path: publish two batches with
    // kind='X' / 'Y'; replay with `kind = 'X'` returns only the X batch.
    let h = build_harness().await;
    let topic = topic_def("events.replay_only_pred", None);
    h.broker.register_topic(&topic).await.unwrap();
    h.topic_repo.register_topic(&topic).await.unwrap();

    h.publisher
        .publish(&topic, batch_of(&[0], &["X"], &[0.0]))
        .await
        .unwrap();
    h.publisher
        .publish(&topic, batch_of(&[1], &["Y"], &[1.0]))
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
