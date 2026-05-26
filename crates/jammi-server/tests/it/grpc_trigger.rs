//! SPEC-04 §15 #5b — `TriggerService.Publish` + `TriggerService.Subscribe`
//! over a real Tonic gRPC transport.
//!
//! The engine-level trigger tests in `crates/jammi-db/tests/it/trigger.rs`
//! exercise the publisher/subscriber surface directly; this module verifies
//! the wire path: an in-process Tonic server hosting `SessionService` +
//! `TriggerService` behind the shared `TenantInterceptor`, two client
//! channels driven through the proto-generated stubs, and the IPC
//! round-trip on the `ArrowBatch` payload.
//!
//! The broker is always the in-memory implementation (no JetStream, no
//! external services); the tests are hermetic and run in microseconds.
//! Each fixture seeds its own `JammiSession` with the topics it needs and
//! is torn down via the fixture's `Drop` impl, which signals the server's
//! `oneshot::Receiver<()>` shutdown future. No background tasks survive a
//! test.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use jammi_db::session::JammiSession;
use jammi_db::TenantId;
use jammi_server::grpc::proto::session::session_service_client::SessionServiceClient;
use jammi_server::grpc::proto::session::{SetTenantRequest, Tenant};
use jammi_server::grpc::proto::trigger::trigger_service_client::TriggerServiceClient;
use jammi_server::grpc::proto::trigger::{
    ArrowBatch, PublishRequest, SubscribeRequest, SubscribedBatch, TopicName,
};
use jammi_server::grpc::session::SessionStore;
use jammi_server::TriggerHandles;
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use super::common::grpc::{channel, tenant_a, with_session, TENANT_A, TENANT_B};

/// User-payload schema used by every test in this module. Matches the
/// engine-level `topic_schema` helper but trimmed to the two columns
/// these tests actually exercise.
fn events_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
    ]))
}

/// Build a `RecordBatch` matching [`events_schema`]. Mirrors the engine-
/// level `batch_of` helper.
fn make_event_batch(ids: &[i64], kinds: &[&str]) -> RecordBatch {
    assert_eq!(
        ids.len(),
        kinds.len(),
        "id/kind columns must be equal length"
    );
    RecordBatch::try_new(
        events_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(kinds.to_vec())),
        ],
    )
    .expect("record batch")
}

/// Encode a batch as Arrow IPC and pack it into the wire shape the server
/// emits — `data_header` empty, full `StreamWriter` payload in `data_body`.
/// `decode_arrow_batch` on the server concatenates the two anyway, so this
/// is the symmetric shape.
fn encode_batch_to_ipc(batch: &RecordBatch) -> ArrowBatch {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new(&mut buf, batch.schema().as_ref()).expect("stream writer");
        writer.write(batch).expect("write batch");
        writer.finish().expect("finish writer");
    }
    ArrowBatch {
        data_header: Vec::new(),
        data_body: buf,
        app_metadata: Vec::new(),
    }
}

/// Inverse of [`encode_batch_to_ipc`]: decode the wire payload back into a
/// `RecordBatch` and assert its schema matches the topic's. Panics on any
/// decode failure since these are integration tests asserting the server
/// produced a well-formed batch.
fn decode_subscribed_batch(s: &SubscribedBatch, expected: &SchemaRef) -> RecordBatch {
    let wire = s
        .batch
        .as_ref()
        .expect("subscribed batch carries an ArrowBatch");
    let mut bytes = Vec::with_capacity(wire.data_header.len() + wire.data_body.len());
    bytes.extend_from_slice(&wire.data_header);
    bytes.extend_from_slice(&wire.data_body);
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = StreamReader::try_new(cursor, None).expect("ipc reader");
    let batch = reader
        .next()
        .expect("stream contains a batch")
        .expect("decode batch");
    assert_eq!(
        batch.schema().as_ref(),
        expected.as_ref(),
        "subscribed batch schema must match topic schema"
    );
    batch
}

/// Declarative seed for a topic the fixture must register before the
/// server starts. `tenant: None` means a globally-scoped topic (anyone
/// with no session tenant can publish to it); `Some(t)` binds the topic
/// to a specific tenant so `lookup_by_name` filters it.
struct TopicSeed {
    name: String,
    schema_ddl: String,
    tenant: Option<TenantId>,
}

impl TopicSeed {
    fn new(name: impl Into<String>, schema_ddl: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            schema_ddl: schema_ddl.into(),
            tenant: None,
        }
    }

    fn with_tenant(mut self, tenant: TenantId) -> Self {
        self.tenant = Some(tenant);
        self
    }
}

/// In-process gRPC server fixture. Wraps the listener address, the
/// shared `SessionStore`, the catalog's `TempDir`, the spawned server
/// task, and a `oneshot::Sender<()>` used to signal shutdown. The
/// `Drop` impl sends `()` so any test that early-returns still tears
/// down the server task cleanly.
struct ServerFixture {
    addr: SocketAddr,
    _store: SessionStore,
    shutdown: Option<oneshot::Sender<()>>,
    _dir: TempDir,
    _handle: JoinHandle<()>,
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            // The receiver may already have been dropped if the server
            // task exited on its own; either way, we don't care about
            // the result — we just want the signal sent before the
            // listener address is freed.
            let _ = tx.send(());
        }
    }
}

/// Spin up an in-process gRPC server hosting `SessionService` +
/// `TriggerService` behind the shared interceptor. Each `TopicSeed` is
/// registered via the engine's `CREATE TOPIC` DDL path (which provisions
/// the backing table and registers the broker channel atomically), so the
/// fixture matches the production code path rather than poking the
/// `TopicRepo` directly.
async fn start_grpc_test_server(seeds: &[TopicSeed]) -> ServerFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(JammiSession::new(cfg).await.expect("session"));

    for seed in seeds {
        match seed.tenant {
            Some(t) => session.bind_tenant(t),
            None => session.unbind_tenant(),
        }
        let ddl = format!("CREATE TOPIC {} ({})", seed.name, seed.schema_ddl);
        session.sql(&ddl).await.expect("create topic");
    }
    // Restore unscoped binding so the gRPC handler uses the per-request
    // `SessionTenant` from the interceptor rather than whatever the seed
    // loop happened to leave behind.
    session.unbind_tenant();

    let store = SessionStore::new();
    let trigger = TriggerHandles {
        topic_repo: session.topic_repo(),
        publisher: session.publisher(),
        subscriber: session.subscriber(),
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let store_for_server = store.clone();
    let handle = tokio::spawn(async move {
        jammi_server::serve_grpc_with_shutdown(addr, store_for_server, Some(trigger), async move {
            let _ = shutdown_rx.await;
        })
        .await
        .expect("grpc server");
    });

    // Give the listener a moment to bind. Matches `grpc_session.rs` —
    // the 50ms window is comfortably above the kernel's bind latency
    // (microseconds on loopback) and the tonic handshake setup.
    tokio::time::sleep(Duration::from_millis(50)).await;

    ServerFixture {
        addr,
        _store: store,
        shutdown: Some(shutdown_tx),
        _dir: dir,
        _handle: handle,
    }
}

/// Bind a tenant to a session via `SessionService.SetTenant`. Returns the
/// raw tonic `Status` on failure so tests can assert error codes.
async fn set_tenant_for_session(
    addr: SocketAddr,
    session_id: &str,
    tenant: &str,
) -> Result<(), tonic::Status> {
    let mut client =
        SessionServiceClient::with_interceptor(channel(addr).await, with_session(session_id));
    client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: tenant.to_string(),
            }),
        })
        .await
        .map(|_| ())
}

/// Drain `expected` items from the subscribe stream, then assert no
/// further items arrive within `quiet_window`. The 100ms quiet window is
/// justified because actual broker→client delivery is microseconds on
/// loopback — anything slower indicates a real engine bug.
async fn drain_n_and_assert_quiet<S>(
    stream: &mut S,
    expected: usize,
    quiet_window: Duration,
) -> Vec<SubscribedBatch>
where
    S: futures::Stream<Item = Result<SubscribedBatch, tonic::Status>> + Unpin,
{
    let mut received = Vec::with_capacity(expected);
    while received.len() < expected {
        let item = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("subscribe stream timed out before expected batch arrived")
            .expect("subscribe stream ended early")
            .expect("subscribe item must be Ok");
        received.push(item);
    }
    // Now confirm the stream is quiet — no spurious deliveries beyond
    // the expected count. A `timeout(...).await` resolving to `Err` is
    // the proof that `.next()` is still pending after the window.
    if tokio::time::timeout(quiet_window, stream.next())
        .await
        .is_ok()
    {
        panic!("subscribe stream produced more than {expected} items");
    }
    received
}

// ---------- Publish tests ----------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_publish_round_trips_batch_to_backing_table() {
    // Publish one batch over gRPC; subscribe with from_offset=0 (which
    // replays from the backing table) and confirm the same payload comes
    // back. Together this proves the wire encode/decode + transactional
    // outbox path is round-tripping correctly.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut publish_client =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-rt"));

    let batch = make_event_batch(&[1, 2, 3], &["X", "Y", "X"]);
    let response = publish_client
        .publish(PublishRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            batch: Some(encode_batch_to_ipc(&batch)),
            tenant_id: String::new(),
        })
        .await
        .expect("publish");
    let assigned = response.into_inner();
    assert_eq!(assigned.offset, 0, "first publish must be offset 0");
    assert!(
        assigned.committed_at.is_some(),
        "publish response must carry committed_at"
    );

    // Now read it back via a from_offset=0 subscription. This drives the
    // backing-table replay path; confirming round-trip end-to-end.
    let mut subscribe_client =
        TriggerServiceClient::with_interceptor(ch, with_session("session-rt"));
    let mut stream = subscribe_client
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: Some(0),
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();

    let received = drain_n_and_assert_quiet(&mut stream, 1, Duration::from_millis(100)).await;
    let got = decode_subscribed_batch(&received[0], &events_schema());
    assert_eq!(got.num_rows(), 3);
    let ids = got
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_publish_rejects_schema_mismatch_with_invalid_argument() {
    // Encode a batch whose schema does not match the topic; the server
    // must reject with `InvalidArgument` (per `decode_arrow_batch`).
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut client = TriggerServiceClient::with_interceptor(ch, with_session("session-mismatch"));

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let wrong = RecordBatch::try_new(wrong_schema, vec![Arc::new(Int64Array::from(vec![1_i64]))])
        .expect("wrong record batch");

    let err = client
        .publish(PublishRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            batch: Some(encode_batch_to_ipc(&wrong)),
            tenant_id: String::new(),
        })
        .await
        .expect_err("schema-mismatched publish must fail");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "expected InvalidArgument; got {:?}: {}",
        err.code(),
        err.message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_publish_on_unknown_topic_returns_not_found() {
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut client = TriggerServiceClient::with_interceptor(ch, with_session("session-missing"));

    let batch = make_event_batch(&[1], &["X"]);
    let err = client
        .publish(PublishRequest {
            topic: Some(TopicName {
                name: "no_such_topic".into(),
            }),
            batch: Some(encode_batch_to_ipc(&batch)),
            tenant_id: String::new(),
        })
        .await
        .expect_err("publish to unknown topic must fail");
    assert_eq!(
        err.code(),
        tonic::Code::NotFound,
        "expected NotFound; got {:?}: {}",
        err.code(),
        err.message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_publish_under_session_tenant_is_invisible_to_other_tenant() {
    // Tenant A registers `notifications`; tenant B should not be able to
    // publish or subscribe to that topic name — the catalog lookup is
    // filtered by tenant, so the server returns NotFound.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "notifications",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )
    .with_tenant(tenant_a())])
    .await;

    // Tenant A binds, publishes — succeeds.
    set_tenant_for_session(fixture.addr, "session-a", TENANT_A)
        .await
        .expect("set tenant A");
    let mut client_a = TriggerServiceClient::with_interceptor(
        channel(fixture.addr).await,
        with_session("session-a"),
    );
    let batch = make_event_batch(&[42], &["X"]);
    client_a
        .publish(PublishRequest {
            topic: Some(TopicName {
                name: "notifications".into(),
            }),
            batch: Some(encode_batch_to_ipc(&batch)),
            tenant_id: String::new(),
        })
        .await
        .expect("tenant A publish");

    // Tenant B binds, tries the same topic — NotFound.
    set_tenant_for_session(fixture.addr, "session-b", TENANT_B)
        .await
        .expect("set tenant B");
    let mut client_b = TriggerServiceClient::with_interceptor(
        channel(fixture.addr).await,
        with_session("session-b"),
    );
    let err = client_b
        .publish(PublishRequest {
            topic: Some(TopicName {
                name: "notifications".into(),
            }),
            batch: Some(encode_batch_to_ipc(&batch)),
            tenant_id: String::new(),
        })
        .await
        .expect_err("tenant B must not see tenant A's topic");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ---------- Subscribe tests ----------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_subscribe_receives_subsequent_publishes_in_order() {
    // Attach the subscriber first (no from_offset → live-only), then
    // publish 5 batches, then assert offsets 0..5 arrived in order.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut subscriber =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-live"));
    let mut publisher = TriggerServiceClient::with_interceptor(ch, with_session("session-live"));

    let mut stream = subscriber
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: None,
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();

    for i in 0..5_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish");
    }

    let received = drain_n_and_assert_quiet(&mut stream, 5, Duration::from_millis(100)).await;
    for (i, item) in received.iter().enumerate() {
        assert_eq!(item.offset, i as u64, "offsets must be contiguous");
        let batch = decode_subscribed_batch(item, &events_schema());
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.value(0), i as i64);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_subscribe_with_from_offset_zero_replays_history() {
    // Publish 3 batches *before* subscribing; with from_offset=0 the
    // subscriber must see every one of them via backing-table replay.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut publisher =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-replay"));
    for i in 0..3_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish");
    }

    let mut subscriber = TriggerServiceClient::with_interceptor(ch, with_session("session-replay"));
    let mut stream = subscriber
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: Some(0),
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();

    let received = drain_n_and_assert_quiet(&mut stream, 3, Duration::from_millis(100)).await;
    let offsets: Vec<u64> = received.iter().map(|b| b.offset).collect();
    assert_eq!(offsets, vec![0, 1, 2], "replay must yield 0, 1, 2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_subscribe_predicate_filters_batches_server_side() {
    // Predicate `kind = 'X'` — odd-indexed publishes (`kind='Y'`) must
    // not arrive. Drives the server-side `Predicate::from_sql` path.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut subscriber =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-pred"));
    let mut stream = subscriber
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: "kind = 'X'".into(),
            from_offset: None,
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();

    let mut publisher = TriggerServiceClient::with_interceptor(ch, with_session("session-pred"));
    for i in 0..6_i64 {
        let kind = if i % 2 == 0 { "X" } else { "Y" };
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &[kind]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish");
    }

    // 3 of 6 published batches match `kind = 'X'`.
    let received = drain_n_and_assert_quiet(&mut stream, 3, Duration::from_millis(150)).await;
    for item in &received {
        let batch = decode_subscribed_batch(item, &events_schema());
        let kinds = batch
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for j in 0..kinds.len() {
            assert_eq!(
                kinds.value(j),
                "X",
                "predicate must filter out non-matching batches"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_subscribe_invalid_predicate_returns_invalid_argument() {
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut client = TriggerServiceClient::with_interceptor(ch, with_session("session-badpred"));
    let err = client
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            // Aggregate predicates are rejected at parse time per SPEC-04 §8.2.
            predicate: "SUM(id) > 0".into(),
            from_offset: None,
            tenant_id: String::new(),
        })
        .await
        .expect_err("aggregate predicate must be rejected");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "expected InvalidArgument; got {:?}: {}",
        err.code(),
        err.message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_subscribe_client_drop_does_not_corrupt_broker_state() {
    // After a client drops its Subscribe stream, the broker and backing
    // table must remain consistent: a fresh subscriber attached after
    // the drop with from_offset = Some(0) must see the full ordered
    // sequence of every batch ever published, including the ones
    // published before, during, and after the dropped subscription.
    // If the gRPC handler corrupted the broker's per-topic state on
    // client drop (e.g., poisoning the broadcast channel or detaching
    // the backing-table sink), the second subscribe would either fail,
    // miss batches, or return them out of order — all of which this
    // test detects deterministically.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "events",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut subscriber =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-drop"));
    let mut publisher = TriggerServiceClient::with_interceptor(ch, with_session("session-drop"));

    // Publish 3 batches before any subscriber attaches.
    for i in 0..3_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish pre-subscribe");
    }

    // Attach a live subscriber, receive the next 3, then drop the stream
    // mid-flight (with no graceful Cancel sent).
    let mut stream = subscriber
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: None,
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();
    for i in 3..6_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish during subscribe");
    }
    let received_live = drain_n_and_assert_quiet(&mut stream, 3, Duration::from_millis(100)).await;
    assert_eq!(received_live.len(), 3, "live tail must see 3 batches");
    drop(stream);

    // Publish 3 more batches after the drop. If the broker corrupted on
    // drop, these publishes would either fail or write to a wrong offset.
    for i in 6..9_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "events".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish post-drop");
    }

    // Attach a fresh subscriber with replay from offset 0; it must see
    // every batch ever published in strict order.
    let mut replay_client = TriggerServiceClient::with_interceptor(
        channel(fixture.addr).await,
        with_session("session-drop"),
    );
    let mut replay_stream = replay_client
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: Some(0),
            tenant_id: String::new(),
        })
        .await
        .expect("replay subscribe")
        .into_inner();
    let replayed =
        drain_n_and_assert_quiet(&mut replay_stream, 9, Duration::from_millis(100)).await;
    for (i, item) in replayed.iter().enumerate() {
        assert_eq!(
            item.offset, i as u64,
            "offset gap or reorder after client drop"
        );
        let batch = decode_subscribed_batch(item, &events_schema());
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.value(0), i as i64, "id mismatch after client drop");
    }
}

// ---------- End-to-end round-trip ----------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_publish_subscribe_round_trip_preserves_order() {
    // Publish 20 batches, drain 20 from the live subscription, assert
    // strict offset ordering 0..20 and matching ids.
    let fixture = start_grpc_test_server(&[TopicSeed::new(
        "audit_log_demo",
        "id BIGINT NOT NULL, kind TEXT NOT NULL",
    )])
    .await;
    let ch = channel(fixture.addr).await;
    let mut subscriber =
        TriggerServiceClient::with_interceptor(ch.clone(), with_session("session-rt20"));
    let mut publisher = TriggerServiceClient::with_interceptor(ch, with_session("session-rt20"));

    let mut stream = subscriber
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "audit_log_demo".into(),
            }),
            predicate: String::new(),
            from_offset: None,
            tenant_id: String::new(),
        })
        .await
        .expect("subscribe")
        .into_inner();

    for i in 0..20_i64 {
        publisher
            .publish(PublishRequest {
                topic: Some(TopicName {
                    name: "audit_log_demo".into(),
                }),
                batch: Some(encode_batch_to_ipc(&make_event_batch(&[i], &["X"]))),
                tenant_id: String::new(),
            })
            .await
            .expect("publish");
    }

    let received = drain_n_and_assert_quiet(&mut stream, 20, Duration::from_millis(100)).await;
    for (i, item) in received.iter().enumerate() {
        assert_eq!(item.offset, i as u64, "offset {i} mismatch");
        let batch = decode_subscribed_batch(item, &events_schema());
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.value(0), i as i64, "id at offset {i} mismatch");
    }
}
