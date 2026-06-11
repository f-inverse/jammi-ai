//! Stage 2C — wire coverage for the remaining `Session`-abstraction verbs: the
//! mutable-table lifecycle, the channel-declaration verbs, and the topic-admin
//! verbs (`RegisterTopic` / `DropTopic`) — all on the control-plane
//! `CatalogService` — plus the `AuditService` data-plane verbs.
//!
//! Each test drives an in-process Tonic server hosting the engine-backed chain
//! behind the shared `TenantInterceptor`, exercises the wire path through the
//! proto-generated client stubs, and asserts the engine-side effect (a Flight-
//! SQL / engine read for the mutable table, the catalog for the channel, the
//! typed fetch + `.verify()` for audit, `ListTopics` for the topic). The broker
//! is the in-memory default; every fixture is hermetic and uses realistic
//! schemas. No network.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use jammi_ai::audit::{verify_with_store, EnvSigningKeyStore, MASTER_KEY_ENV};
use jammi_ai::session::InferenceSession;
use jammi_db::TenantId;
use jammi_server::grpc::proto::audit::audit_service_client::AuditServiceClient;
use jammi_server::grpc::proto::audit::{
    AuditFetchByQueryIdRequest, AuditFetchRecentRequest, AuditLogRequest, PerQueryAudit,
};
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{
    AddChannelColumnsRequest, ChannelColumn, ChannelColumnType, CreateMutableTableRequest,
    DropMutableTableRequest, DropTopicRequest, ListTopicsRequest, MutableTableDefinition,
    RegisterChannelRequest, RegisterTopicRequest,
};
use jammi_server::grpc::session::SessionStore;
use jammi_server::TriggerHandles;
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::common::grpc::{channel, with_session, TENANT_A};

/// 32-byte hex master key for the audit HMAC. Matches the engine-level audit
/// integration tests; deterministic so signature verification is reproducible.
const TEST_KEY: &str = "0000000000000000000000000000000000000000000000000000000000000001";

/// `JAMMI_AUDIT_MASTER_KEY` is process-global. Serialize the audit test that
/// mutates it so a concurrent test does not observe a half-set key. Held across
/// `.await`, so an async mutex (not `std::sync`).
fn env_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

/// In-process gRPC server fixture hosting the full engine-backed chain *with*
/// the trigger handles mounted (so the topic-admin verbs are reachable). Keeps
/// an `Arc<InferenceSession>` clone for engine-side read-back assertions, the
/// catalog `TempDir`, and the shutdown signal.
struct Fixture {
    addr: SocketAddr,
    engine: Arc<InferenceSession>,
    shutdown: Option<oneshot::Sender<()>>,
    _dir: TempDir,
    _handle: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

async fn start_fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(InferenceSession::new(cfg).await.expect("session"));

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
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let engine = Arc::clone(&session);
    let handle = tokio::spawn(async move {
        jammi_server::runtime::serve_grpc_chain(
            jammi_server::runtime::GrpcChain {
                addr,
                flight_ctx,
                flight_binding: binding,
                store,
                trigger: Some(trigger),
                engine: Some(session),
                tiers: jammi_server::tiers::TierSet::all_compiled(),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
        .expect("grpc server");
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    Fixture {
        addr,
        engine,
        shutdown: Some(shutdown_tx),
        _dir: dir,
        _handle: handle,
    }
}

/// A realistic dimension-table schema: a string key plus two payload columns
/// and a monotonic order column.
fn dim_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sku", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("price_cents", DataType::Int64, false),
        Field::new("seq", DataType::Int64, false),
    ]))
}

/// Encode an Arrow schema as a schema-only Arrow IPC stream — the wire shape
/// `MutableTableDefinition.schema` / `RegisterTopicRequest.schema` expect.
fn encode_schema_ipc(schema: &SchemaRef) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref()).expect("stream writer");
        writer.finish().expect("finish writer");
    }
    buf
}

// ───────────────────────── Mutable table ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_then_read_then_drop_mutable_table() {
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client = CatalogServiceClient::with_interceptor(ch, with_session("session-mut"));

    let schema = dim_schema();
    let response = client
        .create_mutable_table(CreateMutableTableRequest {
            definition: Some(MutableTableDefinition {
                id: "dim_products".into(),
                schema: encode_schema_ipc(&schema),
                primary_key: vec!["sku".into()],
                indexes: vec![],
                order_column: "seq".into(),
                chunk_size: 0,
                user_metadata: String::new(),
            }),
        })
        .await
        .expect("create mutable table")
        .into_inner();
    assert_eq!(response.mutable_table_id, "dim_products");

    // Confirm the table exists by reading it through the engine's SQL surface
    // (the same query surface Flight SQL federates). A freshly created mutable
    // table is empty; the query succeeding proves it is registered and
    // queryable.
    let batches = fixture
        .engine
        .sql("SELECT \"sku\" FROM mutable.public.dim_products")
        .await
        .expect("query mutable table");
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 0, "newly created mutable table is empty");

    client
        .drop_mutable_table(DropMutableTableRequest {
            mutable_table_id: "dim_products".into(),
        })
        .await
        .expect("drop mutable table");

    // After drop the table is gone — a query against it must fail.
    let err = fixture
        .engine
        .sql("SELECT \"sku\" FROM mutable.public.dim_products")
        .await;
    assert!(err.is_err(), "querying a dropped mutable table must fail");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_mutable_table_rejects_primary_key_not_in_schema() {
    // The engine builder's validation is enforced on the wire path: a primary
    // key naming a column absent from the schema is a client error.
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client = CatalogServiceClient::with_interceptor(ch, with_session("session-mut-bad"));

    let schema = dim_schema();
    let err = client
        .create_mutable_table(CreateMutableTableRequest {
            definition: Some(MutableTableDefinition {
                id: "dim_bad".into(),
                schema: encode_schema_ipc(&schema),
                primary_key: vec!["does_not_exist".into()],
                indexes: vec![],
                order_column: String::new(),
                chunk_size: 0,
                user_metadata: String::new(),
            }),
        })
        .await
        .expect_err("primary key not in schema must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_missing_mutable_table_is_not_found() {
    // The wire status for a missing table is `NotFound`, not the coarse
    // `Internal` the engine-error catch-all would yield. A remote client's
    // `if_exists` drop reads this code to turn a missing-table drop into a no-op,
    // mirroring the in-process `MutableTableError::NotFound` handling — so the
    // mapping is asserted here at the wire seam.
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client = CatalogServiceClient::with_interceptor(ch, with_session("session-mut-gone"));

    let err = client
        .drop_mutable_table(DropMutableTableRequest {
            mutable_table_id: "never_created".into(),
        })
        .await
        .expect_err("dropping a table that was never created must fail");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ───────────────────────── Topics ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_topic_then_list_then_drop() {
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client = CatalogServiceClient::with_interceptor(ch, with_session("session-topic"));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
    ]));

    let registered = client
        .register_topic(RegisterTopicRequest {
            name: "events".into(),
            schema: encode_schema_ipc(&schema),
            broker_metadata: HashMap::new(),
            // Empty id: this DDL-style caller has no client-minted id, so the
            // server mints a fresh UUIDv7.
            topic_id: String::new(),
        })
        .await
        .expect("register topic")
        .into_inner();
    assert!(
        Uuid::parse_str(&registered.topic_id).is_ok(),
        "register returns a UUID topic id"
    );

    let topics = client
        .list_topics(ListTopicsRequest {
            page_size: 0,
            page_token: String::new(),
            tenant_id: String::new(),
        })
        .await
        .expect("list topics")
        .into_inner();
    assert!(
        topics.topics.iter().any(|t| t.name == "events"),
        "ListTopics must show the freshly registered topic; got {:?}",
        topics.topics
    );

    client
        .drop_topic(DropTopicRequest {
            topic_id: registered.topic_id.clone(),
            if_exists: false,
        })
        .await
        .expect("drop topic");

    let after = client
        .list_topics(ListTopicsRequest {
            page_size: 0,
            page_token: String::new(),
            tenant_id: String::new(),
        })
        .await
        .expect("list topics after drop")
        .into_inner();
    assert!(
        !after.topics.iter().any(|t| t.name == "events"),
        "dropped topic must not appear in ListTopics"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_topic_if_exists_is_noop_for_missing_topic() {
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client =
        CatalogServiceClient::with_interceptor(ch, with_session("session-topic-missing"));

    // A syntactically valid UUID that was never registered.
    let missing = Uuid::now_v7().to_string();

    // if_exists = true on a missing topic is a no-op.
    client
        .drop_topic(DropTopicRequest {
            topic_id: missing.clone(),
            if_exists: true,
        })
        .await
        .expect("drop missing topic with if_exists is a no-op");

    // if_exists = false on a missing topic is NotFound.
    let err = client
        .drop_topic(DropTopicRequest {
            topic_id: missing,
            if_exists: false,
        })
        .await
        .expect_err("drop missing topic without if_exists must fail");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ───────────────────────── Channels ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_channel_then_add_columns_visible_in_catalog() {
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client = CatalogServiceClient::with_interceptor(ch, with_session("session-channel"));

    client
        .register_channel(RegisterChannelRequest {
            channel_id: "retriever".into(),
            priority: 10,
            columns: vec![ChannelColumn {
                name: "score".into(),
                data_type: ChannelColumnType::Float64 as i32,
            }],
        })
        .await
        .expect("register channel");

    client
        .add_channel_columns(AddChannelColumnsRequest {
            channel_id: "retriever".into(),
            columns: vec![ChannelColumn {
                name: "rank".into(),
                data_type: ChannelColumnType::Int64 as i32,
            }],
        })
        .await
        .expect("add channel columns");

    // Assert via the catalog: the channel and both columns are persisted.
    let listing = fixture
        .engine
        .catalog()
        .channels()
        .list()
        .await
        .expect("list channels");
    let retriever = listing
        .iter()
        .find(|spec| spec.id.as_str() == "retriever")
        .expect("retriever channel registered");
    assert_eq!(retriever.priority, 10);
    let col_names: Vec<&str> = retriever.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        col_names,
        vec!["score", "rank"],
        "both the initial and appended columns must be present in order"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_channel_rejects_unspecified_column_type() {
    let fixture = start_fixture().await;
    let ch = channel(fixture.addr).await;
    let mut client =
        CatalogServiceClient::with_interceptor(ch, with_session("session-channel-bad"));

    let err = client
        .register_channel(RegisterChannelRequest {
            channel_id: "bad".into(),
            priority: 0,
            columns: vec![ChannelColumn {
                name: "x".into(),
                data_type: ChannelColumnType::Unspecified as i32,
            }],
        })
        .await
        .expect_err("unspecified column type must be rejected");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

// ───────────────────────── Audit ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_log_then_fetch_and_signature_verifies() {
    let _g = env_lock().lock().await;
    std::env::set_var(MASTER_KEY_ENV, TEST_KEY);

    let fixture = start_fixture().await;

    // Bind a tenant for this session — the audit primitive rejects an unscoped
    // call. SetTenant goes through the same shared SessionStore/interceptor.
    set_tenant(fixture.addr, "session-audit", TENANT_A).await;

    let ch = channel(fixture.addr).await;
    let mut client = AuditServiceClient::with_interceptor(ch, with_session("session-audit"));

    let query_id = Uuid::now_v7();
    client
        .audit_log(AuditLogRequest {
            records: vec![PerQueryAudit {
                query_id: query_id.to_string(),
                tenant_id: String::new(),
                model_id: "openai/clip-vit-base".into(),
                model_version: "rev-3".into(),
                query_lineage: serde_json::json!({ "examiner_id": "42" }).to_string(),
                top_k_result_ids: vec!["doc-1".into(), "doc-2".into()],
                retrieval_scores: vec![0.91, 0.84],
                executed_at_micros: 0,
                signature: String::new(),
            }],
        })
        .await
        .expect("audit log");

    // Fetch by query id and verify the signature the engine computed.
    let fetched = client
        .audit_fetch_by_query_id(AuditFetchByQueryIdRequest {
            query_id: query_id.to_string(),
        })
        .await
        .expect("audit fetch by id")
        .into_inner()
        .record
        .expect("record present after log");
    assert_eq!(fetched.query_id, query_id.to_string());
    assert_eq!(fetched.tenant_id, TENANT_A, "tenant stamped on write");
    assert_eq!(fetched.model_id, "openai/clip-vit-base");
    assert!(!fetched.signature.is_empty(), "signature populated on read");

    // The fetched record round-trips through the engine's verifier — proving
    // the wire decode preserved every field the canonical signature covers.
    let engine_record = jammi_db::PerQueryAudit {
        query_id,
        tenant_id: Some(fetched.tenant_id.clone()),
        model_id: fetched.model_id.clone(),
        model_version: fetched.model_version.clone(),
        query_lineage: serde_json::from_str(&fetched.query_lineage).expect("lineage json"),
        top_k_result_ids: fetched.top_k_result_ids.clone(),
        retrieval_scores: fetched.retrieval_scores.clone(),
        executed_at: chrono::DateTime::from_timestamp_micros(fetched.executed_at_micros)
            .expect("executed_at decodes"),
        signature: fetched.signature.clone(),
    };
    verify_with_store(&engine_record, &EnvSigningKeyStore).expect("signature verifies");

    // AuditFetchRecent surfaces the same record.
    let recent = client
        .audit_fetch_recent(AuditFetchRecentRequest { limit: 10 })
        .await
        .expect("audit fetch recent")
        .into_inner();
    assert!(
        recent
            .records
            .iter()
            .any(|r| r.query_id == query_id.to_string()),
        "fetch_recent must include the logged record"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_log_without_tenant_is_failed_precondition() {
    let _g = env_lock().lock().await;
    std::env::set_var(MASTER_KEY_ENV, TEST_KEY);

    let fixture = start_fixture().await;
    // No SetTenant call — the session is unscoped.
    let ch = channel(fixture.addr).await;
    let mut client =
        AuditServiceClient::with_interceptor(ch, with_session("session-audit-unscoped"));

    let err = client
        .audit_log(AuditLogRequest {
            records: vec![PerQueryAudit {
                query_id: Uuid::now_v7().to_string(),
                tenant_id: String::new(),
                model_id: "m".into(),
                model_version: "v".into(),
                query_lineage: "{}".into(),
                top_k_result_ids: vec![],
                retrieval_scores: vec![],
                executed_at_micros: 0,
                signature: String::new(),
            }],
        })
        .await
        .expect_err("unscoped audit log must fail");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// Bind a tenant to a session via `CatalogService.SetTenant` (shared store).
async fn set_tenant(addr: SocketAddr, session_id: &str, tenant: &str) {
    use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};
    let mut client =
        CatalogServiceClient::with_interceptor(channel(addr).await, with_session(session_id));
    // Confirm the tenant string parses (a malformed fixture is a test bug).
    let _typed: TenantId = tenant.parse().expect("valid tenant uuid");
    client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: tenant.to_string(),
            }),
        })
        .await
        .expect("set tenant");
}
