//! SPEC-03 §12 #4 — Flight SQL `jammi-session-id`-bound tenant isolation.
//!
//! Co-mounted with `SessionService` on one Tonic server. A client binds a
//! tenant via the gRPC `SessionService.SetTenant` call carrying a
//! `jammi-session-id` header; the same header on a Flight SQL query routes
//! through [`TenantBoundProvider`] which updates the engine's shared
//! [`TenantBinding`] for the duration of that query.
//!
//! Concurrency caveat (mirrors `crates/jammi-server/src/flight.rs` rustdoc):
//! the binding is process-global. The tests serialise their Flight SQL
//! queries — two concurrent queries on different tenants would race the
//! binding. The gRPC-only test in `grpc_session.rs` exercises the
//! per-request `SessionTenant` extension path that does NOT race; this
//! file exists specifically to cover the Flight SQL surface.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::Int64Array;
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_engine::session::JammiSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::proto::session::session_service_client::SessionServiceClient;
use jammi_server::grpc::proto::session::{SetTenantRequest, Tenant};
use jammi_server::grpc::session::{SessionStore, SESSION_HEADER};
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::net::TcpListener;

use super::common::grpc::{channel, tenant_a, tenant_b, with_session, TENANT_A, TENANT_B};

/// Spin up the Flight SQL + SessionService server in-process. The fixture
/// pre-seeds a Parquet local source with 10 rows split 6 (tenant A) + 4
/// (tenant B) and registers the source under the engine session, declaring
/// `tenant_id` as the federated tenant column. The two clients we attach
/// later read from this one source through the analyzer rule.
async fn start_flight_test_server() -> (SocketAddr, TempDir, tokio::task::JoinHandle<()>) {
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let dir = tempfile::tempdir().expect("tempdir");
    let pq_path = dir.path().join("notes.parquet");

    // 10 rows split 6/4 between tenants.
    let schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("tenant_id", DataType::Utf8, true),
    ]));
    let note_ids = Int64Array::from((0..10_i64).collect::<Vec<_>>());
    let a_str = tenant_a().to_string();
    let b_str = tenant_b().to_string();
    let tenants: Vec<&str> = (0..10)
        .map(|i| {
            if i < 6 {
                a_str.as_str()
            } else {
                b_str.as_str()
            }
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(note_ids) as ArrayRef,
            Arc::new(StringArray::from(tenants)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(&pq_path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let cfg = test_config(dir.path());
    let session = Arc::new(JammiSession::new(cfg).await.expect("session"));
    session
        .add_source(
            "notes",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", pq_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .expect("add notes source");
    session.set_source_tenant_column("notes", Some("tenant_id".into()));

    let store = SessionStore::new();
    let binding = session.tenant_binding_arc();
    let ctx_clone = session.context().clone();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let handle = tokio::spawn(async move {
        jammi_server::flight::serve_flight_with_session_service(&ctx_clone, binding, addr, store)
            .await
            .expect("flight + session server");
    });

    // Briefly wait for the listener to come up.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, dir, handle)
}

async fn bind_tenant_via_session_service(addr: SocketAddr, session_id: &str, tenant: &str) {
    let mut client =
        SessionServiceClient::with_interceptor(channel(addr).await, with_session(session_id));
    client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: tenant.to_string(),
            }),
        })
        .await
        .expect("set_tenant");
}

async fn flight_count_notes(addr: SocketAddr, session_id: &str) -> i64 {
    let ch = channel(addr).await;
    let mut client = FlightSqlServiceClient::new(ch);
    client.set_header(SESSION_HEADER, session_id);
    let info = client
        .execute(
            "SELECT COUNT(*) AS n FROM notes.public.notes".to_string(),
            None,
        )
        .await
        .expect("execute");
    let endpoint = info
        .endpoint
        .first()
        .cloned()
        .expect("flight info must have an endpoint");
    let ticket = endpoint.ticket.expect("endpoint ticket");
    let mut stream = client.do_get(ticket).await.expect("do_get");
    let mut total: i64 = 0;
    while let Some(batch) = stream.try_next().await.expect("flight stream") {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count column");
        for i in 0..col.len() {
            total += col.value(i);
        }
    }
    total
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_two_session_ids_see_disjoint_row_counts() {
    let (addr, _dir, handle) = start_flight_test_server().await;

    bind_tenant_via_session_service(addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(addr, "session-b", TENANT_B).await;

    // Bound bindings are global → run sequentially.
    let count_a = flight_count_notes(addr, "session-a").await;
    let count_b = flight_count_notes(addr, "session-b").await;

    assert_eq!(count_a, 6, "session A's Flight SQL view must yield 6 rows");
    assert_eq!(count_b, 4, "session B's Flight SQL view must yield 4 rows");

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_clearing_tenant_makes_query_unscoped() {
    let (addr, _dir, handle) = start_flight_test_server().await;

    bind_tenant_via_session_service(addr, "session-c", TENANT_A).await;
    let before = flight_count_notes(addr, "session-c").await;
    assert_eq!(before, 6);

    // Clear the binding via the SessionService.
    let mut client =
        SessionServiceClient::with_interceptor(channel(addr).await, with_session("session-c"));
    client.clear_tenant(()).await.expect("clear_tenant");

    // Subsequent query runs unscoped (Unscoped → IS NULL only). The fixture
    // has zero rows with `tenant_id IS NULL`, so the count is 0.
    let after = flight_count_notes(addr, "session-c").await;
    assert_eq!(after, 0, "after clear_tenant the query must run unscoped");

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_set_tenant_invalid_uuid_returns_invalid_argument() {
    let (addr, _dir, handle) = start_flight_test_server().await;

    let mut client = SessionServiceClient::with_interceptor(
        channel(addr).await,
        with_session("session-bad-uuid"),
    );
    let err = client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: "not-a-uuid".to_string(),
            }),
        })
        .await
        .expect_err("expected InvalidArgument");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "expected InvalidArgument; got {:?}: {}",
        err.code(),
        err.message()
    );

    handle.abort();
}
