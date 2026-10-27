//! Flight SQL `jammi-session-id`-bound tenant isolation.
//!
//! Co-mounted with `CatalogService` on one Tonic server. A client binds a
//! tenant via `CatalogService.SetTenant` carrying a `jammi-session-id`
//! header; the same header on a Flight SQL query is resolved by the server's
//! Flight scope, and the query runs under that tenant alone — concurrently
//! with other tenants' queries, and without leaving a tenant behind for a
//! session that names none.

use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};
use jammi_server::grpc::session::{SessionStore, SESSION_HEADER};
use jammi_test_utils::test_config;
use tempfile::TempDir;

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

    // Flight SQL + the control-plane `CatalogService` on one listener — the same
    // surface the multi-tenant Flight shape serves (Flight scoped through the
    // `ResolvedFlightScope`, `CatalogService.SetTenant` binding the tenant via
    // the engine-default `jammi-session-id` resolver), assembled from the
    // canonical gRPC chain with no engine mounted. Binding eagerly through the
    // chain seam holds the port from bind through serve, so no concurrent test
    // process can steal it in a release-then-rebind window.
    let store = SessionStore::new();
    let chain = jammi_server::runtime::GrpcChain {
        addr: super::common::grpc::ephemeral_addr(),
        flight_ctx: session.context().clone(),
        store: store.clone(),
        trigger: None,
        engine: None,
        tiers: jammi_server::tiers::TierSet::resolve(std::iter::empty()),
        metrics: Arc::new(jammi_server::routes::health::MetricsRegistry::new().unwrap()),
        tenant_resolver: jammi_server::grpc::session::SessionIdTenantResolver::arc(store),
        admin_authorizer: None,
        limits: jammi_db::config::LimitsConfig::default(),
    };
    let bound = jammi_server::runtime::assemble_grpc_chain(chain)
        .expect("assemble")
        .bind()
        .await
        .expect("bind grpc listener");
    let addr = bound.addr();
    // No shutdown channel: the server runs until the test aborts `handle`,
    // exactly as before. `pending()` never resolves, so shutdown is driven
    // solely by the abort — and the listener is held live throughout.
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(std::future::pending::<()>())
            .await
            .expect("flight + catalog server");
    });

    (addr, dir, handle)
}

async fn bind_tenant_via_session_service(addr: SocketAddr, session_id: &str, tenant: &str) {
    let mut client =
        CatalogServiceClient::with_interceptor(channel(addr).await, with_session(session_id));
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

    let count_a = flight_count_notes(addr, "session-a").await;
    let count_b = flight_count_notes(addr, "session-b").await;

    assert_eq!(count_a, 6, "session A's Flight SQL view must yield 6 rows");
    assert_eq!(count_b, 4, "session B's Flight SQL view must yield 4 rows");

    handle.abort();
}

/// Two tenants querying at once each see only their own rows, on every one of
/// many interleaved queries: a request's tenant lives in that request alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flight_concurrent_tenants_each_see_only_their_rows() {
    let (addr, _dir, handle) = start_flight_test_server().await;
    bind_tenant_via_session_service(addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(addr, "session-b", TENANT_B).await;

    const QUERIES: usize = 40;
    let queries = |session: &'static str| async move {
        let mut counts = Vec::with_capacity(QUERIES);
        for _ in 0..QUERIES {
            counts.push(flight_count_notes(addr, session).await);
        }
        counts
    };
    let workers: Vec<_> = (0..4)
        .map(|i| {
            let session = if i % 2 == 0 { "session-a" } else { "session-b" };
            (session, tokio::spawn(queries(session)))
        })
        .collect();
    for (session, worker) in workers {
        let want = if session == "session-a" { 6 } else { 4 };
        let counts = worker.await.expect("worker");
        assert!(
            counts.iter().all(|&n| n == want),
            "{session} must count {want} on every query, counted {counts:?}"
        );
    }

    handle.abort();
}

/// A session that names no tenant reads only global rows, after another
/// tenant's query as before it: nothing a scoped request did is left for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_a_session_naming_no_tenant_stays_global_after_another_tenants_query() {
    let (addr, _dir, handle) = start_flight_test_server().await;
    bind_tenant_via_session_service(addr, "session-a", TENANT_A).await;

    assert_eq!(flight_count_notes(addr, "session-unbound").await, 0);
    assert_eq!(flight_count_notes(addr, "session-a").await, 6);
    assert_eq!(
        flight_count_notes(addr, "session-unbound").await,
        0,
        "a session with no tenant must not inherit tenant A's scope"
    );

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
        CatalogServiceClient::with_interceptor(channel(addr).await, with_session("session-c"));
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

    let mut client = CatalogServiceClient::with_interceptor(
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
