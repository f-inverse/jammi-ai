//! Flight SQL `jammi-session-id`-bound tenant isolation.
//!
//! Co-mounted with `CatalogService` on one Tonic server. A client binds a
//! tenant via the gRPC `CatalogService.SetTenant` call carrying a
//! `jammi-session-id` header; the same header on a Flight SQL query resolves
//! to that tenant through the tenant-binding layer, which runs the request —
//! its planning and every poll of its result stream — inside that tenant's
//! own scope.
//!
//! A request's scope is its own, never a value requests share, so the cases
//! below run tenants concurrently (#738): reads, and the statements whose work
//! happens while their result streams (`INSERT`, `UPDATE`, `DELETE`,
//! `CREATE TABLE … AS`, `DROP TABLE`), each run as their own request's tenant,
//! and a caller that binds no tenant stays global whatever another tenant ran
//! before it.

use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::session::JammiSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::TenantId;
use jammi_server::grpc::proto::catalog as wire;
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};
use jammi_server::grpc::session::{SessionStore, SESSION_HEADER};
use jammi_test_utils::{fixture_url, test_config};
use tempfile::TempDir;
use tokio::task::JoinSet;
use tonic::transport::Channel;

use super::common::grpc::{
    channel, start_engine_server, tenant_a, tenant_b, with_session, TENANT_A, TENANT_B,
};

/// A third fixture tenant, for the cases that need a bystander.
const TENANT_C: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9c";

/// The Flight SQL + `CatalogService` server [`start_flight_test_server`]
/// starts, and the engine session it serves — so a test can read back, in
/// process and under each tenant's own scope, what a statement wrote.
struct FlightTestServer {
    addr: SocketAddr,
    session: Arc<JammiSession>,
    _dir: TempDir,
    handle: tokio::task::JoinHandle<()>,
}

/// Spin up the Flight SQL + SessionService server in-process. The fixture
/// pre-seeds a Parquet local source with 10 rows split 6 (tenant A) + 4
/// (tenant B) and registers the source under the engine session, declaring
/// `tenant_id` as the federated tenant column. The two clients we attach
/// later read from this one source through the analyzer rule. It also creates
/// the mutable table `widgets (id, v)` — once, unscoped, so every tenant
/// writes the same backing table and only the tenant stamp separates its rows.
async fn start_flight_test_server() -> FlightTestServer {
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
    let widgets = MutableTableDefinitionBuilder::new(
        MutableTableId::new("widgets").unwrap(),
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
        ])),
    )
    .primary_key(vec!["id".into()])
    .build()
    .unwrap();
    session
        .create_mutable_table(widgets)
        .await
        .expect("create the widgets table");

    // Flight SQL + the control-plane `CatalogService` on one listener — the same
    // surface the multi-tenant Flight shape serves (Flight bound through the
    // `TenantBoundProvider`, `CatalogService.SetTenant` binding the tenant via
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

    FlightTestServer {
        addr,
        session,
        _dir: dir,
        handle,
    }
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
    let FlightTestServer {
        addr, _dir, handle, ..
    } = start_flight_test_server().await;

    bind_tenant_via_session_service(addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(addr, "session-b", TENANT_B).await;

    let count_a = flight_count_notes(addr, "session-a").await;
    let count_b = flight_count_notes(addr, "session-b").await;

    assert_eq!(count_a, 6, "session A's Flight SQL view must yield 6 rows");
    assert_eq!(count_b, 4, "session B's Flight SQL view must yield 4 rows");

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_clearing_tenant_makes_query_unscoped() {
    let FlightTestServer {
        addr, _dir, handle, ..
    } = start_flight_test_server().await;

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
    let FlightTestServer {
        addr, _dir, handle, ..
    } = start_flight_test_server().await;

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

/// A Flight SQL client whose every call carries `session_id`, so the server
/// resolves the tenant that session bound.
async fn flight_client(addr: SocketAddr, session_id: &str) -> FlightSqlServiceClient<Channel> {
    let mut client = FlightSqlServiceClient::new(channel(addr).await);
    client.set_header(SESSION_HEADER, session_id);
    client
}

/// Run one statement over Flight SQL — `execute` for the ticket, then
/// `do_get` — and drain its rows. A statement does its work while its rows
/// stream, so it has run, and its effects are visible, once this returns.
async fn flight_sql(
    client: &mut FlightSqlServiceClient<Channel>,
    sql: &str,
) -> Result<Vec<RecordBatch>, String> {
    let info = client
        .execute(sql.to_string(), None)
        .await
        .map_err(|e| format!("execute `{sql}`: {e}"))?;
    let ticket = info
        .endpoint
        .first()
        .and_then(|e| e.ticket.clone())
        .ok_or_else(|| format!("`{sql}`: the flight info carries no ticket"))?;
    client
        .do_get(ticket)
        .await
        .map_err(|e| format!("do_get `{sql}`: {e}"))?
        .try_collect()
        .await
        .map_err(|e| format!("stream `{sql}`: {e}"))
}

/// `COUNT(*)` over the notes source, as `client`'s tenant sees it.
async fn count_notes(client: &mut FlightSqlServiceClient<Channel>) -> i64 {
    flight_sql(client, "SELECT COUNT(*) AS n FROM notes.public.notes")
        .await
        .expect("count notes")
        .iter()
        .map(|batch| {
            let counts = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 count column");
            counts.iter().flatten().sum::<i64>()
        })
        .sum()
}

/// The `(id, v)` rows of `widgets` that `tenant` sees, by id, read in
/// process under the tenant's own scope: its own rows and the global ones.
async fn widgets_seen_by(session: &JammiSession, tenant: TenantId) -> Vec<(i64, Option<i64>)> {
    let batches = session
        .with_tenant_scoped(tenant, |scope| async move {
            scope
                .sql("SELECT id, v FROM mutable.public.widgets ORDER BY id")
                .await
        })
        .await
        .expect("read widgets");
    let mut rows = Vec::new();
    for batch in &batches {
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .unwrap_or_else(|| panic!("an Int64 `{name}` column"))
                .clone()
        };
        let (ids, vs) = (column("id"), column("v"));
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), (!vs.is_null(i)).then(|| vs.value(i))));
        }
    }
    rows
}

/// Await every task in `tasks`, failing the test on the first one that
/// panicked or returned an error.
async fn join_all(mut tasks: JoinSet<Result<(), String>>) {
    while let Some(joined) = tasks.join_next().await {
        joined
            .expect("the task ran to completion")
            .expect("its statements succeeded");
    }
}

/// Two tenants querying at the same time each see exactly their own rows,
/// every time (#738): a query runs under its own request's tenant, never
/// under a value the requests share.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_tenants_each_read_only_their_own_rows() {
    let server = start_flight_test_server().await;
    bind_tenant_via_session_service(server.addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(server.addr, "session-b", TENANT_B).await;

    let mut readers = JoinSet::new();
    for _ in 0..4 {
        for (session_id, expected) in [("session-a", 6), ("session-b", 4)] {
            let addr = server.addr;
            readers.spawn(async move {
                let mut client = flight_client(addr, session_id).await;
                let mut wrong = Vec::new();
                for _ in 0..25 {
                    let n = count_notes(&mut client).await;
                    if n != expected {
                        wrong.push(n);
                    }
                }
                (session_id, wrong)
            });
        }
    }
    while let Some(joined) = readers.join_next().await {
        let (session_id, wrong) = joined.expect("reader task");
        assert!(
            wrong.is_empty(),
            "{session_id} counted another tenant's rows {} times out of 25: {wrong:?}",
            wrong.len()
        );
    }

    server.handle.abort();
}

/// Concurrent `INSERT`s over Flight SQL are owned by the tenant that issued
/// them. The sink stamps a row's tenant while the statement's response
/// streams, after its handler has returned — so it must take that tenant
/// from the request, not from whatever was in force when the stream is
/// polled.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_inserts_are_owned_by_the_tenant_that_wrote_them() {
    let server = start_flight_test_server().await;
    bind_tenant_via_session_service(server.addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(server.addr, "session-b", TENANT_B).await;

    let mut writers = JoinSet::new();
    for task in 0..3_i64 {
        for (session_id, base) in [("session-a", 0_i64), ("session-b", 1000)] {
            let addr = server.addr;
            writers.spawn(async move {
                let mut client = flight_client(addr, session_id).await;
                for i in 0..20 {
                    let id = base + task * 20 + i;
                    flight_sql(
                        &mut client,
                        &format!("INSERT INTO mutable.public.widgets (id, v) VALUES ({id}, 0)"),
                    )
                    .await?;
                }
                Ok(())
            });
        }
    }
    join_all(writers).await;

    let ids =
        |rows: Vec<(i64, Option<i64>)>| rows.into_iter().map(|(id, _)| id).collect::<Vec<_>>();
    assert_eq!(
        ids(widgets_seen_by(&server.session, tenant_a()).await),
        (0..60).collect::<Vec<_>>(),
        "tenant A sees exactly the rows it inserted"
    );
    assert_eq!(
        ids(widgets_seen_by(&server.session, tenant_b()).await),
        (1000..1060).collect::<Vec<_>>(),
        "tenant B sees exactly the rows it inserted"
    );

    server.handle.abort();
}

/// Concurrent `UPDATE`s and `DELETE`s over Flight SQL rewrite only the
/// caller's own rows. A rewrite runs while the statement's response streams,
/// so — like an `INSERT` — it must run as the request's tenant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_updates_and_deletes_touch_only_the_callers_rows() {
    let server = start_flight_test_server().await;
    bind_tenant_via_session_service(server.addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(server.addr, "session-b", TENANT_B).await;

    // Twenty rows per tenant, seeded in process under each tenant's own scope.
    for (tenant, base) in [(tenant_a(), 0_i64), (tenant_b(), 1000)] {
        let values = (base..base + 20)
            .map(|id| format!("({id}, 0)"))
            .collect::<Vec<_>>()
            .join(", ");
        server
            .session
            .with_tenant_scoped(tenant, move |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.widgets (id, v) VALUES {values}"
                    ))
                    .await
            })
            .await
            .expect("seed the tenant's rows");
    }

    // Each tenant sets `v` on all of its rows and deletes the even ones, from
    // two tasks over disjoint ids.
    let mut writers = JoinSet::new();
    for (session_id, base, v) in [("session-a", 0_i64, 1_i64), ("session-b", 1000, 2)] {
        for half in 0..2_i64 {
            let addr = server.addr;
            writers.spawn(async move {
                let mut client = flight_client(addr, session_id).await;
                for id in (base + half * 10)..(base + half * 10 + 10) {
                    flight_sql(
                        &mut client,
                        &format!("UPDATE mutable.public.widgets SET v = {v} WHERE id = {id}"),
                    )
                    .await?;
                    if id % 2 == 0 {
                        flight_sql(
                            &mut client,
                            &format!("DELETE FROM mutable.public.widgets WHERE id = {id}"),
                        )
                        .await?;
                    }
                }
                Ok(())
            });
        }
    }
    join_all(writers).await;

    let survivors = |base: i64, v: i64| {
        (base..base + 20)
            .filter(|id| id % 2 == 1)
            .map(|id| (id, Some(v)))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        widgets_seen_by(&server.session, tenant_a()).await,
        survivors(0, 1),
        "tenant A's odd rows survive with its update, and nothing else is visible to it"
    );
    assert_eq!(
        widgets_seen_by(&server.session, tenant_b()).await,
        survivors(1000, 2),
        "tenant B's odd rows survive with its update, and nothing else is visible to it"
    );

    server.handle.abort();
}

/// The catalog row of the result table `name`, read under the admin scope so
/// it is found whichever tenant owns it.
async fn result_table_row(engine: &InferenceSession, name: &str) -> Option<ResultTableRecord> {
    engine
        .with_admin_scope(|_admin| async move { engine.catalog().get_result_table(name).await })
        .await
        .expect("read the result table row")
}

/// Concurrent `CREATE TABLE … AS` and `DROP TABLE` over Flight SQL run as the
/// tenant that issued them. Both execute while the statement's response
/// streams, so each must run as its request's tenant: a table is created
/// owned by its creator, and dropped by it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_create_table_as_and_drop_table_run_as_the_caller() {
    let server = start_engine_server().await;
    bind_tenant_via_session_service(server.addr, "session-a", TENANT_A).await;
    bind_tenant_via_session_service(server.addr, "session-b", TENANT_B).await;
    let callers = [("session-a", "a", TENANT_A), ("session-b", "b", TENANT_B)];
    let names = |prefix: &str| (0..6).map(|i| format!("{prefix}_t{i}")).collect::<Vec<_>>();

    let mut creators = JoinSet::new();
    for (session_id, prefix, _) in callers {
        let (addr, tables) = (server.addr, names(prefix));
        creators.spawn(async move {
            let mut client = flight_client(addr, session_id).await;
            for (i, name) in tables.iter().enumerate() {
                flight_sql(
                    &mut client,
                    &format!("CREATE TABLE {name} AS SELECT {i} AS id"),
                )
                .await?;
            }
            Ok(())
        });
    }
    join_all(creators).await;

    for (_, prefix, owner) in callers {
        for name in names(prefix) {
            let row = result_table_row(&server.engine, &name)
                .await
                .unwrap_or_else(|| panic!("{name} was created"));
            assert_eq!(
                row.tenant_id.as_deref(),
                Some(owner),
                "{name} is owned by the tenant that created it"
            );
        }
    }

    let mut droppers = JoinSet::new();
    for (session_id, prefix, _) in callers {
        let (addr, tables) = (server.addr, names(prefix));
        droppers.spawn(async move {
            let mut client = flight_client(addr, session_id).await;
            for name in &tables {
                flight_sql(&mut client, &format!("DROP TABLE {name}")).await?;
            }
            Ok(())
        });
    }
    join_all(droppers).await;

    for (_, prefix, _) in callers {
        for name in names(prefix) {
            assert!(
                result_table_row(&server.engine, &name).await.is_none(),
                "{name} was dropped by its owner"
            );
        }
    }

    let _ = server.shutdown.send(());
}

/// The source ids `client`'s caller can list.
async fn listed_sources<F>(
    client: &mut CatalogServiceClient<tonic::service::interceptor::InterceptedService<Channel, F>>,
) -> Vec<String>
where
    F: tonic::service::Interceptor,
{
    client
        .list_sources(wire::ListSourcesRequest {})
        .await
        .expect("list_sources")
        .into_inner()
        .sources
        .into_iter()
        .map(|s| s.source_id)
        .collect()
}

/// An `AddSource` request for a CSV source `source_id`, with no tenant column.
fn csv_source(source_id: &str) -> wire::AddSourceRequest {
    wire::AddSourceRequest {
        source_id: source_id.into(),
        source_kind: wire::SourceKind::File as i32,
        connection: Some(wire::SourceConnection {
            url: fixture_url("training_pairs.csv"),
            format: wire::FileFormat::Csv as i32,
            tenant_column: None,
        }),
    }
}

/// A caller that bound no tenant stays global, whatever another tenant ran
/// over Flight SQL before it (#738): it neither lists that tenant's private
/// sources nor registers its own sources as that tenant's. A global source
/// it registers is visible to every tenant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unscoped_caller_stays_global_after_a_tenant_flight_query() {
    let server = start_engine_server().await;
    bind_tenant_via_session_service(server.addr, "session-b", TENANT_B).await;
    bind_tenant_via_session_service(server.addr, "session-c", TENANT_C).await;
    // `session-u` binds nothing, so the resolver scopes it `Global`.
    let addr = server.addr;
    let catalog = |session_id: &'static str| async move {
        CatalogServiceClient::with_interceptor(channel(addr).await, with_session(session_id))
    };
    let (mut b, mut c, mut u) = (
        catalog("session-b").await,
        catalog("session-c").await,
        catalog("session-u").await,
    );

    b.add_source(csv_source("bpriv"))
        .await
        .expect("B adds a private source");
    assert!(
        !listed_sources(&mut u).await.contains(&"bpriv".to_string()),
        "the unscoped caller does not list B's private source"
    );

    let mut flight_b = flight_client(server.addr, "session-b").await;
    flight_sql(&mut flight_b, "SELECT 1 AS one")
        .await
        .expect("B queries over Flight SQL");
    assert!(
        !listed_sources(&mut u).await.contains(&"bpriv".to_string()),
        "after B's Flight SQL query the unscoped caller still does not list B's private source"
    );

    u.add_source(csv_source("u_global"))
        .await
        .expect("the unscoped caller adds a source");
    for (who, client) in [("U", &mut u), ("B", &mut b), ("C", &mut c)] {
        assert!(
            listed_sources(client)
                .await
                .contains(&"u_global".to_string()),
            "the unscoped caller's source is global, so {who} lists it"
        );
    }

    let _ = server.shutdown.send(());
}

/// The Flight service runs a request only inside the tenant scope the
/// tenant-binding layer resolved for it. Mounted without that layer, a query
/// is refused instead of running under whatever tenant happens to be ambient.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_without_the_tenant_binding_layer_refuses_queries() {
    use arrow_flight::flight_service_server::FlightServiceServer;
    use jammi_server::flight::{JammiFlightService, TenantBoundProvider};

    let dir = tempfile::tempdir().expect("tempdir");
    let session = JammiSession::new(test_config(dir.path()))
        .await
        .expect("session");
    let flight = FlightServiceServer::new(JammiFlightService::new(Arc::new(
        TenantBoundProvider::new(session.context().state()),
    )));
    let listener = tokio::net::TcpListener::bind(super::common::grpc::ephemeral_addr())
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(flight)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );

    let mut client = flight_client(addr, "session-a").await;
    let refused = flight_sql(&mut client, "SELECT 1 AS one")
        .await
        .expect_err("a request outside its tenant scope is refused");
    assert!(
        refused.contains("tenant-binding layer"),
        "refused for running outside the layer's scope: {refused}"
    );

    server.abort();
}
