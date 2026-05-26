//! SPEC-03 §12 #5 — gRPC `SessionService.SetTenant` + `TriggerService.ListTopics`
//! end-to-end isolation. An in-process server hosts both services behind the
//! shared [`SessionStore`] + [`TenantInterceptor`]. Two clients distinguished
//! by `jammi-session-id` headers bind different tenants and observe the
//! corresponding `ListTopics` filter; an unbound client sees only globally
//! scoped topics; an invalid UUID is rejected with `InvalidArgument`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use jammi_db::session::JammiSession;
use jammi_server::grpc::proto::session::session_service_client::SessionServiceClient;
use jammi_server::grpc::proto::session::{SetTenantRequest, Tenant};
use jammi_server::grpc::proto::trigger::trigger_service_client::TriggerServiceClient;
use jammi_server::grpc::proto::trigger::ListTopicsRequest;
use jammi_server::grpc::session::SessionStore;
use jammi_server::TriggerHandles;
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

use super::common::grpc::{channel, tenant_a, tenant_b, with_session, TENANT_A, TENANT_B};

/// Spin up an in-process gRPC server hosting `SessionService` + `TriggerService`
/// behind the shared interceptor. Pre-seeds two topics — one bound to
/// tenant A, one to tenant B — so `ListTopics` returns different results per
/// session. Returns `(addr, store, shutdown)` plus the `TempDir` guard that
/// keeps the catalog alive for the duration of the test.
async fn start_grpc_test_server() -> (
    SocketAddr,
    SessionStore,
    oneshot::Sender<()>,
    TempDir,
    tokio::task::JoinHandle<()>,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());

    // The server's catalog & engine state. Create the session, register one
    // topic per tenant via the SQL DDL surface. The catalog row carries the
    // tenant_id from the session binding at registration time.
    let session = Arc::new(JammiSession::new(cfg).await.expect("session"));

    // Topic bound to tenant A.
    session.bind_tenant(tenant_a());
    session
        .sql("CREATE TOPIC tenant_a_topic (id BIGINT NOT NULL)")
        .await
        .expect("create topic A");

    // Topic bound to tenant B.
    session.bind_tenant(tenant_b());
    session
        .sql("CREATE TOPIC tenant_b_topic (id BIGINT NOT NULL)")
        .await
        .expect("create topic B");

    // Restore unscoped binding so the trigger service uses the per-request
    // tenant from the interceptor, not the session's last value.
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

    // Give the server a moment to bind.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, store, shutdown_tx, dir, handle)
}

async fn set_tenant(
    channel: Channel,
    session_id: &str,
    tenant_id: &str,
) -> Result<(), tonic::Status> {
    let mut client = SessionServiceClient::with_interceptor(channel, with_session(session_id));
    client
        .set_tenant(SetTenantRequest {
            tenant: Some(Tenant {
                id: tenant_id.to_string(),
            }),
        })
        .await
        .map(|_| ())
}

async fn list_topics(channel: Channel, session_id: &str) -> Vec<String> {
    let mut client = TriggerServiceClient::with_interceptor(channel, with_session(session_id));
    let response = client
        .list_topics(ListTopicsRequest::default())
        .await
        .expect("list_topics");
    response
        .into_inner()
        .topics
        .into_iter()
        .map(|t| t.name)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_set_tenant_then_list_topics_filters_by_session() {
    let (addr, _store, shutdown, _dir, handle) = start_grpc_test_server().await;
    let channel = channel(addr).await;

    set_tenant(channel.clone(), "session-a", TENANT_A)
        .await
        .unwrap();
    let topics = list_topics(channel, "session-a").await;
    assert!(
        topics.contains(&"tenant_a_topic".to_string()),
        "session-a should see its own topic; got {topics:?}"
    );
    assert!(
        !topics.contains(&"tenant_b_topic".to_string()),
        "session-a must not see tenant B's topic; got {topics:?}"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_two_concurrent_sessions_are_isolated() {
    let (addr, _store, shutdown, _dir, handle) = start_grpc_test_server().await;
    let channel_a = channel(addr).await;
    let channel_b = channel(addr).await;

    set_tenant(channel_a.clone(), "session-a", TENANT_A)
        .await
        .unwrap();
    set_tenant(channel_b.clone(), "session-b", TENANT_B)
        .await
        .unwrap();

    let topics_a = list_topics(channel_a, "session-a").await;
    let topics_b = list_topics(channel_b, "session-b").await;

    assert!(topics_a.contains(&"tenant_a_topic".to_string()));
    assert!(!topics_a.contains(&"tenant_b_topic".to_string()));
    assert!(topics_b.contains(&"tenant_b_topic".to_string()));
    assert!(!topics_b.contains(&"tenant_a_topic".to_string()));

    let intersection: Vec<_> = topics_a.iter().filter(|t| topics_b.contains(t)).collect();
    assert!(
        intersection.is_empty(),
        "session topic lists must be disjoint; got intersection {intersection:?}"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_unset_session_sees_only_global_topics() {
    let (addr, _store, shutdown, _dir, handle) = start_grpc_test_server().await;
    let channel = channel(addr).await;

    // No SetTenant call. The interceptor sees no session and attaches a
    // SessionTenant(None) extension; the TriggerService filters to topics
    // whose tenant_id IS NULL. The fixture pre-seeds zero global topics,
    // so the response is empty.
    let topics = list_topics(channel, "session-c-unbound").await;
    assert!(
        topics.is_empty(),
        "unbound session must see no tenant-bound topics; got {topics:?}"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_set_tenant_rejects_invalid_uuid_typed() {
    let (addr, _store, shutdown, _dir, handle) = start_grpc_test_server().await;
    let channel = channel(addr).await;

    let err = set_tenant(channel, "session-bad", "not-a-uuid")
        .await
        .expect_err("expected InvalidArgument");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "expected InvalidArgument code, got {:?}: {}",
        err.code(),
        err.message()
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}
