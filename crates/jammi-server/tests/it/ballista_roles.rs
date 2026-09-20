//! `[ballista]` role hosting on `OssServer`:
//! unset = no Ballista listener; set = both roles bind, the executor
//! registers, and DRAIN stops both roles within the grace.

use std::net::TcpListener as StdTcpListener;
use std::time::Duration;

use jammi_db::config::{
    BallistaClientConfig, BallistaExecutorConfig, BallistaSchedulerConfig, JammiConfig,
};
use jammi_server::runtime::OssServer;

/// A free localhost port, probed then released — the same "probe, then
/// release" pattern `jammi_ballista::roles::host_executor` uses for an
/// unset `grpc_bind`, applied here because a single process hosting BOTH
/// roles must fix `scheduler.bind` to a KNOWN port before construction (the
/// executor's `scheduler_address` is parsed at the same config-build time,
/// before the scheduler has bound anything real).
fn free_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

fn config_with_ballista(artifact_dir: &std::path::Path, scheduler_port: u16) -> JammiConfig {
    let mut cfg = jammi_test_utils::test_config(artifact_dir);
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    cfg.ballista.scheduler = Some(BallistaSchedulerConfig {
        bind: format!("127.0.0.1:{scheduler_port}"),
        advertise_host: None,
    });
    cfg.ballista.executor = Some(BallistaExecutorConfig {
        scheduler_address: format!("127.0.0.1:{scheduler_port}"),
        bind: "127.0.0.1:0".into(),
        grpc_bind: "127.0.0.1:0".into(),
        advertise_host: Some("127.0.0.1".into()),
        work_dir: None,
        task_slots: 1,
    });
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ballista_roles_bind_executor_registers_and_drain_stops_both() {
    let dir = tempfile::tempdir().expect("tempdir");
    let scheduler_port = free_port();
    let cfg = config_with_ballista(dir.path(), scheduler_port);

    let server = OssServer::new(cfg).await.expect("oss server");
    let bound = server.bind().await.expect("bind all listeners");

    let scheduler_addr = bound
        .scheduler_addr()
        .expect("[ballista.scheduler] was set");
    assert_eq!(scheduler_addr.port(), scheduler_port);
    let (flight_addr, grpc_addr) = bound.executor_addrs().expect("[ballista.executor] was set");
    assert_ne!(
        flight_addr.port(),
        0,
        "the real bound port, not the :0 request"
    );
    assert_ne!(grpc_addr.port(), 0);

    // Registration is not independently observable over gRPC with
    // `ballista-scheduler`'s `rest-api` off (by design: it errors on a
    // restarted scheduler's graph-less status rows); a
    // successful bind is this oracle's determinant, matching
    // `jammi-ballista`'s own hermetic `roles::scheduler_and_executor_host_
    // in_one_process_and_submit_round_trips`, which exercises the stronger
    // "a submitted plan actually completes" property this crate's own
    // scope does not re-test.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("oss server serve")
    });

    // DRAIN (via `serve_with_shutdown`'s single-signal form): both roles
    // stop within the grace.
    let _ = shutdown_tx.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("serve_with_shutdown (DRAIN) returns within 10s")
        .expect("serve task did not panic");

    // Both ports are released: a fresh bind on the exact addresses succeeds.
    assert!(
        std::net::TcpListener::bind(scheduler_addr).is_ok(),
        "the scheduler listener must be closed after DRAIN"
    );
    assert!(
        std::net::TcpListener::bind(flight_addr).is_ok(),
        "the executor flight listener must be closed after DRAIN"
    );
}

#[tokio::test]
async fn unset_ballista_config_has_no_ballista_listener() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    // `cfg.ballista` is left at its default (`hosts_scheduler()` /
    // `hosts_executor()` both `false`) — the config-level property
    // `jammi-ballista`'s own `roles::unset_ballista_config_hosts_no_roles`
    // exercises directly; this asserts the SERVER-level consequence:
    // `OssServer::bind` never calls `host_scheduler`/`host_executor` at all.
    let server = OssServer::new(cfg).await.expect("oss server");
    let session = server.session();
    let bound = server.bind().await.expect("bind all listeners");
    assert!(bound.scheduler_addr().is_none());
    assert!(bound.executor_addrs().is_none());
    assert!(
        session.compute_plane().plane().is_none(),
        "no client role: every materialization and every claimed gang runs in this process"
    );
}

/// `[ballista.client]` alone makes a process a client: `bind` installs the
/// session's compute plane over the named scheduler and opens no Ballista
/// listener of its own. The scheduler need not be up: the role dials at
/// the first submission, so a query tier comes up before its scheduler.
#[tokio::test]
async fn client_role_installs_the_compute_plane_and_binds_no_ballista_listener() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    cfg.ballista.client = Some(BallistaClientConfig {
        scheduler_address: format!("127.0.0.1:{}", free_port()),
    });
    let server = OssServer::new(cfg).await.expect("oss server");
    let session = server.session();
    let bound = server.bind().await.expect("bind all listeners");
    assert!(bound.scheduler_addr().is_none());
    assert!(bound.executor_addrs().is_none());
    assert!(
        session.compute_plane().plane().is_some(),
        "the client role installs the compute plane at bind"
    );
}

/// A `[ballista]` address that collides with a FIXED `[server]` listener is
/// refused at `OssServer::new` — `BallistaConfig::validate`'s cross-section
/// check, called from the `new` constructor a struct-literal `JammiConfig`
/// (this fixture's own shape) always reaches, unlike `JammiConfig::
/// load_from`'s copy of the same call.
#[tokio::test]
async fn colliding_ballista_address_is_refused_at_oss_server_new() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.server.health_listen = "127.0.0.1:19171".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    // Collides with `server.health_listen` above — a FIXED-port clash.
    cfg.ballista.scheduler = Some(BallistaSchedulerConfig {
        bind: "127.0.0.1:19171".into(),
        advertise_host: None,
    });

    let err = match OssServer::new(cfg).await {
        Ok(_) => panic!("a fixed-port collision between [ballista] and [server] must be refused"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("scheduler.bind") && msg.contains("health_listen"),
        "the refusal must name both colliding keys: {msg}"
    );
}
