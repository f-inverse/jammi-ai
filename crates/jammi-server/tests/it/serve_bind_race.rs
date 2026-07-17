//! TOCTOU oracle for the serve seam's eager-bind handoff.
//!
//! The in-process server fixtures used to bind an ephemeral port, read it, DROP
//! the listener, then hand the bare port to a server that re-bound it later — a
//! release-then-rebind window a concurrent `cargo test` process could steal,
//! failing the client with a transport error. The eager-bind seam
//! ([`assemble_grpc_chain`] → [`jammi_server::runtime::AssembledChain::bind`] →
//! `BoundChain`) closes it: the listener is held live from the moment the port
//! is resolved through serving on it.
//!
//! This oracle pins the invariant DIRECTLY. With the port resolved off a live
//! `BoundChain`, an independent bind of that same port must fail with
//! `AddrInUse` at every point — before AND during serve
//! ([`bound_chain_holds_the_port_from_bind_through_serve`]). The companion
//! [`old_drop_then_rebind_shape_leaves_the_port_bindable`] reconstructs the
//! pre-fix drop-then-return shape and shows the window WAS open (the interposing
//! bind succeeds) — the RED the fix turns GREEN.

use std::io::ErrorKind;
use std::sync::Arc;

use jammi_db::session::JammiSession;
use jammi_server::grpc::session::{SessionIdTenantResolver, SessionStore};
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{assemble_grpc_chain, GrpcChain};
use jammi_server::tiers::TierSet;
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::sync::oneshot;

/// Build a transport-only chain (Flight SQL + the control-plane `CatalogService`,
/// no engine) at the loopback ephemeral address — the minimal serve surface,
/// enough to exercise the bind handoff.
async fn transport_only_chain() -> (GrpcChain, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = Arc::new(
        JammiSession::new(test_config(dir.path()))
            .await
            .expect("session"),
    );
    let store = SessionStore::new();
    let chain = GrpcChain {
        addr: "127.0.0.1:0".parse().expect("loopback :0 parses"),
        flight_ctx: session.context().clone(),
        flight_binding: session.tenant_binding_arc(),
        store: store.clone(),
        trigger: None,
        engine: None,
        tiers: TierSet::resolve(std::iter::empty()).expect("core-only tier set resolves"),
        metrics: Arc::new(MetricsRegistry::new().unwrap()),
        tenant_resolver: SessionIdTenantResolver::arc(store),
    };
    (chain, dir)
}

/// GREEN: the eager-bind seam holds the port continuously. Once `BoundChain`
/// reports its address, that port is NOT observably free — an independent bind
/// must fail with `AddrInUse` both before serving and while serving. This is the
/// exact window the old drop-then-rebind shape left open (see the RED control
/// below), now closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_chain_holds_the_port_from_bind_through_serve() {
    let (chain, _dir) = transport_only_chain().await;
    let bound = assemble_grpc_chain(chain)
        .expect("assemble")
        .bind()
        .await
        .expect("bind");
    let addr = bound.addr();

    // (1) Held BEFORE serve: the harness has resolved the port but has not begun
    // serving. An independent bind must ALREADY fail — the listener is live, with
    // no release window between resolving the port and serving on it.
    let err = std::net::TcpListener::bind(addr)
        .expect_err("the port must be held by BoundChain before serve — no release window");
    assert_eq!(
        err.kind(),
        ErrorKind::AddrInUse,
        "an interposing bind before serve must fail with AddrInUse, got {err:?}"
    );

    // (2) Held DURING serve: hand the bound chain to its serve loop. The SAME
    // listener carries through into `serve_with_incoming_shutdown`, so the port
    // stays held.
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });

    let err = std::net::TcpListener::bind(addr)
        .expect_err("the port must stay held while BoundChain serves");
    assert_eq!(
        err.kind(),
        ErrorKind::AddrInUse,
        "an interposing bind during serve must fail with AddrInUse, got {err:?}"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// RED control: the pre-fix shape — bind `:0`, read the port, DROP the listener,
/// then return the bare port — leaves the port OBSERVABLY FREE. An interposing
/// bind of the just-released port SUCCEEDS, which is precisely the window a
/// concurrent test process exploited before a server could re-bind it. This is
/// the RED the eager-bind seam turns GREEN: the identical interposing bind fails
/// with `AddrInUse` once the listener is held live
/// ([`bound_chain_holds_the_port_from_bind_through_serve`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn old_drop_then_rebind_shape_leaves_the_port_bindable() {
    // Reconstruct the pre-fix handoff: bind, read the addr, DROP the listener.
    let released_addr = {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        drop(listener);
        addr
    };

    // The window is open: the just-released port binds again. Under `cargo test`'s
    // parallel test binaries, THIS is where a different process could steal the
    // port between the harness releasing it and the server re-binding it.
    let interposed = std::net::TcpListener::bind(released_addr)
        .expect("the dropped port is observably free — this IS the TOCTOU window the fix closes");
    drop(interposed);
}
