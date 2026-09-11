//! gRPC test helpers shared between `grpc_session`, `grpc_trigger`, and
//! `flight_tenant`. Each of those files previously carried its own copy of
//! the `with_session` interceptor closure, the `channel(addr)` constructor,
//! and the two well-known tenant UUIDs we use as test fixtures — three
//! near-identical copies that violated CLAUDE.md §DRY. Centralising them
//! here keeps the three test surfaces in lockstep and gives new tests one
//! obvious place to plug into.

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::TenantId;
use jammi_server::grpc::session::{SessionStore, SESSION_HEADER};
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

/// Well-known tenant UUIDs used as fixtures across the gRPC integration
/// tests. These are generic UUIDs not coupled to any downstream tenant
/// (jammi is the substrate; accurisk/lace/etc. live in product crates).
pub const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
pub const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b";

/// Parse [`TENANT_A`] into a typed [`TenantId`]. Panics on programmer
/// error (the constant is a valid UUID by construction).
pub fn tenant_a() -> TenantId {
    TenantId::from_str(TENANT_A).expect("TENANT_A is a valid UUID")
}

/// Parse [`TENANT_B`] into a typed [`TenantId`].
pub fn tenant_b() -> TenantId {
    TenantId::from_str(TENANT_B).expect("TENANT_B is a valid UUID")
}

/// Build an HTTP/2 channel to an in-process Tonic server on `addr`. Used by
/// every gRPC test that needs to attach a client — the address is supplied
/// by the per-test fixture (typically backed by a `TcpListener` bound to
/// `127.0.0.1:0`).
pub async fn channel(addr: SocketAddr) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("channel uri")
        .connect()
        .await
        .expect("channel connect")
}

/// The loopback ephemeral-port config address (`127.0.0.1:0`) every in-process
/// fixture assembles its [`jammi_server::runtime::GrpcChain`] at. The real,
/// bindable port is resolved by [`spawn_bound_chain`] at bind time — the
/// fixture never picks a port itself.
pub fn ephemeral_addr() -> SocketAddr {
    "127.0.0.1:0".parse().expect("loopback :0 parses")
}

/// Assemble `chain`, bind its gRPC + Flight SQL listener EAGERLY, and spawn the
/// serve loop on the already-bound listener — returning the ACTUAL bound
/// address (the real ephemeral port) and the serve task's join handle.
///
/// The listener is held continuously from bind through serve: there is no
/// release-then-rebind window in which a concurrent `cargo test` process could
/// steal the port, which is exactly the flake this handoff exists to close. A
/// caller builds its client against the returned address; firing `shutdown_rx`
/// (or dropping its sender) tears the server down.
///
/// Every in-process gRPC fixture builds its own `GrpcChain` (its own tiers /
/// engine / trigger wiring) at [`ephemeral_addr`] and hands it here, so the
/// eager-bind handoff lives in exactly one place.
pub async fn spawn_bound_chain(
    chain: jammi_server::runtime::GrpcChain,
    shutdown_rx: oneshot::Receiver<()>,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let bound = jammi_server::runtime::assemble_grpc_chain(chain)
        .expect("assemble grpc chain")
        .bind()
        .await
        .expect("bind grpc listener");
    let addr = bound.addr();
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("grpc server");
    });
    (addr, handle)
}

/// A [`tokio::task::JoinHandle`] that ABORTS its task on drop rather than
/// merely detaching it (`JoinHandle::drop`'s own documented behaviour). A
/// `Future` in its own right (delegates `poll` to the inner handle), so it
/// slots into `handle.await` and `tokio::time::timeout(dur, handle)`
/// call sites unchanged — only construction (wrapping the raw
/// `JoinHandle`) differs from those call sites' point of view.
///
/// Exists because `shutdown`'s own drop (a field ahead of `handle` in
/// [`EngineServer`]'s declaration order, dropped first) only REQUESTS
/// `serve_with_shutdown` wind down gracefully — it does not guarantee the
/// spawned chain task is gone by the time a caller that never reaches its own
/// explicit `handle.await` (e.g. a test that panics first) moves on. For a
/// broker backed by a live external connection (e.g.
/// `start_engine_server_with_broker`'s `[broker.postgres]`, held via
/// `PostgresBroker`'s own LISTEN connection), that gap leaks a live
/// connection past the guard's scope. Aborting on drop is unconditional and
/// safe either way: a handle that already finished (the explicit-shutdown
/// path every passing test takes) makes `abort` a documented no-op.
pub struct AbortOnDropHandle<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDropHandle<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl<T> std::future::Future for AbortOnDropHandle<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.0).poll(cx)
    }
}

/// Guards that keep an in-process engine-backed gRPC server (and its catalog)
/// alive for the duration of a test. Dropping `shutdown` or letting it fall out
/// of scope tears the server down; `_dir` roots the engine's temp artifact dir.
pub struct EngineServer {
    pub addr: SocketAddr,
    pub shutdown: oneshot::Sender<()>,
    /// RAII guard: roots the engine's temp artifact dir for the server's
    /// lifetime and deletes it on drop. Held, never read.
    pub _dir: TempDir,
    pub handle: AbortOnDropHandle<()>,
    /// The same `Arc<InferenceSession>` the server task drives. Shared so a
    /// test can wrap it in a local `Session` and assert the data-plane client
    /// over the wire returns identical results / errors against the *same*
    /// engine.
    pub engine: Arc<InferenceSession>,
    /// The SAME metrics registry the server's `MetricsLayer` / `crate::limits`
    /// refusal stack drive — a test asserts `jammi_grpc_refused_total{reason}`
    /// against this handle rather than scraping an HTTP `/metrics` route (this
    /// fixture runs no health side-channel).
    pub metrics: Arc<jammi_server::routes::health::MetricsRegistry>,
}

impl EngineServer {
    /// Start a training worker over the SAME engine session this server drives,
    /// returning the RAII guard that owns its loop (dropping the guard stops it).
    ///
    /// The release valve for [`start_engine_server_worker_quiesced`]: a test
    /// reads a submitted job's pre-claim state while nothing can claim it, then
    /// calls this to let the job actually run, and awaits its terminal state
    /// through the public surface. A worker is a worker regardless of who
    /// spawned it — this is the identical `EmbeddedWorker` a `[worker] enabled`
    /// server starts, over the identical session — so releasing here restores
    /// the production shape rather than simulating it.
    pub fn spawn_worker(&self) -> jammi_ai::fine_tune::worker::EmbeddedWorker {
        jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&self.engine)
            .expect("the test config's worker intervals are valid")
    }
}

/// Spin up an in-process gRPC server hosting the chain *with* the engine-backed
/// services, mounting every tier **except** the event tier (no trigger
/// handles). Shared by the `grpc_inference`, `grpc_eval`,
/// `grpc_introspection`, and `grpc_training` suites so they drive the same
/// wiring the embedding suite does.
pub async fn start_engine_server() -> EngineServer {
    start_engine_server_with_tiers(non_event_tiers()).await
}

/// Every optional tier except event — the engine-backed serve + eval surface
/// (job submission is core), without the trigger stream. The tier set
/// [`start_engine_server`] and [`start_engine_server_worker_quiesced`] share,
/// so the two fixtures mount the identical surface.
fn non_event_tiers() -> jammi_server::tiers::TierSet {
    let optional = jammi_server::tiers::ServiceTier::OPTIONAL
        .into_iter()
        .filter(|t| *t != jammi_server::tiers::ServiceTier::Event);
    jammi_server::tiers::TierSet::resolve(optional)
}

/// Like [`start_engine_server`] but also mounts the trigger handles (the event
/// tier), so the `TriggerService` (topics / publish / subscribe) is reachable
/// over the wire. Shared by the data-plane client topic/subscribe/audit parity
/// tests, which drive those surfaces against the same engine a local `Session`
/// wraps.
pub async fn start_engine_server_with_trigger() -> EngineServer {
    start_engine_server_with_tiers(jammi_server::tiers::TierSet::all()).await
}

/// Spin up an in-process engine-backed gRPC server mounting exactly `tiers`.
/// The trigger handles (event tier) are derived from `tiers.contains(Event)`,
/// so what is mounted and what `GetServerInfo` advertises are one decision —
/// no way to construct a fixture whose handshake lies about its mount set.
/// Used by the tier-gating tests to stand up serve-only / serve+train / etc.
pub async fn start_engine_server_with_tiers(tiers: jammi_server::tiers::TierSet) -> EngineServer {
    // Assemble the chain at the loopback ephemeral address and hand it to
    // `spawn_bound_chain`, which binds the listener EAGERLY and serves on it —
    // the port is held from bind through serve, so `addr` names a port no
    // concurrent test process can have stolen.
    let (chain, engine, dir) = engine_chain_at(ephemeral_addr(), tiers).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// An [`jammi_server::grpc::catalog::AdminAuthorizer`] that permits every
/// `all = true` [`Reconcile`](jammi_server::grpc::proto::catalog::ReconcileRequest)
/// pass unconditionally — the test double the K4 embedded/remote parity
/// oracle and the denied-by-default oracle's positive arm wire onto a chain
/// via [`start_engine_server_with_admin`]. Never reached in production: the
/// shipped default is `None`, which every OTHER fixture in this module gets.
pub struct AllowAllAdmin;

impl jammi_server::grpc::catalog::AdminAuthorizer for AllowAllAdmin {
    fn authorize(&self, _metadata: &tonic::metadata::MetadataMap) -> Result<(), tonic::Status> {
        Ok(())
    }
}

/// Like [`start_engine_server_with_tiers`], but with an explicit
/// [`jammi_server::grpc::catalog::AdminAuthorizer`] wired onto the chain, so a
/// test can exercise `Reconcile`'s `all = true` cross-tenant admin pass (or
/// its refusal) without every other fixture in this module having to carry
/// the same seam.
pub async fn start_engine_server_with_admin(
    tiers: jammi_server::tiers::TierSet,
    admin_authorizer: Option<Arc<dyn jammi_server::grpc::catalog::AdminAuthorizer>>,
) -> EngineServer {
    let (chain, engine, dir) =
        engine_chain_at_with_admin(ephemeral_addr(), tiers, admin_authorizer).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// Build the engine-backed [`jammi_server::runtime::GrpcChain`] the fixtures
/// above and below assemble: a fresh engine session over a temp artifact dir,
/// mounting exactly `tiers`, addressed at `addr`. Returns the chain plus the
/// shared engine handle and the `TempDir` that roots its artifacts.
///
/// The single expression of that wiring, so the eager-bind fixture
/// ([`start_engine_server_with_tiers`]) and the worker-quiesced fixture
/// ([`start_engine_server_worker_quiesced`]) cannot drift into serving
/// different surfaces.
async fn engine_chain_at(
    addr: SocketAddr,
    tiers: jammi_server::tiers::TierSet,
) -> (
    jammi_server::runtime::GrpcChain,
    Arc<InferenceSession>,
    TempDir,
) {
    engine_chain_at_with_admin(addr, tiers, None).await
}

/// Like [`engine_chain_at`], but with an explicit
/// [`jammi_server::grpc::catalog::AdminAuthorizer`] wired onto the chain —
/// the seam `Reconcile`'s `all = true` cross-tenant admin pass gates on. The
/// shipped default every other fixture gets is `None` (refuses `all = true`);
/// tests exercising the admin pass (the K4 parity oracle, the denied-by-default
/// oracle) pass `Some(Arc::new(AllowAllAdmin))` here.
async fn engine_chain_at_with_admin(
    addr: SocketAddr,
    tiers: jammi_server::tiers::TierSet,
    admin_authorizer: Option<Arc<dyn jammi_server::grpc::catalog::AdminAuthorizer>>,
) -> (
    jammi_server::runtime::GrpcChain,
    Arc<InferenceSession>,
    TempDir,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let (chain, engine) = engine_chain_from_config(addr, tiers, cfg, admin_authorizer).await;
    (chain, engine, dir)
}

/// The chain-building half of [`engine_chain_at`], parameterised on the
/// `JammiConfig` the engine session is opened with — so a fixture can vary a
/// CONFIG KEY (not a construction seam) and still serve the identical surface
/// every other fixture serves. Returns the chain plus the shared engine handle;
/// the caller owns whatever roots the config's `artifact_dir`.
async fn engine_chain_from_config(
    addr: SocketAddr,
    tiers: jammi_server::tiers::TierSet,
    cfg: jammi_db::config::JammiConfig,
    admin_authorizer: Option<Arc<dyn jammi_server::grpc::catalog::AdminAuthorizer>>,
) -> (jammi_server::runtime::GrpcChain, Arc<InferenceSession>) {
    // `open` (not `new`) so the engine-backed server registers the compound
    // query SQL functions (`annotate`, …) on its context — the same shape the
    // production `OssServer` builds, and what the Flight SQL `annotate` test
    // exercises.
    let session = InferenceSession::open(cfg).await.expect("session");
    chain_over_session(addr, tiers, session, admin_authorizer)
}

/// The chain-building half of [`engine_chain_from_config`] over an ALREADY
/// OPEN engine session — so a test that opened its session itself (e.g. with
/// [`InferenceSession::open_with_placement`]) serves the identical surface
/// every other fixture serves. Returns the chain plus the shared engine handle.
pub fn chain_over_session(
    addr: SocketAddr,
    tiers: jammi_server::tiers::TierSet,
    session: Arc<InferenceSession>,
    admin_authorizer: Option<Arc<dyn jammi_server::grpc::catalog::AdminAuthorizer>>,
) -> (jammi_server::runtime::GrpcChain, Arc<InferenceSession>) {
    let store = SessionStore::new();
    let trigger = tiers
        .contains(jammi_server::tiers::ServiceTier::Event)
        .then(|| jammi_server::TriggerHandles {
            topic_repo: session.topic_repo(),
            publisher: session.publisher(),
            subscriber: session.subscriber(),
        });

    let engine = Arc::clone(&session);
    // Same source production reads (`OssServer::build_grpc_chain`): the
    // config the session was actually opened with, so a test that varies
    // `[server.limits]` through `test_config`/its own override sees that
    // value reach the wire, not a hardcoded default.
    let limits = session.inner_config().server.limits;
    let chain = jammi_server::runtime::GrpcChain {
        addr,
        flight_ctx: session.context().clone(),
        flight_binding: session.tenant_binding_arc(),
        store: store.clone(),
        trigger,
        engine: Some(session),
        tiers,
        metrics: Arc::new(jammi_server::routes::health::MetricsRegistry::new().unwrap()),
        tenant_resolver: jammi_server::grpc::session::SessionIdTenantResolver::arc(store),
        admin_authorizer,
        limits,
    };
    (chain, engine)
}

/// Spin up the SAME engine-backed server [`start_engine_server`] does (identical
/// tier set, identical chain), but with the engine's embedded `EmbeddedWorker`
/// STOPPED AND JOINED before this returns: the fixture hands back a server with
/// **no in-process claimant** for the `jobs` queue.
///
/// Why: `assemble_grpc_chain` spawns that worker as soon as an engine-backed
/// chain assembles under `[worker] enabled` (the test config's default), and
/// its loop claims a `queued` row on its very first tick with no
/// initial sleep. A test that submits a job and then reads a PRE-CLAIM field of
/// it (the submission-time `{"state":"pending"}` acceleration marker, the
/// `queued` status) is therefore racing the worker: on a slow runner the claim
/// lands first and the read observes a post-claim value. Reading at a quiesced
/// point removes that TOCTOU BY CONSTRUCTION, rather than weakening the
/// assertion to "pre-claim or post-claim" (which would stop being an oracle).
///
/// The worker guard reaches the fixture through the engine's OWN public
/// composability seam — `AssembledChain::into_layered_axum_router`'s
/// `ChainParts::worker`, documented as a lifetime the downstream owns —
/// so no test-only construction seam is added to the server.
/// `EmbeddedWorker::stop_and_join` AWAITS the loop task's return, so once this
/// fixture returns the loop is provably gone, not merely signalled; nothing
/// else can claim, because the catalog is this fixture's own temp dir.
///
/// A test releases work when it wants it via
/// [`EngineServer::spawn_worker`], which starts a worker over the very same
/// engine session the server drives.
pub async fn start_engine_server_worker_quiesced() -> EngineServer {
    // Bind the listener FIRST and hold it — its address feeds the chain and
    // `axum::serve` serves on the very same held listener, so there is no
    // release-then-rebind window (the same no-port-steal property
    // `spawn_bound_chain` gives the eager-bind path).
    let listener = tokio::net::TcpListener::bind(ephemeral_addr())
        .await
        .expect("bind grpc listener");
    let addr = listener.local_addr().expect("local_addr");

    let (chain, engine, dir) = engine_chain_at(addr, non_event_tiers()).await;
    let metrics = Arc::clone(&chain.metrics);
    // `into_layered_axum_router` is the SAFE-DEFAULT split: the returned router
    // already carries the engine's canonical transport stack (metrics +
    // gRPC-web trailer repair + gRPC-web framing), so what this fixture serves
    // is the same remote surface `spawn_bound_chain` serves.
    let (router, parts) = jammi_server::runtime::assemble_grpc_chain(chain)
        .expect("assemble grpc chain")
        .into_layered_axum_router();
    // `None` only if the test config disabled the worker (`[worker] enabled =
    // false`) — also quiesced, since no worker was ever spawned.
    if let Some(worker) = parts.worker {
        worker
            .stop_and_join()
            .await
            .expect("stop the fixture's embedded training worker");
    }

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("axum serve");
    });

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// A control-plane client over `addr`. Source registration / model + topic /
/// channel / mutable introspection all live on `CatalogService`, so the
/// engine-backed test suites build one of these to register the sources their
/// compute verbs then read.
pub async fn catalog_client(
    addr: SocketAddr,
) -> jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient<Channel> {
    jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient::new(
        channel(addr).await,
    )
}

/// Build a request-extending interceptor closure that injects the
/// `jammi-session-id` header on every outgoing request. This is the test
/// counterpart to the engine-default `SessionIdTenantResolver`: the
/// server reads the header and binds the tenant; the test passes the same
/// session id on every call so the binding is observable.
pub fn with_session(
    session_id: &str,
) -> impl Fn(Request<()>) -> Result<Request<()>, tonic::Status> + Clone {
    let id: MetadataValue<_> = session_id.parse().expect("session-id ascii");
    move |mut req: Request<()>| {
        req.metadata_mut().insert(SESSION_HEADER, id.clone());
        Ok(req)
    }
}

/// Spin up the SAME engine-backed server [`start_engine_server`] does (identical
/// tier set, identical chain, identical eager bind), with `[worker] enabled`
/// set to `enabled` **through a real `jammi.toml` loaded by
/// `JammiConfig::load`** — the exact path the `jammi-server` binary takes to
/// its config.
///
/// This is the parameterised config variant of the fixture, and it is
/// deliberately NOT a construction seam: nothing here reaches into
/// [`jammi_server::runtime::ChainParts::worker`] or stops a worker after the
/// fact. Whether a claim loop exists in this process is decided by the one
/// TOML key, read by [`jammi_server::runtime::assemble_grpc_chain`] off the
/// session's own config — so a test built on this fixture proves the KNOB, not
/// the seam.
///
/// The loaded config is asserted to actually carry the requested value before
/// the session opens: `JammiConfig::load` applies `JAMMI_*` environment
/// overrides, so an ambient `JAMMI_WORKER__ENABLED` in the test runner's
/// environment would otherwise silently invert the oracle. It fails loud here
/// instead.
///
/// The rest of the config mirrors `jammi_test_utils::test_config` (CPU device,
/// small batch, temp artifact dir) so the surface served is the one every other
/// engine-backed fixture serves.
pub async fn start_engine_server_with_worker_enabled(enabled: bool) -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("jammi.toml");
    std::fs::write(
        &config_path,
        format!(
            "artifact_dir = \"{artifact_dir}\"\n\
             \n\
             [gpu]\n\
             device = -1\n\
             \n\
             [inference]\n\
             batch_size = 8\n\
             \n\
             [logging]\n\
             level = \"debug\"\n\
             \n\
             [worker]\n\
             enabled = {enabled}\n",
            artifact_dir = dir.path().display(),
        ),
    )
    .expect("write the fixture's jammi.toml");

    let cfg = jammi_db::config::JammiConfig::load(Some(&config_path))
        .expect("the fixture's jammi.toml loads");
    assert_eq!(
        cfg.worker.enabled, enabled,
        "the loaded config must carry the `[worker] enabled` this fixture asked \
         for — a mismatch means an ambient JAMMI_WORKER__ENABLED override is \
         inverting the oracle"
    );
    assert_eq!(
        cfg.artifact_dir.as_path(),
        dir.path(),
        "the loaded config must root its artifacts in this fixture's temp dir"
    );

    let (chain, engine) =
        engine_chain_from_config(ephemeral_addr(), non_event_tiers(), cfg, None).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// Like [`start_engine_server_with_worker_enabled`] with the worker always
/// ON, but ALSO overriding `[lease]` to a fast, test-scale cadence
/// (`duration_secs`/`heartbeat_secs`) instead of the production default (30s
/// lease / 10s heartbeat) — for a live it-test that needs to observe a real
/// claimed job's lease-heartbeat-driven behaviour (e.g. the #485
/// cancel-request watcher, which polls at the SAME cadence the lease keeper
/// renews at: `jammi_ai::fine_tune::worker::spawn_cancel_request_watcher`'s
/// doc) inside a live test's own time budget, with no test-hooks park point
/// (`jammi_ai::jobs::compute_test_hooks` is not available to this crate's
/// `it` target — a real small training spec plus a fast REAL cadence is the
/// only lever this binary has). `heartbeat_secs` must be strictly under half
/// of `lease_secs` ([`jammi_db::config::LeaseConfig::intervals`]'s own
/// invariant); the caller picks values that satisfy it.
pub async fn start_engine_server_with_worker_enabled_and_fast_lease(
    lease_secs: u64,
    heartbeat_secs: u64,
) -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("jammi.toml");
    std::fs::write(
        &config_path,
        format!(
            "artifact_dir = \"{artifact_dir}\"\n\
             \n\
             [gpu]\n\
             device = -1\n\
             \n\
             [inference]\n\
             batch_size = 8\n\
             \n\
             [logging]\n\
             level = \"debug\"\n\
             \n\
             [worker]\n\
             enabled = true\n\
             \n\
             [lease]\n\
             duration_secs = {lease_secs}\n\
             heartbeat_secs = {heartbeat_secs}\n",
            artifact_dir = dir.path().display(),
        ),
    )
    .expect("write the fixture's jammi.toml");

    let cfg = jammi_db::config::JammiConfig::load(Some(&config_path))
        .expect("the fixture's jammi.toml loads");
    assert!(
        cfg.worker.enabled,
        "this fixture always requests [worker] enabled = true"
    );
    assert_eq!(cfg.lease.duration_secs, lease_secs);
    assert_eq!(cfg.lease.heartbeat_secs, heartbeat_secs);

    let (chain, engine) =
        engine_chain_from_config(ephemeral_addr(), non_event_tiers(), cfg, None).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// Spin up the SAME engine-backed server [`start_engine_server`] does
/// (identical tier set, identical chain, identical eager bind), but with
/// `[broker]` overridden to `broker` instead of the config default
/// (in-process). Used by the K4 oracle that exercises a NON-DEFAULT runtime
/// broker kind end to end — both the embedded session and the remote server
/// built from the SAME config must report the identical runtime broker
/// (`grpc_introspection.rs`).
pub async fn start_engine_server_with_broker(
    broker: jammi_db::config::BrokerConfig,
) -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = test_config(dir.path());
    cfg.broker = broker;
    let (chain, engine) =
        engine_chain_from_config(ephemeral_addr(), non_event_tiers(), cfg, None).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

/// Spin up the SAME engine-backed server [`start_engine_server`] does
/// (identical tier set, identical chain, identical eager bind), but with
/// `[server.limits]` overridden to `limits` instead of the config default,
/// and `[worker] enabled = false` -- the fixture the `limits` it-suite
/// (`grpc_limits.rs`) builds its oversize-message / timeout / stream-budget
/// refusal cases on. The worker is disabled unconditionally: those cases
/// need a submitted job to stay `queued` forever (no claimant), so a
/// `WaitJob` stream stays genuinely open across the whole test rather than
/// racing an embedded worker to completion.
pub async fn start_engine_server_with_limits(
    limits: jammi_db::config::LimitsConfig,
) -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = test_config(dir.path());
    cfg.server.limits = limits;
    cfg.worker.enabled = false;
    let (chain, engine) =
        engine_chain_from_config(ephemeral_addr(), non_event_tiers(), cfg, None).await;
    let metrics = Arc::clone(&chain.metrics);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle: AbortOnDropHandle(handle),
        engine,
        metrics,
    }
}

// ---------------------------------------------------------------------------
// The peer-listener fixture (`[server] peer_bind`)
// ---------------------------------------------------------------------------

/// Guards for an in-process server started through the PRODUCTION
/// [`jammi_server::runtime::OssServer`] path — `new` → `bind` →
/// `serve_with_shutdown` — with `[server] peer_bind` set, so the third
/// (internal `PeerService`) listener is the real plumbing, not a fixture's
/// re-implementation of it. All three listeners are ephemeral (`:0`).
pub struct PeerEngineServer {
    /// The public gRPC + Flight SQL listener.
    pub public_addr: SocketAddr,
    /// The internal `PeerService` listener (`[server] peer_bind`).
    pub peer_addr: SocketAddr,
    /// The HTTP side-channel (`/healthz`, `/readyz`, `/metrics`).
    pub health_addr: SocketAddr,
    /// The engine session the server drives — a test builds tables on it
    /// directly (the owner's own result store).
    pub engine: Arc<InferenceSession>,
    /// The server's metrics registry — `jammi_peer_requests_total{rpc}` is the
    /// observable that a peer call reached this owner.
    pub metrics: Arc<jammi_server::routes::health::MetricsRegistry>,
    pub shutdown: oneshot::Sender<()>,
    pub handle: AbortOnDropHandle<()>,
    /// RAII root of the engine's artifact dir when this fixture owns it;
    /// `None` when the caller supplied (and roots) a shared dir.
    pub _dir: Option<TempDir>,
}

/// A `test_config` over `artifact_dir` with every listener at loopback `:0`
/// and `peer_bind` set — the config a peer-bound fixture opens.
pub fn peer_bind_config(artifact_dir: &std::path::Path) -> jammi_db::config::JammiConfig {
    let mut cfg = test_config(artifact_dir);
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    cfg.server.peer_bind = Some("127.0.0.1:0".into());
    cfg
}

/// Start a server from `cfg` through the production `OssServer` path. `cfg`
/// must set `peer_bind` (the fixture asserts the third listener bound).
pub async fn start_engine_server_from_config(
    cfg: jammi_db::config::JammiConfig,
    dir: Option<TempDir>,
) -> PeerEngineServer {
    let server = jammi_server::runtime::OssServer::new(cfg)
        .await
        .expect("oss server");
    let engine = server.session();
    let metrics = server.metrics();
    let bound = server.bind().await.expect("bind all listeners");
    let public_addr = bound.flight_addr();
    let health_addr = bound.health_addr();
    let peer_addr = bound
        .peer_addr()
        .expect("peer_bind is set, so the third listener is bound");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        bound
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("oss server serve");
    });
    PeerEngineServer {
        public_addr,
        peer_addr,
        health_addr,
        engine,
        metrics,
        shutdown: shutdown_tx,
        handle: AbortOnDropHandle(handle),
        _dir: dir,
    }
}

/// [`start_engine_server`]'s peer-bound twin: a fresh engine over its own
/// temp artifact dir, every listener ephemeral, `PeerService` served on
/// `peer_addr`.
pub async fn start_engine_server_with_peer_bind() -> PeerEngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = peer_bind_config(dir.path());
    start_engine_server_from_config(cfg, Some(dir)).await
}

/// Serve the engine chain (every tier except event) over an ALREADY OPEN
/// engine session on a loopback ephemeral port — the public listener a test
/// drives `Search` through against a session it opened itself. Returns the
/// bound address, the shutdown trigger and the abort-on-drop serve handle.
pub async fn start_engine_server_over_session(
    session: Arc<InferenceSession>,
) -> (SocketAddr, oneshot::Sender<()>, AbortOnDropHandle<()>) {
    let (chain, _engine) = chain_over_session(ephemeral_addr(), non_event_tiers(), session, None);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;
    (addr, shutdown_tx, AbortOnDropHandle(handle))
}
