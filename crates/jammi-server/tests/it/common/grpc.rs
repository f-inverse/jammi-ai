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

/// Guards that keep an in-process engine-backed gRPC server (and its catalog)
/// alive for the duration of a test. Dropping `shutdown` or letting it fall out
/// of scope tears the server down; `_dir` roots the engine's temp artifact dir.
pub struct EngineServer {
    pub addr: SocketAddr,
    pub shutdown: oneshot::Sender<()>,
    /// RAII guard: roots the engine's temp artifact dir for the server's
    /// lifetime and deletes it on drop. Held, never read.
    pub _dir: TempDir,
    pub handle: tokio::task::JoinHandle<()>,
    /// The same `Arc<InferenceSession>` the server task drives. Shared so a
    /// test can wrap it in a local `Session` and assert the data-plane client
    /// over the wire returns identical results / errors against the *same*
    /// engine.
    pub engine: Arc<InferenceSession>,
}

impl EngineServer {
    /// Start a training worker over the SAME engine session this server drives,
    /// returning the RAII guard that owns its loop (dropping the guard stops it).
    ///
    /// The release valve for [`start_engine_server_worker_quiesced`]: a test
    /// reads a submitted job's pre-claim state while nothing can claim it, then
    /// calls this to let the job actually run, and awaits its terminal state
    /// through the public surface. A worker is a worker regardless of who
    /// spawned it — this is the identical `EmbeddedWorker` the `train` tier
    /// starts, over the identical session — so releasing here restores the
    /// production shape rather than simulating it.
    #[cfg(feature = "train")]
    pub fn spawn_training_worker(&self) -> jammi_ai::fine_tune::worker::EmbeddedWorker {
        jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&self.engine)
            .expect("the test config's worker intervals are valid")
    }
}

/// Spin up an in-process gRPC server hosting the chain *with* the engine-backed
/// services, mounting every compiled-in tier **except** the event tier (no
/// trigger handles). Shared by the `grpc_inference`, `grpc_eval`,
/// `grpc_introspection`, and `grpc_training` suites so they drive the same
/// wiring the embedding suite does.
pub async fn start_engine_server() -> EngineServer {
    start_engine_server_with_tiers(non_event_tiers()).await
}

/// Every compiled-in optional tier except event — the engine-backed serve +
/// eval + (when compiled) train surface, without the trigger stream. The tier
/// set [`start_engine_server`] and [`start_engine_server_worker_quiesced`]
/// share, so the two fixtures mount the identical surface.
fn non_event_tiers() -> jammi_server::tiers::TierSet {
    let optional = jammi_server::tiers::ServiceTier::OPTIONAL
        .into_iter()
        .filter(|t| *t != jammi_server::tiers::ServiceTier::Event && t.compiled_in());
    jammi_server::tiers::TierSet::resolve(optional).expect("non-event tiers resolve")
}

/// Like [`start_engine_server`] but also mounts the trigger handles (the event
/// tier), so the `TriggerService` (topics / publish / subscribe) is reachable
/// over the wire. Shared by the data-plane client topic/subscribe/audit parity
/// tests, which drive those surfaces against the same engine a local `Session`
/// wraps.
pub async fn start_engine_server_with_trigger() -> EngineServer {
    start_engine_server_with_tiers(jammi_server::tiers::TierSet::all_compiled()).await
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
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle,
        engine,
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
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let (chain, engine) = engine_chain_from_config(addr, tiers, cfg).await;
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
) -> (jammi_server::runtime::GrpcChain, Arc<InferenceSession>) {
    // `open` (not `new`) so the engine-backed server registers the compound
    // query SQL functions (`annotate`, …) on its context — the same shape the
    // production `OssServer` builds, and what the Flight SQL `annotate` test
    // exercises.
    let session = InferenceSession::open(cfg).await.expect("session");

    let store = SessionStore::new();
    let trigger = tiers
        .contains(jammi_server::tiers::ServiceTier::Event)
        .then(|| jammi_server::TriggerHandles {
            topic_repo: session.topic_repo(),
            publisher: session.publisher(),
            subscriber: session.subscriber(),
        });

    let engine = Arc::clone(&session);
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
    };
    (chain, engine)
}

/// Spin up the SAME engine-backed server [`start_engine_server`] does (identical
/// tier set, identical chain), but with the `train` tier's embedded
/// `EmbeddedWorker` STOPPED AND JOINED before this returns: the fixture hands
/// back a server with **no in-process claimant** for the `training_jobs` queue.
///
/// Why: `assemble_grpc_chain` spawns that worker as soon as the `train` tier is
/// mounted, and its loop claims a `queued` row on its very first tick with no
/// initial sleep. A test that submits a job and then reads a PRE-CLAIM field of
/// it (the submission-time `{"state":"pending"}` acceleration marker, the
/// `queued` status) is therefore racing the worker: on a slow runner the claim
/// lands first and the read observes a post-claim value. Reading at a quiesced
/// point removes that TOCTOU BY CONSTRUCTION, rather than weakening the
/// assertion to "pre-claim or post-claim" (which would stop being an oracle).
///
/// The worker guard reaches the fixture through the engine's OWN public
/// composability seam — `AssembledChain::into_layered_axum_router`'s
/// `ChainParts::train_worker`, documented as a lifetime the downstream owns —
/// so no test-only construction seam is added to the server.
/// `EmbeddedWorker::stop_and_join` AWAITS the loop task's return, so once this
/// fixture returns the loop is provably gone, not merely signalled; nothing
/// else can claim, because the catalog is this fixture's own temp dir.
///
/// A test releases work when it wants it via
/// [`EngineServer::spawn_training_worker`], which starts a worker over the very
/// same engine session the server drives.
#[cfg(feature = "train")]
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
    // `into_layered_axum_router` is the SAFE-DEFAULT split: the returned router
    // already carries the engine's canonical transport stack (metrics +
    // gRPC-web trailer repair + gRPC-web framing), so what this fixture serves
    // is the same remote surface `spawn_bound_chain` serves.
    let (router, parts) = jammi_server::runtime::assemble_grpc_chain(chain)
        .expect("assemble grpc chain")
        .into_layered_axum_router();
    // `None` only in a build whose `train` tier is not mounted at all — also
    // quiesced (no worker was ever spawned), and such a build fails loudly at
    // the first `StartTraining` rather than silently racing.
    if let Some(worker) = parts.train_worker {
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
        handle,
        engine,
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
/// tier set — the `train` tier included — identical chain, identical eager
/// bind), with `[training] run_worker` set to `run_worker` **through a real
/// `jammi.toml` loaded by `JammiConfig::load`** — the exact path the
/// `jammi-server` binary takes to its config.
///
/// This is the parameterised config variant of the fixture, and it is
/// deliberately NOT a construction seam: nothing here reaches into
/// [`jammi_server::runtime::ChainParts::train_worker`] or stops a worker after
/// the fact. Whether a claim loop exists in this process is decided by the one
/// TOML key, read by [`jammi_server::runtime::assemble_grpc_chain`] off the
/// session's own config — so a test built on this fixture proves the KNOB, not
/// the seam.
///
/// The loaded config is asserted to actually carry the requested value before
/// the session opens: `JammiConfig::load` applies `JAMMI_*` environment
/// overrides, so an ambient `JAMMI_TRAINING__RUN_WORKER` in the test runner's
/// environment would otherwise silently invert the oracle. It fails loud here
/// instead.
///
/// The rest of the config mirrors `jammi_test_utils::test_config` (CPU device,
/// small batch, temp artifact dir) so the surface served is the one every other
/// engine-backed fixture serves.
#[cfg(feature = "train")]
pub async fn start_engine_server_with_run_worker(run_worker: bool) -> EngineServer {
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
             [training]\n\
             run_worker = {run_worker}\n",
            artifact_dir = dir.path().display(),
        ),
    )
    .expect("write the fixture's jammi.toml");

    let cfg = jammi_db::config::JammiConfig::load(Some(&config_path))
        .expect("the fixture's jammi.toml loads");
    assert_eq!(
        cfg.training.run_worker, run_worker,
        "the loaded config must carry the run_worker this fixture asked for — a \
         mismatch means an ambient JAMMI_TRAINING__RUN_WORKER override is \
         inverting the oracle"
    );
    assert_eq!(
        cfg.artifact_dir.as_path(),
        dir.path(),
        "the loaded config must root its artifacts in this fixture's temp dir"
    );

    let (chain, engine) = engine_chain_from_config(ephemeral_addr(), non_event_tiers(), cfg).await;
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr, handle) = spawn_bound_chain(chain, shutdown_rx).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle,
        engine,
    }
}
