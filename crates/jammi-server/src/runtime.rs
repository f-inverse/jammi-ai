//! `OssServer` — the single orchestration entry-point for the OSS
//! `jammi-server` binary.
//!
//! One `OssServer` wires together:
//!
//! - the engine [`InferenceSession`] (catalog, mutable tables, broker)
//! - a [`SessionStore`] shared between Flight SQL and the gRPC services
//! - the Axum side-channel router (`/healthz`, `/readyz`, `/metrics`)
//! - one Tonic server hosting `FlightSqlService + CatalogService +
//!   TriggerService` on a single port
//! - two-mode graceful shutdown (DRAIN on SIGTERM, RELEASE on SIGINT or a
//!   second signal) wired through [`tokio::sync::watch`], so every listener
//!   — HTTP side-channel, gRPC/Flight, and the internal peer listener when
//!   bound — drains in parallel off the same signal
//!
//! The structure is intentionally flat: no `runtime/` directory, no
//! per-component sub-modules. When a second binary materialises the same
//! shape can be reused — the orchestration is the engine of last resort
//! and earns its keep by being grep-able in one place.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Weak};

use arrow_flight::flight_service_server::FlightServiceServer;
use async_trait::async_trait;
use axum::Router;
use datafusion::execution::context::SessionContext;
use datafusion_flight_sql_server::service::FlightSqlService;
use jammi_ai::session::InferenceSession;
use jammi_db::audit::{ensure_master_key_present, EnvSigningKeyStore, FileSigningKeyStore};
use jammi_db::config::{JammiConfig, SigningKeyConfig};
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::{oneshot, watch};
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic_web::GrpcWebLayer;
use tower::Layer;

use crate::flight::TenantBoundProvider;
use crate::grpc::audit::AuditServer;
use crate::grpc::catalog::{AdminAuthorizer, CatalogServer};
use crate::grpc::embedding::EmbeddingServer;
use crate::grpc::eval::EvalServer;
use crate::grpc::inference::InferenceServer;
use crate::grpc::job::JobServer;
use crate::grpc::peer::PeerServer;
use crate::grpc::pipeline::PipelineServer;
use crate::grpc::proto::audit::audit_service_server::AuditServiceServer;
use crate::grpc::proto::catalog::catalog_service_server::CatalogServiceServer;
use crate::grpc::proto::embedding::embedding_service_server::EmbeddingServiceServer;
use crate::grpc::proto::eval::eval_service_server::EvalServiceServer;
use crate::grpc::proto::inference::inference_service_server::InferenceServiceServer;
use crate::grpc::proto::job::job_service_server::JobServiceServer;
use crate::grpc::proto::peer::peer_service_server::PeerServiceServer;
use crate::grpc::proto::pipeline::pipeline_service_server::PipelineServiceServer;
use crate::grpc::proto::trigger::trigger_service_server::TriggerServiceServer;
use crate::grpc::session::{SessionIdTenantResolver, SessionStore, TenantResolver};
use crate::grpc::trigger::TriggerServer;
use crate::grpc_web_trailers::GrpcWebTrailersLayer;
use crate::metrics_layer::MetricsLayer;
use crate::routes::health::MetricsRegistry;
use crate::tenant_resolver_layer::TenantResolverLayer;
use crate::tiers::{ServiceTier, TierSet};
use crate::trace_context_layer::TraceContextLayer;

/// Errors `OssServer::run` can surface to the binary's `main`.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("config error: {0}")]
    Config(String),
    #[error("service tier: {0}")]
    Tier(#[from] crate::tiers::TierError),
    #[error("engine init: {0}")]
    Engine(#[from] jammi_db::error::JammiError),
    #[error("metrics registry: {0}")]
    Metrics(#[from] prometheus::Error),
    #[error("transport: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("addr parse: {0}")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("{0}")]
    AuditMasterKey(String),
    /// A `[server] preload_models` entry could not be loaded at startup —
    /// the model failed to load, a bare id has no `models` row to take its
    /// task from, or the task could not be resolved. The server exits
    /// non-zero instead of serving.
    #[error("preload_models: `{id}`: {reason}")]
    Preload { id: String, reason: String },
}

/// Fail-closed startup check for the audit signing key.
///
/// Classifies the configured source into a four-state lattice — **unset**
/// (`JAMMI_AUDIT_MASTER_KEY` absent for [`SigningKeyConfig::Env`], or ENOENT
/// for [`SigningKeyConfig::File`]'s mounted path), **set-empty** (present but
/// empty, or all-whitespace, after trimming — what `deploy/.env.example`
/// ships as its `JAMMI_AUDIT_MASTER_KEY=` placeholder, and what an unfilled
/// Kubernetes Secret key produces whichever source reads it), **set-malformed**
/// (present, non-empty, but does not decode to 32 bytes of hex), and
/// **set-valid**. Unset and set-empty are BOTH treated as absent — audit
/// signing simply stays unusable until the first `AuditService` write,
/// exactly as before this check existed; this function does not tighten
/// that policy for either state. What this closes is set-malformed: that
/// used to let `jammi-server serve` boot successfully with audit signing
/// silently dead, discovered only on the first signing attempt deep inside a
/// request. A set-malformed key now refuses to start.
///
/// For [`SigningKeyConfig::File`], "present" additionally fails CLOSED on
/// every read error OTHER than the path not existing at all — an
/// existing-but-unreadable mount (wrong owner, `0o000`) is `present`, not
/// silently read as absent, so such a mount refuses startup instead of
/// booting with signing dead.
///
/// The refusal message names only the SHAPE of the failure ("not valid hex",
/// or the character count against the expected 64) — never a character of
/// the configured value. This function derives that shape itself, from the
/// raw value it already read to classify presence, rather than through
/// [`jammi_db::audit::AuditError::MasterKey`]'s `Display`: that type wraps
/// `hex::FromHexError::InvalidHexCharacter`, whose own `Display` names the
/// offending character, and letting that string reach this function's
/// caller (ultimately `stderr`, per `main.rs`) would leak it. The actual
/// pass/fail DECISION (is this key usable) still delegates entirely to
/// [`jammi_db::audit::ensure_master_key_present`] (in turn
/// [`jammi_db::audit::SigningKeyStore::master_key`]) — this function
/// re-derives only the failure's shape, never the decode logic that decides
/// pass/fail.
pub fn validate_audit_master_key(config: &JammiConfig) -> Result<(), ServerError> {
    let raw = match read_configured_key_source(config) {
        Ok(raw) => raw,
        Err(msg) => return Err(ServerError::AuditMasterKey(msg)),
    };
    let trimmed = raw.as_deref().map(str::trim).unwrap_or("");
    if trimmed.is_empty() {
        // Unset or set-empty: both absent for this check.
        return Ok(());
    }

    let result = match &config.signing_key {
        SigningKeyConfig::Env => ensure_master_key_present(&EnvSigningKeyStore),
        SigningKeyConfig::File { path } => {
            ensure_master_key_present(&FileSigningKeyStore::new(path.clone()))
        }
    };
    result.map_err(|_| ServerError::AuditMasterKey(key_shape_error(config, trimmed)))
}

/// Read the configured signing-key source's raw value.
///
/// `Ok(Some(value))` when a value was read (untrimmed — the caller,
/// [`validate_audit_master_key`], trims once, uniformly, for both sources).
/// `Ok(None)` when the source is unset: `JAMMI_AUDIT_MASTER_KEY` absent for
/// [`SigningKeyConfig::Env`], or ENOENT for [`SigningKeyConfig::File`]'s
/// path. `Err` when the source could not be read for a reason OTHER than
/// being unset — a mounted-but-unreadable file — and the error message names
/// only the path, never file content, so it is safe to surface directly as
/// this check's refusal text.
fn read_configured_key_source(config: &JammiConfig) -> Result<Option<String>, String> {
    match &config.signing_key {
        SigningKeyConfig::Env => Ok(std::env::var(jammi_db::audit::MASTER_KEY_ENV).ok()),
        SigningKeyConfig::File { path } => match std::fs::metadata(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!(
                "audit master key file `{}` could not be read: {e}",
                path.display()
            )),
            Ok(_) => std::fs::read_to_string(path).map(Some).map_err(|e| {
                format!(
                    "audit master key file `{}` could not be read: {e}",
                    path.display()
                )
            }),
        },
    }
}

/// Build a refusal message describing ONLY the shape of a present, non-empty,
/// but invalid key — never a character of `trimmed` itself. Mirrors the
/// wording [`jammi_db::audit::AuditError::MasterKey`]'s `Display` uses
/// ("audit master key not configured or invalid — … 32 bytes of hex (64 hex
/// chars): …") so the observable message is unchanged in shape, with only the
/// leaky tail replaced.
fn key_shape_error(config: &JammiConfig, trimmed: &str) -> String {
    let source = match &config.signing_key {
        SigningKeyConfig::Env => format!("set {} to", jammi_db::audit::MASTER_KEY_ENV),
        SigningKeyConfig::File { path } => {
            format!("the file at `{}` must contain", path.display())
        }
    };
    let shape = if is_hex_decodable(trimmed) {
        format!("got {} characters", trimmed.chars().count())
    } else {
        "not valid hex".to_string()
    };
    format!(
        "audit master key not configured or invalid — {source} 32 bytes of hex \
         (64 hex chars): {shape}"
    )
}

/// Would `trimmed` decode as hex at all — an even count of ASCII hex digits?
/// A local re-derivation of exactly what [`hex::decode`] would accept
/// (`jammi-server` carries no non-dev `hex` dependency — a regular
/// dependency addition is a `Cargo.toml` change, the lead/`docs-ci` shared
/// class this crate's own code does not touch on its own), used ONLY to pick
/// which safe wording [`key_shape_error`] prints. The actual pass/fail
/// decision is still [`jammi_db::audit::ensure_master_key_present`]'s, not
/// this function's — see [`validate_audit_master_key`]'s doc.
fn is_hex_decodable(trimmed: &str) -> bool {
    trimmed.len().is_multiple_of(2) && trimmed.chars().all(|c| c.is_ascii_hexdigit())
}

/// A readiness probe: pings whatever resource readiness depends on. The
/// implementation lives behind a trait so tests can substitute a stub
/// that returns deterministic outcomes (the substrate session itself
/// has more moving parts than a probe test cares about).
#[async_trait]
pub trait ReadinessCheck: Send + Sync {
    /// `Ok(())` means the underlying resource responded; `Err(s)` is a
    /// human-readable failure reason surfaced in the `/readyz` body.
    async fn check(&self) -> Result<(), String>;
}

/// Wrapper that holds the active [`ReadinessCheck`] behind an `Arc` so
/// Axum can share it across handlers via `State`, plus the process-level
/// readiness phases a probe cannot know on its own: draining (a shutdown
/// began — `/readyz` 503 `"draining"` so a balancer stops routing here while
/// in-flight work finishes).
pub struct ReadinessProbe {
    inner: Arc<dyn ReadinessCheck>,
    draining: std::sync::atomic::AtomicBool,
    /// Warm-before-ready: `false` until every `[server] preload_models`
    /// entry is cached (`/readyz` 503 `"preloading i/n"` meanwhile).
    warm: std::sync::atomic::AtomicBool,
    preloaded: std::sync::atomic::AtomicUsize,
    preload_total: std::sync::atomic::AtomicUsize,
}

impl ReadinessProbe {
    pub fn new(inner: Arc<dyn ReadinessCheck>) -> Self {
        Self {
            inner,
            draining: std::sync::atomic::AtomicBool::new(false),
            warm: std::sync::atomic::AtomicBool::new(true),
            preloaded: std::sync::atomic::AtomicUsize::new(0),
            preload_total: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Enter the preloading phase: `/readyz` reports 503 `"preloading 0/n"`
    /// until [`Self::set_warm`].
    pub fn begin_preload(&self, total: usize) {
        use std::sync::atomic::Ordering;
        self.preload_total.store(total, Ordering::SeqCst);
        self.preloaded.store(0, Ordering::SeqCst);
        self.warm.store(false, Ordering::SeqCst);
    }

    /// One more entry cached.
    pub fn note_preloaded(&self) {
        self.preloaded
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Every entry cached: `/readyz` may report ready.
    pub fn set_warm(&self) {
        self.warm.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the preload phase has completed (or never existed).
    pub fn is_warm(&self) -> bool {
        self.warm.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// `/readyz` reports 503 `"draining"` from now on — set by the DRAIN and
    /// RELEASE arms of [`BoundServer::serve_with_signals`].
    pub fn begin_drain(&self) {
        self.draining
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether a shutdown has begun.
    pub fn is_draining(&self) -> bool {
        self.draining.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn check(&self) -> Result<(), String> {
        use std::sync::atomic::Ordering;
        if self.is_draining() {
            return Err("draining".to_string());
        }
        if !self.is_warm() {
            return Err(format!(
                "preloading {}/{}",
                self.preloaded.load(Ordering::SeqCst),
                self.preload_total.load(Ordering::SeqCst)
            ));
        }
        self.inner.check().await
    }
}

/// Readiness check backed by the engine's catalog backend. Delegates to
/// [`jammi_db::catalog::Catalog::ping`], which issues a backend-native
/// reachability probe (no transaction, no lock) and surfaces pool failures
/// as [`jammi_db::catalog::backend::BackendError::Unavailable`].
pub struct CatalogPingProbe {
    session: Arc<InferenceSession>,
}

impl CatalogPingProbe {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }
}

#[async_trait]
impl ReadinessCheck for CatalogPingProbe {
    async fn check(&self) -> Result<(), String> {
        self.session
            .catalog()
            .ping()
            .await
            .map_err(|e| e.to_string())
    }
}

/// A liveness probe (`/healthz`): can this process keep its leases and its
/// claim loop alive? Behind a trait so tests substitute a stub; the
/// production implementation is [`EngineLiveness`].
pub trait LivenessCheck: Send + Sync {
    fn check(&self) -> LivenessReport;
}

/// What `/healthz` reports. Unhealthy (503) exactly when the lease keeper
/// thread is dead (every lease this process holds is already lost) or the
/// claim loop task panicked (`failed`); a loop that stopped, was aborted
/// (a RELEASE) or never existed is not a fault, and neither is draining.
/// No slow-step detection: the runtime owns "how long is too long".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivenessReport {
    /// `LeaseKeeper::is_alive`.
    pub lease_keeper: bool,
    /// `running` / `stopped` / `aborted` / `failed`, or `none` on a process
    /// with no claim loop.
    pub claim_loop: &'static str,
}

impl LivenessReport {
    pub fn healthy(&self) -> bool {
        self.lease_keeper && self.claim_loop != "failed"
    }
}

/// Wrapper that holds the active [`LivenessCheck`] behind an `Arc` so Axum
/// can share it across handlers via `State`.
pub struct LivenessProbe {
    inner: Arc<dyn LivenessCheck>,
}

impl LivenessProbe {
    pub fn new(inner: Arc<dyn LivenessCheck>) -> Self {
        Self { inner }
    }

    /// A probe that always reports healthy with no claim loop — for a
    /// router with no engine behind it (`crate::build_router`, fixtures).
    pub fn always_healthy() -> Self {
        struct AlwaysAlive;
        impl LivenessCheck for AlwaysAlive {
            fn check(&self) -> LivenessReport {
                LivenessReport {
                    lease_keeper: true,
                    claim_loop: "none",
                }
            }
        }
        Self::new(Arc::new(AlwaysAlive))
    }

    pub fn check(&self) -> LivenessReport {
        self.inner.check()
    }
}

/// The production liveness: the session's lease keeper and, when this
/// process runs a claim loop, the loop's shared state (a `Weak`, so the
/// probe never keeps it alive; a gone state reads `stopped`).
pub struct EngineLiveness {
    keeper: Arc<jammi_db::catalog::lease_keeper::LeaseKeeper>,
    worker: Option<Weak<jammi_ai::fine_tune::worker::WorkerShared>>,
}

impl EngineLiveness {
    pub fn new(
        keeper: Arc<jammi_db::catalog::lease_keeper::LeaseKeeper>,
        worker: Option<Weak<jammi_ai::fine_tune::worker::WorkerShared>>,
    ) -> Self {
        Self { keeper, worker }
    }
}

impl LivenessCheck for EngineLiveness {
    fn check(&self) -> LivenessReport {
        use jammi_ai::fine_tune::worker::LoopState;
        let claim_loop = match &self.worker {
            None => "none",
            Some(weak) => match weak.upgrade() {
                None => "stopped",
                Some(shared) => match shared.loop_state() {
                    LoopState::Running => "running",
                    LoopState::Stopped => "stopped",
                    LoopState::Aborted => "aborted",
                    LoopState::Failed => "failed",
                },
            },
        };
        LivenessReport {
            lease_keeper: self.keeper.is_alive(),
            claim_loop,
        }
    }
}

/// The OSS server instance. Constructed via [`Self::new`] and consumed
/// by [`Self::run`]. Holds every long-lived dependency the binary
/// orchestrates — bind addresses, the engine session, the shared
/// SessionStore, the metrics registry, and the readiness probe.
pub struct OssServer {
    flight_addr: SocketAddr,
    health_addr: SocketAddr,
    /// `[server] peer_bind`: the internal peer listener's address, `Some`
    /// iff this replica is a segment owner. `None` = no third listener.
    peer_addr: Option<SocketAddr>,
    session: Arc<InferenceSession>,
    session_store: SessionStore,
    metrics: Arc<MetricsRegistry>,
    readiness: Arc<ReadinessProbe>,
    tiers: TierSet,
}

impl OssServer {
    /// Build an OSS server from `JammiConfig`. Validates the server
    /// configuration up front (parses both bind addresses, rejects two
    /// identical FIXED bind addresses — identical `:0` ephemeral addresses
    /// are allowed, since each resolves to a distinct port at bind),
    /// constructs the engine session
    /// (catalog, mutable tables, broker), and prepares the shared
    /// metrics registry and readiness probe.
    pub async fn new(config: JammiConfig) -> Result<Self, ServerError> {
        config
            .server
            .validate()
            .map_err(|e| ServerError::Config(e.to_string()))?;
        // Reject lease timing that violates the heartbeat margin, or a
        // worker poll that is a busy-loop, at construction — before the
        // worker is spawned or a result table is leased.
        let lease = config
            .lease
            .intervals()
            .map_err(|e| ServerError::Config(e.to_string()))?;
        config
            .worker
            .worker_intervals(lease)
            .map_err(|e| ServerError::Config(e.to_string()))?;

        let flight_addr: SocketAddr = config.server.flight_listen.parse()?;
        let health_addr: SocketAddr = config.server.health_listen.parse()?;
        let peer_addr: Option<SocketAddr> = config
            .server
            .peer_bind
            .as_deref()
            .map(str::parse)
            .transpose()?;

        // Resolve the mounted tier set before constructing the engine: a config
        // that names an unknown tier or one whose feature is compiled out is a
        // startup error, not a silent degrade.
        let tiers = TierSet::from_config(&config.server.services)?;

        // `open` (not `new`) registers the `annotate` query UDTF on the engine's
        // DataFusion context — the Flight SQL surface needs it. It already returns
        // an `Arc<InferenceSession>`.
        let session = InferenceSession::open(config).await?;
        // Warm-before-ready: with models to preload, the claim loop must not
        // claim before they are cached — close the session's worker gate
        // before the worker is spawned at bind; `serve_with_signals` opens
        // it once warm. An empty list leaves the gate open (existing
        // deployments unchanged).
        if !session.inner_config().server.preload_models.is_empty() {
            session.close_worker_gate();
        }
        let session_store = SessionStore::new();
        let metrics = Arc::new(MetricsRegistry::new()?);
        // Every process has a keeper: `jammi_lease_heartbeat_age_seconds` is
        // present on every server.
        metrics.attach_keeper(Arc::clone(session.lease_keeper()))?;
        // The placed-search failure-ladder counters live in the engine's
        // result store; the registry reads them at scrape as
        // `jammi_peer_search_failures_total{reason}`. Registered here, once,
        // additively — `MetricsRegistry::new` keeps its arity.
        metrics.install_peer_failures(session.result_store().peer_failures())?;
        let readiness = Arc::new(ReadinessProbe::new(Arc::new(CatalogPingProbe::new(
            Arc::clone(&session),
        ))));

        Ok(Self {
            flight_addr,
            health_addr,
            peer_addr,
            session,
            session_store,
            metrics,
            readiness,
            tiers,
        })
    }

    /// Shared handle to the metrics registry. Test fixtures and the
    /// gRPC services use this to increment counters.
    pub fn metrics(&self) -> Arc<MetricsRegistry> {
        Arc::clone(&self.metrics)
    }

    /// Shared handle to the engine session. Useful in tests that want
    /// to publish to a topic or read a mutable table while the server
    /// is running.
    pub fn session(&self) -> Arc<InferenceSession> {
        Arc::clone(&self.session)
    }

    /// Override the readiness probe. Used by integration tests to make
    /// `/readyz` deterministically return 503.
    pub fn with_readiness(mut self, readiness: Arc<ReadinessProbe>) -> Self {
        self.readiness = readiness;
        self
    }

    /// Bind both listeners eagerly and return a [`BoundServer`] holding them
    /// live. The gRPC + Flight SQL surface and the HTTP side-channel are each
    /// bound here — so their ACTUAL addresses are known (the real ports even
    /// when the config requested an ephemeral `:0`) before a single connection
    /// is served, and neither port is released before
    /// [`BoundServer::serve_with_shutdown`] serves on the same listeners. A
    /// caller reads the resolved ports off the returned handle
    /// ([`BoundServer::flight_addr`] / [`BoundServer::health_addr`]) while the
    /// listeners stay bound, with no observable release-then-rebind window.
    pub async fn bind(self) -> Result<BoundServer, ServerError> {
        let health_listener = TcpListener::bind(self.health_addr).await?;
        let health_addr = health_listener.local_addr()?;
        // The THIRD listener: `PeerService` on `[server] peer_bind`, built
        // OUTSIDE `assemble_grpc_chain` — never added to the public `Routes`,
        // never wrapped by the `TenantResolverLayer`, never advertised by
        // `GetServerInfo`. The public listener answers UNIMPLEMENTED for its
        // paths. Its routes are a plain `tonic::service::Routes`, so a second
        // internal service can be mounted beside `PeerService` here later. The
        // registry is cloned now because `MetricsLayer::new(self.metrics)`
        // moves the `Arc` into the public chain below.
        let peer = match self.peer_addr {
            Some(addr) => {
                let listener = TcpListener::bind(addr).await?;
                let routes = tonic::service::Routes::new(PeerServiceServer::new(PeerServer::new(
                    Arc::clone(&self.session),
                )));
                Some((listener, routes, Arc::clone(&self.metrics)))
            }
            None => None,
        };
        let peer_addr = match &peer {
            Some((listener, _, _)) => Some(listener.local_addr()?),
            None => None,
        };
        // Cloned before `build_grpc_chain`/`assemble_grpc_chain` consume
        // `self` — `AssembledChain`/`BoundChain` hold their own `Arc` clones
        // internally (captured by the mounted services), but neither type
        // carries the session back out as its own field, so this is the one
        // handle `serve_with_shutdown` has to release the catalog through
        // once the serve loop drains.
        let session = Arc::clone(&self.session);
        let readiness = Arc::clone(&self.readiness);
        let mut grpc = assemble_grpc_chain(self.build_grpc_chain())?.bind().await?;
        // Hoist the worker guard out of the chain (D3): ownership decides
        // who can DRAIN or RELEASE, and the server's two-mode shutdown needs
        // the guard alive past the gRPC serve future, which the chain would
        // otherwise drop it with.
        let worker = grpc.take_worker();
        // The worker gauge families exist only where a claim loop does.
        if let Some(w) = &worker {
            self.metrics.attach_worker(w.shared())?;
        }
        // Liveness reads the keeper and (when present) the loop's state, so
        // the side-channel router is built once the worker guard is known.
        let liveness = Arc::new(LivenessProbe::new(Arc::new(EngineLiveness::new(
            Arc::clone(session.lease_keeper()),
            worker.as_ref().map(|w| w.shared()),
        ))));
        let health_router =
            crate::build_health_router(Arc::clone(&readiness), Arc::clone(&self.metrics), liveness);
        Ok(BoundServer {
            grpc,
            health_listener,
            health_addr,
            health_router,
            peer,
            peer_addr,
            session,
            worker,
            readiness,
        })
    }

    /// Drive the server until a shutdown signal arrives — SIGTERM = DRAIN,
    /// SIGINT (or any signal while draining) = RELEASE; see
    /// [`BoundServer::serve`].
    pub async fn run(self) -> Result<(), ServerError> {
        self.bind().await?.serve().await.map(|_| ())
    }

    /// Variant of [`Self::run`] that accepts a caller-provided
    /// shutdown future. Tests use this to drive deterministic
    /// teardown.
    pub async fn run_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        self.bind().await?.serve_with_shutdown(shutdown).await
    }

    /// Assemble the engine's [`GrpcChain`] from this server's config and engine
    /// session. Derives the event-tier trigger handles (only when the event
    /// tier is mounted) and the engine-default `jammi-session-id` tenant
    /// resolver — the OSS-cooperative multitenancy binder.
    fn build_grpc_chain(&self) -> GrpcChain {
        // The event tier (`TriggerService`) is mounted only when the deployment
        // selected it; the handles are derived from the same engine session.
        let trigger = self
            .tiers
            .contains(ServiceTier::Event)
            .then(|| crate::TriggerHandles {
                topic_repo: self.session.topic_repo(),
                publisher: self.session.publisher(),
                subscriber: self.session.subscriber(),
            });
        GrpcChain {
            addr: self.flight_addr,
            flight_ctx: self.session.context().clone(),
            flight_binding: self.session.tenant_binding_arc(),
            store: self.session_store.clone(),
            trigger,
            engine: Some(Arc::clone(&self.session)),
            tiers: self.tiers.clone(),
            metrics: Arc::clone(&self.metrics),
            // The OSS binary binds tenants with the engine-default
            // `jammi-session-id` resolver — the OSS-cooperative multitenancy
            // behavior, now a first-class resolver rather than an implicit
            // interceptor.
            tenant_resolver: SessionIdTenantResolver::arc(self.session_store.clone()),
            // The shipped default: no admin authorizer wired, so `Reconcile`'s
            // `all = true` cross-tenant admin pass is refused outright. A
            // downstream that needs it supplies its own `AdminAuthorizer` by
            // constructing `GrpcChain` directly (this OSS binary's own
            // orchestration path has no seam to configure one yet).
            admin_authorizer: None,
            // `[server.limits]` from the SAME config the engine session was
            // opened with — a wire deployment and an in-process one read the
            // identical knob, exactly as `[worker] enabled` does above.
            limits: self.session.inner_config().server.limits,
        }
    }
}

/// An [`OssServer`] with both listeners bound and held live — the eager-bind
/// half of the serve seam. Returned by [`OssServer::bind`]. The gRPC + Flight
/// SQL surface and the HTTP side-channel are already listening:
/// [`Self::flight_addr`] / [`Self::health_addr`] report the ACTUAL bound
/// addresses (the real ports even when the config requested `:0`), and neither
/// port is released before [`Self::serve_with_shutdown`] serves on the very same
/// listeners — there is no release-then-rebind window a concurrent binder could
/// slip into.
pub struct BoundServer {
    grpc: BoundChain,
    health_listener: TcpListener,
    health_addr: SocketAddr,
    health_router: Router,
    /// The bound internal peer listener with its layer-free routes and the
    /// registry its own `MetricsLayer` is built from at serve time. `None`
    /// when `[server] peer_bind` is unset.
    peer: Option<(TcpListener, tonic::service::Routes, Arc<MetricsRegistry>)>,
    /// The ACTUAL peer listener address (the real port for a `:0` request).
    peer_addr: Option<SocketAddr>,
    /// The engine session, kept alive past [`OssServer::bind`] so
    /// [`Self::serve_with_signals`] can release its catalog connections
    /// (including the lease keeper's own, N3) once the serve loop has fully
    /// drained — the graceful-shutdown release point a `SIGTERM`'d
    /// `jammi-server` needs so a successor process can open the same
    /// SQLite catalog directory immediately, exactly as the embedded
    /// engine's `close()` does.
    session: Arc<InferenceSession>,
    /// The embedded job worker guard, hoisted out of the chain at bind (D3)
    /// so the two-mode shutdown owns it: `None` when `[worker] enabled =
    /// false`.
    worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
    /// The readiness probe, so a shutdown can flip `/readyz` to 503.
    readiness: Arc<ReadinessProbe>,
}

/// How [`BoundServer::serve_with_signals`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownOutcome {
    /// DRAIN completed: the gRPC surface drained and the worker (if any)
    /// finished its in-flight job and was joined.
    Drained {
        /// `true` when a worker loop was actually stopped and joined;
        /// `false` on a worker-less process.
        worker_joined: bool,
    },
    /// RELEASE completed and its evidence CONFIRMS every held lease was
    /// handed back — P-RELEASE: the release call returned `Ok`, its
    /// per-hold pass (P-2B, see `release_outcome`'s doc) was observed with
    /// no per-hold failure, its loop (if any) is certainly no longer able to
    /// claim (P-2F), and its authoritative sweep's `jobs`/`building` fields
    /// both read `Some`. `main` exits the process at once (a detached
    /// training thread may still be running).
    Released,
    /// RELEASE was attempted but its own evidence does NOT confirm every
    /// held lease was handed back — ANY determinant of P-RELEASE was
    /// unobserved or reported failure: the release call itself errored, its
    /// per-hold pass could not be confirmed to run or found a hold it
    /// attempted and failed to release, its loop's terminal state was
    /// neither certainly resolved nor genuinely observed, or one of its
    /// sweep statements failed (`jammi_ai::fine_tune::worker::
    /// ReleaseSweep::jobs`/`building` read `None`, a swallowed statement
    /// failure, never fabricated into a claim of success). Whether a lease
    /// still gets handed back is CONDITIONAL on which determinant degraded,
    /// never universal (contract `CONTRACT-OPS-fix6.md`, round 6, correcting
    /// round 5's operator sentence): when it is a sweep statement for that
    /// lease's own table that read `None`, the lease truly was not written
    /// and falls to the expiry path, so a successor reclaims it within one
    /// lease window rather than one idle poll — but when the sweep itself
    /// confirms (both fields `Some`) and only the hold-observation or
    /// stop-witness evidence is missing (`Unobserved` holds, or
    /// `stop_witnessed == false`), the `UPDATE` that hands the lease back
    /// already ran and committed, so the lease IS NULL with `releases + 1`
    /// and a successor claims it within one idle poll at no attempt cost —
    /// the evidence gap is about hold bookkeeping or loop synchronization,
    /// not about whether the row was released. `main` still exits the
    /// process at once (exit code 3, distinct from [`Self::Released`]'s 0)
    /// — R6: an `Err` here must never propagate through the normal return
    /// path, which would hang on a detached trainer past its grace period
    /// and turn a degraded release into a kill.
    ReleaseDegraded,
}

/// Whether a [`jammi_ai::fine_tune::worker::ReleaseSweep`]'s own evidence
/// supports believing its two RELEASE statements actually ran: `jobs` and
/// `building` both `Some`, never a swallowed statement error read back as
/// `None` (`release_sweep`'s own doc — a failed statement is logged and
/// folded into `Ok(None)`, never an `Err`, so `None` is the only signal
/// left that it failed).
fn sweep_confirms_release(sweep: &jammi_ai::fine_tune::worker::ReleaseSweep) -> bool {
    sweep.jobs.is_some() && sweep.building.is_some()
}

/// One RELEASE call's raw result, from either surface that can issue it —
/// the input to [`release_outcome`].
enum ReleaseAttempt {
    /// `EmbeddedWorker::release_and_stop`: two sweeps (2c, 2g); the FINAL
    /// one (2g, unconditional, idempotent, run after the loop has stopped)
    /// is the authoritative word on what actually got released, so it is
    /// what evidence is read from — sweep #1 can legitimately read `None`
    /// on a transient race and still have the release land via #2.
    Worker(Result<jammi_ai::fine_tune::worker::ReleaseReport, jammi_db::error::JammiError>),
    /// `InferenceSession::release_job_leases`: the worker-less peer, one
    /// sweep, no loop (so P-2F is vacuously satisfied — there is no further
    /// claim any loop could make).
    SessionOnly(
        Result<
            (
                jammi_ai::fine_tune::worker::HoldReleaseOutcome,
                jammi_ai::fine_tune::worker::ReleaseSweep,
            ),
            jammi_db::error::JammiError,
        >,
    ),
}

/// R6's ONE outcome-evidence helper: the preload-exit early return and the
/// main RELEASE arm both call this rather than each constructing
/// [`ShutdownOutcome::Released`] merely from "the RELEASE signal fired".
///
/// Implements **P-RELEASE** (contract `CONTRACT-OPS-fix3.md`):
/// [`ShutdownOutcome::Released`] iff EVERY determinant of "this RELEASE
/// handed back every lease this instance held" was both observed and
/// reports success — [`ShutdownOutcome::ReleaseDegraded`] whenever ANY
/// determinant is unobserved or reports failure. The determinants:
///
/// * **P-2B** — the call itself succeeded (`Ok`) AND its keeper pass
///   [`jammi_ai::fine_tune::worker::HoldReleaseOutcome::confirms_release`]
///   (observed, no per-hold failure).
/// * **P-2F** (`Worker` only) — [`jammi_ai::fine_tune::worker::ReleaseReport::stop_witnessed`]:
///   no further claim by this loop can land. Vacuously true on
///   `SessionOnly` (no loop exists).
/// * **P-EXCLUSION** — only the FINAL sweep (`sweep_two` / the `SessionOnly`
///   sweep) is a determinant; [`sweep_confirms_release`] is never called on
///   `sweep_one`, whose `None` is legitimate on a transient race superseded
///   by the idempotent final sweep.
///
/// Never panics and never itself surfaces an `Err` — a degraded release is
/// a value, not a propagated failure (see [`ShutdownOutcome::ReleaseDegraded`]'s
/// doc).
fn release_outcome(attempt: ReleaseAttempt) -> ShutdownOutcome {
    match attempt {
        ReleaseAttempt::Worker(Ok(report)) => {
            tracing::info!(
                ?report.loop_state,
                ?report.holds,
                stop_witnessed = report.stop_witnessed,
                ?report.sweep_one,
                ?report.sweep_two,
                "RELEASE: leases handed back; the loop is stopped"
            );
            if report.holds.confirms_release()
                && report.stop_witnessed
                && sweep_confirms_release(&report.sweep_two)
            {
                ShutdownOutcome::Released
            } else {
                ShutdownOutcome::ReleaseDegraded
            }
        }
        ReleaseAttempt::Worker(Err(e)) => {
            tracing::error!(error = %e, "RELEASE: the worker release failed");
            ShutdownOutcome::ReleaseDegraded
        }
        ReleaseAttempt::SessionOnly(Ok((holds, sweep))) => {
            tracing::info!(
                ?holds,
                ?sweep,
                "RELEASE on a worker-less process: nothing loop-claimed to release"
            );
            if holds.confirms_release() && sweep_confirms_release(&sweep) {
                ShutdownOutcome::Released
            } else {
                ShutdownOutcome::ReleaseDegraded
            }
        }
        ReleaseAttempt::SessionOnly(Err(e)) => {
            tracing::error!(
                error = %e,
                "RELEASE on a worker-less process: releasing leases failed"
            );
            ShutdownOutcome::ReleaseDegraded
        }
    }
}

impl BoundServer {
    /// The ACTUAL address the gRPC + Flight SQL surface is bound to. When the
    /// config requested `:0`, this is the ephemeral port the kernel assigned —
    /// resolved at [`OssServer::bind`] and held.
    pub fn flight_addr(&self) -> SocketAddr {
        self.grpc.addr()
    }

    /// The ACTUAL address the HTTP side-channel (`/healthz`, `/readyz`,
    /// `/metrics`) is bound to. When the config requested `:0`, this is the
    /// ephemeral port the kernel assigned.
    pub fn health_addr(&self) -> SocketAddr {
        self.health_addr
    }

    /// Whether this server owns an embedded worker guard (`[worker]
    /// enabled`).
    pub fn has_worker(&self) -> bool {
        self.worker.is_some()
    }

    /// A weak handle on the embedded worker's shared state (`None` without a
    /// worker) — what the gauges and liveness read; exposed for oracles.
    pub fn worker_shared(&self) -> Option<Weak<jammi_ai::fine_tune::worker::WorkerShared>> {
        self.worker.as_ref().map(|w| w.shared())
    }

    /// The ACTUAL address the internal `PeerService` listener is bound to —
    /// `Some` iff `[server] peer_bind` was set (the real port for a `:0`
    /// request).
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.peer_addr
    }

    /// Serve both halves on the already-bound listeners until `shutdown`
    /// resolves, then DRAIN: the gRPC surface drains and — concurrently,
    /// gated on the same signal so the worker is never stopped at t = 0 —
    /// the embedded worker finishes its in-flight job and is joined. A
    /// RELEASE is never requested on this entry; [`Self::serve_with_signals`]
    /// is the two-signal form.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        let (drain_tx, drain_rx) = watch::channel(false);
        // Dropped with the task: a closed release sender reads as "never".
        let (_release_tx, release_rx) = watch::channel(false);
        tokio::spawn(async move {
            shutdown.await;
            let _ = drain_tx.send(true);
        });
        self.serve_with_signals(drain_rx, release_rx)
            .await
            .map(|_| ())
    }

    /// Serve both halves until a signal arrives on `drain_rx` (DRAIN) or
    /// `release_rx` (RELEASE) — the two-mode shutdown, PostgreSQL's mapping
    /// (SIGTERM smart, SIGINT fast). A closed sender on either watch reads as
    /// "that signal never comes".
    ///
    /// **DRAIN** (`drain_rx` → `true`): `/readyz` flips to 503 `"draining"`;
    /// tonic's graceful shutdown closes the listener and finishes in-flight
    /// requests while every idle `WaitJob`/`Subscribe` stream is ended with a
    /// typed `UNAVAILABLE` "server draining" trailer (counted under
    /// `jammi_grpc_refused_total{reason="draining"}`); concurrently the
    /// worker's `begin_drain` + `stop_and_join` lets the in-flight job finish
    /// (keeper alive, every epoch bundle lands) and joins the loop. Then the
    /// health task stops, the session closes, OTLP flushes, and this returns
    /// [`ShutdownOutcome::Drained`]. No engine-side timeout: the runtime's
    /// grace period (`terminationGracePeriodSeconds`, `stop_grace_period`)
    /// bounds it. An in-flight UNARY (including an inline `run_now`) is
    /// bounded only by that grace.
    ///
    /// **RELEASE** (`release_rx` → `true`, at any time — it races the whole
    /// DRAIN sequence): the gRPC serve future is dropped (connections
    /// severed; an inline `run_now` future dies here and its row is left to
    /// the inline liveness reclaim), then `EmbeddedWorker::release_and_stop`
    /// hands every lease back and stops the loop (or, with no worker,
    /// `InferenceSession::release_job_leases`, a no-op by construction), then
    /// the same health/session/OTLP tail, and this returns
    /// [`ShutdownOutcome::Released`] — the binary exits the process at once.
    pub async fn serve_with_signals(
        self,
        drain_rx: watch::Receiver<bool>,
        release_rx: watch::Receiver<bool>,
    ) -> Result<ShutdownOutcome, ServerError> {
        let BoundServer {
            grpc,
            health_listener,
            health_addr,
            health_router,
            peer,
            peer_addr,
            session,
            worker,
            readiness,
        } = self;
        tracing::info!(
            address = %health_addr,
            "HTTP side-channel listening (/healthz, /readyz, /metrics)"
        );

        // The health side-channel stays up through a DRAIN (D13) — it is
        // signalled only at the very end, on its own channel.
        let (health_stop_tx, health_stop_rx) = oneshot::channel::<()>();
        // The peer listener: its own tonic server over the pre-bound
        // `TcpListener` (a `TcpListenerStream` — deliberately without the
        // public chain's nodelay tuning: internal unary RPC), carrying only
        // the `MetricsLayer` so `jammi_grpc_requests_total` /
        // `jammi_peer_requests_total{rpc}` count peer calls like any
        // `/jammi.v1.*` request. No tenant layer, no gRPC-web framing, no
        // `[server.limits]` stack — its clients are coordinators (I-PEER).
        // It stays up through a DRAIN exactly like the health side-channel
        // (D13's analogue: a coordinator's fan-out to this owner is never cut
        // early) and is signalled only at the very end, on its own oneshot.
        let (peer_stop_tx, peer_stop_rx) = oneshot::channel::<()>();
        let peer_task = peer.map(|(listener, routes, registry)| {
            tokio::spawn(async move {
                tracing::info!(
                    address = ?peer_addr,
                    "peer listener listening (jammi.v1.peer.PeerService)"
                );
                Server::builder()
                    .layer(MetricsLayer::new(registry))
                    .add_routes(routes)
                    .serve_with_incoming_shutdown(
                        tokio_stream::wrappers::TcpListenerStream::new(listener),
                        async move {
                            let _ = peer_stop_rx.await;
                        },
                    )
                    .await
                    .map_err(ServerError::from)
            })
        });

        let health_task = tokio::spawn(async move {
            axum::serve(health_listener, health_router)
                .with_graceful_shutdown(async move {
                    let _ = health_stop_rx.await;
                })
                .await
                .map_err(ServerError::from)
        });

        // Warm-before-ready: preload every listed model inline, `/readyz`
        // 503 "preloading i/n" meanwhile and the claim loop parked at its
        // gate (`workers.state = warming`), raced against both signals. A
        // signal aborts the preload and the server never serves; a preload
        // error is a startup error. On both exits the worker is stopped and
        // joined (its row deleted) BEFORE the session closes.
        let entries = session.inner_config().server.preload_models.clone();
        if !entries.is_empty() {
            readiness.begin_preload(entries.len());
            let preload = preload_models(&session, &readiness, &entries);
            let mut drain_wait = drain_rx.clone();
            let mut release_wait = release_rx.clone();
            let outcome = tokio::select! {
                result = preload => Some(result),
                _ = drain_wait.wait_for(|v| *v) => None,
                _ = release_wait.wait_for(|v| *v) => None,
            };
            let preempted: Option<Result<(), ServerError>> = match outcome {
                Some(Ok(())) => None,
                Some(Err(e)) => Some(Err(e)),
                None => Some(Ok(())),
            };
            if let Some(preempted) = preempted {
                readiness.begin_drain();
                // W2/R6: the outcome must reflect what THIS arm actually
                // DID, never merely which signal fired. On EITHER exit — a
                // signal preempting the preload, or the preload itself
                // erroring — the worker is still stopped and joined (its
                // row deleted) before the session closes; a preload `Err`
                // is never a release (`is_release` is `false` on it), so
                // that arm always joins, never releases. The computed
                // `outcome` is discarded on the `Err` arm below — only the
                // join's SIDE EFFECT (the row delete) matters there — and
                // is constructed only from its own evidence
                // ([`release_outcome`]) on the `Ok` arm.
                let is_release = matches!(preempted, Ok(())) && *release_rx.borrow();
                let outcome = if is_release {
                    match worker.as_ref() {
                        Some(w) => {
                            release_outcome(ReleaseAttempt::Worker(w.release_and_stop().await))
                        }
                        None => release_outcome(ReleaseAttempt::SessionOnly(
                            session.release_job_leases().await,
                        )),
                    }
                } else {
                    // The gate is closed, so the loop returns without a
                    // claim; the join orders the row's delete after the
                    // task's own upsert. `worker_joined` is the call's own
                    // `StopOutcome` (F4b), never `worker.is_some()` — that
                    // would read `true` even when the join itself errored
                    // or found nothing left to join.
                    let worker_joined = match worker.as_ref() {
                        Some(w) => match w.stop_and_join().await {
                            Ok(jammi_ai::fine_tune::worker::StopOutcome::Joined) => true,
                            Ok(jammi_ai::fine_tune::worker::StopOutcome::NothingToJoin) => false,
                            Err(e) => {
                                tracing::error!(error = %e, "preload exit: the worker join failed");
                                false
                            }
                        },
                        None => false,
                    };
                    ShutdownOutcome::Drained { worker_joined }
                };
                let early: Result<ShutdownOutcome, ServerError> = match preempted {
                    Err(e) => Err(e),
                    Ok(()) => Ok(outcome),
                };
                // R6: the tail this early return shares with the main one —
                // health side-channel, then the PEER listener (previously
                // skipped here entirely: dropping the sender starts its
                // graceful shutdown but nothing established it finished
                // before the session closed — the same divergence class as
                // the outcome itself), then the session and OTLP. FOLDED,
                // never discarded (the main tail's own `result.and(health_
                // result).and(peer_result)`): no hang risk in folding here —
                // this exit precedes the worker gate opening, so no job is
                // claimed and no detached trainer exists to wait on.
                //
                // UNREACHABLE AT THE PINNED VERSIONS, established by reading
                // both crates rather than argued (contract
                // `CONTRACT-OPS-fix6.md`, round 6, correcting round 5's
                // `CONTRACT-OPS-fix5.md`): no test drives `health_task` or
                // `peer_task` to `Err` on this exact preload-exit path
                // because, at the pinned dependency versions, NEITHER serve
                // future can return `Err` from a live socket at all — this
                // is not a tooling gap, it is a fact about the pinned
                // dependencies. `axum::serve(..).with_graceful_shutdown(..)`'s
                // `IntoFuture` is literally `{ self.run().await; Ok(()) }`
                // (`axum-0.8.8/src/serve/mod.rs:344-348`), and its
                // `Listener::accept` loop for a `TcpListener` never returns
                // on error — an EMFILE from descriptor exhaustion is logged
                // and slept on for 1s, then retried
                // (`axum-0.8.8/src/serve/listener.rs:30-36,140-158`).
                // Tonic's `Server::serve_with_incoming_shutdown` loop does
                // `Some(Err(e)) => { trace!(..); continue; }` on an accept
                // error, and its only fallible call is `MakeSvc::call`,
                // which is `future::ready(Ok(svc))`
                // (`tonic-0.14.5/src/transport/server/mod.rs:841-845,1250`).
                // Both futures return `Ok(())` on every path, so raw-fd
                // manipulation or a lowered `RLIMIT_NOFILE` — the routes a
                // prior round named and declined to build — would not have
                // produced an `Err` either; the arm is unreachable by
                // construction, not by a missing fault-injection technique.
                // The arm itself IS still reached
                // (`signal_during_preload_exits_without_serving`,
                // `release_signal_during_preload_actually_releases`,
                // `preload_of_an_unloadable_model_is_a_startup_error` all
                // enter it) — this fold is dead only for `Err`, live for
                // `Ok`, and never dead code — but the fold this comment
                // defends is defence against a FUTURE change to the pinned
                // `axum`/`tonic` versions that makes one of these serve
                // futures fallible on accept-loop exhaustion, not against a
                // producer that exists in the tree today. Whether either
                // task's join-error (`Err(join_err)`) arm just below is
                // covered by anything in this suite is UNMEASURED — a grep
                // for a test that panics either task found none — and is
                // left that way rather than asserted.
                let _ = health_stop_tx.send(());
                let health_result = match health_task.await {
                    Ok(r) => r,
                    Err(join_err) => {
                        Err(ServerError::Io(std::io::Error::other(join_err.to_string())))
                    }
                };
                let _ = peer_stop_tx.send(());
                let peer_result = match peer_task {
                    Some(task) => match task.await {
                        Ok(r) => r,
                        Err(join_err) => {
                            Err(ServerError::Io(std::io::Error::other(join_err.to_string())))
                        }
                    },
                    None => Ok(()),
                };
                session.close().await;
                crate::telemetry::flush_otlp();
                return early.and(health_result).and(peer_result).map(|()| outcome);
            }
        }
        readiness.set_warm();
        session.open_worker_gate();

        enum Arm {
            Drained {
                grpc: Result<(), ServerError>,
                worker_joined: bool,
            },
            Release,
        }

        let arm = {
            let grpc_serve = grpc.serve_with_drain(drain_rx.clone());
            let mut drain_gate = drain_rx.clone();
            let worker_ref = worker.as_ref();
            let readiness_ref = Arc::clone(&readiness);
            // The gated join half: nothing here runs until DRAIN is
            // signalled, so the worker is never stopped at t = 0.
            let gated_join = async move {
                let _ = drain_gate.wait_for(|v| *v).await;
                readiness_ref.begin_drain();
                match worker_ref {
                    Some(w) => {
                        w.begin_drain().await;
                        match w.stop_and_join().await {
                            Ok(jammi_ai::fine_tune::worker::StopOutcome::Joined) => true,
                            Ok(jammi_ai::fine_tune::worker::StopOutcome::NothingToJoin) => false,
                            Err(e) => {
                                tracing::error!(error = %e, "DRAIN: the worker join failed");
                                false
                            }
                        }
                    }
                    None => false,
                }
            };
            let drain_sequence = async { tokio::join!(grpc_serve, gated_join) };
            let mut release_rx = release_rx;
            let release_wait = async move {
                if release_rx.wait_for(|v| *v).await.is_err() {
                    std::future::pending::<()>().await;
                }
            };
            tokio::select! {
                (grpc, worker_joined) = drain_sequence => Arm::Drained { grpc, worker_joined },
                () = release_wait => Arm::Release,
            }
        };

        let (result, outcome) = match arm {
            Arm::Drained {
                grpc,
                worker_joined,
            } => (grpc, ShutdownOutcome::Drained { worker_joined }),
            Arm::Release => {
                // Step 1 already happened: `select!` dropped the drain
                // sequence, and with it the gRPC serve future — connections
                // severed. Step 2: the one release mechanism, whose outcome
                // is constructed only from its own evidence (R6:
                // [`release_outcome`]) — never surfaced as an `Err` here: a
                // degraded release still exits the process at once, exactly
                // like a confirmed one (see [`ShutdownOutcome::
                // ReleaseDegraded`]'s doc for why propagating it as an `Err`
                // would instead hang on a detached trainer past the grace
                // period).
                readiness.begin_drain();
                let outcome = match worker.as_ref() {
                    Some(w) => release_outcome(ReleaseAttempt::Worker(w.release_and_stop().await)),
                    None => release_outcome(ReleaseAttempt::SessionOnly(
                        session.release_job_leases().await,
                    )),
                };
                (Ok(()), outcome)
            }
        };

        // The tail both arms share: stop the health side-channel, release
        // the catalog (the keeper's own connection included, N3 — a
        // successor process can open the same SQLite catalog directory at
        // once), flush telemetry.
        let _ = health_stop_tx.send(());
        let health_result = match health_task.await {
            Ok(r) => r,
            Err(join_err) => Err(ServerError::Io(std::io::Error::other(join_err.to_string()))),
        };
        let _ = peer_stop_tx.send(());
        let peer_result = match peer_task {
            Some(task) => match task.await {
                Ok(r) => r,
                Err(join_err) => Err(ServerError::Io(std::io::Error::other(join_err.to_string()))),
            },
            None => Ok(()),
        };

        // Every listener has stopped accepting and finished draining
        // in-flight requests — the graceful-shutdown release point.
        // `InferenceSession::close` shuts the lease keeper (N3) down and
        // joins its dedicated thread (closing its own catalog connection)
        // before closing the shared pool, so a `SIGTERM`'d `jammi-server`
        // releases the SQLite `unix-excl` lock exactly as the embedded
        // engine's `close()` does — a successor process can open the same
        // catalog directory immediately rather than waiting out the process
        // exit.
        session.close().await;
        crate::telemetry::flush_otlp();

        result.and(health_result).and(peer_result).map(|()| outcome)
    }

    /// Serve both halves until an OS signal arrives — the binary entry
    /// point. One task owns both signal streams from this point on (tokio
    /// coalesces signals only before a stream's first poll): the first
    /// SIGTERM is DRAIN; SIGINT at any time, or any later SIGTERM, is
    /// RELEASE. Ctrl+C on a laptop therefore releases the running job and
    /// exits promptly, exactly as `kill -INT` does in a container.
    pub async fn serve(self) -> Result<ShutdownOutcome, ServerError> {
        let (drain_tx, drain_rx) = watch::channel(false);
        let (release_tx, release_rx) = watch::channel(false);
        tokio::spawn(signal_watcher(drain_tx, release_tx));
        self.serve_with_signals(drain_rx, release_rx).await
    }
}

/// Everything [`serve_grpc_chain`] needs to mount the Tonic chain: the bind
/// address, the Flight SQL context + tenant binding, the shared session store,
/// the optional trigger handles and engine session, and the resolved tier set.
///
/// Grouped into one options object (rather than a long positional argument list)
/// so callers name what they pass and the mount surface has one place to grow.
/// `OssServer` builds this from the engine session; test fixtures construct it
/// directly.
pub struct GrpcChain {
    /// Bind address for the combined gRPC + Flight SQL surface.
    pub addr: SocketAddr,
    /// Flight SQL session context.
    pub flight_ctx: SessionContext,
    /// Tenant binding the Flight SQL provider mutates per request.
    pub flight_binding: jammi_db::tenant_scope::TenantBinding,
    /// Session store shared between the `CatalogService` tenant trio (writers)
    /// and the engine-default `SessionIdTenantResolver` (reader).
    pub store: SessionStore,
    /// Trigger handles — `Some` iff the event tier is mounted.
    pub trigger: Option<crate::TriggerHandles>,
    /// Engine session backing the engine-layer services — `None` for the
    /// transport-only fixtures.
    pub engine: Option<Arc<InferenceSession>>,
    /// The tier set this chain mounts and advertises over `GetServerInfo`.
    pub tiers: TierSet,
    /// Shared metrics registry. The whole-server [`MetricsLayer`] holds it and
    /// drives the substrate counters / latency histogram from the request path;
    /// the Axum `/metrics` route reads the same registry to scrape it.
    pub metrics: Arc<MetricsRegistry>,
    /// The one tenant-binding resolver. Every engine service (and Flight SQL)
    /// binds its request's tenant through THIS resolver via the async
    /// [`crate::tenant_resolver_layer::TenantResolverLayer`] — the single binder,
    /// no interceptor path. Production and the OSS binary pass
    /// [`SessionIdTenantResolver::arc`] (the OSS-cooperative `jammi-session-id`
    /// default); a downstream composing the seam passes its own authenticating
    /// resolver to unify the composability seam with its BYO-auth seam. Only the
    /// services `assemble_grpc_chain` itself builds are bound by it — downstream
    /// services later [`AssembledChain::mount`]ed are NOT wrapped.
    pub tenant_resolver: Arc<dyn TenantResolver>,
    /// Gates `CatalogService.Reconcile`'s `all = true` cross-tenant admin pass
    /// ONLY — every other verb on this service, including `Reconcile` with
    /// `all = false`, is unaffected. The shipped default is `None`, which
    /// refuses every `all = true` request; a downstream that needs the
    /// cross-tenant pass supplies its own [`AdminAuthorizer`] here. See that
    /// trait's doc for the seam's mechanism-not-policy rationale.
    pub admin_authorizer: Option<Arc<dyn AdminAuthorizer>>,
    /// `[server.limits]` — message-size, in-flight concurrency, per-request
    /// timeout, and the two stream budgets. Applied to every service this
    /// function mounts: the message-size cap via each `*ServiceServer`'s own
    /// `max_decoding_message_size` (including Flight SQL), the rest via the
    /// [`crate::limits`] tower layer stack [`BoundChain::serve_with_shutdown`]
    /// applies at serve time. See [`crate::limits`] for the full contract.
    pub limits: jammi_db::config::LimitsConfig,
}

/// The engine's fully-assembled gRPC chain, ready for a downstream to mount
/// additional services onto before serving.
///
/// Holds a [`tonic::service::Routes`] with the engine's services pre-added
/// (Flight SQL + `CatalogService` + the tier/engine services, including
/// `AuditService`) and any resource whose lifetime must span the serve loop (the
/// embedded training worker). A downstream chains [`Self::mount`] (un-gated) or
/// [`Self::mount_tenant_scoped`] (bound by the engine's single tenant resolver)
/// to add its own services beside the engine's, then [`Self::serve`]s — or
/// splits to compose one listener of its own: [`Self::into_layered_axum_router`]
/// is the safe default (the engine's transport stack pre-applied, ready for
/// [`axum::serve()`]), and
/// [`Self::into_axum_router`] is the expert, layer-free split for nesting under a
/// listener that already frames gRPC-web.
///
/// The transport layer stack (`accept_http1` + `MetricsLayer` +
/// `TraceContextLayer` + `GrpcWebTrailersLayer` + `GrpcWebLayer`) is applied by
/// [`Self::serve`], not baked into the routes — see that method,
/// [`Self::into_layered_axum_router`],
/// and [`Self::into_axum_router`] for the seam contract each path honours.
pub struct AssembledChain {
    addr: SocketAddr,
    routes: tonic::service::Routes,
    mounted: Vec<String>,
    // The metrics handle the `MetricsLayer` needs at serve time. Carried forward
    // because the layer stack is deferred to `serve` — the outermost layer
    // observes every request by method path.
    metrics: Arc<MetricsRegistry>,
    // `[server.limits]`, needed at serve time to build the `crate::limits`
    // layer stack (the concurrency/timeout/stream-budget bounds; the
    // message-size cap was already applied per-service in
    // `assemble_grpc_chain`, above).
    limits: jammi_db::config::LimitsConfig,
    // The SAME `TenantResolverLayer` (holding the same `Arc<dyn TenantResolver>`)
    // that wraps every engine service in `assemble_grpc_chain`. Retained
    // (`#[derive(Clone)]`, a cheap `Arc` clone) so `mount_tenant_scoped` can wrap
    // a downstream service through it too — the single-binder invariant holds:
    // there is still exactly ONE resolver, never a second one forked off here.
    tenant_resolver_layer: TenantResolverLayer,
    // The embedded job worker this process runs when `[worker] enabled` is
    // `true`, held RAII for the serve loop. Owned by the chain (not the
    // assemble frame) so it survives the assemble→serve split; `serve` keeps it
    // alive across the serve future and `into_axum_router` hands it onward in
    // [`ChainParts`]. `None` when this process claims nothing.
    _worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

/// The non-routing remainder of an [`AssembledChain`] after
/// [`AssembledChain::into_axum_router`] splits the routes off: the resolved bind
/// address, the mounted-service ledger (for the downstream's startup log), the
/// engine metrics handle (so a single-listener downstream can re-apply the
/// engine's [`MetricsLayer`] on its own listener), and the job-worker guard the
/// downstream must keep alive for the lifetime of its own serve loop.
pub struct ChainParts {
    pub addr: SocketAddr,
    pub mounted: Vec<String>,
    pub metrics: Arc<MetricsRegistry>,
    /// The embedded job worker guard. The downstream MUST hold this for the
    /// lifetime of its serve loop — dropping it stops the worker and submitted
    /// jobs stop running.
    ///
    /// `None` when this process runs no claim loop — `[worker] enabled =
    /// false`, the submit-without-claiming configuration: `JobService`
    /// still serves (it is core), this process just never claims. There is
    /// then nothing for the downstream to hold and nothing for its shutdown to
    /// await.
    pub worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

impl AssembledChain {
    /// Mount a downstream service beside the engine's, **un-gated by the engine's
    /// tenant resolver**. Delegates [`tonic::service::Routes::add_service`] (by
    /// value, chainable) and records the service's `NamedService::NAME` in the
    /// mounted ledger for the startup tracing line — so the ledger cannot drift
    /// from what is actually mounted. Generic: the engine names no consumer. The
    /// service inherits the transport layer stack [`Self::serve`] applies,
    /// exactly as the engine's own services do — but NOT the tenant-binding
    /// layer, so no `SessionTenant` extension is bound on its requests.
    ///
    /// Reach for this for a service that must run BEFORE a tenant is known — a
    /// pre-auth handshake (e.g. a login/bootstrap verb) that a caller reaches
    /// without a resolvable credential yet. Use [`Self::mount_tenant_scoped`]
    /// instead for a service that wants the engine's resolved tenant bound
    /// uniformly.
    pub fn mount<S>(mut self, svc: S) -> Self
    where
        S: tonic::codegen::Service<
                tonic::codegen::http::Request<tonic::body::Body>,
                Error = std::convert::Infallible,
            > + tonic::server::NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Response: axum::response::IntoResponse,
        S::Future: Send + 'static,
    {
        self.mounted.push(S::NAME.to_string());
        self.routes = self.routes.add_service(svc);
        self
    }

    /// Mount a downstream service beside the engine's, wrapped by the engine's
    /// SAME single [`TenantResolverLayer`] every engine service is wrapped with
    /// (retained on `self` from [`assemble_grpc_chain`] — there is still exactly
    /// ONE `Arc<dyn TenantResolver>`, never a second resolver forked off here).
    /// Per request, the wrapper resolves the tenant BEFORE the service's handler
    /// runs: `Ok(scope)` binds the `SessionTenant` extension the handler reads
    /// ([`TenantScope::Tenant`](crate::grpc::session::TenantScope::Tenant) →
    /// `Some`, [`TenantScope::Global`](crate::grpc::session::TenantScope::Global)
    /// → `None`) and calls it; `Err(status)` returns that status and the handler
    /// NEVER runs — see [`crate::tenant_resolver_layer`] for the full per-request
    /// contract. The wrapper forwards the inner service's `NamedService::NAME`,
    /// so the mounted ledger and the proto route both stay keyed to the
    /// service's own name, exactly as [`Self::mount`] records it.
    ///
    /// Reach for this so a downstream service gets `SessionTenant` bound
    /// uniformly with every engine service, dropping its own per-handler
    /// resolve. Use plain [`Self::mount`] instead for a service that must stay
    /// un-gated (e.g. a pre-auth login/bootstrap handshake reached before any
    /// tenant is known).
    ///
    /// **Do NOT** mount a service here that already binds its own
    /// `SessionTenant` internally (or resolves tenant identity some other way)
    /// — the two binds would race and the later one silently wins
    /// (last-writer-wins on the request extension), which is exactly the
    /// double-bind the engine's single-binder invariant exists to prevent. Wrap
    /// a service with this method at most once, and never alongside a
    /// self-resolving service.
    ///
    /// PRE-SPLIT ONLY: this method lives on [`AssembledChain`], before
    /// [`Self::into_axum_router`] / [`Self::into_layered_axum_router`] split the
    /// routes off. [`ChainParts`] does NOT carry the layer forward — a
    /// downstream that splits to compose its own listener re-applies the
    /// (already `pub`) [`TenantResolverLayer`] itself via [`tower::Layer::layer`]
    /// before nesting its service, exactly as this method does internally.
    pub fn mount_tenant_scoped<S, ResBody>(mut self, svc: S) -> Self
    where
        S: tonic::codegen::Service<
                tonic::codegen::http::Request<tonic::body::Body>,
                Response = tonic::codegen::http::Response<ResBody>,
                Error = std::convert::Infallible,
            > + tonic::server::NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
        ResBody: Default + 'static,
        tonic::codegen::http::Response<ResBody>: axum::response::IntoResponse,
    {
        let scoped = self.tenant_resolver_layer.clone().layer(svc);
        self.mounted.push(S::NAME.to_string());
        self.routes = self.routes.add_service(scoped);
        self
    }

    /// The bind address the engine resolved from config. The downstream serves here.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The ledger of mounted service names, in mount order (engine's first, then
    /// any the downstream added via [`Self::mount`]). Read for a startup log; the
    /// ledger cannot drift from what is actually on the routes.
    pub fn mounted(&self) -> &[String] {
        &self.mounted
    }

    /// Bind the gRPC + Flight SQL listener eagerly and return a [`BoundChain`]
    /// holding it live. The listener is opened HERE, so [`BoundChain::addr`]
    /// reports the ACTUAL bound address — the real port even when the chain was
    /// assembled at an ephemeral `:0` — and the port stays held with no release
    /// window until [`BoundChain::serve_with_shutdown`] serves on the very same
    /// listener. A caller that needs the resolved port before serving (a test
    /// harness building a client, a downstream logging its startup address)
    /// reads it off the bound handle rather than pre-binding-and-dropping a
    /// throwaway listener.
    ///
    /// `nodelay` is set to match the [`tonic::transport::Server`] default the
    /// addr-based serve path applies, so the served connections behave
    /// identically to a chain served straight from a configured address.
    pub async fn bind(self) -> Result<BoundChain, ServerError> {
        let listener = TcpListener::bind(self.addr).await?;
        let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
        let addr = incoming.local_addr()?;
        Ok(BoundChain {
            incoming,
            addr,
            routes: self.routes,
            mounted: self.mounted,
            metrics: self.metrics,
            limits: self.limits,
            _worker: self._worker,
        })
    }

    /// Serve the assembled chain (engine core + any downstream-mounted services)
    /// until `shutdown` resolves. A thin composition of [`Self::bind`] +
    /// [`BoundChain::serve_with_shutdown`] — binds the listener, then serves on
    /// it; the job-worker guard stays alive for the whole serve loop. The
    /// transport layer stack is applied by [`BoundChain::serve_with_shutdown`].
    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        self.bind().await?.serve_with_shutdown(shutdown).await
    }

    /// Split into a plain, **LAYER-FREE** [`axum::Router`] (via
    /// [`tonic::service::Routes::into_axum_router`]) plus the [`ChainParts`]
    /// remainder — the EXPERT split, for a downstream that nests the engine's
    /// gRPC routes UNDER a listener that ALREADY carries its own gRPC-web layer
    /// stack. Most single-listener consumers want [`Self::into_layered_axum_router`]
    /// instead (the safe default — see below).
    ///
    /// SEAM CONTRACT: the returned router carries NO transport layers. The
    /// gRPC-web framing + trailer-repair + metrics layers are applied by
    /// [`Self::serve`] on the serve path, NOT baked into the routes. A downstream
    /// that serves this router on its OWN listener must therefore re-apply the
    /// full stack itself — [`GrpcWebLayer`] + [`GrpcWebTrailersLayer`] + the
    /// engine's [`MetricsLayer`] (via [`ChainParts::metrics`]) — or gRPC-web
    /// clients break: a trailers-only error response would miss the in-body
    /// trailer frame.
    ///
    /// PATH-SPECIFIC LAYERING: `accept_http1(true)` is a
    /// [`tonic::transport::Server`] builder method and applies ONLY on the
    /// [`Self::serve`] path — there is no `accept_http1` to call on the axum
    /// path; HTTP/1 is implicit in [`axum::serve()`]. On axum, re-apply the layers
    /// with `Router::layer` in inner→outer call order (axum runs the LAST
    /// `.layer` call as the outermost service, the inverse of the tonic builder),
    /// i.e. `.layer(GrpcWebLayer::new()).layer(GrpcWebTrailersLayer::new())
    /// .layer(MetricsLayer::new(metrics))`. This is exactly what
    /// [`Self::into_layered_axum_router`] does for you — reach for the layer-free
    /// split only when your outer listener already frames gRPC-web (re-applying
    /// here would double-frame).
    ///
    /// The router also carries a gRPC `unimplemented` fallback (from
    /// `Routes`' default), so a composing consumer must nest it under a path
    /// prefix or reconcile its own fallback, NOT blind-`.merge()` it.
    ///
    /// The downstream must hold [`ChainParts`] (specifically its job-worker
    /// guard) alive for the lifetime of its own serve loop.
    pub fn into_axum_router(self) -> (axum::Router, ChainParts) {
        let router = self.routes.into_axum_router();
        let parts = ChainParts {
            addr: self.addr,
            mounted: self.mounted,
            metrics: self.metrics,
            worker: self._worker,
        };
        (router, parts)
    }

    /// Split into a **layered** [`axum::Router`] plus the [`ChainParts`]
    /// remainder — the SAFE DEFAULT for a downstream that composes ONE listener
    /// of its own. The returned router is ready to hand to [`axum::serve()`]
    /// DIRECTLY: it carries the engine's full transport contract, so the consumer
    /// re-applies nothing.
    ///
    /// PARTIAL LAYER STACK — this does NOT apply the SAME stack [`Self::serve`]
    /// applies, despite this method's name; it applies only the gRPC-web +
    /// metrics + trace-context framing (the whole-server [`MetricsLayer`],
    /// outermost, observing every method path, wrapping
    /// [`crate::trace_context_layer::TraceContextLayer`] — opens one span per
    /// request, continuing an incoming W3C `traceparent` — wrapping
    /// [`GrpcWebTrailersLayer`] — the trailers-only error repair — wrapping
    /// [`GrpcWebLayer`], gRPC-web framing, wrapping the routes). axum runs the
    /// LAST `.layer` call as the OUTERMOST service — the
    /// inverse of the tonic [`tonic::transport::Server`] builder, where the FIRST
    /// `.layer` is outermost — so the calls are ordered inner→outer here to land
    /// the same outermost→innermost gRPC-web/metrics stack `serve` builds.
    /// `accept_http1` has no axum analogue: HTTP/1 is implicit in
    /// [`axum::serve()`].
    ///
    /// MISSING, relative to [`Self::serve`]: the WHOLE `[server.limits]`
    /// request-bounds stack — [`crate::limits::RefusalStatusLayer`],
    /// [`crate::limits::GlobalConcurrencyLimitLayer`],
    /// [`crate::limits::PerConnectionLimitLayer`], and
    /// [`crate::limits::MethodClassLayer`] (the message-size cap is unaffected
    /// — it is applied per-service in `assemble_grpc_chain`, before this split,
    /// so it rides along either path). `self.limits` is retained on
    /// [`AssembledChain`] but not consulted here. A downstream serving this
    /// router on ITS OWN listener therefore gets NO in-flight/per-connection/
    /// wait-timeout/stream-budget enforcement from this stack unless it
    /// re-applies `crate::limits`'s layers itself — the per-connection ones in
    /// particular depend on tonic's own [`tonic::transport::server::
    /// TcpConnectInfo`] request extension, which this axum path does not
    /// independently guarantee is populated the same way, so re-applying them
    /// blind here (rather than leaving this an explicit, documented gap for the
    /// downstream to close with its own connection-info wiring) risks a
    /// SILENTLY inert limit — worse than the honest gap this doc now states.
    ///
    /// ERGONOMIC GUARANTEE: the returned value is a plain `axum::Router` (state
    /// `()`, request body [`axum::body::Body`]) that [`axum::serve()`] accepts with
    /// no further ceremony. The layer stack rewrites the response body type; that
    /// normalization is resolved INTERNALLY (the layered routes are re-nested
    /// under a fresh [`axum::Router`]), so the consumer needs no
    /// `Router::<()>::new().merge(...)` re-nest of its own.
    ///
    /// Prefer this over [`Self::into_axum_router`] unless you are nesting under a
    /// listener that ALREADY frames gRPC-web — that expert path is layer-free
    /// precisely so it does not double-frame in that case.
    ///
    /// The downstream must hold [`ChainParts`] (specifically its job-worker
    /// guard) alive for the lifetime of its own serve loop.
    pub fn into_layered_axum_router(self) -> (axum::Router, ChainParts) {
        // The `MetricsLayer` holds a clone; the original moves into `ChainParts`
        // so the downstream can still scrape the same registry from its own
        // `/metrics` route.
        let metrics = Arc::clone(&self.metrics);
        // Apply the canonical stack in axum's inner→outer call order. axum runs
        // the last `.layer` as the outermost service, so ordering the calls
        // GrpcWebLayer → GrpcWebTrailersLayer → TraceContextLayer → MetricsLayer
        // reproduces `serve`'s outermost→innermost stack: Metrics →
        // TraceContext → GrpcWebTrailers → GrpcWebLayer → routes.
        let layered = self
            .routes
            .into_axum_router()
            .layer(GrpcWebLayer::new())
            .layer(GrpcWebTrailersLayer::new())
            .layer(TraceContextLayer::new())
            .layer(MetricsLayer::new(metrics));
        // Re-nest the layered routes under a fresh `Router` so the returned type
        // is a plain `axum::Router` whose request body is `axum::body::Body` —
        // the layer stack's response-body rewrite is absorbed here, and
        // `axum::serve` accepts the result directly with no consumer-side merge.
        let router = axum::Router::new().merge(layered);
        let parts = ChainParts {
            addr: self.addr,
            mounted: self.mounted,
            metrics: self.metrics,
            worker: self._worker,
        };
        (router, parts)
    }
}

/// An [`AssembledChain`] whose gRPC + Flight SQL listener is bound and held
/// live, ready to serve. Returned by [`AssembledChain::bind`]. The listener is
/// open from bind through serve, so [`Self::addr`] reports the ACTUAL bound
/// address (the real port even for a `:0` assembly) and the port is never
/// observably free between resolving it and serving on it.
pub struct BoundChain {
    // The bound listener, wrapped as tonic's incoming stream with the same
    // `nodelay` the addr-based serve path applies. Held from bind through serve
    // — this is the whole point: zero release-then-rebind window.
    incoming: TcpIncoming,
    addr: SocketAddr,
    // The layer-free routes, carried forward so the transport layer stack is
    // still applied at serve time (G1) rather than baked in at bind.
    routes: tonic::service::Routes,
    mounted: Vec<String>,
    metrics: Arc<MetricsRegistry>,
    limits: jammi_db::config::LimitsConfig,
    // The embedded job worker guard, held RAII across the serve loop — its
    // lifetime spans bind → serve, exactly as it did on `AssembledChain`.
    _worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

impl BoundChain {
    /// The ACTUAL address the listener is bound to. For a chain assembled at
    /// `:0`, this is the ephemeral port the kernel assigned — resolved at
    /// [`AssembledChain::bind`] and held live until serve.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The ledger of mounted service names, in mount order. Read for a startup
    /// log; carried through the bind unchanged.
    pub fn mounted(&self) -> &[String] {
        &self.mounted
    }

    /// Take the embedded worker guard out of the chain, so a caller that
    /// needs it to outlive the gRPC serve future (the server's two-mode
    /// shutdown) owns it. After this the chain's own serve paths hold no
    /// worker and their `Drop` stops nothing. `None` when this process runs
    /// no claim loop, or when it was already taken.
    pub fn take_worker(&mut self) -> Option<jammi_ai::fine_tune::worker::EmbeddedWorker> {
        self._worker.take()
    }

    /// Serve the bound chain on its already-open listener until `shutdown`
    /// resolves. Consumes `self`, keeping the training-worker guard alive for
    /// the whole serve loop.
    ///
    /// The transport layers apply HERE, in this order (outermost first):
    /// `accept_http1(true)` then `MetricsLayer` (observes every request by
    /// method path before routing) then `TraceContextLayer` (opens one span
    /// per request, continuing an incoming W3C `traceparent` — see
    /// `crate::trace_context_layer`) then `GrpcWebTrailersLayer` (wraps
    /// `GrpcWebLayer`, repairing the trailers-only error response into the
    /// in-body trailer frame a gRPC-web client requires) then `GrpcWebLayer`
    /// then the `[server.limits]` request-bounds stack
    /// ([`crate::limits::RefusalStatusLayer`] →
    /// [`crate::limits::GlobalConcurrencyLimitLayer`] →
    /// [`crate::limits::PerConnectionLimitLayer`] →
    /// [`crate::limits::MethodClassLayer`] — see [`crate::limits`] for the
    /// full contract and the N4/N5 rationale for this exact position, inside
    /// the gRPC-web layers). Every service mounted via [`AssembledChain::mount`],
    /// engine or downstream, inherits every one of these with no per-service
    /// opt-in — including a downstream's own mounted service, which is
    /// deliberate: the request-bounds refusal is a whole-listener property,
    /// not an engine-only one.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        let (drain_tx, drain_rx) = watch::channel(false);
        tokio::spawn(async move {
            shutdown.await;
            let _ = drain_tx.send(true);
        });
        self.serve_with_drain(drain_rx).await
    }

    /// [`Self::serve_with_shutdown`] on a drain watch: tonic's graceful
    /// shutdown (listener closed, in-flight requests finished) begins when
    /// `drain_rx` reads `true`, and the same watch feeds
    /// [`crate::limits::MethodClassLayer`]'s stream ender, so an idle
    /// `WaitJob`/`Subscribe` stream is ended with `UNAVAILABLE` "server
    /// draining" instead of holding the drain open forever. A closed sender
    /// reads as "never drain".
    pub async fn serve_with_drain(
        self,
        drain_rx: watch::Receiver<bool>,
    ) -> Result<(), ServerError> {
        tracing::info!(
            "gRPC chain ({}) listening on {}",
            self.mounted.join(" + "),
            self.addr
        );
        // The layer stack is deferred to here (G1): holding the post-layer
        // `Router<L>` would leak the concrete `Stack<…>` layer types into
        // `BoundChain`. `Routes` is the layer-free accumulation point;
        // `add_routes` attaches it behind the stack at serve time, then serves
        // on the pre-bound listener via `serve_with_incoming_shutdown`.
        let refusal_metrics = Arc::clone(&self.metrics);
        let drain_metrics = Arc::clone(&self.metrics);
        let limits = self.limits;
        let mut shutdown_rx = drain_rx.clone();
        let shutdown = async move {
            if shutdown_rx.wait_for(|v| *v).await.is_err() {
                std::future::pending::<()>().await;
            }
        };
        let mut server = Server::builder()
            .accept_http1(true)
            .layer(MetricsLayer::new(self.metrics))
            .layer(TraceContextLayer::new())
            .layer(GrpcWebTrailersLayer::new())
            .layer(GrpcWebLayer::new())
            .layer(crate::limits::RefusalStatusLayer::new(refusal_metrics))
            .layer(crate::limits::GlobalConcurrencyLimitLayer::new(
                limits.max_in_flight,
            ))
            .layer(crate::limits::PerConnectionLimitLayer::new(
                limits.max_in_flight_per_connection,
            ))
            .layer(
                crate::limits::MethodClassLayer::new(&limits).with_drain(drain_rx, drain_metrics),
            );
        server
            .add_routes(self.routes)
            .serve_with_incoming_shutdown(self.incoming, shutdown)
            .await
            .map_err(ServerError::from)
        // `self._worker` (if not taken) is dropped here, after the serve
        // future resolves — its RAII lifetime spans the whole serve loop.
    }
}

/// Assemble the engine's gRPC chain from `chain` **without serving it**, so a
/// downstream can [`AssembledChain::mount`] additional services onto the
/// engine's fully-assembled core chain before serving. This is the composability
/// seam.
///
/// **Always mounted** (the core tier + the Flight SQL transport): Flight SQL and
/// the control-plane `CatalogService` (its engine-free tenant trio +
/// `GetServerInfo` answer even when no engine is mounted; its catalog /
/// lifecycle verbs are backed by `engine` when present). When `engine` is
/// `Some`, the core data-plane services also mount: `EmbeddingService`,
/// `InferenceService`, `PipelineService`, `AuditService`, and
/// `JobService` (the job submission surface). These are the serve-path
/// primitives every deployment needs.
///
/// An engine-backed chain also spawns the embedded job worker, unless
/// `chain.engine`'s `[worker] enabled` is `false`: that key decides whether
/// THIS process claims queued jobs, and it does NOT change what is mounted or
/// advertised (`JobService` serves either way, so an `enabled = false`
/// deployment still accepts submissions and just leaves them `queued` for
/// whichever process does claim).
///
/// **Mounted by tier** (only when `tiers` selected them):
/// - `EvalService` ← [`ServiceTier::Eval`]
/// - `TriggerService` ← [`ServiceTier::Event`], driven by `trigger` being
///   `Some` (the caller derives the handles iff the event tier is mounted)
///
/// `engine` and `trigger` are `Option` so the gRPC-Web / control-plane-only
/// fixtures (which construct no `InferenceSession`) can mount just the
/// transport + core handshake. The `tiers` argument is what the
/// `CatalogService.GetServerInfo` handshake advertises, so it must agree with
/// what is actually mounted — the caller is responsible for that agreement
/// (production goes through [`OssServer`], which derives both from one config).
///
/// The engine mounts NO `LifecycleService` — the `jammi.v1.lifecycle` contract
/// is answered by a platform server, not the OSS engine; an OSS server answers
/// `UNIMPLEMENTED` for those verbs.
pub fn assemble_grpc_chain(chain: GrpcChain) -> Result<AssembledChain, ServerError> {
    let GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store,
        trigger,
        engine,
        tiers,
        metrics,
        tenant_resolver,
        admin_authorizer,
        limits,
    } = chain;

    // `[server.limits].max_message_bytes`, applied to every mounted service
    // below (including Flight SQL) via tonic's own per-service
    // `max_decoding_message_size` — see `crate::limits`'s N5 rustdoc for why
    // this is NOT a tower layer. `u64` -> `usize`: a value that would not fit
    // `usize` (only reachable on a 32-bit target with an absurd config) saturates
    // to `usize::MAX` (effectively unbounded) rather than panicking or silently
    // truncating to a SMALLER, surprising cap.
    let max_message_bytes: usize = usize::try_from(limits.max_message_bytes).unwrap_or(usize::MAX);

    // Flight SQL — MUST-FIX 2: cover the `db.sql` lane through the SAME resolver
    // as the gRPC plane. The provider resolves each query's scope and binds it,
    // closing the cross-transport bypass where an authenticated gRPC plane would
    // still let Flight bind from an unauthenticated header (#220).
    let provider = TenantBoundProvider::new(
        flight_ctx.state(),
        flight_binding,
        Arc::clone(&tenant_resolver),
    );
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = FlightServiceServer::new(flight).max_decoding_message_size(max_message_bytes);

    // The single binder. One `TenantResolverLayer` (holding the one resolver)
    // wraps every engine service uniformly — no branch, no separate interceptor,
    // so nothing can double-bind or clobber the resolved tenant. MUST-FIX 1 — the
    // wrapping applies only to the services THIS function builds; downstream
    // services mounted via `AssembledChain::mount` (a platform's own pre-auth
    // `Login`/`Bootstrap`, etc.) stay un-gated by the engine resolver.
    let resolver_layer = TenantResolverLayer::new(tenant_resolver);

    // Bind one engine service onto `routes` under the resolver layer. `$server`
    // is the bare `*ServiceServer::new(inner)`; the layer forwards its
    // `NamedService::NAME` so tonic routing keeps it. Every engine service gets
    // the SAME `max_message_bytes` inbound decode cap (`[server.limits]`).
    macro_rules! mount_engine {
        ($routes:expr, $mounted:expr, $name:literal, $server:expr) => {{
            $routes = $routes.add_service(
                resolver_layer.layer($server.max_decoding_message_size(max_message_bytes)),
            );
            $mounted.push($name.to_string());
        }};
    }

    // Accumulate the services layer-free on a `tonic::service::Routes`. The
    // transport layer stack is deferred to `AssembledChain::serve` (G1): holding
    // the post-`add_service` `Router<L>` would leak the concrete layer-stack type
    // into the seam and cannot grow in place (its `add_service` is by-value with
    // no `Default`). `Routes` is the composition point tonic provides for exactly
    // this — every service mounted onto it, engine or downstream, then inherits
    // the gRPC-web framing + trailer repair the serve path applies. Flight SQL is
    // NOT wrapped by the layer: its `db.sql` lane binds inside the
    // `TenantBoundProvider` (threaded with the same resolver above).
    let mut routes = tonic::service::Routes::new(flight_svc);
    let mut mounted = vec!["Flight SQL".to_string()];

    // The control plane: one `CatalogService` on the always-present core tier.
    // Its engine-free verbs (the tenant trio + `GetServerInfo`) ride the
    // `SessionStore` + `TierSet`, so it mounts even on an engine-light
    // deployment; its catalog / lifecycle verbs delegate to the shared engine
    // when one is present (`engine.clone()` here, with the original moved into
    // the engine-services block below).
    mount_engine!(
        routes,
        mounted,
        "CatalogService",
        CatalogServiceServer::new(CatalogServer::new(
            store,
            tiers.clone(),
            engine.clone(),
            admin_authorizer,
        ))
    );

    // Event tier: TriggerService. Driven by the caller having supplied handles
    // (it does so iff the event tier is mounted).
    if let Some(handles) = trigger {
        mount_engine!(
            routes,
            mounted,
            "TriggerService",
            TriggerServiceServer::new(TriggerServer::new(
                handles.topic_repo,
                handles.publisher,
                handles.subscriber,
            ))
        );
    }

    // The embedded job worker this process runs when `[worker] enabled`. Moved
    // into the returned `AssembledChain` so it outlives the assemble frame and
    // spans the serve loop (RAII). A chain without an engine never sets it.
    let mut worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker> = None;

    if let Some(session) = engine {
        // Core tier engine services: always mounted when an engine is present.
        mount_engine!(
            routes,
            mounted,
            "EmbeddingService",
            EmbeddingServiceServer::new(EmbeddingServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "InferenceService",
            InferenceServiceServer::new(InferenceServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "PipelineService",
            PipelineServiceServer::new(PipelineServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "AuditService",
            AuditServiceServer::new(AuditServer::new(Arc::clone(&session)))
        );

        // Eval tier: EvalService.
        if tiers.contains(ServiceTier::Eval) {
            mount_engine!(
                routes,
                mounted,
                "EvalService",
                EvalServiceServer::new(EvalServer::new(Arc::clone(&session)))
            );
        }

        // Core: JobService — the durable job submission/status/wait surface
        // (replaces TrainingService). `SubmitJob` carries all three
        // training kinds — fine-tune, graph fine-tune, context-predictor.
        // Submission is always mounted; whether THIS process also runs the
        // claim loop is configuration, not a tier and not a build feature:
        // `[worker] enabled` (default `true`). The surface and what
        // `GetServerInfo.services` advertises are the same either way, because
        // the service IS mounted — so a `worker.enabled = false` deployment
        // still accepts submissions; it just never claims them. The embedded
        // arm reads the same key off the same config, so a wire deployment and
        // an in-process one answer the question identically.
        if session.inner_config().worker.enabled {
            // Start the worker that runs submitted jobs of every compiled kind:
            // a "GPU worker pool" is just N processes claiming from the shared
            // catalog, and this server runs one of them. `spawn` borrows
            // `session` before it is moved into `JobServer::new`; the
            // worker is stored in `AssembledChain` so it stops when the serve
            // future resolves.
            worker = Some(jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(
                &session,
            )?);
            tracing::info!(
                worker_enabled = true,
                "JobService mounted; this process claims queued jobs"
            );
        } else {
            // No worker exists to stop, so shutdown has nothing extra to await:
            // `AssembledChain::_worker` stays `None` and its drop is a no-op.
            tracing::info!(
                worker_enabled = false,
                "JobService mounted; this process does not claim jobs"
            );
        }
        mount_engine!(
            routes,
            mounted,
            "JobService",
            JobServiceServer::new(JobServer::new(session))
        );
    }

    Ok(AssembledChain {
        addr,
        routes,
        mounted,
        metrics,
        limits,
        // The SAME layer `mount_engine!` wrapped every engine service with above
        // — retained so `AssembledChain::mount_tenant_scoped` can wrap a
        // downstream service through the identical single resolver.
        tenant_resolver_layer: resolver_layer,
        _worker: worker,
    })
}

/// Build and serve the engine's gRPC chain on `chain.addr` in one call — the
/// OSS-only convenience for a caller serving straight from a configured address.
///
/// A thin composition of the seam: `assemble_grpc_chain(chain)?.serve(...)`,
/// which binds the listener eagerly ([`AssembledChain::bind`]) and serves on it.
/// A caller that needs the RESOLVED port before serving (an ephemeral `:0` bind)
/// drives [`assemble_grpc_chain`] → [`AssembledChain::bind`] →
/// [`BoundChain::serve_with_shutdown`] directly and reads the port off the bound
/// handle. Downstreams that mount their own services go through
/// [`assemble_grpc_chain`] + [`AssembledChain::mount`] instead.
pub async fn serve_grpc_chain(
    chain: GrpcChain,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    assemble_grpc_chain(chain)?.serve(shutdown).await
}

/// Preload every `[server] preload_models` entry into the session's model
/// cache, in order: `ModelSource::parse(id)`; the task is the entry's
/// explicit one, else the catalog's `models` row for the id, else a typed
/// [`ServerError::Preload`] ("no models row; give { id, task }"); a load
/// failure is the same typed error. Each cached entry advances `/readyz`'s
/// "preloading i/n".
async fn preload_models(
    session: &Arc<InferenceSession>,
    readiness: &ReadinessProbe,
    entries: &[jammi_db::config::PreloadEntry],
) -> Result<(), ServerError> {
    for entry in entries {
        let source = jammi_ai::model::ModelSource::parse(&entry.id);
        let task = match entry.task {
            Some(task) => task,
            None => match session.catalog().get_model(&entry.id).await {
                Ok(Some(record)) => record.task,
                Ok(None) => {
                    return Err(ServerError::Preload {
                        id: entry.id.clone(),
                        reason: "no models row; give { id, task }".to_string(),
                    })
                }
                Err(e) => {
                    return Err(ServerError::Preload {
                        id: entry.id.clone(),
                        reason: format!("models row lookup failed: {e}"),
                    })
                }
            },
        };
        tracing::info!(model = %entry.id, ?task, "preloading model");
        session
            .model_cache()
            .preload(&source, task, None)
            .await
            .map_err(|e| ServerError::Preload {
                id: entry.id.clone(),
                reason: e.to_string(),
            })?;
        readiness.note_preloaded();
    }
    Ok(())
}

/// The one task that owns both OS signal streams for the process's
/// lifetime (tokio coalesces a signal only before its stream's first poll;
/// after that every delivery is an item): the first SIGTERM sends DRAIN on
/// `drain_tx`; SIGINT at any time, or any SIGTERM after the first, sends
/// RELEASE on `release_tx` and the task ends. PostgreSQL's mapping — SIGTERM
/// smart, SIGINT fast — and the reason RELEASE is reachable by a distinct
/// signal: an orchestrator sends one stop signal then SIGKILL, so a second
/// SIGTERM never arrives from it; `jammi-server release` (a preStop hook)
/// and Ctrl+C send SIGINT.
async fn signal_watcher(drain_tx: watch::Sender<bool>, release_tx: watch::Sender<bool>) {
    #[cfg(unix)]
    {
        let mut interrupt = match signal::unix::signal(signal::unix::SignalKind::interrupt()) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::error!("Failed to install SIGINT handler: {e}");
                None
            }
        };
        let mut terminate = match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::error!("Failed to install SIGTERM handler: {e}");
                None
            }
        };
        let mut draining = false;
        loop {
            let on_interrupt = async {
                match interrupt.as_mut() {
                    Some(s) => {
                        s.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            let on_terminate = async {
                match terminate.as_mut() {
                    Some(s) => {
                        s.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                () = on_interrupt => {
                    tracing::info!("SIGINT received: RELEASE — handing leases back and exiting");
                    let _ = release_tx.send(true);
                    return;
                }
                () = on_terminate => {
                    if draining {
                        tracing::info!("second SIGTERM while draining: RELEASE");
                        let _ = release_tx.send(true);
                        return;
                    }
                    draining = true;
                    tracing::info!("SIGTERM received: DRAIN — finishing in-flight work");
                    let _ = drain_tx.send(true);
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = &drain_tx;
        match signal::ctrl_c().await {
            Ok(()) => {
                tracing::info!("Ctrl+C received: RELEASE");
                let _ = release_tx.send(true);
            }
            Err(e) => tracing::error!("Failed to install Ctrl+C handler: {e}"),
        }
    }
}

#[cfg(test)]
mod audit_master_key_tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;

    /// Serializes every test below that mutates the process-global
    /// `JAMMI_AUDIT_MASTER_KEY` — a second, independently-declared lock
    /// elsewhere in this binary would race this one, which is exactly the
    /// failure mode `jammi_db::audit::key_store::test_env` (this crate has
    /// no visibility into that `pub(crate)` module, so it declares its own,
    /// same as `tests/it/grpc_mutable_topic_audit.rs::env_lock`) exists to
    /// name.
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RED at base: nothing decoded the key at boot, so a malformed value
    /// was silently accepted and audit signing died until the first write.
    /// This is the control this test now pins GREEN.
    #[test]
    fn malformed_key_refuses_startup() {
        let _guard = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var(jammi_db::audit::MASTER_KEY_ENV, "not-hex");
        let err = validate_audit_master_key(&JammiConfig::default())
            .expect_err("a present-but-malformed key must refuse to start");
        let msg = err.to_string();
        assert!(
            msg.contains(jammi_db::audit::MASTER_KEY_ENV),
            "error must name the offending env var, got: {msg}"
        );
        assert!(
            !msg.contains("not-hex"),
            "error must never echo the value, got: {msg}"
        );
        std::env::remove_var(jammi_db::audit::MASTER_KEY_ENV);
    }

    /// Absence is unchanged: the check does not tighten what was already
    /// allowed to start.
    #[test]
    fn absent_key_is_still_allowed_at_startup() {
        let _guard = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var(jammi_db::audit::MASTER_KEY_ENV);
        assert!(validate_audit_master_key(&JammiConfig::default()).is_ok());
    }

    /// A well-formed key (64 hex chars, the `openssl rand -hex 32` shape)
    /// passes the check.
    #[test]
    fn valid_key_passes_startup() {
        let _guard = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var(jammi_db::audit::MASTER_KEY_ENV, "ab".repeat(32));
        assert!(validate_audit_master_key(&JammiConfig::default()).is_ok());
        std::env::remove_var(jammi_db::audit::MASTER_KEY_ENV);
    }

    /// The DECISION this round pins: `JAMMI_AUDIT_MASTER_KEY=""` (exactly what
    /// `deploy/.env.example` ships as its placeholder, and what an unfilled
    /// Kubernetes Secret key produces) is set-EMPTY, not set-malformed — it
    /// must be treated exactly like the variable being unset, not refuse
    /// startup as a 0-byte key.
    #[test]
    fn empty_env_key_is_treated_as_absent() {
        let _guard = env_lock().lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var(jammi_db::audit::MASTER_KEY_ENV, "");
        assert!(validate_audit_master_key(&JammiConfig::default()).is_ok());
        // All-whitespace is the same "set-empty" arm, not a distinct shape.
        std::env::set_var(jammi_db::audit::MASTER_KEY_ENV, "   ");
        assert!(validate_audit_master_key(&JammiConfig::default()).is_ok());
        std::env::remove_var(jammi_db::audit::MASTER_KEY_ENV);
    }

    fn file_config(path: &std::path::Path) -> JammiConfig {
        JammiConfig {
            signing_key: SigningKeyConfig::File {
                path: path.to_path_buf(),
            },
            ..JammiConfig::default()
        }
    }

    /// `SigningKeyConfig::File`, absent path (ENOENT): allowed to start,
    /// unchanged from today's "absence is fine" policy — the File-source
    /// analogue of `absent_key_is_still_allowed_at_startup`.
    #[test]
    fn file_source_absent_path_is_allowed_at_startup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist");
        assert!(validate_audit_master_key(&file_config(&missing)).is_ok());
    }

    /// `SigningKeyConfig::File`, an existing but empty (or all-whitespace)
    /// file: the File-source analogue of `empty_env_key_is_treated_as_absent`
    /// — an unfilled Kubernetes Secret key mounts as a zero-byte file, and it
    /// must be treated as absent too.
    #[test]
    fn file_source_empty_file_is_treated_as_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("audit-master-key");
        std::fs::write(&path, "").expect("write empty file");
        assert!(validate_audit_master_key(&file_config(&path)).is_ok());

        std::fs::write(&path, "\n").expect("write whitespace-only file");
        assert!(validate_audit_master_key(&file_config(&path)).is_ok());
    }

    /// `SigningKeyConfig::File`, a readable file holding a well-formed key:
    /// passes, exactly like the env source's `valid_key_passes_startup`.
    #[test]
    fn file_source_valid_key_passes_startup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("audit-master-key");
        std::fs::write(&path, "ab".repeat(32)).expect("write valid key");
        assert!(validate_audit_master_key(&file_config(&path)).is_ok());
    }

    /// `SigningKeyConfig::File`, a readable file holding a malformed key:
    /// refuses to start, and the refusal never echoes a character of the
    /// file's content.
    #[test]
    fn file_source_malformed_key_refuses_startup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("audit-master-key");
        std::fs::write(&path, "zzzz-not-hex-zzzz").expect("write malformed key");
        let err = validate_audit_master_key(&file_config(&path))
            .expect_err("a present-but-malformed file key must refuse to start");
        let msg = err.to_string();
        assert!(
            msg.contains("hex"),
            "error must describe the format, got: {msg}"
        );
        assert!(
            !msg.contains("zzzz"),
            "error must never echo the value, got: {msg}"
        );
        assert!(
            !msg.contains("not-hex"),
            "error must never echo the value, got: {msg}"
        );
    }

    /// The require-gate polarity every `chmod` permission-fault probe in the
    /// workspace's test suites shares (esc-089 F1): `probe` performs the
    /// fault-injection premise check itself and returns `true` if the fault
    /// was BYPASSED (root, or a mode-ignoring filesystem). A bypass is
    /// normally a loud, `eprintln`'d skip; under `JAMMI_REQUIRE_POSIX_PERMS=1`
    /// (the CI lane that is SUPPOSED to run unprivileged with real POSIX
    /// permission enforcement) a bypass is instead a hard `panic!` — never a
    /// silent `return`. Each probe file carries its own copy of this wrapper
    /// in the canonical shape the kernel-oracle registry
    /// (`ci/kernel-oracle-helpers.txt`) verifies per file.
    fn chmod_bypassed(test_name: &str, probe: impl FnOnce() -> bool) -> bool {
        let bypassed = probe();
        if bypassed {
            if std::env::var_os("JAMMI_REQUIRE_POSIX_PERMS").is_some() {
                panic!(
                    "JAMMI_REQUIRE_POSIX_PERMS is set but '{test_name}' could not inject its \
                     permission fault (root, or a mode-ignoring filesystem) — the \
                     fault-injection premise this test needs does not hold; a silent skip is \
                     not acceptable here"
                );
            }
            eprintln!("{test_name}: chmod bypassed (root?) — skipping");
        }
        bypassed
    }

    /// `SigningKeyConfig::File`, an existing but UNREADABLE file (`0o000`):
    /// this is the case advisory 2 closes — `std::fs::metadata` still sees
    /// the path (so this is NOT read as absent), and the subsequent read
    /// fails, so the check fails CLOSED (refuses startup) rather than
    /// silently booting with signing dead.
    #[test]
    fn file_source_unreadable_file_refuses_startup() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("audit-master-key");
        std::fs::write(&path, "ab".repeat(32)).expect("write valid key");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");

        let bypassed = chmod_bypassed("file_source_unreadable_file_refuses_startup", || {
            std::fs::read_to_string(&path).is_ok()
        });
        if bypassed {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
                .expect("restore permissions");
            return;
        }

        let err = validate_audit_master_key(&file_config(&path))
            .expect_err("an unreadable file key must refuse to start, not boot with signing dead");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("restore permissions");
        let msg = err.to_string();
        assert!(
            msg.contains(path.to_str().unwrap()),
            "error must name the unreadable path, got: {msg}"
        );
    }
}

/// OPS round 3 (contract `CONTRACT-OPS-fix3.md`) — the non-negotiable test
/// capacity over `release_outcome`'s predicate: round 2 shipped a predicate
/// change (the parity-oracle deletion + evidence-based outcome) with ZERO
/// observing oracle, and the round-3 audit proved it by mutation — forcing
/// the predicate to `true` and reverting the paired change still left an
/// 18-of-18 green suite. This module reaches the private `release_outcome`
/// directly (an in-crate `#[cfg(test)]` table test — no server, no signals,
/// no tokio runtime needed).
///
/// **What actually goes RED, measured, not asserted** (contract
/// `CONTRACT-OPS-fix4.md` M3 — the round-3 audit found this doc previously
/// overstated its own module's capacity: it claimed "every test below
/// fails" against the round-2 revert, when the round-4 audit performed that
/// exact revert and only a minority did). Reverting BOTH arms' predicates to
/// "success iff the call returned `Ok`" (round 2's predicate) turns every
/// test that pins a determinant *this* module folds in RED, and leaves every
/// must-be-`Released` pin and every outer-`Err` pin GREEN — a weaker
/// predicate can only ever be MORE permissive on the `Ok` arms than a
/// stronger one, never less, so a "must be `Released`" or "must be
/// `Degraded`-on-`Err`" pin cannot be falsified by weakening a conjunct on
/// `Ok`. Only the determinant-pinning tests can, and do, go red under that
/// revert; see each test's own doc for which determinant it pins. Measured
/// at this module's current 15 tests: 7 go red, 8 stay green (re-run
/// yourself before trusting this number — it is a property of the test
/// count at the time this sentence was written, not a promise about a
/// future edit to this module).
///
/// # M1 — the determinant enumeration (contract `CONTRACT-OPS-fix4.md`)
///
/// This table test's oracles construct every determinant as a LITERAL — it
/// establishes the predicate reads each determinant correctly, never that
/// any real producer ever emits a failing one. The round-4 audit proved by
/// mutation that the producer half was entirely unobserved: applying all
/// four producer-side collapses at once (forcing `stop_witnessed` constant
/// `true`, re-collapsing the worker's and the session's per-hold `Err` back
/// into `Observed(HoldRelease::default())`, and folding the keeper's
/// per-hold `Err` back into `not_required`) left the WHOLE crate's suite
/// green, counts identical to baseline. The round-5 audit then measured
/// each of those four collapses SEPARATELY (never combined) and found the
/// producer half was in fact 1 of 4 observed, and refuted two of this
/// table's own "NOT producer-driven" claims by execution in ~2 seconds
/// using a technique already in this test tree — closed below. Every
/// determinant, enumerated, and where its producer-driven oracle lives or
/// why one does not exist yet:
///
/// * **P-2B, `HoldReleaseOutcome::Unobserved` (`Worker` arm)** (the keeper's
///   per-hold pass could not be confirmed to run at all): producer-driven in
///   `jammi-server`'s `crates/jammi-server/tests/it/liveness.rs`
///   (`healthz_flips_to_503_within_one_heartbeat_after_the_keeper_thread_dies`)
///   — the test kills the real keeper thread with
///   `LeaseKeeper::kill_thread_for_test`, then drives a real DRAIN+RELEASE
///   through `serve_with_signals`, and asserts the outcome the running
///   system actually returns is `ReleaseDegraded`. No literal `HoldRelease`
///   or `HoldReleaseOutcome` is constructed anywhere in that test.
/// * **P-2B, `HoldReleaseOutcome::Unobserved` (`SessionOnly` arm)** — a
///   SEPARATE producer from the bullet above, which drives the `Worker` arm
///   only: producer-driven in `crates/jammi-ai/tests/it/jobs_shutdown.rs`
///   (`release_job_leases_is_unobserved_when_the_keeper_thread_is_dead`),
///   the identical `LeaseKeeper::kill_thread_for_test` technique applied to
///   `InferenceSession::release_job_leases`'s own collapse.
/// * **P-2B, `HoldRelease.failed > 0`** (a per-hold release attempt itself
///   returning `Err` from `Catalog::release_job_lease`): producer-driven in
///   `crates/jammi-db/tests/it/lease_keeper.rs`
///   (`release_job_holds_reports_failed_from_a_real_backend_fault`). A prior
///   round of this doc claimed this was uncoverable — "no fault-injecting
///   `CatalogBackend` implementation and no reliable way to force one from
///   outside" — which was FALSE: a schema-level fault (`ALTER TABLE jobs
///   DROP COLUMN releases`, through the public `SqliteBackend::open` +
///   `CatalogBackend::transaction`, exactly the pattern
///   `crates/jammi-db/tests/it/migrations.rs` already uses) makes every
///   `release_job_lease` `UPDATE` fail at prepare time on the keeper's own
///   pooled connection, with no new test double. The claim was closed in
///   prose while the code stayed unobserved; the fix is the test, not the
///   sentence.
/// * **P-2B, `HoldRelease.not_required`** (a hold that provably did not need
///   releasing): producer-driven in
///   `crates/jammi-db/tests/it/lease_keeper.rs`
///   (`release_job_holds_flips_lost_and_skips_inline_holds`) — a real
///   inline-claimed row's hold produces a genuine `Ok(false)` from
///   `Catalog::release_job_lease`.
/// * **P-2B totality** (`released + not_required + failed == attempted`,
///   `HoldRelease::attempted` — M5): the production pass's own `assert_eq!`
///   (`crates/jammi-db/src/catalog/lease_keeper.rs`) is not itself bypassable
///   from a test without corrupting the pass, so this is checked at the type
///   level (`confirms_release_catches_an_undercounted_attempted` in
///   `crates/jammi-ai/src/fine_tune/worker.rs`) with a literal — the pass
///   that could produce an inconsistent value by construction refuses to
///   emit one, which is the property, not a gap.
/// * **P-2F, `stop_witnessed == true`** via a genuine abort or join
///   (`stop_resolved`): producer-driven throughout
///   `crates/jammi-server/tests/it/serve_shutdown_modes.rs` (e.g. the
///   sigint-while-draining and abort scenarios) and by the liveness test
///   above (a `Stopped` loop after the keeper dies still resolves the task).
/// * **P-2F, `stop_witnessed == false`** (2e found nothing to take AND 2f's
///   own observation fell back to the last-known proxy read): producer-driven
///   in `crates/jammi-ai/tests/it/jobs_shutdown.rs`
///   (`a_release_racing_an_in_flight_drain_reads_stop_unwitnessed`) — exactly
///   "any signal while draining is a RELEASE" (`deploy-server.md`): a DRAIN
///   (`stop_and_join`, unbounded by design) takes the handle — deterministically,
///   via a single hand-driven poll rather than a scheduler race — and waits
///   on an in-flight job the test parks mid-materialization with a real hold
///   registered (`jammi_db::store::mutable::test_hook`, never a wall clock or
///   the process-global `training_test_hooks::arm_pause_before_spawn_blocking`,
///   which is not scoped per test and was measured to let a concurrently
///   running fine-tune test steal the park), and a concurrent RELEASE loses
///   the handle race (`stop_resolved == false`) and then times out at 2f
///   within one heartbeat (`state_witnessed == false`) because the
///   still-parked loop never transitions. A prior round of this doc claimed
///   this arm was structurally unreachable; that argument is retracted in
///   favour of the test — the reasoning was never wrong about the
///   SINGLE-caller path, only about there being no second caller.
/// * **Sweep confirmation, `ReleaseSweep.jobs == None`** (the jobs sweep
///   statement itself returning `Err`): producer-driven in
///   `crates/jammi-db/tests/it/lease_keeper.rs`
///   (`release_job_holds_reports_failed_from_a_real_backend_fault`, which
///   also asserts `Catalog::release_jobs_claimed_by` errors under the same
///   fault) and end-to-end through a real `EmbeddedWorker::release_and_stop`
///   in `crates/jammi-ai/tests/it/jobs_shutdown.rs`
///   (`release_and_stops_second_sweep_reports_jobs_none_building_some_from_a_real_fault`),
///   which also confirms `ReleaseSweep.building` is UNAFFECTED by the same
///   fault (a different table, a different column) — the asymmetric shape
///   `sweep_confirms_release`'s two conjuncts read independently. Pinned at
///   the predicate itself by the literal
///   `a_jobs_only_unconfirmed_second_sweep_degrades` /
///   `session_only_jobs_only_unconfirmed_sweep_degrades` below: every
///   PRE-EXISTING unconfirmed-sweep pin in this table used `building ==
///   None`, so dropping the `jobs` conjunct out of `sweep_confirms_release`
///   went unnoticed by every test in this file until this round.
/// * **Sweep confirmation, `ReleaseSweep.building == None`** (the linked
///   building-table sweep statement itself returning `Err`): producer-driven
///   in `crates/jammi-ai/tests/it/jobs_shutdown.rs`
///   (`release_job_leases_second_sweep_reports_building_none_jobs_some_from_a_real_fault`).
///   A round-5 version of this doc declared this arm "NOT producer-driven
///   ... no OTHER injection point into this statement is established",
///   reasoning only about the COLUMN `release_building_tables_of_claimant`'s
///   `UPDATE` WRITES (`result_tables.lease_expires_at`); that argument is
///   retracted — the statement also NAMES a second table it reads FROM,
///   `result_tables` itself, and renaming that table out from under the
///   statement (the same public `SqliteBackend::open` +
///   `CatalogBackend::transaction` surface the `jobs`-fault test above
///   uses) faults it while `release_jobs_claimed_by` — which never touches
///   `result_tables` — still confirms: the mirror image of the asymmetric
///   shape above. Covered at the predicate itself by the literal
///   `an_unconfirmed_second_sweep_degrades` / `session_only_unconfirmed_
///   sweep_degrades` below.
/// * **Outer `Err`** (`ReleaseAttempt::Worker(Err(_))` /
///   `SessionOnly(Err(_))`): a single unconditional match arm with no
///   conjunct to collapse — a mutation deleting either arm's body is caught
///   by any test that exercises it at all, so this is lower-value to drive
///   from a producer; covered by the literal `worker_err_is_degraded` /
///   `session_only_err_is_degraded` below.
#[cfg(test)]
mod release_outcome_tests {
    use jammi_ai::fine_tune::worker::{HoldReleaseOutcome, LoopState, ReleaseReport, ReleaseSweep};
    use jammi_db::catalog::lease_keeper::HoldRelease;
    use jammi_db::error::JammiError;

    use super::{release_outcome, ReleaseAttempt, ShutdownOutcome};

    fn sweep(jobs: Option<usize>, building: Option<usize>) -> ReleaseSweep {
        ReleaseSweep { jobs, building }
    }

    /// A sweep whose own evidence confirms it ran (both fields `Some`).
    fn confirmed_sweep() -> ReleaseSweep {
        sweep(Some(0), Some(0))
    }

    fn report(
        loop_state: LoopState,
        holds: HoldReleaseOutcome,
        stop_witnessed: bool,
        sweep_one: ReleaseSweep,
        sweep_two: ReleaseSweep,
    ) -> ReleaseReport {
        ReleaseReport {
            loop_state,
            holds,
            stop_witnessed,
            sweep_one,
            sweep_two,
        }
    }

    // -------------------------------------------------------------------
    // The four arms of `ReleaseAttempt` / `release_outcome`.
    // -------------------------------------------------------------------

    #[test]
    fn worker_ok_fully_confirmed_is_released() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::Released
        );
    }

    #[test]
    fn worker_err_is_degraded() {
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Err(JammiError::Catalog(
                "lease keeper: cannot release job holds, its thread is dead".into()
            )))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    #[test]
    fn session_only_ok_fully_confirmed_is_released() {
        let holds = HoldReleaseOutcome::Observed(HoldRelease::default());
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Ok((holds, confirmed_sweep())))),
            ShutdownOutcome::Released
        );
    }

    #[test]
    fn session_only_err_is_degraded() {
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Err(JammiError::Catalog(
                "boom".into()
            )))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    // -------------------------------------------------------------------
    // Contract `CONTRACT-OPS-fix4.md` M2: the `SessionOnly` arm's own P-2B
    // conjunct (`holds.confirms_release() &&`, folded in this round exactly
    // like the Worker arm's) had NO oracle — the round-4 audit found
    // deleting it left `release_outcome_tests` 12/12 green AND the whole
    // `jammi-server` crate 285/285 green. These pin it, mirroring the two
    // cases already proven load-bearing on the Worker arm
    // (`an_unobserved_hold_pass_degrades`, `a_failed_hold_degrades_even_with_
    // confirmed_sweeps`) rather than adding only the single case the audit
    // wrote out.
    // -------------------------------------------------------------------

    /// The keeper's per-hold pass itself unobserved (a dead keeper) must
    /// degrade a `SessionOnly` release exactly as it does a `Worker` one,
    /// even though the outer call and the sweep are both fine.
    #[test]
    fn session_only_unobserved_hold_pass_degrades() {
        let holds = HoldReleaseOutcome::Unobserved;
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Ok((holds, confirmed_sweep())))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// A per-hold failure degrades a `SessionOnly` release even though the
    /// outer call succeeded and the sweep confirms.
    #[test]
    fn session_only_failed_hold_degrades() {
        let holds = HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 0,
            failed: 1,
            attempted: 1,
        });
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Ok((holds, confirmed_sweep())))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// Sibling check on the OTHER conjunct of the same arm's predicate
    /// (`sweep_confirms_release`, pre-existing but never pinned for
    /// `SessionOnly` specifically): an unconfirmed sweep degrades a
    /// `SessionOnly` release even with a fully confirmed hold pass.
    #[test]
    fn session_only_unconfirmed_sweep_degrades() {
        let holds = HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 0,
            failed: 0,
            attempted: 0,
        });
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Ok((
                holds,
                sweep(Some(0), None)
            )))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// The `SessionOnly` peer of `a_jobs_only_unconfirmed_second_sweep_
    /// degrades`: the same asymmetric arm (`jobs == None`, `building ==
    /// Some(_)`) no existing `SessionOnly` pin exercises either.
    #[test]
    fn session_only_jobs_only_unconfirmed_sweep_degrades() {
        let holds = HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 0,
            failed: 0,
            attempted: 0,
        });
        assert_eq!(
            release_outcome(ReleaseAttempt::SessionOnly(Ok((
                holds,
                sweep(None, Some(0))
            )))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    // -------------------------------------------------------------------
    // The newly folded determinants (P-2B, P-2F, P-EXCLUSION) — each of
    // these fails against the round-2 predicate ("success iff `Ok`").
    // -------------------------------------------------------------------

    /// P-2B: a per-hold failure degrades the outcome even though the OUTER
    /// call succeeded and both sweeps confirm — the collapse this round's
    /// contract closes at the producer (`lease_keeper.rs`), pinned here at
    /// the predicate that reads it.
    #[test]
    fn a_failed_hold_degrades_even_with_confirmed_sweeps() {
        let r = report(
            LoopState::Aborted,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 0,
                not_required: 0,
                failed: 1,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// P-2B: the keeper's per-hold pass itself UNOBSERVED (a dead keeper or
    /// a bound exceeded) is never treated as "zero holds released" —
    /// degraded, never released, even though the outer call and both
    /// sweeps are fine. This is the exact collapse a bare `usize` could not
    /// represent.
    #[test]
    fn an_unobserved_hold_pass_degrades() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Unobserved,
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// P-2F: an unwitnessed stop (2e found nothing to take AND 2f fell back
    /// to the last-known proxy read) degrades even with a confirmed hold
    /// pass and confirmed sweeps.
    #[test]
    fn an_unwitnessed_stop_degrades() {
        let r = report(
            LoopState::Running,
            HoldReleaseOutcome::Observed(HoldRelease::default()),
            false,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// P-EXCLUSION: sweep #1 unconfirmed (`None`) must NEVER degrade the
    /// outcome on its own — folding it in would manufacture a false
    /// degraded on a transient race the idempotent sweep #2 already
    /// superseded.
    #[test]
    fn an_unconfirmed_first_sweep_alone_never_degrades() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            sweep(None, None),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::Released
        );
    }

    /// Sweep #2 unconfirmed still degrades — the pre-existing behaviour,
    /// pinned so this round's changes cannot regress it.
    #[test]
    fn an_unconfirmed_second_sweep_degrades() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            sweep(Some(0), None),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    /// `sweep_confirms_release`'s TWO conjuncts fail independently
    /// (contract `CONTRACT-OPS-fix5.md`, round 5): every existing
    /// unconfirmed-sweep pin above (and `session_only_unconfirmed_sweep_
    /// degrades` below) drops `building` and leaves `jobs` `Some`, so
    /// dropping the `jobs` conjunct out of `sweep_confirms_release`
    /// entirely went unnoticed. Pinned here on the arm none of those
    /// exercise: `jobs == None`, `building == Some(_)`.
    #[test]
    fn a_jobs_only_unconfirmed_second_sweep_degrades() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            sweep(None, Some(0)),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::ReleaseDegraded
        );
    }

    // -------------------------------------------------------------------
    // False-positive pins: a normal cooperative release and a normal abort
    // release must both still report `Released` — the contract's own
    // stop-rule condition (a regression here that only a mutation test
    // would catch is exactly what ends this unit's wave-1 participation).
    // -------------------------------------------------------------------

    /// A normal cooperative release: 2e joined the task after observing
    /// its exit (`stop_witnessed = true`), `loop_state` `Stopped`, no
    /// holds needed, both sweeps confirm.
    #[test]
    fn a_normal_cooperative_release_is_released() {
        let r = report(
            LoopState::Stopped,
            HoldReleaseOutcome::Observed(HoldRelease::default()),
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::Released
        );
    }

    /// A normal abort release: 2e aborted the task directly
    /// (`stop_witnessed = true` via 2e's own certain resolution), a held
    /// lease released, both sweeps confirm — this is
    /// `serve_shutdown_modes::sigint_while_draining_releases_and_returns_
    /// released_within_two_heartbeats`'s (O1's) own scenario, pinned here
    /// at the unit level so a predicate regression is caught without the
    /// full server harness.
    #[test]
    fn a_normal_abort_release_is_released() {
        let r = report(
            LoopState::Aborted,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::Released
        );
    }

    /// The falsifier the design round handed forward, demonstrated
    /// directly: an abort whose in-task guard has not published within one
    /// heartbeat (`loop_state` reads a stale `Running`, the fallback proxy
    /// read) must NOT spuriously degrade a release that 2e itself already
    /// resolved with certainty.
    #[test]
    fn stop_witnessed_via_2e_survives_a_stale_running_read() {
        let r = report(
            LoopState::Running,
            HoldReleaseOutcome::Observed(HoldRelease {
                released: 1,
                not_required: 0,
                failed: 0,
                attempted: 1,
            }),
            true,
            confirmed_sweep(),
            confirmed_sweep(),
        );
        assert_eq!(
            release_outcome(ReleaseAttempt::Worker(Ok(r))),
            ShutdownOutcome::Released
        );
    }
}
