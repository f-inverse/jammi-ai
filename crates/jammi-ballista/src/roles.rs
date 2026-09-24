//! `roles.rs` — hosting a Ballista scheduler and/or executor inside a
//! `jammi-server` process, on jammi's own shutdown (never Ballista's
//! `start_server`/`start_executor_process`, which install their own
//! `ctrl_c` handlers and would race the server's two-mode shutdown — grep
//! `tests/it/roles.rs` for `signal::ctrl_c` to confirm neither is called),
//! and the third role, the client: a process whose submissions — a claimed
//! training attempt, a materialization — go to a scheduler ([`host_client`]). A role is
//! a listener-shaped knob: the scheduler and the executor bind what they
//! serve, the client names what it dials, and a process that hosts a
//! scheduler names itself as a client when its own submissions are to be
//! placed — one way to name a submitter's target, never an implied one.
//! Each role installs the seam it implements on the session: the executor
//! its `PlacedAttemptRunner`, the client its `ComputePlane` — the one submit
//! client every submission the process makes goes through.
//!
//! `ballista-scheduler` in this crate's `Cargo.toml` is
//! `default-features = false`: no `rest-api` surface. This is load-bearing,
//! not merely a smaller build — the REST API's `get_running_jobs` errors on
//! a status row with no execution graph (a scheduler restart keeps
//! `compute_jobs` STATUS rows but never revives the
//! graph itself), so a deployment that turned the REST API back on would
//! regress that property the moment it queried a restarted scheduler.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Channel;

use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionConfig;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_proto::protobuf::{LogicalPlanNode, PhysicalPlanNode};

use ballista_core::config::TaskSchedulingPolicy;
use ballista_core::error::BallistaError;
use ballista_core::extension::SessionConfigExt;
use ballista_core::registry::BallistaFunctionRegistry;
use ballista_core::serde::protobuf::executor_status::Status;
use ballista_core::serde::protobuf::scheduler_grpc_client::SchedulerGrpcClient;
use ballista_core::serde::protobuf::scheduler_grpc_server::SchedulerGrpcServer;
use ballista_core::serde::protobuf::{ExecutorRegistration, ExecutorStatus, HeartBeatParams};
use ballista_core::serde::{BallistaCodec, BallistaLogicalExtensionCodec};
use ballista_core::utils::{create_grpc_client_endpoint, create_grpc_server, GrpcServerConfig};

use ballista_executor::executor::{Executor, TasksDrainedFuture};
use ballista_executor::executor_process::{structure_executor_metadata, ExecutorProcessConfig};
use ballista_executor::executor_server::{self, TERMINATING};
use ballista_executor::flight_service::BallistaFlightService;
use ballista_executor::metrics::LoggingMetricsCollector;
use ballista_executor::shutdown::ShutdownNotifier;

use ballista_scheduler::cluster::{BallistaCluster, ClusterState, JobState};
use ballista_scheduler::config::{SchedulerConfig, TaskDistributionPolicy};
use ballista_scheduler::scheduler_process::create_scheduler;
use ballista_scheduler::scheduler_server::SessionBuilder;

use datafusion::execution::SendableRecordBatchStream;
use jammi_ai::operator::placed_attempt_exec::PlacedAttempt;
use jammi_db::compute_plane::{ComputePlane, Unheld};
use jammi_db::config::{BallistaClientConfig, BallistaExecutorConfig, BallistaSchedulerConfig};

use jammi_ai::session::InferenceSession;

use crate::cluster::{CatalogClusterState, CatalogJobState, PlacedJobs};
use crate::codec::JammiCodec;
use crate::engine::JammiExecutionEngine;
use crate::error::{Error, Result};
use crate::placement::DevicePlacement;

/// The `ConfigProducer` every role hands Ballista: `session`'s own
/// `SessionConfig` upgraded for Ballista, so a scheduler resolves and an
/// executor runs under the same options a plan was built under.
pub fn config_producer(session: &Arc<InferenceSession>) -> ballista_core::ConfigProducer {
    let session = Arc::clone(session);
    Arc::new(move || session.context().copied_config().upgrade_for_ballista())
}

/// The `SessionBuilder` a scheduler decodes submitted plans under:
/// `session`'s own state — its runtime env, holding the object store of
/// every result table and source this process bound, and its function
/// registry — under the submitting session's config. Ballista's default
/// builder builds a bare state whose runtime knows no store but the local
/// filesystem, so a scan of a result table on an object store would fail
/// to decode on the scheduler. The config replaces the state's in place:
/// rebuilding through `SessionStateBuilder` would re-create the default
/// catalog (`QueryContext::single_partition` states why).
pub fn session_builder(session: &Arc<InferenceSession>) -> SessionBuilder {
    let session = Arc::clone(session);
    Arc::new(move |config: SessionConfig| {
        let mut state = session.context().state();
        *state.config_mut() = config;
        Ok(state)
    })
}

/// A hosted Ballista scheduler.
pub struct SchedulerRole {
    /// The bound address (resolved from `:0` if the config asked for one).
    pub addr: SocketAddr,
    /// The identity every task this scheduler places carries —
    /// `advertise_host:port` — which the task's executor dials back to
    /// report the task's status.
    name: String,
    handle: JoinHandle<std::result::Result<(), BallistaError>>,
    stop: oneshot::Sender<()>,
}

impl SchedulerRole {
    /// The name this scheduler stamps into every task it places.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stop serving and await the listener's shutdown.
    pub async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.handle.await;
    }
}

/// Build the scheduler role: `create_scheduler::<LogicalPlanNode,
/// PhysicalPlanNode>` over the catalog-backed cluster —
/// [`CatalogClusterState`] and [`CatalogJobState`] on `session`'s own
/// catalog, owned by this instance, bound by
/// [`crate::placement::DevicePlacement`] — served on `bind` with jammi's
/// own shutdown. There is no other cluster and no other policy: a
/// deployment's topology is configuration, never a fork of the plane.
/// Push-staged, `task_max_failures = stage_max_failures = 0`: a task's
/// retry is the jobs table's, never Ballista's, and an executor's loss
/// fails the jobs bound to it typed ([`crate::cluster::PlacedJobs`],
/// installed here once the scheduler exists) rather than relaunching
/// their stages.
pub async fn host_scheduler(
    session: &Arc<InferenceSession>,
    cfg: &BallistaSchedulerConfig,
) -> Result<SchedulerRole> {
    let addr: SocketAddr = cfg.bind.parse().map_err(|e| {
        Error::Config(format!(
            "invalid ballista scheduler bind '{}': {e}",
            cfg.bind
        ))
    })?;
    // The name every placed task carries, which its executor dials back to
    // report the task's status: the advertised host, never an unspecified
    // bind host. Enforced at config-load time too; re-checked here so a
    // struct-literal config that skipped `load_from` cannot stand up a
    // scheduler no executor could ever report to.
    let external_host = cfg.advertised_host()?;

    let codec: Arc<dyn PhysicalExtensionCodec> = Arc::new(JammiCodec::new(session));

    // Bind FIRST and resolve the REAL port (`addr.port()` may be `0`, "any
    // free port"): `SchedulerConfig::bind_port` is baked into `scheduler_name()`
    // (`external_host:bind_port`), which the scheduler stamps into every
    // task it pushes as the identity the executor's status-report loop
    // dials BACK — a `0` there makes every task-status report fail with
    // "Fail to connect to scheduler ...:0" (`tests/it/roles.rs` fails with
    // `addr.port()` there and passes with `local_addr.port()`).
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;

    let catalog = Arc::clone(session.catalog_arc());
    let cluster_state = Arc::new(CatalogClusterState::new(Arc::clone(&catalog)));
    let job_state: Arc<dyn JobState> = Arc::new(CatalogJobState::new(
        Arc::clone(&catalog),
        session.instance_id(),
        session_builder(session),
        config_producer(session),
    ));
    let cluster = BallistaCluster::new(
        Arc::clone(&cluster_state) as Arc<dyn ClusterState>,
        Arc::clone(&job_state),
    );
    let config = Arc::new(scheduler_config(
        external_host,
        local_addr.ip().to_string(),
        local_addr.port(),
        codec,
        config_producer(session),
        TaskDistributionPolicy::Custom(Arc::new(DevicePlacement::new(catalog))),
    ));
    let name = config.scheduler_name();

    let scheduler = Arc::new(
        create_scheduler::<LogicalPlanNode, PhysicalPlanNode>(cluster, config)
            .await
            .map_err(Error::Ballista)?,
    );
    cluster_state.install_placed_jobs(PlacedJobs::new(&scheduler, job_state));
    let server = SchedulerGrpcServer::from_arc(scheduler);

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        create_grpc_server(&GrpcServerConfig::default())
            .add_service(server)
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = stop_rx.await;
            })
            .await
            .map_err(BallistaError::TonicError)
    });

    Ok(SchedulerRole {
        addr: local_addr,
        name,
        handle,
        stop: stop_tx,
    })
}

/// The `SchedulerConfig` `host_scheduler` builds — pulled into its own pure
/// function so `task_max_failures`/`stage_max_failures` (retries are the
/// jobs table's, not Ballista's) are unit-testable without standing
/// up a live scheduler. `distribution` is always `DevicePlacement` in
/// `host_scheduler`; the parameter keeps the config buildable under a bare
/// policy in the unit test below.
fn scheduler_config(
    external_host: String,
    bind_ip: String,
    bind_port: u16,
    codec: Arc<dyn PhysicalExtensionCodec>,
    config_producer: ballista_core::ConfigProducer,
    distribution: TaskDistributionPolicy,
) -> SchedulerConfig {
    SchedulerConfig {
        external_host,
        bind_host: bind_ip,
        bind_port,
        scheduling_policy: TaskSchedulingPolicy::PushStaged,
        task_distribution: distribution,
        // `None` here falls back to `BallistaLogicalExtensionCodec::default()`
        // inside `create_scheduler` (confirmed by reading
        // `ballista-scheduler-54.1.0/src/scheduler_process.rs:61-69`) — no
        // jammi logical nodes exist, so there is nothing for a
        // `LogicalExtensionCodec` impl on `JammiCodec` to carry.
        override_logical_codec: None,
        override_physical_codec: Some(codec),
        override_config_producer: Some(config_producer),
        task_max_failures: 0,
        stage_max_failures: 0,
        ..SchedulerConfig::default()
    }
}

/// A hosted client role: this process's submissions — a claimed training
/// attempt, a materialization — go to the scheduler at [`Self::scheduler_url`].
/// Nothing listens and nothing stops — the role is its installed seams.
pub struct ClientRole {
    scheduler_url: String,
}

impl ClientRole {
    /// The scheduler this client submits to, as the `http://host:port` URL
    /// the submit client dials.
    pub fn scheduler_url(&self) -> &str {
        &self.scheduler_url
    }
}

/// Build the client role: install the session's [`ComputePlane`] over
/// [`crate::client`] against the scheduler `cfg` names — the one seam a
/// materialization's plan and a claimant's own attempt both submit through.
/// Write-once on the session; a second install of the same session keeps
/// the first (the same shape `install_member_dialer` uses). The scheduler
/// is dialled at the first submission, never here: a client comes up
/// whether or not its scheduler is up yet, and a submission the scheduler
/// cannot take fails typed at that submission.
pub fn host_client(
    session: &Arc<InferenceSession>,
    cfg: &BallistaClientConfig,
) -> Result<ClientRole> {
    // Also enforced at config-load time; re-checked here so a struct-literal
    // config that skipped `load_from` still cannot install a client over a
    // target the submit client could never dial.
    let address = jammi_db::catalog::instance::PeerAddr::parse(&cfg.scheduler_address)
        .map_err(|e| Error::Config(format!("invalid ballista client scheduler_address: {e}")))?;
    let scheduler_url = format!("http://{address}");
    session
        .compute_plane()
        .install(Arc::new(ClientComputePlane {
            session: Arc::clone(session),
            scheduler_url: scheduler_url.clone(),
            device_kind: cfg.device_kind,
        }));
    Ok(ClientRole { scheduler_url })
}

/// The client role's [`ComputePlane`]: [`crate::client::unheld`] is the
/// admission (a refusal, never an error — the plan runs where it was
/// issued), [`crate::client::place`] the submission, whose failure is the
/// submission's own typed error.
struct ClientComputePlane {
    session: Arc<InferenceSession>,
    scheduler_url: String,
    device_kind: Option<jammi_db::store::manifest::ComputeDeviceKind>,
}

impl ComputePlane for ClientComputePlane {
    fn device_kind(&self) -> Option<jammi_db::store::manifest::ComputeDeviceKind> {
        self.device_kind
    }

    fn unheld(
        &self,
        plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ) -> futures::future::BoxFuture<'static, jammi_db::error::Result<Option<Unheld>>> {
        let session = Arc::clone(&self.session);
        let plan = Arc::clone(plan);
        Box::pin(async move {
            crate::client::unheld(&session, &plan)
                .await
                .map_err(jammi_db::error::JammiError::from)
        })
    }

    fn place(
        &self,
        plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ) -> futures::future::BoxFuture<'static, jammi_db::error::Result<SendableRecordBatchStream>>
    {
        let session = Arc::clone(&self.session);
        let url = self.scheduler_url.clone();
        Box::pin(async move {
            crate::client::place(&session, &url, plan)
                .await
                .map_err(jammi_db::error::JammiError::from)
        })
    }
}

/// The executor role's [`jammi_ai::fine_tune::worker::PlacedAttemptRunner`]:
/// runs a placed training attempt's body on THIS process via
/// `JobWorker::run_placed_attempt` — `PlacedAttemptExec::
/// execute` reaches this through the process-global seam `install_
/// placed_attempt_runner` registers, since a Ballista executor's `TaskContext`
/// carries no jammi session.
struct ExecutorPlacedAttemptRunner {
    session: Arc<InferenceSession>,
}

impl jammi_ai::fine_tune::worker::PlacedAttemptRunner for ExecutorPlacedAttemptRunner {
    fn run(
        &self,
        descriptor: PlacedAttempt,
    ) -> futures::future::BoxFuture<
        'static,
        jammi_db::error::Result<jammi_ai::operator::placed_attempt_exec::PlacedOutcome>,
    > {
        let session = Arc::clone(&self.session);
        Box::pin(async move {
            jammi_ai::fine_tune::worker::JobWorker::run_placed_attempt(&session, descriptor).await
        })
    }
}

/// A hosted Ballista executor.
pub struct ExecutorRole {
    /// The Arrow Flight (shuffle) listener address.
    pub flight_addr: SocketAddr,
    /// The gRPC (task) listener address.
    pub grpc_addr: SocketAddr,
    executor: Arc<Executor>,
    scheduler_client: SchedulerGrpcClient<Channel>,
    executor_id: String,
    metadata: ExecutorRegistration,
    devices: Vec<jammi_db::catalog::instance::DeviceFact>,
    notifier: Arc<ShutdownNotifier>,
    flight_handle: JoinHandle<std::result::Result<(), BallistaError>>,
    grpc_handle: JoinHandle<std::result::Result<(), BallistaError>>,
}

impl ExecutorRole {
    /// This executor's own devices (kind × ordinal, no memory field — the
    /// `compute_executors.devices` shape, `jammi_db::catalog::instance::
    /// DeviceFact`). Best-effort: a `[worker]`-disabled process (no
    /// local-rank topology) reports none.
    pub fn devices(&self) -> &[jammi_db::catalog::instance::DeviceFact] {
        &self.devices
    }

    /// This executor's instance id, the same id `HostAdmission`/gang
    /// placement use to identify it (the executor id IS the jammi instance
    /// id).
    pub fn executor_id(&self) -> &str {
        &self.executor_id
    }

    /// RELEASE: stop admitting new work immediately and tear down both
    /// listeners without waiting for in-flight tasks.
    pub async fn stop(self) {
        let _ = self.notifier.notify_shutdown.send(());
        let _ = self.flight_handle.await;
        let _ = self.grpc_handle.await;
    }

    /// DRAIN: stop admitting new tasks — report `Terminating` through the
    /// heartbeat status, the same signal `ballista-executor`'s own shutdown
    /// path sends (`executor_process.rs`'s `TERMINATING` flag +
    /// `heart_beat_from_executor` call) so the scheduler stops binding new
    /// tasks here — then wait for every in-flight task to finish, THEN stop
    /// (never tear down a running placed attempt).
    pub async fn drain(self) {
        self.begin_drain().await;
        TasksDrainedFuture(Arc::clone(&self.executor)).await;
        self.stop().await;
    }

    /// The DRAIN INSTANT's half of [`Self::drain`], callable while the
    /// process's own worker is still draining (the server's DRAIN arm calls
    /// it the moment DRAIN is signalled, before the in-flight worker job is
    /// joined): flip `TERMINATING` — the flag `ballista-executor`'s own
    /// periodic heartbeater consults, so every heartbeat it builds from here
    /// on reports `Terminating` — and report `Terminating` to the scheduler
    /// now, so the catalog row stops reading as live
    /// (`cluster::executor_is_live`) and the binder stops binding new tasks
    /// here NOW, not after the grace period. A heartbeat the heartbeater
    /// built before the flag flipped and delivered after this report still
    /// claims `Active`; the catalog write is monotone in the lifecycle
    /// (`Catalog::record_compute_heartbeat`), so it refreshes the row's
    /// timestamp and never its state. Idempotent; the heartbeat is
    /// best-effort (a scheduler already gone cannot bind anything anyway).
    pub async fn begin_drain(&self) {
        TERMINATING.store(true, std::sync::atomic::Ordering::Release);
        let _ = self
            .scheduler_client
            .clone()
            .heart_beat_from_executor(HeartBeatParams {
                executor_id: self.executor_id.clone(),
                metrics: vec![],
                status: Some(ExecutorStatus {
                    status: Some(Status::Terminating(String::default())),
                }),
                metadata: Some(self.metadata.clone()),
            })
            .await;
    }
}

/// How long an executor role waits for its scheduler to accept a connection
/// before refusing typed (the scheduler is another process and may still be
/// starting), and the pause between attempts.
pub const SCHEDULER_CONNECT_WINDOW: std::time::Duration = std::time::Duration::from_secs(60);
const SCHEDULER_CONNECT_RETRY: std::time::Duration = std::time::Duration::from_millis(250);

/// Build the executor role: the Flight shuffle service on `cfg.bind`, the
/// task/heartbeat gRPC service + scheduler registration on `cfg.grpc_bind`
/// via `executor_server::startup` (push-staged), both under jammi's own
/// shutdown. The executor id IS `session.instance_id()` — the join
/// `DevicePlacement` uses.
pub async fn host_executor(
    session: &Arc<InferenceSession>,
    cfg: &BallistaExecutorConfig,
) -> Result<ExecutorRole> {
    let flight_addr: SocketAddr = cfg.bind.parse().map_err(|e| {
        Error::Config(format!(
            "invalid ballista executor bind '{}': {e}",
            cfg.bind
        ))
    })?;
    let grpc_addr: SocketAddr = cfg.grpc_bind.parse().map_err(|e| {
        Error::Config(format!(
            "invalid ballista executor grpc_bind '{}': {e}",
            cfg.grpc_bind
        ))
    })?;
    let (scheduler_host, scheduler_port) =
        cfg.scheduler_address.rsplit_once(':').ok_or_else(|| {
            Error::Config(format!(
                "invalid ballista executor scheduler_address '{}': expected host:port",
                cfg.scheduler_address
            ))
        })?;
    let scheduler_port: u16 = scheduler_port
        .parse()
        .map_err(|e| Error::Config(format!("invalid scheduler_address port: {e}")))?;

    // The host the scheduler dials back — also enforced at config-load
    // time; re-checked here so a struct-literal config that skipped
    // `load_from` still cannot stand up a role the scheduler could never
    // dial back.
    let advertise_host = cfg.advertised_host()?;

    let work_dir = match &cfg.work_dir {
        Some(p) => {
            std::fs::create_dir_all(p)?;
            p.to_string_lossy().into_owned()
        }
        None => {
            let dir = std::env::temp_dir().join(format!("jammi-ballista-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir)?;
            dir.to_string_lossy().into_owned()
        }
    };

    let codec: Arc<dyn PhysicalExtensionCodec> = Arc::new(JammiCodec::new(session));

    let session_for_runtime = Arc::clone(session);
    let runtime_producer: ballista_core::RuntimeProducer =
        Arc::new(move |_: &SessionConfig| Ok(session_for_runtime.context().runtime_env()));
    let function_registry = Arc::new(BallistaFunctionRegistry::from(&session.context().state()));
    let execution_engine: Arc<dyn ballista_executor::execution_engine::ExecutionEngine> =
        Arc::new(JammiExecutionEngine::new(Arc::clone(session)));

    // Bind the Flight (shuffle) listener FIRST and resolve its REAL port
    // (`flight_addr.port()` may be `0`, "any free port" — the registration
    // metadata below must carry the port the OS actually assigned, not the
    // pre-bind request).
    let flight_listener = TcpListener::bind(flight_addr).await?;
    let flight_local_addr = flight_listener.local_addr()?;

    // `executor_server::startup` binds `grpc_bind` itself, lazily, from
    // `ExecutorProcessConfig.grpc_port` — which the registration metadata
    // ALSO must already carry the real value of. When `cfg.grpc_bind` asks
    // for an ephemeral port (`:0`), resolve one now (bind-then-release) so
    // both the metadata and `startup`'s own later bind agree on one number;
    // a fixed port never takes this branch.
    let resolved_grpc_port = if grpc_addr.port() == 0 {
        let probe = TcpListener::bind(grpc_addr).await?;
        let port = probe.local_addr()?.port();
        drop(probe);
        port
    } else {
        grpc_addr.port()
    };
    let grpc_addr = SocketAddr::new(grpc_addr.ip(), resolved_grpc_port);

    let process_config = Arc::new(ExecutorProcessConfig {
        bind_host: grpc_addr.ip().to_string(),
        external_host: Some(advertise_host),
        port: flight_local_addr.port(),
        grpc_port: grpc_addr.port(),
        scheduler_host: scheduler_host.to_string(),
        scheduler_port,
        task_scheduling_policy: TaskSchedulingPolicy::PushStaged,
        work_dir: Some(work_dir.clone()),
        concurrent_tasks: cfg.task_slots as usize,
        // Documentation only on THIS path: the codec that actually
        // governs decode is the one passed to `executor_server::startup`
        // below, not this field — `startup` never reads
        // `ExecutorProcessConfig.override_physical_codec` (only
        // `start_executor_process`, which this crate never calls, does).
        // Set anyway so a maintainer inspecting this config sees the
        // intended codec, and asserted equal to the `startup` argument in
        // `tests/it/roles.rs`.
        override_physical_codec: Some(Arc::clone(&codec)),
        ..ExecutorProcessConfig::default()
    });

    let executor_id = session.instance_id().to_string();
    let executor_meta = structure_executor_metadata(&executor_id, &process_config, cfg.task_slots);

    let executor = Arc::new(Executor::new(
        executor_meta.clone(),
        &work_dir,
        runtime_producer,
        config_producer(session),
        function_registry,
        Arc::new(LoggingMetricsCollector::default()),
        cfg.task_slots as usize,
        execution_engine,
    ));

    let notifier = Arc::new(ShutdownNotifier::new());
    let mut flight_shutdown = notifier.subscribe_for_shutdown();
    let flight_service = FlightServiceServer::new(BallistaFlightService::new(work_dir.clone()));
    let flight_handle = tokio::spawn(async move {
        create_grpc_server(&GrpcServerConfig::default())
            .add_service(flight_service)
            .serve_with_incoming_shutdown(TcpListenerStream::new(flight_listener), async move {
                flight_shutdown.recv().await;
            })
            .await
            .map_err(BallistaError::TonicError)
    });

    // Connect to the scheduler, register, and start the push-staged task
    // gRPC server + heartbeat + task-runner loops (`executor_server::
    // startup` binds `grpc_bind` itself, using `process_config.bind_host` +
    // `executor_meta.grpc_port`).
    let endpoint =
        create_grpc_client_endpoint(format!("http://{scheduler_host}:{scheduler_port}"), None)
            .map_err(|e| Error::Role(format!("could not build scheduler endpoint: {e}")))?;
    // The scheduler is another process that may still be starting (a compute
    // pod and its scheduler pod come up together; a fleet's three processes
    // spawn concurrently): an executor waits for its scheduler for a bounded
    // window rather than failing on the first refused connect, and refuses
    // typed — naming the address and the window — only when the window ends.
    let channel = {
        let deadline = std::time::Instant::now() + SCHEDULER_CONNECT_WINDOW;
        loop {
            match endpoint.connect().await {
                Ok(channel) => break channel,
                Err(e) if std::time::Instant::now() < deadline => {
                    tracing::warn!(
                        scheduler = %format!("{scheduler_host}:{scheduler_port}"),
                        error = %e,
                        "scheduler not reachable yet; retrying"
                    );
                    tokio::time::sleep(SCHEDULER_CONNECT_RETRY).await;
                }
                Err(e) => {
                    return Err(Error::Role(format!(
                        "scheduler at {scheduler_host}:{scheduler_port} unreachable for {}s: {e}",
                        SCHEDULER_CONNECT_WINDOW.as_secs()
                    )));
                }
            }
        }
    };
    let scheduler_client = SchedulerGrpcClient::new(channel);

    let ballista_codec: BallistaCodec = BallistaCodec::new(
        Arc::new(BallistaLogicalExtensionCodec::default()),
        Arc::clone(&codec),
    );
    let (stop_send, _stop_recv) = mpsc::channel::<bool>(1);
    let grpc_handle = executor_server::startup(
        scheduler_client.clone(),
        Arc::clone(&process_config),
        Arc::clone(&executor),
        ballista_codec,
        stop_send,
        &notifier,
    )
    .await
    .map_err(Error::Ballista)?;

    let devices = device_facts(session);

    // Stamp this executor's OWN device claim onto its `compute_executors`
    // row — `ClusterState::register_executor`'s fixed
    // signature (the gRPC call `executor_server::startup` just completed
    // above triggered) carries no device field, so this process patches its
    // own row directly over the SAME shared catalog right after
    // registration completes, deterministically the LAST write of this
    // startup sequence (`cluster::CatalogClusterState::register_executor`'s
    // doc names this ordering). A missing row here would mean the
    // registration call above never landed — `executor_server::startup`
    // already returned `Ok`, so this is defensive, not the expected path.
    match session.catalog().get_compute_executor(&executor_id).await {
        Ok(Some(existing)) => {
            let rec = jammi_db::catalog::compute_repo::ComputeExecutorRecord {
                devices: devices.clone(),
                ..existing
            };
            if let Err(e) = session.catalog().upsert_compute_executor(&rec).await {
                tracing::warn!(
                    executor_id = %executor_id,
                    error = %e,
                    "jammi-ballista: failed to record this executor's device claim"
                );
            }
        }
        Ok(None) => {
            tracing::warn!(
                executor_id = %executor_id,
                "jammi-ballista: no compute_executors row yet to attach this executor's \
                 device claim to"
            );
        }
        Err(e) => {
            tracing::warn!(
                executor_id = %executor_id,
                error = %e,
                "jammi-ballista: could not read this executor's own row to record its \
                 device claim"
            );
        }
    }

    // Install the `PlacedAttemptRunner` seam: `PlacedAttemptExec::
    // execute` reaches this process's coordinator body through it, since a
    // Ballista executor's `TaskContext` carries no jammi session.
    // Write-once, same shape as `install_member_dialer`.
    session
        .host_admission()
        .install_placed_attempt_runner(Arc::new(ExecutorPlacedAttemptRunner {
            session: Arc::clone(session),
        }));

    Ok(ExecutorRole {
        flight_addr: flight_local_addr,
        grpc_addr,
        executor,
        scheduler_client,
        executor_id,
        metadata: executor_meta,
        devices,
        notifier,
        flight_handle,
        grpc_handle,
    })
}

/// This session's device facts (the `compute_executors.devices` shape):
/// kind × every rank ordinal `[worker]`'s topology names
/// (the CPU sentinel `-1` becomes `0` — a `DeviceFact` ordinal is never
/// negative), or empty when this process runs no local ranks (`[worker]
/// enabled = false` / `local_ranks` unset — an executor-only process is not
/// itself a worker). Calls `jammi_ai::fine_tune::worker::worker_devices`
/// rather than reproducing its mapping — one mapping, never two.
fn device_facts(session: &Arc<InferenceSession>) -> Vec<jammi_db::catalog::instance::DeviceFact> {
    jammi_ai::fine_tune::worker::worker_devices(session.jammi_config(), session.compute_device())
}

#[cfg(test)]
mod scheduler_config_tests {
    use super::*;

    /// Both roles pin `task_max_failures = stage_max_failures =
    /// 0` — retries are the jobs table's, never Ballista's own. Mutation:
    /// dropping either literal (or flipping to `SchedulerConfig::default()`'s
    /// `4`) reds this immediately.
    #[test]
    fn pins_zero_task_and_stage_failures() {
        let codec: Arc<dyn PhysicalExtensionCodec> =
            Arc::new(ballista_core::serde::BallistaPhysicalExtensionCodec::default());
        let config_producer: ballista_core::ConfigProducer =
            Arc::new(ballista_core::utils::default_config_producer);
        let cfg = scheduler_config(
            "127.0.0.1".into(),
            "127.0.0.1".into(),
            0,
            codec,
            config_producer,
            TaskDistributionPolicy::RoundRobin,
        );
        assert_eq!(cfg.task_max_failures, 0);
        assert_eq!(cfg.stage_max_failures, 0);
        assert!(matches!(
            cfg.scheduling_policy,
            ballista_core::config::TaskSchedulingPolicy::PushStaged
        ));
    }
}
