//! `roles.rs` — hosting a Ballista scheduler and/or executor inside a
//! `jammi-server` process, on jammi's own shutdown (never Ballista's
//! `start_server`/`start_executor_process`, which install their own
//! `ctrl_c` handlers and would race the server's two-mode shutdown — grep
//! `tests/it/roles.rs` for `signal::ctrl_c` to confirm neither is called),
//! and the third role, the client: a process whose materializations are
//! submitted to a scheduler ([`host_client`]). Each role installs the seam
//! it implements on the session: the scheduler its `PlacedGangSubmitter`,
//! the executor its `PlacedGangRunner`, the client its `ComputePlane`.
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

use ballista_scheduler::cluster::BallistaCluster;
use ballista_scheduler::config::{SchedulerConfig, TaskDistributionPolicy};
use ballista_scheduler::scheduler_process::create_scheduler;

use jammi_ai::operator::gang_exec::GangDescriptor;
use jammi_db::compute_plane::{ComputePlane, Submission};
use jammi_db::config::{BallistaClientConfig, BallistaExecutorConfig, BallistaSchedulerConfig};

use jammi_ai::session::InferenceSession;

use crate::codec::JammiCodec;
use crate::engine::JammiExecutionEngine;
use crate::error::{Error, Result};

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
/// PhysicalPlanNode>` over `cluster`, served on `bind` with jammi's own
/// shutdown. Push-staged, `task_max_failures = stage_max_failures = 0`
/// (retries are the jobs table's, not Ballista's). `cluster`/`distribution`
/// are the ONE constructor argument pair: `jammi-server`'s own hosting always
/// passes `BallistaCluster::new(CatalogClusterState, CatalogJobState)` and
/// `TaskDistributionPolicy::Custom(Arc::new(DevicePlacement))` — there is no
/// knob, `DevicePlacement` is the shipped policy — kept as parameters here
/// only so the in-memory cluster + a bare policy stay reachable as a TEST
/// fixture (`tests/it/roles.rs`), never a second production path.
pub async fn host_scheduler(
    session: &Arc<InferenceSession>,
    cfg: &BallistaSchedulerConfig,
    cluster: BallistaCluster,
    distribution: TaskDistributionPolicy,
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
    let session_for_config = Arc::clone(session);

    // Bind FIRST and resolve the REAL port (`addr.port()` may be `0`, "any
    // free port"): `SchedulerConfig::bind_port` is baked into `scheduler_name()`
    // (`external_host:bind_port`), which the scheduler stamps into every
    // task it pushes as the identity the executor's status-report loop
    // dials BACK — a `0` there makes every task-status report fail with
    // "Fail to connect to scheduler ...:0" (`tests/it/roles.rs` fails with
    // `addr.port()` there and passes with `local_addr.port()`).
    let listener = TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;

    let config_producer: ballista_core::ConfigProducer = Arc::new(move || {
        session_for_config
            .context()
            .copied_config()
            .upgrade_for_ballista()
    });
    let config = Arc::new(scheduler_config(
        external_host.clone(),
        local_addr.ip().to_string(),
        local_addr.port(),
        codec,
        config_producer,
        distribution,
    ));
    let name = config.scheduler_name();

    let scheduler = create_scheduler::<LogicalPlanNode, PhysicalPlanNode>(cluster, config)
        .await
        .map_err(Error::Ballista)?;
    let server = SchedulerGrpcServer::new(scheduler);

    // Install the `PlacedGangSubmitter` seam: a
    // claimant on THIS host submits its own gang as one Ballista task
    // instead of running it in-process. Write-once on the session; a
    // second `bind` of the same session keeps the first (the same shape
    // `install_member_dialer` uses).
    session
        .host_admission()
        .install_placed_gang_submitter(Arc::new(SchedulerPlacedGangSubmitter {
            session: Arc::clone(session),
            scheduler_url: format!("http://{external_host}:{}", local_addr.port()),
        }));

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
/// up a live scheduler.
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

/// The scheduler role's [`jammi_ai::fine_tune::worker::PlacedGangSubmitter`]:
/// submits a claimant's own gang as one `GangExec` Ballista task instead of
/// running it in-process. `placement_available()` answers
/// "a LIVE registered executor other than this instance exists" from the
/// catalog with `cluster::executor_is_live`, the binder's and the submit
/// edge's own predicate — `run_claimed_job_under` treats `false` as "run
/// in-process" (placement is a property of the claimant's cluster view,
/// decided BEFORE topology).
struct SchedulerPlacedGangSubmitter {
    session: Arc<InferenceSession>,
    scheduler_url: String,
}

impl jammi_ai::fine_tune::worker::PlacedGangSubmitter for SchedulerPlacedGangSubmitter {
    fn submit(
        &self,
        descriptor: GangDescriptor,
    ) -> futures::future::BoxFuture<
        'static,
        jammi_db::error::Result<
            futures::stream::BoxStream<
                'static,
                std::result::Result<arrow::array::RecordBatch, datafusion::error::DataFusionError>,
            >,
        >,
    > {
        let session = Arc::clone(&self.session);
        let url = self.scheduler_url.clone();
        Box::pin(async move {
            let plan: Arc<dyn datafusion::physical_plan::ExecutionPlan> =
                Arc::new(jammi_ai::operator::gang_exec::GangExec::new(descriptor));
            let stream = crate::client::submit_physical_plan(&session, &url, plan)
                .await
                .map_err(jammi_db::error::JammiError::from)?;
            use futures::StreamExt;
            Ok(stream.boxed())
        })
    }

    fn placement_available(&self) -> bool {
        let own_id = self.session.instance_id().to_string();
        let catalog = Arc::clone(self.session.catalog_arc());
        // The catalog read is async; this trait method is not (the seam
        // `jammi-ai`'s `run_claimed_job_under` checks synchronously before
        // deciding whether to place). Same block-in-place shape as
        // `codec.rs`'s `block_on_catalog` — requires a MULTI-THREADED tokio
        // runtime, a precondition every `jammi-server` process satisfies.
        // LIVE executors only (`cluster::executor_is_live`, the binder's and
        // the submit edge's own predicate): a row a dead executor left
        // behind must not divert a claim into the placed path only to have
        // the submit edge refuse it (an attempt spent for nothing).
        let rows = match tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move { catalog.list_compute_executors().await })
        }) {
            Ok(rows) => rows,
            Err(e) => {
                // A catalog fault is not "no peer": say so, then answer
                // "unavailable" — the claim runs in-process (still correct,
                // never parked), and the line names why the topology changed.
                tracing::warn!(
                    error = %e,
                    "placement_available: the catalog read failed; treating placement as unavailable"
                );
                return false;
            }
        };
        let now = chrono::Utc::now();
        rows.iter()
            .any(|r| r.executor_id != own_id && crate::cluster::executor_is_live(r, now))
    }
}

/// A hosted client role: this process's materializations are submitted to
/// the scheduler at [`Self::scheduler_url`]. Nothing listens and nothing
/// stops — the role is the installed seam.
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
/// [`crate::client`] against the scheduler `cfg` names. Write-once on the
/// session; a second install of the same session keeps the first (the
/// same shape the scheduler's `install_placed_gang_submitter` uses). The
/// scheduler is dialled at the first submission, never here: a client
/// comes up whether or not its scheduler is up yet, and a submission the
/// scheduler cannot take fails typed at that submission.
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
        .install(Arc::new(SchedulerComputePlane {
            session: Arc::clone(session),
            scheduler_url: scheduler_url.clone(),
        }));
    Ok(ClientRole { scheduler_url })
}

/// The client role's [`ComputePlane`]: [`crate::client::unheld`] is the
/// refusal (`Submission::Unheld`, never an error — the plan runs where it
/// was issued), [`crate::client::place`] the submission, whose failure is
/// the submission's own typed error.
struct SchedulerComputePlane {
    session: Arc<InferenceSession>,
    scheduler_url: String,
}

impl ComputePlane for SchedulerComputePlane {
    fn submit(
        &self,
        plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ) -> futures::future::BoxFuture<'static, jammi_db::error::Result<Submission>> {
        let session = Arc::clone(&self.session);
        let url = self.scheduler_url.clone();
        Box::pin(async move {
            if let Some(why) = crate::client::unheld(&session, &plan)
                .await
                .map_err(jammi_db::error::JammiError::from)?
            {
                return Ok(Submission::Unheld(why));
            }
            let stream = crate::client::place(&session, &url, plan)
                .await
                .map_err(jammi_db::error::JammiError::from)?;
            Ok(Submission::Placed(stream))
        })
    }
}

/// The executor role's [`jammi_ai::fine_tune::worker::PlacedGangRunner`]:
/// runs a placed gang's coordinator body on THIS process via
/// `JobWorker::run_placed_gang` — `GangExec::
/// execute` reaches this through the process-global seam `install_
/// placed_gang_runner` registers, since a Ballista executor's `TaskContext`
/// carries no jammi session.
struct ExecutorPlacedGangRunner {
    session: Arc<InferenceSession>,
}

impl jammi_ai::fine_tune::worker::PlacedGangRunner for ExecutorPlacedGangRunner {
    fn run(
        &self,
        descriptor: GangDescriptor,
    ) -> futures::future::BoxFuture<
        'static,
        jammi_db::error::Result<jammi_ai::operator::gang_exec::PlacedOutcome>,
    > {
        let session = Arc::clone(&self.session);
        Box::pin(async move {
            jammi_ai::fine_tune::worker::JobWorker::run_placed_gang(&session, descriptor).await
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
    /// (never tear down a running placed gang).
    pub async fn drain(self) {
        self.begin_drain().await;
        TasksDrainedFuture(Arc::clone(&self.executor)).await;
        self.stop().await;
    }

    /// The DRAIN INSTANT's half of [`Self::drain`], callable while the
    /// process's own worker is still draining (the server's DRAIN arm calls
    /// it the moment DRAIN is signalled, before the in-flight worker job is
    /// joined): flip `TERMINATING` and report `Terminating` to the scheduler
    /// so the catalog row stops reading as live (`cluster::executor_is_live`)
    /// and the binder stops binding new tasks here NOW, not after the grace
    /// period. Idempotent; the heartbeat is best-effort (a scheduler already
    /// gone cannot bind anything anyway).
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

    let session_for_config = Arc::clone(session);
    let config_producer: ballista_core::ConfigProducer = Arc::new(move || {
        session_for_config
            .context()
            .copied_config()
            .upgrade_for_ballista()
    });
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
        config_producer,
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

    // Install the `PlacedGangRunner` seam: `GangExec::
    // execute` reaches this process's coordinator body through it, since a
    // Ballista executor's `TaskContext` carries no jammi session.
    // Write-once, same shape as `install_member_dialer`.
    session
        .host_admission()
        .install_placed_gang_runner(Arc::new(ExecutorPlacedGangRunner {
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
/// (now `pub`) rather than reproducing its mapping — one mapping, never two.
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
