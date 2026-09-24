//! Where an encode rung's serve runs, and the proof of it. The call is the
//! same on every rung ([`crate::encode_step`]); what differs is the plane
//! under the session that makes it:
//!
//! - `placed`: the plane's three roles hosted in this process over the
//!   rung's own session (`jammi_ballista::roles`) — the sink is placed
//!   through the scheduler onto the executor, and the scheduler's binding
//!   and the sink's placement line prove it;
//! - `shape-d`: the deployed topology's fleet — the call made through the
//!   query tier's public surface, placed on a compute process, proven by the
//!   query tier's placement line, the scheduler's binding and the compute
//!   process's sink write.
//!
//! A leg that claims either and whose serve ran in the submitting process is
//! refused. The lines this process emits are read through
//! [`PlacementLog`], a layer of the one tracing subscriber the producer
//! installs. The whole module is the `plane` feature's: without it the
//! producer refuses both rungs.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jammi_ai::session::InferenceSession;
use jammi_ballista::placement::BOUND_TASK_LOG;
use jammi_ballista::roles::{
    host_client, host_executor, host_scheduler, ExecutorRole, SchedulerRole,
};
use jammi_db::config::{BallistaClientConfig, BallistaExecutorConfig, BallistaSchedulerConfig};
use jammi_db::source::{SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_wire::request::Modality;

use crate::leg::RanOn;
use crate::plane::fleet::{MemberRole, RunningFleet, SINK_LOCAL_LOG, SINK_PLACED_LOG};
use crate::plane::PlaneParams;

/// The lines about placement this process emits, kept for the proof: one
/// layer of the producer's tracing subscriber, fed every event whose
/// message names a binding or a sink's placement decision.
#[derive(Clone, Default)]
pub struct PlacementLog(Arc<Mutex<Vec<String>>>);

impl PlacementLog {
    /// Every kept line containing `needle`, in emission order.
    pub fn lines_containing(&self, needle: &str) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|line| line.contains(needle))
            .cloned()
            .collect()
    }
}

/// Renders an event's message and fields as one line.
#[derive(Default)]
struct LineVisitor(String);

impl tracing::field::Visit for LineVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0.insert_str(0, &format!("{value:?} "));
        } else {
            self.0.push_str(&format!("{}={value:?} ", field.name()));
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for PlacementLog {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = LineVisitor::default();
        event.record(&mut visitor);
        let line = format!("{}: {}", event.metadata().target(), visitor.0.trim_end());
        if [BOUND_TASK_LOG, SINK_PLACED_LOG, SINK_LOCAL_LOG]
            .iter()
            .any(|needle| line.contains(needle))
        {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(line);
        }
    }
}

/// How many placement lines the log held when a serve began — what the
/// serve's own proof is read against.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlacementMark {
    bound: usize,
    placed: usize,
}

impl PlacementMark {
    pub fn of(log: &PlacementLog) -> Self {
        Self {
            bound: log.lines_containing(BOUND_TASK_LOG).len(),
            placed: log.lines_containing(SINK_PLACED_LOG).len(),
        }
    }
}

/// The plane's three roles over one session in this process: the
/// scheduler and executor on loopback, the client installed on the
/// session so its materializations are placed.
pub struct InProcessRoles {
    scheduler: Option<SchedulerRole>,
    executor: Option<ExecutorRole>,
    executor_id: String,
    log: PlacementLog,
}

impl InProcessRoles {
    /// Host the roles over `session`; the executor's registration is
    /// awaited so the first serve can be bound.
    pub async fn host(
        session: &Arc<InferenceSession>,
        log: PlacementLog,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let scheduler = host_scheduler(
            session,
            &BallistaSchedulerConfig {
                bind: "127.0.0.1:0".to_string(),
                advertise_host: None,
            },
        )
        .await?;
        let scheduler_address = format!("127.0.0.1:{}", scheduler.addr.port());
        let executor = host_executor(
            session,
            &BallistaExecutorConfig {
                scheduler_address: scheduler_address.clone(),
                bind: "127.0.0.1:0".to_string(),
                grpc_bind: "127.0.0.1:0".to_string(),
                advertise_host: Some("127.0.0.1".to_string()),
                work_dir: None,
                task_slots: 1,
            },
        )
        .await?;
        let executor_id = executor.executor_id().to_string();
        host_client(
            session,
            &BallistaClientConfig {
                scheduler_address,
                device_kind: None,
            },
        )?;
        let deadline = Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let registered = session
                .catalog()
                .list_compute_executors()
                .await?
                .iter()
                .any(|e| e.executor_id == executor_id);
            if registered {
                break;
            }
            if Instant::now() >= deadline {
                return Err("the in-process executor never registered with the catalog".into());
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        Ok(Self {
            scheduler: Some(scheduler),
            executor: Some(executor),
            executor_id,
            log,
        })
    }

    pub fn mark(&self) -> PlacementMark {
        PlacementMark::of(&self.log)
    }

    /// Where the serve since `mark` ran: the executor, proven by a new
    /// binding and a new placement line — refused when the sink ran in
    /// this process unplaced.
    pub fn prove(
        &self,
        mark: PlacementMark,
    ) -> Result<RanOn, Box<dyn std::error::Error + Send + Sync>> {
        let bound = self.log.lines_containing(BOUND_TASK_LOG);
        let placed = self.log.lines_containing(SINK_PLACED_LOG);
        if bound.len() <= mark.bound || placed.len() <= mark.placed {
            let local = self.log.lines_containing(SINK_LOCAL_LOG);
            return Err(format!(
                "the placed rung's serve was not placed: the scheduler bound {} new task(s) \
                 and the sink placed {} new materialization(s); the sink's own account: \
                 {local:?}",
                bound.len() - mark.bound,
                placed.len() - mark.placed
            )
            .into());
        }
        Ok(RanOn {
            instance_id: self.executor_id.clone(),
            label: None,
            host: None,
            role: MemberRole::Executor.as_str().to_string(),
            evidence: bound[mark.bound..]
                .iter()
                .chain(placed[mark.placed..].iter())
                .cloned()
                .collect(),
        })
    }

    pub async fn stop(mut self) {
        if let Some(executor) = self.executor.take() {
            executor.stop().await;
        }
        if let Some(scheduler) = self.scheduler.take() {
            scheduler.stop().await;
        }
    }
}

/// The deployed topology's fleet, spawned on this host or joined, with
/// the corpus registered on its query tier.
pub struct ShapeDHost {
    fleet: RunningFleet,
    source: String,
    endpoint: tonic::transport::Endpoint,
}

impl ShapeDHost {
    pub async fn stand_up(
        plane: &PlaneParams,
        corpus: &Path,
        corpus_connection: SourceConnection,
        gpu_device: i32,
        leg: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let fleet = match &plane.query_addr {
            Some(query_addr) => RunningFleet::join_shape_d(query_addr, leg).await?,
            None => RunningFleet::spawn_shape_d(plane, leg, gpu_device).await?,
        };
        let query_addr = fleet
            .query_addr
            .clone()
            .ok_or("the shape-d fleet has no query tier")?;
        let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{query_addr}"))?;
        let admin = jammi_admin::CatalogClient::connect(endpoint.clone()).await?;
        let source = format!("corpus_{}", crate::capture::unique_suffix());
        let mut connection = corpus_connection;
        if let Some(url) = &plane.source_url {
            connection.url = Some(url.clone());
        } else if connection.url.is_none() {
            connection.url = Some(RunningFleet::local_url(corpus));
        }
        admin
            .add_source(&source, SourceType::File, connection)
            .await?;
        Ok(Self {
            fleet,
            source,
            endpoint,
        })
    }

    /// The observer session the fleet's catalog and store are read
    /// through.
    pub fn session(&self) -> &Arc<InferenceSession> {
        &self.fleet.session
    }

    /// One `generate_text_embeddings` call through the query tier,
    /// `CachePolicy::Bypass`, timed around the call alone: the table it
    /// committed, by name.
    pub async fn serve(
        &self,
        model_id: &str,
        text_column: &str,
        key_column: &str,
    ) -> Result<(String, f64), Box<dyn std::error::Error + Send + Sync>> {
        let client = jammi_client::DataClient::connect(self.endpoint.clone()).await?;
        let started = Instant::now();
        let (table, _) = client
            .generate_embeddings(
                &self.source,
                model_id,
                &[text_column.to_string()],
                key_column,
                Modality::Text,
                CachePolicy::Bypass,
            )
            .await?;
        Ok((table.table_name, started.elapsed().as_secs_f64()))
    }

    /// The table's vectors in its storage order, read through the
    /// catalog's own record of it.
    pub async fn vectors(
        &self,
        table_name: &str,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let session = &self.fleet.session;
        let record = session
            .catalog()
            .get_result_table(table_name)
            .await?
            .ok_or_else(|| format!("the served table {table_name} is not in the catalog"))?;
        let pin = session.result_store().pin_current_version(record).await?;
        Ok(session.read_vectors(&pin).await?)
    }

    /// Where the serve that committed `table_name` ran: a compute
    /// process, proven by the query tier's placement line, the scheduler's
    /// binding and the compute process's sink write.
    pub async fn prove(
        &mut self,
        table_name: &str,
    ) -> Result<RanOn, Box<dyn std::error::Error + Send + Sync>> {
        self.fleet
            .placed_sink_ran_on(
                table_name,
                MemberRole::Query,
                MemberRole::Scheduler,
                MemberRole::Compute,
            )
            .await
    }
}
