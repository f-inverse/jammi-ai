//! The fine-tune gang on real GPUs through the product path: a job
//! submitted to the engine, claimed and laid out by the worker's own
//! topology decision, trained, and published — at every topology one host
//! with two devices can form, under each transport.
//!
//! - **1 × 1** — a single-rank job on device 0 (`TopologyDecision::Single`):
//!   the reference.
//! - **1 × 2, one process** — `[worker] local_ranks = 2` over devices 0 and 1
//!   (`TopologyDecision::Local`, an in-process gang).
//! - **2 processes × 1** — a coordinator on device 0 dials a fleet member on
//!   device 1 over the real `GangService` (`TopologyDecision::Peer`, the
//!   compute plane's gang), the same program a two-machine gang runs.
//!
//! Each multi-rank topology runs under `[worker] collective = "cpu"` (the
//! inline transport) and `"nccl"` (the NCCL device transport). The claims:
//!
//! - every multi-rank topology and transport publishes the SAME adapter
//!   bytes — every arm folds through one rank-ordered arithmetic, so neither
//!   the topology nor the transport is visible in the result;
//! - the same gang twice publishes the same bytes (determinism on device);
//! - the job's listing names where it ran (`claimed_by`, `ranks`);
//! - the two-rank gang trains to the single-rank reference's per-epoch loss
//!   at the same global batch, within the trainer's pre-registered ε.
//!
//! The genuinely multi-MACHINE form of the `Peer` row is the GPU topology
//! lane's (`gpu-topology.yml`): the same job across two hosts.
//!
//! Compiled under `live-gpu-gang-tests`: two CUDA devices and NCCL.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::JobWorker;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{CollectiveSelection, JammiConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::gang_rounds::GangDialer;
use tempfile::TempDir;

use crate::gang_chaos::{claim_within, Fleet, Member};
use crate::gang_coordinator::{
    gang_config, published_adapter_bytes, two_rank_spec, write_pairs_csv,
};

const LEASE: Duration = Duration::from_secs(20);
const HEARTBEAT: Duration = Duration::from_secs(2);
const RANK_TIMEOUT_SECS: u64 = 120;
const MAX_ATTEMPTS: u32 = 1;

/// The trainer's pre-registered W=2-vs-W=1 bound
/// (`trainer.rs`'s `gather_exactness_w2_matches_w1_within_pre_registered_epsilon`):
/// the same global batch reaches the same loss whether one rank or a gang
/// computes it.
const W2_VS_W1_LOSS_EPSILON: f64 = 1e-4;

/// Where a multi-rank job runs.
#[derive(Clone, Copy, Debug)]
enum Topology {
    /// One process, `local_ranks = 2` over devices 0 and 1.
    InProcess,
    /// A coordinator on device 0 and a fleet member on device 1.
    Fleet,
}

/// What a finished job left: its published adapter, its per-epoch loss, and
/// where it ran.
struct Trained {
    adapter: Vec<u8>,
    loss_curve: Vec<f64>,
    claimed_by: Option<String>,
    ranks: Vec<String>,
    instances: Vec<String>,
}

/// A host of the fleet, its config adjusted by `configure`.
async fn host(
    fleet: &Fleet,
    dir: &TempDir,
    dials_members: bool,
    configure: impl FnOnce(&mut JammiConfig),
) -> Arc<InferenceSession> {
    let mut cfg = fleet.host_config(dir.path(), LEASE, HEARTBEAT, RANK_TIMEOUT_SECS);
    configure(&mut cfg);
    // Boxed: an engine, a submission and a claimed run are each a large
    // future, and the test's root future lives on its thread's stack.
    let session = Arc::new(Box::pin(InferenceSession::new(cfg)).await.expect("host"));
    if dials_members {
        assert!(session
            .host_admission()
            .install_member_dialer(Arc::new(GangDialer)));
    }
    session
}

async fn add_pairs(session: &Arc<InferenceSession>, fleet: &Fleet) {
    session
        .add_source(
            "pairs",
            SourceType::File,
            SourceConnection {
                url: Some(write_pairs_csv(fleet.dir())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .expect("source");
}

/// Submit `spec` on `session`, claim it as `session`'s worker, run it to its
/// end, and read back what it left.
async fn train(session: &Arc<InferenceSession>, spec: TrainingSpec) -> Trained {
    let job = Box::pin(session.run_training_spec(spec))
        .await
        .expect("submit");
    let worker = JobWorker::new(session).expect("worker");
    let record = claim_within(
        session,
        worker.worker_id(),
        LEASE,
        MAX_ATTEMPTS,
        Duration::from_secs(60),
    )
    .await;
    Box::pin(worker.run_claimed_job(session, record)).await;
    let record = session.catalog().get_job(&job.job_id).await.expect("row");
    assert_eq!(
        record.status, "completed",
        "the job must complete: {:?}",
        record.error
    );
    let result: jammi_ai::jobs::JobResult =
        serde_json::from_str(record.result.as_deref().expect("a result")).expect("result");
    let jammi_ai::jobs::JobResult::Model {
        metrics: Some(metrics),
        ..
    } = result
    else {
        panic!("a fine-tune's result is a model with metrics");
    };
    let metrics: serde_json::Value = serde_json::from_str(&metrics).expect("metrics json");
    let loss_curve = metrics["train_loss_curve"]
        .as_array()
        .expect("a loss curve")
        .iter()
        .map(|point| point["loss"].as_f64().expect("a loss"))
        .collect();
    Trained {
        adapter: published_adapter_bytes(session, &job.job_id).await,
        loss_curve,
        claimed_by: record.claimed_by,
        ranks: record.ranks,
        instances: vec![session.instance_id().to_string()],
    }
}

/// The two-rank job (`two_rank_spec`) at `per_rank_batch` rows per rank.
fn two_rank_spec_at(per_rank_batch: usize) -> TrainingSpec {
    let TrainingSpec::FineTune {
        source,
        columns,
        method,
        task,
        mut common,
    } = two_rank_spec()
    else {
        unreachable!("two_rank_spec is a fine-tune")
    };
    common.config.batch_size = per_rank_batch;
    TrainingSpec::FineTune {
        source,
        columns,
        method,
        task,
        common,
    }
}

/// The two-rank job at `topology` under `collective`.
async fn two_ranks(topology: Topology, collective: CollectiveSelection) -> Trained {
    two_ranks_at(topology, collective, 2).await
}

/// [`two_ranks`] at `per_rank_batch` rows per rank.
async fn two_ranks_at(
    topology: Topology,
    collective: CollectiveSelection,
    per_rank_batch: usize,
) -> Trained {
    let fleet = Fleet::new();
    let dir = TempDir::new().expect("host dir");
    match topology {
        Topology::InProcess => {
            let session = host(&fleet, &dir, false, |cfg| {
                cfg.gpu.device = Some(0);
                cfg.gpu.devices = Some(vec![0, 1]);
                cfg.worker.local_ranks = 2;
                cfg.worker.collective = collective;
            })
            .await;
            add_pairs(&session, &fleet).await;
            train(&session, two_rank_spec_at(per_rank_batch)).await
        }
        Topology::Fleet => {
            let coordinator = host(&fleet, &dir, true, |cfg| {
                cfg.gpu.device = Some(0);
                cfg.server.peer_advertise = Some("127.0.0.1:1".into());
                cfg.worker.collective = collective;
            })
            .await;
            add_pairs(&coordinator, &fleet).await;
            let member = Member::start_configured(&fleet, |cfg| {
                cfg.gpu.device = Some(1);
                cfg.worker.collective = collective;
            })
            .await;
            let mut trained = train(&coordinator, two_rank_spec_at(per_rank_batch)).await;
            trained
                .instances
                .push(member.session.instance_id().to_string());
            trained
        }
    }
}

/// The single-rank reference on device 0, at the two-rank job's GLOBAL
/// batch (two ranks × two rows).
async fn one_rank() -> Trained {
    let fleet = Fleet::new();
    let dir = TempDir::new().expect("host dir");
    let session = host(&fleet, &dir, false, |cfg| {
        cfg.gpu.device = Some(0);
    })
    .await;
    add_pairs(&session, &fleet).await;
    let TrainingSpec::FineTune {
        source,
        columns,
        method,
        task,
        common,
    } = two_rank_spec()
    else {
        unreachable!("two_rank_spec is a fine-tune")
    };
    let mut config = gang_config(2);
    config.batch_size = 4;
    let spec = TrainingSpec::FineTune {
        source,
        columns,
        method,
        task,
        common: TrainingCommon {
            config,
            world_size: 1,
            ..common
        },
    };
    train(&session, spec).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn every_gpu_topology_and_transport_publishes_the_same_adapter() {
    let in_process_inline = two_ranks(Topology::InProcess, CollectiveSelection::Cpu).await;
    let in_process_nccl = two_ranks(Topology::InProcess, CollectiveSelection::Nccl).await;
    let fleet_inline = two_ranks(Topology::Fleet, CollectiveSelection::Cpu).await;
    let fleet_nccl = two_ranks(Topology::Fleet, CollectiveSelection::Nccl).await;

    for (name, trained) in [
        ("in-process, nccl", &in_process_nccl),
        ("fleet, inline", &fleet_inline),
        ("fleet, nccl", &fleet_nccl),
    ] {
        assert!(
            trained.adapter == in_process_inline.adapter,
            "{name}: a topology or a transport must never be visible in the published adapter"
        );
    }

    // Where each ran: the in-process gang's two ranks are this one process;
    // the fleet gang's rank 0 is the coordinator and rank 1 the member.
    let host_of = |t: &Trained| t.instances[0].clone();
    assert_eq!(
        in_process_nccl.ranks,
        vec![host_of(&in_process_nccl); 2],
        "an in-process gang runs every rank in the claiming process"
    );
    assert_eq!(
        fleet_nccl.ranks, fleet_nccl.instances,
        "rank 0 coordinator, rank 1 member"
    );
    assert_eq!(fleet_nccl.claimed_by, Some(fleet_nccl.instances[0].clone()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_nccl_gang_is_deterministic_and_trains_to_the_single_rank_loss() {
    let first = two_ranks(Topology::InProcess, CollectiveSelection::Nccl).await;
    let second = two_ranks(Topology::InProcess, CollectiveSelection::Nccl).await;
    assert!(
        first.adapter == second.adapter,
        "the same NCCL gang twice must publish the same bytes"
    );

    let reference = one_rank().await;
    assert_eq!(first.loss_curve.len(), reference.loss_curve.len());
    for (epoch, (gang, single)) in first
        .loss_curve
        .iter()
        .zip(&reference.loss_curve)
        .enumerate()
    {
        assert!(
            (gang - single).abs() <= W2_VS_W1_LOSS_EPSILON,
            "epoch {epoch}: the two-rank gang's loss {gang} must equal the single rank's \
             {single} within {W2_VS_W1_LOSS_EPSILON}"
        );
    }
}

/// A remainder batch on device: eight rows at three per rank (a global batch
/// of six) leave a last step of two rows — rank 0 takes both, rank 1 NONE.
/// A zero-row rank is a normal case of the partition rule; on CUDA it must
/// train, under either transport and either topology, to the same adapter.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_remainder_step_with_a_zero_row_rank_trains_on_device() {
    let in_process_inline = two_ranks_at(Topology::InProcess, CollectiveSelection::Cpu, 3).await;
    let in_process_nccl = two_ranks_at(Topology::InProcess, CollectiveSelection::Nccl, 3).await;
    let fleet_nccl = two_ranks_at(Topology::Fleet, CollectiveSelection::Nccl, 3).await;
    assert!(in_process_nccl.adapter == in_process_inline.adapter);
    assert!(fleet_nccl.adapter == in_process_inline.adapter);
}
