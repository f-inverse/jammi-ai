//! Plan 67 U5b-1b-ii — the coordinator body end to end over the REAL
//! `GangServer::run_rank` hold loop: coordinator rank 0 in-process (the real
//! claim → `run_claimed_job` → `run_spec` → `coordinate` path over the
//! server's own engine) plus one admitted member session over loopback,
//! whose `Peer` is built over the session's own `MemberLink` (taken through
//! the `test-hooks` tap, the rank body's future seat — for this unit the
//! member runs no rank body of its own: the test drives rank 1's
//! `TrainingLoop` over that link, the exact collective participation
//! U5b-1b-iii's body will own).
//!
//! One scenario, two attempts, RED at the base on both halves (no
//! coordinator body; a submit edge that refused a two-rank job on a
//! one-device host):
//!
//! - **attempt 1** — the member's slot is busy (`Holder::JobRun`
//!   manufactured on its `HostAdmission`), so the real handler answers
//!   `Unavailable` and the dial is refused: the attempt ends
//!   `MemberRefused` → `AssemblyOutcome::Unavailable`, recorded on the row
//!   COOLED and NOT COUNTED (`next_assembly_after` set,
//!   `assembly_failures = 0`), the lease handed back (`releases = 1`, lease
//!   NULL), nothing terminal;
//! - **attempt 2** — after the cooldown the job is claimed again and the
//!   body RE-LISTS (a second assignment is recorded, for attempt 2); the
//!   slot is free, the real handler admits the coordinator's dial, the
//!   coordinator's `Peer` is built over the admitted link and rank 0's run
//!   starts, with rank 1 on the far end of the real hold loop; the run's
//!   end is recorded `Success` (assembly proceeded), and the member session
//!   is ended by the coordinator's `Cancel` — its slot is free afterwards.
//!   The run itself has two admissible ends, pinned exactly: `Published`,
//!   in which case the adapter's bytes must EQUAL a U4b-shaped `LocalGang`
//!   run of the same fixture (the loopback rounds through the real
//!   `run_rank` equal `Local`); or the trainer's typed refusal of a
//!   `Streamed` source at `world > 1` — the ONE failure admitted, because
//!   at this tip a column-source `fine_tune` binds `Streamed` and U4b's
//!   streamed arm is not yet built; the moment it lands, this row demands
//!   the byte equality. Any other end is red.
//!
//! No substitution ever happens: the same member serves both attempts, at
//! rank 1, from the same sorted listing.

#![cfg(feature = "test-hooks")]

use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use jammi_ai::fine_tune::collective::{BlockingCall, LocalGang, Peer};
use jammi_ai::fine_tune::data::TrainingDataLoader;
use jammi_ai::fine_tune::lora::build_projection_head_for_rank;
use jammi_ai::fine_tune::partition::{PartitionRule, PartitionSpec};
use jammi_ai::fine_tune::source::TrainingSource;
use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::target::TrainingTarget;
use jammi_ai::fine_tune::trainer::{RankContext, TrainingLoopBuilder};
use jammi_ai::fine_tune::worker::{training_test_hooks, Holder, JobWorker, TopologyDecision};
use jammi_ai::fine_tune::{EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::instance::{InstanceRegistration, MemberRoot, PeerAddr};
use jammi_db::catalog::jobs_repo::{JobRecord, WorkerState};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::{ArtifactStore, CachePolicy};
use tempfile::TempDir;

/// The deployment lease for this scenario: long enough that no park bound
/// or freshness margin (`2 * lease`) cuts a session or a member row under
/// a training run; the uncounted outcome RELEASES the lease, so no attempt
/// ever waits for it to expire.
const LEASE: Duration = Duration::from_secs(30);
const HEARTBEAT: Duration = Duration::from_secs(10);

/// The U4b gang oracle's fixture: eight `(anchor, positive)` rows.
fn pairs() -> Vec<(String, String)> {
    (0..8)
        .map(|i| (format!("anchor text {i}"), format!("positive text {i}")))
        .collect()
}

fn gang_config(epochs: usize) -> FineTuneConfig {
    FineTuneConfig {
        epochs,
        batch_size: 2,
        validation_fraction: 0.0,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        lora_rank: 2,
        lora_dropout: 0.0,
        seed: 99,
        early_stopping_metric: EarlyStoppingMetric::TrainLoss,
        early_stopping_patience: 10_000,
        learning_rate: 1e-4,
        ..Default::default()
    }
}

fn tiny_bert_model() -> String {
    "local:".to_string()
        + jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .unwrap()
}

fn write_pairs_csv(dir: &std::path::Path) -> String {
    let path = dir.join("pairs.csv");
    let mut body = String::from("anchor,positive\n");
    for (anchor, positive) in pairs() {
        body.push_str(&format!("{anchor},{positive}\n"));
    }
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

/// A peer-bound server that can COORDINATE: `[worker] enabled = false` (the
/// test drives the claim itself), a serveable world of two, a membership
/// (`peer_advertise` set, so the engine's registration carries a
/// `MemberRoot` — the advertised address is never dialed), the CSV source
/// registered. The production `bind` installed the member dialer.
async fn coordinating_server() -> crate::common::grpc::PeerEngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = crate::common::grpc::peer_bind_config(dir.path());
    cfg.worker.enabled = false;
    cfg.lease.duration_secs = LEASE.as_secs();
    cfg.lease.heartbeat_secs = HEARTBEAT.as_secs();
    cfg.distributed.max_world_size = 2;
    cfg.server.peer_advertise = Some("127.0.0.1:1".into());
    let url = write_pairs_csv(dir.path());
    let server = crate::common::grpc::start_engine_server_from_config(cfg, Some(dir)).await;
    server
        .engine
        .add_source(
            "pairs",
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    server
}

fn two_rank_spec() -> TrainingSpec {
    TrainingSpec::FineTune {
        source: "pairs".into(),
        columns: vec!["anchor".into(), "positive".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: gang_config(2),
            world_size: 2,
        },
        cache: CachePolicy::Bypass,
    }
}

/// One fleet member for the coordinator to list and dial: an `instances`
/// row at `addr` carrying the SAME result-root identity as the engine's own
/// registration (the listing's root predicate), and a `claiming` `workers`
/// row over the `fine_tune` kind.
async fn register_member(engine: &Arc<InferenceSession>, id: &str, addr: std::net::SocketAddr) {
    let root = MemberRoot::resolved(engine.inner_config()).expect("the engine's own root");
    engine
        .catalog()
        .upsert_instance(&InstanceRegistration::new(
            id,
            Some("member"),
            Some("host"),
            Some(PeerAddr::parse(&addr.to_string()).unwrap()),
            Some(root),
        ))
        .await
        .unwrap();
    engine
        .catalog()
        .upsert_worker(id, "fine_tune", WorkerState::Claiming)
        .await
        .unwrap();
}

/// Claim the one queued job as the engine's own worker — the claim loop's
/// own two steps (`Catalog::reclaim_expired_jobs`, which requeues the row
/// a released attempt left `running` with a NULL lease, then
/// `Catalog::claim_next`), polling, since the SECOND claim waits out the
/// cooldown the first attempt recorded.
async fn claim(engine: &Arc<InferenceSession>, worker: &JobWorker, within: Duration) -> JobRecord {
    // The loop's reclaim cap (`worker.rs`'s `MAX_ATTEMPTS`), never reached
    // here: a released attempt costs the job no attempt.
    const MAX_ATTEMPTS: u32 = 3;
    let deadline = tokio::time::Instant::now() + within;
    loop {
        engine
            .catalog()
            .reclaim_expired_jobs(LEASE, MAX_ATTEMPTS)
            .await
            .unwrap();
        if let Some(record) = engine
            .catalog()
            .claim_next(worker.worker_id(), &["fine_tune"], LEASE)
            .await
            .unwrap()
        {
            return record;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the job was not claimable within {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    status: String,
    attempts: i32,
    releases: i32,
    lease_expires_at: Option<String>,
    assembly_failures: i32,
    next_assembly_after: Option<String>,
    training_set_ref: Option<String>,
    error: Option<String>,
}

async fn row(engine: &Arc<InferenceSession>, job_id: &str) -> Row {
    let job_id = job_id.to_string();
    engine
        .catalog()
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let job_id = job_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT status, attempts, releases, lease_expires_at, assembly_failures, \
                             next_assembly_after, training_set_ref, error \
                         FROM jobs WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                        |row| {
                            Ok(Row {
                                status: row.get("status")?,
                                attempts: row.get("attempts")?,
                                releases: row.get("releases")?,
                                lease_expires_at: row.try_get("lease_expires_at")?,
                                assembly_failures: row.get("assembly_failures")?,
                                next_assembly_after: row.try_get("next_assembly_after")?,
                                training_set_ref: row.try_get("training_set_ref")?,
                                error: row.try_get("error")?,
                            })
                        },
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the job row exists")
}

async fn published_adapter_bytes(engine: &Arc<InferenceSession>, job_id: &str) -> Vec<u8> {
    let models = engine.catalog().list_models().await.unwrap();
    let model = models
        .iter()
        .find(|m| {
            m.model_id
                .starts_with(&format!("jammi:fine-tuned:{job_id}"))
        })
        .expect("the completed job registered its fine-tuned model");
    let prefix = model
        .artifact_path
        .as_deref()
        .expect("a completed job's model row carries its served artifact_path");
    let local = engine
        .artifact_store()
        .fetch_artifact(&StorageUrl::parse(prefix).unwrap())
        .await
        .expect("the published adapter fetches and verifies");
    std::fs::read(local.dir().join("adapter.safetensors")).unwrap()
}

/// A catalog holding a claimed `running` row for a trainer built directly
/// (`trainer.rs`'s own `test_fixtures::claimed_job` shape).
async fn claimed_loop_env(tag: &str) -> (Arc<jammi_db::catalog::Catalog>, TempDir) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    let model_id = format!("{tag}-model");
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
            job_id: tag,
            kind: "fine_tune",
            execution: jammi_db::catalog::status::JobExecution::Queued,
            spec: "{}",
            model_ref: Some(&format!("{model_id}::1")),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next(
            &format!("{tag}-worker"),
            &["fine_tune"],
            Duration::from_secs(60),
        )
        .await
        .unwrap()
        .expect("queued job claimable");
    (catalog, dir)
}

fn file_store() -> Arc<ArtifactStore> {
    let root_dir = TempDir::new().unwrap().keep();
    let cache = TempDir::new().unwrap().keep();
    let root = StorageUrl::parse(root_dir.to_str().unwrap()).unwrap();
    Arc::new(ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap())
}

/// Everything a directly-built rank's `TrainingLoop` needs, prepared on
/// the runtime (async) so the rank's own thread only builds and runs.
struct RankEnv {
    base: Arc<jammi_ai::model::LoadedModel>,
    hidden: usize,
    catalog: Arc<jammi_db::catalog::Catalog>,
    dir: TempDir,
    store: Arc<ArtifactStore>,
}

async fn rank_env(engine: &Arc<InferenceSession>, tag: &str, store: Arc<ArtifactStore>) -> RankEnv {
    let guard = engine
        .model_cache()
        .get_or_load(
            &ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
            None,
        )
        .await
        .unwrap();
    let base = Arc::clone(&guard.model);
    let hidden = guard.model.embedding_dim().unwrap();
    drop(guard);
    let (catalog, dir) = claimed_loop_env(tag).await;
    RankEnv {
        base,
        hidden,
        catalog,
        dir,
        store,
    }
}

/// Build one rank's `TrainingLoop` over `rank_ctx` (the U4b gang oracle's
/// shape) and run it on the current thread: the saved `adapter.safetensors`
/// bytes, or the run's own error.
fn try_run_rank(
    call: &BlockingCall,
    env: RankEnv,
    job_id: &str,
    worker_id: &str,
    rank_ctx: RankContext,
) -> Result<Vec<u8>, String> {
    let config = gang_config(2);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let head = build_projection_head_for_rank(
        env.hidden,
        &config,
        &varmap,
        &vb,
        rank_ctx.dropout_seed(config.seed),
    )
    .unwrap();
    let mut training_loop =
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(Device::Cpu)
            .job_id(job_id.to_string())
            .worker_id(worker_id.to_string())
            .catalog(env.catalog)
            .artifact_dir(env.dir.path().to_path_buf())
            .base_model(env.base)
            .artifact_store(env.store)
            .rank_context(rank_ctx)
            .build()
            .unwrap();
    let result = training_loop
        .run(
            call,
            TrainingSource::Resident(TrainingDataLoader::from_pairs(pairs())),
        )
        .map_err(|e| e.to_string())?;
    Ok(std::fs::read(result.artifact_dir.path().join("adapter.safetensors")).unwrap())
}

/// [`try_run_rank`] for a rank that must complete.
fn run_rank(
    call: &BlockingCall,
    env: RankEnv,
    job_id: &str,
    worker_id: &str,
    rank_ctx: RankContext,
) -> Vec<u8> {
    try_run_rank(call, env, job_id, worker_id, rank_ctx)
        .unwrap_or_else(|e| panic!("{worker_id} must complete: {e}"))
}

/// The reference: a two-rank `LocalGang` on the CPU, each rank driven
/// directly through `TrainingLoop::run` on its own `spawn_thread`; rank 0's
/// adapter bytes.
async fn reference_rank0_adapter_bytes(engine: &Arc<InferenceSession>, tag: &str) -> Vec<u8> {
    let gang = LocalGang::new(vec![Device::Cpu, Device::Cpu]).unwrap();
    let store = file_store();
    let runtime = tokio::runtime::Handle::current();
    let job_id = format!("{tag}-reference");
    let mut threads = Vec::new();
    for rank in 0..2u32 {
        let local = gang.rank(rank).unwrap();
        let partition =
            PartitionSpec::for_gang(rank as usize, 2, 2, PartitionRule::BlockByGlobalBatch)
                .unwrap();
        let rank_ctx = RankContext::new(Arc::new(local), partition);
        let env = rank_env(engine, &format!("{tag}-ref-{rank}"), Arc::clone(&store)).await;
        let runtime = runtime.clone();
        let job_id = job_id.clone();
        threads.push(BlockingCall::spawn_thread(move |call| {
            let _runtime = runtime.enter();
            run_rank(&call, env, &job_id, &format!("reference-{rank}"), rank_ctx)
        }));
    }
    let mut rank0 = None;
    for (rank, thread) in threads.into_iter().enumerate() {
        let bytes = thread.join().unwrap();
        if rank == 0 {
            rank0 = Some(bytes);
        }
    }
    rank0.expect("rank 0 ran")
}

/// Acceptance (e), end to end, plus the loopback-equals-`Local` oracle —
/// see the module doc.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_answering_unavailable_ends_the_attempt_cooled_and_the_next_attempt_relists_and_runs_the_gang(
) {
    let server = coordinating_server().await;
    let engine = Arc::clone(&server.engine);
    let cap = usize::try_from(engine.inner_config().server.limits.max_message_bytes).unwrap();
    let (addr, mut links) =
        crate::gang_rounds::mount_real_gang_server(Arc::clone(&engine), cap).await;
    register_member(&engine, "member-1", addr).await;

    let job = engine
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job within the serveable world submits on a one-device host");
    let job_id = job.job_id.clone();
    let worker = JobWorker::new(&engine).unwrap();

    // ── attempt 1: the member's slot is busy ────────────────────────────
    let busy = engine.host_admission().hold_for_test(Holder::JobRun);
    let record = claim(&engine, &worker, Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    worker.run_claimed_job(&engine, record).await;
    drop(busy);

    assert_eq!(
        training_test_hooks::topology_for(&job_id),
        Some(TopologyDecision::Peer { world: 2 })
    );
    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_eq!(ends[0].0, 1);
    assert!(
        ends[0].1.starts_with("rank 1 (member-1) refused the dial:")
            && ends[0].1.contains("job slot is busy"),
        "the real handler's Unavailable ends the attempt naming the member: {}",
        ends[0].1
    );
    let listings = training_test_hooks::assembly_listings_for(&job_id);
    assert_eq!(
        listings,
        vec![(1, vec![(1, "member-1".to_string())])],
        "attempt 1 assigned rank 1 to the one listed member"
    );
    let after_one = row(&engine, &job_id).await;
    assert_eq!(
        after_one.status, "running",
        "nothing terminal: {after_one:?}"
    );
    assert_eq!(after_one.error, None);
    assert_eq!(after_one.assembly_failures, 0, "Unavailable is NOT counted");
    assert!(
        after_one.next_assembly_after.is_some(),
        "Unavailable is cooled down: {after_one:?}"
    );
    assert_eq!(after_one.releases, 1, "the lease was handed back at once");
    assert_eq!(after_one.lease_expires_at, None);
    assert!(
        after_one.training_set_ref.is_some(),
        "the CAS wrote the pair"
    );
    assert_eq!(
        engine.host_admission().holder(),
        Holder::Free,
        "no member session was admitted, nothing holds the slot"
    );

    // ── attempt 2: the member is free; rank 1 runs over the real hold loop ─
    let store = file_store();
    let member_env = rank_env(&engine, "member-rank-1", store).await;
    let member = BlockingCall::spawn_blocking(move |call| {
        let link = links
            .blocking_recv()
            .expect("the admitted session offered its MemberLink to the tap");
        let peer = Peer::member(1, 2, link, Device::Cpu, cap)
            .expect("member")
            .with_timeout(Duration::from_secs(120))
            .expect("timeout");
        let partition =
            PartitionSpec::for_gang(1, 2, 2, PartitionRule::BlockByGlobalBatch).unwrap();
        try_run_rank(
            &call,
            member_env,
            "member-rank-1",
            "member-rank-1",
            RankContext::new(Arc::new(peer), partition),
        )
    });

    let record = claim(&engine, &worker, Duration::from_secs(20)).await;
    assert_eq!(record.attempts, 2, "the next attempt, after the cooldown");
    worker.run_claimed_job(&engine, record).await;

    // The attempt's end FIRST: a member parked on the tap (a dial that was
    // refused after all) must surface as the end's own text, never as a
    // silent wait on the member thread.
    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 2, "{ends:?}");
    assert_eq!(ends[1].0, 2);
    // On record under `--nocapture`: which admissible end this tree took.
    eprintln!("attempt-2 end: {} (ordinal {})", ends[1].1, ends[1].2);
    assert!(
        ends[1].2 == 12 || ends[1].2 == 11,
        "attempt 2 must reach the run (Published or a run failure), got: {}",
        ends[1].1
    );
    let member_run = tokio::time::timeout(Duration::from_secs(60), member)
        .await
        .expect("the member thread ends within the bound once the coordinator ended its session")
        .expect("the member thread joined");
    let listings = training_test_hooks::assembly_listings_for(&job_id);
    assert_eq!(
        listings,
        vec![
            (1, vec![(1, "member-1".to_string())]),
            (2, vec![(1, "member-1".to_string())]),
        ],
        "the NEXT attempt re-listed and assigned the same sorted listing"
    );
    let after_two = row(&engine, &job_id).await;
    assert_eq!(after_two.assembly_failures, 0);
    assert_eq!(
        after_two.next_assembly_after, None,
        "assembly proceeded to a run: Success resets the cooldown: {after_two:?}"
    );
    assert_eq!(after_two.attempts, 2);

    const STREAMED_REFUSAL: &str = "a Streamed training source at world > 1 is refused";
    match ends[1].1.as_str() {
        "published" => {
            assert_eq!(after_two.status, "completed", "{after_two:?}");
            assert_eq!(after_two.error, None);
            let member_bytes =
                member_run.expect("rank 1 completed its run over the real hold loop");
            assert!(!member_bytes.is_empty());
            // The published artifact equals the LocalGang reference for
            // the same fixture: the loopback rounds through the real hold
            // loop folded exactly as Local does.
            let published = published_adapter_bytes(&engine, &job_id).await;
            let reference = reference_rank0_adapter_bytes(&engine, "peer-e2e").await;
            assert_eq!(
                published, reference,
                "rank 0's published adapter over Peer must be byte-identical to Local's rank 0"
            );
        }
        end if end.starts_with("the run failed:") => {
            // The one admissible run failure at this tip: the trainer's
            // typed refusal of the Streamed source at world > 1 (U4b's
            // streamed arm not yet built) — recorded `failed` by the
            // caller, exactly as a W=1 run's own failure would be, AFTER
            // assembly proceeded (Success above). Any other failure is red.
            assert!(
                end.contains(STREAMED_REFUSAL),
                "the only run failure this oracle admits is the streamed refusal: {end}"
            );
            assert_eq!(after_two.status, "failed", "{after_two:?}");
            assert!(
                after_two
                    .error
                    .as_deref()
                    .is_some_and(|e| e.contains(STREAMED_REFUSAL)),
                "{after_two:?}"
            );
            let member_error =
                member_run.expect_err("with no round ever opened, rank 1's first collective ends");
            assert!(
                member_error.contains("nothing applied"),
                "rank 1 ends on the coordinator's stream close, never on a fold: {member_error}"
            );
        }
        other => panic!("attempt 2 must end Published or in the streamed refusal, got {other}"),
    }

    // The coordinator ended the member's session: its slot is free again.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while engine.host_admission().holder() != Holder::Free {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the member session must end on the coordinator's Cancel and free its slot, holder: {:?}",
            engine.host_admission().holder()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
