//! Plan 67 U5b-1b-ii/iii — the coordinator body and the REAL rank body end
//! to end over the REAL `GangServer::run_rank` hold loop on the production
//! `peer_bind` listener: coordinator rank 0 in-process (the real claim →
//! `run_claimed_job` → `run_spec` → `coordinate` path over the server's
//! own engine) plus one admitted member session over loopback whose rank
//! body (`run_member_rank`, spawned by the handler at admission) trains
//! rank 1 over the session's own `MemberLink` and ends the session with
//! `RankEvent::Outcome`.
//!
//! Two scenarios, RED at the base (no rank body: the handler parked every
//! admitted session; no runner role; no `Outcome` producer or consumer):
//!
//! - **the healthy gang, two attempts** — attempt 1: the member's slot is
//!   busy (`Holder::JobRun` manufactured on its `HostAdmission`), so the
//!   real handler answers `Unavailable` and the dial is refused: the
//!   attempt ends `MemberRefused` → `AssemblyOutcome::Unavailable`,
//!   recorded on the row COOLED and NOT COUNTED, the lease handed back,
//!   nothing terminal, and the attempt's hold registered as the
//!   `Coordinator`; attempt 2: after the cooldown the job is claimed again
//!   and the body RE-LISTS, the slot is free, the real handler admits the
//!   coordinator's dial and spawns the rank body, rank 0 runs as
//!   `Holder(Coordinator)` and the member as `Rank { 1 }` (both recorded),
//!   the member's session ends `Outcome{Trained{digest}}`, the coordinator
//!   reads it, finds it equal to its own adapter digest, and ONLY THEN
//!   publishes: the row is `completed` through the same
//!   `finish_job_with_model` CAS a loop-claimed run takes, the published
//!   adapter is byte-identical to a U4b-shaped `LocalGang` run of the same
//!   fixture, and the member's slot is free afterwards;
//! - **a member whose body fails** — the member's rank body completes its
//!   run but reports `Outcome{Failed{reason}}` (a `test-hooks` fault
//!   injection at the body's natural end): the coordinator ends the attempt
//!   `TrainingFailed("rank 1: …")`, records `failed` on the row under the
//!   `Coordinator` role with that reason, registers no model row and
//!   publishes nothing — the terminal write on receipt, in its failure
//!   arm.
//!
//! No substitution ever happens: the same member serves every attempt, at
//! rank 1, from the same sorted listing.

#![cfg(feature = "test-hooks")]

use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use jammi_ai::fine_tune::collective::{BlockingCall, LocalGang};
use jammi_ai::fine_tune::data::TrainingDataLoader;
use jammi_ai::fine_tune::lora::build_projection_head_for_rank;
use jammi_ai::fine_tune::partition::{PartitionRule, PartitionSpec};
use jammi_ai::fine_tune::role::{LeaseHolder, RunnerRole};
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

/// The deployment lease for this scenario: long enough that no freshness
/// margin (`2 * lease`) cuts a member row under a training run; the
/// uncounted outcome RELEASES the lease, so no attempt ever waits for it to
/// expire.
const LEASE: Duration = Duration::from_secs(30);
const HEARTBEAT: Duration = Duration::from_secs(10);

/// The U4b gang oracle's fixture: eight `(anchor, positive)` rows.
pub(crate) fn pairs() -> Vec<(String, String)> {
    (0..8)
        .map(|i| (format!("anchor text {i}"), format!("positive text {i}")))
        .collect()
}

pub(crate) fn pairs_loader() -> TrainingDataLoader {
    TrainingDataLoader::from_pairs(pairs())
}

pub(crate) fn gang_config(epochs: usize) -> FineTuneConfig {
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

pub(crate) fn tiny_bert_model() -> String {
    "local:".to_string()
        + jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .unwrap()
}

pub(crate) fn write_pairs_csv(dir: &std::path::Path) -> String {
    let path = dir.join("pairs.csv");
    let mut body = String::from("anchor,positive\n");
    for (anchor, positive) in pairs() {
        body.push_str(&format!("{anchor},{positive}\n"));
    }
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

/// A peer-bound server that can COORDINATE and SERVE A RANK: `[worker]
/// enabled = false` (the test drives the claim itself), a serveable world
/// of two, a membership (`peer_advertise` set, so the engine's registration
/// carries a `MemberRoot` — the advertised address is never dialed), the
/// CSV source registered. The production `bind` mounted the gang listener
/// on `peer_addr` and installed the member dialer.
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

pub(crate) fn two_rank_spec() -> TrainingSpec {
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
/// row over the `fine_tune` kind. `addr` is the server's own `peer_bind`
/// listener: the member is this same process, serving its rank through the
/// production handler.
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
        .upsert_worker(id, "fine_tune", WorkerState::Claiming, &[])
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
pub(crate) struct Row {
    pub(crate) status: String,
    pub(crate) attempts: i32,
    pub(crate) releases: i32,
    pub(crate) lease_expires_at: Option<String>,
    pub(crate) assembly_failures: i32,
    pub(crate) next_assembly_after: Option<String>,
    pub(crate) training_set_ref: Option<String>,
    pub(crate) error: Option<String>,
}

pub(crate) async fn row(engine: &Arc<InferenceSession>, job_id: &str) -> Row {
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

/// The registered fine-tuned model row for `job_id`, if any.
async fn fine_tuned_model(
    engine: &Arc<InferenceSession>,
    job_id: &str,
) -> Option<jammi_db::catalog::model_repo::ModelRecord> {
    let models = engine.catalog().list_models().await.unwrap();
    models.into_iter().find(|m| {
        m.model_id
            .starts_with(&format!("jammi:fine-tuned:{job_id}"))
    })
}

pub(crate) async fn published_adapter_bytes(
    engine: &Arc<InferenceSession>,
    job_id: &str,
) -> Vec<u8> {
    let model = fine_tuned_model(engine, job_id)
        .await
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

pub(crate) fn file_store() -> Arc<ArtifactStore> {
    let root_dir = TempDir::new().unwrap().keep();
    let cache = TempDir::new().unwrap().keep();
    let root = StorageUrl::parse(root_dir.to_str().unwrap()).unwrap();
    Arc::new(ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap())
}

/// The reference: a two-rank `LocalGang` on the CPU, each rank driven
/// directly through `TrainingLoop::run` on its own `spawn_thread` (the U4b
/// gang oracle's shape); rank 0's `adapter.safetensors` bytes.
pub(crate) async fn reference_rank0_adapter_bytes(
    engine: &Arc<InferenceSession>,
    tag: &str,
    loader: fn() -> TrainingDataLoader,
) -> Vec<u8> {
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
        let (catalog, dir) = claimed_loop_env(&format!("{tag}-ref-{rank}")).await;
        let base = Arc::clone(&base);
        let store = Arc::clone(&store);
        let runtime = runtime.clone();
        let job_id = job_id.clone();
        threads.push(BlockingCall::spawn_thread(move |call| {
            let _runtime = runtime.enter();
            let config = gang_config(2);
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
            let head = build_projection_head_for_rank(
                hidden,
                &config,
                &varmap,
                &vb,
                rank_ctx.dropout_seed(config.seed),
            )
            .unwrap();
            let mut training_loop =
                TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
                    .device(Device::Cpu)
                    .job_id(job_id)
                    .worker_id(format!("reference-{rank}"))
                    .catalog(catalog)
                    .artifact_dir(dir.path().to_path_buf())
                    .base_model(base)
                    .artifact_store(store)
                    .rank_context(rank_ctx)
                    .build()
                    .unwrap();
            let result = training_loop
                .run(&call, TrainingSource::Resident(loader()))
                .unwrap_or_else(|e| panic!("reference rank {rank} must complete: {e}"));
            let bytes =
                std::fs::read(result.artifact_dir.path().join("adapter.safetensors")).unwrap();
            drop(dir);
            bytes
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

/// Wait until the engine's job slot is free — the member session ended
/// and its `RankHold` dropped.
async fn expect_slot_free(engine: &Arc<InferenceSession>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while engine.host_admission().holder() != Holder::Free {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the member session must end with the attempt and free its slot, holder: {:?}",
            engine.host_admission().holder()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The healthy gang, end to end (the module doc's first scenario).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_answering_unavailable_ends_the_attempt_cooled_and_the_next_attempt_runs_the_real_rank_body_to_a_published_artifact(
) {
    let server = coordinating_server().await;
    let engine = Arc::clone(&server.engine);
    register_member(&engine, "member-1", server.peer_addr).await;

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
        training_test_hooks::lease_holders_for(&job_id),
        vec![(1, LeaseHolder::Coordinator)],
        "a Peer gang's attempt registers its hold as the Coordinator"
    );
    assert_eq!(engine.host_admission().holder(), Holder::Free);

    // ── attempt 2: the member is free; the REAL rank body runs rank 1 ───
    let record = claim(&engine, &worker, Duration::from_secs(20)).await;
    assert_eq!(record.attempts, 2, "the next attempt, after the cooldown");
    worker.run_claimed_job(&engine, record).await;

    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 2, "{ends:?}");
    assert_eq!(ends[1].0, 2);
    assert_eq!(
        (ends[1].1.as_str(), ends[1].2),
        ("published", 12),
        "attempt 2 publishes over the real rank body: {}",
        ends[1].1
    );
    assert_eq!(
        training_test_hooks::assembly_listings_for(&job_id),
        vec![
            (1, vec![(1, "member-1".to_string())]),
            (2, vec![(1, "member-1".to_string())]),
        ],
        "the NEXT attempt re-listed and assigned the same sorted listing"
    );
    assert_eq!(
        training_test_hooks::lease_holders_for(&job_id),
        vec![(1, LeaseHolder::Coordinator), (2, LeaseHolder::Coordinator)]
    );
    // Both ranks' bodies ran in this process and recorded their roles:
    // rank 0 as the coordinator (attempt 2's only rank-0 run), rank 1 as
    // the member's body.
    let roles = training_test_hooks::runner_roles_for(&job_id);
    assert_eq!(
        roles
            .iter()
            .filter(|r| **r == RunnerRole::Holder(LeaseHolder::Coordinator))
            .count(),
        1,
        "{roles:?}"
    );
    assert_eq!(
        roles
            .iter()
            .filter(|r| **r == RunnerRole::Rank { rank: 1 })
            .count(),
        1,
        "{roles:?}"
    );
    assert!(
        !roles.contains(&RunnerRole::Holder(LeaseHolder::LoopClaimer)),
        "a Peer gang never runs as the loop claimer: {roles:?}"
    );
    // The terminal write on receipt: the coordinator read the member's
    // `Outcome{Trained}` and its digest equalled rank 0's own.
    let member_ends = training_test_hooks::member_ends_for(&job_id);
    assert_eq!(member_ends.len(), 1, "{member_ends:?}");
    assert_eq!(member_ends[0].0, 1);
    assert!(
        member_ends[0]
            .1
            .starts_with("Trained { artifact_digest: \""),
        "the member ended Trained: {}",
        member_ends[0].1
    );

    let after_two = row(&engine, &job_id).await;
    assert_eq!(after_two.status, "completed", "{after_two:?}");
    assert_eq!(after_two.error, None);
    assert_eq!(after_two.assembly_failures, 0);
    assert_eq!(
        after_two.next_assembly_after, None,
        "assembly proceeded to a run: Success resets the cooldown: {after_two:?}"
    );
    assert_eq!(after_two.attempts, 2);

    // The published artifact equals the LocalGang reference for the same
    // fixture: the loopback rounds through the real hold loop and the real
    // rank body folded exactly as Local does.
    let published = published_adapter_bytes(&engine, &job_id).await;
    let reference = reference_rank0_adapter_bytes(&engine, "peer-e2e", pairs_loader).await;
    assert_eq!(
        published, reference,
        "rank 0's published adapter over Peer must be byte-identical to Local's rank 0"
    );
    expect_slot_free(&engine).await;
}

/// The failure arm of the terminal write on receipt (the module doc's
/// second scenario).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_whose_body_fails_ends_the_attempt_failed_under_the_coordinator_and_publishes_nothing(
) {
    let server = coordinating_server().await;
    let engine = Arc::clone(&server.engine);
    register_member(&engine, "member-1", server.peer_addr).await;

    let job = engine
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job within the serveable world submits");
    let job_id = job.job_id.clone();
    training_test_hooks::fail_member_outcome(&job_id, "injected member failure");
    let worker = JobWorker::new(&engine).unwrap();

    let record = claim(&engine, &worker, Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    worker.run_claimed_job(&engine, record).await;

    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_eq!(
        ends[0].2, 11,
        "the attempt ends TrainingFailed on the member's Outcome{{Failed}}: {}",
        ends[0].1
    );
    assert_eq!(
        ends[0].1,
        "the run failed: rank 1: rank 1: injected member failure"
    );
    let member_ends = training_test_hooks::member_ends_for(&job_id);
    assert_eq!(
        member_ends,
        vec![(
            1,
            "Failed { reason: \"rank 1: injected member failure\" }".to_string()
        )]
    );
    let after = row(&engine, &job_id).await;
    assert_eq!(after.status, "failed", "{after:?}");
    assert!(
        after
            .error
            .as_deref()
            .is_some_and(|e| e.contains("rank 1: injected member failure")),
        "the member's reason is the row's error: {after:?}"
    );
    assert_eq!(
        after.next_assembly_after, None,
        "assembly proceeded to a run (Success): {after:?}"
    );
    assert!(
        fine_tuned_model(&engine, &job_id).await.is_none(),
        "nothing is published over a gang that did not complete"
    );
    assert_eq!(
        training_test_hooks::lease_holders_for(&job_id),
        vec![(1, LeaseHolder::Coordinator)]
    );
    let roles = training_test_hooks::runner_roles_for(&job_id);
    assert!(
        roles.contains(&RunnerRole::Holder(LeaseHolder::Coordinator))
            && roles.contains(&RunnerRole::Rank { rank: 1 }),
        "{roles:?}"
    );
    expect_slot_free(&engine).await;
}

// ─── GA8 exit (issue #538): a Peer graph gang is refused by name ──────────
//
// GA8 built the graph-arm Peer path (a member binding the materialised
// `GraphTrainingSet` table via the generic `bind_training_source` path),
// but a closing audit found it unsound by execution, not by inspection:
// reverting only GA1's two `ORDER BY`s made the SAME reference-vs-job byte
// comparison this file's other tests use go green, meaning the Peer
// member's own rank body has no path that reads the table in the SAME
// `_ordinal`-committed order rank 0's read uses — a member reading by a
// column-derived `ORDER BY` (the generic path the tabular arm's member
// takes) partitions a DIFFERENT row order of the SAME rows, so the two
// ranks would shard identically-named rows differently; and there is no
// executed oracle that the resulting per-rank shards combine into a
// correct all-reduced gradient over the graph arm's own loss (rank 0's
// bytes were unchanged when the member's rows were replaced with garbage
// or read reversed). Exited: a `graph_fine_tune` at `world_size > 1` still
// decides `Peer` (the topology decision itself is unchanged — this is a
// refusal downstream of it, not a different decision), but `run_spec`
// refuses it by name before any coordinator dial. The `Single` (`world ==
// 1`) and `Local` (in-process, `world <= local_ranks`) paths are
// unaffected and keep their own byte pins:
// `crates/jammi-ai/tests/it/gang_coordinator.rs`'s
// `a_local_ranks_two_host_fans_a_two_rank_job_out_through_run_spec_and_publishes_the_gangs_bytes`,
// `gang_placed.rs`'s `p2_the_stub_submitter_drives_a_real_run_placed_gang_to_the_same_bytes`,
// and `graph_finetune.rs`'s `fine_tune_graph_end_to_end_completes` (W=1).

/// A small, well-connected graph — a directed 8-cycle plus chords.
fn graph_nodes_edges() -> (
    Vec<jammi_ai::fine_tune::graph_sampler::TextNode>,
    Vec<jammi_ai::fine_tune::graph_sampler::GraphEdge>,
) {
    use jammi_ai::fine_tune::graph_sampler::{GraphEdge, TextNode};
    let n = 8;
    let nodes = (0..n)
        // Every node's text is distinct in-vocabulary tokens for the tiny test
        // model: a text it tokenizes to `[UNK]` makes every sampled row the
        // same row, and no order or shard fault could then change the bytes.
        .map(|i| {
            TextNode::new(
                format!("g{i}"),
                format!("{i} {} {}", (i * 3 + 1) % 8, (i * 5 + 2) % 8),
            )
        })
        .collect();
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push(GraphEdge::declared(
            format!("g{i}"),
            format!("g{}", (i + 1) % n),
        ));
        edges.push(GraphEdge::declared(
            format!("g{}", (i + 1) % n),
            format!("g{i}"),
        ));
    }
    (nodes, edges)
}

/// The in-memory reference for the graph fixture: the same nodes and edges,
/// in the read order the job's own scans commit, through the same seeded
/// sampler — never the job's table.
fn graph_loader() -> TrainingDataLoader {
    use jammi_ai::fine_tune::graph_sampler::{sort_into_graph_read_order, GraphSampler};
    let (mut nodes, mut edges) = graph_nodes_edges();
    sort_into_graph_read_order(&mut nodes, &mut edges);
    let sampler = GraphSampler::build(nodes, edges, graph_sample_config()).unwrap();
    TrainingDataLoader::from_graph(&sampler).unwrap()
}

fn graph_sample_config() -> jammi_ai::fine_tune::graph_sampler::GraphSampleConfig {
    jammi_ai::fine_tune::graph_sampler::GraphSampleConfig {
        walk_length: 2,
        walks_per_node: 1,
        hard_negatives: 0,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 7,
        ..Default::default()
    }
}

fn write_graph_csvs(dir: &std::path::Path) -> (String, String) {
    let (nodes, edges) = graph_nodes_edges();
    let node_path = dir.join("graph_nodes.csv");
    let mut node_body = String::from("id,text\n");
    for n in &nodes {
        node_body.push_str(&format!("{},{}\n", n.id, n.text));
    }
    std::fs::write(&node_path, node_body).unwrap();

    let edge_path = dir.join("graph_edges.csv");
    let mut edge_body = String::from("src,dst\n");
    for e in &edges {
        edge_body.push_str(&format!("{},{}\n", e.src, e.dst));
    }
    std::fs::write(&edge_path, edge_body).unwrap();

    (
        format!("file://{}", node_path.display()),
        format!("file://{}", edge_path.display()),
    )
}

/// [`register_member`]'s graph-arm counterpart: a member claiming the
/// `graph_fine_tune` kind, not `fine_tune` — present and available, so the
/// refusal test below proves a DELIBERATE typed width block, never merely
/// "no member was found".
async fn register_graph_member(
    engine: &Arc<InferenceSession>,
    id: &str,
    addr: std::net::SocketAddr,
) {
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
        .upsert_worker(id, "graph_fine_tune", WorkerState::Claiming, &[])
        .await
        .unwrap();
}

/// [`claim`]'s graph-arm counterpart: claims `graph_fine_tune`, not
/// `fine_tune`.
async fn claim_graph(
    engine: &Arc<InferenceSession>,
    worker: &JobWorker,
    within: Duration,
) -> JobRecord {
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
            .claim_next(worker.worker_id(), &["graph_fine_tune"], LEASE)
            .await
            .unwrap()
        {
            return record;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the graph job was not claimable within {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn coordinating_graph_server() -> crate::common::grpc::PeerEngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = crate::common::grpc::peer_bind_config(dir.path());
    cfg.worker.enabled = false;
    cfg.lease.duration_secs = LEASE.as_secs();
    cfg.lease.heartbeat_secs = HEARTBEAT.as_secs();
    cfg.distributed.max_world_size = 2;
    cfg.server.peer_advertise = Some("127.0.0.1:1".into());
    let (node_url, edge_url) = write_graph_csvs(dir.path());
    let server = crate::common::grpc::start_engine_server_from_config(cfg, Some(dir)).await;
    server
        .engine
        .add_source(
            "graph_nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    server
        .engine
        .add_source(
            "graph_edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    server
}

fn two_rank_graph_spec() -> TrainingSpec {
    use jammi_ai::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources};
    TrainingSpec::GraphFineTune {
        sources: GraphFineTuneSources {
            node_source: "graph_nodes".into(),
            id_column: "id".into(),
            text_column: "text".into(),
            edge_source: "graph_edges".into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            provenance: EdgeProvenance::Declared,
        },
        sample_config: graph_sample_config(),
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: gang_config(2),
            world_size: 2,
        },
    }
}

/// A `graph_fine_tune` at `world_size = 2` runs as a real `Peer` gang: the
/// member binds the graph-sampled training set by the identity on the job
/// row, reads it in its committed order through the same rank body a
/// column-source job uses, and ends `Trained` with the digest the
/// coordinator's terminal write on receipt requires to equal rank 0's own.
/// That digest equality shows the ranks agree with each other; the byte
/// equality against an in-process gang over the in-memory sample shows they
/// cut and read the right shards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn graph_fine_tune_runs_as_a_peer_gang() {
    let server = coordinating_graph_server().await;
    let engine = Arc::clone(&server.engine);
    register_graph_member(&engine, "member-1", server.peer_addr).await;

    let job = engine
        .run_training_spec(two_rank_graph_spec())
        .await
        .unwrap();
    let job_id = job.job_id.clone();
    let worker = JobWorker::new(&engine).unwrap();

    let record = claim_graph(&engine, &worker, Duration::from_secs(20)).await;
    worker.run_claimed_job(&engine, record).await;

    assert_eq!(
        training_test_hooks::topology_for(&job_id),
        Some(TopologyDecision::Peer { world: 2 })
    );
    let after = row(&engine, &job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
    assert_eq!(after.error, None);

    let roles = training_test_hooks::runner_roles_for(&job_id);
    assert!(
        roles.contains(&RunnerRole::Holder(LeaseHolder::Coordinator))
            && roles.contains(&RunnerRole::Rank { rank: 1 }),
        "rank 0 ran as the coordinator and rank 1 as the member's body: {roles:?}"
    );
    let member_ends = training_test_hooks::member_ends_for(&job_id);
    assert_eq!(member_ends.len(), 1, "{member_ends:?}");
    assert!(
        member_ends[0]
            .1
            .starts_with("Trained { artifact_digest: \""),
        "the member ended Trained: {}",
        member_ends[0].1
    );
    let published = published_adapter_bytes(&engine, &job_id).await;
    let reference = reference_rank0_adapter_bytes(&engine, "graph-peer", graph_loader).await;
    assert!(!published.is_empty());
    assert_eq!(
        published, reference,
        "rank 0's published adapter over Peer must be byte-identical to an in-process gang \
         over the in-memory graph sample"
    );
    expect_slot_free(&engine).await;
}
