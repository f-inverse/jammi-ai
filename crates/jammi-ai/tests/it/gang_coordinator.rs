//! The coordinator body and the topology fan-out, driven through the REAL
//! claim → `run_claimed_job` → `run_spec` path over a hermetic session (no
//! server, no member: what a library process can do).
//!
//! Three properties:
//!
//! - (d) a `world_size` within the serveable world but beyond this host's
//!   own devices SUBMITS and REACHES ASSEMBLY: with no fleet member to list,
//!   the attempt ends `ShortListed` — recorded on the row (cooled, NOT
//!   counted), the lease handed back, the training-set identity pair
//!   written by the CAS, nothing terminal;
//! - the CAS `Moved` arm exits with NO write: a claim whose attempt moved
//!   under the coordinator leaves every assembly and identity column exactly
//!   as it found them;
//! - the local fan-out: a `local_ranks = 2` host runs a two-rank job through
//!   the real `run_spec` as an in-process `Local` gang and publishes an
//!   adapter whose bytes EQUAL a two-rank `LocalGang` run of the same
//!   fixture (same rows, config, seed, base model) driven directly through
//!   `TrainingLoop::run` — the artifact is the oracle, not a hook. The job
//!   is a `graph_fine_tune`: the graph arm is the one `Resident` source
//!   `run_spec` binds for a training job (a column-source `fine_tune` binds
//!   `Streamed`, which the trainer refuses at `world > 1` — the same rows
//!   through the same `Local` gang are what the property is about, not the
//!   arm).

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::data::TrainingDataLoader;
use jammi_ai::fine_tune::graph_sampler::{
    sort_into_graph_read_order, EdgeProvenance, GraphEdge, GraphFineTuneSources, GraphSampleConfig,
    GraphSampler, TextNode,
};
use jammi_ai::fine_tune::role::{LeaseHolder, RunnerRole};
use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::{training_test_hooks, JobWorker, TopologyDecision};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use tempfile::TempDir;

use crate::common;
pub(crate) use crate::gang_fixtures::{
    gang_config, gang_config_with_dropout, reference_rank0_adapter_bytes, tiny_bert_model,
    two_rank_spec, write_pairs_csv,
};

/// The graph fixture (`graph_finetune.rs`'s end-to-end shape): six nodes in
/// two triangles joined by one bridge, edges directed both ways.
pub(crate) fn graph_nodes() -> Vec<(String, String)> {
    ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .enumerate()
        // Text the tiny model can tell apart, node from node: were two nodes
        // one token sequence, every sampled row would be the same row, and no
        // order or shard fault could change the bytes.
        .map(|(index, id)| {
            (
                id.to_string(),
                jammi_test_utils::tiny_vocab_text('g', index),
            )
        })
        .collect()
}

pub(crate) fn graph_edges() -> Vec<(String, String)> {
    [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ]
    .iter()
    .map(|(s, d)| (s.to_string(), d.to_string()))
    .collect()
}

pub(crate) fn write_csv(
    dir: &std::path::Path,
    name: &str,
    header: &str,
    rows: &[(String, String)],
) -> String {
    let mut body = String::from(header);
    body.push('\n');
    for (a, b) in rows {
        body.push_str(&format!("{a},{b}\n"));
    }
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

/// Pairs only (no mined negatives), a fixed seed: the sampled rows are a
/// pure function of the graph and this config on both sides.
fn graph_sample_config() -> GraphSampleConfig {
    GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 0,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    }
}

fn graph_sources() -> GraphFineTuneSources {
    GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    }
}

/// The loader the worker's `materialize_graph_training_set` builds — the same
/// nodes and edges the real job's two ordered scans read
/// (`sort_into_graph_read_order`, `GRAPH_READ_ORDER_RULE_V1`), through the
/// same seeded sampler. `graph_nodes`/`graph_edges` are declared in a
/// bidirected shape (two triangles + a bridge), NOT already `(id,
/// text)`/`(src, dst)`-sorted — sorting here is required, not cosmetic:
/// without it the reference's bytes do not match the job's.
pub(crate) fn graph_loader() -> TrainingDataLoader {
    let mut nodes: Vec<TextNode> = graph_nodes()
        .into_iter()
        .map(|(id, text)| TextNode::new(id, text))
        .collect();
    let mut edges: Vec<GraphEdge> = graph_edges()
        .into_iter()
        .map(|(src, dst)| GraphEdge {
            src,
            dst,
            provenance: EdgeProvenance::Declared,
        })
        .collect();
    sort_into_graph_read_order(&mut nodes, &mut edges);
    let sampler = GraphSampler::build(nodes, edges, graph_sample_config()).unwrap();
    TrainingDataLoader::from_graph(&sampler).unwrap()
}

/// The fan-out fixture: `lora_dropout = 0.3`, so the per-rank dropout seed
/// is a live determinant of the run (and of the byte equality below).
pub(crate) fn fan_out_config() -> FineTuneConfig {
    gang_config_with_dropout(2, 0.3)
}

pub(crate) fn two_rank_graph_spec() -> TrainingSpec {
    TrainingSpec::GraphFineTune {
        sources: graph_sources(),
        sample_config: graph_sample_config(),
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: fan_out_config(),
            world_size: 2,
        },
    }
}

/// A session over a deployment that can coordinate: a serveable world of
/// two, fast lease timing, and a membership (`peer_bind` + `peer_advertise`,
/// so the registration carries a `MemberRoot` — the address is never
/// dialed by anything here). `tune` adjusts the rest.
pub(crate) async fn coordinating_session(
    tune: impl FnOnce(&mut JammiConfig),
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.distributed.max_world_size = 2;
    config.lease.duration_secs = 3;
    config.lease.heartbeat_secs = 1;
    config.worker.idle_poll_secs = 1;
    config.server.peer_bind = Some("127.0.0.1:0".into());
    config.server.peer_advertise = Some("127.0.0.1:1".into());
    tune(&mut config);
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    for (name, url) in [
        ("pairs", write_pairs_csv(dir.path())),
        (
            "nodes",
            write_csv(dir.path(), "nodes.csv", "id,text", &graph_nodes()),
        ),
        (
            "edges",
            write_csv(dir.path(), "edges.csv", "src,dst", &graph_edges()),
        ),
    ] {
        session
            .add_source(
                name,
                SourceType::File,
                SourceConnection {
                    url: Some(url),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }
    (session, dir)
}

/// Submit through the real submit edge and claim as this session's own
/// worker — the record `run_claimed_job` takes.
pub(crate) async fn submit_and_claim(
    session: &Arc<InferenceSession>,
    worker: &JobWorker,
    spec: TrainingSpec,
) -> JobRecord {
    let job = session
        .run_training_spec(spec)
        .await
        .expect("a two-rank job within the serveable world submits");
    let record = session
        .catalog()
        .claim_next(
            worker.worker_id(),
            &["fine_tune", "graph_fine_tune"],
            Duration::from_secs(3),
        )
        .await
        .unwrap()
        .expect("the queued job is claimable");
    assert_eq!(record.job_id, job.job_id);
    record
}

/// Every `jobs` column the coordinator body may move, by primary key
/// through raw SQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) status: String,
    pub(crate) claimed_by: Option<String>,
    pub(crate) attempts: i32,
    pub(crate) releases: i32,
    pub(crate) lease_expires_at: Option<String>,
    pub(crate) assembly_failures: i32,
    pub(crate) next_assembly_after: Option<String>,
    pub(crate) training_set_ref: Option<String>,
    pub(crate) training_set_location: Option<String>,
    pub(crate) error: Option<String>,
}

pub(crate) async fn row(catalog: &jammi_db::catalog::Catalog, job_id: &str) -> Row {
    let job_id = job_id.to_string();
    catalog
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
                        "SELECT status, claimed_by, attempts, releases, lease_expires_at, \
                             assembly_failures, next_assembly_after, training_set_ref, \
                             training_set_location, error \
                         FROM jobs WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                        |row| {
                            Ok(Row {
                                status: row.get("status")?,
                                claimed_by: row.try_get("claimed_by")?,
                                attempts: row.get("attempts")?,
                                releases: row.get("releases")?,
                                lease_expires_at: row.try_get("lease_expires_at")?,
                                assembly_failures: row.get("assembly_failures")?,
                                next_assembly_after: row.try_get("next_assembly_after")?,
                                training_set_ref: row.try_get("training_set_ref")?,
                                training_set_location: row.try_get("training_set_location")?,
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

/// The bytes of the adapter the worker published for `job_id` — fetched
/// from the artifact store through the registered model row's own
/// artifact reference, exactly as serving would.
pub(crate) async fn published_adapter_bytes(
    session: &Arc<InferenceSession>,
    job_id: &str,
) -> Vec<u8> {
    let models = session.catalog().list_models().await.unwrap();
    let model = models
        .iter()
        .find(|m| {
            m.model_id
                .starts_with(&format!("jammi:fine-tuned:{job_id}"))
        })
        .expect("the completed job registered its fine-tuned model");
    let local = session
        .artifact_store()
        .fetch_artifact(&crate::common::served_bundle_url(model))
        .await
        .expect("the published adapter fetches and verifies");
    std::fs::read(local.dir().join("adapter.safetensors")).unwrap()
}

/// Acceptance (d), the assembly half: a two-rank job on a ONE-device host
/// within a serveable world of two submits, is claimed, and reaches the
/// coordinator body — `TopologyDecision::Peer { world: 2 }` — which lists
/// no member and ends `ShortListed`: the outcome is recorded (cooled, NOT
/// counted), the lease is handed back for the next attempt, the CAS wrote
/// the training-set identity pair, and nothing terminal was written.
#[tokio::test(flavor = "multi_thread")]
async fn a_two_rank_job_beyond_this_hosts_devices_reaches_assembly_and_lands_short_listed() {
    let (session, _dir) = coordinating_session(|_| {}).await;
    assert_eq!(session.inner_config().gpu.device_list().len(), 1);
    let worker = JobWorker::new(&session).unwrap();
    let record = submit_and_claim(&session, &worker, two_rank_spec()).await;
    let job_id = record.job_id.clone();

    worker.run_claimed_job(&session, record).await;

    assert_eq!(
        training_test_hooks::topology_for(&job_id),
        Some(TopologyDecision::Peer { world: 2 }),
        "a two-rank job on a local_ranks = 1 host is the coordinator body's"
    );
    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 1, "one attempt, one end: {ends:?}");
    assert_eq!(ends[0].0, 1, "the first attempt");
    assert_eq!(
        ends[0].1, "short listing: 0 fresh member(s) where 1 are needed",
        "no fleet member exists to list"
    );
    assert!(
        training_test_hooks::assembly_listings_for(&job_id).is_empty(),
        "a short listing never becomes an assignment"
    );

    let after = row(session.catalog(), &job_id).await;
    assert_eq!(
        after.status, "running",
        "nothing terminal: left for reclaim"
    );
    assert_eq!(after.error, None);
    assert_eq!(after.attempts, 1);
    assert_eq!(after.assembly_failures, 0, "ShortListed is not counted");
    assert!(
        after.next_assembly_after.is_some(),
        "ShortListed is cooled down: {after:?}"
    );
    assert_eq!(
        after.releases, 1,
        "an uncounted outcome hands the lease back at once"
    );
    assert_eq!(
        after.lease_expires_at, None,
        "the lease is NULL: reclaimable now"
    );
    assert!(
        after.training_set_ref.is_some() && after.training_set_location.is_some(),
        "the CAS wrote the pair before membership was listed: {after:?}"
    );
    assert_eq!(after.claimed_by.as_deref(), Some(worker.worker_id()));
}

/// The CAS `Moved` arm: the claim moved (a successor's reclaim bumped
/// `attempts` and took the row) between the claim this worker holds and its
/// coordinator body's CAS — the attempt exits with NO write: no assembly
/// outcome, no pair, no release, nothing terminal.
#[tokio::test(flavor = "multi_thread")]
async fn a_moved_claim_exits_the_coordinator_body_with_no_write() {
    let (session, _dir) = coordinating_session(|_| {}).await;
    let worker = JobWorker::new(&session).unwrap();
    let record = submit_and_claim(&session, &worker, two_rank_spec()).await;
    let job_id = record.job_id.clone();

    // The row moves under the stale claim: a successor at attempt 2.
    let moved_job = job_id.clone();
    session
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let moved_job = moved_job.clone();
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET attempts = attempts + 1, claimed_by = 'successor' \
                     WHERE job_id = $1",
                    &[SqlValue::TextOwned(moved_job)],
                )
                .await
            })
        })
        .await
        .unwrap();
    let before = row(session.catalog(), &job_id).await;
    assert_eq!(before.attempts, 2);

    worker.run_claimed_job(&session, record).await;

    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_eq!(ends[0].1, "the claim moved before assembly (no write)");
    let after = row(session.catalog(), &job_id).await;
    assert_eq!(after, before, "a moved claim writes nothing at all");
    assert_eq!(after.assembly_failures, 0);
    assert_eq!(after.next_assembly_after, None);
    assert_eq!(after.training_set_ref, None);
    assert_eq!(after.releases, 0);
    assert_eq!(after.status, "running");
}

/// The local fan-out: on a `local_ranks = 2` host with two declared devices
/// (real ordinals, degrading to the CPU off the pod — both ranks on one
/// device), a two-rank `graph_fine_tune` runs through the REAL `run_spec`
/// as an in-process `Local` gang (`TopologyDecision::Local { world: 2 }`,
/// no coordinator body), completes, and publishes an adapter whose bytes
/// equal the two-rank `LocalGang` reference's rank-0 adapter over the
/// same sampled rows (`TrainingDataLoader::from_graph` over the same seeded
/// sampler), config, seed and base model.
#[tokio::test(flavor = "multi_thread")]
async fn a_local_ranks_two_host_fans_a_two_rank_job_out_through_run_spec_and_publishes_the_gangs_bytes(
) {
    let (session, _dir) = coordinating_session(|config| {
        config.gpu.device = 0;
        config.gpu.devices = Some(vec![0, 1]);
        config.worker.local_ranks = 2;
        // The gang deadline: a rank left without its peer (a mutation that
        // gives rank 0 a single-rank context, or spawns no rank 1) fails
        // within seconds, never the 120 s default.
        config.worker.rank_timeout_secs = 10;
    })
    .await;
    let worker = JobWorker::new(&session).unwrap();
    let record = submit_and_claim(&session, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();

    worker.run_claimed_job(&session, record).await;

    assert_eq!(
        training_test_hooks::topology_for(&job_id),
        Some(TopologyDecision::Local { world: 2 })
    );
    assert!(
        training_test_hooks::coordinator_ends_for(&job_id).is_empty(),
        "an in-process gang never enters the coordinator body"
    );
    // The roles: the claimant runs the whole job in-process — its hold is
    // the loop claimer's, rank 0 runs as `Holder(LoopClaimer)`, rank 1 as
    // `Rank { 1 }` — never the coordinator.
    assert_eq!(
        training_test_hooks::lease_holders_for(&job_id),
        vec![(1, LeaseHolder::LoopClaimer)]
    );
    let mut roles = training_test_hooks::runner_roles_for(&job_id);
    roles.sort_by_key(|r| r.rank());
    assert_eq!(
        roles,
        vec![
            RunnerRole::Holder(LeaseHolder::LoopClaimer),
            RunnerRole::Rank { rank: 1 }
        ]
    );
    let after = row(session.catalog(), &job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
    assert_eq!(after.error, None);

    // The seed split, through the real `run_spec`: the two
    // ranks' dropout seeds differ (rank 0 keeps `config.seed`, the identity)
    // and every head layer's own dropout Philox seed is the one its rank was
    // given, while the ranks' pre-step adapter weights are byte-identical —
    // the A/B init is keyed by `config.seed` on every rank.
    let mut targets = training_test_hooks::rank_targets_for(&job_id);
    targets.sort_by_key(|t| t.rank);
    assert_eq!(
        targets.iter().map(|t| t.rank).collect::<Vec<_>>(),
        vec![0, 1],
        "one target per rank, both through run_fine_tune_blocking: {targets:?}"
    );
    assert_eq!(
        targets[0].dropout_seed,
        fan_out_config().seed,
        "rank 0 is the identity"
    );
    assert_ne!(
        targets[0].dropout_seed, targets[1].dropout_seed,
        "rank 1 draws its own dropout seed"
    );
    for target in &targets {
        assert!(
            !target.layer_dropout_seeds.is_empty(),
            "a projection head has layers"
        );
        assert!(
            target
                .layer_dropout_seeds
                .iter()
                .all(|s| *s == Some(target.dropout_seed)),
            "every layer's dropout run seed is the rank's own: {target:?}"
        );
    }
    assert_eq!(
        targets[0].weights_digest, targets[1].weights_digest,
        "both ranks start from byte-identical adapter weights"
    );

    let published = published_adapter_bytes(&session, &job_id).await;
    let reference =
        reference_rank0_adapter_bytes(&session, "local-fanout", graph_loader, fan_out_config())
            .await;
    assert!(!published.is_empty());
    assert_eq!(
        published, reference,
        "the fan-out's published adapter must be byte-identical to the LocalGang reference"
    );
}

/// A crashed coordinator's live `building` training-set row is NEVER met
/// by the successor, so the training path needs no `BackOff`
/// disposition. The
/// training-set producer names every table uniquely (`ResultStore::
/// create_table`: `{source}__{task}__{model}__{nanos}_{uuid}`), anchors a
/// registered source `UnpinnedAtInstant` (`training_set::
/// materialize_projection_table`), so its reuse probe short-circuits
/// (`exact_match_candidates`) and only `ready` rows are ever candidates;
/// and the job row's write-once pair is recorded AFTER the table is
/// `ready` (`run_spec` builds the pair from the finished table, then the
/// coordinator body's CAS writes it), so a retry binds a `ready` table by
/// name or materializes anew. Here: a live `building` row over the SAME
/// source and task, its lease renewed by this process's keeper (the handle
/// held, never finished — the crashed writer's row) exists while the
/// successor attempt runs; the attempt materializes ITS OWN table, records
/// a different name on the row, reaches the coordinator body, and leaves
/// the orphan exactly as it found it (`building`, the same writer, its
/// lease live) for the lease to reap after expiry. No `BackOff` disposition
/// exists on the training path because no attempt can reach the state it
/// would answer.
#[tokio::test(flavor = "multi_thread")]
async fn a_live_building_training_set_row_left_by_a_crashed_coordinator_is_never_met_by_the_successor(
) {
    use jammi_db::catalog::result_repo::ResultTableKind;
    use jammi_db::store::TRAINING_SET_MODEL_ID;

    let (session, _dir) = coordinating_session(|_| {}).await;
    let worker = JobWorker::new(&session).unwrap();

    // The crashed coordinator's row: `building` over the same source and
    // task, held under a live, renewing lease.
    let orphan = session
        .result_store()
        .create_table(
            "pairs",
            ModelTask::TextEmbedding,
            ResultTableKind::TrainingSet,
            None,
            TRAINING_SET_MODEL_ID,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("the orphan's building row");
    let orphan_name = orphan.table_name().to_string();
    let before = session
        .catalog()
        .get_result_table(&orphan_name)
        .await
        .unwrap()
        .expect("the orphan row exists");
    assert_eq!(before.status, "building");
    assert!(
        before.lease_expires_at.is_some(),
        "a live lease: {before:?}"
    );
    assert_eq!(before.writer_id.as_deref(), Some(orphan.writer_id()));

    let record = submit_and_claim(&session, &worker, two_rank_spec()).await;
    let job_id = record.job_id.clone();
    worker.run_claimed_job(&session, record).await;

    // The successor attempt built and recorded its OWN ready table and
    // reached the coordinator body (short-listed: no member exists here).
    let ends = training_test_hooks::coordinator_ends_for(&job_id);
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_eq!(
        ends[0].1,
        "short listing: 0 fresh member(s) where 1 are needed"
    );
    let after = row(session.catalog(), &job_id).await;
    let own_name = after
        .training_set_location
        .clone()
        .expect("the CAS recorded the attempt's own table");
    assert_ne!(
        own_name, orphan_name,
        "the successor never binds the orphan: it materializes its own table"
    );
    let own = session
        .catalog()
        .get_result_table(&own_name)
        .await
        .unwrap()
        .expect("the attempt's own row");
    assert_eq!(own.status, "ready");

    // The orphan is exactly as it was: never promoted, failed, deleted or
    // claimed — the lease's to reap after expiry.
    let orphan_after = session
        .catalog()
        .get_result_table(&orphan_name)
        .await
        .unwrap()
        .expect("the orphan row still exists");
    assert_eq!(orphan_after.status, "building");
    assert_eq!(orphan_after.writer_id, before.writer_id);
    assert!(
        orphan_after.lease_expires_at.is_some(),
        "the orphan's lease is still live: {orphan_after:?}"
    );
    orphan.abort().await.expect("the fixture's own abort");
}

/// A `world_size == 1` job through the REAL claim → `run_claimed_job` →
/// `run_spec` path is the plain loop path —
/// the topology is `Single`, the attempt's hold is registered as the
/// `LoopClaimer`, rank 0 runs as `Holder(LoopClaimer)` and nothing else
/// runs, the coordinator body is never entered, and the row completes
/// through the loop's own finalize. A `TopologyDecision::decide` that
/// answered `Peer` for `world_size <= 1` would send the job through the
/// coordinator (a `ShortListed` end, the `Coordinator` role).
#[tokio::test(flavor = "multi_thread")]
async fn a_single_rank_job_runs_as_the_loop_claimer_and_never_traverses_the_coordinator() {
    let (session, _dir) = coordinating_session(|_| {}).await;
    let worker = JobWorker::new(&session).unwrap();
    let spec = TrainingSpec::FineTune {
        source: "pairs".into(),
        columns: vec!["anchor".into(), "positive".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: gang_config(1),
            world_size: 1,
        },
        cache: CachePolicy::Bypass,
    };
    let record = submit_and_claim(&session, &worker, spec).await;
    let job_id = record.job_id.clone();

    worker.run_claimed_job(&session, record).await;

    assert_eq!(
        training_test_hooks::topology_for(&job_id),
        Some(TopologyDecision::Single)
    );
    assert!(
        training_test_hooks::coordinator_ends_for(&job_id).is_empty(),
        "W == 1 never traverses the coordinator body"
    );
    assert!(training_test_hooks::assembly_listings_for(&job_id).is_empty());
    assert_eq!(
        training_test_hooks::lease_holders_for(&job_id),
        vec![(1, LeaseHolder::LoopClaimer)],
        "the single-rank attempt's hold is the loop claimer's"
    );
    assert_eq!(
        training_test_hooks::runner_roles_for(&job_id),
        vec![RunnerRole::Holder(LeaseHolder::LoopClaimer)],
        "exactly one rank ran, as the loop claimer"
    );
    let after = row(session.catalog(), &job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
    assert_eq!(after.error, None);
    assert_eq!(
        after.training_set_ref, None,
        "no coordinator CAS ever wrote the identity pair: {after:?}"
    );
    assert!(!published_adapter_bytes(&session, &job_id).await.is_empty());
}
