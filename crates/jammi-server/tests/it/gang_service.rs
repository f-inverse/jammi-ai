//! `GangService`'s admission-wire tests — `HostAdmission` end to end.
//!
//! The tests below drive the real `RunRank` rpc over the production
//! `peer_bind` listener (`start_engine_server_with_peer_bind` /
//! `start_no_worker_server`): the wire-level edges (`world == 0`,
//! `rank >= world`, refused before I-GANG runs), ambient admin scope
//! (refused before any row is even read), every I-GANG determinant
//! `get_job_for_rank` decides (job not found, not `running`, wrong
//! claimant, wrong attempt, lease not live, an undecodable `world_size`,
//! `assign.world != row.world_size`), the world>1 conjunct (the
//! training-set pair, the row's own tenant pinning a strict resolution, and
//! the sidecar verify — every determinant refusing identically on the wire
//! and distinctly under `test-hooks`, with an ADMITTING control that
//! genuinely resolves for the job's own tenant), the coordinator's
//! freshness, and then admission itself: a call satisfying every
//! determinant receives `Admitted` and is HELD under re-verification; a
//! `world_size == 1` session has no rank body and — absent an earlier exit
//! — ends `Aborted{NoBody}` at the park bound; a `world_size > 1` session
//! hands its link to the REAL rank body (`run_member_rank`) and never
//! parks: the body's end is the session's end.
//!
//! The held session's arms each have their own rows: `Cancel`
//! (`Aborted{Cancelled}`), a second `Assign` (`InvalidArgument`), the
//! host's DRAIN (`Aborted{Drain}`, both through the session's own
//! `HostAdmission` and through the real server shutdown path), the
//! re-verification tick's three ends (`Refuted` / `Unavailable` /
//! `StoreUnavailable`, pairwise distinct on the wire, each manufactured
//! AFTER admission — on `world_size == 2` sessions whose body is alive at
//! its first collective when the tick fires), the park bound, and the
//! body-bearing session that outlives the park bound. Holder contention (a running loop
//! job, a claim probe waited out, another rank, a duplicate assignment, and
//! the same job's greater attempt taking the slot) refuses `Unavailable`
//! or supersedes exactly as the lattice states. Every session end leaves
//! the job row untouched (`row_facts` snapshots, before and after): the
//! peer writes nothing terminal on behalf of a rank.
//!
//! `run_rank_refusal_is_non_disclosing_across_every_determinant`
//! is the ONE table-driven non-disclosure oracle — every I-GANG determinant
//! refuses with the pairwise-identical `(code, message)`. The
//! `test-hooks`-gated `run_rank_last_refusal_reason_distinguishes_every_determinant`
//! drives the exact SAME scenarios and asserts `GangServer::last_refusal_reason`
//! (via `PeerEngineServer::gang_last_refusal_reason`, or — for the
//! `AdminScope` determinant, which no network client can ever trigger — a
//! standalone `GangServer` invoked in-process, see `refusal_scenario`'s own
//! doc) distinguishes every one of them same-process — the plain lane and
//! the `test-hooks` lane therefore run a DIFFERENT number of gang-prefixed
//! test cases (the holder-contention rows that manufacture a holder through
//! `HostAdmission::hold_for_test` are `test-hooks` only as well).

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use jammi_db::TenantId;
use jammi_wire::proto::gang::{rank_event, AbortReason, RankControl, RankEvent};

// ---------------------------------------------------------------------------
// Fixtures shared by every row below.
// ---------------------------------------------------------------------------

fn assign_frame(world: u32, rank: u32) -> jammi_wire::proto::gang::RankControl {
    assign_frame_full("job-1", 0, rank, world, "coord-1")
}

pub(crate) fn assign_frame_full(
    job_id: &str,
    attempt: i64,
    rank: u32,
    world: u32,
    coordinator_instance_id: &str,
) -> jammi_wire::proto::gang::RankControl {
    use jammi_wire::proto::gang::{rank_control, Assign, RankControl};
    RankControl {
        control: Some(rank_control::Control::Assign(Assign {
            job_id: job_id.into(),
            attempt,
            rank,
            world,
            coordinator_instance_id: coordinator_instance_id.into(),
        })),
    }
}

fn cancel_frame() -> RankControl {
    use jammi_wire::proto::gang::{rank_control, Cancel};
    RankControl {
        control: Some(rank_control::Control::Cancel(Cancel {})),
    }
}

/// The deployment lease window every peer-bound fixture below runs with:
/// the held session's PARK BOUND (`Aborted{NoBody}` absent an earlier
/// exit) and `fresh_instance`'s margin input. Short, so a park is observed
/// in seconds; strictly more than twice [`HEARTBEAT`], as the config
/// validator requires.
pub(crate) const LEASE: Duration = Duration::from_secs(3);
/// The re-verification cadence and the longest a `ClaimProbe` is waited
/// on.
pub(crate) const HEARTBEAT: Duration = Duration::from_secs(1);

/// A peer-bound server with `[worker] enabled = false` — every fixture below
/// drives `Catalog::submit_job`/`claim_next` directly (matching
/// `jobs_queue.rs`'s own fixture style) and needs the row's `status`/
/// `claimed_by`/`attempts` to stay exactly what the fixture set, never
/// raced by this same process's own production claim loop (`[worker] enabled`
/// defaults to `true`, `config/mod.rs`) — and with the fast `[lease]`
/// timing above.
pub(crate) async fn start_no_worker_server() -> crate::common::grpc::PeerEngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = crate::common::grpc::peer_bind_config(dir.path());
    cfg.worker.enabled = false;
    cfg.lease.duration_secs = LEASE.as_secs();
    cfg.lease.heartbeat_secs = HEARTBEAT.as_secs();
    crate::common::grpc::start_engine_server_from_config(cfg, Some(dir)).await
}

/// A job whose `spec` names no `world_size` at all — `RankAdmissionRow`
/// decodes it to `1`, `jammi_db`'s own `WORLD_SIZE_IF_ABSENT` default (see
/// `jobs_repo.rs`), the shape a non-training job kind (or a spec predating
/// `world_size`) persists.
pub(crate) const WORLD1_SPEC: &str = "{}";

/// A job whose `spec` names `world_size: 2` under the `common` key — the
/// SAME shape `jammi-ai`'s `TrainingCommon` actually persists, and the SAME
/// literal `crates/jammi-db/tests/it/gang_rank_admission.rs` uses for its own
/// `get_job_for_rank_reflects_the_row_world_size` fixture. Every
/// `world_size > 1` test in this file submits with THIS spec — never a
/// `world_size == 1` spec paired with an `Assign.world == 2`, which the
/// world-mismatch conjunct refuses before the pair conjunct or the sidecar
/// verify ever runs.
const WORLD2_SPEC: &str = r#"{"common":{"world_size":2}}"#;

/// The freshness fixture: the coordinator's own `instances` row, upserted
/// so the "coordinator not fresh" determinant is satisfied and every other
/// row isolates the ONE determinant it names.
pub(crate) async fn fresh_coordinator(server: &crate::common::grpc::PeerEngineServer, coord: &str) {
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            coord,
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
}

/// A job submitted and claimed on `server`'s own engine catalog directly
/// (bypassing the wire `JobService`, matching `jobs_queue.rs`'s own fixture
/// style — the gang admission surface reads the row, not the submission
/// RPC), with `spec` controlling the row's own `world_size` — never the
/// caller's `Assign.world`. `model_ref: None` avoids needing a registered
/// model FK target (the column is nullable). Returns the `attempts` value
/// the claim landed at (always `1`, the first claim), for the caller to
/// build a matching `Assign` frame with.
pub(crate) async fn submit_and_claim(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
    coordinator_instance_id: &str,
    lease: std::time::Duration,
    spec: &str,
) -> i64 {
    use jammi_db::catalog::jobs_repo::SubmitJobParams;
    use jammi_db::catalog::status::JobExecution;

    let catalog = server.engine.catalog();
    catalog
        .submit_job(SubmitJobParams {
            job_id,
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec,
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next(coordinator_instance_id, &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("must claim the only queued job");
    i64::from(claimed.attempts)
}

/// A job submitted under an explicit tenant scope — mirroring
/// [`submit_and_claim`], but pinning the row's own `tenant_id` the way a
/// genuine tenant-bound submission would, via `with_tenant_scoped`
/// (`submit_job`'s own ambient `current_tenant()` picks it up) — then
/// claimed exactly like [`submit_and_claim`]. `claim_next` itself carries no
/// tenant predicate at all (the production worker claims globally, then
/// reads tenant off the claimed row), so the claim step runs unscoped.
/// Returns the claimed `attempts` value.
async fn submit_and_claim_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    job_id: &str,
    coordinator_instance_id: &str,
    lease: std::time::Duration,
    spec: &str,
) -> i64 {
    use jammi_db::catalog::jobs_repo::SubmitJobParams;
    use jammi_db::catalog::status::JobExecution;

    let job_id_owned = job_id.to_string();
    let spec_owned = spec.to_string();
    server
        .engine
        .with_tenant_scoped(tenant, move |scope| {
            let job_id = job_id_owned.clone();
            let spec = spec_owned.clone();
            async move {
                scope
                    .catalog()
                    .submit_job(SubmitJobParams {
                        job_id: &job_id,
                        kind: "fine_tune",
                        execution: JobExecution::Queued,
                        spec: &spec,
                        model_ref: None,
                        output_model_id: None,
                        model_source: None,
                        priority: 0,
                    })
                    .await
                    .unwrap()
            }
        })
        .await;
    let claimed = server
        .engine
        .catalog()
        .claim_next(coordinator_instance_id, &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("must claim the only queued job");
    i64::from(claimed.attempts)
}

/// A genuinely materialized `ready` `result_tables` row (Parquet + sidecar
/// manifest, the same shape a coordinator's own materialization produces)
/// under `tenant`, and what the row carries: the `(training_set_location,
/// training_set_ref, parquet_path)` triple — the table name, the digest read
/// back from the SAME sidecar the handler itself verifies against, and the
/// Parquet URL a fixture rewrites the sidecar beside.
pub(crate) struct ReadyTable {
    table: String,
    digest: String,
    parquet_path: String,
}

async fn materialize_ready_table_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    source_id: &str,
) -> ReadyTable {
    use datafusion::prelude::SessionContext;
    use jammi_db::model_task::ModelTask;
    use jammi_db::session::QueryContext;
    use jammi_db::store::manifest::{
        ComputeDevice, ComputePrecision, InputAnchor, Materialization, MaterializationEnv,
        ModelContentDigest, ModelIdentity, ProducingDescriptor,
    };
    use jammi_db::store::EmbeddingTableSpec;

    let descriptor = ProducingDescriptor::Embedding {
        model_id: "rt-base".into(),
        task: ModelTask::TextEmbedding,
        source_id: source_id.to_string(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: 4,
    };
    let env = MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "rt-base".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("gang-fixture-digest".into()),
            quantization: None,
        }],
    );
    let ctx = QueryContext::from(SessionContext::new());
    let source_id_owned = source_id.to_string();
    let engine_for_scope = Arc::clone(&server.engine);
    let record = server
        .engine
        .with_tenant_scoped(tenant, move |_scope| async move {
            let store = engine_for_scope.result_store();
            store
                .materialize_embedding_table(
                    &ctx,
                    EmbeddingTableSpec {
                        source_id: &source_id_owned,
                        model_id: "rt-base",
                        derived_from: None,
                        dimensions: 4,
                        key_column: None,
                        text_columns: None,
                    },
                    &[],
                    Materialization::new(
                        &descriptor,
                        &env,
                        vec![InputAnchor::mutable_version(&source_id_owned, 1)],
                    ),
                    None,
                )
                .await
                .unwrap()
        })
        .await;
    let store = server.engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("finish() must have written the sidecar");
    ReadyTable {
        table: record.table_name.clone(),
        digest: manifest.artifact.0.clone(),
        parquet_path: record.parquet_path.clone(),
    }
}

/// The in-process gang oracle's fixture rows, as the `pairs` CSV source the
/// world-2 sessions' rank body trains from.
fn write_pairs_csv(dir: &std::path::Path) -> String {
    let path = dir.join("gang_service_pairs.csv");
    let mut body = String::from("anchor,positive\n");
    for i in 0..8 {
        body.push_str(&format!("anchor text {i},positive text {i}\n"));
    }
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

fn tiny_bert_model() -> String {
    "local:".to_string()
        + jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .unwrap()
}

/// The `world_size == 2` job's REAL spec — the SAME shape `jammi-ai`'s
/// `TrainingCommon` persists (a column-source `fine_tune` over `pairs`,
/// `world_size: 2`), so the admitted session's rank body reconstructs a
/// runnable job from the row: it decodes, binds the training set below,
/// verifies its leaves, loads tiny_bert and waits at its first collective
/// for a coordinator that never sends a round.
fn world_two_spec_json() -> String {
    use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
    use jammi_ai::fine_tune::{EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
    use jammi_ai::model::ModelTask;
    use jammi_db::store::CachePolicy;

    serde_json::to_string(&TrainingSpec::FineTune {
        source: "pairs".into(),
        columns: vec!["anchor".into(), "positive".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: FineTuneConfig {
                epochs: 2,
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
            },
            world_size: 2,
            cache: CachePolicy::Bypass,
        },
    })
    .unwrap()
}

/// A genuinely materialized, ready `TrainingSet` table (Parquet + sidecar
/// with its leaf inventory — the same producer a coordinator's own
/// materialization runs, `jammi_ai::fine_tune::training_set::
/// materialize_projection_table`) over the `pairs` source under `tenant`,
/// and what the row carries — the SAME triple [`materialize_ready_table_for_tenant`]
/// returns for an embedding table.
async fn materialize_training_set_for_tenant(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
) -> ReadyTable {
    use jammi_ai::fine_tune::data::TrainingFormat;
    use jammi_ai::model::ModelTask;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    // Registered once per engine (an idempotent upsert of the same URL).
    let dir = tempfile::tempdir().expect("tempdir").keep();
    let url = write_pairs_csv(&dir);
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
    let columns = vec!["anchor".to_string(), "positive".to_string()];
    // `(anchor, positive)` is the pairs shape; the tag is read off the SAME
    // `TrainingFormat::format_tag` the classifier's own tag comes from.
    let format = TrainingFormat::Pairs.format_tag().to_string();
    let engine_for_scope = Arc::clone(&server.engine);
    let record = server
        .engine
        .with_tenant_scoped(tenant, move |_scope| async move {
            let table = jammi_ai::fine_tune::training_set::materialize_projection_table(
                &engine_for_scope,
                "pairs",
                &columns,
                ModelTask::TextEmbedding,
                &format,
            )
            .await
            .unwrap();
            // The whole catalog row, fetched through the catalog under the
            // SAME tenant scope this closure already runs in —
            // `TrainingSetTable` carries no whole-row accessor.
            engine_for_scope
                .catalog()
                .get_result_table(table.table_name())
                .await
                .unwrap()
                .expect("the producer promoted a catalog row")
        })
        .await;
    let store = server.engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("the materialization wrote the sidecar");
    ReadyTable {
        table: record.table_name.clone(),
        digest: manifest.artifact.0.clone(),
        parquet_path: record.parquet_path.clone(),
    }
}

/// A `result_tables` row created with NO tenant scope active (its
/// `tenant_id` lands NULL) whose `parquet_path` names nothing real — the
/// strict resolver must never hand it to a real tenant, so nothing past the
/// resolution is ever reached for it.
fn null_tenant_row(table: &str) -> jammi_db::catalog::result_repo::CreateResultTableParams<'_> {
    use jammi_db::catalog::result_repo::{CreateResultTableParams, ResultTableKind};
    use jammi_db::config::StoragePrecision;
    use jammi_db::model_task::ModelTask;
    CreateResultTableParams {
        table_name: table,
        source_id: "src",
        model_id: "rt-base",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/does-not-exist.parquet",
        dimensions: Some(4),
        key_column: None,
        text_columns: None,
        storage_precision: StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::lease::canonical_stamp_now(),
        writer_id: None,
        lease: None,
        job_attempt: None,
        replaces: None,
    }
}

/// Rewrites a ready table's sidecar WITHOUT its `leaves` inventory, so
/// `ResultStore::read_materialization_manifest` reads it as ABSENT
/// (`Ok(None)`), never as a manifest whose whole artifact is one leaf.
async fn strip_leaves_from_sidecar(
    server: &crate::common::grpc::PeerEngineServer,
    parquet_path: &str,
) {
    let store = server.engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    let bytes = handle.get_bytes(&sidecar).await.unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value
        .as_object_mut()
        .expect("a manifest is a JSON object")
        .remove("leaves")
        .expect("a freshly written sidecar carries a leaf inventory");
    handle
        .put_bytes(&sidecar, serde_json::to_vec(&value).unwrap().into())
        .await
        .unwrap();
    assert!(
        store
            .read_materialization_manifest(&url)
            .await
            .unwrap()
            .is_none(),
        "the pre-leaves sidecar must read as absent"
    );
}

/// Raw SQL against `server`'s engine catalog: the ONE conjunct under test is
/// manufactured directly (mirroring `run_rank_refuses_when_job_not_running`'s
/// technique), never through a path whose own guards could refuse the setup.
/// Every literal here is test-controlled (`unique_suffix`-derived names),
/// never external input.
async fn raw_sql(server: &crate::common::grpc::PeerEngineServer, sql: String) {
    use jammi_db::catalog::backend::TxOptions;
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let sql = sql.clone();
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await
        .unwrap();
}

/// The coordinator's own write-once CAS, landing cleanly on a freshly
/// claimed row.
async fn fill_pair(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
    coord: &str,
    attempt: i64,
    digest: &str,
    table: &str,
) {
    let outcome = server
        .engine
        .catalog()
        .fill_training_set_identity(job_id, coord, attempt as u32, digest, table)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            jammi_db::catalog::jobs_repo::TrainingSetFillOutcome::Filled
        ),
        "the fixture's own CAS must land cleanly on a freshly claimed row, got {outcome:?}"
    );
}

/// Every `jobs` column a terminal (or any) write on behalf of a rank would
/// move — snapshotted by primary key through raw SQL (the tenant-scoped
/// `get_job` cannot see a tenant-bound row from the unscoped fixture), so
/// a session end's "row untouched" claim is a before/after equality over
/// the WHOLE row, not one column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowFacts {
    status: String,
    claimed_by: Option<String>,
    attempts: i32,
    releases: i32,
    lease_expires_at: Option<String>,
    training_set_ref: Option<String>,
    training_set_location: Option<String>,
    error: Option<String>,
    result: Option<String>,
}

pub(crate) async fn row_facts(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
) -> RowFacts {
    use jammi_db::catalog::backend::{SqlValue, TxOptions};
    let job_id = job_id.to_string();
    server
        .engine
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
                        "SELECT status, claimed_by, attempts, releases, lease_expires_at, \
                             training_set_ref, training_set_location, error, result \
                         FROM jobs WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                        |row| {
                            Ok(RowFacts {
                                status: row.get("status")?,
                                claimed_by: row.try_get("claimed_by")?,
                                attempts: row.get("attempts")?,
                                releases: row.get("releases")?,
                                lease_expires_at: row.try_get("lease_expires_at")?,
                                training_set_ref: row.try_get("training_set_ref")?,
                                training_set_location: row.try_get("training_set_location")?,
                                error: row.try_get("error")?,
                                result: row.try_get("result")?,
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

/// An open bidi `RunRank` stream: the client's send side kept open (so a
/// later `Cancel` or second `Assign` can be sent) and the server's event
/// stream.
pub(crate) struct OpenRank {
    pub(crate) outbound: tokio::sync::mpsc::Sender<RankControl>,
    pub(crate) events: tonic::Streaming<RankEvent>,
}

impl std::fmt::Debug for OpenRank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenRank(admitted stream)")
    }
}

/// Opens a `RunRank` stream on `server`'s peer listener with `first` as its
/// opening frame. `Err` is the admission-time refusal status; `Ok` is an
/// admitted stream whose first event a caller reads with [`next_event`].
async fn open_rank(
    server: &crate::common::grpc::PeerEngineServer,
    first: RankControl,
) -> Result<OpenRank, tonic::Status> {
    open_rank_at(server.peer_addr, first).await
}

/// [`open_rank`] against any `GangService` listener — a standalone
/// `GangServer` a test mounted itself, or `server.peer_addr`.
pub(crate) async fn open_rank_at(
    addr: std::net::SocketAddr,
    first: RankControl,
) -> Result<OpenRank, tonic::Status> {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
    let channel = crate::common::grpc::channel(addr).await;
    let mut client = GangServiceClient::new(channel);
    let (outbound, rx) = tokio::sync::mpsc::channel::<RankControl>(4);
    outbound.send(first).await.unwrap();
    let response = client
        .run_rank(tokio_stream::wrappers::ReceiverStream::new(rx))
        .await?;
    Ok(OpenRank {
        outbound,
        events: response.into_inner(),
    })
}

/// The next event on an admitted stream within `within` — `Ok(None)` once
/// the server closed it, `Err(status)` for a status trailer (a protocol
/// violation on the admitted stream).
pub(crate) async fn next_event(
    events: &mut tonic::Streaming<RankEvent>,
    within: Duration,
) -> Result<Option<RankEvent>, tonic::Status> {
    tokio::time::timeout(within, events.message())
        .await
        .unwrap_or_else(|_| panic!("no stream event within {within:?}"))
}

fn is_admitted(event: &RankEvent) -> bool {
    matches!(event.event, Some(rank_event::Event::Admitted(_)))
}

fn aborted_reason(event: &RankEvent) -> Option<AbortReason> {
    match &event.event {
        Some(rank_event::Event::Aborted(aborted)) => AbortReason::try_from(aborted.reason).ok(),
        _ => None,
    }
}

pub(crate) async fn expect_admitted(rank: &mut OpenRank) {
    let event = next_event(&mut rank.events, Duration::from_secs(5))
        .await
        .expect("an admitted stream carries events, not a status")
        .expect("Admitted, not a closed stream");
    assert!(
        is_admitted(&event),
        "expected Admitted first, got {event:?}"
    );
}

fn is_round_event(event: &RankEvent) -> bool {
    matches!(
        event.event,
        Some(
            rank_event::Event::RoundContribution(_)
                | rank_event::Event::RoundChunk(_)
                | rank_event::Event::RoundAck(_)
                | rank_event::Event::RoundFault(_)
        )
    )
}

/// The stream's ONE terminal event — `Aborted{reason}` — followed by the
/// stream closing, within `within`. A body-bearing (`world_size > 1`)
/// session's stream carries the body's round traffic before its end (its
/// round-0 contribution to a coordinator that never answers); round frames
/// are read past, never mistaken for the end. A body-less session's stream
/// carries none.
async fn expect_aborted(rank: &mut OpenRank, reason: AbortReason, within: Duration) {
    let deadline = tokio::time::Instant::now() + within;
    let event = loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(!remaining.is_zero(), "no terminal event within {within:?}");
        let event = next_event(&mut rank.events, remaining)
            .await
            .expect("an Aborted event, not a status")
            .expect("Aborted, not a closed stream");
        if !is_round_event(&event) {
            break event;
        }
    };
    assert_eq!(
        aborted_reason(&event),
        Some(reason),
        "expected Aborted{{{reason:?}}}, got {event:?}"
    );
    let after = next_event(&mut rank.events, Duration::from_secs(5)).await;
    assert!(
        matches!(after, Ok(None)),
        "the stream must close after its terminal Aborted, got {after:?}"
    );
}

/// An admitted `world_size == 1` session over `server` (coordinator fresh,
/// row claimed under a long row lease so no re-verification tick refutes
/// it by itself), with the job row's facts snapshotted BEFORE admission.
async fn admitted_world_one(
    server: &crate::common::grpc::PeerEngineServer,
    job_id: &str,
    coord: &str,
) -> (OpenRank, i64, RowFacts) {
    fresh_coordinator(server, coord).await;
    let attempt =
        submit_and_claim(server, job_id, coord, Duration::from_secs(300), WORLD1_SPEC).await;
    let before = row_facts(server, job_id).await;
    let mut rank = open_rank(server, assign_frame_full(job_id, attempt, 0, 1, coord))
        .await
        .expect("every determinant holds: admitted");
    expect_admitted(&mut rank).await;
    (rank, attempt, before)
}

/// Everything an admitting `world_size == 2` session needs on the row side
/// — a genuinely materialized, ready, digest-verifying TRAINING SET under
/// the job's OWN tenant, a REAL `fine_tune` spec naming it (the rank body
/// the admitted session runs reconstructs its job from these two), a fresh
/// coordinator, the claimed row, the filled pair — with the stream NOT yet
/// opened: the caller opens it on the listener of its choice.
pub(crate) async fn world_two_ready(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    job_id: &str,
    coord: &str,
) -> (i64, RowFacts, ReadyTable) {
    let ready = materialize_training_set_for_tenant(server, tenant).await;
    fresh_coordinator(server, coord).await;
    let attempt = submit_and_claim_for_tenant(
        server,
        tenant,
        job_id,
        coord,
        Duration::from_secs(300),
        &world_two_spec_json(),
    )
    .await;
    fill_pair(server, job_id, coord, attempt, &ready.digest, &ready.table).await;
    let before = row_facts(server, job_id).await;
    (attempt, before, ready)
}

/// An admitted `world_size == 2` session over `server` for `tenant` — the
/// admitting control of the world>1 conjunct ([`world_two_ready`], then the
/// stream opened on `server.peer_addr` as rank 1: rank 0 is the
/// coordinator's own, in-process, and a member body is a rank `>= 1`).
async fn admitted_world_two(
    server: &crate::common::grpc::PeerEngineServer,
    tenant: TenantId,
    job_id: &str,
    coord: &str,
) -> (OpenRank, i64, RowFacts, ReadyTable) {
    let (attempt, before, ready) = world_two_ready(server, tenant, job_id, coord).await;
    let mut rank = open_rank(server, assign_frame_full(job_id, attempt, 1, 2, coord))
        .await
        .expect("a pair that resolves and verifies for the job's own tenant admits");
    expect_admitted(&mut rank).await;
    (rank, attempt, before, ready)
}

pub(crate) fn tenant(n: u8) -> TenantId {
    TenantId::from_str(&format!("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e{n:02x}")).unwrap()
}

// ---------------------------------------------------------------------------
// Wire-level edges (`world == 0`, `rank >= world`), decided before
// I-GANG runs.
// ---------------------------------------------------------------------------

/// Wire-level edge: `world == 0` is refused `INVALID_ARGUMENT`, before I-GANG
/// (which needs no row read at all here — no `jobs` row could ever satisfy this
/// wire-level edge).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_zero() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(0, 0));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("world == 0 must be refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    // `world == 0` also trips `rank >= world` for every unsigned `rank`
    // (`rank >= 0` is trivially true) — the message, not just the code,
    // distinguishes which edge actually refused, so a mutation that
    // deletes the `world == 0` check specifically (leaving `rank >= world`
    // to catch this exact input by coincidence) still fails this assertion.
    assert!(
        err.message().contains("greater than zero"),
        "expected the world == 0 message specifically (not the rank >= world \
         message, which this exact input also trips since rank >= 0 is \
         trivially true for every unsigned rank), got: {}",
        err.message()
    );
}

/// Observability: `jammi_gang_requests_total{rpc="RunRank"}` counts a
/// `RunRank` call reaching this member — the same shape `PeerService`'s own
/// `jammi_peer_requests_total{rpc}` counter proves (`peer_service.rs`), and
/// counted at the whole-server [`jammi_server::metrics_layer`] regardless of
/// how the call is ultimately decided (this one is refused at the wire-level
/// edge, `world == 0`, before I-GANG ever runs).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_increments_gang_requests_metric() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    assert_eq!(
        server
            .metrics
            .gang_requests
            .with_label_values(&["RunRank"])
            .get(),
        0,
        "no RunRank call has reached this member yet"
    );
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(0, 0));
    let _ = client.run_rank(outbound).await;
    assert_eq!(
        server
            .metrics
            .gang_requests
            .with_label_values(&["RunRank"])
            .get(),
        1,
        "the call above must be counted regardless of its own refusal"
    );
}

/// Wire-level edge: `rank >= world` is refused `INVALID_ARGUMENT` — the
/// boundary case (`rank == world`), not just a wildly out-of-range one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_rank_at_world_boundary() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame(2, 2));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("rank >= world must be refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// A `RunRank` stream that opens with `Cancel` rather than `Assign` is a
/// protocol violation refused `INVALID_ARGUMENT` — never silently accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_stream_opening_with_cancel() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
    use jammi_wire::proto::gang::{rank_control, Cancel, RankControl};

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(RankControl {
        control: Some(rank_control::Control::Cancel(Cancel {})),
    });
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a stream must open with Assign");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// An empty stream (closed before any frame at all) is refused
/// `INVALID_ARGUMENT` — never left to hang for the admission bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_stream_closed_before_assign() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
    use jammi_wire::proto::gang::RankControl;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::empty::<RankControl>();
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an empty stream must be refused, never accepted silently");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

// ---------------------------------------------------------------------------
// I-GANG (the full row predicate) + admit-and-hold (`world_size == 1`)
// ---------------------------------------------------------------------------

/// The `world_size == 1` arm: a call satisfying EVERY I-GANG
/// determinant — the row is `running`, claimed by the caller's own
/// `coordinator_instance_id`, at the matching `attempt`, under a live
/// lease, `world_size == 1` (so the training-set pair is not gated), the
/// coordinator's own `instances` row fresh — receives `Admitted`, is HELD
/// under re-verification (at least two ticks pass with no event: the live
/// row keeps re-verifying), and — a `world_size == 1` session having no
/// rank body to run — ends `Aborted{NoBody}` at the park bound (one lease
/// window), the stream closing after it. The job row is byte-identical
/// before and after (nothing terminal is written on behalf of a rank).
/// Mutation proof:
/// a handler that still ends `Unimplemented` fails at `open_rank`; a park
/// bound of one heartbeat ends before the two-tick floor; any `jobs` write
/// on the park path flips the row-facts equality.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_every_i_gang_determinant_satisfied_is_admitted_held_and_parks_no_body() {
    let server = start_no_worker_server().await;
    let started = tokio::time::Instant::now();
    let (mut rank, _attempt, before) = admitted_world_one(&server, "job-full", "coord-full").await;
    expect_aborted(
        &mut rank,
        AbortReason::NoBody,
        LEASE + Duration::from_secs(5),
    )
    .await;
    let held_for = started.elapsed();
    assert!(
        held_for >= HEARTBEAT * 2,
        "the session must be held under at least two re-verification ticks before the park \
         bound, was held {held_for:?}"
    );
    assert_eq!(
        row_facts(&server, "job-full").await,
        before,
        "the park end must leave the job row untouched"
    );
}

/// A job id no row exists for is refused `FAILED_PRECONDITION` — the
/// SAME status and message every other I-GANG determinant refuses with
/// (non-disclosure), never a distinguishing "not found" text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_job_not_found() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = crate::common::grpc::start_engine_server_with_peer_bind().await;
    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full("no-such-job", 0, 0, 1, "coord-1"));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an absent job must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// A `queued` (never claimed) row is refused `FAILED_PRECONDITION` —
/// the "not `running`" determinant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_job_not_running() {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied — this determinant is isolated from "coordinator
    // not fresh" (a separate conjunct), never a confound this test would accidentally also exercise.
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-1",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-completed",
        "coord-1",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    // Force status away from `running` WITHOUT touching `claimed_by` /
    // `attempts` / the lease, so this row fails ONLY the "not running"
    // conjunct — every other conjunct (`claimant_matches`, `attempt_matches`,
    // `lease_live`) still holds. A row this engine's own claim path can
    // reach this way (`finish_job`), manufactured directly via raw SQL so
    // the fixture does not depend on that path's own guards.
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET status = 'completed' WHERE job_id = 'job-completed'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full("job-completed", attempt, 0, 1, "coord-1"));
    let err = client.run_rank(outbound).await.expect_err(
        "a non-running job must be refused, even with a matching claimant/attempt/lease",
    );
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// A lease claimed for 1ms, then allowed to expire, is refused
/// `FAILED_PRECONDITION` — the "lease not live" determinant (never a
/// `NULL`-vs-expired distinction the caller can observe).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_lease_expired() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied — isolates "lease not live" from "coordinator not
    // fresh".
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-expired",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-expired",
        "coord-expired",
        std::time::Duration::from_millis(1),
        WORLD1_SPEC,
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-expired",
        attempt,
        0,
        1,
        "coord-expired",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("an expired lease must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// Every OTHER I-GANG determinant satisfied, but the coordinator's own
/// `instances` row was never upserted (absent) — refused
/// `FAILED_PRECONDITION`, the "coordinator not fresh" determinant I-GANG
/// names beside the row predicate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_coordinator_not_fresh() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Deliberately no `upsert_instance` call for "coord-stale" — every other
    // conjunct is satisfied by a genuine claim.
    let attempt = submit_and_claim(
        &server,
        "job-stale-coord",
        "coord-stale",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-stale-coord",
        attempt,
        0,
        1,
        "coord-stale",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a coordinator with no instances row must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// A caller naming the RIGHT coordinator but the WRONG attempt (a
/// zombie of a prior attempt this job already moved past) is refused
/// `FAILED_PRECONDITION` — the "wrong attempt" determinant, isolated from
/// "not claimed" by using the SAME coordinator the row was actually claimed
/// by.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_attempt_does_not_match() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-1",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-wrong-attempt",
        "coord-1",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-wrong-attempt",
        attempt + 1,
        0,
        1,
        "coord-1",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a caller naming a stale attempt must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// A job claimed by a DIFFERENT coordinator than the caller names is
/// refused `FAILED_PRECONDITION` — the "not claimed [by this caller]"
/// determinant. Caller-supplied identity is never trusted over the row's
/// own `claimed_by` (I-GANG).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_claimed_by_a_different_coordinator() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    // Freshness satisfied for the NAMED (impostor) coordinator — isolates
    // "not claimed by this caller" from "coordinator not fresh": `run_rank`
    // checks freshness against `assign.coordinator_instance_id`
    // ("coord-impostor"), never the row's own `claimed_by`.
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-impostor",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-wrong-claimant",
        "coord-real",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-wrong-claimant",
        attempt,
        0,
        1,
        "coord-impostor",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a caller naming a different coordinator than claimed_by must be refused");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

/// A genuine catalog fault reading `Catalog::fresh_instance` (the
/// `instances` table itself gone, mirroring `jammi-db`'s own
/// `get_job_for_rank_driver_fault_still_surfaces_as_err` DROP-TABLE
/// technique) maps through `admission_catalog_fault` to `Unavailable` — the
/// SAME admission-time classification every other catalog read on this path
/// uses, never `map_engine_error`'s generic mapping (which has no case for a
/// raw backend fault and would fall through to `Internal`). Every OTHER
/// I-GANG determinant is satisfied by construction, isolating this fault.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_fresh_instance_fault_is_unavailable() {
    use jammi_db::catalog::backend::TxOptions;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    server
        .engine
        .catalog()
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "coord-fresh-fault",
            Some("l"),
            Some("h"),
            None,
            None,
        ))
        .await
        .unwrap();
    let attempt = submit_and_claim(
        &server,
        "job-fresh-fault",
        "coord-fresh-fault",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    server
        .engine
        .catalog()
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move { tx.execute("DROP TABLE instances", &[]).await })
        })
        .await
        .unwrap();

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(assign_frame_full(
        "job-fresh-fault",
        attempt,
        0,
        1,
        "coord-fresh-fault",
    ));
    let err = client
        .run_rank(outbound)
        .await
        .expect_err("a dropped instances table must surface as a genuine catalog fault");
    assert_eq!(
        err.code(),
        tonic::Code::Unavailable,
        "fresh_instance erroring must map through admission_catalog_fault, never \
         map_engine_error's generic mapping"
    );
}

// ---------------------------------------------------------------------------
// The world-mismatch determinant, the non-disclosure oracle, and the
// `test-hooks` seam.
// ---------------------------------------------------------------------------

/// Builds a `tonic::Streaming<RankControl>` by hand from `frames`,
/// length-prefixed exactly the way the real gRPC wire frames a message
/// (`tonic-prost`'s own codec tests use the identical shape: a `0`
/// compression byte, a big-endian `u32` length, then the encoded bytes) —
/// never a real network connection. The ONLY caller needing this is
/// [`refusal_scenario`]'s `AdminScope` arm: ambient admin scope is
/// per-process task-local state that never crosses a network hop (a client
/// cannot set it on the server's own request-handling task, nor does it
/// survive a `tokio::spawn` boundary), so exercising that determinant
/// requires calling `GangServer::run_rank` directly, in the SAME task tree a
/// test wraps in `with_admin_scope` — which needs a real `tonic::Streaming`
/// value to satisfy `run_rank`'s own signature, not a bare `Stream`.
fn streaming_of(
    frames: Vec<jammi_wire::proto::gang::RankControl>,
) -> tonic::Streaming<jammi_wire::proto::gang::RankControl> {
    use bytes::{BufMut, Bytes, BytesMut};
    use jammi_wire::proto::gang::RankControl;
    use prost::Message;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tonic::codec::BufferSettings;
    use tonic_prost::ProstCodec;

    /// A one-shot `http_body::Body` handing back every framed byte in a
    /// single poll — sufficient since `tonic::Streaming`'s own decode loop
    /// reads incrementally from whatever bytes a `poll_frame` call yields,
    /// never requiring one HTTP frame per gRPC message.
    struct OnceBody(Option<Bytes>);
    impl http_body::Body for OnceBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;
        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(self.0.take().map(|b| Ok(http_body::Frame::data(b))))
        }
    }

    let mut buf = BytesMut::new();
    for frame in frames {
        let mut msg = BytesMut::new();
        frame.encode(&mut msg).expect("encode RankControl");
        buf.put_u8(0); // uncompressed
        buf.put_u32(msg.len() as u32);
        buf.put_slice(&msg);
    }

    let decoder = ProstCodec::<RankControl, RankControl>::raw_decoder(BufferSettings::default());
    tonic::Streaming::new_request(decoder, OnceBody(Some(buf.freeze())), None, None)
}

/// Drives ONE `RunRank` call end-to-end for a NAMED
/// [`jammi_server::grpc::gang::GangRefusalReason`] — every
/// scenario satisfies every OTHER I-GANG determinant, isolating the named
/// one, the same discipline the individual determinant tests earlier in
/// this file already follow, collected here ONCE so both the plain lane's
/// pairwise non-disclosure oracle and the `test-hooks` lane's
/// reason-distinguishing oracle drive the identical fixtures — never two
/// copies of this setup that could quietly drift apart. Returns the server
/// the call was driven against (a `test-hooks` caller reads
/// `PeerEngineServer::gang_last_refusal_reason` off this SAME instance
/// afterward) paired with the `Status` the RPC returned.
///
/// The world>1 rows: every one starts from the ADMITTING control's
/// own fixture — a genuinely materialized, ready, digest-verifying table
/// under the job's own tenant, the pair filled by the real CAS — and moves
/// exactly ONE determinant off it (the pair left unset; the row's tenant
/// text poisoned; the table owned by another tenant; the table forced back
/// to `building`; the sidecar stripped of its leaf inventory; the pair
/// filled with a digest the sidecar does not carry; the table's Parquet
/// URL moved onto a driver this build cannot construct), so the refusal is
/// attributable to that determinant alone — never to a fixture that would
/// have failed a LATER determinant anyway.
///
/// `GangRefusalReason::AdminScope` is the ONE exception to "drives it over
/// the real listener": no network client can ever make ambient admin scope
/// visible on the SERVER's own request-handling task (see [`streaming_of`]'s
/// own doc), so this arm builds a STANDALONE `GangServer` from the SAME
/// `server.engine` (never the instance mounted on `server.peer_addr`) and
/// calls `run_rank` on it DIRECTLY, inside `server.engine.with_admin_scope`
/// — swapping the standalone server's (`test-hooks` only) refusal handle
/// onto the returned `server` so `PeerEngineServer::gang_last_refusal_reason`
/// still observes the call this function actually made.
async fn refusal_scenario(
    reason: jammi_server::grpc::gang::GangRefusalReason,
) -> (crate::common::grpc::PeerEngineServer, tonic::Status) {
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    #[allow(unused_mut)]
    let mut server = start_no_worker_server().await;

    if reason == GangRefusalReason::AdminScope {
        use jammi_server::grpc::gang::GangServer;
        use jammi_server::grpc::proto::gang::gang_service_server::GangService;

        let coord = "nd-coord-admin-scope";
        fresh_coordinator(&server, coord).await;
        let attempt = submit_and_claim(
            &server,
            "nd-job-admin-scope",
            coord,
            std::time::Duration::from_secs(30),
            WORLD1_SPEC,
        )
        .await;
        // A standalone `GangServer`, never the one mounted on
        // `server.peer_addr` — see this function's own doc for why.
        let standalone = GangServer::new(Arc::clone(&server.engine), LEASE, HEARTBEAT);
        #[cfg(feature = "test-hooks")]
        {
            server.gang_refusal_handle = Some(standalone.refusal_reason_handle());
        }
        let request = tonic::Request::new(streaming_of(vec![assign_frame_full(
            "nd-job-admin-scope",
            attempt,
            0,
            1,
            coord,
        )]));
        let status = server
            .engine
            .with_admin_scope(|_admin| async {
                // `Response<Self::RunRankStream>` (the `Ok` type) is not
                // `Debug` (a boxed `dyn Stream` has no such impl), so
                // `.expect_err` (which requires it) cannot be used here —
                // a plain `match` avoids the bound entirely.
                match standalone.run_rank(request).await {
                    Ok(_) => panic!(
                        "admin scope must be refused, even though every OTHER I-GANG \
                         determinant this job satisfies would otherwise admit it"
                    ),
                    Err(status) => status,
                }
            })
            .await;
        return (server, status);
    }

    /// The world>1 fixture every world>1 row starts from: the admitting
    /// control's own shape, returned with its coordinates so the caller
    /// moves ONE determinant off it.
    async fn world_two_fixture(
        server: &crate::common::grpc::PeerEngineServer,
        tenant: TenantId,
        job_id: &str,
        coord: &str,
    ) -> (i64, ReadyTable) {
        let source_id = format!("nd_w2_src_{}", jammi_test_utils::unique_suffix());
        let ready = materialize_ready_table_for_tenant(server, tenant, &source_id).await;
        fresh_coordinator(server, coord).await;
        let attempt = submit_and_claim_for_tenant(
            server,
            tenant,
            job_id,
            coord,
            std::time::Duration::from_secs(30),
            WORLD2_SPEC,
        )
        .await;
        (attempt, ready)
    }

    let outbound_frame = match reason {
        GangRefusalReason::NotRunning => {
            fresh_coordinator(&server, "nd-coord-not-running").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-not-running",
                "nd-coord-not-running",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            raw_sql(
                &server,
                "UPDATE jobs SET status = 'completed' WHERE job_id = 'nd-job-not-running'".into(),
            )
            .await;
            assign_frame_full("nd-job-not-running", attempt, 0, 1, "nd-coord-not-running")
        }
        GangRefusalReason::WrongClaimant => {
            fresh_coordinator(&server, "nd-coord-impostor").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-wrong-claimant",
                "nd-coord-real",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full("nd-job-wrong-claimant", attempt, 0, 1, "nd-coord-impostor")
        }
        GangRefusalReason::WrongAttempt => {
            fresh_coordinator(&server, "nd-coord-wrong-attempt").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-wrong-attempt",
                "nd-coord-wrong-attempt",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-wrong-attempt",
                attempt + 1,
                0,
                1,
                "nd-coord-wrong-attempt",
            )
        }
        GangRefusalReason::LeaseDead => {
            fresh_coordinator(&server, "nd-coord-lease-dead").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-lease-dead",
                "nd-coord-lease-dead",
                std::time::Duration::from_millis(1),
                WORLD1_SPEC,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assign_frame_full("nd-job-lease-dead", attempt, 0, 1, "nd-coord-lease-dead")
        }
        GangRefusalReason::LeaseUndecodable => {
            // A row fact: `lease_expires_at` text that does
            // not parse as a timestamp — refused the same fixed way
            // `LeaseDead` is, under its own distinguishable variant, never
            // surfaced as `admission_catalog_fault`. Since migration
            // `039_canonical_stamps` the schema-edge domain refuses a
            // shape-invalid value at the WRITE (this harness's SQLite
            // catalog), so the planted text must be shape-valid,
            // calendar-invalid instead (a month of `13` — `catalog::lease`'s
            // own docs state why a leap second does not serve this role) to
            // still reach the Rust decode's `Undecodable` arm.
            fresh_coordinator(&server, "nd-coord-lease-undecodable").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-lease-undecodable",
                "nd-coord-lease-undecodable",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            raw_sql(
                &server,
                "UPDATE jobs SET lease_expires_at = '2026-13-01T00:00:00.000000Z' \
                 WHERE job_id = 'nd-job-lease-undecodable'"
                    .into(),
            )
            .await;
            assign_frame_full(
                "nd-job-lease-undecodable",
                attempt,
                0,
                1,
                "nd-coord-lease-undecodable",
            )
        }
        GangRefusalReason::NotFound => {
            assign_frame_full("nd-job-not-found", 0, 0, 1, "nd-coord-not-found")
        }
        GangRefusalReason::CoordinatorNotFresh => {
            let attempt = submit_and_claim(
                &server,
                "nd-job-coord-not-fresh",
                "nd-coord-stale",
                std::time::Duration::from_secs(30),
                WORLD1_SPEC,
            )
            .await;
            assign_frame_full("nd-job-coord-not-fresh", attempt, 0, 1, "nd-coord-stale")
        }
        GangRefusalReason::SpecUndecodable => {
            fresh_coordinator(&server, "nd-coord-undecodable").await;
            // A `world_size` key present but not a valid non-negative rank
            // count — `world_size_from_spec_json` (jammi-db) classifies this
            // `Undecodable`, never a fault of the read that found it.
            let attempt = submit_and_claim(
                &server,
                "nd-job-undecodable",
                "nd-coord-undecodable",
                std::time::Duration::from_secs(30),
                r#"{"common":{"world_size":"not-a-number"}}"#,
            )
            .await;
            assign_frame_full("nd-job-undecodable", attempt, 0, 1, "nd-coord-undecodable")
        }
        GangRefusalReason::WorldMismatch => {
            // `assign.world` (1) BELOW the row's own `world_size` (2,
            // `WORLD2_SPEC`): refused by the mismatch conjunct BEFORE the
            // pair conjunct is reached (so no pair fixture is needed) —
            // see `run_rank_refuses_when_assign_world_mismatches_row_world_size`
            // for the executed both-directions proof with controls.
            fresh_coordinator(&server, "nd-coord-world-mismatch").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-world-mismatch",
                "nd-coord-world-mismatch",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-world-mismatch",
                attempt,
                0,
                1,
                "nd-coord-world-mismatch",
            )
        }
        GangRefusalReason::TrainingSetPairMissing => {
            // `assign.world` (2) AGREES with the row's own `world_size`
            // (2); the pair was never filled — the coordinator recorded no
            // training set for this job.
            fresh_coordinator(&server, "nd-coord-pair-missing").await;
            let attempt = submit_and_claim(
                &server,
                "nd-job-pair-missing",
                "nd-coord-pair-missing",
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            assign_frame_full(
                "nd-job-pair-missing",
                attempt,
                0,
                2,
                "nd-coord-pair-missing",
            )
        }
        GangRefusalReason::TenantUndecodable => {
            let coord = "nd-coord-tenant-undecodable";
            let (attempt, ready) =
                world_two_fixture(&server, tenant(0x21), "nd-job-tenant-undecodable", coord).await;
            fill_pair(
                &server,
                "nd-job-tenant-undecodable",
                coord,
                attempt,
                &ready.digest,
                &ready.table,
            )
            .await;
            // The row's own `tenant_id` text poisoned (nothing in the
            // engine writes one): a row fact, refused before any
            // tenant-pinned read.
            raw_sql(
                &server,
                "UPDATE jobs SET tenant_id = 'not-a-tenant' WHERE job_id = 'nd-job-tenant-undecodable'"
                    .into(),
            )
            .await;
            assign_frame_full("nd-job-tenant-undecodable", attempt, 0, 2, coord)
        }
        GangRefusalReason::TrainingSetUnresolved => {
            // The table genuinely resolves and verifies — for ANOTHER
            // tenant than the one the calling job is bound to.
            let coord = "nd-coord-other-tenant";
            let tenant_owner = tenant(0x31);
            let tenant_caller = tenant(0x32);
            let source_id = format!("nd_other_src_{}", jammi_test_utils::unique_suffix());
            let ready = materialize_ready_table_for_tenant(&server, tenant_owner, &source_id).await;
            fresh_coordinator(&server, coord).await;
            let attempt = submit_and_claim_for_tenant(
                &server,
                tenant_caller,
                "nd-job-other-tenant",
                coord,
                std::time::Duration::from_secs(30),
                WORLD2_SPEC,
            )
            .await;
            fill_pair(
                &server,
                "nd-job-other-tenant",
                coord,
                attempt,
                &ready.digest,
                &ready.table,
            )
            .await;
            assign_frame_full("nd-job-other-tenant", attempt, 0, 2, coord)
        }
        GangRefusalReason::TrainingSetNotReady => {
            let coord = "nd-coord-not-ready";
            let (attempt, ready) =
                world_two_fixture(&server, tenant(0x41), "nd-job-not-ready", coord).await;
            fill_pair(
                &server,
                "nd-job-not-ready",
                coord,
                attempt,
                &ready.digest,
                &ready.table,
            )
            .await;
            // Forced back to `building` AFTER materialization finished:
            // this row verifies fine; only its `status` isolates the Ready
            // conjunct (a row whose URL never resolved would ALSO fail the
            // verify, hiding a "drop the Ready conjunct" mutation).
            raw_sql(
                &server,
                format!(
                    "UPDATE result_tables SET status = 'building' WHERE table_name = '{}'",
                    ready.table
                ),
            )
            .await;
            assign_frame_full("nd-job-not-ready", attempt, 0, 2, coord)
        }
        GangRefusalReason::TrainingSetSidecarAbsent => {
            let coord = "nd-coord-sidecar-absent";
            let (attempt, ready) =
                world_two_fixture(&server, tenant(0x51), "nd-job-sidecar-absent", coord).await;
            fill_pair(
                &server,
                "nd-job-sidecar-absent",
                coord,
                attempt,
                &ready.digest,
                &ready.table,
            )
            .await;
            // The pre-leaves sidecar: reads as absent, never as a verify.
            strip_leaves_from_sidecar(&server, &ready.parquet_path).await;
            assign_frame_full("nd-job-sidecar-absent", attempt, 0, 2, coord)
        }
        GangRefusalReason::TrainingSetDigestMismatch => {
            let coord = "nd-coord-digest-mismatch";
            let (attempt, ready) =
                world_two_fixture(&server, tenant(0x61), "nd-job-digest-mismatch", coord).await;
            // The pair names a digest the sidecar does not carry.
            fill_pair(
                &server,
                "nd-job-digest-mismatch",
                coord,
                attempt,
                "sha256:not-the-artifact-the-sidecar-attests",
                &ready.table,
            )
            .await;
            assign_frame_full("nd-job-digest-mismatch", attempt, 0, 2, coord)
        }
        GangRefusalReason::TrainingSetStoreFault => {
            let coord = "nd-coord-store-fault";
            let (attempt, ready) =
                world_two_fixture(&server, tenant(0x71), "nd-job-store-fault", coord).await;
            fill_pair(
                &server,
                "nd-job-store-fault",
                coord,
                attempt,
                &ready.digest,
                &ready.table,
            )
            .await;
            // The row's Parquet URL moved onto a scheme this build compiles
            // no driver for: `open_parquet` fails to build the driver —
            // `JammiError::Storage`, THIS host's store, no network.
            raw_sql(
                &server,
                format!(
                    "UPDATE result_tables SET parquet_path = 's3://gang-store-fault/x.parquet' \
                     WHERE table_name = '{}'",
                    ready.table
                ),
            )
            .await;
            assign_frame_full("nd-job-store-fault", attempt, 0, 2, coord)
        }
        GangRefusalReason::AdminScope => {
            unreachable!("handled above, before this match: no network client can trigger it")
        }
    };

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let outbound = tokio_stream::once(outbound_frame);
    let status = client
        .run_rank(outbound)
        .await
        .expect_err("every fixture this function builds must refuse");
    (server, status)
}

/// The byte-identity oracle an undecodable `world_size` must satisfy: a row
/// whose `spec` does not decode a `world_size` — either shape (`world_size`
/// present but non-numeric, or `spec` text that
/// is not valid JSON at all — the identical two shapes
/// `jammi-db`'s own `gang_rank_admission.rs` decodes) — refuses
/// BYTE-IDENTICALLY (code, message) to a job id no row exists for at all:
/// an undecodable spec is a ROW FACT, never `admission_catalog_fault`'s
/// `Unavailable` (reserved for the READ itself faulting). The `test-hooks`
/// seam names `SpecUndecodable` specifically for both shapes, distinguishing
/// them from `NotFound` same-process even though the wire cannot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_an_undecodable_world_size_byte_identically_to_not_found() {
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let (_absent, not_found_status) = refusal_scenario(GangRefusalReason::NotFound).await;

    for (label, coord, job_id, spec) in [
        (
            "non-numeric world_size",
            "nd-coord-poison-numeric",
            "nd-job-poison-numeric",
            r#"{"common":{"world_size":"two"}}"#,
        ),
        (
            "spec not valid JSON",
            "nd-coord-poison-json",
            "nd-job-poison-json",
            "not-even-json",
        ),
    ] {
        let server = start_no_worker_server().await;
        server
            .engine
            .catalog()
            .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
                coord,
                Some("l"),
                Some("h"),
                None,
                None,
            ))
            .await
            .unwrap();
        let attempt = submit_and_claim(
            &server,
            job_id,
            coord,
            std::time::Duration::from_secs(30),
            spec,
        )
        .await;
        let channel = crate::common::grpc::channel(server.peer_addr).await;
        let mut client = GangServiceClient::new(channel);
        let poisoned = client
            .run_rank(tokio_stream::once(assign_frame_full(
                job_id, attempt, 0, 1, coord,
            )))
            .await
            .expect_err("an undecodable world_size must be refused");
        assert_eq!(
            (poisoned.code(), poisoned.message()),
            (not_found_status.code(), not_found_status.message()),
            "{label}: an undecodable world_size must refuse byte-identically to an absent job"
        );
        #[cfg(feature = "test-hooks")]
        assert_eq!(
            server.gang_last_refusal_reason(),
            Some(GangRefusalReason::SpecUndecodable),
            "{label}: the test-hooks seam must name SpecUndecodable specifically"
        );
    }
}

/// The world-mismatch determinant, `assign.world != row.world_size`,
/// exercised in BOTH directions — the caller's `assign.world` below the
/// row's own value, and above it — each with a CONTROL call against the
/// SAME row proving the mismatch conjunct specifically is what refused,
/// never a coincidence with some other determinant.
///
/// **Direction (a)** (`assign.world` BELOW `row.world_size`): the row's own
/// `world_size` (`WORLD2_SPEC`'s `2`) with its pair unset is ALSO refused —
/// by the world>1 conjunct's pair determinant, decided strictly AFTER the
/// mismatch check — so this direction's control is distinguishable ONLY
/// via the `test-hooks` reason (`WorldMismatch` vs
/// `TrainingSetPairMissing`), never the wire status/message (both refuse
/// `FAILED_PRECONDITION` with the identical fixed message, by design —
/// non-disclosure).
///
/// **Direction (b)** (`assign.world` ABOVE `row.world_size`): the row's own
/// `world_size` is `1` (`WORLD1_SPEC`), so its control (`assign.world`
/// matching, `= 1`) is ADMITTED — a WIRE-VISIBLE distinction from the
/// mismatched call's `FAILED_PRECONDITION`, catching the mutation `if false
/// && assign.world != row.world_size` on the PLAIN lane alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_when_assign_world_mismatches_row_world_size() {
    #[cfg(feature = "test-hooks")]
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    // Direction (a): assign.world (1) BELOW row.world_size (2).
    let server_a = start_no_worker_server().await;
    fresh_coordinator(&server_a, "nd-coord-mismatch-below").await;
    let attempt_a = submit_and_claim(
        &server_a,
        "nd-job-mismatch-below",
        "nd-coord-mismatch-below",
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    let channel_a1 = crate::common::grpc::channel(server_a.peer_addr).await;
    let mut client_a1 = GangServiceClient::new(channel_a1);
    let mismatched = client_a1
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-below",
            attempt_a,
            0,
            1,
            "nd-coord-mismatch-below",
        )))
        .await
        .expect_err("assign.world (1) below row.world_size (2) must be refused");
    assert_eq!(mismatched.code(), tonic::Code::FailedPrecondition);
    assert_eq!(mismatched.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_a.gang_last_refusal_reason(),
        Some(GangRefusalReason::WorldMismatch),
        "must record WorldMismatch specifically, not the pair determinant"
    );
    // Control: the SAME row, assign.world (2) matching row.world_size (2):
    // the mismatch conjunct passes and the NEXT determinant — the pair,
    // unset here — is what refuses.
    let channel_a2 = crate::common::grpc::channel(server_a.peer_addr).await;
    let mut client_a2 = GangServiceClient::new(channel_a2);
    let matching = client_a2
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-below",
            attempt_a,
            0,
            2,
            "nd-coord-mismatch-below",
        )))
        .await
        .expect_err("the row's pair is unset, so the pair conjunct refuses");
    assert_eq!(matching.code(), tonic::Code::FailedPrecondition);
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_a.gang_last_refusal_reason(),
        Some(GangRefusalReason::TrainingSetPairMissing),
        "the SAME row at its own world_size must reach the pair conjunct, distinguishing it \
         from the mismatch above"
    );

    // Direction (b): assign.world (2) ABOVE row.world_size (1).
    let server_b = start_no_worker_server().await;
    fresh_coordinator(&server_b, "nd-coord-mismatch-above").await;
    let attempt_b = submit_and_claim(
        &server_b,
        "nd-job-mismatch-above",
        "nd-coord-mismatch-above",
        std::time::Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    let channel_b1 = crate::common::grpc::channel(server_b.peer_addr).await;
    let mut client_b1 = GangServiceClient::new(channel_b1);
    let mismatched_b = client_b1
        .run_rank(tokio_stream::once(assign_frame_full(
            "nd-job-mismatch-above",
            attempt_b,
            0,
            2,
            "nd-coord-mismatch-above",
        )))
        .await
        .expect_err("assign.world (2) above row.world_size (1) must be refused");
    assert_eq!(mismatched_b.code(), tonic::Code::FailedPrecondition);
    assert_eq!(mismatched_b.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server_b.gang_last_refusal_reason(),
        Some(GangRefusalReason::WorldMismatch)
    );
    // Control: the SAME row, assign.world (1) matching row.world_size (1) —
    // ADMITTED, a WIRE-VISIBLE distinction from the mismatched call above,
    // proving the mismatch conjunct (never some other reason) is what
    // refused it.
    let mut matching_b = open_rank(
        &server_b,
        assign_frame_full(
            "nd-job-mismatch-above",
            attempt_b,
            0,
            1,
            "nd-coord-mismatch-above",
        ),
    )
    .await
    .expect("the SAME row at its own world_size (1) must admit");
    expect_admitted(&mut matching_b).await;
}

/// The full set of I-GANG determinants [`refusal_scenario`] can drive,
/// shared by the plain lane's non-disclosure oracle and the `test-hooks`
/// lane's reason-distinguishing oracle below — one definition, so adding a
/// determinant to one automatically covers it in the other.
///
/// Derived from an EXHAUSTIVE match over a witness of each variant, never a
/// hardcoded `[..; N]` array: `assert_every_variant_is_a_witness`'s own
/// match has NO wildcard arm, so a new `GangRefusalReason` variant fails
/// THIS FILE to compile — naming the missing arm — until a matching arm is
/// added there; the exhaustive match forces every variant into a scenario
/// arm. `WITNESSES` below is a separate list that match does NOT force to
/// grow — a fixed arm-list gap the compiler catches is not the same as a
/// witness-list gap it does not; adding the new arm without also adding the
/// variant to `WITNESSES` still compiles.
fn every_gang_refusal_reason() -> Vec<jammi_server::grpc::gang::GangRefusalReason> {
    use jammi_server::grpc::gang::GangRefusalReason as R;

    const WITNESSES: &[R] = &[
        R::AdminScope,
        R::NotFound,
        R::NotRunning,
        R::WrongClaimant,
        R::WrongAttempt,
        R::LeaseDead,
        R::LeaseUndecodable,
        R::SpecUndecodable,
        R::WorldMismatch,
        R::TrainingSetPairMissing,
        R::TenantUndecodable,
        R::TrainingSetUnresolved,
        R::TrainingSetNotReady,
        R::TrainingSetSidecarAbsent,
        R::TrainingSetDigestMismatch,
        R::TrainingSetStoreFault,
        R::CoordinatorNotFresh,
    ];

    fn assert_every_variant_is_a_witness(variant: R) {
        match variant {
            R::AdminScope
            | R::NotFound
            | R::NotRunning
            | R::WrongClaimant
            | R::WrongAttempt
            | R::LeaseDead
            | R::LeaseUndecodable
            | R::SpecUndecodable
            | R::WorldMismatch
            | R::TrainingSetPairMissing
            | R::TenantUndecodable
            | R::TrainingSetUnresolved
            | R::TrainingSetNotReady
            | R::TrainingSetSidecarAbsent
            | R::TrainingSetDigestMismatch
            | R::TrainingSetStoreFault
            | R::CoordinatorNotFresh => {}
        }
    }
    for w in WITNESSES {
        assert_every_variant_is_a_witness(*w);
    }
    WITNESSES.to_vec()
}

/// The ONE non-disclosure oracle. Every I-GANG determinant
/// (ambient admin scope / not found / not running / wrong claimant / wrong
/// attempt / lease dead / lease undecodable / undecodable world_size /
/// world mismatch / the world>1 conjunct's seven — pair missing, tenant
/// undecodable, unresolved under the job's tenant, not ready, sidecar
/// absent, digest mismatch, this host's store faulting — / coordinator not
/// fresh — seventeen total) refuses with
/// the PAIRWISE-IDENTICAL `(code, message)` — compared pairwise so a single
/// differing pair fails naming exactly that pair, never merely "some
/// determinant's message differs somewhere". Mutation proof: make any ONE
/// determinant's message leak (e.g. append the reason) and this fails,
/// naming the leaking pair.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refusal_is_non_disclosing_across_every_determinant() {
    let reasons = every_gang_refusal_reason();
    let mut statuses = Vec::with_capacity(reasons.len());
    for reason in reasons {
        let (_server, status) = refusal_scenario(reason).await;
        statuses.push((reason, status));
    }
    for i in 0..statuses.len() {
        for j in (i + 1)..statuses.len() {
            let (reason_a, status_a) = &statuses[i];
            let (reason_b, status_b) = &statuses[j];
            assert_eq!(
                (status_a.code(), status_a.message()),
                (status_b.code(), status_b.message()),
                "{reason_a:?} and {reason_b:?} must refuse with the pairwise-identical \
                 (code, message) — non-disclosure requires this for every pair, not just \
                 some of them"
            );
        }
    }
}

/// `test-hooks` only: drives the SAME seventeen scenarios
/// [`run_rank_refusal_is_non_disclosing_across_every_determinant`] does, but
/// asserts `PeerEngineServer::gang_last_refusal_reason` names the EXACT
/// determinant each one refused for — the seam that lets this lane
/// distinguish what the plain lane's own non-disclosure oracle just proved
/// is (correctly) indistinguishable on the wire. This is what makes the
/// `test-hooks` lane's `cargo test -p jammi-server --test it -- gang`
/// execute ONE MORE gang-prefixed test-fn than the plain lane (this
/// function itself is compiled only under `test-hooks`; the plain lane's
/// case above still runs the same ten RPC calls, just without this
/// additional reason assertion). One of the seventeen (`AdminScope`) is driven
/// in-process by `refusal_scenario` itself (see its own doc) rather than
/// over the real listener — this test's own assertion still holds, since it
/// reads `PeerEngineServer::gang_last_refusal_reason`, which `refusal_scenario`
/// re-points at the standalone `GangServer` it actually called for that one
/// scenario.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_last_refusal_reason_distinguishes_every_determinant() {
    for reason in every_gang_refusal_reason() {
        let (server, status) = refusal_scenario(reason).await;
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
        assert_eq!(
            server.gang_last_refusal_reason(),
            Some(reason),
            "the served GangServer must record {reason:?} for this scenario, distinguishing \
             it from every other determinant same-process, without that distinction ever \
             reaching the wire"
        );
    }
}

// ---------------------------------------------------------------------------
// The world>1 conjunct: the named rows, and the admitting control.
// ---------------------------------------------------------------------------

/// The cross-tenant-denial case the `GANG_LISTENER_ALLOWLIST`
/// derivation claim stands on: `training_set_location` names a
/// `result_tables` row that genuinely resolves and verifies — but for
/// ANOTHER tenant than the one the calling job's row is bound to. Refused
/// `FAILED_PRECONDITION`, the SAME status and fixed message every other
/// determinant refuses with — the response discloses neither the other
/// tenant's id nor its table name. Mutation proof: dropping the tenant
/// bind from the strict predicate (or resolving through the relaxed
/// `get_result_table`) admits this call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_training_set_another_tenant_owns() {
    use jammi_server::grpc::gang::GangRefusalReason;

    #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
    let (server, status) = refusal_scenario(GangRefusalReason::TrainingSetUnresolved).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
    assert!(
        !status.message().contains("nd_other_src") && !status.message().contains("01906c83"),
        "the refusal must disclose neither the other tenant's table name nor its id, got: {}",
        status.message()
    );
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server.gang_last_refusal_reason(),
        Some(GangRefusalReason::TrainingSetUnresolved)
    );
}

/// The NULL-tenant variant: `training_set_location` names a
/// row created with NO tenant scope active (its `tenant_id` is NULL) while
/// the calling job IS tenant-bound. The strict resolver never matches a
/// NULL-tenant row for a real tenant (`jammi-db`'s own tests prove it
/// against the verb on both backends) — refused through the RPC too,
/// never resolving the orphaned row. Mutation proof: the relaxed predicate
/// (`OR tenant_id IS NULL`) resolves this row and, the URL naming nothing,
/// still refuses — but under a DIFFERENT `test-hooks` reason
/// (`TrainingSetStoreFault`/`SidecarAbsent`), which this row pins.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job() {
    #[cfg(feature = "test-hooks")]
    use jammi_server::grpc::gang::GangRefusalReason;
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let tenant_caller = tenant(0x81);
    let table = format!("gang_null_tenant_rpc_{}", jammi_test_utils::unique_suffix());
    // Created with NO tenant scope active — the row's `tenant_id` lands
    // NULL.
    server
        .engine
        .catalog()
        .create_result_table(null_tenant_row(&table))
        .await
        .unwrap();
    raw_sql(
        &server,
        format!("UPDATE result_tables SET status = 'ready' WHERE table_name = '{table}'"),
    )
    .await;
    let coord = "coord-null-tenant";
    fresh_coordinator(&server, coord).await;
    let attempt = submit_and_claim_for_tenant(
        &server,
        tenant_caller,
        "job-null-tenant",
        coord,
        std::time::Duration::from_secs(30),
        WORLD2_SPEC,
    )
    .await;
    fill_pair(
        &server,
        "job-null-tenant",
        coord,
        attempt,
        "sha256:any",
        &table,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let err = client
        .run_rank(tokio_stream::once(assign_frame_full(
            "job-null-tenant",
            attempt,
            0,
            2,
            coord,
        )))
        .await
        .expect_err("a NULL-tenant training set must never resolve for a tenant-bound job");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert_eq!(err.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server.gang_last_refusal_reason(),
        Some(GangRefusalReason::TrainingSetUnresolved),
        "the strict resolver must not resolve the NULL-tenant row (a later determinant \
         refusing instead would mean it did)"
    );
}

/// A ready table whose sidecar has no leaf inventory (no `leaves`)
/// reads as ABSENT (the inventory-aware read) and refuses — never a
/// verify that treats the whole artifact as one leaf. Mutation proof:
/// a reader that accepts a leaf-less sidecar admits this call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_world_gt_one_when_the_sidecar_predates_the_leaf_inventory() {
    use jammi_server::grpc::gang::GangRefusalReason;

    #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
    let (server, status) = refusal_scenario(GangRefusalReason::TrainingSetSidecarAbsent).await;
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "gang admission refused");
    #[cfg(feature = "test-hooks")]
    assert_eq!(
        server.gang_last_refusal_reason(),
        Some(GangRefusalReason::TrainingSetSidecarAbsent)
    );
}

/// The resolution site's own admin-scope guard (the resolution site
/// refuses admin scope explicitly before ever calling the strict
/// resolver, regardless of which table exists): with a REAL, ready,
/// digest-verifying table for tenant B, `resolve_training_set_identity`
/// for tenant B (i) verifies outside any admin scope — the control proving
/// the fixture is genuinely admissible — and (ii) refuses
/// `AdminScopeRefused` inside `with_admin_scope`, before the resolver ever
/// ran. Driven directly (the handler's own top-of-call guard refuses
/// ambient admin scope before any row is read, so no `RunRank` call can
/// reach this second line — it is the resolution site's own). Mutation
/// proof: deleting the guard returns `Verified` under admin scope.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolution_site_refuses_under_admin_scope_before_the_strict_resolver_runs() {
    use jammi_server::grpc::gang::{resolve_training_set_identity, TrainingSetOutcome};

    let server = start_no_worker_server().await;
    let tenant_b = tenant(0x91);
    let source_id = format!("gang_admin_site_src_{}", jammi_test_utils::unique_suffix());
    let ready = materialize_ready_table_for_tenant(&server, tenant_b, &source_id).await;
    let store = server.engine.result_store();

    let outside =
        resolve_training_set_identity(store.as_ref(), Some(tenant_b), &ready.digest, &ready.table)
            .await
            .unwrap();
    assert_eq!(
        outside,
        TrainingSetOutcome::Verified,
        "the control: a genuinely admissible table verifies outside admin scope"
    );

    let store_for_scope = Arc::clone(&store);
    let inside = server
        .engine
        .with_admin_scope(|_admin| {
            let store = Arc::clone(&store_for_scope);
            let digest = ready.digest.clone();
            let table = ready.table.clone();
            async move {
                resolve_training_set_identity(store.as_ref(), Some(tenant_b), &digest, &table)
                    .await
                    .unwrap()
            }
        })
        .await;
    assert_eq!(
        inside,
        TrainingSetOutcome::AdminScopeRefused,
        "the resolution site refuses ambient admin scope before the strict resolver runs"
    );
}

/// The world>1 parity row: the SAME real training set that
/// admits over the wire is first read back through `get_job_for_rank` on
/// the fixture's SQLite catalog; the Postgres arm of this producer→consumer
/// parity is `gang_training_spec_parity.rs`'s (`test_case`-parameterized
/// over both backends). Here: the ADMITTING CONTROL — a pair that
/// genuinely resolves and verifies for the job's own tenant reaches
/// `Admitted` (proving the conjunct was DECIDED, not blanket-refused), is
/// held under re-verification ticks that re-resolve and re-verify the same
/// identity while its REAL rank body runs — it reconstructs the job from
/// the row, binds and verifies the training set, loads the model, and
/// sends its round-0 contribution (the ONLY frames the stream carries:
/// round traffic, to a coordinator that never answers) — and NEVER parks:
/// past the park bound (`LEASE`) no TERMINAL event has been emitted, and
/// the session ends only when the coordinator's `Cancel` ends it
/// `Aborted{Cancelled}` — the stream closing right after, with no frame of
/// the body's following it — the row untouched, and the slot free.
/// Mutation proof: a hold loop whose park arm fires for a body-bearing
/// session emits `Aborted{NoBody}` inside the bound; a handler that spawns
/// no body parks the same way and never sends a round frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_rank_world_two_own_tenant_training_set_is_admitted_runs_its_body_and_never_parks() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before, _ready) =
        admitted_world_two(&server, tenant(0xa1), "job-w2-own-tenant", "coord-w2-own").await;
    // Past the park bound and at least two ticks: round frames only, no
    // terminal event, and the body did reach its first collective.
    let deadline = tokio::time::Instant::now() + LEASE + Duration::from_secs(1);
    let mut round_frames = 0usize;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, rank.events.message()).await {
            Err(_elapsed) => break,
            Ok(Ok(Some(event))) => {
                assert!(
                    is_round_event(&event),
                    "a body-bearing session emits no terminal event at the park bound: {event:?}"
                );
                round_frames += 1;
            }
            Ok(other) => panic!("the stream must stay open past the park bound: {other:?}"),
        }
    }
    assert!(
        round_frames > 0,
        "the rank body reached its first collective and sent its contribution"
    );
    rank.outbound.send(cancel_frame()).await.unwrap();
    expect_aborted(&mut rank, AbortReason::Cancelled, Duration::from_secs(5)).await;
    assert_eq!(row_facts(&server, "job-w2-own-tenant").await, before);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while server.engine.host_admission().holder() != jammi_ai::fine_tune::worker::Holder::Free {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the slot is freed with the session, holder: {:?}",
            server.engine.host_admission().holder()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// Holder contention: the lattice over the wire.
// ---------------------------------------------------------------------------

/// A `JobRun` holder — a loop-claimed job running on this host —
/// refuses `Unavailable` AT ONCE (well inside one heartbeat), with a fixed
/// message; the I-GANG seam records no determinant for it (every
/// determinant held — the slot alone refused). Manufactured through
/// `HostAdmission::hold_for_test`, so no job need run. Mutation proof: a
/// CAS that waits on `JobRun` the way it waits on a probe fails the
/// elapsed bound; a CAS run BEFORE the determinants would let this call
/// refuse `Unavailable` for a job that does not exist — the sibling
/// `not_found` row below pins the order.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_refuses_unavailable_at_once_while_a_loop_job_runs() {
    use jammi_ai::fine_tune::worker::Holder;

    let server = start_no_worker_server().await;
    fresh_coordinator(&server, "coord-jobrun").await;
    let attempt = submit_and_claim(
        &server,
        "job-jobrun",
        "coord-jobrun",
        Duration::from_secs(30),
        WORLD1_SPEC,
    )
    .await;
    let _busy = server.engine.host_admission().hold_for_test(Holder::JobRun);
    let started = tokio::time::Instant::now();
    let err = open_rank(
        &server,
        assign_frame_full("job-jobrun", attempt, 0, 1, "coord-jobrun"),
    )
    .await
    .expect_err("a running loop job refuses the rank");
    assert_eq!(err.code(), tonic::Code::Unavailable);
    assert_eq!(
        err.message(),
        "gang admission: this host's job slot is busy"
    );
    assert!(
        started.elapsed() < HEARTBEAT,
        "JobRun refuses at once, never after a wait"
    );
    assert_eq!(
        server.gang_last_refusal_reason(),
        None,
        "slot contention is not an I-GANG determinant; nothing is recorded"
    );

    // The order: the decision BEFORE the CAS. A job that does not exist is
    // refused `FailedPrecondition` (the determinant), never `Unavailable`
    // (the slot), even while the slot is busy.
    let err = open_rank(
        &server,
        assign_frame_full("no-such-job", 0, 0, 1, "coord-jobrun"),
    )
    .await
    .expect_err("an absent job refuses before the slot is ever consulted");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert_eq!(
        server.gang_last_refusal_reason(),
        Some(jammi_server::grpc::gang::GangRefusalReason::NotFound)
    );
}

/// A `ClaimProbe` holder is waited on for at most one heartbeat: freed
/// within it, the rank is ADMITTED (and admission follows the release,
/// not the bound); still probing at the bound, refused `Unavailable` — and
/// only then. Mutation proof: refusing a probe at once fails the first
/// half; a wait longer than one heartbeat fails the second half's upper
/// elapsed assertion.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable() {
    use jammi_ai::fine_tune::worker::Holder;

    let server = start_no_worker_server().await;
    fresh_coordinator(&server, "coord-probe").await;
    let attempt = submit_and_claim(
        &server,
        "job-probe",
        "coord-probe",
        Duration::from_secs(300),
        WORLD1_SPEC,
    )
    .await;

    // Freed within the bound: admitted.
    let probe = server
        .engine
        .host_admission()
        .hold_for_test(Holder::ClaimProbe);
    let opening = {
        let addr = server.peer_addr;
        tokio::spawn(async move {
            use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
            let channel = crate::common::grpc::channel(addr).await;
            let mut client = GangServiceClient::new(channel);
            let started = tokio::time::Instant::now();
            let response = client
                .run_rank(tokio_stream::once(assign_frame_full(
                    "job-probe",
                    attempt,
                    0,
                    1,
                    "coord-probe",
                )))
                .await;
            (response.map(|r| r.into_inner()), started.elapsed())
        })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!opening.is_finished(), "still waiting on the probe");
    drop(probe);
    let (response, elapsed) = opening.await.unwrap();
    let mut events = response.expect("freed within the bound: admitted");
    assert!(
        elapsed < HEARTBEAT,
        "admission follows the probe's release, not the bound: {elapsed:?}"
    );
    let first = next_event(&mut events, Duration::from_secs(5))
        .await
        .unwrap()
        .unwrap();
    assert!(is_admitted(&first), "{first:?}");
    // End this session so the slot is free for the second half.
    server.engine.host_admission().begin_drain();
    let end = next_event(&mut events, Duration::from_secs(5))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(aborted_reason(&end), Some(AbortReason::Drain));
    drop(events);

    // A fresh server (the drained one refuses everything now): still
    // probing at the bound, refused — and only at the bound.
    let server = start_no_worker_server().await;
    fresh_coordinator(&server, "coord-probe-2").await;
    let attempt = submit_and_claim(
        &server,
        "job-probe-2",
        "coord-probe-2",
        Duration::from_secs(300),
        WORLD1_SPEC,
    )
    .await;
    let _probe = server
        .engine
        .host_admission()
        .hold_for_test(Holder::ClaimProbe);
    let started = tokio::time::Instant::now();
    let err = open_rank(
        &server,
        assign_frame_full("job-probe-2", attempt, 0, 1, "coord-probe-2"),
    )
    .await
    .expect_err("a probe that never resolves refuses at the bound");
    let elapsed = started.elapsed();
    assert_eq!(err.code(), tonic::Code::Unavailable);
    assert!(
        elapsed >= HEARTBEAT,
        "refused before the bound: {elapsed:?}"
    );
    assert!(
        elapsed < HEARTBEAT * 3,
        "refused long after the bound: {elapsed:?}"
    );
}

/// Another rank held on this host refuses a rank for a DIFFERENT job
/// `Unavailable` at once, and a DUPLICATE assignment of the held session
/// (same job, same attempt) the same way; the SAME job at a GREATER
/// attempt (the row's `attempts` moved on) takes the slot — admitted —
/// while the elder session, superseded, ends `Aborted{Refuted}` at its
/// next re-verification tick (the row no longer names its attempt). Every
/// end leaves the row untouched. Mutation proof: a CAS that refuses the
/// greater attempt fails the successor's `open_rank`; a `RankHold` drop
/// that frees the slot regardless of identity lets the elder's end free
/// the successor's slot — pinned by the successor's own park end still
/// arriving, and by a third rank refused while the successor is held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_held_rank_refuses_other_ranks_and_the_same_job_at_a_greater_attempt_takes_the_slot() {
    let server = start_no_worker_server().await;
    let (mut elder, attempt, before) = admitted_world_one(&server, "job-held", "coord-held").await;

    // Another job: refused at once.
    fresh_coordinator(&server, "coord-other").await;
    let other_attempt = submit_and_claim(
        &server,
        "job-other",
        "coord-other",
        Duration::from_secs(300),
        WORLD1_SPEC,
    )
    .await;
    let started = tokio::time::Instant::now();
    let err = open_rank(
        &server,
        assign_frame_full("job-other", other_attempt, 0, 1, "coord-other"),
    )
    .await
    .expect_err("another rank is held here");
    assert_eq!(err.code(), tonic::Code::Unavailable);
    assert!(started.elapsed() < HEARTBEAT);

    // A duplicate of the held session: refused at once.
    let err = open_rank(
        &server,
        assign_frame_full("job-held", attempt, 0, 1, "coord-held"),
    )
    .await
    .expect_err("the same job at the same attempt is a duplicate of a held session");
    assert_eq!(err.code(), tonic::Code::Unavailable);

    // The job's attempt moves on (a reclaim + re-claim, manufactured
    // directly): the greater attempt takes the slot.
    raw_sql(
        &server,
        "UPDATE jobs SET attempts = attempts + 1 WHERE job_id = 'job-held'".into(),
    )
    .await;
    let mut successor = open_rank(
        &server,
        assign_frame_full("job-held", attempt + 1, 0, 1, "coord-held"),
    )
    .await
    .expect("the same job at a greater attempt takes the slot");
    expect_admitted(&mut successor).await;
    // The elder is refuted by the row at its next tick, not cut by the
    // successor's CAS.
    expect_aborted(&mut elder, AbortReason::Refuted, HEARTBEAT * 3).await;
    // The successor still holds the slot after the elder's end: a third
    // rank is refused, and the successor's own park end still arrives.
    let err = open_rank(
        &server,
        assign_frame_full("job-other", other_attempt, 0, 1, "coord-other"),
    )
    .await
    .expect_err("the successor holds the slot; the elder's end did not free it");
    assert_eq!(err.code(), tonic::Code::Unavailable);
    expect_aborted(
        &mut successor,
        AbortReason::NoBody,
        LEASE + Duration::from_secs(5),
    )
    .await;

    let after = row_facts(&server, "job-held").await;
    assert_eq!(
        after,
        RowFacts {
            attempts: before.attempts + 1,
            ..before.clone()
        },
        "only the fixture's own attempts bump; no session end wrote the row"
    );
}

// ---------------------------------------------------------------------------
// The hold loop's four arms: inbound (Cancel / second Assign), drain, the
// re-verification tick (three ends), the park bound (above).
// ---------------------------------------------------------------------------

/// `Cancel` on an admitted stream ends the session cooperatively —
/// `Aborted{Cancelled}`, the stream closed, the row untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_cancel_on_an_admitted_stream_ends_cancelled() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-cancel", "coord-cancel").await;
    rank.outbound.send(cancel_frame()).await.unwrap();
    expect_aborted(&mut rank, AbortReason::Cancelled, Duration::from_secs(5)).await;
    assert_eq!(row_facts(&server, "job-cancel").await, before);
}

/// A second `Assign` on an already-admitted stream is a protocol
/// violation — the stream ends with `InvalidArgument`, never a second
/// admission and never an `Aborted` reason; the row untouched. Mutation
/// proof: an inbound arm that ignores a second `Assign` parks to `NoBody`
/// instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_second_assign_on_an_admitted_stream_is_invalid_argument() {
    let server = start_no_worker_server().await;
    let (mut rank, attempt, before) =
        admitted_world_one(&server, "job-second-assign", "coord-second-assign").await;
    rank.outbound
        .send(assign_frame_full(
            "job-second-assign",
            attempt,
            0,
            1,
            "coord-second-assign",
        ))
        .await
        .unwrap();
    let err = next_event(&mut rank.events, Duration::from_secs(5))
        .await
        .expect_err("a second Assign is a status, never an event");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert_eq!(row_facts(&server, "job-second-assign").await, before);
}

/// An EMPTY `RankControl` frame (`control: None`) on an admitted stream
/// — the one value of the oneof that is neither the session's (`Assign`,
/// `Cancel`) nor the round protocol's — is a protocol violation: the stream
/// ends with `InvalidArgument`, the row untouched. This is the refusal arm
/// of `HeldSession::dispatch_round_frame`, beside its delivery arm. Mutation
/// proof: a dispatch that hands EVERY frame to the round inbox (the
/// `is_round_frame` guard skipped) keeps the session held — no trailer
/// arrives within this test's bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_empty_frame_on_an_admitted_stream_is_invalid_argument() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-empty-frame", "coord-empty-frame").await;
    rank.outbound
        .send(RankControl { control: None })
        .await
        .unwrap();
    let err = next_event(&mut rank.events, Duration::from_secs(5))
        .await
        .expect_err("an empty frame is a status, never an event");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(
        err.message().contains("empty RankControl frame"),
        "the trailer names the violation: {err}"
    );
    assert_eq!(row_facts(&server, "job-empty-frame").await, before);
}

/// The host DRAIN arm: flipping the session's own `HostAdmission` phase
/// ends every held rank `Aborted{Drain}` at once (well inside one
/// heartbeat — never waiting for a tick or the park bound), the row
/// untouched. Mutation proof: a hold loop without the phase arm parks to
/// `NoBody` at the lease bound instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_drain_when_the_host_drains() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-drain", "coord-drain").await;
    let started = tokio::time::Instant::now();
    assert!(server.engine.host_admission().begin_drain());
    expect_aborted(&mut rank, AbortReason::Drain, Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < HEARTBEAT,
        "a drain cuts the session at once, never at the next tick"
    );
    assert_eq!(row_facts(&server, "job-drain").await, before);
}

/// The SAME drain end through the real server shutdown path (the peer
/// harness's `shutdown` sender drives `serve_with_shutdown`'s DRAIN arm,
/// which flips the session's phase for a worker-less server too): the
/// held session ends `Aborted{Drain}` and the serve task then completes —
/// a held rank never holds the drain open to the park bound. Mutation
/// proof: a DRAIN arm that flips the phase only through
/// `EmbeddedWorker::begin_drain` (absent here: `[worker] enabled = false`)
/// leaves this session parked and the serve future blocked past the
/// park bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_drain_on_server_shutdown() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-shutdown", "coord-shutdown").await;
    let crate::common::grpc::PeerEngineServer {
        shutdown,
        handle,
        engine,
        _dir,
        ..
    } = server;
    let started = tokio::time::Instant::now();
    shutdown.send(()).unwrap();
    expect_aborted(&mut rank, AbortReason::Drain, Duration::from_secs(5)).await;
    assert!(started.elapsed() < HEARTBEAT * 2);
    tokio::time::timeout(Duration::from_secs(20), handle)
        .await
        .expect("the serve task completes once the held session ended")
        .expect("the serve task joins");
    // The row is read back through the (still open) engine handle.
    let after = {
        use jammi_db::catalog::backend::{SqlValue, TxOptions};
        engine
            .catalog()
            .backend_arc()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT status, attempts FROM jobs WHERE job_id = $1",
                            &[SqlValue::TextOwned("job-shutdown".into())],
                            |row| Ok((row.get::<String>("status")?, row.get::<i32>("attempts")?)),
                        )
                        .await
                    })
                },
            )
            .await
    };
    match after {
        Ok(Some((status, attempts))) => {
            assert_eq!((status, attempts), (before.status.clone(), before.attempts));
        }
        // The engine closed its catalog with the serve task: the row's
        // untouched-ness on THIS path is pinned by the drain row above,
        // which reads it back before any shutdown.
        other => eprintln!("catalog closed with the server; row not re-read: {other:?}"),
    }
}

/// Re-verification, the `Refuted` end: a row fact moving after admission (the job
/// flipped off `running`) ends the held session `Aborted{Refuted}` at the
/// next tick — assembly-scoped, the one end that counts toward the
/// assembly's attempts (`ReverifyEnd::counts_toward_assembly_attempts`).
/// The row is untouched by the end itself (only the fixture's own status
/// flip differs).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_refuted_when_the_row_no_longer_holds() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-refuted", "coord-refuted").await;
    raw_sql(
        &server,
        "UPDATE jobs SET status = 'completed' WHERE job_id = 'job-refuted'".into(),
    )
    .await;
    expect_aborted(&mut rank, AbortReason::Refuted, HEARTBEAT * 3).await;
    assert_eq!(
        row_facts(&server, "job-refuted").await,
        RowFacts {
            status: "completed".into(),
            ..before
        }
    );
}

/// Re-verification, the `Unavailable` end: the catalog not answering at re-verification
/// (the `instances` table dropped after admission, so `fresh_instance`'s
/// read faults — the same DROP-TABLE technique
/// `run_rank_fresh_instance_fault_is_unavailable` uses at admission) ends
/// the session `Aborted{Unavailable}` — transient, assembly-scoped, never
/// counted — distinct from `Refuted` on the wire.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_unavailable_when_the_catalog_faults() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before) =
        admitted_world_one(&server, "job-unavailable", "coord-unavailable").await;
    raw_sql(&server, "DROP TABLE instances".into()).await;
    expect_aborted(&mut rank, AbortReason::Unavailable, HEARTBEAT * 3).await;
    assert_eq!(row_facts(&server, "job-unavailable").await, before);
}

/// Re-verification, the `StoreUnavailable` end: THIS host's object store faulting at
/// re-verification — the admitted `world_size == 2` session's training-set
/// row moved onto a scheme this build compiles no driver for, after
/// admission — ends the session `Aborted{StoreUnavailable}`: member-scoped,
/// never counted, distinct from both `Refuted` (a row/artifact fact) and
/// `Unavailable` (the catalog). Together with the two rows above the three
/// ends are pairwise distinct ON THE WIRE, each with its own scope and
/// count rule (`ReverifyEnd`'s own unit test pins the triple). Mutation
/// proof: a re-verification that never re-resolves the identity parks to
/// `NoBody`; one that classifies every store error as `Refuted` fails the
/// reason assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_store_unavailable_when_this_hosts_store_faults() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before, ready) = admitted_world_two(
        &server,
        tenant(0xb1),
        "job-store-unavail",
        "coord-store-unavail",
    )
    .await;
    raw_sql(
        &server,
        format!(
            "UPDATE result_tables SET parquet_path = 's3://gang-store-unavailable/x.parquet' \
             WHERE table_name = '{}'",
            ready.table
        ),
    )
    .await;
    expect_aborted(&mut rank, AbortReason::StoreUnavailable, HEARTBEAT * 3).await;
    assert_eq!(row_facts(&server, "job-store-unavail").await, before);
}

/// Re-verification at the artifact: a `world_size == 2` session whose training set's
/// sidecar is stripped of its leaf inventory AFTER admission (reads as
/// absent) ends `Aborted{Refuted}` — the artifact's fact, assembly-scoped,
/// never this host's `StoreUnavailable`. Pins the split between "the
/// sidecar does not verify" and "this host could not read it".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_held_session_ends_refuted_when_the_sidecar_stops_verifying() {
    let server = start_no_worker_server().await;
    let (mut rank, _attempt, before, ready) = admitted_world_two(
        &server,
        tenant(0xc1),
        "job-sidecar-refuted",
        "coord-sidecar-refuted",
    )
    .await;
    strip_leaves_from_sidecar(&server, &ready.parquet_path).await;
    expect_aborted(&mut rank, AbortReason::Refuted, HEARTBEAT * 3).await;
    assert_eq!(row_facts(&server, "job-sidecar-refuted").await, before);
}

/// The rank body's own pre-collective verify (the per-partition leaf
/// inventory, consumed here): a `world_size == 2` session whose
/// training set's Parquet bytes were corrupted inside a row group AFTER
/// the sidecar attested them — a fault the sidecar-level re-verification
/// tick cannot see (it compares the sidecar's own digest, never the
/// object's bytes) — ends `Aborted{StoreUnavailable}` from the body's
/// leaf verify, member-scoped, within the body's prologue, the row
/// untouched. Mutation proof: a body that skips `verify_partition_leaves`
/// binds the corrupted table, loads the model and waits at its first
/// collective — no event inside the bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_rank_body_refuses_a_partition_whose_leaf_does_not_verify_as_store_unavailable() {
    let server = start_no_worker_server().await;
    let (attempt, before, ready) =
        world_two_ready(&server, tenant(0xe1), "job-bad-leaf", "coord-bad-leaf").await;
    // The first leaf's byte range, from the sidecar's own inventory.
    let store = server.engine.result_store();
    let url = jammi_db::storage::StorageUrl::parse(&ready.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("the sidecar");
    let (offset, length) = match &manifest.leaves[0].key {
        jammi_db::store::manifest::LeafKey::RowGroup { offset, length, .. } => (*offset, *length),
        other => panic!("a Parquet leaf: {other:?}"),
    };
    assert!(length > 0);
    // Flip one byte inside the row group's range on the file-backed store.
    let path = url
        .as_str()
        .strip_prefix("file://")
        .map(str::to_string)
        .unwrap_or_else(|| url.as_str().to_string());
    let mut bytes = std::fs::read(&path).expect("the parquet object is a local file");
    let at = (offset + length / 2) as usize;
    bytes[at] ^= 0xFF;
    std::fs::write(&path, bytes).unwrap();

    let mut rank = open_rank(
        &server,
        assign_frame_full("job-bad-leaf", attempt, 1, 2, "coord-bad-leaf"),
    )
    .await
    .expect("admission reads the sidecar, not the bytes: admitted");
    expect_admitted(&mut rank).await;
    expect_aborted(
        &mut rank,
        AbortReason::StoreUnavailable,
        Duration::from_secs(20),
    )
    .await;
    assert_eq!(row_facts(&server, "job-bad-leaf").await, before);
}

/// I-GANG, "tenant is derived, never accepted", on the wire: the request
/// carries the public tenant layer's own `jammi-session-id` metadata (the
/// ONE header a tenant is ever resolved from on the public listener),
/// naming a session no store binds — and a `world_size == 2` job bound to
/// tenant A, whose training set exists for tenant A alone, is ADMITTED all
/// the same: the peer listener runs no resolver, the handler reads no
/// metadata, and the row's own tenant is what pins the resolution.
/// Mutation proof: a handler that resolved the training set under a
/// caller-derived tenant (any value but the row's) refuses this call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_rank_never_reads_a_caller_supplied_tenant() {
    use jammi_wire::proto::gang::gang_service_client::GangServiceClient;

    let server = start_no_worker_server().await;
    let tenant_a = tenant(0xd1);
    let source_id = format!(
        "gang_caller_tenant_src_{}",
        jammi_test_utils::unique_suffix()
    );
    let ready = materialize_ready_table_for_tenant(&server, tenant_a, &source_id).await;
    let coord = "coord-caller-tenant";
    fresh_coordinator(&server, coord).await;
    let attempt = submit_and_claim_for_tenant(
        &server,
        tenant_a,
        "job-caller-tenant",
        coord,
        Duration::from_secs(300),
        WORLD2_SPEC,
    )
    .await;
    fill_pair(
        &server,
        "job-caller-tenant",
        coord,
        attempt,
        &ready.digest,
        &ready.table,
    )
    .await;

    let channel = crate::common::grpc::channel(server.peer_addr).await;
    let mut client = GangServiceClient::new(channel);
    let mut request = tonic::Request::new(tokio_stream::once(assign_frame_full(
        "job-caller-tenant",
        attempt,
        0,
        2,
        coord,
    )));
    request.metadata_mut().insert(
        "jammi-session-id",
        "a-session-of-some-other-tenant".parse().unwrap(),
    );
    let mut events = client
        .run_rank(request)
        .await
        .expect("the caller's metadata is never read: the row's own tenant admits")
        .into_inner();
    let first = next_event(&mut events, Duration::from_secs(5))
        .await
        .unwrap()
        .unwrap();
    assert!(is_admitted(&first), "{first:?}");
}
