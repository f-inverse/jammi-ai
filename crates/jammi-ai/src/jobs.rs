//! Job specs, results, and handles: the durable-compute entry points that
//! generalise the training queue to every compute verb (`jobs`/`instances`/
//! `workers`, migration 029).
//!
//! `JobSpec` is the tagged specification a producer submits: a training
//! kind is one of the three [`TrainingSpec`](crate::fine_tune::spec::TrainingSpec)
//! variants (unchanged — the worker's from-scratch reconstruction path is
//! untouched by this module), a compute kind is one of `ComputeSpec`'s.
//! `JobHandle` polls/waits/cancels a submitted job by id; `JobResult` is
//! the tagged terminal payload written to `jobs.result`.
//!
//! Two entry points, one executor: `InferenceSession::enqueue` always
//! accepts and returns a `JobHandle` immediately (`execution = 'queued'`,
//! claimed later by a [`crate::fine_tune::worker::JobWorker`]'s poll loop);
//! `InferenceSession::run_now` submits a fresh `execution = 'inline'` row
//! (never selected by that poll loop), claims it by id in the caller's own
//! task, and executes it through `execute_compute` before returning the
//! terminal `JobResult` directly — no worker needed. `execute_compute` is
//! the SAME dispatcher a [`crate::fine_tune::worker::JobWorker`] calls for a
//! claimed compute job, so an embedded synchronous call and a queued-and-
//! claimed compute job of the same kind run identical code.
//!
//! **Kinds that are deliberately NOT jobs.** `import_embeddings` is not a
//! [`crate::jobs::ComputeSpec`] variant: it is a GPU-free promotion of caller-supplied
//! vectors into a result table (`jammi_db::store`'s `import_embeddings`
//! materialises with no job of record for the same reason), not a compute
//! this crate dispatches, so it has no claim/lease/reclaim lifecycle to
//! generalise. `recompute` is not a variant either: it stays an inline,
//! synchronous pipeline verb over an already-materialised table's
//! provenance, and every producer it re-runs is one of the compute kinds
//! below — each of which IS a job when it runs — so a recompute never
//! needs a job row of its own. [`crate::jobs::JobResult`] therefore has exactly the two
//! payload shapes the compiled kinds produce: a `Model` (every training
//! kind) and a `Table` (every compute kind).
//!
//! **Cancellation.** `Catalog::cancel_request` sets `jobs.cancel_requested`;
//! the COMPUTE executor observes it at three checkpoint boundaries — after
//! the claim ([`crate::session::InferenceSession::run_now`],
//! `JobWorker::run_claimed_compute_job`) and before dispatch
//! ([`crate::jobs::execute_compute`]) — through [`crate::jobs::check_cancel`],
//! failing the row with [`jammi_db::error::JammiError::JobCancelled`]'s message. Every compute producer is a
//! single-shot write with no mid-table checkpoint, so a request that lands
//! after the producer has started is honoured only in the sense that it
//! stays recorded on the row: the run completes and the row finishes
//! `completed`.
//!
//! A TRAINING kind (`fine_tune`/`graph_fine_tune`/`context_predictor`) is
//! checkpointed mid-run, so it is checked differently and more often than
//! once: a watcher (`spawn_cancel_request_watcher`, private to
//! [`crate::fine_tune::worker`]) polls the same column at the worker's
//! lease-heartbeat cadence and folds an observed request into the SAME
//! cancel flag the trainer's own epoch-boundary check already reads for a
//! lost lease, so the SAME [`crate::fine_tune::worker::JobWorker::
//! run_claimed_job`] that already stops training on a lost lease also stops
//! it — and fails the row with [`jammi_db::error::JammiError::JobCancelled`]'s
//! message — on a genuine cancel request. See that module's
//! cooperative-cancellation doc.

use std::sync::Arc;

use jammi_db::catalog::jobs_repo::{FinishJobParams, SubmitJobParams};
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::{CacheOutcome, CachePolicy};

use crate::model::ModelTask;
use crate::pipeline::asof::AsofJoinSpec;
use crate::pipeline::graph_propagation::PropagateRequest;
use crate::pipeline::neighbor_graph::BuildNeighborGraph;
use crate::session::InferenceSession;

/// A durable, self-contained description of a compute job: the verb that
/// produced it (the variant) and the inputs [`execute_compute`] reconstructs
/// the run from. Persisted as JSON on `jobs.spec`; the variant's serde tag
/// mirrors into `jobs.kind` — see `crate::fine_tune::worker::is_compute_kind`
/// for the vocabulary this must stay in sync with.
///
/// Every variant now carries its own `cache` (item 3): item 2/K4 routes
/// EVERY embedded synchronous compute verb through
/// [`InferenceSession::run_now`], including the three ([`Self::NeighborGraph`],
/// [`Self::Propagate`]) whose materializer still opts into the
/// definition-hash cache probe under [`CachePolicy::Use`] — N1's "the
/// NeighborGraph/Propagate cache probe stays" wording — so the caller's
/// cache policy has to survive the trip through `jobs.spec` and back, not be
/// silently forced to [`CachePolicy::Bypass`] the way the pre-item-3
/// `execute_compute` did for every compute kind.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComputeSpec {
    /// [`InferenceSession::build_neighbor_graph`]'s inputs.
    NeighborGraph {
        source_id: String,
        embedding_table: Option<String>,
        params: BuildNeighborGraph,
        cache: CachePolicy,
    },
    /// [`InferenceSession::propagate_embeddings`]'s inputs.
    Propagate {
        request: PropagateRequest,
        cache: CachePolicy,
    },
    /// [`InferenceSession::asof_join`]'s inputs. An as-of join has no
    /// cache-opt-in surface (every input is honestly `UnpinnedAtInstant`),
    /// so this carries no `cache` field.
    AsofJoin {
        spine: String,
        facts: String,
        spec: AsofJoinSpec,
    },
    /// [`InferenceSession::generate_embeddings`]'s inputs.
    Embedding {
        source_id: String,
        model_id: String,
        columns: Vec<String>,
        key_column: String,
        modality: jammi_wire::request::Modality,
        cache: CachePolicy,
    },
    /// [`InferenceSession::infer`]'s inputs.
    Infer {
        source_id: String,
        model_id: String,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        cache: CachePolicy,
    },
}

impl ComputeSpec {
    /// The `jobs.model_source` value this spec submits with: the canonical
    /// `ModelSource` string of the model a kind with a model input
    /// (`embedding`, `infer`) resolves against — the same vocabulary
    /// `result_tables.model_id` carries, so the row is a reference edge
    /// `Catalog::delete_model` guards while the job is non-terminal. `None`
    /// for the kinds that take no model.
    pub fn model_source(&self) -> Option<String> {
        match self {
            ComputeSpec::Embedding { model_id, .. } | ComputeSpec::Infer { model_id, .. } => {
                Some(crate::model::ModelSource::parse(model_id).to_string())
            }
            ComputeSpec::NeighborGraph { .. }
            | ComputeSpec::Propagate { .. }
            | ComputeSpec::AsofJoin { .. } => None,
        }
    }

    /// The source-side name a test rendezvous is keyed by (see
    /// [`compute_test_hooks`]): a test arms the hook for a source it alone
    /// registered, so parallel tests of the same kind never park each other.
    #[cfg(feature = "test-hooks")]
    pub fn rendezvous_key(&self) -> &str {
        match self {
            ComputeSpec::NeighborGraph { source_id, .. }
            | ComputeSpec::Embedding { source_id, .. }
            | ComputeSpec::Infer { source_id, .. } => source_id,
            ComputeSpec::Propagate { request, .. } => &request.source_id,
            ComputeSpec::AsofJoin { spine, .. } => spine,
        }
    }

    /// The `jobs.kind` tag this variant submits under — spelled out
    /// explicitly (rather than round-tripped through the serializer) so the
    /// mapping stays greppable, and so `crate::fine_tune::worker::is_compute_kind`
    /// and [`crate::fine_tune::worker::COMPILED_KINDS`] can name the same
    /// strings without invoking serde.
    pub fn kind(&self) -> &'static str {
        match self {
            ComputeSpec::NeighborGraph { .. } => "neighbor_graph",
            ComputeSpec::Propagate { .. } => "propagate",
            ComputeSpec::AsofJoin { .. } => "asof_join",
            ComputeSpec::Embedding { .. } => "embedding",
            ComputeSpec::Infer { .. } => "infer",
        }
    }
}

/// The union of every durable job specification this crate submits: the
/// three [`TrainingSpec`](crate::fine_tune::spec::TrainingSpec) training
/// kinds, unchanged, and the compute kinds in [`ComputeSpec`].
/// `#[serde(untagged)]`: each inner enum already carries its own flat `kind`
/// tag (mirroring `jobs.kind`), so the outer wrapper tries each inner
/// deserializer in turn (training first) rather than adding a second tag
/// layer that would shadow it — a spec whose `kind` is not one of the three
/// training variants simply fails `TrainingSpec`'s deserializer and falls
/// through to `ComputeSpec`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum JobSpec {
    /// A training kind: `fine_tune` / `graph_fine_tune` / `context_predictor`.
    /// Boxed — `TrainingSpec` is far larger than `ComputeSpec`, and boxing
    /// keeps `JobSpec` itself small to pass/return by value.
    Training(Box<crate::fine_tune::spec::TrainingSpec>),
    /// A compute kind: `neighbor_graph` / `propagate` / `asof_join`. Boxed
    /// to match `Training`'s indirection, keeping both variants small.
    Compute(Box<ComputeSpec>),
}

impl JobSpec {
    /// The `jobs.kind` tag this spec submits under.
    pub fn kind(&self) -> &'static str {
        match self {
            JobSpec::Training(t) => t.kind(),
            JobSpec::Compute(c) => c.kind(),
        }
    }
}

impl From<ComputeSpec> for JobSpec {
    fn from(c: ComputeSpec) -> Self {
        JobSpec::Compute(Box::new(c))
    }
}

impl From<crate::fine_tune::spec::TrainingSpec> for JobSpec {
    fn from(t: crate::fine_tune::spec::TrainingSpec) -> Self {
        JobSpec::Training(Box::new(t))
    }
}

/// The tagged terminal payload a job's executor writes to `jobs.result`.
/// [`crate::fine_tune::worker::JobWorker`]'s training path writes
/// [`JobResult::Model`] (folding the old dedicated `metrics` column into
/// this variant — the generalised `jobs` schema has no column of its own for
/// it); [`execute_compute`] writes [`JobResult::Table`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobResult {
    /// A training kind's output model.
    Model {
        model_id: String,
        artifact_path: String,
        /// Run-metrics JSON, or `None` when the run recorded none.
        metrics: Option<String>,
    },
    /// A compute kind's result table.
    Table {
        table: String,
        /// `"computed"`, or `"reused:{table}"` for an exact cache hit.
        cache_outcome: String,
    },
}

/// N1's per-attempt disposition: what a claimed job's worker does BEFORE it
/// ever calls [`execute_compute`], on every attempt after the first.
pub(crate) enum PartialResultDisposition {
    /// A prior attempt's table is `ready` — the job is done; finish it with
    /// this result, no materialize.
    Ready(JobResult),
    /// No usable prior table (absent `partial_result`, a `failed` table, or
    /// an orphaned `building` row this call just reaped) — fall through to
    /// the producer's normal path.
    MaterializeAnew,
    /// A prior attempt's table is still `building` under a LIVE lease (or a
    /// PEER reaper won the race to claim the expired one first) — back off;
    /// do not double-produce. The caller leaves the job `running` and
    /// returns without finishing or failing it; the job's own lease expiry
    /// (or a later attempt) will re-check.
    BackOff,
}

/// N1: on attempt `N > 1`, read `jobs.partial_result` and dispatch on that
/// table's status BEFORE running the producer's normal (materialize-anew)
/// path — `attempt == 1` (every [`InferenceSession::run_now`] inline job;
/// a queued job's first claim) always falls straight through, since there
/// is nothing yet to dispatch on. Called only from
/// [`crate::fine_tune::worker::JobWorker::run_claimed_compute_job`] — a
/// `run_now` inline job always submits a BRAND NEW `job_id`, so it is
/// always attempt 1 and never reaches a real dispatch arm (see
/// [`InferenceSession::run_now`]'s doc).
///
/// The `building` + expired-lease arm claims the row
/// (`Catalog::claim_expired_building_table`) then fails it: no compute
/// producer in this crate can RESUME a partial
/// write today (`neighbor_graph`/`propagate`/`asof_join`/`embedding`/`infer`
/// are each a single-shot write, never checkpointed mid-table — unlike
/// fine-tune's epoch checkpoints), so "adopt (finish) or fail+delete" always
/// takes the fail arm here; a future kind with a genuinely resumable
/// producer would add an adopt arm beside this one. Byte reclaim of the
/// failed row is left to the existing `ResultStore::recover()` sweep (this
/// call site has no access to `jammi-db`'s `pub(crate)
/// delete_objects_after_cas` — the reap-after-fail-CAS pattern
/// `store/mod.rs`'s recovery path itself uses).
pub(crate) async fn dispatch_partial_result(
    session: &Arc<InferenceSession>,
    catalog: &Catalog,
    tenant: Option<jammi_db::TenantId>,
    attempt: u32,
    partial_result: Option<&str>,
    instance_id: &str,
) -> Result<PartialResultDisposition> {
    if attempt <= 1 {
        return Ok(PartialResultDisposition::MaterializeAnew);
    }
    let Some(table_name) = partial_result else {
        return Ok(PartialResultDisposition::MaterializeAnew);
    };
    let Some(record) = catalog.get_result_table(table_name).await? else {
        // The row vanished (e.g. a prior fail+delete already reaped it) —
        // nothing to adopt.
        return Ok(PartialResultDisposition::MaterializeAnew);
    };
    match record.status.as_str() {
        "ready" => Ok(PartialResultDisposition::Ready(JobResult::Table {
            table: record.table_name,
            cache_outcome: "computed".to_string(),
        })),
        "building" => {
            let claim_writer_id = format!("{instance_id}/reclaim-{}", uuid::Uuid::new_v4());
            let lease = session.worker_intervals()?.lease;
            let cas = jammi_db::catalog::result_repo::ResultTableCas::expired(table_name, tenant);
            let claimed = catalog
                .claim_expired_building_table(&cas, &claim_writer_id, lease)
                .await?;
            if !claimed {
                // Either the writer is still renewing (a live lease), or a
                // peer reaper already won the claim — either way, do not
                // double-produce.
                return Ok(PartialResultDisposition::BackOff);
            }
            let fail_cas = jammi_db::catalog::result_repo::ResultTableCas::writer(
                table_name,
                &claim_writer_id,
                tenant,
            );
            // Best-effort: a miss here (a peer somehow raced this claim)
            // changes nothing about THIS attempt's own next step, which is
            // to materialize its own fresh table either way.
            catalog.fail_building_table(&fail_cas).await.ok();
            Ok(PartialResultDisposition::MaterializeAnew)
        }
        _ => Ok(PartialResultDisposition::MaterializeAnew),
    }
}

/// Execute one [`ComputeSpec`] to a terminal [`JobResult`], with NO catalog
/// job-row bookkeeping of its own beyond one read — the caller
/// ([`InferenceSession::run_now`] for an inline job, or
/// [`crate::fine_tune::worker::JobWorker`] for a queued one) owns the
/// claim/lease/finish around this call, and threads `job_attempt` (the
/// claim's own identity) into every producer so the result table's
/// `partial_result` CAS lands under the correct attempt (N1, N11/esc-107).
/// `catalog` is the handle scoped to the job's tenant (the worker's
/// tenant-pinned handle; `run_now`'s caller-scoped one): the one read this
/// function performs is the pre-dispatch cancel checkpoint
/// ([`check_cancel`]), which must resolve the job row under the tenant that
/// submitted it.
///
/// Dispatches to each verb's `*_materialize` method DIRECTLY, never to the
/// public `InferenceSession::build_neighbor_graph` /
/// `propagate_embeddings` / `asof_join` / `generate_embeddings` / `infer`
/// wrappers — those wrappers themselves call `InferenceSession::run_now`,
/// which calls back into this function; dispatching to the wrapper here
/// would recurse forever. A claimed compute job and a direct `run_now`
/// embedded call both bottom out at the SAME `*_materialize` call, so they
/// materialise byte-identical tables (K4).
///
/// Every kind's [`CachePolicy`] rides in its own `spec` field (item 3):
/// [`ComputeSpec::NeighborGraph`] and [`ComputeSpec::Propagate`] are pinned
/// (N1's withdrawn-unpinned-sentence) and genuinely honour `Use`; the other
/// kinds are unpinned, so `Use` is an honest miss for them, but the caller's
/// policy is never silently overridden here.
pub async fn execute_compute(
    session: &Arc<InferenceSession>,
    catalog: &Catalog,
    spec: &ComputeSpec,
    job_attempt: jammi_db::catalog::result_repo::JobAttempt<'_>,
) -> Result<JobResult> {
    #[cfg(feature = "test-hooks")]
    compute_test_hooks::maybe_park(
        spec.rendezvous_key(),
        compute_test_hooks::ParkPoint::BeforeDispatch,
    )
    .await;
    check_cancel(catalog, job_attempt.job_id).await?;
    match spec {
        ComputeSpec::NeighborGraph {
            source_id,
            embedding_table,
            params,
            cache,
        } => {
            let (record, outcome) = session
                .build_neighbor_graph_materialize(
                    source_id,
                    embedding_table.as_deref(),
                    params,
                    *cache,
                    Some(job_attempt),
                )
                .await?;
            Ok(table_result(record, outcome))
        }
        ComputeSpec::Propagate { request, cache } => {
            let (record, outcome) = session
                .propagate_embeddings_materialize(request, *cache, Some(job_attempt))
                .await?;
            Ok(table_result(record, outcome))
        }
        ComputeSpec::AsofJoin { spine, facts, spec } => {
            let record = session
                .asof_join_materialize(spine, facts, spec, Some(job_attempt))
                .await?;
            Ok(JobResult::Table {
                table: record.table_name,
                cache_outcome: "computed".to_string(),
            })
        }
        ComputeSpec::Embedding {
            source_id,
            model_id,
            columns,
            key_column,
            modality,
            cache,
        } => {
            let (record, outcome) = match modality {
                jammi_wire::request::Modality::Text => {
                    session
                        .generate_text_embeddings(
                            source_id,
                            model_id,
                            columns,
                            key_column,
                            *cache,
                            Some(job_attempt),
                        )
                        .await?
                }
                jammi_wire::request::Modality::Image => {
                    let image_column = crate::local_session::single_column(columns, "image")?;
                    session
                        .generate_image_embeddings(
                            source_id,
                            model_id,
                            image_column,
                            key_column,
                            *cache,
                            Some(job_attempt),
                        )
                        .await?
                }
                jammi_wire::request::Modality::Audio => {
                    let audio_column = crate::local_session::single_column(columns, "audio")?;
                    session
                        .generate_audio_embeddings(
                            source_id,
                            model_id,
                            audio_column,
                            key_column,
                            *cache,
                            Some(job_attempt),
                        )
                        .await?
                }
            };
            Ok(table_result(record, outcome))
        }
        ComputeSpec::Infer {
            source_id,
            model_id,
            task,
            content_columns,
            key_column,
            cache,
        } => {
            let source = crate::model::ModelSource::parse(model_id);
            let (table, _batches, outcome) = session
                .infer_materialize(
                    source_id,
                    &source,
                    *task,
                    content_columns,
                    key_column,
                    *cache,
                    Some(job_attempt),
                )
                .await?;
            let cache_outcome = match outcome {
                CacheOutcome::Computed => "computed".to_string(),
                CacheOutcome::Reused { table } => format!("reused:{table}"),
            };
            Ok(JobResult::Table {
                table,
                cache_outcome,
            })
        }
    }
}

/// The cancel checkpoint every executor path shares: read the job row and
/// stop with [`JammiError::JobCancelled`] when `cancel_requested` is set.
/// `catalog` must be scoped to the job's tenant (see [`execute_compute`]).
pub async fn check_cancel(catalog: &Catalog, job_id: &str) -> Result<()> {
    if catalog.get_job(job_id).await?.cancel_requested {
        return Err(JammiError::JobCancelled {
            job_id: job_id.to_string(),
        });
    }
    Ok(())
}

fn table_result(record: ResultTableRecord, outcome: CacheOutcome) -> JobResult {
    let cache_outcome = match outcome {
        CacheOutcome::Computed => "computed".to_string(),
        CacheOutcome::Reused { table } => format!("reused:{table}"),
    };
    JobResult::Table {
        table: record.table_name,
        cache_outcome,
    }
}

/// A handle to a submitted job — training or compute — returned by
/// [`InferenceSession::enqueue`]. Polls/waits/cancels by id through the
/// catalog; carries no in-memory job state of its own (a fresh process can
/// reattach to the same `job_id`).
pub struct JobHandle {
    /// The submitted job's id.
    pub job_id: String,
    /// The `jobs.kind` tag this job submitted under.
    pub kind: String,
    catalog: Arc<Catalog>,
}

impl JobHandle {
    pub(crate) fn new(job_id: String, kind: String, catalog: Arc<Catalog>) -> Self {
        Self {
            job_id,
            kind,
            catalog,
        }
    }

    /// Current status string (`queued`/`running`/`completed`/`failed`).
    pub async fn status(&self) -> Result<String> {
        Ok(self.catalog.get_job(&self.job_id).await?.status)
    }

    /// Block until the job reaches a terminal state, returning the parsed
    /// [`JobResult`] on success. A `failed` job surfaces its recorded error
    /// as a typed [`JammiError::FineTune`] (matching
    /// [`crate::fine_tune::training_job::TrainingJob::wait`]'s error shape,
    /// which this generalises).
    pub async fn wait(&self) -> Result<JobResult> {
        loop {
            let record = self.catalog.get_job(&self.job_id).await?;
            let status: JobStatus = record
                .status
                .parse()
                .map_err(|e| JammiError::Catalog(format!("{e}")))?;
            match status {
                JobStatus::Completed => {
                    let result_json = record.result.ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "job '{}' completed with no recorded result",
                            self.job_id
                        ))
                    })?;
                    return serde_json::from_str(&result_json).map_err(|e| {
                        JammiError::Catalog(format!(
                            "job '{}' recorded an unparseable result: {e}",
                            self.job_id
                        ))
                    });
                }
                JobStatus::Failed => {
                    let msg = record.error.unwrap_or_else(|| "job failed".into());
                    return Err(JammiError::FineTune(msg));
                }
                _ => tokio::time::sleep(std::time::Duration::from_millis(100)).await,
            }
        }
    }

    /// Request cancellation. `true` means the request is recorded on a
    /// still non-terminal row and the executor will honour it at its next
    /// checkpoint boundary (see the module docs' cancellation section):
    /// the job then lands `failed` with [`JammiError::JobCancelled`]'s
    /// message, and [`Self::wait`] surfaces that message. `false` when the
    /// job is already terminal or absent.
    pub async fn cancel(&self) -> Result<bool> {
        self.catalog.cancel_request(&self.job_id).await
    }
}

impl InferenceSession {
    /// Submit `spec` as a `queued` job and return immediately with a
    /// [`JobHandle`] — the durable, always-accepted entry point every
    /// `submit_*` wire verb and client binds to. A worker
    /// ([`crate::fine_tune::worker::JobWorker`]) claims it later; `priority`
    /// is the claim-ordering tie-break before `created_at` (`0` reproduces
    /// FIFO — migration 024).
    ///
    /// The row's model-side links are derived here exactly as the dedicated
    /// training entry points derive them (one rule, [`InferenceSession::
    /// training_job_links`]): a training kind carries `model_ref` (the base
    /// model's catalog PK, registered first when absent) and
    /// `output_model_id`; a compute kind with a model input carries
    /// `model_source` ([`ComputeSpec::model_source`]). A `SubmitJob` that
    /// lands here therefore produces a row `delete_model`'s referential
    /// scan and `JobStatus`'s `output_model_id` resolution both see as
    /// fully linked.
    pub async fn enqueue(self: &Arc<Self>, spec: JobSpec, priority: i32) -> Result<JobHandle> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let kind = spec.kind();
        let spec_json = serde_json::to_string(&spec)?;
        let (model_ref, output_model_id, model_source) = match &spec {
            JobSpec::Training(training) => {
                let links = self.training_job_links(training, &job_id).await?;
                (Some(links.model_ref), Some(links.output_model_id), None)
            }
            JobSpec::Compute(compute) => (None, None, compute.model_source()),
        };
        self.catalog()
            .submit_job(SubmitJobParams {
                job_id: &job_id,
                kind,
                execution: JobExecution::Queued,
                spec: &spec_json,
                model_ref: model_ref.as_deref(),
                output_model_id: output_model_id.as_deref(),
                model_source: model_source.as_deref(),
                priority,
            })
            .await?;
        Ok(JobHandle::new(
            job_id,
            kind.to_string(),
            Arc::clone(self.catalog_arc()),
        ))
    }

    /// Execute `spec` inline, in the caller's own task: submits an
    /// `execution = 'inline'` row (never visible to a
    /// [`crate::fine_tune::worker::JobWorker`]'s poll loop, since
    /// `claim_next` filters on `execution = 'queued'`), claims it by id,
    /// runs it through [`execute_compute`] under a keeper-registered lease
    /// (N3 — no heartbeat task), and finishes the row before returning the
    /// terminal [`JobResult`] directly — no worker needed. Every embedded
    /// synchronous compute verb ([`InferenceSession::build_neighbor_graph`],
    /// [`InferenceSession::propagate_embeddings`],
    /// [`InferenceSession::asof_join`]) is reachable through this same
    /// path, so a `run_now` call and a queued-and-claimed compute job of the
    /// same kind execute byte-identical code and their terminal payloads
    /// match (K4).
    ///
    /// The returned `Ok` is exactly "the row is `completed` with this
    /// result": the finish is the same attempt-guarded compare-and-set the
    /// worker path performs, and a miss — the row went terminal or changed
    /// hands underneath this call (a peer's reclaim, an operator's write)
    /// — is [`JammiError::JobAttemptSuperseded`], never an `Ok` the row
    /// contradicts. A cancel request observed at the post-claim checkpoint
    /// or at [`execute_compute`]'s pre-dispatch checkpoint fails the row and
    /// returns [`JammiError::JobCancelled`].
    pub async fn run_now(self: &Arc<Self>, spec: ComputeSpec) -> Result<JobResult> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let kind = spec.kind();
        let spec_json = serde_json::to_string(&spec)?;
        let model_source = spec.model_source();
        self.catalog()
            .submit_job(SubmitJobParams {
                job_id: &job_id,
                kind,
                execution: JobExecution::Inline,
                spec: &spec_json,
                model_ref: None,
                output_model_id: None,
                model_source: model_source.as_deref(),
                priority: 0,
            })
            .await?;
        let instance_id = self.instance_id().to_string();
        let lease = self.worker_intervals()?.lease;
        let claimed = self
            .catalog()
            .claim_by_id(&job_id, &instance_id, lease)
            .await?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "run_now: failed to claim its own freshly-submitted inline job '{job_id}'"
                ))
            })?;
        let hold = self
            .lease_keeper()
            .hold(jammi_db::catalog::lease_keeper::LeaseTarget::Job {
                job_id: job_id.clone(),
                instance_id: instance_id.clone(),
                attempts: claimed.attempts,
            });
        // `run_now` always submits a BRAND NEW `job_id` (never reused across
        // calls), so `claimed.attempts` is always 1 here — there is no
        // N1 partial_result to dispatch on (an inline job has no requeue
        // arm; it is a fresh row every time). `execute_compute` still runs
        // through the normal producer path with the real `JobAttempt` this
        // claim just won, so `create_table`'s `partial_result` CAS is
        // correctly attributed from attempt 1.
        let job_attempt = jammi_db::catalog::result_repo::JobAttempt {
            job_id: &job_id,
            instance_id: &instance_id,
            attempts: claimed.attempts,
        };
        // Post-claim checkpoint: a cancel that landed between submit and
        // claim is honoured before any producer runs.
        let outcome = match check_cancel(self.catalog(), &job_id).await {
            Ok(()) => execute_compute(self, self.catalog(), &spec, job_attempt).await,
            Err(e) => Err(e),
        };
        drop(hold);
        match outcome {
            Ok(job_result) => {
                let result_json = serde_json::to_string(&job_result)?;
                #[cfg(feature = "test-hooks")]
                compute_test_hooks::maybe_park(
                    spec.rendezvous_key(),
                    compute_test_hooks::ParkPoint::BeforeFinish,
                )
                .await;
                let finished = self
                    .catalog()
                    .finish_job(FinishJobParams {
                        job_id: &job_id,
                        instance_id: &instance_id,
                        attempts: claimed.attempts,
                        result: &result_json,
                    })
                    .await?;
                if !finished {
                    return Err(JammiError::JobAttemptSuperseded { job_id });
                }
                Ok(job_result)
            }
            Err(e) => {
                // #485: a swallowed `Err` here (the old `.ok()`) could leave
                // this inline row `running` for the process lifetime — this
                // function never retries and nothing else finalizes an
                // inline row's lease, so a lost write here is not "left for
                // reclaim" the way a queued job's would be. `Ok(false)` is
                // the benign race (a peer somehow holds this attempt's
                // lease already) and stays a debug log; a genuine catalog
                // `Err` is surfaced at error level so it is never silent.
                match self
                    .catalog()
                    .fail_job(&job_id, &instance_id, claimed.attempts, &e.to_string())
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        tracing::debug!(
                            job_id = %job_id,
                            "run_now: lost the lease before recording the failure"
                        );
                    }
                    Err(fail_err) => {
                        tracing::error!(
                            job_id = %job_id,
                            error = %fail_err,
                            "run_now: failed to record the job's terminal failure -- the row \
                             would otherwise stay `running` for the process lifetime"
                        );
                    }
                }
                Err(e)
            }
        }
    }
}

/// Test-only rendezvous inside the compute executor: a test arms a park
/// point for the source name it alone registered, then the next run of a
/// spec over that source parks there until the test releases it — so a
/// cancel request, a peer's reclaim, or an operator's write can be
/// manufactured at a documented checkpoint rather than raced against a
/// real producer. Compiled only under `feature = "test-hooks"` (this
/// crate's own test targets enable it through a self dev-dependency); no
/// production path observes anything here beyond the `maybe_park` calls
/// themselves, which return immediately when nothing is armed.
#[cfg(feature = "test-hooks")]
pub mod compute_test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use tokio::sync::Notify;

    /// Where a run parks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParkPoint {
        /// Inside `execute_compute`, before the cancel checkpoint and the
        /// producer dispatch — "the run is claimed and about to produce".
        BeforeDispatch,
        /// Inside `run_now`, after the producer returned and before the
        /// finish compare-and-set — "the run produced and is about to
        /// finish".
        BeforeFinish,
    }

    struct Armed {
        key: String,
        point: ParkPoint,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    fn armed() -> &'static Mutex<Vec<Armed>> {
        static ARMED: OnceLock<Mutex<Vec<Armed>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of one armed park: wait for the run to arrive, then
    /// let it continue. Dropping the handle without releasing leaves the
    /// run parked — release it explicitly.
    pub struct ParkHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    impl ParkHandle {
        /// Resolve once the run has reached the park point.
        pub async fn wait_parked(&self) {
            while !self.parked.load(Ordering::SeqCst) {
                self.parked_notify.notified().await;
            }
        }

        /// Let the parked run continue.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_one();
        }
    }

    /// Arm one park for the next run whose spec's rendezvous key is `key`
    /// (see `ComputeSpec::rendezvous_key`) at `point`. One-shot: the park
    /// disarms as soon as a run takes it.
    pub fn arm(key: &str, point: ParkPoint) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Armed {
                key: key.to_string(),
                point,
                parked: Arc::clone(&parked),
                parked_notify: Arc::clone(&parked_notify),
                released: Arc::clone(&released),
                release_notify: Arc::clone(&release_notify),
            });
        ParkHandle {
            parked,
            parked_notify,
            released,
            release_notify,
        }
    }

    pub(super) async fn maybe_park(key: &str, point: ParkPoint) {
        let taken = {
            let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.key == key && a.point == point)
                .map(|i| list.remove(i))
        };
        let Some(armed) = taken else {
            return;
        };
        armed.parked.store(true, Ordering::SeqCst);
        armed.parked_notify.notify_one();
        while !armed.released.load(Ordering::SeqCst) {
            armed.release_notify.notified().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::graph_neighbourhood::EdgeDirection;
    use crate::pipeline::graph_propagation::{PropagationOutput, PropagationWeighting};

    /// `JobSpec`/`ComputeSpec` round-trip through JSON byte-identically to how
    /// they will be persisted on `jobs.spec` and read back by a worker on a
    /// fresh process.
    #[test]
    fn compute_spec_round_trips_through_json() {
        let spec = ComputeSpec::Propagate {
            request: PropagateRequest::new(
                "src",
                crate::pipeline::graph_neighbourhood::EdgeSourceRef::NeighborGraph {
                    table_name: "edges".into(),
                },
            ),
            cache: CachePolicy::Bypass,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: ComputeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "propagate");
        assert_eq!(spec.kind(), back.kind());

        let job_spec: JobSpec = spec.into();
        assert_eq!(job_spec.kind(), "propagate");
        let job_json = serde_json::to_string(&job_spec).unwrap();
        let back_job: JobSpec = serde_json::from_str(&job_json).unwrap();
        assert_eq!(back_job.kind(), "propagate");
    }

    /// A training-kind `JobSpec` round-trips through the SAME `#[serde(tag =
    /// "kind")]` vocabulary `TrainingSpec` itself uses — the untagged wrapper
    /// adds no second tag layer.
    #[test]
    fn training_job_spec_round_trips_and_keeps_the_flat_kind_tag() {
        let spec = JobSpec::Training(Box::new(crate::fine_tune::spec::TrainingSpec::FineTune {
            source: "src".into(),
            columns: vec!["text".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: jammi_db::ModelTask::TextEmbedding,
            common: crate::fine_tune::spec::TrainingCommon {
                base_model: "base".into(),
                config: crate::fine_tune::FineTuneConfig::default(),
            },
        }));
        let json = serde_json::to_string(&spec).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("kind").and_then(|k| k.as_str()),
            Some("fine_tune"),
            "the flat `kind` tag must be reachable at the top level of the JSON, got: {json}"
        );
        let back: JobSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "fine_tune");
    }

    /// A `NeighborGraph`/`AsofJoin` compute spec's param structs
    /// (`BuildNeighborGraph`, `AsofJoinSpec`) round-trip too, not just
    /// `PropagateRequest` — the contract's "param structs gain
    /// Serialize/Deserialize" covers all of them.
    #[test]
    fn neighbor_graph_and_asof_join_specs_round_trip() {
        let ng = ComputeSpec::NeighborGraph {
            source_id: "s".into(),
            embedding_table: None,
            params: BuildNeighborGraph {
                k: 5,
                ..Default::default()
            },
            cache: CachePolicy::Bypass,
        };
        let json = serde_json::to_string(&ng).unwrap();
        let back: ComputeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "neighbor_graph");

        let left = crate::pipeline::asof::AsofKey {
            by: vec!["id".into()],
            time: "t".into(),
        };
        let right = crate::pipeline::asof::AsofKey {
            by: vec!["id".into()],
            time: "t".into(),
        };
        let spec = crate::pipeline::asof::AsofJoinSpecBuilder::new(left, right).build();
        let asof = ComputeSpec::AsofJoin {
            spine: "spine".into(),
            facts: "facts".into(),
            spec,
        };
        let json = serde_json::to_string(&asof).unwrap();
        let back: ComputeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "asof_join");
    }

    /// The `PropagateRequest` builder knobs (direction/hops/weighting/alpha/
    /// output) all survive the round-trip, not just the required fields —
    /// otherwise a persisted job would silently revert a caller's explicit
    /// choice back to the default on the worker's fresh-process replay.
    #[test]
    fn propagate_request_builder_knobs_survive_the_round_trip() {
        let request = PropagateRequest::new(
            "src",
            crate::pipeline::graph_neighbourhood::EdgeSourceRef::NeighborGraph {
                table_name: "edges".into(),
            },
        )
        .with_direction(EdgeDirection::In)
        .with_hops(3)
        .with_weighting(PropagationWeighting::Uniform)
        .with_alpha(0.25)
        .with_output(PropagationOutput::JumpingKnowledge);
        let json = serde_json::to_string(&request).unwrap();
        let back: PropagateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.direction, EdgeDirection::In);
        assert_eq!(back.hops, 3);
        assert_eq!(back.weighting, PropagationWeighting::Uniform);
        assert_eq!(back.alpha, 0.25);
        assert_eq!(back.output, PropagationOutput::JumpingKnowledge);
    }
}
