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
//! **Cancellation.** `Catalog::cancel_request` ends a job no worker has
//! claimed at once (`queued -> cancelled`) and flags a running one
//! (`jobs.cancel_requested`), which its executor ends `cancelled` through the
//! lease-guarded `Catalog::cancel_job`. `UnsuccessfulEnd` decides `failed`,
//! `cancelled` or left-for-reclaim (a placed attempt whose executor the
//! compute plane lost: spent, re-offered to a successor) from the
//! executor's typed error — one rule for every job kind.
//! The COMPUTE executor observes the flag at three checkpoint boundaries — after
//! the claim ([`crate::session::InferenceSession::run_now`],
//! `JobWorker::run_claimed_compute_job`) and before dispatch
//! ([`crate::jobs::execute_compute`]) — through [`crate::jobs::check_cancel`],
//! which returns [`jammi_db::error::JammiError::JobCancelled`]. Every compute producer is a
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
//! it — and ends the row `cancelled` — on a genuine cancel request. See that module's
//! cooperative-cancellation doc.

use std::sync::Arc;

use jammi_db::catalog::jobs_repo::{FinishJobParams, SubmitJobParams};
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::{CacheOutcome, CachePolicy};
use tracing::Instrument;

use crate::model::ModelTask;
use crate::pipeline::asof::AsofJoinSpec;
use crate::pipeline::graph_propagation::PropagateRequest;
use crate::pipeline::graph_structure::StructureRequest;
use crate::pipeline::neighbor_graph::BuildNeighborGraph;
use crate::session::InferenceSession;

/// A durable, self-contained description of a compute job: the verb that
/// produced it (the variant) and the inputs [`execute_compute`] reconstructs
/// the run from. Persisted as JSON on `jobs.spec`; the variant's serde tag
/// mirrors into `jobs.kind` — see `crate::fine_tune::worker::is_compute_kind`
/// for the vocabulary this must stay in sync with.
///
/// Every variant carries its own `cache`: EVERY embedded synchronous compute
/// verb routes through [`InferenceSession::run_now`], including the ones
/// ([`Self::NeighborGraph`], [`Self::Propagate`]) whose materializer opts
/// into the definition-hash cache probe under [`CachePolicy::Use`] — so the
/// caller's cache policy has to survive the trip through `jobs.spec` and
/// back, never be silently forced to [`CachePolicy::Bypass`].
///
/// `#[serde(deny_unknown_fields)]`: a `jobs.spec` row is always
/// engine-written from a decoded spec, so an unknown key can only arrive
/// via a hand edit or corruption, and [`JobSpec`]'s own `Deserialize`
/// already dispatches to this type ONLY when the row's `kind` names one of
/// this enum's own variants — see that type's doc.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
    /// [`InferenceSession::generate_structure_embeddings`]'s inputs.
    GraphStructure {
        request: StructureRequest,
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
            | ComputeSpec::GraphStructure { .. }
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
            ComputeSpec::GraphStructure { request, .. } => &request.source_id,
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
            ComputeSpec::GraphStructure { .. } => "graph_structure",
            ComputeSpec::AsofJoin { .. } => "asof_join",
            ComputeSpec::Embedding { .. } => "embedding",
            ComputeSpec::Infer { .. } => "infer",
        }
    }
}

/// The union of every durable job specification this crate submits: the
/// three [`TrainingSpec`](crate::fine_tune::spec::TrainingSpec) training
/// kinds and the six compute kinds in [`ComputeSpec`], flattened into ONE
/// directly-tagged enum — not a wrapper around either of those two types.
///
/// Derived `#[serde(tag = "kind", deny_unknown_fields)]`, over all eight
/// variants at once, field-for-field identical (same names, same order) to
/// [`crate::fine_tune::spec::TrainingSpec`]'s three variants and
/// [`ComputeSpec`]'s five: a row written through `JobSpec` is byte-identical
/// to one written directly through whichever of those two types the kind
/// belongs to (see the byte-pin tests below), so it round-trips through
/// EITHER type's decode unchanged. `JobSpec` is, in fact, the type every
/// production claim site decodes: `crate::fine_tune::worker`'s loop-claimer
/// and its Peer-rank path both decode a `jobs.spec` row as `JobSpec` and
/// project to [`TrainingSpec`](crate::fine_tune::spec::TrainingSpec) with
/// `JobSpec::as_training_spec`; its compute-claim path decodes `JobSpec`
/// the same way and projects to [`ComputeSpec`] with
/// `JobSpec::as_compute_spec` — `TrainingSpec`/`ComputeSpec` are never
/// deserialized directly from a `jobs.spec` row on any of those three
/// paths (see `JobSpec::as_training_spec`'s own doc for the full
/// caller list and why: a decode under `JobSpec`'s tag/
/// `deny_unknown_fields` pair catches a stray field the SAME way
/// regardless of which kind the row holds, rather than running two
/// independently-tagged decodes that could drift). Only this type's own
/// byte-pin tests below still construct a bare `TrainingSpec`/`ComputeSpec`
/// value directly, to prove the two shapes stay byte-identical.
///
/// This is a flat merge, not a wrapper, because a wrapper does not work:
/// an outer `#[serde(tag = "kind")]` enum whose variant is a newtype around
/// an ALREADY internally-tagged type (`Training(Box<TrainingSpec>)`) tries
/// to write `kind` twice on serialize — once for the outer variant, once
/// for the inner type's own tag — which `serde_json` refuses with a
/// "duplicate field `kind`" error; nesting the payload under an `Adjacent`
/// tag/content pair (`{"kind": ..., "content": {...}}`) round-trips but
/// pushes every training kind's `common` field to depth 2, breaking
/// [`crate::fine_tune::spec::TrainingCommon`]'s depth-1 contract with
/// `jammi-db`'s [`jammi_db::catalog::jobs_repo`] (see this crate and that
/// one's respective `world_size_from_spec_json` fixtures). Only the fully
/// flat merge below keeps `common` at depth 1 while giving `JobSpec` its
/// own single tag layer, so this enum's variants restate
/// `TrainingSpec`'s/`ComputeSpec`'s field lists rather than embedding
/// either type as a value.
///
/// No migration for existing rows: the wire bytes are unchanged (a row
/// written before this reshape already carries `kind` and no unknown
/// fields, since every write path is engine-written from a decoded spec),
/// so every row that decoded before still decodes identically now. A row
/// that CANNOT decode — no `kind` field, an unrecognised `kind`, or a
/// stray field under a recognised one — fails with a typed
/// [`serde_json::Error`] naming the offending field or kind; both
/// production readers of a `jobs.spec` row
/// (`crate::fine_tune::worker::JobWorker::run_claimed_compute_job` and
/// the training claim paths cited above) already fold that error into a
/// failure record keyed by the job's own id (`jobs.job_id` is the row's
/// primary key), so the typed error is never anonymous in practice.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum JobSpec {
    /// Field-for-field identical to
    /// [`TrainingSpec::FineTune`](crate::fine_tune::spec::TrainingSpec::FineTune).
    FineTune {
        source: String,
        columns: Vec<String>,
        method: crate::fine_tune::FineTuneMethod,
        task: ModelTask,
        common: crate::fine_tune::spec::TrainingCommon,
    },
    /// Field-for-field identical to
    /// [`TrainingSpec::GraphFineTune`](crate::fine_tune::spec::TrainingSpec::GraphFineTune).
    GraphFineTune {
        sources: crate::fine_tune::graph_sampler::GraphFineTuneSources,
        sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig,
        common: crate::fine_tune::spec::TrainingCommon,
    },
    /// Field-for-field identical to
    /// [`TrainingSpec::ContextPredictor`](crate::fine_tune::spec::TrainingSpec::ContextPredictor).
    ContextPredictor {
        source: String,
        predictor_spec: crate::pipeline::context_predictor::ContextPredictorTrainConfig,
    },
    /// Field-for-field identical to [`ComputeSpec::NeighborGraph`].
    NeighborGraph {
        source_id: String,
        embedding_table: Option<String>,
        params: BuildNeighborGraph,
        cache: CachePolicy,
    },
    /// Field-for-field identical to [`ComputeSpec::Propagate`].
    Propagate {
        request: PropagateRequest,
        cache: CachePolicy,
    },
    /// Field-for-field identical to [`ComputeSpec::GraphStructure`].
    GraphStructure {
        request: StructureRequest,
        cache: CachePolicy,
    },
    /// Field-for-field identical to [`ComputeSpec::AsofJoin`].
    AsofJoin {
        spine: String,
        facts: String,
        spec: AsofJoinSpec,
    },
    /// Field-for-field identical to [`ComputeSpec::Embedding`].
    Embedding {
        source_id: String,
        model_id: String,
        columns: Vec<String>,
        key_column: String,
        modality: jammi_wire::request::Modality,
        cache: CachePolicy,
    },
    /// Field-for-field identical to [`ComputeSpec::Infer`].
    Infer {
        source_id: String,
        model_id: String,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        cache: CachePolicy,
    },
}

impl JobSpec {
    /// The `jobs.kind` tag this spec submits under.
    pub fn kind(&self) -> &'static str {
        match self {
            JobSpec::FineTune { .. } => "fine_tune",
            JobSpec::GraphFineTune { .. } => "graph_fine_tune",
            JobSpec::ContextPredictor { .. } => "context_predictor",
            JobSpec::NeighborGraph { .. } => "neighbor_graph",
            JobSpec::Propagate { .. } => "propagate",
            JobSpec::GraphStructure { .. } => "graph_structure",
            JobSpec::AsofJoin { .. } => "asof_join",
            JobSpec::Embedding { .. } => "embedding",
            JobSpec::Infer { .. } => "infer",
        }
    }

    /// Reconstructs the equivalent
    /// [`TrainingSpec`](crate::fine_tune::spec::TrainingSpec) when `self` is
    /// one of the three training kinds, `None` for a compute kind. This is
    /// the PROJECTION every `jobs.spec`/`training_spec` production reader
    /// uses: `JobSpec` is the one type every persisted row decodes as (see
    /// the type doc), and a caller that needs the narrower standalone type
    /// — [`crate::fine_tune::spec::admit_training_spec`],
    /// [`InferenceSession::training_job_links`] (both take `&TrainingSpec`),
    /// the loop-claimer and Peer-rank training-claim paths
    /// (`crate::fine_tune::worker`), and [`crate::fine_tune::training_job::
    /// resolve_model_id`] — decodes `JobSpec` first, then projects with
    /// this method, never decoding `TrainingSpec` directly from a `jobs.spec`
    /// row again (only `JobSpec`'s OWN byte-pin tests still construct a bare
    /// `TrainingSpec` value, to prove the two shapes stay byte-identical).
    pub(crate) fn as_training_spec(&self) -> Option<crate::fine_tune::spec::TrainingSpec> {
        use crate::fine_tune::spec::TrainingSpec;
        Some(match self {
            JobSpec::FineTune {
                source,
                columns,
                method,
                task,
                common,
            } => TrainingSpec::FineTune {
                source: source.clone(),
                columns: columns.clone(),
                method: *method,
                task: *task,
                common: common.clone(),
            },
            JobSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            } => TrainingSpec::GraphFineTune {
                sources: sources.clone(),
                sample_config: *sample_config,
                common: common.clone(),
            },
            JobSpec::ContextPredictor {
                source,
                predictor_spec,
            } => TrainingSpec::ContextPredictor {
                source: source.clone(),
                predictor_spec: predictor_spec.clone(),
            },
            JobSpec::NeighborGraph { .. }
            | JobSpec::Propagate { .. }
            | JobSpec::GraphStructure { .. }
            | JobSpec::AsofJoin { .. }
            | JobSpec::Embedding { .. }
            | JobSpec::Infer { .. } => return None,
        })
    }

    /// [`Self::as_training_spec`]'s counterpart: reconstructs the equivalent
    /// [`ComputeSpec`] when `self` is one of the six compute kinds, `None`
    /// for a training kind. The one production reader of a compute-kind
    /// `jobs.spec` row ([`crate::fine_tune::worker::JobWorker::
    /// run_claimed_compute_job`]) decodes `JobSpec` first, then projects
    /// with this method.
    pub(crate) fn as_compute_spec(&self) -> Option<ComputeSpec> {
        Some(match self {
            JobSpec::NeighborGraph {
                source_id,
                embedding_table,
                params,
                cache,
            } => ComputeSpec::NeighborGraph {
                source_id: source_id.clone(),
                embedding_table: embedding_table.clone(),
                params: params.clone(),
                cache: *cache,
            },
            JobSpec::Propagate { request, cache } => ComputeSpec::Propagate {
                request: request.clone(),
                cache: *cache,
            },
            JobSpec::GraphStructure { request, cache } => ComputeSpec::GraphStructure {
                request: request.clone(),
                cache: *cache,
            },
            JobSpec::AsofJoin { spine, facts, spec } => ComputeSpec::AsofJoin {
                spine: spine.clone(),
                facts: facts.clone(),
                spec: spec.clone(),
            },
            JobSpec::Embedding {
                source_id,
                model_id,
                columns,
                key_column,
                modality,
                cache,
            } => ComputeSpec::Embedding {
                source_id: source_id.clone(),
                model_id: model_id.clone(),
                columns: columns.clone(),
                key_column: key_column.clone(),
                modality: *modality,
                cache: *cache,
            },
            JobSpec::Infer {
                source_id,
                model_id,
                task,
                content_columns,
                key_column,
                cache,
            } => ComputeSpec::Infer {
                source_id: source_id.clone(),
                model_id: model_id.clone(),
                task: *task,
                content_columns: content_columns.clone(),
                key_column: key_column.clone(),
                cache: *cache,
            },
            JobSpec::FineTune { .. }
            | JobSpec::GraphFineTune { .. }
            | JobSpec::ContextPredictor { .. } => return None,
        })
    }

    /// [`ComputeSpec::model_source`]'s equivalent for the two compute
    /// kinds that carry a model input; `None` for every other kind
    /// (including every training kind — a training job's model links come
    /// from [`Self::as_training_spec`] + [`InferenceSession::
    /// training_job_links`] instead, which resolves a different `jobs`
    /// column pair).
    fn model_source(&self) -> Option<String> {
        match self {
            JobSpec::Embedding { model_id, .. } | JobSpec::Infer { model_id, .. } => {
                Some(crate::model::ModelSource::parse(model_id).to_string())
            }
            _ => None,
        }
    }
}

impl From<ComputeSpec> for JobSpec {
    fn from(c: ComputeSpec) -> Self {
        match c {
            ComputeSpec::NeighborGraph {
                source_id,
                embedding_table,
                params,
                cache,
            } => JobSpec::NeighborGraph {
                source_id,
                embedding_table,
                params,
                cache,
            },
            ComputeSpec::Propagate { request, cache } => JobSpec::Propagate { request, cache },
            ComputeSpec::GraphStructure { request, cache } => {
                JobSpec::GraphStructure { request, cache }
            }
            ComputeSpec::AsofJoin { spine, facts, spec } => {
                JobSpec::AsofJoin { spine, facts, spec }
            }
            ComputeSpec::Embedding {
                source_id,
                model_id,
                columns,
                key_column,
                modality,
                cache,
            } => JobSpec::Embedding {
                source_id,
                model_id,
                columns,
                key_column,
                modality,
                cache,
            },
            ComputeSpec::Infer {
                source_id,
                model_id,
                task,
                content_columns,
                key_column,
                cache,
            } => JobSpec::Infer {
                source_id,
                model_id,
                task,
                content_columns,
                key_column,
                cache,
            },
        }
    }
}

impl From<crate::fine_tune::spec::TrainingSpec> for JobSpec {
    fn from(t: crate::fine_tune::spec::TrainingSpec) -> Self {
        use crate::fine_tune::spec::TrainingSpec;
        match t {
            TrainingSpec::FineTune {
                source,
                columns,
                method,
                task,
                common,
            } => JobSpec::FineTune {
                source,
                columns,
                method,
                task,
                common,
            },
            TrainingSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            } => JobSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            },
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => JobSpec::ContextPredictor {
                source,
                predictor_spec,
            },
        }
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
        /// [`CacheOutcome::Computed`] for a run that trained;
        /// [`CacheOutcome::Reused`] naming the model artifact (the same one
        /// `artifact_path` names) for a [`CachePolicy::Use`] job that
        /// completed against an already-published model of the same
        /// definition.
        cache_outcome: CacheOutcome,
    },
    /// A compute kind's result table.
    Table {
        table: String,
        /// [`CacheOutcome::Computed`], or [`CacheOutcome::Reused`] naming
        /// the table for an exact cache hit.
        cache_outcome: CacheOutcome,
    },
}

/// The per-attempt disposition: what a claimed job's worker does BEFORE it
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

/// On attempt `N > 1`, read `jobs.partial_result` and dispatch on that
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
/// write today (`neighbor_graph`/`propagate`/`graph_structure`/`asof_join`/`embedding`/`infer`
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
    job_id: &str,
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
        // nothing to adopt; the column still names it, so clear it or this
        // attempt's own `create_result_table` CAS is superseded.
        clear_stale_partial_result(catalog, job_id, instance_id, attempt, table_name).await;
        return Ok(PartialResultDisposition::MaterializeAnew);
    };
    match record.status.as_str() {
        "ready" => Ok(PartialResultDisposition::Ready(JobResult::Table {
            table: record.table_name,
            cache_outcome: CacheOutcome::Computed,
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
            clear_stale_partial_result(catalog, job_id, instance_id, attempt, table_name).await;
            Ok(PartialResultDisposition::MaterializeAnew)
        }
        // `failed` (the predecessor's own abort, or its `BuildingTable::drop`
        // after a RELEASE abort marked it), or any other terminal status:
        // nothing to adopt, and the column must not poison this attempt's
        // own CAS either.
        _ => {
            clear_stale_partial_result(catalog, job_id, instance_id, attempt, table_name).await;
            Ok(PartialResultDisposition::MaterializeAnew)
        }
    }
}

/// Clear the predecessor's `partial_result` before THIS attempt
/// materializes anew, so its own `create_result_table` CAS (`… AND
/// partial_result IS NULL`) can land. Without this every attempt >= 2 that
/// reaches a `MaterializeAnew` arm is superseded by its own predecessor's
/// stale pointer and the job lands `failed`.
/// Attempt-guarded like every jobs CAS (`Catalog::clear_partial_result`);
/// 0 rows = a peer already cleared it or superseded this attempt, which the
/// existing `JobAttemptSuperseded` path then reports. Best-effort: an error
/// is logged and the attempt proceeds to the same CAS.
async fn clear_stale_partial_result(
    catalog: &Catalog,
    job_id: &str,
    instance_id: &str,
    attempt: u32,
    table_name: &str,
) {
    match catalog
        .clear_partial_result(job_id, instance_id, attempt, table_name)
        .await
    {
        Ok(true) => {}
        Ok(false) => tracing::debug!(
            job_id,
            "partial_result already cleared or this attempt superseded; materializing anew regardless"
        ),
        Err(e) => tracing::warn!(
            job_id, error = %e,
            "clear_partial_result failed; this attempt's create_result_table may be superseded"
        ),
    }
}

/// Execute one [`ComputeSpec`] to a terminal [`JobResult`], with NO catalog
/// job-row bookkeeping of its own beyond one read — the caller
/// ([`InferenceSession::run_now`] for an inline job, or
/// [`crate::fine_tune::worker::JobWorker`] for a queued one) owns the
/// claim/lease/finish around this call, and threads `job_attempt` (the
/// claim's own identity) into every producer so the result table's
/// `partial_result` CAS lands under the correct attempt.
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
/// materialise byte-identical tables.
///
/// Every kind's [`CachePolicy`] rides in its own `spec` field:
/// [`ComputeSpec::NeighborGraph`] and [`ComputeSpec::Propagate`] are pinned
/// and genuinely honour `Use`; the other
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
        ComputeSpec::GraphStructure { request, cache } => {
            let (record, outcome) = session
                .generate_structure_embeddings_materialize(request, *cache, Some(job_attempt))
                .await?;
            Ok(table_result(record, outcome))
        }
        ComputeSpec::AsofJoin { spine, facts, spec } => {
            let record = session
                .asof_join_materialize(spine, facts, spec, Some(job_attempt))
                .await?;
            Ok(JobResult::Table {
                table: record.table_name,
                cache_outcome: CacheOutcome::Computed,
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
            Ok(JobResult::Table {
                table,
                cache_outcome: outcome,
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

/// The line a claimant logs, once, as its attempt ends with the plane's
/// loss of the executor holding it and is left for a successor. Carries
/// `job_id`, `attempt` and `error` (the typed loss) as fields.
pub const EXECUTOR_LOST_ATTEMPT_LOG: &str =
    "attempt lost with its executor; left for reclaim, an attempt spent";

/// How a claimed job ends without its result, decided by [`Self::of`]
/// from the typed error — one rule for every job kind: asked not to run,
/// lost with its executor, or could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnsuccessfulEnd {
    /// Terminal `failed`, the error's message on the row.
    Failed(String),
    /// Terminal `cancelled`: the caller asked for this end.
    Cancelled,
    /// Not terminal: the compute plane lost the executor holding the
    /// attempt's placed task ([`JammiError::ExecutorLost`], carried here
    /// as its message). The attempt is spent and the row is left `running`
    /// for the lease reclaim, which re-offers it to a successor claim
    /// while the reclaim cap admits (`Catalog::reclaim_expired_jobs`) —
    /// the end a gang's mid-run fault takes, `attempts + 1` at the
    /// successor's claim and `releases` untouched.
    LeftForReclaim(String),
}

impl UnsuccessfulEnd {
    /// The end `error` decides for a job run under `execution`: a
    /// [`JammiError::JobCancelled`] is `Cancelled`, never a failure; the
    /// plane's [`JammiError::ExecutorLost`] leaves a queued job for its
    /// successor, and is terminal for an inline one — an inline row has no
    /// successor (`Catalog::reclaim_expired_jobs` has no requeue arm for
    /// it) and nothing else would ever end it; everything else is
    /// `Failed`.
    pub(crate) fn of(error: &JammiError, execution: JobExecution) -> Self {
        match (error, execution) {
            (JammiError::JobCancelled { .. }, _) => Self::Cancelled,
            (JammiError::ExecutorLost { .. }, JobExecution::Queued) => {
                Self::LeftForReclaim(error.to_string())
            }
            (other, _) => Self::Failed(other.to_string()),
        }
    }

    /// Settle this end on the row: the lease-guarded terminal write for a
    /// terminal end (`false` when the attempt guard misses — the caller no
    /// longer owns the job), nothing for one left for reclaim, whose next
    /// write is the reclaim's own once this attempt's hold stops renewing
    /// the lease.
    pub(crate) async fn record(
        &self,
        catalog: &Catalog,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
    ) -> Result<bool> {
        match self {
            Self::Failed(error) => catalog.fail_job(job_id, instance_id, attempts, error).await,
            Self::Cancelled => catalog.cancel_job(job_id, instance_id, attempts).await,
            Self::LeftForReclaim(_) => Ok(true),
        }
    }
}

fn table_result(record: ResultTableRecord, outcome: CacheOutcome) -> JobResult {
    JobResult::Table {
        table: record.table_name,
        cache_outcome: outcome,
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
    /// [`JobResult`] on success. A job that ended without its result surfaces
    /// [`JobRecord::unsuccessful_error`](jammi_db::catalog::jobs_repo::JobRecord::unsuccessful_error):
    /// its recorded error for a `failed` job, [`JammiError::JobCancelled`] for
    /// a `cancelled` one.
    pub async fn wait(&self) -> Result<JobResult> {
        loop {
            let record = self.catalog.get_job(&self.job_id).await?;
            let status: JobStatus = record
                .status
                .parse()
                .map_err(|e| JammiError::Catalog(format!("{e}")))?;
            // Derived from `JobStatus::is_terminal_unsuccessful`/`Completed`
            // (the ONE terminality predicate) rather than a per-variant
            // match arm, so a future terminal-unsuccessful status joining
            // the vocabulary ends this wait with a one-line edit, not a
            // hunt for every per-variant match arm.
            if status == JobStatus::Completed {
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
            } else if let Some(error) = record.unsuccessful_error() {
                return Err(error);
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// Request cancellation. `true` means the request took effect: a job no
    /// worker had claimed is `cancelled` already; a running one is flagged and
    /// its executor ends it `cancelled` at its next checkpoint boundary (see
    /// the module docs' cancellation section). [`Self::wait`] then returns
    /// [`JammiError::JobCancelled`]. `false` when the job is already terminal
    /// or absent.
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
        // One of the three durable submit edges for a training spec (the
        // other two are `submit_fine_tune_spec_deduped` and
        // `train_context_predictor_deduped`): this one takes an
        // already-built `JobSpec`, so a training spec reaches the queue
        // through it without passing the per-verb entry points. Same
        // admission — [`crate::fine_tune::spec::admit_training_spec`] — same
        // typed refusals, nothing enqueued on any of them.
        //
        // For a training kind this function builds no `SubmitJobParams` of
        // its own:
        // [`crate::fine_tune::spec::submit_admitted_training`] is the ONE
        // place that construction happens, so this arm submits through it
        // rather than the `submit_job` call below. It writes the
        // RECONSTRUCTED, admitted `TrainingSpec`'s own serialization — not
        // `spec` (the `JobSpec`) — which is a byte-identical but SEPARATE
        // value (see `JobSpec`'s own doc on the two independent shapes; the
        // byte-pin tests are what makes this substitution sound). A compute
        // kind builds `SubmitJobParams` from `spec`'s own
        // `JobSpec::Serialize`.
        let job_id = uuid::Uuid::new_v4().to_string();
        let kind = spec.kind();
        match spec.as_training_spec() {
            Some(training) => {
                let admitted =
                    crate::fine_tune::spec::admit_training_spec(self.jammi_config(), training)?;
                let links = self.training_job_links(admitted.spec(), &job_id).await?;
                let submitted = crate::fine_tune::spec::submit_admitted_training(
                    self.catalog(),
                    &admitted,
                    &job_id,
                    &links.model_ref,
                    &links.output_model_id,
                    priority,
                    None,
                )
                .await?;
                debug_assert_eq!(
                    submitted.recorded_job_id, job_id,
                    "enqueue never dedupes (idempotency_key = None), so the recorded id is \
                     always the one this call minted"
                );
            }
            None => {
                let spec_json = serde_json::to_string(&spec)?;
                let model_source = spec.model_source();
                self.catalog()
                    .submit_job(SubmitJobParams {
                        job_id: &job_id,
                        kind,
                        execution: JobExecution::Queued,
                        spec: &spec_json,
                        model_ref: None,
                        output_model_id: None,
                        model_source: model_source.as_deref(),
                        priority,
                    })
                    .await?;
            }
        }
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
    /// (no heartbeat task), and finishes the row before returning the
    /// terminal [`JobResult`] directly — no worker needed. Every embedded
    /// synchronous compute verb ([`InferenceSession::build_neighbor_graph`],
    /// [`InferenceSession::propagate_embeddings`],
    /// [`InferenceSession::asof_join`]) is reachable through this same
    /// path, so a `run_now` call and a queued-and-claimed compute job of the
    /// same kind execute byte-identical code and their terminal payloads
    /// match.
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
            .instrument(tracing::debug_span!("job.submit"))
            .await?;
        let instance_id = self.instance_id().to_string();
        let lease = self.worker_intervals()?.lease;
        let claimed = self
            .catalog()
            .claim_by_id(&job_id, &instance_id, lease)
            .instrument(tracing::debug_span!("job.claim"))
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
        // partial_result to dispatch on (an inline job has no requeue
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
            Ok(()) => {
                execute_compute(self, self.catalog(), &spec, job_attempt)
                    .instrument(tracing::debug_span!("job.execute"))
                    .await
            }
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
                    .instrument(tracing::debug_span!("job.finish"))
                    .await?;
                if !finished {
                    return Err(JammiError::JobAttemptSuperseded { job_id });
                }
                Ok(job_result)
            }
            Err(e) => {
                // A swallowed `Err` here could leave this inline row `running` for the process
                // lifetime — this function never retries and nothing else finalizes an
                // inline row's lease, so a lost write here is not "left for
                // reclaim" the way a queued job's would be. `Ok(false)` is
                // the benign race (a peer somehow holds this attempt's
                // lease already) and stays a debug log; a genuine catalog
                // `Err` is surfaced at error level so it is never silent.
                match UnsuccessfulEnd::of(&e, JobExecution::Inline)
                    .record(self.catalog(), &job_id, &instance_id, claimed.attempts)
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
        /// Inside a graph propagation, after its adjacency snapshot is
        /// written and before any hop is planned over it — "the graph is
        /// read; the edge source may now move".
        AfterAdjacencySnapshot,
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

    pub(crate) async fn maybe_park(key: &str, point: ParkPoint) {
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

    #[test]
    fn graph_structure_spec_round_trips_through_json() {
        let request = StructureRequest::new(
            "ledger",
            crate::pipeline::graph_neighbourhood::EdgeSourceRef::NeighborGraph {
                table_name: "edges".into(),
            },
        )
        .with_weights([0.0, 1.0, 0.5])
        .with_seed(3);
        let spec = ComputeSpec::GraphStructure {
            request: request.clone(),
            cache: CachePolicy::Use,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: ComputeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "graph_structure");
        let ComputeSpec::GraphStructure {
            request: decoded,
            cache,
        } = back
        else {
            panic!("the kind tag selects the variant");
        };
        assert_eq!(decoded, request);
        assert_eq!(cache, CachePolicy::Use);

        let job_spec: JobSpec = spec.into();
        assert_eq!(job_spec.kind(), "graph_structure");
        let back_job: JobSpec =
            serde_json::from_str(&serde_json::to_string(&job_spec).unwrap()).unwrap();
        assert_eq!(
            back_job.as_compute_spec().unwrap().kind(),
            "graph_structure"
        );
    }

    /// A training-kind `JobSpec` round-trips through the SAME `#[serde(tag =
    /// "kind")]` vocabulary `TrainingSpec` itself uses — `JobSpec`'s own
    /// derive is one flat tag layer over both types' variants, not a
    /// wrapper that would add a second one.
    #[test]
    fn training_job_spec_round_trips_and_keeps_the_flat_kind_tag() {
        let spec: JobSpec = crate::fine_tune::spec::TrainingSpec::FineTune {
            source: "src".into(),
            columns: vec!["text".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: jammi_db::ModelTask::TextEmbedding,
            common: crate::fine_tune::spec::TrainingCommon {
                base_model: "base".into(),
                config: crate::fine_tune::FineTuneConfig::default(),
                world_size: crate::fine_tune::spec::DEFAULT_WORLD_SIZE,
                cache: jammi_db::store::CachePolicy::Bypass,
            },
        }
        .into();
        let json = serde_json::to_string(&spec).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("kind").and_then(|k| k.as_str()),
            Some("fine_tune"),
            "the flat `kind` tag must be reachable at the top level of the JSON, got: {json}"
        );
        assert_eq!(
            value.get("common").and_then(|c| c.get("world_size")),
            Some(&serde_json::json!(
                crate::fine_tune::spec::DEFAULT_WORLD_SIZE
            )),
            "`common` must be reachable at JSON depth 1, never nested under a tag/content \
             wrapper, got: {json}"
        );
        let back: JobSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind(), "fine_tune");
    }

    /// The untagged-collapse case, through `JobSpec` (not `TrainingSpec`
    /// directly): a stray `cache` key hand-edited under a `graph_fine_tune`
    /// row is refused with a typed error naming the field, never `"data did
    /// not match any variant of untagged enum"` — the message a
    /// `#[serde(untagged)]` enum produces when every inner deserializer it
    /// tries in turn fails. Without `deny_unknown_fields` on `JobSpec`'s own
    /// `#[serde(tag = "kind", ...)]` attribute the stray `cache` key would
    /// decode silently and `expect_err` would panic.
    #[test]
    fn a_stray_field_under_a_declared_kind_is_refused_through_job_spec_naming_the_field() {
        let original: JobSpec = crate::fine_tune::spec::TrainingSpec::GraphFineTune {
            sources: crate::fine_tune::graph_sampler::GraphFineTuneSources {
                node_source: "nodes".into(),
                id_column: "id".into(),
                text_column: "text".into(),
                edge_source: "edges".into(),
                src_column: "src".into(),
                dst_column: "dst".into(),
                provenance: crate::fine_tune::graph_sampler::EdgeProvenance::Declared,
            },
            sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig::default(),
            common: crate::fine_tune::spec::TrainingCommon {
                base_model: "local:tiny".into(),
                config: crate::fine_tune::FineTuneConfig::default(),
                world_size: crate::fine_tune::spec::DEFAULT_WORLD_SIZE,
                cache: CachePolicy::Bypass,
            },
        }
        .into();
        let mut value = serde_json::to_value(&original).expect("serialize to a JSON value");
        let object = value
            .as_object_mut()
            .expect("a graph_fine_tune spec is a JSON object");
        assert_eq!(
            object.get("kind").and_then(|k| k.as_str()),
            Some("graph_fine_tune")
        );
        // The hand-edit a real submit path can never produce: `cache` lives
        // inside `common`, so no variant serializes it at the top level.
        object.insert("cache".to_string(), serde_json::json!("use"));

        let err = serde_json::from_value::<JobSpec>(value)
            .expect_err("a stray field under the declared kind must be refused");
        let message = err.to_string();
        assert!(
            message.contains("cache"),
            "the refusal must name the stray field `cache`, got: {message}"
        );
        assert!(
            !message.contains("did not match any variant"),
            "the refusal must come from the ONE flat tag `kind` named, never a trial-and-error \
             fallthrough across variants: {message}"
        );
    }

    /// The same stray-field refusal on a COMPUTE kind (not just a training
    /// one, above): `deny_unknown_fields` covers every one of `JobSpec`'s
    /// eight variants at once, since it is ONE derive over the whole flat
    /// enum, not per-variant.
    #[test]
    fn a_stray_field_under_a_compute_kind_is_refused_naming_the_field() {
        let mut value = serde_json::to_value(JobSpec::NeighborGraph {
            source_id: "s".into(),
            embedding_table: None,
            params: BuildNeighborGraph::default(),
            cache: CachePolicy::Bypass,
        })
        .expect("serialize to a JSON value");
        value
            .as_object_mut()
            .expect("a neighbor_graph spec is a JSON object")
            .insert("unexpected_extra".to_string(), serde_json::json!(1));

        let err = serde_json::from_value::<JobSpec>(value)
            .expect_err("a stray field under a compute kind must be refused");
        assert!(
            err.to_string().contains("unexpected_extra"),
            "the refusal must name the stray field, got: {err}"
        );
    }

    /// Depth-complete refusal: `deny_unknown_fields` on `JobSpec` itself only
    /// refuses a stray field at depth 1 (directly under `kind`) — a stray
    /// field nested inside `common` (depth 2, `TrainingCommon`'s own shape)
    /// is a SEPARATE struct with its own `deny_unknown_fields` requirement.
    /// Without `#[serde(deny_unknown_fields)]` on
    /// `crate::fine_tune::spec::TrainingCommon`, a fixture exactly like this
    /// one decodes clean with the nested key silently dropped.
    #[test]
    fn a_stray_field_nested_inside_common_is_refused_not_silently_dropped() {
        let mut value = serde_json::to_value(JobSpec::FineTune {
            source: "src".into(),
            columns: vec!["text".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: jammi_db::ModelTask::TextEmbedding,
            common: crate::fine_tune::spec::TrainingCommon {
                base_model: "base".into(),
                config: crate::fine_tune::FineTuneConfig::default(),
                world_size: 1,
                cache: CachePolicy::Bypass,
            },
        })
        .expect("serialize to a JSON value");
        value
            .as_object_mut()
            .expect("a fine_tune spec is a JSON object")
            .get_mut("common")
            .expect("common is present")
            .as_object_mut()
            .expect("common is a JSON object")
            .insert("unexpected_nested_extra".to_string(), serde_json::json!(1));

        let err = serde_json::from_value::<JobSpec>(value)
            .expect_err("a stray field nested inside `common` must be refused, not dropped");
        assert!(
            err.to_string().contains("unexpected_nested_extra"),
            "the refusal must name the nested stray field, got: {err}"
        );
    }

    /// BYTE PIN: `JobSpec`'s serialization of a value is
    /// byte-identical to the corresponding `TrainingSpec`/`ComputeSpec`
    /// variant's own serialization of the same logical value, for every
    /// one of the eight compiled kinds — the property the type's own doc
    /// claims ("a row written through `JobSpec` is byte-identical to one
    /// written directly through whichever of those two types the kind
    /// belongs to"), which is what lets the two production training-claim
    /// sites and the one compute-claim site keep reading `TrainingSpec`/
    /// `ComputeSpec` directly with no awareness `JobSpec` exists. Serde's
    /// "adjacent" enum representation (`#[serde(tag = "kind", content =
    /// "content")]`) would move `common` to `.content.common`, desynchronising
    /// the two serializations this test compares byte for byte.
    #[test]
    fn job_spec_byte_pins_every_compiled_kind_against_its_own_type() {
        use crate::fine_tune::spec::TrainingSpec;

        let common = |world_size: u32| crate::fine_tune::spec::TrainingCommon {
            base_model: "base".into(),
            config: crate::fine_tune::FineTuneConfig::default(),
            world_size,
            cache: CachePolicy::Bypass,
        };
        let training_cases: Vec<TrainingSpec> = vec![
            TrainingSpec::FineTune {
                source: "src".into(),
                columns: vec!["text".into()],
                method: crate::fine_tune::FineTuneMethod::Lora,
                task: jammi_db::ModelTask::TextEmbedding,
                common: common(1),
            },
            TrainingSpec::GraphFineTune {
                sources: crate::fine_tune::graph_sampler::GraphFineTuneSources {
                    node_source: "nodes".into(),
                    id_column: "id".into(),
                    text_column: "text".into(),
                    edge_source: "edges".into(),
                    src_column: "src".into(),
                    dst_column: "dst".into(),
                    provenance: crate::fine_tune::graph_sampler::EdgeProvenance::Declared,
                },
                sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig::default(),
                common: common(2),
            },
            TrainingSpec::ContextPredictor {
                source: "src".into(),
                predictor_spec: crate::pipeline::context_predictor::ContextPredictorTrainConfig {
                    model_id: "predictor".into(),
                    architecture: crate::pipeline::context_predictor::ContextArchitecture::Cnp,
                    key_column: "id".into(),
                    task_column: "task".into(),
                    value_column: "y".into(),
                    context_k: 4,
                    hidden_dim: 8,
                    num_heads: 1,
                    num_layers: 1,
                    head: crate::pipeline::context_predictor::PredictiveHead::Gaussian {
                        objective: crate::pipeline::context_predictor::GaussianObjective::Crps,
                    },
                    epochs: 1,
                    learning_rate: 0.01,
                    grad_clip: 1.0,
                    test_task_fraction: 0.2,
                    min_task_count: 2,
                    seed: 1,
                },
            },
        ];
        for training in training_cases {
            let direct = serde_json::to_string(&training).unwrap();
            let via_job_spec = serde_json::to_string(&JobSpec::from(training.clone())).unwrap();
            assert_eq!(
                direct,
                via_job_spec,
                "JobSpec's bytes for kind {:?} must byte-match TrainingSpec's own",
                training.kind()
            );
        }

        let compute_cases: Vec<ComputeSpec> = vec![
            ComputeSpec::NeighborGraph {
                source_id: "s".into(),
                embedding_table: None,
                params: BuildNeighborGraph::default(),
                cache: CachePolicy::Bypass,
            },
            ComputeSpec::Propagate {
                request: PropagateRequest::new(
                    "src",
                    crate::pipeline::graph_neighbourhood::EdgeSourceRef::NeighborGraph {
                        table_name: "edges".into(),
                    },
                ),
                cache: CachePolicy::Bypass,
            },
            ComputeSpec::AsofJoin {
                spine: "spine".into(),
                facts: "facts".into(),
                spec: crate::pipeline::asof::AsofJoinSpecBuilder::new(
                    crate::pipeline::asof::AsofKey {
                        by: vec!["id".into()],
                        time: "t".into(),
                    },
                    crate::pipeline::asof::AsofKey {
                        by: vec!["id".into()],
                        time: "t".into(),
                    },
                )
                .build(),
            },
            ComputeSpec::Embedding {
                source_id: "s".into(),
                model_id: "m".into(),
                columns: vec!["text".into()],
                key_column: "id".into(),
                modality: jammi_wire::request::Modality::Text,
                cache: CachePolicy::Bypass,
            },
            ComputeSpec::Infer {
                source_id: "s".into(),
                model_id: "m".into(),
                task: jammi_db::ModelTask::TextEmbedding,
                content_columns: vec!["text".into()],
                key_column: "id".into(),
                cache: CachePolicy::Bypass,
            },
        ];
        for compute in compute_cases {
            let direct = serde_json::to_string(&compute).unwrap();
            let via_job_spec = serde_json::to_string(&JobSpec::from(compute.clone())).unwrap();
            assert_eq!(
                direct,
                via_job_spec,
                "JobSpec's bytes for kind {:?} must byte-match ComputeSpec's own",
                compute.kind()
            );
        }
    }

    /// A concrete literal pin (not just the cross-type comparison above) for
    /// one representative training kind, so the exact key set/order is
    /// nailed down, not merely "equal to whatever TrainingSpec produces
    /// today": `kind` first (the tag), then `common`/`world_size` reachable
    /// at depth 1.
    #[test]
    fn job_spec_fine_tune_literal_byte_pin() {
        let spec: JobSpec = crate::fine_tune::spec::TrainingSpec::FineTune {
            source: "src".into(),
            columns: vec!["text".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: jammi_db::ModelTask::TextEmbedding,
            common: crate::fine_tune::spec::TrainingCommon {
                base_model: "base".into(),
                config: crate::fine_tune::FineTuneConfig::default(),
                world_size: 1,
                cache: CachePolicy::Bypass,
            },
        }
        .into();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            json.starts_with(r#"{"kind":"fine_tune","source":"src","columns":["text"]"#),
            "the tag must lead, in variant-declaration field order, got: {json}"
        );
        assert!(
            json.contains(r#""common":{"base_model":"base""#),
            "`common` must be an object reachable directly (depth 1), got: {json}"
        );
        assert!(
            json.contains(r#""world_size":1"#),
            "world_size must be nested one level under common, got: {json}"
        );
    }

    /// A row whose fields happen to satisfy a DIFFERENT kind's shape is
    /// still refused under its own declared `kind`, never silently
    /// reinterpreted as that other kind. `neighbor_graph`'s shape
    /// (`source_id`, `embedding_table`, `params`, `cache`) cannot satisfy
    /// `context_predictor`'s (`source`, `predictor_spec`), so a
    /// `context_predictor` row is refused as `context_predictor` (missing
    /// its own fields) rather than silently decoding as some other variant.
    #[test]
    fn an_unrecognised_kind_is_refused_naming_the_kind_not_silently_mapped_to_a_different_variant()
    {
        let value = serde_json::json!({
            "kind": "not_a_real_kind",
            "source_id": "s",
            "embedding_table": null,
            "params": {},
            "cache": "bypass",
        });
        let err = serde_json::from_value::<JobSpec>(value)
            .expect_err("an unrecognised kind must be refused, never guessed at");
        let message = err.to_string();
        assert!(
            message.contains("not_a_real_kind"),
            "the refusal must name the unrecognised kind, got: {message}"
        );
    }

    /// A row with no `kind` field at all is refused as a missing field,
    /// never silently treated as one variant or another.
    #[test]
    fn a_job_spec_row_missing_kind_is_refused() {
        let value = serde_json::json!({ "source_id": "s" });
        let err = serde_json::from_value::<JobSpec>(value)
            .expect_err("a row with no `kind` field must be refused");
        assert!(
            err.to_string().contains("kind"),
            "the refusal must name the missing `kind` field, got: {err}"
        );
    }

    /// A `NeighborGraph`/`AsofJoin` compute spec's param structs
    /// (`BuildNeighborGraph`, `AsofJoinSpec`) round-trip too, not just
    /// `PropagateRequest` — every compute kind's param struct is
    /// Serialize/Deserialize.
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

    /// The one rule every job kind's unsuccessful end follows: a cancel is
    /// `cancelled` under either execution mode; the plane's loss of the
    /// attempt's executor leaves a queued job for its successor and is
    /// terminal for an inline one, which has no successor; every other
    /// error is terminal with its message.
    #[test]
    fn an_unsuccessful_end_is_decided_by_the_typed_error_and_the_execution_mode() {
        let cancelled = JammiError::JobCancelled {
            job_id: "job-1".into(),
        };
        let lost = JammiError::ExecutorLost {
            executor_id: "executor-1".into(),
            job_id: "7bY2".into(),
        };
        let refused = JammiError::SourceNotFound {
            source_id: "patents".into(),
        };
        for execution in [JobExecution::Queued, JobExecution::Inline] {
            assert_eq!(
                UnsuccessfulEnd::of(&cancelled, execution),
                UnsuccessfulEnd::Cancelled
            );
            assert_eq!(
                UnsuccessfulEnd::of(&refused, execution),
                UnsuccessfulEnd::Failed(refused.to_string())
            );
        }
        assert_eq!(
            UnsuccessfulEnd::of(&lost, JobExecution::Queued),
            UnsuccessfulEnd::LeftForReclaim(lost.to_string())
        );
        assert_eq!(
            UnsuccessfulEnd::of(&lost, JobExecution::Inline),
            UnsuccessfulEnd::Failed(lost.to_string())
        );
    }
}
