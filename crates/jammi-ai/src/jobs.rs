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

use std::sync::Arc;

use jammi_db::catalog::jobs_repo::{FinishJobParams, SubmitJobParams};
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::{CacheOutcome, CachePolicy};

use crate::pipeline::asof::AsofJoinSpec;
use crate::pipeline::graph_propagation::PropagateRequest;
use crate::pipeline::neighbor_graph::BuildNeighborGraph;
use crate::session::InferenceSession;

/// A durable, self-contained description of a compute job: the verb that
/// produced it (the variant) and the inputs [`execute_compute`] reconstructs
/// the run from. Persisted as JSON on `jobs.spec`; the variant's serde tag
/// mirrors into `jobs.kind` — see `crate::fine_tune::worker::is_compute_kind`
/// for the vocabulary this must stay in sync with.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComputeSpec {
    /// [`InferenceSession::build_neighbor_graph`]'s inputs.
    NeighborGraph {
        source_id: String,
        embedding_table: Option<String>,
        params: BuildNeighborGraph,
    },
    /// [`InferenceSession::propagate_embeddings`]'s inputs.
    Propagate(PropagateRequest),
    /// [`InferenceSession::asof_join`]'s inputs.
    AsofJoin {
        spine: String,
        facts: String,
        spec: AsofJoinSpec,
    },
}

impl ComputeSpec {
    /// The `jobs.kind` tag this variant submits under — spelled out
    /// explicitly (rather than round-tripped through the serializer) so the
    /// mapping stays greppable, and so `crate::fine_tune::worker::is_compute_kind`
    /// and [`crate::fine_tune::worker::COMPILED_KINDS`] can name the same
    /// strings without invoking serde.
    pub fn kind(&self) -> &'static str {
        match self {
            ComputeSpec::NeighborGraph { .. } => "neighbor_graph",
            ComputeSpec::Propagate(_) => "propagate",
            ComputeSpec::AsofJoin { .. } => "asof_join",
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

/// Execute one [`ComputeSpec`] to a terminal [`JobResult`], with NO catalog
/// job-row bookkeeping of its own — the caller
/// ([`InferenceSession::run_now`] for an inline job, or
/// [`crate::fine_tune::worker::JobWorker`] for a queued one) owns the
/// claim/lease/finish around this call. Dispatches to the SAME pipeline
/// entry points the direct `InferenceSession` verbs call
/// ([`InferenceSession::build_neighbor_graph`],
/// [`InferenceSession::propagate_embeddings`],
/// [`InferenceSession::asof_join`]), so a claimed compute job and a direct
/// embedded call materialise byte-identical tables. Every compute kind
/// bypasses the exact-match cache (`CachePolicy::Bypass`) here: a job's
/// caller has already dispatched on `jobs.partial_result` (N1) before
/// reaching this far, so a second, independent cache probe inside the
/// pipeline itself would be redundant.
pub async fn execute_compute(
    session: &Arc<InferenceSession>,
    spec: &ComputeSpec,
) -> Result<JobResult> {
    match spec {
        ComputeSpec::NeighborGraph {
            source_id,
            embedding_table,
            params,
        } => {
            let (record, outcome) = session
                .build_neighbor_graph(
                    source_id,
                    embedding_table.as_deref(),
                    params,
                    CachePolicy::Bypass,
                )
                .await?;
            Ok(table_result(record, outcome))
        }
        ComputeSpec::Propagate(request) => {
            let (record, outcome) = session
                .propagate_embeddings(request, CachePolicy::Bypass)
                .await?;
            Ok(table_result(record, outcome))
        }
        ComputeSpec::AsofJoin { spine, facts, spec } => {
            let record = session.asof_join(spine, facts, spec).await?;
            Ok(JobResult::Table {
                table: record.table_name,
                cache_outcome: "computed".to_string(),
            })
        }
    }
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

    /// Request cancellation. `false` when the job is already terminal or
    /// absent.
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
    pub async fn enqueue(self: &Arc<Self>, spec: JobSpec, priority: i32) -> Result<JobHandle> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let kind = spec.kind();
        let spec_json = serde_json::to_string(&spec)?;
        self.catalog()
            .submit_job(SubmitJobParams {
                job_id: &job_id,
                kind,
                execution: JobExecution::Queued,
                spec: &spec_json,
                model_ref: None,
                output_model_id: None,
                model_source: None,
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
    pub async fn run_now(self: &Arc<Self>, spec: ComputeSpec) -> Result<JobResult> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let kind = spec.kind();
        let spec_json = serde_json::to_string(&spec)?;
        self.catalog()
            .submit_job(SubmitJobParams {
                job_id: &job_id,
                kind,
                execution: JobExecution::Inline,
                spec: &spec_json,
                model_ref: None,
                output_model_id: None,
                model_source: None,
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
        let registration =
            self.lease_keeper()
                .register(jammi_db::catalog::lease_keeper::LeaseTarget::Job {
                    job_id: job_id.clone(),
                    instance_id: instance_id.clone(),
                    attempts: claimed.attempts,
                });
        let outcome = execute_compute(self, &spec).await;
        drop(registration);
        match outcome {
            Ok(job_result) => {
                let result_json = serde_json::to_string(&job_result)?;
                self.catalog()
                    .finish_job(FinishJobParams {
                        job_id: &job_id,
                        instance_id: &instance_id,
                        attempts: claimed.attempts,
                        result: &result_json,
                    })
                    .await?;
                Ok(job_result)
            }
            Err(e) => {
                self.catalog()
                    .fail_job(&job_id, &instance_id, claimed.attempts, &e.to_string())
                    .await
                    .ok();
                Err(e)
            }
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
        let spec = ComputeSpec::Propagate(PropagateRequest::new(
            "src",
            crate::pipeline::graph_neighbourhood::EdgeSourceRef::NeighborGraph {
                table_name: "edges".into(),
            },
        ));
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
