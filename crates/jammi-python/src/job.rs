use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;

use jammi_ai::fine_tune::training_job::TrainingJob;
use jammi_ai::jobs::JobResult;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::catalog::status::JobStatus;
use jammi_db::error::JammiError;

use crate::convert::serializable_to_pydict;
use crate::error::to_pyerr;

/// The three engine kinds `SubmitJob` accepts today (`_start_training_proto`
/// submits only these). Mirrors `crates/jammi-server/src/grpc/job.rs`'s
/// private `is_training_kind` byte-for-byte: this binding's `Attached` arm
/// must tell a training row from a compute row (`build_neighbor_graph`,
/// `propagate_embeddings`, `asof_join`, `generate_embeddings`, `infer` — every
/// one submitted internally via `InferenceSession::run_now`/`enqueue`, never
/// through this Python surface — also lands in the generalised `jobs` table)
/// the SAME way the wire handler does, so `job(id).output_model_id` agrees
/// with `JobStatusResponse.output_model_id` at every lifecycle state.
fn is_training_kind(kind: &str) -> bool {
    matches!(kind, "fine_tune" | "graph_fine_tune" | "context_predictor")
}

/// The two ways a [`PyJob`] handle comes to exist: freshly minted by a submit
/// call (`_start_training_proto`, which already has the engine's own
/// [`TrainingJob`] to hand over), or reconstructed for a job this connection
/// never submitted (`PyDatabase::job`, the attach-by-id path — the embedded
/// peer of the remote client's `RemoteJob`, which can always be built
/// straight from a `job_id` because every one of its verbs re-fetches state
/// over the wire per call).
///
/// `TrainingJob::new` is `pub(crate)` to `jammi-ai` — deliberately: a
/// `TrainingJob` is minted alongside the catalog row it names, by the same
/// submit call, so its `model_id` is always the value that call itself
/// resolved, never re-derived. The `Attached` arm therefore does not attempt
/// to construct a `jammi-ai` `TrainingJob` at all; it carries exactly the
/// `job_id` / `kind` / resolved `output_model_id` this binding needs and
/// drives `status()` / `wait()` / `progress()` / `cancel()` / `metrics()` /
/// `acceleration_report()` straight off the catalog record, through the same
/// public `Catalog::get_job` read every arm uses.
enum JobState {
    Submitted {
        job: TrainingJob,
        kind: String,
    },
    Attached {
        job_id: String,
        kind: String,
        output_model_id: String,
    },
}

/// Python `Job` handle: the generalised, kind-agnostic peer of the wire's
/// `JobService` (PLAN-C §4). Every job this binding can mint or attach today
/// is one of the three training kinds (`_start_training_proto` never submits
/// a compute spec), but `job(id)`/`list_jobs()` can resolve ANY row the
/// generalised `jobs` table holds — including a compute-kind row created
/// internally by `InferenceSession::run_now`/`enqueue` — so every read here
/// (`kind`, `progress`, `cancel`, `wait`) is written to work for either.
#[pyclass(name = "Job")]
pub struct PyJob {
    inner: JobState,
    runtime: Arc<tokio::runtime::Runtime>,
    /// The session this handle reads through. For the `Submitted` arm this is
    /// also the session `inner` was submitted against; for `Attached` it is
    /// simply the connection `job(job_id)` was called on. Held directly
    /// (rather than reached through `inner`) because `TrainingJob`'s own
    /// `catalog` field is private to `jammi-ai` — this binding carries its
    /// own handle rather than growing that crate's public surface for a read
    /// only this binding needs.
    session: Arc<InferenceSession>,
}

impl PyJob {
    pub fn new(
        job: TrainingJob,
        kind: String,
        runtime: Arc<tokio::runtime::Runtime>,
        session: Arc<InferenceSession>,
    ) -> Self {
        Self {
            inner: JobState::Submitted { job, kind },
            runtime,
            session,
        }
    }

    /// Attach to an existing job by id — a handle for a job this connection
    /// never submitted, reconstructed from the catalog row alone (the embedded
    /// peer of the remote client's always-attach-by-id `RemoteJob`).
    ///
    /// A nonexistent `job_id` raises the SAME typed not-found the catalog read
    /// (`Catalog::get_job`) itself produces — there is no separate
    /// existence check to drift from it.
    ///
    /// `output_model_id` is resolved eagerly here (once, synchronously)
    /// rather than on every access, matching how the `Submitted` arm already
    /// carries it as a plain field: empty for a compute-kind row (mirroring
    /// the wire's `JobStatusResponse.output_model_id` convention — see
    /// [`is_training_kind`]); for a training kind, the catalog's own
    /// `output_model_id` once the job has completed and stamped it, and
    /// before that the same deterministic re-derivation a submit call would
    /// have handed back — see [`resolve_attach_model_id`].
    pub fn attach(
        job_id: String,
        runtime: Arc<tokio::runtime::Runtime>,
        session: Arc<InferenceSession>,
    ) -> PyResult<Self> {
        let record = runtime
            .block_on(session.catalog().get_job(&job_id))
            .map_err(to_pyerr)?;
        let kind = record.kind.clone();
        let output_model_id = if is_training_kind(&kind) {
            resolve_attach_model_id(&job_id, &record)?
        } else {
            String::new()
        };
        Ok(Self {
            inner: JobState::Attached {
                job_id,
                kind,
                output_model_id,
            },
            runtime,
            session,
        })
    }

    fn job_id_str(&self) -> &str {
        match &self.inner {
            JobState::Submitted { job, .. } => &job.job_id,
            JobState::Attached { job_id, .. } => job_id,
        }
    }
}

#[pymethods]
impl PyJob {
    /// The unique job ID.
    #[getter]
    fn job_id(&self) -> &str {
        self.job_id_str()
    }

    /// The `jobs.kind` tag this job submitted under (e.g. `"fine_tune"`).
    #[getter]
    fn kind(&self) -> &str {
        match &self.inner {
            JobState::Submitted { kind, .. } => kind,
            JobState::Attached { kind, .. } => kind,
        }
    }

    /// The output model ID. For the two LoRA fine-tune kinds this is the
    /// deterministic `jammi:fine-tuned:{job_id}` id, resolved at every
    /// lifecycle state (stamped at submission, not only after completion);
    /// the context-predictor kind's is the caller-chosen id. Empty for a
    /// compute-kind job — no compute kind ever registers a model.
    #[getter]
    fn output_model_id(&self) -> &str {
        match &self.inner {
            JobState::Submitted { job, .. } => job.model_id(),
            JobState::Attached {
                output_model_id, ..
            } => output_model_id,
        }
    }

    /// Current status from the catalog.
    fn status(&self) -> PyResult<String> {
        match &self.inner {
            JobState::Submitted { job, .. } => {
                self.runtime.block_on(job.status()).map_err(to_pyerr)
            }
            JobState::Attached { job_id, .. } => {
                let record = self
                    .runtime
                    .block_on(self.session.catalog().get_job(job_id))
                    .map_err(to_pyerr)?;
                Ok(record.status)
            }
        }
    }

    /// This job's progress, as recorded at the executor's checkpoint
    /// boundaries: `{"rows_done", "rows_total", "phase"}`. `rows_done` /
    /// `rows_total` are `None` until the executor has recorded any (a legal
    /// `0` is distinguished from "not recorded yet"); `phase` is the empty
    /// string until the producer names one. Mirrors the wire's `JobProgress`
    /// field-for-field, so a caller reads the same shape on both transports.
    fn progress(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let record = self
            .runtime
            .block_on(self.session.catalog().get_job(self.job_id_str()))
            .map_err(to_pyerr)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("rows_done", record.progress_rows_done)?;
        dict.set_item("rows_total", record.progress_rows_total)?;
        dict.set_item("phase", record.progress_phase.unwrap_or_default())?;
        Ok(dict.into_any().unbind())
    }

    /// Request cancellation. `True` means the request is recorded on a still
    /// non-terminal row and the executor will honour it at its next
    /// checkpoint boundary — the job then lands `failed` with a
    /// `jammi.errors.TrainingError` message, and [`Self::wait`] surfaces it.
    /// `False` when the job is already terminal or absent.
    fn cancel(&self) -> PyResult<bool> {
        self.runtime
            .block_on(self.session.catalog().cancel_request(self.job_id_str()))
            .map_err(to_pyerr)
    }

    /// Block until the job reaches a terminal state, returning the tagged
    /// terminal result as a dict: `{"kind": "model", "model_id",
    /// "artifact_path", "metrics"}` for a training kind (`metrics` is the raw
    /// JSON text of the run-summary blob, or `None` when the run recorded
    /// none — read [`Self::metrics`] for the parsed form), or `{"kind":
    /// "table", "table", "cache_outcome"}` for a compute kind. Raises
    /// `jammi.errors.TrainingError` on a failed job, carrying the executor's
    /// recorded message.
    ///
    /// Reimplemented here (rather than delegating to
    /// `jammi_ai::jobs::JobHandle::wait`, whose constructor is `pub(crate)`
    /// to `jammi-ai`) so ONE code path serves both the `Submitted` and
    /// `Attached` arms identically, and so the terminal payload — not merely
    /// `()` — is available to return: mirrors that handle's polling cadence
    /// and terminal classification exactly.
    fn wait(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let job_id = self.job_id_str().to_string();
        let result = self
            .runtime
            .block_on(wait_for_result(&self.session, &job_id))
            .map_err(to_pyerr)?;
        serializable_to_pydict(py, &result)
    }

    /// Run metrics recorded for this job, as a dict.
    ///
    /// This is exactly what the catalog's `jobs.result` column's nested
    /// `metrics` field carries — the run-summary blob the trainer hands the
    /// worker at completion (`final_loss`, `early_stopping_metric`,
    /// `total_steps`, `started_at`, `completed_at`), or the `error` column a
    /// failed attempt records instead. Returns `{}` for a job that has not
    /// yet recorded any metrics (e.g. still queued or running before its
    /// first stamp — the column is absent). Raises `jammi.errors.BackendError`
    /// if the column IS present but fails to parse as JSON — a catalog
    /// data-integrity fault, never silently folded into the absent `{}` case
    /// (matches the remote transport's `metrics()`).
    ///
    /// Per-epoch train/val loss curves ARE part of this surface (issue #441):
    /// the trainer accumulates `(epoch, avg_train_loss)` / `(epoch,
    /// avg_val_loss)` across `TrainingLoop::run`
    /// (`crates/jammi-ai/src/fine_tune/trainer.rs`) and folds them into the
    /// returned metrics JSON as the `train_loss_curve` / `val_loss_curve`
    /// arrays, so this dict carries them exactly like every other recorded
    /// metric.
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let record = self
            .runtime
            .block_on(self.session.catalog().get_job(self.job_id_str()))
            .map_err(to_pyerr)?;
        // The generalised `jobs` schema (migration 029, C1b) has no dedicated
        // `metrics` column — the raw metrics JSON is nested inside the tagged
        // `result` payload (`jammi_ai::jobs::JobResult::Model.metrics`).
        let metrics_raw = record
            .result
            .as_deref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .and_then(|v| {
                v.get("metrics")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            });
        let value: serde_json::Value = match metrics_raw.as_deref() {
            // Absent — the job has not yet recorded any metrics. `{}`.
            None => serde_json::json!({}),
            // Present — must parse. A present-but-unparseable blob is a catalog
            // data-integrity fault, distinct from "no metrics yet" — surfaced
            // LOUDLY (never silently folded into the absent `{}` case) so it
            // cannot be mistaken for a job that simply hasn't reported yet.
            // Matches the remote transport's `metrics()`, which raises rather
            // than returning `{}` for the same malformed-but-present state.
            Some(raw) => serde_json::from_str(raw).map_err(|parse_err| {
                to_pyerr(JammiError::Catalog(format!(
                    "job {}: metrics blob failed to parse as JSON: {parse_err}",
                    self.job_id_str(),
                )))
            })?,
        };
        serializable_to_pydict(py, &value)
    }

    /// This job's per-attempt acceleration determination (esc-075), as a
    /// dict, or `None`.
    ///
    /// This is exactly what the catalog's `jobs.acceleration_report` column
    /// carries, decoded the same way `metrics()` decodes its column — but
    /// preserving that column's own two-state contract rather than
    /// `metrics()`'s "absent means `{}`" default: SQL `NULL` — a row written
    /// before migration 026, or one this code never touched — maps to Python
    /// `None`, an honest absence of information, never silently coerced to
    /// `{}` or read as any particular acceleration state. A present value
    /// decodes to a dict whose `"state"` is one of four values: `"pending"` —
    /// the submission-time marker, meaning the job exists and no claimant has
    /// computed a determination yet — or one of the three determination
    /// outcomes `"determined"`, `"not_applicable"`, and `"undetermined"`, the
    /// last of which always carries a `"reason"` string. `"pending"` is a
    /// transient marker, not a resting state: a job that reaches a terminal
    /// status without a determination has its marker retired to
    /// `"undetermined"` with a reason naming the edge that retired it, and a
    /// requeued job is reset to `"pending"` for its next attempt — so a read
    /// can legitimately land on any of the four. Everything beside `"state"`
    /// is the payload producer's to define; this binding decodes the blob
    /// as-is and never inspects it.
    ///
    /// Raises `jammi.errors.BackendError` if the column IS present but fails
    /// to parse as JSON — a catalog data-integrity fault, matching
    /// `metrics()`'s same-shaped guard.
    fn acceleration_report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let record = self
            .runtime
            .block_on(self.session.catalog().get_job(self.job_id_str()))
            .map_err(to_pyerr)?;
        match record.acceleration_report.as_deref() {
            // NULL — honest absence, never coerced into a state claim.
            None => Ok(py.None()),
            // Present — must parse. A present-but-unparseable blob is a
            // catalog data-integrity fault, surfaced LOUDLY rather than
            // folded into the `None` case (matches `metrics()`'s treatment
            // of its own malformed-but-present blob).
            Some(raw) => {
                let value: serde_json::Value = serde_json::from_str(raw).map_err(|parse_err| {
                    to_pyerr(JammiError::Catalog(format!(
                        "job {}: acceleration_report blob failed to parse as JSON: \
                         {parse_err}",
                        self.job_id_str(),
                    )))
                })?;
                serializable_to_pydict(py, &value)
            }
        }
    }
}

/// Poll `catalog.get_job(job_id)` until it reaches a terminal state,
/// returning the parsed [`JobResult`] on success. Mirrors
/// `jammi_ai::jobs::JobHandle::wait` exactly (same 100ms cadence, same
/// terminal classification, same `JammiError::FineTune` failure shape) —
/// reimplemented here because that handle's constructor is `pub(crate)` to
/// `jammi-ai`, and this binding must serve BOTH a freshly-submitted
/// (`Submitted`) and an attached-by-id (`Attached`) job through ONE code
/// path.
async fn wait_for_result(
    session: &InferenceSession,
    job_id: &str,
) -> jammi_db::error::Result<JobResult> {
    loop {
        let record = session.catalog().get_job(job_id).await?;
        let status: JobStatus = record
            .status
            .parse()
            .map_err(|e| JammiError::FineTune(format!("{e}")))?;
        match status {
            JobStatus::Completed => {
                let raw = record.result.ok_or_else(|| {
                    JammiError::Catalog(format!("job '{job_id}' completed with no recorded result"))
                })?;
                return serde_json::from_str(&raw).map_err(|e| {
                    JammiError::Catalog(format!(
                        "job '{job_id}' recorded an unparseable result: {e}"
                    ))
                });
            }
            JobStatus::Failed => {
                let msg = record.error.unwrap_or_else(|| "job failed".into());
                return Err(JammiError::FineTune(msg));
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

/// Resolve the output model id an attached handle reports, from the catalog
/// row alone — no in-memory carryover from a submit call, because attach never
/// had one.
///
/// A thin adapter over
/// [`jammi_ai::fine_tune::training_job::resolve_model_id`], which owns the
/// rule: the stamped `output_model_id` once the job has completed, and before
/// then the same re-derivation the submit path applies (the two LoRA
/// fine-tune kinds mint the deterministic
/// [`jammi_ai::fine_tune::training_job::fine_tuned_model_id`]; the
/// context-predictor kind's id is caller-chosen and read out of the persisted
/// `training_spec`). This function adds ONLY the conversion into the Python
/// error taxonomy — no second copy of the naming rule, so this binding cannot
/// drift from the `JobStatus` handler that answers the REMOTE peer of this
/// very handle. That one shared call is what makes
/// `EmbeddedBackend.job(id).output_model_id` and
/// `RemoteDatabase.job(id).output_model_id` equal at every lifecycle state —
/// equal by construction, not by two implementations that happen to agree.
fn resolve_attach_model_id(job_id: &str, record: &JobRecord) -> PyResult<String> {
    jammi_ai::fine_tune::training_job::resolve_model_id(job_id, record).map_err(to_pyerr)
}
