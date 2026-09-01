use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::status::TrainingJobStatus;
use super::Catalog;
use crate::error::{JammiError, Result};
use crate::tenant::TenantId;

/// A row from the `training_jobs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobRecord {
    pub job_id: String,
    pub base_model_id: String,
    pub output_model_id: Option<String>,
    pub training_source: String,
    pub loss_type: String,
    pub hyperparams: String,
    pub status: String,
    pub metrics: Option<String>,
    pub error_message: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    /// Training-job kind discriminator (`'fine_tune'` for the contrastive
    /// adapter path).
    pub kind: String,
    /// Id of the worker holding the lease, or `None` when queued/unclaimed.
    pub claimed_by: Option<String>,
    /// Lease deadline as a canonical UTC timestamp, or `None` when not leased.
    pub lease_expires_at: Option<String>,
    /// Number of times the job has been claimed.
    pub attempts: u32,
    /// The tenant that owns the job, or `None` for unscoped rows. Carried so a
    /// worker that claims across tenants can re-scope subsequent work.
    pub tenant_id: Option<TenantId>,
    /// The self-describing training specification as JSON, or `None` for a row
    /// written without one. A worker deserialises this to reconstruct the run on
    /// a fresh process — the catalog stores it opaquely (the typed shape lives in
    /// the engine crate that produces and consumes it).
    pub training_spec: Option<String>,
    /// The job's per-attempt acceleration determination (esc-075), as an
    /// opaque, self-describing JSON payload whose vocabulary the payload's
    /// *producer* owns — matching `training_spec`'s and `metrics`'
    /// schema-at-the-producer deferral, not a closed enum pinned here
    /// (migration 026). `Option`'s two Rust-level states map to only the
    /// mechanical part of the contract:
    ///
    ///   - `None` (SQL `NULL`) — unknown: a row written before migration 026,
    ///     or one this code never touched. Never read as "accelerated" or
    ///     "eager" — it is an honest absence of information, not a claim.
    ///   - `Some(json)` — a payload landed. The catalog itself writes exactly
    ///     three payload shapes, all of them PENDING-valued transitions (see
    ///     the lifecycle below): `{"state":"pending"}` at `INSERT` time, and
    ///     the two `"undetermined"` retirements of that marker on the terminal
    ///     paths. Every other payload — commonly `{"state":"determined", ...}`
    ///     from the claiming worker via
    ///     [`Catalog::record_acceleration_report`], but not limited to that
    ///     one shape (a non-fine-tune job kind or a pre-device-resolution
    ///     failure path may record a different `"state"`) — is the
    ///     producer's to define; the catalog stores it byte-for-byte and never
    ///     inspects, validates, or enumerates it.
    ///
    /// # The pending marker's lifecycle
    ///
    /// `{"state":"pending"}` asserts one sentence: *the job exists and no
    /// claimant has computed a determination YET*. "Yet" is the load-bearing
    /// word — it is only true while the job can still reach the probe. The
    /// moment the row goes TERMINAL, `pending` describes a state that will
    /// never resolve, which is exactly the state this contract has no reading
    /// for. So the catalog retires it at the ONE edge every terminal write
    /// passes through, never at N caller sites (a caller can be skipped, or a
    /// new terminal caller added without the marker; the reclaim paths have no
    /// live caller at all — the claimant is dead):
    ///
    ///   - [`Catalog::fail_training_job`] — `pending` → `{"state":
    ///     "undetermined","reason":"failed_before_probe"}`, in the SAME
    ///     lease-guarded UPDATE that stamps `failed`.
    ///   - [`Catalog::reclaim_expired_training_jobs`], attempts-exhausted arm
    ///     — `pending` → `{"state":"undetermined","reason":
    ///     "lease_expired_attempts_exhausted"}`, in the SAME UPDATE that
    ///     stamps `failed`.
    ///   - [`Catalog::reclaim_expired_training_jobs`], requeue arm — the row
    ///     returns to `queued` for a NEW attempt that will re-probe, so the
    ///     column is RESET to `{"state":"pending"}`. The dead attempt's report
    ///     described the hardware/config THAT attempt saw; leaving it on a
    ///     queued row would attribute it to an attempt that has not started.
    ///
    /// Both terminal rewrites are strictly `pending`-valued: a payload that is
    /// already `determined` / `not_applicable` / `undetermined` is the last
    /// true thing known about the job and is preserved byte-for-byte, and a
    /// legacy SQL `NULL` stays `NULL` (three-valued `NULL = '…'` is not true,
    /// so the CASE falls through) — "unknown" is never fabricated into a state.
    /// No non-terminal write (claim, heartbeat,
    /// [`Catalog::mark_training_running`], [`Catalog::finalize_training_job`])
    /// touches the column at all.
    pub acceleration_report: Option<String>,
}

const SELECT_COLS: &str = "job_id, base_model_id, output_model_id, training_source, loss_type, \
     hyperparams, status, metrics, created_at, kind, claimed_by, lease_expires_at, attempts, \
     tenant_id, training_spec, acceleration_report";

/// The explicit submission-time marker [`Catalog::create_training_job`] writes
/// into `acceleration_report`: the job exists but no claimant has yet computed
/// an acceleration determination for it. Distinct from SQL `NULL` (a
/// pre-migration-026 row this code never touched) — see
/// [`TrainingJobRecord::acceleration_report`]'s producer-owned-payload
/// contract; this is the one payload shape the catalog itself writes.
const ACCELERATION_REPORT_PENDING: &str = r#"{"state":"pending"}"#;

/// The marker [`Catalog::fail_training_job`] substitutes for a still-`pending`
/// report, in the same lease-guarded UPDATE that stamps `failed`: the attempt
/// died between the claim and the acceleration probe, so no determination
/// exists and none ever will for this attempt. Byte-for-byte vocabulary — the
/// embedded and remote surfaces read these exact bytes; see
/// [`TrainingJobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_FAILED_BEFORE_PROBE: &str =
    r#"{"state":"undetermined","reason":"failed_before_probe"}"#;

/// The marker [`Catalog::reclaim_expired_training_jobs`]'s attempts-exhausted
/// arm substitutes for a still-`pending` report: the claimant is gone (its
/// lease expired) and the job has no attempts left, so the row is terminal with
/// no determination — and, unlike the `fail` path, with no live worker that
/// could ever compensate. Byte-for-byte vocabulary; see
/// [`TrainingJobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED: &str =
    r#"{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}"#;

/// A backend-portable `SET acceleration_report = …` clause that retires the
/// submission-time pending marker and leaves every other payload alone:
/// `$pending_param` is the pending marker to match, `$terminal_param` the
/// marker to write in its place. Rendered into the terminal UPDATEs so the
/// rewrite rides the SAME statement as the status transition — never a
/// read-then-write outside the transaction, and never a second UPDATE whose
/// predicate could drift from the first's.
///
/// The three-valued comparison is the point: a legacy SQL `NULL`
/// (pre-migration-026 "unknown") is neither equal nor unequal to the marker, so
/// `CASE WHEN NULL = '…'` is not true and the `ELSE` arm preserves the `NULL`.
/// `CASE`/`WHEN`/`ELSE` is core SQL — identical on SQLite and Postgres (B4), so
/// no dialect branch is needed the way [`Catalog::claim_next_training_job`]'s
/// `FOR UPDATE SKIP LOCKED` needs one.
fn retire_pending_report_clause(pending_param: u8, terminal_param: u8) -> String {
    format!(
        "acceleration_report = CASE WHEN acceleration_report = ${pending_param} \
         THEN ${terminal_param} ELSE acceleration_report END"
    )
}

/// Format leases write into `lease_expires_at`. Lexicographic ordering of two
/// timestamps in this fixed-width UTC form matches chronological ordering, so
/// the SQL `lease_expires_at < $now` comparison is correct on both backends
/// without dialect-specific interval arithmetic.
const LEASE_TS_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.6fZ";

/// `now`, formatted for an engine-clock lease comparison or stamp.
fn lease_now() -> String {
    chrono::Utc::now().format(LEASE_TS_FORMAT).to_string()
}

/// `now + lease`, formatted as a lease deadline.
fn lease_deadline(lease: Duration) -> String {
    let expiry =
        chrono::Utc::now() + chrono::Duration::from_std(lease).unwrap_or(chrono::Duration::MAX);
    expiry.format(LEASE_TS_FORMAT).to_string()
}

fn parse_row(row: &Row<'_>) -> std::result::Result<TrainingJobRecord, BackendError> {
    let metrics_raw: Option<String> = row.try_get("metrics")?;
    let error_message = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["error_message"].as_str().map(String::from))
    });
    let started_at = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["started_at"].as_str().map(String::from))
    });
    let completed_at = metrics_raw.as_deref().and_then(|m| {
        serde_json::from_str::<serde_json::Value>(m)
            .ok()
            .and_then(|v| v["completed_at"].as_str().map(String::from))
    });
    let tenant_id = row
        .try_get::<String>("tenant_id")?
        .map(|s| {
            s.parse::<TenantId>()
                .map_err(|e| BackendError::TypeConversion {
                    column: "tenant_id".to_string(),
                    detail: e.to_string(),
                })
        })
        .transpose()?;

    Ok(TrainingJobRecord {
        job_id: row.get("job_id")?,
        base_model_id: row.get("base_model_id")?,
        output_model_id: row.try_get("output_model_id")?,
        training_source: row.get("training_source")?,
        loss_type: row.get("loss_type")?,
        hyperparams: row.get("hyperparams")?,
        status: row.get("status")?,
        metrics: metrics_raw,
        error_message,
        created_at: row.get("created_at")?,
        started_at,
        completed_at,
        kind: row.get("kind")?,
        claimed_by: row.try_get("claimed_by")?,
        lease_expires_at: row.try_get("lease_expires_at")?,
        attempts: row.get::<i32>("attempts")? as u32,
        tenant_id,
        training_spec: row.try_get("training_spec")?,
        acceleration_report: row.try_get("acceleration_report")?,
    })
}

/// Input parameters for [`Catalog::create_training_job`]. Grouped into one
/// struct (the `RegisterModelParams` pattern) so the call site names each field
/// and the insert surface has one place to grow.
#[derive(Debug, Clone)]
pub struct CreateTrainingJobParams<'a> {
    /// Unique job id.
    pub job_id: &'a str,
    /// Base-model catalog PK the `base_model_id` FK references.
    pub base_model_id: &'a str,
    /// The source the run reads from (recorded for provenance).
    pub training_source: &'a str,
    /// Human-readable objective tag.
    pub loss_type: &'a str,
    /// Optimisation hyperparameters as JSON.
    pub hyperparams: &'a str,
    /// The verb that produced the job — the worker dispatches on it.
    pub kind: &'a str,
    /// The self-contained JSON specification a worker reconstructs the run from
    /// on a fresh process. Stored opaquely — the typed shape lives in the engine
    /// crate that produces and consumes it.
    pub training_spec: &'a str,
}

/// One retained epoch checkpoint's catalog row, inserted inside the same
/// lease-guarded finalize transaction that commits the output model's
/// served path (unit 348, CONTRACT item 3/4).
///
/// Lifecycle: a row is inserted **only** when the finalize CAS this call
/// rides on actually wins — the insert sits inside the same `if job_updated
/// == 1` guard as the output model's `artifact_path` write. A zombie or
/// lease-lost worker's `finalize_training_job` therefore matches zero job
/// rows and never reaches this insert at all, so it can never register an
/// epoch-checkpoint row for an attempt that lost the race — mirroring the
/// output model's own "served path is written by exactly one writer"
/// contract. Every retained epoch's bytes are durable on the object store
/// under its own attempt-unique prefix (K7) regardless of who wins.
///
/// The now-true residual lattice (round-2 audit, F1/F2) — a residual, not a
/// solved problem, so it is documented precisely rather than glossed as one
/// case:
///   - A **live loser**: the worker's PROCESS survives to reach exactly ONE
///     terminating arm per attempt — `TrainingWorker::run_claimed_job`'s
///     `Cancelled` and `Failed` arms (which also catch a caught-panic or
///     `spawn_blocking` join-error from `TrainingWorker::train_fine_tune`;
///     those do NOT sweep themselves, to avoid a double sweep of the
///     identical range — see `TrainingWorker::gc_epoch_checkpoints`'s doc),
///     and `TrainingWorker::publish_and_finalize`'s publish-failure,
///     `register_model`-failure, `Ok(false)`, and `Err` arms — EVERY one of
///     which calls
///     the same `TrainingWorker::gc_epoch_checkpoints` derived sweep (one
///     reclaim mechanism, not a vec-based one for the lucky arm and
///     something else for the rest). The sweep DERIVES each candidate
///     prefix from the attempt's identity and the run's configured epoch
///     bound rather than reading an in-memory vec — most of these arms never
///     built a `TrainedArtifact` at all, so no such vec exists to read (this
///     is exactly the class of gap a vec-based reclaim could not close by
///     construction). It runs alongside the top-level artifact prefix's own
///     best-effort delete — the nested `checkpoints/epoch_{N}/` prefixes
///     carry their OWN separate manifests, so the top-level prefix's delete
///     alone never reaches them. `keep_last_n_checkpoints` absent (the
///     default) makes this sweep a bound-`0` no-op for every legacy job —
///     zero store requests, not merely zero registered rows.
///   - A **winning attempt with a persistently-failed mid-run prune**: no
///     longer a leak (round-2 F2's fix). `TrainingLoop::save_epoch_checkpoint`'s
///     retention loop leaves a checkpoint whose delete failed in its tracked
///     vector (retrying at each subsequent epoch boundary) rather than
///     losing track of it; `TrainingWorker::publish_and_finalize`'s `Ok(true)`
///     (winner) arm trims the FINAL vector to the true trailing retention
///     window before registering (so a straggler entry never inflates the
///     registered row count past the configured cap) and then sweeps exactly
///     those excluded, still-durable stale entries — reclaiming a
///     persistently-failed prune at termination even if every mid-run retry
///     failed.
///   - A **truly crashed** attempt (the process dies before ever reaching
///     ANY of the terminating arms above, including the winner arm): nothing
///     reclaims either its top-level artifact prefix OR its epoch-checkpoint
///     prefixes — the existing top-level GC is exactly as unreachable in
///     this case as the epoch-checkpoint one, a pre-existing limitation of
///     the "GC on the losing branch of a LIVE process" design this unit does
///     not change. These bytes are durable-but-permanently-unregistered, the
///     expected residual — the ONE case genuinely left, now that F2 closed
///     the winner-path leak.
#[derive(Debug, Clone)]
pub struct EpochCheckpointRow<'a> {
    /// Distinct catalog name: `jammi:fine-tuned:{job_id}:epoch_{N}` — never
    /// an additional VERSION of the output model's name (that would let a
    /// later finalize's version-scoped CAS clobber a different row's path;
    /// see [`Catalog::finalize_training_job`]'s version predicate).
    pub model_id: &'a str,
    /// Catalog `model_type`, mirroring the output model row's convention
    /// (e.g. `"fine-tuned"`).
    pub model_type: &'a str,
    /// Same task the output model row carries.
    pub task: crate::model_task::ModelTask,
    /// Same base-model lineage the output model row carries.
    pub base_model_id: Option<&'a str>,
    /// The attempt-unique object-store prefix these bytes were published
    /// under (`{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{N}/`) — the
    /// bytes are already complete (manifest-last) by the time this row is
    /// built, unlike the output model row, which registers with no path and
    /// gains it only from the CAS below.
    pub artifact_path: &'a str,
}

/// Input parameters for [`Catalog::finalize_training_job`] — grouped (the
/// `RegisterModelParams`/`CreateTrainingJobParams` pattern) so the lease-CAS
/// call site names each field, and so a future addition to the finalize
/// surface grows this struct rather than clippy's `too_many_arguments` limit.
pub struct FinalizeTrainingJobParams<'a> {
    /// The job id to finalize.
    pub job_id: &'a str,
    /// The lease holder's id (`claimed_by`) — the CAS matches only while this
    /// worker still holds the lease.
    pub worker_id: &'a str,
    /// The output model's catalog NAME (`training_jobs.output_model_id`).
    pub output_model_id: &'a str,
    /// The output model's catalog VERSION — together with `output_model_id`
    /// and the tenant, the exact row the served-path `UPDATE` must touch and
    /// no other (B5, unit 348).
    pub output_model_version: i32,
    /// The object-store prefix this worker published the output artifact
    /// under — committed as the model row's `artifact_path`.
    pub artifact_path: &'a str,
    /// Run-metrics JSON to record on the job row, or `None`.
    pub metrics: Option<&'a str>,
    /// Every RETAINED epoch checkpoint to register alongside the output
    /// model (unit 348) — empty for a training kind with no per-epoch
    /// checkpointing.
    pub epoch_checkpoints: &'a [EpochCheckpointRow<'a>],
}

impl Catalog {
    /// Create a new training job record with status = 'queued'. Tenant
    /// bound + asserted (SPEC-03 §7).
    pub async fn create_training_job(&self, params: CreateTrainingJobParams<'_>) -> Result<()> {
        let job_id = params.job_id.to_string();
        let base_model_id = params.base_model_id.to_string();
        let training_source = params.training_source.to_string();
        let loss_type = params.loss_type.to_string();
        let hyperparams = params.hyperparams.to_string();
        let kind = params.kind.to_string();
        let training_spec = params.training_spec.to_string();
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "training_jobs")?;
                    tx.execute(
                        "INSERT INTO training_jobs \
                         (job_id, base_model_id, training_source, loss_type, hyperparams, status, \
                          kind, training_spec, tenant_id, acceleration_report) \
                         VALUES ($1, $2, $3, $4, $5, 'queued', $6, $7, $8, $9)",
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(base_model_id),
                            SqlValue::TextOwned(training_source),
                            SqlValue::TextOwned(loss_type),
                            SqlValue::TextOwned(hyperparams),
                            SqlValue::TextOwned(kind),
                            SqlValue::TextOwned(training_spec),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get a training job by ID. Tenant-filtered.
    pub async fn get_training_job(&self, job_id: &str) -> Result<TrainingJobRecord> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM training_jobs WHERE job_id = $1 \
               AND (tenant_id = $2 OR tenant_id IS NULL)"
        );
        let id = job_id.to_string();
        let id_for_err = id.clone();
        let tenant = self.current_tenant();
        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            &sql,
                            &[
                                SqlValue::TextOwned(id),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| JammiError::Catalog(format!("Training job '{id_for_err}' not found")))
    }

    /// Finalize a training job the caller still owns, as a single lease-guarded
    /// compare-and-set that also commits the output model's served artifact path
    /// and registers every surviving epoch-checkpoint row (unit 348). In one
    /// transaction it flips the job row to `completed`, writes
    /// `output_model_id`, records the run metrics (when `metrics` is `Some`),
    /// and — only if that job-row CAS matched — records `artifact_path` on the
    /// output model's row and inserts one row per `epoch_checkpoints` entry. The
    /// job-row CAS lands **only** while the row is still `running` and
    /// `claimed_by == worker_id`. Returns `true` when the caller held the lease
    /// and is the sole finalizer, `false` when it was not (the lease was lost —
    /// the row is no longer `running`, or another worker reclaimed it).
    ///
    /// `artifact_path` is the object-store prefix this worker published its
    /// artifact under. The model-row update is gated on the job-row CAS matching
    /// (it runs in the same transaction and is skipped when the CAS matched zero
    /// rows), so the finalize CAS is the **sole writer** of the served path: a
    /// loser's finalize matches no job row and therefore writes neither the job
    /// status nor the model's served path, and its orphaned prefix is never the
    /// committed pointer. A `false` return means the caller must not act as the
    /// finalizer; the job is left for [`Self::reclaim_expired_training_jobs`] and
    /// the worker that re-claims it.
    ///
    /// The model row is matched by `name = output_model_id AND version =
    /// output_model_version`, tenant-scoped with the same STRICT predicate
    /// [`Catalog::delete_model`] uses (`tenant_id = $t OR (tenant_id IS NULL AND
    /// $t IS NULL)`) — never the relaxed `OR tenant_id IS NULL` a *read*
    /// resolver uses to also see a global row. Before this predicate the update
    /// matched on `name` alone: any row anywhere sharing that bare name — a
    /// different VERSION, or a DIFFERENT TENANT's row that happens to carry the
    /// same name — would also be clobbered by this worker's `artifact_path`.
    /// Recording the path on the model row (rather than only the job row) keeps
    /// the reload path reading the served pointer straight from `models`,
    /// unchanged.
    ///
    /// The `training_jobs` CAS itself stays NOT tenant-scoped, matching
    /// [`Self::claim_next_training_job`] and [`Self::heartbeat_training_job`]:
    /// the lease identity (`claimed_by`) is the authority there, not the session
    /// tenant, and `job_id` is a global unique PK so no cross-tenant collision is
    /// possible on that row. The `models` predicate above is the one that needed
    /// tenant scoping, because `models.name` is NOT globally unique the way
    /// `job_id` is.
    pub async fn finalize_training_job(
        &self,
        params: FinalizeTrainingJobParams<'_>,
    ) -> Result<bool> {
        let FinalizeTrainingJobParams {
            job_id,
            worker_id,
            output_model_id,
            output_model_version,
            artifact_path,
            metrics,
            epoch_checkpoints,
        } = params;
        let completed = TrainingJobStatus::Completed.to_string();
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let output_model_id = output_model_id.to_string();
        let output_model_version = output_model_version as i64;
        let artifact_path = artifact_path.to_string();
        let metrics = metrics.map(str::to_string);
        let now = lease_now();
        let tenant = self.current_tenant();
        // Owned, 'static-lifetime copies of the epoch rows so the transaction
        // closure (which must be `'static` — see `TxOptions`/`transaction`'s
        // signature) can move them without borrowing `epoch_checkpoints`.
        let epoch_rows: Vec<(String, String, String, Option<String>, String)> = epoch_checkpoints
            .iter()
            .map(|r| {
                (
                    r.model_id.to_string(),
                    r.model_type.to_string(),
                    r.task.as_db_str().to_string(),
                    r.base_model_id.map(str::to_string),
                    r.artifact_path.to_string(),
                )
            })
            .collect();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let job_updated = tx
                        .execute(
                            "UPDATE training_jobs \
                             SET output_model_id = $1, status = $2, \
                                 metrics = COALESCE($3, metrics), updated_at = $4 \
                             WHERE job_id = $5 AND claimed_by = $6 AND status = $7",
                            &[
                                SqlValue::TextOwned(output_model_id.clone()),
                                SqlValue::TextOwned(completed),
                                SqlValue::from(metrics),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(job_id),
                                SqlValue::TextOwned(worker_id),
                                SqlValue::TextOwned(running),
                            ],
                        )
                        .await?;
                    // Commit the served path on the output model's row, and
                    // register every retained epoch-checkpoint row, only when
                    // this worker won the job-row CAS — in the same
                    // transaction, so both are committed atomically with (and
                    // never without) the job's terminal flip. A loser's CAS
                    // matched zero rows and skips this entirely, so it can
                    // neither clobber the served path nor register a row for
                    // bytes a zombie attempt wrote.
                    if job_updated == 1 {
                        tx.assert_tenant_matches(tenant, "models")?;
                        let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));
                        tx.execute(
                            "UPDATE models SET artifact_path = $1, \
                                 updated_at = $2 \
                             WHERE name = $3 AND version = $4 \
                               AND (tenant_id = $5 OR (tenant_id IS NULL AND $5 IS NULL))",
                            &[
                                SqlValue::TextOwned(artifact_path),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(output_model_id),
                                SqlValue::Int(output_model_version),
                                tenant_val.clone(),
                            ],
                        )
                        .await?;

                        for (model_id, model_type, task_db_str, base_model_id, path) in epoch_rows {
                            // Pre-check by NAME ALONE (every version, the
                            // SAME strict tenant predicate the row's own
                            // insert and the output model's UPDATE use) —
                            // never a bare `ON CONFLICT(model_id) DO
                            // NOTHING`, which only catches an EXACT
                            // `(tenant, name, version=1)` PK collision and
                            // silently swallows it with no log. A row
                            // already occupying this NAME at ANY OTHER
                            // version is not a PK collision at all (a
                            // different PK), so a bare INSERT would
                            // SUCCEED — but `get_model`'s `ORDER BY version
                            // DESC` resolves the occupying (higher- or
                            // lower-numbered) row, silently SHADOWING the
                            // checkpoint from every reader (`describe_model`
                            // included) even though its row exists in the
                            // table. Checking by name catches both: the
                            // exact-PK collision AND the version-shadow.
                            //
                            // TOCTOU residual: this SELECT and the INSERT
                            // below are not atomic against a register_model
                            // call EXTERNAL to this transaction landing in
                            // between — with two DIFFERENT outcomes worth
                            // naming rather than assuming away. A race
                            // landing at the SAME (tenant, name, version=1)
                            // this checkpoint would occupy is BOUNDED: the
                            // table's own primary-key uniqueness fails the
                            // INSERT loudly, failing this finalize
                            // transaction (a retried finalize, not a silent
                            // double-write). A race landing at a DIFFERENT
                            // version of the same NAME is the one case this
                            // check cannot close: the INSERT has a distinct
                            // PK, so it succeeds SILENTLY, and the
                            // version-shadow this whole mechanism exists to
                            // prevent can reappear for that one narrow
                            // window — an accepted residual (external
                            // registration racing a finalize CAS in the
                            // sub-transaction interval), not a designed arm.
                            let occupied = tx
                                .query_opt(
                                    "SELECT COUNT(*) AS n FROM models \
                                     WHERE name = $1 \
                                       AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                                    &[SqlValue::TextOwned(model_id.clone()), tenant_val.clone()],
                                    |row| row.get::<i64>("n"),
                                )
                                .await?
                                .unwrap_or(0)
                                > 0;
                            if occupied {
                                // Checkpoints are supplementary — never fail
                                // the job over one name collision. The
                                // checkpoint's bytes stay durable on the
                                // object store (K7) but permanently
                                // unregistered — the SAME residual bucket a
                                // truly-crashed attempt's bytes fall into
                                // (see `EpochCheckpointRow`'s doc), just
                                // reached by a name collision instead of a
                                // dead process.
                                tracing::warn!(
                                    occupied_name = %model_id,
                                    skipped_artifact_path = %path,
                                    "epoch-checkpoint catalog name already occupied by another \
                                     row; skipping registration — the checkpoint's bytes remain \
                                     durable but unregistered"
                                );
                                continue;
                            }

                            let pk = super::model_repo::model_pk(tenant, &model_id, 1);
                            let metadata = serde_json::json!({
                                "base_model_id": base_model_id,
                                "config_json": serde_json::Value::Null,
                            })
                            .to_string();
                            tx.execute(
                                "INSERT INTO models \
                                 (model_id, name, model_type, task, backend, version, \
                                  status, metadata, artifact_path, tenant_id) \
                                 VALUES ($1, $2, $3, $4, 'candle', 1, 'checkpoint', $5, $6, $7)",
                                &[
                                    SqlValue::TextOwned(pk),
                                    SqlValue::TextOwned(model_id),
                                    SqlValue::TextOwned(model_type),
                                    SqlValue::TextOwned(task_db_str),
                                    SqlValue::TextOwned(metadata),
                                    SqlValue::TextOwned(path),
                                    tenant_val.clone(),
                                ],
                            )
                            .await?;
                        }
                    }
                    Ok(job_updated)
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Fail a training job the caller still owns, as a single compare-and-set
    /// gated on lease ownership — the failure peer of [`Self::finalize_training_job`].
    /// Flips the status to `failed` and records `metrics` (the error blob) only
    /// while the row is still `running` and `claimed_by == worker_id`. Returns
    /// `true` when the row was updated and `false` when the lease was lost (the
    /// row is no longer `running`, or another worker reclaimed it).
    ///
    /// Guarding the failure write the same way as the finalize write keeps the
    /// two terminal transitions symmetric: a worker that lost its lease mid-run
    /// cannot stamp `failed` over a job the re-claiming worker is successfully
    /// running (which would otherwise block that worker's finalize). Not
    /// tenant-scoped, matching the other lease-identity operations.
    ///
    /// The same UPDATE retires a still-`pending` `acceleration_report` to
    /// `{"state":"undetermined","reason":"failed_before_probe"}` — the job
    /// died between the claim and the acceleration probe, so "no determination
    /// YET" stops being a true sentence the instant this row goes terminal.
    /// Doing it here rather than at the callers is what makes the property
    /// hold by construction: every `record_failed`-shaped call site funnels
    /// through this one write, so no site can be forgotten and no site added
    /// later can skip it. Strictly `pending`-valued — an already
    /// `determined`/`not_applicable`/`undetermined` payload (including one
    /// carrying a MORE specific reason a worker pre-marked) is preserved
    /// byte-for-byte, and a legacy SQL `NULL` stays `NULL`. See
    /// [`TrainingJobRecord::acceleration_report`]'s lifecycle section and
    /// `retire_pending_report_clause`.
    pub async fn fail_training_job(
        &self,
        job_id: &str,
        worker_id: &str,
        metrics: Option<&str>,
    ) -> Result<bool> {
        let failed = TrainingJobStatus::Failed.to_string();
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let metrics = metrics.map(str::to_string);
        let now = lease_now();
        // Placeholders stay in ascending order of first appearance in the SQL
        // text — SQLite assigns `$N` indices by first appearance, so an
        // out-of-order literal would bind the wrong value.
        let retire = retire_pending_report_clause(3, 4);
        let sql = format!(
            "UPDATE training_jobs \
             SET status = $1, metrics = COALESCE($2, metrics), {retire}, updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8"
        );

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(failed),
                            SqlValue::from(metrics),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::Text(ACCELERATION_REPORT_FAILED_BEFORE_PROBE),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(worker_id),
                            SqlValue::TextOwned(running),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record run-start metrics on a job the caller still owns, gated on lease
    /// ownership — the non-terminal peer of [`Self::finalize_training_job`] and
    /// [`Self::fail_training_job`]. Replaces `metrics` (the run-start blob, e.g.
    /// `started_at`) **only** while the row is still `running` and
    /// `claimed_by == worker_id`; the status is already `running` from the claim,
    /// so this never transitions it — it just stamps metrics under the same lease
    /// guard. Returns `true` when the write landed and `false` when the lease was
    /// lost.
    ///
    /// Every worker write to the job row is lease-guarded for the same reason the
    /// finalize and fail writes are: a worker whose lease was reclaimed mid-run
    /// (a zombie still executing its claimed run) must not be able to touch the
    /// row. Without this guard a zombie's trainer start would stamp `running`
    /// metrics over a job the winner already drove to `completed`, regressing the
    /// terminal status. Not tenant-scoped, matching the other lease-identity
    /// operations.
    pub async fn mark_training_running(
        &self,
        job_id: &str,
        worker_id: &str,
        metrics: Option<&str>,
    ) -> Result<bool> {
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let metrics = metrics.map(str::to_string);
        let now = lease_now();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE training_jobs \
                         SET metrics = COALESCE($1, metrics), updated_at = $2 \
                         WHERE job_id = $3 AND claimed_by = $4 AND status = $5",
                        &[
                            SqlValue::from(metrics),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(worker_id),
                            SqlValue::TextOwned(running),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record the claiming worker's acceleration determination (esc-075) for
    /// this attempt of a job the caller still owns — the report-writing peer of
    /// [`Self::mark_training_running`]. Replaces `acceleration_report`
    /// **only** while the row is still `running`, `claimed_by == worker_id`,
    /// **and `attempts == attempt`**.
    ///
    /// The `attempts` guard is mandatory, not a defensive extra: `claimed_by`
    /// carries `JAMMI_WORKER_ID`, which is deliberately **stable across
    /// process restarts** (worker.rs), so `(job_id, claimed_by, status)` alone
    /// cannot tell "the current claimant, mid-run" from "a zombie of the same
    /// worker identity, from an attempt this job already moved past via
    /// reclaim". A reclaim always bumps `attempts` on re-claim
    /// (`claim_next_training_job`'s `attempts = attempts + 1`), so pinning the
    /// exact attempt closes precisely that gap: a zombie presenting its own
    /// stale `attempt` value matches zero rows and can never overwrite the
    /// current claimant's report, even when it shares the current claimant's
    /// worker id and the row happens to be `running` again under a *later*
    /// attempt.
    ///
    /// Returns `true` when the write landed (the guard matched) and `false`
    /// when it did not — the lease was lost, the job is no longer running, or
    /// the caller's `attempt` is stale. Not tenant-scoped, matching the other
    /// lease-identity operations ([`Self::mark_training_running`],
    /// [`Self::fail_training_job`], [`Self::finalize_training_job`]).
    pub async fn record_acceleration_report(
        &self,
        job_id: &str,
        worker_id: &str,
        attempt: u32,
        report_json: &str,
    ) -> Result<bool> {
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let report_json = report_json.to_string();
        let attempt = attempt as i64;
        let now = lease_now();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE training_jobs \
                         SET acceleration_report = $1, updated_at = $2 \
                         WHERE job_id = $3 AND claimed_by = $4 AND status = $5 AND attempts = $6",
                        &[
                            SqlValue::TextOwned(report_json),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(worker_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempt),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// List training jobs visible to the session tenant, most recent first.
    pub async fn list_training_jobs(&self) -> Result<Vec<TrainingJobRecord>> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM training_jobs \
             WHERE tenant_id = $1 OR tenant_id IS NULL \
             ORDER BY created_at DESC"
        );
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            &sql,
                            &[SqlValue::from(tenant.map(|t| t.to_string()))],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Atomically claim the highest-priority queued training job for
    /// `worker_id`, leasing it for `lease`. The candidate is the queued,
    /// claimable row with the greatest `priority`, ties broken by the oldest
    /// `created_at` — a `claimable = FALSE` row is skipped, not errored, and
    /// stays invisible to the claim until it is flipped back. With every row
    /// at the column defaults (`priority = 0`, `claimable = TRUE`) this
    /// ordering is oldest-first FIFO. On success the row transitions
    /// `queued → running`, stamps `claimed_by = worker_id`, sets
    /// `lease_expires_at = now + lease`, and increments `attempts`; the
    /// claimed record is returned. `Ok(None)` when no job is claimable.
    ///
    /// Deliberately **not** tenant-scoped: a worker serves every tenant's
    /// queue, so this bypasses the `tenant_id` filter the other reads apply.
    /// The returned record carries `tenant_id` so the caller can re-scope the
    /// work it just claimed.
    ///
    /// Atomicity is per-backend. On Postgres the candidate row is selected
    /// `FOR UPDATE SKIP LOCKED`, so concurrent workers each lock a distinct
    /// queued row (or none) and never contend on the same job. On SQLite the
    /// claim is a single `UPDATE … WHERE job_id = (SELECT … LIMIT 1) AND
    /// status = 'queued'` statement: SQLite serialises writers, so of two
    /// concurrent claims exactly one finds the row still `queued` and updates
    /// it while the other matches zero rows. Both backends use `RETURNING` to
    /// read back the claimed row in the same statement.
    pub async fn claim_next_training_job(
        &self,
        worker_id: &str,
        lease: Duration,
    ) -> Result<Option<TrainingJobRecord>> {
        let queued = TrainingJobStatus::Queued.to_string();
        let running = TrainingJobStatus::Running.to_string();
        let worker_id = worker_id.to_string();
        let now = lease_now();
        let deadline = lease_deadline(lease);

        let candidate = match self.backend().backend_kind() {
            super::backend::BackendKind::Postgres => {
                "(SELECT job_id FROM training_jobs WHERE status = $3 AND claimable \
                  ORDER BY priority DESC, created_at LIMIT 1 FOR UPDATE SKIP LOCKED)"
            }
            super::backend::BackendKind::Sqlite => {
                "(SELECT job_id FROM training_jobs WHERE status = $3 AND claimable \
                  ORDER BY priority DESC, created_at LIMIT 1)"
            }
        };
        let sql = format!(
            "UPDATE training_jobs \
             SET status = $1, claimed_by = $2, lease_expires_at = $4, \
                 attempts = attempts + 1, updated_at = $5 \
             WHERE job_id = {candidate} AND status = $3 \
             RETURNING {SELECT_COLS}"
        );

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.query_opt(
                        &sql,
                        &[
                            SqlValue::TextOwned(running),
                            SqlValue::TextOwned(worker_id),
                            SqlValue::TextOwned(queued),
                            SqlValue::TextOwned(deadline),
                            SqlValue::TextOwned(now),
                        ],
                        parse_row,
                    )
                    .await
                })
            })
            .await
            .map_err(Into::into)
    }

    /// Extend the lease on a running job the caller still owns. Renews
    /// `lease_expires_at = now + lease` only when the job is `running` and
    /// `claimed_by == worker_id`, returning `true`. Returns `false` when the
    /// lease was lost — the job is no longer running, or another worker holds
    /// it. Not tenant-scoped, matching [`Self::claim_next_training_job`].
    pub async fn heartbeat_training_job(
        &self,
        job_id: &str,
        worker_id: &str,
        lease: Duration,
    ) -> Result<bool> {
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let now = lease_now();
        let deadline = lease_deadline(lease);

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE training_jobs \
                         SET lease_expires_at = $1, updated_at = $2 \
                         WHERE job_id = $3 AND status = $4 AND claimed_by = $5",
                        &[
                            SqlValue::TextOwned(deadline),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(running),
                            SqlValue::TextOwned(worker_id),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Reclaim running jobs whose lease has expired. For each `running` job
    /// with `lease_expires_at < now`: re-queue it (clearing `claimed_by` and
    /// `lease_expires_at`) when `attempts < max_attempts`, otherwise mark it
    /// `failed` and record the lease-exhaustion reason in `metrics`. Returns
    /// the number of jobs actioned across both branches. Not tenant-scoped —
    /// it sweeps every tenant's expired leases.
    ///
    /// Both arms also move `acceleration_report`, in their OWN `UPDATE` (never
    /// a read-then-write): the claimant whose report this column describes is
    /// gone, and — unlike [`Self::fail_training_job`] — there is no live worker
    /// on the other side that could ever compensate, so the column can only be
    /// correct if this sweep makes it so.
    ///
    ///   - **Requeue** resets the column to `{"state":"pending"}`
    ///     unconditionally. The row goes back to `queued` for a NEW attempt
    ///     that will re-probe, so "no claimant has computed a determination
    ///     yet" is once again exactly true — the same statement
    ///     [`Self::create_training_job`] stamps on a fresh queued row. A
    ///     `determined` payload from the dead attempt described the
    ///     hardware/config THAT attempt saw; surviving onto a queued row it
    ///     would be read as the current attempt's, which is a claim about an
    ///     attempt that has not started. A legacy SQL `NULL` is reset here too
    ///     (rather than preserved as in the terminal arms): `pending` is not a
    ///     fabrication for a row that is, at this instant, genuinely awaiting a
    ///     first determination from its next claimant.
    ///   - **Attempts-exhausted** is terminal, so it retires a still-`pending`
    ///     marker to `{"state":"undetermined","reason":
    ///     "lease_expired_attempts_exhausted"}` under the same strictly
    ///     `pending`-valued `CASE` [`Self::fail_training_job`] uses: a report
    ///     the dead attempt DID record before its lease expired is preserved
    ///     byte-for-byte, and a legacy `NULL` stays `NULL`.
    ///
    /// See [`TrainingJobRecord::acceleration_report`]'s lifecycle section for
    /// the whole transition set in one place.
    pub async fn reclaim_expired_training_jobs(&self, max_attempts: u32) -> Result<usize> {
        let queued = TrainingJobStatus::Queued.to_string();
        let running = TrainingJobStatus::Running.to_string();
        let failed = TrainingJobStatus::Failed.to_string();
        let max_attempts = max_attempts as i64;
        let now = lease_now();
        let failure_metrics = serde_json::json!({
            "error_message": "training job lease expired after exhausting max attempts"
        })
        .to_string();
        // Placeholders stay in ascending order of first appearance in the SQL
        // text (SQLite assigns `$N` indices by first appearance).
        let exhausted_retire = retire_pending_report_clause(3, 4);
        let exhausted_sql = format!(
            "UPDATE training_jobs \
             SET status = $1, metrics = $2, {exhausted_retire}, lease_expires_at = NULL, \
                 updated_at = $5 \
             WHERE status = $6 AND lease_expires_at IS NOT NULL \
               AND lease_expires_at < $7 AND attempts >= $8"
        );

        let actioned = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let requeued = tx
                        .execute(
                            "UPDATE training_jobs \
                             SET status = $1, claimed_by = NULL, lease_expires_at = NULL, \
                                 acceleration_report = $2, updated_at = $3 \
                             WHERE status = $4 AND lease_expires_at IS NOT NULL \
                               AND lease_expires_at < $5 AND attempts < $6",
                            &[
                                SqlValue::TextOwned(queued),
                                SqlValue::Text(ACCELERATION_REPORT_PENDING),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(running.clone()),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::Int(max_attempts),
                            ],
                        )
                        .await?;
                    let exhausted = tx
                        .execute(
                            &exhausted_sql,
                            &[
                                SqlValue::TextOwned(failed),
                                SqlValue::TextOwned(failure_metrics),
                                SqlValue::Text(ACCELERATION_REPORT_PENDING),
                                SqlValue::Text(ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(running),
                                SqlValue::TextOwned(now),
                                SqlValue::Int(max_attempts),
                            ],
                        )
                        .await?;
                    Ok(requeued + exhausted)
                })
            })
            .await?;
        Ok(actioned as usize)
    }
}
