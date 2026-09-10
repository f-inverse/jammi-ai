//! The `jobs` / `instances` / `workers` catalog tables (migration 029):
//! kind-agnostic durable compute jobs, process liveness, and claim-loop
//! membership.
//!
//! `jobs` generalises the training queue's claim/lease/reclaim machinery
//! (the former `training_jobs` table, `catalog::training_repo`, now
//! removed with no shim) to every kind of durable work a worker can claim —
//! training AND the compute verbs that opt into the queue. Every
//! lease-guarded write on a `running` row carries the full attempt guard
//! `WHERE job_id AND claimed_by = $instance AND status = 'running' AND
//! attempts = $n`: `claimed_by` alone cannot distinguish "the current
//! claimant, mid-run" from "a zombie of a prior attempt this job already
//! moved past via reclaim" when an instance id can, in principle, recur
//! (the same defence the removed `training_repo::record_acceleration_report`
//! documented for the training queue).
//!
//! Two execution modes share the one table (`jobs.execution`):
//!
//!   - **`queued`** — claimed by the poll loop
//!     ([`Catalog::claim_next`]), the only rows `claim_next`'s
//!     `WHERE execution = 'queued'` predicate ever selects.
//!   - **`inline`** — claimed exactly once, by id, in the submitting call
//!     itself ([`Catalog::claim_by_id`]); never selected by `claim_next`.
//!     An inline job has no requeue arm: if its executing process dies, the
//!     row is failed once its owning `instances` row is stale or absent
//!     ([`Catalog::reclaim_expired_jobs`]'s liveness arm), never re-claimed.
//!
//! `instances` is the process-liveness table every process upserts at
//! construction and heartbeats; `workers` is the claim-loop-membership table
//! — only a process actually running the claim loop has a row, naming the
//! `kind`s it claims.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::backend::{now_sortable, BackendKind, Row, SqlValue, TxOptions};
use super::lease::{lease_deadline_expr, lease_expired_clause, stale_before_clause};
use super::status::{JobExecution, JobStatus};
use super::Catalog;
use crate::error::{JammiError, Result};
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// A row from the `jobs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub kind: String,
    pub tenant_id: Option<TenantId>,
    pub status: String,
    pub execution: String,
    /// The self-contained, producer-owned tagged JSON specification a worker
    /// reconstructs the run from on a fresh process. Stored opaquely — the
    /// typed shape lives in the engine crate that produces and consumes it.
    pub spec: String,
    /// The name of the `result_tables` row this attempt is (or already has)
    /// materialising into, written inside the SAME transaction as that row's
    /// own INSERT (`Catalog::create_result_table`'s job-CAS). `None` until a
    /// producer that stages through a result table sets it.
    pub partial_result: Option<String>,
    /// The tagged JSON terminal payload a successful attempt writes at
    /// finish. `None` until the job completes.
    pub result: Option<String>,
    /// The terminal failure message. `None` until the job fails.
    pub error: Option<String>,
    pub progress_rows_done: Option<i64>,
    pub progress_rows_total: Option<i64>,
    pub progress_phase: Option<String>,
    /// Set by a cancel request; the executor observes it at a checkpoint
    /// boundary. Never cleared back to `false` once set.
    pub cancel_requested: bool,
    /// The PK-keyed base model a training kind fine-tunes from. `None` for
    /// every compute kind.
    pub model_ref: Option<String>,
    /// The NAME-keyed model a training kind registers on finish. `None`
    /// until (and unless) a training kind finalizes.
    pub output_model_id: Option<String>,
    /// The NAME-keyed, FK-free `ModelSource` string a compute verb resolves
    /// its model against. `None` for a training kind.
    pub model_source: Option<String>,
    /// Id of the instance holding the lease, or `None` while queued/unclaimed.
    pub claimed_by: Option<String>,
    pub attempts: u32,
    /// Lease deadline as a canonical UTC timestamp, or `None` when not leased.
    pub lease_expires_at: Option<String>,
    pub priority: i32,
    /// Migration 024's temporary operator hold: a `false` row is excluded
    /// from `claim_next` without being deleted or given a new status.
    pub claimable: bool,
    /// The job's per-attempt acceleration determination (esc-075), as an
    /// opaque, self-describing JSON payload whose vocabulary the payload's
    /// *producer* owns — matching `spec`'s and `result`'s schema-at-the-
    /// producer deferral, not a closed enum pinned here. Carries forward the
    /// exact contract the removed `training_repo::TrainingJobRecord` field
    /// of the same name documented, generalised from training-only to every
    /// job kind:
    ///
    ///   - `None` (SQL `NULL`) — unknown: a row this code never touched
    ///     (there is no such row on a fresh catalog, since [`Catalog::submit_job`]
    ///     always stamps the pending marker below at insert; a pre-migration-026
    ///     row copied by migration 029 is the one surviving source). Never
    ///     read as "accelerated" or "eager" — it is an honest absence of
    ///     information, not a claim.
    ///   - `Some(json)` — a payload landed. The catalog itself writes exactly
    ///     the pending marker (at [`Catalog::submit_job`] and again on the
    ///     requeue arm of [`Catalog::reclaim_expired_jobs`]) plus the
    ///     `"undetermined"` retirements of that marker — one per terminal
    ///     write. Every other payload — commonly `{"state":"determined", ...}`
    ///     from the claiming instance via
    ///     [`Catalog::record_acceleration_report`], but not limited to that
    ///     one shape — is the producer's to define; the catalog stores it
    ///     byte-for-byte and never inspects, validates, or enumerates it.
    ///
    /// # The pending marker's lifecycle
    ///
    /// `{"state":"pending"}` asserts one sentence: *the job exists and no
    /// claimant has computed a determination YET*. "Yet" is the load-bearing
    /// word — it is only true while the job can still reach the probe. The
    /// moment the row goes TERMINAL, `pending` describes a state that will
    /// never resolve, so the catalog retires it INSIDE each terminal write's
    /// own UPDATE, never at N caller sites: [`Catalog::finish_job`],
    /// [`Catalog::finish_job_with_model`], [`Catalog::fail_job`], and both
    /// terminal arms of [`Catalog::reclaim_expired_jobs`] (lease-exhausted
    /// and inline-executor-dead). The non-terminal requeue arm of
    /// [`Catalog::reclaim_expired_jobs`] instead RESETS the column to
    /// `{"state":"pending"}` unconditionally: the row returns to `queued`
    /// for a NEW attempt that will re-probe, so a `determined` payload from
    /// the dead attempt (which described the hardware/config THAT attempt
    /// saw) must not survive onto a row whose next attempt has not started.
    ///
    /// Every terminal rewrite is strictly `pending`-valued: a payload that
    /// is already `determined` / `not_applicable` / `undetermined` is the
    /// last true thing known about the job and is preserved byte-for-byte,
    /// and a legacy SQL `NULL` stays `NULL` (three-valued `NULL = '…'` is
    /// not true, so the retirement `CASE` falls through) — "unknown" is
    /// never fabricated into a state. Each terminal reason is distinct, so
    /// the retired marker names WHICH edge retired it.
    pub acceleration_report: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl JobRecord {
    /// Whether this row is terminal (`completed` or `failed`) — no further
    /// lease-guarded write is accepted, and the row is eligible for
    /// retention once past `[jobs] retention_days`.
    pub fn is_terminal(&self) -> bool {
        self.status == JobStatus::Completed.to_string()
            || self.status == JobStatus::Failed.to_string()
    }
}

const SELECT_COLS: &str = "job_id, kind, tenant_id, status, execution, spec, partial_result, \
     result, error, progress_rows_done, progress_rows_total, progress_phase, cancel_requested, \
     model_ref, output_model_id, model_source, claimed_by, attempts, lease_expires_at, \
     priority, claimable, acceleration_report, created_at, updated_at";

/// The explicit submission-time marker [`Catalog::submit_job`] writes into
/// `acceleration_report`: the job exists but no claimant has yet computed an
/// acceleration determination for it (esc-075). Distinct from SQL `NULL` (a
/// row this code never touched) — see [`JobRecord::acceleration_report`]'s
/// producer-owned-payload contract. This is the only value the catalog ever
/// writes that is not a retirement of itself: every terminal write matches
/// these exact bytes and replaces them, so `pending` never survives onto a
/// terminal row.
const ACCELERATION_REPORT_PENDING: &str = r#"{"state":"pending"}"#;

/// The marker [`Catalog::fail_job`] substitutes for a still-`pending` report,
/// in the same attempt-guarded UPDATE that stamps `failed`: the attempt died
/// between the claim and the acceleration probe, so no determination exists
/// and none ever will for this attempt. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_FAILED_BEFORE_PROBE: &str =
    r#"{"state":"undetermined","reason":"failed_before_probe"}"#;

/// The marker [`Catalog::reclaim_expired_jobs`]'s queued-execution,
/// attempts-exhausted arm substitutes for a still-`pending` report: the
/// claimant is gone (its lease expired) and the job has no attempts left, so
/// the row is terminal with no determination — and, unlike the `fail` path,
/// with no live worker that could ever compensate. Byte-for-byte vocabulary;
/// see [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED: &str =
    r#"{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}"#;

/// The marker [`Catalog::reclaim_expired_jobs`]'s inline-execution,
/// instance-dead arm substitutes for a still-`pending` report: an inline job
/// has no requeue arm, so once its owning `instances` row is stale or absent
/// the row is failed outright with no determination and no live worker that
/// could ever compensate. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_INLINE_EXECUTOR_DIED: &str =
    r#"{"state":"undetermined","reason":"inline_executor_died"}"#;

/// The marker [`Catalog::finish_job`]/[`Catalog::finish_job_with_model`]
/// substitute for a still-`pending` report, in the same attempt-guarded
/// UPDATE that stamps `completed`: the run SUCCEEDED, but nothing ever
/// recorded a determination for it — the claimant's report-persist step is
/// best-effort (it can lose its lease guard, or hit a catalog error, and the
/// run still proceeds to finish), so success is not evidence that a
/// determination exists. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION: &str =
    r#"{"state":"undetermined","reason":"finalized_without_determination"}"#;

/// A backend-portable `SET acceleration_report = …` clause that retires the
/// submission-time pending marker and leaves every other payload alone:
/// `$pending_param` is the pending marker to match, `$terminal_param` the
/// marker to write in its place. Rendered into the terminal UPDATEs so the
/// rewrite rides the SAME statement as the status transition — never a
/// read-then-write outside the transaction, and never a second UPDATE whose
/// predicate could drift from the first's.
///
/// The three-valued comparison is the point: a legacy SQL `NULL` is neither
/// equal nor unequal to the marker, so `CASE WHEN NULL = '…'` is not true and
/// the `ELSE` arm preserves the `NULL`. `CASE`/`WHEN`/`ELSE` is core SQL —
/// identical on SQLite and Postgres, so no dialect branch is needed.
fn retire_pending_report_clause(pending_param: usize, terminal_param: usize) -> String {
    format!(
        "acceleration_report = CASE WHEN acceleration_report = ${pending_param} \
         THEN ${terminal_param} ELSE acceleration_report END"
    )
}

fn parse_row(row: &Row<'_>) -> std::result::Result<JobRecord, super::backend::BackendError> {
    let tenant_id = row
        .try_get::<String>("tenant_id")?
        .map(|s| {
            s.parse::<TenantId>()
                .map_err(|e| super::backend::BackendError::TypeConversion {
                    column: "tenant_id".to_string(),
                    detail: e.to_string(),
                })
        })
        .transpose()?;
    Ok(JobRecord {
        job_id: row.get("job_id")?,
        kind: row.get("kind")?,
        tenant_id,
        status: row.get("status")?,
        execution: row.get("execution")?,
        spec: row.get("spec")?,
        partial_result: row.try_get("partial_result")?,
        result: row.try_get("result")?,
        error: row.try_get("error")?,
        progress_rows_done: row.try_get("progress_rows_done")?,
        progress_rows_total: row.try_get("progress_rows_total")?,
        progress_phase: row.try_get("progress_phase")?,
        cancel_requested: row.get("cancel_requested")?,
        model_ref: row.try_get("model_ref")?,
        output_model_id: row.try_get("output_model_id")?,
        model_source: row.try_get("model_source")?,
        claimed_by: row.try_get("claimed_by")?,
        attempts: row.get::<i32>("attempts")? as u32,
        lease_expires_at: row.try_get("lease_expires_at")?,
        priority: row.get("priority")?,
        claimable: row.get("claimable")?,
        acceleration_report: row.try_get("acceleration_report")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
    })
}

/// Input parameters for [`Catalog::submit_job`]. Grouped into one struct (the
/// `RegisterModelParams`/`CreateTrainingJobParams` pattern) so the insert
/// surface has one place to grow.
#[derive(Debug, Clone)]
pub struct SubmitJobParams<'a> {
    pub job_id: &'a str,
    pub kind: &'a str,
    /// `Queued` — claimable by the poll loop. `Inline` — claimed once, by id,
    /// in the submitting call itself; every job is inserted `status =
    /// 'queued'` regardless (the row's very first claim, whichever path
    /// claims it, is what performs the `queued -> running` transition).
    pub execution: JobExecution,
    pub spec: &'a str,
    /// The PK-keyed base model a training kind fine-tunes from.
    pub model_ref: Option<&'a str>,
    /// The NAME-keyed model a training kind will register on finish.
    pub output_model_id: Option<&'a str>,
    /// The NAME-keyed, FK-free `ModelSource` string a compute verb resolves
    /// its model against.
    pub model_source: Option<&'a str>,
    /// Claim-ordering tie-break before `created_at`. `0` reproduces plain
    /// oldest-first FIFO — see migration 024.
    pub priority: i32,
}

/// Input parameters for [`Catalog::finish_job`].
#[derive(Debug, Clone)]
pub struct FinishJobParams<'a> {
    pub job_id: &'a str,
    /// The instance the lease-guarded CAS matches against `claimed_by`.
    pub instance_id: &'a str,
    /// The attempt the lease-guarded CAS matches against `attempts`.
    pub attempts: u32,
    /// The tagged JSON terminal payload.
    pub result: &'a str,
}

/// One retained epoch checkpoint's catalog row, inserted inside the same
/// attempt-guarded finish transaction that commits the output model's served
/// path ([`Catalog::finish_job_with_model`]) — ported from the removed
/// `training_repo::EpochCheckpointRow` (unit 348) onto the generalised `jobs`
/// schema.
///
/// Lifecycle: a row is inserted **only** when the finish CAS this call rides
/// on actually wins — the insert sits inside the same `if job_updated == 1`
/// guard as the output model's `artifact_path` write. A zombie or
/// lease-lost caller's `finish_job_with_model` therefore matches zero job
/// rows and never reaches this insert at all, so it can never register an
/// epoch-checkpoint row for an attempt that lost the race — mirroring the
/// output model's own "served path is written by exactly one writer"
/// contract.
#[derive(Debug, Clone)]
pub struct EpochCheckpointRow<'a> {
    /// Distinct catalog name: `jammi:fine-tuned:{job_id}:epoch_{N}` — never
    /// an additional VERSION of the output model's name (that would let a
    /// later finish's version-scoped CAS clobber a different row's path; see
    /// [`Catalog::finish_job_with_model`]'s version predicate).
    pub model_id: &'a str,
    /// Catalog `model_type`, mirroring the output model row's convention
    /// (e.g. `"fine-tuned"`).
    pub model_type: &'a str,
    /// Same task the output model row carries.
    pub task: crate::model_task::ModelTask,
    /// Same base-model lineage the output model row carries.
    pub base_model_id: Option<&'a str>,
    /// The attempt-unique object-store prefix these bytes were published
    /// under — the bytes are already complete (manifest-last) by the time
    /// this row is built, unlike the output model row, which registers with
    /// no path and gains it only from the CAS below.
    pub artifact_path: &'a str,
}

/// Input parameters for [`Catalog::finish_job_with_model`] — grouped (the
/// `RegisterModelParams`/`FinishJobParams` pattern) so the attempt-guarded
/// call site names each field, and so a future addition to the finish
/// surface grows this struct rather than clippy's `too_many_arguments` limit.
pub struct FinishJobWithModelParams<'a> {
    /// The job id to finish.
    pub job_id: &'a str,
    /// The instance the attempt-guarded CAS matches against `claimed_by`.
    pub instance_id: &'a str,
    /// The attempt the attempt-guarded CAS matches against `attempts`.
    pub attempts: u32,
    /// The tagged JSON terminal payload, exactly as [`FinishJobParams::result`].
    pub result: &'a str,
    /// The output model's catalog NAME (`jobs.output_model_id`, already
    /// written at [`Catalog::submit_job`] time for a training kind — see
    /// [`JobRecord::output_model_id`]).
    pub output_model_id: &'a str,
    /// The output model's catalog VERSION — together with `output_model_id`
    /// and the tenant, the exact row the served-path `UPDATE` must touch and
    /// no other (B5, unit 348).
    pub output_model_version: i32,
    /// The object-store prefix this caller published the output artifact
    /// under — committed as the model row's `artifact_path`.
    pub artifact_path: &'a str,
    /// Every RETAINED epoch checkpoint to register alongside the output
    /// model (unit 348) — empty for a training kind with no per-epoch
    /// checkpointing.
    pub epoch_checkpoints: &'a [EpochCheckpointRow<'a>],
}

/// A row from the `workers` table joined with its owning `instances` row —
/// `ListWorkers`' wire shape (N12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub instance_id: String,
    pub label: Option<String>,
    pub host: Option<String>,
    pub kinds: String,
    pub started_at: String,
    pub last_seen_at: String,
}

fn parse_worker_row(
    row: &Row<'_>,
) -> std::result::Result<WorkerRecord, super::backend::BackendError> {
    Ok(WorkerRecord {
        instance_id: row.get("instance_id")?,
        label: row.try_get("label")?,
        host: row.try_get("host")?,
        kinds: row.get("kinds")?,
        started_at: row.get("started_at")?,
        last_seen_at: row.get("last_seen_at")?,
    })
}

impl Catalog {
    /// Submit a new job, `status = 'queued'` — the row's own claim (by
    /// [`Self::claim_next`] for `execution = 'queued'`, or by
    /// [`Self::claim_by_id`] for `execution = 'inline'`) performs the
    /// `queued -> running` transition. Tenant bound + asserted (SPEC-03 §7).
    pub async fn submit_job(&self, p: SubmitJobParams<'_>) -> Result<()> {
        let job_id = p.job_id.to_string();
        let kind = p.kind.to_string();
        let execution = p.execution.to_string();
        let spec = p.spec.to_string();
        let model_ref = p.model_ref.map(str::to_string);
        let output_model_id = p.output_model_id.map(str::to_string);
        let model_source = p.model_source.map(str::to_string);
        let priority = p.priority as i64;
        let tenant = self.current_tenant();
        let now = now_sortable();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "jobs")?;
                    let sql = format!(
                        "INSERT INTO jobs \
                         (job_id, kind, tenant_id, status, execution, spec, model_ref, \
                          output_model_id, model_source, priority, acceleration_report, \
                          created_at, updated_at) \
                         VALUES ($1, $2, $3, '{queued}', $4, $5, $6, $7, $8, $9, $10, $11, $11)",
                        queued = JobStatus::Queued
                    );
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(kind),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                            SqlValue::TextOwned(execution),
                            SqlValue::TextOwned(spec),
                            SqlValue::from(model_ref),
                            SqlValue::from(output_model_id),
                            SqlValue::from(model_source),
                            SqlValue::Int(priority),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Get a job by id. Tenant-filtered; inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the tenant
    /// predicate is dropped and the row resolves by its primary key alone.
    pub async fn get_job(&self, job_id: &str) -> Result<JobRecord> {
        let admin = TenantBinding::is_admin_scope();
        let sql = if admin {
            format!("SELECT {SELECT_COLS} FROM jobs WHERE job_id = $1")
        } else {
            format!(
                "SELECT {SELECT_COLS} FROM jobs WHERE job_id = $1 \
                   AND (tenant_id = $2 OR tenant_id IS NULL)"
            )
        };
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
                        let mut params = vec![SqlValue::TextOwned(id)];
                        if !admin {
                            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                        }
                        tx.query_opt(&sql, &params, parse_row).await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| JammiError::Catalog(format!("Job '{id_for_err}' not found")))
    }

    /// List jobs visible to the session tenant, most recent first. Inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the tenant
    /// predicate is dropped and every tenant's jobs are returned.
    pub async fn list_jobs(&self) -> Result<Vec<JobRecord>> {
        let admin = TenantBinding::is_admin_scope();
        let sql = if admin {
            format!("SELECT {SELECT_COLS} FROM jobs ORDER BY created_at DESC")
        } else {
            format!(
                "SELECT {SELECT_COLS} FROM jobs \
                 WHERE tenant_id = $1 OR tenant_id IS NULL \
                 ORDER BY created_at DESC"
            )
        };
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
                        let params: Vec<SqlValue<'static>> = if admin {
                            Vec::new()
                        } else {
                            vec![SqlValue::from(tenant.map(|t| t.to_string()))]
                        };
                        tx.query(&sql, &params, parse_row).await
                    })
                },
            )
            .await?)
    }

    /// Atomically claim the highest-priority `queued`-execution job of one of
    /// `kinds` for `instance_id`, leasing it for `lease`. Mirrors
    /// `training_repo::claim_next_training_job`'s ordering and per-backend
    /// concurrency (`FOR UPDATE SKIP LOCKED` on Postgres; SQLite's single
    /// serialised writer), generalised with a `kind IN (...)` filter and
    /// `execution = 'queued'` so an `inline` row — however it got stuck at
    /// `queued` — is never selected here. `Ok(None)` when `kinds` is empty or
    /// no matching job is claimable.
    pub async fn claim_next(
        &self,
        instance_id: &str,
        kinds: &[&str],
        lease: Duration,
    ) -> Result<Option<JobRecord>> {
        if kinds.is_empty() {
            return Ok(None);
        }
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let instance_id = instance_id.to_string();
        let now = now_sortable();
        let kind = self.backend().backend_kind();

        let mut params: Vec<SqlValue<'static>> = vec![
            SqlValue::TextOwned(running),
            SqlValue::TextOwned(instance_id),
            SqlValue::TextOwned(queued.clone()),
        ];
        let mut kind_placeholders = Vec::with_capacity(kinds.len());
        for k in kinds {
            params.push(SqlValue::TextOwned((*k).to_string()));
            kind_placeholders.push(format!("${}", params.len()));
        }
        let kind_in = kind_placeholders.join(", ");
        params.push(SqlValue::TextOwned(queued_execution));
        let execution_bind = params.len();

        let candidate = match kind {
            BackendKind::Postgres => format!(
                "(SELECT job_id FROM jobs WHERE status = $3 AND execution = ${execution_bind} \
                  AND claimable AND kind IN ({kind_in}) \
                  ORDER BY priority DESC, created_at LIMIT 1 FOR UPDATE SKIP LOCKED)"
            ),
            BackendKind::Sqlite => format!(
                "(SELECT job_id FROM jobs WHERE status = $3 AND execution = ${execution_bind} \
                  AND claimable AND kind IN ({kind_in}) \
                  ORDER BY priority DESC, created_at LIMIT 1)"
            ),
        };
        // The lease deadline is the DATABASE's own clock on Postgres (never
        // this process's — `catalog::lease`'s module docs).
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        let updated_at_bind = params.len();
        let sql = format!(
            "UPDATE jobs \
             SET status = $1, claimed_by = $2, lease_expires_at = {deadline_expr}, \
                 attempts = attempts + 1, updated_at = ${updated_at_bind} \
             WHERE job_id = {candidate} AND status = $3 \
             RETURNING {SELECT_COLS}"
        );

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.query_opt(&sql, &params, parse_row).await })
            })
            .await
            .map_err(Into::into)
    }

    /// Atomically claim one `inline`-execution job by id for `instance_id`
    /// (`queued -> running`, CAS on `job_id AND status = 'queued' AND
    /// execution = 'inline'`), leasing it for `lease`. `Ok(None)` when the
    /// row is absent, not `inline`, or already claimed.
    pub async fn claim_by_id(
        &self,
        job_id: &str,
        instance_id: &str,
        lease: Duration,
    ) -> Result<Option<JobRecord>> {
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let inline = JobExecution::Inline.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let now = now_sortable();
        let kind = self.backend().backend_kind();

        let mut params: Vec<SqlValue<'static>> = vec![
            SqlValue::TextOwned(running),
            SqlValue::TextOwned(instance_id),
        ];
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        let updated_at_bind = params.len();
        params.push(SqlValue::TextOwned(job_id));
        let job_id_bind = params.len();
        params.push(SqlValue::TextOwned(queued));
        let queued_bind = params.len();
        params.push(SqlValue::TextOwned(inline));
        let inline_bind = params.len();
        let sql = format!(
            "UPDATE jobs \
             SET status = $1, claimed_by = $2, lease_expires_at = {deadline_expr}, \
                 attempts = attempts + 1, updated_at = ${updated_at_bind} \
             WHERE job_id = ${job_id_bind} AND status = ${queued_bind} \
               AND execution = ${inline_bind} \
             RETURNING {SELECT_COLS}"
        );

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.query_opt(&sql, &params, parse_row).await })
            })
            .await
            .map_err(Into::into)
    }

    /// Renew the lease on a running job the caller still owns. `false` when
    /// the guard misses: the lease was lost, the job is no longer running, or
    /// `attempts` is stale (a successor already claimed this job).
    pub async fn heartbeat_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        lease: Duration,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let now = now_sortable();
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        params.push(SqlValue::TextOwned(job_id));
        params.push(SqlValue::TextOwned(running));
        params.push(SqlValue::TextOwned(instance_id));
        params.push(SqlValue::Int(attempts));
        let sql = format!(
            "UPDATE jobs SET lease_expires_at = {deadline_expr}, updated_at = $2 \
             WHERE job_id = $3 AND status = $4 AND claimed_by = $5 AND attempts = $6"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Finish a job the caller still owns: `running -> completed`, recording
    /// the terminal JSON `result`. `false` when the attempt guard misses.
    ///
    /// This is a TERMINAL write, so the same attempt-guarded UPDATE also
    /// retires a still-`pending` `acceleration_report` to
    /// `{"state":"undetermined","reason":"finalized_without_determination"}`
    /// (esc-075) — see [`JobRecord::acceleration_report`]'s lifecycle
    /// section. A training kind that also needs to commit its output
    /// model's served path and epoch checkpoints atomically with this
    /// transition should call [`Catalog::finish_job_with_model`] instead,
    /// never this method followed by a second write.
    pub async fn finish_job(&self, p: FinishJobParams<'_>) -> Result<bool> {
        let completed = JobStatus::Completed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = p.job_id.to_string();
        let instance_id = p.instance_id.to_string();
        let attempts = p.attempts as i64;
        let result = p.result.to_string();
        let now = now_sortable();
        let retire = retire_pending_report_clause(3, 4);
        let sql = format!(
            "UPDATE jobs SET status = $1, result = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(completed),
                            SqlValue::TextOwned(result),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::Text(ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Finish a job the caller still owns as a single attempt-guarded
    /// compare-and-set that ALSO commits the output model's served artifact
    /// path and registers every surviving epoch-checkpoint row (unit 348) —
    /// the atomic finish-with-model-registration peer of [`Self::finish_job`],
    /// ported from the removed `training_repo::finalize_training_job` onto
    /// the generalised `jobs` schema's full attempt guard (`job_id AND
    /// claimed_by AND status = 'running' AND attempts = $n`, where the old
    /// method carried only `job_id AND claimed_by AND status`).
    ///
    /// In one transaction it flips the job row to `completed` and writes
    /// `result` — and, only if that job-row CAS matched, records
    /// `artifact_path` on the output model's row and inserts one row per
    /// `epoch_checkpoints` entry. The job-row CAS lands **only** while the
    /// row is still `running`, `claimed_by == instance_id`, and `attempts ==
    /// attempts`. Returns `true` when the caller held the lease and is the
    /// sole finisher, `false` when it was not (the lease was lost, the row
    /// is no longer `running`, or `attempts` is stale — a successor already
    /// claimed this job). A `false` return means the caller must not act as
    /// the finisher; the job is left for [`Self::reclaim_expired_jobs`] and
    /// whichever instance re-claims it.
    ///
    /// The model row is matched by `name = output_model_id AND version =
    /// output_model_version`, tenant-scoped with the same STRICT predicate
    /// [`Self::delete_model`] uses (`tenant_id = $t OR (tenant_id IS NULL AND
    /// $t IS NULL)`) — never the relaxed `OR tenant_id IS NULL` a *read*
    /// resolver uses to also see a global row. The model-row update is gated
    /// on the job-row CAS matching (it runs in the same transaction and is
    /// skipped when the CAS matched zero rows), so the finish CAS is the
    /// **sole writer** of the served path: a loser's finish matches no job
    /// row and therefore writes neither the job status nor the model's
    /// served path.
    ///
    /// The `jobs` CAS itself stays NOT tenant-scoped, matching every other
    /// lease-identity write in this module: the lease identity (`claimed_by`,
    /// `attempts`) is the authority there, not the session tenant, and
    /// `job_id` is a global unique PK so no cross-tenant collision is
    /// possible on that row. The `models` predicate above is the one that
    /// needed tenant scoping, because `models.name` is NOT globally unique
    /// the way `job_id` is.
    ///
    /// This is a TERMINAL write, so the same CAS also retires a still-pending
    /// `acceleration_report` (esc-075) exactly as [`Self::finish_job`] does
    /// — see [`JobRecord::acceleration_report`]'s lifecycle section.
    ///
    /// An epoch-checkpoint whose catalog NAME is already occupied by another
    /// row is skipped (logged), never failing the whole job over one name
    /// collision: its bytes remain durable on the object store but
    /// permanently unregistered. The pre-check is by NAME ALONE (every
    /// version, the same strict tenant predicate the output model's UPDATE
    /// uses) — never a bare `ON CONFLICT(model_id) DO NOTHING`, which only
    /// catches an EXACT `(tenant, name, version=1)` PK collision and would
    /// silently let a same-name-different-version row SHADOW the checkpoint
    /// from every reader via `ORDER BY version DESC` resolution.
    pub async fn finish_job_with_model(&self, p: FinishJobWithModelParams<'_>) -> Result<bool> {
        let FinishJobWithModelParams {
            job_id,
            instance_id,
            attempts,
            result,
            output_model_id,
            output_model_version,
            artifact_path,
            epoch_checkpoints,
        } = p;
        let completed = JobStatus::Completed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let result = result.to_string();
        let output_model_id = output_model_id.to_string();
        let output_model_version = output_model_version as i64;
        let artifact_path = artifact_path.to_string();
        let now = now_sortable();
        let tenant = self.current_tenant();
        let retire = retire_pending_report_clause(3, 4);
        let job_sql = format!(
            "UPDATE jobs SET status = $1, result = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        // Owned, 'static-lifetime copies of the epoch rows so the transaction
        // closure (which must be `'static`) can move them without borrowing
        // `epoch_checkpoints`.
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
                            &job_sql,
                            &[
                                SqlValue::TextOwned(completed),
                                SqlValue::TextOwned(result),
                                SqlValue::Text(ACCELERATION_REPORT_PENDING),
                                SqlValue::Text(ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(job_id),
                                SqlValue::TextOwned(instance_id),
                                SqlValue::TextOwned(running),
                                SqlValue::Int(attempts),
                            ],
                        )
                        .await?;
                    // The gate every model-side write below hangs off: only
                    // the attempt that won the job-row CAS commits the
                    // served path and the epoch rows — pinned by
                    // `jobs_queue.rs::finish_job_with_model_is_an_attempt_guarded_compare_and_set`,
                    // the finish-side sibling of esc-107's `create_result_table`
                    // control.
                    if job_updated == 1 {
                        tx.assert_tenant_matches(tenant, "models")?;
                        let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));
                        tx.execute(
                            "UPDATE models SET artifact_path = $1, updated_at = $2 \
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

    /// Fail a job the caller still owns: `running -> failed`, recording
    /// `error`. `false` when the attempt guard misses.
    ///
    /// This is a TERMINAL write, so the same attempt-guarded UPDATE also
    /// retires a still-`pending` `acceleration_report` to
    /// `{"state":"undetermined","reason":"failed_before_probe"}` (esc-075) —
    /// see [`JobRecord::acceleration_report`]'s lifecycle section.
    pub async fn fail_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        error: &str,
    ) -> Result<bool> {
        let failed = JobStatus::Failed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let error = error.to_string();
        let now = now_sortable();
        let retire = retire_pending_report_clause(3, 4);
        let sql = format!(
            "UPDATE jobs SET status = $1, error = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(failed),
                            SqlValue::TextOwned(error),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::Text(ACCELERATION_REPORT_FAILED_BEFORE_PROBE),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record a checkpoint's progress on a job the caller still owns. `false`
    /// when the attempt guard misses.
    pub async fn progress_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        rows_done: Option<i64>,
        rows_total: Option<i64>,
        phase: Option<&str>,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let phase = phase.map(str::to_string);
        let attempts = attempts as i64;
        let now = now_sortable();
        let sql = "UPDATE jobs SET progress_rows_done = $1, progress_rows_total = $2, \
                   progress_phase = $3, updated_at = $4 \
                   WHERE job_id = $5 AND claimed_by = $6 AND status = $7 AND attempts = $8";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::from(rows_done),
                            SqlValue::from(rows_total),
                            SqlValue::from(phase),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record `partial_result` on a job the caller still owns, outside a
    /// `create_result_table` transaction (e.g. adopting a predecessor's
    /// already-`ready` table on a reclaimed attempt — N1). `false` when the
    /// attempt guard misses. The narrower, `job_id`-only CAS
    /// [`Self::create_result_table`] performs at table-CREATION time is a
    /// separate, deliberately weaker write — see its doc.
    pub async fn record_partial_result(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        table_name: &str,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let table_name = table_name.to_string();
        let attempts = attempts as i64;
        let now = now_sortable();
        let sql = "UPDATE jobs SET partial_result = $1, updated_at = $2 \
                   WHERE job_id = $3 AND claimed_by = $4 AND status = $5 AND attempts = $6";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record the claiming instance's acceleration determination (esc-075)
    /// for this attempt of a job the caller still owns — the report-writing
    /// peer of [`Self::progress_job`]. Replaces `acceleration_report`
    /// **only** while the row is still `running`, `claimed_by == instance_id`,
    /// **and `attempts == attempt`**.
    ///
    /// The `attempts` guard is mandatory, not a defensive extra:
    /// `claimed_by` can carry an instance id that is stable across process
    /// restarts, so `(job_id, claimed_by, status)` alone cannot tell "the
    /// current claimant, mid-run" from "a zombie of the same instance
    /// identity, from an attempt this job already moved past via reclaim". A
    /// reclaim always bumps `attempts` on re-claim ([`Self::claim_next`]'s
    /// and [`Self::claim_by_id`]'s `attempts = attempts + 1`), so pinning the
    /// exact attempt closes precisely that gap: a zombie presenting its own
    /// stale `attempt` value matches zero rows and can never overwrite the
    /// current claimant's report, even when it shares the current claimant's
    /// instance id and the row happens to be `running` again under a *later*
    /// attempt.
    ///
    /// Returns `true` when the write landed (the guard matched) and `false`
    /// when it did not — the lease was lost, the job is no longer running, or
    /// the caller's `attempt` is stale. Not tenant-scoped, matching the other
    /// lease-identity operations ([`Self::heartbeat_job`],
    /// [`Self::progress_job`], [`Self::finish_job`], [`Self::fail_job`]).
    pub async fn record_acceleration_report(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        report_json: &str,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let report_json = report_json.to_string();
        let attempts = attempts as i64;
        let now = now_sortable();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE jobs SET acceleration_report = $1, updated_at = $2 \
                         WHERE job_id = $3 AND claimed_by = $4 AND status = $5 AND attempts = $6",
                        &[
                            SqlValue::TextOwned(report_json),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Request cancellation of a non-terminal job. Sets `cancel_requested`;
    /// the executor observes it at a checkpoint boundary. Tenant-scoped with
    /// the STRICT predicate [`Self::delete_model`] uses; inside
    /// [`crate::session::JammiSession::with_admin_scope`] the predicate is
    /// dropped. `false` when the row is absent, out of scope, or already
    /// terminal (a terminal job cannot be cancelled retroactively).
    pub async fn cancel_request(&self, job_id: &str) -> Result<bool> {
        let admin = TenantBinding::is_admin_scope();
        let job_id_s = job_id.to_string();
        let tenant = self.current_tenant();
        let now = now_sortable();
        let non_terminal = JobStatus::non_terminal_sql_list();
        let sql = if admin {
            format!(
                "UPDATE jobs SET cancel_requested = TRUE, updated_at = $1 \
                 WHERE job_id = $2 AND status IN ({non_terminal})"
            )
        } else {
            format!(
                "UPDATE jobs SET cancel_requested = TRUE, updated_at = $1 \
                 WHERE job_id = $2 AND status IN ({non_terminal}) \
                   AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL))"
            )
        };
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let mut params = vec![SqlValue::TextOwned(now), SqlValue::TextOwned(job_id_s)];
                    if !admin {
                        params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                    }
                    tx.execute(&sql, &params).await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Reclaim every expired job across the catalog. Two arms, both scoped to
    /// `status = 'running'`:
    ///
    ///   - **queued-execution, lease expired** (absent-or-expired
    ///     `lease_expires_at`, the same `lease_expired_clause` predicate
    ///     `training_repo::reclaim_expired_training_jobs` used): requeue when
    ///     `attempts < max_attempts`, otherwise fail with
    ///     `"job lease expired after exhausting max attempts"`.
    ///   - **inline-execution, owning instance dead**: failed with
    ///     `"inline executor died"` when no `instances` row for `claimed_by`
    ///     has `last_seen_at` within `2 * lease` of now — covering BOTH an
    ///     absent instance row and a present-but-stale one with one `NOT
    ///     EXISTS (… live …)` predicate. Never requeued: an inline job has no
    ///     poll loop to re-claim it.
    ///
    /// `lease` is the deployment's lease WINDOW (the same value passed to
    /// [`Self::claim_next`]/[`Self::heartbeat_job`]) — used both to interpret
    /// `jobs.lease_expires_at` and, doubled, as the inline-liveness margin
    /// against `instances.last_seen_at`. Returns the number of jobs actioned
    /// across both arms. Not tenant-scoped — sweeps every tenant.
    ///
    /// Every arm also moves `acceleration_report`, in its OWN `UPDATE` (never
    /// a read-then-write): a requeue resets it to the pending marker
    /// (esc-075) — the row will re-probe under its next attempt — while both
    /// terminal arms (attempts-exhausted, inline-executor-dead) retire a
    /// still-pending marker with their own distinct reason. See
    /// [`JobRecord::acceleration_report`]'s lifecycle section.
    pub async fn reclaim_expired_jobs(&self, lease: Duration, max_attempts: u32) -> Result<usize> {
        let reclaimed = self
            .reclaim_expired_jobs_inner(lease, max_attempts, None)
            .await?;
        Ok(reclaimed.len())
    }

    /// The row-scoped peer of [`Self::reclaim_expired_jobs`]: given a job
    /// already read (by [`Self::get_job`]/[`Self::list_jobs`] or a wire
    /// `JobStatus`/`WaitJob` call), reclaim it in place if — and only if — it
    /// currently qualifies under either arm. `Ok(None)` when the row does not
    /// qualify (including: it is no longer `running` by the time this runs,
    /// or `record.job_id` no longer exists) — the steady-state case, which
    /// issues the SAME `UPDATE … WHERE …` predicate as the bulk sweep scoped
    /// by `job_id`, so a non-qualifying row costs one no-op statement, never
    /// an unconditional write.
    pub async fn reclaim_job_on_read(
        &self,
        record: &JobRecord,
        lease: Duration,
        max_attempts: u32,
    ) -> Result<Option<JobRecord>> {
        if record.status != JobStatus::Running.to_string() {
            return Ok(None);
        }
        let mut reclaimed = self
            .reclaim_expired_jobs_inner(lease, max_attempts, Some(&record.job_id))
            .await?;
        Ok(reclaimed.pop())
    }

    async fn reclaim_expired_jobs_inner(
        &self,
        lease: Duration,
        max_attempts: u32,
        only_job_id: Option<&str>,
    ) -> Result<Vec<JobRecord>> {
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let failed = JobStatus::Failed.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let inline_execution = JobExecution::Inline.to_string();
        let max_attempts_i = max_attempts as i64;
        let now = now_sortable();
        let kind = self.backend().backend_kind();
        let scope_job = only_job_id.map(str::to_string);
        let exhausted_error = "job lease expired after exhausting max attempts".to_string();
        let inline_error = "inline executor died".to_string();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let mut out = Vec::new();

                    // Arm 1a: queued-execution, lease expired, attempts left -> requeue.
                    // Unconditionally RESETS acceleration_report to the pending
                    // marker (esc-075): the row returns to `queued` for a NEW
                    // attempt that will re-probe, so "no claimant has computed a
                    // determination yet" is once again exactly true, even when
                    // the dead attempt had already recorded a `determined`
                    // payload that describes hardware/config THAT attempt saw
                    // and must not be attributed to the next, not-yet-started
                    // attempt. See `JobRecord::acceleration_report`'s lifecycle
                    // section.
                    {
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(queued.clone()),
                            SqlValue::TextOwned(now.clone()),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                        ];
                        let pending_bind = params.len();
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(queued_execution.clone()));
                        let execution_bind = params.len();
                        params.push(SqlValue::Int(max_attempts_i));
                        let max_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, claimed_by = NULL, \
                                 lease_expires_at = NULL, acceleration_report = ${pending_bind}, \
                                 updated_at = $2 \
                             WHERE status = ${running_bind} AND execution = ${execution_bind} \
                               AND {expired} AND attempts < ${max_bind}{scope_sql} \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    // Arm 1b: queued-execution, lease expired, attempts exhausted -> fail.
                    // TERMINAL, so it retires a still-pending acceleration_report
                    // (esc-075) — see `JobRecord::acceleration_report`'s lifecycle
                    // section.
                    {
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(failed.clone()),
                            SqlValue::TextOwned(exhausted_error.clone()),
                        ];
                        params.push(SqlValue::TextOwned(now.clone()));
                        let now_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_PENDING));
                        let pending_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED));
                        let terminal_bind = params.len();
                        let retire = retire_pending_report_clause(pending_bind, terminal_bind);
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(queued_execution.clone()));
                        let execution_bind = params.len();
                        params.push(SqlValue::Int(max_attempts_i));
                        let max_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, error = $2, \
                                 lease_expires_at = NULL, {retire}, updated_at = ${now_bind} \
                             WHERE status = ${running_bind} AND execution = ${execution_bind} \
                               AND {expired} AND attempts >= ${max_bind}{scope_sql} \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    // Arm 2: inline-execution, owning instance absent or stale -> fail.
                    // TERMINAL (an inline job has no requeue arm), so it retires
                    // a still-pending acceleration_report (esc-075) — see
                    // `JobRecord::acceleration_report`'s lifecycle section.
                    {
                        let margin = lease.saturating_mul(2);
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(failed.clone()),
                            SqlValue::TextOwned(inline_error.clone()),
                        ];
                        params.push(SqlValue::TextOwned(now.clone()));
                        let now_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_PENDING));
                        let pending_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_INLINE_EXECUTOR_DIED));
                        let terminal_bind = params.len();
                        let retire = retire_pending_report_clause(pending_bind, terminal_bind);
                        let stale =
                            stale_before_clause("i.last_seen_at", kind, margin, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(inline_execution.clone()));
                        let execution_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, error = $2, \
                                 lease_expires_at = NULL, {retire}, updated_at = ${now_bind} \
                             WHERE status = ${running_bind} AND execution = ${execution_bind}\
                                 {scope_sql} \
                               AND NOT EXISTS ( \
                                 SELECT 1 FROM instances i \
                                 WHERE i.instance_id = jobs.claimed_by AND NOT ({stale}) \
                               ) \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    Ok(out)
                })
            })
            .await
            .map_err(Into::into)
    }

    /// Upsert this process's `instances` row: insert on first call
    /// (`started_at` stamped once), refresh `label`/`host`/`last_seen_at` on
    /// every later call.
    pub async fn upsert_instance(
        &self,
        instance_id: &str,
        label: Option<&str>,
        host: Option<&str>,
    ) -> Result<()> {
        let instance_id = instance_id.to_string();
        let label = label.map(str::to_string);
        let host = host.map(str::to_string);
        let now = now_sortable();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO instances (instance_id, label, host, started_at, last_seen_at) \
                         VALUES ($1, $2, $3, $4, $4) \
                         ON CONFLICT(instance_id) DO UPDATE SET \
                             label = excluded.label, host = excluded.host, \
                             last_seen_at = excluded.last_seen_at",
                        &[
                            SqlValue::TextOwned(instance_id),
                            SqlValue::from(label),
                            SqlValue::from(host),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// Heartbeat this process's `instances` row. `false` when the row is
    /// absent (recovery pruned it before this beat landed).
    pub async fn touch_instance(&self, instance_id: &str) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let now = now_sortable();
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE instances SET last_seen_at = $1 WHERE instance_id = $2",
                        &[SqlValue::TextOwned(now), SqlValue::TextOwned(instance_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Upsert this process's `workers` row — present only while the process
    /// runs the claim loop. `kinds` is the comma-joined (or otherwise
    /// producer-encoded) kind set this worker claims.
    pub async fn upsert_worker(&self, instance_id: &str, kinds: &str) -> Result<()> {
        let instance_id = instance_id.to_string();
        let kinds = kinds.to_string();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO workers (instance_id, kinds) VALUES ($1, $2) \
                         ON CONFLICT(instance_id) DO UPDATE SET kinds = excluded.kinds",
                        &[SqlValue::TextOwned(instance_id), SqlValue::TextOwned(kinds)],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// Delete this process's `workers` row — the claim loop has stopped, so
    /// the process must no longer advertise itself as a claimant (a
    /// `ListWorkers` read after an `EmbeddedWorker` drop shows no row for
    /// it, rather than a row that lingers until its `instances` row goes
    /// stale and cascades). The `instances` row itself is untouched: the
    /// process is still alive and may still hold inline jobs. `false` when
    /// no row existed.
    pub async fn delete_worker(&self, instance_id: &str) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM workers WHERE instance_id = $1",
                        &[SqlValue::TextOwned(instance_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(deleted == 1)
    }

    /// List every worker, joined with its owning `instances` row (N12).
    pub async fn list_workers(&self) -> Result<Vec<WorkerRecord>> {
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
                            "SELECT w.instance_id AS instance_id, i.label AS label, \
                                    i.host AS host, w.kinds AS kinds, \
                                    i.started_at AS started_at, i.last_seen_at AS last_seen_at \
                             FROM workers w JOIN instances i ON w.instance_id = i.instance_id \
                             ORDER BY i.started_at",
                            &[],
                            parse_worker_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Delete `instances` rows stale for at least `stale_after` — their
    /// `workers` row (if any) cascades with them. Returns the count deleted.
    pub async fn prune_instances(&self, stale_after: Duration) -> Result<usize> {
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let stale = stale_before_clause("last_seen_at", kind, stale_after, &mut params);
        let sql = format!("DELETE FROM instances WHERE {stale}");
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(deleted as usize)
    }

    /// Delete terminal (`completed`/`failed`) job rows whose `updated_at` is
    /// at least `retention` in the past. A non-terminal job is never pruned,
    /// regardless of age. Returns the count deleted.
    pub async fn prune_jobs(&self, retention: Duration) -> Result<usize> {
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let stale = stale_before_clause("updated_at", kind, retention, &mut params);
        let terminal = JobStatus::terminal_sql_list();
        let sql = format!("DELETE FROM jobs WHERE status IN ({terminal}) AND {stale}");
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(deleted as usize)
    }
}
