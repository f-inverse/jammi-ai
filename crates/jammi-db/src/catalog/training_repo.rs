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
}

const SELECT_COLS: &str = "job_id, base_model_id, output_model_id, training_source, loss_type, \
     hyperparams, status, metrics, created_at, kind, claimed_by, lease_expires_at, attempts, \
     tenant_id, training_spec";

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
                          kind, training_spec, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, 'queued', $6, $7, $8)",
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(base_model_id),
                            SqlValue::TextOwned(training_source),
                            SqlValue::TextOwned(loss_type),
                            SqlValue::TextOwned(hyperparams),
                            SqlValue::TextOwned(kind),
                            SqlValue::TextOwned(training_spec),
                            SqlValue::from(tenant.map(|t| t.to_string())),
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

    /// Update a training job's status and optional metrics JSON. Scoped.
    pub async fn update_training_status(
        &self,
        job_id: &str,
        status: super::status::TrainingJobStatus,
        metrics: Option<&str>,
    ) -> Result<()> {
        let status_str = status.to_string();
        let metrics = metrics.map(str::to_string);
        let id = job_id.to_string();
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE training_jobs SET status = $1, metrics = $2, \
                         updated_at = CAST(CURRENT_TIMESTAMP AS TEXT) \
                         WHERE job_id = $3 AND (tenant_id = $4 OR tenant_id IS NULL)",
                        &[
                            SqlValue::TextOwned(status_str),
                            SqlValue::from(metrics),
                            SqlValue::TextOwned(id),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Finalize a training job the caller still owns, as a single compare-and-set
    /// gated on lease ownership. Atomically writes `output_model_id`, flips the
    /// status to `completed`, and (when `metrics` is `Some`) records the run
    /// metrics — but **only** while the row is still `running` and
    /// `claimed_by == worker_id`. Returns `true` when the row was updated
    /// (the caller held the lease and is the sole finalizer) and `false` when it
    /// was not (the lease was lost — the row is no longer `running`, or another
    /// worker reclaimed it). A `false` return means the caller must not register
    /// the output model or otherwise act as the finalizer; the job is left for
    /// [`Self::reclaim_expired_training_jobs`] and the worker that re-claims it.
    ///
    /// Not tenant-scoped, matching [`Self::claim_next_training_job`] and
    /// [`Self::heartbeat_training_job`]: the lease identity (`claimed_by`) is the
    /// authority, not the session tenant. The single `UPDATE … WHERE claimed_by
    /// AND status = 'running'` is the same lease guard the heartbeat uses, so a
    /// worker whose lease was reclaimed mid-run matches zero rows here and the
    /// worker that now owns the job is the only one whose CAS succeeds.
    pub async fn finalize_training_job(
        &self,
        job_id: &str,
        worker_id: &str,
        output_model_id: &str,
        metrics: Option<&str>,
    ) -> Result<bool> {
        let completed = TrainingJobStatus::Completed.to_string();
        let running = TrainingJobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let worker_id = worker_id.to_string();
        let output_model_id = output_model_id.to_string();
        let metrics = metrics.map(str::to_string);
        let now = lease_now();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE training_jobs \
                         SET output_model_id = $1, status = $2, \
                             metrics = COALESCE($3, metrics), updated_at = $4 \
                         WHERE job_id = $5 AND claimed_by = $6 AND status = $7",
                        &[
                            SqlValue::TextOwned(output_model_id),
                            SqlValue::TextOwned(completed),
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

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE training_jobs \
                         SET status = $1, metrics = COALESCE($2, metrics), updated_at = $3 \
                         WHERE job_id = $4 AND claimed_by = $5 AND status = $6",
                        &[
                            SqlValue::TextOwned(failed),
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

    /// Atomically claim the oldest queued training job for `worker_id`, leasing
    /// it for `lease`. On success the row transitions `queued → running`,
    /// stamps `claimed_by = worker_id`, sets `lease_expires_at = now + lease`,
    /// and increments `attempts`; the claimed record is returned. `Ok(None)`
    /// when no job is queued.
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
                "(SELECT job_id FROM training_jobs WHERE status = $3 \
                  ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED)"
            }
            super::backend::BackendKind::Sqlite => {
                "(SELECT job_id FROM training_jobs WHERE status = $3 \
                  ORDER BY created_at LIMIT 1)"
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

        let actioned = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let requeued = tx
                        .execute(
                            "UPDATE training_jobs \
                             SET status = $1, claimed_by = NULL, lease_expires_at = NULL, \
                                 updated_at = $2 \
                             WHERE status = $3 AND lease_expires_at IS NOT NULL \
                               AND lease_expires_at < $4 AND attempts < $5",
                            &[
                                SqlValue::TextOwned(queued),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(running.clone()),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::Int(max_attempts),
                            ],
                        )
                        .await?;
                    let exhausted = tx
                        .execute(
                            "UPDATE training_jobs \
                             SET status = $1, metrics = $2, lease_expires_at = NULL, \
                                 updated_at = $3 \
                             WHERE status = $4 AND lease_expires_at IS NOT NULL \
                               AND lease_expires_at < $5 AND attempts >= $6",
                            &[
                                SqlValue::TextOwned(failed),
                                SqlValue::TextOwned(failure_metrics),
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
