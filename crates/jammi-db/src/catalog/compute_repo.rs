//! The `compute_executors` / `compute_jobs` catalog tables (migration 038,
//! `docs/plans/67-distributed-training/UNITS.md` § U8b): generic CRUD over a
//! distributor-neutral compute-cluster registry (B1/K5 — no `ballista` or
//! any other distributor vocabulary in this module; the Ballista-shaped
//! traits over these verbs live in `jammi-ballista`, not here).
//!
//! `compute_executors` is BOTH the executor registry (host/ports/capacity/
//! liveness) AND, via its `devices` column, the placement policy's sole
//! device authority — [`Catalog::list_compute_executor_devices`] reads it
//! directly, never through a join on `workers.instance_id` (67 pressure-
//! round delta 3: an executor process and a `[worker]` process are
//! different roles that may see different devices, so only the executor's
//! own registration fact is trustworthy for a placement decision).
//! `compute_jobs` mirrors job OWNERSHIP AND STATUS only — the execution
//! graph itself has no serialisation in Ballista 54.1, so a scheduler
//! restart never revives an in-flight graph from this table; jammi's own
//! reclaim re-runs the job.

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::instance::{decode_devices_json, DeviceFact};
use super::Catalog;
use crate::error::Result;

/// A row from `compute_executors`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeExecutorRecord {
    /// The distributor's own executor identity. Opaque to this crate.
    pub executor_id: String,
    /// Carried for display/correlation only — NOT the placement join key
    /// (see [`Catalog::list_compute_executor_devices`]'s doc).
    pub instance_id: String,
    pub host: String,
    /// This executor's shuffle (Arrow Flight) listener port.
    pub port: u16,
    /// This executor's task (gRPC) listener port.
    pub grpc_port: u16,
    /// Total task-slot capacity.
    pub task_slots: u32,
    /// Capacity not currently bound. Always `0 <= available_slots <=
    /// task_slots` in the COMMITTED result of every write this module
    /// makes ([`Catalog::adjust_compute_slots`], [`Catalog::
    /// bind_compute_slots`]) — never enforced by a schema `CHECK`.
    pub available_slots: u32,
    /// The executor's own free-text state. Opaque here.
    pub status: String,
    /// Last liveness signal, `TEXT` in the same lease-timestamp family
    /// other catalog clocks use.
    pub heartbeat_at: String,
    /// Free-form `TEXT` (e.g. the distributor's own JSON executor
    /// description). Never parsed by this crate.
    pub metadata: String,
    /// This executor's OWN device registration — the placement join's sole
    /// authority. A malformed stored value decodes to an empty list with a
    /// `tracing::warn!` naming `executor_id` (see
    /// [`super::instance::decode_devices_json`]), never a read fault.
    pub devices: Vec<DeviceFact>,
}

/// A row from `compute_jobs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeJobRecord {
    pub job_id: String,
    pub owner: String,
    pub status: String,
    pub queued_at: String,
    pub updated_at: String,
}

fn parse_compute_executor_row(
    row: &Row<'_>,
) -> std::result::Result<ComputeExecutorRecord, BackendError> {
    let executor_id: String = row.get("executor_id")?;
    let devices_json: String = row.get("devices")?;
    let devices = decode_devices_json(&devices_json, &executor_id);
    Ok(ComputeExecutorRecord {
        instance_id: row.get("instance_id")?,
        host: row.get("host")?,
        port: row.get::<i32>("port")? as u16,
        grpc_port: row.get::<i32>("grpc_port")? as u16,
        task_slots: row.get::<i32>("task_slots")? as u32,
        available_slots: row.get::<i32>("available_slots")? as u32,
        status: row.get("status")?,
        heartbeat_at: row.get("heartbeat_at")?,
        metadata: row.get("metadata")?,
        executor_id,
        devices,
    })
}

fn parse_compute_job_row(row: &Row<'_>) -> std::result::Result<ComputeJobRecord, BackendError> {
    Ok(ComputeJobRecord {
        job_id: row.get("job_id")?,
        owner: row.get("owner")?,
        status: row.get("status")?,
        queued_at: row.get("queued_at")?,
        updated_at: row.get("updated_at")?,
    })
}

impl Catalog {
    /// Register (or fully replace) a compute executor's row, including its
    /// device claim.
    pub async fn upsert_compute_executor(&self, rec: &ComputeExecutorRecord) -> Result<()> {
        let executor_id = rec.executor_id.clone();
        let instance_id = rec.instance_id.clone();
        let host = rec.host.clone();
        let port = i64::from(rec.port);
        let grpc_port = i64::from(rec.grpc_port);
        let task_slots = i64::from(rec.task_slots);
        let available_slots = i64::from(rec.available_slots);
        let status = rec.status.clone();
        let heartbeat_at = rec.heartbeat_at.clone();
        let metadata = rec.metadata.clone();
        let devices_json = serde_json::to_string(&rec.devices)
            .map_err(|e| crate::error::JammiError::Catalog(format!("devices encode: {e}")))?;
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO compute_executors \
                         (executor_id, instance_id, host, port, grpc_port, task_slots, \
                          available_slots, status, heartbeat_at, metadata, devices) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) \
                         ON CONFLICT(executor_id) DO UPDATE SET \
                             instance_id = excluded.instance_id, \
                             host = excluded.host, \
                             port = excluded.port, \
                             grpc_port = excluded.grpc_port, \
                             task_slots = excluded.task_slots, \
                             available_slots = excluded.available_slots, \
                             status = excluded.status, \
                             heartbeat_at = excluded.heartbeat_at, \
                             metadata = excluded.metadata, \
                             devices = excluded.devices",
                        &[
                            SqlValue::TextOwned(executor_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(host),
                            SqlValue::Int(port),
                            SqlValue::Int(grpc_port),
                            SqlValue::Int(task_slots),
                            SqlValue::Int(available_slots),
                            SqlValue::TextOwned(status),
                            SqlValue::TextOwned(heartbeat_at),
                            SqlValue::TextOwned(metadata),
                            SqlValue::TextOwned(devices_json),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// Every registered compute executor, ordered by `executor_id` (family
    /// J: a fixed, deterministic order — never the backend's own unordered
    /// scan order).
    pub async fn list_compute_executors(&self) -> Result<Vec<ComputeExecutorRecord>> {
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
                            "SELECT executor_id, instance_id, host, port, grpc_port, \
                                    task_slots, available_slots, status, heartbeat_at, \
                                    metadata, devices \
                             FROM compute_executors ORDER BY executor_id",
                            &[],
                            parse_compute_executor_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Look up one compute executor by id.
    pub async fn get_compute_executor(
        &self,
        executor_id: &str,
    ) -> Result<Option<ComputeExecutorRecord>> {
        let executor_id = executor_id.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT executor_id, instance_id, host, port, grpc_port, \
                                    task_slots, available_slots, status, heartbeat_at, \
                                    metadata, devices \
                             FROM compute_executors WHERE executor_id = $1",
                            &[SqlValue::TextOwned(executor_id)],
                            parse_compute_executor_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Update only `status`/`heartbeat_at` on an existing executor row —
    /// every other column (capacity, devices, listeners) is untouched.
    /// `false` when no row exists for `executor_id`.
    pub async fn record_compute_heartbeat(
        &self,
        executor_id: &str,
        status: &str,
        heartbeat_at: &str,
    ) -> Result<bool> {
        let executor_id = executor_id.to_string();
        let status = status.to_string();
        let heartbeat_at = heartbeat_at.to_string();
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE compute_executors SET status = $1, heartbeat_at = $2 \
                         WHERE executor_id = $3",
                        &[
                            SqlValue::TextOwned(status),
                            SqlValue::TextOwned(heartbeat_at),
                            SqlValue::TextOwned(executor_id),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Remove a compute executor's row. `false` when no row existed.
    pub async fn remove_compute_executor(&self, executor_id: &str) -> Result<bool> {
        let executor_id = executor_id.to_string();
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM compute_executors WHERE executor_id = $1",
                        &[SqlValue::TextOwned(executor_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(deleted == 1)
    }

    /// Apply every `(executor_id, delta)` pair to `compute_executors.
    /// available_slots` in ONE transaction: every row moves, or none does.
    /// A `delta` that would take `available_slots` below `0` or above
    /// `task_slots` — for ANY pair in the batch — refuses the WHOLE batch
    /// typed ([`BackendError::Constraint`]) with NO row changed, before any
    /// `UPDATE` in the batch is issued (every pair is validated against a
    /// fresh read of its own row first; the actual writes happen only after
    /// every pair has cleared that check). An `executor_id` naming no row
    /// is the same typed refusal, naming the missing id.
    pub async fn adjust_compute_slots(&self, deltas: &[(&str, i64)]) -> Result<()> {
        let deltas: Vec<(String, i64)> = deltas
            .iter()
            .map(|(id, delta)| ((*id).to_string(), *delta))
            .collect();
        self.backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    // Phase 1: validate every pair against a fresh read of
                    // its own row. No write happens in this phase.
                    for (executor_id, delta) in &deltas {
                        let row = tx
                            .query_opt(
                                "SELECT available_slots, task_slots FROM compute_executors \
                                 WHERE executor_id = $1",
                                &[SqlValue::Text(executor_id)],
                                |row| {
                                    Ok((
                                        row.get::<i32>("available_slots")?,
                                        row.get::<i32>("task_slots")?,
                                    ))
                                },
                            )
                            .await?;
                        let Some((available, task_slots)) = row else {
                            return Err(BackendError::Constraint {
                                table: "compute_executors".to_string(),
                                detail: format!(
                                    "adjust_compute_slots: no row for executor_id '{executor_id}'"
                                ),
                            });
                        };
                        let next = i64::from(available) + delta;
                        if next < 0 || next > i64::from(task_slots) {
                            return Err(BackendError::Constraint {
                                table: "compute_executors".to_string(),
                                detail: format!(
                                    "adjust_compute_slots: executor_id '{executor_id}' delta \
                                     {delta} would take available_slots from {available} to \
                                     {next}, outside [0, {task_slots}]"
                                ),
                            });
                        }
                    }
                    // Phase 2: every pair validated — apply every write.
                    for (executor_id, delta) in &deltas {
                        tx.execute(
                            "UPDATE compute_executors \
                             SET available_slots = available_slots + $1 \
                             WHERE executor_id = $2",
                            &[SqlValue::Int(*delta), SqlValue::Text(executor_id)],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// CAS-decrement `available_slots` by `n` iff `available_slots >= n`.
    /// `false` (no row changed) when no row exists for `executor_id` OR the
    /// executor does not currently have `n` free slots — the two arms are
    /// indistinguishable to the caller BY DESIGN, the same way
    /// [`Self::heartbeat_job`]'s attempt
    /// guard collapses "row gone" and "attempt guard missed" into one
    /// boolean: a placement decision that lost a CAS race retries against a
    /// fresh read either way.
    pub async fn bind_compute_slots(&self, executor_id: &str, n: u32) -> Result<bool> {
        let executor_id = executor_id.to_string();
        let n = i64::from(n);
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE compute_executors \
                         SET available_slots = available_slots - $1 \
                         WHERE executor_id = $2 AND available_slots >= $1",
                        &[SqlValue::Int(n), SqlValue::TextOwned(executor_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Upsert a `compute_jobs` row — every column overwritten from `rec`.
    pub async fn put_compute_job(&self, rec: &ComputeJobRecord) -> Result<()> {
        let job_id = rec.job_id.clone();
        let owner = rec.owner.clone();
        let status = rec.status.clone();
        let queued_at = rec.queued_at.clone();
        let updated_at = rec.updated_at.clone();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO compute_jobs (job_id, owner, status, queued_at, updated_at) \
                         VALUES ($1, $2, $3, $4, $5) \
                         ON CONFLICT(job_id) DO UPDATE SET \
                             owner = excluded.owner, \
                             status = excluded.status, \
                             queued_at = excluded.queued_at, \
                             updated_at = excluded.updated_at",
                        &[
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(owner),
                            SqlValue::TextOwned(status),
                            SqlValue::TextOwned(queued_at),
                            SqlValue::TextOwned(updated_at),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// Look up one compute job by id.
    pub async fn get_compute_job(&self, job_id: &str) -> Result<Option<ComputeJobRecord>> {
        let job_id = job_id.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT job_id, owner, status, queued_at, updated_at \
                             FROM compute_jobs WHERE job_id = $1",
                            &[SqlValue::TextOwned(job_id)],
                            parse_compute_job_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Every compute job, ordered by `job_id` (family J: a fixed order).
    pub async fn list_compute_jobs(&self) -> Result<Vec<ComputeJobRecord>> {
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
                            "SELECT job_id, owner, status, queued_at, updated_at \
                             FROM compute_jobs ORDER BY job_id",
                            &[],
                            parse_compute_job_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Delete a `compute_jobs` row. `false` when no row existed.
    pub async fn delete_compute_job(&self, job_id: &str) -> Result<bool> {
        let job_id = job_id.to_string();
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM compute_jobs WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(deleted == 1)
    }

    /// The placement policy's ONE read: every registered executor's OWN
    /// device claim, decoded, ordered by `executor_id`. Reads
    /// `compute_executors.devices` DIRECTLY — never a join through
    /// `workers.instance_id` (see the module doc and [`ComputeExecutorRecord::
    /// devices`]'s doc for why a `[worker]` row is not the placement join
    /// key). A malformed `devices` value decodes to an empty list for that
    /// executor alone (the row-fact rule; see [`super::instance::
    /// decode_devices_json`]), never a fault of this whole read.
    pub async fn list_compute_executor_devices(&self) -> Result<Vec<(String, Vec<DeviceFact>)>> {
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
                            "SELECT executor_id, devices FROM compute_executors \
                             ORDER BY executor_id",
                            &[],
                            |row| {
                                let executor_id: String = row.get("executor_id")?;
                                let devices_json: String = row.get("devices")?;
                                let devices = decode_devices_json(&devices_json, &executor_id);
                                Ok((executor_id, devices))
                            },
                        )
                        .await
                    })
                },
            )
            .await?)
    }
}
