//! Timeout enforcement for ephemeral sessions.
//!
//! Every open [`EphemeralSession`](super::EphemeralSession) registers a
//! lightweight snapshot of itself — id, tenant, deadline, and the physical
//! table ids it owns — in a process-shared [`ActiveSessions`] map. A background
//! [`scan`] pass (run on an interval by [`spawn`]) force-closes every session
//! whose deadline has passed: it drops the session's tables and publishes a
//! `timed_out` lifecycle event, exactly as an explicit `force_close` would.
//!
//! The scanner does not hold the `EphemeralSession` value itself, so a session
//! that is explicitly closed (or dropped) deregisters its snapshot and the
//! scanner simply never sees it. Conversely, a session that is force-closed by
//! the scanner has its snapshot removed atomically, so a later explicit `close`
//! finds no tables left to drop.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::session::JammiSession;
use crate::store::mutable::definition::MutableTableId;
use crate::tenant::TenantId;

use super::event::{SessionLifecycleEvent, SessionLifecycleRecord};
use super::topic;

/// Default interval between timeout scans. Appropriate for the demo; a
/// production deployment that wants tighter bounds passes its own to [`spawn`].
pub const DEFAULT_SCAN_INTERVAL: Duration = Duration::from_secs(60);

/// One open session's snapshot, owned by the [`ActiveSessions`] map.
#[derive(Clone)]
struct SessionSnapshot {
    tenant: TenantId,
    deadline: DateTime<Utc>,
    tables: Vec<MutableTableId>,
}

/// Process-shared registry of open ephemeral sessions, keyed by session id.
///
/// Cloneable handle over a shared map; the [`EphemeralSession`](super::EphemeralSession)
/// owns one clone to (de)register itself, and the scanner owns another to read
/// expired entries.
#[derive(Clone, Default)]
pub struct ActiveSessions {
    inner: Arc<Mutex<HashMap<Uuid, SessionSnapshot>>>,
}

impl ActiveSessions {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a session's snapshot. Called on open and whenever
    /// the session's table set changes.
    pub(super) fn upsert(
        &self,
        session_id: Uuid,
        tenant: TenantId,
        deadline: DateTime<Utc>,
        tables: Vec<MutableTableId>,
    ) {
        if let Ok(mut map) = self.inner.lock() {
            map.insert(
                session_id,
                SessionSnapshot {
                    tenant,
                    deadline,
                    tables,
                },
            );
        }
    }

    /// Remove a session's snapshot. Returns `true` if it was still present —
    /// i.e. the caller (explicit close / Drop) won the race against the
    /// scanner and now owns the cleanup.
    pub(super) fn remove(&self, session_id: &Uuid) -> bool {
        self.inner
            .lock()
            .map(|mut map| map.remove(session_id).is_some())
            .unwrap_or(false)
    }

    /// Atomically take the snapshots of every session whose deadline has passed
    /// as of `now`, removing them from the map so the scanner owns their
    /// cleanup and no concurrent close double-drops.
    fn take_expired(&self, now: DateTime<Utc>) -> Vec<(Uuid, SessionSnapshot)> {
        let Ok(mut map) = self.inner.lock() else {
            return Vec::new();
        };
        let expired: Vec<Uuid> = map
            .iter()
            .filter(|(_, snap)| now >= snap.deadline)
            .map(|(id, _)| *id)
            .collect();
        expired
            .into_iter()
            .filter_map(|id| map.remove(&id).map(|snap| (id, snap)))
            .collect()
    }

    /// Number of currently-registered sessions. Primarily for tests.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// `true` if no sessions are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Run one timeout pass: force-close every session expired as of now.
///
/// Each expired session has its tables dropped and a `timed_out` lifecycle
/// event published. Drop failures are logged and surface as a
/// `partial_deletion_failure` event for that session; one session's failure
/// does not abort the pass for the others.
pub async fn scan(parent: &Arc<JammiSession>, active: &ActiveSessions) {
    let now = Utc::now();
    for (session_id, snap) in active.take_expired(now) {
        // Best-effort: log the outcome, never propagate (the scanner loops on).
        if let Err(e) = delete_tables_and_emit(
            parent,
            session_id,
            snap.tenant,
            &snap.tables,
            SessionLifecycleEvent::TimedOut,
        )
        .await
        {
            tracing::warn!(
                session_id = %session_id,
                error = %e,
                "ephemeral timeout force-close completed with errors",
            );
        }
    }
}

/// Outcome of a deletion pass: the physical tables that could not be dropped
/// (empty on full success). The deleted-row count is recorded in the published
/// lifecycle event rather than returned, since no caller needs it back.
pub(super) struct DeletionOutcome {
    pub failures: Vec<String>,
}

/// Drop every table in `tables` under `tenant`, sum the rows removed, and
/// publish the terminal lifecycle event — `event` on full success, or
/// `partial_deletion_failure` (listing survivors, tagging the attempted event)
/// if any drop failed. Shared by the timeout scanner, the explicit `close`
/// path, and the best-effort `Drop` impl so all three agree on semantics.
///
/// Row counts and the lifecycle publish are pinned to `tenant` via the parent's
/// task-local scope, so the routine is correct even when called detached (the
/// scanner, `Drop`) with no sticky binding on the parent session.
pub(super) async fn delete_tables_and_emit(
    parent: &Arc<JammiSession>,
    session_id: Uuid,
    tenant: TenantId,
    tables: &[MutableTableId],
    event: SessionLifecycleEvent,
) -> Result<DeletionOutcome, super::error::EphemeralError> {
    let mut deleted_rows: u64 = 0;
    let mut failures: Vec<String> = Vec::new();

    for id in tables {
        // Best-effort count before the drop — a count failure must not block the
        // deletion, which is the operation the PWS guarantee depends on.
        match count_rows(parent, tenant, id).await {
            Ok(n) => deleted_rows += n,
            Err(e) => tracing::warn!(
                session_id = %session_id,
                table = %id.as_str(),
                error = %e,
                "ephemeral row count failed before drop",
            ),
        }
        if let Err(e) = parent.drop_mutable_table(id).await {
            tracing::warn!(
                session_id = %session_id,
                table = %id.as_str(),
                error = %e,
                "ephemeral table drop failed",
            );
            failures.push(id.as_str().to_string());
        }
    }

    let (emit_event, details) = if failures.is_empty() {
        (event, None)
    } else {
        (
            SessionLifecycleEvent::PartialDeletionFailure,
            Some(serde_json::json!({
                "surviving_tables": failures,
                "attempted_event": event,
            })),
        )
    };

    let record = SessionLifecycleRecord {
        session_id,
        tenant_id: tenant.to_string(),
        event: emit_event,
        occurred_at: Utc::now(),
        ephemeral_table_count: tables.len(),
        deleted_row_count: deleted_rows,
        details,
    };
    topic::publish_lifecycle(parent, tenant, &record).await?;

    Ok(DeletionOutcome { failures })
}

/// Row count for one physical table, pinned to `tenant` so the count is correct
/// regardless of the parent session's sticky binding.
async fn count_rows(
    parent: &Arc<JammiSession>,
    tenant: TenantId,
    id: &MutableTableId,
) -> Result<u64, super::error::EphemeralError> {
    let sql = format!(
        "SELECT COUNT(*) AS n FROM mutable.public.\"{}\"",
        id.as_str()
    );
    let batches = parent
        .with_tenant_scoped(tenant, move |scope| async move { scope.sql(&sql).await })
        .await
        .map_err(|e| super::error::EphemeralError::Storage(e.to_string()))?;
    let count = batches
        .first()
        .and_then(|b| b.column_by_name("n"))
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::Int64Array>())
        .filter(|a| !a.is_empty())
        .map(|a| a.value(0))
        .unwrap_or(0);
    Ok(count.max(0) as u64)
}

/// Spawn a detached background task that runs [`scan`] every `interval` until
/// the returned [`tokio::task::JoinHandle`] is aborted (or the runtime shuts
/// down). Library users that want timeout enforcement in-process call this once
/// after opening their database; `jammi-server` runs it for the lifetime of the
/// process.
pub fn spawn(
    parent: Arc<JammiSession>,
    active: ActiveSessions,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // The first tick fires immediately; skip it so we do not scan before
        // any session has had a chance to register.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            scan(&parent, &active).await;
        }
    })
}
