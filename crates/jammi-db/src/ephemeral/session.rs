//! Session-scoped ephemeral storage context.
//!
//! An [`EphemeralSession`] owns a set of mutable tables whose physical ids are
//! prefixed with the session's own UUID. The tables are real mutable companion
//! tables — they federate into the same SQL surface as every other table — but
//! the session is their unit of deletion: `close` (or `Drop`, or the timeout
//! scanner) drops every table the session created and publishes a lifecycle
//! event to `jammi.audit.session_lifecycle.v1`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow_schema::SchemaRef;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::catalog::backend::TxOptions;
use crate::session::JammiSession;
use crate::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use crate::tenant::TenantId;

use super::error::EphemeralError;
use super::event::{SessionLifecycleEvent, SessionLifecycleRecord};
use super::scanner::ActiveSessions;
use super::topic;

/// Physical-id prefix marking a table as session-scoped. The full physical id
/// is `__eph_<32-hex session id>_<logical name>`.
const EPHEMERAL_PREFIX: &str = "__eph_";

/// Characters consumed by the prefix and the simple (hyphenless) session UUID
/// plus the joining underscore: `__eph_` (6) + 32 + `_` (1).
const RESERVED_ID_CHARS: usize = EPHEMERAL_PREFIX.len() + 32 + 1;

/// Maximum logical-name length so the physical id stays within the
/// [`MutableTableId`] 63-character cap.
pub const MAX_LOGICAL_NAME_LEN: usize = 63 - RESERVED_ID_CHARS;

/// A session-scoped storage context.
///
/// Tables created through [`EphemeralSession::create_ephemeral_table`] live only
/// for the lifetime of the session; [`EphemeralSession::close`] (the safe path)
/// or [`Drop`] (best-effort) deletes them. The session is pinned to the tenant
/// bound on its parent at open time and refuses to operate without one.
pub struct EphemeralSession {
    parent: Arc<JammiSession>,
    session_id: Uuid,
    tenant: TenantId,
    created_at: DateTime<Utc>,
    timeout: Duration,
    /// Logical name -> physical mutable-table id, in insertion order.
    tables: BTreeMap<String, MutableTableId>,
    /// Cleared by `close`/`force_close` so `Drop` does not re-attempt deletion
    /// of tables that were already dropped on the safe path.
    closed: bool,
    /// Shared registry the timeout scanner reads. The session keeps its snapshot
    /// in sync on open and on every table creation, and removes it on
    /// close/Drop so the scanner never double-drops.
    active: ActiveSessions,
}

impl EphemeralSession {
    /// Open a session scoped to the tenant currently bound on `parent`,
    /// registering it with `active` so the timeout scanner can force-close it.
    ///
    /// Returns [`EphemeralError::NoTenantBinding`] if no tenant is bound — an
    /// ephemeral session is meaningless without scope. Publishes an `opened`
    /// lifecycle event before returning.
    pub async fn open(
        parent: Arc<JammiSession>,
        timeout: Duration,
        active: ActiveSessions,
    ) -> Result<Self, EphemeralError> {
        let tenant = parent.tenant().ok_or(EphemeralError::NoTenantBinding)?;
        let session = Self {
            parent,
            session_id: Uuid::new_v4(),
            tenant,
            created_at: Utc::now(),
            timeout,
            tables: BTreeMap::new(),
            closed: false,
            active,
        };
        session.register_snapshot();
        session.emit(SessionLifecycleEvent::Opened, 0, None).await?;
        Ok(session)
    }

    /// Refresh this session's snapshot in the shared registry so the scanner
    /// sees the current table set and deadline.
    fn register_snapshot(&self) {
        let tables: Vec<MutableTableId> = self.tables.values().cloned().collect();
        self.active
            .upsert(self.session_id, self.tenant, self.deadline(), tables);
    }

    /// The wall-clock instant past which the scanner considers this session
    /// expired. A timeout that overflows `chrono::Duration` is treated as
    /// effectively unbounded (deadline far in the future).
    fn deadline(&self) -> DateTime<Utc> {
        match chrono::Duration::from_std(self.timeout) {
            Ok(d) => self.created_at + d,
            Err(_) => DateTime::<Utc>::MAX_UTC,
        }
    }

    /// The session's unique identifier.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// The tenant this session is pinned to.
    pub fn tenant(&self) -> TenantId {
        self.tenant
    }

    /// When the session was opened.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// The configured idle timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// `true` if the session has passed its timeout deadline as of `now`.
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.deadline()
    }

    /// The physical mutable-table id for a logical name, if the session created
    /// one under it.
    pub fn physical_table_id(&self, name: &str) -> Option<&MutableTableId> {
        self.tables.get(name)
    }

    /// Build the physical id for a logical name within this session.
    fn physical_id(&self, name: &str) -> Result<MutableTableId, EphemeralError> {
        if name.len() > MAX_LOGICAL_NAME_LEN {
            return Err(EphemeralError::NameTooLong {
                name: name.to_string(),
                max: MAX_LOGICAL_NAME_LEN,
            });
        }
        let raw = format!("{EPHEMERAL_PREFIX}{}_{name}", self.session_id.as_simple());
        MutableTableId::new(raw).map_err(|e| EphemeralError::Storage(e.to_string()))
    }

    /// Create a mutable table whose data lives only within this session.
    ///
    /// The table is created under the session's tenant scope, so only this
    /// tenant can read it back. `schema` and `primary_key` follow the same
    /// contract as [`JammiSession::create_mutable_table`].
    pub async fn create_ephemeral_table(
        &mut self,
        name: &str,
        schema: SchemaRef,
        primary_key: Vec<String>,
    ) -> Result<(), EphemeralError> {
        if self.tables.contains_key(name) {
            return Err(EphemeralError::DuplicateTable(name.to_string()));
        }
        let id = self.physical_id(name)?;
        let def = MutableTableDefinitionBuilder::new(id.clone(), schema)
            .primary_key(primary_key)
            .tenant(Some(self.tenant))
            .build()
            .map_err(|e| EphemeralError::Storage(e.to_string()))?;
        self.parent
            .create_mutable_table(def)
            .await
            .map_err(|e| EphemeralError::Storage(e.to_string()))?;
        self.tables.insert(name.to_string(), id);
        self.register_snapshot();
        Ok(())
    }

    /// Append a batch to an ephemeral table by logical name.
    ///
    /// The batch schema must match the table's registered schema. The insert
    /// runs inside a transaction bound to the session's tenant, so every row
    /// carries the right scope.
    pub async fn insert(&self, name: &str, batch: RecordBatch) -> Result<u64, EphemeralError> {
        let id = self
            .tables
            .get(name)
            .ok_or_else(|| EphemeralError::UnknownTable(name.to_string()))?
            .clone();
        let registry = self.parent.mutable_tables_arc();
        let backend = self.parent.catalog().backend_arc();
        let tenant = self.tenant;
        backend
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    tx.set_tenant(Some(tenant));
                    let rows = registry.insert_batch(tx, &id, &batch).await.map_err(|e| {
                        crate::catalog::backend::BackendError::Execution(e.to_string())
                    })?;
                    Ok(rows)
                })
            })
            .await
            .map_err(|e| EphemeralError::Storage(e.to_string()))
    }

    /// Run a read query against an ephemeral table.
    ///
    /// `query` is a SQL template in which `{table}` is replaced by the fully
    /// qualified reference to the named ephemeral table (e.g.
    /// `SELECT * FROM {table} WHERE image_id = 'x'`). The query runs pinned to
    /// the session's own tenant — regardless of what the parent session's sticky
    /// binding currently is — so the session is self-consistent and only ever
    /// sees its own tenant's rows.
    pub async fn sql(&self, name: &str, query: &str) -> Result<Vec<RecordBatch>, EphemeralError> {
        let table_ref = self.table_ref(name)?;
        let resolved = query.replace("{table}", &table_ref);
        self.scoped_sql(&resolved).await
    }

    /// Run `sql` pinned to the session's tenant via the parent's task-local
    /// scope, so reads do not depend on the parent's current sticky binding.
    async fn scoped_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, EphemeralError> {
        let sql = sql.to_string();
        self.parent
            .with_tenant_scoped(
                self.tenant,
                move |scope| async move { scope.sql(&sql).await },
            )
            .await
            .map_err(|e| EphemeralError::Storage(e.to_string()))
    }

    /// The fully qualified SQL reference for an ephemeral table by logical name.
    pub fn table_ref(&self, name: &str) -> Result<String, EphemeralError> {
        let id = self
            .tables
            .get(name)
            .ok_or_else(|| EphemeralError::UnknownTable(name.to_string()))?;
        Ok(format!("mutable.public.\"{}\"", id.as_str()))
    }

    /// Count the rows currently stored in an ephemeral table (tenant-scoped).
    pub async fn count_rows(&self, name: &str) -> Result<u64, EphemeralError> {
        let id = self
            .tables
            .get(name)
            .ok_or_else(|| EphemeralError::UnknownTable(name.to_string()))?;
        self.count_physical_rows(id).await
    }

    /// Explicitly close the session: delete every ephemeral table it created
    /// and publish a `closed` lifecycle event. This is the recommended path —
    /// the [`Drop`] impl is best-effort.
    ///
    /// On a partial drop failure, a `partial_deletion_failure` event is emitted
    /// listing the surviving physical tables and
    /// [`EphemeralError::PartialDeletionFailure`] is returned; the successfully
    /// dropped tables stay dropped.
    pub async fn close(mut self) -> Result<(), EphemeralError> {
        self.delete_all(SessionLifecycleEvent::Closed).await
    }

    /// Force-close the session with a `timed_out` lifecycle event. Used by the
    /// timeout scanner; semantically identical to [`Self::close`] but tags the
    /// transition as a timeout rather than a deliberate close.
    pub async fn force_close(mut self) -> Result<(), EphemeralError> {
        self.delete_all(SessionLifecycleEvent::TimedOut).await
    }

    /// Drop every ephemeral table, summing deleted rows, then publish the
    /// terminal lifecycle event. Shared by `close` and `force_close`.
    async fn delete_all(&mut self, event: SessionLifecycleEvent) -> Result<(), EphemeralError> {
        // Drain locally so a second call (e.g. Drop after an errored close) is a
        // no-op, and mark closed before any await so Drop never re-enters.
        let tables: Vec<MutableTableId> = std::mem::take(&mut self.tables).into_values().collect();
        self.closed = true;

        // Claim the registry slot. If the timeout scanner already took it, it
        // owns (or already did) the deletion + event for these tables; this
        // explicit close then has nothing left to do.
        if !self.active.remove(&self.session_id) {
            return Ok(());
        }

        let total = tables.len();
        let outcome = super::scanner::delete_tables_and_emit(
            &self.parent,
            self.session_id,
            self.tenant,
            &tables,
            event,
        )
        .await?;

        if outcome.failures.is_empty() {
            Ok(())
        } else {
            Err(EphemeralError::PartialDeletionFailure {
                session_id: self.session_id,
                failed: outcome.failures.len(),
                total,
            })
        }
    }

    /// Row count for an already-resolved physical id, pinned to the session's
    /// tenant so the count is correct independent of the parent's binding.
    async fn count_physical_rows(&self, id: &MutableTableId) -> Result<u64, EphemeralError> {
        let sql = format!(
            "SELECT COUNT(*) AS n FROM mutable.public.\"{}\"",
            id.as_str()
        );
        let batches = self.scoped_sql(&sql).await?;
        let count = batches
            .first()
            .and_then(|b| b.column_by_name("n"))
            .and_then(|c| c.as_any().downcast_ref::<arrow::array::Int64Array>())
            .filter(|a| !a.is_empty())
            .map(|a| a.value(0))
            .unwrap_or(0);
        Ok(count.max(0) as u64)
    }

    /// Publish one lifecycle record to `jammi.audit.session_lifecycle.v1`,
    /// scoped to the session's tenant.
    async fn emit(
        &self,
        event: SessionLifecycleEvent,
        deleted_row_count: u64,
        details: Option<serde_json::Value>,
    ) -> Result<(), EphemeralError> {
        let record = SessionLifecycleRecord {
            session_id: self.session_id,
            tenant_id: self.tenant.to_string(),
            event,
            occurred_at: Utc::now(),
            ephemeral_table_count: self.tables.len(),
            deleted_row_count,
            details,
        };
        topic::publish_lifecycle(&self.parent, self.tenant, &record).await
    }
}

impl Drop for EphemeralSession {
    /// Best-effort deletion of any tables the safe `close`/`force_close` path
    /// did not already remove. Spawns a detached task on the ambient tokio
    /// runtime; failures are logged, not surfaced. The context-manager / explicit
    /// `close` path is the only one that reports deletion outcomes.
    fn drop(&mut self) {
        if self.closed || self.tables.is_empty() {
            return;
        }
        // Claim the registry slot so the timeout scanner does not also try to
        // drop these tables. If the scanner already took it, it owns cleanup.
        if !self.active.remove(&self.session_id) {
            return;
        }
        let parent = Arc::clone(&self.parent);
        let tenant = self.tenant;
        let session_id = self.session_id;
        let tables: Vec<MutableTableId> = self.tables.values().cloned().collect();

        // Only attempt the spawn when a runtime is actually present; outside a
        // runtime (e.g. a test dropping the value on a plain thread) there is
        // nothing to spawn onto and the tables outlive the process anyway.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                session_id = %session_id,
                "ephemeral session dropped outside a tokio runtime; \
                 {} table(s) not deleted — use close() for guaranteed cleanup",
                tables.len(),
            );
            return;
        };

        handle.spawn(async move {
            // Same deletion + lifecycle-publish path as the explicit close and
            // the timeout scanner; Drop is best-effort, so the outcome is logged
            // rather than surfaced.
            if let Err(e) = super::scanner::delete_tables_and_emit(
                &parent,
                session_id,
                tenant,
                &tables,
                SessionLifecycleEvent::Closed,
            )
            .await
            {
                tracing::warn!(
                    session_id = %session_id,
                    error = %e,
                    "ephemeral best-effort Drop cleanup completed with errors",
                );
            }
        });
    }
}
