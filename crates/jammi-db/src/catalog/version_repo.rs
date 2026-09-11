//! Catalog access for versioned result tables (`result_table_versions`,
//! migration 032).
//!
//! One logical table keeps its `result_tables` row; every refreshed state is a
//! row here, keyed `(table_name, version)`, lease-owned while `building` and
//! immutable once `ready`. Three compare-and-sets own the whole lifecycle:
//!
//! - [`Catalog::allocate_result_table_version`] — the monotonic allocator:
//!   `UPDATE result_tables SET next_version = next_version + 1` under the row
//!   lock (no read before the write: a read-then-write transaction on
//!   SQLite/WAL can fail `SQLITE_BUSY_SNAPSHOT`, which `busy_timeout` does not
//!   retry; on Postgres the row lock serialises allocators under the READ
//!   COMMITTED the catalog uses), then the `building` row INSERT in the same
//!   transaction. A number is allocated exactly once and never reused.
//! - [`Catalog::publish_base_version`] — the first refresh's base publish:
//!   the `ready` version row describing today's table and the
//!   `current_version IS NULL AND next_version = $B` CAS, one transaction.
//! - [`Catalog::publish_version`] — the sole commit point of a refresh or a
//!   compaction: renew-by-CAS, the version row's `building -> ready`, and the
//!   `result_tables.current_version = $parent -> $N` swap, one transaction;
//!   anything else rolls back to a typed miss.
//!
//! Every write carries the STRICT tenant pair `tenant_id = $t OR (tenant_id
//! IS NULL AND $t IS NULL)` under a tenant binding (a scoped tenant can read
//! a GLOBAL table but never refresh it), and no arm inside an admin scope.
//! `result_tables.status` / `writer_id` are never touched here — a versioned
//! table never re-enters `building` (invariant I-A2).

use std::time::Duration;

use crate::catalog::backend::{BackendError, Row, SqlValue, Transaction, TxOptions};
use crate::catalog::lease::{lease_deadline_expr, lease_expired_clause};
use crate::catalog::result_repo::{read_cas_target, CasTarget, TenantArm};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::error::{JammiError, Result};
use crate::storage::StorageUrl;
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// One `result_table_versions` row.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ResultTableVersionRecord {
    pub table_name: String,
    pub version: i64,
    /// The version this one was refreshed from; `None` for the base.
    pub parent_version: Option<i64>,
    /// `building` / `ready` / `failed`.
    pub status: String,
    /// The `{table}__v{N}.version.json` manifest URL.
    pub manifest_path: String,
    /// The version identity (K7), set at publish.
    pub identity: Option<String>,
    pub live_rows: Option<usize>,
    pub masked_rows: Option<usize>,
    pub writer_id: Option<String>,
    pub lease_expires_at: Option<String>,
    pub tenant_id: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

fn parse_version(row: &Row<'_>) -> std::result::Result<ResultTableVersionRecord, BackendError> {
    Ok(ResultTableVersionRecord {
        table_name: row.get("table_name")?,
        version: i64::from(row.get::<i32>("version")?),
        parent_version: row.try_get::<i32>("parent_version")?.map(i64::from),
        status: row.get("status")?,
        manifest_path: row.get("manifest_path")?,
        identity: row.try_get("identity")?,
        live_rows: row.try_get::<i32>("live_rows")?.map(|v| v.max(0) as usize),
        masked_rows: row
            .try_get::<i32>("masked_rows")?
            .map(|v| v.max(0) as usize),
        writer_id: row.try_get("writer_id")?,
        lease_expires_at: row.try_get("lease_expires_at")?,
        tenant_id: row.try_get("tenant_id")?,
        created_at: row.get("created_at")?,
        completed_at: row.try_get("completed_at")?,
    })
}

/// Render the tenant arm (`""` inside an admin scope, else the STRICT pair)
/// against column prefix `col`, appending its bind.
fn tenant_arm_sql(arm: &TenantArm, col: &str, params: &mut Vec<SqlValue<'static>>) -> String {
    match arm {
        TenantArm::Admin => String::new(),
        TenantArm::Strict(t) => {
            params.push(SqlValue::from(t.map(|t| t.to_string())));
            let n = params.len();
            format!(" AND ({col}tenant_id = ${n} OR ({col}tenant_id IS NULL AND ${n} IS NULL))")
        }
    }
}

/// The writer's compare-and-set predicate on a `building` version row:
/// `table_name = $t AND version = $n AND status = 'building' AND writer_id =
/// $w [AND <tenant arm>]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionCas {
    pub table: String,
    pub version: i64,
    pub writer_id: String,
    pub tenant_arm: TenantArm,
}

impl VersionCas {
    /// A live writer's CAS on its own version row under the binding in force.
    pub fn writer(table: &str, version: i64, writer_id: &str, tenant: Option<TenantId>) -> Self {
        Self {
            table: table.to_string(),
            version,
            writer_id: writer_id.to_string(),
            tenant_arm: TenantArm::in_force(tenant),
        }
    }

    /// The lease keeper's CAS: the writer match is the whole ownership check
    /// (see [`crate::catalog::result_repo::ResultTableCas::writer_any_tenant`]).
    pub fn writer_any_tenant(table: &str, version: i64, writer_id: &str) -> Self {
        Self {
            table: table.to_string(),
            version,
            writer_id: writer_id.to_string(),
            tenant_arm: TenantArm::Admin,
        }
    }

    fn render(&self, params: &mut Vec<SqlValue<'static>>) -> String {
        params.push(SqlValue::TextOwned(self.table.clone()));
        let t = params.len();
        params.push(SqlValue::Int(self.version));
        let v = params.len();
        params.push(SqlValue::TextOwned(self.writer_id.clone()));
        let w = params.len();
        let mut sql = format!(
            "table_name = ${t} AND version = ${v} AND status = 'building' AND writer_id = ${w}"
        );
        sql.push_str(&tenant_arm_sql(&self.tenant_arm, "", params));
        sql
    }
}

/// The version row a zero-row CAS re-reads by primary key (no tenant
/// predicate) to classify the miss.
#[derive(Debug, Clone)]
struct VersionTarget {
    tenant_id: Option<String>,
    status: String,
    writer_id: Option<String>,
}

async fn read_version_target(
    tx: &mut Transaction<'_>,
    table: &str,
    version: i64,
) -> std::result::Result<Option<VersionTarget>, BackendError> {
    tx.query_opt(
        "SELECT tenant_id, status, writer_id FROM result_table_versions \
         WHERE table_name = $1 AND version = $2",
        &[
            SqlValue::TextOwned(table.to_string()),
            SqlValue::Int(version),
        ],
        |row| {
            Ok(VersionTarget {
                tenant_id: row.try_get("tenant_id")?,
                status: row.get("status")?,
                writer_id: row.try_get("writer_id")?,
            })
        },
    )
    .await
}

/// What [`Catalog::allocate_result_table_version`] handed out.
#[derive(Debug, Clone)]
pub struct AllocatedVersion {
    /// The allocated number `N` (`next_version` before the increment).
    pub version: i64,
    /// `result_tables.current_version` at allocation — the parent.
    pub parent: Option<i64>,
    /// The `{table}__v{N}.version.json` URL the row was stamped with.
    pub manifest_path: String,
    /// The owning tenant, inherited from the table row.
    pub tenant_id: Option<String>,
}

/// The inputs of [`Catalog::publish_version`].
#[derive(Debug, Clone)]
pub struct PublishVersion<'a> {
    pub cas: &'a VersionCas,
    /// The lease window the renew-by-CAS stamps before the flip.
    pub lease: Duration,
    /// The parent `result_tables.current_version` must still equal.
    pub parent: Option<i64>,
    pub identity: &'a str,
    pub live_rows: usize,
    pub masked_rows: usize,
    /// The canonical input-anchors JSON persisted on the table row.
    pub anchors_json: &'a str,
}

/// Outcome of a single version-row CAS statement.
enum VersionCasOutcome {
    Applied,
    Missed(Option<VersionTarget>),
}

/// The transaction-internal rollback sentinels: which CAS missed.
const MISS_TABLE: &str = "table";
const MISS_VERSION: &str = "version";

fn completed_now() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

impl Catalog {
    /// Classify a zero-row compare-and-set on a READY `result_tables` row
    /// (the allocation UPDATE, the base CAS, the publish CAS) from its
    /// re-read: `RowGone`, `TenantMismatch` (the STRICT arm refused), or
    /// `CasFailed { status }` — the row is not `ready`, or it IS `ready` but
    /// its `current_version` / `next_version` arm moved under the caller (a
    /// concurrent publisher). Owner-less on purpose: a `result_tables` row a
    /// concurrent recovery flipped to `building` must never read as
    /// `LeaseLost` to a refresher.
    pub(crate) fn classify_ready_cas_miss(
        table: &str,
        arm: &TenantArm,
        target: Option<CasTarget>,
    ) -> JammiError {
        let table = table.to_string();
        let Some(row) = target else {
            return JammiError::RowGone { table };
        };
        if let TenantArm::Strict(t) = arm {
            if row.tenant_id != t.map(|t| t.to_string()) {
                return JammiError::TenantMismatch { table };
            }
        }
        JammiError::CasFailed {
            table,
            status: row.status,
        }
    }

    fn classify_version_cas_miss(cas: &VersionCas, target: Option<VersionTarget>) -> JammiError {
        let table = cas.table.clone();
        let Some(row) = target else {
            return JammiError::RowGone { table };
        };
        if let TenantArm::Strict(t) = &cas.tenant_arm {
            if row.tenant_id != t.map(|t| t.to_string()) {
                return JammiError::TenantMismatch { table };
            }
        }
        if row.status != ResultTableStatus::Building.to_string() {
            return JammiError::CasFailed {
                table,
                status: row.status,
            };
        }
        if row.writer_id.as_deref() == Some(cas.writer_id.as_str()) {
            return JammiError::CasFailed {
                table,
                status: row.status,
            };
        }
        JammiError::LeaseLost { table }
    }

    /// Allocate the next version number of a `ready` table and insert its
    /// `building` row under `writer_id`'s lease, in ONE transaction whose
    /// first statement is the allocating UPDATE (see the module doc). The
    /// row's `manifest_path` is `{table}__v{N}.version.json` beside the
    /// table's Parquet, its `tenant_id` the table row's own, its
    /// `parent_version` the table's `current_version` at allocation. A
    /// zero-row UPDATE (the table is not `ready`, gone, or another tenant's)
    /// rolls back and is classified by `classify_ready_cas_miss`.
    pub async fn allocate_result_table_version(
        &self,
        table: &str,
        writer_id: &str,
        lease: Duration,
    ) -> Result<AllocatedVersion> {
        let table_name = table.to_string();
        let writer = writer_id.to_string();
        let created_at = crate::catalog::backend::now_sortable();
        let tenant = self.current_tenant();
        let arm = TenantArm::in_force(tenant);
        let arm_in_tx = arm.clone();
        let kind = self.backend().backend_kind();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(table_name.clone())];
                    let arm_sql = tenant_arm_sql(&arm_in_tx, "", &mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "UPDATE result_tables SET next_version = next_version + 1 \
                                 WHERE table_name = $1 AND status = 'ready'{arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    if affected != 1 {
                        return Err(BackendError::Busy(MISS_TABLE.to_string()));
                    }
                    let (next, current, parquet_path, owner_tenant): (
                        i64,
                        Option<i64>,
                        String,
                        Option<String>,
                    ) = tx
                        .query_opt(
                            "SELECT next_version, current_version, parquet_path, tenant_id \
                             FROM result_tables WHERE table_name = $1",
                            &[SqlValue::TextOwned(table_name.clone())],
                            |row| {
                                Ok((
                                    i64::from(row.get::<i32>("next_version")?),
                                    row.try_get::<i32>("current_version")?.map(i64::from),
                                    row.get("parquet_path")?,
                                    row.try_get("tenant_id")?,
                                ))
                            },
                        )
                        .await?
                        .ok_or_else(|| BackendError::Busy(MISS_TABLE.to_string()))?;
                    let version = next - 1;
                    let parquet_url = StorageUrl::parse(&parquet_path).map_err(|e| {
                        BackendError::Execution(format!(
                            "result table '{table_name}': parquet_path is not a storage URL: {e}"
                        ))
                    })?;
                    let manifest_path =
                        crate::store::layout::version_manifest_url(&parquet_url, version)
                            .map_err(|e| BackendError::Execution(e.to_string()))?
                            .as_str()
                            .to_string();
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::TextOwned(table_name.clone()),
                        SqlValue::Int(version),
                        SqlValue::from(current),
                        SqlValue::TextOwned(manifest_path.clone()),
                        SqlValue::TextOwned(writer),
                        SqlValue::from(owner_tenant.clone()),
                        SqlValue::TextOwned(created_at),
                    ];
                    let lease_expr = lease_deadline_expr(kind, lease, &mut params);
                    tx.execute(
                        &format!(
                            "INSERT INTO result_table_versions \
                             (table_name, version, parent_version, status, manifest_path, \
                              writer_id, tenant_id, created_at, lease_expires_at) \
                             VALUES ($1, $2, $3, 'building', $4, $5, $6, $7, {lease_expr})"
                        ),
                        &params,
                    )
                    .await?;
                    Ok(AllocatedVersion {
                        version,
                        parent: current,
                        manifest_path,
                        tenant_id: owner_tenant,
                    })
                })
            })
            .await;
        match outcome {
            Ok(v) => Ok(v),
            Err(BackendError::Busy(_)) => {
                let target = self.read_ready_target(table).await?;
                Err(Self::classify_ready_cas_miss(table, &arm, target))
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn read_ready_target(&self, table: &str) -> Result<Option<CasTarget>> {
        let table = table.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { read_cas_target(tx, &table).await }),
            )
            .await?)
    }

    /// The first refresh's base publish (one transaction): the
    /// `current_version IS NULL AND next_version = $B` CAS on the READY table
    /// row (`current_version = B`, `next_version = B + 1`) and the `ready`
    /// version row `B` describing today's table (`identity`, `live_rows`,
    /// `masked_rows = 0`, no parent, the table's tenant). A zero-row CAS
    /// rolls back: a concurrent base publisher won (the caller re-reads
    /// `current_version`), or the row is not `ready` / gone / another
    /// tenant's — classified by `classify_ready_cas_miss`.
    pub async fn publish_base_version(
        &self,
        table: &str,
        version: i64,
        manifest_path: &str,
        identity: &str,
        live_rows: usize,
    ) -> Result<()> {
        let table_name = table.to_string();
        let manifest_path = manifest_path.to_string();
        let identity = identity.to_string();
        let now = crate::catalog::backend::now_sortable();
        let tenant = self.current_tenant();
        let arm = TenantArm::in_force(tenant);
        let arm_in_tx = arm.clone();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::Int(version),
                        SqlValue::Int(version + 1),
                        SqlValue::TextOwned(table_name.clone()),
                    ];
                    let arm_sql = tenant_arm_sql(&arm_in_tx, "", &mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "UPDATE result_tables SET current_version = $1, next_version = $2 \
                                 WHERE table_name = $3 AND status = 'ready' \
                                   AND current_version IS NULL AND next_version = $1{arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    if affected != 1 {
                        return Err(BackendError::Busy(MISS_TABLE.to_string()));
                    }
                    let owner_tenant: Option<String> = tx
                        .query_opt(
                            "SELECT tenant_id FROM result_tables WHERE table_name = $1",
                            &[SqlValue::TextOwned(table_name.clone())],
                            |row| row.try_get::<String>("tenant_id"),
                        )
                        .await?
                        .flatten();
                    tx.execute(
                        "INSERT INTO result_table_versions \
                         (table_name, version, parent_version, status, manifest_path, identity, \
                          live_rows, masked_rows, tenant_id, created_at, completed_at) \
                         VALUES ($1, $2, NULL, 'ready', $3, $4, $5, 0, $6, $7, $7)",
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::Int(version),
                            SqlValue::TextOwned(manifest_path),
                            SqlValue::TextOwned(identity),
                            SqlValue::Int(live_rows as i64),
                            SqlValue::from(owner_tenant),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await;
        match outcome {
            Ok(()) => Ok(()),
            Err(BackendError::Busy(_)) => {
                let target = self.read_ready_target(table).await?;
                Err(Self::classify_ready_cas_miss(table, &arm, target))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The sole commit point of a refresh or compaction (one transaction):
    /// renew the version lease by CAS; flip the version row `building ->
    /// ready` with its identity and counts (lease cleared); swap
    /// `result_tables.current_version` from `parent` to the version, setting
    /// `row_count = live_rows` and the input anchors. Any zero-row statement
    /// rolls the whole transaction back: a version-row miss is classified
    /// like a building-table miss (`LeaseLost` / `CasFailed` / `RowGone` /
    /// `TenantMismatch`); a table-row miss by
    /// `classify_ready_cas_miss` (a concurrent publisher moved
    /// `current_version`, or the row is not `ready`). Nothing is visible
    /// before this commits.
    pub async fn publish_version(&self, p: PublishVersion<'_>) -> Result<()> {
        let cas = p.cas.clone();
        let cas_in_tx = cas.clone();
        let lease = p.lease;
        let parent = p.parent;
        let identity = p.identity.to_string();
        let live_rows = p.live_rows as i64;
        let masked_rows = p.masked_rows as i64;
        let anchors_json = p.anchors_json.to_string();
        let completed_at = completed_now();
        let tenant = self.current_tenant();
        let kind = self.backend().backend_kind();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let cas = cas_in_tx;
                    // 1. renew by CAS
                    let mut params = Vec::new();
                    let expr = lease_deadline_expr(kind, lease, &mut params);
                    let predicate = cas.render(&mut params);
                    let renewed = tx
                        .execute(
                            &format!(
                                "UPDATE result_table_versions SET lease_expires_at = {expr} \
                                 WHERE {predicate}"
                            ),
                            &params,
                        )
                        .await?;
                    if renewed != 1 {
                        return Err(BackendError::Busy(MISS_VERSION.to_string()));
                    }
                    // 2. building -> ready on the version row
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::TextOwned(identity),
                        SqlValue::Int(live_rows),
                        SqlValue::Int(masked_rows),
                        SqlValue::TextOwned(completed_at),
                    ];
                    let predicate = cas.render(&mut params);
                    let flipped = tx
                        .execute(
                            &format!(
                                "UPDATE result_table_versions SET status = 'ready', identity = $1, \
                                 live_rows = $2, masked_rows = $3, completed_at = $4, \
                                 lease_expires_at = NULL WHERE {predicate}"
                            ),
                            &params,
                        )
                        .await?;
                    if flipped != 1 {
                        return Err(BackendError::Busy(MISS_VERSION.to_string()));
                    }
                    // 3. the current-version swap on the READY table row
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::Int(cas.version),
                        SqlValue::Int(live_rows),
                        SqlValue::TextOwned(anchors_json),
                        SqlValue::TextOwned(cas.table.clone()),
                    ];
                    let parent_arm = match parent {
                        Some(pv) => {
                            params.push(SqlValue::Int(pv));
                            format!("current_version = ${}", params.len())
                        }
                        None => "current_version IS NULL".to_string(),
                    };
                    let arm_sql = tenant_arm_sql(&cas.tenant_arm, "", &mut params);
                    let swapped = tx
                        .execute(
                            &format!(
                                "UPDATE result_tables SET current_version = $1, row_count = $2, \
                                 input_anchors_json = $3 WHERE table_name = $4 AND status = 'ready' \
                                 AND {parent_arm}{arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    if swapped != 1 {
                        return Err(BackendError::Busy(MISS_TABLE.to_string()));
                    }
                    Ok(())
                })
            })
            .await;
        match outcome {
            Ok(()) => Ok(()),
            Err(BackendError::Busy(which)) if which == MISS_VERSION => {
                let target = self
                    .read_version_target_owned(&cas.table, cas.version)
                    .await?;
                Err(Self::classify_version_cas_miss(&cas, target))
            }
            Err(BackendError::Busy(_)) => {
                let target = self.read_ready_target(&cas.table).await?;
                Err(Self::classify_ready_cas_miss(
                    &cas.table,
                    &cas.tenant_arm,
                    target,
                ))
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn read_version_target_owned(
        &self,
        table: &str,
        version: i64,
    ) -> Result<Option<VersionTarget>> {
        let table = table.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { read_version_target(tx, &table, version).await }),
            )
            .await?)
    }

    /// Run `set_clause` as a compare-and-set on the `building` version row
    /// `cas` names; `Ok(())` when exactly one row changed, else the classified
    /// typed miss (re-read inside the same transaction).
    async fn building_version_cas(
        &self,
        cas: &VersionCas,
        set_clause: &str,
        set_params: Vec<SqlValue<'static>>,
    ) -> Result<()> {
        let cas_in_tx = cas.clone();
        let set_clause = set_clause.to_string();
        let tenant = self.current_tenant();
        let outcome = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let cas = cas_in_tx;
                    let mut params = set_params;
                    let predicate = cas.render(&mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "UPDATE result_table_versions SET {set_clause} WHERE {predicate}"
                            ),
                            &params,
                        )
                        .await?;
                    if affected == 1 {
                        return Ok(VersionCasOutcome::Applied);
                    }
                    Ok(VersionCasOutcome::Missed(
                        read_version_target(tx, &cas.table, cas.version).await?,
                    ))
                })
            })
            .await?;
        match outcome {
            VersionCasOutcome::Applied => Ok(()),
            VersionCasOutcome::Missed(target) => Err(Self::classify_version_cas_miss(cas, target)),
        }
    }

    /// Renew the lease on the `building` version row `cas` names — the
    /// writer's heartbeat (the lease keeper's arm for
    /// `LeaseTarget::ResultTableVersion`).
    pub async fn renew_version_lease(&self, cas: &VersionCas, lease: Duration) -> Result<()> {
        let kind = self.backend().backend_kind();
        let mut params = Vec::new();
        let expr = lease_deadline_expr(kind, lease, &mut params);
        self.building_version_cas(cas, &format!("lease_expires_at = {expr}"), params)
            .await
    }

    /// The only `building -> failed` transition on a version row: a CAS that
    /// sets `failed`, `completed_at`, and clears the lease (`writer_id` stays
    /// as history). The caller may reap the version's artifacts only after
    /// this returns `Ok(())`.
    pub async fn fail_building_version(&self, cas: &VersionCas) -> Result<()> {
        self.building_version_cas(
            cas,
            "status = 'failed', completed_at = $1, lease_expires_at = NULL",
            vec![SqlValue::TextOwned(completed_now())],
        )
        .await
    }

    /// Recovery's claim on an expired-lease `building` version row: stamps
    /// `writer_id = new_writer_id` and a fresh lease so the recoverer owns
    /// the row before it reaps. `Ok(false)` — never an error — when the claim
    /// matched zero rows (a live writer, or a peer recoverer got there first).
    pub async fn claim_expired_building_version(
        &self,
        table: &str,
        version: i64,
        new_writer_id: &str,
        lease: Duration,
    ) -> Result<bool> {
        let table = table.to_string();
        let new_writer_id = new_writer_id.to_string();
        let tenant = self.current_tenant();
        let arm = TenantArm::in_force(tenant);
        let kind = self.backend().backend_kind();
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(new_writer_id)];
                    let deadline = lease_deadline_expr(kind, lease, &mut params);
                    params.push(SqlValue::TextOwned(table));
                    let t = params.len();
                    params.push(SqlValue::Int(version));
                    let v = params.len();
                    let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                    let arm_sql = tenant_arm_sql(&arm, "", &mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "UPDATE result_table_versions SET writer_id = $1, \
                                 lease_expires_at = {deadline} \
                                 WHERE table_name = ${t} AND version = ${v} \
                                   AND status = 'building' AND {expired}{arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    Ok(affected == 1)
                })
            })
            .await?)
    }

    /// Every `building` version row whose lease is expired (`live = false`) or
    /// live (`true`) at the instant the query runs, on the backend's own
    /// clock. Inside an admin scope every tenant's rows; outside it the
    /// binding's own and GLOBAL rows.
    async fn building_versions_by_lease_liveness(
        &self,
        live: bool,
    ) -> Result<Vec<ResultTableVersionRecord>> {
        let kind = self.backend().backend_kind();
        let admin = TenantBinding::is_admin_scope();
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
                        let mut params: Vec<SqlValue<'static>> = Vec::new();
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        let predicate = if live {
                            format!("NOT {expired}")
                        } else {
                            expired
                        };
                        let scope = if admin {
                            String::new()
                        } else {
                            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                            format!(
                                " AND (tenant_id = ${n} OR tenant_id IS NULL)",
                                n = params.len()
                            )
                        };
                        tx.query(
                            &format!(
                                "SELECT * FROM result_table_versions WHERE status = 'building' \
                                 AND {predicate}{scope} ORDER BY table_name, version"
                            ),
                            &params,
                            parse_version,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Every `building` version row under a LIVE lease — reconcile's
    /// protective set (its deterministic `__v{N}*` keys and `version = N`
    /// segments are referenced, never orphan candidates).
    pub async fn list_live_building_versions(&self) -> Result<Vec<ResultTableVersionRecord>> {
        self.building_versions_by_lease_liveness(true).await
    }

    /// Every `building` version row whose lease has expired — recovery's
    /// claim-then-fail-then-reap set.
    pub async fn list_expired_building_versions(&self) -> Result<Vec<ResultTableVersionRecord>> {
        self.building_versions_by_lease_liveness(false).await
    }

    fn read_scope_sql(params: &mut Vec<SqlValue<'static>>, tenant: Option<TenantId>) -> String {
        if TenantBinding::is_admin_scope() {
            String::new()
        } else {
            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
            format!(
                " AND (tenant_id = ${n} OR tenant_id IS NULL)",
                n = params.len()
            )
        }
    }

    /// One version row by `(table_name, version)`, under the same read-side
    /// tenant filter `get_result_table` applies.
    pub async fn get_result_table_version(
        &self,
        table: &str,
        version: i64,
    ) -> Result<Option<ResultTableVersionRecord>> {
        let table = table.to_string();
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
                        let mut params: Vec<SqlValue<'static>> =
                            vec![SqlValue::TextOwned(table), SqlValue::Int(version)];
                        let scope = Self::read_scope_sql(&mut params, tenant);
                        tx.query_opt(
                            &format!(
                                "SELECT * FROM result_table_versions \
                                 WHERE table_name = $1 AND version = $2{scope}"
                            ),
                            &params,
                            parse_version,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Every version row of `table`, in version order, under the read-side
    /// tenant filter.
    pub async fn list_result_table_versions(
        &self,
        table: &str,
    ) -> Result<Vec<ResultTableVersionRecord>> {
        let table = table.to_string();
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
                        let mut params: Vec<SqlValue<'static>> = vec![SqlValue::TextOwned(table)];
                        let scope = Self::read_scope_sql(&mut params, tenant);
                        tx.query(
                            &format!(
                                "SELECT * FROM result_table_versions WHERE table_name = $1{scope} \
                                 ORDER BY version"
                            ),
                            &params,
                            parse_version,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Delete one terminal (`ready` / `failed`) version row under the STRICT
    /// tenant arm — expiry's catalog half; the caller reaps the artifacts
    /// afterwards. `Ok(false)` when no such row was deleted (still `building`,
    /// absent, or another tenant's). Never touches `next_version`.
    pub async fn delete_result_table_version(&self, table: &str, version: i64) -> Result<bool> {
        let table = table.to_string();
        let tenant = self.current_tenant();
        let arm = TenantArm::in_force(tenant);
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::TextOwned(table), SqlValue::Int(version)];
                    let arm_sql = tenant_arm_sql(&arm, "", &mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "DELETE FROM result_table_versions WHERE table_name = $1 \
                                 AND version = $2 AND status IN ('ready', 'failed'){arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    Ok(affected == 1)
                })
            })
            .await?)
    }

    /// Recovery's `ready -> failed` on a version row whose manifest is
    /// definitively absent (D14(i)): a status-arm CAS under the tenant arm in
    /// force; `Ok(false)` when it matched nothing. The table row and
    /// `current_version` are untouched.
    pub async fn fail_ready_version(&self, table: &str, version: i64) -> Result<bool> {
        let table = table.to_string();
        let tenant = self.current_tenant();
        let arm = TenantArm::in_force(tenant);
        let completed_at = completed_now();
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let mut params: Vec<SqlValue<'static>> = vec![
                        SqlValue::TextOwned(completed_at),
                        SqlValue::TextOwned(table),
                        SqlValue::Int(version),
                    ];
                    let arm_sql = tenant_arm_sql(&arm, "", &mut params);
                    let affected = tx
                        .execute(
                            &format!(
                                "UPDATE result_table_versions SET status = 'failed', \
                                 completed_at = $1 WHERE table_name = $2 AND version = $3 \
                                 AND status = 'ready'{arm_sql}"
                            ),
                            &params,
                        )
                        .await?;
                    Ok(affected == 1)
                })
            })
            .await?)
    }

    /// Test-only: force the `building` version row `cas` names into an
    /// already-expired lease on the catalog backend's own clock, the version
    /// peer of `expire_lease_for_test`.
    #[cfg(feature = "test-hooks")]
    pub async fn expire_version_lease_for_test(&self, cas: &VersionCas) -> Result<()> {
        use crate::catalog::backend::BackendKind;
        let kind = self.backend().backend_kind();
        let (set_clause, set_params): (String, Vec<SqlValue<'static>>) = match kind {
            BackendKind::Postgres => (
                "lease_expires_at = (now() - interval '1 second')::text".to_string(),
                Vec::new(),
            ),
            BackendKind::Sqlite => {
                let past = (chrono::Utc::now() - chrono::Duration::seconds(1))
                    .format(crate::catalog::lease::LEASE_TS_FORMAT)
                    .to_string();
                (
                    "lease_expires_at = $1".to_string(),
                    vec![SqlValue::TextOwned(past)],
                )
            }
        };
        self.building_version_cas(cas, &set_clause, set_params)
            .await
    }
}
