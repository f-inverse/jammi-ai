//! A lease-owned `building` version of a ready result table — the handle a
//! refresh or compaction holds from [`ResultStore::allocate_version`] to its
//! publish or [`BuildingVersion::abort`].
//!
//! The version peer of [`crate::store::BuildingTable`]: it owns a
//! `result_table_versions` row's `writer_id`, keeps the lease renewed on the
//! process's [`crate::catalog::lease_keeper::LeaseKeeper`]
//! (`LeaseTarget::ResultTableVersion`), routes every transition through the
//! [`VersionCas`] naming that writer, and stamps every segment it appends
//! with its version number. The table's own row is never touched: a versioned
//! table never re-enters `building` (I-A2), and nothing this handle writes is
//! visible before [`crate::catalog::Catalog::publish_version`]'s single
//! transaction.
//!
//! Dropping the handle without a publish, [`BuildingVersion::abort`] or
//! [`BuildingVersion::detach`] spawns a best-effort `building -> failed` CAS on the
//! VERSION row with no byte deletion (recovery reaps the artifacts stamped
//! with this number once the lease expires); `abort` fails the row by CAS and
//! then reaps exactly the artifacts stamped with this version — never the base
//! Parquet, its manifest, or a base segment.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tracing::warn;

use crate::catalog::lease_keeper::{LeaseHold, LeaseTarget};
use crate::catalog::version_repo::VersionCas;
use crate::config::StoragePrecision;
use crate::error::{JammiError, Result};
use crate::index::segment::SegmentId;
use crate::index::sidecar::SidecarIndex;
use crate::storage::StorageUrl;
use crate::store::{layout, ResultStore};
use crate::tenant::TenantId;

/// A `building` version row owned by this process under a leased row.
pub struct BuildingVersion {
    store: ResultStore,
    table_name: String,
    parquet_url: StorageUrl,
    version: i64,
    parent: Option<i64>,
    manifest_url: StorageUrl,
    tenant: Option<TenantId>,
    writer_id: String,
    storage_precision: StoragePrecision,
    done: Arc<AtomicBool>,
    hold: Option<LeaseHold>,
}

impl std::fmt::Debug for BuildingVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildingVersion")
            .field("table_name", &self.table_name)
            .field("version", &self.version)
            .field("parent", &self.parent)
            .field("writer_id", &self.writer_id)
            .field("done", &self.done.load(Ordering::SeqCst))
            .field("lost", &self.hold.as_ref().map(LeaseHold::lost))
            .finish()
    }
}

impl BuildingVersion {
    /// Take ownership of the `building` version row `(table_name, version)`
    /// for `writer_id`, holding it open for renewal when the store carries a
    /// lease keeper. Called by [`ResultStore::allocate_version`] right after
    /// the row's INSERT.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adopt(
        store: ResultStore,
        table_name: String,
        parquet_url: StorageUrl,
        version: i64,
        parent: Option<i64>,
        manifest_url: StorageUrl,
        tenant: Option<TenantId>,
        writer_id: String,
        storage_precision: StoragePrecision,
    ) -> Self {
        let hold = store.lease_keeper().map(|keeper| {
            keeper.hold(LeaseTarget::ResultTableVersion {
                table: table_name.clone(),
                version,
                writer_id: writer_id.clone(),
            })
        });
        Self {
            store,
            table_name,
            parquet_url,
            version,
            parent,
            manifest_url,
            tenant,
            writer_id,
            storage_precision,
            done: Arc::new(AtomicBool::new(false)),
            hold,
        }
    }

    /// The table's catalog name.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// The allocated version number `N`.
    pub fn version(&self) -> i64 {
        self.version
    }

    /// The parent version this one refreshes from.
    pub fn parent_version(&self) -> Option<i64> {
        self.parent
    }

    /// The table's base Parquet URL (never written by this handle).
    pub fn parquet_url(&self) -> &StorageUrl {
        &self.parquet_url
    }

    /// `{table}__v{N}.version.json`.
    pub fn manifest_url(&self) -> &StorageUrl {
        &self.manifest_url
    }

    /// `{table}__v{N}.parquet` — this version's data fragment.
    pub fn fragment_url(&self) -> Result<StorageUrl> {
        layout::version_fragment_url(&self.parquet_url, self.version)
    }

    /// `{table}__v{N}.deletes.parquet` — this version's cumulative mask.
    pub fn deletes_url(&self) -> Result<StorageUrl> {
        layout::version_deletes_url(&self.parquet_url, self.version)
    }

    /// The writer id stamped on the row.
    pub fn writer_id(&self) -> &str {
        &self.writer_id
    }

    /// The tenant the table row carries (`None` for GLOBAL).
    pub fn tenant(&self) -> Option<TenantId> {
        self.tenant
    }

    /// The precision every segment appended to this version must be built at
    /// (the table's persisted precision).
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// `true` while the keeper hold (if any) still owns the lease and the
    /// handle has not published/aborted/detached.
    pub fn is_live(&self) -> bool {
        if self.done.load(Ordering::SeqCst) {
            return false;
        }
        !self.hold.as_ref().map(LeaseHold::lost).unwrap_or(false)
    }

    /// The compare-and-set predicate for this writer's version row under the
    /// tenant arm the binding in force selects.
    pub fn cas(&self) -> VersionCas {
        VersionCas::writer(&self.table_name, self.version, &self.writer_id, self.tenant)
    }

    fn lease_lost(&self) -> JammiError {
        JammiError::LeaseLost {
            table: self.table_name.clone(),
        }
    }

    /// Persist a fully-built [`SidecarIndex`] as a NEW immutable segment
    /// stamped with this version, registered under this writer's lease.
    pub async fn append_segment(&self, index: &SidecarIndex) -> Result<SegmentId> {
        if !self.is_live() {
            return Err(self.lease_lost());
        }
        self.store.append_segment_for_version(self, index).await
    }

    /// The sole commit point (D5): renew-by-CAS, `building -> ready` on the
    /// version row, and the `current_version = parent -> N` swap on the
    /// table row, one transaction ([`crate::catalog::Catalog::publish_version`]).
    /// On success the handle is done (no further renewal, Drop a no-op); on a
    /// typed miss the row is left as the transaction rolled it back and the
    /// caller decides (typically [`Self::abort`]).
    pub async fn publish(
        &mut self,
        identity: &str,
        live_rows: usize,
        masked_rows: usize,
        anchors_json: &str,
    ) -> Result<()> {
        if !self.is_live() {
            return Err(self.lease_lost());
        }
        let cas = self.cas();
        self.store
            .catalog()
            .publish_version(crate::catalog::version_repo::PublishVersion {
                cas: &cas,
                lease: self.store.lease_intervals().lease(),
                parent: self.parent,
                identity,
                live_rows,
                masked_rows,
                anchors_json,
            })
            .await?;
        self.mark_done();
        Ok(())
    }

    /// Mark the handle as published: stop renewing, make Drop a no-op.
    fn mark_done(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.release_hold();
    }

    /// Abort the version: the CAS `building -> failed` on the version row
    /// under this writer's ownership and — ONLY if it applied — reap every
    /// artifact stamped with this version (`__v{N}.parquet`,
    /// `__v{N}.deletes.parquet`, `__v{N}.version.json`, and the `version = N`
    /// segments with their bundles). A miss is returned typed and deletes
    /// nothing. The base Parquet, its manifest and base segments are never
    /// touched.
    pub async fn abort(mut self) -> Result<()> {
        self.done.store(true, Ordering::SeqCst);
        self.release_hold();
        let cas = self.cas();
        self.store.catalog().fail_building_version(&cas).await?;
        let outcome = self
            .store
            .reap_version_artifacts(&self.parquet_url, &self.table_name, self.version)
            .await?;
        if outcome.errored.is_empty() {
            Ok(())
        } else {
            Err(JammiError::Other(format!(
                "abort: {} object delete(s) failed for '{}' version {}: {:?}",
                outcome.errored.len(),
                self.table_name,
                self.version,
                outcome.errored
            )))
        }
    }

    /// Detach with NO catalog transition: stop renewing, issue no CAS, delete
    /// nothing — the row stays `building` under this writer and its lease
    /// simply expires; recovery reaps it as a dead writer's.
    pub fn detach(mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.release_hold();
    }

    /// Test-only: detach AND force the row's lease into the past so a
    /// recovery sweep run immediately afterwards treats it as expired.
    #[cfg(feature = "test-hooks")]
    pub async fn into_detached(self) -> Result<()> {
        let cas = self.cas();
        let catalog = Arc::clone(self.store.catalog());
        self.detach();
        catalog.expire_version_lease_for_test(&cas).await
    }

    fn release_hold(&mut self) {
        self.hold.take();
    }
}

impl Drop for BuildingVersion {
    fn drop(&mut self) {
        let already_lost = self.hold.as_ref().map(LeaseHold::lost).unwrap_or(false);
        self.release_hold();
        if self.done.load(Ordering::SeqCst) || already_lost {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let catalog = Arc::clone(self.store.catalog());
        let cas = self.cas();
        let table = self.table_name.clone();
        let version = self.version;
        handle.spawn(async move {
            if let Err(e) = catalog.fail_building_version(&cas).await {
                warn!(
                    table,
                    version,
                    error = %e,
                    "BuildingVersion dropped without publish/abort: mark-failed CAS did not apply"
                );
            }
        });
    }
}
