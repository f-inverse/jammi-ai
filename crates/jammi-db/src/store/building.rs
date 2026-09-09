//! A lease-owned `building` result table — the handle a writer holds from
//! [`ResultStore::create_table`] to [`BuildingTable::finish`] or
//! [`BuildingTable::abort`].
//!
//! The catalog row a writer creates is published in two steps — bytes first,
//! then a single row flip `building -> ready` — and until the lease landed
//! nothing on the row said who was producing it or whether they were alive.
//! Startup recovery in a peer process therefore reaped a live writer's
//! `building` row, and the writer's unguarded promote then flipped the reaped
//! row to `ready` over deleted bytes (esc-094, issue #479). This handle is the
//! writer's side of the fix: it owns the row's `writer_id`, keeps the lease
//! renewed from a background heartbeat, and routes every transition on the
//! row through the [`ResultTableCas`] predicate naming that writer, so a peer's
//! recovery (which touches only rows whose lease is absent or expired) and a
//! live writer can never both act on one row.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use tracing::warn;

use crate::catalog::lease::{lease_deadline, LeaseIntervals};
use crate::catalog::result_repo::{ResultTableCas, ResultTableRecord};
use crate::catalog::status::ResultTableStatus;
use crate::config::StoragePrecision;
use crate::error::{JammiError, Result};
use crate::index::segment::SegmentId;
use crate::index::sidecar::SidecarIndex;
use crate::storage::StorageUrl;
use crate::store::manifest::Materialization;
use crate::store::ResultStore;
use crate::tenant::TenantId;

/// A `building` result table owned by this process under a heartbeated lease.
///
/// Every catalog transition the handle performs is a compare-and-set on
/// `(table_name, writer_id, status = 'building')` with the tenant arm the
/// binding in force selects ([`crate::catalog::result_repo::TenantArm`]). The
/// heartbeat renews the lease every `heartbeat`; a renew that matches zero
/// rows (recovery claimed the row after the lease expired, or the row went
/// terminal underneath the writer) flips [`Self::is_live`] to `false` and
/// stops beating — a writer checks it at each batch boundary and aborts.
///
/// Dropping the handle without [`Self::finish`] or [`Self::abort`] aborts the
/// heartbeat and, when a tokio runtime is current, spawns a best-effort
/// `building -> failed` CAS (no byte deletion — reconcile reaps the objects
/// later); with no runtime the lease simply expires and recovery reaps the
/// row. After `finish` / `abort` the drop is a no-op by construction: the
/// `status = 'building'` predicate no longer matches.
pub struct BuildingTable {
    store: ResultStore,
    table_name: String,
    parquet_url: StorageUrl,
    tenant: Option<TenantId>,
    writer_id: String,
    storage_precision: StoragePrecision,
    done: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
    heartbeat: Option<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for BuildingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildingTable")
            .field("table_name", &self.table_name)
            .field("parquet_url", &self.parquet_url)
            .field("tenant", &self.tenant)
            .field("writer_id", &self.writer_id)
            .field("done", &self.done.load(Ordering::SeqCst))
            .field("lost", &self.lost.load(Ordering::SeqCst))
            .finish()
    }
}

impl BuildingTable {
    /// Take ownership of the `building` row `table_name` for `writer_id` and
    /// start its heartbeat. Called by [`ResultStore::create_table`] right after
    /// the row's INSERT, and by recovery right after
    /// [`crate::catalog::Catalog::claim_expired_building_table`] stamped the
    /// recoverer's id on an expired-lease row. The lease the caller stamped
    /// must have been `lease_deadline(intervals.lease())`, the same window the
    /// heartbeat renews to.
    pub(crate) fn adopt(
        store: ResultStore,
        table_name: String,
        parquet_url: StorageUrl,
        tenant: Option<TenantId>,
        writer_id: String,
        storage_precision: StoragePrecision,
        intervals: LeaseIntervals,
    ) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let lost = Arc::new(AtomicBool::new(false));
        let heartbeat = Some(spawn_heartbeat(
            store.clone(),
            table_name.clone(),
            tenant,
            writer_id.clone(),
            intervals,
            Arc::clone(&done),
            Arc::clone(&lost),
        ));
        Self {
            store,
            table_name,
            parquet_url,
            tenant,
            writer_id,
            storage_precision,
            done,
            lost,
            heartbeat,
        }
    }

    /// The table's catalog name (the `result_tables` primary key).
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Storage URL of the table's Parquet object — open via
    /// [`ResultStore::open_parquet`] / [`ResultStore::open_writer`].
    pub fn parquet_url(&self) -> &StorageUrl {
        &self.parquet_url
    }

    /// The writer id stamped on the row (`writer-{uuid}`, the owning
    /// [`ResultStore`]'s).
    pub fn writer_id(&self) -> &str {
        &self.writer_id
    }

    /// The tenant the row was created under (`None` for a GLOBAL table),
    /// captured once at creation so every later CAS — including one from the
    /// heartbeat task, which has no task-local tenant scope — names it.
    pub fn tenant(&self) -> Option<TenantId> {
        self.tenant
    }

    /// The precision the row was stamped with at creation; the precision every
    /// segment appended to it must be built at.
    pub fn storage_precision(&self) -> StoragePrecision {
        self.storage_precision
    }

    /// `true` while the heartbeat still owns the lease. Flips to `false` the
    /// first time a renew matches zero rows (the row was claimed by recovery
    /// or went terminal underneath the writer). A writer checks this at each
    /// batch boundary and aborts with [`JammiError::LeaseLost`] when it is
    /// false, rather than streaming bytes into a table it no longer owns.
    pub fn is_live(&self) -> bool {
        !self.lost.load(Ordering::SeqCst) && !self.done.load(Ordering::SeqCst)
    }

    /// The compare-and-set predicate for this writer's row under the tenant
    /// arm the binding in force selects.
    pub fn cas(&self) -> ResultTableCas {
        ResultTableCas::writer(&self.table_name, &self.writer_id, self.tenant)
    }

    /// The typed error a lost lease surfaces as.
    fn lease_lost(&self) -> JammiError {
        JammiError::LeaseLost {
            table: self.table_name.clone(),
        }
    }

    /// Persist a checkpoint (batch number) on the row.
    pub async fn set_checkpoint(&self, batch: usize) -> Result<()> {
        self.store
            .catalog()
            .set_checkpoint(&self.cas(), batch)
            .await
    }

    /// Persist a fully-built [`SidecarIndex`] as a NEW immutable segment of
    /// this table's ANN index and register it under this writer's ownership,
    /// returning the allocated [`SegmentId`]. See [`ResultStore::append_segment`].
    pub async fn append_segment(&self, index: &SidecarIndex) -> Result<SegmentId> {
        if !self.is_live() {
            return Err(self.lease_lost());
        }
        self.store.append_segment(self, index).await
    }

    /// Finish the table behind its materialization contract: the single
    /// `building -> ready` transition every producer routes through.
    ///
    /// In this crash-safe order:
    ///
    /// 1. **renew the lease** by CAS — a writer whose lease was claimed by
    ///    recovery learns it here, before it writes an attestation over bytes
    ///    it no longer owns (K7: the sidecar is written only after a
    ///    successful renew);
    /// 2. [`ResultStore::write_attestation`] — compute the artifact digest
    ///    over the durable Parquet bytes, build the
    ///    [`crate::store::manifest::MaterializationManifest`], write the
    ///    `.materialization.json` sidecar;
    /// 3. [`crate::catalog::Catalog::promote_result_table_with_manifest`] —
    ///    the CAS that flips `building -> ready`, persists the summary columns,
    ///    and clears the lease; and
    /// 4. [`ResultStore::register_table`] in DataFusion under the row's own
    ///    catalog owner.
    ///
    /// A promote that misses with [`JammiError::CasFailed`] and
    /// `status = ready` means recovery promoted this writer's bytes (the lease
    /// expired after the sidecar landed): the outcome is correct, only the
    /// writer's summary write was superseded, so the table is still
    /// registered and the catalog's record returned. Every other miss is
    /// returned as its typed error and deletes nothing — the row's new owner
    /// (or reconcile) is responsible for the bytes.
    ///
    /// The `materialization` test point parks between steps 1 and 2.
    pub async fn finish(
        mut self,
        ctx: &SessionContext,
        rows: usize,
        materialization: Materialization<'_>,
    ) -> Result<ResultTableRecord> {
        if !self.is_live() {
            self.done.store(true, Ordering::SeqCst);
            return Err(self.lease_lost());
        }
        let cas = self.cas();
        let catalog = Arc::clone(self.store.catalog());
        let until = lease_deadline(self.store.lease_intervals().lease());
        if let Err(e) = catalog.renew_lease(&cas, &until).await {
            self.done.store(true, Ordering::SeqCst);
            return Err(e);
        }

        // Crash window the contract must survive: the Parquet is durable but
        // the manifest is not yet written and the status flip has not
        // committed.
        #[cfg(feature = "test-hooks")]
        crate::store::mutable::test_hook::maybe_signal_materialization(&self.writer_id).await;

        let (manifest, anchors_json) = self
            .store
            .write_attestation(&self.parquet_url, materialization)
            .await?;

        let promoted = catalog
            .promote_result_table_with_manifest(
                &cas,
                rows,
                manifest.definition_hash.as_str(),
                &anchors_json,
            )
            .await;
        // Whatever the promote said, this handle's row is no longer `building`
        // under this writer: stop the heartbeat and make Drop a no-op.
        self.done.store(true, Ordering::SeqCst);
        self.stop_heartbeat();
        let owner = match promoted {
            Ok(owner) => owner,
            Err(JammiError::CasFailed { status, .. })
                if status == ResultTableStatus::Ready.to_string() =>
            {
                // Recovery promoted this writer's own bytes after the lease
                // expired; the row's owner is the tenant the row carries.
                warn!(
                    table = self.table_name,
                    "finish: the row was already promoted by recovery; registering it as is"
                );
                self.tenant
            }
            Err(e) => return Err(e),
        };
        self.store
            .register_table(ctx, &self.table_name, &self.parquet_url, owner)
            .await?;
        self.store
            .catalog()
            .get_result_table(&self.table_name)
            .await?
            .ok_or_else(|| JammiError::RowGone {
                table: self.table_name.clone(),
            })
    }

    /// Abort the table: the CAS `building -> failed` under this writer's
    /// ownership, and — ONLY if that CAS affected exactly one row — delete the
    /// Parquet, its `.materialization.json` sidecar, and every ANN segment
    /// bundle plus their catalog rows. A miss is returned as its typed error
    /// and deletes nothing: the row's new owner (recovery), or reconcile, is
    /// responsible for the bytes.
    pub async fn abort(mut self) -> Result<()> {
        self.done.store(true, Ordering::SeqCst);
        self.stop_heartbeat();
        let cas = self.cas();
        self.store.catalog().fail_building_table(&cas).await?;
        self.store
            .delete_objects_after_cas(&self.parquet_url, &cas)
            .await
    }

    /// Detach the handle from its row with no catalog transition — the state
    /// a `SIGKILL` leaves: the heartbeat stops, the row stays `building` under
    /// this writer's id with a lease that then simply expires. Drop marks
    /// nothing. A recovery sweep after the lease expires reconciles the row
    /// exactly as it would a dead writer's.
    pub fn abandon(mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.stop_heartbeat();
    }

    /// Test-only: abandon the handle AND force the row's lease into the past,
    /// so a recovery sweep run immediately afterwards treats the row as a dead
    /// writer's. Runs the lease rewrite as this writer's own CAS, so it fails
    /// loudly (rather than silently leaving a live lease) if the row is not
    /// this writer's `building` row.
    #[cfg(feature = "test-hooks")]
    pub async fn into_abandoned(self) -> Result<()> {
        let cas = self.cas();
        let catalog = Arc::clone(self.store.catalog());
        self.abandon();
        catalog
            .renew_lease(&cas, "1970-01-01T00:00:00.000000Z")
            .await
    }

    fn stop_heartbeat(&mut self) {
        if let Some(handle) = self.heartbeat.take() {
            handle.abort();
        }
    }
}

impl Drop for BuildingTable {
    fn drop(&mut self) {
        self.stop_heartbeat();
        if self.done.load(Ordering::SeqCst) || self.lost.load(Ordering::SeqCst) {
            return;
        }
        // Dropped without finish/abort — an error path unwound through `?`.
        // Best-effort `building -> failed` under this writer's own CAS, no
        // byte deletion (reconcile reaps the objects). Without a runtime the
        // lease expires and recovery reaps the row.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let catalog = Arc::clone(self.store.catalog());
        let cas = self.cas();
        let table = self.table_name.clone();
        handle.spawn(async move {
            if let Err(e) = catalog.fail_building_table(&cas).await {
                warn!(
                    table,
                    error = %e,
                    "BuildingTable dropped without finish/abort: mark-failed CAS did not apply"
                );
            }
        });
    }
}

/// Spawn the lease-renewing heartbeat task (the shape of the training
/// worker's). It renews by CAS on the configured interval; the first miss
/// sets `lost` and stops; a backend error is logged and the task keeps
/// beating (the lease may still be renewed before it expires). Stops when
/// `done` is set.
fn spawn_heartbeat(
    store: ResultStore,
    table_name: String,
    tenant: Option<TenantId>,
    writer_id: String,
    intervals: LeaseIntervals,
    done: Arc<AtomicBool>,
    lost: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(intervals.heartbeat()).await;
            if done.load(Ordering::SeqCst) {
                return;
            }
            let cas = ResultTableCas::writer(&table_name, &writer_id, tenant);
            let until = lease_deadline(intervals.lease());
            match store.catalog().renew_lease(&cas, &until).await {
                Ok(()) => {}
                Err(
                    JammiError::RowGone { .. }
                    | JammiError::TenantMismatch { .. }
                    | JammiError::LeaseLost { .. }
                    | JammiError::CasFailed { .. },
                ) => {
                    lost.store(true, Ordering::SeqCst);
                    return;
                }
                Err(e) => {
                    tracing::error!(table = table_name, error = %e, "result-table heartbeat failed");
                }
            }
        }
    })
}
