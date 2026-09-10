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
//! renewed by registering with the process's [`crate::catalog::lease_keeper::LeaseKeeper`] (N3), and routes
//! every transition on the row through the [`ResultTableCas`] predicate naming
//! that writer, so a peer's recovery (which touches only rows whose lease is
//! absent or expired) and a live writer can never both act on one row.
//!
//! Renewal moved off a per-table `tokio::spawn` heartbeat task onto the
//! shared keeper thread because a heartbeat task competes for the same main
//! runtime worker threads an inline compute job's CPU-bound work occupies —
//! N+1 such jobs on an N-thread runtime can starve the heartbeat task for the
//! whole duration of the blocking work, letting a live writer's lease expire
//! and be reclaimed out from under it. The keeper's dedicated thread and
//! dedicated connection cannot be starved by that contention.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use tracing::warn;

use crate::catalog::lease_keeper::{LeaseTarget, Registration};
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

/// A `building` result table owned by this process under a leased row.
///
/// Every catalog transition the handle performs is a compare-and-set on
/// `(table_name, writer_id, status = 'building')` with the tenant arm the
/// binding in force selects ([`crate::catalog::result_repo::TenantArm`]).
/// When [`ResultStore`] carries a [`crate::catalog::lease_keeper::LeaseKeeper`] handle, this table's row is
/// one of that keeper's registrations: a renew that matches zero rows
/// (recovery claimed the row after the lease expired, or the row went
/// terminal underneath the writer) flips [`Self::is_live`] to `false` — a
/// writer checks it at each batch boundary and aborts. With no keeper
/// attached (a test fixture; see [`ResultStore::with_lease_keeper`]'s doc)
/// nothing renews the row proactively and [`Self::is_live`] reports `true`
/// until [`Self::finish`]/[`Self::abort`]/drop — every CAS this handle issues
/// still fails loudly (`JammiError::CasFailed`/`RowGone`/`LeaseLost`) if the
/// row's lease has genuinely expired underneath it, so correctness never
/// depends on proactive detection, only the "abort early, before more work"
/// optimisation does.
///
/// Dropping the handle without [`Self::finish`], [`Self::abort`], or
/// [`Self::detach`] unregisters the keeper registration (if any) and, when a
/// tokio runtime is current, spawns a best-effort `building -> failed` CAS
/// (no byte deletion — reconcile reaps the objects later); with no runtime
/// the lease simply expires and recovery reaps the row. After
/// `finish`/`abort`/`detach` the drop is a no-op by construction: the
/// `status = 'building'` predicate no longer matches, or (`detach`) the
/// registration is already gone and no CAS is issued at all.
pub struct BuildingTable {
    store: ResultStore,
    table_name: String,
    parquet_url: StorageUrl,
    tenant: Option<TenantId>,
    writer_id: String,
    storage_precision: StoragePrecision,
    done: Arc<AtomicBool>,
    registration: Option<Registration>,
}

impl std::fmt::Debug for BuildingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildingTable")
            .field("table_name", &self.table_name)
            .field("parquet_url", &self.parquet_url)
            .field("tenant", &self.tenant)
            .field("writer_id", &self.writer_id)
            .field("done", &self.done.load(Ordering::SeqCst))
            .field("lost", &self.registration.as_ref().map(Registration::lost))
            .finish()
    }
}

impl BuildingTable {
    /// Take ownership of the `building` row `table_name` for `writer_id` and,
    /// when `store` carries a [`crate::catalog::lease_keeper::LeaseKeeper`], register it for renewal.
    /// Called by [`ResultStore::create_table`] right after the row's INSERT,
    /// and by recovery right after
    /// [`crate::catalog::Catalog::claim_expired_building_table`] stamped the
    /// recoverer's id on an expired-lease row. The lease the caller stamped
    /// must have been `store.lease_intervals().lease()` FROM NOW (the
    /// catalog backend's own clock on Postgres —
    /// [`crate::catalog::lease::lease_deadline_expr`]), the same window the
    /// keeper renews to.
    pub(crate) fn adopt(
        store: ResultStore,
        table_name: String,
        parquet_url: StorageUrl,
        tenant: Option<TenantId>,
        writer_id: String,
        storage_precision: StoragePrecision,
    ) -> Self {
        let registration = store.lease_keeper().map(|keeper| {
            keeper.register(LeaseTarget::ResultTable {
                table: table_name.clone(),
                writer_id: writer_id.clone(),
            })
        });
        Self {
            store,
            table_name,
            parquet_url,
            tenant,
            writer_id,
            storage_precision,
            done: Arc::new(AtomicBool::new(false)),
            registration,
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

    /// `true` while the keeper registration (if any) still owns the lease and
    /// the handle has not finished/aborted/detached. Flips to `false` the
    /// first time the keeper's renew matches zero rows (the row was claimed
    /// by recovery or went terminal underneath the writer). A writer checks
    /// this at each batch boundary and aborts with [`JammiError::LeaseLost`]
    /// when it is false, rather than streaming bytes into a table it no
    /// longer owns. With no keeper attached this is `true` until
    /// finish/abort/detach — see the struct doc's trade-off.
    pub fn is_live(&self) -> bool {
        if self.done.load(Ordering::SeqCst) {
            return false;
        }
        !self
            .registration
            .as_ref()
            .map(Registration::lost)
            .unwrap_or(false)
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
        if let Err(e) = catalog
            .renew_lease(&cas, self.store.lease_intervals().lease())
            .await
        {
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
        // under this writer: unregister from the keeper and make Drop a no-op.
        self.done.store(true, Ordering::SeqCst);
        self.unregister();
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
    ///
    /// Every object is attempted independently
    /// (`ResultStore::delete_objects_after_cas` never lets one failure
    /// suppress an attempt at the others), but a PARTIAL failure is never
    /// silently turned into `Ok`: this fails loudly whenever the delete's own
    /// `DeletionOutcome::errored` is non-empty — a REAL
    /// `delete_if_exists` I/O failure, never a mere 404 — naming every such
    /// key, so this doc comment's "delete the Parquet ... every ANN segment
    /// bundle" is CHECKED, not merely asserted. This deliberately does NOT
    /// diff against `ResultStore::reap_candidate_keys`'s full candidate
    /// superset: that superset intentionally enumerates every POSSIBLE
    /// sidecar extension regardless of this row's actual precision (e.g. a
    /// `.threshold` companion no `F32` table ever writes), so most of it is
    /// legitimately [`crate::storage::DeleteOutcome::Absent`] and never a
    /// failure — `errored` already carries exactly (and only) the keys whose
    /// delete attempt hit a real error, with no need to separately compute
    /// what "should" have existed. The row itself is still `failed` either
    /// way — only the byte cleanup is incomplete, left for `reconcile` to
    /// retry.
    pub async fn abort(mut self) -> Result<()> {
        self.done.store(true, Ordering::SeqCst);
        self.unregister();
        let cas = self.cas();
        self.store.catalog().fail_building_table(&cas).await?;
        let outcome = self
            .store
            .delete_objects_after_cas(&self.parquet_url, &cas)
            .await?;
        if outcome.errored.is_empty() {
            Ok(())
        } else {
            Err(JammiError::Other(format!(
                "abort: {} object delete(s) failed for '{}': {:?}",
                outcome.errored.len(),
                self.table_name,
                outcome.errored
            )))
        }
    }

    /// Detach the handle from its row with NO catalog transition (N2): stop
    /// renewing (unregister from the keeper), mark the handle done, issue no
    /// CAS, delete nothing. The state a job that lost its lease — or whose
    /// attempt was superseded by a reclaim — leaves behind: the row stays
    /// `building` under this writer's id with a lease that then simply
    /// expires (or, if already expired, is immediately reclaimable). Drop
    /// marks nothing further. A recovery sweep after the lease expires
    /// reconciles the row exactly as it would a dead writer's; the successor
    /// attempt that reclaims it adopts or fails it (N1).
    ///
    /// Distinct from [`Self::abort`] (a CAS-fail-and-delete, the OWNER's own
    /// decision that its output should never exist) — `detach` is for the
    /// caller that is no longer sure it IS the owner (lease lost, attempts
    /// mismatch) and so must not act as one: no CAS, because a CAS run by a
    /// non-owner risks nothing structurally, but issuing ANY write here would
    /// contradict the premise that this caller no longer has standing to
    /// decide the row's fate.
    pub fn detach(mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.unregister();
    }

    /// Test-only: detach the handle AND force the row's lease into the past,
    /// so a recovery sweep run immediately afterwards treats the row as a dead
    /// writer's. Runs the lease rewrite as this writer's own CAS, so it fails
    /// loudly (rather than silently leaving a live lease) if the row is not
    /// this writer's `building` row. The rewrite uses the catalog backend's
    /// OWN clock ([`crate::catalog::Catalog::expire_lease_for_test`]), never
    /// this process's — so the row is expired against exactly the clock
    /// [`crate::catalog::lease::lease_expired_clause`] later compares it
    /// with.
    #[cfg(feature = "test-hooks")]
    pub async fn into_detached(self) -> Result<()> {
        let cas = self.cas();
        let catalog = Arc::clone(self.store.catalog());
        self.detach();
        catalog.expire_lease_for_test(&cas).await
    }

    /// Drop the keeper registration (if any), stopping renewal. Mirrors the
    /// old `stop_heartbeat`'s name at every call site above; the mechanism is
    /// now "drop the `Registration`" rather than "abort a `JoinHandle`".
    fn unregister(&mut self) {
        self.registration.take();
    }
}

impl Drop for BuildingTable {
    fn drop(&mut self) {
        let already_lost = self
            .registration
            .as_ref()
            .map(Registration::lost)
            .unwrap_or(false);
        self.unregister();
        if self.done.load(Ordering::SeqCst) || already_lost {
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
