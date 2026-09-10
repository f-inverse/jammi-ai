//! Cross-checks the catalog against the object store: [`ResultStore::reconcile`]
//! is the ONE place this engine performs an object-store `LIST` (the never-LIST
//! hot-path rule lives on `crate::storage::JammiObjectStore::list` itself).
//!
//! Two independent checks, both against the SAME listing:
//!
//! - **row → object** (completeness): a `ready` row's contract requires a set
//!   of objects to exist (`ResultStore::required_row_objects_present`); a row whose
//!   required objects are not ALL present — verified with a live `exists()`
//!   per object, never trusted off the listing snapshot — is driven
//!   `ready -> failed` by the same status-guarded CAS
//!   [`reconcile_ready_manifests`](crate::store::ResultStore) already uses.
//! - **object → row** (attribution): every OTHER listed object is either
//!   referenced by a currently-live row (a `ready` row, a live-lease
//!   `building` row, a `running`/`queued` training job, or a model's
//!   `artifact_path`), or it is not — an orphan candidate, aged against
//!   `grace` before `apply` may delete it, or `unattributed` (an object whose
//!   own key does not even parse through the [`crate::store::layout::TenantSegment`]
//!   allowlist) which is reported but NEVER deleted at any grace.
//!
//! **Order matters**: objects are listed FIRST, rows read SECOND — so the
//! row set a listing is checked against is guaranteed to be a superset of
//! every object's true referencer at the moment of listing (a table
//! materialising concurrently with a reconcile pass always has its `building`
//! row visible by the time reconcile reads rows, because the row is written
//! before any byte the row→object check would look for). This holds for the
//! expired-building pre-pass too — it now runs AFTER the listing, exactly
//! like every other row read this pass performs. The pre-pass still
//! physically deletes bytes (under `apply`) for whatever it reaps, but it
//! never double-counts them: it looks each candidate object up in the
//! listing snapshot taken a moment earlier (so it reports the object's TRUE
//! size, not a size implied after the fact) and records every key it
//! accounts for in a `reaped` set; the object→row loop further down skips
//! any key already in `reaped` — structurally, by set membership, not by
//! ordering — so a key can never be counted twice no matter which arm
//! touches it first. Under `apply=false` the pre-pass claims and deletes
//! nothing, but performs the identical read-only classification and reports
//! the SAME objects and sizes `apply=true` would reclaim (see
//! [`ReconcileOptions::apply`]'s pinned dry-run/apply parity invariant).
//!
//! **A promotion is not a reclaim (#484 design revision).** An expired
//! `building` row with a valid Parquet and its manifest sidecar present is
//! PROMOTED, not reaped — an internal `ExpiredRowOutcome::Promote`
//! classification. Promoting an embedding row's rebuild
//! (`ResultStore::rebuild_index_from_parquet`) purges the row's entire
//! CURRENT segment set and rewrites, at most, a
//! single fresh segment `0` at the same key: a stale second segment (an
//! interrupted multi-segment build), or segment `0` itself over a zero-row
//! Parquet, is deleted and never rewritten. Those bytes are real deletions —
//! `purge_segments` truly removes them — but they are the promotion's OWN
//! internal bookkeeping, never a reclaim this pass reports: both dry-run
//! (which protects the row's WHOLE current key set wholesale, predicting
//! nothing about what the rebuild will purge) and apply (which records
//! exactly what its own `purge_segments` call deleted into a per-pass
//! `promoted_purged` exclusion set, rather than crediting it) report NOTHING
//! for a promoted row's stale segment sidecars in THIS pass. A key
//! `purge_segments` FAILS to delete (a real I/O error, never a mere 404)
//! survives on disk, unreferenced, and is never excluded — it falls through
//! to the ordinary age-gated object→row arm below to be reclaimed normally,
//! in this pass or a later one, exactly like any other orphan candidate.

use std::collections::{BTreeSet, HashMap};
use std::str::FromStr;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use uuid::Uuid;

use crate::catalog::result_repo::ResultTableRecord;
use crate::catalog::status::{ResultTableStatus, TrainingJobStatus};
use crate::error::{JammiError, Result};
use crate::storage::sidecar_layout::{
    required_sidecar_extensions, sidecar_extensions, SidecarKind,
};
use crate::storage::{DeleteOutcome, StorageUrl};
use crate::store::layout::{self, TenantSegment};
use crate::store::{ExpiredRowDeletion, ExpiredRowOutcome, ResultStore};
use crate::tenant_scope::TenantBinding;

/// One reconciliation pass's parameters.
#[derive(Debug, Clone, Copy)]
pub struct ReconcileOptions {
    /// `true` to actually delete orphans past `grace` (and flip an
    /// incomplete `ready` row to `failed`); `false` (the default a caller
    /// should reach for first) reports everything a pass WOULD do without
    /// mutating anything.
    ///
    /// **Precondition: `grace >= ` this store's configured lease duration.**
    /// `apply=true` REFUSES below it ([`ResultStore::reconcile`]'s
    /// `check_apply_grace`, called unconditionally by every mode this store
    /// exposes) — a reclaim window shorter than a live writer's lease may
    /// legitimately run under would race a healthy in-progress
    /// materialization. `apply=false` merely PREVIEWS under whatever `grace`
    /// the caller passed, including one below the floor: a dry-run mutates
    /// nothing, so a too-short `grace` is not yet the hazard it becomes the
    /// moment `apply=true` would act on it — but the invariant below still
    /// holds at whatever `grace` both calls share, refused or not.
    ///
    /// **Pinned invariant, narrowed to the no-real-delete-failure case**: for
    /// the same catalog+object-store state and the same
    /// [`ReconcileOptions::grace`], `apply=false` and `apply=true` report the
    /// identical [`ReconcileReport::rows_failed`], [`ReconcileReport::orphans`],
    /// every `*_count` field, and [`ReconcileReport::bytes_reclaimed`] —
    /// `apply=true`'s numbers are what it actually did; `apply=false`'s are
    /// what it would have done. Only [`ReconcileReport::applied`] itself, and
    /// whatever the catalog/object store actually look like afterward, differ
    /// between the two. A dry-run that under-reports what the matching
    /// `apply` pass would reclaim is a bug in the dry-run arm, not a looser
    /// contract for it.
    ///
    /// This equality holds only when `apply=true`'s own deletes all
    /// succeed as [`crate::storage::DeleteOutcome::Deleted`] — a dry-run
    /// cannot foresee either a REAL delete failure (a permissions error, a
    /// backend outage) or a key that VANISHES between the listing this pass
    /// took and the moment `apply=true` tries to delete it (a peer pass, or
    /// this same pass's own expired-building pre-pass, winning the race).
    /// Under `apply=true` a real failure moves that key from `orphans` to
    /// `pending` (left for the next pass to retry) and a vanish drops it from
    /// every field (nothing was reclaimed, so nothing is credited or
    /// retried) — either way `apply=true` never OVER-reports what it actually
    /// freed. `apply=false` has no way to predict either race, so it still
    /// previews the object as an ordinary orphan at its listed size; the two
    /// modes diverge on that one key rather than the parity invariant itself
    /// being violated.
    pub apply: bool,
    /// An orphan candidate younger than this is `pending`, never deleted —
    /// the window a concurrent writer's just-landed bytes have to grow a
    /// referencing row before a pass would otherwise reclaim them.
    /// `apply = true` REQUIRES `grace >= ` the deployment's configured lease
    /// duration ([`ReconcileOptions`] alone cannot express this — the check
    /// is [`ResultStore::reconcile`]'s), which keeps a lease that is merely
    /// running long from being raced by a reclaim UNDER SYNCHRONIZED CLOCKS.
    ///
    /// **The grace gate compares two DIFFERENT clocks, not one.** The age
    /// check (`obj.last_modified <= now - grace`) reads `last_modified` off
    /// the OBJECT STORE (its own clock, wherever the bytes physically live —
    /// a cloud provider's, or the local filesystem's) and compares it
    /// against THIS REPLICA's `Utc::now()` — never the catalog database's
    /// clock the way a lease predicate does (`catalog::lease`'s module
    /// docs). `grace >= lease.duration` is exact only when the object
    /// store's clock and this replica's clock agree; ordinary NTP-level
    /// skew (milliseconds to low seconds in a well-run fleet) erodes the
    /// margin `grace` provides, it does not remove the mechanism — a `grace`
    /// several multiples of the lease duration is the practical guard against
    /// clock skew the same way it already is against slow writers.
    pub grace: Duration,
}

/// A single list inside a [`ReconcileReport`] gets truncated at this many
/// entries so an unbounded object-store listing can never make the report
/// itself unbounded; `*_count` on the report always carries the TRUE total,
/// and `truncated` says whether any list was cut.
pub const REPORT_LIST_CAP: usize = 10_000;

/// The result of one reconciliation pass. Every list is sorted (and capped at
/// [`REPORT_LIST_CAP`] entries; `*_count` fields always carry the true
/// total); `Serialize` so a wire/CLI mapping can hand this back verbatim.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ReconcileReport {
    /// `"_global"` (unbound), `"tenant:{uuid}"` (a tenant-scoped
    /// [`ResultStore::reconcile`]), or `"all"` ([`ResultStore::reconcile_all`]).
    pub scope: String,
    /// Whether this pass actually deleted anything (mirrors
    /// [`ReconcileOptions::apply`]).
    pub applied: bool,
    /// Result tables flipped `ready -> failed` this pass (an incomplete
    /// object set was found for them) — a row whose fail-CAS missed (someone
    /// else already changed it) is never counted here; it stays `still_ready`
    /// and protected. Capped at [`REPORT_LIST_CAP`]; see
    /// [`Self::rows_failed_count`].
    pub rows_failed: Vec<String>,
    /// The true count of rows flipped `ready -> failed` this pass,
    /// independent of whether [`Self::rows_failed`] was truncated.
    pub rows_failed_count: u64,
    /// Every key this pass reclaims (or, under a dry-run, would reclaim),
    /// admitted through ONE of TWO independent routes:
    ///
    /// - **Age-gated candidates**: an unreferenced listed object whose
    ///   `last_modified` is at least `grace` old (the ordinary object→row
    ///   arm, further down this pass).
    /// - **CAS-licensed reaps**: an expired-lease `building` row's objects,
    ///   admitted by `ExpiredRowOutcome::Reap` — the pre-pass's claim-then-
    ///   fail-CAS licenses the reap regardless of the object's own age; a
    ///   torn row's bytes freshly written a second ago are just as reapable
    ///   as one a week stale, because the lease (not the object's mtime) is
    ///   what proves the writer is gone. See [`ResultStore::reconcile`]'s
    ///   `check_apply_grace`, which is why `apply=true` still requires
    ///   `grace >= ` the configured lease duration even though this second
    ///   route never consults `grace` itself: the floor bounds how long a
    ///   HEALTHY writer's lease may run, not how this route ages its
    ///   candidates.
    ///
    /// NEVER a `Promote` row's own segment sidecars, even a stale one its
    /// rebuild purges and does not rewrite: a promotion is not a reclaim (see
    /// this module's own doc comment) — those keys are excluded from this
    /// pass's accounting entirely, not admitted through either route above.
    ///
    /// Capped at [`REPORT_LIST_CAP`]; see [`Self::orphan_count`].
    pub orphans: Vec<String>,
    /// The true count of orphan candidates found this pass, independent of
    /// whether [`Self::orphans`] was truncated.
    pub orphan_count: u64,
    /// Orphan candidates younger than `grace` — never deleted this pass.
    /// Capped at [`REPORT_LIST_CAP`]; see [`Self::pending_count`].
    pub pending: Vec<String>,
    /// The true count of pending candidates found this pass, independent of
    /// whether [`Self::pending`] was truncated.
    pub pending_count: u64,
    /// Listed keys whose own path does not parse through the tenant/artifact
    /// allowlist at all — reported, never deleted at any grace or `apply`.
    /// Reported ONLY by an admin-scoped pass ([`ResultStore::reconcile_all`]);
    /// a tenant-scoped [`ResultStore::reconcile`] never lists another
    /// tenant's (or nobody's) stray keys — a scoped pass reports NOTHING it
    /// cannot attribute to its own prefix. Capped at
    /// [`REPORT_LIST_CAP`]; see [`Self::unattributed_count`].
    pub unattributed: Vec<String>,
    /// The true count of unattributed keys found this pass, independent of
    /// whether [`Self::unattributed`] was truncated.
    pub unattributed_count: u64,
    /// A prefix named by a `models` row (a trained-model artifact bundle)
    /// whose `manifest.json` is absent: the row says this bundle exists, but
    /// its attestation does not — never reclaimed, at any grace or `apply`,
    /// because the referencing row is still live. Distinct from
    /// [`Self::orphans`], whose entries have NO referencing row at all.
    /// Capped at [`REPORT_LIST_CAP`]; see [`Self::damaged_count`].
    pub damaged: Vec<String>,
    /// The true count of damaged keys found this pass, independent of
    /// whether [`Self::damaged`] was truncated.
    pub damaged_count: u64,
    /// `true` iff any of [`Self::rows_failed`], [`Self::orphans`],
    /// [`Self::pending`], [`Self::unattributed`], [`Self::damaged`] was cut
    /// to [`REPORT_LIST_CAP`] entries — the corresponding `*_count` field is
    /// still the true total either way.
    pub truncated: bool,
    /// Total bytes reclaimed by [`Self::orphans`] — under `apply=true`, the
    /// bytes this pass ACTUALLY deleted; under `apply=false`, the bytes the
    /// matching `apply=true` pass, on the identical state, WOULD delete (see
    /// the pinned dry-run/apply parity invariant on
    /// [`ReconcileOptions::apply`]). Never `0` merely because `!applied` —
    /// only because nothing was reclaimable. Counts orphan and reap
    /// deletions ONLY — a promotion's internal rebuild (its `purge_segments`
    /// call clearing a stale segment sibling to make way for a fresh one) is
    /// never counted here, at any grace or `apply`: it is the promotion's own
    /// bookkeeping, not a reclaim this pass performed (see this module's own
    /// doc comment, "a promotion is not a reclaim").
    pub bytes_reclaimed: u64,
}

/// Push `key` onto `list` unless it is already at [`REPORT_LIST_CAP`], always
/// incrementing `*count` and setting `*truncated` on the first entry a list
/// drops — the one helper every capped list in a [`ReconcileReport`] grows
/// through, so the truncation rule cannot drift between lists.
fn push_capped(list: &mut Vec<String>, count: &mut u64, truncated: &mut bool, key: String) {
    *count += 1;
    if list.len() < REPORT_LIST_CAP {
        list.push(key);
    } else {
        *truncated = true;
    }
}

/// Credit exactly the keys in `keys` that are present in `listed_sizes` (the
/// listing snapshot taken before this pass touched anything) into `reaped`
/// and `orphans`/`bytes_reclaimed` — the ONE place the expired-building
/// pre-pass grows those fields, in EITHER mode.
///
/// A key `reaped` already contains (credited by an earlier iteration of the
/// pre-pass loop, or already present in `keys` itself — a `BTreeSet` cannot
/// duplicate, but a defensive re-check costs nothing) is skipped: a key can
/// never be double-counted. A key ABSENT from `keys` — because
/// [`ResultStore::delete_objects_after_cas`] (apply) or
/// [`ResultStore::reap_candidate_keys`] (dry-run) never named it, most
/// commonly because ITS OWN delete failed — is never credited here: this is
/// the "accounting set == deletion set" invariant made concrete. A key
/// present in `keys` but absent from `listed_sizes` (never actually on disk
/// at listing time) is silently skipped too, never credited a phantom size.
fn credit_reaped(
    keys: BTreeSet<String>,
    listed_sizes: &HashMap<&str, u64>,
    reaped: &mut BTreeSet<String>,
    orphans: &mut Vec<String>,
    orphan_count: &mut u64,
    truncated: &mut bool,
    bytes_reclaimed: &mut u64,
) {
    for key in keys {
        let Some(&size) = listed_sizes.get(key.as_str()) else {
            continue;
        };
        if reaped.insert(key.clone()) {
            push_capped(orphans, orphan_count, truncated, key);
            *bytes_reclaimed += size;
        }
    }
}

/// One listed object, in the coordinates every comparison in this module
/// uses: its key RELATIVE to the store's root (no leading `/`), its size,
/// and its last-modified time (the grace clock).
struct Listed {
    rel: String,
    size: u64,
    last_modified: DateTime<Utc>,
}

/// Which allowlist arm a listed key's path attributes to.
enum Attribution {
    /// First segment parses as a [`TenantSegment`] — a result-table key.
    ResultTable,
    /// `models/{seg}/{job}/…` where `seg` parses as a [`TenantSegment`] and
    /// `job` is a canonical v4 UUID string (job ids are minted with
    /// `Uuid::new_v4().to_string()`).
    Artifact { seg: String, job: String },
    /// Neither — a pre-layout key, or genuine garbage.
    Unattributed,
}

fn attribute(rel: &str) -> Attribution {
    let mut parts = rel.splitn(2, '/');
    let first = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("");
    if first == "models" {
        let mut it = rest.splitn(3, '/');
        let seg = it.next().unwrap_or("");
        let job = it.next().unwrap_or("");
        if TenantSegment::parse(seg).is_some() && is_canonical_v4_uuid(job) {
            return Attribution::Artifact {
                seg: seg.to_string(),
                job: job.to_string(),
            };
        }
        return Attribution::Unattributed;
    }
    if TenantSegment::parse(first).is_some() {
        return Attribution::ResultTable;
    }
    Attribution::Unattributed
}

/// `true` iff `s` is exactly the canonical (lowercase, hyphenated) string
/// form of a v4 UUID — the shape `Uuid::new_v4().to_string()` always
/// produces. Rejects every other valid-UUID spelling the same way
/// [`TenantSegment::parse`] rejects a non-canonical tenant segment.
fn is_canonical_v4_uuid(s: &str) -> bool {
    match Uuid::from_str(s) {
        Ok(u) => u.get_version_num() == 4 && u.to_string() == s,
        Err(_) => false,
    }
}

/// `url`'s key relative to `root` (no leading `/`), or `None` if `url` does
/// not share `root`'s prefix at all. `pub(super)`: [`ResultStore::delete_objects_after_cas`],
/// [`ResultStore::purge_segments`], and [`ResultStore::reap_candidate_keys`]
/// (all in `store::mod`) share this SAME coordinate-space helper so the
/// actual deleter and `reconcile`'s accounting can never compute a key in two
/// different ways.
pub(super) fn relative_to(root: &StorageUrl, url: &StorageUrl) -> Option<String> {
    let root_str = root.as_str().trim_end_matches('/');
    url.as_str()
        .strip_prefix(root_str)
        .map(|s| s.trim_start_matches('/').to_string())
}

impl ResultStore {
    /// Reconcile the catalog against the object store, scoped to the
    /// store's own binding: a tenant-bound store reconciles only its own
    /// `{seg}/` and `models/{seg}/` prefixes; an unbound store reconciles
    /// only `_global/` and `models/_global/`. Use [`Self::reconcile_all`] for
    /// a cross-tenant admin pass.
    pub async fn reconcile(&self, opts: ReconcileOptions) -> Result<ReconcileReport> {
        self.check_apply_grace(&opts)?;
        let tenant = self.catalog.current_tenant();
        let own_seg = TenantSegment::of(tenant.as_ref());
        let scope = match tenant {
            Some(t) => format!("tenant:{t}"),
            None => "_global".to_string(),
        };
        self.reconcile_inner(scope, opts, Some(own_seg)).await
    }

    /// Reconcile every tenant's prefixes in one admin pass. Wraps the WHOLE
    /// pass in [`TenantBinding::admin_scope`] — the enumeration of `ready` /
    /// `building` rows, training jobs, and models ALL see every tenant's rows,
    /// matching the cross-tenant object listing.
    pub async fn reconcile_all(&self, opts: ReconcileOptions) -> Result<ReconcileReport> {
        self.check_apply_grace(&opts)?;
        TenantBinding::admin_scope(self.reconcile_inner("all".to_string(), opts, None)).await
    }

    /// `apply = true` requires `grace >= ` this store's configured lease
    /// duration — a reclaim window shorter than the window a live writer's
    /// lease may legitimately run under would race a healthy in-progress
    /// materialization. Called unconditionally at the top of BOTH
    /// [`Self::reconcile`] (tenant-scoped) and [`Self::reconcile_all`]
    /// (admin-scoped) — every mode this store exposes runs through this ONE
    /// gate, never a copy of it. `apply=false` deliberately tolerates any
    /// `grace`: a dry-run's classification of a given object as `orphan` vs
    /// `pending` still uses whatever `grace` the caller passed (that is the
    /// dry-run/apply parity [`ReconcileOptions::apply`] pins), but a dry-run
    /// never deletes anything, so an operationally-too-short `grace` is not
    /// yet a hazard the way it is the moment `apply=true` would act on it.
    fn check_apply_grace(&self, opts: &ReconcileOptions) -> Result<()> {
        if opts.apply && opts.grace < self.lease.lease() {
            return Err(JammiError::Config(format!(
                "reconcile: apply=true requires grace ({:?}) >= the configured lease duration \
                 ({:?}); a shorter grace could reclaim a live writer's still-in-progress bytes",
                opts.grace,
                self.lease.lease()
            )));
        }
        Ok(())
    }

    /// The pass shared by [`Self::reconcile`] and [`Self::reconcile_all`].
    /// `own_seg`, when `Some`, restricts the pass to objects whose own key
    /// attributes to exactly that tenant segment — a key that fails the
    /// allowlist outright (`Attribution::Unattributed`) is store-wide by
    /// definition, so a `Some` (scoped) pass reports NONE of it: only `None`
    /// (from [`Self::reconcile_all`], already under admin scope,
    /// covering every tenant's prefix) ever populates
    /// [`ReconcileReport::unattributed`].
    async fn reconcile_inner(
        &self,
        scope: String,
        opts: ReconcileOptions,
        own_seg: Option<String>,
    ) -> Result<ReconcileReport> {
        let root_handle = self.open_index(&self.root)?;
        let root_path = root_handle.data_path()?;
        let root_prefix = format!("{root_path}/");

        let listed = root_handle
            .list(&root_path)
            .await?
            .into_iter()
            .filter_map(|m| {
                let full = m.path.to_string();
                full.strip_prefix(&root_prefix).map(|rel| Listed {
                    rel: rel.to_string(),
                    size: m.size,
                    last_modified: m.last_modified,
                })
            })
            .collect::<Vec<_>>();

        // Expired-building pre-pass (esc-094 follow-up): an expired-lease
        // `building` row's objects are NEVER reaped through this pass's
        // orphan arm below (no claim, no CAS) — under `apply` they are
        // reconciled through the SAME recovery arm `ResultStore::recover`
        // uses: claim first (fencing whatever writer is or was alive), then
        // promote-or-fail, then delete only after that CAS. The SAME
        // `list_expired_building_tables` call already respects the binding
        // in force, so a tenant-scoped `reconcile()` reconciles only its own
        // tenant's expired rows and the admin-scoped `reconcile_all()` (via
        // `TenantBinding::admin_scope`) covers every tenant's.
        //
        // Runs AFTER the object listing above, like every other row read
        // this pass performs — `reaped` (built here) records every key this
        // arm CREDITS (with the TRUE size the listing snapshot already
        // captured, before any delete); `promoted_purged` (also built here)
        // records every key a `Promote` row's rebuild ACTUALLY purged, but
        // EXCLUDED from all accounting (never credited, never reported) —
        // a promotion is not a reclaim (this module's own doc comment). The
        // object→row loop further down skips any key already in `reaped`,
        // `promoted_purged`, OR `protected`: a key can never be counted
        // twice, and a promotion's internal purge can never be double-
        // reported against the listing snapshot taken before it ran, by
        // construction, regardless of which arm ran first.
        // [`ExpiredRowOutcome`] is the ONE classification BOTH modes branch
        // on: `apply=true` calls [`ResultStore::reconcile_expired_building_row`],
        // which classifies FIRST and then performs exactly the outcome
        // licenses, returning an [`crate::store::ExpiredRowDeletion`] that
        // tells this loop which accumulator to grow; `apply=false` calls
        // [`ResultStore::classify_expired_row`] alone and only classifies:
        //
        // - `Reap`: this row's candidate keys ([`ResultStore::reap_candidate_keys`]
        //   — the SAME set the real deleter computes) are previewed as
        //   orphans, at their TRUE listed size, regardless of the object's
        //   own age — a CAS-licensed reap is never grace-gated (see
        //   [`ReconcileReport::orphans`]'s two admission routes).
        // - `Promote { keeps, dir_prefixes }`: the row's FULL current key set
        //   is PROTECTED — added to `protected`, never `reaped` — so a
        //   promoted-but-not-yet-`ready` row's objects can never fall
        //   through to the general age-gated arm below. Nothing is
        //   predicted or credited about what the rebuild will purge and not
        //   rewrite: apply's own rebuild records that (whatever it actually
        //   deletes) into `promoted_purged` instead, once it runs — a
        //   dry-run never runs the rebuild, so it has nothing to exclude,
        //   which is exactly why protecting the WHOLE current key set is
        //   the correct (and only) thing a preview can do here.
        // - `Untouched`: nothing to account or protect.
        //
        // `orphans`/`orphan_count`/`bytes_reclaimed`/`truncated` are declared
        // HERE (rather than beside `pending`/`unattributed`/`damaged` further
        // down) because this pre-pass is their first writer; every other
        // report accumulator is declared where it was before.
        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut bytes_reclaimed = 0u64;
        let mut truncated = false;
        let mut reaped: BTreeSet<String> = BTreeSet::new();
        let mut promoted_purged: BTreeSet<String> = BTreeSet::new();
        let mut protected = ReferencedKeys {
            exact: BTreeSet::new(),
            dir_prefixes: BTreeSet::new(),
        };
        // Looked up by root-relative key, built ONCE from the listing
        // snapshot above — every candidate key this pre-pass credits is
        // looked up here rather than re-scanning the whole `listed` vector
        // per expired row.
        let listed_sizes: HashMap<&str, u64> =
            listed.iter().map(|o| (o.rel.as_str(), o.size)).collect();

        for table in self.catalog.list_expired_building_tables().await? {
            if opts.apply {
                match self.reconcile_expired_building_row(table).await? {
                    ExpiredRowDeletion::Untouched => {}
                    ExpiredRowDeletion::Reaped(deleted) => {
                        credit_reaped(
                            deleted,
                            &listed_sizes,
                            &mut reaped,
                            &mut orphans,
                            &mut orphan_count,
                            &mut truncated,
                            &mut bytes_reclaimed,
                        );
                    }
                    ExpiredRowDeletion::PromotedPurged(purged) => {
                        // Real deletions, but a promotion's own bookkeeping,
                        // never a reclaim: excluded from every accounting
                        // field this pass, not credited through
                        // `credit_reaped`.
                        promoted_purged.extend(purged);
                    }
                }
                continue;
            }
            match self.classify_expired_row(&table).await? {
                ExpiredRowOutcome::Reap => {
                    let parquet_url = StorageUrl::parse(&table.parquet_path)?;
                    let candidates = self
                        .reap_candidate_keys(&parquet_url, &table.table_name)
                        .await?;
                    credit_reaped(
                        candidates,
                        &listed_sizes,
                        &mut reaped,
                        &mut orphans,
                        &mut orphan_count,
                        &mut truncated,
                        &mut bytes_reclaimed,
                    );
                }
                ExpiredRowOutcome::Promote {
                    keeps,
                    dir_prefixes,
                } => {
                    // The row's WHOLE current key set is protected wholesale
                    // — a dry-run never runs the rebuild, so it predicts
                    // nothing about which of these keys the rebuild will
                    // purge and not rewrite; see this module's own doc
                    // comment ("a promotion is not a reclaim").
                    protected.exact.extend(keeps);
                    protected.dir_prefixes.extend(dir_prefixes);
                }
                ExpiredRowOutcome::Untouched => {}
            }
        }

        // Rows are read AFTER the listing above (ordering rule): the row set
        // this pass checks against is a superset of every listed object's
        // true referencer at listing time. `list_live_building_tables`
        // filters liveness entirely in SQL against the backend's own clock
        // (never a bound application timestamp — `catalog::lease`'s module
        // docs), so this pass never trusts a Rust-side string compare that a
        // Postgres deployment's stored lease text would not even be shaped
        // for.
        let mut ready_rows = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)
            .await?;
        if let Some(seg) = own_seg.as_deref() {
            // A scoped pass reports NOTHING it cannot act on.
            // `list_result_tables_by_status` is an ordinary READ (GLOBAL rows
            // visible to every tenant, like every other read on this table),
            // but THIS enumeration feeds a MUTATING pass immediately below
            // (`fail_ready_result_table`'s CAS, `Strict` on the caller's own
            // tenant — it never matches a GLOBAL row for a tenant-bound
            // caller). Left unfiltered, a tenant-scoped dry-run would REPORT
            // a GLOBAL row in `rows_failed` while the matching `apply` pass's
            // CAS silently missed it, so dry-run and apply would disagree on
            // the identical state. Filter to the binding's own segment HERE,
            // before either branch below reads `ready_rows` — the same rule
            // `list_expired_building_tables` already applies to its own
            // enumeration. Only `reconcile_all` (`own_seg = None`, already
            // under admin scope) ever acts on or reports a GLOBAL ready row.
            ready_rows.retain(|t| t.tenant_id.as_deref().unwrap_or("_global") == seg);
        }
        let live_building = self.catalog.list_live_building_tables().await?;

        // row -> object: a `ready` row missing a required object is driven
        // to `failed` FIRST (ONLY the CAS is skipped when `!apply`,
        // matching "apply=false mutates nothing" — the catalog row itself
        // is untouched), so its bytes fall out of the referenced set built
        // below (the standard `failed`-rows-are-orphans rule, A12) in BOTH
        // modes. Removing the row from `still_ready` under `!apply` too
        // (never re-adding it, matching the `apply` arm exactly) is what
        // makes the dry-run/apply parity invariant on
        // [`ReconcileOptions::apply`] hold here: the object→row loop further
        // down then classifies this row's now-unreferenced objects through
        // the IDENTICAL orphan/pending age gate apply would use, rather than
        // protecting them from ever being previewed as reclaimable.
        let mut rows_failed = Vec::new();
        let mut rows_failed_count = 0u64;
        let mut still_ready = Vec::with_capacity(ready_rows.len());
        for table in ready_rows.drain(..) {
            if self.required_row_objects_present(&table).await? {
                still_ready.push(table);
                continue;
            }
            if !opts.apply {
                // Dry run: reported, and — matching what `apply` would do —
                // the row is NOT kept in the protected `still_ready` set, so
                // its objects fall through to the orphan/pending age gate
                // below exactly as they would under `apply=true`.
                push_capped(
                    &mut rows_failed,
                    &mut rows_failed_count,
                    &mut truncated,
                    table.table_name.clone(),
                );
                continue;
            }
            if self
                .catalog
                .fail_ready_result_table(&table.table_name)
                .await?
            {
                push_capped(
                    &mut rows_failed,
                    &mut rows_failed_count,
                    &mut truncated,
                    table.table_name.clone(),
                );
            } else {
                // The fail-CAS missed: someone else already changed this row
                // (it may already be `ready` again with its objects restored,
                // or reaped by a concurrent pass) — never counted as THIS
                // pass's fail, and its objects stay in the referenced set
                // (protected), not treated as newly reclaimable.
                still_ready.push(table);
            }
        }
        ready_rows = still_ready;

        let referenced_result_keys = self
            .referenced_result_keys(&ready_rows, &live_building)
            .await?;

        let training_jobs = self.catalog.list_training_jobs().await?;
        let models = self.catalog.list_models().await?;
        let running_prefixes: BTreeSet<String> = training_jobs
            .iter()
            .filter(|j| j.status == TrainingJobStatus::Running.to_string())
            .map(|j| {
                format!(
                    "models/{}/{}",
                    TenantSegment::of(j.tenant_id.as_ref()),
                    j.job_id
                )
            })
            .collect();
        let queued_resume_prefixes: BTreeSet<String> = training_jobs
            .iter()
            .filter(|j| j.status == TrainingJobStatus::Queued.to_string())
            .map(|j| {
                format!(
                    "models/{}/{}/_resume",
                    TenantSegment::of(j.tenant_id.as_ref()),
                    j.job_id
                )
            })
            .collect();
        let artifact_prefixes: BTreeSet<String> = models
            .iter()
            .filter_map(|m| m.artifact_path.as_deref())
            .filter_map(|p| StorageUrl::parse(p).ok())
            .filter_map(|u| relative_to(&self.root, &u))
            .collect();

        let mut pending = Vec::new();
        let mut pending_count = 0u64;
        let mut unattributed = Vec::new();
        let mut unattributed_count = 0u64;
        let mut damaged = Vec::new();
        let mut damaged_count = 0u64;
        // `orphans`/`orphan_count`/`bytes_reclaimed`/`truncated` are shared
        // with the expired-building pre-pass above: a cap hit on ANY list
        // (including `rows_failed`) sets the one report-wide flag, and a key
        // that pass already accounted for (`reaped`), excluded
        // (`promoted_purged`), or protected (`protected`) is skipped below.
        let cutoff =
            Utc::now() - chrono::Duration::from_std(opts.grace).unwrap_or(chrono::Duration::MAX);

        for obj in &listed {
            if reaped.contains(&obj.rel) {
                // Already accounted for by the expired-building pre-pass
                // above (reaped under `apply`, or previewed under a
                // dry-run) — never re-classified here, so it can never be
                // counted a second time no matter which arm ran first.
                continue;
            }
            if promoted_purged.contains(&obj.rel) {
                // A `Promote` row's rebuild ACTUALLY purged this key this
                // pass (apply only) — real bytes are gone, but a promotion
                // is not a reclaim: excluded here so this listing snapshot
                // (taken before the purge ran) never re-reports it as an
                // ordinary orphan/pending candidate. Never populated under a
                // dry-run (which never runs a rebuild), and never populated
                // for a key `purge_segments` itself FAILED to delete — that
                // key survives on disk and falls through normally, below.
                continue;
            }
            if protected.contains(&obj.rel) {
                // A row the pre-pass classified `Promote` (apply promotes it
                // to `ready`; dry-run only previews the promotion) —
                // referenced in BOTH modes, so this key is never even a
                // candidate for the age-gated orphan arm below, regardless
                // of how old the object is.
                continue;
            }
            let attribution = attribute(&obj.rel);
            let in_scope = match (&own_seg, &attribution) {
                (None, _) => true, // admin pass: every tenant in scope
                // A tenant-scoped pass reports NOTHING it cannot
                // attribute to its OWN prefix — an unattributed (stray) key
                // is store-wide by definition, so only an admin-scoped pass
                // (`own_seg = None`, from `reconcile_all`) may ever list it.
                (Some(_), Attribution::Unattributed) => false,
                (Some(seg), Attribution::ResultTable) => obj.rel.starts_with(&format!("{seg}/")),
                (Some(seg), Attribution::Artifact { seg: obj_seg, .. }) => obj_seg == seg,
            };
            if !in_scope {
                continue;
            }

            match attribution {
                Attribution::Unattributed => {
                    push_capped(
                        &mut unattributed,
                        &mut unattributed_count,
                        &mut truncated,
                        obj.rel.clone(),
                    );
                    continue;
                }
                Attribution::ResultTable => {
                    if referenced_result_keys.contains(&obj.rel) {
                        continue;
                    }
                }
                Attribution::Artifact { ref seg, ref job } => {
                    let job_prefix = format!("models/{seg}/{job}");
                    if running_prefixes.contains(&job_prefix)
                        || queued_resume_prefixes
                            .iter()
                            .any(|p| obj.rel.starts_with(p.as_str()))
                    {
                        continue;
                    }
                    let matched_artifact_prefix = artifact_prefixes
                        .iter()
                        .find(|p| obj.rel == p.as_str() || obj.rel.starts_with(&format!("{p}/")));
                    if let Some(prefix_rel) = matched_artifact_prefix {
                        let prefix_url = StorageUrl::parse(&format!(
                            "{}/{prefix_rel}",
                            self.root.as_str().trim_end_matches('/')
                        ))?;
                        // Advisory #8: a present-but-UNREADABLE manifest (a
                        // genuine I/O or parse fault, not "no manifest at all"
                        // — that's `Ok(None)`, handled below) must never abort
                        // the WHOLE pass over one bad bundle. Route this one
                        // prefix to `damaged` and keep going.
                        let expected_objects = match self
                            .artifact_store()
                            .expected_objects(&prefix_url)
                            .await
                        {
                            Ok(e) => e,
                            Err(e) => {
                                tracing::warn!(
                                    prefix = %prefix_url,
                                    error = %e,
                                    "reconcile: manifest present but unreadable; reporting damaged"
                                );
                                push_capped(
                                    &mut damaged,
                                    &mut damaged_count,
                                    &mut truncated,
                                    obj.rel.clone(),
                                );
                                continue;
                            }
                        };
                        match expected_objects {
                            Some(expected) => {
                                // `expected` is already a set of
                                // driver-relative `object_store::path::Path`s
                                // (the SAME coordinate space
                                // `root_path`/`listed` use, since the
                                // artifact store shares this store's
                                // registry) — strip the root prefix to land
                                // in this module's root-relative key space,
                                // exactly like every listed object above.
                                let expected_rel: BTreeSet<String> = expected
                                    .iter()
                                    .filter_map(|p| {
                                        p.to_string().strip_prefix(&root_prefix).map(String::from)
                                    })
                                    .collect();
                                if expected_rel.contains(&obj.rel) {
                                    continue;
                                }
                                // A valid manifest exists but does not name
                                // this key: falls through to the
                                // orphan-candidate arm below (age-gated,
                                // never immediate).
                            }
                            None => {
                                // A `models` row names this prefix
                                // but its `manifest.json` is absent — the row
                                // is still live, so this is NEVER reclaimable
                                // through the orphan arm at any grace or
                                // `apply`. Reported as `damaged`, not orphan.
                                push_capped(
                                    &mut damaged,
                                    &mut damaged_count,
                                    &mut truncated,
                                    obj.rel.clone(),
                                );
                                continue;
                            }
                        }
                    }
                }
            }

            // Orphan candidate: age-gate against `grace`. `bytes_reclaimed`
            // is credited here in BOTH modes — under `apply=true` the delete
            // below must actually succeed first (a failed delete leaves the
            // object `pending` for the next pass, never counted as
            // reclaimed); under `apply=false` a dry-run cannot know whether
            // a future delete would fail, so it credits the size the same
            // way every other preview in this pass does (see the pinned
            // dry-run/apply parity invariant on [`ReconcileOptions::apply`]).
            let key = obj.rel.clone();
            if obj.last_modified <= cutoff {
                if opts.apply {
                    match self.delete_relative(&key).await {
                        Ok(DeleteOutcome::Deleted) => {}
                        // Listed a moment ago, gone now (a peer reconcile
                        // pass, or the same pass's own expired-building
                        // pre-pass, won the race and deleted it first): this
                        // call freed nothing, so it is neither an orphan
                        // this pass reclaimed nor a failure to retry — esp.
                        // never credited (esc-484's vanish-window defect).
                        Ok(DeleteOutcome::Absent) => {
                            tracing::warn!(
                                key,
                                "reconcile: orphan vanished before delete; nothing reclaimed, not credited"
                            );
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(key, error = %e, "reconcile: orphan delete failed; left for the next pass");
                            push_capped(&mut pending, &mut pending_count, &mut truncated, key);
                            continue;
                        }
                    }
                }
                bytes_reclaimed += obj.size;
                push_capped(&mut orphans, &mut orphan_count, &mut truncated, key);
            } else {
                push_capped(&mut pending, &mut pending_count, &mut truncated, key);
            }
        }

        rows_failed.sort();
        orphans.sort();
        pending.sort();
        unattributed.sort();
        damaged.sort();

        Ok(ReconcileReport {
            scope,
            applied: opts.apply,
            rows_failed,
            rows_failed_count,
            orphans,
            orphan_count,
            pending,
            pending_count,
            unattributed,
            unattributed_count,
            damaged,
            damaged_count,
            truncated,
            bytes_reclaimed,
        })
    }

    /// Delete the object at root-relative key `rel` (best-effort; 404 is not
    /// an error) and report which of the two actually happened — the caller
    /// must credit `bytes_reclaimed` only for [`DeleteOutcome::Deleted`],
    /// never for a key that was already [`DeleteOutcome::Absent`] by the
    /// time this ran.
    async fn delete_relative(&self, rel: &str) -> Result<DeleteOutcome> {
        let url = StorageUrl::parse(&format!(
            "{}/{rel}",
            self.root.as_str().trim_end_matches('/')
        ))?;
        let handle = self.open_index(&url)?;
        let path = handle.data_path()?;
        Ok(handle.delete_if_exists(&path).await?)
    }

    /// Whether EVERY object a `ready` row's materialization contract requires
    /// right now is present, verified with a live `exists()` per object (never
    /// trusted off a listing snapshot): the Parquet; the `.materialization.json`
    /// sidecar iff `definition_hash` is set; and, per the table's CURRENT
    /// `index_segments` rows, every [`required_sidecar_extensions`] sibling for
    /// that segment's own `row_count`.
    async fn required_row_objects_present(&self, table: &ResultTableRecord) -> Result<bool> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        let parquet_handle = self.open_parquet(&parquet_url)?;
        if !parquet_handle.exists(&parquet_handle.data_path()?).await? {
            return Ok(false);
        }
        if table.definition_hash.is_some() {
            let sidecar_url = layout::sidecar_url(&parquet_url, "materialization.json")?;
            let handle = self.open_index(&sidecar_url)?;
            if !handle.exists(&handle.data_path()?).await? {
                return Ok(false);
            }
        }
        let precision = table.storage_precision.unwrap_or_default();
        for seg in self.catalog.list_index_segments(&table.table_name).await? {
            let seg_url = StorageUrl::parse(&seg.index_path)?;
            let seg_handle = self.open_index(&seg_url)?;
            for ext in required_sidecar_extensions(SidecarKind::Ann, precision, seg.row_count) {
                let path = seg_handle.sibling_path(ext)?;
                if !seg_handle.exists(&path).await? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// The full set of root-relative keys referenced by a `ready` row or a
    /// live-lease `building` row: the Parquet, the `.materialization.json`
    /// sidecar (referenced-if-present, regardless of `definition_hash` — more
    /// generous than [`Self::required_row_objects_present`] on purpose, since
    /// this side must never delete a legitimately-present object), and every
    /// sidecar sibling of every CURRENT `index_segments` row — enumerated
    /// over the FULL [`SidecarKind`] superset (`Ann` AND `Lexical`), not one
    /// kind, since a segment's actual kind is not itself recorded
    /// on the `index_segments` row and this side must never under-protect. A
    /// directory-shaped sibling (`Lexical`'s `.tantivy`) is referenced by
    /// PREFIX — every key under `{base}.tantivy/…`, not only a key that
    /// equals `{base}.tantivy` exactly — while a plain-file sibling is
    /// still matched exactly.
    ///
    /// Segments are referenced by ROWS, not by filename pattern — a
    /// `{base}__segN.*` object with no row is an orphan candidate, e.g. the
    /// late-landing sidecar of a purge a recoverer's claim already ran.
    ///
    /// `pub(super)`: [`ResultStore::classify_expired_row`] (`store::mod`)
    /// calls this too, to build a [`ExpiredRowOutcome::Promote`] row's
    /// `keeps` payload from the SAME currently-referenced key set this
    /// pass's own pre-pass and object→row arms both read.
    pub(super) async fn referenced_result_keys(
        &self,
        ready: &[ResultTableRecord],
        live_building: &[ResultTableRecord],
    ) -> Result<ReferencedKeys> {
        let mut exact = BTreeSet::new();
        let mut dir_prefixes = BTreeSet::new();
        for table in ready.iter().chain(live_building.iter()) {
            let parquet_url = StorageUrl::parse(&table.parquet_path)?;
            if let Some(rel) = relative_to(&self.root, &parquet_url) {
                exact.insert(rel);
            }
            if let Ok(sidecar_url) = layout::sidecar_url(&parquet_url, "materialization.json") {
                if let Some(rel) = relative_to(&self.root, &sidecar_url) {
                    exact.insert(rel);
                }
            }
            for seg in self.catalog.list_index_segments(&table.table_name).await? {
                let Ok(seg_url) = StorageUrl::parse(&seg.index_path) else {
                    continue;
                };
                if let Some(rel) = relative_to(&self.root, &seg_url) {
                    exact.insert(rel);
                }
                for kind in [SidecarKind::Ann, SidecarKind::Lexical] {
                    for ext in sidecar_extensions(kind) {
                        let Ok(sib) = layout::sidecar_url(&seg_url, ext) else {
                            continue;
                        };
                        let Some(rel) = relative_to(&self.root, &sib) else {
                            continue;
                        };
                        // A directory-shaped sibling (only `tantivy` today)
                        // is referenced by every key under it, not only a
                        // key equal to the base name.
                        if is_directory_sidecar_extension(ext) {
                            dir_prefixes.insert(format!("{rel}/"));
                        } else {
                            exact.insert(rel);
                        }
                    }
                }
            }
        }
        Ok(ReferencedKeys {
            exact,
            dir_prefixes,
        })
    }
}

/// The referenced-object set [`ResultStore::referenced_result_keys`] builds:
/// exact keys, plus directory-sibling prefixes (each carrying a trailing
/// `/`) a listed object is referenced through if its own key starts with one.
///
/// `pub(super)` (fields included): [`ResultStore::classify_expired_row`]
/// (`store::mod`) reads both fields directly to build a `Promote` row's
/// `keeps` payload from its current key set.
pub(super) struct ReferencedKeys {
    pub(super) exact: BTreeSet<String>,
    pub(super) dir_prefixes: BTreeSet<String>,
}

impl ReferencedKeys {
    fn contains(&self, rel: &str) -> bool {
        self.exact.contains(rel)
            || self
                .dir_prefixes
                .iter()
                .any(|p| rel.starts_with(p.as_str()))
    }
}

/// `true` iff `ext` names a sidecar sibling that is a DIRECTORY on disk
/// (`SidecarKind::Lexical`'s `.tantivy` today — sidecar_layout.rs's own
/// doc comment names it the one directory-shaped sibling) rather than a
/// single file, so callers matching a listed object against it must match
/// by PREFIX (`{base}.{ext}/…`), never by exact equality alone.
fn is_directory_sidecar_extension(ext: &str) -> bool {
    ext == "tantivy"
}

#[cfg(test)]
mod attribution_tests {
    use super::*;

    #[test]
    fn result_table_key_attributes_by_first_segment() {
        assert!(matches!(
            attribute("_global/table.parquet"),
            Attribution::ResultTable
        ));
    }

    #[test]
    fn artifact_key_requires_a_canonical_v4_job_id() {
        let job = Uuid::new_v4().to_string();
        assert!(matches!(
            attribute(&format!("models/_global/{job}/worker-a/0/manifest.json")),
            Attribution::Artifact { .. }
        ));
    }

    #[test]
    fn artifact_key_with_a_non_uuid_job_segment_is_unattributed() {
        assert!(matches!(
            attribute("models/_global/not-a-uuid/worker-a/0/manifest.json"),
            Attribution::Unattributed
        ));
    }

    #[test]
    fn a_pre_layout_key_is_unattributed() {
        assert!(matches!(
            attribute("some_old_table.parquet"),
            Attribution::Unattributed
        ));
    }

    #[test]
    fn canonical_v4_uuid_check_rejects_braced_form() {
        let job = Uuid::new_v4().to_string();
        assert!(!is_canonical_v4_uuid(&format!("{{{job}}}")));
    }
}

#[cfg(test)]
mod credit_reaped_tests {
    use super::*;

    /// The precondition every case below shares: two candidate keys the
    /// reaper NAMED (either [`ResultStore::reap_candidate_keys`]'s dry-run
    /// preview, or the actual deleted-key set
    /// [`ResultStore::delete_objects_after_cas`] returns), both present in
    /// the listing snapshot at the given sizes.
    fn listed_sizes() -> HashMap<&'static str, u64> {
        HashMap::from([
            ("table.parquet", 100u64),
            ("table.materialization.json", 7u64),
        ])
    }

    /// A delete that fully succeeded credits every key it named, at its
    /// listed size.
    #[test]
    fn full_success_credits_every_key_at_its_listed_size() {
        let mut reaped = BTreeSet::new();
        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut truncated = false;
        let mut bytes_reclaimed = 0u64;
        let keys: BTreeSet<String> = ["table.parquet", "table.materialization.json"]
            .into_iter()
            .map(String::from)
            .collect();

        credit_reaped(
            keys,
            &listed_sizes(),
            &mut reaped,
            &mut orphans,
            &mut orphan_count,
            &mut truncated,
            &mut bytes_reclaimed,
        );

        assert_eq!(orphan_count, 2);
        assert_eq!(bytes_reclaimed, 107);
        assert_eq!(
            orphans,
            vec![
                "table.materialization.json".to_string(),
                "table.parquet".to_string()
            ]
        );
    }

    /// esc-484 item 2: a PARTIAL delete failure — the deleter's returned key
    /// set omits the key whose `delete_if_exists` errored — must credit ONLY
    /// the keys that actually succeeded, never the one left out. This is the
    /// oracle for "`reap_after_fail_cas` credits only bytes whose delete
    /// succeeded": the failed key is simply never passed to this helper, so
    /// it can never inflate `bytes_reclaimed` or appear in `orphans`.
    #[test]
    fn partial_failure_credits_only_the_keys_that_actually_deleted() {
        let mut reaped = BTreeSet::new();
        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut truncated = false;
        let mut bytes_reclaimed = 0u64;
        // Only the Parquet delete succeeded; the manifest sidecar's delete
        // errored, so the caller never included it here.
        let only_succeeded: BTreeSet<String> =
            ["table.parquet"].into_iter().map(String::from).collect();

        credit_reaped(
            only_succeeded,
            &listed_sizes(),
            &mut reaped,
            &mut orphans,
            &mut orphan_count,
            &mut truncated,
            &mut bytes_reclaimed,
        );

        assert_eq!(orphan_count, 1, "the failed key must never be credited");
        assert_eq!(
            bytes_reclaimed, 100,
            "only the Parquet's true size, never the manifest's"
        );
        assert!(!orphans.contains(&"table.materialization.json".to_string()));
    }

    /// A key already `reaped` (some earlier arm already credited it) is
    /// never double-counted, even if handed to this helper again.
    #[test]
    fn a_key_already_reaped_is_never_double_counted() {
        let mut reaped: BTreeSet<String> = ["table.parquet".to_string()].into_iter().collect();
        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut truncated = false;
        let mut bytes_reclaimed = 0u64;
        let keys: BTreeSet<String> = ["table.parquet"].into_iter().map(String::from).collect();

        credit_reaped(
            keys,
            &listed_sizes(),
            &mut reaped,
            &mut orphans,
            &mut orphan_count,
            &mut truncated,
            &mut bytes_reclaimed,
        );

        assert_eq!(orphan_count, 0);
        assert_eq!(bytes_reclaimed, 0);
    }

    /// A key the deleter named but that was never actually present in the
    /// listing snapshot (never really on disk at listing time) is silently
    /// skipped — never credited a phantom size.
    #[test]
    fn a_key_absent_from_the_listing_snapshot_is_never_credited() {
        let mut reaped = BTreeSet::new();
        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut truncated = false;
        let mut bytes_reclaimed = 0u64;
        let keys: BTreeSet<String> = ["table.usearch"].into_iter().map(String::from).collect();

        credit_reaped(
            keys,
            &listed_sizes(),
            &mut reaped,
            &mut orphans,
            &mut orphan_count,
            &mut truncated,
            &mut bytes_reclaimed,
        );

        assert_eq!(orphan_count, 0);
        assert_eq!(bytes_reclaimed, 0);
    }
}
