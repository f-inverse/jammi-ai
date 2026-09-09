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
//! before any byte the row→object check would look for).

use std::collections::BTreeSet;
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
use crate::storage::StorageUrl;
use crate::store::layout::{self, TenantSegment};
use crate::store::ResultStore;
use crate::tenant_scope::TenantBinding;

/// One reconciliation pass's parameters.
#[derive(Debug, Clone, Copy)]
pub struct ReconcileOptions {
    /// `true` to actually delete orphans past `grace` (and flip an
    /// incomplete `ready` row to `failed`); `false` (the default a caller
    /// should reach for first) reports everything a pass WOULD do without
    /// mutating anything.
    pub apply: bool,
    /// An orphan candidate younger than this is `pending`, never deleted —
    /// the window a concurrent writer's just-landed bytes have to grow a
    /// referencing row before a pass would otherwise reclaim them.
    /// `apply = true` REQUIRES `grace >= ` the deployment's configured lease
    /// duration ([`ReconcileOptions`] alone cannot express this — the check
    /// is [`ResultStore::reconcile`]'s), so a lease that is merely running
    /// long can never be raced by a reclaim.
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
    /// object set was found for them).
    pub rows_failed: Vec<String>,
    /// Orphan candidates at least `grace` old — deleted when `applied`.
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
    /// tenant's (or nobody's) stray keys (block #4). Capped at
    /// [`REPORT_LIST_CAP`]; see [`Self::unattributed_count`].
    pub unattributed: Vec<String>,
    /// The true count of unattributed keys found this pass, independent of
    /// whether [`Self::unattributed`] was truncated.
    pub unattributed_count: u64,
    /// A prefix named by a `models` row (a trained-model artifact bundle)
    /// whose `manifest.json` is absent: the row says this bundle exists, but
    /// its attestation does not — never reclaimed, at any grace or `apply`
    /// (block #3), because the referencing row is still live. Distinct from
    /// [`Self::orphans`], whose entries have NO referencing row at all.
    /// Capped at [`REPORT_LIST_CAP`]; see [`Self::damaged_count`].
    pub damaged: Vec<String>,
    /// The true count of damaged keys found this pass, independent of
    /// whether [`Self::damaged`] was truncated.
    pub damaged_count: u64,
    /// `true` iff any of [`Self::orphans`], [`Self::pending`],
    /// [`Self::unattributed`], [`Self::damaged`] was cut to
    /// [`REPORT_LIST_CAP`] entries — the corresponding `*_count` field is
    /// still the true total either way.
    pub truncated: bool,
    /// Total bytes actually reclaimed (`orphans` deleted this pass; `0` when
    /// `!applied`).
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
/// not share `root`'s prefix at all.
fn relative_to(root: &StorageUrl, url: &StorageUrl) -> Option<String> {
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
    /// materialization.
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
    /// definition, so a `Some` (scoped) pass reports NONE of it (block #4):
    /// only `None` (from [`Self::reconcile_all`], already under admin scope,
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

        // Block #1 (esc-094 follow-up): an expired-lease `building` row's
        // objects are NEVER reaped through this pass's orphan arm below (no
        // claim, no CAS) — they are reconciled through the SAME recovery arm
        // `ResultStore::recover` uses: claim first (fencing whatever writer
        // is or was alive), then promote-or-fail, then delete only after
        // that CAS. Runs BEFORE this pass reads `ready` / live `building`
        // rows, so a row this step promotes or fails is read back in its new
        // terminal state below. Only under `apply` — `apply=false` mutates
        // nothing (an expired row's objects are still reported, as an
        // ordinary orphan candidate, exactly as before this fix); the SAME
        // `list_expired_building_tables` call already respects the binding
        // in force, so a tenant-scoped `reconcile()` reconciles only its own
        // tenant's expired rows and the admin-scoped `reconcile_all()` (via
        // `TenantBinding::admin_scope`) covers every tenant's.
        if opts.apply {
            for table in self.catalog.list_expired_building_tables().await? {
                self.reconcile_expired_building_row(table).await?;
            }
        }

        // Rows are read AFTER the listing above (ordering rule): the row set
        // this pass checks against is a superset of every listed object's
        // true referencer at listing time. `list_live_building_tables`
        // filters liveness entirely in SQL against the backend's own clock
        // (never a bound application timestamp — `catalog::lease`'s module
        // docs), so this pass never trusts a Rust-side string compare that a
        // Postgres deployment's stored lease text would not even be shaped
        // for after block #2's fix.
        let mut ready_rows = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)
            .await?;
        let live_building = self.catalog.list_live_building_tables().await?;

        // row -> object: a `ready` row missing a required object is driven
        // to `failed` FIRST (ONLY when `apply`, matching "apply=false
        // mutates nothing" — a dry-run pass still REPORTS the row in
        // `rows_failed`, but never performs the CAS), so its bytes fall out
        // of the referenced set built below (the standard
        // `failed`-rows-are-orphans rule, A12). Under `!apply` the row stays
        // in `still_ready` so its objects remain protected — nothing new
        // becomes reclaimable from a pass that changed nothing.
        let mut rows_failed = Vec::new();
        let mut still_ready = Vec::with_capacity(ready_rows.len());
        for table in ready_rows.drain(..) {
            if self.required_row_objects_present(&table).await? {
                still_ready.push(table);
                continue;
            }
            rows_failed.push(table.table_name.clone());
            if !opts.apply {
                still_ready.push(table);
                continue;
            }
            // A miss (row moved on already) is not this pass's problem —
            // whoever else changed it owns reporting; the row is already
            // excluded from `still_ready` either way.
            let _ = self
                .catalog
                .fail_ready_result_table(&table.table_name)
                .await?;
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

        let mut orphans = Vec::new();
        let mut orphan_count = 0u64;
        let mut pending = Vec::new();
        let mut pending_count = 0u64;
        let mut unattributed = Vec::new();
        let mut unattributed_count = 0u64;
        let mut damaged = Vec::new();
        let mut damaged_count = 0u64;
        let mut truncated = false;
        let mut bytes_reclaimed = 0u64;
        let cutoff =
            Utc::now() - chrono::Duration::from_std(opts.grace).unwrap_or(chrono::Duration::MAX);

        for obj in &listed {
            let attribution = attribute(&obj.rel);
            let in_scope = match (&own_seg, &attribution) {
                (None, _) => true, // admin pass: every tenant in scope
                // Block #4: a tenant-scoped pass reports NOTHING it cannot
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
                        match self.artifact_store().expected_objects(&prefix_url).await? {
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
                                // this key: outside block #3's scope, falls
                                // through to the orphan-candidate arm below
                                // (age-gated, never immediate).
                            }
                            None => {
                                // Block #3: a `models` row names this prefix
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

            // Orphan candidate: age-gate against `grace`.
            let key = obj.rel.clone();
            if obj.last_modified <= cutoff {
                if opts.apply {
                    if let Err(e) = self.delete_relative(&key).await {
                        tracing::warn!(key, error = %e, "reconcile: orphan delete failed; left for the next pass");
                        push_capped(&mut pending, &mut pending_count, &mut truncated, key);
                        continue;
                    }
                    bytes_reclaimed += obj.size;
                }
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
    /// an error).
    async fn delete_relative(&self, rel: &str) -> Result<()> {
        let url = StorageUrl::parse(&format!(
            "{}/{rel}",
            self.root.as_str().trim_end_matches('/')
        ))?;
        let handle = self.open_index(&url)?;
        let path = handle.data_path()?;
        handle.delete_if_exists(&path).await?;
        Ok(())
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
    /// kind (block #7), since a segment's actual kind is not itself recorded
    /// on the `index_segments` row and this side must never under-protect. A
    /// directory-shaped sibling (`Lexical`'s `.tantivy`) is referenced by
    /// PREFIX — every key under `{base}.tantivy/…`, not only a key that
    /// equals `{base}.tantivy` exactly (A3) — while a plain-file sibling is
    /// still matched exactly.
    /// (A21: segments are referenced by ROWS, not by filename pattern — a
    /// `{base}__segN.*` object with no row is an orphan candidate, e.g. the
    /// late-landing sidecar of a purge a recoverer's claim already ran.)
    async fn referenced_result_keys(
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
struct ReferencedKeys {
    exact: BTreeSet<String>,
    dir_prefixes: BTreeSet<String>,
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
/// by PREFIX (`{base}.{ext}/…`), never by exact equality alone (A3).
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
