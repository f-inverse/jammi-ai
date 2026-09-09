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

/// The result of one reconciliation pass. Every list is sorted; `Serialize`
/// so a wire/CLI mapping (a later commit) can hand this back verbatim.
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
    pub orphans: Vec<String>,
    /// Orphan candidates younger than `grace` — never deleted this pass.
    pub pending: Vec<String>,
    /// Listed keys whose own path does not parse through the tenant/artifact
    /// allowlist at all — reported, never deleted at any grace or `apply`.
    pub unattributed: Vec<String>,
    /// Total bytes actually reclaimed (`orphans` deleted this pass; `0` when
    /// `!applied`).
    pub bytes_reclaimed: u64,
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
    /// attributes to exactly that tenant segment (plus any key that fails
    /// the allowlist outright, which is `unattributed` regardless of scope —
    /// no tenant's scoped pass would ever otherwise report it); `None`
    /// (only from [`Self::reconcile_all`], already under admin scope) covers
    /// every tenant's prefix.
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

        // Rows are read AFTER the listing above (ordering rule): the row set
        // this pass checks against is a superset of every listed object's
        // true referencer at listing time.
        let mut ready_rows = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)
            .await?;
        let building_rows = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Building)
            .await?;
        let now = crate::catalog::lease::lease_now();
        let live_building: Vec<ResultTableRecord> = building_rows
            .into_iter()
            .filter(|t| match &t.lease_expires_at {
                Some(exp) => exp.as_str() > now.as_str(),
                None => false,
            })
            .collect();

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
        let mut pending = Vec::new();
        let mut unattributed = Vec::new();
        let mut bytes_reclaimed = 0u64;
        let cutoff =
            Utc::now() - chrono::Duration::from_std(opts.grace).unwrap_or(chrono::Duration::MAX);

        for obj in &listed {
            let attribution = attribute(&obj.rel);
            let in_scope = match (&own_seg, &attribution) {
                (None, _) => true,                            // admin pass: every tenant in scope
                (Some(_), Attribution::Unattributed) => true, // garbage is always in scope
                (Some(seg), Attribution::ResultTable) => obj.rel.starts_with(&format!("{seg}/")),
                (Some(seg), Attribution::Artifact { seg: obj_seg, .. }) => obj_seg == seg,
            };
            if !in_scope {
                continue;
            }

            match attribution {
                Attribution::Unattributed => {
                    unattributed.push(obj.rel.clone());
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
                        if let Some(expected) =
                            self.artifact_store().expected_objects(&prefix_url).await?
                        {
                            // `expected` is already a set of driver-relative
                            // `object_store::path::Path`s (the SAME
                            // coordinate space `root_path`/`listed` use,
                            // since the artifact store shares this store's
                            // registry) — strip the root prefix to land in
                            // this module's root-relative key space, exactly
                            // like every listed object above.
                            let expected_rel: BTreeSet<String> = expected
                                .iter()
                                .filter_map(|p| {
                                    p.to_string().strip_prefix(&root_prefix).map(String::from)
                                })
                                .collect();
                            if expected_rel.contains(&obj.rel) {
                                continue;
                            }
                        }
                        // No manifest, or this key isn't among the manifest's
                        // own entries: falls through to the orphan-candidate
                        // arm below (age-gated, never immediate).
                    }
                }
            }

            // Orphan candidate: age-gate against `grace`.
            let key = obj.rel.clone();
            if obj.last_modified <= cutoff {
                if opts.apply {
                    if let Err(e) = self.delete_relative(&key).await {
                        tracing::warn!(key, error = %e, "reconcile: orphan delete failed; left for the next pass");
                        pending.push(key);
                        continue;
                    }
                    bytes_reclaimed += obj.size;
                }
                orphans.push(key);
            } else {
                pending.push(key);
            }
        }

        rows_failed.sort();
        orphans.sort();
        pending.sort();
        unattributed.sort();

        Ok(ReconcileReport {
            scope,
            applied: opts.apply,
            rows_failed,
            orphans,
            pending,
            unattributed,
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
    /// [`sidecar_extensions`] sibling of every CURRENT `index_segments` row
    /// (A21: segments are referenced by ROWS, not by filename pattern — a
    /// `{base}__segN.*` object with no row is an orphan candidate, e.g. the
    /// late-landing sidecar of a purge a recoverer's claim already ran).
    async fn referenced_result_keys(
        &self,
        ready: &[ResultTableRecord],
        live_building: &[ResultTableRecord],
    ) -> Result<BTreeSet<String>> {
        let mut set = BTreeSet::new();
        for table in ready.iter().chain(live_building.iter()) {
            let parquet_url = StorageUrl::parse(&table.parquet_path)?;
            if let Some(rel) = relative_to(&self.root, &parquet_url) {
                set.insert(rel);
            }
            if let Ok(sidecar_url) = layout::sidecar_url(&parquet_url, "materialization.json") {
                if let Some(rel) = relative_to(&self.root, &sidecar_url) {
                    set.insert(rel);
                }
            }
            for seg in self.catalog.list_index_segments(&table.table_name).await? {
                let Ok(seg_url) = StorageUrl::parse(&seg.index_path) else {
                    continue;
                };
                if let Some(rel) = relative_to(&self.root, &seg_url) {
                    set.insert(rel);
                }
                for ext in sidecar_extensions(SidecarKind::Ann) {
                    if let Ok(sib) = layout::sidecar_url(&seg_url, ext) {
                        if let Some(rel) = relative_to(&self.root, &sib) {
                            set.insert(rel);
                        }
                    }
                }
            }
        }
        Ok(set)
    }
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
