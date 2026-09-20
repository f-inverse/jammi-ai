//! Model-artifact storage over the shared object store.
//!
//! A model artifact (a fine-tune adapter, a context-predictor weight set) is a
//! small bundle of files — `adapter.safetensors` + `adapter_config.json`, or a
//! single `model.safetensors` — that a worker writes once and inference reloads
//! later, possibly on a different host. [`ArtifactStore`] routes both sides
//! through the same [`StorageRegistry`] result tables use, so a `file://` root
//! keeps single-host deployments byte-identical to today while an `s3://` /
//! `r2://` root lets a worker fleet share trained models across hosts.
//!
//! ## Correctness model: the catalog row is the commit
//!
//! Every bundle is a [`crate::catalog::artifact_repo`] row. A writer stages
//! the row BEFORE its first byte and writes to a **unique per-attempt
//! prefix** — a worker never writes a shared canonical path, so two workers
//! training the same job never collide and no served object is ever
//! overwritten. The finalize transaction that publishes the artifact and
//! attaches the `models` row to it is the single atomic commit; a loser's
//! bundle stays `staged` and is reclaimed. There is no promote/rename step and
//! therefore no torn-promote window.
//!
//! Bytes leave `models/` one way: [`ArtifactStore::reclaim`], which takes the
//! [`ReclaimLicence`] only the catalog's reclaim compare-and-set mints.
//!
//! ## Checkpoints: one immutable prefix per epoch, newest complete wins
//!
//! A job's checkpoint is the same kind of bundle, written by the lease
//! holder at every epoch boundary — the loadable adapter plus the run's
//! optimizer and resume state — and the same rule applies: a bundle is
//! written to a fresh prefix, never over a complete one. Each epoch's bundle
//! lives under its own prefix, `{job_id}/_checkpoints/{attempt}/epoch_{N}`,
//! as its own job-scoped row ([`ArtifactStore::stage_checkpoint`]). A crash
//! mid-write leaves a prefix with no manifest — "nothing published here",
//! which a read skips — never a complete manifest over another epoch's
//! bytes. A resume reads the newest epoch whose manifest exists and verifies
//! ([`ArtifactStore::fetch_newest_checkpoint`]); every write retires the
//! epochs beyond the caller's retention window through the store's own
//! reclaim path, so a job holds its window of complete checkpoints plus,
//! transiently, the one being written. The window's bundles are what a
//! finalize publishes as the job's epoch models.
//!
//! ## Manifest discipline
//!
//! Every staged bundle writes its data files first, then a
//! `manifest.json` **last**, listing the exact relative keys and each file's
//! sha256. [`ArtifactStore::fetch_artifact`] reads the manifest and fetches exactly those keys —
//! it never `LIST`s. Because every attempt is a fresh unique prefix, the only
//! consistency a reader relies on is read-after-write of a *new* object, which
//! every object store (including S3) serves strongly; list-after-write and
//! overwrite-then-read — the eventually-consistent operations — are never on the
//! path. A manifest-listed file whose bytes do not hash to the recorded digest,
//! or that has gone missing after the manifest names it, is a hard integrity
//! error (a partial PUT, not a torn load to be papered over) —
//! [`StorageError::Layout`]. A manifest that is absent ENTIRELY is a distinct
//! outcome, [`StorageError::NotPublished`]: no manifest is in hand, so there is
//! nothing to say is corrupt — the prefix was simply never published, or a
//! catalog pointer names the wrong prefix.
//!
//! ## Local cache
//!
//! `fetch_artifact` materialises the bundle into a **content-addressed** local
//! directory keyed by the manifest's combined hash (immutable for a given
//! training run), downloading into a tempdir and atomically renaming into place
//! so a concurrent fetch of the same artifact is torn-free. A `file://` root
//! short-circuits: the artifact already lives on a local path candle can mmap,
//! so the prefix path is returned directly with no copy.

use std::path::PathBuf;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use std::num::NonZeroUsize;

use crate::catalog::artifact_repo::{
    ArtifactRef, HeldArtifact, ReclaimDecision, ReclaimLicence, StagedArtifact, StagingScope,
};
use crate::catalog::status::ArtifactState;
use crate::catalog::Catalog;
use crate::error::{JammiError, Result};
use crate::storage::{
    sha256_hex, DeleteOutcome, JammiObjectStore, Scheme, StorageError, StorageRegistry, StorageUrl,
};
use crate::store::layout::TenantSegment;
use crate::store::manifest::{
    ArtifactDigest, LeafDigest, LeafKey, Materialization, MaterializationManifest,
};
use crate::store::{manifest_to_jammi, run_id};
use crate::tenant::TenantId;

/// The file every artifact prefix carries last, naming the bundle's exact keys
/// and per-file digests. Written after every data file so its presence proves
/// the bundle is complete.
const MANIFEST_NAME: &str = "manifest.json";

/// The model-artifact peer of a result table's `.materialization.json`
/// sidecar (`crate::store::mod::materialization_sidecar_path`) — the
/// reproducibility attestation over a `FineTune`
/// [`crate::store::manifest::ProducingDescriptor`]. Written by
/// [`ArtifactStore::write_model_materialization`] strictly AFTER
/// [`MANIFEST_NAME`] (which a bundle write puts last among the bundle's own
/// files), so it is the LAST object in the
/// prefix overall: a reader that finds it knows the bundle is not only
/// complete (`manifest.json`'s own guarantee) but carries the definition
/// hash and input anchors this contract exists to attest.
const MATERIALIZATION_NAME: &str = "materialization.json";

/// The prefix segment every epoch checkpoint of a job sits under:
/// `{job_id}/_checkpoints/{attempt}/epoch_{N}`. Distinct from the per-attempt
/// publish prefix (`{job_id}/{worker_id}/{attempt}`) so a checkpoint never
/// collides with a served artifact's prefix; the attempt and the epoch
/// beneath it make every write a fresh prefix — no attempt ever writes one
/// epoch twice, and no two attempts share one.
const CHECKPOINTS_SEGMENT: &str = "_checkpoints";

/// One file in an artifact bundle: its relative name (the candle loader joins
/// this onto the fetched directory) and the sha256 of its bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ManifestEntry {
    /// File name relative to the artifact prefix, e.g. `adapter.safetensors`.
    name: String,
    /// Lowercase-hex sha256 of the file's bytes.
    sha256: String,
}

/// The `manifest.json` payload: the bundle's files in a stable order. The order
/// is fixed by the bundle write (it sorts by name) so the
/// combined hash a reader derives is deterministic for a given content set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Manifest {
    files: Vec<ManifestEntry>,
}

impl Manifest {
    /// The content-address of the whole bundle: sha256 over each entry's
    /// `name` + `sha256` in manifest order. Two bundles with identical file
    /// names and bytes hash equal, so the local cache dir is shared across
    /// repeated fetches of the same training run's artifact.
    fn combined_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for entry in &self.files {
            hasher.update(entry.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(entry.sha256.as_bytes());
            hasher.update(b"\0");
        }
        hex::encode(hasher.finalize())
    }
}

/// A fetched artifact materialised onto the local filesystem: a directory whose
/// files are exactly the manifest's keys, verified against their recorded
/// digests. The candle reload path joins each expected file name
/// (`adapter.safetensors`, `model.safetensors`, …) onto [`Self::dir`].
#[derive(Debug, Clone)]
pub struct LocalArtifact {
    dir: PathBuf,
}

impl LocalArtifact {
    /// The local directory holding the verified artifact files.
    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }
}

/// Stores and reloads model artifacts under a [`StorageUrl`] root, over the same
/// [`StorageRegistry`] the result store uses. Construct one with
/// [`Self::with_root`]; it shares the session's registry so cloud credentials
/// are registered once.
pub struct ArtifactStore {
    root: StorageUrl,
    registry: StorageRegistry,
    /// Root of the content-addressed local fetch cache. A `file://` root never
    /// populates this (fetch short-circuits to the in-place path).
    cache_root: PathBuf,
}

impl ArtifactStore {
    /// Construct an artifact store rooted at `root`, sharing `registry` with the
    /// engine session. `cache_root` is the local directory the fetch cache
    /// materialises cloud artifacts under (ignored for a `file://` root, which
    /// reads in place). For a `file://` root the directory is created so the
    /// first write does not fail; cloud schemes are bucket-rooted and have no
    /// directory concept.
    pub fn with_root(
        root: StorageUrl,
        registry: StorageRegistry,
        cache_root: PathBuf,
    ) -> Result<Self> {
        if root.scheme() == Scheme::File {
            std::fs::create_dir_all(root.path())?;
        }
        std::fs::create_dir_all(&cache_root)?;
        Ok(Self {
            root,
            registry,
            cache_root,
        })
    }

    /// Write a bundle's bytes under `prefix`: each `(name, bytes)` is PUT,
    /// then `manifest.json` is PUT **last** — its presence proves the bundle
    /// is complete. Private: a bundle is only ever written by
    /// [`Self::stage_bundle`], after its `staged` row.
    async fn write_bundle(&self, prefix: &StorageUrl, files: &[(String, Bytes)]) -> Result<()> {
        let handle = self.handle(prefix)?;

        // Sort entries by name so the manifest order — and thus the combined
        // content-hash a reader derives for the cache key — is deterministic for
        // a given content set regardless of caller order.
        let mut sorted: Vec<&(String, Bytes)> = files.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut entries = Vec::with_capacity(sorted.len());
        for (name, bytes) in &sorted {
            let path = self.child(prefix, name)?;
            handle.put_bytes(&path, bytes.clone()).await?;
            entries.push(ManifestEntry {
                name: (*name).clone(),
                sha256: sha256_hex(bytes),
            });
        }

        // The seam a crash between the last data-file PUT and the manifest
        // PUT lands in: a test parks a writer here to leave exactly that
        // prefix behind.
        #[cfg(feature = "test-hooks")]
        artifact_test_hooks::maybe_park_before_manifest(prefix).await;

        // Manifest LAST: a reader that finds it can trust every key it names was
        // already written.
        let manifest = Manifest { files: entries };
        let manifest_bytes = Bytes::from(serde_json::to_vec(&manifest)?);
        let manifest_path = self.child(prefix, MANIFEST_NAME)?;
        handle.put_bytes(&manifest_path, manifest_bytes).await?;
        Ok(())
    }

    /// Compute and write the `materialization.json` attestation for the
    /// caller's own staged bundle — the model-artifact peer of a result
    /// table's attestation (`crate::store::ResultStore::write_attestation`).
    /// It takes the [`StagedArtifact`], so a writer can attest only a bundle
    /// it staged itself. Written strictly AFTER `manifest.json`, making it the
    /// LAST object in the prefix (see `MATERIALIZATION_NAME`'s doc).
    ///
    /// The [`ArtifactDigest`] folded into the [`crate::store::manifest::DefinitionHash`]
    /// is the bundle's `Manifest::combined_hash` — every file name + its own
    /// sha256, in the manifest's stable name-sorted order — the SAME content
    /// address [`Self::fetch_artifact`]'s local cache already keys on, so two
    /// bundles with identical file names and bytes attest identically.
    ///
    /// This reads back the ALREADY-WRITTEN `manifest.json` rather than
    /// accepting a digest parameter, so it can only ever attest a bundle
    /// that is provably complete: a bundle whose manifest is absent fails
    /// with [`StorageError::NotPublished`] — it can never attest a partial
    /// bundle.
    pub async fn write_model_materialization(
        &self,
        staged: &StagedArtifact,
        materialization: Materialization<'_>,
    ) -> Result<MaterializationManifest> {
        let prefix = staged.artifact().url();
        let handle = self.handle(prefix)?;
        let manifest = self.read_manifest(&handle, prefix).await?;
        let digest = ArtifactDigest(manifest.combined_hash());
        // One leaf per file, keyed by NAME (the bundle manifest's own
        // sha256 per file): adding a file changes no existing leaf, and the
        // subject stays `combined_hash` — the content address
        // `fetch_artifact`'s cache already keys on.
        let leaves = manifest
            .files
            .iter()
            .map(|entry| LeafDigest {
                key: LeafKey::File {
                    name: entry.name.clone(),
                },
                digest: ArtifactDigest(entry.sha256.clone()),
            })
            .collect();

        let attestation = MaterializationManifest::compute(
            materialization.descriptor,
            materialization.env,
            materialization.inputs,
            digest,
            leaves,
            run_id().to_string(),
            chrono::Utc::now().to_rfc3339(),
        )
        .map_err(manifest_to_jammi)?;

        let bytes = attestation.to_json_bytes().map_err(manifest_to_jammi)?;
        let path = self.child(prefix, MATERIALIZATION_NAME)?;
        handle.put_bytes(&path, Bytes::from(bytes)).await?;

        Ok(attestation)
    }

    /// Read a model artifact prefix's `materialization.json` sidecar, if
    /// present — the model-artifact peer of
    /// [`crate::store::ResultStore::read_materialization_manifest`]. Returns
    /// `Ok(None)` when no sidecar exists (a model that predates this
    /// contract, or one with no materialization at all).
    ///
    /// The sidecar's path is always DERIVED here — `self.child(prefix,
    /// MATERIALIZATION_NAME)`, the fixed relative name under the given
    /// artifact prefix — never read back from a catalog column: `models`
    /// carries no `manifest_path` column, so there is no separate pointer
    /// that could drift out of sync with where the sidecar actually lives.
    pub async fn read_model_materialization(
        &self,
        prefix: &StorageUrl,
    ) -> Result<Option<MaterializationManifest>> {
        let handle = self.handle(prefix)?;
        let path = self.child(prefix, MATERIALIZATION_NAME)?;
        if !handle.exists(&path).await? {
            return Ok(None);
        }
        let bytes = handle.get_bytes(&path).await?;
        let manifest =
            MaterializationManifest::from_json_bytes(&bytes).map_err(manifest_to_jammi)?;
        Ok(Some(manifest))
    }

    /// Fetch the artifact at `prefix` into a verified local directory candle can
    /// mmap.
    ///
    /// Reads `manifest.json`, fetches exactly its keys (never `LIST`), and
    /// verifies each file's sha256 against the manifest — a mismatch or a missing
    /// key is a hard error. The bundle is materialised into a content-addressed
    /// cache dir keyed by the manifest's combined hash: a download lands in a
    /// tempdir first and is atomically renamed into the cache, so a concurrent
    /// fetch of the same artifact never observes a partial directory, and a
    /// cache hit returns immediately without re-downloading. A `file://` prefix
    /// short-circuits to the in-place path (no copy).
    pub async fn fetch_artifact(&self, prefix: &StorageUrl) -> Result<LocalArtifact> {
        let handle = self.handle(prefix)?;
        let manifest = self.read_manifest(&handle, prefix).await?;

        if prefix.scheme() == Scheme::File {
            // The artifact already lives on a local path candle can mmap. Verify
            // the manifest so a partial write is still caught, then return the
            // prefix path in place — no copy.
            self.verify_files(&handle, prefix, &manifest).await?;
            return Ok(LocalArtifact {
                dir: PathBuf::from(prefix.path()),
            });
        }

        let cache_dir = self.cache_root.join(manifest.combined_hash());

        // Cache hit: the dir was published by a prior fetch's atomic rename, so
        // it is complete by construction.
        if cache_dir.is_dir() {
            return Ok(LocalArtifact { dir: cache_dir });
        }

        // Download into a sibling tempdir, verifying each file, then atomically
        // rename into the cache.
        let tmp = tempfile::tempdir_in(&self.cache_root)?;
        for entry in &manifest.files {
            let path = self.child(prefix, &entry.name)?;
            let bytes = handle
                .get_bytes(&path)
                .await
                .map_err(|e| reclassify_missing_key(e, prefix, &entry.name))?;
            verify_sha256(prefix, entry, &bytes)?;
            std::fs::write(tmp.path().join(&entry.name), &bytes)?;
        }

        // Atomic publish. A concurrent fetch may have won the race and already
        // renamed an identical bundle into place (same content-hash) — that is a
        // benign loss; the existing dir is byte-identical, so keep it.
        match std::fs::rename(tmp.path(), &cache_dir) {
            Ok(()) => Ok(LocalArtifact { dir: cache_dir }),
            Err(_) if cache_dir.is_dir() => Ok(LocalArtifact { dir: cache_dir }),
            Err(e) => Err(e.into()),
        }
    }

    /// Stage and write the bundle one attempt of a job serves, under the
    /// attempt-unique prefix `{job_id}/{worker_id}/{attempt}`.
    ///
    /// The `staged` catalog row is written FIRST, through `catalog` (whose
    /// bound tenant owns both the row and the prefix's tenant segment), so no
    /// byte exists under `models/` that a row does not name. The returned
    /// [`StagedArtifact`] is the writer's claim on the bundle: a finalize
    /// publishes it, and an attempt that gives up reclaims it.
    pub async fn stage_attempt_artifact(
        &self,
        catalog: &Catalog,
        job_id: &str,
        worker_id: &str,
        attempt: u32,
        files: &[(String, Bytes)],
    ) -> Result<StagedArtifact> {
        let attempt_segment = attempt.to_string();
        self.stage_bundle(
            catalog,
            StagingScope::Attempt {
                job_id: job_id.to_string(),
                attempt,
            },
            &[job_id, worker_id, &attempt_segment],
            files,
        )
        .await
    }

    /// Stage and write one epoch's checkpoint under its own prefix,
    /// `{job_id}/_checkpoints/{attempt}/epoch_{epoch}` — a fresh prefix for
    /// every write, so a checkpoint never overwrites a complete one and a
    /// crash before the manifest leaves a prefix with no manifest (skipped by
    /// [`Self::fetch_newest_checkpoint`]), never a complete manifest over
    /// another epoch's bytes. `epoch` is the 0-based index of the epoch the
    /// bundle completes. The bundle is the one every reader of a checkpoint
    /// shares: the loadable adapter a finalize publishes as an epoch model,
    /// plus the optimizer moments and run state a resume restores.
    ///
    /// A checkpoint belongs to the JOB — attempt N+1 resumes from attempt N's
    /// progress — so every epoch's artifact is staged job-scoped: protected
    /// while the job is non-terminal, published by a finalize that retains it
    /// as a model, and otherwise reclaimed once the job ends
    /// ([`Self::reclaim_checkpoints`]).
    ///
    /// Once this epoch's manifest has landed, every epoch of the job beyond
    /// the `retain` newest — older attempts' included — is retired through the
    /// store's own reclaim path, so the job holds its retention window plus,
    /// transiently, the one being written. A retirement that fails is logged,
    /// never an error of the write: the next epoch's write retries it, and the
    /// job's end reclaims whatever remains. Only the lease holder writes (the
    /// trainer gates the call on `!cancel` at the epoch boundary), so a
    /// lost-lease zombie cannot add a stale epoch behind the lease winner's.
    pub async fn stage_checkpoint(
        &self,
        catalog: &Catalog,
        job_id: &str,
        attempt: u32,
        epoch: usize,
        retain: NonZeroUsize,
        files: &[(String, Bytes)],
    ) -> Result<StagedArtifact> {
        let key = CheckpointKey { attempt, epoch };
        let staged = self
            .stage_bundle(
                catalog,
                StagingScope::Job {
                    job_id: job_id.to_string(),
                },
                &checkpoint_segments(job_id, key),
                files,
            )
            .await?;
        let retired = match self.held_checkpoints(catalog, job_id).await {
            Ok(held) => {
                self.reclaim_held(catalog, held.into_iter().skip(retain.get()))
                    .await
            }
            Err(e) => Err(e),
        };
        match retired {
            Ok(unsettled) if unsettled.is_empty() => {}
            Ok(unsettled) => tracing::warn!(
                job_id,
                attempt,
                epoch,
                unsettled = unsettled.len(),
                "checkpoints beyond the retention window were not retired; the next epoch's \
                 write retries"
            ),
            Err(e) => tracing::warn!(
                job_id,
                attempt,
                epoch,
                error = %e,
                "could not list the job's checkpoints; the next epoch's write retries their \
                 retirement"
            ),
        }
        Ok(staged)
    }

    /// The prefix one epoch checkpoint of a job is written under — the one
    /// place that layout is built, so a test observing a specific epoch's
    /// write (the store's test hooks are keyed by prefix) names it through
    /// this rather than the layout string.
    pub fn checkpoint_prefix(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
        attempt: u32,
        epoch: usize,
    ) -> Result<StorageUrl> {
        self.prefix_url(
            tenant,
            &checkpoint_segments(job_id, CheckpointKey { attempt, epoch }),
        )
    }

    /// Reclaim every checkpoint of `job_id` a finalize did not publish, as
    /// the job's own stager — what the job's finisher calls once the job is
    /// terminal and no attempt will read a checkpoint again. Returns the
    /// artifacts left unsettled: a reclaim that was refused or failed (each
    /// logged with its cause), which a reconcile pass converges. An error is
    /// the listing itself failing.
    pub async fn reclaim_checkpoints(
        &self,
        catalog: &Catalog,
        job_id: &str,
    ) -> Result<Vec<ArtifactRef>> {
        let held = self.held_checkpoints(catalog, job_id).await?;
        self.reclaim_held(catalog, held).await
    }

    /// Every checkpoint of `job_id` the catalog still holds unpublished,
    /// newest key first, each with its row's state.
    async fn held_checkpoints(
        &self,
        catalog: &Catalog,
        job_id: &str,
    ) -> Result<Vec<(CheckpointKey, HeldArtifact)>> {
        let mut held = catalog
            .job_scoped_artifacts(job_id)
            .await?
            .into_iter()
            .map(|held| checkpoint_key(held.claim.artifact().url()).map(|key| (key, held)))
            .collect::<Result<Vec<_>>>()?;
        held.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(held)
    }

    /// Reclaim each of `held` as the stager's own bundle: the reclaim
    /// compare-and-set ([`Catalog::reclaim_own_staged_artifact`]) and, under
    /// the licence it mints, [`Self::reclaim`]. A prefix torn before its
    /// manifest has no inventory of its own here (the store never lists —
    /// see [`crate::storage::JammiObjectStore`]); its row is retired and its
    /// bytes are the strays a reconcile pass adopts. Returns the artifacts
    /// left unsettled.
    async fn reclaim_held(
        &self,
        catalog: &Catalog,
        held: impl IntoIterator<Item = (CheckpointKey, HeldArtifact)>,
    ) -> Result<Vec<ArtifactRef>> {
        let mut unsettled = Vec::new();
        for (_, held) in held {
            let artifact = held.claim.artifact().clone();
            let settled = match catalog.reclaim_own_staged_artifact(held.claim).await {
                Ok(ReclaimDecision::Licensed(licence)) => {
                    self.reclaim(catalog, licence, &[]).await.map(|_| true)
                }
                Ok(ReclaimDecision::Absent) => Ok(true),
                Ok(ReclaimDecision::Referenced | ReclaimDecision::Live) => Ok(false),
                Err(e) => Err(e),
            };
            match settled {
                Ok(true) => {}
                Ok(false) => {
                    tracing::debug!(%artifact, "checkpoint reclaim refused");
                    unsettled.push(artifact);
                }
                Err(e) => {
                    tracing::debug!(%artifact, error = %e, "checkpoint reclaim failed");
                    unsettled.push(artifact);
                }
            }
        }
        Ok(unsettled)
    }

    /// Row first, bytes second: the one ordering every staged bundle shares.
    async fn stage_bundle<S: AsRef<str>>(
        &self,
        catalog: &Catalog,
        scope: StagingScope,
        prefix_segments: &[S],
        files: &[(String, Bytes)],
    ) -> Result<StagedArtifact> {
        let tenant = catalog.current_tenant();
        let prefix = self.prefix_url(tenant.as_ref(), prefix_segments)?;
        let staged = catalog.stage_model_artifact(&prefix, scope).await?;
        self.write_bundle(&prefix, files).await?;
        Ok(staged)
    }

    /// Delete a reclaimed artifact's bytes and retire its row — the one
    /// primitive that removes bytes under `models/`, reachable only with the
    /// [`ReclaimLicence`] the catalog's reclaim compare-and-set minted.
    ///
    /// Deletes the files the bundle's `manifest.json` lists, any `listed` key
    /// the caller found under the prefix (a reconcile pass's own listing —
    /// how a bundle torn before its manifest, or one whose manifest no longer
    /// parses, still converges), the `materialization.json` attestation, and
    /// the manifest itself LAST, so an interrupted reclaim still knows what it
    /// had left to delete. Every key passes the licence's always-on
    /// [`ReclaimLicence::covers`] check at the raw delete. Only when every
    /// delete has succeeded is the row retired; any failure leaves the
    /// artifact `reclaiming`, which the next reclaim of it resumes.
    pub async fn reclaim(
        &self,
        catalog: &Catalog,
        licence: ReclaimLicence,
        listed: &[ObjectPath],
    ) -> Result<Vec<ObjectPath>> {
        let prefix = licence.artifact().url().clone();
        let handle = self.handle(&prefix)?;
        let manifest_path = self.child(&prefix, MANIFEST_NAME)?;
        let mut keys: Vec<ObjectPath> = match self.read_manifest(&handle, &prefix).await {
            Ok(manifest) => manifest
                .files
                .iter()
                .map(|entry| self.child(&prefix, &entry.name))
                .collect::<Result<_>>()?,
            // Nothing published, or a manifest that no longer parses: the
            // caller's listing is then the only inventory there is.
            Err(JammiError::Storage(
                StorageError::NotPublished { .. } | StorageError::Layout { .. },
            )) => Vec::new(),
            Err(e) => return Err(e),
        };
        keys.extend(listed.iter().filter(|key| **key != manifest_path).cloned());
        keys.push(self.child(&prefix, MATERIALIZATION_NAME)?);
        keys.sort();
        keys.dedup();
        keys.push(manifest_path);

        let mut deleted = Vec::new();
        for key in keys {
            if handle.delete_licensed(&licence, &key).await? == DeleteOutcome::Deleted {
                deleted.push(key);
            }
        }
        catalog.retire_reclaimed_artifact(licence).await?;
        Ok(deleted)
    }

    /// Fetch the checkpoint a resume of `job_id` restores: the newest epoch
    /// whose manifest exists and verifies, or `None` when no epoch has one
    /// (the job has not completed an epoch boundary, so there is nothing to
    /// resume from — the worker starts from scratch).
    ///
    /// The candidates are the job's `staged` rows in `catalog`
    /// ([`Catalog::job_scoped_artifacts`] — the row is written before the
    /// first byte, so every prefix a writer touched is named), newest key
    /// first. A candidate with no manifest is a write that never completed
    /// (a crash between its first PUT and its manifest PUT), which is
    /// "nothing published here", never corruption: the read moves on to the
    /// epoch before it. A manifest that exists but does not verify — a digest
    /// mismatch, a listed key gone — is corruption and is the hard
    /// [`StorageError::Layout`] error [`Self::fetch_artifact`] raises, never a
    /// silent from-scratch restart.
    pub async fn fetch_newest_checkpoint(
        &self,
        catalog: &Catalog,
        job_id: &str,
    ) -> Result<Option<LocalArtifact>> {
        for (_, held) in self.held_checkpoints(catalog, job_id).await? {
            if held.state != ArtifactState::Staged {
                continue;
            }
            match self.fetch_artifact(held.claim.artifact().url()).await {
                Ok(local) => return Ok(Some(local)),
                Err(JammiError::Storage(StorageError::NotPublished { .. })) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(None)
    }

    /// Read and parse `manifest.json` under `prefix`. A manifest absent
    /// entirely reclassifies to [`StorageError::NotPublished`] — "no bundle
    /// published here", not corruption (see [`reclassify_missing_manifest`]).
    /// A manifest present but malformed is a hard [`StorageError::Layout`]
    /// error — it WAS published, but what's there is not valid JSON.
    async fn read_manifest(
        &self,
        handle: &JammiObjectStore,
        prefix: &StorageUrl,
    ) -> Result<Manifest> {
        let manifest_path = self.child(prefix, MANIFEST_NAME)?;
        let bytes = handle
            .get_bytes(&manifest_path)
            .await
            .map_err(|e| reclassify_missing_manifest(e, prefix))?;
        serde_json::from_slice(&bytes).map_err(|e| {
            JammiError::Storage(StorageError::layout(
                prefix.as_str(),
                format!("malformed artifact manifest: {e}"),
            ))
        })
    }

    /// Verify every manifest key exists under `prefix` and hashes to its recorded
    /// digest. Used by the `file://` in-place path to catch a partial write
    /// without copying any bytes off the local path.
    async fn verify_files(
        &self,
        handle: &JammiObjectStore,
        prefix: &StorageUrl,
        manifest: &Manifest,
    ) -> Result<()> {
        for entry in &manifest.files {
            let path = self.child(prefix, &entry.name)?;
            let bytes = handle
                .get_bytes(&path)
                .await
                .map_err(|e| reclassify_missing_key(e, prefix, &entry.name))?;
            verify_sha256(prefix, entry, &bytes)?;
        }
        Ok(())
    }

    /// Open a [`JammiObjectStore`] handle for a prefix URL.
    fn handle(&self, prefix: &StorageUrl) -> Result<JammiObjectStore> {
        let driver = self.registry.driver_for(prefix, None)?;
        Ok(JammiObjectStore::new(driver, prefix.clone()))
    }

    /// The object-store path of `name` directly under `prefix`. The prefix URL's
    /// own `path` is the artifact directory, so `name` is joined onto it and the
    /// cloud-bucket leading segment stripped (mirroring
    /// [`JammiObjectStore`]'s path parsing).
    fn child(&self, prefix: &StorageUrl, name: &str) -> Result<ObjectPath> {
        Ok(prefix.object_key(&format!("{}/{}", prefix.path(), name))?)
    }

    /// Join the tenant segment plus attempt-unique segments under the store
    /// root to form the artifact prefix URL:
    /// `{root}/{TenantSegment::of(tenant)}/{segments…}`. Each segment is
    /// sanitized so a `job_id`/`worker_id` carrying a `/` cannot escape the
    /// prefix or collide across attempts.
    ///
    /// `pub`: this is the ONE place an artifact prefix is built, so a test
    /// asserting on the SHAPE of an artifact's prefix (never its bytes) calls
    /// this instead of hand-building the layout string.
    pub fn prefix_url<S: AsRef<str>>(
        &self,
        tenant: Option<&TenantId>,
        segments: &[S],
    ) -> Result<StorageUrl> {
        let root = self.root.as_str().trim_end_matches('/');
        let mut joined = String::from(root);
        joined.push('/');
        joined.push_str(&TenantSegment::of(tenant));
        for seg in segments {
            joined.push('/');
            joined.push_str(&sanitize_segment(seg.as_ref()));
        }
        StorageUrl::parse(&joined).map_err(JammiError::from)
    }

    /// The object key of the `manifest.json` a bundle at `prefix` carries.
    pub(crate) fn manifest_path(&self, prefix: &StorageUrl) -> Result<ObjectPath> {
        self.child(prefix, MANIFEST_NAME)
    }

    /// The full set of object keys a published bundle at `prefix` is
    /// expected to carry, read from its `manifest.json`: `None` when no
    /// manifest exists at all (nothing was ever published there — an absent
    /// manifest is not an error, see `reclassify_missing_manifest`);
    /// `Some(keys)` — the manifest itself plus every entry it lists — when
    /// one does. A read error OTHER than "no manifest" propagates: this
    /// method never silently treats a transport fault as "nothing to
    /// expect".
    ///
    /// The sole caller is `reconcile`'s artifact arm, which checks a
    /// referenced artifact's listed keys against this set to report a
    /// damaged bundle.
    pub async fn expected_objects(&self, prefix: &StorageUrl) -> Result<Option<Vec<ObjectPath>>> {
        let handle = self.handle(prefix)?;
        let manifest = match self.read_manifest(&handle, prefix).await {
            Ok(m) => m,
            Err(JammiError::Storage(StorageError::NotPublished { .. })) => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut paths = Vec::with_capacity(manifest.files.len() + 1);
        paths.push(self.child(prefix, MANIFEST_NAME)?);
        for entry in &manifest.files {
            paths.push(self.child(prefix, &entry.name)?);
        }
        Ok(Some(paths))
    }
}

/// Re-type a `manifest.json` `get_bytes` fault as "no bundle is published at
/// this prefix at all" ([`StorageError::NotPublished`]) when the driver
/// reports the manifest does not exist, vs. leaving every other driver
/// failure as the transport/IO fault it is.
///
/// No manifest is in hand at this point, so there is nothing to say the
/// bundle's *content* is broken — the honest claim is narrower: this prefix
/// was never published, or a catalog pointer names the wrong prefix (e.g. a
/// base weights directory). This is a DIFFERENT failure class than
/// [`reclassify_missing_key`]'s "a manifest-listed key is gone" — that one
/// DOES have a manifest in hand naming exactly what's missing, which is
/// genuine bundle corruption. Conflating the two would call "nothing was
/// ever published here" a corrupted bundle, which is a claim this call site
/// carries no evidence for. Any other driver error (network fault,
/// throttling, credential rot, a 5xx) is a genuine transport/IO problem and
/// is left as [`StorageError::Io`] unchanged.
fn reclassify_missing_manifest(err: StorageError, prefix: &StorageUrl) -> JammiError {
    match &err {
        StorageError::NotFound { .. } => {
            JammiError::Storage(StorageError::not_published(prefix.as_str()))
        }
        _ => JammiError::from(err),
    }
}

/// Re-type a `get_bytes` fault reading `name` under `prefix` as an INTEGRITY
/// failure of the bundle when the driver reports the key does not exist, vs.
/// leaving every other driver failure as the transport/IO fault it is.
///
/// A manifest names its keys after they were already written (a bundle write
/// puts the manifest last), so once a fetcher has a manifest in hand, a
/// `NotFound` on a key it names can only mean the object was deleted out from
/// under a completed bundle, or the bundle was a partial/tampered write — the
/// storage layer itself is healthy, the *bundle* is broken. That is the same
/// failure class [`StorageError::Layout`] already carries for a malformed
/// manifest or a digest mismatch, so this folds a missing key into the same
/// variant rather than minting a fourth error path — callers that already
/// match `StorageError::Layout` to detect "this artifact's bytes are wrong"
/// catch a missing key for free. Distinct from [`reclassify_missing_manifest`]
/// (the manifest itself absent, meaning nothing was ever published — never
/// bundle corruption) — see that function's doc for why the two must not be
/// conflated. Any other driver error (network fault, throttling, credential
/// rot, a 5xx) is a genuine transport/IO problem unrelated to this bundle's
/// own integrity and is left as [`StorageError::Io`] unchanged, so a caller
/// can still tell "the store was unreachable" from "this artifact is
/// corrupt".
fn reclassify_missing_key(err: StorageError, prefix: &StorageUrl, name: &str) -> JammiError {
    match &err {
        StorageError::NotFound { .. } => JammiError::Storage(StorageError::layout(
            prefix.as_str(),
            format!("artifact file '{name}' missing under this prefix (manifest lists it)"),
        )),
        _ => JammiError::from(err),
    }
}

/// Verify `bytes` hash to `entry.sha256`, erroring with the prefix context on a
/// mismatch (a partial PUT or a tampered object, never a torn load).
fn verify_sha256(prefix: &StorageUrl, entry: &ManifestEntry, bytes: &[u8]) -> Result<()> {
    let actual = sha256_hex(bytes);
    if actual != entry.sha256 {
        return Err(JammiError::Storage(StorageError::layout(
            prefix.as_str(),
            format!(
                "artifact file '{}' sha256 {actual} does not match manifest {}",
                entry.name, entry.sha256
            ),
        )));
    }
    Ok(())
}

/// The segment naming one epoch's checkpoint: `epoch_{N}`.
fn epoch_segment(epoch: usize) -> String {
    format!("epoch_{epoch}")
}

/// The prefix `epoch_{N}` names.
fn parse_epoch_segment(segment: &str) -> Option<usize> {
    segment.strip_prefix("epoch_")?.parse().ok()
}

/// Which write of a job's checkpoint a prefix holds: the attempt that wrote
/// it and the epoch it completes. Ordered by attempt first, then epoch, so
/// the newest key is the latest attempt's latest epoch — a successor's write
/// is always newer than anything an earlier attempt left, including a bundle
/// a lost-lease writer finished after its successor had already resumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CheckpointKey {
    attempt: u32,
    epoch: usize,
}

/// The prefix segments under the tenant segment one checkpoint write lands
/// at: `{job_id}/_checkpoints/{attempt}/epoch_{N}`.
fn checkpoint_segments(job_id: &str, key: CheckpointKey) -> [String; 4] {
    [
        job_id.to_string(),
        CHECKPOINTS_SEGMENT.to_string(),
        key.attempt.to_string(),
        epoch_segment(key.epoch),
    ]
}

/// Read a checkpoint prefix's key back off the layout [`checkpoint_segments`]
/// built: the last two segments of `{job_id}/_checkpoints/{attempt}/epoch_{N}`.
/// A job-scoped row is only ever staged by [`ArtifactStore::stage_checkpoint`],
/// so any other shape is a layout the store did not write.
fn checkpoint_key(prefix: &StorageUrl) -> Result<CheckpointKey> {
    let mut segments = prefix.path().trim_end_matches('/').rsplit('/');
    let key = segments
        .next()
        .and_then(parse_epoch_segment)
        .and_then(|epoch| {
            let attempt = segments.next()?.parse().ok()?;
            Some(CheckpointKey { attempt, epoch })
        });
    key.ok_or_else(|| {
        JammiError::Storage(StorageError::layout(
            prefix.as_str(),
            "a checkpoint prefix ends in `{attempt}/epoch_{N}`",
        ))
    })
}

/// Test-only rendezvous at the one seam a crash inside a bundle write can
/// land in: between the last data-file PUT and the manifest PUT. A test arms
/// the prefix it wants to see torn, waits for the writer to park there, and
/// then treats that writer as dead — the handle never releases on drop, so
/// the prefix stays exactly as a killed process would leave it: every data
/// file present, no manifest. Compiled only under `feature = "test-hooks"`;
/// the production write observes nothing but the one `maybe_park_*` call
/// (a no-op whenever nothing is armed).
#[cfg(feature = "test-hooks")]
pub mod artifact_test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use tokio::sync::Notify;

    use crate::storage::StorageUrl;

    struct Armed {
        prefix: StorageUrl,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    fn armed() -> &'static Mutex<Vec<Armed>> {
        static ARMED: OnceLock<Mutex<Vec<Armed>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of one armed park: wait for the writer to arrive,
    /// then either let it continue or leave it parked. Dropping the handle
    /// leaves a parked writer parked.
    pub struct ParkHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    impl ParkHandle {
        /// Resolve once the writer has reached the seam. A test bounds this
        /// with its own backstop.
        pub async fn wait_parked(&self) {
            while !self.parked.load(Ordering::SeqCst) {
                self.parked_notify.notified().await;
            }
        }

        /// Whether the writer has reached the seam.
        pub fn is_parked(&self) -> bool {
            self.parked.load(Ordering::SeqCst)
        }

        /// Let the parked writer go on to its manifest PUT.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_one();
        }
    }

    /// Arm a one-shot park for the next bundle written under exactly
    /// `prefix` (an epoch checkpoint of a job names its prefix through
    /// `ArtifactStore::checkpoint_prefix`), at the seam
    /// between its last data-file PUT and its manifest PUT. Keyed by the
    /// full prefix, so sibling tests in one binary never park each other's
    /// writers.
    pub fn arm_park_before_manifest(prefix: &StorageUrl) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Armed {
                prefix: prefix.clone(),
                parked: Arc::clone(&parked),
                parked_notify: Arc::clone(&parked_notify),
                released: Arc::clone(&released),
                release_notify: Arc::clone(&release_notify),
            });
        ParkHandle {
            parked,
            parked_notify,
            released,
            release_notify,
        }
    }

    pub(super) async fn maybe_park_before_manifest(prefix: &StorageUrl) {
        let taken = {
            let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.prefix == *prefix)
                .map(|i| list.remove(i))
        };
        let Some(armed) = taken else {
            return;
        };
        armed.parked.store(true, Ordering::SeqCst);
        armed.parked_notify.notify_one();
        while !armed.released.load(Ordering::SeqCst) {
            armed.release_notify.notified().await;
        }
    }
}

/// Sanitize one prefix segment: replace path-ambiguous characters so a segment
/// is always a single, collision-free path component.
fn sanitize_segment(seg: &str) -> String {
    seg.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | ' ' => '_',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A retention window of one: the newest checkpoint alone survives.
    const ONE: NonZeroUsize = NonZeroUsize::MIN;

    fn store_with_root(root: StorageUrl, cache: PathBuf) -> ArtifactStore {
        ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap()
    }

    /// A fresh catalog for the staged rows a bundle write goes through.
    async fn test_catalog() -> (tempfile::TempDir, Catalog) {
        let dir = tempfile::tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).await.unwrap();
        (dir, catalog)
    }

    /// Stage a bundle as attempt `attempt` of `job`.
    async fn staged(
        store: &ArtifactStore,
        catalog: &Catalog,
        job: &str,
        attempt: u32,
        files: &[(String, Bytes)],
    ) -> StagedArtifact {
        store
            .stage_attempt_artifact(catalog, job, "worker-a", attempt, files)
            .await
            .unwrap()
    }

    /// [`staged`], returning only the bundle's prefix.
    async fn staged_prefix(
        store: &ArtifactStore,
        catalog: &Catalog,
        job: &str,
        attempt: u32,
        files: &[(String, Bytes)],
    ) -> StorageUrl {
        staged(store, catalog, job, attempt, files)
            .await
            .artifact()
            .url()
            .clone()
    }

    fn sample_files() -> Vec<(String, Bytes)> {
        vec![
            (
                "adapter.safetensors".to_string(),
                Bytes::from_static(b"weights-bytes"),
            ),
            (
                "adapter_config.json".to_string(),
                Bytes::from_static(b"{\"adapter_type\":\"x\"}"),
            ),
        ]
    }

    #[tokio::test]
    async fn memory_round_trip_fetches_manifest_keys() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(StorageUrl::memory("artifacts"), cache.path().to_path_buf());
        let files = sample_files();

        let prefix = staged_prefix(&store, &catalog, "job-1", 0, &files).await;
        assert!(prefix
            .as_str()
            .ends_with("artifacts/_global/job-1/worker-a/0"));

        let fetched = store.fetch_artifact(&prefix).await.unwrap();
        for (name, bytes) in &files {
            let got = std::fs::read(fetched.dir().join(name)).unwrap();
            assert_eq!(&got[..], &bytes[..], "fetched file '{name}' differs");
        }
        // The manifest itself is not materialised as a loadable artifact file.
        assert!(!fetched.dir().join(MANIFEST_NAME).exists());
    }

    #[tokio::test]
    async fn file_scheme_reads_in_place_without_copy() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let root_dir = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let root = StorageUrl::parse(root_dir.path().to_str().unwrap()).unwrap();
        let store = store_with_root(root, cache.path().to_path_buf());
        let files = sample_files();

        let prefix = staged_prefix(&store, &catalog, "job-2", 1, &files).await;
        let fetched = store.fetch_artifact(&prefix).await.unwrap();

        // The returned dir is the prefix path itself (in place), under the
        // file:// root — not a copy under the cache root.
        assert_eq!(fetched.dir(), std::path::Path::new(prefix.path()));
        assert!(fetched.dir().starts_with(root_dir.path()));
        assert!(!fetched.dir().starts_with(cache.path()));
        for (name, bytes) in &files {
            let got = std::fs::read(fetched.dir().join(name)).unwrap();
            assert_eq!(&got[..], &bytes[..]);
        }
    }

    #[tokio::test]
    async fn sha256_mismatch_is_a_hard_error() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-corrupt"),
            cache.path().to_path_buf(),
        );
        let prefix = staged_prefix(&store, &catalog, "job-3", 0, &sample_files()).await;

        // Overwrite one data file with different bytes — the manifest digest no
        // longer matches, so fetch must refuse rather than load torn weights.
        let handle = store.handle(&prefix).unwrap();
        let path = store.child(&prefix, "adapter.safetensors").unwrap();
        handle
            .put_bytes(&path, Bytes::from_static(b"tampered"))
            .await
            .unwrap();

        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        assert!(
            err.to_string().contains("does not match manifest"),
            "expected a sha256 mismatch error, got: {err}"
        );
    }

    #[tokio::test]
    async fn missing_manifest_is_not_published_not_corruption() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-nomanifest"),
            cache.path().to_path_buf(),
        );
        // A prefix that was never written — no manifest exists.
        let prefix = store.prefix_url(None, &["ghost", "worker", "0"]).unwrap();
        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        // The manifest GET 404s before any data file is even named. No
        // manifest is in hand, so there is nothing to say the bundle's
        // CONTENT is broken — this is "no bundle published here"
        // (`StorageError::NotPublished`), never the `Layout` integrity-failure
        // class a malformed manifest or a listed-key-absent 404 carries (those
        // DO have a manifest in hand).
        assert!(
            matches!(err, JammiError::Storage(StorageError::NotPublished { .. })),
            "a missing manifest.json must reclassify to StorageError::NotPublished, not the \
             Layout integrity-failure class (no manifest is in hand to say anything is \
             corrupt), got: {err:?}"
        );
        assert!(
            err.to_string().contains(prefix.as_str()),
            "the NotPublished error must name the prefix nothing was published at, got: {err}"
        );
    }

    /// A manifest that DOES exist but names a key the store no longer has (the
    /// object was deleted out from under a completed bundle) is the same
    /// INTEGRITY failure class as a missing manifest or a digest mismatch —
    /// `StorageError::Layout` — not the transport fault a `NotFound` might
    /// otherwise suggest. The reclassified message still names the key so a
    /// caller (e.g. `ModelResolver`) can surface which file is gone.
    #[tokio::test]
    async fn missing_manifest_listed_key_reclassifies_as_integrity_failure() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-missing-key"),
            cache.path().to_path_buf(),
        );
        let prefix = staged_prefix(&store, &catalog, "job-4", 0, &sample_files()).await;
        let handle = store.handle(&prefix).unwrap();
        let path = store.child(&prefix, "adapter.safetensors").unwrap();
        handle.delete_if_exists(&path).await.unwrap();

        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        assert!(
            matches!(err, JammiError::Storage(StorageError::Layout { .. })),
            "a manifest-listed key absent from the store must reclassify to \
             StorageError::Layout (an integrity failure), not stay StorageError::Io (a \
             transport fault), got: {err:?}"
        );
        assert!(
            err.to_string().contains("adapter.safetensors"),
            "the reclassified error must still name the missing key, got: {err}"
        );
    }

    /// The flip side of the two tests above: a key the manifest lists that IS
    /// present but unreadable for a reason that has nothing to do with the
    /// bundle's own content — a permission fault standing in for a transient
    /// object-store outage — must NOT be folded into the same
    /// `StorageError::Layout` integrity bucket. `object_store`'s
    /// `LocalFileSystem` folds a permission-denied open into
    /// `Error::Generic` (its own `UnableToOpenFile` is a private local error,
    /// never constructed outside that crate), never `Error::NotFound`, so
    /// `reclassify_missing_key` must leave it as `StorageError::Io` —
    /// verified with a real `chmod` fault injection (Unix-only), never a
    /// hand-built error the reclassifier was never actually asked to sort.
    #[cfg(all(unix, feature = "unprivileged-tests"))]
    #[tokio::test]
    async fn permission_fault_on_a_present_key_stays_a_transport_error() {
        let (_catalog_dir, catalog) = test_catalog().await;
        use std::os::unix::fs::PermissionsExt;
        jammi_test_resources::assert_permissions_enforced();

        let root_dir = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let root = StorageUrl::parse(root_dir.path().to_str().unwrap()).unwrap();
        let store = store_with_root(root, cache.path().to_path_buf());
        let prefix = staged_prefix(&store, &catalog, "job-5", 0, &sample_files()).await;
        let weights_path = std::path::PathBuf::from(prefix.path()).join("adapter.safetensors");

        std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o644))
            .expect("restore permissions");
        assert!(
            matches!(err, JammiError::Storage(StorageError::Io { .. })),
            "a permission-denied open is a genuine driver/transport fault, not this bundle's \
             own integrity — it must stay StorageError::Io, never be reclassified to Layout, \
             got: {err:?}"
        );
    }

    #[tokio::test]
    async fn cache_hit_avoids_redownload() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-cache"),
            cache.path().to_path_buf(),
        );
        let prefix = staged_prefix(&store, &catalog, "job-4", 0, &sample_files()).await;

        let first = store.fetch_artifact(&prefix).await.unwrap();
        // Mutate the cached file on disk; a second fetch that re-downloaded would
        // overwrite it back, but a cache hit returns the same dir untouched.
        let marker = first.dir().join("adapter.safetensors");
        std::fs::write(&marker, b"locally-edited").unwrap();
        let second = store.fetch_artifact(&prefix).await.unwrap();
        assert_eq!(first.dir(), second.dir());
        assert_eq!(std::fs::read(&marker).unwrap(), b"locally-edited");
    }

    #[tokio::test]
    async fn concurrent_fetch_is_torn_free() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = Arc::new(store_with_root(
            StorageUrl::memory("artifacts-concurrent"),
            cache.path().to_path_buf(),
        ));
        let files = sample_files();
        let prefix = staged_prefix(&store, &catalog, "job-5", 0, &files).await;

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
            let prefix = prefix.clone();
            handles.push(tokio::spawn(
                async move { store.fetch_artifact(&prefix).await },
            ));
        }
        for h in handles {
            let fetched = h.await.unwrap().unwrap();
            for (name, bytes) in &files {
                let got = std::fs::read(fetched.dir().join(name)).unwrap();
                assert_eq!(&got[..], &bytes[..], "torn fetch of '{name}'");
            }
        }
    }

    /// One epoch's checkpoint bundle, distinguishable by its epoch.
    fn resume_files(epoch: usize) -> Vec<(String, Bytes)> {
        vec![
            (
                "adapter.safetensors".to_string(),
                Bytes::from(format!("weights-epoch-{epoch}")),
            ),
            (
                "resume_state.json".to_string(),
                Bytes::from(format!("{{\"last_completed_epoch\":{epoch}}}")),
            ),
        ]
    }

    /// The epoch the fetched resume bundle's state names.
    fn fetched_epoch(local: &LocalArtifact) -> String {
        String::from_utf8(std::fs::read(local.dir().join("resume_state.json")).unwrap()).unwrap()
    }

    /// The prefixes of the job's checkpoint rows the catalog still holds.
    async fn resume_rows(catalog: &Catalog, job: &str) -> Vec<String> {
        catalog
            .job_scoped_artifacts(job)
            .await
            .unwrap()
            .iter()
            .map(|held| held.claim.artifact().url().as_str().to_string())
            .collect()
    }

    /// Whether any object under `prefix` — a bundle's manifest, or a torn
    /// write's data file — still exists.
    async fn object_exists(store: &ArtifactStore, prefix: &StorageUrl, name: &str) -> bool {
        store
            .handle(prefix)
            .unwrap()
            .exists(&store.child(prefix, name).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn the_newest_complete_epoch_is_read_and_epochs_beyond_the_window_are_retired() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume"),
            cache.path().to_path_buf(),
        );

        // No checkpoint written yet → None (start from scratch).
        assert!(store
            .fetch_newest_checkpoint(&catalog, "job-r")
            .await
            .unwrap()
            .is_none());

        // Epoch 0, then epoch 1, each under its own prefix; once epoch 1's
        // manifest has landed, epoch 0 is retired — bytes and row.
        let epoch_0 = store
            .stage_checkpoint(&catalog, "job-r", 1, 0, ONE, &resume_files(0))
            .await
            .unwrap();
        let epoch_1 = store
            .stage_checkpoint(&catalog, "job-r", 1, 1, ONE, &resume_files(1))
            .await
            .unwrap();
        assert_ne!(epoch_0.artifact(), epoch_1.artifact());
        assert_eq!(
            epoch_1.artifact().url(),
            &store.checkpoint_prefix(None, "job-r", 1, 1).unwrap()
        );

        let fetched = store
            .fetch_newest_checkpoint(&catalog, "job-r")
            .await
            .unwrap()
            .expect("a written checkpoint is fetchable");
        assert_eq!(fetched_epoch(&fetched), "{\"last_completed_epoch\":1}");
        assert!(!object_exists(&store, epoch_0.artifact().url(), MANIFEST_NAME).await);
        assert!(!object_exists(&store, epoch_0.artifact().url(), "adapter.safetensors").await);
        assert_eq!(
            resume_rows(&catalog, "job-r").await,
            vec![epoch_1.artifact().url().as_str().to_string()],
            "one complete checkpoint is held once the newer epoch's manifest has landed"
        );

        // No `jobs` row keeps this stager live, so the job's checkpoint is
        // reclaimable: afterwards nothing is held and the next fetch is None.
        assert!(store
            .reclaim_checkpoints(&catalog, "job-r")
            .await
            .unwrap()
            .is_empty());
        assert!(resume_rows(&catalog, "job-r").await.is_empty());
        assert!(!object_exists(&store, epoch_1.artifact().url(), MANIFEST_NAME).await);
        assert!(store
            .fetch_newest_checkpoint(&catalog, "job-r")
            .await
            .unwrap()
            .is_none());
    }

    /// A writer that dies between its last data-file PUT and its manifest
    /// PUT leaves a prefix with no manifest: the read skips it and resumes
    /// from the epoch before, never reporting corruption. The successor's
    /// own write of that epoch lands under its own attempt's prefix and
    /// retires the torn row.
    #[cfg(feature = "test-hooks")]
    #[tokio::test]
    async fn a_checkpoint_write_torn_before_its_manifest_is_skipped_never_corruption() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = Arc::new(store_with_root(
            StorageUrl::memory("artifacts-resume-torn"),
            cache.path().to_path_buf(),
        ));
        let catalog = Arc::new(catalog);
        store
            .stage_checkpoint(&catalog, "job-t", 1, 0, ONE, &resume_files(0))
            .await
            .unwrap();

        // Attempt 1's epoch-1 write parks at the seam and is then killed.
        let torn = store.checkpoint_prefix(None, "job-t", 1, 1).unwrap();
        let park = artifact_test_hooks::arm_park_before_manifest(&torn);
        let writer = tokio::spawn({
            let store = Arc::clone(&store);
            let catalog = Arc::clone(&catalog);
            async move {
                store
                    .stage_checkpoint(&catalog, "job-t", 1, 1, ONE, &resume_files(1))
                    .await
            }
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), park.wait_parked())
            .await
            .expect("the writer reaches the seam");
        writer.abort();
        assert!(writer.await.unwrap_err().is_cancelled());
        assert!(object_exists(&store, &torn, "adapter.safetensors").await);
        assert!(!object_exists(&store, &torn, MANIFEST_NAME).await);

        let fetched = store
            .fetch_newest_checkpoint(&catalog, "job-t")
            .await
            .unwrap()
            .expect("epoch 0 is still the complete checkpoint");
        assert_eq!(fetched_epoch(&fetched), "{\"last_completed_epoch\":0}");

        // Attempt 2 resumes from epoch 0 and completes epoch 1 under its own
        // prefix; the torn row and epoch 0 are both retired behind it.
        let successor = store
            .stage_checkpoint(&catalog, "job-t", 2, 1, ONE, &resume_files(1))
            .await
            .unwrap();
        assert_ne!(successor.artifact().url(), &torn);
        let fetched = store
            .fetch_newest_checkpoint(&catalog, "job-t")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched_epoch(&fetched), "{\"last_completed_epoch\":1}");
        assert_eq!(
            resume_rows(&catalog, "job-t").await,
            vec![successor.artifact().url().as_str().to_string()]
        );
    }

    /// A manifest that exists and does not verify is corruption: the read
    /// fails with the store's integrity error, never a silent from-scratch
    /// restart and never a fall-through to an older epoch.
    #[tokio::test]
    async fn a_checkpoint_manifest_that_does_not_verify_fails_loudly() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume-corrupt"),
            cache.path().to_path_buf(),
        );
        // A complete epoch 0 sits beneath the corrupted epoch 1, inside a
        // window of two: an older epoch is no excuse to read past corruption.
        let two = NonZeroUsize::new(2).unwrap();
        let epoch_1 = store
            .stage_checkpoint(&catalog, "job-c", 1, 1, two, &resume_files(1))
            .await
            .unwrap();
        store
            .stage_checkpoint(&catalog, "job-c", 1, 0, two, &resume_files(0))
            .await
            .unwrap();
        assert_eq!(resume_rows(&catalog, "job-c").await.len(), 2);
        let prefix = epoch_1.artifact().url();
        store
            .handle(prefix)
            .unwrap()
            .put_bytes(
                &store.child(prefix, "adapter.safetensors").unwrap(),
                Bytes::from_static(b"tampered"),
            )
            .await
            .unwrap();

        let err = store
            .fetch_newest_checkpoint(&catalog, "job-c")
            .await
            .unwrap_err();
        assert!(
            matches!(err, JammiError::Storage(StorageError::Layout { .. })),
            "a manifest that does not verify is corruption, got: {err:?}"
        );
        assert!(err.to_string().contains("does not match manifest"), "{err}");
    }

    /// The read's order is attempt first, then epoch: a successor's write is
    /// newer than any epoch an earlier attempt left. Every key round-trips
    /// through the prefix the store builds for it.
    #[test]
    fn checkpoint_keys_order_by_attempt_then_epoch_and_round_trip_their_prefix() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume-keys"),
            cache.path().to_path_buf(),
        );
        let keys = [(1, 0), (1, 3), (2, 1), (2, 2)];
        let parsed: Vec<CheckpointKey> = keys
            .iter()
            .map(|&(attempt, epoch)| {
                let prefix = store
                    .checkpoint_prefix(None, "job-k", attempt, epoch)
                    .unwrap();
                assert!(prefix.as_str().ends_with(&format!(
                    "artifacts-resume-keys/_global/job-k/_checkpoints/{attempt}/epoch_{epoch}"
                )));
                checkpoint_key(&prefix).unwrap()
            })
            .collect();
        assert!(parsed.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            parsed.iter().max().unwrap(),
            &CheckpointKey {
                attempt: 2,
                epoch: 2
            }
        );
        let not_resume = store.prefix_url(None, &["job-k", "worker", "0"]).unwrap();
        assert!(matches!(
            checkpoint_key(&not_resume).unwrap_err(),
            JammiError::Storage(StorageError::Layout { .. })
        ));
    }

    /// A retention window of `n` keeps the `n` newest keys across attempts:
    /// a successor's writes push an earlier attempt's epochs out of the
    /// window one by one, and the window's own bundles all read back whole.
    #[tokio::test]
    async fn the_retention_window_spans_attempts_and_keeps_the_newest_keys() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-window"),
            cache.path().to_path_buf(),
        );
        let two = NonZeroUsize::new(2).unwrap();
        let mut written = Vec::new();
        for (attempt, epoch) in [(1, 0), (1, 1), (2, 2)] {
            let staged = store
                .stage_checkpoint(&catalog, "job-w", attempt, epoch, two, &resume_files(epoch))
                .await
                .unwrap();
            written.push(staged.artifact().url().as_str().to_string());
        }
        assert_eq!(
            resume_rows(&catalog, "job-w").await,
            written[1..].to_vec(),
            "the two newest keys — attempt 1's epoch 1 and attempt 2's epoch 2 — are held"
        );
        let (retired, kept) = written.split_first().unwrap();
        assert!(!object_exists(&store, &StorageUrl::parse(retired).unwrap(), MANIFEST_NAME).await);
        for prefix in kept {
            store
                .fetch_artifact(&StorageUrl::parse(prefix).unwrap())
                .await
                .expect("every bundle inside the window is complete");
        }
        let newest = store
            .fetch_newest_checkpoint(&catalog, "job-w")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched_epoch(&newest), "{\"last_completed_epoch\":2}");
    }

    #[tokio::test]
    async fn the_checkpoint_prefix_is_disjoint_from_the_attempt_prefix() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume-disjoint"),
            cache.path().to_path_buf(),
        );
        // Stage an attempt's served bundle and a resume checkpoint for the
        // same job; neither read sees the other's bytes — the resume side
        // channel never perturbs the served path.
        let served = staged_prefix(&store, &catalog, "job-d", 0, &sample_files()).await;
        store
            .stage_checkpoint(&catalog, "job-d", 0, 3, ONE, &resume_files(3))
            .await
            .unwrap();

        assert!(served.as_str().ends_with("job-d/worker-a/0"));
        let fetched = store.fetch_artifact(&served).await.unwrap();
        assert!(fetched.dir().join("adapter.safetensors").exists());
        assert!(!fetched.dir().join("resume_state.json").exists());

        // Reclaiming the resume checkpoint leaves the served bundle untouched.
        assert!(store
            .reclaim_checkpoints(&catalog, "job-d")
            .await
            .unwrap()
            .is_empty());
        let fetched_again = store.fetch_artifact(&served).await.unwrap();
        assert!(fetched_again.dir().join("adapter.safetensors").exists());
    }

    #[tokio::test]
    async fn a_bundle_lands_under_the_global_segment_when_untenanted() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-seg"),
            cache.path().to_path_buf(),
        );
        let prefix = staged_prefix(&store, &catalog, "job-g", 0, &sample_files()).await;
        assert!(
            prefix
                .as_str()
                .ends_with("artifacts-seg/_global/job-g/worker-a/0"),
            "an untenanted artifact must land under the `_global` segment, got: {prefix}"
        );
    }

    #[tokio::test]
    async fn a_bundle_lands_under_its_catalogs_tenant_segment() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-tenant"),
            cache.path().to_path_buf(),
        );
        let tenant = TenantId::from_uuid(uuid::Builder::from_bytes([42; 16]).into_uuid()).unwrap();
        let pinned = catalog.pinned_to_tenant(Some(tenant));
        let prefix = staged_prefix(&store, &pinned, "job-t", 0, &sample_files()).await;
        assert!(
            prefix
                .as_str()
                .ends_with(&format!("artifacts-tenant/{tenant}/job-t/worker-a/0")),
            "a tenant's artifact must land under its own tenant segment, got: {prefix}"
        );
    }

    #[tokio::test]
    async fn expected_objects_is_none_for_an_unpublished_prefix() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-expected-none"),
            cache.path().to_path_buf(),
        );
        let prefix = store.prefix_url(None, &["ghost", "worker", "0"]).unwrap();
        assert!(store.expected_objects(&prefix).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn expected_objects_lists_the_manifest_and_every_entry() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-expected-some"),
            cache.path().to_path_buf(),
        );
        let prefix = staged_prefix(&store, &catalog, "job-e", 0, &sample_files()).await;
        let objects = store.expected_objects(&prefix).await.unwrap().unwrap();
        let names: Vec<String> = objects.iter().map(|p| p.to_string()).collect();
        assert!(names.iter().any(|n| n.ends_with(MANIFEST_NAME)));
        for (name, _) in sample_files() {
            assert!(
                names.iter().any(|n| n.ends_with(&name)),
                "expected_objects must list every manifest entry, missing '{name}' in {names:?}"
            );
        }
    }

    #[test]
    fn combined_hash_is_stable_for_identical_content() {
        let m1 = Manifest {
            files: vec![
                ManifestEntry {
                    name: "a".into(),
                    sha256: "11".into(),
                },
                ManifestEntry {
                    name: "b".into(),
                    sha256: "22".into(),
                },
            ],
        };
        let m2 = m1.clone();
        assert_eq!(m1.combined_hash(), m2.combined_hash());
    }

    // ─── The model `materialization.json` attestation ──────────────────────

    fn fine_tune_descriptor() -> crate::store::manifest::ProducingDescriptor {
        crate::store::manifest::ProducingDescriptor::FineTune {
            training_set_definition_hash: "a".repeat(64),
            training_set_artifact_digest: "b".repeat(64),
            training_set_row_count: 128,
            spec_canonical: r#"{"base_model":"bert-base"}"#.into(),
            spec_schema_version: 1,
            base_model_id: "bert-base-uncased".into(),
            world_size: 1,
            collective: "noop".into(),
            local_ranks: 1,
        }
    }

    fn fine_tune_env() -> crate::store::manifest::MaterializationEnv {
        crate::store::manifest::MaterializationEnv::new(
            crate::store::manifest::ComputeDevice::Cpu,
            vec![crate::store::manifest::ModelIdentity {
                model_id: "bert-base-uncased".into(),
                backend: "candle".into(),
                compute_precision: crate::store::manifest::ComputePrecision::F32,
                content_digest: crate::store::manifest::ModelContentDigest::Sha256(
                    "fixture-digest".into(),
                ),
                quantization: None,
            }],
        )
    }

    /// `write_model_materialization` writes `materialization.json` into a
    /// model artifact prefix, readable back byte-for-byte, folding the
    /// RIGHT artifact digest (the bundle's `combined_hash`, not an
    /// arbitrary one).
    #[tokio::test]
    async fn write_model_materialization_round_trips_and_folds_the_bundle_digest() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-model-materialization"),
            cache.path().to_path_buf(),
        );
        let files = sample_files();
        let bundle = staged(&store, &catalog, "job-m1", 0, &files).await;
        let prefix = bundle.artifact().url().clone();

        assert!(
            store
                .read_model_materialization(&prefix)
                .await
                .unwrap()
                .is_none(),
            "no sidecar exists before write_model_materialization runs"
        );

        let descriptor = fine_tune_descriptor();
        let env = fine_tune_env();
        // A fixed digest handed straight to `Materialization::new` — this
        // fixture resolves no version, so it is never an anchor a read could
        // disagree with (a resolved anchor comes only from a `PinnedSource`).
        let anchors: Vec<crate::store::manifest::InputAnchor> =
            vec![crate::store::manifest::InputAnchor::result_digest(
                "training-set",
                &ArtifactDigest("c".repeat(64)),
            )];
        let written = store
            .write_model_materialization(
                &bundle,
                Materialization::new(&descriptor, &env, anchors.clone()),
            )
            .await
            .unwrap();

        // The digest folded is the BUNDLE's combined hash, computed from the
        // already-written manifest.json -- not a placeholder.
        let manifest = store
            .read_manifest(&store.handle(&prefix).unwrap(), &prefix)
            .await
            .unwrap();
        assert_eq!(written.artifact.as_str(), manifest.combined_hash());
        // One leaf per bundle file, by NAME, carrying the manifest's own
        // sha256 — the inventory is additive to the subject.
        let mut names: Vec<(String, String)> = written
            .leaves
            .iter()
            .map(|l| match &l.key {
                LeafKey::File { name } => (name.clone(), l.digest.0.clone()),
                other => panic!("a bundle leaf is keyed by file name, got {other:?}"),
            })
            .collect();
        names.sort();
        let mut expected: Vec<(String, String)> = manifest
            .files
            .iter()
            .map(|e| (e.name.clone(), e.sha256.clone()))
            .collect();
        expected.sort();
        assert_eq!(names, expected);
        // Keyed by name, not position: a bundle with ONE MORE file carries
        // the same leaf for every file it shares with this one.
        let mut more = files.clone();
        more.push(("extra.bin".to_string(), Bytes::from_static(b"extra bytes")));
        let bundle_more = staged(&store, &catalog, "job-m1", 1, &more).await;
        let written_more = store
            .write_model_materialization(
                &bundle_more,
                Materialization::new(&descriptor, &env, anchors.clone()),
            )
            .await
            .unwrap();
        assert_eq!(written_more.leaves.len(), written.leaves.len() + 1);
        for leaf in &written.leaves {
            assert!(
                written_more.leaves.contains(leaf),
                "adding a file must change no existing leaf: {leaf:?}"
            );
        }
        assert_ne!(written_more.artifact, written.artifact);

        let read_back = store
            .read_model_materialization(&prefix)
            .await
            .unwrap()
            .expect("the sidecar this call just wrote must read back");
        assert_eq!(read_back, written);
        assert_eq!(read_back.descriptor, descriptor);
        assert_eq!(read_back.input_anchors, anchors);
    }

    /// The attestation cannot cover a bundle whose OWN `manifest.json` is not
    /// in hand: a torn write that never reached `manifest.json` fails with the
    /// same `NotPublished` reclassification `fetch_artifact` uses for a
    /// missing bundle manifest — never a silent attestation of a partial
    /// bundle.
    #[tokio::test]
    async fn write_model_materialization_fails_without_the_bundle_manifest() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-model-materialization-no-manifest"),
            cache.path().to_path_buf(),
        );
        let bundle = staged(&store, &catalog, "job-m2", 0, &sample_files()).await;
        let prefix = bundle.artifact().url();
        let manifest_path = store.child(prefix, MANIFEST_NAME).unwrap();
        store
            .handle(prefix)
            .unwrap()
            .delete_if_exists(&manifest_path)
            .await
            .unwrap();

        let descriptor = fine_tune_descriptor();
        let env = fine_tune_env();
        let anchors = vec![crate::store::manifest::InputAnchor::result_digest(
            "training-set",
            &ArtifactDigest("c".repeat(64)),
        )];
        let err = store
            .write_model_materialization(&bundle, Materialization::new(&descriptor, &env, anchors))
            .await
            .unwrap_err();
        assert!(
            matches!(err, JammiError::Storage(StorageError::NotPublished { .. })),
            "expected NotPublished without the bundle manifest, got: {err:?}"
        );
    }

    /// The materialization sidecar is the LAST object written into the
    /// prefix: every data file, then the bundle's own `manifest.json`, then
    /// `materialization.json`.
    /// Observed through the object store's own `last_modified` timestamps
    /// (the crate-private `list` seam `reconcile.rs` also uses) rather than
    /// asserted from code-reading alone: the materialization sidecar's
    /// timestamp is never EARLIER than any other object's in the prefix.
    #[tokio::test]
    async fn model_materialization_is_the_last_object_written_by_timestamp() {
        let (_catalog_dir, catalog) = test_catalog().await;
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-model-materialization-order"),
            cache.path().to_path_buf(),
        );
        let files = sample_files();
        let bundle = staged(&store, &catalog, "job-m3", 0, &files).await;
        let prefix = bundle.artifact().url().clone();
        let descriptor = fine_tune_descriptor();
        let env = fine_tune_env();
        let anchors = vec![crate::store::manifest::InputAnchor::result_digest(
            "training-set",
            &ArtifactDigest("c".repeat(64)),
        )];
        store
            .write_model_materialization(&bundle, Materialization::new(&descriptor, &env, anchors))
            .await
            .unwrap();

        let handle = store.handle(&prefix).unwrap();
        let listed = handle.list(&handle.data_path().unwrap()).await.unwrap();
        assert!(
            listed.len() >= files.len() + 2,
            "expected every data file plus manifest.json and materialization.json, got {listed:?}"
        );
        let materialization_ts = listed
            .iter()
            .find(|m| m.path.to_string().ends_with(MATERIALIZATION_NAME))
            .expect("materialization.json must be listed")
            .last_modified;
        for m in &listed {
            if m.path.to_string().ends_with(MATERIALIZATION_NAME) {
                continue;
            }
            assert!(
                m.last_modified <= materialization_ts,
                "materialization.json ({materialization_ts:?}) must never be EARLIER than \
                 '{}' ({:?})",
                m.path,
                m.last_modified
            );
        }
    }
}
