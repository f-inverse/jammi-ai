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
//! ## Correctness model: catalog-pointer-as-commit
//!
//! Every write goes to a **unique per-attempt prefix** — a worker never writes a
//! shared canonical path, so two workers training the same job never collide and
//! no object is ever overwritten. The catalog row update that records the
//! prefix (the lease-guarded finalize CAS) is the single atomic commit; losers'
//! prefixes are simply orphaned and GC'd. There is no promote/rename step and
//! therefore no torn-promote window.
//!
//! ## Manifest discipline
//!
//! [`ArtifactStore::put_artifact`] writes the data files first, then a
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

use crate::error::{JammiError, Result};
use crate::storage::{
    sha256_hex, JammiObjectStore, Scheme, StorageError, StorageRegistry, StorageUrl,
};
use crate::store::layout::TenantSegment;
use crate::tenant::TenantId;

/// The file every artifact prefix carries last, naming the bundle's exact keys
/// and per-file digests. Written after every data file so its presence proves
/// the bundle is complete.
const MANIFEST_NAME: &str = "manifest.json";

/// The attempt-shared prefix segment for a job's durable resume checkpoint:
/// `{job_id}/_resume/`. Distinct from the per-attempt publish prefix
/// (`{job_id}/{worker_id}/{attempt}`) so resume state is keyed to the job, not an
/// attempt — and never collides with a published artifact prefix.
const RESUME_SEGMENT: &str = "_resume";

/// The nested segment under an attempt's own publish prefix that per-epoch
/// checkpoints live under: `{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{N}/`
/// (unit 348, CONTRACT item 1 / K7). `N` is the 0-based loop epoch index. This
/// is the ONE place the epoch-checkpoint key shape is spelled — both
/// [`ArtifactStore::put_epoch_checkpoint`] (the trainer's write) and
/// [`ArtifactStore::delete_epoch_checkpoint`] (any terminating path's GC sweep)
/// build the prefix through it, so a GC sweep can never drift out of sync with
/// where the writer actually publishes: reachability is a property of shared
/// code, not of two call sites independently agreeing on a string shape.
const CHECKPOINTS_SEGMENT: &str = "checkpoints";

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
/// is fixed by [`ArtifactStore::put_artifact`] (it sorts by name) so the
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

    /// Write an artifact bundle under a unique per-attempt prefix and return that
    /// prefix as the [`StorageUrl`] the catalog records.
    ///
    /// `tenant` is the owning row's tenant (the job's, or the model's) — it
    /// becomes the FIRST prefix segment (`models/{seg}/…`), via the same
    /// [`TenantSegment`] a result table's key is attributed through, so a
    /// listing pass (`reconcile`) can attribute every artifact key to a
    /// tenant exactly like a result-table key. `prefix_segments` are then
    /// joined under `{root}/{seg}` (the caller passes attempt-unique segments
    /// such as `[job_id, worker_id, attempt]`, so no two attempts ever target
    /// the same prefix and no object is overwritten). Each `(name, bytes)` is
    /// PUT under the prefix, then `manifest.json` is PUT **last** — its
    /// presence proves the bundle is complete. Returns the prefix
    /// [`StorageUrl`].
    pub async fn put_artifact(
        &self,
        tenant: Option<&TenantId>,
        prefix_segments: &[&str],
        files: &[(String, Bytes)],
    ) -> Result<StorageUrl> {
        let prefix = self.prefix_url(tenant, prefix_segments)?;
        let handle = self.handle(&prefix)?;

        // Sort entries by name so the manifest order — and thus the combined
        // content-hash a reader derives for the cache key — is deterministic for
        // a given content set regardless of caller order.
        let mut sorted: Vec<&(String, Bytes)> = files.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut entries = Vec::with_capacity(sorted.len());
        for (name, bytes) in &sorted {
            let path = self.child(&prefix, name)?;
            handle.put_bytes(&path, bytes.clone()).await?;
            entries.push(ManifestEntry {
                name: (*name).clone(),
                sha256: sha256_hex(bytes),
            });
        }

        // Manifest LAST: a reader that finds it can trust every key it names was
        // already written.
        let manifest = Manifest { files: entries };
        let manifest_bytes = Bytes::from(serde_json::to_vec(&manifest)?);
        let manifest_path = self.child(&prefix, MANIFEST_NAME)?;
        handle.put_bytes(&manifest_path, manifest_bytes).await?;

        Ok(prefix)
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
            Err(e) => Err(JammiError::Io(e)),
        }
    }

    /// Best-effort delete of every object under an artifact prefix.
    ///
    /// Used to GC a losing attempt's orphaned prefix. Reads the manifest to learn
    /// the keys and deletes each (plus the manifest); a 404 is not an error — the
    /// caller is paving over already-cleaned or never-completed state. A missing
    /// manifest means the attempt never completed its write; nothing durable to
    /// reclaim, so that is a no-op too.
    pub async fn delete_artifact_prefix(&self, prefix: &StorageUrl) -> Result<()> {
        let handle = self.handle(prefix)?;
        let manifest_path = self.child(prefix, MANIFEST_NAME)?;
        let manifest = if handle.exists(&manifest_path).await? {
            self.read_manifest(&handle, prefix).await.ok()
        } else {
            None
        };
        if let Some(manifest) = manifest {
            for entry in &manifest.files {
                let path = self.child(prefix, &entry.name)?;
                handle.delete_if_exists(&path).await?;
            }
        }
        handle.delete_if_exists(&manifest_path).await?;
        Ok(())
    }

    /// Write a job's durable **resume checkpoint** under the attempt-shared prefix
    /// `{job_id}/_resume/`, overwriting the prior epoch's bundle in place.
    ///
    /// Unlike [`Self::put_artifact`]'s per-attempt publish prefix, the resume
    /// prefix carries no `worker_id`/`attempt` segment: resume state belongs to
    /// the **job**, so attempt N+1 reads attempt N's progress. It is never
    /// registered as a model's served `artifact_path` and never read by the
    /// serving [`Self::fetch_artifact`] path — it is a crash-recovery side channel,
    /// exempt from the catalog-pointer-as-commit publish protocol. The write is
    /// manifest-last (torn-free) like every bundle, and **idempotent latest-wins**:
    /// because every epoch's bundle has the same file set (the same LoRA layer
    /// keys plus `resume_state.json`), each PUT overwrites the prior epoch's keys
    /// in place and the manifest — written last — flips the durable checkpoint to
    /// the new epoch atomically. Only the lease-holder writes (the trainer gates
    /// the call on `!cancel` at the epoch boundary), so a lost-lease zombie cannot
    /// regress the checkpoint to a stale epoch.
    pub async fn put_resume_checkpoint(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
        files: &[(String, Bytes)],
    ) -> Result<StorageUrl> {
        self.put_artifact(tenant, &[job_id, RESUME_SEGMENT], files)
            .await
    }

    /// Fetch a job's durable resume checkpoint, or `None` if no manifest exists
    /// under `{job_id}/_resume/` yet (the job has not completed an epoch boundary,
    /// so there is nothing to resume from — the worker starts from scratch).
    ///
    /// A present-but-corrupt bundle (manifest digest mismatch, missing key) is a
    /// hard error from [`Self::fetch_artifact`], not a silent `None`: a torn resume
    /// checkpoint must fail loudly rather than restart training from scratch and
    /// mask the corruption.
    pub async fn fetch_resume_checkpoint(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
    ) -> Result<Option<LocalArtifact>> {
        let prefix = self.prefix_url(tenant, &[job_id, RESUME_SEGMENT])?;
        let handle = self.handle(&prefix)?;
        let manifest_path = self.child(&prefix, MANIFEST_NAME)?;
        if !handle.exists(&manifest_path).await? {
            return Ok(None);
        }
        self.fetch_artifact(&prefix).await.map(Some)
    }

    /// GC a job's durable resume checkpoint once the job has terminated. Called by
    /// the finalize-CAS winner only: the resume state is dead the moment the job
    /// is `completed`, and the prefix is bounded to one bundle per job
    /// (overwrite-in-place), so this is the single point that reclaims it.
    pub async fn delete_resume_checkpoint(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
    ) -> Result<()> {
        let prefix = self.prefix_url(tenant, &[job_id, RESUME_SEGMENT])?;
        self.delete_artifact_prefix(&prefix).await
    }

    /// Publish one epoch's full loadable adapter checkpoint under the
    /// attempt-unique prefix `{job_id}/{worker_id}/{attempt}/checkpoints/
    /// epoch_{epoch}/` (unit 348, K7) — the same manifest-last, no-overwrite
    /// [`Self::put_artifact`] publish protocol every other bundle uses, with
    /// the `checkpoints/epoch_{N}` segment appended. `epoch` is the 0-based
    /// loop epoch index this checkpoint captures; a resumed attempt writes
    /// its OWN `attempt` segment, so no cross-attempt overwrite is possible by
    /// construction. `files` is the caller's full adapter bundle
    /// (`adapter.safetensors` + `adapter_config.json`), never the weights-only
    /// resume-bundle shape.
    pub async fn put_epoch_checkpoint(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
        worker_id: &str,
        attempt: &str,
        epoch: usize,
        files: &[(String, Bytes)],
    ) -> Result<StorageUrl> {
        let segment = epoch_segment(epoch);
        self.put_artifact(
            tenant,
            &[job_id, worker_id, attempt, CHECKPOINTS_SEGMENT, &segment],
            files,
        )
        .await
    }

    /// Best-effort GC of ONE epoch-checkpoint prefix
    /// (`{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{epoch}/`), tolerant
    /// of an epoch that was never actually written (no manifest — the same
    /// no-op [`Self::delete_artifact_prefix`] already treats a never-completed
    /// attempt as, not an error). This lets a caller derive and sweep a whole
    /// `[0, epochs)` range without first knowing how far training actually
    /// got: indices past the run's real progress are simply no-ops.
    pub async fn delete_epoch_checkpoint(
        &self,
        tenant: Option<&TenantId>,
        job_id: &str,
        worker_id: &str,
        attempt: &str,
        epoch: usize,
    ) -> Result<()> {
        let segment = epoch_segment(epoch);
        let prefix = self.prefix_url(
            tenant,
            &[job_id, worker_id, attempt, CHECKPOINTS_SEGMENT, &segment],
        )?;
        self.delete_artifact_prefix(&prefix).await
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
        let key = format!("{}/{}", prefix.path(), name);
        let stripped = match prefix.scheme() {
            Scheme::File | Scheme::Memory => key.trim_start_matches('/').to_string(),
            _ => key
                .split_once('/')
                .map(|(_, rest)| rest.to_string())
                .unwrap_or_default(),
        };
        ObjectPath::parse(&stripped)
            .map_err(|e| JammiError::Storage(StorageError::layout(&key, e.to_string())))
    }

    /// Join the tenant segment plus attempt-unique segments under the store
    /// root to form the artifact prefix URL:
    /// `{root}/{TenantSegment::of(tenant)}/{segments…}`. Each segment is
    /// sanitized so a `job_id`/`worker_id` carrying a `/` cannot escape the
    /// prefix or collide across attempts.
    fn prefix_url(&self, tenant: Option<&TenantId>, segments: &[&str]) -> Result<StorageUrl> {
        let root = self.root.as_str().trim_end_matches('/');
        let mut joined = String::from(root);
        joined.push('/');
        joined.push_str(&TenantSegment::of(tenant));
        for seg in segments {
            joined.push('/');
            joined.push_str(&sanitize_segment(seg));
        }
        StorageUrl::parse(&joined).map_err(JammiError::from)
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
    /// The sole caller is `reconcile`'s artifact arm: a prefix any
    /// `models.artifact_path` names is checked against this set rather than
    /// deleted as an orphan candidate outright — a manifest with a
    /// still-unpublished object is a torn write in progress, not garbage.
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
/// was never published, a catalog pointer names the wrong prefix, or a
/// pre-fix code path clobbered the pointer to point somewhere else entirely
/// (e.g. a base weights directory). This is a DIFFERENT failure class than
/// [`reclassify_missing_key`]'s "a manifest-listed key is gone" — that one
/// DOES have a manifest in hand naming exactly what's missing, which is
/// genuine bundle corruption. Conflating the two would call "nothing was
/// ever published here" a corrupted bundle, which is a claim this call site
/// carries no evidence for. Any other driver error (network fault,
/// throttling, credential rot, a 5xx) is a genuine transport/IO problem and
/// is left as [`StorageError::Io`] unchanged.
fn reclassify_missing_manifest(err: StorageError, prefix: &StorageUrl) -> JammiError {
    match &err {
        StorageError::Io {
            source: object_store::Error::NotFound { .. },
            ..
        } => JammiError::Storage(StorageError::not_published(prefix.as_str())),
        _ => JammiError::from(err),
    }
}

/// Re-type a `get_bytes` fault reading `name` under `prefix` as an INTEGRITY
/// failure of the bundle when the driver reports the key does not exist, vs.
/// leaving every other driver failure as the transport/IO fault it is.
///
/// A manifest names its keys after they were already written (`put_artifact`
/// writes the manifest last), so once a fetcher has a manifest in hand, a
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
        StorageError::Io {
            source: object_store::Error::NotFound { .. },
            ..
        } => JammiError::Storage(StorageError::layout(
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

/// The `checkpoints/` child segment naming one epoch's checkpoint: `epoch_{N}`.
/// The single spelling both [`ArtifactStore::put_epoch_checkpoint`] and
/// [`ArtifactStore::delete_epoch_checkpoint`] build their prefix from.
fn epoch_segment(epoch: usize) -> String {
    format!("epoch_{epoch}")
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

    fn store_with_root(root: StorageUrl, cache: PathBuf) -> ArtifactStore {
        ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap()
    }

    /// The require-gate polarity every `chmod` permission-fault probe in the
    /// workspace's test suites shares (esc-089 F1): `probe` performs the
    /// fault-injection premise check itself and returns `true` if the fault
    /// was BYPASSED (root, or a mode-ignoring filesystem). A bypass is
    /// normally a loud, `eprintln`'d skip; under `JAMMI_REQUIRE_POSIX_PERMS=1`
    /// (the CI lane that is SUPPOSED to run unprivileged with real POSIX
    /// permission enforcement) a bypass is instead a hard `panic!` — never a
    /// silent `return`. Each probe file carries its own copy of this wrapper
    /// in the canonical shape the kernel-oracle registry
    /// (`ci/kernel-oracle-helpers.txt`) verifies per file.
    fn chmod_bypassed(test_name: &str, probe: impl FnOnce() -> bool) -> bool {
        let bypassed = probe();
        if bypassed {
            if std::env::var_os("JAMMI_REQUIRE_POSIX_PERMS").is_some() {
                panic!(
                    "JAMMI_REQUIRE_POSIX_PERMS is set but '{test_name}' could not inject its \
                     permission fault (root, or a mode-ignoring filesystem) — the \
                     fault-injection premise this test needs does not hold; a silent skip is \
                     not acceptable here"
                );
            }
            eprintln!("{test_name}: chmod bypassed (root?) — skipping");
        }
        bypassed
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
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(StorageUrl::memory("artifacts"), cache.path().to_path_buf());
        let files = sample_files();

        let prefix = store
            .put_artifact(None, &["job-1", "worker-a", "0"], &files)
            .await
            .unwrap();
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
        let root_dir = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let root = StorageUrl::parse(root_dir.path().to_str().unwrap()).unwrap();
        let store = store_with_root(root, cache.path().to_path_buf());
        let files = sample_files();

        let prefix = store
            .put_artifact(None, &["job-2", "worker-b", "1"], &files)
            .await
            .unwrap();
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
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-corrupt"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-3", "worker-c", "0"], &sample_files())
            .await
            .unwrap();

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
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-missing-key"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-4", "worker-d", "0"], &sample_files())
            .await
            .unwrap();
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
    #[cfg(unix)]
    #[tokio::test]
    async fn permission_fault_on_a_present_key_stays_a_transport_error() {
        use std::os::unix::fs::PermissionsExt;

        let root_dir = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let root = StorageUrl::parse(root_dir.path().to_str().unwrap()).unwrap();
        let store = store_with_root(root, cache.path().to_path_buf());
        let prefix = store
            .put_artifact(None, &["job-5", "worker-e", "0"], &sample_files())
            .await
            .unwrap();
        let weights_path = std::path::PathBuf::from(prefix.path()).join("adapter.safetensors");

        // PROBE: root (and a mode-ignoring filesystem) bypasses chmod — in
        // which case the fault-injection premise this test needs never holds.
        // Shared require-gate polarity (esc-089 F1): under
        // `JAMMI_REQUIRE_POSIX_PERMS=1` a bypass panics rather than
        // skipping — it must never be a silent `return`.
        std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let bypassed = chmod_bypassed(
            "permission_fault_on_a_present_key_stays_a_transport_error",
            || std::fs::read(&weights_path).is_ok(),
        );
        if bypassed {
            let _ = std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o644));
            return;
        }

        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        let _ = std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o644));
        assert!(
            matches!(err, JammiError::Storage(StorageError::Io { .. })),
            "a permission-denied open is a genuine driver/transport fault, not this bundle's \
             own integrity — it must stay StorageError::Io, never be reclassified to Layout, \
             got: {err:?}"
        );
    }

    #[tokio::test]
    async fn cache_hit_avoids_redownload() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-cache"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-4", "worker-d", "0"], &sample_files())
            .await
            .unwrap();

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
        let cache = tempfile::tempdir().unwrap();
        let store = Arc::new(store_with_root(
            StorageUrl::memory("artifacts-concurrent"),
            cache.path().to_path_buf(),
        ));
        let files = sample_files();
        let prefix = store
            .put_artifact(None, &["job-5", "worker-e", "0"], &files)
            .await
            .unwrap();

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

    #[tokio::test]
    async fn delete_prefix_removes_objects_and_is_idempotent() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-delete"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-6", "worker-f", "0"], &sample_files())
            .await
            .unwrap();

        store.delete_artifact_prefix(&prefix).await.unwrap();
        // The manifest is gone, so a fetch now fails.
        assert!(store.fetch_artifact(&prefix).await.is_err());
        // Deleting again (already-clean) is a no-op, not an error.
        store.delete_artifact_prefix(&prefix).await.unwrap();
    }

    #[tokio::test]
    async fn resume_checkpoint_round_trips_and_overwrites_latest_wins() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume"),
            cache.path().to_path_buf(),
        );

        // No checkpoint written yet → None (start from scratch).
        assert!(store
            .fetch_resume_checkpoint(None, "job-r")
            .await
            .unwrap()
            .is_none());

        // Persist epoch 0, then epoch 1 with the SAME file set — the second PUT
        // overwrites in place and the manifest (written last) flips the durable
        // checkpoint to epoch 1.
        let epoch0 = vec![(
            "resume_state.json".to_string(),
            Bytes::from_static(b"{\"epoch\":0}"),
        )];
        store
            .put_resume_checkpoint(None, "job-r", &epoch0)
            .await
            .unwrap();
        let epoch1 = vec![(
            "resume_state.json".to_string(),
            Bytes::from_static(b"{\"epoch\":1}"),
        )];
        store
            .put_resume_checkpoint(None, "job-r", &epoch1)
            .await
            .unwrap();

        let fetched = store
            .fetch_resume_checkpoint(None, "job-r")
            .await
            .unwrap()
            .expect("a written checkpoint is fetchable");
        let bytes = std::fs::read(fetched.dir().join("resume_state.json")).unwrap();
        assert_eq!(
            &bytes[..],
            b"{\"epoch\":1}",
            "latest epoch wins on overwrite"
        );

        // GC by the finalize winner: the next fetch is None again.
        store.delete_resume_checkpoint(None, "job-r").await.unwrap();
        assert!(store
            .fetch_resume_checkpoint(None, "job-r")
            .await
            .unwrap()
            .is_none());
        // Deleting again (already-clean) is a no-op.
        store.delete_resume_checkpoint(None, "job-r").await.unwrap();
    }

    #[tokio::test]
    async fn resume_prefix_is_disjoint_from_the_publish_prefix() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-resume-disjoint"),
            cache.path().to_path_buf(),
        );
        // Publish an attempt's served artifact and a resume checkpoint for the
        // same job; neither read sees the other's bytes — the resume side channel
        // never perturbs the served path.
        let published = store
            .put_artifact(None, &["job-d", "worker-a", "0"], &sample_files())
            .await
            .unwrap();
        store
            .put_resume_checkpoint(
                None,
                "job-d",
                &[(
                    "resume_state.json".to_string(),
                    Bytes::from_static(b"{\"epoch\":3}"),
                )],
            )
            .await
            .unwrap();

        // The served prefix does not end at the resume segment, and the published
        // bundle is intact.
        assert!(published.as_str().ends_with("job-d/worker-a/0"));
        let served = store.fetch_artifact(&published).await.unwrap();
        assert!(served.dir().join("adapter.safetensors").exists());
        assert!(!served.dir().join("resume_state.json").exists());

        // GCing the resume checkpoint leaves the served artifact untouched.
        store.delete_resume_checkpoint(None, "job-d").await.unwrap();
        let served_again = store.fetch_artifact(&published).await.unwrap();
        assert!(served_again.dir().join("adapter.safetensors").exists());
    }

    #[tokio::test]
    async fn put_artifact_lands_under_the_global_segment_when_untenanted() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-seg"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-g", "worker-a", "0"], &sample_files())
            .await
            .unwrap();
        assert!(
            prefix
                .as_str()
                .ends_with("artifacts-seg/_global/job-g/worker-a/0"),
            "an untenanted artifact must land under the `_global` segment, got: {prefix}"
        );
    }

    #[tokio::test]
    async fn put_artifact_lands_under_the_tenant_segment() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-tenant"),
            cache.path().to_path_buf(),
        );
        let tenant = TenantId::from_uuid(uuid::Builder::from_bytes([42; 16]).into_uuid()).unwrap();
        let prefix = store
            .put_artifact(Some(&tenant), &["job-t", "worker-a", "0"], &sample_files())
            .await
            .unwrap();
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
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-expected-some"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(None, &["job-e", "worker-a", "0"], &sample_files())
            .await
            .unwrap();
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
}
