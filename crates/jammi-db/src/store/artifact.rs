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
//! path. A manifest absent or a file whose bytes do not hash to the recorded
//! digest is a hard error (a partial PUT, not a torn load to be papered over).
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
use crate::storage::{JammiObjectStore, Scheme, StorageError, StorageRegistry, StorageUrl};

/// The file every artifact prefix carries last, naming the bundle's exact keys
/// and per-file digests. Written after every data file so its presence proves
/// the bundle is complete.
const MANIFEST_NAME: &str = "manifest.json";

/// The attempt-shared prefix segment for a job's durable resume checkpoint:
/// `{job_id}/_resume/`. Distinct from the per-attempt publish prefix
/// (`{job_id}/{worker_id}/{attempt}`) so resume state is keyed to the job, not an
/// attempt — and never collides with a published artifact prefix.
const RESUME_SEGMENT: &str = "_resume";

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

/// Lowercase-hex sha256 of a byte slice.
fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
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
    /// `prefix_segments` are joined under the store root to form the prefix
    /// (the caller passes attempt-unique segments such as
    /// `[job_id, worker_id, attempt]`, so no two attempts ever target the same
    /// prefix and no object is overwritten). Each `(name, bytes)` is PUT under
    /// the prefix, then `manifest.json` is PUT **last** — its presence proves
    /// the bundle is complete. Returns the prefix `StorageUrl`.
    pub async fn put_artifact(
        &self,
        prefix_segments: &[&str],
        files: &[(String, Bytes)],
    ) -> Result<StorageUrl> {
        let prefix = self.prefix_url(prefix_segments)?;
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
            let bytes = handle.get_bytes(&path).await?;
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
        job_id: &str,
        files: &[(String, Bytes)],
    ) -> Result<StorageUrl> {
        self.put_artifact(&[job_id, RESUME_SEGMENT], files).await
    }

    /// Fetch a job's durable resume checkpoint, or `None` if no manifest exists
    /// under `{job_id}/_resume/` yet (the job has not completed an epoch boundary,
    /// so there is nothing to resume from — the worker starts from scratch).
    ///
    /// A present-but-corrupt bundle (manifest digest mismatch, missing key) is a
    /// hard error from [`Self::fetch_artifact`], not a silent `None`: a torn resume
    /// checkpoint must fail loudly rather than restart training from scratch and
    /// mask the corruption.
    pub async fn fetch_resume_checkpoint(&self, job_id: &str) -> Result<Option<LocalArtifact>> {
        let prefix = self.prefix_url(&[job_id, RESUME_SEGMENT])?;
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
    pub async fn delete_resume_checkpoint(&self, job_id: &str) -> Result<()> {
        let prefix = self.prefix_url(&[job_id, RESUME_SEGMENT])?;
        self.delete_artifact_prefix(&prefix).await
    }

    /// Read and parse `manifest.json` under `prefix`. A missing or malformed
    /// manifest is a hard error — without it the bundle's completeness cannot be
    /// established.
    async fn read_manifest(
        &self,
        handle: &JammiObjectStore,
        prefix: &StorageUrl,
    ) -> Result<Manifest> {
        let manifest_path = self.child(prefix, MANIFEST_NAME)?;
        let bytes = handle.get_bytes(&manifest_path).await?;
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
            let bytes = handle.get_bytes(&path).await?;
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

    /// Join attempt-unique segments under the store root to form the artifact
    /// prefix URL. Each segment is sanitized so a `job_id`/`worker_id` carrying a
    /// `/` cannot escape the prefix or collide across attempts.
    fn prefix_url(&self, segments: &[&str]) -> Result<StorageUrl> {
        let root = self.root.as_str().trim_end_matches('/');
        let mut joined = String::from(root);
        for seg in segments {
            joined.push('/');
            joined.push_str(&sanitize_segment(seg));
        }
        StorageUrl::parse(&joined).map_err(JammiError::from)
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
            .put_artifact(&["job-1", "worker-a", "0"], &files)
            .await
            .unwrap();
        assert!(prefix.as_str().ends_with("artifacts/job-1/worker-a/0"));

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
            .put_artifact(&["job-2", "worker-b", "1"], &files)
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
            .put_artifact(&["job-3", "worker-c", "0"], &sample_files())
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
    async fn missing_manifest_is_a_hard_error() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-nomanifest"),
            cache.path().to_path_buf(),
        );
        // A prefix that was never written — no manifest exists.
        let prefix = store.prefix_url(&["ghost", "worker", "0"]).unwrap();
        let err = store.fetch_artifact(&prefix).await.unwrap_err();
        // The manifest GET 404s before any data file is touched.
        assert!(!err.to_string().is_empty());
    }

    #[tokio::test]
    async fn cache_hit_avoids_redownload() {
        let cache = tempfile::tempdir().unwrap();
        let store = store_with_root(
            StorageUrl::memory("artifacts-cache"),
            cache.path().to_path_buf(),
        );
        let prefix = store
            .put_artifact(&["job-4", "worker-d", "0"], &sample_files())
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
            .put_artifact(&["job-5", "worker-e", "0"], &files)
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
            .put_artifact(&["job-6", "worker-f", "0"], &sample_files())
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
            .fetch_resume_checkpoint("job-r")
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
        store.put_resume_checkpoint("job-r", &epoch0).await.unwrap();
        let epoch1 = vec![(
            "resume_state.json".to_string(),
            Bytes::from_static(b"{\"epoch\":1}"),
        )];
        store.put_resume_checkpoint("job-r", &epoch1).await.unwrap();

        let fetched = store
            .fetch_resume_checkpoint("job-r")
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
        store.delete_resume_checkpoint("job-r").await.unwrap();
        assert!(store
            .fetch_resume_checkpoint("job-r")
            .await
            .unwrap()
            .is_none());
        // Deleting again (already-clean) is a no-op.
        store.delete_resume_checkpoint("job-r").await.unwrap();
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
            .put_artifact(&["job-d", "worker-a", "0"], &sample_files())
            .await
            .unwrap();
        store
            .put_resume_checkpoint(
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
        store.delete_resume_checkpoint("job-d").await.unwrap();
        let served_again = store.fetch_artifact(&published).await.unwrap();
        assert!(served_again.dir().join("adapter.safetensors").exists());
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
