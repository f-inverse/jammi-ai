//! Worker-private artifact staging with atomic promotion.
//!
//! A durable training job may be run concurrently by more than one worker (a
//! lost-lease worker that has not yet noticed it lost the lease, plus the worker
//! that re-claimed the job). Both train the *same* `job_id`, and the canonical
//! on-disk artifact path is deterministic and shared across workers. If both
//! wrote that path directly, the slower worker would clobber the winner's
//! weights — or a concurrent reader would observe a torn file — even though the
//! terminal catalog row was lease-guarded.
//!
//! [`StagedArtifact`] closes that window by mirroring the catalog discipline at
//! the filesystem layer: every per-job artifact write goes to a **private** dir
//! unique to this worker+attempt, and the canonical path is written by exactly
//! one worker — the one whose finalize compare-and-set wins — via an atomic
//! same-filesystem rename. A worker that loses the lease discards its staging
//! and never touches the canonical artifact.
//!
//! ## Crash-recovery edge
//!
//! Promotion is per-file `rename` (each atomic), not a single directory swap, so
//! a worker that crashes *between* renaming the first and last file leaves the
//! canonical dir partially overwritten. This is a strictly narrower window than
//! the multi-worker clobber this type closes, and only on a hard crash mid-
//! promotion of the CAS *winner*; the normal concurrent-worker race is fully
//! closed because the loser never reaches `promote` at all. Recovery is the
//! job's re-claim path: a reclaimed job re-stages and re-promotes from scratch.

use std::path::{Path, PathBuf};

use jammi_db::error::{JammiError, Result};

/// A worker-private staging directory paired with its canonical destination.
///
/// Constructed before training begins: the trainer writes every per-job artifact
/// (final adapter, config sidecar, checkpoints) into [`Self::staging_dir`].
/// After training, the worker promotes the staged tree into the canonical path
/// **only if** its finalize CAS wins; otherwise it discards the staging.
///
/// `staging_dir` is `{canonical_parent}/.staging/{leaf}.{worker_id}` — co-located
/// on the same filesystem as the canonical dir so promotion is an atomic rename
/// rather than a cross-device copy.
#[derive(Debug)]
pub struct StagedArtifact {
    staging_dir: PathBuf,
    canonical_dir: PathBuf,
}

impl StagedArtifact {
    /// Stage artifacts destined for `canonical_dir`, privately to `worker_id`.
    ///
    /// The canonical leaf name (`job_id`, sanitized `model_id`, …) is reused as
    /// the staging leaf with the worker id appended, so two concurrent workers
    /// on the same job never share a training-time path. Creates the staging dir
    /// (and parents); does not create the canonical dir — that appears only on
    /// [`Self::promote`].
    pub fn stage(canonical_dir: PathBuf, worker_id: &str) -> Result<Self> {
        let parent = canonical_dir.parent().ok_or_else(|| {
            JammiError::FineTune(format!(
                "artifact path '{}' has no parent to stage beside",
                canonical_dir.display()
            ))
        })?;
        let leaf = canonical_dir.file_name().ok_or_else(|| {
            JammiError::FineTune(format!(
                "artifact path '{}' has no final component to stage",
                canonical_dir.display()
            ))
        })?;
        let mut staging_leaf = leaf.to_os_string();
        staging_leaf.push(".");
        staging_leaf.push(worker_id);
        let staging_dir = parent.join(".staging").join(staging_leaf);

        // A re-run of the same worker+attempt must start clean — a stale staging
        // dir from a prior crashed attempt would otherwise leak old files into
        // the promoted set.
        if staging_dir.exists() {
            std::fs::remove_dir_all(&staging_dir)
                .map_err(|e| JammiError::FineTune(format!("clear stale staging dir: {e}")))?;
        }
        std::fs::create_dir_all(&staging_dir)
            .map_err(|e| JammiError::FineTune(format!("create staging dir: {e}")))?;

        Ok(Self {
            staging_dir,
            canonical_dir,
        })
    }

    /// The private directory the trainer writes every artifact file into.
    pub fn staging_dir(&self) -> &Path {
        &self.staging_dir
    }

    /// The canonical directory the catalog model row points at — written only by
    /// the worker whose finalize CAS wins, via [`Self::promote`].
    pub fn canonical_dir(&self) -> &Path {
        &self.canonical_dir
    }

    /// Atomically promote the staged artifact into its canonical path.
    ///
    /// Called only after the worker's finalize compare-and-set returns `true`
    /// (this worker owns the terminal transition). Each staged file is moved into
    /// the canonical dir with `rename`, which on a single filesystem atomically
    /// replaces any existing file of that name — so a concurrent reader sees
    /// either the old or the new file, never a torn one. The staging dir is
    /// removed afterwards.
    ///
    /// Consumes `self` so a promoted artifact cannot also be discarded.
    pub fn promote(self) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.canonical_dir)
            .map_err(|e| JammiError::FineTune(format!("create canonical artifact dir: {e}")))?;
        for entry in std::fs::read_dir(&self.staging_dir)
            .map_err(|e| JammiError::FineTune(format!("read staging dir: {e}")))?
        {
            let entry =
                entry.map_err(|e| JammiError::FineTune(format!("staging dir entry: {e}")))?;
            let dest = self.canonical_dir.join(entry.file_name());
            std::fs::rename(entry.path(), &dest).map_err(|e| {
                JammiError::FineTune(format!(
                    "promote '{}' -> '{}': {e}",
                    entry.path().display(),
                    dest.display()
                ))
            })?;
        }
        let _ = std::fs::remove_dir_all(&self.staging_dir);
        Ok(self.canonical_dir)
    }

    /// Discard the staged artifact without touching the canonical path.
    ///
    /// Called when the worker's finalize CAS returns `false` (lease lost): a
    /// worker that did not win the terminal transition must leave no trace on the
    /// shared canonical artifact. Consumes `self`.
    pub fn discard(self) {
        // Best-effort: a leftover staging dir is harmless (it is re-created clean
        // on the next attempt) and never observed as a model artifact.
        let _ = std::fs::remove_dir_all(&self.staging_dir);
    }
}
