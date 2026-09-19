use thiserror::Error;

use super::url::Scheme;

/// Typed error returned by every operation in the [`storage`](crate::storage)
/// module. Variants name the failure mode so callers can pattern-match
/// (e.g. retry transient I/O, surface a credential mistake to the user).
#[derive(Debug, Clone, Error)]
pub enum StorageError {
    /// Input string was not a recognisable URL.
    #[error("invalid storage URL '{input}': {reason}")]
    InvalidUrl {
        /// Original input string the caller passed.
        input: String,
        /// Human-readable cause from the URL parser or scheme dispatcher.
        reason: String,
    },

    /// URL parsed but its scheme is not compiled into this build.
    ///
    /// `s3://`, `gs://`, `azure://` are gated by per-cloud Cargo features
    /// (`storage-s3`, `storage-gcs`, `storage-azure`). A user running the
    /// default feature set who passes an `s3://` URL gets this error rather
    /// than a confusing "unknown URL" message.
    #[error(
        "storage scheme {scheme:?} is not enabled in this build. \
         Rebuild with the matching cargo feature (storage-s3 / storage-gcs / \
         storage-azure) to enable it."
    )]
    SchemeNotEnabled {
        /// The scheme variant that the user requested.
        scheme: Scheme,
    },

    /// Cloud-driver construction failed: credentials missing or malformed,
    /// bucket name invalid for the provider, region unknown, etc.
    #[error("driver init failed for {scheme:?}: {reason}")]
    DriverInit {
        /// Scheme whose driver failed to construct.
        scheme: Scheme,
        /// Underlying message from the cloud SDK.
        reason: String,
    },

    /// No object exists at this path. Its own variant, never a shape of
    /// [`Self::Io`]: absence is an answer callers branch on (a probe, an
    /// idempotent delete, a first publish), where an I/O fault is not.
    #[error("no object at '{path}': {detail}")]
    NotFound {
        /// Path inside the bucket / volume that was being accessed.
        path: String,
        /// The driver's own description of the miss.
        detail: String,
    },

    /// Read / write against an already-constructed driver failed for any
    /// reason other than absence. Carries the upstream `object_store::Error`.
    #[error("object-store I/O error at '{path}': {source}")]
    Io {
        /// Path inside the bucket / volume that was being accessed.
        path: String,
        /// Underlying error from the `object_store` crate.
        #[source]
        source: std::sync::Arc<object_store::Error>,
    },

    /// Layout / format error: a file the engine wrote was unreadable, a
    /// manifest was malformed, a row-id was non-UTF8. Distinct from `Io`
    /// because the storage layer is healthy but the bytes it returned are
    /// wrong.
    #[error("layout error at '{path}': {reason}")]
    Layout {
        /// Object path that produced the malformed payload.
        path: String,
        /// Specific layout invariant that was violated.
        reason: String,
    },

    /// No `manifest.json` exists at this prefix at all — distinct from
    /// [`Self::Layout`], which means a manifest WAS read and it names bytes
    /// that turned out to be wrong (a corrupted bundle). This variant means
    /// there is no manifest in hand to judge: nothing was ever published at
    /// this prefix, or a catalog pointer names the wrong prefix entirely
    /// (never published, misdirected, or clobbered to point somewhere else,
    /// e.g. a base weights directory). Conflating the two would call an
    /// absent bundle "corrupt", which is a claim this variant carries no
    /// evidence for.
    #[error("no artifact bundle is published at '{path}' (manifest.json absent)")]
    NotPublished {
        /// Prefix URL a caller expected an artifact bundle under.
        path: String,
    },

    /// A live `models` row — in some tenant scope, not necessarily the
    /// caller's own — still names this prefix as its `artifact_path`,
    /// either exactly or as its immediate containing directory, so the
    /// guarded delete path
    /// ([`crate::store::ResultStore::delete_unreferenced_prefix`]) refused
    /// to remove it. Never surfaced by `ArtifactStore::delete_artifact_prefix`
    /// itself, which stays the unguarded primitive.
    ///
    /// Carries a COUNT only — never row ids, model names, or tenant ids —
    /// since the caller asking to delete `prefix` may itself be
    /// tenant-bound and must never learn identity beyond "still
    /// referenced".
    #[error("cannot delete '{prefix}': still referenced by {count} models row(s)")]
    Referenced {
        /// The prefix (or key) the caller asked to delete.
        prefix: String,
        /// How many `models` rows, across every tenant, still name `prefix`
        /// exactly — never which ones.
        count: usize,
    },
}

impl StorageError {
    /// Classify a bare `object_store::Error` raised at `path`: a miss is
    /// [`StorageError::NotFound`], anything else [`StorageError::Io`]. The one
    /// place a driver error becomes a `StorageError`.
    pub fn io(path: impl Into<String>, source: object_store::Error) -> Self {
        match source {
            object_store::Error::NotFound { .. } => Self::not_found(path, source.to_string()),
            source => Self::Io {
                path: path.into(),
                source: std::sync::Arc::new(source),
            },
        }
    }

    /// Construct a [`StorageError::NotFound`].
    pub fn not_found(path: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::NotFound {
            path: path.into(),
            detail: detail.into(),
        }
    }

    /// Construct a [`StorageError::Layout`] error.
    pub(crate) fn layout(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Layout {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Construct a [`StorageError::NotPublished`] error.
    pub(crate) fn not_published(path: impl Into<String>) -> Self {
        Self::NotPublished { path: path.into() }
    }
}
