use thiserror::Error;

use super::url::Scheme;

/// Typed error returned by every operation in the [`storage`](crate::storage)
/// module. Variants name the failure mode so callers can pattern-match
/// (e.g. retry transient I/O, surface a credential mistake to the user).
#[derive(Debug, Error)]
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

    /// Read / write against an already-constructed driver failed. Carries the
    /// upstream `object_store::Error` so retries / 404 handling can pattern-match.
    #[error("object-store I/O error at '{path}': {source}")]
    Io {
        /// Path inside the bucket / volume that was being accessed.
        path: String,
        /// Underlying error from the `object_store` crate.
        #[source]
        source: object_store::Error,
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
}

impl StorageError {
    /// Construct an [`StorageError::Io`] from a bare `object_store::Error`.
    pub(crate) fn io(path: impl Into<String>, source: object_store::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
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
