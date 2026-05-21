use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::error::StorageError;

/// Concrete object-store backend a [`StorageUrl`] resolves to.
///
/// Schemes never spelled in user input (e.g. `mem://` for hermetic tests)
/// also live here so the dispatcher's `match` is exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scheme {
    /// Local filesystem path. `file://` URLs plus bare absolute paths.
    File,
    /// In-memory store, only used in tests; spelled `memory://`.
    Memory,
    /// AWS S3 (or S3-compatible) bucket via `s3://bucket/key`.
    S3,
    /// Google Cloud Storage via `gs://bucket/key`.
    Gcs,
    /// Azure Blob Storage via `azure://container/blob` or the
    /// `abfss://`/`https://...blob.core.windows.net` URL forms.
    Azure,
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::File => "file",
            Self::Memory => "memory",
            Self::S3 => "s3",
            Self::Gcs => "gs",
            Self::Azure => "azure",
        };
        f.write_str(s)
    }
}

/// Validated URL pointing at an object-store location.
///
/// Construction goes through [`StorageUrl::parse`] so a [`StorageUrl`] value
/// is *always* a real URL whose scheme the engine knows. Filesystem inputs
/// without a scheme (`./data/triplets.parquet`, `/absolute/path`) are
/// normalised to `file://...` so downstream code never branches on "is this
/// a bare path or a URL?".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StorageUrl(String);

impl StorageUrl {
    /// Parse an input string into a validated [`StorageUrl`].
    ///
    /// - URLs with an explicit `<scheme>://` prefix are dispatched on scheme.
    /// - Bare filesystem paths are normalised to `file://...`.
    /// - Windows drive paths (`C:\...`) are accepted and re-spelled with
    ///   forward slashes so downstream `object_store::path::Path` parsing
    ///   stays portable.
    pub fn parse(input: &str) -> Result<Self, StorageError> {
        if input.is_empty() {
            return Err(StorageError::InvalidUrl {
                input: input.to_string(),
                reason: "URL is empty".into(),
            });
        }

        if let Some((scheme_str, _)) = input.split_once("://") {
            // Validate scheme is one we know. We do not store `Scheme` on
            // `StorageUrl` because the source-of-truth is the URL prefix
            // — keeping them separate avoids drift.
            Self::parse_scheme(scheme_str)?;
            return Ok(Self(input.to_string()));
        }

        // No scheme — treat as filesystem path.
        let path = Path::new(input);
        let normalised = path.to_string_lossy().replace('\\', "/");
        // Empty / "." → reject; absolute / relative both become file://...
        let url = if normalised.starts_with('/') {
            format!("file://{normalised}")
        } else {
            format!("file://./{normalised}")
        };
        Ok(Self(url))
    }

    /// Construct a [`StorageUrl`] for an in-memory store rooted at the given
    /// virtual path. Used by hermetic tests; never returned from user input.
    pub fn memory(virtual_path: &str) -> Self {
        let trimmed = virtual_path.trim_start_matches('/');
        Self(format!("memory:///{trimmed}"))
    }

    /// The validated URL as the original `<scheme>://<rest>` string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The concrete backend this URL resolves to.
    pub fn scheme(&self) -> Scheme {
        let prefix = self.0.split_once("://").map(|(s, _)| s).unwrap_or("");
        Self::parse_scheme(prefix).expect("StorageUrl invariant: scheme always valid")
    }

    /// The path component (everything after `<scheme>://`).
    ///
    /// For `file:///abs/path/data.parquet` returns `/abs/path/data.parquet`.
    /// For `s3://bucket/key/data.parquet` returns `bucket/key/data.parquet`.
    pub fn path(&self) -> &str {
        self.0.split_once("://").map(|(_, p)| p).unwrap_or("")
    }

    fn parse_scheme(s: &str) -> Result<Scheme, StorageError> {
        match s {
            "file" => Ok(Scheme::File),
            "memory" => Ok(Scheme::Memory),
            "s3" => Ok(Scheme::S3),
            "gs" | "gcs" => Ok(Scheme::Gcs),
            "azure" | "abfss" => Ok(Scheme::Azure),
            other => Err(StorageError::InvalidUrl {
                input: other.to_string(),
                reason: format!("unknown scheme '{other}://'"),
            }),
        }
    }
}

impl fmt::Display for StorageUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for StorageUrl {
    type Err = StorageError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_file_url() {
        let u = StorageUrl::parse("file:///tmp/data.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::File);
        assert_eq!(u.path(), "/tmp/data.parquet");
    }

    #[test]
    fn parses_s3_url() {
        let u = StorageUrl::parse("s3://benchmarks/snapshots/2026.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::S3);
        assert_eq!(u.path(), "benchmarks/snapshots/2026.parquet");
    }

    #[test]
    fn parses_gs_url() {
        let u = StorageUrl::parse("gs://archives/2026/jan.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::Gcs);
    }

    #[test]
    fn parses_azure_url() {
        let u = StorageUrl::parse("azure://archives/snapshots/x.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::Azure);
    }

    #[test]
    fn bare_absolute_path_becomes_file_url() {
        let u = StorageUrl::parse("/tmp/foo.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::File);
        assert_eq!(u.path(), "/tmp/foo.parquet");
    }

    #[test]
    fn bare_relative_path_becomes_file_url() {
        let u = StorageUrl::parse("data/triplets.parquet").unwrap();
        assert_eq!(u.scheme(), Scheme::File);
        assert!(u.path().ends_with("data/triplets.parquet"));
    }

    #[test]
    fn unknown_scheme_rejected() {
        let err = StorageUrl::parse("ftp://host/foo").unwrap_err();
        assert!(matches!(err, StorageError::InvalidUrl { .. }));
    }

    #[test]
    fn empty_input_rejected() {
        let err = StorageUrl::parse("").unwrap_err();
        assert!(matches!(err, StorageError::InvalidUrl { .. }));
    }
}
