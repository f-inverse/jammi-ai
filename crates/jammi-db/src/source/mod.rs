pub mod file_format;
pub mod mutable;
#[cfg(feature = "mysql")]
pub mod mysql;
#[cfg(feature = "postgres")]
pub mod postgres;
pub mod registry;
pub mod retry;
pub mod schema_provider;
// Note: SQLite external source via datafusion-table-providers is not supported
// because DTP's rusqlite version conflicts with our catalog's rusqlite (different
// libsqlite3-sys links versions). Users can query SQLite files via the File
// source after exporting to Parquet/CSV.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::storage::{CloudConfig, StorageUrl};

/// Supported data source backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// File-shaped source (Parquet / CSV / JSON) read through any
    /// `StorageUrl`-addressable backend — local disk, S3, GCS, Azure.
    File,
    /// PostgreSQL database.
    Postgres,
    /// MySQL / MariaDB database.
    Mysql,
}

/// File format for [`SourceType::File`] sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    /// Apache Parquet columnar format.
    Parquet,
    /// Comma-separated values.
    Csv,
    /// Newline-delimited JSON.
    Json,
    /// Apache Avro binary format.
    Avro,
}

impl std::fmt::Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parquet => write!(f, "parquet"),
            Self::Csv => write!(f, "csv"),
            Self::Json => write!(f, "json"),
            Self::Avro => write!(f, "avro"),
        }
    }
}

impl std::str::FromStr for FileFormat {
    type Err = crate::error::JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "parquet" => Ok(Self::Parquet),
            "csv" => Ok(Self::Csv),
            "json" => Ok(Self::Json),
            "avro" => Ok(Self::Avro),
            other => Err(crate::error::JammiError::Other(format!(
                "Unknown file format '{other}'. Expected: parquet, csv, json, avro"
            ))),
        }
    }
}

/// Connection parameters for a data source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceConnection {
    /// Storage URL (file://, s3://, gs://, azure://) or external-source
    /// connection string (postgres://, mysql://).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// File format (required for [`SourceType::File`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FileFormat>,

    /// Override the default file extension used during directory listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<String>,

    /// Per-cloud credentials. `None` for local / external sources and for
    /// cloud sources that rely on ambient credentials (instance profile,
    /// ADC, Managed Identity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloud: Option<CloudConfig>,

    /// Arbitrary key-value options passed to the source driver.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub options: HashMap<String, String>,

    /// Name of the tenant-discriminator column on this federated source, when
    /// its rows carry tenancy under a column other than `tenant_id`. Consulted
    /// by the tenant-scope analyzer to inject the discriminator predicate on a
    /// scan of this source. Persisted within the source's serialized options so
    /// it round-trips across session restarts (`reload_sources` replays it into
    /// the in-process [`crate::tenant_scope::SourceTenantColumns`] lookup); a
    /// source with no tenant discriminator leaves this `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_column: Option<String>,
}

impl SourceConnection {
    /// Build a connection from a [`StorageUrl`] + file format.
    ///
    /// Used by file-shaped sources regardless of scheme — local paths
    /// (parsed into `file://...`), S3, GCS, Azure. The URL has already
    /// been validated so this never fails.
    pub fn from_url(url: StorageUrl, format: FileFormat) -> Self {
        Self {
            url: Some(url.to_string()),
            format: Some(format),
            ..Default::default()
        }
    }

    /// Parse `path_or_url` into a [`StorageUrl`] and build a connection.
    ///
    /// Convenience wrapper used by CLI / Python callers that accept a free-
    /// form string. Returns the underlying [`crate::storage::StorageError`]
    /// when the input is unrecognisable.
    pub fn parse(
        path_or_url: &str,
        format: FileFormat,
    ) -> Result<Self, crate::storage::StorageError> {
        let url = StorageUrl::parse(path_or_url)?;
        Ok(Self::from_url(url, format))
    }
}

/// Derive a DataFusion table name from a URL by extracting the file stem.
///
/// Handles both Unix forward-slash paths and Windows backslash paths so that
/// `file://C:\Users\...\triplets_train.parquet` correctly yields `triplets_train`
/// rather than the full `C:\...\triplets_train` (which contains a colon that
/// breaks DataFusion's SQL parser).
pub(crate) fn table_name_from_url(url: &str) -> String {
    let path = url.rsplit(['/', '\\']).next().unwrap_or(url);
    path.split('.').next().unwrap_or(path).to_string()
}
