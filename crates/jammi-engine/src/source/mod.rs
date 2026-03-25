pub mod local;
#[cfg(feature = "mysql")]
pub mod mysql;
#[cfg(feature = "postgres")]
pub mod postgres;
pub mod registry;
pub mod retry;
pub mod schema_provider;
// Note: SQLite external source via datafusion-table-providers is not supported
// because DTP's rusqlite version conflicts with our catalog's rusqlite (different
// libsqlite3-sys links versions). Users can query SQLite files via Local source
// after exporting to Parquet/CSV.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported data source backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// Local filesystem path.
    Local,
    /// PostgreSQL database.
    Postgres,
    /// MySQL / MariaDB database.
    Mysql,
}

/// File format for local and object-store sources.
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
    /// Connection URL or filesystem path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// File format (required for local and object-store sources).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FileFormat>,

    /// Override the default file extension used during directory listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<String>,

    /// Arbitrary key-value options passed to the source driver.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub options: HashMap<String, String>,
}

impl SourceConnection {
    /// Create a connection from a local filesystem path and file format.
    pub fn from_path(path: &str, format: FileFormat) -> Self {
        Self {
            url: Some(format!("file://{path}")),
            format: Some(format),
            ..Default::default()
        }
    }
}

/// Derive a DataFusion table name from a URL by extracting the file stem.
pub(crate) fn table_name_from_url(url: &str) -> String {
    let path = url.rsplit('/').next().unwrap_or(url);
    path.split('.').next().unwrap_or(path).to_string()
}
