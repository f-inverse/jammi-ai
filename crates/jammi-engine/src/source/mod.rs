pub mod local;
pub mod registry;
pub mod retry;
pub mod schema_provider;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported data source backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// PostgreSQL database.
    Postgres,
    /// MySQL / MariaDB database.
    Mysql,
    /// SQLite database file.
    Sqlite,
    /// Snowflake data warehouse.
    Snowflake,
    /// Google BigQuery.
    BigQuery,
    /// MongoDB document store.
    MongoDB,
    /// Amazon S3 object store.
    S3,
    /// Google Cloud Storage.
    Gcs,
    /// Azure Blob Storage.
    Azure,
    /// Local filesystem path.
    Local,
    /// Qdrant vector database.
    Qdrant,
    /// User-defined source with custom connection logic.
    Custom,
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
