pub mod local;
pub mod registry;
pub mod schema_provider;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported source types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Postgres,
    Mysql,
    Sqlite,
    Snowflake,
    BigQuery,
    MongoDB,
    S3,
    Gcs,
    Azure,
    Local,
    Qdrant,
    Custom,
}

/// File format for local / object store sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    Parquet,
    Csv,
    Json,
    Avro,
}

/// Connection configuration for a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceConnection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FileFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<String>,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub options: HashMap<String, String>,
}

/// Derive a table name from a URL path (file stem).
pub fn table_name_from_url(url: &str) -> String {
    let path = url.rsplit('/').next().unwrap_or(url);
    path.split('.').next().unwrap_or(path).to_string()
}
