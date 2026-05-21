//! Per-cloud connection configuration.
//!
//! Each cloud backend has its own typed struct so missing credentials
//! surface as a Rust type error (or a [`StorageError::DriverInit`] at
//! construction time) rather than a stringly-typed `HashMap` lookup that
//! silently returns `None`.
//!
//! [`StorageError::DriverInit`]: super::error::StorageError::DriverInit

use serde::{Deserialize, Serialize};

/// AWS S3 (or any S3-compatible) connection details.
///
/// Used to build the S3 driver via [`crate::storage::builder::build_object_store`].
/// Field names mirror the canonical AWS SDK env var conventions so a
/// caller can populate from `std::env::var` 1:1.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct S3Config {
    /// AWS region (e.g. `"us-east-1"`).
    pub region: Option<String>,
    /// Custom endpoint URL — for MinIO / LocalStack / S3-compatible services.
    pub endpoint: Option<String>,
    /// Access key ID. When unset, the SDK's default credential chain is used
    /// (env vars, instance profile, EKS IRSA token, etc).
    pub access_key_id: Option<String>,
    /// Secret access key paired with [`Self::access_key_id`].
    pub secret_access_key: Option<String>,
    /// Optional session token for temporary credentials (STS, SSO).
    pub session_token: Option<String>,
    /// Whether to allow plain HTTP (only used against test endpoints).
    /// Defaults to `false` so production deployments fail closed.
    #[serde(default)]
    pub allow_http: bool,
}

/// Google Cloud Storage connection details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcsConfig {
    /// Inline service-account JSON. When unset, the GCS driver falls back
    /// to Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`,
    /// Workload Identity, gcloud user creds).
    pub service_account_json: Option<String>,
    /// Path to a service-account JSON file. Alternative to
    /// [`Self::service_account_json`] for callers who want to keep the file
    /// out of the catalog row.
    pub service_account_path: Option<String>,
}

/// Azure Blob Storage connection details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Storage account name (e.g. `"myaccount"` for
    /// `myaccount.blob.core.windows.net`).
    pub account_name: Option<String>,
    /// Account access key. Mutually exclusive with [`Self::sas_token`].
    pub account_key: Option<String>,
    /// Shared-access-signature token.
    pub sas_token: Option<String>,
    /// Tenant id for OAuth / Managed Identity auth.
    pub tenant_id: Option<String>,
    /// Client id for OAuth.
    pub client_id: Option<String>,
    /// Client secret paired with [`Self::client_id`].
    pub client_secret: Option<String>,
}

/// Tagged union of per-cloud configuration. The variant selects which
/// driver the [`crate::storage::builder`] will construct.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CloudConfig {
    /// AWS S3 (or S3-compatible) — see [`S3Config`].
    S3(S3Config),
    /// Google Cloud Storage — see [`GcsConfig`].
    Gcs(GcsConfig),
    /// Azure Blob — see [`AzureConfig`].
    Azure(AzureConfig),
}
