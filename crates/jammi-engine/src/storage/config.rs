//! Per-cloud connection configuration.
//!
//! Each cloud backend has its own typed struct so missing credentials
//! surface as a Rust type error (or a [`StorageError::DriverInit`] at
//! construction time) rather than a stringly-typed `HashMap` lookup that
//! silently returns `None`.
//!
//! Each config type exposes a `validate()` method that catches partial
//! credential sets at config-load time — e.g. an `access_key_id` without
//! a `secret_access_key` — before the SDK surfaces the same error
//! deeper inside a request.
//!
//! [`StorageError::DriverInit`]: super::error::StorageError::DriverInit

use serde::{Deserialize, Serialize};

use super::error::StorageError;
use super::url::Scheme;

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

impl CloudConfig {
    /// Validate the variant's credential set at config-load time. Catches
    /// partial-credential mistakes (e.g. an `access_key_id` without its
    /// `secret_access_key`) before the SDK surfaces the same problem deep
    /// inside an I/O call.
    pub fn validate(&self) -> Result<(), StorageError> {
        match self {
            Self::S3(c) => c.validate(),
            Self::Gcs(c) => c.validate(),
            Self::Azure(c) => c.validate(),
        }
    }
}

impl S3Config {
    /// Reject partial explicit credentials. Leaving everything unset is
    /// allowed (and means "fall back to the SDK's default chain"); pairing
    /// `access_key_id` with a missing `secret_access_key` (or vice-versa)
    /// is rejected up front. A `session_token` without an `access_key_id`
    /// is rejected because STS-style credentials require all three.
    pub fn validate(&self) -> Result<(), StorageError> {
        match (
            self.access_key_id.as_deref(),
            self.secret_access_key.as_deref(),
        ) {
            (Some(_), None) => Err(StorageError::DriverInit {
                scheme: Scheme::S3,
                reason: "access_key_id is set but secret_access_key is missing".into(),
            }),
            (None, Some(_)) => Err(StorageError::DriverInit {
                scheme: Scheme::S3,
                reason: "secret_access_key is set but access_key_id is missing".into(),
            }),
            _ => {
                if self.session_token.is_some() && self.access_key_id.is_none() {
                    return Err(StorageError::DriverInit {
                        scheme: Scheme::S3,
                        reason: "session_token requires access_key_id and secret_access_key".into(),
                    });
                }
                Ok(())
            }
        }
    }
}

impl GcsConfig {
    /// `service_account_json` and `service_account_path` are mutually
    /// exclusive — supplying both is ambiguous about which credential the
    /// driver should honour. Either-or-neither (neither => ADC) is valid.
    pub fn validate(&self) -> Result<(), StorageError> {
        if self.service_account_json.is_some() && self.service_account_path.is_some() {
            return Err(StorageError::DriverInit {
                scheme: Scheme::Gcs,
                reason: "service_account_json and service_account_path are mutually exclusive"
                    .into(),
            });
        }
        Ok(())
    }
}

impl AzureConfig {
    /// `account_name` is required whenever any other Azure field is set —
    /// without it the SDK has no host to talk to. `account_key` and
    /// `sas_token` are mutually exclusive (the SDK uses different signing
    /// paths for each). OAuth fields (`tenant_id`/`client_id`/`client_secret`)
    /// must come as a complete triple or not at all.
    pub fn validate(&self) -> Result<(), StorageError> {
        let any_set = self.account_key.is_some()
            || self.sas_token.is_some()
            || self.tenant_id.is_some()
            || self.client_id.is_some()
            || self.client_secret.is_some();
        if any_set && self.account_name.is_none() {
            return Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                reason: "account_name is required when any other Azure credential field is set"
                    .into(),
            });
        }
        if self.account_key.is_some() && self.sas_token.is_some() {
            return Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                reason: "account_key and sas_token are mutually exclusive".into(),
            });
        }
        let oauth_set = [
            self.tenant_id.is_some(),
            self.client_id.is_some(),
            self.client_secret.is_some(),
        ];
        let oauth_count = oauth_set.iter().filter(|b| **b).count();
        if oauth_count != 0 && oauth_count != oauth_set.len() {
            return Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                reason: "tenant_id, client_id, and client_secret must all be set together".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_validate_rejects_half_credentials() {
        let bad = S3Config {
            access_key_id: Some("AKIA…".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::S3,
                ..
            })
        ));

        let also_bad = S3Config {
            secret_access_key: Some("xyz".into()),
            ..Default::default()
        };
        assert!(matches!(
            also_bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::S3,
                ..
            })
        ));
    }

    #[test]
    fn s3_validate_rejects_orphan_session_token() {
        let bad = S3Config {
            session_token: Some("FwoGZ…".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::S3,
                ..
            })
        ));
    }

    #[test]
    fn s3_validate_accepts_default_chain() {
        assert!(S3Config::default().validate().is_ok());
    }

    #[test]
    fn s3_validate_accepts_full_credentials() {
        let good = S3Config {
            access_key_id: Some("AKIA…".into()),
            secret_access_key: Some("xyz".into()),
            ..Default::default()
        };
        assert!(good.validate().is_ok());
    }

    #[test]
    fn gcs_validate_rejects_both_json_and_path() {
        let bad = GcsConfig {
            service_account_json: Some("{...}".into()),
            service_account_path: Some("/key.json".into()),
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::Gcs,
                ..
            })
        ));
    }

    #[test]
    fn gcs_validate_accepts_default_or_one_source() {
        assert!(GcsConfig::default().validate().is_ok());
        assert!(GcsConfig {
            service_account_json: Some("{...}".into()),
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn azure_validate_rejects_missing_account_name() {
        let bad = AzureConfig {
            account_key: Some("k".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                ..
            })
        ));
    }

    #[test]
    fn azure_validate_rejects_key_and_sas_together() {
        let bad = AzureConfig {
            account_name: Some("acct".into()),
            account_key: Some("k".into()),
            sas_token: Some("?sv=…".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                ..
            })
        ));
    }

    #[test]
    fn azure_validate_rejects_partial_oauth() {
        let bad = AzureConfig {
            account_name: Some("acct".into()),
            tenant_id: Some("t".into()),
            client_id: Some("c".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::Azure,
                ..
            })
        ));
    }

    #[test]
    fn azure_validate_accepts_complete_oauth_triple() {
        let good = AzureConfig {
            account_name: Some("acct".into()),
            tenant_id: Some("t".into()),
            client_id: Some("c".into()),
            client_secret: Some("s".into()),
            ..Default::default()
        };
        assert!(good.validate().is_ok());
    }
}
