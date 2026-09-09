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
use crate::config::secret::{deserialize_inline, serialize_exposed};
use crate::config::Secret;

/// AWS S3 (or any S3-compatible) connection details.
///
/// Used to build the S3 driver via [`crate::storage::builder::build_object_store`].
/// Field names mirror the canonical AWS SDK env var conventions so a
/// caller can populate from `std::env::var` 1:1.
///
/// `secret_access_key`/`session_token` are [`Secret`]-typed: `Debug`
/// (derived) redacts them as `Secret(***)`, and each serializes as
/// plaintext ONLY through [`serialize_exposed`] and deserializes a plain
/// string ONLY through [`deserialize_inline`] (an object form is refused,
/// never read as a file) — the shape this struct persists as in a
/// `sources.options` row is unchanged in both directions (still a plain
/// JSON string at each key), but a `{:?}` of a config carrying an
/// `S3Config` never prints either.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct S3Config {
    /// AWS region (e.g. `"us-east-1"`).
    pub region: Option<String>,
    /// Custom endpoint URL — for MinIO / LocalStack / S3-compatible services.
    pub endpoint: Option<String>,
    /// Access key ID. When unset, the SDK's default credential chain is used
    /// (env vars, instance profile, EKS IRSA token, etc).
    pub access_key_id: Option<String>,
    /// Secret access key paired with [`Self::access_key_id`]. See the
    /// struct docs for the redaction/persistence split.
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub secret_access_key: Option<Secret>,
    /// Optional session token for temporary credentials (STS, SSO). See the
    /// struct docs for the redaction/persistence split.
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub session_token: Option<Secret>,
    /// Whether to allow plain HTTP (only used against test endpoints).
    /// Defaults to `false` so production deployments fail closed.
    #[serde(default)]
    pub allow_http: bool,
}

/// Google Cloud Storage connection details.
///
/// `service_account_json` is [`Secret`]-typed — see [`S3Config`]'s docs for
/// the redaction/persistence split this mirrors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcsConfig {
    /// Inline service-account JSON (holds a private key). When unset, the
    /// GCS driver falls back to Application Default Credentials
    /// (`GOOGLE_APPLICATION_CREDENTIALS`, Workload Identity, gcloud user
    /// creds).
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub service_account_json: Option<Secret>,
    /// Path to a service-account JSON file. Alternative to
    /// [`Self::service_account_json`] for callers who want to keep the file
    /// out of the catalog row. A path is not itself a secret.
    pub service_account_path: Option<String>,
}

/// Azure Blob Storage connection details.
///
/// `account_key`/`sas_token`/`client_secret` are [`Secret`]-typed — see
/// [`S3Config`]'s docs for the redaction/persistence split this mirrors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Storage account name (e.g. `"myaccount"` for
    /// `myaccount.blob.core.windows.net`).
    pub account_name: Option<String>,
    /// Account access key. Mutually exclusive with [`Self::sas_token`].
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub account_key: Option<Secret>,
    /// Shared-access-signature token.
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub sas_token: Option<Secret>,
    /// Tenant id for OAuth / Managed Identity auth.
    pub tenant_id: Option<String>,
    /// Client id for OAuth.
    pub client_id: Option<String>,
    /// Client secret paired with [`Self::client_id`].
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub client_secret: Option<Secret>,
}

/// Cloudflare R2 connection details.
///
/// R2 speaks the S3 API, so the driver is the S3 driver underneath. This
/// config exists so a deployer supplies only the R2-shaped inputs and the
/// engine derives the two S3 quirks R2 imposes — an account-scoped endpoint
/// and `region = "auto"` — rather than the deployer hand-rolling an
/// [`S3Config`] and risking either one.
///
/// `secret_access_key` is [`Secret`]-typed — see [`S3Config`]'s docs for the
/// redaction/persistence split this mirrors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct R2Config {
    /// Cloudflare account id. The endpoint is derived as
    /// `https://{account_id}.r2.cloudflarestorage.com` unless
    /// [`Self::endpoint`] overrides it.
    pub account_id: Option<String>,
    /// Explicit endpoint override — an R2 custom domain, or a test endpoint.
    /// Takes precedence over the account-derived endpoint.
    pub endpoint: Option<String>,
    /// R2 access key id (an S3-style token minted in the R2 dashboard / API).
    /// When unset, the S3 SDK's default credential chain applies.
    pub access_key_id: Option<String>,
    /// Secret access key paired with [`Self::access_key_id`].
    #[serde(
        default,
        serialize_with = "serialize_exposed",
        deserialize_with = "deserialize_inline"
    )]
    pub secret_access_key: Option<Secret>,
    /// Allow plain HTTP — only for test endpoints. Defaults `false` (fail closed).
    #[serde(default)]
    pub allow_http: bool,
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
    /// Cloudflare R2 — see [`R2Config`].
    R2(R2Config),
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
            Self::R2(c) => c.validate(),
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
            self.secret_access_key.as_ref(),
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

impl R2Config {
    /// The S3 endpoint the R2 driver talks to: the explicit `endpoint` override
    /// if set, else the account-scoped `https://{account_id}.r2.cloudflarestorage.com`.
    /// `None` when neither is available — a config error caught by [`Self::validate`].
    pub fn resolved_endpoint(&self) -> Option<String> {
        self.endpoint.clone().or_else(|| {
            self.account_id
                .as_ref()
                .map(|a| format!("https://{a}.r2.cloudflarestorage.com"))
        })
    }

    /// Require an addressable endpoint (an `account_id` or an explicit `endpoint`),
    /// and reject a half-set credential pair — the same fail-closed discipline the
    /// other backends apply.
    pub fn validate(&self) -> Result<(), StorageError> {
        if self.resolved_endpoint().is_none() {
            return Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                reason: "R2 requires either account_id or an explicit endpoint".into(),
            });
        }
        match (
            self.access_key_id.as_deref(),
            self.secret_access_key.as_ref(),
        ) {
            (Some(_), None) => Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                reason: "access_key_id is set but secret_access_key is missing".into(),
            }),
            (None, Some(_)) => Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                reason: "secret_access_key is set but access_key_id is missing".into(),
            }),
            _ => Ok(()),
        }
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
    fn r2_derives_endpoint_from_account_id() {
        let cfg = R2Config {
            account_id: Some("abc123".into()),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolved_endpoint().as_deref(),
            Some("https://abc123.r2.cloudflarestorage.com")
        );
    }

    #[test]
    fn r2_explicit_endpoint_overrides_account_id() {
        let cfg = R2Config {
            account_id: Some("abc123".into()),
            endpoint: Some("https://files.example.com".into()),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolved_endpoint().as_deref(),
            Some("https://files.example.com")
        );
    }

    #[test]
    fn r2_validate_requires_account_or_endpoint() {
        assert!(matches!(
            R2Config::default().validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                ..
            })
        ));
    }

    #[test]
    fn r2_validate_rejects_half_credentials() {
        let bad = R2Config {
            account_id: Some("abc".into()),
            access_key_id: Some("k".into()),
            ..Default::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                ..
            })
        ));
    }

    #[test]
    fn r2_validate_accepts_account_plus_full_credentials() {
        let good = R2Config {
            account_id: Some("abc".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
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

    // ── #483: persisted credentials are inline-only on read ──────────────
    //
    // A persisted credential (a `sources.options` row) round-trips through
    // `serde_json` byte-for-byte in BOTH directions, exactly as it did
    // before `Secret` existed: (a)/(b) below pin the literal pre-existing
    // JSON shape and that reloading it exposes the original plaintext; (c)
    // pins that a credential whose text happens to read `{ file = "…" }` is
    // an ordinary opaque string, never treated as a file directive; (d)
    // pins that the object form itself is refused, at every one of the
    // seven credential positions, without ever reading a file or echoing
    // the attempted path; (e) pins the pre-existing unknown-top-level-key
    // behavior (ignored — none of these four structs carries
    // `deny_unknown_fields`).

    #[test]
    fn s3_config_persisted_json_round_trips_byte_for_byte_with_every_credential_set() {
        let cfg = S3Config {
            region: Some("us-east-1".into()),
            endpoint: Some("https://s3.example.com".into()),
            access_key_id: Some("AKIAEXAMPLE".into()),
            secret_access_key: Some(Secret::new("s3-secret-key")),
            session_token: Some(Secret::new("s3-session-token")),
            allow_http: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(
            json,
            r#"{"region":"us-east-1","endpoint":"https://s3.example.com","access_key_id":"AKIAEXAMPLE","secret_access_key":"s3-secret-key","session_token":"s3-session-token","allow_http":false}"#
        );
        let reloaded: S3Config = serde_json::from_str(&json).unwrap();
        assert_eq!(
            reloaded.secret_access_key.unwrap().expose(),
            "s3-secret-key"
        );
        assert_eq!(reloaded.session_token.unwrap().expose(), "s3-session-token");
    }

    #[test]
    fn gcs_config_persisted_json_round_trips_byte_for_byte_with_every_credential_set() {
        let cfg = GcsConfig {
            service_account_json: Some(Secret::new("gcs-service-account-json")),
            service_account_path: Some("/etc/gcs/key.json".into()),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(
            json,
            r#"{"service_account_json":"gcs-service-account-json","service_account_path":"/etc/gcs/key.json"}"#
        );
        let reloaded: GcsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            reloaded.service_account_json.unwrap().expose(),
            "gcs-service-account-json"
        );
    }

    #[test]
    fn azure_config_persisted_json_round_trips_byte_for_byte_with_every_credential_set() {
        let cfg = AzureConfig {
            account_name: Some("myaccount".into()),
            account_key: Some(Secret::new("azure-account-key")),
            sas_token: Some(Secret::new("azure-sas-token")),
            tenant_id: Some("tenant-1".into()),
            client_id: Some("client-1".into()),
            client_secret: Some(Secret::new("azure-client-secret")),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(
            json,
            r#"{"account_name":"myaccount","account_key":"azure-account-key","sas_token":"azure-sas-token","tenant_id":"tenant-1","client_id":"client-1","client_secret":"azure-client-secret"}"#
        );
        let reloaded: AzureConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded.account_key.unwrap().expose(), "azure-account-key");
        assert_eq!(reloaded.sas_token.unwrap().expose(), "azure-sas-token");
        assert_eq!(
            reloaded.client_secret.unwrap().expose(),
            "azure-client-secret"
        );
    }

    #[test]
    fn r2_config_persisted_json_round_trips_byte_for_byte_with_every_credential_set() {
        let cfg = R2Config {
            account_id: Some("r2acct".into()),
            endpoint: Some("https://r2.example.com".into()),
            access_key_id: Some("R2KEY".into()),
            secret_access_key: Some(Secret::new("r2-secret-key")),
            allow_http: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(
            json,
            r#"{"account_id":"r2acct","endpoint":"https://r2.example.com","access_key_id":"R2KEY","secret_access_key":"r2-secret-key","allow_http":false}"#
        );
        let reloaded: R2Config = serde_json::from_str(&json).unwrap();
        assert_eq!(
            reloaded.secret_access_key.unwrap().expose(),
            "r2-secret-key"
        );
    }

    /// (c) A persisted credential whose text IS the file-form spelling is an
    /// ordinary opaque inline string — `deserialize_inline` never applies
    /// `SecretSource`'s "quoted file table" refusal (that refusal is a
    /// `JammiConfig` file/env-layer rule, not a persisted-row rule).
    #[test]
    fn literal_file_table_text_round_trips_as_an_opaque_string_at_every_position() {
        let literal = r#"{ file = "/x" }"#;
        let s3: S3Config =
            serde_json::from_str(&format!(r#"{{"secret_access_key":{:?}}}"#, literal)).unwrap();
        assert_eq!(s3.secret_access_key.unwrap().expose(), literal);

        let gcs: GcsConfig =
            serde_json::from_str(&format!(r#"{{"service_account_json":{:?}}}"#, literal)).unwrap();
        assert_eq!(gcs.service_account_json.unwrap().expose(), literal);

        let azure: AzureConfig =
            serde_json::from_str(&format!(r#"{{"account_key":{:?}}}"#, literal)).unwrap();
        assert_eq!(azure.account_key.unwrap().expose(), literal);

        let r2: R2Config =
            serde_json::from_str(&format!(r#"{{"secret_access_key":{:?}}}"#, literal)).unwrap();
        assert_eq!(r2.secret_access_key.unwrap().expose(), literal);
    }

    /// (d) The object form `{"file": "…"}` is refused at every one of the
    /// seven credential positions across all four cloud variants — an
    /// `invalid_type` error naming the field (via `serde_path_to_error`,
    /// which tracks the struct path correctly for these plain, non-tagged
    /// structs), never a value/path echo, and — the load-bearing half of
    /// this fix — NEVER a file read. `/etc/passwd` is a real, readable file
    /// on every CI runner and dev machine; pre-fix, this exact JSON made
    /// `Secret`'s `Deserialize` (via `SecretSource`) read it and expose its
    /// contents as the "credential" (see the RED capture in this commit's
    /// message).
    #[test]
    fn object_form_is_refused_at_every_credential_position_names_field_no_path_leak() {
        fn assert_refused<T: serde::de::DeserializeOwned>(json: &str, field: &str) {
            let mut de = serde_json::Deserializer::from_str(json);
            let err = serde_path_to_error::deserialize::<_, T>(&mut de)
                .err()
                .unwrap_or_else(|| panic!("{field}: object form must be refused (json = {json})"));
            let msg = err.to_string();
            assert!(msg.contains(field), "{field}: msg = {msg}");
            assert!(!msg.contains("/etc/passwd"), "{field}: msg = {msg}");
        }

        assert_refused::<S3Config>(
            r#"{"secret_access_key":{"file":"/etc/passwd"}}"#,
            "secret_access_key",
        );
        assert_refused::<S3Config>(
            r#"{"session_token":{"file":"/etc/passwd"}}"#,
            "session_token",
        );
        assert_refused::<GcsConfig>(
            r#"{"service_account_json":{"file":"/etc/passwd"}}"#,
            "service_account_json",
        );
        assert_refused::<AzureConfig>(r#"{"account_key":{"file":"/etc/passwd"}}"#, "account_key");
        assert_refused::<AzureConfig>(r#"{"sas_token":{"file":"/etc/passwd"}}"#, "sas_token");
        assert_refused::<AzureConfig>(
            r#"{"client_secret":{"file":"/etc/passwd"}}"#,
            "client_secret",
        );
        assert_refused::<R2Config>(
            r#"{"secret_access_key":{"file":"/etc/passwd"}}"#,
            "secret_access_key",
        );
    }

    /// (e) Pin the pre-existing unknown-top-level-key behavior: none of
    /// these four structs carries `#[serde(deny_unknown_fields)]` (unlike
    /// `crate::config::JammiConfig`'s config-only `*Section` mirrors, which
    /// do), so an old or forward-written row with an unrecognized key still
    /// reloads rather than refusing outright.
    #[test]
    fn unknown_top_level_key_is_ignored_not_denied() {
        let s3: S3Config =
            serde_json::from_str(r#"{"region":"us-east-1","future_field":"whatever"}"#).unwrap();
        assert_eq!(s3.region.as_deref(), Some("us-east-1"));
    }
}
