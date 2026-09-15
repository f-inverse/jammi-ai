//! Driver constructors. Each scheme has one entry point that turns a
//! [`StorageUrl`] (and an optional cloud-config struct) into a
//! [`DynObjectStore`]. Missing-cloud-feature errors come from here so the
//! [`crate::storage::registry`] stays scheme-agnostic.

use std::sync::Arc;

use object_store::ObjectStore;

use super::config::CloudConfig;
use super::error::StorageError;
use super::url::{Scheme, StorageUrl};

/// Trait object alias for a thread-safe, dynamically-dispatched
/// `object_store::ObjectStore` — what every scheme builder returns.
pub type DynObjectStore = Arc<dyn ObjectStore>;

/// Build the concrete `ObjectStore` driver for `url`, optionally applying
/// `config` to fill in credentials / endpoint / region.
///
/// Returns [`StorageError::SchemeNotEnabled`] when the URL points at a
/// cloud that wasn't compiled in.
pub fn build_object_store(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    // Reject partial credentials up front so the caller gets a typed
    // [`StorageError::DriverInit`] at build time rather than a deep-in-the-
    // request SDK error on the first GET.
    if let Some(cfg) = config {
        cfg.validate()?;
    }
    match url.scheme() {
        Scheme::File => build_file(url),
        Scheme::Memory => Ok(Arc::new(object_store::memory::InMemory::new())),
        Scheme::S3 => build_s3(url, config),
        Scheme::Gcs => build_gcs(url, config),
        Scheme::Azure => build_azure(url, config),
        Scheme::R2 => build_r2(url, config),
    }
}

fn build_file(url: &StorageUrl) -> Result<DynObjectStore, StorageError> {
    // `LocalFileSystem::new()` returns a store rooted at "/" — the absolute
    // path lives in each `object_store::path::Path` the caller hands in.
    // This keeps `JammiObjectStore::put / get` portable across schemes
    // (the path component is always relative to the driver's root).
    let _ = url; // scheme already validated; root is always "/"
    Ok(Arc::new(object_store::local::LocalFileSystem::new()))
}

#[cfg(feature = "storage-s3")]
fn build_s3(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    let bucket = url
        .path()
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| StorageError::DriverInit {
            scheme: Scheme::S3,
            reason: "S3 URL has no bucket".into(),
        })?;

    let builder = configure_s3(
        object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket),
        config,
    );
    let store = builder.build().map_err(|e| StorageError::DriverInit {
        scheme: Scheme::S3,
        reason: e.to_string(),
    })?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "storage-s3"))]
fn build_s3(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    Err(StorageError::SchemeNotEnabled { scheme: Scheme::S3 })
}

#[cfg(feature = "storage-gcs")]
fn build_gcs(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    let bucket = url
        .path()
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| StorageError::DriverInit {
            scheme: Scheme::Gcs,
            reason: "GCS URL has no bucket".into(),
        })?;

    let builder = configure_gcs(
        object_store::gcp::GoogleCloudStorageBuilder::from_env().with_bucket_name(bucket),
        config,
    );
    let store = builder.build().map_err(|e| StorageError::DriverInit {
        scheme: Scheme::Gcs,
        reason: e.to_string(),
    })?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "storage-gcs"))]
fn build_gcs(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    Err(StorageError::SchemeNotEnabled {
        scheme: Scheme::Gcs,
    })
}

#[cfg(feature = "storage-azure")]
fn build_azure(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    let container = url
        .path()
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| StorageError::DriverInit {
            scheme: Scheme::Azure,
            reason: "Azure URL has no container".into(),
        })?;

    let builder = configure_azure(
        object_store::azure::MicrosoftAzureBuilder::from_env().with_container_name(container),
        config,
    );
    let store = builder.build().map_err(|e| StorageError::DriverInit {
        scheme: Scheme::Azure,
        reason: e.to_string(),
    })?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "storage-azure"))]
fn build_azure(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    Err(StorageError::SchemeNotEnabled {
        scheme: Scheme::Azure,
    })
}

// R2 is the S3 driver underneath; W2 keeps it a first-class scheme so the two
// quirks R2 imposes — an account-scoped endpoint and `region = "auto"` — are
// derived here instead of being a deployer's hand-rolled `S3Config` incantation.
#[cfg(feature = "storage-r2")]
fn build_r2(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    let bucket = url
        .path()
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| StorageError::DriverInit {
            scheme: Scheme::R2,
            reason: "R2 URL has no bucket".into(),
        })?;

    let (r2, endpoint) = resolve_r2(config)?;
    // R2 speaks S3 with `region = "auto"`, reached at the account-scoped
    // endpoint, with path-style addressing (object_store's default — R2 has no
    // per-bucket subdomains on the account endpoint).
    let builder = configure_r2(
        object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket),
        r2,
        &endpoint,
    );
    let store = builder.build().map_err(|e| StorageError::DriverInit {
        scheme: Scheme::R2,
        reason: e.to_string(),
    })?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "storage-r2"))]
fn build_r2(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
) -> Result<DynObjectStore, StorageError> {
    Err(StorageError::SchemeNotEnabled { scheme: Scheme::R2 })
}

#[cfg(any(
    feature = "storage-s3",
    feature = "storage-gcs",
    feature = "storage-azure",
    feature = "storage-r2"
))]
/// The bucket (or container) a cloud URL names: its first path segment,
/// the SAME split every `build_*` above performs.
fn bucket_of<'a>(url: &'a StorageUrl, scheme: Scheme, what: &str) -> Result<&'a str, StorageError> {
    url.path()
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| StorageError::DriverInit {
            scheme,
            reason: format!("{what} URL has no bucket"),
        })
}

#[cfg(feature = "storage-s3")]
fn configure_s3(
    mut builder: object_store::aws::AmazonS3Builder,
    config: Option<&CloudConfig>,
) -> object_store::aws::AmazonS3Builder {
    if let Some(CloudConfig::S3(s3)) = config {
        if let Some(region) = &s3.region {
            builder = builder.with_region(region);
        }
        if let Some(endpoint) = &s3.endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(key) = &s3.access_key_id {
            builder = builder.with_access_key_id(key);
        }
        if let Some(secret) = &s3.secret_access_key {
            builder = builder.with_secret_access_key(secret.expose());
        }
        if let Some(token) = &s3.session_token {
            builder = builder.with_token(token.expose());
        }
        if s3.allow_http {
            builder = builder.with_allow_http(true);
        }
    }
    builder
}

#[cfg(feature = "storage-gcs")]
fn configure_gcs(
    mut builder: object_store::gcp::GoogleCloudStorageBuilder,
    config: Option<&CloudConfig>,
) -> object_store::gcp::GoogleCloudStorageBuilder {
    if let Some(CloudConfig::Gcs(gcs)) = config {
        if let Some(json) = &gcs.service_account_json {
            builder = builder.with_service_account_key(json.expose());
        }
        if let Some(path) = &gcs.service_account_path {
            builder = builder.with_service_account_path(path);
        }
    }
    builder
}

#[cfg(feature = "storage-azure")]
fn configure_azure(
    mut builder: object_store::azure::MicrosoftAzureBuilder,
    config: Option<&CloudConfig>,
) -> object_store::azure::MicrosoftAzureBuilder {
    if let Some(CloudConfig::Azure(azure)) = config {
        if let Some(name) = &azure.account_name {
            builder = builder.with_account(name);
        }
        if let Some(key) = &azure.account_key {
            builder = builder.with_access_key(key.expose());
        }
        if let Some(tenant) = &azure.tenant_id {
            builder = builder.with_tenant_id(tenant);
        }
        if let Some(client) = &azure.client_id {
            builder = builder.with_client_id(client);
        }
        if let Some(secret) = &azure.client_secret {
            builder = builder.with_client_secret(secret.expose());
        }
        if let Some(sas) = &azure.sas_token {
            // SAS tokens are query-string params; let the SDK parse them.
            let pairs: Vec<(String, String)> = sas
                .expose()
                .trim_start_matches('?')
                .split('&')
                .filter_map(|kv| {
                    kv.split_once('=')
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                })
                .collect();
            builder = builder.with_sas_authorization(pairs);
        }
    }
    builder
}

/// R2's mandatory config and the account-scoped endpoint it resolves to —
/// the refusal `build_r2` makes, shared with the identity derivation.
#[cfg(feature = "storage-r2")]
fn resolve_r2(
    config: Option<&CloudConfig>,
) -> Result<(&super::config::R2Config, String), StorageError> {
    // R2 has no public default endpoint, so unlike S3 it requires a config to
    // know which account to talk to.
    let r2 = match config {
        Some(CloudConfig::R2(r2)) => r2,
        _ => {
            return Err(StorageError::DriverInit {
                scheme: Scheme::R2,
                reason: "R2 requires an R2Config with account_id or endpoint".into(),
            })
        }
    };
    let endpoint = r2
        .resolved_endpoint()
        .ok_or_else(|| StorageError::DriverInit {
            scheme: Scheme::R2,
            reason: "R2 requires either account_id or an explicit endpoint".into(),
        })?;
    Ok((r2, endpoint))
}

#[cfg(feature = "storage-r2")]
fn configure_r2(
    builder: object_store::aws::AmazonS3Builder,
    r2: &super::config::R2Config,
    endpoint: &str,
) -> object_store::aws::AmazonS3Builder {
    // R2 speaks S3 with `region = "auto"`, reached at the account-scoped
    // endpoint, with path-style addressing (object_store's default — R2 has no
    // per-bucket subdomains on the account endpoint).
    let mut builder = builder.with_region("auto").with_endpoint(endpoint);
    if let Some(key) = &r2.access_key_id {
        builder = builder.with_access_key_id(key);
    }
    if let Some(secret) = &r2.secret_access_key {
        builder = builder.with_secret_access_key(secret.expose());
    }
    if r2.allow_http {
        builder = builder.with_allow_http(true);
    }
    builder
}

/// The starting builders [`location_determinants`] derives from: the process
/// environment (`from_env()` — what [`build_object_store`] itself starts
/// from) or, for a test, an explicit variable set fed through the SAME
/// `<PREFIX>_` filter and key table `from_env` applies, so a spelling
/// object_store accepts in the environment is accepted here and nothing
/// this crate spells on its own can drift from the builder.
#[derive(Clone, Default)]
pub struct BuilderSeeds {
    #[cfg(any(feature = "storage-s3", feature = "storage-r2"))]
    s3: Option<object_store::aws::AmazonS3Builder>,
    #[cfg(feature = "storage-gcs")]
    gcs: Option<object_store::gcp::GoogleCloudStorageBuilder>,
    #[cfg(feature = "storage-azure")]
    azure: Option<object_store::azure::MicrosoftAzureBuilder>,
    /// The variables themselves, for the ONE host value object_store reads
    /// with a bare `std::env::var` at `build()` time instead of through its
    /// key tables (`AZURITE_BLOB_STORAGE_URL`, the Azure emulator arm):
    /// `None` means the process environment, as `build()` itself reads it.
    #[cfg_attr(not(feature = "storage-azure"), allow(dead_code))]
    vars: Option<std::collections::BTreeMap<String, String>>,
}

impl BuilderSeeds {
    /// A variable exactly as object_store's own bare `std::env::var` read
    /// at `build()` would see it: from the explicit set, or the process.
    #[cfg(feature = "storage-azure")]
    fn raw(&self, key: &str) -> Option<String> {
        match &self.vars {
            Some(vars) => vars.get(key).cloned(),
            None => std::env::var(key).ok(),
        }
    }
}

impl BuilderSeeds {
    /// The process environment, exactly as `build_object_store` reads it.
    pub fn from_env() -> Self {
        Self {
            #[cfg(any(feature = "storage-s3", feature = "storage-r2"))]
            s3: Some(object_store::aws::AmazonS3Builder::from_env()),
            #[cfg(feature = "storage-gcs")]
            gcs: Some(object_store::gcp::GoogleCloudStorageBuilder::from_env()),
            #[cfg(feature = "storage-azure")]
            azure: Some(object_store::azure::MicrosoftAzureBuilder::from_env()),
            vars: None,
        }
    }

    /// An explicit variable set in place of the process environment: every
    /// `AWS_`/`AZURE_`/`GOOGLE_`-prefixed key is parsed by object_store's own
    /// config-key table (`AmazonS3ConfigKey`/`AzureConfigKey`/
    /// `GoogleConfigKey` `FromStr`, lower-cased — the documented behaviour of
    /// each builder's `from_env`), unknown keys ignored. Nothing from the
    /// process environment is read.
    pub fn from_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        #[cfg(any(feature = "storage-s3", feature = "storage-r2"))]
        let mut s3 = object_store::aws::AmazonS3Builder::new();
        #[cfg(feature = "storage-gcs")]
        let mut gcs = object_store::gcp::GoogleCloudStorageBuilder::new();
        #[cfg(feature = "storage-azure")]
        let mut azure = object_store::azure::MicrosoftAzureBuilder::new();
        let mut raw = std::collections::BTreeMap::new();
        for (key, value) in vars {
            let key = key.as_ref();
            let value: String = value.into();
            raw.insert(key.to_string(), value.clone());
            let lowered = key.to_ascii_lowercase();
            #[cfg(any(feature = "storage-s3", feature = "storage-r2"))]
            if key.starts_with("AWS_") {
                if let Ok(k) = lowered.parse::<object_store::aws::AmazonS3ConfigKey>() {
                    s3 = s3.with_config(k, value.clone());
                }
            }
            #[cfg(feature = "storage-gcs")]
            if key.starts_with("GOOGLE_") {
                if let Ok(k) = lowered.parse::<object_store::gcp::GoogleConfigKey>() {
                    gcs = gcs.with_config(k, value.clone());
                }
            }
            #[cfg(feature = "storage-azure")]
            if key.starts_with("AZURE_") {
                if let Ok(k) = lowered.parse::<object_store::azure::AzureConfigKey>() {
                    azure = azure.with_config(k, value.clone());
                }
            }
            let _ = (&lowered, &value);
        }
        Self {
            #[cfg(any(feature = "storage-s3", feature = "storage-r2"))]
            s3: Some(s3),
            #[cfg(feature = "storage-gcs")]
            gcs: Some(gcs),
            #[cfg(feature = "storage-azure")]
            azure: Some(azure),
            vars: Some(raw),
        }
    }
}

/// The values that decide WHICH service the store dials for `url` under
/// `config` — read back from the SAME builder [`build_object_store`]
/// constructs (the environment first, `config` on top, exactly the order
/// the `build_*` functions apply), so every spelling object_store accepts for
/// an endpoint, account, emulator or base URL (`AWS_ENDPOINT_URL`,
/// `AWS_ENDPOINT`, `AWS_ENDPOINT_URL_S3`, `AZURE_STORAGE_ENDPOINT`,
/// `AZURE_STORAGE_ACCOUNT_NAME`, `GOOGLE_BASE_URL`, …) is honoured here
/// exactly as at dial time. The one host value object_store reads with a
/// bare `std::env::var` outside its key tables — `AZURITE_BLOB_STORAGE_URL`,
/// in the Azure emulator arm — is read the same way here (its default
/// included), so the identity spells exactly the variables the driver
/// spells and no others; and every value is read as the driver reads it
/// (its boolean parser's five spellings, its URL parse).
/// Sorted `(key, value)` pairs; empty when the service's default host is
/// dialled. A scheme whose storage feature is compiled out has no
/// determinants (`Ok(empty)`): such a build cannot dial the scheme at all
/// (`build_object_store` refuses with `SchemeNotEnabled`), so a process that
/// registers such a root never opens a store there.
///
/// # Errors
///
/// The same refusals `build_object_store` makes before dialling: a URL with
/// no bucket, an R2 root with no config to resolve its endpoint.
pub fn location_determinants(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    location_determinants_with(url, config, &BuilderSeeds::from_env())
}

/// [`location_determinants`] over explicit [`BuilderSeeds`].
pub fn location_determinants_with(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
    seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    let mut pairs = match url.scheme() {
        Scheme::File | Scheme::Memory => Vec::new(),
        Scheme::S3 => s3_determinants(url, config, seeds)?,
        Scheme::Gcs => gcs_determinants(url, config, seeds)?,
        Scheme::Azure => azure_determinants(url, config, seeds)?,
        Scheme::R2 => r2_determinants(url, config, seeds)?,
    };
    pairs.sort();
    Ok(pairs)
}

#[cfg(any(
    feature = "storage-s3",
    feature = "storage-gcs",
    feature = "storage-azure",
    feature = "storage-r2"
))]
fn present(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

#[cfg(feature = "storage-s3")]
fn s3_determinants(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
    seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    use object_store::aws::AmazonS3ConfigKey as K;
    let bucket = bucket_of(url, Scheme::S3, "S3")?;
    let base = seeds
        .s3
        .clone()
        .unwrap_or_default()
        .with_bucket_name(bucket);
    let builder = configure_s3(base, config);
    // `build()` dials `s3_endpoint.or(endpoint)`: the S3-specific URL wins
    // over the generic one whatever set either.
    let endpoint = present(builder.get_config_value(&K::S3Endpoint))
        .or_else(|| present(builder.get_config_value(&K::Endpoint)));
    Ok(endpoint.map(|e| ("endpoint", e)).into_iter().collect())
}

#[cfg(not(feature = "storage-s3"))]
fn s3_determinants(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
    _seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    Ok(Vec::new())
}

#[cfg(feature = "storage-r2")]
fn r2_determinants(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
    seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    use object_store::aws::AmazonS3ConfigKey as K;
    let bucket = bucket_of(url, Scheme::R2, "R2")?;
    let (r2, endpoint) = resolve_r2(config)?;
    let base = seeds
        .s3
        .clone()
        .unwrap_or_default()
        .with_bucket_name(bucket);
    let builder = configure_r2(base, r2, &endpoint);
    // The same `s3_endpoint.or(endpoint)` the S3 driver dials: an
    // `AWS_ENDPOINT_URL_S3` in the environment overrides the configured R2
    // endpoint in the store, and therefore here.
    let dialled = present(builder.get_config_value(&K::S3Endpoint))
        .or_else(|| present(builder.get_config_value(&K::Endpoint)));
    Ok(dialled.map(|e| ("endpoint", e)).into_iter().collect())
}

#[cfg(not(feature = "storage-r2"))]
fn r2_determinants(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
    _seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    Ok(Vec::new())
}

#[cfg(feature = "storage-gcs")]
fn gcs_determinants(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
    seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    use object_store::gcp::GoogleConfigKey as K;
    let bucket = bucket_of(url, Scheme::Gcs, "GCS")?;
    let base = seeds
        .gcs
        .clone()
        .unwrap_or_default()
        .with_bucket_name(bucket);
    let builder = configure_gcs(base, config);
    // `GOOGLE_BASE_URL` repoints the whole service (an emulator, a mirror):
    // the bucket namespace is global only at the default base URL.
    let base_url = present(builder.get_config_value(&K::BaseUrl));
    Ok(base_url.map(|u| ("base_url", u)).into_iter().collect())
}

#[cfg(not(feature = "storage-gcs"))]
fn gcs_determinants(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
    _seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    Ok(Vec::new())
}

#[cfg(feature = "storage-azure")]
fn azure_determinants(
    url: &StorageUrl,
    config: Option<&CloudConfig>,
    seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    use object_store::azure::AzureConfigKey as K;
    let container = bucket_of(url, Scheme::Azure, "Azure")?;
    let base = seeds
        .azure
        .clone()
        .unwrap_or_default()
        .with_container_name(container);
    let builder = configure_azure(base, config);
    let mut pairs = Vec::new();
    // `get_config_value` returns the RAW string; `build()` reads it through
    // object_store's own boolean parser, which is not public — so the
    // parser is mirrored here, spelling for spelling, and the oracle below
    // walks every spelling it accepts.
    let flag = |key: K| {
        builder
            .get_config_value(&key)
            .is_some_and(|v| driver_bool(&v))
    };
    if flag(K::UseEmulator) {
        // `build()`'s emulator arm: the host is `AZURITE_BLOB_STORAGE_URL`
        // (a bare env read, default `http://127.0.0.1:10000`) and the
        // account, defaulting to the emulator's, is a path segment; the
        // endpoint and Fabric switch are ignored there.
        pairs.push(("use_emulator", "true".to_string()));
        // The driver parses the URL (`url_from_env`), so `http://h:10000`
        // and `http://h:10000/` are one host here as there; a value the
        // driver could not parse is kept raw (the driver refuses to build).
        let raw = present(seeds.raw("AZURITE_BLOB_STORAGE_URL"))
            .unwrap_or_else(|| "http://127.0.0.1:10000".to_string());
        pairs.push((
            "emulator_url",
            url::Url::parse(&raw).map(|u| u.to_string()).unwrap_or(raw),
        ));
        pairs.push((
            "account",
            present(builder.get_config_value(&K::AccountName))
                .unwrap_or_else(|| "devstoreaccount1".to_string()),
        ));
        return Ok(pairs);
    }
    // The account URL is the endpoint when set, else derived from the
    // account name — on the Fabric host when that switch is on.
    if let Some(account) = present(builder.get_config_value(&K::AccountName)) {
        pairs.push(("account", account));
    }
    match present(builder.get_config_value(&K::Endpoint)) {
        Some(endpoint) => pairs.push(("endpoint", endpoint)),
        None => {
            if flag(K::UseFabricEndpoint) {
                pairs.push(("use_fabric_endpoint", "true".to_string()));
            }
        }
    }
    Ok(pairs)
}

/// object_store's boolean parser (`config.rs`, `impl Parse for bool`), which
/// `build()` applies to every `ConfigValue<bool>` — `1`, `true`, `on`,
/// `yes`, `y` in any case are true; everything else is false — mirrored
/// because it is not public. Any spelling the driver takes as true must
/// switch the identity's arm too, or two members at two hosts would derive
/// one identity (the fourth oracle round's executed refutation).
#[cfg(feature = "storage-azure")]
fn driver_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes" | "y"
    )
}

#[cfg(not(feature = "storage-azure"))]
fn azure_determinants(
    _url: &StorageUrl,
    _config: Option<&CloudConfig>,
    _seeds: &BuilderSeeds,
) -> Result<Vec<(&'static str, String)>, StorageError> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_scheme_always_available() {
        let url = StorageUrl::parse("/tmp").unwrap();
        let store = build_object_store(&url, None).expect("file driver builds");
        // Smoke test: store implements Display.
        let _ = format!("{store}");
    }

    #[test]
    fn memory_scheme_always_available() {
        let url = StorageUrl::memory("test");
        let store = build_object_store(&url, None).expect("memory driver builds");
        let _ = format!("{store}");
    }

    #[cfg(not(feature = "storage-s3"))]
    #[test]
    fn s3_disabled_without_feature() {
        let url = StorageUrl::parse("s3://benchmarks/x").unwrap();
        let err = build_object_store(&url, None).unwrap_err();
        assert!(matches!(
            err,
            StorageError::SchemeNotEnabled { scheme: Scheme::S3 }
        ));
    }

    #[cfg(not(feature = "storage-r2"))]
    #[test]
    fn r2_disabled_without_feature() {
        let url = StorageUrl::parse("r2://archives/x").unwrap();
        let err = build_object_store(&url, None).unwrap_err();
        assert!(matches!(
            err,
            StorageError::SchemeNotEnabled { scheme: Scheme::R2 }
        ));
    }

    #[cfg(feature = "storage-r2")]
    #[test]
    fn r2_builds_driver_from_account_config() {
        use super::super::config::R2Config;
        let url = StorageUrl::parse("r2://archives/x").unwrap();
        let cfg = CloudConfig::R2(R2Config {
            account_id: Some("abc123".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            ..Default::default()
        });
        // No network: AmazonS3Builder::build() only constructs the client.
        let store = build_object_store(&url, Some(&cfg)).expect("r2 driver builds");
        let _ = format!("{store}");
    }

    #[cfg(feature = "storage-r2")]
    #[test]
    fn r2_requires_config() {
        let url = StorageUrl::parse("r2://archives/x").unwrap();
        let err = build_object_store(&url, None).unwrap_err();
        assert!(matches!(
            err,
            StorageError::DriverInit {
                scheme: Scheme::R2,
                ..
            }
        ));
    }

    /// The determinants are read back from the builder itself, so every
    /// spelling object_store accepts for an endpoint reaches them — the
    /// oracle enumerates spellings from object_store's documented key table;
    /// the derivation enumerates nothing.
    #[cfg(feature = "storage-s3")]
    #[test]
    fn s3_determinants_honour_every_endpoint_spelling_the_builder_does() {
        let url = StorageUrl::parse("s3://bucket/prefix").unwrap();
        let of = |vars: &[(&str, &str)]| {
            location_determinants_with(&url, None, &BuilderSeeds::from_vars(vars.iter().copied()))
                .unwrap()
        };
        assert!(of(&[]).is_empty(), "the service default has no determinant");
        for spelling in ["AWS_ENDPOINT_URL", "AWS_ENDPOINT", "AWS_ENDPOINT_URL_S3"] {
            assert_eq!(
                of(&[(spelling, "https://minio.local:9000")]),
                vec![("endpoint", "https://minio.local:9000".to_string())],
                "{spelling}"
            );
        }
        // `AWS_ENDPOINT_URL_S3` wins over the generic endpoint, as `build()` dials it.
        assert_eq!(
            of(&[
                ("AWS_ENDPOINT_URL", "https://generic"),
                ("AWS_ENDPOINT_URL_S3", "https://s3-specific")
            ]),
            vec![("endpoint", "https://s3-specific".to_string())]
        );
        // Config on top of the environment, as `build_s3` applies it.
        let cfg = CloudConfig::S3(super::super::config::S3Config {
            endpoint: Some("https://from-config".to_string()),
            ..Default::default()
        });
        assert_eq!(
            location_determinants_with(
                &url,
                Some(&cfg),
                &BuilderSeeds::from_vars([("AWS_ENDPOINT_URL", "https://from-env")])
            )
            .unwrap(),
            vec![("endpoint", "https://from-config".to_string())]
        );
        // …except that the S3-specific env URL still wins in `build()`, and so here.
        assert_eq!(
            location_determinants_with(
                &url,
                Some(&cfg),
                &BuilderSeeds::from_vars([("AWS_ENDPOINT_URL_S3", "https://s3-env")])
            )
            .unwrap(),
            vec![("endpoint", "https://s3-env".to_string())]
        );
        // Unknown keys and other prefixes are ignored; an empty value is unset.
        assert!(of(&[
            ("AWS_NOT_A_KEY", "x"),
            ("OTHER_ENDPOINT", "y"),
            ("AWS_ENDPOINT", "")
        ])
        .is_empty());
    }

    #[cfg(feature = "storage-r2")]
    #[test]
    fn r2_determinants_are_the_configured_endpoint_unless_the_s3_env_url_overrides_it() {
        let url = StorageUrl::parse("r2://bucket/prefix").unwrap();
        let cfg = |account: &str| {
            CloudConfig::R2(super::super::config::R2Config {
                account_id: Some(account.to_string()),
                ..Default::default()
            })
        };
        let blank = BuilderSeeds::from_vars(std::iter::empty::<(&str, &str)>());
        assert!(
            location_determinants_with(&url, None, &blank).is_err(),
            "R2 needs its config"
        );
        let a = location_determinants_with(&url, Some(&cfg("acct-a")), &blank).unwrap();
        let b = location_determinants_with(&url, Some(&cfg("acct-b")), &blank).unwrap();
        assert_ne!(a, b);
        assert_eq!(
            a,
            vec![(
                "endpoint",
                "https://acct-a.r2.cloudflarestorage.com".to_string()
            )]
        );
        let overridden = location_determinants_with(
            &url,
            Some(&cfg("acct-a")),
            &BuilderSeeds::from_vars([("AWS_ENDPOINT_URL_S3", "https://stray")]),
        )
        .unwrap();
        assert_eq!(overridden, vec![("endpoint", "https://stray".to_string())]);
    }

    #[cfg(feature = "storage-azure")]
    #[test]
    fn azure_determinants_carry_account_endpoint_emulator_and_fabric_as_the_builder_reads_them() {
        let url = StorageUrl::parse("azure://container/prefix").unwrap();
        let of = |vars: &[(&str, &str)]| {
            location_determinants_with(&url, None, &BuilderSeeds::from_vars(vars.iter().copied()))
                .unwrap()
        };
        assert!(of(&[]).is_empty());
        assert_eq!(
            of(&[("AZURE_STORAGE_ACCOUNT_NAME", "acct")]),
            vec![("account", "acct".to_string())]
        );
        for spelling in ["AZURE_STORAGE_ENDPOINT", "AZURE_ENDPOINT"] {
            assert_eq!(
                of(&[
                    ("AZURE_STORAGE_ACCOUNT_NAME", "acct"),
                    (spelling, "https://blob.local")
                ]),
                vec![
                    ("account", "acct".to_string()),
                    ("endpoint", "https://blob.local".to_string())
                ],
                "{spelling}"
            );
        }
        // The emulator arm dials AZURITE_BLOB_STORAGE_URL (a bare env read
        // in object_store, default 127.0.0.1:10000) with the account as a
        // path segment, and ignores the endpoint.
        assert_eq!(
            of(&[("AZURE_STORAGE_USE_EMULATOR", "true")]),
            vec![
                ("account", "devstoreaccount1".to_string()),
                ("emulator_url", "http://127.0.0.1:10000/".to_string()),
                ("use_emulator", "true".to_string()),
            ]
        );
        assert_ne!(
            of(&[
                ("AZURE_STORAGE_USE_EMULATOR", "true"),
                ("AZURITE_BLOB_STORAGE_URL", "http://host-a:10000")
            ]),
            of(&[
                ("AZURE_STORAGE_USE_EMULATOR", "true"),
                ("AZURITE_BLOB_STORAGE_URL", "http://host-b:10000")
            ]),
            "two Azurite hosts are two locations"
        );
        assert_eq!(
            of(&[
                ("AZURE_STORAGE_USE_EMULATOR", "true"),
                ("AZURE_STORAGE_ENDPOINT", "https://ignored")
            ]),
            of(&[("AZURE_STORAGE_USE_EMULATOR", "true")]),
            "the endpoint is ignored in emulator mode, as build() ignores it"
        );
        assert_eq!(
            of(&[
                ("AZURE_STORAGE_ACCOUNT_NAME", "acct"),
                ("AZURE_USE_FABRIC_ENDPOINT", "true")
            ]),
            vec![
                ("account", "acct".to_string()),
                ("use_fabric_endpoint", "true".to_string())
            ]
        );
        // Every spelling object_store's boolean parser accepts (`1`, `true`,
        // `on`, `yes`, `y`, any case) switches the arm exactly as `build()`
        // does; a spelling it rejects does not.
        for spelling in [
            "1", "true", "on", "yes", "y", "TRUE", "True", "Yes", "ON", " y ",
        ] {
            assert_eq!(
                of(&[
                    ("AZURE_STORAGE_USE_EMULATOR", spelling),
                    ("AZURITE_BLOB_STORAGE_URL", "http://host-a:10000"),
                ]),
                of(&[
                    ("AZURE_STORAGE_USE_EMULATOR", "true"),
                    ("AZURITE_BLOB_STORAGE_URL", "http://host-a:10000"),
                ]),
                "emulator spelling {spelling:?}"
            );
            assert_eq!(
                of(&[
                    ("AZURE_STORAGE_ACCOUNT_NAME", "acct"),
                    ("AZURE_USE_FABRIC_ENDPOINT", spelling)
                ]),
                vec![
                    ("account", "acct".to_string()),
                    ("use_fabric_endpoint", "true".to_string())
                ],
                "fabric spelling {spelling:?}"
            );
        }
        for spelling in ["0", "false", "off", "no", "n", ""] {
            assert_eq!(
                of(&[
                    ("AZURE_STORAGE_ACCOUNT_NAME", "acct"),
                    ("AZURE_USE_FABRIC_ENDPOINT", spelling)
                ]),
                vec![("account", "acct".to_string())],
                "not-true spelling {spelling:?}"
            );
        }
        // The emulator URL is parsed as the driver parses it: one host, with
        // or without the trailing slash.
        assert_eq!(
            of(&[
                ("AZURE_STORAGE_USE_EMULATOR", "1"),
                ("AZURITE_BLOB_STORAGE_URL", "http://host-a:10000")
            ]),
            of(&[
                ("AZURE_STORAGE_USE_EMULATOR", "1"),
                ("AZURITE_BLOB_STORAGE_URL", "http://host-a:10000/")
            ])
        );
    }

    #[cfg(feature = "storage-gcs")]
    #[test]
    fn gcs_determinants_carry_the_base_url_the_builder_reads() {
        let url = StorageUrl::parse("gs://bucket/prefix").unwrap();
        let blank = BuilderSeeds::from_vars(std::iter::empty::<(&str, &str)>());
        assert!(location_determinants_with(&url, None, &blank)
            .unwrap()
            .is_empty());
        assert_eq!(
            location_determinants_with(
                &url,
                None,
                &BuilderSeeds::from_vars([("GOOGLE_BASE_URL", "http://fake-gcs:4443")])
            )
            .unwrap(),
            vec![("base_url", "http://fake-gcs:4443".to_string())]
        );
    }
}
