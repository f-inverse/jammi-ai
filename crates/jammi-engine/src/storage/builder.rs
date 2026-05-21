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
    match url.scheme() {
        Scheme::File => build_file(url),
        Scheme::Memory => Ok(Arc::new(object_store::memory::InMemory::new())),
        Scheme::S3 => build_s3(url, config),
        Scheme::Gcs => build_gcs(url, config),
        Scheme::Azure => build_azure(url, config),
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

    let mut builder = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);

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
            builder = builder.with_secret_access_key(secret);
        }
        if let Some(token) = &s3.session_token {
            builder = builder.with_token(token);
        }
        if s3.allow_http {
            builder = builder.with_allow_http(true);
        }
    }

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

    let mut builder = object_store::gcp::GoogleCloudStorageBuilder::from_env()
        .with_bucket_name(bucket);

    if let Some(CloudConfig::Gcs(gcs)) = config {
        if let Some(json) = &gcs.service_account_json {
            builder = builder.with_service_account_key(json);
        }
        if let Some(path) = &gcs.service_account_path {
            builder = builder.with_service_account_path(path);
        }
    }

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
    Err(StorageError::SchemeNotEnabled { scheme: Scheme::Gcs })
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

    let mut builder = object_store::azure::MicrosoftAzureBuilder::from_env()
        .with_container_name(container);

    if let Some(CloudConfig::Azure(azure)) = config {
        if let Some(name) = &azure.account_name {
            builder = builder.with_account(name);
        }
        if let Some(key) = &azure.account_key {
            builder = builder.with_access_key(key);
        }
        if let Some(tenant) = &azure.tenant_id {
            builder = builder.with_tenant_id(tenant);
        }
        if let Some(client) = &azure.client_id {
            builder = builder.with_client_id(client);
        }
        if let Some(secret) = &azure.client_secret {
            builder = builder.with_client_secret(secret);
        }
        if let Some(sas) = &azure.sas_token {
            // SAS tokens are query-string params; let the SDK parse them.
            let pairs: Vec<(String, String)> = sas
                .trim_start_matches('?')
                .split('&')
                .filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
                .collect();
            builder = builder.with_sas_authorization(pairs);
        }
    }

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
            StorageError::SchemeNotEnabled {
                scheme: Scheme::S3
            }
        ));
    }
}
