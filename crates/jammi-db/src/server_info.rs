//! The engine's compile-time self-description.
//!
//! [`ServerInfo`] is the capabilities handshake: a client reads it once and
//! knows the engine's version, the optional feature flags the binary was built
//! with, and the storage URL schemes it can address — so it can negotiate
//! capability without probing each verb. Every field is a compile-time fact, so
//! [`ServerInfo::current`] reads them straight off `cfg!` and the crate version;
//! nothing here depends on session, tenant, or runtime state.
//!
//! This lives in `jammi-db` because that is where the gated capabilities
//! actually compile: the `postgres` / `mysql` federation backends and the
//! `storage-*` object-store drivers are `jammi-db` features (the server crate
//! forwards its own flags onto them), and the storage schemes are the
//! [`crate::storage::Scheme`] variants. Reporting capability anywhere else would
//! have to mirror those flags and drift from them.

use serde::Serialize;

/// The engine's self-description: version, compiled feature flags, and the
/// storage backends this build can address. Built from [`ServerInfo::current`];
/// the values are constant for a given binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServerInfo {
    /// The engine version — the lockstep workspace crate version.
    pub version: String,
    /// Optional engine capabilities compiled into this build (e.g. `"postgres"`,
    /// `"mysql"`, `"jetstream-broker"`), sorted.
    pub features: Vec<String>,
    /// Storage URL schemes this build can address (always `"file"` and
    /// `"memory"`; cloud schemes appear when their driver feature is compiled
    /// in), sorted.
    pub storage_backends: Vec<String>,
}

impl ServerInfo {
    /// The capabilities of the engine compiled into this binary.
    ///
    /// `version` is the crate version; `features` and `storage_backends` are
    /// read from `cfg!`, so each entry is present exactly when its code is —
    /// `file` and `memory` storage are always compiled, each cloud scheme
    /// appears only under its driver feature, and each federation backend only
    /// under its feature. R2 reuses the S3 driver, so `storage-r2` and
    /// `storage-s3` both contribute the `s3` backend.
    pub fn current() -> Self {
        let mut features: Vec<String> = Vec::new();
        if cfg!(feature = "postgres") {
            features.push("postgres".to_string());
        }
        if cfg!(feature = "mysql") {
            features.push("mysql".to_string());
        }
        if cfg!(feature = "jetstream-broker") {
            features.push("jetstream-broker".to_string());
        }
        features.sort();

        let mut storage_backends: Vec<String> = vec!["file".to_string(), "memory".to_string()];
        if cfg!(feature = "storage-s3") || cfg!(feature = "storage-r2") {
            storage_backends.push("s3".to_string());
        }
        if cfg!(feature = "storage-r2") {
            storage_backends.push("r2".to_string());
        }
        if cfg!(feature = "storage-gcs") {
            storage_backends.push("gs".to_string());
        }
        if cfg!(feature = "storage-azure") {
            storage_backends.push("azure".to_string());
        }
        storage_backends.sort();
        storage_backends.dedup();

        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            features,
            storage_backends,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_reports_version_and_always_present_backends() {
        let info = ServerInfo::current();
        // The crate version is the lockstep workspace version.
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        // file + memory are always compiled, regardless of feature set.
        assert!(info.storage_backends.contains(&"file".to_string()));
        assert!(info.storage_backends.contains(&"memory".to_string()));
        // Lists are sorted and free of duplicates.
        let mut sorted = info.storage_backends.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(info.storage_backends, sorted);
        let mut feats = info.features.clone();
        feats.sort();
        feats.dedup();
        assert_eq!(info.features, feats);
    }
}
