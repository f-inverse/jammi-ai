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

/// The engine's self-description: version, compiled feature flags, the storage
/// backends this build can address, and the gRPC service tiers a deployment
/// mounted.
///
/// The first three fields are compile-time facts and are filled by
/// [`ServerInfo::current`]. `services` is a *runtime* fact — which gRPC service
/// tiers the running deployment chose to mount — so it lives outside the
/// engine's knowledge: [`ServerInfo::current`] leaves it empty (the embedded,
/// in-process engine mounts no gRPC services) and the server layer overrides it
/// with the actually-mounted tier set when answering `GetServerInfo` over the
/// wire.
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
    /// The gRPC service tiers this deployment mounted, sorted. Empty for the
    /// embedded engine (it serves no gRPC); the server layer fills it with the
    /// runtime-resolved tier set. See `jammi-server`'s service-tier mechanism.
    pub services: Vec<String>,
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
    ///
    /// `services` is left empty: the mounted gRPC tier set is a runtime fact
    /// the engine library does not know. The embedded engine serves no gRPC, so
    /// empty is its truthful answer; the server layer overrides this field with
    /// the tiers it actually mounted.
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
            services: Vec::new(),
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
        // The embedded engine mounts no gRPC services; the runtime tier set is
        // the server layer's to fill.
        assert!(info.services.is_empty());
    }
}
