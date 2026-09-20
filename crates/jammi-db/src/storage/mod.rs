//! Object-store-backed storage layer for Jammi-owned artifacts (result
//! Parquet, sidecar ANN indexes, model checkpoints).
//!
//! User-facing surface is the [`JammiObjectStore`] handle: parse a URL via
//! [`StorageUrl`], hand it to [`StorageRegistry::handle_for`] (or build a
//! standalone handle with [`JammiObjectStore::open`]) to get a guarded
//! handle (typed [`StorageError`] on failure), then use
//! [`writer::ObjectParquetWriter`] and [`sidecar_layout`] to round-trip
//! Arrow data + USearch indexes.
//!
//! # Sealed: no raw driver leaves this crate via the registry/builder
//!
//! `registry::StorageRegistry::driver_for` and
//! `builder::build_object_store` hand back the raw, unguarded
//! `Arc<dyn object_store::ObjectStore>` — the type `ObjectStoreExt::delete`
//! removes any key from, with no `models/` refusal. Both are `pub(crate)`;
//! each doctest below is compiled as an OUTSIDE consumer of the crate
//! (every rustdoc test is its own crate linked against the built library),
//! so it proves the compiler — not a review — refuses the raw route. The
//! error codes are MEASURED, not assumed (rustdoc enforces the declared
//! code: a `compile_fail,E0603` block whose real error is `E0624` is itself
//! a doctest FAILURE): a sealed INHERENT method reached by method-call
//! syntax is `E0624` ("method is private"); a sealed FREE FUNCTION reached
//! by path is `E0603` ("item is private").
//!
//! ```compile_fail,E0624
//! let registry = jammi_db::storage::StorageRegistry::new();
//! let url = jammi_db::storage::StorageUrl::memory("x");
//! // `driver_for` is a `pub(crate)` inherent method: E0624 outside jammi-db.
//! let _driver = registry.driver_for(&url, None);
//! ```
//!
//! ```compile_fail,E0603
//! let url = jammi_db::storage::StorageUrl::memory("x");
//! // `build_object_store` is a `pub(crate)` free fn: E0603 outside jammi-db.
//! let _driver = jammi_db::storage::builder::build_object_store(&url, None);
//! ```
//!
//! A shorter path, `jammi_db::storage::build_object_store` (skipping the
//! `builder::` segment), is fenced too — but by a DIFFERENT error, because
//! this module's own `pub use` list below never re-exports the function at
//! all (a `pub use` of a `pub(crate)` item is itself a compile error,
//! `E0364`, so there is no re-export to narrow — the path is simply absent):
//!
//! ```compile_fail,E0425
//! let url = jammi_db::storage::StorageUrl::memory("x");
//! // No such path exists at all: E0425, "cannot find function
//! // `build_object_store` in module `storage`".
//! let _driver = jammi_db::storage::build_object_store(&url, None);
//! ```
//!
//! Each doctest bites: making `driver_for`/`build_object_store` `pub` makes
//! the first two snippets compile;
//! adding `pub use builder::build_object_store;` to this module's `pub use`
//! list makes the third compile too (a real path would then exist). Either
//! mutation makes `cargo test -p jammi-db --doc` report the corresponding
//! `compile_fail` example as a FAILURE (it compiled when it should not
//! have) rather than silently passing.
//!
//! The registry/builder route closed here is not the only way to reach a
//! raw driver — `object_store_handle::JammiObjectStore::driver`'s own doc
//! names the two residuals `models_delete_call_sites.rs` still tracks.

pub mod builder;
pub mod config;
pub mod error;
pub mod index_cache;
pub mod object_store_handle;
pub mod read_view;
pub mod reader;
pub mod registry;
pub mod sidecar_layout;
pub mod url;
pub mod writer;

/// Lowercase-hex sha256 of a byte slice — the shared content-address primitive
/// for the model-artifact bundle cache ([`crate::store::artifact`]) and the ANN
/// segment cache ([`index_cache`]).
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

pub use builder::DynObjectStore;
pub use builder::{location_determinants, location_determinants_with, BuilderSeeds};
pub use config::{AzureConfig, CloudConfig, GcsConfig, R2Config, S3Config};
pub use error::StorageError;
pub use object_store_handle::{DeleteOutcome, JammiObjectStore, ObjectMeta};
pub use read_view::ReadView;
pub use registry::StorageRegistry;
pub use url::{Scheme, StorageUrl};
pub use writer::ObjectParquetWriter;
