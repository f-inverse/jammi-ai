//! Object-store-backed storage layer for Jammi-owned artifacts (result
//! Parquet, sidecar ANN indexes, model checkpoints).
//!
//! User-facing surface is the [`JammiObjectStore`] handle: parse a URL via
//! [`StorageUrl`], hand it to the [`StorageRegistry`] to get a driver
//! (typed [`StorageError`] on failure), then use [`writer::ObjectParquetWriter`]
//! and [`sidecar_layout`] to round-trip Arrow data + USearch indexes.

pub mod builder;
pub mod config;
pub mod error;
pub mod object_store_handle;
pub mod reader;
pub mod registry;
pub mod sidecar_layout;
pub mod url;
pub mod writer;

pub use builder::{build_object_store, DynObjectStore};
pub use config::{AzureConfig, CloudConfig, GcsConfig, R2Config, S3Config};
pub use error::StorageError;
pub use object_store_handle::JammiObjectStore;
pub use registry::StorageRegistry;
pub use url::{Scheme, StorageUrl};
pub use writer::ObjectParquetWriter;
