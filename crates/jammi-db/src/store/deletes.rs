//! The per-version deletion mask (`{table}__v{N}.deletes.parquet`).
//!
//! A cumulative sidecar: version N's file is the WHOLE mask. Columns `_row_id
//! Utf8 NOT NULL, _dead_through_version Int64 NOT NULL`, sorted by `_row_id`.
//! An entry `(K, h)` masks `K` in every fragment and segment whose stamped
//! version is `<= h` and nowhere else; a refresh producing N writes `entry[K]
//! = max(entry[K], N - 1)` for every superseded or deleted key, so artifacts
//! stamped N are never self-masked and an updated key present in two segments
//! is unambiguous. Row identity across fragments and segments is `_row_id`,
//! so one mask serves the ANN merge, the SQL/exact scan and
//! `read_vector_by_key`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::error::{JammiError, Result};
use crate::storage::{self, JammiObjectStore, ObjectParquetWriter};
use crate::store::manifest::ArtifactDigest;

/// The mask's on-disk schema.
pub fn deletes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_dead_through_version", DataType::Int64, false),
    ]))
}

/// The in-memory mask: key → dead-through horizon.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeletionMask {
    entries: HashMap<String, i64>,
}

impl DeletionMask {
    /// An empty mask (nothing is hidden).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build from entries (later duplicates keep the larger horizon).
    pub fn from_entries(entries: impl IntoIterator<Item = (String, i64)>) -> Self {
        let mut mask = Self::default();
        for (k, h) in entries {
            mask.raise(k, h);
        }
        mask
    }

    /// `entry[key] = max(entry[key], horizon)`.
    pub fn raise(&mut self, key: String, horizon: i64) {
        let slot = self.entries.entry(key).or_insert(horizon);
        if horizon > *slot {
            *slot = horizon;
        }
    }

    /// Whether `key` is hidden in an artifact stamped `version`.
    pub fn is_masked(&self, key: &str, version: i64) -> bool {
        self.entries.get(key).is_some_and(|h| *h >= version)
    }

    /// The horizon recorded for `key`, if any.
    pub fn horizon(&self, key: &str) -> Option<i64> {
        self.entries.get(key).copied()
    }

    /// Whether any entry can hide a row of an artifact stamped `version`
    /// (`false` lets a scan skip the mask entirely).
    pub fn masks_version(&self, version: i64) -> bool {
        self.entries.values().any(|h| *h >= version)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the mask has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries in `_row_id` order — the on-disk order.
    pub fn sorted_entries(&self) -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = self.entries.iter().map(|(k, h)| (k.clone(), *h)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// The keys, for a segment-membership count.
    pub fn entries(&self) -> impl Iterator<Item = (&str, i64)> {
        self.entries.iter().map(|(k, h)| (k.as_str(), *h))
    }

    /// Write the mask to `handle`'s object as Parquet (sorted by `_row_id`)
    /// and return `(entries, digest over the written bytes)`.
    pub async fn write(&self, handle: &JammiObjectStore) -> Result<(usize, ArtifactDigest)> {
        let schema = deletes_schema();
        let entries = self.sorted_entries();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from_iter_values(
                    entries.iter().map(|(k, _)| k.as_str()),
                )),
                Arc::new(Int64Array::from_iter_values(
                    entries.iter().map(|(_, h)| *h),
                )),
            ],
        )
        .map_err(|e| JammiError::Other(format!("deletes: build batch: {e}")))?;
        let mut writer = ObjectParquetWriter::open(handle, schema).await?;
        writer.write_batch(&batch).await?;
        writer.close().await?;
        let bytes = handle.get_bytes(&handle.data_path()?).await?;
        Ok((entries.len(), ArtifactDigest::of_bytes(&bytes)))
    }

    /// Read a mask back, schema-checked (K2): a file that is not exactly the
    /// two-column mask shape is a typed [`JammiError::IncompatibleFormat`] —
    /// this ENGINE's own sidecar (only [`Self::write`], in this module,
    /// ever produces one), so a corrupt mask is never the caller's fault —
    /// never a misread horizon.
    pub async fn read(handle: &JammiObjectStore, table: &str) -> Result<Self> {
        let batches = storage::reader::read_all_record_batches(handle).await?;
        let mut mask = Self::default();
        for batch in batches {
            let keys = batch
                .column_by_name("_row_id")
                .ok_or_else(|| schema_error(table, "_row_id", "Utf8", "missing"))?;
            let keys = arrow::compute::cast(keys, &DataType::Utf8).map_err(|_| {
                schema_error(table, "_row_id", "Utf8", &format!("{}", keys.data_type()))
            })?;
            let keys = keys
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| schema_error(table, "_row_id", "Utf8", "not a string array"))?;
            let horizons = batch
                .column_by_name("_dead_through_version")
                .ok_or_else(|| schema_error(table, "_dead_through_version", "Int64", "missing"))?;
            let horizons = horizons
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    schema_error(
                        table,
                        "_dead_through_version",
                        "Int64",
                        &format!("{}", horizons.data_type()),
                    )
                })?;
            for i in 0..batch.num_rows() {
                if keys.is_null(i) || horizons.is_null(i) {
                    return Err(schema_error(table, "_row_id", "non-null", "null"));
                }
                let h = horizons.value(i);
                if h < 0 {
                    return Err(schema_error(
                        table,
                        "_dead_through_version",
                        ">= 0",
                        &h.to_string(),
                    ));
                }
                mask.raise(keys.value(i).to_string(), h);
            }
        }
        Ok(mask)
    }
}

/// The mask's own corruption is always the ENGINE's fault: this sidecar is
/// never anything a caller supplies, only ever written by [`DeletionMask::write`]
/// in this module and read back by [`DeletionMask::read`] above.
fn schema_error(table: &str, column: &str, expected: &str, actual: &str) -> JammiError {
    JammiError::IncompatibleFormat {
        artifact: format!("{table}.{column}"),
        found: actual.to_string(),
        supported: expected.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{StorageRegistry, StorageUrl};

    #[test]
    fn horizons_mask_at_or_below_and_never_above() {
        let mask = DeletionMask::from_entries([("k".to_string(), 1), ("k".to_string(), 0)]);
        assert_eq!(mask.horizon("k"), Some(1), "raise keeps the larger horizon");
        assert!(mask.is_masked("k", 0));
        assert!(mask.is_masked("k", 1));
        assert!(!mask.is_masked("k", 2));
        assert!(!mask.is_masked("other", 0));
        assert!(mask.masks_version(1));
        assert!(!mask.masks_version(2));
    }

    /// (DIST round 8) The mask sidecar is ENGINE-owned: only [`DeletionMask::write`]
    /// in this module ever produces one, so a corrupt on-disk mask is never
    /// the caller's fault. `read` used to refuse it with `JammiError::Schema`
    /// (the caller class, gRPC `InvalidArgument`) — this asserts the fixed
    /// engine class, `IncompatibleFormat` (gRPC `Internal`).
    #[tokio::test]
    async fn a_corrupt_mask_file_is_the_engine_class_never_the_caller_s() {
        let dir = tempfile::tempdir().unwrap();
        // Wrong shape entirely: a single Utf8 column, not the two-column
        // `(_row_id Utf8, _dead_through_version Int64)` mask schema.
        let wrong_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "not_a_mask",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&wrong_schema),
            vec![Arc::new(StringArray::from(vec!["x"])) as arrow::array::ArrayRef],
        )
        .unwrap();
        let url = StorageUrl::parse(dir.path().join("corrupt.deletes.parquet").to_str().unwrap())
            .unwrap();
        let registry = StorageRegistry::new();
        let handle = JammiObjectStore::new(registry.driver_for(&url, None).unwrap(), url);
        let mut writer = ObjectParquetWriter::open(&handle, wrong_schema)
            .await
            .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.close().await.unwrap();

        let err = DeletionMask::read(&handle, "docs")
            .await
            .expect_err("a mask file missing its own columns must be refused");
        assert!(
            matches!(&err, JammiError::IncompatibleFormat { artifact, .. } if artifact.contains("docs")),
            "a corrupt ENGINE-owned sidecar must never be billed to the caller: {err:?}"
        );
    }
}
