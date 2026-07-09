//! Typed reads of `FixedSizeList<Float32>` vector columns from a result-table
//! Parquet object. Centralises the downcast-and-collect logic that both the
//! brute-force search path and downstream callers (e.g. resilience checks
//! that need the raw vectors) would otherwise duplicate.

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::DataType;

use crate::error::{JammiError, Result};
use crate::storage::{self, JammiObjectStore};

/// Materialise a `FixedSizeList<Float32>` column from one `RecordBatch` into
/// `Vec<f32>` rows, appending them to `out`.
///
/// Returns a typed [`JammiError::Schema`] when the column is missing, has the
/// wrong Arrow type, or has a non-`Float32` inner item. The `table` argument
/// is folded into the error so the caller does not need to wrap.
///
/// Hidden invariant: this helper is the only place in the engine that should
/// downcast a vector column to `FixedSizeListArray<Float32>`. The brute-force
/// ANN scan, the typed-read API, and the neighbor-graph node reader all call
/// through here rather than re-implementing the downcast.
pub fn extend_with_fixed_size_list_f32(
    batch: &RecordBatch,
    table: &str,
    column: &str,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: "missing".to_string(),
        })?;
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: format!("{:?}", col.data_type()),
        })?;
    if !matches!(list.value_type(), DataType::Float32) {
        return Err(JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: format!("FixedSizeList<{:?}>", list.value_type()),
        });
    }
    let dim = list.value_length() as usize;
    for row in 0..list.len() {
        let v = list.value(row);
        let floats =
            v.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| JammiError::Schema {
                    table: table.to_string(),
                    column: column.to_string(),
                    expected: "FixedSizeList<Float32>".to_string(),
                    actual: format!("FixedSizeList<{:?}>", v.data_type()),
                })?;
        let mut row_vec = Vec::with_capacity(dim);
        for i in 0..dim {
            row_vec.push(floats.value(i));
        }
        out.push(row_vec);
    }
    Ok(())
}

/// Materialise the paired `(key, vector)` rows of one `RecordBatch` — a Utf8
/// `key_column` alongside a `FixedSizeList<Float32>` `vector_column` — appending
/// each `(String, Vec<f32>)` to `out` in row order.
///
/// Returns a typed [`JammiError::Schema`] when either column is missing or has
/// the wrong Arrow type (a non-Utf8 key or a non-`Float32` vector), so a caller
/// reading precomputed vectors sees a typed signal rather than a downcast panic.
/// The vector leg delegates to [`extend_with_fixed_size_list_f32`] so the
/// downcast rules stay defined in exactly one place; this pairs each resulting
/// vector with its key by position.
pub fn extend_with_keyed_fixed_size_list_f32(
    batch: &RecordBatch,
    table: &str,
    key_column: &str,
    vector_column: &str,
    out: &mut Vec<(String, Vec<f32>)>,
) -> Result<()> {
    let keys = batch
        .column_by_name(key_column)
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: "missing".to_string(),
        })?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: format!(
                "{:?}",
                batch.column_by_name(key_column).unwrap().data_type()
            ),
        })?;

    let mut vectors = Vec::with_capacity(keys.len());
    extend_with_fixed_size_list_f32(batch, table, vector_column, &mut vectors)?;
    if vectors.len() != keys.len() {
        return Err(JammiError::Schema {
            table: table.to_string(),
            column: vector_column.to_string(),
            expected: format!("{} vectors (one per key)", keys.len()),
            actual: format!("{} vectors", vectors.len()),
        });
    }

    for (row, vector) in vectors.into_iter().enumerate() {
        out.push((keys.value(row).to_string(), vector));
    }
    Ok(())
}

/// Read every `(key, vector)` pair — a Utf8 `key_column` and a
/// `FixedSizeList<Float32>` `vector_column` — from the Parquet object behind
/// `handle`, in file order.
///
/// The read path behind importing precomputed vectors: streams batches through
/// the engine's `storage::reader` and delegates each to
/// [`extend_with_keyed_fixed_size_list_f32`]. Reads the whole object into
/// memory.
pub async fn read_keyed_vectors_f32(
    handle: &JammiObjectStore,
    table: &str,
    key_column: &str,
    vector_column: &str,
) -> Result<Vec<(String, Vec<f32>)>> {
    let batches = storage::reader::read_all_record_batches(handle).await?;
    let mut out = Vec::new();
    for batch in batches {
        extend_with_keyed_fixed_size_list_f32(&batch, table, key_column, vector_column, &mut out)?;
    }
    Ok(out)
}

/// Read every value of a `FixedSizeList<Float32>` column from the Parquet
/// object behind `handle`, returning one `Vec<f32>` per row.
///
/// Streams batches through the engine's `storage::reader` and delegates each
/// to [`extend_with_fixed_size_list_f32`].
pub(crate) async fn read_fixed_size_list_f32_column(
    handle: &JammiObjectStore,
    table: &str,
    column: &str,
) -> Result<Vec<Vec<f32>>> {
    let batches = storage::reader::read_all_record_batches(handle).await?;
    let mut out = Vec::new();
    for batch in batches {
        extend_with_fixed_size_list_f32(&batch, table, column, &mut out)?;
    }
    Ok(out)
}
