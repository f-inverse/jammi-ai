//! Typed reads of `FixedSizeList<Float32>` vector columns from a result-table
//! Parquet object. Centralises the downcast-and-collect logic that both the
//! brute-force search path and downstream callers (e.g. resilience checks
//! that need the raw vectors) would otherwise duplicate.

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
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
/// downcast a vector column to `FixedSizeListArray<Float32>`. Both the
/// brute-force ANN scan and the typed-read API call through here.
pub(crate) fn extend_with_fixed_size_list_f32(
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
