pub mod adapter;
pub mod observer;
pub mod runner;
pub mod schema;

use arrow::array::{Array, ArrayRef, StringArray};
use jammi_engine::error::{JammiError, Result};

/// Extract text from Arrow StringArray columns.
/// If multiple columns, concatenate with " " separator.
/// Null values produce empty strings (caller handles null tracking).
pub fn arrow_to_texts(columns: &[ArrayRef]) -> Result<Vec<String>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No content columns provided".into()));
    }
    let row_count = columns[0].len();
    let mut texts = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let parts: Vec<&str> = columns
            .iter()
            .filter_map(|col| {
                col.as_any().downcast_ref::<StringArray>().and_then(|arr| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
            })
            .collect();
        texts.push(parts.join(" "));
    }
    Ok(texts)
}

/// Extract named columns from a RecordBatch as ArrayRefs.
pub fn extract_columns(
    batch: &arrow::record_batch::RecordBatch,
    column_names: &[String],
) -> Result<Vec<ArrayRef>> {
    column_names
        .iter()
        .map(|name| {
            batch
                .column_by_name(name)
                .map(std::sync::Arc::clone)
                .ok_or_else(|| {
                    JammiError::Inference(format!("Column '{name}' not found in input batch"))
                })
        })
        .collect()
}

/// Extract a single named column from a RecordBatch.
pub fn extract_column(
    batch: &arrow::record_batch::RecordBatch,
    column_name: &str,
) -> Result<ArrayRef> {
    batch
        .column_by_name(column_name)
        .map(std::sync::Arc::clone)
        .ok_or_else(|| {
            JammiError::Inference(format!("Column '{column_name}' not found in input batch"))
        })
}

/// Slice a set of columns to a sub-range.
pub fn slice_columns(columns: &[ArrayRef], offset: usize, length: usize) -> Vec<ArrayRef> {
    columns
        .iter()
        .map(|col| col.slice(offset, length))
        .collect()
}
