//! Reading the columns an operator names off a batch.

use arrow::array::{ArrayRef, RecordBatch};

use crate::error::{Error, Result};

/// The named columns of `batch`, in the order named.
pub fn extract_columns(batch: &RecordBatch, column_names: &[String]) -> Result<Vec<ArrayRef>> {
    column_names
        .iter()
        .map(|name| extract_column(batch, name))
        .collect()
}

/// One named column of `batch`.
pub fn extract_column(batch: &RecordBatch, column_name: &str) -> Result<ArrayRef> {
    batch
        .column_by_name(column_name)
        .map(std::sync::Arc::clone)
        .ok_or_else(|| Error::Inference(format!("Column '{column_name}' not found in input batch")))
}

/// Every column of `columns` sliced to the same sub-range.
pub fn slice_columns(columns: &[ArrayRef], offset: usize, length: usize) -> Vec<ArrayRef> {
    columns
        .iter()
        .map(|col| col.slice(offset, length))
        .collect()
}
