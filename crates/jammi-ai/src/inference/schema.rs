use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use jammi_db::error::Result;

use super::adapter;
use crate::model::ModelTask;

/// Common prefix columns on every inference output.
pub fn common_prefix_fields() -> Vec<Field> {
    vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source", DataType::Utf8, false),
        Field::new("_model", DataType::Utf8, false),
        Field::new("_status", DataType::Utf8, false),
        Field::new("_error", DataType::Utf8, true),
        Field::new("_latency_ms", DataType::Float32, false),
    ]
}

/// Build the full output schema: prefix + task-specific.
///
/// For embedding tasks, `embedding_dim` must be the model's actual hidden size.
/// For regression tasks, `regression_form` must be the served head's persisted
/// [`DistributionForm`](adapter::DistributionForm) so the planned schema matches
/// the runtime adapter's columns (a quantile head's schema is its level
/// columns, not the Gaussian default of `mean`/`std`). `None` form ⇒ Gaussian.
pub fn build_output_schema(
    task: &ModelTask,
    _input_schema: &SchemaRef,
    _key_column: &str,
    embedding_dim: Option<usize>,
    regression_form: Option<&adapter::DistributionForm>,
) -> Result<SchemaRef> {
    let mut fields = common_prefix_fields();
    let task_adapter = adapter::create_adapter_for_schema(*task, embedding_dim, regression_form);
    fields.extend(task_adapter.output_schema());
    Ok(Arc::new(Schema::new(fields)))
}

/// Build common prefix arrays for an output batch.
pub fn build_prefix_columns(
    keys: &ArrayRef,
    source_id: &str,
    model_id: &str,
    row_status: &[bool],
    row_errors: &[String],
    latency_ms: f32,
    row_count: usize,
) -> Vec<ArrayRef> {
    let status_strs: Vec<&str> = row_status
        .iter()
        .map(|&ok| if ok { "ok" } else { "error" })
        .collect();
    let status = StringArray::from(status_strs);

    // `row_status.get(i)` (never a raw `row_status[i]`): `row_errors` and
    // `row_status` are two independently-sized fields on `BackendOutput`, so
    // a producer that emits more `row_errors` than `row_status` entries must
    // not panic here — a missing status is treated as "not ok" (the error
    // message is still surfaced) rather than indexing out of bounds.
    let errors: StringArray = row_errors
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if row_status.get(i).copied().unwrap_or(false) {
                None
            } else {
                Some(e.as_str())
            }
        })
        .collect();

    // Cast keys to Utf8 if needed (key column may be Int64, etc.)
    let row_ids: ArrayRef = if keys.data_type() == &DataType::Utf8 {
        Arc::clone(keys)
    } else {
        compute::cast(keys, &DataType::Utf8).unwrap_or_else(|_| Arc::clone(keys))
    };

    vec![
        row_ids,                                                               // _row_id
        Arc::new(StringArray::from(vec![source_id; row_count])) as ArrayRef,   // _source
        Arc::new(StringArray::from(vec![model_id; row_count])) as ArrayRef,    // _model
        Arc::new(status) as ArrayRef,                                          // _status
        Arc::new(errors) as ArrayRef,                                          // _error
        Arc::new(Float32Array::from(vec![latency_ms; row_count])) as ArrayRef, // _latency_ms
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    /// `row_errors` longer than `row_status` (two independently-sized fields
    /// on `BackendOutput`) must never panic indexing `row_status[i]` at the
    /// tail entries — the missing status is treated as "not ok". Verified by
    /// temporarily reverting `row_status.get(i).copied().unwrap_or(false)` to
    /// the raw `row_status[i]`: this test goes RED (an out-of-bounds index
    /// panic instead of returning the array below).
    #[test]
    fn build_prefix_columns_never_panics_when_row_errors_is_longer_than_row_status() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true]; // one entry; row_errors has three
        let row_errors = vec![
            String::new(),
            "row 1 failed".to_string(),
            "row 2 failed".to_string(),
        ];
        let cols = build_prefix_columns(&keys, "src", "model", &row_status, &row_errors, 1.0, 3);
        let errors = cols[4].as_any().downcast_ref::<StringArray>().unwrap();
        assert!(errors.is_null(0), "row 0's own recorded status was ok");
        assert_eq!(errors.value(1), "row 1 failed");
        assert_eq!(errors.value(2), "row 2 failed");
    }
}
