use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use jammi_engine::error::Result;

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
/// For embedding tasks, `embedding_dim` must be the model's actual hidden size.
pub fn build_output_schema(
    task: &ModelTask,
    _input_schema: &SchemaRef,
    _key_column: &str,
    embedding_dim: Option<usize>,
) -> Result<SchemaRef> {
    let mut fields = common_prefix_fields();
    let task_adapter = adapter::create_adapter_for_schema(*task, embedding_dim);
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

    let errors: StringArray = row_errors
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if row_status[i] {
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
