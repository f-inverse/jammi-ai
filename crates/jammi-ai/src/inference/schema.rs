use std::sync::Arc;

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
pub fn build_output_schema(
    task: &ModelTask,
    _input_schema: &SchemaRef,
    _key_column: &str,
) -> Result<SchemaRef> {
    let mut fields = common_prefix_fields();
    let task_adapter = adapter::create_adapter_for_schema(*task);
    fields.extend(task_adapter.output_schema());
    Ok(Arc::new(Schema::new(fields)))
}
