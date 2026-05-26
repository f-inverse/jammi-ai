use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// Build the Arrow schema for an embedding result table.
///
/// Columns: `_row_id`, `_source_id`, `_model_id`, `vector` (FixedSizeList of Float32).
pub fn embedding_table_schema(dimensions: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
        Field::new("_model_id", DataType::Utf8, false),
        Field::new_fixed_size_list(
            "vector",
            Field::new("item", DataType::Float32, false),
            dimensions as i32,
            false,
        ),
    ]))
}
