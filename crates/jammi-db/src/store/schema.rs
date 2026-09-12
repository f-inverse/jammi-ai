use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::error::{JammiError, Result};

/// The name of the nullable per-row content-hash column every embedding
/// table carries as its fifth column.
pub const CONTENT_HASH_COLUMN: &str = "_content_hash";

/// Build the Arrow schema for an embedding result table.
///
/// Columns: `_row_id`, `_source_id`, `_model_id`, `vector` (FixedSizeList of
/// Float32), `_content_hash` (nullable Utf8 — the hex SHA-256 the embedding
/// pipeline computes over the embedded source columns, `NULL` on every table
/// built by hand or by a producer that does not read a source row; see
/// [`crate::store::content_hash`]).
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
        Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
    ]))
}

/// Build the `(_row_id, _source_id, _model_id, vector, _content_hash)` batch
/// for an embedding table from per-key vectors, with a NULL `_content_hash`
/// in every row — the one builder every hand-built embedding batch (a
/// materialised context set, a propagation, a test fixture, a bench corpus)
/// routes through. Only the embedding pipeline writes real hashes.
pub fn embedding_batch_with_null_hash(
    schema: &SchemaRef,
    source_id: &str,
    model_id: &str,
    rows: &[(String, Vec<f32>)],
    dimensions: usize,
) -> Result<RecordBatch> {
    for (key, vector) in rows {
        if vector.len() != dimensions {
            return Err(JammiError::Schema {
                table: model_id.to_string(),
                column: "vector".into(),
                expected: format!("FixedSizeList<Float32> width {dimensions}"),
                actual: format!("row '{key}' has width {}", vector.len()),
            });
        }
    }

    let row_ids = StringArray::from_iter_values(rows.iter().map(|(k, _)| k.as_str()));
    let source_ids = StringArray::from_iter_values(rows.iter().map(|_| source_id));
    let model_ids = StringArray::from_iter_values(rows.iter().map(|_| model_id));
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors = FixedSizeListArray::try_new(
        item,
        dimensions as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .map_err(|e| JammiError::Other(format!("materialize: build vector column: {e}")))?;

    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(row_ids),
            Arc::new(source_ids),
            Arc::new(model_ids),
            Arc::new(vectors),
            crate::store::content_hash::null_hash_column(rows.len()),
        ],
    )
    .map_err(|e| JammiError::Other(format!("materialize: build batch: {e}")))
}
