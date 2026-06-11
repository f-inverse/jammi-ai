use arrow::array::{Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;

use jammi_numerics::distance::cosine_distance;

use crate::error::{JammiError, Result};
use crate::store::vectors::extend_with_fixed_size_list_f32;

/// Brute-force vector search over a registered Parquet table via DataFusion.
///
/// Computes cosine distance for every row, returns the `k` closest as
/// `(row_id, cosine_distance)` sorted by ascending distance, with ties broken
/// by ascending `_row_id` so equidistant candidates resolve deterministically
/// regardless of scan order.
pub async fn exact_vector_search(
    ctx: &SessionContext,
    table_name: &str,
    query: &[f32],
    k: usize,
) -> Result<Vec<(String, f32)>> {
    let df = ctx
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await?;
    let batches = df.collect().await?;

    let mut scored: Vec<(String, f32)> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for batch in &batches {
        // `_row_id` is a Utf8 column, but the parquet reader surfaces it as
        // `Utf8View` (`StringViewArray`) under DataFusion's default
        // `schema_force_view_types`, and could be `LargeUtf8` for a wide table.
        // Cast to `Utf8` so a single `StringArray` downcast covers every Utf8
        // family the scan can produce.
        let row_ids_col = batch
            .column_by_name("_row_id")
            .ok_or_else(|| JammiError::Other("Missing _row_id in exact search".into()))?;
        let row_ids_utf8 = cast(row_ids_col, &DataType::Utf8).map_err(|e| {
            JammiError::Other(format!("_row_id column could not be cast to Utf8: {e}"))
        })?;
        let row_ids = row_ids_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                JammiError::Other("_row_id column is not a Utf8-castable string type".into())
            })?;
        let before = vectors.len();
        extend_with_fixed_size_list_f32(batch, table_name, "vector", &mut vectors)?;
        // `extend_with_fixed_size_list_f32` appends exactly one Vec<f32> per
        // row, so the new slice maps 1:1 with `row_ids`.
        for (offset, vec) in vectors[before..].iter().enumerate() {
            let dist = cosine_distance(query, vec);
            scored.push((row_ids.value(offset).to_string(), dist));
        }
    }

    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    Ok(scored)
}
