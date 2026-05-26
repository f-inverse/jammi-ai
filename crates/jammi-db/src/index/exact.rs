use arrow::array::{Array, StringArray};
use datafusion::prelude::SessionContext;

use jammi_numerics::distance::cosine_distance;

use crate::error::{JammiError, Result};
use crate::store::vectors::extend_with_fixed_size_list_f32;

/// Brute-force vector search over a registered Parquet table via DataFusion.
///
/// Computes cosine distance for every row, returns the `k` closest as
/// `(row_id, cosine_distance)` sorted ascending.
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
        let row_ids = batch
            .column_by_name("_row_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| JammiError::Other("Missing _row_id in exact search".into()))?;
        let before = vectors.len();
        extend_with_fixed_size_list_f32(batch, table_name, "vector", &mut vectors)?;
        // `extend_with_fixed_size_list_f32` appends exactly one Vec<f32> per
        // row, so the new slice maps 1:1 with `row_ids`.
        for (offset, vec) in vectors[before..].iter().enumerate() {
            let dist = cosine_distance(query, vec);
            scored.push((row_ids.value(offset).to_string(), dist));
        }
    }

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}
