use arrow::array::{Array, Float32Array, StringArray};
use datafusion::prelude::SessionContext;

use crate::error::{JammiError, Result};
use crate::index::cosine_distance;

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
    for batch in &batches {
        let row_ids = batch
            .column_by_name("_row_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| JammiError::Other("Missing _row_id in exact search".into()))?;

        let vectors = batch
            .column_by_name("vector")
            .and_then(|c| {
                c.as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()
            })
            .ok_or_else(|| JammiError::Other("Missing vector in exact search".into()))?;

        for i in 0..row_ids.len() {
            let row_id = row_ids.value(i).to_string();
            let v = vectors.value(i);
            let float_arr = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| JammiError::Other("Vector not Float32".into()))?;
            let vec: Vec<f32> = (0..float_arr.len()).map(|j| float_arr.value(j)).collect();
            let dist = cosine_distance(query, &vec);
            scored.push((row_id, dist));
        }
    }

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}
