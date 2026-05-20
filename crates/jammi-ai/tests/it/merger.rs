//! SQL-driven null-tolerance test for the data-driven provenance merger.
//!
//! Verifies that channel-declared columns surface in DataFusion query
//! results with the expected nullability — and stay NULL for rows
//! produced by a channel that did not supply a contribution. This is the
//! end-to-end shape of the engine contract that the channel mechanism
//! preserves null semantics across the SQL surface.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_engine::catalog::Catalog;
use jammi_engine::ChannelId;
use tempfile::tempdir;

fn open_catalog() -> (tempfile::TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    (dir, catalog)
}

fn source_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
    ]));
    let row_id = Arc::new(StringArray::from(
        (0..n).map(|i| format!("r{i}")).collect::<Vec<_>>(),
    )) as ArrayRef;
    let src = Arc::new(StringArray::from(vec!["src"; n])) as ArrayRef;
    RecordBatch::try_new(schema, vec![row_id, src]).unwrap()
}

/// SQL `IS NULL` filter over an unsupplied channel column matches every
/// row. SQL `IS NOT NULL` filter over a supplied channel column matches
/// every row. This proves the null/non-null bits propagate end-to-end.
#[tokio::test]
async fn merged_channel_nullability_round_trips_through_datafusion() {
    let (_dir, catalog) = open_catalog();
    let vector = ChannelId::new("vector").unwrap();
    let inference = ChannelId::new("inference").unwrap();

    let n = 5;
    let batch = source_batch(n);

    let vector_contrib = ChannelContribution::single(
        vector.clone(),
        Arc::new(Float32Array::from(vec![0.5_f32; n])) as ArrayRef,
    );
    let merged = merge_channels(
        &catalog,
        &[batch],
        &[vector.clone(), inference.clone()],
        &[vector],
        &[inference],
        &[vec![vector_contrib]],
    )
    .unwrap();
    assert_eq!(merged.len(), 1);

    let ctx = SessionContext::new();
    let schema = merged[0].schema();
    ctx.register_batch("results", merged.into_iter().next().unwrap())
        .unwrap();

    // Sanity: the union of declared columns is present in the surface.
    for name in [
        "similarity",
        "inference_model",
        "inference_task",
        "inference_confidence",
    ] {
        assert!(
            schema.field_with_name(name).is_ok(),
            "expected '{name}' field in the merged result"
        );
    }

    // Every row has the supplied vector column.
    let supplied_rows = ctx
        .sql("SELECT COUNT(*) FROM results WHERE similarity IS NOT NULL")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count_supplied = supplied_rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count_supplied, n as i64);

    // No row has any inference column (the channel contributed nothing).
    for col in ["inference_model", "inference_task", "inference_confidence"] {
        let q = format!("SELECT COUNT(*) FROM results WHERE {col} IS NULL");
        let r = ctx.sql(&q).await.unwrap().collect().await.unwrap();
        let count = r[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(
            count, n as i64,
            "expected every row of '{col}' to be NULL when the channel supplied no contribution"
        );
    }
}
