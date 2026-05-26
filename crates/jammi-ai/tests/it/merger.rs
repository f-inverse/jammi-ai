//! SQL-driven null-tolerance test for the data-driven provenance merger.
//!
//! Verifies that channel-declared columns surface in DataFusion query
//! results with the expected nullability — and stay NULL for rows
//! produced by a channel that did not supply a contribution. This is the
//! end-to-end shape of the engine contract that the channel mechanism
//! preserves null semantics across the SQL surface.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::ChannelId;
use tempfile::tempdir;

async fn open_catalog() -> (tempfile::TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
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
    let (_dir, catalog) = open_catalog().await;
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
    .await
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

/// SPEC-01 §9 — a contribution whose array length doesn't match the batch's
/// row count must surface as a typed `EvidenceChannel` error naming both
/// counts. SPEC-01 §3.7 phrases this as "has N rows, batch has M"; the
/// production wording is "has N rows, expected M" (substring match against
/// "has 2 rows" + "expected 3" + the column name pins both sides of the
/// inequality without depending on the exact connective).
#[tokio::test]
async fn length_mismatched_contribution_returns_typed_error() {
    let (_dir, catalog) = open_catalog().await;
    let vector = ChannelId::new("vector").unwrap();
    let batch = source_batch(3); // 3 rows

    let bad = ChannelContribution::single(
        vector.clone(),
        Arc::new(Float32Array::from(vec![0.5_f32, 0.4])) as ArrayRef, // only 2 rows
    );

    let err = merge_channels(
        &catalog,
        &[batch],
        &[vector.clone()],
        &[vector],
        &[],
        &[vec![bad]],
    )
    .await
    .unwrap_err();

    match err {
        JammiError::EvidenceChannel(msg) => assert!(
            msg.contains("has 2 rows")
                && msg.contains("expected 3")
                && msg.contains("similarity"),
            "expected length-mismatch error naming the column 'similarity' and row counts; got: {msg}"
        ),
        other => panic!("expected JammiError::EvidenceChannel, got {other:?}"),
    }
}

/// SPEC-01 §9 — a contribution whose Arrow dtype doesn't match the
/// catalog-declared `ChannelColumnType` must surface as a typed
/// `EvidenceChannel` error naming the column and both types.
#[tokio::test]
async fn wrong_dtype_contribution_returns_typed_error() {
    let (_dir, catalog) = open_catalog().await;
    let vector = ChannelId::new("vector").unwrap(); // 'similarity' declared Float32
    let batch = source_batch(2);

    let bad = ChannelContribution::single(
        vector.clone(),
        Arc::new(Int32Array::from(vec![1_i32, 2])) as ArrayRef, // wrong: Int32
    );

    let err = merge_channels(
        &catalog,
        &[batch],
        &[vector.clone()],
        &[vector],
        &[],
        &[vec![bad]],
    )
    .await
    .unwrap_err();

    match err {
        JammiError::EvidenceChannel(msg) => assert!(
            msg.contains("Float32") && msg.contains("Int32") && msg.contains("similarity"),
            "expected dtype-mismatch error naming 'similarity' + Int32 + Float32; got: {msg}"
        ),
        other => panic!("expected JammiError::EvidenceChannel, got {other:?}"),
    }
}
