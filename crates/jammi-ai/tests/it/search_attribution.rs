//! Third-tenant integration test: SPEC-01 §8 search-attribution chain.
//!
//! Demonstrates the data-driven provenance channel mechanism for a
//! consumer building a multi-stage retrieval pipeline — a use case
//! unrelated to either flagship plan-group tenant.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, StringArray};
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_ai::session::InferenceSession;
use jammi_engine::catalog::channel_repo::ChannelColumnType;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use jammi_engine::ChannelId;
use jammi_test_utils::register_test_channel;
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "patents",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    (session, dir)
}

#[tokio::test]
async fn third_party_scored_by_channel_columns_appear_in_search_results() {
    let (session, _dir) = session_with_embeddings().await;

    // Register the third-tenant channel before the search runs.
    register_test_channel(
        session.catalog(),
        "scored_by",
        3,
        &[
            ("ranker", ChannelColumnType::Utf8),
            ("rank_score", ChannelColumnType::Float32),
        ],
    )
    .await
    .unwrap();

    // Run a vector search; this produces batches with vector's `similarity`
    // already merged in by the engine.
    let query = vec![0.5_f32; 32];
    let batches = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!batches.is_empty());

    // Build a `scored_by` contribution for each batch. A real reranker
    // would consult an external ranking service; the test synthesises a
    // deterministic score derived from row index.
    let vector = ChannelId::new("vector").unwrap();
    let scored_by = ChannelId::new("scored_by").unwrap();
    let mut contribs: Vec<Vec<ChannelContribution>> = Vec::with_capacity(batches.len());
    for batch in &batches {
        let n = batch.num_rows();
        let ranker: ArrayRef = Arc::new(StringArray::from(vec!["bm25"; n]));
        let rank_score: ArrayRef = Arc::new(Float32Array::from(
            (0..n).map(|i| 1.0 - (i as f32 * 0.1)).collect::<Vec<_>>(),
        ));
        contribs.push(vec![ChannelContribution {
            channel: scored_by.clone(),
            columns: vec![ranker, rank_score],
        }]);
    }

    // Merge again with the third channel participating.
    let merged = merge_channels(
        session.catalog(),
        &batches,
        &[vector.clone(), scored_by.clone()],
        &[vector, scored_by],
        &[],
        &contribs,
    )
    .await
    .unwrap();

    // The catalog-declared third-tenant columns are present in the
    // merged output and carry the supplied values.
    let m = &merged[0];
    let schema = m.schema();
    assert!(schema.field_with_name("ranker").is_ok());
    assert!(schema.field_with_name("rank_score").is_ok());

    let ranker_idx = schema
        .fields()
        .iter()
        .rposition(|f| f.name() == "ranker")
        .unwrap();
    let ranker = m
        .column(ranker_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(ranker.value(0), "bm25");

    let rank_score_idx = schema
        .fields()
        .iter()
        .rposition(|f| f.name() == "rank_score")
        .unwrap();
    let rank_score = m
        .column(rank_score_idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert!((rank_score.value(0) - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn rows_not_touched_by_channel_have_null_in_that_channels_columns() {
    let (session, _dir) = session_with_embeddings().await;

    register_test_channel(
        session.catalog(),
        "scored_by",
        3,
        &[
            ("ranker", ChannelColumnType::Utf8),
            ("rank_score", ChannelColumnType::Float32),
        ],
    )
    .await
    .unwrap();

    let query = vec![0.5_f32; 32];
    let batches = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    // Participate in scored_by but supply no contribution for it. Every
    // row should have NULL in both ranker and rank_score.
    let vector = ChannelId::new("vector").unwrap();
    let scored_by = ChannelId::new("scored_by").unwrap();
    let empty_contribs: Vec<Vec<ChannelContribution>> =
        batches.iter().map(|_| Vec::new()).collect();

    let merged = merge_channels(
        session.catalog(),
        &batches,
        &[vector.clone(), scored_by],
        &[vector],
        &[],
        &empty_contribs,
    )
    .await
    .unwrap();

    let m = &merged[0];
    let ranker = m
        .column_by_name("ranker")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let rank_score = m
        .column_by_name("rank_score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(ranker.null_count(), m.num_rows());
    assert_eq!(rank_score.null_count(), m.num_rows());
}

#[tokio::test]
async fn channel_contribution_arrow_dtypes_must_match_catalog_declaration() {
    let (session, _dir) = session_with_embeddings().await;

    register_test_channel(
        session.catalog(),
        "scored_by",
        3,
        &[("rank_score", ChannelColumnType::Float32)],
    )
    .await
    .unwrap();

    let query = vec![0.5_f32; 32];
    let batches = session
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    let vector = ChannelId::new("vector").unwrap();
    let scored_by = ChannelId::new("scored_by").unwrap();
    // Wrong dtype — declared Float32, supplying Int32.
    let wrong: Vec<Vec<ChannelContribution>> = batches
        .iter()
        .map(|b| {
            let n = b.num_rows();
            vec![ChannelContribution::single(
                scored_by.clone(),
                Arc::new(arrow::array::Int32Array::from(vec![0_i32; n])) as ArrayRef,
            )]
        })
        .collect();

    let err = merge_channels(
        session.catalog(),
        &batches,
        &[vector.clone(), scored_by],
        &[vector],
        &[],
        &wrong,
    )
    .await
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("dtype"), "expected dtype mismatch, got: {msg}");
}
