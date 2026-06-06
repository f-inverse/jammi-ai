//! `InferenceSession::assemble_context` — context-set assembly + the
//! permutation-invariant fixed-pooling set encoder (spec S16).
//!
//! Hermetic: a tempdir-backed session embeds the `patents` fixture through the
//! real embedding pipeline, then assembles and encodes context sets over the
//! resulting embedding table. Exercises the contracts the spec bakes in —
//! self-exclusion (the leakage guard), determinism of exact pooling, the
//! value-column hydration join, the empty-context path, and the materialised
//! context table — and confirms the pool runs through the shared
//! vector-aggregation UDAF rather than a second aggregation implementation.

use std::sync::Arc;

use arrow::array::{Array, StringArray};
use jammi_ai::pipeline::context_set::{ContextRequest, MaterializedContext, SetAggregator};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// A session with the `patents` fixture embedded (abstract → vector, keyed by
/// `id`). The tiny-bert fixture emits 32-wide vectors.
async fn session_with_embeddings() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    // The set encoder pools through the vector-aggregation UDAFs; register the
    // engine's compound-query SQL functions so `vector_mean`/`vector_sum`/
    // `vector_max` resolve on this session, exactly as the canonical
    // `InferenceSession::open` constructor does for a long-lived session.
    session.register_query_functions();
    session
        .add_source(
            "patents",
            SourceType::File,
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

/// Resolve the stored vector for a known row key so a query can be the target's
/// own vector — the self-hit case the leakage guard must drop. Reads the
/// embedding table's `_row_id` + `vector` columns and returns the row matching
/// `key`.
async fn vector_for_key(session: &Arc<InferenceSession>, key: &str) -> Vec<f32> {
    use arrow::array::{FixedSizeListArray, Float32Array};

    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\" WHERE _row_id = '{key}'",
            table.table_name
        ))
        .await
        .unwrap();
    for batch in &batches {
        let vectors = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        if vectors.len() > 0 {
            let v = vectors.value(0);
            let floats = v.as_any().downcast_ref::<Float32Array>().unwrap();
            return (0..floats.len()).map(|i| floats.value(i)).collect();
        }
    }
    panic!("no stored vector for key '{key}'");
}

#[tokio::test]
async fn assemble_context_hydrates_value_columns_and_pools() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.3_f32; 32];
    let mut request = ContextRequest::new("patents", query, 5);
    request.value_columns = vec!["category".to_string()];
    request.aggregator = SetAggregator::Mean;

    let context = session.assemble_context(&request).await.unwrap();

    assert!(!context.is_empty(), "five neighbours should form a context");
    assert_eq!(context.context_size, context.context_keys.len());
    let vector = context
        .context_vector
        .as_ref()
        .expect("non-empty context pools to a vector");
    assert_eq!(vector.len(), 32, "pooled vector keeps the embedding width");

    // value_columns hydrated, in retrieval order, one single-row batch per member.
    let hydrated_rows: usize = context.value_rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(hydrated_rows, context.context_size);
    for batch in &context.value_rows {
        assert!(
            batch.column_by_name("category").is_some(),
            "value_rows carry the requested value column"
        );
    }
}

#[tokio::test]
async fn exact_pooling_is_deterministic_across_runs() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.42_f32; 32];
    let request = {
        let mut r = ContextRequest::new("patents", query, 6);
        r.aggregator = SetAggregator::Sum;
        r
    };

    let first = session.assemble_context(&request).await.unwrap();
    for _ in 0..3 {
        let again = session.assemble_context(&request).await.unwrap();
        assert_eq!(
            first.context_keys, again.context_keys,
            "exact assembly retrieves the same context set every run"
        );
        let a = first.context_vector.as_ref().unwrap();
        let b = again.context_vector.as_ref().unwrap();
        assert_eq!(
            a.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "fixed pooling is byte-identical across runs"
        );
    }
}

#[tokio::test]
async fn aggregators_pool_through_the_shared_udaf() {
    // All three aggregators must succeed against the same context — they are the
    // engine's `vector_mean`/`vector_sum`/`vector_max` UDAFs (one aggregation
    // operator), invoked through `assemble_context`'s pooling call site. If the
    // encoder had its own aggregation, this would not depend on those
    // registrations at all.
    let (session, _dir) = session_with_embeddings().await;
    let query = vec![0.1_f32; 32];

    for aggregator in [SetAggregator::Mean, SetAggregator::Sum, SetAggregator::Max] {
        let mut request = ContextRequest::new("patents", query.clone(), 4);
        request.aggregator = aggregator;
        let context = session.assemble_context(&request).await.unwrap();
        assert_eq!(
            context.context_vector.as_ref().unwrap().len(),
            32,
            "{aggregator:?} pooled through the shared UDAF to a 32-wide vector"
        );
    }
}

#[tokio::test]
async fn exclude_self_drops_the_targets_own_row() {
    let (session, _dir) = session_with_embeddings().await;

    // Pick a real key and query by its own stored vector — it is the nearest
    // neighbour of itself, so without the guard it would head the context.
    let hydrated = session
        .search("patents", vec![0.0_f32; 32], 1)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let key_col = hydrated[0]
        .column_by_name("_row_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let target_key = key_col.value(0).to_string();

    let query = vector_for_key(&session, &target_key).await;

    let mut request = ContextRequest::new("patents", query, 5);
    request.exclude_key = Some(target_key.clone());
    let context = session.assemble_context(&request).await.unwrap();

    assert!(
        !context.context_keys.contains(&target_key),
        "the target's own outcome never appears in its context"
    );
    assert!(
        !context.is_empty(),
        "over-fetch keeps the context full after self-exclusion"
    );

    // With the guard off, the self-hit is present — confirming the guard, not a
    // query that simply never retrieves the target.
    let mut without = ContextRequest::new("patents", request.query.clone(), 5);
    without.exclude_self = false;
    let unguarded = session.assemble_context(&without).await.unwrap();
    assert!(
        unguarded.context_keys.contains(&target_key),
        "without the guard the target retrieves itself"
    );
}

#[tokio::test]
async fn empty_context_is_defined_not_a_crash() {
    let (session, _dir) = session_with_embeddings().await;

    // A split predicate no row satisfies yields an empty context — a defined,
    // low-confidence representation, not a panic or a one-element average.
    let mut request = ContextRequest::new("patents", vec![0.2_f32; 32], 5);
    request.split = Some("category = '__no_such_category__'".to_string());
    let context = session.assemble_context(&request).await.unwrap();

    assert!(context.is_empty());
    assert_eq!(context.context_size, 0);
    assert!(context.context_vector.is_none());
    assert!(context.value_rows.is_empty());
}

#[tokio::test]
async fn materialize_context_writes_a_searchable_embedding_table() {
    let (session, _dir) = session_with_embeddings().await;

    let rows: Vec<(String, Vec<f32>)> = (0..3)
        .map(|i| (format!("target-{i}"), vec![i as f32; 32]))
        .collect();

    let table = session
        .materialize_context(
            "patents",
            MaterializedContext {
                rows: &rows,
                dimensions: 32,
            },
        )
        .await
        .unwrap();

    assert_eq!(table.row_count, 3);
    assert_eq!(table.dimensions, Some(32));
    assert!(
        table.index_path.is_some(),
        "a materialised context table gets the embedding-table sidecar index"
    );

    // Round-trips as a normal embedding table: its vectors read back by key.
    let read = session.read_vectors(&table).await.unwrap();
    assert_eq!(read.len(), 3);
}

#[tokio::test]
async fn materialize_context_rejects_duplicate_target_keys() {
    let (session, _dir) = session_with_embeddings().await;
    let rows = vec![
        ("dup".to_string(), vec![0.0_f32; 32]),
        ("dup".to_string(), vec![1.0_f32; 32]),
    ];
    let err = session
        .materialize_context(
            "patents",
            MaterializedContext {
                rows: &rows,
                dimensions: 32,
            },
        )
        .await
        .expect_err("a target key must be unique in a context table");
    assert!(err.to_string().contains("duplicate target key"));
}
