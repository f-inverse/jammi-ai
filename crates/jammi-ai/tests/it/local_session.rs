//! The transport-agnostic [`Session`] abstraction must be a behavior-preserving
//! seam: driving the in-process [`Session`] yields the same results as
//! calling [`InferenceSession`] directly. These tests run the real
//! source → generate-embeddings → search pipeline over the patents fixture and
//! the tiny BERT cookbook model through both paths and compare.

use std::sync::Arc;

use arrow::array::{Array, StringArray};
use jammi_ai::local_session::{Modality, QueryInput, SearchQuery, SearchRequest};
use jammi_ai::session::InferenceSession;
use jammi_ai::SearchMethod;
use jammi_ai::Session;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// Register the patents fixture and generate text embeddings over `abstract`
/// using the tiny BERT model. Shared setup for both arms of every comparison.
async fn seed(session: &Arc<InferenceSession>) {
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
}

fn tiny_bert() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// `Session::generate_embeddings(Text)` produces the same result table the
/// engine's `generate_text_embeddings` does, and the flattened
/// `Session::search` returns the same hydrated rows as the builder's `.run()`.
#[tokio::test]
async fn local_session_matches_engine_for_embed_and_search() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    seed(&engine).await;

    let session = Session::new(Arc::clone(&engine));

    let record = session
        .generate_embeddings(jammi_ai::local_session::EmbeddingRequest {
            source_id: "patents".to_string(),
            model_id: tiny_bert().to_string(),
            columns: vec!["abstract".to_string()],
            key_column: "id".to_string(),
            modality: Modality::Text,
            dimensions: None,
            cache: jammi_db::store::CachePolicy::Bypass,
        })
        .await
        .unwrap()
        .0;
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);

    // Flattened search through the abstraction.
    let query = vec![0.5_f32; 32];
    let via_session = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(query.clone()),
            k: 5,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            method: SearchMethod::default(),
        })
        .await
        .unwrap();

    // Same query straight through the engine builder.
    let via_engine = engine
        .search("patents", query, 5, None, SearchMethod::default())
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!via_session.is_empty());
    assert_eq!(row_ids(&via_session), row_ids(&via_engine));
}

/// A table generated at `dimensions` serves the model's leading coordinates,
/// renormalised: each stored vector is the full-width embedding's prefix, the
/// catalog and descriptor record the served width, a query encoded at the same
/// width searches it, recompute replays at it, and a width the model does not
/// have is refused.
#[tokio::test]
async fn a_table_generated_at_a_prefix_serves_the_models_leading_coordinates() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    seed(&engine).await;
    let session = Session::new(Arc::clone(&engine));
    let request = |dimensions| jammi_ai::local_session::EmbeddingRequest {
        source_id: "patents".to_string(),
        model_id: tiny_bert(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality: Modality::Text,
        dimensions,
        cache: jammi_db::store::CachePolicy::Bypass,
    };
    let full = session.generate_embeddings(request(None)).await.unwrap().0;
    let prefix = session
        .generate_embeddings(request(Some(8)))
        .await
        .unwrap()
        .0;
    assert_eq!(
        (full.dimensions_raw(), prefix.dimensions_raw()),
        (Some(32), Some(8))
    );

    let full_vectors = common::read_table_vectors(&engine, &full).await;
    let prefix_vectors = common::read_table_vectors(&engine, &prefix).await;
    for (key, served) in &prefix_vectors {
        let expected = jammi_datafusion::matryoshka_prefix(&full_vectors[key], 8);
        let diff: f32 = served
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-4, "row {key}: {served:?} vs {expected:?}");
    }

    let model = tiny_bert();
    let query = |dimensions| {
        session.encode_query(
            &model,
            QueryInput::Text("quantum error correction".into()),
            Modality::Text,
            dimensions,
        )
    };
    let full_query = query(None).await.unwrap();
    assert_eq!(
        query(Some(8)).await.unwrap(),
        jammi_datafusion::matryoshka_prefix(&full_query, 8)
    );
    let hits = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(query(Some(8)).await.unwrap()),
            k: 3,
            embedding_table: Some(prefix.table_name.clone()),
            filter: None,
            select: Vec::new(),
            method: SearchMethod::Exact,
        })
        .await
        .unwrap();
    assert_eq!(row_ids(&hits).len(), 3);

    let replayed = session
        .recompute(
            &prefix.table_name,
            jammi_ai::pipeline::recompute::Cascade::ReportOnly,
        )
        .await
        .unwrap();
    let replay = &replayed.recomputed[0].recomputed;
    let replay = engine
        .catalog()
        .get_result_table(replay)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(replay.dimensions_raw(), Some(8));

    let refused = session
        .generate_embeddings(request(Some(33)))
        .await
        .unwrap_err();
    assert!(
        refused.to_string().contains("cannot serve 33 dimensions"),
        "{refused}"
    );
    assert!(query(Some(0)).await.is_err());
}

/// `Session::encode_query` over the text modality matches the engine's
/// `encode_text_query`, and `Session::search` by row key matches `search_by_id`.
#[tokio::test]
async fn local_session_encode_and_search_by_row_key_match_engine() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    seed(&engine).await;
    let session = Session::new(Arc::clone(&engine));

    session
        .generate_embeddings(jammi_ai::local_session::EmbeddingRequest {
            source_id: "patents".to_string(),
            model_id: tiny_bert().to_string(),
            columns: vec!["abstract".to_string()],
            key_column: "id".to_string(),
            modality: Modality::Text,
            dimensions: None,
            cache: jammi_db::store::CachePolicy::Bypass,
        })
        .await
        .unwrap();

    let via_session = session
        .encode_query(
            &tiny_bert(),
            QueryInput::Text("battery".into()),
            Modality::Text,
            None,
        )
        .await
        .unwrap();
    let via_engine = engine
        .encode_text_query(&tiny_bert(), "battery")
        .await
        .unwrap();
    assert_eq!(via_session, via_engine);

    // Pick a real row key from the source and search-by-example through both paths.
    let key_batches = engine
        .sql("SELECT id FROM patents.public.\"patents\" LIMIT 1")
        .await
        .unwrap();
    let key = key_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|a| a.value(0).to_string())
        .unwrap_or_else(|| {
            // The fixture's id column may not be Utf8; fall back to the formatted value.
            let formatter = arrow::util::display::ArrayFormatter::try_new(
                key_batches[0].column(0),
                &Default::default(),
            )
            .unwrap();
            formatter.value(0).to_string()
        });

    let via_session_key = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::RowKey(key.clone()),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            method: SearchMethod::default(),
        })
        .await
        .unwrap();
    let via_engine_key = engine
        .search_by_id("patents", &key, 3, None, SearchMethod::default())
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert_eq!(row_ids(&via_session_key), row_ids(&via_engine_key));
}

/// A modality/input mismatch on `encode_query` is a typed error, not a silent
/// wrong-tower call.
#[tokio::test]
async fn encode_query_rejects_modality_input_mismatch() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let session = Session::new(engine);

    let err = session
        .encode_query(
            "local:whatever",
            QueryInput::Bytes(vec![0, 1, 2]),
            Modality::Text,
            None,
        )
        .await
        .unwrap_err();
    assert!(
        format!("{err}").contains("requires text input"),
        "expected a modality-mismatch error, got: {err}"
    );
}

/// Collect the `_row_id` provenance column across batches into one ordered
/// vector, the stable identity of a search result set.
fn row_ids(batches: &[arrow::array::RecordBatch]) -> Vec<String> {
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch.column_by_name("_row_id").expect("_row_id present");
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_row_id is Utf8");
        for i in 0..arr.len() {
            ids.push(arr.value(i).to_string());
        }
    }
    ids
}

/// A filtered search returns the `k` nearest rows that satisfy the filter,
/// not the rows of the first `k` that happen to: six patents are from 2021,
/// and the four nearest of them come back even when they are not among the
/// four nearest overall.
#[tokio::test]
async fn a_filtered_search_returns_the_k_nearest_passing_rows() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    seed(&engine).await;
    let session = Session::new(Arc::clone(&engine));
    session
        .generate_embeddings(jammi_ai::local_session::EmbeddingRequest {
            source_id: "patents".to_string(),
            model_id: tiny_bert().to_string(),
            columns: vec!["abstract".to_string()],
            key_column: "id".to_string(),
            modality: Modality::Text,
            dimensions: None,
            cache: jammi_db::store::CachePolicy::Bypass,
        })
        .await
        .unwrap();

    let query = engine
        .encode_text_query(&tiny_bert(), "quantum error correction")
        .await
        .unwrap();
    let filtered = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(query.clone()),
            k: 4,
            embedding_table: None,
            filter: Some("year = 2021".to_string()),
            select: Vec::new(),
            method: SearchMethod::default(),
        })
        .await
        .unwrap();

    // The truth: every row ranked exactly, then filtered.
    let truth = engine
        .search("patents", query.clone(), 20, None, SearchMethod::Exact)
        .await
        .unwrap()
        .filter("year = 2021")
        .unwrap()
        .limit(4)
        .run()
        .await
        .unwrap();
    assert_eq!(row_ids(&filtered).len(), 4);
    assert_eq!(row_ids(&filtered), row_ids(&truth));

    // The first ranked breadth alone could not have answered it: the four
    // nearest rows overall are not the four nearest 2021 rows.
    let nearest = engine
        .search("patents", query, 4, None, SearchMethod::Exact)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert_ne!(row_ids(&nearest), row_ids(&filtered));
}
