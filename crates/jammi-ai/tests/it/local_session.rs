//! The transport-agnostic [`Session`] abstraction must be a behavior-preserving
//! seam: driving the in-process [`LocalSession`] yields the same results as
//! calling [`InferenceSession`] directly. These tests run the real
//! source → generate-embeddings → search pipeline over the patents fixture and
//! the tiny BERT cookbook model through both paths and compare.

use std::sync::Arc;

use arrow::array::{Array, StringArray};
use jammi_ai::local_session::{LocalSession, Modality, QueryInput, SearchQuery, SearchRequest};
use jammi_ai::session::InferenceSession;
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

    let session = Session::Local(LocalSession::new(Arc::clone(&engine)));

    let record = session
        .generate_embeddings(
            "patents",
            &tiny_bert(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .unwrap();
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);

    // Flattened search through the abstraction.
    let query = vec![0.5_f32; 32];
    let via_session = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(query.clone()),
            k: 5,
            filter: None,
            select: Vec::new(),
        })
        .await
        .unwrap();

    // Same query straight through the engine builder.
    let via_engine = engine
        .search("patents", query, 5)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!via_session.is_empty());
    assert_eq!(row_ids(&via_session), row_ids(&via_engine));
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
    let session = Session::Local(LocalSession::new(Arc::clone(&engine)));

    session
        .generate_embeddings(
            "patents",
            &tiny_bert(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .unwrap();

    let via_session = session
        .encode_query(
            &tiny_bert(),
            QueryInput::Text("battery".into()),
            Modality::Text,
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
            filter: None,
            select: Vec::new(),
        })
        .await
        .unwrap();
    let via_engine_key = engine
        .search_by_id("patents", &key, 3)
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
    let session = Session::Local(LocalSession::new(engine));

    let err = session
        .encode_query(
            "local:whatever",
            QueryInput::Bytes(vec![0, 1, 2]),
            Modality::Text,
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
