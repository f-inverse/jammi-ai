use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int64Array, ListArray, StringArray};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();

    (session, dir)
}

// ─── Vector search: results, hydrated columns, provenance, ordering ──────────

#[tokio::test]
async fn search_returns_hydrated_results_with_provenance() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0 && total_rows <= 5);

    // Evidence columns from ANN search
    assert!(batch.schema().field_with_name("_row_id").is_ok());
    assert!(batch.schema().field_with_name("_source_id").is_ok());
    assert!(batch.schema().field_with_name("similarity").is_ok());
    assert!(batch.schema().field_with_name("retrieved_by").is_ok());
    assert!(batch.schema().field_with_name("annotated_by").is_ok());

    // Hydrated columns from original source
    assert!(
        batch.schema().field_with_name("abstract").is_ok(),
        "Hydration should include original 'abstract' column"
    );
    assert!(
        batch.schema().field_with_name("title").is_ok(),
        "Hydration should include original 'title' column"
    );
    assert!(
        batch.schema().field_with_name("assignee_id").is_ok(),
        "Hydration should include original 'assignee_id' column"
    );

    // Similarity descending
    let sim = batch
        .column_by_name("similarity")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    for i in 1..sim.len() {
        assert!(
            sim.value(i - 1) >= sim.value(i),
            "Similarity should be descending: {} < {} at row {i}",
            sim.value(i - 1),
            sim.value(i)
        );
    }

    // retrieved_by = ["vector"], annotated_by = []
    let retrieved_by = batch
        .column_by_name("retrieved_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..retrieved_by.len() {
        let values = retrieved_by.value(i);
        let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
        let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
        assert!(channels.contains(&"vector"));
    }
}

// ─── Query-by-example: search_by_id resolves the row's vector internally ─────

#[tokio::test]
async fn search_by_id_ranks_the_query_row_first() {
    let (session, _dir) = session_with_embeddings().await;

    // Discover a real key by reading one row's `_row_id` from a plain search.
    let seed = session
        .search("patents", vec![0.5_f32; 32], 1, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let row_id_col = seed[0]
        .column_by_name("_row_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let row_key = row_id_col.value(0).to_string();

    // search_by_id resolves that row's stored vector internally and ranks by
    // it; a row is its own nearest neighbor, so it must come back first.
    let results = session
        .search_by_id("patents", &row_key, 5, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    let top = results[0]
        .column_by_name("_row_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(top.value(0), row_key, "the query row ranks first");

    let sim = results[0]
        .column_by_name("similarity")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert!(
        (sim.value(0) - 1.0).abs() < 1e-3,
        "self-match similarity is ~1.0, got {}",
        sim.value(0)
    );
}

#[tokio::test]
async fn search_by_id_rejects_an_unknown_key() {
    let (session, _dir) = session_with_embeddings().await;
    let err = match session
        .search_by_id("patents", "no-such-key", 5, None)
        .await
    {
        Ok(_) => panic!("an unknown key must error, not silently return nothing"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no-such-key"),
        "error must name the missing key, got: {err}"
    );
}

// ─── Sort + limit compose correctly ──────────────────────────────────────────

#[tokio::test]
async fn search_sort_and_limit_compose() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 10, None)
        .await
        .unwrap()
        .sort("similarity", true)
        .unwrap()
        .limit(3)
        .run()
        .await
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total <= 3, "Limit(3) should cap results, got {total}");

    for batch in &results {
        let sim = batch
            .column_by_name("similarity")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 1..sim.len() {
            assert!(sim.value(i - 1) >= sim.value(i));
        }
    }
}

// ─── Search fails without embedding table ────────────────────────────────────

#[tokio::test]
async fn search_fails_without_embedding_table() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

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

    let result = session.search("patents", vec![0.0f32; 32], 5, None).await;
    assert!(
        result.is_err(),
        "Search should fail when no embedding table exists"
    );
}

// ─── Join on real foreign key ────────────────────────────────────────────────

#[tokio::test]
async fn search_with_join_on_real_foreign_key() {
    let (session, _dir) = session_with_embeddings().await;

    session
        .add_source(
            "assignees",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5, None)
        .await
        .unwrap()
        .join("assignees", "assignee_id=id", None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // Joined columns from assignees
    assert!(
        batch.schema().field_with_name("company_name").is_ok(),
        "Join should add company_name from assignees"
    );
    assert!(
        batch.schema().field_with_name("country").is_ok(),
        "Join should add country from assignees"
    );

    // At least one row should have matched (patent assignee_id 101-110 matches assignees id 101-110)
    let company = batch
        .column_by_name("company_name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let has_match = (0..company.len()).any(|i| !company.is_null(i));
    assert!(
        has_match,
        "At least one joined row should have a company_name match"
    );
}

// ─── Annotate on real content column ─────────────────────────────────────────

#[tokio::test]
async fn search_with_annotate_on_real_column() {
    let (session, _dir) = session_with_embeddings().await;

    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 3, None)
        .await
        .unwrap()
        .annotate(
            &tiny_bert_model(),
            ModelTask::TextEmbedding,
            &["abstract".to_string()],
        )
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // annotated_by should contain "inference"
    let annotated_by = batch
        .column_by_name("annotated_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..annotated_by.len() {
        let values = annotated_by.value(i);
        let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
        let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
        assert!(channels.contains(&"inference"));
    }

    // retrieved_by should contain "vector" and NOT "inference"
    let retrieved_by = batch
        .column_by_name("retrieved_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for i in 0..retrieved_by.len() {
        let values = retrieved_by.value(i);
        let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
        let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
        assert!(channels.contains(&"vector"));
        assert!(!channels.contains(&"inference"));
    }
}

// ─── encode_query returns a vector ───────────────────────────────────────────

#[tokio::test]
async fn encode_query_returns_vector_of_correct_dimension() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

    let vector = session
        .encode_text_query(&tiny_bert_model(), "quantum computing")
        .await
        .unwrap();

    assert_eq!(vector.len(), 32, "Vector should be 32-dim for tiny_bert");
    assert!(
        vector.iter().any(|&v| v != 0.0),
        "Vector should not be all zeros"
    );
}

// ─── CP3 UAT 10: multiple tables → search resolves to latest ────────────────

#[tokio::test]
async fn search_resolves_to_latest_embedding_table() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

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

    let model = tiny_bert_model();

    // Generate first embedding table (columns=["title"]).
    let r1 = session
        .generate_text_embeddings(
            "patents",
            &model,
            &["title".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // Generate second embedding table (columns=["abstract"]), created later.
    let r2 = session
        .generate_text_embeddings(
            "patents",
            &model,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert_ne!(
        r1.table_name, r2.table_name,
        "Should produce distinct tables"
    );

    // Both tables should exist in the catalog.
    let tables = session
        .catalog()
        .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    assert_eq!(tables.len(), 2);

    // Resolve without explicit table name should pick the latest (r2).
    let resolved = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    assert_eq!(
        resolved.table_name, r2.table_name,
        "Should resolve to the latest embedding table"
    );

    // Search should work using the resolved (latest) table.
    let query = vec![0.5_f32; 32];
    let results = session
        .search("patents", query, 5, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!results.is_empty(), "Search should succeed on latest table");
}

// ─── Explicit table selector: search picks WHICH embedding table ─────────────

/// The ordered `_row_id` neighbour keys a search returns, flattened across
/// batches. Two searches that hit different embedding tables of the same source
/// must return different neighbour lists — that is what proves the selector
/// actually selects rather than silently using the most-recent table.
fn neighbour_keys(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    let mut keys = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("_row_id")
            .expect("results carry `_row_id`")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("`_row_id` is a StringArray");
        for i in 0..batch.num_rows() {
            keys.push(ids.value(i).to_string());
        }
    }
    keys
}

/// A source can carry several embedding tables (raw / propagated / fine-tuned).
/// `search(..., embedding_table=Some(name))` searches THAT table; two different
/// tables yield table-distinct neighbours for the same query, and
/// `embedding_table=None` preserves the most-recent-table default.
#[tokio::test]
async fn search_embedding_table_selector_picks_the_named_table() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

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

    let model = tiny_bert_model();

    // Two embedding tables on the SAME source, embedding different text columns
    // so their vectors — and therefore the nearest neighbours of a fixed query —
    // differ. `r_title` is created first; `r_abstract` last (the most-recent).
    let r_title = session
        .generate_text_embeddings(
            "patents",
            &model,
            &["title".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let r_abstract = session
        .generate_text_embeddings(
            "patents",
            &model,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    assert_ne!(
        r_title.table_name, r_abstract.table_name,
        "the two columns must produce distinct embedding tables"
    );

    let query = session
        .encode_text_query(&model, "quantum computing")
        .await
        .unwrap();
    let k = 10;
    let run = |table: Option<String>| {
        let session = Arc::clone(&session);
        let query = query.clone();
        async move {
            neighbour_keys(
                &session
                    .search("patents", query, k, table.as_deref())
                    .await
                    .unwrap()
                    .sort("similarity", true)
                    .unwrap()
                    .run()
                    .await
                    .unwrap(),
            )
        }
    };

    let title_keys = run(Some(r_title.table_name.clone())).await;
    let abstract_keys = run(Some(r_abstract.table_name.clone())).await;
    let default_keys = run(None).await;

    assert!(
        !title_keys.is_empty() && !abstract_keys.is_empty(),
        "each named table returns neighbours"
    );

    // The selector selects: naming the title table vs the abstract table for the
    // same query returns table-distinct neighbour lists.
    assert_ne!(
        title_keys, abstract_keys,
        "selecting different embedding tables must return different neighbours \
         (title={title_keys:?}, abstract={abstract_keys:?})"
    );

    // `embedding_table=None` preserves today's behaviour: the most-recent ready
    // table — here the abstract table, created last.
    assert_eq!(
        default_keys, abstract_keys,
        "embedding_table=None must search the most-recent table (abstract)"
    );
    assert_ne!(
        default_keys, title_keys,
        "the default must NOT silently search the title table"
    );
}

// ─── Semantic relevance: encode_query → search returns meaningful results ─────

#[tokio::test]
async fn search_returns_semantically_relevant_results() {
    let (session, _dir) = session_with_embeddings().await;

    // Encode a real query about quantum computing
    let query_vec = session
        .encode_text_query(&tiny_bert_model(), "quantum computing")
        .await
        .unwrap();

    // k=20 is deliberately >= the number of patents to verify we never return
    // more rows than exist.
    let results = session
        .search("patents", query_vec, 20, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    // 1. Results are non-empty
    assert!(
        !results.is_empty(),
        "Search should return at least one batch"
    );

    // Flatten all batches into parallel id / similarity vecs
    let mut all_ids: Vec<i64> = Vec::new();
    let mut all_similarities: Vec<f32> = Vec::new();

    for batch in &results {
        let ids = batch
            .column_by_name("id")
            .expect("results should contain 'id' column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("'id' column should be Int64Array");

        let sims = batch
            .column_by_name("similarity")
            .expect("results should contain 'similarity' column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("'similarity' column should be Float32Array");

        for i in 0..batch.num_rows() {
            all_ids.push(ids.value(i));
            all_similarities.push(sims.value(i));
        }
    }

    let total_rows = all_ids.len();

    // 2. Total row count <= 20 (can't return more rows than exist)
    assert!(
        total_rows <= 20,
        "Should not return more rows than exist, got {total_rows}"
    );

    // 3. At least one of the quantum-related patents {1, 4, 7} appears in the top 5
    let quantum_patent_ids: &[i64] = &[1, 4, 7];
    let top_n = total_rows.min(5);
    let top_ids = &all_ids[..top_n];
    let has_quantum_in_top5 = top_ids.iter().any(|id| quantum_patent_ids.contains(id));
    assert!(
        has_quantum_in_top5,
        "At least one of patents {{1, 4, 7}} should appear in top 5 results, \
         but top 5 IDs were: {top_ids:?}"
    );

    // 4. First result's similarity > 0.0 (query is not orthogonal to all documents)
    assert!(
        all_similarities[0] > 0.0,
        "First result similarity should be > 0.0, got {}",
        all_similarities[0]
    );

    // 5. All similarity values are in (0.0, 1.0] — not inverted or garbage
    for (i, &sim) in all_similarities.iter().enumerate() {
        assert!(
            sim > 0.0 && sim <= 1.0,
            "Similarity at row {i} should be in (0.0, 1.0], got {sim}"
        );
    }
}

// ─── Cross-modal: OpenCLIP text query against image embeddings ──────────────

fn tiny_open_clip_model() -> String {
    "local:".to_string() + common::fixture("tiny_open_clip").to_str().unwrap()
}

/// Embed an image corpus with OpenCLIP vision, embed a text query with the
/// same OpenCLIP checkpoint's text tower, and run `search()`. The text
/// vector lives in the shared latent space the vision tower projects into,
/// so cosine similarity is meaningful and `search()` returns ranked image
/// rows — this is the cross-modal search path that the v0.5.9 text tower
/// makes possible (no separate text encoder, no mismatched latent spaces).
#[tokio::test]
async fn cross_modal_text_to_image_search() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let model_id = tiny_open_clip_model();

    session
        .add_source(
            "figures",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("figures.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // 1. Embed the image corpus with the OpenCLIP vision tower.
    session
        .generate_image_embeddings(
            "figures",
            &model_id,
            "image",
            "figure_id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();

    // 2. Embed a text query with the OpenCLIP text tower (same checkpoint).
    let text_vec = session
        .encode_text_query(&model_id, "a colored rectangle")
        .await
        .unwrap();

    // 3. Run vector search against the image embeddings using the text vector.
    let results = session
        .search("figures", text_vec, 5, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(
        total_rows > 0,
        "Cross-modal search must return at least one image row"
    );

    // Hydrated columns: figure_id from the source, similarity from search.
    let batch = &results[0];
    assert!(
        batch.schema().field_with_name("figure_id").is_ok(),
        "Hydrated batch should carry the source key column"
    );
    let sims = batch
        .column_by_name("similarity")
        .expect("similarity column")
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    for i in 0..sims.len() {
        let sim = sims.value(i);
        assert!(
            sim.is_finite(),
            "Cross-modal similarity at row {i} should be finite, got {sim}"
        );
    }
}
