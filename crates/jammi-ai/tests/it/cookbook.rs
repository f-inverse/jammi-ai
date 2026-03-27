//! Cookbook smoke tests — verifies every code path documented in the cookbook.
//!
//! These are not unit tests. They exercise the exact user-facing API patterns
//! from the cookbook recipes to ensure the documentation is accurate.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray, StringArray};
use jammi_ai::fine_tune::FineTuneMethod;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert_id() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
}

fn tiny_modernbert_id() -> String {
    "local:".to_string() + common::fixture("tiny_modernbert").to_str().unwrap()
}

async fn cookbook_session(dir: &TempDir) -> Arc<InferenceSession> {
    let config = common::test_config(dir.path());
    Arc::new(InferenceSession::new(config).await.unwrap())
}

// ─── Recipe: Query Your Data with SQL ─────────────────────────────────────────

#[tokio::test]
async fn recipe_query_data_with_sql() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;

    // Register a Parquet source (as shown in quickstart + query-data recipe)
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

    // Simple query
    let results = session
        .sql("SELECT id, title, year FROM patents.public.patents WHERE year > 2020 ORDER BY year LIMIT 5")
        .await
        .unwrap();
    assert!(!results.is_empty());
    let batch = &results[0];
    assert!(batch.num_rows() > 0);
    assert!(batch.schema().field_with_name("title").is_ok());
    assert!(batch.schema().field_with_name("year").is_ok());

    // Aggregation
    let agg = session
        .sql("SELECT category, COUNT(*) as count FROM patents.public.patents GROUP BY category ORDER BY count DESC")
        .await
        .unwrap();
    assert!(!agg.is_empty());
    assert!(agg[0].schema().field_with_name("category").is_ok());
    assert!(agg[0].schema().field_with_name("count").is_ok());

    // Register a second source (CSV) and join across sources
    session
        .add_source(
            "companies",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let joined = session
        .sql("SELECT p.title, c.company_name FROM patents.public.patents p JOIN companies.public.assignees c ON p.assignee_id = c.id")
        .await
        .unwrap();
    assert!(!joined.is_empty());
    assert!(joined[0].schema().field_with_name("company_name").is_ok());

    // Source lifecycle
    let sources = session.catalog().list_sources().unwrap();
    assert!(sources.len() >= 2);
}

// ─── Recipe: Generate Embeddings ──────────────────────────────────────────────

#[tokio::test]
async fn recipe_generate_embeddings() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_bert_id();

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

    // Basic generate_embeddings (cookbook recipe)
    let record = session
        .generate_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(record.dimensions.is_some());

    // Parquet file exists
    assert!(std::path::Path::new(&record.parquet_path).exists());

    // Sidecar index files exist
    let base = std::path::Path::new(record.index_path.as_ref().unwrap());
    assert!(base.with_extension("usearch").exists());
    assert!(base.with_extension("rowmap").exists());
    assert!(base.with_extension("manifest.json").exists());

    // Result table queryable via SQL (cookbook: DataFusion integration)
    let sql_results = session
        .sql(&format!(
            "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 5",
            record.table_name
        ))
        .await
        .unwrap();
    assert!(!sql_results.is_empty());
    assert!(sql_results[0].num_rows() > 0);

    // Raw inference without persistence (cookbook: Raw inference)
    let model_source = ModelSource::parse(&model_id);
    let raw = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Embedding,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();
    assert!(!raw.is_empty());
    assert!(raw[0].schema().field_with_name("_status").is_ok());
    assert!(raw[0].schema().field_with_name("vector").is_ok());

    // Multiple text columns (cookbook recipe)
    let multi = session
        .generate_embeddings(
            "patents",
            &model_id,
            &["title".to_string(), "abstract".to_string()],
            "id",
        )
        .await
        .unwrap();
    assert_eq!(multi.status, "ready");
    assert!(multi.row_count > 0);
}

// ─── Recipe: Semantic Search ──────────────────────────────────────────────────

#[tokio::test]
async fn recipe_semantic_search() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_bert_id();

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
        .generate_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    // encode_query (cookbook recipe)
    let query = session
        .encode_query(&model_id, "quantum computing applications")
        .await
        .unwrap();
    assert!(!query.is_empty());

    // Basic search (cookbook recipe)
    let results = session
        .search("patents", query.clone(), 10)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].schema().field_with_name("similarity").is_ok());
    assert!(results[0].schema().field_with_name("title").is_ok());

    // SearchBuilder: filter + sort + limit + select (cookbook recipe)
    let filtered = session
        .search("patents", query.clone(), 20)
        .await
        .unwrap()
        .filter("year > 2020")
        .unwrap()
        .sort("similarity", true)
        .unwrap()
        .limit(5)
        .select(&["_row_id".into(), "title".into(), "similarity".into()])
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!filtered.is_empty());
    let batch = &filtered[0];
    assert!(batch.num_rows() <= 5);
    // 3 selected columns + 2 evidence columns (retrieved_by, annotated_by) always appended
    assert_eq!(batch.schema().fields().len(), 5);
    assert!(batch.schema().field_with_name("_row_id").is_ok());
    assert!(batch.schema().field_with_name("title").is_ok());
    assert!(batch.schema().field_with_name("similarity").is_ok());
    assert!(batch.schema().field_with_name("retrieved_by").is_ok());
    assert!(batch.schema().field_with_name("annotated_by").is_ok());

    // Verify similarity is descending (cookbook guarantee)
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

// ─── Recipe: Enrich Results with Joins and Annotations ────────────────────────

#[tokio::test]
async fn recipe_enrich_results() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_bert_id();

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
        .add_source(
            "assignees",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session
        .generate_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    let query = vec![0.5_f32; 32]; // tiny_bert is 32-dim

    // Join (cookbook recipe)
    let joined = session
        .search("patents", query.clone(), 10)
        .await
        .unwrap()
        .join("assignees", "assignee_id=id", None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!joined.is_empty());
    assert!(joined[0].schema().field_with_name("company_name").is_ok());

    // Annotate (cookbook recipe)
    let annotated = session
        .search("patents", query.clone(), 10)
        .await
        .unwrap()
        .annotate(&model_id, "embedding", &["abstract".to_string()])
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!annotated.is_empty());

    // Evidence provenance: annotated_by should contain "inference"
    let annotated_by = annotated[0]
        .column_by_name("annotated_by")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let values = annotated_by.value(0);
    let str_arr = values.as_any().downcast_ref::<StringArray>().unwrap();
    let channels: Vec<&str> = (0..str_arr.len()).map(|j| str_arr.value(j)).collect();
    assert!(channels.contains(&"inference"));

    // Compose join + filter + sort + limit + select (cookbook recipe)
    let composed = session
        .search("patents", query, 100)
        .await
        .unwrap()
        .join("assignees", "assignee_id=id", None)
        .await
        .unwrap()
        .filter("country = 'US'")
        .unwrap()
        .sort("similarity", true)
        .unwrap()
        .limit(10)
        .select(&["title".into(), "company_name".into(), "similarity".into()])
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!composed.is_empty());
    let batch = &composed[0];
    assert!(batch.num_rows() <= 10);
    assert!(batch.schema().field_with_name("company_name").is_ok());
}

// ─── Recipe: Fine-Tune for Your Domain ────────────────────────────────────────

#[tokio::test]
async fn recipe_fine_tune() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_bert_id();

    // Register training data (cookbook recipe)
    session
        .add_source(
            "training",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Start fine-tuning job with FineTuneMethod::Lora (cookbook recipe)
    let job = session
        .fine_tune(
            "training",
            &model_id,
            &["text_a".into(), "text_b".into(), "score".into()],
            FineTuneMethod::Lora,
            "embedding",
            None,
        )
        .await
        .unwrap();

    assert!(!job.job_id.is_empty());

    // Wait for completion (cookbook recipe)
    job.wait().await.unwrap();

    // Use the fine-tuned model (cookbook recipe)
    let ft_model_id = job.model_id();
    assert!(ft_model_id.starts_with("jammi:fine-tuned:"));

    // Register a source and generate embeddings with the fine-tuned model
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

    let record = session
        .generate_embeddings("patents", ft_model_id, &["abstract".into()], "id")
        .await
        .unwrap();
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);

    // encode_query with fine-tuned model (cookbook recipe)
    let query = session
        .encode_query(ft_model_id, "quantum computing")
        .await
        .unwrap();
    assert!(!query.is_empty());
}

// ─── Recipe: Evaluate and Compare Models ──────────────────────────────────────

#[tokio::test]
async fn recipe_evaluation() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_bert_id();

    // Register source + golden set (cookbook recipe)
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
        .add_source(
            "golden",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("golden_relevance.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate embeddings first
    let record = session
        .generate_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    // eval_embeddings (cookbook recipe)
    let metrics = session
        .eval_embeddings(
            "patents",
            None, // use latest embedding table
            "golden.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // Metrics present and in valid range
    assert!(metrics["recall_at_k"].as_f64().unwrap() >= 0.0);
    assert!(metrics["recall_at_k"].as_f64().unwrap() <= 1.0);
    assert!(metrics["precision_at_k"].as_f64().unwrap() >= 0.0);
    assert!(metrics["mrr"].as_f64().unwrap() >= 0.0);
    assert!(metrics["ndcg"].as_f64().unwrap() >= 0.0);

    // eval_compare with two embedding tables (cookbook recipe)
    let record2 = session
        .generate_embeddings("patents", &model_id, &["title".to_string()], "id")
        .await
        .unwrap();

    let comparison = session
        .eval_compare(
            &[record.table_name.clone(), record2.table_name.clone()],
            "patents",
            "golden.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    assert!(comparison.get("baseline").is_some());
    assert!(comparison.get("delta").is_some());
}

// ─── Recipe: ModernBERT Embeddings ───────────────────────────────────────────

#[tokio::test]
async fn recipe_modernbert_embeddings() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_modernbert_id();

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

    // generate_embeddings — same API as BERT, different model
    let record = session
        .generate_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(record.dimensions.is_some());

    // encode_query with ModernBERT
    let query = session
        .encode_query(&model_id, "quantum computing applications")
        .await
        .unwrap();
    assert_eq!(query.len(), 32, "tiny_modernbert has hidden_size=32");

    // search over ModernBERT-generated embeddings
    let results = session
        .search("patents", query, 10)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].schema().field_with_name("similarity").is_ok());
}

// ─── Recipe: Source Lifecycle ─────────────────────────────────────────────────

#[tokio::test]
async fn recipe_source_lifecycle() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;

    // Register two sources
    session
        .add_source(
            "alpha",
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
        .add_source(
            "beta",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // List sources
    let sources = session.catalog().list_sources().unwrap();
    assert_eq!(sources.len(), 2);
    let ids: Vec<&str> = sources.iter().map(|s| s.source_id.as_str()).collect();
    assert!(ids.contains(&"alpha"));
    assert!(ids.contains(&"beta"));

    // Inspect a source
    let alpha = session
        .catalog()
        .get_source("alpha")
        .unwrap()
        .expect("alpha should exist");
    assert_eq!(alpha.source_type, SourceType::Local);

    // Remove a source
    session.remove_source("alpha").unwrap();

    // Verify removal
    let sources = session.catalog().list_sources().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].source_id, "beta");
    assert!(session.catalog().get_source("alpha").unwrap().is_none());
}

// ─── Recipe: Model Management ────────────────────────────────────────────────

#[tokio::test]
async fn recipe_model_management() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;

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

    // No models registered initially
    let models = session.catalog().list_models().unwrap();
    assert!(models.is_empty(), "No models before first inference");

    // Generate embeddings — auto-registers the model
    session
        .generate_embeddings("patents", &tiny_bert_id(), &["abstract".to_string()], "id")
        .await
        .unwrap();

    // Model now visible in catalog
    let models = session.catalog().list_models().unwrap();
    assert_eq!(models.len(), 1);
    let model = &models[0];
    assert!(model.model_id.contains("tiny_bert"));
    assert_eq!(model.backend, "candle");
    assert_eq!(model.task, "embedding");

    // Inspect specific model
    let found = session
        .catalog()
        .get_model(&model.model_id)
        .unwrap()
        .expect("model should exist");
    assert_eq!(found.model_id, model.model_id);

    // Second model (ModernBERT) on the same source
    session
        .generate_embeddings(
            "patents",
            &tiny_modernbert_id(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    let models = session.catalog().list_models().unwrap();
    assert_eq!(models.len(), 2, "Both models should be registered");
}
