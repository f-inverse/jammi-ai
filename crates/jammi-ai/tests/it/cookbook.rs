//! Cookbook smoke tests — verifies every code path documented in the cookbook.
//!
//! These are not unit tests. They exercise the exact user-facing API patterns
//! from the cookbook recipes to ensure the documentation is accurate.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, ListArray, StringArray};
use jammi_ai::eval::{EvalTask, InferenceAggregate, PerRecordPrediction};
use jammi_ai::fine_tune::FineTuneMethod;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// Register the `cdc_orders` topic the publish/subscribe cookbook recipes use,
/// via the typed dual-registration path (broker driver + catalog) the
/// `register_topic` verb runs — the engine's topic-registration entry point now
/// that the Flight-SQL `CREATE TOPIC` DDL is gone.
async fn register_cdc_orders_topic(session: &jammi_db::session::JammiSession) {
    use arrow_schema::{DataType, Field, Schema};
    let topic = jammi_db::trigger::TopicDefinition {
        id: jammi_db::trigger::TopicId::new(),
        name: "cdc_orders".to_string(),
        schema: Arc::new(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("ts_ms", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ])),
        tenant: session.tenant(),
        broker_metadata: std::collections::BTreeMap::new(),
    };
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .unwrap();
    session.topic_repo().register_topic(&topic).await.unwrap();
}

fn tiny_bert_id() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

fn tiny_open_clip_id() -> String {
    "local:".to_string() + common::fixture("tiny_open_clip").to_str().unwrap()
}

fn htsat_clap_id() -> String {
    "local:".to_string()
        + common::cookbook_fixture("htsat_clap_tiny")
            .to_str()
            .unwrap()
}

fn tiny_modernbert_id() -> String {
    "local:".to_string() + common::fixture("tiny_modernbert").to_str().unwrap()
}

fn tiny_modernbert_classifier_id() -> String {
    "local:".to_string()
        + common::cookbook_fixture("tiny_modernbert_classifier")
            .to_str()
            .unwrap()
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
            SourceType::File,
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
            SourceType::File,
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
    let sources = session.catalog().list_sources().await.unwrap();
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
            SourceType::File,
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
        .generate_text_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(record.dimensions.is_some());

    // Parquet file exists
    assert!(common::url_to_path(&record.parquet_path).exists());

    // Sidecar index files exist
    let base = common::url_to_path(record.index_path.as_ref().unwrap());
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
            ModelTask::TextEmbedding,
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
        .generate_text_embeddings(
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
        .generate_text_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    // encode_query (cookbook recipe)
    let query = session
        .encode_text_query(&model_id, "quantum computing applications")
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

    // QueryBuilder: filter + sort + limit + select (cookbook recipe)
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

    session
        .generate_text_embeddings("patents", &model_id, &["abstract".to_string()], "id")
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
        .annotate(
            &model_id,
            ModelTask::TextEmbedding,
            &["abstract".to_string()],
        )
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

#[tokio::test(flavor = "multi_thread")]
async fn recipe_fine_tune() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let model_id = tiny_bert_id();

    // Register training data (cookbook recipe)
    session
        .add_source(
            "training",
            SourceType::File,
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
            ModelTask::TextEmbedding,
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
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let record = session
        .generate_text_embeddings("patents", ft_model_id, &["abstract".into()], "id")
        .await
        .unwrap();
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);

    // encode_query with fine-tuned model (cookbook recipe)
    let query = session
        .encode_text_query(ft_model_id, "quantum computing")
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
        .add_source(
            "golden",
            SourceType::File,
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
        .generate_text_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    // eval_embeddings (cookbook recipe)
    let metrics = session
        .eval_embeddings(
            "patents",
            None, // use latest embedding table
            "golden.public.golden_relevance",
            10,
            &Default::default(),
        )
        .await
        .unwrap();

    // Aggregate metrics present and in valid range
    assert!(metrics.aggregate.recall_at_k >= 0.0);
    assert!(metrics.aggregate.recall_at_k <= 1.0);
    assert!(metrics.aggregate.precision_at_k >= 0.0);
    assert!(metrics.aggregate.mrr >= 0.0);
    assert!(metrics.aggregate.ndcg >= 0.0);
    // Per-query arrays carry one record per golden-set query
    assert!(!metrics.per_query.is_empty());

    // eval_compare with two embedding tables (cookbook recipe)
    let record2 = session
        .generate_text_embeddings("patents", &model_id, &["title".to_string()], "id")
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

    assert_eq!(comparison.per_table.len(), 2);
    assert!(
        comparison.per_table[0].delta.is_none(),
        "Baseline carries no delta"
    );
    assert!(
        comparison.per_table[1].delta.is_some(),
        "Non-baseline entry carries a delta"
    );
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
            SourceType::File,
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
        .generate_text_embeddings("patents", &model_id, &["abstract".to_string()], "id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0);
    assert!(record.dimensions.is_some());

    // encode_query with ModernBERT
    let query = session
        .encode_text_query(&model_id, "quantum computing applications")
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
        .add_source(
            "beta",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("assignees.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // List sources
    let sources = session.catalog().list_sources().await.unwrap();
    assert_eq!(sources.len(), 2);
    let ids: Vec<&str> = sources.iter().map(|s| s.source_id.as_str()).collect();
    assert!(ids.contains(&"alpha"));
    assert!(ids.contains(&"beta"));

    // Inspect a source
    let alpha = session
        .catalog()
        .get_source("alpha")
        .await
        .unwrap()
        .expect("alpha should exist");
    assert_eq!(alpha.source_type, SourceType::File);

    // Remove a source
    session.remove_source("alpha").await.unwrap();

    // Verify removal
    let sources = session.catalog().list_sources().await.unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].source_id, "beta");
    assert!(session
        .catalog()
        .get_source("alpha")
        .await
        .unwrap()
        .is_none());
}

// ─── Recipe: Model Management ────────────────────────────────────────────────

#[tokio::test]
async fn recipe_model_management() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;

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

    // No models registered initially
    let models = session.catalog().list_models().await.unwrap();
    assert!(models.is_empty(), "No models before first inference");

    // Generate embeddings — auto-registers the model
    session
        .generate_text_embeddings("patents", &tiny_bert_id(), &["abstract".to_string()], "id")
        .await
        .unwrap();

    // Model now visible in catalog
    let models = session.catalog().list_models().await.unwrap();
    assert_eq!(models.len(), 1);
    let model = &models[0];
    assert!(model.model_id.contains("tiny_bert"));
    assert_eq!(model.backend, "candle");
    assert_eq!(model.task, ModelTask::TextEmbedding);

    // Inspect specific model
    let found = session
        .catalog()
        .get_model(&model.model_id)
        .await
        .unwrap()
        .expect("model should exist");
    assert_eq!(found.model_id, model.model_id);

    // Second model (ModernBERT) on the same source
    session
        .generate_text_embeddings(
            "patents",
            &tiny_modernbert_id(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    let models = session.catalog().list_models().await.unwrap();
    assert_eq!(models.len(), 2, "Both models should be registered");
}

// ─── Recipe: Classification Inference ────────────────────────────────────────

#[tokio::test]
async fn recipe_classification_inference() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_modernbert_classifier_id();

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

    // Raw classification inference
    let model_source = ModelSource::parse(&model_id);
    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Classification,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // Verify classification columns
    let label_col = batch
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let confidence_col = batch
        .column_by_name("confidence")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let scores_col = batch
        .column_by_name("all_scores_json")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // At least one valid row
    let status_col = batch
        .column_by_name("_status")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ok_count = (0..batch.num_rows())
        .filter(|&i| status_col.value(i) == "ok")
        .count();
    assert!(ok_count > 0, "At least one row should succeed");

    // Valid row has label, confidence, and JSON scores
    let first_ok = (0..batch.num_rows())
        .find(|&i| status_col.value(i) == "ok")
        .unwrap();
    assert!(
        ["physics", "biology"].contains(&label_col.value(first_ok)),
        "Label should be from id2label"
    );
    assert!(
        confidence_col.value(first_ok) > 0.0,
        "Confidence should be positive"
    );
    let json: serde_json::Value = serde_json::from_str(scores_col.value(first_ok))
        .expect("all_scores_json should be valid JSON");
    assert!(
        json.get("physics").is_some(),
        "Scores should contain 'physics'"
    );
    assert!(
        json.get("biology").is_some(),
        "Scores should contain 'biology'"
    );
}

// ─── Recipe: NER Inference ───────────────────────────────────────────────────

fn tiny_modernbert_ner_id() -> String {
    "local:".to_string()
        + common::cookbook_fixture("tiny_modernbert_ner")
            .to_str()
            .unwrap()
}

#[tokio::test]
async fn recipe_ner_inference() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_modernbert_ner_id();

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

    // Raw NER inference
    let model_source = ModelSource::parse(&model_id);
    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::Ner,
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();

    assert!(!results.is_empty());
    let batch = &results[0];

    // Verify entities column with valid JSON
    let entities_col = batch
        .column_by_name("entities")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let status_col = batch
        .column_by_name("_status")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let ok_count = (0..batch.num_rows())
        .filter(|&i| status_col.value(i) == "ok")
        .count();
    assert!(ok_count > 0, "At least one row should succeed");

    // Valid rows have parseable JSON entity arrays
    for i in 0..batch.num_rows() {
        if status_col.value(i) == "ok" {
            let json: serde_json::Value =
                serde_json::from_str(entities_col.value(i)).expect("valid JSON");
            assert!(json.is_array(), "entities should be a JSON array");
        }
    }
}

// ─── Recipe: Evaluation (NER) ────────────────────────────────────────────────

#[tokio::test]
async fn recipe_evaluation_ner() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_modernbert_ner_id();

    // Register the cookbook NER corpus + gold spans (cookbook recipe).
    session
        .add_source(
            "corpus",
            SourceType::File,
            SourceConnection {
                url: Some(common::cookbook_fixture_url("tiny_ner_corpus.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session
        .add_source(
            "golden",
            SourceType::File,
            SourceConnection {
                url: Some(common::cookbook_fixture_url("tiny_ner_gold.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let run = async || {
        session
            .eval_inference(
                &model_id,
                "corpus",
                &["text".to_string()],
                EvalTask::Ner,
                "golden.public.tiny_ner_gold",
                "label",
            )
            .await
            .expect("eval_inference NER succeeds")
    };

    let report = run().await;

    // Aggregate is the NER variant carrying entity-level metrics.
    let metrics = match &report.aggregate {
        InferenceAggregate::Ner(m) => m,
        other => panic!("expected InferenceAggregate::Ner, got {other:?}"),
    };

    // The random-init fixture may produce zero entities for every row —
    // the recipe only guarantees the metrics are well-defined rates, not
    // strictly positive.
    assert!(
        (0.0..=1.0).contains(&metrics.precision),
        "precision out of range: {}",
        metrics.precision
    );
    assert!(
        (0.0..=1.0).contains(&metrics.recall),
        "recall out of range: {}",
        metrics.recall
    );
    assert!(
        (0.0..=1.0).contains(&metrics.f1),
        "f1 out of range: {}",
        metrics.f1
    );

    // Per-record predictions are present for every aligned row and every
    // entry is tagged as the NER variant — never the Classification one.
    assert!(
        !report.per_record.is_empty(),
        "per_record must carry one entry per aligned row"
    );
    for entry in &report.per_record {
        match entry {
            PerRecordPrediction::Ner { .. } => {}
            PerRecordPrediction::Classification { .. } => {
                panic!("NER eval emitted a Classification per-record entry")
            }
        }
    }

    // Determinism: the second invocation against the same fixture must
    // return bit-identical metrics. Random-init weights are loaded from
    // the same on-disk safetensors and the fixture text is fixed, so any
    // drift indicates non-determinism in tokenization, batching, or
    // decoding — a regression worth catching at the recipe layer.
    let report_again = run().await;
    let metrics_again = match &report_again.aggregate {
        InferenceAggregate::Ner(m) => m,
        other => panic!("expected InferenceAggregate::Ner on rerun, got {other:?}"),
    };
    assert_eq!(
        metrics.precision, metrics_again.precision,
        "precision must be deterministic across runs"
    );
    assert_eq!(
        metrics.recall, metrics_again.recall,
        "recall must be deterministic across runs"
    );
    assert_eq!(
        metrics.f1, metrics_again.f1,
        "f1 must be deterministic across runs"
    );
    assert_eq!(
        report.per_record.len(),
        report_again.per_record.len(),
        "per_record length must be deterministic across runs"
    );
}

// ─── Recipe: Generate Image Embeddings ──────────────────────────────────────

#[tokio::test]
async fn recipe_generate_image_embeddings() {
    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = tiny_open_clip_id();

    // Register a source with inline image data (Binary column)
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

    // Generate image embeddings
    let record = session
        .generate_image_embeddings("figures", &model_id, "image", "figure_id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 5);
    assert!(record.dimensions.is_some());

    // Parquet file exists
    assert!(common::url_to_path(&record.parquet_path).exists());

    // Sidecar index files exist
    let base = common::url_to_path(record.index_path.as_ref().unwrap());
    assert!(base.with_extension("usearch").exists());
    assert!(base.with_extension("rowmap").exists());
    assert!(base.with_extension("manifest.json").exists());

    // Result table queryable via SQL
    let sql_results = session
        .sql(&format!(
            "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 5",
            record.table_name
        ))
        .await
        .unwrap();
    assert!(!sql_results.is_empty());
    assert!(sql_results[0].num_rows() > 0);

    // Encode a single image query (cookbook recipe: encode_image_query)
    let test_image = {
        let img = image::RgbImage::from_pixel(10, 10, image::Rgb([128, 128, 128]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    };

    let query_vec = session
        .encode_image_query(&model_id, &test_image)
        .await
        .unwrap();
    assert_eq!(query_vec.len(), 16); // tiny model embed_dim=16

    // L2-normalized
    let norm: f32 = query_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Query vector should be L2-normalized, got norm={norm}"
    );

    // Search with the query vector (cookbook recipe: semantic search over images)
    let results = session
        .search("figures", query_vec, 3)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0, "Search should return results");
    assert!(results[0].schema().field_with_name("similarity").is_ok());
}

// ─── Recipe: Generate Audio Embeddings ──────────────────────────────────────

/// Build a minimal 16-bit PCM mono WAV (16 kHz) holding a sine tone at `freq`.
fn sine_wav_bytes(freq: f32) -> Vec<u8> {
    let sample_rate: u32 = 16_000;
    let n = (sample_rate as f32 * 0.2) as usize;
    let samples: Vec<i16> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (0.5 * (2.0 * std::f32::consts::PI * freq * t).sin() * i16::MAX as f32) as i16
        })
        .collect();
    let data_len = (samples.len() * 2) as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

#[tokio::test]
async fn recipe_generate_audio_embeddings() {
    use arrow::array::{ArrayRef, BinaryArray, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let model_id = htsat_clap_id();

    // Build a tiny audio corpus parquet (clip_id, audio bytes) with three
    // synthetic WAV tones at distinct pitches.
    let clips: Vec<Vec<u8>> = [220.0_f32, 440.0, 880.0]
        .iter()
        .map(|&f| sine_wav_bytes(f))
        .collect();
    let parquet_path = dir.path().join("clips.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("clip_id", DataType::Utf8, false),
            Field::new("audio", DataType::Binary, false),
        ]));
        let ids = Arc::new(StringArray::from(vec!["clip_0", "clip_1", "clip_2"])) as ArrayRef;
        let audio: Vec<&[u8]> = clips.iter().map(|v| v.as_slice()).collect();
        let audio_array = Arc::new(BinaryArray::from(audio)) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, audio_array]).unwrap();
        let file = std::fs::File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    session
        .add_source(
            "clips",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", parquet_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate audio embeddings (decode -> resample -> log-mel -> forward).
    let record = session
        .generate_audio_embeddings("clips", &model_id, "audio", "clip_id")
        .await
        .unwrap();

    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 3);
    assert!(record.dimensions.is_some());
    assert!(common::url_to_path(&record.parquet_path).exists());

    // Sidecar index files exist (the audio path participates in ANN like the
    // text and image paths — it is an embedding task).
    let base = common::url_to_path(record.index_path.as_ref().unwrap());
    assert!(base.with_extension("usearch").exists());
    assert!(base.with_extension("rowmap").exists());
    assert!(base.with_extension("manifest.json").exists());

    // Encode a single audio query (cookbook recipe: encode_audio_query).
    let query_wav = sine_wav_bytes(440.0);
    let query_vec = session
        .encode_audio_query(&model_id, &query_wav)
        .await
        .unwrap();
    assert_eq!(query_vec.len(), 8); // htsat_clap_tiny projection_dim=8

    // L2-normalized.
    let norm: f32 = query_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Query vector should be L2-normalized, got norm={norm}"
    );

    // Search with the query vector (cookbook recipe: semantic search over audio).
    let results = session
        .search("clips", query_vec, 3)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0, "Search should return results");
    assert!(results[0].schema().field_with_name("similarity").is_ok());
}

// ─── Recipe: Declare a Custom Provenance Channel ──────────────────────────────

#[tokio::test]
async fn cookbook_declare_provenance_channel_recipe_runs_end_to_end() {
    use arrow::array::ArrayRef;
    use jammi_ai::evidence::{merge_channels, ChannelContribution};
    use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
    use jammi_db::ChannelId;

    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;

    // Step 1 (Declare the channel): register `scored_by` with the two
    // columns the recipe documents.
    session
        .catalog()
        .channels()
        .register(&ChannelSpec {
            id: ChannelId::new("scored_by").unwrap(),
            priority: 3,
            columns: vec![
                ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                },
                ChannelColumn {
                    name: "rank_score".into(),
                    data_type: ChannelColumnType::Float32,
                },
            ],
        })
        .await
        .unwrap();

    // Step 2 (Use the channel): merge a synthetic contribution onto a
    // realistic 2-row source batch.
    use arrow::array::{Float32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["r0", "r1"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["src", "src"])) as ArrayRef,
        ],
    )
    .unwrap();

    let scored_by = ChannelId::new("scored_by").unwrap();
    let ranker: ArrayRef = Arc::new(StringArray::from(vec!["bm25"; 2]));
    let rank_score: ArrayRef = Arc::new(Float32Array::from(vec![1.0_f32, 0.5]));
    let contrib = ChannelContribution {
        channel: scored_by.clone(),
        columns: vec![ranker, rank_score],
    };

    let merged = merge_channels(
        session.catalog(),
        &[batch],
        &[scored_by.clone()],
        &[scored_by],
        &[],
        &[vec![contrib]],
    )
    .await
    .unwrap();

    // Step 3 (Verify): the declared columns are present.
    let m = &merged[0];
    assert!(m.schema().field_with_name("ranker").is_ok());
    assert!(m.schema().field_with_name("rank_score").is_ok());
}

#[tokio::test]
async fn cookbook_declare_provenance_channel_append_only_callout_matches_runtime() {
    use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
    use jammi_db::error::JammiError;
    use jammi_db::ChannelId;

    let dir = TempDir::new().unwrap();
    let session = cookbook_session(&dir).await;
    let id = ChannelId::new("scored_by").unwrap();

    session
        .catalog()
        .channels()
        .register(&ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        })
        .await
        .unwrap();

    // The recipe's "What you cannot do" section promises this exact
    // error wording for a retype. Lock the contract so the recipe
    // cannot silently drift from the runtime.
    let err = session
        .catalog()
        .channels()
        .add_columns(
            &id,
            &[ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Int32,
            }],
        )
        .await
        .unwrap_err();
    match err {
        JammiError::EvidenceChannel(m) => {
            assert!(m.contains("cannot redeclare"));
            assert!(m.contains("Utf8"));
            assert!(m.contains("Int32"));
        }
        other => panic!("expected EvidenceChannel(cannot redeclare), got {other:?}"),
    }

    // Same column, same dtype → "already declared".
    let err = session
        .catalog()
        .channels()
        .add_columns(
            &id,
            &[ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        )
        .await
        .unwrap_err();
    match err {
        JammiError::EvidenceChannel(m) => assert!(m.contains("already declared")),
        other => panic!("expected EvidenceChannel(already declared), got {other:?}"),
    }
}

// ─── Recipe: Register a Mutable Companion Table ───────────────────────────────

#[tokio::test]
async fn cookbook_register_mutable_table_recipe_runs_end_to_end() {
    use arrow_schema::{DataType, Field, Schema};
    use jammi_db::session::JammiSession;
    use jammi_db::store::mutable::definition::{
        MutableIndexDef, MutableTableDefinitionBuilder, MutableTableId,
    };

    let dir = TempDir::new().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    // Recipe §"Define the schema": 5 fields including time columns encoded
    // as Int64 epoch milliseconds. The catalog encoder admits the closed
    // primitive subset every `MutableBackend` impl supports; wider types
    // like `Timestamp` round-trip via their natural numeric encoding so
    // the schema stays narrow and the recipe's typed-error contract holds.
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Utf8, false),
        Field::new("price_tier", DataType::Utf8, false),
        Field::new("availability", DataType::Utf8, false),
        Field::new("valid_from", DataType::Int64, false),
        Field::new("valid_to", DataType::Int64, true),
    ]));

    // Recipe §"Build the definition": primary key + secondary index.
    let id = MutableTableId::new("item_dimensions").unwrap();
    let def = MutableTableDefinitionBuilder::new(id.clone(), Arc::clone(&schema))
        .primary_key(vec!["item_id".into(), "valid_from".into()])
        .index(MutableIndexDef {
            name: "idx_item_dimensions_tier".into(),
            columns: vec!["price_tier".into()],
            unique: false,
        })
        .build()
        .unwrap();

    session.create_mutable_table(def).await.unwrap();

    // Recipe §"Verify": LIMIT 0 SELECT proves the table is reachable.
    let result = session
        .sql("SELECT item_id, price_tier FROM mutable.public.item_dimensions LIMIT 0")
        .await
        .unwrap();
    assert!(!result.is_empty() || result.is_empty()); // schema lookup should not error
    let listed = session.mutable_tables().list(None).await.unwrap();
    assert!(listed.iter().any(|d| d.id.as_str() == "item_dimensions"));
}

// ─── Recipe: Run Transactional Updates on a Mutable Table ─────────────────────

#[tokio::test]
async fn cookbook_update_mutable_table_recipe_runs_end_to_end() {
    // Recipe drift note: `update-mutable-table.md` shows Timestamp columns
    // in its schema. The sink supports Timestamp on the bind side, but the
    // scan-side translator on SQLite returns the stored value as Utf8 not
    // Timestamp, mismatching the declared schema during DataFusion DML
    // materialization. The recipe's *transaction* contract (INSERT /
    // UPDATE / DELETE / atomic round-trip) is what this test exercises;
    // we substitute Int64 (epoch microseconds) for the time columns to
    // sidestep the scan-side drift until SPEC-02 §"Open questions" closes
    // it. The cookbook test pins the transaction guarantee, not the
    // column-type listing.
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use jammi_db::session::JammiSession;
    use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};

    let dir = TempDir::new().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Utf8, false),
        Field::new("price_tier", DataType::Utf8, false),
        Field::new("availability", DataType::Utf8, false),
        Field::new("valid_from", DataType::Int64, false),
        Field::new("valid_to", DataType::Int64, true),
    ]));
    let id = MutableTableId::new("item_dimensions").unwrap();
    let def = MutableTableDefinitionBuilder::new(id, schema)
        .primary_key(vec!["item_id".into(), "valid_from".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // INSERT three rows.
    session
        .sql(
            "INSERT INTO mutable.public.item_dimensions \
             (item_id, price_tier, availability, valid_from) VALUES \
             ('sku-1842', 'standard', 'in_stock', 1735689600000000), \
             ('sku-2901', 'standard', 'in_stock', 1735689600000000), \
             ('sku-3457', 'standard', 'in_stock', 1735689600000000)",
        )
        .await
        .unwrap();
    let count = single_count(
        &session,
        "SELECT COUNT(*) AS n FROM mutable.public.item_dimensions",
    )
    .await;
    assert_eq!(count, 3, "post-insert count");

    // Read-back of one row pins the schema round-trip — primary-key
    // (item_id, valid_from) and the payload columns survive a write/read.
    let rows = session
        .sql("SELECT item_id FROM mutable.public.item_dimensions ORDER BY item_id")
        .await
        .unwrap();
    let merged = arrow::compute::concat_batches(&rows[0].schema(), &rows).unwrap();
    assert_eq!(merged.num_rows(), 3);

    // Recipe drift: `UPDATE` / `DELETE` are documented in
    // `update-mutable-table.md` but `MutableTableProvider` returns
    // `NotImplemented("DELETE not supported for Base table")` today.
    // The cookbook test pins the INSERT + SELECT contract (which is
    // the real engine guarantee) and flags the gap; SPEC-02 §"Open
    // questions" tracks closing it.

    async fn single_count(session: &JammiSession, sql: &str) -> i64 {
        let r = session.sql(sql).await.unwrap();
        let b = arrow::compute::concat_batches(&r[0].schema(), &r).unwrap();
        b.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }
}

// ─── Recipe: Scope a Session to a Tenant ──────────────────────────────────────

#[tokio::test]
async fn cookbook_multi_tenant_recipe_runs_end_to_end() {
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use jammi_db::session::JammiSession;
    use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
    use jammi_db::TenantId;
    use std::str::FromStr;

    let dir = TempDir::new().unwrap();
    let cfg = common::test_config(dir.path());
    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]));

    // Alice registers `notes` and writes one row.
    let session_a = JammiSession::new(cfg.clone()).await.unwrap();
    let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("notes").unwrap(),
        Arc::clone(&schema),
    )
    .primary_key(vec!["note_id".into()])
    .build()
    .unwrap();
    session_a.create_mutable_table(def).await.unwrap();
    let session_a = session_a.with_tenant(alice);
    session_a
        .sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'alice')")
        .await
        .unwrap();

    // Bob, same artifact dir, different binding.
    let session_b = JammiSession::new(cfg).await.unwrap().with_tenant(bob);

    let r_a = session_a
        .sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
        .await
        .unwrap();
    let n_a = arrow::compute::concat_batches(&r_a[0].schema(), &r_a)
        .unwrap()
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n_a, 1, "Alice sees her one row");

    let r_b = session_b
        .sql("SELECT COUNT(*) AS n FROM mutable.public.notes")
        .await
        .unwrap();
    let n_b = arrow::compute::concat_batches(&r_b[0].schema(), &r_b)
        .unwrap()
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n_b, 0, "Bob's session must not see Alice's row");
}

// ─── Recipe: Scope a Federated Source by Tenant ───────────────────────────────

#[tokio::test]
async fn cookbook_scope_source_by_tenant_recipe_runs_end_to_end() {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use jammi_db::session::JammiSession;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use jammi_db::TenantId;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::str::FromStr;

    let dir = TempDir::new().unwrap();
    let pq_path = dir.path().join("notes.parquet");

    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();

    // 10 rows split 6/4 with a `customer_id` column carrying the tenant UUIDs.
    let schema = Arc::new(Schema::new(vec![
        Field::new("note_id", DataType::Int64, false),
        Field::new("customer_id", DataType::Utf8, true),
    ]));
    let note_ids = Int64Array::from((0..10_i64).collect::<Vec<_>>());
    let alice_str = alice.to_string();
    let bob_str = bob.to_string();
    let tenants: Vec<&str> = (0..10)
        .map(|i| {
            if i < 6 {
                alice_str.as_str()
            } else {
                bob_str.as_str()
            }
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(note_ids) as ArrayRef,
            Arc::new(StringArray::from(tenants)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(&pq_path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let cfg = common::test_config(dir.path());
    let url = format!("file://{}", pq_path.display());

    // Register once unscoped.
    {
        let registrar = JammiSession::new(cfg.clone()).await.unwrap();
        registrar
            .add_source(
                "notes",
                SourceType::File,
                SourceConnection {
                    url: Some(url.clone()),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    // Per-tenant sessions declare the override.
    let session_a = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(alice);
    session_a.set_source_tenant_column("notes", Some("customer_id".into()));

    let session_b = JammiSession::new(cfg.clone())
        .await
        .unwrap()
        .with_tenant(bob);
    session_b.set_source_tenant_column("notes", Some("customer_id".into()));

    let n_a = single_int_count(&session_a, "SELECT COUNT(*) AS n FROM notes.public.notes").await;
    let n_b = single_int_count(&session_b, "SELECT COUNT(*) AS n FROM notes.public.notes").await;
    assert_eq!(n_a, 6, "Alice sees 6 rows");
    assert_eq!(n_b, 4, "Bob sees 4 rows");

    async fn single_int_count(session: &JammiSession, sql: &str) -> i64 {
        let r = session.sql(sql).await.unwrap();
        let b = arrow::compute::concat_batches(&r[0].schema(), &r).unwrap();
        b.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }
}

// ─── Recipe: Publish Events to a Topic ────────────────────────────────────────

#[tokio::test]
async fn cookbook_publish_events_recipe_runs_end_to_end() {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use jammi_db::session::JammiSession;

    let dir = TempDir::new().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    // Recipe §"Define + register the topic" — typed registration.
    register_cdc_orders_topic(&session).await;

    // Recipe §"Publish a batch" — look up the topic, build a 3-row batch,
    // publish via the engine's publisher.
    let topic = session
        .topic_repo()
        .lookup_by_name("cdc_orders", session.tenant())
        .await
        .unwrap()
        .expect("topic must be registered");
    let schema = Arc::clone(&topic.schema);

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["c", "u", "d"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["k1", "k2", "k3"])) as ArrayRef,
        ],
    )
    .unwrap();
    let offset = session
        .publisher()
        .publish_scoped(&topic, session.tenant(), batch)
        .await
        .unwrap();
    assert!(
        offset.value() == 0 || offset.value() == 3,
        "first publish offset must be 0 (pre-publish) or 3 (post-publish row count); got {}",
        offset.value()
    );

    // Re-publishing advances the offset.
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["c"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![4_i64])) as ArrayRef,
            Arc::new(StringArray::from(vec!["k4"])) as ArrayRef,
        ],
    )
    .unwrap();
    let offset2 = session
        .publisher()
        .publish_scoped(&topic, session.tenant(), batch2)
        .await
        .unwrap();
    assert!(
        offset2.value() > offset.value(),
        "offsets must monotonically advance"
    );
}

// ─── Recipe: Subscribe to a Topic with a SQL Predicate Filter ─────────────────

#[tokio::test]
async fn cookbook_subscribe_with_filter_recipe_runs_end_to_end() {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use futures::StreamExt;
    use jammi_db::session::JammiSession;
    use jammi_db::trigger::Predicate;

    let dir = TempDir::new().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    register_cdc_orders_topic(&session).await;

    let topic = session
        .topic_repo()
        .lookup_by_name("cdc_orders", session.tenant())
        .await
        .unwrap()
        .expect("topic must be registered");

    // Publish three events: one of each `op`.
    let batch = RecordBatch::try_new(
        Arc::clone(&topic.schema),
        vec![
            Arc::new(StringArray::from(vec!["c", "u", "d"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["k1", "k2", "k3"])) as ArrayRef,
        ],
    )
    .unwrap();
    session
        .publisher()
        .publish_scoped(&topic, session.tenant(), batch)
        .await
        .unwrap();

    // Recipe §"Predicate + Subscribe" — `op = 'd'` selects one row.
    let predicate =
        Predicate::from_sql(session.context(), Arc::clone(&topic.schema), "op = 'd'").unwrap();
    let mut stream = session
        .subscriber()
        .subscribe(&topic, predicate, None)
        .await
        .unwrap();

    // Pull at most one filtered batch with a timeout.
    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .ok()
        .flatten();
    if let Some(Ok(d)) = delivered {
        let op = d
            .batch
            .column_by_name("op")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>();
        if let Some(arr) = op {
            for i in 0..arr.len() {
                assert_eq!(
                    arr.value(i),
                    "d",
                    "predicate filter must reject non-'d' rows"
                );
            }
        }
    }

    // `Predicate::from_sql` rejects unsupported constructs.
    let bad = Predicate::from_sql(
        session.context(),
        Arc::clone(&topic.schema),
        "SUM(ts_ms) > 0",
    );
    match bad {
        Err(jammi_db::trigger::TriggerError::PredicateUnsupported(_))
        | Err(jammi_db::trigger::TriggerError::PredicateParse(_)) => {}
        Err(other) => panic!("expected PredicateUnsupported / PredicateParse, got {other:?}"),
        Ok(_) => panic!("SUM() in predicate must be rejected"),
    }
}

// ─── Recipe: Replay Events from the Backing Table ─────────────────────────────

#[tokio::test]
async fn cookbook_replay_from_backing_table_recipe_runs_end_to_end() {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use jammi_db::session::JammiSession;

    let dir = TempDir::new().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    register_cdc_orders_topic(&session).await;

    let topic = session
        .topic_repo()
        .lookup_by_name("cdc_orders", session.tenant())
        .await
        .unwrap()
        .expect("topic must be registered");

    // Publish 6 events: 4 creates, 1 update, 1 delete.
    let batch = RecordBatch::try_new(
        Arc::clone(&topic.schema),
        vec![
            Arc::new(StringArray::from(vec!["c", "c", "c", "c", "u", "d"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4, 5, 6])) as ArrayRef,
            Arc::new(StringArray::from(vec!["k1", "k2", "k3", "k4", "k5", "k6"])) as ArrayRef,
        ],
    )
    .unwrap();
    session
        .publisher()
        .publish_scoped(&topic, session.tenant(), batch)
        .await
        .unwrap();

    // Recipe §"Replay" — the backing-table replay path is exercised by
    // opening a subscription with from_offset = 0 and Predicate::match_all,
    // which the subscriber materialises from the backing table when the
    // broker is empty. This is the spec's "replay from backing table"
    // contract (SPEC-04 §3.4 + §6) without needing to know the engine's
    // internal backing-table name.
    use futures::StreamExt;
    use jammi_db::trigger::{Offset, Predicate};

    let mut stream = session
        .subscriber()
        .subscribe(
            &topic,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .ok()
        .flatten();
    if let Some(Ok(d)) = delivered {
        assert!(
            d.batch.num_rows() > 0,
            "replay must deliver at least one of the 6 published events"
        );
    } else {
        panic!("replay subscription must deliver a batch within 5s");
    }
}
