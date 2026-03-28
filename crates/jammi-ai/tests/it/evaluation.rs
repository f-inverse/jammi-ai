use std::sync::Arc;

use jammi_ai::eval::{
    golden::{ensure_column, RelevanceJudgment},
    metrics::{classification::ClassificationMetrics, retrieval::RetrievalMetrics},
};
use jammi_ai::session::InferenceSession;
use jammi_engine::catalog::eval_repo::EvalRunRecord;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

use arrow::datatypes::{DataType, Field, Schema};
use tempfile::{tempdir, TempDir};

use crate::common;

// ─── Retrieval metrics: known inputs with hand-computed expected values ──────
//
// Guards recall/precision/MRR/nDCG formulas. The partial-match and edge cases
// are where formula bugs hide (e.g. dividing by k vs relevant_set.len()).

#[test]
fn retrieval_metrics_known_inputs() {
    let judgments_3 = vec![
        RelevanceJudgment {
            doc_id: "1".into(),
            grade: 1,
        },
        RelevanceJudgment {
            doc_id: "2".into(),
            grade: 1,
        },
        RelevanceJudgment {
            doc_id: "3".into(),
            grade: 1,
        },
    ];

    // Partial recall: 2 of 3 relevant found in top-5 (guards recall = hits/relevant, not hits/k)
    let partial = RetrievalMetrics::compute_query(
        &["1".into(), "4".into(), "2".into(), "5".into(), "6".into()],
        &judgments_3,
        5,
    );
    assert!(
        (partial.recall - 2.0 / 3.0).abs() < 1e-6,
        "recall = 2/3, got {}",
        partial.recall
    );

    // Precision with noise: 3 relevant in 5 retrieved
    let noisy = RetrievalMetrics::compute_query(
        &["1".into(), "99".into(), "2".into(), "98".into(), "3".into()],
        &judgments_3,
        5,
    );
    assert!(
        (noisy.precision - 3.0 / 5.0).abs() < 1e-6,
        "precision = 3/5"
    );

    // MRR: first relevant at position 2 → reciprocal rank = 1/2
    let mrr_test = RetrievalMetrics::compute_query(
        &["99".into(), "1".into(), "98".into()],
        &[RelevanceJudgment {
            doc_id: "1".into(),
            grade: 1,
        }],
        3,
    );
    assert!((mrr_test.mrr - 0.5).abs() < 1e-6, "MRR = 1/2");

    // No relevant found → all metrics zero (guards against NaN/panic on empty intersection)
    let miss =
        RetrievalMetrics::compute_query(&["99".into(), "98".into(), "97".into()], &judgments_3, 3);
    assert!(miss.mrr.abs() < 1e-6);
    assert!(miss.ndcg.abs() < 1e-6);

    // Empty results → zero (guards divide-by-zero in DCG computation)
    let empty = RetrievalMetrics::compute_query(&[], &judgments_3, 5);
    assert!(empty.ndcg.abs() < 1e-6);
}

// ─── nDCG graded relevance ──────────────────────────────────────────────────
//
// Guards that nDCG uses grade values, not just binary relevant/not-relevant.
// Without this test, someone could simplify nDCG to binary and it would look correct
// on all the uniform-grade tests above.

#[test]
fn retrieval_ndcg_uses_graded_relevance() {
    let judgments = vec![
        RelevanceJudgment {
            doc_id: "high".into(),
            grade: 3,
        },
        RelevanceJudgment {
            doc_id: "low".into(),
            grade: 1,
        },
    ];

    let good = RetrievalMetrics::compute_query(&["high".into(), "low".into()], &judgments, 2);
    let bad = RetrievalMetrics::compute_query(&["low".into(), "high".into()], &judgments, 2);
    assert!(
        good.ndcg > bad.ndcg,
        "Grade-3 at top ({}) must beat grade-1 at top ({})",
        good.ndcg,
        bad.ndcg
    );
}

// ─── Classification metrics ─────────────────────────────────────────────────
//
// Guards the confusion matrix logic. The per-class case (TP=1, FP=1 → precision=0.5)
// is where bugs in the counting logic hide.

#[test]
fn classification_metrics_known_inputs() {
    // 3/4 correct → accuracy = 0.75
    let partial = ClassificationMetrics::compute(
        &["a".into(), "b".into(), "a".into(), "c".into()],
        &["a".into(), "b".into(), "b".into(), "c".into()],
    );
    assert!(
        (partial.accuracy - 0.75).abs() < 1e-6,
        "accuracy = 0.75, got {}",
        partial.accuracy
    );

    // Per-class: physics predicted twice (1 correct, 1 wrong)
    // TP=1, FP=1, FN=0 → precision=0.5, recall=1.0
    let per_class = ClassificationMetrics::compute(
        &[
            "physics".into(),
            "physics".into(),
            "cs".into(),
            "cs".into(),
            "biology".into(),
        ],
        &[
            "physics".into(),
            "cs".into(),
            "cs".into(),
            "biology".into(),
            "biology".into(),
        ],
    );
    let physics = &per_class.per_class["physics"];
    assert!(
        (physics.precision - 0.5).abs() < 1e-6,
        "physics precision = 0.5, got {}",
        physics.precision
    );
    assert!(
        (physics.recall - 1.0).abs() < 1e-6,
        "physics recall = 1.0, got {}",
        physics.recall
    );

    // All wrong → accuracy = 0 (guards against off-by-one in count)
    let wrong = ClassificationMetrics::compute(
        &["a".into(), "a".into(), "a".into()],
        &["b".into(), "c".into(), "d".into()],
    );
    assert!(wrong.accuracy.abs() < 1e-6);
}

// ─── ROUGE-L ────────────────────────────────────────────────────────────────
//
// ─── Golden dataset schema validation ────────────────────────────────────────
//
// Guards that user-facing error messages name the offending column and type.
// This is a user experience guardrail — silent wrong-type acceptance would
// produce confusing metric results downstream.

#[test]
fn golden_schema_validation() {
    let schema = Schema::new(vec![
        Field::new("query_id", DataType::Utf8, false),
        Field::new("query_text", DataType::Utf8, false),
        Field::new("relevant_id", DataType::Utf8, false),
    ]);

    // Missing column → error names the column
    let missing = ensure_column(&schema, "nonexistent", DataType::Utf8);
    assert!(missing.is_err());
    let msg = missing.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent") && msg.contains("missing"),
        "Should name missing column: {msg}"
    );

    // Wrong type → error names column and actual type (Boolean is never string-compatible)
    let wrong_schema = Schema::new(vec![Field::new("query_id", DataType::Boolean, false)]);
    let wrong = ensure_column(&wrong_schema, "query_id", DataType::Utf8);
    assert!(wrong.is_err());
    let msg2 = wrong.unwrap_err().to_string();
    assert!(
        msg2.contains("query_id") && msg2.contains("Boolean"),
        "Should name column and type: {msg2}"
    );

    // Int64 is accepted for Utf8 columns (common for ID columns)
    let int_schema = Schema::new(vec![Field::new("relevant_id", DataType::Int64, false)]);
    assert!(ensure_column(&int_schema, "relevant_id", DataType::Utf8).is_ok());
}

// ─── Contract: metrics always in [0, 1] ─────────────────────────────────────
//
// Property test with non-trivial inputs (mixed grades, partial matches).
// Guards against formula bugs that produce values outside valid range.

#[test]
fn contract_metrics_in_valid_ranges() {
    let r = RetrievalMetrics::compute_query(
        &[
            "1".into(),
            "99".into(),
            "2".into(),
            "98".into(),
            "97".into(),
        ],
        &[
            RelevanceJudgment {
                doc_id: "1".into(),
                grade: 2,
            },
            RelevanceJudgment {
                doc_id: "2".into(),
                grade: 1,
            },
            RelevanceJudgment {
                doc_id: "3".into(),
                grade: 3,
            },
        ],
        5,
    );
    assert!(r.recall >= 0.0 && r.recall <= 1.0);
    assert!(r.precision >= 0.0 && r.precision <= 1.0);
    assert!(r.mrr >= 0.0 && r.mrr <= 1.0);
    assert!(r.ndcg >= 0.0 && r.ndcg <= 1.0);

    let c = ClassificationMetrics::compute(
        &["A".into(), "B".into(), "A".into(), "C".into()],
        &["A".into(), "A".into(), "A".into(), "C".into()],
    );
    assert!(c.accuracy >= 0.0 && c.accuracy <= 1.0);
    assert!(c.f1 >= 0.0 && c.f1 <= 1.0);
}

// ─── Catalog eval_runs: CRUD + latest_eval_run ordering ─────────────────────
//
// Integration test against real SQLite. Guards that latest_eval_run returns
// most recent by created_at (ORDER BY DESC LIMIT 1), not first inserted.

#[test]
fn catalog_eval_run_crud_and_latest() {
    let dir = tempdir().unwrap();
    let catalog = jammi_engine::catalog::Catalog::open(dir.path()).unwrap();

    // Missing model → None (not an error)
    assert!(catalog
        .latest_eval_run("nonexistent", "embedding")
        .unwrap()
        .is_none());

    // Register model (FK constraint on eval_runs.model_id)
    catalog
        .register_model(jammi_engine::catalog::model_repo::RegisterModelParams {
            model_id: "model-a",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "embedding",
            ..Default::default()
        })
        .unwrap();

    // Insert two runs with different created_at
    catalog
        .record_eval_run(&EvalRunRecord {
            eval_run_id: "er-1".into(),
            eval_type: "embedding".into(),
            model_id: "model-a::1".into(),
            source_id: "src".into(),
            golden_source: "golden".into(),
            k: Some(10),
            metrics_json: r#"{"recall_at_k": 0.5}"#.into(),
            status: "completed".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
        })
        .unwrap();
    catalog
        .record_eval_run(&EvalRunRecord {
            eval_run_id: "er-2".into(),
            eval_type: "embedding".into(),
            model_id: "model-a::1".into(),
            source_id: "src".into(),
            golden_source: "golden".into(),
            k: Some(10),
            metrics_json: r#"{"recall_at_k": 0.8}"#.into(),
            status: "completed".into(),
            created_at: "2026-01-02T00:00:00Z".into(),
        })
        .unwrap();

    // latest returns er-2 (most recent), not er-1 (first inserted)
    let latest = catalog
        .latest_eval_run("model-a::1", "embedding")
        .unwrap()
        .unwrap();
    assert_eq!(latest.eval_run_id, "er-2");
    let metrics: serde_json::Value = serde_json::from_str(&latest.metrics_json).unwrap();
    assert!((metrics["recall_at_k"].as_f64().unwrap() - 0.8).abs() < 1e-6);
}

// ─── End-to-end: eval_embeddings pipeline with tiny_bert ─────────────────────
//
// Covers UAT 10, 13, 15, 16. Runs the full pipeline: register source → generate
// embeddings → register golden → eval_embeddings → check metrics + catalog.
// Uses tiny_bert (32-dim, local) so no network access needed.

fn tiny_bert_model() -> String {
    "local:".to_string() + common::fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings_and_golden() -> (Arc<InferenceSession>, String, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    // Register patents source
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

    // Generate embeddings with tiny_bert
    let record = session
        .generate_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
        )
        .await
        .unwrap();
    let table_name = record.table_name.clone();

    // Register golden relevance dataset
    session
        .add_source(
            "golden_rel",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("golden_relevance.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    (session, table_name, dir)
}

#[tokio::test]
async fn eval_embeddings_end_to_end() {
    let (session, table_name, _dir) = session_with_embeddings_and_golden().await;

    // UAT 10: eval_embeddings returns retrieval metrics
    let metrics = session
        .eval_embeddings(
            "patents",
            Some(&table_name),
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // All four metric keys present and in valid range
    for key in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let val = metrics[key]
            .as_f64()
            .unwrap_or_else(|| panic!("Missing metric: {key}"));
        assert!((0.0..=1.0).contains(&val), "{key} = {val} outside [0, 1]");
    }

    // UAT 15: eval run recorded in catalog with golden_source and k
    let runs = session.catalog().list_eval_runs().unwrap();
    assert!(!runs.is_empty(), "Eval run should be recorded");
    let run = &runs[0];
    assert_eq!(run.eval_type, "embedding");
    assert!(!run.metrics_json.is_empty());
    assert!(!run.golden_source.is_empty());
    assert_eq!(run.k, Some(10));
    assert_eq!(run.status, "completed");

    // UAT 16: latest_eval_run retrieves the run we just created
    let latest = session
        .catalog()
        .latest_eval_run(&run.model_id, "embedding")
        .unwrap();
    assert!(latest.is_some());
    let latest = latest.unwrap();
    assert_eq!(latest.eval_type, "embedding");
    assert!(latest.k.is_some());
}

// ─── End-to-end: eval_compare pipeline ──────────────────────────────────────
//
// Covers UAT 14. Compares the same embedding table against itself — deltas
// must be zero. Validates the comparison structure (baseline, delta keys).

#[tokio::test]
async fn eval_compare_self_comparison_has_zero_deltas() {
    let (session, table_name, _dir) = session_with_embeddings_and_golden().await;

    let comparison = session
        .eval_compare(
            &[table_name.clone(), table_name.clone()],
            "patents",
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // Structure: baseline is a string, delta is an object
    assert!(
        comparison["baseline"].is_string(),
        "Should have baseline key"
    );
    assert!(comparison["delta"].is_object(), "Should have delta key");

    // Self-comparison: all deltas must be zero
    let deltas = comparison["delta"].as_object().unwrap();
    assert!(!deltas.is_empty(), "Should have at least one delta entry");
    for (_table, table_deltas) in deltas {
        for (metric, delta) in table_deltas.as_object().unwrap() {
            let abs_delta = delta["absolute"].as_f64().unwrap();
            assert!(
                abs_delta.abs() < 1e-6,
                "Self-comparison {metric} delta should be 0, got {abs_delta}"
            );
        }
    }
}

// ─── Eval determinism: same inputs → identical metrics ──────────────────────
//
// Invariant 3: eval metrics are deterministic for the same model + dataset.
// tiny_bert is deterministic on CPU. If this test fails, something introduced
// non-determinism (e.g. hash-order iteration, parallel execution reordering).

#[tokio::test]
async fn eval_embeddings_is_deterministic() {
    let (session, table_name, _dir) = session_with_embeddings_and_golden().await;

    let m1 = session
        .eval_embeddings(
            "patents",
            Some(&table_name),
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    let m2 = session
        .eval_embeddings(
            "patents",
            Some(&table_name),
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    for key in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let v1 = m1[key].as_f64().unwrap();
        let v2 = m2[key].as_f64().unwrap();
        assert!(
            (v1 - v2).abs() < 1e-12,
            "Determinism: {key} differs between runs: {v1} vs {v2}"
        );
    }
}

// ─── Eval compare with distinct tables ──────────────────────────────────────
//
// Invariant 4: model comparison deltas are consistent with individual evals.
// Generates embeddings from different text columns (title vs abstract) to
// produce genuinely different metrics, then verifies deltas are non-trivial.

#[tokio::test]
async fn eval_compare_distinct_tables_has_nonzero_deltas() {
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
        .add_source(
            "golden_rel",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("golden_relevance.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = tiny_bert_model();

    // Generate embeddings from two different text columns → different metrics
    let rec1 = session
        .generate_embeddings("patents", &model, &["abstract".to_string()], "id")
        .await
        .unwrap();
    let rec2 = session
        .generate_embeddings("patents", &model, &["title".to_string()], "id")
        .await
        .unwrap();

    let comparison = session
        .eval_compare(
            &[rec1.table_name.clone(), rec2.table_name.clone()],
            "patents",
            "golden_rel.public.golden_relevance",
            10,
        )
        .await
        .unwrap();

    // Baseline should be the first table
    assert_eq!(
        comparison["baseline"].as_str().unwrap(),
        rec1.table_name,
        "Baseline should be first table"
    );

    // Deltas should exist and at least one should be non-zero
    // (title and abstract produce different embeddings → different retrieval quality)
    let deltas = comparison["delta"].as_object().unwrap();
    let table2_deltas = &deltas[&rec2.table_name];
    let any_nonzero = ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
        .iter()
        .any(|key| {
            let abs = table2_deltas[key]["absolute"].as_f64().unwrap_or(0.0);
            abs.abs() > 1e-10
        });
    assert!(
        any_nonzero,
        "Different text columns should produce at least one non-zero delta"
    );
}

// ─── Image eval: query_image column instead of query_text ─────────────────

#[tokio::test]
async fn eval_image_embeddings_end_to_end() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    let tiny_open_clip = format!(
        "local:{}",
        common::fixture("tiny_open_clip").to_str().unwrap()
    );

    // Register source with inline images
    session
        .add_source(
            "figures",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("figures.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate image embeddings (single strategy, no rotation for simplicity)
    use jammi_ai::pipeline::image_embedding::EmbeddingStrategy;
    let record = session
        .generate_image_embeddings(
            "figures",
            &tiny_open_clip,
            "image",
            "figure_id",
            EmbeddingStrategy::Single,
        )
        .await
        .unwrap();
    let table_name = record.table_name.clone();

    // Register golden image relevance dataset
    session
        .add_source(
            "golden_img",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("golden_image_relevance.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Run image-aware eval
    let metrics = session
        .eval_embeddings(
            "figures",
            Some(&table_name),
            "golden_img.public.golden_image_relevance",
            5,
        )
        .await
        .unwrap();

    // All four metric keys present and in valid range
    for key in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let val = metrics[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key} missing or not a number"));
        assert!(
            (0.0..=1.0).contains(&val),
            "{key} = {val} out of [0, 1] range"
        );
    }
}
