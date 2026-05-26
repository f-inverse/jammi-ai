use std::sync::Arc;

use jammi_ai::eval::golden::{ensure_column, RelevanceJudgment};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_numerics::classification::ClassificationMetrics;
use jammi_numerics::retrieval::RetrievalMetrics;

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

#[tokio::test]
async fn catalog_eval_run_crud_and_latest() {
    let dir = tempdir().unwrap();
    let catalog = jammi_db::catalog::Catalog::open(dir.path()).await.unwrap();

    // Missing model → None (not an error)
    assert!(catalog
        .latest_eval_run("nonexistent", "embedding")
        .await
        .unwrap()
        .is_none());

    // Register model (FK constraint on eval_runs.model_id)
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: "model-a",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
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
        .await
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
        .await
        .unwrap();

    // latest returns er-2 (most recent), not er-1 (first inserted)
    let latest = catalog
        .latest_eval_run("model-a::1", "embedding")
        .await
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
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings_and_golden() -> (Arc<InferenceSession>, String, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    // Register patents source
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

    // Generate embeddings with tiny_bert
    let record = session
        .generate_text_embeddings(
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
            SourceType::File,
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

    // All four aggregate metrics present and in valid range
    for name in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let val = metrics
            .aggregate
            .field_by_name(name)
            .unwrap_or_else(|| panic!("Missing metric: {name}"));
        assert!((0.0..=1.0).contains(&val), "{name} = {val} outside [0, 1]");
    }

    // Per-query arrays returned alongside the aggregate; the join key is
    // the golden source's `query_id` and every record carries finite metrics.
    assert!(
        !metrics.per_query.is_empty(),
        "per_query must carry one record per golden-set query"
    );
    assert!(
        !metrics.per_query[0].query_id.is_empty(),
        "per_query records must carry the golden-set query_id"
    );
    assert!(
        metrics
            .per_query
            .iter()
            .all(|r| r.metrics.recall.is_finite()
                && r.metrics.precision.is_finite()
                && r.metrics.mrr.is_finite()
                && r.metrics.ndcg.is_finite()),
        "every per_query record must carry finite metrics"
    );

    // UAT 15: eval run recorded in catalog with golden_source and k
    let runs = session.catalog().list_eval_runs().await.unwrap();
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
        .await
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

    // Structure: baseline carries `delta: None`, follow-ups carry `Some`.
    assert_eq!(comparison.per_table.len(), 2);
    assert!(
        comparison.per_table[0].delta.is_none(),
        "Baseline entry must carry no delta"
    );
    let delta = comparison.per_table[1]
        .delta
        .as_ref()
        .expect("Non-baseline entry must carry a delta");

    // Self-comparison: every metric delta must be zero
    for (metric, value) in [
        ("recall_at_k", delta.recall_at_k.absolute),
        ("precision_at_k", delta.precision_at_k.absolute),
        ("mrr", delta.mrr.absolute),
        ("ndcg", delta.ndcg.absolute),
    ] {
        assert!(
            value.abs() < 1e-6,
            "Self-comparison {metric} delta should be 0, got {value}"
        );
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

    for name in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let v1 = m1.aggregate.field_by_name(name).unwrap();
        let v2 = m2.aggregate.field_by_name(name).unwrap();
        assert!(
            (v1 - v2).abs() < 1e-12,
            "Determinism: {name} differs between runs: {v1} vs {v2}"
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
            "golden_rel",
            SourceType::File,
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
        .generate_text_embeddings("patents", &model, &["abstract".to_string()], "id")
        .await
        .unwrap();
    let rec2 = session
        .generate_text_embeddings("patents", &model, &["title".to_string()], "id")
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

    // Baseline should be the first table — no delta, original table_name
    assert_eq!(comparison.per_table.len(), 2);
    assert_eq!(
        comparison.per_table[0].table_name, rec1.table_name,
        "Baseline should be the first table"
    );
    assert!(
        comparison.per_table[0].delta.is_none(),
        "Baseline entry must carry no delta"
    );

    // The second entry carries the delta against the baseline. At least one
    // metric should be non-zero (title and abstract produce different
    // embeddings → different retrieval quality).
    let entry = &comparison.per_table[1];
    assert_eq!(entry.table_name, rec2.table_name);
    let delta = entry
        .delta
        .as_ref()
        .expect("Non-baseline entry must carry a delta");
    let any_nonzero = [
        delta.recall_at_k.absolute,
        delta.precision_at_k.absolute,
        delta.mrr.absolute,
        delta.ndcg.absolute,
    ]
    .iter()
    .any(|v| v.abs() > 1e-10);
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
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("figures.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Generate image embeddings (single strategy, no rotation for simplicity)
    let record = session
        .generate_image_embeddings("figures", &tiny_open_clip, "image", "figure_id")
        .await
        .unwrap();
    let table_name = record.table_name.clone();

    // Register golden image relevance dataset
    session
        .add_source(
            "golden_img",
            SourceType::File,
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

    // All four aggregate metrics present and in valid range
    for name in ["recall_at_k", "precision_at_k", "mrr", "ndcg"] {
        let val = metrics
            .aggregate
            .field_by_name(name)
            .unwrap_or_else(|| panic!("{name} missing"));
        assert!(
            (0.0..=1.0).contains(&val),
            "{name} = {val} out of [0, 1] range"
        );
    }
}
