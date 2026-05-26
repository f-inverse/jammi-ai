use approx::assert_abs_diff_eq;
use jammi_numerics::retrieval::{AggregateMetrics, RelevanceJudgment, RetrievalMetrics};

fn judgments_basic() -> Vec<RelevanceJudgment> {
    vec![
        RelevanceJudgment {
            doc_id: "doc1".into(),
            grade: 3,
        },
        RelevanceJudgment {
            doc_id: "doc2".into(),
            grade: 2,
        },
        RelevanceJudgment {
            doc_id: "doc3".into(),
            grade: 1,
        },
    ]
}

#[test]
fn perfect_top_k_yields_ndcg_one() {
    let judgments = judgments_basic();
    let retrieved = vec!["doc1".into(), "doc2".into(), "doc3".into()];
    let m = RetrievalMetrics::compute_query(&retrieved, &judgments, 3);
    assert_abs_diff_eq!(m.recall, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.precision, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.mrr, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.ndcg, 1.0, epsilon = 1e-12);
}

#[test]
fn partial_recall_and_precision() {
    let judgments = judgments_basic();
    let retrieved = vec!["doc1".into(), "noise".into(), "doc3".into()];
    let m = RetrievalMetrics::compute_query(&retrieved, &judgments, 3);
    assert_abs_diff_eq!(m.recall, 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.precision, 2.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.mrr, 1.0, epsilon = 1e-12);
}

#[test]
fn no_hits_yields_zero() {
    let judgments = judgments_basic();
    let retrieved = vec!["noise1".into(), "noise2".into()];
    let m = RetrievalMetrics::compute_query(&retrieved, &judgments, 2);
    assert_abs_diff_eq!(m.recall, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.precision, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.mrr, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(m.ndcg, 0.0, epsilon = 1e-12);
}

#[test]
fn empty_retrieval_safe() {
    let judgments = judgments_basic();
    let m = RetrievalMetrics::compute_query(&[], &judgments, 5);
    assert_eq!(m.recall, 0.0);
    assert_eq!(m.precision, 0.0);
    assert_eq!(m.mrr, 0.0);
    assert_eq!(m.ndcg, 0.0);
}

#[test]
fn ndcg_uses_graded_relevance() {
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
        "expected better nDCG when high-grade comes first: good={}, bad={}",
        good.ndcg,
        bad.ndcg
    );
}

#[test]
fn aggregate_averages_queries() {
    let q1 = RetrievalMetrics::compute_query(
        &["doc1".into()],
        &[RelevanceJudgment {
            doc_id: "doc1".into(),
            grade: 1,
        }],
        1,
    );
    let q2 = RetrievalMetrics::compute_query(
        &["wrong".into()],
        &[RelevanceJudgment {
            doc_id: "doc1".into(),
            grade: 1,
        }],
        1,
    );
    let agg = RetrievalMetrics::aggregate(&[q1, q2]);
    assert_abs_diff_eq!(agg.recall_at_k, 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(agg.precision_at_k, 0.5, epsilon = 1e-12);
}

#[test]
fn aggregate_field_by_name_round_trips_known_keys() {
    let a = AggregateMetrics {
        recall_at_k: 0.1,
        precision_at_k: 0.2,
        mrr: 0.3,
        ndcg: 0.4,
    };
    assert_eq!(a.field_by_name("recall_at_k"), Some(0.1));
    assert_eq!(a.field_by_name("precision_at_k"), Some(0.2));
    assert_eq!(a.field_by_name("mrr"), Some(0.3));
    assert_eq!(a.field_by_name("ndcg"), Some(0.4));
    assert_eq!(a.field_by_name("unknown"), None);
}
