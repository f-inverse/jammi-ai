use approx::assert_abs_diff_eq;
use jammi_numerics::retrieval::{RelevanceJudgment, RetrievalMetrics};

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
fn recall_at_ks_matches_known_cutoffs() {
    // Three relevant docs; retrieval surfaces doc1 at rank 1, doc2 at rank 3,
    // doc3 at rank 5 (the rest are noise). Recall@k must climb with k:
    //   @1 → 1/3, @3 → 2/3, @5 → 3/3, @10 → 3/3 (clamped to retrieved len).
    let judgments = judgments_basic();
    let retrieved: Vec<String> = vec![
        "doc1".into(),
        "noise1".into(),
        "doc2".into(),
        "noise2".into(),
        "doc3".into(),
    ];
    let recalls = RetrievalMetrics::recall_at_ks(&retrieved, &judgments, &[1, 3, 5, 10]);
    assert_eq!(recalls.len(), 4);
    assert_eq!(recalls[0].0, 1);
    assert_abs_diff_eq!(recalls[0].1, 1.0 / 3.0, epsilon = 1e-12);
    assert_eq!(recalls[1].0, 3);
    assert_abs_diff_eq!(recalls[1].1, 2.0 / 3.0, epsilon = 1e-12);
    assert_eq!(recalls[2].0, 5);
    assert_abs_diff_eq!(recalls[2].1, 1.0, epsilon = 1e-12);
    assert_eq!(recalls[3].0, 10);
    assert_abs_diff_eq!(recalls[3].1, 1.0, epsilon = 1e-12);
}

#[test]
fn recall_at_ks_agrees_with_compute_query_at_same_k() {
    // The multi-K helper must reuse the same recall definition as
    // `compute_query` — for any single k they must agree exactly.
    let judgments = judgments_basic();
    let retrieved: Vec<String> = vec!["doc1".into(), "noise".into(), "doc3".into()];
    for k in [1usize, 2, 3, 5] {
        let single = RetrievalMetrics::compute_query(&retrieved, &judgments, k).recall;
        let multi = RetrievalMetrics::recall_at_ks(&retrieved, &judgments, &[k])[0].1;
        assert_abs_diff_eq!(single, multi, epsilon = 1e-12);
    }
}

#[test]
fn recall_at_ks_no_relevant_is_zero() {
    let retrieved: Vec<String> = vec!["a".into(), "b".into()];
    let recalls = RetrievalMetrics::recall_at_ks(&retrieved, &[], &[1, 3]);
    assert_abs_diff_eq!(recalls[0].1, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(recalls[1].1, 0.0, epsilon = 1e-12);
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
