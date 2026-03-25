//! Retrieval metrics: recall@k, precision@k, MRR, nDCG with graded relevance.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::eval::golden::RelevanceJudgment;

/// Per-query metric values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub recall: f64,
    pub precision: f64,
    pub mrr: f64,
    pub ndcg: f64,
}

/// Aggregate metrics across all queries (mean).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub mrr: f64,
    pub ndcg: f64,
}

pub struct RetrievalMetrics;

impl RetrievalMetrics {
    /// Compute metrics for a single query given retrieved IDs and relevance judgments.
    pub fn compute_query(
        retrieved_ids: &[String],
        judgments: &[RelevanceJudgment],
        k: usize,
    ) -> QueryMetrics {
        let top_k = &retrieved_ids[..k.min(retrieved_ids.len())];

        let relevant_set: HashSet<&str> = judgments
            .iter()
            .filter(|j| j.grade > 0)
            .map(|j| j.doc_id.as_str())
            .collect();

        let hits: Vec<bool> = top_k
            .iter()
            .map(|id| relevant_set.contains(id.as_str()))
            .collect();

        // Recall@k
        let recall = if relevant_set.is_empty() {
            0.0
        } else {
            hits.iter().filter(|&&h| h).count() as f64 / relevant_set.len() as f64
        };

        // Precision@k
        let precision = if k == 0 {
            0.0
        } else {
            hits.iter().filter(|&&h| h).count() as f64 / k as f64
        };

        // MRR (Mean Reciprocal Rank)
        let mrr = hits
            .iter()
            .position(|&h| h)
            .map(|pos| 1.0 / (pos + 1) as f64)
            .unwrap_or(0.0);

        // nDCG with graded relevance
        let grade_map: HashMap<&str, i32> = judgments
            .iter()
            .map(|j| (j.doc_id.as_str(), j.grade))
            .collect();

        let dcg: f64 = top_k
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let gain = *grade_map.get(id.as_str()).unwrap_or(&0) as f64;
                gain / (i as f64 + 2.0).log2()
            })
            .sum();

        let mut ideal_gains: Vec<f64> = judgments.iter().map(|j| j.grade as f64).collect();
        ideal_gains.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let ideal_dcg: f64 = ideal_gains
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &gain)| gain / (i as f64 + 2.0).log2())
            .sum();

        let ndcg = if ideal_dcg > 0.0 {
            dcg / ideal_dcg
        } else {
            0.0
        };

        QueryMetrics {
            recall,
            precision,
            mrr,
            ndcg,
        }
    }

    /// Average metrics across multiple queries.
    pub fn aggregate(queries: &[QueryMetrics]) -> AggregateMetrics {
        let n = queries.len() as f64;
        if n == 0.0 {
            return AggregateMetrics::default();
        }
        AggregateMetrics {
            recall_at_k: queries.iter().map(|q| q.recall).sum::<f64>() / n,
            precision_at_k: queries.iter().map(|q| q.precision).sum::<f64>() / n,
            mrr: queries.iter().map(|q| q.mrr).sum::<f64>() / n,
            ndcg: queries.iter().map(|q| q.ndcg).sum::<f64>() / n,
        }
    }
}
