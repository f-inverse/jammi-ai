//! Entity-level NER evaluation metrics with strict matching (label +
//! boundaries must match). Operates on the single [`Entity`] type from
//! [`crate::ner::types`].

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::ner::types::Entity;

/// Entity-level NER metrics: strict matching (label + boundaries must match).
#[derive(Debug, Clone, Serialize)]
pub struct NerMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub per_type: HashMap<String, TypeMetrics>,
}

/// Per-entity-type precision, recall, F1, and support count.
#[derive(Debug, Clone, Serialize)]
pub struct TypeMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl NerMetrics {
    /// Compute entity-level metrics from predicted and gold entity sets.
    ///
    /// Each input is a list of (entities_for_row) across all rows. Entity
    /// equality ignores `text` and `confidence` (see [`Entity`]) so the
    /// match condition is strict on `(label, start, end)`.
    pub fn compute(predicted: &[Vec<Entity>], gold: &[Vec<Entity>]) -> Self {
        assert_eq!(predicted.len(), gold.len());

        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut total_fn = 0usize;

        let mut type_tp: HashMap<String, usize> = HashMap::new();
        let mut type_fp: HashMap<String, usize> = HashMap::new();
        let mut type_fn: HashMap<String, usize> = HashMap::new();

        for (pred_row, gold_row) in predicted.iter().zip(gold.iter()) {
            let pred_set: HashSet<&Entity> = pred_row.iter().collect();
            let gold_set: HashSet<&Entity> = gold_row.iter().collect();

            for entity in &pred_set {
                if gold_set.contains(entity) {
                    total_tp += 1;
                    *type_tp.entry(entity.label.clone()).or_default() += 1;
                } else {
                    total_fp += 1;
                    *type_fp.entry(entity.label.clone()).or_default() += 1;
                }
            }
            for entity in &gold_set {
                if !pred_set.contains(entity) {
                    total_fn += 1;
                    *type_fn.entry(entity.label.clone()).or_default() += 1;
                }
            }
        }

        let precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };
        let recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let all_types: HashSet<&String> = type_tp
            .keys()
            .chain(type_fp.keys())
            .chain(type_fn.keys())
            .collect();

        let per_type = all_types
            .into_iter()
            .map(|t| {
                let tp = *type_tp.get(t).unwrap_or(&0);
                let fp = *type_fp.get(t).unwrap_or(&0);
                let fn_ = *type_fn.get(t).unwrap_or(&0);
                let p = if tp + fp > 0 {
                    tp as f64 / (tp + fp) as f64
                } else {
                    0.0
                };
                let r = if tp + fn_ > 0 {
                    tp as f64 / (tp + fn_) as f64
                } else {
                    0.0
                };
                let f = if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                };
                (
                    t.clone(),
                    TypeMetrics {
                        precision: p,
                        recall: r,
                        f1: f,
                        support: tp + fn_,
                    },
                )
            })
            .collect();

        NerMetrics {
            precision,
            recall,
            f1,
            per_type,
        }
    }
}
