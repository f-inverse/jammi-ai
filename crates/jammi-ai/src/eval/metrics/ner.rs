//! Entity-level NER evaluation metrics.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

/// A predicted or gold entity span for evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvalEntity {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

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
    /// Each input is a list of (entities_for_row) across all rows.
    pub fn compute(predicted: &[Vec<EvalEntity>], gold: &[Vec<EvalEntity>]) -> Self {
        assert_eq!(predicted.len(), gold.len());

        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut total_fn = 0usize;

        let mut type_tp: HashMap<String, usize> = HashMap::new();
        let mut type_fp: HashMap<String, usize> = HashMap::new();
        let mut type_fn: HashMap<String, usize> = HashMap::new();

        for (pred_row, gold_row) in predicted.iter().zip(gold.iter()) {
            let pred_set: HashSet<&EvalEntity> = pred_row.iter().collect();
            let gold_set: HashSet<&EvalEntity> = gold_row.iter().collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_match() {
        let gold = vec![vec![
            EvalEntity {
                label: "PER".into(),
                start: 0,
                end: 5,
            },
            EvalEntity {
                label: "ORG".into(),
                start: 10,
                end: 15,
            },
        ]];
        let pred = gold.clone();
        let m = NerMetrics::compute(&pred, &gold);
        assert!((m.precision - 1.0).abs() < f64::EPSILON);
        assert!((m.recall - 1.0).abs() < f64::EPSILON);
        assert!((m.f1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_predictions() {
        let gold = vec![vec![EvalEntity {
            label: "PER".into(),
            start: 0,
            end: 5,
        }]];
        let pred = vec![vec![]];
        let m = NerMetrics::compute(&pred, &gold);
        assert!((m.precision).abs() < f64::EPSILON);
        assert!((m.recall).abs() < f64::EPSILON);
        assert!((m.f1).abs() < f64::EPSILON);
    }

    #[test]
    fn partial_overlap() {
        let gold = vec![vec![
            EvalEntity {
                label: "PER".into(),
                start: 0,
                end: 5,
            },
            EvalEntity {
                label: "LOC".into(),
                start: 10,
                end: 15,
            },
        ]];
        let pred = vec![vec![
            EvalEntity {
                label: "PER".into(),
                start: 0,
                end: 5,
            },
            EvalEntity {
                label: "ORG".into(),
                start: 20,
                end: 25,
            },
        ]];
        let m = NerMetrics::compute(&pred, &gold);
        // tp=1, fp=1, fn=1 => P=0.5, R=0.5, F1=0.5
        assert!((m.precision - 0.5).abs() < f64::EPSILON);
        assert!((m.recall - 0.5).abs() < f64::EPSILON);
        assert!((m.f1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn per_type_metrics() {
        let gold = vec![vec![
            EvalEntity {
                label: "PER".into(),
                start: 0,
                end: 5,
            },
            EvalEntity {
                label: "PER".into(),
                start: 10,
                end: 15,
            },
            EvalEntity {
                label: "LOC".into(),
                start: 20,
                end: 25,
            },
        ]];
        let pred = vec![vec![
            EvalEntity {
                label: "PER".into(),
                start: 0,
                end: 5,
            },
            EvalEntity {
                label: "LOC".into(),
                start: 20,
                end: 25,
            },
        ]];
        let m = NerMetrics::compute(&pred, &gold);
        // PER: tp=1, fp=0, fn=1 => P=1.0, R=0.5, F1=0.667
        let per = m.per_type.get("PER").unwrap();
        assert!((per.precision - 1.0).abs() < f64::EPSILON);
        assert!((per.recall - 0.5).abs() < f64::EPSILON);
        assert!(per.support == 2);
        // LOC: tp=1, fp=0, fn=0 => P=1.0, R=1.0, F1=1.0
        let loc = m.per_type.get("LOC").unwrap();
        assert!((loc.precision - 1.0).abs() < f64::EPSILON);
        assert!((loc.recall - 1.0).abs() < f64::EPSILON);
        assert!(loc.support == 1);
    }

    #[test]
    fn empty_inputs() {
        let m = NerMetrics::compute(&[], &[]);
        assert!((m.precision).abs() < f64::EPSILON);
        assert!((m.recall).abs() < f64::EPSILON);
        assert!((m.f1).abs() < f64::EPSILON);
        assert!(m.per_type.is_empty());
    }
}
