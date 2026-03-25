//! Classification metrics: accuracy, macro F1, per-class precision/recall.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Per-class precision, recall, and F1.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Result of classification evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub accuracy: f64,
    pub f1: f64,
    pub per_class: HashMap<String, ClassMetrics>,
}

pub struct ClassificationMetrics;

impl ClassificationMetrics {
    /// Compute accuracy, macro F1, and per-class metrics from predicted and actual labels.
    pub fn compute(predicted: &[String], actual: &[String]) -> ClassificationResult {
        let total = predicted.len().max(1);
        let accuracy =
            predicted.iter().zip(actual).filter(|(p, a)| p == a).count() as f64 / total as f64;

        let classes: HashSet<&str> = actual.iter().map(|s| s.as_str()).collect();
        let per_class: HashMap<String, ClassMetrics> = classes
            .iter()
            .map(|&cls| {
                let tp = predicted
                    .iter()
                    .zip(actual)
                    .filter(|(p, a)| p.as_str() == cls && a.as_str() == cls)
                    .count();
                let fp = predicted
                    .iter()
                    .zip(actual)
                    .filter(|(p, a)| p.as_str() == cls && a.as_str() != cls)
                    .count();
                let fn_ = predicted
                    .iter()
                    .zip(actual)
                    .filter(|(p, a)| p.as_str() != cls && a.as_str() == cls)
                    .count();

                let prec = if tp + fp > 0 {
                    tp as f64 / (tp + fp) as f64
                } else {
                    0.0
                };
                let rec = if tp + fn_ > 0 {
                    tp as f64 / (tp + fn_) as f64
                } else {
                    0.0
                };
                let f1 = if prec + rec > 0.0 {
                    2.0 * prec * rec / (prec + rec)
                } else {
                    0.0
                };

                (
                    cls.to_string(),
                    ClassMetrics {
                        precision: prec,
                        recall: rec,
                        f1,
                    },
                )
            })
            .collect();

        let macro_f1 = if per_class.is_empty() {
            0.0
        } else {
            per_class.values().map(|m| m.f1).sum::<f64>() / per_class.len() as f64
        };

        ClassificationResult {
            accuracy,
            f1: macro_f1,
            per_class,
        }
    }
}
