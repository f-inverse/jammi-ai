//! Evaluation: task-specific metrics against golden datasets.
//!
//! This module computes retrieval (recall@k, MRR, nDCG) and classification
//! (accuracy, F1) metrics. Results are recorded in the catalog for comparison
//! and for Phase 13's ExperimentRunner.

pub mod compare;
pub mod golden;
pub mod metrics;
pub mod runner;

use serde::{Deserialize, Serialize};

/// Task type for inference evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalTask {
    Classification,
}

impl std::fmt::Display for EvalTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Classification => write!(f, "classification"),
        }
    }
}

impl std::str::FromStr for EvalTask {
    type Err = jammi_engine::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "classification" => Ok(Self::Classification),
            other => Err(jammi_engine::error::JammiError::Other(format!(
                "Unknown eval task '{other}'. Expected: classification"
            ))),
        }
    }
}
