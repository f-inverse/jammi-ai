//! Evaluation vocabulary: the transport-neutral report shapes and task/shape
//! enums every eval consumer shares.
//!
//! The report types and the [`EvalTask`] / [`EvalCalibrationShape`] selectors
//! are the terminal results an eval verb returns and the inputs that select
//! which metric family scores a golden source. They carry no engine state, so
//! they live on the wire substrate: the embedded runner builds them, the gRPC
//! converters encode/decode them, and the data-plane client decodes them — none
//! reimplements the shape.

pub mod report;

pub use report::{
    AggregateDelta, CalibrationAggregate, CalibrationEvalReport, CalibrationPrediction,
    CohortCalibration, CompareEvalReport, DeltaSignificance, EmbeddingEvalReport,
    InferenceAggregate, InferenceEvalReport, MetricDelta, MetricSignificance, PerQueryRecord,
    PerRecordCalibration, PerRecordPrediction, TableEvalReport,
};

use serde::{Deserialize, Serialize};

/// The predictive output shape an `eval_calibration` golden source carries.
///
/// A calibration golden source pairs a held-out predictive distribution with
/// its realised outcome. The shape selects which columns the runner reads and
/// which proper-score / diagnostic family scores them — exactly the way
/// [`EvalTask`] selects the inference metric. Both families are scored on the
/// same yardstick (CRPS / NLL / PIT-calibration / sharpness), so a parametric
/// (NP3) and an ensemble (NP4) predictor are directly comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalCalibrationShape {
    /// A parametric predictive `Normal(mean, sd)`: the source carries `mean`,
    /// `sd`, and `outcome` numeric columns.
    Gaussian,
    /// An ensemble of predictive draws: the source carries a `draws` column (a
    /// JSON array of numbers per row) and an `outcome` numeric column.
    Sample,
}

impl std::fmt::Display for EvalCalibrationShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gaussian => write!(f, "calibration_gaussian"),
            Self::Sample => write!(f, "calibration_sample"),
        }
    }
}

impl std::str::FromStr for EvalCalibrationShape {
    type Err = jammi_db::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "gaussian" => Ok(Self::Gaussian),
            "sample" => Ok(Self::Sample),
            other => Err(jammi_db::error::JammiError::Other(format!(
                "Unknown calibration shape '{other}'. Expected: gaussian, sample"
            ))),
        }
    }
}

/// Task type for inference evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalTask {
    Classification,
    Ner,
}

impl std::fmt::Display for EvalTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Classification => write!(f, "classification"),
            Self::Ner => write!(f, "ner"),
        }
    }
}

impl std::str::FromStr for EvalTask {
    type Err = jammi_db::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "classification" => Ok(Self::Classification),
            "ner" => Ok(Self::Ner),
            other => Err(jammi_db::error::JammiError::Other(format!(
                "Unknown eval task '{other}'. Expected: classification, ner"
            ))),
        }
    }
}
