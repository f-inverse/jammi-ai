//! Typed status enums for all catalog tables.
//!
//! Each enum implements `Display` (for SQL storage) and `FromStr` (for SQL retrieval).
//! This replaces scattered string literals like `"ready"`, `"queued"`, `"completed"`
//! with compile-time checked variants.

use std::fmt;
use std::str::FromStr;

use crate::error::JammiError;

/// Status of a result table (Parquet-backed embedding/inference output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultTableStatus {
    /// Table is being built (Parquet/index in progress).
    Building,
    /// Table is complete and queryable.
    Ready,
    /// Build failed or recovery detected corruption.
    Failed,
}

impl fmt::Display for ResultTableStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Building => write!(f, "building"),
            Self::Ready => write!(f, "ready"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl FromStr for ResultTableStatus {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "building" => Ok(Self::Building),
            "ready" => Ok(Self::Ready),
            "failed" => Ok(Self::Failed),
            other => Err(JammiError::Catalog(format!(
                "Unknown result table status: '{other}'"
            ))),
        }
    }
}

/// Status of a fine-tune job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FineTuneJobStatus {
    /// Job created, waiting to start.
    Queued,
    /// Training in progress.
    Running,
    /// Training completed successfully.
    Completed,
    /// Training failed (divergence, error, etc.).
    Failed,
}

impl fmt::Display for FineTuneJobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl FromStr for FineTuneJobStatus {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" | "pending" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(JammiError::Catalog(format!(
                "Unknown fine-tune job status: '{other}'"
            ))),
        }
    }
}

/// Status of an eval run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalRunStatus {
    /// Evaluation completed successfully.
    Completed,
}

impl fmt::Display for EvalRunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
        }
    }
}

impl FromStr for EvalRunStatus {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "completed" => Ok(Self::Completed),
            other => Err(JammiError::Catalog(format!(
                "Unknown eval run status: '{other}'"
            ))),
        }
    }
}

/// Status of a registered model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// Model registered in catalog but not yet loaded.
    Registered,
    /// Model loaded into memory.
    Loaded,
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registered => write!(f, "registered"),
            Self::Loaded => write!(f, "loaded"),
        }
    }
}

impl FromStr for ModelStatus {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "registered" | "available" => Ok(Self::Registered),
            "loaded" => Ok(Self::Loaded),
            other => Err(JammiError::Catalog(format!(
                "Unknown model status: '{other}'"
            ))),
        }
    }
}
