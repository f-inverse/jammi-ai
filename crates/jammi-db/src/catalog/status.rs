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

/// Status of a job (`jobs.status`) — every kind-agnostic unit of work the
/// catalog's claim/lease/reclaim machinery drives
/// ([`crate::catalog::jobs_repo`]), training and compute alike. `Queued` and
/// `Running` are non-terminal; `Completed` and `Failed` are terminal — the
/// two states [`crate::catalog::jobs_repo::JobRecord::is_terminal`] and the
/// retention age-predicate ([`crate::catalog::model_repo`]'s `REFERENCE_EDGES`)
/// both key on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Job created, waiting to be claimed.
    Queued,
    /// Claimed and executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished unsuccessfully (error, divergence, lease exhaustion, cancel).
    Failed,
}

impl JobStatus {
    /// Whether this status is terminal (`Completed` or `Failed`) — a row in
    /// either state accepts no further lease-guarded write and is eligible
    /// for retention once past `[jobs] retention_days`.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl FromStr for JobStatus {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(JammiError::Catalog(format!(
                "Unknown job status: '{other}'"
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
            "registered" => Ok(Self::Registered),
            "loaded" => Ok(Self::Loaded),
            other => Err(JammiError::Catalog(format!(
                "Unknown model status: '{other}'"
            ))),
        }
    }
}

/// Execution mode of a job (`jobs.execution`): whether the poll loop's
/// `claim_next` may pick it up (`Queued`) or it was claimed exactly once, by
/// id, inside the submitting call itself (`Inline`) — an inline row is never
/// selected by `claim_next`'s `WHERE execution = 'queued'` predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobExecution {
    /// Claimable by the poll loop.
    Queued,
    /// Claimed once, by id, in the submitting call; invisible to the poll loop.
    Inline,
}

impl fmt::Display for JobExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Inline => write!(f, "inline"),
        }
    }
}

impl FromStr for JobExecution {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "inline" => Ok(Self::Inline),
            other => Err(JammiError::Catalog(format!(
                "Unknown job execution mode: '{other}'"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_status_round_trips_through_display_and_from_str() {
        for status in [
            JobStatus::Queued,
            JobStatus::Running,
            JobStatus::Completed,
            JobStatus::Failed,
        ] {
            let rendered = status.to_string();
            let parsed = JobStatus::from_str(&rendered).expect("canonical status parses");
            assert_eq!(parsed, status, "round-trip must be identity for {status:?}");
        }
        assert_eq!(JobStatus::Failed.to_string(), "failed");
        assert!(JobStatus::from_str("cancelled").is_err());
    }

    #[test]
    fn job_status_terminality_matches_the_retention_predicate() {
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
    }

    #[test]
    fn job_execution_round_trips_through_display_and_from_str() {
        for exec in [JobExecution::Queued, JobExecution::Inline] {
            let rendered = exec.to_string();
            let parsed = JobExecution::from_str(&rendered).expect("canonical execution parses");
            assert_eq!(parsed, exec, "round-trip must be identity for {exec:?}");
        }
        assert!(JobExecution::from_str("async").is_err());
    }

    #[test]
    fn model_status_round_trips_through_display_and_from_str() {
        for status in [ModelStatus::Registered, ModelStatus::Loaded] {
            let rendered = status.to_string();
            let parsed = ModelStatus::from_str(&rendered).expect("canonical status parses");
            assert_eq!(parsed, status, "round-trip must be identity for {status:?}");
        }
        assert_eq!(ModelStatus::Loaded.to_string(), "loaded");
        assert!(ModelStatus::from_str("available").is_err());
    }
}
