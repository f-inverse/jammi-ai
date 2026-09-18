//! Typed status enums for all catalog tables.
//!
//! Each enum implements `Display` (for SQL storage) and `FromStr` (for SQL retrieval).
//! This replaces scattered string literals like `"ready"`, `"queued"`, `"completed"`
//! with compile-time checked variants.

use std::fmt;
use std::str::FromStr;

use strum::VariantArray;

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
/// `Running` are non-terminal; `Completed`, `Failed` and `Cancelled` are
/// terminal — the states [`crate::catalog::jobs_repo::JobRecord::is_terminal`]
/// and the retention age-predicate ([`crate::catalog::model_repo`]'s
/// `REFERENCE_EDGES`) both key on. `Failed` and `Cancelled` are additionally
/// TERMINAL-UNSUCCESSFUL ([`Self::is_terminal_unsuccessful`]): the job did not
/// produce its result, because it could not (`Failed`) or because it was asked
/// not to (`Cancelled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, VariantArray)]
pub enum JobStatus {
    /// Job created, waiting to be claimed.
    Queued,
    /// Claimed and executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished unsuccessfully (error, divergence, lease exhaustion).
    Failed,
    /// Stopped because a cancel was requested: a queued job retired before
    /// any worker claimed it, or a running one that honoured the request at a
    /// checkpoint.
    Cancelled,
}

impl JobStatus {
    /// Whether this status is terminal (`Completed`, `Failed` or `Cancelled`)
    /// — a row in any of them accepts no further lease-guarded write and is eligible
    /// for retention once past `[jobs] retention_days`.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Whether this status is terminal AND unsuccessful (`Failed`) — DERIVED
    /// from [`Self::is_terminal`] rather than naming `Failed` a second time,
    /// so every future terminal-unsuccessful status this vocabulary ever
    /// gains (e.g. a future `Cancelled`) is picked up here with no second
    /// edit. The one predicate every hand-enumerated terminality decision in
    /// this crate (and the Python client's mirror) derives from — never a
    /// literal string compare.
    pub fn is_terminal_unsuccessful(&self) -> bool {
        self.is_terminal() && !matches!(self, Self::Completed)
    }

    /// Every status, in declaration (lifecycle) order — the one list the
    /// SQL-side helpers below derive their literals from, so a status added
    /// here is reflected in every `status IN (...)` predicate without a
    /// second hand-typed vocabulary. `#[derive(VariantArray)]` (`strum`)
    /// reads this enum's own variant list at macro-expansion time, so a
    /// status added above and covered by every exhaustive match on `Self`
    /// (e.g. [`Display`](fmt::Display) below) is in `ALL` automatically —
    /// unlike a hand-typed array, which the compiler cannot check for
    /// completeness against the variant set.
    pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;

    /// The comma-joined, single-quoted SQL literal list of every TERMINAL
    /// status (`'completed', 'failed'`), for a `status IN (...)` predicate —
    /// rendered from [`Self::ALL`] and [`Self::is_terminal`], never typed
    /// by hand at a query site.
    pub fn terminal_sql_list() -> String {
        Self::sql_list(|s| s.is_terminal())
    }

    /// The comma-joined, single-quoted SQL literal list of every
    /// NON-terminal status (`'queued', 'running'`) — the rows a cancel
    /// request can still reach.
    pub fn non_terminal_sql_list() -> String {
        Self::sql_list(|s| !s.is_terminal())
    }

    fn sql_list(keep: impl Fn(&JobStatus) -> bool) -> String {
        Self::ALL
            .iter()
            .filter(|s| keep(s))
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl FromStr for JobStatus {
    type Err = JammiError;
    /// The inverse of [`Display`](fmt::Display) over [`Self::ALL`], so the
    /// vocabulary is spelled once.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|status| status.to_string() == s)
            .ok_or_else(|| JammiError::Catalog(format!("Unknown job status: '{s}'")))
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

    /// Iterates the DERIVED inventory ([`JobStatus::ALL`]), not a
    /// hand-typed literal list here: a status added to the enum and given a
    /// `Display`/`FromStr` spelling (required to compile) is exercised by
    /// this loop automatically, with no second edit to this test.
    #[test]
    fn job_status_round_trips_through_display_and_from_str() {
        for status in JobStatus::ALL {
            let rendered = status.to_string();
            let parsed = JobStatus::from_str(&rendered).expect("canonical status parses");
            assert_eq!(
                &parsed, status,
                "round-trip must be identity for {status:?}"
            );
        }
        assert_eq!(
            JobStatus::ALL.len(),
            5,
            "every declared JobStatus variant is in ALL"
        );
        assert_eq!(JobStatus::Failed.to_string(), "failed");
        assert_eq!(JobStatus::Cancelled.to_string(), "cancelled");
        assert!(JobStatus::from_str("canceled").is_err());
    }

    #[test]
    fn job_status_terminality_matches_the_retention_predicate() {
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(JobStatus::Failed.is_terminal_unsuccessful());
        assert!(JobStatus::Cancelled.is_terminal_unsuccessful());
        assert!(!JobStatus::Completed.is_terminal_unsuccessful());
    }

    #[test]
    fn terminal_unsuccessful_is_terminal_minus_completed() {
        // Quantified over the WHOLE vocabulary (`JobStatus::ALL`), not a
        // hand-picked subset. Mutation executed (applied to
        // `is_terminal_unsuccessful`'s body, run, reverted): changed
        // `self.is_terminal() && !matches!(self, Self::Completed)` to bare
        // `self.is_terminal()` (dropping the `Completed` exclusion) -> RED,
        // first line: "assertion `left == right` failed: Completed:
        // terminal-unsuccessful must equal terminal-minus-completed
        //   left: true
        //  right: false".
        for status in JobStatus::ALL {
            let expected = status.is_terminal() && *status != JobStatus::Completed;
            assert_eq!(
                status.is_terminal_unsuccessful(),
                expected,
                "{status:?}: terminal-unsuccessful must equal terminal-minus-completed"
            );
        }
        assert!(JobStatus::Failed.is_terminal_unsuccessful());
        assert!(!JobStatus::Completed.is_terminal_unsuccessful());
        assert!(!JobStatus::Queued.is_terminal_unsuccessful());
        assert!(!JobStatus::Running.is_terminal_unsuccessful());
    }

    #[test]
    fn terminal_sql_list_and_non_terminal_sql_list_partition_all() {
        // ALL must split exactly between the two lists with no overlap and
        // no omission — derived from `JobStatus::ALL`, so a status added to
        // `ALL` without a `Display` impl breaking compilation is still
        // caught here if it were ever added to neither predicate.
        let terminal = JobStatus::terminal_sql_list();
        let non_terminal = JobStatus::non_terminal_sql_list();
        for status in JobStatus::ALL {
            let literal = format!("'{status}'");
            if status.is_terminal() {
                assert!(
                    terminal.contains(&literal),
                    "{status:?} must be in terminal_sql_list"
                );
                assert!(!non_terminal.contains(&literal));
            } else {
                assert!(
                    non_terminal.contains(&literal),
                    "{status:?} must be in non_terminal_sql_list"
                );
                assert!(!terminal.contains(&literal));
            }
        }
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
