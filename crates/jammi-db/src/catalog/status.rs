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
        f.pad(match self {
            Self::Building => "building",
            Self::Ready => "ready",
            Self::Failed => "failed",
        })
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

/// Lifecycle state of a model artifact (`model_artifacts.state`) — the bytes
/// of one published bundle under `models/`.
///
/// `Staged` is written before the bundle's first byte and names the writer;
/// `Published` is committed by the finalize transaction that attaches the
/// first `models` row; `Reclaiming` is committed only by the compare-and-set
/// that licenses the byte delete ([`crate::catalog::artifact_repo`]). A
/// `models` row can attach to a `Published` artifact and to no other state,
/// so a `Reclaiming` artifact is never a reuse candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, VariantArray)]
pub enum ArtifactState {
    /// The row exists; the bundle's bytes are being written by its stager.
    Staged,
    /// The bundle is complete and at least one finalize has committed it.
    Published,
    /// The byte delete is licensed and in progress.
    Reclaiming,
}

impl ArtifactState {
    /// The spelling stored in `model_artifacts.state`.
    pub const fn as_db_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::Published => "published",
            Self::Reclaiming => "reclaiming",
        }
    }
}

impl fmt::Display for ArtifactState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_db_str())
    }
}

impl FromStr for ArtifactState {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::VARIANTS
            .iter()
            .copied()
            .find(|state| state.as_db_str() == s)
            .ok_or_else(|| JammiError::Catalog(format!("Unknown model artifact state: '{s}'")))
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
        f.pad(match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        })
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

/// What a job runs (`jobs.kind`). A training kind registers a model under
/// the job's output model id; a compute kind writes a result table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, VariantArray)]
pub enum JobKind {
    /// A text, image or audio fine-tune.
    FineTune,
    /// A fine-tune supervised by a graph's edges.
    GraphFineTune,
    /// An amortized context predictor.
    ContextPredictor,
    /// A k-nearest-neighbour graph over an embedding table.
    NeighborGraph,
    /// Embedding propagation over a graph.
    Propagate,
    /// Structural node embeddings of a graph.
    GraphStructure,
    /// A point-in-time join.
    AsofJoin,
    /// A lexical (BM25) index.
    LexicalIndex,
    /// Embedding generation over a source.
    Embedding,
    /// Model inference over a source.
    Infer,
}

impl JobKind {
    /// Every kind, in declaration order — the vocabulary a worker's `kinds`
    /// selection is drawn from.
    pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;

    /// Whether this kind trains a model rather than computing a result table.
    pub fn trains_a_model(self) -> bool {
        matches!(
            self,
            Self::FineTune | Self::GraphFineTune | Self::ContextPredictor
        )
    }
}

impl fmt::Display for JobKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::FineTune => "fine_tune",
            Self::GraphFineTune => "graph_fine_tune",
            Self::ContextPredictor => "context_predictor",
            Self::NeighborGraph => "neighbor_graph",
            Self::Propagate => "propagate",
            Self::GraphStructure => "graph_structure",
            Self::AsofJoin => "asof_join",
            Self::LexicalIndex => "lexical_index",
            Self::Embedding => "embedding",
            Self::Infer => "infer",
        })
    }
}

impl FromStr for JobKind {
    type Err = JammiError;
    /// The inverse of [`Display`](fmt::Display) over [`Self::ALL`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.to_string() == s)
            .ok_or_else(|| JammiError::InvalidArgument {
                argument: "kind".into(),
                reason: format!(
                    "unknown job kind '{s}'; the kinds are {}",
                    Self::ALL
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })
    }
}

/// What a running job is doing (`jobs.progress_phase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, VariantArray)]
pub enum JobPhase {
    /// Assembling its inputs (a training set); progress counts the rows read.
    Assembling,
    /// Training a model; progress counts the optimizer steps taken.
    Training,
    /// Computing a result table; progress counts the rows read.
    Computing,
    /// Publishing the model or the table.
    Publishing,
}

impl JobPhase {
    /// Every phase, in the order a job passes through them.
    pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;
}

impl fmt::Display for JobPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Assembling => "assembling",
            Self::Training => "training",
            Self::Computing => "computing",
            Self::Publishing => "publishing",
        })
    }
}

impl FromStr for JobPhase {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|phase| phase.to_string() == s)
            .ok_or_else(|| JammiError::Catalog(format!("Unknown job phase: '{s}'")))
    }
}

/// The job kinds one worker claims, stored as the comma-joined spelling of
/// each (`workers.kinds`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JobKinds(Vec<JobKind>);

impl JobKinds {
    /// The set of `kinds`, in the order given.
    pub fn new(kinds: Vec<JobKind>) -> Self {
        Self(kinds)
    }

    /// Every kind.
    pub fn all() -> Self {
        Self(JobKind::ALL.to_vec())
    }

    /// Whether the set holds `kind`.
    pub fn contains(&self, kind: JobKind) -> bool {
        self.0.contains(&kind)
    }

    /// The kinds, in the order given.
    pub fn as_slice(&self) -> &[JobKind] {
        &self.0
    }

    /// Whether the set is empty: a claim loop that runs but claims nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for JobKinds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined = self
            .0
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        f.pad(&joined)
    }
}

impl FromStr for JobKinds {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(JobKind::from_str)
            .collect::<Result<_, _>>()
            .map(Self)
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
        f.pad(match self {
            Self::Completed => "completed",
        })
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
        f.pad(match self {
            Self::Registered => "registered",
            Self::Loaded => "loaded",
        })
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
        f.pad(match self {
            Self::Queued => "queued",
            Self::Inline => "inline",
        })
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
    fn job_kind_round_trips_and_names_what_it_trains() {
        for kind in JobKind::ALL {
            assert_eq!(JobKind::from_str(&kind.to_string()).unwrap(), *kind);
        }
        let training: Vec<_> = JobKind::ALL.iter().filter(|k| k.trains_a_model()).collect();
        assert_eq!(
            training,
            [
                &JobKind::FineTune,
                &JobKind::GraphFineTune,
                &JobKind::ContextPredictor
            ]
        );
        assert!(JobKind::from_str("finetune").is_err());
    }

    #[test]
    fn job_kinds_round_trip_through_their_column_spelling() {
        let kinds = JobKinds::new(vec![JobKind::FineTune, JobKind::Embedding]);
        assert_eq!(kinds.to_string(), "fine_tune,embedding");
        assert_eq!(JobKinds::from_str("fine_tune,embedding").unwrap(), kinds);
        assert!(JobKinds::from_str("").unwrap().is_empty());
        assert_eq!(
            JobKinds::from_str(&JobKinds::all().to_string()).unwrap(),
            JobKinds::all()
        );
        assert!(JobKinds::from_str("fine_tune,bogus").is_err());
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
