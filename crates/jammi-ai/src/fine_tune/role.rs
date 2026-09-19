//! Who runs one attempt's training body, and therefore who may write.
//!
//! The gang's single-writer rule: the process that claimed a
//! job — the lease holder — is the ONE writer of that job's row, of its
//! durable checkpoints and of its published artifact. Every other rank of a
//! gang computes, reduces and writes nothing durable: no per-rank fragment,
//! no peer-side write into the artifact prefix, no job-row write. The two
//! types here make that a TYPE fact rather than a discipline:
//!
//! - [`LeaseHolder`](crate::fine_tune::role::LeaseHolder) is what every job-row-writing site in
//!   `crate::fine_tune::worker` takes as a REQUIRED parameter (the lease-hold
//!   registration and its `Releasing` self-release arm, the holder
//!   accounting, the acceleration-report write, every `record_failed` site,
//!   `finish_job_with_model`'s call site, the coordinator's assembly-outcome
//!   and lease-release writes). It has exactly two values — the in-process
//!   loop path and the `Peer` gang's coordinator — and no value for a rank.
//! - [`RunnerRole`](crate::fine_tune::role::RunnerRole) is what a training body runs AS: a lease holder, or a
//!   rank `>= 1` of a gang. A `Rank` carries no `LeaseHolder`, so a rank
//!   body that tried to call a job-row writer would have nothing to pass:
//!   the write is unreachable by type, not refused at run time. The
//!   trainer's own durable writes (the resume checkpoint, the epoch
//!   checkpoints) are gated on the same role
//!   (`crate::fine_tune::trainer::TrainingLoopBuilder::runner_role`).
//!
//! `W == 1` is [`LeaseHolder::LoopClaimer`](crate::fine_tune::role::LeaseHolder::LoopClaimer) on
//! every path: the single-rank run never traverses the coordinator body.

use std::fmt;

/// The lease holder of one attempt — the ONE writer of the job row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseHolder {
    /// The in-process loop path: the claimant runs the whole job in-process — a
    /// single rank, or rank 0 of an in-process `Local` gang whose other
    /// ranks are threads of the same process. Never traverses the
    /// coordinator body.
    LoopClaimer,
    /// Rank 0 of a `Peer` gang: the claimant that assembled fleet members
    /// through the coordinator body (`JobWorker::coordinate`). The same
    /// writer, one more thing to write (the assembly outcome).
    Coordinator,
}

impl fmt::Display for LeaseHolder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LoopClaimer => "loop_claimer",
            Self::Coordinator => "coordinator",
        })
    }
}

/// What one training body runs as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerRole {
    /// Rank 0, holding the lease: writes the row, the checkpoints, the
    /// artifact.
    Holder(LeaseHolder),
    /// Rank `rank >= 1` of a gang — an in-process `Local` rank, or a `Peer`
    /// member on the far end of an admitted `RunRank` session. Computes and
    /// reduces in lockstep; writes nothing durable.
    Rank { rank: u32 },
}

impl RunnerRole {
    /// The role a rank index implies when nothing names one explicitly:
    /// rank 0 is the loop claimer (the single-rank default every pre-gang
    /// caller gets), every other rank is a `Rank`.
    pub fn implied_by_rank(rank: u32) -> Self {
        if rank == 0 {
            Self::Holder(LeaseHolder::LoopClaimer)
        } else {
            Self::Rank { rank }
        }
    }

    /// This role's rank index in the gang: `0` for a holder.
    pub fn rank(self) -> u32 {
        match self {
            Self::Holder(_) => 0,
            Self::Rank { rank } => rank,
        }
    }

    /// The lease holder this role may write as — `None` for a rank, which
    /// is what makes every durable write unreachable from a rank body.
    pub fn lease_holder(self) -> Option<LeaseHolder> {
        match self {
            Self::Holder(holder) => Some(holder),
            Self::Rank { .. } => None,
        }
    }
}

impl fmt::Display for RunnerRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Holder(holder) => write!(f, "{holder}"),
            Self::Rank { rank } => write!(f, "rank_{rank}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivation table: rank 0 alone implies a holder (the loop
    /// claimer), every other rank a `Rank`; a holder's rank is 0; only a
    /// holder may write.
    #[test]
    fn rank_zero_alone_implies_a_holder_and_only_a_holder_may_write() {
        assert_eq!(
            RunnerRole::implied_by_rank(0),
            RunnerRole::Holder(LeaseHolder::LoopClaimer)
        );
        for rank in 1..5 {
            assert_eq!(RunnerRole::implied_by_rank(rank), RunnerRole::Rank { rank });
            assert_eq!(RunnerRole::Rank { rank }.rank(), rank);
            assert_eq!(RunnerRole::Rank { rank }.lease_holder(), None);
        }
        for holder in [LeaseHolder::LoopClaimer, LeaseHolder::Coordinator] {
            assert_eq!(RunnerRole::Holder(holder).rank(), 0);
            assert_eq!(RunnerRole::Holder(holder).lease_holder(), Some(holder));
        }
    }
}
