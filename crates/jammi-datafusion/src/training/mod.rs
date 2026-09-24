//! The training stage: run a claimed training job as a physical stage.
//!
//! [`exec::TrainingExec`] is one task that runs a [`exec::TrainingJob`] —
//! a job's coordinates, never its spec — through the
//! [`exec::TrainingRunner`] it was bound to, and yields the job's outcome as
//! one row. The stage is placeable on another process through its wire form
//! ([`wire`]) and the runner that process binds: the process whose executor
//! holds the job's device runs it there.

pub mod exec;
pub mod wire;
