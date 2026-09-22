//! The compute plane under a leg: the rungs of every workload above the
//! in-process ones. `placed` hosts the plane's roles — scheduler, executor,
//! client — and `shape-d` runs the deployed topology's exact role configs
//! as a fleet of `jammi-server` processes on Postgres and an S3-class store,
//! or joins one already running across hosts. Every leg records where its
//! work ran and refuses to be filed as placed when it ran in the submitting
//! process.
//!
//! Built only with the `plane` feature; without it every rung above the
//! in-process ones refuses by name.

use std::path::PathBuf;

/// Where a rung above the in-process ones runs.
#[derive(Debug, Clone, Default)]
pub struct PlaneParams {
    /// The `jammi-server` binary a one-host fleet is spawned from.
    pub server_bin: Option<PathBuf>,
    /// A shape-d fleet already running across hosts: its query tier's gRPC
    /// address the job is submitted through.
    pub query_addr: Option<String>,
    /// The repository checkout whose committed shape-d role configs a
    /// spawned shape-d fleet runs; this workspace when unset.
    pub repo_root: Option<PathBuf>,
    /// For a joined fleet: the URL every member reads the leg's rows from
    /// (a JSONL of the training rows, a parquet of the corpus), already put
    /// where the fleet's hosts can read it. A spawned fleet reads a file
    /// under the work dir.
    pub source_url: Option<String>,
}

impl PlaneParams {
    /// What was given, for a refusal that names it.
    pub fn summary(&self) -> String {
        format!(
            "--server-bin {:?}, --query-addr {:?}, --repo-root {:?}, --source-url {:?}",
            self.server_bin, self.query_addr, self.repo_root, self.source_url
        )
    }
}

#[cfg(feature = "plane")]
pub mod encode_host;
#[cfg(feature = "plane")]
pub mod fleet;
#[cfg(feature = "plane")]
pub mod train_run;

#[cfg(not(feature = "plane"))]
pub mod train_run {
    use crate::finetune_run::{FinetuneRunParams, RunContext, TrainedRun};

    /// The `placed` and `shape-d` train-run rungs need the compute plane.
    pub fn train_on_fleet(
        params: &FinetuneRunParams,
        _ctx: &RunContext,
    ) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
        Err(format!(
            "finetune-run: --rung {} needs the compute plane ({} were given): build jammi-bench \
             with --features plane",
            params.rung.as_str(),
            params.plane.summary(),
        )
        .into())
    }
}
