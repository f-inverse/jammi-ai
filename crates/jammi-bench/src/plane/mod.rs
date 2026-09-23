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

    /// The same plane, as the flags a child leg process is given.
    pub fn child_args(&self) -> Vec<std::ffi::OsString> {
        let paths = [
            ("--server-bin", &self.server_bin),
            ("--repo-root", &self.repo_root),
        ];
        let values = [
            ("--query-addr", &self.query_addr),
            ("--source-url", &self.source_url),
        ];
        paths
            .into_iter()
            .filter_map(|(flag, path)| path.as_ref().map(|p| [flag.into(), p.into()]))
            .chain(
                values
                    .into_iter()
                    .filter_map(|(flag, value)| value.as_ref().map(|v| [flag.into(), v.into()])),
            )
            .flatten()
            .collect()
    }

    /// The refusal of a rung above the in-process ones in a build without
    /// the plane.
    #[cfg(not(feature = "plane"))]
    pub fn refusal(
        &self,
        subcommand: &str,
        rung: &str,
    ) -> Box<dyn std::error::Error + Send + Sync> {
        format!(
            "{subcommand}: --rung {rung} needs the compute plane ({} were given): build jammi-bench \
             with --features plane",
            self.summary(),
        )
        .into()
    }
}

/// The plane's flags, as every rung-bearing subcommand takes them: a
/// `jammi-server` binary a fleet is spawned from on one host, or a fleet
/// already running across hosts whose query tier is at `--query-addr`.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct PlaneArgs {
    /// The `jammi-server` binary a one-host fleet is spawned from (the
    /// `placed` rung always; `shape-d` unless `--query-addr` names a fleet
    /// already running).
    #[arg(long)]
    server_bin: Option<PathBuf>,
    /// A shape-d fleet already running across hosts: its query tier's
    /// gRPC address (`host:port`) the job is submitted through. The fleet's
    /// catalog and store are the same backends this process reads.
    #[arg(long)]
    query_addr: Option<String>,
    /// The repository checkout whose committed shape-d role configs
    /// (`deploy/kubernetes/overlays/shape-d`) a spawned shape-d fleet runs;
    /// this workspace when omitted.
    #[arg(long)]
    repo_root: Option<PathBuf>,
    /// With `--query-addr`: the URL the fleet's members read the leg's rows
    /// from (the training rows as JSONL with `anchor`/`positive`/`negative`
    /// columns; the corpus as parquet), already put where they can read it.
    #[arg(long)]
    source_url: Option<String>,
}

impl From<PlaneArgs> for PlaneParams {
    fn from(args: PlaneArgs) -> Self {
        Self {
            server_bin: args.server_bin,
            query_addr: args.query_addr,
            repo_root: args.repo_root,
            source_url: args.source_url,
        }
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
        Err(params.plane.refusal("finetune-run", params.rung.as_str()))
    }
}
