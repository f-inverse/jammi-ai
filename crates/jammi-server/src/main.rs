//! `jammi-server` — OSS server binary.
//!
//! Parses a single optional `--config` flag, loads the workspace
//! [`JammiConfig`], initialises tracing per the resolved logging
//! settings, binds the server's listeners, announces the resolved bind
//! addresses on stdout, and serves until a shutdown signal arrives.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use jammi_db::config::JammiConfig;
use jammi_server::runtime::OssServer;
use jammi_server::telemetry::init_tracing;

/// CLI for the OSS `jammi-server`.
///
/// Only the `--config` flag is exposed today. Everything else (bind
/// addresses, logging level, catalog backend, broker backend) is
/// driven from the resolved [`JammiConfig`]. Environment variables
/// override individual fields per the engine's `JAMMI_*` convention.
#[derive(Parser)]
#[command(
    name = "jammi-server",
    version,
    about = "OSS Jammi server: Arrow Flight SQL, CatalogService, TriggerService, and /healthz /readyz /metrics"
)]
struct Cli {
    /// Path to the configuration file. Falls back to `JAMMI_CONFIG`,
    /// `./jammi.toml`, and the platform-default config directory in
    /// that order. When no file is found the workspace defaults are
    /// used (SQLite catalog + in-memory broker under `.jammi/`).
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    let config = match JammiConfig::load(cli.config.as_deref()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("jammi-server: failed to load config: {e}");
            return ExitCode::FAILURE;
        }
    };

    init_tracing(&config);

    let server = match OssServer::new(config).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "failed to construct OSS server");
            return ExitCode::FAILURE;
        }
    };

    // Bind both listeners eagerly, then announce the ACTUAL bound addresses on
    // stdout before serving. When the config requests an ephemeral `:0` port,
    // this line is the only channel a supervisor — or a subprocess-driving test
    // harness — has to learn the port the kernel actually assigned. The format
    // is deliberately fixed and machine-parseable:
    // `jammi-server listening flight=<addr> health=<addr>`. `println!` flushes at
    // the newline (stdout is a `LineWriter`), so the line is visible immediately
    // even when stdout is a pipe rather than a terminal.
    let bound = match server.bind().await {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "failed to bind OSS server listeners");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "jammi-server listening flight={} health={}",
        bound.flight_addr(),
        bound.health_addr()
    );

    if let Err(e) = bound.serve().await {
        tracing::error!(error = %e, "OSS server exited with error");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
