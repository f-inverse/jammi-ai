//! `jammi-server` — OSS server binary.
//!
//! Parses a single optional `--config` flag, loads the workspace
//! [`JammiConfig`], initialises tracing per the resolved logging
//! settings, and hands control to [`OssServer::run`].

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use jammi_db::config::{JammiConfig, LogFormat};
use jammi_server::runtime::OssServer;
use tracing_subscriber::EnvFilter;

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
    about = "OSS Jammi server: Arrow Flight SQL, SessionService, TriggerService, and /healthz /readyz /metrics"
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

    if let Err(e) = server.run().await {
        tracing::error!(error = %e, "OSS server exited with error");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Configure the global tracing subscriber from the engine config.
/// Honours `RUST_LOG` when set; otherwise falls back to the config's
/// `logging.level`. Format is JSON when `logging.format = "json"`,
/// otherwise the default text layer.
fn init_tracing(config: &JammiConfig) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.logging.level.clone()));

    match config.logging.format {
        LogFormat::Json => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .init();
        }
        LogFormat::Text => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }
}
