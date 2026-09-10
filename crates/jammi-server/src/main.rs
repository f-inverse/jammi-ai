//! `jammi-server` — OSS server binary.
//!
//! Two subcommands: `serve` (parses a single optional `--config` flag,
//! loads the workspace [`JammiConfig`], initialises tracing per the
//! resolved logging settings, binds the server's listeners, announces the
//! resolved bind addresses on stdout, and serves until a shutdown signal
//! arrives) and `probe` (a generic readiness check — see [`jammi_server::probe`]).
//! `serve` is the default: `jammi-server`, `jammi-server --config X`, and
//! `jammi-server serve --config X` all serve.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use jammi_db::config::JammiConfig;
use jammi_server::runtime::{validate_audit_master_key, OssServer};
use jammi_server::telemetry::init_tracing;

/// CLI for the OSS `jammi-server`.
///
/// `serve` is the implicit default subcommand: a bare `jammi-server` or
/// `jammi-server --config X` (no subcommand keyword) serves, identically to
/// `jammi-server serve` / `jammi-server serve --config X`. Once a subcommand
/// keyword IS given, the top-level flattened flags conflict with it — e.g.
/// `jammi-server --config X probe` is a parse error, since `--config` only
/// means something for `serve` and `probe` does not take it.
#[derive(Parser, Debug)]
#[command(
    name = "jammi-server",
    version,
    about = "OSS Jammi server: Arrow Flight SQL, CatalogService, TriggerService, and /healthz /readyz /metrics",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    serve: ServeArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Bind the Flight SQL + gRPC + health listeners and serve until a
    /// shutdown signal arrives. The default when no subcommand is given.
    Serve(ServeArgs),
    /// GET the server's `/readyz` endpoint once and exit 0 on HTTP 200, 1
    /// otherwise. A generic supervisor readiness check — see
    /// [`jammi_server::probe`].
    Probe(ProbeArgs),
}

#[derive(Args, Debug)]
struct ServeArgs {
    /// Path to the configuration file. Falls back to `JAMMI_CONFIG`,
    /// `./jammi.toml`, `/etc/jammi/jammi.toml`, and the platform-default
    /// config directory, in that order. When no file is found the
    /// workspace defaults are used (SQLite catalog + in-memory broker
    /// under `.jammi/`).
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ProbeArgs {
    /// Path to the configuration file, consulted only when `--url` is
    /// absent. Falls back to `JAMMI_CONFIG`, `./jammi.toml`,
    /// `/etc/jammi/jammi.toml`, and the platform-default config directory,
    /// in that order — byte-for-byte the same resolution `serve`'s
    /// `--config` runs.
    #[arg(long)]
    config: Option<PathBuf>,
    /// URL to GET. Defaults to `http://<host>:<port>/readyz`, derived from
    /// the resolved `[server] health_listen` (the SAME config resolution
    /// `serve` uses — `--config`, `JAMMI_CONFIG`, `./jammi.toml`,
    /// `/etc/jammi/jammi.toml`, the platform config directory), with a
    /// wildcard bind host (`0.0.0.0`, `::`) rewritten to its loopback
    /// equivalent.
    #[arg(long)]
    url: Option<String>,
    /// Request timeout, in seconds. Must be at least 1.
    #[arg(
        long,
        default_value_t = jammi_server::probe::DEFAULT_TIMEOUT_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    timeout_secs: u64,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve(cli.serve)) {
        Command::Serve(args) => serve(args).await,
        Command::Probe(args) => probe(args).await,
    }
}

async fn serve(args: ServeArgs) -> ExitCode {
    let config = match JammiConfig::load(args.config.as_deref()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("jammi-server: failed to load config: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Fail closed before tracing (and everything else) initializes: a
    // configured-but-undecodable audit master key must never let the
    // process reach a listening state with audit signing silently dead.
    // Checked here — pre-tracing, like the config-load failure above — so
    // the failure prints via `eprintln!` to stderr rather than through the
    // tracing subscriber, which writes to stdout (see
    // `telemetry::init_tracing`). Absence of a configured key is left
    // exactly as it was: this only closes the malformed case.
    if let Err(e) = validate_audit_master_key(&config) {
        eprintln!("jammi-server: {e}");
        return ExitCode::FAILURE;
    }

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

async fn probe(args: ProbeArgs) -> ExitCode {
    let timeout = Duration::from_secs(args.timeout_secs);
    let url = match args.url {
        Some(url) => url,
        None => {
            // Same config resolution `serve` runs — `JammiConfig::load`
            // with the same optional explicit path, falling back through
            // `JAMMI_CONFIG` / `./jammi.toml` / `/etc/jammi/jammi.toml` /
            // the platform config dir in that order.
            let config = match JammiConfig::load(args.config.as_deref()) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("jammi-server probe: failed to load config: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match jammi_server::probe::default_url(&config.server) {
                Ok(url) => url,
                Err(e) => {
                    eprintln!("jammi-server probe: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };
    match jammi_server::probe::check(&url, timeout).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("jammi-server probe: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_is_the_default_subcommand() {
        // Bare invocation.
        let cli = Cli::try_parse_from(["jammi-server"]).expect("bare invocation parses");
        assert!(cli.command.is_none());

        // Top-level `--config` with no subcommand keyword.
        let cli = Cli::try_parse_from(["jammi-server", "--config", "x.toml"])
            .expect("top-level --config parses");
        assert!(cli.command.is_none());
        assert_eq!(cli.serve.config, Some(PathBuf::from("x.toml")));

        // Explicit `serve --config`.
        let cli = Cli::try_parse_from(["jammi-server", "serve", "--config", "x.toml"])
            .expect("serve --config parses");
        match cli.command {
            Some(Command::Serve(args)) => assert_eq!(args.config, Some(PathBuf::from("x.toml"))),
            other => panic!("expected Some(Command::Serve(_)), got {}", describe(&other)),
        }

        // Explicit `probe --url`.
        let cli = Cli::try_parse_from(["jammi-server", "probe", "--url", "http://x/readyz"])
            .expect("probe --url parses");
        match cli.command {
            Some(Command::Probe(args)) => {
                assert_eq!(args.url.as_deref(), Some("http://x/readyz"))
            }
            other => panic!("expected Some(Command::Probe(_)), got {}", describe(&other)),
        }

        // Explicit `probe --config`, same shape as `serve --config`.
        let cli = Cli::try_parse_from(["jammi-server", "probe", "--config", "x.toml"])
            .expect("probe --config parses");
        match cli.command {
            Some(Command::Probe(args)) => {
                assert_eq!(args.config, Some(PathBuf::from("x.toml")));
                assert_eq!(args.url, None);
            }
            other => panic!("expected Some(Command::Probe(_)), got {}", describe(&other)),
        }
    }

    #[test]
    fn probe_timeout_secs_rejects_zero() {
        let err = Cli::try_parse_from(["jammi-server", "probe", "--timeout-secs", "0"])
            .expect_err("--timeout-secs 0 must be rejected at the CLI boundary");
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn top_level_config_conflicts_with_a_subcommand() {
        let err = Cli::try_parse_from(["jammi-server", "--config", "x.toml", "probe"])
            .expect_err("top-level --config combined with a subcommand must be rejected");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    fn describe(command: &Option<Command>) -> &'static str {
        match command {
            None => "None",
            Some(Command::Serve(_)) => "Some(Command::Serve(_))",
            Some(Command::Probe(_)) => "Some(Command::Probe(_))",
        }
    }
}
