//! `jammi` — the strict gRPC client CLI.
//!
//! The CLI talks to a running `jammi-server` over the `jammi.v1` wire surface
//! and never touches the catalog or storage in-process: it opens a remote
//! [`Session`] against a `--target` endpoint and dispatches
//! each subcommand to one or more session verbs. There is no embedded engine
//! here — `jammi serve` lives in the `jammi-server` binary, not this CLI.

mod commands;

use clap::{Parser, Subcommand};
use jammi_ai::{Jammi, Session, Target};

/// Default endpoint: the server's Flight-SQL + gRPC listener
/// (`flight_listen = "0.0.0.0:8081"`).
const DEFAULT_TARGET: &str = "grpc://127.0.0.1:8081";

#[derive(Parser)]
#[command(name = "jammi", version, about = "Jammi AI — gRPC client CLI")]
struct Cli {
    /// Server endpoint. Accepts `grpc://host:port` (plaintext h2),
    /// `grpcs://host:port` (TLS), `http(s)://host:port`, or a bare `host:port`
    /// (treated as plaintext). The CLI is a strict client — every verb runs on
    /// the server reached here, never in-process.
    #[arg(long, global = true, default_value = DEFAULT_TARGET)]
    target: String,

    /// Bind a tenant scope (UUID v4 / v7) for the session. Every verb then runs
    /// scoped to that tenant; omit for an unscoped session.
    #[arg(long, global = true)]
    tenant: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Report the server's capabilities (version, features, storage backends,
    /// mounted services). Reachability is confirmed by the RPC succeeding.
    Status,
    /// Manage data sources
    Sources {
        #[command(subcommand)]
        action: commands::sources::SourceAction,
    },
    /// Manage models
    Models {
        #[command(subcommand)]
        action: commands::models::ModelAction,
    },
    /// Manage trigger-stream topics
    Trigger {
        #[command(subcommand)]
        action: commands::trigger::TriggerAction,
    },
    /// Manage evidence channels
    Channels {
        #[command(subcommand)]
        action: commands::channels::ChannelAction,
    },
    /// Manage mutable companion tables
    Mutable {
        #[command(subcommand)]
        action: commands::mutable::MutableAction,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // `--help` / `--version` / no-subcommand must not connect: `Jammi::open`
    // eagerly dials the endpoint, so the no-verb path prints help and returns
    // before any connection is attempted.
    let Some(command) = cli.command else {
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        cmd.print_help().expect("print help");
        println!();
        return;
    };

    if let Err(e) = run(command, &cli.target, cli.tenant.as_deref()).await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(
    command: Commands,
    target: &str,
    tenant: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = endpoint_from_target(target)?;
    let session = Jammi::open(Target::Remote(endpoint)).await?;

    // Bind the tenant before any verb. An unbound/unknown session id maps to an
    // *unscoped* (all-tenants) view server-side with no error, so a `--tenant`
    // query that skipped this bind would silently read across tenants. Binding
    // first stamps the session's tenant against the same session id every verb
    // (gRPC header / Flight SQL header) carries.
    if let Some(t) = tenant {
        use std::str::FromStr;
        session
            .bind_tenant(jammi_db::TenantId::from_str(t)?)
            .await?;
    }

    dispatch(&session, command).await
}

async fn dispatch(session: &Session, command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Status => commands::status::run(session).await,
        Commands::Sources { action } => commands::sources::run(session, action).await,
        Commands::Models { action } => commands::models::run(session, action).await,
        Commands::Trigger { action } => commands::trigger::run(session, action).await,
        Commands::Channels { action } => commands::channels::run(session, action).await,
        Commands::Mutable { action } => commands::mutable::run(session, action).await,
    }
}

/// Translate a `--target` value into a [`tonic::transport::Endpoint`].
///
/// `grpc://` and a bare `host:port` are plaintext h2 (`http://`); `grpcs://` is
/// TLS (`https://`); `http://` / `https://` pass through. An unrecognised scheme
/// is rejected with a typed error rather than silently coerced — a misspelled
/// scheme should fail loudly, not connect somewhere unexpected.
fn endpoint_from_target(
    target: &str,
) -> Result<tonic::transport::Endpoint, Box<dyn std::error::Error>> {
    let url = match target.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("grpcs", rest)) => format!("https://{rest}"),
        Some(("http", _)) | Some(("https", _)) => target.to_string(),
        Some((scheme, _)) => {
            return Err(format!(
                "unsupported --target scheme '{scheme}://'; use grpc://, grpcs://, \
                 http://, https://, or a bare host:port"
            )
            .into());
        }
        // No scheme: a bare `host:port` is plaintext h2.
        None => format!("http://{target}"),
    };
    Ok(tonic::transport::Endpoint::try_from(url)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_grpc_becomes_http() {
        let ep = endpoint_from_target("grpc://127.0.0.1:8081").unwrap();
        assert_eq!(ep.uri().scheme_str(), Some("http"));
        assert_eq!(ep.uri().host(), Some("127.0.0.1"));
        assert_eq!(ep.uri().port_u16(), Some(8081));
    }

    #[test]
    fn endpoint_grpcs_becomes_https() {
        let ep = endpoint_from_target("grpcs://host.example:443").unwrap();
        assert_eq!(ep.uri().scheme_str(), Some("https"));
    }

    #[test]
    fn endpoint_bare_host_is_plaintext() {
        let ep = endpoint_from_target("localhost:9000").unwrap();
        assert_eq!(ep.uri().scheme_str(), Some("http"));
        assert_eq!(ep.uri().port_u16(), Some(9000));
    }

    #[test]
    fn endpoint_http_passthrough() {
        let ep = endpoint_from_target("http://127.0.0.1:8081").unwrap();
        assert_eq!(ep.uri().scheme_str(), Some("http"));
    }

    #[test]
    fn endpoint_rejects_unknown_scheme() {
        let err = endpoint_from_target("ftp://host:21").unwrap_err();
        assert!(err.to_string().contains("unsupported --target scheme"));
    }
}
