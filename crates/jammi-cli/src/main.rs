//! `jammi` — the strict gRPC control-plane client CLI.
//!
//! The CLI talks to a running `jammi-server` over the `jammi.v1` wire surface
//! and never touches the catalog or storage in-process: it opens a
//! [`CatalogClient`] against a `--target` endpoint and dispatches each
//! subcommand to one or more control verbs. There is no embedded engine here —
//! `jammi serve` lives in the `jammi-server` binary, not this CLI — and the
//! crate depends on `jammi-admin`, not `jammi-ai`, so the candle stack never
//! reaches the `jammi` binary.

mod commands;

use clap::{Parser, Subcommand};
use jammi_admin::CatalogClient;

/// Default endpoint: the server's Flight-SQL + gRPC listener
/// (`flight_listen = "0.0.0.0:8081"`).
const DEFAULT_TARGET: &str = "grpc://127.0.0.1:8081";

#[derive(Parser)]
#[command(name = "jammi", version, about = "Jammi AI — gRPC client CLI")]
struct Cli {
    /// Server endpoint. Accepts `grpc://host:port` or a bare `host:port`
    /// (both plaintext h2 — a bare value is treated as plaintext). TLS
    /// termination is the consumer's runtime, not the CLI's: put a
    /// TLS-terminating proxy in front and point `--target` at it in
    /// plaintext. The CLI is a strict client — every verb runs on the server
    /// reached here, never in-process.
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
    /// Observe/manage durable jobs (list, per-job status, cancel, prune).
    /// Jobs are submitted through the data-plane client / SDK — this surface
    /// is the control-plane read + cancel + prune peer, over every job kind
    /// (training and compute alike, PLAN-C §3).
    Jobs {
        #[command(subcommand)]
        action: commands::jobs::JobAction,
    },
    /// List the engine processes currently running the claim loop.
    Workers {
        #[command(subcommand)]
        action: commands::workers::WorkerAction,
    },
    /// Cross-check the catalog against the object store and report (or
    /// reclaim) drift.
    Reconcile {
        /// Actually reclaim drift past `--grace-secs` (and flip an incomplete
        /// `ready` row to `failed`). Omit for a dry run that reports without
        /// mutating anything.
        #[arg(long)]
        apply: bool,
        /// An orphan candidate younger than this (seconds) is never reclaimed
        /// this pass. `--apply` requires this to be at least the server's
        /// configured lease duration or the call fails naming both values.
        #[arg(long, default_value_t = commands::reconcile::DEFAULT_GRACE_SECS)]
        grace_secs: u64,
        /// Run the cross-tenant admin pass instead of this session's own
        /// tenant scope. The server refuses this with `PERMISSION_DENIED`
        /// unless a deployment-supplied admin authorizer is wired.
        #[arg(long)]
        all: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // `--help` / `--version` / no-subcommand must not connect: `CatalogClient
    // ::connect` eagerly dials the endpoint, so the no-verb path prints help and
    // returns before any connection is attempted.
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
    let client = CatalogClient::connect(endpoint).await?;

    // Bind the tenant before any verb. An unbound/unknown session id maps to an
    // *unscoped* (all-tenants) view server-side with no error, so a `--tenant`
    // query that skipped this bind would silently read across tenants. Binding
    // first stamps the session's tenant against the same session id every verb
    // (gRPC header) carries.
    if let Some(t) = tenant {
        use std::str::FromStr;
        client.bind_tenant(jammi_db::TenantId::from_str(t)?).await?;
    }

    dispatch(&client, command).await
}

async fn dispatch(
    client: &CatalogClient,
    command: Commands,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Status => commands::status::run(client).await,
        Commands::Sources { action } => commands::sources::run(client, action).await,
        Commands::Models { action } => commands::models::run(client, action).await,
        Commands::Trigger { action } => commands::trigger::run(client, action).await,
        Commands::Channels { action } => commands::channels::run(client, action).await,
        Commands::Mutable { action } => commands::mutable::run(client, action).await,
        Commands::Jobs { action } => commands::jobs::run(client, action).await,
        Commands::Workers { action } => commands::workers::run(client, action).await,
        Commands::Reconcile {
            apply,
            grace_secs,
            all,
        } => commands::reconcile::run(client, apply, grace_secs, all).await,
    }
}

/// Translate a `--target` value into a [`tonic::transport::Endpoint`].
///
/// `grpc://` and a bare `host:port` are plaintext h2 (`http://`); `http://`
/// passes through. Those are the only two accepted forms: TLS termination is
/// the consumer's runtime, not the engine's (see `philosophy.md`), so a
/// `grpcs://` or `https://` target is refused rather than silently trusted —
/// terminate TLS in a proxy in front of the server and point `--target` at it
/// in plaintext. An unrecognised scheme is rejected with a typed error rather
/// than silently coerced — a misspelled scheme should fail loudly, not
/// connect somewhere unexpected.
fn endpoint_from_target(
    target: &str,
) -> Result<tonic::transport::Endpoint, Box<dyn std::error::Error>> {
    let url = match target.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("http", _)) => target.to_string(),
        Some((scheme, _)) => {
            return Err(format!(
                "unsupported --target scheme '{scheme}://'; use grpc://, \
                 http://, or a bare host:port"
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
    fn refuses_tls_schemes_naming_the_supported_ones() {
        for target in ["grpcs://h:1", "https://h:1"] {
            let err = endpoint_from_target(target).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("grpc://"), "{msg}");
            assert!(msg.contains("http://"), "{msg}");
        }
        // Control: the two supported schemes still parse.
        assert!(endpoint_from_target("grpc://h:1").is_ok());
        assert!(endpoint_from_target("http://h:1").is_ok());
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
