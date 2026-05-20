mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "jammi", version, about = "Jammi AI — inference engine CLI")]
struct Cli {
    /// Path to config file
    #[arg(long, global = true)]
    config: Option<String>,

    /// Bind a tenant scope (UUID v4 / v7). Every catalog read and write
    /// observes `tenant_id = <UUID> OR tenant_id IS NULL`; writes record
    /// `tenant_id = <UUID>`. Omit for an unscoped (single-tenant) session.
    #[arg(long, global = true)]
    tenant: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP + Flight SQL server
    Serve,
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
    /// Run a SQL query
    Query {
        /// SQL query string
        sql: String,
    },
    /// Show the execution plan for a query
    Explain {
        /// SQL query string
        sql: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let config_path = cli.config.as_deref();
    let tenant = cli.tenant.as_deref();

    if let Err(e) = run(cli.command, config_path, tenant).await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(
    command: Option<Commands>,
    config_path: Option<&str>,
    tenant: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = jammi_engine::config::JammiConfig::load(config_path.map(std::path::Path::new))?;
    let tenant = match tenant {
        None => None,
        Some(s) => {
            use std::str::FromStr;
            Some(jammi_engine::TenantId::from_str(s)?)
        }
    };

    match command {
        Some(Commands::Serve) => commands::serve::run(config).await,
        Some(Commands::Sources { action }) => commands::sources::run(config, tenant, action).await,
        Some(Commands::Models { action }) => commands::models::run(config, tenant, action).await,
        Some(Commands::Query { sql }) => commands::query::run(config, tenant, &sql).await,
        Some(Commands::Explain { sql }) => commands::query::explain(config, tenant, &sql).await,
        None => {
            // No subcommand — print help
            use clap::CommandFactory;
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}
