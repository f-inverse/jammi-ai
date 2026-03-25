mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "jammi", version, about = "Jammi AI — inference engine CLI")]
struct Cli {
    /// Path to config file
    #[arg(long, global = true)]
    config: Option<String>,

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

    if let Err(e) = run(cli.command, config_path).await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(
    command: Option<Commands>,
    config_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = jammi_engine::config::JammiConfig::load(config_path.map(std::path::Path::new))?;

    match command {
        Some(Commands::Serve) => commands::serve::run(config).await,
        Some(Commands::Sources { action }) => commands::sources::run(config, action).await,
        Some(Commands::Models { action }) => commands::models::run(config, action).await,
        Some(Commands::Query { sql }) => commands::query::run(config, &sql).await,
        Some(Commands::Explain { sql }) => commands::query::explain(config, &sql).await,
        None => {
            // No subcommand — print help
            use clap::CommandFactory;
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}
