//! `jammi sources` subcommand.
//!
//! Register and list data sources over the remote `Session`. `list` prints
//! each `SourceDescriptor`: its registry identity plus how many embedding
//! result tables have been produced from it.

use clap::Subcommand;
use jammi_ai::Session;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

#[derive(Subcommand)]
pub enum SourceAction {
    /// List registered sources
    List,
    /// Add a file-shaped data source. `--url` accepts a local path
    /// (parsed into `file://...`) or any storage URL the server was
    /// compiled with: `s3://bucket/key`, `gs://bucket/key`,
    /// `azure://container/blob`.
    Add {
        /// Source name
        name: String,
        /// Storage URL or local path
        #[arg(long)]
        url: String,
        /// File format (parquet, csv, json)
        #[arg(long)]
        format: String,
    },
}

pub async fn run(
    session: &Session,
    action: SourceAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        SourceAction::List => {
            let sources = session.list_sources().await?;
            if sources.is_empty() {
                println!("No sources registered.");
            } else {
                println!("{:<20} {:<12} {:<10} Embeddings", "Name", "Type", "Status");
                println!("{}", "-".repeat(60));
                for s in sources {
                    println!(
                        "{:<20} {:<12} {:<10} {}",
                        s.source_id,
                        format!("{:?}", s.source_type),
                        s.status,
                        s.result_tables.len()
                    );
                }
            }
        }
        SourceAction::Add { name, url, format } => {
            let file_format: FileFormat = format
                .parse()
                .map_err(|e: jammi_db::error::JammiError| e.to_string())?;
            let connection = SourceConnection::parse(&url, file_format)?;
            session
                .add_source(&name, SourceType::File, connection)
                .await?;
            println!("Source '{name}' added.");
        }
    }
    Ok(())
}
