use clap::Subcommand;
use jammi_ai::session::InferenceSession;
use jammi_engine::catalog::Catalog;
use jammi_engine::config::JammiConfig;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

#[derive(Subcommand)]
pub enum SourceAction {
    /// List registered sources
    List,
    /// Add a local file source
    Add {
        /// Source name
        name: String,
        /// Path to the data file
        #[arg(long)]
        path: String,
        /// File format (parquet, csv, json)
        #[arg(long)]
        format: String,
    },
}

pub async fn run(
    config: JammiConfig,
    action: SourceAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        SourceAction::List => {
            let catalog = Catalog::open(&config.artifact_dir)?;
            let sources = catalog.list_sources()?;
            if sources.is_empty() {
                println!("No sources registered.");
            } else {
                println!("{:<20} {:<15} Created", "Name", "Type");
                println!("{}", "-".repeat(55));
                for s in sources {
                    println!(
                        "{:<20} {:<15} {}",
                        s.source_id,
                        format!("{:?}", s.source_type),
                        s.created_at
                    );
                }
            }
        }
        SourceAction::Add { name, path, format } => {
            let file_format: FileFormat = format
                .parse()
                .map_err(|e: jammi_engine::error::JammiError| e.to_string())?;
            let connection = SourceConnection::from_path(&path, file_format);
            let session = InferenceSession::new(config).await?;
            session
                .add_source(&name, SourceType::Local, connection)
                .await?;
            println!("Source '{name}' added.");
        }
    }
    Ok(())
}
