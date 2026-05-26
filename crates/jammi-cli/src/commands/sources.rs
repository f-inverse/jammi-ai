use clap::Subcommand;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

#[derive(Subcommand)]
pub enum SourceAction {
    /// List registered sources
    List,
    /// Add a file-shaped data source. `--url` accepts a local path
    /// (parsed into `file://...`) or any storage URL the build was
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
    config: JammiConfig,
    tenant: Option<jammi_db::TenantId>,
    action: SourceAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        SourceAction::List => {
            let session = InferenceSession::new(config).await?;
            if let Some(t) = tenant {
                session.bind_tenant(t);
            }
            let sources = session.catalog().list_sources().await?;
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
        SourceAction::Add { name, url, format } => {
            let file_format: FileFormat = format
                .parse()
                .map_err(|e: jammi_db::error::JammiError| e.to_string())?;
            let connection = SourceConnection::parse(&url, file_format)?;
            let session = InferenceSession::new(config).await?;
            if let Some(t) = tenant {
                session.bind_tenant(t);
            }
            session
                .add_source(&name, SourceType::File, connection)
                .await?;
            println!("Source '{name}' added.");
        }
    }
    Ok(())
}
