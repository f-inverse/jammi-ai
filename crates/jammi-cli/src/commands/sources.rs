use clap::Subcommand;
use jammi_engine::catalog::Catalog;
use jammi_engine::config::JammiConfig;

#[derive(Subcommand)]
pub enum SourceAction {
    /// List registered sources
    List,
}

pub async fn run(
    config: JammiConfig,
    action: SourceAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = Catalog::open(&config.artifact_dir)?;

    match action {
        SourceAction::List => {
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
    }
    Ok(())
}
