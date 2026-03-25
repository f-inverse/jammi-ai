use clap::Subcommand;
use jammi_engine::catalog::Catalog;
use jammi_engine::config::JammiConfig;

#[derive(Subcommand)]
pub enum ModelAction {
    /// List registered models
    List,
}

pub async fn run(
    config: JammiConfig,
    action: ModelAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = Catalog::open(&config.artifact_dir)?;

    match action {
        ModelAction::List => {
            let models = catalog.list_models()?;
            if models.is_empty() {
                println!("No models registered.");
            } else {
                println!("{:<40} {:<12} {:<10} Status", "Model ID", "Backend", "Task");
                println!("{}", "-".repeat(75));
                for m in models {
                    println!(
                        "{:<40} {:<12} {:<10} {}",
                        m.model_id, m.backend, m.task, m.status
                    );
                }
            }
        }
    }
    Ok(())
}
