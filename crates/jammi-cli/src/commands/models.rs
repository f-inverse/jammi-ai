use clap::Subcommand;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;

#[derive(Subcommand)]
pub enum ModelAction {
    /// List registered models
    List,
}

pub async fn run(
    config: JammiConfig,
    tenant: Option<jammi_db::TenantId>,
    action: ModelAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let session = InferenceSession::new(config).await?;
    if let Some(t) = tenant {
        session.bind_tenant(t);
    }

    match action {
        ModelAction::List => {
            let models = session.catalog().list_models().await?;
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
