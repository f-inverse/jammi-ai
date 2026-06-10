//! `jammi models` subcommand.
//!
//! List the models registered to the session's tenant over the remote
//! [`Session`].

use clap::Subcommand;
use jammi_ai::Session;

#[derive(Subcommand)]
pub enum ModelAction {
    /// List registered models
    List,
}

pub async fn run(session: &Session, action: ModelAction) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        ModelAction::List => {
            let models = session.list_models().await?;
            if models.is_empty() {
                println!("No models registered.");
            } else {
                println!("{:<40} {:<12} {:<14} Status", "Model ID", "Backend", "Task");
                println!("{}", "-".repeat(78));
                for m in models {
                    println!(
                        "{:<40} {:<12} {:<14} {}",
                        m.model_id,
                        m.backend,
                        format!("{:?}", m.task),
                        m.status
                    );
                }
            }
        }
    }
    Ok(())
}
