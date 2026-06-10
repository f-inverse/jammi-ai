//! `jammi models` subcommand.
//!
//! List or describe the models registered to the session's tenant over the
//! remote [`Session`].

use clap::Subcommand;
use jammi_ai::Session;
use jammi_db::catalog::model_repo::ModelRecord;

#[derive(Subcommand)]
pub enum ModelAction {
    /// List registered models
    List,
    /// Describe one registered model by id
    Describe {
        /// Model id (e.g. an HF repo id or a fine-tuned id).
        model_id: String,
    },
}

pub async fn run(session: &Session, action: ModelAction) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        ModelAction::List => {
            let models = session.list_models().await?;
            if models.is_empty() {
                println!("No models registered.");
            } else {
                print_header();
                for m in models {
                    print_row(&m);
                }
            }
        }
        ModelAction::Describe { model_id } => match session.describe_model(&model_id).await? {
            Some(m) => {
                print_header();
                print_row(&m);
            }
            None => println!("No model '{model_id}' registered."),
        },
    }
    Ok(())
}

fn print_header() {
    println!("{:<40} {:<12} {:<14} Status", "Model ID", "Backend", "Task");
    println!("{}", "-".repeat(78));
}

fn print_row(m: &ModelRecord) {
    println!(
        "{:<40} {:<12} {:<14} {}",
        m.model_id,
        m.backend,
        format!("{:?}", m.task),
        m.status
    );
}
