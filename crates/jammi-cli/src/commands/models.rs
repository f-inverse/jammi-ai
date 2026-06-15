//! `jammi models` subcommand.
//!
//! List or describe the models registered to the session's tenant over the
//! remote [`CatalogClient`].

use clap::Subcommand;
use jammi_admin::CatalogClient;
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
    /// Soft-retire a model: hide it from listings and refuse to serve it, while
    /// keeping it resolvable as a reference target. Retires the latest version.
    Retire {
        /// Model id (e.g. an HF repo id or a fine-tuned id).
        model_id: String,
    },
    /// Hard-delete a model row. Refused while any reference still points at the
    /// model. Targets the latest version.
    Delete {
        /// Model id (e.g. an HF repo id or a fine-tuned id).
        model_id: String,
        /// Treat a missing model as a no-op rather than an error.
        #[arg(long)]
        if_exists: bool,
    },
    /// Promote a model, marking it the promoted version for its name. Any
    /// previously-promoted sibling is demoted. Promotes the latest version.
    Promote {
        /// Model id (e.g. an HF repo id or a fine-tuned id).
        model_id: String,
    },
}

pub async fn run(
    session: &CatalogClient,
    action: ModelAction,
) -> Result<(), Box<dyn std::error::Error>> {
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
        ModelAction::Retire { model_id } => {
            session.retire_model(&model_id, None).await?;
            println!("Retired model '{model_id}'.");
        }
        ModelAction::Delete {
            model_id,
            if_exists,
        } => {
            session.delete_model(&model_id, None, if_exists).await?;
            println!("Deleted model '{model_id}'.");
        }
        ModelAction::Promote { model_id } => {
            session.promote_model(&model_id, None).await?;
            println!("Promoted model '{model_id}'.");
        }
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
