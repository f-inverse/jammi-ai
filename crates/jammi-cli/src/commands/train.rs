//! `jammi train` subcommand.
//!
//! Read-only training-job observation over the remote [`CatalogClient`]: list
//! the jobs visible to the session's tenant, or read one job's lifecycle
//! status by id. Submission stays on the data-plane client / SDK — this is the
//! control-plane read peer, and there is no progress surface to read (the
//! engine persists run metrics only at job finalization).

use clap::Subcommand;
use jammi_admin::{CatalogClient, TrainingJobSummary};

#[derive(Subcommand)]
pub enum TrainAction {
    /// List training jobs visible to the session's tenant, most recent first
    List,
    /// Read one training job's lifecycle status by id
    Status {
        /// Job id returned when the job was submitted.
        job_id: String,
    },
}

pub async fn run(
    session: &CatalogClient,
    action: TrainAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        TrainAction::List => {
            let jobs = session.list_training_jobs().await?;
            if jobs.is_empty() {
                println!("No training jobs.");
            } else {
                print_header();
                for j in jobs {
                    print_row(&j);
                }
            }
        }
        TrainAction::Status { job_id } => {
            let info = session.training_status(&job_id).await?;
            println!("job_id:   {job_id}");
            println!("status:   {}", info.status);
            if !info.model_id.is_empty() {
                println!("model_id: {}", info.model_id);
            }
            if !info.error.is_empty() {
                println!("error:    {}", info.error);
            }
            if let Some(metrics) = &info.metrics_json {
                // Opaque blob — printed verbatim, not parsed or reformatted.
                // Its schema is documented at the trainer, not the CLI.
                println!("metrics:  {metrics}");
            }
        }
    }
    Ok(())
}

fn print_header() {
    println!(
        "{:<38} {:<18} {:<10} {:<26} Model ID",
        "Job ID", "Kind", "Status", "Created"
    );
    println!("{}", "-".repeat(110));
}

fn print_row(j: &TrainingJobSummary) {
    println!(
        "{:<38} {:<18} {:<10} {:<26} {}",
        j.job_id, j.kind, j.status, j.created_at, j.output_model_id,
    );
}
