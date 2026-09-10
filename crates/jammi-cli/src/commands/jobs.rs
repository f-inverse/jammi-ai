//! `jammi jobs` subcommand.
//!
//! Read-mostly observation over the remote [`CatalogClient`]'s `JobService`
//! surface (replaces `jammi train list/status`): list the jobs
//! visible to the session's tenant, read one job's lifecycle status by id,
//! request cancellation, or eagerly sweep terminal rows past the
//! deployment's retention window. Submission stays on the data-plane client
//! / SDK — this is the control-plane read (+ cancel/prune) peer, generalised
//! from training jobs to every job kind.

use clap::Subcommand;
use jammi_admin::{CatalogClient, JobSummary};

#[derive(Subcommand)]
pub enum JobAction {
    /// List jobs visible to the session's tenant, most recent first
    List,
    /// Read one job's lifecycle status by id
    Status {
        /// Job id returned when the job was submitted.
        job_id: String,
    },
    /// Request cancellation of a job by id
    Cancel {
        /// Job id returned when the job was submitted.
        job_id: String,
    },
    /// Eagerly delete terminal (completed/failed) job rows older than the
    /// deployment's `[jobs] retention_days`
    Prune,
}

pub async fn run(
    session: &CatalogClient,
    action: JobAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        JobAction::List => {
            let jobs = session.list_jobs().await?;
            if jobs.is_empty() {
                println!("No jobs.");
            } else {
                print_header();
                for j in jobs {
                    print_row(&j);
                }
            }
        }
        JobAction::Status { job_id } => {
            let info = session.job_status(&job_id).await?;
            println!("job_id:   {job_id}");
            println!("kind:     {}", info.kind);
            println!("status:   {}", info.status);
            if !info.output_model_id.is_empty() {
                println!("model_id: {}", info.output_model_id);
            }
            if !info.error.is_empty() {
                println!("error:    {}", info.error);
            }
            if let Some(metrics) = &info.metrics_json {
                // Opaque blob — printed verbatim, not parsed or reformatted.
                // Its schema is documented at the trainer, not the CLI.
                println!("metrics:  {metrics}");
            }
            if let Some(report) = &info.acceleration_report_json {
                // Opaque blob — printed verbatim, not parsed or reformatted.
                // Its schema (including the `"state"` vocabulary) is
                // documented at the trainer, not the CLI. Absent entirely
                // when the row predates the column (SQL `NULL`) — no
                // fabricated placeholder in that case.
                println!("acceleration_report: {report}");
            }
        }
        JobAction::Cancel { job_id } => {
            let cancelled = session.cancel_job(&job_id).await?;
            if cancelled {
                println!("cancellation requested for {job_id}");
            } else {
                println!("{job_id} was already terminal or absent; no effect");
            }
        }
        JobAction::Prune => {
            let deleted = session.prune_jobs().await?;
            println!("pruned {deleted} terminal job row(s)");
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

fn print_row(j: &JobSummary) {
    println!(
        "{:<38} {:<18} {:<10} {:<26} {}",
        j.job_id, j.kind, j.status, j.created_at, j.output_model_id,
    );
}
