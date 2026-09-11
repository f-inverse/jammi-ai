//! `jammi workers` subcommand.
//!
//! Reports the engine processes currently running the claim loop
//! (`[worker] enabled = true`), joined from `instances`/`workers`
//! (migration 029) via [`CatalogClient::list_workers`].

use clap::Subcommand;
use jammi_admin::{CatalogClient, WorkerSummary};

#[derive(Subcommand)]
pub enum WorkerAction {
    /// List the engine processes currently running the claim loop
    List,
}

pub async fn run(
    session: &CatalogClient,
    action: WorkerAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        WorkerAction::List => {
            let workers = session.list_workers().await?;
            if workers.is_empty() {
                println!("No workers.");
            } else {
                print_header();
                for w in workers {
                    print_row(&w);
                }
            }
        }
    }
    Ok(())
}

fn print_header() {
    println!(
        "{:<38} {:<16} {:<20} {:<12} {:<9} {:<26} Last Seen",
        "Instance ID", "Label", "Host", "Kinds", "State", "Started"
    );
    println!("{}", "-".repeat(140));
}

fn print_row(w: &WorkerSummary) {
    println!(
        "{:<38} {:<16} {:<20} {:<12} {:<9} {:<26} {}",
        w.instance_id,
        if w.label.is_empty() { "—" } else { &w.label },
        w.host,
        w.kinds,
        w.state,
        w.started_at,
        w.last_seen_at,
    );
}
