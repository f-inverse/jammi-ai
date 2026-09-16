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
        "{:<38} {:<16} {:<20} {:<12} {:<9} {:<26} {:<26} Devices",
        "Instance ID", "Label", "Host", "Kinds", "State", "Started", "Last Seen"
    );
    println!("{}", "-".repeat(160));
}

fn print_row(w: &WorkerSummary) {
    println!(
        "{:<38} {:<16} {:<20} {:<12} {:<9} {:<26} {:<26} {}",
        w.instance_id,
        if w.label.is_empty() { "—" } else { &w.label },
        w.host,
        w.kinds,
        w.state,
        w.started_at,
        w.last_seen_at,
        format_devices(&w.devices),
    );
}

/// `kind0, kind1, ...` in ordinal order, `—` for an empty device list —
/// matching the empty-label convention above.
fn format_devices(devices: &[jammi_admin::DeviceFact]) -> String {
    if devices.is_empty() {
        return "—".to_string();
    }
    devices
        .iter()
        .map(|d| format!("{}{}", d.kind, d.ordinal))
        .collect::<Vec<_>>()
        .join(", ")
}
