//! `jammi workers` subcommand.
//!
//! Reports the engine processes currently running the claim loop
//! (`[worker] enabled = true`), joined from `instances`/`workers`
//! (migration 029) via [`CatalogClient::list_workers`].

use clap::Subcommand;
use jammi_admin::{CatalogClient, WorkerSummary};
use jammi_db::catalog::instance::DeviceFact;
use jammi_db::catalog::status::JobKinds;

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
        w.label.as_deref().unwrap_or("—"),
        w.host.as_deref().unwrap_or("—"),
        format_kinds(&w.kinds),
        w.state,
        w.started_at,
        w.last_seen_at,
        format_devices(&w.devices),
    );
}

/// `all` for a worker that claims every kind, else the kinds it claims.
fn format_kinds(kinds: &JobKinds) -> String {
    if *kinds == JobKinds::all() {
        "all".to_string()
    } else {
        kinds.to_string()
    }
}

/// `kind0, kind1, ...` in rank order (the configured `[gpu] devices` order), `—` for an empty device list —
/// matching the empty-label convention above.
fn format_devices(devices: &[DeviceFact]) -> String {
    if devices.is_empty() {
        return "—".to_string();
    }
    devices
        .iter()
        .map(|d| format!("{}{}", d.kind, d.ordinal))
        .collect::<Vec<_>>()
        .join(", ")
}
