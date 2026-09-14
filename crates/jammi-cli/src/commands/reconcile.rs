//! `jammi reconcile` subcommand.
//!
//! Cross-checks the catalog against the object store and reports (or, when
//! `--apply` is set, reclaims) what has drifted, by calling
//! [`CatalogClient::reconcile`]. `--apply` defaults to `false` — a dry run
//! that reports without mutating anything. `--all` requests the cross-tenant
//! admin pass; the server gates it behind a deployment-supplied admin
//! authorizer, and the shipped default (no authorizer wired) refuses with
//! `PERMISSION_DENIED`, surfaced verbatim by this CLI's ordinary error path
//! (no special-casing here — the same `Err(e) => eprintln!("Error: {e}")`
//! every other verb's failure takes).

use jammi_admin::CatalogClient;
use jammi_db::store::ReconcileReport;

/// `--grace-secs` default: 3600 seconds. This CLI always sends an explicit
/// value, so the server's own `grace_secs`-unset default
/// (`jammi_wire::DEFAULT_RECONCILE_GRACE_SECS`, also 3600) never actually
/// applies here — the two are kept at the same number so a caller who never
/// overrides either surface sees identical behaviour regardless of which one
/// technically decided it. `jammi-cli` depends on `jammi-admin` only (no
/// `jammi-wire`), so the number is duplicated rather than imported.
pub const DEFAULT_GRACE_SECS: u64 = 3600;

pub async fn run(
    session: &CatalogClient,
    apply: bool,
    grace_secs: u64,
    all: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = session.reconcile(apply, Some(grace_secs), all).await?;
    print!("{}", render(&report));
    Ok(())
}

/// Render a [`ReconcileReport`] into the label-shaped lines `jammi reconcile`
/// prints — every field the report carries, none dropped, in the same order
/// [`jammi_db::store::ReconcileReport`] declares them and the same field
/// names the python client's `_reconcile_report_to_dict` projects (so a
/// caller cross-referencing the two surfaces sees identical names). Kept
/// separate from [`run`] so a unit test can exercise it against a
/// hand-built report with no server round trip.
fn render(report: &ReconcileReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "scope:           {}", report.scope);
    let _ = writeln!(out, "applied:         {}", report.applied);
    let _ = writeln!(
        out,
        "rows_failed:     {}",
        join_capped(&report.rows_failed, report.rows_failed_count)
    );
    let _ = writeln!(
        out,
        "orphans:         {}",
        join_capped(&report.orphans, report.orphan_count)
    );
    let _ = writeln!(
        out,
        "pending:         {}",
        join_capped(&report.pending, report.pending_count)
    );
    let _ = writeln!(
        out,
        "unattributed:    {}",
        join_capped(&report.unattributed, report.unattributed_count)
    );
    let _ = writeln!(
        out,
        "damaged:         {}",
        join_capped(&report.damaged, report.damaged_count)
    );
    let _ = writeln!(
        out,
        "referenced:      {}",
        join_capped(&report.referenced, report.referenced_count)
    );
    let _ = writeln!(out, "bytes_reclaimed: {}", report.bytes_reclaimed);
    out
}

/// Render a key list as a comma-separated line, or `—` when empty — the same
/// label-style rendering [`super::status::run`] uses for `jammi status`.
fn join_or_dash(values: &[String]) -> String {
    if values.is_empty() {
        "—".to_string()
    } else {
        values.join(", ")
    }
}

/// [`join_or_dash`], plus a trailing "… and N more" when the report's `true`
/// count exceeds the (possibly capped) list actually printed — so a huge
/// pass's output stays bounded on the terminal while still saying the whole
/// truth about how much was found.
fn join_capped(values: &[String], true_count: u64) -> String {
    let shown = values.len() as u64;
    let mut out = join_or_dash(values);
    if true_count > shown {
        out.push_str(&format!(" … and {} more", true_count - shown));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{render, ReconcileReport, DEFAULT_GRACE_SECS};

    #[test]
    fn default_grace_secs_matches_the_wire_default() {
        assert_eq!(DEFAULT_GRACE_SECS, 3600);
    }

    /// A report whose `referenced` list is non-empty (objects the reap
    /// skipped because a live `models` row still names their prefix) renders
    /// both the entries and the true count — the one field this command used
    /// to drop entirely.
    #[test]
    fn render_prints_referenced_entries_and_count() {
        let report = ReconcileReport {
            scope: "_global".to_string(),
            referenced: vec![
                "models/tenant-a/m1/manifest.json".to_string(),
                "models/tenant-a/m2/manifest.json".to_string(),
            ],
            referenced_count: 2,
            ..Default::default()
        };

        let out = render(&report);

        assert!(
            out.contains("referenced:      models/tenant-a/m1/manifest.json, models/tenant-a/m2/manifest.json"),
            "missing referenced entries in output:\n{out}"
        );
    }

    /// A `referenced` list truncated below its true count still reports the
    /// true total via the same "… and N more" suffix every other capped
    /// field in this report uses.
    #[test]
    fn render_reports_the_true_referenced_count_when_truncated() {
        let report = ReconcileReport {
            scope: "_global".to_string(),
            referenced: vec!["models/tenant-a/m1/manifest.json".to_string()],
            referenced_count: 5,
            truncated: true,
            ..Default::default()
        };

        let out = render(&report);

        assert!(
            out.contains("referenced:      models/tenant-a/m1/manifest.json … and 4 more"),
            "missing capped referenced count in output:\n{out}"
        );
    }
}
