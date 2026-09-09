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
    println!("scope:           {}", report.scope);
    println!("applied:         {}", report.applied);
    println!("rows_failed:     {}", join_or_dash(&report.rows_failed));
    println!(
        "orphans:         {}",
        join_capped(&report.orphans, report.orphan_count)
    );
    println!(
        "pending:         {}",
        join_capped(&report.pending, report.pending_count)
    );
    println!(
        "unattributed:    {}",
        join_capped(&report.unattributed, report.unattributed_count)
    );
    println!(
        "damaged:         {}",
        join_capped(&report.damaged, report.damaged_count)
    );
    println!("bytes_reclaimed: {}", report.bytes_reclaimed);
    Ok(())
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
    use super::DEFAULT_GRACE_SECS;

    #[test]
    fn default_grace_secs_matches_the_wire_default() {
        assert_eq!(DEFAULT_GRACE_SECS, 3600);
    }
}
