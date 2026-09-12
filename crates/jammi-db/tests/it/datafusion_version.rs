//! Version oracle for the `datafusion` this crate links.
//!
//! `jammi-db` owns the query engine and the whole workspace shares one
//! `datafusion` pin (`Cargo.toml` `[workspace.dependencies]`), so the linked
//! crate's own version constant is the oracle for the line this binary was
//! built against: it fails on any other major line. It says nothing about the
//! rest of the graph — a second `datafusion` elsewhere in the lockfile is
//! invisible here.
//!
//! That bound — exactly ONE `datafusion`/`arrow`/`parquet`/`object_store`
//! line, and it is the 54/58 line — is `deny.toml`'s `[bans] deny` entries
//! (`deny-multiple-versions` plus the version fences). The `dep-audit` CI job
//! runs them over TWO graphs: default features, and `--all-features` (the
//! second is what covers crates reachable only through optional features, such
//! as the `datafusion-table-providers` subtree behind jammi-db's
//! `postgres`/`mysql`). This test's own reach is the default-feature unit
//! graph of this one binary.

#[test]
fn datafusion_version_is_the_54_line() {
    let version = datafusion::DATAFUSION_VERSION;
    assert!(
        version.starts_with("54."),
        "DATAFUSION_VERSION = {version}; the workspace pins the DataFusion 54 line"
    );
}
