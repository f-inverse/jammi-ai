//! Workspace-pin oracle for the DataFusion line every crate compiles against.
//!
//! `jammi-db` owns the query engine; the whole workspace shares one
//! `datafusion` pin (`Cargo.toml` `[workspace.dependencies]`), so a single
//! assertion on the linked crate's own version constant is the oracle for
//! the line the workspace is on. Fails on any other major line, so a
//! `cargo update` that silently walks the pin (or a sibling crate that pulls
//! a second DataFusion) is caught here rather than at the Flight SQL seam.

#[test]
fn datafusion_version_is_the_54_line() {
    let version = datafusion::DATAFUSION_VERSION;
    assert!(
        version.starts_with("54."),
        "DATAFUSION_VERSION = {version}; the workspace pins the DataFusion 54 line"
    );
}
