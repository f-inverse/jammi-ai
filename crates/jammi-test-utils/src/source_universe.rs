//! The file universe every source-enumerating oracle quantifies over, in one
//! place: "every `.rs` file cargo compiles that is not a test target". Two
//! oracles in two crates (`jammi-ai`'s submit-seam oracle, `jammi-db`'s raw
//! byte-delete oracle) each once carried their own copy of this filter; a
//! copy drifts, and a universe narrower than what cargo compiles lets the
//! exact defect an enumerating gate exists to catch pass green.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root, derived from this crate's own manifest directory
/// (`crates/jammi-test-utils`) rather than the process's `cwd` — `cargo
/// test` can be invoked from anywhere, but `CARGO_MANIFEST_DIR` is fixed.
pub fn repo_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("CARGO_MANIFEST_DIR has no grandparent: {manifest_dir:?}"))
        .to_path_buf()
}

/// Every git-TRACKED `.rs` file under `root`-relative `dir`, sorted as
/// `git ls-files` emits them. `git ls-files` recurses on its own and is
/// the same list CI checks out, so a file it reports as tracked is scanned
/// or the scan fails loudly — never a `read_dir` walk that could stop
/// early, and never a second, parallel enumeration.
pub fn tracked_rs_files(root: &Path, dir: &str) -> Vec<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "--", dir])
        .output()
        .unwrap_or_else(|e| panic!("git ls-files {dir}: {e}"));
    assert!(
        output.status.success(),
        "git ls-files {dir} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap_or_else(|e| panic!("git ls-files {dir}: non-utf8 output: {e}"))
        .lines()
        .filter(|l| l.ends_with(".rs"))
        .map(str::to_string)
        .collect()
}

/// Whether a repo-relative `.rs` path is one cargo COMPILES outside a test
/// target: every workspace member's `src/` (the crates and the `ci/tools/*`
/// members), every `build.rs`, and every `examples/` and `benches/` target
/// (`cargo clippy --workspace --all-targets` builds them, so a hand-built
/// call placed in one compiles). Excluded: a crate's top-level `tests/`
/// directory (its integration-test targets — `#[cfg(test)]` items inside
/// compiled files are the scanner's own concern) and the `ci/fixtures/`
/// tokenizer inputs, which are parsed by a checker's fixtures and never
/// compiled.
pub fn is_compiled_non_test_source(rel: &str) -> bool {
    if rel.starts_with("ci/fixtures/") {
        return false;
    }
    // A crate's integration-test targets live in ITS top-level `tests/`
    // (`crates/<c>/tests/**`, `ci/tools/<t>/tests/**`). A `tests` component
    // deeper down (`examples/tests/main.rs`, `src/x/tests/mod.rs`) is a
    // compiled target or module and stays in the universe.
    let parts: Vec<&str> = rel.split('/').collect();
    !matches!(
        parts.as_slice(),
        ["crates", _, "tests", ..] | ["ci", "tools", _, "tests", ..]
    )
}

/// Every tracked `.rs` file under `root` that [`is_compiled_non_test_source`]
/// admits — the one universe both call-site oracles scan.
pub fn compiled_non_test_rs_files(root: &Path) -> Vec<String> {
    let files: Vec<String> = tracked_rs_files(root, ".")
        .into_iter()
        .filter(|f| is_compiled_non_test_source(f))
        .collect();
    assert!(
        files.len() > 100,
        "git ls-files returned suspiciously few compiled non-test .rs files ({}); the universe \
         quantifier is broken, not the tree",
        files.len()
    );
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_universe_admits_every_compiled_shape_and_refuses_test_targets_and_fixtures() {
        for admitted in [
            "crates/jammi-db/src/store/mod.rs",
            "crates/jammi-kernels/build.rs",
            "crates/jammi-bench/examples/frontend_serial_tail.rs",
            "crates/jammi-bench/benches/anything.rs",
            "ci/tools/symbol-index/src/main.rs",
            "crates/jammi-bench/examples/tests/main.rs",
            "crates/jammi-db/src/store/tests/helpers.rs",
        ] {
            assert!(
                is_compiled_non_test_source(admitted),
                "{admitted} must be in the universe"
            );
        }
        for refused in [
            "crates/jammi-ai/tests/it/rank_admission.rs",
            "crates/jammi-db/tests/it/models_delete_call_sites.rs",
            "ci/fixtures/kernel-oracle-tokenizer/chars.rs",
            "ci/tools/symbol-index/tests/smoke.rs",
        ] {
            assert!(
                !is_compiled_non_test_source(refused),
                "{refused} must be outside the universe"
            );
        }
    }
}
