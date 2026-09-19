//! The typed root an `instances` row carries is constructible in production
//! ONLY from `resolved_result_root()`; any other constructor is test-only.
//! `MemberRoot::resolved`
//! (`crates/jammi-db/src/catalog/instance.rs`) is the ONE production
//! constructor — it calls
//! [`crate::config::JammiConfig::resolved_result_root`] itself.
//! `MemberRoot::new`, the arbitrary-string wrapper, is compiled only under
//! `feature = "test-hooks"`, so it cannot even link into a production
//! build. A `cfg` gate proves that half; it does NOT prove no in-tree
//! caller reaches for the constructor from ordinary (non-test) source —
//! that is an enumeration this test performs directly, on the literal call
//! syntax, so a future `feature = "test-hooks"` production dependent (e.g.
//! a crate that turns the feature on for a non-test reason) cannot smuggle
//! an arbitrary string into `instances.result_root` unnoticed.
//!
//! **Universe**: every `.rs` file under `crates/<name>/
//! src/` for every crate in the workspace that HAS a `src/` directory
//! (found by walking `crates/`, not a hand-maintained list — a new crate is
//! automatically in scope). `tests/`, `benches/`, and `examples/`
//! directories are excluded — those are the test/fixture surfaces
//! `feature = "test-hooks"` already exists to cover, the same way
//! `crates/jammi-db/tests/it/gang_membership.rs` and
//! `crates/jammi-ai/tests/it/instance_identity.rs` use the constructor.
//! Nothing inside `src/` is further excluded on the theory that it might be
//! an inline `#[cfg(test)] mod tests` — such a module belongs in `tests/`,
//! not in `src/`, so this test treats one as a real finding rather than
//! carving out an exception for it.

use std::path::{Path, PathBuf};

/// The workspace root, walked up from this crate's manifest directory
/// (`<root>/crates/jammi-db`) — same pattern `shipped_feature_exposure.rs`
/// uses, so this test cannot silently stop checking anything if the layout
/// ever changes.
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .find(|dir| dir.join("deny.toml").is_file() && dir.join("Cargo.lock").is_file())
        .unwrap_or_else(|| {
            panic!(
                "no ancestor of {} holds both deny.toml and Cargo.lock",
                manifest.display()
            )
        });
    root.to_path_buf()
}

fn rs_files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading directory {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("reading an entry of {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            rs_files_under(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The enumerating half: no `crates/<name>/src/**/*.rs` file across
/// the whole workspace calls `MemberRoot::new(` (the test-only, arbitrary-
/// string constructor). Runs with no feature requirement — this is a pure
/// static sweep of the tree, so it exercises on every `cargo test`
/// invocation, never only the `test-hooks` lane.
#[test]
fn member_root_new_has_no_production_caller() {
    let root = workspace_root();
    let crates_dir = root.join("crates");
    let mut crate_src_dirs = 0usize;
    let mut files_scanned = 0usize;
    let mut offenders = Vec::new();

    let crate_entries = std::fs::read_dir(&crates_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", crates_dir.display()));
    for entry in crate_entries {
        let entry = entry.unwrap();
        let crate_dir = entry.path();
        if !crate_dir.is_dir() {
            continue;
        }
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        crate_src_dirs += 1;

        let mut files = Vec::new();
        rs_files_under(&src_dir, &mut files);
        for file in files {
            files_scanned += 1;
            let text = std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("reading {}: {e}", file.display()));
            for (idx, line) in text.lines().enumerate() {
                if line.contains("MemberRoot::new(") {
                    offenders.push(format!("{}:{}: {}", file.display(), idx + 1, line.trim()));
                }
            }
        }
    }

    // Sanity on the universe itself: a walk that silently found nothing
    // (a wrong root, an empty `crates/`) would make the assertion below
    // pass vacuously, which is exactly the failure mode this test exists
    // to avoid for `MemberRoot::new`. The workspace has 14 crates with a
    // `src/` directory and 400+ `.rs` files under them at the time this
    // test was written; both floors are set well below that so a modest
    // future removal never flakes this test for an unrelated reason.
    assert!(
        crate_src_dirs >= 10,
        "expected at least 10 crates with a src/ directory under {}, found {crate_src_dirs} \
         -- the walk itself is broken, so the sweep below proves nothing",
        crates_dir.display()
    );
    assert!(
        files_scanned >= 300,
        "expected at least 300 .rs files under crates/*/src, scanned {files_scanned} -- the \
         walk itself is broken, so the sweep below proves nothing"
    );

    assert!(
        offenders.is_empty(),
        "MemberRoot::new (the test-only, arbitrary-string constructor gated behind \
         feature = \"test-hooks\") has a production (crates/*/src, outside tests/) caller -- \
         the ONLY production constructor must be MemberRoot::resolved, which calls \
         JammiConfig::resolved_result_root itself:\n{}",
        offenders.join("\n")
    );
}
