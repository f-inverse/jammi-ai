//! `CONTRACT-U5a.md` §I1(a) / §W2 Resolution — two enumerating-caller
//! oracles, each a MEASURED claim (never prose): `Catalog::get_job_for_rank`
//! and `Catalog::get_result_table_for_tenant` are each called from nowhere
//! outside the gang `RunRank` handler (plus each function's own crate's
//! tests, which call it directly to exercise it in isolation).
//!
//! The scanned surface is derived from `git ls-files` (never a hand-rolled
//! directory walk) over the whole tracked tree — `crates/**` and everything
//! else — matching `whose_fault_gate.rs`'s own precedent for this shape of
//! claim. A tracked file `git ls-files` reports that this process cannot
//! then read is a hard failure naming the file. The detector is a plain
//! substring match on the call-token (`name(`) — cheap and sufficient here:
//! neither function name collides with any other identifier in this tree
//! (checked by the fact that the allowlist below is exhaustive and the
//! test is green), so no comment/string masking is needed the way
//! `whose_fault_gate.rs`'s heavier detector requires for a much more common
//! token (`JammiError::Schema {`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("git rev-parse must run");
    assert!(out.status.success(), "git rev-parse --show-toplevel failed");
    PathBuf::from(
        String::from_utf8(out.stdout)
            .expect("utf8 path")
            .trim()
            .to_string(),
    )
}

fn git_ls_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .arg("ls-files")
        .current_dir(root)
        .output()
        .expect("git ls-files must run");
    assert!(out.status.success(), "git ls-files failed");
    String::from_utf8(out.stdout)
        .expect("utf8 file list")
        .lines()
        .map(str::to_string)
        .collect()
}

/// Every tracked `.rs` file containing `token` as a literal substring,
/// relative to the repo root. Hard-fails (naming the file) if `git
/// ls-files` reports a tracked file this process cannot then read.
fn files_containing(token: &str) -> HashSet<String> {
    let root = repo_root();
    let mut hits = HashSet::new();
    for rel in git_ls_files(&root) {
        if !rel.ends_with(".rs") {
            continue;
        }
        let path = root.join(&rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("git ls-files tracked {rel} but it could not be read: {e}"));
        if text.contains(token) {
            hits.insert(rel);
        }
    }
    hits
}

/// `CONTRACT-U5a.md` §I1(a): "the enumerating-caller oracle over `crates/**`"
/// — `get_job_for_rank`'s only production caller is the gang `RunRank`
/// handler; the sole other hit is `jammi-db`'s own unit test exercising the
/// method directly.
#[test]
fn only_the_gang_run_rank_handler_calls_get_job_for_rank() {
    let hits = files_containing("get_job_for_rank(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/jobs_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",     // the ONE production caller
        "crates/jammi-db/tests/it/gang_rank_admission.rs", // jammi-db's own unit tests
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_job_for_rank(` occurrence outside the allowed set: {hit} \
             (allowed: {allowed:?}) — a new caller of this primary-key-only, \
             non-tenant-scoped verb must be reviewed against CONTRACT-U5a.md §I1(a) \
             before this allowlist grows"
        );
    }
    for must_hit in &allowed {
        assert!(
            hits.contains(*must_hit),
            "{must_hit} is in the allowlist but no longer contains \
             `get_job_for_rank(` — shrink the allowlist rather than leaving a stale entry"
        );
    }
}

/// `CONTRACT-U5a.md` §W2 Resolution (round-11 fold, ruling 5): "no caller
/// other than the gang `RunRank` handler resolves `training_set_location`"
/// — `get_result_table_for_tenant`'s only production caller is
/// `resolve_training_set_identity` inside the gang handler; the two other
/// hits are `gang_service.rs`'s own b1' tests (rulings 4/5) exercising the
/// raw verb directly to demonstrate the hazard those rulings name.
#[test]
fn only_resolve_training_set_identity_calls_get_result_table_for_tenant() {
    let hits = files_containing("get_result_table_for_tenant(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/result_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",       // the ONE production caller
        "crates/jammi-server/tests/it/gang_service.rs", // this crate's own b1' tests
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_result_table_for_tenant(` occurrence outside the allowed \
             set: {hit} (allowed: {allowed:?}) — a new caller of this strict-tenant \
             verb must be reviewed against CONTRACT-U5a.md §W2 Resolution's admin-scope \
             hazard (ruling 4) before this allowlist grows"
        );
    }
    for must_hit in &allowed {
        assert!(
            hits.contains(*must_hit),
            "{must_hit} is in the allowlist but no longer contains \
             `get_result_table_for_tenant(` — shrink the allowlist rather than leaving \
             a stale entry"
        );
    }
}
