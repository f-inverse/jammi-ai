//! G6 (#515): terminality is ONE predicate,
//! [`jammi_db::catalog::status::JobStatus::is_terminal`]. This gate asserts
//! — by scanning tracked source, never by prose — that no OTHER site in
//! either language hand-enumerates the job-status terminality vocabulary as
//! a literal string compare (`status == "completed"`, `status != "failed"`,
//! and their reversed/`match`-arm shapes). A literal compare outside the one
//! predicate is exactly the class of defect that made `WaitJob`, the
//! embedded `Job.wait`/`TrainingJob.wait`/`RemoteJob.wait`, and the Python
//! client's `_TERMINAL_STATES` liable to silently hang or misclassify the
//! moment this vocabulary ever grows a new terminal member (the job-
//! dependency graph, #515, deferred — this gate is the standalone
//! terminality-hygiene fix that survives its deferral, without adding the
//! `Cancelled` status that unit would need: a status with no writer is dead
//! vocabulary) — see `crates/jammi-ai/tests/it/pinned_source_gate.rs` for
//! the established idiom this file follows: the scanned universe is derived
//! from `git ls-files`, never a hand-rolled directory walk, so a file the
//! scan should reach but cannot read is a hard failure naming it, never a
//! silent skip.
//!
//! **Scope.** The WHOLE tracked tree: `crates/*/src/**/*.rs`,
//! `crates/*/tests/**/*.rs`, `clients/python/jammi/**/*.py` AND
//! `clients/python/tests/**/*.py`. Test fixtures were originally scoped out
//! ("a fixture polling for one known literal is a different risk class from
//! a production dispatch path"), but a fixture that polls for exactly ONE
//! terminal literal (`wait_until(|r| r.status == "completed")`) hangs (or, in
//! the harnesses with a bounded backstop, burns the FULL timeout) the moment
//! its job reaches a DIFFERENT terminal status than the one literal it
//! checks — every one of the ~25 prior occurrences is now rewritten to
//! compare against `JobStatus::<Variant>.to_string()` (or `JobRecord::
//! is_terminal`/`is_terminal_unsuccessful` where the intent was "any
//! terminal", not one specific literal), so the universe below is
//! enumerated over the WHOLE tree and holds the line for both scopes
//! identically going forward.
//!
//! **The SUCCESS predicate is a separate, exempt decision.** Whether a job's
//! `result` is populated is decided by comparing DIRECTLY against
//! `JobStatus::Completed` (never a bare `"completed"` literal in Rust; the
//! Python client's one instance is line-allowlisted below) — `grpc/job.rs`'s
//! `job_status_response_from_record` and `clients/python/jammi/_database.py`'s
//! `RemoteJob.wait` success arm. This is NOT a terminality decision and is
//! never folded into `is_terminal`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The repo root, derived from this crate's manifest dir
/// (`crates/jammi-db`), matching `pinned_source_gate.rs`'s own derivation.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-db has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-TRACKED file under `root`-relative `dir` whose name ends with
/// `ext`, sorted. `git ls-files` recurses on its own.
fn tracked_files(root: &Path, dir: &str, ext: &str) -> Vec<String> {
    let output = Command::new("git")
        .args([
            "-C",
            root.to_str().expect("utf8 repo root"),
            "ls-files",
            "--",
            dir,
        ])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- {dir} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut files: Vec<String> = String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(ext))
        .map(str::to_string)
        .collect();
    files.sort();
    files
}

/// Every tracked `.rs` file under every `crates/*/src` OR `crates/*/tests`
/// directory — production AND test source alike. Discovered via `git
/// ls-files` with a glob pathspec (letting git enumerate the crate list)
/// rather than a hand-typed crate name list, so a new crate's `src/`/
/// `tests/` is scanned automatically.
// `concat!` splits every `/` immediately followed by `*` onto opposite
// sides of a literal boundary, so NO 2-byte `/*`-shaped substring survives
// anywhere in the SOURCE TEXT (`"crates/*/src/**"` has two: the mid-path
// `/*` glob and the unclosed trailing `/**`) — `ci/scripts/
// check_kernel_oracles.py`'s independent comment-only cross-checker has NO
// string/char awareness by design and reads a raw `/*` inside ANY string
// as a real (and here, never-closed) block-comment opener. The runtime
// STRING VALUE is unaffected, only how it is spelled in source.
const RUST_SURFACE_PATHSPECS: &[&str] = &[
    concat!("crates/", "*", "/src/", "*", "*"),
    concat!("crates/", "*", "/tests/", "*", "*"),
];
/// Every tracked `.py` file under the embed client package AND its own test
/// suite.
const PYTHON_SURFACE_PATHSPECS: &[&str] = &["clients/python/jammi", "clients/python/tests"];

/// `status.rs` is the one predicate's own file: its `Display`/`FromStr`
/// implementations ARE the literal-to-variant mapping, and its own
/// `#[cfg(test)]` round-trip assertions compare `JobStatus::Failed.to_string()`
/// against `"failed"` BY DESIGN (the oracle that the mapping is correct).
///
/// `recovery.rs`'s one hit is a DIFFERENT vocabulary entirely —
/// `JammiError::CasFailed { status, .. }` names a `result_tables`
/// (`ResultTableStatus`) CAS failure, not `jobs.status`; the destructured
/// binding is merely also spelled `status`. Exempted BY LINE CONTENT (this
/// file has no OTHER hit, so a future `jobs`-status literal landing here
/// still fails the gate).
const RUST_EXEMPT_FILES: &[&str] = &["crates/jammi-db/src/catalog/status.rs"];
const RUST_EXEMPT_LINE_SUBSTRINGS: &[&str] = &["JammiError::CasFailed { table, status }"];

/// Python's ONE predicate definition site and its one, separately-named
/// SUCCESS-predicate line (see module docs) — both in
/// `clients/python/jammi/_database.py`. Exempted BY LINE CONTENT, not by
/// file, so a NEW literal compare landing anywhere else in this same file
/// still fails the gate.
const PYTHON_EXEMPT_LINE_SUBSTRINGS: &[&str] = &[
    "_TERMINAL_STATES = {",
    "_TERMINAL_UNSUCCESSFUL_STATES = {",
    // The SUCCESS predicate (module docs): mirrors `grpc/job.rs`'s
    // `record.status == JobStatus::Completed.to_string()`.
    "if resp.status == \"completed\":",
];

/// One hit: `(repo-relative path, 1-based line number, the line's own
/// trimmed text)`. Every field is read only through the `Debug` derive (the
/// failure message), which rustc's `dead_code` lint does not credit as a
/// use.
#[derive(Debug)]
#[allow(dead_code)]
struct Hit {
    path: String,
    line_no: usize,
    text: String,
}

/// The identifier-ish token (`.`/`_`/alnum) immediately touching `s`'s END —
/// e.g. for `"    record.status"` this returns `"record.status"`.
fn trailing_ident(s: &str) -> &str {
    let end = s.len();
    let start = s
        .rfind(|c: char| !(c.is_alphanumeric() || c == '.' || c == '_'))
        .map(|i| i + 1)
        .unwrap_or(0);
    &s[start..end]
}

/// The identifier-ish token (`.`/`_`/alnum) immediately touching `s`'s
/// START — e.g. for `"record.status {"` this returns `"record.status"`.
fn leading_ident(s: &str) -> &str {
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '.' || c == '_'))
        .unwrap_or(s.len());
    &s[..end]
}

/// A literal job-status terminality compare, in either direction: `<expr
/// ending in "status"> (==|!=) "<vocab>"` or the reverse `"<vocab>"
/// (==|!=) <expr ending in "status">`. The vocabulary scanned is
/// [`jammi_db::catalog::status::JobStatus::ALL`]'s rendered set, read from the
/// enum, so a status joining it is scanned for with no edit here.
fn find_literal_status_compares(text: &str) -> Vec<(usize, String)> {
    let vocabulary: Vec<String> = jammi_db::catalog::status::JobStatus::ALL
        .iter()
        .map(|status| status.to_string())
        .collect();
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        // Comments and doc comments never execute a compare.
        if trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        for vocab in &vocabulary {
            let forward = format!("\"{vocab}\"");
            if let Some(pos) = line.find(&forward) {
                let before = line[..pos].trim_end();
                let after = line[pos + forward.len()..].trim_start();
                // Forward: `<operand> (==|!=) "<vocab>"` — the operand ends
                // the text before the operator.
                let forward_hit = (before.ends_with("==") || before.ends_with("!="))
                    && trailing_ident(before[..before.len() - 2].trim_end()).ends_with("status");
                // Reverse: `"<vocab>" (==|!=) <operand>` — the operand
                // starts the text after the operator.
                let reverse_hit = (after.starts_with("==") || after.starts_with("!="))
                    && leading_ident(after[2..].trim_start()).ends_with("status");
                if forward_hit || reverse_hit {
                    out.push((idx + 1, line.trim().to_string()));
                }
            }
        }
    }
    out
}

fn scan(
    root: &Path,
    pathspecs: &[&str],
    ext: &str,
    exempt_files: &[&str],
    exempt_line_substrings: &[&str],
) -> Vec<Hit> {
    let exempt: HashSet<&str> = exempt_files.iter().copied().collect();
    let mut hits = Vec::new();
    for pathspec in pathspecs {
        let files = tracked_files(root, pathspec, ext);
        assert!(
            !files.is_empty(),
            "git ls-files -- {pathspec} returned no tracked {ext} files — the pathspec is \
             wrong, which would make this gate vacuously pass"
        );
        for path in files {
            if exempt.contains(path.as_str()) {
                continue;
            }
            let full = root.join(&path);
            let text = std::fs::read_to_string(&full).unwrap_or_else(|e| {
                panic!("git ls-files reported {path} as tracked, but it could not be read: {e}")
            });
            for (line_no, line_text) in find_literal_status_compares(&text) {
                if exempt_line_substrings.iter().any(|s| line_text.contains(s)) {
                    continue;
                }
                hits.push(Hit {
                    path: path.clone(),
                    line_no,
                    text: line_text,
                });
            }
        }
    }
    hits
}

/// The Rust half. Scans `crates/*/src` AND `crates/*/tests` (the WHOLE
/// tracked Rust tree) for a literal job-status terminality compare outside
/// `status.rs`. RED before this unit's fixes (executed): reverting
/// `grpc/job.rs`'s success-result guard from `record.status ==
/// JobStatus::Completed.to_string()` back to the pre-fix
/// `record.status == "completed"` reds naming
/// `crates/jammi-server/src/grpc/job.rs`; reverting `acceleration_report.rs`'s
/// `wait_for_any_terminal` from `record.is_terminal()` back to
/// `matches!(record.status.as_str(), "completed" | "failed")` (a `matches!`
/// arm, not an `==`, so NOT independently caught by this test — the class
/// this detector's `==`/`!=` shape cannot see; tracked in Uncovered) is a
/// separate, already-fixed site; the CANONICAL red for the widened universe
/// is reverting `crates/jammi-ai/tests/it/jammi.rs`'s
/// `status != jammi_db::catalog::status::JobStatus::Queued.to_string()` back
/// to `status != "queued"`, which reds naming that exact file/line.
#[test]
fn no_literal_job_status_terminality_compare_outside_the_one_predicate_rust() {
    let root = repo_root();
    let hits = scan(
        &root,
        RUST_SURFACE_PATHSPECS,
        ".rs",
        RUST_EXEMPT_FILES,
        RUST_EXEMPT_LINE_SUBSTRINGS,
    );
    assert!(
        hits.is_empty(),
        "literal job-status terminality compare(s) found outside \
         crates/jammi-db/src/catalog/status.rs (derive from JobStatus::is_terminal \
         instead): {hits:#?}"
    );
}

/// The Python half. Scans `clients/python/jammi` AND `clients/python/tests`
/// for a literal job-status terminality compare outside the two designated
/// set constants and the one named success line. RED before this unit's fix
/// (executed): reverting `RemoteJob.wait`'s
/// `if resp.status in _TERMINAL_UNSUCCESSFUL_STATES:` back to the pre-fix
/// `if resp.status == "failed":` reintroduces exactly the bare-literal shape
/// this test forbids, and it reports that one hit, naming
/// `clients/python/jammi/_database.py`.
#[test]
fn no_literal_job_status_terminality_compare_outside_the_one_predicate_python() {
    let root = repo_root();
    let hits = scan(
        &root,
        PYTHON_SURFACE_PATHSPECS,
        ".py",
        &[],
        PYTHON_EXEMPT_LINE_SUBSTRINGS,
    );
    assert!(
        hits.is_empty(),
        "literal job-status terminality compare(s) found outside \
         clients/python/jammi/_database.py's _TERMINAL_STATES/_TERMINAL_UNSUCCESSFUL_STATES \
         (derive from those sets instead): {hits:#?}"
    );
}

#[cfg(test)]
mod detector_self_tests {
    use super::find_literal_status_compares;

    /// Mutation executed: remove the `status` suffix requirement from
    /// `find_literal_status_compares` (match ANY `"completed"`/`"failed"`
    /// literal regardless of context) -> this test's negative case
    /// (`"the connection failed"`, an unrelated error message, no `status`
    /// anywhere on the line) starts reporting a false-positive hit, which
    /// this test's `assert!(hits.is_empty())` catches (RED).
    #[test]
    fn an_unrelated_string_literal_is_not_a_hit() {
        let text = "    return Err(format!(\"the connection failed: {e}\"));\n";
        assert!(
            find_literal_status_compares(text).is_empty(),
            "an unrelated 'failed' substring with no status compare must not be flagged"
        );
    }

    #[test]
    fn a_direct_literal_compare_is_a_hit() {
        let text = "    if record.status == \"completed\" {\n";
        let hits = find_literal_status_compares(text);
        assert_eq!(hits.len(), 1, "must flag record.status == \"completed\"");
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn a_reversed_literal_compare_is_a_hit() {
        let text = "    if \"failed\" == record.status {\n";
        let hits = find_literal_status_compares(text);
        assert_eq!(hits.len(), 1, "must flag the reversed form too");
    }

    #[test]
    fn a_derived_to_string_compare_is_not_a_hit() {
        let text = "    if record.status == JobStatus::Completed.to_string() {\n";
        assert!(
            find_literal_status_compares(text).is_empty(),
            "a compare against a rendered enum variant (not a bare literal) is the CORRECT \
             idiom and must not be flagged"
        );
    }

    #[test]
    fn a_comment_containing_the_shape_is_not_a_hit() {
        let text = "    // record.status == \"completed\" is the OLD shape, fixed below\n";
        assert!(
            find_literal_status_compares(text).is_empty(),
            "a comment must never be flagged"
        );
    }
}
