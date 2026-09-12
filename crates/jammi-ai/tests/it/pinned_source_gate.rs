//! The DELTA contract's source gate (`CONTRACT-DELTA-fix3.md`'s Oracles
//! section): "One gate over `crates/jammi-ai/src/pipeline/` for
//! `ctx.table("jammi.` and `sql(`, so a new producer cannot reintroduce the
//! unpinned read. This is the enforcement that survives you; a site list is
//! not the gate."
//!
//! What actually makes either of those two APIs dangerous is the STRING they
//! are handed: a literal `"jammi."` prefix — the bare, session-registered
//! table reference [`jammi_db::store::ResultStore::bind_result_table`]'s doc
//! names as "NOT reliably the catalog's `current_version`". A new persisting
//! producer that types that literal into a `.sql(` or `ctx.table(` call has
//! reintroduced exactly the unpinned-read shape M1 closes, regardless of
//! which of the two APIs it used to do it — so this gate counts occurrences
//! of the literal itself, in non-comment source, per file (walked
//! RECURSIVELY — round 5, M3: a non-recursive `read_dir` silently skipped
//! every file under `src/pipeline/asof/`), against a fixed allowlist of the
//! sites already audited as reading an UNVERSIONED edge/source relation
//! (never the pinned embedding table): see each allowlist entry's comment
//! for its own audit note. A count above its file's allowance is RED: either
//! the new site is a straddle waiting to happen (route it through
//! `ResultStore::pin_current_version` / `pinned_provider` instead), or it is
//! a genuinely new, reviewed exception — raise this file's allowance for it,
//! in the same commit as the review.
//!
//! **Stated limit (round 5, M3), not silently absorbed into the count:**
//! this is a SOURCE-TEXT heuristic scoped to non-comment lines under this
//! crate's `src/pipeline/`. It catches the literal spelled as an
//! interpolated string (`"jammi.{table}"`), a runtime concatenation
//! (`"jammi." + table`), or a `const` prefix declared in-scope — anywhere
//! the substring `"jammi.` appears in quoted text in a scanned file. It
//! CANNOT catch a session-registered reference assembled from a name or
//! constant that never spells `"jammi.` inside this directory at all — e.g.
//! a helper defined in `jammi-db` that returns the fully-built name, or a
//! producer that reaches a version-resolved read through the public store
//! API surface directly (`ResultStore::resolve_version_manifest` +
//! `build_masked_provider` without going through `pin_current_version`)
//! rather than through a `jammi.{table}` string at all. That class is the
//! job of the round-5 M1 sweep documented on
//! [`jammi_db::store::ResultStore::pin_current_version`], not this gate —
//! this gate is deliberately mechanical text-matching, not a semantic
//! understanding of every way to reach an unpinned read.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// `(path relative to `src/pipeline/`, allowed count)`. Every entry here was
/// reviewed as reading an UNVERSIONED edge/source relation, never the pinned
/// embedding table.
///
/// **Keyed on the PATH, not the bare file name (round 6 advisory).** The
/// walk this gate checks against is recursive (round 5, M3), so a bare file
/// name would let a same-named file in a DIFFERENT subdirectory (e.g. a
/// future `src/pipeline/asof/graph_propagation.rs`) silently inherit an
/// allowance reviewed for a wholly different file, with no new review at
/// all. A path is unique per file; a bare name is not.
const ALLOWED: &[(&str, usize)] = &[
    // `edge_scan_sql`'s S9 neighbor_graph edge scan — the EDGE relation, not
    // the embedding table `PinnedSource` covers. Reviewed in the DELTA
    // round-4 contract (M2's graph_propagation.rs`:816` carve-out).
    ("graph_propagation.rs", 1),
    // `load_neighbor_graph_edges` — same class, the S9 edge relation.
    ("graph_neighbourhood.rs", 1),
];

/// Count non-comment-line occurrences of the literal `"jammi.` — the
/// ingredient common to `ctx.table("jammi.{...}")`, a raw
/// `.sql("... jammi.{...} ...")`, and a runtime concatenation or `const`
/// prefix that still spells the quoted text `"jammi."` somewhere in the
/// file — in `text`. A doc-comment mention (used throughout this crate's
/// rustdoc to reference the registration by name) is explicitly NOT a read
/// and must not trip the gate, so `///`/`//!`/`//` lines are skipped.
fn count_bare_jammi_table_literal(text: &str) -> usize {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//")
        })
        .filter(|line| line.contains("\"jammi."))
        .count()
}

/// Every `.rs` file under `dir`, walked RECURSIVELY (round 5, M3: the
/// original walk was `std::fs::read_dir`'s single level, which silently
/// never descended into a subdirectory such as `src/pipeline/asof/`).
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The ground truth this gate's own coverage is checked against: every
/// git-TRACKED `.rs` file under `crates/jammi-ai/src/pipeline` (recursively;
/// `git ls-files` itself recurses), independent of whatever bug the walk
/// above might have. Tracked, not merely present-on-disk, because a tracked
/// file is what CI's own checkout — and therefore what this gate must see —
/// actually contains (the same idiom `ci/scripts/check_ci_guard_wiring.py`
/// uses for its own completeness tripwire).
fn tracked_pipeline_rs_files() -> Vec<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output = std::process::Command::new("git")
        .args(["-C", manifest_dir, "ls-files", "--", "src/pipeline"])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- src/pipeline failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(".rs"))
        .map(str::to_string)
        .collect()
}

#[test]
fn no_new_unpinned_jammi_table_literal_in_pipeline() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest_dir.join("src/pipeline");
    let mut files = Vec::new();
    collect_rs_files(&dir, &mut files);

    let mut reached: HashSet<String> = HashSet::new();
    for path in &files {
        let rel = path
            .strip_prefix(manifest_dir)
            .expect("file under manifest dir")
            .to_str()
            .expect("utf8 path")
            .replace(std::path::MAIN_SEPARATOR, "/");
        reached.insert(rel);

        // Keyed on the path relative to `src/pipeline/` (round 6 advisory),
        // never the bare file name — see `ALLOWED`'s own doc.
        let pipeline_relative = path
            .strip_prefix(&dir)
            .expect("file under src/pipeline")
            .to_str()
            .expect("utf8 path")
            .replace(std::path::MAIN_SEPARATOR, "/");
        let text = std::fs::read_to_string(path).expect("read pipeline source file");
        let count = count_bare_jammi_table_literal(&text);
        let allowed = ALLOWED
            .iter()
            .find(|(f, _)| *f == pipeline_relative)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        assert!(
            count <= allowed,
            "{pipeline_relative}: {count} occurrence(s) of the bare `jammi.{{table}}` literal, \
             {allowed} audited/allowed. A NEW site must read through \
             `ResultStore::pin_current_version`/`pinned_provider`, never construct the \
             session-registered `jammi.{{table}}` reference directly (see \
             `ResultStore::bind_result_table`'s staleness-residual doc). If this site IS an \
             audited exception (an unversioned edge/source relation, not the pinned embedding \
             table), add it to this test's `ALLOWED` list — keyed on this same path, never the \
             bare file name — with the same review its existing entries had; a site list alone \
             is not this gate."
        );
    }

    // The anti-vacuity sentinel (round 5, M3): NOT a bare count threshold —
    // a structurally blind walk (a non-recursive `read_dir`, or an extension
    // filter that quietly excluded a real source file) can clear a bare
    // threshold just as easily as a correct one, which is exactly how the
    // `asof/` subdirectory went unscanned while this test reported success.
    // Instead: name every git-tracked `.rs` file the walk above did NOT
    // reach, which cannot pass while any such file exists, independent of
    // how many files happened to be reached.
    let tracked = tracked_pipeline_rs_files();
    assert!(
        !tracked.is_empty(),
        "git ls-files -- src/pipeline returned no tracked .rs files — the manifest dir, cwd, or \
         pathspec is wrong, which would make the coverage check below vacuously pass"
    );
    let missed: Vec<&String> = tracked.iter().filter(|f| !reached.contains(*f)).collect();
    assert!(
        missed.is_empty(),
        "the pipeline source walk did not reach {} git-tracked .rs file(s), so this gate never \
         scanned them for the bare `jammi.{{` literal: {missed:?}",
        missed.len()
    );
}

#[test]
fn allowed_entries_are_still_present_and_at_their_audited_count() {
    // The inverse control: if an allowlisted site's audited occurrence
    // disappears entirely (the code was refactored away), the allowance
    // should shrink with it rather than sit as permanent dead slack that
    // could hide a LATER, different unpinned read reusing the same budget.
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/pipeline"));
    for (relative_path, allowed) in ALLOWED {
        let text = std::fs::read_to_string(dir.join(relative_path))
            .unwrap_or_else(|e| panic!("read {relative_path}: {e}"));
        let count = count_bare_jammi_table_literal(&text);
        assert_eq!(
            count, *allowed,
            "{relative_path}: expected exactly {allowed} audited occurrence(s) of `jammi.{{`, \
             found {count} — update this allowlist to match, with the same review its other \
             entries had"
        );
    }
}
