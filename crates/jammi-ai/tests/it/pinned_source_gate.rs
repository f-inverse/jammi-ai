//! The DELTA contract's source gate (`CONTRACT-DELTA-fix3.md`'s Oracles
//! section): "One gate over `crates/jammi-ai/src/pipeline/` for
//! `ctx.table("jammi.` and `sql(`, so a new producer cannot reintroduce the
//! unpinned read. This is the enforcement that survives you; a site list is
//! not the gate."
//!
//! What actually makes either of those two APIs dangerous is the STRING they
//! are handed: a literal `"jammi.{...}"` — the bare, session-registered
//! table reference [`jammi_db::store::ResultStore::bind_result_table`]'s doc
//! names as "NOT reliably the catalog's `current_version`". A new persisting
//! producer that types that literal into a `.sql(` or `ctx.table(` call has
//! reintroduced exactly the unpinned-read shape M1 closes, regardless of
//! which of the two APIs it used to do it — so this gate counts occurrences
//! of the literal itself, in non-comment source, per file, against a fixed
//! allowlist of the sites already audited as reading an UNVERSIONED
//! edge/source relation (never the pinned embedding table): see each
//! allowlist entry's comment for its own audit note. A count above its
//! file's allowance is RED: either the new site is a straddle waiting to
//! happen (route it through `ResultStore::pin_current_version` /
//! `pinned_provider` instead), or it is a genuinely new, reviewed exception
//! — raise this file's allowance for it, in the same commit as the review.

use std::path::Path;

/// `(file name, allowed count)`. Every entry here was reviewed as reading an
/// UNVERSIONED edge/source relation, never the pinned embedding table:
const ALLOWED: &[(&str, usize)] = &[
    // `edge_scan_sql`'s S9 neighbor_graph edge scan — the EDGE relation, not
    // the embedding table `PinnedSource` covers. Reviewed in the DELTA
    // round-4 contract (M2's graph_propagation.rs`:816` carve-out).
    ("graph_propagation.rs", 1),
    // `load_neighbor_graph_edges` — same class, the S9 edge relation.
    ("graph_neighbourhood.rs", 1),
];

/// Count non-comment-line occurrences of the literal `jammi.{` — the
/// ingredient common to both `ctx.table("jammi.{...}")` and a raw
/// `.sql("... jammi.{...} ...")` — in `text`. A doc-comment mention (used
/// throughout this crate's rustdoc to reference the registration by name)
/// is explicitly NOT a read and must not trip the gate, so `///`/`//!`/`//`
/// lines are skipped.
fn count_bare_jammi_table_literal(text: &str) -> usize {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//")
        })
        .filter(|line| line.contains("jammi.{"))
        .count()
}

#[test]
fn no_new_unpinned_jammi_table_literal_in_pipeline() {
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/pipeline"));
    let mut checked = 0usize;
    for entry in std::fs::read_dir(dir).expect("read src/pipeline") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        checked += 1;
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf8 file name")
            .to_string();
        let text = std::fs::read_to_string(&path).expect("read pipeline source file");
        let count = count_bare_jammi_table_literal(&text);
        let allowed = ALLOWED
            .iter()
            .find(|(f, _)| *f == file_name)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        assert!(
            count <= allowed,
            "{file_name}: {count} occurrence(s) of the bare `jammi.{{table}}` literal, {allowed} \
             audited/allowed. A NEW site must read through \
             `ResultStore::pin_current_version`/`pinned_provider`, never construct the \
             session-registered `jammi.{{table}}` reference directly (see \
             `ResultStore::bind_result_table`'s staleness-residual doc). If this site IS an \
             audited exception (an unversioned edge/source relation, not the pinned embedding \
             table), add it to this test's `ALLOWED` list with the same review its existing \
             entries had — a site list alone is not this gate."
        );
    }
    // The gate itself must be exercising a non-trivial directory — a
    // directory-listing bug that silently iterated zero files would make
    // every assertion above vacuously true.
    assert!(
        checked > 10,
        "expected to check more than 10 pipeline source files, checked {checked} — \
         the pipeline directory did not resolve as expected"
    );
}

#[test]
fn allowed_entries_are_still_present_and_at_their_audited_count() {
    // The inverse control: if an allowlisted site's audited occurrence
    // disappears entirely (the code was refactored away), the allowance
    // should shrink with it rather than sit as permanent dead slack that
    // could hide a LATER, different unpinned read reusing the same budget.
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/pipeline"));
    for (file_name, allowed) in ALLOWED {
        let text = std::fs::read_to_string(dir.join(file_name))
            .unwrap_or_else(|e| panic!("read {file_name}: {e}"));
        let count = count_bare_jammi_table_literal(&text);
        assert_eq!(
            count, *allowed,
            "{file_name}: expected exactly {allowed} audited occurrence(s) of `jammi.{{`, found \
             {count} — update this allowlist to match, with the same review its other entries \
             had"
        );
    }
}
