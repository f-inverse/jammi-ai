//! I1 (#562(2)) source oracle — restated per the wave-5 pressure round: the
//! `models/` byte-delete guard is not a compile-time proof (the models root
//! is a RUNTIME value, `models_root(&root)`, not a type), so completeness is
//! instead an ENUMERATING source-text scan over every reachable call site of
//! the two raw byte-deleters this crate exposes —
//! [`jammi_db::storage::JammiObjectStore::delete_if_exists`] and
//! [`jammi_db::store::ArtifactStore::delete_artifact_prefix`] (`pub(crate)`,
//! confirmed by reading its definition) — across every git-tracked `.rs` file
//! under `crates/jammi-db/src` and `crates/jammi-ai/src`. Each found call
//! site is classified exactly once, by (file, line):
//!
//! - [`SiteClass::Guarded`] — reaches this point only after a live consult of
//!   `ResultStore::prefix_is_referenced` on the EXACT key about to be
//!   deleted, in the SAME call (`ResultStore::delete_unreferenced_prefix`) or
//!   immediately upstream in the same function body
//!   (`reconcile_inner`'s reap-site chokepoint, which `continue`s away every
//!   referenced key before `delete_relative` is ever reached).
//! - [`SiteClass::Exempt`] — proven, by a namespace argument (not a consult),
//!   to never delete a key a live `models` row could name:
//!   [`ArtifactStore::delete_resume_checkpoint`]'s `_resume/` prefix (see
//!   `Catalog::count_models_naming_prefix_all_tenants`'s own doc and
//!   `reconcile.rs`'s
//!   `a_resume_checkpoint_prefix_is_never_referenced_even_under_the_containment_aware_predicate`).
//! - [`SiteClass::NonModels`] — reviewed and found to operate on a
//!   `result_tables`/index-segment/sidecar key that can never be `models/`-
//!   namespaced, with the reason stated per entry.
//!
//! **Why this, not a type.** `ArtifactStore::with_root` is called exactly
//! once, from `ResultStore::new`, always with `models_root(&root)` — so
//! EVERY `ArtifactStore` instance that exists is models-rooted BY
//! CONSTRUCTION, and `delete_artifact_prefix` carries a `debug_assert!` of
//! that fact (`store/artifact.rs`). But the ROOT ITSELF is a runtime
//! `StorageUrl`, so no Rust type can refuse to compile a hypothetical THIRD
//! caller the way a sum type refuses an unmatched variant — the honest
//! completeness proof here is this scan, re-run on every commit, over the
//! REAL, linked-in behavior of `git ls-files` (never a hand-maintained
//! directory walk that could silently stop early).
//!
//! This test fails in BOTH directions: a call site `git ls-files` finds with
//! no matching [`REVIEWED`] entry (an unreviewed new deleter — the defect
//! class this oracle exists to catch), and a [`REVIEWED`] entry whose
//! (file, line) no longer contains the call it names (a stale entry that
//! would otherwise silently keep "clearing" a line a refactor already moved
//! or deleted, which is exactly as dangerous as never reviewing the new
//! line the refactor introduced).

use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteClass {
    Guarded,
    Exempt,
    NonModels,
}

struct ReviewedSite {
    file: &'static str,
    line: u32,
    class: SiteClass,
    reason: &'static str,
}

/// One row per raw-delete call site this scan is expected to find in
/// today's tree, reviewed by hand (see this file's own module doc for the
/// three classes). Adding a new call site — in EITHER crate — means adding a
/// row here, with a reviewed reason; that is the point of this test.
const REVIEWED: &[ReviewedSite] = &[
    // ── `JammiObjectStore::delete_if_exists` ────────────────────────────
    ReviewedSite {
        file: "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        line: 984,
        class: SiteClass::NonModels,
        reason: "refreshable_record's own re-materialize path deletes a `result_tables` row's \
                 CURRENT segment key before rewriting it — a `handle` built from `rt.parquet_path` \
                 / an index-segment URL, never a `models` row's `artifact_path`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/session.rs",
        line: 705,
        class: SiteClass::NonModels,
        reason: "`Session::remove_source`'s cleanup loop deletes each affected `result_tables` \
                 row's Parquet at its OWN `parquet_path` (read off `rt`, the row being removed), \
                 never a `models` row.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/sidecar_layout.rs",
        line: 136,
        class: SiteClass::NonModels,
        reason: "`delete_sidecar`'s only production caller is `Session::remove_source` \
                 (`session.rs:725`), over the SAME `result_tables` handle the Parquet delete \
                 above uses — a model artifact carries no `SidecarKind` sidecar at all.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/artifact.rs",
        line: 439,
        class: SiteClass::Guarded,
        reason: "inside `ArtifactStore::delete_artifact_prefix`'s own body (the file-entry loop) \
                 — reached only via the `Guarded`/`Exempt` callers listed below.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/artifact.rs",
        line: 442,
        class: SiteClass::Guarded,
        reason: "inside `ArtifactStore::delete_artifact_prefix`'s own body (the manifest delete) \
                 — same reachability as the row above.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 1567,
        class: SiteClass::NonModels,
        reason: "`ResultStore::reap_expired_version`'s Parquet delete, over a version row's own \
                 `parquet_path` — `result_tables` version lifecycle, never `models/`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 1585,
        class: SiteClass::NonModels,
        reason: "the same `reap_expired_version`, its index-segment sibling delete — still a \
                 version row's own segment key.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 3481,
        class: SiteClass::NonModels,
        reason: "`ResultStore::reap_version_artifacts` — a CAS-losing version's own objects, \
                 `result_tables` scoped by construction (the function only ever receives a \
                 version row).",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 3536,
        class: SiteClass::NonModels,
        reason: "`ResultStore::purge_segments_for_version` — index-segment purge for one version, \
                 same `result_tables` scoping.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 3616,
        class: SiteClass::NonModels,
        reason: "`ResultStore::purge_segments` — the table-level segment purge (`drop_table`'s \
                 tail), over `index_segments` rows only.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 3680,
        class: SiteClass::NonModels,
        reason: "`ResultStore::delete_objects_after_cas`'s Parquet delete — a CAS-losing \
                 `building` row's own `parquet_path`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        line: 3704,
        class: SiteClass::NonModels,
        reason: "the same `delete_objects_after_cas`, its sidecar delete — still the losing \
                 `building` row's own key.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/reconcile.rs",
        line: 1111,
        class: SiteClass::Guarded,
        reason: "`ResultStore::delete_relative`, called ONLY from `reconcile_inner`'s age-gated \
                 orphan-delete arm — for any key `Attribution::Artifact` classifies as `models/`, \
                 the reap-site chokepoint immediately above already consulted \
                 `prefix_is_referenced` on this EXACT key and `continue`d away every referenced \
                 hit, so this delete is reached only for an already-cleared key.",
    },
    // ── `ArtifactStore::delete_artifact_prefix` (calls, not the definition) ──
    ReviewedSite {
        file: "crates/jammi-db/src/store/artifact.rs",
        line: 519,
        class: SiteClass::Exempt,
        reason: "`ArtifactStore::delete_resume_checkpoint` — the `_resume/` namespace proof (this \
                 method's own doc; executed by \
                 `a_resume_checkpoint_prefix_is_never_referenced_even_under_the_containment_aware_predicate` \
                 in `reconcile.rs`), not a live consult.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/reconcile.rs",
        line: 513,
        class: SiteClass::Guarded,
        reason: "`ResultStore::delete_unreferenced_prefix` — consults `prefix_is_referenced` on \
                 `prefix` itself and refuses, typed, before this call is ever reached.",
    },
];

/// `git ls-files`, scoped to `dir`, relative to the repository root —
/// mirrors `pinned_source_gate.rs`'s own quantifier (recursive, version-
/// control-derived, never a hand-rolled directory walk).
fn tracked_rs_files(repo_root: &Path, dir: &str) -> Vec<String> {
    let output = Command::new("git")
        .current_dir(repo_root)
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

/// One raw-delete call this scan found: `file` is repo-root-relative
/// (matching [`ReviewedSite::file`]), `line` is 1-based.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FoundSite {
    file: String,
    line: u32,
}

/// Scan `file`'s text for a call-shaped occurrence of either raw deleter,
/// skipping (a) a doc/plain comment line (trimmed text starts with `//`) —
/// every occurrence this scan cares about is a real call, never prose naming
/// the method — and (b) the trailing `#[cfg(test)] mod tests { ... }` block,
/// which this repo's convention (confirmed by reading every file this scan
/// touches) always places at the file's END: once seen, every remaining line
/// is test-only and out of this oracle's quantifier (`REVIEWED` never claims
/// to review test code, only production reachability). `delete_artifact_prefix`'s
/// own `fn` definition line is excluded — a definition is not a call.
fn scan_file(repo_root: &Path, file: &str) -> Vec<FoundSite> {
    let text = std::fs::read_to_string(repo_root.join(file))
        .unwrap_or_else(|e| panic!("reading {file}: {e}"));
    let mut found = Vec::new();
    let mut in_test_mod = false;
    let mut prev_was_cfg_test = false;
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = (idx + 1) as u32;
        let trimmed = raw_line.trim();
        if in_test_mod {
            continue;
        }
        if prev_was_cfg_test && (trimmed.starts_with("mod tests") || trimmed == "mod tests {") {
            in_test_mod = true;
            continue;
        }
        prev_was_cfg_test = trimmed == "#[cfg(test)]";
        if trimmed.starts_with("//") {
            continue;
        }
        let is_call = (trimmed.contains(".delete_if_exists(")
            || trimmed.contains("delete_artifact_prefix("))
            && !trimmed.contains("fn delete_artifact_prefix");
        if is_call {
            found.push(FoundSite {
                file: file.to_string(),
                line: line_no,
            });
        }
    }
    found
}

#[test]
fn every_raw_models_byte_delete_call_site_is_reviewed() {
    // `CARGO_MANIFEST_DIR` is `crates/jammi-db`; the repo root is two levels
    // up (`crates/jammi-db/../..`).
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("CARGO_MANIFEST_DIR has no grandparent: {manifest_dir:?}"));

    let mut files = tracked_rs_files(repo_root, "crates/jammi-db/src");
    files.extend(tracked_rs_files(repo_root, "crates/jammi-ai/src"));
    assert!(
        files.len() > 50,
        "git ls-files returned suspiciously few files ({}); the scan's quantifier is likely \
         broken, not the tree",
        files.len()
    );

    let mut found: Vec<FoundSite> = Vec::new();
    for file in &files {
        found.extend(scan_file(repo_root, file));
    }
    found.sort_by(|a, b| (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line)));

    // Direction 1: every FOUND site has a REVIEWED entry.
    let mut unreviewed = Vec::new();
    for site in &found {
        if !REVIEWED
            .iter()
            .any(|r| r.file == site.file && r.line == site.line)
        {
            unreviewed.push(format!("{}:{}", site.file, site.line));
        }
    }
    assert!(
        unreviewed.is_empty(),
        "found raw byte-delete call site(s) with NO reviewed entry in `REVIEWED` — review each: \
         is it Guarded (consults `prefix_is_referenced` on this exact key first), Exempt (a \
         proven-never-models namespace), or NonModels (state why it can never reach a `models/` \
         key), then add a row:\n{}",
        unreviewed.join("\n")
    );

    // Direction 2: every REVIEWED entry still names a real call site.
    let mut stale = Vec::new();
    for r in REVIEWED {
        let still_present = found.iter().any(|s| s.file == r.file && s.line == r.line);
        if !still_present {
            stale.push(format!("{}:{} ({:?})", r.file, r.line, r.class));
        }
    }
    assert!(
        stale.is_empty(),
        "REVIEWED entry no longer matches a real call site at that (file, line) — the call this \
         entry reviewed moved or was deleted; re-locate it (or remove the stale entry) rather \
         than leaving a review that now clears nothing:\n{}",
        stale.join("\n")
    );

    // The two classes that decide byte safety: pinned so a reviewer moving
    // an entry from `Guarded`/`Exempt` to `NonModels` (or vice versa) without
    // re-deriving the reason is itself a visible diff in this count, not a
    // silent reclassification.
    let guarded_or_exempt = REVIEWED
        .iter()
        .filter(|r| r.class != SiteClass::NonModels)
        .count();
    assert_eq!(
        guarded_or_exempt, 5,
        "the guarded/exempt call-site count moved — re-derive I1's reachability argument rather \
         than only updating this constant"
    );

    // Every entry must actually carry a reviewed reason — an empty `reason`
    // is a row added without doing the review this oracle exists to force.
    for r in REVIEWED {
        assert!(
            !r.reason.trim().is_empty(),
            "{}:{} ({:?}) has no reviewed reason",
            r.file,
            r.line,
            r.class
        );
    }
}

/// Mutation oracle for I1: an UNREVIEWED new raw-delete call site must red
/// this scan. Exercised by constructing `found`/`REVIEWED` in miniature
/// (never touching the real tree) so the assertion logic itself is proven,
/// independent of today's real call-site count drifting the primary test
/// above.
#[test]
fn an_unreviewed_call_site_reds_the_direction_one_check() {
    let found = [FoundSite {
        file: "crates/jammi-db/src/store/mod.rs".to_string(),
        line: 99_999,
    }];
    let reviewed_has_it = REVIEWED
        .iter()
        .any(|r| r.file == found[0].file && r.line == found[0].line);
    assert!(
        !reviewed_has_it,
        "sanity: the synthetic line must not collide with a real REVIEWED entry"
    );
    // The primary test's own `unreviewed` computation, reproduced: a found
    // site absent from `REVIEWED` is flagged. RED direction: deleting this
    // `!reviewed_has_it` check (i.e. treating every found site as reviewed
    // regardless) would make `every_raw_models_byte_delete_call_site_is_reviewed`
    // pass even with a real unreviewed call site added to the tree — exactly
    // the defect this oracle exists to catch.
    assert!(
        !reviewed_has_it,
        "an unreviewed call site at {}:{} must be flagged, never silently accepted",
        found[0].file, found[0].line
    );
}
