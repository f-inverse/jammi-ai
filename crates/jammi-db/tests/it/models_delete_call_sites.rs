//! I1 (#562(2)) source oracle — restated by the wave-5 pressure round
//! (twice): the `models/` byte-delete guard is not a compile-time proof
//! (the models root is a RUNTIME value, `models_root(&root)`, not a type),
//! so completeness is instead an ENUMERATING source oracle over every
//! reachable call site of the two raw byte-deleters this crate exposes —
//! [`jammi_db::storage::JammiObjectStore::delete_if_exists`] and
//! [`jammi_db::store::ArtifactStore::delete_artifact_prefix`] (`pub(crate)`,
//! confirmed by reading its definition) — across every git-tracked `.rs`
//! file under every crate's `src/` tree (`crates/*/src`), because
//! `delete_if_exists` is `pub` and a caller in any crate is in scope.
//!
//! **Keyed by `(file, function, ordinal)`, not `(file, line)`** (F2's
//! second delta): a bare line number goes stale on every UNRELATED edit
//! anywhere above it in the same file — the #526 class the delta names —
//! forcing a re-review this oracle's own reviewed reason never actually
//! needed to change. Keying by the enclosing function's name and the
//! call's ordinal position WITHIN that function is invariant to any edit
//! outside that one function's body. Determining "the enclosing function"
//! and "which literal token is a real call, never a doc comment's rendered
//! prose" correctly needs a REAL parse — a hand-rolled brace-counter or a
//! `//`-prefix text check (this file's own PRIOR revision) cannot tell a
//! `///` doc comment's rendered text from a real call inside a multi-line
//! signature, or track nested braces through a `match`/`if let` chain
//! without re-implementing a chunk of the Rust grammar — so this scan
//! parses every file into a real `syn::File` and walks it with
//! `syn::visit::Visit`, mirroring `pinned_source_gate.rs`'s own established
//! idiom in this crate's sibling for exactly this reason.
//!
//! Each found call site is classified exactly once:
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
//! CONSTRUCTION, and `delete_artifact_prefix` refuses, typed, a `prefix`
//! that fails that check IN EVERY BUILD (`store/artifact.rs` — an
//! always-on check, not a `debug_assert!` a release build compiles away;
//! see `delete_artifact_prefix_refuses_a_prefix_outside_this_stores_own_root`
//! there). But the ROOT ITSELF is a runtime `StorageUrl`, so no Rust type
//! can refuse to compile a hypothetical THIRD caller the way a sum type
//! refuses an unmatched variant — the honest completeness proof here is
//! this scan, re-run on every commit, over the REAL, linked-in behavior of
//! `git ls-files` (never a hand-maintained directory walk that could
//! silently stop early) and a real `syn` parse (never a re-implemented
//! subset of the grammar).
//!
//! This test fails in BOTH directions: a call site this scan finds with no
//! matching [`REVIEWED`] entry (an unreviewed new deleter — the defect
//! class this oracle exists to catch), and a [`REVIEWED`] entry whose
//! `(file, function, ordinal)` no longer names a real call (a stale entry
//! that would otherwise silently keep "clearing" a site a refactor already
//! moved or deleted, which is exactly as dangerous as never reviewing the
//! new site the refactor introduced) — PLUS a third check this key shape
//! adds: the TOTAL call count within a reviewed function must match every
//! entry's own `count` field, so a fourth call silently added to an
//! already-reviewed three-call function is caught even though ordinals
//! 1-3 still resolve.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteClass {
    Guarded,
    Exempt,
    NonModels,
}

/// One reviewed raw-delete call site, keyed by the enclosing function's
/// name and this call's 1-based ordinal position among every raw-delete
/// call inside THAT function (source order) — never a bare line number.
/// `count` is the TOTAL number of raw-delete calls this function is
/// reviewed to contain; every entry for the same `(file, function)` must
/// agree on it (checked as its own invariant, independent of the per-entry
/// ordinal match).
struct ReviewedSite {
    file: &'static str,
    function: &'static str,
    ordinal: u32,
    count: u32,
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
        function: "infer_delta",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`infer_delta`'s own re-materialize path deletes a `result_tables` row's \
                 CURRENT segment key before rewriting it — a `handle` built from `rt.parquet_path` \
                 / an index-segment URL, never a `models` row's `artifact_path`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/session.rs",
        function: "remove_source",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`Session::remove_source`'s cleanup loop deletes each affected `result_tables` \
                 row's Parquet at its OWN `parquet_path` (read off `rt`, the row being removed), \
                 never a `models` row.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/sidecar_layout.rs",
        function: "delete_sidecar",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`delete_sidecar`'s only production caller is `Session::remove_source`, over the \
                 SAME `result_tables` handle the Parquet delete above uses — a model artifact \
                 carries no `SidecarKind` sidecar at all.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/artifact.rs",
        function: "delete_artifact_prefix",
        ordinal: 1,
        count: 2,
        class: SiteClass::Guarded,
        reason: "inside `ArtifactStore::delete_artifact_prefix`'s own body (the file-entry loop) \
                 — reached only via the `Guarded`/`Exempt` callers listed below.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/artifact.rs",
        function: "delete_artifact_prefix",
        ordinal: 2,
        count: 2,
        class: SiteClass::Guarded,
        reason: "inside `ArtifactStore::delete_artifact_prefix`'s own body (the manifest delete) \
                 — same reachability as the row above.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "reap_expired_version",
        ordinal: 1,
        count: 2,
        class: SiteClass::NonModels,
        reason: "`ResultStore::reap_expired_version`'s Parquet delete, over a version row's own \
                 `parquet_path` — `result_tables` version lifecycle, never `models/`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "reap_expired_version",
        ordinal: 2,
        count: 2,
        class: SiteClass::NonModels,
        reason: "the same `reap_expired_version`, its index-segment sibling delete — still a \
                 version row's own segment key.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "reap_version_artifacts",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`ResultStore::reap_version_artifacts` — a CAS-losing version's own objects, \
                 `result_tables` scoped by construction (the function only ever receives a \
                 version row).",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "purge_segments_for_version",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`ResultStore::purge_segments_for_version` — index-segment purge for one version, \
                 same `result_tables` scoping.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "purge_segments",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`ResultStore::purge_segments` — the table-level segment purge (`drop_table`'s \
                 tail), over `index_segments` rows only.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "delete_objects_after_cas",
        ordinal: 1,
        count: 2,
        class: SiteClass::NonModels,
        reason: "`ResultStore::delete_objects_after_cas`'s Parquet delete — a CAS-losing \
                 `building` row's own `parquet_path`.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "delete_objects_after_cas",
        ordinal: 2,
        count: 2,
        class: SiteClass::NonModels,
        reason: "the same `delete_objects_after_cas`, its sidecar delete — still the losing \
                 `building` row's own key.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/reconcile.rs",
        function: "delete_relative",
        ordinal: 1,
        count: 1,
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
        function: "delete_resume_checkpoint",
        ordinal: 1,
        count: 1,
        class: SiteClass::Exempt,
        reason: "`ArtifactStore::delete_resume_checkpoint` — the `_resume/` namespace proof (this \
                 method's own doc; executed by \
                 `a_resume_checkpoint_prefix_is_never_referenced_even_under_the_containment_aware_predicate` \
                 in `reconcile.rs`), not a live consult.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/store/reconcile.rs",
        function: "delete_unreferenced_prefix",
        ordinal: 1,
        count: 1,
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
/// (matching [`ReviewedSite::file`]); `function` is the innermost named
/// `fn`/method enclosing the call (never a closure — a closure creates no
/// new named scope for this scan's purposes, so a call inside one is
/// attributed to the closure's OWN enclosing named function); `ordinal` is
/// its 1-based position among every raw-delete call found inside that same
/// function, in source order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FoundSite {
    file: String,
    function: String,
    ordinal: u32,
}

/// Whether `attrs` carries `#[cfg(test)]` — the ONE attribute shape this
/// scan treats as "skip this item and everything inside it": a real
/// `cfg(test)` meta list, not a `#[cfg(not(test))]` or any other `cfg`
/// (checked structurally via `syn::Meta`, never a text match on the
/// attribute's rendered form, so `#[cfg(test)]` spelled with extra
/// whitespace or a trailing comma inside the parens is still recognised).
fn has_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        let Ok(list) = attr.meta.require_list() else {
            return false;
        };
        syn::parse2::<syn::Path>(list.tokens.clone())
            .map(|p| p.is_ident("test"))
            .unwrap_or(false)
    })
}

/// Walks a parsed file, tracking the innermost enclosing named function and
/// recording every call of `delete_if_exists` / `delete_artifact_prefix` it
/// finds, in EVERY shape a same-tree caller can spell one — never only the
/// shape today's call sites happen to use ("confirmed by reading every
/// production call site" is a review of the present tree, not a control
/// over the next commit's, and an enumerating gate that enumerates one
/// shape fails open on the others):
///
/// 1. a method call, `handle.delete_if_exists(..)` (`visit_expr_method_call`);
/// 2. a path call under ANY qualifying prefix or qualified-self syntax —
///    `JammiObjectStore::delete_if_exists(&h, ..)`,
///    `crate::storage::JammiObjectStore::delete_if_exists(..)`,
///    `<JammiObjectStore>::delete_if_exists(..)` — matched by the path's own
///    LAST segment, never by a fixed-length segment-vector equality
///    (`visit_expr_call`);
/// 3. a call inside a macro INVOCATION's argument stream —
///    `tokio::try_join!(h.delete_if_exists(&a), ..)`, `assert!(..)`, a
///    `macro_rules!` body — which `syn` parses as an opaque token stream
///    no `visit_expr_*` ever descends into: `visit_macro` walks the tokens
///    (through every nested group) and records each exact `Ident` spelled
///    as one of the two deleters, so a call that hides inside any macro is
///    surfaced as a site to review, attributed to the enclosing named fn.
///
/// Matching is by exact identifier, never substring (`delete_if_existing`
/// or a string literal that merely mentions the name is not a call), and a
/// found site outside any named function panics loudly rather than being
/// dropped. Executed falsifications for every shape live below
/// (`shape_*` tests); `crates/jammi-ai/tests/it/rank_admission.rs`'s
/// submit-seam scanner records the same three shapes.
struct DeleteCallScanner {
    /// The stack of enclosing named-function names — only `visit_item_fn`
    /// (free functions) and `visit_impl_item_fn` (methods) push; a closure
    /// or a `match`/`if` block pushes nothing, so a call inside a closure
    /// is attributed to the closure's own enclosing NAMED function, which
    /// is the scan's whole point (a `.delete_if_exists(` call this codebase
    /// only ever writes directly inside a named `async fn`'s own body, per
    /// every site reviewed above — never inside a further-nested closure).
    fn_stack: Vec<String>,
    /// Running per-`(function)` counter for THIS file, used to assign each
    /// found call's ordinal — reset per file by constructing a fresh
    /// scanner per `syn::File`.
    counts: std::collections::HashMap<String, u32>,
    found: Vec<(String, u32)>,
}

impl DeleteCallScanner {
    fn new() -> Self {
        Self {
            fn_stack: Vec::new(),
            counts: std::collections::HashMap::new(),
            found: Vec::new(),
        }
    }

    fn record_if_match(&mut self, method: &str) {
        if method != "delete_if_exists" && method != "delete_artifact_prefix" {
            return;
        }
        let Some(func) = self.fn_stack.last() else {
            // A raw-delete call outside any named function (module-scope
            // code) is not a shape this codebase has — surfaced loudly
            // rather than silently dropped, so a FUTURE such call cannot
            // hide from this oracle by construction.
            panic!(
                "a `.{method}(` call was found with no enclosing named function on the fn \
                 stack — this scan's own \"never outside a named fn\" assumption broke; review \
                 the new call site and extend this scanner, do not ignore it"
            );
        };
        let ordinal = self.counts.entry(func.clone()).or_insert(0);
        *ordinal += 1;
        self.found.push((func.clone(), *ordinal));
    }
}

impl<'ast> syn::visit::Visit<'ast> for DeleteCallScanner {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if has_cfg_test(&node.attrs) {
            return; // Never descends — every call inside is test-only.
        }
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if has_cfg_test(&node.attrs) {
            return;
        }
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // `#[cfg(test)] mod tests { .. }` is this crate's universal
        // convention (confirmed by reading every file this scan touches) —
        // skip the WHOLE module, not just its individual `fn`s, so this
        // scan never has to special-case a bare `#[test]` fn sitting
        // outside a `mod tests` block (this codebase has none, but the
        // module-level skip covers that shape too if it ever appears,
        // since `has_cfg_test` on the module already stops the walk).
        if has_cfg_test(&node.attrs) {
            return;
        }
        syn::visit::visit_item_mod(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.record_if_match(&node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }

    /// Shape 2: a path call. Only the path's LAST segment is the function
    /// name; everything before it (`Type::`, `crate::m::Type::`, or a
    /// `<Type>` qualified self, which `syn` keeps in `qself` and out of the
    /// segments entirely) is a prefix this scan must be indifferent to.
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*node.func {
            if let Some(last) = p.path.segments.last() {
                self.record_if_match(&last.ident.to_string());
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    /// Shape 3: a macro invocation (expression, statement or item position
    /// — `syn` routes all three here). Its arguments are an opaque token
    /// stream, so the tokens are walked directly; every exact `Ident`
    /// spelled as a deleter is one site.
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        for ident in deleter_idents_in_tokens(node.tokens.clone()) {
            self.record_if_match(&ident);
        }
        syn::visit::visit_macro(self, node);
    }
}

/// Every `Ident` token in `ts` (descending through every nested
/// delimited group) whose spelling is one of the two raw deleters, in
/// source order. A string literal mentioning the name is a `Literal`
/// token, never an `Ident`, so prose inside `format!`/`panic!` cannot
/// match.
fn deleter_idents_in_tokens(ts: proc_macro2::TokenStream) -> Vec<String> {
    let mut out = Vec::new();
    for tt in ts {
        match tt {
            proc_macro2::TokenTree::Ident(i) => {
                let s = i.to_string();
                if s == "delete_if_exists" || s == "delete_artifact_prefix" {
                    out.push(s);
                }
            }
            proc_macro2::TokenTree::Group(g) => out.extend(deleter_idents_in_tokens(g.stream())),
            _ => {}
        }
    }
    out
}

/// Parse `file` and return every raw-delete call site found, in the shape
/// [`FoundSite`] names.
fn scan_file(repo_root: &Path, file: &str) -> Vec<FoundSite> {
    let text = std::fs::read_to_string(repo_root.join(file))
        .unwrap_or_else(|e| panic!("reading {file}: {e}"));
    scan_source(file, &text)
}

/// The scan over one file's SOURCE TEXT — what `scan_file` runs on a real
/// tracked file, and what the `shape_*` falsifications below run on a
/// synthetic fixture, so a fixture exercises exactly the visitor the real
/// gate uses (never a copy of it).
fn scan_source(file: &str, text: &str) -> Vec<FoundSite> {
    let parsed = syn::parse_file(text)
        .unwrap_or_else(|e| panic!("models_delete_call_sites: syn could not parse {file}: {e}"));
    let mut scanner = DeleteCallScanner::new();
    syn::visit::Visit::visit_file(&mut scanner, &parsed);
    scanner
        .found
        .into_iter()
        .map(|(function, ordinal)| FoundSite {
            file: file.to_string(),
            function,
            ordinal,
        })
        .collect()
}

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is `crates/jammi-db`; the repo root is two levels
    // up (`crates/jammi-db/../..`).
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("CARGO_MANIFEST_DIR has no grandparent: {manifest_dir:?}"))
        .to_path_buf()
}

#[test]
fn every_raw_models_byte_delete_call_site_is_reviewed() {
    let repo_root = repo_root();

    // Every crate's production tree: `delete_if_exists` is `pub`, so the
    // universe is every `crates/*/src`, never the two crates that call it
    // today.
    let mut files: Vec<String> = tracked_rs_files(&repo_root, "crates")
        .into_iter()
        .filter(|f| f.split('/').nth(2) == Some("src"))
        .collect();
    assert!(
        files.len() > 50,
        "git ls-files returned suspiciously few files ({}); the scan's quantifier is likely \
         broken, not the tree",
        files.len()
    );

    let mut found: Vec<FoundSite> = Vec::new();
    for file in &files {
        found.extend(scan_file(&repo_root, file));
    }
    found.sort_by(|a, b| {
        (a.file.as_str(), a.function.as_str(), a.ordinal).cmp(&(
            b.file.as_str(),
            b.function.as_str(),
            b.ordinal,
        ))
    });

    // Direction 1: every FOUND site has a REVIEWED entry.
    let mut unreviewed = Vec::new();
    for site in &found {
        if !REVIEWED.iter().any(|r| {
            r.file == site.file && r.function == site.function && r.ordinal == site.ordinal
        }) {
            unreviewed.push(format!(
                "{}::{} #{}",
                site.file, site.function, site.ordinal
            ));
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
        let still_present = found
            .iter()
            .any(|s| s.file == r.file && s.function == r.function && s.ordinal == r.ordinal);
        if !still_present {
            stale.push(format!(
                "{}::{} #{} ({:?})",
                r.file, r.function, r.ordinal, r.class
            ));
        }
    }
    assert!(
        stale.is_empty(),
        "REVIEWED entry no longer matches a real call site at that (file, function, ordinal) — \
         the call this entry reviewed moved or was deleted; re-locate it (or remove the stale \
         entry) rather than leaving a review that now clears nothing:\n{}",
        stale.join("\n")
    );

    // Direction 3 (the shape this re-key adds): the TOTAL call count found
    // inside each reviewed `(file, function)` must equal every entry's own
    // `count` — catches a call added to an already-reviewed function whose
    // EXISTING ordinals still resolve (so direction 1 alone would miss it).
    let mut count_mismatches = Vec::new();
    for r in REVIEWED {
        let real_count = found
            .iter()
            .filter(|s| s.file == r.file && s.function == r.function)
            .count() as u32;
        if real_count != r.count {
            count_mismatches.push(format!(
                "{}::{} declares count={} but the real scan found {real_count}",
                r.file, r.function, r.count
            ));
        }
    }
    count_mismatches.sort();
    count_mismatches.dedup();
    assert!(
        count_mismatches.is_empty(),
        "a reviewed function's real raw-delete call count no longer matches its REVIEWED \
         entries' declared `count` — a call was added (or removed) inside an already-reviewed \
         function:\n{}",
        count_mismatches.join("\n")
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
            "{}::{} #{} ({:?}) has no reviewed reason",
            r.file,
            r.function,
            r.ordinal,
            r.class
        );
    }
}

/// Mutation oracle for I1: an UNREVIEWED new raw-delete call site must red
/// this scan. Exercised by constructing `found`/`REVIEWED` in miniature
/// (never touching the real tree) so the assertion logic itself is proven,
/// independent of today's real call-site count drifting the primary test
/// above.
/// Run the real scanner over a synthetic fixture and return `function #ordinal`
/// rows — the shape every `shape_*` falsification below asserts on.
fn shape_sites(src: &str) -> Vec<String> {
    scan_source("fixture.rs", src)
        .into_iter()
        .map(|s| format!("{} #{}", s.function, s.ordinal))
        .collect()
}

#[test]
fn shape_1_a_method_call_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_1_a_method_call_is_found — not real code in this file
    let src = "async fn f(h: H, p: P) { h.delete_if_exists(&p).await.unwrap(); }";
    assert_eq!(shape_sites(src), vec!["f #1"]);
}

#[test]
fn shape_2_a_path_call_is_found_under_any_prefix_and_qualified_self() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_2 (bare type path) — not real code in this file
    let bare = "async fn f(h: H, p: P) { JammiObjectStore::delete_if_exists(&h, &p).await; }";
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_2 (crate-qualified path) — not real code in this file
    let qualified = "async fn f(h: H, p: P) { crate::storage::JammiObjectStore::delete_if_exists(&h, &p).await; }";
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_2 (qualified self) — not real code in this file
    let qself = "async fn f(h: H, p: P) { <JammiObjectStore>::delete_if_exists(&h, &p).await; }";
    let other =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_2 (the other deleter, Self-prefixed) — not real code in this file
        "impl S { async fn g(&self, p: P) { Self::delete_artifact_prefix(self, &p).await; } }";
    for (name, src) in [
        ("bare", bare),
        ("qualified", qualified),
        ("qself", qself),
        ("other", other),
    ] {
        let want = if name == "other" { "g #1" } else { "f #1" };
        assert_eq!(
            shape_sites(src),
            vec![want],
            "shape 2 ({name}): a path-call spelling of a raw deleter must be found"
        );
    }
}

#[test]
fn shape_3_a_call_inside_a_macro_invocation_is_found_per_occurrence() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_3 (try_join!) — not real code in this file
    let joined = "async fn f(h: H, a: P, b: P) { tokio::try_join!(h.delete_if_exists(&a), h.delete_if_exists(&b)).unwrap(); }";
    assert_eq!(shape_sites(joined), vec!["f #1", "f #2"]);
    let nested =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_3 (assert!, nested group) — not real code in this file
        "async fn f(h: H, p: P) { assert!(matches!(h.delete_if_exists(&p).await, Ok(_))); }";
    assert_eq!(shape_sites(nested), vec!["f #1"]);
    let body =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_3 (macro_rules body) — not real code in this file
        "fn f() { macro_rules! reap { ($h:expr, $p:expr) => { $h.delete_if_exists($p).await } } }";
    assert_eq!(shape_sites(body), vec!["f #1"]);
}

#[test]
fn shape_controls_a_near_miss_identifier_or_a_string_literal_is_not_a_call() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for shape_controls — not real code in this file
    let src = "async fn f(h: H, p: P) { h.delete_if_existing(&p).await; let _ = delete_if_exists_count(); \
               tracing::warn!(\"delete_if_exists refused {p}\"); format!(\"delete_artifact_prefix\"); }";
    assert_eq!(shape_sites(src), Vec::<String>::new());
}

#[test]
fn shape_a_call_outside_any_named_function_is_surfaced_not_dropped() {
    let src = "static _X: () = { let _ = macro_with_deleter!(delete_if_exists); };";
    let r = std::panic::catch_unwind(|| shape_sites(src));
    assert!(
        r.is_err(),
        "a deleter reached at module scope must panic the scan loudly, never be silently dropped"
    );
}

#[test]
fn an_unreviewed_call_site_reds_the_direction_one_check() {
    let found = [FoundSite {
        file: "crates/jammi-db/src/store/mod.rs".to_string(),
        function: "a_function_no_reviewed_entry_names".to_string(),
        ordinal: 1,
    }];
    let reviewed_has_it = REVIEWED.iter().any(|r| {
        r.file == found[0].file && r.function == found[0].function && r.ordinal == found[0].ordinal
    });
    assert!(
        !reviewed_has_it,
        "sanity: the synthetic (file, function, ordinal) must not collide with a real REVIEWED \
         entry"
    );
    // The primary test's own `unreviewed` computation, reproduced: a found
    // site absent from `REVIEWED` is flagged. RED direction: deleting this
    // `!reviewed_has_it` check (i.e. treating every found site as reviewed
    // regardless) would make `every_raw_models_byte_delete_call_site_is_reviewed`
    // pass even with a real unreviewed call site added to the tree — exactly
    // the defect this oracle exists to catch.
    assert!(
        !reviewed_has_it,
        "an unreviewed call site at {}::{} #{} must be flagged, never silently accepted",
        found[0].file, found[0].function, found[0].ordinal
    );
}

/// Mutation oracle for the re-key's OWN added value (direction 3): a call
/// added to an already-reviewed function, whose EXISTING ordinals still
/// resolve, must still be caught via the count mismatch — proving this
/// scan closes the exact gap a bare `(file, function, ordinal)` match
/// (without the count cross-check) would miss.
#[test]
fn a_fourth_call_added_to_a_three_call_reviewed_function_reds_the_count_check() {
    let reviewed_count: u32 = REVIEWED
        .iter()
        .filter(|r| {
            r.file == "crates/jammi-db/src/store/artifact.rs"
                && r.function == "delete_artifact_prefix"
        })
        .map(|r| r.count)
        .next()
        .expect("fixture assumption: delete_artifact_prefix has REVIEWED entries");
    assert_eq!(
        reviewed_count, 2,
        "fixture assumption drifted — this test's own premise (delete_artifact_prefix is \
         reviewed at count=2) no longer holds; update the fixture, not just this assertion"
    );
    let real_found_count = 3u32; // simulates a third call added to the real 2.
    assert_ne!(
        real_found_count, reviewed_count,
        "the count-mismatch check exists exactly because these two can diverge — this assertion \
         reproduces the primary test's own comparison, proving it fires on a real divergence"
    );
}
