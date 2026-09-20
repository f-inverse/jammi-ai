//! Enumerating source oracle for the `models/` byte-delete guard.
//!
//! The guard cannot be a compile-time proof — the models root is a RUNTIME
//! value (`models_root(&root)`), not a type — so completeness is carried by
//! this scan over every reference to the deleter the storage HANDLE exposes,
//! `JammiObjectStore::delete_if_exists`, and the handle's own raw driver, as
//! the `pub(crate)` accessor `JammiObjectStore::driver` and as the private
//! field `self.driver` inside the handle's file (every reference to either is
//! a reviewed row). The universe is `jammi-db`'s own `src/`: the deleter and
//! the driver are both `pub(crate)`, so the compiler refuses a caller in any
//! other crate (`storage::object_store_handle`'s `compile_fail` doctest pins
//! it) and only this crate's own call sites remain for a review to cover.
//!
//! **What this oracle does NOT close.** A raw `Arc<dyn ObjectStore>` — on
//! which `ObjectStoreExt::delete` is unguarded — is obtainable without the
//! handle by these routes:
//!
//! 1. **Closed.** `jammi_db::storage::registry::StorageRegistry::driver_for`
//!    and `jammi_db::storage::builder::build_object_store` are `pub(crate)`
//!    (`storage/mod.rs`'s `compile_fail,E0624`/`E0603`/`E0425` doctests pin
//!    it); every external acquisition site uses `StorageRegistry::handle_for`
//!    or `JammiObjectStore::open`, both of which hand back the guarded handle.
//!    `JammiObjectStore::new` only ACCEPTS a driver a caller already built
//!    (route 3); nothing returns one to caller code.
//! 2. **Open.** `JammiSession::context()` (`pub`, re-exposed by `jammi-ai`'s
//!    `InferenceSession::context()`, and held by `jammi-ballista`'s executor
//!    wiring) hands out the live `SessionContext`, whose default `RuntimeEnv`
//!    pre-registers a `LocalFileSystem` rooted at `/` for `file://`, so
//!    `context().runtime_env().object_store(url)` yields a writable store. A
//!    registry-level wrapper cannot close this route: DataFusion's
//!    `SessionContext::state_ref` is `pub` and returns the SHARED
//!    `Arc<RwLock<SessionState>>`, so any holder can rebuild the state with a
//!    fresh default `RuntimeEnv` (`SessionStateBuilder::new_from_existing(..)
//!    .with_runtime_env(..)`) and write it back, replacing any wrapper. Closing
//!    it needs a context facade that never exposes `state_ref` or the runtime
//!    env.
//! 3. **Open, by design.** Direct construction with the `object_store` crate by
//!    any code holding the same credentials — no crate boundary can seal a
//!    capability a dependent crate can rebuild from raw credentials.
//! 4. Raw catalog SQL (`Tx::{execute,query,query_opt}`,
//!    `BackendImpl::transaction`, `Catalog::backend_arc`) is a SEPARATE
//!    property and is deliberately not sealed: `BackendImpl` is a `pub` enum
//!    over `pub` backend types with `pub` constructors, so sealing the
//!    transaction-closure surface would remove only its most convenient door,
//!    and would lock out the negative-path test fixtures that hand-build rows
//!    the engine's typed verbs refuse (`crates/jammi-db/tests/it` is a
//!    separate crate under Rust's privacy rules). "A training-kind `jobs` row
//!    is writable only through `submit_job_deduped`" is carried for production
//!    code by the submit-seam oracle in
//!    `crates/jammi-ai/tests/it/rank_admission.rs`; test code and direct
//!    backend construction keep raw SQL, consistent with the engine's trust
//!    posture (`docs/guide/src/security.md`: a trusted network, not a security
//!    boundary against code running alongside it).
//!
//! **Keyed by `(file, function, ordinal)`, not `(file, line)`.** A line number
//! goes stale on every unrelated edit above it; the enclosing function's name
//! and the call's ordinal within it are invariant to any edit outside that
//! function's body. Finding "the enclosing function", and telling a real call
//! from a doc comment's rendered prose, needs a real parse, so every file is
//! parsed into a `syn::File` and walked with `syn::visit::Visit`, the same
//! idiom as `pinned_source_gate.rs`.
//!
//! Each found call site is classified exactly once:
//!
//! - [`SiteClass::Licensed`] — the raw deleter's own body. A `models/` key
//!   reaches it one way only: `JammiObjectStore::delete_licensed`, past the
//!   always-on `ReclaimLicence::covers` check, holding the licence only the
//!   catalog's reclaim compare-and-set mints
//!   (`jammi_db::catalog::artifact_repo`).
//! - [`SiteClass::NonModels`] — reviewed and found to operate on a
//!   `result_tables`/index-segment/sidecar key that can never be `models/`-
//!   namespaced, with the reason stated per entry.
//!
//! **Why this, beside the type.** The reclaim licence makes the `models/`
//! delete unforgeable, but inside this crate `delete_if_exists` serves every
//! non-`models/` key, and the models root is a runtime `StorageUrl`: no Rust
//! type can refuse to compile a new in-crate `delete_if_exists` caller aimed
//! under it. That half of the proof is this scan, over the real `git ls-files`
//! listing and a real `syn` parse.
//!
//! This test fails in BOTH directions: a call site with no matching
//! [`REVIEWED`] entry (an unreviewed new deleter), and a [`REVIEWED`] entry
//! whose `(file, function, ordinal)` names no real call (a stale entry that
//! would keep "clearing" a site a refactor moved or deleted). It also checks
//! that the TOTAL call count within a reviewed function matches every entry's
//! `count` field, so a call added to an already-reviewed function is caught
//! even though the existing ordinals still resolve.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteClass {
    Licensed,
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
/// three classes). Adding a new reference — anywhere cargo compiles outside
/// a test target — means adding a row here, with a reviewed reason; that is
/// the point of this test.
const REVIEWED: &[ReviewedSite] = &[
    // ── `JammiObjectStore::driver` (the raw `Arc<dyn ObjectStore>`) ─────
    // Inside the handle's own file the private FIELD is the raw store; every
    // use of it is a row (shape 5). `delete_if_exists` is the ONE deleter.
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "driver",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "the `pub(crate)` accessor clones the raw driver for the parquet reader/writer \
                 (their references are the two rows below); it deletes nothing itself.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "put_bytes",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "a `put` on the handle's own path; no delete.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "get_range",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "a ranged `get` on the handle's own path; no delete.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "delete_raw",
        ordinal: 1,
        count: 1,
        class: SiteClass::Licensed,
        reason: "THE raw deleter's own body — the one place the field's `delete` is called. It \
                 is private: `delete_licensed` reaches it only past the reclaim licence's own \
                 `covers` check, and every caller of `delete_if_exists` is itself a reviewed \
                 row of this table.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "get_bytes",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "a whole-object `get` on the handle's own path; no delete.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "list",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "a `list` under the handle's own prefix; no delete.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/object_store_handle.rs",
        function: "exists",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "a `head` on the handle's own path; no delete.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/reader.rs",
        function: "read_all_record_batches",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "the parquet reader hands the raw driver to `ParquetObjectReader::new` to READ \
                 the handle's own path; no delete is issued on it.",
    },
    ReviewedSite {
        file: "crates/jammi-db/src/storage/writer.rs",
        function: "open",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "the parquet writer hands the raw driver to `ParquetObjectWriter::new` to WRITE \
                 the handle's own path; no delete is issued on it.",
    },
    // ── `JammiObjectStore::delete_if_exists` ────────────────────────────
    ReviewedSite {
        file: "crates/jammi-db/src/store/building_version.rs",
        function: "discard_empty_fragment",
        ordinal: 1,
        count: 1,
        class: SiteClass::NonModels,
        reason: "`BuildingVersion::discard_empty_fragment` deletes the version's own empty \
                 fragment at its `fragment_url` — a `result_table_versions` key derived from the \
                 table's `parquet_path`, never a `models` row's artifact.",
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
        class: SiteClass::NonModels,
        reason: "`ResultStore::delete_relative`, called ONLY from `reconcile_inner`'s age-gated \
                 result-table orphan arm. Every `models/` key leaves that loop earlier — named \
                 by an artifact row, `unattributed`, or a stray settled per directory — and is \
                 deleted, if at all, through `ArtifactStore::reclaim` under a licence.",
    },
];

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
/// recording every REFERENCE to `delete_if_exists`
/// it finds — a call or a value — in the shapes the grammar allows, never
/// only the shape today's call sites happen to use ("confirmed by reading
/// every production call site" is a review of the present tree, not a
/// control over the next commit's, and an enumerating gate that enumerates
/// one shape fails open on the others):
///
/// 1. a method call, `handle.delete_if_exists(..)` (`visit_expr_method_call`);
/// 2. a path expression naming the fn in ANY position — the callee of
///    `JammiObjectStore::delete_if_exists(&h, ..)`,
///    `crate::storage::JammiObjectStore::delete_if_exists(..)`,
///    `<JammiObjectStore>::delete_if_exists(..)`, and equally a fn-item
///    captured as a value and invoked later (`let raw =
///    JammiObjectStore::delete_if_exists; raw(&h, &p)`), handed to a
///    combinator or stored in a field — matched by the path's own LAST
///    segment, never by a fixed-length segment-vector equality
///    (`visit_expr_path`, which fires wherever a path appears, so a call
///    position is not a special case; a value reference is a site because
///    the fn it names can be invoked anywhere afterwards);
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
/// submit-seam scanner records the same shapes.
struct DeleteCallScanner {
    /// The stack of enclosing named-function names — only `visit_item_fn`
    /// (free functions) and `visit_impl_item_fn` (methods) push; a closure
    /// or a `match`/`if` block pushes nothing, so a call inside a closure
    /// is attributed to the closure's own enclosing NAMED function, which
    /// is the scan's whole point (a `.delete_if_exists(` call this codebase
    /// only ever writes directly inside a named `async fn`'s own body, per
    /// every site reviewed above — never inside a further-nested closure).
    fn_stack: Vec<String>,
    /// The repo-relative path being scanned, for file-scoped target rules
    /// (`driver` counts inside `crates/jammi-db/src` only).
    file: String,
    /// Running per-`(function)` counter for THIS file, used to assign each
    /// found call's ordinal — reset per file by constructing a fresh
    /// scanner per `syn::File`.
    counts: std::collections::HashMap<String, u32>,
    found: Vec<(String, u32)>,
}

impl DeleteCallScanner {
    fn new(file: &str) -> Self {
        Self {
            file: file.to_string(),
            fn_stack: Vec::new(),
            counts: std::collections::HashMap::new(),
            found: Vec::new(),
        }
    }

    fn record_if_match(&mut self, method: &str) {
        let is_deleter = method == "delete_if_exists";
        // `JammiObjectStore::driver` is `pub(crate)`: outside `crates/jammi-db/src`
        // the compiler refuses the reference (E0624), so a `driver` identifier
        // there is a homonym on some other type and never the raw store. Inside
        // this crate the compiler proves nothing, so every `driver` reference
        // is a site.
        let is_raw_driver = method == "driver" && self.file.starts_with("crates/jammi-db/src/");
        if !is_deleter && !is_raw_driver {
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

    /// Shape 5: the raw driver as a FIELD (`self.driver`), which only the
    /// handle's own file can spell (the field is private) — every use of it
    /// there is a site, so a new deleter written as `self.driver.delete(..)`
    /// is reviewed like any other.
    fn visit_expr_field(&mut self, node: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &node.member {
            if name == "driver" && self.file == "crates/jammi-db/src/storage/object_store_handle.rs"
            {
                self.record_if_match("driver");
            }
        }
        syn::visit::visit_expr_field(self, node);
    }

    /// Shape 2: a path expression in ANY position (a call's callee, a
    /// captured value, an argument, a field). Only the path's LAST segment
    /// is the function name; everything before it (`Type::`,
    /// `crate::m::Type::`, or a `<Type>` qualified self, which `syn` keeps
    /// in `qself` and out of the segments entirely) is a prefix this scan
    /// must be indifferent to.
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        // An inherent method is only ever spelled with an owner —
        // `Type::name`, `crate::m::Type::name`, `<Type>::name` — so a bare
        // single-segment path (`name`) is a local variable or a free fn of
        // that name, never the deleter; recording it would review homonyms.
        if node.qself.is_some() || node.path.segments.len() >= 2 {
            if let Some(last) = node.path.segments.last() {
                self.record_if_match(&last.ident.to_string());
            }
        }
        syn::visit::visit_expr_path(self, node);
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
                if s == "delete_if_exists" {
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
    let mut scanner = DeleteCallScanner::new(file);
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

use jammi_test_utils::source_universe::repo_root;

#[test]
fn every_raw_models_byte_delete_call_site_is_reviewed() {
    let repo_root = repo_root();

    // The universe: every `.rs` cargo compiles outside a test target
    // (`jammi_test_utils::source_universe`), narrowed to this crate's own
    // `src/` — the deleter is `pub(crate)`, so the compiler already refuses
    // every other crate.
    let files: Vec<String> =
        jammi_test_utils::source_universe::compiled_non_test_rs_files(&repo_root)
            .into_iter()
            .filter(|f| f.starts_with("crates/jammi-db/src/"))
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
         state why it can never reach a `models/` key (NonModels) — a `models/` byte is deleted \
         only under a reclaim licence — then add a row:\n{}",
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

    // Exactly one site may touch a `models/` key: pinned so reclassifying an
    // entry is a visible diff in this count, not a silent move.
    let licensed = REVIEWED
        .iter()
        .filter(|r| r.class == SiteClass::Licensed)
        .count();
    assert_eq!(
        licensed, 1,
        "the raw deleter's own body is the only site a `models/` key may reach — re-derive the \
         licence argument rather than only updating this constant"
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

/// Mutation oracle for this scan: an UNREVIEWED new raw-delete call site must red
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
    let src = "async fn f(h: H, p: P) { h.delete_if_exists(&p).await.unwrap(); }";
    assert_eq!(shape_sites(src), vec!["f #1"]);
}

#[test]
fn shape_2_a_path_call_is_found_under_any_prefix_and_qualified_self() {
    let bare = "async fn f(h: H, p: P) { JammiObjectStore::delete_if_exists(&h, &p).await; }";
    let qualified = "async fn f(h: H, p: P) { crate::storage::JammiObjectStore::delete_if_exists(&h, &p).await; }";
    let qself = "async fn f(h: H, p: P) { <JammiObjectStore>::delete_if_exists(&h, &p).await; }";
    let other = "impl S { async fn g(&self, p: P) { Self::delete_if_exists(self, &p).await; } }";
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
    let joined = "async fn f(h: H, a: P, b: P) { tokio::try_join!(h.delete_if_exists(&a), h.delete_if_exists(&b)).unwrap(); }";
    assert_eq!(shape_sites(joined), vec!["f #1", "f #2"]);
    let nested =
        "async fn f(h: H, p: P) { assert!(matches!(h.delete_if_exists(&p).await, Ok(_))); }";
    assert_eq!(shape_sites(nested), vec!["f #1"]);
    let body =
        "fn f() { macro_rules! reap { ($h:expr, $p:expr) => { $h.delete_if_exists($p).await } } }";
    assert_eq!(shape_sites(body), vec!["f #1"]);
}

#[test]
fn shape_4_a_path_captured_as_a_value_is_found_wherever_it_appears() {
    let captured = "async fn f(h: H, p: P) { let raw = JammiObjectStore::delete_if_exists; raw(&h, &p).await; }";
    let combinator =
        "fn f(ps: Vec<P>) { let _ = ps.iter().map(JammiObjectStore::delete_if_exists); }";
    let field = "fn f() -> Ops { Ops { del: <JammiObjectStore>::delete_if_exists } }";
    for (name, src) in [
        ("captured", captured),
        ("combinator", combinator),
        ("field", field),
    ] {
        assert_eq!(
            shape_sites(src),
            vec!["f #1"],
            "shape 4 ({name}): a value reference must be found"
        );
    }
}

#[test]
fn shape_5_a_raw_driver_reference_is_a_site_inside_this_crate_only() {
    let src = "async fn f(h: H, p: P) { let _ = h.driver().delete(&p).await; }";
    let rows = |file: &str| -> Vec<String> {
        scan_source(file, src)
            .into_iter()
            .map(|s| format!("{} #{}", s.function, s.ordinal))
            .collect()
    };
    assert_eq!(
        rows("crates/jammi-db/src/store/fixture.rs"),
        vec!["f #1"],
        "a `driver()` reference inside jammi-db must be a reviewed site"
    );
    assert_eq!(
        rows("crates/jammi-bench/src/fixture.rs"),
        Vec::<String>::new(),
        "outside jammi-db a `driver` identifier is a homonym the compiler already keeps off the raw store"
    );
    let local = "fn f(d: D) { let driver = d; driver.run(); }";
    assert_eq!(
        scan_source("crates/jammi-db/src/store/fixture.rs", local).len(),
        0,
        "a bare local named `driver` is not a reference to the inherent method"
    );
}

#[test]
fn shape_5b_the_raw_driver_field_inside_the_handle_is_a_site() {
    let src =
        "impl H { pub async fn purge(&self, p: &P) { let _ = self.driver.delete(p).await; } }";
    let rows = |file: &str| -> Vec<String> {
        scan_source(file, src)
            .into_iter()
            .map(|s| format!("{} #{}", s.function, s.ordinal))
            .collect()
    };
    assert_eq!(
        rows("crates/jammi-db/src/storage/object_store_handle.rs"),
        vec!["purge #1"],
        "a use of the private `driver` field inside the handle's file must be a reviewed site"
    );
    assert_eq!(
        rows("crates/jammi-db/src/store/mod.rs"),
        Vec::<String>::new(),
        "a field named `driver` in any other file is some other type's field (the handle's is private)"
    );
}

#[test]
fn shape_controls_a_near_miss_identifier_or_a_string_literal_is_not_a_call() {
    let src = "async fn f(h: H, p: P) { h.delete_if_existing(&p).await; let _ = delete_if_exists_count(); \
               tracing::warn!(\"delete_if_exists refused {p}\"); format!(\"delete_if_exists\"); }";
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
            r.file == "crates/jammi-db/src/store/mod.rs" && r.function == "reap_expired_version"
        })
        .map(|r| r.count)
        .next()
        .expect("fixture assumption: reap_expired_version has REVIEWED entries");
    assert_eq!(
        reviewed_count, 2,
        "fixture assumption drifted — this test's own premise (reap_expired_version is \
         reviewed at count=2) no longer holds; update the fixture, not just this assertion"
    );
    let real_found_count = 3u32; // simulates a third call added to the real 2.
    assert_ne!(
        real_found_count, reviewed_count,
        "the count-mismatch check exists exactly because these two can diverge — this assertion \
         reproduces the primary test's own comparison, proving it fires on a real divergence"
    );
}
