//! `symbol-index`: a real `syn` AST index of every Rust item, impl method,
//! enum variant, struct field, and CALL SITE under the given root
//! directories, with file path + line, emitted as JSON.
//!
//! CI-only. Never a product binary, never published. The checks that read
//! Rust source (`check_no_consumer_names.py`'s public-declaration scan, the
//! eager-disable key sweep) resolve against ONE real parse of the tree instead
//! of each carrying its own regex reader: syn's parser cannot be fooled by a
//! brace inside a string literal the way a line-based brace counter can.
//!
//! Usage: `cargo run --release -p symbol-index -- <root-dir>... [--out <path>]`
//! (no `--out` → JSON on stdout). Malformed/unparseable files are SKIPPED
//! with a warning on stderr and counted in the summary line — never a hard
//! abort of the whole index build (a single fixture file mid-edit, or a
//! generated file this tool does not need, must not deny every gate that
//! reads the index).
//!
//! Scope, stated: free items (`fn`/`struct`/`enum`/`trait`/`type`/`const`/
//! `static`/`mod`), every method defined directly inside an `impl Type { }`
//! / `impl Trait for Type { }` block (qualified `Type::method`), every
//! enum variant (`Enum::Variant`), every named struct field
//! (`Struct::field`), and every CALL EXPRESSION (`f(...)`/`recv.f(...)`),
//! by the callee's own bare name (a method call's RECEIVER type is not
//! resolved — that needs real type inference, out of scope for a syntax
//! index; a caller cross-references by NAME, same as this tool's own
//! item index does for a bare-hint citation). Trait-body method
//! SIGNATURES (never given a body in the trait itself) are NOT separately
//! indexed as `Trait::method` — this repo's plan-doc citations name a
//! concrete `impl`'s own method, never a trait interface declaration; a
//! checked scope limit, not an unverified assumption (grepped
//! `docs/plans/**` for a `TraitName::` citation form at the time this
//! tool was written: none).
//!
//! `is_test`/`in_test` (on `fn` items and call sites respectively): a
//! `#[test]`/`#[tokio::test]`/… -attributed fn, or ANY item/call lexically
//! inside a `#[cfg(test)] mod { … }` block, is marked — the same
//! "non-test" boundary issue #557 item 2 needs to exclude test-only call
//! sites from the required-attack-site set.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use syn::spanned::Spanned;
use syn::visit::Visit;

#[derive(Serialize)]
struct IndexedItem {
    path: String,
    kind: &'static str,
    name: String,
    qualified: Option<String>,
    vis: String,
    line: usize,
    line_end: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    is_test: bool,
}

#[derive(Serialize)]
struct CallSite {
    path: String,
    callee: String,
    line: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    in_test: bool,
}

#[derive(Serialize)]
struct Index {
    items: Vec<IndexedItem>,
    calls: Vec<CallSite>,
    files_scanned: usize,
    files_skipped: Vec<String>,
}

/// `#[cfg(test)]` — a real `Meta` inspection (never a text-tokenstream
/// grep for `test` alone, which would also fire on `#[cfg(feature =
/// "test-hooks")]`): the meta must be `List` (has parens), its path must
/// be exactly `cfg`, and one of its comma-separated tokens must be the
/// bare identifier `test` (not merely a substring — `nottest` inside
/// `#[cfg(nottest)]` must not match).
fn is_cfg_test_attr(attr: &syn::Attribute) -> bool {
    if attr.path().segments.last().map(|s| s.ident.to_string()) != Some("cfg".to_string()) {
        return false;
    }
    let syn::Meta::List(list) = &attr.meta else {
        return false;
    };
    list.tokens
        .to_string()
        .split(&[' ', ','][..])
        .any(|tok| tok == "test")
}

/// `#[test]` / `#[tokio::test]` / `#[actix_rt::test]` / … — the attribute's
/// OWN last path segment is the bare identifier `test`, whatever prefix
/// crate it's qualified through.
fn is_test_attr(attr: &syn::Attribute) -> bool {
    attr.path()
        .segments
        .last()
        .map(|s| s.ident == "test")
        .unwrap_or(false)
}

/// `syn::Visibility` stringified to the same vocabulary a plan-doc reader
/// would recognize (`"pub"`, `"pub(crate)"`, `"pub(in a::b)"`, `""` for
/// private) — never via `quote`/`ToTokens` (avoids a dependency this tool
/// does not otherwise need): a `Visibility::Restricted` path is always a
/// short `crate`/`super`/`self`/`in a::b` form, cheap to join by hand.
fn vis_string(vis: &syn::Visibility) -> String {
    match vis {
        syn::Visibility::Public(_) => "pub".to_string(),
        syn::Visibility::Restricted(r) => {
            let path_str = r
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if r.in_token.is_some() {
                format!("pub(in {path_str})")
            } else {
                format!("pub({path_str})")
            }
        }
        syn::Visibility::Inherited => String::new(),
    }
}

/// The short (last-segment, generics stripped) name of a `syn::Type` — an
/// `impl <TypeName>` / `impl Trait for <TypeName>` block's own self type.
/// `None` for a shape a plan-doc citation would never name as `Type::`
/// (a tuple type, a bare `dyn Trait` with no concrete name, etc.).
fn type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => type_name(&r.elem),
        syn::Type::Group(g) => type_name(&g.elem),
        syn::Type::Paren(p) => type_name(&p.elem),
        _ => None,
    }
}

/// Bundled args for `FileVisitor::push` — clippy's `too_many_arguments`
/// lint (>7) is a real signal here, not noise: eight positional
/// same-typed-ish args (three `String`s, two `Span`s) is exactly the shape
/// that silently swaps two arguments at a call site with no compiler
/// error. A named-field struct makes a swap a compile error instead.
struct PushArgs {
    kind: &'static str,
    name: String,
    qualified: Option<String>,
    vis: String,
    name_span: proc_macro2::Span,
    whole_span: proc_macro2::Span,
    is_test: bool,
}

struct FileVisitor<'a> {
    path: &'a str,
    items: Vec<IndexedItem>,
    calls: Vec<CallSite>,
    impl_stack: Vec<String>,
    /// >0 while walking inside a `#[cfg(test)] mod { … }` block.
    test_mod_depth: usize,
    /// >0 while walking inside a `#[test]`-attributed fn's own body.
    test_fn_depth: usize,
}

impl<'a> FileVisitor<'a> {
    fn new(path: &'a str) -> Self {
        FileVisitor {
            path,
            items: Vec::new(),
            calls: Vec::new(),
            impl_stack: Vec::new(),
            test_mod_depth: 0,
            test_fn_depth: 0,
        }
    }

    fn in_test(&self) -> bool {
        self.test_mod_depth > 0 || self.test_fn_depth > 0
    }

    fn push(&mut self, args: PushArgs) {
        self.items.push(IndexedItem {
            path: self.path.to_string(),
            kind: args.kind,
            name: args.name,
            qualified: args.qualified,
            vis: args.vis,
            line: args.name_span.start().line,
            line_end: args.whole_span.end().line,
            is_test: args.is_test,
        });
    }

    fn push_call(&mut self, callee: String, span: proc_macro2::Span) {
        self.calls.push(CallSite {
            path: self.path.to_string(),
            callee,
            line: span.start().line,
            in_test: self.in_test(),
        });
    }
}

impl<'a, 'ast> Visit<'ast> for FileVisitor<'a> {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let name = node.sig.ident.to_string();
        let vis = vis_string(&node.vis);
        let is_test_fn = node.attrs.iter().any(is_test_attr);
        let was_test = self.in_test();
        self.push(PushArgs {
            kind: "fn",
            name,
            qualified: None,
            vis,
            name_span: node.sig.ident.span(),
            whole_span: node.span(),
            is_test: was_test || is_test_fn,
        });
        if is_test_fn {
            self.test_fn_depth += 1;
        }
        syn::visit::visit_item_fn(self, node);
        if is_test_fn {
            self.test_fn_depth -= 1;
        }
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "struct",
            name: name.clone(),
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
        if let syn::Fields::Named(named) = &node.fields {
            for f in &named.named {
                if let Some(ident) = &f.ident {
                    let fvis = vis_string(&f.vis);
                    self.push(PushArgs {
                        kind: "field",
                        name: ident.to_string(),
                        qualified: Some(format!("{name}::{ident}")),
                        vis: fvis,
                        name_span: ident.span(),
                        whole_span: f.span(),
                        is_test: self.in_test(),
                    });
                }
            }
        }
    }

    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "enum",
            name: name.clone(),
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
        for v in &node.variants {
            self.push(PushArgs {
                kind: "variant",
                name: v.ident.to_string(),
                qualified: Some(format!("{name}::{}", v.ident)),
                vis: String::new(),
                name_span: v.ident.span(),
                whole_span: v.span(),
                is_test: self.in_test(),
            });
        }
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "trait",
            name,
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
        // Deliberately NOT recursed into: trait-body method SIGNATURES are
        // out of this tool's stated scope (see module doc).
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "type",
            name,
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
    }

    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "const",
            name,
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        self.push(PushArgs {
            kind: "static",
            name,
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test(),
        });
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let name = node.ident.to_string();
        let vis = vis_string(&node.vis);
        let is_test_mod = node.attrs.iter().any(is_cfg_test_attr);
        self.push(PushArgs {
            kind: "mod",
            name,
            qualified: None,
            vis,
            name_span: node.ident.span(),
            whole_span: node.span(),
            is_test: self.in_test() || is_test_mod,
        });
        if is_test_mod {
            self.test_mod_depth += 1;
        }
        syn::visit::visit_item_mod(self, node);
        if is_test_mod {
            self.test_mod_depth -= 1;
        }
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let target = type_name(&node.self_ty).unwrap_or_else(|| "?".to_string());
        self.impl_stack.push(target);
        syn::visit::visit_item_impl(self, node);
        self.impl_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let name = node.sig.ident.to_string();
        let vis = vis_string(&node.vis);
        let qualified = self.impl_stack.last().map(|t| format!("{t}::{name}"));
        let is_test_fn = node.attrs.iter().any(is_test_attr);
        let was_test = self.in_test();
        self.push(PushArgs {
            kind: "fn",
            name,
            qualified,
            vis,
            name_span: node.sig.ident.span(),
            whole_span: node.span(),
            is_test: was_test || is_test_fn,
        });
        if is_test_fn {
            self.test_fn_depth += 1;
        }
        syn::visit::visit_impl_item_fn(self, node);
        if is_test_fn {
            self.test_fn_depth -= 1;
        }
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = node.func.as_ref() {
            if let Some(seg) = p.path.segments.last() {
                self.push_call(seg.ident.to_string(), seg.ident.span());
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.push_call(node.method.to_string(), node.method.span());
        syn::visit::visit_expr_method_call(self, node);
    }
}

/// Parses `text` (one file's full source, `display_path` its recorded
/// path) and returns `(items, calls)`; `None` on a `syn` parse error —
/// the ONE seam both the real file-walking driver and this tool's own
/// unit tests (below) call, so a test exercises the SAME code the real
/// index build runs, never a second, parallel implementation.
fn index_source(display_path: &str, text: &str) -> Option<(Vec<IndexedItem>, Vec<CallSite>)> {
    let parsed = syn::parse_file(text).ok()?;
    let mut visitor = FileVisitor::new(display_path);
    for item in &parsed.items {
        visitor.visit_item(item);
    }
    Some((visitor.items, visitor.calls))
}

fn find_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| e.file_name() != "target" && e.file_name() != "_generated")
    {
        let Ok(entry) = entry else { continue };
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|e| e.to_str()) == Some("rs")
        {
            out.push(entry.path().to_path_buf());
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut roots: Vec<String> = Vec::new();
    let mut out_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--out" {
            i += 1;
            out_path = args.get(i).cloned();
        } else {
            roots.push(args[i].clone());
        }
        i += 1;
    }
    if roots.is_empty() {
        roots.push("crates".to_string());
    }

    let mut all_paths: Vec<PathBuf> = Vec::new();
    let mut seen_roots: HashSet<PathBuf> = HashSet::new();
    for root in &roots {
        let root_path = PathBuf::from(root);
        if !seen_roots.insert(root_path.clone()) {
            continue;
        }
        all_paths.extend(find_rust_files(&root_path));
    }
    all_paths.sort();
    all_paths.dedup();

    let mut items: Vec<IndexedItem> = Vec::new();
    let mut calls: Vec<CallSite> = Vec::new();
    let mut files_skipped: Vec<String> = Vec::new();
    let files_scanned = all_paths.len();

    for path in &all_paths {
        let display_path = path.to_string_lossy().replace('\\', "/");
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("symbol-index: WARN cannot read {display_path}: {e}");
                files_skipped.push(display_path);
                continue;
            }
        };
        match index_source(&display_path, &text) {
            Some((file_items, file_calls)) => {
                items.extend(file_items);
                calls.extend(file_calls);
            }
            None => {
                eprintln!("symbol-index: WARN cannot parse {display_path}");
                files_skipped.push(display_path);
            }
        }
    }

    let index = Index {
        items,
        calls,
        files_scanned,
        files_skipped: files_skipped.clone(),
    };
    let json = serde_json::to_string(&index).expect("index must serialize");

    match out_path {
        Some(p) => {
            std::fs::write(&p, json).unwrap_or_else(|e| panic!("cannot write {p}: {e}"));
        }
        None => {
            println!("{json}");
        }
    }
    eprintln!(
        "symbol-index: scanned {} file(s), {} item(s) indexed, {} file(s) skipped{}",
        files_scanned,
        index.items.len(),
        files_skipped.len(),
        if files_skipped.is_empty() {
            String::new()
        } else {
            format!(" ({})", files_skipped.join(", "))
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(src: &str) -> (Vec<IndexedItem>, Vec<CallSite>) {
        index_source("t.rs", src).expect("fixture source must parse")
    }

    fn find<'a>(items: &'a [IndexedItem], qualified: &str) -> Option<&'a IndexedItem> {
        items
            .iter()
            .find(|i| i.qualified.as_deref() == Some(qualified))
    }

    #[test]
    fn impl_method_is_qualified_by_its_self_type() {
        let (items, _) = idx("struct Catalog; impl Catalog { pub fn claim_next(&self) {} }");
        let hit = find(&items, "Catalog::claim_next").expect("Catalog::claim_next must be indexed");
        assert_eq!(hit.vis, "pub");
        assert_eq!(hit.kind, "fn");
    }

    #[test]
    fn one_line_method_does_not_pop_the_enclosing_impl() {
        // The exact real-tree defect this tool's own construction found
        // and fixed (jobs_repo.rs): a one-line method's own `{ ... }`
        // must not spuriously close the `impl` block for every method
        // AFTER it. Two methods, the first one-line, the second not.
        let src = "struct Catalog; impl Catalog { \
                    pub fn is_empty(&self) -> bool { true } \
                    pub fn fail_job(&self) {} \
                    }";
        let (items, _) = idx(src);
        assert!(find(&items, "Catalog::is_empty").is_some());
        assert!(
            find(&items, "Catalog::fail_job").is_some(),
            "fail_job lost its Catalog:: qualification -- the one-line-method bug is back"
        );
    }

    #[test]
    fn enum_variants_and_struct_fields_are_qualified() {
        let (items, _) =
            idx("enum ResultTableKind { TrainingSet } struct ResumeState { pub dropout_positions: i32 }");
        assert!(find(&items, "ResultTableKind::TrainingSet").is_some());
        let field = find(&items, "ResumeState::dropout_positions").expect("field must be indexed");
        assert_eq!(field.kind, "field");
        assert_eq!(field.vis, "pub");
    }

    #[test]
    fn cfg_test_mod_marks_every_item_and_call_inside_it_in_test() {
        let src = "fn live() { helper(); }\n\
                    #[cfg(test)] mod tests { fn probe() { helper(); } }\n";
        let (items, calls) = idx(src);
        let live = items.iter().find(|i| i.name == "live").unwrap();
        assert!(
            !live.is_test,
            "a free fn outside #[cfg(test)] must not be marked is_test"
        );
        let probe = items.iter().find(|i| i.name == "probe").unwrap();
        assert!(
            probe.is_test,
            "a fn inside #[cfg(test)] mod must be marked is_test"
        );
        let helper_calls: Vec<&CallSite> = calls.iter().filter(|c| c.callee == "helper").collect();
        assert_eq!(
            helper_calls.len(),
            2,
            "expected exactly the two `helper()` call sites"
        );
        assert!(
            helper_calls.iter().any(|c| !c.in_test),
            "the call inside `live` must NOT be marked in_test"
        );
        assert!(
            helper_calls.iter().any(|c| c.in_test),
            "the call inside `tests::probe` must be marked in_test"
        );
    }

    #[test]
    fn test_attribute_marks_the_fn_and_its_own_calls_without_a_cfg_test_mod() {
        let src = "#[test] fn it_works() { assert_thing(); }";
        let (items, calls) = idx(src);
        let it_works = items.iter().find(|i| i.name == "it_works").unwrap();
        assert!(it_works.is_test);
        assert!(
            calls
                .iter()
                .find(|c| c.callee == "assert_thing")
                .unwrap()
                .in_test
        );
    }

    #[test]
    fn method_calls_are_recorded_by_the_method_name_alone() {
        let (_, calls) = idx("fn f(x: Vec<i32>) { x.iter().map(|v| v).count(); }");
        let callees: Vec<&str> = calls.iter().map(|c| c.callee.as_str()).collect();
        assert!(callees.contains(&"iter"));
        assert!(callees.contains(&"map"));
        assert!(callees.contains(&"count"));
    }

    #[test]
    fn cfg_feature_test_hooks_is_not_mistaken_for_cfg_test() {
        // `is_cfg_test_attr`'s own stated non-substring guard: a MOD
        // gated on a feature literally named "test-hooks" must not be
        // read as `#[cfg(test)]` -- the exact substring trap the guard's
        // own doc comment names (`#[cfg(feature = "test-hooks")]` contains
        // the substring "test" but is not `cfg(test)`).
        let src = "#[cfg(feature = \"test-hooks\")] mod wired { pub fn helper() {} }";
        let (items, _) = idx(src);
        let wired_mod = items
            .iter()
            .find(|i| i.name == "wired" && i.kind == "mod")
            .unwrap();
        assert!(
            !wired_mod.is_test,
            "cfg(feature = \"test-hooks\") must not be read as cfg(test)"
        );
        let helper = items.iter().find(|i| i.name == "helper").unwrap();
        assert!(
            !helper.is_test,
            "an item inside a non-test cfg-gated mod must not be marked is_test"
        );
    }
}
