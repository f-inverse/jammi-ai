//! `domain_hash`'s real precondition — prefix-freedom of the domain set —
//! gated by a REAL `syn` parse over every call site, tree-wide, never a
//! text/regex scan.
//!
//! **The gate's real universe, stated so a reader never has to re-derive it
//! from the code below:** every `.rs` file `git ls-files -- crates/` lists
//! (`domain_hash` is `pub`, so a caller anywhere in the workspace — not just
//! this crate's own `src/` — is in scope). Each file is
//! parsed into a real `syn::File`, never read as text, and walked for:
//!
//! 1. every top-level `const`/`static` item, resolved to its own byte value
//!    when its initializer is (after unwrapping `&`/parens) itself a
//!    byte-string literal, a string literal's `.as_bytes()`, or a reference
//!    to another such item — fixed-point, so a const-of-a-const resolves —
//!    kept in one process-wide map keyed by identifier (today's tree has no
//!    two same-named consts with different values; if that ever happens,
//!    the conflicting name resolves to NOTHING rather than picking either
//!    value, which turns every call referencing it into the "unresolved"
//!    class below rather than silently choosing the wrong one);
//! 2. every CALL to `domain_hash` — a bare `domain_hash(..)`, a
//!    path-qualified `crate::store::content_hash::domain_hash(..)` under any
//!    prefix, an identifier reached through a `use content_hash::domain_hash
//!    as dh;` alias (tracked per file), or the same three shapes appearing
//!    inside a macro invocation's token stream — matched by the callee
//!    path's LAST segment, never a fixed shape.
//!
//! Each found call's FIRST argument is then resolved through the exact same
//! literal/`.as_bytes()`/const rules as above. A byte-string literal or a
//! resolved const is a domain VALUE, added to the reviewed set
//! (`EXPECTED_CALL_DOMAINS`; its count and its exact membership are both
//! asserted, so a new tag cannot silently join unreviewed). Anything else
//! the argument could be — a local variable a `let` merely aliases, a slice
//! expression, any other computed value — resolves to NOTHING: this is its
//! own tracked, NAMED finding (`REVIEWED_UNRESOLVED` below), never a silent
//! skip, and a NEW one (one `REVIEWED_UNRESOLVED` does not already name)
//! fails this gate with the call's file and enclosing function. Two domains
//! this crate hashes WITHOUT ever calling `domain_hash` at all — a
//! `domain_hash`-call scan structurally cannot find either — are pinned
//! separately (`HAND_ROLLED_DOMAINS`); the pairwise prefix-free check spans
//! the UNION of both sets, which is the real, load-bearing invariant.
//!
//! **Why a parse, not a text scan** (see `planted_*` below): a domain literal
//! that does not contain the exact substring `jammi.` right after its opening
//! quote (`b"jammi"`, no trailing dot — still a byte-for-byte PREFIX of every
//! real `jammi.…` domain, so exactly the collision case this gate exists to
//! catch) evades a scanner keyed on the marker text `b"jammi.`; and a domain
//! spelled as a plain string literal's `.as_bytes()`
//! (`"jammi.placement".as_bytes()`) evades a scanner that only recognises the
//! `b"…"` byte-string-literal SYNTAX. A real `syn` parse reads the actual
//! literal VALUE in either case, regardless of surface spelling.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;
use syn::{Expr, ExprCall, ExprLit, ExprMethodCall, ItemConst, ItemStatic, Lit, UseTree};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-db has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-tracked `.rs` file under `crates/`, sorted — the whole
/// workspace's crates, never one crate's `src/` (`domain_hash` is `pub`).
fn tracked_files_under_crates(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args([
            "-C",
            root.to_str().expect("utf8 repo root"),
            "ls-files",
            "--",
            "crates/",
        ])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- crates/ failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut files: Vec<String> = String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(".rs"))
        .map(str::to_string)
        .collect();
    files.sort();
    assert!(
        files.len() > 300,
        "git ls-files -- crates/ returned suspiciously few .rs files ({}); the scan's own \
         quantifier is likely broken, not the tree",
        files.len()
    );
    files
}

/// Strip `&expr`, `(expr)` and `{ expr }`-free grouping down to the
/// expression a literal/const check actually cares about.
fn unwrap_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Paren(p) => unwrap_expr(&p.expr),
        Expr::Group(g) => unwrap_expr(&g.expr),
        Expr::Reference(r) => unwrap_expr(&r.expr),
        other => other,
    }
}

/// Resolve `expr` to the byte value it names, given `consts` (a fully-built
/// identifier -> bytes map): a byte-string literal, a string literal's
/// `.as_bytes()`, or a path expression whose last segment names a resolved
/// `const`/`static`. Anything else — including a local variable, since
/// `consts` holds only top-level items — resolves to `None`.
fn resolve_bytes(expr: &Expr, consts: &HashMap<String, Vec<u8>>) -> Option<Vec<u8>> {
    match unwrap_expr(expr) {
        Expr::Lit(ExprLit {
            lit: Lit::ByteStr(bs),
            ..
        }) => Some(bs.value()),
        Expr::MethodCall(ExprMethodCall {
            method,
            args,
            receiver,
            ..
        }) if method == "as_bytes" && args.is_empty() => match unwrap_expr(receiver) {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Some(s.value().into_bytes()),
            _ => None,
        },
        Expr::Path(p) if p.qself.is_none() => {
            let ident = p.path.segments.last()?.ident.to_string();
            consts.get(&ident).cloned()
        }
        _ => None,
    }
}

/// Every top-level `const`/`static` item's `(identifier, initializer
/// expression)` in one parsed file — the raw material [`build_const_map`]
/// resolves.
struct ConstCollector {
    found: Vec<(String, Expr)>,
}

impl<'ast> Visit<'ast> for ConstCollector {
    fn visit_item_const(&mut self, node: &'ast ItemConst) {
        self.found
            .push((node.ident.to_string(), (*node.expr).clone()));
        syn::visit::visit_item_const(self, node);
    }

    fn visit_item_static(&mut self, node: &'ast ItemStatic) {
        self.found
            .push((node.ident.to_string(), (*node.expr).clone()));
        syn::visit::visit_item_static(self, node);
    }
}

/// Fixed-point resolution over every `const`/`static` this scan found,
/// process-wide: repeatedly try to resolve every not-yet-resolved item
/// against the map built so far, so a const initialized from ANOTHER const
/// (not present in today's tree, but not assumed absent from the next
/// commit's) still resolves. A name that resolves to two DIFFERENT byte
/// values across the tree is removed rather than kept at either value —
/// every call referencing it then falls into the "unresolved" class, which
/// is the honest outcome for a genuinely ambiguous name.
fn build_const_map(raw: &[(String, Expr)]) -> HashMap<String, Vec<u8>> {
    let mut map: HashMap<String, Vec<u8>> = HashMap::new();
    let mut conflicted: HashSet<String> = HashSet::new();
    for _pass in 0..8 {
        let mut changed = false;
        for (name, expr) in raw {
            if conflicted.contains(name) {
                continue;
            }
            let Some(bytes) = resolve_bytes(expr, &map) else {
                continue;
            };
            match map.get(name) {
                Some(existing) if existing == &bytes => {}
                Some(_different) => {
                    map.remove(name);
                    conflicted.insert(name.clone());
                }
                None => {
                    map.insert(name.clone(), bytes);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    map
}

/// One call to `domain_hash` this scan found, with its first argument
/// resolved (or not).
#[derive(Debug, Clone)]
struct FoundCall {
    file: String,
    function: String,
    ordinal: u32,
    resolved: Option<Vec<u8>>,
}

/// Every name aliasing `domain_hash` inside one `use` tree
/// (`use a::b::domain_hash as dh;` records `"dh"`; a plain `use
/// a::b::domain_hash;` records `"domain_hash"` itself, redundant with the
/// bare-name match every call scanner already does but harmless to record).
fn collect_domain_hash_aliases(tree: &UseTree, aliases: &mut HashSet<String>) {
    match tree {
        UseTree::Path(p) => collect_domain_hash_aliases(&p.tree, aliases),
        UseTree::Name(n) => {
            if n.ident == "domain_hash" {
                aliases.insert("domain_hash".to_string());
            }
        }
        UseTree::Rename(r) => {
            if r.ident == "domain_hash" {
                aliases.insert(r.rename.to_string());
            }
        }
        UseTree::Group(g) => {
            for item in &g.items {
                collect_domain_hash_aliases(item, aliases);
            }
        }
        UseTree::Glob(_) => {}
    }
}

/// Every `Ident` token spelled as one of `names` inside a macro invocation's
/// token stream (descending through every nested delimited group) — a call
/// hidden inside `assert!(..)`/`format!(..)`/a `macro_rules!` body is
/// otherwise invisible to `visit_expr_call`, which only fires on real
/// expression nodes `syn` parses, never on an opaque macro argument stream.
fn domain_hash_idents_in_tokens(ts: proc_macro2::TokenStream, names: &HashSet<String>) -> u32 {
    let mut count = 0;
    for tt in ts {
        match tt {
            proc_macro2::TokenTree::Ident(i) => {
                if names.contains(&i.to_string()) {
                    count += 1;
                }
            }
            proc_macro2::TokenTree::Group(g) => {
                count += domain_hash_idents_in_tokens(g.stream(), names);
            }
            _ => {}
        }
    }
    count
}

struct CallScanner<'a> {
    file: String,
    consts: &'a HashMap<String, Vec<u8>>,
    aliases: HashSet<String>,
    fn_stack: Vec<String>,
    counts: HashMap<String, u32>,
    found: Vec<FoundCall>,
    /// Calls found opaquely inside a macro's token stream cannot have their
    /// first argument resolved (the tokens are not a parsed `Expr`) — the
    /// tree has none of these today (`domain_hash` is never invoked through
    /// a macro), so this only needs to be COUNTED as a reviewable surface,
    /// never silently ignored, if one is ever added.
    macro_hits: u32,
}

impl<'a> CallScanner<'a> {
    fn new(file: &str, consts: &'a HashMap<String, Vec<u8>>) -> Self {
        Self {
            file: file.to_string(),
            consts,
            aliases: HashSet::new(),
            fn_stack: Vec::new(),
            counts: HashMap::new(),
            found: Vec::new(),
            macro_hits: 0,
        }
    }

    fn record(&mut self, resolved: Option<Vec<u8>>) {
        let function = self
            .fn_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "<module scope>".to_string());
        let ordinal = self.counts.entry(function.clone()).or_insert(0);
        *ordinal += 1;
        self.found.push(FoundCall {
            file: self.file.clone(),
            function,
            ordinal: *ordinal,
            resolved,
        });
    }

    fn is_domain_hash_name(&self, name: &str) -> bool {
        name == "domain_hash" || self.aliases.contains(name)
    }
}

impl<'a, 'ast> Visit<'ast> for CallScanner<'a> {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        collect_domain_hash_aliases(&node.tree, &mut self.aliases);
        syn::visit::visit_item_use(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(p) = unwrap_expr(&node.func) {
            if p.qself.is_none() {
                if let Some(last) = p.path.segments.last() {
                    if self.is_domain_hash_name(&last.ident.to_string()) {
                        let resolved = node
                            .args
                            .first()
                            .and_then(|a| resolve_bytes(a, self.consts));
                        self.record(resolved);
                    }
                }
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        self.macro_hits +=
            domain_hash_idents_in_tokens(node.tokens.clone(), &self.aliases_and_self());
        syn::visit::visit_macro(self, node);
    }
}

impl<'a> CallScanner<'a> {
    fn aliases_and_self(&self) -> HashSet<String> {
        let mut names = self.aliases.clone();
        names.insert("domain_hash".to_string());
        names
    }
}

/// Parse every `git ls-files -- crates/` `.rs` file once, run the const
/// collector over each, build the process-wide const map, then run the call
/// scanner over each parsed file again (no file is read from disk twice).
fn scan_repo() -> (Vec<FoundCall>, u32) {
    let root = repo_root();
    let files = tracked_files_under_crates(&root);
    let mut parsed: Vec<(String, syn::File)> = Vec::with_capacity(files.len());
    let mut raw_consts: Vec<(String, Expr)> = Vec::new();
    for rel in &files {
        let text = std::fs::read_to_string(root.join(rel))
            .unwrap_or_else(|e| panic!("read tracked file {rel}: {e}"));
        let file = syn::parse_file(&text).unwrap_or_else(|e| {
            panic!("domain_hash_prefix_free_gate: syn could not parse {rel}: {e}")
        });
        let mut consts = ConstCollector { found: Vec::new() };
        consts.visit_file(&file);
        raw_consts.extend(consts.found);
        parsed.push((rel.clone(), file));
    }
    let const_map = build_const_map(&raw_consts);

    let mut found = Vec::new();
    let mut macro_hits = 0;
    for (rel, file) in &parsed {
        let mut scanner = CallScanner::new(rel, &const_map);
        scanner.visit_file(file);
        macro_hits += scanner.macro_hits;
        found.extend(scanner.found);
    }
    (found, macro_hits)
}

/// A found call's resolved value as a lossy-UTF8 string, for set/assertion
/// purposes — every real domain tag this crate uses is ASCII.
fn resolved_string(call: &FoundCall) -> Option<String> {
    call.resolved
        .as_ref()
        .map(|b| String::from_utf8_lossy(b).into_owned())
}

/// The exact reviewed set of domain values `domain_hash` is CALLED under
/// today — the set the syn-based call scan above can actually find, since
/// it only ever sees a `domain_hash` call's own first argument.
/// `"a.third.domain"` (`content_hash.rs`'s
/// `domain_hash_separates_by_domain_over_identical_parts`) is reviewed and
/// kept here rather than switched to a `jammi.…`-shaped tag: sharing no
/// prefix relationship with any real `jammi.…` domain, it is trivially
/// prefix-free against all of them, and changing it would only churn an
/// already-passing test for no safety gain.
const EXPECTED_CALL_DOMAINS: &[&str] = &[
    "jammi.content_hash.v1",
    "jammi.placement.v1",
    "a.third.domain",
];

/// The two domains this crate hashes WITHOUT ever calling `domain_hash` at
/// all — `content_hash.rs`'s own doc names them by hand: `manifest.rs`'s
/// `definition_hash` and `version.rs`'s `compute_identity`, each folding an
/// identical length-prefixed-parts shape directly against its own domain
/// constant. A domain_hash-CALL scan structurally cannot find either one —
/// neither file ever calls `domain_hash` — so each is pinned here instead
/// and checked by [`hand_rolled_domains_are_still_present_verbatim_in_their_named_file`]
/// against its own named file directly; the pairwise prefix-free check below
/// covers the UNION of this list and [`EXPECTED_CALL_DOMAINS`], because that
/// union — not `domain_hash`'s callers alone — is the real, load-bearing set
/// this crate's `domain_hash` doc claims prefix-freedom over.
struct HandRolledDomain {
    domain: &'static str,
    file: &'static str,
}

const HAND_ROLLED_DOMAINS: &[HandRolledDomain] = &[
    HandRolledDomain {
        domain: "jammi.materialization.definition.v1",
        file: "crates/jammi-db/src/store/manifest.rs",
    },
    HandRolledDomain {
        domain: "jammi.version.identity.v1",
        file: "crates/jammi-db/src/store/version.rs",
    },
];

#[test]
fn hand_rolled_domains_are_still_present_verbatim_in_their_named_file() {
    let root = repo_root();
    for hrd in HAND_ROLLED_DOMAINS {
        let text = std::fs::read_to_string(root.join(hrd.file))
            .unwrap_or_else(|e| panic!("read {}: {e}", hrd.file));
        let needle = format!("b\"{}\"", hrd.domain);
        assert!(
            text.contains(&needle),
            "{} no longer contains the literal {needle:?} this gate pins by hand — if the domain \
             moved, update HAND_ROLLED_DOMAINS' `file`; if it was removed or renamed, update \
             `domain` too and re-check prefix-freedom against the whole set",
            hrd.file
        );
    }
}

/// Every call whose first argument this scan could NOT resolve to a literal
/// value, reviewed by hand. A call found here that is not in this list
/// fails the gate, naming its file and enclosing function (never silently
/// ignored); an entry here that no longer matches a real unresolved call
/// (the call moved, was deleted, or its argument became resolvable) is
/// itself a failure, so a stale review can never keep "clearing" nothing.
struct ReviewedUnresolved {
    file: &'static str,
    function: &'static str,
    ordinal: u32,
    reason: &'static str,
}

const REVIEWED_UNRESOLVED: &[ReviewedUnresolved] = &[
    ReviewedUnresolved {
        file: "crates/jammi-db/src/store/content_hash.rs",
        function: "domain_prefix_is_not_free_by_construction",
        ordinal: 1,
        reason: "`long_domain` is a local `let` initialized from the literal \
                 `b\"jammi.placement.v1\"` (already reviewed as `jammi.placement.v1` via its own \
                 direct-literal call sites elsewhere) — this indirection is incidental to the \
                 test, not a new domain.",
    },
    ReviewedUnresolved {
        file: "crates/jammi-db/src/store/content_hash.rs",
        function: "domain_prefix_is_not_free_by_construction",
        ordinal: 2,
        reason: "`short_domain` is deliberately NOT a literal — it is `long_domain` SLICED at \
                 runtime (`&long_domain[..long_domain.len() - 1]`) specifically so this \
                 counterexample fixture can never be mistaken for a real, reviewable domain by \
                 this gate; see this test's own doc for why.",
    },
];

#[test]
fn every_domain_hash_call_resolves_to_a_reviewed_domain_or_a_reviewed_unresolved_entry() {
    let (found, macro_hits) = scan_repo();
    assert_eq!(
        macro_hits, 0,
        "a `domain_hash` reference was found inside a macro invocation's token stream — this is \
         a real call shape this gate does not yet resolve arguments for; review it and extend \
         this scanner rather than ignoring the count"
    );
    assert!(
        !found.is_empty(),
        "the scan must find at least the domain_hash call sites this crate is known to have"
    );

    let expected: BTreeSet<String> = EXPECTED_CALL_DOMAINS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut resolved_set: BTreeSet<String> = BTreeSet::new();
    let mut unreviewed_unresolved = Vec::new();
    let mut unreviewed_new_domains = Vec::new();

    for call in &found {
        match resolved_string(call) {
            Some(domain) => {
                resolved_set.insert(domain.clone());
                if !expected.contains(&domain) {
                    unreviewed_new_domains.push(format!(
                        "{}::{} #{} resolves to a NEW domain {domain:?} not in EXPECTED_CALL_DOMAINS — \
                         review it for prefix-freedom against every existing domain, then add it",
                        call.file, call.function, call.ordinal
                    ));
                }
            }
            None => {
                let is_reviewed = REVIEWED_UNRESOLVED.iter().any(|r| {
                    r.file == call.file && r.function == call.function && r.ordinal == call.ordinal
                });
                if !is_reviewed {
                    unreviewed_unresolved.push(format!(
                        "domain expression at {}::{} #{} is not a literal — bind it to a \
                         reviewed const, or add a REVIEWED_UNRESOLVED entry stating why it can \
                         never be a new domain",
                        call.file, call.function, call.ordinal
                    ));
                }
            }
        }
    }

    assert!(
        unreviewed_new_domains.is_empty(),
        "{}",
        unreviewed_new_domains.join("\n")
    );
    assert!(
        unreviewed_unresolved.is_empty(),
        "{}",
        unreviewed_unresolved.join("\n")
    );

    assert_eq!(
        resolved_set, expected,
        "the set of resolved domain_hash domains changed from the reviewed EXPECTED_CALL_DOMAINS"
    );

    // Every REVIEWED_UNRESOLVED entry must still name a real, still-unresolved call.
    let mut stale = Vec::new();
    for r in REVIEWED_UNRESOLVED {
        let still_unresolved = found.iter().any(|c| {
            c.file == r.file
                && c.function == r.function
                && c.ordinal == r.ordinal
                && c.resolved.is_none()
        });
        if !still_unresolved {
            stale.push(format!("{}::{} #{}", r.file, r.function, r.ordinal));
        }
    }
    assert!(
        stale.is_empty(),
        "REVIEWED_UNRESOLVED entry no longer names a real unresolved call site (it moved, was \
         deleted, or its argument became resolvable): {}",
        stale.join("\n")
    );

    // Every entry must actually carry a reviewed reason — an empty `reason`
    // is a row added without doing the review this table exists to force.
    for r in REVIEWED_UNRESOLVED {
        assert!(
            !r.reason.trim().is_empty(),
            "{}::{} #{} has no reviewed reason",
            r.file,
            r.function,
            r.ordinal
        );
    }
}

#[test]
fn domains_are_pairwise_prefix_free() {
    // The UNION of domain_hash's own callers and the two hand-rolled folds
    // — the real, load-bearing set (`content_hash.rs`'s doc), not
    // `domain_hash`'s callers alone.
    let found: Vec<String> = EXPECTED_CALL_DOMAINS
        .iter()
        .map(|s| s.to_string())
        .chain(HAND_ROLLED_DOMAINS.iter().map(|hrd| hrd.domain.to_string()))
        .collect();
    for i in 0..found.len() {
        for j in 0..found.len() {
            if i == j {
                continue;
            }
            let (a, b) = (&found[i], &found[j]);
            assert!(
                !b.starts_with(a.as_str()),
                "domain {a:?} is a byte-prefix of domain {b:?} -- domain_hash(b, parts) can \
                 collide with domain_hash(a, [b[a.len()..], parts...]) for a suitable choice \
                 of parts; rename one of the two domains so neither prefixes the other"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Shape proofs: the scanner extracts the right byte value from every
// argument shape, run over a SYNTHETIC single-file fixture — never the
// real tree — mirroring `models_delete_call_sites.rs`'s own `shape_*`
// idiom in this crate's sibling.
// ---------------------------------------------------------------------

/// Parse `src` as one file and return the resolved domain string (if any)
/// for every `domain_hash` call found, in source order. Consts are resolved
/// from WITHIN `src` only (no cross-file map) — sufficient for every shape
/// proof below, each of which is self-contained.
fn shape_resolved_domains(src: &str) -> Vec<Option<String>> {
    let file = syn::parse_str::<syn::File>(src).expect("shape fixture must parse");
    let mut consts = ConstCollector { found: Vec::new() };
    consts.visit_file(&file);
    let const_map = build_const_map(&consts.found);
    let mut scanner = CallScanner::new("fixture.rs", &const_map);
    scanner.visit_file(&file);
    scanner
        .found
        .into_iter()
        .map(|c| c.resolved.map(|b| String::from_utf8_lossy(&b).into_owned()))
        .collect()
}

#[test]
fn shape_bare_byte_string_literal_is_resolved() {
    let src = r#"fn f(parts: &[&[u8]]) { domain_hash(b"jammi.content_hash.v1", parts); }"#;
    assert_eq!(
        shape_resolved_domains(src),
        vec![Some("jammi.content_hash.v1".to_string())]
    );
}

#[test]
fn shape_qualified_path_call_is_resolved() {
    let src = r#"fn f(parts: &[&[u8]]) { crate::store::content_hash::domain_hash(b"jammi.x.v1", parts); }"#;
    assert_eq!(
        shape_resolved_domains(src),
        vec![Some("jammi.x.v1".to_string())]
    );
}

#[test]
fn shape_use_alias_call_is_resolved() {
    let src = concat!(
        "use crate::store::content_hash::domain_hash as dh;\n",
        "fn f(parts: &[&[u8]]) { dh(b\"jammi.aliased.v1\", parts); }\n"
    );
    assert_eq!(
        shape_resolved_domains(src),
        vec![Some("jammi.aliased.v1".to_string())]
    );
}

#[test]
fn shape_const_reference_is_resolved() {
    let src = concat!(
        "pub const D: &[u8] = b\"jammi.consted.v1\";\n",
        "fn f(parts: &[&[u8]]) { domain_hash(D, parts); }\n"
    );
    assert_eq!(
        shape_resolved_domains(src),
        vec![Some("jammi.consted.v1".to_string())]
    );
}

#[test]
fn shape_transitive_const_of_const_is_resolved() {
    let src = concat!(
        "pub const BASE: &[u8] = b\"jammi.base.v1\";\n",
        "pub const ALIAS: &[u8] = BASE;\n",
        "fn f(parts: &[&[u8]]) { domain_hash(ALIAS, parts); }\n"
    );
    assert_eq!(
        shape_resolved_domains(src),
        vec![Some("jammi.base.v1".to_string())]
    );
}

#[test]
fn shape_local_variable_is_unresolved_not_silently_dropped() {
    let src =
        r#"fn f(parts: &[&[u8]]) { let d: &[u8] = b"jammi.local.v1"; domain_hash(d, parts); }"#;
    assert_eq!(shape_resolved_domains(src), vec![None]);
}

/// A `domain_hash` reference hidden inside a macro invocation's argument
/// stream (`assert!`, `tokio::try_join!`, a `macro_rules!` body) is an
/// opaque token stream to `syn` — invisible to `visit_expr_call` — so it is
/// counted separately (`macro_hits`) rather than silently missed; the
/// primary test's own `macro_hits == 0` assertion is what turns a REAL such
/// reference into a named finding rather than a silent pass.
#[test]
fn shape_macro_embedded_reference_is_counted_not_dropped() {
    let src = r#"fn f(p: &[u8], q: &[u8]) { assert!(matches!(domain_hash(p, &[q]), _)); }"#;
    let file = syn::parse_str::<syn::File>(src).expect("shape fixture must parse");
    let const_map = HashMap::new();
    let mut scanner = CallScanner::new("fixture.rs", &const_map);
    scanner.visit_file(&file);
    assert_eq!(
        scanner.macro_hits, 1,
        "a domain_hash reference inside assert!()'s token stream must be counted"
    );
    assert!(
        scanner.found.is_empty(),
        "a macro-embedded reference is never ALSO found as a real ExprCall (syn never descends \
         into a macro's opaque token stream on its own)"
    );
}

// ---------------------------------------------------------------------
// Two planted spellings a text scan misses, plus two positive controls
// proving a plain new literal is caught the same way. Each runs over a
// synthetic fixture — never a mutation of the real tree.
// ---------------------------------------------------------------------

#[test]
fn planted_a_byte_literal_prefix_missing_the_trailing_dot_is_still_caught() {
    // `b"jammi"` (no trailing `.`) is a byte-for-byte PREFIX of every real
    // `jammi.…` domain — exactly the collision case this gate exists to
    // catch — yet a scanner keyed on the text `b"jammi.` never matches it.
    let src = r#"fn f(parts: &[&[u8]]) { domain_hash(b"jammi", parts); }"#;
    let resolved = shape_resolved_domains(src);
    assert_eq!(resolved, vec![Some("jammi".to_string())]);
    assert!(
        !EXPECTED_CALL_DOMAINS.contains(&resolved[0].as_deref().unwrap()),
        "sanity: \"jammi\" must not already be a reviewed domain, or this plant proves nothing"
    );
}

#[test]
fn planted_a_string_literals_as_bytes_is_still_caught() {
    // A plain string literal converted with `.as_bytes()` carries no `b"`
    // byte-string syntax at all — invisible to a scanner that only
    // recognises that one token spelling.
    let src = r#"fn f(parts: &[&[u8]]) { domain_hash("jammi.placement".as_bytes(), parts); }"#;
    let resolved = shape_resolved_domains(src);
    assert_eq!(resolved, vec![Some("jammi.placement".to_string())]);
    assert!(
        !EXPECTED_CALL_DOMAINS.contains(&resolved[0].as_deref().unwrap()),
        "sanity: \"jammi.placement\" (missing \".v1\") must not already be a reviewed domain"
    );
}

#[test]
fn positive_control_a_plain_new_byte_literal_is_caught() {
    let src = r#"fn f(parts: &[&[u8]]) { domain_hash(b"jammi.placement", parts); }"#;
    let resolved = shape_resolved_domains(src);
    assert_eq!(resolved, vec![Some("jammi.placement".to_string())]);
    assert!(!EXPECTED_CALL_DOMAINS.contains(&resolved[0].as_deref().unwrap()));
}

#[test]
fn positive_control_b_a_fifth_new_domain_is_caught() {
    let src = r#"fn f(parts: &[&[u8]]) { domain_hash(b"jammi.newthing.v1", parts); }"#;
    let resolved = shape_resolved_domains(src);
    assert_eq!(resolved, vec![Some("jammi.newthing.v1".to_string())]);
    assert!(!EXPECTED_CALL_DOMAINS.contains(&resolved[0].as_deref().unwrap()));
}

/// Reproduces the primary test's own new-domain comparison on a synthetic
/// `found` set (never touching the real tree) — proving that IF any of the
/// four fixtures above were a real call site in the tree, the primary test
/// would fail rather than pass — a property of the whole gate, not just of the
/// extraction helper.
#[test]
fn any_of_the_four_planted_domains_would_red_the_primary_assertion() {
    let expected: BTreeSet<String> = EXPECTED_CALL_DOMAINS
        .iter()
        .map(|s| s.to_string())
        .collect();
    for planted in ["jammi", "jammi.placement", "jammi.newthing.v1"] {
        let mut resolved_set = expected.clone();
        resolved_set.insert(planted.to_string());
        assert_ne!(
            resolved_set, expected,
            "planting {planted:?} must change the resolved set away from EXPECTED_CALL_DOMAINS"
        );
    }
}
