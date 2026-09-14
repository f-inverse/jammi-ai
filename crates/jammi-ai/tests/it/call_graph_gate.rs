//! #500 U2a fix round 7 — the reachability gate is a real Rust call graph.
//!
//! `pinned_source_gate.rs`'s `no_session_registration_under_fine_tune` polices
//! DIRECT session-binding call sites written inside
//! `crates/jammi-ai/src/fine_tune/**` itself. It structurally cannot see an
//! INDIRECT one: a `fine_tune/` function calling an in-tree helper defined
//! elsewhere (anywhere in `crates/jammi-ai/src` or `crates/jammi-db/src`)
//! that itself — possibly several calls further in — binds a name on the
//! shared `SessionContext` or its catalog. `fine_tune/`'s own architecture
//! makes this real: `training_set.rs`'s tabular arm calls
//! `ResultStore::materialize_training_set` (`crates/jammi-db/src/store/mod.rs`),
//! which several calls later reaches `SessionContext`/`CatalogProvider`
//! registration verbs.
//!
//! A previous version of this gate answered that question with a name-keyed,
//! line-regex fixed point (`session_binding_reachable_names`/`callee_names`,
//! formerly in `pinned_source_gate.rs`). A closing audit found it unsound on
//! three independent axes: its reach universe was scoped to
//! `crates/jammi-db/src/store/**` only, so a real binder defined anywhere
//! else in either crate (several such sites exist today — see
//! [`FINE_TUNE_REACHABLE_NAME_COLLISIONS`]'s doc) was invisible; its edges
//! came only from `.ident(` method-call TEXT, so a free-function call, a
//! `Self::`/`crate::`/`super::`-qualified call, or a call made from inside a
//! closure produced no edge at all (`crates/jammi-db/src/store/mod.rs:1278`'s
//! free-function call to `build_result_table_provider` is exactly this
//! shape, and it is the call that reaches `register_object_store`); and it
//! had no DDL-literal awareness in the fixed point at all (only the
//! separate, also-buggy `fine_tune_ddl_relation_binding_hits` had one, and
//! only for the DIRECT-call-site layer).
//!
//! This file replaces the mechanism, not merely patches it (the closing
//! audit's own ruling): every tracked `.rs` file under `crates/jammi-ai/src`
//! and `crates/jammi-db/src` is parsed as real Rust with `syn`, and a genuine
//! `syn::visit::Visit` walk over every parsed function/method body produces
//! the call-graph edges — a free-function call, a `Self::`/`crate::`/
//! `super::`-qualified call, a plain method call, and a call made from inside
//! a closure are all just `syn::Expr` nodes the SAME walk already visits, so
//! none of them need special-casing to be seen. Every string-literal TOKEN
//! (including one that lives inside a macro invocation such as
//! `format!(..)`, whose arguments `syn` otherwise leaves as an opaque token
//! stream) is checked against [`pinned_source_gate::ddl_statement_shape`].
//! Comments are never tokens in the first place — `syn`'s lexer (via
//! `proc_macro2`) discards them before this file ever sees a token — so a
//! DDL string mentioned in a `//`/`/* */` comment can never be mistaken for
//! a call site, with no masking pass required at all. A DOC comment
//! (`///`/`//!`) is a partial exception — see [`FileVisitor::visit_attribute`].
//!
//! **The graph is still name-keyed, deliberately.** Without full type
//! resolution (this is a `syn` parse, not a `rustc` one), a bare call
//! `foo()` cannot always be resolved to the ONE function it invokes when two
//! functions share the name `foo` — the SAME limitation the old design had,
//! and the same safe direction: a name-keyed graph can only ever make the
//! reachable set LARGER than the true one (a caller of "the wrong" `foo` is
//! still marked reaching if the right one binds), never smaller, so nothing
//! this gate declares safe can be a false negative for this reason. The
//! qualification a real `syn::Path` gives beyond a bare identifier (multiple
//! segments for `Self::`/`crate::`/`super::`/fully-qualified calls) is
//! collapsed to its LAST segment before matching, for the same
//! over-approximating reason `callee_names`'s own doc gave: keeping the
//! reachable set a superset, never a subset, of the true one. The measured
//! cost of that choice on today's tree is
//! [`FINE_TUNE_REACHABLE_NAME_COLLISIONS`]'s eight entries — reviewed, not
//! hidden.
//!
//! **`finish` needed no such entry.** `fine_tune/**` calls `.finish()` in
//! three other, unrelated shapes — `tracing_subscriber::fmt().finish()`
//! (`worker.rs`), a `std::fmt::Formatter::debug_struct(..).finish()` (a
//! `Debug` impl in `training_job.rs`), and
//! `jammi_kernels::admission::ProbeCaptureGuard::finish()`
//! (`worker.rs::probe_acceleration` — the predecessor review named this type
//! `AdmissionProbeCapture`, which is not this codebase's type; corrected
//! here). None of the three is `crates/jammi-db/src/store/building.rs`'s
//! `BuildingTable::finish`, but that one — `materialize_training_set`'s OWN
//! call, not a fine_tune-local one — genuinely IS what the fixed point finds
//! (it calls `bind_result_table` directly), so this is a case where the
//! name-keyed over-approximation happens to land on the true call regardless
//! of the three false ones sharing its name; nothing about that requires a
//! reviewed-collision entry, since the flagged binder itself
//! (`bind_result_table`) is independently, correctly a real crossing.
use std::collections::{BTreeSet, HashMap};

use proc_macro2::TokenTree;
use syn::visit::Visit;
use syn::{Expr, ExprCall, ExprMethodCall, ImplItemFn, ItemFn, LitStr, Macro, TraitItemFn};

use super::pinned_source_gate::{
    ddl_statement_shape, scan_surface, PAIRED_REGISTRATION_VERBS, UNPAIRED_REGISTRATION_VERBS,
};

/// The binder-verb half of the fixed point's seed set: every
/// `register_`/`deregister_` call NAME [`PAIRED_REGISTRATION_VERBS`] /
/// [`UNPAIRED_REGISTRATION_VERBS`] already reviews as binding a
/// caller-chosen token on the shared session (`pinned_source_gate.rs`'s own
/// module-level doc derives and reviews this 24-verb set from the pinned
/// `datafusion = "54.1"` source; this gate reuses that already-reviewed set
/// rather than re-deriving or re-reviewing it, so the two layers can never
/// silently drift onto two different verb lists).
fn registration_verb_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for verb in PAIRED_REGISTRATION_VERBS {
        names.insert(format!("register_{verb}"));
        names.insert(format!("deregister_{verb}"));
    }
    for verb in UNPAIRED_REGISTRATION_VERBS {
        names.insert(format!("register_{verb}"));
    }
    names
}

/// What a [`CgNode`] was observed to bind directly — kept (not just a bool)
/// so a failing assertion can name WHAT a node bound, not just THAT it did.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Binder {
    /// A call whose name is a member of [`registration_verb_names`].
    Verb(String),
    /// A string-literal token matching [`ddl_statement_shape`].
    Ddl(String),
}

/// One function-like item (a free `fn`, an `impl` method, or a trait
/// default-method body) found while parsing the surface: its defining file,
/// its bare name, an ordinal disambiguating same-named functions in the same
/// file (mirrors `pinned_source_gate.rs`'s own allowlist-key discipline —
/// see its module comment on why `(file, name, ordinal)` is the key, never a
/// declaration line), every call-graph edge its body makes (by bare callee
/// name, over-approximated per this file's module doc), and what it binds
/// directly, if anything.
#[derive(Debug, Clone)]
struct CgNode {
    file: String,
    name: String,
    ordinal: usize,
    callees: BTreeSet<String>,
    binder: Option<Binder>,
}

/// Walks one already-parsed file, building its [`CgNode`]s. A stack (not a
/// single "current" slot) because a function item can nest inside another
/// function's body (a local `fn`) or inside an `impl`/`trait` block that
/// itself sits inside a function body; the top of the stack is always the
/// innermost function whose body is being walked, so a call made inside a
/// nested item is attributed to the nested item, never leaked to its
/// enclosing one.
struct FileVisitor<'v> {
    file: String,
    verb_names: &'v BTreeSet<String>,
    nodes: Vec<CgNode>,
    stack: Vec<usize>,
    name_counts: HashMap<String, usize>,
}

impl<'v> FileVisitor<'v> {
    fn new(file: String, verb_names: &'v BTreeSet<String>) -> Self {
        Self {
            file,
            verb_names,
            nodes: Vec::new(),
            stack: Vec::new(),
            name_counts: HashMap::new(),
        }
    }

    fn enter_fn(&mut self, name: String) {
        let counter = self.name_counts.entry(name.clone()).or_insert(0);
        *counter += 1;
        let ordinal = *counter;
        self.nodes.push(CgNode {
            file: self.file.clone(),
            name,
            ordinal,
            callees: BTreeSet::new(),
            binder: None,
        });
        self.stack.push(self.nodes.len() - 1);
    }

    fn exit_fn(&mut self) {
        self.stack.pop();
    }

    /// Record a call-graph edge to `name` from whichever function is
    /// currently on top of the stack, if any (a call at file/module scope —
    /// a `const` initializer, say — has no enclosing function and is simply
    /// not attributed to one; nothing in `fine_tune/**` binds outside a
    /// function body today, and a future one would need a function to house
    /// it before this gate could see it as a crossing at all, the same
    /// disclosed limit `session_binding_reachable_names`'s predecessor
    /// always had for module-level code).
    fn record_callee(&mut self, name: String) {
        let verb_names = self.verb_names;
        if let Some(&idx) = self.stack.last() {
            let node = &mut self.nodes[idx];
            if node.binder.is_none() && verb_names.contains(&name) {
                node.binder = Some(Binder::Verb(name.clone()));
            }
            node.callees.insert(name);
        }
    }

    fn record_ddl(&mut self, text: &str) {
        if let Some(&idx) = self.stack.last() {
            let node = &mut self.nodes[idx];
            if node.binder.is_none() && ddl_statement_shape(text) {
                node.binder = Some(Binder::Ddl(text.to_string()));
            }
        }
    }

    /// String-literal tokens inside a macro invocation
    /// (`format!("CREATE TABLE {name}")`'s argument, say) are not parsed
    /// into `syn::Expr`s by default — `syn::Macro` keeps its argument as an
    /// opaque `proc_macro2::TokenStream`. Walked recursively (a `Group`
    /// token is itself a nested token stream, e.g. the parenthesised
    /// argument list) so a literal nested inside another macro invocation is
    /// still found.
    fn scan_macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        for tt in tokens {
            match tt {
                TokenTree::Literal(lit) => {
                    if let Ok(syn::Lit::Str(s)) = syn::parse_str::<syn::Lit>(&lit.to_string()) {
                        self.record_ddl(&s.value());
                    }
                }
                TokenTree::Group(g) => self.scan_macro_tokens(g.stream()),
                _ => {}
            }
        }
    }
}

impl<'v, 'ast> Visit<'ast> for FileVisitor<'v> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.enter_fn(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.exit_fn();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.enter_fn(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.exit_fn();
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        self.enter_fn(node.sig.ident.to_string());
        syn::visit::visit_trait_item_fn(self, node);
        self.exit_fn();
    }

    /// A free-function call, a `Self::f(..)` call, and a `crate::`/`super::`-
    /// qualified call are ALL `Expr::Call` with an `Expr::Path` callee — the
    /// exact shape the old `.ident(`-only scan could never see
    /// (`store/mod.rs:1278`'s `build_result_table_provider(ctx, ..)` is
    /// this shape). Only the LAST path segment is kept, per this file's
    /// module doc on over-approximation.
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(p) = node.func.as_ref() {
            if let Some(seg) = p.path.segments.last() {
                self.record_callee(seg.ident.to_string());
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    /// A plain `receiver.method(..)` call — over-approximated by method name
    /// alone, the receiver's own type is never inspected (no type
    /// resolution is available from a `syn`-only parse).
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.record_callee(node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }

    /// Every plain and raw string literal `syn` parses as an expression
    /// (`"..."`, `r"..."`, `r#"..."#`, all normalised to the same
    /// `syn::LitStr` by `syn`'s own lexer — no separate raw-string handling
    /// is needed here, unlike `mask_non_code`'s hand-rolled text scan).
    fn visit_lit_str(&mut self, node: &'ast LitStr) {
        self.record_ddl(&node.value());
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        self.scan_macro_tokens(node.tokens.clone());
        syn::visit::visit_macro(self, node);
    }

    /// A `///`/`//!`/`/** */`/`/*! */` DOC comment is NOT stripped the way a
    /// plain `//`/`/* */` comment is — the Rust grammar desugars it into a
    /// real `#[doc = "..."]` attribute carrying the comment text as a
    /// genuine `Expr::Lit(Lit::Str(..))`, which the default `Visit` walk
    /// would otherwise recurse into and hand to [`Self::visit_lit_str`] like
    /// any other string literal — exactly how a module doc's own prose
    /// mentioning `CREATE TABLE` (this file's, or `store/mutable/sqlite.rs`'s
    /// literal DDL-builder doc) would otherwise be misread as a DDL call
    /// site. A `doc` attribute's value is never visited; every other
    /// attribute (`#[cfg(..)]`, a derive's arguments, etc.) still is.
    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        if node.path().is_ident("doc") {
            return;
        }
        syn::visit::visit_attribute(self, node);
    }
}

/// Parse every `(file, text)` pair in `surface` as a whole Rust file and
/// return every [`CgNode`] found, across every file — fails CLOSED (panics
/// naming the file) rather than silently skipping a file `syn` cannot parse,
/// the same discipline `pinned_source_gate::scan_surface`'s own doc commits
/// to for a tracked file it cannot read.
fn parse_call_graph(surface: &[(String, String)], verb_names: &BTreeSet<String>) -> Vec<CgNode> {
    let mut nodes = Vec::new();
    for (file, text) in surface {
        let parsed = syn::parse_file(text).unwrap_or_else(|e| {
            panic!(
                "{file} failed to parse as Rust via `syn::parse_file` ({e}) — this gate fails \
                 closed rather than silently treating an unparseable file as call-free"
            )
        });
        let mut visitor = FileVisitor::new(file.clone(), verb_names);
        visitor.visit_file(&parsed);
        nodes.extend(visitor.nodes);
    }
    nodes
}

/// The names reachable, by call-graph edge, from `roots` — a forward BFS
/// over `nodes`' callee sets, seeded at `roots` and closed under "any name a
/// visited node calls is also visited". Over-approximated by name (this
/// file's module doc): two functions sharing a name are indistinguishable,
/// so visiting either visits both.
fn reachable_from(nodes: &[CgNode], roots: &BTreeSet<String>) -> BTreeSet<String> {
    let mut visited = roots.clone();
    loop {
        let mut changed = false;
        for node in nodes {
            if !visited.contains(&node.name) {
                continue;
            }
            for callee in &node.callees {
                if visited.insert(callee.clone()) {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    visited
}

/// Every name a `crates/jammi-ai/src/fine_tune/**` function's body actually
/// CALLS — the fixed point's root set, per this gate's own doc: "the fixed
/// point runs from every fn item under `crates/jammi-ai/src/fine_tune/**`".
///
/// **Deliberately NOT the fine_tune functions' own names.** An earlier draft
/// of this gate seeded `visited` with every fine_tune-defined function's OWN
/// name; since a name-keyed graph cannot distinguish two functions sharing a
/// name, that seeded a totally unrelated same-named function ELSEWHERE in
/// the two-crate surface as "reached" whenever `fine_tune/**` happened to
/// define its own function of the same generic name (`build`, `finish`,
/// `register_source_tables`-shaped names are common), even though nothing in
/// `fine_tune/` ever called it — a false-positive explosion the OLD
/// `store/**`-scoped design never hit only because it restricted the
/// PROPAGATION universe, not because it seeded correctly. Seeding from the
/// actual outgoing edges keeps the over-approximation in the one place this
/// file's module doc commits to (a shared NAME between the true callee and
/// something unrelated), not in the seed step itself.
fn fine_tune_root_names(nodes: &[CgNode]) -> BTreeSet<String> {
    let mut roots = BTreeSet::new();
    for n in nodes {
        if n.file.starts_with("crates/jammi-ai/src/fine_tune/") {
            roots.extend(n.callees.iter().cloned());
        }
    }
    roots
}

/// Every DISTINCT name reachable from `crates/jammi-ai/src/fine_tune/**`
/// (via [`reachable_from`], starting at [`fine_tune_root_names`]) that is
/// ITSELF a binder — i.e. every [`Binder`] this fixed point's walk actually
/// crosses, keyed by the NAME of the function whose own body performs the
/// bind (not by the bare verb string alone, so `install_result_schema` and
/// `register_schema` are never conflated even though the former's binder is
/// the latter — see [`fine_tune_reachable_bindings_match_allowlist_exactly`]
/// for the reviewed set this must equal).
fn fine_tune_reachable_bindings(nodes: &[CgNode]) -> Vec<(&CgNode, &Binder)> {
    let roots = fine_tune_root_names(nodes);
    let reach = reachable_from(nodes, &roots);
    nodes
        .iter()
        .filter(|n| reach.contains(&n.name))
        .filter_map(|n| n.binder.as_ref().map(|b| (n, b)))
        .collect()
}

/// Whether a binder's own honest property holds — never "this call site
/// happens not to collide today".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingArm {
    /// The bound token is derived fresh on every call (a per-call UUID
    /// suffix, a per-call timestamp+UUID name) — two overlapping calls can
    /// never bind the same name, so there is nothing to collide on.
    #[allow(dead_code)]
    // no current entry uses this arm; kept for the property's own completeness
    UniquePerCall,
    /// The bound token is a FIXED key (a constant name, or a value derived
    /// from an immutable input such as a URL) and every call with that key
    /// binds the identical provider/value — a second call is a redundant
    /// rebind of what the first already installed, never a different value
    /// silently displacing it.
    IdempotentRebind,
}

struct BindingAllowlistEntry {
    file: &'static str,
    name: &'static str,
    ordinal: usize,
    #[allow(dead_code)] // read by a human reviewer at review time, not by the assertion
    arm: BindingArm,
    #[allow(dead_code)] // read by a human reviewer at review time, not by the assertion
    reason: &'static str,
}

/// The reviewed set of GENUINE bindings `fine_tune/**` reaches today — every
/// entry keyed by the binder's OWN `(file, name, ordinal)` identity (mirrors
/// the predecessor `IN_TREE_SESSION_BINDING_ALLOWLIST`'s discipline: never a
/// declaration line, which drifts on an unrelated edit above it).
///
/// **The one real crossing, and its two halves.** `fine_tune/`'s tabular arm
/// makes exactly one direct call out of `fine_tune/**`:
/// `training_set.rs::materialize_and_read` ->
/// `ResultStore::materialize_training_set` (`crates/jammi-db/src/store/mod.rs`).
/// That function does not itself bind anything (it has no entry here for
/// that reason — [`fine_tune_reachable_bindings`] only reports nodes whose
/// OWN body binds), but its two halves each reach a real one:
///
///   - **The fresh path**, `ResultStore::create_table`, names the row
///     `{source_id}__{task}__{model}__{ns-timestamp}_{uuid8}`
///     (`store/mod.rs:1183`). It is not itself a [`Binder`] (it calls no
///     registration verb and executes no DDL — the row it inserts is a
///     catalog INSERT under a UNIQUE `table_name` column, not a DataFusion
///     bind), so it carries no entry in THIS list; its own uniqueness
///     property is proven directly by
///     `materialization::create_table_names_a_concurrent_burst_uniquely_over_one_definition`.
///   - **The reuse/promote path**, `BuildingTable::finish` ->
///     `ResultStore::bind_result_table` -> `ResultStore::register_table` ->
///     {`install_result_schema`, `build_result_table_provider`} — three real
///     binders, each listed below (`register_table` itself never appears as
///     its own entry: it does not directly call a verb, only forwards to the
///     two below — the same "forwarder needs no entry, only a genuine binder
///     does" rule `create_table` falls under).
///
/// Derived from THIS gate's own output at `HEAD` (round 7), not hand-typed:
/// run [`fine_tune_reachable_bindings_match_allowlist_exactly`] with both
/// lists emptied and transcribe the panic's `found` set, then read each hit's
/// actual call site to classify it as a real crossing (here) or a name
/// collision ([`FINE_TUNE_REACHABLE_NAME_COLLISIONS`]).
const FINE_TUNE_REACHABLE_BINDING_ALLOWLIST: &[BindingAllowlistEntry] = &[
    BindingAllowlistEntry {
        file: "crates/jammi-db/src/store/mod.rs",
        name: "bind_result_table",
        ordinal: 1,
        arm: BindingArm::IdempotentRebind,
        reason: "binds directly by calling `self.register_table(..)` -- jammi's OWN method, a \
                 name collision this over-approximating (name-keyed) graph cannot itself tell \
                 apart from DataFusion's `SessionContext::register_table`, but the resolution IS \
                 the real one here (confirmed by reading store/mod.rs:2714-2726): \
                 `bind_result_table` rebinds a table_name an EARLIER call already created, over \
                 that call's own already-written, immutable Parquet bytes -- every call for the \
                 same table_name rebinds the identical artifact; executed proof: \
                 materialization::two_runs_over_one_pinned_definition_share_one_training_set",
    },
    BindingAllowlistEntry {
        file: "crates/jammi-db/src/store/mod.rs",
        name: "install_result_schema",
        ordinal: 1,
        arm: BindingArm::IdempotentRebind,
        reason: "calls `catalog.register_schema(&catalog_opts.default_schema, ..)` under the \
                 session's FIXED default-schema name -- this function's own doc: \"Idempotent: \
                 re-installing the same provider preserves the tables it already holds\" -- every \
                 call binds the schema under the same constant key, never a per-call one",
    },
    BindingAllowlistEntry {
        file: "crates/jammi-db/src/store/mod.rs",
        name: "build_result_table_provider",
        ordinal: 1,
        arm: BindingArm::IdempotentRebind,
        reason: "calls `ctx.runtime_env().register_object_store(&parsed, driver)` keyed by the \
                 storage URL's own scheme+authority -- DataFusion's own key for this registry -- \
                 so every call for the same URL binds the identical driver; this is exactly the \
                 free-function call shape (store/mod.rs:1278, no receiver, reached via \
                 `ResultStore::register_table`) the predecessor `.ident(`-only scan could never \
                 see, and is one of the closing audit's named real sites",
    },
];

/// A [`fine_tune_reachable_bindings`] hit whose NAME happens to collide with
/// an unrelated function elsewhere in the two-crate surface — this
/// over-approximating, name-keyed graph cannot tell the two apart (the same
/// disclosed limit the predecessor `FINE_TUNE_REVIEWED_STORE_NAME_COLLISIONS`
/// reviewed by hand for the single name `finish`), so each is read here,
/// individually, by tracing the ACTUAL call chain from a `fine_tune/**` root
/// rather than trusting the shared name. None is a real crossing: the
/// `fine_tune/**` call that seeds each chain (`.build()`, `.open()`, `.new()`)
/// resolves — read at the actual call site — to an unrelated builder/
/// constructor, never to the `jammi_db::Session`/`InferenceSession`
/// constructor that happens to share the bare name. Kept as DATA, not
/// silently widened away: a FUTURE `fine_tune/` call that genuinely resolves
/// to one of these still needs a real review, the same "reviewed collision,
/// not a blanket exemption" discipline the predecessor used.
struct CollisionEntry {
    file: &'static str,
    name: &'static str,
    ordinal: usize,
    /// The literal call this collision's name traces back to at the
    /// `fine_tune/**` root, and why it is not the same function.
    #[allow(dead_code)] // read by a human reviewer at review time, not by the assertion
    not_the_real_path: &'static str,
}

const FINE_TUNE_REACHABLE_NAME_COLLISIONS: &[CollisionEntry] = &[
    CollisionEntry {
        file: "crates/jammi-db/src/session.rs",
        name: "build",
        ordinal: 1,
        not_the_real_path:
            "root `build`: every `.build()` call under fine_tune (recursively, 38 sites, \
            `grep -rc '.build()' crates/jammi-ai/src/fine_tune` over every `.rs` file there) \
            resolves to an unrelated \
            builder (`hard_negative_miner.rs`'s USearch index builder, `worker.rs`'s \
            `TrainingLoopBuilder`, etc.), never to `jammi_db::session::Session::build` \
            (session.rs:181), which only the engine's own session-construction path calls",
    },
    CollisionEntry {
        file: "crates/jammi-db/src/session.rs",
        name: "register_source_tables",
        ordinal: 1,
        not_the_real_path: "downstream of the `build` collision above (`Session::build` -> \
            `reload_sources` -> `register_source_tables`) -- not an independent crossing",
    },
    CollisionEntry {
        file: "crates/jammi-db/src/source/file_format.rs",
        name: "register_driver_for_url",
        ordinal: 1,
        not_the_real_path: "downstream of the same `build` collision (`register_source_tables` \
            -> `create_listing_table` -> `register_driver_for_url`)",
    },
    CollisionEntry {
        file: "crates/jammi-db/src/store/mutable/postgres.rs",
        name: "create_table_ddl",
        ordinal: 1,
        not_the_real_path: "downstream of the same `build` collision (`Session::build` -> \
            `register` -> `register_in_tx` -> `create_table_ddl`); this DDL targets the mutable- \
            table SQL backend, not a DataFusion `SessionContext`, and is unreachable from \
            `fine_tune/` by any real call regardless",
    },
    CollisionEntry {
        file: "crates/jammi-db/src/store/mutable/sqlite.rs",
        name: "create_table_ddl",
        ordinal: 1,
        not_the_real_path: "same as the Postgres arm above, the SQLite mutable-table backend",
    },
    CollisionEntry {
        file: "crates/jammi-ai/src/session.rs",
        name: "register_query_functions",
        ordinal: 1,
        not_the_real_path: "root `open`: `fine_tune/`'s test fixtures call `.open()` exclusively \
            as `jammi_db::catalog::Catalog::open` (dozens of sites, `grep -rc 'Catalog::open' \
            crates/jammi-ai/src/fine_tune` over every `.rs` file there), never \
            `InferenceSession::open`/`with_observer`/ \
            `with_broker` (session.rs:96,120,140), which only the engine's own session \
            construction calls",
    },
    CollisionEntry {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        name: "register_vector_agg_udafs",
        ordinal: 1,
        not_the_real_path: "downstream of the same `open` collision (`InferenceSession::open` -> \
            `register_query_functions` -> `register_vector_agg_udafs`)",
    },
    CollisionEntry {
        file: "crates/jammi-ai/src/query/content_hash_udf.rs",
        name: "register_content_hash_udf",
        ordinal: 1,
        not_the_real_path: "root `new`: called by virtually every constructor in both crates, \
            including several with no relation to session construction at all; the specific \
            chain (`new` -> `with_observer` -> `wrap` -> `wrap_with` -> \
            `register_content_hash_udf`, session.rs:120,169,167 respectively) is \
            `InferenceSession::with_observer`'s OWN internal call graph, never reached by any \
            `fine_tune/` call to an unrelated `.new()`",
    },
];

/// The mechanical property: every binding [`fine_tune_reachable_bindings`]
/// finds against the REAL surface is either a reviewed real crossing
/// ([`FINE_TUNE_REACHABLE_BINDING_ALLOWLIST`]) or a reviewed name collision
/// ([`FINE_TUNE_REACHABLE_NAME_COLLISIONS`]) — an unlisted hit fails, naming
/// the exact `(file, function, ordinal)` and how to review it.
#[test]
fn fine_tune_reachable_bindings_match_allowlist_exactly() {
    let surface = scan_surface();
    let verb_names = registration_verb_names();
    let nodes = parse_call_graph(&surface, &verb_names);
    let hits: BTreeSet<(String, String, usize)> = fine_tune_reachable_bindings(&nodes)
        .into_iter()
        .map(|(node, _binder)| (node.file.clone(), node.name.clone(), node.ordinal))
        .collect();
    let mut allow: BTreeSet<(String, String, usize)> = FINE_TUNE_REACHABLE_BINDING_ALLOWLIST
        .iter()
        .map(|e| (e.file.to_string(), e.name.to_string(), e.ordinal))
        .collect();
    allow.extend(
        FINE_TUNE_REACHABLE_NAME_COLLISIONS
            .iter()
            .map(|e| (e.file.to_string(), e.name.to_string(), e.ordinal)),
    );
    assert_eq!(
        hits, allow,
        "every binding reachable from crates/jammi-ai/src/fine_tune (recursively) must be either \
         a reviewed real crossing (FINE_TUNE_REACHABLE_BINDING_ALLOWLIST) or a reviewed name \
         collision (FINE_TUNE_REACHABLE_NAME_COLLISIONS) -- found {hits:?}, allowlisted \
         {allow:?}. A name in `found` but not `allow` is an UNREVIEWED path: trace it by \
         grepping the fine_tune call site the offending bare name traces back to and reading \
         whether it is really the \
         flagged function (the same technique every entry above documents)."
    );
}

/// V3(v): with BOTH review lists emptied, the mechanical computation still
/// names the real crossing — i.e. the property genuinely depends on the
/// allowlists (nothing here always passes vacuously). Run directly against
/// [`fine_tune_reachable_bindings`]'s raw output rather than the `#[test]`
/// above (which would need to fail to demonstrate this).
#[test]
fn falsification_the_real_crossing_is_found_without_any_review_list() {
    let surface = scan_surface();
    let verb_names = registration_verb_names();
    let nodes = parse_call_graph(&surface, &verb_names);
    let hits: BTreeSet<(String, String, usize)> = fine_tune_reachable_bindings(&nodes)
        .into_iter()
        .map(|(node, _binder)| (node.file.clone(), node.name.clone(), node.ordinal))
        .collect();
    let real_crossing = (
        "crates/jammi-db/src/store/mod.rs".to_string(),
        "bind_result_table".to_string(),
        1usize,
    );
    assert!(
        hits.contains(&real_crossing),
        "with no allowlist consulted at all, the mechanical computation must still name the real \
         crossing bind_result_table (fine_tune/'s one direct call, materialize_training_set, \
         reaches it via BuildingTable::finish) -- got {hits:?}"
    );
}

/// V3(i): the lead's M3 shape — a free-function wrapper defined OUTSIDE
/// `crates/jammi-db/src/store/**` (this synthetic lives in `pipeline/`) that
/// itself binds a session name, called from `fine_tune/` as a bare
/// free-function call (`helper(ctx)`, no receiver) rather than a method
/// call. The predecessor's `.ident(`-only scan could never see this shape at
/// all; this gate's `Expr::Call` handling must.
#[test]
fn falsification_free_function_wrapper_outside_store_is_flagged() {
    let verb_names = registration_verb_names();
    let wrapper = (
        "crates/jammi-ai/src/pipeline/__probe_freefn__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the free-function-wrapper shape — synthetic producer text fed to parse_call_graph, not real code in this file
            "fn synthetic_free_fn_wrapper(ctx: &SessionContext, name: &str) {\n",
            "    ctx.register_table(name, provider).unwrap();\n",
            "}\n",
        )
        .to_string(),
    );
    let caller = (
        "crates/jammi-ai/src/fine_tune/__probe_freefn__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture — the fine_tune/ call site into the wrapper above, synthetic producer text, not real code in this file
            "fn caller_in_fine_tune(ctx: &SessionContext) {\n",
            "    synthetic_free_fn_wrapper(ctx, \"probe\");\n",
            "}\n",
        )
        .to_string(),
    );
    let surface = vec![wrapper, caller];
    let nodes = parse_call_graph(&surface, &verb_names);
    let hits: BTreeSet<&str> = fine_tune_reachable_bindings(&nodes)
        .into_iter()
        .map(|(node, _)| node.name.as_str())
        .collect();
    assert!(
        hits.contains("synthetic_free_fn_wrapper"),
        "a bare free-function call (no receiver) from fine_tune/ into a wrapper defined outside \
         store (recursively) that itself binds must be flagged, got {hits:?}"
    );
}

/// V3(ii): `Self::`- and `crate::`-qualified calls both produce edges. Two
/// independent checks in one test: (a) a method that binds reached via
/// `Self::inner(..)` from a sibling method (never a bare, unqualified call)
/// still marks the sibling as reachable; (b) a `fine_tune/` function calling
/// a fully `crate::`-qualified path still resolves (by last segment) to the
/// wrapper it names.
#[test]
fn falsification_self_and_crate_qualified_calls_produce_edges() {
    let verb_names = registration_verb_names();
    let helper_file = (
        "crates/jammi-ai/src/pipeline/__probe_qualified__.rs".to_string(),
        concat!(
            "struct Helper;\n",
            "impl Helper {\n",
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the Self::-qualified-call shape — synthetic producer text fed to parse_call_graph, not real code in this file
            "    fn outer(ctx: &SessionContext) {\n",
            "        Self::inner(ctx);\n",
            "    }\n",
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the Self::-qualified-call shape — synthetic producer text fed to parse_call_graph, not real code in this file
            "    fn inner(ctx: &SessionContext) {\n",
            "        ctx.register_table(\"x\", provider).unwrap();\n",
            "    }\n",
            "}\n",
        )
        .to_string(),
    );
    let caller_file = (
        "crates/jammi-ai/src/fine_tune/__probe_qualified__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the crate::-qualified-call shape — synthetic producer text, not real code in this file
            "fn caller_in_fine_tune(ctx: &SessionContext) {\n",
            "    crate::pipeline::Helper::outer(ctx);\n",
            "}\n",
        )
        .to_string(),
    );
    let surface = vec![helper_file, caller_file];
    let nodes = parse_call_graph(&surface, &verb_names);
    // (a) `Self::inner` propagates the bind up to `outer`, which itself has
    // no direct verb call in its own body.
    let outer_node = nodes
        .iter()
        .find(|n| n.name == "outer")
        .expect("outer must be a parsed node");
    assert!(
        outer_node.binder.is_none(),
        "outer's OWN body never calls a verb directly; it becomes reachable only via `inner`"
    );
    assert!(
        outer_node.callees.contains("inner"),
        "a `Self::inner(..)` call must produce an edge to `inner`, got {:?}",
        outer_node.callees
    );
    // (b) the fine_tune caller's fully `crate::`-qualified call resolves (by
    // last segment) and the whole chain is reachable end to end.
    let hits: BTreeSet<&str> = fine_tune_reachable_bindings(&nodes)
        .into_iter()
        .map(|(node, _)| node.name.as_str())
        .collect();
    assert!(
        hits.contains("inner"),
        "a `crate::`-qualified call (`crate::pipeline::Helper::outer(ctx)`) from fine_tune/ must \
         resolve by its last path segment and reach the binder two hops in, got {hits:?}"
    );
}

/// V3(iii): [`ddl_statement_shape`] catches `CREATE OR REPLACE VIEW` (the
/// form DataFusion itself implements as deregister-then-register on the same
/// relation) and lower-case `create view` — both missed by the predecessor's
/// case-sensitive, `OR REPLACE`-blind literal scan.
#[test]
fn falsification_ddl_shape_catches_or_replace_and_lower_case() {
    assert!(ddl_statement_shape("CREATE OR REPLACE VIEW t AS SELECT 1"));
    assert!(ddl_statement_shape("create view t as select 1"));
    assert!(ddl_statement_shape(
        "create or replace external table t stored as parquet location 'x'"
    ));
    assert!(!ddl_statement_shape(
        "this sentence merely mentions create and view separately"
    ));
}

/// V3(iv): a DDL-shaped string sitting inside a `//` or `/* */` comment (not
/// a real string literal) must NOT be flagged — comments are never lexed
/// into tokens at all, so [`FileVisitor::visit_lit_str`] never sees them.
#[test]
fn falsification_ddl_inside_a_comment_is_not_flagged() {
    let verb_names = registration_verb_names();
    let surface = vec![(
        "crates/jammi-ai/src/fine_tune/__probe_ddl_comment__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the DDL-inside-a-comment shape — synthetic producer text, not real code in this file
            "fn commented_only(ctx: &SessionContext) {\n",
            "    // CREATE VIEW t AS SELECT 1 -- never executed, just a comment\n",
            "    /* CREATE TABLE t (x INT) -- also never executed */\n",
            "    let _ = ctx;\n",
            "}\n",
        )
        .to_string(),
    )];
    let nodes = parse_call_graph(&surface, &verb_names);
    let node = nodes
        .iter()
        .find(|n| n.name == "commented_only")
        .expect("commented_only must be a parsed node");
    assert!(
        node.binder.is_none(),
        "a DDL-shaped string inside a comment must never be treated as a binder, got \
         {:?}",
        node.binder
    );

    // Non-vacuousness (the "executed mutation" for this control, applied to
    // the FIXTURE rather than to production code, since comment-stripping is
    // `syn`'s own lexer, not a line of this file's): the identical text, no
    // longer inside a comment but a real string-literal argument, MUST be
    // flagged -- proving the assertion above is discriminating on the
    // comment, not merely on the words never appearing anywhere at all.
    let uncommented = vec![(
        "crates/jammi-ai/src/fine_tune/__probe_ddl_comment__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: non-vacuousness fixture — the same DDL text as a real string-literal argument, not real code in this file
            "fn really_executes(ctx: &SessionContext) {\n",
            "    ctx.sql(\"CREATE VIEW t AS SELECT 1\");\n",
            "}\n",
        )
        .to_string(),
    )];
    let uncommented_nodes = parse_call_graph(&uncommented, &verb_names);
    let uncommented_node = uncommented_nodes
        .iter()
        .find(|n| n.name == "really_executes")
        .expect("really_executes must be a parsed node");
    assert!(
        uncommented_node.binder.is_some(),
        "the SAME DDL text, as a real string-literal argument rather than inside a comment, must \
         be flagged -- otherwise the comment assertion above would be vacuous"
    );
}
