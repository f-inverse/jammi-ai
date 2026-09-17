//! I2 adversarial-audit fix: [`jammi_kernels::admission::ProbedOp`] is
//! sealed against every crate but `jammi-kernels` itself — `#[non_exhaustive]`
//! plus every field `pub(crate)` with public read accessors
//! (`admission.rs`'s own doc on `ProbedOp`/`Sealed` has the full argument).
//! `#[non_exhaustive]` alone is NOT the seal: it refuses struct-EXPRESSION
//! construction and exhaustive matching from outside the crate, but does
//! nothing about a caller that COPIES an existing `&'static ProbedOp`
//! const (`Copy`, no struct expression involved) and then ASSIGNS one of
//! its OWN fields on that copy — a probe reproduced exactly this against
//! an earlier revision (fields still `pub`): a cross-crate integration
//! test copied `LAYER_NORM`, assigned `report_key`/`registry`/`dry_run`
//! directly, and `admit` honoured the forged value. Field privacy closes
//! THAT gap (field assignment needs the field to be VISIBLE at the
//! assignment site, exactly as field READ does); `#[non_exhaustive]`
//! closes construction. Together they cover every OTHER crate. Rust has
//! no "seal from myself" mechanism, so a same-crate forgery inside
//! `jammi-kernels` itself is a residual neither touches — this file
//! closes it mechanically instead: a real `syn` AST walk over
//! `crates/jammi-kernels/src/**` AND `crates/jammi-kernels/tests/**` (the
//! stated universe — a same-crate forgery could live in either), proving
//! every EXPRESSION that can produce a fresh `ProbedOp` VALUE is
//! reviewed:
//!
//!   1. `ProbedOp::new(...)` calls (`syn::ExprCall`), matched by their
//!      LAST TWO path segments under ANY qualifying prefix — a bare
//!      `ProbedOp::new`, a fully qualified `crate::admission::ProbedOp::new`,
//!      a qualified-self `<ProbedOp>::new` (a DIFFERENT `syn::Expr::Path`
//!      shape — the type lives in the expression's own `qself` field, not
//!      in `path`'s segments, so this direction reads `qself` FIRST and
//!      only falls back to the path's own second-to-last segment when
//!      `qself` is absent), `Self::new` inside `impl ProbedOp`, and
//!      `Alias::new`/`<Alias>::new` for any same-crate `type Alias =
//!      ProbedOp;` this file's own alias-collection pass resolved — never
//!      an exact, fixed-length segment-vector equality, which matches
//!      ONLY the unqualified two-segment shape and silently misses every
//!      other one. Name-keyed against the REAL, linked-in [`PROBED_OPS`]
//!      constant by each call's own first-argument string literal (its
//!      `report_key`), not by bare count — see "Why name-keyed, not
//!      count-keyed" below.
//!   2. `ProbedOp { ... }` struct-literal expressions (`syn::ExprStruct`,
//!      also checked against `qself` first, then the path's last segment
//!      or a resolved alias) — name-keyed against ONE reviewed site,
//!      `ProbedOp::new`'s own constructor body (the canonical construction
//!      this whole file exists to make the ONLY one). A same-crate `pub
//!      const X: ProbedOp = ProbedOp { ..., _sealed: Sealed }` literal
//!      ANYWHERE ELSE — invisible to an earlier revision of this oracle,
//!      which counted only `ProbedOp::new` calls — REDs here.
//!   3. Any `fn` (free or `impl` method) whose own return type names
//!      `ProbedOp` (directly, as `Self` inside `impl ProbedOp`, or as a
//!      resolved same-crate alias) — name-keyed against the SAME one
//!      reviewed site. A new fn returning a `ProbedOp` by some OTHER
//!      construction path (built from a mutated copy, forwarded from
//!      another crate through an FFI-shaped boundary, anything syn cannot
//!      see the BODY of well enough to classify directly) is caught at
//!      this coarser, signature-level grain instead.
//!   4. Every same-crate `type Alias = ProbedOp;` declaration
//!      (`syn::ItemType`), resolved in a SEPARATE first pass over the same
//!      file universe before the main scan runs, so directions 1–3 above
//!      see `Alias::new`/`-> Alias` exactly as they see `ProbedOp::new`/
//!      `-> ProbedOp` — single-level only (an alias of an alias is not
//!      chased further; none exists in this crate today, and the
//!      `falsification_*` fixtures below exercise a direct alias, the
//!      shape a real bypass attempt used).
//!   5. Every macro INVOCATION's own token stream (`syn::Macro`, visited
//!      via `visit_macro` — a DIFFERENT trait method from `visit_item_macro`
//!      below, which sees only macro *definitions*), token-walked
//!      recursively through every group for an `Ident` token spelled
//!      `ProbedOp` or a resolved alias — `vec![ProbedOp::new(...)]` and
//!      `vec![ProbedOp { ..., _sealed: Sealed }]` are both opaque to
//!      directions 1/2's typed `Expr` traversal (a macro's own arguments
//!      are an unparsed `TokenStream` to `syn`, not `Expr`/`ExprStruct`
//!      nodes), so this direction is the ONLY one that sees them.
//!      Matching by exact `Ident` token (never a rendered-string
//!      `.contains(...)` substring check) means `ProbedOpKind` — this
//!      crate's own, unrelated enum, mentioned inside `matches!`/`assert!`
//!      throughout its real test suite — never false-positives: it is one
//!      `Ident` token, not two, and never equals `ProbedOp`. Every
//!      `macro_rules!` DEFINITION's own body (`node.path` naming
//!      `macro_rules` itself) is excluded here and reviewed instead by
//!      direction 6, so a legitimate fixture macro's own construction
//!      is never double-reported under two different directions.
//!   6. `macro_rules!` DEFINITION bodies (`visit_item_macro`, an opaque
//!      token stream to a normal `syn::visit::Visit` traversal — its
//!      pattern/template syntax, e.g. `($key:expr) => { ... };`, is not
//!      valid standalone Rust expression syntax `syn` can parse as an
//!      `Expr`, so none of directions 1–3 structurally descend into one):
//!      `test_two_arm!`/`test_cascade!` each contain exactly one
//!      `ProbedOp::new(...)` call in their own body; this direction finds
//!      that with the SAME exact-`Ident`-token walk direction 5 uses
//!      (reused, not a second `.contains(...)` implementation to drift out
//!      of sync with it), reviewed as a SEPARATE category, never folded
//!      into 1/2 (a `test_two_arm!` INVOCATION, e.g.
//!      `test_two_arm!("test_op")`, is a macro invocation — direction 5's
//!      own territory, not direction 1's — but its own argument tokens
//!      are just a string literal, never an `Ident` spelled `ProbedOp`, so
//!      it never appears there either; only the two macro DEFINITIONS are
//!      counted, once each, regardless of how many times the fixture
//!      macros are invoked across this crate's own tests).
//!   7. `mem::transmute::<_, ProbedOp>(...)`-shaped calls (any path
//!      prefix — `std::mem::transmute`, `core::mem::transmute`, a bare
//!      `use`-imported `transmute`) whose own turbofish explicitly names
//!      `ProbedOp`/a resolved alias as either type argument, AND a
//!      `let binding: ProbedOp = unsafe { transmute(...) };`-shaped
//!      pattern (the type read off the `let`'s own annotation when the
//!      turbofish is omitted and inferred instead). **The one HONESTLY
//!      NAMED residual, not closed and not silently omitted**: this crate
//!      does NOT carry `#![forbid(unsafe_code)]` (checked directly against
//!      `crates/jammi-kernels/src/lib.rs`; the crate's own CUDA FFI paths
//!      use real `unsafe` blocks), so a `transmute` (or a raw-pointer
//!      cast) whose target `ProbedOp` type is established SOME OTHER WAY
//!      syntax alone cannot resolve — inferred through a function's own
//!      return position several calls away, assigned into a field of
//!      another struct whose own field type is read from a THIRD file,
//!      anything requiring a real type checker rather than a syntax walk
//!      — remains outside what a `syn`-only, type-checker-free oracle can
//!      structurally rule out. This is not a claim of impossibility: it
//!      is the honestly-stated boundary of what directions 1–7 prove nothing
//!      beyond, exactly as this whole file's `#[non_exhaustive]`+field-privacy
//!      argument is honest about covering every OTHER crate, never THIS one,
//!      without the `syn` walk.
//!
//! **Why name-keyed, not count-keyed, for direction 1.** A prior revision
//! compared `direct_call_sites.len()` against `PROBED_OPS.len()` by bare
//! `==` — a shape that hides an "invisible extra row": if one legitimate
//! call site is MISSED by an under-matching path check (exactly the bug
//! direction 1's exact-segment-vector-equality had before this fix) at
//! the SAME time an unrelated forged call site is ADDED elsewhere, the
//! two errors cancel in the total count and `==` reports nothing wrong.
//! `PROBED_OPS`'s own array holds VALUES at runtime, not the Rust const
//! IDENTIFIERS (`LAYER_NORM`, …) that produced them, but each row's own
//! `report_key()` IS recoverable from a real value, and each real
//! `ProbedOp::new(...)` call's own first argument names that SAME
//! `report_key` as a string literal — so direction 1 now compares the SET
//! of `report_key` literals FOUND at direct call sites against the SET
//! `PROBED_OPS` itself reports, three ways, each individually named on
//! failure: any found key ABSENT from `PROBED_OPS` (forged/orphaned), any
//! `PROBED_OPS` key never found at a call site (missing), and any key
//! found at MORE than one call site (duplicated) — a floor
//! (`direct_call_sites.len() >= PROBED_OPS.len()`) is asserted directly
//! alongside, never trusted as an implicit corollary. A call site whose
//! own first argument is not a string literal (so its `report_key`
//! cannot be read back out at all) is itself reported as unreviewed,
//! never silently skipped.
//!
//! **Why name-keyed for directions 2/3 specifically.** Both have EXACTLY
//! one legitimate site each (`ProbedOp::new`'s own body, and
//! `ProbedOp::new` itself, respectively) — a fixed, tiny, by-NAME
//! reviewed set is the more precise check available for a set of size
//! one, and reads better in a failure message than "1 != 1, somewhere".

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;

/// `macro_rules!` definitions this crate reviews as legitimate
/// `ProbedOp`-constructing TEST fixtures — `crates/jammi-kernels/src/
/// admission.rs`'s own `test_two_arm`/`test_cascade` macros, used
/// exclusively inside its `#[cfg(test)] mod tests`. A NEW macro_rules!
/// definition anywhere in the scanned universe whose own body contains an
/// `Ident` token spelled `ProbedOp` (or a resolved alias) and is NOT one
/// of these two names REDs this oracle's macro-definition direction.
const REVIEWED_TEST_FIXTURE_MACROS: &[&str] = &["test_two_arm", "test_cascade"];

/// The ONE reviewed (file, fn name) pair allowed to contain a
/// `ProbedOp { ... }` struct literal, or to have a return type naming
/// `ProbedOp`/`Self`/an alias — `ProbedOp::new` itself, the sole
/// canonical constructor this whole file exists to keep sole.
const REVIEWED_CONSTRUCTOR: (&str, &str) = ("crates/jammi-kernels/src/admission.rs", "new");

/// `CARGO_MANIFEST_DIR` is `crates/jammi-kernels`; the repo root is two
/// levels up — mirrors `crates/jammi-db/tests/it/models_delete_call_sites.rs`'s
/// own helper of the same shape.
fn repo_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("CARGO_MANIFEST_DIR has no grandparent: {manifest_dir:?}"))
        .to_path_buf()
}

/// `git ls-files`, scoped to `dir`, relative to the repository root —
/// never a hand-maintained directory walk that could silently stop early.
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

/// `owner` (a path segment's own bare name) identifies `ProbedOp` — either
/// literally, as `Self` while the caller reports the enclosing `impl`
/// block's own `Self` type IS `ProbedOp`, or as a same-crate type alias
/// this file's own [`AliasCollector`] pass resolved. The ONE place every
/// "does this name mean `ProbedOp`" question in this file is answered, so
/// the call-site/struct-literal/return-type directions can never drift
/// out of sync with each other on what counts.
fn owner_names_probed_op(owner: &str, self_is_probed_op: bool, aliases: &HashSet<String>) -> bool {
    owner == "ProbedOp" || (owner == "Self" && self_is_probed_op) || aliases.contains(owner)
}

/// `ty` names `ProbedOp` (directly, as `Self`, or as a resolved alias) —
/// the only shapes a `-> ProbedOp`-equivalent return type, or a
/// `<ProbedOp>::`-equivalent qualified-self type, can take in this
/// crate's own source.
fn type_names_probed_op(
    ty: &syn::Type,
    self_is_probed_op: bool,
    aliases: &HashSet<String>,
) -> bool {
    let syn::Type::Path(p) = ty else {
        return false;
    };
    p.path
        .segments
        .last()
        .map(|s| owner_names_probed_op(&s.ident.to_string(), self_is_probed_op, aliases))
        .unwrap_or(false)
}

fn returns_probed_op(
    output: &syn::ReturnType,
    self_is_probed_op: bool,
    aliases: &HashSet<String>,
) -> bool {
    match output {
        syn::ReturnType::Type(_, ty) => type_names_probed_op(ty, self_is_probed_op, aliases),
        syn::ReturnType::Default => false,
    }
}

/// Whether a call expression's own `func` — `qself`/`path`, exactly the
/// two fields `syn::ExprPath` carries — is a `ProbedOp::new`-equivalent
/// call, under ANY qualifying prefix. A qualified-self call
/// (`<ProbedOp>::new(...)`) parses with the type in `qself.ty` and ONLY
/// `new` left in `path`'s own segments — checked first, since a bare
/// `path.segments.last()` read would see `new` alone and never the type
/// at all. Otherwise, the owning type is `path`'s own SECOND-TO-LAST
/// segment, regardless of how many segments come before it
/// (`ProbedOp::new`, `crate::admission::ProbedOp::new`, `Self::new`
/// inside `impl ProbedOp`, `Alias::new`) — never an exact, fixed-length
/// segment-VECTOR equality, which matches only the two-segment
/// unqualified shape.
fn is_probed_op_new_call(
    qself: &Option<syn::QSelf>,
    path: &syn::Path,
    self_is_probed_op: bool,
    aliases: &HashSet<String>,
) -> bool {
    if path
        .segments
        .last()
        .map(|s| s.ident != "new")
        .unwrap_or(true)
    {
        return false;
    }
    if let Some(qself) = qself {
        return type_names_probed_op(&qself.ty, self_is_probed_op, aliases);
    }
    path.segments
        .iter()
        .rev()
        .nth(1)
        .map(|s| owner_names_probed_op(&s.ident.to_string(), self_is_probed_op, aliases))
        .unwrap_or(false)
}

/// `path`'s own last segment is `transmute` — any prefix (`mem::`,
/// `std::mem::`, `core::mem::`, or a bare `use`-imported name) — AND its
/// own turbofish explicitly names `ProbedOp`/a resolved alias as either
/// type argument. See this file's own module doc, direction 7, for what
/// this does NOT close: a `transmute` whose target type is established
/// any other way.
fn transmute_turbofish_names_probed_op(path: &syn::Path, aliases: &HashSet<String>) -> bool {
    let Some(last) = path.segments.last() else {
        return false;
    };
    if last.ident != "transmute" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return false;
    };
    args.args.iter().any(|arg| {
        matches!(
            arg,
            syn::GenericArgument::Type(ty) if type_names_probed_op(ty, false, aliases)
        )
    })
}

/// Peels a single-tail-expression `unsafe { ... }`/`{ ... }` block down to
/// its own inner expression, repeatedly — the shape a `let x: ProbedOp =
/// unsafe { transmute(...) };` binding's own initializer takes.
fn unwrap_block_tail(mut expr: &syn::Expr) -> &syn::Expr {
    loop {
        expr = match expr {
            syn::Expr::Unsafe(u) => match u.block.stmts.as_slice() {
                [syn::Stmt::Expr(inner, None)] => inner,
                _ => return expr,
            },
            syn::Expr::Block(b) => match b.block.stmts.as_slice() {
                [syn::Stmt::Expr(inner, None)] => inner,
                _ => return expr,
            },
            _ => return expr,
        };
    }
}

/// `expr`, once unwrapped through any enclosing `unsafe`/plain block, is a
/// call whose own callee's last path segment is `transmute` — checked
/// WITHOUT requiring a turbofish (the shape a `let`-binding's own type
/// annotation, checked separately by the caller, supplies the target type
/// for instead).
fn is_bare_transmute_call(expr: &syn::Expr) -> bool {
    let syn::Expr::Call(call) = unwrap_block_tail(expr) else {
        return false;
    };
    let syn::Expr::Path(p) = &*call.func else {
        return false;
    };
    p.path
        .segments
        .last()
        .map(|s| s.ident == "transmute")
        .unwrap_or(false)
}

/// `expr`, if a bare string-literal expression — the shape every real
/// `ProbedOp::new(...)` call's own first (`report_key`) argument takes.
/// `None` for anything else (a variable, a `const`, a computed
/// expression) — direction 1's own name-keyed check reports such a call
/// site as unreviewed rather than silently skipping it (see this file's
/// module doc).
fn expr_as_str_literal(expr: &syn::Expr) -> Option<String> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(s),
        ..
    }) = expr
    {
        Some(s.value())
    } else {
        None
    }
}

/// An `Ident` token spelled `ProbedOp` (or a resolved alias) appears
/// ANYWHERE in `ts`, recursively through every `Group` — the shape both
/// direction 5 (macro invocations) and direction 6 (macro_rules! bodies)
/// share, so the two never drift out of sync on what counts as a hit.
/// Matches by EXACT `Ident` token, never a rendered-string
/// `.contains(...)` substring check — `ProbedOpKind` is one `Ident`
/// token, never equal to `ProbedOp`, so it can never false-positive here
/// regardless of how it is spaced/rendered.
fn probed_op_ident_in_tokens(ts: proc_macro2::TokenStream, aliases: &HashSet<String>) -> bool {
    for tt in ts {
        match tt {
            proc_macro2::TokenTree::Ident(id) => {
                let name = id.to_string();
                if name == "ProbedOp" || aliases.contains(&name) {
                    return true;
                }
            }
            proc_macro2::TokenTree::Group(g) => {
                if probed_op_ident_in_tokens(g.stream(), aliases) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// First pass over the SAME file universe [`scan`] walks: every
/// same-crate `type Alias = ProbedOp;` declaration, single-level (an
/// alias of an alias is not chased further — none exists in this crate
/// today; see this file's module doc, direction 4).
#[derive(Default)]
struct AliasCollector {
    aliases: HashSet<String>,
}

impl<'ast> Visit<'ast> for AliasCollector {
    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        if type_names_probed_op(&node.ty, false, &HashSet::new()) {
            self.aliases.insert(node.ident.to_string());
        }
        syn::visit::visit_item_type(self, node);
    }
}

/// One `ProbedOp::new(...)`-equivalent call site.
#[derive(Debug, Clone)]
struct DirectCallSite {
    file: String,
    function: String,
    /// The call's own first-argument string literal — this row's
    /// `report_key`, when statically resolvable. `None` when the first
    /// argument is some other expression shape; the main test reports
    /// this as unreviewed rather than silently excluding it from the
    /// name-keyed comparison.
    key_literal: Option<String>,
}

#[derive(Default)]
struct ProbedOpValueScanner {
    file: String,
    /// Same-crate `type Alias = ProbedOp;` names, resolved once by
    /// [`AliasCollector`] before the main scan runs and held constant
    /// across every file this scanner visits.
    aliases: HashSet<String>,
    /// Names of `impl` blocks currently open whose own `Self` type is
    /// `ProbedOp` (or a resolved alias of it) — a stack (never just a
    /// bool) because `impl` blocks nest inside modules, never inside EACH
    /// OTHER, but tracking a stack costs nothing and is the honest shape
    /// for a recursive visit.
    self_is_probed_op_stack: Vec<bool>,
    /// The innermost enclosing `fn`'s bare name, for every construction
    /// site/return-type match found — a stack for the same reason.
    fn_name_stack: Vec<String>,

    /// `ProbedOp::new(...)`-equivalent call sites — name-keyed against
    /// `PROBED_OPS`'s own `report_key`s (see this file's module doc for
    /// why name, not bare count).
    direct_call_sites: Vec<DirectCallSite>,
    /// `(file, enclosing fn)` for every `ProbedOp { ... }`-equivalent
    /// struct literal.
    struct_literal_sites: Vec<(String, String)>,
    /// `(file, fn name)` for every fn whose own return type names
    /// `ProbedOp`-equivalent.
    fns_returning_probed_op: Vec<(String, String)>,
    /// `macro_rules!` items whose own token-stream body contains a
    /// `ProbedOp`-equivalent `Ident`, by name.
    macro_bodies_constructing: Vec<String>,
    /// `(file, enclosing fn, macro name)` for every macro INVOCATION
    /// (never a `macro_rules!` definition — see direction 5's own doc)
    /// whose own token stream contains a `ProbedOp`-equivalent `Ident`.
    macro_invocation_sites: Vec<(String, String, String)>,
    /// `(file, enclosing fn)`-style descriptors for every `transmute`
    /// call/`let`-binding this scanner classified as
    /// `ProbedOp`-targeting — see direction 7's own doc for exactly what
    /// this does and does not close.
    transmute_sites: Vec<String>,
}

impl ProbedOpValueScanner {
    fn current_fn(&self) -> String {
        self.fn_name_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "<module level>".to_string())
    }

    fn self_is_probed_op(&self) -> bool {
        self.self_is_probed_op_stack
            .last()
            .copied()
            .unwrap_or(false)
    }
}

impl<'ast> Visit<'ast> for ProbedOpValueScanner {
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let is_probed_op = type_names_probed_op(&node.self_ty, false, &self.aliases);
        self.self_is_probed_op_stack.push(is_probed_op);
        syn::visit::visit_item_impl(self, node);
        self.self_is_probed_op_stack.pop();
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let name = node.sig.ident.to_string();
        if returns_probed_op(&node.sig.output, self.self_is_probed_op(), &self.aliases) {
            self.fns_returning_probed_op
                .push((self.file.clone(), name.clone()));
        }
        self.fn_name_stack.push(name);
        syn::visit::visit_item_fn(self, node);
        self.fn_name_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let name = node.sig.ident.to_string();
        if returns_probed_op(&node.sig.output, self.self_is_probed_op(), &self.aliases) {
            self.fns_returning_probed_op
                .push((self.file.clone(), name.clone()));
        }
        self.fn_name_stack.push(name);
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_name_stack.pop();
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let syn::Pat::Type(pt) = &node.pat {
            if type_names_probed_op(&pt.ty, self.self_is_probed_op(), &self.aliases) {
                if let Some(init) = &node.init {
                    if is_bare_transmute_call(&init.expr) {
                        self.transmute_sites.push(format!(
                            "{}::{} (let-binding-inferred transmute)",
                            self.file,
                            self.current_fn()
                        ));
                    }
                }
            }
        }
        syn::visit::visit_local(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*node.func {
            if is_probed_op_new_call(&p.qself, &p.path, self.self_is_probed_op(), &self.aliases) {
                let key_literal = node.args.first().and_then(expr_as_str_literal);
                self.direct_call_sites.push(DirectCallSite {
                    file: self.file.clone(),
                    function: self.current_fn(),
                    key_literal,
                });
            } else if p.qself.is_none()
                && transmute_turbofish_names_probed_op(&p.path, &self.aliases)
            {
                self.transmute_sites.push(format!(
                    "{}::{} (turbofish transmute)",
                    self.file,
                    self.current_fn()
                ));
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        let names = if let Some(qself) = &node.qself {
            type_names_probed_op(&qself.ty, self.self_is_probed_op(), &self.aliases)
        } else {
            node.path
                .segments
                .last()
                .map(|s| {
                    owner_names_probed_op(
                        &s.ident.to_string(),
                        self.self_is_probed_op(),
                        &self.aliases,
                    )
                })
                .unwrap_or(false)
        };
        if names {
            self.struct_literal_sites
                .push((self.file.clone(), self.current_fn()));
        }
        syn::visit::visit_expr_struct(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if let Some(ident) = &node.ident {
            if probed_op_ident_in_tokens(node.mac.tokens.clone(), &self.aliases) {
                self.macro_bodies_constructing.push(ident.to_string());
            }
        }
        syn::visit::visit_item_macro(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let name = node.path.segments.last().map(|s| s.ident.to_string());
        // `macro_rules!` DEFINITIONS are reviewed separately by
        // `visit_item_macro` above (direction 6) — excluded here so a
        // legitimate fixture macro's own body is never double-reported
        // under both directions.
        if name.as_deref() != Some("macro_rules")
            && probed_op_ident_in_tokens(node.tokens.clone(), &self.aliases)
        {
            self.macro_invocation_sites.push((
                self.file.clone(),
                self.current_fn(),
                name.unwrap_or_default(),
            ));
        }
        syn::visit::visit_macro(self, node);
    }
}

fn scan() -> ProbedOpValueScanner {
    let root = repo_root();
    // The STATED universe: every same-crate forgery vector named in this
    // file's own module doc could live under either tree — `src/**`
    // (production code) or `tests/**` (an integration test, the exact
    // shape a cross-crate probe used, and the shape a same-crate one
    // could equally take).
    let mut files = tracked_rs_files(&root, "crates/jammi-kernels/src");
    files.extend(tracked_rs_files(&root, "crates/jammi-kernels/tests"));
    assert!(
        files.len() > 20,
        "git ls-files returned suspiciously few files ({}) under crates/jammi-kernels/src and \
         crates/jammi-kernels/tests; the scan's own quantifier is likely broken (a bad cwd, a \
         moved directory)",
        files.len()
    );

    let parsed: Vec<(String, syn::File)> = files
        .iter()
        .map(|file| {
            let full = root.join(file);
            let text =
                std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("read {full:?}: {e}"));
            let ast: syn::File =
                syn::parse_file(&text).unwrap_or_else(|e| panic!("syn::parse_file {file}: {e}"));
            (file.clone(), ast)
        })
        .collect();

    // Pass 1: resolve every same-crate `type Alias = ProbedOp;` BEFORE the
    // main scan runs, so an alias used textually earlier in the same file
    // it is declared in (or in a DIFFERENT file entirely) is still seen —
    // never dependent on declaration order.
    let mut alias_collector = AliasCollector::default();
    for (_, ast) in &parsed {
        alias_collector.visit_file(ast);
    }

    // Pass 2: the full scan, with the resolved alias set held constant.
    let mut scanner = ProbedOpValueScanner {
        aliases: alias_collector.aliases,
        ..Default::default()
    };
    for (file, ast) in &parsed {
        scanner.file = file.clone();
        scanner.visit_file(ast);
    }
    scanner
}

#[test]
fn every_probed_op_construction_site_is_reviewed() {
    let scanner = scan();

    // Direction 6: every macro_rules! body constructing a ProbedOp is one
    // of the two reviewed test-fixture macros.
    let mut unreviewed_macros: Vec<&String> = scanner
        .macro_bodies_constructing
        .iter()
        .filter(|name| !REVIEWED_TEST_FIXTURE_MACROS.contains(&name.as_str()))
        .collect();
    unreviewed_macros.sort();
    assert!(
        unreviewed_macros.is_empty(),
        "macro_rules! definition(s) constructing a ProbedOp with no review entry in \
         REVIEWED_TEST_FIXTURE_MACROS: {unreviewed_macros:?}"
    );

    // Direction 5: no macro INVOCATION anywhere in the scanned universe
    // has a ProbedOp-equivalent Ident in its own token stream today —
    // every real one is empty, so any hit at all is unreviewed and named
    // directly (never a count comparison, since the reviewed set is the
    // empty set).
    assert!(
        scanner.macro_invocation_sites.is_empty(),
        "macro invocation(s) whose own token stream names ProbedOp/a resolved alias, with no \
         review entry (the reviewed set is empty — no legitimate call constructs a ProbedOp \
         through a macro invocation today): {:?}",
        scanner.macro_invocation_sites
    );

    // Direction 1 (name-keyed, never bare count — see this file's module
    // doc for why): every direct ProbedOp::new(...)-equivalent call site
    // must carry a first-argument string literal, and the SET of literals
    // found must match PROBED_OPS's own report_keys exactly, with every
    // deviation individually named.
    // Reads `file`/`function` directly (not merely through the derived
    // `Debug` impl, which rustc's dead-code analysis does not count) — the
    // `(file, fn)` descriptor is itself the reviewable identity of an
    // unnamed call site.
    let unnamed: Vec<String> = scanner
        .direct_call_sites
        .iter()
        .filter(|s| s.key_literal.is_none())
        .map(|s| format!("{}::{}", s.file, s.function))
        .collect();
    assert!(
        unnamed.is_empty(),
        "direct ProbedOp::new(...)-equivalent call site(s) whose own first argument is not a \
         string literal, so this oracle cannot read back which report_key they build (reported \
         rather than silently excluded from the name-keyed comparison below): {unnamed:?}"
    );

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for site in &scanner.direct_call_sites {
        *counts
            .entry(site.key_literal.clone().expect("checked above"))
            .or_default() += 1;
    }
    let found: BTreeSet<String> = counts.keys().cloned().collect();
    let expected: BTreeSet<String> = jammi_kernels::admission::PROBED_OPS
        .iter()
        .map(|op| op.report_key().to_string())
        .collect();

    let extra: Vec<&String> = found.difference(&expected).collect();
    assert!(
        extra.is_empty(),
        "direct ProbedOp::new(...)-equivalent call site(s) whose first-argument literal names a \
         report_key NOT present in PROBED_OPS — forged or orphaned: {extra:?}"
    );
    let missing: Vec<&String> = expected.difference(&found).collect();
    assert!(
        missing.is_empty(),
        "PROBED_OPS row(s) whose own report_key was never found among direct \
         ProbedOp::new(...)-equivalent call sites: {missing:?}"
    );
    let duplicated: Vec<(&String, &usize)> = counts.iter().filter(|(_, &n)| n > 1).collect();
    assert!(
        duplicated.is_empty(),
        "report_key(s) constructed at MORE THAN ONE direct call site — each PROBED_OPS row must \
         be built exactly once: {duplicated:?}"
    );
    assert!(
        scanner.direct_call_sites.len() >= jammi_kernels::admission::PROBED_OPS.len(),
        "found fewer direct call sites ({}) than PROBED_OPS rows ({}) — asserted directly as a \
         floor rather than trusted as an implicit corollary of the checks above",
        scanner.direct_call_sites.len(),
        jammi_kernels::admission::PROBED_OPS.len(),
    );

    // Direction 2 (name-keyed): every ProbedOp { ... }-equivalent
    // struct-literal expression is the ONE reviewed site — ProbedOp::new's
    // own body. This is exactly the shape an earlier revision of this
    // oracle could not see at all (it counted only ProbedOp::new calls): a
    // same-crate `pub const X: ProbedOp = ProbedOp { ..., _sealed: Sealed
    // }` literal never goes through `new` and was invisible to that count.
    let unreviewed_literals: Vec<&(String, String)> = scanner
        .struct_literal_sites
        .iter()
        .filter(|(file, func)| (file.as_str(), func.as_str()) != REVIEWED_CONSTRUCTOR)
        .collect();
    assert!(
        unreviewed_literals.is_empty(),
        "ProbedOp {{ ... }}-equivalent struct-literal construction site(s) outside the one \
         reviewed constructor {REVIEWED_CONSTRUCTOR:?}: {unreviewed_literals:?}"
    );
    assert_eq!(
        scanner
            .struct_literal_sites
            .iter()
            .filter(|site| (site.0.as_str(), site.1.as_str()) == REVIEWED_CONSTRUCTOR)
            .count(),
        1,
        "the one reviewed constructor must contain EXACTLY one ProbedOp {{ ... }} literal — \
         zero would mean this oracle's own premise (ProbedOp::new builds one) drifted, more \
         than one is not a shape ProbedOp::new's own body has today"
    );

    // Direction 3 (name-keyed): every fn whose own return type names
    // ProbedOp-equivalent (directly, Self inside impl ProbedOp, or a
    // resolved alias) is the ONE reviewed constructor — a new fn
    // returning a ProbedOp by some OTHER construction path this scanner
    // cannot classify at the body level is caught here instead, at the
    // coarser signature grain.
    let unreviewed_fns: Vec<&(String, String)> = scanner
        .fns_returning_probed_op
        .iter()
        .filter(|(file, func)| (file.as_str(), func.as_str()) != REVIEWED_CONSTRUCTOR)
        .collect();
    assert!(
        unreviewed_fns.is_empty(),
        "fn(s) whose own return type names ProbedOp-equivalent outside the one reviewed \
         constructor {REVIEWED_CONSTRUCTOR:?}: {unreviewed_fns:?}"
    );

    // Direction 7: no transmute anywhere in the scanned universe targets
    // ProbedOp today — see this file's module doc for the residual this
    // direction does NOT close.
    assert!(
        scanner.transmute_sites.is_empty(),
        "transmute call(s)/let-binding(s) this oracle classified as ProbedOp-targeting, with no \
         review entry: {:?}",
        scanner.transmute_sites
    );
}

/// Mutation oracle for direction 1's own name-keyed premise (executed
/// directly against a synthetic file tree during this fix's own
/// development, reproduced here as a standing fixture-level proof rather
/// than a real-tree mutation, since mutating the crate's own tracked
/// source from inside a committed test would leave a real forged row on
/// disk between the mutation and its revert): a fixture tree carrying an
/// EXTRA, forged `ProbedOp::new(...)` call is found and its own key
/// literal recorded — the main test's set-difference check is what
/// classifies it as unreviewed; this fixture proves the scanner itself
/// finds it at all.
#[test]
fn falsification_an_extra_direct_call_site_moves_the_discovered_count() {
    let dir = std::env::temp_dir().join(format!(
        "probed_op_construction_sites_falsification_call_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fake.rs");
    std::fs::write(
        &path,
        "const A: ProbedOp = ProbedOp::new(\"a\", ProbedOpKind::TwoArm, &[], f);\n\
         const B: ProbedOp = ProbedOp::new(\"b\", ProbedOpKind::TwoArm, &[], f);\n\
         // A THIRD, forged construction site not present in any real PROBED_OPS row.\n\
         const FORGED: ProbedOp = ProbedOp::new(\"forged\", ProbedOpKind::TwoArm, &[], f);\n",
    )
    .unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let parsed: syn::File = syn::parse_file(&text).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);

    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(
        scanner.direct_call_sites.len(),
        3,
        "the fixture's own THREE direct call sites must all be found — a scanner that only \
         finds the first two would silently pass a count check pinned at 2, masking exactly \
         the forged-row class this oracle exists to catch"
    );
    let keys: BTreeSet<String> = scanner
        .direct_call_sites
        .iter()
        .filter_map(|s| s.key_literal.clone())
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from(["a".to_string(), "b".to_string(), "forged".to_string()]),
        "each call site's own first-argument literal must be read back correctly — the \
         name-keyed comparison in the real oracle above depends on this"
    );
}

/// Mutation oracle for direction 2 — the exact gap the audit named: a
/// same-crate `pub const X: ProbedOp = ProbedOp {{ ..., _sealed: Sealed }}`
/// struct literal, never routed through `ProbedOp::new` at all, was
/// invisible to the earlier, `ProbedOp::new`-only scan (`direct_call_sites`
/// alone would stay unchanged — no `ExprCall` exists in this fixture at
/// all). The struct-literal scanner must find it and classify it as
/// UNREVIEWED (its own enclosing fn, `sneaky`, is not `REVIEWED_CONSTRUCTOR`).
#[test]
fn falsification_a_same_crate_struct_literal_outside_new_is_found_and_unreviewed() {
    let dir = std::env::temp_dir().join(format!(
        "probed_op_construction_sites_falsification_literal_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fake.rs");
    std::fs::write(
        &path,
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_a_same_crate_struct_literal_outside_new_is_found_and_unreviewed — not real code in this file
        "fn sneaky() -> ProbedOp {\n\
         \x20\x20\x20\x20ProbedOp {\n\
         \x20\x20\x20\x20\x20\x20\x20\x20report_key: \"forged\",\n\
         \x20\x20\x20\x20\x20\x20\x20\x20kind: ProbedOpKind::TwoArm,\n\
         \x20\x20\x20\x20\x20\x20\x20\x20registry: &[],\n\
         \x20\x20\x20\x20\x20\x20\x20\x20dry_run: f,\n\
         \x20\x20\x20\x20\x20\x20\x20\x20_sealed: Sealed,\n\
         \x20\x20\x20\x20}\n\
         }\n",
    )
    .unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let parsed: syn::File = syn::parse_file(&text).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);

    std::fs::remove_dir_all(&dir).ok();

    // The EXACT regression this direction closes: zero ExprCall sites
    // (no ProbedOp::new anywhere in the fixture), so the OLD, call-only
    // scanner would have found NOTHING here at all.
    assert_eq!(scanner.direct_call_sites.len(), 0);
    assert_eq!(
        scanner.struct_literal_sites,
        vec![("fake.rs".to_string(), "sneaky".to_string())]
    );
    assert_eq!(
        scanner.fns_returning_probed_op,
        vec![("fake.rs".to_string(), "sneaky".to_string())]
    );
    assert_ne!(
        ("fake.rs".to_string(), "sneaky".to_string()),
        (
            REVIEWED_CONSTRUCTOR.0.to_string(),
            REVIEWED_CONSTRUCTOR.1.to_string()
        ),
        "sanity: the fixture's own (file, fn) must NOT match REVIEWED_CONSTRUCTOR — proving the \
         real oracle's direction-2/3 checks above genuinely fire on this shape rather than \
         vacuously matching"
    );
}

/// Auditor bypass 1/5 — a fully qualified path
/// (`crate::admission::ProbedOp::new`, here shortened to a two-module
/// prefix over the SAME shape since the fixture is single-file): the
/// exact shape an exact-segment-vector-equality check (`segments ==
/// ["ProbedOp", "new"]`) misses, since a 3+-segment path is never equal
/// to a 2-element vector.
#[test]
fn falsification_qualified_path_call_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_qualified_path_call_is_found — not real code in this file
    let src = "fn f() { let _ = some::module::path::ProbedOp::new(\"x\", ProbedOpKind::TwoArm, &[], g); }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.direct_call_sites.len(), 1);
    assert_eq!(
        scanner.direct_call_sites[0].key_literal.as_deref(),
        Some("x")
    );
}

/// Auditor bypass 2/5 — qualified-self syntax (`<ProbedOp>::new(...)`): a
/// DIFFERENT `syn::Expr::Path` shape entirely (the type lives in the
/// expression's own `qself` field, and `path` carries only `new`), so a
/// check that only ever reads `path`'s own segments — never `qself` —
/// misses it completely regardless of how the segment-matching itself is
/// written.
#[test]
fn falsification_qualified_self_type_call_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_qualified_self_type_call_is_found — not real code in this file
    let src = "fn f() { let _ = <ProbedOp>::new(\"x\", ProbedOpKind::TwoArm, &[], g); }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.direct_call_sites.len(), 1);
    assert_eq!(
        scanner.direct_call_sites[0].key_literal.as_deref(),
        Some("x")
    );
}

/// Auditor bypass 3/5 — a same-crate type alias (`type PO = ProbedOp; PO
/// ::new(...)`): the exact shape [`AliasCollector`]'s own first pass
/// exists to resolve.
#[test]
fn falsification_type_alias_call_is_found() {
    let src =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_type_alias_call_is_found — not real code in this file
        "type PO = ProbedOp; fn f() { let _ = PO::new(\"x\", ProbedOpKind::TwoArm, &[], g); }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut alias_collector = AliasCollector::default();
    alias_collector.visit_file(&parsed);
    assert_eq!(
        alias_collector.aliases,
        HashSet::from(["PO".to_string()]),
        "sanity: the fixture's own alias must be resolved before the main scan runs"
    );
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        aliases: alias_collector.aliases,
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.direct_call_sites.len(), 1);
    assert_eq!(
        scanner.direct_call_sites[0].key_literal.as_deref(),
        Some("x")
    );
}

/// Auditor bypass 4/5 — a `ProbedOp::new(...)` call wrapped inside a
/// macro INVOCATION (`vec![ProbedOp::new(...)]`): opaque to direction 1's
/// typed `ExprCall` traversal (a macro's own arguments are an unparsed
/// token stream, never an `Expr` node syn descends into), found only by
/// direction 5's own token-stream walk.
#[test]
fn falsification_macro_invocation_wrapped_new_call_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_macro_invocation_wrapped_new_call_is_found — not real code in this file
    let src = "fn f() { let _ = vec![ProbedOp::new(\"x\", ProbedOpKind::TwoArm, &[], g)]; }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.direct_call_sites.len(), 0, "opaque to direction 1");
    assert_eq!(
        scanner.macro_invocation_sites,
        vec![("fake.rs".to_string(), "f".to_string(), "vec".to_string())]
    );
}

/// Auditor bypass 5/5 — a raw `ProbedOp { .. }` struct literal wrapped
/// inside a macro invocation (`vec![ProbedOp { .., _sealed: Sealed }]`):
/// opaque to BOTH direction 1 (no call at all) and direction 2 (no typed
/// `ExprStruct` node — the whole literal is inside an unparsed macro
/// argument token stream), found only by direction 5.
#[test]
fn falsification_macro_invocation_wrapped_struct_literal_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_macro_invocation_wrapped_struct_literal_is_found — not real code in this file
    let src = "fn f() { let _ = vec![ProbedOp { report_key: \"x\", kind: ProbedOpKind::TwoArm, \
               registry: &[], dry_run: g, _sealed: Sealed }]; }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(
        scanner.struct_literal_sites.len(),
        0,
        "opaque to direction 2"
    );
    assert_eq!(
        scanner.macro_invocation_sites,
        vec![("fake.rs".to_string(), "f".to_string(), "vec".to_string())]
    );
}

/// Control/GREEN — the exact false-positive shape a rendered-string
/// `.contains("ProbedOp")` substring check (rather than an exact `Ident`
/// token match) would trip on: `ProbedOpKind` mentioned inside
/// `matches!`/`assert!` macro invocations, plus an unrelated `vec!` with
/// no `ProbedOp` involvement at all. Neither is a construction site;
/// direction 5 must find nothing here.
#[test]
fn falsification_unrelated_macro_invocation_is_not_flagged() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_unrelated_macro_invocation_is_not_flagged — not real code in this file
    let src = "fn f(k: ProbedOpKind) -> bool {\n\
               \x20\x20\x20\x20let _ = vec![1, 2, 3];\n\
               \x20\x20\x20\x20assert!(matches!(k, ProbedOpKind::TwoArm));\n\
               \x20\x20\x20\x20matches!(k, ProbedOpKind::Cascade)\n\
               }\n";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert!(
        scanner.macro_invocation_sites.is_empty(),
        "ProbedOpKind is a DIFFERENT Ident token from ProbedOp and must never false-positive: \
         {:?}",
        scanner.macro_invocation_sites
    );
    assert!(scanner.direct_call_sites.is_empty());
    assert!(scanner.struct_literal_sites.is_empty());
}

/// `Self::new` inside `impl ProbedOp` — not one of the auditor's named six
/// fixtures, but the same root-cause path-matching fix this round makes,
/// exercised directly: an exact `segments == ["ProbedOp", "new"]` check
/// (or one that only compares by NAME without ALSO tracking whether the
/// enclosing `impl`'s own `Self` type is `ProbedOp`) never matches
/// `Self::new` at all.
#[test]
fn falsification_self_new_inside_impl_probed_op_is_found() {
    let src =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_self_new_inside_impl_probed_op_is_found — not real code in this file
        "impl ProbedOp { fn make() -> Self { Self::new(\"x\", ProbedOpKind::TwoArm, &[], g) } }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.direct_call_sites.len(), 1);
    assert_eq!(
        scanner.direct_call_sites[0].key_literal.as_deref(),
        Some("x")
    );
}

/// Direction 7's own RED: a turbofish `transmute` explicitly naming
/// `ProbedOp` is found.
#[test]
fn falsification_transmute_turbofish_targeting_probed_op_is_found() {
    // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_transmute_turbofish_targeting_probed_op_is_found — not real code in this file
    let src = "fn f(bytes: [u8; 64]) -> ProbedOp { unsafe { std::mem::transmute::<[u8; 64], ProbedOp>(bytes) } }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.transmute_sites.len(), 1);
}

/// Direction 7's own RED, the `let`-binding-inferred shape (no turbofish
/// at all — the target type comes from the binding's own annotation).
#[test]
fn falsification_transmute_let_binding_inferred_targeting_probed_op_is_found() {
    let src =
        // kernel-oracles: fn-in-literal reviewed: synthetic source fixture for falsification_transmute_let_binding_inferred_targeting_probed_op_is_found — not real code in this file
        "fn f(bytes: [u8; 64]) { let op: ProbedOp = unsafe { std::mem::transmute(bytes) }; }";
    let parsed: syn::File = syn::parse_file(src).unwrap();
    let mut scanner = ProbedOpValueScanner {
        file: "fake.rs".to_string(),
        ..Default::default()
    };
    scanner.visit_file(&parsed);
    assert_eq!(scanner.transmute_sites.len(), 1);
}
