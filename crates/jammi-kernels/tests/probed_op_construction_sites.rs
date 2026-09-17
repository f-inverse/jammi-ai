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
//!   1. `ProbedOp::new(...)` calls (`syn::ExprCall`) — count-keyed
//!      against the REAL, linked-in [`PROBED_OPS`] constant.
//!   2. `ProbedOp { ... }` struct-literal expressions (`syn::ExprStruct`
//!      whose path's last segment is `ProbedOp`) — name-keyed against ONE
//!      reviewed site, `ProbedOp::new`'s own constructor body (the
//!      canonical construction this whole file exists to make the ONLY
//!      one). A same-crate `pub const X: ProbedOp = ProbedOp { ...,
//!      _sealed: Sealed }` literal ANYWHERE ELSE — invisible to an
//!      earlier revision of this oracle, which counted only
//!      `ProbedOp::new` calls — REDs here.
//!   3. Any `fn` (free or `impl` method) whose own return type names
//!      `ProbedOp` (directly, or as `Self` inside `impl ProbedOp`) —
//!      name-keyed against the SAME one reviewed site. A new fn
//!      returning a `ProbedOp` by some OTHER construction path (built
//!      from a mutated copy, forwarded from another crate through an
//!      FFI-shaped boundary, anything syn cannot see the BODY of well
//!      enough to classify directly) is caught at this coarser,
//!      signature-level grain instead.
//!
//! **Why count-keyed for (1) but name-keyed for (2) and (3).**
//! `PROBED_OPS`'s own array holds VALUES at runtime, not the Rust const
//! IDENTIFIERS (`LAYER_NORM`, …) that produced them — there is no way to
//! read "the const was named `LAYER_NORM`" back out of a `ProbedOp`
//! value, so (1) can only be proved by COUNT. (2) and (3) both have
//! EXACTLY one legitimate site each (`ProbedOp::new`'s own body, and
//! `ProbedOp::new` itself, respectively) — a fixed, tiny, by-NAME
//! reviewed set is the more precise check available for a set of size
//! one, and reads better in a failure message than "1 != 1, somewhere".
//!
//! **Why `macro_rules!` bodies are handled separately.** A `macro_rules!`
//! definition's own body is an opaque token stream to a normal
//! `syn::visit::Visit` traversal (its pattern/template syntax, e.g.
//! `($key:expr) => { ... };`, is not valid standalone Rust expression
//! syntax syn can parse as an `Expr`) — none of the three visitors above
//! structurally descend into one. `test_two_arm!`/`test_cascade!` each
//! contain exactly one `ProbedOp::new(...)` call in their own body; this
//! scanner finds that by name (`visit_item_macro`, checking each
//! macro_rules! item's own token-stream text for either construction
//! shape), reviewed as a SEPARATE category, never folded into (1)/(2) (a
//! `test_two_arm!` INVOCATION, e.g. `test_two_arm!("test_op")`, is a
//! macro invocation — `syn::Expr::Macro`, not `syn::Expr::Call` — so it
//! is invisible to the call-site visitor regardless of how many times the
//! fixture macros are invoked across this crate's own tests; only the two
//! macro DEFINITIONS are counted, once each).

use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;

/// `macro_rules!` definitions this crate reviews as legitimate
/// `ProbedOp`-constructing TEST fixtures — `crates/jammi-kernels/src/
/// admission.rs`'s own `test_two_arm`/`test_cascade` macros, used
/// exclusively inside its `#[cfg(test)] mod tests`. A NEW macro_rules!
/// definition anywhere in the scanned universe whose own body contains
/// either construction shape (token-stream-stringified: `ProbedOp ::
/// new`, or a struct literal opening `ProbedOp {`) and is NOT one of
/// these two names REDs this oracle's macro direction.
const REVIEWED_TEST_FIXTURE_MACROS: &[&str] = &["test_two_arm", "test_cascade"];

/// The ONE reviewed (file, fn name) pair allowed to contain a
/// `ProbedOp { ... }` struct literal, or to have a return type naming
/// `ProbedOp`/`Self` (inside `impl ProbedOp`) — `ProbedOp::new` itself,
/// the sole canonical constructor this whole file exists to keep sole.
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

/// `ty` names `ProbedOp` directly, or is the bare `Self` path while the
/// caller says the enclosing `impl` block's own `Self` type IS
/// `ProbedOp`. The only two shapes a `-> ProbedOp` return type can take
/// in this crate's own source.
fn type_names_probed_op(ty: &syn::Type, self_is_probed_op: bool) -> bool {
    let syn::Type::Path(p) = ty else {
        return false;
    };
    match p
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .as_deref()
    {
        Some("ProbedOp") => true,
        Some("Self") => self_is_probed_op,
        _ => false,
    }
}

fn returns_probed_op(output: &syn::ReturnType, self_is_probed_op: bool) -> bool {
    match output {
        syn::ReturnType::Type(_, ty) => type_names_probed_op(ty, self_is_probed_op),
        syn::ReturnType::Default => false,
    }
}

#[derive(Default)]
struct ProbedOpValueScanner {
    file: String,
    /// Names of `impl` blocks currently open whose own `Self` type is
    /// `ProbedOp` — a stack (never just a bool) because `impl` blocks
    /// nest inside modules, never inside EACH OTHER, but tracking a
    /// stack costs nothing and is the honest shape for a recursive visit.
    self_is_probed_op_stack: Vec<bool>,
    /// The innermost enclosing `fn`'s bare name, for every construction
    /// site/return-type match found — a stack for the same reason.
    fn_name_stack: Vec<String>,

    /// `ProbedOp::new(...)` call sites — count-keyed against `PROBED_OPS`.
    direct_call_sites: Vec<String>,
    /// `(file, enclosing fn)` for every `ProbedOp { ... }` struct literal.
    struct_literal_sites: Vec<(String, String)>,
    /// `(file, fn name)` for every fn whose own return type names
    /// `ProbedOp`.
    fns_returning_probed_op: Vec<(String, String)>,
    /// `macro_rules!` items whose own token-stream body contains either
    /// construction shape, by name.
    macro_bodies_constructing: Vec<String>,
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
        let is_probed_op = matches!(
            &*node.self_ty,
            syn::Type::Path(p) if p.path.segments.last().map(|s| s.ident == "ProbedOp").unwrap_or(false)
        );
        self.self_is_probed_op_stack.push(is_probed_op);
        syn::visit::visit_item_impl(self, node);
        self.self_is_probed_op_stack.pop();
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let name = node.sig.ident.to_string();
        if returns_probed_op(&node.sig.output, self.self_is_probed_op()) {
            self.fns_returning_probed_op
                .push((self.file.clone(), name.clone()));
        }
        self.fn_name_stack.push(name);
        syn::visit::visit_item_fn(self, node);
        self.fn_name_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let name = node.sig.ident.to_string();
        if returns_probed_op(&node.sig.output, self.self_is_probed_op()) {
            self.fns_returning_probed_op
                .push((self.file.clone(), name.clone()));
        }
        self.fn_name_stack.push(name);
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_name_stack.pop();
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*node.func {
            let segments: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments == ["ProbedOp", "new"] {
                self.direct_call_sites
                    .push(format!("{}::{}", self.file, self.current_fn()));
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast syn::ExprStruct) {
        if node
            .path
            .segments
            .last()
            .map(|s| s.ident == "ProbedOp")
            .unwrap_or(false)
        {
            self.struct_literal_sites
                .push((self.file.clone(), self.current_fn()));
        }
        syn::visit::visit_expr_struct(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if let Some(ident) = &node.ident {
            let body = node.mac.tokens.to_string();
            if body.contains("ProbedOp :: new") || body.contains("ProbedOp {") {
                self.macro_bodies_constructing.push(ident.to_string());
            }
        }
        syn::visit::visit_item_macro(self, node);
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
    let mut scanner = ProbedOpValueScanner::default();
    for file in &files {
        let full = root.join(file);
        let text = std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("read {full:?}: {e}"));
        let parsed: syn::File =
            syn::parse_file(&text).unwrap_or_else(|e| panic!("syn::parse_file {file}: {e}"));
        scanner.file = file.clone();
        scanner.visit_file(&parsed);
    }
    scanner
}

#[test]
fn every_probed_op_construction_site_is_reviewed() {
    let scanner = scan();

    // Direction 1: every macro_rules! body constructing a ProbedOp is one
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

    // Direction 2 (count-keyed): every DIRECT ProbedOp::new(...) call
    // site — which structurally excludes the two macro bodies above —
    // must equal exactly PROBED_OPS's own row count. See this file's own
    // module doc for why count, not name, is the right key here.
    assert_eq!(
        scanner.direct_call_sites.len(),
        jammi_kernels::admission::PROBED_OPS.len(),
        "the number of direct (non-test-macro) ProbedOp::new(...) call sites this crate's own \
         source contains ({}) must equal PROBED_OPS's own row count ({}) exactly — a mismatch \
         means a ProbedOp is constructed somewhere PROBED_OPS/dry_run_all/the eager-disable \
         sweep never sees it:\n  {}",
        scanner.direct_call_sites.len(),
        jammi_kernels::admission::PROBED_OPS.len(),
        scanner.direct_call_sites.join("\n  "),
    );

    // Direction 3 (name-keyed): every ProbedOp { ... } struct-literal
    // expression is the ONE reviewed site — ProbedOp::new's own body.
    // This is exactly the shape an earlier revision of this oracle could
    // not see at all (it counted only ProbedOp::new calls): a same-crate
    // `pub const X: ProbedOp = ProbedOp { ..., _sealed: Sealed }` literal
    // never goes through `new` and was invisible to that count.
    let unreviewed_literals: Vec<&(String, String)> = scanner
        .struct_literal_sites
        .iter()
        .filter(|(file, func)| (file.as_str(), func.as_str()) != REVIEWED_CONSTRUCTOR)
        .collect();
    assert!(
        unreviewed_literals.is_empty(),
        "ProbedOp {{ ... }} struct-literal construction site(s) outside the one reviewed \
         constructor {REVIEWED_CONSTRUCTOR:?}: {unreviewed_literals:?}"
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

    // Direction 4 (name-keyed): every fn whose own return type names
    // ProbedOp (or Self inside impl ProbedOp) is the ONE reviewed
    // constructor — a new fn returning a ProbedOp by some OTHER
    // construction path this scanner cannot classify at the body level
    // is caught here instead, at the coarser signature grain.
    let unreviewed_fns: Vec<&(String, String)> = scanner
        .fns_returning_probed_op
        .iter()
        .filter(|(file, func)| (file.as_str(), func.as_str()) != REVIEWED_CONSTRUCTOR)
        .collect();
    assert!(
        unreviewed_fns.is_empty(),
        "fn(s) whose own return type names ProbedOp outside the one reviewed constructor \
         {REVIEWED_CONSTRUCTOR:?}: {unreviewed_fns:?}"
    );
}

/// Mutation oracle for direction 2's own count-keyed premise (executed
/// directly against a synthetic file tree during this fix's own
/// development, reproduced here as a standing fixture-level proof rather
/// than a real-tree mutation, since mutating the crate's own tracked
/// source from inside a committed test would leave a real forged row on
/// disk between the mutation and its revert): a fixture tree carrying an
/// EXTRA `ProbedOp::new(...)` call moves the discovered count.
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
    assert_ne!(
        scanner.direct_call_sites.len(),
        2,
        "sanity: the fixture's REAL count (3) must differ from what a REVIEWED, two-row table \
         would expect (2) — proving the count comparison in the real oracle above is genuinely \
         load-bearing against exactly this shape of drift, not vacuously true"
    );
}

/// Mutation oracle for direction 3 — the exact gap the audit named: a
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
         real oracle's direction-3/4 checks above genuinely fire on this shape rather than \
         vacuously matching"
    );
}
