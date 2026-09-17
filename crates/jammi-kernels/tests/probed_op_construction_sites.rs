//! I2 adversarial-audit fix (wave-5 third round): [`jammi_kernels::admission::ProbedOp`]
//! is sealed against every crate but `jammi-kernels` itself
//! (`#[non_exhaustive]` plus a private field — `admission.rs`'s own doc
//! on `ProbedOp`/`Sealed` has the full argument, and the probe that
//! confirms an OUTSIDE forgery fails to compile: `E0639: cannot create
//! non-exhaustive struct using struct expression`, reproduced against
//! `crates/jammi-encoders/src/layer_norm.rs` while building this fix,
//! then reverted). Rust has no "seal from myself" mechanism, so a
//! same-crate forgery inside `jammi-kernels` itself is a residual
//! `#[non_exhaustive]`/private-field sealing cannot touch — this file
//! closes it mechanically instead: a real `syn` AST walk over
//! `crates/jammi-kernels/src`, proving every `ProbedOp::new(...)` call
//! this crate's own source contains is either one of [`PROBED_OPS`]'s own
//! rows (count-keyed against the REAL, linked-in [`PROBED_OPS`] constant —
//! never a hand-copied number) or lives inside one of the two named,
//! `#[cfg(test)]`-only fixture macros `admission.rs`'s own test module
//! defines (`test_two_arm`/`test_cascade`).
//!
//! **Why count-keyed, not name-keyed.** `PROBED_OPS`'s own array holds
//! VALUES at runtime, not the Rust const IDENTIFIERS (`LAYER_NORM`, …)
//! that produced them — there is no way to read "the const was named
//! `LAYER_NORM`" back out of a `ProbedOp` value. A forged row hidden
//! somewhere in this crate's source but never listed in `PROBED_OPS`'s
//! array moves the SYN-DISCOVERED direct-call-site count without moving
//! `PROBED_OPS.len()` — exactly the mismatch this oracle exists to catch,
//! without needing a name-to-value correspondence syn cannot give it.
//!
//! **Why macro_rules! bodies are handled separately.** A `macro_rules!`
//! definition's own body is an opaque token stream to a normal
//! `syn::visit::Visit` traversal (its pattern/template syntax, e.g.
//! `($key:expr) => { ... };`, is not valid standalone Rust expression
//! syntax syn can parse as an `Expr`) — `visit_expr_call` structurally
//! never descends into one. `test_two_arm!`/`test_cascade!` each contain
//! exactly one `ProbedOp::new(...)` call in their own body; this scanner
//! finds that by name (`visit_item_macro`, checking each macro_rules!
//! item's own token-stream text), reviewed as a SEPARATE category from
//! the direct-call count above, never folded into it (a `test_two_arm!`
//! INVOCATION, e.g. `test_two_arm!("test_op")`, is a macro invocation —
//! `syn::Expr::Macro`, not `syn::Expr::Call` — so it is invisible to
//! `visit_expr_call` regardless of how many times the fixture macros are
//! invoked across this crate's own tests; only the two macro
//! DEFINITIONS are counted, once each).

use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;

/// `macro_rules!` definitions this crate reviews as legitimate
/// `ProbedOp::new(...)` construction sites, TEST fixtures only —
/// `crates/jammi-kernels/src/admission.rs`'s own `test_two_arm`/
/// `test_cascade` macros, used exclusively inside its
/// `#[cfg(test)] mod tests`. A NEW macro_rules! definition anywhere in
/// this crate whose own body contains the text `ProbedOp :: new`
/// (token-stream-stringified, hence the spaces around `::`) and is NOT
/// one of these two names REDs this oracle's first direction.
const REVIEWED_TEST_FIXTURE_MACROS: &[&str] = &["test_two_arm", "test_cascade"];

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

#[derive(Default)]
struct ProbedOpNewScanner {
    /// Direct (non-macro-body) `ProbedOp::new(...)` call sites, one entry
    /// per site (the exact source text of the callee path, for a
    /// human-readable failure message only).
    direct_call_sites: Vec<String>,
    /// `macro_rules!` items whose own token-stream body contains
    /// `ProbedOp :: new`, by name.
    macro_bodies_constructing: Vec<String>,
}

impl<'ast> Visit<'ast> for ProbedOpNewScanner {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*node.func {
            let segments: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments == ["ProbedOp", "new"] {
                self.direct_call_sites.push(segments.join("::"));
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if let Some(ident) = &node.ident {
            if node.mac.tokens.to_string().contains("ProbedOp :: new") {
                self.macro_bodies_constructing.push(ident.to_string());
            }
        }
        syn::visit::visit_item_macro(self, node);
    }
}

fn scan() -> ProbedOpNewScanner {
    let root = repo_root();
    let files = tracked_rs_files(&root, "crates/jammi-kernels/src");
    assert!(
        files.len() > 20,
        "git ls-files returned suspiciously few files ({}) under crates/jammi-kernels/src; the \
         scan's own quantifier is likely broken (a bad cwd, a moved directory)",
        files.len()
    );
    let mut scanner = ProbedOpNewScanner::default();
    for file in &files {
        let full = root.join(file);
        let text = std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("read {full:?}: {e}"));
        let parsed: syn::File =
            syn::parse_file(&text).unwrap_or_else(|e| panic!("syn::parse_file {file}: {e}"));
        scanner.visit_file(&parsed);
    }
    scanner
}

#[test]
fn every_probed_op_construction_site_is_reviewed() {
    let scanner = scan();

    // Direction 1: every macro_rules! body constructing a ProbedOp is one
    // of the two reviewed test-fixture macros — an unreviewed THIRD macro
    // constructing one REDs here.
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

    // Direction 2 (count-keyed): every DIRECT ProbedOp::new(...) call site
    // — which structurally excludes the two macro bodies above, since
    // macro_rules! bodies are opaque token streams to a normal expr visit
    // — must equal exactly PROBED_OPS's own row count. See this file's
    // own module doc for why count, not name, is the right key here.
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
}

/// Mutation oracle for direction 2's own count-keyed premise (executed
/// directly against a synthetic file tree during this fix's own
/// development, reproduced here as a standing fixture-level proof rather
/// than a real-tree mutation, since mutating the crate's own tracked
/// source from inside a committed test would leave a real forged row on
/// disk between the mutation and its revert): a fixture tree carrying an
/// EXTRA `ProbedOp::new(...)` call beyond what `REVIEWED_TEST_FIXTURE_MACROS`
/// and a caller-supplied expected count allow must disagree, proving the
/// count comparison itself is load-bearing, not a tautology.
#[test]
fn falsification_an_extra_direct_call_site_moves_the_discovered_count() {
    let dir = std::env::temp_dir().join(format!(
        "probed_op_construction_sites_falsification_{}",
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
    let mut scanner = ProbedOpNewScanner::default();
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
