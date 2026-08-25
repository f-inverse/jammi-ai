//! Grep-based regression test for [`jammi_kernels::ops::StatefulKernelOp`]'s
//! own doc: candle's PUBLIC `Tensor::apply_op1_arc`/`apply_op2_arc`/
//! `apply_op3_arc` (`candle-core` 0.11.0 `custom_op.rs:216-234,236-243`),
//! `apply_op1_no_bwd`/`apply_op2_no_bwd`/`apply_op3_no_bwd`
//! (`custom_op.rs:159-176`), AND a bare `apply_op1`/`apply_op2`/`apply_op3`
//! call can all back a `CustomOpN` value with no `KernelOp`/
//! `StatefulKernelOp` bound at all — nothing in candle's own `CustomOpN`
//! bounds stops that. This crate's enforcement (`apply1`/`apply2`/`apply3`/
//! `apply_stateful1`/`apply_stateful3`, all in `ops/mod.rs`) is a
//! DISCIPLINE over call sites, not something the type system checks on its
//! own — the real hazard is a STATEFUL op (one holding a `Saved` field)
//! reaching `apply_op1`/`apply_op3` directly, bypassing
//! `apply_stateful1`/`apply_stateful3`'s gate (the `_arc`/`_no_bwd` variants
//! are ALSO a hazard for stateful ops specifically — an already-boxed
//! `Arc` can back many forward calls from one instance, which is exactly
//! what `Saved`'s per-call-freshness argument depends on NOT happening).
//!
//! Two checks, at two different scopes:
//!
//! 1. `FORBIDDEN_EVERYWHERE` (the `_arc`/`_no_bwd` names): banned in every
//!    file. Pre-existing STATELESS ops in this crate never call these
//!    (only `apply1`/`apply2`/`apply3` do, indirectly via the safe
//!    `apply_op1`/`apply_op2`/`apply_op3`), so this has no legitimate
//!    exception.
//! 2. Bare `.apply_op1(`/`.apply_op3(` calls: legitimate and COMMON in
//!    this crate for STATELESS (`KernelOp`) ops calling their OWN helper
//!    ops directly (e.g. `crate::ops::attention_block`'s `qkv.apply_op3(
//!    rope_pack, mask, op)`, `crate::ops::low_rank_residual_linear`'s
//!    several `x.apply_op3(&w, &ab, op)` sites — both pass a `Copy`
//!    construction-data-only op, so reusing one instance across calls is
//!    harmless by construction; banning these globally would flag
//!    ALREADY-REVIEWED, correct code). The REAL hazard is scoped to
//!    `ops/flash_attention.rs` specifically — the ONLY file defining a
//!    `StatefulKernelOp` today (`FlashVarlenAttention`,
//!    `FlashVarlenBwdHelper`) — which must reach candle's `apply_op1`/
//!    `apply_op3` ONLY through `super::apply_stateful1`/
//!    `super::apply_stateful3`, never directly. This test enforces the
//!    bare-call ban ONLY in that one file, not crate-wide.
//!
//! Comment lines (trimmed text starting `//`) are skipped in both checks:
//! several files discuss these names in PROSE without calling them (e.g.
//! `ops/softmax.rs`'s esc-037 disposition section: `` (`apply_op1_no_bwd`) ``,
//! no trailing paren — a bare mention; `jammi-encoders/src/attention.rs`'s
//! own doc comment additionally quotes a REAL call-syntax example,
//! `` xs.apply_op1_no_bwd(&SoftmaxLastDim) ``, WITH a trailing paren, of
//! candle_nn's own pre-existing (non-jammi) softmax path — exactly the
//! shape that would false-positive without comment-skipping; that file is
//! outside this test's scanned tree (`jammi-kernels/src` only) today, but
//! the skip is correctness regardless of scope). No CUDA needed (pure
//! source-text scan); runs in every default `cargo test -p jammi-kernels`.

use std::fs;
use std::path::Path;

/// Needles forbidden EVERYWHERE in `src/` (no file is exempt).
const FORBIDDEN_EVERYWHERE: &[&str] = &[
    "apply_op1_arc(",
    "apply_op2_arc(",
    "apply_op3_arc(",
    "apply_op1_no_bwd(",
    "apply_op2_no_bwd(",
    "apply_op3_no_bwd(",
];

/// Needles forbidden ONLY in `ops/flash_attention.rs` — see the module doc's
/// point 2 for why this is scoped to that one file rather than crate-wide.
const FORBIDDEN_IN_FLASH_ATTENTION_RS: &[&str] = &[".apply_op1(", ".apply_op2(", ".apply_op3("];

fn walk_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

fn is_flash_attention_rs(path: &Path) -> bool {
    path.file_name().is_some_and(|f| f == "flash_attention.rs")
}

#[test]
fn no_src_file_bypasses_the_kernelop_stateful_kernelop_gate() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk_rs_files(&src_dir, &mut files);
    assert!(
        files.len() > 10,
        "sanity: expected to find more than 10 .rs files under {}, found {} — the walk is \
         probably broken, not that this crate shrank",
        src_dir.display(),
        files.len()
    );

    let mut violations = Vec::new();
    for path in &files {
        let text =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let check_bare_calls = is_flash_attention_rs(path);
        for (line_no, line) in text.lines().enumerate() {
            if is_comment_line(line) {
                continue;
            }
            for needle in FORBIDDEN_EVERYWHERE {
                if line.contains(needle) {
                    violations.push(format!(
                        "{}:{}: {needle:?} — {}",
                        path.display(),
                        line_no + 1,
                        line.trim()
                    ));
                }
            }
            if check_bare_calls {
                for needle in FORBIDDEN_IN_FLASH_ATTENTION_RS {
                    if line.contains(needle) {
                        violations.push(format!(
                            "{}:{}: {needle:?} (this file's stateful ops must reach candle only \
                             via super::apply_stateful1/apply_stateful3) — {}",
                            path.display(),
                            line_no + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        !files.is_empty() && files.iter().any(|p| is_flash_attention_rs(p)),
        "sanity: ops/flash_attention.rs was not found by the walk (feature-gated files are \
         still on disk regardless of which features this test binary was built with) — the \
         bare-call check above would be silently untested"
    );
    assert!(
        violations.is_empty(),
        "found direct call(s) that bypass KernelOp/StatefulKernelOp:\n{}",
        violations.join("\n")
    );
}

/// Negative control: the needle-matching logic itself is not vacuous.
#[test]
fn the_forbidden_needle_match_is_not_vacuous() {
    let injected_arc = "        let _ = x.apply_op3_arc(y, z, op);";
    assert!(
        FORBIDDEN_EVERYWHERE
            .iter()
            .any(|n| injected_arc.contains(n)),
        "the _arc needle scan does not catch an obvious violation"
    );

    let injected_bare = "    let out = qkv.apply_op3(o, grad_o, helper);";
    assert!(
        FORBIDDEN_IN_FLASH_ATTENTION_RS
            .iter()
            .any(|n| injected_bare.contains(n)),
        "the bare apply_op3( needle scan does not catch an obvious violation"
    );

    // Bare-name doc mentions (no trailing paren) must NOT false-positive.
    let doc_mention = "//! (`apply_op1_no_bwd`) and `QMatMul`, the natural entry point";
    assert!(
        !FORBIDDEN_EVERYWHERE.iter().any(|n| doc_mention.contains(n)),
        "a bare-name doc mention must not be flagged as a call"
    );

    // A doc-comment call-syntax EXAMPLE (trailing paren) must be skipped
    // via comment detection, not needle avoidance.
    let doc_example = "//!     xs.apply_op1_no_bwd(&SoftmaxLastDim)";
    assert!(
        is_comment_line(doc_example),
        "comment detection must catch this line"
    );
    assert!(
        FORBIDDEN_EVERYWHERE.iter().any(|n| doc_example.contains(n)),
        "sanity: the needle itself must still match inside the comment string — proves the \
         comment SKIP, not the needle, is what protects this case"
    );

    // Existing, already-reviewed bare `.apply_op3(` call sites in OTHER
    // files (attention_block.rs, low_rank_residual_linear.rs — both pass a
    // Copy, stateless op) must NOT be flagged — the scope restriction to
    // `flash_attention.rs` is load-bearing, not incidental.
    assert!(!is_flash_attention_rs(Path::new(
        "crates/jammi-kernels/src/ops/attention_block.rs"
    )));
    assert!(!is_flash_attention_rs(Path::new(
        "crates/jammi-kernels/src/ops/low_rank_residual_linear.rs"
    )));
    assert!(is_flash_attention_rs(Path::new(
        "crates/jammi-kernels/src/ops/flash_attention.rs"
    )));
}
