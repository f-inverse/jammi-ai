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
//! Three checks:
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
//!    whichever file(s) actually DEFINE a `Saved`-bearing op — which must
//!    reach candle's `apply_op1`/`apply_op3` ONLY through
//!    `super::apply_stateful1`/`super::apply_stateful3`, never directly.
//!
//!    **Scoped by PROPERTY, not by filename** — `10b1f3b`'s audit found the
//!    previous version hardcoded `flash_attention.rs` by name (advisory
//!    finding: "discipline test keyed on filename"). A file is in scope iff
//!    it declares a struct field of type `Saved<...>` OR `Arc<...>` on a
//!    non-comment line ([`declares_a_stateful_field`]) — `Saved<...>` is
//!    the exact shape every `Saved`-bearing `StatefulKernelOp` in this
//!    crate uses to hold its interior-mutable state (`ops::saved`'s module
//!    doc); `Arc<...>` WIDENS the property (round-5, `ops::quant_matmul_grad`
//!    landing) to the SAME hazard class for a struct that carries its
//!    interior-mutable/shareable state through an `Arc` it did not
//!    construct itself, rather than an owned `Saved<T>` — `QuantMatMulGrad`
//!    holds `Arc<candle_core::quantized::QTensor>`, and `QTensor` itself
//!    owns a `OnceLock` (`repacked_qs` — see that op's own module doc's
//!    "repacked_qs" section for the full safety argument, argued not
//!    denied) — the property this test enforces is "a file whose op struct
//!    carries state that is NOT plain, owned, `Copy`-able construction
//!    data", and an `Arc<...>` field is exactly as much a structural signal
//!    of that as a `Saved<...>` one, even though `QuantMatMulGrad` itself
//!    declares no `Saved<T>` field. A future stateful op added under ANY
//!    OTHER filename, using EITHER shape, is caught automatically; the
//!    previous filename-keyed version would have silently stopped
//!    enforcing the ban on it, and the previous `Saved<`-only property
//!    would have silently let an `Arc`-only op like `QuantMatMulGrad`
//!    reach `.apply_op1(`/`.apply_op3(` directly, bypassing
//!    `apply_stateful1`/`apply_stateful3`'s gate, unchecked.
//! 3. No `Saved`-bearing struct may derive `Clone`/`Copy`, and no field may
//!    be `Arc<Saved<...>>` in place of an owned `Saved<...>` — the shape
//!    `10b1f3b`'s audit demonstrated compiles and ALIASES (`Arc<X>` is
//!    `Clone` regardless of whether `X` itself is; wrapping the `Saved`
//!    field in an extra `Arc` makes the OUTER op struct cheaply `Clone`-able
//!    through `&self`, sidestepping the move-out-of-`&self` compile error
//!    the crate's whole "cannot hoist a stateful op" argument rests on —
//!    see `ops/mod.rs`'s `StatefulKernelOp` doc for the corrected claim).
//!
//! Comment lines (trimmed text starting `//`) are skipped in all three
//! checks: several files discuss these names/types in PROSE without using
//! them (e.g. `ops/softmax.rs`'s esc-037 disposition section:
//! `` (`apply_op1_no_bwd`) ``, no trailing paren — a bare mention;
//! `jammi-encoders/src/attention.rs`'s own doc comment additionally quotes
//! a REAL call-syntax example, `` xs.apply_op1_no_bwd(&SoftmaxLastDim) ``,
//! WITH a trailing paren, of candle_nn's own pre-existing (non-jammi)
//! softmax path — exactly the shape that would false-positive without
//! comment-skipping; that file is outside this test's scanned tree
//! (`jammi-kernels/src` only) today, but the skip is correctness
//! regardless of scope; `ops/mod.rs`'s own doc discusses `Saved<T>` in
//! prose extensively and must NOT be swept into check 2/3's scope by that
//! alone). No CUDA needed (pure source-text scan); runs in every default
//! `cargo test -p jammi-kernels`.

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

/// Needles forbidden ONLY in files [`declares_a_stateful_field`] scopes.
const FORBIDDEN_IN_SAVED_FIELD_FILES: &[&str] = &[".apply_op1(", ".apply_op2(", ".apply_op3("];

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

/// PROPERTY-based scope check (see module doc, point 2): a file is in scope
/// for the bare-call ban iff it declares a struct FIELD of type `Saved<` OR
/// `Arc<` somewhere on a non-comment line — WIDENED (round-5,
/// `ops::quant_matmul_grad` landing) from the original `Saved<`-only
/// property, per this file's own module doc's scope section: an
/// `Arc<...>`-carried field is the same "not plain, owned, `Copy`-able
/// construction data" hazard shape a `Saved<T>` field is, even when the op
/// struct declares no `Saved<T>` field of its own (`QuantMatMulGrad` holds
/// `Arc<candle_core::quantized::QTensor>` only). Broad enough to also
/// (harmlessly) scope `ops/saved.rs` itself (`Saved<T>`'s own definition,
/// whose `inner` field is itself `Arc<Mutex<Option<T>>>` — doubly in scope
/// — plus its `#[cfg(test)]` module, none of which calls `apply_op1/2/3`)
/// — a false POSITIVE there would just mean an extra file gets checked and
/// passes; the property must never produce a false NEGATIVE (a real
/// stateful op file escaping scope), which filename-keying could.
fn declares_a_stateful_field(text: &str) -> bool {
    text.lines()
        .any(|line| !is_comment_line(line) && (line.contains("Saved<") || line.contains("Arc<")))
}

/// Check 3 (module doc, point 3): a `Saved`-scoped file must not derive
/// `Clone`/`Copy` and must not wrap its `Saved` field in an `Arc`.
fn clone_copy_or_arc_saved_violations(path: &Path, text: &str) -> Vec<String> {
    let mut hits = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if is_comment_line(line) {
            continue;
        }
        if line.contains("#[derive(") && (line.contains("Clone") || line.contains("Copy")) {
            hits.push(format!(
                "{}:{}: derives Clone/Copy in a Saved-bearing file — {}",
                path.display(),
                line_no + 1,
                line.trim()
            ));
        }
        if line.contains("Arc<Saved") {
            hits.push(format!(
                "{}:{}: Arc<Saved<...>> field — Arc is Clone regardless of Saved's own refusal, \
                 aliasing the interior-mutable slot — {}",
                path.display(),
                line_no + 1,
                line.trim()
            ));
        }
    }
    hits
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
    let mut stateful_field_files_seen = 0usize;
    for path in &files {
        let text =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let in_scope = declares_a_stateful_field(&text);
        if in_scope {
            stateful_field_files_seen += 1;
        }
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
            if in_scope {
                for needle in FORBIDDEN_IN_SAVED_FIELD_FILES {
                    if line.contains(needle) {
                        violations.push(format!(
                            "{}:{}: {needle:?} (a file declaring a Saved<...> or Arc<...> field \
                             must reach candle only via \
                             super::apply_stateful1/apply_stateful3) — {}",
                            path.display(),
                            line_no + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
        // Check 3 is ALSO scoped to `in_scope` files: every stateless
        // (KernelOp) op in this crate deliberately DOES derive Clone/Copy
        // (that is the whole point of KernelOp's `Copy` bound) — flagging
        // that crate-wide would ban the correct, common case. The hazard
        // this check exists for is a Saved-bearing struct ALSO deriving
        // Clone/Copy (see this file's module doc, point 3).
        if in_scope {
            violations.extend(clone_copy_or_arc_saved_violations(path, &text));
        }
    }
    assert!(
        stateful_field_files_seen >= 3,
        "sanity: expected at least 3 files declaring a Saved<...> or Arc<...> field \
         (ops/saved.rs's own definition, ops/flash_attention.rs's two op structs, and \
         ops/quant_matmul_grad.rs's Arc<QTensor> field) — found {stateful_field_files_seen}; \
         the widened PROPERTY scope is untested if this stays at the pre-widening 0/1/2 count"
    );
    assert!(
        violations.is_empty(),
        "found violation(s) of the KernelOp/StatefulKernelOp discipline:\n{}",
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
        FORBIDDEN_IN_SAVED_FIELD_FILES
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

    // --- Scope property (check 2): a struct field, not a doc mention.
    let real_field = "    lse: Saved<CudaSlice<f32>>,";
    assert!(
        declares_a_stateful_field(real_field),
        "a real Saved<...> struct field must scope the file in"
    );
    let no_mention = "    lse: CudaSlice<f32>,";
    assert!(
        !declares_a_stateful_field(no_mention),
        "a file mentioning nothing about Saved/Arc must not be scoped in"
    );
    let comment_only = "//! discusses `Saved<T>` in prose only, never as a field";
    assert!(
        !declares_a_stateful_field(comment_only),
        "a Saved<...> MENTION inside a comment must not scope the file in — ops/mod.rs's own \
         module doc does exactly this and must stay out of scope"
    );

    // --- Scope property widening (round-5, QuantMatMulGrad landing): an
    // Arc<...>-carried field (no Saved<T> field at all) must ALSO scope a
    // file in — this is what mechanically forces ops/quant_matmul_grad.rs
    // into the discipline apparatus despite holding Arc<QTensor>, not
    // Saved<T>.
    let arc_field = "    w: Arc<QTensor>,";
    assert!(
        declares_a_stateful_field(arc_field),
        "an Arc<...>-carried struct field (no Saved<T> present) must ALSO scope the file in"
    );
    let arc_comment_only = "// w: Arc<QTensor> — considered and rejected, do not add it";
    assert!(
        !declares_a_stateful_field(arc_comment_only),
        "an Arc<...> MENTION inside a comment must not scope the file in, mirroring Saved<...>'s \
         own comment-skip"
    );

    // Existing, already-reviewed bare `.apply_op3(` call sites in OTHER
    // (non-Saved-declaring, non-Arc-declaring) files must NOT be flagged —
    // the scope restriction is load-bearing, not incidental.
    // kernel-oracles: fn-in-literal reviewed: grep-discipline fixture text, not code
    let attention_block_text = "pub(crate) fn foo() { qkv.apply_op3(rope_pack, mask, op) }";
    assert!(
        !declares_a_stateful_field(attention_block_text),
        "a file with no Saved<...>/Arc<...> field must stay out of scope even if it calls \
         apply_op3"
    );

    // --- Check 3: Clone/Copy derive and Arc<Saved<...>> detection.
    let bad_derive = "#[derive(Clone)]\nstruct Evil { lse: Saved<u32> }";
    assert!(
        !clone_copy_or_arc_saved_violations(Path::new("evil.rs"), bad_derive).is_empty(),
        "a #[derive(Clone)] on a Saved-bearing struct must be flagged"
    );
    let bad_arc = "struct Evil { lse: Arc<Saved<u32>> }";
    assert!(
        !clone_copy_or_arc_saved_violations(Path::new("evil.rs"), bad_arc).is_empty(),
        "an Arc<Saved<...>> field must be flagged"
    );
    let fine = "struct Fine { lse: Saved<u32> }\nimpl super::sealed::Sealed for Fine {}";
    assert!(
        clone_copy_or_arc_saved_violations(Path::new("fine.rs"), fine).is_empty(),
        "a plain owned Saved<...> field with no Clone/Copy derive must NOT be flagged"
    );
    let comment_derive = "// #[derive(Clone)] would be wrong here, do not add it";
    assert!(
        clone_copy_or_arc_saved_violations(Path::new("commented.rs"), comment_derive).is_empty(),
        "a #[derive(Clone)] MENTIONED inside a comment must not be flagged"
    );
}
