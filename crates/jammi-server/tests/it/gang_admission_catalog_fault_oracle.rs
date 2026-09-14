//! A source-scan oracle over `crates/jammi-server/src/grpc/gang.rs`'s
//! `fn run_rank` span: every admission-time catalog read this handler makes
//! maps its `Err` through `admission_catalog_fault`, never
//! `crate::grpc::wire::map_engine_error` — see that function's own doc for
//! why the generic mapping is wrong here (no case for a raw backend fault,
//! falls through to `Internal`, never tells a retrying caller this was
//! transient).
//!
//! **Honest universe.** The scanned span is located textually — the byte
//! offset of the literal `fn run_rank` declaration, then the FIRST `{`
//! after it (the function body's opening brace, since the signature itself,
//! `(&self, request: Request<tonic::Streaming<RankControl>>) -> Result<...>`,
//! contains no brace of its own), then a brace-depth walk to the MATCHING
//! closing `}` — never "the rest of the file", which would let an unrelated
//! later `map_engine_error` call (in a different fn entirely) produce a
//! false failure, and never "the whole `impl` block", which would let a
//! genuine violation hide past `run_rank`'s own closing brace. Comments,
//! doc comments, and string/char literals are masked to spaces first
//! (`mask_non_code`, ported verbatim from
//! `gang_rank_admission_oracle.rs`/`crates/jammi-db/tests/it/whose_fault_gate.rs`'s
//! function of the same name) — this very file's own doc comments name both
//! call-tokens, so an unmasked scan would self-hit.
//!
//! Two assertions over that ONE span: it must NOT contain
//! `map_engine_error` as code (a plain substring check is deliberately
//! stricter than a call-site-boundary check here — ANY appearance of that
//! identifier inside `run_rank`, even one that turns out on inspection not
//! to be a call, is exactly the drift this oracle exists to catch, so a
//! false positive costs nothing and a false negative is the failure mode
//! that matters); and it MUST contain `admission_catalog_fault` at least
//! twice (`get_job_for_rank`'s and `fresh_instance`'s own `Err` arms) — a
//! sanity floor so a version of this file that deleted BOTH call sites
//! (leaving no catalog-fault mapping at all) still fails, rather than
//! passing vacuously on "no `map_engine_error` found because nothing reads
//! a catalog at all".

use std::path::PathBuf;
use std::process::Command;

const GANG_RS: &str = "crates/jammi-server/src/grpc/gang.rs";

fn repo_root() -> PathBuf {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("git rev-parse must run");
    assert!(out.status.success(), "git rev-parse --show-toplevel failed");
    PathBuf::from(
        String::from_utf8(out.stdout)
            .expect("utf8 path")
            .trim()
            .to_string(),
    )
}

/// Ported verbatim from `gang_rank_admission_oracle.rs`'s function of the
/// same name (itself ported from `crates/jammi-db/tests/it/whose_fault_gate.rs`):
/// replaces every line comment, block comment, string literal (plain and
/// raw), and char literal in `text` with spaces — same length, same
/// newlines, so byte offsets computed against the masked text still index
/// correctly into the ORIGINAL text.
fn mask_non_code(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<char> = chars.clone();
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            let mut j = i;
            while j < n && chars[j] != '\n' {
                out[j] = ' ';
                j += 1;
            }
            i = j;
            continue;
        }
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            let mut j = i + 2;
            while j + 1 < n && !(chars[j] == '*' && chars[j + 1] == '/') {
                j += 1;
            }
            let end = (j + 2).min(n);
            for k in i..end {
                if chars[k] != '\n' {
                    out[k] = ' ';
                }
            }
            i = end;
            continue;
        }
        if c == 'r' && i + 1 < n && (chars[i + 1] == '"' || chars[i + 1] == '#') {
            let mut k = i + 1;
            let mut hashes = 0usize;
            while k < n && chars[k] == '#' {
                hashes += 1;
                k += 1;
            }
            if k < n && chars[k] == '"' {
                let content_start = k + 1;
                let mut j = content_start;
                let end = loop {
                    if j >= n {
                        break n;
                    }
                    if chars[j] == '"'
                        && chars[j + 1..(j + 1 + hashes).min(n)]
                            .iter()
                            .all(|ch| *ch == '#')
                        && j + 1 + hashes <= n
                    {
                        break j + 1 + hashes;
                    }
                    j += 1;
                };
                for k2 in i..end {
                    if chars[k2] != '\n' {
                        out[k2] = ' ';
                    }
                }
                i = end;
                continue;
            }
        }
        if c == '"' {
            let mut j = i + 1;
            while j < n {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == '"' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            let end = j.min(n);
            for k in i..end {
                if chars[k] != '\n' {
                    out[k] = ' ';
                }
            }
            i = end;
            continue;
        }
        if c == '\'' {
            if i + 1 < n && chars[i + 1] == '\\' {
                let mut j = i + 2;
                let mut steps = 0;
                while j < n && chars[j] != '\'' && steps < 10 {
                    j += 1;
                    steps += 1;
                }
                if j < n && chars[j] == '\'' {
                    let end = j + 1;
                    for k in i..end {
                        if chars[k] != '\n' {
                            out[k] = ' ';
                        }
                    }
                    i = end;
                    continue;
                }
            } else if i + 2 < n && chars[i + 2] == '\'' {
                out[i] = ' ';
                out[i + 1] = ' ';
                out[i + 2] = ' ';
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    out.into_iter().collect()
}

/// Locates `fn run_rank`'s own body span (the byte range strictly between
/// its opening `{` and its matching closing `}`, both exclusive) inside
/// `masked` — the SAME text `contains`/`matches` below scan, so offsets
/// line up. Panics naming what went wrong (never a silent empty span) if
/// `fn run_rank` is renamed/removed, or its signature grows a brace this
/// scanner does not expect.
fn run_rank_body_span(masked: &str) -> &str {
    let fn_start = masked.find("fn run_rank").expect(
        "gang.rs must still declare `fn run_rank` — this oracle's scanned fn was \
                 renamed or removed",
    );
    let sig_and_body = &masked[fn_start..];
    let open = sig_and_body
        .find('{')
        .expect("fn run_rank's signature must be followed by a body opening `{`");
    let bytes = sig_and_body.as_bytes();
    let mut depth = 0i32;
    let mut close = None;
    for (idx, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(idx);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.expect(
        "fn run_rank's opening `{` has no matching closing `}` in the scanned text — a \
         brace-depth bug in this scanner or a genuinely unbalanced source file",
    );
    &sig_and_body[open + 1..close]
}

/// The oracle itself: within `fn run_rank`'s own body span, `map_engine_error`
/// is never named as code, and `admission_catalog_fault` is named at least
/// twice (`get_job_for_rank`'s and `fresh_instance`'s own `Err` arms).
#[test]
fn run_rank_never_calls_map_engine_error() {
    let path = repo_root().join(GANG_RS);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{GANG_RS} must be readable: {e}"));
    let masked = mask_non_code(&text);
    let body = run_rank_body_span(&masked);

    assert!(
        !body.contains("map_engine_error"),
        "fn run_rank must never call map_engine_error — every admission-time catalog \
         read on this path must map through admission_catalog_fault instead \
         (Unavailable, transient), never the generic Internal-catch-all mapping"
    );
    let admission_catalog_fault_count = body.matches("admission_catalog_fault").count();
    assert!(
        admission_catalog_fault_count >= 2,
        "fn run_rank must call admission_catalog_fault at least twice (get_job_for_rank's \
         and fresh_instance's own Err arms) — found {admission_catalog_fault_count}; a \
         version of this file with `map_engine_error` simply deleted, rather than replaced, \
         would otherwise pass this oracle vacuously"
    );
}

/// The span-extraction self-test this oracle depends on: a decoy `fn` BEFORE
/// `run_rank` (naming `map_engine_error`) must NOT be included in the
/// extracted span, and a decoy `fn` AFTER it (naming `map_engine_error` too)
/// must likewise be excluded — proving the brace-depth walk stops at
/// `run_rank`'s OWN closing brace, never "the rest of the file".
#[test]
fn run_rank_body_span_excludes_neighboring_functions() {
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — decoy fn before run_rank
    let before = "fn before() { map_engine_error(1) }\n";
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — the scanned fn itself
    let target = "fn run_rank() { admission_catalog_fault(1); admission_catalog_fault(2); }\n";
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — decoy fn after run_rank
    let after = "fn after() { map_engine_error(2) }\n";
    let fixture = format!("{before}{target}{after}");
    let masked = mask_non_code(&fixture);
    let body = run_rank_body_span(&masked);
    assert!(
        !body.contains("map_engine_error"),
        "the extracted span must exclude both neighboring decoy functions, got: {body:?}"
    );
    assert_eq!(
        body.matches("admission_catalog_fault").count(),
        2,
        "the extracted span must include exactly run_rank's own two calls, got: {body:?}"
    );
}

/// The masking self-test this oracle depends on: a call-token inside a
/// comment or string literal is not a code occurrence — otherwise this very
/// file's own doc comments (which name both `map_engine_error` and
/// `admission_catalog_fault` in prose) would self-hit.
#[test]
fn mask_non_code_hides_comments_and_strings_but_not_code() {
    let commented = "// calls map_engine_error( here, not really\nfn f() {}\n";
    assert!(!mask_non_code(commented).contains("map_engine_error("));

    let doc_commented = "/// mentions map_engine_error( in prose\nfn f() {}\n";
    assert!(!mask_non_code(doc_commented).contains("map_engine_error("));

    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code
    let string_literal = "fn f() { let s = \"map_engine_error(\"; }\n";
    assert!(!mask_non_code(string_literal).contains("map_engine_error("));

    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code
    let real_call = "fn f() { map_engine_error(e) }\n";
    assert!(mask_non_code(real_call).contains("map_engine_error("));
}
