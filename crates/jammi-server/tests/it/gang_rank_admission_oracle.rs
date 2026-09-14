//! `CONTRACT-U5a.md` §I1(a) / §W2 Resolution — two enumerating-caller
//! oracles, each a MEASURED claim (never prose): `Catalog::get_job_for_rank`
//! and `Catalog::get_result_table_for_tenant` are each called from nowhere
//! outside the gang `RunRank` handler (plus each function's own crate's
//! tests, which call it directly to exercise it in isolation).
//!
//! The scanned surface is derived from `git ls-files` (never a hand-rolled
//! directory walk) over the whole tracked tree — `crates/**` and everything
//! else — matching `crates/jammi-db/tests/it/whose_fault_gate.rs`'s own
//! precedent for this shape of claim. A tracked file `git ls-files` reports
//! that this process cannot then read is a hard failure naming the file.
//!
//! The detector is a substring match on the call-token (`name(`), but ONLY
//! over CODE: every line comment, block comment, string literal (plain and
//! raw), and char literal is masked to spaces first (`mask_non_code`,
//! ported verbatim from `whose_fault_gate.rs`'s own function of the same
//! name). This is load-bearing, not cosmetic — this very oracle file names
//! both call-tokens in its own doc comments, in its assert messages, and as
//! string-literal arguments to `files_containing` itself; an unmasked
//! substring scan would find those and self-hit, and the fix must never be
//! to allowlist this file (a self-allowlisted oracle could hide a real new
//! call site behind its own comments and never notice). Masking is what
//! keeps this file honest without an allowlist entry: its comments and
//! string literals are masked away.
//!
//! Masking alone is not quite enough, though: this file's OWN test
//! functions are named `only_the_gang_run_rank_handler_calls_get_job_for_\
//! rank` and `only_resolve_training_set_identity_calls_get_result_table_\
//! for_tenant`, so their declarations (`fn ...calls_get_job_for_rank() {`)
//! are a genuine CODE occurrence of the substring `get_job_for_rank(` that
//! is nonetheless not a call — it is a longer identifier that happens to
//! end in the token, immediately followed by its own empty parameter
//! list's `(`. `contains_code_token` closes that gap with an
//! identifier-boundary check (the `boundary_ok` idiom
//! `whose_fault_gate.rs`'s `find_fn_regions` already uses): a match is
//! only a hit if the byte immediately before it is not itself an
//! identifier byte, which a real call site (`catalog.get_job_for_rank(`,
//! preceded by `.`) always satisfies and a same-tokened longer identifier
//! never does.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn git_ls_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .arg("ls-files")
        .current_dir(root)
        .output()
        .expect("git ls-files must run");
    assert!(out.status.success(), "git ls-files failed");
    String::from_utf8(out.stdout)
        .expect("utf8 file list")
        .lines()
        .map(str::to_string)
        .collect()
}

/// Replace every line comment, block comment, string literal (plain and
/// raw), and char literal in `text` with spaces — same length, same
/// newlines, so line numbers still match the original file. Ported
/// verbatim from `crates/jammi-db/tests/it/whose_fault_gate.rs`'s
/// `mask_non_code` (same stated limit: a nested block comment's interior is
/// treated as code — this surface has none, checked by the fact that the
/// masking self-test below and the two oracle tests are both green).
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

/// `true` for an ASCII identifier byte (`[A-Za-z0-9_]`) — this surface's
/// identifiers are ASCII throughout, the same assumption
/// `whose_fault_gate.rs`'s own `is_ident_byte` makes.
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `true` if `masked` contains `token` at a genuine call-site boundary: the
/// byte immediately before the match is not an identifier byte (or the
/// match starts at byte 0). See this file's header doc for why a plain
/// substring test is not enough — a longer identifier that merely ENDS in
/// `token` (this file's own test names, `..._get_job_for_rank()`) must not
/// count, while a real call (`catalog.get_job_for_rank(...)`, preceded by
/// `.`) always does.
fn contains_code_token(masked: &str, token: &str) -> bool {
    let bytes = masked.as_bytes();
    let tbytes = token.as_bytes();
    let n = bytes.len();
    let tn = tbytes.len();
    if tn == 0 || tn > n {
        return false;
    }
    for i in 0..=(n - tn) {
        if &bytes[i..i + tn] == tbytes {
            let boundary_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            if boundary_ok {
                return true;
            }
        }
    }
    false
}

/// Every tracked `.rs` file containing `token` as a genuine call-site
/// occurrence in CODE — never inside a comment, doc comment, or
/// string/char literal (`mask_non_code` strips those first), and never a
/// longer identifier that merely ends in `token` (`contains_code_token`'s
/// identifier-boundary check) — relative to the repo root. Hard-fails
/// (naming the file) if `git ls-files` reports a tracked file this process
/// cannot then read.
fn files_containing(token: &str) -> HashSet<String> {
    let root = repo_root();
    let mut hits = HashSet::new();
    for rel in git_ls_files(&root) {
        if !rel.ends_with(".rs") {
            continue;
        }
        let path = root.join(&rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("git ls-files tracked {rel} but it could not be read: {e}"));
        if contains_code_token(&mask_non_code(&text), token) {
            hits.insert(rel);
        }
    }
    hits
}

/// `CONTRACT-U5a.md` §I1(a): "the enumerating-caller oracle over `crates/**`"
/// — `get_job_for_rank`'s only production caller is the gang `RunRank`
/// handler; the sole other hit is `jammi-db`'s own unit test exercising the
/// method directly.
#[test]
fn only_the_gang_run_rank_handler_calls_get_job_for_rank() {
    let hits = files_containing("get_job_for_rank(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/jobs_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",     // the ONE production caller
        "crates/jammi-db/tests/it/gang_rank_admission.rs", // jammi-db's own unit tests
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_job_for_rank(` occurrence outside the allowed set: {hit} \
             (allowed: {allowed:?}) — a new caller of this primary-key-only, \
             non-tenant-scoped verb must be reviewed against CONTRACT-U5a.md §I1(a) \
             before this allowlist grows"
        );
    }
    for must_hit in &allowed {
        assert!(
            hits.contains(*must_hit),
            "{must_hit} is in the allowlist but no longer contains \
             `get_job_for_rank(` — shrink the allowlist rather than leaving a stale entry"
        );
    }
}

/// `CONTRACT-U5a.md` §W2 Resolution (round-11 fold, ruling 5): "no caller
/// other than the gang `RunRank` handler resolves `training_set_location`"
/// — `get_result_table_for_tenant`'s only production caller is
/// `resolve_training_set_identity` inside the gang handler; the two other
/// hits are `gang_service.rs`'s own b1' tests (rulings 4/5) exercising the
/// raw verb directly to demonstrate the hazard those rulings name.
#[test]
fn only_resolve_training_set_identity_calls_get_result_table_for_tenant() {
    let hits = files_containing("get_result_table_for_tenant(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/result_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",       // the ONE production caller
        "crates/jammi-server/tests/it/gang_service.rs", // this crate's own b1' tests
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_result_table_for_tenant(` occurrence outside the allowed \
             set: {hit} (allowed: {allowed:?}) — a new caller of this strict-tenant \
             verb must be reviewed against CONTRACT-U5a.md §W2 Resolution's admin-scope \
             hazard (ruling 4) before this allowlist grows"
        );
    }
    for must_hit in &allowed {
        assert!(
            hits.contains(*must_hit),
            "{must_hit} is in the allowlist but no longer contains \
             `get_result_table_for_tenant(` — shrink the allowlist rather than leaving \
             a stale entry"
        );
    }
}

/// The masking self-test both enumerating-caller oracles above depend on:
/// a call-token inside a `//` line comment, a `///` doc comment, or a
/// string literal is NOT a code occurrence; the same token appearing as an
/// actual call in code IS one. Without this, this very file's own doc
/// comments and its `files_containing("get_job_for_rank(")` /
/// `files_containing("get_result_table_for_tenant(")` string-literal
/// arguments would self-hit and force a self-allowlist entry — the one
/// outcome this design rejects.
#[test]
fn mask_non_code_hides_comments_and_strings_but_not_code() {
    let commented = "// calls get_job_for_rank( here, not really\nfn f() {}\n";
    assert!(
        !mask_non_code(commented).contains("get_job_for_rank("),
        "a line-comment occurrence must be masked"
    );

    let doc_commented = "/// this doc mentions get_job_for_rank( in prose\nfn f() {}\n";
    assert!(
        !mask_non_code(doc_commented).contains("get_job_for_rank("),
        "a doc-comment occurrence must be masked"
    );

    let block_commented = "/* get_job_for_rank( inside a block comment */\nfn f() {}\n";
    assert!(
        !mask_non_code(block_commented).contains("get_job_for_rank("),
        "a block-comment occurrence must be masked"
    );

    let string_literal = "fn f() { let s = \"get_job_for_rank(\"; }\n";
    assert!(
        !mask_non_code(string_literal).contains("get_job_for_rank("),
        "a string-literal occurrence must be masked"
    );

    let raw_string_literal = "fn f() { let s = r#\"get_job_for_rank(\"#; }\n";
    assert!(
        !mask_non_code(raw_string_literal).contains("get_job_for_rank("),
        "a raw-string-literal occurrence must be masked"
    );

    let real_call = "fn f() { catalog.get_job_for_rank(&id); }\n";
    assert!(
        mask_non_code(real_call).contains("get_job_for_rank("),
        "an actual call in code must NOT be masked"
    );
}

/// The identifier-boundary self-test `contains_code_token` depends on: a
/// longer identifier that merely ENDS in the token (exactly this file's own
/// test function declarations, `fn only_..._calls_get_job_for_rank() {`)
/// is NOT a call-site hit, while a real call (preceded by `.`, whitespace,
/// or nothing at all) IS.
#[test]
fn contains_code_token_rejects_a_same_tokened_longer_identifier() {
    let fn_declaration = "fn only_the_gang_run_rank_handler_calls_get_job_for_rank() {}\n";
    assert!(
        !contains_code_token(fn_declaration, "get_job_for_rank("),
        "a longer identifier merely ending in the token must not count as a call"
    );

    let method_call = "fn f() { catalog.get_job_for_rank(&id); }\n";
    assert!(
        contains_code_token(method_call, "get_job_for_rank("),
        "a real call preceded by `.` must count"
    );

    let bare_call_at_start = "get_job_for_rank(&id)";
    assert!(
        contains_code_token(bare_call_at_start, "get_job_for_rank("),
        "a call at byte 0 (no preceding byte at all) must count"
    );

    let whitespace_preceded = "fn f() { let _ = get_job_for_rank(&id); }\n";
    assert!(
        contains_code_token(whitespace_preceded, "get_job_for_rank("),
        "a call preceded by whitespace must count"
    );
}
