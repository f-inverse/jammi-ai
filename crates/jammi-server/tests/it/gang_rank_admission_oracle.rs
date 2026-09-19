//! Two enumerating-caller oracles for the I-GANG row predicate, MEASURED
//! claims (never prose): `Catalog::get_job_for_rank`
//! is called from nowhere outside the gang `RunRank` handler (plus
//! `jammi-db`'s own tests, which call it directly to exercise it in
//! isolation, and the producer→consumer parity test), and the strict
//! tenant-pinned resolver `Catalog::get_result_table_for_tenant` — the
//! world>1 conjunct's ONE tenant-scoped read — is called from
//! nowhere outside `gang.rs`'s `resolve_training_set_identity` (plus
//! `jammi-db`'s own strict-predicate tests).
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
//! the call-token in its own doc comments, in its assert messages, and as a
//! string-literal argument to `files_containing` itself; an unmasked
//! substring scan would find those and self-hit, and the fix must never be
//! to allowlist this file (a self-allowlisted oracle could hide a real new
//! call site behind its own comments and never notice). Masking is what
//! keeps this file honest without an allowlist entry: its comments and
//! string literals are masked away.
//!
//! Masking alone is not quite enough, though: this file's OWN test
//! function is named `only_the_gang_run_rank_handler_calls_get_job_for_\
//! rank`, so its declaration (`fn ...calls_get_job_for_rank() {`)
//! is a genuine CODE occurrence of the substring `get_job_for_rank(` that
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
    // Byte-preserving: a masked character becomes as many spaces as it
    // had bytes, so every byte offset computed on the masked text indexes
    // the SAME position in the original — the fn-body slices below are
    // taken from the original (its SQL string literals intact) at offsets
    // found on the masked copy, which a multi-byte character in a comment
    // (an em-dash) otherwise shifts.
    let mut masked = String::with_capacity(text.len());
    for (orig, m) in chars.iter().zip(out.iter()) {
        if *m == ' ' && *orig != ' ' {
            for _ in 0..orig.len_utf8() {
                masked.push(' ');
            }
        } else {
            masked.push(*m);
        }
    }
    debug_assert_eq!(masked.len(), text.len());
    masked
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

/// The enumerating-caller oracle over `crates/**`: `get_job_for_rank`'s only
/// production caller is the gang `RunRank` handler; the other hits are
/// `jammi-db`'s own unit tests exercising the method directly and the
/// producer→consumer parity test, which calls it directly (never through
/// the RPC) to compare its own decode against a real `TrainingSpec`
/// producer's serialization.
#[test]
fn only_the_gang_run_rank_handler_calls_get_job_for_rank() {
    let hits = files_containing("get_job_for_rank(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/jobs_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",     // the ONE production caller
        "crates/jammi-db/tests/it/gang_rank_admission.rs", // jammi-db's own unit tests
        "crates/jammi-server/tests/it/gang_training_spec_parity.rs", // producer/consumer parity, direct call
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_job_for_rank(` occurrence outside the allowed set: {hit} \
             (allowed: {allowed:?}) — a new caller of this primary-key-only, \
             non-tenant-scoped verb must be reviewed and this allowlist \
             deliberately grown, never left stale"
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

/// The strict resolver's enumerating-caller oracle: `get_result_table_for_tenant`'s
/// only production caller is `gang.rs`'s `resolve_training_set_identity`
/// (the world>1 conjunct's resolution site, whose own callers are
/// `run_rank` and the hold loop's re-verification); the other hits are the
/// definition and `jammi-db`'s own strict-predicate tests. A new caller of
/// this verb is a new tenant-pinned read site and must be reviewed here.
#[test]
fn only_the_gang_resolution_site_calls_get_result_table_for_tenant() {
    let hits = files_containing("get_result_table_for_tenant(");
    let allowed: HashSet<&str> = [
        "crates/jammi-db/src/catalog/result_repo.rs", // the definition itself
        "crates/jammi-server/src/grpc/gang.rs",       // the ONE production caller
        "crates/jammi-db/tests/it/result_tables.rs",  // jammi-db's own strict-predicate tests
        // The SECOND production caller, reviewed: `JammiCodec`'s `AnnSearchExec`
        // decode rebuilds the operator on a Ballista executor from the table
        // name AND the tenant the SUBMITTER's own session carried onto the
        // wire — a read pinned to the carried tenant, never to the decoding
        // process's ambient tenant (a scheduler/executor process has none),
        // over the internal Ballista listeners, which are the peer listener's
        // trust class (every client of them is a jammi role, I-PEER).
        "crates/jammi-ballista/src/codec.rs",
    ]
    .into_iter()
    .collect();
    for hit in &hits {
        assert!(
            allowed.contains(hit.as_str()),
            "unexpected `get_result_table_for_tenant(` occurrence outside the allowed set: {hit} \
             (allowed: {allowed:?}) — a new tenant-pinned read site must be reviewed and this \
             allowlist deliberately grown, never left stale"
        );
    }
    for must_hit in &allowed {
        assert!(
            hits.contains(*must_hit),
            "{must_hit} is in the allowlist but no longer contains \
             `get_result_table_for_tenant(` — shrink the allowlist rather than leaving a stale entry"
        );
    }
}

/// The masking self-test the enumerating-caller oracles above depend on:
/// a call-token inside a `//` line comment, a `///` doc comment, or a
/// string literal is NOT a code occurrence; the same token appearing as an
/// actual call in code IS one. Without this, this very file's own doc
/// comments and its `files_containing("get_job_for_rank(")` string-literal
/// argument would self-hit and force a self-allowlist entry — the one
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
