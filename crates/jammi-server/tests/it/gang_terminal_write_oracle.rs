//! A source-scan oracle for the peer's terminal-write scope (g2', OPS
//! D6/D10): the gang admission handler and its hold loop
//! (`crates/jammi-server/src/grpc/gang.rs`) write NOTHING to the `jobs`
//! table on behalf of a rank — a reclaim, a drain, a refutation or a park
//! ends the STREAM with the matching reason and leaves the job row exactly
//! as it was.
//!
//! **Honest universe, derived not hand-listed.** The set of `jobs` writers
//! is computed from `crates/jammi-db/src/catalog/jobs_repo.rs` itself: every
//! `pub async fn NAME(` declared there whose own body (a brace-depth walk
//! from the declaration to its matching `}`) contains a SQL statement that
//! writes the `jobs` table (`UPDATE jobs`, `INSERT INTO jobs`, `DELETE FROM
//! jobs`, matched case-insensitively inside the body's string literals —
//! the ONE place SQL lives) is a writer. That set is then asserted
//! non-empty and to contain the writers this program is known to have
//! (`fail_job`, `fill_training_set_identity`, `claim_next`), so a scanner
//! that silently found nothing fails rather than passing vacuously. The
//! oracle proper: none of those writers' call-tokens (`NAME(`) occurs as
//! CODE anywhere in `gang.rs` (comments and string literals masked to
//! spaces first — `mask_non_code`, ported verbatim from
//! `gang_rank_admission_oracle.rs`; identifier-boundary check, so a longer
//! identifier merely ending in a writer's name is not a hit). Read-only
//! verbs (`get_job_for_rank`, `fresh_instance`, `get_result_table_for_tenant`)
//! are not writers and are exactly what the handler is allowed to call.
//!
//! What this oracle does NOT prove: a writer added to another catalog
//! module (a `jobs` write outside `jobs_repo.rs`) is outside its universe —
//! `jobs_repo.rs` is the catalog's one `jobs` module today, and a second
//! one would be a reviewable structural change, not a silent hole here.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

const GANG_RS: &str = "crates/jammi-server/src/grpc/gang.rs";
const JOBS_REPO_RS: &str = "crates/jammi-db/src/catalog/jobs_repo.rs";

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
/// same name: replaces every line comment, block comment, string literal
/// (plain and raw), and char literal in `text` with spaces — same length,
/// same newlines.
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

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether `masked` contains `token` as code with an identifier boundary
/// before it (so `fn only_..._calls_fail_job(` is not a hit for `fail_job(`).
fn contains_code_token(masked: &str, token: &str) -> bool {
    let bytes = masked.as_bytes();
    let mut from = 0usize;
    while let Some(pos) = masked[from..].find(token) {
        let at = from + pos;
        let boundary_ok = at == 0 || !is_ident_byte(bytes[at - 1]);
        if boundary_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

/// Every `pub async fn NAME(` in `jobs_repo.rs` that writes the `jobs`
/// table: its body carries a write statement, or calls a fn that writes —
/// followed to a fixpoint, so a writer that delegates to a private shared
/// writer is still a writer. Declarations are found on the MASKED text (so a
/// `fn` in a comment is not a declaration); each body span is taken from the
/// ORIGINAL text (so its SQL string literals are visible to the write scan).
fn jobs_writers(original: &str) -> BTreeSet<String> {
    let masked = mask_non_code(original);
    // (name, is_pub, masked body, writes directly)
    let mut fns: Vec<(String, bool, String, bool)> = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = masked[from..].find("async fn ") {
        let keyword = from + pos;
        let is_pub = masked[..keyword].trim_end().ends_with("pub");
        let decl = keyword + "async fn ".len();
        let name_end = masked[decl..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map(|off| decl + off)
            .unwrap_or(masked.len());
        let name = masked[decl..name_end].to_string();
        // The body: the first `{` after the signature's own `(`...`)` —
        // walk to the paren that closes the parameter list, then the first
        // brace after it (a where-clause or a return type carries none of
        // its own on these declarations).
        let open_paren = masked[name_end..]
            .find('(')
            .map(|off| name_end + off)
            .expect("a fn declaration has a parameter list");
        let mut depth = 0i32;
        let mut close_paren = None;
        for (idx, b) in masked.as_bytes().iter().enumerate().skip(open_paren) {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        close_paren = Some(idx);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close_paren = close_paren.expect("balanced parameter list");
        let open_brace = masked[close_paren..]
            .find('{')
            .map(|off| close_paren + off)
            .expect("a fn declaration has a body");
        let mut depth = 0i32;
        let mut close_brace = None;
        for (idx, b) in masked.as_bytes().iter().enumerate().skip(open_brace) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close_brace = Some(idx);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close_brace = close_brace.expect("balanced fn body");
        let body = original[open_brace..=close_brace].to_ascii_lowercase();
        let writes = ["update jobs", "insert into jobs", "delete from jobs"]
            .iter()
            .any(|needle| body.contains(needle));
        fns.push((
            name,
            is_pub,
            masked[open_brace..=close_brace].to_string(),
            writes,
        ));
        from = close_brace;
    }
    let mut writers: BTreeSet<String> = fns
        .iter()
        .filter(|(_, _, _, writes)| *writes)
        .map(|(name, ..)| name.clone())
        .collect();
    // Grow the set until a pass adds nothing: each pass admits the fns that
    // call a fn already in it. Bounded by the number of fns.
    loop {
        let callers: Vec<String> = fns
            .iter()
            .filter(|(name, _, body, _)| {
                !writers.contains(name)
                    && writers
                        .iter()
                        .any(|writer| contains_code_token(body, &format!("{writer}(")))
            })
            .map(|(name, ..)| name.clone())
            .collect();
        if callers.is_empty() {
            break;
        }
        writers.extend(callers);
    }
    fns.into_iter()
        .filter(|(name, is_pub, ..)| *is_pub && writers.contains(name))
        .map(|(name, ..)| name)
        .collect()
}

/// The oracle: no `jobs` writer of the catalog is named as code in
/// `gang.rs`.
#[test]
fn the_gang_handler_names_no_jobs_writer() {
    let root = repo_root();
    let jobs_repo = std::fs::read_to_string(root.join(JOBS_REPO_RS))
        .unwrap_or_else(|e| panic!("{JOBS_REPO_RS} must be readable: {e}"));
    let writers = jobs_writers(&jobs_repo);
    for known in [
        "fail_job",
        "cancel_job",
        "fill_training_set_identity",
        "claim_next",
    ] {
        assert!(
            writers.contains(known),
            "the derived jobs-writer set must contain `{known}` — the scanner found {writers:?}; \
             a universe that misses a known writer proves nothing"
        );
    }
    let gang = std::fs::read_to_string(root.join(GANG_RS))
        .unwrap_or_else(|e| panic!("{GANG_RS} must be readable: {e}"));
    let masked = mask_non_code(&gang);
    for writer in &writers {
        let token = format!("{writer}(");
        assert!(
            !contains_code_token(&masked, &token),
            "gang.rs names the jobs writer `{token}` as code — the peer writes nothing \
             terminal on behalf of a rank; a session end is a stream event, never a row write"
        );
    }
}

/// The scanner's own self-test: a writer is recognised by a write statement
/// inside its body (case-insensitive, inside the SQL string literal), a
/// read-only fn is not, and a `pub async fn` mentioned only in a comment is
/// not a declaration.
#[test]
fn jobs_writers_scanner_recognises_writes_and_ignores_reads_and_comments() {
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — a writer
    let writer = "pub async fn bump(&self) -> Result<()> { tx.execute(\"UPDATE jobs SET x = 1\", &[]).await }\n";
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — a reader
    let reader = "pub async fn peek(&self) -> Result<()> { tx.query_opt(\"SELECT status FROM jobs\", &[], f).await }\n";
    let commented = "// pub async fn ghost(&self) { \"DELETE FROM jobs\" }\n";
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — a private shared writer
    let shared = "async fn write_end(&self) -> Result<()> { tx.execute(\"UPDATE jobs SET y = 2\", &[]).await }\n";
    // kernel-oracles: fn-in-literal reviewed: fixture string, not real code — writers only by delegation
    let delegating = "pub async fn end(&self) -> Result<()> { self.write_end().await }\n";
    let twice_removed = "pub async fn end_all(&self) -> Result<()> { self.end().await }\n";
    let fixture = format!("{writer}{reader}{commented}{shared}{delegating}{twice_removed}");
    let writers = jobs_writers(&fixture);
    assert_eq!(
        writers,
        BTreeSet::from(["bump".to_string(), "end".to_string(), "end_all".to_string()]),
        "the direct writer and the two that reach a write by delegation — never the reader, \
         the commented-out fn, or the private shared writer itself"
    );
}

/// The masking + boundary self-test: a writer named in a comment or a
/// string is not a code occurrence; a longer identifier ending in the
/// writer's name is not a hit; a real call is.
#[test]
fn contains_code_token_hits_calls_not_comments_strings_or_longer_identifiers() {
    assert!(!contains_code_token(
        &mask_non_code("// fail_job( in prose\nfn f() {}\n"),
        "fail_job("
    ));
    assert!(!contains_code_token(
        // kernel-oracles: fn-in-literal reviewed: fixture string, not real code
        &mask_non_code("fn f() { let s = \"fail_job(\"; }\n"),
        "fail_job("
    ));
    assert!(!contains_code_token(
        // kernel-oracles: fn-in-literal reviewed: fixture string, not real code
        &mask_non_code("fn never_fail_job() {}\n"),
        "fail_job("
    ));
    assert!(contains_code_token(
        // kernel-oracles: fn-in-literal reviewed: fixture string, not real code
        &mask_non_code("fn f() { catalog.fail_job(&id).await }\n"),
        "fail_job("
    ));
}
