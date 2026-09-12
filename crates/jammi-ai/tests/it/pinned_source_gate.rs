//! The DELTA contract's source gate — round 8 (`CONTRACT-DELTA-fix8.md`,
//! closing two measured defects the round-7 audit found by compiling this
//! file's own detector functions into a standalone harness and driving
//! synthetic producers through them).
//!
//! **Round 8, D1 — the escape.** Every round-7 detector required the
//! straddle-shaped function to receive the table's [`ResultTableRecord`] as
//! a PARAMETER. A producer that instead takes a bare table NAME and resolves
//! the record itself through the catalog (`self.catalog.get_result_table(name)`
//! — the prevailing in-tree idiom) before reading `.current_version` off the
//! record it just fetched was invisible to all three: measured empty against
//! every one. This is not hypothetical — [`self_fetched_record_version_hits`]
//! (pattern 4, below) finds `InferenceSession::refreshable_record`
//! (`crates/jammi-ai/src/pipeline/embedding_refresh.rs`) already doing
//! exactly this on the surface as it stands today, undetected by any of the
//! first three patterns.
//!
//! **Round 8, D2 — the parser bug.** [`find_fn_regions`]'s optional generic
//! parameter list used to be matched with the SAME depth-counter
//! [`find_matching`] uses for parens and braces, which does not know that
//! the `>` in a closure/`Fn`-trait bound's `->` return arrow is not a
//! closing angle bracket. A function written
//! `fn f<F: Fn(&str) -> String>(&self, rec: &ResultTableRecord) -> Result<InputAnchor>`
//! had its generic list close at the arrow, which desynchronized the scan
//! from `(` and dropped the WHOLE function — invisible to every detector,
//! including the anchor-shaped-return one, despite a return type that is
//! literally `Result<InputAnchor>`. [`find_matching_angle`] now treats a
//! `->` as a single unit that never changes angle-bracket depth. Checked,
//! not assumed: an independent `fn <ident>` token count over today's surface
//! (4136) equals the number of regions `find_fn_regions` produces both
//! before and after this fix, so no live function was affected — the bug
//! was latent, not live, but it was a silent fail-OPEN in the shared parsing
//! layer of what round 7 made the sole enforcement, and the comment that
//! used to sit above the old match asserted the opposite.
//!
//! **The ruling this file exists to satisfy.** Six rounds tried to close the
//! straddle class — a producer that resolves a result table's version once
//! for its provenance anchor and again for its content, so a version publish
//! landing between the two calls makes a durable artifact whose provenance
//! names one version while its rows came from another — by arguing the shape
//! was "closed by construction." Every round was wrong, most recently in
//! three lines of published API (`ResultStore::current_anchor`) one module
//! over from the one the hand-written sweep enumerated. The two artifacts
//! that were supposed to prevent the class each delegated it to the other:
//! this gate walked one directory of one crate, and a prose sweep on
//! [`jammi_db::store::ResultStore::pin_current_version`] enumerated "every
//! `pub`/`pub(crate)` function in this module" for a property that says "no
//! public interface" — and missed members six rounds running, including
//! three sites the unit's own plan document had already listed together as
//! one reader class.
//!
//! **What changed here.** The prose sweep is DELETED, not corrected — it is
//! the artifact that failed six times, and a human enumeration beside a
//! machine one guarantees a seventh round arguing which is authoritative.
//! This file is now the only enforcement, and its quantifier is derived from
//! version control, not typed by hand: every git-TRACKED `.rs` file under
//! `crates/jammi-db/src` and `crates/jammi-ai/src`, recursively (`git
//! ls-files` itself recurses, so there is no second, hand-rolled directory
//! walk that could silently stop at one level the way round 5's did), is
//! read and scanned by every check below. A file `git ls-files` lists that
//! this process cannot then read is a hard failure naming the file, not a
//! silently skipped one.
//!
//! Whether a function is `pub` or crate-private is NOT part of any check
//! here (the lead's round-7 ruling: a visibility keyword protects nobody in
//! a greenfield crate with no external consumers, and six rounds of chasing
//! "constructible from published surface" produced keyword churn instead of
//! closing anything). Every detector below scans every function regardless
//! of visibility.
//!
//! **The four patterns, not one string literal.** Each is a shape that
//! reproduces the straddle, independent of which public/private API a
//! producer happens to use:
//!
//! 1. [`anchor_shaped_return_hits`] — a function whose return type carries
//!    [`jammi_db::store::manifest::InputAnchor`] or
//!    [`jammi_db::store::CurrentAnchor`] verbatim: a version-resolved anchor
//!    value with no content attached, exactly what let
//!    `current_anchor`/`result_digest_anchor` escape as a "return-type"
//!    clearance that was never a value clearance.
//! 2. [`bare_record_version_branch_hits`] — a function that takes a bare
//!    [`jammi_db::catalog::result_repo::ResultTableRecord`] (by value or by
//!    reference — this is a source-text check, not a type-resolution one, so
//!    a fully-qualified path such as
//!    `crate::catalog::result_repo::ResultTableRecord` still matches, since
//!    the identifier `ResultTableRecord` occurs in the parameter text either
//!    way) and re-derives `.current_version` from it inside its own body,
//!    rather than taking an already-resolved version/manifest/[`PinnedSource`]
//!    as a parameter — the shape that let `read_vectors` reach an unpinned,
//!    version-branched content read through the session's own registration,
//!    outside the module the old sweep ever looked at.
//! 3. [`session_registration_literal_sites`] — the session-registered
//!    `jammi.{table}` reference spelled out directly in source text (the
//!    original round-3/5/6 gate's own check), now over the WHOLE two-crate
//!    surface rather than one crate's `src/pipeline/` directory, and (round
//!    8, D3) bound to its enclosing function rather than to the file as a
//!    whole.
//! 4. [`self_fetched_record_version_hits`] (round 8, D1) — a function that
//!    does NOT take a [`jammi_db::catalog::result_repo::ResultTableRecord`]
//!    as a parameter (pattern 2's precondition) but instead resolves one
//!    itself in its own body via the catalog idiom `.get_result_table(` and
//!    then reads the `.current_version` FIELD off the record it just
//!    fetched — the parameter shape and the self-fetch shape partition the
//!    surface between patterns 2 and 4 rather than overlapping it. This is
//!    pattern 2's own precondition inverted, closing exactly the gap the
//!    round-7 audit measured: all three of the original detectors return
//!    empty on this shape.
//!
//! **What this file does NOT claim.** It is source-text pattern matching
//! over a hand-written (but string/char/comment-literal-aware — see
//! [`mask_non_code`]) approximation of Rust's grammar, not a real parser and
//! not a call-graph or dataflow analysis. Stated failure modes, per R-A:
//! nested (non-doc) block comments are treated as non-nesting (the first
//! `*/` closes them — Rust allows `/* /* */ */` to nest; this crate's source
//! was checked and contains no such nesting today, but a future one would
//! have its interior treated as code); a function-pointer type parameter
//! written with unconventional spacing (`fn (i32) -> bool`, a space after
//! `fn`) would be mistaken for a function item — checked: `grep -rn "fn (["
//! crates/jammi-db/src crates/jammi-ai/src` finds none; and detectors 2 and
//! 4 do not follow a value ACROSS function boundaries (a helper that reads
//! `.current_version` — off a parameter for pattern 2, off a self-fetched
//! record for pattern 4 — and hands the bare version to a second,
//! separately-reviewed function is invisible to that detector unless the
//! second function is itself reviewed — which is why every hit, closed or
//! not, is enumerated in an `ALLOWED` list below with its own review note,
//! rather than silently passing). Pattern 4 additionally requires an EXACT
//! field-boundary match on `.current_version` (not merely the substring —
//! see [`reads_current_version_field`]'s doc) precisely so this cross-
//! function case (e.g. a self-fetching function that only calls
//! `.current_version_identity(...)`, never reads the field itself) is not
//! mistaken for a direct field read by coincidence of one identifier
//! prefixing another.
//!
//! **What this file's surface is NOT (round 8, D4 — disclosure, not a
//! claim of absolute fail-closed).** [`tracked_rs_files`] derives the
//! scanned surface from `git ls-files`, so a new `.rs` file present on disk
//! but not yet `git add`ed is invisible to every check here — this process
//! passes locally on code it has never read. The "hard failure naming the
//! file" property below is about a file `git ls-files` DOES list that this
//! process then cannot read (deleted on disk without being staged, or a
//! worktree race), not about a file `git ls-files` never lists at all. In
//! continuous integration the working tree IS the committed tree, so every
//! `.rs` file under either surface directory is necessarily tracked and this
//! gap does not exist there; this note is about a local `cargo test` run
//! against an uncommitted new file, not about the enforcement CI relies on.
//!
//! Every entry in every `ALLOWED` list below is a site this gate's own scan
//! finds TODAY (verified by running the scan without an allowlist and
//! transcribing every hit — the allowlists are not a guess at what might
//! exist). A hit not in its list is RED. An `ALLOWED` entry whose site no
//! longer produces that hit (the code was fixed or removed) is ALSO RED
//! (`allowlists_match_current_hits_exactly`, below) — an allowance is never
//! permanent slack a later, different site can spend.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The whole surface this gate's property quantifies over
/// (`CONTRACT-DELTA-fix7.md`: "every Rust source in `crates/jammi-db/src`
/// and `crates/jammi-ai/src`"), relative to the repo root.
const SURFACE_DIRS: &[&str] = &["crates/jammi-db/src", "crates/jammi-ai/src"];

/// The repo root, derived from this crate's manifest dir
/// (`crates/jammi-ai`) rather than the process's `cwd` — `cargo test` can be
/// invoked from anywhere, but `CARGO_MANIFEST_DIR` is always this crate's
/// directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-TRACKED `.rs` file under `root`-relative `dir`, sorted. `git
/// ls-files` recurses on its own — this is deliberately NOT a hand-rolled
/// `read_dir` walk, because that is exactly the shape (round 5, M3) that
/// silently stopped at one directory level and never scanned
/// `src/pipeline/asof/`. There is nothing here for a "did the walk reach
/// every tracked file" sentinel to check, because there is no second walk
/// to disagree with the tracked list — the tracked list IS what is scanned.
fn tracked_rs_files(root: &Path, dir: &str) -> Vec<String> {
    let output = Command::new("git")
        .args([
            "-C",
            root.to_str().expect("utf8 repo root"),
            "ls-files",
            "--",
            dir,
        ])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- {dir} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut files: Vec<String> = String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(".rs"))
        .map(str::to_string)
        .collect();
    files.sort();
    files
}

/// The whole scanned surface: every tracked `.rs` file under every
/// [`SURFACE_DIRS`] entry, as `(repo-relative path, source text)`. Reading a
/// file `git ls-files` reports as tracked is not optional — a tracked file
/// this process cannot read (deleted on disk without being staged, or a
/// worktree race) is a hard failure naming the file, never a silent skip,
/// per `CONTRACT-DELTA-fix7.md`'s "failing closed" requirement.
fn scan_surface() -> Vec<(String, String)> {
    let root = repo_root();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for dir in SURFACE_DIRS {
        let files = tracked_rs_files(&root, dir);
        assert!(
            !files.is_empty(),
            "git ls-files -- {dir} returned no tracked .rs files — the repo root, or the \
             pathspec, is wrong, which would make every check below vacuously pass"
        );
        for rel in files {
            assert!(
                seen.insert(rel.clone()),
                "{rel} is tracked under more than one of {SURFACE_DIRS:?} — the surface dirs \
                 must be disjoint or this gate double-counts it"
            );
            let text = std::fs::read_to_string(root.join(&rel)).unwrap_or_else(|e| {
                panic!(
                    "{rel} is git-tracked but could not be read ({e}) — this gate fails closed \
                     rather than silently scanning fewer files than `git ls-files` reports"
                )
            });
            out.push((rel, text));
        }
    }
    out
}

// ── A comment/string/char-literal-aware mask, so brace/paren counting and
// pattern search never mistake a `format!("jammi.{}")`'s own braces, or a
// doc comment's prose, for code. ──────────────────────────────────────────

/// Replace every line comment, block comment, string literal (plain and
/// raw), and char literal in `text` with spaces — same length, same
/// newlines, so every downstream line/column number still matches the
/// original file, and brace/paren counting on the result never miscounts a
/// `{`/`}` that appears inside a string (e.g. `format!("jammi.{}", ..)`) or
/// treats commented-out code as live.
///
/// **Stated limit (R-A):** block comments are treated as non-nesting (the
/// first `*/` closes one opened by `/*`, even though Rust itself nests
/// them). Checked rather than assumed: `grep -rn '/\*.*/\*' crates/jammi-db/src
/// crates/jammi-ai/src` finds no nested block comment in this surface today,
/// so this limit is inert on the current tree; a future nested block
/// comment would have its interior treated as code, which could only ever
/// make a detector below fire MORE often (spurious code seen inside a dead
/// comment), never mask a real hit.
fn mask_non_code(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<char> = chars.clone();
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        // Line comment: `//` to end of line.
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            let mut j = i;
            while j < n && chars[j] != '\n' {
                out[j] = ' ';
                j += 1;
            }
            i = j;
            continue;
        }
        // Block comment: `/* ... */`, non-nesting (see doc above).
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
        // Raw string: `r`/`r#`.../`r###..."`, matched against the same
        // number of trailing `#` after the closing `"`.
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
        // Plain string literal: `"..."`, with `\`-escapes.
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
        // Char literal, distinguished from a lifetime (`'a`, `'static`) by
        // requiring a closing `'` within an escape's width: `'\''`, `'\n'`,
        // `'\u{2019}'`, or a plain `'x'`.
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

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Find the index just past the character matching `chars[open_idx]`
/// (assumed to be `open`), by depth-counting `open`/`close` over `chars`
/// starting at `open_idx`. `chars` must already be comment/string/char
/// masked, so a `(`/`)`/`{`/`}` inside a literal never perturbs the count.
fn find_matching(chars: &[char], open_idx: usize, open: char, close: char) -> Option<usize> {
    debug_assert_eq!(chars[open_idx], open);
    let mut depth = 1i64;
    let mut j = open_idx + 1;
    while j < chars.len() {
        if chars[j] == open {
            depth += 1;
        } else if chars[j] == close {
            depth -= 1;
            if depth == 0 {
                return Some(j + 1);
            }
        }
        j += 1;
    }
    None
}

/// Find the index just past the `>` matching the `<` at `chars[open_idx]`
/// (a generic parameter list), depth-counting `<`/`>` the way [`find_matching`]
/// counts parens/braces — with ONE exception (round 8, D2): the `>` of a
/// `->` return arrow, e.g. a closure/`Fn`-trait bound written
/// `Fn(&str) -> String`, is never treated as a closing angle bracket. Before
/// this exception existed, `find_matching(chars, m, '<', '>')` closed the
/// list at that arrow, desynchronized the scan from the following `(`, and
/// dropped the WHOLE function from [`find_fn_regions`]'s output — invisible
/// to every detector below even when its own return type carried
/// `InputAnchor` verbatim. Reachability was checked, not assumed: an
/// independent `fn <ident>` token count over the real surface (4136) equals
/// the number of regions produced with or without this fix, so no function
/// on today's tree used this shape — the bug was latent, not live — but it
/// was a silent fail-OPEN in the only enforcement, so it is fixed rather
/// than left as a disclosed limit.
fn find_matching_angle(chars: &[char], open_idx: usize) -> Option<usize> {
    debug_assert_eq!(chars[open_idx], '<');
    let mut depth = 1i64;
    let mut j = open_idx + 1;
    while j < chars.len() {
        if chars[j] == '-' && j + 1 < chars.len() && chars[j + 1] == '>' {
            // The arrow is one token; skip both characters without
            // touching depth so its `>` can never close the list early.
            j += 2;
            continue;
        }
        if chars[j] == '<' {
            depth += 1;
        } else if chars[j] == '>' {
            depth -= 1;
            if depth == 0 {
                return Some(j + 1);
            }
        }
        j += 1;
    }
    None
}

/// One `fn` item found in a masked source: its name, 1-based source line of
/// the `fn` keyword, the 1-based source line of its LAST character (the
/// closing `}` for a function with a body, or the terminating `;` for a
/// trait declaration — used to bind a whole-file textual hit, such as a
/// session-registration literal, to its enclosing function rather than to
/// the file as a whole; see [`session_registration_literal_sites`]), its
/// parameter-list text (masked), and — when it has a body rather than a
/// trait-declaration `;` — its return-type text and its body text (both
/// masked).
struct FnRegion {
    name: String,
    line: usize,
    end_line: usize,
    params: String,
    return_type: Option<String>,
    body: Option<String>,
}

/// Every `fn` item in `masked` (an `impl`/free/trait/nested function — this
/// is deliberately unfiltered by visibility or nesting, per this file's
/// module doc: a visibility keyword is not part of any property this gate
/// checks). Source-text scanning, not a parser: see [`mask_non_code`]'s doc
/// for the stated limits this inherits.
fn find_fn_regions(masked: &str) -> Vec<FnRegion> {
    let chars: Vec<char> = masked.chars().collect();
    let n = chars.len();
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i + 1 < n {
        // `fn` as a whole word, followed by whitespace then an identifier —
        // excludes a `Fn(...)` trait-bound type (capital F) and a
        // function-pointer type written `fn(i32)` with no space (checked
        // absent from this surface, see the module doc).
        let boundary_ok = i == 0 || !is_ident_char(chars[i - 1]);
        if boundary_ok
            && chars[i] == 'f'
            && chars[i + 1] == 'n'
            && i + 2 < n
            && chars[i + 2].is_whitespace()
        {
            let fn_start = i;
            let mut k = i + 2;
            while k < n && chars[k].is_whitespace() {
                k += 1;
            }
            let name_start = k;
            while k < n && is_ident_char(chars[k]) {
                k += 1;
            }
            if k > name_start {
                let name: String = chars[name_start..k].iter().collect();
                let mut m = k;
                while m < n && chars[m].is_whitespace() {
                    m += 1;
                }
                // Optional generic parameter list `<...>` — see
                // `find_matching_angle`'s doc for why this is NOT the same
                // depth-counter `find_matching` uses for parens/braces (a
                // `->` inside a `Fn` bound must not close it early).
                if m < n && chars[m] == '<' {
                    if let Some(after) = find_matching_angle(&chars, m) {
                        m = after;
                        while m < n && chars[m].is_whitespace() {
                            m += 1;
                        }
                    }
                }
                if m < n && chars[m] == '(' {
                    if let Some(params_end) = find_matching(&chars, m, '(', ')') {
                        let params: String = chars[m + 1..params_end - 1].iter().collect();
                        // Scan forward for the first `;` or `{` at paren
                        // depth 0 — the signature terminator. A tuple
                        // return type or a `where` clause's own parens are
                        // depth-tracked so they cannot be mistaken for the
                        // terminator.
                        let mut depth = 0i64;
                        let mut t = params_end;
                        let mut terminator = None;
                        while t < n {
                            match chars[t] {
                                '(' => depth += 1,
                                ')' => depth -= 1,
                                ';' | '{' if depth == 0 => {
                                    terminator = Some(t);
                                    break;
                                }
                                _ => {}
                            }
                            t += 1;
                        }
                        if let Some(term) = terminator {
                            let sig: String = chars[params_end..term].iter().collect();
                            let return_type = sig.split_once("->").map(|(_, rt)| rt.to_string());
                            // The region's own end: one past the closing
                            // `}` for a function with a body, or one past
                            // the terminating `;` for a trait declaration.
                            // A brace this file's own `find_matching` cannot
                            // match (malformed/unclosed source) falls back
                            // to end-of-file rather than leaving the region
                            // unbounded, so a downstream line-range lookup
                            // (`session_registration_literal_sites`) never
                            // panics on an out-of-order range.
                            let (body, region_end) = if chars[term] == '{' {
                                match find_matching(&chars, term, '{', '}') {
                                    Some(end) => {
                                        (Some(chars[term + 1..end - 1].iter().collect()), end)
                                    }
                                    None => (None, n),
                                }
                            } else {
                                (None, term + 1)
                            };
                            let line = chars[..fn_start].iter().filter(|c| **c == '\n').count() + 1;
                            let end_line = chars[..region_end.min(n)]
                                .iter()
                                .filter(|c| **c == '\n')
                                .count()
                                + 1;
                            regions.push(FnRegion {
                                name,
                                line,
                                end_line,
                                params,
                                return_type,
                                body,
                            });
                        }
                    }
                }
            }
        }
        i += 1;
    }
    regions
}

/// One hit of pattern 1 (`crates/jammi-ai/tests/it/pinned_source_gate.rs`
/// module doc's list) — a function whose return type carries `InputAnchor`
/// or `CurrentAnchor` verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hit {
    file: String,
    name: String,
    line: usize,
}

fn anchor_shaped_return_hits(surface: &[(String, String)]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        for region in find_fn_regions(&masked) {
            if let Some(rt) = &region.return_type {
                if rt.contains("InputAnchor") || rt.contains("CurrentAnchor") {
                    hits.push(Hit {
                        file: file.clone(),
                        name: region.name,
                        line: region.line,
                    });
                }
            }
        }
    }
    hits
}

fn bare_record_version_branch_hits(surface: &[(String, String)]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        for region in find_fn_regions(&masked) {
            if region.params.contains("ResultTableRecord") {
                if let Some(body) = &region.body {
                    if body.contains(".current_version") {
                        hits.push(Hit {
                            file: file.clone(),
                            name: region.name,
                            line: region.line,
                        });
                    }
                }
            }
        }
    }
    hits
}

/// True when `body` reads the `.current_version` FIELD, not merely a longer
/// identifier that happens to start with the same text — `.current_version`
/// is also a PREFIX of `.current_version_identity(` and
/// `.current_version_provider(`, two already-reviewed helper methods
/// (`RECORD_VERSION_BRANCH_ALLOWED`'s own entries) that a self-fetching
/// caller can delegate to without ever reading the field itself (the
/// disclosed cross-function limit this file's module doc names for pattern
/// 4). A field access is never followed by an identifier character; a
/// longer method name always is. Without this boundary check,
/// `self_fetched_record_version_hits` would flag `ResultStore::current_anchor`
/// (which only calls `self.current_version_identity(&parent)`, never reads
/// the field) as a false positive of pattern 4 — checked directly: with the
/// boundary check, it does not appear in `self_fetched_record_version_hits`'s
/// output; a bare `.contains(".current_version")` substring check does flag
/// it.
fn reads_current_version_field(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let needle: Vec<char> = ".current_version".chars().collect();
    let n = chars.len();
    let m = needle.len();
    if n < m {
        return false;
    }
    for i in 0..=(n - m) {
        if chars[i..i + m] == needle[..] && !(i + m < n && is_ident_char(chars[i + m])) {
            return true;
        }
    }
    false
}

/// Pattern 4 (round 8, D1) — the escape the round-7 audit measured: a
/// function that does NOT take a bare `ResultTableRecord` as a parameter
/// (pattern 2's own precondition, excluded here via `!region.params.contains(
/// "ResultTableRecord")` so patterns 2 and 4 partition the surface instead
/// of double-flagging the same site under two different names) but instead
/// resolves one itself, in its own body, via the catalog idiom
/// `.get_result_table(` — the prevailing in-tree idiom, not this crate's
/// only one, but the one every known instance of this shape uses — and then
/// reads the resolved record's `.current_version` FIELD (see
/// [`reads_current_version_field`]'s doc for why this must be a field match,
/// not a substring one). See this file's module doc for why patterns 1–3
/// alone cannot see this shape.
fn self_fetched_record_version_hits(surface: &[(String, String)]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        for region in find_fn_regions(&masked) {
            if region.params.contains("ResultTableRecord") {
                continue;
            }
            if let Some(body) = &region.body {
                if body.contains(".get_result_table(") && reads_current_version_field(body) {
                    hits.push(Hit {
                        file: file.clone(),
                        name: region.name,
                        line: region.line,
                    });
                }
            }
        }
    }
    hits
}

/// Non-comment-line occurrences of the session-registered literal
/// `"jammi.{` (the ingredient common to `TableReference::bare(format!(
/// "jammi.{table}"))`, `ctx.table("jammi.{table}")`, and a raw
/// `.sql(&format!("... \"jammi.{table}\" ..."))`), bound to the enclosing
/// FUNCTION rather than to the file as a whole (round 8, D3 — the round-7
/// audit's advisory: a per-file `usize` allowance is fungible across every
/// site inside that file, so a NEW unpinned read added to an already-
/// allowlisted file passes review-free as long as an existing one is
/// deleted in the same commit; the site was discarded before the old
/// per-file allowlist ever saw it). This is the same site-binding
/// [`anchor_shaped_return_hits`] and [`bare_record_version_branch_hits`]
/// already use, keyed on `(file, function name)` instead of `file` alone.
///
/// The narrowing itself is unchanged from round 7: deliberately narrower
/// than a bare `"jammi.` substring check — `"jammi.toml"`,
/// `"jammi.audit.search.v1"`, `"jammi.topic.{}.batch"` and this crate's
/// other domain-separator/config literals all contain `"jammi.` but are
/// never followed immediately by `{` — only a session table-reference
/// literal spells the table name as an interpolation directly after the
/// dot. Verified against every `"jammi.` occurrence in this surface
/// (`grep -rn '"jammi\.' crates/jammi-db/src crates/jammi-ai/src`) before
/// narrowing: every non-`"jammi.{` hit is one of the domain/config literals
/// above, and every session-registration site this file's own module doc
/// and the round-6 audit named is `"jammi.{`.
///
/// Attribution runs on the ORIGINAL, unmasked text (the literal itself is a
/// string, which [`mask_non_code`] would blank), using [`find_fn_regions`]'s
/// `(line, end_line)` only to find which function's line range contains a
/// given hit line — the innermost (smallest-range) containing region wins,
/// so a hit inside a nested function is never double-counted against its
/// enclosing one too. A hit whose line falls inside no region at all (a
/// module-level literal, which does not occur on today's surface) is bound
/// to the sentinel site `"<module-scope>"`, which no `ALLOWED` entry ever
/// names, so it fails loudly rather than being silently mis-attributed.
fn session_registration_literal_sites(
    surface: &[(String, String)],
) -> std::collections::HashMap<(String, String), usize> {
    let mut counts = std::collections::HashMap::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        let regions = find_fn_regions(&masked);
        for (line_idx, line) in text.lines().enumerate() {
            let line_no = line_idx + 1;
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            let hits = line.matches("\"jammi.{").count();
            if hits == 0 {
                continue;
            }
            let mut best: Option<&FnRegion> = None;
            for region in &regions {
                if region.line <= line_no && line_no <= region.end_line {
                    let is_smaller = match best {
                        None => true,
                        Some(b) => (region.end_line - region.line) < (b.end_line - b.line),
                    };
                    if is_smaller {
                        best = Some(region);
                    }
                }
            }
            let site = best
                .map(|r| r.name.clone())
                .unwrap_or_else(|| "<module-scope>".to_string());
            *counts.entry((file.clone(), site)).or_insert(0) += hits;
        }
    }
    counts
}

// ── Allowlists. Every entry is a hit this gate's scan finds TODAY (verified
// by running each detector with an empty allowlist and transcribing every
// hit), never a guess. Each carries its own review note — a bare list of
// paths is not a review, it is the same failure mode this contract exists
// to close for the sweep it replaces. ─────────────────────────────────────

/// Pattern 1 — `(file, function name)`.
const ANCHOR_RETURN_ALLOWED: &[(&str, &str)] = &[
    (
        "crates/jammi-db/src/store/mod.rs",
        "input_anchor",
        // `PinnedSource::input_anchor` — SAFE BY CONSTRUCTION: only reachable
        // through a `&PinnedSource`, which already carries the record and
        // (for a versioned table) the manifest `ResultStore::pinned_provider`
        // reads from — the SAME resolution. Extracting the anchor from an
        // already-pinned source cannot itself create a second, independent
        // resolution; the risk this gate exists to catch is a caller that
        // gets an anchor WITHOUT ALSO holding the paired read, which this
        // accessor structurally cannot produce.
    ),
    (
        "crates/jammi-db/src/store/freshness.rs",
        "current_anchor",
        // `ResultStore::current_anchor` — NOT CLOSED, disclosed rather than
        // hidden (round 6's audit, BLOCK finding 1). Its only in-tree
        // callers (`freshness.rs`'s own staleness comparison, same file) use
        // the returned `CurrentAnchor::ResultDigest(String)` for an
        // equality check against a RECORDED anchor and then discard it —
        // never persist it as a new artifact's provenance. But its input is
        // publicly mintable from a bare table name
        // (`InputAnchor::result_digest` over `ArtifactDigest(pub String)`),
        // and the value it returns is byte-identical to
        // `PinnedSource::input_anchor`'s on both arms. Nothing in the type
        // system stops a FUTURE caller from pairing this anchor with an
        // independently-resolved read and persisting the pair — the exact
        // straddle this contract names. Closing that (narrow the type so it
        // cannot be separated from content, or fold this crate's callers
        // onto `pin_current_version`) is out of round 7's scope (the ruling:
        // "Folds, small ... Nothing else"); this entry is the gate's record
        // that the residual is real, watched, and not silently absorbed.
        //
        // **Round 8, D3 disclosure:** the "only in-tree callers" claim above
        // is a POINT-IN-TIME grep result (re-verified at round 8:
        // `grep -rn 'current_anchor(' crates/jammi-db/src crates/jammi-ai/src`
        // still shows only `freshness.rs`'s own `staleness`), not a
        // machine-checked invariant — nothing in this file re-runs that grep
        // or otherwise re-derives it, so a future commit that adds a second
        // caller would not turn any test here red. This allowlist entry
        // clears the SHAPE (an anchor-shaped return), not the callers; the
        // callers claim is carried in prose, in an artifact whose purpose is
        // to replace prose claims with machine ones. Cheaply machine-
        // checking "no OTHER caller persists this value" would require call-
        // graph analysis this file's source-text matching does not attempt
        // (see the module doc's "not a call-graph or dataflow analysis").
    ),
    (
        "crates/jammi-ai/src/pipeline/graph_propagation.rs",
        "edge_source_anchor",
        // Delegates to `pin_current_version(record).await?.input_anchor()`
        // — the sanctioned pattern (the same one every `result_digest_anchor`
        // caller was migrated to in round 6), not an independent anchor-only
        // resolve. The residual here is one level up, at this function's own
        // caller: `edge_scan_sql` reads the SAME edge table's content
        // through an unpinned, session-registered scan
        // (`session_registration_literal_sites`'s allowlist entry for this
        // same file), so the anchor and the edge content are NOT from one
        // resolution. This is the already-disclosed, reviewed exception for
        // the S9 edge relation (never the pinned embedding table) from the
        // DELTA round-4 contract (M2's `graph_propagation.rs:816` carve-out) —
        // carried forward here rather than re-litigated.
    ),
];

/// Pattern 2 — `(file, function name)`.
const RECORD_VERSION_BRANCH_ALLOWED: &[(&str, &str)] = &[
    (
        "crates/jammi-db/src/store/mod.rs",
        "current_version_identity",
        // The anchor leg `current_anchor` calls (crate-private in name, not
        // in the property this gate checks): same residual, see
        // `ANCHOR_RETURN_ALLOWED`'s `current_anchor` entry — its only
        // callers are same-crate freshness comparisons, never a persisted
        // anchor.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "verify_materialization",
        // Read-only integrity check: compares a version's RECORDED identity
        // against a freshly recomputed one and reports a `MatchVerdict`. It
        // never persists a new anchor or a new artifact.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "resolve_search_mode_local",
        // Disclosed, not closed (M4, carried from the deleted sweep):
        // candidate SELECTION (which rows a producer's pooled read gathers)
        // is out of this contract's scope. `pin_current_version`'s own doc
        // states the residual: a pinned producer's POOLED VECTORS are
        // single-version by construction, but its MEMBER SET may have been
        // chosen from a different, unpinned view via this function.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "bind_result_table",
        // Documented "Read class" residual on its own doc comment: serves a
        // possibly-stale session-bound registration, never persists an
        // anchor. Every producer that DOES persist an anchor is required
        // (by that same doc) to route through `pin_current_version` /
        // `pinned_provider` instead.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "current_version_provider",
        // Private; its only caller is `pinned_provider`'s own unversioned
        // arm (verified: `grep -n 'current_version_provider' crates/jammi-db/src`
        // shows one call site, `store/mod.rs`'s `pinned_provider`). Reading
        // `.current_version` off the SAME `pin.record` `pinned_provider` was
        // itself called with cannot straddle anything — there is no second,
        // independent resolution here.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "pin_current_version",
        // The seam itself: reads `record.current_version` once to decide
        // which arm to take, then returns a `PinnedSource` that carries the
        // record, the resolved version, and (for a versioned table) the
        // manifest from that SAME resolution — safe by construction, not by
        // convention.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "allocate_version",
        // Reads `table.current_version` only as the CAS's EXPECTED PARENT
        // (refuses with `ParentMoved` on mismatch); never reads content
        // under a version it resolves itself.
    ),
    (
        "crates/jammi-db/src/store/freshness.rs",
        "producing_descriptor",
        // NOT CLOSED, disclosed (round 6's audit named this alongside
        // `current_anchor` and `read_vectors` as the plan's own reader
        // class, `DELTA-INCREMENTAL-EMBEDDING.md:180`). Its callers
        // (`embedding_refresh.rs`'s `refresh_embeddings`,
        // `recompute.rs`'s `recompute_one`) use the returned
        // `ProducingDescriptor` only to select WHICH producer verb/params to
        // replay — never as content or as a persisted anchor — and each
        // producer that then materializes a new artifact performs its own,
        // independent `pin_current_version` resolution for that artifact's
        // actual anchor and rows. A version drift between this read and that
        // later pin could select a stale REPLAY TARGET, never corrupt a
        // persisted anchor/content pairing. Closing the shape itself (this
        // function still re-derives `table.current_version` from a bare
        // record) is out of round 7's scope; recorded here rather than
        // silently absorbed.
        //
        // **Round 8, D3 disclosure:** the "callers use it only to select a
        // replay target" claim is a POINT-IN-TIME grep result (re-verified
        // at round 8: `grep -rn 'producing_descriptor(' crates/jammi-db/src
        // crates/jammi-ai/src` shows exactly the two callers named above),
        // not a machine-checked invariant this file re-derives on every run
        // — see `current_anchor`'s entry above for the same disclosure and
        // why closing it would need call-graph analysis this file does not
        // attempt.
    ),
    (
        "crates/jammi-db/src/session.rs",
        "read_vectors",
        // NOT CLOSED, disclosed (round 6's audit, the third plan-listed
        // reader-class member: `DELTA-INCREMENTAL-EMBEDDING.md:180`). The
        // versioned arm reads content through this SESSION's own
        // `jammi.{table}` registration (see
        // `session_registration_literal_sites`'s allowlist entry for this
        // same file) rather than through `pinned_provider` — an unpinned,
        // version-branched content read, re-exported publicly at
        // `jammi-ai/src/session.rs:1018` and
        // `jammi-ai/src/local_session.rs:363`. No in-tree caller currently
        // pairs this read with an independently-resolved anchor to persist
        // as provenance (verified: `grep -rn '\.read_vectors(' crates/jammi-db/src
        // crates/jammi-ai/src` — every caller is a plain vector read, not a
        // provenance-persisting producer). Closing this (route the versioned
        // arm through `pin_current_version`/`pinned_provider`, or require a
        // caller-supplied `PinnedSource`) is out of round 7's scope; the
        // straddle this function makes constructible is exactly the class
        // this contract names as real and unclosed.
        //
        // **Round 8, D3 disclosure:** re-verified rather than re-argued
        // (the grep above was re-run at round 8 with the same two-caller
        // result), but still a POINT-IN-TIME claim, not a machine-checked
        // one — nothing here re-runs it on a future commit; see
        // `current_anchor`'s entry above for the general shape of this
        // disclosure.
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "ensure_base_version",
        // Guard clause only: `record.current_version.is_some()` short-
        // circuits to "already versioned, return unchanged" — no content
        // read branches on the value; the function that DOES read content
        // for the base publish path is gated separately.
    ),
];

/// Pattern 4 (round 8, D1) — `(file, function name)`, same shape as
/// `ANCHOR_RETURN_ALLOWED`/`RECORD_VERSION_BRANCH_ALLOWED`.
const SELF_FETCHED_RECORD_ALLOWED: &[(&str, &str)] = &[(
    "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
    "refreshable_record",
    // `InferenceSession::refreshable_record` — step 0's readiness GATE,
    // and the round-8 audit's own escape shape found LIVE on this
    // surface: it self-fetches the record from a bare table name via
    // `self.catalog().get_result_table(table)`, then reads
    // `record.current_version` to reject a table whose CURRENT version
    // row is not `ready` (`NotRefreshableReason::CurrentVersionUnavailable`).
    // That read is used ONLY for this readiness check — it is never
    // returned, never becomes an anchor, and never pairs with a content
    // read here. Its one in-tree caller (`refresh_embeddings`) performs
    // every later anchor/content pairing from a SEPARATE, later
    // resolution: `ensure_base_version` re-fetches the record itself
    // (`RECORD_VERSION_BRANCH_ALLOWED`'s own entry for that function),
    // and `refresh_embeddings` reads `parent_version` off THAT fresh
    // record, not this one's. A version publish landing between this
    // gate and that later resolution can only make this gate stale
    // (reject a table that just became refreshable, or admit one whose
    // checked version has since moved and gets re-validated downstream
    // by `resolve_version_manifest`) — never mint a mismatched
    // anchor/content pair. Same disclosed-residual shape as
    // `bind_result_table`'s "Read class" note, not closed by this round.
)];

/// Pattern 3 — `(file, function name, allowed occurrence count)`. Keyed on
/// the (path, function) SITE (round 8, D3 — the round-7 audit's advisory: a
/// per-file `usize` allowance let a NEW unpinned read inside an already-
/// allowlisted file pass review-free whenever an existing one in a
/// DIFFERENT function of that same file was deleted in the same commit),
/// never a bare file name (round 6 advisory: a same-named file in a
/// different subdirectory must not silently inherit an allowance reviewed
/// for a wholly different file).
const SESSION_LITERAL_ALLOWED: &[(&str, &str, usize)] = &[
    (
        "crates/jammi-ai/src/pipeline/graph_propagation.rs",
        "edge_scan_sql",
        1,
        // The S9 `neighbor_graph` edge scan — the EDGE relation, never the
        // pinned embedding table `PinnedSource` covers. Reviewed in the
        // DELTA round-4 contract (M2's `graph_propagation.rs:816` carve-out);
        // see this same file's `edge_source_anchor` note in
        // `ANCHOR_RETURN_ALLOWED` for the anchor/content pairing this
        // residual leaves open.
    ),
    (
        "crates/jammi-ai/src/pipeline/graph_neighbourhood.rs",
        "load_neighbor_graph_edges",
        1,
        // Same class, the S9 edge relation.
    ),
    (
        "crates/jammi-db/src/index/exact.rs",
        "exact_vector_search",
        1,
        // The exact-match ANN fallback, reading THIS session's own
        // registration. `pin_current_version`'s own doc names this exact
        // function as the disclosed "candidate SELECTION is not pinned"
        // residual (M4): a pinned producer's pooled vectors are single-
        // version, but the candidate set this function returns may have
        // been chosen from a different, unpinned view.
    ),
    (
        "crates/jammi-db/src/session.rs",
        "read_vectors",
        1,
        // See `RECORD_VERSION_BRANCH_ALLOWED`'s entry for this same
        // function — builds a `TableReference::bare(format!("jammi.{table}"))`
        // to read through this session's own registration rather than a
        // pin. Disclosed, not closed.
    ),
    (
        "crates/jammi-db/src/session.rs",
        "read_vector_by_key",
        1,
        // Same shape as `read_vectors`, a different function in the same
        // file — kept as its own site so the two allowances cannot be
        // spent interchangeably.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "register_table",
        1,
        // The registration write itself — defines what `jammi.{name}` maps
        // to, never a read.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "bind_result_table",
        2,
        // Its two `add_result_table` calls (its documented "Read class"
        // residual — see `RECORD_VERSION_BRANCH_ALLOWED`'s entry). Both
        // sites live in this one function, so the site-bound count here is
        // 2, not 1 — round 8 binds the allowance to the FUNCTION, not to
        // each individual occurrence, so two reviewed sites in one already-
        // reviewed function still share one entry.
    ),
    (
        "crates/jammi-db/src/store/result_schema.rs",
        "deregister_result_tables",
        1,
        // REMOVES a registration (`provider.remove`); not a read of any
        // kind.
    ),
    (
        "crates/jammi-ai/src/session.rs",
        "infer_ordered_read_back_sql",
        1,
        // An INFERENCE task-result table's own read-back of what
        // `InferenceSession::infer` just wrote in the same call, immediately
        // after the write, in-process. This is a different table kind (task
        // results, never a `current_version`-bearing embedding table
        // `PinnedSource` covers) and a different hazard shape (read-your-
        // own-write, not a version straddle across two independent
        // resolutions) — listed here because the literal check cannot
        // distinguish table kinds, not because it shares the embedding-
        // provenance risk this contract is about.
    ),
];

#[test]
fn no_new_anchor_shaped_return_without_review() {
    let surface = scan_surface();
    let hits = anchor_shaped_return_hits(&surface);
    for hit in &hits {
        let allowed_name = ANCHOR_RETURN_ALLOWED
            .iter()
            .find(|(f, n)| *f == hit.file && *n == hit.name);
        assert!(
            allowed_name.is_some(),
            "{}:{} `fn {}` returns a type carrying `InputAnchor`/`CurrentAnchor` — a version-\
             resolved anchor value with no paired content. This is either a NEW straddle-shaped \
             site (route it through `ResultStore::pin_current_version`/`PinnedSource::input_anchor` \
             instead) or a reviewed exception that belongs in this test's `ANCHOR_RETURN_ALLOWED` \
             list, keyed on (file, function name), with the same review its existing entries carry.",
            hit.file, hit.line, hit.name
        );
    }
}

#[test]
fn no_new_bare_record_version_branch_without_review() {
    let surface = scan_surface();
    let hits = bare_record_version_branch_hits(&surface);
    for hit in &hits {
        let allowed_name = RECORD_VERSION_BRANCH_ALLOWED
            .iter()
            .find(|(f, n)| *f == hit.file && *n == hit.name);
        assert!(
            allowed_name.is_some(),
            "{}:{} `fn {}` takes a bare `ResultTableRecord` and re-derives `.current_version` \
             from it in its own body, rather than taking an already-resolved version/manifest/\
             `PinnedSource` as a parameter. This is either a NEW straddle-shaped site or a \
             reviewed exception that belongs in `RECORD_VERSION_BRANCH_ALLOWED`, keyed on (file, \
             function name), with the same review its existing entries carry.",
            hit.file,
            hit.line,
            hit.name
        );
    }
}

#[test]
fn no_new_self_fetched_record_version_without_review() {
    let surface = scan_surface();
    let hits = self_fetched_record_version_hits(&surface);
    for hit in &hits {
        let allowed_name = SELF_FETCHED_RECORD_ALLOWED
            .iter()
            .find(|(f, n)| *f == hit.file && *n == hit.name);
        assert!(
            allowed_name.is_some(),
            "{}:{} `fn {}` resolves a `ResultTableRecord` itself in its own body via \
             `.get_result_table(` and reads the resolved record's `.current_version` field, \
             rather than taking an already-resolved record/version/manifest/`PinnedSource` as a \
             parameter (pattern 2's shape) or content read (pattern 3's shape). This is either a \
             NEW straddle-shaped site (route it through `ResultStore::pin_current_version` \
             instead) or a reviewed exception that belongs in `SELF_FETCHED_RECORD_ALLOWED`, \
             keyed on (file, function name), with the same review its existing entry carries.",
            hit.file,
            hit.line,
            hit.name
        );
    }
}

#[test]
fn no_new_unpinned_session_registration_literal() {
    let surface = scan_surface();
    let sites = session_registration_literal_sites(&surface);
    for ((file, name), count) in &sites {
        let allowed = SESSION_LITERAL_ALLOWED
            .iter()
            .find(|(f, n, _)| f == file && n == name)
            .map(|(_, _, c)| *c)
            .unwrap_or(0);
        assert!(
            *count <= allowed,
            "{file}: fn {name} has {count} occurrence(s) of the bare session-registration \
             literal `\"jammi.{{`, {allowed} audited/allowed for THIS SITE (function) — an \
             allowance in a DIFFERENT function of the same file never covers this one. A NEW \
             site must read through `ResultStore::pin_current_version`/`pinned_provider`, never \
             construct the session-registered `jammi.{{table}}` reference directly. If this IS \
             an audited exception, add it to `SESSION_LITERAL_ALLOWED` — keyed on this same \
             (path, function name) — with the same review its existing entries had."
        );
    }
}

// ── Falsification (R-A): these prove the four detectors and the mask/region
// finder they share actually fire on the shapes they claim to catch, on
// synthetic snippets that never touch git or the real tree — the file-scan
// tests above already prove the real surface is reached; these prove the
// PATTERN LOGIC itself is not vacuous. Each was run against the detector
// BEFORE any allowlist existed for the synthetic name, confirmed red, then
// this comment and the assertion were written — there is no allowlist
// entry anywhere for `sneaky_anchor`/`sneaky_read`/`anchor_identity_for`/
// `anchor_with`/`__probe__.rs`, so a regression that made a detector (or
// `find_fn_regions`'s generic-list matching) stop firing would fail these
// directly, not merely by coincidence of what the real tree happens to
// contain today.

#[test]
fn falsification_anchor_shaped_return_is_detected() {
    let src = r#"
        impl ResultStore {
            pub async fn sneaky_anchor(
                &self,
                t: &ResultTableRecord,
            ) -> Result<InputAnchor> {
                Ok(InputAnchor::result_digest("x", &digest))
            }
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = anchor_shaped_return_hits(&surface);
    assert_eq!(
        hits.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
        vec!["sneaky_anchor"],
        "the anchor-shaped-return detector did not fire on a synthetic function whose return \
         type is `Result<InputAnchor>` — this detector would silently pass a real new straddle \
         site"
    );
}

#[test]
fn falsification_anchor_shaped_return_ignores_non_anchor_signatures() {
    // Negative control (family F): a function that takes an `InputAnchor`
    // as a PARAMETER (not a return type) and returns something else must
    // NOT fire — otherwise the detector would be vacuously triggered by any
    // function mentioning the identifier at all.
    let src = r#"
        pub async fn compare_anchor(&self, anchor: &InputAnchor) -> Result<CurrentAnchor> {
            Ok(CurrentAnchor::Undecidable)
        }
        pub fn benign(&self, anchor: &InputAnchor) -> bool {
            true
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = anchor_shaped_return_hits(&surface);
    let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["compare_anchor"],
        "expected only the function whose RETURN type carries an anchor-shaped type to fire, \
         not the one that merely takes one as a parameter"
    );
}

#[test]
fn falsification_bare_record_version_branch_is_detected() {
    let src = r#"
        async fn sneaky_read(&self, table: &ResultTableRecord) -> Result<Vec<u8>> {
            if let Some(v) = table.current_version {
                return read_version(v).await;
            }
            Ok(vec![])
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = bare_record_version_branch_hits(&surface);
    assert_eq!(
        hits.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
        vec!["sneaky_read"],
        "the bare-record version-branch detector did not fire on a synthetic function that \
         takes `&ResultTableRecord` and reads `.current_version` off it — this detector would \
         silently pass a real new `read_vectors`-shaped site"
    );
}

#[test]
fn falsification_bare_record_version_branch_ignores_explicit_version_param() {
    // Negative control: a function taking an EXPLICIT version number (never
    // re-deriving `.current_version` from a bare record) must not fire —
    // this is `resolve_version_manifest`'s own shape, clause 2(a), safe by
    // construction.
    let src = r#"
        pub async fn resolve_version_manifest(
            &self,
            table: &ResultTableRecord,
            version: i64,
        ) -> Result<VersionManifest> {
            self.read_version_manifest(&table.table_name, version).await
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = bare_record_version_branch_hits(&surface);
    assert!(
        hits.is_empty(),
        "a function taking an explicit `version: i64` parameter and never reading \
         `.current_version` off the bare record must not be flagged: {hits:?}"
    );
}

#[test]
fn falsification_self_fetched_record_version_is_detected() {
    // The round-7 audit's own escape shape ("P2 EVASION" in its harness at
    // `scratchpad/audit-r7/harness.rs`, reused verbatim per the round-8
    // contract): a producer that takes a bare table NAME, resolves the
    // record itself through the catalog, reads `.current_version` off the
    // record it just fetched, and returns the version identity as a plain
    // `String`. Measured by the round-7 audit with the gate's own detector
    // code compiled standalone: patterns 1-3 all return empty on this exact
    // shape.
    let src = r#"
        impl ResultStore {
            pub async fn anchor_identity_for(&self, table_name: &str) -> Result<Option<String>> {
                let record = self.catalog.get_result_table(table_name).await?.unwrap();
                let Some(version) = record.current_version else { return Ok(None); };
                let row = self.catalog.get_result_table_version(table_name, version).await?;
                Ok(row.and_then(|r| r.identity))
            }
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = self_fetched_record_version_hits(&surface);
    assert_eq!(
        hits.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
        vec!["anchor_identity_for"],
        "the self-fetched-record-version detector did not fire on a synthetic function that \
         resolves a `ResultTableRecord` itself via `.get_result_table(` from a bare table name \
         and reads `.current_version` off it — this is the round-7 audit's own escape shape"
    );
    // The property this new detector exists to close, re-verified rather
    // than asserted: patterns 1 and 2 are confirmed still blind to this
    // exact shape (its return type is `Result<Option<String>>`, not
    // anchor-shaped; it never takes `ResultTableRecord` as a parameter) — a
    // change here would mean the escape has moved, not closed.
    assert!(
        anchor_shaped_return_hits(&surface).is_empty(),
        "pattern 1 must not fire on this shape"
    );
    assert!(
        bare_record_version_branch_hits(&surface).is_empty(),
        "pattern 2 must not fire on this shape"
    );
}

#[test]
fn falsification_self_fetched_record_version_ignores_reviewed_delegation_and_parameterized_shapes()
{
    // Negative control 1 (family F): self-fetches the record but delegates
    // the version branch to a SEPARATELY REVIEWED helper
    // (`current_version_identity`, `RECORD_VERSION_BRANCH_ALLOWED`'s own
    // entry) rather than reading the `.current_version` field itself.
    // Without `reads_current_version_field`'s exact field-boundary check,
    // the substring `.current_version` inside `.current_version_identity(`
    // would false-positive here — this is exactly `ResultStore::current_anchor`'s
    // real shape on today's surface.
    let src_delegates = r#"
        impl ResultStore {
            pub async fn wraps_identity(&self, table_name: &str) -> Result<Option<String>> {
                let record = self.catalog.get_result_table(table_name).await?.unwrap();
                self.current_version_identity(&record).await
            }
        }
    "#;
    let surface = vec![("__probe__.rs".to_string(), src_delegates.to_string())];
    let hits = self_fetched_record_version_hits(&surface);
    assert!(
        hits.is_empty(),
        "a function that self-fetches a record and delegates its version branch to a \
         separately-reviewed helper (never reading `.current_version` itself) must not be \
         flagged: {hits:?}"
    );

    // Negative control 2: takes the record as a PARAMETER (pattern 2's own
    // precondition) rather than self-fetching it. Must not ALSO fire
    // pattern 4 — the two patterns partition the surface, they do not both
    // claim the same site.
    let src_parameterized = r#"
        async fn sneaky_read(&self, table: &ResultTableRecord) -> Result<Vec<u8>> {
            if let Some(v) = table.current_version {
                return read_version(v).await;
            }
            Ok(vec![])
        }
    "#;
    let surface2 = vec![("__probe__.rs".to_string(), src_parameterized.to_string())];
    let hits2 = self_fetched_record_version_hits(&surface2);
    assert!(
        hits2.is_empty(),
        "a function that takes `&ResultTableRecord` as a parameter (pattern 2's own shape) must \
         not also be flagged by pattern 4: {hits2:?}"
    );
}

#[test]
fn falsification_generic_arrow_bound_region_is_found() {
    // Round 8, D2: before `find_matching_angle` existed, the `->` inside
    // this `Fn` trait bound closed the generic parameter list early via the
    // depth-counter parens/braces use, desynchronized the scan from the
    // following `(`, and dropped this WHOLE function from
    // `find_fn_regions`'s output — invisible to every detector despite a
    // return type that is literally `Result<InputAnchor>` (the round-7
    // audit's "P4 EVASION", reused verbatim).
    let src = r#"
        impl ResultStore {
            pub async fn anchor_with<F: Fn(&str) -> String>(&self, rec: &ResultTableRecord, f: F) -> Result<InputAnchor> {
                let v = rec.current_version.unwrap();
                Ok(InputAnchor::result_digest(f(&rec.table_name), v))
            }
        }
    "#;
    let masked = mask_non_code(src);
    let regions = find_fn_regions(&masked);
    assert_eq!(
        regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        vec!["anchor_with"],
        "a function whose generic bound contains a `->` return arrow must still be found as a \
         region — before the fix this function vanished entirely"
    );
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    assert_eq!(
        anchor_shaped_return_hits(&surface)
            .iter()
            .map(|h| h.name.as_str())
            .collect::<Vec<_>>(),
        vec!["anchor_with"],
        "pattern 1 must fire on this function's `Result<InputAnchor>` return type now that the \
         region is found"
    );
    assert_eq!(
        bare_record_version_branch_hits(&surface)
            .iter()
            .map(|h| h.name.as_str())
            .collect::<Vec<_>>(),
        vec!["anchor_with"],
        "pattern 2 must also fire (it takes `&ResultTableRecord` and reads `.current_version`)"
    );
}

#[test]
fn falsification_generic_arrow_bound_does_not_break_plain_generics() {
    // Negative-control regression: an ordinary generic parameter list with
    // no arrow (this file's own real surface carries plenty, e.g.
    // `find_matching<T>` a few hundred lines up) must still close at its
    // own `>`, not run away looking for a `->` that never comes.
    let src = r#"
        impl Foo {
            pub fn plain<T: Clone, U>(&self, a: T, b: U) -> Result<InputAnchor> {
                todo!()
            }
        }
    "#;
    let masked = mask_non_code(src);
    let regions = find_fn_regions(&masked);
    assert_eq!(
        regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        vec!["plain"],
        "an ordinary generic parameter list with no arrow must still be found as one region"
    );
}

#[test]
fn falsification_session_registration_literal_is_detected() {
    let src = "let table_ref = TableReference::bare(format!(\"jammi.{}\", table.table_name));\n";
    let sites =
        session_registration_literal_sites(&[("__probe__.rs".to_string(), src.to_string())]);
    assert_eq!(
        sites.get(&("__probe__.rs".to_string(), "<module-scope>".to_string())),
        Some(&1),
        "the session-registration-literal detector did not fire on a synthetic \
         `TableReference::bare(format!(\"jammi.{{}}\", ..))` call at module scope"
    );
}

#[test]
fn falsification_session_registration_literal_binds_to_its_enclosing_function() {
    // Round 8, D3: the SAME literal shape, once inside a named function,
    // must be bound to that function's SITE, not to the file (or to
    // "<module-scope>") — proving the site-binding fix actually attributes
    // the hit correctly rather than merely still finding it somewhere.
    let src = concat!(
        "fn one(table: &str) {\n",
        "    let a = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "}\n",
        "fn two(table: &str) {\n",
        "    let b = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "    let c = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "}\n",
    );
    let sites =
        session_registration_literal_sites(&[("__probe__.rs".to_string(), src.to_string())]);
    assert_eq!(
        sites.get(&("__probe__.rs".to_string(), "one".to_string())),
        Some(&1),
        "fn `one`'s single occurrence must be bound to `one`, not to the file total: {sites:?}"
    );
    assert_eq!(
        sites.get(&("__probe__.rs".to_string(), "two".to_string())),
        Some(&2),
        "fn `two`'s two occurrences must be bound to `two`, and only `two`: {sites:?}"
    );
    assert!(
        !sites.contains_key(&("__probe__.rs".to_string(), "<module-scope>".to_string())),
        "no occurrence here is outside a function, so the module-scope sentinel must not appear: \
         {sites:?}"
    );
}

#[test]
fn falsification_session_registration_literal_ignores_unrelated_jammi_dot_literals() {
    // Negative control: the domain/config literals this surface actually
    // carries (`jammi.toml`, `jammi.audit.search.v1`, `jammi.topic.{}.batch`,
    // `jammi.version.identity.v1`) must NOT fire — only `"jammi.{` (the dot
    // immediately followed by an interpolation brace) is the table-
    // registration shape.
    let src = concat!(
        "let p = dir.path().join(\"jammi.toml\");\n",
        "pub const T: &str = \"jammi.audit.search.v1\";\n",
        "let s = format!(\"jammi.topic.{}.batch\", id);\n",
        "h.update(b\"jammi.version.identity.v1\");\n",
    );
    let sites =
        session_registration_literal_sites(&[("__probe__.rs".to_string(), src.to_string())]);
    assert!(
        sites.is_empty(),
        "a domain-separator/config literal that merely starts with `jammi.` must not be counted \
         as a session-registration reference: {sites:?}"
    );
}

#[test]
fn mask_non_code_ignores_comments_and_string_braces() {
    // If the mask did not blank comments and string-literal contents, this
    // synthetic file would either (a) see the commented-out `fn ignored`
    // as a second, spurious function, or (b) have `find_matching`'s brace
    // counting thrown off by the `{}`/`{v}` inside the two string literals,
    // corrupting where `real`'s body is judged to end.
    let src = concat!(
        "// fn ignored_in_comment(x: ResultTableRecord) { x.current_version }\n",
        "fn real(x: i32) -> i32 {\n",
        "    let s = format!(\"jammi.{}\", x);\n",
        "    let t = format!(\"{v}\", v = x);\n",
        "    x\n",
        "}\n",
    );
    let masked = mask_non_code(src);
    let regions = find_fn_regions(&masked);
    assert_eq!(
        regions.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        vec!["real"],
        "the mask must blank the commented-out `fn` so it is never treated as a second function \
         region, and must blank string contents so their `{{`/`}}` never perturb brace counting"
    );
    assert!(
        regions[0].body.is_some(),
        "`real`'s body must be found despite the braces in its own string literals"
    );
}

#[test]
fn allowlists_match_current_hits_exactly() {
    // The inverse control (round 6 advisory, generalized to all four
    // patterns): an allowance whose site no longer produces that hit — the
    // code was fixed, renamed, or removed — must shrink with it. A
    // permanent allowance is dead slack a LATER, different site could spend
    // without any new review at all, which is exactly the failure mode this
    // contract names for the sweep it replaces.
    let surface = scan_surface();

    let anchor_hits: HashSet<(String, String)> = anchor_shaped_return_hits(&surface)
        .into_iter()
        .map(|h| (h.file, h.name))
        .collect();
    for (file, name) in ANCHOR_RETURN_ALLOWED {
        assert!(
            anchor_hits.contains(&(file.to_string(), name.to_string())),
            "ANCHOR_RETURN_ALLOWED lists {file}:{name}, but the current scan no longer finds an \
             anchor-shaped return there — shrink this list to match (the fix landed, or the \
             function moved/was removed)."
        );
    }

    let record_hits: HashSet<(String, String)> = bare_record_version_branch_hits(&surface)
        .into_iter()
        .map(|h| (h.file, h.name))
        .collect();
    for (file, name) in RECORD_VERSION_BRANCH_ALLOWED {
        assert!(
            record_hits.contains(&(file.to_string(), name.to_string())),
            "RECORD_VERSION_BRANCH_ALLOWED lists {file}:{name}, but the current scan no longer \
             finds a bare-record version-branch there — shrink this list to match."
        );
    }

    let self_fetched_hits: HashSet<(String, String)> = self_fetched_record_version_hits(&surface)
        .into_iter()
        .map(|h| (h.file, h.name))
        .collect();
    for (file, name) in SELF_FETCHED_RECORD_ALLOWED {
        assert!(
            self_fetched_hits.contains(&(file.to_string(), name.to_string())),
            "SELF_FETCHED_RECORD_ALLOWED lists {file}:{name}, but the current scan no longer \
             finds a self-fetched-record version read there — shrink this list to match."
        );
    }

    let literal_sites = session_registration_literal_sites(&surface);
    for (file, name, allowed) in SESSION_LITERAL_ALLOWED {
        let current = literal_sites
            .get(&(file.to_string(), name.to_string()))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            current, *allowed,
            "{file}: fn {name}: SESSION_LITERAL_ALLOWED expects exactly {allowed} occurrence(s) \
             of `\"jammi.{{`, the current scan finds {current} — update this allowlist to match."
        );
    }
}
