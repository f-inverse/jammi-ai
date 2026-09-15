//! The DELTA contract's source gate — a source-text scan over
//! `crates/jammi-db/src` and `crates/jammi-ai/src`, verified by compiling
//! this file's own detector functions into a standalone harness and driving
//! synthetic producers through them.
//!
//! **Self-fetch is a distinct shape from a parameter.** A producer that
//! takes a bare table NAME and resolves the record itself through the
//! catalog (`self.catalog.get_result_table(name)` — the prevailing in-tree
//! idiom) before reading `.current_version` off the record it just fetched
//! is a shape none of the parameter-typed detectors (patterns 1-3, below)
//! can see, since none of them requires (or inspects) a
//! [`ResultTableRecord`] taken by parameter at all in this case.
//! [`self_fetched_record_version_hits`] (pattern 4, below) exists for
//! exactly this shape, and finds `InferenceSession::refreshable_record`
//! (`crates/jammi-ai/src/pipeline/embedding_refresh.rs`) doing it on the
//! surface as it stands today.
//!
//! **A `->` inside a generic bound is not a closing angle bracket.**
//! [`find_fn_regions`]'s optional generic parameter list is matched by
//! [`find_matching_angle`], which treats a `->` (as in a closure/`Fn`-trait
//! bound written `Fn(&str) -> String`) as a single unit that never changes
//! angle-bracket depth — unlike the plain depth-counter [`find_matching`]
//! uses for parens and braces, which does not know that a `->`'s `>` is not
//! a closing bracket. A function written
//! `fn f<F: Fn(&str) -> String>(&self, rec: &ResultTableRecord) -> Result<InputAnchor>`
//! would otherwise have its generic list close at the arrow, desynchronizing
//! the scan from `(` and dropping the WHOLE function from every detector's
//! output — invisible even to the anchor-shaped-return detector, despite a
//! return type that is literally `Result<InputAnchor>`. Checked, not
//! assumed: an independent `fn <ident>` token count over today's surface
//! equals the number of regions `find_fn_regions` produces, so no live
//! function needs this exception to be found today — it is handled anyway,
//! because a silent fail-OPEN in the shared parsing layer every detector
//! below depends on is exactly the class of defect this gate exists to
//! prevent elsewhere.
//!
//! **Why a machine-derived quantifier, not a hand-enumerated one.** The
//! straddle class this file exists to catch — a producer that resolves a
//! result table's version once for its provenance anchor and again for its
//! content, so a version publish landing between the two calls makes a
//! durable artifact whose provenance names one version while its rows came
//! from another — is easy to argue "closed by construction" and hard to
//! prove closed by prose: a hand-written sweep enumerating "every `pub`/
//! `pub(crate)` function in this module" for a property that says "no public
//! interface" misses a member the moment the module gains one, a property
//! stated over one crate's `src/pipeline/` directory misses a sibling crate
//! entirely, and a prose sweep next to a machine scan checking the SAME
//! property in two different places invites disagreement about which one is
//! authoritative when they diverge. This file is the ONLY enforcement for
//! the straddle class, and its quantifier is derived from version control,
//! not typed by hand: every git-TRACKED `.rs` file under
//! `crates/jammi-db/src` and `crates/jammi-ai/src`, recursively (`git
//! ls-files` itself recurses, so there is no second, hand-rolled directory
//! walk that could silently stop at one level), is read and scanned by every
//! check below. A file `git ls-files` lists that this process cannot then
//! read is a hard failure naming the file, not a silently skipped one.
//!
//! Whether a function is `pub` or crate-private is NOT part of any check
//! here: a visibility keyword protects nobody in a greenfield crate with no
//! external consumers, and "constructible from published surface" is not a
//! property worth enforcing here. Every detector below scans every function
//! regardless of visibility.
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
//!    `jammi.{table}` reference spelled out directly in source text, over the
//!    WHOLE two-crate surface rather than one crate's `src/pipeline/`
//!    directory, bound to its enclosing function rather than to the file as
//!    a whole.
//! 4. [`self_fetched_record_version_hits`] — a function that does NOT take a
//!    [`jammi_db::catalog::result_repo::ResultTableRecord`] as a parameter
//!    (pattern 2's precondition) but reads the `.current_version` FIELD off
//!    a record it holds by some OTHER means — the parameter shape and the
//!    self-fetch shape partition the surface between patterns 2 and 4 rather
//!    than overlapping it. This is pattern 2's own precondition inverted, and
//!    all three of the other detectors return empty on this shape (see
//!    [`self_fetched_record_version_hits`]'s own doc for its history of
//!    conjuncts and their measured cost). Destructuring a `ResultTableRecord`
//!    into a bare `current_version` local remains disclosed, not covered —
//!    see that same doc for why.
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
//! `fn`) would be mistaken for a function item — checked: `grep -rn 'fn (\['
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
//! **What this file's surface is NOT (disclosure, not a claim of absolute
//! fail-closed).** [`tracked_rs_files`] derives the
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
//!
//! **The allowlist's own prose is itself a claim that needs checking.**
//! Deleting the hand-written sweep this file's enforcement replaced did not
//! eliminate unverified human claims; every allowlist entry below still
//! clears its site with a
//! hand-written justification, and three of those justifications named a
//! CALLER SET in prose — an enumeration, not a dataflow property, and
//! therefore exactly as derivable as everything else this file checks. Two
//! were false: `producing_descriptor`'s declared set omitted a real caller
//! (`compact_embeddings`), and `refreshable_record`'s omitted TWO — the
//! destructive ones, `compact_embeddings` (publishes a new version) and
//! `expire_versions` (permanently reaps old ones). Both are now DATA
//! (`PRODUCING_DESCRIPTOR_CALLERS`, `REFRESHABLE_RECORD_CALLERS`, below —
//! see [`callers_of`]'s doc), checked by `caller_set_claims_match_reality`
//! against the real surface; a claim that cannot be expressed as a caller
//! set (what a caller DOES with the value, rather than who the caller is)
//! stays prose. One allowlisted entry's MECHANISM claim was also false, not
//! just its caller set — `refreshable_record`'s entry cleared itself by
//! asserting every later pairing comes from a fresh re-fetch, when the
//! function it named (`ensure_base_version`) documents, in its OWN entry,
//! that its dominant arm returns the same record unchanged; both entries are
//! corrected together. One negative control asserted that the exact escape
//! shape pattern 4 exists to catch (self-fetch, then resolve the version
//! identity as a plain string) must never be flagged merely because it
//! delegates to a named, reviewed helper — clearance by a callee's name is
//! the same failure this file's own module doc disowns for a return TYPE's
//! name; it is corrected to make the delegate's caller set the thing that is
//! actually checked, rather than asserting the shape itself is safe.

use std::collections::BTreeSet;
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
/// `read_dir` walk, since a hand-rolled walk can silently stop at one
/// directory level and never scan a nested directory such as
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
pub(crate) fn scan_surface() -> Vec<(String, String)> {
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

/// Blank every line comment and block comment in `text` (same length and
/// newlines preserved, so line numbers still match the original), leaving
/// string and char literal CONTENT untouched — unlike [`mask_non_code`],
/// which blanks comments AND string/char literals and so cannot be used to
/// find a DDL keyword that lives inside a string
/// ([`fine_tune_ddl_relation_binding_hits`]'s exact requirement). String
/// literals are still recognised (and skipped over without masking) so that
/// a `//` or `/*` appearing inside one — a URL, say — is never mistaken for
/// the start of a comment on the next iteration; raw strings and char
/// literals are not specially recognised here. This function has TWO
/// callers today, with two different scopes: [`fine_tune_ddl_relation_binding_hits`],
/// scoped to `crates/jammi-ai/src/fine_tune/**`, and [`ddl_literal_occurrences`],
/// unscoped over both crates' whole `src` trees ([`SURFACE_DIRS`]). Not
/// recognising raw strings or char literals is a LIVE desync today, not a
/// disclosed-but-inert limit: this scanner pairs the FIRST `"` it meets with
/// the NEXT `"` it meets, with no notion of an `r#`/`r##` delimiter or of
/// the hash count a raw string's real closing quote must match, so its
/// "inside a string" bookkeeping can invert and run across lines — this is
/// not one enumerable mechanism, so no fixed list of triggers is claimed
/// here. Measured with this exact function, compiled verbatim (not a
/// transcription), over both `src` trees at this head: 22 code lines get
/// blanked as if they were comments and 12 real `//` comment lines are left
/// completely unblanked, across `storage/config.rs:555`/`:619`,
/// `config/tests.rs`, `config/secret.rs`, and `sql/ident.rs` — the complete
/// set. The `grep -rn 'r#*".*\(//\|/\*\)'
/// crates/jammi-db/src crates/jammi-ai/src` command finds exactly the two
/// SINGLE-LINE raw strings responsible for part of the code-blanking above —
/// `storage/config.rs:555` and `:619` — and nothing else, but that describes
/// the one grep, not the tree: a delimiter-aware scan of `config/tests.rs`
/// alone finds 12 further raw strings that span MULTIPLE physical lines
/// (URL-carrying TOML fixtures), invisible to a single-line pattern by
/// construction — the grep is a description of what it matches, not a
/// completeness instrument over what raw strings exist. The
/// consequence for this function's callers: a DDL-shaped string literal
/// sitting on a desynced line is invisible to [`ddl_literal_occurrences`] —
/// a FIFTH residual alongside the four the "literal-occurrence gate" section
/// below already names, tracked on the same issue,
/// <https://github.com/f-inverse/jammi-ai/issues/554> (amended with this
/// residual; the rebuild that closes it replaces this hand-rolled masking
/// with a real tokenizer that knows raw-string hash counts).
fn mask_comments_only(text: &str) -> String {
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
            i = j.min(n);
            continue;
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
/// counts parens/braces — with ONE exception: the `>` of a
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
/// ORDINAL (see [`assign_ordinals`]'s doc for why this, rather than the
/// declaration line, is the allowlist key), its parameter-list text
/// (masked), and — when it has a body rather than a trait-declaration `;` —
/// its return-type text and its body text (both masked).
struct FnRegion {
    name: String,
    line: usize,
    end_line: usize,
    ordinal: usize,
    params: String,
    return_type: Option<String>,
    body: Option<String>,
}

/// Assign each region its ORDINAL: the 1-based count of functions sharing
/// its name in `regions`, in the order `regions` already lists them —
/// [`find_fn_regions`] appends regions in a single left-to-right scan of the
/// file, so that order is source (declaration) order, and the first
/// function named `f` in a file gets ordinal 1, the second gets 2, and so
/// on, independent of every OTHER function's name.
///
/// **Why ordinal, not line, and not a bare name.** A `(file, name,
/// declaration line)` key is stable only until something above the site in
/// the same file changes line count — a merge that adds a four-line doc
/// comment above every allowlisted site below it desyncs every one of them
/// from the code they were reviewed against, with no change to the reviewed
/// function itself. An ordinal has neither failure mode: it is stable under
/// any edit strictly ABOVE the site that does not insert or remove a
/// same-named sibling before it (checked below,
/// [`falsification_ordinal_survives_a_line_shift_above_it`] makes exactly
/// that edit and shows the gate still passes), and it still distinguishes
/// same-named siblings in one file the way a bare `(file, name)` key cannot
/// — the surface carries 109 colliding `(file, name)` pairs today, and an
/// ordinal partitions every one of them by declaration order instead.
///
/// The residual this trades in: inserting or deleting a same-named sibling
/// ABOVE an allowlisted site (never touched by this program's edits so far
/// on the sites the allowlists below name) shifts that site's ordinal — this
/// key is not claimed immune to every edit, only to the specific, common
/// shape (edits that add/remove lines, comments, or unrelated functions).
/// Renaming, reordering, or deleting a same-named sibling remains
/// a real edit that requires updating the allowlist entry it displaces, the
/// same way it always would have under any positional key.
fn assign_ordinals(regions: &mut [FnRegion]) {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for region in regions.iter_mut() {
        let counter = seen.entry(region.name.clone()).or_insert(0);
        *counter += 1;
        region.ordinal = *counter;
    }
}

/// Every `fn` item in `masked` (an `impl`/free/trait/nested function — this
/// is deliberately unfiltered by visibility or nesting, per this file's
/// module doc: a visibility keyword is not part of any property this gate
/// checks). Source-text scanning, not a parser: see [`mask_non_code`]'s doc
/// for the stated limits this inherits. Ordinals are assigned by
/// [`assign_ordinals`] over the whole per-file result before it is
/// returned, so every caller sees them already populated.
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
                                ordinal: 0, // assigned below, by `assign_ordinals`
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
    assign_ordinals(&mut regions);
    regions
}

/// One hit of pattern 1 (`crates/jammi-ai/tests/it/pinned_source_gate.rs`
/// module doc's list) — a function whose return type carries `InputAnchor`
/// or `CurrentAnchor` verbatim. `line` is carried for human-readable
/// diagnostics only; `ordinal` is the field every allowlist match is
/// actually keyed on — see [`assign_ordinals`]'s doc for why.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hit {
    file: String,
    name: String,
    line: usize,
    ordinal: usize,
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
                        ordinal: region.ordinal,
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
                            ordinal: region.ordinal,
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

/// Pattern 4 — a function that does NOT take a bare `ResultTableRecord`
/// as a parameter (pattern 2's own precondition, excluded here via
/// `!region.params.contains("ResultTableRecord")` so patterns 2 and 4
/// partition the surface instead of double-flagging the same site under two
/// different names) but reads a self-obtained record's `.current_version`
/// FIELD (see [`reads_current_version_field`]'s doc for why this must be a
/// field match, not a substring one).
///
/// **No idiom conjunct.** An earlier design of this detector additionally
/// required the literal idiom `.get_result_table(` to appear in the same
/// function's body. Measuring that conjunct's cost directly (`p4_no_idiom` in
/// a standalone harness) rather than arguing it showed: dropping it raises
/// the real-surface hit count from 1 to 6 — five more allowlist entries, not
/// hundreds — and the conjunct was never load-bearing against false
/// positives, only against coverage. It was a narrowing that let three of the
/// module doc's own
/// named evasions through: a record resolved through ANY OTHER catalog
/// method (`probe_cache_record`, `exact_match_candidates`), the SAME method
/// called in fully-qualified/UFCS form (`ResultRepo::get_result_table(...)`,
/// which never spells `.get_result_table(`), and a parameter typed through a
/// local alias of `ResultTableRecord` (which fails pattern 2's own
/// parameter-text precondition and, before this fix, ALSO failed pattern 4's
/// idiom precondition since such a function never self-fetches at all — it
/// already holds the record, just not under the literal type name pattern 2
/// looks for). All three are closed by dropping the conjunct: none of them
/// require the self-fetch idiom, they only require a `.current_version`
/// FIELD read outside a function whose parameter TEXT names
/// `ResultTableRecord`. A fourth named evasion, destructuring
/// (`let ResultTableRecord { current_version, .. } = record;`, binding the
/// field to a bare local identifier with no `.current_version` text at all),
/// remains genuinely uncovered by this or any other conjunct here — see the
/// module doc's disclosure list; a text-only check cannot distinguish that
/// shape from an ordinary struct-literal field-init shorthand
/// (`ResultTableRecord { current_version, .. other }`) without becoming a
/// parser, so per R-A it stays disclosed rather than "fixed" by a check that
/// would also flag unrelated row-construction code (checked: relaxing the
/// field match to a bare, both-sides-bounded `current_version` identifier —
/// dropping the leading-dot requirement — pulls in
/// `catalog/result_repo.rs`'s `from_wire_projection`/`parse_row`/
/// `read_cas_target`, three row-PARSING functions that construct a
/// `ResultTableRecord` rather than branch on an already-resolved one; that
/// is the "hundreds" scale this same measurement warns against, not the
/// "five" the idiom-conjunct removal costs).
fn self_fetched_record_version_hits(surface: &[(String, String)]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        for region in find_fn_regions(&masked) {
            if region.params.contains("ResultTableRecord") {
                continue;
            }
            if let Some(body) = &region.body {
                if reads_current_version_field(body) {
                    hits.push(Hit {
                        file: file.clone(),
                        name: region.name,
                        line: region.line,
                        ordinal: region.ordinal,
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
/// FUNCTION rather than to the file as a whole (a per-file `usize` allowance
/// would be fungible across every site inside that file, so a NEW unpinned
/// read added to an already-allowlisted file would pass review-free as long
/// as an existing one is deleted in the same commit). This is the same
/// site-binding [`anchor_shaped_return_hits`] and
/// [`bare_record_version_branch_hits`] already use, keyed on `(file,
/// function name)` instead of `file` alone.
///
/// The narrowing is deliberately narrower than a bare `"jammi.` substring
/// check — `"jammi.toml"`, `"jammi.audit.search.v1"`, `"jammi.topic.{}.batch"`
/// and this crate's other domain-separator/config literals all contain
/// `"jammi.` but are never followed immediately by `{` — only a session
/// table-reference literal spells the table name as an interpolation
/// directly after the dot. Verified against every `"jammi.` occurrence in
/// this surface (`grep -rn '"jammi\.' crates/jammi-db/src crates/jammi-ai/src`):
/// every non-`"jammi.{` hit is one of the domain/config literals above, and
/// every session-registration site this file's own module doc names is
/// `"jammi.{`.
///
/// Attribution runs on the ORIGINAL, unmasked text (the literal itself is a
/// string, which [`mask_non_code`] would blank), using [`find_fn_regions`]'s
/// `(line, end_line)` only to find which function's line range contains a
/// given hit line — the innermost (smallest-range) containing region wins,
/// so a hit inside a nested function is never double-counted against its
/// enclosing one too. A hit whose line falls inside no region at all (a
/// module-level literal, which does not occur on today's surface) is bound
/// to the sentinel site `"<module-scope>"` at ordinal 0 (no real function
/// has ordinal 0 — [`assign_ordinals`] starts counting at 1), which no
/// `ALLOWED` entry ever names, so it fails loudly rather than being silently
/// mis-attributed.
///
/// **Why the key has three elements, and why the third is an ordinal.** A
/// bare `(file, function name)` key collides for a same-named sibling in the
/// same file (a trait method and an inherent method both called
/// `read_vectors`, say), summing their counts into one allowance — the
/// surface carries 109 colliding `(file, name)` pairs today, so this is not
/// a hypothetical. The third element distinguishes them; it is the
/// function's ORDINAL (see [`assign_ordinals`]'s doc), not its declaration
/// line, because a line number drifts every time something above the site in
/// the same file gains or loses a line (an unrelated merge that adds a doc
/// comment desyncs every allowlist entry below the insertion point from the
/// code it was reviewed against, with the reviewed function itself
/// unchanged), while an ordinal is stable under exactly that class of edit
/// and still partitions same-named siblings the way a line number would.
fn session_registration_literal_sites(
    surface: &[(String, String)],
) -> std::collections::HashMap<(String, String, usize), usize> {
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
            let (site, site_ordinal) = best
                .map(|r| (r.name.clone(), r.ordinal))
                .unwrap_or_else(|| ("<module-scope>".to_string(), 0));
            *counts
                .entry((file.clone(), site, site_ordinal))
                .or_insert(0) += hits;
        }
    }
    counts
}

// ── Caller-set claims, machine-checked. ─────────────────────────────────────
//
// Several allowlist entries below clear their site by naming who calls it —
// "its one in-tree caller is", "its only in-tree callers are", "every
// caller is" — and R-H holds that a caller SET is an enumeration over
// source text, not a dataflow property, so it is derivable exactly the way
// [`session_registration_literal_sites`] already derives a hit's enclosing
// function: attribute every `.method(` call site to its innermost
// containing region. What a caller DOES with the value it gets (whether it
// persists it, discards it, or pairs it with something else) is NOT
// enumerable this way and stays prose in the entry itself, per R-A.

/// The `(file, enclosing function name)` sites whose masked body contains a
/// call `.method(` — an ENUMERATION, not a dataflow analysis: it names every
/// function that calls `method` this way, nothing about what that function
/// does with the result. Deliberately does NOT exclude a hit whose
/// enclosing function shares `method`'s own name: a forwarding wrapper
/// (`JammiSession::read_vectors` calling `self.inner.read_vectors(...)`) is
/// a REAL caller, and an early draft of this check that skipped "the
/// enclosing function's name equals the callee's name" as a guard against
/// spurious recursive self-matches hid exactly that forwarding call —
/// checked directly: without the guard, `callers_of(surface,
/// "read_vectors")` finds both `LocalSession::read_vectors` and
/// `JammiSession::read_vectors`; a version of this function carrying that
/// guard finds neither.
fn callers_of(surface: &[(String, String)], method: &str) -> BTreeSet<(String, String)> {
    let needle = format!(".{method}(");
    let mut out = BTreeSet::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        let regions = find_fn_regions(&masked);
        for (line_idx, line) in masked.lines().enumerate() {
            let line_no = line_idx + 1;
            if !line.contains(&needle) {
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
            out.insert((file.clone(), site));
        }
    }
    out
}

/// `ANCHOR_RETURN_ALLOWED`'s `current_anchor` entry's caller-set claim, as
/// DATA: `staleness`, same file, is its only in-tree caller of
/// `.current_anchor(`.
const CURRENT_ANCHOR_CALLERS: &[(&str, &str)] =
    &[("crates/jammi-db/src/store/freshness.rs", "staleness")];

/// `RECORD_VERSION_BRANCH_ALLOWED`'s `producing_descriptor` entry's caller-
/// set claim, as DATA: three in-tree callers, `compact_embeddings` and
/// `refresh_embeddings` (`embedding_refresh.rs`) plus `recompute_one`
/// (`recompute.rs`).
const PRODUCING_DESCRIPTOR_CALLERS: &[(&str, &str)] = &[
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "compact_embeddings",
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "refresh_embeddings",
    ),
    ("crates/jammi-ai/src/pipeline/recompute.rs", "recompute_one"),
];

/// `SELF_FETCHED_RECORD_ALLOWED`'s `refreshable_record` entry's caller-set
/// claim, as DATA: THREE in-tree callers, `refresh_embeddings` plus the two
/// destructive ones, `compact_embeddings` and `expire_versions` (see that
/// entry's review for what each actually does with the value).
const REFRESHABLE_RECORD_CALLERS: &[(&str, &str)] = &[
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "compact_embeddings",
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "expire_versions",
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "refresh_embeddings",
    ),
];

/// `RECORD_VERSION_BRANCH_ALLOWED`'s `current_version_provider` entry's
/// caller-set claim, as DATA: `pinned_provider`'s unversioned arm is its
/// only in-tree caller.
const CURRENT_VERSION_PROVIDER_CALLERS: &[(&str, &str)] =
    &[("crates/jammi-db/src/store/mod.rs", "pinned_provider")];

/// `RECORD_VERSION_BRANCH_ALLOWED`'s `current_version_identity` entry's
/// caller-set claim, as DATA: TWO in-tree callers, both already-reviewed
/// non-persisting reads: `current_anchor` (`ANCHOR_RETURN_ALLOWED`'s entry)
/// and
/// `verify_materialization` (`RECORD_VERSION_BRANCH_ALLOWED`'s own entry,
/// "never persists a new anchor"). Pinning this set is what actually
/// protects the delegation shape `falsification_self_fetched_record_version_ignores_reviewed_delegation_and_parameterized_shapes`
/// exercises: a THIRD caller appearing here — such as a hypothetical
/// producer that self-fetches a record and returns
/// `current_version_identity`'s value as a plain string, pattern 4's own
/// disclosed cross-function blind spot — turns `caller_set_claims_match_reality`
/// red the moment it is added, even though pattern 4 itself cannot see it.
const CURRENT_VERSION_IDENTITY_CALLERS: &[(&str, &str)] = &[
    ("crates/jammi-db/src/store/freshness.rs", "current_anchor"),
    ("crates/jammi-db/src/store/mod.rs", "verify_materialization"),
];

/// `RECORD_VERSION_BRANCH_ALLOWED`'s `read_vectors` entry's caller-set
/// claim, as DATA: its only two in-tree callers are the forwarding wrappers
/// of the same name in
/// `jammi-ai` (`LocalSession::read_vectors`, `JammiSession::read_vectors`),
/// each a one-line delegate to this function, never a provenance-persisting
/// producer.
const READ_VECTORS_CALLERS: &[(&str, &str)] = &[
    ("crates/jammi-ai/src/local_session.rs", "read_vectors"),
    ("crates/jammi-ai/src/session.rs", "read_vectors"),
];

#[test]
fn caller_set_claims_match_reality() {
    let surface = scan_surface();
    let checks: &[(&str, &[(&str, &str)])] = &[
        ("current_anchor", CURRENT_ANCHOR_CALLERS),
        ("producing_descriptor", PRODUCING_DESCRIPTOR_CALLERS),
        ("refreshable_record", REFRESHABLE_RECORD_CALLERS),
        ("current_version_provider", CURRENT_VERSION_PROVIDER_CALLERS),
        ("current_version_identity", CURRENT_VERSION_IDENTITY_CALLERS),
        ("read_vectors", READ_VECTORS_CALLERS),
    ];
    for (method, declared) in checks {
        let real = callers_of(&surface, method);
        let declared_set: BTreeSet<(String, String)> = declared
            .iter()
            .map(|(f, n)| (f.to_string(), n.to_string()))
            .collect();
        let omitted: Vec<_> = real.difference(&declared_set).collect();
        assert!(
            omitted.is_empty(),
            "{method}: the allowlist's caller-set claim omits real in-tree caller(s) of \
             `.{method}(` — {omitted:?}. R-H requires every caller-set claim to be machine-\
             checked; add these to the declared `_CALLERS` list AND review what each does with \
             the value in the allowlist entry's prose, or retract the caller-set claim entirely."
        );
        let stale: Vec<_> = declared_set.difference(&real).collect();
        assert!(
            stale.is_empty(),
            "{method}: the declared caller-set lists {stale:?}, but the current scan finds no \
             call to `.{method}(` inside them — the claim no longer matches reality (a caller was \
             renamed, removed, or never called this way). Shrink the declared set to match, the \
             same inverse-control discipline `allowlists_match_current_hits_exactly` applies to \
             shape hits."
        );
    }
}

// ── Allowlists. Every entry is a hit this gate's scan finds TODAY (verified
// by running each detector with an empty allowlist and transcribing every
// hit), never a guess. Each carries its own review note — a bare list of
// paths is not a review, it is the same failure mode this contract exists
// to close for the sweep it replaces.
//
// Keyed on `(file, function name, ORDINAL)` (see [`assign_ordinals`]'s doc),
// not `(file, function name)` alone (a same-named sibling in the same file
// would collide under the two-element key and inherit an unrelated review —
// the surface carries 109 colliding `(file, name)` pairs today) and not
// `(file, function name, declaration LINE)` either (a line number drifts
// every time an edit ABOVE the site in the same file changes that file's
// line count, and an unrelated merge that added a four-line doc comment
// above `crates/jammi-ai/src/pipeline/embedding_refresh.rs`'s
// `compact_embeddings`/`expire_versions` desynchronized both of those
// entries from the functions they were reviewed against without either
// function itself changing at all). Every entry below carries ordinal 1
// today (no two functions sharing a name collide at any currently-
// allowlisted site — see each entry's own inline note for the traceability
// line number, kept for human review only, never matched on). ─────────────

/// Pattern 1 — `(file, function name, ordinal)`.
const ANCHOR_RETURN_ALLOWED: &[(&str, &str, usize)] = &[
    (
        "crates/jammi-db/src/store/mod.rs",
        "input_anchor",
        1, // ordinal 1 — the only `input_anchor` in this file; line 306 today
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
        1, // ordinal 1 — the only `current_anchor` in this file; line 458 today
           // `ResultStore::current_anchor` — NOT CLOSED, disclosed rather than
           // hidden. Its only in-tree callers (`freshness.rs`'s own staleness
           // comparison, same file) use the returned
           // `CurrentAnchor::ResultDigest(String)` for an equality check
           // against a RECORDED anchor and then discard it — never persist it
           // as a new artifact's provenance. But its input is publicly
           // mintable from a bare table name (`InputAnchor::result_digest`
           // over `ArtifactDigest(pub String)`), and the value it returns is
           // byte-identical to `PinnedSource::input_anchor`'s on both arms.
           // Nothing in the type system stops a FUTURE caller from pairing
           // this anchor with an independently-resolved read and persisting
           // the pair — the exact straddle this contract names. Closing that
           // (narrow the type so it cannot be separated from content, or fold
           // this crate's callers onto `pin_current_version`) is a design
           // change this gate does not itself make; this entry is the gate's
           // record that the residual is real, watched, and not silently
           // absorbed.
           //
           // The "only in-tree callers" claim is not carried in prose alone —
           // `CURRENT_ANCHOR_CALLERS` above is machine-checked by
           // `caller_set_claims_match_reality` against
           // `callers_of(&surface, "current_anchor")` on every run, and a
           // future second caller turns that test red the moment it lands.
    ),
    (
        "crates/jammi-ai/src/pipeline/graph_propagation.rs",
        "edge_source_anchor",
        1, // ordinal 1 — the only `edge_source_anchor` in this file; line 821 today
           // Delegates to `pin_current_version(record).await?.input_anchor()`
           // — the sanctioned pattern every `result_digest_anchor` caller is
           // migrated to, not an independent anchor-only resolve. The
           // residual here is one level up, at this function's own caller:
           // `edge_scan_sql` reads the SAME edge table's content through an
           // unpinned, session-registered scan
           // (`session_registration_literal_sites`'s allowlist entry for this
           // same file), so the anchor and the edge content are NOT from one
           // resolution. This is a disclosed, reviewed exception for the S9
           // edge relation specifically (never the pinned embedding table).
    ),
    (
        "crates/jammi-ai/src/pipeline/recompute.rs",
        "reresolve_recorded_anchor",
        1, // ordinal 1 -- the only `reresolve_recorded_anchor` in this file; line 567 today
           // `InferenceSession::reresolve_recorded_anchor` -- never mints a NEW
           // pinned anchor from arbitrary live state: it dispatches on the KIND
           // already recorded in the table's OWN `.materialization.json`
           // sidecar (read by its caller, `recompute_training_set`, into
           // `recorded_anchors`) and only re-resolves that SAME recorded
           // relation name (`anchor.source`). The `UnpinnedAtInstant` arm
           // re-stamps the read instant but stays unpinned -- the same weak
           // guarantee the original recording made, never upgraded. The
           // `MutableVersion`/`ResultDigest` arm fetches the named relation and
           // routes it through `pin_current_version(current).await?
           // .input_anchor()` -- the same sanctioned pattern
           // `edge_source_anchor`'s entry above uses, not an independent
           // anchor-only resolve -- and REFUSES (`JammiError::NotRecomputable`,
           // naming the anchor's own source) when the relation no longer
           // resolves, rather than silently downgrading to unpinned. Both
           // determinants are standing tests, not merely reviewed once:
           // `recompute_training_set_re_resolves_a_pinned_result_digest_anchor_pinned`
           // (a recorded `ResultDigest` anchor replays PINNED, at the source's
           // CURRENT digest, never the stale recorded one) and
           // `recompute_training_set_refuses_a_pinned_anchor_whose_target_is_gone`
           // (a recorded anchor whose target no longer resolves is
           // `NotRecomputable`, never silently treated as unpinned). No
           // in-tree producer writes a `MutableVersion`/`ResultDigest` anchor
           // onto a `TrainingSet`-kind table today (`training_set.rs`'s own
           // "Anchors" doc: both `materialize_projection` and
           // `materialize_sampled_pairs` record `UnpinnedAtInstant` only) --
           // this arm is exercised by the two tests above via a forged
           // manifest, future-proofing the exhaustive `AnchorKind` match (K7)
           // for a producer that does, the same disclosed-but-unreached shape
           // `recompute_fine_tune`'s own doc names for its arm.
    ),
];

/// Pattern 2 — `(file, function name, ordinal)`.
const RECORD_VERSION_BRANCH_ALLOWED: &[(&str, &str, usize)] = &[
    (
        "crates/jammi-db/src/store/mod.rs",
        "current_version_identity",
        1, // ordinal 1 — the only `current_version_identity` in this file; line 1164 today
           // `CURRENT_VERSION_IDENTITY_CALLERS` above is the machine-checked
           // claim: its two in-tree callers are `current_anchor`
           // (`ANCHOR_RETURN_ALLOWED`'s entry — discards the value after an
           // equality check) and `verify_materialization` (this list's own
           // entry below — never persists a new anchor). Neither pairs this
           // value with an independent content read to persist as provenance.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "verify_materialization",
        1, // ordinal 1 — the only `verify_materialization` in this file; line 1301 today
           // Read-only integrity check: compares a version's RECORDED identity
           // against a freshly recomputed one and reports a `MatchVerdict`. It
           // never persists a new anchor or a new artifact.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "resolve_search_mode_local",
        1, // ordinal 1 — the only `resolve_search_mode_local` in this file; line 2280 today
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
        1, // ordinal 1 — the only `bind_result_table` in this file; line 2516 today
           // Documented "Read class" residual on its own doc comment: serves a
           // possibly-stale session-bound registration, never persists an
           // anchor. Every producer that DOES persist an anchor is required
           // (by that same doc) to route through `pin_current_version` /
           // `pinned_provider` instead.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "current_version_provider",
        1, // ordinal 1 — the only `current_version_provider` in this file; line 2675 today
           // Private; `CURRENT_VERSION_PROVIDER_CALLERS` above is the machine-
           // checked claim: its only in-tree caller is `pinned_provider`'s own
           // unversioned arm. Reading `.current_version` off the SAME
           // `pin.record` `pinned_provider` was itself called with cannot
           // straddle anything — there is no second, independent resolution
           // here.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "pin_current_version",
        1, // ordinal 1 — the only `pin_current_version` in this file; line 2756 today
           // The seam itself: reads `record.current_version` once to decide
           // which arm to take, then returns a `PinnedSource` that carries the
           // record, the resolved version, and (for a versioned table) the
           // manifest from that SAME resolution — safe by construction, not by
           // convention.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "allocate_version",
        1, // ordinal 1 — the only `allocate_version` in this file; line 2917 today
           // Reads `table.current_version` only as the CAS's EXPECTED PARENT
           // (refuses with `ParentMoved` on mismatch); never reads content
           // under a version it resolves itself.
    ),
    (
        "crates/jammi-db/src/store/freshness.rs",
        "producing_descriptor",
        1, // ordinal 1 — the only `producing_descriptor` in this file; line 505 today
           // NOT CLOSED, disclosed rather than hidden. Its callers use the
           // returned `ProducingDescriptor` only to select WHICH producer
           // verb/params to replay — never as content or as a persisted anchor
           // — and each producer that then materializes a new artifact
           // performs its own, independent `pin_current_version` resolution
           // for that artifact's actual anchor and rows. A version drift
           // between this read and that later pin could select a stale REPLAY
           // TARGET, never corrupt a persisted anchor/content pairing. Closing
           // the shape itself (this function still re-derives
           // `table.current_version` from a bare record) is a design change
           // this gate does not itself make; recorded here rather than
           // silently absorbed.
           //
           // `PRODUCING_DESCRIPTOR_CALLERS` above carries all three real
           // callers (`refresh_embeddings`, `recompute_one`,
           // `compact_embeddings`) as DATA, machine-checked by
           // `caller_set_claims_match_reality` rather than re-verified by hand
           // each time; all three use the descriptor the same way (select the
           // producer verb and its inputs).
    ),
    (
        "crates/jammi-db/src/session.rs",
        "read_vectors",
        1, // ordinal 1 — the only `read_vectors` in this file; line 977 today
           // NOT CLOSED, disclosed rather than hidden. The versioned arm reads
           // content through this SESSION's own `jammi.{table}` registration
           // (see `session_registration_literal_sites`'s allowlist entry for
           // this same file) rather than through `pinned_provider` — an
           // unpinned, version-branched content read, re-exported publicly at
           // `read_vectors`, `jammi-ai/src/session.rs:1082` and
           // `read_vectors`, `jammi-ai/src/local_session.rs:365`.
           // `READ_VECTORS_CALLERS` above is the machine-checked claim: its
           // only two in-tree callers are those two forwarding wrappers, each
           // a one-line delegate, never a provenance-persisting producer.
           // Closing this (route the versioned arm through
           // `pin_current_version`/`pinned_provider`, or require a
           // caller-supplied `PinnedSource`) is a design change this gate does
           // not itself make; the straddle this function makes constructible
           // is exactly the class this contract names as real and unclosed.
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "ensure_base_version",
        1, // ordinal 1 — the only `ensure_base_version` in this file; line 657 today
           // Guard clause only on its DOMINANT arm:
           // `record.current_version.is_some()` short-circuits to "already
           // versioned, return the SAME record UNCHANGED" — no re-fetch, no
           // content read branches on the value. Only the non-dominant,
           // never-based-before arm re-fetches (`embedding_refresh.rs:751`,
           // after its own base-publish CAS) before returning. See
           // `SELF_FETCHED_RECORD_ALLOWED`'s `refreshable_record` entry for
           // why this dominant-arm behaviour is what that entry's review
           // actually depends on.
    ),
];

/// Pattern 4 — `(file, function name, ordinal)` (see [`assign_ordinals`]'s
/// doc), same shape as
/// `ANCHOR_RETURN_ALLOWED`/`RECORD_VERSION_BRANCH_ALLOWED`. Six entries, not
/// one: dropping the `.get_result_table(` idiom conjunct
/// (`self_fetched_record_version_hits`'s own doc) raised the real-surface
/// hit count from 1 to 6 — the measured "five more, not hundreds" this
/// contract pays for wider coverage.
///
/// The `compact_embeddings`/`expire_versions` entries below cite their
/// current declaration lines for human traceability only — an unrelated
/// merge elsewhere in this same file could still drift a line number without
/// the function itself changing, desynchronizing a line-keyed allowlist from
/// the code it names, which is exactly the failure the ordinal key avoids.
const SELF_FETCHED_RECORD_ALLOWED: &[(&str, &str, usize)] = &[
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "refresh_embeddings",
        1, // ordinal 1 — the only `refresh_embeddings` in this file; line 340 today
           // This function never itself calls `.get_result_table(` — it calls
           // the wrapper `refreshable_record`. Reads `record.current_version`
           // once (`parent_version`, line 357) off the record
           // `refreshable_record` returned, then uses that SAME value to
           // resolve the parent manifest (`store.resolve_version_manifest(
           // &record, parent_version)`, line 366) — one fetch, one field, one
           // value threaded through to both the version DECISION and the
           // content READ. There is no second, independent resolution to
           // straddle against.
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "refreshable_record",
        1, // ordinal 1 — the only `refreshable_record` in this file; line 579 today
           // `InferenceSession::refreshable_record` — step 0's readiness GATE:
           // it self-fetches the record from a bare table name via
           // `self.catalog().get_result_table(table)`, then reads
           // `record.current_version` to reject a table whose CURRENT version
           // row is not `ready` (`NotRefreshableReason::CurrentVersionUnavailable`).
           // That read is used ONLY for this readiness check — it is never
           // returned, never becomes an anchor, and never pairs with a content
           // read here.
           //
           // `REFRESHABLE_RECORD_CALLERS` above, machine-checked, shows THREE
           // in-tree callers: `refresh_embeddings`, and the two destructive
           // ones, `compact_embeddings` (PUBLISHES a new version —
           // `version.publish(...)`, `embedding_refresh.rs:1170`) and
           // `expire_versions` (PERMANENTLY REAPS old ones —
           // `store.reap_expired_version(...)`, its own doc comment calls
           // this "a PERMANENT delete"). All three are reviewed in their own
           // entries.
           //
           // The safety mechanism is NOT that `ensure_base_version` re-fetches
           // the record before `refresh_embeddings`/`compact_embeddings` ever
           // pair this value with content — that is false on the DOMINANT arm.
           // `RECORD_VERSION_BRANCH_ALLOWED`'s own `ensure_base_version` entry
           // documents that when `record.current_version.is_some()` — true for
           // every refresh/compaction after a table's first-ever base publish —
           // it is a guard clause returning the SAME record UNCHANGED, no
           // re-fetch at all.
           //
           // What actually protects `refresh_embeddings` and
           // `compact_embeddings` is not a re-fetch on that arm; it is that
           // there is only ONE resolution of the record/version in the whole
           // call, threaded through unchanged: this function's single catalog
           // read supplies both the readiness check's `current_version` AND
           // (via the record it returns, carried forward unchanged by
           // `ensure_base_version`'s guard clause on the dominant arm) the SAME
           // field the caller later reads as `parent_version` to resolve the
           // manifest. One fetch, one field, one value — never two independent
           // resolutions to straddle. On the non-dominant, never-based arm,
           // `ensure_base_version` DOES re-fetch once
           // (`embedding_refresh.rs:751`, after its own base-publish CAS), and
           // THAT fresh record is what flows forward instead — still a single
           // resolution per call, just a different one depending on the arm.
           // Either way the anchor and the content pairing this gate polices
           // come from the SAME record object, never two.
           //
           // `expire_versions` never calls `ensure_base_version` at all: it
           // reads `record.current_version` from this function's single fetch
           // and uses that SAME value, once, to resolve the retention manifest
           // (`store.resolve_version_manifest(&record, current)`) its deletion
           // loop reaps against — again one resolution, not a pairing of two.
           // It never constructs or persists an `InputAnchor`; the value read
           // here never becomes a provenance artifact, only a retention-set
           // selector for a destructive delete — the same "candidate SELECTION
           // is out of this contract's scope" residual `resolve_search_mode_local`'s
           // entry above discloses for a different function. A version publish
           // landing between this read and the reap could pick a stale
           // retention set (a garbage-collection race its own `M5` doc comment
           // already tracks) — never mint a mismatched anchor/content pair,
           // because no anchor is ever minted on this path.
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "compact_embeddings",
        1, // ordinal 1 — the only `compact_embeddings` in this file; line 1024 today
           // No direct `.get_result_table(` call in its own body. Same
           // mechanism as `refresh_embeddings`: calls `refreshable_record`
           // then `ensure_base_version`, reads `record.current_version` once
           // as `parent_version` (line 1027), and uses that SAME value to
           // resolve `store.resolve_version_manifest(&record, parent_version)`
           // (line 1034) for the content it compacts. See `refreshable_record`'s
           // entry above for the full mechanism review.
    ),
    (
        "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
        "expire_versions",
        1, // ordinal 1 — the only `expire_versions` in this file; line 1206 today
           // Calls `refreshable_record` (never `ensure_base_version`), reads
           // `record.current_version` once (`current`, line 1209) and uses
           // that SAME value to resolve the retention manifest
           // (`store.resolve_version_manifest(&record, current)`, line 1228)
           // its deletion loop reaps every OTHER version against. See
           // `refreshable_record`'s entry above for why this never mints an
           // anchor.
    ),
    (
        "crates/jammi-db/src/catalog/version_repo.rs",
        "classify_ready_cas_miss",
        1, // ordinal 1 — the only `classify_ready_cas_miss` in this file; line 245 today
           // Takes `target: Option<CasTarget>`, never a `ResultTableRecord` —
           // `CasTarget` merely happens to name its own version field
           // `current_version` too, which is what this text-only check
           // (correctly) matches on. Reads `row.current_version` only to
           // build a typed comparison error (`ParentMoved { expected, found:
           // row.current_version }` vs. `CasFailed`); never persists an
           // anchor or reads content.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "reconcile_ready_manifests",
        1, // ordinal 1 — the only `reconcile_ready_manifests` in this file; line 1941 today
           // A read-only recovery sweep over already-`ready` tables: for
           // each, checks whether the CURRENT version's manifest
           // sidecar exists on disk and fails the row/version if it does not
           // (corruption repair). Never constructs or persists an
           // `InputAnchor`; no content is read here at all, only manifest
           // EXISTENCE.
    ),
];

/// Pattern 3 — `(file, function name, declaration line, allowed occurrence
/// count)`. Keyed on the (path, function, ORDINAL) SITE, not on `file` alone
/// (a per-file `usize` allowance would let a NEW unpinned read inside an
/// already-allowlisted file pass review-free whenever an existing one in a
/// DIFFERENT function of that same file was deleted in the same commit),
/// not on the bare file path either (a same-named file in a different
/// subdirectory must not silently inherit an allowance reviewed for a wholly
/// different file), and not on a declaration LINE for the third element (a
/// same-named sibling FUNCTION in the same file would collide under a
/// two-element `(file, name)` key and have its count summed into this one —
/// see [`assign_ordinals`]'s doc for why an ORDINAL, not a line number, is
/// the stable third element).
const SESSION_LITERAL_ALLOWED: &[(&str, &str, usize, usize)] = &[
    (
        "crates/jammi-ai/src/pipeline/graph_propagation.rs",
        "edge_scan_sql",
        1, // ordinal 1 — the only `edge_scan_sql` in this file; line 854 today
        1,
        // The S9 `neighbor_graph` edge scan — the EDGE relation, never the
        // pinned embedding table `PinnedSource` covers. See this same file's
        // `edge_source_anchor` note in `ANCHOR_RETURN_ALLOWED` for the
        // anchor/content pairing this residual leaves open.
    ),
    (
        "crates/jammi-ai/src/pipeline/graph_neighbourhood.rs",
        "load_neighbor_graph_edges",
        1, // ordinal 1 — the only `load_neighbor_graph_edges` in this file; line 511 today
        1,
        // Same class, the S9 edge relation.
    ),
    (
        "crates/jammi-db/src/index/exact.rs",
        "exact_vector_search",
        1, // ordinal 1 — the only `exact_vector_search` in this file; line 135 today
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
        1, // ordinal 1 — the only `read_vectors` in this file; line 977 today
        1,
        // See `RECORD_VERSION_BRANCH_ALLOWED`'s entry for this same
        // function — builds a `TableReference::bare(format!("jammi.{table}"))`
        // to read through this session's own registration rather than a
        // pin. Disclosed, not closed.
    ),
    (
        "crates/jammi-db/src/session.rs",
        "read_vector_by_key",
        1, // ordinal 1 — the only `read_vector_by_key` in this file; line 1045 today
        1,
        // Same shape as `read_vectors`, a different function in the same
        // file — kept as its own site so the two allowances cannot be
        // spent interchangeably.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "registered_name",
        1, // ordinal 1 — the only `registered_name` in this file
        1,
        // `TrainingSetTable::registered_name` — the key a materialised
        // training set is bound under, the accessor `sql_relation` (and
        // through it `jammi_ai::fine_tune::training_set::read_back_sql`)
        // quotes for the read-back. There is no version to straddle here.
        //
        // Two independent reasons, the second of which is the operative one:
        //
        // (1) KIND-INDEPENDENT: neither `registered_name` nor `sql_relation`
        // resolves a version at ALL. They format `jammi.{table_name}` off the
        // record the caller already holds; no `current_version` is read on
        // this path, so there is no anchor/content pair here to mismatch —
        // only a bare registration key. What the straddle question turns on
        // is whether the RELATION that key names can change under a reader,
        // which is (2).
        //
        // (2) The relation is versionless because every verb that could
        // publish a version over a result table REFUSES this kind, in
        // `jammi-ai`, before it writes: `InferenceSession::refreshable_record`
        // (`crates/jammi-ai/src/pipeline/embedding_refresh.rs`) returns
        // `NotRefreshable { reason: NotEmbeddingTable }` for any record whose
        // `kind` is not `ResultTableKind::Model`, and it dominates the
        // enumerated writer set: `refresh_embeddings` and `compact_embeddings`
        // both call it before `ensure_base_version` (→
        // `version_repo.rs::publish_base_version`) and before
        // `allocate_version` (→ `building_version.rs`), and `expire_versions`
        // — the deleting third verb through that gate — calls it too. Pinned
        // by `training_set::refresh_and_compaction_refuse_a_training_set_leaving_it_versionless`,
        // which materialises a real `ready` `TrainingSet` row, drives all
        // three verbs at it, and asserts the typed refusal plus
        // `current_version == None`, `next_version` unmoved and zero version
        // rows afterwards.
        //
        // R-A — this is CALLER DISCIPLINE, not a storage-layer
        // impossibility, and the earlier version of this entry got that
        // wrong. The db owner's executed probe publishes a base version on a
        // `TrainingSet` row through `Catalog::publish_base_version`
        // (`current_version` None → `Some(0)`) and allocates a second through
        // `ResultStore::allocate_version` (`Ok(1)`): jammi-db accepts both.
        // The `kind = 'model'` predicate in `resolve_embedding_table`
        // (`result_repo.rs`, pinned by the db-side oracle
        // `a_training_set_never_resolves_as_a_sources_embedding_table`) is
        // still true and still reviewed, but it covers only the
        // source_id-ADDRESSED half — how a source resolves to ITS embedding
        // table — and `refresh_embeddings` / `compact_embeddings` never call
        // it. So the claim held by this entry is exactly: no caller in this
        // workspace publishes a version on a training set, because the one
        // choke point they all pass through refuses the kind. A future verb
        // that reaches `publish_base_version` WITHOUT going through
        // `refreshable_record` would break it and nothing here would catch
        // that; the writer set above is a hand enumeration (`grep -rn
        // 'publish_base_version\|allocate_version' crates/jammi-ai/src`),
        // re-derived when a version-writing path is added.
        //
        // The table is also immutable by construction: it is written once
        // through the single `building -> ready` funnel and never rewritten.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "register_table",
        1, // ordinal 1 — the only `register_table` in this file; line 1073 today
        1,
        // The registration write itself — defines what `jammi.{name}` maps
        // to, never a read.
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "bind_result_table",
        1, // ordinal 1 — the only `bind_result_table` in this file; line 2516 today
        2,
        // Its two `add_result_table` calls (its documented "Read class"
        // residual — see `RECORD_VERSION_BRANCH_ALLOWED`'s entry). Both
        // sites live in this one function, so the site-bound count here is
        // 2, not 1 — the allowance binds to the FUNCTION, not to each
        // individual occurrence, so two reviewed sites in one already-
        // reviewed function still share one entry.
    ),
    (
        "crates/jammi-db/src/store/result_schema.rs",
        "deregister_result_tables",
        1, // ordinal 1 — the only `deregister_result_tables` in this file; line 193 today
        1,
        // Read in full (`deregister_result_tables`,
        // `crates/jammi-db/src/store/result_schema.rs:193-211`
        // today): this function makes exactly ONE call against the schema
        // provider it resolves — `provider.remove(&format!("jammi.{name}"))`
        // (line 209) — and no other. It never calls `.table(`/
        // `.table_exist(`, never reads `.current_version` off any record,
        // never constructs or returns an `InputAnchor`/`CurrentAnchor`.
        // `ResultTableSchemaProvider::remove` (this same file, the
        // `impl ResultTableSchemaProvider` block above `deregister_table`)
        // only pops an entry out of the in-memory registration map and
        // returns the provider that WAS there — no catalog read, no version
        // resolution, no content access happens anywhere on this path. That
        // makes it a genuinely different class from the anchor/content
        // straddle this gate exists to catch, which requires resolving a
        // VERSION and then reading CONTENT under it: there is no content
        // read here to straddle against, only a registration entry being
        // torn down.
        //
        // Per R-A: this is a REVIEWED CLAIM about this one function's
        // behaviour (its body was read in full and every call in it
        // enumerated by hand above), not a machine-checked one — nothing in
        // this gate's detectors verifies "this function never resolves a
        // version or reads content" the way `caller_set_claims_match_reality`
        // machine-verifies a caller SET. A future edit that adds a `.table(`/
        // `.get_result_table(`/`.current_version` read to this same function
        // would not be caught by anything here and would need a fresh
        // review, not a renewed allowlist entry.
    ),
    (
        "crates/jammi-ai/src/session.rs",
        "infer_ordered_read_back_sql",
        1, // ordinal 1 — the only `infer_ordered_read_back_sql` in this file; line 2094 today
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
            .find(|(f, n, o)| *f == hit.file && *n == hit.name && *o == hit.ordinal);
        assert!(
            allowed_name.is_some(),
            "{}:{} (ordinal {}) `fn {}` returns a type carrying `InputAnchor`/`CurrentAnchor` — a \
             version-resolved anchor value with no paired content. This is either a NEW straddle-\
             shaped site (route it through `ResultStore::pin_current_version`/\
             `PinnedSource::input_anchor` instead) or a reviewed exception that belongs in this \
             test's `ANCHOR_RETURN_ALLOWED` list, keyed on (file, function name, ordinal — see \
             `assign_ordinals`'s doc), with the same review its existing entries carry.",
            hit.file,
            hit.line,
            hit.ordinal,
            hit.name
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
            .find(|(f, n, o)| *f == hit.file && *n == hit.name && *o == hit.ordinal);
        assert!(
            allowed_name.is_some(),
            "{}:{} (ordinal {}) `fn {}` takes a bare `ResultTableRecord` and re-derives \
             `.current_version` from it in its own body, rather than taking an already-resolved \
             version/manifest/`PinnedSource` as a parameter. This is either a NEW straddle-shaped \
             site or a reviewed exception that belongs in `RECORD_VERSION_BRANCH_ALLOWED`, keyed \
             on (file, function name, ordinal — see `assign_ordinals`'s doc), with the same review \
             its existing entries carry.",
            hit.file,
            hit.line,
            hit.ordinal,
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
            .find(|(f, n, o)| *f == hit.file && *n == hit.name && *o == hit.ordinal);
        assert!(
            allowed_name.is_some(),
            "{}:{} (ordinal {}) `fn {}` reads a self-obtained record's `.current_version` field, \
             rather than taking an already-resolved record/version/manifest/`PinnedSource` as a \
             parameter (pattern 2's shape) or content read (pattern 3's shape). This is either a \
             NEW straddle-shaped site (route it through `ResultStore::pin_current_version` \
             instead) or a reviewed exception that belongs in `SELF_FETCHED_RECORD_ALLOWED`, \
             keyed on (file, function name, ordinal — see `assign_ordinals`'s doc), with the same \
             review its existing entries carry.",
            hit.file,
            hit.line,
            hit.ordinal,
            hit.name
        );
    }
}

#[test]
fn no_new_unpinned_session_registration_literal() {
    let surface = scan_surface();
    let sites = session_registration_literal_sites(&surface);
    for ((file, name, ordinal), count) in &sites {
        let allowed = SESSION_LITERAL_ALLOWED
            .iter()
            .find(|(f, n, o, _)| f == file && n == name && o == ordinal)
            .map(|(_, _, _, c)| *c)
            .unwrap_or(0);
        assert!(
            *count <= allowed,
            "{file}: fn {name} (ordinal {ordinal}) has {count} occurrence(s) of the bare session-\
             registration literal `\"jammi.{{`, {allowed} audited/allowed for THIS SITE (the \
             ordinal-th function of this name in this file — see `assign_ordinals`'s doc) — an \
             allowance in a DIFFERENT function of the same file, even one with the same NAME, \
             never covers this one. A NEW site must read through \
             `ResultStore::pin_current_version`/`pinned_provider`, never construct the session-\
             registered `jammi.{{table}}` reference directly. If this IS an audited exception, \
             add it to `SESSION_LITERAL_ALLOWED` — keyed on this same (path, function name, \
             ordinal) — with the same review its existing entries had."
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
    let src = concat!(
        r#"
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for `anchor_shaped_return_hits` — synthetic producer text fed to that detector, not real code in this file
        r#"            pub async fn sneaky_anchor(
                &self,
                t: &ResultTableRecord,
            ) -> Result<InputAnchor> {
                Ok(InputAnchor::result_digest("x", &digest))
            }
        }
    "#,
    );
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
    let src = concat!(
        r#"
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for `anchor_shaped_return_hits`'s negative control — synthetic producer text, not real code in this file
        r#"        pub async fn compare_anchor(&self, anchor: &InputAnchor) -> Result<CurrentAnchor> {
            Ok(CurrentAnchor::Undecidable)
        }
"#,
        // kernel-oracles: fn-in-literal reviewed: same negative-control fixture's second synthetic function — not real code in this file
        r#"        pub fn benign(&self, anchor: &InputAnchor) -> bool {
            true
        }
    "#,
    );
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
    let src = concat!(
        r#"
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for `bare_record_version_branch_hits` — synthetic producer text fed to that detector, not real code in this file
        r#"        async fn sneaky_read(&self, table: &ResultTableRecord) -> Result<Vec<u8>> {
            if let Some(v) = table.current_version {
                return read_version(v).await;
            }
            Ok(vec![])
        }
    "#,
    );
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
    let src = concat!(
        r#"
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the explicit-version negative control — synthetic producer text fed to `bare_record_version_branch_hits`, not real code in this file
        r#"        pub async fn resolve_version_manifest(
            &self,
            table: &ResultTableRecord,
            version: i64,
        ) -> Result<VersionManifest> {
            self.read_version_manifest(&table.table_name, version).await
        }
    "#,
    );
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
    // A producer that takes a bare table NAME, resolves the record itself
    // through the catalog, reads `.current_version` off the record it just
    // fetched, and returns the version identity as a plain `String`.
    // Patterns 1-3 all return empty on this exact shape.
    let src = concat!(
        r#"
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the self-fetched-record-version detector (a self-fetch-then-read shape) — synthetic producer text, not real code in this file
        r#"            pub async fn anchor_identity_for(&self, table_name: &str) -> Result<Option<String>> {
                let record = self.catalog.get_result_table(table_name).await?.unwrap();
                let Some(version) = record.current_version else { return Ok(None); };
                let row = self.catalog.get_result_table_version(table_name, version).await?;
                Ok(row.and_then(|r| r.identity))
            }
        }
    "#,
    );
    let surface = vec![("__probe__.rs".to_string(), src.to_string())];
    let hits = self_fetched_record_version_hits(&surface);
    assert_eq!(
        hits.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
        vec!["anchor_identity_for"],
        "the self-fetched-record-version detector did not fire on a synthetic function that \
         resolves a `ResultTableRecord` itself via `.get_result_table(` from a bare table name \
         and reads `.current_version` off it — this is exactly pattern 4's own reason to exist"
    );
    // The property pattern 4 exists to close, re-verified rather than
    // asserted: patterns 1 and 2 are confirmed still blind to this exact
    // shape (its return type is `Result<Option<String>>`, not anchor-shaped;
    // it never takes `ResultTableRecord` as a parameter).
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
    // A producer taking a bare table name, self-fetching the record, and
    // returning `current_version_identity`'s value as a plain `String` — the
    // exact shape pattern 4 exists to catch — delegates to a NAMED, reviewed
    // helper here rather than reading `.current_version` itself. Clearance by
    // a callee's NAME alone is the same failure this file's own module doc
    // disowns for a return TYPE's name; the actual protection is the
    // machine-checked caller SET below, not an assumption that this shape can
    // never be flagged.
    //
    // The field-boundary regression this control also exercises is real and
    // stays: without `reads_current_version_field`'s exact-boundary check,
    // the substring `.current_version` inside `.current_version_identity(`
    // would false-positive on this exact shape — `ResultStore::current_anchor`'s
    // real body on today's surface. That assertion is kept below, but framed
    // as what it is: pattern 4 has a disclosed, uncovered blind spot for
    // ANY delegation to a differently-named helper (the module doc's own
    // "cross-function" limit), not a proof this shape is safe.
    let src_delegates = concat!(
        r#"
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for pattern 4's reviewed-delegation control — synthetic producer text, not real code in this file
        r#"            pub async fn wraps_identity(&self, table_name: &str) -> Result<Option<String>> {
                let record = self.catalog.get_result_table(table_name).await?.unwrap();
                self.current_version_identity(&record).await
            }
        }
    "#,
    );
    let surface = vec![("__probe__.rs".to_string(), src_delegates.to_string())];
    let hits = self_fetched_record_version_hits(&surface);
    assert!(
        hits.is_empty(),
        "UNCOVERED, per R-A, not a guarantee: pattern 4 does not see this \
         self-fetch-then-delegate shape — a change here would mean the field-boundary check \
         regressed to a bare substring match: {hits:?}"
    );
    // What actually protects today's real surface against this exact shape
    // is NOT pattern 4 (which cannot see it) but `current_version_identity`'s
    // own machine-checked caller set (`CURRENT_VERSION_IDENTITY_CALLERS`,
    // checked by `caller_set_claims_match_reality`): a real `wraps_identity`
    // calling `.current_version_identity(` would be a THIRD caller, and that
    // check would go red the moment it landed, even though this detector
    // never would. Demonstrated directly: this synthetic probe's own
    // `(file, fn)` pair is not in the declared set, and IS what
    // `callers_of` would report if this were real production code sharing
    // this file's surface.
    assert!(
        !CURRENT_VERSION_IDENTITY_CALLERS
            .iter()
            .any(|(f, n)| *f == "__probe__.rs" && *n == "wraps_identity"),
        "the pinned caller set must not already contain this synthetic probe's site"
    );

    // Negative control 2: takes the record as a PARAMETER (pattern 2's own
    // precondition) rather than self-fetching it. Must not ALSO fire
    // pattern 4 — the two patterns partition the surface, they do not both
    // claim the same site.
    let src_parameterized = concat!(
        r#"
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for pattern 4's negative control 2 — synthetic producer text fed to `self_fetched_record_version_hits`, not real code in this file
        r#"        async fn sneaky_read(&self, table: &ResultTableRecord) -> Result<Vec<u8>> {
            if let Some(v) = table.current_version {
                return read_version(v).await;
            }
            Ok(vec![])
        }
    "#,
    );
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
    // Without `find_matching_angle`'s `->`-as-one-token handling, the `->`
    // inside this `Fn` trait bound would close the generic parameter list
    // early via the plain depth-counter parens/braces use, desynchronizing
    // the scan from the following `(` and dropping this WHOLE function from
    // `find_fn_regions`'s output — invisible to every detector despite a
    // return type that is literally `Result<InputAnchor>`.
    let src = concat!(
        r#"
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the generic-arrow-bound region finder — synthetic producer text (a `Fn(&str) -> String` trait bound), not real code in this file
        r#"            pub async fn anchor_with<F: Fn(&str) -> String>(&self, rec: &ResultTableRecord, f: F) -> Result<InputAnchor> {
                let v = rec.current_version.unwrap();
                Ok(InputAnchor::result_digest(f(&rec.table_name), v))
            }
        }
    "#,
    );
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
    let src = concat!(
        r#"
        impl Foo {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the plain-generics negative control — synthetic producer text, not real code in this file
        r#"            pub fn plain<T: Clone, U>(&self, a: T, b: U) -> Result<InputAnchor> {
                todo!()
            }
        }
    "#,
    );
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
        sites.get(&("__probe__.rs".to_string(), "<module-scope>".to_string(), 0)),
        Some(&1),
        "the session-registration-literal detector did not fire on a synthetic \
         `TableReference::bare(format!(\"jammi.{{}}\", ..))` call at module scope"
    );
}

#[test]
fn falsification_session_registration_literal_binds_to_its_enclosing_function() {
    // The SAME literal shape, once inside a named function, must be bound to
    // that function's SITE, not to the file (or to "<module-scope>") —
    // proving the site-binding attributes the hit correctly rather than
    // merely still finding it somewhere. The third key element is each
    // function's ORDINAL, not its declaration line — both `one` and `two`
    // are the first (and only)
    // function of their name in this synthetic file, so both key as
    // ordinal 1 regardless of which source line either starts on.
    let src = concat!(
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the site-binding fix — synthetic producer text fed to `session_registration_literal_sites`, not real code in this file
        "fn one(table: &str) {\n",
        "    let a = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "}\n",
        // kernel-oracles: fn-in-literal reviewed: same site-binding fixture's second synthetic function — not real code in this file
        "fn two(table: &str) {\n",
        "    let b = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "    let c = TableReference::bare(format!(\"jammi.{}\", table));\n",
        "}\n",
    );
    let sites =
        session_registration_literal_sites(&[("__probe__.rs".to_string(), src.to_string())]);
    assert_eq!(
        sites.get(&("__probe__.rs".to_string(), "one".to_string(), 1)),
        Some(&1),
        "fn `one`'s single occurrence must be bound to `one` (ordinal 1, its only occurrence in \
         this file), not to the file total: {sites:?}"
    );
    assert_eq!(
        sites.get(&("__probe__.rs".to_string(), "two".to_string(), 1)),
        Some(&2),
        "fn `two`'s two occurrences must be bound to `two` (ordinal 1, its only occurrence in \
         this file), and only `two`: {sites:?}"
    );
    assert!(
        !sites
            .keys()
            .any(|(f, n, _)| f == "__probe__.rs" && n == "<module-scope>"),
        "no occurrence here is outside a function, so the module-scope sentinel must not appear: \
         {sites:?}"
    );
}

#[test]
fn falsification_ordinal_survives_a_line_shift_above_it() {
    // The ordinal-vs-line stability claim, proved rather than argued: two
    // synthetic files, otherwise byte-identical, differ only by FOUR extra
    // doc-comment lines inserted ABOVE the reviewed function — exactly the
    // shape of edit that would desynchronize a line-keyed allowlist from
    // `compact_embeddings`/`expire_versions` if an unrelated merge added a
    // four-line comment above both.
    let src_before = concat!(
        r#"
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for the ordinal-vs-line stability property — synthetic producer text, not real code in this file
        r#"            pub async fn drifting_anchor(
                &self,
                t: &ResultTableRecord,
            ) -> Result<InputAnchor> {
                Ok(InputAnchor::result_digest("x", &digest))
            }
        }
    "#,
    );
    // The SAME function, four lines further down — the shape of an
    // unrelated merge adding a doc comment above it, with the reviewed
    // function's own text byte-identical.
    let src_after = concat!(
        r#"
        // one
        // two
        // three
        // four
        impl ResultStore {
"#,
        // kernel-oracles: fn-in-literal reviewed: same ordinal-vs-line stability fixture, four lines further down — not real code in this file
        r#"            pub async fn drifting_anchor(
                &self,
                t: &ResultTableRecord,
            ) -> Result<InputAnchor> {
                Ok(InputAnchor::result_digest("x", &digest))
            }
        }
    "#,
    );

    let surface_before = vec![("__probe__.rs".to_string(), src_before.to_string())];
    let surface_after = vec![("__probe__.rs".to_string(), src_after.to_string())];
    let hits_before = anchor_shaped_return_hits(&surface_before);
    let hits_after = anchor_shaped_return_hits(&surface_after);
    assert_eq!(
        hits_before.len(),
        1,
        "expected exactly one hit before the shift"
    );
    assert_eq!(
        hits_after.len(),
        1,
        "expected exactly one hit after the shift"
    );
    let (before, after) = (&hits_before[0], &hits_after[0]);

    // The property a LINE-keyed allowlist relies on breaks: the site's line
    // moves.
    assert_ne!(
        before.line, after.line,
        "this fixture is supposed to move the function's line by inserting four lines above it — \
         if the lines are equal the fixture itself is broken, not the property under test"
    );
    assert_eq!(
        after.line,
        before.line + 4,
        "the fixture inserts exactly four lines above the function; its line must shift by \
         exactly four"
    );

    // The property an ORDINAL-keyed allowlist relies on holds: the site's
    // ordinal (1st `drifting_anchor` in this file, both before and after) is
    // unchanged by the shift.
    assert_eq!(
        before.ordinal, after.ordinal,
        "the ordinal must be identical before and after a same-line-count edit made strictly \
         ABOVE the site that inserts no same-named sibling — this is the whole property the \
         ordinal key rests on"
    );

    // Demonstrated concretely, not just asserted on the `Hit` fields
    // directly: build a one-entry `ANCHOR_RETURN_ALLOWED`-shaped allowlist
    // from the BEFORE hit (as a reviewer would, reviewing the pre-edit code)
    // and confirm it still matches the AFTER hit — the real
    // `no_new_anchor_shaped_return_without_review` lookup, reproduced
    // here on a controlled fixture so the "the gate still passes" claim is
    // executed, not narrated.
    let allowlist_from_before: &[(&str, &str, usize)] =
        &[("__probe__.rs", "drifting_anchor", before.ordinal)];
    let still_allowed = allowlist_from_before
        .iter()
        .any(|(f, n, o)| *f == after.file && *n == after.name && *o == after.ordinal);
    assert!(
        still_allowed,
        "an allowlist entry keyed on the ordinal derived from the PRE-shift code must still match \
         the POST-shift hit — a line-keyed allowlist would desync from exactly this edit instead"
    );
    // The failure mode this replaces, shown directly rather than merely
    // asserted away: the equivalent LINE-keyed lookup from the same review
    // does NOT survive the shift.
    let line_keyed_allowlist_from_before: &[(&str, &str, usize)] =
        &[("__probe__.rs", "drifting_anchor", before.line)];
    let would_have_failed_under_line_key = !line_keyed_allowlist_from_before
        .iter()
        .any(|(f, n, l)| *f == after.file && *n == after.name && *l == after.line);
    assert!(
        would_have_failed_under_line_key,
        "this fixture is supposed to reproduce a line key's own failure mode — if a line-keyed \
         lookup from the pre-shift review ALSO still matched post-shift, the fixture no longer \
         demonstrates what a line key actually breaks on"
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
        // kernel-oracles: fn-in-literal reviewed: falsification fixture for `mask_non_code`'s brace-counting — synthetic producer text, not real code in this file
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
    // The inverse control, generalized to all four patterns: an allowance
    // whose site no longer produces that hit — the
    // code was fixed, renamed, or removed — must shrink with it. A
    // permanent allowance is dead slack a LATER, different site could spend
    // without any new review at all, which is exactly the failure mode this
    // contract names for the sweep it replaces.
    let surface = scan_surface();

    let anchor_hits: HashSet<(String, String, usize)> = anchor_shaped_return_hits(&surface)
        .into_iter()
        .map(|h| (h.file, h.name, h.ordinal))
        .collect();
    for (file, name, ordinal) in ANCHOR_RETURN_ALLOWED {
        assert!(
            anchor_hits.contains(&(file.to_string(), name.to_string(), *ordinal)),
            "ANCHOR_RETURN_ALLOWED lists {file} fn {name} (ordinal {ordinal}), but the current \
             scan no longer finds an anchor-shaped return there — shrink this list to match (the \
             fix landed, the function moved/was renamed, or a same-named sibling was added/removed \
             ahead of it, shifting its ordinal)."
        );
    }

    let record_hits: HashSet<(String, String, usize)> = bare_record_version_branch_hits(&surface)
        .into_iter()
        .map(|h| (h.file, h.name, h.ordinal))
        .collect();
    for (file, name, ordinal) in RECORD_VERSION_BRANCH_ALLOWED {
        assert!(
            record_hits.contains(&(file.to_string(), name.to_string(), *ordinal)),
            "RECORD_VERSION_BRANCH_ALLOWED lists {file} fn {name} (ordinal {ordinal}), but the \
             current scan no longer finds a bare-record version-branch there — shrink this list to \
             match."
        );
    }

    let self_fetched_hits: HashSet<(String, String, usize)> =
        self_fetched_record_version_hits(&surface)
            .into_iter()
            .map(|h| (h.file, h.name, h.ordinal))
            .collect();
    for (file, name, ordinal) in SELF_FETCHED_RECORD_ALLOWED {
        assert!(
            self_fetched_hits.contains(&(file.to_string(), name.to_string(), *ordinal)),
            "SELF_FETCHED_RECORD_ALLOWED lists {file} fn {name} (ordinal {ordinal}), but the \
             current scan no longer finds a self-fetched-record version read there — shrink this \
             list to match."
        );
    }

    let literal_sites = session_registration_literal_sites(&surface);
    for (file, name, ordinal, allowed) in SESSION_LITERAL_ALLOWED {
        let current = literal_sites
            .get(&(file.to_string(), name.to_string(), *ordinal))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            current, *allowed,
            "{file}: fn {name} (ordinal {ordinal}): SESSION_LITERAL_ALLOWED expects exactly \
             {allowed} occurrence(s) of `\"jammi.{{`, the current scan finds {current} — update \
             this allowlist to match."
        );
    }

    let anchor_callers = callers_of(&surface, "current_anchor");
    let producing_callers = callers_of(&surface, "producing_descriptor");
    let refreshable_callers = callers_of(&surface, "refreshable_record");
    let provider_callers = callers_of(&surface, "current_version_provider");
    let identity_callers = callers_of(&surface, "current_version_identity");
    let vectors_callers = callers_of(&surface, "read_vectors");
    // Inverse control for the caller-set consts themselves: a
    // declared caller that no longer calls `.method(` at all — renamed,
    // removed, or refactored away — must shrink the const, the same
    // discipline the four shape-hit allowlists above already carry.
    // `caller_set_claims_match_reality` already asserts this per-check; this
    // is the same assertion folded into the one test every entry's inverse
    // control lives in, so a reviewer checking "does every allowlist here
    // have a matching inverse control" finds all five without hunting.
    for (f, n) in CURRENT_ANCHOR_CALLERS {
        assert!(
            anchor_callers.contains(&(f.to_string(), n.to_string())),
            "CURRENT_ANCHOR_CALLERS lists {f}::{n}, but it no longer calls `.current_anchor(`"
        );
    }
    for (f, n) in PRODUCING_DESCRIPTOR_CALLERS {
        assert!(
            producing_callers.contains(&(f.to_string(), n.to_string())),
            "PRODUCING_DESCRIPTOR_CALLERS lists {f}::{n}, but it no longer calls \
             `.producing_descriptor(`"
        );
    }
    for (f, n) in REFRESHABLE_RECORD_CALLERS {
        assert!(
            refreshable_callers.contains(&(f.to_string(), n.to_string())),
            "REFRESHABLE_RECORD_CALLERS lists {f}::{n}, but it no longer calls \
             `.refreshable_record(`"
        );
    }
    for (f, n) in CURRENT_VERSION_PROVIDER_CALLERS {
        assert!(
            provider_callers.contains(&(f.to_string(), n.to_string())),
            "CURRENT_VERSION_PROVIDER_CALLERS lists {f}::{n}, but it no longer calls \
             `.current_version_provider(`"
        );
    }
    for (f, n) in CURRENT_VERSION_IDENTITY_CALLERS {
        assert!(
            identity_callers.contains(&(f.to_string(), n.to_string())),
            "CURRENT_VERSION_IDENTITY_CALLERS lists {f}::{n}, but it no longer calls \
             `.current_version_identity(`"
        );
    }
    for (f, n) in READ_VECTORS_CALLERS {
        assert!(
            vectors_callers.contains(&(f.to_string(), n.to_string())),
            "READ_VECTORS_CALLERS lists {f}::{n}, but it no longer calls `.read_vectors(`"
        );
    }
}

// ── #500 U2a — the graph arm's excised registration guard ──────────────────
//
// The graph arm's per-call sampled-pairs relation (`register_table`/
// `deregister_table` on the shared `SessionContext`, keyed by spec identity +
// job id) collided under reclaim: two overlapping materializations of ONE job
// id (a lease lost mid-sampling, then reclaimed by a second worker) still
// collide, because the resource the guard named is scoped to one CALL while
// the name was unique only per JOB. The guard is deleted outright, not
// narrowed: the graph arm is `origin/main`'s shape (sample in memory, train
// directly, no table); the follow-on unit that gives it a table of its own is
// <https://github.com/f-inverse/jammi-ai/issues/538>.
//
// This is the standing oracle for the DIRECT half of the underlying property:
// no file under `crates/jammi-ai/src/fine_tune/**` may itself WRITE a call to
// one of the verbs below, ever, because the session is not a per-call
// namespace — any token that is not unique per CALL (not per job, not per
// spec) collides under reclaim. Unlike every allowlist above, there is no
// allowance list for THIS half: the target count is zero, unconditionally, so
// a future re-introduction of a direct session-scoped call under `fine_tune/`
// fails this test rather than needing a reviewed entry.
//
// **This is a claim about literal call sites, not about the tree's runtime
// behaviour.** The true property this file exists to protect is that every
// name `fine_tune/` ever causes to be bound on the shared session — direct or
// several calls removed — is unique per CALL over an IMMUTABLE artifact; a
// zero-direct-hits count is sufficient for the direct half, but
// `training_set.rs`'s own call into `ResultStore::materialize_training_set`
// (which binds a name several calls further in) is exactly the kind of
// indirection no amount of widening the literal set below could ever make a
// directory-scoped scan see. That half is answered by enumerating every
// occurrence of these verbs anywhere under BOTH crates' `src` trees, not
// merely `fine_tune/**` — [`registration_verb_occurrences_are_all_reviewed`]
// and [`REGISTRATION_VERB_SITES`], further down this file — a SEPARATE
// oracle, not a hand-waved exception to this one. Tracing which of those
// sites a `fine_tune/` call can actually REACH (as opposed to reviewing every
// site unconditionally, reachable or not) is a distinct question, filed as
// <https://github.com/f-inverse/jammi-ai/issues/549>.
//
// **The DataFusion-wrapper shape this verb list closes.** A two-literal
// detector naming only `register_table(`/`deregister_table(` misses
// `SessionContext::register_batch` (DataFusion 54.1.0,
// `execution/context/mod.rs:537`), a one-line wrapper —
// `MemTable::try_new(..)` then `self.register_table(table_ref, Arc::new(table))`
// at `mod.rs:543` — that binds the exact same shared-session table name, one
// call deeper than the literal string a narrower detector would search for.
// The class this oracle enforces is not "the two literal spellings
// `register_table`/`deregister_table`"; it is a per-call resource bound to
// the shared `SessionContext` under a token that is not unique per call — and
// `SessionContext` exposes many more verbs shaped exactly like that, covered
// below.
//
// **Three further shapes this file closes, each by its own mechanism because
// each is its own shape, not one wider literal set:**
//   (i) `CatalogProvider::register_schema`/`deregister_schema` — not a
//       `SessionContext` method at all, reached as
//       `ctx.catalog(name).unwrap().register_schema(..)`. Closed by widening
//       [`PAIRED_REGISTRATION_VERBS`] (below) to cover it, since it is a
//       plain literal call-site pattern like every other entry there.
//   (ii) A DDL literal (`CREATE VIEW`/`CREATE TABLE`/`CREATE EXTERNAL
//       TABLE`/`CREATE SCHEMA`) handed to `SessionContext::sql` as a raw SQL
//       string — invisible to ANY verb-literal scan because the keyword
//       lives INSIDE a string literal, which [`mask_non_code`] blanks by
//       design. Closed by a SEPARATE detector,
//       [`fine_tune_ddl_relation_binding_hits`], that scans
//       [`mask_comments_only`]'s output (comments blanked, string content
//       intact) the same way [`session_registration_literal_sites`] scans
//       the unmasked text for `"jammi.{`.
//   (iii) An in-tree wrapper DEFINED OUTSIDE `fine_tune/` (a helper calling
//       `ctx.register_table(..)`, itself called from inside `fine_tune/`) —
//       invisible to a directory-scoped scan no matter how many verbs it
//       knows, because the LITERAL call site never appears inside
//       `fine_tune/**` at all. This is not a wider literal set's job for
//       THIS test: the wrapper's own call site is reviewed by
//       [`registration_verb_occurrences_are_all_reviewed`] instead, which
//       scans BOTH crates' whole `src` trees rather than one directory;
//       whether `fine_tune/` can actually reach that wrapper is a separate,
//       unanswered question (<https://github.com/f-inverse/jammi-ai/issues/549>),
//       not one this test or that whole-surface review resolves.
//
// **The whole verb surface, enumerated from the pinned source
// (`datafusion = "54.1"`, locked at `54.1.0` in `Cargo.lock`;
// `~/.cargo/registry/.../datafusion-54.1.0/src/execution/context/`), not
// hand-guessed** — every `pub fn`/`pub async fn` on `SessionContext` whose
// name starts `register_`/`deregister_`:
//
// IN — binds a resource under an explicit, caller-chosen token (a
// `TableReference`/`String`/`Url`/`&str`, distinct from the value being
// registered) into the shared session (or its shared `RuntimeEnv`), so two
// overlapping calls under the same token collide the same way
// `register_table` does under reclaim:
//   - `register_table`/`deregister_table` (`mod.rs:1925,1941`) — the
//     original pair; keyed by `table_ref: impl Into<TableReference>`.
//   - `register_batch` (`mod.rs:537`) — a one-line wrapper that calls
//     `self.register_table(table_ref, ..)` at `mod.rs:543`; same
//     `table_ref` token, one call removed — the DataFusion-wrapper shape
//     named above, closed by including it here rather than only the two
//     original literal spellings.
//   - `register_listing_table` (`mod.rs:1823`) — also calls
//     `self.register_table(table_ref, Arc::new(table))?` directly
//     (`mod.rs:1844`); same token.
//   - `register_arrow` (`mod.rs:1873`), `register_csv`
//     (`execution/context/csv.rs:63`), `register_json`
//     (`execution/context/json.rs:42`), `register_parquet`
//     (`execution/context/parquet.rs:66`), `register_avro`
//     (`execution/context/avro.rs:40`) — each is a thin format-specific
//     wrapper that builds `ListingOptions` and calls
//     `self.register_listing_table(table_ref, ..)`; same `table_ref` token,
//     two calls removed.
//   - `register_catalog` (`mod.rs:1898`) — binds `name: impl Into<String>`
//     into the shared `catalog_list()`; a second call with the same name
//     silently replaces the first ("Returns the CatalogProvider previously
//     registered for this name, if any"), so a caller that discards the
//     return value has no signal a collision even happened.
//   - `register_object_store`/`deregister_object_store` (`mod.rs:521,532`)
//     — bind `url: &Url` into the shared `RuntimeEnv`, itself shared across
//     every `SessionContext` built from it.
//   - `register_udtf`/`deregister_udtf` (`mod.rs:1603,1693`) —
//     `register_udtf(&self, name: &str, ..)` takes an explicit `name`
//     token, the identical shape to `register_table`.
//   - `register_udf`/`deregister_udf` (`mod.rs:1616,1670`),
//     `register_udaf`/`deregister_udaf` (`mod.rs:1642,1683`),
//     `register_udwf`/`deregister_udwf` (`mod.rs:1653,1688`) — the token is
//     the function's own `.name()` rather than a separate parameter, but it
//     is still a caller-determined string bound into the shared session's
//     function registry, and `register_udf`'s own doc states the collision
//     outcome directly (`mod.rs:1614`): "Any functions registered with the
//     udf name or its aliases will be OVERWRITTEN with this new function" —
//     the same silent-overwrite failure `register_catalog` has, not merely
//     an analogy.
//   - `register_higher_order_function`/`deregister_higher_order_function`
//     (`mod.rs:1630,1675`) — same shape as `register_udf`, one indirection
//     further (`HigherOrderUDF`'s own name).
//   - `register_schema`/`deregister_schema`
//     (`datafusion-catalog-54.1.0/src/catalog.rs:121,142`) — NOT a
//     `SessionContext` method itself: a `CatalogProvider` trait method,
//     reached from a `fine_tune/` caller as
//     `ctx.catalog(name).unwrap().register_schema(schema_name, provider)`.
//     `register_schema`'s own doc states the collision outcome directly ("If
//     a schema of the same name existed before, it is replaced in the
//     catalog and returned"), the identical silent-overwrite shape
//     `register_catalog`/`register_udf` already have above;
//     `deregister_schema` is its inverse. `ResultStore`'s own
//     `install_result_schema`, `crates/jammi-db/src/store/mod.rs:1183` calls
//     exactly this verb — which is why this literal set had to widen past
//     `SessionContext`'s own surface rather than staying a pure enumeration
//     of it.
//
// OUT — no caller-chosen per-resource token exists at all, so there is no
// name a `fine_tune/` arm could pick non-uniquely and no reclaim-shaped
// collision is possible regardless of call count; each is checked directly
// by `falsification_fine_tune_session_registration_is_detected_and_scoped`'s
// negative control, not merely omitted:
//   - `register_variable` (`mod.rs:1591`) — keyed by `VarType`, a two-value
//     enum (`System`/`UserDefined`), never a caller string; a fixed,
//     session-wide config slot, not a per-job/per-spec resource namespace.
//   - `register_relation_planner` (`mod.rs:1662`) — appends to an ordered
//     `Vec` of planners ("Planners are invoked in reverse registration
//     order"); there is no name to collide on and no way for a second
//     registration to replace or interfere with the first.
//   - `register_catalog_list` (`mod.rs:2053`) — replaces the WHOLE
//     `CatalogProviderList` in one call; a single global slot, not a
//     per-resource token — the same shape as `add_analyzer_rule`, session
//     configuration rather than a named per-call resource.
//   - `register_table_options_extension::<T>` (`mod.rs:2059`) — keyed by
//     the extension type's own `TypeId` (a generic parameter), never a
//     caller-supplied string; nothing under `fine_tune/` picks the Rust
//     TYPE it instantiates per job or per spec.

/// Verbs (54.1.0) with BOTH a `register_`/`deregister_` form, both binding
/// the shared session under the SAME caller-chosen token — see the
/// module-level comment above this section for why each of these eight is
/// IN scope. Seven are `SessionContext` methods; `schema` is a
/// `CatalogProvider` method reached through `ctx.catalog(..)` rather than
/// called on `SessionContext` directly — this array polices literal call
/// SHAPES, not one specific receiver type.
pub(crate) const PAIRED_REGISTRATION_VERBS: &[&str] = &[
    "table",
    "object_store",
    "udtf",
    "udf",
    "udaf",
    "udwf",
    "higher_order_function",
    "schema",
];

/// DataFusion `SessionContext` verbs (54.1.0) with only a `register_` form
/// (no matching `deregister_`), each binding the shared session under a
/// caller-chosen token — see the module-level comment above for why each of
/// these eight is IN scope.
pub(crate) const UNPAIRED_REGISTRATION_VERBS: &[&str] = &[
    "batch",
    "csv",
    "json",
    "parquet",
    "avro",
    "listing_table",
    "arrow",
    "catalog",
];

/// The four DataFusion `SessionContext` verbs (54.1.0) this gate deliberately
/// does NOT flag — see the module-level comment above for why each has no
/// caller-chosen per-call token to collide on.
const EXCLUDED_REGISTRATION_VERB_LITERALS: &[&str] = &[
    "register_variable(",
    "register_relation_planner(",
    "register_catalog_list(",
    "register_table_options_extension(",
];

/// Every literal call-site pattern [`fine_tune_session_registration_hits`]
/// treats as in-scope, derived from [`PAIRED_REGISTRATION_VERBS`] and
/// [`UNPAIRED_REGISTRATION_VERBS`] — the SAME two arrays the detector reads,
/// so the falsification test's expected set can never drift from what the
/// detector actually checks (the same caller-set-claim drift `callers_of`'s
/// own doc names, applied here to a literal set instead of a caller set).
fn included_registration_literals() -> Vec<String> {
    let mut out = Vec::new();
    for verb in PAIRED_REGISTRATION_VERBS {
        out.push(format!("register_{verb}("));
        out.push(format!("deregister_{verb}("));
    }
    for verb in UNPAIRED_REGISTRATION_VERBS {
        out.push(format!("register_{verb}("));
    }
    out
}

/// Every hit of the class above under `crates/jammi-ai/src/fine_tune/**`, on
/// the masked surface (so a doc comment or a string literal that merely
/// NAMES a method, e.g. this file's own module comment, is never mistaken
/// for a call). For each [`PAIRED_REGISTRATION_VERBS`] entry, the
/// `deregister_` form is checked BEFORE the `register_` form on the same
/// line: `"register_X("` is itself a substring of `"deregister_X("` (`de` +
/// `register_X(`), so checking both unconditionally would double-count every
/// `deregister_X(` call as a `register_X(` hit too — the SAME reasoning the
/// original two-verb version used for `table`, now applied per paired verb.
/// Checked exhaustively rather than assumed: no other pair among the full
/// 24-literal surface is a substring of another (see
/// [`falsification_no_included_literal_is_a_substring_of_another_unless_the_declared_pair`],
/// below), so no other verb needs this ordering.
fn fine_tune_session_registration_hits(
    surface: &[(String, String)],
) -> Vec<(String, usize, String)> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        if !file.starts_with("crates/jammi-ai/src/fine_tune/") {
            continue;
        }
        let masked = mask_non_code(text);
        for (line_idx, line) in masked.lines().enumerate() {
            let line_no = line_idx + 1;
            for verb in PAIRED_REGISTRATION_VERBS {
                let de_pat = format!("deregister_{verb}(");
                let re_pat = format!("register_{verb}(");
                if line.contains(de_pat.as_str()) {
                    hits.push((file.clone(), line_no, de_pat));
                } else if line.contains(re_pat.as_str()) {
                    hits.push((file.clone(), line_no, re_pat));
                }
            }
            for verb in UNPAIRED_REGISTRATION_VERBS {
                let re_pat = format!("register_{verb}(");
                if line.contains(re_pat.as_str()) {
                    hits.push((file.clone(), line_no, re_pat));
                }
            }
        }
    }
    hits
}

/// SQL DDL literal keywords that bind a NEW relation into the session's
/// catalog when executed through `SessionContext::sql` — the string-literal
/// analogue of a `register_`/`deregister_` verb call. Invisible to
/// [`fine_tune_session_registration_hits`] because the keyword lives INSIDE a
/// string literal, which [`mask_non_code`] blanks by design (the same reason
/// [`session_registration_literal_sites`] scans the ORIGINAL, unmasked text
/// for `"jammi.{` rather than the masked surface, above) — this scan instead
/// uses [`mask_comments_only`], which blanks comments but leaves string
/// content visible. `CREATE EXTERNAL TABLE` is listed separately from
/// `CREATE TABLE`: neither is a substring of the other (`EXTERNAL ` sits
/// between them), so there is no `register_X`/`deregister_X`-shaped ordering
/// hazard here. The canonical (upper-case) spelling below is what every hit
/// is reported as, regardless of the source line's own casing or an `OR
/// REPLACE` qualifier — see [`ddl_statement_shape`] for the actual match
/// (case-insensitive, `CREATE OR REPLACE` aware).
const DDL_RELATION_BINDING_LITERALS: &[&str] = &[
    "CREATE VIEW",
    "CREATE EXTERNAL TABLE",
    "CREATE TABLE",
    "CREATE SCHEMA",
];

/// Whether `text` contains a DDL statement that binds a relation into a
/// DataFusion session's catalog when handed to `SessionContext::sql` —
/// `CREATE [OR REPLACE] {VIEW | TABLE | EXTERNAL TABLE | SCHEMA}`, matched
/// case-insensitively over whitespace-separated tokens (a prior, case-
/// sensitive, upper-case-only substring match against the four
/// [`DDL_RELATION_BINDING_LITERALS`] missed both a lower-case `create table`
/// and DataFusion's `CREATE OR REPLACE` form — implemented as
/// deregister-then-register on the SAME relation name,
/// `datafusion-54.1.0/src/execution/context/mod.rs`'s `SessionContext::sql`
/// DDL dispatch — which binds under exactly the same caller-chosen token the
/// bare form does and so is not a distinct, safer shape). Used both by
/// [`fine_tune_ddl_relation_binding_hits`] below (the `fine_tune/`-scoped,
/// direct-call-site layer) and by [`ddl_literal_occurrences`] (the
/// whole-two-crate-surface layer) — one shape definition, not two
/// independently-drifting copies.
pub(crate) fn ddl_statement_shape(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    // Split on anything that is neither alphanumeric nor `_`, not merely on
    // whitespace: real SQL text like `ctx.sql("create view ...")` glues the
    // opening quote directly against the keyword with no space
    // (`("create`), so a plain `.split_whitespace()` would never isolate
    // `create` as its own token. `_` stays PART of a token (not a separator)
    // so a Rust identifier such as `create_view_something` is one token,
    // never mistaken for the two words `create`/`view` — only genuine
    // whitespace-or-punctuation-separated SQL keyword text matches.
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .collect();
    for (i, tok) in tokens.iter().enumerate() {
        if *tok != "create" {
            continue;
        }
        let mut j = i + 1;
        if tokens.get(j) == Some(&"or") && tokens.get(j + 1) == Some(&"replace") {
            j += 2;
        }
        match tokens.get(j) {
            Some(&"view") | Some(&"table") | Some(&"schema") => return true,
            Some(&"external") if tokens.get(j + 1) == Some(&"table") => return true,
            _ => {}
        }
    }
    false
}

/// Every [`ddl_statement_shape`] hit under `crates/jammi-ai/src/fine_tune/**`,
/// on [`mask_comments_only`]'s output (comments blanked, string content
/// intact) rather than the fully-masked surface [`mask_non_code`] produces
/// (which would blank the very string literals this scan needs to read).
/// Reported against the canonical (upper-case) [`DDL_RELATION_BINDING_LITERALS`]
/// spelling regardless of the source line's own casing or `OR REPLACE`
/// qualifier, so callers checking for `"CREATE VIEW"` keep working whether the
/// source wrote `create view`, `CREATE VIEW`, or `CREATE OR REPLACE VIEW`.
fn fine_tune_ddl_relation_binding_hits(
    surface: &[(String, String)],
) -> Vec<(String, usize, String)> {
    let mut hits = Vec::new();
    for (file, text) in surface {
        if !file.starts_with("crates/jammi-ai/src/fine_tune/") {
            continue;
        }
        let masked = mask_comments_only(text);
        for (line_idx, line) in masked.lines().enumerate() {
            if !ddl_statement_shape(line) {
                continue;
            }
            let line_no = line_idx + 1;
            let lower = line.to_ascii_lowercase();
            let literal = DDL_RELATION_BINDING_LITERALS
                .iter()
                .find(|lit| {
                    let bare = lit.to_ascii_lowercase();
                    let replaced = bare.replacen("create ", "create or replace ", 1);
                    lower.contains(bare.as_str()) || lower.contains(replaced.as_str())
                })
                .copied()
                .unwrap_or("CREATE (unrecognised canonical form)");
            hits.push((file.clone(), line_no, literal.to_string()));
        }
    }
    hits
}

/// RED at `fe96bf39` (the excised commit's parent): `training_set.rs` had a
/// `ctx.register_table(relation.as_str(), ...)` call and `DeregisterOnDrop`'s
/// `self.ctx.deregister_table(...)` — two hits under the original two-verb
/// detector. GREEN once the guard and its `MemTable` machinery are removed
/// and the graph arm reverts to sampling in memory: zero hits,
/// unconditionally, no allowlist — now checked against the full 24-literal
/// verb surface above (`fine_tune_session_registration_hits`) PLUS the DDL
/// surface (`fine_tune_ddl_relation_binding_hits`).
///
/// **This test's property is DIRECT call sites under `fine_tune/**` only.**
/// It says nothing about whether a name gets bound INDIRECTLY, through an
/// in-tree function defined outside `fine_tune/` — that half is
/// [`registration_verb_occurrences_are_all_reviewed`] /
/// [`ddl_literal_occurrences_are_all_reviewed`], further down this file,
/// which review every occurrence anywhere under both crates' `src` trees
/// (never tracing whether `fine_tune/` can actually reach any given one —
/// that reachability question is <https://github.com/f-inverse/jammi-ai/issues/549>).
/// The module-level comment above states why the split is real rather than
/// an oversight: a directory-scoped literal scan structurally cannot see a
/// call that never appears inside `fine_tune/**`, no matter how many verbs
/// it knows.
#[test]
fn no_session_registration_under_fine_tune() {
    let surface = scan_surface();
    let verb_hits = fine_tune_session_registration_hits(&surface);
    let ddl_hits = fine_tune_ddl_relation_binding_hits(&surface);
    assert!(
        verb_hits.is_empty() && ddl_hits.is_empty(),
        "no file under crates/jammi-ai/src/fine_tune (recursively) may DIRECTLY write a call to \
         a session-binding verb (a `SessionContext`/`CatalogProvider` register_/deregister_ verb) \
         or a DDL literal that binds a relation (CREATE VIEW/TABLE/EXTERNAL TABLE/SCHEMA) — the \
         shared session is not a per-call namespace, and any name that is not unique per CALL \
         collides under reclaim (see https://github.com/f-inverse/jammi-ai/issues/538). This is a \
         DIRECT-call-site property only; the reviewed bindings anywhere under both crates' `src` \
         trees (including those reached through `ResultStore::materialize_training_set`) are a \
         separate, pinned property — see pinned_source_gate::REGISTRATION_VERB_SITES. verb hits: \
         {verb_hits:?}; DDL hits: {ddl_hits:?}"
    );
}

/// Falsification (R-A): every INCLUDED verb literal must fire the detector
/// on its own line, scoped to `fine_tune/`, and masked out of a comment; and
/// every EXCLUDED verb literal (a real `SessionContext` registration method
/// this gate deliberately does not police) must NOT fire even when called
/// from inside `fine_tune/` — proving the OUT decision by running the exact
/// call shape through the detector, not merely by omitting it from the
/// include list.
#[test]
fn falsification_fine_tune_session_registration_is_detected_and_scoped() {
    let included = included_registration_literals();
    assert_eq!(
        included.len(),
        24,
        "expected 8 paired verbs * 2 forms + 8 unpaired verbs = 24 literals; the module-level \
         doc's IN enumeration and PAIRED_REGISTRATION_VERBS/UNPAIRED_REGISTRATION_VERBS have \
         drifted apart if this count changes without both being updated together"
    );

    // One call per included literal, each on its own line inside its own
    // synthetic function, scoped as if it lived under `fine_tune/`.
    let mut hit_src = String::from("\n");
    for lit in &included {
        hit_src.push_str(&format!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for `fine_tune_session_registration_hits` — synthetic producer text fed to that detector, not real code in this file
            "async fn synthetic(ctx: &SessionContext) {{\n    ctx.{lit}ARG).unwrap();\n}}\n"
        ));
    }
    let surface = [(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        hit_src.clone(),
    )];
    let hits = fine_tune_session_registration_hits(&surface);
    assert_eq!(
        hits.len(),
        included.len(),
        "expected exactly one hit per included verb literal, got {hits:?}"
    );
    let found: BTreeSet<String> = hits.iter().map(|(_, _, lit)| lit.clone()).collect();
    let expected: BTreeSet<String> = included.iter().cloned().collect();
    assert_eq!(
        found, expected,
        "every included verb literal must fire the detector on its own falsification line"
    );

    // Same text, a file OUTSIDE fine_tune/ — must not count (this gate's
    // property is scoped to the excised arm's own module tree, not the
    // whole crate).
    let outside_hits = fine_tune_session_registration_hits(&[(
        "crates/jammi-ai/src/pipeline/embedding.rs".to_string(),
        hit_src.clone(),
    )]);
    assert!(
        outside_hits.is_empty(),
        "a hit outside the fine_tune tree must not be counted, got {outside_hits:?}"
    );

    // A comment naming every included method, never calling it — must not
    // count (masked out, same discipline `session_registration_literal_sites`
    // uses).
    let mut comment_src = String::new();
    for lit in &included {
        comment_src.push_str(&format!("// see ctx.{lit} for context\n"));
    }
    let comment_hits = fine_tune_session_registration_hits(&[(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        comment_src,
    )]);
    assert!(
        comment_hits.is_empty(),
        "a comment naming the methods must not count as a call site, got {comment_hits:?}"
    );

    // Negative control: the OUT verbs — real DataFusion `SessionContext`
    // registration methods this gate deliberately does NOT flag, because
    // none binds a caller-chosen per-call token (see the module-level
    // comment above for why each is excluded). Run inside `fine_tune/`
    // itself so the control proves the OUT decision, not merely the
    // directory scoping already proven above.
    let mut excluded_src = String::from("\n");
    for lit in EXCLUDED_REGISTRATION_VERB_LITERALS {
        excluded_src.push_str(&format!(
            // kernel-oracles: fn-in-literal reviewed: negative-control fixture for the excluded `SessionContext` verbs — synthetic producer text fed to `fine_tune_session_registration_hits`, not real code in this file
            "async fn synthetic_excluded(ctx: &SessionContext) {{\n    ctx.{lit}ARG);\n}}\n"
        ));
    }
    let excluded_hits = fine_tune_session_registration_hits(&[(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        excluded_src,
    )]);
    assert!(
        excluded_hits.is_empty(),
        "a SessionContext verb with no caller-chosen per-call token must never be flagged, got \
         {excluded_hits:?}"
    );
}

/// Falsification (R-H, applied to a literal set rather than a
/// caller set — see [`included_registration_literals`]'s doc): the ONE
/// ordering hazard `fine_tune_session_registration_hits` corrects for
/// (`"register_X("` is a substring of `"deregister_X("`) is the ONLY such
/// hazard among the full 24-literal surface, checked exhaustively rather
/// than assumed — a future addition to either verb array that silently
/// introduces a SECOND hazard (e.g. two unpaired verbs where one's literal
/// is a substring of the other's) would go undetected by the per-verb
/// `if`/`else if` structure above, exactly the way `register_batch` went
/// undetected by the original two-literal version.
#[test]
fn falsification_no_included_literal_is_a_substring_of_another_unless_the_declared_pair() {
    let included = included_registration_literals();
    for a in &included {
        for b in &included {
            if a == b {
                continue;
            }
            if b.contains(a.as_str()) {
                let is_declared_pair = PAIRED_REGISTRATION_VERBS.iter().any(|verb| {
                    *a == format!("register_{verb}(") && *b == format!("deregister_{verb}(")
                });
                assert!(
                    is_declared_pair,
                    "{a:?} is a substring of {b:?}, an UNDECLARED collision — \
                     `fine_tune_session_registration_hits`'s if/else-if ordering only accounts \
                     for the declared register_X/deregister_X pairs, so this pair would silently \
                     under-count one of the two calls on a shared line"
                );
            }
        }
    }
}

/// A probe shape this scan must catch: `ctx.catalog(..).unwrap().register_schema(..)`,
/// reproduced under `fine_tune/` and run through the real detector — not
/// merely the generic per-verb falsification loop above, which never spells
/// `ctx.catalog(..).unwrap()` as the receiver: `CatalogProvider::register_schema`
/// is reached exactly this way, never as a bare `SessionContext` method, so
/// this is the shape that mattered.
#[test]
fn falsification_probe_a_catalog_register_schema_is_caught() {
    let src = concat!(
        // kernel-oracles: fn-in-literal reviewed: falsification fixture reproducing the `ctx.catalog(..).unwrap().register_schema(..)` shape — synthetic producer text fed to `fine_tune_session_registration_hits`, not real code in this file
        "async fn probe(ctx: &SessionContext, name: &str) {\n",
        "    ctx.catalog(\"datafusion\").unwrap().register_schema(name, \
         std::sync::Arc::new(MemorySchemaProvider::new())).unwrap();\n",
        "}\n",
    );
    let hits = fine_tune_session_registration_hits(&[(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        src.to_string(),
    )]);
    assert!(
        hits.iter().any(|(_, _, lit)| lit == "register_schema("),
        "the `ctx.catalog(..).unwrap().register_schema(..)` shape must be \
         caught now that `schema` is in PAIRED_REGISTRATION_VERBS, got {hits:?}"
    );
}

/// A probe shape this scan must catch: a `CREATE VIEW` DDL string handed to
/// `SessionContext::sql`, reproduced under `fine_tune/` and run through the
/// DDL detector — the shape no verb-literal scan could ever catch, because
/// the binding text lives inside a string literal.
#[test]
fn falsification_probe_b_create_view_ddl_is_caught() {
    let src = concat!(
        // kernel-oracles: fn-in-literal reviewed: falsification fixture reproducing the `CREATE VIEW` DDL-string shape — synthetic producer text fed to `fine_tune_ddl_relation_binding_hits`, not real code in this file
        "async fn probe(ctx: &SessionContext, name: &str) {\n",
        "    ctx.sql(&format!(\"CREATE VIEW {name} AS SELECT 1\")).await.unwrap();\n",
        "}\n",
    );
    let hits = fine_tune_ddl_relation_binding_hits(&[(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        src.to_string(),
    )]);
    assert!(
        hits.iter().any(|(_, _, lit)| lit == "CREATE VIEW"),
        "a `CREATE VIEW` DDL string handed to `SessionContext::sql` must be caught by the \
         comment-aware DDL scan, got {hits:?}"
    );
}

/// Every [`DDL_RELATION_BINDING_LITERALS`] shape is caught on its own
/// falsification line, scoped to `fine_tune/`; a comment naming all four
/// (never issuing one) and a hit outside `fine_tune/` must both be silent —
/// the same three controls [`falsification_fine_tune_session_registration_is_detected_and_scoped`]
/// runs for the verb-literal surface, run here for the DDL surface.
#[test]
fn falsification_every_ddl_literal_is_detected_and_scoped() {
    for lit in DDL_RELATION_BINDING_LITERALS {
        let src = format!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for `fine_tune_ddl_relation_binding_hits` — synthetic producer text, not real code in this file
            "async fn probe(ctx: &SessionContext) {{\n    ctx.sql(\"{lit} t AS SELECT 1\").await.unwrap();\n}}\n"
        );
        let hits = fine_tune_ddl_relation_binding_hits(&[(
            "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
            src,
        )]);
        assert!(
            hits.iter().any(|(_, _, l)| l == lit),
            "{lit} must be caught on its own falsification line, got {hits:?}"
        );
    }

    let mut comment_src = String::new();
    for lit in DDL_RELATION_BINDING_LITERALS {
        comment_src.push_str(&format!("// see {lit} for context\n"));
    }
    let comment_hits = fine_tune_ddl_relation_binding_hits(&[(
        "crates/jammi-ai/src/fine_tune/training_set.rs".to_string(),
        comment_src,
    )]);
    assert!(
        comment_hits.is_empty(),
        "a comment naming a DDL literal must not count as a call site, got {comment_hits:?}"
    );

    let outside_hits = fine_tune_ddl_relation_binding_hits(&[(
        "crates/jammi-ai/src/pipeline/embedding.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for the directory-scoping control on `fine_tune_ddl_relation_binding_hits` — synthetic producer text, not real code in this file
            "async fn probe(ctx: &SessionContext) {\n",
            "    ctx.sql(\"CREATE VIEW t AS SELECT 1\").await.unwrap();\n",
            "}\n",
        )
        .to_string(),
    )]);
    assert!(
        outside_hits.is_empty(),
        "a DDL hit outside fine_tune/ must not be counted, got {outside_hits:?}"
    );
}

// ── The literal-occurrence gate replaces call-graph reachability. ─────────
//
// The two checks above (`no_session_registration_under_fine_tune` and its
// falsifications) are the DIRECT-call-site layer: no file under
// `crates/jammi-ai/src/fine_tune/**` may itself spell a session-binding verb
// or a DDL literal. They stay here, cheap and text-based, as the first line
// of defense.
//
// The INDIRECT half -- an in-tree function defined OUTSIDE `fine_tune/` that
// itself binds a session/catalog name, reached through some chain of in-tree
// calls `fine_tune/` makes -- is no longer answered by tracing a call graph
// at all. A `syn`-based AST call-graph test file (deleted; it lived
// alongside this one under `crates/jammi-ai/tests/it/`) missed a
// registration verb reached through a function-pointer argument,
// `.map(Self::f)`, a call inside a macro invocation such as
// `assert!`/`tokio::select!`, or a
// fn-pointer struct field, and missed a DDL keyword sitting in a module-level
// `const SQL = "..."`, split across two `format!`/`concat!` arguments, or
// pulled in via `include_str!` -- a SOUNDNESS gap in what any finite set of
// AST node-kind handlers can promise to cover exhaustively. That gate is
// deleted, along with its `syn`/`proc-macro2` dev-dependencies (no
// production code ever depended on either). The dependency-closure/
// reachability question it existed to answer is its own unit,
// <https://github.com/f-inverse/jammi-ai/issues/549>.
//
// **What replaces it, stated as its own honest universe.** Rather than
// trace which binder a `fine_tune/` call can REACH, this gate finds every
// SINGLE-LINE occurrence of one of the 24 `register_*`/`deregister_*`
// call-site patterns ([`PAIRED_REGISTRATION_VERBS`]/
// [`UNPAIRED_REGISTRATION_VERBS`]) or the DDL-statement shape
// ([`ddl_statement_shape`]) in `crates/jammi-ai/src` and
// `crates/jammi-db/src` (both whole trees, [`SURFACE_DIRS`]) -- not only
// `crates/jammi-ai/src/fine_tune/**`, which
// [`fine_tune_session_registration_hits`]/[`fine_tune_ddl_relation_binding_hits`]
// above already police as the cheap, zero-tolerance DIRECT layer -- and keys
// each hit to `(file, enclosing function, ordinal)`, so a human reviews and
// pins one property per site regardless of how many matching lines that
// site contains.
//
// That key is also this gate's own residual, not a solved problem the
// deleted gate merely approximated. It counts SITES, never per-site
// OCCURRENCES: a second `register_table(..)` call planted inside an
// already-reviewed function, or a second module-scope DDL `const` added to
// a file that already has one, keys to the SAME `(file, function, ordinal)`
// an earlier, different line already occupies, and so raises no new hit for
// [`registration_verb_occurrences_are_all_reviewed`]/
// [`ddl_literal_occurrences_are_all_reviewed`] to catch. This scan is also
// strictly line-based: a DDL literal split across two `concat!`/`format!`
// arguments on separate lines, or pulled in through `include_str!`, is
// invisible to it -- the same two shapes the deleted AST gate also missed,
// carried over rather than closed by this replacement. And it is scoped to
// exactly [`SURFACE_DIRS`]: a registration verb or DDL literal living
// anywhere outside those two `src` trees is outside its universe entirely --
// under `tests/it/` in either crate (e.g. the five `.register_table(`
// calls this file's own review list keys to, `.register_table(`,
// `crates/jammi-db/tests/it/materialization.rs:567/:616/:669/:671/:1024`,
// none of them under `crates/jammi-db/src`), or in a third crate, both
// count the same way. A fifth gap sits inside the scan itself, not at its
// boundary: [`mask_comments_only`]'s masking step desyncs on a raw string
// today (its own doc above states the raw-string/char-literal residual
// and the file/line evidence), so a DDL literal sitting on a desynced
// line is invisible to [`ddl_literal_occurrences`] regardless of which
// directory it lives in.
// None of these five gaps (per-site counts, split literals, `include_str!`
// targets, anything outside `SURFACE_DIRS` including `tests/it/`, the
// masking step's raw-string desync) is closed here; the rebuild that would
// close them is <https://github.com/f-inverse/jammi-ai/issues/554>.
//
// What this gate DOES buy over the deleted call-graph gate is recall over
// call SHAPE, not occurrence count or literal assembly: every function, in
// either crate's `src` tree ([`SURFACE_DIRS`] -- never the whole
// repository; `tests/it/` and any third crate are the gap named above),
// whose own body contains at least one line
// matching one of the 24 patterns or the DDL shape is found and reviewed
// here, whether or not anything under `fine_tune/` can reach it -- a site
// with zero real callers (e.g. a trait method a language feature requires
// but nothing in-tree invokes) is still listed and reviewed here, the same
// as a site with a hundred callers -- see each entry's own prose for which
// case it is.

/// One occurrence a review has cleared: the registration-verb call/
/// declaration or DDL-shaped string literal at `(file, function, ordinal)` --
/// never a declaration LINE, which drifts under an unrelated edit above it
/// (see [`assign_ordinals`]'s doc) -- carries a `property`, the reviewed,
/// human-written account of what this site actually does and why a second
/// call/occurrence at the same site can never silently corrupt state. Every
/// field is read: `file`/`function`/`ordinal` key the comparison against the
/// live scan ([`registration_verb_occurrences_are_all_reviewed`],
/// [`ddl_literal_occurrences_are_all_reviewed`]); `property` is asserted
/// non-trivial by [`every_reviewed_registration_site_states_its_property`]
/// and printed in every failure message via `#[derive(Debug)]` -- no field
/// here is decorative the way the deleted call-graph gate's `#[allow(dead_code)]`
/// `arm`/`reason` fields were.
#[derive(Debug)]
struct ReviewedRegistrationSite {
    file: &'static str,
    function: &'static str,
    ordinal: usize,
    property: &'static str,
}

impl ReviewedRegistrationSite {
    fn key(&self) -> (String, String, usize) {
        (
            self.file.to_string(),
            self.function.to_string(),
            self.ordinal,
        )
    }
}

/// The same "smallest enclosing region wins" attribution
/// [`session_registration_literal_sites`] and [`callers_of`] already use, cut
/// out as a shared helper for the two whole-surface scans below (both scan
/// UNFILTERED by directory, unlike those two, which is the entire point: the
/// occurrence, not the reachability, is what is being enumerated here).
fn attribute_hit_to_enclosing_fn(regions: &[FnRegion], line_no: usize) -> (String, usize) {
    let mut best: Option<&FnRegion> = None;
    for region in regions {
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
    best.map(|r| (r.name.clone(), r.ordinal))
        .unwrap_or_else(|| ("<module-scope>".to_string(), 0))
}

/// Every occurrence, anywhere under [`SURFACE_DIRS`] (both crates, every
/// directory -- not scoped to `fine_tune/`), of a
/// [`PAIRED_REGISTRATION_VERBS`]/[`UNPAIRED_REGISTRATION_VERBS`] call-site
/// PATTERN, attributed to its enclosing function. Deliberately a superset of
/// "genuine calls": the same substring match `fine_tune_session_registration_hits`
/// uses also matches the verb's own `fn register_x(`/`fn deregister_x(`
/// DECLARATION line, which this file's hand-rolled scan cannot distinguish
/// from a call site without becoming a real parser -- disclosed, not hidden:
/// [`REGISTRATION_VERB_SITES`]'s entries for `store/mod.rs::register_table`
/// and `store/result_schema.rs::{register_table,deregister_table}` say so
/// directly.
fn registration_verb_occurrences(
    surface: &[(String, String)],
) -> BTreeSet<(String, String, usize)> {
    let mut hits = BTreeSet::new();
    for (file, text) in surface {
        let masked = mask_non_code(text);
        let regions = find_fn_regions(&masked);
        for (line_idx, line) in masked.lines().enumerate() {
            let line_no = line_idx + 1;
            let mut hit_here = false;
            for verb in PAIRED_REGISTRATION_VERBS {
                if line.contains(format!("deregister_{verb}(").as_str())
                    || line.contains(format!("register_{verb}(").as_str())
                {
                    hit_here = true;
                }
            }
            for verb in UNPAIRED_REGISTRATION_VERBS {
                if line.contains(format!("register_{verb}(").as_str()) {
                    hit_here = true;
                }
            }
            if hit_here {
                let (name, ordinal) = attribute_hit_to_enclosing_fn(&regions, line_no);
                hits.insert((file.clone(), name, ordinal));
            }
        }
    }
    hits
}

/// Every [`ddl_statement_shape`] occurrence anywhere under [`SURFACE_DIRS`]
/// (both crates, every directory), on [`mask_comments_only`]'s output (string
/// content visible, comments blanked -- the same reasoning
/// [`fine_tune_ddl_relation_binding_hits`] already documents), attributed to
/// its enclosing function.
fn ddl_literal_occurrences(surface: &[(String, String)]) -> BTreeSet<(String, String, usize)> {
    let mut hits = BTreeSet::new();
    for (file, text) in surface {
        let regions = find_fn_regions(&mask_non_code(text));
        let masked_comments = mask_comments_only(text);
        for (line_idx, line) in masked_comments.lines().enumerate() {
            if !ddl_statement_shape(line) {
                continue;
            }
            let line_no = line_idx + 1;
            let (name, ordinal) = attribute_hit_to_enclosing_fn(&regions, line_no);
            hits.insert((file.clone(), name, ordinal));
        }
    }
    hits
}

/// This list IS the gate's own output at this head, never a hand-typed
/// guess: every occurrence of a [`PAIRED_REGISTRATION_VERBS`]/
/// [`UNPAIRED_REGISTRATION_VERBS`] call-site pattern under [`SURFACE_DIRS`]
/// TODAY, transcribed by running [`registration_verb_occurrences`] against
/// `scan_surface()` with an empty allowlist and reading each hit's real call
/// site -- 18 entries: 17 on the production/test call and declaration sites
/// `registration_verb_occurrences` finds unassisted across both
/// [`SURFACE_DIRS`] trees, plus the 18th being `build_result_table_provider`'s
/// own new EXECUTED oracle,
/// `crates/jammi-db/src/store/mod.rs::tests::register_object_store_twice_for_one_url_rebinds_the_same_driver_and_errors_on_neither`
/// (a `#[cfg(test)]` module inside that `src` file, so still inside
/// [`SURFACE_DIRS`], not this file) -- that test itself calls
/// `register_object_store` twice, which the same scan also finds because it
/// is DIRECTORY-restricted to [`SURFACE_DIRS`], not test-vs-production
/// restricted: every occurrence of the pattern anywhere under those two
/// trees is a site, whether the enclosing function is production code or a
/// test. Kept in
/// sync by [`registration_verb_occurrences_are_all_reviewed`]: an entry here
/// whose site no longer produces a hit, or a hit with no entry here, both
/// fail that test — this comment's own entry count is the only part of this
/// relationship that is NOT machine-checked, so treat it as documentation
/// for a human re-deriving the list, not as a fact the gate itself asserts.
const REGISTRATION_VERB_SITES: &[ReviewedRegistrationSite] = &[
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/content_hash_udf.rs",
        function: "register_content_hash_udf",
        ordinal: 1,
        property: "register_udf(..) under the UDF's own FIXED `.name()` (`jammi_content_hash`), \
                   called once per session at construction (`InferenceSession`'s own \
                   `with_observer`/`wrap_with` chain) -- a session-wide singleton, never a \
                   per-job/per-call resource; a second call on the same session silently \
                   overwrites the first (`register_udf`'s own doc), which is fine here because \
                   every call registers the identical function.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/content_hash_udf.rs",
        function: "udf_hashes_the_runner_rendering",
        ordinal: 1,
        property: "a unit test's own `SessionContext::new()`, local and discarded at the end of \
                   the test -- never the shared production session, so there is no reclaim-shaped \
                   collision surface here at all.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        function: "empty_group_is_null_vector",
        ordinal: 1,
        property: "a unit test's own `SessionContext::new()`, local and discarded at the end of \
                   the test -- same as `content_hash_udf.rs::udf_hashes_the_runner_rendering`.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        function: "grouped_reduction_per_group",
        ordinal: 1,
        property: "a unit test's own `SessionContext::new()`, local and discarded at the end of \
                   the test.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        function: "register_vector_agg_udafs",
        ordinal: 1,
        property: "register_udaf(..) three times (`vector_mean`/`vector_sum`/`vector_max`), each \
                   under that UDAF's own FIXED `.name()`, called once per session at construction \
                   (`InferenceSession::register_query_functions`'s own call) -- the same \
                   session-wide singleton shape as `register_content_hash_udf`.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        function: "run_reduce",
        ordinal: 1,
        property: "a unit-test helper's own `SessionContext::new()`, local and discarded at the \
                   end of each call.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/query/vector_agg_udaf.rs",
        function: "wrong_argument_type_is_planning_error",
        ordinal: 1,
        property: "a unit test's own `SessionContext::new()`, local and discarded at the end of \
                   the test.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-ai/src/session.rs",
        function: "register_query_functions",
        ordinal: 1,
        property: "register_udtf(..) under the FIXED `AnnotateTableFunction::NAME` -- this \
                   function's own doc: \"must be called once per session, after the session is \
                   behind an Arc\" -- a session-construction-time singleton, never called from \
                   `fine_tune/` or per job.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/session.rs",
        function: "build",
        ordinal: 1,
        property: "register_catalog(\"mutable\", ..) under the FIXED literal name \"mutable\", \
                   once at session-build time -- a session-construction-time singleton.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/session.rs",
        function: "register_source_tables",
        ordinal: 1,
        property: "register_catalog(source_id, ..) keyed by the data SOURCE's own stable, \
                   admin-configured identifier, called once per configured source at session \
                   build/reload time -- never per fine_tune call, never per job; two DIFFERENT \
                   sources never share a `source_id`, and re-registering the SAME source's \
                   catalog at reload time is a deliberate refresh of that source's own tables.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/source/file_format.rs",
        function: "register_driver_for_url",
        ordinal: 1,
        property: "register_object_store(..) keyed by the URL's own scheme+authority, with the \
                   driver resolved through `StorageRegistry::driver_for`'s per-(scheme,root) \
                   cache -- the identical idempotent-rebind shape reviewed for \
                   `store/mod.rs::build_result_table_provider` below, executed by \
                   `store::tests::register_object_store_twice_for_one_url_rebinds_the_same_driver_and_errors_on_neither`.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "bind_result_table",
        ordinal: 1,
        property: "binds by calling `self.register_table(..)` -- jammi's OWN 4-argument method, a \
                   name collision this substring scan cannot itself tell apart from DataFusion's \
                   `SessionContext::register_table`, but the resolution IS the real one here: \
                   `bind_result_table` rebinds a `table_name` an EARLIER call already created, \
                   over that call's own already-written, immutable Parquet bytes -- every call for \
                   the same `table_name` rebinds the identical artifact. EXECUTED oracle: \
                   `crates/jammi-db/tests/it/materialization.rs::two_runs_over_one_pinned_definition_share_one_training_set`.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "build_result_table_provider",
        ordinal: 1,
        property: "for a non-file/-memory URL, calls `ctx.runtime_env().register_object_store(&parsed, driver)` \
                   keyed by the URL's own scheme+authority, where `driver` is \
                   `StorageRegistry::driver_for`'s CACHED value for that key -- two calls for one \
                   URL rebind the identical driver, and DataFusion's own `register_object_store` \
                   signature (`Option<Arc<dyn ObjectStore>>`, no `Result`) cannot error on either \
                   call. EXECUTED oracle (NEW): \
                   `crates/jammi-db/src/store/mod.rs::tests::register_object_store_twice_for_one_url_rebinds_the_same_driver_and_errors_on_neither` \
                   -- pins the exact primitive this function calls; the function's own cloud-scheme \
                   branch cannot be driven end-to-end in this crate's default test build, disclosed \
                   in that test's own doc.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "install_result_schema",
        ordinal: 1,
        property: "calls `catalog.register_schema(&catalog_opts.default_schema, ..)` under the \
                   session's FIXED default-schema name -- this function's own doc: \"Idempotent: \
                   re-installing the same provider preserves the tables it already holds\" -- \
                   every call binds the SAME `Arc<ResultTableSchemaProvider>` under the same \
                   constant key, never a per-call one. EXECUTED oracle (NEW): \
                   `crates/jammi-db/tests/it/materialization.rs::install_result_schema_twice_on_one_session_binds_the_same_schema_and_errors_on_neither`.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "register_table",
        ordinal: 1,
        // kernel-oracles: fn-in-literal reviewed: the property string below names the literal shape `fn register_table(` in prose, describing a real declaration elsewhere in this file — not a fn-keyword desync in this line
        property: "this hit is the `fn register_table(` DECLARATION line, not a call site (see \
                   `registration_verb_occurrences`'s own doc on this scan's inability to tell the \
                   two apart). The function itself never calls a `register_table`/`deregister_table` \
                   verb directly; it calls `build_result_table_provider` + `install_result_schema` \
                   (both reviewed above) then `self.result_schema.add_result_table(..)` -- a \
                   DISTINCT, non-trait method (its own name does not match this gate's verb \
                   pattern) keyed by the caller-supplied `jammi.{name}` string, which is always \
                   `record.table_name` -- unique per table by construction \
                   (`create_table_names_a_concurrent_burst_uniquely_over_one_definition`, \
                   `materialization.rs`), so two DIFFERENT names never collide here.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/result_schema.rs",
        function: "register_table",
        ordinal: 1,
        property: "this hit is the `SchemaProvider::register_table` trait-method DECLARATION for \
                   `ResultTableSchemaProvider`, not a call site written in this crate: the only \
                   in-tree paths that dispatch to it are DataFusion's own `SessionContext::register_table` \
                   top-level API and `CREATE TABLE` DDL execution, when the target schema resolves \
                   to this provider (i.e. after `install_result_schema` runs). Checked, not assumed, \
                   and the command run is stated exactly because an earlier draft of this entry got \
                   it wrong: `grep -rn '\\.register_table(' crates/jammi-db/src crates/jammi-ai/src` \
                   -- the two `src` trees [`SURFACE_DIRS`] scans -- returns exactly ONE hit, the \
                   4-argument `.register_table(ctx, &record.table_name, &url, owner)` call at \
                   `store/mod.rs:2724`; it does NOT find the five 2-argument \
                   `.register_table(name, provider)` calls, because all five live under \
                   `crates/jammi-db/tests/it/materialization.rs`, outside both `src` trees entirely. \
                   The command that actually produces the five is repo-wide: \
                   `grep -rn '\\.register_table(' --include='*.rs' crates/` returns \
                   `materialization.rs:565/:616/:669/:671/:1024` (plus that same `store/mod.rs:2724` \
                   line, and several prose mentions of the verb inside this very file that are text, \
                   not call sites). Of those five 2-argument calls, only the one at `:1024`, inside \
                   `install_result_schema_twice_on_one_session_binds_the_same_schema_and_errors_on_neither`, \
                   actually dispatches to THIS implementation: it is the only one of the five whose \
                   `ctx` already had `install_result_schema` called on it earlier in the same \
                   function, which is what makes the target schema resolve here (traced, not \
                   assumed: `install_result_schema`'s call at that test's line 1018 precedes its \
                   `:1024` `register_table` call; the other four calls' enclosing functions --\
                   `ts_session` (`:565`), `pinned_session` (`:616`), and `pinned_session_two` \
                   (`:669`, `:671`) -- never call `install_result_schema` on their `ctx` at all, \
                   so they resolve to DataFusion's \
                   own default `MemorySchemaProvider` instead). That one call is deliberate, to \
                   prove the \"preserves the tables it already holds\" property survives a second \
                   install -- no PRODUCTION call reaches this implementation today. Its own body inserts \
                   UNCONDITIONALLY and returns the displaced provider on a name collision (a \
                   SILENT overwrite, closing audit #3's own finding, `CONTRACT-U2a-fix1.md` round \
                   3) -- moot for `fine_tune/` (#549 is the tracked follow-on for a table of its \
                   own), since no in-tree caller picks a per-job/per-call name through this path.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/result_schema.rs",
        function: "deregister_table",
        ordinal: 1,
        property: "the `SchemaProvider::deregister_table` trait-method DECLARATION, the inverse of \
                   `register_table` immediately above -- same disclosure, both commands stated \
                   exactly: `grep -rn '\\.deregister_table(' crates/jammi-db/src crates/jammi-ai/src` \
                   (the two `src` trees) finds nothing, and the repo-wide \
                   `grep -rn '\\.deregister_table(' --include='*.rs' crates/` finds no call site -- \
                   every hit it returns is this file's own prose, quoting the verb in backticks -- no \
                   real call site exists anywhere in the tree, under `src`, under `tests/it/`, or \
                   elsewhere; it exists to satisfy the trait.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mod.rs",
        function: "register_object_store_twice_for_one_url_rebinds_the_same_driver_and_errors_on_neither",
        ordinal: 1,
        property: "the unit test that is `build_result_table_provider`'s own \
                   EXECUTED oracle above -- it calls `ctx.runtime_env().register_object_store(..)` \
                   directly, twice, against an in-memory driver and a raw `url::Url`, on a session \
                   this test owns and discards at its end; not the shared production session, no \
                   reclaim-shaped collision surface.",
    },
];

/// This list IS the gate's own output at this head, transcribed the same
/// way [`REGISTRATION_VERB_SITES`] was: every [`ddl_statement_shape`]
/// occurrence under [`SURFACE_DIRS`] TODAY -- 5 entries, kept in sync by
/// [`ddl_literal_occurrences_are_all_reviewed`] the same way.
const DDL_LITERAL_SITES: &[ReviewedRegistrationSite] = &[
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/catalog/migrations.rs",
        function: "<module-scope>",
        ordinal: 0,
        property: "the ordered `MIGRATIONS` table naming each migration's SQL constant -- the DDL \
                   text itself lives in `catalog/schema.rs` (reviewed below); this file only lists \
                   the constants. Every migration executes through `CatalogBackend`'s own SQL \
                   connection (SQLite/Postgres), never through a DataFusion `SessionContext::sql` \
                   call -- outside the shape this gate's property (a DataFusion catalog bind) \
                   describes, enumerated and reviewed anyway per this gate's completeness \
                   requirement. Unreachable from any DataFusion `SessionContext`, and so from \
                   `fine_tune/`, regardless.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/catalog/schema.rs",
        function: "<module-scope>",
        ordinal: 0,
        property: "the migration SQL constants themselves (`CREATE TABLE sources`, \
                   `result_tables`, etc.) -- same disclosure as `catalog/migrations.rs`: executed \
                   only through `CatalogBackend`'s own connection, never a DataFusion \
                   `SessionContext::sql` call, so this gate's property does not describe them; \
                   listed and reviewed for completeness.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mutable/postgres.rs",
        function: "create_table_ddl",
        ordinal: 1,
        property: "builds a `CREATE TABLE ..` STRING for the companion \"mutable table\" Postgres \
                   backend, executed through that backend's own direct SQL connection -- never a \
                   DataFusion `SessionContext::sql` call, and never reachable from `fine_tune/` by \
                   any real call regardless (it targets an entirely separate SQL backend from the \
                   DataFusion catalog this gate's property is about).",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mutable/sqlite.rs",
        function: "create_table_ddl",
        ordinal: 1,
        property: "the SQLite arm of the same \"mutable table\" backend -- same disclosure as the \
                   Postgres arm above.",
    },
    ReviewedRegistrationSite {
        file: "crates/jammi-db/src/store/mutable/sqlite.rs",
        function: "create_table_ddl_emits_implicit_tenant_id",
        ordinal: 1,
        property: "a unit test asserting on the built DDL STRING's own content \
                   (`ddl.starts_with(\"CREATE TABLE \\\"widgets\\\"\")`) -- the DDL text lives in a \
                   test assertion, never executed as SQL by this test at all.",
    },
];

/// Both directions of the comparison every allowlist in this file already
/// checks (`allowlists_match_current_hits_exactly`'s own discipline, applied
/// here to the two whole-surface scans): an occurrence with no reviewed entry
/// is UNREVIEWED (fails naming the site); a reviewed entry whose site no
/// longer produces a hit is STALE (also fails -- an allowance is never
/// permanent slack a later, different site can spend).
fn assert_occurrences_reviewed(
    found: &BTreeSet<(String, String, usize)>,
    allow: &BTreeSet<(String, String, usize)>,
    what: &str,
) {
    let unreviewed: Vec<_> = found.difference(allow).collect();
    assert!(
        unreviewed.is_empty(),
        "{what}: unreviewed occurrence(s) {unreviewed:?} -- add a reviewed entry naming the \
         (file, function, ordinal) and its property, or remove the offending call/literal."
    );
    let stale: Vec<_> = allow.difference(found).collect();
    assert!(
        stale.is_empty(),
        "{what}: reviewed entr(y/ies) {stale:?} no longer produce a hit -- shrink the allowlist \
         to match reality."
    );
}

#[test]
fn registration_verb_occurrences_are_all_reviewed() {
    let surface = scan_surface();
    let found = registration_verb_occurrences(&surface);
    let allow: BTreeSet<_> = REGISTRATION_VERB_SITES
        .iter()
        .map(ReviewedRegistrationSite::key)
        .collect();
    assert_occurrences_reviewed(&found, &allow, "registration verb");
}

#[test]
fn ddl_literal_occurrences_are_all_reviewed() {
    let surface = scan_surface();
    let found = ddl_literal_occurrences(&surface);
    let allow: BTreeSet<_> = DDL_LITERAL_SITES
        .iter()
        .map(ReviewedRegistrationSite::key)
        .collect();
    assert_occurrences_reviewed(&found, &allow, "DDL literal");
}

/// No field on [`ReviewedRegistrationSite`] is decorative: `property` is read
/// here directly, never merely present for a human to skim.
#[test]
fn every_reviewed_registration_site_states_its_property() {
    for entry in REGISTRATION_VERB_SITES
        .iter()
        .chain(DDL_LITERAL_SITES.iter())
    {
        assert!(
            entry.property.trim().len() > 20,
            "{}::{} (ordinal {}) must state its reviewed property in prose, got {:?}",
            entry.file,
            entry.function,
            entry.ordinal,
            entry.property
        );
    }
}

/// The name-keyed collision control this literal gate needs: a NEW verb
/// occurrence in a NEW file is found by the scan and is NOT already on the
/// reviewed list -- exactly the mutation "a planted `ctx.register_table(`
/// appears in a new file under store/" that a fixed allowlist must not
/// silently absorb.
#[test]
fn falsification_new_verb_occurrence_in_a_new_file_is_flagged() {
    let surface = vec![(
        "crates/jammi-db/src/store/__probe_new_registration_site__.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture for `registration_verb_occurrences` -- synthetic producer text, not real code in this file
            "fn planted_caller(ctx: &SessionContext, provider: Arc<dyn TableProvider>) {\n",
            "    ctx.register_table(\"planted\", provider).unwrap();\n",
            "}\n",
        )
        .to_string(),
    )];
    let found = registration_verb_occurrences(&surface);
    let allow: BTreeSet<_> = REGISTRATION_VERB_SITES
        .iter()
        .map(ReviewedRegistrationSite::key)
        .collect();
    let key = (
        "crates/jammi-db/src/store/__probe_new_registration_site__.rs".to_string(),
        "planted_caller".to_string(),
        1usize,
    );
    assert!(
        found.contains(&key),
        "a register_table( call in a new file must be found by the scan, got {found:?}"
    );
    assert!(
        !allow.contains(&key),
        "the planted site must not already be on the reviewed list -- this control is vacuous \
         otherwise"
    );
}

/// Non-vacuousness for [`registration_verb_occurrences_are_all_reviewed`]:
/// reproduce ONE real hit on a minimal synthetic surface (rather than the
/// real, 600+-file `scan_surface()`, so this control is fast and
/// self-contained) and show [`assert_occurrences_reviewed`]'s own comparison
/// reports it unreviewed when the allowlist omits it -- i.e. removing a
/// reviewed entry from the real list would turn the real test RED, executed
/// here against a stand-in rather than by literally mutating the const (the
/// same class of control this file's other `falsification_*` tests use).
#[test]
fn falsification_removing_a_reviewed_entry_leaves_its_site_unreviewed() {
    let surface = vec![(
        "crates/jammi-ai/src/query/content_hash_udf.rs".to_string(),
        concat!(
            // kernel-oracles: fn-in-literal reviewed: falsification fixture reproducing a real reviewed site -- synthetic producer text, not read from the real file
            "pub fn register_content_hash_udf(ctx: &SessionContext) {\n",
            "    ctx.register_udf(ScalarUDF::new_from_impl(ContentHashUdf::default()));\n",
            "}\n",
        )
        .to_string(),
    )];
    let found = registration_verb_occurrences(&surface);
    let empty_allow: BTreeSet<(String, String, usize)> = BTreeSet::new();
    let unreviewed: Vec<_> = found.difference(&empty_allow).collect();
    assert!(
        !unreviewed.is_empty(),
        "an allowlist missing a real hit's entry must report it unreviewed, not silently pass -- \
         found {found:?} against an empty allowlist"
    );
}
