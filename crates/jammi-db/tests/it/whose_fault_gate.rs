//! The whose-fault gate — DIST round 8 (`CONTRACT-DIST-fix5.md`'s M1,
//! closing the ruling: "wrong attribution is still constructible, and
//! worse, it is live and pinned").
//!
//! **The invariant.** [`jammi_db::error::JammiError::Schema`] is the
//! CALLER-fault class (`crates/jammi-db/src/error.rs`'s own doc: "a CALLER's
//! vector is the caller's fault — the schema class"; the wire mapping sends
//! it out as gRPC `InvalidArgument`). Six prior rounds enforced this
//! invariant at exactly ONE seam — [`jammi_numerics::query::ValidatedQuery::
//! require_width`], the downstream, artifact-only width check, which is
//! structurally incapable of expressing a caller fault because its error
//! variant carries no [`jammi_numerics::query::QuerySource`] at all — and
//! each round found the NEXT construction of the caller-fault class that
//! bypassed that one seam and billed an engine-owned artifact's own
//! corruption to the caller instead (`crates/jammi-db/src/index/exact.rs`'s
//! corrupt scan width, `crates/jammi-db/src/index/placed.rs`'s absent
//! catalog width — both fixed the round this gate was written, alongside
//! two more of the same shape this gate's own derivation found:
//! `crates/jammi-db/src/store/vectors.rs`'s shared column-extraction
//! helpers and `crates/jammi-db/src/store/deletes.rs`'s mask reader, both
//! billing THIS table's or THIS mask's own stored corruption to the
//! caller).
//!
//! **What changed here.** The site-by-site sweep is REPLACED, not
//! supplemented, by a mechanical enumeration of every construction of
//! `JammiError::Schema` across the whole surface the property quantifies
//! over — `crates/jammi-db/src` and `crates/jammi-numerics/src` — modelled
//! on the sibling DELTA unit's gate
//! (`crates/jammi-ai/tests/it/pinned_source_gate.rs`): the scanned surface
//! is derived from `git ls-files` (not a hand-rolled directory walk), a
//! file `git ls-files` reports tracked that this process cannot then read
//! is a hard failure naming the file, and every hit the detector finds
//! TODAY is enumerated in [`ALLOWED`] with its own disclosure note — an
//! allowlist entry whose site no longer produces that hit is ALSO RED
//! ([`allowlist_entries_still_produce_their_hit`]), so an allowance can
//! never become permanent slack a later, different site spends.
//!
//! **The detector.** A construction is the source-text token
//! `JammiError::Schema` immediately followed by `{` (this variant is a
//! struct variant; every occurrence in this tree is a brace-enclosed field
//! list), over a comment/string/char-literal-aware mask ([`mask_non_code`],
//! ported verbatim from the sibling gate) so a doc comment's mention of the
//! name, or a `panic!("expected JammiError::Schema, got {other:?}")`
//! message, is never mistaken for a real construction.
//!
//! **Production vs. test.** Every PATTERN match on `JammiError::Schema`
//! (`match err { JammiError::Schema { .. } => `, `matches!(&err,
//! JammiError::Schema { .. })`) found across this surface today is inside a
//! `#[cfg(test)]` item. The detector excludes exactly the byte range each
//! `#[cfg(test)]` marker's own item spans ([`test_ranges`]) — NOT "from the
//! first marker to EOF": `crates/jammi-db/src/config/mod.rs` gates an
//! external-file module (`#[cfg(test)] mod tests;`, whose body is a
//! SEPARATE tracked file this gate scans on its own) declared near the TOP
//! of the file, with ordinary production code immediately following it, and
//! only gates a second, unrelated item (a bare `#[cfg(test)] fn`) much
//! later — a "first marker onward" split would have wrongly swallowed that
//! production code as test-only. [`end_of_next_test_item`] resolves each
//! marker's item to one of three shapes (an external-file `mod ident;`, an
//! inline `mod ident { ... }`, or a `fn ... { ... }`) and fails closed,
//! naming the file and byte offset, on anything else.
//! [`every_test_marker_resolves_to_a_recognized_item_shape`] forces that
//! resolution to actually run over every marker on the whole surface.
//!
//! **What this file does NOT claim.** Source-text pattern matching over a
//! hand-written approximation of Rust's grammar, not a parser. It does not
//! enumerate constructions of `JammiError::IncompatibleFormat` (the engine
//! class) at all — that class can never misattribute to the caller by
//! definition, so it is out of this property's scope. It does not judge
//! `crates/jammi-numerics/src`'s surface as trivially clean: it is scanned
//! identically to `crates/jammi-db/src`, and
//! [`the_numerics_crate_never_constructs_the_caller_fault_class`] asserts,
//! rather than assumes, that it produces zero hits — `JammiError` is not
//! even a type `jammi-numerics` knows about, so this is expected to hold by
//! construction, but the assertion is what makes that a checked fact, not a
//! claim.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The whole surface this gate's property quantifies over (`CONTRACT-DIST-
/// fix5.md`, M1: "over `crates/jammi-db/src` and `crates/jammi-numerics/src`").
const SURFACE_DIRS: &[&str] = &["crates/jammi-db/src", "crates/jammi-numerics/src"];

/// Every reviewed production construction of `JammiError::Schema` this
/// gate's scan finds TODAY, each with its own disclosure note. Keyed on
/// `(file, enclosing function name, ordinal, occurrence)` — round 10, see
/// [`FnRegion`]'s and [`SchemaHit`]'s docs for why: a bare declaration LINE
/// desyncs from the reviewed function under an edit that only shifts lines
/// above it (measured stale against PR #521's CI: this whole list drifted
/// off `crates/jammi-db/src/error.rs` after an unrelated history rewrite
/// moved its two entries from lines 584/605 to 545/566 with neither
/// reviewed `fn from` arm itself changing). An entry whose site no longer
/// produces the hit is RED ([`allowlist_entries_still_produce_their_hit`]);
/// a hit not in this list is RED
/// ([`every_production_schema_construction_is_reviewed_as_a_caller_fault_entry`]).
const ALLOWED: &[(&str, &str, usize, usize, &str)] = &[
    (
        "crates/jammi-db/src/error.rs",
        "from",
        2, // ordinal 2 — the second `fn from` in this file (the first is
        // `impl From<datafusion::error::DataFusionError>`); this one is
        // `impl From<jammi_numerics::query::QueryValidationError>`, line
        // 534 today.
        1, // occurrence 1 (line 545 today) — see the note below.
        "The settled mechanism itself: `QueryValidationError::NonFinite`'s \
         `QuerySource::Caller` arm. This IS the entry — `validate_query` \
         checked the query's own finiteness against nothing but the \
         query's own components, so a non-finite CALLER-provenance \
         component is the caller's fault by definition, not a downstream \
         artifact read.",
    ),
    (
        "crates/jammi-db/src/error.rs",
        "from",
        2,
        2, // occurrence 2 (line 566 today) — same enclosing `fn from`
        // (ordinal 2) as the entry above; a DIFFERENT construction
        // inside it, so `occurrence` (not `ordinal`) is what tells the
        // two apart.
        "The settled mechanism itself: `QueryValidationError::Width`'s \
         `QuerySource::Caller` arm — the construction-time or deferred \
         (`require_authority_width`) width check, both entries that had \
         the authority in hand when they ran. `ArtifactMismatch` (the \
         downstream, artifact-only class `require_width` raises) has no \
         `QuerySource` field to match here at all, by construction.",
    ),
    (
        "crates/jammi-db/src/store/content_hash.rs",
        "from_hex",
        1, // ordinal 1 — the only `from_hex` in this file; line 64 today
        1,
        "`ContentHash::from_hex`'s length/charset check on the stored \
         `_content_hash` column's hex text. Reviewed and left CALLER-class \
         (not re-classified this round): every production caller renders \
         this column from a source row's OWN content via \
         `content_hash_row`, which always emits exactly 64 lowercase hex \
         characters by construction (`hasher.finalize()` over a `Sha256`, \
         `hex::encode`) — a malformed value here would mean this crate's \
         own encoder is broken, not a corrupt READ of a stored artifact in \
         the same sense as `exact.rs`'s scan-width class. Not probed \
         further this round; flagged here for the next sweep rather than \
         silently cleared.",
    ),
    (
        "crates/jammi-db/src/store/content_hash.rs",
        "from_hex",
        1,
        2, // occurrence 2, same `from_hex` — line 72 today
        "`ContentHash::from_hex`'s `hex::decode_to_slice` failure arm — \
         same review as occurrence 1 above (length/charset already passed, \
         so this is an internal-consistency check on the same value).",
    ),
    (
        "crates/jammi-db/src/store/content_hash.rs",
        "cell",
        1, // ordinal 1 — the only `cell` in this file; line 126 today
        1,
        "`cell()`'s unknown-Arrow-type arm inside `content_hash_columns`. \
         Reviewed and left CALLER-class: the module doc states rendering \
         is `jammi_content_hash`'s (the `jammi-ai` UDF's) responsibility, \
         and this kernel refuses anything it was NOT told to render \
         first — a wrongly-typed column here is a caller of THIS crate \
         (the UDF) skipping its own contract, not a stored artifact's \
         drift. Not probed this round whether an end user's SQL can drive \
         an unrendered column into this kernel; flagged for the next \
         sweep.",
    ),
    (
        "crates/jammi-db/src/store/content_hash.rs",
        "content_hash_columns",
        1, // ordinal 1 — the only `content_hash_columns` in this file
        1, // occurrence 1 — line 144 today
        "`content_hash_columns`'s empty-column-list guard — an argument- \
         shape check on the kernel's own parameters, not a data read.",
    ),
    (
        "crates/jammi-db/src/store/content_hash.rs",
        "content_hash_columns",
        1,
        2, // occurrence 2 — line 154 today
        "`content_hash_columns`'s row-count-mismatch guard — an \
         argument-shape check across the kernel's own parameters (unequal \
         column lengths), not a data read.",
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "materialize_computed_embedding_table",
        1, // ordinal 1 — the only `materialize_computed_embedding_table`
        // in this file
        1, // occurrence 1 — line 3538 today
        "`materialize_computed_embedding_table`'s reserved-provenance-key \
         guard: refuses the CALLER's own `provenance.params` for carrying \
         a key this call reserves. The caller's own argument, checked at \
         the entry — no downstream artifact is read yet.",
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "materialize_computed_embedding_table",
        1,
        2, // occurrence 2 — line 3551 today
        "`materialize_computed_embedding_table`'s per-row width check \
         against `spec.dimensions` — an explicit parameter the SAME call \
         supplies alongside `rows`, both the caller's own arguments. The \
         entry for this call's own input, not a stored artifact.",
    ),
    (
        "crates/jammi-db/src/store/mod.rs",
        "materialize_computed_embedding_table",
        1,
        3, // occurrence 3 — line 3560 today
        "`materialize_computed_embedding_table`'s non-finite/zero-norm \
         check on the caller's own `rows` before they are normalized and \
         stored — the entry, before anything is written.",
    ),
    (
        "crates/jammi-db/src/store/schema.rs",
        "embedding_batch_with_null_hash",
        1, // ordinal 1 — the only `embedding_batch_with_null_hash` in this
        // file; line 48 today
        1,
        "`embedding_batch_with_null_hash`'s per-row width check against an \
         explicit `dimensions` parameter the same call receives — the \
         caller's own arguments, checked before a batch is built.",
    ),
    (
        "crates/jammi-db/src/store/vectors.rs",
        "into_caller_fault",
        1, // ordinal 1 — the only `into_caller_fault` in this file; line 65
        // today
        1,
        "`VectorColumnError::into_caller_fault` — the EXPLICIT override \
         this round's reshape introduced: every extraction helper in this \
         module defaults to the ENGINE class through `?`'s `From` \
         (`crates/jammi-db/src/store/vectors.rs`'s own module doc), and \
         this is the one call `read_keyed_vectors_f32` makes instead of \
         `?`, reserved for the one path that reads an object the CALLER \
         supplied directly (the file behind `import_embeddings`). Naming \
         the artifact type here, not gating a bool.",
    ),
];

/// The repo root, derived from this crate's manifest dir (`crates/jammi-db`)
/// rather than the process's `cwd`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-db has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-TRACKED `.rs` file under `root`-relative `dir`, sorted.
/// `git ls-files` recurses on its own — no second, hand-rolled directory
/// walk to silently stop early.
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
/// [`SURFACE_DIRS`] entry, as `(repo-relative path, source text)`. A
/// tracked file this process cannot read is a hard failure naming the
/// file, never a silent skip.
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

/// Replace every line comment, block comment, string literal (plain and
/// raw), and char literal in `text` with spaces — same length, same
/// newlines, so line numbers still match the original file. Ported
/// verbatim from `crates/jammi-ai/tests/it/pinned_source_gate.rs`'s
/// `mask_non_code` (same stated limit: a nested block comment's interior is
/// treated as code — checked clean on this surface by the same `grep -rn
/// '/\*.*/\*'` this gate's sibling already ran over the workspace).
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
/// `crates/jammi-ai/tests/it/pinned_source_gate.rs`'s own `is_ident_char`
/// makes.
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// One `fn` item found in `masked`: its name, its ORDINAL (round — DELTA
/// round 10, ported from `crates/jammi-ai/tests/it/pinned_source_gate.rs`'s
/// `assign_ordinals`: the 1-based count of functions sharing its name in
/// this file, in declaration order — stable under any edit strictly ABOVE
/// the site that does not insert/remove a same-named sibling before it,
/// unlike a declaration line, which desyncs from an unrelated doc-comment
/// or merge inserting lines above it with the reviewed function itself
/// untouched), and its own byte span `[start, end)` in `masked` — used to
/// bind a hit's byte offset to its enclosing function the same way
/// `pinned_source_gate.rs`'s `session_registration_literal_sites` binds a
/// whole-file textual hit to its enclosing function.
struct FnRegion {
    name: String,
    ordinal: usize,
    start: usize,
    end: usize,
}

/// Byte offset one past the character matching `bytes[open_idx]` (assumed
/// to be `open`), depth-counting `open`/`close` over `bytes` starting at
/// `open_idx`. `bytes` must already be comment/string/char masked (via
/// [`mask_non_code`]) so a `(`/`)`/`{`/`}` inside a literal never perturbs
/// the count. Ported from `pinned_source_gate.rs`'s `find_matching`, byte-
/// indexed rather than char-indexed to match this file's own offset space
/// (every other offset in this file — [`schema_construction_offsets`],
/// [`test_ranges`] — is already a `str::find` BYTE offset into `masked`,
/// never a char index; see [`line_of`]'s doc for why the two are not
/// interchangeable here).
fn find_matching_byte(bytes: &[u8], open_idx: usize, open: u8, close: u8) -> Option<usize> {
    debug_assert_eq!(bytes[open_idx], open);
    let mut depth = 1i64;
    let mut j = open_idx + 1;
    while j < bytes.len() {
        if bytes[j] == open {
            depth += 1;
        } else if bytes[j] == close {
            depth -= 1;
            if depth == 0 {
                return Some(j + 1);
            }
        }
        j += 1;
    }
    None
}

/// Byte offset one past the `>` matching the `<` at `bytes[open_idx]` (a
/// generic parameter list) — ported from `pinned_source_gate.rs`'s
/// `find_matching_angle`, including its round-8 fix: a `->` return arrow's
/// `>` is never treated as a closing angle bracket, so a closure/`Fn`-trait
/// bound (`Fn(&str) -> String`) in a function's generic bounds cannot
/// desync this scan from the parameter list that follows.
fn find_matching_angle_byte(bytes: &[u8], open_idx: usize) -> Option<usize> {
    debug_assert_eq!(bytes[open_idx], b'<');
    let mut depth = 1i64;
    let mut j = open_idx + 1;
    while j < bytes.len() {
        if bytes[j] == b'-' && j + 1 < bytes.len() && bytes[j + 1] == b'>' {
            j += 2;
            continue;
        }
        if bytes[j] == b'<' {
            depth += 1;
        } else if bytes[j] == b'>' {
            depth -= 1;
            if depth == 0 {
                return Some(j + 1);
            }
        }
        j += 1;
    }
    None
}

/// Assign each region its ordinal — see [`FnRegion`]'s doc. Ported verbatim
/// (in effect) from `pinned_source_gate.rs`'s `assign_ordinals`.
fn assign_ordinals(regions: &mut [FnRegion]) {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for region in regions.iter_mut() {
        let counter = seen.entry(region.name.clone()).or_insert(0);
        *counter += 1;
        region.ordinal = *counter;
    }
}

/// Every `fn` item in `masked` (unfiltered by visibility or nesting, same
/// as `pinned_source_gate.rs`'s `find_fn_regions` — a nested `fn` is found
/// too, since this scan advances one byte at a time and re-triggers on the
/// `fn` keyword inside a body exactly the way it would at the top level).
/// Ordinals are assigned by [`assign_ordinals`] over the whole per-file
/// result before it is returned.
fn find_fn_regions(masked: &str) -> Vec<FnRegion> {
    let bytes = masked.as_bytes();
    let n = bytes.len();
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i + 1 < n {
        let boundary_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
        if boundary_ok
            && bytes[i] == b'f'
            && bytes[i + 1] == b'n'
            && i + 2 < n
            && (bytes[i + 2] as char).is_whitespace()
        {
            let fn_start = i;
            let mut k = i + 2;
            while k < n && (bytes[k] as char).is_whitespace() {
                k += 1;
            }
            let name_start = k;
            while k < n && is_ident_byte(bytes[k]) {
                k += 1;
            }
            if k > name_start {
                let name = masked[name_start..k].to_string();
                let mut m = k;
                while m < n && (bytes[m] as char).is_whitespace() {
                    m += 1;
                }
                if m < n && bytes[m] == b'<' {
                    if let Some(after) = find_matching_angle_byte(bytes, m) {
                        m = after;
                        while m < n && (bytes[m] as char).is_whitespace() {
                            m += 1;
                        }
                    }
                }
                if m < n && bytes[m] == b'(' {
                    if let Some(params_end) = find_matching_byte(bytes, m, b'(', b')') {
                        let mut depth = 0i64;
                        let mut t = params_end;
                        let mut terminator = None;
                        while t < n {
                            match bytes[t] {
                                b'(' => depth += 1,
                                b')' => depth -= 1,
                                b';' | b'{' if depth == 0 => {
                                    terminator = Some(t);
                                    break;
                                }
                                _ => {}
                            }
                            t += 1;
                        }
                        if let Some(term) = terminator {
                            let region_end = if bytes[term] == b'{' {
                                match find_matching_byte(bytes, term, b'{', b'}') {
                                    Some(end) => end,
                                    None => n,
                                }
                            } else {
                                term + 1
                            };
                            regions.push(FnRegion {
                                name,
                                ordinal: 0, // assigned below, by `assign_ordinals`
                                start: fn_start,
                                end: region_end,
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

/// Bind byte offset `at` to its enclosing function: the SMALLEST region
/// (by byte span) among `regions` whose `[start, end)` contains `at` — the
/// innermost containing function, same tie-break
/// `session_registration_literal_sites` uses. `("<module-scope>", 0)` when
/// no region contains it (module-scope code — ordinal 0 is a sentinel: no
/// real function has ordinal 0, since [`assign_ordinals`] starts counting
/// at 1).
fn enclosing_function(regions: &[FnRegion], at: usize) -> (String, usize) {
    let mut best: Option<&FnRegion> = None;
    for region in regions {
        if region.start <= at && at < region.end {
            let is_smaller = match best {
                None => true,
                Some(b) => (region.end - region.start) < (b.end - b.start),
            };
            if is_smaller {
                best = Some(region);
            }
        }
    }
    best.map(|r| (r.name.clone(), r.ordinal))
        .unwrap_or_else(|| ("<module-scope>".to_string(), 0))
}

/// Every `[start, end)` byte range in `masked` that a `#[cfg(test)]`
/// attribute (found there) and the single item it gates together span —
/// computed PER MARKER via [`end_of_next_test_item`], never as "one
/// boundary point to EOF". This matters concretely on this surface:
/// `crates/jammi-db/src/config/mod.rs` gates an EXTERNAL-FILE module
/// (`#[cfg(test)] mod tests;`, whose body lives in a separate tracked file
/// this gate scans on its own) declared alongside its other `mod`
/// statements near the TOP of the file, with ordinary production code
/// (`pub use secret::{Secret, SecretSource};` and more) immediately
/// following it — a "first marker to EOF" split would wrongly swallow that
/// production code as "test". Each range here is exactly the marker's own
/// item, so unrelated code between two markers (as in that file) is
/// correctly left OUTSIDE every range, i.e. production.
fn test_ranges(rel: &str, masked: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut cursor = 0usize;
    while let Some(pos) = masked[cursor..].find("#[cfg(test)]") {
        let marker = cursor + pos;
        let end = end_of_next_test_item(rel, masked, marker);
        ranges.push((marker, end));
        cursor = end;
    }
    ranges
}

/// Every byte offset in `masked` where the CONSTRUCTION token
/// `JammiError::Schema` immediately followed by `{` occurs — a struct
/// variant's construction (or a pattern binding on it; region-split by the
/// caller distinguishes the two).
fn schema_construction_offsets(masked: &str) -> Vec<usize> {
    let needle = "JammiError::Schema";
    let bytes = masked.as_bytes();
    let mut offsets = Vec::new();
    let mut start = 0usize;
    while let Some(pos) = masked[start..].find(needle) {
        let abs = start + pos;
        let after = abs + needle.len();
        // Must be followed (skipping ASCII whitespace only — Rust allows no
        // other separator between a struct variant path and its `{`) by
        // `{`, and NOT be a substring of a longer identifier on the left
        // (Rust identifiers cannot contain `:`, so `JammiError::Schema` is
        // already a token boundary on its own left edge for this exact
        // spelling).
        let mut j = after;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'{' {
            offsets.push(abs);
        }
        start = after;
    }
    offsets
}

/// 1-based line number of byte offset `at` in `text`.
///
/// `at` MUST be an offset into the SAME string this counts newlines in —
/// never the masked offset against the original, unmasked source. Masking
/// preserves every newline's position but replaces a comment or string's
/// own multi-byte characters (an em dash, a curly quote — both common in
/// this codebase's doc comments) with a single ASCII space each, so the
/// masked text's total BYTE length differs from the original's whenever a
/// multi-byte character appears inside a comment or string before the
/// point in question. Counting in the wrong string silently shifts the
/// reported line number without erroring (measured while building this
/// gate: `crates/jammi-db/src/error.rs`'s and `store/mod.rs`'s hits were
/// off by one and by eight-to-ten lines respectively before this was
/// caught).
fn line_of(text: &str, at: usize) -> usize {
    text[..at].matches('\n').count() + 1
}

/// One production construction of `JammiError::Schema`: the file, its
/// ENCLOSING FUNCTION's name and ordinal (round 10 — see [`FnRegion`]'s
/// doc; `("<module-scope>", 0)` for a construction outside any function),
/// its OCCURRENCE (the 1-based position of this construction among every
/// `JammiError::Schema` construction inside that same enclosing function,
/// in source order — the ordinal's own idea applied one level deeper: this
/// gate's detector is per-CONSTRUCTION, not per-function like its sibling's
/// patterns are, so a single reviewed function can hold more than one
/// `JammiError::Schema {` site with two genuinely different review notes —
/// `crates/jammi-db/src/error.rs`'s `from` (`QueryValidationError` ->
/// `JammiError`) is exactly this shape, one `NonFinite` arm and one
/// `Width` arm, both inside the SAME function/ordinal. `(file, name,
/// ordinal)` alone would silently collapse the two into one allowlist
/// slot; `occurrence` keeps them distinguishable the same way `ordinal`
/// keeps two same-named functions distinguishable, and for the same
/// reason — stable under any edit that does not itself add/remove a
/// `JammiError::Schema` construction ABOVE this one inside the same
/// function), and the line — carried for human-readable diagnostics only,
/// never matched on (`ALLOWED` is keyed on `(file, name, ordinal,
/// occurrence)`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SchemaHit {
    file: String,
    name: String,
    ordinal: usize,
    occurrence: usize,
    line: usize,
}

/// Every [`SchemaHit`] where `text` constructs `JammiError::Schema` in
/// PRODUCTION code — i.e., outside every `#[cfg(test)]`-gated item's own
/// range (see [`test_ranges`]).
fn production_schema_constructions(rel: &str, text: &str) -> Vec<SchemaHit> {
    let masked = mask_non_code(text);
    let ranges = test_ranges(rel, &masked);
    let regions = find_fn_regions(&masked);
    let offsets: Vec<usize> = schema_construction_offsets(&masked)
        .into_iter()
        .filter(|&abs| !ranges.iter().any(|&(s, e)| abs >= s && abs < e))
        .collect();
    // `occurrence` is assigned per (name, ordinal) group, in ascending
    // offset order — offsets is already ascending (schema_construction_
    // offsets scans left to right), so a running per-key counter suffices.
    let mut seen: std::collections::HashMap<(String, usize), usize> =
        std::collections::HashMap::new();
    offsets
        .into_iter()
        .map(|abs| {
            let (name, ordinal) = enclosing_function(&regions, abs);
            let counter = seen.entry((name.clone(), ordinal)).or_insert(0);
            *counter += 1;
            SchemaHit {
                file: rel.to_string(),
                name,
                ordinal,
                occurrence: *counter,
                line: line_of(&masked, abs),
            }
        })
        .collect()
}

/// The gate: every PRODUCTION construction of `JammiError::Schema` across
/// the whole surface is in [`ALLOWED`], and nothing in [`ALLOWED`] is
/// extraneous — asserted as one equality so a diff names both an
/// unreviewed new site and a stale allowlist entry in the same failure.
/// Compared on `(file, name, ordinal, occurrence)` only (round 10 — see
/// [`FnRegion`]'s doc for why a declaration line is not a stable key: a
/// merge or an unrelated doc comment inserted above a reviewed site shifts
/// its line with no change to the reviewed function itself, desyncing every
/// allowlist entry below it from the code it was reviewed against — exactly
/// the failure this gate's own sibling,
/// `crates/jammi-ai/tests/it/pinned_source_gate.rs`, hit and fixed the same
/// way; `occurrence`, see [`SchemaHit`]'s doc, is this gate's own addition
/// on top of that fix, needed because a single function can hold more than
/// one `JammiError::Schema` construction); `line` is dropped before
/// comparing, kept on each [`SchemaHit`] only so a failure message can show
/// a human where to look.
#[test]
fn every_production_schema_construction_is_reviewed_as_a_caller_fault_entry() {
    let mut found: Vec<(String, String, usize, usize)> = Vec::new();
    for (rel, text) in scan_surface() {
        found.extend(
            production_schema_constructions(&rel, &text)
                .into_iter()
                .map(|h| (h.file, h.name, h.ordinal, h.occurrence)),
        );
    }
    found.sort();
    found.dedup();

    let mut allowed: Vec<(String, String, usize, usize)> = ALLOWED
        .iter()
        .map(|(path, name, ordinal, occurrence, _note)| {
            (path.to_string(), name.to_string(), *ordinal, *occurrence)
        })
        .collect();
    allowed.sort();

    assert_eq!(
        found, allowed,
        "the live `JammiError::Schema` construction sites, keyed on (file, enclosing function \
         name, ordinal, occurrence-within-that-function) (LEFT), must equal the reviewed \
         ALLOWED list (RIGHT) exactly — a site present only on the left is an UNREVIEWED new \
         construction of the caller-fault class (classify it: is this an entry that validates \
         against an authority the SAME call holds, or a downstream read of an artifact the \
         engine itself owns, which must be JammiError::IncompatibleFormat instead?); a site \
         present only on the right is a STALE allowlist entry whose code moved or was fixed — \
         remove it rather than let it become slack a different site spends"
    );
}

/// The inverse control, named separately per the contract ("carry an
/// inverse control that fires when an allowlisted site stops producing its
/// hit"): every entry in [`ALLOWED`] is a hit AGAINST TODAY'S SCAN, checked
/// one at a time so a failure names exactly which entry went stale.
/// Redundant with the equality above by construction, but the contract asks
/// for this as its own named check, and a per-entry loop gives a clearer
/// single-entry failure message than a whole-vector diff would.
#[test]
fn allowlist_entries_still_produce_their_hit() {
    let root = repo_root();
    for (path, name, ordinal, occurrence, note) in ALLOWED {
        let text = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|e| panic!("ALLOWED names {path}, which could not be read: {e}"));
        let hits: HashSet<(String, usize, usize)> = production_schema_constructions(path, &text)
            .into_iter()
            .map(|h| (h.name, h.ordinal, h.occurrence))
            .collect();
        assert!(
            hits.contains(&(name.to_string(), *ordinal, *occurrence)),
            "ALLOWED entry {path} fn {name} (ordinal {ordinal}, occurrence {occurrence}) no \
             longer produces a production JammiError::Schema construction hit there (its own \
             review note: {note:?}) — the code moved or was fixed, or a same-named sibling / an \
             earlier same-function construction was added/removed ahead of it shifting its \
             ordinal/occurrence; remove or re-point this entry rather than let it sit as unused \
             slack"
        );
    }
}

/// `crates/jammi-numerics/src` is scanned identically to `crates/jammi-db/src`
/// and asserted, not assumed, to produce zero hits — `JammiError` is not
/// even a type this crate depends on, so a hit here would mean a NEW
/// dependency edge this gate should be told about, not silently absorbed.
#[test]
fn the_numerics_crate_never_constructs_the_caller_fault_class() {
    let root = repo_root();
    let mut any = false;
    for rel in tracked_rs_files(&root, "crates/jammi-numerics/src") {
        let text = std::fs::read_to_string(root.join(&rel)).unwrap();
        let masked = mask_non_code(&text);
        if !schema_construction_offsets(&masked).is_empty() {
            any = true;
        }
    }
    assert!(
        !any,
        "crates/jammi-numerics/src now constructs JammiError::Schema somewhere — this crate did \
         not depend on jammi_db::error::JammiError when this gate was written; re-derive the \
         ALLOWED list against the new surface rather than ignoring this"
    );
}

/// Find the `mod <ident> { ... }` item starting at or after `from` in
/// `masked`, brace-balance its body, and return the byte offset one past
/// its closing `}`. Panics naming `rel` if the shape is not "`mod` then a
/// brace-balanced body" — this gate only knows how to verify that one
/// shape, the only one `#[cfg(test)]` gates anywhere on this surface today.
/// Byte offset, relative to `s`, of the character one past the bracket
/// matching `s`'s own first character (which must be `open`) — a plain
/// depth counter over `s`, which is safe here because `s` is already
/// comment/string/char-masked.
fn balanced_close(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Find the item — `mod <ident>;` (an external-file module: the item ends
/// at the `;`, its body is a SEPARATE tracked file this gate scans on its
/// own), `mod <ident> { ... }`, or `fn <ident>(...) [-> T] { ... }` — that
/// `from` (a `#[cfg(test)]` attribute's byte offset) gates, and return the
/// byte offset one past that item's end. These are the two shapes
/// `#[cfg(test)]` gates anywhere on this surface today (a bare external-file
/// `mod`, or a bare `fn`); anything else is a hard failure naming `rel`
/// rather than a silent guess.
fn end_of_next_test_item(rel: &str, masked: &str, from: usize) -> usize {
    let region = &masked[from..];
    let mod_pos = region.find("mod ");
    let fn_pos = region.find("fn ");
    let is_mod = match (mod_pos, fn_pos) {
        (Some(m), Some(f)) => m < f,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => panic!(
            "{rel}: found '#[cfg(test)]' at byte {from} with no following 'mod ' or 'fn ' — this \
             gate only knows how to verify those two shapes"
        ),
    };
    if is_mod {
        let mod_kw = mod_pos.unwrap();
        let after = &region[mod_kw..];
        let semi = after.find(';');
        let brace = after.find('{');
        match (semi, brace) {
            (Some(s), Some(b)) if s < b => from + mod_kw + s + 1,
            (Some(s), None) => from + mod_kw + s + 1,
            (_, Some(b)) => {
                let close = balanced_close(&after[b..], '{', '}').unwrap_or_else(|| {
                    panic!("{rel}: a '#[cfg(test)] mod ... {{' body never brace-balances")
                });
                from + mod_kw + b + close + 1
            }
            (None, None) => {
                panic!("{rel}: '#[cfg(test)] mod <ident>' with neither ';' nor '{{' found")
            }
        }
    } else {
        let fn_kw = fn_pos.unwrap();
        let after_fn = &region[fn_kw..];
        let paren = after_fn
            .find('(')
            .unwrap_or_else(|| panic!("{rel}: a '#[cfg(test)] fn' with no '(' found"));
        let params_close = balanced_close(&after_fn[paren..], '(', ')')
            .unwrap_or_else(|| panic!("{rel}: a '#[cfg(test)] fn's parameter list never closes"));
        let after_params = &after_fn[paren + params_close + 1..];
        let brace_rel = after_params.find('{').unwrap_or_else(|| {
            panic!(
                "{rel}: a '#[cfg(test)] fn' with no body '{{' found after its parameter list \
                 (a bodyless fn is not valid outside a trait, which this surface does not gate \
                 behind cfg(test))"
            )
        });
        let close = balanced_close(&after_params[brace_rel..], '{', '}')
            .unwrap_or_else(|| panic!("{rel}: a '#[cfg(test)] fn's body never brace-balances"));
        from + fn_kw + paren + params_close + 1 + brace_rel + close + 1
    }
}

/// Stated-limit self-check (R-A): [`test_ranges`] resolves EVERY
/// `#[cfg(test)]` marker on the whole surface to one of the three shapes
/// [`end_of_next_test_item`] recognizes (an external-file `mod ident;`, an
/// inline `mod ident { ... }`, or a `fn ... { ... }`) — a marker gating
/// anything else is a hard failure naming the file and byte offset, from
/// inside `end_of_next_test_item` itself, so this test exists to actually
/// FORCE that code to run over every marker on the surface rather than
/// trusting it never fires. Ranges are also asserted well-formed (`start <
/// end`, strictly increasing as the scan advances) — a malformed range
/// would silently mis-exclude or mis-include a construction near it.
#[test]
fn every_test_marker_resolves_to_a_recognized_item_shape() {
    for (rel, text) in scan_surface() {
        let masked = mask_non_code(&text);
        let ranges = test_ranges(&rel, &masked);
        let mut prev_end = 0usize;
        for &(start, end) in &ranges {
            assert!(
                start < end,
                "{rel}: a #[cfg(test)] item range [{start}, {end}) is not well-formed"
            );
            assert!(
                start >= prev_end,
                "{rel}: #[cfg(test)] item ranges are not in non-overlapping, increasing order \
                 near byte {start} (previous ended at {prev_end}) — test_ranges' own scan is \
                 broken, not just this file's shape"
            );
            prev_end = end;
        }
    }
}
