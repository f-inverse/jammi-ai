#!/usr/bin/env python3
"""Re-resolve every `finetune_step.rs:<n>` / `grad_oracle.rs:<n>` PATH:LINE
citation under `crates/jammi-bench/**`, `ci/scripts/perf/**`, and
`crates/jammi-kernels/artifacts/cuda-runs/**` — against the actual file
content AT HEAD for ordinary living files, or against the CITING artifact's
own recorded `git_sha` for committed evidence (see this doc's own
"Committed artifacts are append-only evidence" section below) — advisory
(i), round-2 audit fix on PR #372; the artifact sha-relative resolution is
the M1b audit round's own fix.

WHY THIS EXISTS: round 1 of this fix round "corrected" a stale citation in
`finetune_step.rs` (originally naming Rust source lines 253 through 264,
"fixed" to name line 290) by eyeballing the diff — but the code had ALREADY
moved again by the time that commit landed, so the "fixed" citation (line
290) was ALSO stale (the real line was 299) the moment it was committed.
`p1_softmax_scale_fold_ab.json` carried this SAME stale line-290 citation in
TWO fields, and `torch_finetune_step.py`/`README.md` independently carried a
DIFFERENT stale citation naming Rust source lines 112 and 233 together
(the second of those two line numbers pointed at unrelated LoRA-builder
code, not the VRAM baseline capture it was meant to cite) that neither round
noticed by eye either. A
citation that is only ever checked "by eye" at commit time is exactly the
kind of claim this repo's own `implementer-acceptance-clause` ("resolvable
citations") exists to stop being trusted on prose alone — this script is the
mechanical re-check, run every CI, not a one-time manual pass.

CONVENTION THIS SCRIPT ENFORCES: every citation in scope must be immediately
preceded by a backtick-quoted CODE IDENTIFIER (allowing only
whitespace/commas/parens/an apostrophe-s/the literal phrase "at the time of
writing" between the identifier's closing backtick and the citation itself —
see `_find_adjacent_identifier`). For example, this crate's own
`finetune_step.rs` module already carries a self-citation of exactly this
resolvable shape, right next to its own `peak.saturating_sub` expression —
see that function's test-suite doc for the literal text this script accepts
as a positive control. A BARE citation with no adjacent identifier (naming
two Rust source lines together with no code quoted next to either one — the
shape this fix replaces) is a HARD FAIL, not a lint warning — it is exactly
the shape that let a citation go stale
unnoticed, since there is nothing in the text itself a script (or a human)
can mechanically check it against. For each citation found, this script:

    1. Resolves `<file>.rs` to its one known location under
       `crates/jammi-bench/src/` (this script only knows about
       `finetune_step.rs`/`grad_oracle.rs` — the two files this crate's own
       fix-round dispatches have named; extending `_KNOWN_FILES` is a one-line
       change if a THIRD `.rs` file starts being cited this way).
    2. Checks the cited LINE NUMBER is in-bounds for that file's CURRENT
       (HEAD) line count.
    3. Checks the adjacent backtick-quoted identifier's text (whitespace-
       normalized) is a SUBSTRING of that exact line's CURRENT content
       (also whitespace-normalized) — this is what actually catches
       "the line number is in-bounds but points at different code now",
       which an in-bounds-only check (or eyeballing the diff) would miss.

## Committed artifacts are append-only evidence, not living prose (M1b audit)

A citation living inside a file under an `artifacts/` directory (any path
segment named exactly `artifacts`, not just this repo's one instance today)
that itself carries a top-level, well-formed `git_sha` is evidence ABOUT
THAT SHA'S TREE — the run the artifact's `_comment`/`provenance` fields
narrate was measured against the code as it stood at `git_sha`, and the
artifact is never edited again to track later refactors (the class this
repo's own `check_cuda_run_artifacts.py` schema already enforces: `git_sha`
is "the measured tip, kept verbatim"). Re-resolving such a citation against
HEAD is a category error, not a stricter check — it demands an edit to
historical evidence every time ANY later commit moves the cited line,
whether or not the artifact's claim about ITS OWN tree was ever wrong. Every
`file.rs:<n>` citation found inside such a file is instead resolved via `git
show <git_sha>:<path>` (this same repo's sha-relative provenance discipline,
`check_cuda_run_artifacts.py`'s ancestry model, reused rather than
reinvented) — a citation that was true at recording time PASSES regardless
of how far the working tree has since drifted; a citation that was NEVER
true even at its own declared sha still FAILS, now correctly attributed to
the artifact's own authoring mistake rather than misreported as "the code
moved". Every OTHER citing file (not under `artifacts/`, or an `artifacts/`
file with no resolvable `git_sha`) keeps the HEAD-resolution behaviour above
unchanged — living docs/scripts describe the code as it IS today, and their
citations should track that.

This needs REAL commit history to mean anything, the identical shallow-
checkout hazard `check_cuda_run_artifacts.py`'s own rule (d) already
documents: `git show <sha>:<path>` on a shallow clone (`actions/checkout`'s
default `fetch-depth: 1`) fails to find a sha outside the single fetched
commit, indistinguishable from a genuinely bad citation without checking
first. `_require_history` checks `git rev-parse --is-shallow-repository`
before the FIRST such lookup and raises one explicit, named `CitationError`
("shallow checkout") instead of N misleading per-citation findings that
would look like real drift — `.github/workflows/ci.yml`'s `citation
resolver` leg is given `fetch_depth: "0"` for exactly this reason (only that
one leg pays the deeper-clone cost; every other leg in that matrix stays at
the normal shallow default).

## The discriminator is ANCESTRY, never local object presence (post-#411 CI fix)

The FIRST version of this section resolved every artifact citation via `git
show <git_sha>:<path>` unconditionally, on the theory that a well-typed
`git_sha` is enough. It is not: `check_cuda_run_artifacts.py`'s own schema
(this same repo's ancestry model) already distinguishes a `git_sha` that IS
an ancestor of `HEAD` from one that legitimately never can be again — a
tip that was squash-merged (that gate's `merged_as`/`merged_via_pr` pair),
or, for a handful of pre-schema artifacts, a `git_sha` grandfathered into
`LEGACY_NONE_ALLOWLIST` with no ancestry claim made about it at all (see
that file's own module doc). `bf8e807` (the P1 softmax-fold artifact this
round's own audit traced) is exactly the second shape: it predates this
repo's merge-commit discipline, was never an ancestor of `main` even the
day it was recorded, and `check_cuda_run_artifacts.py` accepts it ONLY
through that reviewed-legacy arm — never by claiming its tree is reachable.
A fresh, fully-fetched clone (`fetch-depth: 0`, real history, zero shallow
ambiguity) can still legitimately never contain `bf8e807`'s objects at all,
because nothing on `main`'s own history line ever points at it. `git show`
unconditionally attempting to read it anyway does not fail closed on that —
it fails OPEN, exactly backwards: on a developer's long-lived local
checkout, an old feature branch (or its reflog) can leave `bf8e807`'s
objects sitting in the local object store even though `main` never
reaches it, so the SAME citation reads GREEN on that machine and RED on a
CI runner's fresh clone. Object presence is an accident of which branches
happened to touch a given checkout, never a property of the citation's own
correctness — the exact fail-open, environment-dependent green this gate
family exists to kill (`check_cuda_run_artifacts.py`'s own rule (d) already
refuses to let ancestry-adjacent questions turn on anything but `git
merge-base`).

Every artifact-scoped citation is instead resolved by a DETERMINISTIC
three-way split, decided by ancestry alone:

  1. **`git_sha` IS an ancestor of `HEAD`** (`git merge-base --is-ancestor
     <sha> HEAD`) and the file reads at that sha — sha-relative resolve, the
     behaviour described above, unchanged.
  2. **Ancestor, but `git show` still fails** (the sha/path genuinely does
     not resolve even though `merge-base` calls it reachable) — an ordinary
     `Violation`, same shape as before; this is a real finding about the
     citation, not an environment question.
  3. **`git_sha` is NOT an ancestor of `HEAD`** — the artifact is historical
     evidence whose tree is not on this branch's history line AT ALL, by
     construction, on every checkout with equal fetch depth. Every citation
     inside it is reported as a NAMED, non-failing **EXEMPT** line (never
     silent, never a `Violation`, never gates CI) stating plainly that it is
     historical prose this script cannot mechanically verify from this
     repository line, and that the artifact's own acceptance rests on
     `check_cuda_run_artifacts.py`'s reviewed-legacy arm
     (`LEGACY_NONE_ALLOWLIST`) instead — never on this citation resolving.
     This branch NEVER falls back to HEAD resolution and NEVER attempts
     `git show` at all: doing either would silently reintroduce the exact
     object-presence dependency this fix closes (a `git show` that happens
     to succeed on one machine's stale object store and fail on another's
     clean clone is not more informative than skipping it honestly).

Ancestry itself needs the SAME shallow guard case 1 does — `git merge-base
--is-ancestor` on a shallow clone reads back every sha as a false
non-ancestor, indistinguishable from a genuine case 3 without checking
first (`check_cuda_run_artifacts.py`'s own rule (d) documents this exact
trap for its ancestry check). `_require_history` therefore runs BEFORE any
ancestry conclusion is drawn, not just before the case-1 `git show` —
computed once per artifact file (the same sha decides every citation inside
it), never per citation.

## A maintainer guide's citations are resolved by FULL PATH, not by basename

`_KNOWN_FILES` is a BASENAME map, and that shape is load-bearing for the
roots above: a handful of files are cited by bare filename dozens of times,
so a hand-registered mapping is what makes a typo'd filename fail loudly
(`KeyError`-shaped) instead of silently resolving to the wrong file. It does
not scale past a handful. `docs/maintainer/pod-build-guide.md` alone cites
NINE distinct scripts and `docs/maintainer/cuda-kernel-guide.md` another
seven source files; registering every one of them by basename would also
make the map ambiguous the first time two directories hold a same-named file
(`layer_norm.rs` exists under both `jammi-kernels/src/ops/` and
`jammi-encoders/src/`, today).

Citing files under `_DOC_SEARCH_ROOTS` (the maintainer guides) therefore get
a SECOND, additional citation form: a repo-root-relative FULL PATH, e.g.
`rp_tree_dir` (`ci/scripts/runpod_lib.sh:687`). The full-path form is
resolved directly against the working tree, and the loud-failure property
the basename map provides is preserved by a DIFFERENT mechanism rather than
dropped: a full path that does not exist under `REPO_ROOT` is a
`Violation` ("cites <path> but that file does not exist"), never a silently
skipped citation. Both forms are subject to the IDENTICAL adjacent-
identifier rule and in-bounds check — the full-path form buys a citation no
registration step, never a weaker check. A full path is constrained to the
prefixes this repo's own sources live under (`_FULL_PATH_ROOT_PREFIXES`) so
a citation of a VENDORED third-party file (`candle-core-0.11.0/src/op.rs:…`,
which `cuda-kernel-guide.md` legitimately cites and which is not in this
tree at all) is not matched and then reported as a missing path.

The basename form still applies everywhere it did before, including inside
the doc roots. Where both forms match the same text — a full path whose last
component happens to be a registered basename — the FULL-PATH match wins and
the basename match nested inside it is dropped, so one citation is never
reported twice.

## The full-path form's coverage extension: `ci/scripts/perf/**` and crate comments

The full-path form was originally `_DOC_SEARCH_ROOTS`-only; a DOCUMENTED
RESIDUAL paragraph here used to name two real gaps this left open (found by
a survey when the form was first added). Both are now closed, each by its
own scope, never by widening `_DOC_SEARCH_ROOTS` itself (that tuple stays
"the maintainer guides", a distinct citing-audience from either extension
below):

  * **`_PERF_FULL_PATH_ROOTS`** (`ci/scripts/perf/**`, `.sh`/`.py` files
    only — not the `.json` fixtures or `.md` provenance notes living
    alongside them, which are not citation-bearing prose): the SAME
    whole-file-text scan `_DOC_SEARCH_ROOTS` gets, since a real, resolvable
    full-path citation in this scope lives in ordinary comment prose OR a
    module/class docstring (a Python docstring is not syntactically a
    comment, but it is the SAME kind of load-bearing prose a maintainer
    guide's Markdown is — `test_finetune_ab_disable_op_keys.py`'s own
    module docstring cites real call sites this way). `check_citations.py`
    (this file) and `test_check_citations.py` are excluded from this scope
    (`_PERF_FULL_PATH_EXCLUDE`): the former's own module doc and the
    latter's fixtures construct `path:line`-shaped EXAMPLE/TEST-INPUT text
    ABOUT this convention (or deliberately-broken citations exercising its
    failure paths) — never a real citation about this repo's own code —
    and mechanically re-checking prose or test data that is DESCRIBING or
    EXERCISING the rule, rather than USING it, is the same category error
    the "Committed artifacts are append-only evidence" section above
    already names for a different case.
  * **`_CRATE_COMMENT_ROOTS`** (`crates/**/*.rs`, comment/doc text ONLY): a
    `path:line`-shaped token inside an ordinary Rust string literal or in
    executable code is NEVER a citation — only text that a reader would
    recognize as documentation prose is in scope, via
    `_rust_comment_line_spans`'s lightweight lexical scan. That scan tracks
    ordinary/raw string literals, CHAR LITERALS (`'x'`, an escaped quote
    or backslash, a `\\x` byte escape, a `\\u` unicode escape, and the
    byte forms `b'x'`/`b'\\xFF'` — every one consumed whole so an
    escaped or literal `"`/`'` inside one can never be
    misread as opening a real string), and LIFETIMES/LABELS (`'a`,
    `'static`, `'_`, `'outer:`) — disambiguated from a char literal by
    whether a matching closing `'` immediately follows a single character
    or a recognized escape (a lifetime/label has no closing quote at all,
    by grammar, so this is never actually ambiguous in valid Rust; on a
    non-match the lone `'` is consumed as an ordinary character and the
    identifier after it is left for normal processing, since it triggers no
    further lexical state on its own). Getting char-literal handling right
    here is not cosmetic: BEFORE this fix, an unhandled `'"'`/`b'"'` char
    literal opened a phantom ordinary-string state on its embedded `"` that
    was never closed until the NEXT unrelated `"` anywhere later in the
    file — silently swallowing every real `//`/`///`/`//!` comment line in
    between as unscanned "string content" (measured: dozens of lines lost
    in this repo's own `crates/jammi-encoders/src/layer_norm.rs` and
    `crates/jammi-kernels/src/ops/launch_domain.rs`).

    Block comments (`/* ... */`) are tracked (including nesting) so a
    `//`-shaped substring inside one is never misread as a line comment,
    but their CONTENT is added to the scanned spans only when the block is
    itself a doc comment — `/** ... */` (exactly two asterisks opening, not
    `/***` or more — mirrors the same "exactly N, no more" rule `///`
    vs. `////` already uses for line comments) or `/*! ... */` (inner doc).
    An ordinary `/* ... */` block comment's content is deliberately NOT
    scanned: it is not doc prose a maintainer reads as documentation, and
    treating it as citation-bearing would be scanning code commentary this
    convention was never meant to reach. `#[doc = "..."]` / `#![doc =
    "..."]` attributes (the desugared form `///`/`//!` themselves compile
    to) are ALSO scanned — the attribute's string literal content (plain or
    raw `r#"..."#`) is doc text exactly like a `///` line, found by a
    separate whole-text regex pass (`_DOC_ATTR_RE`) independent of the main
    lexical scan, since attribute syntax is not itself string/comment
    lexical state.

    Unlike the two scopes above, this one is `"comments"` MODE, not
    `"text"` mode: `_full_path_citation_re` only ever sees the extracted
    comment/doc substrings, never a whole file's text. This scope ALSO
    recognizes a SECOND full-path shape unique to crate-internal doc
    comments, `_crate_relative_citation_re` (`jammi-<name>/src/...:<n>`,
    no `crates/` prefix — a crate names a sibling by its published crate
    name, not by this workspace's own directory layout, which is invisible
    from a crate's own doc-comment perspective): see that function's own
    doc for how it coexists with, and never double-counts against, the
    `crates/`-rooted form.

    `test_check_citations.py`'s `RustCommentLexerCoverageTests` is the
    coverage proof for this scope: it runs `_rust_comment_line_spans` over
    every real `.rs` file under `crates/**` and asserts every line that a
    NAIVE (lexically-blind) scan would call comment-only (`line.lstrip()`
    starts with `//`) is covered by at least one real span — the exact
    regression class the char-literal fix above closes.

Both new scopes are subject to the IDENTICAL adjacent-identifier rule and
in-bounds check the original `_DOC_SEARCH_ROOTS` form uses — the extension
buys coverage, never a weaker check.

## A frozen pre-registration's citations are pinned to their own epoch

A `docs/plans/<N>-<slug>/CONTRACT.md`-shaped pre-registration is reviewed
and FROZEN before any measurement, then kept byte-identical to that frozen
copy for the rest of the unit's life (the doctrine `docs/plans/66-tower-
profile/CONTRACT.md` names explicitly) — its Scope-facts citations describe
the code as it stood the day the freeze was reviewed, never as it reads at
whatever LATER HEAD re-runs this gate. Re-resolving them against HEAD is
the exact category error the "Committed artifacts are append-only
evidence" section above already names for a JSON artifact's own `git_sha`
field, and the inline-pin section names for a single citation's own "at
HEAD `<sha>`" phrase — this is the WHOLE-FILE form of the same principle,
for a class neither of those two covers: a file that is not JSON evidence
and carries no per-citation pin phrase, but whose ENTIRE body is frozen at
once. A file opts in with an HTML comment naming the epoch within its own
first few lines, e.g. `<!-- citations-resolve-at:
bff1fad65683760f6a6b2f6677b74e61f20d96f5 -->` (`_file_citations_epoch`) —
every `path:line` citation the rest of THIS SAME FILE carries then resolves
against that sha's tree (`git show <sha>:<path>`), never HEAD, under the
IDENTICAL adjacent-identifier and in-bounds checks every other class above
already gets. Two ways this differs from the artifacts/ arm, both
deliberate: (1) the epoch sha MUST be an ancestor of `HEAD` — fails CLOSED,
never EXEMPT, if it is not (a frozen contract's own declared epoch is
reviewed, reachable prose about THIS branch's history, not
squashed-away-and-ungraftable legacy evidence, so a non-ancestor pin here
is a wrong or fabricated header, not a legitimate historical artifact); (2)
a malformed header value (present but not a well-formed 40-character hex
sha) is ALSO a hard fail, never silently treated as "no header" — a typo'd
epoch must never quietly fall back to ordinary HEAD-relative resolution,
which would paper over the very drift this convention exists to pin
against. Scoped to `.md` files only (the pre-registration's own shape);
every other file's citations keep the resolution behaviour described above
completely unchanged by this.

## An unseen-suffix path-like token is a diagnostic, never a Violation

`_FULL_PATH_SUFFIXES` is deliberately closed (see that tuple's own module
comment: "a permissive suffix set turns ordinary prose mentioning a path
plus a number into one") -- every citation recognizer in this file
(`_citation_re`, `_full_path_citation_re`, `_crate_relative_citation_re`,
`_plan_contract_citation_like_re`) can therefore only ever MATCH a
`path:line` token whose extension is one of those nine. A backtick-quoted
token of the identical shape but a DIFFERENT extension -- `` `notes.txt:12`
`` -- is invisible to every one of them: it is neither resolved nor
reported, a genuine blind spot rather than a deliberately excluded case,
since nothing distinguishes "a real citation this repo simply has not
registered a suffix for yet" from "ordinary prose that happens to look
path-shaped" without a human reading it.

For each EPOCH-PINNED file (one that carries a well-formed
`citations-resolve-at:` header -- the class most likely to accumulate
exactly this kind of unregistered citation, since it is reviewed prose
citing a wide variety of source files by hand), `_unseen_path_like_tokens`
additionally scans for a backtick-quoted `<path>.<any-suffix>:<line-spec>`
token whose suffix is NOT in `_FULL_PATH_SUFFIXES`, and `main()` prints
each one found as a NAMED, non-failing diagnostic line -- never a
`Violation`: widening `_FULL_PATH_SUFFIXES` itself to close the blind spot
would risk exactly the false-positive class its own module comment already
rules out, and failing CI on a token that might just be ordinary prose
would be a worse mistake than the blind spot itself. The diagnostic exists
so the blind spot is VISIBLE to a reader (or a future gate) rather than
silent -- the same "never-checked must never read as checked-clean" spirit
the mandatory frozen-contract header rule above already applies to a
different blind spot, applied here to a narrower, print-only degree
because there is no reliable mechanical rule for graduating a diagnostic
into a Violation.

## A frozen pre-registration's OWN citation shape is prose, not identifier-adjacent

`docs/plans/66-tower-profile/CONTRACT.md`'s "Scope facts" section cites
Rust source lines as bare backticked tokens -- `` `trainer.rs:1952-1965,
2067-2106` ``, `` `finetune_run.rs:394` ``, occasionally a full path
(`` `crates/jammi-lora/src/lora_linear.rs:973-1005` ``), occasionally a
RELATIVE SUB-PATH that is neither (`` `ops/attention_block.rs:467,472` ``:
has a slash, but does not start with a recognized full-path root prefix),
and occasionally a bare CONTINUATION with no path half at all (`` `:1635`
``, eliding a path already named earlier on the SAME LINE -- "``
`htsat_audio.rs:1064` `` and `` `:1635` `` call ...") -- never
identifier-adjacent the way every OTHER citation class in this file is
(`_find_adjacent_identifier`'s connector convention does not apply here at
all). `_PLAN_CONTRACT_ROOTS` (scoped to this one plan group; widen it, not
the shape, the day a second frozen contract adopts the same prose
convention) opts a `.md` file under it into a SEPARATE citation form. The
RECOGNIZER (`_plan_contract_citation_like_re`) is deliberately as broad as
the punctuation alone can support: ANY backtick-to-backtick token of the
shape `` `<path-ish>:<line-spec>` `` where `<path-ish>` ends in a
`_FULL_PATH_SUFFIXES` extension, OR a bare `` `:<line-spec>` `` continuation,
is a citation this file owes a classification to -- never gated on the
line-spec ALSO being well-formed digits in the same pattern that determines
recognition, and never blind to a shape the punctuation alone does not
distinguish from ordinary prose (a relative sub-path like
`ops/attention_block.rs:467,472` has neither a bare basename, which forbids
`/`, nor a recognized full-path prefix, so it needs the sub-path arm below;
a bare continuation has no path half at all, so it needs its own scope-
tracking arm, `_check_plan_contract_citations`'s own `pending_relpath`).
Classification, once a candidate is recognized:

  - the line-spec half must fully match `<line>[-<line>][, <line>[-<line>]]*`
    (bare digit ranges only) -- a citation-shaped token whose line-spec does
    not is a `Violation` ("could not be classified"), never silently
    dropped from the checked set (module doc's "assert coverage" clause
    below);
  - a FULL PATH (starts with a recognized `_FULL_PATH_ROOT_PREFIXES`
    prefix) resolves directly, checked to exist in the pinned tree;
  - a RELATIVE SUB-PATH (contains a `/` but is not a full path) resolves by
    a UNIQUE SUFFIX match against the pinned tree at that epoch (the one
    path that IS the sub-path or ENDS WITH `/<sub-path>`) -- FAILING CLOSED
    if absent or ambiguous, never a silent first-match guess;
  - a BARE BASENAME (no `/`) resolves by searching the pinned tree at that
    epoch (`git ls-tree -r --name-only <sha>`) for a unique basename match,
    FAILING CLOSED if it is absent OR ambiguous (this repo genuinely reuses
    generic filenames like `main.rs`/`layer_norm.rs`/`attention_block.rs`
    across crates -- the same scaling limit `_KNOWN_FILES`'s own module doc
    names for why a hand-registered map does not extend past a handful of
    files), unless a `citations-basename-map` header entry narrows it (see
    below);
  - a bare CONTINUATION (no path half) resolves against the path most
    recently resolved earlier on the SAME LINE (never an earlier line, and
    never a path citation that itself failed to resolve) -- FAILING CLOSED
    with no such scope in effect, never guessed against whichever path
    happens to be nearest in the document.

Gated STRICTLY by resolution and in-bounds -- never a content re-check,
since the adjacent-identifier convention this file's other forms use has
nothing to pair against here. Only engages for a file that ALSO carries the
`citations-resolve-at:` header above (opt-IN twice over). Every individual
line/range in a comma-separated spec is checked, not just the first. Every
one of `CONTRACT.md`'s 19 citation-shaped tokens (18 named-path citations
plus the one bare continuation) is classified this way -- see `check-
citations: frozen-contract citation coverage` in this script's own `main()`
output for the live count.

## Never-checked must never read as checked-clean

A frozen pre-registration (`_is_frozen_contract_file` -- a `_PLAN_CONTRACT_
ROOTS` `.md` file whose first `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` lines
carry a FROZEN MARKER, the ONE file class the "frozen pre-registration's
citations are pinned to their own epoch" section above describes)
recognized by that DECLARED marker, never by the literal filename
`CONTRACT.md`: either the `<!-- Frozen ledger ts: ... -->` HTML comment
this contract group's own freeze convention opens every frozen file with,
or a FIRST `#`-prefixed heading line in that same window whose text
contains "CONTRACT" or "frozen" (case-insensitive) -- a LATER heading is
never consulted, so recognition cannot be smuggled in by a deep section
title that happens to use either word. A name-based check would silently
stop firing the moment a frozen contract is renamed or versioned into its
own title (`CONTRACT-v2.5.md`); a marker-based one keeps firing regardless,
because it is the declared property, not the filename, that makes a file
frozen. A recognized frozen file that does NOT carry a well-formed
`citations-resolve-at:` header in that SAME window -- whether the header is
genuinely absent, or it landed past the search window (which reads
byte-for-byte identically to absent) -- is itself a `Violation`, never a
silent skip: skipping the whole plan-contract citation scan and reporting
zero violations would be mechanically indistinguishable from "every
citation in this file resolves" -- exactly the false-negative shape a
reviewer, or a later gate, would trust.

Deliberately scoped to the frozen file ITSELF, not to every `.md` file
`_plan_contract_scope` recognizes: a companion doc in the same directory
(e.g. `docs/plans/66-tower-profile/README.md`) routinely cites a MIX of the
contract's own frozen-epoch facts (quoted verbatim, for discussion) and the
CURRENT tree (a stale-name deviation, a driver script's real line numbers)
in the same paragraph -- a single whole-file epoch pin cannot honestly
cover both, so such a file is never forced to declare one, and its
citations keep resolving HEAD-relative through the ordinary (non-plan-
contract) citation forms above. `_plan_contract_scope` itself stays
name-agnostic: any `.md` file under the root MAY still opt a citation into
this bare `path:line` form by carrying a well-formed header voluntarily
(unchanged) -- only the FROZEN file is ever forced to.

Coverage is also ASSERTED, not merely implied by the absence of a
violation: `main()` prints, per successfully-scanned `_plan_contract_scope`
file that carries a header, the exact count of citation-shaped tokens
`_plan_contract_citation_like_re` found and classified (`check-citations:
frozen-contract citation coverage: - <path>: <N> citation(s) checked`) -- a
reviewer (or a future regression) can see that number move the day a
citation is added, removed, or silently stops matching.

## An ambiguous bare basename can be disambiguated by a declared header map, never re-written in the frozen body

`docs/plans/66-tower-profile/CONTRACT.md` itself hit the ambiguity case the
section above describes: at its own pinned epoch (`bff1fad6`) this repo
already carries three `layer_norm.rs` files and ten `main.rs` files, so the
frozen body's `` `layer_norm.rs:129, 552-583` `` and `` `main.rs:115-223` ``/
`` `main.rs:1389-1400` `` citations are genuinely ambiguous by basename
alone -- and the frozen body can never be edited to spell them out as full
paths (the freeze doctrine the section above already names). The fix lives
in the HEADER ZONE instead, which is not frozen prose: an optional SECOND
HTML-comment line, immediately after `citations-resolve-at:`, e.g.

    <!-- citations-basename-map: layer_norm.rs=crates/jammi-encoders/src/layer_norm.rs; main.rs=crates/jammi-bench/src/main.rs -->

(`;`-separated `<basename>=<repo-root-relative-path>` entries; searched over
the same leading `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` lines the epoch
header is, parsed by `_file_citations_basename_map`). A malformed entry
(no `=`, or an empty key/value) is a hard fail for the WHOLE file, the same
shape a malformed `citations-resolve-at:` value already is -- never silently
treated as "no map".

This is a NARROWING of the search space `_resolve_plan_contract_target`
already does, never a way to assert a fact this script cannot itself check:
every mapped entry is validated, at the moment a citation actually uses that
basename, against the SAME pinned-tree lookup the ambiguity check itself
uses, IN THIS ORDER --

  1. if the basename is NOT actually ambiguous (one unique match already
     exists in the pinned tree, without consulting the map at all), the
     map's declared path must BE that unique match -- checked FIRST, ahead
     of the two checks below, and regardless of whether the declared path
     would otherwise pass them: a map entry is never allowed to re-point a
     citation that was already resolving correctly on its own, and a bogus
     declared value must never masquerade as a mere existence/basename
     typo by having that more specific finding hidden behind a generic one.
     This is the load-bearing property: a map can only ever RESOLVE a
     genuine ambiguity the pinned tree itself has, never silently
     substitute a different file for one a reviewer could otherwise have
     verified by eye;
  2. the declared path must exist in the pinned tree at all (a map entry
     naming a path the epoch's tree never had is a `Violation`, not a
     dead-but-harmless header line); and
  3. the declared path's OWN basename must equal the map key (a map entry
     `layer_norm.rs=crates/foo/attention.rs` is a `Violation` -- the key and
     the target must actually agree on what they claim to name).

An unmapped ambiguous basename still fails closed exactly as before -- the
map is opt-in per basename, not a blanket relaxation of the ambiguity check.

Run: `python3 ci/scripts/perf/check_citations.py`
Hermetic for every non-artifact citation (reads only files in the working
tree; no network, no build). An artifact-scoped citation additionally shells
out to `git show`/`git rev-parse --is-shallow-repository` against the local
checkout's own object database — still no network, no build, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The repo root every git subprocess call below runs against — a module
# variable (not a hardcoded `REPO_ROOT` reference inside each helper) so
# `test_check_citations.py` can monkeypatch it onto a throwaway `git
# init`'d fixture repo, the SAME pattern this file's own `_KNOWN_FILES`/
# `_SEARCH_ROOTS` already use for isolating a test from this repo's real
# citation inventory.
_GIT_REPO_ROOT = REPO_ROOT

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

SHALLOW_CHECKOUT_MESSAGE = (
    "shallow checkout — sha-relative artifact-citation resolution needs real "
    "commit history; use fetch-depth: 0"
)


class CitationError(Exception):
    """Uncomputable input (a shallow checkout, when sha-relative resolution
    is needed) — fails closed with ONE explicit message, never N misleading
    per-citation findings that would look like real drift."""

# The only `.rs` files this script currently knows how to resolve citations
# against — both live under `crates/jammi-bench/src/`. Add a new mapping
# here (not a heuristic search) the day a THIRD file starts being cited this
# way, so a typo'd filename fails loudly (KeyError-shaped) rather than
# silently resolving to the wrong file.
_KNOWN_FILES = {
    "finetune_step.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "finetune_step.rs",
    "grad_oracle.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "grad_oracle.rs",
    # round-4 audit fold-in on PR #372: the determinant tables in
    # `grad_oracle.rs`/`ab_merge.py` cite dozens of `.py:<n>` lines in the
    # torch reference scripts — those citations were NEVER mechanically
    # re-checked (this script only knew about the two `.rs` files above),
    # which is exactly how the `.py` line-drift this round's own audit
    # caught went unnoticed.
    "torch_grad_oracle.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_grad_oracle.py",
    "torch_finetune_step.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_finetune_step.py",
}

# The roots this advisory names, all searched recursively for every file
# (not just `.py`/`.json` -- a `.md` doc citation is just as resolvable
# and just as capable of going stale, see README.md's own citation this
# round fixed).
#
# Unification contract C8.4/NF15: `crates/jammi-kernels/artifacts/cuda-runs`
# joined this tuple in phase 2, the same PR that `git mv`s the two
# `finetune_step_reference.json`/`p1_softmax_scale_fold_ab.json` baselines
# OUT of `crates/jammi-bench/baselines/` and into this directory (contract
# C8) -- without this addition, the moved p1 record's own two citations of
# finetune_step.rs's batched-forward concatenation call site (see that
# record's own `_comment`) would silently drop OUT of this script's coverage
# the moment the move landed (a citation this script used to check would
# simply never be visited again, not a citation that fails loudly), which is
# precisely the "coverage regression" pressure-v2 pin H / NF15 named. (This
# comment deliberately avoids spelling out a bare `file.rs:N` citation of its
# own -- this script scans `ci/scripts/perf/**`, itself included.)
_SEARCH_ROOTS = (
    REPO_ROOT / "crates" / "jammi-bench",
    REPO_ROOT / "ci" / "scripts" / "perf",
    REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs",
)

# The citing roots whose files ALSO get the full-path citation form (see the
# module doc's "A maintainer guide's citations are resolved by FULL PATH"
# section). A separate tuple from `_SEARCH_ROOTS`, not an entry appended to
# it, precisely because the two are not interchangeable: everything here is
# additionally scanned for full-path citations, and every existing test that
# monkeypatches `_SEARCH_ROOTS` onto a throwaway fixture keeps meaning
# exactly what it meant before.
_DOC_SEARCH_ROOTS = (
    REPO_ROOT / "docs" / "maintainer",
)

# The coverage extension named in the module doc's "The full-path form's
# coverage extension" section: `ci/scripts/perf/**`'s own `.sh`/`.py`
# scripts get the SAME whole-file-text full-path scan `_DOC_SEARCH_ROOTS`
# does (never the `.json` fixtures or `.md` provenance notes also living
# under this root -- those are not citation-bearing prose). A separate
# tuple from `_DOC_SEARCH_ROOTS` (never appended to it) so `_DOC_SEARCH_
# ROOTS` keeps meaning exactly "the maintainer guides" for every existing
# test that monkeypatches it, and so this root's suffix restriction can be
# enforced independently of that tuple's (which allows any suffix).
_PERF_FULL_PATH_ROOTS = (
    REPO_ROOT / "ci" / "scripts" / "perf",
)
_PERF_FULL_PATH_SUFFIXES = (".sh", ".py")

# `check_citations.py` (this file) and `test_check_citations.py` are the
# ONE named exception inside `_PERF_FULL_PATH_ROOTS`: this file's own
# module doc constructs `path:line`-shaped EXAMPLE text describing the
# convention, and `test_check_citations.py`'s fixtures construct
# deliberately-synthetic (often deliberately-BROKEN) `path:line` text as
# PYTHON STRING LITERAL test input -- never a real citation about this
# repo's own code. Mechanically re-checking prose or test data that is
# DESCRIBING or EXERCISING this rule, rather than USING it, is the same
# category error the module doc's "Committed artifacts are append-only
# evidence" section already names for a different case.
_PERF_FULL_PATH_EXCLUDE = (
    REPO_ROOT / "ci" / "scripts" / "perf" / "check_citations.py",
    REPO_ROOT / "ci" / "scripts" / "perf" / "test_check_citations.py",
)

# The OTHER coverage extension: `crates/**/*.rs` doc/comment lines
# (`//!`/`///`/`//` only -- see `_rust_comment_line_spans`). `.rs` only
# (not every suffix `_ALL_SCAN_SUFFIXES` allows for the other roots) --
# this scope exists specifically for Rust source comments, not for a
# crate's `Cargo.toml`/README/fixtures.
_CRATE_COMMENT_ROOTS = (
    REPO_ROOT / "crates",
)

# A full-path citation must start with one of these — the roots this repo's
# OWN sources live under. Without this constraint a maintainer guide's
# legitimate citation of a VENDORED third-party file (cuda-kernel-guide.md
# cites `candle-core-0.11.0/src/op.rs` and `.../cpu_backend/mod.rs`, neither
# of which is in this tree) would match and then be reported as a missing
# path — a false finding about a citation that is correct as written.
_FULL_PATH_ROOT_PREFIXES = ("ci/scripts/", "crates/", "docs/", ".github/")

# The suffixes a full-path citation may name. Deliberately explicit (not
# "any extension"): the trailing `:<digits>` is the only structural signal
# that a path-shaped token is a citation at all, and a permissive suffix
# set turns ordinary prose mentioning a path plus a number into one.
_FULL_PATH_SUFFIXES = ("sh", "py", "rs", "cu", "toml", "md", "yml", "yaml", "json")

# A `docs/plans/<N>-<slug>/` group whose citation PROSE is a bare backticked
# `<basename>.<ext>:<line>[-<line>][, <line>[-<line>]]*` token (never
# identifier-adjacent -- `docs/plans/66-tower-profile/CONTRACT.md`'s own
# "Scope facts" section is exactly this shape: `` `trainer.rs:1952-1965,
# 2067-2106` ``, `` `finetune_run.rs:394` ``, alongside an occasional
# full-path token, `` `crates/jammi-lora/src/lora_linear.rs:973-1005` ``).
# Scoped narrowly to this one plan group (not every `docs/plans/**`
# pre-registration) -- widen this tuple, not the citation shape, the day a
# second frozen contract adopts the same prose convention.
_PLAN_CONTRACT_ROOTS = (
    REPO_ROOT / "docs" / "plans" / "66-tower-profile",
)


# --------------------------------------------------------------------------- #
# Artifact sha-relative resolution -- see the module doc's "Committed
# artifacts are append-only evidence" section above.
# --------------------------------------------------------------------------- #
def _run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_GIT_REPO_ROOT, capture_output=True, text=True)


def _is_shallow_repository() -> bool:
    proc = _run_git(["rev-parse", "--is-shallow-repository"])
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _require_history() -> None:
    """Checked before the FIRST ancestry conclusion OR `git show` an
    artifact-scoped citation needs -- one explicit `CitationError` naming
    the shallow checkout, never N misleading per-citation "does not
    resolve" findings that would look like real drift (the same discipline
    `check_cuda_run_artifacts.py`'s `run_gate` already applies to its own
    ancestry rule). Ancestry needs this EXACT same guard case 1 does: `git
    merge-base --is-ancestor` on a shallow clone reads back every sha as a
    false non-ancestor, indistinguishable from a genuine case-3 (non-
    ancestor, EXEMPT) citation without checking first.
    """
    if _is_shallow_repository():
        raise CitationError(SHALLOW_CHECKOUT_MESSAGE)


def _is_ancestor(sha: str, target: str = "HEAD") -> bool:
    """Whether `sha` is an ancestor of `target` -- the ONE deterministic
    discriminator for artifact-citation resolution (never local object
    presence, which differs by checkout history and is exactly the
    environment-dependent green this function replaces). A non-zero exit
    covers both "genuinely not an ancestor" and "not a resolvable object at
    all in this checkout" -- both correctly fall through to case 3 (EXEMPT)
    rather than attempting a `git show` whose success would depend on
    which stale branches this particular checkout happens to still hold.
    """
    proc = _run_git(["merge-base", "--is-ancestor", sha, target])
    return proc.returncode == 0


def _artifact_git_sha(path: Path) -> str | None:
    """The `git_sha` this citing file declares as ITS OWN evidence tree, or
    `None` if `path` does not qualify for sha-relative resolution at all
    (not under an `artifacts/` path segment, not JSON, unparsable, not an
    object, or no well-formed top-level `git_sha`). A file that does not
    qualify keeps the ordinary HEAD-resolution behaviour -- this is an
    opt-IN narrowing (only files that both live under `artifacts/` AND
    self-declare a resolved `git_sha` are evidence in the append-only
    sense this function exists to detect), never a heuristic guess.
    """
    if "artifacts" not in path.parts or path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    sha = data.get("git_sha")
    if isinstance(sha, str) and GIT_SHA_RE.match(sha):
        return sha
    return None


# The whole-file epoch pin -- see the module doc's "A frozen
# pre-registration's citations are pinned to their own epoch" section.
_CITATIONS_EPOCH_HEADER_RE = re.compile(r"citations-resolve-at:\s*(\S+)")

# How many leading lines of a Markdown file are searched for the header --
# generous enough to cover a SECOND HTML comment line immediately after the
# file's own opening comment (the shape a frozen contract's freeze-ledger
# comment already uses), never the whole file: a `citations-resolve-at:`-
# shaped phrase appearing deep in ordinary prose, unrelated to this
# convention, must never be misread as the header.
_CITATIONS_EPOCH_HEADER_SEARCH_LINES = 6


def _file_citations_epoch(path: Path, text: str) -> tuple[str | None, str | None]:
    """The commit sha a Markdown file's own `<!-- citations-resolve-at:
    <sha> -->` header declares every one of ITS OWN `path:line` citations
    resolves against -- the WHOLE-FILE analogue of the artifacts/ arm's
    per-file `git_sha` JSON field and the inline arm's per-citation "at
    HEAD `<sha>`" phrase, for a class neither of those two covers: a frozen
    pre-registration whose body is reviewed BEFORE measurement and then
    kept byte-identical to that frozen copy forever after -- its citations
    describe the code as it stood at review time, never as it reads at
    whatever later HEAD re-runs this gate, and the freeze doctrine forbids
    re-pointing them the moment a later, unrelated commit inserts lines
    above them.

    Returns `(sha, None)` if a well-formed header is found in the first
    `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` lines; `(None, None)` if no
    header line is present at all (ordinary HEAD-relative resolution,
    completely unchanged -- this is an opt-IN convention, never a
    heuristic guess); `(None, message)` if a header LINE is present but its
    value is not a well-formed 40-character hex commit sha -- a malformed
    pin is a HARD FAIL, never silently treated as "no header" (which would
    silently fall back to HEAD-relative resolution for a file whose author
    explicitly declared otherwise -- the same reviewability argument the
    inline commit-pin arm's own module doc section already makes for a
    fabricated sha, just for a value that is not even well-typed rather
    than one this checkout's object database cannot find).
    """
    if path.suffix != ".md":
        return None, None
    for line in text.splitlines()[:_CITATIONS_EPOCH_HEADER_SEARCH_LINES]:
        m = _CITATIONS_EPOCH_HEADER_RE.search(line)
        if not m:
            continue
        candidate = m.group(1)
        if GIT_SHA_RE.match(candidate):
            return candidate, None
        return None, (
            f"declares a 'citations-resolve-at: {candidate}' header, but {candidate!r} "
            "is not a well-formed 40-character hex commit sha"
        )
    return None, None


# The disambiguation-map header -- see the module doc's "An ambiguous bare
# basename can be disambiguated by a declared header map" section. A
# non-greedy capture up to the closing `-->` so this matches a SINGLE header
# line even though the whole thing is one HTML comment.
_CITATIONS_BASENAME_MAP_HEADER_RE = re.compile(r"citations-basename-map:\s*(.*?)\s*-->")


def _file_citations_basename_map(path: Path, text: str) -> tuple[dict[str, str] | None, str | None]:
    """The `{basename: repo-root-relative-path}` map a Markdown file's own
    `<!-- citations-basename-map: <basename>=<path>; <basename>=<path>; ... -->`
    header declares, or `(None, None)` if no such header line is present at
    all (ordinary ambiguity-checked resolution, completely unchanged -- this
    is an opt-IN convention, same as `_file_citations_epoch`'s own header).
    Searched over the SAME leading `_CITATIONS_EPOCH_HEADER_SEARCH_LINES`
    lines the epoch header is (the map is documented as living immediately
    after that header, but this function does not itself require ordering
    relative to it -- only line-count proximity to the top of the file).

    A malformed entry (missing `=`, or an empty key/value on either side of
    it) is a hard fail for the WHOLE file, `(None, <message>)` -- never
    silently dropped or treated as "no map", the same non-negotiable
    strictness `_file_citations_epoch` already applies to a malformed sha:
    a typo'd map entry must never quietly fall back to ordinary
    ambiguity-checked resolution, which would silently re-enable the exact
    AMBIGUOUS failure this header exists to name a fix for.

    This function only PARSES the header -- it does not validate that a
    declared path exists, or that its basename actually matches the key, or
    that it agrees with an unambiguous match already in the tree. Those
    three checks all need the PINNED TREE (`_ls_tree_paths(epoch_sha)`),
    which this function has no epoch to look up against; they are instead
    applied by `_resolve_plan_contract_target`, at the moment a citation
    actually uses the mapped basename (see that function's own doc).
    """
    if path.suffix != ".md":
        return None, None
    for line in text.splitlines()[:_CITATIONS_EPOCH_HEADER_SEARCH_LINES]:
        m = _CITATIONS_BASENAME_MAP_HEADER_RE.search(line)
        if not m:
            continue
        mapping: dict[str, str] = {}
        for entry in m.group(1).split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                return None, (
                    f"declares a 'citations-basename-map' header entry {entry!r} that is not "
                    "of the form <basename>=<repo-root-relative-path>"
                )
            key, _, value = entry.partition("=")
            key, value = key.strip(), value.strip()
            if not key or not value:
                return None, (
                    f"declares a 'citations-basename-map' header entry {entry!r} that is not "
                    "of the form <basename>=<repo-root-relative-path>"
                )
            mapping[key] = value
        return mapping, None
    return None, None


def _git_relpath(target_path: Path) -> str | None:
    """`target_path` (a `_KNOWN_FILES` entry, always absolute) expressed
    relative to `_GIT_REPO_ROOT` in POSIX form, the shape `git show
    <sha>:<relpath>` needs -- `None` if `target_path` does not sit under
    `_GIT_REPO_ROOT` at all (never raises; the caller turns this into an
    ordinary `Violation` instead of an uncomputable-input `CitationError`,
    since an unresolvable relpath is a real finding about THIS citation,
    not a global "history is missing" condition).
    """
    try:
        return target_path.resolve().relative_to(_GIT_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _lines_at_sha(sha: str, relpath: str) -> list[str] | None:
    """`relpath`'s content at `sha`, split into lines -- `None` if `git
    show` cannot read it there (the sha or the path does not resolve in
    this checkout's object database at all; a distinct condition from "the
    citation is stale", which needs a successful read to even evaluate).
    """
    _require_history()
    proc = _run_git(["show", f"{sha}:{relpath}"])
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


# --------------------------------------------------------------------------- #
# Inline commit-pinned citations -- "at HEAD `<sha>`" / "at `<sha>`" / "as of
# `<sha7+>`" in the SAME sentence as a citation, scoped to the two full-path
# coverage-extension scopes this fix round adds (`_PERF_FULL_PATH_ROOTS`/
# `_CRATE_COMMENT_ROOTS`), never the original `_DOC_SEARCH_ROOTS` scope --
# see `_sha_pin_eligible`. This is the SAME "committed evidence is
# append-only, not living prose" principle the artifacts/ arm above already
# applies, extended to an INLINE pin rather than a whole citing FILE's own
# `git_sha` field: `adamw_step.rs`'s own `crates/jammi-ai/src/fine_tune/
# adamw.rs:81-107` citation is explicit about which commit its claim is
# true at ("at HEAD `2c1a68d`") -- re-resolving it against a later HEAD the
# moment ANY unrelated commit moves that line is the exact category error
# the artifacts/ arm's own module doc section already names, just for a
# citation that carries its own pin inline instead of via a sibling
# `git_sha` JSON field.
#
# An unresolvable pin sha (unknown to this checkout's object database, per
# `_require_history`'s shallow guard aside) EXEMPTS by design -- it is
# never treated as a `Violation`, and never silently re-resolved against
# HEAD instead. This is deliberate, not a gap: a FABRICATED sha (typo'd, or
# copy-pasted wrong) is a reviewer-visible artefact the moment a human
# reads the diff introducing it -- exactly the same reviewability argument
# the artifacts/ arm's own `git_sha` field already rests on (see "Committed
# artifacts are append-only evidence" above). Mechanically distinguishing
# "genuinely bad sha" from "valid sha this shallow-relative clone simply
# doesn't have yet" would need the exact object-presence check the
# ancestry arm's own module doc section already rules out as
# environment-dependent and fails-open; EXEMPT is the honest, fails-closed
# answer for a signal this script cannot compute reliably, with the human
# review step (not this script) catching a fabricated pin.
# --------------------------------------------------------------------------- #
_SHA_PIN_RE = re.compile(
    r"\b(?:at(?:\s+HEAD)?|as\s+of)\s*`([0-9a-f]{7,40})`", re.IGNORECASE
)


def _sha_pin_eligible(path: Path) -> bool:
    """Whether `path` is one of the TWO NEW full-path coverage-extension
    scopes (module doc's "coverage extension" section) where an inline
    commit pin gets sha-relative resolution -- `_PERF_FULL_PATH_ROOTS`
    (`ci/scripts/perf/**`'s `.sh`/`.py`, excluding this checker's own
    implementation/test file) and `_CRATE_COMMENT_ROOTS` (`crates/**/*.rs`
    comment text). Deliberately excludes the ORIGINAL `_DOC_SEARCH_ROOTS`
    scope (the maintainer guides): that scope's citations describe the
    system as it IS today, never a historically-pinned claim, and keeps
    its existing HEAD-only behaviour completely unchanged by this.
    """
    return (
        path.suffix in _PERF_FULL_PATH_SUFFIXES
        and path not in _PERF_FULL_PATH_EXCLUDE
        and _is_under(path, _PERF_FULL_PATH_ROOTS)
    ) or (path.suffix == ".rs" and _is_under(path, _CRATE_COMMENT_ROOTS))


# The character-distance cap `_find_pin_sha` additionally applies, ON TOP
# OF the same-line-or-adjacent-line rule -- found necessary against this
# repo's own text: `ab_merge.py`'s determinant table packs an entire row
# (many unrelated citations, each in its own `|`-delimited cell) onto ONE
# physical text line, so "same line" alone let an unrelated "landed on
# `main` at `c0f0e98`" provenance aside -- INCHES away in line-count terms
# but HUNDREDS of characters away in the same giant row -- read as if it
# pinned a completely different cell's citation. The real, intended shape
# (`adamw_step.rs`'s own `at HEAD `2c1a68d`` immediately after its
# citation) sits well under 100 characters away; this cap is set generously
# above that (still far below the false-positive's ~180-290 characters) so
# a genuine pin is never rejected while a same-line-but-unrelated mention
# in a different table cell is.
_SHA_PIN_MAX_DISTANCE_CHARS = 120


def _find_pin_sha(text: str, citation_start: int) -> str | None:
    """The commit sha pinning the citation starting at `citation_start`
    (a character offset into `text`), or `None` if no pin phrase
    (`_SHA_PIN_RE`) is BOTH (a) on the citation's own line or an
    immediately preceding/following line, AND (b) within
    `_SHA_PIN_MAX_DISTANCE_CHARS` characters of it -- a sha mentioned
    further away, mentioned without one of the recognized pin phrases
    immediately before it (e.g. a commit named in unrelated prose), or
    sitting in a DIFFERENT cell of the same physical (giant, table-row-
    shaped) line, is NOT a pin and leaves the citation on the ordinary
    HEAD-relative path. Ties (more than one pin phrase within range)
    resolve to whichever is CLOSEST (by character distance) to the
    citation itself -- there is no real case in this repo's own text
    where two different pins compete for the same citation, but "closest
    wins" is the least surprising tie-break if one ever appears.
    """
    citation_line_no = text.count("\n", 0, citation_start) + 1
    best_sha: str | None = None
    best_distance: int | None = None
    for m in _SHA_PIN_RE.finditer(text):
        pin_line_no = text.count("\n", 0, m.start()) + 1
        if abs(pin_line_no - citation_line_no) > 1:
            continue
        char_distance = abs(m.start() - citation_start)
        if char_distance > _SHA_PIN_MAX_DISTANCE_CHARS:
            continue
        if best_distance is None or char_distance < best_distance:
            best_distance = char_distance
            best_sha = m.group(1)
    return best_sha


# A known filename, a colon, and a line number -- optionally wrapped in a
# matched pair of backticks around the whole citation (both forms are used
# across this repo's own docs; see this script's own module doc for an
# example of each). A bare `,:<n>` continuation immediately after a
# citation (naming a SECOND Rust source line with no `<filename>:` prefix
# of its own) is NOT matched as part of it, and is not itself a citation
# shape this pattern recognizes at all: an isolated `,:<n>` fragment,
# unattached to a `<filename>:` prefix, simply fails to match past the
# first number, degrading to "only the first number is checked" rather
# than being silently mis-parsed as one unit.
def _citation_re() -> re.Pattern:
    """Built fresh from the CURRENT `_KNOWN_FILES` on every call (never
    compiled once at import time against whatever `_KNOWN_FILES` happened to
    be at that moment) -- `test_check_citations.py` monkeypatches
    `_KNOWN_FILES` per test to isolate one fixture at a time, and a
    module-load-time-frozen pattern would silently keep matching only the
    ORIGINAL file list regardless of that patch.
    """
    return re.compile(
        r"`?(?P<file>" + "|".join(re.escape(f) for f in _KNOWN_FILES) + r"):(?P<line>\d+)`?"
    )


def _full_path_citation_re() -> re.Pattern:
    """The FULL-PATH citation form, built fresh from the CURRENT
    `_FULL_PATH_ROOT_PREFIXES`/`_FULL_PATH_SUFFIXES` on every call — same
    reason `_citation_re` is: `test_check_citations.py` monkeypatches those
    module variables per test, and a module-load-time-frozen pattern would
    silently keep matching only whatever they held at import time.

    A leading `(?<![A-Za-z0-9_./-])` guard keeps a prefix from matching in
    the MIDDLE of a longer path (`vendor/crates/foo.rs:1` must not be read
    as a `crates/`-rooted citation of a file this repo does not have).
    """
    prefixes = "|".join(re.escape(p) for p in _FULL_PATH_ROOT_PREFIXES)
    suffixes = "|".join(re.escape(s) for s in _FULL_PATH_SUFFIXES)
    return re.compile(
        r"(?<![A-Za-z0-9_./-])`?(?P<path>(?:" + prefixes + r")[A-Za-z0-9_./-]+\.(?:"
        + suffixes + r")):(?P<line>\d+)`?"
    )


def _crate_relative_citation_re() -> re.Pattern:
    """The CRATE-RELATIVE shorthand full-path form `_CRATE_COMMENT_ROOTS`
    (comments-mode) ALSO recognizes, alongside `_full_path_citation_re`'s
    `crates/...`-rooted form: a crate's own doc comment routinely names a
    SIBLING crate by its published crate name (`jammi-encoders/src/
    layer_norm.rs:353-370`), never by this workspace's `crates/` directory
    layout, which is a repo-layout implementation detail invisible from a
    crate's own doc-comment perspective (`admission.rs`'s own module doc
    uses exactly this shape, e.g. `` `"layer_norm_fused"` (`jammi-encoders/
    src/layer_norm.rs:189`) ``, right alongside its OTHER doc comments that
    spell the same crate out fully-qualified as `crates/jammi-encoders/
    src/...` — both shapes coexist in this repo's real crate comments, so
    both must resolve). A match is resolved by treating `crates/` as
    implicitly prepended. Built fresh from `_FULL_PATH_SUFFIXES` on every
    call, same reason `_full_path_citation_re` is. The SAME leading
    negative-lookbehind guard `_full_path_citation_re` uses keeps this from
    matching in the MIDDLE of a longer path -- in particular, the tail of
    an ALREADY-`crates/`-rooted citation (`crates/jammi-encoders/src/
    layer_norm.rs:91` literally contains `jammi-encoders/src/
    layer_norm.rs:91` as a substring); `_cited_targets` additionally drops
    any crate-relative match whose span nests inside a `crates/`-rooted
    match on the same segment, so a fully-qualified citation is never
    double-reported by this shorthand form too.
    """
    suffixes = "|".join(re.escape(s) for s in _FULL_PATH_SUFFIXES)
    return re.compile(
        r"(?<![A-Za-z0-9_./-])`?(?P<path>jammi-[A-Za-z0-9_-]+/(?:src|tests)/[A-Za-z0-9_./-]+\.(?:"
        + suffixes + r")):(?P<line>\d+)`?"
    )


_LINE_SPEC_FRAGMENT = r"\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*"
_LINE_SPEC_ONLY_RE = re.compile(r"^" + _LINE_SPEC_FRAGMENT + r"$")

# ANY backtick-quoted `<path>.<suffix>:<line-spec>` token, regardless of
# suffix -- module doc's "An unseen-suffix path-like token is a diagnostic,
# never a Violation" section. `_unseen_path_like_tokens` filters this down
# to the suffixes NOT in `_FULL_PATH_SUFFIXES` -- every OTHER citation
# recognizer in this file is suffix-restricted, so this is the one pattern
# that deliberately is not, purely to surface what those recognizers can
# never see.
_UNSEEN_SUFFIX_TOKEN_RE = re.compile(
    r"`(?P<path>[^`\s:]+\.(?P<suffix>[A-Za-z0-9]+)):(?P<lines>" + _LINE_SPEC_FRAGMENT + r")`"
)


def _unseen_path_like_tokens(text: str) -> list[tuple[int, str]]:
    """Every backtick-quoted, path-line-shaped token in `text` whose
    extension is NOT one of `_FULL_PATH_SUFFIXES` -- diagnostic-only (see
    module doc's "An unseen-suffix path-like token is a diagnostic, never a
    Violation" section), never gating CI. Returns `(line_no, token)` pairs
    in document order; the caller (`_check_file_impl`) scopes this to
    EPOCH-PINNED files only (a well-formed `citations-resolve-at:` header),
    not every file this script scans.
    """
    found: list[tuple[int, str]] = []
    for m in _UNSEEN_SUFFIX_TOKEN_RE.finditer(text):
        if m.group("suffix").lower() in _FULL_PATH_SUFFIXES:
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        found.append((line_no, m.group(0)))
    return found


def _plan_contract_citation_like_re() -> re.Pattern:
    r"""Every backtick-quoted, path-ish, colon-line-numbered token in a
    `_PLAN_CONTRACT_ROOTS` file's prose -- module doc's "A frozen
    pre-registration's OWN citation shape" section: ANY backtick-to-backtick
    `` `<path-ish>:<line-spec>` `` where `<path-ish>` ends in a
    `_FULL_PATH_SUFFIXES` extension is a citation this file OWES a
    classification to, whether that citation is a bare basename
    (`trainer.rs:1952-1965`), a full path
    (`crates/jammi-lora/src/lora_linear.rs:973-1005`), a RELATIVE SUB-PATH
    that is neither (`ops/attention_block.rs:467,472` -- has a slash, but
    does not start with a recognized `_FULL_PATH_ROOT_PREFIXES` prefix --
    see `_resolve_plan_contract_target`'s sub-path arm), OR a bare
    CONTINUATION token with no path half at all (`` `:1635` ``, matched via
    the SECOND alternative below into `clines` instead of `path`/`lines`) --
    CONTRACT.md's own "`` `htsat_audio.rs:1064` `` and `` `:1635` `` call
    ..." elision, where a second citation on the SAME LINE names only its
    line number and leaves the path implicit. A continuation token resolves
    against whichever path citation was most recently named earlier on that
    SAME LINE (`_check_plan_contract_citations`'s own scope-tracking state);
    one with no such preceding path is itself a `Violation`, never silently
    dropped or guessed against some earlier line's path.

    Deliberately LOOSE on each line-spec half: `(?P<lines>[^`]+)` /
    `(?P<clines>\d[^`]*)`, not `_LINE_SPEC_FRAGMENT` -- this regex's whole
    job is coverage (module doc's "assert coverage" clause: every
    citation-shaped backtick token found here must be resolved OR reported,
    never silently dropped because its line-spec half turns out to be
    malformed). A match whose line-spec half does not fully match
    `_LINE_SPEC_FRAGMENT` is itself reported as an unparsed citation by
    `_check_plan_contract_citations`, never silently excluded by a stricter
    regex that gated recognition on the line-spec ALSO being well-formed in
    the same pattern. The PATH arm's own anchor of legitimacy is its
    `_FULL_PATH_SUFFIXES` extension (any backtick span ending in `.rs`/
    `.py`/etc followed by `:<anything>` is unambiguously citation-shaped);
    the bare CONTINUATION arm has no path half to anchor on at all, so it
    additionally requires its first character to be a DIGIT (`\d`) -- this
    repo's prose routinely closes a backtick-quoted term with a bare colon
    immediately after (`` `<keys>`: refuses ... `` ), and without the digit
    anchor that ordinary prose shape reads as an (empty, non-numeric)
    citation-shaped token every bit as much as `` `:1635` `` does.

    Built fresh from `_FULL_PATH_SUFFIXES` on every call, same reason every
    other citation regex here is. The two arms are mutually exclusive by
    construction (`path` is set XOR `clines` is set), so a single
    `finditer` pass sees every citation-shaped token, in document order --
    required for the continuation arm to know which path citation came
    immediately before it.
    """
    suffixes = "|".join(re.escape(s) for s in _FULL_PATH_SUFFIXES)
    path = r"[^`\s:]+\.(?:" + suffixes + r")"
    return re.compile(r"`(?:(?P<path>" + path + r"):(?P<lines>[^`]+)|:(?P<clines>\d[^`]*))`")


def _plan_contract_scope(path: Path) -> bool:
    """Whether `path` is a `.md` file under `_PLAN_CONTRACT_ROOTS` -- the
    convention is opt-IN twice over: the file must live under the named
    plan group AND (checked by the caller) carry its own
    `citations-resolve-at:` header. Scoped to `.md` (the pre-registration's
    own shape), same restriction `_file_citations_epoch` already applies.
    Deliberately name-agnostic (a companion doc in the SAME plan directory
    -- e.g. `README.md` -- may also opt a citation into this form by
    carrying the header; see `_is_frozen_contract_file` for the NARROWER
    predicate that makes the header itself MANDATORY).
    """
    return path.suffix == ".md" and _is_under(path, _PLAN_CONTRACT_ROOTS)


# The declared FROZEN marker `_is_frozen_contract_file` recognizes -- see
# that function's own doc and the module doc's "Never-checked must never
# read as checked-clean" section. Two independent, either-is-sufficient
# shapes: an HTML comment opening with "Frozen" (the freeze-ledger header
# this contract group's own convention uses -- `<!-- Frozen ledger ts: ...
# -->`), or a FIRST heading whose text names it a contract/frozen document.
_FROZEN_MARKER_COMMENT_RE = re.compile(r"<!--\s*Frozen\b", re.IGNORECASE)
_FROZEN_HEADING_WORD_RE = re.compile(r"contract|frozen", re.IGNORECASE)


def _is_frozen_contract_file(path: Path, text: str) -> bool:
    """Whether `path` (together with its OWN text, since recognition reads
    the file's content, not its name) is a frozen pre-registration --
    module doc's "A frozen pre-registration's citations are pinned to their
    own epoch" section -- the ONLY file class the "never-checked must never
    read as checked-clean" mandatory-header rule applies to.

    Recognized by a DECLARED MARKER in the first
    `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` lines, never by the literal
    filename `CONTRACT.md`: a filename check silently stops firing the
    moment a frozen contract is renamed or versioned into its own title
    (`CONTRACT-v2.5.md`) -- exactly the drift this rule exists to make
    impossible to miss. Either marker shape is sufficient on its own:

      - `_FROZEN_MARKER_COMMENT_RE` matches anywhere in the window (the
        `<!-- Frozen ledger ts: ... -->` HTML comment this plan group's own
        freeze convention opens every frozen file with); or
      - the FIRST `#`-prefixed heading line in that same window, if its
        text matches `_FROZEN_HEADING_WORD_RE` ("contract" or "frozen",
        case-insensitive). A LATER heading is never consulted once the
        first one is found -- whether or not it matches -- so a document
        whose first heading is unrelated (a companion `README.md` titled
        "66 — tower profile close-out") is never accidentally caught by
        some deeper section heading that happens to mention either word.

    Deliberately NARROWER than `_plan_contract_scope`: a plan group's
    OTHER `.md` files (a close-out `README.md` discussing or quoting the
    frozen contract's own citations, for instance) live under the same
    `_PLAN_CONTRACT_ROOTS` directory but carry NEITHER marker shape -- they
    are not the frozen, single-epoch artifact this convention exists to
    guard, and typically cite a MIX of the contract's own frozen-epoch
    facts and the CURRENT (HEAD-relative) tree in the same paragraph -- a
    single whole-file epoch pin cannot honestly cover both, so such a file
    is never forced to declare one. Requiring a `citations-resolve-at:`
    header from every `.md` file in the directory (not just the one that
    actually carries a frozen marker) would either force those genuinely-
    HEAD-relative citations to mis-resolve against a stale pinned tree, or
    force the companion doc to stop quoting the contract's own citation
    text altogether -- neither of which the "never-checked equals
    checked-clean" property this rule protects actually requires: that
    property is about a genuinely FROZEN file never silently losing its
    own check, not about every neighboring doc opting into the same
    one-epoch-per-file model.
    """
    if not _plan_contract_scope(path):
        return False
    window = text.splitlines()[:_CITATIONS_EPOCH_HEADER_SEARCH_LINES]
    for line in window:
        if _FROZEN_MARKER_COMMENT_RE.search(line):
            return True
    for line in window:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return bool(_FROZEN_HEADING_WORD_RE.search(stripped.lstrip("#").strip()))
    return False


_LS_TREE_CACHE: dict[tuple[str, str], list[str]] = {}


def _ls_tree_paths(sha: str) -> list[str]:
    """Every path in the tree at `sha`, repo-root-relative POSIX form --
    `git ls-tree -r --name-only <sha>`, memoized per `(repo root, sha)` (a
    throwaway test fixture repo and this real checkout never collide on the
    same sha, but keying on both keeps that true by construction rather
    than by accident). Empty on a `git` failure (an unknown sha reaching
    here is already a `CitationError`-shaped case the caller's own
    `_require_history`/ancestor check rules out first)."""
    key = (str(_GIT_REPO_ROOT), sha)
    if key not in _LS_TREE_CACHE:
        proc = _run_git(["ls-tree", "-r", "--name-only", sha])
        _LS_TREE_CACHE[key] = proc.stdout.splitlines() if proc.returncode == 0 else []
    return _LS_TREE_CACHE[key]


def _resolve_plan_contract_target(
    cited_path: str, epoch_sha: str, basename_map: dict[str, str] | None = None
) -> tuple[str | None, str | None]:
    """`cited_path` (the `path` group of `_plan_contract_citation_like_re`)
    resolved to a repo-root-relative POSIX path in the tree at `epoch_sha`,
    or `(None, <error>)` if it cannot be resolved AT ALL -- FAIL CLOSED,
    never a guess:

      - A FULL-PATH citation (starts with a recognized
        `_FULL_PATH_ROOT_PREFIXES` prefix) resolves directly, checked to
        actually exist in the pinned tree (a full path is unambiguous by
        construction, so there is no uniqueness question -- only
        existence).
      - A RELATIVE SUB-PATH (contains a `/` but does NOT start with a
        recognized `_FULL_PATH_ROOT_PREFIXES` prefix -- e.g.
        `ops/attention_block.rs`, CONTRACT.md's own Scope-facts shape)
        resolves by a UNIQUE SUFFIX match against the pinned tree: the one
        path that either EQUALS `cited_path` or ENDS WITH `/<cited_path>`.
        Absent (no match) or AMBIGUOUS (more than one match) are BOTH hard
        failures, never a silent first-match guess -- the same fail-closed
        posture the bare-basename arm below already has, for the same
        reason (this repo's crate layout genuinely repeats path suffixes
        across crates, e.g. multiple `ops/attention_block.rs`-shaped
        trees, not just bare filenames).
      - A BARE BASENAME (no `/` at all) resolves by searching every path in
        the pinned tree whose own basename matches exactly: absent (no
        match) or AMBIGUOUS (more than one match -- this repo genuinely
        reuses generic filenames like `main.rs`/`layer_norm.rs` across
        crates, the exact class `_KNOWN_FILES`'s own module doc already
        names as the reason a hand-registered map does not scale) are BOTH
        hard failures, never a silent first-match guess -- UNLESS
        `basename_map` (the file's own `citations-basename-map:` header,
        see that function's doc) declares this exact basename, in which
        case it is resolved via the map instead, subject to three checks
        that make the map narrow the search space rather than assert an
        unverifiable fact (module doc's "An ambiguous bare basename can be
        disambiguated" section), checked in this ORDER (deliberately, see
        below): (1) if `cited_path` is not even ambiguous to begin with --
        one unique match already exists in the pinned tree without
        consulting the map at all -- the declared path must BE that one
        match, checked FIRST and independently of whether the declared
        path would otherwise pass existence/basename validation (a map
        entry disagreeing with an already-correct, unique resolution is
        the more specific, more actionable finding -- "you added an
        unnecessary and WRONG map entry" -- than a generic "does not
        exist"/"wrong basename" one, and reporting it first means a bogus
        declared value never masquerades as a mere existence/basename
        typo); (2) the declared path must exist in this same pinned tree;
        (3) its own basename must equal the map key. Any of the three
        failing is a `Violation`, never a silent fall-through to the
        unmapped ambiguity check below. `basename_map` is never consulted
        for a relative sub-path (its own module doc's disambiguation
        section is scoped to bare basenames only) -- an ambiguous sub-path
        is cited more precisely instead.
    """
    if cited_path.startswith(_FULL_PATH_ROOT_PREFIXES):
        tree = _ls_tree_paths(epoch_sha)
        if cited_path not in tree:
            return None, f"{cited_path} does not exist in the tree at pinned epoch {epoch_sha}"
        return cited_path, None
    tree = _ls_tree_paths(epoch_sha)
    if "/" in cited_path:
        sub_matches = sorted(
            p for p in tree if p == cited_path or p.endswith("/" + cited_path)
        )
        if not sub_matches:
            return None, (
                f"{cited_path!r} (a relative sub-path) does not exist anywhere in the tree "
                f"at pinned epoch {epoch_sha}"
            )
        if len(sub_matches) > 1:
            return None, (
                f"{cited_path!r} (a relative sub-path) is AMBIGUOUS in the tree at pinned "
                f"epoch {epoch_sha} ({len(sub_matches)} matches: {', '.join(sub_matches)}) -- "
                "cite the full path instead"
            )
        return sub_matches[0], None
    matches = sorted(p for p in tree if p.rsplit("/", 1)[-1] == cited_path)
    if basename_map is not None and cited_path in basename_map:
        mapped = basename_map[cited_path]
        if len(matches) == 1 and matches[0] != mapped:
            # Checked FIRST, ahead of existence/basename validation below --
            # see the docstring's ordering rationale. `cited_path` was
            # ALREADY resolving correctly, unambiguously, on its own; the
            # map has no business disagreeing with that, whether or not its
            # own declared value happens to separately be well-formed.
            return None, (
                f"'citations-basename-map' maps {cited_path!r} to {mapped!r}, but {cited_path!r} "
                f"already resolves uniquely to {matches[0]!r} in the pinned tree -- a map entry "
                "must never re-point a citation away from its one true match"
            )
        if mapped not in tree:
            return None, (
                f"'citations-basename-map' maps {cited_path!r} to {mapped!r}, but {mapped!r} "
                f"does not exist in the tree at pinned epoch {epoch_sha}"
            )
        mapped_basename = mapped.rsplit("/", 1)[-1]
        if mapped_basename != cited_path:
            return None, (
                f"'citations-basename-map' maps {cited_path!r} to {mapped!r}, but that path's "
                f"own basename is {mapped_basename!r}, not {cited_path!r} -- fix the map entry"
            )
        return mapped, None
    if not matches:
        return None, f"{cited_path!r} does not exist anywhere in the tree at pinned epoch {epoch_sha}"
    if len(matches) > 1:
        return None, (
            f"{cited_path!r} is AMBIGUOUS in the tree at pinned epoch {epoch_sha} "
            f"({len(matches)} matches: {', '.join(matches)}) -- cite the full path instead, or "
            "add a 'citations-basename-map' header entry"
        )
    return matches[0], None


def _check_plan_contract_citations(
    path: Path, text: str, epoch_sha: str, basename_map: dict[str, str] | None = None
) -> tuple[list[Violation], int]:
    """Every `_plan_contract_citation_like_re` match in `text`, resolved
    against the tree at `epoch_sha` (never HEAD -- this file's own
    `citations-resolve-at` header already fixed that for the whole file).
    Returns `(violations, checked)` -- `checked` is the total number of
    citation-shaped tokens found (module doc's "assert coverage" clause),
    whether they ultimately resolved, failed resolution, or were REPORTED
    AS UNPARSED (never silently dropped from the count).

    Two match shapes, walked in ONE `finditer` pass (document order matters
    -- see below): a PATH citation (`path` + `lines` groups) and a bare
    CONTINUATION (`clines` only, no path half -- `` `:1635` `` eliding a
    path already named earlier on the SAME LINE). Per PATH citation: first
    its `lines` half must fully match `_LINE_SPEC_FRAGMENT` -- a token
    whose line-spec is not a bare digit range (e.g. stray trailing text, a
    non-numeric spec) is itself a `Violation` ("could not be classified"),
    never silently excluded from being a citation at all. Otherwise:
    resolve the path (`_resolve_plan_contract_target`, fail-closed on
    absent/ambiguous), then every individual line and range in the
    comma-separated spec must be in-bounds for that file at that sha -- ALL
    of them, not just the first (a `1-2, 10` spec where only `10` is out of
    range is still a violation). A successfully-resolved path citation
    becomes the CONTINUATION SCOPE for any bare `:<line-spec>` token that
    follows it later on the same line; the scope resets (to "none") the
    moment a match on a DIFFERENT line is seen, and also resets past a
    PATH citation that itself failed to resolve (a broken citation is not a
    valid basis for a later elision to lean on). A bare continuation seen
    with no such scope in effect is itself a `Violation`, never silently
    dropped or resolved against some earlier line's path. No
    adjacent-identifier / content-match check for either shape (unlike
    every other citation form in this file): this convention's own doc
    names only resolution + in-bounds as what it gates, not a content
    re-check.
    """
    violations: list[Violation] = []
    checked = 0
    lines_cache: dict[str, list[str] | None] = {}

    def _lines(relpath: str) -> list[str] | None:
        if relpath not in lines_cache:
            lines_cache[relpath] = _lines_at_sha(epoch_sha, relpath)
        return lines_cache[relpath]

    def _check_bounds(cited_label: str, relpath: str, lines_spec: str, line_no: int) -> None:
        target_lines = _lines(relpath)
        if target_lines is None:
            violations.append(
                Violation(
                    path, line_no,
                    f"cites {cited_label} (resolved to {relpath}), but "
                    f"`git show {epoch_sha}:{relpath}` could not read that file",
                )
            )
            return
        for part in lines_spec.split(","):
            part = part.strip()
            lo_s, _, hi_s = part.partition("-")
            lo = int(lo_s)
            hi = int(hi_s) if hi_s else lo
            if lo < 1 or hi > len(target_lines) or lo > hi:
                violations.append(
                    Violation(
                        path, line_no,
                        f"cites {cited_label} but {relpath} only has "
                        f"{len(target_lines)} lines at pinned epoch {epoch_sha} (range {part!r} "
                        "does not fit)",
                    )
                )

    pending_relpath: str | None = None
    pending_line: int | None = None

    for m in _plan_contract_citation_like_re().finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        if pending_line is not None and line_no != pending_line:
            # The continuation scope is SAME-LINE only -- a path citation
            # named on an earlier line is never a valid basis for a later
            # line's own bare `:<n>` elision.
            pending_relpath = None
        checked += 1

        if m.group("path") is not None:
            cited_path = m.group("path")
            lines_spec = m.group("lines")
            cited_label = f"{cited_path}:{lines_spec}"
            if not _LINE_SPEC_ONLY_RE.match(lines_spec):
                violations.append(
                    Violation(
                        path, line_no,
                        f"citation-shaped token `{cited_label}` could not be "
                        "classified: its line-spec half is not a bare "
                        "`<line>[-<line>][, <line>[-<line>]]*` digit range -- fix the token, "
                        "or rewrite it so it does not read as a path:line citation",
                    )
                )
                pending_relpath, pending_line = None, line_no
                continue
            relpath, error = _resolve_plan_contract_target(cited_path, epoch_sha, basename_map)
            if error is not None:
                violations.append(Violation(path, line_no, f"cites {cited_label} but {error}"))
                pending_relpath, pending_line = None, line_no
                continue
            _check_bounds(cited_label, relpath, lines_spec, line_no)
            pending_relpath, pending_line = relpath, line_no
            continue

        # Bare continuation: `clines` is set, `path` is not (the two arms
        # are mutually exclusive by construction -- see `_plan_contract_
        # citation_like_re`'s own doc).
        lines_spec = m.group("clines")
        cited_label = f":{lines_spec}"
        if not _LINE_SPEC_ONLY_RE.match(lines_spec):
            violations.append(
                Violation(
                    path, line_no,
                    f"citation-shaped token `{cited_label}` could not be "
                    "classified: its line-spec half is not a bare "
                    "`<line>[-<line>][, <line>[-<line>]]*` digit range -- fix the token, "
                    "or rewrite it so it does not read as a path:line citation",
                )
            )
            pending_line = line_no
            continue
        if pending_relpath is None:
            violations.append(
                Violation(
                    path, line_no,
                    f"cites the bare continuation `{cited_label}` but no path:line citation "
                    "resolves earlier on this same line for it to elide -- a bare line-spec "
                    "token with no preceding path IN SCOPE is refused, never silently guessed "
                    "against an earlier line's path",
                )
            )
            pending_line = line_no
            continue
        _check_bounds(
            f"{cited_label} (eliding the path resolved earlier on this line)",
            pending_relpath, lines_spec, line_no,
        )
        pending_line = line_no
        # `pending_relpath` is left in place: a THIRD elided `:<n>` later on
        # the SAME line continues to resolve against the SAME preceding
        # path, not just the immediately-prior token.
    return violations, checked


def _rust_string_prefix_len(text: str, i: int, n: int) -> int:
    """Length (0, 1, or 2) of a string-literal prefix (`r`, `b`, `br`, `rb`)
    starting at `text[i]`, without consuming the hashes or the opening
    quote -- the caller decides what follows. `0` means "no such prefix
    here" (the caller falls through to treating `text[i]` as an ordinary
    character, e.g. an identifier that merely happens to start with `r`/
    `b`, like `result`).
    """
    if i >= n or text[i] not in ("r", "b"):
        return 0
    if text[i] == "b" and i + 1 < n and text[i + 1] == "r":
        return 2
    if text[i] == "r" and i + 1 < n and text[i + 1] == "b":
        return 2
    return 1


# A Rust CHAR (or byte-char, `b'...'`) literal, anchored at its opening
# `'` via `re.match(text, pos=i)`: a `\u{..}` unicode escape (1-6 hex
# digits), a `\x..` byte escape (exactly 2 hex digits), any OTHER
# single-character escape (`\n`, `\r`, `\t`, `\0`, `\\`, `\'`, `\"`, ...,
# via the catch-all `\\.`), or one ordinary (non-quote, non-backslash,
# non-newline) character -- always followed by the closing `'`. A `'` that
# does NOT match this (a lifetime like `'a`/`'static`/`'_`, or a label like
# `'outer:`) never has a closing quote at all by Rust's own grammar, so
# there is no real ambiguity between the two shapes: if this matches, it IS
# a char literal; if it does not, the `'` is a lifetime/label sigil.
_CHAR_LITERAL_RE = re.compile(
    r"'(?:\\u\{[0-9a-fA-F]{1,6}\}|\\x[0-9a-fA-F]{2}|\\.|[^'\\\n])'"
)

# `#[doc = "..."]` / `#![doc = "..."]` attributes -- the desugared form
# `///`/`//!` doc comments themselves compile down to. Matched over the
# WHOLE file text independently of the main lexical scan below (attribute
# syntax is not itself string/comment lexical state): `raw`/`hashes` cover
# the `r#"..."#` raw-string form (any hash count, via the `(?P=hashes)`
# backreference so the closing hash run must match the opening one
# exactly), `content` covers the plain `"..."` form (ordinary escapes
# tolerated via the `(?:[^"\\]|\\.)*` body so an escaped `"` inside the
# attribute string never prematurely closes the match).
_DOC_ATTR_RE = re.compile(
    r'#!?\s*\[\s*doc\s*=\s*(?:r(?P<hashes>#*)"(?P<raw>.*?)"(?P=hashes)|"(?P<content>(?:[^"\\]|\\.)*)")\s*\]',
    re.DOTALL,
)


def _rust_comment_line_spans(text: str) -> list[tuple[int, int]]:
    """Character-offset `(start, end)` ranges in `text` that are Rust
    DOC/COMMENT prose: line comment content ("//", "///", "//!" -- the
    substring strictly after the leading slashes, up to the newline), the
    content of a DOC block comment (`/** ... */`, `/*! ... */` -- never an
    ordinary `/* ... */`, see below), and the string content of a `#[doc =
    "..."]`/`#![doc = "..."]` attribute.

    A lightweight structural scan, not a full parser: it tracks just enough
    Rust lexical state -- ordinary string literals, raw strings (`r"..."`,
    `r#"..."#`, `br"..."`, `rb"..."`, any hash count), CHAR LITERALS
    (`_CHAR_LITERAL_RE` -- consumed whole, so an escaped or literal
    `"`/`'` inside one can never be misread as opening or closing a real
    string), LIFETIMES/LABELS (a `'` that is not a char literal -- consumed
    as a single ordinary character; the identifier after it needs no
    special handling since it triggers no further lexical state on its
    own), and block comments (nested) -- to keep a `//`-shaped substring
    inside any of those from being misread as a real line comment (the
    module doc's "never string literals or code" requirement for
    `_CRATE_COMMENT_ROOTS`).

    Getting the char-literal case right is load-bearing, not cosmetic: an
    unhandled `'"'`/`b'"'` char literal's embedded `"` would otherwise be
    misread as OPENING an ordinary string, which would then stay open
    (there is no closing `"` inside the char literal to pair against) until
    the NEXT unrelated `"` anywhere later in the file -- silently
    swallowing every real comment line in between as unscanned "string
    content".

    Block comments are ALWAYS tracked (so their `//`-shaped or `"`-shaped
    content never confuses the rest of this scan) but their CONTENT is
    only added to the returned spans when the block is a DOC comment --
    `/**` (exactly two asterisks opening; `/***` or more is a regular,
    non-doc comment, mirroring `///` vs. `////` for line comments) or
    `/*!` (inner doc). A bare `/**/` (four characters total) has no room
    for both a 3-char `/**` marker and a 2-char `*/` closer without
    overlapping the same `*`, so it is unambiguously the ordinary, empty,
    non-doc `/*` + `*/` -- handled as a special case rather than by the
    general marker arithmetic below.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    NORMAL, STRING, RAW_STRING, BLOCK = range(4)
    state = NORMAL
    raw_hashes = 0
    block_depth = 0
    block_is_doc = False
    block_doc_start = 0
    while i < n:
        c = text[i]
        if state == NORMAL:
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                start = i + 2
                nl = text.find("\n", start)
                end = nl if nl != -1 else n
                spans.append((start, end))
                i = end
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                if text[i : i + 4] == "/**/":
                    # See docstring: not enough room for a 3-char `/**`
                    # marker plus a distinct 2-char `*/` closer -- always
                    # the ordinary, empty, non-doc block comment.
                    i += 4
                    continue
                is_doc_outer = (
                    i + 2 < n and text[i + 2] == "*" and not (i + 3 < n and text[i + 3] == "*")
                )
                is_doc_inner = i + 2 < n and text[i + 2] == "!"
                is_doc = is_doc_outer or is_doc_inner
                state = BLOCK
                block_depth = 1
                block_is_doc = is_doc
                if is_doc:
                    block_doc_start = i + 3
                    i += 3
                else:
                    i += 2
                continue
            if c == "'":
                m = _CHAR_LITERAL_RE.match(text, i)
                if m:
                    i = m.end()
                    continue
                # Not a char literal -- a lifetime (`'a`) or label
                # (`'outer:`) instead (see module doc: never actually
                # ambiguous, since neither has a closing quote to match
                # against). Consumed as a single ordinary character; the
                # following identifier is left for NORMAL to process
                # untouched.
                i += 1
                continue
            if c == '"':
                state = STRING
                i += 1
                continue
            plen = _rust_string_prefix_len(text, i, n)
            if plen:
                j = i + plen
                hashes = 0
                k = j
                while k < n and text[k] == "#":
                    hashes += 1
                    k += 1
                if k < n and text[k] == '"':
                    is_raw = text[i] == "r" or plen == 2
                    if is_raw:
                        state = RAW_STRING
                        raw_hashes = hashes
                    else:
                        state = STRING
                    i = k + 1
                    continue
            i += 1
        elif state == STRING:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                state = NORMAL
            i += 1
        elif state == RAW_STRING:
            if c == '"':
                k = i + 1
                cnt = 0
                while k < n and cnt < raw_hashes and text[k] == "#":
                    cnt += 1
                    k += 1
                if cnt == raw_hashes:
                    state = NORMAL
                    i = k
                    continue
            i += 1
        else:  # BLOCK
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                block_depth += 1
                i += 2
                continue
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                block_depth -= 1
                i += 2
                if block_depth == 0:
                    if block_is_doc:
                        spans.append((block_doc_start, i - 2))
                    state = NORMAL
                continue
            i += 1

    for m in _DOC_ATTR_RE.finditer(text):
        if m.group("raw") is not None:
            spans.append((m.start("raw"), m.end("raw")))
        else:
            spans.append((m.start("content"), m.end("content")))
    spans.sort()
    return spans


def _full_path_mode(path: Path) -> str | None:
    """Which full-path citation scanning mode applies to `path`, or `None`
    if the full-path form is disabled for it entirely (module doc's "The
    full-path form's coverage extension" section):

      - `"text"`: the ENTIRE file text is scanned -- `_DOC_SEARCH_ROOTS`
        (any suffix; the original scope) and `_PERF_FULL_PATH_ROOTS`
        (`.sh`/`.py` only, excluding this checker's own implementation and
        test file, `_PERF_FULL_PATH_EXCLUDE`).
      - `"comments"`: only Rust LINE-comment text is scanned (via
        `_rust_comment_line_spans`) -- `_CRATE_COMMENT_ROOTS` (`.rs` only).
      - `None`: the full-path form does not apply; only the basename form
        (`_citation_re`, always on) does.
    """
    if _is_under(path, _DOC_SEARCH_ROOTS):
        return "text"
    if (
        path.suffix in _PERF_FULL_PATH_SUFFIXES
        and path not in _PERF_FULL_PATH_EXCLUDE
        and _is_under(path, _PERF_FULL_PATH_ROOTS)
    ):
        return "text"
    if path.suffix == ".rs" and _is_under(path, _CRATE_COMMENT_ROOTS):
        return "comments"
    return None


def _cited_targets(path: Path, text: str) -> list[tuple[int, str, Path, int]]:
    """Every citation in `text`, as `(match_start, label, target_path,
    cited_line)`, ordered by position.

    The basename form always applies; the full-path form applies only per
    `_full_path_mode` (module doc's "resolved by FULL PATH" / "coverage
    extension" sections). Where a full-path match SPANS a basename match —
    a path whose last component is a registered `_KNOWN_FILES` name — the
    nested basename match is dropped, so one citation is reported once, by
    its more specific form.
    """
    spans: list[tuple[int, int, str, Path, int]] = []
    mode = _full_path_mode(path)
    if mode == "text":
        for m in _full_path_citation_re().finditer(text):
            rel = m.group("path")
            spans.append(
                (m.start(), m.end(), f"{rel}:{m.group('line')}", REPO_ROOT / rel, int(m.group("line")))
            )
    elif mode == "comments":
        for c_start, c_end in _rust_comment_line_spans(text):
            segment = text[c_start:c_end]
            seg_spans: list[tuple[int, int, str, Path, int]] = []
            for m in _full_path_citation_re().finditer(segment):
                rel = m.group("path")
                seg_spans.append(
                    (m.start(), m.end(), f"{rel}:{m.group('line')}", REPO_ROOT / rel, int(m.group("line")))
                )
            seg_full_ranges = [(s, e) for (s, e, _l, _t, _n) in seg_spans]
            for m in _crate_relative_citation_re().finditer(segment):
                if any(s <= m.start() and m.end() <= e for (s, e) in seg_full_ranges):
                    continue
                rel = m.group("path")
                seg_spans.append(
                    (
                        m.start(), m.end(), f"{rel}:{m.group('line')}",
                        REPO_ROOT / "crates" / rel, int(m.group("line")),
                    )
                )
            for s, e, label, target, n in seg_spans:
                spans.append((c_start + s, c_start + e, label, target, n))
    full_spans = [(s, e) for (s, e, _l, _t, _n) in spans]
    for m in _citation_re().finditer(text):
        if any(s <= m.start() and m.end() <= e for (s, e) in full_spans):
            continue
        spans.append(
            (
                m.start(), m.end(),
                f"{m.group('file')}:{m.group('line')}",
                _KNOWN_FILES[m.group("file")],
                int(m.group("line")),
            )
        )
    spans.sort(key=lambda t: t[0])
    return [(s, label, target, n) for (s, _e, label, target, n) in spans]


def _is_under(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False

# A backtick-quoted span -- candidate identifiers.
_IDENT_RE = re.compile(r"`([^`\n]{1,200})`")

# What is allowed to sit BETWEEN an identifier's closing backtick and the
# citation that names it: whitespace, commas, parens, an apostrophe-`s`, and
# the literal phrase "at the time of writing" (the one hedge phrase this
# repo's own self-citation convention uses, see `finetune_step.rs`'s own
# `vram_sampler_finish_reports_true_delta_not_floored_by_a_baseline_at_the_peak`
# test docstring). Anything else in the gap means the identifier just
# preceding is NOT actually attached to this citation.
_CONNECTOR_RE = re.compile(
    r"^(?:'s)?[\s,\(]*(?:at the time of writing[\s,\(]*)?$", re.IGNORECASE
)

_SEARCH_WINDOW = 300


# A wrapped Rust doc-comment (`/// ...`) or line comment (`// `/`# `)
# continuation onto the next source line -- collapsed to a single space
# before the connector check below, so a citation whose adjacent identifier
# sits one wrapped comment-line above it (e.g. `finetune_step.rs`'s own
# self-citation at the time of writing this script, split across two `///`
# lines) is not penalized for the comment SYNTAX carrying it; only the
# actual prose content of the gap is checked.
_COMMENT_CONTINUATION_RE = re.compile(r"\n[ \t]*(?:///|//|#)?[ \t]*")


def _find_adjacent_identifier(ident_spans: list[re.Match], citation_start: int) -> str | None:
    """`ident_spans` is EVERY backtick-quoted span in the FULL source text
    (`_IDENT_RE.finditer(text)`, computed once per file by the caller) --
    never re-paired from a text[start:end] slice. A slice cut at an
    arbitrary character offset can land INSIDE a real backtick pair (its
    opening backtick outside the slice, only its closing backtick inside),
    which silently shifts which backticks pair with which for every
    subsequent match in that slice -- a dense table row with several
    citations close together can trip this (round-4 audit fold-in on PR
    #372's own row_lengths addition hit it: a `` `...` `` pair straddling
    the 300-char boundary made the NEXT citation's adjacent-identifier
    lookup misfire on a truncated fragment like `') | '`). Operating on
    globally-paired spans and only then filtering to the lookback window
    makes that class of misparse structurally unreachable.
    """
    window_start = max(0, citation_start - _SEARCH_WINDOW)
    candidates = [m for m in ident_spans if m.end() <= citation_start and m.start() >= window_start]
    if not candidates:
        return None
    last = candidates[-1]
    gap = _COMMENT_CONTINUATION_RE.sub(" ", last.string[last.end() : citation_start])
    if _CONNECTOR_RE.match(gap):
        return last.group(1)
    return None


def _normalize(s: str) -> str:
    return " ".join(s.split())


class Violation:
    def __init__(self, source_path: Path, line_no: int, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        try:
            rel = self.source_path.relative_to(REPO_ROOT)
        except ValueError:
            # A path outside REPO_ROOT (e.g. a test fixture in a throwaway
            # tempdir) -- print it absolute rather than raising, so a test
            # driving `main()`/`check_file()` against a fixture never trips
            # on formatting a violation it deliberately provoked.
            rel = self.source_path
        return f"{rel}:{self.line_no}: {self.message}"


class Exemption:
    """Case 3 of the artifact-citation three-way split (see module doc):
    the citing artifact's own `git_sha` is NOT an ancestor of `HEAD` --
    historical evidence off this branch's history line, reported NAMED and
    printed unconditionally (never silent), but NEVER a `Violation` -- it
    does not gate CI, and no attempt is made to resolve it (against HEAD OR
    via `git show`, which would reintroduce the exact object-presence
    dependency this class exists to avoid). Deliberately a SEPARATE class
    from `Violation` (not a subclass) -- `main()` keeps the two in
    independent lists throughout, so an `Exemption` can never be
    accidentally counted toward a FAIL by a future refactor that forgets
    to filter by type.
    """

    def __init__(self, source_path: Path, line_no: int, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        try:
            rel = self.source_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.source_path
        return f"{rel}:{self.line_no}: {self.message}"


_ALL_SCAN_SUFFIXES = (".py", ".json", ".md", ".rs", ".sh")


def _iter_source_files():
    """Every file this script scans, deduplicated across roots. `_SEARCH_
    ROOTS`/`_DOC_SEARCH_ROOTS` keep their original any-of-`_ALL_SCAN_
    SUFFIXES` breadth; `_CRATE_COMMENT_ROOTS` is `.rs`-only (module doc's
    "coverage extension" section) -- a SEPARATE per-root suffix filter,
    not a blanket one, so `crates/**` is walked once for its `.rs` files
    without also pulling in every `Cargo.toml`/README/fixture the rest of
    a crate tree carries (keeping the runtime this adds bounded to what
    the new scope actually needs).
    """
    seen = set()
    scoped_roots: list[tuple[Path, tuple[str, ...]]] = [
        (root, _ALL_SCAN_SUFFIXES)
        for root in (*_SEARCH_ROOTS, *_DOC_SEARCH_ROOTS, *_PLAN_CONTRACT_ROOTS)
    ]
    scoped_roots += [(root, (".rs",)) for root in _CRATE_COMMENT_ROOTS]
    for root, suffixes in scoped_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in suffixes:
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def _check_file_impl(
    path: Path,
) -> tuple[list[Violation], list[Exemption], list[tuple[Path, int]], list[tuple[Path, int, str]]]:
    """The real per-file scan: `check_file` (kept, unchanged signature, for
    every existing caller/test that only wants the failing findings) is a
    thin wrapper discarding the other three elements. `main()` calls this
    directly so it can also print the non-failing EXEMPT lines the
    three-way split's case 3 produces (see module doc), the frozen-contract
    CITATION COVERAGE the third element carries: a `(path, count)` entry
    per `_plan_contract_scope` file this call actually scanned for that
    citation form, `count` being the number of citation-shaped tokens
    `_check_plan_contract_citations` found and classified (module doc's
    "assert coverage" clause) -- empty for every other file, and empty
    (never populated) for a plan-contract file whose missing epoch header
    made this scan a whole-file `Violation` instead (there is nothing to
    count when the file was never opted in) -- and the UNSEEN-SUFFIX
    diagnostic tokens the fourth element carries: a `(path, line_no, token)`
    entry per backtick-quoted path-like token this EPOCH-PINNED file
    carries whose extension `_FULL_PATH_SUFFIXES` does not recognize
    (module doc's "An unseen-suffix path-like token is a diagnostic, never
    a Violation" section) -- empty for every non-epoch-pinned file, and
    never gating a `Violation` regardless of how many are found.
    """
    violations: list[Violation] = []
    exemptions: list[Exemption] = []
    plan_contract_coverage: list[tuple[Path, int]] = []
    unseen_suffix_diagnostics: list[tuple[Path, int, str]] = []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations, exemptions, plan_contract_coverage, unseen_suffix_diagnostics

    # This citing file's own evidence sha, if it is a qualifying artifact --
    # computed ONCE per file (not per citation): every citation inside the
    # SAME artifact resolves against the SAME tree, by the SAME ancestry
    # verdict.
    artifact_sha = _artifact_git_sha(path)
    ident_spans = list(_IDENT_RE.finditer(text))

    # Shallow-first ordering (module doc's "The discriminator is ANCESTRY"
    # section): the ancestry test itself needs the SAME shallow guard case
    # 1's `git show` does, checked ONCE per file, before any ancestry
    # conclusion is drawn -- never per citation, never deferred until a
    # case-1 `git show` is about to run.
    sha_is_ancestor: bool | None = None
    if artifact_sha is not None:
        _require_history()
        sha_is_ancestor = _is_ancestor(artifact_sha)

    # A frozen pre-registration's own whole-file epoch pin (module doc's "A
    # frozen pre-registration's citations are pinned to their own epoch"
    # section) -- computed ONCE per file, same as `artifact_sha`, and
    # mutually exclusive with it by construction (`.md` vs `.json` under
    # `artifacts/` never overlap on the same file). Unlike `artifact_sha`'s
    # non-ancestor case (EXEMPT, historical evidence that may legitimately
    # predate this repo's merge-commit discipline), a bad epoch header
    # here FAILS CLOSED for the whole file, immediately, before any
    # per-citation resolution: a frozen contract's declared epoch is
    # reviewed, reachable prose about THIS branch's own history, so a
    # malformed value or a non-ancestor sha is a wrong or fabricated
    # header, never a legitimate reason to fall back to ordinary
    # HEAD-relative resolution (which would silently paper over the exact
    # drift this convention exists to pin against).
    epoch_sha, epoch_header_error = _file_citations_epoch(path, text)
    if epoch_header_error is not None:
        violations.append(
            Violation(
                path, 1,
                f"{epoch_header_error} -- fix the header (or remove it, which reverts this "
                "file to ordinary HEAD-relative citation resolution)",
            )
        )
        return violations, exemptions, plan_contract_coverage, unseen_suffix_diagnostics
    if epoch_sha is not None:
        _require_history()
        if not _is_ancestor(epoch_sha):
            violations.append(
                Violation(
                    path, 1,
                    f"declares 'citations-resolve-at: {epoch_sha}', but that sha is NOT an "
                    "ancestor of HEAD -- a frozen pre-registration's declared epoch must be "
                    "real, reachable history on this branch (never EXEMPT the way pre-merge-"
                    "commit-discipline artifact evidence can be); fix the pinned sha",
                )
            )
            return violations, exemptions, plan_contract_coverage, unseen_suffix_diagnostics

        # This file IS epoch-pinned, and its declared epoch is real,
        # reachable history -- module doc's "An unseen-suffix path-like
        # token is a diagnostic, never a Violation" section: a diagnostic
        # only, collected here (never in the `else` branch below, which
        # covers ordinary HEAD-relative files this scope was never meant to
        # widen into).
        for line_no, token in _unseen_path_like_tokens(text):
            unseen_suffix_diagnostics.append((path, line_no, token))

    # This file's own optional disambiguation map (module doc's "An
    # ambiguous bare basename can be disambiguated by a declared header
    # map" section) -- parsed unconditionally alongside the epoch header
    # (a malformed entry is a whole-file FAIL, same shape as a malformed
    # epoch sha), but only ever CONSULTED below when a plan-contract bare
    # basename actually needs it (`_resolve_plan_contract_target`).
    basename_map, map_header_error = _file_citations_basename_map(path, text)
    if map_header_error is not None:
        violations.append(
            Violation(
                path, 1,
                f"{map_header_error} -- fix the header (or remove it, which reverts every "
                "ambiguous basename in this file to the ordinary fail-closed AMBIGUOUS check)",
            )
        )
        return violations, exemptions, plan_contract_coverage, unseen_suffix_diagnostics

    # A `_PLAN_CONTRACT_ROOTS` file's own bare-basename / full-path /
    # relative-sub-path `path:line[-line][, line[-line]]*` citations
    # (module doc's "A frozen pre-registration's citations are pinned to
    # their own epoch" section) -- opt-IN twice over (the file must both
    # live under the named plan group AND carry the epoch header checked
    # above), never identifier-adjacent, so this is a SEPARATE scan from
    # the `_cited_targets` loop below (whose forms never match this file's
    # own bare-basename prose at all: none of `trainer.rs`/
    # `finetune_run.rs`/etc. are `_KNOWN_FILES` entries or under any of the
    # other full-path roots).
    #
    # NEVER-CHECKED MUST NEVER READ AS CHECKED-CLEAN (module doc's
    # "coverage" clause), scoped to `_is_frozen_contract_file` (a file
    # recognized by its OWN declared frozen marker, NEVER by its filename,
    # and NEVER every `.md` file under `_PLAN_CONTRACT_ROOTS` -- see that
    # predicate's own doc for why a companion doc like `README.md` is
    # deliberately exempt from the MANDATORY-header rule while still being
    # free to opt a citation into this form voluntarily): a frozen contract
    # that does NOT carry a well-formed epoch header (`epoch_sha is None`
    # -- either the header is genuinely absent, or it landed past
    # `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` and therefore reads
    # byte-for-byte like absent) is a whole-file `Violation` here, not a
    # silent skip -- the OLD behaviour (skip, zero violations) is
    # indistinguishable from "every citation in this file resolves", which
    # is exactly the false-negative shape a reviewer or a later gate would
    # trust.
    if _is_frozen_contract_file(path, text) and epoch_sha is None:
        violations.append(
            Violation(
                path, 1,
                f"{path} carries a declared FROZEN marker under _PLAN_CONTRACT_ROOTS "
                "(an HTML comment opening with 'Frozen', or a first heading naming it a "
                "contract) but declares no 'citations-resolve-at:' header in its "
                f"first {_CITATIONS_EPOCH_HEADER_SEARCH_LINES} lines -- a frozen "
                "pre-registration's path:line citations must be pinned to an epoch and "
                "mechanically checked, never silently skipped because the header is absent "
                "or landed past the search window (which reads identically to absent); add "
                "the header near the top of the file",
            )
        )
    elif epoch_sha is not None and _plan_contract_scope(path):
        pc_violations, pc_checked = _check_plan_contract_citations(
            path, text, epoch_sha, basename_map
        )
        violations.extend(pc_violations)
        plan_contract_coverage.append((path, pc_checked))

    for citation_start, cited_label, target_path, cited_line in _cited_targets(path, text):
        # `cited_file` keeps naming the citation exactly as the doc wrote it
        # (a bare basename, or a full path) so every message below quotes
        # back the text a reader has to go find and fix.
        cited_file = cited_label.rsplit(":", 1)[0]
        source_line_no = text.count("\n", 0, citation_start) + 1

        # An inline commit pin ("at HEAD `<sha>`"/"at `<sha>`"/"as of
        # `<sha7+>`") on this exact citation's own line or an immediately
        # adjacent one -- computed PER CITATION (unlike `artifact_sha`,
        # which is one verdict for the whole file): two citations in the
        # SAME file can carry two DIFFERENT pins, or one pinned and one
        # not. Mutually exclusive with `artifact_sha` by construction
        # (`_sha_pin_eligible` and `_artifact_git_sha`'s own qualifying
        # conditions -- `.rs`/`.sh`/`.py` under the two NEW coverage-
        # extension scopes vs `.json` under an `artifacts/` path segment --
        # never overlap on the same file).
        pin_sha = None
        if artifact_sha is None and _sha_pin_eligible(path):
            pin_sha = _find_pin_sha(text, citation_start)

        if artifact_sha is not None and not sha_is_ancestor:
            # Case 3: the artifact's own git_sha is NOT an ancestor of
            # HEAD -- historical evidence off this branch's history line,
            # by construction, on every checkout with equal fetch depth.
            # NEVER resolved (not against HEAD, not via `git show`, which
            # would silently reintroduce the exact object-presence
            # dependency this split exists to close) -- named, printed,
            # non-failing.
            exemptions.append(
                Exemption(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} sha-relative to this artifact's own git_sha "
                    f"{artifact_sha}, which is NOT an ancestor of HEAD -- historical evidence "
                    "predating this repo's merge-commit discipline (typically squash-merged away), "
                    "so its tree is not reachable on this branch's history line at all, on any "
                    "checkout with equal fetch depth. EXEMPT: this citation is historical prose "
                    "that cannot be mechanically verified from this repository line; the artifact's "
                    "own acceptance rests on check_cuda_run_artifacts.py's reviewed-legacy arm "
                    "(LEGACY_NONE_ALLOWLIST), never on this citation resolving.",
                )
            )
            continue

        if artifact_sha is not None:
            # Case 1 (sha_is_ancestor is True here): sha-relative resolve.
            relpath = _git_relpath(target_path)
            if relpath is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file} sha-relative to this artifact's own git_sha "
                        f"{artifact_sha}, but {target_path} does not resolve under "
                        f"{_GIT_REPO_ROOT} for `git show`",
                    )
                )
                continue
            target_lines = _lines_at_sha(artifact_sha, relpath)
            if target_lines is None:
                # Case 2: an ancestor sha whose `git show` still fails --
                # a real finding about the citation, not an environment
                # question (ancestry already ruled that out).
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} sha-relative to this artifact's own "
                        f"recorded git_sha {artifact_sha} (an ancestor of HEAD; committed evidence "
                        "is append-only -- never re-resolved against HEAD), but "
                        f"`git show {artifact_sha}:{relpath}` could not read that file at that sha",
                    )
                )
                continue
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has "
                        f"{len(target_lines)} lines at this artifact's own git_sha {artifact_sha}",
                    )
                )
                continue
        elif epoch_sha is not None:
            # This citing file's own whole-file epoch header pins it --
            # sha-relative resolve against THAT commit's tree, never
            # against HEAD. Ancestry was already checked, fail-closed,
            # once for the whole file above -- reaching here means
            # `epoch_sha` IS an ancestor, so a `git show` miss below is a
            # real finding about THIS citation (case 2's shape), never an
            # EXEMPT (there is no non-ancestor case left to reach this
            # branch at all).
            relpath = _git_relpath(target_path)
            if relpath is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file} sha-relative to this file's own "
                        f"'citations-resolve-at: {epoch_sha}' header, but {target_path} does not "
                        f"resolve under {_GIT_REPO_ROOT} for `git show`",
                    )
                )
                continue
            target_lines = _lines_at_sha(epoch_sha, relpath)
            if target_lines is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} sha-relative to this file's own "
                        f"'citations-resolve-at: {epoch_sha}' header (an ancestor of HEAD), but "
                        f"`git show {epoch_sha}:{relpath}` could not read that file at that sha",
                    )
                )
                continue
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has "
                        f"{len(target_lines)} lines at this file's own pinned epoch {epoch_sha}",
                    )
                )
                continue
        elif pin_sha is not None:
            # This citation's OWN text pins it to a specific commit --
            # sha-relative resolve against THAT commit's tree, never
            # against HEAD (the same append-only-evidence principle the
            # artifacts/ arm applies, just keyed off an inline phrase
            # instead of a sibling `git_sha` JSON field). No ancestry
            # check here (unlike the artifacts/ arm): a `git show` that
            # fails is classified EXEMPT directly -- "the sha is unknown
            # to this clone" covers both a genuinely bad/typo'd sha and a
            # perfectly valid one this checkout's object database simply
            # does not (yet) contain, and this repository line cannot
            # mechanically tell those apart without deeper history than a
            # citation check should require.
            relpath = _git_relpath(target_path)
            if relpath is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file} sha-relative to a commit pin ({pin_sha}) in this "
                        f"citation's own text, but {target_path} does not resolve under "
                        f"{_GIT_REPO_ROOT} for `git show`",
                    )
                )
                continue
            target_lines = _lines_at_sha(pin_sha, relpath)
            if target_lines is None:
                exemptions.append(
                    Exemption(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} sha-relative to a commit pin ({pin_sha}) in "
                        "this citation's own text (an 'at HEAD `<sha>`'/'at `<sha>`'/'as of `<sha>`' "
                        f"phrase), but `git show {pin_sha}:{relpath}` cannot read that file at that sha "
                        "in this checkout. EXEMPT: the sha is unknown to this clone (a shallow fetch "
                        "depth would raise before reaching here instead -- see `_require_history`), so "
                        "this citation is historical prose that cannot be mechanically verified from "
                        "this repository line; never resolved against HEAD instead, which would "
                        "silently reintroduce the exact 'code moved since' false positive an explicit "
                        "commit pin exists to avoid.",
                    )
                )
                continue
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has "
                        f"{len(target_lines)} lines at the pinned commit {pin_sha}",
                    )
                )
                continue
        else:
            if not target_path.exists():
                violations.append(
                    Violation(path, source_line_no, f"cites {cited_file} but {target_path} does not exist")
                )
                continue
            target_lines = target_path.read_text().splitlines()
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has {len(target_lines)} lines",
                    )
                )
                continue

        ident = _find_adjacent_identifier(ident_spans, citation_start)
        if ident is None:
            violations.append(
                Violation(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} with no resolvable adjacent backtick-quoted "
                    "identifier -- a bare PATH:LINE citation cannot be mechanically re-checked "
                    "(this is exactly the shape that went stale unnoticed on PR #372); rewrite as "
                    f"`` `some_identifier`, {cited_file}:{cited_line} `` naming what is actually at "
                    "that line",
                )
            )
            continue

        cited_line_text = target_lines[cited_line - 1]
        if _normalize(ident) not in _normalize(cited_line_text):
            if artifact_sha is not None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line "
                        f"reads {cited_line_text.strip()!r} at this artifact's own recorded git_sha "
                        f"{artifact_sha} -- the citation was never true at the tree this evidence "
                        "describes (committed artifacts are append-only, sha-relative evidence -- "
                        "never re-resolve this class of citation against HEAD; the citation's line "
                        "number, or the sha it should have cited, is wrong and needs a hand fix)",
                    )
                )
            elif epoch_sha is not None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line "
                        f"reads {cited_line_text.strip()!r} at this file's own pinned epoch "
                        f"{epoch_sha} -- the citation was never true at the tree this file's header "
                        "declares (a frozen pre-registration's citations are append-only, "
                        "sha-relative to its own epoch -- never re-resolved against HEAD; the "
                        "citation's line number needs a hand fix, which requires un-freezing this "
                        "file first)",
                    )
                )
            elif pin_sha is not None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line "
                        f"reads {cited_line_text.strip()!r} at the pinned commit {pin_sha} -- the "
                        "citation is STALE even at its own pinned commit (a wrong line number, or the "
                        "wrong sha was pinned); never re-resolved against HEAD instead, which would "
                        "misattribute this as ordinary code drift rather than a wrong pin",
                    )
                )
            else:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line is "
                        f"currently {cited_line_text.strip()!r} -- the citation is STALE (the code moved "
                        "since this was written); re-resolve it against the file at HEAD",
                    )
                )
    return violations, exemptions, plan_contract_coverage, unseen_suffix_diagnostics


def check_file(path: Path) -> list[Violation]:
    violations, _exemptions, _plan_contract_coverage, _unseen_suffix_diagnostics = _check_file_impl(path)
    return violations


def main() -> int:
    all_violations: list[Violation] = []
    all_exemptions: list[Exemption] = []
    all_plan_contract_coverage: list[tuple[Path, int]] = []
    all_unseen_suffix_diagnostics: list[tuple[Path, int, str]] = []
    checked = 0
    try:
        for path in _iter_source_files():
            checked += 1
            v, e, pc, u = _check_file_impl(path)
            all_violations.extend(v)
            all_exemptions.extend(e)
            all_plan_contract_coverage.extend(pc)
            all_unseen_suffix_diagnostics.extend(u)
    except CitationError as exc:
        print(f"check-citations: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if all_exemptions:
        print(
            f"check-citations: {len(all_exemptions)} EXEMPT citation(s) -- historical evidence "
            "whose git_sha is not an ancestor of HEAD, never mechanically verified from this "
            "repository line (see check_cuda_run_artifacts.py's reviewed-legacy arm instead):"
        )
        for e in all_exemptions:
            print(f"  - EXEMPT: {e}")

    if all_plan_contract_coverage:
        # Module doc's "assert coverage" clause: a per-pinned-file count of
        # citations CHECKED, printed unconditionally (pass or fail) so
        # "checked" is never conflated with "checked clean" -- a reader (or
        # a future regression) can see the count move.
        print("check-citations: frozen-contract citation coverage:")
        for cpath, count in all_plan_contract_coverage:
            try:
                rel = cpath.relative_to(REPO_ROOT)
            except ValueError:
                rel = cpath
            print(f"  - {rel}: {count} citation(s) checked")

    if all_unseen_suffix_diagnostics:
        # Module doc's "An unseen-suffix path-like token is a diagnostic,
        # never a Violation" section: named and printed unconditionally
        # (pass or fail), never gating CI -- the blind spot every citation
        # recognizer in this file has for a suffix `_FULL_PATH_SUFFIXES`
        # does not list, made visible rather than silent.
        print(
            "check-citations: unseen path-like token(s) outside _FULL_PATH_SUFFIXES "
            "(diagnostic only, never a Violation):"
        )
        for upath, line_no, token in all_unseen_suffix_diagnostics:
            try:
                rel = upath.relative_to(REPO_ROOT)
            except ValueError:
                rel = upath
            print(f"  - {rel}:{line_no}: {token}")

    if all_violations:
        print("check-citations: FAIL", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(f"check-citations: {checked} file(s) scanned, all PATH:LINE citations resolve "
          "(HEAD for living files, each artifact's own recorded git_sha for committed evidence "
          f"reachable from HEAD; {len(all_exemptions)} exempt as non-ancestor legacy evidence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
