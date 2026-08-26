#!/usr/bin/env python3
"""Every measurement-shaped number in a kernel/test doc comment or `assert!`
message must cite what produced it.

The class this closes: a doc comment or an `assert!`/`panic!` message states
a precise-looking measurement — "26% of elements", "1/6144", "100%
(6144/6144)", a bare cosine like "0.796" — that nothing in the tree actually
produces. The number reads as evidence (a real run, a real oracle) but is
unverifiable: no test, no artifact, no script anywhere computes it, so a
reader (or an implementer citing it as ground truth) cannot re-derive or even
locate where it came from. `docs/maintainer/cuda-kernel-guide.md` §3.9 states
this rule in one paragraph, alongside the adjacent oracle disciplines §3.7
(write comparisons affirmatively) and §3.8 (no absolute ULP floor) this is a
third leg of. §3.9 is also `KO-4` of that guide's kernel-acceptance-oracle
standard.

ROUND-6 (KO-4) WIDENING: the same "unproduced number reads as evidence"
class applies to a FLOOR TERM in CODE, not just a number in a doc comment —
a `const`/`let` whose name ends `_floor` (covers `_FLOOR`, `_ABS_FLOOR`, and
the lowercase `abs_floor` local-variable idiom), or a `.max(<literal>)`/
`+ <literal>` inside a fn whose name or `-> bool` return reads as
comparison-feeding. `find_floor_hits`/its call site in `scan_lines` below
carry the full rule and citation-scope rationale; the shape is otherwise
independent of (never a repeat of) the three number shapes below.

BINDING UNIT: the contiguous doc-comment BLOCK — a maximal run of `///`/
`//!`/`//` lines — containing the number (round-3 audit fix: a 12-line
window both missed a same-block citation written one sentence AWAY from the
number it covers, and could reach a citation several lines off that has
nothing to do with the number it happened to sit near). A citation ANYWHERE
in the block covers every number in that block. `no-producer: <reason>` is
the one exception — it stays LINE-scoped: it exempts only the number(s) on
its own line, never the rest of the block, so a stray opt-out cannot
silently launder an unrelated number several lines away. An
`assert!`/`panic!`-family message is always its own singleton block (it is
code, not a comment run).

WHAT COUNTS AS "measurement-shaped": three number shapes, each REQUIRED to
have one of `ADJACENT_WORD_ALTS` appear (case-insensitively) ANYWHERE in the
number's own BLOCK — not a plain N/M, X%, or 0.XXX floating around in math
notation, an enumeration, a fixed statistical constant, or a defined
threshold, all of which carry no claim of an observed run:

  - `N/M` element/mismatch counts, REQUIRED to have `M >= 64` (a smaller
    denominator is almost always an index/enumeration/small ratio in this
    codebase, never a real mismatch-element count, which starts in the
    hundreds) and NOT: immediately preceded by `.` (a decimal-embedded
    fragment, e.g. the `3/16` inside `1.3/16`), `/` or immediately followed
    by `/` (a fragment of a longer `a/b/c` chain — excluding both
    directions kills every non-first segment of a chain in one rule), `^`
    (an exponent, `x^2/2`), or an `n=`-prefixed shape enumeration
    (`n=32/900`).
  - percentages (`X%`, `X.Y%`).
  - bare cosines (`0.dd` through `0.dddddd` — 2 to 6 decimal digits).

  A version string (`0.11.0`, or a known crate name immediately before a
  bare `X.Y` — `candle 0.11`, `candle-core 0.11's`) is excluded from all
  three shapes: it is never a measurement.

  Round-3 audit fix: the round-2 "definitional operator" exclusion
  (anything right of `=`/`>=`/etc. was dropped) and the "same line as a
  `[`CONST`]` doc-link" exclusion (a whole line was blanket-suppressed by
  the mere presence of an unrelated link on it) are BOTH REMOVED — a
  measured value restated as an equality (`measured on the A100: 3.1e-3 =
  0.39`) still counts.

WHERE: doc comments (`///`, `//!`) and plain `//` comment blocks, plus the
message string of an `assert!`/`assert_eq!`/`assert_ne!`/`debug_assert!`/
`panic!` call (single-line message only — a multi-line `format!` message
inside the macro call is out of scope for this pass, the same "cheap,
visible, first-pass tripwire" cost `check_sqlite_isms.py`'s own doc accepts),
under `crates/jammi-kernels/{src,tests}`, `crates/jammi-encoders/src`,
`crates/jammi-lora/src`, `crates/jammi-bench/src` — `.rs` and CUDA source
(`.cu`/`.cuh`) files.

PRODUCER CITATION GRAMMAR (round-3 audit fix: the round-2 grammar only
recognised a BARE identifier after `see`/`printed by`, rejecting this
repo's own dominant idioms — `` see [`fn`] `` alone outnumbers the bare
form several times over — while ALSO accepting `see below`/`see table
3`/`see step 2`, none of which name a real producer). A citation is one of:

  - `` see [`<path>`] ``, `` see `<path>` ``, or `see <path>` — `<path>` is
    a `::`-separated identifier chain (`ops::softmax::MAX_LAST_DIM`,
    `Self::with_scale`, a bare `check_bit_identity_fixture`); resolved by
    its LAST segment.
  - `printed by` in the same three forms, resolved the same way.
  - `measured by <path>` — `<path>` is a filesystem path; resolved by
    `git ls-files` membership (a committed artifact), never a bare
    `Path.exists()` (which would accept an untracked, gitignored file).
  - `` [producer: <path or fn form>] `` — the same resolution as `see`,
    OR (if the content isn't a `::`-path/identifier shape) a tracked path.
  - a BARE `` [`<path>`] `` rustdoc link, with NO `see`/`printed by`
    lead-in at all (round-4 audit fix) — accepted ONLY for this bracketed
    form; a bare/backtick-only path with no keyword is still NOT a
    citation (this repo writes far more `` [`Type::method`] `` cross-
    references that are not citations than ones that are — the bracket
    form alone is not evidence; it becomes evidence ONLY because its
    resolved fn, checked the same way as every other form, turns out to
    be a real `#[test]`).

A resolved fn name is REQUIRED to be a `#[test]`-attributed function
(including `#[tokio::test]`/any attribute whose text contains "test"), or a
`pub fn` declared in a file under a `tests/` directory — never "any fn with
that name exists somewhere", which is what let `see below` resolve against
a real, but wholly unrelated, non-test `fn below` a prior round of this
gate shipped with. This is the one piece of real discipline in the
grammar: it is what makes `see table 3` and `see step 2` correctly stay
unresolved (neither names a real identifier at all) and what makes `see
below` correctly stay unresolved EVEN THOUGH a real `fn below` exists
elsewhere in the tree (it is not a test).

FAIL-CLOSED, file:line, offending number. A committed allowlist
(`ALLOWLIST_PATH`) carries pre-existing debt, keyed on `file:number:sha1` —
`sha1` of the NORMALIZED text of the number's own line (not the line
number), so a line shift elsewhere in the file does not silently re-trigger
an already-tracked entry, and does not silently un-flag a genuinely edited
line either (the hash changes with the text). `ci/doc_number_allowlist_
classification.md` marks every seeded entry real/noise by hand, with the
noise rate computed directly from that table — not asserted in prose.

The allowlist may only shrink: `--check-allowlist-only-shrinks` fetches
`origin/main` (fails CLOSED — `check=True` — if the fetch itself fails,
never silently proceeding on stale/absent data), requires `origin/main` to
resolve at all (`git rev-parse --verify origin/main`, fails the leg if
not), and only THEN treats "no allowlist file at `origin/main`" as a
legitimate bootstrap (this PR is the one introducing the file) rather than
conflating it with "the ref didn't resolve" — the message states which
case fired.

The bare scan (no flags) is ADVISORY in CI (`continue-on-error: true`) and
reports until this lexical heuristic's precision on main reaches >= 80%
real — currently 33/69 = 47.8% real (36/69 = 52.2% noise; see
`ci/doc_number_allowlist_classification.md`) — while `--self-test` and
`--check-allowlist-only-shrinks` stay REQUIRED.

Run: `python3 ci/scripts/check_doc_numbers_have_producers.py`
Self-test: `python3 ci/scripts/check_doc_numbers_have_producers.py --self-test`
Allowlist-only-shrinks leg (network: fetches `origin/main`):
`python3 ci/scripts/check_doc_numbers_have_producers.py --check-allowlist-only-shrinks`
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = [
    REPO_ROOT / "crates" / "jammi-kernels" / "src",
    REPO_ROOT / "crates" / "jammi-kernels" / "tests",
    REPO_ROOT / "crates" / "jammi-encoders" / "src",
    REPO_ROOT / "crates" / "jammi-lora" / "src",
    REPO_ROOT / "crates" / "jammi-bench" / "src",
]

# Matches the module doc's WHERE section exactly: `.rs` and CUDA source only.
SCAN_EXTENSIONS = {".rs", ".cu", ".cuh"}

ALLOWLIST_PATH = REPO_ROOT / "ci" / "doc_number_allowlist.txt"
CLASSIFICATION_PATH = REPO_ROOT / "ci" / "doc_number_allowlist_classification.md"

# --- trigger words -----------------------------------------------------

ADJACENT_WORD_ALTS = [
    r"cosine",
    r"mismatch(?:es)?",
    r"elements",
    r"ceiling",
    r"bit-match\w*",
    r"diverg\w*",
    r"measure[sd]",  # "measured" (past) or "measures" (present) — round-5 ship fix
    r"observe[sd]",  # "observed" (past) or "observes" (present) — same fix
    r"differ(?:s|ed)?",
    r"disagree\w*",
    r"values",
    r"run",
    r"pod",
    r"a100",
    r"seed\w*",
]
_ADJACENT_RE = re.compile(r"\b(?:" + "|".join(ADJACENT_WORD_ALTS) + r")\b", re.IGNORECASE)

# --- number shapes -----------------------------------------------------

NM_RE = re.compile(r"\b\d{1,7}\s*/\s*\d{1,7}\b")
PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,4})?\s*%")
COSINE_RE = re.compile(r"\b0\.\d{2,6}\b")
VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")

# A crate name (this workspace's own deps that show up in prose as "candle
# 0.11", "candle-core 0.11's") immediately before a bare `X.Y` reads as a
# version mention, not a measurement — `VERSION_RE` above only catches
# 3-component versions (`0.11.0`); this catches the 2-component form this
# codebase actually writes crate versions as. Kept per the round-3 audit's
# "keep the version-string exclusion" — this is the same exclusion, just
# covering the 2-component spelling `VERSION_RE` cannot.
_CRATE_NAME_PREFIX_RE = re.compile(
    r"(?i)\b(?:candle(?:-[a-z]+)*|torch|cudarc|half|rayon|tokio|serde(?:-[a-z]+)*|"
    r"anyhow|thiserror|usearch)\s*$"
)

_N_EQUALS_CHAIN_RE = re.compile(r"(?i)\bn\s*=\s*$")

ASSERT_CALL_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|panic)!\s*\("
)
STRING_LITERAL_RE = re.compile(r'"')

NO_PRODUCER_RE = re.compile(r"\bno-producer:")

_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://!|///|//)")
_FN_LINE_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
_ATTR_LINE_RE = re.compile(r"^\s*#!?\[")

# --- citation grammar ----------------------------------------------------

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_PATH = rf"{_IDENT}(?:::{_IDENT})*"

_BRACKET_BACKTICK_TARGET_RE = re.compile(rf"^\[`({_PATH})`\]")
_BACKTICK_TARGET_RE = re.compile(rf"^`({_PATH})`")
_BARE_TARGET_RE = re.compile(rf"^({_PATH})")

_SEE_KW_RE = re.compile(r"\bsee\s+")
_PRINTED_BY_KW_RE = re.compile(r"\bprinted by\s+")
_MEASURED_BY_KW_RE = re.compile(r"\bmeasured by\s+")
_PRODUCER_TAG_RE = re.compile(r"\[producer:\s*([^\]]+)\]")
_ARTIFACT_PATH_RE = re.compile(r"^`?([^\s`,;]+)")

# Round-4 audit fix: a BARE rustdoc intra-link — `` [`fn`] `` with no `see`/
# `printed by` prefix at all — counts as a citation, but ONLY for the
# bracketed-link form (never a bare/backtick-only path, which still
# requires an explicit `see`/`printed by` keyword — this repo writes far
# too many non-citation `` [`Type::method`] `` cross-references for a bare
# link alone to mean "this is where the number came from"; the bracket
# form is reserved for exactly that ambiguity because a doc-comment
# convention this repo already follows is to link the covering `#[test]`
# fn directly, e.g. `` (fixture from [`some_test_fn`]) `` with no lead-in
# word at all).
_BARE_BRACKET_LINK_RE = re.compile(rf"\[`({_PATH})`\]")


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


def build_test_fn_index() -> set[str]:
    """Every `fn <name>` in a tracked `.rs` file that is either
    `#[test]`-attributed (any attribute whose text contains "test",
    case-insensitively — covers `#[test]`, `#[tokio::test]`,
    `#[test_log::test]`, ...; walks backward through however many stacked
    attribute lines sit directly above the `fn` line) or a `pub fn`
    declared in a file under a `tests/` directory.

    Round-3 audit fix: the round-2 index accepted ANY `fn` with a matching
    name, which is what let `` see [`below`] ``/`` see below `` resolve
    against a real, but wholly unrelated, non-test `fn below` — a citation
    is only real evidence if it points at something that actually
    PRODUCED a result (a test), not merely at some function that happens
    to share a word with ordinary English prose ("below", "table", "step"
    are all plausible fn names somewhere in a repo this size).
    """
    names: set[str] = set()
    for rel in _run(["git", "ls-files", "--", "*.rs"]).splitlines():
        path = REPO_ROOT / rel
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        under_tests_dir = f"/{rel}".find("/tests/") != -1 or rel.startswith("tests/")
        for i, line in enumerate(lines):
            m = _FN_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            is_pub = bool(re.match(r"^\s*pub\b", line))
            has_test_attr = False
            j = i - 1
            while j >= 0 and _ATTR_LINE_RE.match(lines[j]):
                if "test" in lines[j].lower():
                    has_test_attr = True
                j -= 1
            if has_test_attr or (is_pub and under_tests_dir):
                names.add(name)
    return names


def is_test_fn(name: str, fn_index: set[str]) -> bool:
    return name in fn_index


def strip_comment_prefix(line: str) -> str:
    return _COMMENT_PREFIX_RE.sub("", line, count=1).lstrip()


def _citation_target(text_after_keyword: str) -> str | None:
    """`text_after_keyword` is everything right after `see `/`printed by `
    in a joined, comment-prefix-stripped block string. Returns the full
    matched `::`-path (e.g. `ops::softmax::MAX_LAST_DIM`), trying the
    rustdoc-link form first, then a bare backtick-wrapped form, then a
    fully bare form — or `None` if none match.
    """
    for pattern in (_BRACKET_BACKTICK_TARGET_RE, _BACKTICK_TARGET_RE, _BARE_TARGET_RE):
        m = pattern.match(text_after_keyword)
        if m:
            return m.group(1)
    return None


def _last_segment(path: str) -> str:
    return path.rsplit("::", 1)[-1]


def is_block_bound(block_text: str, fn_index: set[str], tracked: set[str]) -> bool:
    """`block_text` is the WHOLE comment block's text, comment prefixes
    stripped and lines joined by a single space (so a `see`/`printed by`
    keyword at one line's end and its target on the next line resolve
    exactly like an unwrapped one-line citation would). True iff ANY
    citation anywhere in it resolves.
    """
    for kw_re in (_SEE_KW_RE, _PRINTED_BY_KW_RE):
        for m in kw_re.finditer(block_text):
            target = _citation_target(block_text[m.end() :])
            if target and is_test_fn(_last_segment(target), fn_index):
                return True

    # A BARE `` [`fn`] `` rustdoc link, with no `see`/`printed by` lead-in —
    # accepted ONLY for this bracketed form (see `_BARE_BRACKET_LINK_RE`'s
    # own comment for why the bare/backtick-only forms still require the
    # keyword).
    for m in _BARE_BRACKET_LINK_RE.finditer(block_text):
        if is_test_fn(_last_segment(m.group(1)), fn_index):
            return True

    for m in _MEASURED_BY_KW_RE.finditer(block_text):
        am = _ARTIFACT_PATH_RE.match(block_text[m.end() :])
        if am:
            candidate = am.group(1).strip("`").rstrip(".,;")
            if candidate in tracked:
                return True

    for m in _PRODUCER_TAG_RE.finditer(block_text):
        raw = m.group(1).strip()
        target = _citation_target(raw)
        if target and is_test_fn(_last_segment(target), fn_index):
            return True
        candidate = raw.strip("`").rstrip(".,;")
        if candidate in tracked:
            return True

    return False


# --- scanning ------------------------------------------------------------


@dataclass(frozen=True)
class NumberHit:
    text: str  # the matched number's literal text
    kind: str  # "N/M" | "%" | "cosine"


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def is_scan_line(line: str) -> bool:
    """A "doc comment / // block" line, OR a line invoking one of the
    assert-family macros with a string-literal message on it (single-line
    only — see module doc's WHERE section for the multi-line limitation).
    """
    stripped = line.lstrip()
    if stripped.startswith("//"):
        return True
    if ASSERT_CALL_RE.search(line) and STRING_LITERAL_RE.search(line):
        return True
    return False


def is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("//")


# A comment line with NOTHING after the `//`/`///`/`//!` marker (only
# trailing whitespace) — this repo's own paragraph-break convention inside
# a doc comment. Round-4 audit fix: `compute_blocks` originally treated an
# ENTIRE module doc (every `//!` line from the top of a file down to the
# first blank/code line — e.g. all 571 lines of softmax.rs's header) as
# ONE binding unit, so a single resolving citation anywhere in that doc
# laundered every number in it; the auditor proved this by injecting a
# phantom, uncited `5145/16384`/`31.4%`/`0.796123` into the MIDDLE of
# softmax.rs's module doc and the gate stayed green. A bare separator line
# is still a comment line (`is_comment_line` stays true for it — it must
# remain part of the scanned text, never silently dropped), but it now
# ENDS the current binding block the same way a blank/code line would, so
# a citation in one paragraph of a module doc no longer reaches a number
# several paragraphs away.
_BARE_SEPARATOR_RE = re.compile(r"^\s*(?://!|///|//)\s*$")


def is_bare_separator_line(line: str) -> bool:
    return bool(_BARE_SEPARATOR_RE.match(line))


def compute_blocks(lines: list[str]) -> list[int]:
    """`block_of[i]` is the block id line `i` belongs to. Contiguous
    non-separator comment lines (any run of lines starting with `//`,
    covering `///`, `//!`, and plain `//`) share one id; a bare
    `//`/`///`/`//!` separator line, code (including an `assert!`/
    `panic!` line), or a blank line each ENDS the current block and starts
    its own singleton one.
    """
    block_of = [0] * len(lines)
    next_id = 0
    current: int | None = None
    for i, line in enumerate(lines):
        if is_comment_line(line) and not is_bare_separator_line(line):
            if current is None:
                current = next_id
                next_id += 1
            block_of[i] = current
        else:
            block_of[i] = next_id
            next_id += 1
            current = None
    return block_of


def find_number_hits(line: str) -> list[NumberHit]:
    """Measurement-shaped numbers on `line`, passed through the shape
    exclusions the module doc's WHAT COUNTS AS section documents. Trigger-
    word and binding checks are NOT done here — both are BLOCK-scoped (see
    `scan_lines`), not line-scoped.
    """
    version_spans = [m.span() for m in VERSION_RE.finditer(line)]

    def in_version(span: tuple[int, int]) -> bool:
        return any(_overlaps(span, v) for v in version_spans)

    def is_versioned(start: int) -> bool:
        tail = line[:start].rstrip().rstrip("`").rstrip()
        return bool(_CRATE_NAME_PREFIX_RE.search(tail))

    hits: list[NumberHit] = []

    for m in NM_RE.finditer(line):
        span = m.span()
        start, end = span
        if in_version(span) or is_versioned(start):
            continue
        text = m.group(0)
        _, denom_str = re.split(r"\s*/\s*", text)
        try:
            denom = int(denom_str)
        except ValueError:
            denom = None
        if denom is None or denom < 64:
            continue
        prev_char = line[start - 1] if start > 0 else ""
        if prev_char in (".", "/", "^"):
            continue
        if end < len(line) and line[end] == "/":
            continue
        if _N_EQUALS_CHAIN_RE.search(line[:start].rstrip()):
            continue
        hits.append(NumberHit(text, "N/M"))

    pct_spans: list[tuple[int, int]] = []
    for m in PCT_RE.finditer(line):
        span = m.span()
        if in_version(span) or is_versioned(span[0]):
            continue
        pct_spans.append(span)
        hits.append(NumberHit(m.group(0), "%"))

    for m in COSINE_RE.finditer(line):
        span = m.span()
        if in_version(span) or is_versioned(span[0]):
            continue
        # One hit per token: a percentage's own fractional digits (e.g. the
        # `0.52` inside `0.52%`) are already claimed above as the `%` hit —
        # don't also flag the identical span as a separate bare cosine.
        if any(_overlaps(span, p) for p in pct_spans):
            continue
        hits.append(NumberHit(m.group(0), "cosine"))

    return hits


# --- KO-4: floor terms (round-6, kernel-acceptance-oracle standard) -----
#
# A FLOOR is a literal numeric constant added to a bound to keep a
# discriminating assertion from charging a near-zero-magnitude element the
# full relative tolerance (§3.8 of the cuda-kernel-guide). Two shapes, both
# code (never a doc-comment number the shapes above already catch):
#
#   (B) a `const`/`let` declaration whose name ends `_floor`, case-
#       insensitive — covers `_FLOOR`, `_ABS_FLOOR` (both END in `_floor`),
#       and the lowercase `abs_floor` local-variable idiom this repo's own
#       oracles use (`abs_floor` itself ends in `_floor`), all with ONE
#       suffix check.
#   (A) a `.max(<numeric literal>)` or `+ <numeric literal>` (the latter
#       only on a line that ALSO carries a comparison operator, `<=`/`>=` —
#       otherwise too many ordinary arithmetic `+`s would fire) inside a fn
#       whose SIGNATURE (name or return type) reads as comparison-feeding:
#       returns `bool`, or its name contains `bound`/`close`/`tol`/`floor`/
#       `threshold`/`within`. Conservative on purpose — this shape's
#       precision is unproven at scale, so it rides the SAME advisory bare-
#       scan leg every other shape here does; it never gates on its own.
#
# CITATION SCOPE for a floor line is narrower than a doc-comment number's:
# a `const`/`let` line is CODE, so `compute_blocks` gives it its own
# singleton block (code lines never merge with a preceding comment run) —
# there is no comment block "the floor sits inside" the way a `///` number
# does. The citation text checked is that singleton line's own text PLUS,
# only when the line immediately above is itself a real (non-separator)
# comment line, that comment block's text — i.e. a doc comment DIRECTLY
# attached to the declaration (rustdoc's own attachment convention) or an
# inline trailing `// see fn` on the declaration line itself. A citation
# one comment-block further up (separated by so much as one other code
# line) does NOT reach a floor two lines below it — exactly why
# `GROSS_REGRESSION_FLOOR` (preceded by another `const` line, not a
# comment) is uncited even though a `see`-citable comment sits three lines
# above the *other* const beside it.
FLOOR_DECL_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|let(?:\s+mut)?)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*(?P<rest>.+?);?\s*$"
)
FLOOR_NAME_SUFFIX_RE = re.compile(r"(?i)_floor$")
FLOOR_LITERAL_RE = re.compile(r"-?\d[\d_]*(?:\.\d+)?(?:[eE]-?\d+)?(?:f32|f64)?")

# Round-6 audit fix: the ORIGINAL category (A) design keyed comparison-
# feeding-ness off the ENCLOSING fn's OWN name/return type, via a
# single-line-only `FN_SIG_RE` — both were real gaps found against
# `cuda_parity.rs`'s actual `.max(1.0)` floor sites: (i) `assert_ln_parity_
# bf16`/`assert_softmax_scale_parity_bf16` (lines 476/1515) span their
# signature across multiple lines, so the single-line anchor never matched
# at all and every floor inside those fns was invisible; (ii) the floor
# usually lives in a LOCAL BINDING (`let bf16_bound = |c, g| ... .max(1.0);`)
# whose enclosing fn is a generic `assert_*_parity_*` helper with no
# "bound"/"tol"/"floor" in ITS OWN name — the comparison-feeding-ness lives
# on the LOCAL NAME (`bf16_bound`), not the fn. The redesign below fixes
# both: fn boundaries are found via brace-balanced extraction over the
# WHOLE FILE (any signature shape, any number of lines — mirrors
# `check_kernel_oracles.py`'s `find_fns`, kept as an independent copy per
# this repo's no-cross-script-import gate convention), and comparison-
# feeding-ness is decided per STATEMENT: a `let <name> = <rhs>;` whose RHS
# carries a floor literal is a hit iff `<name>` is later used, in the SAME
# fn body, inside an `assert!`-family macro call or a `<=`/`>=`-bearing
# statement (the LOCAL-BINDING rule); a floor literal with NO enclosing
# `let` is a hit iff ITS OWN statement already satisfies that same
# assert/comparison test (the INLINE rule, e.g. `magnitude.max(1e-12)`
# used directly inside `... <= REL`).
FN_HEAD_RE_KO4 = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
LET_BINDING_RE = re.compile(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=")
ASSERT_MACRO_RE_KO4 = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne)!\s*\("
)
FLOOR_MAX_RE = re.compile(
    r"\.max\(\s*(-?\d[\d_]*(?:\.\d+)?(?:[eE]-?\d+)?(?:f32|f64)?)\s*\)"
)
FLOOR_ADD_RE = re.compile(r"\+\s*(-?\d[\d_]*\.\d+(?:[eE]-?\d+)?(?:f32|f64)?)\b")
_COMPARISON_OP_RE = re.compile(r"<=|>=")


def _blank_comments_and_strings_ko4(text: str) -> str:
    """Length-preserving `//`-comment and `"..."`-string-content blanker,
    used ONLY to find real `fn` keyword POSITIONS (never to build the body
    text itself, which is still extracted from RAW `text` at those verified
    positions) — a `fn foo(...)  { ... }` shape merely quoted in a comment
    or a string literal (e.g. an example in a doc comment) must not be
    discovered as a real function. Block comments and raw strings are not
    this repo's kernel-test convention and are left unhandled; imprecision
    here only risks UNDER-stripping, which fails toward scanning MORE text
    on this advisory-leg shape, never less.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                in_string = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if text[i : i + 2] == "//":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_fn_body_ko4(text: str, fn_kw_end: int) -> tuple[str, int, int]:
    brace_start = text.find("{", fn_kw_end)
    if brace_start == -1:
        return "", -1, -1
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1], brace_start, i
    return text[brace_start:], brace_start, len(text) - 1


def _find_stmt_terminator(text: str, start: int) -> int:
    """Index just PAST the depth-balanced `;` that terminates the `let`
    statement beginning at `start` (right after its own `let name =`) — a
    `let` is always `;`-terminated in valid Rust, regardless of what
    surrounds it (a `for`/`if`/`match` block with NO trailing `;` of its
    own, which is why a general "split the whole fn body into top-level
    `;`-terminated statements" pass is the WRONG tool here: a block
    expression used as a statement needs no `;`, so naive `;`-splitting
    silently merges everything after it into one giant tail statement —
    the round-6 bug this function replaces. Scanning forward from the
    KNOWN start of a specific `let`'s RHS sidesteps that entirely: it never
    needs to know where any OTHER statement begins or ends.
    """
    depth = 0
    i, n = start, len(text)
    while i < n:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i  # closing bracket of an ENCLOSING scope — bail
            depth -= 1
        elif c == ";" and depth == 0:
            return i + 1
        i += 1
    return n


def _extract_paren_balanced_ko4(text: str, open_paren_idx: int) -> str:
    depth = 0
    for i in range(open_paren_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx : i + 1]
    return text[open_paren_idx:]


def _text_has_assert_or_comparison_usage(text: str, name_re: re.Pattern) -> bool:
    for m in ASSERT_MACRO_RE_KO4.finditer(text):
        args = _extract_paren_balanced_ko4(text, m.end() - 1)
        if name_re.search(args):
            return True
    for stmt in re.split(r"[;{}]", text):
        if name_re.search(stmt) and _COMPARISON_OP_RE.search(stmt):
            return True
    return False


def _first_floor_literal(text: str) -> str | None:
    m = FLOOR_MAX_RE.search(text)
    if m:
        return m.group(1)
    m = FLOOR_ADD_RE.search(text)
    if m:
        return m.group(1)
    return None


@dataclass(frozen=True)
class FloorHit:
    line_idx: int  # 0-indexed
    text: str
    # For a category (A) hit: the 0-indexed line of the ENCLOSING fn's own
    # `fn` signature — its citation home is that fn's attached doc comment
    # (directly above the `fn` line), not necessarily the call-site line
    # itself, which usually sits deep in the fn body with no comment
    # immediately above it. `None` for a category (B) declaration hit (whose
    # citation home is its OWN directly-attached line, per `scan_lines`).
    fn_sig_line_idx: int | None = None


def find_floor_hits(lines: list[str]) -> list[FloorHit]:
    """Category (B) declaration scan (whole-file, name-suffix driven) plus
    category (A) fn-scoped `.max(<num>)`/`+ <num>` scan — fn boundaries via
    brace-balanced extraction over the WHOLE FILE (any signature shape,
    multi-line included). Within each fn body:

      - LOCAL-BINDING rule: for every `let <name> = <rhs>;` (found by
        `re.finditer`, its OWN terminating `;` found by scanning FORWARD
        from that specific match via `_find_stmt_terminator` — never by
        splitting the whole fn body into top-level statements first, which
        silently merges everything after a semicolon-less block expression
        like `for {...}`/`if {...}` into one giant tail statement; that WAS
        this function's round-6 bug, reproduced and fixed in the same
        round), a floor literal in the RHS hits iff `<name>` is later used,
        anywhere in the REST of the fn body, inside an `assert!`-family
        call or a `<=`/`>=`-bearing statement.
      - INLINE rule: a floor literal with no owning `let` (not already
        claimed by the rule above) hits iff it sits inside an `assert!`-
        family call's own argument list, OR on a line that also carries a
        `<=`/`>=` comparison operator.
    """
    hits: list[FloorHit] = []

    # (B) — declaration name ends `_floor`.
    for i, line in enumerate(lines):
        m = FLOOR_DECL_RE.match(line)
        if not m:
            continue
        if not FLOOR_NAME_SUFFIX_RE.search(m.group("name")):
            continue
        lit = FLOOR_LITERAL_RE.search(m.group("rest"))
        if lit:
            hits.append(FloorHit(i, lit.group(0)))

    # (A) — local-binding / inline, fn-scoped.
    text = "\n".join(lines)
    stripped_for_fn_detection = _blank_comments_and_strings_ko4(text)
    for fm in FN_HEAD_RE_KO4.finditer(stripped_for_fn_detection):
        body, body_start, _body_end = _extract_fn_body_ko4(text, fm.end())
        if not body:
            continue
        fn_sig_line_idx = text.count("\n", 0, fm.start())

        assert_spans: list[tuple[int, int]] = []
        for am in ASSERT_MACRO_RE_KO4.finditer(body):
            args = _extract_paren_balanced_ko4(body, am.end() - 1)
            assert_spans.append((am.end() - 1, am.end() - 1 + len(args)))

        def _same_line(pos: int, _body: str = body) -> str:
            ls = _body.rfind("\n", 0, pos) + 1
            le = _body.find("\n", pos)
            if le == -1:
                le = len(_body)
            return _body[ls:le]

        claimed_positions: set[int] = set()

        # LOCAL-BINDING rule.
        for let_m in LET_BINDING_RE.finditer(body):
            name = let_m.group(1)
            rhs_start = let_m.end()
            rhs_end = _find_stmt_terminator(body, rhs_start)
            rhs = body[rhs_start:rhs_end]
            lit_m = FLOOR_MAX_RE.search(rhs) or FLOOR_ADD_RE.search(rhs)
            if lit_m is None:
                continue
            name_re = re.compile(rf"\b{re.escape(name)}\b")
            later_text = body[rhs_end:]
            if not _text_has_assert_or_comparison_usage(later_text, name_re):
                continue
            abs_pos = body_start + let_m.start()
            line_idx = text.count("\n", 0, abs_pos)
            hits.append(FloorHit(line_idx, lit_m.group(1), fn_sig_line_idx))
            claimed_positions.add(rhs_start + lit_m.start())

        # INLINE rule — never re-claims a position the LOCAL-BINDING rule
        # already reported (its own `let` RHS naturally also matches this
        # loop's regexes; double-counting the SAME literal twice is a
        # duplicate finding, not a second genuine instance).
        for fre in (FLOOR_MAX_RE, FLOOR_ADD_RE):
            for m in fre.finditer(body):
                pos = m.start()
                if pos in claimed_positions:
                    continue
                in_assert = any(a <= pos < b for a, b in assert_spans)
                if not (in_assert or _COMPARISON_OP_RE.search(_same_line(pos))):
                    continue
                abs_pos = body_start + pos
                line_idx = text.count("\n", 0, abs_pos)
                hits.append(FloorHit(line_idx, m.group(1), fn_sig_line_idx))

    return hits


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    number: str
    kind: str
    line_text: str  # the raw text of the number's own line, for allowlist hashing


def scan_lines(
    lines: list[str], file_label: str, fn_index: set[str], tracked: set[str]
) -> list[Finding]:
    """`lines` is 0-indexed; line numbers reported are 1-indexed."""
    n = len(lines)
    scan_mask = [is_scan_line(l) for l in lines]
    block_of = compute_blocks(lines)

    block_line_indices: dict[int, list[int]] = {}
    for i in range(n):
        block_line_indices.setdefault(block_of[i], []).append(i)

    trigger_text_cache: dict[int, str] = {}
    clean_text_cache: dict[int, str] = {}

    def trigger_text(bid: int) -> str:
        if bid not in trigger_text_cache:
            trigger_text_cache[bid] = "\n".join(lines[j] for j in block_line_indices[bid])
        return trigger_text_cache[bid]

    def clean_text(bid: int) -> str:
        if bid not in clean_text_cache:
            clean_text_cache[bid] = " ".join(
                strip_comment_prefix(lines[j]) for j in block_line_indices[bid]
            )
        return clean_text_cache[bid]

    findings: list[Finding] = []
    for i, line in enumerate(lines):
        if not scan_mask[i]:
            continue
        hits = find_number_hits(line)
        if not hits:
            continue
        bid = block_of[i]
        if not _ADJACENT_RE.search(trigger_text(bid)):
            continue
        if NO_PRODUCER_RE.search(line):
            continue
        if is_block_bound(clean_text(bid), fn_index, tracked):
            continue
        for hit in hits:
            findings.append(Finding(file_label, i + 1, hit.text, hit.kind, line))

    # KO-4 — floor terms. Not gated by `_ADJACENT_RE` (the floor SHAPE
    # itself is the trigger, unlike a bare number in prose); citation scope
    # is the floor's own singleton (code) line plus a directly-attached
    # preceding comment block, per `find_floor_hits`'s own doc.
    def attached_comment_text(above_idx: int) -> str:
        """The comment block directly above `lines[above_idx]`, if any (empty
        string otherwise) — a citation must be DIRECTLY attached (no
        intervening code line), never reaching across one.
        """
        j = above_idx - 1
        if j >= 0 and is_comment_line(lines[j]) and not is_bare_separator_line(lines[j]):
            return clean_text(block_of[j])
        return ""

    for fh in find_floor_hits(lines):
        i = fh.line_idx
        if NO_PRODUCER_RE.search(lines[i]):
            continue
        own_text = clean_text(block_of[i])
        parts = [own_text, attached_comment_text(i)]
        if fh.fn_sig_line_idx is not None:
            # category (A): also check the ENCLOSING fn's own attached doc
            # comment (directly above its `fn` line) — the natural citation
            # home for a floor buried in the fn body, not the call-site
            # line itself.
            parts.append(attached_comment_text(fh.fn_sig_line_idx))
        combined = " ".join(p for p in parts if p)
        if is_block_bound(combined, fn_index, tracked):
            continue
        findings.append(Finding(file_label, i + 1, fh.text, "floor", lines[i]))

    return findings


def scan_tree() -> list[Finding]:
    fn_index = build_test_fn_index()
    tracked = tracked_files()
    findings: list[Finding] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            raise SystemExit(f"check_doc_numbers_have_producers: scan root not found: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_EXTENSIONS:
                continue
            text = path.read_text(errors="ignore")
            rel = str(path.relative_to(REPO_ROOT))
            findings.extend(scan_lines(text.splitlines(), rel, fn_index, tracked))
    return findings


# --- allowlist -------------------------------------------------------------


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def line_hash(text: str) -> str:
    return hashlib.sha1(normalize_line(text).encode("utf-8")).hexdigest()


def parse_allowlist_text(text: str) -> set[tuple[str, str, str]]:
    """`file:number:sha1` triples — `#`-comments and blank lines ignored.
    `number` may itself contain no `:` (the shapes this gate matches never
    do), so a 3-way `split(":", 2)` is exact.
    """
    entries: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(":", 2)
        if len(parts) != 3:
            continue
        file_part, number_part, sha_part = parts
        entries.add((file_part, number_part, sha_part))
    return entries


def load_allowlist() -> set[tuple[str, str, str]]:
    if not ALLOWLIST_PATH.exists():
        return set()
    return parse_allowlist_text(ALLOWLIST_PATH.read_text())


def report(findings: list[Finding]) -> int:
    allowlist = load_allowlist()
    unallowlisted = [
        f for f in findings if (f.file, f.number, line_hash(f.line_text)) not in allowlist
    ]

    if not unallowlisted:
        print(
            "doc-numbers-have-producers: OK — "
            f"{len(findings)} measurement-shaped number(s) found, "
            f"all cited or allowlisted."
        )
        return 0

    print("doc-numbers-have-producers: FAIL", file=sys.stderr)
    for f in unallowlisted:
        print(
            f"  {f.file}:{f.line_no}: {f.kind} `{f.number}` has no producer citation "
            "resolvable anywhere in its comment block (see [`fn`] / see mod::fn / "
            "printed by <same> / measured by <path> / [producer: <same>]), and no "
            "`no-producer:` opt-out on its own line.",
            file=sys.stderr,
        )
    print(
        "\ndoc-numbers-have-producers: a measurement-shaped number in a kernel/test doc "
        "comment or assert! message claims an observed run with nothing in the tree that "
        "produces it. Cite the real producer anywhere in the same comment block (`see "
        "[`fn`]`, `see mod::fn`, `printed by <same>`, `measured by <path>`, or `[producer: "
        "<same>]`, where the resolved fn must be a #[test] function or a pub fn under "
        "tests/), or tag the number `no-producer: <reason>` on its own line if it is "
        f"genuinely derived — never add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)}, "
        "which only ever shrinks.",
        file=sys.stderr,
    )
    return 1


def check_allowlist_only_shrinks() -> int:
    """The allowlist tracks PRE-EXISTING debt only and may only shrink.

    Round-3 audit fix: the round-2 version used `check=False` on `git
    fetch` and a single `returncode == 0` test on `git show`, so a FAILED
    fetch (network flake, `origin` misconfigured) and a genuinely ABSENT
    file on `origin/main` were indistinguishable — both silently took the
    "nothing to compare against, pass" branch, meaning a fetch failure
    made this leg fail OPEN instead of closed. Now: `git fetch` raises via
    `check=True`; `git rev-parse --verify origin/main` must succeed (a
    real ref) or the leg FAILS outright with that fact stated; only once
    the ref is confirmed to resolve is "the allowlist file itself doesn't
    exist there" checked via `git cat-file -e`, and ONLY THAT case is
    treated as the legitimate bootstrap (this PR is the one introducing
    the file) — the message states which case fired.
    """
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(
            f"  git fetch origin main failed (exit {exc.returncode}): "
            f"{exc.stderr.strip() if exc.stderr else '<no stderr>'}",
            file=sys.stderr,
        )
        print(
            "\ndoc-number-allowlist-only-shrinks: cannot compare against origin/main "
            "because the fetch itself failed — this fails CLOSED (a network/config "
            "problem is not license to skip the only-shrinks check), not open.",
            file=sys.stderr,
        )
        return 1

    rev = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if rev.returncode != 0:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(
            "  origin/main does not resolve after a successful fetch — "
            f"`git rev-parse --verify origin/main` exit {rev.returncode}: "
            f"{rev.stderr.strip()}",
            file=sys.stderr,
        )
        print(
            "\ndoc-number-allowlist-only-shrinks: this is NOT the bootstrap case (that "
            "requires origin/main to resolve, just without the allowlist file) — it is an "
            "unresolvable ref, which fails CLOSED.",
            file=sys.stderr,
        )
        return 1

    file_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", "origin/main:ci/doc_number_allowlist.txt"],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    current = load_allowlist()

    if not file_exists:
        print(
            "doc-number-allowlist-only-shrinks: OK (bootstrap) — origin/main resolves "
            f"({rev.stdout.strip()}) but has no {ALLOWLIST_PATH.relative_to(REPO_ROOT)} "
            f"yet: this branch's {len(current)} entries establish the baseline."
        )
        return 0

    base_text = _run(["git", "show", "origin/main:ci/doc_number_allowlist.txt"])
    base = parse_allowlist_text(base_text)
    added = current - base

    if added:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for entry in sorted(added):
            print(f"  + {':'.join(entry)}", file=sys.stderr)
        print(
            "\ndoc-number-allowlist-only-shrinks: this branch adds a NEW entry to "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}. The allowlist tracks pre-existing "
            "debt from before this gate landed and may never grow. Do not add an entry "
            "here — cite the real producer instead anywhere in the number's comment block "
            "(`see [`fn`]`, `see mod::fn`, `printed by <same>`, `measured by <path>`, or "
            "`[producer: <same>]`), or tag the number `no-producer: <reason>` on its own "
            "line if it is genuinely derived, not measured.",
            file=sys.stderr,
        )
        return 1

    print(
        f"doc-number-allowlist-only-shrinks: OK — {len(current)} entries "
        f"({len(base) - len(current)} shrunk vs origin/main)."
    )
    return 0


# --- self-test ---------------------------------------------------------


def _pick_real_test_fn(fn_index: set[str]) -> str:
    """A REAL `#[test]`/pub-fn-under-tests name, picked from the live
    tree at self-test time — never hardcoded, so this fixture never rots
    as the repo's own test names change. Deterministic (sorted) so the
    self-test's own output is stable across runs.
    """
    if not fn_index:
        raise SystemExit(
            "check_doc_numbers_have_producers self-test: build_test_fn_index() found "
            "ZERO test functions in the tracked tree — cannot build the positive-"
            "resolution fixture. This means either the tree has no #[test] fns at all "
            "(implausible) or build_test_fn_index() itself is broken."
        )
    return sorted(fn_index)[0]


def _scan_fragment(text: str, label: str, fn_index: set[str], tracked: set[str]) -> list[Finding]:
    return scan_lines(text.strip("\n").splitlines(), label, fn_index, tracked)


def self_test() -> int:
    failures: list[str] = []
    fn_index = build_test_fn_index()
    tracked = tracked_files()
    real_fn = _pick_real_test_fn(fn_index)

    def flagged(name: str, text: str) -> None:
        hits = _scan_fragment(text, f"<self-test: {name}>", fn_index, tracked)
        if not hits:
            failures.append(f"self-test FAILED ({name}): expected a finding, got none")

    def clean(name: str, text: str) -> None:
        hits = _scan_fragment(text, f"<self-test: {name}>", fn_index, tracked)
        if hits:
            failures.append(f"self-test FAILED ({name}): expected no finding, got {hits}")

    # --- (A) citation grammar: positive forms, resolved via a REAL test fn ---
    clean(
        "see [`fn`] rustdoc-link form",
        f"""
// measured 5145/16384 elements differed from the eager chain — see [`{real_fn}`]
// for the run that produced this count.
""",
    )
    clean(
        "see mod::path::fn form, resolved by last segment",
        f"""
// measured 5145/16384 elements differed from the eager chain — see
// some::nested::module::{real_fn} for the run that produced this count.
""",
    )
    clean(
        "printed by `fn` form",
        f"""
// 100% (6144/6144) elements bit-match — printed by `{real_fn}` for the run
// that produced this count.
""",
    )
    clean(
        "[producer: mod::fn] tag form",
        f"""
// worst observed 0.796123 cosine mismatch [producer: some::mod::{real_fn}] on
// this fixture.
""",
    )
    clean(
        "measured by <tracked path>",
        """
// 100% (6144/6144) elements bit-match — measured by Cargo.toml, a real
// tracked path (used here only as a stand-in artifact).
""",
    )
    clean(
        "round-4: a BARE [`fn`] rustdoc link, no see/printed-by lead-in at "
        "all, resolves for the bracketed form (the esc-045 branch's "
        "reference_call_path_matches_torch_dx_on_torchs_own_f32_y_from_"
        "fixture shape: a citation embedded as a plain doc link, not "
        "introduced by a keyword)",
        f"""
// (fixture from [`{real_fn}`]) the fused arm measured 5145/16384 elements
// differing from the eager chain on that same fixture.
""",
    )
    flagged(
        "round-4: a BARE backtick-only (non-bracketed) path with NO "
        "see/printed-by lead-in does NOT resolve — the keyword-optional "
        "carve-out is for the bracketed [`fn`] link form only",
        f"""
// (fixture from `{real_fn}`) the fused arm measured 5145/16384 elements
// differing from the eager chain on that same fixture.
""",
    )

    # --- (A) citation grammar: must NOT resolve, even if a real fn exists ---
    flagged(
        "see below must not resolve (fn below is not a #[test]/pub-under-tests fn)",
        """
// measured 87/128 elements differed from the eager chain — see below for
// the run that produced this count.
fn below() {}
""",
    )
    flagged(
        "see table 3 must not resolve (names no real identifier)",
        """
// measured 87/128 elements differed from the eager chain — see table 3
// for the breakdown.
""",
    )
    flagged(
        "see step 2 must not resolve (names no real identifier)",
        """
// measured 87/128 elements differed from the eager chain — see step 2 for
// the setup.
""",
    )

    # --- (B) binding unit is the whole block ---
    clean(
        "citation one sentence earlier in the same block covers a later number",
        f"""
// The bound is verified directly against a real run — see [`{real_fn}`] for
// the harness. Over that run the worst divergence measured is 0.512345 at
// the fixture's peak magnitude, comfortably inside tolerance.
""",
    )
    flagged(
        "a blank-line-separated (different) block is NOT covered by an earlier block's citation",
        f"""
// see [`{real_fn}`] for the harness this whole module relies on.

fn unrelated_marker() {{}}

// measured 87/128 elements differed from the eager chain in a totally
// separate block with no citation of its own.
""",
    )
    flagged(
        "round-4 audit fixture: a bare '//!' paragraph separator inside a "
        "module doc breaks the block, so a citation in paragraph 1 does NOT "
        "cover a phantom number injected into paragraph 2 (the exact "
        "reproduction the round-4 audit verified against softmax.rs, where "
        "a single resolving citation anywhere in the ~570-line module doc "
        "was previously laundering every number in it)",
        f"""
//! Fused masked softmax-last-dim forward + backward — see [`{real_fn}`] for
//! the harness the whole module doc's claims are anchored to.
//!
//! Paragraph two, several sentences after that citation: the fused arm
//! measured 5145/16384 elements differing from the eager chain, a 31.4%
//! divergence rate, cosine similarity 0.796123 against the reference —
//! none of this paragraph's own numbers are cited anywhere near it.
""",
    )
    clean(
        "no-producer: on the number's own line",
        """
// bf16 mantissa ceiling is 0.780000 observed — no-producer: derived from
// the IEEE 754 bf16 format, not measured.
""",
    )
    flagged(
        "no-producer: on a DIFFERENT line in the SAME block must still flag (line-scoped, not block-scoped)",
        """
// bf16 mantissa ceiling is 0.780000 observed, on this fixture.
// no-producer: derived from the IEEE 754 bf16 format, not measured.
""",
    )

    # --- KO-4: floor terms (round-6) ---
    clean(
        f"category (B) floor const cited by a DIRECTLY-attached preceding doc comment — see [`{real_fn}`]",
        f"""
/// Derived directly from the fixture's own measured tail — see [`{real_fn}`].
const BF16_ABS_FLOOR: f32 = 0.03125;
""",
    )
    flagged(
        "category (B) floor const with NO directly-attached comment (blocked by an "
        "intervening code line) is uncited — the exact GROSS_REGRESSION_FLOOR "
        "reproduction: a citable comment sits above the OTHER const beside it, not "
        "directly above this one",
        """
// A SANITY backstop — see some_unrelated_test for the real discriminating oracle.
const GROSS_REGRESSION_MULTIPLE: f64 = 8.0;
const GROSS_REGRESSION_FLOOR: f64 = 0.05;
""",
    )
    flagged(
        "category (B) lowercase `abs_floor` local-variable idiom is caught too, and a "
        "self-referential 'cited in this test's doc' is NOT a real citation",
        """
// A small floor covers the f32-only summation-order noise (measured negligible
// by the forward test's own standalone probe, cited in its doc).
let abs_floor = 1e-1f64;
""",
    )
    clean(
        "category (A) `.max(<num>)` floor inside a comparison-feeding (`bound`-named, "
        f"`-> bool`) fn, cited on the fn's own attached doc comment — see [`{real_fn}`]",
        f"""
/// Divide-by-zero guard on the denominator — see [`{real_fn}`].
fn within_some_bound(diff: f64, magnitude: f64) -> bool {{
    diff <= (diff) / magnitude.max(1e-12)
}}
""",
    )
    flagged(
        "category (A) `.max(<num>)` floor inside a comparison-feeding fn with NO "
        "citation at all is flagged",
        """
fn within_some_bound(diff: f64, magnitude: f64) -> bool {
    diff <= (diff) / magnitude.max(1e-12)
}
""",
    )
    flagged(
        "round-6 audit fix: category (A) LOCAL-BINDING rule — a `+ <num>` floor "
        "assigned to a local name (not on the same line as any comparison) still "
        "hits when that name is used in a comparison LATER in the fn body, uncited",
        """
fn within_some_bound(diff: f64, magnitude: f64) -> bool {
    let padded = diff + 0.05;
    padded <= magnitude
}
""",
    )
    clean(
        "category (A) `+ <num>` assigned to a local name that is NEVER used in a "
        "comparison or assert! anywhere in the fn is genuinely ordinary arithmetic "
        "— not a floor hit",
        """
fn compute_something(diff: f64) -> f64 {
    let padded = diff + 0.05;
    padded * 2.0
}
""",
    )

    # --- KO-4's seven motivating instances (round-6 audit item 5) ---------
    # cuda_parity.rs (pre-#386, origin/main e77805f) carries seven uncited
    # `.max(1.0)` floor sites: 227/2269 are the "let ulp = ...; assert!(...
    # <= ulp, ...)" inline-let shape; 510/823/1181/1544/1949 are the "let
    # bf16_bound = |c, g| ... .max(1.0); assert!(... <= bf16_bound(*c, *g),
    # ...)" closure-call shape. Both shapes reproduced here (not the file
    # itself) — the ORIGINAL fn-name-keyed design missed both classes
    # entirely (the enclosing `assert_*_parity_*` helper fns carry none of
    # bound/tol/floor/threshold/within in their OWN names, and two of the
    # seven sit inside multi-line-signature fns the old single-line
    # `FN_SIG_RE` could not even see).
    flagged(
        "KO-4 motivating instance (cuda_parity.rs:227/2269 shape): an inline "
        "`let ulp = ... .max(1.0); assert!(... <= ulp, ...)` inside a fn whose "
        "OWN name carries no bound/tol/floor keyword at all",
        """
fn assert_parity_f32(cuda: &Device, alpha: f64, xv: &[f32], yv: &[f32]) {
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        let ulp = 2.0f32.powi(-7) * c.abs().max(*g).max(1.0);
        assert!(
            (c - g).abs() <= 2.0 * ulp,
            "mismatch at {i}: cpu {c} vs cuda {g}"
        );
    }
}
""",
    )
    flagged(
        "KO-4 motivating instance (cuda_parity.rs:510/823/1181/1544/1949 shape): "
        "a `let bf16_bound = |c, g| ... .max(1.0);` closure whose enclosing fn "
        "signature spans MULTIPLE LINES (the old single-line FN_SIG_RE could not "
        "see this fn at all) and whose OWN name carries no bound/tol/floor "
        "keyword; bf16_bound is called only much later, inside assert!",
        """
fn assert_ln_parity_bf16(
    cuda: &Device,
    eps: f64,
    rows: usize,
    hidden: usize,
    xv: &[f32],
    gv: &[f32],
) {
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "ln bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
}
""",
    )
    flagged(
        "category (B) `no-producer:` opt-out stays LINE-scoped for a floor const too "
        "— on a DIFFERENT line in the same file it does not launder this one",
        """
const SOME_ABS_FLOOR: f64 = 0.25;
// no-producer: this opt-out is written on the wrong line on purpose.
""",
    )

    # --- (C) number shapes ---
    flagged(
        "definitional '=' no longer suppresses a measured value",
        """
// measured on the A100: 3.1e-3 = 0.390000 of the sup; a looser bound let
// a mutation pass.
""",
    )
    flagged(
        "a [`CONST`] doc-link on the line no longer blanket-suppresses it",
        """
// [`SOME_CONST`] aside, the worst observed divergence measured on an
// otherwise-idle box was 0.273456, run three of three.
""",
    )
    flagged(
        "5-6 decimal cosines are now in COSINE_RE's range",
        """
// worst observed cosine mismatch measured on this run: 0.123456.
""",
    )
    clean(
        "N/M with M < 64 stays excluded (index/enumeration, not a mismatch count)",
        """
// measured 29/32 elements differed on this fixture, a small run.
""",
    )
    clean(
        "N/M immediately preceded by '.' stays excluded (decimal-embedded fragment)",
        """
// measured on the A100: swept scaling 1.3/1600 against the fixture and
// observed no divergence worth flagging in this run.
""",
    )
    clean(
        "N/M chain (a/b/c) is fully excluded in both directions",
        """
// measured on a run: n=32/900/1024/6144 elements differed across the
// sweep, a shape enumeration not a single mismatch count.
""",
    )
    clean(
        "N/M immediately preceded by '^' stays excluded (exponent)",
        """
// measured phi(x) = exp(-x^2/6144) on this run, observed to match the
// standard normal density.
""",
    )
    clean(
        "version strings (3- and 2-component) stay excluded",
        """
// candle-kernels-0.11.0 measured this; candle-core 0.11's own run
// observed no divergence worth flagging at all in this fixture.
""",
    )
    flagged(
        "a genuine N/M (M >= 64, no chain/decimal/exponent context) is still caught",
        """
// candle-core 0.11's own run observed no divergence worth flagging, but
// separately measured 128/6144 elements as a genuine finding.
""",
    )

    # --- (F) fn_index has no extra_fns escape hatch in the production path ---
    if is_test_fn("definitely_not_a_real_fn_anywhere", fn_index):
        failures.append(
            "self-test FAILED: is_test_fn resolved a name that isn't in the real fn_index "
            "— there is no extra_fns parameter to explain this; the resolver itself is "
            "broken."
        )

    # --- the eight audit-reworded probes (all MUST be caught) ----------------
    probes = {
        "probe 1: measured X = Y (definitional-= no longer kills it)": """
// The RED drill's fused arm measured on the A100: 67546/262144 elements
// differed = 0.257600 of the total, no producer cited.
""",
        "probe 2: [`CONST`] link no longer blanket-suppresses the line": """
// [`REL_TOL`] context aside, bf16 scored 0.610000 against the f32 truth
// run, versus 0.810000 for the f32-only baseline — neither is cited.
""",
        "probe 3: differ on N/M values, M >= 64": """
// the two arms differ on 176/4096 values in this sweep, uncited.
""",
        "probe 4: percentage measured on a pod": """
// the fused kernel gives a 39% memory reduction measured on that pod,
// with nothing here saying how.
""",
        "probe 5: 5-6 decimal cosine": """
// worst observed cosine mismatch measured on this run: 0.512345, uncited.
""",
        "probe 6: see below (real fn, wrong kind)": """
// measured 512/4096 elements differed on this run — see below for the
// harness.
fn below() {}
""",
        "probe 7: see table 3 (no real identifier)": """
// measured 512/4096 elements differed on this run — see table 3 for the
// per-shape breakdown.
""",
        "probe 8: no-producer one line off in the same block": """
// measured 512/4096 elements differed on this run, uncited on this exact
// line.
// no-producer: this reasoning applies to a DIFFERENT number, not this one.
""",
        "probe 9: present-tense 'measures'/'observes' trigger (round-5 ship "
        "fix — the exact geglu_oracles.rs:218 gap: a narrower block no "
        "longer reaches a DIFFERENT paragraph's past-tense 'measured' by "
        "accident, so the present-tense form must be its own trigger)": """
// the residual relative requirement, maximized over every element in the
// sweep, measures 1.06% and observes no case above it, uncited.
""",
    }
    for name, text in probes.items():
        flagged(name, text)

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("doc-numbers-have-producers self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "doc-numbers-have-producers self-test: OK — citation grammar (rustdoc-link/path/"
        "bare forms, plus a keyword-optional bracketed link) resolves via a REAL #[test] "
        "fn and rejects see-below/table/step; a bare '//!' paragraph separator breaks a "
        "binding block (the softmax.rs module-doc injection reproduction); block-level "
        "binding covers a same-block citation and refuses a different block's; "
        "no-producer: stays line-scoped; every shape exclusion (M>=64, decimal/chain/"
        "exponent/n=/version) holds without over-firing; the definitional-= and "
        "[`CONST`]-link exclusions are gone; all nine reworded probes caught "
        "(including present-tense measures/observes)."
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()

    findings = scan_tree()
    if findings and "--list-by-file" in argv:
        by_file: dict[str, int] = {}
        for f in findings:
            by_file[f.file] = by_file.get(f.file, 0) + 1
        for file, count in sorted(by_file.items()):
            print(f"{file}: {count}")
    return report(findings)


if __name__ == "__main__":
    sys.exit(main())
