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
third leg of.

WHAT COUNTS AS "measurement-shaped": three number shapes, each REQUIRED to
sit within `TRIGGER_WINDOW` lines of one of `ADJACENT_WORDS` — a plain `N/M`,
`X%`, or `0.XXX` floating around in math notation, an enumeration ("cell
3/7"), a domain bound ("rank 1/2/3"), a fixed statistical constant ("a 95%
CI"), or a defined threshold ("DEFAULT_REGRESSION_THRESHOLD, 30%, see the
module docs") is NOT measurement-shaped by this gate — it carries no claim of
an observed run, so it needs no producer. Only a number sitting near a word
that signals "this was seen to happen" is treated as a claimed measurement:

  - `N/M` element/mismatch counts:      `\\d+\\s*/\\s*\\d+`
  - percentages:                        `\\d+(\\.\\d+)?\\s*%`
  - bare cosines (e.g. `0.610`,
    `0.796`):                           `0\\.\\d{2,4}`

Three SHAPE EXCLUSIONS keep the above from firing on plain math/notation that
happens to sit near a trigger word in the same paragraph (see `NM_EXCLUDED`):
a version string (`0.11.0`), an exponent/grouped-division numerator
immediately preceded by `^` or `)` (`x^2/2`), an indeterminate form
(`= 0/0`), an `n=`-prefixed shape enumeration (`n=32/900`), and a `k/N` ratio
whose denominator `N < 16` (almost always an index/enumeration/small ratio,
never a real mismatch-element count in this codebase's own measured spans,
which start in the hundreds). A percentage's own fractional digits
(`0.52%`'s `0.52`) are claimed by the `%` match and never separately
re-flagged as a bare cosine — one hit per token.

WHERE: doc comments (`///`, `//!`) and plain `//` comment blocks, plus the
message string of an `assert!`/`assert_eq!`/`assert_ne!`/`debug_assert!`/
`panic!` call (single-line message only — a multi-line `format!` message
inside the macro call is out of scope for this pass, the same "cheap,
visible, first-pass tripwire" cost `check_sqlite_isms.py`'s own doc accepts),
under `crates/jammi-kernels/{src,tests}`, `crates/jammi-encoders/src`,
`crates/jammi-lora/src`, `crates/jammi-bench/src` — `.rs` and CUDA source
(`.cu`/`.cuh`) files, since the class this closes was found in a `.cu` file
(`adamw_step.cu`) as often as a `.rs` one.

PRODUCER CITATION — TIGHTLY BOUND to the flagged number, never a wide
line-window (a prior version of this gate accepted any citation within 12
lines of the number; a two-line shift in an unrelated edit re-triggered an
already-cited number on a live branch — see `is_bound`). A number at line L
is bound iff ONE of:

  - a `see <fn>` / `printed by <fn>` / `measured by <path>` citation sits on
    line L itself, with `<fn>` resolving to a real `fn <fn>` in some tracked
    `.rs` file (see `FN_INDEX`) or `<path>` a `git ls-files`-tracked path.
  - line L (rstripped of trailing punctuation) ENDS with `see` / `printed
    by` / `measured by` — a doc-comment line wrap — and the immediately
    FOLLOWING line supplies the `<fn>`/`<path>` (the wrap case; the citation
    keyword and its target may legitimately fall on either side of a
    comment-width line break, but the number's own line must be the one
    that starts the citation).
  - an explicit per-number tag `[producer: <fn_or_path>]` sits on line L,
    resolved the same way as `<fn>`/`<path>` above.
  - `no-producer: <reason>` sits on line L — covers ONLY its own line, never
    a neighboring one, so a stray opt-out elsewhere in the same comment
    block cannot silently launder an unrelated number.

Everything else in the citation space (a bare "see the module docs", "see
that commit's message") does NOT satisfy this gate: it names no
grep-verifiable function and no tracked artifact, so a reader still cannot
resolve the claim. That is intentional, not a bug — those are exactly the
un-resolvable citations this gate exists to catch.

FAIL-CLOSED, file:line, offending number. A committed allowlist
(`ALLOWLIST_PATH`) carries pre-existing debt, keyed on `file:number:sha1` —
`sha1` of the NORMALIZED text of the number's own line (not the line
number), so a line shift elsewhere in the file does not silently re-trigger
an already-tracked entry, and does not silently un-flag a genuinely edited
line either (the hash changes with the text). The allowlist may only shrink:
`--check-allowlist-only-shrinks` diffs it against `origin/main` and fails
closed on any ADDED entry — never on a removed one — so it is mechanically
impossible to launder a brand-new instance of this class through the
allowlist; the only sanctioned way to clear a violation is a real producer
citation or a `no-producer:` tag.

Run: `python3 ci/scripts/check_doc_numbers_have_producers.py`
Self-test (positive: unproduced measurements — including four reworded
probes drawn from a prior audit round — are flagged; negative: a citation
resolving to a REAL tracked fn, a wrapped citation, a per-number tag, a
no-producer-tagged derived constant, and every shape exclusion are clean):
`python3 ci/scripts/check_doc_numbers_have_producers.py --self-test`
Allowlist-only-shrinks leg (network: fetches `origin/main` to diff against —
the one leg of this gate that is not otherwise hermetic, matching
`swarm.yml`'s own TOUCHED-guard precedent):
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

# How many lines around a number's own line count as "nearby" for the
# TRIGGER-WORD check only (NOT the producer-citation binding, which is
# tightly bound to the number's own line ± one wrap line — see `is_bound`).
# Prose commonly wraps a phrase like "5145/16384 `m` elements differed"
# across a line boundary in a doc comment, so same-line-only trigger
# detection would miss it; going wider than a couple of lines starts roping
# in an unrelated trigger word from elsewhere in the same paragraph.
TRIGGER_WINDOW = 3

ADJACENT_WORD_ALTS = [
    r"cosine",
    r"mismatch(?:es)?",
    r"elements",
    r"ceiling",
    r"bit-match\w*",
    r"diverg\w*",
    r"measured",
    r"observed",
    r"differ(?:s|ed)?",
    r"disagree\w*",
    r"values",
    r"run",
    r"pod",
    r"a100",
    r"seed\w*",
]
_ADJACENT_RE = re.compile(r"\b(?:" + "|".join(ADJACENT_WORD_ALTS) + r")\b", re.IGNORECASE)

NM_RE = re.compile(r"\b\d{1,7}\s*/\s*\d{1,7}\b")
PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,4})?\s*%")
COSINE_RE = re.compile(r"\b0\.\d{2,4}\b")
VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")

# A crate name (this workspace's own deps that show up in prose as "candle
# 0.11", "candle-core 0.11's") immediately before a bare `X.Y` reads as a
# version mention, not a measurement — `VERSION_RE` above only catches
# 3-component versions (`0.11.0`); this catches the 2-component form this
# codebase actually writes crate versions as.
_CRATE_NAME_PREFIX_RE = re.compile(
    r"(?i)\b(?:candle(?:-[a-z]+)*|torch|cudarc|half|rayon|tokio|serde(?:-[a-z]+)*|"
    r"anyhow|thiserror|usearch)\s*$"
)

# A number immediately preceded (ignoring backticks/whitespace) by a
# definitional/comparison operator — `=`, `==`, `~=`, `>=`, `<=`, `≥`, `≤` —
# is the RHS of a formula, threshold, or named-constant definition ("FLOOR =
# 2^-2 = 0.25", "REL = 2^-6 = 1.5625%", "recall@k >= 0.95"), not a claimed
# observation. A bare comparison/approx symbol with nothing bound to it
# (`~5%`, `~0.008`) is NOT in this set — that is the ordinary hedge a real
# measurement is reported with, not a definition.
_DEFINITIONAL_OPERATOR_RE = re.compile(r"(==|~=|>=|<=|≥|≤|=)\s*$")

# A rustdoc intra-link to a SCREAMING_SNAKE constant — `` [`FLOOR_NAME`] `` —
# anywhere on the same line means the sentence is discussing a NAMED,
# already-defined code constant (its own producer is the `const`/`static`
# declaration the link resolves to), not reporting a fresh unproduced
# measurement. Deliberately restricted to ALL-CAPS identifiers (the Rust
# convention for `const`/`static`) so a lowercase/CamelCase link to a
# function or type (which says nothing about whether a NEARBY number is a
# measurement) does not trigger this.
_NAMED_CONST_LINK_RE = re.compile(r"\[`[A-Z][A-Z0-9_]*`\]")

# "95% CI" / "90% interval" / "95% confidence" — a fixed statistical
# convention (a chosen significance level), never itself a reported
# measurement, across every stats-adjacent file in this tree.
_STATISTICAL_CONVENTION_TAIL_RE = re.compile(r"(?i)^\s*(?:CI\b|interval\b|confidence\b)")

SEE_RE = re.compile(r"\bsee\s+`?([A-Za-z_][A-Za-z0-9_]*)\b")
PRINTED_BY_RE = re.compile(r"\bprinted by\s+`?([A-Za-z_][A-Za-z0-9_]*)\b")
MEASURED_BY_RE = re.compile(r"\bmeasured by\s+`?([^\s`,;]+)")
PRODUCER_TAG_RE = re.compile(r"\[producer:\s*([^\]]+)\]")
NO_PRODUCER_RE = re.compile(r"\bno-producer:")

# A line "ending with" one of the three citation keywords (rstripped of
# trailing whitespace and common trailing punctuation/dashes first) is the
# WRAP case: the keyword's target identifier/path continues on the next
# line. Longest alternative first so "measured by"/"printed by" don't get
# shadowed by a bare "see" (not applicable here since the three phrases
# share no textual overlap, but kept for the same discipline as elsewhere in
# this repo's own regex-ordering convention).
_WRAP_TAIL_RE = re.compile(r"(?i)\b(measured by|printed by|see)\s*$")

ASSERT_CALL_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|panic)!\s*\("
)
STRING_LITERAL_RE = re.compile(r'"')

_FN_DEF_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://!|///|//)")
_CONST_DECL_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+[A-Za-z_][A-Za-z0-9_]*\s*:")


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


def build_fn_index() -> set[str]:
    """Every `fn <name>` defined in any tracked `.rs` file, resolved entirely
    in Python (no `git grep -E`): BSD/Apple grep (macOS's system `grep`,
    which `git grep` shells out to on that platform) does not support the
    `\\b`/`\\s` PCRE-style escapes `-E`/POSIX ERE define, so the previous
    `git grep -InE '\\bfn\\s+NAME\\s*[(<]'` resolved NOTHING on macOS while
    working on Linux CI — a platform-dependent gate that silently diverged
    (44 findings on macOS vs 42 on Linux, two allowlist entries dead on the
    CI platform that actually runs this check). Reading every tracked `.rs`
    file's text once and matching with Python's own `re` module is
    platform-independent by construction.
    """
    names: set[str] = set()
    for rel in _run(["git", "ls-files", "--", "*.rs"]).splitlines():
        path = REPO_ROOT / rel
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for m in _FN_DEF_RE.finditer(text):
            names.add(m.group(1))
    return names


def fn_exists(name: str, fn_index: set[str], extra: set[str] | None = None) -> bool:
    """True if `name` is a real `fn` definition somewhere in the tracked
    `.rs` tree (`fn_index`, built once by `build_fn_index`), OR `name` is in
    `extra` — the self-test's injected set of SYNTHETIC-negative fixture
    names only (a fn that deliberately does NOT exist in the real tree, used
    to prove the resolver correctly rejects it). The self-test's
    positive-resolution leg cites a REAL tracked fn (`main`) and goes
    through `fn_index` like every other caller — `extra` is never used to
    fake a positive resolution.
    """
    if name in fn_index:
        return True
    if extra and name in extra:
        return True
    return False


def strip_comment_prefix(line: str) -> str:
    return _COMMENT_PREFIX_RE.sub("", line, count=1).lstrip()


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


def _tail_before(line: str, start: int) -> str:
    """The text immediately before `start`, with trailing whitespace and
    backticks stripped — the shared lookbehind window every shape exclusion
    below tests against.
    """
    return line[:start].rstrip().rstrip("`").rstrip()


def find_number_hits(
    line: str, trigger_context: str, prev_line: str = "", next_line: str = ""
) -> list[NumberHit]:
    """Measurement-shaped numbers on `line`, each REQUIRED to sit within
    `TRIGGER_WINDOW` lines (i.e. somewhere in `trigger_context`) of one of
    `ADJACENT_WORD_ALTS` (see module doc's WHAT COUNTS AS section), and each
    passed through the shape exclusions documented there. `prev_line` is
    only consulted when `line`'s own text before the number is empty (a
    doc-comment line-wrap put the number right after the `//`/`///` prefix
    with nothing else before it) — a wrapped `lr =\\n0.001` counts as
    definitional the same as an unwrapped `lr = 0.001` would.
    """
    if not _ADJACENT_RE.search(trigger_context):
        return []
    if _NAMED_CONST_LINK_RE.search(line):
        return []

    version_spans = [m.span() for m in VERSION_RE.finditer(line)]

    def in_version(span: tuple[int, int]) -> bool:
        return any(_overlaps(span, v) for v in version_spans)

    def is_definitional_or_versioned(start: int) -> bool:
        tail = _tail_before(line, start)
        if not strip_comment_prefix(line[:start]).strip() and prev_line:
            tail = _tail_before(prev_line, len(prev_line))
        return bool(_DEFINITIONAL_OPERATOR_RE.search(tail)) or bool(
            _CRATE_NAME_PREFIX_RE.search(tail)
        )

    def is_call_argument(start: int) -> bool:
        """A number that is a non-first, comma-separated argument inside an
        UNCLOSED `identifier(` on this same line — `Normal(0, 0.02)`,
        `swept ... values (\\`0.1\\`, \\`1.3/16\\`, ...)` — is a literal
        parameter in an input list, not free-standing prose reporting an
        observed result.
        """
        tail = _tail_before(line, start)
        if not tail.endswith(","):
            return False
        prefix = line[:start]
        return prefix.count("(") > prefix.count(")")

    claimed: list[tuple[int, int]] = []
    hits: list[NumberHit] = []

    for m in NM_RE.finditer(line):
        span = m.span()
        if in_version(span):
            continue
        text = m.group(0)
        start = span[0]
        if is_definitional_or_versioned(start) or is_call_argument(start):
            continue
        _, denom_str = re.split(r"\s*/\s*", text)
        try:
            denom = int(denom_str)
        except ValueError:
            denom = None
        # k/N where N < 16: an index/enumeration/small ratio, not a
        # mismatch-element count (this codebase's own real measured spans
        # start in the hundreds).
        if denom is not None and denom < 16:
            continue
        prev_char = line[start - 1] if start > 0 else ""
        # x^2/2-style exponent, or a grouped-division numerator right after
        # a closing paren — neither is a measured ratio.
        if prev_char in ("^", ")"):
            continue
        # "= 0/0" indeterminate form: covered by `is_definitional_or_versioned`
        # above (an immediately-preceding "="). A bare "0/0" with NO
        # preceding "=" is deliberately left flagged — the audit scoped this
        # exclusion to the "= 0/0" shape specifically, not every "0/0".
        # "n=32/900"-style shape enumeration.
        if re.search(r"(?i)\bn\s*=\s*$", line[:start].rstrip()):
            continue
        claimed.append(span)
        hits.append(NumberHit(text, "N/M"))

    for m in PCT_RE.finditer(line):
        span = m.span()
        if in_version(span):
            continue
        if is_definitional_or_versioned(span[0]) or is_call_argument(span[0]):
            continue
        after = line[span[1] :]
        if not after.strip() and next_line:
            after = strip_comment_prefix(next_line)
        if _STATISTICAL_CONVENTION_TAIL_RE.match(after):
            continue
        claimed.append(span)
        hits.append(NumberHit(m.group(0), "%"))

    for m in COSINE_RE.finditer(line):
        span = m.span()
        if in_version(span):
            continue
        if is_definitional_or_versioned(span[0]) or is_call_argument(span[0]):
            continue
        # One hit per token: a percentage's own fractional digits (e.g.
        # `0.52` inside `0.52%`) are already claimed above; don't also flag
        # them as a separate bare cosine.
        if any(_overlaps(span, c) for c in claimed):
            continue
        hits.append(NumberHit(m.group(0), "cosine"))

    return hits


def _resolve_wrap_target(
    lines: list[str],
    i: int,
    keyword: str,
    fn_index: set[str],
    extra_fns: set[str] | None,
    tracked: set[str],
) -> bool:
    if i + 1 >= len(lines):
        return False
    nxt = strip_comment_prefix(lines[i + 1])
    if keyword in ("see", "printed by"):
        m = re.match(r"`?([A-Za-z_][A-Za-z0-9_]*)", nxt)
        return bool(m) and fn_exists(m.group(1), fn_index, extra_fns)
    if keyword == "measured by":
        m = re.match(r"`?([^\s`,;]+)", nxt)
        if not m:
            return False
        candidate = m.group(1).strip("`").rstrip(".,;")
        return candidate in tracked
    return False


def _resolve_tag_or_citation(
    candidate: str, fn_index: set[str], extra_fns: set[str] | None, tracked: set[str]
) -> bool:
    candidate = candidate.strip().strip("`").rstrip(".,;")
    if not candidate:
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate) and fn_exists(
        candidate, fn_index, extra_fns
    ):
        return True
    return candidate in tracked


def is_bound(
    lines: list[str],
    i: int,
    fn_index: set[str],
    extra_fns: set[str] | None,
    tracked: set[str],
) -> bool:
    """True iff the number on `lines[i]` has a producer citation TIGHTLY
    bound to it — see module doc's PRODUCER CITATION section for the exact
    four accepted shapes. No line-window search: a citation elsewhere in the
    same comment block, even one line further than the wrap case allows,
    does not bind.
    """
    line = lines[i]

    m = SEE_RE.search(line)
    if m and fn_exists(m.group(1), fn_index, extra_fns):
        return True
    m = PRINTED_BY_RE.search(line)
    if m and fn_exists(m.group(1), fn_index, extra_fns):
        return True
    m = MEASURED_BY_RE.search(line)
    if m:
        candidate = m.group(1).strip("`").rstrip(".,;")
        if candidate in tracked:
            return True
    m = PRODUCER_TAG_RE.search(line)
    if m and _resolve_tag_or_citation(m.group(1), fn_index, extra_fns, tracked):
        return True
    if NO_PRODUCER_RE.search(line):
        return True

    wrap = _WRAP_TAIL_RE.search(line)
    if wrap and _resolve_wrap_target(
        lines, i, wrap.group(1).lower(), fn_index, extra_fns, tracked
    ):
        return True

    return False


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    number: str
    kind: str
    line_text: str  # the raw text of the number's own line, for allowlist hashing


def scan_lines(
    lines: list[str],
    file_label: str,
    fn_index: set[str],
    tracked: set[str],
    extra_fns: set[str] | None = None,
) -> list[Finding]:
    """`lines` is 0-indexed; line numbers reported are 1-indexed."""
    n = len(lines)
    scan_mask = [is_scan_line(l) for l in lines]

    findings: list[Finding] = []
    for i, line in enumerate(lines):
        if not scan_mask[i]:
            continue
        # A doc-comment line immediately preceding (zero gap) a `const NAME:
        # T = ...;` declaration is that constant's own closing summary
        # line — its "producer" is the declaration one line below, not a
        # fresh measurement claim. Deliberately zero-gap only: a number on a
        # line separated from its const by even one more `///` continuation
        # line is left flagged (two files in this tree place a genuinely
        # MEASURED result, not a definition, exactly one comment line above
        # their own const declaration — a wider lookahead could not tell
        # those two shapes apart, so this stays the narrower, safer rule).
        if i + 1 < n and _CONST_DECL_RE.match(lines[i + 1]):
            continue
        trig_lo = max(0, i - TRIGGER_WINDOW)
        trig_hi = min(n - 1, i + TRIGGER_WINDOW)
        trigger_context = "\n".join(lines[trig_lo : trig_hi + 1])
        prev_line = lines[i - 1] if i > 0 else ""
        next_line = lines[i + 1] if i + 1 < n else ""
        for hit in find_number_hits(line, trigger_context, prev_line, next_line):
            if not is_bound(lines, i, fn_index, extra_fns, tracked):
                findings.append(Finding(file_label, i + 1, hit.text, hit.kind, line))
    return findings


def scan_tree() -> list[Finding]:
    fn_index = build_fn_index()
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


# --- allowlist ---------------------------------------------------------


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
            "bound to it (see <fn> / printed by <fn> / measured by <path> / "
            "[producer: <fn_or_path>]), and no `no-producer:` opt-out on its own line.",
            file=sys.stderr,
        )
    print(
        "\ndoc-numbers-have-producers: a measurement-shaped number in a kernel/test doc "
        "comment or assert! message claims an observed run with nothing in the tree that "
        "produces it. Cite the real producer (`see <fn>`, `printed by <fn>`, `measured by "
        "<path>`, or `[producer: <fn_or_path>]`), or tag it `no-producer: <reason>` if it "
        "is genuinely derived (not measured) — never add it to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}, which only ever shrinks.",
        file=sys.stderr,
    )
    return 1


def check_allowlist_only_shrinks() -> int:
    """The allowlist tracks PRE-EXISTING debt only. This diffs the working
    tree's `ci/doc_number_allowlist.txt` against the version committed on
    `origin/main` and fails on any ADDED entry (never on a removed one — a
    shrink is always welcome). The one network call in this gate (`git
    fetch origin main`), matching `swarm.yml`'s own TOUCHED-guard precedent
    for comparing a branch against its merge base.
    """
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"], cwd=REPO_ROOT, check=False
    )
    base_out = subprocess.run(
        ["git", "show", "origin/main:ci/doc_number_allowlist.txt"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if base_out.returncode != 0:
        # `origin/main` has no allowlist file at all yet — this IS the PR
        # introducing it. There is no baseline to shrink relative to, so
        # every entry is, by construction, "new" against an empty file;
        # trivially pass rather than permanently red the one PR that lands
        # the mechanism. Strict enforcement (fails on any entry ADDED
        # relative to a real baseline) starts on the very next PR, once
        # this one merges and `origin/main` has the file.
        current = load_allowlist()
        print(
            "doc-number-allowlist-only-shrinks: OK (bootstrap) — "
            f"origin/main has no {ALLOWLIST_PATH.relative_to(REPO_ROOT)} yet "
            f"({len(current)} entries in this branch's copy establish the baseline)."
        )
        return 0
    base_text = base_out.stdout
    base = parse_allowlist_text(base_text)
    current = load_allowlist()
    added = current - base

    if added:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for entry in sorted(added):
            print(f"  + {':'.join(entry)}", file=sys.stderr)
        print(
            "\ndoc-number-allowlist-only-shrinks: this branch adds a NEW entry to "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}. The allowlist tracks pre-existing "
            "debt from before this gate landed and may never grow. Do not add an entry "
            "here — cite the real producer instead (`see <fn>`, `printed by <fn>`, "
            "`measured by <path>`, or `[producer: <fn_or_path>]`), or tag the number "
            "`no-producer: <reason>` if it is genuinely derived, not measured.",
            file=sys.stderr,
        )
        return 1

    print(
        f"doc-number-allowlist-only-shrinks: OK — {len(current)} entries "
        f"({len(base) - len(current)} shrunk vs origin/main)."
    )
    return 0


# --- self-test -------------------------------------------------------------

REAL_TRACKED_FN = "main"  # a real fn this repo's own tracked .rs tree defines many times over

POSITIVE_FRAGMENT = """
// measured on jammi-a100: 5145/16384 `m` elements differed from the eager
// CUDA chain at t=3 with nonzero prior moments — no producer cited anywhere
// near this claim.
"""

POSITIVE_PROBE_1 = """
// The RED drill measured 67546/262144 elements against the f32 truth run,
// with no citation anywhere near this line.
"""

POSITIVE_PROBE_2 = """
// bf16 scored 0.610 against the f32 truth run over the whole fixture,
// versus 0.810 for the f32-only baseline — neither number is cited.
"""

POSITIVE_PROBE_3 = """
// the two arms differ on 176/4096 values in this sweep, uncited.
"""

POSITIVE_PROBE_4 = """
// the fused kernel gives a 39% memory reduction measured on that pod,
// with nothing here saying how.
"""

NEGATIVE_CITED_SAME_LINE_FRAGMENT = f"""
// 100% (6144/6144) elements bit-match — see {REAL_TRACKED_FN} for the run
// that produced this count.
"""

NEGATIVE_CITED_WRAP_FRAGMENT = """
// 100% (6144/6144) elements bit-match — measured by
// Cargo.toml, a real tracked path (used here only as a stand-in artifact).
"""

NEGATIVE_TAG_FRAGMENT = f"""
// worst observed 0.796 cosine mismatch [producer: {REAL_TRACKED_FN}] on this
// fixture.
"""

NEGATIVE_NO_PRODUCER_OWN_LINE_FRAGMENT = """
// bf16 mantissa ceiling is 0.78% observed — no-producer: derived from the
// IEEE 754 bf16 format, not measured.
"""

NEGATIVE_NO_PRODUCER_WRONG_LINE_FRAGMENT = """
// bf16 mantissa ceiling is 0.78% observed, on this fixture.
// no-producer: derived from the IEEE 754 bf16 format, not measured.
"""

NEGATIVE_NO_TRIGGER_WORD_FRAGMENT = """
// the significance level is a fixed 95% CI (DEFAULT_REGRESSION_THRESHOLD is
// a fixed 30%, see the module docs), plain statistical constants with no
// claim attached to either one.
"""

NEGATIVE_UNRESOLVABLE_CITATION_FRAGMENT = """
// 26% of elements diverge — see the module docs for context.
"""

NEGATIVE_SHAPE_EXCLUSIONS_FRAGMENT = """
// candle-kernels-0.11.0 measured phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2),
// and the degenerate-variance regime is bound/max|signal| = 0/0 — the
// strongest possible measurement of an eps-sign flip. Elsewhere n=32/900
// values were observed, and 1/8 of elements differed in a small run.
// candle-core 0.11's backend has no observed BF16 impl. FLOOR = 2^-2 = 0.25
// covers every element measured; recall@k >= 0.95 is the observed gate
// floor. Reports a 95% CI alongside the observed point mean, a fixed
// significance level, not a measured value: Both A and B ~ Normal(0, 0.02),
// independent name-keyed streams, differed on the observed run. The
// two-tailed significance level for this measured, observed gate is a 95%
// interval, the same level another module's bootstrap significance CI uses.
// The wrapped form: this measured, observed gate's significance level is a
// 95%
// interval too, wrapped mid-phrase across the doc-comment line break.
"""


def _assert_flagged(name: str, fragment: str, fn_index: set[str], tracked: set[str], failures):
    hits = scan_lines(fragment.strip("\n").splitlines(), f"<self-test: {name}>", fn_index, tracked)
    if not hits:
        failures.append(f"self-test FAILED ({name}): expected a finding, got none")


def _assert_clean(
    name: str, fragment: str, fn_index: set[str], tracked: set[str], failures, extra_fns=None
):
    hits = scan_lines(
        fragment.strip("\n").splitlines(), f"<self-test: {name}>", fn_index, tracked, extra_fns
    )
    if hits:
        failures.append(f"self-test FAILED ({name}): expected no finding, got {hits}")


def self_test() -> int:
    failures: list[str] = []
    fn_index = build_fn_index()
    tracked = tracked_files()

    if REAL_TRACKED_FN not in fn_index:
        failures.append(
            f"self-test FAILED: precondition — {REAL_TRACKED_FN!r} must be a real fn "
            "somewhere in the tracked .rs tree for the positive-resolution leg to mean "
            "anything; build_fn_index() did not find it"
        )

    _assert_flagged("positive: unproduced measurement", POSITIVE_FRAGMENT, fn_index, tracked, failures)
    _assert_flagged("probe 1: RED drill 67546/262144", POSITIVE_PROBE_1, fn_index, tracked, failures)
    _assert_flagged("probe 2: 0.610 vs 0.810", POSITIVE_PROBE_2, fn_index, tracked, failures)
    _assert_flagged("probe 3: differ on 176/4096 values", POSITIVE_PROBE_3, fn_index, tracked, failures)
    _assert_flagged("probe 4: 39% memory reduction on a pod", POSITIVE_PROBE_4, fn_index, tracked, failures)

    _assert_clean(
        "cited same-line, real fn (no extra_fns)",
        NEGATIVE_CITED_SAME_LINE_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    _assert_clean(
        "cited via wrap (measured by <path> continuing on next line)",
        NEGATIVE_CITED_WRAP_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    _assert_clean(
        "per-number [producer: ...] tag, real fn",
        NEGATIVE_TAG_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    _assert_clean(
        "no-producer: on the number's own line",
        NEGATIVE_NO_PRODUCER_OWN_LINE_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    # no-producer: on a DIFFERENT line must NOT bind (finding 2: "no-producer:
    # covers ONLY its own line") — this fragment's number is expected to be
    # FLAGGED, not clean.
    _assert_flagged(
        "no-producer: on the WRONG line must not bind",
        NEGATIVE_NO_PRODUCER_WRONG_LINE_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    _assert_clean(
        "no adjacent trigger word",
        NEGATIVE_NO_TRIGGER_WORD_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )
    _assert_clean(
        "shape exclusions (version, exponent, indeterminate, n=, k/N<16)",
        NEGATIVE_SHAPE_EXCLUSIONS_FRAGMENT,
        fn_index,
        tracked,
        failures,
    )

    hits = scan_lines(
        NEGATIVE_UNRESOLVABLE_CITATION_FRAGMENT.strip("\n").splitlines(),
        "<self-test: unresolvable citation>",
        fn_index,
        tracked,
    )
    if not hits:
        failures.append(
            "self-test FAILED: 'see the module docs' (names no grep-verifiable fn, no "
            "tracked artifact) incorrectly satisfied the producer requirement"
        )

    # fn_exists: extra_fns is for SYNTHETIC negatives only — a name not in
    # the real tree must still resolve when explicitly injected, but must
    # NOT resolve without it.
    if fn_exists("definitely_not_a_real_fn_anywhere", fn_index, None):
        failures.append("self-test FAILED: fn_exists resolved a nonexistent fn with no extra_fns")
    if not fn_exists("definitely_not_a_real_fn_anywhere", fn_index, {"definitely_not_a_real_fn_anywhere"}):
        failures.append("self-test FAILED: fn_exists did not resolve a name explicitly in extra_fns")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("doc-numbers-have-producers self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "doc-numbers-have-producers self-test: OK — unproduced measurements (including all "
        "four reworded probes) flagged; same-line/wrap/tagged/no-producer-on-its-own-line "
        "citations clean; no-producer on the wrong line still flags; shape exclusions hold; "
        "fn_exists resolves only real or explicitly-injected names."
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
