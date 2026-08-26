#!/usr/bin/env python3
"""Rule (h): every numeric token in an in-scope performance table cites a
tracked artifact, or is escaped into a committed, shrink-only ledger.

THE CLASS THIS CLOSES. `docs/maintainer/fine-tune-performance-guide.md`
prints ~200 measurement-shaped numbers across nine pipe tables. Nothing
mechanically checked that any of them still agreed with the tracked cuda-run
artifacts they claim to summarize — a number could drift from its source
(estimator shopping: any of min/mean/max/r1/r2 "looks right"), or never have
had one (a session-ledger number with no committed producer), and the guide
would read as equally authoritative either way. This gate makes that
distinction mechanical, per CELL, at the doc's own printed precision.

SCOPE. A pipe table (a header row, a `|---|...` separator, and its
contiguous body rows) under `docs/maintainer/**` or
`docs/plans/61-perf-unification/**` is IN SCOPE if its header row contains
one of the trigger substrings below (the GENERAL, future-proof rule), OR if
it is one of the nine tables this contract's own census names in
`docs/maintainer/fine-tune-performance-guide.md` (`KNOWN_TABLES` below) —
UNLESS its header is a CITATION table (`_is_citation_table_header`: a header
containing both "rule" and "escape", case-insensitively — this guide's own
§7 rule/KO/escape table is exactly this shape and is never a measurement
table even though the audit that added this rule confirmed its header
carries no `HEADER_TRIGGERS` substring today; the exclusion is stated and
mechanical, not an accident of the trigger list staying narrow). Three of
the nine named tables (T3, T5, T8) have header rows that, verified
byte-exact against the guide, carry NONE of the eight trigger substrings —
a gap between the general heuristic and the census it requires, closed by
`KNOWN_TABLES` rather than by widening `HEADER_TRIGGERS`.

SCOPE INVARIANTS (audit A3). Every `KNOWN_TABLES` header must match EXACTLY
ONE table in its file — a header rename or a vanished table must RED, never
silently drop a table out of scope. `EXPECTED_TABLE_COUNTS` and
`EXPECTED_DENOMINATOR` pin the per-file in-scope table count and the
tokenizer's own total count as standing constants the bare run checks; a
change to either (a renamed column, a new/removed table, a widened/narrowed
exclusion class) must update the constant in the same diff or the gate REDs
— this is what makes a *silent* scope collapse (a table quietly falling out
of scope, its tokens no longer checked at all) impossible to miss.

TOKENIZER. A numeric token is a maximal match of
`[-−+]?\\d+(?:[.,]\\d+)*` not immediately preceded by a word character
or `.` (so a token embedded in a git-sha-like run of letters+digits, e.g.
`eee7e6a`, or in a longer decimal, is never split out), scanned over every
BODY-ROW cell (label column included). A comma-grouped token (`159,183`,
read from bracket-list prose like `[159,183]`) is a single token under this
regex and can never resolve through the pointer grammar (no artifact field
is ever comma-grouped) — it always routes to `ledger`; this is a stated
limitation of the literal tokenizer regex, not a silent misparse, and no
artifact in this tree has ever needed a thousands-separated integer.

Every token is one of:
  - EXCLUDED (`is_excluded_span`): a shape label (`b8s512`, `s512`, `b8`,
    `d0`/`d0.05`), an issue/PR ref (`#377`) or the escape-ledger's own
    `esc-NNN` row id, a version string (`2.13.0+cu126`) or driver-style
    `cu126`, a ledger cite (`s2:89`, `s2:245-300`, `row 5`, `cont row 11`,
    `fusion rows 30, 36` — ONE unified pattern now covers a bare `row(s) N`
    and a `(fusion )?row(s) N, M, ...` list identically, audit advisory: the
    old two-pattern form excluded only the first number of a bare list), a
    date (`2026-08-23`), a seed label (`seed 42`), a section cross-reference
    (`§6`), a `<digit>/√<var>` formula fragment (`1/√d`), a slash-versioned
    type name (`InplaceOp2/3`), or a hyphen-versioned proper noun / config
    constant (`FlashAttention-2`, `rank-16`) — audit A5: these five classes
    were previously ledgered as "no producer", which overstated the
    ledger's honest debt: a formula digit, a type name, a section ref, and
    a seed label were never measurements needing a producer at all; they
    are lexical noise the tokenizer should never have surfaced as a claim
    in the first place. `\\d+ (layers|tensors|launches|memcpys|sites|
    seeds)` is explicitly NOT excluded (P2) — those are measurements, not
    labels, even though "launches" is also a header trigger word.
  - BOUND by exactly one comment-tag entry (below).
  - ESCAPED into the ledger via the tag entry `ledger` (bare word — this is
    the ONLY escape spelling; there is no inline `no-producer: <reason>`
    form, C11), cross-checked against a committed allowlist entry keyed
    `file:token:sha1(normalized line):col` — audit advisory: the key now
    includes the token's own COLUMN offset so two identical tokens on one
    line (e.g. `112 / 112`) get two distinct, individually-shrinkable
    entries instead of collapsing into one that silently covers both.
An uncovered token (no comment, or fewer tag entries than tokens) is a
FINDING at `file:line:col`.

TAG PLACEMENT. An HTML comment on the line immediately above a body row:
`<!-- claims: c1=<expr>; c2=<expr>; c3=ledger; ... -->` — one entry per
NON-EXCLUDED token, left to right, cell by cell. Comments do not render, so
the tables are visually unchanged.

GRAMMAR (closed; six value forms + `ledger`; anything else is a parse
error, never a silent pass):
  P              <tracked-path>#<json-pointer>, resolving to a JSON number.
  min/mean/max   over >=2 pointers.
  diff(A,B)      A - B, A/B pointer or aggregate.
  ratio(A,B)     A / B.            pct(A,B)   (A/B - 1) * 100.
  <form> as U    U in {ms,s,GB,GiB,MiB,%}; FIXED factors (s->ms x1000,
                 bytes->GB /1e9, bytes->GiB /2^30, bytes->MiB /2^20,
                 x100 for %) applied to the form's raw value, never a
                 unit-aware conversion from the field's true unit.
  neg(<form>)    the printed cell is the ADDITIVE INVERSE of the evaluated
                 form (audit advisory, REPLACES the old `abs(...)`: `abs()`
                 discarded sign on BOTH sides, so it would have silently
                 accepted a doc token with the WRONG sign — e.g. `+38.5`
                 for an artifact value whose negation is `-38.5` — as long
                 as the magnitudes agreed; `neg(...)` computes `-value` and
                 then compares it SIGN-INCLUSIVE against the token, so a
                 sign error is a mismatch like any other).
  legacy(<form>) same evaluation and equality rule as <form>; reported as
                 V-legacy, not V. RESTRICTED (audit advisory) to the exact
                 pointers in `LEGACY_POINTER_ALLOWLIST` — the two named
                 AdamW-summary cells C12.2 calls out — never accepted on an
                 arbitrary tag; rule (g)'s v2-schema classification, which
                 would derive V-legacy mechanically, is out of phase-3
                 scope, so this is an explicit, closed stand-in, not an
                 author-trusted marker.

ARTIFACT-FIELD PRECEDENCE (P3, C10.4; rewritten per audit A1). For a
`diff`/`ratio`/`pct` tag, gather every leaf pointer among its operands; if
all leaves share the same tracked file, search the WHOLE FILE (not merely
the operands' common-ancestor subtree — the original walk stopped there and
missed a computed field living in a SIBLING subtree, e.g. cast-w1's own
`/deltas/b8_s512_p50_ms/delta_ms` relative to operands under `/legs/...`,
which are siblings of `/deltas` at the artifact's top level, not ancestors
of it) for a leaf whose OWN key matches `delta*`/`ratio*`/`speedup*`/
`*_pct`/`spread*` AND whose OWN json-pointer path shares every
digit-bearing identifier token (`_path_digit_tokens`) common to ALL the
operands' own paths, EXCLUDING any per-replicate suffix (`r1`, `r2`,
`rep...`) — round-3 audit fix: a per-rep suffix is explicitly STRIPPED
before intersecting, not left to "never survive the intersection" as an
earlier revision's docstring claimed. That claim was false: a SAME-rep
pair (`.../legs/b8_s512_disabled_r1/s_per_step_p50` vs `.../legs/
b8_s512_fused_r1/s_per_step_p50` — both `r1`) keeps `r1` in the natural
intersection, over-constrains the candidate search, and silently misses
the artifact's own `/deltas/b8_s512_p50_ms/delta_ms` (whose path carries
no rep suffix at all) — binding `39.565658` bit-for-bit as an unchecked
free diff. Stripping the suffix outright closes the class regardless of
whether the two operands happen to share or differ on their own rep index.
The remaining identifier tokens must ALSO be consistent with a matching
UNIT FAMILY (`_unit_family`: `ms`/`s`/`bytes` inferred from each leaf key's
own suffix) — a `_ms`-family candidate is only offered to `_ms`/`s_per_
step`-shaped operands, a `_gb`/bytes-family candidate only to VRAM-shaped
operands; a shared shape token like `b8_s512` alone is not enough (it
would otherwise let a VRAM diff match a millisecond field of the same
shape, or vice versa). When the operands span more than one tracked file,
or share NO digit-bearing identifier token at all, precedence is
UNDECIDABLE and the tag is a FINDING naming that explicitly ("precedence
undecidable: ...") — never a silent pass just because the mechanism could
not determine an answer. NUMBER and STRING leaves are both hits (audit A1:
a string-valued field, e.g. a hand-written `"+0.235 to +0.369 GB"` range,
is a FINDING too — "artifact states a range: bind two tokens or ledger it"
— never a silently blessed pass just because it isn't a JSON number the
pointer grammar could resolve to). If a hit exists, the tag is a FINDING
naming the field's own pointer, unless the tag's own operand pointers
already include that exact leaf.

EQUALITY (P3). The token is parsed as `Decimal` at ITS OWN printed
precision (digits after `.`); the evaluated expression is quantized to that
same precision with `ROUND_HALF_EVEN` and compared as the Decimal STRING
(never float — this is what makes `1.2650`/`0.8300` bind, since a trailing
zero is significant to a string compare but invisible to a float one). A
mismatch prints the evaluated value at full precision.

POINTER ROOTS. Tracked `*.json` under
`crates/jammi-kernels/artifacts/cuda-runs/**` or
`crates/jammi-bench/baselines/*.json` only. A pointer at any other path, a
non-`.json` payload (a `LEGACY_RAW_NONJSON` raw leg, `*.json.raw`), or an
untracked file is a FINDING ("unprovenanced producer") — never silently
resolved.

THE LEDGER is `ci/perf_claims_allowlist.txt` (grows ONLY via a human-
reviewed PR; `--check-allowlist-only-shrinks` is the ratchet) plus
`ci/perf_claims_allowlist_classification.md` (one row per entry, a closed
reason: `ledger-only | modeled | issue-text | superseded-run`) — verified
mechanically by `check_classification_file` (audit advisory: these two
constants used to be dead weight; every allowlist entry now must have
EXACTLY one classification row with a reason in the closed set, or the bare
run FAILS naming the drift).

KNOWN LEDGER-KEY LIMITATION (inherited from `check_doc_numbers_have_
producers.py`, `ac2c5cb`; out of scope to redesign here — stated, not
fixed). The key is `file:token:sha1(normalized line):col`. A purely
EDITORIAL edit to an already-ledgered row (a typo fix, a rewording that
never touches the token itself) changes that row's normalized-line hash and
therefore its key: the OLD key becomes an orphaned allowlist entry (a
`check_classification_file` finding) and the still-unproducible token needs
a NEW entry under the new hash — which `--check-allowlist-only-shrinks`
reads as a plain ADDITION and REDS, indistinguishable from genuinely new
debt. The remedy is a human one, stated in the same PR: remove the orphaned
old entry, add the new one with the SAME classification-row content
(reason and note unchanged), and say in the PR body that this is a
net-zero editorial swap, not new debt — the ratchet cannot tell the two
apart mechanically, and does not try to.

F1 (audit A4, lead decision — `docs/plans/61-perf-unification/CONTRACT.md`
§6). The acceptance bar is `>= 1 fully-bound table AND V >= 50 AND the
ledger may only shrink` — a standing ratchet, not a PR-body number.
`--report --min-fully-bound=N --min-v=N` exits non-zero if either floor is
missed; wired as a REQUIRED fourth `guard` leg beside the bare/self-test/
ratchet legs.

Run:      python3 ci/scripts/check_perf_claims.py
Report:   python3 ci/scripts/check_perf_claims.py --report [--min-fully-bound=N --min-v=N]
Self-test:python3 ci/scripts/check_perf_claims.py --self-test
Ratchet:  python3 ci/scripts/check_perf_claims.py --check-allowlist-only-shrinks
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = [
    REPO_ROOT / "docs" / "maintainer",
    REPO_ROOT / "docs" / "plans" / "61-perf-unification",
]

HEADER_TRIGGERS = ["s/step", "VRAM", "ratio", "cosine", "×", "ms", "GB", "launches"]

# The nine tables this contract's own census names (§2 row E6), identified
# by their LITERAL header-row text (stable across a claims-tag edit, unlike
# a line-number range). Three of the nine (T3, T5, T8) carry a header row
# with none of HEADER_TRIGGERS — see the module doc's SCOPE section.
KNOWN_TABLES = {
    "docs/maintainer/fine-tune-performance-guide.md": {
        "| stack | s/step | triplets/s | peak VRAM | GPU util | mem-ctrl util |",  # T1 — §1 (#352)
        "| tip (b8·s128, PCIe, 15 steps) | s/step | Δ | modeled |",  # T2 — §3 C-series
        "| lever | mechanism | predicted | measured (same box, SXM4) | verdict |",  # T3
        "| kernel | ms/step | launches | what it is |",  # T4 — §3 census
        "| lever | projection history | measured (one build, forced arm off/on) | PR |",  # T5
        "| shape | jammi stacked s/step | torch-sdpa | jammi all-off | torch ÷ jammi |",  # T6
        "| pair | mean cosine | notes |",  # T7 — §6 esc-045 cosines
        "| operating point (b4·s128) | block-fused | eager | torch bf16 | flash (FA2 tip) |",  # T8
        "| b8·s512, dropout 0 | s/step | VRAM | box |",  # T9 — §10
    }
}

# audit A3: pinned scope constants, checked by the bare run. Recompute both
# after ANY change to KNOWN_TABLES, the exclusion class, or the guide's own
# tagged tables — the bare run FAILS on a mismatch, naming both numbers.
EXPECTED_TABLE_COUNTS = {
    "docs/maintainer/fine-tune-performance-guide.md": 9,
}
EXPECTED_DENOMINATOR = 242

ALLOWLIST_PATH = REPO_ROOT / "ci" / "perf_claims_allowlist.txt"
CLASSIFICATION_PATH = REPO_ROOT / "ci" / "perf_claims_allowlist_classification.md"
CLASSIFICATION_REASONS = {"ledger-only", "modeled", "issue-text", "superseded-run"}

POINTER_ROOTS = [
    "crates/jammi-kernels/artifacts/cuda-runs",
    "crates/jammi-bench/baselines",
]

# audit advisory: `legacy(...)` is restricted to an explicit, named
# allowlist of (path, pointer) pairs rather than accepted on any tag — the
# exact two AdamW-summary cells C12.2 names, never inferred.
LEGACY_POINTER_ALLOWLIST = {
    (
        "crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json",
        "/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/0",
    ),
    (
        "crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json",
        "/a100b_full_step_ab_reference/summary/s512/disabled_eager_p50_r1_r2/1",
    ),
    (
        "crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json",
        "/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/0",
    ),
    (
        "crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4.json",
        "/a100b_full_step_ab_reference/summary/s512/fused_p50_r1_r2/1",
    ),
}

# --- tokenizer -----------------------------------------------------------

NUMBER_RE = re.compile(r"(?<![\w.])[−\-+]?\d+(?:[.,]\d+)*")

_EXCLUSION_PATTERNS = [
    # ledger cites — checked BEFORE the bare shape-label patterns below so
    # their spans win the union regardless of pattern application order
    # (finditer per-pattern; excluded spans are unioned across patterns).
    # audit advisory: ONE unified pattern for `row(s) N` and
    # `(fusion )?row(s) N, M, ...` — the old two-pattern form excluded only
    # the FIRST number of a bare `row 5, 6` list while the `fusion rows`
    # form excluded the whole list; both now share one pattern that
    # consumes the entire comma-separated run either way.
    re.compile(r"\bs\d+:\d+(?:–\d+|-\d+)?\b"),
    re.compile(r"\b(?:fusion\s+)?rows?\s+[\d,\s]*\d\b"),
    re.compile(r"\bcont row \d+\b"),
    # dates
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # version strings / driver-style cuNNN
    re.compile(r"\b\d+\.\d+\.\d+(?:\+\w+)?\b"),
    re.compile(r"\bcu\d{3}\b"),
    # issue/PR refs, and the escape-ledger's own `esc-NNN` identifier (the
    # same reasoning as `#\d+`: a row ID, never a measurement)
    re.compile(r"#\d+\b"),
    re.compile(r"\besc-\d+\b"),
    # audit A5: seed labels, section cross-refs, a `1/√d`-style formula
    # digit, a slash-versioned type name, and a hyphen-versioned proper
    # noun / config constant — none of these is a measurement; each was
    # previously ledgered, overstating the ledger's honest debt.
    re.compile(r"\bseeds?\s+\d+\b"),
    re.compile(r"§\s*\d+\b"),
    re.compile(r"\b\d+/√[A-Za-z]*"),
    re.compile(r"\b[A-Za-z]+\d+/\d+\b"),
    re.compile(r"\b[A-Za-z]{3,}-\d{1,2}\b"),
    # shape labels
    re.compile(r"\bb\d+[·x]?s\d+\b"),
    re.compile(r"\bs\d+\b"),
    re.compile(r"\bb\d+\b"),
    re.compile(r"\bd0(?:\.\d+)?\b"),
]


def excluded_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pat in _EXCLUSION_PATTERNS:
        spans.extend(m.span() for m in pat.finditer(text))
    return spans


def _in_any_span(pos_span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    a, b = pos_span
    return any(a >= s and b <= e for s, e in spans)


@dataclass(frozen=True)
class Token:
    text: str
    line_no: int  # 1-indexed
    col: int  # 0-indexed
    line_text: str


def find_tokens_in_row(line: str, line_no: int) -> list[Token]:
    """Numeric tokens over the WHOLE row line (every cell, label column
    included), minus anything covered by an exclusion span."""
    ex = excluded_spans(line)
    out: list[Token] = []
    for m in NUMBER_RE.finditer(line):
        if _in_any_span(m.span(), ex):
            continue
        out.append(Token(m.group(0), line_no, m.start(), line))
    return out


# --- table discovery -------------------------------------------------------

_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_COMMENT_RE = re.compile(r"^\s*<!--\s*claims:\s*(?P<body>.*?)\s*-->\s*$")


@dataclass
class Table:
    file: str
    header_line_no: int
    header_text: str
    body_line_nos: list[int]  # 1-indexed


def header_in_scope(header_text: str) -> bool:
    return any(t in header_text for t in HEADER_TRIGGERS)


def is_citation_table_header(header_text: str) -> bool:
    """audit A3: a table whose header contains BOTH "rule" and "escape"
    (case-insensitive) is a citation table (rule -> kernel-guide section ->
    KO check -> the escape that paid for it), never a measurement table —
    excluded unconditionally, overriding HEADER_TRIGGERS/KNOWN_TABLES, so a
    future header edit that happens to introduce a trigger substring into
    this shape of table still does not sweep it into scope by accident."""
    low = header_text.lower()
    return "rule" in low and "escape" in low


def find_tables(
    rel_path: str, lines: list[str], header_hits: dict[str, int] | None = None
) -> list[Table]:
    """`lines` is 0-indexed; table line numbers reported are 1-indexed.
    `header_hits` (audit A3), if given, is incremented once per line whose
    text exactly matches a `KNOWN_TABLES` header for `rel_path` — used to
    assert every known header matched EXACTLY one table across the file."""
    tables: list[Table] = []
    n = len(lines)
    i = 0
    known_headers = KNOWN_TABLES.get(rel_path, set())
    while i < n - 1:
        if _ROW_RE.match(lines[i]) and _SEP_RE.match(lines[i + 1]):
            header_line_no = i + 1
            header_text = lines[i]
            j = i + 2
            body_line_nos: list[int] = []
            while j < n and (_ROW_RE.match(lines[j]) or _COMMENT_RE.match(lines[j])):
                if _ROW_RE.match(lines[j]):
                    body_line_nos.append(j + 1)
                j += 1
            stripped = header_text.strip()
            if header_hits is not None and stripped in known_headers:
                header_hits[stripped] = header_hits.get(stripped, 0) + 1
            in_scope = (
                header_in_scope(header_text) or stripped in known_headers
            ) and not is_citation_table_header(header_text)
            if in_scope and body_line_nos:
                tables.append(Table(rel_path, header_line_no, header_text, body_line_nos))
            i = j
        else:
            i += 1
    return tables


def check_known_table_usage(rel_path: str, header_hits: dict[str, int]) -> list[str]:
    """audit A3: every `KNOWN_TABLES` header for `rel_path` must appear
    EXACTLY once across the scanned file — zero (renamed/removed) or more
    than one (duplicated) is a named problem, never a silent pass."""
    problems: list[str] = []
    for header in sorted(KNOWN_TABLES.get(rel_path, set())):
        count = header_hits.get(header, 0)
        if count == 0:
            problems.append(f"{rel_path}: KNOWN_TABLES header never matched (renamed/removed?): {header!r}")
        elif count > 1:
            problems.append(
                f"{rel_path}: KNOWN_TABLES header matched {count} times (expected exactly 1): {header!r}"
            )
    return problems


# --- claim-tag grammar -----------------------------------------------------


class ClaimParseError(Exception):
    pass


class PrecedenceUndecidable(ClaimParseError):
    """round-3 audit fix: raised (never silently swallowed to `None`) when
    the artifact-field precedence check (P3) cannot determine an answer —
    the operands span more than one tracked file, or share no digit-
    bearing identifier token to key a same-quantity search off of. A
    subclass of `ClaimParseError` so it is caught by the same handler and
    reported as a FINDING, exactly like a genuine precedence violation."""


@dataclass(frozen=True)
class PointerRef:
    path: str  # repo-relative
    pointer: str  # "/a/b/c"


@dataclass(frozen=True)
class Expr:
    kind: str  # "pointer" | "min" | "mean" | "max" | "diff" | "ratio" | "pct"
    args: tuple  # PointerRef or Expr, per kind
    unit: str | None = None
    is_neg: bool = False
    is_legacy: bool = False


_UNIT_FACTORS = {
    "ms": Decimal(1000),
    "s": Decimal(1),
    "GB": Decimal(1) / Decimal(10**9),
    "GiB": Decimal(1) / Decimal(2**30),
    "MiB": Decimal(1) / Decimal(2**20),
    "%": Decimal(100),
}


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


_POINTER_RE = re.compile(r"^(?P<path>[^\s#()]+)#(?P<ptr>/\S*)$")
_AGG_RE = re.compile(r"^(min|mean|max)\((.*)\)$", re.DOTALL)
_DIFF_RE = re.compile(r"^diff\((.*)\)$", re.DOTALL)
_RATIO_RE = re.compile(r"^(ratio|pct)\((.*)\)$", re.DOTALL)
_NEG_RE = re.compile(r"^neg\((.*)\)$", re.DOTALL)
_LEGACY_RE = re.compile(r"^legacy\((.*)\)$", re.DOTALL)
_AS_UNIT_RE = re.compile(r"^(.*)\s+as\s+(ms|s|GB|GiB|MiB|%)$", re.DOTALL)


def parse_pointer(s: str) -> PointerRef:
    m = _POINTER_RE.match(s.strip())
    if not m:
        raise ClaimParseError(f"not a pointer: {s!r}")
    return PointerRef(m.group("path"), m.group("ptr"))


def parse_operand(s: str) -> PointerRef | Expr:
    s = s.strip()
    m = _AGG_RE.match(s)
    if m:
        kind = m.group(1)
        pointers = [parse_pointer(p) for p in _split_top_level(m.group(2))]
        if len(pointers) < 2:
            raise ClaimParseError(f"{kind}(...) needs >=2 pointers: {s!r}")
        return Expr(kind, tuple(pointers))
    return parse_pointer(s)


def parse_base_form(s: str) -> Expr:
    s = s.strip()
    m = _AGG_RE.match(s)
    if m:
        kind = m.group(1)
        pointers = [parse_pointer(p) for p in _split_top_level(m.group(2))]
        if len(pointers) < 2:
            raise ClaimParseError(f"{kind}(...) needs >=2 pointers: {s!r}")
        return Expr(kind, tuple(pointers))
    m = _DIFF_RE.match(s)
    if m:
        parts = _split_top_level(m.group(1))
        if len(parts) != 2:
            raise ClaimParseError(f"diff(...) needs exactly 2 operands: {s!r}")
        return Expr("diff", tuple(parse_operand(p) for p in parts))
    m = _RATIO_RE.match(s)
    if m:
        kind, inner = m.group(1), m.group(2)
        parts = _split_top_level(inner)
        if len(parts) != 2:
            raise ClaimParseError(f"{kind}(...) needs exactly 2 operands: {s!r}")
        return Expr(kind, tuple(parse_operand(p) for p in parts))
    # bare pointer
    return Expr("pointer", (parse_pointer(s),))


def parse_expr(raw: str) -> Expr:
    """Full grammar: `legacy(...)`? wraps everything; then an optional
    trailing ` as <unit>`; then an optional `neg(...)` wrap; then a base
    form (pointer / aggregate / diff / ratio / pct)."""
    s = raw.strip()
    is_legacy = False
    m = _LEGACY_RE.match(s)
    if m:
        is_legacy = True
        s = m.group(1).strip()

    unit = None
    m = _AS_UNIT_RE.match(s)
    if m:
        s = m.group(1).strip()
        unit = m.group(2)

    is_neg = False
    m = _NEG_RE.match(s)
    if m:
        is_neg = True
        s = m.group(1).strip()

    base = parse_base_form(s)
    return Expr(base.kind, base.args, unit=unit, is_neg=is_neg, is_legacy=is_legacy)


# --- JSON pointer resolution / precedence check -----------------------------


class ResolutionError(Exception):
    pass


def _rfc6901_walk(doc, pointer: str):
    if pointer == "" or pointer == "/":
        segs = []
    else:
        segs = pointer.lstrip("/").split("/")
    cur = doc
    for seg in segs:
        seg = seg.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError as exc:
                raise ResolutionError(f"non-integer list index {seg!r} in {pointer!r}") from exc
            cur = cur[idx]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise ResolutionError(f"key {seg!r} missing (pointer {pointer!r})")
            cur = cur[seg]
        else:
            raise ResolutionError(f"cannot descend into a scalar at {seg!r} ({pointer!r})")
    return cur, segs


_REP_SUFFIX_RE = re.compile(r"^r\d+$|^rep\w*$")


def _path_digit_tokens(pointer: str) -> set[str]:
    """Digit-bearing identifier tokens in a JSON-pointer path — `b8`,
    `s512`, `p50` — split generically on any non-alphanumeric character. A
    token with NO digit (`legs`, `deltas`, `fused`) is structural prose,
    never an identifier, and is dropped. A per-REPLICATE suffix (`r1`,
    `r2`, `rep...`) is ALSO dropped (round-3 audit fix) — it identifies
    WHICH RUN an operand came from, never the QUANTITY being computed, and
    keeping it let a same-rep operand pair over-constrain the search (see
    the module doc's PRECEDENCE section)."""
    segs = re.split(r"[^A-Za-z0-9]+", pointer)
    return {
        s.lower()
        for s in segs
        if s and any(ch.isdigit() for ch in s) and not _REP_SUFFIX_RE.match(s.lower())
    }


_UNIT_FAMILY_TIME_RE = re.compile(r"(?:^|_)(?:ms|s)(?:$|_)|s_per_step")
_UNIT_FAMILY_BYTES_RE = re.compile(r"(?:^|_)(?:gb|gib|mib|bytes)(?:$|_)|bytes")


def _unit_family(key: str) -> str | None:
    """A coarse unit-family guess from a leaf key's own suffix (round-3
    audit fix): `time` for a seconds- OR milliseconds-shaped name
    (`s_per_step_p50`, `..._s`, `delta_ms` — a delta computed FROM raw
    seconds is routinely REPORTED in ms, same physical quantity, so `s`
    and `ms` are ONE family, not two — an earlier draft of this function
    split them and vetoed the real cast-w1 s_per_step-vs-delta_ms match,
    caught by this file's own self-test), `bytes` for a VRAM/byte-shaped
    name (`_gb`/`_gib`/`_mib`/`bytes`). `None` if undetermined —
    undetermined NEVER blocks a match by itself; it only means the family
    check cannot VETO a token-identifier match. Used to stop a shared
    shape token (`b8_s512`) alone from matching a candidate of the WRONG
    family — a VRAM diff must never bind to a time field of the same
    shape, or vice versa."""
    low = key.lower()
    if _UNIT_FAMILY_TIME_RE.search(low):
        return "time"
    if _UNIT_FAMILY_BYTES_RE.search(low):
        return "bytes"
    return None


def _operand_family(refs: list[PointerRef]) -> str | None:
    families = {_unit_family(r.pointer.rsplit("/", 1)[-1]) for r in refs}
    families.discard(None)
    return next(iter(families)) if len(families) == 1 else None


def _iter_leaves(doc, prefix: str = ""):
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield from _iter_leaves(v, f"{prefix}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from _iter_leaves(v, f"{prefix}/{i}")
    else:
        yield prefix, doc


class Loader:
    """Resolves `<path>#<pointer>` refs against tracked JSON under
    POINTER_ROOTS, with per-file caching. Injectable for tests."""

    def __init__(self, tracked: set[str] | None = None, base_dir: Path | None = None):
        self._tracked = tracked
        self._base_dir = base_dir or REPO_ROOT
        self._cache: dict[str, object] = {}

    def _check_path(self, path: str) -> None:
        if not path.endswith(".json"):
            raise ResolutionError(f"unprovenanced producer: not a .json payload: {path}")
        if not any(path == r or path.startswith(r + "/") for r in POINTER_ROOTS):
            raise ResolutionError(f"pointer path is not under an allowed root: {path}")
        if self._tracked is not None and path not in self._tracked:
            raise ResolutionError(f"pointer path is not a tracked file: {path}")

    def doc(self, path: str):
        self._check_path(path)
        if path not in self._cache:
            fp = self._base_dir / path
            if not fp.is_file():
                raise ResolutionError(f"pointer path does not exist on disk: {path}")
            with open(fp, encoding="utf-8") as f:
                self._cache[path] = json.load(f, parse_float=Decimal, parse_int=Decimal)
        return self._cache[path]

    def value(self, ref: PointerRef) -> Decimal:
        doc = self.doc(ref.path)
        val, _ = _rfc6901_walk(doc, ref.pointer)
        if isinstance(val, bool) or not isinstance(val, (Decimal, int)):
            raise ResolutionError(f"pointer does not resolve to a number: {ref.path}{ref.pointer}")
        return Decimal(val)

    def same_quantity_field(self, refs: list[PointerRef]) -> tuple[str, object] | None:
        """audit A1 + round-3 fix: whole-FILE search for a computed field
        describing the SAME quantity as `refs`' common leg/shape
        identifiers (rep suffix stripped) AND unit family — NOT limited to
        the operands' common-ancestor SUBTREE (the original walk stopped
        there and missed a field like cast-w1's own
        `/deltas/b8_s512_p50_ms/delta_ms` sitting in a SIBLING subtree of
        the operands' own `/legs/...`). Returns (pointer, value) of the
        first match; NUMBER and STRING leaves both count. Raises
        `PrecedenceUndecidable` — never returns `None` for this case —
        when the operands span more than one file or share no digit-
        bearing identifier token at all."""
        paths = {r.path for r in refs}
        if len(paths) != 1:
            raise PrecedenceUndecidable(
                f"precedence undecidable: operands span {len(paths)} different tracked "
                f"files ({sorted(paths)}) — point directly at a field or ledger this cell"
            )
        path = next(iter(paths))
        doc = self.doc(path)
        operand_pointers = {r.pointer for r in refs}
        operand_token_sets = [_path_digit_tokens(r.pointer) for r in refs]
        common = set.intersection(*operand_token_sets) if operand_token_sets else set()
        if not common:
            raise PrecedenceUndecidable(
                "precedence undecidable: the operands share no digit-bearing identifier "
                f"token to key a same-quantity search off of ({sorted(operand_pointers)}) — "
                "point directly at a field or ledger this cell"
            )
        operand_family = _operand_family(refs)
        for leaf_pointer, leaf_value in _iter_leaves(doc):
            if leaf_pointer in operand_pointers:
                continue
            last_seg = leaf_pointer.rsplit("/", 1)[-1]
            if not _COMPUTED_FIELD_RE.match(last_seg):
                continue
            if isinstance(leaf_value, bool):
                continue
            if not isinstance(leaf_value, (Decimal, int, str)):
                continue
            if not (common <= _path_digit_tokens(leaf_pointer)):
                continue
            candidate_family = _unit_family(last_seg)
            if operand_family is not None and candidate_family is not None and operand_family != candidate_family:
                continue
            return leaf_pointer, leaf_value
        return None


_COMPUTED_FIELD_RE = re.compile(r"^(delta|ratio|speedup).*$|.*_pct$|^spread.*$")


def leaves_of(expr_or_ref) -> list[PointerRef]:
    if isinstance(expr_or_ref, PointerRef):
        return [expr_or_ref]
    out: list[PointerRef] = []
    for a in expr_or_ref.args:
        out.extend(leaves_of(a))
    return out


def precedence_violation(expr: Expr, loader: Loader) -> str | None:
    """None, or the offending field's own json-pointer, for a
    diff/ratio/pct tag whose operands describe a quantity the pointed
    artifact ALSO states elsewhere via a delta*/ratio*/speedup*/*_pct/
    spread* field (number OR string) sharing the operands' identifiers."""
    if expr.kind not in ("diff", "ratio", "pct"):
        return None
    hit = loader.same_quantity_field(leaves_of(expr))
    return hit[0] if hit is not None else None


def eval_value(expr_or_ref, loader: Loader) -> Decimal:
    if isinstance(expr_or_ref, PointerRef):
        return loader.value(expr_or_ref)
    e = expr_or_ref
    if e.kind == "pointer":
        return loader.value(e.args[0])
    if e.kind in ("min", "mean", "max"):
        vals = [loader.value(p) for p in e.args]
        if e.kind == "min":
            return min(vals)
        if e.kind == "max":
            return max(vals)
        return sum(vals) / Decimal(len(vals))
    if e.kind == "diff":
        a, b = e.args
        return eval_value(a, loader) - eval_value(b, loader)
    if e.kind == "ratio":
        a, b = e.args
        return eval_value(a, loader) / eval_value(b, loader)
    if e.kind == "pct":
        a, b = e.args
        return (eval_value(a, loader) / eval_value(b, loader) - 1) * Decimal(100)
    raise ClaimParseError(f"unknown expr kind: {e.kind}")


def evaluate_expr(expr: Expr, loader: Loader) -> tuple[Decimal, bool]:
    """Returns (value, is_legacy). Applies the precedence check, the
    legacy() allowlist, sign negation, and the unit factor, in that order."""
    hit_pointer = precedence_violation(expr, loader)
    if hit_pointer is not None:
        field = hit_pointer.rsplit("/", 1)[-1]
        raise ClaimParseError(
            f"artifact states this quantity elsewhere at {hit_pointer} ({field}); "
            "bind directly to it or ledger this cell — a free aggregate is not legal here"
        )
    if expr.is_legacy:
        for leaf in leaves_of(expr):
            if (leaf.path, leaf.pointer) not in LEGACY_POINTER_ALLOWLIST:
                raise ClaimParseError(
                    f"legacy(...) is not allowlisted for {leaf.path}{leaf.pointer} "
                    "(LEGACY_POINTER_ALLOWLIST is closed — add the exact cell by hand)"
                )
    val = eval_value(Expr(expr.kind, expr.args), loader)
    if expr.is_neg:
        val = -val
    if expr.unit:
        val = val * _UNIT_FACTORS[expr.unit]
    return val, expr.is_legacy


# --- token <-> Decimal string equality --------------------------------------


def normalize_token_for_compare(token_text: str) -> tuple[str, int]:
    """Unicode-minus -> ascii hyphen, and a leading '+' dropped (Decimal's
    own str() never emits one) — SIGN is otherwise preserved, so a token's
    sign must match the (possibly `neg()`-flipped) evaluated value's own
    sign exactly; there is no longer a sign-discarding comparison mode."""
    t = token_text.replace("−", "-").lstrip("+")
    try:
        d = Decimal(t)
    except InvalidOperation as exc:
        raise ClaimParseError(f"not a valid number: {token_text!r}") from exc
    exp = d.as_tuple().exponent
    places = -exp if isinstance(exp, int) and exp < 0 else 0
    return t, places


def quantize_str(value: Decimal, places: int) -> str:
    quantum = Decimal(1).scaleb(-places) if places else Decimal(1)
    q = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    return str(q)


def compare_token(token_text: str, value: Decimal) -> tuple[bool, str]:
    norm, places = normalize_token_for_compare(token_text)
    got = quantize_str(value, places)
    return got == norm, str(value)


# --- allowlist (shape copied from check_doc_numbers_have_producers.py) -----


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def line_hash(text: str) -> str:
    return hashlib.sha1(normalize_line(text).encode("utf-8")).hexdigest()


def parse_allowlist_text(text: str) -> set[tuple[str, str, str, str]]:
    """`file:token:sha1:col` quadruples (audit advisory: `col` makes the
    key injective — two identical tokens on one line, e.g. `112 / 112`,
    used to collapse into a single allowlist entry covering both)."""
    entries: set[tuple[str, str, str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(":", 3)
        if len(parts) != 4:
            continue
        entries.add((parts[0], parts[1], parts[2], parts[3]))
    return entries


def load_allowlist() -> set[tuple[str, str, str, str]]:
    if not ALLOWLIST_PATH.exists():
        return set()
    return parse_allowlist_text(ALLOWLIST_PATH.read_text())


def check_classification_file(allowlist: set[tuple[str, str, str, str]]) -> list[str]:
    """audit advisory: CLASSIFICATION_PATH/CLASSIFICATION_REASONS were
    previously unread constants. Every allowlist entry must have EXACTLY
    one row in CLASSIFICATION_PATH (matched by file+token+sha1+col) whose
    reason is in the closed set; a missing or malformed row, a duplicate,
    or an orphaned classification row with no matching allowlist entry are
    all named problems."""
    problems: list[str] = []
    if not CLASSIFICATION_PATH.exists():
        return [f"{CLASSIFICATION_PATH.name} does not exist but the allowlist has entries"]
    row_re = re.compile(
        r"^\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|\s*([A-Za-z-]+)\s*\|"
    )
    seen: dict[tuple[str, str, str, str], int] = {}
    for line in CLASSIFICATION_PATH.read_text().splitlines():
        m = row_re.match(line.strip())
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3), m.group(4))
        reason = m.group(5)
        seen[key] = seen.get(key, 0) + 1
        if reason not in CLASSIFICATION_REASONS:
            problems.append(f"classification row for {key} has an unrecognized reason: {reason!r}")
    for key, count in seen.items():
        if count > 1:
            problems.append(f"classification row for {key} appears {count} times (expected 1)")
    for entry in allowlist:
        if entry not in seen:
            problems.append(f"allowlist entry {entry} has no classification row")
    for key in seen:
        if key not in allowlist:
            problems.append(f"classification row for {key} has no matching allowlist entry")
    return problems


# --- scanning / classification ----------------------------------------------


@dataclass
class Finding:
    file: str
    line_no: int
    col: int
    token: str
    reason: str


@dataclass
class TableStats:
    file: str
    header_line_no: int
    excluded: int = 0
    v: int = 0
    v_legacy: int = 0
    ledger: int = 0
    findings: int = 0


def _entries_for_row(lines: list[str], row_line_no: int) -> list[str] | None:
    """`row_line_no` is 1-indexed. Returns the parsed `c1..cN` expr strings
    from the claims comment on the line immediately above, or None if there
    is no such comment."""
    above_idx = row_line_no - 2  # 0-indexed line immediately above
    if above_idx < 0:
        return None
    m = _COMMENT_RE.match(lines[above_idx])
    if not m:
        return None
    body = m.group("body")
    entries: dict[int, str] = {}
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        km = re.match(r"^c(\d+)\s*=\s*(.*)$", part)
        if not km:
            continue
        entries[int(km.group(1))] = km.group(2).strip()
    if not entries:
        return []
    n = max(entries)
    return [entries.get(i, "") for i in range(1, n + 1)]


def scan_table(
    lines: list[str], table: Table, loader: Loader, allowlist: set[tuple[str, str, str, str]]
) -> tuple[TableStats, list[Finding]]:
    stats = TableStats(table.file, table.header_line_no)
    findings: list[Finding] = []
    for row_line_no in table.body_line_nos:
        line = lines[row_line_no - 1]
        tokens = find_tokens_in_row(line, row_line_no)
        row_excluded = len(NUMBER_RE.findall(line)) - len(tokens)
        stats.excluded += max(row_excluded, 0)
        if not tokens:
            continue
        entries = _entries_for_row(lines, row_line_no)
        if entries is None or len(entries) != len(tokens):
            for tok in tokens:
                findings.append(
                    Finding(
                        table.file,
                        tok.line_no,
                        tok.col,
                        tok.text,
                        f"uncovered numeric token (expected a claims comment above the row "
                        f"with {len(tokens)} entr{'y' if len(tokens) == 1 else 'ies'}, "
                        f"found {0 if entries is None else len(entries)})",
                    )
                )
            stats.findings += len(tokens)
            continue
        for tok, entry in zip(tokens, entries):
            if entry == "ledger":
                key = (table.file, tok.text, line_hash(tok.line_text), str(tok.col))
                if key not in allowlist:
                    findings.append(
                        Finding(
                            table.file,
                            tok.line_no,
                            tok.col,
                            tok.text,
                            "marked `ledger` but has no entry in "
                            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} keyed "
                            f"{':'.join(key)}",
                        )
                    )
                    stats.findings += 1
                else:
                    stats.ledger += 1
                continue
            try:
                expr = parse_expr(entry)
                value, is_legacy = evaluate_expr(expr, loader)
            except (ClaimParseError, ResolutionError) as exc:
                findings.append(Finding(table.file, tok.line_no, tok.col, tok.text, str(exc)))
                stats.findings += 1
                continue
            ok, full_precision = compare_token(tok.text, value)
            if not ok:
                findings.append(
                    Finding(
                        table.file,
                        tok.line_no,
                        tok.col,
                        tok.text,
                        f"mismatch: doc prints {tok.text!r}, expression `{entry}` evaluates to "
                        f"{full_precision} (full precision)",
                    )
                )
                stats.findings += 1
            elif is_legacy:
                stats.v_legacy += 1
            else:
                stats.v += 1
    return stats, findings


def scan_tree() -> tuple[list[TableStats], list[Finding], list[str]]:
    """Returns (per-table stats, per-token findings, scope-invariant
    problems — audit A3 — which the caller folds into the gate's exit)."""
    loader = Loader(tracked=tracked_files())
    allowlist = load_allowlist()
    all_stats: list[TableStats] = []
    all_findings: list[Finding] = []
    scope_problems: list[str] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            rel = str(path.relative_to(REPO_ROOT))
            lines = path.read_text(errors="ignore").splitlines()
            header_hits: dict[str, int] = {}
            tables = find_tables(rel, lines, header_hits)
            scope_problems.extend(check_known_table_usage(rel, header_hits))
            if rel in EXPECTED_TABLE_COUNTS and len(tables) != EXPECTED_TABLE_COUNTS[rel]:
                scope_problems.append(
                    f"{rel}: {len(tables)} in-scope table(s), expected "
                    f"{EXPECTED_TABLE_COUNTS[rel]} (EXPECTED_TABLE_COUNTS) — a table fell out "
                    "of (or into) scope; update the pinned constant only after confirming why"
                )
            for table in tables:
                stats, findings = scan_table(lines, table, loader, allowlist)
                all_stats.append(stats)
                all_findings.extend(findings)
    scope_problems.extend(check_classification_file(allowlist))
    denom = token_denominator(all_stats)
    if denom != EXPECTED_DENOMINATOR:
        scope_problems.append(
            f"tokenizer denominator is {denom}, expected {EXPECTED_DENOMINATOR} "
            "(EXPECTED_DENOMINATOR) — update the pinned constant only after confirming why "
            "(a new number, a retag, or an exclusion-class change)"
        )
    return all_stats, all_findings, scope_problems


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


# --- report / gate -----------------------------------------------------


def token_denominator(stats: list[TableStats]) -> int:
    return sum(s.v + s.v_legacy + s.ledger + s.findings for s in stats)


def report(stats: list[TableStats], findings: list[Finding], argv: list[str]) -> int:
    total_v = sum(s.v for s in stats)
    total_v_legacy = sum(s.v_legacy for s in stats)
    total_ledger = sum(s.ledger for s in stats)
    total_excluded = sum(s.excluded for s in stats)
    total_findings = sum(s.findings for s in stats)
    denom = token_denominator(stats)
    fully_bound = 0
    print("check_perf_claims --report")
    print(f"{'table':50s} {'excl':>5s} {'V':>5s} {'V-leg':>6s} {'L':>5s} {'find':>5s}")
    for s in stats:
        print(
            f"{s.file + ':' + str(s.header_line_no):50s} {s.excluded:5d} {s.v:5d} "
            f"{s.v_legacy:6d} {s.ledger:5d} {s.findings:5d}"
        )
        if s.findings == 0 and (s.v + s.v_legacy) > 0 and s.ledger == 0:
            fully_bound += 1
    print("-" * 80)
    print(f"tokenizer count (denominator): {denom}")
    print(f"excluded: {total_excluded}")
    print(f"V: {total_v}   V-legacy: {total_v_legacy}   ledger: {total_ledger}   findings: {total_findings}")
    if denom:
        print(f"bind rate (V+V-legacy)/denominator: {100 * (total_v + total_v_legacy) / denom:.1f}%")
    print(f"fully-bound tables (0 findings, 0 ledger, >0 tokens): {fully_bound}")

    min_fully_bound = None
    min_v = None
    for a in argv:
        if a.startswith("--min-fully-bound="):
            min_fully_bound = int(a.split("=", 1)[1])
        if a.startswith("--min-v="):
            min_v = int(a.split("=", 1)[1])
    exit_code = 0
    if min_fully_bound is not None and fully_bound < min_fully_bound:
        print(
            f"F1 RATCHET FAIL: {fully_bound} fully-bound table(s) < --min-fully-bound={min_fully_bound}",
            file=sys.stderr,
        )
        exit_code = 1
    if min_v is not None and total_v < min_v:
        print(f"F1 RATCHET FAIL: V={total_v} < --min-v={min_v}", file=sys.stderr)
        exit_code = 1
    return exit_code


def gate(stats: list[TableStats], findings: list[Finding], scope_problems: list[str]) -> int:
    if not findings and not scope_problems:
        denom = token_denominator(stats)
        print(
            f"check_perf_claims: OK — {denom} numeric token(s) across {len(stats)} in-scope "
            "table(s), every one excluded, bound, or ledgered; scope invariants (A3) hold."
        )
        return 0
    print("check_perf_claims: FAIL", file=sys.stderr)
    for p in scope_problems:
        print(f"  SCOPE: {p}", file=sys.stderr)
    for f in findings:
        print(f"  {f.file}:{f.line_no}:{f.col}: {f.token!r}: {f.reason}", file=sys.stderr)
    print(
        "\ncheck_perf_claims: either a scope invariant broke (a KNOWN_TABLES header renamed/"
        "vanished/duplicated, or the pinned table-count/denominator constant is stale) or a "
        "numeric token in an in-scope performance table is not excluded, not bound by a "
        "resolving `claims:` tag entry, and not escaped via `ledger` + a committed allowlist "
        "entry. Tag it, ledger it with a classification row, or fix the pinned constant after "
        "confirming why it moved — never widen the grammar past the six forms.",
        file=sys.stderr,
    )
    return 1


def check_allowlist_only_shrinks() -> int:
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print("perf-claims-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(
            f"  git fetch origin main failed (exit {exc.returncode}): "
            f"{exc.stderr.strip() if exc.stderr else '<no stderr>'}",
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
        print("perf-claims-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(f"  origin/main does not resolve: {rev.stderr.strip()}", file=sys.stderr)
        return 1

    rel = ALLOWLIST_PATH.relative_to(REPO_ROOT).as_posix()
    file_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"origin/main:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    current = load_allowlist()

    if not file_exists:
        print(
            "perf-claims-allowlist-only-shrinks: OK (bootstrap) — origin/main resolves "
            f"({rev.stdout.strip()}) but has no {rel} yet: this branch's {len(current)} "
            "entries establish the baseline."
        )
        return 0

    base_text = _run(["git", "show", f"origin/main:{rel}"])
    base = parse_allowlist_text(base_text)
    added = current - base
    if added:
        print("perf-claims-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for entry in sorted(added):
            print(f"  + {':'.join(entry)}", file=sys.stderr)
        print(
            "\nperf-claims-allowlist-only-shrinks: this branch adds a NEW entry. The "
            "allowlist tracks pre-existing debt and may never grow — bind the token to a "
            "real producer instead.",
            file=sys.stderr,
        )
        return 1

    print(
        f"perf-claims-allowlist-only-shrinks: OK — {len(current)} entries "
        f"({len(base) - len(current)} shrunk vs origin/main)."
    )
    return 0


# --- self-test ---------------------------------------------------------


def _mem_loader(doc: dict) -> Loader:
    loader = Loader(tracked=None)
    loader._check_path = lambda path: None  # type: ignore[method-assign]
    loader._cache["fixture.json"] = doc
    return loader


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(f"self-test FAILED: {name}")

    # --- (A1) whole-artifact same-quantity precedence, reproducing the
    # real cast-w1 shape: operands under /legs/..., the computed field
    # lives at /deltas/... — a SIBLING subtree, not an ancestor. The OLD
    # common-ancestor-only walk missed this; the audit's end-to-end probe
    # found it binding -38.5 beside an unread delta_ms=39.565658. ---
    cast_w1_doc = {
        "legs": {
            "b8_s512_fused_r1": {"s_per_step_p50": Decimal("0.634850336")},
            "b8_s512_fused_r2": {"s_per_step_p50": Decimal("0.635428075")},
            "b8_s512_disabled_r1": {"s_per_step_p50": Decimal("0.674415994")},
            "b8_s512_disabled_r2": {"s_per_step_p50": Decimal("0.672819099")},
        },
        "deltas": {
            "b8_s512_p50_ms": {
                "fused_r1": Decimal("634.850336"),
                "disabled_r1": Decimal("674.415994"),
                "delta_ms": Decimal("39.565658"),
            }
        },
    }
    loader_cw1 = _mem_loader(cast_w1_doc)
    end_to_end_expr = parse_expr(
        "neg(diff(mean(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_disabled_r2/s_per_step_p50),"
        "mean(fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r2/s_per_step_p50))) as ms"
    )
    try:
        evaluate_expr(end_to_end_expr, loader_cw1)
        failures.append(
            "self-test FAILED (A1 end-to-end reproduction): the real cast-w1 shape "
            "(operands under /legs, computed field under a sibling /deltas) did not raise"
        )
    except ClaimParseError as exc:
        check("A1 end-to-end: violation names the sibling-subtree field", "delta_ms" in str(exc))

    # direct pointer AT the flagged field is exempt (bare pointer, not a
    # diff/ratio/pct tag at all).
    direct_expr = parse_expr("neg(fixture.json#/deltas/b8_s512_p50_ms/delta_ms)")
    direct_val, _ = evaluate_expr(direct_expr, loader_cw1)
    check("direct pointer to the computed field is legal", direct_val == Decimal("-39.565658"))

    # (A1) a STRING-valued same-quantity field is ALSO a finding — never a
    # blessed pass (this INVERTS the pre-audit self-test, which wrongly
    # asserted a string sibling does not block).
    doc2 = {
        "peak": {
            "b8_s512": {
                "fused_r1_gb": Decimal("16.0"),
                "fused_r2_gb": Decimal("16.2"),
                "disabled_r1_gb": Decimal("15.9"),
                "disabled_r2_gb": Decimal("15.9"),
                "delta_gb": "+0.1 to +0.3 GB",
            }
        }
    }
    loader2 = _mem_loader(doc2)
    expr3 = parse_expr(
        "diff(mean(fixture.json#/peak/b8_s512/fused_r1_gb,fixture.json#/peak/b8_s512/fused_r2_gb),"
        "mean(fixture.json#/peak/b8_s512/disabled_r1_gb,fixture.json#/peak/b8_s512/disabled_r2_gb))"
    )
    try:
        evaluate_expr(expr3, loader2)
        failures.append(
            "self-test FAILED (A1): a STRING-valued delta_gb sibling did not block a free diff "
            "— it must be a finding ('bind two tokens or ledger it'), never a silent pass"
        )
    except ClaimParseError as exc:
        check("A1: string-valued same-quantity field named in the finding", "delta_gb" in str(exc))

    # (round-3 audit) a quantity with NO shared digit-identifier across
    # operands is UNDECIDABLE, not silently clean — FAILS LOUDLY rather
    # than returning None (an earlier revision let this bind freely).
    doc4 = {"a": {"x1": Decimal("1"), "delta_x": Decimal("9")}, "b": {"y2": Decimal("2")}}
    loader4 = _mem_loader(doc4)
    expr4 = parse_expr("diff(fixture.json#/a/x1,fixture.json#/b/y2)")
    try:
        evaluate_expr(expr4, loader4)
        failures.append(
            "self-test FAILED (round-3, digit-free-leg): operands sharing no digit-bearing "
            "identifier token must be UNDECIDABLE (a finding), not a silent free bind"
        )
    except PrecedenceUndecidable as exc:
        check("digit-free-leg case fails loudly and says 'undecidable'", "undecidable" in str(exc))
    except ClaimParseError:
        failures.append(
            "self-test FAILED (round-3, digit-free-leg): raised a plain ClaimParseError, "
            "not the more specific PrecedenceUndecidable"
        )

    # (round-3 audit) operands spanning more than one tracked file are
    # ALSO undecidable, not silently clean.
    loader4b = _mem_loader({"v": Decimal("1")})
    loader4b._cache["other.json"] = {"w": Decimal("2")}
    expr4b = parse_expr("diff(fixture.json#/v,other.json#/w)")
    try:
        evaluate_expr(expr4b, loader4b)
        failures.append("self-test FAILED (round-3, multi-file): cross-file operands did not raise")
    except PrecedenceUndecidable as exc:
        check("multi-file operands fail loudly and say 'undecidable'", "undecidable" in str(exc))

    # --- (round-3 audit, THE CLASS FIX) a SAME-REP operand pair — both
    # named `r1` — must still be caught. Reproduces the auditor's
    # end-to-end probe: pre-fix, diff(disabled_r1, fused_r1) bound
    # 39.565658 bit-for-bit as a free diff because `r1`, shared by BOTH
    # operands, survived the (un-stripped) intersection and over-
    # constrained the search past delta_ms (whose own path carries no rep
    # suffix at all). ---
    same_rep_doc = {
        "legs": {
            "b8_s512_disabled_r1": {"s_per_step_p50": Decimal("0.674415994")},
            "b8_s512_fused_r1": {"s_per_step_p50": Decimal("0.634850336")},
        },
        "deltas": {
            "b8_s512_p50_ms": {
                "delta_ms": Decimal("39.565658"),
                "delta_pct": Decimal("6.232"),
            }
        },
    }
    loader_same_rep = _mem_loader(same_rep_doc)
    same_rep_diff_expr = parse_expr(
        "diff(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50)"
    )
    try:
        val, _ = evaluate_expr(same_rep_diff_expr, loader_same_rep)
        failures.append(
            f"self-test FAILED (round-3, same-rep diff): bound FREELY to {val} — the exact "
            "class the audit's end-to-end probe found live on 5c8eef5"
        )
    except ClaimParseError as exc:
        check(
            "same-rep diff (both r1) is still caught — names a delta*/*_pct field",
            "delta" in str(exc),
        )
    same_rep_pct_expr = parse_expr(
        "pct(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50)"
    )
    try:
        evaluate_expr(same_rep_pct_expr, loader_same_rep)
        failures.append("self-test FAILED (round-3, same-rep pct): bound freely, not caught")
    except ClaimParseError as exc:
        check("same-rep pct (both r1) is also caught", "delta" in str(exc))

    # (round-3 audit) a multi-computed-field artifact: the search must
    # name WHICH field it found (not just "a" field) — and the UNIT-FAMILY
    # veto must pick the family-consistent one, never a same-shape field
    # of the WRONG family (a `_ms` operand must never be blocked by a
    # `_gb` field of the identical shape, and vice versa).
    multi_field_doc = {
        "legs": {
            "b8_s512_disabled_r1": {"s_per_step_p50": Decimal("0.674415994")},
            "b8_s512_fused_r1": {"s_per_step_p50": Decimal("0.634850336")},
            "b8_s512_disabled_r1_gb": {"peak_vram_bytes": Decimal("1000")},
            "b8_s512_fused_r1_gb": {"peak_vram_bytes": Decimal("900")},
        },
        "deltas": {
            "b8_s512_p50_ms": {"delta_ms": Decimal("39.565658")},
            "b8_s512_p50_gb": {"delta_gb": Decimal("100")},
        },
    }
    loader_multi = _mem_loader(multi_field_doc)
    ms_operand_expr = parse_expr(
        "diff(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50)"
    )
    try:
        evaluate_expr(ms_operand_expr, loader_multi)
        failures.append("self-test FAILED (round-3, multi-field): ms-family operand pair not caught")
    except ClaimParseError as exc:
        check(
            "multi-field artifact: an s_per_step (ms-family) operand pair is matched to "
            "delta_ms, never the same-shape delta_gb field",
            "delta_ms" in str(exc) and "delta_gb" not in str(exc),
        )
    gb_operand_expr = parse_expr(
        "diff(fixture.json#/legs/b8_s512_disabled_r1_gb/peak_vram_bytes,"
        "fixture.json#/legs/b8_s512_fused_r1_gb/peak_vram_bytes)"
    )
    try:
        evaluate_expr(gb_operand_expr, loader_multi)
        failures.append("self-test FAILED (round-3, multi-field): gb-family operand pair not caught")
    except ClaimParseError as exc:
        check(
            "multi-field artifact: a peak_vram_bytes (bytes-family) operand pair is matched "
            "to delta_gb, never the same-shape delta_ms field",
            "delta_gb" in str(exc) and "delta_ms" not in str(exc),
        )

    # (ii) Decimal + ROUND_HALF_EVEN string equality.
    loader3 = _mem_loader({"v": Decimal("1.265")})
    ok, _ = compare_token("1.2650", eval_value(Expr("pointer", (PointerRef("fixture.json", "/v"),)), loader3))
    check("Decimal string-equality binds a trailing zero (1.2650 == 1.265)", ok)
    half_even = quantize_str(Decimal("0.125"), 2)
    check("ROUND_HALF_EVEN on an exact .5 boundary rounds to the even digit", half_even == "0.12")

    # (iii) an expression outside the six forms is a parse error.
    try:
        parse_expr("sqrt(fixture.json#/v)")
        failures.append("self-test FAILED: unsupported form did not raise ClaimParseError")
    except ClaimParseError:
        pass

    # (iv) an uncovered token is a finding with file:line:col.
    lines = ["| shape | s/step |", "|---|---|", "| b8s512 | 0.421 |"]
    table = Table("t.md", 1, lines[0], [3])
    stats, findings = scan_table(lines, table, Loader(tracked=set()), set())
    check("uncovered token yields exactly one finding", len(findings) == 1)
    if findings:
        check("finding carries file:line:col", findings[0].line_no == 3 and findings[0].col >= 0)

    # (v) shape/version/PR/date/seed/section/formula/type-name/product-name
    # tokens are excluded, not findings, even with no claims comment.
    lines5 = [
        "| shape | s/step |",
        "|---|---|",
        "| b8·s512 | see #377, cu126, 2.13.0+cu126, 2026-08-23, s2:89, row 5, "
        "seed 42, §6, 1/√d, Op2/3, FlashAttention-2 |",
    ]
    table5 = Table("t.md", 1, lines5[0], [3])
    stats5, findings5 = scan_table(lines5, table5, Loader(tracked=set()), set())
    check("shape/version/ref/date/ledger-cite/seed/section/formula/type/product tokens all excluded", len(findings5) == 0)
    raw_numbers = len(NUMBER_RE.findall(lines5[2]))
    check(
        "every numeric token on the all-excluded line is accounted for as excluded",
        stats5.excluded == raw_numbers,
    )

    # unified rows-list pattern: a bare "row 5, 6" list excludes BOTH
    # numbers now (audit advisory), not just the first.
    lines5b = ["| shape | s/step |", "|---|---|", "| x | see row 5, 6 for detail |"]
    table5b = Table("t.md", 1, lines5b[0], [3])
    stats5b, findings5b = scan_table(lines5b, table5b, Loader(tracked=set()), set())
    check("unified rows-list pattern excludes every number in a bare 'row N, M' list", len(findings5b) == 0)

    # layer/tensor/launch counts are explicitly NOT excluded.
    lines6 = ["| kernel | launches |", "|---|---|", "| badd | 563 launches |"]
    table6 = Table("t.md", 1, lines6[0], [3])
    _, findings6 = scan_table(lines6, table6, Loader(tracked=set()), set())
    check("a launch count is NOT excluded (still needs a tag)", len(findings6) == 1)

    # (vi) a pointer into a .json.raw payload is refused.
    loader_raw = Loader(tracked={"crates/jammi-kernels/artifacts/cuda-runs/x-raw-runs/leg.json.raw"})
    try:
        loader_raw.value(
            PointerRef("crates/jammi-kernels/artifacts/cuda-runs/x-raw-runs/leg.json.raw", "/v")
        )
        failures.append("self-test FAILED: .json.raw pointer did not raise")
    except ResolutionError as exc:
        check("json.raw pointer refusal names it unprovenanced", "unprovenanced" in str(exc))

    try:
        Loader(tracked={"docs/maintainer/foo.json"}).value(PointerRef("docs/maintainer/foo.json", "/v"))
        failures.append("self-test FAILED: out-of-root pointer did not raise")
    except ResolutionError as exc:
        check("out-of-root pointer refusal names the root rule", "allowed root" in str(exc))

    try:
        Loader(tracked=set()).value(
            PointerRef("crates/jammi-bench/baselines/__self_test_untracked__.json", "/v")
        )
        failures.append("self-test FAILED: untracked pointer did not raise (or file existed)")
    except ResolutionError as exc:
        check(
            "untracked pointer refusal fires on tracked-set membership, not disk existence",
            "tracked" in str(exc) or "does not exist" in str(exc),
        )

    # neg() sign-preserving compare (REPLACES the old abs()): a positive
    # raw artifact value negates to match a printed negative token, and a
    # WRONG-signed token now correctly mismatches.
    loader7 = _mem_loader({"delta": Decimal("39.565658")})
    val7, _ = evaluate_expr(parse_expr("neg(fixture.json#/delta)"), loader7)
    ok7, _ = compare_token("−39.6", val7)
    check("neg() binds a unicode-minus-prefixed doc token to -raw_value", ok7)
    wrong_sign_ok, _ = compare_token("+39.6", val7)
    check("neg(): a WRONG-signed token (would have passed under abs()) now mismatches", not wrong_sign_ok)

    # legacy() is restricted to the closed allowlist.
    loader8 = _mem_loader({"v": Decimal("16.3959")})
    try:
        evaluate_expr(parse_expr("legacy(fixture.json#/v)"), loader8)
        failures.append("self-test FAILED: legacy() on a non-allowlisted pointer did not raise")
    except ClaimParseError as exc:
        check("legacy() refusal names the allowlist", "allowlist" in str(exc).lower())
    allowlisted_ref = next(iter(LEGACY_POINTER_ALLOWLIST))
    loader8b = _mem_loader({})
    loader8b._cache = {}
    # build a doc matching the allowlisted path's own pointer shape
    walk_doc: dict = {}
    cur = walk_doc
    segs = allowlisted_ref[1].lstrip("/").split("/")
    for seg in segs[:-1]:
        cur[seg] = {}
        cur = cur[seg]
    cur[segs[-1]] = Decimal("1.5")
    loader8b._cache[allowlisted_ref[0]] = walk_doc
    val8b, is_legacy8b = evaluate_expr(
        parse_expr(f"legacy({allowlisted_ref[0]}#{allowlisted_ref[1]})"), loader8b
    )
    check("legacy() on the closed allowlist evaluates normally", val8b == Decimal("1.5"))
    check("legacy() on the closed allowlist reports V-legacy", is_legacy8b is True)

    # (vii) mismatch prints the evaluated value at full precision.
    loader9 = _mem_loader({"v": Decimal("16.408")})
    ok9, full9 = compare_token("16.5", eval_value(Expr("pointer", (PointerRef("fixture.json", "/v"),)), loader9))
    check("mismatch is detected", ok9 is False)
    check("mismatch reports the FULL precision evaluated value, not the rounded one", full9 == "16.408")

    # `ledger` marker requires a matching allowlist entry, now keyed
    # file:token:sha1:col (audit advisory: injective on column).
    lines10 = ["| shape | s/step |", "|---|---|", "| b8s128 | -16.5 |"]
    lines10.insert(2, "<!-- claims: c1=ledger -->")
    table10 = Table("t.md", 1, lines10[0], [4])
    _, findings10 = scan_table(lines10, table10, Loader(tracked=set()), set())
    check("`ledger` with no allowlist entry is a finding", len(findings10) == 1)
    tok_col = lines10[3].index("-16.5")
    key10 = ("t.md", "-16.5", line_hash(lines10[3]), str(tok_col))
    _, findings10b = scan_table(lines10, table10, Loader(tracked=set()), {key10})
    check("`ledger` with a matching allowlist entry binds clean", len(findings10b) == 0)

    # injective key: two identical tokens on one line get two distinct
    # allowlist entries (different columns), and only listing one still
    # leaves the other uncovered.
    lines11 = ["| a | b |", "|---|---|", "| 112 / 112 | x |"]
    lines11.insert(2, "<!-- claims: c1=ledger; c2=ledger -->")
    table11 = Table("t.md", 1, lines11[0], [4])
    col1 = lines11[3].index("112")
    col2 = lines11[3].index("112", col1 + 1)
    key11a = ("t.md", "112", line_hash(lines11[3]), str(col1))
    _, findings11_partial = scan_table(lines11, table11, Loader(tracked=set()), {key11a})
    check("an injective key: allowlisting only the first '112' still flags the second", len(findings11_partial) == 1)
    key11b = ("t.md", "112", line_hash(lines11[3]), str(col2))
    _, findings11_full = scan_table(lines11, table11, Loader(tracked=set()), {key11a, key11b})
    check("both distinct-column keys allowlisted clears both", len(findings11_full) == 0)

    # (A3-S1) KNOWN_TABLES exactly-one-match invariant: renaming one
    # table's header must FAIL the usage check.
    renamed_header = "| lever | RENAMED COLUMN | measured (same box, SXM4) | verdict |"
    header_hits_s1: dict[str, int] = {}
    fake_lines = [
        renamed_header,
        "|---|---|---|---|",
        "| x | y | z | w |",
    ]
    find_tables("docs/maintainer/fine-tune-performance-guide.md", fake_lines, header_hits_s1)
    problems_s1 = check_known_table_usage(
        "docs/maintainer/fine-tune-performance-guide.md", header_hits_s1
    )
    check(
        "A3-S1: renaming a KNOWN_TABLES header is caught as 'never matched'",
        any("never matched" in p and "measured (same box, SXM4)" in p for p in problems_s1),
    )

    # a duplicated header is also caught.
    header_hits_dup: dict[str, int] = {}
    dup_lines = [
        "| pair | mean cosine | notes |",
        "|---|---|---|",
        "| a | 0.1 | n |",
        "",
        "| pair | mean cosine | notes |",
        "|---|---|---|",
        "| b | 0.2 | n |",
    ]
    find_tables("docs/maintainer/fine-tune-performance-guide.md", dup_lines, header_hits_dup)
    problems_dup = check_known_table_usage(
        "docs/maintainer/fine-tune-performance-guide.md", header_hits_dup
    )
    check(
        "A3: a duplicated KNOWN_TABLES header is caught as 'matched N times'",
        any("matched 2 times" in p for p in problems_dup),
    )

    # a citation table (header contains both 'rule' and 'escape') is
    # excluded even if it happens to contain a trigger word too.
    check(
        "citation-table header (rule + escape) is recognized",
        is_citation_table_header("| # | rule | guide | KO | the escape that paid for it |"),
    )
    citation_lines = [
        "| # | rule | guide | KO | ms escape that paid for it |",  # 'ms' trigger present too
        "|---|---|---|---|---|",
        "| 1 | Two references | §3.3 | pending | some note |",
    ]
    citation_tables = find_tables("t.md", citation_lines)
    check("a citation-table header stays OUT of scope even with a trigger substring present", citation_tables == [])

    # classification-file verification: an allowlist entry with no
    # classification row is a named problem.
    problems_cls = check_classification_file({("x.md", "9", "deadbeef", "0")})
    check(
        "an allowlist entry missing its classification row is a named problem",
        any("has no classification row" in p for p in problems_cls),
    )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check_perf_claims self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "check_perf_claims self-test: OK — whole-artifact same-quantity precedence (A1, incl. "
        "the real cast-w1 sibling-subtree reproduction and the string-field finding), Decimal/"
        "ROUND_HALF_EVEN string equality, the six-form grammar's parse-error refusal, "
        "uncovered-token findings with file:line:col, the extended lexical exclusion class "
        "(seed/section/formula/type-name/product-name, plus the unified rows-list pattern and "
        "the layer/launch-count non-exclusion), the .json.raw/out-of-root/untracked pointer "
        "refusals, neg()'s sign-preserving compare, legacy()'s closed allowlist, the injective "
        "file:token:sha1:col allowlist key, the KNOWN_TABLES exactly-one-match invariant (A3-S1 "
        "rename + duplicate), the citation-table header exclusion, the classification-file "
        "verification, the round-3 same-rep diff/pct fix (rep-suffix stripped before "
        "intersecting; reproduces and closes the audit's live 5c8eef5 free-bind of 39.565658), "
        "the multi-computed-field unit-family veto (names the family-consistent field, never "
        "a same-shape field of the wrong family), and the multi-file/digit-free "
        "PrecedenceUndecidable fail-loud (never a silent None) all confirmed."
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()
    stats, findings, scope_problems = scan_tree()
    if "--report" in argv:
        exit_code = report(stats, findings, argv)
        if findings or scope_problems:
            for p in scope_problems:
                print(f"SCOPE: {p}", file=sys.stderr)
            for f in findings:
                print(f"{f.file}:{f.line_no}:{f.col}: {f.token!r}: {f.reason}", file=sys.stderr)
            exit_code = 1
        return exit_code
    return gate(stats, findings, scope_problems)


if __name__ == "__main__":
    sys.exit(main())
