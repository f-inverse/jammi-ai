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

ARTIFACT-FIELD PRECEDENCE (P3, C10.3; VALUE-based, round-4 audit rewrite,
round-5 audit fixes). Two token-based mechanisms were both found
live-broken by an adversarial sweep of the real tracked tree: a
token-subset test always required the CANDIDATE field to be at least as
specific as the OPERANDS, but a real computed field is almost always LESS
specific (`delta_ms` carries no rep suffix; the operands that produce it
do). Identity/token reasoning is abandoned entirely. The mechanism is
VALUE-based: for EVERY non-pointer node in a tag's expr tree — round-5
fix, class A: not only an outermost `diff`/`ratio`/`pct`; a top-level OR
NESTED `min`/`mean`/`max` is ITSELF a free computation (`mean` is
literally the estimator-shopping form this module doc has always named)
and is checked independently at every level, innermost first
(`_non_pointer_subexprs`) — evaluate that node's own FREE result (its raw
value over ITS OWN operands, in their native unit, before the OUTERMOST
tag's `neg`/`as unit`/`legacy` wrapping, which only ever applies to the
root); if all its operand leaves share one tracked file, scan the WHOLE
FILE for every NUMBER leaf whose OWN key matches `delta*`/`ratio*`/
`speedup*`/`*_pct`/`spread*` (`_COMPUTED_FIELD_RE`) and test whether that
leaf's value equals the free result — same sign, opposite sign, or after a
`×1000`/`÷1000`/`÷1e9`/`×1e9` unit rescale (`s<->ms`, `bytes<->GB` — round-5
advisory: the prior scale set was asymmetric, missing `×1e9`) — within a
5e-4 RELATIVE tolerance (`_value_matches_any_scale`/`_values_match`, THE
PRODUCTION matcher) — EXCLUDING the operand leaves themselves (`same_quantity_field`'s
own `operand_pointers` check), so a bare pointer's tag can never flag itself as a
'match' against its own value. ANY match is a FINDING naming the matched field's own
pointer, REGARDLESS of which two leaves produced either number — a
coincidental match is still reported, on purpose, so a human decides
"point at it" or "ledger this cell, it's a coincidence," never a silent
mechanical exemption. Operands spanning more than one tracked file are
UNDECIDABLE (a FINDING). A STRING-valued same-family field can never be
value-checked (free text) — round-5 fix, class A: "the ONE live case" was
false (three exist: cast-w1's two `delta_gb` rows and flash-arm-encoder-
oracle's), so this is now closed by a CLOSED REGISTRY
(`STRING_FIELDS_PATH`, `check_string_field_registry` — REDs on any
unregistered string-valued computed field in the tracked tree, part of
the bare run) plus a registry-object check inside `same_quantity_field`:
a free expression whose operands live under a REGISTERED string field's
own parent object is a finding too, even with no numeric twin to rescue
it. A zero denominator in `ratio`/`pct` (1,390 zero-valued leaves live in
the tracked tree) is a `ClaimParseError` naming the offending pointer —
round-5 advisory — never a raw `decimal.DivisionByZero`.

FREE-BIND-EVASION SWEEP (`--sweep`, required guard leg; round-5 fixes,
class B). A regression oracle over the real tracked tree, independent of
any specific guide tag: for every dict in every tracked cuda-run JSON with
>=2 child containers sharing a common relative leaf sub-path (the exact
shape every `legs`/`runs`/`shapes`/`summary` container in this tree uses),
every pair of same-sub-path values is tried as `diff`/`ratio`/`pct`/
`mean`/`min`/`max` (round-5 fix, class A: aggregates too) operands and
checked against every computed-family leaf in that file — but with an
INDEPENDENT reference matcher (`_reference_values_match`, a deliberately
separate implementation), NEVER the production matcher
(`_value_matches_any_scale`) — round-5 fix, class B: reusing one matcher
for both ENUMERATION and VERIFICATION meant a narrowed production matcher
shrank the reported population right alongside its catch rate, so a
fully-broken matcher (M2: always `False`) could print "0 candidates, 0
uncaught" and look clean. `run_real_tree_sweep` re-verifies every
independently-enumerated candidate through the REAL `evaluate_expr` path;
`--sweep` fails on EITHER an uncaught match OR a population below
`EXPECTED_SWEEP_MATCHES` (a pinned floor — catches an enumeration that
narrowed, which "uncaught == 0" alone cannot see since zero candidates
trivially yields zero uncaught).

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

# round-5 audit fix (class B, 2c): a POPULATION FLOOR for `--sweep`'s
# enumerated candidate count — a matcher mutated to always return False
# (M2) still enumerates candidates via the INDEPENDENT reference matcher,
# so `uncaught == 0` alone does not catch it (0 candidates -> 0 uncaught
# trivially); this floor does. Bump ONLY deliberately, after confirming
# why the population grew (a new artifact, a new computed field) — a drop
# below this number is itself a finding. round-6 audit fix: the real
# measured population (`--sweep` over the current tracked tree, after the
# class-A scale-factor fix) is 416 — this floor is set BELOW that
# (round-5's own "zero margin" floor = measured was itself flagged as
# brittle) so ordinary, expected run-to-run noise in which SPECIFIC
# coincidental matches land inside 5e-4 tolerance does not itself red the
# gate; a real regression (a mutated/narrowed matcher) drops the count far
# below this margin, not by one or two.
EXPECTED_SWEEP_MATCHES = 400

ALLOWLIST_PATH = REPO_ROOT / "ci" / "perf_claims_allowlist.txt"
CLASSIFICATION_PATH = REPO_ROOT / "ci" / "perf_claims_allowlist_classification.md"
CLASSIFICATION_REASONS = {"ledger-only", "modeled", "issue-text", "superseded-run"}

# round-5 audit fix (class A): the closed registry of every string-valued
# computed-family leaf in the tracked tree (delta_gb-shaped hand-written
# ranges — value equality can never be tested against free text, so these
# are the one class of computed field the round-4 value rule structurally
# cannot see). Format: `<path>#<pointer> | <reason>`, one per line.
# `check_string_field_registry` REDs on any unregistered string-valued
# computed field (growth) AND on a registered entry that no longer exists
# (shrinkage without editing the registry) — both directions checked.
STRING_FIELDS_PATH = REPO_ROOT / "ci" / "perf_claims_string_fields.txt"

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


_COMPUTED_FIELD_RE = re.compile(r"^(delta|ratio|speedup).*$|.*_pct$|^spread.*$")

# round-6 audit fix (class A): the scale set MUST be derived from the
# grammar's OWN unit factors (`_UNIT_FACTORS` — ms/s/GB/GiB/MiB/%), not a
# hand-picked subset — an earlier set covered only s<->ms and bytes<->GB,
# so `ratio(...) as %` (or `as GiB`/`as MiB`) could evade precedence
# entirely: the PRE-unit raw value (e.g. a raw fraction ~0.0623) never
# gets compared against a stored `_pct` field (e.g. 6.232) unless x100 is
# among the tried scales. 745 live uncaught free binds on the tracked
# tree (686 `%`, 32 GiB, 27 MiB) before this fix. Every direct unit
# factor, its reciprocal, and every PAIRWISE ratio between two unit
# factors is included, so "raw is in unit A's native scale, candidate is
# stated in unit B" is covered for every (A, B) pair the grammar can
# express — not only the pairs a previous round happened to think of.
_VALUE_REL_TOL = Decimal("0.0005")


def _derive_value_scale_factors(base_values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    factors: set[Decimal] = {Decimal(1)}
    for f in base_values:
        factors.add(f)
        if f != 0:
            factors.add(Decimal(1) / f)
    for f1 in base_values:
        for f2 in base_values:
            if f2 != 0:
                factors.add(f1 / f2)
    return tuple(sorted(factors))


_VALUE_SCALE_FACTORS = _derive_value_scale_factors(tuple(_UNIT_FACTORS.values()))


def _values_match(a: Decimal, b: Decimal, rel_tol: Decimal = _VALUE_REL_TOL) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0:
        return False
    return abs(a - b) <= rel_tol * denom


def _value_matches_any_scale(free_value: Decimal, candidate: Decimal) -> bool:
    """THE PRODUCTION matcher — used by the real runtime precedence check
    (`Loader.same_quantity_field`) ONLY. round-5 audit fix (class B): the
    sweep's own candidate ENUMERATION must never call this function — see
    `_reference_values_match`, a deliberately independent implementation,
    for that purpose, so a narrowing of THIS function (fewer scale
    factors, a smaller/zero tolerance, a dropped sign direction) shows up
    as an UNCAUGHT sweep candidate instead of silently shrinking the
    enumerated population too."""
    for scale in _VALUE_SCALE_FACTORS:
        scaled = free_value * scale
        if _values_match(scaled, candidate) or _values_match(-scaled, candidate):
            return True
    return False


# round-5 audit fix (class B): a SEPARATE, deliberately simpler reference
# matcher, used ONLY by `sibling_operand_pairs`/`sweep_free_bind_evasions`
# to ENUMERATE sweep candidates — verification always goes through the
# real `evaluate_expr` (which calls `_value_matches_any_scale` above via
# `Loader.same_quantity_field`), never this function. Reusing one matcher
# for both enumeration and verification was the exact defect the audit
# named: a narrowed production matcher would shrink the sweep's reported
# population right alongside its catch rate, so a fully-broken matcher
# (e.g. one that always returns False) could still print "0 candidates,
# 0 uncaught" and look clean.
# round-6 audit fix (class A): a SEPARATE derivation function (never
# calling `_derive_value_scale_factors`, and never importing
# `_UNIT_FACTORS` — its own independently-typed-out base values below),
# covering the SAME mathematical set: every direct unit factor, its
# reciprocal, and every pairwise cross-unit ratio.
_REFERENCE_BASE_UNIT_VALUES = (
    Decimal(1000),  # ms
    Decimal(1),  # s
    Decimal(1) / Decimal(10**9),  # GB
    Decimal(1) / Decimal(2**30),  # GiB
    Decimal(1) / Decimal(2**20),  # MiB
    Decimal(100),  # %
)


def _build_reference_scale_factors() -> tuple[Decimal, ...]:
    out: set[Decimal] = {Decimal(1)}
    for f in _REFERENCE_BASE_UNIT_VALUES:
        out.add(f)
        if f != 0:
            out.add(Decimal(1) / f)
    for f1 in _REFERENCE_BASE_UNIT_VALUES:
        for f2 in _REFERENCE_BASE_UNIT_VALUES:
            if f2 != 0:
                out.add(f1 / f2)
    return tuple(sorted(out))


_REFERENCE_SCALE_FACTORS = _build_reference_scale_factors()
_REFERENCE_REL_TOL = Decimal("0.0005")


def _reference_values_match(a: Decimal, b: Decimal) -> bool:
    for scale in _REFERENCE_SCALE_FACTORS:
        for sign in (1, -1):
            scaled = a * scale * sign
            if scaled == b:
                return True
            hi = max(abs(scaled), abs(b))
            if hi != 0 and abs(scaled - b) <= _REFERENCE_REL_TOL * hi:
                return True
    return False


def _iter_leaves(doc, prefix: str = ""):
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield from _iter_leaves(v, f"{prefix}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from _iter_leaves(v, f"{prefix}/{i}")
    else:
        yield prefix, doc


def _collect_dicts(doc, prefix: str = ""):
    if isinstance(doc, dict):
        yield prefix or "/", doc
        for k, v in doc.items():
            yield from _collect_dicts(v, f"{prefix}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from _collect_dicts(v, f"{prefix}/{i}")


def _numeric_leaf_map(node, prefix: str = "") -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for ptr, val in _iter_leaves(node, prefix):
        if isinstance(val, bool) or not isinstance(val, (Decimal, int)):
            continue
        out[ptr] = Decimal(val)
    return out


def sibling_operand_pairs(doc) -> list[tuple[str, Decimal, str, Decimal]]:
    """Every pair of numeric leaves a real author could plausibly write as
    diff/ratio/pct operands, generated PURELY STRUCTURALLY — no
    identifier-token reasoning (round-4 audit: token reasoning is exactly
    what round 2/3's precedence check used and both were found live-broken
    by an adversarial sweep). For every dict with >=2 CHILD dicts, and
    every relative leaf sub-path the children hold in COMMON, every pair
    of children's values at that sub-path — the exact shape every real
    `legs`/`runs`/`shapes`/`summary` container in this tree uses (two
    sibling leg objects both carrying `s_per_step_p50`; two sibling run
    objects both carrying a stat)."""
    pairs: list[tuple[str, Decimal, str, Decimal]] = []
    for base_ptr, d in _collect_dicts(doc):
        child_containers = {k: v for k, v in d.items() if isinstance(v, (dict, list))}
        if len(child_containers) < 2:
            continue
        rel_maps: dict[str, dict[str, tuple[str, Decimal]]] = {}
        for k, v in child_containers.items():
            child_prefix = f"{base_ptr.rstrip('/')}/{k}"
            leaves = _numeric_leaf_map(v, child_prefix)
            rel: dict[str, tuple[str, Decimal]] = {}
            for full_ptr, val in leaves.items():
                rel[full_ptr[len(child_prefix):]] = (full_ptr, val)
            if rel:
                rel_maps[k] = rel
        keys = sorted(rel_maps)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                common_subs = set(rel_maps[keys[i]]) & set(rel_maps[keys[j]])
                for sub in common_subs:
                    pa, va = rel_maps[keys[i]][sub]
                    pb, vb = rel_maps[keys[j]][sub]
                    pairs.append((pa, va, pb, vb))
    return pairs


def sweep_free_bind_evasions(doc, own_pointers: frozenset[str] = frozenset()) -> list[tuple[str, str, str, str, str]]:
    """For one parsed JSON document: every plausible operand pair
    (`sibling_operand_pairs`), every one of the FIVE free forms —
    `diff`/`ratio`/`pct` AND `mean`/`min`/`max` (round-5 audit fix, class
    A: an aggregate is itself a free computation, not exempt just because
    it isn't a diff/ratio/pct — the real T6 evasion is a bare `mean(...)`)
    — checked against every computed-family leaf in the SAME document via
    the INDEPENDENT reference matcher (`_reference_values_match` — NEVER
    `_value_matches_any_scale`, the production matcher; see that
    function's own docstring for why). Returns `(form, op_a_pointer,
    op_b_pointer, matched_field_pointer, free_value_str)` candidates — the
    POPULATION the real runtime path is verified against by
    `run_real_tree_sweep`, not itself a pass/fail signal."""
    evasions: list[tuple[str, str, str, str, str]] = []
    pairs = sibling_operand_pairs(doc)
    targets = [
        (ptr, val)
        for ptr, val in _numeric_leaf_map(doc).items()
        if _COMPUTED_FIELD_RE.match(ptr.rsplit("/", 1)[-1]) and ptr not in own_pointers
    ]
    for pa, va, pb, vb in pairs:
        forms: list[tuple[str, Decimal | None]] = [
            ("diff", va - vb),
            ("mean", (va + vb) / Decimal(2)),
            ("min", min(va, vb)),
            ("max", max(va, vb)),
        ]
        if vb != 0:
            forms.append(("ratio", va / vb))
            forms.append(("pct", (va / vb - Decimal(1)) * Decimal(100)))
        for form, free in forms:
            if free is None:
                continue
            for target_ptr, target_val in targets:
                if target_ptr in (pa, pb):
                    continue
                if _reference_values_match(free, target_val):
                    evasions.append((form, pa, pb, target_ptr, str(free)))
    return evasions


def run_real_tree_sweep() -> tuple[int, list[tuple[str, str, str, str, str]], int]:
    """Runs `sweep_free_bind_evasions` over every tracked cuda-run JSON
    under `POINTER_ROOTS`, then re-checks each candidate match through the
    REAL runtime path (`evaluate_expr` on an actually-constructed
    diff/ratio/pct `Expr`) rather than trusting the sweep's own duplicate
    matcher. Returns `(total_matches, uncaught, files_swept)` — `uncaught`
    must be empty; a nonzero `total_matches` alone is not a failure (it is
    the population of coincidences the mechanism is SUPPOSED to catch)."""
    tracked = tracked_files()
    loader = Loader(tracked=tracked)
    total_matches = 0
    uncaught: list[tuple[str, str, str, str, str]] = []
    files_swept = 0
    for rel in sorted(tracked):
        if not any(rel == r or rel.startswith(r + "/") for r in POINTER_ROOTS):
            continue
        if not rel.endswith(".json"):
            continue
        try:
            doc = loader.doc(rel)
        except ResolutionError:
            continue
        files_swept += 1
        for form, pa, pb, target_ptr, free_str in sweep_free_bind_evasions(doc):
            total_matches += 1
            expr = Expr(form, (PointerRef(rel, pa), PointerRef(rel, pb)))
            try:
                evaluate_expr(expr, loader)
                uncaught.append((rel, form, pa, pb, target_ptr))
            except ClaimParseError:
                pass
    return total_matches, uncaught, files_swept


class Loader:
    """Resolves `<path>#<pointer>` refs against tracked JSON under
    POINTER_ROOTS, with per-file caching. Injectable for tests."""

    def __init__(
        self,
        tracked: set[str] | None = None,
        base_dir: Path | None = None,
        string_registry: dict[tuple[str, str], str] | None = None,
    ):
        self._tracked = tracked
        self._base_dir = base_dir or REPO_ROOT
        self._cache: dict[str, object] = {}
        # `None` means "load STRING_FIELDS_PATH lazily, on first use" — an
        # explicit `{}` (what `_mem_loader` passes by default) means "no
        # registry, isolated from the real committed file," so self-test
        # fixtures never accidentally collide with real registered paths.
        self._string_registry = string_registry

    def string_field_registry(self) -> dict[tuple[str, str], str]:
        if self._string_registry is None:
            self._string_registry = parse_string_fields_registry()
        return self._string_registry

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

    def same_quantity_field(self, free_value: Decimal, refs: list[PointerRef]) -> tuple[str, Decimal] | None:
        """round-4 audit rewrite: VALUE-based whole-FILE search — NOT
        limited to the operands' common-ancestor subtree, and NOT keyed on
        any identifier-token/unit-family reasoning (both were found
        live-broken by an adversarial sweep of the real tracked tree: the
        candidate is almost always LESS token-specific than the operands,
        e.g. `delta_ms` carries no rep suffix while its own operands do).
        Scans every NUMBER leaf in the file whose key matches
        `_COMPUTED_FIELD_RE` and returns the first whose value equals
        `free_value` within `_value_matches_any_scale` (5e-4 relative,
        either sign, or after a x1000/1000/1e9/1e-9 rescale). Raises
        `PrecedenceUndecidable` — never returns `None` for this case —
        when the operands span more than one tracked file. round-5 audit
        fix (class A): BEFORE the numeric scan, also checks whether any
        operand lives under a REGISTERED string-valued computed field's
        own parent object (`STRING_FIELDS_PATH`) — string leaves can never
        be value-checked, so this is the only mechanical guard against a
        free aggregate binding cleanly beside a hand-written range."""
        paths = {r.path for r in refs}
        if len(paths) != 1:
            raise PrecedenceUndecidable(
                f"precedence undecidable: operands span {len(paths)} different tracked "
                f"files ({sorted(paths)}) — point directly at a field or ledger this cell"
            )
        path = next(iter(paths))
        doc = self.doc(path)
        operand_pointers = {r.pointer for r in refs}
        for (reg_path, reg_ptr) in self.string_field_registry():
            if reg_path != path:
                continue
            parent = reg_ptr.rsplit("/", 1)[0] or "/"
            parent_prefix = parent.rstrip("/") + "/"
            if any(p == parent or p.startswith(parent_prefix) for p in operand_pointers):
                return reg_ptr, Decimal(0)
        for leaf_pointer, leaf_value in _iter_leaves(doc):
            if leaf_pointer in operand_pointers:
                continue
            last_seg = leaf_pointer.rsplit("/", 1)[-1]
            if not _COMPUTED_FIELD_RE.match(last_seg):
                continue
            if isinstance(leaf_value, bool) or not isinstance(leaf_value, (Decimal, int)):
                continue
            leaf_dec = Decimal(leaf_value)
            if _value_matches_any_scale(free_value, leaf_dec):
                return leaf_pointer, leaf_dec
        return None


def leaves_of(expr_or_ref) -> list[PointerRef]:
    if isinstance(expr_or_ref, PointerRef):
        return [expr_or_ref]
    out: list[PointerRef] = []
    for a in expr_or_ref.args:
        out.extend(leaves_of(a))
    return out


def _non_pointer_subexprs(expr_or_ref):
    """Every node in the expr tree that is itself a free computation —
    i.e. every `Expr` whose `kind != 'pointer'` (a bare `PointerRef`, or a
    `pointer`-kind `Expr` wrapping one, is never a free computation and
    stays exempt) — innermost first. round-5 audit fix (class A): a
    top-level OR NESTED `min`/`mean`/`max` is itself a free computation
    and must be checked independently of any enclosing diff/ratio/pct —
    the real T6 evasion is a bare `mean(...)` tag, not nested inside
    anything."""
    if isinstance(expr_or_ref, PointerRef):
        return
    for a in expr_or_ref.args:
        yield from _non_pointer_subexprs(a)
    if expr_or_ref.kind != "pointer":
        yield expr_or_ref


def precedence_violation(expr: Expr, loader: Loader) -> str | None:
    """None, or the offending field's own json-pointer. round-5 audit fix
    (class A): recursively checks EVERY non-pointer node in the expr tree
    — not only an outermost diff/ratio/pct, which is what let a top-level
    `mean(A,B)` bind freely even though its own value reproduced a
    declared field. For each such node, evaluates its OWN free result
    (ignoring the outer tag's neg/unit/legacy wrapping, which only ever
    applies to the outermost node) and asks whether the pointed artifact
    ALSO states that value elsewhere via a delta*/ratio*/speedup*/*_pct/
    spread* field or a registered string-valued field's own object,
    regardless of which leaves produced either number."""
    for sub in _non_pointer_subexprs(expr):
        sub_value = eval_value(sub, loader)
        hit = loader.same_quantity_field(sub_value, leaves_of(sub))
        if hit is not None:
            return hit[0]
    return None


def _describe_operand(op) -> str:
    if isinstance(op, PointerRef):
        return f"{op.path}#{op.pointer}"
    return f"{op.kind}(...)"


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
        b_val = eval_value(b, loader)
        if b_val == 0:
            # round-5 advisory: a zero denominator previously raised a raw
            # decimal.DivisionByZero straight out of this function (1,390
            # zero-valued leaves live in the tracked tree) — converted to
            # a ClaimParseError naming the offending pointer, so it reads
            # as an ordinary gate finding, not an unhandled crash.
            raise ClaimParseError(f"ratio(...) has a zero denominator at {_describe_operand(b)}")
        return eval_value(a, loader) / b_val
    if e.kind == "pct":
        a, b = e.args
        b_val = eval_value(b, loader)
        if b_val == 0:
            raise ClaimParseError(f"pct(...) has a zero denominator at {_describe_operand(b)}")
        return (eval_value(a, loader) / b_val - 1) * Decimal(100)
    raise ClaimParseError(f"unknown expr kind: {e.kind}")


def evaluate_expr(expr: Expr, loader: Loader) -> tuple[Decimal, bool]:
    """Returns (value, is_legacy). Runs the (now whole-tree) precedence
    check first, then applies the legacy() allowlist, sign negation, and
    the unit factor, in that order, to the OUTERMOST node's own free
    result."""
    hit_pointer = precedence_violation(expr, loader)
    if hit_pointer is not None:
        field = hit_pointer.rsplit("/", 1)[-1]
        raise ClaimParseError(
            f"artifact states this value elsewhere at {hit_pointer} ({field}) — a free "
            "expression in this tag (possibly a nested aggregate) matches within tolerance; "
            "bind directly to it or ledger this cell — a free aggregate matching a stated "
            "computed field is not legal here"
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


def parse_string_fields_registry() -> dict[tuple[str, str], str]:
    if not STRING_FIELDS_PATH.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    for line in STRING_FIELDS_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key_part, sep, reason = stripped.partition("|")
        if not sep:
            continue
        path, sep2, pointer = key_part.strip().partition("#")
        if not sep2:
            continue
        out[(path.strip(), pointer.strip())] = reason.strip()
    return out


def check_string_field_registry(
    tracked: set[str] | None = None, loader: "Loader | None" = None
) -> list[str]:
    """round-5 audit fix (class A): every string-valued leaf in the
    tracked tree whose key matches the computed-family pattern must be
    registered in STRING_FIELDS_PATH — an unregistered one (growth) is a
    named problem, since it is exactly the precedence blind spot the
    round-4 value rule structurally cannot see (value equality cannot be
    tested against free text). A registered entry that no longer exists
    as a string leaf (the field was removed, or became numeric) is ALSO a
    named problem — the registry must track reality, not accumulate.
    round-6 audit fix (class B): `tracked`/`loader` are injectable (never
    used by production, which always passes neither) so `--self-test` can
    plant a fixture string leaf on a fake tracked-path WITHOUT touching
    the real committed registry or filesystem — this function had NO
    oracle at all before round 6: unwiring its call from `scan_tree`, or
    mutating its body to `return []`, left every gate green."""
    if tracked is None:
        tracked = tracked_files()
    if loader is None:
        registry = parse_string_fields_registry()
        loader = Loader(tracked=tracked, string_registry=registry)
    else:
        registry = loader.string_field_registry()
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for rel in sorted(tracked):
        if not any(rel == r or rel.startswith(r + "/") for r in POINTER_ROOTS):
            continue
        if not rel.endswith(".json"):
            continue
        try:
            doc = loader.doc(rel)
        except ResolutionError:
            continue
        for ptr, val in _iter_leaves(doc):
            last = ptr.rsplit("/", 1)[-1]
            if _COMPUTED_FIELD_RE.match(last) and isinstance(val, str):
                seen.add((rel, ptr))
                if (rel, ptr) not in registry:
                    problems.append(
                        f"unregistered string-valued computed field: {rel}#{ptr} — add it to "
                        f"{STRING_FIELDS_PATH.relative_to(REPO_ROOT)} with a reason"
                    )
    for key in registry:
        if key not in seen:
            problems.append(
                f"{STRING_FIELDS_PATH.relative_to(REPO_ROOT)} registers {key[0]}#{key[1]}, "
                "which no longer exists in the tracked tree as a string leaf"
            )
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
    scope_problems.extend(check_string_field_registry())
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


def _mem_loader(doc: dict, string_registry: dict[tuple[str, str], str] | None = None) -> Loader:
    # `string_registry` defaults to `{}` (never `None`, which would lazily
    # read the REAL committed STRING_FIELDS_PATH) — fixtures stay isolated
    # from the real file unless a test explicitly opts in.
    loader = Loader(tracked=None, string_registry=string_registry if string_registry is not None else {})
    loader._check_path = lambda path: None  # type: ignore[method-assign]
    loader._cache["fixture.json"] = doc
    return loader


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(f"self-test FAILED: {name}")

    # --- round-4 audit: VALUE-based precedence. Reproduces the real
    # cast-w1 shape: operands under /legs/..., the computed field lives at
    # /deltas/... — a SIBLING subtree, not an ancestor. ---
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
                "delta_pct": Decimal("6.232281178"),
            }
        },
    }
    loader_cw1 = _mem_loader(cast_w1_doc)

    # direct pointer AT the flagged field is exempt (bare pointer, not a
    # diff/ratio/pct tag at all).
    direct_expr = parse_expr("neg(fixture.json#/deltas/b8_s512_p50_ms/delta_ms)")
    direct_val, _ = evaluate_expr(direct_expr, loader_cw1)
    check("direct pointer to the computed field is legal", direct_val == Decimal("-39.565658"))

    # --- REQUIRED FIXTURE 1 (round-4 acceptance): real cast-w1 SAME-REP
    # diff/pct. `diff(disabled_r1, fused_r1)` (both literally `r1`) in raw
    # seconds equals `delta_ms` after x1000 EXACTLY (0.039565658 * 1000 =
    # 39.565658) — the identity/token mechanisms in rounds 2-3 both missed
    # this (round 2: no rep-stripping at all; round 3: rep IS shared by
    # both operands so it survived the intersection and over-constrained
    # the candidate search past delta_ms, whose own path carries none).
    # Value-based matching does not care what either leaf is CALLED. ---
    same_rep_diff_expr = parse_expr(
        "diff(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50)"
    )
    try:
        val, _ = evaluate_expr(same_rep_diff_expr, loader_cw1)
        failures.append(
            f"self-test FAILED (REQUIRED FIXTURE 1, same-rep diff): bound FREELY to {val} "
            "— the exact evasion class the round-3/4 audits found live"
        )
    except ClaimParseError as exc:
        check(
            "same-rep diff (both r1) is caught by VALUE match, names delta_ms",
            "delta_ms" in str(exc),
        )
    same_rep_pct_expr = parse_expr(
        "pct(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50)"
    )
    try:
        evaluate_expr(same_rep_pct_expr, loader_cw1)
        failures.append("self-test FAILED (REQUIRED FIXTURE 1, same-rep pct): bound freely, not caught")
    except ClaimParseError as exc:
        check("same-rep pct (both r1) is also caught by value match", "delta" in str(exc))

    # --- REQUIRED FIXTURE 2 (round-4 acceptance): the REAL adamw
    # evasion the audit named — `speedup_median_run1 = 9.2065` reproduced
    # freely by `ratio(eager_arm.run_1.median_ms, fused_arm.run_1.
    # median_ms)`. `run_1` (with an underscore, inside a longer path
    # segment) never matched the old `^r\d+$` rep-suffix regex at all —
    # value-based matching does not need to recognize "run" as a
    # replicate marker in the first place. ---
    adamw_doc = {
        "optimizer_phase_wall_time_ms": {
            "eager_arm": {"run_1": {"median_ms": Decimal("24.7481")}},
            "fused_arm": {"run_1": {"median_ms": Decimal("2.6881")}},
            "speedup_median_run1": Decimal("9.2065"),
        }
    }
    loader_adamw = _mem_loader(adamw_doc)
    adamw_run1_ratio_expr = parse_expr(
        "ratio(fixture.json#/optimizer_phase_wall_time_ms/eager_arm/run_1/median_ms,"
        "fixture.json#/optimizer_phase_wall_time_ms/fused_arm/run_1/median_ms)"
    )
    try:
        evaluate_expr(adamw_run1_ratio_expr, loader_adamw)
        failures.append(
            "self-test FAILED (REQUIRED FIXTURE 2, adamw run_1 ratio): bound freely to "
            "~9.2065 without naming speedup_median_run1 — the real, named live evasion"
        )
    except ClaimParseError as exc:
        check(
            "adamw run_1 ratio is caught by value match, names speedup_median_run1",
            "speedup_median_run1" in str(exc),
        )

    # --- REQUIRED FIXTURE 3 (round-4 acceptance): a NUMERIC bytes pair
    # whose diff equals a `delta_gb` field (never `delta_ms`, even though
    # a `delta_ms` field of unrelated value sits in the SAME file) —
    # value equality, not a unit-family guess, is what selects the right
    # field; there is no family veto anymore because there is no family
    # heuristic anymore. ---
    bytes_doc = {
        "legs": {
            "b8_s512_disabled_r1_gb": {"peak_vram_bytes": Decimal("1000")},
            "b8_s512_fused_r1_gb": {"peak_vram_bytes": Decimal("900")},
        },
        "deltas": {
            "b8_s512_p50_ms": {"delta_ms": Decimal("39.565658")},
            "b8_s512_p50_gb": {"delta_gb": Decimal("100")},
        },
    }
    loader_bytes = _mem_loader(bytes_doc)
    gb_operand_expr = parse_expr(
        "diff(fixture.json#/legs/b8_s512_disabled_r1_gb/peak_vram_bytes,"
        "fixture.json#/legs/b8_s512_fused_r1_gb/peak_vram_bytes)"
    )
    try:
        evaluate_expr(gb_operand_expr, loader_bytes)
        failures.append("self-test FAILED (REQUIRED FIXTURE 3, bytes pair): not caught")
    except ClaimParseError as exc:
        check(
            "a numeric bytes-shaped diff (=100) is matched to delta_gb (=100), never the "
            "unrelated delta_ms (=39.565658) in the same file",
            "delta_gb" in str(exc) and "delta_ms" not in str(exc),
        )

    # --- REQUIRED FIXTURE 4 (round-4 acceptance): the CONTROL — a free
    # aggregate that matches NOTHING stays legal. Reuses the pre-round-4
    # MEAN-based cast-w1 diff: mean(disabled_r1,disabled_r2) -
    # mean(fused_r1,fused_r2) = 38.478341 ms, genuinely 2.7% away from
    # delta_ms=39.565658 (nowhere near the 5e-4 relative tolerance) — a
    # real, different number, not a reproduction, and must bind cleanly. ---
    control_expr = parse_expr(
        "neg(diff(mean(fixture.json#/legs/b8_s512_disabled_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_disabled_r2/s_per_step_p50),"
        "mean(fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50,"
        "fixture.json#/legs/b8_s512_fused_r2/s_per_step_p50))) as ms"
    )
    control_val, _ = evaluate_expr(control_expr, loader_cw1)
    check(
        "CONTROL: a free aggregate matching NOTHING (mean-based, 2.7% away from delta_ms) "
        "stays legal, no exception",
        control_val.quantize(Decimal("0.001")) == Decimal("-38.478"),
    )

    # A string-valued same-family field is NEVER value-checked directly
    # (value equality cannot be tested against free text) — but round-5
    # audit fix (class A) closes the resulting blind spot with a
    # REGISTRY: if the field is registered (STRING_FIELDS_PATH), a free
    # aggregate whose operands live under its own object IS caught; if it
    # is genuinely unregistered (this fixture's empty in-memory registry —
    # `_mem_loader` never reads the real committed file), it binds
    # cleanly, honestly demonstrating why `check_string_field_registry`
    # (part of the bare run) must RED on any unregistered string leaf in
    # the real tree — there is no other guard once a string field exists
    # unregistered.
    string_field_doc = {
        "peak": {
            "b8_s512": {
                "fused_r1_gb": Decimal("16.0"),
                "disabled_r1_gb": Decimal("15.7"),
                "delta_gb": "+0.1 to +0.3 GB",
            }
        }
    }
    loader_string = _mem_loader(string_field_doc)  # empty registry
    string_field_expr = parse_expr(
        "diff(fixture.json#/peak/b8_s512/fused_r1_gb,fixture.json#/peak/b8_s512/disabled_r1_gb)"
    )
    val_str, _ = evaluate_expr(string_field_expr, loader_string)
    check(
        "an UNREGISTERED string-valued delta_gb field is not caught (motivates the registry, "
        "not a silent forever-gap) — the free diff binds cleanly",
        val_str == Decimal("0.3"),
    )

    # --- REQUIRED FIXTURE 5 (round-5 acceptance, class A): the SAME
    # string field, now REGISTERED — a free diff whose operands live
    # under the registered field's own parent object is a finding,
    # regardless of whether any NUMBER also happens to match. Reproduces
    # the real, live gap the audit re-opened: cast-w1's b8_s128 and
    # flash-arm-encoder-oracle's delta_gb rows bound freely with no
    # numeric twin to rescue them. ---
    loader_string_reg = _mem_loader(
        string_field_doc, string_registry={("fixture.json", "/peak/b8_s512/delta_gb"): "test fixture"}
    )
    try:
        evaluate_expr(string_field_expr, loader_string_reg)
        failures.append(
            "self-test FAILED (REQUIRED FIXTURE 5, registered string field): a free diff "
            "beside a REGISTERED string field bound cleanly — the registry-object check is dead"
        )
    except ClaimParseError as exc:
        check(
            "a REGISTERED string field's object blocks a free diff over its siblings, names it",
            "delta_gb" in str(exc),
        )

    # --- REQUIRED FIXTURE 6 (round-5 acceptance, class A): a TOP-LEVEL
    # `mean(...)` tag — not nested inside any diff/ratio/pct — is itself a
    # free computation and must be checked. Reproduces the real T6
    # evasion the audit demonstrated end-to-end: mean(ratio_torch_over_
    # stacked@b8s128, ratio_torch_over_stacked@b8s1024) = 1.07425
    # reproduces a THIRD shape's own ratio_torch_over_stacked = 1.0737811
    # within 5e-4 — bound V=1 on 0a1a317 because `precedence_violation`
    # exempted every kind outside diff/ratio/pct. ---
    t6_mean_doc = {
        "shapes": {
            "b8s128": {"ratio_torch_over_stacked": Decimal("1.0085995163106791")},
            "b8s1024": {"ratio_torch_over_stacked": Decimal("1.1399039368319384")},
            "b16s512": {"ratio_torch_over_stacked": Decimal("1.0737811162910524")},
        }
    }
    loader_t6 = _mem_loader(t6_mean_doc)
    t6_mean_expr = parse_expr(
        "mean(fixture.json#/shapes/b8s128/ratio_torch_over_stacked,"
        "fixture.json#/shapes/b8s1024/ratio_torch_over_stacked)"
    )
    try:
        val, _ = evaluate_expr(t6_mean_expr, loader_t6)
        failures.append(
            f"self-test FAILED (REQUIRED FIXTURE 6, top-level mean): bound FREELY to {val} — "
            "the exact live T6 evasion the round-5 audit demonstrated end-to-end"
        )
    except ClaimParseError as exc:
        check(
            "a top-level mean() aggregate is caught, names the reproduced field",
            "ratio_torch_over_stacked" in str(exc) and "b16s512" in str(exc),
        )

    # a mean() NESTED inside a diff() is ALSO checked independently of the
    # outer diff's own combined value (round-5 fix (1)'s "nested" clause).
    nested_doc = {
        "legs": {
            "a1": {"v": Decimal("1.0085995163106791")},
            "a2": {"v": Decimal("1.1399039368319384")},
            "b1": {"v": Decimal("5")},
            "b2": {"v": Decimal("5")},
        },
        "shapes": {"x": {"ratio_torch_over_stacked": Decimal("1.0737811162910524")}},
    }
    loader_nested = _mem_loader(nested_doc)
    nested_expr = parse_expr(
        "diff(mean(fixture.json#/legs/a1/v,fixture.json#/legs/a2/v),"
        "mean(fixture.json#/legs/b1/v,fixture.json#/legs/b2/v))"
    )
    try:
        evaluate_expr(nested_expr, loader_nested)
        failures.append(
            "self-test FAILED (nested mean inside diff): the inner mean(a1,a2) alone "
            "reproduces shapes/x/ratio_torch_over_stacked but was not independently checked"
        )
    except ClaimParseError as exc:
        check("a mean() NESTED inside a diff() is independently precedence-checked", "ratio_torch_over_stacked" in str(exc))

    # --- REQUIRED FIXTURE 7 (round-5 acceptance, class B): boundary
    # pins — opposite sign, a 4.9e-4-relative match (just inside
    # tolerance, caught), a 6.0e-4 near-miss (just outside, legal), and
    # each rescale direction including the round-5-added x1e9. ---
    boundary_doc = {
        "a": {"x": Decimal("39.565658")},  # ms-scale target
        "deltas": {"y": {"delta_ms": Decimal("-39.565658")}},  # opposite-sign target
        "b": {"gb": Decimal("0.302")},  # bytes-scale target (GB)
        "c": {"bytes_field": Decimal("302000000")},  # x1e9 rescale target
    }
    loader_boundary = _mem_loader(boundary_doc)
    # opposite sign: free result +39.565658, target is -39.565658 — must match.
    opp_expr = parse_expr("fixture.json#/a/x")
    opp_val, _ = evaluate_expr(opp_expr, loader_boundary)
    check(
        "opposite-sign candidate value is a genuine match under _value_matches_any_scale",
        _value_matches_any_scale(opp_val, Decimal("-39.565658")),
    )
    # 4.9e-4 relative (just inside 5e-4 tolerance) is caught.
    check(
        "a 4.9e-4 relative difference IS caught (inside the 5e-4 tolerance)",
        _value_matches_any_scale(Decimal("1.00049"), Decimal("1.0")),
    )
    # 6.0e-4 relative (just outside) is legal.
    check(
        "a 6.0e-4 relative difference is NOT caught (outside the 5e-4 tolerance) — a near-miss stays legal",
        not _value_matches_any_scale(Decimal("1.0006"), Decimal("1.0")),
    )
    # each rescale direction, including the round-5-added x1e9 (GB->bytes).
    check("x1000 rescale (s->ms) matches", _value_matches_any_scale(Decimal("0.039565658"), Decimal("39.565658")))
    check("/1000 rescale (ms->s) matches", _value_matches_any_scale(Decimal("39.565658"), Decimal("0.039565658")))
    check("/1e9 rescale (bytes->GB) matches", _value_matches_any_scale(Decimal("302000000"), Decimal("0.302")))
    check(
        "x1e9 rescale (GB->bytes, round-5 advisory fix — the prior scale set was asymmetric) matches",
        _value_matches_any_scale(Decimal("0.302"), Decimal("302000000")),
    )

    # --- REQUIRED FIXTURE 8 (round-6 acceptance, class A): the REAL
    # `as %` evasion the audit drove through the real scan_table --
    # `ratio(...delta_ms, ...fused_r1) as %` binds V=1, findings=0 while
    # reproducing that object's own delta_pct to 14 digits, because the
    # PRE-unit raw ratio (~0.0623) was never rescaled by x100 before being
    # compared. Uses the real cast-w1 numbers. 745 live uncaught free
    # binds on the tracked tree before this fix (% 686, GiB 32, MiB 27). ---
    as_pct_doc = {
        "deltas": {
            "b8_s512_p50_ms": {
                "delta_ms": Decimal("39.565658"),
                "delta_pct": Decimal("6.232281178157861"),
            }
        },
        "legs": {"b8_s512_fused_r1": {"s_per_step_p50": Decimal("0.634850336")}},
    }
    loader_as_pct = _mem_loader(as_pct_doc)
    as_pct_expr = parse_expr(
        "ratio(fixture.json#/deltas/b8_s512_p50_ms/delta_ms,"
        "fixture.json#/legs/b8_s512_fused_r1/s_per_step_p50) as %"
    )
    try:
        val, _ = evaluate_expr(as_pct_expr, loader_as_pct)
        failures.append(
            f"self-test FAILED (REQUIRED FIXTURE 8, as %% evasion): bound FREELY to {val} "
            "— the real live class-A evasion the round-6 audit drove through scan_table"
        )
    except ClaimParseError as exc:
        check(
            "ratio(...) as %% is caught by the whole-expr-tree value rule, names delta_pct",
            "delta_pct" in str(exc),
        )
    # a GiB-wrapped free result is caught too (same class, same fix): the
    # RAW free result is in bytes; the declared field is stated in GiB.
    as_gib_doc = {
        "a": {"peak_bytes": Decimal("324398043136")},
        "b": {"peak_bytes": Decimal("0")},
        "spread_peak_gib": Decimal("302.0"),
    }
    loader_as_gib = _mem_loader(as_gib_doc)
    as_gib_expr = parse_expr("diff(fixture.json#/a/peak_bytes,fixture.json#/b/peak_bytes) as GiB")
    try:
        evaluate_expr(as_gib_expr, loader_as_gib)
        failures.append("self-test FAILED (as GiB evasion): bound freely, not caught")
    except ClaimParseError as exc:
        check("diff(...) as GiB is caught by the whole-expr-tree value rule", "spread_peak_gib" in str(exc))

    # --- round-5 advisory: a zero denominator is a ClaimParseError naming
    # the offending pointer, never a raw decimal.DivisionByZero crash
    # (1,390 zero-valued leaves live in the tracked tree). ---
    zero_doc = {"a": {"num": Decimal("5")}, "b": {"den": Decimal("0")}}
    loader_zero = _mem_loader(zero_doc)
    for form in ("ratio", "pct"):
        zero_expr = parse_expr(f"{form}(fixture.json#/a/num,fixture.json#/b/den)")
        try:
            evaluate_expr(zero_expr, loader_zero)
            failures.append(f"self-test FAILED (zero denominator, {form}): did not raise at all")
        except ZeroDivisionError:
            failures.append(
                f"self-test FAILED (zero denominator, {form}): raised a raw ZeroDivisionError/"
                "decimal exception, not a ClaimParseError naming the pointer"
            )
        except ClaimParseError as exc:
            check(f"zero denominator ({form}) raises ClaimParseError naming the pointer", "/b/den" in str(exc))

    # operands spanning more than one tracked file are STILL undecidable
    # (round-3 mechanism kept for this one case, per the audit).
    loader4b = _mem_loader({"v": Decimal("1")})
    loader4b._cache["other.json"] = {"w": Decimal("2")}
    expr4b = parse_expr("diff(fixture.json#/v,other.json#/w)")
    try:
        evaluate_expr(expr4b, loader4b)
        failures.append("self-test FAILED (multi-file): cross-file operands did not raise")
    except PrecedenceUndecidable as exc:
        check("multi-file operands fail loudly and say 'undecidable'", "undecidable" in str(exc))

    # round-4 acceptance (1): the free-bind-evasion sweep over the real
    # tracked tree. `sweep_free_bind_evasions` enumerates every plausible
    # operand pair whose free VALUE equals a declared computed field
    # (the population — informational; cross-shape numeric coincidences
    # are EXPECTED here and are supposed to be caught, not absent). What
    # must be exactly zero is how many of those matches slip PAST the
    # real runtime path uncaught: for each match, the actual
    # diff/ratio/pct Expr is constructed and run through `evaluate_expr`
    # — the same call a real guide tag makes — and must raise. This is
    # the regression oracle the audit named: "no free tag reproduces a
    # stated computed field" (without being caught).
    sweep_matches, sweep_uncaught, swept_files = run_real_tree_sweep()
    check(
        f"real-tree sweep ({swept_files} tracked cuda-run JSONs): {sweep_matches} candidate "
        "free-bind match(es) found, ALL caught by the real evaluate_expr path (0 slip through)",
        len(sweep_uncaught) == 0,
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

    # --- round-6 audit fix (class B): check_string_field_registry had NO
    # oracle at all — M12 (unwiring its call from scan_tree) and M13
    # (mutating it to `return []`) both left every gate green. Two
    # fixtures: (i) the FUNCTION's own correctness, on a planted fixture
    # leaf under a fake but POINTER_ROOTS-shaped tracked path, injected
    # via the loader/tracked params (never touching the real committed
    # registry or filesystem); (ii) that scan_tree ACTUALLY CALLS it
    # (wiring), verified by substituting the module-level function with a
    # sentinel and confirming scan_tree's own scope_problems propagate it. ---
    fake_string_path = "crates/jammi-kernels/artifacts/cuda-runs/__self_test_string_field_fixture__.json"
    fake_string_doc = {"some": {"delta_gb": "1.0-2.0 GB (unregistered on purpose)"}}
    fake_string_loader = Loader(tracked={fake_string_path}, string_registry={})
    fake_string_loader._check_path = lambda path: None  # type: ignore[method-assign]
    fake_string_loader._cache[fake_string_path] = fake_string_doc
    problems_string = check_string_field_registry(tracked={fake_string_path}, loader=fake_string_loader)
    check(
        "an UNREGISTERED string-valued computed field on a tracked-path fixture is one "
        "named problem (M13 regression: mutating the function to `return []` misses this)",
        len(problems_string) == 1 and fake_string_path in problems_string[0] and "delta_gb" in problems_string[0],
    )

    _module = sys.modules[__name__]
    _original_check_string_field_registry = _module.check_string_field_registry
    _module.check_string_field_registry = lambda: ["__ROUND6_WIRING_SENTINEL__"]
    try:
        _, _, wiring_scope_problems = scan_tree()
    finally:
        _module.check_string_field_registry = _original_check_string_field_registry
    check(
        "check_string_field_registry is WIRED into scan_tree's scope_problems (M12 "
        "regression: unwiring the call leaves the sentinel — and any real problem — invisible)",
        "__ROUND6_WIRING_SENTINEL__" in wiring_scope_problems,
    )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check_perf_claims self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "check_perf_claims self-test: OK — round-4/5 VALUE-based precedence over the WHOLE "
        "expr tree (real cast-w1 same-rep diff/pct, the real adamw speedup_median_run1 "
        "evasion, a numeric bytes pair naming delta_gb over an unrelated delta_ms, a "
        "mean-based CONTROL that matches nothing and stays legal, a top-level AND a nested "
        "mean() reproducing the real T6 evasion, a REGISTERED string field blocking its "
        "sibling operands vs. an unregistered one binding cleanly, boundary/near-miss/"
        "opposite-sign/all-four-rescale pins, zero-denominator ClaimParseErrors, and "
        "multi-file PrecedenceUndecidable), the real-tree sweep (independent reference "
        "matcher, population floor) finding zero uncaught matches, Decimal/ROUND_HALF_EVEN "
        "string equality, the six-form grammar's parse-error refusal, uncovered-token "
        "findings with file:line:col, the lexical exclusion class (seed/section/formula/"
        "type-name/product-name, the unified rows-list pattern, and the layer/launch-count "
        "non-exclusion), the .json.raw/out-of-root/untracked pointer refusals, neg()'s "
        "sign-preserving compare, legacy()'s closed allowlist, the injective "
        "file:token:sha1:col allowlist key, the KNOWN_TABLES exactly-one-match invariant "
        "(rename + duplicate), the citation-table header exclusion, and the "
        "classification-file verification all confirmed."
    )
    return 0


def sweep_command() -> int:
    """`--sweep`: the free-bind-evasion regression oracle (round-4 audit
    acceptance (1); round-5 fix, class B) as its own CI leg, independent
    of `--self-test`'s in-memory fixtures — runs over the REAL tracked
    cuda-run tree. Prints the match population and the uncaught count;
    fails on EITHER an uncaught match (the production verify path missed
    a real, independently-enumerated candidate) OR a population below
    `EXPECTED_SWEEP_MATCHES` (the enumeration itself narrowed — e.g. a
    mutated reference matcher — which `uncaught == 0` alone cannot see,
    since zero candidates trivially yields zero uncaught)."""
    total_matches, uncaught, files_swept = run_real_tree_sweep()
    print(f"check_perf_claims --sweep: {files_swept} tracked cuda-run JSON(s) swept")
    print(
        f"  candidate free-bind value-matches found: {total_matches} "
        f"(population floor: {EXPECTED_SWEEP_MATCHES})"
    )
    print(f"  uncaught (must be 0): {len(uncaught)}")
    failed = False
    if total_matches < EXPECTED_SWEEP_MATCHES:
        print(
            f"check_perf_claims --sweep: FAIL — population {total_matches} < "
            f"EXPECTED_SWEEP_MATCHES {EXPECTED_SWEEP_MATCHES} (the enumeration itself "
            "narrowed; bump the constant only after confirming why)",
            file=sys.stderr,
        )
        failed = True
    if uncaught:
        print("check_perf_claims --sweep: FAIL — uncaught matches", file=sys.stderr)
        for rel, form, pa, pb, target_ptr in uncaught:
            print(f"  {rel}: {form}({pa}, {pb}) reproduces {target_ptr} UNCAUGHT", file=sys.stderr)
        failed = True
    if failed:
        return 1
    print("check_perf_claims --sweep: OK — every free-bind value-match is caught")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()
    if "--sweep" in argv:
        return sweep_command()
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
