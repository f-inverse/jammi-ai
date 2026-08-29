#!/usr/bin/env python3
"""esc-065's write-time oracle: every artifact-referencing measurement claim
in a `docs/plans/63-how-well/measurements/**/README.md` or the
`docs/plans/63-how-well/mutants/README.md` "Measured" record must resolve
against a committed artifact AT WRITE TIME — never rely on an audit round N
commits later to catch a wrong pointer or a numerically false relation
(esc-065: 4 such instances, rounds 13-16, each caught by adversarial audit
instead of a gate).

THIS IS A COVERAGE EXTENSION OF RULE (h), NOT A SECOND MECHANISM. Rule (h)
(`check_perf_claims.py`) proved this idiom on `docs/maintainer/**` /
`docs/plans/61-perf-unification/**` pipe tables: a closed tag grammar bound
to numeric tokens, Decimal/ROUND_HALF_EVEN equality at the doc's own printed
precision, a shrink-only ledger for genuinely unproducible numbers, and an
independent-matcher anti-vacuity leg. This module targets a DIFFERENT record
shape — prose paragraphs and markdown tables that cite JSON pointers and
relations, not `docs/maintainer`'s pipe-table cells — so it is a SIBLING
checker, not an extension of rule (h)'s own scope (`~:16` pins rule (h) to
`docs/maintainer/**`/`docs/plans/61-perf-unification/**` only). It REUSES
rule (h)'s resolution core by IMPORTING it (never re-implementing): pointer
walking (`_rfc6901_walk`), the tokenizer's exclusion-span machinery
(`excluded_spans`/`_in_any_span`), and the Decimal-precision equality idiom
(reused via `_decimal_places_of`/`_quantize` below, generalized to also
accept scientific notation — rule (h)'s own corpus never needed exponents;
this one does, `mutants/README.md`'s CPU-verifiable demonstration table is
printed in `1.7542565e-4` form). `check_perf_claims.py` itself is NEVER
edited by this module; its own constants (`EXPECTED_DENOMINATOR` etc.) stay
exactly as they were — the reuse is by IMPORT, read-only.

TAG GRAMMAR (closed; comment on the line immediately above a claim-bearing
line — a prose line or a markdown table body row — exactly rule (h)'s own
"HTML comment on the line above" placement convention, generalized from
"one entry per table-row token" to "one entry per token on ANY line", since
this corpus's claims live in paragraphs and table rows both):

  <!-- claims63: default=<path>; c1=<expr>; c2=<expr>; ... -->

`default=<path>` (optional) sets the artifact JSON a bare `#/pointer` (no
`<path>` prefix) resolves against for every `cK` on this one tag. EVERY
non-excluded numeric token on EVERY line of an in-scope file (there is no
covered-region allowlist — see "INVERTED COVERAGE MODEL" below) must have a
`cK` entry, left to right — a token with no entry, or an entry the closed
grammar below cannot parse, is a FINDING, never a silent pass (rule 9: this
gate fails closed on its OWN errors too — an unresolvable path, a JSON parse
error, a missing artifact, or an ambiguous binding is a FINDING, never
skipped).

EXPR GRAMMAR (closed, FULL enumeration below; anything else is a
`ClaimParseError`, never a silent pass):
  <path>#/<rfc6901-pointer>        a JSON pointer (path relative to repo
                                    root, or bare `#/...` under `default=`).
  abs(E) / max(E) / min(E) / mean(E)
                                    E resolves to a dict/list of JSON
                                    numbers (e.g. a `per_seed` map or a
                                    `d_values` map); the aggregate is taken
                                    over every numeric LEAF under E (`max`/
                                    `min` take the leaf's ABSOLUTE value —
                                    a magnitude claim over a committed
                                    artifact's own leaves; contrast
                                    `maxd`/`mind` below, which are SIGNED
                                    and operate over raw-recipe deltas, not
                                    artifact leaves). An optional second arg
                                    (a quoted top-level key) is excluded
                                    from the aggregate before it runs — the
                                    "the OTHER N seeds" idiom.
  count(E, '>0' | '<0' | 'len')     count of leaves satisfying the sign
                                    predicate, or the leaf/list count.
  poscount(globA, globB, field) / negcount(...) / meand(...) / maxd(...) /
  mind(...) / zerocount(...) / paircount(globA, globB, field)
                                    THE RAW-CONCORDANCE RECIPE: pairs every
                                    file matching globA with the file in
                                    globB sharing the SAME embedded seed
                                    number (`__seed(\\d+)`/`seed(\\d+)__` in
                                    the filename), computes
                                    d_i = fieldA(A) - fieldA(B) reading
                                    `field` under `.tiers.finetune_run` in
                                    each raw leg (rule (h)'s own domain has
                                    no equivalent — reused idiom, new
                                    recipe: this is the SAME "recompute the
                                    free value independently, compare"
                                    discipline rule (h)'s P3/precedence
                                    checks apply, just over raw legs instead
                                    of a merged field), then reports:
                                    `poscount`/`negcount` the positive/
                                    negative sign count; `zerocount` the
                                    EXACT-zero count (a "N/M bit-identical"
                                    claim — neither positive nor negative,
                                    so deliberately its own function, not a
                                    derived complement of the other two);
                                    `meand` the SIGNED mean; `maxd`/`mind`
                                    the SIGNED max/min over all pairs (a raw
                                    d-column's own stated endpoint, e.g.
                                    "effects to +X" — signed, unlike the
                                    unrelated `max(E)`/`min(E)` pointer
                                    aggregates above); `paircount` the
                                    number of matched pairs itself (the
                                    denominator every other member of this
                                    family shares).
  code(label, step)                 a pointer into THIS FILE's OWN nearest
                                    preceding fenced ``` code block, parsed
                                    as `<label>:` or `eps=<label> (...):`
                                    sections of `step=N l2_divergence=X`
                                    lines — the doc-internal,
                                    self-referential artifact the
                                    CPU-verifiable-demonstration tables
                                    cite (never committed as a separate
                                    JSON; the fenced block IS the committed
                                    artifact for these numbers).
  numer(D, E) / denom(N, E)         an exact rational restatement of an
                                    already-resolved E (e.g. doc prose
                                    "p = 2/4096 = 0.00048828125"): `numer`
                                    derives the numerator `round(E*D)`,
                                    `denom` the denominator `round(N/E)` —
                                    both exact integer arithmetic, so a
                                    wrong numerator/denominator FAILS at
                                    0-decimal-place precision, never
                                    silently rounds away. E is itself any
                                    expr from this grammar (a pointer, or a
                                    literal-only `ratio(...)`, letting a
                                    doc-internal fraction like "2/4096 =
                                    1/2048" be cross-checked without an
                                    artifact pointer at all).
  ratio(A, B) / absdiff(A, B)       A/B and |A-B|, full precision; A and B
                                    are each EITHER an expr from this
                                    grammar OR a bare Decimal literal (a
                                    rule PARAMETER/design constant, e.g.
                                    verifying a stated ratio like
                                    "0.10 / 0.02 = 5" against the literal
                                    inputs themselves, or deriving an exact
                                    count like "N total legs minus M clean
                                    legs" from two independently-bound
                                    pointers/recipes).
  rel(A, OP, B)                     OP in {'>','<','>=','<=','==','!='}; A/B
                                    are themselves exprs from this grammar,
                                    compared at FULL PRECISION (the
                                    resolved Decimal, never the printed
                                    rounding) — B MAY be a bare numeric
                                    literal (a rule PARAMETER like the
                                    pre-registered threshold 11 or the
                                    sign baseline 0, never a measurement
                                    itself; C10.3's own P3 precedent: a
                                    non-pointer node is checked on its own
                                    terms).
  within(A, B, tol)                 |A - B| <= tol, full precision.
  interval(A, lo, hi)               lo <= A <= hi, full precision (N/M
                                    "seeds land in [lo..hi]" containment
                                    claims).
  hist                              HISTORICAL/QUOTED marker — binds to
                                    NOTHING; the token is a deliberately
                                    quoted superseded/wrong value (the
                                    correction-of-record discipline
                                    campaign-v1/README.md:29-35,
                                    red-proof/README.md's pre-amendment
                                    stands-as-is note, and mutants/
                                    README.md's M1/M_signflip-v1 retirement
                                    notes all rely on — including a value
                                    quoted ONLY to be explicitly rejected,
                                    e.g. "the exact two-sided tail — NEVER
                                    1/4096": the rejected fraction is a
                                    quotation for contrast, not a claim)
                                    — punishing a quotation as though it
                                    were a live claim would make the
                                    correction discipline itself
                                    un-writable.
  const                             RULE-PARAMETER/PREDICTION marker —
                                    binds to NOTHING, same "consumes a tag
                                    slot, no binding attempted" treatment
                                    as `hist`, but semantically distinct: a
                                    stated design constant (a formula
                                    coefficient, a dose LABEL, an
                                    Adam-bias-correction multiplier
                                    evaluated by hand at a fixed step —
                                    values this closed grammar has no
                                    exponentiation/formula-evaluation
                                    primitive to recompute) or a
                                    PRE-REGISTERED PREDICTION (a value
                                    explicitly labeled "predicted", stated
                                    BEFORE the run it forecasts, from a
                                    hand-fit secant/linear-extrapolation
                                    model over already-bound operating-point
                                    data — never itself a fresh measurement
                                    to reconcile against a committed
                                    artifact). Under the SAME shrink-only
                                    ratchet as `ledger`
                                    (`--check-allowlist-only-shrinks`
                                    covers both), so a doc author cannot
                                    silently reclassify a real measurement
                                    as `const` to dodge binding.
  ledger                            ESCAPE — cross-checked against
                                    `ci/measurement_claims_allowlist.txt`
                                    (own shrink-only ratchet,
                                    `--check-allowlist-only-shrinks`,
                                    exact `check_allowlist_only_shrinks`
                                    shape reused). Reserved for figures this
                                    module's OWN recipe/pointer machinery
                                    genuinely cannot reach — a CLI/shell
                                    exit code (never written to any
                                    committed JSON), a product of two
                                    independently-bound factors (no
                                    multiplication primitive in this
                                    grammar), or a sum of two independent
                                    `paircount` legs (no addition
                                    primitive) — each ledger entry's
                                    trailing `# <note>` (see the allowlist
                                    file's own header) states WHICH of
                                    these classes applies; a token that IS
                                    reachable by an existing pointer or
                                    recipe function is never ledgered — see
                                    the allowlist file for the full,
                                    per-entry disposition.

EQUALITY (reused idiom, generalized for scientific notation). The doc token
is parsed as `Decimal` (native to both plain and `1.23e-4` scientific
forms); `places = -exponent` from the token's OWN `as_tuple()` (rule (h)'s
identical P3 rule — the number of digits printed after the decimal point,
which for a token like `6e-6` is 6, exactly matching what "6 zeros then a
6" means printed in full). The resolved value is quantized to that SAME
`places` with `ROUND_HALF_EVEN` and compared by DECIMAL VALUE (not string,
unlike rule (h) — a deliberate, documented adaptation: rule (h)'s corpus is
always plain-decimal so string comparison and value comparison coincide;
this corpus mixes scientific and plain notation for the SAME quantities
(`1.7542565e-4` vs `0.00017542565`), and a string compare would spuriously
FAIL two representations of an identical rounded value — Decimal value
equality after quantization preserves rule (h)'s exact precision semantics
(trailing zeros ARE significant: they set `places`, hence the quantum)
while being representation-independent).

ANTI-VACUITY (two legs, reusing rule (h)'s own two-leg discipline —
`EXPECTED_DENOMINATOR`/`--sweep`, generalized):
  (a) COVERAGE. There is no covered-region allowlist (see "INVERTED
      COVERAGE MODEL" above) — the WHOLE file, every line except a fenced
      ``` block, is scanned with the SAME tokenizer used to award tag
      entries; the resulting count is compared against a PINNED per-file
      denominator (FILE_TOKEN_DENOMINATOR) — any drift (a token entering
      or leaving scope: an edit, or an exclusion-class/fence change,
      without updating the pin) is itself a FINDING, never silently
      absorbed. Run/checkpoint PROVENANCE prose (timestamps, config
      hashes, hyperparameter settings) is out of scope not because it
      sits outside a hand-picked region, but because it is excluded via a
      closed, documented lexical class (same discipline as rule (h)'s own
      shape-label/date/version-string exclusions, extended with the
      classes this corpus's provenance prose needs: hex/sha prefixes,
      ISO-8601 timestamps, GB/MB sizes, epoch/step/t axis labels,
      named-hyperparameter assignments, and section/item/round/unit
      cross-references) — every class is enumerated at its definition
      site (`_EXTRA_EXCLUSIONS` below) with a one-line reason and a
      self-test fixture proving it excludes ONLY its own shape; there is
      no per-file/per-line carve-out that could silently narrow past that
      documented boundary the way a zone allowlist could.
  (b) AMBIGUITY. For every DIRECT pointer binding (not `hist`/`ledger`/a
      derived aggregate), an INDEPENDENT scan (`_check_ambiguity`, walking
      `_iter_leaf_paths` — a traversal never used by the production
      `Evaluator.eval` resolution path) counts how many OTHER leaves round
      to the identical value at the token's own stated precision; more
      than one DISTINCT pointer (under a DIFFERENT field name — a same-
      named mirrored field, e.g. `decision.p_value` vs `sign_test.p_value`,
      is an intentional duplicate, not a coincidence) bound to the same
      rounded value inside one artifact is a FINDING ("ambiguous
      binding"). Two classes are exempted, both because the leg's OWN
      purpose is catching a DECIMAL measurement bound to the wrong
      pointer: an INTEGER-valued token (`n_pos`/`n_neg`/dispatch-count-
      shaped fields collide constantly and harmlessly over a small state
      space — flagging every collision would be pure noise), and an
      EXACT-ZERO token (the single most common possible measurement —
      "many other leaves are also exactly 0.0" carries no signal).

RELATIONAL CLAIMS ARE IN SCOPE (this module's own reading of esc-065,
correcting an earlier narrower draft): a comparison/interval-containment
claim over operands this gate CAN resolve is exactly the "checkable core"
— `rel`/`within`/`interval` above. What stays OUT of scope is
CHARACTERIZATION: a value's qualitative label ("catastrophic", "the
detector's sensitivity ceiling", "of the same ORDER as the noise band" as
free prose, "smallest of the 12" as an unquantified superlative) is policed
by audit and the pointer-only writing style this file's own docs went
through (round 17/18), never by this gate — inventing a semantic
truth-value for a qualitative characterization is exactly the kind of
inference this module's design explicitly refuses to attempt.

SCOPE NOTE. `docs/plans/63-how-well/CONTRACT.md` (25 field=value-shaped
tokens) and `PLAN.md` (7) are NOT in `MEASUREMENT_FILES` — named
follow-up, not a silent gap: both are living design documents
(amendment-by-amendment) far larger in scope than a measurement record's
own citation surface, and under the inverted (allowlist-free) coverage
model this module now uses, adding either would mean the WHOLE of two
multi-thousand-word documents entering scope by default (never a
hand-picked subset — that judgment call is exactly what the inversion
removed) — better done as its own follow-up unit, with its own tag/binding
sweep, than smuggled into this one's already large surface.

WIRED (see `.github/workflows/ci.yml`, `Guard` matrix): a real run, a
`--self-test` run (RED fixtures from the actual `82253c1b`/`56989368`
pre-round-16-fix/`84ed6e33` historical commits plus synthetic RED cases per
rule), and `--check-allowlist-only-shrinks`.
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
assert (REPO_ROOT / "Cargo.toml").is_file(), (
    f"REPO_ROOT resolved to {REPO_ROOT}, which has no Cargo.toml — "
    "this file sits at ci/scripts/<name>.py (parents[2] == repo root); "
    "a future move must update this constant, never fail silently "
    "downstream as an empty scan (esc-063's own class)."
)

# --- reuse rule (h)'s resolution core, by IMPORT, never by copy -----------
_RH_PATH = REPO_ROOT / "ci" / "scripts" / "check_perf_claims.py"
_rh_spec = importlib.util.spec_from_file_location("_rule_h", _RH_PATH)
assert _rh_spec is not None and _rh_spec.loader is not None
rule_h = importlib.util.module_from_spec(_rh_spec)
sys.modules["_rule_h"] = rule_h  # dataclasses needs this to resolve cls.__module__
_rh_spec.loader.exec_module(rule_h)  # noqa: this IS the reuse — read-only

# --- this corpus's own scan roots ------------------------------------------

MEASUREMENT_FILES = [
    "docs/plans/63-how-well/measurements/campaign-v1/README.md",
    "docs/plans/63-how-well/measurements/campaign-v2/README.md",
    "docs/plans/63-how-well/measurements/dose-ladder/README.md",
    "docs/plans/63-how-well/measurements/red-proof/README.md",
    "docs/plans/63-how-well/measurements/red-proof/dstar/README.md",
    "docs/plans/63-how-well/mutants/README.md",
]

# INVERTED COVERAGE MODEL (round-21 finding: an ALLOWLIST of covered zones
# is structurally blind to its own incompleteness — a token OUTSIDE every
# zone was silently never scanned at all, not even a candidate for a
# finding; `mutants/README.md`'s committed-benchmark table and a fourth
# sibling of the exact class the prior zone list had already special-cased
# both sat outside every zone, undetected until adversarial perturbation).
# There is no allowlist of covered regions any more. Every claim token in
# every file in MEASUREMENT_FILES is IN SCOPE BY DEFAULT; a token escapes
# checking ONLY via one of three closed, enumerable mechanisms:
#   (a) a fenced ``` block (`_fenced_line_numbers` below) — the block's own
#       content is either a non-claim artifact (a diff/shell snippet) or,
#       for `mutants/README.md`'s CPU-demo tables, the doc-internal
#       artifact `code()` reads directly (parsed by `parse_code_blocks`,
#       which walks the SAME raw lines independently of token scanning —
#       excluding a fence from claim-token scanning and reading it as the
#       `code()` artifact are two different traversals over the same text,
#       never in tension).
#   (b) a CLOSED lexical exclusion class (`rule_h.excluded_spans` +
#       `_EXTRA_EXCLUSIONS` below, plus the heading-line and
#       `claims63:`-tag-line skips in `find_claim_tokens`) — every class is
#       named with a one-line reason at its definition site, and every
#       class has a self-test fixture proving it excludes ONLY its own
#       shape (`self_test`'s "exclusion class" fixtures below).
#   (c) the `ledger`/`const` ratchets (ci/measurement_claims_allowlist.txt,
#       shrink-only) — for a token that IS in scope but this grammar's own
#       recipe/pointer machinery genuinely cannot reach.
# A measured claim appended or edited ANYWHERE in an in-scope file is a
# claim token under this default; it REDs unless one of the three
# mechanisms above already covers it — there is no fourth way to leave
# scope, and in particular no per-file/per-line allowlist to silently miss
# updating.


def _fenced_line_numbers(lines: list[str]) -> set[int]:
    """Every 1-indexed line number that is a ``` fence delimiter OR strictly
    between a matched pair — mechanism (a) above. Reuses the exact
    delimiter-toggle rule `parse_code_blocks` already uses for its own,
    separate traversal (fenced content is read as the `code()` artifact
    there; it is EXCLUDED from claim-token scanning here — two different
    passes over identical fence boundaries, kept in sync by construction
    since both key off the same `stripped.startswith("```")` toggle)."""
    fenced: set[int] = set()
    in_block = False
    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            fenced.add(i)  # the delimiter itself is not claim-bearing text
            in_block = not in_block
            continue
        if in_block:
            fenced.add(i)
    return fenced


# Pinned per-file TOTAL in-scope token counts (mechanism (a)/(b) applied,
# whole file — no zone allowlist), using THIS module's own
# tokenizer/exclusions. This is what now catches silent scope shrink: since
# every line is a candidate, a token entering OR leaving scope (an edit, an
# exclusion-class change, a fence added/removed) changes this count and
# fails the bare run, naming both numbers — exactly
# `check_perf_claims.py`'s own `EXPECTED_DENOMINATOR` discipline, now
# applied to the WHOLE file instead of a hand-picked subset of it.
FILE_TOKEN_DENOMINATOR: dict[str, int] = {
    "docs/plans/63-how-well/measurements/campaign-v1/README.md": 39,
    "docs/plans/63-how-well/measurements/campaign-v2/README.md": 17,
    "docs/plans/63-how-well/measurements/dose-ladder/README.md": 22,
    "docs/plans/63-how-well/measurements/red-proof/README.md": 46,
    "docs/plans/63-how-well/measurements/red-proof/dstar/README.md": 31,
    "docs/plans/63-how-well/mutants/README.md": 320,
}

LEDGER_PATH = REPO_ROOT / "ci" / "measurement_claims_allowlist.txt"

# --- tokenizer: rule (h)'s NUMBER_RE widened for scientific notation ------
# (this corpus's CPU-demo table prints `1.7542565e-4` — rule (h)'s own
# corpus never needed an exponent; the widening is additive: every string
# rule (h)'s NUMBER_RE matches, this one matches identically, PLUS an
# optional `[eE][+-]?\d+` suffix.)
SCI_NUMBER_RE = re.compile(r"(?<![\w.])[−\-+]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?")

# Extra, closed, documented exclusion classes this corpus's PROVENANCE
# prose needs beyond rule (h)'s own list (reused via `rule_h.excluded_spans`
# for its own classes; these are ADDITIVE, never a narrowing of rule (h)'s
# own patterns).
_EXTRA_EXCLUSIONS = [
    # hex/sha-like run (checkpoint/patch/commit prefixes): >=6 chars,
    # mixing at least one digit and one a-f letter.
    re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{6,}\b"),
    # a date carrying a same-day revision letter (`2026-08-29c`) — rule
    # (h)'s own date pattern has no trailing-letter form.
    re.compile(r"\b\d{4}-\d{2}-\d{2}[a-z]\b"),
    # compact ISO-8601 timestamp (`20260829T055912Z`).
    re.compile(r"\b\d{8}T\d{6}Z\b"),
    # a size in GB/MB/TB/GiB/MiB, digits+unit as one label.
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:GB|MB|TB|GiB|MiB)\b"),
    # an epoch/step/t axis index (`epoch 1`, `step=1`, `t=50`, `t+1`).
    re.compile(r"\b(?:epoch|step|[Tt])\s*[=:+-]?\s*\d+\b"),
    # a named hyperparameter SETTING (experimental design, not a measured
    # outcome) — `lr=2e-4`, `eps=1e-8`, `alpha2=0.0064`, `beta1=0.9`.
    re.compile(
        r"\b(?:lr|eps|alpha2|beta1|beta2|weight_decay|margin|warmup_steps|batch|seq)"
        r"\s*=\s*[\d.eE+-]+\b"
    ),
    # a section/item cross-reference (`item 2(ii)`).
    re.compile(r"\bitem\s+\d+(?:\([ivx]+\))?\b"),
    # an exit-code label (`RC:0`).
    re.compile(r"\bRC:\d+\b"),
    # a repeat-count descriptor (`x2 seeds`).
    re.compile(r"\bx\d+\b"),
    # a dose-eps LABEL (`eps-0.50`, `eps0.02`) — a shape label, same class
    # as rule (h)'s own `b8s512`-style exclusions, never a measurement.
    re.compile(r"\beps[+-]?\d+(?:\.\d+)?\b"),
    # a markdown ordered-list marker at line start (`1. `, `3. `).
    re.compile(r"^\s*\d+\.(?=\s)"),
    # the `lr` hyperparameter in parenthetical-mention form (`lr (2e-4)`).
    re.compile(r"\blr\s*\(\s*[\d.eE+-]+\s*\)"),
    # the `(1+eps)`/`|1+eps|` dose-family name, and its negative-branch
    # sibling `(1-eps)` (a formula/family label, same class as rule (h)'s
    # own `1/√d`-style formula-fragment exclusion).
    re.compile(r"[(|]1[+-]eps[)|]"),
    # an "Acceptance N" / "finding N" / "Step N" / "Step N-M" / "Step N/M"
    # cross-reference (a CONTRACT.md item number / this doc's own numbered
    # findings list / this doc's own numbered "Step" sections).
    re.compile(r"\b(?:Acceptance|finding|Step)\s+\d+(?:[-/]\d+)?\b", re.IGNORECASE),
    # this repo's own `docs/plans/<N>-<slug>` directory-numbering
    # convention (`63-how-well`, `61-perf-unification`) — a path segment,
    # never a measurement.
    re.compile(r"\b\d+-how-well\b"),
    # this unit's own `CONTRACT N` / `unit N` / `unit-N` / `PLAN v2 delta N`
    # self-reference (the doc's own contract/unit/plan-delta number, never
    # a measurement) — same class as the `item N`/`Acceptance N` cross-refs
    # above, widened to this file's own additional numbered-reference
    # vocabulary.
    re.compile(r"\bCONTRACT\s+\d+\b"),
    re.compile(r"\bunit[\s-]+\d+\b", re.IGNORECASE),
    re.compile(r"\bdelta\s+\d+\b"),
    # an `H<n>(<m>)` requirement/heading cross-reference (`H5(1)`).
    re.compile(r"\bH\d+\(\d+\)"),
    # a `*.patch` filename (`M_eps_-0.10.patch`, `M_nobc.patch`) — the
    # patch's OWN identifier, never a measurement; the leading `M_` +
    # trailing `.patch` bracket the whole filename so an embedded dose
    # value (`M_eps_-0.10.patch`) is excluded as part of the name, not
    # read as a bare decimal.
    re.compile(r"\bM_[A-Za-z0-9_.+-]*\.patch\b"),
    # this section's own `s: 0 -> 1` / `s=0` / `s=1` interpolation-parameter
    # axis notation (the untrained/trained operating-point labels for the
    # secant slope) — same class as the epoch/step/t axis-label exclusion
    # above, for the one additional axis variable this section names.
    re.compile(r"\bs\s*[:=]\s*\d+(?:\s*->\s*\d+)?\b"),
    # an `eps < 0` / `eps > 0` sign-domain condition (which branch of the
    # signed dose family a column belongs to) — a formula/domain
    # description, same class as the named-hyperparameter-assignment
    # exclusion above, not a measurement of any particular eps value.
    re.compile(r"\beps\s*[<>]\s*0\b"),
    # a shell/CLI process exit-code label (`exit 0`) — never a stored
    # measurement, same class as the existing `RC:\d+` exit-code
    # exclusion, for this corpus's prose form of the same idiom.
    re.compile(r"\bexit\s+\d+\b"),
    # a `seed{N,M}` brace-set notation — the same "seed list" idiom the
    # `seeds? \d+(?:,|and)...` exclusion above covers for space-separated
    # lists, widened to this corpus's brace-delimited form.
    re.compile(r"\bseeds?\{[\d,]+\}"),
    # a bracketed numeric-literal test FIXTURE (`[0.5,-1.25,3.0,0.0]`) — the
    # CPU-demo's fixed input, a design constant never a measurement.
    re.compile(r"\[[-0-9.,\s]+\]"),
    # an "N-element"/"N-leg"/"N consecutive" descriptive count label.
    # (deliberately NOT "N-seed": that shape also carries a real,
    # bindable measurement — the campaign's own `12-seed gate` restating
    # `gate_seed_count` — so it is handled per-occurrence via a binding,
    # never excluded wholesale.)
    re.compile(r"\b\d+-(?:element|leg)\b"),
    re.compile(r"\b\d+\s+consecutive\b"),
    # a seed-number LIST (`seeds 9 and 12`, `seeds 9, 12`) — reused idiom,
    # generalizing rule (h)'s own single-seed `seeds? \d+` exclusion to a
    # conjunction/comma list (the exact widening its own `rows?` pattern
    # already needed — see rule (h)'s module doc audit-advisory note).
    re.compile(r"\bseeds?\s+\d+(?:\s*(?:,|and)\s*\d+)*\b"),
    # a zero/negative-index array-position label (`series[0]`, `series[-1]`
    # — the SAME class as rule (h)'s own formula-fragment exclusion: an
    # index, never a measurement).
    re.compile(r"\[-?\d+\]"),
]


def _extra_excluded_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pat in _EXTRA_EXCLUSIONS:
        spans.extend(m.span() for m in pat.finditer(text))
    return spans


def _all_excluded_spans(text: str) -> list[tuple[int, int]]:
    return rule_h.excluded_spans(text) + _extra_excluded_spans(text)


def find_claim_tokens(line: str, line_no: int) -> list[rule_h.Token]:
    """Same shape as `rule_h.find_tokens_in_row`, over `SCI_NUMBER_RE` and
    the UNION of rule (h)'s own exclusions plus this corpus's additions —
    heading lines (`^#`) are never scanned (a title/heading is a label,
    never a measurement claim; this is what keeps a bare `status: GREEN`
    in a heading from colliding with the field-adjacent form inside a
    claim paragraph). A `claims63:` TAG line itself is never scanned either
    — it is a directive, not claim-bearing content, and its own expr text
    (artifact paths, function args) is not a second population of claims
    needing its own tag one line up."""
    if line.lstrip().startswith("#"):
        return []
    if _TAG_RE.match(line):
        return []
    ex = _all_excluded_spans(line)
    out: list[rule_h.Token] = []
    for m in SCI_NUMBER_RE.finditer(line):
        if rule_h._in_any_span(m.span(), ex):
            continue
        out.append(rule_h.Token(m.group(0), line_no, m.start(), line))
    return out


# --- claim tag parsing (rule (h)'s own comment-above-line idiom) ----------

_TAG_RE = re.compile(r"^\s*<!--\s*claims63:\s*(?P<body>.*?)\s*-->\s*$")


def _entries_for_line(lines: list[str], line_no: int) -> tuple[str | None, list[str] | None]:
    """`line_no` 1-indexed. Returns `(default_path, [c1..cN])`, or
    `(None, None)` if there is no `claims63:` comment on the line
    immediately above — identical placement rule to rule (h)'s own
    `_entries_for_row`."""
    above_idx = line_no - 2
    if above_idx < 0:
        return None, None
    m = _TAG_RE.match(lines[above_idx])
    if not m:
        return None, None
    body = m.group("body")
    default_path: str | None = None
    entries: dict[int, str] = {}
    for part in _split_top_level(body, ";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("default="):
            default_path = part[len("default=") :].strip()
            continue
        km = re.match(r"^c(\d+)\s*=\s*(.*)$", part)
        if not km:
            continue
        entries[int(km.group(1))] = km.group(2).strip()
    if not entries:
        return default_path, []
    n = max(entries)
    return default_path, [entries.get(i, "") for i in range(1, n + 1)]


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep` only outside of `(...)`/`'...'` nesting — reused
    idiom (same purpose as rule (h)'s own `_split_top_level`, written fresh
    here since this grammar's quoting/nesting differs)."""
    out: list[str] = []
    depth = 0
    in_quote = False
    cur: list[str] = []
    for ch in s:
        if in_quote:
            cur.append(ch)
            if ch == "'":
                in_quote = False
            continue
        if ch == "'":
            in_quote = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


# --- errors -----------------------------------------------------------------


class ClaimParseError(Exception):
    pass


class ResolutionError(Exception):
    pass


# --- artifact loading + pointer resolution (reusing rule_h._rfc6901_walk) -


class Loader:
    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def load(self, rel_path: str) -> object:
        if rel_path in self._cache:
            return self._cache[rel_path]
        abspath = REPO_ROOT / rel_path
        if not abspath.is_file():
            raise ResolutionError(f"artifact not found: {rel_path}")
        try:
            with open(abspath) as fh:
                doc = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ResolutionError(f"invalid JSON: {rel_path} ({exc})") from exc
        self._cache[rel_path] = doc
        return doc

    def resolve(self, rel_path: str, pointer: str) -> object:
        doc = self.load(rel_path)
        try:
            value, _ = rule_h._rfc6901_walk(doc, pointer)
        except (rule_h.ResolutionError, IndexError, KeyError, TypeError) as exc:
            raise ResolutionError(f"{rel_path}#{pointer}: {exc}") from exc
        return value


def _iter_numeric_leaves(node) -> list[Decimal]:
    out: list[Decimal] = []

    def walk(n):
        if isinstance(n, bool):
            return
        if isinstance(n, (int, float)):
            out.append(Decimal(str(n)))
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return out


def _iter_leaf_paths(node, prefix: str = "") -> list[tuple[str, Decimal]]:
    """Every `(pointer, Decimal-value)` numeric leaf under `node` — used by
    the anti-vacuity ambiguity leg's independent scan."""
    out: list[tuple[str, Decimal]] = []

    def walk(n, p):
        if isinstance(n, bool):
            return
        if isinstance(n, (int, float)):
            out.append((p, Decimal(str(n))))
        elif isinstance(n, dict):
            for k, v in n.items():
                walk(v, f"{p}/{k}")
        elif isinstance(n, list):
            for i, v in enumerate(n):
                walk(v, f"{p}/{i}")

    walk(node, prefix)
    return out


# --- fenced-code-block self-referential artifact (mutants/README.md) ------

_CODE_BLOCK_HEADER_RE = re.compile(r"^eps=([+-]?[\d.]+)\s*\(")
# a bare `<Label>:` header (no `eps=`/parenthetical) — the M1-family
# demonstration blocks (`M_nobc:` / `M_signflip:` / `M_signflip_v2:`) use
# this simpler shape since they are not part of the signed-`eps` family and
# have no ratio-scaling claim attached to their own label.
_CODE_BLOCK_LABEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")
_CODE_BLOCK_STEP_RE = re.compile(r"^step=(\d+)\s+l2_divergence=([0-9.eE+-]+)\s*$")


def parse_code_blocks(lines: list[str]) -> dict[str, dict[int, Decimal]]:
    """Parse EVERY fenced ``` block in `lines` for `eps=<label> (...):` /
    `<Label>:` / `step=N l2_divergence=X` lines — this doc-internal table IS
    the committed artifact for the CPU-verifiable-demonstration numbers
    (never a separate JSON; `git log` on this file is this data's own
    history)."""
    out: dict[str, dict[int, Decimal]] = {}
    in_block = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_block = not in_block
            current = None
            continue
        if not in_block:
            continue
        m = _CODE_BLOCK_HEADER_RE.match(stripped)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            continue
        m = _CODE_BLOCK_LABEL_RE.match(stripped)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            continue
        m = _CODE_BLOCK_STEP_RE.match(stripped)
        if m and current is not None:
            step = int(m.group(1))
            out[current][step] = Decimal(m.group(2))
    return out


# --- raw-concordance recipe (poscount/negcount/meand) ----------------------

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(path: str) -> str | None:
    m = _SEED_RE.search(Path(path).name)
    return m.group(1) if m else None


def _raw_field(path: str, field_name: str) -> Decimal:
    with open(REPO_ROOT / path) as fh:
        doc = json.load(fh)
    tiers = doc.get("tiers", {})
    fr = tiers.get("finetune_run")
    if fr is None or field_name not in fr:
        raise ResolutionError(f"{path}: tiers.finetune_run.{field_name} missing")
    return Decimal(str(fr[field_name]))


def _paired_deltas(glob_a: str, glob_b: str, field_name: str) -> list[Decimal]:
    files_a = sorted(globmod.glob(str(REPO_ROOT / glob_a)))
    files_b = sorted(globmod.glob(str(REPO_ROOT / glob_b)))
    by_seed_b = {}
    for fb in files_b:
        seed = _seed_of(fb)
        if seed is not None:
            by_seed_b[seed] = fb
    if not files_a or not files_b:
        raise ResolutionError(f"empty glob: {glob_a!r} or {glob_b!r}")
    deltas: list[Decimal] = []
    for fa in files_a:
        seed = _seed_of(fa)
        if seed is None or seed not in by_seed_b:
            raise ResolutionError(f"no seed-paired file for {fa!r} under {glob_b!r}")
        va = _raw_field(str(Path(fa).relative_to(REPO_ROOT)), field_name)
        vb = _raw_field(str(Path(by_seed_b[seed]).relative_to(REPO_ROOT)), field_name)
        deltas.append(va - vb)
    return deltas


# --- expression parser + evaluator ------------------------------------------


@dataclass
class Expr:
    kind: str
    args: list = field(default_factory=list)


def _tokenize_expr(s: str) -> list[str]:
    toks: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "(),":
            toks.append(c)
            i += 1
            continue
        if c == "'":
            j = s.index("'", i + 1)
            toks.append(s[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < n and s[j] not in "(),":
            j += 1
        toks.append(s[i:j].strip())
        i = j
    return [t for t in toks if t != ""]


def parse_expr(raw: str) -> Expr:
    raw = raw.strip()
    if raw == "":
        raise ClaimParseError("empty expression")
    if raw == "hist":
        return Expr("hist", [])
    if raw == "ledger":
        return Expr("ledger", [])
    if raw == "const":
        return Expr("const", [])
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", raw, re.DOTALL)
    if m:
        fname = m.group(1)
        args_raw = _split_top_level(m.group(2), ",")
        if fname not in _FUNCS:
            raise ClaimParseError(f"unknown function {fname!r}")
        return Expr(fname, [a.strip() for a in args_raw])
    # a bare pointer: <path>#/<ptr>  or  #/<ptr>  (uses tag's default=)
    if "#" in raw:
        return Expr("ptr", [raw])
    raise ClaimParseError(f"unparseable expression: {raw!r}")


_FUNCS = {
    "abs",
    "max",
    "min",
    "mean",
    "count",
    "poscount",
    "negcount",
    "meand",
    "maxd",
    "mind",
    "zerocount",
    "paircount",
    "code",
    "rel",
    "within",
    "interval",
    "numer",
    "denom",
    "ratio",
    "absdiff",
}


@dataclass
class Binding:
    value: Decimal | None  # None for hist
    is_hist: bool = False
    is_ledger: bool = False
    source_pointer: str | None = None  # for the ambiguity leg
    source_path: str | None = None


class Evaluator:
    def __init__(self, loader: Loader, code_blocks: dict[str, dict[int, Decimal]], default_path: str | None):
        self.loader = loader
        self.code_blocks = code_blocks
        self.default_path = default_path

    def _resolve_ptr_literal(self, raw: str) -> tuple[str, str, Decimal | object]:
        if "#" not in raw:
            raise ClaimParseError(f"pointer missing '#': {raw!r}")
        path_part, ptr = raw.split("#", 1)
        path_part = path_part.strip()
        if path_part == "":
            if self.default_path is None:
                raise ClaimParseError(f"bare pointer {raw!r} with no default= on this tag")
            path_part = self.default_path
        value = self.loader.resolve(path_part, ptr)
        return path_part, ptr, value

    def eval(self, expr: Expr) -> Binding:
        if expr.kind == "hist":
            return Binding(None, is_hist=True)
        if expr.kind == "ledger":
            return Binding(None, is_ledger=True)
        if expr.kind == "const":
            return Binding(None, is_hist=True)  # a rule PARAMETER, never a
            # measurement — same "consumes a tag slot, no binding attempted"
            # treatment as `hist`, but semantically distinct (see module
            # doc's `rel()` note): a stated threshold/baseline, not a quote.
        if expr.kind == "ptr":
            path, ptr, value = self._resolve_ptr_literal(expr.args[0])
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ClaimParseError(f"pointer {expr.args[0]!r} did not resolve to a scalar")
            if isinstance(value, str):
                # string-valued leaf (e.g. detected/status/red_proof_verdict)
                return Binding(value, source_pointer=ptr, source_path=path)  # type: ignore[arg-type]
            return Binding(Decimal(str(value)), source_pointer=ptr, source_path=path)
        if expr.kind in ("abs", "max", "min", "mean"):
            path, ptr, node = self._resolve_ptr_literal(expr.args[0])
            if len(expr.args) > 1 and isinstance(node, dict):
                # an optional SECOND arg (a quoted top-level key) is
                # EXCLUDED before aggregating — the dstar per-seed record's
                # own "the OTHER 11 seeds" claim (seed 2 named and quoted
                # separately, the remaining per-seed dict aggregated
                # without it).
                excl_key = expr.args[1].strip().strip("'")
                node = {k: v for k, v in node.items() if k != excl_key}
            leaves = _iter_numeric_leaves(node)
            if not leaves:
                raise ResolutionError(f"{expr.args[0]}: no numeric leaves to aggregate")
            if expr.kind == "abs":
                if len(leaves) != 1:
                    raise ClaimParseError("abs() takes a scalar pointer, not an aggregate")
                return Binding(abs(leaves[0]), source_pointer=ptr, source_path=path)
            if expr.kind == "max":
                v = max(abs(x) for x in leaves)
            elif expr.kind == "min":
                v = min(abs(x) for x in leaves)
            else:
                v = sum(leaves) / Decimal(len(leaves))
            return Binding(v, source_pointer=ptr, source_path=path)
        if expr.kind == "count":
            path, ptr, node = self._resolve_ptr_literal(expr.args[0])
            leaves = _iter_numeric_leaves(node)
            pred = expr.args[1].strip("'")
            if pred == "len":
                v = Decimal(len(node) if isinstance(node, (list, dict)) else 1)
            elif pred == ">0":
                v = Decimal(sum(1 for x in leaves if x > 0))
            elif pred == "<0":
                v = Decimal(sum(1 for x in leaves if x < 0))
            else:
                raise ClaimParseError(f"unknown count predicate {pred!r}")
            return Binding(v, source_pointer=ptr, source_path=path)
        if expr.kind in ("poscount", "negcount", "meand", "maxd", "mind", "zerocount"):
            glob_a, glob_b, field_name = (a.strip().strip("'") for a in expr.args)
            deltas = _paired_deltas(glob_a, glob_b, field_name)
            if expr.kind == "poscount":
                v = Decimal(sum(1 for d in deltas if d > 0))
            elif expr.kind == "negcount":
                v = Decimal(sum(1 for d in deltas if d < 0))
            elif expr.kind == "zerocount":
                # a "N/M bit-identical" claim: the count of EXACT-zero
                # paired deltas — deliberately distinct from
                # poscount/negcount (a bit-identical leg is neither
                # positive nor negative), reusing the identical
                # `_paired_deltas` recompute discipline.
                v = Decimal(sum(1 for d in deltas if d == 0))
            elif expr.kind == "maxd":
                # the SIGNED max (not abs — a "sits inside/effects to +X"
                # endpoint claim over a raw d-column states the actual
                # signed extreme, not a magnitude; distinct from the
                # existing `max(E)` pointer-aggregate, which DOES take abs
                # by design for its own, unrelated corpus of claims).
                v = max(deltas)
            elif expr.kind == "mind":
                v = min(deltas)
            else:
                v = sum(deltas) / Decimal(len(deltas))
            return Binding(v)
        if expr.kind == "paircount":
            glob_a, glob_b, field_name = (a.strip().strip("'") for a in expr.args)
            deltas = _paired_deltas(glob_a, glob_b, field_name)
            return Binding(Decimal(len(deltas)))
        if expr.kind in ("numer", "denom"):
            # An exact rational restatement of an already-pointer-bound
            # value (e.g. doc prose "p = 2/4096 = 0.00048828125"): the
            # SECOND operand is the pointer/expr the fraction restates;
            # `numer(D, E)` derives the exact numerator (round(E * D)),
            # `denom(N, E)` derives the exact denominator (round(N / E)) —
            # both exact integer arithmetic when the restatement is
            # correct, so a wrong numerator/denominator FAILS at integer
            # (0 decimal place) precision, never silently rounds away.
            other = int(expr.args[0].strip())
            e = self.eval(parse_expr(expr.args[1]))
            assert isinstance(e.value, Decimal)
            if expr.kind == "numer":
                v = (e.value * Decimal(other)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
            else:
                v = (Decimal(other) / e.value).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
            return Binding(v)
        if expr.kind in ("ratio", "absdiff"):
            a = self._const_or_eval(expr.args[0].strip())
            b = self._const_or_eval(expr.args[1].strip())
            v = (a / b) if expr.kind == "ratio" else abs(a - b)
            return Binding(v)
        if expr.kind == "code":
            label = expr.args[0].strip().strip("'")
            step = int(expr.args[1])
            table = self.code_blocks.get(label)
            if table is None or step not in table:
                raise ResolutionError(f"code block {label!r} step {step} not found")
            return Binding(table[step])
        if expr.kind == "rel":
            a = self.eval(parse_expr(expr.args[0]))
            op = expr.args[1].strip("'")
            b_raw = expr.args[2].strip()
            b_val = self._const_or_eval(b_raw)
            assert isinstance(a.value, Decimal) and isinstance(b_val, Decimal)
            ok = {
                ">": a.value > b_val,
                "<": a.value < b_val,
                ">=": a.value >= b_val,
                "<=": a.value <= b_val,
                "==": a.value == b_val,
                "!=": a.value != b_val,
            }[op]
            if not ok:
                raise ResolutionError(f"relation false: {a.value} {op} {b_val}")
            return Binding(a.value)
        if expr.kind == "within":
            a = self.eval(parse_expr(expr.args[0]))
            b_val = self._const_or_eval(expr.args[1].strip())
            tol = Decimal(expr.args[2].strip())
            assert isinstance(a.value, Decimal) and isinstance(b_val, Decimal)
            if abs(a.value - b_val) > tol:
                raise ResolutionError(f"within() false: |{a.value} - {b_val}| > {tol}")
            return Binding(a.value)
        if expr.kind == "interval":
            a = self.eval(parse_expr(expr.args[0]))
            lo = Decimal(expr.args[1].strip())
            hi = Decimal(expr.args[2].strip())
            assert isinstance(a.value, Decimal)
            if not (lo <= a.value <= hi):
                raise ResolutionError(f"interval() false: {a.value} not in [{lo}, {hi}]")
            return Binding(a.value)
        raise ClaimParseError(f"unhandled expr kind {expr.kind!r}")

    def _const_or_eval(self, raw: str) -> Decimal:
        try:
            return Decimal(raw)
        except InvalidOperation:
            return self.eval(parse_expr(raw)).value  # type: ignore[return-value]


# --- equality (reused idiom, generalized for scientific notation) ---------


def _decimal_places_of(token_text: str) -> tuple[Decimal, int]:
    t = token_text.replace("−", "-").lstrip("+")
    try:
        d = Decimal(t)
    except InvalidOperation as exc:
        raise ClaimParseError(f"not a valid number: {token_text!r}") from exc
    exp = d.as_tuple().exponent
    places = -exp if isinstance(exp, int) and exp < 0 else 0
    return d, places


def _quantize(value: Decimal, places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-places) if places else Decimal(1)
    return value.quantize(quantum, rounding=ROUND_HALF_EVEN)


def compare_token(token_text: str, value: Decimal) -> tuple[bool, Decimal]:
    tok_d, places = _decimal_places_of(token_text)
    got = _quantize(value, places)
    return got == tok_d, got


def compare_string_token(token_text: str, value: str) -> bool:
    return token_text.strip("`") == value


# --- findings / scanning ----------------------------------------------------


@dataclass
class Finding:
    file: str
    line: int
    col: int
    message: str


def _in_scope_lines(lines: list[str]) -> set[int]:
    """Every 1-indexed line number in the file EXCEPT fenced lines — mechanism
    (a). This is the WHOLE file, replacing the old CLAIM_ZONES allowlist: a
    token escapes scanning from here only via mechanism (b) (a lexical
    exclusion class, applied per-token inside `find_claim_tokens`) or (c)
    (a `ledger`/`const` tag entry, applied per-token inside `scan_file`)."""
    fenced = _fenced_line_numbers(lines)
    return {i for i in range(1, len(lines) + 1) if i not in fenced}


# the binding-breakdown categories the gate's headline prints (B4: "print
# the binding breakdown ... in the gate's output line" — never a flat
# bound-vs-findings count that hides how much of "bound" is a real
# pointer/recipe recompute vs an allowlisted escape).
_BREAKDOWN_KINDS = ("ptr", "recipe", "hist", "const", "ledger")
_RECIPE_KINDS = {
    "abs",
    "max",
    "min",
    "mean",
    "count",
    "poscount",
    "negcount",
    "meand",
    "maxd",
    "mind",
    "zerocount",
    "paircount",
    "code",
    "rel",
    "within",
    "interval",
    "numer",
    "denom",
    "ratio",
    "absdiff",
}


def scan_file(
    rel_path: str, loader: Loader, ledger: set[str] | None = None
) -> tuple[list[Finding], int, int, dict[str, int]]:
    """Returns `(findings, tokens_in_zone, tokens_bound, breakdown)` —
    `breakdown` counts successfully-bound tokens per `_BREAKDOWN_KINDS`
    category (a finding is never counted in any category)."""
    ledger = ledger if ledger is not None else set()
    abspath = REPO_ROOT / rel_path
    text = abspath.read_text()
    lines = text.splitlines()
    code_blocks = parse_code_blocks(lines) if "mutants" in rel_path else {}
    scope_lines = _in_scope_lines(lines)
    findings: list[Finding] = []
    tokens_in_zone = 0
    tokens_bound = 0
    breakdown: dict[str, int] = {k: 0 for k in _BREAKDOWN_KINDS}

    for line_no, line in enumerate(lines, start=1):
        if line_no not in scope_lines:
            continue
        toks = find_claim_tokens(line, line_no)
        if not toks:
            continue
        tokens_in_zone += len(toks)
        default_path, entries = _entries_for_line(lines, line_no)
        if entries is None:
            findings.append(
                Finding(rel_path, line_no, toks[0].col, f"{len(toks)} claim token(s), no claims63 tag above")
            )
            continue
        if len(entries) < len(toks):
            findings.append(
                Finding(
                    rel_path,
                    line_no,
                    toks[0].col,
                    f"{len(toks)} claim token(s) but only {len(entries)} tag entries",
                )
            )
        evaluator = Evaluator(loader, code_blocks, default_path)
        for tok, entry in zip(toks, entries):
            if entry == "":
                findings.append(Finding(rel_path, line_no, tok.col, f"token {tok.text!r}: empty tag entry"))
                continue
            try:
                expr = parse_expr(entry)
            except ClaimParseError as exc:
                findings.append(Finding(rel_path, line_no, tok.col, f"token {tok.text!r}: parse error: {exc}"))
                continue
            if expr.kind == "hist":
                tokens_bound += 1
                breakdown["hist"] += 1
                continue
            if expr.kind in ("ledger", "const"):
                # `const` is under the SAME shrink-only ratchet as `ledger`
                # (round-20 audit finding B4) -- a `const:`-prefixed key in
                # the SAME file, so a doc author cannot silently reclassify
                # a real measurement as `const` to dodge both the pointer/
                # recipe binding AND the ledger's own review gate.
                prefix = "const:" if expr.kind == "const" else ""
                key = f"{prefix}{rel_path}:{tok.text}:{line_hash(line)}:{tok.col}"
                if key not in ledger:
                    findings.append(
                        Finding(
                            rel_path,
                            line_no,
                            tok.col,
                            f"token {tok.text!r}: {expr.kind}-escaped but key {key!r} not in "
                            f"{LEDGER_PATH.name} (grows ONLY via a human-reviewed PR)",
                        )
                    )
                else:
                    tokens_bound += 1
                    breakdown[expr.kind] += 1
                continue
            try:
                binding = evaluator.eval(expr)
            except (ResolutionError, ClaimParseError) as exc:
                findings.append(Finding(rel_path, line_no, tok.col, f"token {tok.text!r} ({entry}): {exc}"))
                continue
            if isinstance(binding.value, str):
                ok = compare_string_token(tok.text, binding.value)
                if not ok:
                    findings.append(
                        Finding(
                            rel_path,
                            line_no,
                            tok.col,
                            f"token {tok.text!r} != resolved string {binding.value!r} ({entry})",
                        )
                    )
                else:
                    tokens_bound += 1
                    breakdown["ptr"] += 1
                continue
            ok, got = compare_token(tok.text, binding.value)  # type: ignore[arg-type]
            if not ok:
                findings.append(
                    Finding(
                        rel_path,
                        line_no,
                        tok.col,
                        f"token {tok.text!r} != resolved {got} (full precision {binding.value}) via {entry}",
                    )
                )
                continue
            tokens_bound += 1
            breakdown["ptr" if expr.kind == "ptr" else "recipe"] += 1
            # anti-vacuity leg (b) applies only to DIRECT pointer bindings
            # (`expr.kind == "ptr"`) — an aggregate (`max`/`min`/`mean`)
            # binding's `source_pointer` names the CONTAINER it aggregated
            # over, so "another leaf under that same container equals the
            # aggregate" is expected BY CONSTRUCTION (that leaf is very
            # likely the one the aggregate picked), never a sign of a
            # wrong pointer.
            if expr.kind == "ptr" and binding.source_path is not None and binding.source_pointer is not None:
                amb = _check_ambiguity(loader, binding.source_path, binding.source_pointer, tok.text, binding.value)
                if amb:
                    findings.append(Finding(rel_path, line_no, tok.col, amb))
    return findings, tokens_in_zone, tokens_bound, breakdown


def _check_ambiguity(loader: Loader, path: str, pointer: str, token_text: str, value: Decimal) -> str | None:
    """Anti-vacuity leg (b): an INDEPENDENT leaf scan (never calling
    `Loader.resolve`/`rule_h._rfc6901_walk` again) — walks the artifact's
    OWN already-loaded document (no re-parse, but a structurally distinct
    traversal: `_iter_leaf_paths`, never used by the production `eval()`
    path) counting other leaves that round to the SAME token at the SAME
    precision."""
    doc = loader.load(path)
    tok_d, places = _decimal_places_of(token_text)
    if tok_d == 0:
        # exact zero (a floor/delta/clean value) is the single most common
        # possible measurement — a "many other leaves are also exactly
        # 0.0" finding carries no information about a wrong pointer.
        return None
    if places == 0:
        # INTEGER-valued fields (dispatch/seed/n_pos-shaped counters) share
        # small state spaces and collide constantly and harmlessly by
        # construction — this leg's real purpose is catching a DECIMAL
        # measurement (mean_d/p_value/spread-shaped) bound to the wrong
        # pointer, where a coincidental match at full printed precision is
        # a meaningful signal; a real artifact CAN (and here does) carry
        # two unrelated integer counters that happen to share one value
        # (`attention_block_flash_fused_dispatches` ==
        # `geglu_fused_dispatches` == 3276 in one committed probe leg) —
        # never itself evidence of a wrong pointer.
        return None
    source_field = pointer.rsplit("/", 1)[-1]
    matches = []
    for leaf_ptr, leaf_val in _iter_leaf_paths(doc):
        if leaf_ptr == pointer:
            continue
        # A leaf under a DIFFERENT parent but the SAME field name (e.g.
        # `decision/p_value` vs `sign_test/p_value`) is an intentionally
        # mirrored quantity, not a coincidental collision — only a
        # DIFFERENT field name matching is a real ambiguity.
        if leaf_ptr.rsplit("/", 1)[-1] == source_field:
            continue
        q = _quantize(leaf_val, places)
        if q == tok_d:
            matches.append(leaf_ptr)
    if matches:
        return f"ambiguous: {path}#{pointer} == {token_text} but so does {matches[0]} (+{len(matches)-1} more)"
    return None


# --- ledger (shrink-only ratchet, reused idiom) -----------------------------


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def line_hash(text: str) -> str:
    return hashlib.sha1(normalize_line(text).encode("utf-8")).hexdigest()


def load_ledger() -> set[str]:
    """Every entry line is `key` or `key  # <note>` — the note (required for
    every surviving entry as of the round-20 sweep: WHY this specific token
    is genuinely unreachable by any pointer/recipe binding, never a bare
    unexplained escape) is split off before membership-checking; it is
    documentation only and never part of the key itself."""
    if not LEDGER_PATH.is_file():
        return set()
    out = set()
    for line in LEDGER_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split(" #", 1)[0].strip()
        if key:
            out.add(key)
    return out


def check_allowlist_only_shrinks() -> int:
    """Exact `check_perf_claims.py --check-allowlist-only-shrinks` shape,
    reused idiom: fetch `origin/main`'s copy, fail CLOSED on a failed fetch
    or unresolvable ref, an explicit `git cat-file -e` bootstrap arm (this
    branch's entries establish the baseline when `origin/main` has no
    ledger file yet — never a silent `old_text = ""` fallthrough, which
    would let `check` swallow a `git show` failure that is NOT actually a
    missing-file bootstrap, e.g. a detached/shallow fetch that resolved the
    ref but not the blob), then fail on any entry THIS branch adds relative
    to it."""
    try:
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: could not fetch origin/main: {exc}", file=sys.stderr)
        return 1

    rev = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if rev.returncode != 0:
        print(f"FAIL: origin/main does not resolve: {rev.stderr.strip()}", file=sys.stderr)
        return 1

    rel = LEDGER_PATH.relative_to(REPO_ROOT).as_posix()
    file_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"origin/main:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    new = load_ledger()

    if not file_exists:
        print(
            "measurement-claims-allowlist-only-shrinks: OK (bootstrap) — origin/main "
            f"resolves ({rev.stdout.strip()}) but has no {rel} yet: this branch's "
            f"{len(new)} entries establish the baseline."
        )
        return 0

    old_text = subprocess.run(
        ["git", "show", f"origin/main:{rel}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    old = {
        ln.strip().split(" #", 1)[0].strip()
        for ln in old_text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    old.discard("")
    added = new - old
    if added:
        print("FAIL: ledger grew relative to origin/main:", file=sys.stderr)
        for entry in sorted(added):
            print(f"  + {entry}", file=sys.stderr)
        return 1
    print(f"PASS: ledger only shrunk or held ({len(old)} -> {len(new)})")
    return 0


# --- coverage anti-vacuity leg (a) -----------------------------------------


def check_coverage() -> list[str]:
    problems = []
    for rel_path in MEASUREMENT_FILES:
        abspath = REPO_ROOT / rel_path
        lines = abspath.read_text().splitlines()
        scope_lines = _in_scope_lines(lines)
        total = 0
        for line_no in sorted(scope_lines):
            total += len(find_claim_tokens(lines[line_no - 1], line_no))
        expected = FILE_TOKEN_DENOMINATOR.get(rel_path)
        if expected is None:
            problems.append(f"{rel_path}: no FILE_TOKEN_DENOMINATOR pinned")
        elif total != expected:
            problems.append(f"{rel_path}: file token count drifted: pinned {expected}, counted {total}")
    return problems


# --- report / gate -----------------------------------------------------------


def run_real_tree() -> tuple[list[Finding], int, int, dict[str, int]]:
    ledger = load_ledger()
    loader = Loader()
    all_findings: list[Finding] = []
    total_zone = 0
    total_bound = 0
    total_breakdown: dict[str, int] = {k: 0 for k in _BREAKDOWN_KINDS}
    for rel_path in MEASUREMENT_FILES:
        findings, in_zone, bound, breakdown = scan_file(rel_path, loader, ledger)
        all_findings.extend(findings)
        total_zone += in_zone
        total_bound += bound
        for k, v in breakdown.items():
            total_breakdown[k] += v
    return all_findings, total_zone, total_bound, total_breakdown


def gate() -> int:
    problems = check_coverage()
    findings, in_zone, bound, breakdown = run_real_tree()
    ok = True
    for p in problems:
        print(f"FAIL (coverage): {p}", file=sys.stderr)
        ok = False
    for f in findings:
        print(f"FAIL: {f.file}:{f.line}:{f.col}: {f.message}", file=sys.stderr)
        ok = False
    # B4 (round-20 audit): the headline breakdown -- how much of "bound" is
    # a real pointer/recipe recompute vs an allowlisted escape (`ledger`),
    # a design constant/prediction (`const`, SAME ratchet as `ledger`), or a
    # deliberately-quoted historical value (`hist`, binds to nothing) --
    # never a flat "N/N bound" that launders an escape as though it were a
    # recompute.
    print(
        f"in-scope tokens: {in_zone}, bound OK: {bound}, findings: {len(findings)} "
        f"[ptr={breakdown['ptr']} recipe={breakdown['recipe']} "
        f"ledger={breakdown['ledger']} const={breakdown['const']} hist={breakdown['hist']}]"
    )
    if ok:
        print("PASS")
        return 0
    return 1


# --- self-test ---------------------------------------------------------------


def _write_tmp(tmpdir: Path, name: str, content: str) -> Path:
    p = tmpdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def self_test() -> int:
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool):
        if not cond:
            failures.append(name)
        print(("PASS" if cond else "FAIL") + f": {name}")

    # --- equality core (generalized P3) ------------------------------------
    check("equality: legit coarse rounding p=0.00635", compare_token("0.00635", Decimal("13") / Decimal("2048"))[0])
    check("equality: legit coarse rounding p=0.34", compare_token("0.34", Decimal("0.34375"))[0])
    check(
        "equality: sci-notation exact",
        compare_token("1.7542565e-4", Decimal("1.7542565e-4"))[0],
    )
    check(
        "equality: LIVE DEFECT 25.076 vs true 25.074... is FALSE",
        not compare_token("25.076", Decimal("4.3986836e-3") / Decimal("1.7542565e-4"))[0],
    )
    check(
        "equality: fixed 25.074 vs true is TRUE",
        compare_token("25.074", Decimal("4.3986836e-3") / Decimal("1.7542565e-4"))[0],
    )
    check(
        "equality: LIVE DEFECT 5.017 vs true 5.0175... is FALSE",
        not compare_token("5.017", Decimal("8.801983e-4") / Decimal("1.7542565e-4"))[0],
    )
    check(
        "equality: fixed 5.018 vs true is TRUE",
        compare_token("5.018", Decimal("8.801983e-4") / Decimal("1.7542565e-4"))[0],
    )
    check(
        "equality: LIVE DEFECT ~6e-6 vs true diff ~5.664e-7 is FALSE",
        not compare_token("6e-6", abs(Decimal("4.39925e-3") - Decimal("4.3986836e-3")))[0],
    )
    check(
        "equality: fixed ~6e-7 vs true diff is TRUE",
        compare_token("6e-7", abs(Decimal("4.39925e-3") - Decimal("4.3986836e-3")))[0],
    )

    # --- recipe: raw-concordance recompute (validated against the real tree)
    deltas = _paired_deltas(
        "docs/plans/63-how-well/measurements/red-proof/raw/nobc__seed*.json",
        "docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json",
        "held_out_example_mean",
    )
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    mean_d = sum(deltas) / Decimal(len(deltas))
    check("recipe: nobc raw n_pos=5", n_pos == 5)
    check("recipe: nobc raw n_neg=7", n_neg == 7)
    check("recipe: nobc raw mean_d rounds to -0.0183", compare_token("-0.0183", mean_d)[0])

    # --- rule 1: pointer resolution over the real tree ----------------------
    loader = Loader()
    ok1 = True
    try:
        loader.resolve(
            "docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json",
            "/mutant_dose_ladder/doses/0/per_seed/2/d_i",
        )
    except ResolutionError:
        ok1 = False
    check("rule1: doses[0].per_seed[2].d_i resolves on the real tree", ok1)

    ok2 = False
    try:
        loader.resolve(
            "docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json",
            "/mutant_dose_ladder/red_proof/0/per_seed/2/d_i",
        )
    except ResolutionError:
        ok2 = True
    check("rule1 RED fixture: red_proof[0].per_seed does not exist (8-key projection)", ok2)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # --- (a) 82253c1b/56989368-shaped RED: wrong pointer `red_proof[0].per_seed`
        art_dir = tmp / "artifact"
        art_dir.mkdir()
        artifact = {
            "mutant_dose_ladder": {
                "doses": [{"per_seed": {"2": {"d_i": 0.17900}}}],
                "red_proof": [{"dose_label": "redproof-signflip-v2", "n_pos": 12}],
            }
        }
        (art_dir / "report.json").write_text(json.dumps(artifact))
        loader2 = Loader()
        rel = str((art_dir / "report.json").relative_to(REPO_ROOT)) if _under_repo(art_dir) else None
        # can't relativize outside REPO_ROOT via Loader (it always joins
        # REPO_ROOT); test the resolver directly against the parsed doc
        # instead, using rule_h's own walker (the exact function reused in
        # production).
        red_ok = True
        try:
            rule_h._rfc6901_walk(artifact, "/mutant_dose_ladder/red_proof/0/per_seed")
        except rule_h.ResolutionError:
            red_ok = False
        check("fixture(a) 82253c1b/56989368-shaped: red_proof[0].per_seed FAILS to resolve", not red_ok)
        doses_ok = True
        try:
            rule_h._rfc6901_walk(artifact, "/mutant_dose_ladder/doses/0/per_seed/2/d_i")
        except rule_h.ResolutionError:
            doses_ok = False
        check("fixture(a) corrected pointer doses[0].per_seed[2].d_i resolves", doses_ok)

        # --- (b) 84ed6e33-shaped RED: "+11.2..+20.1" is < 4 printed decimal
        # digits of precision relative to the artifact's own values,
        # meaning the doc token, quantized at ITS OWN precision (1 decimal
        # place), does NOT bind to a unique per_seed value -- multiple
        # distinct seeds round to "11.2"/"20.1" (representable, but
        # AMBIGUOUS: this is exactly why round-17 replaced it with the
        # 5-decimal-place form). Demonstrated directly against the real
        # dstar artifact.
        real_doses = loader.load(
            "docs/plans/63-how-well/measurements/red-proof/dstar/finetune_run_ab_report.json"
        )
        per_seed = real_doses["mutant_dose_ladder"]["doses"][0]["per_seed"]  # type: ignore[index]
        others = [Decimal(str(v["d_i"])) for k, v in per_seed.items() if k != "2"]
        low_prec_matches = [v for v in others if compare_token("11.2", v)[0] or compare_token("20.1", v)[0]]
        check(
            "fixture(b) 84ed6e33-shaped: coarse '11.2'/'20.1' endpoints are ambiguous "
            f"({len(low_prec_matches)} of 11 seeds round-match at 1 decimal place)",
            len(low_prec_matches) >= 2,
        )
        precise_min = min(others)
        precise_max = max(others)
        check(
            "fixture(b) fixed: precise 5-decimal endpoints uniquely bind",
            compare_token("11.19540", precise_min)[0] and compare_token("20.08700", precise_max)[0],
        )

        # --- 33f6ebae-shaped RED: "0.179 sits inside |d_i| up to 0.152"
        d_values = real_doses["d_values"]  # type: ignore[index]
        noise_band = max(abs(Decimal(str(v))) for v in d_values.values())
        seed2 = Decimal(str(per_seed["2"]["d_i"]))
        check(
            "fixture: 'sits inside |d_i| up to 0.152' relation is FALSE (0.179 > 0.152)",
            not (seed2 <= noise_band),
        )
        check("fixture: corrected 'sits OUTSIDE' relation (0.179 > 0.152) is TRUE", seed2 > noise_band)

        # --- synthetic REDs, >=2 per rule -------------------------------------
        # rule 1, synthetic #1: nonexistent key
        r1a = True
        try:
            rule_h._rfc6901_walk(artifact, "/mutant_dose_ladder/doses/0/nonexistent_key")
        except (rule_h.ResolutionError, IndexError, KeyError, TypeError):
            r1a = False
        check("rule1 synthetic#1: nonexistent key FAILS to resolve", not r1a)
        # rule 1, synthetic #2: index off-by-one (doses[1] when only doses[0] exists)
        r1b = True
        try:
            rule_h._rfc6901_walk(artifact, "/mutant_dose_ladder/doses/1/per_seed")
        except (rule_h.ResolutionError, IndexError, KeyError, TypeError):
            r1b = False
        check("rule1 synthetic#2: doses[1] index off-by-one FAILS to resolve", not r1b)

        # rule 2, synthetic #1: 5th-sig-fig drift mean_d=+15.96715 vs true 15.96714...
        true_mean_d = Decimal("15.967140335279206")
        check(
            "rule2 synthetic#1: mean_d=15.96715 (5th-digit drift) is FALSE",
            not compare_token("15.96715", true_mean_d)[0],
        )
        # rule 2, synthetic #2: transposed cross_seed_spread 0.08256 vs true 0.08265
        true_spread = Decimal("0.08264997071681932")
        check(
            "rule2 synthetic#2: transposed cross_seed_spread=0.08256 is FALSE",
            not compare_token("0.08256", true_spread)[0],
        )
        # rule 2, synthetic #3: dropped-sign +0.0183 vs true -0.01830683834850788
        check(
            "rule2 synthetic#3: dropped-sign mean_d=0.0183 (vs true -0.0183) is FALSE",
            not compare_token("0.0183", Decimal("-0.01830683834850788"))[0],
        )
        # rule 2, synthetic #4: denominator swap n_pos=5/11 (true gate_seed_count=12)
        # -- driven end-to-end through Loader.resolve (a pre-seeded cache
        # entry, never a bare dict compare) + the REAL compare_token, not a
        # tautological `Decimal("11") == Decimal("12")` (round-20 audit
        # advisory: the prior form here would pass unconditionally, proving
        # nothing about the module's own resolution/equality code).
        fixture_loader = Loader()
        fixture_loader._cache["fixture-denom.json"] = {"decision": {"gate_seed_count": 12}}
        denom_binding = Evaluator(fixture_loader, {}, None).eval(parse_expr("fixture-denom.json#/decision/gate_seed_count"))
        check(
            "rule2 synthetic#4: denominator swap 5/11 (true gate_seed_count=12) is FALSE",
            not compare_token("11", denom_binding.value)[0],  # type: ignore[arg-type]
        )
        check(
            "rule2 synthetic#4 fixed: 5/12 matches the resolved gate_seed_count",
            compare_token("12", denom_binding.value)[0],  # type: ignore[arg-type]
        )

        # rule 3, synthetic #1 (rel()): false relation 0.08265 > 0.15239 --
        # driven end-to-end through `rel()`'s OWN `Evaluator.eval` (never a
        # bare `Decimal(...) > Decimal(...)` — round-20 audit advisory: the
        # prior form never called `rel()` at all, so a bug INSIDE `rel()`
        # itself could not have failed this check).
        fixture_loader._cache["fixture-rel.json"] = {"spread": 0.08265, "noise_band": 0.15239}
        rel_ev = Evaluator(fixture_loader, {}, "fixture-rel.json")
        rel_false_raised = False
        try:
            rel_ev.eval(parse_expr("rel(#/spread, '>', #/noise_band)"))
        except ResolutionError:
            rel_false_raised = True
        check("rule3 synthetic#1 (rel()): 0.08265 > 0.15239 correctly raises ResolutionError", rel_false_raised)
        rel_true_ok = True
        try:
            rel_ev.eval(parse_expr("rel(#/noise_band, '>', #/spread)"))
        except ResolutionError:
            rel_true_ok = False
        check("rule3 synthetic#1 fixed (rel()): 0.15239 > 0.08265 resolves cleanly", rel_true_ok)

        # rule 3, synthetic #2 (interval()): false containment (11.1954
        # outside [11.2, 20.1]) -- driven end-to-end through `interval()`'s
        # OWN `Evaluator.eval`, same discipline as the `rel()` fixture above.
        fixture_loader._cache["fixture-interval.json"] = {"d_i": 11.1954}
        interval_ev = Evaluator(fixture_loader, {}, "fixture-interval.json")
        interval_false_raised = False
        try:
            interval_ev.eval(parse_expr("interval(#/d_i, 11.2, 20.1)"))
        except ResolutionError:
            interval_false_raised = True
        check(
            "rule3 synthetic#2 (interval()): 11.1954 correctly NOT contained in [11.2, 20.1]",
            interval_false_raised,
        )
        interval_true_ok = True
        try:
            interval_ev.eval(parse_expr("interval(#/d_i, 11.19, 20.1)"))
        except ResolutionError:
            interval_true_ok = False
        check(
            "rule3 synthetic#2 fixed (interval()): 11.1954 IS contained in [11.19, 20.1]",
            interval_true_ok,
        )

        # within(): unused by the real corpus today, but part of the closed
        # grammar (module doc) -- driven end-to-end through its OWN
        # `Evaluator.eval` so a bug in it cannot ship silently for lack of
        # ANY exercising fixture (round-20 audit advisory).
        fixture_loader._cache["fixture-within.json"] = {"a": 1.0005, "b": 1.0}
        within_ev = Evaluator(fixture_loader, {}, "fixture-within.json")
        within_true_ok = True
        try:
            within_ev.eval(parse_expr("within(#/a, #/b, 0.001)"))
        except ResolutionError:
            within_true_ok = False
        check("within() synthetic: |1.0005 - 1.0| <= 0.001 resolves cleanly", within_true_ok)
        within_false_raised = False
        try:
            within_ev.eval(parse_expr("within(#/a, #/b, 0.0001)"))
        except ResolutionError:
            within_false_raised = True
        check("within() synthetic: |1.0005 - 1.0| > 0.0001 correctly raises ResolutionError", within_false_raised)

        # mind(): unused by the real corpus today (its sibling `maxd()` IS
        # used, e.g. mutants/README.md's own `redproof-signflip-v2` maxd
        # binding), same "no member ships without an exercising fixture"
        # discipline as `within()` above -- driven end-to-end through the
        # REAL tree's own `signflip_v2` vs `alloff` raw legs (the same pair
        # `maxd()` is bound against elsewhere), not a synthetic dict, so the
        # fixture doubles as a real-tree recompute cross-check.
        mind_ev = Evaluator(loader, {}, None)
        mind_binding = mind_ev.eval(
            parse_expr(
                "mind('docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed*.json', "
                "'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', "
                "'held_out_example_mean')"
            )
        )
        maxd_binding = mind_ev.eval(
            parse_expr(
                "maxd('docs/plans/63-how-well/measurements/red-proof/raw/signflip_v2__seed*.json', "
                "'docs/plans/63-how-well/measurements/campaign-v2/raw/seed*__alloff__r1.json', "
                "'held_out_example_mean')"
            )
        )
        check(
            "mind() end-to-end: signflip_v2-vs-alloff smallest signed delta rounds to 0.17900",
            compare_token("0.17900", mind_binding.value)[0],  # type: ignore[arg-type]
        )
        check(
            "mind() correctly the SMALLER endpoint, maxd() the LARGER, over the SAME pair",
            mind_binding.value < maxd_binding.value,  # type: ignore[operator]
        )

        # wrong-file binding: dstar's n_pos=12 must NOT validate against
        # red-proof's own (pre-D*) report — `mutant_dose_ladder.red_proof[0]`
        # there (the `redproof-nobc` column) reads n_pos=3.
        red_proof_doc = loader.load(
            "docs/plans/63-how-well/measurements/red-proof/finetune_run_ab_report.json"
        )
        wrong_file_n_pos_0 = red_proof_doc["mutant_dose_ladder"]["red_proof"][0]["n_pos"]  # type: ignore[index]
        check(
            "wrong-file-binding fixture: dstar n_pos=12 != red-proof's own red_proof[0].n_pos "
            f"({wrong_file_n_pos_0})",
            Decimal(str(wrong_file_n_pos_0)) == Decimal("3") and Decimal(str(wrong_file_n_pos_0)) != Decimal("12"),
        )
        # ...and red_proof[1] there (the co-scheduled `redproof-signflip-v2`
        # column, pre-D*) is INVALID with n_pos=0 — a second independent
        # wrong-file mismatch in the SAME artifact.
        wrong_file_n_pos_1 = red_proof_doc["mutant_dose_ladder"]["red_proof"][1]["n_pos"]  # type: ignore[index]
        check(
            "wrong-file-binding fixture #2: dstar n_pos=12 != red-proof's own red_proof[1].n_pos "
            f"({wrong_file_n_pos_1})",
            Decimal(str(wrong_file_n_pos_1)) != Decimal("12"),
        )

    # --- a deleted tag REDs the coverage leg (mutation-adequacy: the
    # coverage leg itself must be able to fire) --------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        content = "prose line with a bare 3 and 12 and no tag above\n"
        p = _write_tmp(tmp, "x.md", content)
        toks = find_claim_tokens(content.rstrip("\n"), 1)
        default_path, entries = _entries_for_line([content.rstrip("\n")], 1)
        check(
            "coverage-leg fixture: an untagged claim line has no entries -> would FIND",
            len(toks) >= 1 and entries is None,
        )

    # --- new (round-21) closed lexical exclusion classes: each excludes
    # ONLY its own shape -- a measured token dressed up in the SAME
    # syntactic neighborhood (adversarial fixture per class) must still be
    # found. -----------------------------------------------------------
    def toks_of(line: str) -> list[str]:
        return [t.text for t in find_claim_tokens(line, 1)]

    check("exclusion(Step N): 'Step 2 predicted the effect' excludes cleanly", toks_of("Step 2 predicted the effect") == [])
    check(
        "exclusion(Step N) adversarial: a measured value beside 'Step 2' is still found",
        toks_of("the measured value was 2.5 in Step 2") == ["2.5"],
    )
    check("exclusion(CONTRACT/unit/delta N): all three excluded", toks_of("CONTRACT 63 unit 63 delta 7") == [])
    check(
        "exclusion(CONTRACT N) adversarial: a measured value beside 'CONTRACT 63' is still found",
        toks_of("measured 63.5 in CONTRACT 63") == ["63.5"],
    )
    check("exclusion(H\\d+(\\d+)): \"per H5(1)'s own rule\" excludes cleanly", toks_of("per H5(1)'s own rule") == [])
    check(
        "exclusion(H\\d+(\\d+)) adversarial: a measured value beside 'H5(1)' is still found",
        toks_of("measured 5.5 per H5(1)") == ["5.5"],
    )
    check(
        "exclusion(*.patch filename): a dose value embedded in a patch filename excludes cleanly",
        toks_of("see `M_eps_-0.10.patch` for the patch") == [],
    )
    check(
        "exclusion(*.patch filename) adversarial: a standalone measured value beside the SAME filename is still found",
        toks_of("measured -0.10 as recorded in `M_eps_-0.10.patch`") == ["-0.10"],
    )
    check("exclusion(s-scale): 's: 0 -> 1' and 's=1' exclude cleanly", toks_of("the operating point `s: 0 -> 1`, `s=1` was reached") == [])
    check(
        "exclusion(s-scale) adversarial: a measured value beside 's=1' is still found",
        toks_of("measured 1.5 at `s=1`") == ["1.5"],
    )
    check("exclusion(eps sign condition): 'eps < 0' and 'eps > 0' exclude cleanly", toks_of("for eps < 0 dose, unlike eps > 0") == [])
    check(
        "exclusion(eps sign condition) adversarial: a measured value beside 'eps < 0' is still found",
        toks_of("measured 0.5 when eps < 0") == ["0.5"],
    )
    check("exclusion(exit N): 'cargo build (exit 0)' excludes cleanly", toks_of("cargo build (exit 0)") == [])
    check(
        "exclusion(exit N) adversarial: a measured value beside 'exit 0' is still found",
        toks_of("measured 0.5 at (exit 0)") == ["0.5"],
    )
    check("exclusion(seed{N,M}): 'seed{1,2} committed' excludes cleanly", toks_of("seed{1,2} committed") == [])
    check(
        "exclusion(seed{N,M}) adversarial: a measured value beside 'seed{1,2}' is still found",
        toks_of("measured 1.5 for seed{1,2}") == ["1.5"],
    )
    check(
        "exclusion((1-eps) formula fragment): both '(1+eps)' and '(1-eps)' exclude cleanly",
        toks_of("the `(1+eps)`/`(1-eps)` dose-family shape") == [],
    )

    check(
        "self-test: real tree tokenizer matches pinned FILE_TOKEN_DENOMINATOR",
        len(check_coverage()) == 0,
    )

    # ledger cross-check: mutation-adequacy — a `ledger` tag whose key is
    # NOT registered in the committed allowlist must FAIL, never silently
    # pass (the escape must be human-reviewed-and-committed, not merely
    # self-declared inline) — driven end-to-end through the REAL `scan_file`
    # entry point (round-20 audit advisory: the prior GREEN form here was
    # `real_key in (load_ledger() | {real_key})`, which is true by set-union
    # construction REGARDLESS of `scan_file`'s own ledger-membership logic —
    # a tautology that could never fail). `Path(REPO_ROOT) / <absolute path>`
    # returns the absolute path unchanged (stdlib `pathlib` semantics), so a
    # tempfile path works as `scan_file`'s `rel_path` without touching the
    # real tree; under the inverted (allowlist-free) coverage model,
    # `scan_file` derives its scope purely from the fixture file's OWN
    # lines (`_in_scope_lines`) — no separate zone registration for a
    # throwaway fixture path is needed any more.
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _td:
        fixture_md = Path(_td) / "fixture-ledger.md"
        fixture_content = "<!-- claims63: c1=ledger -->\nunregistered 42 claim\n"
        fixture_md.write_text(fixture_content)
        fixture_rel = str(fixture_md)  # absolute -- see comment above
        fixture_line = fixture_content.splitlines()[1]
        fixture_tok = find_claim_tokens(fixture_line, 2)[0]
        real_key = f"{fixture_rel}:{fixture_tok.text}:{line_hash(fixture_line)}:{fixture_tok.col}"
        findings_absent, _, _, _ = scan_file(fixture_rel, Loader(), ledger=set())
        check(
            "ledger synthetic RED: scan_file FINDS an unregistered ledger key (driven end-to-end)",
            any("ledger-escaped" in f.message for f in findings_absent),
        )
        findings_present, _, bound_present, _ = scan_file(fixture_rel, Loader(), ledger={real_key})
        check(
            "ledger synthetic GREEN: scan_file BINDS the SAME key once registered (driven end-to-end)",
            not findings_present and bound_present == 1,
        )

    # --- round-21 perturbation probes, reproduced as regression fixtures --
    # (INVERTED-MODEL mutation-adequacy: the auditor perturbed four sites in
    # the real, committed `mutants/README.md` and found :1111 correctly
    # REDded while :929/:153/:551 sat OUTSIDE every CLAIM_ZONES range and
    # were silently never scanned at all. Under the inverted, allowlist-free
    # model every one of these MUST RED, driven end-to-end through the REAL
    # `scan_file` entry point on a MUTATED COPY of the real committed file —
    # never a synthetic 2-line fixture, so a regression in the inversion
    # itself (a zone/scope narrowing creeping back in) cannot hide behind a
    # too-small fixture the way the pre-inversion CLAIM_ZONES bug did.)
    real_mutants_path = REPO_ROOT / "docs/plans/63-how-well/mutants/README.md"
    real_mutants_text = real_mutants_path.read_text()
    real_ledger = load_ledger()

    def _mutate_and_scan(old: str, new: str) -> list[Finding]:
        assert old in real_mutants_text, f"probe anchor text not found: {old!r}"
        mutated = real_mutants_text.replace(old, new, 1)
        with _tempfile.TemporaryDirectory() as td:
            p = Path(td) / "README.md"
            p.write_text(mutated)
            findings, _, _, _ = scan_file(str(p), Loader(), ledger=real_ledger)
            return findings

    # probe 1 (round-21 :929 analog): M_signflip v1's own "RETIRED, measured
    # INERT on GPU (12/12 bit-identical)" perturbed to a near-miss 11/12 --
    # was silently OUTSIDE every CLAIM_ZONES range before the inversion.
    f929 = _mutate_and_scan(
        "RETIRED, measured INERT on GPU (12/12 bit-identical)",
        "RETIRED, measured INERT on GPU (11/12 bit-identical)",
    )
    check("round-21 probe :929 (M_signflip v1 12/12->11/12) REDs under the inverted model", bool(f929))

    # probe 2 (round-21 :153 analog): Step 1's own committed-benchmark table,
    # one full-precision `held_out_example_mean` value perturbed in its last
    # digit -- was silently OUTSIDE every CLAIM_ZONES range before the
    # inversion (the "committed-benchmark table" round-21 itself named).
    f153 = _mutate_and_scan("`3.218041628599167`", "`3.218041628599168`")
    check("round-21 probe :153 (Step-1 table full-precision value, last digit) REDs under the inverted model", bool(f153))

    # probe 3 (round-21 :551 analog): M1's own "8/12, well under either
    # threshold" restatement perturbed to a near-miss 9/12 -- was silently
    # OUTSIDE every CLAIM_ZONES range before the inversion.
    f551 = _mutate_and_scan(
        "sign-flipping-transient shape: 8/12, well under either threshold);",
        "sign-flipping-transient shape: 9/12, well under either threshold);",
    )
    check("round-21 probe :551 (M1 8/12->9/12 restatement) REDs under the inverted model", bool(f551))

    # probe 4 (round-21 "EOF-append"): a brand-new, untagged measured-shaped
    # claim appended past the end of the file -- the exact shape a
    # covered-region ALLOWLIST is structurally blind to (nothing further
    # than the last zone was ever scanned); under the inverted (default-in-
    # scope) model this REDs exactly like any other untagged line.
    f_eof = _mutate_and_scan(
        real_mutants_text[-40:],
        real_mutants_text[-40:] + "\nA freshly appended, untagged measured claim: 42.42.\n",
    )
    check("round-21 probe (EOF-append, untagged) REDs under the inverted model", bool(f_eof))

    # negative control (round-21 :1111 analog): the "Files" section's own
    # `M_signflip_v2.patch` restatement of "measured RED, 12/12" was ALREADY
    # inside a CLAIM_ZONES range pre-inversion and already REDded correctly
    # -- perturbing it must STILL RED post-inversion (a continuity check:
    # the inversion must never silently narrow what a prior, working zone
    # already covered).
    f1111 = _mutate_and_scan(
        "member of this pair, **measured RED, 12/12, `red_proof_verdict=PROVEN`",
        "member of this pair, **measured RED, 11/12, `red_proof_verdict=PROVEN`",
    )
    check("round-21 probe :1111 analog (Files-section 12/12 restatement) STILL REDs post-inversion", bool(f1111))

    findings, in_zone, bound, breakdown = run_real_tree()
    check(
        f"self-test: real tree run is clean ({in_zone} tokens, {bound} bound "
        f"[ptr={breakdown['ptr']} recipe={breakdown['recipe']} ledger={breakdown['ledger']} "
        f"const={breakdown['const']} hist={breakdown['hist']}], {len(findings)} findings)",
        not findings,
    )

    print(f"\n{len(failures)} failing check(s)" if failures else "\nall checks GREEN")
    return 1 if failures else 0


def _under_repo(p: Path) -> bool:
    try:
        p.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-allowlist-only-shrinks", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.check_allowlist_only_shrinks:
        return check_allowlist_only_shrinks()
    return gate()


if __name__ == "__main__":
    sys.exit(main())
