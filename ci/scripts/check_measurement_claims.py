#!/usr/bin/env python3
r"""esc-065's write-time oracle: every artifact-referencing measurement claim
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

ANTI-VACUITY (FOUR legs — round-22 audit added (c)/(d) to rule (h)'s
original two-leg discipline, `EXPECTED_DENOMINATOR`/`--sweep`, generalized):
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
      site (`_EXTRA_EXCLUSIONS` below), with a one-line reason, and has
      EXACTLY one row in the `_EXCLUSION_FIXTURES` registry immediately
      below `find_claim_tokens` — a positive "excludes cleanly" case AND
      an adversarial "a measured value in the SAME syntactic neighborhood
      is still found" case, both driven through `find_claim_tokens`.
      `len(_EXCLUSION_FIXTURES) == len(_EXTRA_EXCLUSIONS)` is an `assert`
      that runs at IMPORT time (round-25: not deferred to `--self-test`,
      and not a recount by hand each round) — so "every one of the 31
      classes has a fixture proving it excludes ONLY its own shape" is
      true BY CONSTRUCTION: a class with a missing, empty, or misindexed
      row fails to even import, and `self_test` additionally runs a BITE
      sweep (`_worst_case_span_extension`) that monkeypatches EVERY
      class's own pattern into its worst-case span-extending widening and
      confirms the SAME adversarial fixture goes RED — proving each row
      is load-bearing against a future widening, not merely a decorative
      pass on the pattern as currently written (round-25's own live
      finding: two classes, `unit 63` and `PLAN v2 delta N`, had an
      anchoring/"different-N" fixture but no adjacency fixture at all, so
      a widening of either was FULLY SILENT — closed by registration, not
      by patching those two instances alone). There is no per-file/
      per-line carve-out that could silently narrow past that documented
      boundary the way a zone allowlist could. A SUBSET of
      classes are additionally ANCHORED to the specific literal idiom/
      digit they name (round-22 audit B4, round-23 F1:
      `Step`/`Acceptance`/`finding`/`unit 63`/`CONTRACT 63`/
      `PLAN v2 delta`/`exit 0`/the `s`-axis `{0,1}` domain/the CPU-demo's
      `4-element`/`5 consecutive` shape) — never a bare `\bWORD\s+\d+\b`
      wide enough to also swallow an adjacent real measured claim sharing
      the same word (e.g. "Step" as a doc-internal section cross-reference
      vs "step" as a training-step count; round-23 F1: "N-leg" as a
      doc-internal CPU-demo design constant vs "N-leg" as this corpus's
      OWN gate-seed-count measurement — the exact shape that escaped
      round-22's own sweep). The remaining classes — the `epoch|step|t`
      axis-index label, `item N`, and the seed-number LIST — are
      DELIBERATELY left unbounded in N: each is a citation/axis-position
      SHAPE (not itself a measured outcome) that this corpus uses at
      genuinely many different, real N (`epoch 1`, `step=1`..`step=7`,
      `item 2`/`item 3`/`item 4`, `seed 1`/`seed 2`/`seed 4`/
      `seeds 9 and 12`); bounding any of these to one literal value would
      only forbid a future genuine occurrence at a different N, never
      close a real collision the way the round-22 B4 fixes above did.
  (b) AMBIGUITY. For every DIRECT pointer binding (not `hist`/`ledger`/a
      derived aggregate — round-22 B3: an `abs(E)`-wrapped scalar pointer
      is UNWRAPPED to a bare pointer wherever the doc token is not itself
      printed in `|...|` magnitude notation, precisely so this leg (and
      sign sensitivity) apply to it), an INDEPENDENT scan
      (`_check_ambiguity`, walking `_iter_leaf_paths` — a traversal never
      used by the production `Evaluator.eval` resolution path) counts how
      many OTHER leaves round to the identical value at the token's own
      stated precision; more than one DISTINCT pointer (under a DIFFERENT
      field name — a same-named mirrored field, e.g. `decision.p_value` vs
      `sign_test.p_value`, is an intentional duplicate, not a coincidence)
      bound to the same rounded value inside one artifact is a FINDING
      ("ambiguous binding"). Three classes are exempted: an INTEGER-valued
      token (`n_pos`/`n_neg`/dispatch-count-shaped fields collide
      constantly and harmlessly over a small state space — flagging every
      collision would be pure noise), an EXACT-ZERO token ("many other
      leaves are also exactly 0.0" carries no signal), and a CLOSED
      MIRROR-EXEMPTION list (`_MIRROR_EXEMPT_FIELD_PAIRS`, round-22 B3) of
      field-NAME pairs that are a KNOWN, legitimate same-artifact
      duplicate (today: `held_out_example_mean`/`held_out_mean`, the
      untrained checkpoint logged both as the summary field and under
      every `trajectory[i]`) — grown only via a human-reviewed PR, never a
      wildcard, and self-test-proven to exempt ONLY its own listed pairs.
  (c) FENCE INTEGRITY (round-22 audit B1). The fence carve-out mechanism
      (a) relies on has its OWN fail-closed check, `check_fence_integrity`:
      an odd ``` delimiter count (an unclosed fence) is a FINDING by
      itself — the prior version had no balance check at all, so an
      unclosed fence silently carved out everything through EOF, never
      even a candidate for a finding — and, independently, the total
      fenced-line count is pinned per file (`FENCE_LINE_DENOMINATOR`,
      parallel to `FILE_TOKEN_DENOMINATOR`), so a line appended INSIDE an
      already-closed fence — which moves no token count, since fenced
      content is never tokenized — still drifts a pinned number.
  (d) UNUSED ENTRIES (round-22 audit A2, round-23 audit F2). TWO
      complementary checks, both catching "an entry that claims to bind a
      token but is consumed by NOTHING":
        - the ALLOWLIST FILE side: `check_unused_ledger_entries` diffs
          every `ledger`/`const` key an actual scan of the real tree
          consumed against the full committed allowlist — an entry
          consumed by NOTHING (e.g. left behind after a lexical exclusion
          class widened to cover what it used to escape) is a FINDING; a
          shrink-only ratchet with no liveness check on its own rows can
          otherwise accumulate dead escapes forever.
        - the IN-DOC `claims63:` TAG side (round-23 F2): `scan_file`
          itself now checks `len(entries)` against `len(toks)` in BOTH
          directions, not just `entries < toks` (a token with no entry).
          `entries > toks` — a tag entry bound to NOTHING, including the
          degenerate case where the line below has ZERO claim tokens at
          all (every numeral on it excluded by a lexical class) — is
          symmetrically a FINDING; the prior one-sided form's `if not
          toks: continue` made this second shape structurally invisible,
          not merely under-counted (three committed instances: a lexical
          exclusion class widening out from under a `ledger` tag, twice,
          and once resolving naturally once F1's own exclusion-class fix
          restored the token it was already correctly written to bind).

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
    "docs/plans/63-how-well/measurements/f16-sweep/README.md",
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
#       `_EXTRA_EXCLUSIONS` below, plus the `claims63:`-tag-line skip in
#       `find_claim_tokens` — a directive line, never claim-bearing content)
#       — every class is named with a one-line reason at its definition
#       site, and every class has exactly one row in the
#       `_EXCLUSION_FIXTURES` registry (next to `find_claim_tokens`)
#       proving it excludes ONLY its own shape — a STRUCTURAL guarantee
#       (round-25: `assert len(_EXCLUSION_FIXTURES) ==
#       len(_EXTRA_EXCLUSIONS)` runs at import time, plus a BITE sweep in
#       `self_test` that widens every class's own pattern into its
#       worst-case span-extension and confirms its fixture goes RED),
#       superseding round-23 F3's hand-maintained one-`check()`-per-class
#       relay (round-24's own commit message named this shape "F10"; two
#       classes, `unit 63` and `PLAN v2 delta N`, still had no adjacency
#       fixture under that relay — closed by the registry, not by a third
#       instance patch). A1 (round-22 audit): a heading line
#       (`^#`) is NOT its own
#       mechanism — there used to be a whole-line heading skip here, but it
#       swallowed a live threshold flip AND an injected false claim written
#       directly into a heading (nothing under a heading was ever even a
#       candidate for a finding); heading lines are in scope exactly like
#       any other line, escaping only through (a)/(b)/(c) same as prose.
#   (c) the `ledger`/`const` ratchets (ci/measurement_claims_allowlist.txt,
#       shrink-only) — for a token that IS in scope but this grammar's own
#       recipe/pointer machinery genuinely cannot reach.
# A measured claim appended or edited ANYWHERE in an in-scope file is a
# claim token under this default; it REDs unless one of the three
# mechanisms above already covers it — there is no fourth way to leave
# scope, and in particular no per-file/per-line allowlist to silently miss
# updating.


def _fence_scan(lines: list[str]) -> tuple[set[int], int]:
    """Single pass over `lines` producing BOTH (1) every 1-indexed line
    number that is a ``` fence delimiter OR strictly between a matched pair
    — mechanism (a) — and (2) the raw COUNT of delimiter lines seen (B1,
    round-22 audit: the prior version had no balance check at all, so an
    unclosed fence silently toggled `in_block` True for the rest of the
    file and fenced everything through EOF — the "laundering probe": a
    false `Measured` line appended behind a lone, unclosed ``` was
    silently carved out of scanning entirely, never even a candidate for a
    finding). The delimiter count's PARITY is the caller's fail-closed
    check (`check_fence_integrity` below) — an odd count means the file's
    OWN fence nesting is broken and every line from the last delimiter to
    EOF was carved out on a false premise, which must itself be a finding,
    never a silent widen of what "the whole file" scans."""
    fenced: set[int] = set()
    in_block = False
    delimiter_count = 0
    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            fenced.add(i)  # the delimiter itself is not claim-bearing text
            delimiter_count += 1
            in_block = not in_block
            continue
        if in_block:
            fenced.add(i)
    return fenced, delimiter_count


def _fenced_line_numbers(lines: list[str]) -> set[int]:
    """Every 1-indexed line number that is a ``` fence delimiter OR strictly
    between a matched pair — mechanism (a) above. Reuses the exact
    delimiter-toggle rule `parse_code_blocks` already uses for its own,
    separate traversal (fenced content is read as the `code()` artifact
    there; it is EXCLUDED from claim-token scanning here — two different
    passes over identical fence boundaries, kept in sync by construction
    since both key off the same `stripped.startswith("```")` toggle).
    Thin wrapper over `_fence_scan` — see it for the PARITY check this
    function itself does not (and structurally cannot, having thrown away
    the delimiter count) perform; every caller that needs fail-closed
    behavior on an unbalanced file must go through `check_fence_integrity`,
    never this function alone."""
    return _fence_scan(lines)[0]


# Pinned per-file TOTAL in-scope token counts (mechanism (a)/(b) applied,
# whole file — no zone allowlist), using THIS module's own
# tokenizer/exclusions. This is what now catches silent scope shrink: since
# every line is a candidate, a token entering OR leaving scope (an edit, an
# exclusion-class change, a fence added/removed) changes this count and
# fails the bare run, naming both numbers — exactly
# `check_perf_claims.py`'s own `EXPECTED_DENOMINATOR` discipline, now
# applied to the WHOLE file instead of a hand-picked subset of it.
FILE_TOKEN_DENOMINATOR: dict[str, int] = {
    # +1 (round-22 A1): heading lines are now in scope — campaign-v1's
    # "## probe/ — H5 step-0 (4 legs, a100)" heading's own "4" token.
    "docs/plans/63-how-well/measurements/campaign-v1/README.md": 40,
    "docs/plans/63-how-well/measurements/campaign-v2/README.md": 17,
    "docs/plans/63-how-well/measurements/dose-ladder/README.md": 22,
    # The f16 sweep, registered in this list rather than left to its own
    # "this README is not registered" notice — a measurement record that
    # documents its own exemption is still an unchecked measurement record.
    # Its `raw/*.json` legs (not just the merged report) are producers here:
    # the run's stated configuration (epochs/batch/lr/patience) binds to the
    # fields the legs actually recorded, and the per-seed d-column binds both
    # to `#/per_seed/{N}/d_i` and, for the sign summary, to the raw-
    # concordance recipe recomputed from the legs themselves.
    "docs/plans/63-how-well/measurements/f16-sweep/README.md": 53,
    "docs/plans/63-how-well/measurements/red-proof/README.md": 46,
    "docs/plans/63-how-well/measurements/red-proof/dstar/README.md": 31,
    # +2 (round-22 A1): heading lines are now in scope — the "### Step 3 —
    # ... >=11/12+mean rule" heading's own "11"/"12" tokens.
    # +7 (round-23 F1): the "N-leg" exclusion class was removed (a real,
    # bindable sample-size measurement, same discipline as "N-seed" — see
    # `_EXTRA_EXCLUSIONS` above); the "12" in "12-leg GPU" at all 7
    # committed Measured-record sites now tokenizes and is bound
    # per-occurrence (322 round-22 baseline + 7 round-23 F1 "12-leg"
    # tokens = 329).
    "docs/plans/63-how-well/mutants/README.md": 329,
}

# B1 (round-22 audit): a SECOND, INDEPENDENT pin — the total count of fenced
# lines (delimiters + interior) per file, exactly `check_perf_claims.py`'s
# own `EXPECTED_DENOMINATOR` discipline applied to the region the token
# denominator above CANNOT see. Appending a line INSIDE an already-closed
# fence changes NOTHING in `FILE_TOKEN_DENOMINATOR` (fenced content is never
# tokenized) — it must instead drift THIS pin, or the append is invisible to
# both anti-vacuity legs at once. A file with no fenced content at all pins
# to 0 (still checked: a 0 -> nonzero drift, e.g. a first fence appearing in
# a file that previously had none, is exactly the kind of scope-shrink this
# leg exists to catch).
#
# DISCLOSED LIMIT (round-23 audit A2, count-only compensating edit): this
# pin is a single scalar TOTAL, not a per-fence-block or per-line-position
# record — it catches any net change in the total fenced-line count, but a
# COMPENSATING edit (one fenced line deleted from one block, a DIFFERENT
# fenced line added to a DIFFERENT block, net delta zero) leaves the total
# unchanged and is NOT caught by this leg. This is a known, accepted
# narrowing of what "the pin" verifies (a total, not a shape), not a bug —
# closing it would need a per-block or line-content pin, which this leg
# does not attempt; `self_test`'s "A2 known-negative" fixture asserts this
# CURRENT behavior explicitly, as the disclosed limit it is, never as a
# fake RED.
FENCE_LINE_DENOMINATOR: dict[str, int] = {
    "docs/plans/63-how-well/measurements/campaign-v1/README.md": 0,
    "docs/plans/63-how-well/measurements/campaign-v2/README.md": 0,
    "docs/plans/63-how-well/measurements/dose-ladder/README.md": 0,
    "docs/plans/63-how-well/measurements/f16-sweep/README.md": 0,
    "docs/plans/63-how-well/measurements/red-proof/README.md": 0,
    "docs/plans/63-how-well/measurements/red-proof/dstar/README.md": 0,
    "docs/plans/63-how-well/mutants/README.md": 91,
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
    # B4 (round-22 audit): "Acceptance N" / "finding N" / "Step N" / "Step
    # N-M" / "Step N/M" cross-references — each ANCHORED to its own actual
    # idiom rather than one shared case-insensitive catch-all. The prior
    # combined form's IGNORECASE made lowercase prose "step" (a training-
    # step COUNT, e.g. "detected at step 12/12 seeds") collide with the
    # doc-internal "Step N" section cross-reference and get swallowed whole
    # — losing the adjacent real measured claim, not just the cross-ref
    # itself (round-22 audit probes: "detected at step 12/12 seeds" and
    # "Step 8/12 seeds" must be FOUND, not excluded, since neither is
    # actually this doc's own Step-1/2/3 cross-reference).
    #   - "Step" is ALWAYS capitalized in this doc's own cross-references,
    #     and always one of this doc's own three numbered "Step" sections
    #     (`### Step 1` / `### Step 2` / `### Step 3`) — bounded 1-3, never
    #     IGNORECASE, so lowercase "step" (a training-step count) is never
    #     swallowed by this class at all.
    re.compile(r"\bStep\s+[1-3](?:[-/][1-3])?\b"),
    #   - "Acceptance" (this doc's single acceptance criterion, "Acceptance
    #     5") appears both capitalized and lowercase — bounded to a single
    #     digit (this doc names only "5"; a wider criterion set is a
    #     human-reviewed widening, never silently re-opened to \d+).
    re.compile(r"\bAcceptance\s+[1-9]\b", re.IGNORECASE),
    #   - "finding" (this doc's own numbered audit-finding cross-references,
    #     "finding 1".."finding 4") is always lowercase here — bounded to a
    #     single digit for the same reason.
    re.compile(r"\bfinding\s+[1-9]\b"),
    # this repo's own `docs/plans/<N>-<slug>` directory-numbering
    # convention (`63-how-well`, `61-perf-unification`) — a path segment,
    # never a measurement.
    re.compile(r"\b\d+-how-well\b"),
    # this unit's own `CONTRACT N` / `unit N` / `unit-N` / `PLAN v2 delta N`
    # self-reference (the doc's own contract/unit/plan-delta number, never
    # a measurement) — same class as the `item N`/`Acceptance N` cross-refs
    # above, widened to this file's own additional numbered-reference
    # vocabulary. round-23 audit F3: bounded to the literal "63" (this
    # doc's own, only, `CONTRACT` number in every in-scope file today) —
    # same discipline as `unit 63` below, so a bare "CONTRACT 12" (a
    # genuinely different number) would surface as a real claim rather
    # than being silently swallowed as though it were this doc's own
    # self-reference.
    re.compile(r"\bCONTRACT\s+63\b"),
    # B4 (round-22 audit): "unit N" — this repo's own unit-numbering
    # cross-reference is ALWAYS this unit's own number, "63" (`unit 63`,
    # `unit-63`, `Unit 63`) in every in-scope file today — bounded to that
    # literal rather than `\d+`, so a bare "unit 12" (a genuinely different
    # number, e.g. a training-unit COUNT) would surface as a real claim
    # instead of being silently swallowed as though it were this doc's own
    # self-reference.
    re.compile(r"\bunit[\s-]+63\b", re.IGNORECASE),
    # B4 (round-22 audit): "delta N" was over-broad enough to swallow ANY
    # "delta N" occurrence, anywhere — anchored to the actual idiom this
    # corpus uses it for, the doc's own "PLAN v2 delta N" self-reference
    # (`CONTRACT 63 (PLAN v2 delta 7; ...)`), never a bare "delta N" in
    # isolation. NOTE (future-inclusion hazard): `CONTRACT.md` itself (not
    # in `MEASUREMENT_FILES` today — see the module doc's SCOPE NOTE) uses
    # a BARE "delta N" idiom per-item (`delta 0`, `delta 3`, ... one per
    # contract item) that is a genuine per-item cross-reference, NOT a
    # measurement, but does NOT share the "PLAN v2" prefix this pattern
    # anchors to; a future unit that brings CONTRACT.md into scope must
    # NOT simply widen this pattern back to `\bdelta\s+\d+\b` (that
    # regresses this exact fix) — it needs its own anchored class (or
    # per-item `const` bindings) for CONTRACT.md's bare "delta N" idiom.
    re.compile(r"\bPLAN\s+v2\s+delta\s+\d+\b"),
    # an `H<n>(<m>)` requirement/heading cross-reference (`H5(1)`).
    re.compile(r"\bH\d+\(\d+\)"),
    # a `*.patch` filename (`M_eps_-0.10.patch`, `M_nobc.patch`) — the
    # patch's OWN identifier, never a measurement; the leading `M_` +
    # trailing `.patch` bracket the whole filename so an embedded dose
    # value (`M_eps_-0.10.patch`) is excluded as part of the name, not
    # read as a bare decimal.
    re.compile(r"\bM_[A-Za-z0-9_.+-]*\.patch\b"),
    # B4 (round-22 audit): this section's own `s: 0 -> 1` / `s=0` / `s=1`
    # interpolation-parameter axis notation (the untrained/trained
    # operating-point labels for the secant slope) — bounded to the axis's
    # own literal domain, `{0, 1}` (every real occurrence names one or both
    # endpoints; the variable is DEFINED as running over `[0, 1]`), rather
    # than an unbounded `\d+`, so an adjacent real measured claim written
    # as `s=<value>` for any OTHER value is never silently swallowed.
    re.compile(r"\bs\s*[:=]\s*[01](?:\s*->\s*[01])?\b"),
    # round-23 audit F1: the SAME `{0, 1}` interpolation-parameter axis
    # domain, restated in bracket-range prose instead of `s:`/`s=` notation
    # (`the secant is measured over the entire [0, 1] range`) — the exact
    # literal endpoints only, never a bare `\[\d+,\s*\d+\]`, so an unrelated
    # bracketed pair naming two OTHER numbers is never silently swallowed.
    re.compile(r"\[0,\s*1\]"),
    # an `eps < 0` / `eps > 0` sign-domain condition (which branch of the
    # signed dose family a column belongs to) — a formula/domain
    # description, same class as the named-hyperparameter-assignment
    # exclusion above, not a measurement of any particular eps value.
    re.compile(r"\beps\s*[<>]\s*0\b"),
    # B4 (round-22 audit): a shell/CLI process exit-code label — every real
    # occurrence in this corpus is "exit 0" (the merger/build ALWAYS exits
    # 0 in the quoted scenarios; a non-zero exit is described in prose
    # without a digit, e.g. "non-zero exit"), so bounded to that literal
    # rather than `\d+` — an adjacent real measured claim written as
    # "exit N" for any other N is never silently swallowed.
    re.compile(r"\bexit\s+0\b"),
    # a `seed{N,M}` brace-set notation — the same "seed list" idiom the
    # `seeds? \d+(?:,|and)...` exclusion above covers for space-separated
    # lists, widened to this corpus's brace-delimited form.
    re.compile(r"\bseeds?\{[\d,]+\}"),
    # a bracketed numeric-literal test FIXTURE (`theta=[0.5,-1.25,3.0,0.0]`,
    # `g=[0.1,-0.2,0.05,0.0]`) — the CPU-demo's fixed input, a design
    # constant never a measurement. round-23 audit F1: ANCHORED to its own
    # `theta=`/`g=` prefix (this corpus's only two bracketed-literal-array
    # occurrences) rather than a bare `\[[-0-9.,\s]+\]`, which also matched
    # an UNPREFIXED bracket pair naming a real value (e.g. an axis-domain
    # `[0, 1]` restatement, now its own anchored class above) — never a
    # measurement dressed up in brackets next to an unrelated word.
    re.compile(r"\b(?:theta|g)=\[[-0-9.,\s]+\]"),
    # round-23 audit F1 (LIVE false-pass): "N-element"/"N consecutive" are a
    # genuinely fixed CPU-demo design constant in this corpus (always the
    # SAME 4-element theta/g input, always 5 consecutive steps) — bounded to
    # those literal digits, exactly like `Acceptance N`/`exit N` above,
    # rather than an unbounded `\d+`. "N-leg" was REMOVED from this class
    # entirely (round-22's own comment already reasoned "N-seed" needed the
    # same per-occurrence-binding treatment for exactly this reason but
    # missed that "N-leg" is the SAME shape: the auditor's own perturbation
    # probe, `12-leg GPU` -> `9-leg GPU` at all 7 committed sites, proved
    # "N-leg" carries a real, bindable sample-size measurement identical to
    # `gate_seed_count`/`clean_pair_count` — it is never blanket-excluded
    # any more; every "N-leg" occurrence in this corpus is bound
    # per-occurrence via its own `claims63` tag entry, same discipline as
    # "N-seed").
    re.compile(r"\b4-element\b"),
    re.compile(r"\b5\s+consecutive\b"),
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


# moved above `find_claim_tokens` (round-25): the exclusion-class self-test
# registry below calls `find_claim_tokens` at IMPORT time (structural
# registration, not deferred to `--self-test`), which needs `_TAG_RE`
# already bound -- same regex, same claims63-tag-line skip, just defined
# earlier so the eager registry check can resolve it.
_TAG_RE = re.compile(r"^\s*<!--\s*claims63:\s*(?P<body>.*?)\s*-->\s*$")


def find_claim_tokens(line: str, line_no: int) -> list[rule_h.Token]:
    """Same shape as `rule_h.find_tokens_in_row`, over `SCI_NUMBER_RE` and
    the UNION of rule (h)'s own exclusions plus this corpus's additions.
    A1 (round-22 audit): heading lines (`^#`) are IN SCOPE, exactly like any
    other line — the prior whole-line `^#` skip swallowed BOTH a live
    threshold flip written directly into a heading AND an injected false
    heading claim (the auditor's own probes), since nothing under a `#`
    line was ever even a candidate for a finding. There is no bespoke
    heading carve-out any more: a numeric token in a heading binds/excludes
    through the exact same tag/exclusion-class machinery as prose (most
    heading numbers already fall under an existing lexical class — a
    self-referencing `unit 63`, a dated amendment, a bounded `Step N`/
    `Acceptance N` cross-reference — the few that do not get an explicit
    `claims63` tag on the line above, same as any prose line). A
    `claims63:` TAG line itself is still never scanned — it is a directive,
    not claim-bearing content, and its own expr text (artifact paths,
    function args) is not a second population of claims needing its own
    tag one line up."""
    if _TAG_RE.match(line):
        return []
    ex = _all_excluded_spans(line)
    out: list[rule_h.Token] = []
    for m in SCI_NUMBER_RE.finditer(line):
        if rule_h._in_any_span(m.span(), ex):
            continue
        out.append(rule_h.Token(m.group(0), line_no, m.start(), line))
    return out


# --- exclusion-class self-test REGISTRY (round-25 audit: class-terminal fix)
# Round-23 F3 gave every `_EXTRA_EXCLUSIONS` class a positive/adversarial
# fixture pair, but as 62 individual hand-written `check()` calls in
# `self_test` — a RELAY shape (round-24's own commit message: "the
# one-instance-at-a-time relay shape (F10)"), never checked for completeness
# against `_EXTRA_EXCLUSIONS` ITSELF. Round-25's live finding: two classes
# (`unit[\s-]+63` and `PLAN v2 delta N`) had an anchoring/"different-N"
# fixture but NO adjacency fixture at all — a measured value written right
# beside the idiom was never even a candidate check, so a monkeypatch sweep
# that widened either pattern was FULLY SILENT (`self_test` stayed green).
# This registry closes that CLASS, not just the two instances: every row of
# `_EXTRA_EXCLUSIONS` has EXACTLY one row here, at the SAME index, and
# `self_test` below ITERATES this table rather than listing one `check()`
# per class by hand. A class with a missing, empty, or misindexed
# positive/adversarial pair fails STRUCTURALLY at import time (the asserts
# immediately below the table), never merely at self-test-review time — so
# the module docstring's ":247" claim ("every one of the 31 classes has a
# fixture") is true BY CONSTRUCTION: `len(_EXCLUSION_FIXTURES) ==
# len(_EXTRA_EXCLUSIONS)` is asserted, not recounted by hand each round.
@dataclass(frozen=True)
class _ExclusionFixture:
    """One row: binds `_EXTRA_EXCLUSIONS[index]` to its own `name` (used in
    `self_test`'s PASS/FAIL print lines, matching the pre-registry wording),
    a `positive` probe line that must exclude CLEANLY (`positive_expected`,
    always `()` for this corpus — every class's whole purpose is to produce
    zero claim tokens on its own idiom), and an `adversarial` probe line — a
    REAL measured value in the SAME syntactic neighborhood as the idiom —
    that must still be FOUND (`adversarial_expected`, always non-empty:
    round-25's own registration assert below enforces this, since an empty
    adversarial_expected would make the row untestable-by-construction, the
    exact silent-pass shape this registry exists to close)."""

    index: int
    name: str
    positive: str
    positive_expected: tuple[str, ...]
    adversarial: str
    adversarial_expected: tuple[str, ...]


_EXCLUSION_FIXTURES: tuple[_ExclusionFixture, ...] = (
    _ExclusionFixture(0, "hex/sha run",
        "checkpoint dc1cfc3b committed", (),
        "the run recorded 42 alongside checkpoint dc1cfc3b", ("42",)),
    _ExclusionFixture(1, "date+revision-letter",
        "recorded on 2026-08-29c after the fix", (),
        "measured 2.5 on 2026-08-29c", ("2.5",)),
    _ExclusionFixture(2, "compact ISO-8601 timestamp",
        "output dir 20260829T055912Z", (),
        "measured 2.5 at 20260829T055912Z", ("2.5",)),
    _ExclusionFixture(3, "GB/MB size",
        "A100-SXM4-80GB", (),
        "measured 2.5 on the 80GB A100", ("2.5",)),
    _ExclusionFixture(4, "epoch/step/t axis index",
        "detected at epoch 1", (),
        "measured 2.5 at epoch 1", ("2.5",)),
    _ExclusionFixture(5, "hyperparameter setting",
        "lr=2e-4, eps=1e-8, alpha2=0.0064, beta1=0.9", (),
        "measured 2.5 at lr=2e-4", ("2.5",)),
    _ExclusionFixture(6, "item N cross-ref",
        "see item 2(ii) above", (),
        "measured 2.5 near item 2(ii)", ("2.5",)),
    _ExclusionFixture(7, "RC:N exit code",
        "RC:0", (),
        "measured 2.5 at RC:0", ("2.5",)),
    _ExclusionFixture(8, "eps-N dose label",
        "eps-0.50 dose", (),
        "measured -0.50 near eps-0.50", ("-0.50",)),
    _ExclusionFixture(9, "ordered-list marker",
        "3. another line", (),
        "2. the measured value is 3.5", ("3.5",)),
    _ExclusionFixture(10, "lr parenthetical mention",
        "the lr (2e-4) setting", (),
        "measured 2.5 near lr (2e-4)", ("2.5",)),
    _ExclusionFixture(11, "(1+eps)/(1-eps) formula fragment",
        "the `(1+eps)`/`(1-eps)` dose-family shape", (),
        "measured 1.5 near (1+eps)", ("1.5",)),
    _ExclusionFixture(12, "Step N",
        "Step 2 predicted the effect", (),
        "the measured value was 2.5 in Step 2", ("2.5",)),
    _ExclusionFixture(13, "Acceptance N",
        "Acceptance 5 discharged", (),
        "measured 5.5 near Acceptance 5", ("5.5",)),
    _ExclusionFixture(14, "finding N",
        "finding 4 resolved", (),
        "measured 4.4 near finding 4", ("4.4",)),
    _ExclusionFixture(15, "63-how-well dir segment",
        "docs/plans/63-how-well/x", (),
        "measured 63.5 in 63-how-well context", ("63.5",)),
    _ExclusionFixture(16, "CONTRACT 63",
        "CONTRACT 63", (),
        "measured 63.5 in CONTRACT 63", ("63.5",)),
    _ExclusionFixture(17, "unit 63",
        "unit 63", (),
        # round-25 audit: this row's adversarial half DID NOT EXIST before
        # this fix -- only a "different unit number" differential (below,
        # `unit 12`) existed, never an ADJACENT real value beside the SAME
        # `unit 63` idiom. A monkeypatch sweep that widened the exclusion
        # span leftward (e.g. a stray `.*` prefix) was therefore fully
        # silent; this line closes exactly that gap.
        "measured 2.5 in unit 63", ("2.5",)),
    _ExclusionFixture(18, "PLAN v2 delta N",
        "PLAN v2 delta 7", (),
        # round-25 audit: same gap as class 17 above -- only the "bare delta
        # N" differential (below) existed, never an adjacent-value probe.
        "shift 0.7 per PLAN v2 delta 7", ("0.7",)),
    _ExclusionFixture(19, r"H\d+(\d+)",
        "per H5(1)'s own rule", (),
        "measured 5.5 per H5(1)", ("5.5",)),
    _ExclusionFixture(20, "*.patch filename",
        "see `M_eps_-0.10.patch` for the patch", (),
        "measured -0.10 as recorded in `M_eps_-0.10.patch`", ("-0.10",)),
    _ExclusionFixture(21, "s-scale",
        "the operating point `s: 0 -> 1`, `s=1` was reached", (),
        "measured 1.5 at `s=1`", ("1.5",)),
    _ExclusionFixture(22, "[0, 1] axis-domain bracket",
        "the axis range is [0, 1] here", (),
        "measured 0.5 alongside the [0, 1] axis range", ("0.5",)),
    _ExclusionFixture(23, "eps sign condition",
        "for eps < 0 dose, unlike eps > 0", (),
        "measured 0.5 when eps < 0", ("0.5",)),
    _ExclusionFixture(24, "exit N",
        "cargo build (exit 0)", (),
        "measured 0.5 at (exit 0)", ("0.5",)),
    _ExclusionFixture(25, "seed{N,M}",
        "seed{1,2} committed", (),
        "measured 1.5 for seed{1,2}", ("1.5",)),
    _ExclusionFixture(26, "theta=/g= bracket array",
        "theta=[0.5,-1.25,3.0,0.0] fixed input", (),
        "measured 0.5 near theta=[0.5,-1.25,3.0,0.0]", ("0.5",)),
    _ExclusionFixture(27, "4-element CPU-demo input",
        "the same fixed 4-element input", (),
        "measured 4.5 near the 4-element input", ("4.5",)),
    _ExclusionFixture(28, "5-consecutive CPU-demo steps",
        "for 5 consecutive steps", (),
        "measured 5.5 over 5 consecutive steps", ("5.5",)),
    _ExclusionFixture(29, "seed-number list",
        "seeds 9 and 12 committed", (),
        "measured 9.5 near seeds 9 and 12", ("9.5",)),
    _ExclusionFixture(30, "zero/negative array-index bracket",
        "series[0] and series[-1]", (),
        "measured 2.5 near series[0]", ("2.5",)),
)

# STRUCTURAL registration checks (round-25: not deferred to `--self-test` --
# these run on EVERY import, including a plain `gate()` invocation, so a
# regressed row is a FINDING before any CI job even reaches `--self-test`).
# (1) the table is index-complete against `_EXTRA_EXCLUSIONS` itself -- the
#     :247 "all 31 classes" universal is this assert, not a comment.
assert len(_EXCLUSION_FIXTURES) == len(_EXTRA_EXCLUSIONS), (
    f"_EXCLUSION_FIXTURES has {len(_EXCLUSION_FIXTURES)} rows but "
    f"_EXTRA_EXCLUSIONS has {len(_EXTRA_EXCLUSIONS)} classes -- every "
    "class must have exactly one registry row (round-25: this assert IS "
    "the :247 universal, not a recount)."
)
assert tuple(fx.index for fx in _EXCLUSION_FIXTURES) == tuple(range(len(_EXTRA_EXCLUSIONS))), (
    "_EXCLUSION_FIXTURES must be index-ordered 1:1 with _EXTRA_EXCLUSIONS "
    "(a reordered or skipped index binds the wrong pattern to the wrong "
    "fixture, silently)."
)
for _fx in _EXCLUSION_FIXTURES:
    # (2) both halves non-empty -- an empty positive/adversarial string, or
    #     an adversarial fixture that asserts NO surviving token, is
    #     untestable by construction (exactly the silent-pass shape this
    #     registry exists to forbid).
    assert _fx.positive.strip(), f"class {_fx.index} ({_fx.name}): empty positive fixture"
    assert _fx.adversarial.strip(), f"class {_fx.index} ({_fx.name}): empty adversarial fixture"
    assert _fx.adversarial_expected, (
        f"class {_fx.index} ({_fx.name}): adversarial fixture must assert "
        ">=1 surviving token, or it proves nothing"
    )
    # (3) both halves EXECUTE with the asserted results, at import time --
    #     never merely typed into the table and trusted.
    _got_pos = tuple(t.text for t in find_claim_tokens(_fx.positive, 1))
    assert _got_pos == _fx.positive_expected, (
        f"class {_fx.index} ({_fx.name}): positive fixture {_fx.positive!r} "
        f"produced {_got_pos}, expected {_fx.positive_expected}"
    )
    _got_adv = tuple(t.text for t in find_claim_tokens(_fx.adversarial, 1))
    assert _got_adv == _fx.adversarial_expected, (
        f"class {_fx.index} ({_fx.name}): adversarial fixture {_fx.adversarial!r} "
        f"produced {_got_adv}, expected {_fx.adversarial_expected}"
    )
del _fx


def _worst_case_span_extension(
    pattern: re.Pattern, adversarial: str, expected: tuple[str, ...]
) -> tuple[str | None, str]:
    """Round-25 BITE sweep helper: construct the WORST-CASE span-extending
    widening of `pattern` -- a `.*` consuming everything between the
    idiom's own match and the line boundary -- on WHICHEVER SIDE of that
    match the adversarial fixture's real value sits (auto-detected by
    string position, never hand-coded per class, so a future 32nd class
    needs no new wiring here to be swept). Returns `(widened_source,
    "leading"|"trailing")`, or `(None, reason)` if that side is
    structurally forbidden (a `^` line-start anchor forbids ANY leftward
    extension -- class 9, the ordered-list marker, is the one class in
    this corpus where this fires; every other class's own adversarial
    value sits on a side its own pattern can, in principle, be widened
    into, which is exactly the vulnerability this sweep exists to close)."""
    src = pattern.pattern
    m = pattern.search(adversarial)
    assert m is not None, "adversarial fixture must itself contain the idiom"
    value_pos = adversarial.find(expected[0])
    assert value_pos != -1, "adversarial fixture must itself contain its own expected value"
    if value_pos < m.start():
        if src.startswith("^"):
            return None, "IMPOSSIBLE: '^' line-start anchor forbids leftward span-extension"
        core = src[2:] if src.startswith(r"\b") else src
        return ".*" + core, "leading"
    core = src[:-2] if src.endswith(r"\b") else src
    return core + ".*", "trailing"


# --- claim tag parsing (rule (h)'s own comment-above-line idiom) ----------
# (`_TAG_RE` itself now lives above, next to `find_claim_tokens` -- see the
# round-25 comment there.)


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
    rel_path: str,
    loader: Loader,
    ledger: set[str] | None = None,
    consumed_ledger_keys: set[str] | None = None,
) -> tuple[list[Finding], int, int, dict[str, int]]:
    """Returns `(findings, tokens_in_zone, tokens_bound, breakdown)` —
    `breakdown` counts successfully-bound tokens per `_BREAKDOWN_KINDS`
    category (a finding is never counted in any category). When
    `consumed_ledger_keys` is passed, every `ledger`/`const` key that
    successfully matches an entry gets added to it — A2 (round-22 audit):
    the unused-entry leg (`check_unused_ledger_entries` below) diffs this
    set against `load_ledger()`'s full contents to find an entry the doc no
    longer cites at all (an orphan, e.g. left behind after a lexical
    exclusion class widened to cover what the entry used to escape).

    round-23 audit F2: the entries-vs-tokens relation was audited
    ONE-SIDED — `len(entries) < len(toks)` (a token with no matching
    entry) was always a finding, but `len(entries) > len(toks)` (a tag
    entry with no matching token — a DEAD directive, e.g. left behind
    after a lexical exclusion class widened to swallow the token it used
    to bind, the exact class `check_unused_ledger_entries` already polices
    for ledger/const ROWS in the allowlist FILE, but never for an ordinary
    tag entry inside the doc itself) was silently absorbed by `zip()`
    truncating to the shorter sequence, never even inspected. This is now
    checked on BOTH sides, including the degenerate `toks == []` case (a
    tag sits above a line with ZERO claim tokens at all — the prior
    control flow `continue`d on an empty `toks` BEFORE ever looking at the
    tag above it, so a tag whose token was fully excluded out from under
    it was invisible by construction, not merely under-counted)."""
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
            # round-23 F2: even with ZERO claim tokens on this line, a
            # `claims63` tag directly above it may still declare entries —
            # every one of those is DEAD (bound to nothing) and must be a
            # finding, never silently skipped just because there is no
            # token to anchor the finding's own column to.
            _, dead_entries = _entries_for_line(lines, line_no)
            if dead_entries:
                findings.append(
                    Finding(
                        rel_path,
                        line_no - 1,
                        0,
                        f"claims63 tag above has {len(dead_entries)} entr"
                        f"{'y' if len(dead_entries) == 1 else 'ies'} but this line has 0 "
                        "claim token(s) -- dead directive",
                    )
                )
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
        elif len(entries) > len(toks):
            findings.append(
                Finding(
                    rel_path,
                    line_no,
                    toks[0].col,
                    f"{len(toks)} claim token(s) but {len(entries)} tag entries "
                    f"-- {len(entries) - len(toks)} dead (unconsumed) entr"
                    f"{'y' if len(entries) - len(toks) == 1 else 'ies'}",
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
                    if consumed_ledger_keys is not None:
                        consumed_ledger_keys.add(key)
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


# B3 (round-22 audit): MIRROR-EXEMPTION — a CLOSED list of field-NAME pairs
# that are a KNOWN, legitimate same-artifact duplicate, never a
# coincidental collision the ambiguity leg exists to catch. The one
# member today: `held_out_example_mean` (the top-level per-leg summary
# field) and `held_out_mean` (logged once per entry under
# `tiers.finetune_run.trajectory[i]`) — the untrained (`lr=0`) checkpoint's
# held-out mean is recorded BOTH as the summary field AND at every
# trajectory step before the model has moved, so they are BIT-IDENTICAL by
# construction, not an accidental match. Growing this list is a
# human-reviewed PR (never a wildcard/prefix rule) — each entry states the
# reason inline; a self-test fixture proves it exempts ONLY this pair (a
# different field-name collision must still report).
_MIRROR_EXEMPT_FIELD_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"held_out_example_mean", "held_out_mean"}),  # untrained checkpoint, logged twice
    }
)


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
        leaf_field = leaf_ptr.rsplit("/", 1)[-1]
        if leaf_field == source_field:
            continue
        # B3 MIRROR-EXEMPTION: a different-but-KNOWN-legit field-name pair
        # (see `_MIRROR_EXEMPT_FIELD_PAIRS` above) — closed, reviewed,
        # never a wildcard.
        if frozenset({source_field, leaf_field}) in _MIRROR_EXEMPT_FIELD_PAIRS:
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


def check_fence_integrity() -> list[str]:
    r"""B1 (round-22 audit): the fence carve-out's own two-part fail-closed
    check. (i) PARITY — an odd `\`\`\`` delimiter count means the file's own
    fence nesting is unbalanced; every line from the last delimiter to EOF
    was silently treated as fenced (carved out of `check_coverage`'s token
    count too) on a false premise — reported HERE, never absorbed into a
    token-count drift message that would misname the defect. (ii) SPAN —
    even when balanced, the total fenced-line count is pinned separately
    from `FILE_TOKEN_DENOMINATOR` (`FENCE_LINE_DENOMINATOR`), so a line
    appended INSIDE an already-closed fence — which changes NO token count,
    since fenced content is never tokenized — still drifts a pinned number
    and is caught. DISCLOSED LIMIT (round-23 audit A2): the SPAN leg pins a
    single scalar TOTAL per file, not a per-block shape, so a COMPENSATING
    edit (a fenced line removed from one block, a different fenced line
    added to another, net delta zero) is NOT caught — see
    `FENCE_LINE_DENOMINATOR`'s own comment for the full disclosure."""
    problems = []
    for rel_path in MEASUREMENT_FILES:
        abspath = REPO_ROOT / rel_path
        lines = abspath.read_text().splitlines()
        fenced, delimiter_count = _fence_scan(lines)
        if delimiter_count % 2 != 0:
            problems.append(
                f"{rel_path}: odd number of ``` fence delimiters "
                f"({delimiter_count}) — an unclosed fence would silently carve out "
                "the remainder of the file from claim-token scanning"
            )
            continue  # the span pin is meaningless over an unbalanced file
        expected = FENCE_LINE_DENOMINATOR.get(rel_path)
        if expected is None:
            problems.append(f"{rel_path}: no FENCE_LINE_DENOMINATOR pinned")
        elif len(fenced) != expected:
            problems.append(
                f"{rel_path}: fenced-line count drifted: pinned {expected}, counted "
                f"{len(fenced)} (a fence was added, removed, or resized without "
                "updating the pin)"
            )
    return problems


def check_unused_ledger_entries() -> list[str]:
    """A2 (round-22 audit): a `ledger`/`const` allowlist entry that is
    consumed by NOTHING in the real tree is itself a FINDING — a ledger
    row is a standing claim "this specific token, at this specific line
    hash, genuinely needs the escape"; once that token is deleted, edited,
    or (per B4/A1) brought back in-scope by a tightened exclusion class or
    the heading-line skip's removal, the row becomes a dead escape no
    scan_file run will ever match again, and a shrink-only ratchet with no
    liveness check on its OWN entries can silently accumulate orphans
    forever (the auditor found exactly one: the `[exit-code]` row for
    `mutants/README.md`'s "(exit 0)" token — already fully covered by the
    `exit N` lexical exclusion class, so `scan_file` never reaches the
    ledger-membership check for it at all)."""
    ledger = load_ledger()
    loader = Loader()
    consumed: set[str] = set()
    for rel_path in MEASUREMENT_FILES:
        scan_file(rel_path, loader, ledger, consumed_ledger_keys=consumed)
    unused = ledger - consumed
    return [f"ledger entry never consumed by any scan: {key!r}" for key in sorted(unused)]


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
    problems = check_coverage() + check_fence_integrity() + check_unused_ledger_entries()
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

        # --- B3 (round-22 audit): abs()-unwrap sign sensitivity + mirror
        # exemption. The five sites at mutants/README.md:156/158/160/162/238
        # used to wrap a scalar pointer in `abs()`, which (i) is sign-blind
        # (a negated artifact value still equals the SAME printed token,
        # since both sides go through `abs()`) and (ii) skips the ambiguity
        # leg entirely (`expr.kind != "ptr"`). Now unwrapped to bare `ptr`
        # bindings — restoring both properties, demonstrated directly.
        sign_loader = Loader()
        sign_loader._cache["fixture-sign.json"] = {"held_out_example_mean": -3.422172799706459}
        sign_ev = Evaluator(sign_loader, {}, "fixture-sign.json")
        bare_binding = sign_ev.eval(parse_expr("#/held_out_example_mean"))
        check(
            "B3 sign sensitivity restored: a negated artifact value FAILS a bare ptr "
            "binding against the (positive-printed) real token",
            not compare_token("3.422172799706459", bare_binding.value)[0],  # type: ignore[arg-type]
        )
        abs_binding_would_absorb = sign_ev.eval(parse_expr("abs(#/held_out_example_mean)"))
        check(
            "B3 contrast: the OLD abs()-wrapped form would have silently absorbed the "
            "identical sign flip (why the wrap was sign-blind, not why it's still used here)",
            compare_token("3.422172799706459", abs_binding_would_absorb.value)[0],  # type: ignore[arg-type]
        )

        # MIRROR-EXEMPTION: the real `held_out_example_mean` sites now bind
        # as bare `ptr`, so the ambiguity leg (b) runs on them — it must NOT
        # flag the KNOWN legit `held_out_example_mean`/`held_out_mean`
        # mirror (the untrained checkpoint logged both as the summary field
        # and under every `trajectory[i]`), driven end-to-end through the
        # REAL committed artifact (never a synthetic stand-in for this
        # specific pair — the exemption's whole point is this real
        # duplicate).
        mirror_doc_path = "docs/plans/63-how-well/measurements/campaign-v1/raw/seed1__fused__lr0.json"
        mirror_amb = _check_ambiguity(
            loader,
            mirror_doc_path,
            "/tiers/finetune_run/held_out_example_mean",
            "3.422172799706459",
            Decimal("3.422172799706459"),
        )
        check(
            "B3 mirror exemption: held_out_example_mean vs trajectory[].held_out_mean "
            "does NOT report ambiguous (known legit same-artifact mirror)",
            mirror_amb is None,
        )
        # adversarial: a DIFFERENT ambiguous pointer (not the exempted pair)
        # must still report — proving the exemption is narrow, not a
        # blanket "skip decimal collisions in this artifact" escape.
        amb_fixture_doc = {"decision": {"mean_d": 0.023799}, "unrelated": {"other_field": 0.023799}}
        amb_loader = Loader()
        amb_loader._cache["fixture-amb.json"] = amb_fixture_doc
        non_exempt_amb = _check_ambiguity(
            amb_loader, "fixture-amb.json", "/decision/mean_d", "0.023799", Decimal("0.023799")
        )
        check(
            "B3 mirror exemption is NARROW: a different (non-exempted) field-name "
            "collision still reports ambiguous",
            non_exempt_amb is not None,
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

    # --- round-23 audit F2: the entries-vs-tokens relation was audited
    # ONE-SIDED (entries < toks reported; entries > toks, a DEAD tag
    # directive, never did) -- driven end-to-end through the REAL
    # `scan_file` entry point, both shapes: (i) a tag with excess entries
    # above a line that STILL has some real tokens (partial dead entry),
    # and (ii) a tag with entries above a line with ZERO real tokens at
    # all (fully dead directive -- the exact shape the pre-fix `if not
    # toks: continue` made structurally invisible).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # (i) partial: one real token, two tag entries.
        content_partial = "<!-- claims63: c1=const; c2=const -->\na claim of 7 here\n"
        p_partial = _write_tmp(tmp, "dead-partial.md", content_partial)
        rel_partial = str(p_partial)
        findings_partial, _, _, _ = scan_file(rel_partial, Loader(), ledger=set())
        check(
            "F2 fixture (i): a tag with MORE entries than real tokens on the line "
            "below (1 token, 2 entries) REDs with a 'dead (unconsumed) entry' finding",
            any("dead (unconsumed) entry" in f.message for f in findings_partial),
        )
        # (ii) fully dead: a tag directive sits above a line with ZERO real
        # claim tokens (e.g. every numeral on the line is excluded by a
        # lexical class) -- the exact shape of the pre-fix mutants/README.md
        # :751/:819/:979 orphans.
        content_dead = "<!-- claims63: c1=const -->\nmerge exit 0 (nothing measured here)\n"
        p_dead = _write_tmp(tmp, "dead-full.md", content_dead)
        rel_dead = str(p_dead)
        findings_dead, in_zone_dead, _, _ = scan_file(rel_dead, Loader(), ledger=set())
        check(
            "F2 fixture (ii): a tag directive above a line with ZERO real claim "
            "tokens (every numeral lexically excluded) REDs with a 'dead directive' "
            "finding, not silently skipped",
            in_zone_dead == 0 and any("dead directive" in f.message for f in findings_dead),
        )
        # negative control: a tag whose entry count matches its line's real
        # token count exactly must NOT fire either leg.
        content_balanced = "<!-- claims63: c1=const -->\na claim of 7 here\n"
        p_balanced = _write_tmp(tmp, "balanced.md", content_balanced)
        rel_balanced = str(p_balanced)
        findings_balanced, _, _, _ = scan_file(rel_balanced, Loader(), ledger={f"const:{rel_balanced}:7:{line_hash('a claim of 7 here')}:12"})
        check(
            "F2 negative control: a tag whose entry count matches its real token "
            "count exactly fires NEITHER the dead-entry NOR the missing-entry leg",
            not any("dead" in f.message or "only" in f.message for f in findings_balanced),
        )

    # --- exclusion-class coverage (round-21 origin; round-25 STRUCTURAL
    # rewrite): each of the 31 `_EXTRA_EXCLUSIONS` classes excludes ONLY its
    # own shape -- a measured token dressed up in the SAME syntactic
    # neighborhood (adversarial fixture) must still be found. This is no
    # longer 62 hand-written `check()` calls (round-24's own commit message:
    # "the one-instance-at-a-time relay shape (F10)") -- it is a single
    # loop over `_EXCLUSION_FIXTURES` (defined next to `find_claim_tokens`
    # above), which was ALREADY asserted structurally at import time; this
    # loop re-runs the SAME table through `check()` purely so the existing
    # PASS/FAIL print-line audit trail (one line per class, matching every
    # prior round's convention) is preserved for a human `--self-test` read.
    # -----------------------------------------------------------------
    def toks_of(line: str) -> list[str]:
        return [t.text for t in find_claim_tokens(line, 1)]

    check(
        f"F-registry: _EXCLUSION_FIXTURES is index-complete against "
        f"_EXTRA_EXCLUSIONS ({len(_EXCLUSION_FIXTURES)} of {len(_EXTRA_EXCLUSIONS)}) "
        "-- the :247 'all 31 classes' universal, true by construction",
        len(_EXCLUSION_FIXTURES) == len(_EXTRA_EXCLUSIONS),
    )
    for _fx in _EXCLUSION_FIXTURES:
        check(
            f"exclusion({_fx.name}): {_fx.positive!r} excludes cleanly",
            tuple(toks_of(_fx.positive)) == _fx.positive_expected,
        )
        check(
            f"exclusion({_fx.name}) adversarial: a measured value beside "
            f"{_fx.positive!r} is still found",
            tuple(toks_of(_fx.adversarial)) == _fx.adversarial_expected,
        )

    # --- round-25 BITE sweep, WIRED (not left as an external one-off audit
    # script): for every one of the 31 classes, construct the WORST-CASE
    # (`.*`) span-extension on whichever side of the idiom the adversarial
    # value sits (auto-detected, `_worst_case_span_extension` above -- never
    # hand-coded per class) and confirm the widened pattern is CAUGHT (the
    # SAME adversarial fixture goes RED under it), proving each fixture is
    # load-bearing against the exact failure mode the round-25 audit found
    # (":637"/":651" silently surviving an unbounded widening) -- not just
    # a decorative pass on the CURRENT pattern. Measured cost: ~31 monkey-
    # patch-and-rescan probes, <10ms wall time on this corpus (round-25
    # audit measurement) -- cheap enough to run on every `--self-test`
    # rather than staying an external, human-rerun sweep.
    for _fx in _EXCLUSION_FIXTURES:
        _orig_pat = _EXTRA_EXCLUSIONS[_fx.index]
        _wsrc, _dir = _worst_case_span_extension(_orig_pat, _fx.adversarial, _fx.adversarial_expected)
        if _wsrc is None:
            check(f"BITE sweep({_fx.name}): the relevant widening direction is {_dir}", True)
            continue
        _wpat = re.compile(_wsrc, _orig_pat.flags)
        _EXTRA_EXCLUSIONS[_fx.index] = _wpat
        try:
            _widened_got = tuple(t.text for t in find_claim_tokens(_fx.adversarial, 1))
        finally:
            _EXTRA_EXCLUSIONS[_fx.index] = _orig_pat
        check(
            f"BITE sweep({_fx.name}): a worst-case {_dir} span-extension of this "
            "class's own pattern is CAUGHT -- the adversarial fixture goes RED",
            _widened_got != _fx.adversarial_expected,
        )

    # supplementary shape coverage beyond the registry's one canonical
    # positive per class (kept from round-22/23; not migrated into the
    # registry since each row there is exactly ONE positive + ONE
    # adversarial by design -- these probe additional REAL surface forms
    # of the SAME class, a strictly additive check).
    check(
        "exclusion(CONTRACT/unit/PLAN-v2-delta N): all three compose cleanly on one line",
        toks_of("CONTRACT 63 unit 63 PLAN v2 delta 7") == [],
    )
    check(
        "exclusion(theta=/g= bracket array): the 'g=' alternation form also excludes cleanly",
        toks_of("g=[0.1,-0.2,0.05,0.0] fixed input") == [],
    )
    check(
        "exclusion(seed-number list): the comma-separated form also excludes cleanly",
        toks_of("seeds 9, 12 committed") == [],
    )

    # B4 (round-22 audit): old-vs-new differential probes — the auditor's
    # EXACT examples. Under the pre-fix combined, IGNORECASE "Step N"
    # class, BOTH of these swallowed their own real measured claim whole
    # (`toks_of(...) == []`); anchored per-idiom, both numerals must now be
    # FOUND.
    check(
        "B4 differential (round-22 audit exact probe): 'detected at step 12/12 seeds' "
        "no longer swallowed WHOLE by the (now-anchored, capital-only) Step class — "
        "the second '12' surfaces; the first is still legitimately excluded by the "
        "SEPARATE, pre-existing epoch/step/t axis-label class (a genuine 'step N' "
        "training-step index, out of THIS fix's scope), not by a lowercase collision",
        toks_of("detected at step 12/12 seeds") == ["12"],
    )
    check(
        "B4 differential (round-22 audit exact probe): 'Step 8/12 seeds' no longer "
        "swallowed by the unbounded 'Step N/M' range — both numerals found (8 is "
        "outside this doc's own Step 1-3 range)",
        toks_of("Step 8/12 seeds") == ["8", "12"],
    )
    # ...while the doc's own REAL cross-reference sites (bounded 1-3) still
    # exclude cleanly, both singular and range/slash forms.
    check(
        "B4: real Step cross-refs still exclude cleanly (Step 1, Step 2/3, Step 1-3)",
        toks_of("Step 1") == [] and toks_of("Step 2/3") == [] and toks_of("Step 1-3") == [],
    )
    check(
        "B4: 'unit 63'/'unit-63' still excludes cleanly, but a DIFFERENT unit number "
        "(e.g. a training-unit count) is now found, not silently swallowed",
        toks_of("unit 63") == [] and toks_of("unit-63 round-7") == [] and toks_of("unit 12") == ["12"],
    )
    check(
        "round-23 F3: 'CONTRACT 63' still excludes cleanly, but a DIFFERENT CONTRACT "
        "number is now found, not silently swallowed (same discipline as 'unit 63')",
        toks_of("CONTRACT 63") == [] and toks_of("CONTRACT 12") == ["12"],
    )
    check(
        "B4: 'PLAN v2 delta 7' still excludes cleanly, but a bare 'delta 3' "
        "(e.g. CONTRACT.md's own per-item idiom) is now found, not silently swallowed",
        toks_of("PLAN v2 delta 7") == [] and toks_of("delta 3") == ["3"],
    )
    check(
        "B4: 's=1'/'s: 0 -> 1' still exclude cleanly, but 's=2' (outside the "
        "axis's own {0,1} domain) is now found",
        toks_of("s=1") == [] and toks_of("s: 0 -> 1") == [] and toks_of("s=2") == ["2"],
    )
    check(
        "B4: 'exit 0' still excludes cleanly, but 'exit 1' (a different, real exit "
        "code) is now found, not silently swallowed",
        toks_of("cargo build (exit 0)") == [] and toks_of("cargo build (exit 1)") == ["1"],
    )
    check(
        "self-test: real tree tokenizer matches pinned FILE_TOKEN_DENOMINATOR",
        len(check_coverage()) == 0,
    )

    # --- B1 (round-22 audit): fence carve-out fail-closed fixtures — the
    # auditor's exact laundering probe, both variants, plus a clean-tree and
    # a restore proof.
    check(
        "B1: check_fence_integrity() is clean on the real committed tree",
        check_fence_integrity() == [],
    )

    def _fence_problems_for(rel_path: str, text: str) -> list[str]:
        """Same two-part logic as `check_fence_integrity`, over an
        arbitrary text blob instead of the real committed file — lets the
        laundering probes below mutate a COPY without touching the real
        tree, while still exercising the exact parity/span-pin checks."""
        problems: list[str] = []
        fenced, delimiter_count = _fence_scan(text.splitlines())
        if delimiter_count % 2 != 0:
            problems.append(f"{rel_path}: odd fence delimiter count ({delimiter_count})")
            return problems
        expected = FENCE_LINE_DENOMINATOR.get(rel_path)
        if len(fenced) != expected:
            problems.append(f"{rel_path}: fenced-line count drifted: pinned {expected}, counted {len(fenced)}")
        return problems

    real_mutants_rel_for_fence = "docs/plans/63-how-well/mutants/README.md"
    real_mutants_text_for_fence = (REPO_ROOT / real_mutants_rel_for_fence).read_text()

    # laundering probe (a): append a false Measured line behind an UNCLOSED
    # ``` fence — the prior version had NO balance check at all, so this
    # silently fenced everything through EOF (never even a candidate for a
    # finding, regardless of the token-count pin). Must RED via PARITY.
    unclosed_text = real_mutants_text_for_fence + "\n```\nA laundered false Measured claim: 99.9.\n"
    check(
        "B1 laundering probe (a): a false claim behind an UNCLOSED ``` fence REDs "
        "via the PARITY check (odd delimiter count)",
        bool(_fence_problems_for(real_mutants_rel_for_fence, unclosed_text)),
    )

    # laundering probe (b): append a new line INSIDE an already-CLOSED
    # fence — balanced parity (even delimiter count), so probe (a)'s check
    # does not fire, and `FILE_TOKEN_DENOMINATOR` does not move either
    # (fenced content is never tokenized) — must RED via the SPAN pin
    # instead (`FENCE_LINE_DENOMINATOR`), the second, independent leg B1
    # adds.
    _fence_lines = real_mutants_text_for_fence.splitlines(keepends=True)
    _first_open_idx = next(i for i, l in enumerate(_fence_lines) if l.strip().startswith("```"))
    # one line PAST the opening delimiter — strictly interior to the block,
    # never the delimiter line itself (which would instead flip parity).
    inside_closed_fence_text = "".join(
        _fence_lines[: _first_open_idx + 1]
        + ["A laundered false Measured claim inside a CLOSED fence: 88.8.\n"]
        + _fence_lines[_first_open_idx + 1 :]
    )
    _, inside_delim = _fence_scan(inside_closed_fence_text.splitlines())
    check(
        "B1 laundering probe (b) setup: appending inside an existing fence keeps "
        f"parity EVEN ({inside_delim}) — probe (a)'s parity check does NOT fire here",
        inside_delim % 2 == 0,
    )
    check(
        "B1 laundering probe (b): a false claim appended INSIDE an already-CLOSED "
        "fence REDs via the SPAN pin (FENCE_LINE_DENOMINATOR), even though token "
        "count and fence parity are both unchanged",
        bool(_fence_problems_for(real_mutants_rel_for_fence, inside_closed_fence_text)),
    )

    # restore: an unmutated copy of the real file must stay clean under the
    # SAME per-text check the two probes above use — proving the probes
    # above discriminate on the injected mutation, not on the harness.
    check(
        "B1 restore: an unmutated copy of the real file is clean under the same "
        "per-text fence check",
        _fence_problems_for(real_mutants_rel_for_fence, real_mutants_text_for_fence) == [],
    )

    # A2 (round-23 audit): the DISCLOSED count-only compensating-edit limit
    # (see `FENCE_LINE_DENOMINATOR`'s own comment), demonstrated end-to-end
    # as a KNOWN-NEGATIVE fixture — asserting the CURRENT, disclosed
    # behavior (no finding), never a fake RED. A line is removed from ONE
    # fenced block and a DIFFERENT line is added to a DIFFERENT fenced
    # block, net fenced-line delta zero: parity stays even (no block was
    # un/re-closed) and the SPAN total is unchanged, so neither B1 leg
    # fires, even though the file's fenced CONTENT genuinely changed. If a
    # future fix closes this gap (a per-block or line-content pin), this
    # fixture's own expectation must flip from "not caught" to "caught" in
    # the SAME change — never left silently asserting a since-fixed limit.
    compensating_text = real_mutants_text_for_fence.replace(
        "+    const LR_INFLATION_FACTOR: f64 = 1.02_f64;  // 1.10 / 1.50 for the other doses\n",
        "",
        1,
    ).replace(
        "slope(seed) = held_out_mean(s=0) - held_out_mean(s=1)\n",
        "slope(seed) = held_out_mean(s=0) - held_out_mean(s=1)\n"
        "A laundered line added to a DIFFERENT fenced block, compensating the deletion above\n",
        1,
    )
    _, compensating_delim = _fence_scan(compensating_text.splitlines())
    _compensating_total = len(_fence_scan(compensating_text.splitlines())[0])
    _real_total = len(_fence_scan(real_mutants_text_for_fence.splitlines())[0])
    check(
        "A2 known-negative setup: the compensating edit removes one fenced line and "
        f"adds a different one elsewhere, net delta zero ({_real_total} -> {_compensating_total}), "
        f"parity stays even ({compensating_delim})",
        _compensating_total == _real_total and compensating_delim % 2 == 0,
    )
    check(
        "A2 known-negative (DISCLOSED LIMIT): a compensating edit across two "
        "DIFFERENT fenced blocks is NOT caught by either B1 leg — this is the "
        "documented count-only narrowing, asserted as CURRENT behavior, not a bug",
        _fence_problems_for(real_mutants_rel_for_fence, compensating_text) == [],
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
    real_mutants_rel = "docs/plans/63-how-well/mutants/README.md"
    real_mutants_path = REPO_ROOT / real_mutants_rel
    real_mutants_text = real_mutants_path.read_text()
    real_ledger = load_ledger()

    def _remap_ledger_to_tmp(new_rel: str) -> set[str]:
        """B2 (round-22 audit fix): the real ledger's `ledger`/`const` keys
        are prefixed with THIS file's real repo-relative path
        (`real_mutants_rel`). The prior version scanned a temp copy under a
        DIFFERENT path (`str(Path(td) / "README.md")`) without remapping,
        so every one of the ~173 mutants-scoped keys mismatched — a
        188-finding baseline (one per ledger/const entry, since none of
        their keys could ever match at the wrong path) loud enough that
        ANY perturbation, even a no-op, still "REDded": the auditor ran
        old==new and still got 188 findings, proving the five probes below
        were vacuous. Remapping every real key for THIS file onto the temp
        path (same trailing `:<token>:<hash>:<col>` shape, only the path
        prefix swapped) restores a byte-identical copy to a CLEAN scan, so
        each probe's finding count reflects ONLY the perturbation it
        introduces."""
        remapped: set[str] = set()
        prefix_plain = f"{real_mutants_rel}:"
        prefix_const = f"const:{real_mutants_rel}:"
        for key in real_ledger:
            if key.startswith(prefix_const):
                remapped.add(f"const:{new_rel}:{key[len(prefix_const):]}")
            elif key.startswith(prefix_plain):
                remapped.add(f"{new_rel}:{key[len(prefix_plain):]}")
        return remapped

    def _scan_tmp_copy(text: str) -> list[Finding]:
        with _tempfile.TemporaryDirectory() as td:
            # B2: the temp copy's OWN directory is named `mutants/` — not
            # just cosmetic: `scan_file` gates `parse_code_blocks` (the
            # fenced-code-block self-referential `code()` artifact) on
            # `"mutants" in rel_path`; a temp path with no such substring
            # would silently disable that artifact too, a SECOND
            # independent source of baseline noise the auditor's original
            # fixture never isolated from the ledger-key mismatch.
            p = Path(td) / "mutants" / "README.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            new_rel = str(p)
            remapped_ledger = _remap_ledger_to_tmp(new_rel)
            findings, _, _, _ = scan_file(new_rel, Loader(), ledger=remapped_ledger)
            return findings

    def _mutate_and_scan(old: str, new: str) -> list[Finding]:
        assert old in real_mutants_text, f"probe anchor text not found: {old!r}"
        mutated = real_mutants_text.replace(old, new, 1)
        return _scan_tmp_copy(mutated)

    # B2 baseline proof: a byte-identical temp copy (ledger keys remapped
    # to the temp path) must scan CLEAN — the corrected variant's own
    # anti-vacuity proof. Every probe below relies on this baseline being
    # 0; if it drifts nonzero again, every probe below becomes vacuous
    # again exactly as the pre-fix version was, so this is asserted
    # FIRST and by itself.
    baseline_findings = _scan_tmp_copy(real_mutants_text)
    check(
        "B2 baseline (round-22 fix): byte-identical temp copy, ledger remapped, "
        f"scans CLEAN ({len(baseline_findings)} findings; the pre-fix mismatched-path "
        "form carried ~188 findings on this SAME input)",
        len(baseline_findings) == 0,
    )

    # B2 no-op-mutation control: with the baseline now clean, a NO-OP
    # "mutation" (the anchor text replaced by itself) must produce ZERO
    # findings — i.e. it correctly FAILS the "expects a finding" shape
    # every real probe below asserts. This is a deliberate NEGATIVE
    # control (asserting the fixture discriminates, not that it always
    # fires) — printed and checked explicitly, never silently assumed.
    noop_findings = _mutate_and_scan(
        "RETIRED, measured INERT on GPU (12/12 bit-identical)",
        "RETIRED, measured INERT on GPU (12/12 bit-identical)",
    )
    check(
        f"B2 no-op-mutation control: a no-op edit produces NO finding ({len(noop_findings)}, "
        "not the pre-fix ~188) — proves the probes below discriminate rather than firing "
        "unconditionally",
        len(noop_findings) == 0,
    )

    # probe 1 (round-21 :929 analog): M_signflip v1's own "RETIRED, measured
    # INERT on GPU (12/12 bit-identical)" perturbed to a near-miss 11/12 --
    # was silently OUTSIDE every CLAIM_ZONES range before the inversion.
    f929 = _mutate_and_scan(
        "RETIRED, measured INERT on GPU (12/12 bit-identical)",
        "RETIRED, measured INERT on GPU (11/12 bit-identical)",
    )
    check(
        "round-21 probe :929 (M_signflip v1 12/12->11/12) REDs under the inverted model "
        f"with EXACTLY ONE new finding ({len(f929)}) — B2: the corrected baseline+remap "
        "makes the count discriminating, not just nonzero",
        len(f929) == 1,
    )

    # probe 2 (round-21 :153 analog): Step 1's own committed-benchmark table,
    # one full-precision `held_out_example_mean` value perturbed in its last
    # digit -- was silently OUTSIDE every CLAIM_ZONES range before the
    # inversion (the "committed-benchmark table" round-21 itself named).
    f153 = _mutate_and_scan("`3.218041628599167`", "`3.218041628599168`")
    check(
        "round-21 probe :153 (Step-1 table full-precision value, last digit) REDs under "
        f"the inverted model with EXACTLY ONE new finding ({len(f153)})",
        len(f153) == 1,
    )

    # probe 3 (round-21 :551 analog): M1's own "8/12, well under either
    # threshold" restatement perturbed to a near-miss 9/12 -- was silently
    # OUTSIDE every CLAIM_ZONES range before the inversion.
    f551 = _mutate_and_scan(
        "sign-flipping-transient shape: 8/12, well under either threshold);",
        "sign-flipping-transient shape: 9/12, well under either threshold);",
    )
    check(
        "round-21 probe :551 (M1 8/12->9/12 restatement) REDs under the inverted model "
        f"with EXACTLY ONE new finding ({len(f551)})",
        len(f551) == 1,
    )

    # probe 4 (round-21 "EOF-append"): a brand-new, untagged measured-shaped
    # claim appended past the end of the file -- the exact shape a
    # covered-region ALLOWLIST is structurally blind to (nothing further
    # than the last zone was ever scanned); under the inverted (default-in-
    # scope) model this REDs exactly like any other untagged line.
    f_eof = _mutate_and_scan(
        real_mutants_text[-40:],
        real_mutants_text[-40:] + "\nA freshly appended, untagged measured claim: 42.42.\n",
    )
    check(
        "round-21 probe (EOF-append, untagged) REDs under the inverted model with "
        f"EXACTLY ONE new finding ({len(f_eof)})",
        len(f_eof) == 1,
    )

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
    check(
        "round-21 probe :1111 analog (Files-section 12/12 restatement) STILL REDs "
        f"post-inversion with EXACTLY ONE new finding ({len(f1111)})",
        len(f1111) == 1,
    )

    # --- round-23 audit F1 (LIVE false-pass): the auditor's EXACT
    # perturbation — `12-leg GPU` -> `9-leg GPU` — reproduced at all 7
    # committed sites the exclusion-class fix now exposes (before the fix,
    # the "N-leg" wholesale exclusion silently swallowed this numeral at
    # every one of the 7 sites, so this same probe left the gate at
    # 478/478 PASS; the class no longer exists, and every site is bound
    # per-occurrence to its own artifact's `gate_seed_count`/`paircount`,
    # so each must now RED with exactly one new finding).
    f_leg_1 = _mutate_and_scan(
        "**Measured (12-leg GPU, a100, `redproof-nobc`): NOT DETECTED (raw).**",
        "**Measured (9-leg GPU, a100, `redproof-nobc`): NOT DETECTED (raw).**",
    )
    check(
        f"F1 probe 1/7 (mutants/README.md:752, M_nobc raw): 12-leg -> 9-leg REDs "
        f"with EXACTLY ONE new finding ({len(f_leg_1)})",
        len(f_leg_1) == 1,
    )
    f_leg_2 = _mutate_and_scan(
        "**Measured (12-leg GPU, a100, `redproof-signflip`): INERT — 12/12 legs",
        "**Measured (9-leg GPU, a100, `redproof-signflip`): INERT — 12/12 legs",
    )
    check(
        f"F1 probe 2/7 (mutants/README.md:820, M_signflip v1 INERT): 12-leg -> 9-leg "
        f"REDs with EXACTLY ONE new finding ({len(f_leg_2)})",
        len(f_leg_2) == 1,
    )
    f_leg_3 = _mutate_and_scan(
        "reaches the arm the 12-leg GPU gate actually exercises. This is the",
        "reaches the arm the 9-leg GPU gate actually exercises. This is the",
    )
    check(
        f"F1 probe 3/7 (mutants/README.md:839, dispatch-invariant lesson): 12-leg -> "
        f"9-leg REDs with EXACTLY ONE new finding ({len(f_leg_3)})",
        len(f_leg_3) == 1,
    )
    f_leg_4 = _mutate_and_scan(
        "**Measured (12-leg GPU, a100, `redproof-signflip-v2`, D*-gated): RED —",
        "**Measured (9-leg GPU, a100, `redproof-signflip-v2`, D*-gated): RED —",
    )
    check(
        f"F1 probe 4/7 (mutants/README.md:966, M_signflip_v2 D*-gated RED): 12-leg -> "
        f"9-leg REDs with EXACTLY ONE new finding ({len(f_leg_4)})",
        len(f_leg_4) == 1,
    )
    f_leg_5 = _mutate_and_scan(
        "**Measured record (12-leg GPU, a100), current-truth discipline:** committed",
        "**Measured record (9-leg GPU, a100), current-truth discipline:** committed",
    )
    check(
        f"F1 probe 5/7 (mutants/README.md:1079, RED-proof mutants summary): 12-leg -> "
        f"9-leg REDs with EXACTLY ONE new finding ({len(f_leg_5)})",
        len(f_leg_5) == 1,
    )
    f_leg_6 = _mutate_and_scan(
        "patch-file-only; **measured NOT-DETECTED (raw)** on 12-leg GPU",
        "patch-file-only; **measured NOT-DETECTED (raw)** on 9-leg GPU",
    )
    check(
        f"F1 probe 6/7 (mutants/README.md:1175, Files: M_nobc.patch): 12-leg -> 9-leg "
        f"REDs with EXACTLY ONE new finding ({len(f_leg_6)})",
        len(f_leg_6) == 1,
    )
    f_leg_7 = _mutate_and_scan(
        "patch-file-only; **RETIRED — measured INERT on 12-leg GPU**",
        "patch-file-only; **RETIRED — measured INERT on 9-leg GPU**",
    )
    check(
        f"F1 probe 7/7 (mutants/README.md:1185, Files: M_signflip.patch): 12-leg -> "
        f"9-leg REDs with EXACTLY ONE new finding ({len(f_leg_7)})",
        len(f_leg_7) == 1,
    )

    # --- round-23 audit F2 regression: reproducing the auditor's exact
    # finding on the real, committed tree -- mutants/README.md:979 (pre-fix)
    # carried a `c1=ledger` tag directly above a line whose only numeral
    # ("exit 0") is lexically excluded, so the directive bound to NOTHING
    # and was invisible to the one-sided pre-fix check (now removed as
    # dead -- see the markdown itself). Driven end-to-end on a mutated
    # COPY that reintroduces the exact orphan, through the real `scan_file`
    # entry point (never a synthetic-only fixture, so a regression of the
    # fix itself on the real file is caught).
    f_dead_directive_regression = _mutate_and_scan(
        "`detected=RED`, two-sided `p = 2/4096 = 1/2048 = 0.00048828125` exact;\n"
        "`red_proof_verdict = PROVEN`;",
        "`detected=RED`, two-sided `p = 2/4096 = 1/2048 = 0.00048828125` exact;\n"
        "<!-- claims63: c1=ledger -->\n"
        "`red_proof_verdict = PROVEN`;",
    )
    check(
        "F2 regression probe (mutants/README.md, the pre-fix dead-directive shape "
        f"reintroduced): a dead directive above 'exit 0' REDs with EXACTLY "
        f"ONE new finding ({len(f_dead_directive_regression)})",
        len(f_dead_directive_regression) == 1,
    )

    # --- A1 (round-22 audit): heading lines are now IN SCOPE. Unit-level:
    # a heading line is tokenized exactly like prose (no more whole-line
    # `^#` skip). End-to-end: a threshold flip written directly INTO a
    # heading, and an injected false claim appended to a heading, must
    # both RED through the real `scan_file` entry point on the real,
    # committed file — the auditor's exact two probes, reproduced as
    # regression fixtures the same way the round-21 probes are above.
    check(
        "A1 unit: a heading line is tokenized like prose (no whole-line '#' skip)",
        toks_of("# status: threshold 11 met, gate_seed_count 12") == ["11", "12"],
    )
    # probe A1-a (threshold flip IN a heading): mutants/README.md's own
    # "Step 3" heading names the pre-registered ">=11/12+mean" rule
    # directly in its title; flipping "11" to "10" there must RED (this
    # exact heading now carries a real `claims63` tag one line above it,
    # bound to `#/decision/threshold` / `#/decision/gate_seed_count`).
    f_heading_threshold = _mutate_and_scan(
        "### Step 3 — predicted detection verdict under the >=11/12+mean rule",
        "### Step 3 — predicted detection verdict under the >=10/12+mean rule",
    )
    check(
        f"A1 probe (threshold flip IN a heading) REDs with EXACTLY ONE new finding "
        f"({len(f_heading_threshold)}) — the pre-fix whole-line heading skip would "
        "have silently absorbed this",
        len(f_heading_threshold) == 1,
    )
    # probe A1-b (injected false heading claim): appending a brand-new,
    # untagged measured-shaped claim INTO the file's own H1 title line —
    # the exact shape the pre-fix heading skip was structurally blind to,
    # since nothing on a `#` line was ever even a candidate for a finding.
    f_heading_inject = _mutate_and_scan(
        "# Unit 63 H5 — kernel-mutant RED column",
        "# Unit 63 H5 — kernel-mutant RED column (spurious measured 77.7)",
    )
    check(
        f"A1 probe (injected false claim INTO a heading) REDs with EXACTLY ONE new "
        f"finding ({len(f_heading_inject)}) — untagged, so no claims63 tag above covers it",
        len(f_heading_inject) == 1,
    )

    # --- A2 (round-22 audit): the unused-ledger-entry leg. Unit-level: a
    # synthetic ledger with one consumed key and one orphan must report
    # ONLY the orphan (never the consumed one, never a false positive on
    # the whole ledger going unused because the real tree wasn't scanned).
    with _tempfile.TemporaryDirectory() as _td2:
        fixture_md = Path(_td2) / "fixture-unused.md"
        fixture_content = "<!-- claims63: c1=ledger -->\nconsumed 7 claim\n"
        fixture_md.write_text(fixture_content)
        fixture_rel2 = str(fixture_md)
        fixture_line2 = fixture_content.splitlines()[1]
        fixture_tok2 = find_claim_tokens(fixture_line2, 2)[0]
        consumed_key = f"{fixture_rel2}:{fixture_tok2.text}:{line_hash(fixture_line2)}:{fixture_tok2.col}"
        orphan_key = f"{fixture_rel2}:999:deadbeefdeadbeefdeadbeefdeadbeefdeadbeef:0"
        consumed_seen: set[str] = set()
        scan_file(fixture_rel2, Loader(), ledger={consumed_key, orphan_key}, consumed_ledger_keys=consumed_seen)
        check(
            "A2 unit: a ledger key an actual scan matches is marked consumed",
            consumed_key in consumed_seen,
        )
        check(
            "A2 unit: an orphan ledger key (never matched by any scan) is correctly "
            "NOT marked consumed",
            orphan_key not in consumed_seen,
        )

    check(
        "A2: check_unused_ledger_entries() is clean on the real committed tree/ledger "
        "(the [exit-code] mutants/README.md:0 row, orphaned by the pre-existing "
        "'exit N' exclusion class, has been removed)",
        check_unused_ledger_entries() == [],
    )

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
