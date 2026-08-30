#!/usr/bin/env python3
"""Merge + table stage for `ci/scripts/perf/gpu_inference_ab.sh`'s issue #335
within-run GPU perf A/B — parent-HEAD vs a PR change, measured back to back
on the SAME rented pod so the device and its conditions cancel by
construction (never a resurrected absolute baseline).

Importable (never an inline heredoc, `ab_merge.py`'s own B3 lesson: zero
automated coverage otherwise) — `test_gpu_inference_ab.py` in this same
directory drives the REAL entry point (`main`) against fixture leg
directories shaped like `gpu_inference_ab.sh`'s own `.exit`/`.json`/`.stderr`
output, never a hand-rolled call with literal tuples standing in for a
report.

## Recording-only by default; enforcing only when explicitly opted into

The multi-pod/both-device-model validation issue #335's own exit criterion
required before any enforcement is now DONE — the `--aa-null` empirical-null
campaign committed under `ci/artifacts/gpu-perf-aa-null/` — and this
module's advisory band is now a real, PRE-REGISTERED one derived from that
evidence (see this module's own "ADVISORY classification" doc below). That
did not flip this module's DEFAULT posture: NON-enforcing (the default)
NEVER fails a run over the measured *ratio* — the only thing it refuses
(nonzero exit, status `INVALID`/`INVALID_MEASUREMENT`) is a PREMISE mismatch
(the two legs did not measure the same thing at all, so no ratio is even
meaningful) or a correctness-of-measurement defect. A measured ratio,
however large, is still always recorded and printed. ENFORCING mode
(`GPU_INFERENCE_AB_ENFORCE=1`, opt-in per invocation — see this module's own
"Enforcement flip" doc below) is the one addition: a GREEN-premise run whose
ratio falls outside the pre-registered band now also refuses, with its own
NAMED verdict, distinct from every premise-mismatch refusal above.

## Refusal core: reused, never hand-rolled

Leg-premise identity is checked via `ab_merge.generic_leg_identity_fields`/
`ab_merge.generic_leg_premise_violations` — the SAME shared refusal core
`encode_ab.sh`'s own merge stage already builds on
(`identity_fields.ENCODE_IDENTITY_FIELDS`), now driven against
`identity_fields.GPU_INFERENCE_IDENTITY_FIELDS`
(`GpuInferenceTier::IDENTITY_FIELDS`, `report.rs`). This module hand-rolls
NO second identity comparator.

## Order-balanced legs: A, B, B, A

`gpu_inference_ab.sh` runs FOUR legs in the fixed order A, B, B, A —
`a1` (parent), `b1` (pr), `b2` (pr), `a2` (parent) — never A, A, B, B.

### What actually cancels, and what does not (round-1 adversarial audit B4 correction)

An EARLIER version of this doc claimed adjacent-pair averaging (below) was
a SUPERIOR estimator to a naive mean-of-all-A-vs-mean-of-all-B one — that
claim was FALSE, and is corrected here rather than quietly dropped.

The FIRST-ORDER cancellation this design relies on is bought by the A, B,
B, A ORDER ITSELF, under a MULTIPLICATIVE linear drift model — the
physically relevant one for a clock/thermal effect (a GPU that throttles
increasingly over a run scales EVERY measurement's wall-time by a growing
FACTOR, not by a fixed absolute offset). Placing the two `b`-role legs
symmetrically BETWEEN the two `a`-role legs makes the MEAN measurement TIME
of the `a`-role legs equal the mean measurement time of the `b`-role legs,
so a multiplicative drift trend's first-order term cancels under EITHER
reasonable combining convention: BOTH adjacent-pair averaging
([`combined_embed_p50_ratio`] below) AND a naive mean(all `b`) /
mean(all `a`) estimator are unbiased to first order under THIS order.
Adjacent-pairing is therefore a REPORTING convention, not a smaller-bias
estimator — it additionally surfaces two per-pair ratios on the merged
report (`adjacent_pair_ratios`) for diagnostic visibility, a genuine
benefit, but not the source of the cancellation itself.
`test_gpu_inference_ab.py::DriftCancellationTests` proves both halves of
this mechanically: a synthetic multiplicative drift recovers the true
ratio to first order under the REAL A, B, B, A order, and the SAME drift
would NOT have cancelled under an A, A, B, B order (never this producer's
actual order).

Under an ADDITIVE linear drift model (a fixed absolute offset per unit
time, rather than a percentage), NEITHER convention cancels the
first-order term when `b_true != a_true` — the residual bias is of
comparable magnitude and shape under both. This residual is an HONEST,
DOCUMENTED limitation of v1, not silently closed: the `--aa-null`
empirical-null instrument (D6, `gpu_inference_ab.sh`'s own header) measures
the REAL combined drift-plus-noise distribution this residual (and every
other source of run-to-run variance) actually produces on real hardware —
the intended route to a real, evidence-derived tolerance band, never a
closed-form correction applied here.

## ONE pre-registered primary endpoint: embed `p50_ms` ratio (PR / parent)

The gated (in a FUTURE, fleet-validated unit — never this one) comparison is
`b.embed.p50_ms / a.embed.p50_ms`, per adjacent pair, then averaged over the
two pairs — [`combined_embed_p50_ratio`]. This is the ONE metric this module
treats as "the" A/B signal. Everything else on the tier is RECORDED, never
part of any classification:

  * `embed.rows_per_s` / `infer.rows_per_s` — `rows_per_s` is `rows /
    (p50_ms / 1000)` BY CONSTRUCTION (`gpu_inference.rs`'s own
    `rows_per_s` helper divides the SAME `p50_ms` this endpoint already
    reads); a second ratio over it would not be an independent signal, it
    would be (up to the constant `rows` term) `1 / (the same ratio inverted)`
    — restating the primary endpoint under a different name, not a second
    measurement.
  * `embed.p99_ms` / `infer.p99_ms` — the max (or near-max) of a SHORT
    (`iters`, typically 20) sample; an order statistic that far into the
    tail is high-variance run to run by construction, not a stable ratio
    target for a v1 recording-only instrument.
  * The WHOLE `infer` (classification) lane — recorded for visibility, never
    folded into the primary endpoint: this tier states one workload as
    primary (embed), matching `GpuInferenceTier::compute_precision`'s own
    "one endpoint, one identity field" framing (see that field's own doc in
    `report.rs`).

## ADVISORY classification: a PRE-REGISTERED band, derived from the D6
## empirical-null campaign (issue #335 final unit)

[`PRE_REGISTERED_ADVISORY_BAND`] below is what this module classifies
`combined_embed_p50_ratio` against for the `advisory` column in the printed
table. It IS a pre-registered decision rule (D6): derived from the `--aa-null`
empirical A/A null campaign committed under `ci/artifacts/gpu-perf-aa-null/`
— THREE PRIMARY runs (six primary pairs), two rented pods, spanning BOTH
A100 device models (SXM4, PCIe), all measured 2026-08-30 against the same
`main` tip, all `status=GREEN`. Two further runs (`pcie-p1`, `pcie-p2`) were
also executed and are also committed, but are AUXILIARY, not primary — that
directory's own README.md "Disclosure" section documents why (the two ran
CONCURRENTLY on one shared PCIe pod, a GPU-contention confound this
campaign's isolation assumption does not cover; `pcie-p2` separately shows
an anomalous driver return code). That same README carries the full
campaign protocol, per-run table, and the characterization findings the
band below is derived from (a binary-level build-to-build effect whose sign
tracks device model, a within-run adjacent-pair spread, and the sensitivity
this band actually has) — never re-copied here (a fact lives in exactly one
place; this doc cites it), EXCEPT the sensitivity paragraph immediately
below, restated VERBATIM per this module's own citation discipline (a
number this consequential earns a second, independently-checkable copy,
never a single point of truth a reader must trust was transcribed
correctly).

**Sensitivity** (restated verbatim from `ci/artifacts/gpu-perf-aa-null/README.md`'s
own "(c) Sensitivity: what this band can and cannot catch" section): With
the upper edge at `1.33`, the smallest SLOWDOWN this band can catch on an
idealized (zero build-offset) pod is `> 33%` (`1.33 − 1`), not the `≥ 25%`
an earlier draft of this doc claimed — `25%` is the LOWER edge's own
distance from 1.0 (`1 − 0.75`), a different, asymmetric threshold that does
not describe the slowdown-catching side at all. On a REAL pod, that nominal
33% figure is itself optimistic: this campaign's own two primary SXM4
combined ratios (`0.8706549652288303`, `0.8821655548443332`) show the
binary-level build offset (finding (a) above) already suppresses the observed ratio by ≈12.9% and ≈11.8% respectively on that
device model, working AGAINST detection of a real slowdown (a slowdown and
a favorable build offset partially cancel in the observed ratio). To still
cross the `1.33` upper edge despite that offset, a real slowdown needs to
overcome it first: `1.33 / 0.8706549652288303 − 1 ≈ 52.8%` under the worse
of the two observed SXM4 offsets, or `1.33 / 0.8821655548443332 − 1 ≈
50.8%` under the milder one — i.e. on a pod exhibiting an SXM4-like build
offset, the REAL smallest reliably-catchable slowdown is on the order of
**~51–53%**, not 33% and certainly not 25%. Tightening this requires
endpoint-precision work (more iters/rows, replicate medians, or
characterizing and correcting for the build-offset effect itself), never
band tuning alone.

**Derivation**: 1.5x the largest `|log deviation|` observed among the
THREE PRIMARY runs' six pair ratios (worst pair ratio `0.8315173238022384`
-> `|log|` `0.18450314616782526`; 1.5x -> `0.27675471925173794`;
`exp(∓0.27675471925173794)` ≈ `[0.7582404560899295, 1.3188428445994143]`)
rounded outward via [`derive_advisory_band`]'s own rule (floored lower
edge; upper edge = the WIDER of the floored reciprocal and the per-edge
ceiling — see that function's own PROVENANCE note for why, and for the
disclosure that the rule was formalized after the constant)
to `[0.75, 1.33]` — mechanically re-derived from the committed evidence and
its `manifest.json` classification, and checked equal to
[`PRE_REGISTERED_ADVISORY_BAND`], by `ci/scripts/check_aa_null_band.py`
(wired into `ci.yml`'s Guard matrix) — this is ENFORCED, never merely
asserted in prose. See `ci/artifacts/gpu-perf-aa-null/README.md`'s own
"Band derivation" section for the full worked numeric example, including
why the rounding rule produces `1.33` rather than the `1.32` an
independent per-edge ceiling would give.

`advisory` classifying `"outside_band"` NEVER changes this module's exit
code by itself — see [`GPU_INFERENCE_AB_ENFORCE`] / the `enforce` marker
below for the ONE thing that does, and only when explicitly opted into.

## Enforcement flip: `GPU_INFERENCE_AB_ENFORCE` (issue #335 final unit,
## DIRECTION-HONEST on a two-sided null band — round-4 delta-audit F1/F4 fix)

Two modes, both driven by the `enforce` marker `gpu_inference_ab.sh` writes
into `raw_dir` before any leg runs (the SAME file-based state-passing
convention the `mode` marker already uses — see [`load_mode`] — never an
env var threaded directly into this module):

  * **non-enforcing (the default)** — `enforce_verdict` on a `GREEN` report
    reads `"NOT_ENFORCED"`; a ratio outside [`PRE_REGISTERED_ADVISORY_BAND`]
    is recorded and printed, exactly as before this unit, but NEVER changes
    the exit code. This is still v1's own recording-only posture — pre-
    registering a real band did not, by itself, turn this into a gate.
  * **enforcing** (`GPU_INFERENCE_AB_ENFORCE=1`) — REQUIRES `mode == "ab"`
    (see the `ENFORCE_INVALID_MODE` arm below) and, once that premise
    holds, on a `GREEN` report (the four legs' premises are confirmed to
    agree) whose `combined_embed_p50_ratio` falls OUTSIDE the pre-registered
    band, refuses with a DIRECTION-HONEST verdict — this band is a NULL
    band (derived from a parent-vs-parent A/A campaign, never a "faster is
    always fine" one), so a ratio below the lower edge is NOT assumed
    favorable:
      - `combined_embed_p50_ratio` ABOVE the upper edge —
        `enforce_verdict` reads `"PERF_REGRESSION"`: a real signal the
        `b`-role (PR) leg measured SLOWER than the null distribution can
        explain.
      - `combined_embed_p50_ratio` BELOW the lower edge —
        `enforce_verdict` reads `"OUTSIDE_BAND_FAST"`, deliberately a
        DIFFERENT name, never folded into `PERF_REGRESSION`: the `b`-role
        leg measured FASTER than the null distribution can explain, which
        is AMBIGUOUS on its own — either a genuine large improvement, OR a
        `b`-role leg that silently short-circuited or broke and finished
        suspiciously fast (the same signature this repo's own torch-cu130
        CPU-fallback escape carries: "too fast" is what a leg not actually
        doing the work looks like, not proof of a real speedup). This
        module CANNOT distinguish the two from the ratio alone, so it
        never guesses; the `outside_band_fast_reason` states the dual
        possibility explicitly and demands human adjudication, never
        auto-classifying either way.
    BOTH shapes exit `1` — fail CLOSED on any GREEN-premise ratio outside
    the null band, regardless of direction; only a ratio inside the band
    (INCLUSIVE of both edges — `lo <= ratio <= hi`, exactly what
    `classify_advisory` computes; a ratio landing exactly ON an edge is
    within the pre-registered null) earns `enforce_verdict = "PASS"`,
    exit `0`. Both refusal
    verdicts are deliberately carried on a SEPARATE field (`enforce_verdict`,
    never folded into `status`, which stays `"GREEN"`) from every
    correctness-refusal reason below, so a report reader can always tell
    "the premises were wrong" (`status != GREEN`) apart from "the premises
    were fine but the measured ratio itself fell outside the null band"
    (`status == GREEN`, `enforce_verdict` one of `"PERF_REGRESSION"`/
    `"OUTSIDE_BAND_FAST"`) without parsing prose.
  * **`ENFORCE_INVALID_MODE`** — enforcement is defined ONLY for a real A/B
    run (`mode == "ab"`): under `--aa-null`, both `b`-role legs are ALSO
    parent-sha clones (no PR exists to enforce against — the null band was
    ITSELF derived from exactly this shape of run, so enforcing it against
    another null run would misfire the very instrument the band comes
    from), and an unconfirmed/absent `mode` (an older producer) means this
    module cannot even confirm which shape the run is. Requesting
    enforcement (`GPU_INFERENCE_AB_ENFORCE=1`) on a `GREEN` report whose
    `mode` is NOT `"ab"` refuses (`enforce_verdict = "ENFORCE_INVALID_MODE"`,
    exit `1`) rather than silently downgrading to `"NOT_ENFORCED"` — an
    operator's explicit enforcement request must never be swallowed without
    any record it was ignored; this is validated BEFORE the band is even
    consulted (a mode mismatch refuses regardless of the ratio's own value).
    Enforcement NEVER overrides a correctness refusal — every `status !=
    "GREEN"` exit path below is entirely unaffected by the `enforce`
    marker's value; this module only ever consults it once premises are
    already confirmed clean.

## Exit codes (round-3 adversarial audit B2/B3's reconciled lattice, extended
## by issue #335's final unit's enforcement flip (e)/(f)/(g) below — matches
## `gpu_inference_ab.sh`'s own header table and `gpu-perf-ab.yml`'s own step
## annotations verbatim)

  * `0`  — GREEN: all four legs are `OK`, their RECORDED run order verifies
           as A,B,B,A ([`verify_recorded_order`]), and they agree on
           [`identity_fields.GPU_INFERENCE_IDENTITY_FIELDS`]. The ratio is
           computed and printed regardless of its own value; under
           enforcing mode, `0` also requires `mode == "ab"` AND the ratio
           landed INSIDE [`PRE_REGISTERED_ADVISORY_BAND`]
           (`enforce_verdict="PASS"`) — see (e)/(f)/(g) below for the three
           cases that still exit `1` from this same `GREEN` status.
  * `1`  — a real correctness-of-measurement refusal, ONLY ever raised once
           this module can CONFIRM the signal is real, in FOUR shapes: (a)
           INVALID (identity) — at least two `OK` legs DIFFER on a declared
           identity field's actual VALUE; (b) INVALID_MEASUREMENT — the
           legs' premises agree, but computing the primary endpoint's ratio
           itself raised (a malformed lane/metric, a zero baseline —
           [`_measurement_value`]/[`adjacent_pair_ratio`] deliberately
           raise rather than silently substitute a placeholder); (c) a
           `b`-role leg (`b1`/`b2`) RAN but did not produce an `OK` report
           (`outcome == "FAIL"`, never `MISSING`/`DRY_RUN` — "nothing ran"
           carries no runtime signal at all) **AND** the producer's own
           `mode` marker confirms `"ab"` (round-3 adversarial audit B2:
           under `--aa-null`, `b`-role legs are ALSO parent-sha clones — no
           PR exists to blame; an UNKNOWN mode, an older producer, never
           escalates here either — this module cannot claim a signal it
           cannot confirm); (d) the four legs' RECORDED start order does
           not verify as A,B,B,A AND every timestamp actually PARSED
           (round-3 adversarial audit B3: a missing/unparseable timestamp
           is a SEPARATE, neutral case — `INCOMPLETE_ORDER`/`75` below,
           never this bucket). All four refuse loudly (never silently drop
           the offending leg/field and compute a ratio anyway), and all
           four still WRITE a report (never an uncaught traceback with
           nothing recorded at all). PLUS, under ENFORCING mode only, THREE
           further, entirely different shapes never present when `enforce`
           is unset — in every one of them `status` stays `"GREEN"` (the
           four legs' premises really did agree; NONE of these are a
           correctness refusal), distinguishing them from (a)-(d) above by
           `status` alone, never by the bare exit code:
             - (e) `PERF_REGRESSION` — `combined_embed_p50_ratio` fell
               ABOVE the upper edge of [`PRE_REGISTERED_ADVISORY_BAND`], so
               `enforce_verdict` reads `"PERF_REGRESSION"` with its own
               `perf_regression_reason` — a real, unambiguous slowdown
               signal.
             - (f) `OUTSIDE_BAND_FAST` — `combined_embed_p50_ratio` fell
               BELOW the lower edge, so `enforce_verdict` reads
               `"OUTSIDE_BAND_FAST"` with its own
               `outside_band_fast_reason` — an AMBIGUOUS signal (a genuine
               improvement, or a broken/short-circuited leg reading
               suspiciously fast); this module never guesses which, it
               only refuses and names the ambiguity for a human to
               adjudicate.
             - (g) `ENFORCE_INVALID_MODE` — enforcement was requested
               (`GPU_INFERENCE_AB_ENFORCE=1`) but `mode != "ab"`
               (`"aa-null"` or unconfirmed/`None`) — `enforce_verdict`
               reads `"ENFORCE_INVALID_MODE"`, checked BEFORE the band is
               even consulted, so this arm can fire regardless of the
               ratio's own value.
  * `75` — neutral "nothing to compare safely", never a sign of a code
           problem, in THREE distinct shapes:
             - `INCOMPLETE`: fewer than all four legs produced an `OK`
               report, and the CONFIRMED-real b-role-FAIL-in-ab-mode
               precondition in (c) above does not hold — a `MISSING`/
               `DRY_RUN` leg of EITHER role ("nothing ran" — round-3
               adversarial audit B2), a PARENT-side (`a`-role) build/
               runtime failure, or a `b`-role FAIL under `--aa-null`/an
               unconfirmed `mode` (no PR to blame, or nothing confirmed).
             - `INCOMPLETE_IDENTITY` (round-1 adversarial audit B3): every
               `OK` leg's identity DISAGREEMENT is a field MISSING entirely
               from one side's record OR present but explicit JSON `null`
               (never a genuine differing VALUE) — the honest shape a
               PARENT leg built before issue #335's own identity contract
               landed produces (it simply cannot EMIT a field this version
               of the tool never knew to record). This is NOT the same
               claim as "the two legs proved they ran a different premise"
               (that is `1` above) — it is "this comparator cannot even ASK
               the question of one of the fields", which is closer in kind
               to "could not run" than to "ran and disagreed". A SINGLE
               differing-VALUE violation among the same set immediately
               promotes the WHOLE refusal to `1` — a genuine divergence is
               never masked by an ALSO-missing field elsewhere.
             - `INCOMPLETE_ORDER` (round-3 adversarial audit B3): one or
               more legs' `.started_at` files could not be read at all
               (missing) or did not parse as a plain integer (e.g. a
               non-GNU `date` binary emitting a different format than
               `%s%N`) — this is an environment/producer-version gap, NOT
               itself proof the A,B,B,A order was violated, so it must
               never land in the SAME bucket a genuine, PARSED
               out-of-order timestamp earns (that stays `1`, case (d)
               above).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_merge  # noqa: E402
from identity_fields import GPU_INFERENCE_IDENTITY_FIELDS  # noqa: E402

# The fixed A, B, B, A leg order `gpu_inference_ab.sh` runs — see this
# module's own doc for why this exact order (never A, A, B, B) matters.
# `ROLE_OF_LEG` names which clone (`"a"` = parent-shaped, `"b"` =
# pr-shaped/comparison-shaped) each leg name is, WITHOUT hardcoding
# "parent"/"pr" into the comparison machinery itself — `--aa-null` reuses
# this exact module with role `"b"` played by a THIRD parent-sha clone
# rather than a PR clone (see this module's own doc), so the machinery below
# never assumes role `"b"` means "the PR".
LEG_ORDER = ("a1", "b1", "b2", "a2")
ROLE_OF_LEG = {"a1": "a", "b1": "b", "b2": "b", "a2": "a"}
ADJACENT_PAIRS = (("a1", "b1"), ("b2", "a2"))

# PRE-REGISTERED (D6, issue #335 final unit) — see this module's own
# "ADVISORY classification" doc section above for the full derivation,
# restated verbatim from `ci/artifacts/gpu-perf-aa-null/README.md`'s own
# "Band derivation" section. A ratio inside this closed interval classifies
# `advisory="within_band"`; outside it, `advisory="outside_band"`. Neither
# outcome changes this module's exit code UNLESS `enforce` is set (see
# `load_enforce`/[`build_report`]'s own enforcement-flip doc above).
PRE_REGISTERED_ADVISORY_BAND = (0.75, 1.33)

# The committed evidence this band is derived from — cited verbatim on the
# merged report's own `advisory.band_derivation` field and printed on the
# GREEN table row, so a reader never has to trust the band's two numbers on
# prose alone. "2 pods" is a true count of PHYSICAL HARDWARE rented for the
# whole campaign — it is NOT a claim that all five committed runs had
# exclusive use of that hardware; see the cited README's own "Disclosure"
# section for the one pair of runs (pcie-p1/pcie-p2) that did not, and why
# the PRIMARY evidence base is three runs (six pairs), not four (eight).
BAND_DERIVATION_ARTIFACT_PATH = (
    "ci/artifacts/gpu-perf-aa-null/ (3 primary runs / six pairs, 2 pods, both A100 device models, "
    "2026-08-30 -- see that directory's own README.md Disclosure section for the two excluded, "
    "contention-contaminated auxiliary runs)"
)


def derive_advisory_band(worst_abs_log_deviation, safety_factor=1.5):
    """Mechanically derive an advisory `(lo, hi)` band from
    `worst_abs_log_deviation` (the largest `|ln(ratio)|` observed over some
    empirical-null evidence set). Extracted here (never inlined only
    at the one call site that produced the committed constant) so
    `ci/scripts/check_aa_null_band.py` can RE-DERIVE the band from the
    `ci/artifacts/gpu-perf-aa-null/` committed evidence and assert equality
    against the constant, rather than trusting prose that the two agree —
    "never hand-tunes the two numbers" is ENFORCED via that gate, not
    merely asserted here.

    PROVENANCE, disclosed plainly: [`PRE_REGISTERED_ADVISORY_BAND`] was
    committed BEFORE this function existed, under prose that said only
    "rounded outward" — which, applied per-edge, yields `(0.75, 1.32)`,
    not the committed `(0.75, 1.33)`. The committed upper edge came from
    flooring the reciprocal of the floored lower edge (`1/0.75 = 1.333…`).
    This function was reverse-engineered AFTER the fact to state, as a
    single checkable rule, a mechanism that (a) reproduces the committed
    literals exactly and (b) is genuinely outward (band-widening) on BOTH
    edges for every valid input. An earlier draft claimed the reciprocal
    alone is always outward; that claim was FALSE (for small spreads the
    floored reciprocal can land a hair INSIDE the raw upper edge), so the
    upper edge is defined as the MAX of the two candidates below.

    Rounding rule: the raw lower edge
    `exp(-safety_factor * worst_abs_log_deviation)` is FLOORED to 2
    decimal places (outward for a lower bound). The upper edge is
    `max(floor2(1 / lo), ceil2(exp(+safety_factor * w)))` — the
    reciprocal-symmetric candidate (which produced the committed `1.33`)
    or the independent per-edge ceiling, whichever is WIDER, so the
    outward guarantee holds by construction rather than by assertion.

    Valid input domain: `worst_abs_log_deviation` must be a finite number
    `> 0`, small enough that the floored lower edge stays positive
    (`worst_abs_log_deviation < ln(100)/safety_factor`); anything else
    raises `ValueError` by name rather than returning a degenerate,
    inverted, or divide-by-zero band.
    """
    w = worst_abs_log_deviation
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
        raise ValueError(
            f"derive_advisory_band: worst_abs_log_deviation must be a finite number, got {w!r}"
        )
    if w <= 0:
        raise ValueError(
            "derive_advisory_band: worst_abs_log_deviation must be > 0 "
            f"(a |ln(ratio)| spread), got {w!r} — a zero/negative spread has no band to derive"
        )
    lo_raw = math.exp(-safety_factor * w)
    hi_raw = math.exp(safety_factor * w)
    lo = math.floor(lo_raw * 100) / 100
    if lo <= 0:
        raise ValueError(
            f"derive_advisory_band: spread {w!r} (x{safety_factor}) floors the lower edge to 0 — "
            "a deviation this large is not band material; it is the broken-leg shape the "
            "OUTSIDE_BAND_FAST verdict exists for"
        )
    hi = max(math.floor((1.0 / lo) * 100) / 100, math.ceil(hi_raw * 100) / 100)
    return lo, hi


def load_leg(raw_dir, name):
    """Read one leg's `<name>.exit`/`<name>.json` pair out of `raw_dir` — the
    SAME `{"outcome": ..., "report": ...}` shape `encode_ab.sh`'s own
    (embedded) `load_leg` builds, extracted here so it is directly testable.
    `outcome` is one of `"OK"`, `"FAIL"`, `"DRY_RUN"`, or `"MISSING"` (no
    `.exit` file at all).
    """
    exit_path = os.path.join(raw_dir, f"{name}.exit")
    out_path = os.path.join(raw_dir, f"{name}.json")
    if not os.path.exists(exit_path):
        return {"outcome": "MISSING", "report": None}
    with open(exit_path, encoding="utf-8") as fh:
        exit_code = fh.read().strip()
    try:
        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError):
        report = None
    if report is not None and report.get("ab_dry_run") is True:
        return {"outcome": "DRY_RUN", "report": None}
    if exit_code != "0" or report is None:
        return {"outcome": "FAIL", "report": None}
    return {"outcome": "OK", "report": report}


def load_mode(raw_dir):
    """Read the `mode` marker (round-3 adversarial audit B2) the producer
    writes into `raw_dir` before any leg runs: `"ab"` (the normal
    parent-vs-PR A/B), `"aa-null"` (the D6 empirical-null instrument —
    BOTH `b`-role legs are ALSO parent-sha clones, no PR exists), or
    `"dry-run"`. `None` when absent (an older producer that predates this
    marker) — [`build_report`] treats an unknown mode conservatively:
    never claims a `b`-role runtime failure is "the PR's own problem"
    without CONFIRMED `ab` mode.
    """
    path = os.path.join(raw_dir, "mode")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        value = fh.read().strip()
    return value if value in ("ab", "aa-null", "dry-run") else None


def load_enforce(raw_dir):
    """Read the `enforce` marker (issue #335's final unit, the enforcement
    flip) the producer writes into `raw_dir` before any leg runs — the SAME
    file-based state-passing convention [`load_mode`] already uses, never an
    env var read directly by this module. `True` only when the file exists
    AND its stripped content is exactly `"1"`; ANY other content, or an
    ABSENT file (an older producer that predates this flip, or a
    hand-crafted fixture that never calls the equivalent of
    `gpu_inference_ab.sh`'s own `printf '1' > "$RAW_DIR/enforce"`), reads
    `False` — the SAFE default (v1's own recording-only posture): a
    missing/garbled marker must never accidentally start gating a run that
    never opted in. [`build_report`] consults this ONLY once a report's
    `status` is already confirmed `"GREEN"` — enforcement never overrides a
    correctness refusal (see this module's own "Enforcement flip" doc).
    """
    path = os.path.join(raw_dir, "enforce")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip() == "1"


def load_enforce_marker_present(raw_dir):
    """`True` iff `raw_dir/enforce` exists at all (regardless of its own
    content) — distinguishing "an OLDER producer that never wrote this
    marker at all" from "a producer that wrote it with an EXPLICIT `'0'`",
    the same shape [`load_mode`] already carries by returning `None` for an
    absent file versus a real value for a present one. [`load_enforce`]
    itself folds BOTH of those shapes into the SAME safe `False` (correct
    for control flow — either shape must stay non-enforcing), so this
    separate probe exists purely for AUDITABILITY: [`build_report`] records
    it on the merged report's own `enforce_marker_present` field, letting a
    reader tell "this producer predates the enforcement flip entirely" apart
    from "this producer supports it and this run simply did not request it"
    without re-deriving anything from `enforce`'s own boolean value.
    """
    return os.path.isfile(os.path.join(raw_dir, "enforce"))


def load_pod_id(raw_dir):
    """Read the `pod_id` marker `gpu_inference_ab.sh` writes into `raw_dir`
    before any leg runs (`${RUNPOD_POD_ID:-$(hostname)}` — see that
    script's own doc) — the STRUCTURAL fix for the class of contamination
    `ci/artifacts/gpu-perf-aa-null/README.md`'s own "Disclosure" section
    documents by hand (two independent producer invocations sharing one
    physical rented pod CONCURRENTLY, silently contending for the same
    GPU): a future committed report now carries its own pod identity
    directly, so that exact class is ADJUDICABLE from the artifact alone
    (two reports naming the same `pod_id` with overlapping
    `recorded_order` windows), never requiring out-of-band operator session
    records the way that disclosure currently must. `None` when the file is
    absent (an older producer that predates this field — every report
    committed under `ci/artifacts/gpu-perf-aa-null/` today is exactly this
    case) AND when the file exists but is EMPTY (a host where
    `RUNPOD_POD_ID` was unset and `hostname` itself failed under the
    producer's `set -u`-no-`-e` shell): both fold to `None` here because a
    fabricated or sentinel value would FALSE-MATCH across two broken hosts
    in the exact same-pod-id contamination check this field exists to
    enable. The two shapes stay distinguishable via
    [`load_pod_id_marker_present`] (recorded on the report as
    `pod_id_marker_present`, the same absent-vs-present split
    [`load_enforce_marker_present`] carries for `enforce`) —
    [`build_report`] folds this `None` into every OK leg's own
    `provenance.pod_id`, never raising or defaulting to a fabricated value.
    """
    path = os.path.join(raw_dir, "pod_id")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        value = fh.read().strip()
    return value or None


def load_pod_id_marker_present(raw_dir):
    """`True` iff `raw_dir/pod_id` exists at all (regardless of content) —
    the same auditability probe [`load_enforce_marker_present`] provides
    for `enforce`, applied to `pod_id` (round-2 delta-audit B5): without
    it, "an older producer that never wrote pod identity" and "a producer
    that wrote the marker but whose capture came back EMPTY (no
    `RUNPOD_POD_ID`, failing `hostname`)" both surface as the SAME
    `provenance.pod_id = null`, and a campaign-wide contamination checker
    cannot tell 'field predates the producer' from 'field failed on this
    host' — the identical absent-vs-explicit collapse the `enforce` marker
    already had fixed one field over.
    """
    return os.path.isfile(os.path.join(raw_dir, "pod_id"))


def load_leg_started_at(raw_dir, name):
    """Read `<name>.started_at` (a plain-text nanosecond-epoch timestamp,
    `date +%s%N` — round-2 adversarial audit F3's order-binding evidence)
    out of `raw_dir`. Returns `(value, unavailable_reason)`: `value` is an
    `int` on success, else `None`; `unavailable_reason` is `None` on
    success, `"missing"` when the file does not exist at all, or a short
    parse-failure string (round-3 adversarial audit B3) when the file
    EXISTS but its content did not parse as a plain integer — e.g. a
    non-GNU `date` binary emitting a different format than `%s%N`
    produces. The producer (`gpu_inference_ab.sh`'s own `run_leg`) writes
    this file BEFORE invoking each leg's binary, in EVERY mode including
    `--dry-run`, so `"missing"` at this call site (only ever reached once
    a leg's OWN outcome is confirmed `OK` — see [`build_report`]) means an
    OLDER producer that predates this feature, never a normal outcome.
    """
    path = os.path.join(raw_dir, f"{name}.started_at")
    if not os.path.isfile(path):
        return None, "missing"
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        return int(raw), None
    except (OSError, ValueError) as exc:
        return None, f"timestamp unparseable (non-GNU date?): {exc}"


def verify_recorded_order(started_at_by_leg):
    """The MACHINE-CHECKED half of the A, B, B, A order binding (round-2
    adversarial audit F3): `gpu_inference_ab.sh`'s own module doc states
    the producer always runs legs in that fixed physical order, and the
    whole first-order drift-cancellation rationale
    (`combined_embed_p50_ratio`'s own doc) depends on that being true — but
    nothing PREVIOUSLY verified it was, beyond trusting that the producer's
    own source code calls `run_leg` in that sequence. This asserts the
    FOUR legs' own RECORDED start timestamps ([`load_leg_started_at`]) are
    non-decreasing in [`LEG_ORDER`]'s declared `a1, b1, b2, a2` sequence.

    `started_at_by_leg` is `{name: (value, unavailable_reason)}` —
    [`load_leg_started_at`]'s own return shape. Returns a list of
    `(kind, message)` tuples (empty when the recorded order is clean AND
    every timestamp parsed): `kind` is `"order"` for a GENUINE order
    violation (a real signal the A,B,B,A premise was not observed — round-3
    adversarial audit B3's own `"differs:"`-shaped precedent: this is a
    real divergence, never neutral) or `"unavailable"` for a missing/
    unparseable timestamp (round-3 adversarial audit B3: this is an
    environment/producer-version gap, NOT itself proof the order was
    violated, so [`build_report`] must never fold it into the SAME
    PR-blame bucket a genuine `"order"` violation earns).
    """
    findings = []
    prev_name, prev_ts = None, None
    for name in LEG_ORDER:
        ts, unavailable_reason = started_at_by_leg.get(name, (None, "missing"))
        if unavailable_reason is not None:
            findings.append(("unavailable", f"leg {name!r}: {unavailable_reason} -- cannot verify run order"))
            continue
        if prev_ts is not None and ts < prev_ts:
            findings.append((
                "order",
                f"recorded run-order violation: {name!r} started at {ts} BEFORE {prev_name!r} started at "
                f"{prev_ts} -- the required A,B,B,A order (a1,b1,b2,a2) was not observed",
            ))
        prev_name, prev_ts = name, ts
    return findings


def gpu_inference_tier(report):
    """Read `tiers.gpu_inference` off a full `jammi-bench gpu-inference-scale`
    report — `{}` when absent (a caller checking presence gets an honest
    empty dict, never a `KeyError`).
    """
    return (report or {}).get("tiers", {}).get("gpu_inference") or {}


def leg_identity(tier):
    """`ab_merge.generic_leg_identity_fields` over
    `identity_fields.GPU_INFERENCE_IDENTITY_FIELDS` — the ONE identity
    comparator this module uses, never a hand-rolled second one.
    """
    return ab_merge.generic_leg_identity_fields(tier, GPU_INFERENCE_IDENTITY_FIELDS)


def identity_violations_across_legs(identity_by_leg):
    """Pairwise-refuse every `OK` leg's identity against a single reference
    leg (the first, in [`LEG_ORDER`], that is present) — reusing
    `ab_merge.generic_leg_premise_violations` per pair, never a hand-rolled
    N-way comparator. Two legs sharing the SAME identity as the reference
    necessarily share it with EACH OTHER too (identity comparison is
    transitive equality on a fixed field set), so one reference is
    sufficient to certify the whole set — this is not weaker than a full
    pairwise sweep, just cheaper.

    Returns a list of violation strings (empty when every present leg
    agrees), each prefixed with the leg pair it came from.
    """
    present = [name for name in LEG_ORDER if name in identity_by_leg]
    if len(present) < 2:
        return []
    reference = present[0]
    violations = []
    for other in present[1:]:
        violations += ab_merge.generic_leg_premise_violations(
            GPU_INFERENCE_IDENTITY_FIELDS,
            identity_by_leg[reference],
            identity_by_leg[other],
            reference,
            other,
        )
    return violations


def is_missing_field_violation(violation_text):
    """`True` iff `violation_text` (one entry of
    `ab_merge.generic_leg_premise_violations`'s own return list) names a
    field MISSING entirely from one side's record — that function's own
    two, and ONLY two, violation shapes are `"...missing from [...] leg's
    record -- cannot verify..."` and `"...differs: a=... b=..."` (see that
    function's own source for both literal f-strings this substring test
    matches against); `False` for the latter (a genuine VALUE divergence).
    round-1 adversarial audit B3's own classifier: [`build_report`] uses
    this to route an all-missing violation set to the neutral
    `INCOMPLETE_IDENTITY`/75 outcome rather than the hard `INVALID`/1
    refusal a real divergence earns.
    """
    return "missing from" in violation_text


def _measurement_value(tier, lane, metric):
    """Read `tier[lane][metric]["value"]` (a `Measurement`'s own JSON shape,
    `report.rs::Measurement`) — raises `KeyError`/`TypeError` on a malformed
    tier rather than silently returning `None`/`0`, since a caller computing
    a ratio must never divide by a silently-substituted placeholder.
    """
    return tier[lane][metric]["value"]


def adjacent_pair_ratio(legs, pair, lane="embed", metric="p50_ms"):
    """The ratio for one [`ADJACENT_PAIRS`] entry: `b / a`, where `a`/`b`
    are decided by ROLE ([`ROLE_OF_LEG`]), never by the pair tuple's own
    member order — pair `("a1", "b1")` runs a-then-b physically, but pair
    `("b2", "a2")` runs b-then-a; both must still read as `b-role /
    a-role`, so this reads correctly regardless of which leg physically ran
    first within the pair.
    """
    roles = {ROLE_OF_LEG[name]: name for name in pair}
    if set(roles) != {"a", "b"}:
        raise ValueError(
            f"pair {pair!r} is not one (a-role, b-role) leg — got roles "
            f"{[ROLE_OF_LEG[n] for n in pair]!r}"
        )
    a_name, b_name = roles["a"], roles["b"]
    a_value = _measurement_value(legs[a_name], lane, metric)
    b_value = _measurement_value(legs[b_name], lane, metric)
    if a_value is None or a_value == 0:
        raise ZeroDivisionError(f"leg {a_name!r}'s {lane}.{metric} is {a_value!r} — cannot ratio")
    return b_value / a_value


def combined_embed_p50_ratio(legs):
    """THE primary endpoint (see this module's own doc): the mean of the two
    [`ADJACENT_PAIRS`] ratios over `embed.p50_ms` — `b / a` (PR / parent)
    orientation, first-order drift-cancelled by the A, B, B, A leg order.
    `legs` must carry every [`LEG_ORDER`] name with a real `gpu_inference`
    tier dict (the caller — [`build_report`] — only calls this once all four
    legs are confirmed `OK`).
    """
    ratios = [adjacent_pair_ratio(legs, pair) for pair in ADJACENT_PAIRS]
    return statistics.mean(ratios), ratios


def classify_advisory(ratio):
    """`"within_band"` when `ratio` falls inside
    [`PRE_REGISTERED_ADVISORY_BAND`], `"outside_band"` otherwise. Deliberately
    NOT spelled `"pass"`/`"fail"` (round-1 adversarial audit advisory): those
    words read as a GATE verdict to a human skimming the table — this
    classification ALONE still never gates: it states the FACT the ratio
    fell outside the pre-registered band, without implying anything was
    refused. Whether that fact ALSO refuses the run is decided entirely by
    `enforce_verdict` in [`build_report`] (the enforcement flip), a separate
    field this function knows nothing about.
    """
    lo, hi = PRE_REGISTERED_ADVISORY_BAND
    return "within_band" if lo <= ratio <= hi else "outside_band"


def lane_measurements(tier, lane):
    """Every RECORDED-never-gated measurement for one lane
    (`rows`/`rows_per_s`/`p50_ms`/`p99_ms`/`deterministic`) — carried through
    to the merged report verbatim for visibility, per this module's own
    "recorded, not gated" doctrine for everything but the primary endpoint.
    """
    block = tier.get(lane) or {}
    return {
        "rows": block.get("rows"),
        "rows_per_s": block.get("rows_per_s"),
        "p50_ms": block.get("p50_ms"),
        "p99_ms": block.get("p99_ms"),
        "deterministic": block.get("deterministic"),
    }


def build_report(raw_dir, a_sha=None, b_sha=None):
    """Load all four [`LEG_ORDER`] legs out of `raw_dir`, refuse (INVALID) on
    an identity mismatch among the `OK` ones, and otherwise assemble the
    merged report + exit code. Returns `(merged_dict, exit_code)`.

    `a_sha`/`b_sha` are the two clones' own git shas (the producer's own
    provenance cross-check already confirmed each binary's baked
    `build_sha` matches its clone's `HEAD`, and that the two differ — see
    `gpu_inference_ab.sh`'s own header) — recorded on the merged report,
    never re-derived here.

    Whether this run is ENFORCING (issue #335's final unit) is read from
    `raw_dir`'s own `enforce` marker ([`load_enforce`]), the SAME
    file-based convention `mode` already uses — never a second parameter
    here, so every existing caller (this module's own `main`,
    `gpu_inference_ab.sh`) keeps working unchanged the moment the producer
    starts writing that file.
    """
    legs_raw = {name: load_leg(raw_dir, name) for name in LEG_ORDER}
    ok_legs = {name: entry for name, entry in legs_raw.items() if entry["outcome"] == "OK"}
    started_at_by_leg = {name: load_leg_started_at(raw_dir, name) for name in LEG_ORDER}
    mode = load_mode(raw_dir)
    enforce = load_enforce(raw_dir)
    enforce_marker_present = load_enforce_marker_present(raw_dir)
    pod_id = load_pod_id(raw_dir)
    pod_id_marker_present = load_pod_id_marker_present(raw_dir)

    legs_out = {name: {"outcome": entry["outcome"]} for name, entry in legs_raw.items()}
    tiers = {}
    identity_by_leg = {}
    for name, entry in ok_legs.items():
        tier = gpu_inference_tier(entry["report"])
        tiers[name] = tier
        legs_out[name]["identity"] = {k: tier.get(k) for k in GPU_INFERENCE_IDENTITY_FIELDS}
        legs_out[name]["provenance"] = {
            k: tier.get(k)
            for k in ("device_name", "kernels_disabled_requested", "flash_compiled", "build_features")
        }
        legs_out[name]["provenance"]["build_sha"] = (entry["report"].get("provenance") or {}).get("build_sha")
        # issue #335 delta-audit F3(d): pod identity, structural fix for the
        # concurrent-invocations-sharing-one-pod contamination class
        # `ci/artifacts/gpu-perf-aa-null/README.md`'s own "Disclosure"
        # section documents by hand -- see `load_pod_id`'s own doc. Recorded
        # PER LEG (mirroring `build_sha`'s own placement) even though all
        # four legs of one invocation share the SAME pod -- this is the
        # "record pod identity into provenance" contract a future consumer
        # (a campaign-wide contamination checker) can rely on without
        # special-casing "read it once at the top" vs "read it per leg".
        legs_out[name]["provenance"]["pod_id"] = pod_id
        legs_out[name]["measurements"] = {
            "embed": lane_measurements(tier, "embed"),
            "infer": lane_measurements(tier, "infer"),
        }
        identity_by_leg[name] = leg_identity(tier)

    base = {
        "schema_version": 1,
        "a_sha": a_sha,
        "b_sha": b_sha,
        "producer": {
            "path": "ci/scripts/perf/gpu_inference_ab.sh",
            "kind": "script",
            "invocation": "ci/scripts/perf/gpu_inference_ab.sh",
            "gating": "none",
        },
        "primary_endpoint": "embed.p50_ms ratio (b / a), mean of the two A,B,B,A adjacent-pair ratios",
        "identity_fields": list(GPU_INFERENCE_IDENTITY_FIELDS),
        "leg_order": list(LEG_ORDER),
        "legs": legs_out,
        # round-2 adversarial audit F3 (round-3 adversarial audit B3 schema
        # fix): the RAW recorded evidence the order binding is checked
        # against, folded into the merged JSON regardless of outcome -- an
        # auditor reading a committed report can see the actual
        # timestamps AND, honestly, why one was unavailable when it was,
        # not just this module's own verdict about them.
        "mode": mode,
        # issue #335's final unit: recorded on EVERY report regardless of
        # status (mirrors `mode`'s own unconditional placement) -- an
        # auditor reading a refused (INVALID/INCOMPLETE/...) report can see
        # whether enforcement was even requested for this run, even though
        # (per this module's own "Enforcement flip" doc) the marker's value
        # never changes the outcome on any non-GREEN status.
        "enforce": enforce,
        # issue #335 delta-audit advisory (4): distinguishes "an older
        # producer that predates the enforcement flip entirely" (`False`)
        # from "this producer supports it and simply was not asked for it"
        # (`True`, `enforce` still `False`) -- see `load_enforce_marker_present`'s
        # own doc; `enforce` alone (a plain bool) cannot carry this
        # distinction, since both shapes fold to the SAME safe `False`.
        "enforce_marker_present": enforce_marker_present,
        # round-2 delta-audit B5: the same absent-vs-explicit split for the
        # pod identity marker — `provenance.pod_id = null` alone cannot
        # distinguish "older producer, field never existed" (`False` here)
        # from "producer wrote the marker but the capture came back empty"
        # (`True` here, `pod_id` still `null`); see
        # `load_pod_id_marker_present`'s own doc.
        "pod_id_marker_present": pod_id_marker_present,
        "recorded_order": {
            name: {"value": value, "unavailable_reason": reason}
            for name, (value, reason) in started_at_by_leg.items()
        },
    }

    missing = [name for name in LEG_ORDER if name not in ok_legs]
    if missing:
        # round-3 adversarial audit B2 (correcting round-2's own F5, which
        # collapsed EVERY b-role absence -- MISSING, DRY_RUN, or a genuine
        # FAIL -- into the SAME "PR's own problem" bucket regardless of
        # whether the PR binary ever even ran, and regardless of whether a
        # PR exists at all under --aa-null): a b-role leg only carries a
        # real "the PR's own problem" SIGNAL when (a) it actually RAN and
        # errored (outcome FAIL, never MISSING/DRY_RUN -- "nothing ran" is
        # not a runtime signal about the PR binary at all) and (b) this run
        # is CONFIRMED `ab` mode (under --aa-null, b-role legs are ALSO
        # parent-sha clones -- no PR exists to blame, matching the
        # PRODUCER's own build-failure routing, which already treats an
        # --aa-null b-role BUILD failure the same as a parent's). An
        # unknown `mode` (an older producer) never escalates to INVALID --
        # this module cannot confirm the ab-mode precondition, so it stays
        # in the neutral bucket.
        failed_b_in_ab_mode = [
            name
            for name in missing
            if ROLE_OF_LEG[name] == "b" and legs_raw[name]["outcome"] == "FAIL" and mode == "ab"
        ]
        if failed_b_in_ab_mode:
            base["status"] = "INVALID"
            base["leg_premise_violations"] = []
            base["missing_legs"] = missing
            base["invalid_reason"] = (
                f"PR-side leg(s) {failed_b_in_ab_mode} RAN but did not produce an OK report (mode=ab, "
                f"confirmed a real PR exists) -- a runtime failure on an already-built PR binary is a "
                f"stronger signal than a non-compiling one; a real correctness-of-measurement refusal, "
                f"never neutral"
            )
            return base, 1
        base["status"] = "INCOMPLETE"
        base["leg_premise_violations"] = []
        base["missing_legs"] = missing
        base["incomplete_reason"] = (
            f"leg(s) {missing} did not produce an OK report -- nothing ran for at least one leg "
            f"(a MISSING/DRY_RUN outcome), or the leg that failed is parent-side or this run has no "
            f"confirmed PR to blame (mode={mode!r}); neutral, never a proven runtime signal about a PR"
        )
        return base, 75

    order_findings = verify_recorded_order(started_at_by_leg)
    if order_findings:
        # round-3 adversarial audit B3: a GENUINE order violation ("order")
        # is a real signal and wins outright (the SAME "a stronger signal
        # dominates" precedent the identity-violation classification below
        # already sets) -- but an "unavailable" (missing/unparseable)
        # timestamp is an environment/producer-version gap, NEVER itself
        # proof the order was violated, so it must land in the SAME neutral
        # bucket a "nothing to compare" leg does, not the PR-blame one.
        order_violations = [msg for kind, msg in order_findings if kind == "order"]
        if order_violations:
            base["status"] = "INVALID"
            base["leg_premise_violations"] = order_violations
            return base, 1
        unavailable = [msg for kind, msg in order_findings if kind == "unavailable"]
        base["status"] = "INCOMPLETE_ORDER"
        base["leg_premise_violations"] = unavailable
        base["incomplete_order_reason"] = (
            "one or more legs' recorded start timestamps could not be read (missing file or "
            "unparseable content, e.g. a non-GNU `date` binary) -- cannot verify the A,B,B,A order "
            "was observed, but this is not itself proof it was not; neutral, never blamed on the PR"
        )
        return base, 75

    violations = identity_violations_across_legs(identity_by_leg)
    if violations:
        # round-1 adversarial audit B3 (round-2 adversarial audit advisory
        # correction: the ORIGINAL wording here claimed "never present-but-
        # null on a leg that DOES emit it" -- that was FALSE. `ab_merge.
        # generic_leg_identity_fields` folds BOTH a genuinely absent key
        # AND a present-but-JSON-`null` value into the SAME `_MISSING`
        # sentinel for every `GPU_INFERENCE_IDENTITY_FIELDS` member (none
        # of them are `null_is_value_fields`), so a "missing from" violation
        # here honestly covers EITHER shape, not only a key's outright
        # absence): a violation whose text names a field this way is not
        # the same claim as a proven premise MISMATCH -- it is "this
        # comparator cannot even ask the question", the honest shape a
        # parent leg built before issue #335's own identity contract landed
        # produces (or, less commonly, a leg that emits the key but with an
        # explicit `null` value). A structured `(field, kind, sides)`
        # violation record that distinguished "absent key" from "present-
        # null value" mechanically, rather than folding both into one
        # prose-matched string, is a natural follow-on residual -- NOT
        # built now. A single genuine VALUE divergence anywhere in the set
        # still promotes the WHOLE refusal to INVALID/1 -- `all(...)` below
        # is FALSE the instant one "differs:"-shaped violation exists among
        # possibly-several "missing from"-shaped ones.
        if all(is_missing_field_violation(v) for v in violations):
            base["status"] = "INCOMPLETE_IDENTITY"
            base["leg_premise_violations"] = violations
            base["incomplete_identity_reason"] = (
                "every identity disagreement among the OK legs is a field MISSING entirely from one "
                "side's record, never a differing VALUE -- likely a parent leg built before issue #335's "
                "own identity contract landed; neutral, not a proven premise mismatch"
            )
            return base, 75
        base["status"] = "INVALID"
        base["leg_premise_violations"] = violations
        return base, 1

    # round-1 adversarial audit advisory: an identity-clean leg set can
    # STILL carry a malformed measurement (a lane/metric key genuinely
    # absent, a non-numeric `p50_ms.value`, a zero baseline) --
    # `_measurement_value`/`adjacent_pair_ratio` deliberately raise rather
    # than silently substitute a placeholder (see their own doc), so this
    # is the ONE place that catches those exceptions and turns them into a
    # TYPED refusal with a report still WRITTEN (never an uncaught
    # traceback that crashes this script with NO report at all).
    try:
        ratio, pair_ratios = combined_embed_p50_ratio(tiers)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        base["status"] = "INVALID_MEASUREMENT"
        base["leg_premise_violations"] = []
        base["invalid_measurement_reason"] = (
            f"the four legs' premises agree, but computing the primary endpoint's ratio raised "
            f"{type(exc).__name__}: {exc} -- a real correctness-of-measurement defect (a malformed "
            f"lane/metric, a zero baseline), never a perf regression itself"
        )
        return base, 1

    advisory = classify_advisory(ratio)
    base["status"] = "GREEN"
    base["leg_premise_violations"] = []
    base["combined_embed_p50_ratio"] = ratio
    base["adjacent_pair_ratios"] = {f"{a}/{b}": r for (a, b), r in zip(ADJACENT_PAIRS, pair_ratios)}
    base["advisory"] = {
        "band_not_pre_registered": False,
        "band": list(PRE_REGISTERED_ADVISORY_BAND),
        "band_derivation": BAND_DERIVATION_ARTIFACT_PATH,
        "classification": advisory,
    }

    # issue #335's final unit: the ONE place `enforce` is ever consulted —
    # only reached once `status` is already confirmed "GREEN" above, so
    # enforcement can never override a correctness refusal (see this
    # module's own "Enforcement flip" doc). `status` itself is NEVER
    # changed by ANY branch below (it stays "GREEN" -- the premises really
    # did agree) -- every enforcement verdict lives entirely on its own
    # `enforce_verdict`/`*_reason` fields, deliberately never folded into
    # `status`, so a report reader can always distinguish a correctness
    # refusal from an enforcement one without parsing prose.
    if not enforce:
        base["enforce_verdict"] = "NOT_ENFORCED"
        return base, 0

    # round-4 delta-audit F4: enforcement is defined ONLY for a real A/B run
    # -- checked BEFORE the band is even consulted, so this refuses
    # regardless of the ratio's own value, and NEVER silently downgrades to
    # NOT_ENFORCED (which would swallow an operator's explicit enforcement
    # request without any record it was ignored). See this module's own
    # "Enforcement flip" doc for the full rationale.
    if mode != "ab":
        base["enforce_verdict"] = "ENFORCE_INVALID_MODE"
        base["enforce_invalid_mode_reason"] = (
            f"GPU_INFERENCE_AB_ENFORCE=1 was requested, but this run's mode is {mode!r}, not 'ab' -- "
            f"enforcement is only defined for a real parent-vs-PR A/B run: under --aa-null both b-role "
            f"legs are ALSO parent-sha clones (no PR exists to enforce against -- the null band itself "
            f"was derived from exactly this shape of run, so enforcing it here would misfire the very "
            f"instrument the band comes from), and an unconfirmed/absent mode means this module cannot "
            f"even confirm which shape this run is. Refusing rather than silently treating this as "
            f"NOT_ENFORCED, which would swallow the operator's explicit enforcement request with no "
            f"record it was ignored."
        )
        return base, 1

    if advisory == "within_band":
        base["enforce_verdict"] = "PASS"
        return base, 0

    # round-4 delta-audit F1: this is a NULL band (derived from a
    # parent-vs-parent A/A campaign, never a "faster is always fine" one) --
    # BOTH directions outside it fail closed (exit 1), but the VERDICT and
    # REASON are direction-honest, never a one-sided "regression" label
    # applied to a ratio that could just as easily mean the b-role leg
    # measured suspiciously, unexplainably FAST (a silently short-circuited/
    # broken leg -- cf. this repo's own torch-cu130 CPU-fallback escape:
    # "too fast" is the signature of a leg not doing the work, not proof of
    # a real speedup). See this module's own "Enforcement flip" doc's own
    # two named arms.
    lo, hi = PRE_REGISTERED_ADVISORY_BAND
    if ratio > hi:
        base["enforce_verdict"] = "PERF_REGRESSION"
        base["perf_regression_reason"] = (
            f"ENFORCING mode (GPU_INFERENCE_AB_ENFORCE=1, mode=ab): the four legs' premises are GREEN, "
            f"but combined_embed_p50_ratio={ratio:.6f} falls ABOVE the pre-registered advisory band's "
            f"upper edge ({hi}, band={list(PRE_REGISTERED_ADVISORY_BAND)}, derived from "
            f"{BAND_DERIVATION_ARTIFACT_PATH}) -- a real, unambiguous signal the b-role (PR) leg measured "
            f"SLOWER than the null distribution can explain, refused separately from every "
            f"correctness-refusal reason above; status stays GREEN because the premises themselves were fine"
        )
        return base, 1

    base["enforce_verdict"] = "OUTSIDE_BAND_FAST"
    base["outside_band_fast_reason"] = (
        f"ENFORCING mode (GPU_INFERENCE_AB_ENFORCE=1, mode=ab): the four legs' premises are GREEN, but "
        f"combined_embed_p50_ratio={ratio:.6f} falls BELOW the pre-registered advisory band's lower edge "
        f"({lo}, band={list(PRE_REGISTERED_ADVISORY_BAND)}, derived from {BAND_DERIVATION_ARTIFACT_PATH}) "
        f"-- this is AMBIGUOUS, not a favorable result by default: it means the b-role (PR) leg measured "
        f"FASTER than the null distribution can explain, which is EITHER a genuine large improvement OR a "
        f"leg that silently short-circuited/broke and finished suspiciously fast (the same signature this "
        f"repo's own torch-cu130 CPU-fallback escape carries -- 'too fast' is what a leg not actually "
        f"doing the work looks like). This module cannot distinguish the two from the ratio alone and "
        f"never guesses; refusing (exit 1, fail closed) and naming the ambiguity for a human to adjudicate, "
        f"never silently accepted as a win"
    )
    return base, 1


def render_table(merged):
    """A short, greppable plain-text table naming the ratio, the primary
    endpoint, and the advisory classification — printed to stdout alongside
    the merged JSON, mirroring `finetune_run_ab.sh`'s own
    report-json-plus-table pair.
    """
    lines = [
        f"status={merged['status']}",
        f"primary_endpoint={merged['primary_endpoint']}",
    ]
    if merged["status"] == "GREEN":
        # round-1 adversarial audit advisory: "GREEN" alone reads as a
        # gate-shaped verdict to a skimming human -- this line states what
        # it actually certifies (the four legs' PREMISES agree) and what it
        # explicitly does NOT (anything about the measured ratio's own
        # magnitude, which is recorded below regardless of its value).
        lines.append("premises=GREEN (validity only -- a separate enforce_verdict line below states the perf verdict)")
        lines.append(f"combined_embed_p50_ratio={merged['combined_embed_p50_ratio']:.6f} (b/a, PR/parent)")
        for pair_label, r in merged["adjacent_pair_ratios"].items():
            lines.append(f"  pair {pair_label}: ratio={r:.6f}")
        adv = merged["advisory"]
        lines.append(
            f"advisory={adv['classification']} band={adv['band']} "
            f"(PRE-REGISTERED — derived from {adv['band_derivation']}, see that directory's own README.md)"
        )
        enforce_verdict = merged.get("enforce_verdict")
        if enforce_verdict == "NOT_ENFORCED":
            lines.append("enforce_verdict=NOT_ENFORCED (recording-only; set GPU_INFERENCE_AB_ENFORCE=1 to gate on the band above)")
        elif enforce_verdict == "ENFORCE_INVALID_MODE":
            lines.append("enforce_verdict=ENFORCE_INVALID_MODE (enforcement requested but mode != 'ab'; exit 1, refused before the band was even consulted)")
            lines.append(f"enforce_invalid_mode_reason={merged['enforce_invalid_mode_reason']}")
        elif enforce_verdict == "PASS":
            lines.append("enforce_verdict=PASS (enforcing mode; ratio landed inside the pre-registered band)")
        elif enforce_verdict == "PERF_REGRESSION":
            lines.append("enforce_verdict=PERF_REGRESSION (enforcing mode; ratio ABOVE the upper edge, a real slowdown signal, exit 1)")
            lines.append(f"perf_regression_reason={merged['perf_regression_reason']}")
        elif enforce_verdict == "OUTSIDE_BAND_FAST":
            lines.append("enforce_verdict=OUTSIDE_BAND_FAST (enforcing mode; ratio BELOW the lower edge, AMBIGUOUS -- improvement or a broken leg -- exit 1)")
            lines.append(f"outside_band_fast_reason={merged['outside_band_fast_reason']}")
    elif merged["status"] == "INVALID":
        # round-2 adversarial audit F5 (round-3 adversarial audit B2/B3
        # refinement): the b-role-FAIL-in-confirmed-ab-mode and genuine
        # order-violation refusals both fold their evidence into this SAME
        # status; `invalid_reason` (the b-role case) is a single-string
        # summary, `leg_premise_violations` (the identity-mismatch and
        # order-violation cases) is a list -- print whichever is present,
        # never silently drop the b-role case's own reason just because
        # its own `leg_premise_violations` list happens to be empty.
        if merged.get("invalid_reason"):
            lines.append(f"reason={merged['invalid_reason']}")
            lines.append(f"missing_legs={merged.get('missing_legs')}")
        if merged["leg_premise_violations"]:
            lines.append("leg_premise_violations:")
            for v in merged["leg_premise_violations"]:
                lines.append(f"  - {v}")
    elif merged["status"] == "INCOMPLETE":
        lines.append(f"missing_legs={merged['missing_legs']} mode={merged.get('mode')!r}")
        if merged.get("incomplete_reason"):
            lines.append(f"reason={merged['incomplete_reason']}")
    elif merged["status"] == "INCOMPLETE_IDENTITY":
        lines.append(f"reason={merged['incomplete_identity_reason']}")
        lines.append("leg_premise_violations (all missing-field, no genuine value divergence):")
        for v in merged["leg_premise_violations"]:
            lines.append(f"  - {v}")
    elif merged["status"] == "INCOMPLETE_ORDER":
        lines.append(f"reason={merged['incomplete_order_reason']}")
        lines.append("leg_premise_violations (timestamp unavailable, not a proven order violation):")
        for v in merged["leg_premise_violations"]:
            lines.append(f"  - {v}")
    elif merged["status"] == "INVALID_MEASUREMENT":
        lines.append(f"reason={merged['invalid_measurement_reason']}")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(
            "usage: gpu_inference_ab.py RAW_DIR OUT_DIR [A_SHA] [B_SHA]",
            file=sys.stderr,
        )
        return 2
    raw_dir, out_dir = argv[0], argv[1]
    a_sha = argv[2] if len(argv) > 2 else None
    b_sha = argv[3] if len(argv) > 3 else None

    merged, exit_code = build_report(raw_dir, a_sha=a_sha, b_sha=b_sha)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gpu_inference_ab_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)

    table = render_table(merged)
    table_path = os.path.join(out_dir, "gpu_inference_ab_table.txt")
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")

    print(f"=== merged report: {out_path} ===")
    print(table)
    print(f"=== exit={exit_code} ===")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
