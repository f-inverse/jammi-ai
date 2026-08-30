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

## v1 is RECORDING-ONLY — never a gate

Issue #335's own exit criterion forbids enforcement before multi-pod/
both-device-model validation. This module NEVER fails a run over the
measured *ratio* — the only thing it refuses (nonzero exit, status
`INVALID`) is a PREMISE mismatch (the two legs did not measure the same
thing at all, so no ratio is even meaningful). A measured ratio, however
large, is always recorded and printed, never gated.

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

## ADVISORY classification: a placeholder band, NOT YET PRE-REGISTERED

[`PLACEHOLDER_ADVISORY_BAND`] below is what this module classifies
`combined_embed_p50_ratio` against for the `advisory` column in the printed
table — but it is EXPLICITLY NOT a pre-registered decision rule (D6): no
empirical A/A null distribution exists yet to derive a real band from. The
`--aa-null` producer mode (`gpu_inference_ab.sh`) is the instrument that will
produce one (comparing the SAME parent sha built twice, from two independent
clones, against itself) — its output is shaped for committing under
`ci/artifacts/gpu-perf-aa-null/`. Until that artifact exists and a real band
is derived from it, `advisory` is observability only: it NEVER changes this
module's exit code, and a future unit that wires this endpoint into an
actual gate must replace [`PLACEHOLDER_ADVISORY_BAND`] with a band derived
from that artifact, never keep this placeholder as the real threshold.

## Exit codes (round-2 adversarial audit F5's final reconciled lattice —
## matches `gpu_inference_ab.sh`'s own header table and `gpu-perf-ab.yml`'s
## own step annotations verbatim)

  * `0`  — GREEN: all four legs are `OK`, their RECORDED run order verifies
           as A,B,B,A ([`verify_recorded_order`]), and they agree on
           [`identity_fields.GPU_INFERENCE_IDENTITY_FIELDS`]. The ratio is
           computed and printed regardless of its own value (recording-only
           — see above).
  * `1`  — a real correctness-of-measurement refusal, always the PR's own
           problem or a genuine measurement-validity defect, never the
           parent's, in FOUR shapes: (a) INVALID (identity) — at least two
           `OK` legs DIFFER on a declared identity field's actual VALUE;
           (b) INVALID_MEASUREMENT — the legs' premises agree, but
           computing the primary endpoint's ratio itself raised (a
           malformed lane/metric, a zero baseline —
           [`_measurement_value`]/[`adjacent_pair_ratio`] deliberately
           raise rather than silently substitute a placeholder); (c) a
           PR-side (`b`-role) leg (`b1`/`b2`) did not produce an `OK`
           report (round-2 adversarial audit F5: a RUNTIME failure on an
           already-built PR binary is a stronger signal than a
           non-compiling one — the shell producer's own build-failure exit
           is a SEPARATE, earlier gate; this is the comparator's own
           classification of a leg that ran but errored/crashed); (d) the
           four legs' RECORDED start order does not verify as A,B,B,A
           ([`verify_recorded_order`]) — the drift-cancellation rationale
           this comparator's own primary endpoint depends on was never
           actually observed. All four refuse loudly (never silently drop
           the offending leg/field and compute a ratio anyway), and all
           four still WRITE a report (never an uncaught traceback with
           nothing recorded at all).
  * `75` — neutral "nothing to compare safely", never a sign of a code
           problem, in TWO distinct shapes:
             - `INCOMPLETE`: fewer than all four legs produced an `OK`
               report, and NONE of the missing/failed ones are `b`-role
               (a PARENT-side (`a`-role) build/runtime failure, a capacity
               miss, a `DRY_RUN` stub — see (c) above for the `b`-role
               case, which is `1`, not `75`).
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
"""

from __future__ import annotations

import json
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

# NOT PRE-REGISTERED (D6) — see this module's own "ADVISORY classification"
# doc section above. A ratio inside this half-open-on-neither-side interval
# classifies `advisory=pass`; outside it, `advisory=fail`. Neither outcome
# ever changes this module's exit code in v1.
PLACEHOLDER_ADVISORY_BAND = (0.90, 1.10)


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


def load_leg_started_at(raw_dir, name):
    """Read `<name>.started_at` (a plain-text nanosecond-epoch timestamp,
    `date +%s%N` — round-2 adversarial audit F3's order-binding evidence)
    out of `raw_dir`. `None` when absent or unparsable — the producer
    (`gpu_inference_ab.sh`'s own `run_leg`) writes this file BEFORE
    invoking each leg's binary, in EVERY mode including `--dry-run`, so a
    genuinely absent file means an OLDER producer that predates this
    feature, never a normal outcome for a leg this comparator otherwise
    reads as `OK`.
    """
    path = os.path.join(raw_dir, f"{name}.started_at")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


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

    Returns a list of violation strings (empty when the recorded order is
    clean) — a leg with NO recorded timestamp is itself a violation (an
    older producer, or a hand-crafted fixture, that cannot state this
    premise at all).
    """
    violations = []
    prev_name, prev_ts = None, None
    for name in LEG_ORDER:
        ts = started_at_by_leg.get(name)
        if ts is None:
            violations.append(f"leg {name!r} has no recorded start timestamp -- cannot verify run order")
            continue
        if prev_ts is not None and ts < prev_ts:
            violations.append(
                f"recorded run-order violation: {name!r} started at {ts} BEFORE {prev_name!r} started at "
                f"{prev_ts} -- the required A,B,B,A order (a1,b1,b2,a2) was not observed"
            )
        prev_name, prev_ts = name, ts
    return violations


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
    """`"within_placeholder_band"` when `ratio` falls inside
    [`PLACEHOLDER_ADVISORY_BAND`], `"outside_placeholder_band"` otherwise —
    NEVER gates (see this module's own doc); purely a printed/recorded
    classification. Deliberately NOT spelled `"pass"`/`"fail"` (round-1
    adversarial audit advisory): those words read as a GATE verdict to a
    human skimming the table, which this v1 recording-only instrument is
    not — `"outside_placeholder_band"` states the FACT (the ratio fell
    outside a band that is not even pre-registered yet) without implying
    anything failed.
    """
    lo, hi = PLACEHOLDER_ADVISORY_BAND
    return "within_placeholder_band" if lo <= ratio <= hi else "outside_placeholder_band"


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
    """
    legs_raw = {name: load_leg(raw_dir, name) for name in LEG_ORDER}
    ok_legs = {name: entry for name, entry in legs_raw.items() if entry["outcome"] == "OK"}
    started_at_by_leg = {name: load_leg_started_at(raw_dir, name) for name in LEG_ORDER}

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
        # round-2 adversarial audit F3: the RAW recorded evidence the order
        # binding is checked against, folded into the merged JSON
        # regardless of outcome -- an auditor reading a committed report
        # can see the actual timestamps, not just this module's own verdict
        # about them.
        "recorded_order": {name: started_at_by_leg.get(name) for name in LEG_ORDER},
    }

    missing = [name for name in LEG_ORDER if name not in ok_legs]
    if missing:
        # round-2 adversarial audit F5: a PR-side (`b`-role) leg that ran
        # but did not produce an OK report is a STRONGER signal than a
        # non-compiling one (the binary built fine; something in the
        # MEASURED serve itself crashed/errored) -- the PR's own problem, a
        # real correctness-of-measurement refusal (1), never the neutral
        # "nothing to compare" bucket a PARENT-side (`a`-role) runtime
        # failure still falls into (75, the SAME bucket a parent BUILD
        # failure already occupies -- see gpu_inference_ab.sh's own exit
        # table). A single b-role failure dominates even alongside an
        # ALSO-missing a-role leg, the same "a stronger signal wins"
        # precedent B3's identity-violation classification already sets.
        missing_b = [name for name in missing if ROLE_OF_LEG[name] == "b"]
        if missing_b:
            base["status"] = "INVALID"
            base["leg_premise_violations"] = []
            base["missing_legs"] = missing
            base["invalid_reason"] = (
                f"PR-side leg(s) {missing_b} did not produce an OK report -- a runtime failure on an "
                f"already-built PR binary is a stronger signal than a non-compiling one; a real "
                f"correctness-of-measurement refusal, never neutral"
            )
            return base, 1
        base["status"] = "INCOMPLETE"
        base["leg_premise_violations"] = []
        base["missing_legs"] = missing
        return base, 75

    order_violations = verify_recorded_order(started_at_by_leg)
    if order_violations:
        base["status"] = "INVALID"
        base["leg_premise_violations"] = order_violations
        return base, 1

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
        "band_not_pre_registered": True,
        "band": list(PLACEHOLDER_ADVISORY_BAND),
        "classification": advisory,
    }
    return base, 0


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
        lines.append("premises=GREEN (validity only -- not a perf verdict; v1 is recording-only)")
        lines.append(f"combined_embed_p50_ratio={merged['combined_embed_p50_ratio']:.6f} (b/a, PR/parent)")
        for pair_label, r in merged["adjacent_pair_ratios"].items():
            lines.append(f"  pair {pair_label}: ratio={r:.6f}")
        adv = merged["advisory"]
        lines.append(
            f"advisory={adv['classification']} band={adv['band']} "
            f"(NOT PRE-REGISTERED — placeholder until the --aa-null A/A artifact lands; "
            f"observability only, never a gate verdict)"
        )
    elif merged["status"] == "INVALID":
        # round-2 adversarial audit F5: the b-role-missing-leg and
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
        lines.append(f"missing_legs={merged['missing_legs']}")
    elif merged["status"] == "INCOMPLETE_IDENTITY":
        lines.append(f"reason={merged['incomplete_identity_reason']}")
        lines.append("leg_premise_violations (all missing-field, no genuine value divergence):")
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
