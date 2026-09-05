#!/usr/bin/env python3
"""Merge stage for `ci/scripts/perf/lora_bias_ab.sh`'s #428 P2b sweep --
turns the raw per-leg reports + `manifest.json` that driver writes into one
pre-registered ACTIVATE/NEUTRAL/REGRESSION/INVALID verdict per model, plus a
`crates/jammi-kernels/artifacts/cuda-runs/` schema-v1 artifact.

Pure Python 3 stdlib (plus this directory's own `ab_merge`/`identity_fields`
modules, themselves stdlib-only) -- no third-party dependency.

## Why this is a NEW merger, not another `ab_merge.py` mode

`ab_merge.py`'s `finetune-run` mode computes a DIFFERENT quantity
(`train_run_wall_s` as one coarse per-run TOTAL, sign-tested across many
seeds against a torch/alloff reference) and its own dispatch-proof lattice
assumes the `fused`-vs-`alloff` CASCADE shape (`CASCADE_BASES`,
`REQUIRED_PAIRS`) -- neither fits this contract's per-step DIFFERENCING
design (`(wall_600 - wall_100) / 500`, `report.rs`'s own
`FinetuneRunTier::train_run_wall_s` doc) or its single-op
(`lora_linear_fused`) proof shape. What IS reused, directly imported rather
than re-derived: `identity_fields.FINETUNE_RUN_IDENTITY_FIELDS` /
`FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS` (the SAME 33-field identity tuple
`report.rs`'s own `FinetuneRunTier::IDENTITY_FIELDS` const declares) and
`ab_merge.generic_leg_identity_fields` / `generic_leg_premise_violations`
(the same generic premise-refusal core `encode_ab.sh`'s/`finetune_run_ab.sh`'s
own merge steps already build on) -- one canonicalizer table, one
identity-comparison core, every jammi-vs-jammi and jammi-vs-torch comparator
in this directory shares.

## Cell model

A **measurement cell** is one `(model, shape)` pair at one of the two
activating shapes (`b8W512f32` "wire", `b32W64bf16` "chapter"), covering the
`fused`/`lora_eager` arms. A **control series** is `(model,)` alone,
covering the `control` arm at the wire shape (a fused-vs-fused repeated
measurement -- the noise floor). CONTRACT's own "an INVALID leg voids its
(model, shape) cell, never the whole file silently" is read LITERALLY and
CONSERVATIVELY here: any single leg violation (a dispatch-proof failure, a
non-finite/non-positive wall, an asymmetric `JAMMI_KERNELS_DISABLE`, or an
identity mismatch against its own `(model, shape, steps)` group) voids the
WHOLE cell/series that leg belongs to -- never a partial, repeat-level
salvage of an otherwise-contaminated cell. A cell/series with zero complete
`(N, M)` repeat pairs (e.g. an operator's `LORA_BIAS_AB_LEGS_ONLY` filter
left a gap) is ALSO invalid, even with zero leg-level violations -- there is
simply no datum to report.

## Verdict (pre-registered, per model)

`floor = |fused_median_wire - control_median| / fused_median_wire` -- ONE
number per model (the wire shape's own fused-arm median against the control
series' median), reused as the floor for BOTH shapes' verdicts. Undefined
(model verdict INVALID) unless BOTH the wire measurement cell's `fused` arm
AND the control series are themselves valid.

Priority (checked in this order, matching `docs/plans/428-lora-bias/
plan-428-p2b.md`'s v1-amendment "bar met -> guide row; missed-but-not-slower
-> keep fusion ...; slower on any leg -> ... and file" cascade):
  1. **ACTIVATE** -- ANY valid activating shape has `gain >= 0.05 and
     gain > floor` (`gain = 1 - fused_median / eager_median`).
  2. **REGRESSION** -- ANY valid activating shape has `gain < -floor`
     (fused slower than eager by more than the floor).
  3. **NEUTRAL** -- every activating shape is valid, none activated or
     regressed (a claim about EVERY shape, so it requires full coverage).
  4. **INVALID** -- floor undefined, OR neither ACTIVATE nor REGRESSION
     fired and at least one activating shape's own cell is invalid (not
     enough coverage to honestly assert NEUTRAL either).

Run: `python3 ci/scripts/perf/lora_bias_ab_merge.py RAW_DIR OUT.json`
(`RAW_DIR` is the sweep's own `$OUT_DIR` -- the directory holding
`manifest.json` and `raw/*.json`, exactly what `lora_bias_ab.sh` writes).
Hermetic self-test: `python3 ci/scripts/perf/test_lora_bias_ab_merge.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_merge import generic_leg_identity_fields, generic_leg_premise_violations  # noqa: E402
from identity_fields import (  # noqa: E402
    FINETUNE_RUN_IDENTITY_FIELDS,
    FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS,
)

SCHEMA_VERSION = 1
LORA_OP = "lora_linear_fused"
WIRE_SHAPE = "b8W512f32"
CHAPTER_SHAPE = "b32W64bf16"
ACTIVATING_SHAPES = (WIRE_SHAPE, CHAPTER_SHAPE)
CONTROL_SHAPE = WIRE_SHAPE
MODELS = ("bert", "distilbert")
STEPS_N = 100
STEPS_M = 600
GAIN_BAR = 0.05


def _is_finite_real(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_manifest(raw_dir):
    path = os.path.join(raw_dir, "manifest.json")
    if not os.path.isfile(path):
        raise SystemExit(f"::error::lora_bias_ab_merge: no manifest.json under {raw_dir!r}")
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"::error::lora_bias_ab_merge: {path} is not a JSON array")
    return rows


def load_tier(raw_dir, row):
    """Returns `(tier_dict, None)` or `(None, error_string)` -- NEVER raises:
    a report that fails to load/parse is a leg-level violation, not a
    merge-fatal error (`finetune_run_ab.sh`'s own "a missed bar is data"
    doctrine, applied at the leg granularity)."""
    report_path = row.get("report_path")
    if not report_path:
        return None, f"{row.get('leg_id')}: manifest row has no report_path"
    full = os.path.join(raw_dir, report_path)
    try:
        with open(full) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{row.get('leg_id')}: could not load/parse {report_path}: {e}"
    tiers = data.get("tiers")
    tier = tiers.get("finetune_run") if isinstance(tiers, dict) else None
    if not isinstance(tier, dict):
        return None, f"{row.get('leg_id')}: {report_path} has no tiers.finetune_run object"
    return tier, None


def wall_violations(leg_id, tier):
    wall = tier.get("train_run_wall_s")
    if not _is_finite_real(wall) or wall <= 0:
        return [f"{leg_id}: train_run_wall_s is not finite and > 0 (got {wall!r})"]
    return []


def dispatch_violations(row, tier):
    """The single-op (`lora_linear_fused`) dispatch proof (CONTRACT):
      * `fused`/`control` legs: `fused > 0`, `eager == 0`, and
        `lora_linear_fused` NOT in `kernels_disabled_requested` (the fused
        arm makes no disable claim at all -- see `finetune_run.rs`'s own
        `Arm::Fused` doc).
      * `lora_eager` legs: `fused == 0`, `eager > 0`, and
        `lora_linear_fused` present in BOTH `kernels_disabled_requested`
        AND `kernels_disabled_fired` (disable wins over Strict --
        `crate::admission::admit_inner`'s own doc -- so a genuine eager leg
        must show the op both requested-disabled and actually fired).
    Plus the EXTRA_DISABLE symmetry check every leg carries regardless of
    arm: `kernels_disabled_requested` minus `{lora_linear_fused}` must equal
    the leg's OWN recorded `extra_disable` set -- an asymmetric
    `JAMMI_KERNELS_DISABLE` (extra entries on one arm only) would silently
    change which op each arm's numbers actually describe.
    """
    leg_id = row.get("leg_id")
    arm = row.get("arm")
    fused = tier.get("lora_linear_fused_dispatches")
    eager = tier.get("lora_linear_eager_dispatches")
    requested = tier.get("kernels_disabled_requested")
    fired = tier.get("kernels_disabled_fired")

    v = []
    if not isinstance(requested, list):
        v.append(f"{leg_id}: kernels_disabled_requested missing or not a list")
        requested = []
    if not isinstance(fired, list):
        v.append(f"{leg_id}: kernels_disabled_fired missing or not a list")
        fired = []
    if not isinstance(fused, (int, float)) or isinstance(fused, bool) or not isinstance(
        eager, (int, float)
    ) or isinstance(eager, bool):
        v.append(f"{leg_id}: lora_linear_{{fused,eager}}_dispatches missing or not numeric")
        return v

    if arm in ("fused", "control"):
        if not (fused > 0 and eager == 0 and LORA_OP not in requested):
            v.append(
                f"{leg_id}: fused/control dispatch proof failed "
                f"(fused={fused}, eager={eager}, {LORA_OP!r} in requested={LORA_OP in requested})"
            )
    elif arm == "lora_eager":
        if not (fused == 0 and eager > 0 and LORA_OP in requested and LORA_OP in fired):
            v.append(
                f"{leg_id}: lora_eager dispatch proof failed "
                f"(fused={fused}, eager={eager}, requested={requested!r}, fired={fired!r})"
            )
    else:
        v.append(f"{leg_id}: unrecognized arm label {arm!r}")

    extra_recorded = set(row.get("extra_disable") or [])
    requested_minus_op = set(requested) - {LORA_OP}
    if requested_minus_op != extra_recorded:
        v.append(
            f"{leg_id}: kernels_disabled_requested minus {{{LORA_OP}}} = "
            f"{sorted(requested_minus_op)!r} != this leg's own recorded extra_disable "
            f"{sorted(extra_recorded)!r} (asymmetric JAMMI_KERNELS_DISABLE)"
        )
    return v


def build_legs(raw_dir, rows):
    """One record per manifest row: `{"row", "tier", "violations"}` --
    `violations` empty means this leg individually passed every leg-level
    check (status, envelope load, wall, dispatch proof, extra-disable
    symmetry). Identity cross-checking (a GROUP property, not a per-leg
    one) is applied afterward by `apply_identity_checks`.
    """
    legs = []
    for row in rows:
        violations = []
        tier = None
        if row.get("status") != "ok":
            violations.append(f"{row.get('leg_id')}: {row.get('reason') or 'manifest row status != ok'}")
        else:
            tier, err = load_tier(raw_dir, row)
            if err:
                violations.append(err)
        if tier is not None:
            violations.extend(wall_violations(row.get("leg_id"), tier))
            violations.extend(dispatch_violations(row, tier))
        legs.append({"row": row, "tier": tier, "violations": violations})
    return legs


def apply_identity_checks(legs):
    """Within each `(model, shape, steps)` group (spanning EVERY arm that
    shares that shape+steps, including `control` at the wire shape -- the
    driver deliberately reuses the SAME corpus files across fused/eager/
    control at a given shape, so their identity fields are expected to
    agree exactly), every pair of legs must agree on every
    `FINETUNE_RUN_IDENTITY_FIELDS` entry. A mismatch appends a violation to
    BOTH legs of the disagreeing pair -- there is no way to tell which one
    is "wrong", so neither is trusted.
    """
    groups = {}
    for leg in legs:
        if leg["tier"] is None:
            continue
        row = leg["row"]
        key = (row.get("model"), row.get("shape"), row.get("steps"))
        groups.setdefault(key, []).append(leg)

    for group in groups.values():
        if len(group) < 2:
            continue
        ids = [
            generic_leg_identity_fields(
                leg["tier"], FINETUNE_RUN_IDENTITY_FIELDS, FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS
            )
            for leg in group
        ]
        ref_leg, ref_id = group[0], ids[0]
        for leg, this_id in zip(group[1:], ids[1:]):
            vs = generic_leg_premise_violations(
                FINETUNE_RUN_IDENTITY_FIELDS,
                ref_id,
                this_id,
                label_a=ref_leg["row"].get("leg_id"),
                label_b=leg["row"].get("leg_id"),
            )
            if vs:
                ref_leg["violations"].extend(vs)
                leg["violations"].extend(vs)


def _bucket_key(row):
    if row.get("arm") == "control":
        return ("control", row.get("model"))
    return ("measurement", row.get("model"), row.get("shape"))


def compute_buckets(legs):
    """Groups legs into measurement cells / control series (see module
    doc's "Cell model"), computes per-arm `(n, min, median, max)` stats over
    the complete `(N, M)` repeat pairs, and applies the LITERAL,
    conservative "any violation anywhere in this bucket voids the whole
    bucket" rule. Returns `dict[bucket_key] -> BucketResult`.
    """
    buckets = {}
    for leg in legs:
        buckets.setdefault(_bucket_key(leg["row"]), []).append(leg)

    results = {}
    for key, bucket_legs in buckets.items():
        reasons = []
        for leg in bucket_legs:
            reasons.extend(leg["violations"])

        # Cross-arm EXTRA_DISABLE symmetry (CONTRACT: "applied to BOTH arms
        # symmetrically") -- `LORA_BIAS_AB_EXTRA_DISABLE` is ONE value for
        # the whole sweep, so every leg's own recorded `extra_disable` set
        # (fused, lora_eager, or control, whichever arms this bucket holds)
        # must agree; `dispatch_violations` above only checks a leg's
        # SELF-consistency (its own `kernels_disabled_requested` vs its own
        # recorded `extra_disable`) -- this catches the CROSS-arm drift a
        # self-consistency check cannot (both arms individually honest
        # about what they saw, but disagreeing with each other).
        extra_sets = {frozenset(leg["row"].get("extra_disable") or []) for leg in bucket_legs}
        if len(extra_sets) > 1:
            reasons.append(
                f"{key}: inconsistent extra_disable recorded across this bucket's legs "
                f"(asymmetric JAMMI_KERNELS_DISABLE): {[sorted(s) for s in extra_sets]}"
            )

        arms = sorted({leg["row"].get("arm") for leg in bucket_legs})
        per_arm = {}
        for arm in arms:
            by_repeat = {}
            for leg in bucket_legs:
                if leg["row"].get("arm") != arm:
                    continue
                # A leg whose report never loaded (rc != 0, a missing/
                # unparseable file) already contributed its own violation
                # via `build_legs` -- it must NEVER populate `steps_map`
                # here (there is no `tier` to read a wall out of), so it is
                # correctly treated as an ABSENT N/M leg below, not a
                # `NoneType` crash.
                if leg["tier"] is None:
                    continue
                by_repeat.setdefault(leg["row"].get("repeat"), {})[leg["row"].get("steps")] = leg
            values = []
            for repeat in sorted(by_repeat):
                steps_map = by_repeat[repeat]
                if STEPS_N not in steps_map or STEPS_M not in steps_map:
                    reasons.append(
                        f"{key}/{arm}/r{repeat}: missing its N={STEPS_N} or M={STEPS_M} leg"
                    )
                    continue
                wall_n = steps_map[STEPS_N]["tier"]["train_run_wall_s"]
                wall_m = steps_map[STEPS_M]["tier"]["train_run_wall_s"]
                s = (wall_m - wall_n) / (STEPS_M - STEPS_N)
                if not _is_finite_real(s) or s <= 0:
                    reasons.append(
                        f"{key}/{arm}/r{repeat}: non-positive/non-finite per-step wall "
                        f"differencing result ({s!r}) -- wall_m={wall_m!r} wall_n={wall_n!r}"
                    )
                    continue
                values.append(s)
            if not values:
                reasons.append(f"{key}/{arm}: no complete, valid (N, M) repeat pair")
                per_arm[arm] = None
            else:
                per_arm[arm] = {
                    "n": len(values),
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                    "values": values,
                }

        valid = not reasons and all(v is not None for v in per_arm.values())
        results[key] = {
            "valid": valid,
            "reasons": sorted(set(reasons)),
            "per_arm": per_arm,
        }
    return results


def compute_model_verdict(model, buckets):
    control = buckets.get(("control", model))
    wire = buckets.get(("measurement", model, WIRE_SHAPE))
    chapter = buckets.get(("measurement", model, CHAPTER_SHAPE))

    if control is None or not control["valid"]:
        return {
            "verdict": "INVALID",
            "reason": "control series invalid or missing -- floor undefined",
            "control_reasons": (control or {}).get("reasons", []),
        }
    if wire is None or not wire["valid"] or wire["per_arm"].get("fused") is None:
        return {
            "verdict": "INVALID",
            "reason": "wire-shape (b8W512f32) fused-arm cell invalid or missing -- floor undefined",
            "wire_reasons": (wire or {}).get("reasons", []),
        }

    fused_median_wire = wire["per_arm"]["fused"]["median"]
    control_median = control["per_arm"]["control"]["median"]
    if fused_median_wire == 0:
        return {"verdict": "INVALID", "reason": "wire fused median is zero -- floor undefined"}
    floor = abs(fused_median_wire - control_median) / fused_median_wire

    per_shape_gain = {}
    missing_shapes = []
    per_shape_cells = {WIRE_SHAPE: wire, CHAPTER_SHAPE: chapter}
    for shape, cell in per_shape_cells.items():
        if cell is None or not cell["valid"]:
            missing_shapes.append(shape)
            continue
        fused_m = cell["per_arm"].get("fused")
        eager_m = cell["per_arm"].get("lora_eager")
        if fused_m is None or eager_m is None or fused_m["median"] == 0:
            missing_shapes.append(shape)
            continue
        per_shape_gain[shape] = 1.0 - (fused_m["median"] / eager_m["median"])

    activated = {s: g for s, g in per_shape_gain.items() if g >= GAIN_BAR and g > floor}
    regressed = {s: g for s, g in per_shape_gain.items() if g < -floor}

    if activated:
        verdict = "ACTIVATE"
    elif regressed:
        verdict = "REGRESSION"
    elif not missing_shapes:
        verdict = "NEUTRAL"
    else:
        verdict = "INVALID"

    result = {
        "verdict": verdict,
        "floor": floor,
        "fused_median_wire_s_per_step": fused_median_wire,
        "control_median_s_per_step": control_median,
        "gain_by_shape": per_shape_gain,
    }
    if missing_shapes:
        result["missing_or_invalid_shapes"] = missing_shapes
    if verdict == "INVALID" and not missing_shapes:
        result["reason"] = "no shape activated or regressed, but coverage was somehow incomplete"
    elif verdict == "INVALID":
        result["reason"] = f"shape(s) invalid/missing, no ACTIVATE/REGRESSION signal from the rest: {missing_shapes}"
    return result


def _resolve_git_sha(explicit):
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001 -- any failure here is fatal, never a fabricated sha
        raise SystemExit(f"::error::lora_bias_ab_merge: could not resolve git sha (pass --git-sha): {e}")


def print_table(model_verdicts, buckets):
    print(f"{'model':<12} {'shape':<12} {'arm':<12} {'n':>3} {'median s/step':>16}")
    for model in MODELS:
        for shape in (WIRE_SHAPE, CHAPTER_SHAPE):
            cell = buckets.get(("measurement", model, shape))
            if cell is None:
                continue
            for arm in ("fused", "lora_eager"):
                stats = cell["per_arm"].get(arm)
                n = stats["n"] if stats else 0
                median = f"{stats['median']:.6f}" if stats else "n/a"
                print(f"{model:<12} {shape:<12} {arm:<12} {n:>3} {median:>16}")
        control = buckets.get(("control", model))
        if control is not None:
            stats = control["per_arm"].get("control")
            n = stats["n"] if stats else 0
            median = f"{stats['median']:.6f}" if stats else "n/a"
            print(f"{model:<12} {'control':<12} {'control':<12} {n:>3} {median:>16}")
    print()
    for model, verdict in model_verdicts.items():
        gain_str = ", ".join(f"{s}={g:.4f}" for s, g in verdict.get("gain_by_shape", {}).items())
        floor_str = f"{verdict['floor']:.4f}" if "floor" in verdict else "n/a"
        print(f"{model}: {verdict['verdict']} (floor={floor_str}, gain={{{gain_str}}})")


def build_artifact(git_sha, box, invocation, model_verdicts, legs, buckets):
    legs_out = []
    for leg in legs:
        row = leg["row"]
        legs_out.append(
            {
                "leg_id": row.get("leg_id"),
                "model": row.get("model"),
                "shape": row.get("shape"),
                "arm": row.get("arm"),
                "steps": row.get("steps"),
                "repeat": row.get("repeat"),
                "valid": not leg["violations"],
                "violations": leg["violations"],
                "wall_s": (leg["tier"] or {}).get("train_run_wall_s") if leg["tier"] else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "box": box,
        "producer": {
            "path": "ci/scripts/perf/lora_bias_ab.sh",
            "kind": "script",
            "invocation": invocation,
            "gating": "none",
        },
        "status": "GREEN",
        "notes": {
            "what": (
                "GH #428 P2b: same-box fused-vs-eager per-step wall A/B for BERT/DistilBERT at "
                "the two shapes issue #356's own close-out profile ACTIVATED the C-LORA port on, "
                "proving out the bias-carrying-base widening once it lands."
            ),
            "method": (
                "per-repeat differencing s_per_step = (wall_600 - wall_100) / 500; "
                "gain = 1 - fused_median / eager_median; "
                "floor = |fused_median_wire - control_median| / fused_median_wire (one per model); "
                "activation_bar: gain >= 0.05 and gain > floor on >= 1 activating shape."
            ),
            "verdicts": model_verdicts,
        },
        "legs": legs_out,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_dir", help="the sweep's OUT_DIR (holds manifest.json + raw/*.json)")
    ap.add_argument("out_json", help="where to write the merged cuda-runs artifact")
    ap.add_argument("--git-sha", default=None, help="40-hex sha (default: resolve HEAD)")
    ap.add_argument("--box", default="unknown", help="the physical/pod box identifier")
    ap.add_argument(
        "--producer-invocation",
        default=None,
        help="the driver invocation string to record in producer.invocation",
    )
    args = ap.parse_args(argv)

    git_sha = _resolve_git_sha(args.git_sha)
    invocation = args.producer_invocation or "ci/scripts/perf/lora_bias_ab.sh"

    rows = load_manifest(args.raw_dir)
    legs = build_legs(args.raw_dir, rows)
    apply_identity_checks(legs)
    buckets = compute_buckets(legs)

    model_verdicts = {model: compute_model_verdict(model, buckets) for model in MODELS}

    artifact = build_artifact(git_sha, args.box, invocation, model_verdicts, legs, buckets)
    with open(args.out_json, "w") as f:
        json.dump(artifact, f, indent=1)

    print_table(model_verdicts, buckets)
    print(f"\n=== merged artifact: {args.out_json} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
