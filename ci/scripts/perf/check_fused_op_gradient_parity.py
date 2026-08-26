#!/usr/bin/env python3
"""CI-shaped gate for a fused-op PR: fail if a SINGLE-op kernel ablation
moves the gradient-direction cosine (against an f32-precision truth run)
too far from the all-fused reference arm, AT BOTH of two required shapes.

Consumes TWO `jammi-bench grad-oracle --ablate-each-op` `AblationReport`s
(`crates/jammi-bench/src/grad_oracle_ablation.rs`'s own schema — the SAME
files `ci/scripts/perf/compare_grad_oracle.py --ablation` reads), one at
shape `b4-s128` (batch=4, seq=128) and one at `b8-s128` (batch=8, seq=128).
For every `ablate:<op_key>` arm (one op forced eager via
`JAMMI_KERNELS_DISABLE=<op_key>`, every OTHER op still strictly proven fused
— see that module's own doc), computes the MEDIAN-over-seeds delta
`|median(cosine) - median(all_fused cosine)|` at EACH shape and FAILS the
op only if BOTH shapes' deltas exceed THAT SHAPE'S OWN derived budget.

## Round-7 audit fix (PR #383): why this is no longer a fixed-fraction threshold

Round 6 of this gate derived a single fixed threshold (`0.20 whole-gap /
10`) from ONE run at ONE shape/seed. Round 7's own two-shape run on the
SAME build inverted the story entirely (`b4-s128`: `all_fused=0.610 <
all_off=0.810`; `b8-s128`: `all_fused=0.843 > all_off=0.773`) — a
single-seed cosine at this operating point is CHAOTIC, so ANY fixed
constant fitted to one measurement is unsound. The FIX is not a better
constant; it is deriving the budget from the MEASURED SPREAD of the
`all_fused` arm's OWN cosine across `--seeds` (see `AblationReport::
derived_per_op_budget`'s own Rust-side doc: `3 * (max - min)` of
`all_fused`'s median-per-seed cosine, AT THAT SHAPE), and requiring a
median-delta to exceed that spread-derived budget at BOTH `b4-s128` and
`b8-s128` before calling it a real, shape-independent defect — a single
shape's own instability (however it happens to land relative to any FIXED
threshold) must not, alone, gate a PR.

This module does NOT recompute the budget from scratch itself: it reads
`derived_per_op_budget` from EACH report (already Rust-computed from that
report's own `per_seed` data) and INDEPENDENTLY RE-DERIVES it from the same
`per_seed` values as a cross-check (`_recompute_budget`) — a disagreement
between the two computations of the identical formula is itself a
refusal, never silently accepted.

## What else this gate refuses (never only the cosine delta)

Reuses `compare_grad_oracle._ablation_provenance_problems` (imported, not
reimplemented) so a report whose provenance is not self-describing (an
unmatched `JAMMI_KERNELS_DISABLE` entry, a nonzero `vacuous_tensor_count`
on any seed, a non-finite cosine) FAILS here too, never silently passing a
budget check computed off untrustworthy inputs. A report with ZERO
`ablate:<op_key>` arms is ALSO a refusal (family F non-vacuous control:
this gate proves nothing about any fused op without at least one).

Usage:
    python3 check_fused_op_gradient_parity.py B4_S128.json B8_S128.json
    python3 check_fused_op_gradient_parity.py --self-test

Exit codes: 0 PASS, 1 FAIL (both reports loaded, a real problem found), 2
REFUSED (malformed/insufficient input — cannot even attempt the check).
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grad_oracle import _ablation_provenance_problems, _recompute_seed_stat  # noqa: E402

# The two shapes this gate REQUIRES, both present, before it will render any
# verdict — see this module's own doc's "why this is no longer a
# fixed-fraction threshold" section. Keyed by (batch, seq); the label is
# purely for messages.
REQUIRED_SHAPES = {(4, 128): "b4-s128", (8, 128): "b8-s128"}

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2


def _shape_label(report: dict) -> str:
    batch = report.get("batch")
    seq = report.get("seq")
    return REQUIRED_SHAPES.get((batch, seq), f"b{batch}-s{seq}")


def _recompute_budget(report: dict) -> tuple[float | None, list[str]]:
    """Independently re-derive `derived_per_op_budget` from the
    `all_fused` arm's OWN `per_seed` `full_tensor_cosine` values — `3 *
    (max - min)`, the SAME formula `grad_oracle_ablation.rs` uses, computed
    here from scratch rather than trusted verbatim (family F). Returns
    `(budget_or_None, problems)`.
    """
    arms = report.get("arms") or []
    reference = next((a for a in arms if a.get("arm") == "all_fused"), None)
    if reference is None:
        return None, ["no 'all_fused' reference arm present"]
    per_seed = reference.get("per_seed") or []
    values = [
        s["full_tensor_cosine"]
        for s in per_seed
        if isinstance(s.get("full_tensor_cosine"), (int, float)) and math.isfinite(s["full_tensor_cosine"])
    ]
    if not values:
        return None, ["all_fused arm has no usable per_seed full_tensor_cosine values"]
    stat = _recompute_seed_stat(values)
    recomputed = 3.0 * (stat["max"] - stat["min"])
    reported = report.get("derived_per_op_budget")
    problems = []
    if not isinstance(reported, (int, float)) or abs(reported - recomputed) > 1e-9:
        problems.append(
            f"reported derived_per_op_budget={reported!r} disagrees with this gate's own "
            f"recomputation from all_fused's per_seed data ({recomputed!r})"
        )
    return recomputed, problems


def _median_cosine(arm: dict) -> float | None:
    c = arm.get("full_tensor_cosine") or {}
    m = c.get("median")
    return m if isinstance(m, (int, float)) and math.isfinite(m) else None


def check_reports(reports: list[dict]) -> tuple[bool, list[str]]:
    """`(passed, problems)` — `problems` is empty iff `passed`. Never
    raises on a malformed report (every lookup is defensive).
    """
    problems: list[str] = []

    if len(reports) != 2:
        return False, [f"expected exactly 2 shape reports (b4-s128, b8-s128), got {len(reports)}"]

    seen_shapes: dict[str, dict] = {}
    for report in reports:
        problems.extend(_ablation_provenance_problems(report))
        label = _shape_label(report)
        seen_shapes[label] = report

    required_labels = set(REQUIRED_SHAPES.values())
    missing = sorted(required_labels - seen_shapes.keys())
    if missing:
        problems.append(
            f"missing required shape(s) {missing} -- both b4-s128 and b8-s128 must be present, "
            "a single shape's own instability must not, alone, gate this PR"
        )
        return False, problems

    per_shape_budget: dict[str, float] = {}
    per_shape_ref_median: dict[str, float] = {}
    per_shape_ablate_arms: dict[str, dict[str, dict]] = {}
    for label, report in seen_shapes.items():
        budget, budget_problems = _recompute_budget(report)
        problems.extend([f"{label}: {p}" for p in budget_problems])
        if budget is None:
            continue
        per_shape_budget[label] = budget

        arms = report.get("arms") or []
        reference = next((a for a in arms if a.get("arm") == "all_fused"), None)
        ref_median = _median_cosine(reference) if reference else None
        if ref_median is None:
            problems.append(f"{label}: all_fused arm has no finite full_tensor_cosine.median")
            continue
        per_shape_ref_median[label] = ref_median

        ablate_arms = {
            a["op_key"]: a
            for a in arms
            if isinstance(a.get("arm"), str) and a["arm"].startswith("ablate:") and a.get("op_key")
        }
        if not ablate_arms:
            problems.append(
                f"{label}: zero 'ablate:<op_key>' arms present -- this gate proves nothing about "
                "any fused op without at least one; refusing to report a vacuous PASS"
            )
            continue
        per_shape_ablate_arms[label] = ablate_arms

    if problems:
        return False, problems

    all_op_keys = sorted(set().union(*(set(d.keys()) for d in per_shape_ablate_arms.values())))
    if not all_op_keys:
        return False, ["no op key was tested (ablate arm) at any required shape"]

    for op_key in all_op_keys:
        shapes_that_fail: list[str] = []
        for label in sorted(required_labels):
            arms = per_shape_ablate_arms.get(label, {})
            arm = arms.get(op_key)
            if arm is None:
                # This op was not live/ablated at this particular shape --
                # not itself a failure (a real, structural absence, e.g. a
                # dtype-gated op that never admits at one shape); only
                # shapes where the op WAS actually tested count toward the
                # "both shapes" requirement below.
                continue
            median = _median_cosine(arm)
            if median is None:
                problems.append(f"{label}: ablate:{op_key} has no finite full_tensor_cosine.median")
                continue
            delta = abs(median - per_shape_ref_median[label])
            budget = per_shape_budget[label]
            if delta > budget:
                shapes_that_fail.append(
                    f"{label} (delta={delta:.6f} > budget={budget:.6f}, all_fused_median="
                    f"{per_shape_ref_median[label]:.6f}, {op_key}_median={median:.6f})"
                )
        if len(shapes_that_fail) >= 2:
            problems.append(
                f"{op_key}: median cosine delta exceeds its shape's own derived budget at BOTH "
                f"required shapes: {shapes_that_fail}"
            )

    return (len(problems) == 0), problems


def _self_test() -> int:
    """RED/GREEN cases against SYNTHETIC reports — never a real GPU run.
    Includes a dedicated pair of MATERIALLY DIVERGING bf16-vs-f32-shaped
    synthetic reports (round-7 audit item 6, PR #383): `all_fused` itself
    sits well below `1.0` (a real bf16-vs-f32 divergence, not the trivial
    f32-vs-f32 vacuous case), exercised BOTH ways -- once engineered to
    PASS (the per-op deltas stay inside the derived budget at both shapes)
    and once to FAIL (one op's delta exceeds the budget at both shapes).
    """
    failures = 0

    def sample(seed: int, cosine: float, op_key: str | None) -> dict:
        return {
            "seed": seed,
            "full_tensor_cosine": cosine,
            "vacuous_tensor_count": 0,
            "kernels_disabled_requested": [] if op_key is None else [op_key],
            "kernels_disabled_fired": [] if op_key is None else [op_key],
            "unmatched_disables": [],
        }

    def arm(label: str, op_key: str | None, cosines: dict[int, float]) -> dict:
        per_seed = [sample(seed, c, op_key) for seed, c in cosines.items()]
        values = list(cosines.values())
        return {
            "arm": label,
            "op_key": op_key,
            "full_tensor_cosine": {
                "median": sorted(values)[len(values) // 2],
                "min": min(values),
                "max": max(values),
            },
            "per_seed": per_seed,
        }

    def report(batch: int, seq: int, ref_cosines: dict[int, float], ablate: dict[str, dict[int, float]]) -> dict:
        arms = [arm("all_fused", None, ref_cosines)]
        for key, cosines in ablate.items():
            arms.append(arm(f"ablate:{key}", key, cosines))
        arms.append(arm("all_off", None, {s: c - 0.05 for s, c in ref_cosines.items()}))
        ref_values = list(ref_cosines.values())
        budget = 3.0 * (max(ref_values) - min(ref_values))
        return {"batch": batch, "seq": seq, "seeds": list(ref_cosines.keys()), "derived_per_op_budget": budget, "arms": arms}

    def expect(name: str, reports: list[dict], want_pass: bool, needle: str | None = None) -> None:
        nonlocal failures
        passed, problems = check_reports(reports)
        if passed != want_pass:
            print(f"SELF-TEST FAIL [{name}]: expected passed={want_pass}, got {passed} ({problems})")
            failures += 1
            return
        if needle is not None and not any(needle in p for p in problems):
            print(f"SELF-TEST FAIL [{name}]: expected a problem containing {needle!r}, got {problems}")
            failures += 1

    seeds = {42: 0.80, 43: 0.82, 44: 0.78}  # spread 0.04, budget = 0.12

    # GREEN: within-budget ablations at BOTH shapes.
    b4_clean = report(4, 128, seeds, {"layer_norm_fused": {42: 0.75, 43: 0.79, 44: 0.83}})
    b8_clean = report(8, 128, seeds, {"layer_norm_fused": {42: 0.78, 43: 0.80, 44: 0.82}})
    expect("green: within budget at both shapes", [b4_clean, b8_clean], True)

    # RED: an op that exceeds the budget at BOTH shapes.
    b4_bad = report(4, 128, seeds, {"geglu_fused": {42: 0.30, 43: 0.32, 44: 0.28}})
    b8_bad = report(8, 128, seeds, {"geglu_fused": {42: -0.20, 43: -0.10, 44: -0.30}})
    expect("red: exceeds budget at both shapes", [b4_bad, b8_bad], False, "geglu_fused")

    # A shape-SPECIFIC defect (fails at ONE shape only) must NOT fail the
    # gate alone -- this is the exact "b4/b8 inverted story" property this
    # round's fix is FOR.
    b4_one_shape_bad = report(4, 128, seeds, {"lora_linear_fused": {42: 0.30, 43: 0.32, 44: 0.28}})
    b8_one_shape_ok = report(8, 128, seeds, {"lora_linear_fused": {42: 0.79, 43: 0.81, 44: 0.83}})
    expect(
        "green: fails budget at ONLY one shape -- must not gate alone",
        [b4_one_shape_bad, b8_one_shape_ok],
        True,
    )

    # A MATERIALLY DIVERGING bf16-vs-f32-shaped pair (round-7 audit item 6):
    # all_fused itself sits at ~0.62 (a real cross-dtype divergence, NOT
    # the trivial f32-vs-f32 near-1.0 vacuous case) -- exercised both ways.
    bf16_seeds = {1: 0.60, 2: 0.65, 3: 0.62}  # spread 0.05, budget 0.15
    bf16_pass_b4 = report(4, 128, bf16_seeds, {"layer_norm_fused": {1: 0.55, 2: 0.70, 3: 0.60}})
    bf16_pass_b8 = report(8, 128, bf16_seeds, {"layer_norm_fused": {1: 0.58, 2: 0.68, 3: 0.63}})
    expect("bf16-shaped: within-budget PASS", [bf16_pass_b4, bf16_pass_b8], True)

    bf16_fail_b4 = report(4, 128, bf16_seeds, {"layer_norm_fused": {1: -0.10, 2: -0.05, 3: -0.08}})
    bf16_fail_b8 = report(8, 128, bf16_seeds, {"layer_norm_fused": {1: -0.20, 2: -0.15, 3: -0.18}})
    expect("bf16-shaped: exceeds-budget FAIL", [bf16_fail_b4, bf16_fail_b8], False, "layer_norm_fused")

    # RED: missing a required shape.
    expect("red: only one shape provided", [b4_clean], False)
    duplicate_shape = report(4, 128, seeds, {"layer_norm_fused": {42: 0.75, 43: 0.79, 44: 0.83}})
    expect("red: same shape twice, b8-s128 missing", [b4_clean, duplicate_shape], False, "missing")

    # RED (non-vacuous control): zero ablate arms at one shape.
    b4_no_ablate = report(4, 128, seeds, {})
    expect("red: zero ablate arms at one shape", [b4_no_ablate, b8_clean], False, "zero")

    # RED: provenance not self-describing (unmatched disable) at one shape.
    b4_unmatched = report(4, 128, seeds, {"layer_norm_fused": {42: 0.75, 43: 0.79, 44: 0.83}})
    b4_unmatched["arms"][1]["per_seed"][0]["kernels_disabled_fired"] = []
    expect("red: unmatched disable", [b4_unmatched, b8_clean], False, "not self-describing")

    # RED: vacuous_tensor_count nonzero anywhere.
    b4_vacuous = report(4, 128, seeds, {"layer_norm_fused": {42: 0.75, 43: 0.79, 44: 0.83}})
    b4_vacuous["arms"][0]["per_seed"][0]["vacuous_tensor_count"] = 5
    expect("red: nonzero vacuous_tensor_count", [b4_vacuous, b8_clean], False, "vacuous_tensor_count")

    # RED: budget cross-check disagreement (a tampered derived_per_op_budget).
    b4_tampered_budget = report(4, 128, seeds, {"layer_norm_fused": {42: 0.75, 43: 0.79, 44: 0.83}})
    b4_tampered_budget["derived_per_op_budget"] = 999.0
    expect("red: tampered derived_per_op_budget", [b4_tampered_budget, b8_clean], False, "disagrees")

    if failures:
        print(f"SELF-TEST: {failures} case(s) failed")
        return EXIT_FAIL
    print("SELF-TEST: all cases passed")
    return EXIT_PASS


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="*", help="Exactly 2 AblationReport JSON paths: b4-s128 and b8-s128.")
    p.add_argument("--self-test", action="store_true", default=False)
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if len(args.reports) != 2:
        print(
            f"REFUSED: expected exactly 2 report paths (b4-s128, b8-s128), got {len(args.reports)} "
            "(or --self-test)",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    reports = []
    for path in args.reports:
        try:
            with open(path) as fh:
                reports.append(json.load(fh))
        except (OSError, json.JSONDecodeError) as e:
            print(f"REFUSED: could not load {path!r}: {e}", file=sys.stderr)
            return EXIT_REFUSED

    passed, problems = check_reports(reports)
    for msg in problems:
        print(f"PROBLEM: {msg}", file=sys.stderr)
    print("PASS" if passed else "FAIL")
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
