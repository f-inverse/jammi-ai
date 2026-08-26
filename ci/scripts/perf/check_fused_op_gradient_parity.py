#!/usr/bin/env python3
"""CI-shaped gate for a fused-op PR: fail if a SINGLE-op kernel ablation
moves the gradient-direction cosine (against an f32-precision truth run)
too far from the all-fused reference arm.

Consumes a `jammi-bench grad-oracle --ablate-each-op` `AblationReport`
(`crates/jammi-bench/src/grad_oracle_ablation.rs`'s own schema — the SAME
file `ci/scripts/perf/compare_grad_oracle.py --ablation` reads) and, for
every `ablate:<op_key>` arm (one op forced eager via
`JAMMI_KERNELS_DISABLE=<op_key>`, every OTHER op still strictly proven fused
via `JAMMI_KERNELS_STRICT=1` — see that module's own doc), computes
`|cosine(ablate:<op_key>) - cosine(all_fused)|` and FAILS if it exceeds
[`FUSED_OP_COSINE_DELTA_THRESHOLD`].

## Threshold derivation (ledger row 240) — never fitted to a specific PR

A live A100 run of this feature's own commissioning dispatch (ModernBERT-
large, `--lora-init gaussian`, so `dL/dA` and the LoRA epilogue's own
gradient term are LIVE — see `grad_oracle.rs`'s module doc's "Structural
limitation" section for why a `ZerosB`-init run cannot even see this)
measured:

    all_fused (bf16, every fused op ON)  cosine vs f32 truth = 0.610
    all_off   (bf16, every fused op OFF) cosine vs f32 truth = 0.810

The WHOLE gap between "every fused op forced eager" and "every fused op
forced on" on this architecture is `|0.610 - 0.810| = 0.20`, distributed
across (at most) the four op keys that were LIVE that run
(`attention_block_fused`, `geglu_fused`, `layer_norm_fused`,
`lora_linear_fused` — see `GradOracleReport::live_admit_keys`'s own doc for
why the exact set is discovered per run, never hardcoded). A SINGLE-op
ablation — every OTHER op still strictly proven fused in the SAME run —
should therefore move the cosine by, at most, a SMALL FRACTION of that
whole-composition gap: `FUSED_OP_COSINE_DELTA_THRESHOLD` picks 1/10th of the
measured whole-gap (`0.20 / 10 = 0.02`) as the per-op budget. This is a
derived, STATED number, not a value chosen to make any specific PR's own
measurement pass — if a future measured whole-gap differs meaningfully from
`0.20`, re-derive this constant from the new number and cite it here, never
silently widen the threshold to clear a failing PR.

## What else this gate refuses (never only the cosine delta)

Reuses `compare_grad_oracle._ablation_provenance_problems` (imported, not
reimplemented — the SAME mechanism `compare_grad_oracle.py --ablation`
checks) so a report whose provenance is not self-describing (an unmatched
`JAMMI_KERNELS_DISABLE` entry, a nonzero `vacuous_tensor_count`, a
non-finite `overall_cosine_vs_f32_truth`) FAILS here too, never silently
passing a threshold check computed off untrustworthy inputs.

Usage:
    python3 check_fused_op_gradient_parity.py ABLATION_REPORT_JSON
    python3 check_fused_op_gradient_parity.py --self-test

Exit codes: 0 PASS, 1 FAIL (report loaded, a real problem found), 2 REFUSED
(malformed input — cannot even attempt the check).
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grad_oracle import _ablation_provenance_problems  # noqa: E402

# See this module's own doc's "Threshold derivation" section: 1/10th of the
# measured 0.20 whole-composition (all_fused vs all_off) gap on ModernBERT-
# large at a live-gradient (`--lora-init gaussian`) fixture.
FUSED_OP_COSINE_DELTA_THRESHOLD = 0.02

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2


def check_report(report: dict, threshold: float = FUSED_OP_COSINE_DELTA_THRESHOLD) -> tuple[bool, list[str]]:
    """`(passed, problems)` — `problems` is empty iff `passed`. Never raises
    on a malformed report (every lookup is defensive); a structurally
    missing/malformed field is reported as a PROBLEM (which makes `passed`
    `False`), never silently skipped.
    """
    problems: list[str] = []

    provenance_problems = _ablation_provenance_problems(report)
    problems.extend(provenance_problems)

    arms = report.get("arms")
    if not isinstance(arms, list) or not arms:
        return False, problems + ["ablation report has no non-empty 'arms' list"]

    reference = next((a for a in arms if a.get("arm") == "all_fused"), None)
    if reference is None:
        return False, problems + ["no 'all_fused' reference arm present"]
    ref_cosine = reference.get("overall_cosine_vs_f32_truth")
    if not isinstance(ref_cosine, (int, float)) or not math.isfinite(ref_cosine):
        return False, problems + [f"all_fused arm's overall_cosine_vs_f32_truth is not finite: {ref_cosine!r}"]

    ablate_arms = [
        a for a in arms if isinstance(a.get("arm"), str) and a["arm"].startswith("ablate:")
    ]
    if not ablate_arms:
        # NON-VACUOUS CONTROL (family F): a report with zero per-op ablation
        # arms would otherwise silently PASS this gate (an empty loop below
        # finds no violation) despite proving NOTHING about any fused op --
        # this is the exact "controls are non-vacuous" trap this repo's own
        # constitution names. A report with no live admit keys on this
        # checkpoint (e.g. an all-CPU-eager fixture with no fused kernel
        # reachable at all) is a REFUSAL, not a silent pass.
        return False, problems + [
            "ablation report has zero 'ablate:<op_key>' arms -- this gate proves nothing about "
            "any fused op without at least one; refusing to report a vacuous PASS"
        ]

    for arm in ablate_arms:
        label = arm.get("arm", "<unnamed>")
        c = arm.get("overall_cosine_vs_f32_truth")
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            problems.append(f"{label}: overall_cosine_vs_f32_truth is not finite: {c!r}")
            continue
        delta = abs(c - ref_cosine)
        if delta > threshold:
            problems.append(
                f"{label}: |cosine_vs_f32_truth - all_fused_cosine| = {delta:.6f} exceeds the "
                f"per-op threshold {threshold} (all_fused={ref_cosine:.6f}, {label}={c:.6f}) -- "
                "this single op's own gradient contribution moved the direction more than the "
                "derived per-op budget allows"
            )

    return (len(problems) == 0), problems


def _self_test() -> int:
    """RED/GREEN cases against SYNTHETIC reports — never a real GPU run.
    Every field this function's own `check_report` reads gets at least one
    case that flips it from clean to a problem, one at a time.
    """
    failures = 0

    def clean_arm(label: str, cosine: float, op_key: str | None = None) -> dict:
        return {
            "arm": label,
            "op_key": op_key,
            "kernels_disabled_requested": [] if op_key is None else [op_key],
            "kernels_disabled_fired": [] if op_key is None else [op_key],
            "unmatched_disables": [],
            "vacuous_tensor_count": 0,
            "overall_cosine_vs_f32_truth": cosine,
        }

    def clean_report(deltas: dict[str, float], ref_cosine: float = 0.90) -> dict:
        arms = [clean_arm("all_fused", ref_cosine), clean_arm("f32_truth", 1.0)]
        for key, cosine in deltas.items():
            arms.append(clean_arm(f"ablate:{key}", cosine, op_key=key))
        arms.append(clean_arm("all_off", 0.70))
        return {"arms": arms}

    def expect(
        name: str,
        report: dict,
        want_pass: bool,
        needle: str | None = None,
        threshold: float = FUSED_OP_COSINE_DELTA_THRESHOLD,
    ) -> None:
        nonlocal failures
        passed, problems = check_report(report, threshold)
        if passed != want_pass:
            print(f"SELF-TEST FAIL [{name}]: expected passed={want_pass}, got {passed} ({problems})")
            failures += 1
            return
        if needle is not None and not any(needle in p for p in problems):
            print(f"SELF-TEST FAIL [{name}]: expected a problem containing {needle!r}, got {problems}")
            failures += 1

    # GREEN: every per-op arm within the threshold of all_fused (0.90).
    expect("green: within threshold", clean_report({"layer_norm_fused": 0.885, "geglu_fused": 0.905}), True)

    # RED: one op's ablation moves the cosine beyond the threshold.
    expect(
        "red: one op exceeds threshold",
        clean_report({"layer_norm_fused": 0.60, "geglu_fused": 0.905}),
        False,
        "layer_norm_fused",
    )

    # RED: exactly AT the threshold boundary must still PASS (`>`, not
    # `>=`) -- pinned so a mutant flipping the comparison operator is
    # detectable. Uses a reference/threshold/cosine triple that is EXACT in
    # binary floating point (0.75, 0.25, 0.5 are all exact powers-of-two
    # fractions) so the boundary itself is not obscured by ordinary float
    # subtraction noise (`0.90 - 0.02`, for example, is NOT exact in
    # binary64 -- an earlier draft of this test used exactly that pair and
    # false-failed on rounding noise it never intended to test).
    expect(
        "boundary: exactly at threshold passes",
        clean_report({"layer_norm_fused": 0.5}, ref_cosine=0.75),
        True,
        threshold=0.25,
    )
    expect(
        "boundary: just past threshold fails",
        clean_report({"layer_norm_fused": 0.5 - 1e-9}, ref_cosine=0.75),
        False,
        threshold=0.25,
    )

    # RED: missing arms list entirely.
    expect("red: no arms key", {}, False, "arms")

    # RED (non-vacuous control): zero ablate: arms must never silently pass.
    expect(
        "red: zero ablate arms",
        {"arms": [clean_arm("all_fused", 0.90), clean_arm("f32_truth", 1.0), clean_arm("all_off", 0.70)]},
        False,
        "zero",
    )

    # RED: missing all_fused reference arm.
    no_ref = clean_report({"layer_norm_fused": 0.89})
    no_ref["arms"] = [a for a in no_ref["arms"] if a["arm"] != "all_fused"]
    expect("red: no all_fused arm", no_ref, False, "all_fused")

    # AFFIRMATIVE non-finite refusal (family F): NaN must not silently
    # compare `False` on `> threshold` and slip through as a pass.
    nan_report = clean_report({"layer_norm_fused": float("nan")})
    expect("red: nan cosine on an ablate arm", nan_report, False, "not finite")

    nan_ref = clean_report({"layer_norm_fused": 0.89})
    nan_ref["arms"][0]["overall_cosine_vs_f32_truth"] = float("nan")
    expect("red: nan cosine on all_fused", nan_ref, False, "not finite")

    # RED: provenance not self-describing (unmatched disable).
    unmatched_report = clean_report({"layer_norm_fused": 0.89})
    for arm in unmatched_report["arms"]:
        if arm["arm"] == "ablate:layer_norm_fused":
            arm["kernels_disabled_fired"] = []  # requested but never fired
    expect("red: unmatched disable", unmatched_report, False, "not self-describing")

    # RED: vacuous_tensor_count nonzero anywhere.
    vacuous_report = clean_report({"layer_norm_fused": 0.89})
    vacuous_report["arms"][0]["vacuous_tensor_count"] = 3
    expect("red: nonzero vacuous_tensor_count", vacuous_report, False, "vacuous_tensor_count")

    if failures:
        print(f"SELF-TEST: {failures} case(s) failed")
        return EXIT_FAIL
    print("SELF-TEST: all cases passed")
    return EXIT_PASS


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ablation_report", nargs="?")
    p.add_argument("--self-test", action="store_true", default=False)
    p.add_argument("--threshold", type=float, default=FUSED_OP_COSINE_DELTA_THRESHOLD)
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.ablation_report:
        print("REFUSED: an ABLATION_REPORT_JSON path (or --self-test) is required", file=sys.stderr)
        return EXIT_REFUSED

    try:
        with open(args.ablation_report) as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"REFUSED: could not load {args.ablation_report!r}: {e}", file=sys.stderr)
        return EXIT_REFUSED

    passed, problems = check_report(report, args.threshold)
    for msg in problems:
        print(f"PROBLEM: {msg}", file=sys.stderr)
    print(f"threshold: {args.threshold}")
    print("PASS" if passed else "FAIL")
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
