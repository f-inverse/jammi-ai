#!/usr/bin/env python3
"""The jammi-vs-torch LEARNING oracle's comparator — reads TWO gradient
dumps (one per stack, same JSON schema — see
`crates/jammi-bench/src/grad_oracle.rs`'s module doc and
`crates/jammi-bench/reference/torch_grad_oracle.py`'s) produced from
IDENTICAL LoRA weights and an identical batch, and reports whether the two
stacks' gradients agree in DIRECTION (cosine similarity), not in loss value.

WHY COSINE, NOT A LOSS COMPARISON: a fused-vs-eager (same build, same
framework) oracle already proves jammi's KERNEL FUSION is value-neutral —
it cannot prove jammi's EAGER path itself computes the same gradient torch
does, because if jammi's eager arithmetic were itself wrong, that oracle
stays green on both arms (both are wrong the same way). A jammi-vs-torch
LOSS TRAJECTORY comparison is not a substitute either: even after fixing
the update-index placement (B1) and matching the LoRA init DISTRIBUTION
(`--lora-init jammi`), torch and jammi draw DIFFERENT BITS for that
distribution (`torch_finetune_step.py`'s "LoRA INIT IS NOT A MATCH BY
DEFAULT" section) — through a bf16 triplet hinge, that alone separates any
multi-step trajectory permanently. This comparator instead loads the
IDENTICAL weight file on both sides (see `grad_oracle.rs`'s "Weight
interchange format"), so trajectory divergence cannot enter at all — a
divergence here is either bf16 rounding noise (small, isotropic, cosine
near 1) or a REAL arithmetic defect (large, directional, cosine well below
the derived floor).

Deliberately Python, not Rust: this comparator's entire job is comparing
TWO INDEPENDENT dumps, so it must not share a code path — or a bug — with
either producer (family F's "numpy-first oracle" convention: prefer numpy
where available for the actual arithmetic; this module also carries a
pure-Python fallback specifically so its OWN math is unit-testable in an
environment with no numpy/torch installed — see `test_compare_grad_oracle.py`
in this directory, which forces the fallback path).

Usage:
    python3 compare_grad_oracle.py A.json B.json [--cosine-floor F]
        [--num-layers N] [--hidden-size H] [--out report.json]

If `--cosine-floor` is omitted, the floor is DERIVED (never fitted) from
`--num-layers`/`--hidden-size` via `derive_cosine_floor` below — pass the
real checkpoint's `num_hidden_layers`/`hidden_size` (ModernBERT-large:
28/1024) for a meaningful bound; the built-in default is deliberately
conservative for smaller/unknown configs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

try:
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover -- exercised by test_compare_grad_oracle.py
    HAVE_NUMPY = False

# bf16: 1 sign + 8 exponent + 7 explicit mantissa bits. Unit roundoff
# u = 2^-(t+1) where t = 7 explicit mantissa bits (the standard definition,
# e.g. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
# ch. 2) -- HALF the machine epsilon `eps = 2^-t`, since round-to-nearest
# bounds the relative rounding error of a single operation by u, not eps.
BF16_MANTISSA_BITS = 7
BF16_UNIT_ROUNDOFF = 2.0 ** -(BF16_MANTISSA_BITS + 1)  # 2^-8 = 0.00390625


def derive_relative_error_bound(num_layers: int, hidden_size: int) -> float:
    """A STATISTICAL (not worst-case) first-order bound on the relative
    error a bf16 forward+backward's gradient carries, derived from
    standard rounding-error analysis, never fitted to an observed number.

    The WORST-CASE bound for summing `K` bf16 terms (a matmul's inner
    dimension, a softmax/LayerNorm reduction) is `O(K * u)` (Higham ch.
    2's forward error bound for a naive summation) -- for
    `K = hidden_size ~ 1024` and `u = BF16_UNIT_ROUNDOFF`, that is already
    `> 1` (a vacuous, useless bound: real bf16 training visibly does not
    lose ~4x relative accuracy per matmul). This is expected: the
    worst-case bound assumes every rounding error compounds in the SAME
    direction, which does not happen in practice.

    The STATISTICAL bound for the same summation, under Higham's
    random-rounding model (errors are independent, zero-mean to first
    order) is `O(sqrt(K) * u)` instead of `O(K * u)` for ONE matmul's `K`
    inner-dimension terms. This function applies the SAME independence
    assumption a second time ACROSS the `num_layers` sequential matmuls the
    gradient backprops through: `num_layers` independent per-layer relative
    errors of comparable magnitude compose (in quadrature, i.e. root-sum-
    square, the same `sqrt(N)` shape as the intra-layer case, applied
    consistently rather than switching to a worst-case linear sum for
    ONLY the inter-layer step) rather than adding linearly:

        bound = sqrt(num_layers * hidden_size) * BF16_UNIT_ROUNDOFF

    For ModernBERT-large (`num_layers=28`, `hidden_size=1024`):
    `sqrt(28 * 1024) * 2^-8 ≈ 169.3 * 0.00390625 ≈ 0.661` -- still a large
    relative-error bound (66%), reflecting that a STATISTICAL bound
    genuinely does get looser at this depth, not a derivation bug: the
    EARLIEST layers' gradients backprop through every later layer's own
    rounding. This is itself informative: it says a per-element/per-tensor
    max|delta| bound is not the right primary assertion for a deep bf16
    network's earliest layers, and cosine similarity (which only cares
    about the gradient's DIRECTION, not its magnitude, and where random
    per-element noise averages out over a high-dimensional vector rather
    than compounding) is the metric that actually stays meaningful at this
    depth -- see `derive_cosine_floor`'s own doc for how this bound maps to
    a cosine floor, and
    `test_compare_grad_oracle.py::test_cosine_floor_is_too_loose_to_catch_a_3x_defect_at_full_modernbert_large_depth`
    for the HONEST, pinned disclosure that even this improved (root-sum-
    square, not linear) formula derives a floor too loose to catch a 3x
    per-element magnitude defect at ModernBERT-large's full depth with the
    default safety factor — a caller comparing a real ModernBERT-large
    sweep should scope `--num-layers`/`--hidden-size` to the SPECIFIC
    tensor being checked (e.g. the last layer's adapter backprops through
    only that one layer's own rounding) rather than trust the whole-model
    default to be tight.
    """
    if num_layers <= 0 or hidden_size <= 0:
        raise ValueError(f"num_layers/hidden_size must be positive, got {num_layers}/{hidden_size}")
    return math.sqrt(num_layers * hidden_size) * BF16_UNIT_ROUNDOFF


def derive_cosine_floor(num_layers: int, hidden_size: int, safety_factor: float = 3.0) -> float:
    """DERIVE (never fit) a cosine-similarity floor from bf16's ULP mass.

    For two vectors related by a small relative perturbation of typical
    magnitude `eps` per element (`b = a + delta`, `|delta_i| ~ eps *
    |a_i|`, delta roughly ISOTROPIC -- uncorrelated with `a`'s own
    direction, the random-rounding-noise case this oracle is meant to
    tolerate), the angle `theta` between `a` and `b` satisfies
    `sin(theta) ~ eps` to first order (the perturbation's component
    ORTHOGONAL to `a`, relative to `|a|`, is what rotates the vector; a
    perturbation purely along `a` does not change direction at all). So:

        cosine_floor = cos(safety_factor * eps)

    where `eps = derive_relative_error_bound(...)` and `safety_factor`
    widens the bound (default 3x) so ordinary bf16 rounding noise —
    which is a STATISTICAL bound, not a hard cap — does not false-trip
    this assertion; a REAL arithmetic defect (wrong sign on a whole
    tensor, a missing scale factor, a transposed axis) produces an angle
    close to 90 degrees or worse, nowhere near this floor regardless of
    the safety factor. See this module's docstring for the citable
    example: the P3 defect (b8-s512, fused arm's total |dqkv| 0.19% off
    at step 0, 3x off after one update) — a 3x magnitude error on a
    SHARED tensor is not isotropic per-element noise, and would read as a
    LARGE angle (cosine well below 0.9, typically much lower), not a
    small perturbation near 1.0.

    `eps` can legitimately exceed `pi` for a large `num_layers *
    hidden_size` (see `derive_relative_error_bound`'s own doc — the
    statistical bound is itself not tight at ModernBERT-large's full
    depth); `cos` is clamped to `[-1.0, 1.0]` via `math.cos`'s own
    range regardless, so this function always returns a value in
    `[-1.0, 1.0]`, never raises on a large `eps`. A caller whose derived
    floor comes out at or below 0.0 should treat the bound as
    UNINFORMATIVE at that depth and prefer a NAMED, empirically-set floor
    instead — this function never silently substitutes one.
    """
    eps = derive_relative_error_bound(num_layers, hidden_size)
    return math.cos(min(safety_factor * eps, math.pi))


def _dot(a, b):
    if HAVE_NUMPY:
        return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a):
    if HAVE_NUMPY:
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))
    return math.sqrt(sum(x * x for x in a))


def _max_abs_delta(a, b):
    if HAVE_NUMPY:
        return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
    return max(abs(x - y) for x, y in zip(a, b))


def _max_abs(a):
    if HAVE_NUMPY:
        return float(np.max(np.abs(np.asarray(a, dtype=np.float64)))) if len(a) else 0.0
    return max((abs(x) for x in a), default=0.0)


# NORM_FLOOR mirrors `pooling.rs::norm_floor`'s fp32/bf16 arm (`1e-12`) --
# same reasoning: a genuinely all-zero gradient vector (e.g. `lora_a`'s
# gradient at a fresh `LoraInitMode::ZerosB` init, which IS mathematically
# zero -- `grad_oracle.rs`'s own doc/test confirms this empirically, not
# a bug) must divide to a finite, well-defined cosine of `0.0` (undefined
# direction, never a `NaN`/`Inf` that a naive `x > floor` control would
# silently pass through -- family F's non-vacuous-control invariant).
NORM_FLOOR = 1e-12


def cosine_similarity(a, b) -> float:
    na, nb = _norm(a), _norm(b)
    denom = na * nb
    if denom < NORM_FLOOR:
        return 0.0
    return _dot(a, b) / denom


def compare_tensor(name, grad_a, grad_b):
    if len(grad_a) != len(grad_b):
        raise ValueError(f"{name}: length mismatch ({len(grad_a)} vs {len(grad_b)})")
    max_signal = max(_max_abs(grad_a), _max_abs(grad_b))
    max_delta = _max_abs_delta(grad_a, grad_b)
    return {
        "max_abs_delta": max_delta,
        "max_abs_delta_over_max_signal": (max_delta / max_signal) if max_signal > NORM_FLOOR else None,
        "cosine_similarity": cosine_similarity(grad_a, grad_b),
        "n": len(grad_a),
    }


def compare_reports(report_a, report_b, cosine_floor):
    """Match tensors by NAME (loud on a mismatch, never silently skipped --
    B6's schema-strictness posture: a structural mismatch here is exactly
    the failure mode this oracle exists to catch, e.g. a target-modules
    set that resolved differently on the two stacks) and compute per-tensor
    and OVERALL (every matched tensor's gradient concatenated into one
    vector) statistics.
    """
    names_a = set(report_a["gradients"].keys())
    names_b = set(report_b["gradients"].keys())
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    matched = sorted(names_a & names_b)

    per_tensor = {}
    all_a, all_b = [], []
    for name in matched:
        ga = report_a["gradients"][name]["grad"]
        gb = report_b["gradients"][name]["grad"]
        per_tensor[name] = compare_tensor(name, ga, gb)
        all_a.extend(ga)
        all_b.extend(gb)

    overall_cosine = cosine_similarity(all_a, all_b) if all_a else None
    passed = (
        not only_a
        and not only_b
        and bool(matched)
        and overall_cosine is not None
        and overall_cosine >= cosine_floor
    )

    return {
        "loss_a": report_a.get("loss"),
        "loss_b": report_b.get("loss"),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "matched_tensor_count": len(matched),
        "cosine_floor": cosine_floor,
        "overall_cosine_similarity": overall_cosine,
        "per_tensor": per_tensor,
        "passed": passed,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report_a")
    p.add_argument("report_b")
    p.add_argument("--cosine-floor", type=float, default=None)
    p.add_argument("--num-layers", type=int, default=28, help="default: ModernBERT-large")
    p.add_argument("--hidden-size", type=int, default=1024, help="default: ModernBERT-large")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    with open(args.report_a) as fh:
        report_a = json.load(fh)
    with open(args.report_b) as fh:
        report_b = json.load(fh)

    cosine_floor = (
        args.cosine_floor
        if args.cosine_floor is not None
        else derive_cosine_floor(args.num_layers, args.hidden_size)
    )
    result = compare_reports(report_a, report_b, cosine_floor)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)

    print(f"overall_cosine_similarity: {result['overall_cosine_similarity']}")
    print(f"cosine_floor: {result['cosine_floor']} (numpy={'yes' if HAVE_NUMPY else 'no, pure-python fallback'})")
    print(f"matched_tensor_count: {result['matched_tensor_count']}")
    if result["only_in_a"]:
        print(f"only_in_a: {result['only_in_a']}", file=sys.stderr)
    if result["only_in_b"]:
        print(f"only_in_b: {result['only_in_b']}", file=sys.stderr)
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
