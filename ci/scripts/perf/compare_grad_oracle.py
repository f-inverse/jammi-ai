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
    this assertion.

    THIS IS ONLY TRUE WHEN THE DERIVED FLOOR IS INFORMATIVE (`> 0`, in
    practice close to `1.0`). An earlier draft of this docstring claimed a
    REAL arithmetic defect (wrong sign on a whole tensor, a missing scale
    factor, a transposed axis) "produces an angle close to 90 degrees or
    worse, nowhere near this floor regardless of the safety factor" — that
    claim is FALSE at the depth/width this module's OWN default arguments
    select (`--num-layers 28 --hidden-size 1024`, ModernBERT-large):
    `eps = sqrt(28*1024) * 2^-8 ~= 0.661`, so `safety_factor * eps ~=
    1.984` radians, and `cos(1.984) ~= -0.402` — a floor BELOW zero, which
    a full 90-degree defect (cosine exactly `0.0`) clears with room to
    spare, and which even a WORSE-than-90-degree defect (cosine down to
    `-0.402`) still clears. `derive_relative_error_bound`'s own doc already
    discloses that its statistical bound is not tight at this depth; this
    function's docstring must not separately claim the resulting floor is
    always discriminating when it is not. See
    `test_compare_grad_oracle.py::test_derive_cosine_floor_is_non_positive_at_modernbert_large_defaults`
    for the pinned numeric reproduction, and `main()`'s own doc for the
    refusal this module now performs instead of silently comparing against
    a non-positive floor.

    A perturbation that genuinely IS small and isotropic (the ordinary
    bf16-rounding case this bound is meant to size) does satisfy
    `sin(theta) ~ eps` to first order (the perturbation's component
    ORTHOGONAL to `a`, relative to `|a|`, is what rotates the vector; a
    perturbation purely along `a` does not change direction at all) — the
    derivation above is correct AS A DERIVATION. What breaks at depth is
    only the SIZE of `eps` itself, which the safety-factor multiplication
    and `cos` can push at or past the point where the resulting floor stops
    discriminating between "rounding noise" and "wrong direction entirely".

    `eps` can legitimately exceed `pi` for a large `num_layers *
    hidden_size` (see `derive_relative_error_bound`'s own doc — the
    statistical bound is itself not tight at ModernBERT-large's full
    depth); `cos` is clamped to `[-1.0, 1.0]` via `math.cos`'s own
    range regardless, so this function always returns a value in
    `[-1.0, 1.0]`, never raises on a large `eps`. A caller whose derived
    floor comes out at or below 0.0 should treat the bound as
    UNINFORMATIVE at that depth and prefer a NAMED, empirically-set floor
    instead — this function never silently substitutes one; `main()`
    enforces this by REFUSING (non-zero exit, no `PASS` ever printed)
    rather than running a vacuous comparison.

    EMPIRICAL ANCHOR FOR CHOOSING THAT NAMED FLOOR: a live A100 run
    (ModernBERT-large, `--batch 8 --seq 128 --seed 42`, jammi tip
    `e62c8a8` — reported by the lead who dispatched that pod job, not
    reproduced by this module's own test suite, which has no GPU) measured
    these overall cosine similarities: torch-eager vs torch-sdpa `0.825`;
    torch-bf16 vs torch-f32 `0.924`; jammi-f32 vs torch-f32 `0.9999998`
    (near-perfect, as expected — f32 has no bf16 rounding to diverge on).
    A separately-introduced real defect on that same run scored `0.30` to
    `0.53`. So real bf16-through-a-28-layer-triplet-loss noise sits around
    `0.82`-`0.93`, nowhere near this function's derived `~-0.40` at these
    dimensions, and a real defect sits well below both — an explicit
    `--cosine-floor` picked from THIS band (e.g. `0.7`-`0.8`, between the
    noise band and the defect band) is what a real bf16 sweep should pass,
    not the derived worst-case bound, which this function's own doc above
    already says to treat as uninformative once it goes non-positive. This
    paragraph is a citation of an EXTERNALLY reported measurement, not a
    claim this module's local test suite verifies (no GPU here) — see
    `test_compare_grad_oracle.py` for what IS locally verified (the pure
    arithmetic these external numbers are consistent with, never the
    external numbers themselves).
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


def is_vacuous_pair(a, b) -> bool:
    """`True` iff BOTH `a` and `b` are (numerically) the zero vector.

    Confirmed live on a real A100 run (ModernBERT-large, tip e62c8a8): at a
    fresh `LoraInitMode::ZerosB` init, `dL/dA` is EXACTLY `0.0` on BOTH
    stacks for every `lora_a` tensor (`max|dL/dA| == 0` on all four dumps
    from that run) -- LoRA's forward is `base(x) + scaling *
    dropout(x @ A^T @ B^T)` (`lora_linear.rs`'s own doc), and with `B ==
    0` the chain rule's `B^T @ dL/d(output)` factor that backprops into
    `dL/dA` is the zero matrix, for ANY `A`. In that run this was HALF the
    matched tensors (112 of 224) — every `lora_a` entry, none of the
    `lora_b` ones. `cosine_similarity`'s `denom < NORM_FLOOR` branch
    already returns a well-defined `0.0` rather than `NaN` for this case
    (family F's non-vacuous-control invariant), but a bare `0.0` in
    `per_tensor[name]["cosine_similarity"]` does not, on its own, say
    WHETHER that `0.0` means "the two stacks disagree" or "neither stack
    has a signal here at all, by construction" — those are opposite
    conclusions from the same number. `compare_tensor`/`compare_reports`
    use this to classify and report that distinction explicitly rather
    than let a vacuous `0.0` masquerade as either a pass or a fail.
    """
    return _norm(a) < NORM_FLOOR and _norm(b) < NORM_FLOOR


def compare_tensor(name, grad_a, grad_b):
    if len(grad_a) != len(grad_b):
        raise ValueError(f"{name}: length mismatch ({len(grad_a)} vs {len(grad_b)})")
    max_signal = max(_max_abs(grad_a), _max_abs(grad_b))
    max_delta = _max_abs_delta(grad_a, grad_b)
    return {
        "max_abs_delta": max_delta,
        "max_abs_delta_over_max_signal": (max_delta / max_signal) if max_signal > NORM_FLOOR else None,
        "cosine_similarity": cosine_similarity(grad_a, grad_b),
        "vacuous": is_vacuous_pair(grad_a, grad_b),
        "n": len(grad_a),
    }


# Run-identity fields this comparator's premise (module docstring line 6:
# "IDENTICAL LoRA weights", plus an identical batch — see
# `_premise_violations`'s own doc) actually depends on. `lora_alpha` and
# `lora_dropout` are DELIBERATELY excluded: `lora_alpha` only rescales the
# LoRA delta by a constant both stacks apply identically post-matmul (a
# mismatch there would show up as a magnitude difference, which
# `max_abs_delta_over_max_signal` already surfaces, not a premise
# violation), and `lora_dropout` is unconditionally forced to `0.0` by both
# producers (`grad_oracle.rs`'s and `torch_grad_oracle.py`'s own module
# docs) so it can never legitimately differ.
RUN_IDENTITY_FIELDS = (
    "seed",
    "batch",
    "seq",
    "lora_rank",
    "target_modules",
    "batched_forward",
    "backbone_dtype",
)

# How tightly a `weight` array recorded by the two INDEPENDENT producers
# (jammi's `grad_oracle.rs`, torch's `torch_grad_oracle.py`) must agree to
# count as "the identical weight file". Both producers read the SAME
# safetensors file and re-serialize its values through JSON at f32/f64
# precision, so a genuine same-file load should agree far tighter than
# this — this bound is deliberately loose relative to that (not a fitted
# noise floor) so it never false-trips on JSON round-trip precision, while
# staying many orders of magnitude below a real content mismatch (the
# reproduction that motivated this check compared `weight = [0, 0, 0, 0]`
# against `[9, 9, 9, 9]`).
WEIGHT_MATCH_ATOL = 1e-4


def _weight_mismatches(report_a, report_b, matched_names):
    """Per-tensor `weight` agreement over the tensors matched by name.

    This is the check the comparator's own premise (module docstring line
    6, "IDENTICAL LoRA weights") depends on and, before this fix, never
    ran: both `grad_oracle.rs` and `torch_grad_oracle.py` dump the exact
    weight value the forward actually used specifically so a comparator
    can verify this — reading only `grad` and ignoring `weight` compares
    gradients that may have been taken at DIFFERENT weights and calls that
    a pass.
    """
    mismatches = []
    for name in matched_names:
        ta = report_a["gradients"][name]
        tb = report_b["gradients"][name]
        wa = ta.get("weight")
        wb = tb.get("weight")
        if not wa or not wb:
            mismatches.append(f"{name}: missing/empty 'weight' field in one or both dumps")
            continue
        if len(wa) != len(wb):
            mismatches.append(f"{name}: weight length mismatch ({len(wa)} vs {len(wb)})")
            continue
        delta = _max_abs_delta(wa, wb)
        if delta > WEIGHT_MATCH_ATOL:
            mismatches.append(
                f"{name}: weight mismatch, max|delta|={delta} (> {WEIGHT_MATCH_ATOL}) -- the two "
                "dumps were NOT produced from identical LoRA weights"
            )
    return mismatches


def _premise_violations(report_a, report_b):
    """Everything besides the gradient arrays themselves that this
    comparator's stated premise (module docstring line 6: "IDENTICAL LoRA
    weights", and an identical batch) depends on, and that
    `compare_reports` used to never look at: whether each side actually
    loaded a shared weight file at all, whether the two runs were
    configured identically (seed/shape/LoRA rank/target modules/forward
    mode/backbone dtype), and whether the two runs fed the encoder the
    SAME synthetic tokens (`batch_token_id_sums`, jammi's batch digest --
    see `grad_oracle.rs`'s field doc; `torch_grad_oracle.py` now emits the
    same field in the same schema).
    """
    violations = []
    for label, report in (("A", report_a), ("B", report_b)):
        if not report.get("lora_weights_in"):
            violations.append(
                f"report {label} records no loaded --lora-weights-in file -- this comparator's "
                "premise is IDENTICAL LoRA weights loaded from a SHARED file on both sides; a "
                "fresh/unloaded run compares gradients taken at DIFFERENT, independently-seeded "
                "weights, which this comparator cannot certify agrees with anything"
            )

    for field in RUN_IDENTITY_FIELDS:
        va = report_a.get(field)
        vb = report_b.get(field)
        if va != vb:
            violations.append(f"run-identity field {field!r} differs: A={va!r} B={vb!r}")

    sums_a = report_a.get("batch_token_id_sums")
    sums_b = report_b.get("batch_token_id_sums")
    if sums_a is None or sums_b is None:
        violations.append(
            "batch_token_id_sums missing from one or both dumps -- cannot verify the two runs "
            "fed the encoder the SAME synthetic batch"
        )
    elif list(sums_a) != list(sums_b):
        violations.append(f"batch_token_id_sums differ: A={sums_a!r} B={sums_b!r}")

    return violations


def compare_reports(report_a, report_b, cosine_floor):
    """Match tensors by NAME (loud on a mismatch, never silently skipped --
    B6's schema-strictness posture: a structural mismatch here is exactly
    the failure mode this oracle exists to catch, e.g. a target-modules
    set that resolved differently on the two stacks), verify the
    comparator's own premise (`_premise_violations`, `_weight_mismatches`
    -- IDENTICAL weights, identical batch, identical run configuration),
    and compute per-tensor and OVERALL (every matched tensor's gradient
    concatenated into one vector) gradient-direction statistics.

    `passed` requires ALL of: no name-set mismatch, at least one matched
    tensor, no premise violation, no weight mismatch, and the overall
    cosine similarity at or above `cosine_floor`. A premise violation or a
    weight mismatch forces `passed = False` regardless of how well the
    gradients happen to agree -- agreement computed at the WRONG premise
    (different weights, different batch, different config) proves nothing
    about the two stacks' arithmetic.

    VACUOUS TENSORS (confirmed live on a real A100 run: at a fresh
    `LoraInitMode::ZerosB` init, every `lora_a` tensor's gradient is
    EXACTLY zero on BOTH stacks -- see `is_vacuous_pair`'s own doc): these
    are counted in `vacuous_tensor_count`/named in `vacuous_tensor_names`,
    separately from `matched_tensor_count`, and are NEVER excluded from
    `overall_cosine_similarity`'s concatenated vector (a both-sides-zero
    segment contributes exactly `0` to both the dot product and each
    vector's sum-of-squares, so including vs excluding it is
    mathematically IDENTICAL -- this is a reporting/classification fix,
    not a correction to the overall statistic, which was never corrupted
    by these tensors). What was missing before this fix is visibility: a
    reader of `per_tensor[name]["cosine_similarity"] == 0.0` could not
    tell "these two stacks disagree here" apart from "this tensor has no
    signal on EITHER side, by construction, regardless of correctness" --
    opposite conclusions from the identical number. See
    `crates/jammi-bench/reference/README.md`'s own disclosure that a
    single fresh-init forward+backward tests ONLY `dL/dB`; `dL/dA`
    agreement (or disagreement) cannot be observed this way at all.
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

    vacuous_tensor_names = sorted(name for name in matched if per_tensor[name]["vacuous"])

    overall_cosine = cosine_similarity(all_a, all_b) if all_a else None
    weight_mismatches = _weight_mismatches(report_a, report_b, matched)
    premise_violations = _premise_violations(report_a, report_b)

    passed = (
        not only_a
        and not only_b
        and bool(matched)
        and overall_cosine is not None
        and overall_cosine >= cosine_floor
        and not weight_mismatches
        and not premise_violations
    )

    loss_a = report_a.get("loss")
    loss_b = report_b.get("loss")
    # INFORMATIONAL ONLY -- never gates `passed`. jammi and torch are
    # different arithmetic implementations of the same architecture; even
    # at identical weights and an identical batch, their losses are not
    # expected to be BIT-identical (different reduction/op order, same
    # "not associative" reasoning `finetune_step.rs`'s own
    # batched-vs-unbatched test documents), so asserting equality here
    # would risk a false FAIL on ordinary cross-framework rounding. It is
    # still computed and surfaced (previously dead-lettered: recorded in
    # the report, read by nothing) so a caller can eyeball whether it is
    # "small rounding noise" or "wildly different", the same qualitative
    # read the cosine floor's own derivation relies on.
    loss_relative_diff = None
    if loss_a is not None and loss_b is not None:
        denom = max(abs(loss_a), abs(loss_b), NORM_FLOOR)
        loss_relative_diff = abs(loss_a - loss_b) / denom

    return {
        "loss_a": loss_a,
        "loss_b": loss_b,
        "loss_relative_diff": loss_relative_diff,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "matched_tensor_count": len(matched),
        "vacuous_tensor_count": len(vacuous_tensor_names),
        "vacuous_tensor_names": vacuous_tensor_names,
        "cosine_floor": cosine_floor,
        "overall_cosine_similarity": overall_cosine,
        "per_tensor": per_tensor,
        "weight_mismatches": weight_mismatches,
        "premise_violations": premise_violations,
        "passed": passed,
    }


# Exit codes `main()` returns. `REFUSED` is deliberately distinct from
# `FAIL`: `FAIL` means the comparison RAN and did not pass (a real
# gradient/premise disagreement); `REFUSED` means the comparison never ran
# at all because the floor in use could not have discriminated a real
# defect from noise -- printing `PASS` (or even `FAIL`) in that state would
# claim a certification this invocation cannot back up. Never print `PASS`
# on a `REFUSED` exit.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report_a")
    p.add_argument("report_b")
    p.add_argument("--cosine-floor", type=float, default=None)
    p.add_argument("--num-layers", type=int, default=28, help="default: ModernBERT-large")
    p.add_argument("--hidden-size", type=int, default=1024, help="default: ModernBERT-large")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    cosine_floor = (
        args.cosine_floor
        if args.cosine_floor is not None
        else derive_cosine_floor(args.num_layers, args.hidden_size)
    )

    # REFUSE before even opening the report files: a floor `<= 0` cannot
    # discriminate a real arithmetic defect (which can itself read as
    # cosine `<= 0`, e.g. a 90-degree-or-worse rotation) from noise, so
    # attempting the comparison at all would be theater regardless of what
    # the two dumps contain. This enforces `derive_cosine_floor`'s own doc
    # ("A caller whose derived floor comes out at or below 0.0 should ...
    # prefer a NAMED, empirically-set floor instead") and closes the
    # reproduction where ModernBERT-large's OWN default `--num-layers
    # 28 --hidden-size 1024` derives a floor of ~-0.402 and two EXACTLY
    # ORTHOGONAL gradient vectors (cosine 0.0) print `PASS`.
    #
    # This also refuses an EXPLICIT `--cosine-floor` at or below `0.0`:
    # such a floor is uninformative regardless of where it came from (see
    # this module's docstring), and silently accepting one would just move
    # the same vacuous-pass hazard from the derived path to the explicit
    # one.
    if cosine_floor <= 0.0:
        source = "an explicit --cosine-floor" if args.cosine_floor is not None else (
            f"the DERIVED floor at --num-layers {args.num_layers} --hidden-size {args.hidden_size}"
        )
        print(
            f"REFUSED: cosine_floor = {cosine_floor} (<= 0.0), from {source}. A non-positive "
            "cosine floor cannot distinguish ordinary bf16 rounding noise from a real arithmetic "
            "defect (a 90-degree-or-worse rotation can itself score <= 0.0) -- this comparator "
            "will not run a comparison that cannot fail. Pass an explicit --cosine-floor > 0.0 "
            "(see derive_cosine_floor's docstring), or scope --num-layers/--hidden-size to the "
            "specific tensor(s) actually being checked rather than the whole-model default.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    with open(args.report_a) as fh:
        report_a = json.load(fh)
    with open(args.report_b) as fh:
        report_b = json.load(fh)

    result = compare_reports(report_a, report_b, cosine_floor)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)

    print(f"overall_cosine_similarity: {result['overall_cosine_similarity']}")
    print(f"cosine_floor: {result['cosine_floor']} (numpy={'yes' if HAVE_NUMPY else 'no, pure-python fallback'})")
    print(f"matched_tensor_count: {result['matched_tensor_count']}")
    print(
        f"vacuous_tensor_count: {result['vacuous_tensor_count']} "
        "(both sides exactly zero -- e.g. lora_a at a fresh ZerosB init; carries "
        "NO evidence about agreement either way; see is_vacuous_pair's doc)"
    )
    print(f"loss_a: {result['loss_a']}  loss_b: {result['loss_b']}  loss_relative_diff: {result['loss_relative_diff']}")
    if result["only_in_a"]:
        print(f"only_in_a: {result['only_in_a']}", file=sys.stderr)
    if result["only_in_b"]:
        print(f"only_in_b: {result['only_in_b']}", file=sys.stderr)
    for msg in result["premise_violations"]:
        print(f"PREMISE VIOLATION: {msg}", file=sys.stderr)
    for msg in result["weight_mismatches"]:
        print(f"WEIGHT MISMATCH: {msg}", file=sys.stderr)
    print("PASS" if result["passed"] else "FAIL")
    return EXIT_PASS if result["passed"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
