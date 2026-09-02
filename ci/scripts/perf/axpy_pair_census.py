#!/usr/bin/env python3
"""Campaign #446 W2-A: the axpy-SHAPED PAIR share `P`, differenced over the
same `(N, M)` same-workload pair `kernel_census.py` differences, plus the
fused-vs-eager AdamW SANITY check each leg's validity depends on.

`kernel_census.py` attributes KERNELS; this attributes ADJACENCY. The
pre-registered rule (`decision_rule` in the committed artifact, written
before any number was measured) defines an axpy-shaped PAIR as

    an `affine_*` launch whose output feeds a `badd_*`/`bsub_*` on
    identically-shaped operands

so a count-only match ("there are 173 affines and 858 adds, so up to 173
could pair") is NOT the measurement — it is an upper bound that assumes the
conclusion. This module confirms the pairing from the nsys sqlite's own
KERNEL START ORDERING: within one `(device, context, stream)`, kernels are
serialized, so the immediate successor of an `affine_*` launch is the launch
that consumed its output if anything did. A pair is counted only when

  1. the successor (by start, same stream) is a `badd_*`/`bsub_*`, AND
  2. its `(gridX, gridY, gridZ, blockX, blockY, blockZ)` signature is
     IDENTICAL to the affine's — an elementwise launch's grid is a function
     of its element count, so an equal signature is the census-visible
     witness of "identically-shaped operands", and
  3. both are the same dtype suffix family (`_f32`/`_bf16`/`_f16`), since
     `Axpy` is same-dtype by domain (`ops/axpy.rs`).

`P` = (paired affine time + paired add time) per step / `gpu_kernel_us_per_
step`. BOTH launches are counted because that is exactly what one `Axpy`
launch would replace: one `affine(alpha)` + one `badd` (`ops/axpy.rs`'s own
"It fuses exactly one `affine(alpha)` + one `badd` — two launches and one
extra memory pass — into one launch").

## Differencing, not raw totals

Every quantity is the `(M-N)` DIFFERENCE of the two exports divided by
`M-N`, the same isolation `kernel_census.py` performs: model load, the init
probe, the held-out eval and checkpointing are identical across the pinned
pair and cancel. A raw single-run pair count would fold all of that in.

## Refusals (the validity contract — this script REFUSES, never degrades)

  - either export lacking / carrying an EMPTY `CUPTI_ACTIVITY_KIND_KERNEL`
    table (the same instrument failure `kernel_census.py` refuses on);
  - a NEGATIVE differenced pair count or time beyond tolerance — the
    signature of two exports that are not the declared same-workload pair;
  - any headline number that is not FINITE. A non-finite `P` must never
    reach a threshold comparison: `float("nan") >= 0.02` is `False`, so a
    diverged run would silently read as "below threshold, DELETE" — a
    negative control that passes on every way the bad path can fail has to
    reject non-finite explicitly, not rely on the comparison;
  - a leg whose declared arm and MEASURED AdamW dispatch counters disagree
    (see `sanity` below). A `P` computed on a leg that did not actually run
    the arm it claims is a number about nothing.

## The AdamW sanity check

`AdamW::step` dispatches per-`Var`: the fused arm is ONE
`adamw_step_fused_t` call = THREE launches (two `InplaceOp2` moment updates
-> `adamw_moment_update_f32`, one `InplaceOp3` theta update ->
`adamw_theta_update_f32`); the eager arm is `step_eager_one`'s candle chain
(`affine_*` x8, `badd_*` x2, `bsub_*`, `bdiv_*`, `usqr`, `usqrt`) plus three
`Var::set` copies. `V`, the number of trainable `Var`s stepped per step, is
MEASURED, not assumed: it is the differenced `adamw_fused_dispatches` (or
`adamw_eager_dispatches`) from the two runs' own report JSONs, divided by
`M-N`. The check then asserts the census actually shows

  - fused legs: `adamw_moment_update_f32` ~ `2V` launches/step and
    `adamw_theta_update_f32` ~ `1V`, eager dispatches == 0;
  - the eager control leg: BOTH fused kernels ABSENT, fused dispatches == 0,
    and eager dispatches == `V` per step.

`--sanity-rel-tolerance` bounds the launch-count agreement (default 5%): the
counters and the trace are two independent instruments, so an exact integer
match is not guaranteed across a differenced pair, but a factor-level
disagreement is a real INVALID.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys

# candle's elementwise CUDA kernels are named `<op>_<dtype>`; the dtype
# suffixes this profile can see (the three shipped backbone dtypes plus the
# f32 the LoRA `Var`s and the optimizer state always are).
_DTYPE_SUFFIXES = ("f32", "bf16", "f16", "f64", "u8", "u32", "i64")

_AFFINE_RE = re.compile(r"^affine_(" + "|".join(_DTYPE_SUFFIXES) + r")$")
_ADD_RE = re.compile(r"^(badd|bsub)_(" + "|".join(_DTYPE_SUFFIXES) + r")$")

ADAMW_FUSED_KERNELS = ("adamw_moment_update_f32", "adamw_theta_update_f32")
# `step_eager_one`'s own candle chain, by kernel family (see that fn).
EAGER_CHAIN_FAMILIES = ("affine", "badd", "bsub", "bdiv", "usqr", "usqrt")


class Refusal(Exception):
    """A validity-contract violation: refuse loudly, emit no report."""


def _finite(x: float, what: str) -> float:
    """Every headline number passes through here.

    `NaN`/`inf` must never reach a threshold comparison: `nan >= 0.02` is
    `False`, so a diverged leg would read as a clean below-threshold DELETE.
    """
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise Refusal(f"{what} is not finite ({x!r}) — refusing to report a non-finite measurement")
    return float(x)


def _kernels(path: str) -> list[tuple]:
    """Every kernel launch, ordered by stream then start.

    Returns `(device, context, stream, start, end, name, gx, gy, gz, bx, by, bz)`.
    """
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        present = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
        ).fetchone()
        if not present:
            raise Refusal(
                f"{path}: no CUPTI_ACTIVITY_KIND_KERNEL table in this nsys sqlite export — "
                "the export never recorded a kernel-level trace (leg INVALID)"
            )
        rows = cur.execute(
            """SELECT k.deviceId, k.contextId, k.streamId, k.start, k.end, s.value,
                      k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
               ORDER BY k.deviceId, k.contextId, k.streamId, k.start"""
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise Refusal(
            f"{path}: CUPTI_ACTIVITY_KIND_KERNEL is PRESENT but has zero rows — the same "
            "instrument failure as a missing table (leg INVALID)"
        )
    return rows


def scan_pairs(rows: list[tuple]) -> tuple[dict, dict, dict]:
    """Immediate-successor axpy-shaped pairs, plus the per-kernel totals.

    Returns `(pairs_by_key, kernel_counts, kernel_time_ns)`.
    `pairs_by_key[(affine_name, add_name, grid, block)] = [count, time_ns]`
    where `time_ns` is BOTH launches' duration (what one `Axpy` replaces).
    """
    pairs: dict[tuple, list] = {}
    counts: dict[str, int] = {}
    times: dict[str, int] = {}
    for i, r in enumerate(rows):
        name = r[5]
        counts[name] = counts.get(name, 0) + 1
        times[name] = times.get(name, 0) + (r[4] - r[3])

        m = _AFFINE_RE.match(name)
        if not m or i + 1 >= len(rows):
            continue
        nxt = rows[i + 1]
        # Same (device, context, stream): only within one stream is
        # "immediate successor" a serialization fact rather than an artifact
        # of interleaving two independent streams in the global ordering.
        if r[0:3] != nxt[0:3]:
            continue
        n = _ADD_RE.match(nxt[5])
        if not n:
            continue
        if n.group(2) != m.group(1):  # same dtype family (Axpy is same-dtype by domain)
            continue
        if r[6:12] != nxt[6:12]:  # identical (grid, block) => identically-shaped operands
            continue
        key = (name, nxt[5], tuple(r[6:9]), tuple(r[9:12]))
        e = pairs.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += (r[4] - r[3]) + (nxt[4] - nxt[3])
    return pairs, counts, times


def _tier(path: str) -> dict:
    d = json.load(open(path))
    t = d.get("tiers", {}).get("finetune_run")
    if t is None:
        raise Refusal(f"{path}: no tiers.finetune_run in the run report")
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite-a", required=True)
    ap.add_argument("--sqlite-b", required=True)
    ap.add_argument("--steps-a", type=int, required=True)
    ap.add_argument("--steps-b", type=int, required=True)
    ap.add_argument("--census", required=True, help="this leg's kernel_census.py report")
    ap.add_argument("--run-json-a", required=True)
    ap.add_argument("--run-json-b", required=True)
    ap.add_argument("--leg-id", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--disable-keys", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count-tolerance", type=float, default=1.0)
    ap.add_argument("--sanity-rel-tolerance", type=float, default=0.05)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if args.steps_b <= args.steps_a:
            raise Refusal(f"steps_b ({args.steps_b}) must exceed steps_a ({args.steps_a})")
        d = args.steps_b - args.steps_a

        rows_a = _kernels(args.sqlite_a)
        rows_b = _kernels(args.sqlite_b)
        pa, ca, ta = scan_pairs(rows_a)
        pb, cb, tb = scan_pairs(rows_b)

        census = json.load(open(args.census))
        gpu_us = _finite(census["gpu_kernel_us_per_step"], "gpu_kernel_us_per_step")
        if gpu_us <= 0:
            raise Refusal(f"gpu_kernel_us_per_step is {gpu_us} — cannot form a share against it")

        # --- differenced pair table -------------------------------------
        pair_rows = []
        total_us = 0.0
        for key in set(pa) | set(pb):
            na, sa = pa.get(key, (0, 0))
            nb, sb = pb.get(key, (0, 0))
            dn, dns = nb - na, sb - sa
            if dn <= args.count_tolerance:
                # No differencing-relevant work in this bucket (fixed cost:
                # the init probe / held-out eval launch it identically in
                # both runs). Excluded, exactly as kernel_census.py excludes
                # a non-positive count delta — never divided by `d`, which
                # would emit a negative launches/step into the headline.
                continue
            if dns < 0:
                raise Refusal(
                    f"pair bucket {key} has a POSITIVE count delta ({dn}) but a NEGATIVE time "
                    f"delta ({dns} ns) — the two exports are not the declared same-workload pair"
                )
            us = _finite(dns / d / 1000.0, f"pair {key} us_per_step")
            total_us += us
            pair_rows.append(
                {
                    "affine_kernel": key[0],
                    "add_kernel": key[1],
                    "grid": list(key[2]),
                    "block": list(key[3]),
                    "elements_hint": key[2][0] * key[2][1] * key[2][2] * key[3][0] * key[3][1] * key[3][2],
                    "pairs_per_step": _finite(dn / d, "pairs_per_step"),
                    "us_per_step": us,
                }
            )
        pair_rows.sort(key=lambda r: -r["us_per_step"])
        total_us = _finite(total_us, "paired us_per_step total")
        for r in pair_rows:
            r["share_of_P"] = r["us_per_step"] / total_us if total_us else 0.0

        p_frac = _finite(total_us / gpu_us, "P")

        # --- AdamW sanity (the leg's own validity) ----------------------
        ta_j, tb_j = _tier(args.run_json_a), _tier(args.run_json_b)
        fused_d = tb_j["adamw_fused_dispatches"] - ta_j["adamw_fused_dispatches"]
        eager_d = tb_j["adamw_eager_dispatches"] - ta_j["adamw_eager_dispatches"]
        if fused_d < 0 or eager_d < 0:
            raise Refusal(
                f"differenced AdamW dispatch counters are negative (fused {fused_d}, eager "
                f"{eager_d}) — not a same-workload pair"
            )
        expect_eager = "adamw_step_fused" in [k for k in args.disable_keys.split(",") if k]

        def per_step(name: str) -> float:
            dn = cb.get(name, 0) - ca.get(name, 0)
            return _finite(dn / d, f"{name} launches_per_step")

        moment = per_step("adamw_moment_update_f32")
        theta = per_step("adamw_theta_update_f32")
        sanity: dict = {
            "arm_expected": "eager" if expect_eager else "fused",
            "adamw_fused_dispatches_per_step": _finite(fused_d / d, "fused dispatches/step"),
            "adamw_eager_dispatches_per_step": _finite(eager_d / d, "eager dispatches/step"),
            "adamw_moment_update_f32_launches_per_step": moment,
            "adamw_theta_update_f32_launches_per_step": theta,
            "eager_chain_launches_per_step": {
                fam: _finite(
                    sum(
                        cb.get(k, 0) - ca.get(k, 0)
                        for k in set(ca) | set(cb)
                        if k.startswith(fam + "_") or k == fam
                    )
                    / d,
                    f"{fam} launches_per_step",
                )
                for fam in EAGER_CHAIN_FAMILIES
            },
            "violations": [],
        }
        v = sanity["violations"]
        if expect_eager:
            trainable_vars = _finite(eager_d / d, "trainable Vars per step (eager)")
            if fused_d != 0:
                v.append(f"eager CONTROL leg dispatched the FUSED arm {fused_d} times")
            if eager_d == 0:
                v.append("eager CONTROL leg recorded ZERO eager AdamW dispatches")
            for k, n in (("adamw_moment_update_f32", moment), ("adamw_theta_update_f32", theta)):
                if n > args.count_tolerance / d:
                    v.append(f"eager CONTROL leg still launched {k} at {n:.3f}/step — must be ABSENT")
            for fam in ("affine", "badd", "bsub", "bdiv", "usqr", "usqrt"):
                if sanity["eager_chain_launches_per_step"][fam] <= 0:
                    v.append(f"eager CONTROL leg shows NO {fam}_* launches — the eager chain is absent")
        else:
            trainable_vars = _finite(fused_d / d, "trainable Vars per step (fused)")
            if eager_d != 0:
                v.append(f"SHIPPED leg fell back to the eager AdamW arm {eager_d} times")
            if fused_d == 0:
                v.append("SHIPPED leg recorded ZERO fused AdamW dispatches")
            for k, n, mult in (
                ("adamw_moment_update_f32", moment, 2.0),
                ("adamw_theta_update_f32", theta, 1.0),
            ):
                want = mult * trainable_vars
                if want <= 0 or abs(n - want) > args.sanity_rel_tolerance * want:
                    v.append(
                        f"{k} launched {n:.3f}/step, expected ~{want:.3f} "
                        f"({mult:g} x {trainable_vars:.3f} trainable Vars/step)"
                    )
        sanity["trainable_vars_per_step"] = trainable_vars
        sanity["ok"] = not v

        report = {
            "leg_id": args.leg_id,
            "dtype": args.dtype,
            "kernels_disable_keys": args.disable_keys,
            "steps_a": args.steps_a,
            "steps_b": args.steps_b,
            "steps_diff": d,
            "gpu_kernel_us_per_step": gpu_us,
            "wall_s_per_step": census.get("wall_s_per_step"),
            "launches_per_step": census.get("launches_per_step"),
            "axpy_pair_us_per_step": total_us,
            "P": p_frac,
            "P_percent": p_frac * 100.0,
            "pairs": pair_rows,
            "sanity": sanity,
        }
        if not sanity["ok"]:
            print(
                "::error::axpy_pair_census: leg "
                + args.leg_id
                + " FAILED the AdamW sanity check: "
                + "; ".join(v),
                file=sys.stderr,
            )
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(
            f"{args.leg_id}: P={p_frac * 100:.3f}% "
            f"(paired {total_us:.1f} us/step of {gpu_us:.1f} us/step), "
            f"{len(pair_rows)} pair bucket(s), sanity_ok={sanity['ok']}"
        )
        return 0 if sanity["ok"] else 4
    except Refusal as e:
        print(f"::error::axpy_pair_census: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
