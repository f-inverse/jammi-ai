#!/usr/bin/env python3
"""esc-045 round 6: numpy-first per-boundary/per-site comparator for the
safetensors dumps produced by jammi's `esc045_round4_per_layer_activation_gradient_dump`
(`boundary.{1..N}` / `qkv.{0..N-1}` / `mlp_input.{0..N-1}`, all dL/d(activation))
and `torch_round6_layer_dump.py` (`boundary.{1..N}` / `forward.{0..N}` /
`sublayer.{i}.{site}` / `sublayer.{i}.{site}.fwd`).

Deliberately Python + numpy (family F's numpy-first-oracle convention):
this is a SEPARATE codepath from both producers.

Usage: round6_layer_compare.py DUMP_A.safetensors DUMP_B.safetensors
    [--common-prefix boundary] [--out report.json]

Prints, per matched key: cos(a,b), max|a|, max|b|, relerr = ||a-b||/||b||
(ratio of L2 norms is also reported: ||a||/||b||). Never uses an absolute
ULP floor (guide 3.8) -- every number is relative.
"""
import argparse
import json
import sys

import numpy as np
from safetensors.numpy import load_file


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def relerr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / nb)


def norm_ratio(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return float("nan")
    return float(np.linalg.norm(a) / nb)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dump_a")
    p.add_argument("dump_b")
    p.add_argument("--prefix", action="append", default=None, help="only compare keys starting with this prefix (repeatable)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    a = load_file(args.dump_a)
    b = load_file(args.dump_b)
    common = sorted(set(a) & set(b))
    if args.prefix:
        common = [k for k in common if any(k.startswith(pfx) for pfx in args.prefix)]
    if not common:
        print(f"NO COMMON KEYS between {args.dump_a} and {args.dump_b} (prefixes={args.prefix})", file=sys.stderr)
        sys.exit(2)

    rows = []
    for k in common:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            rows.append({"key": k, "shape_mismatch": [list(ta.shape), list(tb.shape)]})
            continue
        finite_a = bool(np.isfinite(ta).all())
        finite_b = bool(np.isfinite(tb).all())
        row = {
            "key": k,
            "cos": cos(ta, tb),
            "relerr": relerr(ta, tb),
            "norm_ratio": norm_ratio(ta, tb),
            "max_abs_a": float(np.abs(ta).max()) if ta.size else 0.0,
            "max_abs_b": float(np.abs(tb).max()) if tb.size else 0.0,
            "finite_a": finite_a,
            "finite_b": finite_b,
        }
        rows.append(row)

    for r in rows:
        if "shape_mismatch" in r:
            print(f"{r['key']}: SHAPE MISMATCH {r['shape_mismatch']}")
        else:
            print(
                f"{r['key']}: cos={r['cos']:.4f} relerr={r['relerr']:.4f} "
                f"norm_ratio={r['norm_ratio']:.4f} finite={r['finite_a'] and r['finite_b']}"
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"dump_a": args.dump_a, "dump_b": args.dump_b, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
