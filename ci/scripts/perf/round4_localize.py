#!/usr/bin/env python3
"""esc-045 (GH#374) round 4/5 per-layer activation-gradient localization
comparator.

## What this replaces (A3, GH#374 phase-4 audit)

Round 5's E2 finding (`crates/jammi-kernels/artifacts/cuda-runs/
2026-08-25-esc045-r5-725d116-a100-pcie.json`'s `E2_attention_logit_
precision_mechanism`, `e2_full_table.txt`'s 84-row cos/relerr table) was
produced by an EPHEMERAL, never-committed script during that round's pod
session -- the artifact's `shared_upcast` row therefore had no committed
PRODUCER, and the comparator itself could not be re-run against a fresh
dump. This file is that comparator, committed for the first time; see
`crates/jammi-encoders/src/modernbert.rs`'s `round5_maybe_upcast_scores`
for the `shared_upcast` arm's own first committed producer.

## Input format

Reads N safetensors dumps, one per attention ARM, all produced by
`esc045_round4_per_layer_activation_gradient_dump`
(`crates/jammi-encoders/src/modernbert.rs`) against IDENTICAL LoRA weights
and an identical batch -- only the admission-cascade DISABLE list (and,
for `shared_upcast`, `JAMMI_ROUND5_UPCAST_SCORES`) differs between arms;
see that Rust test's own doc for the exact per-arm env-var recipe. Each
dump is keyed `boundary.{i}` / `qkv.{i}` / `mlp_input.{i}` (`i` a layer
index), every value an `F32` tensor at the real captured
activation-gradient shape (`dL/d(that activation)`).

## What this reports

Per captured tensor (every key present in EVERY supplied dump, including
the designated "truth" one): `cos(arm_flat, truth_flat)`, using the SAME
`cosine_similarity`/`_dot`/`_norm`/`NORM_FLOOR` this directory's
`compare_grad_oracle.py` already defines and this file imports rather than
re-derives (one definition, family F: a numeric primitive computed once,
never forked). Plus, per arm, the L2 RELATIVE ERROR of the FULL
concatenated vector across every matched tensor (deterministic
concatenation order: sorted by key name) against the truth arm's
concatenation -- `crates/jammi-kernels/artifacts/cuda-runs/
2026-08-25-esc045-r5-725d116-a100-pcie.json`'s `summary_all_84_relerr`
field's exact metric.

## Dependencies

Pure Python + the standard library for BOTH the safetensors parsing
(`_read_safetensors_header`/`load_f32_dump` hand-parse the format's own
8-byte-length-prefixed-JSON-header-then-raw-bytes spec directly -- no
`safetensors`/`torch` package needed) and the aggregation math (numpy used
when available via `compare_grad_oracle`'s own `HAVE_NUMPY` dual-path
convention, but never required) -- so `test_round4_localize.py` can run,
and does, in an environment with NONE of numpy/torch/safetensors
installed, the same guarantee `compare_grad_oracle.py`'s own self-test
carries (family F: a comparator's own math must be independently
testable).

Usage:
    python3 round4_localize.py --truth f32_truth.safetensors \\
        --arm fused=fused.safetensors --arm eager=eager.safetensors \\
        [--arm shared=shared.safetensors --arm shared_upcast=su.safetensors] \\
        [--band-split 18] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grad_oracle import (  # noqa: E402
    HAVE_NUMPY,
    NORM_FLOOR,
    _dot,
    _norm,
    cosine_similarity,
)

if HAVE_NUMPY:
    import numpy as np  # type: ignore


# ---------------------------------------------------------------------
# Pure-Python safetensors reader (F32 tensors only -- this comparator's
# entire domain, since the round-4 dump always writes F32; a tensor of any
# other dtype in a supplied file is a typed refusal, not a silent
# misread).
# ---------------------------------------------------------------------


def _read_safetensors_header(data: bytes) -> tuple[dict, int]:
    if len(data) < 8:
        raise ValueError(f"file too short to be safetensors ({len(data)} bytes)")
    (header_len,) = struct.unpack("<Q", data[:8])
    if 8 + header_len > len(data):
        raise ValueError(
            f"safetensors header claims {header_len} bytes but file has only "
            f"{len(data) - 8} bytes after the length prefix"
        )
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    return header, 8 + header_len


def load_f32_dump(path: str) -> dict[str, list[float]]:
    with open(path, "rb") as fh:
        data = fh.read()
    header, data_start = _read_safetensors_header(data)
    out: dict[str, list[float]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype")
        if dtype != "F32":
            raise ValueError(
                f"{path}: tensor {name!r} is {dtype!r}, expected F32 (this comparator's "
                "entire domain -- the round-4 dump always writes F32 activation gradients)"
            )
        start, end = meta["data_offsets"]
        raw = data[data_start + start : data_start + end]
        n = (end - start) // 4
        if len(raw) != n * 4:
            raise ValueError(
                f"{path}: tensor {name!r} data_offsets span {end - start} bytes, not a "
                "multiple of 4 (F32 element size)"
            )
        if HAVE_NUMPY:
            out[name] = np.frombuffer(raw, dtype="<f4", count=n).astype("float64").tolist()
        else:
            out[name] = list(struct.unpack(f"<{n}f", raw)) if n else []
    return out


# ---------------------------------------------------------------------
# Aggregation math -- operates on ALREADY-LOADED dicts (never touches a
# file directly), so the self-test exercises this in isolation from the
# safetensors parser above.
# ---------------------------------------------------------------------


def matched_keys(arms: dict[str, dict], truth: dict) -> list[str]:
    """Keys present in EVERY supplied arm AND `truth` -- never assumed
    from a fixed layer-index range (family D: derive the domain from what
    is actually present, don't hardcode `0..N`)."""
    keys = set(truth.keys())
    for d in arms.values():
        keys &= set(d.keys())
    return sorted(keys)


def per_tensor_cosines(
    arms: dict[str, dict], truth: dict, keys: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """`{key: {arm_name: cos(arm[key], truth[key])}}` for every key in
    `keys` (default: `matched_keys(arms, truth)`)."""
    if keys is None:
        keys = matched_keys(arms, truth)
    rows: dict[str, dict[str, float]] = {}
    for key in keys:
        t = truth[key]
        rows[key] = {name: cosine_similarity(d[key], t) for name, d in arms.items()}
    return rows


def concatenated_relerr(arms: dict[str, dict], truth: dict, keys: list[str] | None = None) -> dict[str, float]:
    """Per arm, the L2 relative error of the FULL concatenated vector
    (every matched tensor, sorted-key order -- deterministic, family J)
    against `truth`'s own concatenation: `||arm - truth|| /
    max(||truth||, NORM_FLOOR)`."""
    if keys is None:
        keys = matched_keys(arms, truth)
    truth_concat: list[float] = []
    for key in keys:
        truth_concat.extend(truth[key])
    truth_norm = _norm(truth_concat)
    denom = max(truth_norm, NORM_FLOOR)
    out: dict[str, float] = {}
    for name, d in arms.items():
        arm_concat: list[float] = []
        for key in keys:
            arm_concat.extend(d[key])
        if HAVE_NUMPY:
            diff = (np.asarray(arm_concat, dtype="float64") - np.asarray(truth_concat, dtype="float64")).tolist()
        else:
            diff = [a - b for a, b in zip(arm_concat, truth_concat)]
        out[name] = _norm(diff) / denom
    return out


def _category_and_index(key: str) -> tuple[str, int]:
    category, _, idx = key.rpartition(".")
    return category, int(idx)


def band_means(
    rows: dict[str, dict[str, float]], arm_names: list[str], band_split: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Per category, per arm: mean cos for layer index `< band_split`
    ("low", the esc-045 hotspot band in every historical measurement) and
    `>= band_split` ("high"). A single `band_split` applied uniformly
    across categories -- the historical artifact used a per-category
    off-by-one split (`boundary` 1-18 vs 19-28, `qkv` 0-18 vs 19-27);
    this function's single-parameter convention is a disclosed
    simplification of that, not a re-derivation of the exact historical
    bands."""
    by_category: dict[str, list[tuple[int, dict[str, float]]]] = {}
    for key, cos_by_arm in rows.items():
        category, idx = _category_and_index(key)
        by_category.setdefault(category, []).append((idx, cos_by_arm))

    out: dict[str, dict[str, dict[str, float]]] = {}
    for category, entries in by_category.items():
        low = [c for idx, c in entries if idx < band_split]
        high = [c for idx, c in entries if idx >= band_split]
        out[category] = {
            "low": _mean_per_arm(low, arm_names),
            "high": _mean_per_arm(high, arm_names),
        }
    return out


def _mean_per_arm(entries: list[dict[str, float]], arm_names: list[str]) -> dict[str, float]:
    if not entries:
        return {name: float("nan") for name in arm_names}
    return {name: sum(e[name] for e in entries) / len(entries) for name in arm_names}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_report(
    arms: dict[str, dict], truth: dict, band_split: int
) -> dict:
    keys = matched_keys(arms, truth)
    if not keys:
        raise ValueError(
            "no key is present in every supplied arm AND --truth -- nothing to compare "
            "(check that every dump came from the same run: same num_hidden_layers, same "
            "activation_capture keys)"
        )
    rows = per_tensor_cosines(arms, truth, keys)
    relerr = concatenated_relerr(arms, truth, keys)
    arm_names = sorted(arms.keys())
    bands = band_means(rows, arm_names, band_split)
    return {
        "keys": keys,
        "arm_names": arm_names,
        "cos_per_tensor": rows,
        "band_means": bands,
        "all_matched_relerr": relerr,
    }


def format_table(report: dict) -> str:
    lines: list[str] = []
    arm_names = report["arm_names"]
    by_category: dict[str, list[str]] = {}
    for key in report["keys"]:
        category, _ = _category_and_index(key)
        by_category.setdefault(category, []).append(key)
    for category, keys in by_category.items():
        keys_sorted = sorted(keys, key=lambda k: _category_and_index(k)[1], reverse=True)
        lines.append(f"\n## {category} -- cos vs truth, per arm (backward order)")
        header = f"{'name':>14}" + "".join(f"{name:>16}" for name in arm_names)
        lines.append(header)
        for key in keys_sorted:
            row = report["cos_per_tensor"][key]
            lines.append(f"{key:>14}" + "".join(f"{row[name]:16.6f}" for name in arm_names))
    lines.append("\n## all-matched relerr")
    for name in arm_names:
        lines.append(f"{name:<16} {report['all_matched_relerr'][name]:.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True, help="path to the F32-truth dump")
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="one attention arm's dump, repeatable (e.g. --arm fused=fused.safetensors)",
    )
    p.add_argument(
        "--band-split",
        type=int,
        default=18,
        help="layer index splitting 'low' (< split) from 'high' (>= split) bands; default 18 "
        "matches the historical esc-045 round-5 boundary-category split",
    )
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    arms: dict[str, dict] = {}
    for spec in args.arm:
        if "=" not in spec:
            print(f"REFUSED: --arm {spec!r} must be NAME=PATH", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        arms[name] = load_f32_dump(path)
    truth = load_f32_dump(args.truth)

    report = build_report(arms, truth, args.band_split)
    print(f"round4_localize: numpy={'yes' if HAVE_NUMPY else 'no, pure-python fallback'}")
    print(format_table(report))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
