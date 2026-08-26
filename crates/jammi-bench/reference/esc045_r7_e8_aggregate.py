#!/usr/bin/env python3
"""esc-045 round 7 (GH #374) E8 per-tensor aggregation.

Reads a `jammi-bench grad-oracle` dump, its `jammi_f32` truth dump, and
`ci/scripts/perf/compare_grad_oracle.py`'s own `--out` comparison JSON
(the `per_tensor` dict it emits -- this script does NOT recompute cosine
itself; it reuses that comparator's numbers, family F's "one numpy-first
oracle, not a second independently-drifting copy of the same arithmetic"),
and writes one row per matched LoRA tensor: `layer` / `module` (`Wi`,
`Wo`, `Wqkv`, `mlp.Wo`) / `ab` (`lora_a`/`lora_b`), parsed from the tensor
NAME jammi's own `grad_oracle.rs` emits (`layer.{L}.{module}.{lora_a|
lora_b}`); the comparator's own `cosine_similarity`/`vacuous`/
`one_sided_zero`/`has_nonfinite` per tensor; and this SCRIPT's own
`norm_dump`/`norm_truth`/`norm_ratio` (the gradient's L2 norm on each
side, from the two RAW dumps directly -- the comparator's own per-tensor
report does not carry a norm ratio field, only `max_abs_delta_over_
max_signal`).

Usage:
    esc045_r7_e8_aggregate.py DUMP.json TRUTH.json COMPARE.json OUT.json

`DUMP.json`/`TRUTH.json` are `jammi-bench grad-oracle --out` reports;
`COMPARE.json` is `compare_grad_oracle.py --out`'s report comparing them
(same two files, same order: `dump_a=DUMP`, `dump_b=TRUTH`).
"""

from __future__ import annotations

import json
import math
import re
import sys


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


TENSOR_NAME_RE = re.compile(r"^layer\.(\d+)\.(.+)\.(lora_a|lora_b)$")


def aggregate(dump_path: str, truth_path: str, compare_path: str) -> list[dict]:
    dump = json.load(open(dump_path))
    truth = json.load(open(truth_path))
    compare = json.load(open(compare_path))
    per_tensor = compare["per_tensor"]

    rows = []
    for name, stats in per_tensor.items():
        m = TENSOR_NAME_RE.match(name)
        if not m:
            print(f"esc045_r7_e8_aggregate: unparsed tensor name {name!r}", file=sys.stderr)
            continue
        layer, module, ab = int(m.group(1)), m.group(2), m.group(3)
        g_dump = dump["gradients"][name]["grad"]
        g_truth = truth["gradients"][name]["grad"]
        norm_dump = _norm(g_dump)
        norm_truth = _norm(g_truth)
        rows.append(
            {
                "name": name,
                "layer": layer,
                "module": module,
                "ab": ab,
                "cosine": stats["cosine_similarity"],
                "vacuous": stats["vacuous"],
                "one_sided_zero": stats["one_sided_zero"],
                "has_nonfinite": stats["has_nonfinite"],
                "norm_dump": norm_dump,
                "norm_truth": norm_truth,
                "norm_ratio": (norm_dump / norm_truth) if norm_truth > 1e-30 else None,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        print(__doc__)
        return 2
    dump_path, truth_path, compare_path, out_path = argv
    rows = aggregate(dump_path, truth_path, compare_path)
    json.dump(rows, open(out_path, "w"))
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
