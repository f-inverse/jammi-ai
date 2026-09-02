#!/usr/bin/env python3
"""Timeline-neighbourhood evidence for campaign #446's axpy-shaped pairs.

`kernel_census.py` attributes KERNELS; a CALL SITE is attributed by reading
code. The gap between those two is where an attribution claim usually goes
unchecked ("these pairs are obviously the LoRA epilogue") — so this dumps
the evidence that makes the reading falsifiable instead of asserted: for
every axpy-shaped pair (the SAME detection rule `axpy_pair_census.py` uses,
so the two agree by construction on which launches are pairs), the ordered
kernel names immediately before and after it on the same stream, grouped
into distinct neighbourhood SIGNATURES with counts.

A signature is a fingerprint of the surrounding computation. It cannot name
a source line by itself, but it discriminates sharply between candidate
sites: a pair surrounded by `uexp/ulog/gather_u32/sa_u32/fast_sum` is in a
cross-entropy backward, not in an encoder block; one surrounded by
`ampere_*gemm/layer_norm/flash_*` is in the transformer stack. Attribution
is then the READING that explains the observed signature, and a reader who
disagrees can check the reading against the same signatures.

Purely descriptive: it refuses nothing and decides nothing. Its output is
evidence attached to a verdict that is reached elsewhere.
"""
import argparse
import collections
import json
import re
import sqlite3

DT = ("f32", "bf16", "f16", "f64", "u8", "u32", "i64")
AFF = re.compile(r"^affine_(" + "|".join(DT) + r")$")
ADD = re.compile(r"^(badd|bsub)_(" + "|".join(DT) + r")$")

ap = argparse.ArgumentParser()
ap.add_argument("--sqlite", required=True)
ap.add_argument("--before", type=int, default=6)
ap.add_argument("--after", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()

con = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
rows = con.execute(
    """SELECT k.deviceId,k.contextId,k.streamId,k.start,k.end,s.value,
              k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
       ORDER BY k.deviceId,k.contextId,k.streamId,k.start"""
).fetchall()
con.close()

sigs = collections.Counter()
by_shape = collections.defaultdict(collections.Counter)
for i, r in enumerate(rows):
    m = AFF.match(r[5])
    if not m or i + 1 >= len(rows):
        continue
    nx = rows[i + 1]
    if r[0:3] != nx[0:3]:
        continue
    n = ADD.match(nx[5])
    if not n or n.group(2) != m.group(1) or r[6:12] != nx[6:12]:
        continue
    lo = max(0, i - a.before)
    hi = min(len(rows), i + 2 + a.after)
    ctx = []
    for j in range(lo, hi):
        if rows[j][0:3] != r[0:3]:
            continue
        mark = ">>" if j in (i, i + 1) else "  "
        ctx.append(f"{mark}{rows[j][5]}[g{rows[j][6]}x{rows[j][7]}x{rows[j][8]}]")
    sig = " | ".join(ctx)
    shape = f"grid{list(r[6:9])}block{list(r[9:12])}"
    sigs[sig] += 1
    by_shape[shape][sig] += 1

out = {
    "sqlite": a.sqlite,
    "total_pairs": sum(sigs.values()),
    "distinct_signatures": len(sigs),
    "top_signatures": [{"count": c, "signature": s} for s, c in sigs.most_common(25)],
    "by_pair_shape": {
        k: {
            "count": sum(v.values()),
            "top_signatures": [{"count": c, "signature": s} for s, c in v.most_common(6)],
        }
        for k, v in sorted(by_shape.items(), key=lambda kv: -sum(kv[1].values()))
    },
}
json.dump(out, open(a.out, "w"), indent=1)
print(f"{a.sqlite}: {out['total_pairs']} pairs, {out['distinct_signatures']} distinct neighbourhoods")
for e in out["top_signatures"][:8]:
    print(f"  {e['count']:6d}  {e['signature'][:240]}")
