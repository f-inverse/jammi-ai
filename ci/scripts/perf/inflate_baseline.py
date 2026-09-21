#!/usr/bin/env python3
"""Inflate a committed graph-train baseline 100x, in place.

The teeth-proof half of the workflow-layer rate gate (perf.yml): with
`baseline_pairs_per_s` inflated 100x the derived floor (`baseline * (1 - 0.30)`)
is unreachable, so the tier MUST exit non-zero — proving the exit-code gate is
wired through, not decorative. The caller perturbs an ephemeral CI checkout
only and restores the file afterwards.

Usage: inflate_baseline.py <path-to-graph_train.json>
"""

import json
import sys

p = sys.argv[1]
d = json.load(open(p))
d["baseline_pairs_per_s"] *= 100
json.dump(d, open(p, "w"), indent=2)
print(f"inflated baseline_pairs_per_s to {d['baseline_pairs_per_s']} (floor now unreachable)")
