#!/usr/bin/env python3
"""Inflate one committed same-box rate baseline 100x, in place.

The teeth-proof half of the workflow-layer rate gate (perf.yml): with the named
baseline rate inflated 100x the derived floor (`baseline * (1 - 0.30)`) is
unreachable, so the tier MUST exit non-zero — proving the exit-code gate is
wired through, not decorative. The caller perturbs an ephemeral CI checkout
only and restores the file afterwards.

Usage: inflate_baseline.py <path-to-baseline.json> <rate-field>
"""

import json
import sys

path, field = sys.argv[1], sys.argv[2]
d = json.load(open(path))
d[field] *= 100
json.dump(d, open(path, "w"), indent=2)
print(f"inflated {field} to {d[field]} (floor now unreachable)")
