#!/usr/bin/env python3
"""`HOWWELL_DRY_RUN=1` runs of `runpod_gpu_howwell.sh`: the shares a pod count
splits the seeds into, where the control seeds land, and that nothing is
rented. Drives the real script; the RunPod library is sourced but never
called past its argument checks."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runpod_gpu_howwell.sh")


def dry_run(**env):
    with tempfile.TemporaryDirectory() as out:
        full = dict(os.environ, HOWWELL_DRY_RUN="1", HOWWELL_MODEL_DIR="/checkpoints/dry", HOWWELL_OBJECTIVE="mnrl",
                    HOWWELL_ARTIFACT_DIR=out, RUNPOD_API_KEY="dry", **env)
        proc = subprocess.run(["bash", SCRIPT], env=full, capture_output=True, text=True, timeout=120)
        shares = {}
        for path in sorted(os.listdir(out)):
            if path.startswith("share-") and path.endswith(".log"):
                with open(os.path.join(out, path)) as fh:
                    shares[path] = fh.read()
    return proc, shares


def share_lines(text):
    return re.findall(r"^--- share: seeds=(\S+) lr0=(\S+) lane_tests=(\S+) -> ", text, flags=re.M)


class FanOut(unittest.TestCase):
    def test_one_pod_runs_every_seed_and_judges_its_own_legs(self):
        proc, shares = dry_run()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(share_lines(proc.stdout), [("1,2,3,4,5,6,7,8,9,10,11,12", "none", "1")])
        self.assertNotIn("--- judge", proc.stdout)
        self.assertEqual(shares, {})

    def test_pods_split_the_seeds_round_robin_and_the_controls_stay_with_their_seeds(self):
        proc, shares = dry_run(HOWWELL_PODS="4", HOWWELL_LR0_SEEDS="1,2")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(share_lines(proc.stdout), [("1,5,9", "1", "1")])
        self.assertEqual(
            {name: share_lines(text)[0] for name, text in shares.items()},
            {"share-2.log": ("2,6,10", "2", "0"), "share-3.log": ("3,7,11", "none", "0"), "share-4.log": ("4,8,12", "none", "0")},
        )
        self.assertIn("--- judge: jammi-bench ladder train-run merged/raw --from torch --to resident", proc.stdout)

    def test_more_pods_than_seeds_is_refused(self):
        proc, _ = dry_run(HOWWELL_PODS="3", HOWWELL_SEEDS="1,2")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("exceeds the 2 seeds", proc.stderr)

    def test_a_dry_run_never_calls_the_api(self):
        proc, _ = dry_run(HOWWELL_PODS="2")
        self.assertNotIn("sweep", proc.stdout + proc.stderr)
        self.assertNotIn("provisioning a live", proc.stdout)


if __name__ == "__main__":
    unittest.main()
