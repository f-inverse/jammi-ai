#!/usr/bin/env python3
"""The shell producers of the ladder's GPU edges, under their dry-run flags:
each runs its legs in a balanced order, files them under the ladder's rung
names, and hands the directory to `jammi-bench ladder`. Nothing is built,
rented or measured.

Run: `python3 ci/scripts/perf/test_ladder_producers_dry_run.py`
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent


def dry_run(script: str, env: dict[str, str]) -> str:
    with tempfile.TemporaryDirectory() as out:
        full = {**os.environ, **env, "TORCH_VENV": os.path.join(out, "venv")}
        proc = subprocess.run(
            ["bash", str(PERF_DIR / script)],
            capture_output=True,
            text=True,
            env=full,
            cwd=out,
            timeout=120,
        )
    if proc.returncode != 0:
        raise AssertionError(f"{script} dry run exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


class TrainStepProducer(unittest.TestCase):
    def test_legs_run_balanced_per_shape_and_the_ladder_judges_them(self):
        out = dry_run(
            "finetune_step_ab.sh",
            {"FINETUNE_STEP_AB_DRY_RUN": "1", "FINETUNE_STEP_AB_SHAPES": "8:128:0,8:512:0.05"},
        )
        legs = re.findall(r"^--- (\S+): ", out, flags=re.M)
        self.assertEqual(
            legs,
            [
                "fused__b8s128d0__r1",
                "torch__b8s128d0__r1",
                "torch__b8s128d0__r2",
                "fused__b8s128d0__r2",
                "fused__b8s512d0p05__r1",
                "torch__b8s512d0p05__r1",
                "torch__b8s512d0p05__r2",
                "fused__b8s512d0p05__r2",
            ],
        )
        self.assertIn("ladder train-step", out)
        self.assertRegex(out, r"--axes speed\\?,space")
        # Every jammi leg claims the fused arm: nothing requested off.
        for argv in re.findall(r"^--- fused__\S+: (.*)$", out, flags=re.M):
            self.assertIn("--expect-kernels-disabled ''", argv)


class EncodeRevisionProducer(unittest.TestCase):
    def test_three_sides_run_balanced_and_the_revision_edge_is_judged(self):
        out = dry_run(
            "gpu_inference_ab.sh",
            {"GPU_INFERENCE_AB_DRY_RUN": "1", "GPU_INFERENCE_AB_SKIP_GPU_CHECK": "1"},
        )
        legs = re.findall(r"^--- (\S+): ", out, flags=re.M)
        self.assertEqual(
            legs,
            [
                "direct@base__rows256__r1",
                "direct@rebuilt__rows256__r1",
                "direct@revised__rows256__r1",
                "direct@revised__rows256__r2",
                "direct@rebuilt__rows256__r2",
                "direct@base__rows256__r2",
            ],
        )
        self.assertIn("ladder encode", out)
        self.assertIn("--revision direct", out)
        self.assertEqual(out.count("cargo build --release -p jammi-bench"), 3)


if __name__ == "__main__":
    sys.exit(unittest.main())
