#!/usr/bin/env python3
"""Hermetic `ENCODE_AB_DRY_RUN=1` run of `encode_ab.sh` itself: its env
parsing, the legs it would run and the order it would run them in, the
markers it hands its merge stage, and the merge stage it really runs over the
stubs. No build, no torch, no network — every leg's command is PRINTED, never
executed. Drives the REAL `bash ci/scripts/perf/encode_ab.sh` subprocess,
never a re-implementation of its control flow.

Run: `python3 ci/scripts/perf/test_encode_ab_sh_dry_run.py`
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "encode_ab.sh")
sys.path.insert(0, PERF_DIR)
import encode_ab  # noqa: E402 -- the script and its merge stage must agree on the legs.

LEG_LINE = re.compile(r"^--- (\S+): (.*)$", re.MULTILINE)


def run_dry(out_dir, **env):
    full_env = dict(os.environ, ENCODE_AB_DRY_RUN="1", ENCODE_AB_OUT_DIR=out_dir, **env)
    return subprocess.run(["bash", SCRIPT], env=full_env, capture_output=True, text=True, timeout=60)


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)

    def legs(self, **env):
        result = run_dry(self.out.name, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, dict(LEG_LINE.findall(result.stdout))

    def test_it_runs_end_to_end_without_building_or_running_a_leg(self):
        result, _ = self.legs()
        self.assertNotIn("+ cargo", result.stdout, "a dry run must not even print a build it skipped")
        with open(os.path.join(self.out.name, "encode_ab_report.json")) as fh:
            merged = json.load(fh)
        self.assertEqual(merged["status"], "DRY_RUN")
        self.assertEqual({leg["outcome"] for leg in merged["legs"].values()}, {"DRY_RUN"})

    def test_the_legs_run_in_the_merge_stages_order(self):
        result, _ = self.legs()
        self.assertEqual([leg for leg, _ in LEG_LINE.findall(result.stdout)], list(encode_ab.LEG_ORDER))

    def test_each_jammi_arm_is_served_at_its_partitions_and_leaves_an_exchange_directory(self):
        _, legs = self.legs(ENCODE_AB_PARTITIONS="6", ENCODE_AB_ROWS="16,64", ENCODE_AB_DTYPE="bf16")
        for leg, partitions in (("jammi-p1", "1"), ("jammi-pN", "6"), ("jammi-pN-2", "6"), ("jammi-p1-2", "1")):
            self.assertIn(f"encode-step --rows 16\\,64 --partitions {partitions} ", legs[leg])
            self.assertIn("--compute-precision bf16", legs[leg])
            self.assertIn(f"--exchange-dir {self.out.name}/raw/{leg}.exchange", legs[leg])
            self.assertNotIn("--cuda", legs[leg])
            self.assertNotIn("--model-dir", legs[leg], "unset, the jammi legs serve their compiled-in fixture")

    def test_each_torch_arm_reads_the_first_jammi_legs_exchange_in_its_own_order(self):
        _, legs = self.legs(ENCODE_AB_DTYPE="bf16")
        exchange = f"{self.out.name}/raw/jammi-p1.exchange"
        for leg, order, attn in (
            ("torch-corpus", "corpus", "eager"),
            ("torch-sorted", "length-sorted", "sdpa"),
            ("torch-sorted-2", "length-sorted", "sdpa"),
            ("torch-corpus-2", "corpus", "eager"),
        ):
            self.assertIn(f"--exchange-dir {exchange} ", legs[leg])
            self.assertIn(f"--model-dir {exchange}/model ", legs[leg], "unset, the fixture the jammi leg left")
            self.assertIn(f"--order {order} --attn {attn}", legs[leg])
            self.assertIn("--dtype bf16", legs[leg])
            self.assertIn("--ann-index", legs[leg])

    def test_a_named_checkpoint_and_device_reach_every_leg(self):
        _, legs = self.legs(ENCODE_AB_MODEL_DIR="/models/encoder", ENCODE_AB_CUDA_ORDINAL="0")
        self.assertEqual(set(legs), set(encode_ab.LEG_ORDER))
        for command in legs.values():
            self.assertIn("--model-dir /models/encoder", command)
            self.assertIn("--cuda 0", command)

    def test_the_torch_legs_can_stop_at_the_parquet_file(self):
        _, legs = self.legs(ENCODE_AB_TORCH_ANN_INDEX="0")
        for arm in encode_ab.TORCH_ARMS:
            self.assertNotIn("--ann-index", legs[arm])

    def test_the_merge_stage_is_handed_this_runs_markers(self):
        self.legs(ENCODE_AB_PARTITIONS="6", ENCODE_AB_PASS_RATIO="0.8", ENCODE_AB_COSINE_FLOOR="0.95")
        with open(os.path.join(self.out.name, "encode_ab_report.json")) as fh:
            merged = json.load(fh)
        self.assertEqual(
            (merged["partitions_n"], merged["torch_ann_index"], merged["pass_ratio"], merged["cosine_floor"]),
            (6, True, 0.8, 0.95),
        )


if __name__ == "__main__":
    unittest.main()
