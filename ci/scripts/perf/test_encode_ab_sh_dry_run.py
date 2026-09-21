#!/usr/bin/env python3
"""Hermetic `ENCODE_AB_DRY_RUN=1` run of `encode_ab.sh` itself: its env
parsing, the producer invocations it would make and the order it would make
them in, and the comparator it hands the legs to. No build, no torch, no
network — every command is PRINTED, never executed. Drives the REAL
`bash ci/scripts/perf/encode_ab.sh` subprocess, never a re-implementation of
its control flow.

Run: `python3 ci/scripts/perf/test_encode_ab_sh_dry_run.py`
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "encode_ab.sh")

LEG_LINE = re.compile(r"^--- (\S+): (.*)$", re.MULTILINE)
CMD_LINE = re.compile(r"^\+ (.*)$", re.MULTILINE)


def run_dry(out_dir, **env):
    full_env = dict(os.environ, ENCODE_AB_DRY_RUN="1", ENCODE_AB_OUT_DIR=out_dir, **env)
    return subprocess.run(["bash", SCRIPT], env=full_env, capture_output=True, text=True, timeout=60)


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)

    def run_and_parse(self, **env):
        result = run_dry(self.out.name, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, dict(LEG_LINE.findall(result.stdout)), CMD_LINE.findall(result.stdout)

    def test_it_runs_end_to_end_without_building_or_running_a_leg(self):
        result, _, commands = self.run_and_parse()
        self.assertFalse(any("cargo" in c for c in commands), "a dry run must not even print a build it skipped")
        self.assertNotIn("torch_encode.py", "".join(commands), "legs are printed as invocations, never run")

    def test_the_arms_run_as_a_palindrome_and_the_comparator_judges_each_order(self):
        _, legs, commands = self.run_and_parse()
        self.assertEqual(
            list(legs),
            [
                "jammi-interleaved",
                "torch-corpus",
                "torch-sorted",
                "jammi-direct",
                "jammi-plan",
                "jammi-plan-partitioned",
            ],
        )
        ladder = [c for c in commands if " ladder encode " in c]
        self.assertEqual(len(ladder), 2, commands)
        self.assertIn(f"ladder encode {self.out.name}/legs-corpus --out {self.out.name}/verdict-corpus", ladder[0])
        self.assertIn(f"ladder encode {self.out.name}/legs-sorted --out {self.out.name}/verdict-sorted", ladder[1])

    def test_the_engine_rungs_run_interleaved_then_each_alone(self):
        _, legs, _ = self.run_and_parse(ENCODE_AB_PARTITIONS="6", ENCODE_AB_ROWS="16,64", ENCODE_AB_DTYPE="bf16", ENCODE_AB_TAKES="3")
        interleaved = legs["jammi-interleaved"]
        self.assertIn("encode-step --task embed --rows 16\\,64 --takes 3 --partitions 6 ", interleaved)
        self.assertIn("--rung direct --rung plan --rung plan-partitioned", interleaved)
        self.assertIn("--compute-precision bf16", interleaved)
        self.assertIn(f"--legs-dir {self.out.name}/legs-corpus ", interleaved)
        self.assertIn(f"--exchange-dir {self.out.name}/exchange ", interleaved)
        for label, rung in (("jammi-direct", "direct"), ("jammi-plan", "plan"), ("jammi-plan-partitioned", "plan-partitioned")):
            self.assertTrue(legs[label].endswith(f"--rung {rung} "), legs[label])
            self.assertIn(f"--legs-dir {self.out.name}/legs-corpus/space ", legs[label])
        for command in legs.values():
            if "encode-step" in command:
                self.assertNotIn("--cuda", command)
                self.assertNotIn("--model-dir", command, "unset, the jammi legs serve their compiled-in fixture")

    def test_each_torch_arm_reads_the_exchange_and_runs_under_the_sampler(self):
        _, legs, _ = self.run_and_parse(ENCODE_AB_DTYPE="bf16")
        exchange = f"{self.out.name}/exchange"
        for label, order, attn, legs_dir in (
            ("torch-corpus", "corpus", "eager", "legs-corpus"),
            ("torch-sorted", "length-sorted", "sdpa", "legs-sorted"),
        ):
            command = legs[label]
            self.assertIn(f"--exchange-dir {exchange} ", command)
            self.assertIn(f"--model-dir {exchange}/model ", command, "unset, the fixture the jammi leg left")
            self.assertIn(f"--legs-dir {self.out.name}/{legs_dir} ", command)
            self.assertIn("--sampler-bin ", command)
            self.assertIn(f"--order {order} --attn {attn}", command)
            self.assertIn("--dtype bf16", command)
            self.assertIn("--ann-index", command)

    def test_a_named_checkpoint_and_device_reach_every_leg(self):
        _, legs, _ = self.run_and_parse(ENCODE_AB_MODEL_DIR="/models/encoder", ENCODE_AB_CUDA_ORDINAL="0")
        for command in legs.values():
            self.assertIn("--model-dir /models/encoder", command)
            self.assertIn("--cuda 0", command)

    def test_the_torch_legs_can_stop_at_the_parquet_file(self):
        _, legs, _ = self.run_and_parse(ENCODE_AB_TORCH_ANN_INDEX="0")
        for label in ("torch-corpus", "torch-sorted"):
            self.assertNotIn("--ann-index", legs[label])


if __name__ == "__main__":
    unittest.main()
