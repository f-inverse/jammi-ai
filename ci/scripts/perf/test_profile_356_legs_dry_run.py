#!/usr/bin/env python3
"""Hermetic `PROFILE_356_LEGS_DRY_RUN=1` smoke test for
`profile_356_legs.sh` itself (P4, CONTRACT
`scratchpad/contract-356-profile.md` v4) -- mirrors `test_finetune_ab_sh_
dry_run.py`/`test_gpu_inference_ab_sh_dry_run.py`'s own "the shell
PRODUCER itself must run in CI at least once, zero-execution is RED not a
skip" doctrine: drives the REAL `bash ci/scripts/perf/profile_356_legs.sh`
subprocess end to end (leg table construction, per-leg corpus/work-dir
setup, the provenance cross-check, per-leg manifest stamping), never a
re-implementation of its control flow. No GPU, no `nsys`, no real
checkpoint, no network -- every `nsys`/`$BENCH_BIN` invocation is PRINTED
via `run_cmd`, never executed.

This test does NOT exercise `preflight_probe`'s real (non-dry) precondition
probe (that function returns immediately under DRY_RUN, by design -- there
is nothing to probe against a placeholder binary) -- it proves the
SCRIPT's own control flow (leg iteration, corpus/manifest wiring) is
correct, not that a real pod run's preflight gate fires correctly (that
half needs a real `jammi-bench` binary and is exercised, if at all, on the
GPU pod itself).

Run: `python3 ci/scripts/perf/test_profile_356_legs_dry_run.py`
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "profile_356_legs.sh")


def run_dry(out_dir, legs_only=None):
    env = dict(os.environ)
    env["PROFILE_356_LEGS_DRY_RUN"] = "1"
    env["OUT_DIR"] = out_dir
    env["NSYS_BIN"] = "/nonexistent/nsys-DRY-RUN-PLACEHOLDER"
    env["BENCH_BIN"] = "/nonexistent/jammi-bench-DRY-RUN-PLACEHOLDER"
    if legs_only:
        env["PROFILE_356_LEGS_ONLY"] = legs_only
    result = subprocess.run(
        ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=120
    )
    return result


def _fail_msg(result):
    return f"stdout={result.stdout}\nstderr={result.stderr}"


class DryRunSmokeTests(unittest.TestCase):
    def test_dry_run_all_14_legs_exits_zero_and_stamps_every_manifest(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            expected_legs = [
                f"{model}-{leg}"
                for model in ("bert", "distilbert")
                for leg in ("A1", "A2", "A3", "A4", "N1", "N3", "E1")
            ]
            self.assertEqual(len(expected_legs), 14)
            for leg_id in expected_legs:
                manifest_path = os.path.join(out_dir, leg_id, "manifest.json")
                self.assertTrue(
                    os.path.isfile(manifest_path),
                    f"no manifest for {leg_id}\nstdout={result.stdout}\nstderr={result.stderr}",
                )
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                self.assertEqual(manifest["leg_id"], leg_id)
                self.assertIn("git_sha", manifest)
                self.assertIn("driver", manifest)
                self.assertIn("lora_counters_n_run", manifest)
                self.assertIn("lora_counters_m_run", manifest)

    def test_legs_only_filter_runs_a_single_leg(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-E1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "bert-E1", "manifest.json")))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "bert-A1", "manifest.json")))
            self.assertIn("skipping bert-A1", result.stdout)

    def test_e1_leg_uses_the_heldout_corpus_mode(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="distilbert-E1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("corpus=heldout", result.stdout)

    def test_non_e1_leg_uses_the_synthetic_corpus_mode(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("corpus=synthetic", result.stdout)


if __name__ == "__main__":
    unittest.main()
