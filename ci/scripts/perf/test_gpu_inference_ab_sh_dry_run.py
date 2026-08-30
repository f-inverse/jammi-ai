#!/usr/bin/env python3
"""Hermetic `--dry-run` smoke test for `gpu_inference_ab.sh` itself (round-1
adversarial audit advisory): before this test, the shell PRODUCER had ZERO
automated execution anywhere in this repo's CI — only `gpu_inference_ab.py`
(the Python merge/table module it drives) and `gpu_inference_ab_git.sh`
(the git-shape library it sources) had wired test suites; the SHELL SCRIPT
ITSELF, its arg/env parsing, its `run_cmd` dry-run wrapper, and its own
control flow through the clone/build/leg/merge pipeline, ran in CI exactly
zero times. `ci.yml`'s own "zero-execution is RED, not a skip" doctrine
(the same reasoning `test_ab_merge.py`/`compare_grad_oracle`'s own suites
were wired in to close) applies here too.

`GPU_INFERENCE_AB_DRY_RUN=1` makes the whole pipeline safe to execute
hermetically: no real `git clone`, no `cargo build`, no `nvidia-smi`, no
network — every `run_cmd`-wrapped command is PRINTED, never executed, and
every leg writes a `{"tool":"dry-run","ab_dry_run":true,...}` stub so the
merge stage still runs end-to-end against real (if fabricated-empty) files.
Drives the REAL `bash ci/scripts/perf/gpu_inference_ab.sh` subprocess, never
a re-implementation of its control flow.

Run: `python3 ci/scripts/perf/test_gpu_inference_ab_sh_dry_run.py`
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "gpu_inference_ab.sh")


def run_dry(out_dir, extra_env=None):
    env = dict(os.environ)
    env["GPU_INFERENCE_AB_DRY_RUN"] = "1"
    env["GPU_INFERENCE_AB_OUT_DIR"] = out_dir
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    return result


class DryRunSmokeTests(unittest.TestCase):
    def test_dry_run_runs_end_to_end_and_never_touches_the_network_or_a_real_build(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)

            # Every real leg is a stub under GPU_INFERENCE_AB_DRY_RUN=1 (see
            # gpu_inference_ab.py's own MISSING/INCOMPLETE doctrine), so the
            # merge status is deterministically INCOMPLETE and the exit
            # code is the documented neutral 75 -- proven here against the
            # REAL process exit code, not assumed.
            self.assertEqual(
                result.returncode,
                75,
                f"dry-run's own four stub legs are never 'OK' (dry-run reports), so the merge status "
                f"must be INCOMPLETE/exit 75 deterministically\nstdout={result.stdout}\nstderr={result.stderr}",
            )

            # The printed command trace shows the commands run_cmd WOULD
            # have run (proving the control flow reached them), but no real
            # clone/build ever happened -- checked below by asserting
            # neither target-a nor target-b was ever created under the
            # (real, on-disk) work dir this invocation used.
            self.assertIn("git clone", result.stdout)
            self.assertIn("cargo build", result.stdout)

            report_path = os.path.join(out_dir, "gpu_inference_ab_report.json")
            self.assertTrue(os.path.isfile(report_path), f"no report written under {out_dir}\nstdout={result.stdout}")
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertEqual(sorted(report["missing_legs"]), ["a1", "a2", "b1", "b2"])

    def test_dry_run_never_creates_a_real_cargo_target_dir(self):
        """The strongest available proof this run touched no real build:
        `run_cmd`'s dry-run branch never executes `cargo build`, so neither
        `target-a/release/jammi-bench` nor `target-b/release/jammi-bench`
        (the two binaries a real run would produce) exists anywhere under
        `GPU_INFERENCE_AB_WORK_DIR` afterward.
        """
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, extra_env={"GPU_INFERENCE_AB_WORK_DIR": work_dir})
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")

            for sub in ("target-a", "target-b", "clone-a", "clone-b"):
                self.assertFalse(
                    os.path.exists(os.path.join(work_dir, sub)),
                    f"dry-run must never create {sub}/ under the work dir -- run_cmd's dry-run branch "
                    "prints every clone/build command instead of executing it",
                )

    def test_dry_run_skips_the_gpu_idle_check_without_needing_gpu_skip_flag(self):
        """`GPU_INFERENCE_AB_DRY_RUN=1` alone must be sufficient to skip the
        `nvidia-smi` idle gate (no GPU exists in this hermetic test
        environment at all) -- proven by NOT setting
        `GPU_INFERENCE_AB_SKIP_GPU_CHECK` and confirming the run still
        completes rather than erroring out looking for `nvidia-smi`.
        """
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertNotIn("nvidia-smi", result.stderr)


if __name__ == "__main__":
    unittest.main()
