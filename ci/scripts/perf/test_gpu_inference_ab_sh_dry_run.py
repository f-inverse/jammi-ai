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
import sys
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "gpu_inference_ab.sh")

sys.path.insert(0, PERF_DIR)
import gpu_inference_ab  # noqa: E402 -- round-3 adversarial audit B3: the producer-to-comparator round trip needs the REAL parser, never a re-implementation.


def run_dry(out_dir, work_dir, extra_env=None):
    """`work_dir` is REQUIRED (round-2 adversarial audit advisory): every
    caller passes an explicit `GPU_INFERENCE_AB_WORK_DIR` (a tempdir the
    caller owns and cleans up), never the script's own default (a sibling
    directory of the checkout, `$(dirname "$REPO_ROOT")/gpu-perf-ab-<ts>`)
    -- `mkdir -p "$WORK_DIR"` runs UNCONDITIONALLY in `gpu_inference_ab.sh`,
    even under `--dry-run`, so a caller that omitted this would leave a
    real, empty, timestamped directory behind next to this very checkout
    on every single test run.
    """
    env = dict(os.environ)
    env["GPU_INFERENCE_AB_DRY_RUN"] = "1"
    env["GPU_INFERENCE_AB_OUT_DIR"] = out_dir
    env["GPU_INFERENCE_AB_WORK_DIR"] = work_dir
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    return result


class DryRunSmokeTests(unittest.TestCase):
    def test_dry_run_runs_end_to_end_and_never_touches_the_network_or_a_real_build(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir)

            # round-3 adversarial audit B2 (the auditor's own reproduction):
            # every real leg is a `DRY_RUN`-outcome stub under
            # GPU_INFERENCE_AB_DRY_RUN=1, NEVER `FAIL` -- "nothing ran"
            # carries no runtime signal about a PR binary at all, so the
            # merge status must be the NEUTRAL INCOMPLETE/75, never the
            # PR-blame INVALID/1 an earlier (round-2) version of this
            # routing incorrectly produced. Proven here against the REAL
            # process exit code, not assumed.
            self.assertEqual(
                result.returncode,
                75,
                f"dry-run's own four stub legs are never 'OK' NOR 'FAIL' (outcome is 'DRY_RUN' -- "
                f"nothing ran), so the merge status must be the neutral INCOMPLETE/exit 75 "
                f"deterministically (round-3 adversarial audit B2 correction)\n"
                f"stdout={result.stdout}\nstderr={result.stderr}",
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
            self.assertEqual(report["mode"], "dry-run")
            self.assertIn("nothing ran", report["incomplete_reason"])
            # issue #335's final unit: GPU_INFERENCE_AB_ENFORCE defaults
            # unset/0 -- the producer must still write a well-formed
            # "enforce" marker (round-trip through the real
            # gpu_inference_ab.py::load_enforce, not re-implemented here).
            self.assertEqual(report["enforce"], False)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "raw", "enforce")))

    def test_dry_run_with_enforce_env_writes_a_true_enforce_marker(self):
        """`GPU_INFERENCE_AB_ENFORCE=1` must round-trip through the real
        producer-to-comparator boundary: the shell script writes
        `$RAW_DIR/enforce` containing `"1"`, and `gpu_inference_ab.py`'s own
        `load_enforce` (never re-implemented here) reads it back `True` --
        even on a dry run, whose four MISSING legs keep this INCOMPLETE/75
        regardless (enforcement is only ever consulted on a GREEN status).
        """
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir, extra_env={"GPU_INFERENCE_AB_ENFORCE": "1"})
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")

            raw_dir = os.path.join(out_dir, "raw")
            self.assertTrue(gpu_inference_ab.load_enforce(raw_dir))

            report_path = os.path.join(out_dir, "gpu_inference_ab_report.json")
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["enforce"], True)
            self.assertEqual(report["status"], "INCOMPLETE", "enforcement never fires on a non-GREEN status")

    def test_dry_run_never_creates_a_real_cargo_target_dir(self):
        """The strongest available proof this run touched no real build:
        `run_cmd`'s dry-run branch never executes `cargo build`, so neither
        `target-a/release/jammi-bench` nor `target-b/release/jammi-bench`
        (the two binaries a real run would produce) exists anywhere under
        `GPU_INFERENCE_AB_WORK_DIR` afterward.
        """
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir)
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
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir)
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertNotIn("nvidia-smi", result.stderr)

    def test_dry_run_prints_the_four_legs_in_the_a1_b1_b2_a2_order(self):
        """round-2 adversarial audit F3: the printed leg-trace sequence
        (`--- a1: `, `--- b1: `, `--- b2: `, `--- a2: `) must appear in
        EXACTLY that order in stdout -- the visible, human-readable half of
        the order binding (the MACHINE-CHECKED half lives in
        `gpu_inference_ab.py::verify_recorded_order`, driven against the
        `.started_at` files this same run writes -- see the NEXT test for
        the actual producer-to-comparator round trip over those files).
        """
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir)
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")

            markers = ["--- a1: ", "--- b1: ", "--- b2: ", "--- a2: "]
            indices = [result.stdout.index(m) for m in markers]
            self.assertEqual(
                indices,
                sorted(indices),
                f"the four leg-trace markers must appear in a1,b1,b2,a2 order in stdout; got indices "
                f"{indices} for {markers}\nstdout={result.stdout}",
            )

    def test_the_real_started_at_files_round_trip_through_the_comparators_own_parser(self):
        """round-3 adversarial audit B3: the producer-to-comparator round
        trip, pinned WITHOUT a GPU -- reads the FOUR REAL `.started_at`
        files `gpu_inference_ab.sh`'s own `run_leg` wrote (real `date
        +%s%N` output, not a Python-constructed fixture), parses EACH one
        through `gpu_inference_ab.py`'s OWN
        `load_leg_started_at`/`verify_recorded_order` functions (imported
        directly, never re-implemented), and asserts they parse as ints in
        non-decreasing a1,b1,b2,a2 order -- proving the two halves of this
        system (what the shell writes, what the Python reads) actually
        agree on the file's own format, not merely on paper.
        """
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as work_dir:
            result = run_dry(out_dir, work_dir)
            self.assertEqual(result.returncode, 75, f"stdout={result.stdout}\nstderr={result.stderr}")

            raw_dir = os.path.join(out_dir, "raw")
            started_at_by_leg = {}
            for name in gpu_inference_ab.LEG_ORDER:
                path = os.path.join(raw_dir, f"{name}.started_at")
                self.assertTrue(os.path.isfile(path), f"the real producer must write {path}")
                value, reason = gpu_inference_ab.load_leg_started_at(raw_dir, name)
                self.assertIsNone(reason, f"leg {name!r}'s real .started_at file failed to parse: {reason}")
                self.assertIsInstance(value, int)
                started_at_by_leg[name] = (value, reason)

            values = [started_at_by_leg[name][0] for name in gpu_inference_ab.LEG_ORDER]
            self.assertEqual(
                values,
                sorted(values),
                f"the four REAL .started_at files must parse into a non-decreasing a1,b1,b2,a2 "
                f"sequence; got {dict(zip(gpu_inference_ab.LEG_ORDER, values))}",
            )
            self.assertEqual(
                gpu_inference_ab.verify_recorded_order(started_at_by_leg),
                [],
                "the real producer's own timestamps, read through the comparator's own parser, must "
                "verify as a clean A,B,B,A order with no findings at all",
            )


if __name__ == "__main__":
    unittest.main()
