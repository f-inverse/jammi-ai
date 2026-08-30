#!/usr/bin/env python3
"""Hermetic `AB_DRY_RUN=1` smoke test for `finetune_ab.sh` itself — mirrors
`test_gpu_inference_ab_sh_dry_run.py`'s own "the shell PRODUCER itself had
ZERO automated execution" doctrine, applied here to the F5 (empty
`--expect-kernels-disabled` on both fused legs) and F2 (the
`TWO_RUN_PROTOCOL_MARKER` file) fold-ins specifically — neither is
observable from `ab_merge.py`'s own test suite alone, since that suite
drives fixture `raw_dir`s directly and never invokes the shell script's
own argv construction.

`AB_DRY_RUN=1` makes the whole pipeline safe to execute hermetically: no
real `cargo build`, no `uv`, no GPU, no network — every `run_cmd`/
`run_leg`-wrapped command is PRINTED, never executed, and every leg
writes a `{"tool":"dry-run",...}` stub so the merge stage still runs
end-to-end. Drives the REAL `bash ci/scripts/perf/finetune_ab.sh`
subprocess, never a re-implementation of its control flow.

Run: `python3 ci/scripts/perf/test_finetune_ab_sh_dry_run.py`
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "finetune_ab.sh")

sys.path.insert(0, PERF_DIR)
import ab_merge  # noqa: E402 -- the real TWO_RUN_PROTOCOL_MARKER constant, never a re-typed literal.


def run_dry(out_dir, extra_env=None):
    env = dict(os.environ)
    env["AB_DRY_RUN"] = "1"
    env["AB_OUT_DIR"] = out_dir
    # A single config/dropout keeps this hermetic smoke test fast; the
    # sweep's own CONFIGS/DROPOUTS control flow is exercised for real by
    # every OTHER config regardless -- this test's job is the SHELL
    # SCRIPT's own argv construction and control flow, not re-proving the
    # sweep is 3x2.
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    return result


class DryRunSmokeTests(unittest.TestCase):
    def test_dry_run_runs_end_to_end_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(
                result.returncode, 0,
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            report_path = os.path.join(out_dir, "finetune_ab_report.json")
            self.assertTrue(os.path.isfile(report_path), f"no report written\nstdout={result.stdout}")
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertIn("configs", report)
            self.assertTrue(report["configs"])

    def test_two_run_protocol_marker_is_written_before_any_leg_runs(self):
        """F2: the marker's own promise -- written unconditionally, even
        under AB_DRY_RUN, so `ab_merge.py`'s own `two_run_protocol_active`
        reads `True` and the merged report records `two_run_protocol:
        true` for THIS run.
        """
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            marker_path = os.path.join(out_dir, "raw", ab_merge.TWO_RUN_PROTOCOL_MARKER)
            self.assertTrue(
                os.path.isfile(marker_path),
                f"{marker_path} was never written -- finetune_ab.sh must touch it before any leg runs",
            )
            with open(os.path.join(out_dir, "finetune_ab_report.json"), encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertTrue(report["two_run_protocol"])

    def test_both_fused_legs_pass_an_empty_expect_kernels_disabled(self):
        """F5: `jammi-fused` AND `jammi-fused-2` both carry
        `--expect-kernels-disabled ''` (an EMPTY argument, hard-failing on
        any ambient `JAMMI_KERNELS_DISABLE`) -- never omitted the way an
        earlier version of this script left them unguarded.
        """
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            # The printed command trace shows exactly what run_leg would
            # execute -- `--expect-kernels-disabled` immediately followed
            # by an EMPTY shell-quoted argument (`''`) on the two fused
            # legs' own printed invocations.
            fused_lines = [
                line for line in result.stdout.splitlines()
                if "/ jammi-fused:" in line or "/ jammi-fused-2:" in line
            ]
            self.assertTrue(fused_lines, f"no jammi-fused/jammi-fused-2 leg lines found\nstdout={result.stdout}")
            for line in fused_lines:
                self.assertIn(
                    "--expect-kernels-disabled ''", line,
                    f"jammi-fused leg must pass an EMPTY --expect-kernels-disabled: {line!r}",
                )
                # And it must NOT ALSO carry JAMMI_KERNELS_DISABLE=<9 keys>
                # -- that would defeat the whole point of a "fused" leg.
                self.assertNotIn("layer_norm_fused", line)

    def test_jammi_eager_leg_passes_all_nine_disable_keys(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            eager_lines = [line for line in result.stdout.splitlines() if "/ jammi-eager:" in line]
            self.assertTrue(eager_lines, f"no jammi-eager leg lines found\nstdout={result.stdout}")
            nine_keys = [
                "layer_norm_fused", "geglu_fused", "attention_block_flash", "attention_block_fused",
                "rope_fused", "softmax_last_dim_fused", "lora_linear_fused", "adamw_step_fused",
                "mem_efficient_attention",
            ]
            for line in eager_lines:
                for key in nine_keys:
                    self.assertIn(key, line, f"jammi-eager leg line missing {key!r}: {line!r}")

    def test_bar_legs_appear_in_the_a_b_b_a_order_per_config(self):
        """The visible half of the A,B,B,A order binding: within the
        FIRST config's own leg-trace block, jammi-fused, torch-sdpa,
        torch-sdpa-2, jammi-fused-2 must appear in exactly that order.
        """
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            markers = [
                "/ jammi-fused: ", "/ torch-sdpa: ", "/ torch-sdpa-2: ", "/ jammi-fused-2: ",
            ]
            first_indices = [result.stdout.index(m) for m in markers]
            self.assertEqual(
                first_indices, sorted(first_indices),
                f"the first config's own bar legs must appear in jammi-fused, torch-sdpa, "
                f"torch-sdpa-2, jammi-fused-2 order; got indices {first_indices}\nstdout={result.stdout}",
            )

    def test_dry_run_never_creates_a_real_cargo_target_dir_or_touches_git(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as fake_target:
            result = run_dry(out_dir, extra_env={"CARGO_TARGET_DIR": fake_target})
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertFalse(
                os.path.exists(os.path.join(fake_target, "release", "jammi-bench")),
                "AB_DRY_RUN=1 must never actually build the binary",
            )


if __name__ == "__main__":
    unittest.main()
