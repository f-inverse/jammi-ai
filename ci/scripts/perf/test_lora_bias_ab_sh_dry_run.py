#!/usr/bin/env python3
"""Hermetic tests for `lora_bias_ab.sh` itself (#428 P2b, docs-ci) --
mirrors `test_profile_356_legs_dry_run.py`'s own "the shell PRODUCER itself
must run in CI at least once, zero-execution is RED not a skip" doctrine:
drives the REAL `bash ci/scripts/perf/lora_bias_ab.sh` subprocess end to
end under `LORA_BIAS_AB_DRY_RUN=1`, never a re-implementation of its control
flow.

Two independent test surfaces:

  `DryRunSmokeTests`: the whole sweep, hermetically -- no GPU, no real
  `jammi-bench`, no checkpoint, no network. `$BENCH_BIN` is swapped for a
  hermetic fake stand-in (the script's own `$DRY_RUN_STUB_DIR` mechanism)
  and ACTUALLY EXECUTED through the real capture path (redirect + envelope
  validation), the same discipline `profile_356_legs.sh`'s own DRY_RUN arm
  follows. Asserts the manifest row count/shape, the per-arm env
  (`JAMMI_KERNELS_STRICT` on every arm, `JAMMI_KERNELS_DISABLE` naming
  `lora_linear_fused` ONLY on `lora_eager`), `LORA_BIAS_AB_EXTRA_DISABLE`
  landing symmetrically on every arm, the fused/eager order-balance across
  repeats, the `LORA_BIAS_AB_LEGS_ONLY` filter's own guards (unknown id
  refuses, a re-run against the same `OUT_DIR` refuses rather than silently
  appending).

  `PreflightRefusalTests`: drives the `LORA_BIAS_AB_DRY_RUN_FAIL_OP` hook
  the fake bench stub reads (see `lora_bias_ab.sh`'s own module doc) to
  prove the STRICT preflight's own refusal path actually stops the sweep
  BEFORE any leg runs (an empty `manifest.json`, never a partially-run
  sweep) and names the offending op key plus the pre-registered remedy.

  `AbOpLnTests` (#460): drives `AB_OP=ln` -- the disable key becomes
  `layer_norm_fused`, the eager-arm label becomes `ln_eager`, and every
  manifest row records `ab_op == "ln"`. `AB_OP=lora_linear` (every test
  above, run with no `AB_OP` set at all) is byte-for-byte what it always
  was; `AbOpLnTests` also pins that an unrecognized `AB_OP` refuses before
  any manifest is even created.

Run: `python3 ci/scripts/perf/test_lora_bias_ab_sh_dry_run.py`
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "lora_bias_ab.sh")

MODELS = ("bert", "distilbert")
SHAPES = ("b8W512f32", "b32W64bf16")
ARMS = ("fused", "lora_eager")


def _fail_msg(result):
    return f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"


def run_dry(out_dir, extra_env=None, timeout=120):
    env = dict(os.environ)
    env["LORA_BIAS_AB_DRY_RUN"] = "1"
    env["OUT_DIR"] = out_dir
    env["BENCH_BIN"] = "/nonexistent/jammi-bench-DRY-RUN-PLACEHOLDER"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=timeout
    )


def load_manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


class DryRunSmokeTests(unittest.TestCase):
    def test_exits_zero_and_manifest_has_the_expected_row_count(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            # main: 2 models x 2 shapes x 2 arms x 3 repeats x 2 steps = 48
            # control: 2 models x 3 repeats x 2 steps = 12
            self.assertEqual(len(rows), 60, [r["leg_id"] for r in rows])
            for row in rows:
                self.assertEqual(row["status"], "ok", row)
                self.assertEqual(row["rc"], 0, row)

    def test_every_declared_leg_table_column_reaches_the_manifest(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            by_id = {r["leg_id"]: r for r in rows}
            expected_id = "bert-b8W512f32-fused-100-r1"
            self.assertIn(expected_id, by_id, sorted(by_id))
            row = by_id[expected_id]
            self.assertEqual(row["model"], "bert")
            self.assertEqual(row["shape"], "b8W512f32")
            self.assertEqual(row["batch"], 8)
            self.assertEqual(row["width"], 512)
            self.assertEqual(row["dtype"], "f32")
            self.assertEqual(row["arm"], "fused")
            self.assertEqual(row["steps"], 100)
            self.assertEqual(row["repeat"], 1)
            self.assertTrue(row["report_path"])
            self.assertTrue(os.path.isfile(os.path.join(out_dir, row["report_path"])))
            self.assertEqual(len(row["git_sha"]), 40)
            self.assertTrue(row["box"])
            self.assertIn("--target-modules", row["argv"])
            self.assertIn("--max-seq-length", row["argv"])

    def test_env_is_strict_on_every_arm_and_disable_only_on_eager(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            by_arm = {}
            for row in rows:
                by_arm.setdefault(row["arm"], []).append(row)
            for arm, arm_rows in by_arm.items():
                for row in arm_rows:
                    self.assertEqual(row["env"]["JAMMI_KERNELS_STRICT"], "1", row)
                    if arm == "lora_eager":
                        self.assertEqual(row["env"]["JAMMI_KERNELS_DISABLE"], "lora_linear_fused", row)
                    else:
                        self.assertEqual(row["env"]["JAMMI_KERNELS_DISABLE"], "", row)

    def test_extra_disable_lands_symmetrically_on_every_arm(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, {"LORA_BIAS_AB_EXTRA_DISABLE": "some_bad_op"})
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            for row in rows:
                self.assertEqual(row["extra_disable"], ["some_bad_op"], row)
                disable = row["env"]["JAMMI_KERNELS_DISABLE"]
                self.assertIn("some_bad_op", disable, row)
                if row["arm"] == "lora_eager":
                    self.assertEqual(disable, "lora_linear_fused,some_bad_op", row)
                else:
                    self.assertEqual(disable, "some_bad_op", row)

    def test_order_balanced_repeats_alternate_fused_eager(self):
        with tempfile.TemporaryDirectory() as out_dir:
            legs_only = ",".join(
                f"bert-b8W512f32-{arm}-r{r}" for r in (1, 2, 3) for arm in ARMS
            )
            result = run_dry(out_dir, {"LORA_BIAS_AB_LEGS_ONLY": legs_only})
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            # Each arm-visit (one repeat's one arm) emits exactly two
            # CONSECUTIVE manifest rows -- its N-step run, then its M-step
            # run (`run_cell`'s own call order) -- so every even-indexed row
            # starts a new visit. Reading the visit-level arm directly this
            # way (rather than collapsing consecutive same-arm ROWS) avoids
            # conflating two ADJACENT visits that happen to share an arm
            # (repeat 1 ends on `lora_eager`, repeat 2 starts on
            # `lora_eager` too -- the A,B,B,A shape's own middle seam).
            self.assertEqual(len(rows), 12, [r["leg_id"] for r in rows])
            for i in range(0, len(rows), 2):
                self.assertEqual(rows[i]["arm"], rows[i + 1]["arm"], rows)
                self.assertEqual(rows[i]["repeat"], rows[i + 1]["repeat"], rows)
                self.assertEqual({rows[i]["steps"], rows[i + 1]["steps"]}, {100, 600}, rows)
            visit_arms = [rows[i]["arm"] for i in range(0, len(rows), 2)]
            self.assertEqual(
                visit_arms,
                ["fused", "lora_eager", "lora_eager", "fused", "fused", "lora_eager"],
                [r["arm"] for r in rows],
            )

    def test_legs_only_filter_runs_only_the_named_cells(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, {"LORA_BIAS_AB_LEGS_ONLY": "bert-control-r1"})
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            self.assertEqual(len(rows), 2, rows)  # one cell = N-step + M-step
            for row in rows:
                self.assertEqual(row["model"], "bert")
                self.assertEqual(row["arm"], "control")
                self.assertEqual(row["repeat"], 1)

    def test_legs_only_unknown_id_refuses_before_any_leg_runs(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, {"LORA_BIAS_AB_LEGS_ONLY": "bert-totally-bogus-r99"})
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("unknown cell id", result.stderr)
            rows = load_manifest(out_dir)
            self.assertEqual(rows, [])

    def test_rerun_against_the_same_out_dir_refuses(self):
        with tempfile.TemporaryDirectory() as out_dir:
            first = run_dry(out_dir, {"LORA_BIAS_AB_LEGS_ONLY": "bert-control-r1"})
            self.assertEqual(first.returncode, 0, _fail_msg(first))
            second = run_dry(out_dir, {"LORA_BIAS_AB_LEGS_ONLY": "bert-control-r1"})
            self.assertNotEqual(second.returncode, 0, _fail_msg(second))
            self.assertIn("already has a manifest.json", second.stderr)


class PreflightRefusalTests(unittest.TestCase):
    def test_strict_refusal_stops_before_any_leg_and_names_the_key(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir,
                {
                    "LORA_BIAS_AB_DRY_RUN_FAIL_OP": "some_bad_op",
                    "LORA_BIAS_AB_DRY_RUN_FAIL_PREDICATE": "some_predicate",
                },
            )
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("some_bad_op", result.stderr)
            self.assertIn("STRICT preflight refused", result.stderr)
            self.assertIn("pre-registered remedy", result.stderr)
            self.assertIn("LORA_BIAS_AB_EXTRA_DISABLE=some_bad_op", result.stderr)
            # No leg ever ran -- the manifest was initialized ([]) but never
            # appended to.
            rows = load_manifest(out_dir)
            self.assertEqual(rows, [])
            # No raw report files were ever written.
            raw_dir = os.path.join(out_dir, "raw")
            self.assertEqual(os.listdir(raw_dir) if os.path.isdir(raw_dir) else [], [])

    def test_remedy_appends_onto_an_existing_extra_disable(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir,
                {
                    "LORA_BIAS_AB_EXTRA_DISABLE": "already_disabled_op",
                    "LORA_BIAS_AB_DRY_RUN_FAIL_OP": "newly_found_op",
                },
            )
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn(
                "LORA_BIAS_AB_EXTRA_DISABLE=already_disabled_op,newly_found_op",
                result.stderr,
            )


class AbOpLnTests(unittest.TestCase):
    """#460: `AB_OP=ln` forces `layer_norm_fused` eager instead of
    `lora_linear_fused` -- same driver, same STRICT/negative-control/corpus
    mechanism, only the disable key and the eager-arm label change. Every
    test here uses `LORA_BIAS_AB_LEGS_ONLY` to keep the sweep itself small
    (the KNOWN_CELL_IDS table and the STRICT preflight -- 4 probes,
    unconditional -- still run in full either way, exactly like every
    `DryRunSmokeTests` case above)."""

    def test_manifest_rows_record_ab_op_ln_and_the_ln_eager_arm_label(self):
        with tempfile.TemporaryDirectory() as out_dir:
            legs_only = "bert-b8W512f32-fused-r1,bert-b8W512f32-ln_eager-r1"
            result = run_dry(out_dir, {"AB_OP": "ln", "LORA_BIAS_AB_LEGS_ONLY": legs_only})
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            self.assertEqual(len(rows), 4, [r["leg_id"] for r in rows])  # 2 cells x (N, M)
            arms_seen = {r["arm"] for r in rows}
            self.assertEqual(arms_seen, {"fused", "ln_eager"}, rows)
            for row in rows:
                self.assertEqual(row["ab_op"], "ln", row)
                self.assertEqual(row["status"], "ok", row)

    def test_env_is_strict_on_every_arm_and_disable_only_on_ln_eager(self):
        with tempfile.TemporaryDirectory() as out_dir:
            legs_only = "bert-b8W512f32-fused-r1,bert-b8W512f32-ln_eager-r1"
            result = run_dry(out_dir, {"AB_OP": "ln", "LORA_BIAS_AB_LEGS_ONLY": legs_only})
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            for row in rows:
                self.assertEqual(row["env"]["JAMMI_KERNELS_STRICT"], "1", row)
                if row["arm"] == "ln_eager":
                    self.assertEqual(row["env"]["JAMMI_KERNELS_DISABLE"], "layer_norm_fused", row)
                else:
                    self.assertEqual(row["env"]["JAMMI_KERNELS_DISABLE"], "", row)

    def test_control_cell_still_works_under_ab_op_ln(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir, {"AB_OP": "ln", "LORA_BIAS_AB_LEGS_ONLY": "bert-control-r1"}
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            self.assertEqual(len(rows), 2, rows)
            for row in rows:
                self.assertEqual(row["model"], "bert")
                self.assertEqual(row["arm"], "control")
                self.assertEqual(row["ab_op"], "ln", row)
                self.assertEqual(row["env"]["JAMMI_KERNELS_DISABLE"], "", row)

    def test_extra_disable_lands_symmetrically_under_ab_op_ln(self):
        with tempfile.TemporaryDirectory() as out_dir:
            legs_only = "bert-b8W512f32-fused-r1,bert-b8W512f32-ln_eager-r1"
            result = run_dry(
                out_dir,
                {
                    "AB_OP": "ln",
                    "LORA_BIAS_AB_LEGS_ONLY": legs_only,
                    "LORA_BIAS_AB_EXTRA_DISABLE": "some_bad_op",
                },
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            rows = load_manifest(out_dir)
            for row in rows:
                self.assertEqual(row["extra_disable"], ["some_bad_op"], row)
                disable = row["env"]["JAMMI_KERNELS_DISABLE"]
                self.assertIn("some_bad_op", disable, row)
                if row["arm"] == "ln_eager":
                    self.assertEqual(disable, "layer_norm_fused,some_bad_op", row)
                else:
                    self.assertEqual(disable, "some_bad_op", row)

    def test_known_cell_ids_use_the_ln_eager_label_not_lora_eager(self):
        with tempfile.TemporaryDirectory() as out_dir:
            # `lora_eager` is not a known arm label for an AB_OP=ln sweep --
            # the cell-id table is built from THIS sweep's own eager-arm
            # label, never the other op's.
            result = run_dry(
                out_dir, {"AB_OP": "ln", "LORA_BIAS_AB_LEGS_ONLY": "bert-b8W512f32-lora_eager-r1"}
            )
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("unknown cell id", result.stderr)

    def test_unrecognized_ab_op_refuses_before_any_manifest_is_created(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, {"AB_OP": "bogus_op"})
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("AB_OP must be", result.stderr)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "manifest.json")))


if __name__ == "__main__":
    unittest.main()
