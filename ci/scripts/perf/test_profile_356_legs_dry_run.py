#!/usr/bin/env python3
"""Hermetic tests for `profile_356_legs.sh` itself (P4, CONTRACT
`scratchpad/contract-356-profile.md` v4; phase-4 adversarial audit
CLASSES 1-3) -- mirrors `test_finetune_ab_sh_dry_run.py`/
`test_gpu_inference_ab_sh_dry_run.py`'s own "the shell PRODUCER itself
must run in CI at least once, zero-execution is RED not a skip" doctrine:
drives the REAL `bash ci/scripts/perf/profile_356_legs.sh` subprocess end
to end, never a re-implementation of its control flow.

Two independent test surfaces:

  `DryRunSmokeTests` (`PROFILE_356_LEGS_DRY_RUN=1`): the whole 14-leg
  sweep, hermetically -- no GPU, no `nsys`, no real checkpoint, no
  network. Every `nsys`/`$BENCH_BIN` invocation is PRINTED via `run_cmd`,
  never executed; each run's own JSON stub is DERIVED from the committed
  `finetune_run_golden` fixture (`_dry_run_stub_report`), so this exercises
  the SAME `tiers.finetune_run` envelope binding every real reader in the
  script uses (CLASS 1: a hand-shaped top-level stub previously let three
  real reader bugs -- the preflight field probe, `_wall_s`, `_lora_
  counters` -- ship undetected). Also asserts every declared leg-table
  column (CLASS 3) reaches both the `manifest.json` record and (via the
  printed command lines) the `cmd` array itself.

  `PreflightArmTests` (`PROFILE_356_LEGS_PREFLIGHT_ONLY=1`,
  `PROFILE_356_LEGS_DRY_RUN=0`): drives the REAL (non-dry) preflight probe
  against a FAKE `$BENCH_BIN` stub script (`_write_fake_bench_stub`),
  covering the pass case and each of the four distinguishable failure arms
  (CLASS 2): (a) distilbert-dispatch missing, (b) the wall field missing,
  (c) the `--layers-to-transform` flag missing, and a genuinely-unrelated
  failure -- proving the four `missing[]` messages in `preflight_probe`
  are not merely "any failure looks the same". The fake stub also answers
  `provenance` (the script's OWN build_sha cross-check, which runs BEFORE
  preflight) with this checkout's REAL `git rev-parse HEAD`, and answers
  `finetune-run --target-modules q_lin,...` (never the CLI's ModernBERT-
  only default) with a SUCCESSFUL zero-trainable check, exercising the
  CLASS 2 selector fix directly: a stub that refused on any `target_modules`
  value would make every arm indistinguishable from "(a) distilbert
  missing", the exact class this fix closes.

Run: `python3 ci/scripts/perf/test_profile_356_legs_dry_run.py`
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "profile_356_legs.sh")
REPO_ROOT = os.path.abspath(os.path.join(PERF_DIR, "..", "..", ".."))


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
                # CLASS 3: every declared leg-table column reaches the
                # manifest, not just LoRA counters.
                for key in (
                    "dtype", "width", "target_modules", "layers_to_transform",
                    "eval_cadence", "steps_declared", "steps_measured",
                    "status", "reason", "census_ok",
                    "lora_counters_n_run", "lora_counters_m_run",
                ):
                    self.assertIn(key, manifest, f"{leg_id} manifest missing {key!r}: {manifest}")
                # Every leg in the hermetic dry-run sweep is a clean, fully
                # recorded run (the golden-derived stub always succeeds).
                self.assertEqual(manifest["status"], "ok", manifest)
                self.assertEqual(manifest["reason"], "")
                self.assertTrue(manifest["census_ok"])
                self.assertEqual(manifest["eval_cadence"], 1)
                # CLASS 1: the LoRA counters came from a REAL
                # tiers.finetune_run read (never a silent {}) -- all four
                # expected keys, non-empty.
                for run in ("lora_counters_n_run", "lora_counters_m_run"):
                    self.assertEqual(
                        set(manifest[run]),
                        {
                            "lora_linear_eager_dispatches", "lora_linear_fused_dispatches",
                            "lora_epilogue_eager_dispatches", "lora_epilogue_fused_dispatches",
                        },
                        manifest,
                    )
                # steps_measured mirrors steps_declared in this hermetic
                # stub (the stub is built FROM steps_declared) -- proves
                # the cross-check plumbing reads a real, non-null value.
                self.assertEqual(manifest["steps_measured"], manifest["steps_declared"], manifest)

    def test_bert_a1_width_reaches_both_manifest_and_cmd_array(self):
        """CLASS 3's headline finding: --max-seq-length was never passed
        at all, so A1 (the contract's mandatory W=512 leg) silently ran at
        clap's own default of 64. Asserts BOTH surfaces: the manifest
        record AND the literal --max-seq-length token in the printed cmd
        line for the leg whose bug this class names explicitly."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            with open(os.path.join(out_dir, "bert-A1", "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["width"], 512, manifest)
            self.assertEqual(manifest["target_modules"], ["query", "key", "value", "dense"])
            self.assertIsNone(manifest["layers_to_transform"])
            # The printed nsys/finetune-run command line must carry the
            # real flag, not merely the manifest's own recollection of it.
            self.assertIn("--max-seq-length 512", result.stdout)
            self.assertIn("--eval-cadence 1", result.stdout)
            self.assertNotIn("--layers-to-transform", result.stdout)

    def test_bert_n1_layers_to_transform_reaches_both_manifest_and_cmd_array(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-N1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            with open(os.path.join(out_dir, "bert-N1", "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["layers_to_transform"], "0", manifest)
            self.assertEqual(manifest["target_modules"], ["query"])
            self.assertIn("--layers-to-transform 0", result.stdout)

    def test_e1_leg_stamps_excluded_from_chain_attribution_in_census_cmd(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-E1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--excluded-from-chain-attribution", result.stdout)
            with open(os.path.join(out_dir, "bert-E1", "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["steps_declared"], {"n": 10, "m": 40}, manifest)

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

    def test_wall_pair_derived_from_the_stub_is_strictly_ordered(self):
        """CLASS 4 (kernel_census.py) needs wall_b > wall_a > 0 -- proves
        the golden-derived DRY_RUN stub (CLASS 1) does not accidentally
        emit an equal/inverted pair that would make the printed
        kernel_census.py command line demonstrate an invalid call."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--wall-a 1.0 --wall-b 6.0", result.stdout)


# ---------------------------------------------------------------------
# CLASS 2: preflight arm-differentiation tests, against a fake $BENCH_BIN.
# ---------------------------------------------------------------------

_FAKE_BENCH_STUB = r"""#!/usr/bin/env bash
# Fake jammi-bench binary for profile_356_legs.sh's preflight-only tests
# (test_profile_356_legs_dry_run.py). Behavior selected by $FAKE_BENCH_MODE:
#   pass               -- --help shows --layers-to-transform; the probe
#                          finetune-run succeeds with a real-shaped
#                          tiers.finetune_run.train_run_wall_s.
#   flag_missing        -- --help does NOT show --layers-to-transform; the
#                          probe finetune-run itself still succeeds (isolates
#                          the (c) arm from (a)/(b)).
#   distilbert_missing  -- --help is fine; the probe finetune-run FAILS with
#                          the exact "unsupported model_type" wording.
#   wall_missing        -- --help is fine; the probe finetune-run succeeds
#                          but its JSON has no train_run_wall_s field.
#   broken              -- --help is fine; the probe finetune-run FAILS with
#                          an unrelated error.
set -euo pipefail

if [ "$1" = "provenance" ]; then
  printf '{"build_sha": "%s"}\n' "$FAKE_BENCH_BUILD_SHA"
  exit 0
fi

if [ "$1" = "finetune-run" ]; then
  shift
  for a in "$@"; do
    if [ "$a" = "--help" ]; then
      if [ "${FAKE_BENCH_MODE:-pass}" = "flag_missing" ]; then
        echo "Usage: jammi-bench finetune-run [OPTIONS]"
        echo "  --model-dir <MODEL_DIR>"
      else
        echo "Usage: jammi-bench finetune-run [OPTIONS]"
        echo "  --layers-to-transform <LAYERS_TO_TRANSFORM>"
      fi
      exit 0
    fi
  done
  case "${FAKE_BENCH_MODE:-pass}" in
    distilbert_missing)
      echo "finetune-run: unsupported model_type 'distilbert' -- supports bert/modernbert" >&2
      exit 1
      ;;
    broken)
      echo "finetune-run failed: some completely unrelated catastrophic error" >&2
      exit 1
      ;;
    wall_missing)
      echo '{"tool":"finetune-run","tiers":{"finetune_run":{"seed":42}}}'
      exit 0
      ;;
    pass|flag_missing)
      echo '{"tool":"finetune-run","tiers":{"finetune_run":{"seed":42,"train_run_wall_s":1.23}}}'
      exit 0
      ;;
    *)
      echo "unknown FAKE_BENCH_MODE: ${FAKE_BENCH_MODE:-pass}" >&2
      exit 1
      ;;
  esac
fi

echo "fake_jammi_bench: unhandled subcommand: $*" >&2
exit 1
"""


def _write_fake_bench_stub(path: Path) -> None:
    path.write_text(_FAKE_BENCH_STUB)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def run_preflight(mode: str) -> subprocess.CompletedProcess:
    real_head = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "fake_jammi_bench.sh"
        _write_fake_bench_stub(stub)
        env = dict(os.environ)
        env["PROFILE_356_LEGS_DRY_RUN"] = "0"
        env["PROFILE_356_LEGS_PREFLIGHT_ONLY"] = "1"
        env["BENCH_BIN"] = str(stub)
        env["NSYS_BIN"] = str(stub)  # never invoked in PREFLIGHT_ONLY mode.
        env["MODEL_DIR_BERT"] = "/fake/bert-checkpoint-dir"
        env["MODEL_DIR_DISTILBERT"] = "/fake/distilbert-checkpoint-dir"
        env["OUT_DIR"] = tmp
        env["FAKE_BENCH_MODE"] = mode
        env["FAKE_BENCH_BUILD_SHA"] = real_head
        return subprocess.run(
            ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60
        )


class PreflightArmTests(unittest.TestCase):
    """CLASS 2: exercises the REAL (non-dry) `preflight_probe` against a
    fake `$BENCH_BIN`, one test per distinguishable outcome."""

    def test_pass_exits_zero_and_reports_all_three_preconditions_ok(self):
        result = run_preflight("pass")
        self.assertEqual(result.returncode, 0, _fail_msg(result))
        self.assertIn("preflight OK", result.stderr + result.stdout)

    def test_distilbert_arm_missing_is_distinguishable(self):
        result = run_preflight("distilbert_missing")
        self.assertNotEqual(result.returncode, 0, _fail_msg(result))
        combined = result.stdout + result.stderr
        self.assertIn("(a)", combined)
        self.assertIn("model_type 'distilbert'", combined)
        self.assertNotIn("(b) the probe", combined)
        self.assertNotIn("(c) '", combined)

    def test_wall_field_missing_is_distinguishable(self):
        result = run_preflight("wall_missing")
        self.assertNotEqual(result.returncode, 0, _fail_msg(result))
        combined = result.stdout + result.stderr
        self.assertIn("(b)", combined)
        self.assertIn("train_run_wall_s", combined)
        self.assertNotIn("(a) finetune-run refuses", combined)
        self.assertNotIn("(c) '", combined)

    def test_layers_to_transform_flag_missing_is_distinguishable(self):
        result = run_preflight("flag_missing")
        self.assertNotEqual(result.returncode, 0, _fail_msg(result))
        combined = result.stdout + result.stderr
        self.assertIn("(c)", combined)
        self.assertIn("--layers-to-transform", combined)
        self.assertNotIn("(a) finetune-run refuses", combined)
        self.assertNotIn("(b) the probe", combined)

    def test_genuinely_broken_probe_is_distinguishable_from_the_known_gaps(self):
        result = run_preflight("broken")
        self.assertNotEqual(result.returncode, 0, _fail_msg(result))
        combined = result.stdout + result.stderr
        self.assertIn("reason other than the known distilbert gap", combined)
        self.assertNotIn("(a) finetune-run refuses", combined)
        self.assertNotIn("(b) the probe finetune-run's own JSON", combined)
        self.assertNotIn("(c) '", combined)

    def test_preflight_only_exits_before_the_leg_sweep(self):
        """A passing preflight under PREFLIGHT_ONLY must exit 0 WITHOUT
        ever entering run_leg (no manifest.json anywhere under OUT_DIR) --
        proves the early-exit hook actually short-circuits, not merely
        that the (fake, non-GPU-capable) sweep happens to fail silently
        afterward."""
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_jammi_bench.sh"
            _write_fake_bench_stub(stub)
            real_head = subprocess.run(
                ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            env = dict(os.environ)
            env["PROFILE_356_LEGS_DRY_RUN"] = "0"
            env["PROFILE_356_LEGS_PREFLIGHT_ONLY"] = "1"
            env["BENCH_BIN"] = str(stub)
            env["NSYS_BIN"] = str(stub)
            env["MODEL_DIR_BERT"] = "/fake/bert-checkpoint-dir"
            env["MODEL_DIR_DISTILBERT"] = "/fake/distilbert-checkpoint-dir"
            env["OUT_DIR"] = str(out_dir)
            env["FAKE_BENCH_MODE"] = "pass"
            env["FAKE_BENCH_BUILD_SHA"] = real_head
            result = subprocess.run(
                ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifests = list(out_dir.rglob("manifest.json"))
            self.assertEqual(manifests, [], f"expected no legs to run: {manifests}")


if __name__ == "__main__":
    unittest.main()
