#!/usr/bin/env python3
"""Hermetic tests for `profile_356_legs.sh` itself (P4, CONTRACT
`scratchpad/contract-356-profile.md` v4; phase-4 adversarial audit
CLASSES 1-3, round-2 BLOCKS 1-2) -- mirrors `test_finetune_ab_sh_dry_run.py`/
`test_gpu_inference_ab_sh_dry_run.py`'s own "the shell PRODUCER itself
must run in CI at least once, zero-execution is RED not a skip" doctrine:
drives the REAL `bash ci/scripts/perf/profile_356_legs.sh` subprocess end
to end, never a re-implementation of its control flow.

Three independent test surfaces:

  `DryRunSmokeTests` (`PROFILE_356_LEGS_DRY_RUN=1`): the whole 14-leg
  sweep, hermetically -- no GPU, no real `nsys`, no real checkpoint, no
  network. `$NSYS_BIN`/`$BENCH_BIN` are swapped for hermetic fake
  stand-ins (`profile_356_legs.sh`'s own `$DRY_RUN_STUB_DIR` mechanism)
  and ACTUALLY EXECUTED through the real capture path (round-2 audit
  BLOCK 1's own "CRITICAL": a hand-shaped stub that bypassed the capture
  machinery entirely structurally could not catch a bug IN that
  machinery -- fixed by making DRY_RUN drive the SAME `bash -c
  'exec ... > "$0"'` exec-wrapper a real leg uses, just against a
  deliberately CHATTY fake nsys and a fake bench that emits the
  golden-derived envelope on its OWN stdout). Also asserts every declared
  leg-table column (CLASS 3) reaches both the `manifest.json` record and
  (via the printed, stderr-only command lines) the `cmd` array itself.

  `PreflightArmTests` (`PROFILE_356_LEGS_PREFLIGHT_ONLY=1`,
  `PROFILE_356_LEGS_DRY_RUN=0`): drives the REAL (non-dry) preflight probe
  against a FAKE `$BENCH_BIN` stub script (`_write_fake_bench_stub`),
  covering the pass case and each of the four distinguishable failure arms
  (CLASS 2): (a) distilbert-dispatch missing, (b) the wall field missing,
  (c) the `--layers-to-transform` flag missing, and a genuinely-unrelated
  failure -- proving the four `missing[]` messages in `preflight_probe`
  are not merely "any failure looks the same". The stub's SUCCESSFUL
  `finetune-run` JSON is itself DERIVED from the committed golden fixture
  (round-2 audit advisory 3 -- never hand-shaped), the same discipline
  `profile_356_legs.sh`'s own DRY_RUN stand-in follows. The stub also
  answers `provenance` (the script's OWN build_sha cross-check, which runs
  BEFORE preflight) with this checkout's REAL `git rev-parse HEAD`, and
  answers `finetune-run --target-modules q_lin,...` (never the CLI's
  ModernBERT-only default) with a SUCCESSFUL zero-trainable check,
  exercising the CLASS 2 selector fix directly: a stub that refused on any
  `target_modules` value would make every arm indistinguishable from "(a)
  distilbert missing", the exact class this fix closes.

  `MidSweepNsysFailureTests` (round-2 audit BLOCK 2): a REAL (non-dry,
  non-preflight-only), multi-leg run against a fake `$NSYS_BIN` whose
  `--version` deliberately fails while `profile`/`export` still work --
  the auditor's own empirical reproduction (a failing `nsys --version`
  used to abort the WHOLE sweep mid-leg via an unguarded `set -e`-fatal
  command, with no manifest for the failing leg). Asserts the sweep still
  completes for EVERY leg, each with a manifest recording the degraded
  `nsys_version` string -- proves the fix (moved to a ONCE, guarded,
  start-of-script computation) eliminates the whole failure class, not
  merely "the next leg still runs". Both legs ARE legitimately recorded
  status="invalid" in this test (this bare fake nsys cannot produce a
  real CUPTI-table sqlite export at all, so `kernel_census.py` refuses
  each leg on its own, unrelated domain check) -- the assertion that
  actually matters is that neither leg's recorded `reason` mentions the
  version failure, proving the INVALID status has nothing to do with the
  bug this test targets.

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
GOLDEN_FIXTURE = os.path.join(PERF_DIR, "fixtures", "finetune_run_golden", "bert_fused.json")


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


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
                    "lora_counters_n_run", "lora_counters_m_run", "dry_run",
                    "fixed_cost_buckets", "fixed_cost_time_us",
                ):
                    self.assertIn(key, manifest, f"{leg_id} manifest missing {key!r}: {manifest}")
                # `run_cmd` (the wrapper `census_cmd` itself goes through)
                # only ECHOES under DRY_RUN and never executes
                # kernel_census.py -- census.json is never actually
                # written in this mode, so the fixed-cost tally this
                # producer surfaces from it (phase-4 audit round-5 BLOCK
                # 2) must stay null here, never a fabricated stand-in the
                # way `census_ok=true` already is for this whole mode.
                self.assertIsNone(manifest["fixed_cost_buckets"])
                self.assertIsNone(manifest["fixed_cost_time_us"])
                # Every leg in the hermetic dry-run sweep is a clean, fully
                # recorded run (the real capture path against the fake
                # nsys/bench stand-ins always succeeds) -- round-2 audit
                # BLOCK 1's regression class: this assertion is exactly
                # what would go RED again if a report envelope ever got
                # polluted by trace/nsys stdout noise (json.load would
                # fail, `run_traced` would return nonzero, and every leg
                # would show status="invalid" instead).
                self.assertEqual(manifest["status"], "ok", manifest)
                self.assertEqual(manifest["reason"], "")
                self.assertTrue(manifest["census_ok"])
                self.assertTrue(manifest["dry_run"])
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
                # fake (the fake bench is built FROM steps_declared) --
                # proves the cross-check plumbing reads a real, non-null
                # value.
                self.assertEqual(manifest["steps_measured"], manifest["steps_declared"], manifest)

    def test_chatty_fake_nsys_stdout_never_pollutes_the_report_file(self):
        """Round-2 audit BLOCK 1(b)'s own regression proof: the built-in
        DRY_RUN fake nsys prints stdout noise on BOTH `profile` and
        `export`; asserts the WRITTEN report file (run_n.json under the
        leg dir, not merely the manifest's own recollection of success)
        parses as clean JSON carrying the real envelope shape, never a
        `+ nsys ...`/`fake_nsys: ...` trace line."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            # The chatty noise must be visible SOMEWHERE in this process's
            # own captured output (proving the fake really is chatty, not
            # merely silent-by-accident) -- but nowhere inside the report
            # file itself.
            self.assertIn("fake_nsys: chatty stdout noise", result.stdout)
            report_path = os.path.join(out_dir, "bert-A1", "run_n.json")
            with open(report_path, encoding="utf-8") as f:
                report_text = f.read()
            self.assertNotIn("fake_nsys", report_text)
            self.assertNotIn("+ ", report_text[:5])
            report = json.loads(report_text)  # raises if any noise leaked in
            self.assertIn("tiers", report)
            self.assertIn("finetune_run", report["tiers"])
            self.assertIn("train_run_wall_s", report["tiers"]["finetune_run"])

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
            # The printed nsys/finetune-run command line (now on stderr --
            # round-2 audit BLOCK 1(a)) must carry the real flag, not
            # merely the manifest's own recollection of it.
            self.assertIn("--max-seq-length 512", result.stderr)
            self.assertIn("--eval-cadence 1", result.stderr)
            self.assertNotIn("--layers-to-transform", result.stderr)

    def test_bert_n1_layers_to_transform_reaches_both_manifest_and_cmd_array(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-N1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            with open(os.path.join(out_dir, "bert-N1", "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["layers_to_transform"], "0", manifest)
            self.assertEqual(manifest["target_modules"], ["query"])
            self.assertIn("--layers-to-transform 0", result.stderr)

    def test_e1_leg_stamps_excluded_from_chain_attribution_in_census_cmd(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-E1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--excluded-from-chain-attribution", result.stderr)
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
        the golden-derived fake bench (CLASS 1) does not accidentally emit
        an equal/inverted pair that would make the printed
        kernel_census.py command line demonstrate an invalid call."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--wall-a 1.0 --wall-b 6.0", result.stderr)

    def test_steps_measured_pair_reaches_the_census_cmd(self):
        """Round-2 audit advisory 3's own explicit pin (mirroring the
        --wall pin above): --steps-measured-a/-b must reach
        kernel_census.py's own argv, not just live in the manifest."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="bert-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--steps-measured-a 100 --steps-measured-b 600", result.stderr)


# ---------------------------------------------------------------------
# CLASS 2 / BLOCK 2: fake $BENCH_BIN / $NSYS_BIN stub scripts, generated
# at test time. The fake bench's SUCCESSFUL finetune-run JSON is derived
# from the committed golden fixture (round-2 audit advisory 3), never
# hand-shaped -- $FAKE_BENCH_GOLDEN names the committed fixture path.
# ---------------------------------------------------------------------

_FAKE_BENCH_STUB = r"""#!/usr/bin/env bash
# Fake jammi-bench binary for profile_356_legs.sh's preflight/mid-sweep
# tests (test_profile_356_legs_dry_run.py). Behavior selected by $FAKE_BENCH_MODE:
#   pass               -- --help shows --layers-to-transform; every
#                          finetune-run invocation succeeds with a
#                          golden-derived tiers.finetune_run envelope
#                          (train_run_wall_s added).
#   flag_missing        -- --help does NOT show --layers-to-transform; the
#                          probe finetune-run itself still succeeds (isolates
#                          the (c) arm from (a)/(b)).
#   distilbert_missing  -- --help is fine; the probe finetune-run FAILS with
#                          the exact "unsupported model_type" wording.
#   wall_missing        -- --help is fine; finetune-run succeeds with the
#                          RAW golden tier (which predates train_run_wall_s
#                          entirely -- no override needed to omit it).
#   broken              -- --help is fine; the probe finetune-run FAILS with
#                          an unrelated error.
#   broken_envelope     -- --help is fine; finetune-run exits 0 but prints a
#                          stray banner line BEFORE its JSON, corrupting the
#                          report file the exec-wrapper captures (round-2
#                          audit BLOCK 1(c)'s own _validate_report_envelope
#                          failing path, driven for real through the actual
#                          capture machinery, not just unit-tested in
#                          isolation).
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
    broken_envelope)
      # The preflight probe ALWAYS targets $MODEL_DIR_DISTILBERT -- answer
      # IT with a clean, pass-shaped envelope regardless, so this mode
      # only corrupts a REAL leg's own finetune-run invocation (whichever
      # model-dir that leg's own architecture actually uses), never the
      # one-time probe call every real-mode run makes first.
      is_probe=0
      prev_arg=""
      for a in "$@"; do
        if [ "$prev_arg" = "--model-dir" ] && [ "$a" = "$MODEL_DIR_DISTILBERT" ]; then
          is_probe=1
        fi
        prev_arg="$a"
      done
      if [ "$is_probe" = "1" ]; then
        python3 -c '
import copy, json, sys
golden = json.load(open(sys.argv[1]))
tier = copy.deepcopy(golden["tiers"]["finetune_run"])
tier["train_run_wall_s"] = 1.23
tier.pop("steps_measured", None)
print(json.dumps({"tool": "finetune-run", "tiers": {"finetune_run": tier}}))
' "$FAKE_BENCH_GOLDEN"
        exit 0
      fi
      # A stray line on stdout BEFORE the JSON -- the exec-wrapper
      # redirects THIS process's entire stdout to the report file, so this
      # banner line lands in the report file too, breaking json.load.
      echo "unexpected startup banner noise from a buggy bench binary"
      python3 -c '
import json, sys
golden = json.load(open(sys.argv[1]))
tier = golden["tiers"]["finetune_run"]
print(json.dumps({"tool": "finetune-run", "tiers": {"finetune_run": tier}}))
' "$FAKE_BENCH_GOLDEN"
      exit 0
      ;;
    wall_missing)
      python3 -c '
import json, sys
golden = json.load(open(sys.argv[1]))
tier = golden["tiers"]["finetune_run"]
print(json.dumps({"tool": "finetune-run", "tiers": {"finetune_run": tier}}))
' "$FAKE_BENCH_GOLDEN"
      exit 0
      ;;
    pass|flag_missing)
      # steps_measured deliberately OMITTED -- this one fake bench answers
      # every finetune-run invocation identically regardless of the real
      # --batch/--epochs/corpus-size arguments it was actually given, so a
      # fixed value here would (correctly) trip kernel_census.py's own
      # declared-vs-measured cross-check for any REAL leg whose declared
      # step count differs from that fixed value. steps_measured is
      # cross-checked "when available" only (driver's own doc) -- omitting
      # it here is honest (this fake genuinely does not know the real
      # step count), not a workaround.
      python3 -c '
import copy, json, sys
golden = json.load(open(sys.argv[1]))
tier = copy.deepcopy(golden["tiers"]["finetune_run"])
tier["train_run_wall_s"] = 1.23
# The COMMITTED golden fixture carries its OWN steps_measured (a real,
# frozen value unrelated to whatever a caller declares) -- popped, not
# merely left unset, so it does not leak through as a fixed value that
# would mismatch every real legs declared step count.
tier.pop("steps_measured", None)
print(json.dumps({"tool": "finetune-run", "tiers": {"finetune_run": tier}}))
' "$FAKE_BENCH_GOLDEN"
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

# Fake nsys binary that FAILS `--version` specifically, while `profile`/
# `export` still behave normally (round-2 audit BLOCK 2's own empirical
# reproduction: the auditor's fake nsys was exactly this shape).
_FAKE_NSYS_VERSION_FAILING_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "--version" ]; then
  echo "fake_nsys: --version deliberately fails for this test" >&2
  exit 1
fi
if [ "$1" = "export" ]; then
  shift
  out=""
  for a in "$@"; do
    case "$a" in
      --output=*) out="${a#--output=}" ;;
    esac
  done
  if [ -n "$out" ]; then : > "$out"; fi
  exit 0
fi
if [ "$1" = "profile" ]; then
  shift
  args=("$@")
  for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--" ]; then
      rest=("${args[@]:$((i+1))}")
      exec "${rest[@]}"
    fi
  done
  echo "fake_nsys: no -- separator found" >&2
  exit 1
fi
echo "fake_nsys: unknown subcommand $1" >&2
exit 1
"""


def _write_stub(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_fake_bench_stub(path: Path) -> None:
    _write_stub(path, _FAKE_BENCH_STUB)


def run_preflight(mode: str) -> subprocess.CompletedProcess:
    real_head = _real_head()
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
        env["FAKE_BENCH_GOLDEN"] = GOLDEN_FIXTURE
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
            env["FAKE_BENCH_BUILD_SHA"] = _real_head()
            env["FAKE_BENCH_GOLDEN"] = GOLDEN_FIXTURE
            result = subprocess.run(
                ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifests = list(out_dir.rglob("manifest.json"))
            self.assertEqual(manifests, [], f"expected no legs to run: {manifests}")


class MidSweepNsysFailureTests(unittest.TestCase):
    """Round-2 audit BLOCK 2: a REAL (non-dry), multi-leg run against a
    fake $NSYS_BIN whose --version deliberately fails. Reproduces the
    auditor's own finding (an unguarded per-leg `nsys --version` capture
    aborted the WHOLE sweep) and proves the fix: `NSYS_VERSION` is now
    computed ONCE, guarded, before any leg runs, so a failing --version
    degrades to a placeholder string and affects NOTHING else."""

    def test_failing_nsys_version_does_not_abort_the_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bench = Path(tmp) / "fake_bench.sh"
            fake_nsys = Path(tmp) / "fake_nsys.sh"
            _write_fake_bench_stub(fake_bench)
            _write_stub(fake_nsys, _FAKE_NSYS_VERSION_FAILING_STUB)
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            env = dict(os.environ)
            env["PROFILE_356_LEGS_DRY_RUN"] = "0"
            env["PROFILE_356_LEGS_PREFLIGHT_ONLY"] = "0"
            env["BENCH_BIN"] = str(fake_bench)
            env["NSYS_BIN"] = str(fake_nsys)
            env["MODEL_DIR_BERT"] = "/fake/bert-checkpoint-dir"
            env["MODEL_DIR_DISTILBERT"] = "/fake/distilbert-checkpoint-dir"
            env["OUT_DIR"] = str(out_dir)
            env["FAKE_BENCH_MODE"] = "pass"
            env["FAKE_BENCH_BUILD_SHA"] = _real_head()
            env["FAKE_BENCH_GOLDEN"] = GOLDEN_FIXTURE
            # Two small, non-E1 legs (E1 needs the real, network-
            # provisioned train_pairs.jsonl, out of scope for this test).
            env["PROFILE_356_LEGS_ONLY"] = "bert-N1,distilbert-N1"
            result = subprocess.run(
                ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=180
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            # BOTH legs must be recorded -- the sweep is never aborted
            # mid-leg by the failing --version (this bare fake nsys cannot
            # produce a real CUPTI-table sqlite export at all, so
            # kernel_census.py legitimately refuses each leg on ITS OWN,
            # UNRELATED domain check -- that is expected and is not what
            # this test is about; what matters is that BOTH legs got a
            # manifest, in a run whose --version genuinely failed, and
            # that failure is never itself the recorded reason).
            for leg_id in ("bert-N1", "distilbert-N1"):
                manifest_path = out_dir / leg_id / "manifest.json"
                self.assertTrue(
                    manifest_path.is_file(), f"no manifest for {leg_id}: {_fail_msg(result)}"
                )
                manifest = json.loads(manifest_path.read_text())
                self.assertIn("nsys --version failed", manifest["nsys_version"], manifest)
                self.assertNotIn("--version", manifest["reason"], manifest)
                self.assertNotIn("nsys_version", manifest["reason"], manifest)


class ReportEnvelopeValidationFailureTests(unittest.TestCase):
    """Round-3 audit item 5: drives `_validate_report_envelope`'s own
    FAILING path through the REAL (non-dry) capture machinery, not just
    the pure function in isolation -- a fake bench that emits a stray
    stdout banner line before its JSON, corrupting the exact file the
    exec-wrapper writes. Also pins the fix for the "operator pointed at a
    file that does not contain the cause" gap: `run_traced` tees the
    validation error into `$run_prefix.stderr`, so the leg's own recorded
    `reason` (which names that file) is no longer misleading."""

    def test_stray_stdout_banner_corrupts_the_report_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_bench = Path(tmp) / "fake_bench.sh"
            fake_nsys = Path(tmp) / "fake_nsys.sh"
            _write_fake_bench_stub(fake_bench)
            _write_stub(fake_nsys, _FAKE_NSYS_VERSION_FAILING_STUB)
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            env = dict(os.environ)
            env["PROFILE_356_LEGS_DRY_RUN"] = "0"
            env["PROFILE_356_LEGS_PREFLIGHT_ONLY"] = "0"
            env["BENCH_BIN"] = str(fake_bench)
            env["NSYS_BIN"] = str(fake_nsys)
            env["MODEL_DIR_BERT"] = "/fake/bert-checkpoint-dir"
            env["MODEL_DIR_DISTILBERT"] = "/fake/distilbert-checkpoint-dir"
            env["OUT_DIR"] = str(out_dir)
            env["FAKE_BENCH_MODE"] = "broken_envelope"
            env["FAKE_BENCH_BUILD_SHA"] = _real_head()
            env["FAKE_BENCH_GOLDEN"] = GOLDEN_FIXTURE
            env["PROFILE_356_LEGS_ONLY"] = "bert-N1"
            result = subprocess.run(
                ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=120
            )
            # The sweep itself still completes (one leg, recorded invalid,
            # never a script-wide abort).
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifest_path = out_dir / "bert-N1" / "manifest.json"
            self.assertTrue(manifest_path.is_file(), _fail_msg(result))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "invalid", manifest)
            # The recorded reason names report-envelope validation as ONE
            # of the possible causes (round-3 audit item 5's "distinguish
            # envelope-validation failure from a run failure" -- satisfied
            # here via the file it points at containing the real cause,
            # asserted next, rather than a separate reason-per-cause code).
            self.assertIn("report-envelope validation", manifest["reason"], manifest)
            # And the file that reason POINTS AT must actually contain the
            # real cause -- not merely a silent, unhelpful nsys/bench log.
            stderr_path = out_dir / "bert-N1" / "run_n.stderr"
            self.assertTrue(stderr_path.is_file(), f"no {stderr_path}: {_fail_msg(result)}")
            stderr_text = stderr_path.read_text()
            self.assertIn("does not parse as JSON", stderr_text, stderr_text)
            # The report file itself is genuinely corrupted (the banner
            # line really did reach it, via the real exec-wrapper).
            report_path = out_dir / "bert-N1" / "run_n.json"
            report_text = report_path.read_text()
            self.assertIn("unexpected startup banner noise", report_text)
            with self.assertRaises(json.JSONDecodeError):
                json.loads(report_text)


if __name__ == "__main__":
    unittest.main()
