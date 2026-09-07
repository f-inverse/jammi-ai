#!/usr/bin/env python3
"""Hermetic tests for `profile_421_legs.sh` itself (issue #421 P1-b(vi);
CONTRACT `scratchpad/contract-421-profile.md` v2) -- the same "the shell
PRODUCER itself must run in CI at least once, zero-execution is RED not a
skip" doctrine `test_profile_356_legs_dry_run.py` /
`test_finetune_ab_sh_dry_run.py` already carry: this drives the REAL `bash
ci/scripts/perf/profile_421_legs.sh` subprocess end to end, never a
re-implementation of its control flow.

Two independent test surfaces:

  `DryRunSmokeTests` (`PROFILE_421_LEGS_DRY_RUN=1`): the whole 12-leg
  sweep, hermetically -- no GPU, no real `nsys`, no real checkpoint, no
  network. `$NSYS_BIN`/`$BENCH_BIN` are swapped for the driver's own
  hermetic fake stand-ins and ACTUALLY EXECUTED through the real capture
  path (a hand-shaped stub that bypassed the capture machinery could not
  catch a bug IN that machinery), and every pinned leg-table column is
  asserted on BOTH surfaces it must reach: the `manifest.json` record AND
  the literal command line the driver prints (on stderr) for the
  invocation it would run on a pod.

  `PreflightArmTests` (`PROFILE_421_LEGS_PREFLIGHT_ONLY=1`,
  `PROFILE_421_LEGS_DRY_RUN=0`): drives the REAL (non-dry) preflight probe
  against a FAKE `$BENCH_BIN` stub, covering the pass case and each
  distinguishable failure arm -- one per NEW flag this unit's legs depend
  on (`--expect-kernels-disabled`, `--lora-init`) plus the two pinned
  pre-existing ones (`--task`, `--objective`) -- proving the preflight's
  messages are not merely "any failure looks the same". The producers'
  held-out mode is probed by ACTUALLY RUNNING all three real producers, so
  that arm needs no stub at all: it passes iff the committed producers
  really do emit `heldout_ids.txt` + `heldout_triplets.jsonl`.

Run: `python3 ci/scripts/perf/test_profile_421_legs_dry_run.py`
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "profile_421_legs.sh")
REPO_ROOT = os.path.abspath(os.path.join(PERF_DIR, "..", "..", ".."))
GOLDEN_FIXTURE = os.path.join(PERF_DIR, "fixtures", "finetune_run_golden", "bert_fused.json")

# The contract's own leg table, spelled out here as a LITERAL rather than
# parsed out of the script: a test that asked the driver which legs it
# defines would agree with itself after any drift.
TOWERS = ("clip-text", "clip-vision", "htsat")
LEG_SUFFIXES = ("A1", "A2", "D1", "D2")
ALL_LEG_IDS = [f"{tower}-{leg}" for tower in TOWERS for leg in LEG_SUFFIXES]

# The D-leg disable lists (contract legs table). D1 is PER-TOWER:
# `gelu_erf_fused` is reachable on HTSAT only -- the CLIP towers' MLP
# activation is `quick_gelu`, which has no fused seam and therefore no
# `admit` key, so naming it on a CLIP leg would put an entry in
# `JAMMI_KERNELS_DISABLE` that never disables a live dispatch and
# `finetune-run` would refuse the leg outright (`unmatched_disables()`).
# Spelled out as literals here, never read back from the driver.
D1_KEYS_CLIP = "lora_linear_fused,layer_norm_fused"
D1_KEYS_HTSAT = "lora_linear_fused,layer_norm_fused,gelu_erf_fused"
D2_KEYS = "lora_linear_fused"


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


# Small pinned step counts for the ordinary hermetic case: `run_corpus_cmd`
# now runs the three real corpus producers even under DRY_RUN (esc-088), so
# every `run_dry()` call that leaves the driver's own N=100/M=600 defaults
# in place generates a real rows_m = BATCH * 600 = 4800-row corpus per
# selected leg/tower -- CPU-hermetic and correct, but needlessly slow for
# tests that only care about the CONTROL FLOW around corpus provisioning,
# not the pinned production row counts themselves. These two are small but
# still N < M (so N/M wall-differencing assertions that merely check
# `front["n"] < front["m"]` still hold) and both a nonzero multiple's worth
# of rows above the fixed `HELDOUT_ROWS = 8` (`--batch 8`), so no producer's
# own row-count invariants are violated.
_SMALL_STEPS_N = "2"
_SMALL_STEPS_M = "6"


def run_dry(out_dir, legs_only=None, extra_env=None, pin_workload_steps=False):
    """Drive the real `profile_421_legs.sh` under `PROFILE_421_LEGS_DRY_RUN=1`.

    `pin_workload_steps=True` opts OUT of the small step-count override
    below and leaves `PROFILE_421_STEPS_N`/`_M` unset, so the driver falls
    back to its OWN pinned production defaults (100/600) -- for the small
    number of tests that assert on those literal, contract-pinned values
    themselves (`DryRunSmokeTests.test_every_leg_pins_the_contracts_workload_constants`,
    `test_the_n_and_m_corpora_differ_only_in_row_count`,
    `test_the_text_legs_heldout_split_is_disjoint_from_BOTH_train_corpora`).
    Every other test only exercises CONTROL FLOW that is step-count
    agnostic, so it runs against the small values instead.
    """
    env = dict(os.environ)
    env["PROFILE_421_LEGS_DRY_RUN"] = "1"
    env["OUT_DIR"] = out_dir
    env["NSYS_BIN"] = "/nonexistent/nsys-DRY-RUN-PLACEHOLDER"
    env["BENCH_BIN"] = "/nonexistent/jammi-bench-DRY-RUN-PLACEHOLDER"
    if not pin_workload_steps:
        env["PROFILE_421_STEPS_N"] = _SMALL_STEPS_N
        env["PROFILE_421_STEPS_M"] = _SMALL_STEPS_M
    else:
        env.pop("PROFILE_421_STEPS_N", None)
        env.pop("PROFILE_421_STEPS_M", None)
    # A leftover JAMMI_KERNELS_DISABLE in the caller's environment must not
    # silently become part of what the legs measure.
    env.pop("JAMMI_KERNELS_DISABLE", None)
    if legs_only:
        env["PROFILE_421_LEGS_ONLY"] = legs_only
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=300)


def _fail_msg(result):
    return f"stdout={result.stdout}\nstderr={result.stderr}"


def _manifest(out_dir, leg_id):
    with open(os.path.join(out_dir, leg_id, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


class DryRunSmokeTests(unittest.TestCase):
    def test_dry_run_all_12_legs_exits_zero_and_stamps_every_manifest(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertEqual(len(ALL_LEG_IDS), 12)
            for leg_id in ALL_LEG_IDS:
                manifest_path = os.path.join(out_dir, leg_id, "manifest.json")
                self.assertTrue(os.path.isfile(manifest_path), f"no manifest for {leg_id}\n{_fail_msg(result)}")
                manifest = _manifest(out_dir, leg_id)
                self.assertEqual(manifest["leg_id"], leg_id)
                # Every declared leg-table column reaches the record, not
                # just the ones this suite happens to assert on below.
                for key in (
                    "git_sha", "driver", "nsys_version", "tower", "task", "dtype",
                    "target_modules", "kernels_disabled", "max_seq_length", "batch",
                    "objective", "lora_init", "eval_cadence", "heldout_rows",
                    "steps_declared", "steps_measured", "media_front_end_wall_s",
                    "checkpoint_weights_sha256", "fusible_site_census",
                    "status", "reason", "census_ok", "census_exit", "dry_run",
                ):
                    self.assertIn(key, manifest, f"{leg_id} manifest missing {key!r}: {manifest}")
                self.assertEqual(manifest["status"], "ok", manifest)
                self.assertEqual(manifest["reason"], "")
                self.assertTrue(manifest["dry_run"])
                self.assertEqual(manifest["git_sha"], _real_head())

    def test_every_leg_pins_the_contracts_workload_constants(self):
        """The pinned flags are pinned on EVERY leg -- batch 8, one epoch's
        worth of N=100/M=600 steps, `--objective triplet` (which is what
        makes `rows = 3B` on all three towers), `--lora-init zeros_b`, and a
        held-out split that is a nonzero multiple of the batch.

        `pin_workload_steps=True`: this is the one test whose OWN point is
        the literal 100/600 the contract pins, read off the driver's own
        printout/manifest with no override in play -- every other test in
        this suite runs against the small step-count override instead
        (see `run_dry`'s own doc)."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, pin_workload_steps=True)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for leg_id in ALL_LEG_IDS:
                manifest = _manifest(out_dir, leg_id)
                self.assertEqual(manifest["batch"], 8, manifest)
                self.assertEqual(manifest["objective"], "triplet", manifest)
                self.assertEqual(manifest["lora_init"], "zeros_b", manifest)
                self.assertEqual(manifest["eval_cadence"], 1, manifest)
                self.assertEqual(manifest["steps_declared"], {"n": 100, "m": 600}, manifest)
                self.assertEqual(manifest["heldout_rows"], 8, manifest)
                self.assertEqual(
                    manifest["heldout_rows"] % manifest["batch"], 0,
                    "finetune-run refuses a held-out fixture that is not a nonzero multiple of "
                    f"--batch: {manifest}",
                )
            # And the pinned flags appear LITERALLY in the printed
            # would-be-production command line, not only in the manifest's
            # own recollection of them.
            for token in (
                "--objective triplet",
                "--lora-init zeros_b",
                "--batch 8",
                "--epochs 1",
                "--grad-accum 1",
                "--validation-fraction 0",
                "--early-stopping-metric train_loss",
                "--seed 42",
            ):
                self.assertIn(token, result.stderr, f"{token!r} never reached a command line")

    def test_each_tower_drives_its_own_task_and_producer(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            expected_task = {
                "clip-text": "text_embedding",
                "clip-vision": "image_embedding",
                "htsat": "audio_embedding",
            }
            for leg_id in ALL_LEG_IDS:
                manifest = _manifest(out_dir, leg_id)
                self.assertEqual(manifest["task"], expected_task[manifest["tower"]], manifest)
                self.assertIn(f"--task {manifest['task']}", result.stderr)
            # The three producers, each with the shape its tower's contract
            # clause pins: 224 px images, 9.5 s @ 48 kHz audio, and text at
            # `--min-wordpieces 77` (the CLIP-text context width).
            self.assertIn("gen_fixed_shape_image_corpus.py", result.stderr)
            self.assertIn("--size 224", result.stderr)
            self.assertIn("gen_fixed_length_audio_corpus.py", result.stderr)
            self.assertIn("--seconds 9.5", result.stderr)
            self.assertIn("--sample-rate 48000", result.stderr)
            self.assertIn("gen_fixed_width_corpus.py", result.stderr)
            self.assertIn("--min-wordpieces 77", result.stderr)
            # Every producer invocation carries the held-out flags.
            self.assertIn("--heldout-rows 8", result.stderr)
            self.assertIn("--heldout-batch 8", result.stderr)

    def test_the_n_and_m_corpora_differ_only_in_row_count(self):
        """The whole differencing method rests on this: `rows = batch *
        steps`, so N asks for 800 rows and M for 4800 at the SAME seed.
        `pin_workload_steps=True`: this asserts the literal contract-pinned
        row counts, so it must run against the real N=100/M=600 defaults,
        not this suite's small override."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1", pin_workload_steps=True)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("--rows 800", result.stderr)
            self.assertIn("--rows 4800", result.stderr)
            self.assertIn("--seed 42", result.stderr)

    def test_d_legs_set_the_disable_env_and_claim_it_back(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-D1,clip-text-D2")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            d1 = _manifest(out_dir, "clip-text-D1")
            self.assertEqual(d1["kernels_disabled"], D1_KEYS_CLIP.split(","), d1)
            d2 = _manifest(out_dir, "clip-text-D2")
            self.assertEqual(d2["kernels_disabled"], D2_KEYS.split(","), d2)
            # BOTH halves of the claim reach the command line: the env var
            # itself AND the argv claim that must match it.
            unescaped = result.stderr.replace("\\,", ",")
            self.assertIn(f"JAMMI_KERNELS_DISABLE={D1_KEYS_CLIP}", unescaped)
            self.assertIn(f"--expect-kernels-disabled {D1_KEYS_CLIP}", unescaped)

    def test_d1_names_gelu_erf_fused_on_htsat_and_never_on_a_clip_tower(self):
        """`gelu_erf_fused` has no `admit` call site on either CLIP tower
        (their MLP activation is `quick_gelu`, which has no fused seam), so
        naming it there would make `unmatched_disables()` non-empty and
        `finetune-run` would refuse the leg as INVALID -- every CLIP D1 leg
        would fail before producing a datum. This test pins the per-tower
        split so a future edit cannot quietly re-unify the two lists."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-D1,clip-vision-D1,htsat-D1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertEqual(
                _manifest(out_dir, "htsat-D1")["kernels_disabled"],
                D1_KEYS_HTSAT.split(","),
            )
            for leg_id in ("clip-text-D1", "clip-vision-D1"):
                keys = _manifest(out_dir, leg_id)["kernels_disabled"]
                self.assertEqual(keys, D1_KEYS_CLIP.split(","), leg_id)
                self.assertNotIn(
                    "gelu_erf_fused", keys,
                    f"{leg_id}: a CLIP tower has no gelu_erf seam, so disabling that key would "
                    "invalidate the leg rather than force an eager arm",
                )
            # And D2 is the SAME single key on every tower -- it isolates
            # C-LORA, which is architecture-independent.
            for tower in TOWERS:
                self.assertEqual(
                    _manifest(out_dir, f"{tower}-D1")["kernels_disabled"][:2],
                    D2_KEYS.split(",") + ["layer_norm_fused"],
                    f"{tower}-D1 must be a superset of D2's key, in a stable order",
                )

    def test_a_legs_carry_no_disable_env_and_make_no_claim(self):
        """The negative control for the test above: an A leg is the FUSED
        arm, so neither the env var nor the claim may appear on its command
        line -- an inherited `JAMMI_KERNELS_DISABLE` would silently turn the
        decision leg into a second eager twin."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1,clip-text-A2")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for leg_id in ("clip-text-A1", "clip-text-A2"):
                manifest = _manifest(out_dir, leg_id)
                self.assertEqual(manifest["kernels_disabled"], [], manifest)
            self.assertNotIn("JAMMI_KERNELS_DISABLE", result.stderr)
            self.assertNotIn("--expect-kernels-disabled", result.stderr)

    def test_a_leg_dtypes_follow_the_contract_table(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for tower in TOWERS:
                self.assertEqual(_manifest(out_dir, f"{tower}-A1")["dtype"], "f32")
                self.assertEqual(_manifest(out_dir, f"{tower}-A2")["dtype"], "bf16")
                # Both D legs are F32 twins of A1 -- a bf16 eager twin would
                # not be A1's twin at all.
                self.assertEqual(_manifest(out_dir, f"{tower}-D1")["dtype"], "f32")
                self.assertEqual(_manifest(out_dir, f"{tower}-D2")["dtype"], "f32")

    def test_each_tower_uses_its_full_lora_site_set(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            clip = ["in_proj", "out_proj", "c_fc", "c_proj"]
            clap = [
                "query", "key", "value", "attention_output", "intermediate_dense",
                "output_dense", "reduction", "linear1", "linear2",
            ]
            for leg in LEG_SUFFIXES:
                self.assertEqual(_manifest(out_dir, f"clip-text-{leg}")["target_modules"], clip)
                self.assertEqual(_manifest(out_dir, f"clip-vision-{leg}")["target_modules"], clip)
                self.assertEqual(_manifest(out_dir, f"htsat-{leg}")["target_modules"], clap)

    def test_chatty_fake_nsys_stdout_never_pollutes_the_report_file(self):
        """The capture path's own regression class: the fake nsys prints
        stdout noise on BOTH subcommands; the WRITTEN report file must
        still parse as clean JSON carrying the real envelope shape."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-vision-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("fake_nsys: chatty stdout noise", result.stdout)
            report_text = Path(out_dir, "clip-vision-A1", "run_n.json").read_text()
            self.assertNotIn("fake_nsys", report_text)
            report = json.loads(report_text)  # raises if any noise leaked in
            tier = report["tiers"]["finetune_run"]
            self.assertIn("train_run_wall_s", tier)
            self.assertEqual(tier["task"], "image_embedding")

    def test_media_front_end_wall_is_read_off_the_report_on_both_arms(self):
        """P1-b(v): the direct front-end timer is READ off each run's report,
        never back-filled from `wall - busy` (a difference is not a
        measurement). A media leg carries a real number for both the N and
        the M run; a TEXT leg carries `null` on both -- the trainer reports
        `Duration::ZERO` there by construction, which is not a measurement
        of anything, and a fabricated 0.0 would be indistinguishable from a
        real zero. Both arms are asserted so the reader cannot pass by
        degrading everything to null."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="htsat-A1,clip-vision-A1,clip-text-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for leg_id in ("htsat-A1", "clip-vision-A1"):
                front = _manifest(out_dir, leg_id)["media_front_end_wall_s"]
                for run in ("n", "m"):
                    self.assertIsInstance(front[run], float, f"{leg_id} {run}: {front}")
                    self.assertGreater(front[run], 0.0, f"{leg_id} {run}: {front}")
                self.assertLess(
                    front["n"], front["m"],
                    f"{leg_id}: the M run decodes 6x the rows, so its front-end wall must exceed "
                    f"the N run's: {front}",
                )
            self.assertEqual(
                _manifest(out_dir, "clip-text-A1")["media_front_end_wall_s"],
                {"n": None, "m": None},
                "a text leg has no media front end at all",
            )

    def test_the_text_legs_heldout_split_is_disjoint_from_BOTH_train_corpora(self):
        """Contract v2.3 §D4 item 3, proven by DRIVING THE REAL PRODUCER.

        `gen_fixed_width_corpus.py` draws `rows + heldout_rows` rows from one
        seeded stream and slices the held-out half off the END, so at
        `--rows R` the held-out rows are stream indices `[R, R+8)`. The two
        text runs share `--seed 42` and differ only in `--rows`, which makes
        the N run's held-out rows (indices `[800, 808)`) BYTE-IDENTICAL to
        rows the M run TRAINS on (indices `[0, 4800)`). Only the M corpus's
        own held-out slice sits past both train corpora.

        This test takes the driver's OWN printed command lines -- the two
        producer invocations AND the `--heldout-ids`/`--heldout-jsonl` paths
        it hands `finetune-run` -- executes the (hermetic, CPU-only)
        producers for real, and asserts no held-out text appears in either
        train file. It then asserts the CONVERSE for the `corpus_n` split
        the driver used to pass, so the check is demonstrably not vacuous:
        if this test could pass with either choice, it would prove nothing.

        `pin_workload_steps=True`: the byte-identical-indices argument above
        is stated in terms of the literal 100/600-step row counts, so this
        test must run against the real defaults, not this suite's small
        override.
        """
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1", pin_workload_steps=True)
            self.assertEqual(result.returncode, 0, _fail_msg(result))

            producer_cmds = []
            run_cmds = []
            for line in result.stderr.splitlines():
                if not line.startswith("+ "):
                    continue
                argv = shlex.split(line[2:])
                if any(a.endswith("gen_fixed_width_corpus.py") for a in argv):
                    producer_cmds.append(argv)
                elif "--heldout-ids" in argv:
                    run_cmds.append(argv)
            self.assertEqual(len(producer_cmds), 2, result.stderr)
            self.assertEqual(len(run_cmds), 2, result.stderr)

            # Run the producers for real, exactly as the driver would.
            for argv in producer_cmds:
                produced = subprocess.run(argv, capture_output=True, text=True, timeout=600)
                self.assertEqual(produced.returncode, 0,
                                 f"{argv}\n{produced.stdout}\n{produced.stderr}")

            heldout_paths = set()
            train_paths = []
            for argv in run_cmds:
                heldout_paths.add(argv[argv.index("--heldout-jsonl") + 1])
                heldout_paths.add(argv[argv.index("--heldout-ids") + 1])
                train_paths.append(argv[argv.index("--train-jsonl") + 1])
            # BOTH runs are handed the SAME held-out split, and it is the M
            # corpus's -- the only one past both train corpora.
            self.assertEqual(
                sorted(heldout_paths),
                sorted([
                    os.path.join(out_dir, "clip-text-A1", "corpus_m", "heldout_ids.txt"),
                    os.path.join(out_dir, "clip-text-A1", "corpus_m", "heldout_triplets.jsonl"),
                ]),
                result.stderr,
            )

            def texts(path):
                out = set()
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        out.update((row["anchor_text"], row["positive_text"],
                                    row["negative_text"]))
                return out

            heldout_m = texts(
                os.path.join(out_dir, "clip-text-A1", "corpus_m", "heldout_triplets.jsonl")
            )
            self.assertEqual(len(heldout_m), 3 * 8, "8 held-out triplets, 3 distinct texts each")
            train_n_texts = texts(train_paths[0])
            train_m_texts = texts(train_paths[1])
            # The producers really did run at the pinned row counts (batch 8
            # x steps 100/600) -- otherwise "no overlap" could just mean
            # "one of these files is empty".
            def rows(path):
                with open(path, encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())

            self.assertEqual(rows(train_paths[0]), 800)
            self.assertEqual(rows(train_paths[1]), 4800)
            self.assertEqual(heldout_m & train_n_texts, set(),
                             "a held-out text appears in the N train corpus")
            self.assertEqual(heldout_m & train_m_texts, set(),
                             "a held-out text appears in the M train corpus")

            # Non-vacuity: the `corpus_n` split -- what this driver passed
            # before §D4 item 3 -- IS inside the M train corpus. Were the
            # driver to go back to it, the assertions above would fire.
            heldout_n = texts(
                os.path.join(out_dir, "clip-text-A1", "corpus_n", "heldout_triplets.jsonl")
            )
            self.assertTrue(
                heldout_n and heldout_n <= train_m_texts,
                "the N corpus's held-out rows must be a SUBSET of the M train corpus -- if they "
                "were not, this test would pass no matter which split the driver chose",
            )

    def test_esc_088_corpus_producer_stdout_never_pollutes_the_traced_paths(self):
        """esc-088 (`.jammi/escapes.jsonl`): `provision_corpus` used to hand
        `run_leg` its 4-tuple ("train_n<TAB>train_m<TAB>heldout_ids<TAB>
        heldout_jsonl") over ITS OWN stdout
        (`corpus_line="$(provision_corpus ...)"` / `IFS=$'\\t' read`), and
        the three real corpus producers ALSO print a one-line summary to
        THEIR OWN stdout. On a real pod run that summary line became
        `train_n` and every other field (`heldout_ids` included) came out
        empty, and every one of the 12 real legs refused with clap's own
        "a value is required for '--heldout-ids <HELDOUT_IDS>' but none was
        supplied" -- while the hermetic dry-run suite stayed green, because
        under the OLD dry-run behaviour the corpus producers never actually
        ran (a touch-empty stand-in took their place), so the capture bug
        never had a producer's real stdout to be polluted BY.

        This test closes that gap for real: the corpus producers now
        execute for REAL under DRY_RUN too (never a fake stand-in -- they
        are CPU-hermetic and cheap, see `run_corpus_cmd`), so this drives
        the actual `provision_corpus` capture path with actual producer
        stdout. It asserts, off the driver's OWN printed (real,
        would-be-production) command line, that `--train-jsonl`,
        `--heldout-ids` and `--heldout-jsonl` each name a REAL, EXISTING,
        NON-EMPTY file for every N/M run across all three towers -- proving
        the mechanism, not merely that some file happens to exist (a
        touch-empty placeholder would pass an `isfile` check but fail the
        non-empty one)."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1,clip-vision-A1,htsat-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for leg_id in ("clip-text-A1", "clip-vision-A1", "htsat-A1"):
                manifest = _manifest(out_dir, leg_id)
                self.assertEqual(manifest["status"], "ok", manifest)
                self.assertEqual(manifest["reason"], "", manifest)

            run_cmds = []
            for line in result.stderr.splitlines():
                if not line.startswith("+ "):
                    continue
                argv = shlex.split(line[2:])
                if "--heldout-ids" in argv:
                    run_cmds.append(argv)
            # 3 towers x 2 runs (N, M) each.
            self.assertEqual(len(run_cmds), 6, result.stderr)
            for argv in run_cmds:
                for flag in ("--train-jsonl", "--heldout-ids", "--heldout-jsonl"):
                    value = argv[argv.index(flag) + 1]
                    self.assertTrue(value, f"{flag} is empty on the traced command line: {argv}")
                    self.assertTrue(
                        os.path.isfile(value),
                        f"{flag}={value!r} does not name a real file -- a corpus producer's own "
                        f"stdout must never leak into this value: {argv}",
                    )
                    self.assertGreater(
                        os.path.getsize(value), 0,
                        f"{flag}={value!r} exists but is EMPTY -- the real producer must have "
                        "actually written real content here, not a touch-empty stand-in",
                    )
            # Non-vacuity: the file-existence/non-empty checks above would
            # ALSO pass if some OTHER mechanism (not the real producer) had
            # written those files. Assert the real producer's own one-line
            # summary actually reached this driver's stderr during THIS
            # run, proving the producer process itself executed rather than
            # merely that its declared output path happens to be populated.
            self.assertIn("gen_fixed_width_corpus: wrote", result.stderr)
            self.assertIn("gen_fixed_shape_image_corpus: wrote", result.stderr)
            self.assertIn("gen_fixed_length_audio_corpus: wrote", result.stderr)

    def test_each_legs_report_satisfies_the_positive_proof_equation(self):
        """The dry-run reports the capture path actually produces must be
        SHAPED like a real leg's, or this whole suite proves the plumbing
        while leaving the plumbing's payload untested.

        `profile_421_merge.py` checks `fused + eager == fusible_site_census
        x steps_measured` per key on every real leg; the same equation is
        asserted here on what the stub emits, on both arms that matter: an A
        leg (fused non-zero for lora/ln) and a D leg (fused exactly 0 for
        every key it claims to have disabled). The `gelu_erf_fused` key is
        the load-bearing one — a CLIP tower's `quick_gelu` has no fused
        seam, so its census is 0 and its counters must read 0/0, while HTSAT
        carries a real per-forward count.
        """
        keys = {
            "lora_linear_fused": (
                "lora_linear_fused_dispatches",
                "lora_linear_eager_dispatches",
                "lora_sites_wrapped",
            ),
            "layer_norm_fused": ("ln_fused_dispatches", "ln_eager_dispatches", "layer_norms"),
            "gelu_erf_fused": (
                "gelu_fused_dispatches",
                "gelu_eager_dispatches",
                "gelu_seam_calls_per_forward",
            ),
        }
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1,htsat-A1,clip-text-D1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            for leg_id, disabled in (
                ("clip-text-A1", ()),
                ("htsat-A1", ()),
                ("clip-text-D1", tuple(D1_KEYS_CLIP.split(","))),
            ):
                for run in ("n", "m"):
                    path = os.path.join(out_dir, leg_id, f"run_{run}.json")
                    with open(path, encoding="utf-8") as f:
                        tier = json.load(f)["tiers"]["finetune_run"]
                    # The pinned convention the equation is defined under.
                    self.assertEqual(tier["epochs"], 1, f"{leg_id} {run}")
                    self.assertEqual(tier["grad_accum"], 1, f"{leg_id} {run}")
                    steps = tier["steps_measured"]
                    census = tier["fusible_site_census"]
                    for key, (fused_f, eager_f, census_f) in keys.items():
                        calls = census[census_f]
                        self.assertEqual(
                            tier[fused_f] + tier[eager_f], calls * steps,
                            f"{leg_id} run_{run}: {key} violates fused + eager == "
                            f"{census_f} x steps_measured",
                        )
                        if key in disabled:
                            self.assertEqual(tier[fused_f], 0,
                                             f"{leg_id} run_{run}: {key} is disabled")
                        elif key != "gelu_erf_fused":
                            self.assertGreater(tier[fused_f], 0,
                                               f"{leg_id} run_{run}: {key} on a fused arm")
                    # The one structural split the stub must mirror.
                    expected_gelu = 0 if leg_id.startswith("clip") else 9
                    self.assertEqual(
                        census["gelu_seam_calls_per_forward"], expected_gelu,
                        f"{leg_id}: a CLIP tower's quick_gelu has no fused seam (census 0); "
                        f"HTSAT's Swin MLP does",
                    )

    def test_legs_only_filter_runs_exactly_the_named_legs(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="htsat-A2,clip-vision-D2")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            ran = {
                leg_id for leg_id in ALL_LEG_IDS
                if os.path.isfile(os.path.join(out_dir, leg_id, "manifest.json"))
            }
            self.assertEqual(ran, {"htsat-A2", "clip-vision-D2"}, result.stdout)
            for skipped in sorted(set(ALL_LEG_IDS) - ran):
                self.assertIn(f"skipping {skipped}", result.stdout)

    def test_legs_only_typo_refuses_before_any_leg_runs(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="htsat-A2,htsat-A9999-TYPO")
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("::error::", result.stderr)
            self.assertIn("htsat-A9999-TYPO", result.stderr)
            self.assertEqual(list(Path(out_dir).rglob("manifest.json")), [], _fail_msg(result))

    def test_existing_manifest_for_a_leg_about_to_run_refuses(self):
        with tempfile.TemporaryDirectory() as out_dir:
            first = run_dry(out_dir, legs_only="htsat-D1")
            self.assertEqual(first.returncode, 0, _fail_msg(first))
            manifest_path = os.path.join(out_dir, "htsat-D1", "manifest.json")
            before = Path(manifest_path).read_text()

            second = run_dry(out_dir, legs_only="htsat-D1")
            self.assertNotEqual(second.returncode, 0, _fail_msg(second))
            self.assertIn("fresh OUT_DIR", second.stderr)
            self.assertEqual(Path(manifest_path).read_text(), before)

            # A leg never yet run in this OUT_DIR is unaffected.
            third = run_dry(out_dir, legs_only="htsat-D2")
            self.assertEqual(third.returncode, 0, _fail_msg(third))

    def test_checkpoint_weights_sha256_agrees_within_a_tower_and_differs_across_towers(self):
        """Unit-467 finding R3, witnessed through the manifest: the stub
        mirrors production`s real shape (both CLIP towers share ONE
        checkpoint directory, HTSAT a different one), so every leg of one
        tower must report the SAME sha and a leg of a DIFFERENT tower must
        report a DIFFERENT one -- proving the field genuinely reaches the
        manifest per run, not merely that the key exists."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            clip_shas = set()
            for tower in ("clip-text", "clip-vision"):
                for leg in LEG_SUFFIXES:
                    sha = _manifest(out_dir, f"{tower}-{leg}")["checkpoint_weights_sha256"]
                    self.assertEqual(sha["n"], sha["m"], f"{tower}-{leg}: {sha}")
                    clip_shas.add(sha["n"])
            self.assertEqual(len(clip_shas), 1, f"both CLIP towers must share one sha: {clip_shas}")
            htsat_shas = set()
            for leg in LEG_SUFFIXES:
                sha = _manifest(out_dir, f"htsat-{leg}")["checkpoint_weights_sha256"]
                self.assertEqual(sha["n"], sha["m"], f"htsat-{leg}: {sha}")
                htsat_shas.add(sha["n"])
            self.assertEqual(len(htsat_shas), 1, f"every htsat leg must share one sha: {htsat_shas}")
            self.assertNotEqual(
                clip_shas, htsat_shas,
                "CLIP and HTSAT load DIFFERENT checkpoint directories in production, so their "
                "witnessed shas must differ",
            )


class CheckpointIdentityPreflightTests(unittest.TestCase):
    """Unit-467 finding R3(i): `_checkpoint_identity_probe` refuses BEFORE
    any leg runs when `$MODEL_DIR_CLIP`/`$MODEL_DIR_CLAP` hold the wrong
    checkpoint shape -- exercised here against REAL temp dirs, under
    `PROFILE_421_LEGS_DRY_RUN=1` so no GPU/nsys/bench is needed, proving the
    probe runs even in dry-run mode (unlike the rest of the preflight,
    which is a DRY_RUN no-op)."""

    def test_a_model_dir_clip_carrying_config_json_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip"
            clip_dir.mkdir()
            (clip_dir / "config.json").write_text("{}", encoding="utf-8")
            with tempfile.TemporaryDirectory() as out_dir:
                result = run_dry(
                    out_dir, legs_only="clip-text-A1",
                    extra_env={"MODEL_DIR_CLIP": str(clip_dir)},
                )
                self.assertNotEqual(result.returncode, 0, _fail_msg(result))
                self.assertIn("MODEL_DIR_CLIP", result.stderr)
                self.assertIn("config.json", result.stderr)
                self.assertEqual(
                    list(Path(out_dir).rglob("manifest.json")), [], _fail_msg(result),
                )

    def test_a_model_dir_clip_carrying_model_safetensors_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip"
            clip_dir.mkdir()
            (clip_dir / "model.safetensors").write_bytes(b"")
            with tempfile.TemporaryDirectory() as out_dir:
                result = run_dry(
                    out_dir, legs_only="clip-text-A1",
                    extra_env={"MODEL_DIR_CLIP": str(clip_dir)},
                )
                self.assertNotEqual(result.returncode, 0, _fail_msg(result))
                self.assertIn("MODEL_DIR_CLIP", result.stderr)
                self.assertIn("model.safetensors", result.stderr)

    def test_a_model_dir_clap_missing_preprocessor_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            clap_dir = Path(tmp) / "clap"
            clap_dir.mkdir()
            (clap_dir / "config.json").write_text("{}", encoding="utf-8")
            (clap_dir / "model.safetensors").write_bytes(b"")
            # preprocessor_config.json deliberately absent.
            with tempfile.TemporaryDirectory() as out_dir:
                result = run_dry(
                    out_dir, legs_only="htsat-A1",
                    extra_env={"MODEL_DIR_CLAP": str(clap_dir)},
                )
                self.assertNotEqual(result.returncode, 0, _fail_msg(result))
                self.assertIn("MODEL_DIR_CLAP", result.stderr)
                self.assertIn("preprocessor_config.json", result.stderr)
                self.assertEqual(
                    list(Path(out_dir).rglob("manifest.json")), [], _fail_msg(result),
                )

    def test_a_well_shaped_pair_of_real_temp_dirs_is_not_refused(self):
        """The non-vacuity control: real, EXISTING directories that carry
        the correct shape must not trip the probe -- otherwise the two
        refusal tests above would prove nothing about which shape is
        actually being checked for."""
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp) / "clip"
            clip_dir.mkdir()
            (clip_dir / "open_clip_config.json").write_text("{}", encoding="utf-8")
            (clip_dir / "open_clip_model.safetensors").write_bytes(b"")
            (clip_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            clap_dir = Path(tmp) / "clap"
            clap_dir.mkdir()
            (clap_dir / "config.json").write_text("{}", encoding="utf-8")
            (clap_dir / "model.safetensors").write_bytes(b"")
            (clap_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            with tempfile.TemporaryDirectory() as out_dir:
                result = run_dry(
                    out_dir, legs_only="clip-text-A1,htsat-A1",
                    extra_env={"MODEL_DIR_CLIP": str(clip_dir), "MODEL_DIR_CLAP": str(clap_dir)},
                )
                self.assertEqual(result.returncode, 0, _fail_msg(result))


class AmbientDisableEnvGuardTests(unittest.TestCase):
    """Unit-467 finding F1, driver half: `JAMMI_KERNELS_DISABLE` reaching
    this driver`s OWN environment (as opposed to being scoped, per D leg,
    onto a single child invocation) must refuse before any leg runs -- an
    ambient value here would otherwise leak into every A leg`s environment
    too. Deliberately does NOT go through `run_dry` (which always pops the
    var defensively), so the ambient value genuinely reaches the driver."""

    def test_an_ambient_kernels_disable_env_var_refuses_before_any_leg_runs(self):
        with tempfile.TemporaryDirectory() as out_dir:
            env = dict(os.environ)
            env["PROFILE_421_LEGS_DRY_RUN"] = "1"
            env["OUT_DIR"] = out_dir
            env["NSYS_BIN"] = "/nonexistent/nsys-DRY-RUN-PLACEHOLDER"
            env["BENCH_BIN"] = "/nonexistent/jammi-bench-DRY-RUN-PLACEHOLDER"
            env["JAMMI_KERNELS_DISABLE"] = "lora_linear_fused"
            result = subprocess.run(
                ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=300
            )
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("JAMMI_KERNELS_DISABLE", result.stderr)
            self.assertIn("lora_linear_fused", result.stderr)
            self.assertEqual(list(Path(out_dir).rglob("manifest.json")), [], _fail_msg(result))

    def test_an_unset_ambient_var_is_the_ordinary_unaffected_case(self):
        """The non-vacuity control: the SAME invocation with the var
        genuinely absent (never merely empty) must run normally."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))


class DLegExpectedDisablesEqualityTests(unittest.TestCase):
    """Unit-467 finding F1, D-leg half: `_check_expected_disables` must
    refuse a D leg whose report's `kernels_disabled_requested` is a STRICT
    SUPERSET of what it claimed on `--expect-kernels-disabled` -- a SUBSET
    test (claimed keys present in `requested`, extras allowed) would let an
    extra ambient key through, force-eagering an op the leg assumed fused
    and OVERSTATING the realized gain. Driven through the REAL
    `PROFILE_421_LEGS_DRY_RUN_EXTRA_REQUESTED_KEY` lever the hermetic fake
    bench stub reads (never a re-implementation of the check), so this
    drives the actual driver code path, not a python mirror of it."""

    def test_a_d_leg_report_with_an_extra_requested_key_is_refused(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir, legs_only="clip-text-D2",
                extra_env={"PROFILE_421_LEGS_DRY_RUN_EXTRA_REQUESTED_KEY": "layer_norm_fused"},
            )
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifest = _manifest(out_dir, "clip-text-D2")
            self.assertEqual(manifest["status"], "invalid", manifest)
            self.assertIn("layer_norm_fused", manifest["reason"])

    def test_a_d_leg_report_with_exactly_the_claimed_keys_still_succeeds(self):
        """The non-vacuity control: the SAME leg with the injection lever
        left unset (the ordinary, uncontaminated case) must pass -- proving
        the refusal above fires on the mismatch, not on this D leg shape in
        general."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-D2")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifest = _manifest(out_dir, "clip-text-D2")
            self.assertEqual(manifest["status"], "ok", manifest)


class CorpusPostConditionTests(unittest.TestCase):
    """`run_leg`'s post-condition on a REPORTED-success `provision_corpus`:
    a producer that exits 0 but leaves one of the four corpus paths
    missing/empty must mark the leg INVALID by name, before `run_traced`
    ever sees it -- rather than letting an empty `--train-jsonl` (etc.)
    reach a real `finetune-run` and get blamed on the wrong layer.

    `gen_fixed_width_corpus.py` refuses `--rows 0` outright (so it cannot
    itself produce an EMPTY-but-successful corpus), so this class drives
    the driver's own test-only `PROFILE_421_LEGS_DRY_RUN_TRUNCATE_CORPUS_VAR`
    lever -- never set by the real leg sweep -- which truncates one of
    `provision_corpus`'s own four output variables to empty AFTER it
    reported success, proving the post-condition check fires on the
    mechanism rather than being assumed."""

    def test_each_truncated_corpus_var_marks_the_leg_invalid_by_name(self):
        for var in ("train_n", "train_m", "heldout_ids", "heldout_jsonl"):
            with self.subTest(var=var):
                with tempfile.TemporaryDirectory() as out_dir:
                    result = run_dry(
                        out_dir, legs_only="clip-text-A1",
                        extra_env={"PROFILE_421_LEGS_DRY_RUN_TRUNCATE_CORPUS_VAR": var},
                    )
                    self.assertEqual(result.returncode, 0, _fail_msg(result))
                    manifest = _manifest(out_dir, "clip-text-A1")
                    self.assertEqual(manifest["status"], "invalid", manifest)
                    self.assertIn(f"{var} path is missing/empty", manifest["reason"])

    def test_the_lever_left_unset_is_the_ordinary_unaffected_case(self):
        """The non-vacuity control: the SAME leg with the lever genuinely
        unset (the ordinary case) must pass -- proving the refusal above
        fires on the truncation, not on this leg shape in general."""
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, legs_only="clip-text-A1")
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            manifest = _manifest(out_dir, "clip-text-A1")
            self.assertEqual(manifest["status"], "ok", manifest)


def _write_fake_bench_stub(path: Path, *, missing_flag: str | None = None) -> None:
    """A fake `$BENCH_BIN` answering the two subcommands the preflight uses:
    `provenance` (the driver's own build_sha cross-check, which runs BEFORE
    preflight) and `finetune-run --help`.

    `missing_flag` omits exactly ONE flag from the help text, so each
    preflight arm can be driven independently -- proving the driver's
    per-flag messages are genuinely distinguishable rather than one generic
    failure wearing different words.
    """
    flags = [
        "--model-dir", "--arm", "--task", "--train-jsonl", "--heldout-ids",
        "--heldout-jsonl", "--objective", "--lora-init", "--expect-kernels-disabled",
        "--target-modules", "--backbone-dtype", "--max-seq-length", "--eval-cadence",
    ]
    if missing_flag is not None:
        assert missing_flag in flags, missing_flag
        flags = [f for f in flags if f != missing_flag]
    # `--lora-init`'s accepted value is probed separately by the driver, so
    # the stub prints it exactly when the flag itself is present.
    help_lines = [f"  {f} <VALUE>" for f in flags]
    if "--lora-init" in flags:
        help_lines.append("      zeros_b or gaussian")
    help_text = "\n".join(["Usage: jammi-bench finetune-run [OPTIONS]", *help_lines])
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "$1" = "provenance" ]; then\n'
        f'  echo \'{{"build_sha": "{_real_head()}"}}\'\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "finetune-run" ]; then\n'
        "  cat <<'HELP_EOF'\n" + help_text + "\nHELP_EOF\n"
        "  exit 0\n"
        "fi\n"
        'echo "fake_bench: unexpected subcommand $1" >&2\n'
        "exit 3\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class PreflightArmTests(unittest.TestCase):
    """The REAL (non-dry) preflight, driven against a fake `$BENCH_BIN`.

    The producers' half of the preflight runs the THREE REAL committed
    producers (no stub), so the pass case below is a genuine end-to-end
    proof that `gen_fixed_width_corpus.py`,
    `gen_fixed_shape_image_corpus.py` and `gen_fixed_length_audio_corpus.py`
    each accept `--heldout-rows`/`--heldout-batch` AND emit the two files
    this driver passes to `--heldout-ids`/`--heldout-jsonl` under exactly
    those names.
    """

    def _run_preflight(self, tmp, *, missing_flag=None):
        bench = Path(tmp) / "fake_bench.sh"
        _write_fake_bench_stub(bench, missing_flag=missing_flag)
        env = dict(os.environ)
        env["PROFILE_421_LEGS_DRY_RUN"] = "0"
        env["PROFILE_421_LEGS_PREFLIGHT_ONLY"] = "1"
        env["OUT_DIR"] = str(Path(tmp) / "out")
        env["BENCH_BIN"] = str(bench)
        # `nsys` is never invoked on the preflight path, but the driver
        # checks it is executable before preflight -- point it at the stub.
        env["NSYS_BIN"] = str(bench)
        env["MODEL_DIR_CLIP"] = str(Path(tmp) / "clip")
        env["MODEL_DIR_CLAP"] = str(Path(tmp) / "clap")
        env.pop("JAMMI_KERNELS_DISABLE", None)
        return subprocess.run(
            ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=300
        )

    def test_preflight_passes_when_every_flag_and_producer_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_preflight(tmp)
            self.assertEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("preflight OK", result.stdout)
            self.assertIn("all three producers emit a held-out split", result.stdout)
            self.assertIn("exiting before the leg sweep", result.stdout)
            self.assertEqual(
                list(Path(tmp, "out").rglob("manifest.json")), [],
                "PREFLIGHT_ONLY must run NO leg",
            )

    def test_each_missing_flag_is_reported_on_its_own(self):
        """One arm per flag the legs depend on -- the two this unit adds
        (`--expect-kernels-disabled`, `--lora-init`) and the two it PINS
        (`--task`, `--objective`, pre-existing but silently workload-
        changing if absent). Each refusal must name ITS flag and no other,
        which is what proves these are four distinguishable outcomes rather
        than one generic "something failed"."""
        for flag in ("--expect-kernels-disabled", "--lora-init", "--task", "--objective"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                result = self._run_preflight(tmp, missing_flag=flag)
                self.assertNotEqual(result.returncode, 0, _fail_msg(result))
                self.assertIn("precondition(s) not yet landed", result.stderr)
                self.assertIn(f"has no {flag} flag", result.stderr)
                for other in ("--expect-kernels-disabled", "--lora-init", "--task", "--objective"):
                    if other != flag and not other.startswith(flag):
                        self.assertNotIn(
                            f"has no {other} flag", result.stderr,
                            f"the {flag} arm must not also blame {other}",
                        )

    def test_a_missing_heldout_flag_on_the_binary_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_preflight(tmp, missing_flag="--heldout-jsonl")
            self.assertNotEqual(result.returncode, 0, _fail_msg(result))
            self.assertIn("has no --heldout-jsonl flag", result.stderr)


if __name__ == "__main__":
    unittest.main()
