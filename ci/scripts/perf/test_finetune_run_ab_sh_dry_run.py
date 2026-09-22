#!/usr/bin/env python3
"""Hermetic `FINETUNE_RUN_AB_DRY_RUN=1` smoke test for `finetune_run_ab.sh`
itself — the shell PRODUCER executed, not only its merge stage
(`test_finetune_ab_sh_dry_run.py`'s doctrine, applied to the run-level
driver).

What only running the script shows: the order each seed's legs run in, the
command line each leg is handed, and — with the torch arm on — that a torch
leg is the SAME run description given to the other producer, started from the
adapter a jammi leg of its seed dumped, and refused outright when the pairing
premise cannot hold. None of that is visible from `ab_merge.py`'s suite, which
reads fixture leg files and never builds an argv.

`FINETUNE_RUN_AB_DRY_RUN=1` makes the pipeline hermetic: no `cargo build`, no
venv probe, no GPU, no network — every leg's command is PRINTED, never
executed, and every leg writes a `{"tool":"dry-run",...}` stub so the merge
stage still runs end to end. Drives the REAL `bash
ci/scripts/perf/finetune_run_ab.sh` subprocess.

Run: `python3 ci/scripts/perf/test_finetune_run_ab_sh_dry_run.py`
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "finetune_run_ab.sh")

# The flags that describe the RUN: both producers must be handed every one,
# with the same value, on every leg of a seed.
RUN_FLAGS = (
    "--model-dir",
    "--train-jsonl",
    "--heldout-ids",
    "--heldout-jsonl",
    "--seed",
    "--batch",
    "--objective",
    "--early-stopping-patience",
    "--backbone-dtype",
    "--lora-dropout",
    "--cuda",
)


def run_dry(out_dir, **extra_env):
    env = dict(os.environ)
    env.update(
        FINETUNE_RUN_AB_DRY_RUN="1",
        FINETUNE_RUN_AB_OUT_DIR=out_dir,
        FINETUNE_RUN_AB_SEEDS="1,2",
        # A dry run has no lr=0 control legs to merge; the opt-out is the
        # merger's own recorded one, not a way around it.
        FINETUNE_RUN_AB_ALLOW_NO_LR0="1",
    )
    # Whatever venv the host names is irrelevant to a dry run, which must not
    # probe one.
    env.pop("TORCH_VENV", None)
    env.update(extra_env)
    return subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=120)


def leg_commands(stdout):
    """`{(seed, arm, repeat): argv}` in the order the script printed them."""
    legs = {}
    for line in stdout.splitlines():
        if not line.startswith("--- seed"):
            continue
        label, _, command = line.partition(": ")
        seed, arm, repeat = label[len("--- seed") :].split("/")
        legs[(seed, arm, repeat)] = shlex.split(command)
    return legs


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def derive_command(stdout):
    """The `kernel-arm` argv the script derives the reference arm with."""
    for line in stdout.splitlines():
        if line.startswith("--- kernel-arm: "):
            return shlex.split(line[len("--- kernel-arm: ") :])
    raise AssertionError("the script never printed its kernel-arm derive")


class JammiOnlyDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.result = run_dry(cls._tmp.name)
        cls.legs = leg_commands(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_dry_run_hands_the_legs_to_the_ladder_and_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")
        self.assertIn("ladder train-run", self.result.stdout)
        self.assertIn("--from resident-reference --to resident", self.result.stdout)
        raw = os.listdir(os.path.join(self._tmp.name, "raw"))
        self.assertIn("resident__seed1__r1.json", raw)
        self.assertIn("resident-reference__seed1__r2.json", raw)

    def test_each_seeds_legs_run_order_balanced(self):
        self.assertEqual(
            list(self.legs),
            [
                (seed, arm, repeat)
                for seed in ("1", "2")
                for arm, repeat in (("fused", "r1"), ("alloff", "r1"), ("alloff", "r2"), ("fused", "r2"))
            ],
        )

    def test_no_leg_names_torch(self):
        self.assertNotIn("torch", {arm for _seed, arm, _repeat in self.legs})

    def test_the_reference_arm_is_derived_on_the_sites_the_legs_adapt(self):
        derive = derive_command(self.result.stdout)
        self.assertEqual(derive[1:2], ["kernel-arm"])
        self.assertEqual(flag_value(derive, "--off"), "flash-attention,adam-w")
        # No selector named: the derive and every leg take the one constant.
        self.assertIsNone(flag_value(derive, "--target-modules"))
        for argv in self.legs.values():
            self.assertIsNone(flag_value(argv, "--target-modules"))


class TargetModulesDryRun(unittest.TestCase):
    """`FINETUNE_RUN_AB_TARGET_MODULES` reaches the derive and every leg of every arm."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.result = run_dry(
            cls._tmp.name,
            FINETUNE_RUN_AB_TORCH="1",
            FINETUNE_RUN_AB_LORA_DROPOUT="0.0",
            FINETUNE_RUN_AB_TARGET_MODULES="Wqkv,Wo",
        )
        cls.legs = leg_commands(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_derive_and_every_leg_name_the_same_sites(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")
        self.assertEqual(flag_value(derive_command(self.result.stdout), "--target-modules"), "Wqkv,Wo")
        self.assertTrue(self.legs)
        for key, argv in self.legs.items():
            self.assertEqual(flag_value(argv, "--target-modules"), "Wqkv,Wo", key)


class TorchArmDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.result = run_dry(
            cls._tmp.name, FINETUNE_RUN_AB_TORCH="1", FINETUNE_RUN_AB_LORA_DROPOUT="0.0"
        )
        cls.legs = leg_commands(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_dry_run_merges_and_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")

    def test_the_torch_legs_sit_in_the_middle_of_each_seeds_block(self):
        self.assertEqual(
            list(self.legs),
            [
                (seed, arm, repeat)
                for seed in ("1", "2")
                for arm, repeat in (
                    ("fused", "r1"),
                    ("alloff", "r1"),
                    ("torch", "r1"),
                    ("torch-natural", "r1"),
                    ("torch-natural", "r2"),
                    ("torch", "r2"),
                    ("alloff", "r2"),
                    ("fused", "r2"),
                )
            ],
        )

    def test_the_two_torch_arms_are_the_twins_two_widths(self):
        for (_seed, arm, _repeat), argv in self.legs.items():
            expected = {"torch": "bucketed", "torch-natural": "natural"}.get(arm)
            self.assertEqual(flag_value(argv, "--width"), expected, arm)

    def test_a_torch_leg_is_the_same_run_handed_to_the_other_producer(self):
        for seed in ("1", "2"):
            jammi = self.legs[(seed, "alloff", "r1")]
            for arm, repeat in itertools.product(("torch", "torch-natural"), ("r1", "r2")):
                torch_leg = self.legs[(seed, arm, repeat)]
                self.assertTrue(torch_leg[1].endswith("reference/torch_finetune_run.py"), torch_leg[:2])
                for flag in RUN_FLAGS:
                    self.assertIsNotNone(flag_value(jammi, flag), f"{flag} missing on the jammi leg")
                    self.assertEqual(flag_value(torch_leg, flag), flag_value(jammi, flag), flag)
                self.assertEqual(flag_value(torch_leg, "--lora-dropout"), "0.0")
                # jammi-only: the arm label is not a flag the twin takes.
                self.assertNotIn("--arm", torch_leg)

    def test_a_torch_leg_loads_the_adapter_its_seeds_first_jammi_leg_writes(self):
        for seed in ("1", "2"):
            # A jammi leg writes its untrained adapter into its own work dir.
            first_work_dir = flag_value(self.legs[(seed, "fused", "r1")], "--work-dir")
            written = os.path.join(first_work_dir, "initial_adapter.safetensors")
            for arm, repeat in itertools.product(("torch", "torch-natural"), ("r1", "r2")):
                torch_leg = self.legs[(seed, arm, repeat)]
                self.assertEqual(flag_value(torch_leg, "--initial-adapter"), written)
                self.assertEqual(flag_value(torch_leg, "--lora-init"), "zeros_b")

    def test_every_leg_has_a_work_dir_of_its_own(self):
        work_dirs = [flag_value(argv, "--work-dir") for argv in self.legs.values()]
        self.assertNotIn(None, work_dirs)
        self.assertEqual(len(set(work_dirs)), len(work_dirs), "two legs would share one work dir")

    def test_torch_legs_are_filed_under_the_torch_rung_and_the_ladder_starts_there(self):
        self.assertIn("--from torch --to resident", self.result.stdout)
        raw = os.listdir(os.path.join(self._tmp.name, "raw"))
        self.assertIn("torch__seed1__r1.json", raw)
        self.assertNotIn("torch__seed1__r1.json", os.listdir(os.path.join(self._tmp.name, "raw", "natural")) and [])
        self.assertIn("torch__seed1__r1.json", os.listdir(os.path.join(self._tmp.name, "raw", "natural")))


class ControlLegsDryRun(unittest.TestCase):
    """A control leg is the SAME job run with `--zero-lr-control`, on both
    producers — never `--lr 0`, which is not a job either producer admits."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.result = run_dry(
            cls._tmp.name,
            FINETUNE_RUN_AB_LR="0.0003",
            FINETUNE_RUN_AB_LR0_SEEDS="101,102",
        )
        cls.legs = leg_commands(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_each_control_seed_runs_both_arms_under_the_lr0_label(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")
        controls = [leg for leg in self.legs if leg[2] == "lr0"]
        self.assertEqual(
            controls, [(seed, arm, "lr0") for seed in ("101", "102") for arm in ("fused", "alloff")]
        )

    def test_a_control_leg_is_the_sweeps_job_with_the_control_flag(self):
        sweep = self.legs[("1", "fused", "r1")]
        self.assertNotIn("--zero-lr-control", sweep)
        for leg, argv in self.legs.items():
            self.assertEqual(flag_value(argv, "--lr"), "0.0003", leg)
            self.assertEqual("--zero-lr-control" in argv, leg[2] == "lr0", leg)


    def test_with_the_torch_arm_on_each_control_seed_has_a_torch_control_leg(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir,
                FINETUNE_RUN_AB_TORCH="1",
                FINETUNE_RUN_AB_LORA_DROPOUT="0",
                FINETUNE_RUN_AB_LR0_SEEDS="101",
            )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        legs = leg_commands(result.stdout)
        torch_control = legs[("101", "torch", "lr0")]
        self.assertIn("--zero-lr-control", torch_control)
        # A control seed has no r1 leg: the adapter is its fused control leg's.
        self.assertEqual(
            flag_value(torch_control, "--initial-adapter"),
            os.path.join(flag_value(legs[("101", "fused", "lr0")], "--work-dir"), "initial_adapter.safetensors"),
        )


class MaxSeqLengthDryRun(unittest.TestCase):
    """`max_seq_length` is an identity field: set, it reaches every leg of
    every arm with one value; unset, no leg names it and each producer's own
    default — the engine's — applies."""

    def legs_with(self, **env):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir,
                FINETUNE_RUN_AB_TORCH="1",
                FINETUNE_RUN_AB_LORA_DROPOUT="0",
                FINETUNE_RUN_AB_LR0_SEEDS="101",
                **env,
            )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        legs = leg_commands(result.stdout)
        self.assertEqual(
            {arm for _seed, arm, _repeat in legs}, {"fused", "alloff", "torch", "torch-natural"}
        )
        self.assertIn(("101", "torch", "lr0"), legs)
        return legs

    def test_set_it_is_forwarded_to_every_leg_of_every_arm(self):
        for leg, argv in self.legs_with(FINETUNE_RUN_AB_MAX_SEQ_LENGTH="512").items():
            self.assertEqual(flag_value(argv, "--max-seq-length"), "512", leg)

    def test_unset_no_leg_names_it(self):
        for leg, argv in self.legs_with().items():
            self.assertNotIn("--max-seq-length", argv, leg)


class ProtocolDefaultsDryRun(unittest.TestCase):
    """The run protocol (learning rate, epochs) is one source, the producers'
    own defaults: unset, no leg names either flag; set, every leg of every
    arm carries the one value."""

    def legs_with(self, **env):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(
                out_dir, FINETUNE_RUN_AB_TORCH="1", FINETUNE_RUN_AB_LORA_DROPOUT="0",
                FINETUNE_RUN_AB_LR0_SEEDS="101", **env,
            )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        return leg_commands(result.stdout)

    def test_unset_no_leg_names_the_learning_rate_or_the_epoch_count(self):
        for leg, argv in self.legs_with().items():
            self.assertNotIn("--lr", argv, leg)
            self.assertNotIn("--epochs", argv, leg)

    def test_set_both_reach_every_leg_of_every_arm(self):
        legs = self.legs_with(FINETUNE_RUN_AB_LR="1e-4", FINETUNE_RUN_AB_EPOCHS="6")
        self.assertEqual({arm for _s, arm, _r in legs}, {"fused", "alloff", "torch", "torch-natural"})
        for leg, argv in legs.items():
            self.assertEqual(flag_value(argv, "--lr"), "1e-4", leg)
            self.assertEqual(flag_value(argv, "--epochs"), "6", leg)


class TorchArmPremises(unittest.TestCase):
    """The pairing premise is refused before a single leg is printed."""

    def assert_refused(self, needle, **env):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, FINETUNE_RUN_AB_TORCH="1", **env)
        self.assertEqual(result.returncode, 2, f"{result.stdout}\n{result.stderr}")
        self.assertIn(needle, result.stderr)
        self.assertEqual(leg_commands(result.stdout), {})

    def test_the_default_lora_dropout_is_refused(self):
        self.assert_refused("FINETUNE_RUN_AB_LORA_DROPOUT=0")

    def test_a_nonzero_lora_dropout_is_refused(self):
        self.assert_refused("FINETUNE_RUN_AB_LORA_DROPOUT=0", FINETUNE_RUN_AB_LORA_DROPOUT="0.05")

    def test_an_objective_the_twin_does_not_train_is_refused(self):
        self.assert_refused(
            "FINETUNE_RUN_AB_OBJECTIVE=mnrl",
            FINETUNE_RUN_AB_LORA_DROPOUT="0",
            FINETUNE_RUN_AB_OBJECTIVE="triplet",
        )


if __name__ == "__main__":
    unittest.main()
