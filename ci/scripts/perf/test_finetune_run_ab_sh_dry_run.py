#!/usr/bin/env python3
"""Hermetic `FINETUNE_RUN_AB_DRY_RUN=1` smoke test for `finetune_run_ab.sh`
itself — the shell PRODUCER executed, not only its merge stage
(`test_finetune_ab_sh_dry_run.py`'s doctrine, applied to the run-level
driver).

What only running the script shows: the order each seed's legs run in, the
command line each leg is handed, and that a torch leg is the SAME run
description given to the other producer, started from the adapter a jammi
leg of its seed dumped, and refused outright when the pairing premise cannot
hold. None of that is visible from the ladder's own tests, which read leg
files and never build an argv.

`FINETUNE_RUN_AB_DRY_RUN=1` makes the pipeline hermetic: no `cargo build`, no
venv probe, no GPU, no network — every leg's command is PRINTED, never
executed, and every leg writes a `{"tool":"dry-run",...}` stub so the ladder
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
    """The script under `FINETUNE_RUN_AB_DRY_RUN=1` with the torch arm's
    pairing premise met; an entry of `None` in `extra_env` unsets the
    variable, so a test can ask for a producer's own default."""
    env = dict(os.environ)
    env.update(
        FINETUNE_RUN_AB_DRY_RUN="1",
        FINETUNE_RUN_AB_OUT_DIR=out_dir,
        FINETUNE_RUN_AB_SEEDS="1,2",
        FINETUNE_RUN_AB_LORA_DROPOUT="0",
        # A dry run has no lr=0 control legs to judge; the opt-out is the
        # ladder's own recorded one, not a way around it.
        FINETUNE_RUN_AB_ALLOW_NO_LR0="1",
    )
    # Whatever venv the host names is irrelevant to a dry run, which must not
    # probe one.
    env.pop("TORCH_VENV", None)
    for name, value in extra_env.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
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


class DefaultDryRun(unittest.TestCase):
    """Every arm of the run, in the block's order, handed to the ladder."""

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
        self.assertIn("--from torch --to resident", self.result.stdout)
        raw = os.listdir(os.path.join(self._tmp.name, "raw"))
        self.assertIn("resident__seed1__r1.json", raw)
        self.assertIn("torch__seed1__r2.json", raw)
        self.assertIn("torch__seed1__r1.json", os.listdir(os.path.join(self._tmp.name, "raw", "natural")))

    def test_each_seeds_legs_run_order_balanced(self):
        self.assertEqual(
            list(self.legs),
            [
                (seed, arm, repeat)
                for seed in ("1", "2")
                for arm, repeat in (
                    ("fused", "r1"),
                    ("torch", "r1"),
                    ("torch-natural", "r1"),
                    ("torch-natural", "r2"),
                    ("torch", "r2"),
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
            jammi = self.legs[(seed, "fused", "r1")]
            for arm, repeat in itertools.product(("torch", "torch-natural"), ("r1", "r2")):
                torch_leg = self.legs[(seed, arm, repeat)]
                self.assertTrue(torch_leg[1].endswith("reference/torch_finetune_run.py"), torch_leg[:2])
                for flag in RUN_FLAGS:
                    self.assertIsNotNone(flag_value(jammi, flag), f"{flag} missing on the jammi leg")
                    self.assertEqual(flag_value(torch_leg, flag), flag_value(jammi, flag), flag)
                self.assertEqual(flag_value(torch_leg, "--lora-dropout"), "0")

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

    def test_no_leg_names_a_kernel_arm(self):
        for key, argv in self.legs.items():
            self.assertNotIn("--arm", argv, key)
            self.assertIsNone(flag_value(argv, "--target-modules"), key)


class TargetModulesDryRun(unittest.TestCase):
    """`FINETUNE_RUN_AB_TARGET_MODULES` reaches every leg of every arm."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.result = run_dry(cls._tmp.name, FINETUNE_RUN_AB_TARGET_MODULES="Wqkv,Wo")
        cls.legs = leg_commands(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_leg_names_the_same_sites(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")
        self.assertTrue(self.legs)
        for key, argv in self.legs.items():
            self.assertEqual(flag_value(argv, "--target-modules"), "Wqkv,Wo", key)


class ArmFilterDryRun(unittest.TestCase):
    """`FINETUNE_RUN_AB_ARMS` runs one stack's legs alone, in the block's order."""

    def test_the_torch_arms_alone_run_in_their_block_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dry(tmp, FINETUNE_RUN_AB_ARMS="torch,torch-natural")
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
            legs = leg_commands(result.stdout)
        self.assertEqual(
            list(legs),
            [
                (seed, arm, repeat)
                for seed in ("1", "2")
                for arm, repeat in (
                    ("torch", "r1"),
                    ("torch-natural", "r1"),
                    ("torch-natural", "r2"),
                    ("torch", "r2"),
                )
            ],
        )
        # The legs still load the adapter the seed's fused leg wrote into this
        # OUT_DIR: the pairing is by file, not by having run in this process.
        for (seed, _arm, _repeat), argv in legs.items():
            self.assertTrue(
                flag_value(argv, "--initial-adapter").endswith(
                    f"/work/seed{seed}__fused__r1/initial_adapter.safetensors"
                )
            )

    def test_the_control_loop_honours_the_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dry(
                tmp,
                FINETUNE_RUN_AB_LR0_SEEDS="1",
                FINETUNE_RUN_AB_ALLOW_NO_LR0="0",
                FINETUNE_RUN_AB_ARMS="torch",
            )
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
            legs = leg_commands(result.stdout)
        self.assertEqual(
            [leg for leg in legs if leg[2] == "lr0"], [("1", "torch", "lr0")]
        )

    def test_an_arm_the_run_does_not_have_is_refused_before_any_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dry(tmp, FINETUNE_RUN_AB_ARMS="fused,eager")
            self.assertEqual(result.returncode, 2)
            self.assertIn("'eager'", result.stderr)
            self.assertEqual(leg_commands(result.stdout), {})

    def test_the_default_is_every_arm_of_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dry(tmp)
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
            arms = {arm for _seed, arm, _repeat in leg_commands(result.stdout)}
        self.assertEqual(arms, {"fused", "torch", "torch-natural"})


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

    def test_each_control_seed_runs_the_fused_and_torch_arms_under_the_lr0_take(self):
        self.assertEqual(self.result.returncode, 0, f"{self.result.stdout}\n{self.result.stderr}")
        controls = [leg for leg in self.legs if leg[2] == "lr0"]
        self.assertEqual(
            controls, [(seed, arm, "lr0") for seed in ("101", "102") for arm in ("fused", "torch")]
        )

    def test_a_control_leg_is_the_sweeps_job_with_the_control_flag(self):
        sweep = self.legs[("1", "fused", "r1")]
        self.assertNotIn("--zero-lr-control", sweep)
        for leg, argv in self.legs.items():
            self.assertEqual(flag_value(argv, "--lr"), "0.0003", leg)
            self.assertEqual("--zero-lr-control" in argv, leg[2] == "lr0", leg)


    def test_the_torch_control_leg_is_the_control_run_from_the_fused_controls_adapter(self):
        torch_control = self.legs[("101", "torch", "lr0")]
        self.assertIn("--zero-lr-control", torch_control)
        # A control seed has no r1 leg: the adapter is its fused control leg's.
        self.assertEqual(
            flag_value(torch_control, "--initial-adapter"),
            os.path.join(flag_value(self.legs[("101", "fused", "lr0")], "--work-dir"), "initial_adapter.safetensors"),
        )


class MaxSeqLengthDryRun(unittest.TestCase):
    """`max_seq_length` is an identity field: set, it reaches every leg of
    every arm with one value; unset, no leg names it and each producer's own
    default — the engine's — applies."""

    def legs_with(self, **env):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, FINETUNE_RUN_AB_LR0_SEEDS="101", **env)
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        legs = leg_commands(result.stdout)
        self.assertEqual({arm for _seed, arm, _repeat in legs}, {"fused", "torch", "torch-natural"})
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
            result = run_dry(out_dir, FINETUNE_RUN_AB_LR0_SEEDS="101", **env)
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        return leg_commands(result.stdout)

    def test_unset_no_leg_names_the_learning_rate_or_the_epoch_count(self):
        for leg, argv in self.legs_with().items():
            self.assertNotIn("--lr", argv, leg)
            self.assertNotIn("--epochs", argv, leg)

    def test_set_both_reach_every_leg_of_every_arm(self):
        legs = self.legs_with(FINETUNE_RUN_AB_LR="1e-4", FINETUNE_RUN_AB_EPOCHS="6")
        self.assertEqual({arm for _s, arm, _r in legs}, {"fused", "torch", "torch-natural"})
        for leg, argv in legs.items():
            self.assertEqual(flag_value(argv, "--lr"), "1e-4", leg)
            self.assertEqual(flag_value(argv, "--epochs"), "6", leg)


class TorchArmPremises(unittest.TestCase):
    """The pairing premise is refused before a single leg is printed."""

    def assert_refused(self, needle, **env):
        with tempfile.TemporaryDirectory() as out_dir:
            result = run_dry(out_dir, **env)
        self.assertEqual(result.returncode, 2, f"{result.stdout}\n{result.stderr}")
        self.assertIn(needle, result.stderr)
        self.assertEqual(leg_commands(result.stdout), {})

    def test_the_default_lora_dropout_is_refused(self):
        self.assert_refused("FINETUNE_RUN_AB_LORA_DROPOUT=0", FINETUNE_RUN_AB_LORA_DROPOUT=None)

    def test_a_nonzero_lora_dropout_is_refused(self):
        self.assert_refused("FINETUNE_RUN_AB_LORA_DROPOUT=0", FINETUNE_RUN_AB_LORA_DROPOUT="0.05")

    def test_an_objective_the_twin_does_not_train_is_refused(self):
        self.assert_refused("FINETUNE_RUN_AB_OBJECTIVE=mnrl", FINETUNE_RUN_AB_OBJECTIVE="triplet")


if __name__ == "__main__":
    unittest.main()
