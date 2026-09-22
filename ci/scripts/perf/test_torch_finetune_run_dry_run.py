#!/usr/bin/env python3
# lane: torch-host
# needs: torch-venv
"""The leg a real `torch_finetune_run.py --dry-run` writes: a whole two-epoch
fine-tune of a tiny random ModernBERT on a CPU, through the same code path a
GPU leg takes.

What only a real run can hold: that every identity field a jammi
`finetune-run` leg is compared on is a key of the torch leg with the right
nullness, that the outcome and cost fields carry the shapes
`FinetuneRunTier` gives them, and that the host-side per-example loss agrees
with torch's own cross-entropy.

REQUIRES the torch venv `torch_venv.py` resolves. This suite is in the `torch-host` lane (its `# lane:` line):
nothing installs the venv, so the CI image's run does not select this suite, and
where it is selected a missing venv fails naming it.

Run: `python3 ci/scripts/run_guards.py --lane torch-host`
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from check_cuda_run_artifacts import build_identity_tuples  # noqa: E402


def run_identity_fields():
    """The `train-run` identity as `impl Payload for TrainRunPayload` declares
    it: the field names in order, and the names whose `null` is a value."""
    fields = [f for f in build_identity_tuples()[("finetune_run", "jammi")]["fields"] if f[1] == "tier"]
    return tuple(f[0] for f in fields), frozenset(f[0] for f in fields if f[2] == "NullMeans")

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"

# `torch.nn.functional.cross_entropy(reduction="none")` against the twin's
# host-side f32 fold, over logits wide enough to exercise the max-subtraction.
PER_ROW_PROBE = """
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
import torch_finetune_run as tfr
torch.manual_seed(0)
logits = torch.randn(7, 7) * 20.0
ours = tfr.cross_entropy_per_row(logits.numpy())
theirs = F.cross_entropy(logits, torch.arange(7), reduction="none").tolist()
print(json.dumps({"ours": ours, "theirs": theirs}))
"""


class TorchFinetuneRunDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/needs.toml)")
        # The producer writes its report to standard output.
        cls.report = json.loads(
            torch_venv.run(REFERENCE_DIR / "torch_finetune_run.py", "--dry-run", timeout=600)
        )
        cls.tier = cls.report["tiers"]["finetune_run"]

    def test_every_identity_field_is_present_with_its_declared_nullness(self):
        names, null_is_a_value = run_identity_fields()
        for field in names:
            self.assertIn(field, self.tier, f"identity field {field!r} is not a key of the torch leg")
            if field not in null_is_a_value:
                self.assertIsNotNone(self.tier[field], f"identity field {field!r} is null")

    def test_the_outcome_has_one_point_per_epoch_on_a_time_axis(self):
        epochs = self.tier["epochs"]
        self.assertIsInstance(self.tier["held_out_at_init"], float)
        self.assertEqual([p["epoch"] for p in self.tier["trajectory"]], list(range(epochs)))
        self.assertEqual(len(self.tier["train_probe_series"]), epochs + 1)
        walls = self.tier["epoch_walls"]
        self.assertEqual(len(walls), epochs)
        for wall in walls:
            self.assertEqual(
                set(wall), {"run_s", "steps_s", "validation_s", "checkpoint_s", "step_walls"}
            )
            # The dry run monitors val_loss, so every phase is paid, and the
            # phases are disjoint spans inside the epoch's wall.
            phases = ("run_s", "steps_s", "validation_s", "checkpoint_s")
            self.assertTrue(all(wall[phase] > 0.0 for phase in phases), wall)
            self.assertLessEqual(wall["steps_s"] + wall["validation_s"] + wall["checkpoint_s"], wall["run_s"])
            # Every optimizer step's wall, inside the epoch's step span.
            self.assertTrue(wall["step_walls"] and all(w > 0.0 for w in wall["step_walls"]), wall)
            self.assertLessEqual(sum(wall["step_walls"]), wall["steps_s"] * 1.01 + 1e-3)
        # The padded transport, stated as the fact the ladder's premise reads.
        self.assertIs(self.tier["admission_is_dense"], False)
        # A training run's timed iteration is its optimizer step.
        self.assertEqual(len(self.tier["iter_wall_s"]), self.tier["steps_measured"])
        self.assertEqual(
            self.tier["iter_wall_s"], [w for wall in walls for w in wall["step_walls"]]
        )
        last = self.tier["trajectory"][-1]
        # The epochs lie inside the run, which also writes the final adapter
        # (`finish`), as jammi's own leg does.
        self.assertLessEqual(sum(w["run_s"] for w in walls), self.tier["train_run_wall_s"])
        self.assertEqual(last["run_wall_s_cumulative"], sum(w["run_s"] for w in walls))
        self.assertEqual(last["steps_wall_s_cumulative"], sum(w["steps_s"] for w in walls))
        self.assertEqual(last["held_out_mean"], self.tier["held_out_example_mean"])

    def test_every_optimizer_step_is_counted_once(self):
        # 11 rows, 2 held out for validation: 9 train rows at --batch 2 is 5
        # batches, which --grad-accum 2 makes 3 steps an epoch (the last a
        # trailing partial window), over 2 epochs.
        self.assertEqual(self.tier["steps_measured"], 6)

    def test_memory_is_measured_where_the_host_can_say(self):
        self.assertGreater(self.tier["peak_rss_bytes"]["value"], 0.0)
        self.assertIn(self.report["provenance"]["peak_rss_source"], ("proc_vm_hwm", "getrusage_maxrss"))
        # The dry run is a CPU run: no device, no sample, never a zero.
        self.assertIsNone(self.tier["peak_vram_bytes"]["value"])

    def test_a_peft_initialized_leg_says_so(self):
        self.assertEqual(self.tier["lora_init"], "peft")
        self.assertIsNone(self.tier["initial_adapter_sha256"])

    def test_the_leg_names_its_width(self):
        self.assertEqual(self.tier["width"], "bucketed")

    def test_the_stack_that_ran_is_recorded(self):
        provenance = self.report["provenance"]
        for field in ("torch_version", "transformers_version", "peft_version", "tokenizer_sha256"):
            self.assertTrue(provenance[field], f"provenance.{field} is empty")

    def test_the_zero_lr_control_runs_every_step_and_learns_nothing(self):
        control = json.loads(
            torch_venv.run(
                REFERENCE_DIR / "torch_finetune_run.py", "--dry-run", "--zero-lr-control", timeout=600
            )
        )["tiers"]["finetune_run"]
        self.assertEqual(control["lr"], 0.0)
        self.assertEqual(control["steps_measured"], self.tier["steps_measured"])
        series = control["train_probe_series"]
        self.assertEqual(series[0] - series[-1], 0.0, series)
        self.assertEqual(len(set(series)), 1, series)
        self.assertTrue(
            all(p["held_out_mean"] == control["held_out_at_init"] for p in control["trajectory"]), control
        )
        # The control's own control: the ordinary dry run's probe moved.
        ordinary = self.tier["train_probe_series"]
        self.assertNotEqual(ordinary[0], ordinary[-1], ordinary)

    def test_a_non_positive_learning_rate_is_refused_naming_the_control(self):
        done = subprocess.run(
            [str(torch_venv.TORCH_PY), str(REFERENCE_DIR / "torch_finetune_run.py"), "--dry-run", "--lr", "0"],
            cwd=torch_venv.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("--zero-lr-control", done.stderr)

    def test_the_host_side_per_example_loss_is_torchs_cross_entropy(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "per_row_probe.py"
            probe.write_text(PER_ROW_PROBE)
            result = json.loads(torch_venv.run(probe, str(REFERENCE_DIR), timeout=300))
        for ours, theirs in zip(result["ours"], result["theirs"], strict=True):
            self.assertAlmostEqual(ours, theirs, delta=1e-4 * max(1.0, abs(theirs)))


if __name__ == "__main__":
    unittest.main()
