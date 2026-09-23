#!/usr/bin/env python3
# lane: torch-host
# needs: cargo, torch-venv
"""The two REAL `finetune-run` producers, paired: `jammi-bench finetune-run`
(a real `cargo` build and run) and `torch_finetune_run.py` (a real torch
subprocess), each fine-tuning the same committed tiny checkpoint over the same
committed text on a CPU in f32, the torch run starting from the adapter the
jammi run dumped.

This is where the twin's claims are held against the thing it twins, rather
than against itself:

* TOKEN-ID PARITY. Both legs digest the token batches they fed their encoders
  (`train_token_ids_sha256`, `heldout_token_ids_sha256`). The text is the
  committed held-out fixture's — real abstracts, long enough to truncate and
  ragged enough to pad — so equal digests mean jammi's tokenizer path and the
  HuggingFace one agree on ids, truncation, padding and bucketing over it.
* IDENTITY. The two legs agree on every field a jammi leg is compared on,
  judged by the merger's own premise check, not by a second comparison
  written here.
* THE PAIRING PREMISE. Both legs record the same `initial_adapter_sha256`.
* TWO WIDTHS. The twin's `--width natural` leg — every batch padded to its
  longest row instead of up jammi's shape ladder — is a different computation
  that must arrive at the same losses: padded positions are masked. It is
  held to jammi within the same tolerance, while both its token digests
  DIFFER from jammi's (the rows are cut ragged so that a batch's longest row
  is not already a ladder rung).
* THE ORIGIN. Both legs evaluate the untrained model once before step 1
  (`held_out_at_init`): same adapter, no step taken, so the two must agree
  like any other loss point.
* LEARNING. With the initial adapter shared and LoRA dropout off, nothing
  random separates the runs, so every held-out and probe loss must agree to
  f32 rounding — while the run demonstrably learns, so the agreement is not
  two flat lines. A trainer behaviour the twin failed to reproduce (warmup,
  the trailing accumulation window, weight decay, the clip) shows up here as
  a gap orders of magnitude above the tolerance.

REQUIRES a `cargo` toolchain that can build `jammi-bench`, and the torch venv
`torch_venv.py` resolves. Both are needs of this suite's guard in
`ci/needs.toml`; this suite is in the `torch-host` lane: nothing installs either,
so the CI image's lane does not select it, and where it is selected a missing
one fails naming it.

Run: `python3 ci/scripts/run_guards.py --lane torch-host`
"""

from __future__ import annotations

import json
import os
import shutil
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

REPO_ROOT = torch_venv.REPO_ROOT
REFERENCE_DIR = REPO_ROOT / "crates" / "jammi-bench" / "reference"
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
HELDOUT_PAIRS = FIXTURES / "finetune_heldout" / "heldout_pairs.jsonl"

# One committed checkpoint per architecture the twin names tensors for.
CHECKPOINTS = {
    "bert": (FIXTURES / "tiny_bert", "query,value"),
    "modernbert": (FIXTURES / "tiny_modernbert_classifier", "Wqkv,Wo,Wi"),
}

TRAIN_ROWS = 96
HELDOUT_ROWS = 32

# The run, chosen to reach every arm of the loop that can disagree: 96 train
# rows at a 0.1 validation fraction leave 86, which `--batch 8` cuts into 11
# batches with a ragged last one, and `--grad-accum 2` into 5 full windows plus
# a trailing partial one; two warmup steps, then a cosine decay whose horizon
# is the whole run (so a stack that derived it per epoch would diverge at once);
# weight decay and the clip both on; a validation pass every epoch; three
# epochs, so jammi resumes twice.
SHARED_FLAGS = [
    "--seed", "1",
    "--epochs", "3",
    "--eval-cadence", "1",
    "--batch", "8",
    "--lr", "0.01",
    "--schedule", "cosine_decay",
    "--warmup-steps", "2",
    "--weight-decay", "0.01",
    "--grad-accum", "2",
    "--validation-fraction", "0.1",
    "--early-stopping-patience", "10000",
    "--early-stopping-metric", "val_loss",
    "--max-grad-norm", "1.0",
    "--objective", "mnrl",
    "--temperature", "20.0",
    "--lora-rank", "4",
    "--lora-alpha", "8",
    "--lora-dropout", "0.0",
    "--backbone-dtype", "f32",
    "--max-seq-length", "32",
]
EXPECTED_STEPS = 18  # 6 optimizer steps an epoch, 3 epochs

# Both stacks compute in f32 and report losses near 2, where one unit in the
# last place is 2.4e-7; rounding accumulated over 18 updates stays well inside
# this.
LOSS_TOLERANCE = 1e-5
# The run must move its train probe by far more than the tolerance, or
# agreement within it would say nothing about the optimizer.
MIN_LEARNING = 100 * LOSS_TOLERANCE


def cut_ragged(line: str, index: int) -> str:
    """Row `index` with each text cut to its first 2-13 words. The committed
    abstracts all overflow `--max-seq-length`, which would make every batch
    exactly that wide; cut ragged, batches need real padding, some rows still
    truncate, and a batch's longest row is rarely a shape-ladder rung."""
    row = json.loads(line)
    for offset, field in enumerate(("anchor_text", "positive_text", "negative_text")):
        words = row[field].split()
        row[field] = " ".join(words[: 2 + (index * 5 + offset * 3) % 12])
    return json.dumps(row)


def write_inputs(root: Path) -> list[str]:
    lines = [line for line in HELDOUT_PAIRS.read_text().split("\n") if line.strip()]
    assert len(lines) >= TRAIN_ROWS + HELDOUT_ROWS, len(lines)
    lines = [cut_ragged(line, index) for index, line in enumerate(lines)]
    train, heldout = lines[:TRAIN_ROWS], lines[TRAIN_ROWS : TRAIN_ROWS + HELDOUT_ROWS]
    (root / "train.jsonl").write_text("\n".join(train) + "\n")
    (root / "heldout.jsonl").write_text("\n".join(heldout) + "\n")
    rows = [json.loads(line) for line in heldout]
    (root / "heldout_ids.txt").write_text(
        "".join(f"{r['anchor_id']}\t{r['positive_id']}\t{r['negative_id']}\n" for r in rows)
    )
    return [
        "--train-jsonl", str(root / "train.jsonl"),
        "--heldout-ids", str(root / "heldout_ids.txt"),
        "--heldout-jsonl", str(root / "heldout.jsonl"),
    ]


def run_jammi(model_dir: Path, targets: str, data_flags: list[str], work: Path) -> dict:
    work.mkdir()
    done = subprocess.run(
        [
            "cargo", "run", "--quiet", "-p", "jammi-bench", "--bin", "jammi-bench", "--",
            "finetune-run",
            "--model-dir", str(model_dir),
            "--target-modules", targets,
            "--work-dir", str(work),
            *data_flags,
            *SHARED_FLAGS,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        raise AssertionError(f"jammi-bench finetune-run exited {done.returncode}:\n{done.stderr}")
    return json.loads(done.stdout)["tiers"]["finetune_run"]


def run_torch(
    model_dir: Path, targets: str, data_flags: list[str], root: Path, adapter: Path, width: str
) -> dict:
    work = root / f"torch-work-{width}"
    work.mkdir()
    stdout = torch_venv.run(
        REFERENCE_DIR / "torch_finetune_run.py",
        "--model-dir", str(model_dir),
        "--target-modules", targets,
        "--lora-init", "zeros_b",
        "--initial-adapter", str(adapter),
        # The semantic twin of jammi's attention composition on a CPU.
        "--attn", "eager",
        "--width", width,
        "--work-dir", str(work),
        *data_flags,
        *SHARED_FLAGS,
        timeout=3600,
    )
    return json.loads(stdout)["tiers"]["finetune_run"]


class FinetuneRunCrossProducerParity(unittest.TestCase):
    """Drives both real producers once per architecture (class-level: a real
    cargo build and real torch training runs) and asserts on what they
    actually wrote."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("cargo") is None:
            raise AssertionError(
                "cargo is not on PATH: jammi-bench cannot be built (`cargo` in ci/needs.toml)"
            )
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/needs.toml)")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.legs = {}
        cls.natural = {}
        for arch, (model_dir, targets) in CHECKPOINTS.items():
            root = Path(cls._tmp.name) / arch
            root.mkdir()
            data_flags = write_inputs(root)
            jammi_work = root / "jammi-work"
            jammi = run_jammi(model_dir, targets, data_flags, jammi_work)
            # The untrained adapter every jammi run writes into its work dir.
            adapter = jammi_work / "initial_adapter.safetensors"
            torch_leg = run_torch(model_dir, targets, data_flags, root, adapter, "bucketed")
            cls.natural[arch] = run_torch(model_dir, targets, data_flags, root, adapter, "natural")
            cls.legs[arch] = (jammi, torch_leg)
            print(f"\n{arch}: held-out / probe losses, jammi vs torch", file=sys.stderr)
            natural_losses = [n for _name, _a, n in cls.loss_pairs(jammi, cls.natural[arch])]
            for (name, a, b), n in zip(cls.loss_pairs(jammi, torch_leg), natural_losses, strict=True):
                print(
                    f"  {name:<18} jammi={a:.9f} torch={b:.9f} d={a - b:+.2e}"
                    f" torch-natural={n:.9f} d={a - n:+.2e}",
                    file=sys.stderr,
                )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def loss_pairs(jammi: dict, torch_leg: dict):
        pairs = [("held-out at init", jammi["held_out_at_init"], torch_leg["held_out_at_init"])]
        pairs += [
            (f"held-out epoch {a['epoch']}", a["held_out_mean"], b["held_out_mean"])
            for a, b in zip(jammi["trajectory"], torch_leg["trajectory"], strict=True)
        ]
        pairs += [
            (f"probe[{i}]", a, b)
            for i, (a, b) in enumerate(
                zip(jammi["train_probe_series"], torch_leg["train_probe_series"], strict=True)
            )
        ]
        return pairs

    def test_the_two_legs_agree_on_every_identity_field(self):
        for arch, (jammi, torch_leg) in self.legs.items():
            names, null_is_a_value = run_identity_fields()
            violations = []
            for field in names:
                a, b = jammi.get(field, "<absent>"), torch_leg.get(field, "<absent>")
                if (a is None or b is None) and field not in null_is_a_value:
                    violations.append(f"{field}: null on a NonNull field (jammi={a!r}, torch={b!r})")
                elif json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    violations.append(f"{field}: jammi={a!r} torch={b!r}")
            self.assertEqual(violations, [], arch)

    def test_both_tokenizer_paths_fed_their_encoders_the_same_token_batches(self):
        for arch, (jammi, torch_leg) in self.legs.items():
            for field in ("train_token_ids_sha256", "heldout_token_ids_sha256"):
                self.assertRegex(jammi[field], r"^[0-9a-f]{64}$", f"{arch} {field}")
                self.assertEqual(jammi[field], torch_leg[field], f"{arch} {field}")

    def test_both_legs_started_from_the_same_adapter_bytes(self):
        for arch, (jammi, torch_leg) in self.legs.items():
            self.assertRegex(jammi["initial_adapter_sha256"], r"^[0-9a-f]{64}$", arch)
            self.assertEqual(jammi["initial_adapter_sha256"], torch_leg["initial_adapter_sha256"], arch)

    def test_both_legs_report_outcome_and_cost_under_the_same_names(self):
        measured = (
            "held_out_at_init",
            "held_out_example_mean",
            "held_out_count",
            "tie_fraction",
            "final_epoch",
            "train_probe_series",
            "train_run_wall_s",
            "steps_measured",
            "initial_adapter_sha256",
        )
        for arch, (jammi, torch_leg) in self.legs.items():
            for field in measured:
                self.assertIn(field, jammi, f"{arch} jammi {field}")
                self.assertIn(field, torch_leg, f"{arch} torch {field}")
            for series in ("trajectory", "epoch_walls"):
                self.assertEqual(len(jammi[series]), len(torch_leg[series]), f"{arch} {series}")
                self.assertEqual(set(jammi[series][0]), set(torch_leg[series][0]), f"{arch} {series}")
            for memory in ("peak_rss_bytes", "peak_vram_bytes"):
                self.assertEqual(set(jammi[memory]), {"value", "unit"}, f"{arch} jammi {memory}")
                self.assertEqual(set(torch_leg[memory]), {"value", "unit"}, f"{arch} torch {memory}")
                self.assertEqual(jammi[memory]["unit"], torch_leg[memory]["unit"])

    def test_a_natural_width_leg_is_another_computation_of_the_same_losses(self):
        for arch, (jammi, bucketed) in self.legs.items():
            natural = self.natural[arch]
            self.assertEqual(bucketed["width"], "bucketed", arch)
            self.assertEqual(natural["width"], "natural", arch)
            # Different training batches, by construction; the same held-out ones.
            for field in ("train_token_ids_sha256", "heldout_token_ids_sha256"):
                self.assertNotEqual(natural[field], jammi[field], f"{arch} {field}")
            self.assertEqual(natural["steps_measured"], EXPECTED_STEPS, arch)
            for name, a, n in self.loss_pairs(jammi, natural):
                self.assertLessEqual(
                    abs(a - n),
                    LOSS_TOLERANCE,
                    f"{arch} {name}: jammi {a!r} vs torch at natural width {n!r} — masked padding "
                    "must not move a loss beyond rounding",
                )

    def test_both_legs_took_the_same_optimizer_steps(self):
        for arch, (jammi, torch_leg) in self.legs.items():
            self.assertEqual(jammi["steps_measured"], EXPECTED_STEPS, arch)
            self.assertEqual(torch_leg["steps_measured"], EXPECTED_STEPS, arch)

    def test_both_legs_learn_and_every_loss_agrees_to_f32_rounding(self):
        for arch, (jammi, torch_leg) in self.legs.items():
            for leg_name, leg in (("jammi", jammi), ("torch", torch_leg)):
                probe = leg["train_probe_series"]
                self.assertGreater(
                    probe[0] - probe[-1],
                    MIN_LEARNING,
                    f"{arch} {leg_name}: the train probe barely moved ({probe}), so loss agreement "
                    "within the tolerance would not show the optimizers agree",
                )
            for name, a, b in self.loss_pairs(jammi, torch_leg):
                self.assertLessEqual(
                    abs(a - b),
                    LOSS_TOLERANCE,
                    f"{arch} {name}: jammi {a!r} vs torch {b!r} — a gap this size is an "
                    "unreproduced trainer behaviour, not rounding",
                )


if __name__ == "__main__":
    unittest.main()
