#!/usr/bin/env python3
"""A cross-producer PARITY test that
drives the two REAL emitters — `jammi-bench grad-oracle` (a real `cargo`
build/run, tiny CPU fixture) and `torch_grad_oracle.py --dry-run` (a real
torch subprocess, tiny random-init 2-layer ModernBERT donor checkpoint) —
and diffs their emitted `backbone_dtype` SPELLING and top-level key set,
never two hand-built fixture dicts standing in for what the producers
actually emit: every field compared across the two producers has a parity
test that drives the REAL emitters on both sides.

WHY REAL EMITTERS: jammi's `grad_oracle.rs` emits `backbone_dtype: "f32"`
(`format!("{:?}", ComputePrecision::F32).to_lowercase()`), while torch's bare
`--dtype` CLI spelling is `"fp32"`. `compare_grad_oracle.py`'s
`RUN_IDENTITY_FIELDS` premise check compares this field after canonicalizing
it — if the torch producer ever emitted a spelling the canonicalizer does not
cover, EVERY jammi-f32-vs-torch-f32 comparison (including this oracle's own
near-perfect control, overall cosine 0.9999998 on a real A100 run) would be
refused on a spurious spelling mismatch. `test_compare_grad_oracle.py`'s
fixture-based suite cannot see that: its `make_report` helper puts `"f32"`
on BOTH sides by construction, which is why this test lives in a SEPARATE
file that drives the real producers.

REQUIRES a `cargo` toolchain that can build `jammi-bench`, and the torch
venv `torch_venv.py` resolves (`TORCH_VENV`, default `<repo>/.venv-torch-ref`,
the one `ci/scripts/perf/finetune_ab.sh`'s `setup_torch_venv` provisions).
Both are needs of this suite's guard in `ci/guards.toml`, which is in the
`torch-host` lane: nothing installs either, so the CI image's lane does not
select it, and where it is selected a missing one fails naming it.

Run (after `uv venv "$TORCH_VENV" && uv pip install --python
"$TORCH_VENV/bin/python3" torch transformers peft safetensors`):
    python3 ci/scripts/run_guards.py --lane torch-host
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
import compare_grad_oracle as cgo  # noqa: E402
import torch_venv  # noqa: E402

REPO_ROOT = torch_venv.REPO_ROOT
REFERENCE_DIR = REPO_ROOT / "crates" / "jammi-bench" / "reference"
TINY_FIXTURE_DIR = REPO_ROOT / "cookbook" / "fixtures" / "tiny_modernbert_classifier"


def _run_torch_dry_run(out_path: Path) -> dict:
    # `--lora-rank`/`--lora-alpha`/`--target-modules`/`--seed` match
    # `_run_jammi_grad_oracle` below EXACTLY -- `--dry-run` only overrides
    # `--batch`/`--seq`/`--model-dir` (see `torch_grad_oracle.py`'s `run()`),
    # every other flag is left as the caller passed it. A rank mismatch here
    # (torch's own `--lora-rank` default is 16, jammi's tiny-fixture default
    # below is 2) makes every LoRA tensor's SHAPE differ between the two
    # dumps -- caught, the hard way, by `compare_tensor`'s own length-
    # mismatch guard the first time this test ran without this alignment.
    torch_venv.run(
        REFERENCE_DIR / "torch_grad_oracle.py",
        "--dry-run",
        "--lora-rank", "2",
        "--lora-alpha", "4.0",
        "--target-modules", "Wqkv,Wo,Wi",
        "--seed", "42",
        "--out", str(out_path),
        timeout=300,
    )
    with open(out_path) as fh:
        return json.load(fh)


def _run_jammi_grad_oracle(out_path: Path) -> dict:
    built = subprocess.run(
        [
            "cargo", "run", "--quiet", "-p", "jammi-bench", "--bin", "jammi-bench", "--",
            "grad-oracle",
            "--model-dir", str(TINY_FIXTURE_DIR),
            "--batch", "3",
            "--seq", "8",
            "--lora-rank", "2",
            "--lora-alpha", "4.0",
            "--target-modules", "Wqkv,Wo,Wi",
            "--backbone-dtype", "f32",
            "--seed", "42",
            "--out", str(out_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if built.returncode != 0:
        raise AssertionError(f"jammi-bench grad-oracle exited {built.returncode}:\n{built.stderr}")
    with open(out_path) as fh:
        return json.load(fh)


class CrossProducerDtypeSpellingParity(unittest.TestCase):
    """Drives BOTH real emitters exactly once (class-level, expensive: a
    real cargo build/run and a real torch subprocess) and asserts on their
    ACTUAL output, never a hand-built stand-in.
    """

    @classmethod
    def setUpClass(cls):
        if shutil.which("cargo") is None:
            raise AssertionError(
                "cargo is not on PATH: jammi-bench cannot be built (`cargo` in ci/guards.toml)"
            )
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/guards.toml)")
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.torch_report = _run_torch_dry_run(tmp / "torch_grad.json")
        cls.jammi_report = _run_jammi_grad_oracle(tmp / "jammi_grad.json")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_real_emitters_write_the_canonical_f32_spelling(self):
        """Driven at the real entry points: BOTH dumps'
        `backbone_dtype` must be the literal string `"f32"` -- not merely
        "equal to each other under some normalization", but the ACTUAL
        canonical jammi spelling on both sides, unnormalized.
        """
        self.assertEqual(self.jammi_report["backbone_dtype"], "f32")
        self.assertEqual(self.torch_report["backbone_dtype"], "f32")

    def test_backbone_dtype_field_never_appears_in_premise_violations(self):
        """Feeds the two REAL dumps into the REAL `compare_reports` (never
        a hand-built fixture pair) and asserts that whatever premise
        violations it finds (the two dumps are NOT a matching pair --
        different checkpoints, no shared --lora-weights-in, different
        target token content -- so OTHER violations are EXPECTED and
        correct here), none of them name `backbone_dtype`. This is the
        actual mechanism check: not "does a hand-built pair with a spelling
        mismatch get refused", but "do the two REAL emitters' REAL output,
        run through the REAL comparator, ever disagree on this field's
        spelling".
        """
        result = cgo.compare_reports(self.jammi_report, self.torch_report, cosine_floor=0.5)
        backbone_dtype_violations = [v for v in result["premise_violations"] if "backbone_dtype" in v]
        self.assertEqual(
            backbone_dtype_violations,
            [],
            f"backbone_dtype spelling mismatch survived into a REAL comparator run: "
            f"jammi={self.jammi_report['backbone_dtype']!r} torch={self.torch_report['backbone_dtype']!r} "
            f"-- all premise_violations: {result['premise_violations']!r}",
        )

    def test_run_identity_key_set_present_on_both_real_dumps(self):
        """Every field `compare_grad_oracle.RUN_IDENTITY_FIELDS` reads must
        actually be PRESENT on both real dumps -- a missing key would make
        `_premise_violations`'s `report.get(field)` silently compare `None
        == None` and never flag a real absence as a violation at all.
        """
        for field in cgo.RUN_IDENTITY_FIELDS:
            self.assertIn(field, self.jammi_report, f"jammi grad-oracle dump missing {field!r}")
            self.assertIn(field, self.torch_report, f"torch_grad_oracle.py dump missing {field!r}")

    def test_layer_zero_tensor_naming_scheme_matches_between_real_emitters(self):
        """The two real dumps come from DIFFERENT models (jammi's tiny
        1-layer CPU fixture vs. torch's 2-layer --dry-run donor checkpoint),
        so their FULL key sets are not expected to match -- but both used
        `target_modules = Wqkv,Wo,Wi`, so the NAMING SCHEME for layer 0
        (which site names existed, `Wqkv`/`Wo`/`Wi`/`mlp.Wo`, each with a
        `lora_a`/`lora_b` pair) must match EXACTLY -- this is
        `torch_grad_oracle.py`'s own `translate_peft_name_to_jammi` name-
        translation table proving itself correct against jammi's REAL
        naming, not just against `test_torch_grad_oracle_names.py`'s
        stdlib-only unit tests of the translation function in isolation.
        """
        jammi_layer0 = {k for k in self.jammi_report["gradients"] if k.startswith("layer.0.")}
        torch_layer0 = {k for k in self.torch_report["gradients"] if k.startswith("layer.0.")}
        self.assertEqual(jammi_layer0, torch_layer0)
        self.assertEqual(
            jammi_layer0,
            {
                "layer.0.Wqkv.lora_a", "layer.0.Wqkv.lora_b",
                "layer.0.Wo.lora_a", "layer.0.Wo.lora_b",
                "layer.0.Wi.lora_a", "layer.0.Wi.lora_b",
                "layer.0.mlp.Wo.lora_a", "layer.0.mlp.Wo.lora_b",
            },
        )

    def test_lora_dropout_is_forced_zero_on_both_real_dumps(self):
        """Both module docs claim `lora_dropout` is unconditionally forced
        to `0.0` -- checked here against the REAL dumps, not just asserted
        in prose.
        """
        self.assertEqual(self.jammi_report["lora_dropout"], 0.0)
        self.assertEqual(self.torch_report["lora_dropout"], 0.0)


if __name__ == "__main__":
    unittest.main()
