#!/usr/bin/env python3
"""`torch_encode.py --dry-run`, really run: the loader, the tokenizer, the
pooling, the persisted Parquet, and the child-per-leg sweep over the
repository's own tiny checkpoint — and every identity field the producer
declares, read off the legs that run writes by the ladder's leg contract.

REQUIRES the torch venv `torch_venv.py` resolves. It is the `torch-venv` need
of this suite's guard in `ci/guards.toml`, which is in the `torch-host` lane:
nothing installs it, so the CI image's lane does not select this suite, and
where it is selected a missing venv fails naming it.

Run: `python3 ci/scripts/run_guards.py --lane torch-host`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
import torch_encode  # noqa: E402

# `EncodePayload::IDENTITY_FIELDS`' one `NullMeans` entry: the dry run's
# checkpoint ships no `1_Pooling/config.json`.
NULL_MEANS = {"checkpoint_pooling_sha256"}


class TorchEncodeDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/guards.toml)")
        cls.legs_dir = tempfile.TemporaryDirectory()
        cls.report = json.loads(
            torch_venv.run(REFERENCE_DIR / "torch_encode.py", "--dry-run", "--legs-dir", cls.legs_dir.name, timeout=600)
        )
        cls.legs = [leg["encode_step"] for leg in cls.report["legs"]]

    @classmethod
    def tearDownClass(cls):
        cls.legs_dir.cleanup()

    def test_every_declared_identity_field_is_stated_on_every_leg(self):
        for leg in self.legs:
            missing = [f for f in torch_encode.IDENTITY_FIELDS if f not in leg]
            null = [f for f in torch_encode.IDENTITY_FIELDS if leg.get(f) is None and f not in NULL_MEANS]
            self.assertFalse(missing or null, f"absent: {missing}; null but declared non-null: {null}")

    def test_the_legs_follow_the_ladders_contract(self):
        expected = [(rows, take) for rows in torch_encode.DRY_RUN_ROWS for take in range(1, torch_encode.DRY_RUN_TAKES + 1)]
        self.assertEqual([(leg["rows"], leg["take"]) for leg in self.legs], expected)
        for leg in self.legs:
            stem = f"torch__rows{leg['rows']}__r{leg['take']}"
            with open(os.path.join(self.legs_dir.name, stem + ".json")) as fh:
                self.assertEqual(json.load(fh)["encode_step"]["outcome_digest"], leg["outcome_digest"])
            self.assertEqual((leg["rung"], leg["work"]), ("torch", leg["rows"]))
            self.assertEqual(len(leg["iter_wall_s"]), leg["iters_measured"])
            if leg["take"] == 1:
                self.assertEqual(leg["vectors_file"], stem + ".vectors.f32")
                size = os.path.getsize(os.path.join(self.legs_dir.name, leg["vectors_file"]))
                self.assertEqual(size, leg["rows"] * leg["vector_dim"] * 4)
            else:
                self.assertIsNone(leg["vectors_file"])
        first, second = self.legs[0], self.legs[1]
        self.assertEqual(first["outcome_digest"], second["outcome_digest"], "two takes of one unit embed alike")

    def test_a_leg_is_a_real_variable_length_serve(self):
        for leg in self.legs:
            self.assertGreater(leg["padded_tokens"], leg["tokens"], "variable-length rows must pad")
            self.assertLessEqual(leg["row_tokens_max"], leg["max_sequence_length"])
            self.assertGreater(leg["serve_ms_min"], 0.0)
            self.assertLessEqual(leg["serve_ms_min"], leg["serve_ms_p50"])
            self.assertIsNone(leg["peak_vram_bytes"]["value"], "no sampler wrapped this dry run")
            self.assertIsNone(leg["peak_vram_allocator_bytes"], "a CPU leg touches no device allocator")

    def test_the_resolved_attention_and_pooling_are_read_back(self):
        leg = self.legs[0]
        self.assertEqual(leg["attn_implementation"], leg["attn_requested"])
        self.assertEqual((leg["pooling"], leg["checkpoint_pooling_sha256"]), ("mean", None))


if __name__ == "__main__":
    unittest.main()
