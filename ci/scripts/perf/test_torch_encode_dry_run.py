#!/usr/bin/env python3
"""`torch_encode.py --dry-run`, really run: the loader, the tokenizer, the
pooling, the persisted Parquet and the child-per-point sweep over the
repository's own tiny checkpoint — and every identity field the producer
declares, read off the report that run writes.

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
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
import torch_encode  # noqa: E402

# `EncodeStepTier::IDENTITY_FIELDS`' one `NullMeans` entry: the dry run's
# checkpoint ships no `1_Pooling/config.json`.
NULL_MEANS = {"checkpoint_pooling_sha256"}


class TorchEncodeDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/guards.toml)")
        cls.report = json.loads(torch_venv.run(REFERENCE_DIR / "torch_encode.py", "--dry-run", timeout=600))
        cls.tier = cls.report["encode_step"]

    def test_every_declared_identity_field_is_stated(self):
        missing = [f for f in torch_encode.IDENTITY_FIELDS if f not in self.tier]
        null = [f for f in torch_encode.IDENTITY_FIELDS if self.tier.get(f) is None and f not in NULL_MEANS]
        self.assertFalse(missing or null, f"absent: {missing}; null but declared non-null: {null}")

    def test_the_sweep_is_one_point_per_row_count_with_both_fits(self):
        self.assertEqual(self.tier["rows"], list(torch_encode.DRY_RUN_ROWS))
        self.assertEqual([p["rows"] for p in self.tier["points"]], self.tier["rows"])
        self.assertEqual([c["rows"] for c in self.tier["corpus"]], self.tier["rows"])
        for fit in (self.tier["fit_p50"], self.tier["fit_min"]):
            self.assertGreater(fit["per_row_ms"], 0.0)

    def test_a_point_is_a_real_variable_length_serve(self):
        for point, corpus in zip(self.tier["points"], self.tier["corpus"]):
            self.assertGreater(point["padded_tokens"], corpus["tokens"], "variable-length rows must pad")
            self.assertLessEqual(point["row_tokens_max"], self.tier["max_sequence_length"])
            self.assertGreater(point["serve_ms_min"], 0.0)
            self.assertLessEqual(point["serve_ms_min"], point["serve_ms_p50"])
            self.assertIsNone(point["agreement"], "a dry run has no jammi vectors to agree with")
            self.assertIsNone(point["peak_vram_delta_bytes"], "a CPU leg touches no device allocator")

    def test_the_resolved_attention_and_pooling_are_read_back(self):
        self.assertEqual(self.tier["attn_implementation"], self.tier["attn_requested"])
        self.assertEqual((self.tier["pooling"], self.tier["checkpoint_pooling_sha256"]), ("mean", None))


if __name__ == "__main__":
    unittest.main()
