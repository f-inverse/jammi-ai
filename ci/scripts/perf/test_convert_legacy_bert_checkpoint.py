#!/usr/bin/env python3
"""`convert_legacy_bert_checkpoint.py`'s own suite (unit #356 census
execution fix, defect 2): drives the real `renamed_name`/`convert`
functions.

`renamed_name` (the suffix-anchored `.gamma`->`.weight` / `.beta`->`.bias`
string logic) is a PURE function of a name string -- covered without the
`safetensors` package, the same "pure function first" split
`fixture_width_report.py`'s own suite already uses for its
package-independent half.

`convert`'s file-level behavior (renames happen, values are byte-identical,
a rename collision refuses, a zero-rename result refuses, `out_path ==
in_path` refuses) needs a REAL synthetic safetensors fixture built in-test
via the real `safetensors` package (`TensorSpec`/`serialize`) -- exercised
only when `safetensors` is actually importable, skipped (never silently
treated as passing) otherwise, exactly the `tokenizers`-dependent-half
precedent `fixture_width_report.py`'s own suite already established.

The "package not importable" loud-refusal path is exercised via a real
subprocess, skipped if the real package IS installed in this environment
(there is then nothing to hide it behind without also breaking other
tests' imports) -- the same `NotImportableRefusalTests` shape
`test_fixture_width_report.py` uses for `tokenizers`.

Run: `python3 ci/scripts/perf/test_convert_legacy_bert_checkpoint.py`
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_legacy_bert_checkpoint as clbc  # noqa: E402

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "convert_legacy_bert_checkpoint.py")

_SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None


class RenamedNameSuffixAnchoredTests(unittest.TestCase):
    """`renamed_name` needs no `safetensors` import at all -- pure string
    logic, covered unconditionally."""

    def test_gamma_suffix_renames_to_weight(self):
        self.assertEqual(
            clbc.renamed_name("encoder.layer.0.attention.output.LayerNorm.gamma"),
            "encoder.layer.0.attention.output.LayerNorm.weight",
        )

    def test_beta_suffix_renames_to_bias(self):
        self.assertEqual(
            clbc.renamed_name("encoder.layer.0.attention.output.LayerNorm.beta"),
            "encoder.layer.0.attention.output.LayerNorm.bias",
        )

    def test_already_correct_name_is_untouched(self):
        self.assertEqual(
            clbc.renamed_name("embeddings.word_embeddings.weight"),
            "embeddings.word_embeddings.weight",
        )

    def test_mid_string_gamma_not_a_trailing_component_is_untouched(self):
        # "gamma" appears in the name but NOT as its own trailing dotted
        # component -- suffix-anchored means this must be left alone.
        self.assertEqual(clbc.renamed_name("gamma_scale.other"), "gamma_scale.other")

    def test_mid_string_beta_not_a_trailing_component_is_untouched(self):
        self.assertEqual(clbc.renamed_name("beta_version.other"), "beta_version.other")

    def test_bare_gamma_with_no_leading_dot_is_untouched(self):
        # ".gamma" is the suffix pattern -- a name ending in "gamma" but
        # without the preceding dot (not a real dotted component boundary)
        # must not match.
        self.assertEqual(clbc.renamed_name("somegamma"), "somegamma")


@unittest.skipUnless(_SAFETENSORS_AVAILABLE, "safetensors package not importable in this env")
class ConvertFileLevelTests(unittest.TestCase):
    """Drives the real `convert()` against a TINY synthetic safetensors
    fixture built in-test via the real `safetensors` package -- never a
    real BERT checkpoint."""

    @staticmethod
    def _make_fixture(path: str, tensors: dict) -> None:
        """`tensors`: name -> (dtype ctor name, shape, raw bytes)."""
        import ctypes

        from safetensors import TensorSpec, serialize

        buffers = []
        tensor_dict = {}
        for name, (dtype, shape, data) in tensors.items():
            buf = bytearray(data)
            arr = (ctypes.c_char * len(buf)).from_buffer(buf)
            buffers.append((buf, arr))
            tensor_dict[name] = TensorSpec(
                dtype=dtype, shape=shape, data_ptr=ctypes.addressof(arr), data_len=len(buf)
            )
        blob = serialize(tensor_dict)
        Path(path).write_bytes(blob)

    @staticmethod
    def _read_back(path: str) -> dict:
        from safetensors import deserialize

        raw = Path(path).read_bytes()
        return {name: info for name, info in deserialize(raw)}

    def test_renames_happen_and_values_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            ln_weight_bytes = bytes(range(16))  # 4 x float32
            ln_bias_bytes = bytes(range(16, 32))
            embed_bytes = bytes(range(32, 40))  # 2 x float32
            self._make_fixture(
                in_path,
                {
                    "encoder.layer.0.LayerNorm.gamma": ("float32", [4], ln_weight_bytes),
                    "encoder.layer.0.LayerNorm.beta": ("float32", [4], ln_bias_bytes),
                    "embeddings.word_embeddings.weight": ("float32", [2], embed_bytes),
                },
            )
            n = clbc.convert(in_path, out_path)
            self.assertEqual(n, 2)

            out_items = self._read_back(out_path)
            self.assertEqual(
                set(out_items),
                {
                    "encoder.layer.0.LayerNorm.weight",
                    "encoder.layer.0.LayerNorm.bias",
                    "embeddings.word_embeddings.weight",
                },
            )
            self.assertEqual(
                bytes(out_items["encoder.layer.0.LayerNorm.weight"]["data"]), ln_weight_bytes
            )
            self.assertEqual(
                bytes(out_items["encoder.layer.0.LayerNorm.bias"]["data"]), ln_bias_bytes
            )
            self.assertEqual(
                bytes(out_items["embeddings.word_embeddings.weight"]["data"]), embed_bytes
            )

    def test_collision_refuses_no_output_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            # Renaming "encoder.LayerNorm.gamma" -> "encoder.LayerNorm.weight"
            # collides with an ALREADY-present "encoder.LayerNorm.weight".
            self._make_fixture(
                in_path,
                {
                    "encoder.LayerNorm.gamma": ("float32", [1], bytes(4)),
                    "encoder.LayerNorm.weight": ("float32", [1], bytes(4)),
                },
            )
            with self.assertRaises(clbc.CheckpointConversionError):
                clbc.convert(in_path, out_path)
            self.assertFalse(os.path.exists(out_path))

    def test_zero_renames_refuses_no_output_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "already_correct.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            # Already jammi-compatible names -- nothing to rename.
            self._make_fixture(
                in_path,
                {
                    "encoder.LayerNorm.weight": ("float32", [1], bytes(4)),
                    "encoder.LayerNorm.bias": ("float32", [1], bytes(4)),
                },
            )
            with self.assertRaises(clbc.CheckpointConversionError):
                clbc.convert(in_path, out_path)
            self.assertFalse(os.path.exists(out_path))

    def test_in_place_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.safetensors")
            self._make_fixture(path, {"encoder.LayerNorm.gamma": ("float32", [1], bytes(4))})
            with self.assertRaises(clbc.CheckpointConversionError):
                clbc.convert(path, path)

    def test_main_writes_report_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            self._make_fixture(in_path, {"encoder.LayerNorm.gamma": ("float32", [1], bytes(4))})
            rc = clbc.main([in_path, out_path])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))

    def test_main_zero_renames_exit_3_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "already_correct.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            self._make_fixture(in_path, {"encoder.LayerNorm.weight": ("float32", [1], bytes(4))})
            rc = clbc.main([in_path, out_path])
            self.assertEqual(rc, 3)
            self.assertFalse(os.path.exists(out_path))

    def test_main_in_place_exit_2_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.safetensors")
            self._make_fixture(path, {"encoder.LayerNorm.gamma": ("float32", [1], bytes(4))})
            rc = clbc.main([path, path])
            self.assertEqual(rc, 2)


@unittest.skipUnless(_SAFETENSORS_AVAILABLE, "safetensors package not importable in this env")
class MainCliUsageTests(unittest.TestCase):
    """Usage-level refusals below the "is safetensors importable" check --
    package-gated the same as `ConvertFileLevelTests` (`main()`'s own
    importability check runs FIRST, same order as `fixture_width_report.py`'s
    precedent, so exercising anything past it needs the real package)."""

    def test_main_missing_input_file_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "does_not_exist.safetensors")
            out_path = os.path.join(tmp, "out.safetensors")
            rc = clbc.main([in_path, out_path])
            self.assertEqual(rc, 2)


class NotImportableRefusalTests(unittest.TestCase):
    def test_missing_safetensors_package_refuses_loudly_no_output(self):
        """Simulates the "safetensors not importable" path via a real
        subprocess -- skipped if the real package IS installed in this
        environment (there is then nothing to hide it behind without
        also breaking other tests' imports)."""
        if _SAFETENSORS_AVAILABLE:
            self.skipTest(
                "safetensors package is installed in this environment; cannot exercise the "
                "ImportError path without uninstalling it"
            )
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "legacy.safetensors"
            in_path.write_bytes(b"not a real safetensors file, just needs to exist")
            out_path = Path(tmp) / "converted.safetensors"
            result = subprocess.run(
                [sys.executable, SCRIPT, str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
