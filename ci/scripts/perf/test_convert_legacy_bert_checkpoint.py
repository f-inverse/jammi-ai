#!/usr/bin/env python3
"""`convert_legacy_bert_checkpoint.py`'s own suite (unit #356 census
execution fix, defect 2): drives the real `renamed_name`/`convert`
functions.

`renamed_name` (the suffix-anchored `.gamma`->`.weight` / `.beta`->`.bias`
string logic) is a PURE function of a name string -- covered without the
`safetensors` package, the same "pure function first" split
`fixture_width_report.py`'s own suite already uses for its
package-independent half.

`convert`'s file-level behavior (renames happen, values AND shape/dtype
are byte-identical/preserved for both f32 and a non-f32 dtype, an
unsupported/packed dtype refuses, the input's `__metadata__` block is
carried through unchanged, a rename collision refuses, a zero-rename
result refuses, `out_path == in_path` refuses) needs a REAL synthetic
safetensors fixture built in-test via the real `safetensors` package
(`TensorSpec`/`serialize`) -- exercised only when `safetensors` is
actually importable, skipped (never silently treated as passing)
otherwise, exactly the `tokenizers`-dependent-half precedent
`fixture_width_report.py`'s own suite already established. `ci.yml`'s
Guard job installs `safetensors` for THIS ONE matrix leg only (phase-4
audit round-2 re-audit advisory 4, the same `if:`-gated exception the
"pod build substrate" leg's own Rust toolchain already uses) so these
arms are actually CI-EXECUTED, not merely proven in a local venv and
silently skipped in the real gate.

The "package not importable" loud-refusal path is exercised via a real
subprocess, skipped if the real package IS installed in this environment
(there is then nothing to hide it behind without also breaking other
tests' imports) -- the same `NotImportableRefusalTests` shape
`test_fixture_width_report.py` uses for `tokenizers`. Because CI now
installs `safetensors` for this leg, THIS ONE test is the one arm that
skips in CI (and runs for real only in an environment without the
package, e.g. this repo's own base dev checkout) -- the inverse trade-off
of the file-level tests above, and the same trade-off
`test_fixture_width_report.py` already accepts for `tokenizers` (never
installed there at all).

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

    def test_literal_bare_gamma_name_is_untouched(self):
        # Phase-4 audit advisory 4: `renamed_name` is a literal
        # `str.endswith('.gamma')` suffix check, NOT a tokenized "last
        # dot-separated path component" one -- a name that IS just the
        # bare word "gamma", with no leading dot at all, has no
        # `.gamma` suffix and must be left untouched (a tokenized
        # last-component check would have wrongly renamed this).
        self.assertEqual(clbc.renamed_name("gamma"), "gamma")
        self.assertEqual(clbc.renamed_name("beta"), "beta")


@unittest.skipUnless(_SAFETENSORS_AVAILABLE, "safetensors package not importable in this env")
class ConvertFileLevelTests(unittest.TestCase):
    """Drives the real `convert()` against a TINY synthetic safetensors
    fixture built in-test via the real `safetensors` package -- never a
    real BERT checkpoint."""

    @staticmethod
    def _make_fixture(path: str, tensors: dict, metadata: dict | None = None) -> None:
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
        blob = serialize(tensor_dict, metadata=metadata)
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
            # A non-f32 dtype arm (phase-4 audit advisory 3) -- exercises
            # the `_DTYPE_CODE_TO_CTOR` on-disk-code round-trip for a real
            # checkpoint dtype other than the stock bert-base-uncased F32.
            bf16_bytes = bytes(range(40, 44))  # 2 x bfloat16
            self._make_fixture(
                in_path,
                {
                    "encoder.layer.0.LayerNorm.gamma": ("float32", [4], ln_weight_bytes),
                    "encoder.layer.0.LayerNorm.beta": ("float32", [4], ln_bias_bytes),
                    "embeddings.word_embeddings.weight": ("float32", [2], embed_bytes),
                    "encoder.layer.0.LayerNorm.bf16.gamma": ("bfloat16", [2], bf16_bytes),
                },
            )
            n = clbc.convert(in_path, out_path)
            self.assertEqual(n, 3)

            out_items = self._read_back(out_path)
            self.assertEqual(
                set(out_items),
                {
                    "encoder.layer.0.LayerNorm.weight",
                    "encoder.layer.0.LayerNorm.bias",
                    "embeddings.word_embeddings.weight",
                    "encoder.layer.0.LayerNorm.bf16.weight",
                },
            )
            # Shape AND dtype are preserved, not just the raw data bytes --
            # a rename must never reinterpret a tensor's own type.
            self.assertEqual(out_items["encoder.layer.0.LayerNorm.weight"]["shape"], [4])
            self.assertEqual(out_items["encoder.layer.0.LayerNorm.weight"]["dtype"], "F32")
            self.assertEqual(out_items["encoder.layer.0.LayerNorm.bf16.weight"]["shape"], [2])
            self.assertEqual(out_items["encoder.layer.0.LayerNorm.bf16.weight"]["dtype"], "BF16")
            self.assertEqual(
                bytes(out_items["encoder.layer.0.LayerNorm.weight"]["data"]), ln_weight_bytes
            )
            self.assertEqual(
                bytes(out_items["encoder.layer.0.LayerNorm.bias"]["data"]), ln_bias_bytes
            )
            self.assertEqual(
                bytes(out_items["embeddings.word_embeddings.weight"]["data"]), embed_bytes
            )
            self.assertEqual(
                bytes(out_items["encoder.layer.0.LayerNorm.bf16.weight"]["data"]), bf16_bytes
            )

    def test_unsupported_dtype_refuses_no_output_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            # float8_e4m3fn is a real, constructible safetensors dtype (its
            # on-disk code is "F8_E4M3") but deliberately NOT in
            # `_DTYPE_CODE_TO_CTOR` (module doc: packed/exotic dtypes
            # refuse rather than risk a silent mis-encode).
            self._make_fixture(
                in_path,
                {"encoder.LayerNorm.gamma": ("float8_e4m3fn", [2], bytes([1, 2]))},
            )
            with self.assertRaises(clbc.CheckpointConversionError) as ctx:
                clbc.convert(in_path, out_path)
            self.assertIn("F8_E4M3", str(ctx.exception))
            self.assertFalse(os.path.exists(out_path))

    def test_metadata_block_carried_through_unchanged(self):
        # Phase-4 audit advisory 4: `safetensors.deserialize()` itself
        # drops `__metadata__` entirely -- `convert()` must still carry it
        # through to the output UNCHANGED via `_read_metadata` reading the
        # input's own header bytes directly.
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            self._make_fixture(
                in_path,
                {"encoder.LayerNorm.gamma": ("float32", [1], bytes(4))},
                metadata={"format": "pt", "custom_annotation": "hello"},
            )
            clbc.convert(in_path, out_path)
            out_raw = Path(out_path).read_bytes()
            self.assertEqual(
                clbc._read_metadata(out_raw), {"format": "pt", "custom_annotation": "hello"}
            )

    def test_metadata_block_absent_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "legacy.safetensors")
            out_path = os.path.join(tmp, "converted.safetensors")
            self._make_fixture(in_path, {"encoder.LayerNorm.gamma": ("float32", [1], bytes(4))})
            clbc.convert(in_path, out_path)
            out_raw = Path(out_path).read_bytes()
            self.assertIsNone(clbc._read_metadata(out_raw))

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
            # Both the exact exit code AND the named `::error::` marker
            # (phase-4 audit advisory 2): `assertNotEqual(rc, 0)` alone
            # cannot distinguish this module's own LOUD, declared
            # ImportError refusal (exit 1, a clear stderr message) from an
            # UNHANDLED traceback elsewhere in the script also exiting
            # nonzero -- exactly the failure state this test exists to
            # exclude.
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("::error::convert_legacy_bert_checkpoint", result.stderr)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
