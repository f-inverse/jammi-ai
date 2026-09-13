#!/usr/bin/env python3
"""Tests for `packaging/server-cu12/verify_link_set.py`'s classification.

The property under test is fail-CLOSED classification: every `DT_NEEDED`
soname of the cu12 binary is either platform, driver-provided or delivered by
a declared wheel component, and anything else FAILS. The earlier predicate
asked the opposite question — "does this soname start with `libcu` or `libnv`,
and if not, skip it" — which passes a wheel that bundles no NCCL while its
binary hard-links `libnccl.so.2`, because that soname starts with neither. The
`libnccl` case below is the standing oracle for that shape; the others hold
the classification honest in the directions it could over-reach.

The PASS fixture is the measured `DT_NEEDED` list of the binary the check
actually runs on (`_pypi-server.yml:144`, `target/release/jammi-server`), read
off the `server-cu12-binary` artifact of run 34717957779 — not a hand-written
approximation of it.

Hermetic: no readelf, no binary, no network. `needed_libs` is replaced by a
fixture, so this exercises `check()`'s classification and its exit code, which
is the whole of what CI consumes.

Run: `python3 ci/scripts/test_verify_link_set.py`
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "packaging" / "server-cu12" / "verify_link_set.py"

# The real link set, measured. Any change to it is a change to what the wheel
# must deliver, so this list is a fixture, never a paraphrase.
MEASURED_LINK_SET = [
    "libcudart.so.12",
    "libstdc++.so.6",
    "libcuda.so.1",
    "libnvrtc.so.12",
    "libcurand.so.10",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libdl.so.2",
    "libgcc_s.so.1",
    "librt.so.1",
    "libpthread.so.0",
    "libm.so.6",
    "libmvec.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
]


def load_module():
    spec = importlib.util.spec_from_file_location("verify_link_set", MODULE_PATH)
    assert spec is not None and spec.loader is not None, MODULE_PATH
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_link_set"] = mod
    spec.loader.exec_module(mod)
    return mod


class LinkSetCheck(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def run_check(self, needed: list[str]) -> tuple[int, str, str]:
        self.mod.needed_libs = lambda _binary: list(needed)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.mod.check("/nonexistent/jammi-server")
        return rc, out.getvalue(), err.getvalue()

    def test_measured_link_set_passes(self):
        rc, out, err = self.run_check(MEASURED_LINK_SET)
        self.assertEqual(rc, 0, f"the shipped link set must pass; stderr={err}")
        self.assertIn("OK", out)

    def test_nccl_is_not_silently_skipped(self):
        """The fail-open shape: `libnccl` starts with neither `libcu` nor
        `libnv`, and the wheel declares no NCCL component."""
        rc, out, err = self.run_check(MEASURED_LINK_SET + ["libnccl.so.2"])
        self.assertEqual(rc, 1, f"libnccl.so.2 must FAIL the check; stdout={out}")
        self.assertIn("libnccl.so.2", err)
        self.assertNotIn("OK", out)

    def test_any_unclassified_library_fails(self):
        """Not a check for NCCL by name: an arbitrary unclassified soname —
        a system library the platform does not promise, a bundled third-party
        `.so` — fails the same way."""
        for soname in ("libfoo.so.1", "libz.so.1", "libssl.so.3", "libnuma.so.1"):
            with self.subTest(soname=soname):
                rc, _out, err = self.run_check(MEASURED_LINK_SET + [soname])
                self.assertEqual(rc, 1)
                self.assertIn(soname, err)

    def test_new_cuda_component_still_fails(self):
        """The behaviour the check already had stays: a CUDA library no
        declared component delivers is a FAIL."""
        rc, _out, err = self.run_check(MEASURED_LINK_SET + ["libcusparse.so.12"])
        self.assertEqual(rc, 1)
        self.assertIn("libcusparse.so.12", err)

    def test_classification_of_each_measured_soname(self):
        expected = {
            "libcudart.so.12": "covered",
            "libcublas.so.12": "covered",
            "libcublasLt.so.12": "covered",
            "libcurand.so.10": "covered",
            "libnvrtc.so.12": "covered",
            "libcuda.so.1": "driver",
            "libstdc++.so.6": "platform",
            "libdl.so.2": "platform",
            "libgcc_s.so.1": "platform",
            "librt.so.1": "platform",
            "libpthread.so.0": "platform",
            "libm.so.6": "platform",
            "libmvec.so.1": "platform",
            "libc.so.6": "platform",
            "ld-linux-x86-64.so.2": "platform",
        }
        self.assertEqual(sorted(expected), sorted(MEASURED_LINK_SET))
        for soname, bucket in expected.items():
            with self.subTest(soname=soname):
                self.assertEqual(self.mod.classify(soname), bucket)
        self.assertEqual(self.mod.classify("libnccl.so.2"), "unclassified")

    def test_loader_prefix_admits_only_the_loader(self):
        self.assertTrue(self.mod.is_platform("ld-linux-aarch64"))
        self.assertFalse(self.mod.is_platform("libnccl"))
        self.assertFalse(self.mod.is_platform("libcudart"))

    def test_buckets_are_disjoint(self):
        covered = {s for stems in self.mod.COVERED.values() for s in stems}
        self.assertFalse(covered & self.mod.DRIVER_PROVIDED)
        self.assertFalse(covered & self.mod.PLATFORM)
        self.assertFalse(self.mod.DRIVER_PROVIDED & self.mod.PLATFORM)

    def test_covered_keys_match_the_entry_point_components(self):
        """`COVERED`'s keys, `_CUDA_COMPONENTS` and the `nvidia-*-cu12` pins
        are one contract stated three times; the first two are checkable
        here."""
        entry = (MODULE_PATH.parent / "jammi_server" / "_entry.py").read_text()
        for component in self.mod.COVERED:
            self.assertIn(f'"{component}"', entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
