#!/usr/bin/env python3
"""Tests for `packaging/server-cu12/verify_link_set.py`'s classification.

The property under test is fail-CLOSED classification: every `DT_NEEDED`
soname of the cu12 binary is either platform, driver-provided or delivered by
a declared wheel component, and anything else FAILS. A `libcu`/`libnv` prefix
test alone is not sufficient: it fails OPEN on any soname that starts with
neither prefix, which is exactly how a wheel could once print "OK" while
bundling no NCCL and its binary hard-linked `libnccl.so.2` regardless — that
soname is now `covered` (the `nccl` component), so the standing fail-open
oracle below is `libmpi.so.40`, a plausible future link that starts with
neither prefix and is declared by no component; the other cases hold the
classification honest in the directions it could over-reach.

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
import subprocess
import sys
import types
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

    def test_an_unclassified_library_is_not_silently_skipped(self):
        """The fail-open shape: `libmpi` starts with neither `libcu` nor
        `libnv`, and the wheel declares no MPI component. This is the
        standing oracle for "a soname naming no prefix and no component" —
        `libnccl.so.2` cannot serve this role because the `nccl` component
        declares it (see `test_classification_of_each_measured_soname`), so
        this soname must stay absent from every declared component or the
        oracle silently stops testing anything."""
        oracle = "libmpi.so.40"
        declared = {s for stems in self.mod.COVERED.values() for s in stems}
        self.assertNotIn(
            self.mod.soname_stem(oracle),
            declared,
            f"{oracle} must be declared by no component for this oracle to hold",
        )
        rc, out, err = self.run_check(MEASURED_LINK_SET + [oracle])
        self.assertEqual(rc, 1, f"{oracle} must FAIL the check; stdout={out}")
        self.assertIn(oracle, err)
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
        # libnccl.so.2 is delivered by the `nccl` component (a `candle-core/nccl`
        # build links it), not by MEASURED_LINK_SET's own binary -- classified
        # separately here rather than folded into `expected` above.
        self.assertEqual(self.mod.classify("libnccl.so.2"), "covered")

    def test_loader_prefix_admits_only_the_loader(self):
        self.assertTrue(self.mod.is_platform("ld-linux-aarch64"))
        self.assertFalse(self.mod.is_platform("libnccl"))
        self.assertFalse(self.mod.is_platform("libcudart"))

    def test_buckets_are_disjoint(self):
        covered = {s for stems in self.mod.COVERED.values() for s in stems}
        self.assertFalse(covered & self.mod.DRIVER_PROVIDED)
        self.assertFalse(covered & self.mod.PLATFORM)
        self.assertFalse(self.mod.DRIVER_PROVIDED & self.mod.PLATFORM)

    def test_an_empty_link_set_fails(self):
        """Zero classified entries is not "everything is classified". The
        `needed_libs` fixture is bypassed the same way the tests above do
        it; the suite below drives the REAL parser."""
        rc, out, err = self.run_check([])
        self.assertEqual(rc, 1, f"an empty DT_NEEDED list must FAIL; stdout={out}")
        self.assertNotIn("OK", out)

    def test_covered_keys_match_the_entry_point_components(self):
        """`COVERED`'s keys, `_CUDA_COMPONENTS` and the `nvidia-*-cu12` pins
        are one contract stated three times; the first two are checkable
        here."""
        entry = (MODULE_PATH.parent / "jammi_server" / "_entry.py").read_text()
        for component in self.mod.COVERED:
            self.assertIn(f'"{component}"', entry)


class ExtractionThroughTheRealParser(unittest.TestCase):
    """The suite above replaces `needed_libs` wholesale, so it can never see
    a failure of `needed_libs` ITSELF. These cases patch one level lower —
    the module's own `subprocess`, so the REAL `needed_libs` runs its real
    regex over a real (fixture) `readelf -d` dump — which is the only way
    the "the tool produced something this parser does not match" case is
    reachable at all. Still hermetic: no readelf, no binary."""

    BINARY = "/nonexistent/target/release/jammi-server"

    #: A real `readelf -d` line, as the parser expects it.
    WELL_FORMED_DUMP = "\n".join(
        f" 0x0000000000000001 (NEEDED)             Shared library: [{soname}]"
        for soname in MEASURED_LINK_SET
    )
    #: The same information in a shape the parser does NOT match — the
    #: bracket-less spelling. Not a hypothetical taste in fixtures: this is
    #: what an output-format change or a different binutils build looks like
    #: from inside `needed_libs`, and `libnccl.so.2` is here so the case is
    #: unmistakably "the one soname this check exists to catch, extracted by
    #: nothing".
    UNPARSEABLE_DUMP = "\n".join(
        [
            "Dynamic section at offset 0x1234 contains 30 entries:",
            "  Tag        Type                         Name/Value",
            " 0x0000000000000001 (NEEDED) Shared library: libnccl.so.2",
            " 0x0000000000000001 (NEEDED) Shared library: libcudart.so.12",
        ]
    )
    EMPTY_DUMP = ""

    def setUp(self):
        self.mod = load_module()

    def run_with_dump(self, dump: str) -> tuple[int, str, str, list]:
        """`check()` driven through the REAL `needed_libs`, with only the
        subprocess call replaced. The module's own `subprocess` attribute is
        rebound (never `subprocess.run` on the shared module object), so no
        other test — in this file or any other — sees the patch."""
        calls: list = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout=dump, stderr="")

        self.mod.subprocess = types.SimpleNamespace(
            run=fake_run, CalledProcessError=subprocess.CalledProcessError
        )
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.mod.check(self.BINARY)
        return rc, out.getvalue(), err.getvalue(), calls

    def test_well_formed_dump_is_parsed_and_passes(self):
        """The control: the same plumbing, a dump the parser DOES match,
        must extract the measured link set and pass — so a FAIL below is the
        extraction's, not the fixture harness's."""
        rc, out, err, calls = self.run_with_dump(self.WELL_FORMED_DUMP)
        self.assertEqual(calls, [["readelf", "-d", self.BINARY]])
        self.assertEqual(
            self.mod.needed_libs(self.BINARY), MEASURED_LINK_SET, "the real parser must read the dump"
        )
        self.assertEqual(rc, 0, f"stderr={err}")
        self.assertIn("OK", out)

    def test_unparseable_dump_fails_naming_the_binary_and_the_tool(self):
        rc, out, err, _calls = self.run_with_dump(self.UNPARSEABLE_DUMP)
        self.assertEqual(
            self.mod.needed_libs(self.BINARY), [], "fixture premise: this dump must not parse"
        )
        self.assertEqual(rc, 1, f"an unparseable readelf dump must FAIL; stdout={out}")
        self.assertNotIn("OK", out)
        self.assertIn(self.BINARY, err)
        self.assertIn("readelf", err)

    def test_empty_dump_fails_naming_the_binary_and_the_tool(self):
        rc, out, err, _calls = self.run_with_dump(self.EMPTY_DUMP)
        self.assertEqual(rc, 1, f"an empty readelf dump must FAIL; stdout={out}")
        self.assertNotIn("OK", out)
        self.assertIn(self.BINARY, err)
        self.assertIn("readelf", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
