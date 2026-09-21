#!/usr/bin/env python3
"""Hermetic tests of `torch_venv.py`'s provisioning verb: no package is
downloaded, the installer is a stand-in.

Run: python3 ci/scripts/perf/test_torch_venv.py
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(venv_dir: Path):
    """`torch_venv`, resolved against `venv_dir`."""
    os.environ["TORCH_VENV"] = str(venv_dir)
    import torch_venv

    return importlib.reload(torch_venv)


def installer(returncode: int, stderr: str = ""):
    calls = []

    def install(python, requirements):
        calls.append((python, requirements))
        return subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)

    return install, calls


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "TORCH_VENV", None)
        self.venv = Path(self.tmp.name) / "venv"

    def test_the_venv_is_built_from_the_interpreter_running_the_verb(self):
        torch_venv = load(self.venv)
        install, calls = installer(1, "no wheel")
        torch_venv.provision(install)
        self.assertEqual(calls, [(torch_venv.TORCH_PY, torch_venv.REQUIREMENTS)])
        built_from = subprocess.run(
            [str(torch_venv.TORCH_PY), "-c", "import sys; print(sys.base_prefix, sys.version.split()[0])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        self.assertEqual(built_from, [sys.base_prefix, sys.version.split()[0]])

    def test_an_interpreter_the_packages_cannot_be_installed_for_is_refused_by_name(self):
        torch_venv = load(self.venv)
        install, _ = installer(1, "ERROR: No matching distribution found for torch")
        why = torch_venv.provision(install)
        self.assertIsNotNone(why)
        self.assertIn(sys.executable, why)
        self.assertIn(sys.version.split()[0], why)
        self.assertIn("No matching distribution found for torch", why)

    def test_an_install_that_succeeds_but_leaves_a_package_unimportable_is_refused(self):
        torch_venv = load(self.venv)
        install, _ = installer(0)
        why = torch_venv.provision(install)
        self.assertIsNotNone(why)
        self.assertIn("does not import", why)

    def test_a_usable_venv_is_reused_without_installing(self):
        torch_venv = load(self.venv)
        torch_venv.PACKAGES = ("json", "os")
        install, calls = installer(0)
        self.assertIsNone(torch_venv.provision(install))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(torch_venv.provision(install))
        self.assertEqual(len(calls), 1, "a venv that already imports every package is not reinstalled")


if __name__ == "__main__":
    unittest.main()
