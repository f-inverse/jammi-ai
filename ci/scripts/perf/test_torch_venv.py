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

    def install(python, requirements, index=None):
        calls.append((python, requirements))
        return subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)

    return install, calls


def NO_DRIVER():
    return None


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "TORCH_VENV", None)
        self.venv = Path(self.tmp.name) / "venv"

    def test_the_venv_is_built_from_the_interpreter_running_the_verb(self):
        torch_venv = load(self.venv)
        install, calls = installer(1, "no wheel")
        torch_venv.provision(install, driver=NO_DRIVER)
        self.assertEqual(calls, [(torch_venv.TORCH_PY, (torch_venv.TORCH_REQUIREMENT,))])
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
        why = torch_venv.provision(install, driver=NO_DRIVER)
        self.assertIsNotNone(why)
        self.assertIn(sys.executable, why)
        self.assertIn(sys.version.split()[0], why)
        self.assertIn("No matching distribution found for torch", why)

    def test_an_install_that_succeeds_but_leaves_a_package_unimportable_is_refused(self):
        torch_venv = load(self.venv)
        install, _ = installer(0)
        why = torch_venv.provision(install, driver=NO_DRIVER)
        self.assertIsNotNone(why)
        self.assertIn("does not import", why)

    def test_a_usable_venv_is_reused_without_installing(self):
        torch_venv = load(self.venv)
        torch_venv.PACKAGES = ("json", "os")
        install, calls = installer(0)
        self.assertIsNone(torch_venv.provision(install, driver=NO_DRIVER))
        self.assertEqual(len(calls), 2, "torch from its index, then the rest")
        self.assertIsNone(torch_venv.provision(install, driver=NO_DRIVER))
        self.assertEqual(len(calls), 2, "a venv that already imports every package is not reinstalled")


class ProvisionGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "TORCH_VENV", None)
        self.torch_venv = load(Path(self.tmp.name) / "venv")
        self.torch_venv.PACKAGES = ("json", "os")

    def test_the_graph_packages_install_on_the_usable_venv_the_source_build_last(self):
        self.torch_venv.GRAPH_PACKAGES = ("json", "no_such_graph_package")
        install, installed = installer(0)
        build, built = installer(0)
        why = self.torch_venv.provision_graph(install, build, driver=NO_DRIVER)
        self.assertEqual(installed[-1], (self.torch_venv.TORCH_PY, self.torch_venv.GRAPH_REQUIREMENTS))
        self.assertEqual(built, [(self.torch_venv.TORCH_PY, self.torch_venv.GRAPH_SOURCE_BUILDS)])
        self.assertIn("does not import", why, "an install that leaves a package unimportable is refused")

    def test_a_failed_source_build_is_refused_by_name(self):
        self.torch_venv.GRAPH_PACKAGES = ("json", "no_such_graph_package")
        install, _ = installer(0)
        build, _ = installer(1, "error: command 'g++' failed")
        why = self.torch_venv.provision_graph(install, build, driver=NO_DRIVER)
        self.assertIn("torch_cluster", why)
        self.assertIn("command 'g++' failed", why)

    def test_a_venv_with_the_graph_packages_is_reused_without_installing(self):
        self.torch_venv.GRAPH_PACKAGES = ("json",)
        install, installed = installer(0)
        build, built = installer(0)
        self.assertIsNone(self.torch_venv.provision_graph(install, build, driver=NO_DRIVER))
        self.assertEqual((len(installed), built), (2, []), "the base venv only: torch, then the rest")


BANNER = "| NVIDIA-SMI 570.172.08    Driver Version: 570.172.08    CUDA Version: {} |"


class DriverMatchedWheel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "TORCH_VENV", None)
        self.torch_venv = load(Path(self.tmp.name) / "venv")

    def test_the_driver_version_is_read_from_the_banner(self):
        read = self.torch_venv.driver_cuda_version
        self.assertEqual(read(lambda: BANNER.format("12.8")), (12, 8))
        self.assertIsNone(read(lambda: None), "no driver on this box")

    def test_the_index_is_the_newest_the_driver_supports(self):
        index = self.torch_venv.wheel_index
        self.assertEqual(index((12, 8)), "cu128", "a cu130 wheel would run on the CPU here")
        self.assertEqual(index((13, 1)), "cu130")
        self.assertEqual(index((12, 7)), "cu126")
        self.assertIsNone(index((11, 4)))

    def test_torch_is_installed_from_the_drivers_index_and_the_rest_from_the_default(self):
        calls = []

        def install(python, requirements, index=None):
            calls.append((requirements, index))
            return subprocess.CompletedProcess([], 1, "", "stop here")

        self.torch_venv.provision(install, driver=lambda: (12, 8), check=lambda: None)
        self.assertEqual(calls, [((self.torch_venv.TORCH_REQUIREMENT,), "cu128")])

    def test_a_driver_older_than_every_index_is_refused_by_name(self):
        why = self.torch_venv.provision(lambda *a: None, driver=lambda: (11, 4))
        self.assertIn("CUDA 11.4", why)
        self.assertIn("update the driver", why)

    def test_a_failed_preflight_names_the_missing_headers(self):
        failed = subprocess.CompletedProcess([], 1, "", "fatal error: Python.h: No such file or directory")
        why = self.torch_venv.preflight(lambda: failed)
        self.assertIn("forward and backward on cuda:0", why)
        self.assertIn("python3-devel", why)
        self.assertIsNone(self.torch_venv.preflight(lambda: subprocess.CompletedProcess([], 0, "", "")))


if __name__ == "__main__":
    unittest.main()
