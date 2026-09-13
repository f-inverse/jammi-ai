#!/usr/bin/env python3
"""The cu12 wheel's CUDA-component contract, stated in three files, checked as
one.

`packaging/server-cu12/` says the same thing three times, and it has to:

  * `verify_link_set.py`'s `COVERED` maps a component directory to the SONAME
    stems it delivers — the CI-side classification of the binary's `DT_NEEDED`
    entries;
  * `jammi_server/_entry.py`'s `_CUDA_COMPONENTS` is the runtime half: the
    directories the console script actually prepends to `LD_LIBRARY_PATH`
    before `execve`;
  * `pyproject.toml`'s `nvidia-*-cu12` dependencies are what pip installs, so
    they decide whether those directories exist at all.

Any one of the three alone is a claim the other two can silently contradict. A
component in `COVERED` but not in `_CUDA_COMPONENTS` means CI declares a
library delivered while the shim never puts it on the path; a component in both
but with no pin means the shim looks for a directory pip never installed. Both
end the same way — the binary fails at `execve` on a user's machine, with every
CI lane green, because the `DT_NEEDED` entry the loader wanted resolved
nowhere. `verify_link_set.py`'s own header already says the three "describe one
contract"; this file is what makes that sentence checkable.

The component -> distribution mapping is mechanical, not a table:
`nvidia-{component with `_` as `-`}-cu12`, which is how all five of
`cuda_runtime`, `cublas`, `curand`, `cuda_nvrtc` and `nccl` are named on PyPI.
A table would be a fourth statement of the same contract.

Scope, stated so it is not mistaken for more: this suite reads the three
enumerations against each other. It does NOT check that a pinned wheel really
ships the SONAME `COVERED` claims (that is a fact about someone else's artifact
— the `nvidia/nccl/lib/libnccl.so.2` measurement is recorded where the entry
was added, in `COVERED`'s own comment), and it does not run the link-set check
itself; `ci/scripts/test_verify_link_set.py` drives `classify()`/`check()` over
a measured `DT_NEEDED` fixture and states the entry-point leg below a second
time, which is a duplication worth having: these two suites fail for different
reasons and neither depends on the other landing.

Hermetic: three file reads, stdlib only, no network, no wheel, no binary.

Run: `python3 ci/scripts/test_cu12_component_contract.py`
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGING = REPO_ROOT / "packaging" / "server-cu12"
VERIFY_PATH = PACKAGING / "verify_link_set.py"
ENTRY_PATH = PACKAGING / "jammi_server" / "_entry.py"
PYPROJECT_PATH = PACKAGING / "pyproject.toml"
README_PATH = PACKAGING / "README.md"


def load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_link_set", VERIFY_PATH)
    assert spec is not None and spec.loader is not None, VERIFY_PATH
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_link_set"] = mod
    spec.loader.exec_module(mod)
    return mod


def entry_components() -> tuple:
    """`_entry.py`'s `_CUDA_COMPONENTS`, read from the AST rather than by
    importing the module — importing it is harmless today but the file exists
    to `execve` a binary, and a suite should not be one edit away from doing
    that."""
    tree = ast.parse(ENTRY_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_CUDA_COMPONENTS":
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"_CUDA_COMPONENTS not found in {ENTRY_PATH}")


def pinned_distributions() -> dict:
    """The `nvidia-*-cu12` requirements in `pyproject.toml`'s `dependencies`,
    as {distribution: specifier}. A plain-text scan, not a TOML parse: python
    3.9 (this package's own `requires-python` floor) has no `tomllib`, and the
    property here is about the literal lines a reviewer reads."""
    text = PYPROJECT_PATH.read_text()
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert block, f"no `dependencies = [...]` block in {PYPROJECT_PATH}"
    pins = {}
    for name, spec in re.findall(r'"(nvidia-[a-z0-9-]+-cu12)([^"]*)"', block.group(1)):
        pins[name] = spec
    return pins


def distribution_for(component: str) -> str:
    """`cuda_runtime` -> `nvidia-cuda-runtime-cu12`."""
    return f"nvidia-{component.replace('_', '-')}-cu12"


class ComponentContract(unittest.TestCase):
    def setUp(self):
        self.covered = load_verify_module().COVERED
        self.components = entry_components()
        self.pins = pinned_distributions()

    def test_the_three_enumerations_are_the_same_set(self):
        self.assertEqual(
            sorted(self.covered),
            sorted(self.components),
            "`COVERED`'s keys and `_CUDA_COMPONENTS` disagree",
        )
        self.assertEqual(
            sorted(distribution_for(c) for c in self.components),
            sorted(self.pins),
            "`_CUDA_COMPONENTS` and the `nvidia-*-cu12` pins disagree",
        )

    def test_every_component_is_pinned_to_a_version(self):
        """A dependency with no specifier floats to whatever PyPI holds on the
        install day — for a library the binary links by SONAME, that is a
        different library than the one it was built against."""
        for component in self.components:
            with self.subTest(component=component):
                dist = distribution_for(component)
                self.assertIn(dist, self.pins, f"{dist} is in no `dependencies` entry")
                self.assertTrue(
                    self.pins[dist].startswith("=="),
                    f"{dist} is not pinned with `==` (got {self.pins[dist]!r})",
                )

    def test_nccl_is_in_the_contract(self):
        """The component this unit adds, named in all three files at once —
        `jammi-ai`'s `cuda` feature includes `candle-core/nccl`, so the binary
        carries `DT_NEEDED libnccl.so.2` and the wheel must deliver it."""
        self.assertIn("nccl", self.covered)
        self.assertEqual(self.covered["nccl"], {"libnccl"})
        self.assertIn("nccl", self.components)
        self.assertEqual(self.pins["nvidia-nccl-cu12"], "==2.23.4.*")

    def test_the_readme_names_every_component(self):
        """The prose the user reads is a fourth copy of this list and rots the
        same way the other three would, so a component that appears nowhere in
        it fails here.

        Per COMPONENT, not per stem: the README names the libraries a reader
        would recognise, and a component may deliver a second object that rides
        along inside it (`libnvrtc-builtins` inside the nvrtc wheel) which the
        prose has no reason to enumerate. What must never happen is a whole
        component — a new `nvidia-*-cu12` dependency the wheel now installs —
        going unmentioned."""
        readme = README_PATH.read_text()
        for component, stems in self.covered.items():
            with self.subTest(component=component):
                self.assertTrue(
                    any(stem in readme for stem in stems),
                    f"{README_PATH.name} names none of {sorted(stems)} for component {component!r}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
