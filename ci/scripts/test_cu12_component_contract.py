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

## A fourth binding: the cu12 TARBALL restates the same partition

`ci/scripts/bundle_cuda_libs.sh` answers the identical classification
question — "is this SONAME the tarball's to carry, the host's, or the
driver's" — for the cu12 TARBALL rather than the WHEEL. Two artifacts, one
partition. The failure mode is a SECOND, independent enumeration in the shell
script (a `BUNDLE_FLOOR_STEMS`/platform/driver list restated by hand): never
checked against `verify_link_set.py`, it would be a fifth statement of the
same fact, free to drift. `BundleScriptContract` below is that binding, in the
same idiom as
`ComponentContract` above: read each side's OWN enumeration (here, by
sourcing the real shell script in a subprocess and asking its real
functions/constants — no re-parsing of shell text by regex) and assert
agreement.

Run: `python3 ci/scripts/test_cu12_component_contract.py`
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGING = REPO_ROOT / "packaging" / "server-cu12"
VERIFY_PATH = PACKAGING / "verify_link_set.py"
ENTRY_PATH = PACKAGING / "jammi_server" / "_entry.py"
PYPROJECT_PATH = PACKAGING / "pyproject.toml"
README_PATH = PACKAGING / "README.md"
BUNDLE_SCRIPT = REPO_ROOT / "ci" / "scripts" / "bundle_cuda_libs.sh"
NATIVE_PYPROJECT_PATH = REPO_ROOT / "packaging" / "native-cu12" / "pyproject.toml"
NATIVE_BUILD_RS = REPO_ROOT / "crates" / "jammi-python" / "build.rs"


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


def pinned_distributions(pyproject: Path = PYPROJECT_PATH) -> dict:
    """The `nvidia-*-cu12` requirements in `pyproject`'s `dependencies`,
    as {distribution: specifier}. A plain-text scan, not a TOML parse: python
    3.9 (this package's own `requires-python` floor) has no `tomllib`, and the
    property here is about the literal lines a reviewer reads."""
    text = pyproject.read_text()
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert block, f"no `dependencies = [...]` block in {pyproject}"
    pins = {}
    for name, spec in re.findall(r'"(nvidia-[a-z0-9-]+-cu12)([^"]*)"', block.group(1)):
        pins[name] = spec
    return pins


def distribution_for(component: str) -> str:
    """`cuda_runtime` -> `nvidia-cuda-runtime-cu12`."""
    return f"nvidia-{component.replace('_', '-')}-cu12"


def _run_bundle_script(body: str) -> str:
    """Sources `bundle_cuda_libs.sh` in a real `bash` subprocess (never a
    regex over its text) and runs `body` in that same shell, returning
    stdout. Sourcing (not executing) means `bundle_main` never runs — the
    script's own `if [ "${BASH_SOURCE[0]}" = "$0" ]; then bundle_main "$@";
    fi` guard is false under `source`, the exact seam
    `test_bundle_cuda_libs.sh` already relies on."""
    script = f"set -e; . {shlex.quote(str(BUNDLE_SCRIPT))}; {body}"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout


def bundle_floor_stems() -> set:
    """`bundle_cuda_libs.sh`'s `BUNDLE_FLOOR_STEMS` — the fixed seven stems
    the tarball's floor stages independently of the `DT_NEEDED` closure."""
    out = _run_bundle_script('printf "%s" "$BUNDLE_FLOOR_STEMS"')
    return set(out.split())


def bundle_classify(soname: str) -> str:
    """`platform` | `driver` | `neither`, via the script's OWN predicate
    functions (`bundle_is_platform_soname`/`bundle_is_driver_soname`) — the
    ONE classifier `bundle_verify_loader_resolution` (the loader arm) and
    `bundle_resolve_closure` (the derivation's host-provided exclusion) both
    call, never a second copy restated for either."""
    q = shlex.quote(soname)
    out = _run_bundle_script(
        f"if bundle_is_platform_soname {q}; then echo platform; "
        f"elif bundle_is_driver_soname {q}; then echo driver; "
        f"else echo neither; fi"
    )
    return out.strip()


def native_runpath_components() -> tuple:
    """`crates/jammi-python/build.rs`'s `CUDA_COMPONENTS`: the directories the
    CUDA extension's RUNPATH names, in order."""
    m = re.search(r"const CUDA_COMPONENTS: \[&str; \d+\] = \[(.*?)\];",
                  NATIVE_BUILD_RS.read_text(), re.S)
    assert m, f"no `CUDA_COMPONENTS` array in {NATIVE_BUILD_RS}"
    return tuple(re.findall(r'"([^"]+)"', m.group(1)))


class NativeWheelContract(unittest.TestCase):
    """`jammi-ai-native-cu12` states the same contract twice more: its
    extension's RUNPATH (`build.rs`) is the loader-path half the server's
    `_entry.py` is, and its `nvidia-*-cu12` pins decide which of those
    directories exist. Its link set is checked by the same `verify_link_set.py`
    (`ci/scripts/build_native_cu12_wheel.sh`), so the components must be the
    server's exactly."""

    def test_the_runpath_names_the_servers_components(self):
        self.assertEqual(native_runpath_components(), entry_components())

    def test_the_native_wheel_pins_what_the_server_wheel_pins(self):
        self.assertEqual(pinned_distributions(NATIVE_PYPROJECT_PATH), pinned_distributions())


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


class BundleScriptContract(unittest.TestCase):
    """`ci/scripts/bundle_cuda_libs.sh` (the cu12 TARBALL's derivation +
    loader-verification arm) restates the SAME COVERED/PLATFORM/
    DRIVER_PROVIDED partition `verify_link_set.py` states for the cu12
    WHEEL. This is the cross-file agreement for that
    second statement: neither script's classification can drift from the
    other without going red here."""

    def setUp(self):
        self.verify = load_verify_module()

    def test_floor_stems_equal_covered_union(self):
        """The floor's fixed seven stems must be EXACTLY the union of every
        `COVERED` component's stems — the tarball's floor and the wheel's
        `COVERED` map both enumerate "every CUDA/NCCL library stem this
        binary needs that a bare `DT_NEEDED` walk cannot be trusted to find
        on its own for every path" from the SAME measured facts."""
        covered_union = {stem for stems in self.verify.COVERED.values() for stem in stems}
        self.assertEqual(
            bundle_floor_stems(),
            covered_union,
            "bundle_cuda_libs.sh's BUNDLE_FLOOR_STEMS and verify_link_set.py's "
            "COVERED union disagree",
        )

    def test_platform_stems_agree(self):
        """Every `verify_link_set.py` `PLATFORM` stem, restated as a
        representative versioned soname, classifies as `platform` under
        `bundle_cuda_libs.sh`'s own predicate — and the loader's own prefix
        family (`PLATFORM_PREFIXES`) does too."""
        for stem in sorted(self.verify.PLATFORM):
            with self.subTest(stem=stem):
                self.assertEqual(
                    bundle_classify(f"{stem}.so.6"),
                    "platform",
                    f"{stem}.so.6 not classified platform by bundle_cuda_libs.sh",
                )
        for prefix in self.verify.PLATFORM_PREFIXES:
            soname = f"{prefix}-x86-64.so.2"
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    bundle_classify(soname),
                    "platform",
                    f"{soname} (PLATFORM_PREFIXES member) not classified platform by bundle_cuda_libs.sh",
                )

    def test_driver_stems_agree(self):
        """Every `verify_link_set.py` `DRIVER_PROVIDED` stem, restated as a
        representative versioned soname, classifies as `driver` under
        `bundle_cuda_libs.sh`'s own predicate."""
        for stem in sorted(self.verify.DRIVER_PROVIDED):
            with self.subTest(stem=stem):
                self.assertEqual(
                    bundle_classify(f"{stem}.so.1"),
                    "driver",
                    f"{stem}.so.1 not classified driver by bundle_cuda_libs.sh",
                )

    def test_covered_stems_are_neither_platform_nor_driver(self):
        """Every stem the wheel's `COVERED` map delivers (and therefore the
        tarball's floor stages) must NOT be classified platform or driver by
        `bundle_cuda_libs.sh` — the three buckets are a true partition, on
        both sides at once."""
        covered_union = {stem for stems in self.verify.COVERED.values() for stem in stems}
        for stem in sorted(covered_union):
            with self.subTest(stem=stem):
                self.assertEqual(
                    bundle_classify(f"{stem}.so.12"),
                    "neither",
                    f"{stem}.so.12 (a COVERED/floor member) misclassified as platform or driver",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
