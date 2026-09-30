#!/usr/bin/env python3
"""Every wheel published to PyPI is built, and held to PyPI's size limit, on
the pull request that changes it.

**Guarded property**: a workflow that publishes to PyPI

  1. runs `ci/scripts/assert_wheel_size.sh` in the build it publishes from, so
     a wheel over PyPI's per-file limit fails its own build;
  2. has a `pull_request` trigger, so that build runs before a tag does; and
  3. when it compiles the workspace, lists every compiled input
     (`COMPILED_INPUTS`) in that trigger's `paths:` — or carries no `paths:`
     filter at all — so the pull request that grows the binary is the one
     that rebuilds the wheel.

PyPI refuses an oversized file at the tag's upload, after the other wheels of
the same lockstep release are already published; without all three, the first
run to see the size is that upload.

**Universe**: the workflows under `.github/workflows/` with a
`pypa/gh-action-pypi-publish` step. A workflow's build is everything it
reaches: the local reusable workflows it calls (`uses: ./…`) and the
`ci/scripts/*.sh` its `run:` lines and those scripts name, transitively.

**Decided over parsed documents**: a workflow is read with a YAML parser; a
command counts where it is executed — a `run:` string, or a script line that
is not a comment — never where prose mentions it. A workflow compiles the
workspace when its build runs `cargo build` or `maturin build`, or uses the
maturin action.

Fail-closed: an empty universe, an unparseable workflow, or a `uses: ./…` or
script reference that resolves to nothing is a finding, never a silent pass.

Run: `python3 ci/scripts/check_wheel_gates.py [--self-test]`
"""

from __future__ import annotations

import re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = Path(".github/workflows")
SIZE_CHECK = "ci/scripts/assert_wheel_size.sh"
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"
MATURIN_ACTION = "PyO3/maturin-action"

# What a `cargo build` of the workspace reads: a change to any of these can
# change the bytes a compiled wheel carries.
COMPILED_INPUTS = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/**",
    ".cargo/**",
    "rust-toolchain.toml",
)

_SCRIPT_REF = re.compile(r"ci/scripts/[\w./-]+\.sh")
_COMPILES = re.compile(r"\bcargo\s+build\b|\bmaturin\s+build\b")


def scalars(node, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    """Every string scalar in a parsed YAML document, with its path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from scalars(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from scalars(value, path + (str(index),))
    elif isinstance(node, str):
        yield path, node


def executed_lines(script: str) -> str:
    """A shell script without its comment lines."""
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def resolve(root: Path, target: Path) -> Path | None:
    """The file a reference names: itself, or a composite action's `action.yml`."""
    if (root / target).is_file():
        return target
    for name in ("action.yml", "action.yaml"):
        if (root / target / name).is_file():
            return target / name
    return None


class Build:
    """What one workflow's build executes and uses, over everything it reaches."""

    def __init__(self, root: Path, workflow: Path):
        self.commands: list[str] = []
        self.actions: list[str] = []
        self.findings: list[str] = []
        self.document = None
        pending = [workflow]
        seen: set[Path] = set()
        while pending:
            named = pending.pop()
            path = resolve(root, named)
            if path is None:
                self.findings.append(f"{workflow.name}: reaches `{named}`, which does not exist")
                continue
            if path in seen:
                continue
            seen.add(path)
            text = (root / path).read_text()
            if path.suffix == ".sh":
                body = executed_lines(text)
                self.commands.append(body)
            else:
                try:
                    document = yaml.safe_load(text)
                except yaml.YAMLError as error:
                    self.findings.append(f"{path}: not parseable as YAML ({error})")
                    continue
                if path == workflow:
                    self.document = document
                runs = [value for where, value in scalars(document) if where[-1:] == ("run",)]
                uses = [value for where, value in scalars(document) if where[-1:] == ("uses",)]
                self.commands.extend(runs)
                self.actions.extend(uses)
                pending.extend(Path(use[2:].split("@")[0]) for use in uses if use.startswith("./"))
                body = "\n".join(runs)
            pending.extend(Path(ref) for ref in _SCRIPT_REF.findall(body))

    def publishes(self) -> bool:
        return any(use.startswith(PUBLISH_ACTION) for use in self.actions)

    def checks_size(self) -> bool:
        return any(SIZE_CHECK.rsplit("/", 1)[1] in command for command in self.commands)

    def compiles(self) -> bool:
        return any(_COMPILES.search(command) for command in self.commands) or any(
            use.startswith(MATURIN_ACTION) for use in self.actions
        )


def pull_request_trigger(document) -> tuple[bool, list[str] | None]:
    """Whether a workflow runs on pull requests, and its `paths:` filter."""
    # YAML 1.1 reads the bare key `on` as the boolean true.
    triggers = document.get("on", document.get(True)) if isinstance(document, dict) else None
    if isinstance(triggers, str):
        return triggers == "pull_request", None
    if isinstance(triggers, list):
        return "pull_request" in triggers, None
    if not isinstance(triggers, dict) or "pull_request" not in triggers:
        return False, None
    paths = (triggers["pull_request"] or {}).get("paths")
    return True, paths


def findings(root: Path) -> list[str]:
    out: list[str] = []
    publishers = 0
    for path in sorted((root / WORKFLOWS).glob("*.y*ml")):
        build = Build(root, path.relative_to(root))
        if not build.publishes():
            continue
        publishers += 1
        out.extend(build.findings)
        if build.document is None:
            continue
        name = path.name
        if not build.checks_size():
            out.append(
                f"{name}: publishes to PyPI but its build never runs `{SIZE_CHECK}` — a wheel "
                "over PyPI's per-file limit would first fail at the tag's upload"
            )
        on_pr, paths = pull_request_trigger(build.document)
        if not on_pr:
            out.append(
                f"{name}: publishes to PyPI but has no `pull_request` trigger — its wheel is "
                "first built when a tag is pushed"
            )
        elif build.compiles() and paths is not None:
            missing = [entry for entry in COMPILED_INPUTS if entry not in paths]
            if missing:
                out.append(
                    f"{name}: compiles the workspace but its `pull_request` `paths:` leave out "
                    f"{', '.join(missing)} — a pull request changing those would not rebuild "
                    "the wheel it changes"
                )
    if publishers == 0:
        out.append(f"no workflow under {WORKFLOWS} publishes to PyPI — nothing to check")
    return out


def _tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp())
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text))
    return root


def _self_test() -> int:
    paths = "\n".join(f'              - "{entry}"' for entry in COMPILED_INPUTS)
    publish = """
          publish:
            runs-on: ubuntu-latest
            steps:
              - uses: pypa/gh-action-pypi-publish@v1
    """
    good = f"""
        on:
          push:
            tags: ["py-v*"]
          pull_request:
            paths:
{paths}
              - "packaging/**"
        jobs:
          build:
            runs-on: ubuntu-latest
            steps:
              - run: bash ci/scripts/build.sh
    {publish}"""
    script = "#!/usr/bin/env bash\nmaturin build --release\nbash ci/scripts/assert_wheel_size.sh dist/*.whl\n"
    check = "#!/usr/bin/env bash\n"

    def tree(workflow: str, build: str = script) -> Path:
        return _tree(
            {
                ".github/workflows/wheel.yml": workflow,
                "ci/scripts/build.sh": build,
                "ci/scripts/assert_wheel_size.sh": check,
            }
        )

    cases = [
        ("a gated compiled wheel passes", tree(good), []),
        (
            "a size check named only in a comment does not count",
            tree(good, "#!/usr/bin/env bash\nmaturin build --release\n# bash ci/scripts/assert_wheel_size.sh\n"),
            ["never runs"],
        ),
        (
            "a compiled input left out of paths is named",
            tree(good.replace('              - "Cargo.lock"\n', "")),
            ["leave out Cargo.lock"],
        ),
        (
            "a publisher with no pull_request trigger is refused",
            tree(good.replace("  pull_request:\n", "  workflow_dispatch:\n")),
            ["no `pull_request` trigger"],
        ),
        (
            "an unfiltered pull_request trigger covers every input",
            tree(good.split("            paths:")[0] + "        jobs:" + good.split("        jobs:")[1]),
            [],
        ),
        (
            "a wheel that compiles nothing needs the trigger and the check, not the inputs",
            tree(
                good.replace(paths + "\n", ""),
                "#!/usr/bin/env bash\npython -m build --wheel\nbash ci/scripts/assert_wheel_size.sh dist/*.whl\n",
            ),
            [],
        ),
        (
            "a script the build names but the tree lacks is a finding",
            _tree({".github/workflows/wheel.yml": good}),
            ["does not exist", "never runs"],
        ),
        (
            "a tree with no publisher is a finding, not a pass",
            _tree({".github/workflows/ci.yml": "on: pull_request\njobs: {}\n"}),
            ["nothing to check"],
        ),
    ]
    failures = 0
    for name, root, expected in cases:
        got = findings(root)
        ok = len(got) == len(expected) and all(
            any(fragment in finding for finding in got) for fragment in expected
        )
        print(f"self-test[{name}]: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            print(f"  expected findings containing {expected}, got {got}", file=sys.stderr)
    if failures:
        print(f"wheel-gates --self-test: {failures} case(s) FAILED", file=sys.stderr)
        return 1
    print(f"wheel-gates --self-test: all {len(cases)} case(s) passed.")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    found = findings(REPO_ROOT)
    for finding in found:
        print(f"::error::{finding}", file=sys.stderr)
    if found:
        return 1
    print("wheel gates: every PyPI publisher builds and checks its wheel on the pull request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
