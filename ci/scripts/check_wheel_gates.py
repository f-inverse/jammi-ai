#!/usr/bin/env python3
"""Every wheel published to PyPI is built, and held to PyPI's size limit, on
the pull request that changes it — once — and a release publishes that build.

**Guarded property**:

  1. `ci.yml` runs on every pull request with no `paths:` filter, and every
     wheel it builds — every local reusable workflow it calls whose build
     runs `maturin build`, the maturin action, `python -m build` or
     `wheel tags` — runs `ci/scripts/assert_wheel_size.sh`, so a wheel over
     PyPI's per-file limit fails the pull request that grew it;
  2. a workflow that publishes to PyPI publishes what `_proven-artifacts.yml`
     resolves — the artifacts of the `ci.yml` run that proved the released
     tree — and compiles nothing itself, so the bytes published are the ones
     that run built and checked; and
  3. a publisher has no `pull_request` trigger of its own, so no pull request
     builds a wheel twice.

PyPI refuses an oversized file at the tag's upload, after the other wheels of
the same lockstep release are already published; without the first, the
first run to see the size is that upload.

**Universe**: `ci.yml`'s local reusable workflows, and the workflows under
`.github/workflows/` with a `pypa/gh-action-pypi-publish` step. A build is
everything a workflow reaches: the local reusable workflows and composite
actions it uses (`uses: ./…`) and the `ci/scripts/*.sh` its `run:` lines and
those scripts name, transitively.

**Decided over parsed documents**: a workflow is read with a YAML parser; a
command counts where it is executed — a `run:` string, or a script line that
is not a comment — never where prose mentions it.

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
PROVING_WORKFLOW = WORKFLOWS / "ci.yml"
SIZE_CHECK = "ci/scripts/assert_wheel_size.sh"
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"
MATURIN_ACTION = "PyO3/maturin-action"
LOCAL_WORKFLOW = "./.github/workflows/"
PROMOTION = WORKFLOWS / "_proven-artifacts.yml"

_SCRIPT_REF = re.compile(r"ci/scripts/[\w./-]+\.sh")
_BUILDS_WHEEL = re.compile(r"\bmaturin\s+build\b|\bpython3?\s+-m\s+build\b|\bwheel\s+tags\b")
_COMPILES = re.compile(r"\bcargo\s+build\b|\bnpm\s+run\s+build\b")


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


def load(root: Path, path: Path, findings: list[str]):
    try:
        return yaml.safe_load((root / path).read_text())
    except yaml.YAMLError as error:
        findings.append(f"{path}: not parseable as YAML ({error})")
        return None


class Build:
    """What a workflow executes and uses, over everything it reaches."""

    def __init__(self, root: Path, workflow: Path):
        self.commands: list[str] = []
        self.actions: list[str] = []
        self.findings: list[str] = []
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
                document = load(root, path, self.findings)
                if document is None:
                    continue
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

    def builds_wheel(self) -> bool:
        return any(_BUILDS_WHEEL.search(command) for command in self.commands) or any(
            use.startswith(MATURIN_ACTION) for use in self.actions
        )

    def compiles(self) -> bool:
        return self.builds_wheel() or any(_COMPILES.search(command) for command in self.commands)


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


def called_reusables(document) -> set[Path]:
    """The local reusable workflows a workflow's jobs call."""
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        return set()
    return {
        WORKFLOWS / job["uses"][len(LOCAL_WORKFLOW):].split("@")[0]
        for job in jobs.values()
        if isinstance(job, dict) and isinstance(job.get("uses"), str) and job["uses"].startswith(LOCAL_WORKFLOW)
    }


def findings(root: Path) -> list[str]:
    out: list[str] = []
    proving = load(root, PROVING_WORKFLOW, out) if (root / PROVING_WORKFLOW).is_file() else None
    if proving is None:
        out.append(f"{PROVING_WORKFLOW} is missing — no workflow builds the wheels on a pull request")
    else:
        on_pr, paths = pull_request_trigger(proving)
        if not on_pr or paths is not None:
            out.append(
                f"{PROVING_WORKFLOW.name}: must run on every pull request, with no `paths:` "
                "filter — it is where every wheel is built before a tag"
            )
        wheel_builds = 0
        for reusable in sorted(called_reusables(proving)):
            build = Build(root, reusable)
            out.extend(build.findings)
            if not build.builds_wheel():
                continue
            wheel_builds += 1
            if not build.checks_size():
                out.append(
                    f"{reusable.name}: builds a wheel `ci.yml` ships but never runs `{SIZE_CHECK}` — "
                    "a wheel over PyPI's per-file limit would first fail at the tag's upload"
                )
        if wheel_builds == 0:
            out.append(f"{PROVING_WORKFLOW.name} builds no wheel — nothing a release could publish")
    publishers = 0
    for path in sorted((root / WORKFLOWS).glob("*.y*ml")):
        workflow = path.relative_to(root)
        build = Build(root, workflow)
        if not build.publishes():
            continue
        publishers += 1
        out.extend(build.findings)
        document = load(root, workflow, out)
        if document is None:
            continue
        name = path.name
        if PROMOTION not in called_reusables(document):
            out.append(
                f"{name}: publishes to PyPI but not what `{PROMOTION.name}` resolves — a release "
                "publishes the artifacts of the run that proved its tree"
            )
        if build.compiles():
            out.append(
                f"{name}: compiles at release — it must publish the bytes `ci.yml` built and checked, "
                "not a second build"
            )
        if pull_request_trigger(document)[0]:
            out.append(
                f"{name}: has a `pull_request` trigger of its own — `ci.yml` builds its wheel "
                "on every pull request already, so a pull request would build it twice"
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
    publisher = """
        on:
          push:
            tags: ["py-v*"]
        jobs:
          artifacts:
            uses: ./.github/workflows/_proven-artifacts.yml
            with:
              artifacts: wheel
          publish:
            needs: artifacts
            runs-on: ubuntu-latest
            steps:
              - uses: pypa/gh-action-pypi-publish@v1
    """
    promotion = """
        on: workflow_call
        jobs:
          resolve:
            runs-on: ubuntu-latest
            steps:
              - run: python3 ci/scripts/proven_artifacts.py
    """
    wheel = """
        on: workflow_call
        jobs:
          build:
            runs-on: ubuntu-latest
            steps:
              - run: bash ci/scripts/build.sh
    """
    proving = """
        on:
          push:
            branches: [main]
          pull_request:
        jobs:
          wheel:
            uses: ./.github/workflows/_wheel.yml
    """
    script = "#!/usr/bin/env bash\nmaturin build --release\nbash ci/scripts/assert_wheel_size.sh dist/*.whl\n"

    def tree(
        publisher: str = publisher,
        proving: str | None = proving,
        build: str = script,
    ) -> Path:
        files = {
            ".github/workflows/pypi.yml": publisher,
            ".github/workflows/_proven-artifacts.yml": promotion,
            ".github/workflows/_wheel.yml": wheel,
            "ci/scripts/build.sh": build,
            "ci/scripts/assert_wheel_size.sh": "#!/usr/bin/env bash\n",
        }
        if proving is not None:
            files[".github/workflows/ci.yml"] = proving
        return _tree(files)

    cases = [
        ("a wheel ci.yml builds and checks, promoted at release, passes", tree(), []),
        (
            "a size check named only in a comment does not count",
            tree(build="#!/usr/bin/env bash\nmaturin build --release\n# bash ci/scripts/assert_wheel_size.sh\n"),
            ["never runs"],
        ),
        (
            "a publisher that compiles at release is refused",
            tree(
                publisher=publisher.replace(
                    "              - uses: pypa/gh-action-pypi-publish@v1",
                    "              - run: bash ci/scripts/build.sh\n              - uses: pypa/gh-action-pypi-publish@v1",
                )
            ),
            ["compiles at release"],
        ),
        (
            "a publisher that bypasses the proving run is refused",
            tree(publisher=publisher.replace("uses: ./.github/workflows/_proven-artifacts.yml", "uses: ./.github/workflows/_wheel.yml")),
            ["not what `_proven-artifacts.yml` resolves", "compiles at release"],
        ),
        (
            "a ci.yml filtered by paths is refused",
            tree(proving=proving.replace("  pull_request:\n", "  pull_request:\n            paths: [crates/**]\n")),
            ["no `paths:` filter"],
        ),
        (
            "a ci.yml that builds no wheel is a finding",
            tree(proving=proving.replace("uses: ./.github/workflows/_wheel.yml", "runs-on: ubuntu-latest\n            steps: []")),
            ["builds no wheel"],
        ),
        (
            "a publisher that also builds on pull requests is refused",
            tree(publisher=publisher.replace('    tags: ["py-v*"]\n', '    tags: ["py-v*"]\n          pull_request:\n')),
            ["build it twice"],
        ),
        (
            "a tree with no ci.yml is a finding",
            tree(proving=None),
            ["is missing"],
        ),
        (
            "a script the build names but the tree lacks is a finding",
            _tree(
                {
                    ".github/workflows/pypi.yml": publisher,
                    ".github/workflows/_proven-artifacts.yml": promotion,
                    ".github/workflows/_wheel.yml": wheel,
                    ".github/workflows/ci.yml": proving,
                }
            ),
            ["does not exist", "builds no wheel"],
        ),
        (
            "a tree with no publisher is a finding, not a pass",
            _tree(
                {
                    ".github/workflows/ci.yml": proving,
                    ".github/workflows/_wheel.yml": wheel,
                    "ci/scripts/build.sh": script,
                    "ci/scripts/assert_wheel_size.sh": "#!/usr/bin/env bash\n",
                }
            ),
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
    print("wheel gates: every wheel ci.yml builds is size-checked on every pull request, and every PyPI publisher promotes that build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
