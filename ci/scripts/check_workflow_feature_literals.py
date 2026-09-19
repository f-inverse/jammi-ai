#!/usr/bin/env python3
"""No release workflow spells a `jammi-server` cargo feature; the manifest does.

**Guarded property**: every workflow that builds a shipped `jammi-server`
artifact takes its cargo feature list from `ci/release-feature-manifest.json`
and from nowhere else. A feature name typed into a workflow — a `build-args`
line, a `--features=…` inside a `run:` string, an input default, a matrix
value — is a second source that silently diverges from the manifest.

**Universe**: the workflow files the manifest's own lanes name, plus every
local reusable workflow and composite action they reach through `uses: ./…`,
transitively. Test lanes elsewhere legitimately name test features and are
outside it.

**Decided by value, over the whole parsed document**: each file is read with a
real YAML parser and EVERY string scalar in it is examined, wherever it sits.
A scalar carries a feature literal when a `jammi-server` feature name (from
the crate's own `[features]` table) appears in a feature-list position:

  * after `--features` / `-F` (`=` or whitespace separated);
  * as the value of an assignment whose name contains `FEATURES`
    (`CARGO_FEATURES=cuda,…` in a `build-args` block or a shell line);
  * under a mapping key whose name contains `features`;
  * as a scalar that is nothing but a list of feature names, anywhere the
    workflow schema lets data flow into a build: under `env`, `with`,
    `matrix`, an input `default`, or `outputs` — never a label such as a step
    `id` or a job `name`.

Prose that mentions a feature, an expression such as
`${{ steps.cuda.outputs.image }}`, and a list made of shell or workflow
variables are none of these and pass. The manifest-read idiom
(`jq -r '.lanes[…].cargo_features | join(",")'`) names no feature at all.

Fail-closed: a lane naming a workflow that does not exist, a `uses: ./…` that
resolves to nothing, an unparseable document, or an empty universe is a
finding, never a silent pass.

Run: `python3 ci/scripts/check_workflow_feature_literals.py [--self-test]`
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ci" / "release-feature-manifest.json"
SERVER_CARGO = REPO_ROOT / "crates" / "jammi-server" / "Cargo.toml"

_WORKFLOW_REF = re.compile(r"\.github/workflows/[\w.-]+\.ya?ml")
# Mapping keys under which the workflow schema carries data, as opposed to
# labels (`id`, `name`) and control (`if`, `needs`, `uses`).
_DATA_KEYS = {"env", "with", "matrix", "default", "outputs"}
_FLAG = re.compile(r"(?:--features|(?<![\w-])-F)(?:=|\s+)(\"[^\"]*\"|'[^']*'|\S+)")
_ASSIGNMENT = re.compile(r"(?<![\w$])\w*FEATURES\w*\s*[=:]\s*(\"[^\"]*\"|'[^']*'|\S+)")


def server_features() -> set[str]:
    features = tomllib.loads(SERVER_CARGO.read_text())["features"]
    return {name for name in features if name != "default"}


def tokens(feature_list: str) -> list[str]:
    return [t for t in re.split(r"[\s,\"']+", feature_list) if t]


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


def literal_features(path: tuple[str, ...], value: str, features: set[str]) -> set[str]:
    """The feature names `value` spells in a feature-list position."""
    candidates = [m.group(1) for m in _FLAG.finditer(value)]
    candidates += [m.group(1) for m in _ASSIGNMENT.finditer(value)]
    key = next((segment for segment in reversed(path) if not segment.isdigit()), "")
    if "features" in key.lower():
        candidates.append(value)
    whole = tokens(value)
    if _DATA_KEYS & set(path) and whole and all(token in features for token in whole):
        candidates.append(value)
    return {token for candidate in candidates for token in tokens(candidate)} & features


def local_uses(document) -> set[str]:
    """Repo-relative targets of every `uses: ./…` in a document."""
    return {
        value[2:].split("@")[0]
        for path, value in scalars(document)
        if path and path[-1] == "uses" and value.startswith("./")
    }


def resolve_use(target: str) -> Path | None:
    base = REPO_ROOT / target
    if base.is_file():
        return base
    for name in ("action.yml", "action.yaml"):
        if (base / name).is_file():
            return base / name
    return None


def universe() -> tuple[list[Path], list[str]]:
    """The release workflows and everything local they reach, plus findings
    for anything named that does not resolve."""
    findings: list[str] = []
    lanes = json.loads(MANIFEST.read_text())["lanes"]
    pending = sorted(
        {ref for lane in lanes.values() for ref in _WORKFLOW_REF.findall(lane["workflow"])}
    )
    if not pending:
        findings.append(f"{MANIFEST.name}: no lane names a workflow — nothing to check")
    seen: dict[Path, None] = {}
    while pending:
        rel = pending.pop()
        path = resolve_use(rel)
        if path is None:
            findings.append(f"`{rel}` is named by the manifest or a `uses:` but does not exist")
            continue
        if path in seen:
            continue
        seen[path] = None
        try:
            document = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            findings.append(f"{path.relative_to(REPO_ROOT)}: unparseable YAML: {error}")
            continue
        pending.extend(sorted(local_uses(document)))
    return list(seen), findings


def check_document(rel: str, document, features: set[str]) -> list[str]:
    return [
        f"{rel}: `{'/'.join(path)}` spells jammi-server feature(s) "
        f"{sorted(found)} in {value.strip()[:160]!r} — read the lane's list from "
        f"{MANIFEST.relative_to(REPO_ROOT)} instead"
        for path, value in scalars(document)
        if (found := literal_features(path, value, features))
    ]


def check() -> list[str]:
    features = server_features()
    files, findings = universe()
    for path in files:
        document = yaml.safe_load(path.read_text())
        findings += check_document(str(path.relative_to(REPO_ROOT)), document, features)
    return findings


def self_test() -> int:
    features = {"cuda", "flash-attn", "storage-cloud", "jetstream-broker"}
    failures: list[str] = []

    def expect(label: str, text: str, want: set[str]) -> None:
        got: set[str] = set()
        for path, value in scalars(yaml.safe_load(textwrap.dedent(text))):
            got |= literal_features(path, value, features)
        if got != want:
            failures.append(f"{label}: expected {sorted(want)}, got {sorted(got)}")

    expect(
        "a build-args assignment is refused",
        """
        jobs:
          b:
            steps:
              - with:
                  build-args: |
                    BASE=x
                    CARGO_FEATURES=cuda,flash-attn
        """,
        {"cuda", "flash-attn"},
    )
    expect(
        "a --features= inside a run string is refused",
        """
        jobs:
          b:
            steps:
              - run: cargo build -p jammi-server --features=storage-cloud
        """,
        {"storage-cloud"},
    )
    expect(
        "a space-separated, quoted -F list is refused",
        """
        jobs:
          b:
            steps:
              - run: cargo build -F 'cuda jetstream-broker'
        """,
        {"cuda", "jetstream-broker"},
    )
    expect(
        "an input default that is nothing but features is refused",
        """
        on:
          workflow_call:
            inputs:
              flavour:
                default: cuda,flash-attn
        """,
        {"cuda", "flash-attn"},
    )
    expect(
        "a value under a features key is refused",
        """
        jobs:
          b:
            strategy:
              matrix:
                cargo_features: ["cuda extra"]
        """,
        {"cuda"},
    )
    expect(
        "the manifest-read idiom passes",
        """
        jobs:
          b:
            steps:
              - run: |
                  features="$(jq -r '.lanes["cu12-wheel"].cargo_features | join(",")' m.json)"
                  cargo build -p jammi-server --features "$features"
        """,
        set(),
    )
    expect(
        "labels, prose and expressions that mention a feature pass",
        """
        jobs:
          b:
            name: Build the cuda image (flash-attn kernels)
            outputs:
              image_cuda: ${{ steps.cuda.outputs.image }}
            steps:
              - id: cuda
        """,
        set(),
    )
    expect(
        "a workflow-expression feature list passes",
        """
        jobs:
          b:
            steps:
              - with:
                  build-args: |
                    CARGO_FEATURES=${{ steps.features.outputs.list }}
        """,
        set(),
    )

    files, findings = universe()
    if findings:
        failures += findings
    lanes = json.loads(MANIFEST.read_text())["lanes"]
    named = {ref for lane in lanes.values() for ref in _WORKFLOW_REF.findall(lane["workflow"])}
    reached = {str(path.relative_to(REPO_ROOT)) for path in files}
    if not named <= reached:
        failures.append(f"universe misses manifest-named workflows: {sorted(named - reached)}")
    if not any("/actions/" in rel or "/_" in rel for rel in reached):
        failures.append("universe reached no reusable workflow or composite action")

    if failures:
        print("workflow-feature-literals self-test: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"workflow-feature-literals self-test: OK ({len(reached)} documents in the universe)")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    findings = check()
    if findings:
        print("workflow-feature-literals: FAIL", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print("workflow-feature-literals: OK — no release workflow spells a jammi-server feature")
    return 0


if __name__ == "__main__":
    sys.exit(main())
