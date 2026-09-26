#!/usr/bin/env python3
"""Every dist ships at the workspace version, and every lockstep pin names it.

The Rust crates inherit `[workspace.package] version`; the Python and npm dists
each state it in their own manifest, and some pin a sibling at it exactly
(`jammi-ai[embedded]` → `jammi-ai-native==X`). The cookbook is no published
dist — a notebook installs it from the release's tag — but it carries the same
version and pins `jammi-ai==X`. A release bump that misses one publishes a dist
whose pin cannot resolve, or a notebook whose install line names a release that
never shipped. This lists
every such site once and fails on any that disagrees with `Cargo.toml`.

Run: `python3 ci/scripts/check_lockstep_versions.py`
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text())


def _pin(requirements: list[str], name: str) -> str | None:
    """The exact version ``requirements`` pins ``name`` at, if any."""
    for req in requirements:
        m = re.fullmatch(rf"{re.escape(name)}==([^\s;]+)", req.strip())
        if m:
            return m.group(1)
    return None


def sites() -> dict[str, str | None]:
    """Each lockstep site and the version it states."""
    client = _toml("clients/python/pyproject.toml")["project"]
    cookbook = _toml("cookbook/book/pyproject.toml")["project"]
    return {
        "clients/python/pyproject.toml version": client["version"],
        "clients/python/pyproject.toml [embedded] jammi-ai-native pin":
            _pin(client["optional-dependencies"]["embedded"], "jammi-ai-native"),
        "cookbook/book/pyproject.toml version": cookbook["version"],
        "cookbook/book/pyproject.toml jammi-ai pin": _pin(cookbook["dependencies"], "jammi-ai"),
        "packaging/server-cpu/pyproject.toml version":
            _toml("packaging/server-cpu/pyproject.toml")["project"]["version"],
        "packaging/server-cu12/pyproject.toml version":
            _toml("packaging/server-cu12/pyproject.toml")["project"]["version"],
        "clients/typescript/package.json version":
            json.loads((ROOT / "clients/typescript/package.json").read_text())["version"],
    }


def main() -> int:
    workspace = _toml("Cargo.toml")["workspace"]["package"]["version"]
    wrong = {site: found for site, found in sites().items() if found != workspace}
    if wrong:
        print(f"lockstep versions: FAIL — the workspace is at {workspace}:", file=sys.stderr)
        for site, found in wrong.items():
            print(f"  {site}: {found or 'no exact pin'}", file=sys.stderr)
        return 1
    print(f"lockstep versions: every dist and pin is at {workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
