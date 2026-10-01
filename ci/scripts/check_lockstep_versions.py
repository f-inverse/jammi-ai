#!/usr/bin/env python3
"""Every dist ships at the workspace version, and every lockstep pin names it.

The Rust crates inherit `[workspace.package] version`; the Python and npm dists
each state it in their own manifest, and some pin a sibling at it exactly
(`jammi-ai[embedded]` → `jammi-ai-native==X`). The cookbook is no published
dist — a notebook installs it from the release's tag — but it carries the same
version and pins `jammi-ai==X`. The changelog's newest release section names
the version the workspace ships, and every version the guide or a README
states (a dependency on a jammi crate, a `"version"` field in an example
response) is it. A release bump that misses one publishes a dist
whose pin cannot resolve, a notebook whose install line names a release that
never shipped, or a release whose changes the changelog files under another
version. This lists
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


def _latest_release(changelog: str) -> str | None:
    """The version of the changelog's first `## [X.Y.Z]` section, if any."""
    m = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
    return m.group(1) if m else None


# A version a document states: a Cargo dependency on a jammi crate, or a
# `"version"` field in an example response. The docs show only Jammi's own
# responses (health, server info); one that shows another system's would have
# to say so in a way this pattern does not read as Jammi's.
_DOC_VERSION = re.compile(
    r'^jammi-[\w-]+ = (?:\{ *version = )?"([^"]+)"|"version": ?"([^"]+)"', re.MULTILINE
)


def _documented() -> dict[str, str]:
    """Each version the guide and the READMEs state, by `file:line`."""
    docs = [*ROOT.glob("docs/guide/src/**/*.md"), ROOT / "README.md", *ROOT.glob("crates/*/README.md")]
    return {
        f"{doc.relative_to(ROOT)}:{text.count(chr(10), 0, m.start()) + 1}": m.group(1) or m.group(2)
        for doc in docs
        for text in [doc.read_text()]
        for m in _DOC_VERSION.finditer(text)
    }


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
        "clients/typescript/package-lock.json version":
            json.loads((ROOT / "clients/typescript/package-lock.json").read_text())["version"],
        "clients/typescript/package-lock.json root package version":
            json.loads((ROOT / "clients/typescript/package-lock.json").read_text())["packages"][""]["version"],
        "CHANGELOG.md newest release section": _latest_release((ROOT / "CHANGELOG.md").read_text()),
        **_documented(),
    }


def main() -> int:
    workspace = _toml("Cargo.toml")["workspace"]["package"]["version"]
    wrong = {site: found for site, found in sites().items() if found != workspace}
    if wrong:
        print(f"lockstep versions: FAIL — the workspace is at {workspace}:", file=sys.stderr)
        for site, found in wrong.items():
            print(f"  {site}: {found or 'none'}", file=sys.stderr)
        return 1
    print(f"lockstep versions: every site is at {workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
