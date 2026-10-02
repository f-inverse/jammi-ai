#!/usr/bin/env python3
"""Whether a release is installable from every place a reader installs it from.

A published notebook installs a release the way a reader does: its setup cell
installs the Python dists from PyPI, a chapter's Rust program builds against
the crates on crates.io, its TypeScript program against the client on npm, and
its CLI comes from the GitHub release. The publishers finish at different
times, and a registry serves a version some minutes after it accepted it, so a
reader lane started too early fails notebook by notebook on installs that
would have worked an hour later. The lanes run this first, and it refuses,
naming what is missing.

Each place is asked through the view its installer reads:

  * PyPI: pip's simple index (PEP 691 JSON), for every dist whose manifest is
    `clients/python/pyproject.toml` or `packaging/*/pyproject.toml`;
  * crates.io: cargo's sparse index, for every crate `Cargo.toml`'s
    `[workspace.dependencies]` names with both a `version` and a `path` (the
    workspace crates that publish);
  * npm: the registry document, for `clients/typescript/package.json`'s name;
  * GitHub: the `v<version>` release's assets, for the Linux x86_64 CLI tarball
    a notebook downloads (`jammi_cookbook.programs` names it the same way).

    release_installable.py --tag py-vX.Y.Z

Prints one line per artifact and exits 1 when any is missing. It checks once
and never waits: dispatch the lane again once every publisher has finished.
`GITHUB_TOKEN`, when set, authenticates the GitHub request.

Self-test: `python3 ci/scripts/release_installable.py --self-test`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_REPO = "f-inverse/jammi-ai"
READER_CLI_TRIPLE = "x86_64-unknown-linux-gnu"

# url, headers -> body, or None when the resource does not exist (HTTP 404).
Fetch = Callable[[str, dict[str, str]], "bytes | None"]


def default_fetch(url: str, headers: dict[str, str]) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "jammi-release-check", **headers})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def python_dists(root: Path) -> list[str]:
    manifests = [root / "clients" / "python" / "pyproject.toml",
                 *sorted((root / "packaging").glob("*/pyproject.toml"))]
    return [tomllib.loads(m.read_text())["project"]["name"] for m in manifests]


def published_crates(root: Path) -> list[str]:
    deps = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["dependencies"]
    return sorted(
        name for name, spec in deps.items()
        if isinstance(spec, dict) and "path" in spec and "version" in spec
    )


def npm_package(root: Path) -> str:
    return json.loads((root / "clients" / "typescript" / "package.json").read_text())["name"]


def sparse_index_path(crate: str) -> str:
    """Cargo's sparse-index path for `crate`."""
    name = crate.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def check(version: str, *, root: Path, fetch: Fetch, token: str | None) -> list[tuple[str, bool]]:
    """`(artifact, served)` for everything a reader installs at `version`."""
    results: list[tuple[str, bool]] = []

    for dist in python_dists(root):
        body = fetch(f"https://pypi.org/simple/{dist}/",
                     {"Accept": "application/vnd.pypi.simple.v1+json"})
        served = body is not None and version in json.loads(body).get("versions", [])
        results.append((f"PyPI {dist}=={version}", served))

    for crate in published_crates(root):
        body = fetch(f"https://index.crates.io/{sparse_index_path(crate)}", {})
        versions = [json.loads(line)["vers"] for line in (body or b"").decode().splitlines() if line]
        results.append((f"crates.io {crate} {version}", version in versions))

    package = npm_package(root)
    body = fetch(f"https://registry.npmjs.org/{package}", {})
    served = body is not None and version in json.loads(body).get("versions", {})
    results.append((f"npm {package}@{version}", served))

    asset = f"jammi-{version}-{READER_CLI_TRIPLE}.tar.gz"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = fetch(f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/v{version}", headers)
    assets = [a["name"] for a in json.loads(body).get("assets", [])] if body else []
    results.append((f"GitHub release v{version} {asset}", asset in assets))

    return results


def version_of(tag: str) -> str:
    match = re.fullmatch(r"py-v(\d+\.\d+\.\d+)", tag)
    if match is None:
        raise SystemExit(f"release_installable: '{tag}' is not a py-vX.Y.Z release tag")
    return match.group(1)


def run(tag: str, *, root: Path, fetch: Fetch, token: str | None, out=sys.stdout) -> int:
    results = check(version_of(tag), root=root, fetch=fetch, token=token)
    for artifact, served in results:
        print(f"{'served ' if served else 'MISSING'}  {artifact}", file=out)
    missing = [a for a, served in results if not served]
    if missing:
        print(f"::error::{tag} is not installable as published: {len(missing)} of {len(results)} "
              "artifacts are not served. If a publisher is still running, run the lane again once "
              "it has finished.", file=out)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Self-test: the real manifests, fixture registries.
# --------------------------------------------------------------------------- #


def _fixture_fetch(version: str, drop: str | None = None) -> Fetch:
    """Registries serving `version` for everything, except the one artifact
    whose URL contains `drop`."""

    def fetch(url: str, headers: dict[str, str]) -> bytes | None:
        if drop and drop in url:
            if "api.github.com" in url:
                return None
            if "index.crates.io" in url:
                return json.dumps({"vers": "0.0.1"}).encode()
            return json.dumps({"versions": ["0.0.1"] if "pypi" in url else {"0.0.1": {}}}).encode()
        if "pypi.org" in url:
            assert headers.get("Accept") == "application/vnd.pypi.simple.v1+json", url
            return json.dumps({"versions": ["0.0.1", version]}).encode()
        if "index.crates.io" in url:
            return "\n".join(json.dumps({"vers": v}) for v in ("0.0.1", version)).encode()
        if "registry.npmjs.org" in url:
            return json.dumps({"versions": {"0.0.1": {}, version: {}}}).encode()
        if "api.github.com" in url:
            return json.dumps({"assets": [{"name": f"jammi-{version}-{READER_CLI_TRIPLE}.tar.gz"}]}).encode()
        raise AssertionError(f"unexpected url {url}")

    return fetch


def _self_test() -> int:
    import io

    failures: list[str] = []

    def expect(name: str, cond: bool, detail: str = "") -> None:
        print(f"self-test[{name}]: {'ok' if cond else 'FAIL'}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    expect("sparse-path-long", sparse_index_path("jammi-ai") == "ja/mm/jammi-ai")
    expect("sparse-path-three", sparse_index_path("abc") == "3/a/abc")
    expect("sparse-path-short", sparse_index_path("ab") == "2/ab")
    expect("the-real-manifests-name-dists-crates-and-the-client",
           "jammi-ai" in python_dists(REPO_ROOT)
           and {"jammi-ai", "jammi-db"} <= set(published_crates(REPO_ROOT))
           and "jammi-test-utils" not in published_crates(REPO_ROOT)
           and npm_package(REPO_ROOT).endswith("jammi-client"))

    out = io.StringIO()
    rc = run("py-v9.9.9", root=REPO_ROOT, fetch=_fixture_fetch("9.9.9"), token=None, out=out)
    expect("a-fully-published-release-is-installable", rc == 0, out.getvalue())

    for drop, needle in (("pypi.org/simple/jammi-ai-native/", "PyPI jammi-ai-native==9.9.9"),
                         ("index.crates.io/ja/mm/jammi-db", "crates.io jammi-db 9.9.9"),
                         ("registry.npmjs.org", "npm "),
                         ("api.github.com", "GitHub release v9.9.9")):
        out = io.StringIO()
        rc = run("py-v9.9.9", root=REPO_ROOT, fetch=_fixture_fetch("9.9.9", drop), token=None, out=out)
        missing = [line for line in out.getvalue().splitlines() if line.startswith("MISSING")]
        expect(f"a-release-missing-from-{drop.split('/')[0]}-is-refused-by-name",
               rc == 1 and len(missing) == 1 and needle in missing[0], out.getvalue())

    try:
        version_of("v9.9.9")
        expect("a-non-py-tag-is-refused", False, "accepted")
    except SystemExit:
        expect("a-non-py-tag-is-refused", True)

    if failures:
        print(f"self-test: FAIL {failures}", file=sys.stderr)
        return 1
    print("self-test: all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", help="the release tag, py-vX.Y.Z")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.tag:
        parser.error("--tag is required")
    return run(args.tag, root=REPO_ROOT, fetch=default_fetch, token=os.environ.get("GITHUB_TOKEN"))


if __name__ == "__main__":
    sys.exit(main())
