#!/usr/bin/env python3
"""Assert the OSS engine workspace depends on no proprietary code.

The open-core boundary is one-way: a consumer may depend on Jammi; Jammi
depends on no consumer. Every crate resolved into the shippable OSS surface is
either a local workspace path or comes from the public crates.io registry —
nothing from an enterprise namespace, a git source, or a private registry is
allowed to be linked into the engine.

Scope is the NORMAL-dependency closure of the workspace default-members (the
shippable OSS crates). jammi-python (a PyO3 cdylib built via maturin, not
`cargo build --workspace`) is excluded by construction because it is not a
default-member; dev- and build-dependencies are excluded by walking only normal
edges, because a git-sourced dev tool is not proprietary runtime linkage and
must not produce a false positive. If runtime-vs-dev coverage ever needs to
widen, the closure walk below is the single place to adjust.

Two independent filters run over that closure:
  - SOURCE filter (load-bearing, prefix-independent): any crate with a non-null
    source that is not the crates.io registry is an offender. This catches any
    git or private-registry crate regardless of its name.
  - NAME filter (defense-in-depth): any crate whose name starts with
    `jammi-enterprise` is an offender, so a proprietary crate is caught even if
    it were somehow published to crates.io.

Input is `cargo metadata --format-version 1` JSON on stdin. Any offender prints
a `::error::` annotation naming the crate and its source, and the script exits
non-zero. A clean closure prints a one-line OK with the closure size and exits 0.
"""

import json
import sys

CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
ENTERPRISE_PREFIX = "jammi-enterprise"


def main() -> int:
    metadata = json.load(sys.stdin)

    packages_by_id = {pkg["id"]: pkg for pkg in metadata["packages"]}
    resolve_nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    default_members = metadata["workspace_default_members"]

    # BFS the normal-dependency closure. A `deps` entry contributes a normal
    # edge when any of its dep_kinds has kind == null (cargo metadata encodes
    # the normal kind as a null `kind`); dev (kind == "dev") and build
    # (kind == "build") edges are not followed.
    closure: set[str] = set()
    work = list(default_members)
    while work:
        node_id = work.pop()
        if node_id in closure:
            continue
        closure.add(node_id)
        for dep in resolve_nodes[node_id]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                if dep["pkg"] not in closure:
                    work.append(dep["pkg"])

    name_offenders = []
    source_offenders = []
    for pkg_id in closure:
        pkg = packages_by_id[pkg_id]
        name = pkg["name"]
        source = pkg["source"]  # null/None for local path crates
        if name.startswith(ENTERPRISE_PREFIX):
            name_offenders.append((name, source))
        if source is not None and source != CRATES_IO_SOURCE:
            source_offenders.append((name, source))

    for name, source in name_offenders:
        print(
            f"::error::OSS engine dep closure contains enterprise-namespaced "
            f"crate '{name}' (source: {source}) — Jammi must depend on no consumer"
        )
    for name, source in source_offenders:
        print(
            f"::error::OSS engine dep closure contains non-crates.io crate "
            f"'{name}' (source: {source}) — the OSS edition links only local "
            f"workspace paths and crates.io"
        )

    if name_offenders or source_offenders:
        return 1

    print(
        f"OK: {len(closure)} crates in the OSS default-members normal-dep "
        f"closure; all are local workspace paths or crates.io"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
