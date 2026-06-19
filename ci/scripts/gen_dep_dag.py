#!/usr/bin/env python3
"""Generate the intra-workspace crate dependency DAG into the maintainer guide.

build-graph is the maintainer-chosen source of truth for the crate graph: it is
extracted directly from `cargo build` artifacts, not from `cargo metadata`. This
script runs the build-graph CLI, reads the resolved graph, keeps only the
intra-workspace crate `depends_on` edges, and renders a deterministic
crate-name-only block spliced strictly between the seeded markers in
`docs/maintainer/MAINTAINER-GUIDE.md`.

Determinism is load-bearing: this is wired as a CI freshness gate. The
generator must be idempotent — running it twice leaves the doc byte-identical —
so the gate (`git diff --exit-code` on the guide) only reds a PR when the crate
graph actually drifted from the committed block.

The block is crate NAMES only: file nodes, node ids, paths, and versions are all
dropped. Crate names sort the node lines; each node's dependency list is sorted
too. A crate with no intra-workspace dependencies renders as a bare name.

Run: `python3 ci/scripts/gen_dep_dag.py`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_JSON = REPO_ROOT / "target" / "build-graph" / "graph.json"
GUIDE = REPO_ROOT / "docs" / "maintainer" / "MAINTAINER-GUIDE.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED: dep-dag -->"
END_MARKER = "<!-- END GENERATED: dep-dag -->"


def refresh_graph() -> None:
    """Refresh `target/build-graph/graph.json` from the current workspace.

    If the plain (uncompressed) graph already exists, `update` re-extracts it
    from the present `target/` without re-running `cargo build`; otherwise a full
    `build` produces it. Both emit the uncompressed `graph.json` (`--no-compress`)
    so the reader below can parse it directly.
    """
    subcommand = "update" if GRAPH_JSON.exists() else "build"
    subprocess.run(
        ["cargo", "build-graph", subcommand, "--no-compress"],
        cwd=REPO_ROOT,
        check=True,
    )


def render_block() -> str:
    """Render the deterministic crate-dependency block from the build graph.

    Keeps only intra-workspace crate→crate `depends_on` edges (both endpoints
    are crate nodes), maps node ids to their hyphenated crate names, and renders
    one name-sorted line per crate with a name-sorted dependency list. Crates
    with no intra-workspace dependency render as a bare name.
    """
    graph = json.loads(GRAPH_JSON.read_text())
    nodes = {node["id"]: node for node in graph["nodes"]}
    crate_name = {
        node_id: node["attributes"]["crate"]
        for node_id, node in nodes.items()
        if node["attributes"].get("kind") == "crate"
    }

    deps: dict[str, set[str]] = {name: set() for name in crate_name.values()}
    for edge in graph["edges"]:
        if edge["relation"] != "depends_on":
            continue
        src, dst = edge["source"], edge["target"]
        if src in crate_name and dst in crate_name:
            deps[crate_name[src]].add(crate_name[dst])

    lines = []
    for name in sorted(deps):
        targets = sorted(deps[name])
        if targets:
            lines.append(f"{name} -> {', '.join(targets)}")
        else:
            lines.append(name)
    return "\n".join(lines)


def splice(block: str) -> None:
    """Replace the content strictly between the markers, byte-for-byte preserving
    the marker lines and everything outside them. Idempotent."""
    text = GUIDE.read_text()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        sys.exit(f"markers not found in {GUIDE}")

    before, rest = text.split(BEGIN_MARKER, 1)
    _stale, after = rest.split(END_MARKER, 1)
    new_text = f"{before}{BEGIN_MARKER}\n```\n{block}\n```\n{END_MARKER}{after}"
    GUIDE.write_text(new_text)


def main() -> int:
    refresh_graph()
    splice(render_block())
    print(f"dep-dag block written to {GUIDE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
