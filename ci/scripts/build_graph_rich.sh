#!/usr/bin/env bash
# Build the RICH build-graph (Layer 2) on demand and verify it is actually rich.
#
# Why this wrapper exists: `cargo build-graph build --rich --references` SILENTLY
# degrades to the Layer 1 crate/file graph and STILL EXITS 0 when the nightly
# toolchain or rust-analyzer is missing — it just logs "rich layer: skipped".
# Exit 0 is therefore NOT proof of a rich build. This script post-verifies the
# emitted graph and FAILS LOUD if it degraded.
#
# Output path is target/build-graph-rich/ — DISTINCT from target/build-graph/,
# which the dep-dag automation owns. Both live under target/ (gitignored).
#
# Usage: ci/scripts/build_graph_rich.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="target/build-graph-rich"
GRAPH_JSON="${OUT_DIR}/graph.json"

cd "${REPO_ROOT}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# --- Preconditions: build-graph + nightly + rust-analyzer ---------------------
command -v cargo-build-graph >/dev/null 2>&1 \
  || fail "cargo-build-graph not found. Install the pinned version: \
cargo install build-graph --version 0.1.0 --locked"

rustup toolchain list 2>/dev/null | grep -q '^nightly' \
  || fail "no nightly toolchain. The rich (Layer 2) graph needs nightly rustdoc: \
rustup toolchain install nightly"

rustup component list --toolchain nightly 2>/dev/null \
  | grep -q '^rust-analyzer.*(installed)' \
  || fail "rust-analyzer is not installed on the nightly toolchain (needed for \
--references reference edges): rustup component add rust-analyzer --toolchain nightly"

# --- Build (S3: distinct --out; --no-compress emits plain graph.json) ---------
echo "[build_graph_rich] building rich graph -> ${OUT_DIR} (this can take ~3-4 min cold) ..."
cargo build-graph build --rich --references --no-compress --out "${OUT_DIR}"

[ -f "${GRAPH_JSON}" ] || fail "build-graph reported success but ${GRAPH_JSON} is absent."

# --- F1 post-verification: the graph must actually be RICH --------------------
# A degraded Layer 1 graph contains only kind crate/file nodes and only
# depends_on/contains edges. A rich graph adds item-level nodes (struct, method,
# trait, ...) and semantic edges (calls, implements, ...). Require representative
# node kinds AND edge relations from BOTH the rich item layer (--rich) and the
# reference layer (--references); their absence means we silently got Layer 1.
python3 - "${GRAPH_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as fh:
    graph = json.load(fh)

nodes = graph.get("nodes", [])
edges = graph.get("edges", [])
node_kinds = {(n.get("attributes") or {}).get("kind") for n in nodes}
edge_relations = {e.get("relation") for e in edges}

# Rich indicators: item-level kinds from --rich, semantic edges from
# --rich (calls/implements) and --references (member_calls/member_uses).
required_kinds = {"struct", "method", "trait"}
required_relations = {"calls", "implements", "member_calls"}

missing_kinds = sorted(required_kinds - node_kinds)
missing_relations = sorted(required_relations - edge_relations)

if missing_kinds or missing_relations:
    print(
        "rich build degraded to Layer 1 — need nightly + rust-analyzer.\n"
        f"  {len(nodes)} nodes / {len(edges)} edges in {path}\n"
        f"  node kinds present: {sorted(k for k in node_kinds if k)}\n"
        f"  edge relations present: {sorted(r for r in edge_relations if r)}\n"
        f"  MISSING rich node kinds: {missing_kinds}\n"
        f"  MISSING rich edge relations: {missing_relations}\n"
        "The rich item layer (--rich) and/or reference edges (--references) were "
        "skipped. Confirm: rustup toolchain install nightly; rustup component add "
        "rust-analyzer --toolchain nightly.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"[build_graph_rich] VERIFIED rich: {len(nodes)} nodes / {len(edges)} edges\n"
    f"  node kinds: {sorted(k for k in node_kinds if k)}\n"
    f"  edge relations: {sorted(r for r in edge_relations if r)}"
)
PY

echo "[build_graph_rich] OK -> ${REPO_ROOT}/${GRAPH_JSON}"
echo "[build_graph_rich] serve it:  python -m graphify.serve ${GRAPH_JSON}"
