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
# --- Pin: nightly-2026-06-07 / build-graph 0.1.0 -------------------------------
# build-graph 0.1.0 (published 2026-06-07) parses that era's rustdoc JSON
# output shape. `--nightly` defaults to auto-detecting the NEWEST installed
# nightly toolchain; a newer nightly (empirically nightly-2026-07-23) silently
# zeroes the rich layer instead of failing outright — the tool still logs
# success, and only the post-verify below catches the degradation. Pinning
# `--nightly` explicitly means a nightly installed on the host for an
# unrelated reason (an experiment, a different tool) can never become "newest"
# and silently degrade this graph out from under us.
#
# This pin has a reciprocal home in .devcontainer/Dockerfile, which bakes
# nightly-2026-06-07 (and the matching rust-analyzer component) into the dev
# image so the toolchain this script names is actually installed there. Bump
# both together, in the same PR — this is the same two-location lockstep
# idiom as `CUDA_COMPUTE_CAP` in `.docker/ci-cuda.Dockerfile` mirrored by
# `MIN_CUDA_COMPUTE_CAP` in `crates/jammi-ai/src/model/backend/candle.rs`:
# each side names its counterpart in a comment so a bump is one reviewable
# diff, never silent drift between the two files.
#
# The dated nightly coexists with rust-toolchain.toml's 1.94.0 stable default;
# it is never the ambient toolchain, only invoked explicitly via `--nightly`.
#
# Usage:
#   ci/scripts/build_graph_rich.sh              full clean rebuild + verify,
#                                                always (today's behavior: the
#                                                OUT_DIR is always wiped before
#                                                the build, never trusted
#                                                incremental), plus now also
#                                                stamps the graph with HEAD
#                                                once verification passes.
#   ci/scripts/build_graph_rich.sh --if-stale    skip the rebuild if the graph
#                                                is already stamped fresh at
#                                                HEAD; otherwise falls through
#                                                to the exact same clean full
#                                                rebuild above.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="target/build-graph-rich"
GRAPH_JSON="${OUT_DIR}/graph.json"
STAMP="${OUT_DIR}/.built-sha"
readonly NIGHTLY="nightly-2026-06-07"
readonly BUILD_GRAPH_VERSION="0.1.0"

cd "${REPO_ROOT}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# --- Argument parsing -----------------------------------------------------
IF_STALE=0
for arg in "$@"; do
  case "${arg}" in
    --if-stale) IF_STALE=1 ;;
    *) fail "unknown argument: '${arg}' (supported: --if-stale)" ;;
  esac
done

# --- --if-stale: cheap freshness check, skip the rebuild if HEAD is unchanged -
# The graph is extracted from the WORKING TREE, not from a commit object, so
# HEAD alone is not the full determinant of freshness. The complete
# determinant set is TWO conditions, both required:
#   1. HEAD-sufficiency for committed state: every toolchain pin this graph
#      depends on (the --nightly pin above, build-graph ${BUILD_GRAPH_VERSION}) lives in a
#      HEAD-tracked file (this script, .devcontainer/Dockerfile) rather than
#      in ambient host state — a pin bump is necessarily a commit, so the SHA
#      moves with it and can never falsely read fresh. Do NOT widen the
#      stamp to also record tool versions: that would let an
#      unpinned/ambient tool change go undetected by a diff review, which is
#      exactly what the pin exists to avoid.
#   2. Clean-tree requirement, closing the working-tree gap HEAD leaves open:
#      a dirty tree at an unchanged HEAD can still shape the extracted graph
#      (uncommitted edits) or leave a graph that no longer matches HEAD after
#      a revert — either way, `git status --porcelain` empty is required
#      alongside the SHA match. `git status --porcelain` (tracked
#      modifications AND untracked non-ignored files) is exactly "does the
#      working tree differ from HEAD" — the same predicate the stamp-writer
#      below re-checks before stamping.
if [ "${IF_STALE}" -eq 1 ]; then
  if [ -f "${GRAPH_JSON}" ] && [ -f "${STAMP}" ]; then
    CURRENT_SHA="$(git rev-parse HEAD)"
    STAMPED_SHA="$(cat "${STAMP}")"
    PORCELAIN="$(git status --porcelain)"
    if [ "${CURRENT_SHA}" = "${STAMPED_SHA}" ] && [ -z "${PORCELAIN}" ]; then
      echo "[build_graph_rich] fresh at ${CURRENT_SHA}"
      exit 0
    fi
  fi
  # Stale, absent, or dirty tree: fall through to the same clean full rebuild
  # the no-arg invocation always does (below) — see the OUT_DIR rationale
  # there. A dirty-tree session therefore rebuilds on every --if-stale call;
  # accepted cost — the doc-currency pipeline runs from clean worktrees /
  # committed states, and correctness beats a several-minute saving during
  # active editing.
fi

# --- Clean before every build, both invocation shapes ------------------------
# build-graph's own incremental cache empirically rewrites a previously
# degraded Layer-1 graph with "no changed crates to re-document" even when
# re-run under the good nightly — it treats the crates as already documented
# from the earlier (degraded) run and skips re-extraction entirely. The
# post-verify below still catches this loudly, but leaving it there alone is
# a retry-trap: an operator fixes the toolchain, re-runs, hits the same
# masked incremental state, and the error message points at toolchains, not
# the stale dir. Only a clean OUT_DIR forces true re-extraction under the
# pinned nightly, so nuke it unconditionally before every build — the
# --if-stale early-exit above is the only path that skips this.
rm -rf "${OUT_DIR}"

# --- Preconditions: build-graph (exact pinned version) + pinned nightly + ----
# --- rust-analyzer on that exact nightly ---------------------------------
command -v cargo-build-graph >/dev/null 2>&1 \
  || fail "cargo-build-graph not found. Install the pinned version: \
cargo install build-graph --version ${BUILD_GRAPH_VERSION} --locked"

# The `|| true` keeps a non-zero --version exit from aborting this assignment
# under `set -e` — without it, a broken/incompatible cargo-build-graph binary
# would kill the script here instead of falling into the `<unknown>` fallback
# and the authored skew message below, defeating the point of the check.
INSTALLED_BUILD_GRAPH_VERSION="$({ cargo-build-graph --version 2>/dev/null || true; } | awk '{print $2}')"
[ "${INSTALLED_BUILD_GRAPH_VERSION}" = "${BUILD_GRAPH_VERSION}" ] \
  || fail "cargo-build-graph is version '${INSTALLED_BUILD_GRAPH_VERSION:-<unknown>}', but this \
graph is pinned to build-graph ${BUILD_GRAPH_VERSION} (it parses ${NIGHTLY}'s rustdoc JSON \
shape specifically — a version skew here is the other half of the toolchain lockstep, and can \
silently mis-parse a different nightly's format). Install the pinned version: \
cargo install build-graph --version ${BUILD_GRAPH_VERSION} --locked"

rustup toolchain list 2>/dev/null | grep -q "^${NIGHTLY}" \
  || fail "pinned nightly toolchain ${NIGHTLY} is not installed (this graph is pinned to it, in \
lockstep with build-graph ${BUILD_GRAPH_VERSION} — see the pin comment at the top of this \
script): rustup toolchain install ${NIGHTLY}"

rustup component list --toolchain "${NIGHTLY}" 2>/dev/null \
  | grep -q '^rust-analyzer.*(installed)' \
  || fail "rust-analyzer is not installed on the pinned ${NIGHTLY} toolchain (needed for \
--references reference edges): rustup component add rust-analyzer --toolchain ${NIGHTLY}"

# --- Build (S3: distinct --out; --no-compress emits plain graph.json) ---------
echo "[build_graph_rich] building rich graph -> ${OUT_DIR} (this can take ~3-4 min cold) ..."
cargo build-graph build --rich --references --no-compress --out "${OUT_DIR}" --nightly "${NIGHTLY}"

[ -f "${GRAPH_JSON}" ] || fail "build-graph reported success but ${GRAPH_JSON} is absent."

# --- F1 post-verification: the graph must actually be RICH --------------------
# A degraded Layer 1 graph contains only kind crate/file nodes and only
# depends_on/contains edges. A rich graph adds item-level nodes (struct, method,
# trait, ...) and semantic edges (calls, implements, ...). Require representative
# node kinds AND edge relations from BOTH the rich item layer (--rich) and the
# reference layer (--references); their absence means we silently got Layer 1.
python3 - "${GRAPH_JSON}" "${NIGHTLY}" <<'PY'
import json
import sys

path, nightly = sys.argv[1], sys.argv[2]
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
        "rich build degraded to Layer 1 — need the pinned nightly + rust-analyzer.\n"
        f"  {len(nodes)} nodes / {len(edges)} edges in {path}\n"
        f"  node kinds present: {sorted(k for k in node_kinds if k)}\n"
        f"  edge relations present: {sorted(r for r in edge_relations if r)}\n"
        f"  MISSING rich node kinds: {missing_kinds}\n"
        f"  MISSING rich edge relations: {missing_relations}\n"
        "The rich item layer (--rich) and/or reference edges (--references) were "
        f"skipped. Confirm: rustup toolchain install {nightly}; rustup component add "
        f"rust-analyzer --toolchain {nightly}.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"[build_graph_rich] VERIFIED rich: {len(nodes)} nodes / {len(edges)} edges\n"
    f"  node kinds: {sorted(k for k in node_kinds if k)}\n"
    f"  edge relations: {sorted(r for r in edge_relations if r)}"
)
PY

# --- Stamp: only after verification above has exited 0, AND only on a -------
# --- clean tree ------------------------------------------------------------
# A degraded build must never stamp — `set -e` already guarantees we never
# reach this line if the post-verify above failed (it exits 1 on missing
# node kinds/edge relations). A build from a DIRTY tree must never stamp
# either — a stamp is a claim that "the graph at this SHA is genuinely
# rich," and a dirty tree means the graph reflects uncommitted state that
# the SHA alone doesn't capture (see the --if-stale determinant-set comment
# above). So an existing stamp is proof the graph it points at was genuinely
# rich AND built from a clean tree at that SHA.
if [ -z "$(git status --porcelain)" ]; then
  git rev-parse HEAD > "${STAMP}"
else
  echo "[build_graph_rich] built from a DIRTY working tree — not stamping; the next --if-stale call will rebuild."
fi

echo "[build_graph_rich] OK -> ${REPO_ROOT}/${GRAPH_JSON}"
echo "[build_graph_rich] serve it:  python -m graphify.serve ${GRAPH_JSON}"
