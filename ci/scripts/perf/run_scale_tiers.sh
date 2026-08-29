#!/usr/bin/env bash
# Runs the five CPU-hermetic `*-scale` bench tiers against their committed
# same-box baselines. Each tier emits its JSON report to stdout and maps its
# gate verdict to the process exit code: a throughput below the committed
# floor (`baseline·(1 − 0.30)`) or a determinism-digest drift exits non-zero,
# which fails this script at that tier — the log names the regressed tier via
# its ::group:: header.
#
# The tier list is stated ONCE, here, and consumed by both callers: the
# nightly early-warning lane (perf.yml, never merge-gating) and the
# release-blocking perf gate (crates.yml). The documented contract is
# docs/guide/src/performance-slos.md.
#
# Expects a release-profile jammi-bench binary (the committed baselines are
# release-profile numbers) at ./target/release/jammi-bench, overridable via
# JAMMI_BENCH_BIN. Callers pin RAYON_NUM_THREADS=1 in the job env — the
# posture the committed baselines were emitted under.
set -euo pipefail

BIN="${JAMMI_BENCH_BIN:-./target/release/jammi-bench}"

# --- provenance cross-check (unification contract C5.1), same shape as
# finetune_ab.sh's/stacked_sweep.sh's/pod_build_timings.sh's own
# check_bin_provenance(): every tier below reports a throughput/digest
# verdict that BOTH callers (crates.yml's release-blocking gate,
# perf.yml's nightly early-warning lane) attribute to the commit currently
# checked out — refuse BEFORE running any tier if `$BIN`'s own baked
# `build_sha` (`report::Provenance::baked`, filled by `build.rs` at
# COMPILE time) does not match that commit. `build_sha` only refreshes
# when `cargo build` actually recompiles; a `target/` directory carried
# over from a previous checkout (a stale actions/cache restore, or a
# manual re-run against an unrebuilt tree) would otherwise let a GREEN
# leg here silently attest to a commit `$BIN` was never built from.
#
# `-c safe.directory='*'` (a PER-INVOCATION override, never a lasting
# `git config --global` mutation): both merge-path callers run this
# script inside a `container:` job (crates.yml's release-blocking
# `perf-gate`, perf.yml's `perf-scale`), where a bare `git` hits "fatal:
# detected dubious ownership" (exit 128) — the checkout is owned by a
# different uid than the container's git, and the `safe.directory`
# `actions/checkout` itself sets is not visible inside the container (the
# same trap this repo's own ci.yml/dep-dag.yml document and work around
# with a `git config --global --add safe.directory "$GITHUB_WORKSPACE"`
# step; the per-invocation `-c` form here avoids needing a second step in
# every caller, and needs no `$GITHUB_WORKSPACE` — it trusts whatever
# directory this invocation's own `git` command actually runs in).
sha="$(git -c safe.directory='*' rev-parse HEAD)"
if ! [[ "$sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "::error::provenance cross-check: HEAD did not resolve to a 40-hex commit ('$sha') -- refusing" >&2
  exit 1
fi
bin_prov_json="$("$BIN" provenance 2>&1)" || { echo "::error::'$BIN provenance' failed: $bin_prov_json" >&2; exit 1; }
bin_prov_sha="$(printf '%s' "$bin_prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
  || { echo "::error::could not parse build_sha from '$BIN provenance' output: $bin_prov_json" >&2; exit 1; }
if [ -z "$bin_prov_sha" ] || [ "$bin_prov_sha" != "$sha" ]; then
  echo "::error::'$BIN provenance' reports build_sha=$bin_prov_sha, but this checkout is at sha=$sha -- refusing before any tier below runs off this binary." >&2
  exit 1
fi

# tier — what it gates
#   train-scale               — fine-tune throughput + OOM control
#   graph-train-scale         — graph sampler throughput + digest
#   context-predictor-scale   — predictor train throughput + predict digest
#   model-inference-scale     — serving throughput + output digests
#   arxiv                     — held-out ANN-vs-exact recall over the committed corpus
TIERS=(
  train-scale
  graph-train-scale
  context-predictor-scale
  model-inference-scale
  arxiv
)

for tier in "${TIERS[@]}"; do
  echo "::group::${tier}"
  "$BIN" "$tier"
  echo "::endgroup::"
done
