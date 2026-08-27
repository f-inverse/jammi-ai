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
