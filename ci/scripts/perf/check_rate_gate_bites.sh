#!/usr/bin/env bash
# The measured-rate gate has teeth: with a committed baseline inflated 100x in
# this checkout, the cheapest tier with a rate gate (`train-scale`, one
# GradCache step over synthetic pairs on the CPU) must exit non-zero. The
# workflow-layer mirror of the unit test `committed_baseline_gates_with_teeth`:
# a lane whose exit-code gate is not wired through would pass every night and
# prove nothing. The baseline is restored by copy, never by git, and the
# checkout this runs in is discarded with its runner.
#
# The tier runs through `run_scale_tiers.sh`, the one place a tier is invoked
# and the binary's provenance cross-checked.
#
#   bash ci/scripts/perf/check_rate_gate_bites.sh   # after `cargo build -p jammi-bench --release`
set -euo pipefail

baseline=crates/jammi-bench/baselines/training.json
cp "$baseline" "$baseline.orig"
trap 'mv -f "$baseline.orig" "$baseline"' EXIT
python3 ci/scripts/perf/inflate_baseline.py "$baseline" pairs_per_s
if bash "$(dirname "${BASH_SOURCE[0]}")/run_scale_tiers.sh" train-scale; then
  echo "::error::train-scale exited 0 against a 100x-inflated baseline: the rate gate is decorative"
  exit 1
fi
echo "the rate gate bites: train-scale exited non-zero against the inflated baseline"
