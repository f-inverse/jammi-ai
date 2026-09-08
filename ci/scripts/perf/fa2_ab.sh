#!/usr/bin/env bash
# a100b (exclusive timing box): P6 B3-dense @5886c6b — one-build A/B via JAMMI_KERNELS_DISABLE=attention_block_flash
set -uo pipefail; [ -f /root/.jammi_env ] && . /root/.jammi_env; export PATH="$HOME/.cargo/bin:$PATH"
# The per-(shape,leg,repeat) step body lives in fa2_ab_leg.sh, sourced from
# beside THIS file (never a hardcoded /root path) so it resolves the same
# way whether this script runs from a repo checkout or a git worktree.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fa2_ab_leg.sh"
while [ -f /root/TIMING_IN_PROGRESS ]; do sleep 20; done; echo "lead-fa2-ab $(date -u +%FT%TZ)" > /root/TIMING_IN_PROGRESS
cd /root/jammi-ai && git fetch origin perf/p6-fa2-dense -q && { [ -d /root/wt-fa2 ] || git worktree add -q /root/wt-fa2 5886c6b; }
cd /root/wt-fa2 && git checkout -q --detach 5886c6b && git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass 2>&1 | tail -1 && echo "HEAD=$(git rev-parse HEAD)"; export CARGO_TARGET_DIR=/root/target-fa2
echo "=== SECTION build $(date -u +%FT%TZ) ==="; cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 1; echo "BUILD_RC=${PIPESTATUS[0]}"
B=/root/target-fa2/release/jammi-bench; MD=/root/checkpoints/ModernBERT-large; OUT=/root/fa2-ab; mkdir -p $OUT
# --- provenance cross-check (unification contract C5.1), same shape as
# stacked_sweep.sh/clip_artifact_producer.sh/finetune_ab.sh: refuse BEFORE
# any leg runs if the binary's own baked identity does not match the sha
# this checkout is actually at. `unknown`/a `-dirty` suffix can never equal
# a resolved 40-hex $SHA, so a single string-equality check catches
# mismatch/unknown/dirty uniformly; an empty reading is ALSO a refusal
# (never silently skipped) -- never a leg silently marked GREEN off a
# binary that was not built cleanly at $SHA.
SHA="$(git rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2; exit 2; fi
BIN_PROV_JSON="$("$B" provenance 2>&1)" || { echo "::error::'$B provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" || { echo "::error::could not parse build_sha from '$B provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then echo "::error::'$B provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
c=(--model-dir "$MD" --lora-rank 16 --lora-alpha 32 --target-modules "Wqkv,Wo,Wi" --backbone-dtype bf16 --cuda 0 --seed 42 --batched-forward true --steps 25 --warmup 5 --lora-dropout 0)
K=attention_block_flash
overall_rc=0
for shape in "8 512" "8 128"; do set -- $shape
  for leg in flash block; do for r in r1 r2; do
    # `fa2_ab_run_leg` (fa2_ab_leg.sh) is the per-repeat step: it runs the
    # binary, sets `step_rc`/`parse_rc`, and folds a refusal (the binary's
    # own START/END check, or a JSON-parse failure on the emitted report)
    # into `overall_rc` -- so a refused leg moves THIS SCRIPT's own exit
    # status, never merely a `FAILED` line a human has to notice in
    # scrollback.
    fa2_ab_run_leg "$B" "$OUT" "$K" "$leg" "$1" "$2" "$r" "${c[@]}"
  done; done
done
rm -f /root/TIMING_IN_PROGRESS; echo "FA2AB_EXIT=$overall_rc $(date -u +%FT%TZ)"; exit $overall_rc
