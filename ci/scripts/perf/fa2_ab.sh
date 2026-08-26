#!/usr/bin/env bash
# a100b (exclusive timing box): P6 B3-dense @5886c6b — one-build A/B via JAMMI_KERNELS_DISABLE=attention_block_flash
set -uo pipefail; [ -f /root/.jammi_env ] && . /root/.jammi_env; export PATH="$HOME/.cargo/bin:$PATH"
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
for shape in "8 512" "8 128"; do set -- $shape
  for leg in flash block; do for r in r1 r2; do
    if [ $leg = block ]; then JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=$K "$B" finetune-step "${c[@]}" --batch $1 --seq $2 > $OUT/b$1_s$2_$leg.$r.json 2> $OUT/b$1_s$2_$leg.$r.err
    else JAMMI_KERNELS_STRICT=1 "$B" finetune-step "${c[@]}" --batch $1 --seq $2 > $OUT/b$1_s$2_$leg.$r.json 2> $OUT/b$1_s$2_$leg.$r.err; fi
    python3 -c "
import json,sys
try:
  t=json.load(open(sys.argv[1]))['tiers']['finetune_step']; c={k:v for k,v in t.items() if 'flash' in k or k.startswith('attention_block')}
  print('FA2AB',sys.argv[2],sys.argv[3],'p50',round(t['s_per_step_p50']['value'],4),c,'req',t.get('kernels_disabled_requested'),'fired',t.get('kernels_disabled_fired'))
except Exception as e: print('FA2AB',sys.argv[2],sys.argv[3],'FAILED',e, open(sys.argv[4]).read()[-300:].replace(chr(10),' | '))" $OUT/b$1_s$2_$leg.$r.json "b$1s$2" "$leg-$r" $OUT/b$1_s$2_$leg.$r.err
  done; done
done
rm -f /root/TIMING_IN_PROGRESS; echo "FA2AB_EXIT=0 $(date -u +%FT%TZ)"
