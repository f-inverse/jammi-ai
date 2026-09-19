#!/usr/bin/env bash
# Producer for a per-arch flash validation artifact
# (`crates/jammi-kernels/artifacts/cuda-runs/<date>-arch-set-<sha7>-<box>.json`),
# the evidence `ci/scripts/check_arch_validation_freshness.py` reads for each
# entry of `crates/jammi-kernels/build.rs::VALIDATED_SMS`.
#
# Runs ON A GPU POD from a clean checkout of the exact commit being proven:
#   1. the producer test: ModernBERT-large's padded flash arm against the block
#      arm on real rows (`live-flash-oracle-tests`, 8 seeds);
#   2. on a device with >= 79 GiB, the whole `live-flash-oracle-tests` suite
#      (its encoder-level oracles need that much memory).
# The rest of the arch's proof (kernels, encoders, engine, served suites) is the
# prove lane's run at the same commit; pass its run URL as PROVE_RUN.
#
# Env:
#   PROVE_RUN   the gpu-prove.yml run that proved this commit on this arch (required)
#   OUT_DIR     where the logs and the artifact go (default /root/proof-out)
#   CKPT        ModernBERT-large checkpoint dir (default /root/checkpoints/ModernBERT-large;
#               fetched from the Hugging Face hub when absent)
set -uo pipefail
: "${PROVE_RUN:?PROVE_RUN must name the gpu-prove.yml run that proved this commit on this arch}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-/root/proof-out}"
CKPT="${CKPT:-/root/checkpoints/ModernBERT-large}"
cd "$REPO" || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "::error::the checkout is dirty; the artifact must name the tree it measured" >&2; exit 2; }
SHA="$(git rev-parse HEAD)"
mkdir -p "$OUT_DIR" "$CKPT"

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
export CUDA_COMPUTE_CAP="${cap//./}"
mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass || exit 1
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  [ -s "$CKPT/$f" ] || curl -fsSL "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/$f" -o "$CKPT/$f" || exit 1
done
export JAMMI_FLASH_ORACLE_MODEL_DIR="$CKPT" CARGO_TERM_COLOR=never RUST_BACKTRACE=1

PRODUCER_TEST=modernbert::tests::flash_arm_padded_matches_block_arm_on_real_rows_cuda
PRODUCER_CMD=(cargo test -p jammi-encoders --lib --features cuda,flash-attn,live-flash-oracle-tests -- --exact "$PRODUCER_TEST" --nocapture)
SUITE_CMD=(cargo test -p jammi-encoders --lib --features cuda,flash-attn,live-flash-oracle-tests -- --test-threads=1)

leg() {
  local name="$1"; shift
  local t0=$SECONDS
  "$@" > "$OUT_DIR/$name.log" 2>&1
  printf '%s\t%s\t%s\t%s\n' "$name" "$?" "$((SECONDS - t0))" "$*" >> "$OUT_DIR/legs.tsv"
}
: > "$OUT_DIR/legs.tsv"
leg producer "${PRODUCER_CMD[@]}"
[ "$mem_mib" -ge 79000 ] && leg flash-oracle-suite "${SUITE_CMD[@]}"

python3 - "$OUT_DIR" "$SHA" "$PROVE_RUN" "$PRODUCER_TEST" "${PRODUCER_CMD[*]}" <<'PY'
import json, re, subprocess, sys
from datetime import datetime, timezone
out, sha, prove_run, test, invocation = sys.argv[1:]
q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()[0].strip()
name, driver, cap, mem, uuid = (q(f) for f in ("name", "driver_version", "compute_cap", "memory.total", "uuid"))
nvcc = subprocess.run("nvcc --version | tail -n 2 | head -n 1", shell=True, capture_output=True, text=True).stdout.strip()
legs = []
for line in open(f"{out}/legs.tsv"):
    leg, rc, wall, cmd = line.rstrip("\n").split("\t")
    text = open(f"{out}/{leg}.log").read()
    summary = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    passed = sum(int(p) for p, _, _ in summary)
    failed = sum(int(f) for _, f, _ in summary)
    legs.append({"name": leg, "command": cmd, "exit": int(rc), "wall_seconds": int(wall),
                 "passed": passed, "failed": failed,
                 "status": "ok" if rc == "0" and passed > 0 and failed == 0 else "failed"})
producer_leg = next(l for l in legs if l["name"] == "producer")
green = all(l["status"] == "ok" for l in legs) and producer_leg["passed"] == 1
tag = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("nvidia", "")).strip("-")
now = datetime.now(timezone.utc)
date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
art = {
    "schema_version": 1,
    "unit": f"arch-set-{tag}",
    "git_sha": sha,
    "date": date,
    "box": f"RunPod {name} ({mem}, driver {driver}, {nvcc}, compute_cap {cap}, {uuid})",
    "gpu": name,
    "driver": driver,
    "features": "cuda,flash-attn,live-flash-oracle-tests",
    "status": "GREEN" if green else "RED",
    "producer": {
        "path": "crates/jammi-encoders/src/modernbert.rs",
        "kind": "cargo-test",
        "invocation": invocation,
        "gating": "feature:live-flash-oracle-tests",
    },
    "producer_script": "ci/scripts/perf/arch_set_producer.sh",
    "prove_run": prove_run,
    "legs": legs,
    "note": "Per-arch flash validation: the producer test on this device, plus the whole "
            "live-flash-oracle-tests suite where the device has >= 79 GiB. The arch's kernel, "
            "encoder, engine and served suites ran in prove_run.",
}
path = f"{out}/{now:%Y-%m-%d}-arch-set-{sha[:7]}-{tag}.json"
json.dump(art, open(path, "w"), indent=1)
print("wrote", path, art["status"])
PY
