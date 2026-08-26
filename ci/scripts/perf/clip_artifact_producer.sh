#!/bin/bash
# Producer for the PR #381 device-side-clip cuda-run artifact
# (`crates/jammi-kernels/artifacts/cuda-runs/<date>-clip-<sha7>-<box>.json`).
#
# Runs ON A GPU BOX from a checkout (or an rsync'd tree — see SHA below) of the
# exact tip being proven, and folds four legs into one schema-valid artifact
# (`ci/scripts/check_cuda_run_artifacts.py` is the gate):
#   1. jammi-ai `fine_tune::optimizer` with `--features cuda,live-gpu-tests`
#      (both CUDA legs of the clip; JAMMI_REQUIRE_CUDA=1 makes a missing
#      device a hard failure, never a skip);
#   2. jammi-bench `finetune_step::tests` + `report::tests` with
#      `--features cuda` (clip-on active arm vs host reference, cross-process
#      bit-identical losses, counted clip_invocations, attention_arm,
#      the shared-identity pin, the on-GPU peak_vram_bytes arm);
#   3. the `--exact` producer test the artifact's `producer` field names;
#   4. CLIP-ON-FLASH: one real `jammi-bench finetune-step --max-grad-norm 1.0`
#      leg on a bf16 ModernBERT checkpoint with the FlashAttention-2 arm
#      compiled in (`--features cuda,jammi-encoders/flash-attn`) and
#      JAMMI_KERNELS_STRICT=1, so the artifact records the clip and flash
#      COUNTERS of the same step: `attention_block_flash_fused_dispatches > 0`,
#      `..._declined == 0`, `clip_invocations == steps + warmup + 1`,
#      `attention_arm == "fused"`, `max_grad_norm == 1.0`. The clip runs after
#      `loss.backward()` in `finetune_step.rs`'s `step_once` — i.e. over the
#      flash arm's gradients — which is what this leg proves end to end.
# Zero parsed tests in any log, or any counter predicate above failing, writes
# `"status": "INVALID"` — never green.
#
# Env:
#   REPO         checkout/tree to run in (default: the repo this script is in)
#   SHA          40-hex tip being proven; REQUIRED when REPO has no .git (an
#                rsync'd tree), otherwise `git rev-parse HEAD`
#   BOX          artifact file-name box tag (default a100b-pcie) and label
#   MODEL_DIR    bf16 ModernBERT checkpoint for leg 4 (default
#                /root/checkpoints/ModernBERT-large)
#   CARGO_TARGET_DIR        target dir for legs 1-3 (default /root/target-clip)
#   FLASH_TARGET_DIR        target dir for the flash build of leg 4
#                           (default /root/target-fa2 — reuse the FA2 build)
#   FLASH_PROFILE           release (default) | debug — leg 4 only needs the
#                           COUNTERS, so a debug build whose flash kernels
#                           are already compiled (nvcc is the long pole) is
#                           acceptable; the artifact records which profile.
#   PROOF_OUT    log dir (default /root/proof-out)
#   SKIP_FLASH_LEG=1        omit leg 4 (artifact is then INVALID by rule)
set -u
[ -f /root/.jammi_env ] && . /root/.jammi_env
export PATH="$HOME/.cargo/bin:$PATH"
export RUST_BACKTRACE=1 JAMMI_REQUIRE_CUDA=1 JAMMI_REQUIRE_FLASH_ORACLE=1
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$REPO" || exit 2
if [ -d .git ]; then SHA=${SHA:-$(git rev-parse HEAD)}; fi
: "${SHA:?SHA must be set (40-hex tip) when REPO is not a git checkout}"
SHA7=${SHA:0:7}; DATE=$(date -u +%F); TS=$(date -u +%FT%TZ)
BOX=${BOX:-a100b-pcie}
MODEL_DIR=${MODEL_DIR:-/root/checkpoints/ModernBERT-large}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/root/target-clip}
FLASH_TARGET_DIR=${FLASH_TARGET_DIR:-/root/target-fa2}
FLASH_PROFILE=${FLASH_PROFILE:-release}
case "$FLASH_PROFILE" in release) FLASH_BUILD_FLAG=--release ;; debug) FLASH_BUILD_FLAG= ;; *) echo "FLASH_PROFILE must be release|debug"; exit 2 ;; esac
PROOF_OUT=${PROOF_OUT:-/root/proof-out}; mkdir -p "$PROOF_OUT"
OUT=crates/jammi-kernels/artifacts/cuda-runs/${DATE}-clip-${SHA7}-${BOX}.json
EXACT=fine_tune::optimizer::tests::multi_var_clip_matches_host_reference_on_cuda_and_is_bit_identical_to_cpu
L1=$PROOF_OUT/clip_${SHA7}_optimizer.log; L2=$PROOF_OUT/clip_${SHA7}_bench.log
L3=$PROOF_OUT/clip_${SHA7}_exact.log;     L4=$PROOF_OUT/clip_${SHA7}_flash_leg.json
L4E=$PROOF_OUT/clip_${SHA7}_flash_leg.err; L4B=$PROOF_OUT/clip_${SHA7}_flash_build.log
STEPS=5; WARMUP=2
echo "### clip artifact producer: sha=$SHA box=$BOX $(date -u)"
cargo test -p jammi-ai --features cuda,live-gpu-tests --lib fine_tune::optimizer -- --test-threads=1 2>&1 | tee "$L1"
cargo test -p jammi-bench --features cuda -- finetune_step::tests report::tests --test-threads=1 2>&1 | tee "$L2"
cargo test -p jammi-ai --features cuda,live-gpu-tests --lib -- --exact "$EXACT" 2>&1 | tee "$L3"
if [ "${SKIP_FLASH_LEG:-0}" != "1" ]; then
  CARGO_TARGET_DIR=$FLASH_TARGET_DIR cargo build $FLASH_BUILD_FLAG -p jammi-bench --features cuda,jammi-encoders/flash-attn 2>&1 | tail -n 3 | tee "$L4B"
  BIN="$FLASH_TARGET_DIR/$FLASH_PROFILE/jammi-bench"
  # --- provenance cross-check (unification contract C5.1), same shape as
  # stacked_sweep.sh: refuse BEFORE the flash leg runs if the flash binary's
  # own baked identity does not match the sha this invocation claims to
  # prove. `unknown`/a `-dirty` suffix can never equal the 40-hex $SHA above,
  # so a single string-equality check catches mismatch/unknown/dirty
  # uniformly -- never a clip_on_flash_leg record folded into a GREEN
  # artifact off a binary that was not built cleanly at $SHA.
  BIN_PROV_JSON="$("$BIN" provenance 2>&1)" || {
    echo "::error::'$BIN provenance' failed: $BIN_PROV_JSON" >&2
    exit 1
  }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" || {
    echo "::error::could not parse build_sha from '$BIN provenance' output: $BIN_PROV_JSON" >&2
    exit 1
  }
  if [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before the flash leg runs. This single check covers three cases uniformly: a genuine mismatch, build_sha=unknown, and a '-dirty' suffix (none can ever equal the 40-hex \$SHA) -- the flash binary was not built cleanly at the sha this run claims." >&2
    exit 1
  fi
  JAMMI_KERNELS_STRICT=1 "$BIN" finetune-step \
    --model-dir "$MODEL_DIR" --batch 4 --seq 256 --steps $STEPS --warmup $WARMUP \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --target-modules "Wqkv,Wo,Wi" \
    --backbone-dtype bf16 --cuda 0 --seed 42 --batched-forward true --max-grad-norm 1.0 \
    > "$L4" 2> "$L4E"; echo "FLASH_LEG_RC=$?"
fi
python3 - "$SHA" "$OUT" "$TS" "$L1" "$L2" "$L3" "$EXACT" "$L4" "$BOX" "$STEPS" "$WARMUP" "$MODEL_DIR" "$FLASH_PROFILE" <<'PY'
import json, os, re, subprocess, sys
sha, out, ts, l1, l2, l3, exact, l4, box, steps, warmup, model_dir, flash_profile = sys.argv[1:14]
steps, warmup = int(steps), int(warmup)
def tests(path):
    rows=[]; res=None
    for line in open(path, errors="replace"):
        m = re.match(r"^test (\S+) \.\.\. (ok|FAILED|ignored)", line)
        if m: rows.append({"name": m.group(1), "status": m.group(2)})
        r = re.match(r"^test result: (\w+)\. (\d+) passed; (\d+) failed", line)
        if r:
            # ONE `test result:` line PER TEST TARGET — aggregate, never keep the last.
            cur = {"result": r.group(1), "passed": int(r.group(2)), "failed": int(r.group(3)), "targets": 1}
            res = cur if res is None else {
                "result": "ok" if res["result"] == "ok" and cur["result"] == "ok" else "FAILED",
                "passed": res["passed"] + cur["passed"], "failed": res["failed"] + cur["failed"],
                "targets": res["targets"] + cur["targets"]}
    return rows, res
t1,r1 = tests(l1); t2,r2 = tests(l2); t3,r3 = tests(l3)
ok = all(r and r["result"] == "ok" and r["passed"] > 0 for r in (r1, r2, r3))
flash = {"status": "MISSING", "predicates": {}}
if os.path.isfile(l4) and os.path.getsize(l4) > 0:
    try:
        fs = json.load(open(l4))["tiers"]["finetune_step"]
        keep = ["max_grad_norm", "clip_invocations", "attention_arm", "warmup", "steps_measured",
                "attention_block_flash_fused_dispatches", "attention_block_flash_declined_dispatches",
                "attention_block_fused_dispatches", "attention_block_eager_dispatches", "flash_compiled",
                "kernels_disabled_requested", "kernels_disabled_fired", "backbone_dtype", "batch", "seq",
                "loss_first", "loss_last", "s_per_step_p50", "triplets_per_s", "peak_vram_bytes", "device_name",
                "checkpoint_weights_sha256"]
        rec = {k: fs.get(k) for k in keep}
        preds = {
            "flash_fused_gt_0": (fs.get("attention_block_flash_fused_dispatches") or 0) > 0,
            "flash_declined_eq_0": fs.get("attention_block_flash_declined_dispatches") == 0,
            "clip_invocations_eq_steps_plus_warmup_plus_1": fs.get("clip_invocations") == steps + warmup + 1,
            "attention_arm_fused": fs.get("attention_arm") == "fused",
            "max_grad_norm_1": fs.get("max_grad_norm") == 1.0,
            "flash_compiled": fs.get("flash_compiled") is True,
            "nothing_disabled": fs.get("kernels_disabled_requested") == [],
        }
        flash = {"status": "GREEN" if all(preds.values()) else "INVALID", "predicates": preds, "record": rec,
                 "model_dir": model_dir, "profile": flash_profile, "invocation": f"JAMMI_KERNELS_STRICT=1 jammi-bench finetune-step --batch 4 --seq 256 --steps {steps} --warmup {warmup} --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --target-modules Wqkv,Wo,Wi --backbone-dtype bf16 --cuda 0 --seed 42 --batched-forward true --max-grad-norm 1.0 ({flash_profile} build, --features cuda,jammi-encoders/flash-attn)"}
    except Exception as e:  # noqa: BLE001
        flash = {"status": "INVALID", "error": f"{type(e).__name__}: {e}"}
ok = ok and flash["status"] == "GREEN"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
art = {
 "schema_version": 1, "unit": "perf/device-clip-narrow (PR #381): device-side gradient clip, clip-on-flash", "git_sha": sha, "date": ts,
 "box": f"jammi-{box} (" + sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader") + ")",
 "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"), "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
 "nvcc": sh("nvcc --version | tail -n 2 | head -n 1"), "rustc": sh("rustc -V"), "features": "cuda,live-gpu-tests (tests); cuda,jammi-encoders/flash-attn (clip-on-flash leg)",
 "status": "GREEN" if ok else "INVALID",
 "producer": {"path": "crates/jammi-ai/src/fine_tune/optimizer.rs", "kind": "cargo-test",
              "invocation": "JAMMI_REQUIRE_CUDA=1 cargo test -p jammi-ai --features cuda,live-gpu-tests --lib -- --exact " + exact,
              "gating": "env:JAMMI_REQUIRE_CUDA"},
 "producer_script": "ci/scripts/perf/clip_artifact_producer.sh",
 "note": "Execution provenance for PR #381's device-side clip at this tip: jammi-ai fine_tune::optimizer CUDA legs, jammi-bench finetune_step::tests + report::tests (clip active arm vs host reference, cross-process bit-identical clip-on losses, counted clip_invocations, attention_arm, shared identity pin, on-GPU peak_vram_bytes arm), the --exact producer test, and one real clip-on-flash finetune-step leg whose flash and clip counters are recorded under clip_on_flash_leg. Zero parsed tests or any failed counter predicate => INVALID, never green.",
 "jammi_ai_optimizer_cuda": {"log": l1, "summary": r1, "tests": t1},
 "jammi_bench_cuda": {"log": l2, "summary": r2, "tests": t2},
 "producer_exact_run": {"log": l3, "summary": r3, "tests": t3},
 "clip_on_flash_leg": flash,
}
json.dump(art, open(out, "w"), indent=1); print("wrote", out, art["status"], json.dumps(flash.get("predicates")))
PY
python3 ci/scripts/check_cuda_run_artifacts.py && echo "ARTIFACT GATE PASS"
