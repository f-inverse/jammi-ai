#!/usr/bin/env bash
# stacked_sweep.sh -- the committed, parameterised producer for the P6 Stage
# B "stacked" throughput sweep: jammi-fused STACKED (FA2 dense attention arm
# + the fused AdamW step, both admitted together under
# JAMMI_KERNELS_STRICT=1) vs an ALL-OFF eager baseline (the same two ops
# forced eager) vs the PyTorch/PEFT sdpa reference, across 8 batch/seq
# shapes.
#
# This is the tracked form of the sweep section of the lead's own pod driver
# (scratchpad/a100c-chain.sh's "SECTION sweep" block, run by hand at
# perf/p6-fa2-dense@10846dd -- see .jammi/ledger/perf-s4-20260826.jsonl row
# 5; both are session-local working files, untracked -- this script IS the
# reproducible record) -- same 8 shapes, same per-shape leg structure (2x
# stacked, 1x all-off,
# 2x torch-sdpa), same fixed hyperparameters, now parameterised as
# `<worktree> <sha> <out_dir>` and committed so a re-proof of any tip is a
# one-line invocation, not a hand-typed heredoc that lives only in a pod
# session's scrollback.
#
# Usage:
#   ci/scripts/perf/stacked_sweep.sh <worktree> <sha> <out_dir>
#
# Not a CI job (no GPU on the CI image) -- invoked by hand (or via
# `ci/scripts/gpu-dev.sh run <session> ...`) on a pod that already has
# <worktree> checked out at <sha>. This script never switches git refs
# itself; see "refuses unless already at <sha>" below.
#
# Preconditions this script itself enforces (never silently worked around):
#   * `git -C <worktree> rev-parse HEAD` must equal <sha> EXACTLY (40-hex,
#     case-sensitive string compare) -- this script never checks out a ref
#     itself, so the sha this run measures is never ambiguous. The CALLER
#     owns `git checkout --detach <sha>` before invoking this script.
#   * $SWEEP_LOCK (default /root/TIMING_IN_PROGRESS) must not already exist
#     -- refuses rather than queues, so two concurrent sweeps on the same
#     shared pod can never interleave and corrupt each other's numbers via
#     GPU contention. Acquired for the duration of this script's run
#     (content: this invocation's sha + start time) and released on ANY
#     exit path (`trap ... EXIT`), including a failure partway through.
#   * `nvidia-smi --query-compute-apps=pid --format=csv,noheader` must be
#     empty -- an already-busy GPU makes every timing number this script
#     would produce meaningless (contention, not the kernel under test).
#     Override: SWEEP_SKIP_GPU_CHECK=1 (CPU-only / dry-run smoke test only).
#
# Env vars:
#   MODEL_DIR             checkpoint dir both stacks load from. Required
#                          unless SWEEP_DRY_RUN=1.
#   TORCH_PY               torch venv's python3 (default
#                          "$(dirname <worktree>)/.venv-torch-ref/bin/python3").
#                          Override explicitly when the venv is shared
#                          across worktrees, e.g. the pod's
#                          /root/jammi-ai/.venv-torch-ref.
#   CARGO_TARGET_DIR       build output dir (default <worktree>/target),
#                          forwarded verbatim to `cargo build`.
#   SWEEP_LOCK              exclusivity lock path (default
#                          /root/TIMING_IN_PROGRESS).
#   SWEEP_CUDA_ORDINAL     CUDA device ordinal both stacks target (default 0).
#   SWEEP_SKIP_GPU_CHECK=1  skip the nvidia-smi idle check.
#   SWEEP_SKIP_BUILD=1      skip `cargo build` (binary already fresh at <sha>
#                           in $CARGO_TARGET_DIR).
#   SWEEP_DRY_RUN=1         print every command instead of running it; writes
#                           `{"tool":"dry-run",...}` stub files so the
#                           summary stage still runs end-to-end against real
#                           (if fabricated-empty) files. Never touches the
#                           lock, the GPU, or the network; never claims a
#                           real number.
#   SWEEP_FAKE_BIN_SHA      unification contract C5.2: under SWEEP_DRY_RUN=1
#                           ONLY, injects a fake `jammi-bench provenance`
#                           build_sha so the provenance-mismatch refusal
#                           path is exercisable without a GPU or a real
#                           build. Inert otherwise: set it with
#                           SWEEP_DRY_RUN unset or 0 and the script refuses
#                           outright, before touching MODEL_DIR/the lock/the
#                           GPU/the build -- a real run can never launder a
#                           fabricated provenance answer through this knob.
#   SWEEP_BOX               physical/pod box tag (e.g. `a100c`) stamped
#                           mechanically into every raw leg's `box` field
#                           (unification contract C5.3's `stamp_leg()`) and
#                           into env.json. Required unless SWEEP_DRY_RUN=1
#                           (then defaults to a `dry-run-box` placeholder).
#
# Stale-build note: this script, like finetune_ab.sh (which no longer
# switches git refs within its own run either -- every leg there runs off
# ONE binary, built once, see that script's own header), only ever measures
# the ONE <sha> the caller already checked out before invoking it -- there
# is no in-script ref switch for cargo's fingerprint to get confused by. It
# still forces a `cargo clean -p jammi-kernels --release` before the one
# build this script performs (skippable via SWEEP_SKIP_BUILD=1), because
# <worktree>'s $CARGO_TARGET_DIR is typically warm from a PREVIOUS
# invocation at a DIFFERENT sha -- the SAME cargo-fingerprint hazard class
# (jammi-kernels' CUDA build.rs can leave cargo's fingerprint satisfied by
# a PREVIOUS invocation's compiled artifact even though the caller's own
# checkout moved since), just triggered by a different cause here
# (a warm, cross-invocation $CARGO_TARGET_DIR) than the in-script
# git-checkout switch this hazard was originally documented against.
#
# Shapes (batch, seq), FIXED, in this order: 8x512 8x128 1x128 1x512 16x128
# 8x256 16x512 8x1024. Fixed hyperparameters (not CLI-configurable -- this
# sweep's whole point is a repeatable, comparable-across-runs number): 25
# measured steps, 5 warmup, seed 42, LoRA rank 16 / alpha 32 / dropout 0,
# target modules Wqkv,Wo,Wi, backbone bf16.
#
# Per shape, three arms:
#   * 2x stacked    -- JAMMI_KERNELS_STRICT=1, jammi-bench finetune-step, no
#                      JAMMI_KERNELS_DISABLE (every eligible fused op --
#                      including the FA2 dense attention arm and the fused
#                      AdamW step -- admits or the run ERRORS, never
#                      silently falls back to eager wearing a fused label).
#   * 1x all-off    -- JAMMI_KERNELS_DISABLE=$ALLOFF
#                      --expect-kernels-disabled $ALLOFF (also under
#                      JAMMI_KERNELS_STRICT=1) -- the eager baseline this
#                      sweep's "stacked" ratio is measured against. ALLOFF
#                      names exactly attention_block_flash,adamw_step_fused
#                      -- this branch's OTHER documented disable names
#                      (cast_scale_bf16_f32, cast_add_bf16 --
#                      ops/low_rank_residual_linear.rs, real dispatch sites)
#                      are deliberately NOT included: finetune_step.rs's own
#                      report schema (report.rs's FinetuneStepTier) carries
#                      no counter fields for either name, so requesting
#                      their disable here would be unverifiable noise (the
#                      counters this script could read back would be None),
#                      not a real, checkable all-off leg. The two names
#                      above are the entire admitted set finetune-step's
#                      report can actually attest to.
#   * 2x torch-sdpa -- crates/jammi-bench/reference/torch_finetune_step.py
#                      --attn sdpa, matched batch/seq/steps/warmup/seed/LoRA
#                      shape, --lora-init peft (a throughput row, not a
#                      trajectory-equivalence row -- see
#                      torch_finetune_step.py's own LoRA-init section /
#                      finetune_ab.sh's header for why "peft" is right
#                      here).
#
# Non-vacuous control, checked per leg (never just "did it not crash"):
#   * every stacked leg's OWN report must show
#     attention_block_flash_fused_dispatches > 0,
#     attention_block_flash_declined_dispatches == 0,
#     attention_block_fused_dispatches == 0, adamw_fused_dispatches > 0,
#     adamw_eager_dispatches == 0, kernels_disabled_requested == [] and
#     kernels_disabled_fired == [] -- a stacked leg reading fused == 0
#     anywhere is INVALID, not a fast number.
#   * the all-off leg's OWN report must show
#     attention_block_flash_declined_dispatches > 0,
#     attention_block_flash_fused_dispatches == 0, adamw_eager_dispatches
#     > 0, adamw_fused_dispatches == 0, kernels_disabled_requested ==
#     kernels_disabled_fired == sorted($ALLOFF) -- an all-off leg that
#     silently kept a fused dispatch alive is INVALID, not a slow number.
#   * every p50 read back is checked for finiteness (NaN/inf/null all fail
#     the same way a diverged-but-still-"passing" run would slip past a
#     naive `x > 0` check -- see summarize()'s `_finite` guard).
# A leg failing this check is written into summary.json with
# "status": "INVALID" (never silently dropped, never silently reclassified
# as a slow but valid datum) -- summarize()'s own doc has the full rule.
#
# Records nvidia-smi name+driver and the torch venv's torch/transformers/
# peft versions ONCE, into $OUT_DIR/env.json -- not re-queried per leg (the
# box does not change mid-sweep; a per-leg query would just be 40 redundant
# subprocess calls for a fact that cannot change within one script run).
#
# NOT a comparator: this script only RUNS the sweep and writes raw per-leg
# JSON + one summary.json folding p50/dispatch-counters/ratios per shape --
# it does not compute a PASS/FAIL bar the way ab_merge.py's fused_proof
# does. ab_merge.py's own LEGS/CASCADE_BASES tables assume the
# jammi-eager/jammi-fused/torch-eager/torch-sdpa 4-leg shape finetune_ab.sh
# produces; this sweep's per-shape shape (stacked/all-off/torch-sdpa, no
# jammi-eager leg at all) does not fit it, so this script carries its own,
# narrower summarize() instead of forcing an ill-fitting import. A human
# (or the cuda-run artifact this script's output feeds) reads summary.json's
# per-shape ratios.

set -uo pipefail

if [ "$#" -ne 3 ]; then
  echo "::error::usage: $0 <worktree> <sha> <out_dir>" >&2
  exit 2
fi

WORKTREE="$1"
SHA="$2"
OUT_DIR="$3"

if [ ! -d "$WORKTREE" ]; then
  echo "::error::worktree '$WORKTREE' does not exist" >&2
  exit 2
fi

SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::<sha> must be a 40-hex commit, got '$SHA' -- a short sha is ambiguous, refusing to guess" >&2
  exit 2
fi
SHA="$(printf '%s' "$SHA" | tr 'A-F' 'a-f')"

SWEEP_DRY_RUN="${SWEEP_DRY_RUN:-0}"
SWEEP_SKIP_GPU_CHECK="${SWEEP_SKIP_GPU_CHECK:-0}"
SWEEP_SKIP_BUILD="${SWEEP_SKIP_BUILD:-0}"
SWEEP_CUDA_ORDINAL="${SWEEP_CUDA_ORDINAL:-0}"
SWEEP_LOCK="${SWEEP_LOCK:-/root/TIMING_IN_PROGRESS}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORKTREE/target}"
TORCH_PY="${TORCH_PY:-$(cd "$WORKTREE/.." && pwd)/.venv-torch-ref/bin/python3}"

# --- SWEEP_FAKE_BIN_SHA is a DRY-RUN-ONLY test knob (contract C5.2): it
# exists to exercise the provenance-mismatch refusal path below without a
# GPU or a real build, never to let a REAL run supply its own answer to the
# question that check exists to ask. Checked here, before ANY other
# precondition, so a real invocation with the knob set can never reach the
# build/GPU/lock stages on the strength of a fabricated provenance value --
# inert unless SWEEP_DRY_RUN=1, and a real run with it set REFUSES outright
# rather than silently ignoring it.
if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then
  echo "::error::SWEEP_FAKE_BIN_SHA is set but SWEEP_DRY_RUN != 1 -- this is a dry-run-only test knob (contract C5.2) for exercising the '\$BIN provenance' mismatch refusal without a real binary; a REAL run may never inject its own provenance answer. Refusing." >&2
  exit 2
fi

# --- refuse unless the worktree is already checked out at exactly <sha> ---
# Checked BEFORE any other precondition (MODEL_DIR, the lock, the GPU) --
# the sha mismatch is the one thing that means "this run would not measure
# what it claims to", so it must never be masked by a downstream refusal
# that happens to trip first.
ACTUAL_HEAD="$(git -C "$WORKTREE" rev-parse HEAD 2>&1)" || {
  echo "::error::'git -C $WORKTREE rev-parse HEAD' failed: $ACTUAL_HEAD" >&2
  exit 2
}
if [ "$ACTUAL_HEAD" != "$SHA" ]; then
  echo "::error::worktree '$WORKTREE' is at $ACTUAL_HEAD, not the requested $SHA -- this script never checks out a ref itself; the caller must 'git -C $WORKTREE checkout --detach $SHA' first." >&2
  exit 2
fi

if [ -z "${MODEL_DIR:-}" ]; then
  if [ "$SWEEP_DRY_RUN" = "1" ]; then
    MODEL_DIR="/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER"
    echo "::warning::SWEEP_DRY_RUN=1 and MODEL_DIR unset -- printed commands use a placeholder path; nothing is read from it."
  else
    echo "::error::MODEL_DIR must name a checkpoint directory" >&2
    exit 2
  fi
fi

if [ -z "${SWEEP_BOX:-}" ]; then
  if [ "$SWEEP_DRY_RUN" = "1" ]; then
    SWEEP_BOX="dry-run-box"
  else
    echo "::error::SWEEP_BOX must name the physical/pod box this run measures on (stamped into every raw leg mechanically, contract C5.3)" >&2
    exit 2
  fi
fi

mkdir -p "$OUT_DIR"

# --- exclusivity lock: refuse (never queue) if another job holds it ---
if [ "$SWEEP_DRY_RUN" != "1" ]; then
  if [ -e "$SWEEP_LOCK" ]; then
    echo "::error::$SWEEP_LOCK already exists (held by: $(cat "$SWEEP_LOCK" 2>/dev/null || echo '<unreadable>')) -- refusing to start a concurrent sweep on the same box." >&2
    exit 1
  fi
  echo "stacked_sweep.sh $SHA $(date -u +%FT%TZ)" > "$SWEEP_LOCK"
  trap 'rm -f "$SWEEP_LOCK"' EXIT
fi

# --- GPU must be idle before the first leg starts ---
if [ "$SWEEP_SKIP_GPU_CHECK" != "1" ]; then
  BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "::error::'nvidia-smi --query-compute-apps' failed (rc=$RC): $BUSY -- refusing to proceed without a confirmed-idle GPU. Set SWEEP_SKIP_GPU_CHECK=1 only for a CPU/dry-run smoke test." >&2
    exit 1
  fi
  if [ -n "$BUSY" ]; then
    echo "::error::GPU is not idle -- nvidia-smi reports compute processes:" >&2
    echo "$BUSY" >&2
    exit 1
  fi
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$SWEEP_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# One measurement leg. NEVER aborts the sweep: a leg failure is recorded as
# this leg's outcome (its .exit file + stderr tail), not propagated as a
# script error -- the whole point of a sweep is that one shape OOM-ing
# tells you something; it must not hide the other seven.
run_leg() {
  local tag="$1"
  shift
  local -a cmd=("$@")
  local out_file="$OUT_DIR/${tag}.json"
  local err_file="$OUT_DIR/${tag}.err"
  local exit_file="$OUT_DIR/${tag}.exit"

  printf -- '--- %s: ' "$tag"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$SWEEP_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","sweep_dry_run":true,"tag":"%s"}\n' "$tag" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${tag} FAILED (exit ${rc}) -- recorded as a leg outcome; sweep continues."
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

# --- build ---
if [ "$SWEEP_SKIP_BUILD" != "1" ]; then
  echo "=== build $(date -u +%FT%TZ) ==="
  run_cmd cargo clean -p jammi-kernels --release --manifest-path "$WORKTREE/Cargo.toml" || {
    echo "::error::cargo clean -p jammi-kernels failed" >&2
    exit 1
  }
  CARGO_TARGET_DIR="$CARGO_TARGET_DIR" run_cmd cargo build --release -p jammi-bench \
    --features cuda,jammi-encoders/flash-attn --manifest-path "$WORKTREE/Cargo.toml" || {
    echo "::error::cargo build -p jammi-bench --features cuda,jammi-encoders/flash-attn failed" >&2
    exit 1
  }
fi

BIN="$CARGO_TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$WORKTREE/crates/jammi-bench/reference/torch_finetune_step.py"

if [ "$SWEEP_DRY_RUN" != "1" ]; then
  if [ ! -x "$BIN" ]; then
    echo "::error::jammi-bench binary not found at $BIN -- build it first, or unset SWEEP_SKIP_BUILD." >&2
    exit 1
  fi
  if [ ! -x "$TORCH_PY" ]; then
    echo "::error::torch venv python not found/executable at $TORCH_PY -- set TORCH_PY explicitly." >&2
    exit 1
  fi
fi

# --- provenance cross-check (unification contract C5.1): refuse BEFORE any
# leg runs if the binary's own baked identity does not match the sha this
# invocation claims to prove. `unknown`/a `-dirty` suffix can never equal a
# 40-hex `$SHA` (already validated above), so a single string-equality
# check catches mismatch/unknown/dirty uniformly -- never a leg silently
# marked GREEN off a binary that was not built cleanly at $SHA. In REAL mode
# this ALWAYS queries the real binary -- SWEEP_FAKE_BIN_SHA was already
# refused above if set outside SWEEP_DRY_RUN=1, so there is no real-mode
# path that can inject a fake answer here. Under SWEEP_DRY_RUN=1 with no
# injected SWEEP_FAKE_BIN_SHA there is no real binary to query
# (SWEEP_SKIP_BUILD may also be set) -- the refusal path is instead
# exercised via SWEEP_FAKE_BIN_SHA (contract C5.2), so the check is skipped
# only in that one dry-run-and-no-fake-sha case.
BIN_PROV_SHA=""
if [ "$SWEEP_DRY_RUN" = "1" ]; then
  if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ]; then
    BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"
  fi
else
  BIN_PROV_JSON="$("$BIN" provenance 2>&1)" || {
    echo "::error::'$BIN provenance' failed: $BIN_PROV_JSON" >&2
    exit 1
  }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" || {
    echo "::error::could not parse build_sha from '$BIN provenance' output: $BIN_PROV_JSON" >&2
    exit 1
  }
fi
if [ -n "$BIN_PROV_SHA" ] && [ "$BIN_PROV_SHA" != "$SHA" ]; then
  echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg. This single check covers three cases uniformly: a genuine mismatch, build_sha=unknown, and a '-dirty' suffix (none can ever equal the 40-hex \$SHA validated above) -- the binary was not built cleanly at the sha this run claims." >&2
  exit 1
fi

# --- env.json: box identity, queried once ---
echo "=== env $(date -u +%FT%TZ) ==="
{
  echo "{"
  echo "  \"git_sha\": \"$SHA\","
  echo "  \"box\": \"$SWEEP_BOX\","
  if [ "$SWEEP_DRY_RUN" = "1" ]; then
    echo "  \"nvidia_smi\": \"[dry-run]\","
    echo "  \"torch_versions\": \"[dry-run]\""
  else
    SMI="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tr -d '\r')"
    VERS="$("$TORCH_PY" -c 'import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)' 2>&1)"
    echo "  \"nvidia_smi\": \"$(printf '%s' "$SMI" | sed 's/"/\\"/g')\","
    echo "  \"torch_versions\": \"$(printf '%s' "$VERS" | sed 's/"/\\"/g')\""
  fi
  echo "}"
} > "$OUT_DIR/env.json"
cat "$OUT_DIR/env.json"

ALLOFF="attention_block_flash,adamw_step_fused"

# jammi common flags (both stacked and all-off arms share these).
JAMMI_COMMON=(
  --model-dir "$MODEL_DIR"
  --steps 25 --warmup 5
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0
  --target-modules "Wqkv,Wo,Wi"
  --backbone-dtype bf16
  --cuda "$SWEEP_CUDA_ORDINAL" --seed 42
  --batched-forward true
)
# torch common flags -- matched shape/steps/seed/LoRA; --lora-init peft is
# the throughput-row default (see the ALLOFF comment block above / this
# script's header for why "peft" is right here, not "jammi").
TORCH_COMMON=(
  --model-dir "$MODEL_DIR"
  --steps 25 --warmup 5
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0
  --target-modules "Wqkv,Wo,Wi"
  --dtype bf16 --lora-init peft
  --cuda "$SWEEP_CUDA_ORDINAL" --seed 42
)

SHAPES=("8 512" "8 128" "1 128" "1 512" "16 128" "8 256" "16 512" "8 1024")

echo "=== sweep $(date -u +%FT%TZ) ==="
for shape in "${SHAPES[@]}"; do
  set -- $shape
  BATCH="$1"; SEQ="$2"
  TAG="b${BATCH}s${SEQ}"

  for r in r1 r2; do
    JAMMI_KERNELS_STRICT=1 run_leg "${TAG}_stacked.${r}" \
      "$BIN" finetune-step "${JAMMI_COMMON[@]}" --batch "$BATCH" --seq "$SEQ"
  done

  JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$ALLOFF" run_leg "${TAG}_alloff.r1" \
    "$BIN" finetune-step "${JAMMI_COMMON[@]}" --batch "$BATCH" --seq "$SEQ" \
    --expect-kernels-disabled "$ALLOFF"

  for r in r1 r2; do
    run_leg "${TAG}_torch.${r}" \
      "$TORCH_PY" "$REF_SCRIPT" "${TORCH_COMMON[@]}" --attn sdpa --batch "$BATCH" --seq "$SEQ"
  done
done

echo "=== summarize $(date -u +%FT%TZ) ==="
# Always runs (even under SWEEP_DRY_RUN=1) -- it only reads the leg output
# files run_leg already wrote (real files even in dry-run, just dry-run
# stub JSON), it never issues a GPU/network/build command itself, so
# run_cmd's "print instead of execute" gate does not apply here.
python3 - "$OUT_DIR" "$SHA" "$SWEEP_BOX" <<'PYEOF'
import json, math, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
SWEEP_GIT_SHA = sys.argv[2]
SWEEP_BOX_TAG = sys.argv[3]
ALLOFF_SORTED = sorted(["attention_block_flash", "adamw_step_fused"])


def stamp_leg(full_tag, tier, producer_kind, status):
    """Unification contract C5.3: mechanically writes the artifact schema
    (rules (a)-(f): schema_version/git_sha/box/producer/status) PLUS the v2
    identity stamp (leg_schema_version, identity{tier,producer_kind,
    leg_shape:"raw"}) into the RAW leg file this summarize stage just
    folded a number from -- replacing the hand-applied stamp every
    previously-committed stacked raw leg carried (CV4). A dry-run stub
    (`{"tool":"dry-run",...}`) or an unparsable/missing raw file is left
    untouched -- there is nothing real to stamp."""
    path = out_dir / f"{full_tag}.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict) or data.get("tool") == "dry-run":
        return
    data["schema_version"] = 2
    data["leg_schema_version"] = 2
    data["git_sha"] = SWEEP_GIT_SHA
    data["box"] = SWEEP_BOX_TAG
    data["status"] = status
    data["producer"] = {
        "path": "ci/scripts/perf/stacked_sweep.sh",
        "kind": "script",
        "invocation": "ci/scripts/perf/stacked_sweep.sh <worktree> <sha> <out_dir>",
        "gating": "none",
    }
    data["identity"] = {"tier": tier, "producer_kind": producer_kind, "leg_shape": "raw"}
    path.write_text(json.dumps(data, indent=2, sort_keys=False))

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def load(tag):
    """Returns (data_or_none, error_str_or_none). A dry-run stub, a
    nonzero exit, or a JSON parse failure are all reported as errors --
    NEVER silently treated as a missing-but-fine leg."""
    exit_file = out_dir / f"{tag}.exit"
    json_file = out_dir / f"{tag}.json"
    if exit_file.exists():
        rc = exit_file.read_text().strip()
        if rc not in ("0",):
            return None, f"leg exited {rc}"
    if not json_file.exists():
        return None, "no output json"
    try:
        data = json.loads(json_file.read_text())
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    if data.get("tool") == "dry-run":
        return data, None
    return data, None

def jammi_tier(data):
    return (data or {}).get("tiers", {}).get("finetune_step") or {}

def torch_tier(data):
    return (data or {}).get("finetune_step") or {}

def p50(tier):
    v = tier.get("s_per_step_p50")
    if isinstance(v, dict):
        v = v.get("value")
    return v

def counter(tier, key):
    v = tier.get(key)
    if isinstance(v, dict):
        v = v.get("value")
    return v

def vram(tier):
    v = tier.get("peak_vram_bytes")
    if isinstance(v, dict):
        v = v.get("value")
    return v

def check_stacked(tier):
    """Non-vacuous positive control: every counter this leg's own report
    carries must show the fused path actually admitted, not merely 'no
    error' -- and every p50 must be finite (a NaN/inf p50 is exactly the
    diverged-but-'passing' shape a naive `x > 0` check lets through)."""
    problems = []
    if not _finite(p50(tier)):
        problems.append(f"s_per_step_p50 not finite: {p50(tier)!r}")
    if counter(tier, "attention_block_flash_fused_dispatches") in (None, 0):
        problems.append("attention_block_flash_fused_dispatches not > 0")
    if counter(tier, "attention_block_flash_declined_dispatches") not in (0, None):
        problems.append("attention_block_flash_declined_dispatches != 0")
    if counter(tier, "attention_block_fused_dispatches") not in (0, None):
        problems.append("attention_block_fused_dispatches != 0 (block arm should be idle when flash admits)")
    if counter(tier, "adamw_fused_dispatches") in (None, 0):
        problems.append("adamw_fused_dispatches not > 0")
    if counter(tier, "adamw_eager_dispatches") not in (0, None):
        problems.append("adamw_eager_dispatches != 0")
    if tier.get("kernels_disabled_requested") not in ([], None):
        problems.append(f"kernels_disabled_requested not empty: {tier.get('kernels_disabled_requested')!r}")
    if tier.get("kernels_disabled_fired") not in ([], None):
        problems.append(f"kernels_disabled_fired not empty: {tier.get('kernels_disabled_fired')!r}")
    return problems

def check_alloff(tier):
    """Non-vacuous negative control: the disabled ops must show real
    fallback traffic (declined/eager > 0), not just an absence of the
    fused counter -- a leg that silently kept a fused dispatch alive while
    still reporting 'requested'/'fired' correctly must fail here too."""
    problems = []
    if not _finite(p50(tier)):
        problems.append(f"s_per_step_p50 not finite: {p50(tier)!r}")
    if counter(tier, "attention_block_flash_declined_dispatches") in (None, 0):
        problems.append("attention_block_flash_declined_dispatches not > 0")
    if counter(tier, "attention_block_flash_fused_dispatches") not in (0, None):
        problems.append("attention_block_flash_fused_dispatches != 0")
    if counter(tier, "adamw_eager_dispatches") in (None, 0):
        problems.append("adamw_eager_dispatches not > 0")
    if counter(tier, "adamw_fused_dispatches") not in (0, None):
        problems.append("adamw_fused_dispatches != 0")
    req = sorted(tier.get("kernels_disabled_requested") or [])
    fired = sorted(tier.get("kernels_disabled_fired") or [])
    if req != ALLOFF_SORTED:
        problems.append(f"kernels_disabled_requested {req!r} != {ALLOFF_SORTED!r}")
    if fired != ALLOFF_SORTED:
        problems.append(f"kernels_disabled_fired {fired!r} != {ALLOFF_SORTED!r}")
    return problems

def check_torch(tier):
    problems = []
    if not _finite(p50(tier)):
        problems.append(f"s_per_step_p50 not finite: {p50(tier)!r}")
    return problems

SHAPES = [(8, 512), (8, 128), (1, 128), (1, 512), (16, 128), (8, 256), (16, 512), (8, 1024)]

summary = {}
for batch, seq in SHAPES:
    tag = f"b{batch}s{seq}"
    shape_out = {"batch": batch, "seq": seq, "legs": {}}

    stacked_p50s = []
    for r in ("r1", "r2"):
        data, err = load(f"{tag}_stacked.{r}")
        leg = {"file": f"{tag}_stacked.{r}.json"}
        if err:
            leg["status"] = "INVALID"
            leg["problems"] = [err]
        else:
            tier = jammi_tier(data)
            problems = [] if data.get("tool") == "dry-run" else check_stacked(tier)
            leg["status"] = "INVALID" if problems else "GREEN"
            if problems:
                leg["problems"] = problems
            leg["s_per_step_p50"] = p50(tier)
            leg["peak_vram_bytes"] = vram(tier)
            leg["attention_block_flash_fused_dispatches"] = counter(tier, "attention_block_flash_fused_dispatches")
            leg["attention_block_flash_declined_dispatches"] = counter(tier, "attention_block_flash_declined_dispatches")
            leg["adamw_fused_dispatches"] = counter(tier, "adamw_fused_dispatches")
            leg["adamw_eager_dispatches"] = counter(tier, "adamw_eager_dispatches")
            if leg["status"] == "GREEN" and _finite(leg["s_per_step_p50"]):
                stacked_p50s.append(leg["s_per_step_p50"])
        stamp_leg(f"{tag}_stacked.{r}", "finetune_step", "jammi", leg["status"])
        shape_out["legs"][f"stacked_{r}"] = leg

    data, err = load(f"{tag}_alloff.r1")
    alloff_leg = {"file": f"{tag}_alloff.r1.json"}
    alloff_p50 = None
    if err:
        alloff_leg["status"] = "INVALID"
        alloff_leg["problems"] = [err]
    else:
        tier = jammi_tier(data)
        problems = [] if data.get("tool") == "dry-run" else check_alloff(tier)
        alloff_leg["status"] = "INVALID" if problems else "GREEN"
        if problems:
            alloff_leg["problems"] = problems
        alloff_leg["s_per_step_p50"] = p50(tier)
        alloff_leg["peak_vram_bytes"] = vram(tier)
        alloff_leg["attention_block_flash_declined_dispatches"] = counter(tier, "attention_block_flash_declined_dispatches")
        alloff_leg["adamw_eager_dispatches"] = counter(tier, "adamw_eager_dispatches")
        alloff_leg["kernels_disabled_requested"] = tier.get("kernels_disabled_requested")
        alloff_leg["kernels_disabled_fired"] = tier.get("kernels_disabled_fired")
        if alloff_leg["status"] == "GREEN" and _finite(alloff_leg["s_per_step_p50"]):
            alloff_p50 = alloff_leg["s_per_step_p50"]
    stamp_leg(f"{tag}_alloff.r1", "finetune_step", "jammi", alloff_leg["status"])
    shape_out["legs"]["alloff_r1"] = alloff_leg

    torch_p50s = []
    for r in ("r1", "r2"):
        data, err = load(f"{tag}_torch.{r}")
        leg = {"file": f"{tag}_torch.{r}.json"}
        if err:
            leg["status"] = "INVALID"
            leg["problems"] = [err]
        else:
            tier = torch_tier(data)
            problems = [] if data.get("tool") == "dry-run" else check_torch(tier)
            leg["status"] = "INVALID" if problems else "GREEN"
            if problems:
                leg["problems"] = problems
            leg["s_per_step_p50"] = p50(tier)
            if leg["status"] == "GREEN" and _finite(leg["s_per_step_p50"]):
                torch_p50s.append(leg["s_per_step_p50"])
        stamp_leg(f"{tag}_torch.{r}", "finetune_step", "torch", leg["status"])
        shape_out["legs"][f"torch_{r}"] = leg

    if stacked_p50s:
        shape_out["stacked_p50_min_s"] = min(stacked_p50s)
    if torch_p50s:
        shape_out["torch_p50_min_s"] = min(torch_p50s)
    if alloff_p50 is not None:
        shape_out["alloff_p50_s"] = alloff_p50
    if stacked_p50s and torch_p50s:
        shape_out["ratio_torch_over_stacked"] = shape_out["torch_p50_min_s"] / shape_out["stacked_p50_min_s"]
    if alloff_p50 is not None and stacked_p50s:
        shape_out["ratio_alloff_over_stacked"] = alloff_p50 / shape_out["stacked_p50_min_s"]

    summary[tag] = shape_out

env = json.loads((out_dir / "env.json").read_text())
report = {"git_sha": env.get("git_sha"), "env": env, "shapes": summary}
(out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=False))
print(json.dumps(report, indent=2, sort_keys=False))
PYEOF
SUMMARY_RC=$?

echo
echo "=== stacked sweep done: ${OUT_DIR} ==="
exit "$SUMMARY_RC"
