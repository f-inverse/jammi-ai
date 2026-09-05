#!/usr/bin/env bash
# The #428 P2b measurement driver (docs-ci domain): a same-box fused-vs-
# eager per-step wall A/B for BERT and DistilBERT at the two activating
# shapes issue #356's own close-out profile already ACTIVATED the C-LORA
# port on (`b8/W512/f32` "wire", `b32/W64/bf16` "chapter") -- proving out
# the bias-carrying-base widening (#428's own engine half, a separate
# branch) once it lands.
#
# THIS SCRIPT IS A GPU-POD DRIVER, not a CI step -- same shape as
# `profile_356_legs.sh`/`finetune_run_ab.sh`: real legs need a real GPU, a
# built `jammi-bench` release binary (`$BENCH_BIN`), and two real checkpoint
# directories (`$MODEL_DIR_BERT`/`$MODEL_DIR_DISTILBERT`).
# `LORA_BIAS_AB_DRY_RUN=1` makes the WHOLE pipeline safe to exercise
# hermetically: `$BENCH_BIN` is swapped for a hermetic fake stand-in
# generated into a throwaway stub dir and ACTUALLY EXECUTED (never a
# hand-shaped bypass) through the EXACT SAME capture path a real leg uses.
#
# Both arms pass `--arm fused` to the binary -- CONTRACT: `finetune_run.rs`'s
# `Arm::Fused` makes no `JAMMI_KERNELS_DISABLE` claim of its own (only
# `Arm::Alloff` hard-pins a disable set and cross-checks it); the eager arm
# here is manufactured entirely via `JAMMI_KERNELS_DISABLE=lora_linear_fused`
# under `JAMMI_KERNELS_STRICT=1` (disable wins over Strict --
# `crate::admission::admit_inner`'s own doc), which is what turns a SINGLE
# build into a one-op A/B: `lora_linear_fused` is forced eager while every
# other op the same run touches is still strictly proven fused. The leg's
# arm label (`fused` / `lora_eager` / `control`) is OUR OWN bookkeeping,
# recorded in every manifest row and independently recoverable downstream
# from `kernels_disabled_requested` (the merger's own dispatch-proof check).
#
# NEGATIVE CONTROL: one extra fused-vs-fused series per model at
# `b8/W512/f32` (arm label "control", same underlying config as that
# shape's own `fused` cell -- reuses the SAME corpus and every other flag)
# gives a downstream merger a measured noise floor: two independent draws
# of the identical config, differenced, bound how much of the fused-vs-
# eager gap could be pod jitter rather than the LoRA-linear op itself.
#
# ORDER-BALANCED REPEATS: within one (model, shape) pair the fused/eager
# cells alternate which one runs first on odd/even repeats (repeat 1:
# fused,eager; repeat 2: eager,fused; ...) -- the same A,B,B,A
# drift-cancellation shape `finetune_ab.sh`'s own bar legs and
# `gpu_inference_ab.py`'s `ADJACENT_PAIRS` already use, generalized past a
# single pair to however many `$LORA_BIAS_AB_REPEATS` repeats are
# configured.
#
# STRICT PRE-FLIGHT: before the sweep, one SHORT probe (16 rows, batch 4,
# epochs 1) per (model, arm) under `JAMMI_KERNELS_STRICT=1` -- if the binary
# refuses with a typed Strict-mode fallback
# (`crate::error::KernelError::StrictModeFallback`'s own Display: "fused op
# `<op>` refused in STRICT admission mode: predicate `<predicate>` failed"),
# this script STOPS before any measured leg runs, prints the offending op
# key, and names the pre-registered remedy: re-run with
# `LORA_BIAS_AB_EXTRA_DISABLE=<op>` (comma-joined onto any value already
# set) -- appended to `JAMMI_KERNELS_DISABLE` on BOTH arms symmetrically,
# never only on one (an asymmetric disable would silently change which op
# each arm's own numbers describe). This preflight probes a small,
# arbitrary shape (width 64, dtype f32) -- it is a best-effort catch of a
# STRUCTURAL admission refusal (the bias-carrying-base predicate itself),
# not an exhaustive per-shape/per-dtype proof; a refusal specific to a
# DIFFERENT shape/dtype combination the sweep itself touches still shows up
# as that one leg's own INVALID manifest row, never silently swallowed.
#
# Every leg writes `$OUT_DIR/raw/<model>-<shape>-<arm>-<steps>-r<k>.json`
# (the run's own JSON report) plus one row in `$OUT_DIR/manifest.json` (a
# single JSON array, appended to after each leg) -- this script's own exit
# code reflects whether the SWEEP RAN, never whether every leg passed: a
# leg with `rc != 0` (or a report-envelope validation failure) marks THAT
# leg `status: invalid` in its manifest row and the sweep continues,
# `finetune_run_ab.sh`'s own "a missed bar is data" doctrine.
#
# Env vars (all required for a real run; DRY_RUN relaxes all but OUT_DIR):
#   MODEL_DIR_BERT         bert-base-uncased checkpoint dir.
#   MODEL_DIR_DISTILBERT   distilbert-base-uncased checkpoint dir.
#   BENCH_BIN              path to the jammi-bench binary (default:
#                          `$CARGO_TARGET_DIR/release/jammi-bench` or
#                          `$REPO_ROOT/target/release/jammi-bench`).
#   OUT_DIR                output directory for every leg's artifacts
#                          (default: `$REPO_ROOT/.lora-bias-ab/<ts>`).
#   LORA_BIAS_AB_REPEATS   order-balanced repeats per (model, shape, arm)
#                          cell, and per (model, control) series (default 3).
#   LORA_BIAS_AB_EXTRA_DISABLE
#                          comma-separated extra `JAMMI_KERNELS_DISABLE`
#                          entries, applied identically to BOTH arms (the
#                          preflight's own pre-registered remedy channel;
#                          default empty).
#   LORA_BIAS_AB_DRY_RUN   "1" swaps in a hermetic fake `$BENCH_BIN`
#                          stand-in and drives it through the real capture
#                          path (default "0").
#   LORA_BIAS_AB_LEGS_ONLY optional comma-separated CELL-id filter (e.g.
#                          "bert-b8W512f32-fused-r1,distilbert-control-r1")
#                          -- default empty means every cell. A cell covers
#                          BOTH its steps=100 and steps=600 runs together.
#                          Every named id is validated against the known
#                          cell table BEFORE any leg runs.
#   LORA_BIAS_AB_DRY_RUN_FAIL_OP / LORA_BIAS_AB_DRY_RUN_FAIL_PREDICATE
#                          DRY_RUN-only hooks: the hermetic fake bench stub
#                          reads these and, when `_FAIL_OP` is set, always
#                          refuses with the SAME Strict-mode-fallback text a
#                          real Strict refusal would produce -- the seam
#                          `test_lora_bias_ab_sh_dry_run.py` drives to
#                          exercise the preflight-refusal path without a
#                          GPU or a real predicate failure.
#
# Hermetic self-test: `python3 ci/scripts/perf/test_lora_bias_ab_sh_dry_run.py`
# drives this script under `LORA_BIAS_AB_DRY_RUN=1` end to end.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

LORA_BIAS_AB_DRY_RUN="${LORA_BIAS_AB_DRY_RUN:-0}"
LORA_BIAS_AB_REPEATS="${LORA_BIAS_AB_REPEATS:-3}"
LORA_BIAS_AB_EXTRA_DISABLE="${LORA_BIAS_AB_EXTRA_DISABLE:-}"
LORA_BIAS_AB_LEGS_ONLY="${LORA_BIAS_AB_LEGS_ONLY:-}"
LORA_BIAS_AB_DRY_RUN_FAIL_OP="${LORA_BIAS_AB_DRY_RUN_FAIL_OP:-}"
LORA_BIAS_AB_DRY_RUN_FAIL_PREDICATE="${LORA_BIAS_AB_DRY_RUN_FAIL_PREDICATE:-base_has_no_bias}"

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BENCH_BIN="${BENCH_BIN:-$TARGET_DIR/release/jammi-bench}"
MODEL_DIR_BERT="${MODEL_DIR_BERT:-}"
MODEL_DIR_DISTILBERT="${MODEL_DIR_DISTILBERT:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.lora-bias-ab/$TS}"

# CONTRACT legs table selector sets -- same constants `profile_356_legs.sh`
# already pins (`BERT_FULL`/`DISTIL_FULL`), reused verbatim: both models'
# FULL LoRA target-module set, the shape every activating leg in #356's own
# close-out profile measured over.
BERT_FULL="query,key,value,dense"
DISTIL_FULL="q_lin,k_lin,v_lin,out_lin,lin1,lin2"

# The two activating shapes (#356 close-out): "wire" (b8/W512/f32) and
# "chapter" (b32/W64/bf16) -- `id:batch:width:dtype`.
SHAPE_WIRE="b8W512f32:8:512:f32"
SHAPE_CHAPTER="b32W64bf16:32:64:bf16"
CONTROL_SHAPE="$SHAPE_WIRE"

# Fixed step-count pin (CONTRACT): N=100, M=600 for every leg, both
# multiples of 50, M>N with a wide separation for a clean per-step average
# -- mirrors `profile_356_legs.sh`'s own "STEP-COUNT PIN" doctrine.
STEPS_N=100
STEPS_M=600

mkdir -p "$OUT_DIR"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"
CORPUS_DIR="$OUT_DIR/corpus"
mkdir -p "$CORPUS_DIR"
MANIFEST="$OUT_DIR/manifest.json"

if [ -f "$MANIFEST" ]; then
  echo "::error::OUT_DIR ($OUT_DIR) already has a manifest.json -- refusing to silently append to a prior run's results; use a fresh OUT_DIR." >&2
  exit 2
fi
printf '[]' > "$MANIFEST"

if [ "$LORA_BIAS_AB_DRY_RUN" != "1" ]; then
  if [ -z "$MODEL_DIR_BERT" ] || [ -z "$MODEL_DIR_DISTILBERT" ]; then
    echo "::error::MODEL_DIR_BERT and MODEL_DIR_DISTILBERT must both be set for a real run." >&2
    exit 2
  fi
  if [ ! -x "$BENCH_BIN" ]; then
    echo "::error::$BENCH_BIN is not an executable file -- refusing before any leg runs." >&2
    exit 2
  fi
else
  MODEL_DIR_BERT="${MODEL_DIR_BERT:-/root/checkpoints/bert-base-uncased-DRY-RUN-PLACEHOLDER}"
  MODEL_DIR_DISTILBERT="${MODEL_DIR_DISTILBERT:-/root/checkpoints/distilbert-base-uncased-DRY-RUN-PLACEHOLDER}"
  echo "::warning::LORA_BIAS_AB_DRY_RUN=1 -- nothing is read from MODEL_DIR_*/BENCH_BIN."
fi

# --- trace-echo helper (same shape as profile_356_legs.sh's `_print_cmd`):
# ALWAYS to stderr, never stdout -- a trace line sharing stdout with a
# captured child process's own JSON report would corrupt the report.
_print_cmd() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

# --- provenance cross-check (unification contract C5.1), same shape as
# profile_356_legs.sh/finetune_run_ab.sh/fa2_ab.sh/finetune_ab.sh/
# encode_ab.sh/stacked_sweep.sh/clip_artifact_producer.sh: refuse BEFORE any
# leg runs if the binary's own baked identity does not match the sha this
# checkout is actually at.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$LORA_BIAS_AB_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BENCH_BIN" provenance 2>&1)" || { echo "::error::'$BENCH_BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BENCH_BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BENCH_BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

BOX="$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true) | head -1)"
if [ -z "$BOX" ]; then
  BOX="unknown ($(hostname))"
fi

HELDOUT_DIR="$REPO_ROOT/cookbook/fixtures/finetune_heldout"
HELDOUT_IDS="$HELDOUT_DIR/heldout_ids.txt"
HELDOUT_JSONL="$HELDOUT_DIR/heldout_pairs.jsonl"

# --- hermetic fake bench stand-in, used ONLY under LORA_BIAS_AB_DRY_RUN=1
# -- generated ONCE, reused for every leg/probe, so the DRY_RUN sweep drives
# the EXACT SAME capture path (redirect, envelope validation) a real
# GPU-pod run uses. `$1`=steps `$2`=model `$3`=batch `$4`=width `$5`=dtype.
# Reads `JAMMI_KERNELS_DISABLE`/`JAMMI_KERNELS_STRICT` off ITS OWN
# environment -- the same env this driver exports for the real binary --
# and derives fused/eager dispatch counters from whether
# `lora_linear_fused` is named in `JAMMI_KERNELS_DISABLE`, exactly the real
# admission lattice's own "disable wins" rule.
# `LORA_BIAS_AB_DRY_RUN_FAIL_OP` (if set) makes it refuse unconditionally
# with the exact Strict-mode-fallback text
# `crate::error::KernelError::StrictModeFallback` renders, for
# `test_lora_bias_ab_sh_dry_run.py`'s own preflight-refusal arm.
DRY_RUN_STUB_DIR=""
if [ "$LORA_BIAS_AB_DRY_RUN" = "1" ]; then
  DRY_RUN_STUB_DIR="$(mktemp -d)"
  trap 'rm -rf "$DRY_RUN_STUB_DIR"' EXIT

  cat > "$DRY_RUN_STUB_DIR/fake_bench.sh" <<'FAKE_BENCH_EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "fake_bench: chatty stdout noise on purpose, mirroring real bench progress output" >&2
if [ -n "${LORA_BIAS_AB_DRY_RUN_FAIL_OP:-}" ]; then
  echo "finetune-run failed: fused op \`${LORA_BIAS_AB_DRY_RUN_FAIL_OP}\` refused in STRICT admission mode: predicate \`${LORA_BIAS_AB_DRY_RUN_FAIL_PREDICATE:-base_has_no_bias}\` failed" >&2
  exit 1
fi
python3 -c '
import json, os, sys

steps, model, batch, width, dtype = sys.argv[1:6]
steps = int(steps)
batch = int(batch)
width = int(width)
disable_raw = os.environ.get("JAMMI_KERNELS_DISABLE", "")
requested = sorted({x for x in disable_raw.split(",") if x})
lora_disabled = "lora_linear_fused" in requested or "all" in requested
fired = ["lora_linear_fused"] if lora_disabled else []

target_modules = {
    "bert": ["query", "key", "value", "dense"],
    "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
}[model]

tier = {
    "arm": "fused",
    "device_name": "dry-run-cpu",
    "kernels_disabled_requested": requested,
    "kernels_disabled_fired": fired,
    "flash_compiled": False,
    "build_features": [],
    "attention_arm": "fused",
    "split_rule": "positional_fraction_split",
    "batched_forward": True,
    "steps_measured": steps,
    "seed": 42,
    "batch": batch,
    "seq": width,
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.05,
    "margin": None,
    "target_modules": target_modules,
    "layers_to_transform": None,
    "backbone_dtype": dtype,
    "checkpoint_config_sha256": "0" * 64,
    "checkpoint_weights_sha256": "1" * 64,
    "checkpoint_weights_size_bytes": 102608,
    "max_grad_norm": None,
    "warmup": None,
    "row_lengths": None,
    "epochs": 1,
    "lr": 0.0002,
    "schedule": "constant",
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "grad_accum": 1,
    "validation_fraction": 0.0,
    "train_pairs_file_sha256": "2" * 64,
    "heldout_ids_sha256": "3" * 64,
    "heldout_pairs_sha256": "4" * 64,
    "heldout_batch_partition_sha256": "5" * 64,
    "embedding_loss": "mnrl",
    "temperature": 20.0,
    "matryoshka_dims": [],
    "early_stopping_patience": 10000,
    "early_stopping_metric": "train_loss",
    "eval_cadence": 1,
    "train_run_wall_s": 0.01 * steps,
    "lora_linear_fused_dispatches": 0 if lora_disabled else 1,
    "lora_linear_eager_dispatches": 1 if lora_disabled else 0,
}
report = {"tool": "dry-run", "lora_bias_ab_dry_run": True, "tiers": {"finetune_run": tier}}
json.dump(report, sys.stdout)
' "$@"
echo "fake_bench: stderr noise too, never captured into the report" >&2
FAKE_BENCH_EOF
  chmod +x "$DRY_RUN_STUB_DIR/fake_bench.sh"
fi

# Validates "$1" parses as JSON and carries a `tiers.finetune_run` OBJECT --
# same shape as `profile_356_legs.sh`'s `_validate_report_envelope`.
_validate_report_envelope() {
  python3 -c '
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        d = json.load(f)
except (OSError, json.JSONDecodeError) as e:
    print("::error::_validate_report_envelope: " + path + " does not parse as JSON: " + str(e), file=sys.stderr)
    sys.exit(1)
tiers = d.get("tiers")
tier = tiers.get("finetune_run") if isinstance(tiers, dict) else None
if not isinstance(tier, dict):
    print("::error::_validate_report_envelope: " + path + " has no tiers.finetune_run object", file=sys.stderr)
    sys.exit(1)
' "$1"
}

_wall_s() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
tier = d.get("tiers", {}).get("finetune_run")
if tier is None or "train_run_wall_s" not in tier:
    print("::error::_wall_s: no tiers.finetune_run.train_run_wall_s in " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
print(tier["train_run_wall_s"])
' "$1"
}

# Appends one row to the single `$MANIFEST` array -- read-modify-write is
# safe here because this driver is strictly single-threaded/sequential
# (never parallelized across legs).
_append_manifest_row() {
  MANIFEST_ROW_JSON="$1" python3 -c '
import json, os
path = os.environ["MANIFEST_PATH"]
row = json.loads(os.environ["MANIFEST_ROW_JSON"])
with open(path) as f:
    rows = json.load(f)
rows.append(row)
with open(path, "w") as f:
    json.dump(rows, f, indent=1)
'
}
export MANIFEST_PATH="$MANIFEST"

# Comma-joined `JAMMI_KERNELS_DISABLE` value for a given arm label --
# `lora_eager` always names `lora_linear_fused`; `fused`/`control` never
# do. `$LORA_BIAS_AB_EXTRA_DISABLE` (the preflight's own remedy channel) is
# appended identically on every arm -- symmetry the merger's own dispatch
# proof checks for.
_disable_for_arm() {
  local arm="$1"
  local base=""
  if [ "$arm" = "lora_eager" ]; then
    base="lora_linear_fused"
  fi
  if [ -n "$base" ] && [ -n "$LORA_BIAS_AB_EXTRA_DISABLE" ]; then
    printf '%s,%s' "$base" "$LORA_BIAS_AB_EXTRA_DISABLE"
  elif [ -n "$base" ]; then
    printf '%s' "$base"
  else
    printf '%s' "$LORA_BIAS_AB_EXTRA_DISABLE"
  fi
}

# =====================================================================
# STRICT preflight (see module doc) -- runs BEFORE any measured leg.
# =====================================================================
preflight_probe() {
  local model="$1" target_modules="$2" arm="$3"
  local model_dir
  if [ "$model" = "bert" ]; then model_dir="$MODEL_DIR_BERT"; else model_dir="$MODEL_DIR_DISTILBERT"; fi

  local probe_dir
  probe_dir="$(mktemp -d)"
  local probe_corpus="$probe_dir/probe.jsonl"
  if ! python3 "$DIR/gen_fixed_width_corpus.py" --rows 16 --min-wordpieces 32 --seed 1 --out "$probe_corpus" >/dev/null 2>&1; then
    if [ "$LORA_BIAS_AB_DRY_RUN" != "1" ]; then
      echo "::error::preflight_probe: could not generate the probe corpus." >&2
      rm -rf "$probe_dir"
      exit 1
    fi
    : > "$probe_corpus"
  fi

  local disable
  disable="$(_disable_for_arm "$arm")"
  local probe_out="$probe_dir/probe.json"
  local probe_err="$probe_dir/probe.stderr"
  local probe_rc=0

  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused
    --train-jsonl "$probe_corpus" --heldout-ids "$HELDOUT_IDS" --heldout-jsonl "$HELDOUT_JSONL"
    --seed 42 --epochs 1 --batch 4 --objective mnrl
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype f32
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --target-modules "$target_modules"
    --max-seq-length 64 --eval-cadence 1
    --work-dir "$probe_dir/work" --cuda 0
  )
  _print_cmd "${cmd[@]}"

  local -a exec_cmd=("${cmd[@]}")
  if [ "$LORA_BIAS_AB_DRY_RUN" = "1" ]; then
    exec_cmd=("$DRY_RUN_STUB_DIR/fake_bench.sh" 4 "$model" 4 64 f32)
  fi

  JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$disable" "${exec_cmd[@]}" > "$probe_out" 2> "$probe_err" || probe_rc=$?

  if [ "$probe_rc" -ne 0 ]; then
    local match
    match="$(grep -oE 'fused op `[^`]+` refused in STRICT admission mode: predicate `[^`]+` failed' "$probe_err" | head -1 || true)"
    if [ -n "$match" ]; then
      local op
      op="$(printf '%s' "$match" | sed -E 's/^fused op `([^`]+)`.*/\1/')"
      local remedy="$op"
      if [ -n "$LORA_BIAS_AB_EXTRA_DISABLE" ]; then
        remedy="${LORA_BIAS_AB_EXTRA_DISABLE},${op}"
      fi
      echo "::error::lora_bias_ab: STRICT preflight refused for model=$model arm=$arm -- $match" >&2
      echo "::error::lora_bias_ab: pre-registered remedy -- re-run with LORA_BIAS_AB_EXTRA_DISABLE=$remedy (applied to BOTH arms symmetrically)." >&2
      rm -rf "$probe_dir"
      exit 1
    fi
    echo "::error::lora_bias_ab: STRICT preflight probe FAILED for model=$model arm=$arm for a reason other than a recognized Strict-mode fallback (exit $probe_rc) -- see $probe_err" >&2
    cat "$probe_err" >&2 || true
    rm -rf "$probe_dir"
    exit 1
  fi
  rm -rf "$probe_dir"
}

for model_spec in "bert:$BERT_FULL" "distilbert:$DISTIL_FULL"; do
  model="${model_spec%%:*}"
  target_modules="${model_spec#*:}"
  for arm in fused lora_eager; do
    preflight_probe "$model" "$target_modules" "$arm"
  done
done
echo "::notice::lora_bias_ab: STRICT preflight OK for every (model, arm) -- proceeding to the sweep."

# =====================================================================
# Corpus provisioning: one (model, shape) M-row corpus + its N-row prefix,
# generated ONCE and reused across every arm/repeat/control leg that shares
# that (model, shape) -- guarantees `train_pairs_file_sha256` (an identity
# field) is byte-identical across arms/repeats within a group by
# construction, not by chance.
# =====================================================================
# Sets `CORPUS_N_FILE`/`CORPUS_FULL_FILE` (globals, NEVER a stdout return
# value) -- called directly, never inside a `$(...)` command substitution,
# so a genuine corpus-generation failure's own `exit 1` actually terminates
# THIS script rather than only the subshell a command substitution would
# otherwise confine it to.
_corpus_for() {
  local model="$1" shape_id="$2" width="$3" batch="$4"
  local full="$CORPUS_DIR/${model}-${shape_id}-m${STEPS_M}.jsonl"
  local n_file="$CORPUS_DIR/${model}-${shape_id}-n${STEPS_N}.jsonl"
  if [ ! -f "$full" ]; then
    local rows_m=$(( batch * STEPS_M ))
    local rows_n=$(( batch * STEPS_N ))
    if [ "$LORA_BIAS_AB_DRY_RUN" != "1" ]; then
      python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" --min-wordpieces "$width" \
        --seed 42 --out "$full" || { echo "::error::corpus generation failed for $model/$shape_id" >&2; exit 1; }
      head -n "$rows_n" "$full" > "$n_file" || { echo "::error::corpus slicing failed for $model/$shape_id" >&2; exit 1; }
    else
      : > "$full"
      : > "$n_file"
    fi
  fi
  CORPUS_N_FILE="$n_file"
  CORPUS_FULL_FILE="$full"
}

# =====================================================================
# One measurement leg: one (model, shape, arm, repeat, steps) unit --
# writes `$RAW_DIR/<model>-<shape>-<arm>-<steps>-r<k>.json` +
# `.stderr` + one `$MANIFEST` row. NEVER aborts the sweep on a leg-level
# failure (a leg's own rc!=0/envelope failure lives in its manifest row's
# `status`/`reason`; `finetune_run_ab.sh`'s own doctrine).
# =====================================================================
run_leg() {
  local model="$1" shape_id="$2" batch="$3" width="$4" dtype="$5" \
        target_modules="$6" arm="$7" repeat="$8" steps="$9" train_jsonl="${10}"

  local model_dir
  if [ "$model" = "bert" ]; then model_dir="$MODEL_DIR_BERT"; else model_dir="$MODEL_DIR_DISTILBERT"; fi

  local leg_id="${model}-${shape_id}-${arm}-${steps}-r${repeat}"
  local out_json="$RAW_DIR/${leg_id}.json"
  local err_file="$RAW_DIR/${leg_id}.stderr"
  local work_dir="$OUT_DIR/work/${leg_id}"
  mkdir -p "$work_dir"

  local disable
  disable="$(_disable_for_arm "$arm")"

  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused
    --train-jsonl "$train_jsonl" --heldout-ids "$HELDOUT_IDS" --heldout-jsonl "$HELDOUT_JSONL"
    --seed 42 --epochs 1 --batch "$batch" --objective mnrl
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$dtype"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --target-modules "$target_modules"
    --max-seq-length "$width" --eval-cadence 1
    --work-dir "$work_dir" --cuda 0
  )
  _print_cmd "${cmd[@]}"

  local -a exec_cmd=("${cmd[@]}")
  if [ "$LORA_BIAS_AB_DRY_RUN" = "1" ]; then
    exec_cmd=("$DRY_RUN_STUB_DIR/fake_bench.sh" "$steps" "$model" "$batch" "$width" "$dtype")
  fi

  local rc=0
  JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$disable" "${exec_cmd[@]}" > "$out_json" 2> "$err_file" || rc=$?

  local status="ok" reason="" wall="null"
  if [ "$rc" -ne 0 ]; then
    status="invalid"
    reason="finetune-run exited $rc -- see $err_file"
  else
    local venv_err
    if ! venv_err="$(_validate_report_envelope "$out_json" 2>&1)"; then
      status="invalid"
      reason="$venv_err"
    else
      local w
      if w="$(_wall_s "$out_json" 2>/dev/null)"; then
        wall="$w"
      else
        status="invalid"
        reason="could not read tiers.finetune_run.train_run_wall_s"
      fi
    fi
  fi

  local extra_disable_json="[]"
  if [ -n "$LORA_BIAS_AB_EXTRA_DISABLE" ]; then
    extra_disable_json="$(python3 -c 'import json,sys; print(json.dumps([x for x in sys.argv[1].split(",") if x]))' "$LORA_BIAS_AB_EXTRA_DISABLE")"
  fi

  local argv_json
  argv_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${cmd[@]}")"

  local row_json
  row_json="$(python3 -c '
import json, sys
(leg_id, model, shape_id, batch, width, dtype, arm, steps, repeat, disable, extra_disable_json,
 argv_json, rc, wall, status, reason, git_sha, box, dry_run, out_json, err_file) = sys.argv[1:]
row = {
    "leg_id": leg_id,
    "model": model,
    "shape": shape_id,
    "batch": int(batch),
    "width": int(width),
    "dtype": dtype,
    "arm": arm,
    "steps": int(steps),
    "repeat": int(repeat),
    "env": {"JAMMI_KERNELS_STRICT": "1", "JAMMI_KERNELS_DISABLE": disable},
    "extra_disable": json.loads(extra_disable_json),
    "argv": json.loads(argv_json),
    "rc": int(rc),
    "wall_s": None if wall == "null" else float(wall),
    "status": status,
    "reason": reason,
    "git_sha": git_sha,
    "box": box,
    "dry_run": dry_run == "true",
    "report_path": "raw/" + leg_id + ".json",
    "stderr_path": "raw/" + leg_id + ".stderr",
}
print(json.dumps(row))
' "$leg_id" "$model" "$shape_id" "$batch" "$width" "$dtype" "$arm" "$steps" "$repeat" \
    "$disable" "$extra_disable_json" "$argv_json" "$rc" "$wall" "$status" "$reason" \
    "$SHA" "$BOX" "$([ "$LORA_BIAS_AB_DRY_RUN" = "1" ] && echo true || echo false)" \
    "$out_json" "$err_file")"

  if ! _append_manifest_row "$row_json"; then
    echo "::error::$leg_id: could not append to $MANIFEST -- this leg's result is unrecoverable; aborting the sweep now rather than continuing silently unrecorded." >&2
    exit 1
  fi

  if [ "$status" != "ok" ]; then
    echo "::warning::$leg_id: INVALID -- $reason (recorded in $MANIFEST; sweep continues)." >&2
  fi
}

# One (model, shape, arm, repeat) cell -- both its N-step and M-step runs.
run_cell() {
  local model="$1" shape_id="$2" batch="$3" width="$4" dtype="$5" \
        target_modules="$6" arm="$7" repeat="$8" n_file="$9" full_file="${10}"
  run_leg "$model" "$shape_id" "$batch" "$width" "$dtype" "$target_modules" "$arm" "$repeat" "$STEPS_N" "$n_file"
  run_leg "$model" "$shape_id" "$batch" "$width" "$dtype" "$target_modules" "$arm" "$repeat" "$STEPS_M" "$full_file"
}

_cell_selected() {
  local cell_id="$1"
  if [ -z "$LORA_BIAS_AB_LEGS_ONLY" ]; then
    return 0
  fi
  case ",$LORA_BIAS_AB_LEGS_ONLY," in
    *",$cell_id,"*) return 0 ;;
    *) return 1 ;;
  esac
}

# --- LORA_BIAS_AB_LEGS_ONLY validation (before any leg runs): every named
# cell id must be a KNOWN cell -- a typo refuses loudly rather than
# silently running a smaller-than-intended sweep.
KNOWN_CELL_IDS=()
for model_spec in "bert:$BERT_FULL" "distilbert:$DISTIL_FULL"; do
  model="${model_spec%%:*}"
  for shape_spec in "$SHAPE_WIRE" "$SHAPE_CHAPTER"; do
    shape_id="${shape_spec%%:*}"
    for arm in fused lora_eager; do
      for r in $(seq 1 "$LORA_BIAS_AB_REPEATS"); do
        KNOWN_CELL_IDS+=("${model}-${shape_id}-${arm}-r${r}")
      done
    done
  done
  for r in $(seq 1 "$LORA_BIAS_AB_REPEATS"); do
    KNOWN_CELL_IDS+=("${model}-control-r${r}")
  done
done
if [ -n "$LORA_BIAS_AB_LEGS_ONLY" ]; then
  IFS=',' read -r -a _requested <<< "$LORA_BIAS_AB_LEGS_ONLY"
  _unknown=()
  for req in "${_requested[@]}"; do
    _known=0
    for k in "${KNOWN_CELL_IDS[@]}"; do
      if [ "$req" = "$k" ]; then _known=1; break; fi
    done
    [ "$_known" -eq 0 ] && _unknown+=("$req")
  done
  if [ "${#_unknown[@]}" -gt 0 ]; then
    echo "::error::LORA_BIAS_AB_LEGS_ONLY names unknown cell id(s): ${_unknown[*]} -- known ids: ${KNOWN_CELL_IDS[*]}" >&2
    exit 2
  fi
fi

# =====================================================================
# The sweep: main fused/lora_eager cells (order-balanced across repeats),
# then the per-model negative-control series.
# =====================================================================
for model_spec in "bert:$BERT_FULL" "distilbert:$DISTIL_FULL"; do
  model="${model_spec%%:*}"
  target_modules="${model_spec#*:}"

  for shape_spec in "$SHAPE_WIRE" "$SHAPE_CHAPTER"; do
    shape_id="${shape_spec%%:*}"
    batch="$(printf '%s' "$shape_spec" | cut -d: -f2)"
    width="$(printf '%s' "$shape_spec" | cut -d: -f3)"
    dtype="$(printf '%s' "$shape_spec" | cut -d: -f4)"

    _corpus_for "$model" "$shape_id" "$width" "$batch"
    n_file="$CORPUS_N_FILE"
    full_file="$CORPUS_FULL_FILE"

    for r in $(seq 1 "$LORA_BIAS_AB_REPEATS"); do
      if [ $(( r % 2 )) -eq 1 ]; then
        order=(fused lora_eager)
      else
        order=(lora_eager fused)
      fi
      for arm in "${order[@]}"; do
        cell_id="${model}-${shape_id}-${arm}-r${r}"
        if ! _cell_selected "$cell_id"; then
          echo "::notice::skipping $cell_id (not in LORA_BIAS_AB_LEGS_ONLY)"
          continue
        fi
        run_cell "$model" "$shape_id" "$batch" "$width" "$dtype" "$target_modules" "$arm" "$r" "$n_file" "$full_file"
      done
    done
  done

  # --- negative control: fused-vs-fused at the wire shape only.
  control_shape_id="${CONTROL_SHAPE%%:*}"
  control_batch="$(printf '%s' "$CONTROL_SHAPE" | cut -d: -f2)"
  control_width="$(printf '%s' "$CONTROL_SHAPE" | cut -d: -f3)"
  control_dtype="$(printf '%s' "$CONTROL_SHAPE" | cut -d: -f4)"
  _corpus_for "$model" "$control_shape_id" "$control_width" "$control_batch"
  control_n="$CORPUS_N_FILE"
  control_full="$CORPUS_FULL_FILE"
  for r in $(seq 1 "$LORA_BIAS_AB_REPEATS"); do
    cell_id="${model}-control-r${r}"
    if ! _cell_selected "$cell_id"; then
      echo "::notice::skipping $cell_id (not in LORA_BIAS_AB_LEGS_ONLY)"
      continue
    fi
    run_cell "$model" "$control_shape_id" "$control_batch" "$control_width" "$control_dtype" \
      "$target_modules" "control" "$r" "$control_n" "$control_full"
  done
done

echo "lora_bias_ab: done -- artifacts under $OUT_DIR (raw legs + $MANIFEST)"
