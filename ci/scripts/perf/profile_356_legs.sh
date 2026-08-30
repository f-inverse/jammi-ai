#!/usr/bin/env bash
# The unit-356 close-out profile leg driver (P4, docs-ci domain; CONTRACT
# `scratchpad/contract-356-profile.md` v4). Drives the 14 pre-registered
# legs (7 per model x {bert-base-uncased, distilbert-base-uncased} --
# CONTRACT's own legs table) two `jammi-bench finetune-run` invocations per
# leg (N steps, then M steps -- the SAME declared workload, only the
# train-corpus row count differs), each traced with `nsys profile
# --trace=cuda`, exported to sqlite, and differenced with this directory's
# own `kernel_census.py`.
#
# THIS SCRIPT IS A GPU-POD DRIVER, not a CI step: real legs need an A100 (or
# equivalent), an installed `nsys` (CONTRACT: 2025.3.2), and two real
# checkpoint directories (`$MODEL_DIR_BERT`/`$MODEL_DIR_DISTILBERT`).
# `PROFILE_356_LEGS_DRY_RUN=1` makes the WHOLE pipeline safe to exercise
# hermetically (every `nsys`/`$BENCH_BIN` invocation is printed via
# `run_cmd`, never executed; each run's own JSON is a small stub carrying
# just enough shape -- `train_run_wall_s` and the LoRA counter fields --
# for the wall-denominator/counter-stamping arithmetic below to run
# end-to-end against real, if fabricated, numbers).
#
# PRECONDITION GUARD (round-3 contract pressure-test, v4): this script
# REFUSES, before any leg runs, unless the bench binary carries THREE
# things P1 (bench, in flight) lands separately:
#   (a) the `finetune-run` DistilBERT dispatch arm (today: `finetune_run.rs`
#       matches only `"modernbert"`/`"bert"`, refusing every other
#       `model_type` with "unsupported model_type '<other>'" -- see this
#       script's own `preflight_probe`).
#   (b) `train_run_wall_s` on `FinetuneRunTier` (today: absent -- the
#       WITHDRAWN `s_per_step_p50`/`s_per_step_mean` spec lived on
#       `FinetuneStepTier` only, never this tier).
#   (c) a `--layers-to-transform` CLI flag on `finetune-run` (today:
#       absent -- `FinetuneRunParams` has no such field at all, and
#       `finetune_run.rs`'s own `LoraBuildConfig`/`FineTuneConfig`
#       construction hardcodes `layers_to_transform: None` unconditionally,
#       so even a caller that could pass the flag would have it silently
#       dropped internally).
# Each is probed CHEAPLY (CPU-hermetic dry finetune-run + a `--help` scan),
# never requiring a GPU or a full training run just to check readiness.
#
# STEP-COUNT PIN (round-3 contract pressure-test, v4): N=100, M=600 for
# every non-E1 leg -- both multiples of 50 (`trainer.rs`'s own norm-check
# cadence), M>N with a wide (500-step) separation for a clean per-step
# average. E1 is the one leg that cannot take this pair: its train corpus
# is the REAL, network-provisioned `train_pairs.jsonl` (CONTRACT P2 --
# 1372 committed-hash-verified pairs), and `--epochs 1` is pinned for
# EVERY leg (`## Declared workload`), so E1's own max step count in one
# epoch at its B=32 is `floor(1372/32) = 42` -- strictly less than even
# N=100. E1 therefore uses its OWN smaller, corpus-bounded (N,M) pair
# (default 10/40, both well under the 42-step ceiling), NOT the 100/600
# pin -- CONTRACT MISMATCH, flagged rather than silently forced: applying
# the global 100/600 pin to E1 would either be impossible (fewer real rows
# exist than the pin needs) or would require repeating rows across
# multiple epochs, contradicting the pinned `--epochs 1` differencing
# argument (`## Method`/`### Census`'s "epochs 1 also kills the
# loss-dependent checkpoint_best divergence"). Overridable via
# `PROFILE_356_E1_STEPS_N`/`PROFILE_356_E1_STEPS_M`.
#
# WALL DENOMINATOR (v4): each run's own `train_run_wall_s` (P1) is read
# from its JSON report and passed to `kernel_census.py` as `--wall-a`/
# `--wall-b`, which derives `wall_s_per_step = (wall_b-wall_a)/(M-N)` --
# the same (M-N)-step differencing this script already applies to the
# kernel trace, applied to the wall clock too.
#
# CHAIN-ATTRIBUTION EXCLUSION (v4): E1 is variable-width (`BatchLongest`
# over the real fixture) -- `### Attribution`'s element-count/shape
# signatures assume a FIXED width, so E1's own kernel_census invocation
# passes `--excluded-from-chain-attribution`; E1's role is the ecological
# wall anchor, the LoRA counter check, and the width report
# (`fixture_width_report.py`), never a chain share.
#
# Env vars (all required for a real run; DRY_RUN relaxes all but OUT_DIR):
#   NSYS_BIN               path to the nsys binary (default: CONTRACT's
#                          pinned `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys`)
#   BENCH_BIN              path to the jammi-bench binary (default:
#                          `$CARGO_TARGET_DIR/release/jammi-bench` or
#                          `$REPO_ROOT/target/release/jammi-bench`)
#   MODEL_DIR_BERT         bert-base-uncased checkpoint dir
#   MODEL_DIR_DISTILBERT   distilbert-base-uncased checkpoint dir
#   OUT_DIR                output directory for every leg's artifacts
#                          (default: `$REPO_ROOT/.profile-356-legs/<ts>`)
#   PROFILE_356_LEGS_DRY_RUN   "1" prints every command, executes nothing
#                              (default "0")
#   PROFILE_356_E1_STEPS_N/M   E1's own (N,M) pair (default 10/40 -- see
#                              "STEP-COUNT PIN" above)
#   PROFILE_356_LEGS_ONLY      optional comma-separated leg-id filter
#                              (e.g. "bert-A1,distilbert-E1") -- default
#                              empty means every one of the 14 legs.
#
# Hermetic self-test: `python3 ci/scripts/perf/test_profile_356_legs_dry_run.py`
# drives this script under PROFILE_356_LEGS_DRY_RUN=1 end to end.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

PROFILE_356_LEGS_DRY_RUN="${PROFILE_356_LEGS_DRY_RUN:-0}"
NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/2025.3.2/bin/nsys}"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BENCH_BIN="${BENCH_BIN:-$TARGET_DIR/release/jammi-bench}"
MODEL_DIR_BERT="${MODEL_DIR_BERT:-}"
MODEL_DIR_DISTILBERT="${MODEL_DIR_DISTILBERT:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.profile-356-legs/$TS}"
PROFILE_356_E1_STEPS_N="${PROFILE_356_E1_STEPS_N:-10}"
PROFILE_356_E1_STEPS_M="${PROFILE_356_E1_STEPS_M:-40}"
PROFILE_356_LEGS_ONLY="${PROFILE_356_LEGS_ONLY:-}"

mkdir -p "$OUT_DIR"

if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
  if [ -z "$MODEL_DIR_BERT" ] || [ -z "$MODEL_DIR_DISTILBERT" ]; then
    echo "::error::MODEL_DIR_BERT and MODEL_DIR_DISTILBERT must both be set for a real run." >&2
    exit 2
  fi
  for f in "$NSYS_BIN" "$BENCH_BIN"; do
    if [ ! -x "$f" ]; then
      echo "::error::$f is not an executable file -- refusing before any leg runs." >&2
      exit 2
    fi
  done
else
  MODEL_DIR_BERT="${MODEL_DIR_BERT:-/root/checkpoints/bert-base-uncased-DRY-RUN-PLACEHOLDER}"
  MODEL_DIR_DISTILBERT="${MODEL_DIR_DISTILBERT:-/root/checkpoints/distilbert-base-uncased-DRY-RUN-PLACEHOLDER}"
  echo "::warning::PROFILE_356_LEGS_DRY_RUN=1 -- nothing is read from MODEL_DIR_*/NSYS_BIN/BENCH_BIN."
fi

# --- state-changing command wrapper (same shape as finetune_run_ab.sh's
# own run_cmd): always echoes what it would run; under DRY_RUN never
# executes.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# --- provenance cross-check (unification contract C5.1), same shape as
# finetune_run_ab.sh/fa2_ab.sh/encode_ab.sh/stacked_sweep.sh/
# clip_artifact_producer.sh: refuse BEFORE any leg runs if the binary's own
# baked identity does not match the sha this checkout is actually at.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BENCH_BIN" provenance 2>&1)" || { echo "::error::'$BENCH_BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BENCH_BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BENCH_BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# =====================================================================
# Precondition guard (P1 cross-check) -- see module doc.
# =====================================================================
preflight_probe() {
  if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then
    echo "::notice::PROFILE_356_LEGS_DRY_RUN=1 -- skipping the real preflight probe (nothing to probe)."
    return 0
  fi

  local missing=()

  # (c) --layers-to-transform CLI flag.
  if ! "$BENCH_BIN" finetune-run --help 2>&1 | grep -q -- '--layers-to-transform'; then
    missing+=("(c) 'finetune-run --help' has no --layers-to-transform flag (FinetuneRunParams/FineTuneConfig hardcode layers_to_transform: None today)")
  fi

  # (a)+(b): one minimal CPU-hermetic dry finetune-run against
  # MODEL_DIR_DISTILBERT -- cheap (1 pair, 1 epoch, no --cuda), reused to
  # probe BOTH the distilbert dispatch arm and the train_run_wall_s field
  # in one shot.
  local probe_dir; probe_dir="$(mktemp -d)"
  local probe_corpus="$probe_dir/probe.jsonl"
  python3 "$DIR/gen_fixed_width_corpus.py" --rows 1 --min-wordpieces 4 --seed 1 --out "$probe_corpus" >/dev/null
  local probe_ids="$probe_dir/probe_ids.txt"
  local anchor_id; anchor_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["anchor_id"])')"
  local positive_id; positive_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["positive_id"])')"
  local negative_id; negative_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["negative_id"])')"
  printf '%s\t%s\t%s\n' "$anchor_id" "$positive_id" "$negative_id" > "$probe_ids"
  local probe_work; probe_work="$probe_dir/work"
  mkdir -p "$probe_work"
  local probe_out="$probe_dir/probe.json"
  local probe_err="$probe_dir/probe.stderr"
  local probe_rc=0
  "$BENCH_BIN" finetune-run \
    --model-dir "$MODEL_DIR_DISTILBERT" --arm fused \
    --train-jsonl "$probe_corpus" --heldout-ids "$probe_ids" --heldout-jsonl "$probe_corpus" \
    --seed 42 --epochs 1 --batch 1 --objective mnrl \
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1 \
    --early-stopping-patience 10000 --backbone-dtype f32 \
    --work-dir "$probe_work" \
    > "$probe_out" 2> "$probe_err" || probe_rc=$?

  if [ "$probe_rc" -ne 0 ]; then
    if grep -q "unsupported model_type 'distilbert'" "$probe_err"; then
      missing+=("(a) finetune-run refuses model_type 'distilbert' (dispatch arm not landed)")
    else
      missing+=("(a)/(b) probe finetune-run FAILED for a reason other than the known distilbert gap (exit $probe_rc) -- cannot verify preconditions; see $probe_err")
    fi
  else
    if ! python3 -c 'import json,sys; sys.exit(0 if "train_run_wall_s" in json.load(open(sys.argv[1])) else 1)' "$probe_out"; then
      missing+=("(b) the probe finetune-run's own JSON report has no train_run_wall_s field")
    fi
  fi
  rm -rf "$probe_dir"

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::profile_356_legs: refusing -- precondition(s) not yet landed (P1, in flight):" >&2
    local m
    for m in "${missing[@]}"; do
      echo "  - $m" >&2
    done
    exit 1
  fi
  echo "::notice::profile_356_legs: preflight OK -- distilbert arm dispatches, train_run_wall_s present, --layers-to-transform available."
}

preflight_probe

# =====================================================================
# Leg table (CONTRACT legs table, 7 legs x 2 models = 14) --
# "id|model|B|W|dtype|target_modules|layers_to_transform|corpus_mode|n_steps|m_steps"
# corpus_mode: "synthetic" (gen_fixed_width_corpus.py) or "heldout" (E1
# only -- the committed/provisioned real fixture).
# =====================================================================
BERT_FULL="query,key,value,dense"
BERT_ONE="query"
DISTIL_FULL="q_lin,k_lin,v_lin,out_lin,lin1,lin2"
DISTIL_ONE="q_lin"

LEGS=(
  "bert-A1|bert|8|512|f32|$BERT_FULL||synthetic|100|600"
  "bert-A2|bert|8|512|bf16|$BERT_FULL||synthetic|100|600"
  "bert-A3|bert|32|64|bf16|$BERT_FULL||synthetic|100|600"
  "bert-A4|bert|32|64|f32|$BERT_FULL||synthetic|100|600"
  "bert-N1|bert|8|512|f32|$BERT_ONE|0|synthetic|100|600"
  "bert-N3|bert|32|64|bf16|$BERT_ONE|0|synthetic|100|600"
  "bert-E1|bert|32|64|bf16|$BERT_FULL||heldout|$PROFILE_356_E1_STEPS_N|$PROFILE_356_E1_STEPS_M"
  "distilbert-A1|distilbert|8|512|f32|$DISTIL_FULL||synthetic|100|600"
  "distilbert-A2|distilbert|8|512|bf16|$DISTIL_FULL||synthetic|100|600"
  "distilbert-A3|distilbert|32|64|bf16|$DISTIL_FULL||synthetic|100|600"
  "distilbert-A4|distilbert|32|64|f32|$DISTIL_FULL||synthetic|100|600"
  "distilbert-N1|distilbert|8|512|f32|$DISTIL_ONE|0|synthetic|100|600"
  "distilbert-N3|distilbert|32|64|bf16|$DISTIL_ONE|0|synthetic|100|600"
  "distilbert-E1|distilbert|32|64|bf16|$DISTIL_FULL||heldout|$PROFILE_356_E1_STEPS_N|$PROFILE_356_E1_STEPS_M"
)

FIXTURE_DIR="$REPO_ROOT/cookbook/fixtures/finetune_heldout"
HELDOUT_IDS="$FIXTURE_DIR/heldout_ids.txt"
HELDOUT_JSONL="$FIXTURE_DIR/heldout_pairs.jsonl"
TRAIN_JSONL_REAL="${TRAIN_JSONL:-$FIXTURE_DIR/train_pairs.jsonl}"

# One nsys-traced finetune-run: writes "$out_json" (the run's own JSON
# report) and "$out_sqlite" (the nsys sqlite export). Returns 0 always
# (DRY_RUN or a real run) -- a leg's own run failure is a leg-level
# refusal decided by the caller, not a script-wide abort.
run_traced() {
  local model_dir="$1" train_jsonl="$2" heldout_ids="$3" heldout_jsonl="$4" \
        seed="$5" batch="$6" dtype="$7" target_modules="$8" layers_to_transform="$9" \
        work_dir="${10}" run_prefix="${11}" out_json="${12}" out_sqlite="${13}"

  mkdir -p "$work_dir"
  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused
    --train-jsonl "$train_jsonl" --heldout-ids "$heldout_ids" --heldout-jsonl "$heldout_jsonl"
    --seed "$seed" --epochs 1 --batch "$batch" --objective mnrl
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$dtype"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --target-modules "$target_modules"
    --work-dir "$work_dir" --cuda 0
  )
  if [ -n "$layers_to_transform" ]; then
    cmd+=(--layers-to-transform "$layers_to_transform")
  fi

  if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then
    run_cmd "$NSYS_BIN" profile --trace=cuda -o "$run_prefix" --force-overwrite=true -- "${cmd[@]}"
    run_cmd "$NSYS_BIN" export --type=sqlite --output="$out_sqlite" --force-overwrite=true "$run_prefix.nsys-rep"
    printf '{"tool":"dry-run","profile_356_dry_run":true,"train_run_wall_s":1.0,"lora_linear_eager_dispatches":1,"lora_linear_fused_dispatches":0,"lora_epilogue_eager_dispatches":1,"lora_epilogue_fused_dispatches":0}' > "$out_json"
    : > "$out_sqlite"
    return 0
  fi

  run_cmd "$NSYS_BIN" profile --trace=cuda -o "$run_prefix" --force-overwrite=true -- "${cmd[@]}" > "$out_json" 2> "$run_prefix.stderr"
  run_cmd "$NSYS_BIN" export --type=sqlite --output="$out_sqlite" --force-overwrite=true "$run_prefix.nsys-rep"
}

_wall_s() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_run_wall_s"])' "$1"
}

_lora_counters() {
  python3 -c 'import json,sys
d = json.load(open(sys.argv[1]))
print(json.dumps({k: d[k] for k in (
    "lora_linear_eager_dispatches", "lora_linear_fused_dispatches",
    "lora_epilogue_eager_dispatches", "lora_epilogue_fused_dispatches") if k in d}))' "$1"
}

run_leg() {
  local spec="$1"
  IFS='|' read -r leg_id model batch width dtype target_modules layers_to_transform corpus_mode n_steps m_steps <<< "$spec"

  if [ -n "$PROFILE_356_LEGS_ONLY" ]; then
    case ",$PROFILE_356_LEGS_ONLY," in
      *",$leg_id,"*) ;;
      *) echo "::notice::skipping $leg_id (not in PROFILE_356_LEGS_ONLY)"; return 0 ;;
    esac
  fi

  echo "=== leg $leg_id: model=$model batch=$batch width=$width dtype=$dtype target=$target_modules layers_to_transform=${layers_to_transform:-<none>} corpus=$corpus_mode steps=$n_steps/$m_steps ==="

  local model_dir
  if [ "$model" = "bert" ]; then model_dir="$MODEL_DIR_BERT"; else model_dir="$MODEL_DIR_DISTILBERT"; fi

  local leg_dir="$OUT_DIR/$leg_id"
  mkdir -p "$leg_dir"

  local train_n train_m heldout_ids heldout_jsonl
  if [ "$corpus_mode" = "synthetic" ]; then
    local full_corpus="$leg_dir/corpus_m${m_steps}.jsonl"
    local rows_m=$(( batch * m_steps ))
    local rows_n=$(( batch * n_steps ))
    if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
      python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" --min-wordpieces "$width" --seed 42 --out "$full_corpus"
      head -n "$rows_n" "$full_corpus" > "$leg_dir/corpus_n${n_steps}.jsonl"
    else
      run_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" --min-wordpieces "$width" --seed 42 --out "$full_corpus"
      : > "$full_corpus"
      : > "$leg_dir/corpus_n${n_steps}.jsonl"
    fi
    train_n="$leg_dir/corpus_n${n_steps}.jsonl"
    train_m="$full_corpus"
    heldout_ids="$HELDOUT_IDS"
    heldout_jsonl="$HELDOUT_JSONL"
  else
    # E1: the REAL, network-provisioned + byte-verified train_pairs.jsonl
    # (CONTRACT P2) -- pre-run provisioning is this script's caller's job
    # (mirrors finetune_run_ab.sh's own PRE-RUN PROVISIONING step); this
    # driver only refuses if the file is absent/fails verification.
    if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
      python3 "$DIR/verify_train_pairs.py" --pairs "$TRAIN_JSONL_REAL" \
        || { echo "::error::$leg_id: $TRAIN_JSONL_REAL failed byte-verification -- refusing this leg." >&2; return 1; }
      head -n "$(( batch * n_steps ))" "$TRAIN_JSONL_REAL" > "$leg_dir/corpus_n${n_steps}.jsonl"
      head -n "$(( batch * m_steps ))" "$TRAIN_JSONL_REAL" > "$leg_dir/corpus_m${m_steps}.jsonl"
    else
      : > "$leg_dir/corpus_n${n_steps}.jsonl"
      : > "$leg_dir/corpus_m${m_steps}.jsonl"
    fi
    train_n="$leg_dir/corpus_n${n_steps}.jsonl"
    train_m="$leg_dir/corpus_m${m_steps}.jsonl"
    heldout_ids="$HELDOUT_IDS"
    heldout_jsonl="$HELDOUT_JSONL"
  fi

  local out_n="$leg_dir/run_n.json" out_m="$leg_dir/run_m.json"
  local sqlite_n="$leg_dir/run_n.sqlite" sqlite_m="$leg_dir/run_m.sqlite"

  run_traced "$model_dir" "$train_n" "$heldout_ids" "$heldout_jsonl" 42 "$batch" "$dtype" \
    "$target_modules" "$layers_to_transform" "$leg_dir/work_n" "$leg_dir/run_n" "$out_n" "$sqlite_n"
  run_traced "$model_dir" "$train_m" "$heldout_ids" "$heldout_jsonl" 42 "$batch" "$dtype" \
    "$target_modules" "$layers_to_transform" "$leg_dir/work_m" "$leg_dir/run_m" "$out_m" "$sqlite_m"

  local census_json="$leg_dir/census.json"
  local -a census_cmd=(
    python3 "$DIR/kernel_census.py" "$sqlite_n" "$sqlite_m" "$n_steps" "$m_steps" "$census_json"
  )
  if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
    local wall_n wall_m
    wall_n="$(_wall_s "$out_n")"
    wall_m="$(_wall_s "$out_m")"
    census_cmd+=(--wall-a "$wall_n" --wall-b "$wall_m")
  fi
  case "$leg_id" in
    *-E1) census_cmd+=(--excluded-from-chain-attribution) ;;
  esac
  run_cmd "${census_cmd[@]}" || echo "::warning::$leg_id: kernel_census.py FAILED -- leg INVALID, recorded, sweep continues." >&2

  # Per-leg provenance/LoRA-counter stamp (CONTRACT: "Stamp per leg: git
  # sha, box, driver, nsys version, dtype, leg id, LoRA counter fields
  # pulled from the tier report").
  local nsys_version="unknown"
  if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
    nsys_version="$("$NSYS_BIN" --version 2>&1 | head -1)"
  fi
  local lora_n="{}" lora_m="{}"
  if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ] && [ -s "$out_n" ] && [ -s "$out_m" ]; then
    lora_n="$(_lora_counters "$out_n")"
    lora_m="$(_lora_counters "$out_m")"
  fi
  python3 -c '
import json, sys
leg_id, git_sha, box, nsys_version, dtype, lora_n, lora_m, out = sys.argv[1:]
manifest = {
    "leg_id": leg_id,
    "git_sha": git_sha,
    "box": box,
    "driver": "ci/scripts/perf/profile_356_legs.sh",
    "nsys_version": nsys_version,
    "dtype": dtype,
    "lora_counters_n_run": json.loads(lora_n),
    "lora_counters_m_run": json.loads(lora_m),
}
json.dump(manifest, open(out, "w"), indent=1)
' "$leg_id" "$SHA" "$(hostname)" "$nsys_version" "$dtype" "$lora_n" "$lora_m" "$leg_dir/manifest.json"

  # E1's own width evidence (Artifacts: "the E1 width histogram produced
  # OFFLINE by the tracked producer re-tokenizing the fixture in the same
  # anchors+positives join and batch order"). Runs against the COMMITTED
  # heldout_pairs.jsonl (never the synthetic corpus) regardless of model,
  # since the fixture text itself is model-independent -- only the
  # tokenizer differs, and the caller supplies model_dir's own
  # tokenizer.json.
  case "$leg_id" in
    *-E1)
      if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
        python3 "$DIR/fixture_width_report.py" "$HELDOUT_JSONL" \
          --tokenizer "$model_dir/tokenizer.json" --cap "$width" --batch-size "$batch" \
          --out "$leg_dir/width_report.json" \
          || echo "::warning::$leg_id: fixture_width_report.py FAILED (see module doc's 'degrade LOUDLY' clause) -- no width evidence for this leg." >&2
      fi
      ;;
  esac
}

for spec in "${LEGS[@]}"; do
  run_leg "$spec"
done

echo "profile_356_legs: done -- artifacts under $OUT_DIR"
