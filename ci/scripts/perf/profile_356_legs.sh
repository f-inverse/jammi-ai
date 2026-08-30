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
# hermetically: `$NSYS_BIN`/`$BENCH_BIN` are swapped for hermetic fake
# stand-ins generated into `$DRY_RUN_STUB_DIR` (see that variable's own
# doc) and ACTUALLY EXECUTED (never a hand-shaped bypass) through the
# EXACT SAME capture path a real leg uses -- `_print_cmd` still prints the
# real, would-be production command line for operator visibility.
#
# PRECONDITION GUARD (round-3 contract pressure-test, v4): this script
# REFUSES, before any leg runs, unless the bench binary carries THREE
# things `bench/356-finetune-run-distilbert` lands in the SAME atomic PR
# (B6) as this producer -- a docs-ci branch cannot itself carry crate
# changes, so this preflight, not a doc claim, is the mechanical proof the
# LANDED tree actually contains all three:
#   (a) the `finetune-run` DistilBERT dispatch arm (`finetune_run.rs`'s
#       `model_type` match previously admitted only `"modernbert"`/`"bert"`,
#       refusing every other value with "unsupported model_type '<other>'").
#   (b) `train_run_wall_s` on `FinetuneRunTier` (previously only
#       `FinetuneStepTier` carried a wall-clock pair -- the WITHDRAWN
#       `s_per_step_p50`/`s_per_step_mean` spec).
#   (c) a `--layers-to-transform` CLI flag on `finetune-run` (previously
#       absent entirely -- `FinetuneRunParams` had no such field, and
#       `finetune_run.rs`'s own `LoraBuildConfig`/`FineTuneConfig`
#       construction hardcoded `layers_to_transform: None` unconditionally).
# Each is probed CHEAPLY (a CPU-hermetic dry finetune-run + a `--help`
# scan), never requiring a GPU or a full training run just to check
# readiness -- and never merely trusted because a sibling PR claims to
# have landed it. The probe uses ARCHITECTURE-CORRECT LoRA selectors
# (`$DISTIL_FULL`, the same selectors the real DistilBERT legs use) --
# `finetune_run.rs`'s own unconditional zero-trainable-LoRA refusal fires
# on ANY `target_modules` that matches nothing on the probed architecture
# (the CLI's own default, `Wqkv,Wo,Wi`, is ModernBERT-only and matches
# nothing on DistilBERT), so an unqualified probe would refuse on THAT
# gap and never reach the distilbert-dispatch/wall-field checks it exists
# to make. `PROFILE_356_LEGS_PREFLIGHT_ONLY=1` runs the preflight probe
# and exits before the leg sweep -- the hook a hermetic test drives
# against a fake `$BENCH_BIN` stub to exercise each of the four
# distinguishable outcomes (pass; (a) distilbert-arm missing; (b)
# wall-field missing; (c) flag missing; a genuinely-unrelated failure)
# without a GPU or a real checkpoint.
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
# EVERY DECLARED LEG PARAMETER REACHES THE BINARY AND IS RECORDED
# (phase-4 audit CLASS 3): width (`--max-seq-length`), LoRA selectors
# (`--target-modules`), `--layers-to-transform` (N legs only), the fixed
# `--eval-cadence` (`$EVAL_CADENCE`, same value every leg -- "fixed eval
# cadence" cancels in the N/M differencing the same way every other pinned
# flag does), and `dtype` all appear literally in `run_traced`'s own `cmd`
# array; steps N/M are what SIZE the corpus (there is no standalone
# "--steps" flag -- steps are `rows/batch` at `--epochs 1`) and are
# recorded, alongside every other column, in each leg's `manifest.json`
# (`steps_declared`), cross-checked there against the report's own
# MEASURED `steps_measured` (also passed to `kernel_census.py` as
# `--steps-measured-a`/`-b`, its own domain check -- see that module's
# doc). `manifest.json` also carries `status`/`reason`/`census_ok` per leg
# -- "leg INVALID, recorded, sweep continues" is a RECORD, not just a
# warning line: a leg-level failure anywhere in this script never aborts
# the whole sweep, and every leg gets a manifest, always (phase-4 round-2
# audit BLOCK 2 -- see `run_leg`'s own doc for how this is now enforced
# by EXPLICIT guards, not by relying on `set -e`'s errexit-suspension
# quirk alone).
#
# MANIFEST PATH -- TWO SHAPES: the normal, expected one is
# `$OUT_DIR/<leg_id>/manifest.json` (this leg's own subdirectory,
# alongside its corpus/run/census files). If `mkdir -p` for that
# subdirectory itself fails (round-2 audit BLOCK 2's own "write a minimal
# manifest even there"), this producer falls back to a FLAT
# `$OUT_DIR/<leg_id>.manifest.json` instead (no subdirectory, since one
# could not be created) -- the body still carries `leg_id`, so a consumer
# can always identify which leg a manifest belongs to either way, but a
# future reader globbing `*/manifest.json` under `$OUT_DIR` alone would
# silently miss this flat, fallback-shaped one; glob BOTH
# `*/manifest.json` and `*.manifest.json` (or simply `**/*manifest.json`)
# to see every leg's result.
#
# WALL DENOMINATOR (v4): each run's own `train_run_wall_s` is read from
# `tiers.finetune_run` in its JSON report and passed to `kernel_census.py`
# as `--wall-a`/`--wall-b`, which derives `wall_s_per_step =
# (wall_b-wall_a)/(M-N)` -- the same (M-N)-step differencing this script
# already applies to the kernel trace, applied to the wall clock too.
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
#   PROFILE_356_LEGS_DRY_RUN   "1" swaps in hermetic fake nsys/bench
#                              stand-ins and drives them through the real
#                              capture path (default "0")
#   PROFILE_356_LEGS_PREFLIGHT_ONLY
#                              "1" runs the preflight probe then exits
#                              (before the leg sweep) -- the hook a
#                              hermetic test drives against a fake
#                              `$BENCH_BIN` (default "0")
#   PROFILE_356_E1_STEPS_N/M   E1's own (N,M) pair (default 10/40 -- see
#                              "STEP-COUNT PIN" above)
#   PROFILE_356_LEGS_ONLY      optional comma-separated leg-id filter
#                              (e.g. "bert-A1,distilbert-E1") -- default
#                              empty means every one of the 14 legs.
#
# Hermetic self-tests: `python3 ci/scripts/perf/test_profile_356_legs_dry_run.py`
# drives this script under PROFILE_356_LEGS_DRY_RUN=1 end to end (every
# leg, every reader, the manifest schema, a chatty-fake-nsys stdout-
# pollution arm); its own preflight arm drives
# `PROFILE_356_LEGS_PREFLIGHT_ONLY=1` against a fake `$BENCH_BIN` stub
# covering the pass case and each of the four distinguishable failure
# arms; a fifth arm drives a REAL (non-dry, non-preflight-only) multi-leg
# run against a fake `$NSYS_BIN` whose `--version` deliberately fails,
# asserting the WHOLE sweep still completes (both legs recorded, neither
# one's reason naming the version failure) -- `--version` is captured
# ONCE, guarded, before any leg runs (`$NSYS_VERSION` below), so a failing
# `--version` degrades that ONE recorded string for every leg and aborts
# nothing; it is not a per-leg concept the way it was before this fix.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

PROFILE_356_LEGS_DRY_RUN="${PROFILE_356_LEGS_DRY_RUN:-0}"
PROFILE_356_LEGS_PREFLIGHT_ONLY="${PROFILE_356_LEGS_PREFLIGHT_ONLY:-0}"
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

# The one fixed eval cadence every leg shares (`## Declared workload`'s
# "fixed eval cadence") -- a script-level constant, never a per-leg
# column, referenced directly by `run_traced`/the manifest builder rather
# than threaded through as yet another positional parameter.
EVAL_CADENCE=1

# CONTRACT legs table selector sets (verified: 72 BERT sites = 12x6 via
# `ends_with` matching, 36 DistilBERT = 6x6) -- defined HERE, before
# `preflight_probe`, so the probe itself can use `$DISTIL_FULL` (phase-4
# audit CLASS 2: an unqualified probe -- CLI default `target_modules`,
# ModernBERT-only -- matches nothing on DistilBERT and trips
# `finetune_run.rs`'s own unconditional zero-trainable-LoRA refusal before
# ever reaching the dispatch/wall-field checks the probe exists to make).
BERT_FULL="query,key,value,dense"
BERT_ONE="query"
DISTIL_FULL="q_lin,k_lin,v_lin,out_lin,lin1,lin2"
DISTIL_ONE="q_lin"

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

# `nsys --version` ONCE at startup (phase-4 round-2 audit BLOCK 2):
# previously re-invoked, unguarded, inside `run_leg` for EVERY leg -- a
# failing `nsys --version` (an unguarded plain assignment) aborted the
# WHOLE sweep mid-leg via `set -e`, with no manifest for the failing leg.
# Guarded here, computed once, reused by every leg -- a failure here
# degrades to a recorded "unknown" string, never a script-wide abort.
NSYS_VERSION="unknown"
if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
  NSYS_VERSION="$("$NSYS_BIN" --version 2>&1 | head -1)" || NSYS_VERSION="unknown (nsys --version failed)"
fi

# --- trace-echo helper (phase-4 round-2 audit BLOCK 1(a)): ALWAYS to
# stderr, never stdout -- a trace line sharing stdout with a captured
# child process's own JSON report was exactly how the report-envelope
# capture bug (BLOCK 1(b), see run_traced's own doc) went undetected.
_print_cmd() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

# --- state-changing command wrapper (same shape as finetune_run_ab.sh's
# own run_cmd): always echoes what it would run (to stderr -- see
# `_print_cmd`); under DRY_RUN never executes. Used for invocations whose
# own stdout is never captured into a report file (corpus generation,
# `verify_train_pairs.py`, `kernel_census.py`) -- `run_traced`'s own nsys/
# bench invocation does NOT go through this wrapper; see its own doc for
# why.
run_cmd() {
  _print_cmd "$@"
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
# Precondition guard (P1 cross-check) -- see module doc. Every setup step
# below is explicitly guarded (phase-4 round-2 audit BLOCK 2's "sweep
# preflight_probe too") -- a setup failure here is loud and intentional
# (`exit 1`, a clear message), never a raw, unexplained `set -e` abort;
# aborting the WHOLE script on a genuine preflight-setup failure is
# correct (preflight gates the entire run), the fix here is HOW it aborts,
# not WHETHER it does.
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
  # in one shot. `--target-modules "$DISTIL_FULL"` (phase-4 audit CLASS
  # 2): an unqualified probe would fail on the unconditional
  # zero-trainable-LoRA refusal instead of ever reaching the dispatch
  # check.
  local probe_dir
  if ! probe_dir="$(mktemp -d)"; then
    echo "::error::preflight_probe: mktemp -d failed -- cannot set up the probe corpus." >&2
    exit 1
  fi
  local probe_corpus="$probe_dir/probe.jsonl"
  if ! python3 "$DIR/gen_fixed_width_corpus.py" --rows 1 --min-wordpieces 4 --seed 1 --out "$probe_corpus" >/dev/null; then
    echo "::error::preflight_probe: could not generate the probe corpus at $probe_corpus." >&2
    rm -rf "$probe_dir"
    exit 1
  fi
  local probe_ids="$probe_dir/probe_ids.txt"
  local anchor_id positive_id negative_id
  if ! anchor_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["anchor_id"])')" \
      || ! positive_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["positive_id"])')" \
      || ! negative_id="$(python3 -c 'import json; print(json.loads(open("'"$probe_corpus"'").readline())["negative_id"])')"; then
    echo "::error::preflight_probe: could not parse ids out of the probe corpus." >&2
    rm -rf "$probe_dir"
    exit 1
  fi
  if ! printf '%s\t%s\t%s\n' "$anchor_id" "$positive_id" "$negative_id" > "$probe_ids"; then
    echo "::error::preflight_probe: could not write $probe_ids." >&2
    rm -rf "$probe_dir"
    exit 1
  fi
  local probe_work="$probe_dir/work"
  if ! mkdir -p "$probe_work"; then
    echo "::error::preflight_probe: could not create $probe_work." >&2
    rm -rf "$probe_dir"
    exit 1
  fi
  local probe_out="$probe_dir/probe.json"
  local probe_err="$probe_dir/probe.stderr"
  local probe_rc=0
  "$BENCH_BIN" finetune-run \
    --model-dir "$MODEL_DIR_DISTILBERT" --arm fused \
    --train-jsonl "$probe_corpus" --heldout-ids "$probe_ids" --heldout-jsonl "$probe_corpus" \
    --seed 42 --epochs 1 --batch 1 --objective mnrl \
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1 \
    --early-stopping-patience 10000 --backbone-dtype f32 \
    --target-modules "$DISTIL_FULL" \
    --work-dir "$probe_work" \
    > "$probe_out" 2> "$probe_err" || probe_rc=$?

  if [ "$probe_rc" -ne 0 ]; then
    if grep -q "unsupported model_type 'distilbert'" "$probe_err"; then
      missing+=("(a) finetune-run refuses model_type 'distilbert' (dispatch arm not landed)")
    else
      missing+=("(a)/(b) probe finetune-run FAILED for a reason other than the known distilbert gap (exit $probe_rc) -- cannot verify preconditions; see $probe_err")
    fi
  else
    if ! python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
tier = d.get("tiers", {}).get("finetune_run")
sys.exit(0 if tier is not None and "train_run_wall_s" in tier else 1)
' "$probe_out"; then
      missing+=("(b) the probe finetune-run's own JSON report has no tiers.finetune_run.train_run_wall_s field")
    fi
  fi
  rm -rf "$probe_dir"

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::profile_356_legs: refusing -- precondition(s) not yet landed:" >&2
    local m
    for m in "${missing[@]}"; do
      echo "  - $m" >&2
    done
    exit 1
  fi
  echo "::notice::profile_356_legs: preflight OK -- distilbert arm dispatches, tiers.finetune_run.train_run_wall_s present, --layers-to-transform available."
}

preflight_probe

if [ "$PROFILE_356_LEGS_PREFLIGHT_ONLY" = "1" ]; then
  echo "::notice::PROFILE_356_LEGS_PREFLIGHT_ONLY=1 -- preflight passed, exiting before the leg sweep."
  exit 0
fi

# =====================================================================
# Leg table (CONTRACT legs table, 7 legs x 2 models = 14) --
# "id|model|B|W|dtype|target_modules|layers_to_transform|corpus_mode|n_steps|m_steps"
# corpus_mode: "synthetic" (gen_fixed_width_corpus.py) or "heldout" (E1
# only -- the committed/provisioned real fixture).
# =====================================================================
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
GOLDEN_FIXTURE="$REPO_ROOT/ci/scripts/perf/fixtures/finetune_run_golden/bert_fused.json"

# Hermetic fake nsys/bench stand-ins, used ONLY under
# PROFILE_356_LEGS_DRY_RUN=1 -- generated ONCE, reused for every leg, so
# the DRY_RUN sweep drives the EXACT SAME capture path (the exec-wrapper,
# the stderr redirect, the post-write envelope validation in `run_traced`)
# a real GPU-pod run uses, never a hand-shaped bypass (phase-4 round-2
# audit BLOCK 1's own "CRITICAL": a stub JSON written directly by this
# script, skipping the capture machinery entirely, structurally cannot
# catch a bug IN that machinery). `fake_nsys.sh` is deliberately CHATTY on
# stdout for BOTH its subcommands (mirroring real nsys's own progress
# output) to prove the capture path discards it regardless; `fake_bench.sh`
# emits the golden-derived envelope on ITS OWN stdout (never writes a file
# directly), exercising the exec-wrapper's redirect for real.
DRY_RUN_STUB_DIR=""
if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then
  DRY_RUN_STUB_DIR="$(mktemp -d)"
  trap 'rm -rf "$DRY_RUN_STUB_DIR"' EXIT

  cat > "$DRY_RUN_STUB_DIR/fake_nsys.sh" <<'FAKE_NSYS_EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "fake_nsys: chatty stdout noise on purpose, mirroring real nsys progress output"
if [ "$1" = "export" ]; then
  shift
  out=""
  for a in "$@"; do
    case "$a" in
      --output=*) out="${a#--output=}" ;;
    esac
  done
  echo "fake_nsys: export writing to '$out' (more stdout noise)"
  if [ -n "$out" ]; then : > "$out"; fi
  exit 0
fi
if [ "$1" = "profile" ]; then
  shift
  args=("$@")
  for i in "${!args[@]}"; do
    if [ "${args[$i]}" = "--" ]; then
      rest=("${args[@]:$((i+1))}")
      echo "fake_nsys: launching the traced command (even more stdout noise)"
      exec "${rest[@]}"
    fi
  done
  echo "fake_nsys: no -- separator found in profile args" >&2
  exit 1
fi
echo "fake_nsys: unknown subcommand $1" >&2
exit 1
FAKE_NSYS_EOF
  chmod +x "$DRY_RUN_STUB_DIR/fake_nsys.sh"

  cat > "$DRY_RUN_STUB_DIR/fake_bench.sh" <<'FAKE_BENCH_EOF'
#!/usr/bin/env bash
set -euo pipefail
# $1=steps_measured $2=golden_json_path -- emits the golden-derived Report
# envelope on ITS OWN stdout (never writes a file directly), so the exec-
# wrapper's own "redirect this child's stdout to the report file"
# mechanism is what actually produces the report, for real.
python3 -c '
import copy, json, sys
steps_measured, golden_path = int(sys.argv[1]), sys.argv[2]
golden = json.load(open(golden_path))
tier = copy.deepcopy(golden["tiers"]["finetune_run"])
# Proportional to steps_measured (never a flat constant): the M-step run
# always declares more steps than the N-step run, so this keeps
# wall_m > wall_a > 0 genuinely true across the fake pair, exercising the
# wall-pair domain check in kernel_census.py the same way a real
# same-workload N<M pair would satisfy it.
tier["train_run_wall_s"] = 0.01 * steps_measured
tier["steps_measured"] = steps_measured
tier["lora_linear_eager_dispatches"] = 1
tier["lora_linear_fused_dispatches"] = 0
tier["lora_epilogue_eager_dispatches"] = 1
tier["lora_epilogue_fused_dispatches"] = 0
report = {"tool": "dry-run", "profile_356_dry_run": True, "tiers": {"finetune_run": tier}}
json.dump(report, sys.stdout)
' "$1" "$2"
echo "fake_bench: stderr noise too, never captured into the report" >&2
FAKE_BENCH_EOF
  chmod +x "$DRY_RUN_STUB_DIR/fake_bench.sh"
fi

# Validates "$1" parses as JSON and carries a `tiers.finetune_run` OBJECT
# (phase-4 round-2 audit BLOCK 1(c)) -- called right after the traced
# invocation writes the report, BEFORE any reader (`_wall_s`/
# `_steps_measured`/`_lora_counters`) touches it. A parse failure or a
# missing/wrong-shaped `tiers.finetune_run` is a recorded leg-INVALID
# reason, never a downstream KeyError/TypeError surfacing from one of
# those readers instead.
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

# One nsys-traced finetune-run: writes "$out_json" (the run's own JSON
# report, VALIDATED before return -- see `_validate_report_envelope`) and
# "$out_sqlite" (the nsys sqlite export). Returns the underlying command's
# exit status (0 on success) -- `run_leg` decides what a nonzero return
# means for this leg's own status.
#
# NEVER CAPTURES THE NSYS-WRAPPED PROCESS'S OWN STDOUT AS THE REPORT
# (phase-4 round-2 audit BLOCK 1(b)): the previous
# `run_cmd ... -- "${cmd[@]}" > "$out_json"` redirected run_cmd's OWN
# trace echo AND nsys's OWN progress output AND the bench child's stdout
# all into one file -- `json.load` then failed on every REAL leg (the
# report always started with a `+ nsys profile ...` trace line), which
# every reader below folded into a "leg invalid" status, so a real GPU-pod
# run of this script recorded status=invalid on all 14 legs unconditionally,
# never because of anything the profiled workload itself did. The fix:
# the bench child's stdout is redirected to "$out_json" INSIDE the traced
# invocation via a `bash -c 'exec ... > "$0"'` wrapper, so ONLY that
# child's own stdout ever reaches the report file, regardless of what nsys
# prints to ITS OWN stdout -- `_print_cmd` (stderr-only, see its own doc)
# shows the real, would-be production command line for operator
# visibility, entirely separate from the actual exec below.
#
# EVERY declared leg parameter reaches this cmd array (phase-4 audit
# CLASS 3): `--max-seq-length "$width"` (previously OMITTED entirely --
# clap's own default of 64 silently ran every W=512 leg, including the
# contract's mandatory A1/N1, at W=64) and `--eval-cadence "$EVAL_CADENCE"`
# (previously relied on the CLI's own default; passed explicitly here so
# the fixed value is self-documenting rather than an unstated default that
# could silently change).
run_traced() {
  local model_dir="$1" train_jsonl="$2" heldout_ids="$3" heldout_jsonl="$4" \
        seed="$5" batch="$6" dtype="$7" target_modules="$8" layers_to_transform="$9" \
        width="${10}" steps_this_run="${11}" \
        work_dir="${12}" run_prefix="${13}" out_json="${14}" out_sqlite="${15}"

  if ! mkdir -p "$work_dir"; then
    echo "::error::run_traced: could not create $work_dir" >&2
    return 1
  fi

  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused
    --train-jsonl "$train_jsonl" --heldout-ids "$heldout_ids" --heldout-jsonl "$heldout_jsonl"
    --seed "$seed" --epochs 1 --batch "$batch" --objective mnrl
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$dtype"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --target-modules "$target_modules"
    --max-seq-length "$width"
    --eval-cadence "$EVAL_CADENCE"
    --work-dir "$work_dir" --cuda 0
  )
  if [ -n "$layers_to_transform" ]; then
    cmd+=(--layers-to-transform "$layers_to_transform")
  fi

  # What actually execs -- swapped for the hermetic fakes under DRY_RUN
  # (see $DRY_RUN_STUB_DIR's own doc); $NSYS_BIN/cmd[] (below, via
  # _print_cmd) always show the REAL, would-be production command line
  # regardless of which binaries actually run.
  local exec_nsys_bin="$NSYS_BIN"
  local -a exec_cmd=("${cmd[@]}")
  if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then
    exec_nsys_bin="$DRY_RUN_STUB_DIR/fake_nsys.sh"
    exec_cmd=("$DRY_RUN_STUB_DIR/fake_bench.sh" "$steps_this_run" "$GOLDEN_FIXTURE")
  fi

  local -a wrapped=(bash -c 'exec "$1" "${@:2}" > "$0"' "$out_json" "${exec_cmd[@]}")

  _print_cmd "$NSYS_BIN" profile --trace=cuda -o "$run_prefix" --force-overwrite=true -- "${cmd[@]}"
  local rc=0
  "$exec_nsys_bin" profile --trace=cuda -o "$run_prefix" --force-overwrite=true -- "${wrapped[@]}" \
    2> "$run_prefix.stderr" || rc=$?

  if [ "$rc" -eq 0 ]; then
    _print_cmd "$NSYS_BIN" export --type=sqlite --output="$out_sqlite" --force-overwrite=true "$run_prefix.nsys-rep"
    "$exec_nsys_bin" export --type=sqlite --output="$out_sqlite" --force-overwrite=true "$run_prefix.nsys-rep" \
      2>> "$run_prefix.stderr" || rc=$?
  fi

  if [ "$rc" -eq 0 ]; then
    local validate_err
    if ! validate_err="$(_validate_report_envelope "$out_json" 2>&1)"; then
      # Teed into "$run_prefix.stderr" too (phase-4 round-3 audit item 5):
      # run_leg's own recorded reason points the operator at THIS file for
      # every run_traced failure, including this one -- without the tee,
      # that file would contain only the (silent, rc=0) nsys/bench
      # streams, never the actual envelope-validation cause.
      printf '%s\n' "$validate_err" | tee -a "$run_prefix.stderr" >&2
      echo "::error::run_traced: $out_json failed report-envelope validation -- see $run_prefix.stderr" >&2
      rc=1
    fi
  fi
  return "$rc"
}

# Reads `tiers.finetune_run.train_run_wall_s` -- REFUSES (nonzero exit,
# clear stderr message, no stray stdout) rather than KeyError-ing when
# either the envelope or the field is absent.
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

# Reads `tiers.finetune_run.steps_measured` -- same REFUSE-dont-degrade
# posture as `_wall_s`. Call sites tolerate absence explicitly (`||
# steps_measured_x=""`) since this field is cross-checked "when available"
# (CONTRACT/audit CLASS 4(c)), never required the way `train_run_wall_s`
# is.
_steps_measured() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
tier = d.get("tiers", {}).get("finetune_run")
if tier is None or "steps_measured" not in tier:
    print("::error::_steps_measured: no tiers.finetune_run.steps_measured in " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
print(tier["steps_measured"])
' "$1"
}

# Reads the four LoRA dispatch counters off `tiers.finetune_run` --
# REFUSES (nonzero exit, no stdout) when ANY of the four is absent, rather
# than silently yielding `{}` (phase-4 audit CLASS 1: `{}` is the
# contract's OWN positive-proof channel -- `### Positive proof`'s
# `eager == site_count x batches` equation -- so silently emitting it
# empty on a read failure would make a genuinely un-provable leg look
# identical to a real, if trivially empty, measurement).
_lora_counters() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
tier = d.get("tiers", {}).get("finetune_run")
required = ("lora_linear_eager_dispatches", "lora_linear_fused_dispatches",
            "lora_epilogue_eager_dispatches", "lora_epilogue_fused_dispatches")
if tier is None:
    print("::error::_lora_counters: no tiers.finetune_run in " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
missing = [k for k in required if k not in tier]
if missing:
    print("::error::_lora_counters: missing LoRA counter field(s) " + repr(missing) +
          " in tiers.finetune_run of " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: tier[k] for k in required}))
' "$1"
}

# Reads `fixed_cost_buckets`/`fixed_cost_time_us`/`fixed_cost_jitter_max_rel`
# off a `census.json` (kernel_census.py's own top-level report fields,
# round-1/2 fix -- see that module's own doc) -- REFUSES (nonzero exit, no
# stdout) if any is absent, the same "never degrade to a silent {}" posture
# `_lora_counters` above uses: this is only ever called after
# `census_ok=true` (kernel_census.py exited 0 and wrote a report), so an
# absent field here means a report shape this script does not understand,
# not a legitimately-missing measurement. Surfaced into the per-leg
# manifest (phase-4 audit BLOCK 2, widened round-2 re-audit BLOCK 3(a) to
# include `fixed_cost_jitter_max_rel`) so the emitted-for-visibility claim
# in kernel_census.py's own doc has an actual consumer, not just an
# informational field nothing ever reads.
_census_fixed_cost() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
required = ("fixed_cost_buckets", "fixed_cost_time_us", "fixed_cost_jitter_max_rel")
missing = [k for k in required if k not in d]
if missing:
    print("::error::_census_fixed_cost: missing field(s) " + repr(missing) +
          " in " + sys.argv[1], file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: d[k] for k in required}))
' "$1"
}

# Writes a leg's manifest.json -- callers guard the CALL itself (a write
# failure is sweep-fatal, see `run_leg`'s own doc). Every declared
# leg-table column (width, target_modules, layers_to_transform, dtype,
# eval_cadence, steps_declared) plus the report-measured steps_measured
# (when readable) and the census/LoRA-counter outcome. `git_sha`/`box`/
# `driver`/`nsys_version` are this producer's own per-leg elaboration of
# the contract's own legs-table sentence, "one session, same box, one
# build, binary git-sha stamped per leg" -- that sentence names the
# session-level invariant; it does not itself enumerate `driver`/
# `nsys_version` as separate fields, which this producer adds for its own
# auditability. `dry_run` (phase-4 round-2 audit advisory 1) marks every
# manifest written under `PROFILE_356_LEGS_DRY_RUN=1` so a reader never
# mistakes a hermetic fake-execution manifest for a real GPU-pod result --
# `census_ok`/`status: ok` under DRY_RUN describe the FAKE pipeline
# completing, never a real measurement. `fixed_cost_buckets`/
# `fixed_cost_time_us`/`fixed_cost_jitter_max_rel` (phase-4 audit round-2
# BLOCK 2, widened round-2 re-audit BLOCK 3(a)) surface `kernel_census.py`'s
# own informational fixed-cost tally into the manifest -- all three `null`
# unless `census_ok` (there is no fixed-cost tally without a real census
# report to read it from). `census_exit` (round-2 re-audit BLOCK 3(b))
# persists the ACTUAL exit code `kernel_census.py` (or `run_cmd`'s own
# DRY_RUN stand-in) returned -- `null` only when this section never even
# attempted the invocation (an earlier leg-level failure) -- so exit 9
# (empty differenced census) vs exit 4 (a per-key/fixed-cost violation) is
# machine-readable in the persisted record even on refusal, alongside
# `reason`'s own first `::error::` line from kernel_census.py's stderr.
_write_manifest() {
  MANIFEST_LEG_ID="$1" MANIFEST_GIT_SHA="$2" MANIFEST_BOX="$3" MANIFEST_NSYS_VERSION="$4" \
  MANIFEST_DTYPE="$5" MANIFEST_WIDTH="$6" MANIFEST_TARGET_MODULES="$7" \
  MANIFEST_LAYERS_TO_TRANSFORM="$8" MANIFEST_EVAL_CADENCE="$9" \
  MANIFEST_STEPS_DECLARED_N="${10}" MANIFEST_STEPS_DECLARED_M="${11}" \
  MANIFEST_STEPS_MEASURED_N="${12}" MANIFEST_STEPS_MEASURED_M="${13}" \
  MANIFEST_STATUS="${14}" MANIFEST_REASON="${15}" MANIFEST_CENSUS_OK="${16}" \
  MANIFEST_LORA_N="${17}" MANIFEST_LORA_M="${18}" MANIFEST_DRY_RUN="${19}" \
  MANIFEST_FIXED_COST_BUCKETS="${20}" MANIFEST_FIXED_COST_TIME_US="${21}" \
  MANIFEST_FIXED_COST_JITTER_MAX_REL="${22}" MANIFEST_CENSUS_EXIT="${23}" \
  MANIFEST_OUT="${24}" \
  python3 -c '
import json, os


def _int_or_none(v):
    return int(v) if v not in ("", None) else None


def _float_or_none(v):
    return float(v) if v not in ("", None) else None


manifest = {
    "leg_id": os.environ["MANIFEST_LEG_ID"],
    "git_sha": os.environ["MANIFEST_GIT_SHA"],
    "box": os.environ["MANIFEST_BOX"],
    "driver": "ci/scripts/perf/profile_356_legs.sh",
    "nsys_version": os.environ["MANIFEST_NSYS_VERSION"],
    "dtype": os.environ["MANIFEST_DTYPE"],
    "width": int(os.environ["MANIFEST_WIDTH"]),
    "target_modules": os.environ["MANIFEST_TARGET_MODULES"].split(","),
    "layers_to_transform": os.environ["MANIFEST_LAYERS_TO_TRANSFORM"] or None,
    "eval_cadence": int(os.environ["MANIFEST_EVAL_CADENCE"]),
    "steps_declared": {
        "n": int(os.environ["MANIFEST_STEPS_DECLARED_N"]),
        "m": int(os.environ["MANIFEST_STEPS_DECLARED_M"]),
    },
    "steps_measured": {
        "n": _int_or_none(os.environ["MANIFEST_STEPS_MEASURED_N"]),
        "m": _int_or_none(os.environ["MANIFEST_STEPS_MEASURED_M"]),
    },
    "status": os.environ["MANIFEST_STATUS"],
    "reason": os.environ["MANIFEST_REASON"],
    "census_ok": os.environ["MANIFEST_CENSUS_OK"] == "true",
    "lora_counters_n_run": json.loads(os.environ["MANIFEST_LORA_N"]),
    "lora_counters_m_run": json.loads(os.environ["MANIFEST_LORA_M"]),
    "dry_run": os.environ["MANIFEST_DRY_RUN"] == "true",
    "fixed_cost_buckets": _int_or_none(os.environ["MANIFEST_FIXED_COST_BUCKETS"]),
    "fixed_cost_time_us": _float_or_none(os.environ["MANIFEST_FIXED_COST_TIME_US"]),
    "fixed_cost_jitter_max_rel": _float_or_none(os.environ["MANIFEST_FIXED_COST_JITTER_MAX_REL"]),
    "census_exit": _int_or_none(os.environ["MANIFEST_CENSUS_EXIT"]),
}
json.dump(manifest, open(os.environ["MANIFEST_OUT"], "w"), indent=1)
'
}

# One leg, start to finish. Returns 0 for every OUTCOME this leg's own
# workload can produce -- a leg's own failure (corpus/run_traced/census/
# counter-read) lives in its `manifest.json` (`status`/`reason`), never in
# this function's own exit code, so one leg's OOM/refusal never discards
# any other leg. TWO EXCEPTIONS, both deliberate and both a genuine
# `exit 1` OUT OF THE WHOLE SCRIPT, never a silent `return`: a
# `manifest.json` write itself failing (the fallback-manifest branch's own
# `exit 1`, and the main end-of-leg one below) -- a manifest write failing
# means this leg's result is UNRECOVERABLE (there is nowhere else to
# record it), which this script treats as sweep-fatal by deliberate
# design, stated loudly, never silently swallowed the way every OTHER
# leg-level failure is.
#
# EVERY OTHER RISKY COMMAND IS EXPLICITLY GUARDED (phase-4 round-2 audit
# BLOCK 2): round-1's fix relied in part on `set -e`'s own "errexit is
# suspended for a command tested by if/||" behavior propagating through
# nested calls -- true, but ONLY for commands reachable that way; a
# handful of PLAIN, untested commands (the old per-leg `nsys --version`
# capture, the corpus `head` redirects) were still exposed, and a real pod
# run reproduced exactly that: a failing `nsys --version` aborted the
# WHOLE 14-leg sweep mid-leg with NO manifest for the failing leg. Every
# one of those is now an explicit `if ... ; then ... ; else
# leg_status=invalid; fi` -- this function's own "never aborts on a
# WORKLOAD failure" guarantee no longer depends on the calling convention
# at its OWN call site (the bottom loop calls it plainly: a workload bug
# inside `run_leg` cannot escape `run_leg` at all now, regardless of how
# it is invoked -- ONLY the two named manifest-write exceptions above
# still can, by design). A small number of near-infallible commands
# (`mktemp -d`/`rm -rf` cleanup, the `: > file` truncations) remain
# unguarded as an ACCEPTED residual -- these fail only under
# resource-exhaustion conditions (disk full, out of file descriptors) that
# would make continuing the sweep meaningless regardless, and guarding
# them would add verbosity without a plausible corresponding gain.
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

  local dry_run_flag="false"
  if [ "$PROFILE_356_LEGS_DRY_RUN" = "1" ]; then dry_run_flag="true"; fi

  local leg_dir="$OUT_DIR/$leg_id"
  if ! mkdir -p "$leg_dir"; then
    # Cannot write ANYTHING under leg_dir -- fall back to a FLAT manifest
    # path directly under OUT_DIR (phase-4 round-2 audit BLOCK 2: "write a
    # minimal manifest even there", never a silent return with nothing
    # recorded for this leg).
    echo "::error::$leg_id: could not create $leg_dir -- writing a fallback manifest instead." >&2
    if ! _write_manifest \
        "$leg_id" "$SHA" "$(hostname)" "$NSYS_VERSION" "$dtype" "$width" "$target_modules" \
        "$layers_to_transform" "$EVAL_CADENCE" "$n_steps" "$m_steps" \
        "" "" "invalid" "could not create leg directory $leg_dir" "false" \
        "null" "null" "$dry_run_flag" "" "" "" "" "$OUT_DIR/${leg_id}.manifest.json"; then
      echo "::error::$leg_id: could not write even the fallback manifest.json -- this IS sweep-fatal (results cannot be recorded); aborting." >&2
      exit 1
    fi
    return 0
  fi

  local leg_status="ok"
  local leg_reason=""

  # --- corpus provisioning ---
  local train_n="" train_m="" heldout_ids="$HELDOUT_IDS" heldout_jsonl="$HELDOUT_JSONL"
  local verify_tok_args=()
  if [ -f "$model_dir/tokenizer.json" ]; then
    verify_tok_args=(--verify-tokenizer "$model_dir/tokenizer.json")
  fi

  if [ "$corpus_mode" = "synthetic" ]; then
    local full_corpus="$leg_dir/corpus_m${m_steps}.jsonl"
    local rows_m=$(( batch * m_steps ))
    local rows_n=$(( batch * n_steps ))
    if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
      if python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" --min-wordpieces "$width" \
          --seed 42 --out "$full_corpus" "${verify_tok_args[@]}"; then
        if head -n "$rows_n" "$full_corpus" > "$leg_dir/corpus_n${n_steps}.jsonl"; then
          train_n="$leg_dir/corpus_n${n_steps}.jsonl"
          train_m="$full_corpus"
        else
          leg_status="invalid"
          leg_reason="head -n $rows_n $full_corpus failed while slicing the N-step corpus"
        fi
      else
        leg_status="invalid"
        leg_reason="gen_fixed_width_corpus.py failed (see leg dir for any partial output)"
      fi
    else
      run_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" --min-wordpieces "$width" \
        --seed 42 --out "$full_corpus" "${verify_tok_args[@]}"
      : > "$full_corpus"
      : > "$leg_dir/corpus_n${n_steps}.jsonl"
      train_n="$leg_dir/corpus_n${n_steps}.jsonl"
      train_m="$full_corpus"
    fi
  else
    # E1: the REAL, network-provisioned + byte-verified train_pairs.jsonl
    # (CONTRACT P2) -- pre-run provisioning is this script's caller's job
    # (mirrors finetune_run_ab.sh's own PRE-RUN PROVISIONING step); this
    # driver only refuses (this leg only, never the sweep) if the file is
    # absent/fails verification.
    if [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
      if python3 "$DIR/verify_train_pairs.py" --pairs "$TRAIN_JSONL_REAL"; then
        if head -n "$(( batch * n_steps ))" "$TRAIN_JSONL_REAL" > "$leg_dir/corpus_n${n_steps}.jsonl" \
            && head -n "$(( batch * m_steps ))" "$TRAIN_JSONL_REAL" > "$leg_dir/corpus_m${m_steps}.jsonl"; then
          train_n="$leg_dir/corpus_n${n_steps}.jsonl"
          train_m="$leg_dir/corpus_m${m_steps}.jsonl"
        else
          leg_status="invalid"
          leg_reason="head -n ... $TRAIN_JSONL_REAL failed while slicing the E1 real corpus"
        fi
      else
        leg_status="invalid"
        leg_reason="$TRAIN_JSONL_REAL failed byte-verification against the committed train_ids_sha256.json"
      fi
    else
      : > "$leg_dir/corpus_n${n_steps}.jsonl"
      : > "$leg_dir/corpus_m${m_steps}.jsonl"
      train_n="$leg_dir/corpus_n${n_steps}.jsonl"
      train_m="$leg_dir/corpus_m${m_steps}.jsonl"
    fi
  fi

  local out_n="$leg_dir/run_n.json" out_m="$leg_dir/run_m.json"
  local sqlite_n="$leg_dir/run_n.sqlite" sqlite_m="$leg_dir/run_m.sqlite"

  if [ "$leg_status" = "ok" ]; then
    if ! run_traced "$model_dir" "$train_n" "$heldout_ids" "$heldout_jsonl" 42 "$batch" "$dtype" \
        "$target_modules" "$layers_to_transform" "$width" "$n_steps" \
        "$leg_dir/work_n" "$leg_dir/run_n" "$out_n" "$sqlite_n"; then
      leg_status="invalid"
      # "nsys/bench execution or report-envelope validation" -- run_traced
      # returns nonzero for either cause; $run_n.stderr now ACTUALLY
      # contains whichever one it was (validate_report_envelope's own
      # error text is teed there too, round-3 audit item 5), so this file
      # reference is a genuine pointer at the cause, not a guess.
      leg_reason="run_traced (N-step run) failed (nsys/bench execution or report-envelope validation) -- see $leg_dir/run_n.stderr for the specific cause"
    fi
  fi
  if [ "$leg_status" = "ok" ]; then
    if ! run_traced "$model_dir" "$train_m" "$heldout_ids" "$heldout_jsonl" 42 "$batch" "$dtype" \
        "$target_modules" "$layers_to_transform" "$width" "$m_steps" \
        "$leg_dir/work_m" "$leg_dir/run_m" "$out_m" "$sqlite_m"; then
      leg_status="invalid"
      leg_reason="run_traced (M-step run) failed (nsys/bench execution or report-envelope validation) -- see $leg_dir/run_m.stderr for the specific cause"
    fi
  fi

  local wall_n="" wall_m=""
  if [ "$leg_status" = "ok" ]; then
    if ! wall_n="$(_wall_s "$out_n")"; then
      leg_status="invalid"
      leg_reason="could not read tiers.finetune_run.train_run_wall_s from the N-step run"
    fi
  fi
  if [ "$leg_status" = "ok" ]; then
    if ! wall_m="$(_wall_s "$out_m")"; then
      leg_status="invalid"
      leg_reason="could not read tiers.finetune_run.train_run_wall_s from the M-step run"
    fi
  fi

  # steps_measured: cross-checked "when available" -- absence alone is
  # never a leg failure (a build predating that field, or a fake that
  # chose not to populate it, is not an error), only a DISAGREEMENT with
  # the declared count is (kernel_census.py's own domain check, fed via
  # --steps-measured-a/-b below).
  local steps_measured_n="" steps_measured_m=""
  if [ "$leg_status" = "ok" ]; then
    steps_measured_n="$(_steps_measured "$out_n" 2>/dev/null)" || steps_measured_n=""
    steps_measured_m="$(_steps_measured "$out_m" 2>/dev/null)" || steps_measured_m=""
  fi

  local census_ok="false"
  local census_json="$leg_dir/census.json"
  local census_stderr="$leg_dir/census.stderr"
  # Persisted so exit 9 (empty differenced census) vs exit 4 (a
  # negative/fixed-cost-jitter violation) -- and WHICH rule fired -- is
  # machine-readable in the manifest even on refusal (phase-4 audit round-2
  # re-audit BLOCK 3(b)). "" (-> null in the manifest) unless this section
  # actually attempts the real invocation below.
  local census_exit=""
  if [ "$leg_status" = "ok" ]; then
    local -a census_cmd=(
      python3 "$DIR/kernel_census.py" "$sqlite_n" "$sqlite_m" "$n_steps" "$m_steps" "$census_json"
      --wall-a "$wall_n" --wall-b "$wall_m"
    )
    if [ -n "$steps_measured_n" ]; then census_cmd+=(--steps-measured-a "$steps_measured_n"); fi
    if [ -n "$steps_measured_m" ]; then census_cmd+=(--steps-measured-b "$steps_measured_m"); fi
    case "$leg_id" in
      *-E1) census_cmd+=(--excluded-from-chain-attribution) ;;
    esac
    # `run_cmd`'s own echoed "+ ..." trace line and (on a real, non-DRY_RUN
    # invocation) kernel_census.py's own stderr are captured into a shell
    # variable first (`2>&1 1>/dev/null` -- the same "capture only stderr"
    # idiom, never a live pipe/process-substitution race) so BOTH the
    # exit code (`$?`, immediately after the command substitution) and the
    # text are available -- then re-emitted live (`>&2`, so this leg's
    # trace/error visibility on the real console/CI log is unchanged) AND
    # persisted to `$census_stderr` (so `leg_reason`'s own pointer below
    # resolves to a real file, the same convention `run_traced`'s
    # `run_n.stderr`/`run_m.stderr` already use).
    # `&&`/`||` here, never a bare assignment (phase-4 round-2 audit
    # BLOCK 2's own "every risky command is explicitly guarded" doctrine)
    # -- a bare `x="$(cmd)"` assignment is a PLAIN command under
    # `set -euo pipefail`; if `cmd` (kernel_census.py refusing) exits
    # nonzero, that would abort the WHOLE SCRIPT immediately, before
    # `census_exit=$?` ever ran, exactly the class of live regression
    # this doctrine already exists to catch.
    local census_err
    census_err="$(run_cmd "${census_cmd[@]}" 2>&1 1>/dev/null)" && census_exit=0 || census_exit=$?
    printf '%s\n' "$census_err" >&2
    printf '%s\n' "$census_err" > "$census_stderr"
    if [ "$census_exit" -eq 0 ]; then
      census_ok="true"
    else
      leg_status="invalid"
      local census_first_violation
      census_first_violation="$(printf '%s\n' "$census_err" | grep -m1 '^::error::' || true)"
      if [ -n "$census_first_violation" ]; then
        leg_reason="kernel_census.py refused (exit $census_exit): $census_first_violation -- see $census_stderr"
      else
        leg_reason="kernel_census.py refused (exit $census_exit, leg INVALID -- see $census_stderr; no census.json written)"
      fi
    fi
  fi

  # Surfaces kernel_census.py's own informational fixed-cost tally into
  # this leg's manifest (phase-4 audit round-2 BLOCK 2, widened round-2
  # re-audit BLOCK 3(a) to include `fixed_cost_jitter_max_rel`) so the
  # emitted-for-visibility claim in that module's doc has an actual
  # consumer. All three stay "" (-> null in the manifest) unless census_ok
  # AND NOT DRY_RUN -- census_cmd itself goes through `run_cmd`, which
  # under DRY_RUN only ECHOES the command and never executes it (see
  # `run_cmd`'s own doc), so `census_json` is never actually written in
  # that mode; `census_ok=true` there is the FAKE pipeline-completed
  # stand-in this producer's own doc already names, and reading a
  # fixed-cost tally out of a file that was never written would be a read
  # against a nonexistent report, not a genuinely-missing field.
  local fixed_cost_buckets="" fixed_cost_time_us="" fixed_cost_jitter_max_rel=""
  if [ "$census_ok" = "true" ] && [ "$PROFILE_356_LEGS_DRY_RUN" != "1" ]; then
    local census_fixed_cost_json
    if census_fixed_cost_json="$(_census_fixed_cost "$census_json")"; then
      fixed_cost_buckets="$(printf '%s' "$census_fixed_cost_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["fixed_cost_buckets"])')"
      fixed_cost_time_us="$(printf '%s' "$census_fixed_cost_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["fixed_cost_time_us"])')"
      fixed_cost_jitter_max_rel="$(printf '%s' "$census_fixed_cost_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["fixed_cost_jitter_max_rel"])')"
    else
      leg_status="invalid"
      leg_reason="_census_fixed_cost refused on $census_json (missing fixed_cost_buckets/fixed_cost_time_us/fixed_cost_jitter_max_rel field -- an unrecognized report shape)"
    fi
  fi

  local lora_n="null" lora_m="null"
  if [ "$leg_status" = "ok" ]; then
    if ! lora_n="$(_lora_counters "$out_n")"; then
      leg_status="invalid"
      leg_reason="_lora_counters refused on the N-step run (missing LoRA counter field -- the contract's only positive-proof channel)"
      lora_n="null"
    fi
  fi
  if [ "$leg_status" = "ok" ]; then
    if ! lora_m="$(_lora_counters "$out_m")"; then
      leg_status="invalid"
      leg_reason="_lora_counters refused on the M-step run (missing LoRA counter field -- the contract's only positive-proof channel)"
      lora_m="null"
    fi
  fi

  # A manifest write failure IS sweep-fatal (this is the ONE thing this
  # script cannot degrade gracefully from -- without it, this leg's
  # result is unrecorded and unrecoverable) -- but it fails LOUDLY and
  # deliberately (an explicit `exit 1` with a clear message), never a raw
  # `set -e` trace (phase-4 round-2 audit BLOCK 2).
  if ! _write_manifest \
      "$leg_id" "$SHA" "$(hostname)" "$NSYS_VERSION" "$dtype" "$width" "$target_modules" \
      "$layers_to_transform" "$EVAL_CADENCE" "$n_steps" "$m_steps" \
      "$steps_measured_n" "$steps_measured_m" "$leg_status" "$leg_reason" "$census_ok" \
      "$lora_n" "$lora_m" "$dry_run_flag" \
      "$fixed_cost_buckets" "$fixed_cost_time_us" "$fixed_cost_jitter_max_rel" "$census_exit" \
      "$leg_dir/manifest.json"; then
    echo "::error::$leg_id: could not write $leg_dir/manifest.json -- this IS sweep-fatal (this leg's result cannot be recorded); aborting the sweep now rather than continuing silently unrecorded." >&2
    exit 1
  fi

  if [ "$leg_status" != "ok" ]; then
    echo "::warning::$leg_id: INVALID -- $leg_reason (recorded in $leg_dir/manifest.json; sweep continues)." >&2
  fi

  # E1's own width evidence (Artifacts: "the E1 width histogram produced
  # OFFLINE by the tracked producer re-tokenizing the fixture in the same
  # anchors+positives join and batch order"). Runs against the COMMITTED
  # heldout_pairs.jsonl (never the synthetic corpus) regardless of model,
  # since the fixture text itself is model-independent -- only the
  # tokenizer differs, and the caller supplies model_dir's own
  # tokenizer.json. Best-effort: never affects this leg's own status
  # (E1's width evidence is a separate artifact, not a census input).
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
  return 0
}

for spec in "${LEGS[@]}"; do
  run_leg "$spec"
done

echo "profile_356_legs: done -- artifacts under $OUT_DIR"
