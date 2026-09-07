#!/usr/bin/env bash
# frontend_ab.sh -- issue #421 follow-on ("media front-end parallelization")
# contract's pre-registered A/B: two PREBUILT binaries (`$BASE_BIN`,
# `$TIP_BIN` -- never built by this script, and never compared against ONE
# repo's own HEAD; each is provenance-checked against the sha the OPERATOR
# declares for it, `$BASE_SHA`/`$TIP_SHA`, because base and tip are two
# DIFFERENT checkouts on the pod, so "this repo's HEAD" names neither of
# them correctly), driving untraced `jammi-bench finetune-run` over the
# HTSAT (`--task audio_embedding`) and CLIP-vision (`--task image_embedding`)
# towers at the profile's pinned leg parameters (`--objective triplet`,
# `--lora-init zeros_b`, the full per-tower LoRA site set, `--eval-cadence
# 1`, N = 100 steps, batch 8 -- 24 items/step under triplet), interleaved
# base, tip, base, tip per tower.
#
# DECISION QUANTITY (per report): `media_front_end_wall_s / steps_measured`.
# COMPANION (report-only): `train_run_wall_s / steps_measured`.
#
# THE BAR (HTSAT only; CLIP-vision is report-only -- contract "Measurement"):
# with `P` read from the TIP binary's own `rayon_pool_threads` (the rayon
# GLOBAL pool size that build's process resolved to), `n = 24`, `ideal =
# n / ceil(n / P)`, and `r` (the CPU-local serial-tail ratio) taken from the
# REQUIRED `--serial-tail-ratio` argument (the operator's own
# `cargo run -p jammi-bench --example frontend_serial_tail` measurement,
# divided by the corresponding tier's own per-step wall):
#
#   ratio = mean(front_tip) / mean(front_base)
#   upper_bound = r + (1 - r) / (0.5 * ideal)   -- at least a real win
#   lower_bound = r + (1 - r) / ideal           -- never beats the ideal
#
# `ratio` inside `[lower_bound, upper_bound]` is PASS (the bar holds);
# above `upper_bound` is FAIL (not enough speedup); below `lower_bound` is
# INVALID_BEATS_IDEAL (the instrument, not the code, is broken -- nothing
# can go faster than the machine model's own ideal). The base-to-base
# spread of the interleaved runs (`|front_base_r1 - front_base_r2|`) is the
# error bar: when that spread, added on either side of `ratio`, would cross
# EITHER bound, the decision is UNRESOLVED rather than a confident
# PASS/FAIL/INVALID_BEATS_IDEAL call.
#
# VERDICT (contract's own words): ACTIVATE (keep the change) iff the HTSAT
# bar holds (PASS); this script only RECORDS the bar's outcome -- it never
# reverts or gates a build on it. A human reads `htsat_bar.verdict` off the
# written artifact.
#
# ARTIFACT: writes the merged report to `$OUT_DIR/report.json`, and (real
# runs only -- see `FRONTEND_AB_DRY_RUN` below) the NEW, schema-conformant
# (`ci/scripts/check_cuda_run_artifacts.py`) cuda-run artifact to
# `crates/jammi-kernels/artifacts/cuda-runs/<date>-frontend-<tip
# short-sha>-<box slug>.json`, `producer.path` naming THIS script
# (`kind: "script"`) -- never touching the pre-existing #421 profile
# artifact under that same directory. Refuses (rather than overwrites) if
# that exact path is already present.
#
# Env vars:
#   FRONTEND_AB_DRY_RUN     "1" swaps `$BASE_BIN`/`$TIP_BIN` for hermetic
#                            fake stand-ins that VALIDATE their own argv
#                            (mirroring `profile_421_legs.sh`'s own
#                            `fake_bench.sh`) and never touches the network
#                            or the real committed artifacts directory
#                            (the merged report still lands under
#                            `$OUT_DIR`, tagged with a `-dry-run` filename).
#                            The two media corpus producers ALWAYS run for
#                            real, even under DRY_RUN (esc-088,
#                            `.jammi/escapes.jsonl`: a stubbed-out corpus
#                            step would make this driver's own corpus-
#                            provisioning code path invisible to every
#                            hermetic test, exactly the gap that escape
#                            documents) -- their output paths are always
#                            validated non-empty before any leg reads them.
#   BASE_BIN / TIP_BIN       paths to the two prebuilt `jammi-bench`
#                            binaries. Required unless DRY_RUN.
#   BASE_SHA / TIP_SHA       40-hex shas the operator DECLARES each binary
#                            was built from -- checked against that binary's
#                            OWN `provenance` output, never against this
#                            checkout's `git rev-parse HEAD` (base and tip
#                            are two different checkouts; neither need equal
#                            this repo's current HEAD). Required unless
#                            DRY_RUN.
#   BOX                      the JSON artifact's own `box` field (free-form,
#                            e.g. a pod hostname/container id). Required
#                            unless DRY_RUN.
#   BOX_SLUG                 filename-safe token for the artifact's own
#                            path (default: `$BOX` lowercased with every
#                            run of non-`[a-z0-9]` collapsed to a single
#                            `-`).
#   MODEL_DIR_CLIP           OpenCLIP ViT-B-32 checkpoint dir (CLIP-vision).
#   MODEL_DIR_CLAP           HF `laion/clap-htsat-fused` checkpoint dir
#                            (HTSAT). Both required unless DRY_RUN.
#   FRONTEND_AB_SERIAL_TAIL_RATIO
#                            `r` -- REQUIRED (dry-run included, so the
#                            parsing/validation path is always exercised):
#                            a float in `[0, 1]`.
#   FRONTEND_AB_STEPS_N      override the pinned N = 100 (default 100).
#   FRONTEND_AB_BACKBONE_DTYPE
#                            `--backbone-dtype` passthrough, every leg
#                            (default "bf16" -- irrelevant to the CPU-side
#                            front end this A/B measures, but pinned
#                            identically across every leg so the GPU-side
#                            forward this front end feeds is never itself a
#                            confound).
#   FRONTEND_AB_CUDA         CUDA ordinal (default 0). Set
#                            FRONTEND_AB_CPU=1 to omit --cuda entirely
#                            (CPU-hermetic smoke path) -- never both.
#   OUT_DIR                  where the raw legs + merged report land
#                            (default "<repo>/.frontend-ab-report/<UTC ts>").
#
# Hermetic self-test: `python3 ci/scripts/perf/test_frontend_ab_dry_run.py`.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

FRONTEND_AB_DRY_RUN="${FRONTEND_AB_DRY_RUN:-0}"

# --- ambient JAMMI_KERNELS_DISABLE guard (mirrors profile_421_legs.sh's own
# unit-467 finding F1 guard): every leg this driver runs is an ordinary
# `fused`-arm leg, and none claims `--expect-kernels-disabled` -- an ambient
# value here would silently contaminate every leg's own dispatch counters
# and, worse, could make a leg refuse outright mid-sweep for a reason that
# has nothing to do with the front end this A/B exists to measure. Refusing
# loudly, once, before any leg runs, is cheaper than debugging a
# contaminated sweep after the fact.
if [ -n "${JAMMI_KERNELS_DISABLE:-}" ]; then
  echo "::error::JAMMI_KERNELS_DISABLE is set in this driver's own environment ('$JAMMI_KERNELS_DISABLE') -- refusing before any leg runs. Every leg this driver runs is an ordinary fused-arm leg and claims no disable list; an ambient value would silently contaminate the front-end measurement. Unset it before running this driver." >&2
  exit 2
fi

FRONTEND_AB_SERIAL_TAIL_RATIO="${FRONTEND_AB_SERIAL_TAIL_RATIO:-}"
if [ -z "$FRONTEND_AB_SERIAL_TAIL_RATIO" ]; then
  echo "::error::FRONTEND_AB_SERIAL_TAIL_RATIO is required (the operator's own 'cargo run -p jammi-bench --example frontend_serial_tail' measurement, divided by the relevant tier's per-step wall) -- refusing before any leg runs." >&2
  exit 2
fi
if ! python3 -c "
import sys
r = float('$FRONTEND_AB_SERIAL_TAIL_RATIO')
sys.exit(0 if 0.0 <= r <= 1.0 else 1)
" 2>/dev/null; then
  echo "::error::FRONTEND_AB_SERIAL_TAIL_RATIO ('$FRONTEND_AB_SERIAL_TAIL_RATIO') must be a float in [0, 1] -- refusing before any leg runs." >&2
  exit 2
fi

BASE_BIN="${BASE_BIN:-}"
TIP_BIN="${TIP_BIN:-}"
BASE_SHA="${BASE_SHA:-}"
TIP_SHA="${TIP_SHA:-}"
BOX="${BOX:-}"
MODEL_DIR_CLIP="${MODEL_DIR_CLIP:-}"
MODEL_DIR_CLAP="${MODEL_DIR_CLAP:-}"
FRONTEND_AB_STEPS_N="${FRONTEND_AB_STEPS_N:-100}"
FRONTEND_AB_BACKBONE_DTYPE="${FRONTEND_AB_BACKBONE_DTYPE:-bf16}"
FRONTEND_AB_CUDA="${FRONTEND_AB_CUDA:-0}"
FRONTEND_AB_CPU="${FRONTEND_AB_CPU:-0}"

SHA_RE='^[0-9a-f]{40}$'

if [ "$FRONTEND_AB_DRY_RUN" != "1" ]; then
  missing=()
  [ -z "$BASE_BIN" ] && missing+=("BASE_BIN")
  [ -z "$TIP_BIN" ] && missing+=("TIP_BIN")
  [ -z "$BASE_SHA" ] && missing+=("BASE_SHA")
  [ -z "$TIP_SHA" ] && missing+=("TIP_SHA")
  [ -z "$BOX" ] && missing+=("BOX")
  [ -z "$MODEL_DIR_CLIP" ] && missing+=("MODEL_DIR_CLIP")
  [ -z "$MODEL_DIR_CLAP" ] && missing+=("MODEL_DIR_CLAP")
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::missing required env var(s) for a real run: ${missing[*]}" >&2
    exit 2
  fi
  for f in "$BASE_BIN" "$TIP_BIN"; do
    if [ ! -x "$f" ]; then
      echo "::error::$f is not an executable file -- refusing before any leg runs." >&2
      exit 2
    fi
  done
  for sha in "$BASE_SHA" "$TIP_SHA"; do
    if ! [[ "$sha" =~ $SHA_RE ]]; then
      echo "::error::declared sha '$sha' is not 40 lowercase hex chars -- refusing before any leg runs." >&2
      exit 2
    fi
  done
else
  BASE_SHA="${BASE_SHA:-0000000000000000000000000000000000000base}"
  TIP_SHA="${TIP_SHA:-0000000000000000000000000000000000000tip0}"
  # The two placeholder shas above are intentionally NOT valid hex (they
  # spell "base"/"tip0" in their last four chars) -- a dry run never reaches
  # `check_cuda_run_artifacts.py`'s schema gate (the artifact write is
  # skipped entirely under DRY_RUN, see below), so nothing ever validates
  # these against `$SHA_RE`; spelling them memorably is strictly for a
  # human reading dry-run trace output, not a schema requirement.
  BOX="${BOX:-dry-run-box}"
  MODEL_DIR_CLIP="${MODEL_DIR_CLIP:-/root/checkpoints/open-clip-vit-b-32-DRY-RUN-PLACEHOLDER}"
  MODEL_DIR_CLAP="${MODEL_DIR_CLAP:-/root/checkpoints/clap-htsat-fused-DRY-RUN-PLACEHOLDER}"
  # Trace-only placeholders (never executed -- see `run_leg`'s own DRY_RUN
  # branch, which always execs the hermetic stub instead): keeps
  # `_print_cmd`'s echoed command line naming a real-looking path rather
  # than an empty string.
  BASE_BIN="${BASE_BIN:-/root/base-checkout/target/release/jammi-bench-DRY-RUN-PLACEHOLDER}"
  TIP_BIN="${TIP_BIN:-/root/tip-checkout/target/release/jammi-bench-DRY-RUN-PLACEHOLDER}"
  echo "::warning::FRONTEND_AB_DRY_RUN=1 -- nothing is read from BASE_BIN/TIP_BIN/MODEL_DIR_*; the real committed artifacts directory is never written."
fi

BOX_SLUG_DEFAULT="$(printf '%s' "$BOX" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
BOX_SLUG="${BOX_SLUG:-$BOX_SLUG_DEFAULT}"
if [ -z "$BOX_SLUG" ]; then
  echo "::error::BOX_SLUG (or a sanitizable BOX) is required -- got BOX='$BOX', sanitized to an empty string." >&2
  exit 2
fi

# ── Contract-pinned workload constants ("Measurement" / "Workload shape") ──
BATCH=8
SEED=42
EVAL_CADENCE=1
OBJECTIVE=triplet
LORA_INIT=zeros_b
MEDIA_SEQ=77            # a no-op cap on a media task; pinned for a visible,
                         # constant command line (profile_421_legs.sh's own
                         # convention for every non-text tower).
IMAGE_SIZE=224
AUDIO_SECONDS=9.5
AUDIO_SAMPLE_RATE=48000
HELDOUT_ROWS=8
MEDIA_FAMILIES=6
MEDIA_HELDOUT_FAMILIES=2
N_ITEMS_PER_STEP=24      # BATCH * 3 (triplet) -- the machine model's `n`.

CLIP_FULL="in_proj,out_proj,c_fc,c_proj"
CLAP_FULL="query,key,value,attention_output,intermediate_dense,output_dense,reduction,linear1,linear2"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
DATE_ONLY="$(date -u +%Y-%m-%d)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.frontend-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

# --- trace echo: always stderr, never stdout -- a trace line sharing stdout
# with a captured child's own JSON report is how a report envelope gets
# polluted (profile_421_legs.sh's own lesson).
_print_cmd() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

# --- corpus-producer wrapper: ALWAYS executes, even under DRY_RUN (esc-088
# -- see module doc). The child's own stdout is forwarded to THIS script's
# stderr, never left on a channel a later capture point could pick up.
run_corpus_cmd() {
  _print_cmd "$@"
  "$@" >&2
}

# ── Hermetic DRY_RUN stand-ins ──────────────────────────────────────────
# Generated once, reused for every leg. Unlike profile_421_legs.sh's
# fake_bench.sh (which threads its overrides through fixed POSITIONAL
# arguments because it sits behind an nsys wrapper this driver has no
# analogue of), this stub reads its OWN real argv directly -- the exact
# argv `finetune-run` itself would see -- and VALIDATES it, so a regression
# that drops/garbles a required flag fails HERE, hermetically, rather than
# only on a real pod run.
DRY_RUN_STUB_DIR=""
if [ "$FRONTEND_AB_DRY_RUN" = "1" ]; then
  DRY_RUN_STUB_DIR="$(mktemp -d)"
  trap 'rm -rf "$DRY_RUN_STUB_DIR"' EXIT

  # $1 = role ("base" or "tip") -- controls the fabricated
  # rayon_pool_threads/media_front_end_wall_s pair so a dry run exercises a
  # REAL (if fabricated) speedup the bar-decision arithmetic can chew on.
  for role in base tip; do
    stub="$DRY_RUN_STUB_DIR/fake_bench_${role}.sh"
    # `FRONTEND_AB_DRY_RUN_${ROLE}_PROVENANCE_SHA_OVERRIDE` is a TEST-ONLY
    # lever (never set by a real run, and inert on one -- these stubs never
    # exist outside DRY_RUN): lets `test_frontend_ab_dry_run.py` simulate a
    # binary that reports a DIFFERENT sha than the operator declared for it,
    # exercising `_check_provenance`'s own mismatch refusal hermetically.
    role_upper="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
    override_var="FRONTEND_AB_DRY_RUN_${role_upper}_PROVENANCE_SHA_OVERRIDE"
    reported_sha_expr="\${FRONTEND_AB_STUB_SHA}"
    if [ -n "${!override_var:-}" ]; then
      reported_sha_expr="${!override_var}"
    fi
    cat > "$stub" <<STUBEOF
#!/usr/bin/env bash
set -euo pipefail
if [ "\${1:-}" = "provenance" ]; then
  printf '{"build_sha":"%s","target":"dry-run","profile":"dry-run"}\\n' "$reported_sha_expr"
  exit 0
fi
if [ "\${1:-}" != "finetune-run" ]; then
  echo "fake_bench_${role}: unknown subcommand '\${1:-}'" >&2
  exit 2
fi
shift
task="" train_jsonl="" heldout_ids="" heldout_jsonl="" objective="" lora_init="" \\
  target_modules="" eval_cadence="" arm="" seed="" batch=""
while [ "\$#" -gt 0 ]; do
  case "\$1" in
    --task) task="\$2"; shift 2 ;;
    --train-jsonl) train_jsonl="\$2"; shift 2 ;;
    --heldout-ids) heldout_ids="\$2"; shift 2 ;;
    --heldout-jsonl) heldout_jsonl="\$2"; shift 2 ;;
    --objective) objective="\$2"; shift 2 ;;
    --lora-init) lora_init="\$2"; shift 2 ;;
    --target-modules) target_modules="\$2"; shift 2 ;;
    --eval-cadence) eval_cadence="\$2"; shift 2 ;;
    --arm) arm="\$2"; shift 2 ;;
    --seed) seed="\$2"; shift 2 ;;
    --batch) batch="\$2"; shift 2 ;;
    *) shift ;;
  esac
done
_require_nonempty_file() {
  if [ -z "\$1" ] || [ ! -s "\$1" ]; then
    echo "error: a value is required for '\$2' but none was supplied (got '\$1')" >&2
    exit 2
  fi
}
_require_nonempty_file "\$train_jsonl" --train-jsonl
_require_nonempty_file "\$heldout_ids" --heldout-ids
_require_nonempty_file "\$heldout_jsonl" --heldout-jsonl
for required in "\$task:--task" "\$objective:--objective" "\$lora_init:--lora-init" \\
  "\$target_modules:--target-modules" "\$eval_cadence:--eval-cadence" "\$arm:--arm" \\
  "\$seed:--seed" "\$batch:--batch"; do
  val="\${required%%:*}" flag="\${required##*:}"
  if [ -z "\$val" ]; then
    echo "error: a value is required for '\$flag' but none was supplied" >&2
    exit 2
  fi
done
if [ "\$objective" != "triplet" ]; then
  echo "error: fake_bench_${role} expected --objective triplet, got '\$objective'" >&2
  exit 2
fi
if [ "\$lora_init" != "zeros_b" ]; then
  echo "error: fake_bench_${role} expected --lora-init zeros_b, got '\$lora_init'" >&2
  exit 2
fi
# TEST-ONLY (never set by a real run): forces THIS leg's stub to fail
# outright, proving the merge step's own INVALID handling fires on a real
# leg failure rather than only being exercised by a synthetic report.
# Format: "<task>__<role>__<repeat>", e.g. "audio_embedding__base__r1".
_force_fail_leg="\${FRONTEND_AB_DRY_RUN_FAIL_LEG:-}"
if [ -n "\$_force_fail_leg" ] && [ "\$_force_fail_leg" = "\${task}__${role}__\${FRONTEND_AB_STUB_REPEAT:-}" ]; then
  echo "error: fake_bench_${role}: forced failure for '\$_force_fail_leg' (FRONTEND_AB_DRY_RUN_FAIL_LEG)" >&2
  exit 3
fi
STUBEOF
    # role-specific fabricated numbers, appended after the shared argv
    # validation above -- base is deliberately slower and single-threaded;
    # tip is faster and multi-threaded, so the dry run exercises a real
    # (if fabricated) PASS-shaped bar at the DEFAULT knob values below (P =
    # 13 -> ideal = 12; ratio = 0.10 sits inside [lower_bound, upper_bound]
    # for every r in [0, 1] -- see this script's own module doc). Every
    # number here is a TEST-ONLY knob (env-var overridable so
    # `test_frontend_ab_dry_run.py` can drive the bar's FAIL/
    # INVALID_BEATS_IDEAL/UNRESOLVED branches deterministically): a real run
    # never sets these, and never reaches this stub at all.
    if [ "$role" = "base" ]; then
      cat >> "$stub" <<STUBEOF
rayon_threads=1
front_per_step=${FRONTEND_AB_DRY_RUN_BASE_FRONT_PER_STEP:-0.020}
# TEST-ONLY, read at STUB RUNTIME (not baked in like the default above):
# lets a test give the base tower's r2 repeat a DIFFERENT front-per-step
# than r1, so the base-to-base spread ("error bar") is genuinely nonzero --
# otherwise two hardcoded-identical base legs always spread exactly 0 and
# the UNRESOLVED branch (spread straddling a bound) could never be
# exercised. \${FRONTEND_AB_STUB_REPEAT} is exported by \`run_leg\` per call.
if [ "\${FRONTEND_AB_STUB_REPEAT:-}" = "r2" ] && [ -n "\${FRONTEND_AB_DRY_RUN_BASE_FRONT_PER_STEP_R2:-}" ]; then
  front_per_step="\$FRONTEND_AB_DRY_RUN_BASE_FRONT_PER_STEP_R2"
fi
STUBEOF
    else
      cat >> "$stub" <<STUBEOF
rayon_threads=${FRONTEND_AB_DRY_RUN_TIP_RAYON_THREADS:-13}
front_per_step=${FRONTEND_AB_DRY_RUN_TIP_FRONT_PER_STEP:-0.0020}
STUBEOF
    fi
    cat >> "$stub" <<STUBEOF
steps_measured=$FRONTEND_AB_STEPS_N
train_per_step=0.150
python3 -c "
import json
steps = int('\$steps_measured')
# Mirrors the REAL 'jammi-bench finetune-run' envelope (report.rs's
# FinetuneRunTier, serialized under tiers.finetune_run -- see
# frontend_ab_merge.py's own doc for why a flat shape here is exactly the
# bug this stub used to hide): steps_measured / media_front_end_wall_s /
# train_run_wall_s / rayon_pool_threads all live UNDER tiers.finetune_run,
# never at the report's top level. A real BASE-role binary (pre-#421
# follow-on) never emits rayon_pool_threads at all -- this stub's '${role}'
# arm mirrors that exactly, never fabricating a key the real binary that
# role stands in for would not have written.
tier = {
    'task': '\$task',
    'steps_measured': steps,
    'media_front_end_wall_s': float('\$front_per_step') * steps,
    'train_run_wall_s': float('\$train_per_step') * steps,
}
if '${role}' == 'tip':
    tier['rayon_pool_threads'] = int('\$rayon_threads')
print(json.dumps({
    'host': {'logical_cpus': int('\$rayon_threads')},
    'tiers': {'finetune_run': tier},
}))
"
STUBEOF
    chmod +x "$stub"
  done
fi

# --- provenance cross-check: each binary's OWN `provenance` output against
# the sha the OPERATOR declares for it -- NEVER against `git rev-parse
# HEAD` (base and tip are two DIFFERENT checkouts; this repo's own HEAD
# names neither).
_check_provenance() {
  local role="$1" bin_path="$2" declared_sha="$3"
  local prov_json rc=0
  if [ "$FRONTEND_AB_DRY_RUN" = "1" ]; then
    prov_json="$(FRONTEND_AB_STUB_SHA="$declared_sha" "$DRY_RUN_STUB_DIR/fake_bench_${role}.sh" provenance)" || rc=$?
  else
    prov_json="$("$bin_path" provenance 2>&1)" || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    echo "::error::'$bin_path provenance' failed: $prov_json" >&2
    exit 1
  fi
  local prov_sha
  prov_sha="$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$bin_path provenance' output: $prov_json" >&2; exit 1; }
  if [ "$prov_sha" != "$declared_sha" ]; then
    local declared_var_name
    declared_var_name="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')_SHA"
    echo "::error::$role binary '$bin_path' reports build_sha=$prov_sha, but the operator declared $declared_var_name=$declared_sha -- refusing before any leg runs." >&2
    exit 1
  fi
}
_check_provenance base "$BASE_BIN" "$BASE_SHA"
_check_provenance tip "$TIP_BIN" "$TIP_SHA"

# --- corpus provisioning: ONE corpus per tower, rows = BATCH * STEPS_N,
# shared by every leg (base and tip read the identical bytes -- the whole
# point of an A/B). Reuses profile_421_legs.sh's own producer invocations
# verbatim (same flags, same producers) so this driver's corpora are
# byte-shaped exactly like that campaign's.
ROWS=$(( BATCH * FRONTEND_AB_STEPS_N ))
CORPUS_DIR="$OUT_DIR/corpus"

provision_tower_corpus() {
  local tower="$1" dir="$2"
  case "$tower" in
    clip-vision)
      run_corpus_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$ROWS" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$dir" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      ;;
    htsat)
      run_corpus_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$ROWS" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$dir" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      ;;
    *)
      echo "::error::provision_tower_corpus: unknown tower '$tower'" >&2
      return 1
      ;;
  esac
}

# esc-088: every corpus path this driver is about to hand a leg is
# validated non-empty HERE, before any leg reads it -- never merely assumed
# because the producer reported exit 0 (a producer that exits 0 having
# written nothing is exactly the failure mode esc-088 documents).
_require_nonempty() {
  local path="$1" label="$2"
  if [ ! -s "$path" ]; then
    echo "::error::$label ('$path') is missing or empty after corpus provisioning -- refusing before any leg runs." >&2
    exit 1
  fi
}

declare -A TRAIN_JSONL HELDOUT_IDS HELDOUT_JSONL TASK TARGET_MODULES MODEL_DIR
for tower in clip-vision htsat; do
  dir="$CORPUS_DIR/$tower"
  mkdir -p "$dir"
  provision_tower_corpus "$tower" "$dir" \
    || { echo "::error::corpus provisioning failed for tower '$tower'" >&2; exit 1; }
  TRAIN_JSONL[$tower]="$dir/triplets.jsonl"
  HELDOUT_IDS[$tower]="$dir/heldout_ids.txt"
  HELDOUT_JSONL[$tower]="$dir/heldout_triplets.jsonl"
  _require_nonempty "${TRAIN_JSONL[$tower]}" "$tower train_jsonl"
  _require_nonempty "${HELDOUT_IDS[$tower]}" "$tower heldout_ids"
  _require_nonempty "${HELDOUT_JSONL[$tower]}" "$tower heldout_jsonl"
  case "$tower" in
    clip-vision) TASK[$tower]=image_embedding; TARGET_MODULES[$tower]="$CLIP_FULL"; MODEL_DIR[$tower]="$MODEL_DIR_CLIP" ;;
    htsat) TASK[$tower]=audio_embedding; TARGET_MODULES[$tower]="$CLAP_FULL"; MODEL_DIR[$tower]="$MODEL_DIR_CLAP" ;;
  esac
done

# --- one leg: an untraced `finetune-run`, N steps, over one tower's shared
# corpus, on one of the two binaries. Never aborts the sweep on its own
# failure (encode_ab.sh/finetune_run_ab.sh's own convention) -- recorded as
# this leg's own outcome; the merge step decides what a missing/failed leg
# means for the bar.
run_leg() {
  local tower="$1" role="$2" bin_path="$3" declared_sha="$4" repeat="$5"
  local work_dir="$OUT_DIR/work/${tower}__${role}__${repeat}"
  local out_file="$RAW_DIR/${tower}__${role}__${repeat}.json"
  local err_file="$RAW_DIR/${tower}__${role}__${repeat}.stderr"
  local exit_file="$RAW_DIR/${tower}__${role}__${repeat}.exit"
  mkdir -p "$work_dir"

  local -a cmd=(
    "$bin_path" finetune-run
    --model-dir "${MODEL_DIR[$tower]}" --arm fused --task "${TASK[$tower]}"
    --train-jsonl "${TRAIN_JSONL[$tower]}" --heldout-ids "${HELDOUT_IDS[$tower]}" \
    --heldout-jsonl "${HELDOUT_JSONL[$tower]}"
    --seed "$SEED" --epochs 1 --batch "$BATCH" --objective "$OBJECTIVE"
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$FRONTEND_AB_BACKBONE_DTYPE"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --lora-init "$LORA_INIT"
    --target-modules "${TARGET_MODULES[$tower]}"
    --max-seq-length "$MEDIA_SEQ"
    --eval-cadence "$EVAL_CADENCE"
    --work-dir "$work_dir"
  )
  if [ "$FRONTEND_AB_CPU" != "1" ]; then
    cmd+=(--cuda "$FRONTEND_AB_CUDA")
  fi

  _print_cmd "${cmd[@]}"
  local rc=0
  if [ "$FRONTEND_AB_DRY_RUN" = "1" ]; then
    local stub="$DRY_RUN_STUB_DIR/fake_bench_${role}.sh"
    local -a stub_cmd=("$stub" finetune-run)
    stub_cmd+=("${cmd[@]:2}")
    FRONTEND_AB_STUB_SHA="$declared_sha" FRONTEND_AB_STUB_REPEAT="$repeat" \
      "${stub_cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  else
    "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  fi
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${tower}/${role}/${repeat} FAILED (exit ${rc}) -- recorded as this leg's own outcome; sweep continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

for tower in htsat clip-vision; do
  run_leg "$tower" base "$BASE_BIN" "$BASE_SHA" r1
  run_leg "$tower" tip "$TIP_BIN" "$TIP_SHA" r1
  run_leg "$tower" base "$BASE_BIN" "$BASE_SHA" r2
  run_leg "$tower" tip "$TIP_BIN" "$TIP_SHA" r2
done

# ── merge + bar decision (`frontend_ab_merge.py`, this directory) ────────
# Extracted into an importable module (mirrors `finetune_run_ab.sh`'s own
# `python3 "$DIR/ab_merge.py" ...` convention) specifically so
# `test_frontend_ab_merge.py` can drive the real report-reading code path
# with a fixture directory -- including a REAL, committed, envelope-
# trimmed `finetune-run` report -- rather than only ever exercising it
# through this script's own DRY_RUN stub (see that module's own doc for
# why an inline heredoc could not catch this driver's own pod-p421b
# defect).
MERGE_OUT="$OUT_DIR/report.json"
python3 "$DIR/frontend_ab_merge.py" "$RAW_DIR" "$MERGE_OUT" "$FRONTEND_AB_SERIAL_TAIL_RATIO" \
  "$N_ITEMS_PER_STEP" "$TIP_SHA" "$BASE_SHA" "$BOX" "$FRONTEND_AB_DRY_RUN"
MERGE_RC=$?
if [ "$MERGE_RC" -ne 0 ]; then
  echo "::error::merge step failed (exit $MERGE_RC)" >&2
  exit 1
fi

REPORT_STATUS="$(python3 -c "import json; print(json.load(open('$MERGE_OUT'))['status'])")"

# ── artifact write (real runs only) ─────────────────────────────────────
TIP_SHORT="${TIP_SHA:0:7}"
ARTIFACT_NAME="${DATE_ONLY}-frontend-${TIP_SHORT}-${BOX_SLUG}.json"
ARTIFACT_PATH="$REPO_ROOT/crates/jammi-kernels/artifacts/cuda-runs/$ARTIFACT_NAME"

if [ "$FRONTEND_AB_DRY_RUN" = "1" ]; then
  DRY_ARTIFACT_PATH="$OUT_DIR/artifact-dry-run.json"
  python3 - "$MERGE_OUT" "$DRY_ARTIFACT_PATH" "$TIP_SHA" "$BOX" <<'PYEOF'
import json
import sys
from pathlib import Path

merge_path, out_path, tip_sha, box = sys.argv[1:5]
report = json.loads(Path(merge_path).read_text())
artifact = {
    "schema_version": 1,
    "git_sha_unresolved": tip_sha,
    "box": box,
    "producer": {
        "path": "ci/scripts/perf/frontend_ab.sh",
        "kind": "none",
        "invocation": "ci/scripts/perf/frontend_ab.sh (FRONTEND_AB_DRY_RUN=1)",
        "gating": "none",
    },
    "status": report["status"],
    "report": report,
}
Path(out_path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
PYEOF
  echo "::notice::FRONTEND_AB_DRY_RUN=1 -- artifact written to $DRY_ARTIFACT_PATH, NOT to $ARTIFACT_PATH (the real committed artifacts directory is never touched under DRY_RUN)."
else
  if [ -e "$ARTIFACT_PATH" ]; then
    echo "::error::$ARTIFACT_PATH already exists -- refusing to overwrite a committed artifact. Remove it first if this run is meant to replace it." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$ARTIFACT_PATH")"
  python3 - "$MERGE_OUT" "$ARTIFACT_PATH" "$TIP_SHA" "$BOX" <<'PYEOF'
import json
import sys
from pathlib import Path

merge_path, out_path, tip_sha, box = sys.argv[1:5]
report = json.loads(Path(merge_path).read_text())
artifact = {
    "schema_version": 1,
    "git_sha": tip_sha,
    "box": box,
    "producer": {
        "path": "ci/scripts/perf/frontend_ab.sh",
        "kind": "script",
        "invocation": "ci/scripts/perf/frontend_ab.sh",
        "gating": "none",
    },
    "status": report["status"],
    "report": report,
}
Path(out_path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
PYEOF
  echo "=== artifact written: $ARTIFACT_PATH ==="
fi

echo
echo "=== raw legs + merged report: ${OUT_DIR} ==="
if [ "$REPORT_STATUS" != "GREEN" ]; then
  exit 1
fi
exit 0
