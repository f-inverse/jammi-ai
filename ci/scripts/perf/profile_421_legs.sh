#!/usr/bin/env bash
# The issue #421 tower training-step profile leg driver (P1-b(vi); CONTRACT
# `scratchpad/contract-421-profile.md` v2). Generalizes
# `profile_356_legs.sh` -- same method skeleton (two `jammi-bench
# finetune-run` invocations per leg, N steps then M steps over the SAME
# declared workload with only the train-corpus row count differing, each
# traced with `nsys profile --trace=cuda`, exported to sqlite, differenced
# with `kernel_census.py`) -- by `--task`/TOWER instead of by BERT-family
# model: CLIP-text (`text_embedding`), OpenCLIP-vision (`image_embedding`)
# and HTSAT (`audio_embedding`), 4 legs each = 12 legs.
#
# WHAT IS DIFFERENT FROM `profile_356_legs.sh`, and why (every difference is
# a CONTRACT clause, not a preference):
#
#   * `--task` selects the TOWER, so the corpus PRODUCER differs per leg:
#     `gen_fixed_width_corpus.py` (text), `gen_fixed_shape_image_corpus.py`
#     (224 px images), `gen_fixed_length_audio_corpus.py` (9.5 s @ 48 kHz).
#     All three now emit a HELD-OUT split too (`--heldout-rows`), which this
#     driver requires: `finetune-run` demands `--heldout-ids` +
#     `--heldout-jsonl` on EVERY leg, and a media leg has no committed
#     fixture to fall back on.
#   * `--objective triplet` is PINNED on every leg (contract "Workload
#     shape"): media triplets and TEXT triplets both encode as ONE joined
#     forward of `rows = 3B`, while MNRL/pairs gives `2B` -- and MNRL is
#     refused outright for a media task. Pinning triplet is what makes the
#     three towers' walls comparable at all. `profile_356_legs.sh` pinned
#     `mnrl`; this is NOT an inherited value.
#   * D LEGS (the eager twins) set `JAMMI_KERNELS_DISABLE` **and** claim it
#     back with `--expect-kernels-disabled`. A/D legs are otherwise
#     identical. `JAMMI_KERNELS_DISABLE` is exported for the D legs ONLY --
#     never process-wide -- so an A leg cannot inherit it.
#   * `--lora-init zeros_b` is passed EXPLICITLY on every leg. It is also
#     the default, so this changes nothing about what runs; it is stated so
#     the pinned value is visible in the recorded command line rather than
#     resting on a default that could move.
#
# THIS SCRIPT IS A GPU-POD DRIVER, not a CI step: real legs need an A100 (or
# equivalent), an installed `nsys`, and two real checkpoint directories
# (`$MODEL_DIR_CLIP` for both CLIP towers, `$MODEL_DIR_CLAP` for HTSAT).
# `PROFILE_421_LEGS_DRY_RUN=1` makes the WHOLE pipeline safe to exercise
# hermetically: `$NSYS_BIN`/`$BENCH_BIN` are swapped for hermetic fake
# stand-ins generated into `$DRY_RUN_STUB_DIR` and ACTUALLY EXECUTED through
# the EXACT SAME capture path a real leg uses (never a hand-shaped bypass --
# a stub report written directly by this script, skipping the capture
# machinery, structurally could not catch a bug IN that machinery);
# `_print_cmd` still prints the real, would-be production command line, on
# stderr, for operator visibility.
#
# PRECONDITION GUARD: this script REFUSES, before any leg runs, unless the
# bench binary carries every P1-b flag these legs depend on --
# `--expect-kernels-disabled`, `--lora-init`, `--task`, `--objective` -- and
# unless the three corpus producers accept `--heldout-rows`/`--heldout-batch`.
# Each is probed CHEAPLY (a `--help` scan plus one CPU-hermetic producer
# invocation into a tempdir), never requiring a GPU or a training run just to
# check readiness, and never merely trusted because a sibling PR claims to
# have landed it. `PROFILE_421_LEGS_PREFLIGHT_ONLY=1` runs the preflight and
# exits before the leg sweep -- the hook the hermetic test drives against a
# fake `$BENCH_BIN` stub to exercise each distinguishable outcome.
#
# STEP-COUNT PIN (contract "Legs per tower"): N=100, M=600 for every leg,
# batch 8 -- so a leg's train corpus is `8 * steps` rows and the differenced
# denominator is `M - N = 500` steps. `--epochs 1` everywhere, so steps ARE
# `rows / batch` (there is no `--steps` flag).
#
# EVERY DECLARED LEG PARAMETER REACHES THE BINARY AND IS RECORDED: task,
# batch, dtype, LoRA selectors, `--lora-init`, `--objective`, the fixed
# `--eval-cadence`, `--max-seq-length`, and (D legs) the disable list and its
# `--expect-kernels-disabled` claim all appear literally in `run_traced`'s
# own `cmd` array, and every one of them is written to the leg's
# `manifest.json` alongside `status`/`reason`/`census_ok`. A leg-level
# failure NEVER aborts the sweep: it is RECORDED in that leg's manifest and
# the next leg runs.
#
# Env vars (all required for a real run; DRY_RUN relaxes all but OUT_DIR):
#   NSYS_BIN         path to the nsys binary (default: the contract's pinned
#                    `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys`)
#   BENCH_BIN        path to the jammi-bench binary (default:
#                    `$CARGO_TARGET_DIR/release/jammi-bench` or
#                    `$REPO_ROOT/target/release/jammi-bench`)
#   MODEL_DIR_CLIP   OpenCLIP ViT-B-32 checkpoint dir (BOTH CLIP towers)
#   MODEL_DIR_CLAP   HF `laion/clap-htsat-fused` checkpoint dir
#   OUT_DIR          output directory (default `$REPO_ROOT/.profile-421-legs/<ts>`)
#   PROFILE_421_LEGS_DRY_RUN         "1" swaps in the hermetic fakes
#   PROFILE_421_LEGS_PREFLIGHT_ONLY  "1" preflight then exit
#   PROFILE_421_LEGS_ONLY            optional comma-separated leg-id filter
#                                    (e.g. "clip-text-A1,htsat-D2"); every
#                                    named id is validated against the leg
#                                    table BEFORE any leg runs (a typo
#                                    refuses loudly, exit 2), and every leg
#                                    about to run is refused if a manifest
#                                    for it already exists under $OUT_DIR
#   PROFILE_421_STEPS_N/M            override the pinned 100/600
#
# Hermetic self-tests: `python3 ci/scripts/perf/test_profile_421_legs_dry_run.py`.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

PROFILE_421_LEGS_DRY_RUN="${PROFILE_421_LEGS_DRY_RUN:-0}"
PROFILE_421_LEGS_PREFLIGHT_ONLY="${PROFILE_421_LEGS_PREFLIGHT_ONLY:-0}"
NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/2025.3.2/bin/nsys}"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BENCH_BIN="${BENCH_BIN:-$TARGET_DIR/release/jammi-bench}"
MODEL_DIR_CLIP="${MODEL_DIR_CLIP:-}"
MODEL_DIR_CLAP="${MODEL_DIR_CLAP:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.profile-421-legs/$TS}"
PROFILE_421_LEGS_ONLY="${PROFILE_421_LEGS_ONLY:-}"
STEPS_N="${PROFILE_421_STEPS_N:-100}"
STEPS_M="${PROFILE_421_STEPS_M:-600}"

# ── Contract-pinned workload constants (`## Declared workload`) ──────────
# Every one of these is a CONTRACT value, not a tuning knob: changing one
# here changes what the pre-registered legs measure.
BATCH=8                    # rows = 3B = 24 per step under --objective triplet
SEED=42
EVAL_CADENCE=1
OBJECTIVE=triplet          # PINNED: rows = 3B on every tower (see module doc)
LORA_INIT=zeros_b          # PINNED: the wire default, stated explicitly
CLIP_TEXT_SEQ=77           # CLIP-text context; the corpus exceeds it per row
IMAGE_SIZE=224             # OpenCLIP ViT-B-32 input side
AUDIO_SECONDS=9.5          # strictly below nb_max_samples: repeat-pad branch
AUDIO_SAMPLE_RATE=48000    # 9.5 s @ 48 kHz = 456000 frames exactly
HELDOUT_ROWS=8             # one --batch of held-out rows (8 % 8 == 0)
MEDIA_FAMILIES=6           # 4 train + 2 reserved held-out (producer refuses <4)
MEDIA_HELDOUT_FAMILIES=2

# Full LoRA site sets per tower (`finetune_run.rs::tower_site_names`).
CLIP_FULL="in_proj,out_proj,c_fc,c_proj"
CLAP_FULL="query,key,value,attention_output,intermediate_dense,output_dense,reduction,linear1,linear2"

# The D-leg disable lists (contract legs table). D1 is the JOINT eager twin
# over that tower's realized-gain chains; D2 isolates C-LORA. Set as
# `JAMMI_KERNELS_DISABLE` on those legs ONLY, and claimed back on the SAME
# command line with `--expect-kernels-disabled`.
#
# D1 IS PER-TOWER, and this is NOT a simplification of the contract's single
# D1 row -- it is what that row means once the towers' actual seams are read
# (contract "Scope facts"): `gelu_erf_fused` is reachable on HTSAT ONLY.
# The CLIP towers' MLP activation is `quick_gelu`
# (`jammi-encoders/src/activations.rs`, `open_clip_vision.rs:21`), which has
# no fused seam and therefore no `admit` key at all -- C-MLP-CLIP is a port
# CANDIDATE, not a realized chain. Naming `gelu_erf_fused` on a CLIP leg
# would put an op key in `JAMMI_KERNELS_DISABLE` that never disables a live
# dispatch, which `jammi_kernels::admission::unmatched_disables()` reports
# and `finetune-run` refuses as an INVALID run: every CLIP D1 leg would fail
# outright. So:
#
#   CLIP D1  = lora_linear_fused + layer_norm_fused  -> C-LN = D1 - D2
#   HTSAT D1 = those two + gelu_erf_fused            -> C-LN + C-GELU = D1 - D2
#
# The HTSAT D1 leg additionally PRESUPPOSES P1-a (HTSAT's MLP routed through
# `activations::gelu_erf(x, training)` rather than calling `Tensor::gelu_erf`
# directly at `htsat_audio.rs:1064`/`:1635`). If P1-a has not landed, that
# leg refuses with "gelu_erf_fused ... never disabled a live dispatch this
# run" -- the CORRECT outcome (the leg is invalid, not a datum), recorded in
# its manifest while the other 11 legs continue.
D1_KEYS_CLIP="lora_linear_fused,layer_norm_fused"
D1_KEYS_HTSAT="lora_linear_fused,layer_norm_fused,gelu_erf_fused"
D2_KEYS="lora_linear_fused"

mkdir -p "$OUT_DIR"

if [ "$PROFILE_421_LEGS_DRY_RUN" != "1" ]; then
  if [ -z "$MODEL_DIR_CLIP" ] || [ -z "$MODEL_DIR_CLAP" ]; then
    echo "::error::MODEL_DIR_CLIP and MODEL_DIR_CLAP must both be set for a real run." >&2
    exit 2
  fi
  for f in "$NSYS_BIN" "$BENCH_BIN"; do
    if [ ! -x "$f" ]; then
      echo "::error::$f is not an executable file -- refusing before any leg runs." >&2
      exit 2
    fi
  done
else
  MODEL_DIR_CLIP="${MODEL_DIR_CLIP:-/root/checkpoints/open-clip-vit-b-32-DRY-RUN-PLACEHOLDER}"
  MODEL_DIR_CLAP="${MODEL_DIR_CLAP:-/root/checkpoints/clap-htsat-fused-DRY-RUN-PLACEHOLDER}"
  echo "::warning::PROFILE_421_LEGS_DRY_RUN=1 -- nothing is read from MODEL_DIR_*/NSYS_BIN/BENCH_BIN."
fi

# `nsys --version` ONCE at startup, guarded (the `profile_356_legs.sh`
# lesson: an unguarded per-leg `--version` capture aborted a whole sweep
# mid-leg under `set -e`, with no manifest for the failing leg). A failure
# here degrades to a recorded string, never a script-wide abort.
NSYS_VERSION="unknown"
if [ "$PROFILE_421_LEGS_DRY_RUN" != "1" ]; then
  NSYS_VERSION="$("$NSYS_BIN" --version 2>&1 | head -1)" || NSYS_VERSION="unknown (nsys --version failed)"
fi

# --- trace echo: ALWAYS to stderr, never stdout -- a trace line sharing
# stdout with a captured child's own JSON report is exactly how a report
# envelope gets polluted.
_print_cmd() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

# --- state-changing command wrapper: echoes what it would run (stderr);
# under DRY_RUN never executes. Used for invocations whose stdout is never
# captured into a report file (corpus generation, `kernel_census.py`).
# `run_traced`'s own nsys/bench invocation does NOT go through this.
run_cmd() {
  _print_cmd "$@"
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# --- provenance cross-check (unification contract C5.1): refuse BEFORE any
# leg runs if the binary's own baked identity does not match this checkout.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$PROFILE_421_LEGS_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BENCH_BIN" provenance 2>&1)" || { echo "::error::'$BENCH_BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BENCH_BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BENCH_BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# =====================================================================
# Preflight: every P1-b surface these legs depend on must be PRESENT in
# the landed tree, probed rather than assumed. A setup failure here is
# loud and intentional (`exit 1`), never a raw `set -e` abort -- preflight
# gates the entire run, so aborting is correct; the fix is HOW it aborts.
# =====================================================================
preflight_probe() {
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    echo "::notice::PROFILE_421_LEGS_DRY_RUN=1 -- skipping the real preflight probe (nothing to probe)."
    return 0
  fi

  local missing=()
  local help_out
  if ! help_out="$("$BENCH_BIN" finetune-run --help 2>&1)"; then
    echo "::error::preflight_probe: '$BENCH_BIN finetune-run --help' failed:" >&2
    echo "$help_out" >&2
    exit 1
  fi
  # Every flag this driver PASSES, probed by name. `--objective`/`--task`
  # predate P1-b but are pinned by this driver, so an absent one would
  # silently change the workload rather than fail -- probed too.
  local flag
  for flag in --expect-kernels-disabled --lora-init --task --objective --heldout-ids --heldout-jsonl; do
    if ! printf '%s' "$help_out" | grep -q -- "$flag"; then
      missing+=("'finetune-run --help' has no $flag flag")
    fi
  done
  # `--lora-init`'s accepted VALUE, not merely the flag: a build carrying
  # the flag but not this driver's pinned token would refuse every leg.
  if ! printf '%s' "$help_out" | grep -q -- "$LORA_INIT"; then
    missing+=("'finetune-run --help' does not mention the pinned --lora-init value '$LORA_INIT'")
  fi

  # The three producers' held-out mode, probed by ACTUALLY RUNNING each one
  # into a tempdir (a `--help` scan would not catch a flag that parses but
  # emits nothing). Cheap: tiny corpora, no network, no GPU.
  local probe_dir
  if ! probe_dir="$(mktemp -d)"; then
    echo "::error::preflight_probe: mktemp -d failed -- cannot set up the producer probes." >&2
    exit 1
  fi
  if ! python3 "$DIR/gen_fixed_width_corpus.py" --rows 2 --min-wordpieces 4 --seed 1 \
      --out "$probe_dir/text/train.jsonl" --heldout-rows 2 --heldout-batch 2 >/dev/null 2>&1; then
    # `--out`'s parent must exist for the text producer; create and retry
    # once so a missing directory is not misreported as a missing flag.
    mkdir -p "$probe_dir/text"
    if ! python3 "$DIR/gen_fixed_width_corpus.py" --rows 2 --min-wordpieces 4 --seed 1 \
        --out "$probe_dir/text/train.jsonl" --heldout-rows 2 --heldout-batch 2 >/dev/null 2>&1; then
      missing+=("gen_fixed_width_corpus.py does not support --heldout-rows/--heldout-batch")
    fi
  fi
  if ! python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows 2 --size 8 --seed 1 \
      --out-dir "$probe_dir/img" --families "$MEDIA_FAMILIES" \
      --heldout-rows 2 --heldout-batch 2 >/dev/null 2>&1; then
    missing+=("gen_fixed_shape_image_corpus.py does not support --heldout-rows/--heldout-batch")
  fi
  if ! python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows 2 --seconds 0.05 \
      --sample-rate 16000 --seed 1 --out-dir "$probe_dir/aud" --families "$MEDIA_FAMILIES" \
      --heldout-rows 2 --heldout-batch 2 >/dev/null 2>&1; then
    missing+=("gen_fixed_length_audio_corpus.py does not support --heldout-rows/--heldout-batch")
  fi
  # The held-out FILES themselves, by the exact names this driver passes to
  # `--heldout-ids`/`--heldout-jsonl`: a producer that accepted the flags
  # but wrote them elsewhere would fail every leg at load time.
  local d
  for d in text img aud; do
    if [ ! -f "$probe_dir/$d/heldout_ids.txt" ] || [ ! -f "$probe_dir/$d/heldout_triplets.jsonl" ]; then
      missing+=("the $d producer did not emit heldout_ids.txt + heldout_triplets.jsonl")
    fi
  done
  rm -rf "$probe_dir"

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::profile_421_legs: refusing -- precondition(s) not yet landed:" >&2
    local m
    for m in "${missing[@]}"; do
      echo "  - $m" >&2
    done
    exit 1
  fi
  echo "::notice::profile_421_legs: preflight OK -- every pinned flag is present and all three producers emit a held-out split."
}

preflight_probe

if [ "$PROFILE_421_LEGS_PREFLIGHT_ONLY" = "1" ]; then
  echo "::notice::PROFILE_421_LEGS_PREFLIGHT_ONLY=1 -- preflight passed, exiting before the leg sweep."
  exit 0
fi

# =====================================================================
# Leg table (contract legs table, 4 legs x 3 towers = 12) --
# "id|tower|task|dtype|target_modules|disable_keys"
# `disable_keys` empty = an A leg (no JAMMI_KERNELS_DISABLE, no claim).
# =====================================================================
LEGS=(
  "clip-text-A1|clip-text|text_embedding|f32|$CLIP_FULL|"
  "clip-text-A2|clip-text|text_embedding|bf16|$CLIP_FULL|"
  "clip-text-D1|clip-text|text_embedding|f32|$CLIP_FULL|$D1_KEYS_CLIP"
  "clip-text-D2|clip-text|text_embedding|f32|$CLIP_FULL|$D2_KEYS"
  "clip-vision-A1|clip-vision|image_embedding|f32|$CLIP_FULL|"
  "clip-vision-A2|clip-vision|image_embedding|bf16|$CLIP_FULL|"
  "clip-vision-D1|clip-vision|image_embedding|f32|$CLIP_FULL|$D1_KEYS_CLIP"
  "clip-vision-D2|clip-vision|image_embedding|f32|$CLIP_FULL|$D2_KEYS"
  "htsat-A1|htsat|audio_embedding|f32|$CLAP_FULL|"
  "htsat-A2|htsat|audio_embedding|bf16|$CLAP_FULL|"
  "htsat-D1|htsat|audio_embedding|f32|$CLAP_FULL|$D1_KEYS_HTSAT"
  "htsat-D2|htsat|audio_embedding|f32|$CLAP_FULL|$D2_KEYS"
)

ALL_LEG_IDS=()
for _leg_spec in "${LEGS[@]}"; do
  ALL_LEG_IDS+=("${_leg_spec%%|*}")
done

if [ -n "$PROFILE_421_LEGS_ONLY" ]; then
  IFS=',' read -r -a _requested_leg_ids <<< "$PROFILE_421_LEGS_ONLY"
  _unknown_leg_ids=()
  for _requested in "${_requested_leg_ids[@]}"; do
    _known=0
    for _leg_id in "${ALL_LEG_IDS[@]}"; do
      if [ "$_requested" = "$_leg_id" ]; then
        _known=1
        break
      fi
    done
    if [ "$_known" -eq 0 ]; then
      _unknown_leg_ids+=("$_requested")
    fi
  done
  if [ "${#_unknown_leg_ids[@]}" -gt 0 ]; then
    echo "::error::PROFILE_421_LEGS_ONLY names unknown leg id(s): ${_unknown_leg_ids[*]} -- known ids are: ${ALL_LEG_IDS[*]}" >&2
    exit 2
  fi
fi

# Refuse before any leg runs if $OUT_DIR already carries a manifest for a
# leg about to run -- both manifest shapes this driver can write (the normal
# per-leg-subdirectory one and the flat fallback) -- rather than silently
# overwriting a prior run's recorded result.
_existing_manifest_conflicts=()
for _leg_id in "${ALL_LEG_IDS[@]}"; do
  if [ -n "$PROFILE_421_LEGS_ONLY" ]; then
    case ",$PROFILE_421_LEGS_ONLY," in
      *",$_leg_id,"*) ;;
      *) continue ;;
    esac
  fi
  if [ -f "$OUT_DIR/$_leg_id/manifest.json" ] || [ -f "$OUT_DIR/${_leg_id}.manifest.json" ]; then
    _existing_manifest_conflicts+=("$_leg_id")
  fi
done
if [ "${#_existing_manifest_conflicts[@]}" -gt 0 ]; then
  echo "::error::OUT_DIR ($OUT_DIR) already has a manifest for leg(s): ${_existing_manifest_conflicts[*]} -- refusing to silently overwrite; use a fresh OUT_DIR or narrow PROFILE_421_LEGS_ONLY to exclude them." >&2
  exit 2
fi

GOLDEN_FIXTURE="$REPO_ROOT/ci/scripts/perf/fixtures/finetune_run_golden/bert_fused.json"

# Hermetic fake nsys/bench stand-ins, used ONLY under DRY_RUN -- generated
# ONCE, reused for every leg, so the DRY_RUN sweep drives the EXACT SAME
# capture path (the exec-wrapper, the stderr redirect, the post-write
# envelope validation) a real GPU-pod run uses. `fake_nsys.sh` is
# deliberately CHATTY on stdout for BOTH subcommands (mirroring real nsys)
# to prove the capture path discards it; `fake_bench.sh` emits the
# golden-derived envelope on ITS OWN stdout (never writes a file directly),
# exercising the exec-wrapper's redirect for real.
DRY_RUN_STUB_DIR=""
if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
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
# $1=steps_measured $2=golden_json_path $3=task $4=disable_keys(csv, may be
# empty) -- emits the golden-derived Report envelope on ITS OWN stdout
# (never writes a file directly), so the exec-wrapper's own "redirect this
# child's stdout to the report file" mechanism is what actually produces
# the report, for real. The overridden fields are exactly the ones this
# driver's own readers consume, so a reader bug is exercised here rather
# than papered over.
python3 -c '
import copy, json, sys
steps_measured, golden_path, task, disable_csv = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
)
golden = json.load(open(golden_path))
tier = copy.deepcopy(golden["tiers"]["finetune_run"])
# Proportional to steps_measured (never a flat constant): the M-step run
# always declares more steps than the N-step run, so wall_m > wall_n > 0
# holds across the fake pair and the wall-pair domain check in
# kernel_census.py is exercised as a real same-workload N<M pair would.
tier["train_run_wall_s"] = 0.01 * steps_measured
tier["steps_measured"] = steps_measured
tier["task"] = task
tier["lora_init"] = "zeros_b"
expected = sorted(k for k in disable_csv.split(",") if k)
tier["kernels_disabled_expected"] = expected
tier["kernels_disabled_requested"] = expected
tier["kernels_disabled_fired"] = expected
# A D leg is an EAGER twin: its disabled keys dispatched eager, never fused.
tier["lora_linear_fused_dispatches"] = 0 if expected else 1
tier["lora_linear_eager_dispatches"] = 1 if expected else 0
tier["lora_epilogue_fused_dispatches"] = 0
tier["lora_epilogue_eager_dispatches"] = 1
if task != "text_embedding":
    tier["train_media_sha256"] = "a" * 64
    tier["heldout_media_sha256"] = "b" * 64
report = {"tool": "dry-run", "profile_421_dry_run": True, "tiers": {"finetune_run": tier}}
json.dump(report, sys.stdout)
' "$1" "$2" "$3" "$4"
echo "fake_bench: stderr noise too, never captured into the report" >&2
FAKE_BENCH_EOF
  chmod +x "$DRY_RUN_STUB_DIR/fake_bench.sh"
fi

# Validates "$1" parses as JSON and carries a `tiers.finetune_run` OBJECT --
# called right after the traced invocation writes the report, BEFORE any
# reader touches it. A parse failure or a missing/wrong-shaped tier is a
# recorded leg-INVALID reason, never a downstream KeyError from a reader.
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

# Reads one required numeric/string field off `tiers.finetune_run` --
# REFUSES (nonzero exit, clear stderr message, no stray stdout) rather than
# KeyError-ing or degrading to an empty value when it is absent.
_tier_field() {
  python3 -c '
import json, sys
path, field = sys.argv[1], sys.argv[2]
d = json.load(open(path))
tier = d.get("tiers", {}).get("finetune_run")
if tier is None or field not in tier:
    print("::error::_tier_field: no tiers.finetune_run." + field + " in " + path, file=sys.stderr)
    sys.exit(1)
print(json.dumps(tier[field]))
' "$1" "$2"
}

# The D-leg eager-twin proof, read off the REPORT rather than trusted
# because the env var was exported: `kernels_disabled_expected` must equal
# the keys this leg claimed, and `kernels_disabled_requested` must contain
# every one of them. The binary itself already refuses a leg that fails
# either (that is what `--expect-kernels-disabled` IS), so this is the
# driver's own independent record of the same fact -- and it catches the one
# case the binary cannot: a leg whose report came from a DIFFERENT
# invocation than the one this script believes it ran.
_check_expected_disables() {
  python3 -c '
import json, sys
path, claimed_csv = sys.argv[1], sys.argv[2]
claimed = sorted(k for k in claimed_csv.split(",") if k)
d = json.load(open(path))
tier = d.get("tiers", {}).get("finetune_run") or {}
expected = tier.get("kernels_disabled_expected")
requested = tier.get("kernels_disabled_requested")
if expected is None or requested is None:
    print("::error::_check_expected_disables: the report carries no "
          "kernels_disabled_expected/kernels_disabled_requested pair (a build predating "
          "issue #421 P1-b(i)): " + path, file=sys.stderr)
    sys.exit(1)
if sorted(expected) != claimed:
    print("::error::_check_expected_disables: report claims " + repr(sorted(expected)) +
          " but this leg declared " + repr(claimed) + ": " + path, file=sys.stderr)
    sys.exit(1)
missing = [k for k in claimed if k not in requested]
if missing:
    print("::error::_check_expected_disables: claimed key(s) " + repr(missing) +
          " absent from the process-resolved JAMMI_KERNELS_DISABLE " + repr(sorted(requested)) +
          ": " + path, file=sys.stderr)
    sys.exit(1)
' "$1" "$2"
}

# One nsys-traced finetune-run: writes "$out_json" (the run's own JSON
# report, VALIDATED before return) and "$out_sqlite" (the nsys sqlite
# export). Returns the underlying command's exit status; `run_leg` decides
# what a nonzero return means for the leg.
#
# NEVER CAPTURES THE NSYS-WRAPPED PROCESS'S OWN STDOUT AS THE REPORT: the
# bench child's stdout is redirected to "$out_json" INSIDE the traced
# invocation via a `bash -c 'exec ... > "$0"'` wrapper, so ONLY that child's
# own stdout reaches the report file regardless of what nsys prints to ITS
# stdout (`profile_356_legs.sh`'s own recorded regression: a naive
# `run_cmd ... > "$out_json"` captured the trace echo AND nsys progress AND
# the child, so `json.load` failed on EVERY real leg).
#
# `JAMMI_KERNELS_DISABLE` is exported for THIS INVOCATION ONLY (via `env`),
# never process-wide: an A leg run after a D leg must not inherit it.
run_traced() {
  local model_dir="$1" task="$2" train_jsonl="$3" heldout_ids="$4" heldout_jsonl="$5" \
        dtype="$6" target_modules="$7" seq_len="$8" disable_keys="$9" \
        steps_this_run="${10}" work_dir="${11}" run_prefix="${12}" out_json="${13}" \
        out_sqlite="${14}"

  if ! mkdir -p "$work_dir"; then
    echo "::error::run_traced: could not create $work_dir" >&2
    return 1
  fi

  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused --task "$task"
    --train-jsonl "$train_jsonl" --heldout-ids "$heldout_ids" --heldout-jsonl "$heldout_jsonl"
    --seed "$SEED" --epochs 1 --batch "$BATCH" --objective "$OBJECTIVE"
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$dtype"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --lora-init "$LORA_INIT"
    --target-modules "$target_modules"
    --max-seq-length "$seq_len"
    --eval-cadence "$EVAL_CADENCE"
    --work-dir "$work_dir" --cuda 0
  )
  local -a env_prefix=()
  if [ -n "$disable_keys" ]; then
    env_prefix=(env "JAMMI_KERNELS_DISABLE=$disable_keys")
    cmd+=(--expect-kernels-disabled "$disable_keys")
  fi

  # What actually execs -- swapped for the hermetic fakes under DRY_RUN;
  # $NSYS_BIN/cmd[] (via _print_cmd) always show the REAL, would-be
  # production command line regardless of which binaries actually run.
  local exec_nsys_bin="$NSYS_BIN"
  local -a exec_cmd=("${env_prefix[@]}" "${cmd[@]}")
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    exec_nsys_bin="$DRY_RUN_STUB_DIR/fake_nsys.sh"
    exec_cmd=(
      "${env_prefix[@]}"
      "$DRY_RUN_STUB_DIR/fake_bench.sh" "$steps_this_run" "$GOLDEN_FIXTURE" "$task" "$disable_keys"
    )
  fi

  local -a wrapped=(bash -c 'exec "$1" "${@:2}" > "$0"' "$out_json" "${exec_cmd[@]}")

  _print_cmd "${env_prefix[@]}" "$NSYS_BIN" profile --trace=cuda -o "$run_prefix" --force-overwrite=true -- "${cmd[@]}"
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
      printf '%s\n' "$validate_err" | tee -a "$run_prefix.stderr" >&2
      echo "::error::run_traced: $out_json failed report-envelope validation -- see $run_prefix.stderr" >&2
      rc=1
    fi
  fi
  return "$rc"
}

# One leg's corpus pair (N and M) plus its held-out split, per TOWER.
# Echoes "train_n<TAB>train_m<TAB>heldout_ids<TAB>heldout_jsonl" on success;
# returns nonzero (with a stderr message) on failure, which `run_leg`
# records as this leg's own INVALID reason.
#
# The N corpus is a PREFIX of the M corpus for text (the producer's own
# stream property) and a smaller row list over the SAME image/audio files
# for media -- either way the two runs differ ONLY in row count, which is
# what the N/M wall differencing requires. The HELD-OUT split is generated
# ONCE per leg and shared by both runs (it is not part of the differenced
# workload -- `--eval-cadence` is fixed, so its cost cancels).
provision_corpus() {
  local tower="$1" leg_dir="$2"
  local rows_n=$(( BATCH * STEPS_N ))
  local rows_m=$(( BATCH * STEPS_M ))
  local dir_n="$leg_dir/corpus_n" dir_m="$leg_dir/corpus_m"

  case "$tower" in
    clip-text)
      mkdir -p "$dir_n" "$dir_m"
      run_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_n" \
        --min-wordpieces "$CLIP_TEXT_SEQ" --seed "$SEED" --out "$dir_n/train.jsonl" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" \
        --min-wordpieces "$CLIP_TEXT_SEQ" --seed "$SEED" --out "$dir_m/train.jsonl" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
        : > "$dir_n/train.jsonl"; : > "$dir_m/train.jsonl"
        : > "$dir_n/heldout_ids.txt"; : > "$dir_n/heldout_triplets.jsonl"
      fi
      printf '%s\t%s\t%s\t%s\n' "$dir_n/train.jsonl" "$dir_m/train.jsonl" \
        "$dir_n/heldout_ids.txt" "$dir_n/heldout_triplets.jsonl"
      ;;
    clip-vision)
      run_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$rows_n" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$dir_n" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$rows_m" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$dir_m" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
        mkdir -p "$dir_n" "$dir_m"
        : > "$dir_n/triplets.jsonl"; : > "$dir_m/triplets.jsonl"
        : > "$dir_n/heldout_ids.txt"; : > "$dir_n/heldout_triplets.jsonl"
      fi
      printf '%s\t%s\t%s\t%s\n' "$dir_n/triplets.jsonl" "$dir_m/triplets.jsonl" \
        "$dir_n/heldout_ids.txt" "$dir_n/heldout_triplets.jsonl"
      ;;
    htsat)
      run_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$rows_n" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$dir_n" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$rows_m" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$dir_m" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
        mkdir -p "$dir_n" "$dir_m"
        : > "$dir_n/triplets.jsonl"; : > "$dir_m/triplets.jsonl"
        : > "$dir_n/heldout_ids.txt"; : > "$dir_n/heldout_triplets.jsonl"
      fi
      printf '%s\t%s\t%s\t%s\n' "$dir_n/triplets.jsonl" "$dir_m/triplets.jsonl" \
        "$dir_n/heldout_ids.txt" "$dir_n/heldout_triplets.jsonl"
      ;;
    *)
      echo "::error::provision_corpus: unknown tower '$tower'" >&2
      return 1
      ;;
  esac
}

# Writes a leg's manifest.json -- callers guard the CALL itself (a write
# failure is sweep-fatal: without it this leg's result is unrecorded and
# unrecoverable, which this driver treats as fatal by deliberate design,
# stated loudly, never silently swallowed the way every OTHER leg-level
# failure is).
_write_manifest() {
  MANIFEST_LEG_ID="$1" MANIFEST_GIT_SHA="$2" MANIFEST_BOX="$3" MANIFEST_NSYS_VERSION="$4" \
  MANIFEST_TOWER="$5" MANIFEST_TASK="$6" MANIFEST_DTYPE="$7" MANIFEST_TARGET_MODULES="$8" \
  MANIFEST_DISABLE_KEYS="$9" MANIFEST_SEQ="${10}" MANIFEST_BATCH="${11}" \
  MANIFEST_OBJECTIVE="${12}" MANIFEST_LORA_INIT="${13}" MANIFEST_EVAL_CADENCE="${14}" \
  MANIFEST_STEPS_DECLARED_N="${15}" MANIFEST_STEPS_DECLARED_M="${16}" \
  MANIFEST_STEPS_MEASURED_N="${17}" MANIFEST_STEPS_MEASURED_M="${18}" \
  MANIFEST_STATUS="${19}" MANIFEST_REASON="${20}" MANIFEST_CENSUS_OK="${21}" \
  MANIFEST_CENSUS_EXIT="${22}" MANIFEST_DRY_RUN="${23}" \
  MANIFEST_HELDOUT_ROWS="${24}" MANIFEST_MEDIA_FRONT_END_N="${25}" \
  MANIFEST_MEDIA_FRONT_END_M="${26}" MANIFEST_OUT="${27}" \
  python3 -c '
import json, os


def _int_or_none(v):
    return int(v) if v not in ("", None) else None


def _json_or_none(v):
    return json.loads(v) if v not in ("", None) else None


manifest = {
    "leg_id": os.environ["MANIFEST_LEG_ID"],
    "git_sha": os.environ["MANIFEST_GIT_SHA"],
    "box": os.environ["MANIFEST_BOX"],
    "driver": "ci/scripts/perf/profile_421_legs.sh",
    "nsys_version": os.environ["MANIFEST_NSYS_VERSION"],
    "tower": os.environ["MANIFEST_TOWER"],
    "task": os.environ["MANIFEST_TASK"],
    "dtype": os.environ["MANIFEST_DTYPE"],
    "target_modules": os.environ["MANIFEST_TARGET_MODULES"].split(","),
    # The D-leg disable list, [] on an A leg -- the declared arm, recorded
    # next to the report-read proof the run actually honoured it.
    "kernels_disabled": [k for k in os.environ["MANIFEST_DISABLE_KEYS"].split(",") if k],
    "max_seq_length": int(os.environ["MANIFEST_SEQ"]),
    "batch": int(os.environ["MANIFEST_BATCH"]),
    "objective": os.environ["MANIFEST_OBJECTIVE"],
    "lora_init": os.environ["MANIFEST_LORA_INIT"],
    "eval_cadence": int(os.environ["MANIFEST_EVAL_CADENCE"]),
    "heldout_rows": int(os.environ["MANIFEST_HELDOUT_ROWS"]),
    "steps_declared": {
        "n": int(os.environ["MANIFEST_STEPS_DECLARED_N"]),
        "m": int(os.environ["MANIFEST_STEPS_DECLARED_M"]),
    },
    "steps_measured": {
        "n": _int_or_none(os.environ["MANIFEST_STEPS_MEASURED_N"]),
        "m": _int_or_none(os.environ["MANIFEST_STEPS_MEASURED_M"]),
    },
    # The DIRECT media front-end timer (contract P1-b(v)), read off each
    # run`s own report. `null` on a build whose `TrainingResult` does not
    # yet carry the seam -- recorded as null rather than omitted, so a
    # consumer can tell "not measured" from "measured as zero".
    "media_front_end_wall_s": {
        "n": _json_or_none(os.environ["MANIFEST_MEDIA_FRONT_END_N"]),
        "m": _json_or_none(os.environ["MANIFEST_MEDIA_FRONT_END_M"]),
    },
    "status": os.environ["MANIFEST_STATUS"],
    "reason": os.environ["MANIFEST_REASON"],
    "census_ok": os.environ["MANIFEST_CENSUS_OK"] == "true",
    "census_exit": _int_or_none(os.environ["MANIFEST_CENSUS_EXIT"]),
    "dry_run": os.environ["MANIFEST_DRY_RUN"] == "true",
}
json.dump(manifest, open(os.environ["MANIFEST_OUT"], "w"), indent=1)
'
}

# One leg, start to finish. Returns 0 for every OUTCOME this leg's own
# workload can produce -- a leg's own failure (corpus/run_traced/census/
# report-read) lives in its `manifest.json` (`status`/`reason`), never in
# this function's exit code, so one leg's OOM/refusal never discards any
# other leg. The ONE deliberate exception is a manifest WRITE failing,
# which is sweep-fatal and says so.
run_leg() {
  local spec="$1"
  IFS='|' read -r leg_id tower task dtype target_modules disable_keys <<< "$spec"

  if [ -n "$PROFILE_421_LEGS_ONLY" ]; then
    case ",$PROFILE_421_LEGS_ONLY," in
      *",$leg_id,"*) ;;
      *) echo "::notice::skipping $leg_id (not in PROFILE_421_LEGS_ONLY)"; return 0 ;;
    esac
  fi

  echo "=== leg $leg_id: tower=$tower task=$task dtype=$dtype target=$target_modules disable=${disable_keys:-<none>} steps=$STEPS_N/$STEPS_M ==="

  # BOTH CLIP towers live behind ONE checkpoint directory -- `--task` is
  # what selects which of the two gets the adapters (exactly why `task` is
  # an identity field on the emitted tier). `--max-seq-length` is the
  # TOKENIZER truncation cap: load-bearing for clip-text (it pins the
  # `[3B, 77]` shape) and inert for the two media towers, which never
  # tokenize -- passed there anyway, at the same value, so the flag set is
  # identical across legs and cannot itself be a difference between them.
  local model_dir seq_len
  case "$tower" in
    clip-text)   model_dir="$MODEL_DIR_CLIP"; seq_len="$CLIP_TEXT_SEQ" ;;
    clip-vision) model_dir="$MODEL_DIR_CLIP"; seq_len="$CLIP_TEXT_SEQ" ;;
    htsat)       model_dir="$MODEL_DIR_CLAP"; seq_len="$CLIP_TEXT_SEQ" ;;
  esac

  local dry_run_flag="false"
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then dry_run_flag="true"; fi

  local leg_dir="$OUT_DIR/$leg_id"
  if ! mkdir -p "$leg_dir"; then
    echo "::error::$leg_id: could not create $leg_dir -- writing a fallback manifest instead." >&2
    if ! _write_manifest \
        "$leg_id" "$SHA" "$(hostname)" "$NSYS_VERSION" "$tower" "$task" "$dtype" \
        "$target_modules" "$disable_keys" "$seq_len" "$BATCH" "$OBJECTIVE" "$LORA_INIT" \
        "$EVAL_CADENCE" "$STEPS_N" "$STEPS_M" "" "" "invalid" \
        "could not create leg directory $leg_dir" "false" "" "$dry_run_flag" \
        "$HELDOUT_ROWS" "" "" "$OUT_DIR/${leg_id}.manifest.json"; then
      echo "::error::$leg_id: could not write even the fallback manifest.json -- this IS sweep-fatal (results cannot be recorded); aborting." >&2
      exit 1
    fi
    return 0
  fi

  local leg_status="ok"
  local leg_reason=""
  local train_n="" train_m="" heldout_ids="" heldout_jsonl=""

  local corpus_line
  if corpus_line="$(provision_corpus "$tower" "$leg_dir")"; then
    IFS=$'\t' read -r train_n train_m heldout_ids heldout_jsonl <<< "$corpus_line"
  else
    leg_status="invalid"
    leg_reason="corpus provisioning failed for tower $tower (see this leg's stderr above)"
  fi

  local out_n="$leg_dir/run_n.json" out_m="$leg_dir/run_m.json"
  local sqlite_n="$leg_dir/run_n.sqlite" sqlite_m="$leg_dir/run_m.sqlite"

  if [ "$leg_status" = "ok" ]; then
    if ! run_traced "$model_dir" "$task" "$train_n" "$heldout_ids" "$heldout_jsonl" \
        "$dtype" "$target_modules" "$seq_len" "$disable_keys" "$STEPS_N" \
        "$leg_dir/work_n" "$leg_dir/run_n" "$out_n" "$sqlite_n"; then
      leg_status="invalid"
      leg_reason="run_traced (N-step run) failed (nsys/bench execution or report-envelope validation) -- see $leg_dir/run_n.stderr"
    fi
  fi
  if [ "$leg_status" = "ok" ]; then
    if ! run_traced "$model_dir" "$task" "$train_m" "$heldout_ids" "$heldout_jsonl" \
        "$dtype" "$target_modules" "$seq_len" "$disable_keys" "$STEPS_M" \
        "$leg_dir/work_m" "$leg_dir/run_m" "$out_m" "$sqlite_m"; then
      leg_status="invalid"
      leg_reason="run_traced (M-step run) failed (nsys/bench execution or report-envelope validation) -- see $leg_dir/run_m.stderr"
    fi
  fi

  # D legs only: the eager-twin proof, read off BOTH reports. An A leg
  # makes no claim, so this check does not fire for it -- which is exactly
  # why the D legs' own claim has to be checked rather than assumed.
  if [ "$leg_status" = "ok" ] && [ -n "$disable_keys" ]; then
    local disable_err
    if ! disable_err="$(_check_expected_disables "$out_n" "$disable_keys" 2>&1)" \
        || ! disable_err="$(_check_expected_disables "$out_m" "$disable_keys" 2>&1)"; then
      leg_status="invalid"
      leg_reason="the forced-eager claim was not honoured: $disable_err"
    fi
  fi

  local wall_n="" wall_m=""
  if [ "$leg_status" = "ok" ]; then
    if ! wall_n="$(_tier_field "$out_n" train_run_wall_s)"; then
      leg_status="invalid"
      leg_reason="could not read tiers.finetune_run.train_run_wall_s from the N-step run"
    fi
  fi
  if [ "$leg_status" = "ok" ]; then
    if ! wall_m="$(_tier_field "$out_m" train_run_wall_s)"; then
      leg_status="invalid"
      leg_reason="could not read tiers.finetune_run.train_run_wall_s from the M-step run"
    fi
  fi

  # The DIRECT media front-end timer (contract P1-b(v)). Read "when
  # available", never required: the field is `null` until the `jammi-ai`
  # seam it needs lands, and a missing DIRECT measurement is a gap in the
  # attribution, not an invalid leg. It is NEVER back-filled from
  # `wall - busy` -- a difference is not a measurement.
  local front_n="" front_m=""
  if [ "$leg_status" = "ok" ]; then
    front_n="$(_tier_field "$out_n" media_front_end_wall_s 2>/dev/null)" || front_n=""
    front_m="$(_tier_field "$out_m" media_front_end_wall_s 2>/dev/null)" || front_m=""
  fi

  local steps_measured_n="" steps_measured_m=""
  if [ "$leg_status" = "ok" ]; then
    steps_measured_n="$(_tier_field "$out_n" steps_measured 2>/dev/null)" || steps_measured_n=""
    steps_measured_m="$(_tier_field "$out_m" steps_measured 2>/dev/null)" || steps_measured_m=""
  fi

  local census_ok="false"
  local census_json="$leg_dir/census.json"
  local census_stdout="$leg_dir/census.stdout"
  local census_stderr="$leg_dir/census.stderr"
  local census_exit=""
  if [ "$leg_status" = "ok" ]; then
    local -a census_cmd=(
      python3 "$DIR/kernel_census.py" "$sqlite_n" "$sqlite_m" "$STEPS_N" "$STEPS_M" "$census_json"
      --wall-a "$wall_n" --wall-b "$wall_m"
    )
    if [ -n "$steps_measured_n" ]; then census_cmd+=(--steps-measured-a "$steps_measured_n"); fi
    if [ -n "$steps_measured_m" ]; then census_cmd+=(--steps-measured-b "$steps_measured_m"); fi
    # `&&`/`||`, never a bare assignment: under `set -euo pipefail` a plain
    # command exiting nonzero (kernel_census.py legitimately REFUSING)
    # would abort the whole sweep before `census_exit` was ever set.
    run_cmd "${census_cmd[@]}" >"$census_stdout" 2>"$census_stderr" && census_exit=0 || census_exit=$?
    cat "$census_stdout" || true
    cat "$census_stderr" >&2 || true
    if [ "$census_exit" -eq 0 ]; then
      census_ok="true"
    else
      leg_status="invalid"
      local census_first_violation
      census_first_violation="$(grep -A1 -m1 '^::error::' "$census_stderr" | tail -n1)" ||
        census_first_violation=""
      if [ -n "$census_first_violation" ]; then
        leg_reason="kernel_census.py refused (exit $census_exit): $census_first_violation -- see $census_stderr"
      else
        leg_reason="kernel_census.py refused (exit $census_exit, leg INVALID -- see $census_stderr; no census.json written)"
      fi
    fi
  fi

  if ! _write_manifest \
      "$leg_id" "$SHA" "$(hostname)" "$NSYS_VERSION" "$tower" "$task" "$dtype" \
      "$target_modules" "$disable_keys" "$seq_len" "$BATCH" "$OBJECTIVE" "$LORA_INIT" \
      "$EVAL_CADENCE" "$STEPS_N" "$STEPS_M" "$steps_measured_n" "$steps_measured_m" \
      "$leg_status" "$leg_reason" "$census_ok" "$census_exit" "$dry_run_flag" \
      "$HELDOUT_ROWS" "$front_n" "$front_m" "$leg_dir/manifest.json"; then
    echo "::error::$leg_id: could not write $leg_dir/manifest.json -- this IS sweep-fatal (this leg's result cannot be recorded); aborting the sweep now rather than continuing silently unrecorded." >&2
    exit 1
  fi

  if [ "$leg_status" != "ok" ]; then
    echo "::warning::$leg_id: INVALID -- $leg_reason (recorded in $leg_dir/manifest.json; sweep continues)." >&2
  fi
  return 0
}

for spec in "${LEGS[@]}"; do
  run_leg "$spec"
done

echo "profile_421_legs: done -- artifacts under $OUT_DIR"
