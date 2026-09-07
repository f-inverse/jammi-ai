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
#   PROFILE_421_LEGS_DRY_RUN_EXTRA_REQUESTED_KEY
#                                    TEST-ONLY (never set by a real run):
#                                    the hermetic fake bench stub appends
#                                    this op key to `kernels_disabled_requested`
#                                    on top of the leg's genuine claim,
#                                    simulating an ambient JAMMI_KERNELS_DISABLE
#                                    contaminating a D leg (unit-467 finding
#                                    F1's D-leg half) so
#                                    `_check_expected_disables`'s equality
#                                    refusal can be exercised hermetically.
#   PROFILE_421_P2_BF16              "1" runs the BF16 PRE-FLIGHT MODE
#                                    (contract "## P2" / v2.3 §D4 item 4)
#                                    instead of the 12-leg sweep, and exits:
#                                    one untraced `finetune-run` per tower at
#                                    `--backbone-dtype bf16 --lora-init
#                                    gaussian`, 16 rows / batch 8 / 1 epoch
#                                    plus an 8-row held-out, into
#                                    `$OUT_DIR/p2-bf16/<tower>/`. The
#                                    ASSERTIONS on those reports live in
#                                    `profile_421_merge.py --p2-dir`.
#
# MERGE STEP: `ci/scripts/perf/profile_421_merge.py` reads this driver's
# `$OUT_DIR` and turns it into the per-(tower, dtype, leg) table the artifact
# producer embeds -- the positive-proof equation (`fused + eager == census x
# steps_measured`), the D-leg forced-eager proof, and the per-step
# wall/front/busy/residual decomposition. This driver deliberately does not
# judge its own output.
#
# Hermetic self-tests: `python3 ci/scripts/perf/test_profile_421_legs_dry_run.py`
# and `python3 ci/scripts/perf/test_profile_421_merge.py`.

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
# The BF16 pre-flight mode (contract "## P2", restated in emitted terms by
# v2.3 §D4 item 4). A MODE of this driver, not a separate script, so the P2
# invocation is pinned in exactly the same file the 12 legs are -- see the
# `p2_bf16_sweep` function below for what it runs and why.
PROFILE_421_P2_BF16="${PROFILE_421_P2_BF16:-0}"

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

# The P2 BF16 pre-flight's own pinned shape (contract §D4 item 4): 16 train
# rows at --batch 8 --epochs 1 = exactly 2 optimizer steps, which is what
# makes `train_probe_series` two entries long (index 0 = the untrained
# init probe, index 1 = after the single epoch) and therefore what makes
# the "the Gaussian adapter MOVED the loss at step 1" assertion expressible
# at all. `--lora-init gaussian` is the whole point of the pre-flight: under
# the wire default ZerosB, `dL/dA == 0` at step 1 by construction, so the
# two probes could not differ and a dtype bug would be indistinguishable
# from a correctly-frozen adapter.
P2_ROWS=16
P2_HELDOUT_ROWS=8
P2_BACKBONE_DTYPE=bf16
P2_LORA_INIT=gaussian

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
# The HTSAT D1 leg PRESUPPOSES P1-a (HTSAT's two GELU sites routed through
# the house `activations::gelu_erf(x, training)` seam rather than calling
# `Tensor::gelu_erf` directly), which landed on this branch. Were it ever
# reverted, that leg would refuse with "gelu_erf_fused ... never disabled a
# live dispatch this run" -- the CORRECT outcome (the leg is invalid, not a
# datum), recorded in its manifest while the other 11 legs continue.
D1_KEYS_CLIP="lora_linear_fused,layer_norm_fused"
D1_KEYS_HTSAT="lora_linear_fused,layer_norm_fused,gelu_erf_fused"
D2_KEYS="lora_linear_fused"

# --- ambient-var guard (unit-467 finding F1, driver half -- "pick one and
# say why: explicit-empty vs. preflight refusal"; this driver picks
# REFUSAL): `JAMMI_KERNELS_DISABLE` reaching THIS driver's own process
# environment at all would silently leak into every A leg's environment too
# (a child process inherits its parent's environment unless a var is
# explicitly cleared for it), turning the DECISION legs' own fused-vs-
# disabled claim into a lie -- an ambient value here IS what a hand-run D
# leg (or ANY prior export) left behind. `run_traced` scopes
# `JAMMI_KERNELS_DISABLE` to a SINGLE D-leg invocation via `env VAR=...
# cmd`, deliberately never touching this script's own environment, so this
# check running ONCE, here, before any leg runs, is sufficient: nothing
# below this line can make the var newly appear in this process's own
# environment over the sweep's lifetime. Refusing here (rather than
# stamping an explicit `JAMMI_KERNELS_DISABLE=` onto every A-leg
# invocation) keeps an A leg's command line identical to what it always
# was -- an operator reading `_print_cmd`'s trace sees NO disable-related
# token on an A leg either way, which is what
# `test_a_legs_carry_no_disable_env_and_make_no_claim` pins.
if [ -n "${JAMMI_KERNELS_DISABLE:-}" ]; then
  echo "::error::JAMMI_KERNELS_DISABLE is set in this driver's own environment ('$JAMMI_KERNELS_DISABLE') -- refusing before any leg runs. Each D leg scopes this var to its own single finetune-run invocation via 'env VAR=... cmd'; an ambient value here would leak into every A leg's environment too and silently contaminate the fused-vs-disabled decision legs (unit-467 finding F1). Unset it in the calling shell before running this driver." >&2
  exit 2
fi

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
# under DRY_RUN never executes. Used for invocations that need `$NSYS_BIN`/
# `$BENCH_BIN` or otherwise cannot run hermetically (`kernel_census.py`,
# which reads a real `nsys`-exported sqlite that only exists on a real
# run). Corpus generation uses `run_corpus_cmd` below instead -- see
# esc-088. `run_traced`'s own nsys/bench invocation does NOT go through
# either wrapper.
run_cmd() {
  _print_cmd "$@"
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# --- corpus-producer wrapper: unlike `run_cmd`, this ALWAYS executes, even
# under `PROFILE_421_LEGS_DRY_RUN=1` -- the three corpus producers
# (`gen_fixed_width_corpus.py` and the two media producers) are CPU-
# hermetic (no GPU, no network, no `$NSYS_BIN`/`$BENCH_BIN`) and cheap: each
# writes a FIXED family x instances-per-family media pool regardless of
# `--rows` (`preflight_probe` above already runs all three for real on
# every non-dry invocation, at an even smaller scale, for exactly this
# reason). Running them for real under DRY_RUN too is what makes
# `test_profile_421_legs_dry_run.py` exercise `provision_corpus`'s own
# nameref-out-parameter path with a REAL producer's real stdout, rather
# than a path that never ran hermetically at all (esc-088,
# `.jammi/escapes.jsonl`): under the OLD behaviour (touch empty placeholder
# files, never invoke the producer, under DRY_RUN) no producer ever wrote a
# byte of real stdout in any hermetic test, so a stdout-capture bug in
# `provision_corpus` was invisible to the whole suite and only surfaced on
# a real pod run. The child's own stdout is forwarded to THIS SCRIPT's
# stderr -- the same place `_print_cmd`'s trace line already goes, and for
# the same reason: never leave it on a channel a future capture point
# could pick up again.
run_corpus_cmd() {
  _print_cmd "$@"
  "$@" >&2
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

# --- checkpoint identity (unit-467 R3 pressure-test finding): refuse
# BEFORE any leg runs if `$MODEL_DIR_CLIP`/`$MODEL_DIR_CLAP` hold the WRONG
# checkpoint shape. `arch.rs`'s `Checkpoint::resolve` (`arch.rs:45,64`)
# PREFERS a `config.json`/`model.safetensors` pair over the `open_clip_*`
# siblings whenever both are present in the same directory, so a
# `$MODEL_DIR_CLIP` that also happens to carry a bare `config.json`/
# `model.safetensors` (a stray HF snapshot co-located in the checkpoint
# directory, or the wrong directory entirely) would silently mis-resolve to
# the WRONG architecture family rather than the OpenCLIP one every CLIP leg
# declares -- every downstream check in this driver (and in
# `profile_421_merge.py`) would still pass, because nothing here reads
# WHICH family actually got built, only whether the run and census agree
# with themselves. Symmetrically, `$MODEL_DIR_CLAP` (the HF
# `laion/clap-htsat-fused` checkpoint) MUST carry the standard HF triad --
# `config.json` + `model.safetensors` + `preprocessor_config.json` -- a
# directory missing any of the three is not the checkpoint shape the htsat
# legs declare, whatever else it contains.
#
# Runs UNCONDITIONALLY, dry run included, but ONLY when the named directory
# actually EXISTS on disk: a DRY_RUN caller that leaves
# `$MODEL_DIR_CLIP`/`$MODEL_DIR_CLAP` at their non-existent placeholder
# default (set just above) has nothing to check yet -- this is what lets
# the hermetic dry-run test exercise BOTH refusals against real temp dirs
# without requiring a full pod checkpoint set for every OTHER dry-run
# assertion in this suite.
_checkpoint_identity_probe() {
  local violations=()
  if [ -d "$MODEL_DIR_CLIP" ]; then
    if [ -f "$MODEL_DIR_CLIP/config.json" ] || [ -f "$MODEL_DIR_CLIP/model.safetensors" ]; then
      violations+=("MODEL_DIR_CLIP ($MODEL_DIR_CLIP) carries config.json and/or model.safetensors -- arch.rs's Checkpoint::resolve prefers those over the open_clip_config.json/open_clip_model.safetensors pair every CLIP leg declares, so this directory would silently resolve to the WRONG architecture family")
    fi
  fi
  if [ -d "$MODEL_DIR_CLAP" ]; then
    local f
    for f in config.json model.safetensors preprocessor_config.json; do
      if [ ! -f "$MODEL_DIR_CLAP/$f" ]; then
        violations+=("MODEL_DIR_CLAP ($MODEL_DIR_CLAP) is missing $f -- the HTSAT/CLAP checkpoint shape every htsat leg declares requires config.json + model.safetensors + preprocessor_config.json all present")
      fi
    done
  fi
  if [ "${#violations[@]}" -gt 0 ]; then
    echo "::error::_checkpoint_identity_probe: refusing before any leg runs -- checkpoint identity mismatch(es):" >&2
    local v
    for v in "${violations[@]}"; do
      echo "  - $v" >&2
    done
    exit 2
  fi
}

_checkpoint_identity_probe

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
  # In P2 mode no leg manifest is ever written, so a pre-existing one is not
  # a conflict; `p2_bf16_sweep` runs the same guard over its OWN outputs.
  if [ "$PROFILE_421_P2_BF16" = "1" ]; then break; fi
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
# empty) $5=lora_init $6=backbone_dtype $7=heldout_ids_path -- emits the
# golden-derived Report envelope on ITS OWN stdout
# (never writes a file directly), so the exec-wrapper's own "redirect this
# child's stdout to the report file" mechanism is what actually produces
# the report, for real. The overridden fields are exactly the ones this
# driver's own readers consume, so a reader bug is exercised here rather
# than papered over.
#
# esc-088 (`.jammi/escapes.jsonl`): $7 mirrors the REAL `finetune-run`'s own
# `--heldout-ids <HELDOUT_IDS>` refusal -- a non-empty, EXISTING path is
# required, exactly what clap's "a value is required ... but none was
# supplied" enforces on a real pod run. `provision_corpus`'s stdout-capture
# bug used to hand this driver an EMPTY `heldout_ids`, which every real leg
# then passed straight through to a real `finetune-run` and every real leg
# refused; the DRY_RUN path never caught it because nothing on this path
# ever inspected the value. This stub now refuses the same way, so a
# regression back to an empty/garbled `--heldout-ids` fails HERE, hermetically,
# rather than only on a GPU pod.
heldout_ids_path="${7:-}"
if [ -z "$heldout_ids_path" ] || [ ! -f "$heldout_ids_path" ]; then
  echo "error: a value is required for '--heldout-ids <HELDOUT_IDS>' but none was supplied (got '${heldout_ids_path}')" >&2
  exit 2
fi
python3 -c '
import copy, json, os, sys
steps_measured, golden_path, task, disable_csv, lora_init, dtype = (
    int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
)
golden = json.load(open(golden_path))
tier = copy.deepcopy(golden["tiers"]["finetune_run"])
# Proportional to steps_measured (never a flat constant): the M-step run
# always declares more steps than the N-step run, so wall_m > wall_n > 0
# holds across the fake pair and the wall-pair domain check in
# kernel_census.py is exercised as a real same-workload N<M pair would.
tier["train_run_wall_s"] = 0.01 * steps_measured
tier["steps_measured"] = steps_measured
# Mirror the invocation this stub stands in for, not the golden it was
# derived from: every leg (and the P2 pre-flight) runs `--epochs 1
# --grad-accum 1`, which is the ONLY convention under which
# `steps_measured` is the positive-proof equation`s `batches` term
# (`profile_421_merge.py` refuses any other by name). The committed golden
# was produced at `--epochs 2`, so leaving its value here would make the
# dry-run output describe a run this driver cannot issue.
tier["epochs"] = 1
tier["grad_accum"] = 1
tier["task"] = task
tier["lora_init"] = lora_init
tier["backbone_dtype"] = dtype
# Checkpoint identity (unit-467 finding R3), mirrored per TOWER rather than
# left at the golden`s one inherited constant: the two CLIP towers really do
# share ONE checkpoint directory in production (MODEL_DIR_CLIP) and HTSAT a
# DIFFERENT one (MODEL_DIR_CLAP), so this is the one split that makes
# profile_421_merge.py`s cross-tower identity check exercisable from this
# stub`s own output -- a stub that left every task at the golden`s single
# inherited value could never distinguish "every leg of a tower agrees" from
# "this stub never varies at all".
tier["checkpoint_weights_sha256"] = "c" * 64 if task != "audio_embedding" else "d" * 64
# The ZerosB-vs-Gaussian PROBE SPLIT, mirrored rather than faked flat: under
# `zeros_b` the B matrix is zero at step 0, so `dL/dA == 0` and the init
# probe and the post-epoch probe are the SAME number by construction; under
# `gaussian` they must differ (that IS the P2 pre-flight`s assertion). A
# stub that emitted a moving series on BOTH arms would make the P2 check
# pass vacuously, so this fake reproduces the real split.
_p0 = 0.6
tier["train_probe_series"] = [_p0, _p0 - 0.05] if lora_init == "gaussian" else [_p0, _p0]
expected = sorted(k for k in disable_csv.split(",") if k)
tier["kernels_disabled_expected"] = expected
# Test-only contamination lever (never set by the real leg sweep): a
# hermetic RED-side proof that this driver`s own `_check_expected_disables`
# refuses a D leg`s report whose `kernels_disabled_requested` is a
# SUPERSET of what it claimed -- an ambient env var reaching the process
# beyond the declared disable list, unit-467 finding F1`s D-leg half. Left
# unset, `requested` is exactly `expected`, the ONLY thing a genuine,
# uncontaminated leg`s report ever carries.
_extra_requested = os.environ.get("PROFILE_421_LEGS_DRY_RUN_EXTRA_REQUESTED_KEY", "")
requested = sorted(expected + [_extra_requested]) if _extra_requested else list(expected)
tier["kernels_disabled_requested"] = requested
tier["kernels_disabled_fired"] = expected
# The WITNESSED per-forward seam census the real binary derives from the
# encoder it built, and the dispatch counters DERIVED FROM IT so this
# stub`s output satisfies the same positive-proof equation
# `profile_421_merge.py` applies to a real leg (`fused + eager == census x
# steps_measured`). The site counts are stand-ins -- this stub builds no
# model and claims nothing about any checkpoint -- but the ONE
# structurally load-bearing split is mirrored exactly: the CLIP towers`
# MLP activation is `quick_gelu`, which has no fused seam, so their
# `gelu_seam_calls_per_forward` is 0 and their gelu counters must read
# 0/0, while an HTSAT leg carries a real per-forward count. A stub that
# emitted a non-zero gelu census on a CLIP leg would make that whole arm
# of the merger untestable from here.
census = {
    "lora_sites_wrapped": 48 if task != "audio_embedding" else 77,
    "layer_norms": 25 if task != "audio_embedding" else 30,
    "gelu_seam_calls_per_forward": 0 if task != "audio_embedding" else 9,
}
tier["fusible_site_census"] = census
for key, (fused_field, eager_field, census_field) in {
    "lora_linear_fused": (
        "lora_linear_fused_dispatches", "lora_linear_eager_dispatches", "lora_sites_wrapped",
    ),
    "layer_norm_fused": ("ln_fused_dispatches", "ln_eager_dispatches", "layer_norms"),
    "gelu_erf_fused": (
        "gelu_fused_dispatches", "gelu_eager_dispatches", "gelu_seam_calls_per_forward",
    ),
}.items():
    total = census[census_field] * steps_measured
    # A D leg is an EAGER twin: its disabled keys dispatched eager, never fused.
    if key in expected:
        tier[fused_field], tier[eager_field] = 0, total
    else:
        tier[fused_field], tier[eager_field] = total, 0
tier["lora_epilogue_fused_dispatches"] = 0
tier["lora_epilogue_eager_dispatches"] = 1
if task != "text_embedding":
    tier["train_media_sha256"] = "a" * 64
    tier["heldout_media_sha256"] = "b" * 64
    # The DIRECT front-end timer: a real producer reports it non-null on a
    # media leg and NULL on a text one (`Duration::ZERO` there is not a
    # measurement of anything -- see the field'"'"'s own doc), so the fake
    # mirrors that split and this driver'"'"'s reader is exercised on BOTH arms.
    # Proportional to steps, like the wall above, and strictly under it.
    tier["media_front_end_wall_s"] = 0.004 * steps_measured
else:
    tier["media_front_end_wall_s"] = None
report = {"tool": "dry-run", "profile_421_dry_run": True, "tiers": {"finetune_run": tier}}
json.dump(report, sys.stdout)
' "$1" "$2" "$3" "$4" "$5" "$6"
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
# the keys this leg claimed, and `kernels_disabled_requested` must equal
# them EXACTLY too -- not merely contain them. A SUBSET check here (claimed
# keys present in `requested`, extras allowed) would let a D leg whose
# `JAMMI_KERNELS_DISABLE` carries an EXTRA ambient key beyond what it
# claimed pass silently: that extra key force-eagers an op the leg assumed
# fused, inflating the D-leg wall and OVERSTATING the realized gain -- a
# distinct failure mode from the missing-key case below, and just as
# invalidating. The binary itself already refuses a leg that fails the
# equality `--expect-kernels-disabled` check (that is what the flag IS), so
# this is the driver's own independent record of the same fact -- and it
# catches the one case the binary cannot: a leg whose report came from a
# DIFFERENT invocation than the one this script believes it ran.
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
requested_sorted = sorted(requested)
if requested_sorted != claimed:
    missing = [k for k in claimed if k not in requested_sorted]
    extra = [k for k in requested_sorted if k not in claimed]
    print("::error::_check_expected_disables: claimed " + repr(claimed) +
          " does not exactly equal the process-resolved JAMMI_KERNELS_DISABLE " +
          repr(requested_sorted) + " (missing=" + repr(missing) + " extra=" + repr(extra) +
          "): " + path, file=sys.stderr)
    sys.exit(1)
' "$1" "$2"
}

# The A-leg witness (unit-467 finding F1): `finetune-run --arm fused` makes
# NO claim about `JAMMI_KERNELS_DISABLE` at all (an operator may legitimately
# run it with OTHER, unrelated op keys disabled -- see
# `FinetuneRunParams::expect_kernels_disabled`'s doc) -- the binary itself
# does not, and must not, refuse an unlabeled fused leg on this basis. The
# driver's own preflight ambient-var guard above already makes a
# contaminated A leg unreachable for a FRESH invocation of this script, but
# the REPORT is the independent witness of what the binary actually saw,
# regardless of how it got invoked -- a leg produced by an older build of
# this same script (one predating that guard), or a report copied in from
# somewhere else entirely, must still be caught here. Refuses if
# `kernels_disabled_requested` is non-empty on a leg this driver declared an
# A leg (no `JAMMI_KERNELS_DISABLE`, no `--expect-kernels-disabled` claim) --
# this driver and `profile_421_merge.py`'s own A-leg refusal are the ONLY
# witnesses that an A leg was genuinely unlabeled; nothing inside the binary
# checks this.
_check_no_ambient_disables() {
  python3 -c '
import json, sys
path = sys.argv[1]
d = json.load(open(path))
tier = d.get("tiers", {}).get("finetune_run") or {}
requested = tier.get("kernels_disabled_requested")
if requested is None:
    print("::error::_check_no_ambient_disables: the report carries no "
          "kernels_disabled_requested (a build predating issue #421 P1-b(i)): " + path,
          file=sys.stderr)
    sys.exit(1)
if requested:
    print("::error::_check_no_ambient_disables: this leg declared itself an A leg (no "
          "JAMMI_KERNELS_DISABLE, no --expect-kernels-disabled claim) but the report'"'"'s "
          "kernels_disabled_requested=" + repr(sorted(requested)) + " -- an ambient env var "
          "contaminated this run (unit-467 finding F1): " + path, file=sys.stderr)
    sys.exit(1)
' "$1"
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
      "$DRY_RUN_STUB_DIR/fake_bench.sh" "$steps_this_run" "$GOLDEN_FIXTURE" "$task" \
        "$disable_keys" "$LORA_INIT" "$dtype" "$heldout_ids"
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
# Sets the CALLER's own train_n/train_m/heldout_ids/heldout_jsonl variables
# via bash nameref out-parameters ($3-$6) -- NEVER this function's own
# stdout; returns nonzero (with a stderr message) on failure, which
# `run_leg` records as this leg's own INVALID reason.
#
# esc-088 (`.jammi/escapes.jsonl`): this function used to ECHO its 4-tuple
# ("train_n<TAB>train_m<TAB>heldout_ids<TAB>heldout_jsonl") on ITS OWN
# stdout, and `run_leg` captured the WHOLE function call with
# `corpus_line="$(provision_corpus ...)"`. But the three real producers
# (`gen_fixed_width_corpus.py`, the two media producers) ALSO print their
# own one-line summary to stdout, and (at the time) that summary was never
# redirected away -- `run_leg`'s command substitution therefore captured
# the producer's summary line as its FIRST line and the tuple as its
# SECOND, and `IFS=$'\t' read -r train_n train_m heldout_ids heldout_jsonl
# <<< "$corpus_line"` only ever sees the first line: `train_n` became the
# producer's prose, `heldout_ids` came out empty, and every real leg's
# `finetune-run` invocation refused with `--heldout-ids` missing. Nameref
# out-parameters carry no stdout at all, so there is no shared channel left
# for a producer's own report to land on -- structural, not a per-call
# redirect a future new producer could bypass by omission. See also
# `run_corpus_cmd` below, which is what actually runs these producers now
# (even under `PROFILE_421_LEGS_DRY_RUN=1`, unlike `run_cmd`) so this
# capture-free path is itself hermetically exercised.
#
# The N corpus is a PREFIX of the M corpus for text (the producer's own
# stream property) and a smaller row list over the SAME image/audio files
# for media -- either way the two runs differ ONLY in row count, which is
# what the N/M wall differencing requires. The HELD-OUT split is generated
# ONCE per leg and shared by both runs (it is not part of the differenced
# workload -- `--eval-cadence` is fixed, so its cost cancels).
#
# WHICH held-out split is shared is NOT a free choice for the TEXT tower
# (contract v2.3 §D4 item 3). `gen_fixed_width_corpus.py` draws
# `rows + heldout_rows` rows from ONE seeded stream and slices the held-out
# half OFF THE END: at `--rows R` the held-out rows are stream indices
# `[R, R + heldout_rows)`. Both text runs share `--seed 42` and differ only
# in `--rows`, so the N run's held-out rows (indices `[800, 808)`) are
# INSIDE the M run's train corpus (indices `[0, 4800)`) -- byte-identical
# texts under the same stream. Only the M corpus's own held-out slice
# (indices `[4800, 4808)`) sits past BOTH train corpora, so `$dir_m`'s
# held-out pair is what both runs are given. The two media producers are
# NOT affected: they reserve `--heldout-families` out of the family pool, so
# a held-out row's family never appears in EITHER train split regardless of
# row count, and each of those arms keeps using its own `$dir_n` split.
# `test_profile_421_legs_dry_run.py` drives the real (hermetic) text
# producer and asserts no held-out anchor text appears in either train file.
provision_corpus() {
  local tower="$1" leg_dir="$2"
  local -n _pc_train_n="$3" _pc_train_m="$4" _pc_heldout_ids="$5" _pc_heldout_jsonl="$6"
  local rows_n=$(( BATCH * STEPS_N ))
  local rows_m=$(( BATCH * STEPS_M ))
  local dir_n="$leg_dir/corpus_n" dir_m="$leg_dir/corpus_m"

  case "$tower" in
    clip-text)
      mkdir -p "$dir_n" "$dir_m"
      run_corpus_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_n" \
        --min-wordpieces "$CLIP_TEXT_SEQ" --seed "$SEED" --out "$dir_n/train.jsonl" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_corpus_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$rows_m" \
        --min-wordpieces "$CLIP_TEXT_SEQ" --seed "$SEED" --out "$dir_m/train.jsonl" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      # `$dir_m`, NOT `$dir_n` -- see this function's own doc: only the M
      # corpus's held-out slice is past BOTH train corpora's stream indices.
      _pc_train_n="$dir_n/train.jsonl"
      _pc_train_m="$dir_m/train.jsonl"
      _pc_heldout_ids="$dir_m/heldout_ids.txt"
      _pc_heldout_jsonl="$dir_m/heldout_triplets.jsonl"
      ;;
    clip-vision)
      run_corpus_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$rows_n" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$dir_n" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_corpus_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$rows_m" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$dir_m" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      _pc_train_n="$dir_n/triplets.jsonl"
      _pc_train_m="$dir_m/triplets.jsonl"
      _pc_heldout_ids="$dir_n/heldout_ids.txt"
      _pc_heldout_jsonl="$dir_n/heldout_triplets.jsonl"
      ;;
    htsat)
      run_corpus_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$rows_n" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$dir_n" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      run_corpus_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$rows_m" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$dir_m" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      _pc_train_n="$dir_n/triplets.jsonl"
      _pc_train_m="$dir_m/triplets.jsonl"
      _pc_heldout_ids="$dir_n/heldout_ids.txt"
      _pc_heldout_jsonl="$dir_n/heldout_triplets.jsonl"
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
  MANIFEST_MEDIA_FRONT_END_M="${26}" \
  MANIFEST_CHECKPOINT_SHA_N="${27}" MANIFEST_CHECKPOINT_SHA_M="${28}" \
  MANIFEST_CENSUS_N="${29}" MANIFEST_CENSUS_M="${30}" \
  MANIFEST_OUT="${31}" \
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
    # Checkpoint identity (unit-467 finding R3), read off EACH run`s own
    # report -- `null` when the run never reached the point of reporting it
    # (an INVALID leg`s runs may not have produced a report at all). Recorded
    # PER RUN, like `steps_measured`/`media_front_end_wall_s` above, rather
    # than collapsed to one value here: the within-leg N/M agreement and the
    # cross-leg (this tower`s OTHER legs) agreement are both
    # `profile_421_merge.py`s job, and both need the raw per-run values to
    # check, not a value this driver already decided was canonical.
    "checkpoint_weights_sha256": {
        "n": _json_or_none(os.environ["MANIFEST_CHECKPOINT_SHA_N"]),
        "m": _json_or_none(os.environ["MANIFEST_CHECKPOINT_SHA_M"]),
    },
    # The witnessed per-forward fusible-site census (issue #421 D4 item 1),
    # same per-run/null convention as the checkpoint sha above -- this is
    # what `profile_421_merge.py`s cross-tower identity check compares
    # alongside it.
    "fusible_site_census": {
        "n": _json_or_none(os.environ["MANIFEST_CENSUS_N"]),
        "m": _json_or_none(os.environ["MANIFEST_CENSUS_M"]),
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
        "$HELDOUT_ROWS" "" "" "" "" "" "" "$OUT_DIR/${leg_id}.manifest.json"; then
      echo "::error::$leg_id: could not write even the fallback manifest.json -- this IS sweep-fatal (results cannot be recorded); aborting." >&2
      exit 1
    fi
    return 0
  fi

  local leg_status="ok"
  local leg_reason=""
  local train_n="" train_m="" heldout_ids="" heldout_jsonl=""

  if ! provision_corpus "$tower" "$leg_dir" train_n train_m heldout_ids heldout_jsonl; then
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

  # A legs only, the mirror image (unit-467 finding F1): a POSITIVE claim
  # that nothing was disabled, witnessed off BOTH reports. The preflight
  # ambient-var guard near the top of this script already makes a
  # contaminated A leg unreachable for a fresh run of this driver -- this is
  # the belt-and-braces second read, off the artifact the run actually left
  # behind, exactly as the D-leg check above is a second read of ITS claim.
  if [ "$leg_status" = "ok" ] && [ -z "$disable_keys" ]; then
    local no_ambient_err
    if ! no_ambient_err="$(_check_no_ambient_disables "$out_n" 2>&1)" \
        || ! no_ambient_err="$(_check_no_ambient_disables "$out_m" 2>&1)"; then
      leg_status="invalid"
      leg_reason="an ambient JAMMI_KERNELS_DISABLE contaminated this A leg: $no_ambient_err"
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

  # Checkpoint identity (unit-467 finding R3), read off EACH run's own
  # report -- "when available", same posture as `front_n`/`front_m` above:
  # a leg already marked invalid for some other reason may not have a
  # readable report at all, and a manifest that cannot record this yet must
  # say so with `null` rather than fail the whole write. `_tier_field`
  # already emits the field as a JSON-encoded string/object, so these
  # variables carry EXACTLY what `_write_manifest`'s `_json_or_none` expects.
  local checkpoint_sha_n="" checkpoint_sha_m="" census_n="" census_m=""
  if [ "$leg_status" = "ok" ]; then
    checkpoint_sha_n="$(_tier_field "$out_n" checkpoint_weights_sha256 2>/dev/null)" || checkpoint_sha_n=""
    checkpoint_sha_m="$(_tier_field "$out_m" checkpoint_weights_sha256 2>/dev/null)" || checkpoint_sha_m=""
    census_n="$(_tier_field "$out_n" fusible_site_census 2>/dev/null)" || census_n=""
    census_m="$(_tier_field "$out_m" fusible_site_census 2>/dev/null)" || census_m=""
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
      "$HELDOUT_ROWS" "$front_n" "$front_m" \
      "$checkpoint_sha_n" "$checkpoint_sha_m" "$census_n" "$census_m" \
      "$leg_dir/manifest.json"; then
    echo "::error::$leg_id: could not write $leg_dir/manifest.json -- this IS sweep-fatal (this leg's result cannot be recorded); aborting the sweep now rather than continuing silently unrecorded." >&2
    exit 1
  fi

  if [ "$leg_status" != "ok" ]; then
    echo "::warning::$leg_id: INVALID -- $leg_reason (recorded in $leg_dir/manifest.json; sweep continues)." >&2
  fi
  return 0
}

# =====================================================================
# P2 -- the BF16 pre-flight (contract "## P2", restated in emitted terms by
# v2.3 §D4 item 4). Its own MODE of this driver (`PROFILE_421_P2_BF16=1`),
# never a separate script, so the pod session's P2 invocation is pinned in
# the SAME file the 12 legs are and cannot drift from them.
#
# Per tower, ONE untraced `finetune-run`:
#   --backbone-dtype bf16 --lora-init gaussian, 16 train rows at --batch 8
#   --epochs 1 (= 2 optimizer steps), plus an 8-row held-out split.
#
# UNTRACED on purpose: nsys buys nothing here. P2 asks a QUALITATIVE
# question ("does the bf16 backbone train at all, and does a non-degenerate
# adapter actually move the loss?"), not a timing one, and wrapping it in a
# profiler would make the pre-flight depend on the very tool whose absence
# it is meant to be able to report before the legs. The stdout capture still
# goes through the same `bash -c 'exec ... > "$0"'` redirect the traced legs
# use, so the report file is produced by the same mechanism.
#
# This driver only RUNS P2 and records what came back (the report + the exit
# status); the ASSERTIONS on it -- exit 0, `final_loss_diagnostic` finite,
# `train_probe_series[0] != train_probe_series[1]`, `lora_linear_fused` and
# `layer_norm_fused` fused counters > 0 -- live in
# `ci/scripts/perf/profile_421_merge.py`'s `--p2-dir` mode, one reader, one
# place, tested hermetically there. A driver that also judged its own output
# would be two implementations of the same rule.
p2_run_one() {
  local tower="$1" task="$2" model_dir="$3" p2_dir="$4" target_modules="$5"
  local corpus_dir="$p2_dir/corpus"
  local train_jsonl heldout_ids heldout_jsonl

  case "$tower" in
    clip-text)
      mkdir -p "$corpus_dir"
      run_cmd python3 "$DIR/gen_fixed_width_corpus.py" --rows "$P2_ROWS" \
        --min-wordpieces "$CLIP_TEXT_SEQ" --seed "$SEED" --out "$corpus_dir/train.jsonl" \
        --heldout-rows "$P2_HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      train_jsonl="$corpus_dir/train.jsonl"
      ;;
    clip-vision)
      run_cmd python3 "$DIR/gen_fixed_shape_image_corpus.py" --rows "$P2_ROWS" \
        --size "$IMAGE_SIZE" --seed "$SEED" --out-dir "$corpus_dir" \
        --families "$MEDIA_FAMILIES" --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$P2_HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      train_jsonl="$corpus_dir/triplets.jsonl"
      ;;
    htsat)
      run_cmd python3 "$DIR/gen_fixed_length_audio_corpus.py" --rows "$P2_ROWS" \
        --seconds "$AUDIO_SECONDS" --sample-rate "$AUDIO_SAMPLE_RATE" --seed "$SEED" \
        --out-dir "$corpus_dir" --families "$MEDIA_FAMILIES" \
        --heldout-families "$MEDIA_HELDOUT_FAMILIES" \
        --heldout-rows "$P2_HELDOUT_ROWS" --heldout-batch "$BATCH" || return 1
      train_jsonl="$corpus_dir/triplets.jsonl"
      ;;
    *)
      echo "::error::p2_run_one: unknown tower '$tower'" >&2
      return 1
      ;;
  esac
  heldout_ids="$corpus_dir/heldout_ids.txt"
  heldout_jsonl="$corpus_dir/heldout_triplets.jsonl"
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    mkdir -p "$corpus_dir"
    : > "$train_jsonl"; : > "$heldout_ids"; : > "$heldout_jsonl"
  fi

  # The full P2 command line, pinned here. Everything the legs pin is
  # repeated verbatim EXCEPT the two knobs P2 exists to vary
  # (`--backbone-dtype bf16`, `--lora-init gaussian`), so a P2 failure
  # cannot be blamed on some third difference from the legs.
  local -a cmd=(
    "$BENCH_BIN" finetune-run
    --model-dir "$model_dir" --arm fused --task "$task"
    --train-jsonl "$train_jsonl" --heldout-ids "$heldout_ids" --heldout-jsonl "$heldout_jsonl"
    --seed "$SEED" --epochs 1 --batch "$BATCH" --objective "$OBJECTIVE"
    --validation-fraction 0 --early-stopping-metric train_loss --grad-accum 1
    --early-stopping-patience 10000 --backbone-dtype "$P2_BACKBONE_DTYPE"
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05
    --lora-init "$P2_LORA_INIT"
    --target-modules "$target_modules"
    --max-seq-length "$CLIP_TEXT_SEQ"
    --eval-cadence "$EVAL_CADENCE"
    --work-dir "$p2_dir/work" --cuda 0
  )
  local -a exec_cmd=("${cmd[@]}")
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then
    exec_cmd=(
      "$DRY_RUN_STUB_DIR/fake_bench.sh" 2 "$GOLDEN_FIXTURE" "$task" "" \
        "$P2_LORA_INIT" "$P2_BACKBONE_DTYPE" "$heldout_ids"
    )
  fi

  mkdir -p "$p2_dir/work" || return 1
  _print_cmd "${cmd[@]}"
  local rc=0
  bash -c 'exec "$1" "${@:2}" > "$0"' "$p2_dir/run.json" "${exec_cmd[@]}" \
    2> "$p2_dir/run.stderr" || rc=$?
  P2_EXIT="$rc"
  if [ "$rc" -eq 0 ]; then
    local validate_err
    if ! validate_err="$(_validate_report_envelope "$p2_dir/run.json" 2>&1)"; then
      printf '%s\n' "$validate_err" | tee -a "$p2_dir/run.stderr" >&2
      return 1
    fi
  fi
  return "$rc"
}

# Writes one tower's P2 manifest. Like `_write_manifest`, a write failure
# here is FATAL: an unrecorded pre-flight result is worse than none.
_write_p2_manifest() {
  P2M_TOWER="$1" P2M_TASK="$2" P2M_GIT_SHA="$3" P2M_BOX="$4" P2M_DTYPE="$5" \
  P2M_LORA_INIT="$6" P2M_ROWS="$7" P2M_HELDOUT_ROWS="$8" P2M_BATCH="$9" \
  P2M_EPOCHS="${10}" P2M_OBJECTIVE="${11}" P2M_TARGET_MODULES="${12}" \
  P2M_EXIT="${13}" P2M_STATUS="${14}" P2M_REASON="${15}" P2M_DRY_RUN="${16}" \
  P2M_OUT="${17}" \
  python3 -c '
import json, os

manifest = {
    "mode": "p2-bf16-preflight",
    "driver": "ci/scripts/perf/profile_421_legs.sh",
    "tower": os.environ["P2M_TOWER"],
    "task": os.environ["P2M_TASK"],
    "git_sha": os.environ["P2M_GIT_SHA"],
    "box": os.environ["P2M_BOX"],
    "backbone_dtype": os.environ["P2M_DTYPE"],
    "lora_init": os.environ["P2M_LORA_INIT"],
    "rows": int(os.environ["P2M_ROWS"]),
    "heldout_rows": int(os.environ["P2M_HELDOUT_ROWS"]),
    "batch": int(os.environ["P2M_BATCH"]),
    "epochs": int(os.environ["P2M_EPOCHS"]),
    "objective": os.environ["P2M_OBJECTIVE"],
    "target_modules": os.environ["P2M_TARGET_MODULES"].split(","),
    # The RUN`s own exit status, recorded as an integer -- `profile_421_merge.py`
    # asserts it is 0 rather than inferring success from the report existing.
    "exit": int(os.environ["P2M_EXIT"]),
    "status": os.environ["P2M_STATUS"],
    "reason": os.environ["P2M_REASON"],
    "dry_run": os.environ["P2M_DRY_RUN"] == "true",
}
json.dump(manifest, open(os.environ["P2M_OUT"], "w"), indent=1)
'
}

p2_bf16_sweep() {
  local dry_run_flag="false"
  if [ "$PROFILE_421_LEGS_DRY_RUN" = "1" ]; then dry_run_flag="true"; fi
  local root="$OUT_DIR/p2-bf16"

  local tower
  local conflicts=()
  for tower in clip-text clip-vision htsat; do
    if [ -f "$root/$tower/manifest.json" ]; then conflicts+=("$tower"); fi
  done
  if [ "${#conflicts[@]}" -gt 0 ]; then
    echo "::error::OUT_DIR ($OUT_DIR) already has a P2 manifest for tower(s): ${conflicts[*]} -- refusing to silently overwrite; use a fresh OUT_DIR." >&2
    exit 2
  fi

  for tower in clip-text clip-vision htsat; do
    local task model_dir target_modules
    case "$tower" in
      clip-text)   task=text_embedding;  model_dir="$MODEL_DIR_CLIP"; target_modules="$CLIP_FULL" ;;
      clip-vision) task=image_embedding; model_dir="$MODEL_DIR_CLIP"; target_modules="$CLIP_FULL" ;;
      htsat)       task=audio_embedding; model_dir="$MODEL_DIR_CLAP"; target_modules="$CLAP_FULL" ;;
    esac
    echo "=== P2 bf16 pre-flight: tower=$tower task=$task dtype=$P2_BACKBONE_DTYPE lora_init=$P2_LORA_INIT rows=$P2_ROWS batch=$BATCH epochs=1 ==="

    local p2_dir="$root/$tower"
    local status="ok" reason=""
    P2_EXIT=1
    if ! mkdir -p "$p2_dir"; then
      echo "::error::P2 $tower: could not create $p2_dir -- aborting (a pre-flight result that cannot be recorded is not a pre-flight)." >&2
      exit 1
    fi
    if ! p2_run_one "$tower" "$task" "$model_dir" "$p2_dir" "$target_modules"; then
      status="invalid"
      reason="the P2 bf16 pre-flight run failed (exit $P2_EXIT, or its report failed envelope validation) -- see $p2_dir/run.stderr"
    fi

    if ! _write_p2_manifest "$tower" "$task" "$SHA" "$(hostname)" "$P2_BACKBONE_DTYPE" \
        "$P2_LORA_INIT" "$P2_ROWS" "$P2_HELDOUT_ROWS" "$BATCH" 1 "$OBJECTIVE" \
        "$target_modules" "$P2_EXIT" "$status" "$reason" "$dry_run_flag" \
        "$p2_dir/manifest.json"; then
      echo "::error::P2 $tower: could not write $p2_dir/manifest.json -- aborting." >&2
      exit 1
    fi
    if [ "$status" != "ok" ]; then
      echo "::warning::P2 $tower: INVALID -- $reason (recorded in $p2_dir/manifest.json; the sweep continues)." >&2
    fi
  done
  echo "profile_421_legs: P2 bf16 pre-flight done -- artifacts under $root"
  echo "::notice::assert them with: python3 ci/scripts/perf/profile_421_merge.py --p2-dir $root --out $root/p2_verdicts.json"
}

if [ "$PROFILE_421_P2_BF16" = "1" ]; then
  p2_bf16_sweep
  exit 0
fi

for spec in "${LEGS[@]}"; do
  run_leg "$spec"
done

echo "profile_421_legs: done -- artifacts under $OUT_DIR"
