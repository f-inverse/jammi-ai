#!/usr/bin/env bash
# The how-well producer: drives `jammi-bench finetune-run` over the
# committed `cookbook/fixtures/finetune_heldout/` held-out fixture, one
# leg per (seed, arm, repeat) — `{fused, alloff}` arms, `{r1, r2}` same-seed
# repeats — against the SAME committed fixture and objective every leg of a
# run shares. With FINETUNE_RUN_AB_TORCH=1 a third arm, `torch`, runs
# `crates/jammi-bench/reference/torch_finetune_run.py` — the same run in
# PyTorch + PEFT — beside them (see "THE TORCH ARM" below).
#
# NOT `stacked_sweep.sh`-shaped for its measured legs: no cookbook book
# stack, no server. Every input a MEASURED leg reads is a committed repo
# path (the fixture under `cookbook/fixtures/finetune_heldout/`, a local
# `--model-dir` checkpoint the operator already has on-box); no leg itself
# builds the cookbook corpus, starts a `jammi-server`, or touches the
# network. The ONE exception is the PRE-RUN provisioning step below,
# which runs strictly BEFORE any measured
# leg and is never counted as one: it may invoke the book-side
# `cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs`
# (network-backed, checksum-gated) to (re)populate `train_pairs.jsonl`, then
# ALWAYS byte-verifies it against the committed `train_ids_sha256.json`
# before letting any leg proceed.
#
# HELD-OUT FIXTURE LAYOUT (cookbook/fixtures/finetune_heldout/):
#   heldout_ids.txt      the committed held-out id list -- what
#                         `heldout_ids_sha256` hashes.
#   heldout_pairs.jsonl  the FULL held-out pair text (committed).
#   train_ids_sha256.json ids + a per-pair SHA-256 for the 1372 TRAIN-side
#                         pairs -- deliberately NOT full text (repo-size
#                         discipline, that directory's own README.md "Why
#                         train text isn't committed" section). This means
#                         `--train-jsonl` (a required `jammi-bench
#                         finetune-run` flag) has no committed source of its
#                         own text in this checkout.
#
# PRE-RUN PROVISIONING: before any measured
# leg, if `$TRAIN_JSONL` (default `$REPO_ROOT/cookbook/fixtures/
# finetune_heldout/train_pairs.jsonl`, gitignored -- never committed) is
# absent, this script invokes the book-side producer's own
# `--emit-train-pairs` mode (network-backed, checksum-gated; reuses the
# exact `mine_pairs()`/`_text()` code path `--check` already re-derives
# against) to write it. Then -- REGARDLESS of whether the file was just
# emitted or was already present on this pod from a prior run -- this
# script ALWAYS byte-verifies every pair against the committed
# `train_ids_sha256.json` via the standalone
# `ci/scripts/perf/verify_train_pairs.py` (sha256 per pair id, exact count
# 1372, no extras/duplicates), refusing loudly with the first mismatching
# id on any divergence, before a single leg runs. A pre-existing
# `train_pairs.jsonl` is never trusted on name alone: a stale or
# hand-edited file left over from an earlier checkout fails this exactly
# like a corrupted fresh fetch would.
#
# Batch size: 32 (`cookbook/fixtures/finetune_heldout/README.md` -- the
# chapter-config value every real `db.fine_tune(...)` call over this exact
# pair set already uses; 128 held-out pairs is a multiple of both 32 and the engine's own
# unset-default 8, so this pick does not change the fixture's own held-out
# count, only which "N held-out = k batches" framing is reported).
#
# Objective: MNRL by default -- override with FINETUNE_RUN_AB_OBJECTIVE=
# triplet. The objective is chosen by a dynamic-range probe outside this
# script; this script just runs whichever one it is told, over both arms, at
# every pre-registered seed.
#
# Seeds: the pre-registered 12-seed gate set (N=12 seeds x 2 arms), 1..12
# by default -- override with FINETUNE_RUN_AB_SEEDS (a
# comma-separated list, no spaces).
#
# LEG ORDER, per seed: fused r1, alloff r1, [torch r1, torch-natural r1,
# torch-natural r2, torch r2,] alloff r2, fused r2 — each arm's two repeats sit
# symmetrically about the middle of the seed's block (A, B, [T, N, N, T,] B, A;
# never A, A, B, B), so a first-order
# clock/thermal drift across the block shifts every arm's r1/r2 mean by the
# same amount instead of landing on whichever arm ran last. Same rationale
# as `finetune_ab.sh`'s "ORDER-BALANCED BAR LEGS". FINETUNE_RUN_AB_ARMS
# selects which of the run's arms have their legs run, in that same order
# (default: all of them); the legs of an unselected arm are neither run nor
# filed, so one stack's legs can re-run into an OUT_DIR whose other legs
# stand.
#
# THE TORCH ARM (FINETUNE_RUN_AB_TORCH=1). A torch leg is PAIRED with the
# jammi legs of its seed, not merely run next to them: every jammi leg
# writes its untrained adapter into its work dir
# (`initial_adapter.safetensors`, recorded as `initial_adapter_sha256`), and
# the torch leg loads the seed's first one (`--lora-init zeros_b
# --initial-adapter ...`), so both stacks start from byte-identical LoRA
# tensors. When FINETUNE_RUN_AB_ARMS leaves the `fused` arm out, that
# adapter is the one an earlier run of this OUT_DIR wrote, and the arm
# REFUSES, before any leg, naming every seed whose adapter is missing.
# That leaves LoRA dropout as the only
# randomness the two stacks cannot share, so the arm REFUSES, before any
# leg, unless FINETUNE_RUN_AB_LORA_DROPOUT is 0. Both producers take the
# SAME flags by the same names, built once (`run_leg`'s `shared`), so the two
# command lines cannot drift apart. The torch arm is TWO arms, the twin's two
# widths: `torch` runs `--width bucketed` (jammi's bucket ladder — the
# semantic twin, the leg that pairs with jammi on outcome) and
# `torch-natural` runs `--width natural` (pad to the batch's longest row —
# what a PyTorch user does, and so the practical bar for speed and space).
# The torch venv is the one `torch_venv.py` resolves (TORCH_VENV, default
# "<repo>/.venv-torch-ref"); it is probed before any leg and never
# provisioned here. Every leg is filed under the ladder's rung names —
# `resident` (fused), `resident-reference` (the flash cascade and fused AdamW
# off), `torch` — as `<rung>__seed<N>__<take>.json`, and `jammi-bench ladder
# train-run` judges them; a `torch-natural` leg is filed under `raw/natural/`
# as a reader's diagnostic (its token batches are another computation and
# never pair with the jammi legs on identity).
#
# Env vars:
#   MODEL_DIR                 checkpoint dir (config.json + model.safetensors
#                              + tokenizer.json). Required unless
#                              FINETUNE_RUN_AB_DRY_RUN=1.
#   FINETUNE_RUN_AB_SEEDS      comma-separated seed list (default: 1..12,
#                              the pre-registered gate set).
#   FINETUNE_RUN_AB_OBJECTIVE  "mnrl" or "triplet" (default: mnrl).
#   FINETUNE_RUN_AB_EPOCHS     --epochs passthrough (default: unset, so
#                              each producer's own default, the tier's
#                              protocol of 4, is used).
#   FINETUNE_RUN_AB_BATCH      batch size (default: 32 -- see "Batch size"
#                              above).
#   FINETUNE_RUN_AB_LR         --lr passthrough for every leg (default:
#                              unset, so each producer's own default, the
#                              tier's protocol of 5e-5, is used).
#   FINETUNE_RUN_AB_TARGET_MODULES
#                              --target-modules passthrough: the LoRA sites
#                              every leg adapts AND the sites the reference
#                              arm is derived on (`kernel-arm`'s census
#                              trains one step on them). Default: unset, so
#                              both read `finetune_run::DEFAULT_TARGET_MODULES`
#                              -- one constant, never two spellings here.
#   FINETUNE_RUN_AB_LR0_SEEDS  comma-separated seed list for the lr=0 RED
#                              control (an lr=0 arm over >= 2 seeds must fail
#                              learning-happened); default
#                              empty = skipped). Each seed here runs BOTH
#                              arms with --zero-lr-control, filed as the
#                              `lr0` take -- a control leg, never a measured
#                              repeat: the ladder checks each one ran at
#                              lr=0 and FAILS learning-happened, and never
#                              counts it into the paired statistic.
#   FINETUNE_RUN_AB_ALLOW_NO_LR0
#                              Default "0":
#                              when FINETUNE_RUN_AB_LR0_SEEDS is empty, the
#                              ladder REFUSES the edge (INVALID) -- the
#                              pre-registered lr=0 control is not silently
#                              optional. Set to "1" to pass the ladder
#                              `--waive-control`, a deliberate, visible
#                              opt-out recorded in the verdict.
#   FINETUNE_RUN_AB_MUTANT_LEGS
#                              OPTIONAL,
#                              ';'-separated list of
#                              'DOSE_LABEL:PATCH_SHA256:SEED1,SEED2,...'
#                              specs, forwarded verbatim as one
#                              '--mutant SPEC' per entry to the ladder.
#                              PURE pass-through: this script never runs a
#                              mutant leg itself (docs/plans/63-how-well/
#                              mutants/README.md's own scratch-worktree
#                              on-pod procedure does that, against a
#                              patched jammi-kernels build); this variable
#                              only tells the merge step where to find
#                              already-produced 'mutant-<dose_label>'-tagged
#                              legs under THIS run's own $RAW_DIR. Default
#                              empty = no dose ladder in this merge.
#   FINETUNE_RUN_AB_BACKBONE_DTYPE
#                              --backbone-dtype passthrough for EVERY leg
#                              `run_leg` runs -- both A/B arms, every seed,
#                              AND the lr=0 RED control below (default
#                              "bf16"). `backbone_dtype` is IDENTITY
#                              FIELD #10 on `FINETUNE_RUN_IDENTITY_FIELDS`
#                              (identity_fields.py) -- cross-arm AND
#                              cross-seed homogeneity requires every leg to
#                              report the SAME value, so this is read ONCE
#                              here and forwarded unconditionally from the
#                              one `run_leg` both loops (main sweep, lr=0
#                              control) share, never overridden per-arm/
#                              per-loop. `main.rs`'s own CLI accepts "f32",
#                              "f16", or "bf16" (`FinetuneRunArgs::
#                              backbone_dtype`'s match arms) -- this script
#                              does not itself re-validate the value, it
#                              relies on the binary's own refusal of an
#                              unrecognized spelling.
#   FINETUNE_RUN_AB_LORA_DROPOUT
#                              --lora-dropout passthrough for EVERY leg
#                              (default: unset, so the CLI's own default
#                              (0.05) is used). `lora_dropout` is an identity
#                              field, so it is read once and forwarded from
#                              the one `run_leg` every loop shares. Must be 0
#                              when the torch arm is on (see "THE TORCH ARM").
#   FINETUNE_RUN_AB_MAX_SEQ_LENGTH
#                              --max-seq-length passthrough for EVERY leg of
#                              every arm, control legs included (default:
#                              unset, so each producer's own default is used
#                              -- the engine's, 512, on both).
#                              `max_seq_length` is an identity field, so it
#                              is read once and forwarded from the one
#                              `run_leg` every loop shares.
#   FINETUNE_RUN_AB_TORCH=1    also run the `torch` arm (default: 0).
#   FINETUNE_RUN_AB_ARMS       comma-separated subset of the run's arms
#                              (`fused`, `alloff`, and with the torch arm
#                              on, `torch`, `torch-natural`) whose legs run
#                              (default: all of them). An arm the run does
#                              not have is refused; leaving `fused` out
#                              while a torch arm is in requires each seed's
#                              initial adapter in OUT_DIR already (see "THE
#                              TORCH ARM").
#   FINETUNE_RUN_AB_TORCH_ATTN the torch arm's `--attn` (default: sdpa —
#                              torch's best case; `eager` is the semantic twin
#                              of jammi's `alloff` attention composition).
#   TORCH_VENV                 the torch venv (default: torch_venv.py's,
#                              "<repo>/.venv-torch-ref"). Read only when the
#                              torch arm is on.
#   FINETUNE_RUN_AB_CUDA       CUDA ordinal (default: 0). Unset
#                              FINETUNE_RUN_AB_CPU=1 to omit --cuda entirely
#                              (the CPU-hermetic smoke path finetune-run's
#                              own CLI doc names) -- never both.
#   TRAIN_JSONL / HELDOUT_IDS / HELDOUT_JSONL
#                              override the committed-fixture paths (see
#                              "HELD-OUT FIXTURE LAYOUT" / "PRE-RUN
#                              PROVISIONING" above -- TRAIN_JSONL's default
#                              is auto-provisioned + byte-verified before any
#                              leg runs, never committed itself).
#   FINETUNE_RUN_AB_OUT_DIR    where the raw legs + merged report land
#                              (default "<repo>/.finetune-run-ab-report/
#                              <UTC timestamp>").
#   FINETUNE_RUN_AB_PROVISION_PYTHON
#                              the python interpreter invoked for the ONE
#                              provisioning step above (`derive_heldout_
#                              fixture.py --emit-train-pairs`) -- default
#                              "python3" (a bare checkout's system
#                              interpreter, which is enough when
#                              `train_pairs.jsonl` is already pre-staged, so
#                              this step never actually runs). A pod driver
#                              (e.g. `ci/scripts/runpod_gpu_howwell.sh`) that
#                              provisions a dedicated venv for
#                              `jammi_cookbook`/numpy/pyarrow/requests (a
#                              bare pod has no pip on PATH and this script's
#                              own producer binary
#                              build/run never needs any of those packages)
#                              points this at that venv's own interpreter
#                              instead; every OTHER step in this script
#                              (verification, the cargo build, every measured
#                              leg) stays on plain "python3"/the system
#                              toolchain -- MEASURED legs are deliberately
#                              venv-free, only this one pre-run provisioning
#                              call is not.
#   FINETUNE_RUN_AB_DRY_RUN=1  print every command this script would run
#                              instead of executing it, and write a
#                              `{"tool":"dry-run",...}` stub per leg so the
#                              merge stage still runs end-to-end against
#                              real (if fabricated-empty) files. Never
#                              mutates the checkout, never touches the
#                              network, never claims a real number -- same
#                              contract `finetune_ab.sh`/`encode_ab.sh`'s
#                              own `*_DRY_RUN` knobs already carry.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

FINETUNE_RUN_AB_DRY_RUN="${FINETUNE_RUN_AB_DRY_RUN:-0}"
FINETUNE_RUN_AB_SEEDS="${FINETUNE_RUN_AB_SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12}"
FINETUNE_RUN_AB_OBJECTIVE="${FINETUNE_RUN_AB_OBJECTIVE:-mnrl}"
case "$FINETUNE_RUN_AB_OBJECTIVE" in
  mnrl|triplet) ;;
  *)
    echo "::error::FINETUNE_RUN_AB_OBJECTIVE must be 'mnrl' or 'triplet', got '${FINETUNE_RUN_AB_OBJECTIVE}'." >&2
    exit 2
    ;;
esac
# --epochs / --lr passthrough. Unset means "omit the flag": both producers
# then run the tier's own protocol (`finetune_run::DEFAULT_LEARNING_RATE`
# 5e-5 over `DEFAULT_EPOCHS` 4, evaluated every epoch -- that constant's
# doc says why it is not the engine's 2e-4 over 3), read from one source,
# never a value fabricated here.
FINETUNE_RUN_AB_EPOCHS="${FINETUNE_RUN_AB_EPOCHS:-}"
FINETUNE_RUN_AB_BATCH="${FINETUNE_RUN_AB_BATCH:-32}"
FINETUNE_RUN_AB_LR="${FINETUNE_RUN_AB_LR:-}"
# --target-modules passthrough, to the derive and to every leg alike (unset
# = omit on both, so both take the one constant).
FINETUNE_RUN_AB_TARGET_MODULES="${FINETUNE_RUN_AB_TARGET_MODULES:-}"
# lr=0 RED control seeds -- comma-separated,
# default empty (skipped). NEVER added to FINETUNE_RUN_AB_SEEDS/the main
# sweep loop below; run through their own dedicated loop as the `lr0` take.
FINETUNE_RUN_AB_LR0_SEEDS="${FINETUNE_RUN_AB_LR0_SEEDS:-}"
# --backbone-dtype passthrough for EVERY leg (see env-var doc above).
FINETUNE_RUN_AB_BACKBONE_DTYPE="${FINETUNE_RUN_AB_BACKBONE_DTYPE:-bf16}"
# --lora-dropout passthrough. Unset means "omit the flag", i.e. the CLI's own
# default -- never a second copy of that default here.
FINETUNE_RUN_AB_LORA_DROPOUT="${FINETUNE_RUN_AB_LORA_DROPOUT:-}"
FINETUNE_RUN_AB_MAX_SEQ_LENGTH="${FINETUNE_RUN_AB_MAX_SEQ_LENGTH:-}"
FINETUNE_RUN_AB_TORCH="${FINETUNE_RUN_AB_TORCH:-0}"
FINETUNE_RUN_AB_TORCH_ATTN="${FINETUNE_RUN_AB_TORCH_ATTN:-sdpa}"
FINETUNE_RUN_AB_CUDA="${FINETUNE_RUN_AB_CUDA:-0}"
FINETUNE_RUN_AB_CPU="${FINETUNE_RUN_AB_CPU:-0}"

# The torch arm's premises, refused BEFORE any leg runs -- see "THE TORCH ARM".
TORCH_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_finetune_run.py"
if [ "$FINETUNE_RUN_AB_TORCH" = "1" ]; then
  # Numeric, so 0, 0.0 and 0.00 all read as "no dropout".
  if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) == 0.0 else 1)' "${FINETUNE_RUN_AB_LORA_DROPOUT:-unset}" 2>/dev/null; then
    echo "::error::FINETUNE_RUN_AB_TORCH=1 requires FINETUNE_RUN_AB_LORA_DROPOUT=0 (got '${FINETUNE_RUN_AB_LORA_DROPOUT:-<unset: the CLI default, 0.05>}') -- a torch leg is paired with the jammi legs of its seed from a shared initial adapter, and a LoRA dropout mask is the one draw the two stacks cannot share." >&2
    exit 2
  fi
  if [ "$FINETUNE_RUN_AB_OBJECTIVE" != "mnrl" ]; then
    echo "::error::FINETUNE_RUN_AB_TORCH=1 requires FINETUNE_RUN_AB_OBJECTIVE=mnrl (got '${FINETUNE_RUN_AB_OBJECTIVE}') -- torch_finetune_run.py twins the MNRL objective only." >&2
    exit 2
  fi
  # The torch venv and its default are resolved in one place, torch_venv.py.
  TORCH_VENV="$(python3 "$DIR/torch_venv.py" --path)"
  TORCH_PY="$TORCH_VENV/bin/python3"
  if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
    python3 "$DIR/torch_venv.py" \
      || { echo "::error::FINETUNE_RUN_AB_TORCH=1 but the torch venv is not usable (see above) -- refusing before any leg runs." >&2; exit 1; }
  fi
fi
# The run's arms, in leg order, and the selected subset -- see "LEG ORDER".
RUN_ARMS=(fused alloff)
if [ "$FINETUNE_RUN_AB_TORCH" = "1" ]; then
  RUN_ARMS+=(torch torch-natural)
fi
FINETUNE_RUN_AB_ARMS="${FINETUNE_RUN_AB_ARMS:-$(IFS=','; echo "${RUN_ARMS[*]}")}"
IFS=',' read -r -a SELECTED_ARMS <<< "$FINETUNE_RUN_AB_ARMS"
for arm in "${SELECTED_ARMS[@]}"; do
  case " ${RUN_ARMS[*]} " in *" $arm "*) continue ;; esac
  case "$arm" in
    torch|torch-natural)
      echo "::error::FINETUNE_RUN_AB_ARMS names '$arm', which requires FINETUNE_RUN_AB_TORCH=1." >&2
      ;;
    *)
      echo "::error::FINETUNE_RUN_AB_ARMS names '$arm'; this run's arms are: ${RUN_ARMS[*]}." >&2
      ;;
  esac
  exit 2
done
arm_selected() {
  case " ${SELECTED_ARMS[*]} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}
# Interpreter for the one provisioning step -- see the env-var doc above.
FINETUNE_RUN_AB_PROVISION_PYTHON="${FINETUNE_RUN_AB_PROVISION_PYTHON:-python3}"

FIXTURE_DIR="$REPO_ROOT/cookbook/fixtures/finetune_heldout"
TRAIN_JSONL="${TRAIN_JSONL:-$FIXTURE_DIR/train_pairs.jsonl}"
HELDOUT_IDS="${HELDOUT_IDS:-$FIXTURE_DIR/heldout_ids.txt}"
HELDOUT_JSONL="${HELDOUT_JSONL:-$FIXTURE_DIR/heldout_pairs.jsonl}"

MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ]; then
  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    MODEL_DIR="/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER"
    echo "::warning::FINETUNE_RUN_AB_DRY_RUN=1 and MODEL_DIR unset — printed commands use a placeholder path; nothing is read from it."
  else
    echo "::error::MODEL_DIR must name a checkpoint directory (config.json + model.safetensors + tokenizer.json)." >&2
    exit 2
  fi
fi

# Refuse loudly, before any leg runs, if the fixture's real held-out files
# are absent -- a real run over a missing/renamed fixture must not silently
# produce a stub-shaped FAIL row indistinguishable from a real training
# failure.
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  for f in "$HELDOUT_IDS" "$HELDOUT_JSONL"; do
    if [ ! -f "$f" ]; then
      echo "::error::committed fixture file not found: $f (cookbook/fixtures/finetune_heldout/) — refusing before any leg runs." >&2
      exit 1
    fi
  done

  # --- PRE-RUN provisioning -- see module
  # doc "PRE-RUN PROVISIONING" above. Outside every measured leg: this runs
  # once, before the sweep loop, never inside run_leg. Emit is SKIPPED
  # whenever `$TRAIN_JSONL` is already present (an operator/pod driver may
  # pre-stage it) -- byte-verification below still ALWAYS runs regardless.
  #
  # Invoked from `cookbook/book` as cwd, per that
  # directory's own fixture README ("cd cookbook/book && python scripts/
  # derive_heldout_fixture.py ...") -- `derive_heldout_fixture.py` itself
  # resolves every path it reads/writes off `__file__`, never cwd, so this
  # is the DOCUMENTED invocation convention, not a functional requirement of
  # that script; `$FINETUNE_RUN_AB_PROVISION_PYTHON` (default "python3") is
  # this call's own interpreter knob -- a bare checkout's system Python
  # cannot `import jammi_cookbook`/numpy, so a pod driver that provisions a
  # dedicated venv for this ONE step points this env var at that venv's
  # interpreter instead (see the env-var's own doc above).
  if [ ! -f "$TRAIN_JSONL" ]; then
    echo "::notice::$TRAIN_JSONL not found -- provisioning via 'derive_heldout_fixture.py --emit-train-pairs' (network-backed, checksum-gated fetch of train text; outside measured legs)."
    (cd "$REPO_ROOT/cookbook/book" && "$FINETUNE_RUN_AB_PROVISION_PYTHON" scripts/derive_heldout_fixture.py --emit-train-pairs) \
      || { echo "::error::train-pairs provisioning failed (cookbook/book/scripts/derive_heldout_fixture.py --emit-train-pairs, invoked from cookbook/book via \$FINETUNE_RUN_AB_PROVISION_PYTHON='$FINETUNE_RUN_AB_PROVISION_PYTHON') — refusing before any leg runs." >&2; exit 1; }
  else
    echo "::notice::$TRAIN_JSONL already present -- skipping the emit step (pre-staged); byte-verification below still always runs."
  fi
  # ALWAYS byte-verify -- whether train_pairs.jsonl was just emitted above or
  # was already present on this pod from a prior run. A stale/hand-edited
  # file must fail exactly like a corrupted fresh fetch would; this is the
  # ONE reviewable unit both this producer and any other caller share
  # (ci/scripts/perf/verify_train_pairs.py), never a second hand-rolled
  # comparator.
  python3 "$DIR/verify_train_pairs.py" --pairs "$TRAIN_JSONL" \
    || { echo "::error::$TRAIN_JSONL failed byte-verification against cookbook/fixtures/finetune_heldout/train_ids_sha256.json — refusing before any leg runs." >&2; exit 1; }
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${FINETUNE_RUN_AB_OUT_DIR:-$REPO_ROOT/.finetune-run-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

leg_work_dir() {
  echo "$OUT_DIR/work/seed${1}__${2}__${3}"
}

IFS=',' read -r -a SEEDS <<< "$FINETUNE_RUN_AB_SEEDS"

# A torch leg's premise when the `fused` arm is not selected: the adapter the
# seed's first jammi leg wrote into this OUT_DIR on an earlier run. Refused
# before any leg, every missing seed named. A dry run reads nothing.
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ] && ! arm_selected fused \
  && { arm_selected torch || arm_selected torch-natural; }; then
  missing=()
  for seed in "${SEEDS[@]}"; do
    f="$(leg_work_dir "$seed" fused r1)/initial_adapter.safetensors"
    [ -f "$f" ] || missing+=("$f")
  done
  if [ -n "$FINETUNE_RUN_AB_LR0_SEEDS" ]; then
    IFS=',' read -r -a LR0_SEEDS <<< "$FINETUNE_RUN_AB_LR0_SEEDS"
    for seed in "${LR0_SEEDS[@]}"; do
      f="$(leg_work_dir "$seed" fused lr0)/initial_adapter.safetensors"
      [ -f "$f" ] || missing+=("$f")
    done
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::FINETUNE_RUN_AB_ARMS=${FINETUNE_RUN_AB_ARMS} runs torch legs without the fused arm, but these seeds' initial adapters are not in OUT_DIR -- run their fused legs first:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 2
  fi
fi


TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN="$TARGET_DIR/release/jammi-bench"

# --- state-changing command wrapper (same shape as finetune_ab.sh/
# encode_ab.sh's own run_cmd): always echoes what it would run; under
# FINETUNE_RUN_AB_DRY_RUN never executes.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# The arms are pre-registered as "fused cascade vs ALLOFF=
# attention_block_flash,adamw_step_fused" -- the A/B's own differential IS
# the flash cascade. Building WITHOUT flash-attn makes
# attention_block_flash unable to dispatch in EITHER arm, nulling the
# experiment -- mirrors
# stacked_sweep.sh's own flash-A/B build feature list exactly
# (`--features cuda,jammi-encoders/flash-attn`), never a second,
# independently-drifting feature-list spelling.
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  run_cmd cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda,jammi-encoders/flash-attn failed" >&2; exit 1; }
fi

# --- provenance cross-check, same shape as
# finetune_ab.sh/encode_ab.sh/stacked_sweep.sh/
# clip_artifact_producer.sh: refuse BEFORE any leg runs if the binary's own
# baked identity does not match the sha this checkout is actually at.
SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHA_RE='^[0-9a-fA-F]{40}$'
if ! [[ "$SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$SHA') -- refusing" >&2
  exit 2
fi
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  BIN_PROV_JSON="$("$BIN" provenance 2>&1)" || { echo "::error::'$BIN provenance' failed: $BIN_PROV_JSON" >&2; exit 1; }
  BIN_PROV_SHA="$(printf '%s' "$BIN_PROV_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$BIN provenance' output: $BIN_PROV_JSON" >&2; exit 1; }
  if [ -z "$BIN_PROV_SHA" ] || [ "$BIN_PROV_SHA" != "$SHA" ]; then
    echo "::error::'$BIN provenance' reports build_sha=$BIN_PROV_SHA, but this run proves sha=$SHA -- refusing before any leg." >&2
    exit 1
  fi
fi

# --- one measurement leg (mirrors finetune_ab.sh/encode_ab.sh's own
# run_leg: NEVER aborts the sweep -- a leg failure is recorded as this
# leg's own outcome, so one seed's OOM/refusal does not discard every other
# seed's row).
#
# `arm` selects BOTH the CLI's own `--arm` flag (recorded on the report,
# report.rs's own PROVENANCE_FIELDS) AND, for the `alloff` arm
# only, the `JAMMI_KERNELS_DISABLE` env var this binary's own CLI doc names
# as the CALLER's responsibility ("the caller is responsible for setting
# JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused itself
# before invoking this binary for the alloff arm" -- main.rs's own
# `FinetuneRunArgs::arm` doc).
#
# Every leg is the SAME job: `$FINETUNE_RUN_AB_LR`, when set, is forwarded as
# `--lr` (unset omits it, i.e. the CLI's own 2e-4 default). A control leg
# (`repeat` = `lr0`) is that job run with `--zero-lr-control` -- every
# optimizer step applied at learning rate zero, reported as `lr: 0.0`. It is
# never `--lr 0`: a job that cannot learn is refused at admission, and the
# control is a way of RUNNING a valid job, not a job.
#
# `shared` is every flag that describes the RUN. `torch_finetune_run.py`
# takes them under the same names, so a `torch` leg is this same array handed
# to the other producer; only the producer-specific head (`cmd`) differs.
# The reference rung's arm — the flash cascade and fused AdamW off — as the
# `JAMMI_KERNELS_DISABLE` value `jammi-bench kernel-arm` derives from this
# checkpoint's admission census: never a list typed here.
# The derive adapts the SAME sites the legs do: `--target-modules` reaches it
# exactly when it reaches every leg (see `shared` below), else both default.
DERIVE=("$BIN" kernel-arm --model-dir "$MODEL_DIR" --off flash-attention,adam-w)
if [ -n "$FINETUNE_RUN_AB_TARGET_MODULES" ]; then
  DERIVE+=(--target-modules "$FINETUNE_RUN_AB_TARGET_MODULES")
fi
printf -- '--- kernel-arm: '; printf '%q ' "${DERIVE[@]}"; printf '\n'
REFERENCE_DISABLE="[dry-run]"
if [ "$FINETUNE_RUN_AB_DRY_RUN" != "1" ]; then
  REFERENCE_DISABLE="$("${DERIVE[@]}")" \
    || { echo "::error::'$BIN kernel-arm' failed on $MODEL_DIR -- refusing before any leg." >&2; exit 1; }
  echo "=== reference arm: JAMMI_KERNELS_DISABLE=$REFERENCE_DISABLE ==="
fi

# The rung a leg is filed under, and the directory: a natural-width torch
# leg is a diagnostic beside the run, never a rung of the ladder.
leg_rung() {
  case "$1" in
    fused) echo resident ;;
    alloff) echo resident-reference ;;
    torch|torch-natural) echo torch ;;
  esac
}
leg_dir() {
  if [ "$1" = "torch-natural" ]; then echo "$RAW_DIR/natural"; else echo "$RAW_DIR"; fi
}

run_leg() {
  local seed="$1" arm="$2" repeat="$3" work_dir="$4"
  local leg="$(leg_dir "$arm")/$(leg_rung "$arm")__seed${seed}__${repeat}"
  mkdir -p "$(leg_dir "$arm")"
  local out_file="$leg.json" err_file="$leg.stderr" exit_file="$leg.exit"

  local -a shared=(
    --model-dir "$MODEL_DIR"
    --train-jsonl "$TRAIN_JSONL"
    --heldout-ids "$HELDOUT_IDS"
    --heldout-jsonl "$HELDOUT_JSONL"
    --seed "$seed"
    --batch "$FINETUNE_RUN_AB_BATCH"
    --objective "$FINETUNE_RUN_AB_OBJECTIVE"
    # Early stopping DISABLED both arms -- the "never
    # stops before the pre-registered epoch budget" idiom, so a seed's
    # trajectory is never truncated by an early-stopping decision the sign
    # test would then have to account for.
    --early-stopping-patience 10000
    # `main.rs`'s own `--backbone-dtype` default
    # is `f32` (`FinetuneRunArgs::backbone_dtype`'s own `#[arg(long,
    # default_value = "f32")]`), and `flash_capability_gates` DomainMisses
    # the whole flash cascade whenever `dtype` is neither `DType::BF16` nor
    # `DType::F16` (`jammi-encoders/src/modernbert.rs`'s
    # `dtype_is_bf16_or_f16` gate — `f32` is outside that admitted set, and
    # this script's own default, `bf16`, is inside it). The flash cascade
    # is the `fused` arm's pre-registered admitted branch, so an unset
    # `--backbone-dtype` (silently f32) makes `attention_block_flash`
    # unable to fire on EITHER arm's real leg -- the same null differential
    # the flash-attn build feature above prevents. `backbone_dtype` is
    # also IDENTITY FIELD #10 on `FINETUNE_RUN_IDENTITY_FIELDS`
    # (`identity_fields.py`) -- cross-arm AND cross-seed homogeneity
    # requires every leg (both arms, every seed, INCLUDING the lr=0
    # control below) to report the SAME value, so this is passed
    # unconditionally here in the one `run_leg` both loops share, never
    # only on the `fused` arm. Value comes from
    # `$FINETUNE_RUN_AB_BACKBONE_DTYPE` (default "bf16" -- see that
    # env-var's own doc above).
    --backbone-dtype "$FINETUNE_RUN_AB_BACKBONE_DTYPE"
    --work-dir "$work_dir"
  )
  if [ -n "$FINETUNE_RUN_AB_EPOCHS" ]; then
    shared+=(--epochs "$FINETUNE_RUN_AB_EPOCHS")
  fi
  if [ -n "$FINETUNE_RUN_AB_LR" ]; then
    shared+=(--lr "$FINETUNE_RUN_AB_LR")
  fi
  if [ "$repeat" = "lr0" ]; then
    shared+=(--zero-lr-control)
  fi
  if [ -n "$FINETUNE_RUN_AB_MAX_SEQ_LENGTH" ]; then
    shared+=(--max-seq-length "$FINETUNE_RUN_AB_MAX_SEQ_LENGTH")
  fi
  if [ -n "$FINETUNE_RUN_AB_LORA_DROPOUT" ]; then
    shared+=(--lora-dropout "$FINETUNE_RUN_AB_LORA_DROPOUT")
  fi
  if [ -n "$FINETUNE_RUN_AB_TARGET_MODULES" ]; then
    shared+=(--target-modules "$FINETUNE_RUN_AB_TARGET_MODULES")
  fi
  if [ "$FINETUNE_RUN_AB_CPU" != "1" ]; then
    shared+=(--cuda "$FINETUNE_RUN_AB_CUDA")
  fi

  local -a cmd
  if [ "$arm" = "torch" ] || [ "$arm" = "torch-natural" ]; then
    local width=bucketed
    if [ "$arm" = "torch-natural" ]; then
      width=natural
    fi
    # The untrained adapter the seed's FIRST jammi leg wrote into its work
    # dir: every jammi leg of a seed writes the same bytes, and this one has
    # always run by the time a torch leg does. A control seed has no r1 leg;
    # its first jammi leg is its fused control leg.
    local first_jammi_repeat=r1
    if [ "$repeat" = "lr0" ]; then
      first_jammi_repeat=lr0
    fi
    cmd=(
      "$TORCH_PY" "$TORCH_SCRIPT"
      --lora-init zeros_b
      --initial-adapter "$(leg_work_dir "$seed" fused "$first_jammi_repeat")/initial_adapter.safetensors"
      --attn "$FINETUNE_RUN_AB_TORCH_ATTN"
      --width "$width"
      "${shared[@]}"
    )
  else
    cmd=("$BIN" finetune-run --arm "$arm" "${shared[@]}")
  fi

  printf -- '--- seed%s/%s/%s: ' "$seed" "$arm" "$repeat"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$FINETUNE_RUN_AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"seed":%s,"arm":"%s","repeat":"%s"}\n' \
      "$seed" "$arm" "$repeat" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  if [ "$arm" = "alloff" ]; then
    JAMMI_KERNELS_DISABLE="$REFERENCE_DISABLE" "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  else
    "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  fi
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::seed${seed}/${arm}/${repeat} FAILED (exit ${rc}) — recorded as a leg outcome; sweep continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

# One seed's legs, in run order -- see "LEG ORDER" in the header.
SEED_LEGS=(fused:r1 alloff:r1)
if [ "$FINETUNE_RUN_AB_TORCH" = "1" ]; then
  SEED_LEGS+=(torch:r1 torch-natural:r1 torch-natural:r2 torch:r2)
fi
SEED_LEGS+=(alloff:r2 fused:r2)

for seed in "${SEEDS[@]}"; do
  for leg in "${SEED_LEGS[@]}"; do
    arm="${leg%%:*}"
    repeat="${leg##*:}"
    arm_selected "$arm" || continue
    work_dir="$(leg_work_dir "$seed" "$arm" "$repeat")"
    mkdir -p "$work_dir"
    run_leg "$seed" "$arm" "$repeat" "$work_dir"
  done
done

# --- lr=0 RED control legs:
# both arms, with --zero-lr-control, tagged with ab_merge.py's own FINETUNE_RUN_LR0_REPEAT
# label ("lr0") -- a DISTINCT repeat token from r1/r2, so these legs are
# never picked up by the main sweep's own r1/r2 loader and never enter the
# A/B set. Skipped entirely (no legs, no wiring cost) when
# FINETUNE_RUN_AB_LR0_SEEDS is unset -- an operator opts in explicitly.
if [ -n "$FINETUNE_RUN_AB_LR0_SEEDS" ]; then
  IFS=',' read -r -a LR0_SEEDS <<< "$FINETUNE_RUN_AB_LR0_SEEDS"
  LR0_ARMS=(fused alloff)
  if [ "$FINETUNE_RUN_AB_TORCH" = "1" ]; then
    LR0_ARMS+=(torch)
  fi
  for seed in "${LR0_SEEDS[@]}"; do
    for arm in "${LR0_ARMS[@]}"; do
      arm_selected "$arm" || continue
      work_dir="$(leg_work_dir "$seed" "$arm" lr0)"
      mkdir -p "$work_dir"
      run_leg "$seed" "$arm" "lr0" "$work_dir"
    done
  done
fi

# --- the verdict: the kernel edge of the `train-run` ladder, and the
# framework edge below it when the torch arm ran. Every premise, the identity
# check, the paired sign test, the derived margin and the mutant columns are
# the ladder's. The lr=0 control is not silently optional: the ladder refuses
# the edge without it unless the operator waives it, which the verdict
# records. Mutant legs produced outside this script (docs/plans/63-how-well/
# mutants/README.md) under `mutant-<dose_label>` rung names in $RAW_DIR are
# named to the ladder as `--mutant DOSE_LABEL:PATCH_SHA256`, one per
# ';'-separated FINETUNE_RUN_AB_MUTANT_LEGS entry.
FINETUNE_RUN_AB_ALLOW_NO_LR0="${FINETUNE_RUN_AB_ALLOW_NO_LR0:-0}"
LADDER_FROM=resident-reference
if [ "$FINETUNE_RUN_AB_TORCH" = "1" ]; then
  LADDER_FROM=torch
fi
LADDER_ARGS=(ladder train-run "$RAW_DIR" --from "$LADDER_FROM" --to resident --axes outcome --out "$OUT_DIR")
if [ "$FINETUNE_RUN_AB_ALLOW_NO_LR0" = "1" ]; then
  LADDER_ARGS+=(--waive-control)
fi
FINETUNE_RUN_AB_MUTANT_LEGS="${FINETUNE_RUN_AB_MUTANT_LEGS:-}"
if [ -n "$FINETUNE_RUN_AB_MUTANT_LEGS" ]; then
  IFS=';' read -r -a MUTANT_LEG_SPECS <<< "$FINETUNE_RUN_AB_MUTANT_LEGS"
  for spec in "${MUTANT_LEG_SPECS[@]}"; do
    [ -n "$spec" ] && LADDER_ARGS+=(--mutant "$spec")
  done
fi
run_cmd "$BIN" "${LADDER_ARGS[@]}"
LADDER_RC=$?
echo
echo "=== raw legs + ladder verdict: ${OUT_DIR} ==="
exit "$LADDER_RC"
