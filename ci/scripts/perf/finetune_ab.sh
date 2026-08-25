#!/usr/bin/env bash
# The #352 A/B: jammi eager vs jammi fused vs the PyTorch/PEFT reference —
# runs ON THE POD, invoked either via
#   ci/scripts/gpu-dev.sh run <session> bash ci/scripts/perf/finetune_ab.sh
# or directly over ssh once the checkout is on the pod. NOT a CI job (no
# GPU on the CI image, and this script switches git refs in place — never
# run it against a checkout you care about keeping put).
#
# What it does, for each of {b8 s128, b8 s512, b16 s128} x {dropout 0,
# dropout 0.05} (6 configs):
#   1. jammi eager   — the pre-fusion commit, JAMMI_KERNELS_STRICT unset.
#   2. jammi fused   — HEAD, JAMMI_KERNELS_STRICT=1 (an admission failure on
#      any fused op ERRORS instead of silently falling back — see
#      jammi-encoders/src/layer_norm.rs's `admission_mode`), so the run
#      cannot pass on a silent eager fallback.
#   3. torch eager   — crates/jammi-bench/reference/torch_finetune_step.py
#      --attn eager --lora-init peft --dtype bf16 (jammi eager's semantic
#      twin: no fused attention kernel).
#   4. torch sdpa    — the same script --attn sdpa (torch's best-case
#      number; what the #352 throughput ratio is measured against).
# Emits one merged JSON report + a printed table: s/step p50, triplets/s,
# peak VRAM (delta, comparable across stacks, and absolute where torch has
# it — see "VRAM columns" below), EVERY fused-kernel dispatch counter PAIR
# present in the jammi-fused report (the positive-proof channel `ab_merge.py`'s
# `fused_proof` computes: `eager == 0` on EVERY pair, ALWAYS, no exceptions —
# but `fused > 0` is NOT a flat per-pair rule; it is the classification
# `ab_merge.py`'s `REQUIRED_PAIRS`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
# `LORA_SITE_EXCLUSIVE_GROUP` (their union, `ALL_BASES`) declares and that
# module's own doc is the authority on — restated here only for a reader of
# THIS script, not duplicated logic:
#     * ln, geglu, attention_block   — each MUST independently show fused > 0.
#     * rope, softmax                — MUST be present; may read (0, 0) ONLY
#                                       when attention_block's OWN fused count
#                                       is > 0 this run (its fused arm folds
#                                       RoPE+softmax into itself and never
#                                       calls their independent sites at all);
#                                       otherwise each independently needs
#                                       fused > 0.
#     * lora_epilogue, lora_linear   — MUST both be present; mutually
#                                       EXCLUSIVE call sites for the same LoRA
#                                       adapted forward, so only their SUM
#                                       needs fused > 0 — either one alone may
#                                       legitimately read (0, 0).
# Every classified base (present in `ALL_BASES`) must be PRESENT in the
# report at all — an ABSENT pair (the field renamed, deleted, or
# feature-gated off) is a hard fail for every one of them, never silently
# excluded. A pair discovered in the report but NOT in `ALL_BASES` (a NEW
# fused kernel this script's classification tables were never updated for)
# is a loud per-config schema-drift ERROR, not a silently-ignored base — see
# `fused_proof`/`dispatch_pairs` in the merge stage for the full rule table
# and "whatever lands next" being read off the report's own keys rather than
# a hardcoded name list), each leg's per-step loss_first->loss_last and a
# loss_final_ratio jammi-fused/torch-sdpa column (SAME DATA, COST FIXTURE —
# NOT A QUALITY RESULT, printed only so a large divergence is visible — see
# "loss_final_ratio" below), the ratio jammi-fused/torch-sdpa, and a
# PASS/FAIL against #352's bar (>= 0.9x torch-sdpa throughput at matched
# batch/seq, no OOM on a config torch itself completed). Like every
# jammi-bench tier, this RECORDS — it does not gate the process exit code
# on a config missing the bar; a FAIL row is data for a human to read, not
# an infrastructure failure (see finetune_step.rs's own module doc). The
# script's own exit code reflects whether the sweep RAN, not whether every
# config passed.
#
# NOT covered here: loss-TRAJECTORY equivalence between jammi-fused and
# jammi-eager (the #352 quality constraint) is a REAL-TRAINER check over
# >= 5 seeds reusing C0's distributional oracle machinery — a different,
# slower harness than this one-step-timing sweep. Run it separately. The
# loss_first/loss_last/loss_final_ratio columns THIS script prints are a
# different, weaker thing: one synthetic-data cost-fixture step count from
# `finetune-step`/`torch_finetune_step.py` itself, printed for visibility,
# never a substitute for that real-trainer check.
#
# loss_final_ratio (jammi-fused loss_last / torch-sdpa loss_last): SAME
# DATA (both stacks feed the identical synthetic token ids — the two
# scripts' `synthetic_ids` are a literal LCG port of each other; see
# torch_finetune_step.py's module doc), but NOT a quality result: the two
# stacks run different attention-kernel arithmetic and, at this script's
# `--lora-init peft` default, different LoRA init distributions, so a ratio
# far from 1.0 does not indicate either stack regressed — it indicates the
# comparison's preconditions (matched init, matched attention arithmetic)
# were not met, which is exactly why this rides as a printed reference, not
# a gated bar.
#
# Leg resolution is BY COMMIT SUBJECT, never by position in history or a
# hardcoded SHA (the stale-build lesson generalized: a script that names a
# SHA goes stale the moment someone rebases; a script that greps a subject
# survives every commit after it). "eager base" is the commit whose subject
# contains BASE_SUBJECT_MATCH below (the last commit before any fused
# kernel landed); "fused tip" is always HEAD at invocation time.
#
# Stale-build guard: every git-ref switch is followed by
# `cargo clean -p jammi-kernels --release` THEN a full release rebuild —
# see checkout_and_build()'s comment for why this is not optional. A run
# that skips it risks comparing a cached binary against itself, not eager
# against fused.
#
# Torch env: `uv venv "$TORCH_VENV"` (default crates/jammi-bench/reference/
# README.md's own `.venv-torch-ref` convention, resolved under the repo
# root) then `uv pip install --python "$TORCH_VENV/bin/python3" torch
# "transformers>=4.48" peft`. Tolerates an existing venv: if the interpreter
# is already there AND can `import torch, transformers, peft`, it is reused
# rather than reprovisioned (each `uv pip install` re-downloads real GPU
# wheels, and that cost belongs on the first pod session of the day, not
# every invocation of this script). torch/transformers/peft are ORACLE
# dependencies per crates/jammi-bench/reference/README.md's own B2 section —
# never a Cargo dependency, never installed by any CI job, never vendored;
# this script's `uv venv`/`uv pip install` calls are the only place they are
# installed, and only on a pod, only into a venv this script owns.
#
# VRAM columns (binds C8 contract section 2's rules — read
# torch_finetune_step.py's own module doc / crates/jammi-bench/reference/
# README.md for the full derivation; not re-derived here, only the two
# columns this table prints):
#   * "vram_delta(comparable)" — jammi's `peak_vram_bytes` next to torch's
#     `peak_vram_delta_bytes`: same concept (device-memory growth over a
#     baseline read after model+optimizer are resident), the pair this
#     table's numbers are meant to be read against each other.
#   * "vram_absolute(torch only)" — torch's `peak_vram_absolute_bytes`
#     (raw bytes live at the peak, no baseline subtraction). jammi has no
#     analogous field; jammi rows print "n/a" here, not a zero.
#
# Every LoRA-shaped and dtype flag is passed EXPLICITLY and identically to
# both stacks (r=16, alpha=32, dropout as swept, targets Wqkv,Wo,Wi, bf16),
# matching the C8 contract's section 1 spec — this deliberately overrides
# jammi-bench's own CLI defaults (rank 8 / alpha 16), which differ from the
# torch reference script's C8-contract defaults (rank 16 / alpha 32) on
# purpose (see torch_finetune_step.py's --lora-rank/--lora-alpha help text).
# --lora-init defaults to `peft` (AB_TORCH_LORA_INIT, below) — right for
# throughput rows, where the adapter's initial values do not matter. Set
# AB_TORCH_LORA_INIT=jammi to sweep the loss-TRAJECTORY-equivalence
# precondition instead (torch_finetune_step.py re-draws every lora_A matrix
# from jammi's own bound — see that script's own LoRA-init section). jammi's
# own leg always uses its LoraInitMode::ZerosB init; it has no CLI knob for
# this today, so it is not itself switchable — the merged report's
# top-level "lora_init" block records which value each side actually used.
#
# Env vars:
#   MODEL_DIR            checkpoint dir (config.json + weights) BOTH stacks
#                         load from. Required unless AB_DRY_RUN=1.
#   JAMMI_MODEL_DIR       overrides MODEL_DIR for the jammi legs only.
#   TORCH_MODEL_DIR       overrides MODEL_DIR for the torch legs only.
#   AB_CUDA_ORDINAL       CUDA device ordinal both stacks target (default 0).
#   AB_STEPS / AB_WARMUP  measured / warmup step counts (default 20 / 5,
#                         matching both CLIs' own defaults).
#   AB_SEED               synthetic-data + init seed (default 42).
#   AB_PASS_RATIO         the #352 throughput bar (default 0.9).
#   AB_TORCH_LORA_INIT    torch_finetune_step.py's --lora-init for BOTH
#                         torch legs (default "peft" — throughput rows;
#                         "jammi" is required before a loss_final_ratio
#                         column means anything as a trajectory-equivalence
#                         signal — see torch_finetune_step.py's own LoRA-init
#                         section). Must be "peft" or "jammi".
#   AB_OUT_DIR            where the merged report + table land (default
#                         "<repo>/.ab-report/<UTC timestamp>").
#   TORCH_VENV            torch venv path (default "<repo>/.venv-torch-ref").
#   BASE_SUBJECT_MATCH    override the eager-base commit-subject substring
#                         (default below — the lead names the real one).
#   AB_DRY_RUN=1          print every command this script would run (git,
#                         cargo, uv, the bench binary, the torch script)
#                         instead of executing it, and write a
#                         `{"tool":"dry-run",...}` stub per leg so the
#                         merge/table stage still runs end-to-end against
#                         real (if fabricated-empty) files. Never mutates
#                         the checkout, never touches the network, never
#                         claims a real number — every dry-run row prints
#                         outcome DRY_RUN, never OK/FAIL/OOM.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

# The eager-base commit's subject substring — NOT its SHA. A SHA in this
# script would go stale the first time this branch is rebased or the base
# commit is cherry-picked elsewhere; a subject substring survives that
# because it names WHAT the commit is, not WHERE it sits in history.
BASE_SUBJECT_MATCH="${BASE_SUBJECT_MATCH:-make the K3 standardization oracles distributional}"

AB_DRY_RUN="${AB_DRY_RUN:-0}"
AB_CUDA_ORDINAL="${AB_CUDA_ORDINAL:-0}"
AB_STEPS="${AB_STEPS:-20}"
AB_WARMUP="${AB_WARMUP:-5}"
AB_SEED="${AB_SEED:-42}"
AB_PASS_RATIO="${AB_PASS_RATIO:-0.9}"
AB_TORCH_LORA_INIT="${AB_TORCH_LORA_INIT:-peft}"
case "$AB_TORCH_LORA_INIT" in
  peft|jammi) ;;
  *)
    echo "::error::AB_TORCH_LORA_INIT must be 'peft' or 'jammi', got '${AB_TORCH_LORA_INIT}'."
    exit 2
    ;;
esac
TORCH_VENV="${TORCH_VENV:-$REPO_ROOT/.venv-torch-ref}"

MODEL_DIR="${MODEL_DIR:-}"
JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-$MODEL_DIR}"
TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-$MODEL_DIR}"

if [ -z "$JAMMI_MODEL_DIR" ] || [ -z "$TORCH_MODEL_DIR" ]; then
  if [ "$AB_DRY_RUN" = "1" ]; then
    JAMMI_MODEL_DIR="${JAMMI_MODEL_DIR:-/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER}"
    TORCH_MODEL_DIR="${TORCH_MODEL_DIR:-/root/checkpoints/ModernBERT-large-DRY-RUN-PLACEHOLDER}"
    echo "::warning::AB_DRY_RUN=1 and MODEL_DIR unset — printed commands use a placeholder path; nothing is read from it."
  else
    echo "::error::MODEL_DIR (or JAMMI_MODEL_DIR/TORCH_MODEL_DIR) must name a checkpoint directory."
    exit 2
  fi
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${AB_OUT_DIR:-$REPO_ROOT/.ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

CONFIGS=("8:128" "8:512" "16:128")
DROPOUTS=("0" "0.05")

# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

# A state-changing command (git/cargo/uv). Always echoes what it would run;
# under AB_DRY_RUN it never executes. Returns the real exit status otherwise.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# One measurement leg (a jammi or torch invocation). NEVER aborts the
# sweep: OOM or any other failure is recorded as this row's outcome, not
# propagated as a script error — the whole point of an A/B sweep is that
# one config OOM-ing tells you something; it must not hide the other five.
run_leg() {
  local config_slug="$1" leg="$2"
  shift 2
  local -a cmd=("$@")
  local out_file="$RAW_DIR/${config_slug}__${leg}.json"
  local err_file="$RAW_DIR/${config_slug}__${leg}.stderr"
  local exit_file="$RAW_DIR/${config_slug}__${leg}.exit"

  printf -- '--- %s / %s: ' "$config_slug" "$leg"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [ "$AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"config":"%s","leg":"%s"}\n' \
      "$config_slug" "$leg" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "${cmd[@]}" > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${config_slug}/${leg} FAILED (exit ${rc}) — recorded as a row outcome; sweep continues."
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

slug_for() {
  local batch="$1" seq="$2" dropout="$3"
  local dslug
  case "$dropout" in
    0) dslug="d0" ;;
    *) dslug="d${dropout//./p}" ;;
  esac
  printf 'b%s-s%s-%s' "$batch" "$seq" "$dslug"
}

run_jammi_leg() {
  local strict="$1" leg="$2" binary="$3" model_dir="$4" batch="$5" seq="$6" dropout="$7" config_slug="$8"
  local -a cmd=(
    "$binary" finetune-step
    --model-dir "$model_dir"
    --batch "$batch" --seq "$seq"
    --steps "$AB_STEPS" --warmup "$AB_WARMUP"
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$dropout"
    --target-modules "Wqkv,Wo,Wi"
    --backbone-dtype bf16
    --cuda "$AB_CUDA_ORDINAL" --seed "$AB_SEED"
    --batched-forward true
  )
  if [ "$strict" = "1" ]; then
    # Strict admission: an eligible-but-failed fused op ERRORS instead of
    # falling back, so this leg cannot silently "pass" on eager numbers
    # wearing a fused label (jammi-encoders/src/layer_norm.rs::admission_mode).
    JAMMI_KERNELS_STRICT=1 run_leg "$config_slug" "$leg" "${cmd[@]}"
  else
    run_leg "$config_slug" "$leg" "${cmd[@]}"
  fi
}

run_torch_leg() {
  local attn="$1" leg="$2" python_bin="$3" ref_script="$4" model_dir="$5" batch="$6" seq="$7" dropout="$8" config_slug="$9"
  local -a cmd=(
    "$python_bin" "$ref_script"
    --model-dir "$model_dir"
    --batch "$batch" --seq "$seq"
    --steps "$AB_STEPS" --warmup "$AB_WARMUP"
    --lora-rank 16 --lora-alpha 32 --lora-dropout "$dropout"
    --target-modules "Wqkv,Wo,Wi"
    --dtype bf16 --attn "$attn" --lora-init "$AB_TORCH_LORA_INIT"
    --cuda "$AB_CUDA_ORDINAL" --seed "$AB_SEED"
  )
  run_leg "$config_slug" "$leg" "${cmd[@]}"
}

# Provision (or reuse) the torch reference venv — crates/jammi-bench/
# reference/README.md's own `.venv-torch-ref` convention.
setup_torch_venv() {
  local py="$TORCH_VENV/bin/python3"
  if [ -x "$py" ] && [ "$AB_DRY_RUN" != "1" ] && "$py" -c 'import torch, transformers, peft' >/dev/null 2>&1; then
    echo "torch venv at $TORCH_VENV already has torch+transformers+peft — reusing."
    return 0
  fi
  if [ -x "$py" ] && [ "$AB_DRY_RUN" = "1" ]; then
    echo "torch venv at $TORCH_VENV exists — [dry-run] would verify torch+transformers+peft import before reprovisioning."
  fi
  echo "provisioning torch venv at $TORCH_VENV"
  run_cmd uv venv "$TORCH_VENV" || { echo "::error::uv venv failed"; exit 1; }
  run_cmd uv pip install --python "$py" torch "transformers>=4.48" peft \
    || { echo "::error::uv pip install (torch/transformers/peft) failed"; exit 1; }
}

# Switch the checkout to $1 and force a full jammi-kernels rebuild.
#
# STALE-BUILD GUARD, and why it is not optional: `cargo build` decides
# whether to recompile jammi-kernels off its own fingerprint (source hashes,
# env vars, build.rs's recorded outputs) — NOT off "did `git checkout` just
# run". jammi-kernels' build.rs drives a feature-gated CUDA compile, and a
# checkout switch between two refs that both already have a jammi-kernels
# build.rs (eager base and fused tip both do — jammi-kernels' scaffolding
# predates the fused ops) can leave cargo's fingerprint satisfied by the
# PREVIOUS ref's compiled artifact, because nothing about that artifact's
# recorded inputs necessarily changed shape across the switch. A run that
# skipped this step measured exactly that failure: two "different" legs
# (eager and fused) whose reported numbers were bit-identical across a
# checkout that changed jammi-kernels' own admission logic — i.e. the sweep
# was silently comparing one stale kernels binary against itself, not eager
# against fused. `cargo clean -p jammi-kernels --release` forces every
# checkout switch to pay a full jammi-kernels rebuild, which is the only way
# to be sure the binary under test was actually built FROM the ref that is
# checked out, not carried over from whichever ref built last.
checkout_and_build() {
  local ref="$1"
  echo "=== checking out ${ref} ==="
  run_cmd git -C "$REPO_ROOT" checkout --quiet "$ref" \
    || { echo "::error::git checkout ${ref} failed"; exit 1; }
  run_cmd cargo clean -p jammi-kernels --release --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo clean -p jammi-kernels failed"; exit 1; }
  run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$REPO_ROOT/Cargo.toml" \
    || { echo "::error::cargo build -p jammi-bench --features cuda failed"; exit 1; }
}

# ---------------------------------------------------------------------- #
# resolve legs (by commit SUBJECT, never position — see header)
# ---------------------------------------------------------------------- #

if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ] && [ "$AB_DRY_RUN" != "1" ]; then
  echo "::error::working tree at $REPO_ROOT is not clean — this script switches git refs in place; commit or stash first."
  exit 1
fi

ORIGINAL_REF="$(git -C "$REPO_ROOT" symbolic-ref -q --short HEAD || git -C "$REPO_ROOT" rev-parse HEAD)"
FUSED_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [ "$AB_DRY_RUN" != "1" ] && [ "$(git -C "$REPO_ROOT" rev-parse --is-shallow-repository)" = "true" ]; then
  git -C "$REPO_ROOT" fetch --unshallow --quiet 2>/dev/null || true
fi

BASE_MATCHES="$(git -C "$REPO_ROOT" log --all --format='%H%x09%s' | grep -F -- "$BASE_SUBJECT_MATCH" || true)"
if [ -z "$BASE_MATCHES" ]; then
  echo "::error::no commit subject contains BASE_SUBJECT_MATCH='${BASE_SUBJECT_MATCH}' — set BASE_SUBJECT_MATCH to the real eager-base commit's subject substring."
  exit 1
fi
BASE_MATCH_COUNT="$(printf '%s\n' "$BASE_MATCHES" | grep -c .)"
if [ "$BASE_MATCH_COUNT" -gt 1 ]; then
  echo "::error::BASE_SUBJECT_MATCH='${BASE_SUBJECT_MATCH}' matches ${BASE_MATCH_COUNT} commits — ambiguous, refusing to guess:"
  printf '%s\n' "$BASE_MATCHES"
  exit 1
fi
BASE_SHA="$(printf '%s\n' "$BASE_MATCHES" | cut -f1)"

echo "=== eager base:  ${BASE_SHA}  (subject contains '${BASE_SUBJECT_MATCH}') ==="
echo "=== fused tip:    ${FUSED_SHA}  (HEAD at invocation) ==="
echo "=== restoring to: ${ORIGINAL_REF} when the sweep finishes ==="

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
JAMMI_BIN="$TARGET_DIR/release/jammi-bench"
REF_SCRIPT="$REPO_ROOT/crates/jammi-bench/reference/torch_finetune_step.py"
TORCH_PY="$TORCH_VENV/bin/python3"

# ---------------------------------------------------------------------- #
# phase A — fused tip: jammi-fused legs + both torch legs
# ---------------------------------------------------------------------- #
checkout_and_build "$FUSED_SHA"
setup_torch_venv

for cfg in "${CONFIGS[@]}"; do
  BATCH="${cfg%%:*}"; SEQ="${cfg##*:}"
  for DROPOUT in "${DROPOUTS[@]}"; do
    SLUG="$(slug_for "$BATCH" "$SEQ" "$DROPOUT")"
    run_jammi_leg 1 jammi-fused "$JAMMI_BIN" "$JAMMI_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
    run_torch_leg eager torch-eager "$TORCH_PY" "$REF_SCRIPT" "$TORCH_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
    run_torch_leg sdpa  torch-sdpa  "$TORCH_PY" "$REF_SCRIPT" "$TORCH_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
  done
done

# ---------------------------------------------------------------------- #
# phase B — eager base: jammi-eager legs only
# ---------------------------------------------------------------------- #
checkout_and_build "$BASE_SHA"

for cfg in "${CONFIGS[@]}"; do
  BATCH="${cfg%%:*}"; SEQ="${cfg##*:}"
  for DROPOUT in "${DROPOUTS[@]}"; do
    SLUG="$(slug_for "$BATCH" "$SEQ" "$DROPOUT")"
    run_jammi_leg 0 jammi-eager "$JAMMI_BIN" "$JAMMI_MODEL_DIR" "$BATCH" "$SEQ" "$DROPOUT" "$SLUG"
  done
done

# ---------------------------------------------------------------------- #
# phase C — leave the pod on the ref it started on
# ---------------------------------------------------------------------- #
checkout_and_build "$ORIGINAL_REF"

# ---------------------------------------------------------------------- #
# merge + table
# ---------------------------------------------------------------------- #
# B3: this used to be an inline heredoc with zero automated coverage
# (AB_DRY_RUN=1 only ever exercised the DRY_RUN arm). It is now
# ci/scripts/perf/ab_merge.py, an importable module `ci/scripts/perf/
# test_ab_merge.py` drives directly against fixture leg directories — this
# call is exactly the "real entry point" that test suite exercises.
python3 "$DIR/ab_merge.py" "$RAW_DIR" "$OUT_DIR" "$AB_STEPS" "$AB_WARMUP" "$AB_PASS_RATIO" "$AB_TORCH_LORA_INIT"
PY_RC=$?

echo
echo "=== merged report + table: ${OUT_DIR} ==="
exit "$PY_RC"

