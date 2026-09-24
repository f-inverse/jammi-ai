#!/usr/bin/env bash
# A GPU campaign from a roster: rent the pods, pin each to one pushed
# commit, run each pod's producer, watch them, pull their legs, merge the
# ladders a campaign spreads over pods, and tear the pods down. It runs
# producers and decides nothing; every verdict is `jammi-bench ladder`'s.
#
# Every pod is reached through `gpu-dev.sh`'s own verbs — up, wait-seed,
# target, run, exec, pull, down — never its session files. Every per-pod
# child runs with stdin from /dev/null, stated rather than left to the
# shell: bash gives a background child /dev/null only while job control is
# off, and a shell that does not (zsh) lets the first child that reads stdin
# (ssh does) swallow the roster the loop is still reading.
#
# THE ROSTER. One pod per line, `#` comments and blank lines ignored:
#
#   <session> <arch> <producer> [KEY=VALUE ...]
#
#   producer   train-run    finetune_run_ab.sh
#              train-step   finetune_step_ab.sh
#              encode       encode_ab.sh
#              cpu-ladder   cpu_ladders_ab.sh
#   KEY=VALUE  the producer's own knobs, exported on the pod before it runs
#              (e.g. `FINETUNE_RUN_AB_SEEDS=3,4`, `CPU_AB_WORKLOAD=propagate`)
#
# Sessions sharing a producer pool into one ladder only through `merge`,
# and only if they ran on one GPU model: `rent` refuses a roster whose
# lines of one pooled producer name different arches.
#
# VERBS
#   campaign.sh rent ROSTER REF       rent every roster session not yet live, on REF
#                                      (a pushed branch), retrying on capacity
#   campaign.sh dispatch ROSTER SHA   per pod: wait for its seed build, clone the warm
#                                      target, check out SHA (pushed) and refuse unless
#                                      HEAD is SHA with a clean tree, launch its producer
#   campaign.sh status ROSTER         one line per pod: seed, job, HEAD, the log's last line
#   campaign.sh pull ROSTER DEST      each pod's outputs → DEST/<session>/
#   campaign.sh merge ROSTER DEST PRODUCER TO
#                                      pool the legs of every PRODUCER session in DEST
#                                      into DEST/<producer>/legs and run its ladder from
#                                      torch to the rung TO
#   campaign.sh down ROSTER           tear down every roster session
#   campaign.sh pod-job SESSION PRODUCER [KEY=VALUE ...]
#                                      ON A POD, from its checkout: provision what the
#                                      producer needs, run it into .campaign-out/SESSION/
#
# CAMPAIGN_DRY_RUN=1 prints every pod-side command instead of running it;
# GPU_DEV names the pod CLI (default: this checkout's `ci/scripts/gpu-dev.sh`).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"
GPU_DEV="${GPU_DEV:-$REPO_ROOT/ci/scripts/gpu-dev.sh}"
CAMPAIGN_DRY_RUN="${CAMPAIGN_DRY_RUN:-0}"
OUT_ROOT=".campaign-out"
LOG_DIR="${CAMPAIGN_LOG_DIR:-$REPO_ROOT/$OUT_ROOT/logs}"
MODEL_DIR_ON_POD="/root/checkpoints/ModernBERT-large"
MODEL_REPO="answerdotai/ModernBERT-large"

die() { echo "campaign.sh: $*" >&2; exit 2; }

# The roster's lines, comments and blanks dropped: `session arch producer knobs...`.
roster_lines() {
  [ -f "$1" ] || die "no roster at $1"
  grep -v '^\s*#' "$1" | grep -v '^\s*$'
}

# Run `$@ SESSION ARCH PRODUCER KNOB...` once per roster line, each in its
# own process with stdin from /dev/null; wait for all.
per_pod() {
  local roster="$1"; shift
  local line
  local -a fields
  while read -r line; do
    read -r -a fields <<< "$line"
    "$@" "${fields[@]}" < /dev/null &
  done < <(roster_lines "$roster")
  wait
}

producer_script() {
  case "$1" in
    train-run) echo finetune_run_ab.sh ;;
    train-step) echo finetune_step_ab.sh ;;
    encode) echo encode_ab.sh ;;
    cpu-ladder) echo cpu_ladders_ab.sh ;;
    *) die "unknown producer '$1' (train-run, train-step, encode, cpu-ladder)" ;;
  esac
}

# The producer's own output-directory knob.
producer_out_var() {
  case "$1" in
    train-run) echo FINETUNE_RUN_AB_OUT_DIR ;;
    train-step) echo FINETUNE_STEP_AB_OUT_DIR ;;
    encode) echo ENCODE_AB_OUT_DIR ;;
    cpu-ladder) echo CPU_AB_OUT_DIR ;;
  esac
}

# The legs directories a producer files under its output directory.
producer_legs_dirs() {
  case "$1" in
    train-run|train-step) echo raw ;;
    encode) echo "legs-plan legs-sorted" ;;
    cpu-ladder) echo legs ;;
  esac
}

gpu_dev() {
  if [ "$CAMPAIGN_DRY_RUN" = 1 ]; then
    printf 'gpu-dev'; printf ' %q' "$@"; printf '\n'
  else
    "$GPU_DEV" "$@"
  fi
}

# ------------------------------------------------------------------ rent
pooled_arch_check() {
  roster_lines "$1" | awk '{ if (($3 in arch) && arch[$3] != $2) { print $3 ": " arch[$3] " and " $2; bad = 1 } arch[$3] = $2 }
    END { exit bad }' \
    || die "a pooled producer's sessions name different arches — one ladder pools one GPU model"
}

# A session answering `exec` is live. Asked three times before a session is
# judged absent: renting over a live one that missed one probe would bill
# a second pod under the same name.
session_live() {
  local try
  for try in 1 2 3; do
    gpu_dev exec "$1" true >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

rent_one() {
  local ref="$1" session="$2" arch="$3"
  if [ "$CAMPAIGN_DRY_RUN" != 1 ] && session_live "$session"; then
    echo "$session: live"; return 0
  fi
  local try
  for try in 1 2 3; do
    RP_SESSION="$session" gpu_dev up "$arch" --ref "$ref" --replace > "$LOG_DIR/rent-$session.log" 2>&1 \
      && { echo "$session: rented ($arch)"; return 0; }
    [ "$CAMPAIGN_DRY_RUN" = 1 ] && return 0
    sleep 60
  done
  echo "$session: NOT RENTED after 3 tries — see $LOG_DIR/rent-$session.log"
  return 1
}

# ------------------------------------------------------------------ dispatch
dispatch_one() {
  local sha="$1" session="$2" _arch="$3" producer="$4"; shift 4
  producer_script "$producer" >/dev/null
  gpu_dev wait-seed "$session" --timeout 5400 >/dev/null || { echo "$session: seed build failed"; return 1; }
  gpu_dev target "$session" jammi-ai --with-cutlass >/dev/null 2>&1 \
    || gpu_dev target "$session" jammi-ai --adopt >/dev/null \
    || { echo "$session: no warm target"; return 1; }
  # Checked out and verified BEFORE the job launches: a leg's build_sha is
  # this HEAD, a dirty tree's legs are refused by every producer, and a SHA
  # the remote does not have fails the checkout here.
  local seen
  seen="$(gpu_dev exec "$session" "git fetch -q origin && git checkout -q --detach $sha && git submodule update --init -q crates/jammi-kernels/third_party/cutlass && echo \"\$(git rev-parse HEAD) \$(git status --porcelain --untracked-files=no | wc -l)\"" 2>/dev/null | tail -1)"
  if [ "$CAMPAIGN_DRY_RUN" != 1 ] && [ "$seen" != "$sha 0" ]; then
    echo "$session: NOT at a clean $sha — saw '$seen'"; return 1
  fi
  local knobs=""
  [ $# -gt 0 ] && knobs="$(printf ' %q' "$@")"
  gpu_dev run "$session" "bash ci/scripts/perf/campaign.sh pod-job $session $producer$knobs" \
    >/dev/null || { echo "$session: launch refused"; return 1; }
  echo "$session: launched $producer$knobs at $sha"
}

# ------------------------------------------------------------------ status
status_one() {
  local session="$1" producer="$3"
  # What the job itself writes, and nothing of gpu-dev's own bookkeeping: a
  # log gone silent is a stale age, and a finished job names its exit.
  local probe='[ -f .jammi.log ] || { echo "head=$(git rev-parse --short HEAD) | not launched"; exit 0; }
age=$(( $(date +%s) - $(stat -c %Y .jammi.log) ))
done=$(grep -o "CAMPAIGN JOB DONE exit=[0-9]*" .jammi.log | tail -1)
echo "head=$(git rev-parse --short HEAD) log-age=${age}s ${done:-running} | $(grep -v "^\s*$" .jammi.log | tail -1 | cut -c1-110)"'
  printf '%-12s %-11s %s\n' "$session" "$producer" "$(gpu_dev exec "$session" "$probe" 2>/dev/null | tail -1)"
}

down_one() { gpu_dev down "$1"; }

# ------------------------------------------------------------------ pull
pull_one() {
  local dest="$1" session="$2"
  gpu_dev pull "$session" "$OUT_ROOT/$session" >/dev/null || { echo "$session: pull failed"; return 1; }
  [ "$CAMPAIGN_DRY_RUN" = 1 ] && { echo "$session: pulled"; return 0; }
  mkdir -p "$dest"
  rm -rf "${dest:?}/$session"
  mv "$REPO_ROOT/.gpu-pull/$session" "$dest/$session" && echo "$session: pulled → $dest/$session"
}

# ------------------------------------------------------------------ merge
merge() {
  local roster="$1" dest="$2" producer="$3" to="$4"
  local legs="$dest/$producer/legs" session arch p _knobs sub
  mkdir -p "$legs"
  while read -r session arch p _knobs; do
    [ "$p" = "$producer" ] || continue
    for sub in $(producer_legs_dirs "$producer"); do
      [ -d "$dest/$session/$producer/$sub" ] || { echo "$session: no $producer/$sub legs pulled" >&2; continue; }
      cp -R "$dest/$session/$producer/$sub"/. "$legs"/
    done
  done < <(roster_lines "$roster")
  echo "merged $(find "$legs" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ') legs into $legs"
  local rel="${legs#"$REPO_ROOT"/}"
  [ "$rel" != "$legs" ] || die "merge: DEST must be inside the checkout, where the CI image can read it"
  local cmd=("$REPO_ROOT/ci/dev.sh" cargo run -q -p jammi-bench -- ladder "$producer" "$rel" --from torch --to "$to" --out "${rel%/legs}")
  if [ "$CAMPAIGN_DRY_RUN" = 1 ]; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
}

# ------------------------------------------------------------------ pod-job
pod_job() {
  local session="$1" producer="$2"; shift 2
  local script out_var kv
  script="$(producer_script "$producer")"
  out_var="$(producer_out_var "$producer")"
  for kv in "$@"; do
    [[ "$kv" == *=* ]] || die "pod-job: '$kv' is not KEY=VALUE"
    export "${kv?}"
  done
  export "$out_var=$REPO_ROOT/$OUT_ROOT/$session/$producer"
  export TORCH_VENV="$REPO_ROOT/.venv-torch-ref"
  echo "=== $(date -u +%FT%TZ) $session $producer at $(git -C "$REPO_ROOT" rev-parse HEAD)"
  run_step() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    [ "$CAMPAIGN_DRY_RUN" = 1 ] || "$@"
  }
  case "$producer" in
    train-run|train-step|encode)
      run_step python3 "$DIR/torch_venv.py" --provision || return 1
      run_step python3 "$DIR/torch_venv.py" --preflight || return 1
      run_step "$TORCH_VENV/bin/python3" "$DIR/checkpoint_files.py" --fetch "$MODEL_REPO" "$MODEL_DIR_ON_POD" || return 1
      export MODEL_DIR="$MODEL_DIR_ON_POD" ;;
  esac
  case "$producer" in
    # The producer derives its train pairs from the book's pinned sources;
    # the interpreter it derives them with imports the book.
    train-run)
      local book="$REPO_ROOT/.venv-cookbook"
      [ -x "$book/bin/python3" ] && "$book/bin/python3" -c "import jammi_cookbook" 2>/dev/null \
        || run_step bash -c "python3 -m venv '$book' && '$book/bin/pip' -q install -e '$REPO_ROOT/cookbook/book'" || return 1
      export FINETUNE_RUN_AB_PROVISION_PYTHON="$book/bin/python3" ;;
    encode) export ENCODE_AB_MODEL_DIR="$MODEL_DIR" ENCODE_AB_CUDA_ORDINAL="${ENCODE_AB_CUDA_ORDINAL:-0}" ;;
  esac
  local rc=0
  run_step bash "$DIR/$script" || rc=$?
  echo "CAMPAIGN JOB DONE exit=$rc"
  return "$rc"
}

# ------------------------------------------------------------------ verbs
[ $# -ge 1 ] || die "usage: rent|dispatch|status|pull|merge|down ROSTER … | pod-job SESSION PRODUCER [KEY=VALUE …]"
verb="$1"; shift
case "$verb" in
  rent)
    [ $# -eq 2 ] || die "rent ROSTER REF"
    pooled_arch_check "$1"
    mkdir -p "$LOG_DIR"
    per_pod "$1" rent_one "$2" ;;
  dispatch)
    [ $# -eq 2 ] || die "dispatch ROSTER SHA"
    [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die "dispatch: SHA must be a full 40-hex commit"
    per_pod "$1" dispatch_one "$2" ;;
  status)
    [ $# -eq 1 ] || die "status ROSTER"
    per_pod "$1" status_one | sort ;;
  pull)
    [ $# -eq 2 ] || die "pull ROSTER DEST"
    per_pod "$1" pull_one "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")" ;;
  merge)
    [ $# -eq 4 ] || die "merge ROSTER DEST PRODUCER TO"
    producer_script "$3" >/dev/null
    merge "$1" "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")" "$3" "$4" ;;
  down)
    [ $# -eq 1 ] || die "down ROSTER"
    per_pod "$1" down_one ;;
  pod-job)
    [ $# -ge 2 ] || die "pod-job SESSION PRODUCER [KEY=VALUE …]"
    pod_job "$@" ;;
  *) die "unknown verb '$verb'" ;;
esac
