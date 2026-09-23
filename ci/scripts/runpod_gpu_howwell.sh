#!/usr/bin/env bash
# how-well GPU driver: rents a real A100 (sm_80), clones the
# checkout at GIT_REF onto it, and runs `ci/scripts/perf/finetune_run_ab.sh`
# remotely — the producer of the `train-run` ladder's torch edge, executed
# against a live checkpoint. Shared deploy/run/teardown
# lives in runpod_lib.sh, the same machinery `runpod_gpu_prove.sh` already
# uses (rp_sweep/rp_init/rp_deploy_live_a100/rp_run_remote) — this script is
# that one's how-well sibling, not a second orchestration mechanism.
#
# Off the merge path (see gpu-howwell.yml's own doc): this driver is invoked
# ONLY by that workflow's workflow_dispatch / `run-howwell` PR-label triggers,
# never a schedule.
#
# Exit 0 = the ladder's status was GREEN; non-zero = status is RED,
# RED_FOR_INVESTIGATION, or INVALID (the pre-registered decision rule fired,
# or a correctness-of-measurement problem was found — see the "ladder status"
# log line below for WHICH one and its named causes, rather than a bare exit
# code an operator has to cross-reference against the pulled artifact);
# 75 = no A100 capacity (neutral skip, the SAME convention runpod_gpu_prove.sh
# uses).
#
# Env vars (forwarded verbatim into the remote leg — see
# finetune_run_ab.sh's own doc for each one's meaning/default):
#   GIT_REPO / GIT_REF          what to clone (defaults mirror
#                                runpod_gpu_prove.sh's own).
#   HOWWELL_MODEL_DIR            REQUIRED — the pod's own provisioned
#                                checkpoint directory (ModernBERT-large
#                                primary — this driver does not provision
#                                the checkpoint itself, only forwards the
#                                path an operator/pod-seed step already
#                                placed there).
#   HOWWELL_SEEDS                forwarded as FINETUNE_RUN_AB_SEEDS
#                                (default: the pre-registered 12-seed gate
#                                set, finetune_run_ab.sh's own default).
#   HOWWELL_OBJECTIVE             forwarded as FINETUNE_RUN_AB_OBJECTIVE.
#                                REQUIRED -- no default (the choice must be
#                                made deliberately on every dispatch); this
#                                script refuses loudly if unset.
#   HOWWELL_LR0_SEEDS             forwarded as FINETUNE_RUN_AB_LR0_SEEDS
#                                (default: empty — the lr=0 RED control is
#                                opt-in).
#   HOWWELL_ARTIFACT_DIR          where the ladder verdict/table is pulled
#                                back to once the remote run finishes
#                                (default: "<repo>/.gpu-pull/how-well" —
#                                this mirrors gpu-dev.sh's own `pull` subcommand's rsync
#                                invocation rather than inventing a second
#                                retrieval mechanism. `.gpu-pull/` is
#                                already gitignored — a human commits the
#                                specific run(s) that matter as evidence,
#                                this driver
#                                only makes them retrievable).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

HOWWELL_MODEL_DIR="${HOWWELL_MODEL_DIR:-}"
if [ -z "$HOWWELL_MODEL_DIR" ]; then
  echo "::error::HOWWELL_MODEL_DIR must name the pod's own provisioned checkpoint directory." >&2
  exit 2
fi
HOWWELL_SEEDS="${HOWWELL_SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12}"
# No silent default -- gpu-howwell.yml's own workflow_dispatch `objective`
# input is REQUIRED for exactly this reason, refusing before this script is
# even invoked; a `mnrl` fallback here would only be reachable via a DIRECT
# invocation of this script that bypasses that workflow, silently choosing
# an objective on the caller's behalf rather than making the choice
# deliberate on every dispatch).
HOWWELL_OBJECTIVE="${HOWWELL_OBJECTIVE:-}"
if [ -z "$HOWWELL_OBJECTIVE" ]; then
  echo "::error::HOWWELL_OBJECTIVE must be set to 'mnrl' or 'triplet' -- no default (the objective is chosen deliberately on every dispatch); gpu-howwell.yml's workflow_dispatch 'objective' input already enforces this on the merge path, but a direct invocation of this script must refuse just as loudly." >&2
  exit 2
fi
HOWWELL_LR0_SEEDS="${HOWWELL_LR0_SEEDS:-}"
HOWWELL_ARTIFACT_DIR="${HOWWELL_ARTIFACT_DIR:-${REPO_ROOT}/.gpu-pull/how-well}"

# Sweep before renting anything — same orphan-bounding reasoning
# runpod_gpu_prove.sh's own header states (this workflow ALSO sets
# cancel-in-progress: false at the job level, so a cancelled run never
# orphans a pod, but the sweep stays cheap insurance regardless).
rp_sweep

rp_init
echo "=== provisioning a live A100 (sm_80) for the how-well producer ==="
rp_deploy_live_a100 || exit $?

echo "=== running finetune_run_ab.sh on ${RP_HOST}:${RP_PORT} ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
# the depth-1 clone above carries no submodule content; jammi-kernels/build.rs
# panics loudly ("CUTLASS submodule is not checked out") the moment a
# jammi-encoders/flash-attn build reaches it, so init it explicitly before
# any build step runs.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass

# --- python provisioning: a bare pod has no pip
# on PATH and finetune_run_ab.sh's own cargo build/jammi-bench run never
# needs python beyond the stdlib (verify_train_pairs.py) --
# the ONE exception is that script's own PRE-RUN provisioning step
# (\`derive_heldout_fixture.py --emit-train-pairs\`), which imports
# jammi_cookbook + numpy and pulls pyarrow/requests transitively
# through \`jammi_cookbook.datasets\`. This venv exists FOR THAT ONE STEP
# ONLY -- every measured leg, and the jammi-bench build/binary itself, stay
# venv-free (system python3, when they touch python at all). Fails loudly
# (never silently skips provisioning) if python3/venv is unavailable.
echo "::group::python provisioning (jammi_cookbook/numpy/pyarrow/requests -- the finetune_run_ab.sh PRE-RUN provisioning step ONLY, never the measured legs)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "::error::python3 not found on this pod -- refusing (the PRE-RUN provisioning step cannot run without it)." >&2
  exit 1
fi
if ! python3 -m venv /root/howwell-venv; then
  echo "::error::'python3 -m venv /root/howwell-venv' failed -- refusing (python3-venv unavailable on this pod)." >&2
  exit 1
fi
# Only what derive_heldout_fixture.py --emit-train-pairs's own import chain
# actually touches (jammi_cookbook/__init__.py -> contracts/determinism/
# rails, stdlib only, + datasets -> pyarrow/requests; the script itself
# imports numpy directly) -- version pins read verbatim off cookbook/book/
# pyproject.toml's own \`dependencies\` list. \`--no-deps\` on jammi_cookbook
# itself so this venv never pulls in that pyproject's jammi-ai/usearch/
# numkong entries -- the FULL book's own heavier deps (a maturin-built
# engine wheel, a pinned ANN backend), unneeded for this one script and
# requiring toolchains this bare pod does not carry.
/root/howwell-venv/bin/pip install --quiet --upgrade pip \
  || { echo "::error::'pip install --upgrade pip' failed in /root/howwell-venv -- refusing." >&2; exit 1; }
/root/howwell-venv/bin/pip install --quiet 'numpy>=1.26,<3' 'pyarrow>=15' 'requests>=2.31' \
  || { echo "::error::provisioning numpy/pyarrow/requests into /root/howwell-venv failed -- refusing." >&2; exit 1; }
/root/howwell-venv/bin/pip install --quiet --no-deps -e cookbook/book \
  || { echo "::error::'pip install --no-deps -e cookbook/book' into /root/howwell-venv failed -- refusing." >&2; exit 1; }
echo "::endgroup::"

echo "::group::torch venv (torch_venv.py --provision)"
python3 ci/scripts/perf/torch_venv.py --provision \
  || { echo "::error::the torch venv could not be provisioned on this pod -- refusing before any leg." >&2; exit 1; }
echo "::endgroup::"

echo "::group::train-run, torch edge (finetune_run_ab.sh)"
MODEL_DIR="${HOWWELL_MODEL_DIR}" \
  FINETUNE_RUN_AB_SEEDS="${HOWWELL_SEEDS}" \
  FINETUNE_RUN_AB_OBJECTIVE="${HOWWELL_OBJECTIVE}" \
  FINETUNE_RUN_AB_LR0_SEEDS="${HOWWELL_LR0_SEEDS}" \
  FINETUNE_RUN_AB_LORA_DROPOUT=0 \
  FINETUNE_RUN_AB_PROVISION_PYTHON=/root/howwell-venv/bin/python3 \
  bash ci/scripts/perf/finetune_run_ab.sh
rc=\$?
echo "::endgroup::"
echo "HOWWELL_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== how-well A/B exit=${rc} ==="

# --- artifact retrieval: the artifact
# never otherwise leaves the pod — this driver is the ONE place still able to
# reach it, since the EXIT trap (rp_cleanup, installed by rp_init) tears the
# pod down once THIS script itself exits). Mirrors gpu-dev.sh's own `pull`
# subcommand's rsync invocation, EXCEPT for the `work/` exclusion below —
# never a second, independently-drifting retrieval
# mechanism otherwise. Best-effort and unconditional (pulled regardless of
# ${rc} — a RED/RED_FOR_INVESTIGATION/INVALID run's own artifact is exactly
# the evidence to keep, not less so than a GREEN one's).
#
# `finetune_run_ab.sh`'s own `$OUT_DIR` layout is `ladder_verdict.json` +
# `ladder_table.txt` (the ladder's decision — the ACTUAL payload), `raw/`
# (one small leg `.json` plus its `.exit`/`.stderr` per leg — useful
# debugging context, cheap), and `work/` (one `--work-dir` per leg, 12 seeds
# x 2 rungs x 2 repeats = 48+ dirs, EACH holding that leg's own LoRA
# checkpoint/optimizer state — multi-GB bulk). `--exclude 'work/'` keeps the
# pull to the two decision files plus `raw/`; a human who needs a specific
# leg's own checkpoint can re-run that one leg.
mkdir -p "$HOWWELL_ARTIFACT_DIR"
if [ -n "${RP_HOST:-}" ] && [ -n "${RP_PORT:-}" ]; then
  rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    --exclude 'work/' \
    "root@${RP_HOST}:/root/jammi-ai/.finetune-run-ab-report/" "${HOWWELL_ARTIFACT_DIR}/" \
    && echo "=== pulled how-well artifact (ladder verdict + table + raw/ legs, work/ excluded) -> ${HOWWELL_ARTIFACT_DIR} ===" \
    || echo "::warning::how-well artifact pull failed -- ${rc} above is still authoritative; the pod is torn down on this script's own exit, so this evidence is now unrecoverable for this invocation."
else
  echo "::warning::no live pod (RP_HOST/RP_PORT unset) -- skipping artifact pull."
fi

# --- surface the ladder's status by NAME, with every cause it names: the
# verdict's `status` already folds the torch edge, every refusal and the
# mutant columns, and `causes` says which. Defensive: if the remote's own exit
# code somehow read 0 despite a non-GREEN status, force non-zero here rather
# than let a mismatch pass silently.
VERDICT_JSON="$(find "$HOWWELL_ARTIFACT_DIR" -name ladder_verdict.json 2>/dev/null | sort | tail -1)"
if [ -n "$VERDICT_JSON" ] && [ -f "$VERDICT_JSON" ]; then
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$VERDICT_JSON" 2>/dev/null || echo "UNKNOWN")"
  CAUSES="$(python3 -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["causes"]))' "$VERDICT_JSON" 2>/dev/null || echo "unknown (could not read ${VERDICT_JSON})")"
  echo "=== ladder status: ${STATUS} (${VERDICT_JSON}) ==="
  if [ "$STATUS" != "GREEN" ]; then
    echo "::error::how-well status=${STATUS} -- ${CAUSES}"
    if [ "$rc" -eq 0 ]; then
      echo "::error::remote exit was 0 but the ladder's status=${STATUS} is non-GREEN -- forcing a non-zero exit."
      rc=1
    fi
  elif [ "$rc" -ne 0 ]; then
    echo "::error::how-well status=GREEN but exit=${rc} -- the failure is outside the ladder (a build, a provisioning step, or the pod)."
  fi
else
  echo "::warning::no ladder_verdict.json found under ${HOWWELL_ARTIFACT_DIR} -- cannot name the status."
fi

exit "$rc"
