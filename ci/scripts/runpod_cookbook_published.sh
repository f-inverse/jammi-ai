#!/usr/bin/env bash
# The published cookbook notebooks on a GPU: rents an L4 (sm_89, the class of
# GPU a Colab reader gets; an RTX 4090 when no L4 is free) and runs
# `cookbook/book/scripts/run_published_notebooks.py` there over the notebooks
# at NOTEBOOKS_TAG, so each setup cell installs the CUDA wheels from PyPI. The
# deploy, run and teardown are runpod_lib.sh's, as for every other pod lane.
#
# Invoked by cookbook-published-gpu.yml, dispatched after a release.
#
# Requires env: RUNPOD_API_KEY, NOTEBOOKS_TAG (py-vX.Y.Z).
# Optional env: GIT_REPO, DRIVER_REF (the commit whose driver runs; default
# main), RP_TTL_HOURS (default 5).
# Exit 0 = every notebook ran; non-zero = a notebook failed (the log names
# each); 75 = no capacity.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-5}"
# One invocation runs every notebook in turn; the pod's own deadline is the
# cost bound, this only bounds the ssh session inside it.
RP_TIMEOUT="${RP_TIMEOUT:-$(( (RP_TTL_HOURS - 1) * 3600 ))}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/f-inverse/jammi-ai.git}"
DRIVER_REF="${DRIVER_REF:-main}"
NOTEBOOKS_TAG="${NOTEBOOKS_TAG:-}"
case "$NOTEBOOKS_TAG" in
  py-v[0-9]*) ;;
  *) echo "::error::NOTEBOOKS_TAG must name a py-vX.Y.Z release tag (got '${NOTEBOOKS_TAG}')." >&2; exit 2 ;;
esac

rp_sweep
rp_init
echo "=== provisioning an L4 for the notebooks at ${NOTEBOOKS_TAG} ==="
rp_deploy_arch l4 || exit $?

rp_run_remote <<REMOTE
set -uo pipefail
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"
cd /root && rm -rf driver published
git init -q driver && git -C driver fetch -q --depth 1 "${GIT_REPO}" "${DRIVER_REF}" && git -C driver checkout -q FETCH_HEAD
git clone -q --depth 1 --branch "${NOTEBOOKS_TAG}" --filter=blob:none --sparse "${GIT_REPO}" published
git -C published sparse-checkout set cookbook/notebooks
python3.12 driver/cookbook/book/scripts/run_published_notebooks.py \
  --notebooks published/cookbook/notebooks --work /root/notebooks
REMOTE
rc=$?
echo "=== notebooks at ${NOTEBOOKS_TAG} on an L4: exit=${rc} ==="
exit "$rc"
