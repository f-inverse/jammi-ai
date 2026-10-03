#!/usr/bin/env bash
# The cookbook on a GPU, over a tree's own builds: rents an L4 (sm_89, the class
# of GPU a Colab reader gets; an RTX 4090 when no L4 is free), copies to it the
# CUDA engine wheel, the client wheel, the CUDA server and the CLI a `ci.yml`
# run built for the tree, checks out the tree's commit, runs every recipe and
# renders every page of the book under the `assemble` profile, and copies the
# pages' recorded execution (`_freeze`) back for `quarto render --profile
# assemble` to assemble into the book a release publishes. Every recipe and
# chapter runs real Hub models, which is why this runs on a rented GPU rather
# than in `ci.yml`. The deploy, run and teardown are runpod_lib.sh's, as for
# every other pod lane.
#
# Invoked by cookbook-gpu.yml.
#
# Requires env:
#   RUNPOD_API_KEY
#   PROVE_EXPECT_SHA  the commit to check out on the pod (the tree measured)
#   ARTIFACTS         a directory holding native/ (the CUDA engine wheel),
#                     client/ (the client wheel), server/ (`jammi-server`) and
#                     cli/ (the `jammi` release tarball)
#   FREEZE_OUT        where the pages' `_freeze` directory is copied back to
# Optional env: GIT_REPO, RP_TTL_HOURS (default 4).
# Exit 0 = every recipe ran and every page rendered with its claims holding;
# non-zero = a recipe or a page failed (the log names each); 75 = no capacity.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RP_TTL_HOURS="${RP_TTL_HOURS:-4}"
# One invocation runs every recipe and page in turn; the pod's own deadline is
# the cost bound, this only bounds the ssh session inside it.
RP_TIMEOUT="${RP_TIMEOUT:-$(( (RP_TTL_HOURS - 1) * 3600 ))}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/f-inverse/jammi-ai.git}"
: "${PROVE_EXPECT_SHA:?PROVE_EXPECT_SHA must name the commit to check out}"
: "${ARTIFACTS:?ARTIFACTS must name the directory holding the tree's builds}"
: "${FREEZE_OUT:?FREEZE_OUT must name where the pages' _freeze is copied back to}"
for part in native client server cli; do
  [ -d "$ARTIFACTS/$part" ] || { echo "::error::ARTIFACTS has no $part/ directory (${ARTIFACTS})" >&2; exit 2; }
done

rp_sweep
rp_init
echo "=== provisioning an L4 for the cookbook at ${PROVE_EXPECT_SHA} ==="
rp_deploy_arch l4 || exit $?

scp -q -r "${RP_SSHO[@]}" -P "$RP_PORT" "$ARTIFACTS" "root@${RP_HOST}:/root/artifacts" \
  || { echo "::error::could not copy the tree's builds to the pod" >&2; exit 1; }

{
  rp_remote_checkout_lines "$PROVE_EXPECT_SHA" "$GIT_REPO"
  cat <<'REMOTE'
set -uo pipefail
# A run that cannot reach the GPU fails instead of falling back to the CPU.
export JAMMI_GPU__REQUIRE_GPU=true
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "::endgroup::"

echo "::group::install the tree's builds"
A=/root/artifacts
mkdir -p /root/bin
install -m 0755 "$A"/server/jammi-server /root/bin/jammi-server || exit 1
tar -xzf "$A"/cli/*.tar.gz -C /root/bin || exit 1
export PATH=/root/bin:$PATH JAMMI_CLI_BIN=/root/bin/jammi
python3.12 -m venv /root/venv && . /root/venv/bin/activate || exit 1
pip install -q --disable-pip-version-check "$A"/client/*.whl "$A"/native/*.whl || exit 1
pip install -q --disable-pip-version-check -e 'cookbook/book[book,dev,cloud]' || exit 1
jammi --version && jammi-server --version || exit 1
echo "::endgroup::"

failed=0
echo "::group::recipes"
python tests/cookbook_smoke.py || failed=1
echo "::endgroup::"

cd cookbook/book || exit 1
pages="$(python - <<'PY'
import yaml
book = yaml.safe_load(open("_quarto.yml"))["book"]
pages = []
def walk(entries):
    for entry in entries:
        if isinstance(entry, str):
            pages.append(entry)
        elif isinstance(entry, dict):
            walk(entry.get("chapters", []))
walk(book["chapters"])
walk(book.get("appendices", []))
print("\n".join(pages))
PY
)" || exit 1
for page in $pages; do
  start=$(date +%s)
  if python -m jammi.session_journal -- quarto render "$page" --profile assemble >"/root/render.log" 2>&1; then
    echo "ok   $(( $(date +%s) - start ))s $page"
  else
    failed=1
    echo "FAIL $(( $(date +%s) - start ))s $page"
    sed 's/\x1b\[[0-9;]*m//g' /root/render.log | grep -a -E "does not hold|Error" | tail -5
  fi
done
exit "$failed"
REMOTE
} | rp_run_remote
rc=$?

if [ "$rc" -eq 0 ]; then
  mkdir -p "$(dirname "$FREEZE_OUT")"
  scp -q -r "${RP_SSHO[@]}" -P "$RP_PORT" "root@${RP_HOST}:/root/jammi-ai/cookbook/book/_freeze" "$FREEZE_OUT" \
    || { echo "::error::could not copy the pages' recorded execution back from the pod" >&2; rc=1; }
fi
echo "=== cookbook on an L4 at ${PROVE_EXPECT_SHA}: exit=${rc} ==="
exit "$rc"
