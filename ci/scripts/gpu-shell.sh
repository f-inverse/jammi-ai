#!/usr/bin/env bash
# Interactive GPU debug shell on an ephemeral RunPod pod — real silicon for the
# open GPU issues (#277 arch, #305 JIT cache, #307 scheduler, #308 fp8, …).
#
# Usage: ci/scripts/gpu-shell.sh [a100|l40s|h100|a40|l4]   (default: a100)
# Env:
#   RUNPOD_API_KEY   RunPod API key (falls back to ~/.config/runpod/key).
#   RP_IMAGE         Pod image. Default: the CUDA CI image (full toolchain, for
#                    building/debugging). Set it to the shipped runtime image
#                    (e.g. nvidia/cuda:12.6.3-runtime-ubi8) to reproduce runtime
#                    behaviour such as the uid-65532 JIT-cache case in #305.
#
# Provisions the chosen arch (capacity + liveness failover), drops you into an
# SSH shell on the pod, and ALWAYS terminates the pod when you exit.
set -uo pipefail

ARCH="${1:-a100}"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "${HOME}/.config/runpod/key" 2>/dev/null || true)}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# arch → RunPod GPU-type candidates (SECURE then COMMUNITY). A100 is the #277
# floor (sm_80); l40s/l4 are Ada (sm_89, fp8 #308); h100 is Hopper (sm_90);
# a40 is Ampere-workstation (sm_86). (RunPod has no Tesla T4 — #306 is Ampere+.)
case "$ARCH" in
  a100) CAND=("SECURE|NVIDIA A100 80GB PCIe" "COMMUNITY|NVIDIA A100 80GB PCIe" "SECURE|NVIDIA A100-SXM4-80GB" "COMMUNITY|NVIDIA A100-SXM4-80GB") ;;
  l40s) CAND=("SECURE|NVIDIA L40S" "COMMUNITY|NVIDIA L40S") ;;
  h100) CAND=("SECURE|NVIDIA H100 80GB HBM3" "SECURE|NVIDIA H100 PCIe" "COMMUNITY|NVIDIA H100 80GB HBM3" "COMMUNITY|NVIDIA H100 PCIe") ;;
  a40)  CAND=("SECURE|NVIDIA A40" "COMMUNITY|NVIDIA A40") ;;
  l4)   CAND=("SECURE|NVIDIA L4" "COMMUNITY|NVIDIA L4") ;;
  *) echo "usage: $(basename "$0") [a100|l40s|h100|a40|l4]   (default: a100)"; exit 2 ;;
esac

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY or write it to ~/.config/runpod/key}"

rp_init
echo "=== provisioning ${ARCH} (image: ${RP_IMAGE}) ==="
rp_deploy_live "${CAND[@]}" || exit $?

echo "=== pod ${RP_POD_ID} up on ${RP_HOST}:${RP_PORT} — opening shell; the pod TERMINATES when you exit ==="
echo "    tip: load the build env inside with:"
echo "         while IFS= read -r -d '' e; do export \"\$e\"; done < /proc/1/environ"
ssh "${RP_SSHO[@]}" -t -p "$RP_PORT" "root@${RP_HOST}" || true
echo "=== shell closed — terminating pod (trap) ==="
