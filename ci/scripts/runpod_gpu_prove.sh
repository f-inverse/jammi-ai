#!/usr/bin/env bash
#
# GPU prove-lane orchestrator: run the gated GPU suites on a real NVIDIA device.
#
# jammi has no GPU in its default CI (GPU tests only compile-check on CPU
# runners), so the recommended shape — a CUDA server served over gRPC — and the
# engine's CUDA kernels are never actually exercised on a GPU before release.
# This script closes that gap without a standing GPU runner: it rents an
# ephemeral RunPod GPU pod from the CUDA CI image, runs the gated suites on it,
# streams the output, and ALWAYS terminates the pod on exit.
#
# It reproduces #277 as a regression gate: the pod builds candle's kernels at the
# CI image's baked CUDA_COMPUTE_CAP and runs them on an A100 (sm_80, the arch
# floor). If the cap ever regresses above 8.0, the served path breaks here.
#
# Env (required):
#   RUNPOD_API_KEY   RunPod API key.
# Env (optional):
#   GIT_REPO         Public repo to clone on the pod (default: this repo via GITHUB_REPOSITORY).
#   GIT_REF          Branch/tag/sha to check out (default: GITHUB_SHA, else main).
#   POD_MAX_SECONDS  Hard cap on the remote run (default 3000).
#
# Exit code = the remote suite's exit code (0 = all GPU suites passed).
set -uo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (GitHub secret)}"
GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"
POD_MAX_SECONDS="${POD_MAX_SECONDS:-3000}"
IMAGE="ghcr.io/f-inverse/jammi-ai-ci-cuda:latest"

WORK="$(mktemp -d)"
SSH_KEY="$WORK/id_ed25519"
POD_ID=""
gql() { curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$1"; }
cleanup() {
  [ -n "$POD_ID" ] && gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${POD_ID}\\\"}) }\"}" >/dev/null 2>&1 && echo "::notice::terminated RunPod pod ${POD_ID}"
  rm -rf "$WORK"
}
trap cleanup EXIT

# Ephemeral SSH keypair — CI has no user key; the public half is injected into
# the pod via the PUBLIC_KEY env, the private half drives the session.
ssh-keygen -t ed25519 -N '' -f "$SSH_KEY" -q
PUBKEY="$(cat "${SSH_KEY}.pub")"

# The remote job: import the container's Dockerfile ENV (an SSH shell does not
# inherit it), clone the ref, and run both gated GPU suites at the image's baked
# compute cap. --test-threads=1 is the suites' mandated run mode.
REMOTE_JOB=$(cat <<'REMOTE'
set -uo pipefail
while IFS= read -r -d '' e; do export "$e"; done < /proc/1/environ
export CARGO_TERM_COLOR=never
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-<unset>}"; echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "$GIT_REF" "$GIT_REPO" jammi-ai 2>&1 | tail -1
cd jammi-ai
rc=0
echo "::group::served client/server GPU proof (jammi-server)"
cargo test -p jammi-server --features cuda,live-gpu-tests --test it grpc_embedding_gpu -- --nocapture --test-threads=1 || rc=$?
echo "::endgroup::"
echo "::group::engine-core GPU correctness (jammi-ai gpu_capability)"
cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1 || rc=$?
echo "::endgroup::"
echo "PROVE_EXIT=${rc}"
exit "$rc"
REMOTE
)

# Deploy with SUPPLY_CONSTRAINT failover: A100 (sm_80) is the ONLY arch that
# proves the #277 floor, so try its variants across both cloud tiers before
# giving up. Availability is intermittent — this is the arch-availability risk
# of a single provider, handled with a candidate list.
deploy() {
  local cloud="$1" gpu="$2"
  # Emit the GraphQL deploy payload as JSON on stdout; the caller POSTs it with
  # curl (RunPod's Cloudflare WAF rejects urllib's user-agent).
  python3 - "$cloud" "$gpu" "$IMAGE" "$PUBKEY" "$GIT_REPO" "$GIT_REF" <<'PY'
import json, sys
cloud, gpu, image, pub, repo, ref = sys.argv[1:7]
setup = ("yum install -y openssh-server openssh-clients >/dev/null 2>&1; ssh-keygen -A; "
         "mkdir -p /root/.ssh; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
         "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D")
inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "name": "jammi-gpu-prove",
       "imageName": image, "containerDiskInGb": 60, "volumeInGb": 0, "ports": "22/tcp",
       "dockerArgs": "bash -c '%s'" % setup,
       "env": [{"key": "PUBLIC_KEY", "value": pub}, {"key": "GIT_REPO", "value": repo}, {"key": "GIT_REF", "value": ref}]}
q = "mutation D($i: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$i){ id } }"
print(json.dumps({"query": q, "variables": {"i": inp}}))
PY
}

echo "=== provisioning an A100 (sm_80) with supply failover ==="
for combo in "SECURE|NVIDIA A100 80GB PCIe" "COMMUNITY|NVIDIA A100 80GB PCIe" "SECURE|NVIDIA A100-SXM4-80GB" "COMMUNITY|NVIDIA A100-SXM4-80GB"; do
  cloud="${combo%%|*}"; gpu="${combo##*|}"
  payload="$(deploy "$cloud" "$gpu")"
  resp="$(gql "$payload")"
  POD_ID="$(printf '%s' "$resp" | python3 -c 'import sys,json
d=json.load(sys.stdin)
if "errors" in d: print("", end=""); sys.exit()
print((d.get("data",{}).get("podFindAndDeployOnDemand") or {}).get("id",""), end="")')"
  if [ -n "$POD_ID" ]; then echo "deployed ${POD_ID} on ${cloud} / ${gpu}"; break; fi
  echo "  no capacity: ${cloud} / ${gpu}"
done
[ -z "$POD_ID" ] && { echo "::error::no A100 capacity on RunPod (SUPPLY_CONSTRAINT across all candidates); retry later"; exit 75; }

# Poll for the public SSH endpoint, then wait for sshd.
HOST=""; PORT=""
for _ in $(seq 1 60); do
  R="$(gql "{\"query\":\"query{ pod(input:{podId:\\\"${POD_ID}\\\"}){ runtime{ ports{ ip publicPort privatePort isIpPublic type } } } }\"}")"
  read -r HOST PORT < <(printf '%s' "$R" | python3 -c 'import sys,json
p=(json.load(sys.stdin).get("data",{}).get("pod") or {}).get("runtime") or {}
[print(x["ip"], x["publicPort"]) for x in (p.get("ports") or []) if x.get("privatePort")==22 and x.get("isIpPublic")]' | head -1)
  [ -n "${HOST:-}" ] && break; sleep 10
done
[ -z "${HOST:-}" ] && { echo "::error::pod ${POD_ID} never exposed SSH"; exit 1; }
SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i "$SSH_KEY")
for _ in $(seq 1 40); do ssh "${SSHO[@]}" -p "$PORT" "root@${HOST}" true 2>/dev/null && break; sleep 10; done

echo "=== running GPU prove suites (${HOST}:${PORT}) ==="
ssh "${SSHO[@]}" -p "$PORT" "root@${HOST}" "timeout ${POD_MAX_SECONDS} bash -s" <<REMOTE
export GIT_REPO="${GIT_REPO}" GIT_REF="${GIT_REF}"
${REMOTE_JOB}
REMOTE
rc=$?
echo "=== GPU prove suites exit=${rc} ==="
exit "$rc"
