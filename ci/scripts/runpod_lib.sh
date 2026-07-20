#!/usr/bin/env bash
# Shared RunPod GPU primitive — one seam for every caller.
#
# Deploys a live A100 (sm_80) with two failover dimensions (capacity +
# liveness), runs a remote job on it importing the container's real ENV, and
# ALWAYS terminates the pod on exit. Sourced by:
#   * runpod_gpu_prove.sh          — build-from-source + run the gated GPU suites
#   * runpod_gpu_smoke_artifact.sh — run a prebuilt CUDA artifact on the device
#
# Requires env: RUNPOD_API_KEY.
# Optional env: RP_IMAGE (pod image), RP_TIMEOUT (remote timeout s, default 3000).
# Sets globals: RP_POD_ID, RP_HOST, RP_PORT. Installs an EXIT trap for teardown.

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (GitHub secret)}"
RP_IMAGE="${RP_IMAGE:-ghcr.io/f-inverse/jammi-ai-ci-cuda:latest}"
RP_WORK="$(mktemp -d)"
RP_SSH_KEY="$RP_WORK/id_ed25519"
RP_POD_ID=""; RP_HOST=""; RP_PORT=""; RP_PUBKEY=""; RP_SSHO=()

# An SSH login shell does NOT inherit the container's Dockerfile ENV (CC=gcc-13,
# PATH with cuda+mold+rust). Every remote job imports PID 1's real environment.
RP_ENV_PREAMBLE='while IFS= read -r -d "" __e; do export "$__e"; done < /proc/1/environ'

rp_gql() { curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$1"; }

rp_cleanup() {
  if [ -n "$RP_POD_ID" ]; then
    rp_gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${RP_POD_ID}\\\"}) }\"}" >/dev/null 2>&1
    echo "::notice::terminated RunPod pod ${RP_POD_ID}"
  fi
  rm -rf "$RP_WORK"
}
trap rp_cleanup EXIT

# Generate the ephemeral SSH keypair used to reach the pod.
rp_init() {
  ssh-keygen -t ed25519 -N '' -f "$RP_SSH_KEY" -q
  RP_PUBKEY="$(cat "${RP_SSH_KEY}.pub")"
  RP_SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i "$RP_SSH_KEY")
}

_rp_deploy_payload() { # $1=cloudType $2=gpuTypeId
  python3 - "$1" "$2" "$RP_IMAGE" "$RP_PUBKEY" <<'PY'
import json, sys
cloud, gpu, image, pub = sys.argv[1:5]
setup = ("yum install -y openssh-server openssh-clients >/dev/null 2>&1; ssh-keygen -A; "
         "mkdir -p /root/.ssh; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
         "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D")
inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "name": "jammi-gpu",
       "imageName": image, "containerDiskInGb": 60, "volumeInGb": 0, "ports": "22/tcp",
       "dockerArgs": "bash -c '%s'" % setup, "env": [{"key": "PUBLIC_KEY", "value": pub}]}
print(json.dumps({"query": "mutation D($i: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$i){ id } }",
                  "variables": {"i": inp}}))
PY
}

# Deploy a live GPU pod, failing over across a candidate list of "CLOUD|GPU_TYPE"
# args (capacity) and terminating any pod that never becomes reachable
# (liveness). On success sets RP_POD_ID/RP_HOST/RP_PORT and returns 0. Returns 75
# (neutral skip) when no candidate yields a reachable pod — a provider condition,
# not a code failure.
rp_deploy_live() {
  local supply_seen=0 combo cloud gpu R
  for combo in "$@"; do
    cloud="${combo%%|*}"; gpu="${combo##*|}"
    RP_POD_ID="$(rp_gql "$(_rp_deploy_payload "$cloud" "$gpu")" | python3 -c 'import sys,json
d=json.load(sys.stdin)
print("" if "errors" in d else (d.get("data",{}).get("podFindAndDeployOnDemand") or {}).get("id",""), end="")')"
    if [ -z "$RP_POD_ID" ]; then echo "  no capacity: ${cloud} / ${gpu}"; continue; fi
    supply_seen=1
    echo "  deployed ${RP_POD_ID} on ${cloud} / ${gpu}; waiting for SSH (≤4m)..."
    RP_HOST=""; RP_PORT=""
    for _ in $(seq 1 24); do
      R="$(rp_gql "{\"query\":\"query{ pod(input:{podId:\\\"${RP_POD_ID}\\\"}){ runtime{ ports{ ip publicPort privatePort isIpPublic type } } } }\"}")"
      read -r RP_HOST RP_PORT < <(printf '%s' "$R" | python3 -c 'import sys,json
p=(json.load(sys.stdin).get("data",{}).get("pod") or {}).get("runtime") or {}
[print(x["ip"], x["publicPort"]) for x in (p.get("ports") or []) if x.get("privatePort")==22 and x.get("isIpPublic")]' | head -1)
      if [ -n "${RP_HOST:-}" ] && ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" true 2>/dev/null; then
        echo "  SSH up on ${RP_HOST}:${RP_PORT}"; return 0
      fi
      RP_HOST=""; sleep 10
    done
    echo "  pod ${RP_POD_ID} never became reachable; terminating and trying next candidate"
    rp_gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"${RP_POD_ID}\\\"}) }\"}" >/dev/null 2>&1
    RP_POD_ID=""
  done
  if [ "$supply_seen" = "0" ]; then echo "::error::no GPU capacity on RunPod for the requested candidates (SUPPLY_CONSTRAINT); retry later"
  else echo "::error::GPU pod(s) deployed but none became reachable over SSH; retry later"; fi
  return 75
}

# A100 (sm_80) — the arch that proves the compute_80 floor / #277. Wrapper over
# rp_deploy_live with the A100 candidate list (PCIe + SXM, both cloud tiers).
rp_deploy_live_a100() {
  rp_deploy_live \
    "SECURE|NVIDIA A100 80GB PCIe" "COMMUNITY|NVIDIA A100 80GB PCIe" \
    "SECURE|NVIDIA A100-SXM4-80GB" "COMMUNITY|NVIDIA A100-SXM4-80GB"
}

# Run a bash script (read from stdin) on the pod, with the container ENV imported
# first and a hard timeout. Returns the remote script's exit code.
rp_run_remote() {
  { printf '%s\n' "$RP_ENV_PREAMBLE"; cat; } \
    | ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" "timeout ${RP_TIMEOUT:-3000} bash -s"
}
