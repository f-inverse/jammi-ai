#!/usr/bin/env bash
# GPU cluster leg: two RunPod PODS on one RunPod CLUSTER (REST v2), one A100
# each, joined over the cluster's own private overlay network — the only
# place a real cross-HOST NCCL gang is exercised before release. The pod leg
# (`runpod_gpu_gang.sh`) proves a two-DEVICE gang inside one pod; this driver
# proves the two-HOST bootstrap (`ncclCommInitRank`, an out-of-band id file)
# that pod leg cannot reach at all. Never `runpod_gpu_gang.sh` itself — a
# SEPARATE driver, a SEPARATE workflow, a SEPARATE RunPod object type (a
# cluster is retired by deleting the CLUSTER, never a member pod — see
# runpod_lib.sh's own `rp_cluster_delete`/`rp_cluster_sweep`).
#
# WHAT IT RENTS: one RunPod CLUSTER, `podCount: 2`, `gpuCountPerPod: 1`, the
# `NVIDIA A100-SXM4-80GB` SECURE candidate (`RP_CLUSTER_GPU_TYPE` below) —
# the one shape S4 measured co-placed on ONE data center. `dataCenterIds` is
# never left to the scheduler: this driver reads per-data-center
# availability itself (`GET /v2/catalog/gpus?include=AVAILABILITY&
# product=CLUSTER&count=1&cloud=SECURE`) and passes only the data
# center(s) at `RP_CLUSTER_MIN_AVAILABILITY` (MEDIUM) or better — co-
# placement needs ONE data center, and the overall (account-wide) figure
# alone does not establish that any single one actually has it (A1).
#
# COST BOUND (human-approved, S4's measured $1.908/GPU/h for a SECURE
# cluster GPU — the catalog's own $1.59 is the POD price, a different rate):
#
#   2 GPUs x $1.908/GPU/h = $3.816/h.
#
#   (i) terminate-succeeds (the ordinary path): ONE create (no candidate
#       walk — a cluster create names its data center directly, unlike a
#       pod's failover search), billing to RP_TTL_HOURS=1:
#         1 h x $3.816/h = $3.82 per run.
#   (ii) sweep-only (the worst path — the EXIT trap's own `rp_cluster_delete`
#       call fails, AND member self-removal is UNMEASURED — see
#       runpod_lib.sh's own cluster-primitives header): the cluster bills to
#       its own TTL, then `gpu-reap.yml`'s 6-hourly `rp_cluster_sweep` is the
#       backstop:
#         (1 + 6) h x $3.816/h = $26.71.
#
# `ci/scripts/test_gpu_cluster_lane.sh` re-derives both figures from this
# script's own RP_TTL_HOURS/the $1.908 rate/the 2-GPU shape and fails if the
# printed figure and the mechanism disagree. `≤ 1 h billed, ≤ 2 runs` is the
# standing spend authorization (2026-09-13) this lane holds to; `MAX_ATTEMPTS`
# lives in `.github/workflows/gpu-cluster.yml`, exactly the gang lane's own
# shape (this driver attempts ONE create per invocation; the workflow's own
# retry loop is capacity-only, bounded, and never doubles the billed TTL).
#
# WHAT IT PROVES: `gang_nccl_two_hosts_reduce_a_known_vector`
# (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`) — the two-HOST NCCL
# leg (`ncclCommInitRank`, not the pod leg's single-process
# `ncclCommInitAll`). Test env contract (see that module's own doc):
# `JAMMI_GANG_TWO_HOSTS_RANK`/`_WORLD`/`_ID_FILE`, `JAMMI_GANG_ARTIFACT_DIR`,
# `JAMMI_REQUIRE_CUDA_TWO_HOSTS`. `NCCL_SOCKET_IFNAME=ens1`/`NCCL_DEBUG=INFO`
# are exported by THIS driver, never read by the test itself. The build+run
# text is ONE shared function (`_rpc_remote_script`), expanded once per rank
# — `world`/the member count derive from `RP_CLUSTER_POD_COUNT`
# (the create payload's own `podCount`), never a second, independently
# duplicated literal.
#
# THE ID CROSSING: rank 0 mints the 128-byte NCCL id and writes it,
# atomically, to its own host's id file; this driver polls that file (via
# `stat` over ssh) until it reports EXACTLY 128 bytes, `scp`s it down to a
# LOCAL staging copy (`$RP_WORK/nccl.id`, mode 0600), then `scp`s that
# staging copy up to the member host — ONLY THEN does rank 1 start. The id
# never rides inside `JAMMI_GANG_ARTIFACT_DIR` and never reaches this
# driver's own stdout/log in the clear; `ci/scripts/gang_id_secrecy_scan.py`
# (F5) scans every carrier this run produces (the pulled artifact dir, the
# run log, the assembled artifact, the staging copy's own directory
# listing) for the id in every encoding the ship step could emit (raw, hex
# either case, base64) before the staging copy is deleted.
#
# THE ARTIFACT: one `gang` artifact (`gang.leg = "cluster"`, `producer.path
# = "ci/scripts/runpod_gpu_cluster.sh"` — THIS driver names itself as the
# sole writer, bound by `check_cuda_run_artifacts.py`'s own
# GANG_LEG_PRODUCER_PATH so the leg cannot be misdeclared to dodge the pod
# leg's registry) assembled by THIS driver (the two ranks only report) from
# both `rank-<r>.json` reports, schema-gated by
# `ci/scripts/check_cuda_run_artifacts.py`'s rule (k) cluster-leg rows
# (`world`, `collective`, `hosts`, `ranks[]` — each carrying its OWN
# `reduced_vector_digest`, kept per rank and asserted equal on `pass`,
# never collapsed here — `verdict`, `pod_count`, `gpu_count_per_pod`,
# `ttl_hours`). Assembly itself REFUSES (named, never an artifact) when
# either rank's own `hostname`/`nccl_socket_ifname` is empty or `unknown`,
# or the two ranks report the SAME host. A human reviews the pulled
# artifact and commits it under `crates/jammi-kernels/artifacts/cuda-runs/`.
#
# EXIT CONTRACT: 0 pass; 75 no cluster capacity (no data center at
# RP_CLUSTER_MIN_AVAILABILITY or better — a neutral provider condition); 76
# inactivity kill (a hung leg, watched across BOTH ranks' output); 77 wrong
# tree (a rank's own echoed PROVE_SHA disagreed with PROVE_EXPECT_SHA); 97
# wrong shape (a member's launch-time read-back failed: `Pod.args` does not
# echo the shared entrypoint text, or NEITHER `ssh.direct` nor an overlay-ip
# proxy path is reachable for a member — see F2/F3); 124 budget cut (T-10m,
# with the per-phase wall-clock breakdown printed); else this driver's own
# post-run refusal, by name (a failed artifact pull, a failed id-secrecy
# scan, a missing `ens1` line in a rank's log).
#
# TRIGGERS: `.github/workflows/gpu-cluster.yml` only — the `run-cluster` PR
# label and manual dispatch, deliberately no `push:`/`workflow_call:`/
# `schedule:` trigger, and nothing may `uses:` it
# (`ci/scripts/check_gpu_prove_once.py`'s P7/P8 rules pin this by name).
# Needs the `RUNPOD_API_KEY` repo secret.
#
# ALL of this driver's own stdout+stderr goes through ONE `tee`'d run log
# (F6) written INSIDE the pulled/uploaded directory itself
# (`${CLUSTER_ARTIFACT_DIR}/run.log`, never a bare `mktemp` elsewhere) — the
# SAME file the id-secrecy scan's own artifact-dir walk treats as a carrier
# and the workflow's `actions/upload-artifact` step uploads (`path:
# .gpu-pull/gpu-cluster/`), so the uploaded log IS a scanned carrier, never
# a second, unscanned copy of one. Both ranks' own remote logs
# (`rank0.log`/`rank1.log`) are copied into the same directory after both
# ranks finish, pass or fail, for the same reason.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RP_TTL_HOURS="${RP_TTL_HOURS:-1}"
export RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-300}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# The one gpuTypeId this lane rents, and the availability floor a data
# center must clear to be offered to the create call (A1): the ACCOUNT-WIDE
# figure alone never establishes that any SINGLE data center can co-place
# both members, so this driver always reads the per-data-center breakdown
# itself rather than leaving `dataCenterIds` to the scheduler.
RP_CLUSTER_GPU_TYPE="${RP_CLUSTER_GPU_TYPE:-NVIDIA A100-SXM4-80GB}"
RP_CLUSTER_MIN_AVAILABILITY="${RP_CLUSTER_MIN_AVAILABILITY:-MEDIUM}"

# The shape this lane rents — a 2x1 cluster. Both members derive `world`
# from this, never a second literal.
RP_CLUSTER_POD_COUNT=2
RP_CLUSTER_GPU_COUNT_PER_POD=1

# sm_80 (A100) is this leg's device, the same floor the pod leg proves.
NATIVE_COMPUTE_CAP=80

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

# The ONE place the two-host test's own name lives — a rename moves this
# line and nothing else. `test_gpu_cluster_lane.sh` re-derives the exact
# `cargo test` tuple from this value.
CLUSTER_TEST_FILTER="${CLUSTER_TEST_FILTER:-gang_nccl_two_hosts}"

# Where the two-host test writes its per-rank report on EACH member, and
# where this driver pulls both back to locally. The NCCL id rides NO path
# under this directory — see the module doc's "THE ID CROSSING".
CLUSTER_REMOTE_ARTIFACT_DIR="/root/jammi-ai/.gang-artifact"
CLUSTER_ARTIFACT_DIR="${CLUSTER_ARTIFACT_DIR:-.gpu-pull/gpu-cluster}"
# The remote path rank 0 mints the id to and rank 1 reads it from — on
# EACH host's own filesystem (never shared storage; this driver is what
# ships it between the two).
CLUSTER_REMOTE_ID_FILE="/root/nccl.id"

# The gating groups this driver's own verdict rule reads `PROVE_GROUP_RC`
# markers for, per rank — `::group::` names in `_rpc_remote_script`,
# verbatim. Declared BEFORE the sourced-execution guard so a fixture can
# `source` this file and see it (same convention `runpod_gpu_gang.sh`'s
# GANG_GROUPS uses).
CLUSTER_GROUPS=(cluster-build cluster-proof)

# F13's shared zero-test tripwire text, computed HERE (before the
# sourced-execution guard, same reason CLUSTER_GROUPS is) so a fixture can
# read the exact text `_rpc_remote_script` splices in via a plain
# `${...}` expansion — never a bare `$(...)` inside either rank's own
# heredoc body.
cluster_zero_test_tripwire="$(_rp_zero_test_tripwire_lines grc '$rank_log' "${CLUSTER_TEST_FILTER}")"

# --------------------------------------------------------------------------- #
# Pure, unit-testable helpers. Every one of these is callable by merely
# SOURCING this file — no network, no ssh, no RunPod account — which is
# exactly what test_gpu_cluster_lane.sh does.
# --------------------------------------------------------------------------- #

# AvailabilityLevel's own closed set, worst-to-best (RunPod REST v2 schema).
_RPC_AVAILABILITY_ORDER="NONE LOW MEDIUM HIGH"

# $1=min level $2=candidate level. 0 when candidate >= min in the ranking
# above; 1 when candidate is a recognized level below min; 2 when EITHER
# level is not one of the four documented spellings (a schema drift, never
# silently read as passing or failing).
_rpc_availability_at_least() {
  local min="${1:?_rpc_availability_at_least needs a min level}" have="${2:?_rpc_availability_at_least needs a candidate level}"
  local order="$_RPC_AVAILABILITY_ORDER" min_i=-1 have_i=-1 i=0 lvl
  for lvl in $order; do
    [ "$lvl" = "$min" ] && min_i=$i
    [ "$lvl" = "$have" ] && have_i=$i
    i=$((i + 1))
  done
  if [ "$min_i" -lt 0 ] || [ "$have_i" -lt 0 ]; then
    echo "::error::_rpc_availability_at_least: unrecognized AvailabilityLevel (min='${min}' have='${have}')" >&2
    return 2
  fi
  [ "$have_i" -ge "$min_i" ]
}

# The per-data-center availability read (A1): $1=gpuTypeId $2=min level;
# the catalog response BODY (the `GET /v2/catalog/gpus?...` JSON) on stdin.
# Prints a SPACE-SEPARATED list of qualifying data center ids (possibly
# empty — "the gpu type was found, but no data center clears the floor",
# read by the caller as no-capacity, exit 75) on stdout. Returns 0 on ANY
# successful parse (including zero qualifying data centers, or the gpu type
# entirely absent from the catalog — both are "no capacity", never a hard
# error); 2 when the body itself could not be read as the documented shape
# (a genuinely different failure — the catalog endpoint itself is
# unreachable or its schema moved).
_rpc_pick_data_centers() {
  local gpu="${1:?_rpc_pick_data_centers needs a gpuTypeId}" min="${2:?_rpc_pick_data_centers needs a min level}"
  python3 -c '
import json, sys
gpu, min_level = sys.argv[1], sys.argv[2]
order = ["NONE", "LOW", "MEDIUM", "HIGH"]
try:
    min_i = order.index(min_level)
except ValueError:
    print("PARSE_ERROR: unrecognized min level %r" % min_level)
    sys.exit(2)
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE_ERROR: could not parse the catalog response: %s" % e)
    sys.exit(2)
gpus = d.get("gpus")
if gpus is None:
    print("PARSE_ERROR: catalog response carries no gpus key")
    sys.exit(2)
entry = next((g for g in gpus if g.get("id") == gpu), None)
if entry is None:
    print("")
    sys.exit(0)
out = []
for dc in (entry.get("dataCenters") or []):
    lvl = dc.get("availability")
    try:
        i = order.index(lvl)
    except ValueError:
        continue
    if i >= min_i:
        did = dc.get("id")
        if did:
            out.append(did)
print(" ".join(out))
' "$gpu" "$min"
}

# The launch-time READ-BACK refusal (F2/F3): $1=the RAW `GET
# /v2/clusters/{id}/pods` response BODY $2=the expected entrypoint setup
# TEXT (the SAME string `_rp_entrypoint_setup` builds, unwrapped from its
# `bash -c '...'` wrapper). Prints one line per member:
# `<podId> <rank> STATE <overlay_ip> <direct_host_or_dash>
# <direct_port_or_dash>`, STATE one of READBACK_OK / READBACK_ARGS_MISMATCH
# / READBACK_NO_SSH_PATH. `READBACK_NO_SSH_PATH` fires only when the member
# carries NEITHER a non-null `ssh.direct` NOR a usable overlay `ip` to proxy
# through — a member with a live overlay ip but no direct ssh is still
# `READBACK_OK` (the no-public-port fallback, F3, reaches it via `ssh -J`
# through the primary — which itself MUST carry a direct host/port, checked
# by the caller). Returns 0 when the body parsed at all (even when it
# reports a member failing); 2 when the body itself could not be read as
# the documented `Pod` list shape.
_rpc_check_readback() {
  local body="${1:?_rpc_check_readback needs the pods response body}" setup="${2:?_rpc_check_readback needs the expected setup text}"
  python3 -c '
import json, sys
body, setup = sys.argv[1], sys.argv[2]
try:
    d = json.loads(body)
except Exception as e:
    print("PARSE_ERROR: could not parse the pods response: %s" % e)
    sys.exit(2)
pods = d.get("pods")
if pods is None:
    print("PARSE_ERROR: pods response carries no pods key")
    sys.exit(2)
for p in pods:
    pid = p.get("id") or "?"
    cl = p.get("cluster") or {}
    rank = cl.get("rank")
    rank = str(rank) if rank is not None else "?"
    ip = cl.get("ip") or "-"
    args = p.get("args") or (p.get("dockerArgs") or "")
    ssh_direct = (p.get("ssh") or {}).get("direct") or {}
    dhost = ssh_direct.get("host") or "-"
    dport = ssh_direct.get("port")
    dport = str(dport) if dport is not None else "-"
    if setup not in (args or ""):
        print("%s %s READBACK_ARGS_MISMATCH %s %s %s" % (pid, rank, ip, dhost, dport))
        continue
    if not ssh_direct and ip == "-":
        print("%s %s READBACK_NO_SSH_PATH %s %s %s" % (pid, rank, ip, dhost, dport))
        continue
    print("%s %s READBACK_OK %s %s %s" % (pid, rank, ip, dhost, dport))
' "$body" "$setup"
}

# F11's reader-side gate, mirrored on the SHIPPING side: refuses to `scp` a
# staging copy anywhere unless `stat` reports EXACTLY 128 bytes. $1=path.
# Returns 0 only at exactly 128 bytes; 1 otherwise (any other size,
# including a missing file, which `stat` itself fails on).
_rpc_id_file_ready() {
  local f="${1:?_rpc_id_file_ready needs a path}" sz
  sz="$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)" || return 1
  [ "$sz" = "128" ]
}

# The `ens1` proof (A5's "ens1 proof" fold): $1=a rank's own log file.
# Returns 0 when the log names `ens1` for NCCL's own NET/socket transport
# line (`NCCL INFO NET/Socket : Using [...] ens1` is the real string this
# driver's own `NCCL_DEBUG=INFO` export produces); 1 when the log has no
# such line — a missing line is this driver's own refusal, by name, never a
# silent pass.
_rpc_ens1_seen() {
  local log="${1:?_rpc_ens1_seen needs a log path}"
  grep -q 'NCCL INFO NET/Socket.*ens1' "$log" 2>/dev/null
}

# The shared verdict rule (mirrors `runpod_gpu_gang.sh`'s own
# `rp_gang_verdict`, generalized to ONE RANK's own ssh exit code + log —
# the driver calls this ONCE PER RANK and combines both below). $1=raw ssh
# rc $2=that rank's own log file.
rp_cluster_rank_verdict() {
  local raw_rc="$1" log="$2"
  declare -A grc_map=()
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    if rp_parse_prove_marker "$line"; then
      grc_map["$RP_PARSED_MARKER_NAME"]="$RP_PARSED_MARKER_RC"
    fi
  done < "$log"

  local has_prove_exit=0
  if grep -q '^PROVE_EXIT=' "$log" 2>/dev/null; then # tripwire-ok: an unreadable/missing log independently loses every group marker above, which the all_pass rule below reports by name.
    has_prove_exit=1
  fi

  local all_pass=1 missing_or_failed=() g v
  for g in "${CLUSTER_GROUPS[@]}"; do
    v="${grc_map[$g]:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -ne 0 ]; then
      all_pass=0
      missing_or_failed+=("${g}=${v:-<missing>}")
    fi
  done

  local rc="$raw_rc"
  if [ "$raw_rc" -eq 0 ]; then
    if [ "$all_pass" -ne 1 ]; then
      echo "::error::GPU cluster: ssh exited 0 but CLUSTER_GROUPS member(s) missing or non-zero: ${missing_or_failed[*]:-<none>} — PROVE_EXIT disagrees with its own markers" >&2
      rc=1
    fi
  elif [ "$has_prove_exit" -eq 1 ]; then
    rc="$raw_rc"
  else
    if [ "$all_pass" -ne 1 ]; then
      echo "::error::GPU cluster: cut/hang (raw rc=${raw_rc}) with group(s) unresolved: ${missing_or_failed[*]:-<none>}" >&2
    fi
    rc="$raw_rc"
  fi
  return "$rc"
}

# Combines both ranks' own verdicts into the leg's single exit code. $1=rank0
# rc $2=rank1 rc. Neither rank passing silently masks the other: the WORST
# (highest-priority) code wins, in the order the module doc's EXIT CONTRACT
# lists (a named refusal always outranks a bare nonzero).
rp_cluster_verdict() {
  local r0="$1" r1="$2"
  [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] && return 0
  for special in 75 76 77 97 124; do
    [ "$r0" -eq "$special" ] && return "$special"
    [ "$r1" -eq "$special" ] && return "$special"
  done
  [ "$r0" -ne 0 ] && return "$r0"
  return "$r1"
}

# The shared per-rank remote script text (F13: the zero-test tripwire lives
# HERE, spliced via the pre-computed `${cluster_zero_test_tripwire}`
# variable — never a bare `$(...)` inside this heredoc). $1=rank (0|1).
# `world`/`pod_count`/`gpu_count_per_pod` derive from
# RP_CLUSTER_POD_COUNT/RP_CLUSTER_GPU_COUNT_PER_POD, never a second literal.
_rpc_remote_script() {
  local rank="${1:?_rpc_remote_script needs a rank}"
  cat <<EOF
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (ledger row 17: no cross-target-dir reuse on this image)
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
export JAMMI_GANG_ARTIFACT_DIR=${CLUSTER_REMOTE_ARTIFACT_DIR}
export JAMMI_REQUIRE_CUDA_TWO_HOSTS=1
export JAMMI_GANG_TWO_HOSTS_RANK=${rank}
export JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT}
export JAMMI_GANG_TWO_HOSTS_ID_FILE=${CLUSTER_REMOTE_ID_FILE}
export NCCL_SOCKET_IFNAME=ens1
export NCCL_DEBUG=INFO
echo "::group::device"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
gpu_seen="\$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)"
if [ "\${gpu_seen}" != "${RP_CLUSTER_GPU_COUNT_PER_POD}" ]; then
  echo "::error::device count mismatch: nvidia-smi reports \${gpu_seen} GPU(s) but this member rented ${RP_CLUSTER_GPU_COUNT_PER_POD} -- refusing to build"
  exit 97
fi
compute_cap_raw="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
compute_cap_norm="\${compute_cap_raw//./}"
if [ "\${compute_cap_norm}" != "\${CUDA_COMPUTE_CAP:-}" ]; then
  echo "::error::compute_cap mismatch: nvidia-smi reports compute_cap=\${compute_cap_raw} but CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>} -- refusing to build"
  exit 97
fi
echo "::endgroup::"
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
echo "PROVE_SHA=\$(git rev-parse HEAD)"
rc=0
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \\
  || { echo "::error::CUTLASS submodule init failed -- refusing to attempt the flash-attn build" >&2; exit 1; }

echo "::group::cluster-build"
grc=0
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=cluster-build rc=\${grc}"
echo "::endgroup::"

echo "::group::cluster-proof"
grc=0
mkdir -p "\${JAMMI_GANG_ARTIFACT_DIR}"
rank_log=/tmp/cluster_proof_${rank}.log
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability ${CLUSTER_TEST_FILTER} -- --nocapture --test-threads=1 2>&1 | tee "\$rank_log"
grc=\${PIPESTATUS[0]}
${cluster_zero_test_tripwire}
if [ "\$grc" -eq 0 ] && [ -z "\$(ls -A "\${JAMMI_GANG_ARTIFACT_DIR}" 2>/dev/null)" ]; then # tripwire-ok: ls's stderr on a missing dir is not evidence; an empty result is exactly the "no artifact written" case this arm reports by name on the next line.
  echo "::error::the cluster test passed but wrote NO artifact to \${JAMMI_GANG_ARTIFACT_DIR}" >&2
  grc=1
fi
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=cluster-proof rc=\${grc}"
echo "::endgroup::"

echo "PROVE_EXIT=\${rc}"; exit \$rc
EOF
}

# Assembles the ONE `gang` (leg=cluster) artifact from both ranks' own
# `rank-<r>.json` reports (this driver is the SOLE writer of the assembled
# artifact; the ranks only report). $1=rank0 json path $2=rank1 json path
# $3=git_sha $4=box label $5=out path. Prints nothing; returns 0 on a
# successful write (regardless of the recorded verdict — a `fail` is
# representable, never refused at assembly time), 1 when EITHER rank report
# is missing or malformed (an artifact with no evidence to assemble from is
# refused outright, never synthesized).
_rpc_assemble_gang_artifact() {
  local r0="${1:?needs rank-0.json}" r1="${2:?needs rank-1.json}" git_sha="${3:?needs git_sha}" \
        box="${4:?needs a box label}" out="${5:?needs an output path}"
  python3 -c '
import json, sys
r0_path, r1_path, git_sha, box, out_path = sys.argv[1:6]
try:
    r0 = json.load(open(r0_path))
    r1 = json.load(open(r1_path))
except Exception as e:
    print("could not read one or both rank reports: %s" % e, file=sys.stderr)
    sys.exit(1)
reports = sorted([r0, r1], key=lambda r: r.get("rank", 0))

def _resolved(v):
    return isinstance(v, str) and v.strip() and v.strip().lower() != "unknown"

# F4: a named driver refusal, never an artifact -- this assembler is the
# SOLE writer, and check_cuda_run_artifacts.py (rule k, F4) refuses to
# accept "unknown" or a repeated host on the far side anyway; catching it
# HERE means a bad run never even reaches a committed file for a human to
# accidentally review as real evidence.
for r in reports:
    host, iface = r.get("hostname"), r.get("nccl_socket_ifname")
    if not _resolved(host):
        print("refusing to assemble: rank %r own hostname is unresolved (%r)" % (r.get("rank"), host), file=sys.stderr)
        sys.exit(2)
    if not _resolved(iface):
        print("refusing to assemble: rank %r own nccl_socket_ifname is unresolved (%r)" % (r.get("rank"), iface), file=sys.stderr)
        sys.exit(2)
if reports[0].get("hostname") == reports[1].get("hostname"):
    print("refusing to assemble: both ranks report the SAME host (%r) -- not the two-host bootstrap this leg proves" % reports[0].get("hostname"), file=sys.stderr)
    sys.exit(2)

ranks = []
for r in reports:
    ranks.append({
        "rank": r.get("rank"),
        "host": r.get("hostname"),
        "device": "cuda:%s" % r.get("device_ordinal", 0),
        "iface": r.get("nccl_socket_ifname"),
        # F4: kept PER RANK, never collapsed here -- the checker asserts
        # equality across ranks itself rather than trusting an
        # already-collapsed value.
        "reduced_vector_digest": r.get("reduced_vector_digest_sha256"),
    })
both_pass = all(r.get("verdict") == "pass" for r in reports)
digests = [r.get("reduced_vector_digest_sha256") for r in reports]
digest_equal = both_pass and digests[0] is not None and digests[0] == digests[1]
if both_pass and digest_equal:
    verdict = "pass"
    reason = ""
    reduced_digest = digests[0]
    status = "GREEN"
elif both_pass and not digest_equal:
    verdict = "fail"
    reason = "both ranks report pass but their reduced_vector_digest_sha256 disagree: %r vs %r" % (digests[0], digests[1])
    reduced_digest = None
    status = "RED"
else:
    verdict = "fail"
    reasons = [r.get("reason") or "" for r in reports if r.get("verdict") != "pass"]
    reason = "; ".join(x for x in reasons if x) or "at least one rank did not pass"
    reduced_digest = None
    status = "RED"
artifact = {
    "schema_version": 1,
    "git_sha": git_sha,
    "box": box,
    "producer": {
        # F4: bound to THIS driver -- the sole writer of the cluster-leg
        # artifact (check_cuda_run_artifacts.py GANG_LEG_PRODUCER_PATH).
        "path": "ci/scripts/runpod_gpu_cluster.sh",
        "kind": "script",
        "invocation": "bash ci/scripts/runpod_gpu_cluster.sh",
        "gating": "env:JAMMI_REQUIRE_CUDA_TWO_HOSTS",
    },
    "status": status,
    "artifact_kind": "gang",
    "gang": {
        "leg": "cluster",
        "world": 2,
        "collective": "nccl",
        "hosts": 2,
        "ranks": ranks,
        "reduced_vector_digest": reduced_digest,
        "verdict": verdict,
        "pod_count": 2,
        "gpu_count_per_pod": 1,
        "ttl_hours": int((sys.argv[6] if len(sys.argv) > 6 else "1")),
    },
}
if verdict == "fail":
    artifact["gang"]["reason"] = reason
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)
    f.write("\n")
' "$r0" "$r1" "$git_sha" "$box" "$out" "${RP_TTL_HOURS}"
}

# Wraps the id-secrecy scan (F5, `gang_id_secrecy_scan.py`) invocation in
# ONE place so both the real orchestration below and
# test_gpu_cluster_lane.sh's own fixtures call it identically. $1=staging
# id file $2=pulled artifact dir $3=run log $4=assembled artifact path.
# Deletes the staging copy ONLY on a clean (exit 0) scan.
_rpc_run_id_secrecy_scan() {
  local staging="${1:?needs the staging id file}" artifact_dir="${2:?needs the artifact dir}" \
        log="${3:?needs the run log}" assembled="${4:?needs the assembled artifact}"
  python3 "$DIR/gang_id_secrecy_scan.py" \
    --staging-file "$staging" --artifact-dir "$artifact_dir" --log "$log" \
    --assembled-artifact "$assembled" --delete-staging
}

# The first executed run's own record of whether MEMBER SELF-REMOVAL
# actually works on a cluster (S4: members expose `actions: []`; the
# module doc's own header states this is otherwise UNMEASURED). Checked
# BEFORE this driver's own delete call: a 404 on the cluster's own GET
# means every member already self-terminated and RunPod retired the
# cluster object on its own -- "ok". Any other status (200, meaning the
# cluster is still present) means self-removal has NOT happened by the
# time this driver's own EXIT trap fires -- "refused" -- and the driver's
# own `rp_cluster_delete` call is what actually retires it.
_rpc_self_remove_status() {
  local id="${1:?needs a cluster id}" resp status
  resp="$(_rp_rest GET "/v2/clusters/${id}")"
  status="$(printf '%s\n' "$resp" | head -n1)"
  case "$status" in
    404) echo "ok" ;;
    *) echo "refused" ;;
  esac
}

# F2/F3(c): captures the PENDING exit status FIRST ($? here is whatever the
# script was about to exit with), chains runpod_lib.sh's own `rp_cleanup`
# (this trap REPLACES the `trap rp_cleanup EXIT` that sourcing runpod_lib.sh
# already installed -- rp_cleanup is not "installed by rp_init", it is
# installed unconditionally at source time, and never runs again unless
# called explicitly here), and — a failed cluster delete is a LEAKED
# resource, never a warning folded into an otherwise-green exit — joins that
# failure into the exit status this trap finally exits with. Reads the
# GLOBAL `cluster_id` (set by the executed-only orchestration below; unset
# when merely sourced for a fixture, which is exactly "no cluster to clean
# up yet" — the `-n` guard below).
_rpc_cleanup_cluster() {
  local rc=$?
  if [ -n "${cluster_id:-}" ]; then
    local self_remove
    self_remove="$(_rpc_self_remove_status "$cluster_id")"
    echo "cluster-self-remove: ${self_remove}"
    if [ "$self_remove" = "ok" ]; then
      : # already gone -- a driver-initiated delete against a 404 would be a spurious failure.
    elif rp_cluster_delete "$cluster_id"; then
      echo "cluster deleted by the driver's own EXIT trap (member self-removal had not taken by then)"
    else
      echo "::error::LEAKED cluster ${cluster_id}: could not delete on exit -- gpu-reap.yml's 6-hourly sweep is the backstop"
      [ "$rc" -eq 0 ] && rc=1
    fi
  fi
  rp_cleanup  # F2: chain the library's own EXIT cleanup (rm -rf "$RP_WORK" -- the staging id file and the ssh keypair).
  exit "$rc"
}

# --------------------------------------------------------------------------- #
# Everything below runs only when this file is EXECUTED, never when it is
# `source`d (the same guard runpod_gpu_gang.sh/runpod_gpu_prove.sh use) --
# test_gpu_cluster_lane.sh sources this file and drives every helper above
# directly, renting nothing and calling no network primitive.
# --------------------------------------------------------------------------- #
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

# F6: the run log lives INSIDE the pulled/uploaded directory from the very
# first byte -- created BEFORE the tee starts, so nothing this driver ever
# emits is written to an unuploaded, unscanned path.
mkdir -p "$CLUSTER_ARTIFACT_DIR"
RUN_LOG="$CLUSTER_ARTIFACT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1   # F6: ONE tee'd stream for every byte this driver emits.

_rpc_phase() { echo "=== PHASE ($(( SECONDS )))s: $* ==="; }

DEADLINE=$(( SECONDS + RP_TTL_HOURS * 3600 - 600 ))  # T-10m budget cut (F16).

_rpc_phase "availability read (product=CLUSTER, cloud=SECURE)"
avail_resp="$(_rp_rest GET "/v2/catalog/gpus?include=AVAILABILITY&product=CLUSTER&count=1&cloud=SECURE")"
avail_status="$(printf '%s\n' "$avail_resp" | head -n1)"
avail_body="$(printf '%s\n' "$avail_resp" | tail -n +2)"
if [ "$avail_status" != "200" ]; then
  echo "::error::catalog availability read failed (status ${avail_status}) -- treating as no capacity"
  exit 75
fi
dcs="$(printf '%s' "$avail_body" | _rpc_pick_data_centers "$RP_CLUSTER_GPU_TYPE" "$RP_CLUSTER_MIN_AVAILABILITY")"
case "$dcs" in
  PARSE_ERROR*) echo "::error::${dcs}"; exit 75 ;;
esac
if [ -z "$dcs" ]; then
  echo "::error::no data center offers ${RP_CLUSTER_GPU_TYPE} at ${RP_CLUSTER_MIN_AVAILABILITY} or better (SUPPLY_CONSTRAINT)"
  exit 75
fi
echo "candidate data center(s): ${dcs}"

# F1: rp_init BEFORE the create call, exactly as runpod_gpu_gang.sh's own
# pod leg does — it is what generates the SSH keypair (RP_PUBKEY, read by
# `_rp_cluster_payload` into the create body's own `env.PUBLIC_KEY`) and
# populates RP_SSHO (StrictHostKeyChecking/IdentitiesOnly/-i), which every
# ssh/scp/rsync call below this point depends on. Without it the create
# body ships an EMPTY authorized-key and every later ssh call runs with an
# empty RP_SSHO array (no `-i`, no IdentitiesOnly) — a silent, wrong-key
# failure that reads exactly like "not yet reachable".
rp_init

_rpc_phase "cluster create"
cluster_id="$(rp_cluster_create "$RP_CLUSTER_GPU_TYPE" "$dcs")" || { echo "::error::cluster create failed"; exit 75; }
echo "cluster ${cluster_id} created"

# `_rpc_self_remove_status`/`_rpc_cleanup_cluster` are defined ABOVE, in the
# pure-helpers section (so test_gpu_cluster_lane.sh can drive them by
# merely sourcing this file, mocking `_rp_rest`/`rp_cluster_delete`) — only
# the trap REGISTRATION itself is an executed-only action.
trap _rpc_cleanup_cluster EXIT

_rpc_phase "waiting for both members RUNNING with a usable ssh path"
primary_host="" primary_port="" member_host="" member_port="" member_ip="" pods_body="" ok_count=0
deadline_ssh=$(( SECONDS + RP_SSH_WAIT_SECS ))
while [ "$SECONDS" -lt "$deadline_ssh" ]; do
  resp="$(_rp_rest GET "/v2/clusters/${cluster_id}/pods")"
  status="$(printf '%s\n' "$resp" | head -n1)"
  pods_body="$(printf '%s\n' "$resp" | tail -n +2)"
  if [ "$status" = "200" ]; then
    readback="$(_rpc_check_readback "$pods_body" "$(_rp_entrypoint_setup "$RP_TTL_HOURS")")"
    readback_rc=$?
    if [ "$readback_rc" -eq 0 ]; then
      ok_count=0
      mismatch=0
      while IFS=' ' read -r _pid rank state ip dhost dport; do
        [ -n "$rank" ] || continue
        case "$state" in
          READBACK_ARGS_MISMATCH) mismatch=1 ;;
          READBACK_NO_SSH_PATH) : ;;
          READBACK_OK)
            ok_count=$((ok_count + 1))
            if [ "$rank" = "0" ]; then
              primary_host="$dhost"; primary_port="$dport"
            else
              member_host="$dhost"; member_port="$dport"; member_ip="$ip"
            fi ;;
        esac
      done <<< "$readback"
      if [ "$mismatch" -eq 1 ]; then
        echo "::error::a member's Pod.args does not echo the shared entrypoint text -- refusing"
        exit 97
      fi
      [ "$ok_count" -ge "$RP_CLUSTER_POD_COUNT" ] && break
    fi
  fi
  sleep 5
done
if [ "$ok_count" -lt "$RP_CLUSTER_POD_COUNT" ]; then
  echo "::error::not every member reached a usable ssh path (direct or overlay-proxy) within ${RP_SSH_WAIT_SECS}s"
  exit 97
fi
if [ "$primary_host" = "-" ] || [ -z "$primary_host" ]; then
  echo "::error::the primary (rank 0) member carries no direct ssh endpoint -- there is no jump host for the no-public-port fallback either"
  exit 97
fi
# F3's no-public-port fallback: the member's own `ssh.direct` was absent
# (dhost="-") but its overlay `ip` is known -- proxy through the primary,
# which the check above already confirmed carries a direct endpoint.
# `member_extra_sshopts` carries ONLY the proxy option (never a port flag,
# which differs between ssh's `-p` and scp/rsync's `-P` -- each call site
# below supplies its own port flag explicitly instead of one shared array
# with an embedded, flag-specific `-p`/`-P`).
member_extra_sshopts=()
if [ "$member_host" = "-" ] || [ -z "$member_host" ]; then
  [ -n "$member_ip" ] && [ "$member_ip" != "-" ] || { echo "::error::the member carries neither a direct ssh endpoint nor an overlay ip -- no path reaches it"; exit 97; }
  member_host="$member_ip"
  member_port=22
  member_extra_sshopts=(-o "ProxyJump=root@${primary_host}:${primary_port}")
fi

_rpc_phase "build + two-host proof"
rank0_log="$(mktemp)"; rank1_log="$(mktemp)"
STAGING_ID_FILE="$RP_WORK/nccl.id"

_rpc_remote_script 0 | ssh "${RP_SSHO[@]}" -p "$primary_port" "root@${primary_host}" "timeout ${RP_TIMEOUT:-3000} bash -s" > "$rank0_log" 2>&1 &
rank0_pid=$!

_rpc_phase "polling the primary for the 128-byte id file"
id_deadline=$(( SECONDS + RP_SSH_WAIT_SECS ))
id_ready=0
while [ "$SECONDS" -lt "$id_deadline" ]; do
  if ! kill -0 "$rank0_pid" 2>/dev/null; then
    break  # rank 0 already exited -- stop polling; `wait` below reads its rc.
  fi
  remote_size="$(ssh "${RP_SSHO[@]}" -p "$primary_port" "root@${primary_host}" "stat -c%s '${CLUSTER_REMOTE_ID_FILE}' 2>/dev/null")"
  if [ "$remote_size" = "128" ]; then
    id_ready=1
    break
  fi
  sleep 3
done
if [ "$id_ready" -ne 1 ]; then
  echo "::error::rank 0 never produced a 128-byte id file within ${RP_SSH_WAIT_SECS}s"
  kill -TERM "$rank0_pid" 2>/dev/null; wait "$rank0_pid" 2>/dev/null
  exit 76
fi

scp "${RP_SSHO[@]}" -P "$primary_port" "root@${primary_host}:${CLUSTER_REMOTE_ID_FILE}" "$STAGING_ID_FILE" \
  || { echo "::error::could not scp the id down from the primary"; kill -TERM "$rank0_pid" 2>/dev/null; wait "$rank0_pid" 2>/dev/null; exit 76; }
chmod 600 "$STAGING_ID_FILE"
_rpc_id_file_ready "$STAGING_ID_FILE" \
  || { echo "::error::the staged id copy is not exactly 128 bytes -- refusing to ship it"; kill -TERM "$rank0_pid" 2>/dev/null; wait "$rank0_pid" 2>/dev/null; exit 76; }

scp "${RP_SSHO[@]}" "${member_extra_sshopts[@]}" -P "$member_port" "$STAGING_ID_FILE" "root@${member_host}:${CLUSTER_REMOTE_ID_FILE}" \
  || { echo "::error::could not scp the id up to the member"; kill -TERM "$rank0_pid" 2>/dev/null; wait "$rank0_pid" 2>/dev/null; exit 76; }

_rpc_remote_script 1 | ssh "${RP_SSHO[@]}" "${member_extra_sshopts[@]}" -p "$member_port" "root@${member_host}" "timeout ${RP_TIMEOUT:-3000} bash -s" > "$rank1_log" 2>&1 &
rank1_pid=$!

_rpc_phase "watching both ranks (inactivity + wrong-tree + budget)"
last_growth=$SECONDS
last_size0=0; last_size1=0
while kill -0 "$rank0_pid" 2>/dev/null || kill -0 "$rank1_pid" 2>/dev/null; do
  sleep 5
  sz0=$(wc -c < "$rank0_log" 2>/dev/null || echo 0)
  sz1=$(wc -c < "$rank1_log" 2>/dev/null || echo 0)
  if [ "$sz0" -gt "$last_size0" ] || [ "$sz1" -gt "$last_size1" ]; then
    last_growth=$SECONDS
    last_size0=$sz0; last_size1=$sz1
  fi
  if [ -n "${PROVE_EXPECT_SHA:-}" ]; then
    wrong_tree=0
    for candidate_log in "$rank0_log" "$rank1_log"; do
      while IFS= read -r line; do
        if rp_parse_prove_sha "$line" && [ "$RP_PARSED_PROVE_SHA" != "$PROVE_EXPECT_SHA" ]; then
          wrong_tree=1
          break
        fi
      done < "$candidate_log"
      [ "$wrong_tree" -eq 1 ] && break
    done
    if [ "$wrong_tree" -eq 1 ]; then
      echo "::error::wrong tree: a rank's own PROVE_SHA disagreed with PROVE_EXPECT_SHA=${PROVE_EXPECT_SHA}"
      kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null
      exit 77
    fi
  fi
  if [ $(( SECONDS - last_growth )) -ge "${RP_INACTIVITY}" ]; then
    echo "::error::inactivity: no new output on either rank's log for ${RP_INACTIVITY}s"
    kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null
    exit 76
  fi
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    echo "::error::budget cut at T-10m (per-phase breakdown above)"
    kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null
    exit 124
  fi
done
wait "$rank0_pid"; rank0_rc=$?
wait "$rank1_pid"; rank1_rc=$?

# F6 advisory: both ranks' own logs land in the uploaded/scanned directory
# too, pass or fail alike -- copied here, unconditionally, before any
# pass/fail branching below, rather than only on a path that might exit
# early.
mkdir -p "$CLUSTER_ARTIFACT_DIR"
cp -f "$rank0_log" "${CLUSTER_ARTIFACT_DIR}/rank0.log" 2>/dev/null || echo "::warning::could not copy rank 0's own log into ${CLUSTER_ARTIFACT_DIR}"
cp -f "$rank1_log" "${CLUSTER_ARTIFACT_DIR}/rank1.log" 2>/dev/null || echo "::warning::could not copy rank 1's own log into ${CLUSTER_ARTIFACT_DIR}"

rp_cluster_rank_verdict "$rank0_rc" "$rank0_log"; rank0_final=$?
rp_cluster_rank_verdict "$rank1_rc" "$rank1_log"; rank1_final=$?
rp_cluster_verdict "$rank0_final" "$rank1_final"; rc=$?

if [ "$rc" -eq 0 ]; then
  if ! _rpc_ens1_seen "$rank0_log" || ! _rpc_ens1_seen "$rank1_log"; then
    echo "::error::one or both ranks' logs never named ens1 for the NCCL NET transport -- refusing to read this as a proof"
    rc=1
  fi
fi

_rpc_phase "pulling both ranks' artifacts"
# CLUSTER_ARTIFACT_DIR already exists (created before the run log's own tee
# started, F6) -- rsync below just fills it in further.
pull_rc=0
rsync -az -e "ssh ${RP_SSHO[*]} -p ${primary_port}" "root@${primary_host}:${CLUSTER_REMOTE_ARTIFACT_DIR}/" "${CLUSTER_ARTIFACT_DIR}/" || pull_rc=$?
rsync -az -e "ssh ${RP_SSHO[*]} ${member_extra_sshopts[*]} -p ${member_port}" "root@${member_host}:${CLUSTER_REMOTE_ARTIFACT_DIR}/" "${CLUSTER_ARTIFACT_DIR}/" || pull_rc=$?
if [ "$pull_rc" -ne 0 ]; then
  echo "::error::artifact pull failed (rsync rc=${pull_rc}) -- a leg with no retrievable evidence proves nothing reviewable" >&2
  [ "$rc" -eq 0 ] && rc="$pull_rc"
fi

_rpc_phase "assembling the gang artifact"
ASSEMBLED="${CLUSTER_ARTIFACT_DIR}/gang-cluster-$(date -u +%Y%m%d%H%M%S).json"
if [ -f "${CLUSTER_ARTIFACT_DIR}/rank-0.json" ] && [ -f "${CLUSTER_ARTIFACT_DIR}/rank-1.json" ]; then
  measured_sha="${PROVE_EXPECT_SHA:-$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)}"
  _rpc_assemble_gang_artifact "${CLUSTER_ARTIFACT_DIR}/rank-0.json" "${CLUSTER_ARTIFACT_DIR}/rank-1.json" \
    "$measured_sha" "a100-sxm4-cluster" "$ASSEMBLED" \
    || { echo "::error::could not assemble the gang artifact" >&2; [ "$rc" -eq 0 ] && rc=1; }
else
  echo "::error::one or both rank reports were not pulled -- cannot assemble the gang artifact" >&2
  [ "$rc" -eq 0 ] && rc=1
fi

_rpc_phase "id-secrecy scan"
if [ -f "$STAGING_ID_FILE" ]; then
  if [ -f "$ASSEMBLED" ]; then
    _rpc_run_id_secrecy_scan "$STAGING_ID_FILE" "$CLUSTER_ARTIFACT_DIR" "$RUN_LOG" "$ASSEMBLED"
    scan_rc=$?
  else
    scan_rc=2
    echo "::error::no assembled artifact exists to scan" >&2
  fi
  if [ "$scan_rc" -ne 0 ]; then
    echo "::error::id-secrecy scan FAILED (rc=${scan_rc}) -- a carrier may hold the NCCL id, or could not be examined" >&2
    [ "$rc" -eq 0 ] && rc="$scan_rc"
  fi
else
  echo "::warning::no staging id file present to scan (a prior phase never reached id-shipping)"
fi

_rpc_phase "billing read (F16)"
billing_resp="$(_rp_rest GET /v2/billing/clusters)"
echo "billing read status for this run window: $(printf '%s\n' "$billing_resp" | head -n1)"

echo "=== GPU cluster leg exit=${rc} ==="
exit "$rc"

fi # end sourced-execution guard
