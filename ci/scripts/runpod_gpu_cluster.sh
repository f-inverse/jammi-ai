#!/usr/bin/env bash
# GPU cluster leg: the two-HOST NCCL bootstrap, rented over ONE of TWO
# transports selected by `RP_TWO_HOST_TRANSPORT` (`pods` default, `cluster`)
# — the only place a real cross-HOST NCCL gang is exercised before release.
# The pod leg (`runpod_gpu_gang.sh`) proves a two-DEVICE gang inside one
# pod; this driver proves the two-HOST bootstrap (`ncclCommInitRank`, an
# out-of-band id file) that pod leg cannot reach at all. Never
# `runpod_gpu_gang.sh` itself — a SEPARATE driver, a SEPARATE workflow, and
# (under `cluster`) a SEPARATE RunPod object type (a cluster is retired by
# deleting the CLUSTER, never a member pod — see runpod_lib.sh's own
# `rp_cluster_delete`/`rp_cluster_sweep`).
#
# WHAT IT RENTS, per transport:
#
#   `pods` (default — ordinary pods provision reliably where INSTANT
#   CLUSTERs do not; measured: 8 of 9 cluster creates refused `Insufficient
#   resources` in one session). TWO ORDINARY pods (`POST /v2/pods`,
#   `rp_two_host_pod_create`), each `gpu: {id: RP_CLUSTER_GPU_TYPE,
#   count: 1}`, `cloud: SECURE`, `globalNetworking: true`, co-located in the
#   SAME data center — chosen by intersecting "offers this GPU type at
#   RP_CLUSTER_MIN_AVAILABILITY or better" (`GET /v2/catalog/gpus?
#   include=AVAILABILITY&product=POD&count=1&cloud=SECURE`, the SAME read
#   `runpod_gpu_prove.sh`/`runpod_gpu_gang.sh` already make) with "carries
#   Global Networking" (`GET /v2/catalog/datacenters`, `globalNetwork: true` — READ
#   LIVE at run time, never a hard-coded list; snapshot verified 2026-09-16:
#   CA-MTL-1, CA-MTL-3, EU-CZ-1, EU-FR-1, EU-NL-1, EU-RO-1, EU-SE-1,
#   EUR-IS-2, EUR-IS-4, OC-AU-1, US-CA-2, US-GA-2, US-IL-1, US-KS-2,
#   US-NC-1, US-TX-3, US-TX-4, US-WA-1). Rank is OURS to assign: the first
#   pod created is rank 0, the second rank 1. Each member derives its OWN
#   NCCL_SOCKET_IFNAME from its Global-Networking ip at run time (the
#   kernel route table, /proc/net/route, longest matching prefix — the image
#   ships no `ip` binary) — never a literal interface name; a member whose
#   route table covers no such ip refuses (97) by name.
#
#   `cluster` (kept, never removed — the SAME proof over a different rental
#   mechanism). One RunPod CLUSTER, `podCount: 2`, `gpuCountPerPod: 1`, the
# `RP_CLUSTER_GPU_TYPE` SECURE candidate (default `NVIDIA A100-SXM4-80GB`,
# the one shape S4 measured co-placed on ONE data center; the workflow's
# `gpu_type` input may name another part in the sm_80/86/89/90 domain
# `_rpc_compute_cap_for_gpu_type` maps, and the leg then proves THAT part's
# SASS — any other gpuTypeId is refused before phase 0). `dataCenterIds` is
# never left to the scheduler: this driver reads per-data-center
# availability itself (`GET /v2/catalog/gpus?include=AVAILABILITY&
# product=CLUSTER&count=1&cloud=SECURE`) and passes only the data
# center(s) at `RP_CLUSTER_MIN_AVAILABILITY` (MEDIUM) or better — co-
# placement needs ONE data center, and the overall (account-wide) figure
# alone does not establish that any single one actually has it (A1). Both
# transports assemble the SAME `gang` artifact (`gang.leg == "cluster"` —
# the two-HOST leg is the fact that matters downstream — `gang.transport`
# is the sub-fact naming WHICH mechanism carried it: `instant-cluster` or
# `global-networking`, closed set, checked by `check_cuda_run_artifacts.py`
# rule (k)).
#
# COST BOUND (human-approved), per transport:
#
#   `cluster` (S4's measured $1.908/GPU/h for a SECURE cluster GPU — the
#   catalog's own $1.59 is the POD price, a different rate). The figures
#   below are for the DEFAULT part; an operator-chosen `gpu_type` bills at
#   RunPod's cluster rate for that part, which this header and
#   `test_gpu_cluster_lane.sh`'s G4 re-derivation do not bound — choosing
#   it is the cost decision (an H100 SXM cluster ran at ~$6.6/h in run
#   35127869122):
#
#     2 GPUs x $1.908/GPU/h = $3.816/h.
#
#     (i) terminate-succeeds (the ordinary path): ONE create (no candidate
#         walk — a cluster create names its data center directly, unlike a
#         pod's failover search), billing to RP_TTL_HOURS=1:
#           1 h x $3.816/h = $3.82 per run.
#     (ii) sweep-only (the worst path — the EXIT trap's own
#         `rp_cluster_delete` call fails, AND member self-removal is
#         UNMEASURED — see runpod_lib.sh's own cluster-primitives header):
#         the cluster bills to its own TTL, then `gpu-reap.yml`'s 6-hourly
#         `rp_cluster_sweep` is the backstop:
#           (1 + 6) h x $3.816/h = $26.71.
#
#   `pods` (the catalog's own SECURE POD rate, $1.59/GPU/h for the same
#   default A100 SXM4 part):
#
#     2 GPUs x $1.59/GPU/h = $3.18/h.
#
#     (i) terminate-succeeds: two ordinary creates (no candidate walk — the
#         co-located data center is chosen up front), billing to
#         RP_TTL_HOURS=1:
#           1 h x $3.18/h = $3.18 per run.
#     (ii) sweep-only (both pods carry the ORDINARY pod-name shape, so
#         `gpu-reap.yml`'s SAME 6-hourly `rp_sweep` is the backstop — no
#         separate sweep primitive for this transport):
#           (1 + 6) h x $3.18/h = $22.26.
#
# `ci/scripts/test_gpu_cluster_lane.sh` re-derives every one of these four
# figures from this script's own RP_TTL_HOURS/each transport's own rate/the
# 2-GPU shape and fails if the printed figure and the mechanism disagree.
# `≤ 1 h billed, ≤ 2 runs` is the standing spend authorization
# (2026-09-13) this lane holds to; `MAX_ATTEMPTS` lives in `.github/
# workflows/gpu-cluster.yml`, exactly the gang lane's own shape (this
# driver attempts ONE create sequence per invocation; the workflow's own
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
# staging copy up to the member host — ONLY THEN does rank 1's PROOF start
# (both ranks clone and build concurrently; rank 1's script blocks between
# its build and its proof until that file holds 128 bytes). The crossing is
# a phase of the one watch loop, bounded by rank 0's liveness, log growth
# within RP_INACTIVITY and the T-10m budget — never a separate wall clock
# (`_rpc_run_two_ranks`). The id
# never rides inside `JAMMI_GANG_ARTIFACT_DIR` and never reaches this
# driver's own stdout/log in the clear. The moment the download from the
# primary is ATTEMPTED, this driver's own cleanup trap starts scanning: on
# EVERY exit arm from that point on — the happy path, a failed pull, a
# refused assembly, an inactivity/wrong-tree/budget cut, or the process
# being SIGNALLED (the trap is registered on EXIT, INT, TERM AND HUP, never
# EXIT alone — round 3 F1: an untrapped SIGINT, what a CI runner's own
# cancellation sends first, otherwise skips an EXIT-only trap entirely) —
# the trap runs the scan FIRST, before either of its own two REST calls
# (self-removal status, `rp_cluster_delete` — round 3 F1: previously
# sequenced behind both, so a hang in an untimed `curl` call could delay
# it), over every carrier the run has produced so far (the pulled artifact
# dir, the run log, the staging copy's own directory listing, and the
# assembled artifact — but ONLY when THIS run's own assembly step actually
# claims to have written one; P-A2: a refusal arm that never reached
# assembly, or whose assembly step itself refused and wrote nothing, is not
# penalised for a file it never promised — its clean run.log and rank logs
# reach the upload step exactly as they are) for the id in every encoding
# the ship step could emit (raw, hex either case — each whitespace-stripped
# on a miss, round 3 A2 — base64, including a line-wrapped base64
# encoding). A scan that is NOT clean DESTROYS the whole carrier directory,
# SYNCHRONOUSLY, before this process exits (moved outside the uploaded path
# and deleted right there, in the same trap invocation — round 3 F2: never
# deferred to `rp_cleanup`'s own `RP_SESSION`-conditional `$RP_WORK` teardown,
# which a real interactive/resumed run skips entirely; emptied in place,
# matching `..`-prefixed names too — round 3 F3 — when the move itself
# fails), so `actions/upload-artifact`'s own `if: always()` step can never
# see an unscanned or dirty byte; the staging copy is deleted on every
# exit, unconditionally, once the trap has run; and this process explicitly
# waits for its own `tee`'d run log to finish writing (round 3 A3) before
# it actually exits.
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
# `ttl_hours`). `world`/`hosts`/`pod_count`/`gpu_count_per_pod` are the
# MEASURED shape read back from the cluster RunPod actually created (`GET
# /v2/clusters/{id}`'s own `compute` block, via `_rpc_parse_cluster_shape`)
# — never a hardcoded literal, and refused (97, wrong shape) before any
# member work starts if it disagrees with what this driver requested.
# Assembly itself REFUSES (named, never an artifact) when either rank's own
# `hostname`/`nccl_socket_ifname` is empty or `unknown`, or the two ranks
# report the SAME host (compared case-insensitively). A human reviews the
# pulled artifact and commits it under
# `crates/jammi-kernels/artifacts/cuda-runs/`.
#
# EXIT CONTRACT: 0 pass; 75 no cluster capacity (no data center at
# RP_CLUSTER_MIN_AVAILABILITY or better — a neutral provider condition); 76
# inactivity kill (a hung leg, watched across BOTH ranks' output); 77 wrong
# tree (a rank's own echoed PROVE_SHA disagreed with PROVE_EXPECT_SHA); 97
# wrong shape (the MEASURED cluster shape, read back right after create,
# disagrees with what this driver requested; or a member's launch-time
# read-back failed: `Pod.args` does not echo the shared entrypoint text, or
# NEITHER `ssh.direct` nor an overlay-ip proxy path is reachable for a
# member — see F2/F3); 124 budget cut (T-10m, with the per-phase
# wall-clock breakdown printed); else this driver's own post-run refusal,
# by name (a failed artifact pull, a failed id-secrecy scan — which also
# DESTROYS the carrier directory before this process exits — a missing
# `ens1` line in a rank's log). A refusal arm whose scan itself comes back
# CLEAN (P-A2: no id ever landed anywhere examinable) keeps its own named
# exit code and its run.log/rank logs at the upload path, unaltered.
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

# Which RunPod object type carries the two-HOST NCCL bootstrap: `cluster`
# (`POST /v2/clusters`, an INSTANT CLUSTER — near-zero capacity, see the
# module header) or `pods` (`POST /v2/pods` x2, joined by RunPod's Global
# Networking — the default, since ordinary pods provision reliably where
# clusters do not). Validated in the executed-only block below (never here
# — a sourced fixture must never `exit`); `RP_TWO_HOST_TRANSPORT_INVALID`
# is set instead so the refusal fires at the one place it is checked.
RP_TWO_HOST_TRANSPORT="${RP_TWO_HOST_TRANSPORT:-pods}"
case "$RP_TWO_HOST_TRANSPORT" in
  pods|cluster) : ;;
  *) RP_TWO_HOST_TRANSPORT_INVALID=1 ;;
esac

# The rented part's compute capability, DERIVED from the gpuTypeId (never a
# second literal): each member exports it as `CUDA_COMPUTE_CAP` and refuses
# to build when `nvidia-smi`'s own `compute_cap` disagrees (the tripwire in
# `_rpc_remote_script`), so the SASS this leg proves is the SASS the rented
# silicon runs. The domain is the sm_XX set `runpod_gpu_prove.sh` names
# (sm_80/86/89/90); a gpuTypeId outside it is refused HERE, before phase 0
# and before anything is rented (exit 2), never on the members after a
# cluster is billing. `$1`=gpuTypeId -> stdout: the bare numeric cap; rc 1
# when unknown.
_rpc_compute_cap_for_gpu_type() {
  case "${1:?_rpc_compute_cap_for_gpu_type needs a gpuTypeId}" in
    "NVIDIA A100-SXM4-80GB"|"NVIDIA A100 80GB PCIe"|"NVIDIA A100-PCIE-40GB") echo 80 ;; # sm_80 (Ampere floor)
    "NVIDIA A40"|"NVIDIA RTX A6000"|"NVIDIA RTX A5000") echo 86 ;;                       # sm_86 (Ampere workstation)
    "NVIDIA L4"|"NVIDIA L40S"|"NVIDIA L40"|"NVIDIA GeForce RTX 4090") echo 89 ;;         # sm_89 (Ada)
    "NVIDIA H100 80GB HBM3"|"NVIDIA H100 PCIe"|"NVIDIA H100 NVL"|"NVIDIA H200") echo 90 ;; # sm_90 (Hopper)
    *) return 1 ;;
  esac
}
# Empty when the gpuTypeId is outside the domain: the main path below
# refuses on it before phase 0 (a sourced fixture must never `exit`).
NATIVE_COMPUTE_CAP="$(_rpc_compute_cap_for_gpu_type "$RP_CLUSTER_GPU_TYPE" || true)"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

# The ONE place the two-host test's own name lives — a rename moves this
# line and nothing else. `test_gpu_cluster_lane.sh` re-derives the exact
# `cargo test` tuple from this value.
CLUSTER_TEST_FILTER="${CLUSTER_TEST_FILTER:-gang_nccl_two_hosts}"

# Where the two-host test writes its per-rank report on EACH member, and
# where this driver pulls both back to locally. The NCCL id rides NO path
# under this directory — see the module doc's "THE ID CROSSING".
CLUSTER_REMOTE_ARTIFACT_DIR="${RP_REMOTE_ROOT}/jammi-ai/.gang-artifact"
CLUSTER_ARTIFACT_DIR="${CLUSTER_ARTIFACT_DIR:-.gpu-pull/gpu-cluster}"
# The remote path rank 0 mints the id to and rank 1 reads it from — on
# EACH host's own filesystem (never shared storage; this driver is what
# ships it between the two).
CLUSTER_REMOTE_ID_FILE="${RP_REMOTE_ROOT}/nccl.id"

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

# The reachability wait loop, extracted into its own function (round 3
# A1) so it is sourceable and testable exactly like `_rpc_check_readback`
# above, which it calls. $1=cluster id $2=RP_SSH_WAIT_SECS $3=RP_TTL_HOURS.
# Polls `GET /v2/clusters/{id}/pods` every 5s until the deadline; the loop
# is satisfied early only when BOTH members carry a direct endpoint, and
# otherwise runs to the deadline and settles for what the last read back
# carried (rank 0 direct + rank 1 overlay-only = the F3 proxy fallback; rank
# 0 without a direct endpoint = refusal). On success prints ONE line: `<primary_host> <primary_port> <member_host> <member_port>
# <member_ip> <proxy_flag>` and returns 0 -- `proxy_flag` is `1` when the
# member's own `ssh.direct` was absent and its overlay `ip` is what the
# caller must proxy through (F3's no-public-port fallback), `0` when the
# member has its own direct endpoint. On any refusal, prints nothing to
# stdout, names the reason on stderr, and returns 97.
#
# A1 (round 3): a raw tally (`ok_count -ge RP_CLUSTER_POD_COUNT`) was
# satisfiable by the SAME rank appearing twice in one `pods` response (a
# duplicate/stale row RunPod's own listing is under no documented obligation
# never to return) without rank 1 ever having actually been seen at all --
# the loop would have broken "ready" on a cluster that is not. This driver's
# own shape is FIXED at exactly ranks {0, 1} (A4: podCount=2 is a literal,
# not a parameter), so readiness is tracked as two DISTINCT rank flags,
# never a count: the loop exits only when rank 0 AND rank 1 have EACH been
# read back READBACK_OK at least once, regardless of how many rows the
# response carries or in what order.
_rpc_wait_for_members_ready() {
  local wait_cluster_id="${1:?_rpc_wait_for_members_ready needs a cluster id}" \
        wait_ssh_secs="${2:?_rpc_wait_for_members_ready needs RP_SSH_WAIT_SECS}" \
        wait_ttl_hours="${3:?_rpc_wait_for_members_ready needs RP_TTL_HOURS}"
  local primary_host="" primary_port="" member_host="" member_port="" member_ip="" pods_body=""
  local rank0_seen=0 rank1_seen=0 resp status readback readback_rc mismatch
  local deadline_ssh=$(( SECONDS + wait_ssh_secs ))
  while [ "$SECONDS" -lt "$deadline_ssh" ]; do
    resp="$(_rp_rest GET "/v2/clusters/${wait_cluster_id}/pods")"
    status="$(printf '%s\n' "$resp" | head -n1)"
    pods_body="$(printf '%s\n' "$resp" | tail -n +2)"
    if [ "$status" = "200" ]; then
      readback="$(_rpc_check_readback "$pods_body" "$(_rp_entrypoint_setup "$wait_ttl_hours")")"
      readback_rc=$?
      if [ "$readback_rc" -eq 0 ]; then
        mismatch=0
        while IFS=' ' read -r _pid rank state ip dhost dport; do
          [ -n "$rank" ] || continue
          case "$state" in
            READBACK_ARGS_MISMATCH) mismatch=1 ;;
            READBACK_NO_SSH_PATH) : ;;
            READBACK_OK)
              if [ "$rank" = "0" ]; then
                rank0_seen=1
                primary_host="$dhost"; primary_port="$dport"
              elif [ "$rank" = "1" ]; then
                rank1_seen=1
                member_host="$dhost"; member_port="$dport"; member_ip="$ip"
              fi ;;
          esac
        done <<< "$readback"
        if [ "$mismatch" -eq 1 ]; then
          echo "::error::a member's Pod.args does not echo the shared entrypoint text -- refusing" >&2
          return 97
        fi
        # Ready only when rank 0 carries its DIRECT endpoint (it is the jump
        # host for F3's fallback and the scp/rsync peer, which the proxy path
        # cannot serve) and rank 1 has been read back at all; a member whose
        # overlay ip is assigned seconds after create while its `ssh.direct`
        # is still null is PROVISIONING, not ready -- keep polling until the
        # deadline instead of refusing on the first read (run 35127869122
        # refused 1 s into phase 4 on exactly that read).
        if [ "$rank0_seen" = "1" ] && [ "$rank1_seen" = "1" ] \
           && [ -n "$primary_host" ] && [ "$primary_host" != "-" ] \
           && [ -n "$member_host" ] && [ "$member_host" != "-" ]; then
          break
        fi
      fi
    fi
    sleep 5
  done
  if [ "$rank0_seen" != "1" ] || [ "$rank1_seen" != "1" ]; then
    echo "::error::not every member reached a usable ssh path (direct or overlay-proxy) within ${wait_ssh_secs}s (rank 0 seen: ${rank0_seen}, rank 1 seen: ${rank1_seen})" >&2
    return 97
  fi
  if [ "$primary_host" = "-" ] || [ -z "$primary_host" ]; then
    echo "::error::the primary (rank 0) member carries no direct ssh endpoint -- there is no jump host for the no-public-port fallback either" >&2
    return 97
  fi
  local proxy_flag=0
  if [ "$member_host" = "-" ] || [ -z "$member_host" ]; then
    if [ -z "$member_ip" ] || [ "$member_ip" = "-" ]; then
      echo "::error::the member carries neither a direct ssh endpoint nor an overlay ip -- no path reaches it" >&2
      return 97
    fi
    member_host="$member_ip"
    member_port=22
    proxy_flag=1
  fi
  printf '%s %s %s %s %s %s\n' "$primary_host" "$primary_port" "$member_host" "$member_port" "$member_ip" "$proxy_flag"
  return 0
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

# ─────────────────────────────────────────────────────────────────────────
# The `pods` transport's own pure helpers (RP_TWO_HOST_TRANSPORT=pods): two
# ORDINARY RunPod pods joined by Global Networking rather than an INSTANT
# CLUSTER. Every read below reuses `_rp_rest`/`_rpc_pick_data_centers` —
# never a second literal for the availability read the cluster path already
# performs.
# ─────────────────────────────────────────────────────────────────────────

# $1(stdin)=the raw `GET /v2/catalog/datacenters` response BODY. Prints a
# SPACE-SEPARATED, SORTED list of every data center id carrying
# `globalNetwork: true` — read LIVE at run time (never a hard-coded list;
# see the module doc for the snapshot this was verified against on
# 2026-09-16). Accepts either a bare list or an object carrying a
# `dataCenters` key (RunPod's documented REST v2 shape has varied across
# endpoints elsewhere in this file — `rp_cluster_pods` vs `rp_cluster_list`
# — so both shapes are read rather than assumed). Returns 0 on any
# successful parse (including zero GN data centers — "no capacity", read by
# the caller as SUPPLY_CONSTRAINT); 2 when the body could not be read as
# either documented shape at all.
_rpc_parse_global_network_datacenters() {
  python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE_ERROR: could not parse the datacenters response: %s" % e)
    sys.exit(2)
if isinstance(d, list):
    items = d
elif isinstance(d, dict) and isinstance(d.get("dataCenters"), list):
    items = d["dataCenters"]
else:
    print("PARSE_ERROR: datacenters response is neither a list nor an object carrying a dataCenters list")
    sys.exit(2)
out = []
for dc in items:
    if isinstance(dc, dict) and dc.get("globalNetwork") is True:
        did = dc.get("id")
        if did:
            out.append(did)
print(" ".join(sorted(out)))
'
}

# The co-placement intersection (P2): $1=space-separated candidate list A
# $2=space-separated candidate list B. Prints the SORTED intersection,
# space-separated (possibly empty). Pure set arithmetic — no parsing, no
# network — so the two upstream reads (pod availability, Global-Networking
# data centers) stay independently testable and this join is tested on its
# own.
_rpc_intersect_data_centers() {
  local a="${1:-}" b="${2:-}"
  python3 -c '
import sys
a = set(sys.argv[1].split())
b = set(sys.argv[2].split())
print(" ".join(sorted(a & b)))
' "$a" "$b"
}

# The pods-transport readback parser (P2), mirroring `_rpc_check_readback`'s
# own three-way judged shape for the cluster path's `Pod` list, but reading
# TWO independent `GET /v2/pods/{id}` bodies (one per rank — pods rented
# individually carry no shared `cluster.rank`/`cluster.ip` block at all,
# unlike a cluster member). $1=rank 0's raw Pod body $2=rank 1's raw Pod
# body. Prints ONE line per rank: `<podId> <rank> <status> <dataCenterId>
# <gn_enabled 0|1> <gn_ip_or_dash> <ssh_host_or_dash> <ssh_port_or_dash>`.
# Returns 0 when BOTH bodies parse (even when a pod is not yet RUNNING or
# not yet GN-enabled — the caller's own poll loop reads the per-rank
# fields); 2 when EITHER body could not be read as the documented `Pod`
# object shape at all.
_rpc_check_pods_readback() {
  local body0="${1:?_rpc_check_pods_readback needs rank 0 own Pod body}" \
        body1="${2:?_rpc_check_pods_readback needs rank 1 own Pod body}"
  python3 -c '
import json, sys

def parse(body, rank):
    try:
        d = json.loads(body)
    except Exception as e:
        return None, "PARSE_ERROR: could not parse rank %d Pod body: %s" % (rank, e)
    if not isinstance(d, dict):
        return None, "PARSE_ERROR: rank %d Pod body is not an object" % rank
    return d, None

for rank, body in ((0, sys.argv[1]), (1, sys.argv[2])):
    d, err = parse(body, rank)
    if err:
        print(err)
        sys.exit(2)
    pid = d.get("id") or "?"
    status = d.get("desiredStatus") or d.get("status") or "?"
    dc = d.get("dataCenterId") or "-"
    gn = d.get("globalNetworking") if isinstance(d.get("globalNetworking"), dict) else {}
    gn_enabled = "1" if gn.get("enabled") is True else "0"
    gn_ip = gn.get("ip") or "-"
    ssh = d.get("ssh") if isinstance(d.get("ssh"), dict) else {}
    ssh_direct = ssh.get("direct") if isinstance(ssh.get("direct"), dict) else {}
    shost = ssh_direct.get("host") or "-"
    sport = ssh_direct.get("port")
    sport = str(sport) if sport is not None else "-"
    print("%s %d %s %s %s %s %s %s" % (pid, rank, status, dc, gn_enabled, gn_ip, shost, sport))
' "$body0" "$body1"
}

# The pods-transport reachability wait loop (P2), mirroring
# `_rpc_wait_for_members_ready`'s own shape but over TWO independently
# polled pods rather than one cluster's member listing. Refuses (97) BEFORE
# any build starts when: either pod never reaches RUNNING within the
# window, either pod never reports `globalNetworking.enabled` with a GN ip,
# the two pods' OWN `dataCenterId` disagree (co-placement did not actually
# happen — measured, never trusted from the create-time request alone), or
# either pod carries no `ssh.direct` (the pods transport does not use the
# cluster path's overlay-ip proxy fallback: `scp`/`rsync` cannot ride
# through it either way, so a direct endpoint is required for BOTH pods,
# not merely the primary). $1=rank 0 pod id $2=rank 1 pod id
# $3=RP_SSH_WAIT_SECS. On success prints ONE line: `<host0> <port0> <host1>
# <port1> <dataCenterId> <gn_ip0> <gn_ip1>` and returns 0. On any refusal,
# prints nothing to stdout, names the reason on stderr, returns 97.
_rpc_wait_for_two_host_pods_ready() {
  local pod0_id="${1:?_rpc_wait_for_two_host_pods_ready needs rank 0 pod id}" \
        pod1_id="${2:?_rpc_wait_for_two_host_pods_ready needs rank 1 pod id}" \
        wait_secs="${3:?_rpc_wait_for_two_host_pods_ready needs RP_SSH_WAIT_SECS}"
  local deadline=$(( SECONDS + wait_secs ))
  local body0 body1 readback readback_rc
  local host0="" port0="" host1="" port1="" dc0="" dc1="" gn_ip0="" gn_ip1=""
  local ready0=0 ready1=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    body0="$(rp_pod_get "$pod0_id" 2>/dev/null)"
    body1="$(rp_pod_get "$pod1_id" 2>/dev/null)"
    if [ -n "$body0" ] && [ -n "$body1" ]; then
      readback="$(_rpc_check_pods_readback "$body0" "$body1")"
      readback_rc=$?
      if [ "$readback_rc" -eq 0 ]; then
        ready0=0; ready1=0
        while IFS=' ' read -r _pid rank status dc gn_en gn_ip shost sport; do
          [ -n "$rank" ] || continue
          if [ "$status" = "RUNNING" ] && [ "$gn_en" = "1" ] && [ "$gn_ip" != "-" ] && [ "$shost" != "-" ]; then
            if [ "$rank" = "0" ]; then
              ready0=1; host0="$shost"; port0="$sport"; dc0="$dc"; gn_ip0="$gn_ip"
            else
              ready1=1; host1="$shost"; port1="$sport"; dc1="$dc"; gn_ip1="$gn_ip"
            fi
          fi
        done <<< "$readback"
        [ "$ready0" -eq 1 ] && [ "$ready1" -eq 1 ] && break
      fi
    fi
    sleep 5
  done
  if [ "$ready0" -ne 1 ] || [ "$ready1" -ne 1 ]; then
    echo "::error::not every two-host pod reached RUNNING with Global Networking enabled and a direct ssh endpoint within ${wait_secs}s (rank 0 ready: ${ready0}, rank 1 ready: ${ready1})" >&2
    return 97
  fi
  if [ "$dc0" != "$dc1" ]; then
    echo "::error::the two pods landed in DIFFERENT data centers (rank 0: ${dc0}, rank 1: ${dc1}) -- co-placement failed; refusing before any build starts" >&2
    return 97
  fi
  printf '%s %s %s %s %s %s %s\n' "$host0" "$port0" "$host1" "$port1" "$dc0" "$gn_ip0" "$gn_ip1"
  return 0
}

# P4: reads a rank's own log for the `DERIVED_NCCL_IFACE=<iface>` marker the
# pods-transport remote script echoes right after deriving its interface
# from the Global-Networking ip (never before — a marker line the remote
# script never reaches is exactly "the derivation was never attempted or
# refused", read by the caller as an empty string, never a stale guess).
# $1=log path. Prints the interface name, or nothing when the marker is
# absent.
_rpc_parse_derived_iface() {
  local log="${1:?_rpc_parse_derived_iface needs a log path}"
  grep -oE '^DERIVED_NCCL_IFACE=.*$' "$log" 2>/dev/null | tail -1 | cut -d= -f2-
}

# P4's log-side proof, generalizing `_rpc_ens1_seen` to an ARBITRARY
# interface name (the pods transport's own derived one, never the cluster
# path's `ens1` literal). $1=a rank's own log file $2=the interface name to
# look for. Returns 0 when the log names that interface for NCCL's own
# NET/socket transport line; 1 when it does not (a missing line, or an
# empty $2, is this driver's own refusal, by name, never a silent pass).
_rpc_net_iface_seen() {
  local log="${1:?_rpc_net_iface_seen needs a log path}" iface="${2:-}"
  [ -n "$iface" ] || return 1
  grep -qF "NCCL INFO NET/Socket" "$log" 2>/dev/null && grep -q "NCCL INFO NET/Socket.*${iface}" "$log" 2>/dev/null
}

# P-C: the MEASURED cluster shape, read from the SAME `Cluster` object
# `rp_cluster_get` already returns (`GET /v2/clusters/{id}`, which echoes
# the create request's own `compute` block back) -- never the request-side
# RP_CLUSTER_POD_COUNT/RP_CLUSTER_GPU_COUNT_PER_POD literals re-asserted
# uninspected. $1(stdin)=the raw Cluster response body. Prints
# "podCount gpuCountPerPod" on a successful parse (both required, positive
# integers); returns 2 when the body does not carry that shape at all --
# closing the class the round-2 audit named: four literals duplicated into
# the assembled artifact made check_cuda_run_artifacts.py's own cross-field
# check (`_gang_check_cluster_shape`) a tautology.
_rpc_parse_cluster_shape() {
  python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE_ERROR: could not parse the cluster response: %s" % e, file=sys.stderr)
    sys.exit(2)
compute = d.get("compute") or {}
pod_count = compute.get("podCount")
gpu_count_per_pod = compute.get("gpuCountPerPod")
def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1
if not _pos_int(pod_count):
    print("PARSE_ERROR: compute.podCount missing or not a positive integer (%r)" % (pod_count,), file=sys.stderr)
    sys.exit(2)
if not _pos_int(gpu_count_per_pod):
    print("PARSE_ERROR: compute.gpuCountPerPod missing or not a positive integer (%r)" % (gpu_count_per_pod,), file=sys.stderr)
    sys.exit(2)
print("%d %d" % (pod_count, gpu_count_per_pod))
'
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

# Point 4 of the pods-transport design: the ONE place the two transports'
# remote text actually differs. $1=rank. Under `RP_TWO_HOST_TRANSPORT=
# cluster` prints EXACTLY the pre-existing literal (`export
# NCCL_SOCKET_IFNAME=ens1`) — byte-for-byte, so the cluster path's own text
# (and every fixture asserting it) is unchanged. Under `pods`, prints shell
# TEXT that DERIVES the interface ON THE REMOTE HOST, at run time, from that
# rank's own Global-Networking ip (`RP_TWO_HOST_GN_IP_<rank>`, set by the
# executed block once the readback wait resolves it — never a literal
# interface name): /proc/net/route lists every route as little-endian hex
# Destination/Mask per Iface; the interface whose route covers the GN ip by
# LONGEST prefix wins (the default route, mask 0, never matches); no match
# is a NAMED refusal (97), never a silent fall-through to some other
# interface. No iproute2 anywhere — the image ships none. Echoes `DERIVED_NCCL_IFACE=<iface>` right after a successful
# derivation so the LOCAL driver's own post-run proof
# (`_rpc_parse_derived_iface`/`_rpc_net_iface_seen`) reads which interface
# this rank actually used, never re-guessing it.
_rpc_two_host_iface_lines() {
  local rank="${1:?_rpc_two_host_iface_lines needs a rank}"
  if [ "${RP_TWO_HOST_TRANSPORT:-pods}" = "cluster" ]; then
    printf 'export NCCL_SOCKET_IFNAME=ens1\n'
    return 0
  fi
  local gn_ip_var="RP_TWO_HOST_GN_IP_${rank}"
  # The interface is read from the kernel's own route table (/proc/net/route:
  # little-endian hex Destination/Mask per Iface), never from iproute2 — the
  # CI image ships no `ip` binary (run 35166917277: both ranks refused with
  # "ip: command not found"). Longest matching prefix wins; the default
  # route (mask 0) never matches. RP_ROUTE_TABLE lets the lane suite feed a
  # fixture table; a member always reads the real one.
  cat <<IFACE
gn_ip="${!gn_ip_var}"
route_table="\${RP_ROUTE_TABLE:-/proc/net/route}"
IFS=. read -r _a _b _c _d <<< "\$gn_ip"
gn_le=\$(( (_d << 24) | (_c << 16) | (_b << 8) | _a ))
iface=""; best_mask=-1
while read -r _ifn _dest _gw _flags _ref _use _metric _mask _rest; do
  [ "\$_ifn" = "Iface" ] && continue
  [ -n "\$_mask" ] || continue
  _m=\$(( 16#\$_mask )); _dst=\$(( 16#\$_dest ))
  [ "\$_m" -eq 0 ] && continue
  if [ \$(( gn_le & _m )) -eq "\$_dst" ] && [ "\$_m" -gt "\$best_mask" ]; then
    best_mask=\$_m; iface="\$_ifn"
  fi
done < "\$route_table"
if [ -z "\$iface" ]; then
  echo "::error::no route in \$route_table covers the Global-Networking ip \$gn_ip -- refusing to derive NCCL_SOCKET_IFNAME" >&2
  exit 97
fi
echo "DERIVED_NCCL_IFACE=\$iface"
export NCCL_SOCKET_IFNAME="\$iface"
IFACE
}

# Every rank log the run produced ships in the artifact on EVERY exit arm —
# a refusal that leaves only "rank 0 ended (rc=97)" in run.log and the
# reason on a terminated pod is not a diagnosis (run 35166195386). The
# id-secrecy scan already covers `$CLUSTER_ARTIFACT_DIR`, so the logs are
# scanned like every other carrier before upload.
_rpc_ship_rank_logs() {
  mkdir -p "$CLUSTER_ARTIFACT_DIR"
  [ -n "${rank0_log:-}" ] && cp -f "$rank0_log" "${CLUSTER_ARTIFACT_DIR}/rank0.log" 2>/dev/null || echo "::warning::could not copy rank 0's own log into ${CLUSTER_ARTIFACT_DIR}"
  [ -n "${rank1_log:-}" ] && cp -f "$rank1_log" "${CLUSTER_ARTIFACT_DIR}/rank1.log" 2>/dev/null || echo "::warning::could not copy rank 1's own log into ${CLUSTER_ARTIFACT_DIR}"
}

_rpc_run_two_ranks() {
  local rc=0
  _rpc_run_two_ranks_inner "$@" || rc=$?
  _rpc_ship_rank_logs
  return "$rc"
}

# The size of rank 0's id file, read over ssh: the ONE probe the id crossing
# polls. A helper so the lane suite can drive `_rpc_run_two_ranks` with a
# fixture that mints the id after N polls.
_rpc_remote_id_size() { # $1=host $2=port
  ssh "${RP_SSHO[@]}" -p "$2" "root@${1}" "stat -c%s '${CLUSTER_REMOTE_ID_FILE}' 2>/dev/null"
}

# Run the two ranks, shared by BOTH transports (the same defect lived in two
# copies of this text). Both members clone and build CONCURRENTLY — only the
# PROOF needs the id — and the id crossing is a phase of the watch loop,
# bounded by the SAME three rules as everything else (rank 0's liveness and
# log growth within RP_INACTIVITY, the wrong-tree check, the T-10m budget),
# never by a wall clock unrelated to the work it waits on (run 35164486335:
# the crossing was bounded by RP_SSH_WAIT_SECS=300 s while rank 0's cold
# build alone takes far longer, so no cold run could ever pass). Rank 1's
# own remote script blocks between its build and its proof until the id file
# it was shipped holds exactly 128 bytes; the id still travels ONLY through
# the staging copy, after rank 0 minted it (`_rpc_remote_script`'s doc).
#   $1=primary host $2=primary port $3=member host $4=member port;
#   `member_extra_sshopts[@]` (global; empty under `pods`) rides on every
#   member call. Sets the globals the assembly step reads: rank0_log,
#   rank1_log, rank0_rc, rank1_rc, STAGING_ID_FILE, id_landed.
_rpc_run_two_ranks_inner() {
  local primary_host="${1:?_rpc_run_two_ranks needs the primary host}" \
        primary_port="${2:?_rpc_run_two_ranks needs the primary port}" \
        member_host="${3:?_rpc_run_two_ranks needs the member host}" \
        member_port="${4:?_rpc_run_two_ranks needs the member port}"
  _rpc_phase "build (both ranks, concurrently) + two-host proof"
  rank0_log="$(mktemp)"; rank1_log="$(mktemp)"
  STAGING_ID_FILE="$RP_WORK/nccl.id"
  # Every remote job imports PID 1's environment FIRST (RP_ENV_PREAMBLE, the
  # library's one definition, prepended exactly as rp_run_remote /
  # rp_run_remote_watched do): an sshd session does not inherit the
  # container's Dockerfile ENV, so without it the member builds with the
  # system gcc, which does not know the mold link flag the workspace's cargo
  # config asks for (run 35169035056: rank 0's build died at the linker).
  { printf '%s\n' "$RP_ENV_PREAMBLE"; _rpc_remote_script 0; } | ssh "${RP_SSHO[@]}" -p "$primary_port" "root@${primary_host}" "timeout ${RP_TIMEOUT:-3000} bash -s" > "$rank0_log" 2>&1 &
  rank0_pid=$!
  { printf '%s\n' "$RP_ENV_PREAMBLE"; _rpc_remote_script 1; } | ssh "${RP_SSHO[@]}" "${member_extra_sshopts[@]}" -p "$member_port" "root@${member_host}" "timeout ${RP_TIMEOUT:-3000} bash -s" > "$rank1_log" 2>&1 &
  rank1_pid=$!
  _rpc_phase "watching both ranks (id crossing + inactivity + wrong-tree + budget)"
  local last_growth=$SECONDS last_size0=0 last_size1=0 sz0 sz1 remote_size wrong_tree candidate_log line
  while kill -0 "$rank0_pid" 2>/dev/null || kill -0 "$rank1_pid" 2>/dev/null; do
    sleep 5
    if [ "${id_landed:-0}" -ne 1 ]; then
      if ! kill -0 "$rank0_pid" 2>/dev/null; then
        wait "$rank0_pid" 2>/dev/null; rank0_rc=$?
        echo "::error::rank 0 ended (rc=${rank0_rc}) before minting the 128-byte id file -- the proof never started (its log is in the artifact)"
        kill -TERM "$rank1_pid" 2>/dev/null; wait "$rank1_pid" 2>/dev/null
        return 76
      fi
      remote_size="$(_rpc_remote_id_size "$primary_host" "$primary_port")"
      if [ "$remote_size" = "128" ]; then
        id_landed=1
        scp "${RP_SSHO[@]}" -P "$primary_port" "root@${primary_host}:${CLUSTER_REMOTE_ID_FILE}" "$STAGING_ID_FILE" \
          || { echo "::error::could not scp the id down from the primary"; kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null; return 76; }
        chmod 600 "$STAGING_ID_FILE"
        _rpc_id_file_ready "$STAGING_ID_FILE" \
          || { echo "::error::the staged id copy is not exactly 128 bytes -- refusing to ship it"; kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null; return 76; }
        scp "${RP_SSHO[@]}" "${member_extra_sshopts[@]}" -P "$member_port" "$STAGING_ID_FILE" "root@${member_host}:${CLUSTER_REMOTE_ID_FILE}" \
          || { echo "::error::could not scp the id up to the member"; kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null; return 76; }
        echo "the id crossed to the member at $(( SECONDS ))s"
      fi
    fi
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
        return 77
      fi
    fi
    if [ $(( SECONDS - last_growth )) -ge "${RP_INACTIVITY}" ]; then
      echo "::error::inactivity: no new output on either rank's log for ${RP_INACTIVITY}s"
      kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null
      return 76
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
      echo "::error::budget cut at T-10m (per-phase breakdown above)"
      kill -TERM "$rank0_pid" "$rank1_pid" 2>/dev/null; wait "$rank0_pid" "$rank1_pid" 2>/dev/null
      return 124
    fi
  done
  wait "$rank0_pid"; rank0_rc=$?
  wait "$rank1_pid"; rank1_rc=$?
  if [ "${id_landed:-0}" -ne 1 ]; then
    echo "::error::both ranks ended (rank 0 rc=${rank0_rc}, rank 1 rc=${rank1_rc}) and the id never crossed"
    return 76
  fi
  return 0
}

# The shared per-rank remote script text (F13: the zero-test tripwire lives
# HERE, spliced via the pre-computed `${cluster_zero_test_tripwire}`
# variable — never a bare `$(...)` inside this heredoc). $1=rank (0|1).
# `world`/`pod_count`/`gpu_count_per_pod` derive from
# RP_CLUSTER_POD_COUNT/RP_CLUSTER_GPU_COUNT_PER_POD, never a second literal.
# The NCCL interface line itself is transport-specific
# (`_rpc_two_host_iface_lines`, spliced via the pre-computed
# `${two_host_iface_lines}` variable — same "never a bare `$(...)` inside
# this heredoc" rule F13 already states) — everything else is ONE script,
# shared, transport-agnostic.
_rpc_remote_script() {
  local rank="${1:?_rpc_remote_script needs a rank}"
  local two_host_iface_lines id_wait_lines="" remote_checkout_lines
  two_host_iface_lines="$(_rpc_two_host_iface_lines "$rank")"
  remote_checkout_lines="$(rp_remote_checkout_lines "${GIT_REF}" "${GIT_REPO}")"
  # Every rank but 0 blocks between its build and its proof until the id
  # rank 0 minted has been shipped to it (128 bytes exactly); the heartbeat
  # line keeps the driver's inactivity watchdog fed while it waits. Rank 0
  # mints the id inside its own proof and never waits.
  if [ "$rank" != "0" ]; then
    id_wait_lines='echo "=== id-wait: rank 1 built; waiting for the 128-byte id file rank 0 mints ==="
while [ "$(stat -c%s "${JAMMI_GANG_TWO_HOSTS_ID_FILE}" 2>/dev/null)" != "128" ]; do
  echo "id-wait: not yet (${SECONDS}s)"
  sleep 10
done
echo "=== id-wait: the id landed ==="'
  fi
  cat <<EOF
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (ledger row 17: no cross-target-dir reuse on this image)
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
export JAMMI_GANG_ARTIFACT_DIR=${CLUSTER_REMOTE_ARTIFACT_DIR}
export JAMMI_REQUIRE_CUDA_TWO_HOSTS=1
export JAMMI_GANG_TWO_HOSTS_RANK=${rank}
export JAMMI_GANG_TWO_HOSTS_WORLD=${RP_CLUSTER_POD_COUNT}
export JAMMI_GANG_TWO_HOSTS_ID_FILE=${CLUSTER_REMOTE_ID_FILE}
${two_host_iface_lines}
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
${remote_checkout_lines}
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

${id_wait_lines}
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
# $3=git_sha $4=box label $5=out path $6=pod_count (MEASURED, from
# `_rpc_parse_cluster_shape` under `cluster`, or the literal 2 under `pods`
# — both transports rent exactly 2 pods, 1 GPU each) $7=gpu_count_per_pod
# (MEASURED) $8=ttl_hours (optional, default 1) $9=transport — REQUIRED,
# exactly `instant-cluster` or `global-networking` (P7: a reader must never
# mistake one rental mechanism's run for the other's). Prints nothing;
# returns 0 on a successful write (regardless of the recorded verdict — a
# `fail` is representable, never refused at assembly time), 1 when EITHER
# rank report is missing or malformed, or `transport` is outside the closed
# set (an artifact with no evidence to assemble from, or an unrecorded
# rental mechanism, is refused outright, never synthesized), 2 on the named
# per-rank refusals below.
_rpc_assemble_gang_artifact() {
  local r0="${1:?needs rank-0.json}" r1="${2:?needs rank-1.json}" git_sha="${3:?needs git_sha}" \
        box="${4:?needs a box label}" out="${5:?needs an output path}" \
        pod_count="${6:?needs the MEASURED pod_count}" gpu_count_per_pod="${7:?needs the MEASURED gpu_count_per_pod}" \
        transport="${9:?needs the transport (instant-cluster|global-networking)}"
  case "$transport" in
    instant-cluster|global-networking) : ;;
    *) echo "::error::_rpc_assemble_gang_artifact: transport must be 'instant-cluster' or 'global-networking', got '${transport}'" >&2; return 1 ;;
  esac
  python3 -c '
import json, sys
r0_path, r1_path, git_sha, box, out_path, pod_count_s, gpu_count_per_pod_s = sys.argv[1:8]
transport = sys.argv[9] if len(sys.argv) > 9 else ""
try:
    r0 = json.load(open(r0_path))
    r1 = json.load(open(r1_path))
except Exception as e:
    print("could not read one or both rank reports: %s" % e, file=sys.stderr)
    sys.exit(1)
try:
    pod_count = int(pod_count_s)
    gpu_count_per_pod = int(gpu_count_per_pod_s)
    if pod_count < 1 or gpu_count_per_pod < 1:
        raise ValueError("must be >= 1")
except ValueError as e:
    print("could not read the measured pod_count/gpu_count_per_pod: %s" % e, file=sys.stderr)
    sys.exit(1)
reports = sorted([r0, r1], key=lambda r: r.get("rank", 0))

def _resolved(v):
    return isinstance(v, str) and v.strip() and v.strip().lower() != "unknown"

def _norm_host(v):
    # F4 advisory: compared case-insensitively (and stripped) on BOTH this
    # assembler and check_cuda_run_artifacts own duplicate-host check --
    # "Host-A" and "host-a" are the same host, never a false "two hosts".
    return (v or "").strip().casefold()

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
if _norm_host(reports[0].get("hostname")) == _norm_host(reports[1].get("hostname")):
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
        # P-C: derived from the MEASURED shape (the cluster own `compute`
        # block, threaded in from `_rpc_parse_cluster_shape`), never a
        # hardcoded literal -- a fixture that requests a different shape
        # gets a different artifact, and check_cuda_run_artifacts own
        # cross-field check (`_gang_check_cluster_shape`) is falsifiable
        # against it.
        "world": pod_count * gpu_count_per_pod,
        "collective": "nccl",
        "hosts": pod_count,
        "ranks": ranks,
        "reduced_vector_digest": reduced_digest,
        "verdict": verdict,
        "pod_count": pod_count,
        "gpu_count_per_pod": gpu_count_per_pod,
        "ttl_hours": int((sys.argv[8] if len(sys.argv) > 8 else "1")),
        # P7: which RENTAL MECHANISM produced this run -- a reader must
        # never mistake a Global-Networking run for an Instant-Cluster one.
        # Bash-side already refused a value outside the closed set before
        # this python process ever started; asserted again here is
        # deliberately redundant (belt), the checker (rule (k)) is the
        # authoritative gate (suspenders).
        "transport": transport,
    },
}
if verdict == "fail":
    artifact["gang"]["reason"] = reason
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)
    f.write("\n")
' "$r0" "$r1" "$git_sha" "$box" "$out" "$pod_count" "$gpu_count_per_pod" "${RP_TTL_HOURS}" "$transport"
}

# Wraps the id-secrecy scan (F5, `gang_id_secrecy_scan.py`) invocation in
# ONE place so both the real orchestration below and
# test_gpu_cluster_lane.sh's own fixtures call it identically. $1=staging
# id file $2=pulled artifact dir $3=run log $4=assembled artifact path, or
# the EMPTY STRING (P-A2: the scanner's own `--assembled-artifact` is
# optional -- pass it only when this run's own assembly step actually
# claims to have written that file; an empty $4 omits the flag entirely, so
# a refusal arm that never reached assembly is not penalised for a file it
# never promised). Deletes the staging copy ONLY on a clean (exit 0) scan.
_rpc_run_id_secrecy_scan() {
  local staging="${1:?needs the staging id file}" artifact_dir="${2:?needs the artifact dir}" \
        log="${3:?needs the run log}" assembled="${4:-}"
  local -a scan_args=(--staging-file "$staging" --artifact-dir "$artifact_dir" --log "$log" --delete-staging)
  [ -n "$assembled" ] && scan_args+=(--assembled-artifact "$assembled")
  python3 "$DIR/gang_id_secrecy_scan.py" "${scan_args[@]}"
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

# P-A: runs from the EXIT trap, on EVERY arm, once the id has landed on
# this runner (`id_landed=1`, set the moment the download from the primary
# is ATTEMPTED -- see the executed block below -- never only on the
# happy-path tail). Scans the WHOLE carrier directory (F5/M4) via the SAME
# `_rpc_run_id_secrecy_scan` the happy path already used.
#
# P-A2 (absent vs. unexaminable): the assembled artifact is passed to the
# scan ONLY when `assembly_ok=1` -- set (below, in the executed block)
# immediately after `_rpc_assemble_gang_artifact` itself returns 0, i.e.
# this run's own assembly step actually claims to have written a file at
# `$ASSEMBLED`. On a refusal arm that never reached assembly at all (a
# failed pull, a wrong-tree/inactivity/budget cut before assembly), or
# whose assembly step reached this phase and REFUSED (missing/malformed
# rank reports, an unresolved hostname/iface, a repeated host) and so never
# wrote anything, `assembly_ok` stays 0 and the scan is called with an
# EMPTY 4th argument -- the scanner does not require a file nobody
# promised, so a clean run.log/rank-logs on that arm reads CLEAN, not
# UNEXAMINABLE, and is never destroyed for lack of a file the run never
# claimed to produce. `$ASSEMBLED` itself always lives INSIDE
# `$CLUSTER_ARTIFACT_DIR`, so omitting the flag never widens what gets
# scanned: a stray/leaked file sitting at that path is still caught by the
# directory walk `_rpc_run_id_secrecy_scan` already performs.
#
# P-A3 (honest wording): a scan that is NOT clean has this driver DESTROY
# the entire carrier directory before this process exits, never merely
# "quarantine" it -- the relocation below exists ONLY so the upload step
# (which starts concurrently with, and could otherwise race, this trap's
# own deletion) can never see a half-removed directory; the relocation
# target lives under `$RP_WORK`, which the chained `rp_cleanup` call
# (below) unconditionally `rm -rf`s before this process exits. A CI runner
# torn down at process exit is not somewhere a human can later inspect
# anything, so this IS destruction, and every message below says so.
#
# Globals read: STAGING_ID_FILE, CLUSTER_ARTIFACT_DIR, RUN_LOG, ASSEMBLED,
# assembly_ok (all set early in the executed block, before any exit arm can
# fire, so they are stable by the time this runs regardless of which phase
# the process is exiting from). Returns 0 (clean, or nothing to scan yet)
# or the scan's own non-zero status (1 hit, 2 unexaminable) for the caller
# to join into the pending exit code.
#
# F2 (round 3): the ORIGINAL shape here moved the dirty carrier under
# `$RP_WORK` and left its actual deletion to `rp_cleanup`'s own conditional
# `rm -rf "$RP_WORK"` (only run when `RP_WORK_IS_TEMP=1`, which an exported
# `RP_SESSION` CLEARS -- `runpod_lib.sh:294`) -- so a run under `RP_SESSION`
# (exactly the interactive/resumed shape a maintainer's by-hand fallback
# uses) left the id-bearing directory ON DISK, past this process's own
# exit, while its own log line claimed "WILL BE DESTROYED". Fixed: this
# function destroys the relocated copy ITSELF, synchronously, unconditional
# on RP_SESSION/RP_WORK_IS_TEMP/rp_cleanup ever running at all -- the `mv`
# is still what avoids racing the upload step's own concurrent read of
# `$CLUSTER_ARTIFACT_DIR` (an atomic rename empties that path instantly);
# once relocated, nothing further reads the copy, so deleting it immediately
# after is not a second race, only a earlier, unconditional one. The log
# line is now PAST tense ("was destroyed"), stated truthfully.
_rpc_scan_or_destroy() {
  [ "${id_landed:-0}" = "1" ] || return 0
  local scan_rc assembled_for_scan=""
  [ "${assembly_ok:-0}" = "1" ] && assembled_for_scan="$ASSEMBLED"
  _rpc_run_id_secrecy_scan "$STAGING_ID_FILE" "$CLUSTER_ARTIFACT_DIR" "$RUN_LOG" "$assembled_for_scan"
  scan_rc=$?
  if [ "$scan_rc" -ne 0 ]; then
    local pending_destroy_dir="${RP_WORK:-${TMPDIR:-/tmp}}/gpu-cluster-destroy-$$"
    if [ -d "$CLUSTER_ARTIFACT_DIR" ] && mv "$CLUSTER_ARTIFACT_DIR" "$pending_destroy_dir" 2>/dev/null; then
      # The relocated directory's own run.log is the SAME open file this
      # process has been tee'ing into (mv preserves the inode) -- this line
      # therefore lands as the run log's own LAST line, never a second,
      # unscanned append to whatever remains (nothing remains) at the
      # uploaded path.
      echo "::error::id-secrecy scan was not clean (rc=${scan_rc}) -- the carrier directory was moved to ${pending_destroy_dir} (outside ${CLUSTER_ARTIFACT_DIR}) and destroyed there, now, unconditionally (never left to rp_cleanup's own RP_SESSION-conditional teardown -- F2); the upload step finds nothing there"
      # F2: destroyed HERE, synchronously -- never deferred to rp_cleanup's
      # own conditional `rm -rf "$RP_WORK"`, which a real RP_SESSION run
      # would skip entirely, leaving this exact directory on disk.
      rm -rf "${pending_destroy_dir:?}" 2>/dev/null
    else
      # F3: the ORIGINAL fallback globbed `/*` and `/.[!.]*` only -- `/*`
      # never matches a dotfile, and `.[!.]*` explicitly excludes any name
      # whose SECOND character is also `.`, so a name starting `..` (e.g.
      # `..leak`) followed by more characters matched NEITHER glob and
      # survived this in-place destroy untouched even though Python's own
      # `iterdir()` (the scanner's walk) sees it. `..?*` closes exactly
      # that gap: `..` followed by at least one more character, never the
      # bare `..` parent-directory entry itself (which has no third glob
      # character to match `?`).
      rm -rf "${CLUSTER_ARTIFACT_DIR:?}"/* "${CLUSTER_ARTIFACT_DIR:?}"/.[!.]* "${CLUSTER_ARTIFACT_DIR:?}"/..?* 2>/dev/null
      echo "::error::id-secrecy scan was not clean (rc=${scan_rc}) -- the carrier directory could not be relocated, so it was destroyed in place; nothing reaches the upload step"
    fi
  fi
  return "$scan_rc"
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
# F1 (round 3): this function is now registered on INT/TERM/HUP as well as
# EXIT (below) -- an untrapped SIGINT (what a CI runner's own cancellation
# sends first) previously terminated the process WITHOUT running an
# EXIT-only trap at all, so the `if: always()` upload step could publish
# whatever the pulled carrier directory held, unscanned. Each signal's own
# registration passes its conventional 128+n exit code explicitly ($1
# below) rather than relying on `$?` at trap-entry time to already carry
# it -- deterministic across bash versions/build configurations, never an
# assumption about what a signal leaves in `$?`.
_rpc_cleanup_cluster() {
  local rc="${1:-$?}"
  # Disarm immediately: a second signal (or a hang in this very function's
  # own REST calls below, e.g. `rp_cluster_delete` itself timing out) must
  # never re-enter this trap while it is already running.
  trap - EXIT INT TERM HUP

  # P-A / F1: the id-secrecy scan now runs FIRST, before either untimed
  # REST call below (`_rpc_self_remove_status`/`rp_cluster_delete`, both
  # through `_rp_rest`) -- round 3 found it sequenced THIRD, behind both,
  # so a hang in either could have delayed the ONE thing that must run on
  # every exit arm before any byte reaches the upload step.
  local scan_rc
  _rpc_scan_or_destroy
  scan_rc=$?
  [ "$scan_rc" -ne 0 ] && [ "$rc" -eq 0 ] && rc="$scan_rc"

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

  # P5 (pods transport): terminates BOTH pods on every exit arm, exactly
  # like the cluster branch above does for its one cluster -- a SIBLING
  # block, gated on the pods-transport's OWN globals
  # (`two_host_pod_id_<rank>`, set the moment each create call returns an
  # id, never only on the happy path) rather than on `RP_TWO_HOST_
  # TRANSPORT` itself, so the cluster branch above is untouched byte-for-
  # byte and this block is simply INERT (both variables unset) on the
  # cluster path. `rp_terminate` (runpod_lib.sh) is the SAME GraphQL
  # `podTerminate` mutation every other pod lane in this tree already uses
  # -- no new termination primitive for this transport.
  local two_host_pid two_host_leaked=0
  for two_host_pid in "${two_host_pod_id_0:-}" "${two_host_pod_id_1:-}"; do
    [ -n "$two_host_pid" ] || continue
    if rp_terminate "$two_host_pid" >/dev/null 2>&1; then
      echo "two-host pod ${two_host_pid} terminated by the driver's own EXIT trap"
    else
      echo "::error::LEAKED two-host pod ${two_host_pid}: could not terminate on exit -- gpu-reap.yml's own pod sweep is the backstop"
      two_host_leaked=1
    fi
  done
  [ "$two_host_leaked" -eq 1 ] && [ "$rc" -eq 0 ] && rc=1

  # F5 advisory: the staging id file is deleted on EVERY exit AFTER the
  # scan above, regardless of RP_SESSION/RP_WORK_IS_TEMP -- never left to
  # rp_cleanup's own conditional `rm -rf "$RP_WORK"` (which only fires when
  # RP_WORK_IS_TEMP=1), and never left behind for "inspection" on a dirty
  # scan either: a destroyed CI runner is not somewhere a human can inspect
  # anything, so unconditional deletion here is strictly safer than the
  # scanner's own generic (reusable) leave-it-for-a-caller default.
  [ -n "${STAGING_ID_FILE:-}" ] && rm -f "$STAGING_ID_FILE" 2>/dev/null

  rp_cleanup  # F2: chain the library's own EXIT cleanup (rm -rf "$RP_WORK" when RP_WORK_IS_TEMP=1 -- the ssh keypair).

  # A3 (round 3): `exec > >(tee -a "$RUN_LOG") 2>&1` (below, in the
  # executed-only block) forks `tee` as a background process-substitution
  # job that bash never implicitly waits for -- every line this trap
  # itself echoed above (including the scan's own DESTROY reason line)
  # could still be sitting unflushed in the pipe the instant this process
  # exits and the CI runner's next step starts reading `$RUN_LOG` off
  # disk. Closing the fds that feed `tee` (so it sees EOF and can finish
  # writing) and then explicitly `wait`ing on its own pid (captured right
  # after the `exec` line as `$_RPC_TEE_PID`) makes every byte this
  # process ever wrote durable on disk BEFORE the process that wrote them
  # actually exits.
  exec 1>&- 2>&-
  [ -n "${_RPC_TEE_PID:-}" ] && wait "$_RPC_TEE_PID" 2>/dev/null
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
# P-A: a STABLE path, computed once, here — before the tee even starts —
# so the EXIT trap's own scan-or-destroy (`_rpc_scan_or_destroy`) knows
# exactly where to look on EVERY exit arm, including one that fires long
# before assembly is ever reached. P-A2: on those arms `assembly_ok` stays
# 0 (below) and the scanner is never told to require this path at all, so
# the file simply not existing yet is NOT read as UNEXAMINABLE — see
# gang_id_secrecy_scan.py's own optional --assembled-artifact handling.
ASSEMBLED="${CLUSTER_ARTIFACT_DIR}/gang-cluster-$(date -u +%Y%m%d%H%M%S).json"
# P-A: whether the NCCL id has landed locally on this runner yet (never
# only "the happy path completed") -- set the moment the download from the
# primary is ATTEMPTED, below. The EXIT trap only ever scans once this is 1:
# before that point the id exists only on rank 0's own remote host, which
# this driver has not yet read the CONTENTS of (only its remote SIZE, via
# `stat` over ssh), so there is nothing local yet that could carry it.
id_landed=0
# P-A2: whether THIS run's own assembly step claims to have written
# $ASSEMBLED (never "the happy path completed" either -- a `pass` and a
# recorded `fail` verdict both set this to 1, since `_rpc_assemble_gang_
# artifact` returns 0 for either; only a refusal that never wrote a file
# at all leaves this 0). Set immediately after that call, below.
assembly_ok=0
exec > >(tee -a "$RUN_LOG") 2>&1   # F6: ONE tee'd stream for every byte this driver emits.
# A3: `$!` immediately after a process substitution captures ITS OWN pid (a
# bash-specific, but well-established, idiom) -- the cleanup trap explicitly
# `wait`s on this before the process finally exits, so `tee`'s own write is
# never racing this process's own termination.
_RPC_TEE_PID=$!

_rpc_phase() { echo "=== PHASE ($(( SECONDS )))s: $* ==="; }

DEADLINE=$(( SECONDS + RP_TTL_HOURS * 3600 - 600 ))  # T-10m budget cut (F16).

if [ -z "$NATIVE_COMPUTE_CAP" ]; then
  echo "::error::RP_CLUSTER_GPU_TYPE '${RP_CLUSTER_GPU_TYPE}' is outside this leg's domain (no compute capability mapping in _rpc_compute_cap_for_gpu_type: sm_80/86/89/90 parts only) -- refused before any rent"
  exit 2
fi
# P1: RP_TWO_HOST_TRANSPORT is validated at source time (above, before the
# sourced-execution guard) into RP_TWO_HOST_TRANSPORT_INVALID rather than
# exiting there directly (a sourced fixture must never exit) -- refused
# HERE, before any rent, on the one path that actually executes.
if [ -n "${RP_TWO_HOST_TRANSPORT_INVALID:-}" ]; then
  echo "::error::RP_TWO_HOST_TRANSPORT must be 'pods' or 'cluster', got '${RP_TWO_HOST_TRANSPORT}'"
  exit 2
fi
echo "two-host transport: ${RP_TWO_HOST_TRANSPORT}"

if [ "$RP_TWO_HOST_TRANSPORT" = "cluster" ]; then
# --------------------------------------------------------------------------- #
# The `cluster` transport (RunPod INSTANT CLUSTER, POST /v2/clusters) --
# UNCHANGED from before this transport existed (P1: byte-for-byte).
# --------------------------------------------------------------------------- #
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
#
# F1 (round 3): EXIT alone is not enough -- an untrapped SIGINT (what a CI
# runner sends first on cancellation) terminates a non-interactive bash
# process WITHOUT ever running an EXIT-only trap, skipping the id-secrecy
# scan (and the cluster delete) entirely. Registered on INT/TERM/HUP too,
# each passing its own conventional 128+n code explicitly so `_rpc_cleanup_
# cluster`'s own `rc` is deterministic regardless of what `$?` happens to
# hold at the moment a signal (rather than a normal `exit`) invokes it.
trap _rpc_cleanup_cluster EXIT
trap '_rpc_cleanup_cluster 129' HUP
trap '_rpc_cleanup_cluster 130' INT
trap '_rpc_cleanup_cluster 143' TERM

# P-C: the MEASURED shape, read back from the cluster RunPod actually
# created (never the request-side RP_CLUSTER_POD_COUNT/
# RP_CLUSTER_GPU_COUNT_PER_POD literals re-asserted uninspected) — this is
# also the wrong-shape refusal the module doc's own EXIT CONTRACT already
# names (97: "member count or GPU count != payload"), just not actually
# checked against a measurement until now.
cluster_body="$(rp_cluster_get "$cluster_id")" || { echo "::error::could not read back the created cluster's own shape"; exit 97; }
shape_line="$(printf '%s' "$cluster_body" | _rpc_parse_cluster_shape)" \
  || { echo "::error::the cluster's own GET response does not carry a valid compute.podCount/gpuCountPerPod -- refusing to trust the request-side literal instead"; exit 97; }
read -r MEASURED_POD_COUNT MEASURED_GPU_COUNT_PER_POD <<< "$shape_line"
if [ "$MEASURED_POD_COUNT" != "$RP_CLUSTER_POD_COUNT" ] || [ "$MEASURED_GPU_COUNT_PER_POD" != "$RP_CLUSTER_GPU_COUNT_PER_POD" ]; then
  echo "::error::RunPod granted a ${MEASURED_POD_COUNT}x${MEASURED_GPU_COUNT_PER_POD} cluster, not the ${RP_CLUSTER_POD_COUNT}x${RP_CLUSTER_GPU_COUNT_PER_POD} this driver requested -- refusing (wrong shape)"
  exit 97
fi
echo "measured cluster shape: ${MEASURED_POD_COUNT}x${MEASURED_GPU_COUNT_PER_POD}"

_rpc_phase "waiting for both members RUNNING with a usable ssh path"
members_line="$(_rpc_wait_for_members_ready "$cluster_id" "$RP_SSH_WAIT_SECS" "$RP_TTL_HOURS")"
wait_rc=$?
[ "$wait_rc" -eq 0 ] || exit "$wait_rc"
IFS=' ' read -r primary_host primary_port member_host member_port member_ip proxy_flag <<< "$members_line"
member_extra_sshopts=()
# F3's no-public-port fallback: `_rpc_wait_for_members_ready` already proxies
# the member's ssh endpoint through the overlay ip when its own `ssh.direct`
# was absent -- `proxy_flag=1` says so; `member_extra_sshopts` carries ONLY
# the proxy option (never a port flag, which differs between ssh's `-p` and
# scp/rsync's `-P` -- each call site below supplies its own port flag
# explicitly instead of one shared array with an embedded, flag-specific
# `-p`/`-P`).
[ "$proxy_flag" = "1" ] && member_extra_sshopts=(-o "ProxyJump=root@${primary_host}:${primary_port}")
_rpc_phase "waiting for sshd on both members (an executed connect per member)"
rp_wait_sshd "$primary_host" "$primary_port" "$RP_SSH_WAIT_SECS" "rank 0" || exit 76
rp_wait_sshd "$member_host" "$member_port" "$RP_SSH_WAIT_SECS" "rank 1" "${member_extra_sshopts[@]}" || exit 76

_rpc_run_two_ranks "$primary_host" "$primary_port" "$member_host" "$member_port" || exit $?

# F6 advisory: both ranks' own logs land in the uploaded/scanned directory
# too, pass or fail alike -- copied here, unconditionally, before any
# pass/fail branching below, rather than only on a path that might exit
# early.
mkdir -p "$CLUSTER_ARTIFACT_DIR"  # rank logs already shipped by _rpc_run_two_ranks, on every arm

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
# ASSEMBLED is the STABLE path computed at the very top of this block
# (before the tee even started) -- never recomputed here, so the EXIT
# trap's own scan-or-destroy (which reads the same global) always looks in
# the place this write actually lands. P-A2: `assembly_ok` is set to 1
# ONLY on a successful write here -- never on the two refusal arms below,
# each of which never writes $ASSEMBLED at all -- so the EXIT trap's scan
# never requires a file that this run did not claim to produce.
if [ -f "${CLUSTER_ARTIFACT_DIR}/rank-0.json" ] && [ -f "${CLUSTER_ARTIFACT_DIR}/rank-1.json" ]; then
  measured_sha="${PROVE_EXPECT_SHA:-$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)}"
  if _rpc_assemble_gang_artifact "${CLUSTER_ARTIFACT_DIR}/rank-0.json" "${CLUSTER_ARTIFACT_DIR}/rank-1.json" \
    "$measured_sha" "a100-sxm4-cluster" "$ASSEMBLED" \
    "$MEASURED_POD_COUNT" "$MEASURED_GPU_COUNT_PER_POD" "" "instant-cluster"; then
    assembly_ok=1
  else
    echo "::error::could not assemble the gang artifact" >&2
    [ "$rc" -eq 0 ] && rc=1
  fi
else
  echo "::error::one or both rank reports were not pulled -- cannot assemble the gang artifact" >&2
  [ "$rc" -eq 0 ] && rc=1
fi

# P-A: the id-secrecy scan itself now runs from the EXIT trap
# (`_rpc_scan_or_destroy`, via `_rpc_cleanup_cluster`) on EVERY exit arm --
# including this one, the natural fall-through below -- not only here on
# the happy-path tail. Nothing further to do in THIS phase; the trap reads
# the same STAGING_ID_FILE/CLUSTER_ARTIFACT_DIR/RUN_LOG/ASSEMBLED/
# assembly_ok globals this block has been populating all along.

_rpc_phase "billing read (F16)"
billing_resp="$(_rp_rest GET /v2/billing/clusters)"
echo "billing read status for this run window: $(printf '%s\n' "$billing_resp" | head -n1)"

else
# --------------------------------------------------------------------------- #
# The `pods` transport (two ORDINARY RunPod pods, joined by Global
# Networking) -- the default. Same phase shape as the cluster branch above,
# reusing every shared primitive it uses (`_rpc_pick_data_centers`,
# `rp_init`, `_rpc_remote_script`, `rp_cluster_rank_verdict`/`rp_cluster_
# verdict`, `_rpc_run_id_secrecy_scan`, `_rpc_assemble_gang_artifact`) --
# only the rent/readback/interface/cleanup primitives differ (point 4 of
# the design: "one script, transport-specific lines only").
# --------------------------------------------------------------------------- #
_rpc_phase "availability read (product=POD, cloud=SECURE)"
pod_avail_resp="$(_rp_rest GET "/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=1&cloud=SECURE")"
pod_avail_status="$(printf '%s\n' "$pod_avail_resp" | head -n1)"
pod_avail_body="$(printf '%s\n' "$pod_avail_resp" | tail -n +2)"
if [ "$pod_avail_status" != "200" ]; then
  echo "::error::pod catalog availability read failed (status ${pod_avail_status}) -- treating as no capacity"
  exit 75
fi
pod_dcs="$(printf '%s' "$pod_avail_body" | _rpc_pick_data_centers "$RP_CLUSTER_GPU_TYPE" "$RP_CLUSTER_MIN_AVAILABILITY")"
case "$pod_dcs" in
  PARSE_ERROR*) echo "::error::${pod_dcs}"; exit 75 ;;
esac

_rpc_phase "global-networking data center read"
gn_resp="$(_rp_rest GET /v2/catalog/datacenters)"
gn_status="$(printf '%s\n' "$gn_resp" | head -n1)"
gn_body="$(printf '%s\n' "$gn_resp" | tail -n +2)"
if [ "$gn_status" != "200" ]; then
  echo "::error::datacenters read failed (status ${gn_status}) -- treating as no capacity"
  exit 75
fi
gn_dcs="$(printf '%s' "$gn_body" | _rpc_parse_global_network_datacenters)"
case "$gn_dcs" in
  PARSE_ERROR*) echo "::error::${gn_dcs}"; exit 75 ;;
esac

# P2: co-placement is a PROPERTY, not a hope -- ONE data center for BOTH
# pods, picked from the intersection of "offers this GPU type at the
# floor" and "carries Global Networking", never left to two independent
# per-pod choices.
co_dcs="$(_rpc_intersect_data_centers "$pod_dcs" "$gn_dcs")"
if [ -z "$co_dcs" ]; then
  echo "::error::no data center offers both ${RP_CLUSTER_GPU_TYPE} at ${RP_CLUSTER_MIN_AVAILABILITY} or better AND Global Networking (SUPPLY_CONSTRAINT)"
  exit 75
fi
chosen_dc="$(printf '%s\n' "$co_dcs" | tr ' ' '\n' | sort | head -n1)"
echo "candidate co-located data center(s): ${co_dcs} -- chosen: ${chosen_dc}"

rp_init

# P5/P9: initialized BEFORE either create call and BEFORE the trap is
# registered, so the trap's own pods-transport cleanup block (in
# _rpc_cleanup_cluster, above) never reads an unset variable regardless of
# which phase this process exits from.
two_host_pod_id_0=""
two_host_pod_id_1=""
trap _rpc_cleanup_cluster EXIT
trap '_rpc_cleanup_cluster 129' HUP
trap '_rpc_cleanup_cluster 130' INT
trap '_rpc_cleanup_cluster 143' TERM

# P3: rank assignment is DETERMINISTIC and RECORDED -- the FIRST pod
# created is rank 0, the SECOND is rank 1, never the reverse and never
# inferred from anything RunPod returns.
_rpc_phase "pod create (rank 0)"
two_host_pod_id_0="$(rp_two_host_pod_create "$RP_CLUSTER_GPU_TYPE" "$chosen_dc" 0)" \
  || { echo "::error::two-host pod create (rank 0) failed"; exit 75; }
echo "pod ${two_host_pod_id_0} (rank 0) created in ${chosen_dc}"

_rpc_phase "pod create (rank 1)"
two_host_pod_id_1="$(rp_two_host_pod_create "$RP_CLUSTER_GPU_TYPE" "$chosen_dc" 1)" \
  || { echo "::error::two-host pod create (rank 1) failed"; exit 75; }
echo "pod ${two_host_pod_id_1} (rank 1) created in ${chosen_dc}"

_rpc_phase "waiting for both pods RUNNING with Global Networking + a direct ssh endpoint"
two_host_ready_line="$(_rpc_wait_for_two_host_pods_ready "$two_host_pod_id_0" "$two_host_pod_id_1" "$RP_SSH_WAIT_SECS")"
wait_rc=$?
[ "$wait_rc" -eq 0 ] || exit "$wait_rc"
IFS=' ' read -r primary_host primary_port member_host member_port measured_dc RP_TWO_HOST_GN_IP_0 RP_TWO_HOST_GN_IP_1 <<< "$two_host_ready_line"
echo "both pods RUNNING, GN-enabled, co-located in ${measured_dc} (rank 0 GN ip: ${RP_TWO_HOST_GN_IP_0}, rank 1 GN ip: ${RP_TWO_HOST_GN_IP_1})"
# P4: threaded through global variables `_rpc_remote_script`/`_rpc_two_host_
# iface_lines` already read (declared, not local, so they are visible to
# those functions defined earlier in this file) -- never a literal
# interface name.
export RP_TWO_HOST_GN_IP_0 RP_TWO_HOST_GN_IP_1
member_extra_sshopts=()  # the pods transport never proxies -- both pods carry their own direct endpoint (P2's own refusal above).
_rpc_phase "waiting for sshd on both pods (an executed connect per member)"
rp_wait_sshd "$primary_host" "$primary_port" "$RP_SSH_WAIT_SECS" "rank 0" || exit 76
rp_wait_sshd "$member_host" "$member_port" "$RP_SSH_WAIT_SECS" "rank 1" || exit 76

_rpc_run_two_ranks "$primary_host" "$primary_port" "$member_host" "$member_port" || exit $?

mkdir -p "$CLUSTER_ARTIFACT_DIR"  # rank logs already shipped by _rpc_run_two_ranks, on every arm

rp_cluster_rank_verdict "$rank0_rc" "$rank0_log"; rank0_final=$?
rp_cluster_rank_verdict "$rank1_rc" "$rank1_log"; rank1_final=$?
rp_cluster_verdict "$rank0_final" "$rank1_final"; rc=$?

# P4: "the derived interface's line" is this transport's own version of the
# cluster path's ens1 proof -- each rank's OWN derived interface (read from
# the `DERIVED_NCCL_IFACE=` marker it echoed), never a shared literal.
if [ "$rc" -eq 0 ]; then
  for two_host_check_rank in 0 1; do
    if [ "$two_host_check_rank" = "0" ]; then two_host_check_log="$rank0_log"; else two_host_check_log="$rank1_log"; fi
    two_host_iface="$(_rpc_parse_derived_iface "$two_host_check_log")"
    if [ -z "$two_host_iface" ] || ! _rpc_net_iface_seen "$two_host_check_log" "$two_host_iface"; then
      echo "::error::rank ${two_host_check_rank}'s log never named its own derived interface ('${two_host_iface:-<none>}') for the NCCL NET transport -- refusing to read this as a proof"
      rc=1
    fi
  done
fi

_rpc_phase "pulling both ranks' artifacts"
pull_rc=0
rsync -az -e "ssh ${RP_SSHO[*]} -p ${primary_port}" "root@${primary_host}:${CLUSTER_REMOTE_ARTIFACT_DIR}/" "${CLUSTER_ARTIFACT_DIR}/" || pull_rc=$?
rsync -az -e "ssh ${RP_SSHO[*]} -p ${member_port}" "root@${member_host}:${CLUSTER_REMOTE_ARTIFACT_DIR}/" "${CLUSTER_ARTIFACT_DIR}/" || pull_rc=$?
if [ "$pull_rc" -ne 0 ]; then
  echo "::error::artifact pull failed (rsync rc=${pull_rc}) -- a leg with no retrievable evidence proves nothing reviewable" >&2
  [ "$rc" -eq 0 ] && rc="$pull_rc"
fi

_rpc_phase "assembling the gang artifact"
if [ -f "${CLUSTER_ARTIFACT_DIR}/rank-0.json" ] && [ -f "${CLUSTER_ARTIFACT_DIR}/rank-1.json" ]; then
  measured_sha="${PROVE_EXPECT_SHA:-$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)}"
  if _rpc_assemble_gang_artifact "${CLUSTER_ARTIFACT_DIR}/rank-0.json" "${CLUSTER_ARTIFACT_DIR}/rank-1.json" \
    "$measured_sha" "a100-sxm4-two-host-pods" "$ASSEMBLED" \
    "$RP_CLUSTER_POD_COUNT" "$RP_CLUSTER_GPU_COUNT_PER_POD" "" "global-networking"; then
    assembly_ok=1
  else
    echo "::error::could not assemble the gang artifact" >&2
    [ "$rc" -eq 0 ] && rc=1
  fi
else
  echo "::error::one or both rank reports were not pulled -- cannot assemble the gang artifact" >&2
  [ "$rc" -eq 0 ] && rc=1
fi

_rpc_phase "billing read"
billing_resp="$(_rp_rest GET /v2/billing/pods 2>/dev/null || true)"
echo "billing read status for this run window: $(printf '%s\n' "$billing_resp" | head -n1)"

fi

echo "=== GPU cluster leg exit=${rc} ==="
exit "$rc"

fi # end sourced-execution guard
