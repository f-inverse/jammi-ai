#!/usr/bin/env bash
# needs: pyyaml
# GPU-topology-lane fixture suite. Mocks-only: no network, no GPU, no RunPod
# account, no pod. It drives the REAL objects rather than paraphrases of
# them: `runpod_gpu_topology.sh` is `source`d, not executed — its
# sourced-execution guard skips the rent/run flow — so `rp_topology_verdict`,
# `rp_topology_pick_type`, the group arrays and the lane's own defaults are
# the committed ones; the assembler and the artifact gate are the committed
# programs.
#
# Cases:
#   T0  sourcing the driver invokes no curl/ssh/scp/rsync — measured through
#       a PATH shim, not asserted in a comment.
#   T1  `rp_topology_verdict` over every rc arm: a clean pass; ssh 0 with a
#       missing marker; ssh 0 with a non-zero group; a cut (76) with no
#       `PROVE_EXIT`; an in-suite exit (97) returned verbatim; an abrupt cut
#       whose final line has no newline.
#   T2  every `PROVE_GROUP_RC name=` the driver writes literally names a
#       declared gating group, and every declared group is written.
#   T3  `rp_fleet_candidates` over the lane's own knobs and a catalog
#       fixture: every co-located
#       Global-Networking data center of every type within the rate ceiling,
#       in preference order; a type above the ceiling or without co-located
#       capacity is passed over; a named data center narrows the list; none
#       qualifying is 75; a type with no compute capability is refused (2)
#       before anything is rented.
#   T4  the cost bound is what the MECHANISM produces: both figures printed
#       in the driver's header and in `gpu-topology.yml` are re-derived from
#       the driver's own defaults (hosts x RP_GPU_COUNT x
#       TOPOLOGY_MAX_GPU_RATE x RP_TTL_HOURS, and the TTL plus
#       `gpu-reap.yml`'s sweep interval).
#   T5  `gpu-topology.yml`'s `on:` block is exactly dispatch + PR label —
#       no push, no workflow_call, no schedule — read through the one shared
#       `on:` reader.
#   T6  every `cargo test` the driver runs on a pod enables
#       `live-gpu-gang-tests`, the feature that compiles the tests its
#       filters select.
#   T7  the assembler over fixture runs: agreeing transports produce a
#       `pass` artifact that `check_cuda_run_artifacts.py`'s rule (k)
#       accepts; disagreeing curves, the wrong transport in a coordinator's
#       log, a gang that published only after a retry, and a gang loss
#       beyond ε each produce a `fail` naming why.
#
# Run: bash ci/scripts/test_gpu_topology_lane.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"
DRIVER="$DIR/runpod_gpu_topology.sh"
WORKFLOW="$REPO_ROOT/.github/workflows/gpu-topology.yml"
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

pass=0 fail=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; fail=$((fail + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# Source the driver in a subshell with a key present (rp_init is never
# reached) and every network tool shimmed to log its own invocation.
SHIM="$SANDBOX/shim"; mkdir -p "$SHIM"
for tool in curl ssh scp rsync; do
  printf '#!/usr/bin/env bash\necho %s >> "%s/calls"\nexit 1\n' "$tool" "$SANDBOX" > "$SHIM/$tool"
  chmod +x "$SHIM/$tool"
done
in_driver() {
  ( export PATH="$SHIM:$PATH" RUNPOD_API_KEY=fixture-key
    # shellcheck source=ci/scripts/runpod_gpu_topology.sh
    source "$DRIVER" >/dev/null 2>&1
    eval "$1" )
}

# --- T0 ---------------------------------------------------------------------
in_driver ":"
check "T0 sourcing the driver makes no network call" '[ ! -s "$SANDBOX/calls" ]'

# --- T1 ---------------------------------------------------------------------
LOG="$SANDBOX/log"
verdict() { # $1=raw rc, stdin=log text; prints the verdict rc
  cat > "$LOG"
  in_driver "rp_topology_verdict $1 '$LOG' a b 2>/dev/null; echo \$?"
}
got="$(printf 'PROVE_GROUP_RC name=a rc=0\nPROVE_GROUP_RC name=b rc=0\nPROVE_EXIT=0\n' | verdict 0)"
check "T1 a clean pass is 0" '[ "$got" = 0 ]'
got="$(printf 'PROVE_GROUP_RC name=a rc=0\nPROVE_EXIT=0\n' | verdict 0)"
check "T1 ssh 0 with a missing marker is 1" '[ "$got" = 1 ]'
got="$(printf 'PROVE_GROUP_RC name=a rc=0\nPROVE_GROUP_RC name=b rc=3\nPROVE_EXIT=0\n' | verdict 0)"
check "T1 ssh 0 with a non-zero group is 1" '[ "$got" = 1 ]'
got="$(printf 'PROVE_GROUP_RC name=a rc=0\n' | verdict 76)"
check "T1 a cut with no PROVE_EXIT keeps its 76" '[ "$got" = 76 ]'
got="$(printf 'PROVE_GROUP_RC name=a rc=97\nPROVE_EXIT=97\n' | verdict 97)"
check "T1 an in-suite exit is returned verbatim" '[ "$got" = 97 ]'
got="$(printf 'PROVE_GROUP_RC name=a rc=0\nPROVE_GROUP_RC name=b rc=0' | verdict 0)"
check "T1 an unterminated final marker still counts" '[ "$got" = 0 ]'

# --- T2 ---------------------------------------------------------------------
declared="$(in_driver 'printf "%s\n" "${HOST0_GROUPS[@]}" "${HOST1_GROUPS[@]}" "${FLEET_GROUPS[@]}"' | sort)"
# Literal markers, the `group_lines <name>` helper's argument, and the fleet
# phases' `fleet-${phase/cpu/inline}` over both phases.
written="$( { grep -oE 'PROVE_GROUP_RC name=[a-z0-9-]+ ' "$DRIVER" | sed -E 's/.*name=([a-z0-9-]+) /\1/'
             grep -oE 'group_lines [a-z0-9-]+' "$DRIVER" | awk '{print $2}'
             grep -q 'fleet-${1/cpu/inline}' "$DRIVER" && printf 'fleet-nccl\nfleet-inline\n'; } | sort -u)"
check "T2 the written groups are exactly the declared gating groups" '[ "$declared" = "$written" ]'

# --- T3 ---------------------------------------------------------------------
CATALOG='{"gpus":[
 {"id":"NVIDIA A40","price":{"secure":0.4},"dataCenters":[{"id":"DC-NOGN","availability":"HIGH"}]},
 {"id":"NVIDIA H100 NVL","price":{"secure":9.0},"dataCenters":[{"id":"DC-A","availability":"HIGH"}]},
 {"id":"NVIDIA RTX A6000","price":{"secure":0.53},"dataCenters":[{"id":"DC-A","availability":"LOW"},{"id":"DC-B","availability":"NONE"},{"id":"DC-C","availability":"HIGH"}]},
 {"id":"NVIDIA A100-SXM4-80GB","price":{"secure":1.59},"dataCenters":[{"id":"DC-C","availability":"LOW"}]}
]}'
pick() { # $1=candidate types [$2=named data center]; prints "<rc>|<stdout, one line>"
  in_driver "out=\"\$(rp_fleet_candidates '$CATALOG' 'DC-A DC-B DC-C' '$1' \"\$TOPOLOGY_MIN_AVAILABILITY\" \"\$TOPOLOGY_MAX_GPU_RATE\" '${2:-}' 2>/dev/null)\"; echo \"\$?|\$(echo \"\$out\" | tr '\n' ' ')\""
}
got="$(pick 'NVIDIA A40|NVIDIA H100 NVL|NVIDIA RTX A6000|NVIDIA A100-SXM4-80GB')"
check "T3 every co-located place within the ceiling, in preference order" '[ "$got" = "0|NVIDIA RTX A6000|0.53|DC-A NVIDIA RTX A6000|0.53|DC-C NVIDIA A100-SXM4-80GB|1.59|DC-C " ]'
got="$(pick 'NVIDIA RTX A6000|NVIDIA A100-SXM4-80GB' DC-C)"
check "T3 a named data center narrows the candidates" '[ "$got" = "0|NVIDIA RTX A6000|0.53|DC-C NVIDIA A100-SXM4-80GB|1.59|DC-C " ]'
got="$(pick 'NVIDIA A40|NVIDIA H100 NVL')"
check "T3 no qualifying type is 75" '[ "${got%%|*}" = 75 ]'
got="$(pick 'NVIDIA B200|NVIDIA RTX A6000')"
check "T3 a type with no compute capability is refused before renting" '[ "${got%%|*}" = 2 ]'

# --- T4 ---------------------------------------------------------------------
bound="$(in_driver 'python3 -c "import sys; h,g,r,t,s=map(float,sys.argv[1:]); print(\"%.2f %.2f\" % (h*g*r*t, h*g*r*(t+s)))" 2 "$RP_GPU_COUNT" "$TOPOLOGY_MAX_GPU_RATE" "$RP_TTL_HOURS" "$(grep -oE "\*/[0-9]+ \* \* \*" "'"$REPO_ROOT"'/.github/workflows/gpu-reap.yml" | grep -oE "[0-9]+" | head -1)"')"
read -r bound_i bound_ii <<< "$bound"
for f in "$DRIVER" "$WORKFLOW"; do
  check "T4 $(basename "$f") prints bound (i) \$${bound_i}" 'grep -q "\$${bound_i}" "$f"'
  check "T4 $(basename "$f") prints bound (ii) \$${bound_ii}" 'grep -q "\$${bound_ii}" "$f"'
done
got="$(in_driver 'echo "$RP_GPU_COUNT"')"
check "T4 the lane rents two GPUs per host" '[ "$got" = 2 ]'

# --- T5 ---------------------------------------------------------------------
keys="$(python3 -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print("\n".join(d.get("on", d.get(True)).keys()))' "$WORKFLOW" | sort | tr '\n' ' ')"
check "T5 gpu-topology.yml triggers only on dispatch and the PR label" '[ "$keys" = "pull_request workflow_dispatch " ]'

# --- T6 ---------------------------------------------------------------------
missing="$(grep -E '^cargo test ' "$DRIVER" | grep -v 'live-gpu-gang-tests' || true)" # tripwire-ok: no surviving line is the pass condition, asserted next.
runs="$(grep -cE '^cargo test ' "$DRIVER")"
check "T6 every pod-side cargo test enables live-gpu-gang-tests ($runs runs)" '[ -z "$missing" ] && [ "$runs" -ge 4 ]'

# --- T7 ---------------------------------------------------------------------
# One fixture fleet: two transports' records, the servers' logs, the
# one-host logs. `mutate` edits it per case.
fixture() { # $1=dir
  local d="$1"
  mkdir -p "$d/host0" "$d/host1"
  printf 'index, name, compute_cap, driver_version\n0, NVIDIA RTX A6000, 8.6, 570.1\n' > "$d/host0/device.csv"
  printf 'test gang_nccl::an_abort_ends_a_real_nccl_wait_on_a_rank_that_never_joins ... ok\n' > "$d/host0/one-host-device.log"
  printf 'test gpu::topology::every_gpu_topology_and_transport_publishes_the_same_adapter ... ok\n' > "$d/host0/one-host-product.log"
  python3 - "$d" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
def record(phase, reference):
    workers = {f"i{r}-{phase}": {"label": f"h{r // 2}-gpu{r % 2}", "host": f"pod{r // 2}"} for r in range(4)}
    gang = {"job_id": f"job-{phase}", "status": "completed", "claimed_by": f"i0-{phase}",
            "ranks": list(workers), "loss_curve": [0.9, 0.7, 0.6], "embeddings": [[0.1, 0.2]]}
    out = {"workers": workers, "gang": gang, "verdict": "pass", "reasons": []}
    if reference:
        out["reference"] = {"job_id": "job-ref", "status": "completed", "claimed_by": "i1-nccl",
                            "ranks": ["i1-nccl"], "loss_curve": [0.9, 0.70002, 0.6], "embeddings": [[0.1, 0.2]]}
    return out
for phase, transport in (("nccl", "Nccl"), ("cpu", "Inline")):
    (d / "host0" / f"fleet-{phase}.json").write_text(json.dumps(record(phase, phase == "nccl")))
    (d / "host0" / f"{phase}-h0-g0.log").write_text(
        f"INFO gang transport selected job_id=job-{phase} topology=\"peer\" world=4 transport={transport}\n"
        f"INFO coordinator attempt ended job_id=job-{phase} attempt=1 world=4 end=published end_ordinal=12\n"
        + ("NCCL version 2.23.4+cuda12.4\n" if phase == "nccl" else ""))
    (d / "host1" / f"{phase}-h1-g0.log").write_text("INFO member joined\n")
PY
}
assemble() { # $1=dir; prints "<rc> <verdict>" and leaves $1/topology.json
  python3 "$DIR/gpu_topology_assemble.py" "$1" --sha "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    --gpu-type "NVIDIA RTX A6000" --data-center DC-A --gpus-per-host 2 > "$1/topology.json" 2>/dev/null
  echo "$? $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["topology"]["verdict"])' "$1/topology.json")"
}
gate_accepts() { # $1=artifact
  python3 - "$1" "$DIR" "$REPO_ROOT" <<'PY'
import json, sys, pathlib
sys.path.insert(0, sys.argv[2])
import check_cuda_run_artifacts as g
data = json.load(open(sys.argv[1]))
failures = g.check_topology_artifact(data, "2026-01-01-topology-fixture.json", pathlib.Path(sys.argv[3]))
print("\n".join(failures))
sys.exit(1 if failures else 0)
PY
}

T7="$SANDBOX/t7-pass"; fixture "$T7"
got="$(assemble "$T7")"
check "T7 agreeing transports assemble a pass" '[ "$got" = "0 pass" ]'
check "T7 the pass artifact satisfies rule (k)" 'gate_accepts "$T7/topology.json" >/dev/null'

T7="$SANDBOX/t7-curves"; fixture "$T7"
python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["gang"]["loss_curve"][2]=0.61; json.dump(d,open(p,"w"))' "$T7/host0/fleet-cpu.json"
got="$(assemble "$T7")"
check "T7 disagreeing transports assemble a fail" '[ "$got" = "1 fail" ] && grep -q "loss curves differ" "$T7/topology.json"'
check "T7 the fail artifact still satisfies rule (k)" 'gate_accepts "$T7/topology.json" >/dev/null'

T7="$SANDBOX/t7-transport"; fixture "$T7"
sed -i.bak 's/transport=Nccl/transport=Inline/' "$T7/host0/nccl-h0-g0.log"
got="$(assemble "$T7")"
check "T7 an nccl phase whose coordinator selected inline is a fail" '[ "$got" = "1 fail" ] && grep -q "exactly one Nccl transport" "$T7/topology.json"'

T7="$SANDBOX/t7-retry"; fixture "$T7"
sed -i.bak '/attempt=1 world=4 end=published/d' "$T7/host0/nccl-h0-g0.log"
printf 'INFO coordinator attempt ended job_id=job-nccl attempt=1 world=4 end=the gang faulted\nINFO coordinator attempt ended job_id=job-nccl attempt=2 world=4 end=published\n' >> "$T7/host0/nccl-h0-g0.log"
got="$(assemble "$T7")"
check "T7 a gang that published only after a retry is a fail" '[ "$got" = "1 fail" ] && grep -q "publish on its first attempt" "$T7/topology.json"'

T7="$SANDBOX/t7-epsilon"; fixture "$T7"
python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["reference"]["loss_curve"][1]=0.71; json.dump(d,open(p,"w"))' "$T7/host0/fleet-nccl.json"
got="$(assemble "$T7")"
check "T7 a gang loss beyond ε of the single rank is a fail" '[ "$got" = "1 fail" ] && grep -q "strays from the single rank" "$T7/topology.json"'

echo "test_gpu_topology_lane: ${pass} passed, ${fail} failed"
[ "$fail" -eq 0 ]
