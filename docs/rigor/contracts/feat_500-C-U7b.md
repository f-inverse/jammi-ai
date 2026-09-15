# CONTRACT — feat/500-C-U7b: the RunPod cluster leg (2 hosts x 1 A100) proves the two-host NCCL bootstrap; the reap treats a cluster as its own object type

**Contract of record.** slug: `feat_500-C-U7b` — the committed mechanism contract
`ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` before this unit's
rigor record at `docs/rigor/feat_500-C-U7b.jsonl` (the lead's own export, landed separately)
satisfies that checker's disclosure requirement. This unit is **U7b-A2** in the decomposition
`docs/plans/67-distributed-training/UNITS.md § U7b` now states: **A1-pull** (the pod-tier smoke's
CI scaffolding — `gpu-gang.yml`, `runpod_gpu_gang.sh`, P7) merged already, as PR-B1 (contract
`docs/rigor/contracts/feat_500-PR-B1.md`); **A2** is this unit — the cluster leg, its own reap
arm, and P8; **A3** — a 6-hourly cron re-add on `gpu-gang.yml` — is a FUTURE, separately
authorized unit this contract does not build, made a reviewed, human-visible act (rather than a
silent default) by P8 below. `test_gpu_gang_lane.sh`'s G7 case (no `schedule:` key anywhere in
`gpu-gang.yml`) is UNCHANGED by this unit and stays green — this unit never touches that workflow.

Owner: **docs-ci** (dispatched to write this contract; commits c1-c3, the code itself, are
already committed on this branch and lead-verified green before this dispatch). Every citation
below was read directly against this worktree's tree at commit `ed3612e6` (branch
`feat/500-C-U7b`, five commits atop `10787947`/`f772e2e8`) — `grep -n`/direct file reads, never a
scratchpad working document's own line numbers.

## 0. What this unit is, in one paragraph

Two RunPod PODS on one RunPod CLUSTER (REST v2, `POST /v2/clusters`), one A100 each, joined over
the cluster's own private overlay network — the only place a real cross-HOST NCCL gang
(`ncclCommInitRank`, an out-of-band id file) is exercised before release; the pod-tier gang leg
(PR-B1) proves a two-DEVICE collective inside ONE pod (`ncclCommInitAll`) and cannot reach this
bootstrap at all. Six mechanisms: **M1** the REST v2 cluster primitives in `runpod_lib.sh` plus
the reap arm that now treats a cluster as its own object type; **M2** the two-host NCCL test body
(`gang_nccl.rs`); **M3** the driver, `ci/scripts/runpod_gpu_cluster.sh`; **M4** the id-secrecy
scan; **M5** `check_gpu_prove_once.py`'s new P8 arm (schedule visibility) plus the `RENTING_ROOTS`
derivation that makes the cluster driver visible to the existing P7 arm at all; **M6** the
artifact registry's leg discrimination in `check_cuda_run_artifacts.py`'s rule (k).

## 1. The pre-flight gap — stated honestly before anything else

**F2 of `CONTRACT-U7b.md` (the session's own working contract) calls for a ~$0.05 pod pre-flight
BEFORE any cluster run**: one `POST /v2/pods` (an RTX A5000 SECURE candidate, the SAME
`_rp_entrypoint_setup` string in `args`), read back `Pod.args` and `Pod.ssh.direct`, ssh in,
delete — to settle, on real hardware, whether RunPod's REST v2 `args` field actually reaches
`bash -c` on `RP_IMAGE` the way the GraphQL `dockerArgs` field measurably does (S4's own probe
measured the GraphQL path; REST v2's `args` field was never itself executed against a live pod
by anyone on this unit). **This pre-flight has NOT been executed.** The lead's own attempt to run
it was BLOCKED by the harness's permission classifier (a real-world transaction touching billed
infrastructure) — it needs a human to run it or to explicitly allow it. This is not a gap in the
driver's own design; it is a gap in what has been verified about RunPod's own API surface.

Until that pre-flight runs, this unit's actual guard is the driver's own LAUNCH-TIME READ-BACK
refusal, never an independent confirmation that the mechanism works: `_rpc_check_readback`
(`ci/scripts/runpod_gpu_cluster.sh::_rpc_check_readback`) reads `GET /v2/clusters/{id}/pods`
after create and refuses (exit 97) the moment either (a) a member's own `Pod.args` field does not
contain the exact entrypoint text this driver sent, or (b) neither `ssh.direct` nor a usable
overlay `ip` is present for it:

```
$ grep -n 'READBACK_ARGS_MISMATCH\|READBACK_NO_SSH_PATH\|READBACK_OK' ci/scripts/runpod_gpu_cluster.sh
273:        print("%s %s READBACK_ARGS_MISMATCH %s %s %s" % (pid, rank, ip, dhost, dport))
276:        print("%s %s READBACK_NO_SSH_PATH %s %s %s" % (pid, rank, ip, dhost, dport))
278:        print("%s %s READBACK_OK %s %s %s" % (pid, rank, ip, dhost, dport))
```

**This read-back is a REFUSAL mechanism, not a proof that the API behaves as documented** — it
can only ever catch the failure AFTER a real cluster has already been created and billed for the
time it took to observe the mismatch (bounded by `RP_SSH_WAIT_SECS=300`s). Nothing in this unit's
own committed text claims the pre-flight ran, or that the REST v2 `args` field is confirmed to
reach the container. **§2 below — "the one real cluster run" the plan document schedules at the
end of commit c3 — has also NOT happened.** No real RunPod cluster has been created, no real cost
has been billed, and no real answer exists yet to any of the questions a real run would settle:
whether member self-removal works on a cluster object (S4 only measured that members expose
`actions: []`, a LISTING attribute — never an OBSERVED refusal or success of a real termination
attempt), whether the `ens1` NCCL transport line actually appears in a real run's log, and
whether the two-host id ship (`scp` through a local staging copy) actually completes within the
driver's own wait budgets. Every cost figure this contract cites below (`$3.816/h`, `$3.82`/run,
`$26.71` worst-case) is a COMMITTED, re-derived-by-test FIGURE — never a measured bill from an
actual run.

## 2. M1 — cluster primitives in `runpod_lib.sh`, REST v2, and the reap's fail-closed member exclusion

`_rp_rest` (`ci/scripts/runpod_lib.sh::_rp_rest`) is the ONE REST v2 transport every cluster
primitive goes through — Bearer auth, capture-then-parse via a temp file (never a pipe under
`pipefail`, so a transport failure's own exit code is never aliased against a parse failure's):

```
$ sed -n '369,386p' ci/scripts/runpod_lib.sh
```
```
_rp_rest() {
  local method="${1:?_rp_rest needs a METHOD}" path="${2:?_rp_rest needs a PATH}" body="${3-}"
  local body_file status rc
  body_file="$(mktemp "${TMPDIR:-/tmp}/jammi-rp-rest.XXXXXX")" \
    || { echo "::error::_rp_rest could not create a capture file" >&2; return 1; }
  if [ -n "$body" ]; then
    status="$(curl -s -o "$body_file" -w '%{http_code}' -X "$method" "https://api.runpod.io${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H 'Content-Type: application/json' --data-binary "$body")"
  else
    status="$(curl -s -o "$body_file" -w '%{http_code}' -X "$method" "https://api.runpod.io${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}")"
  fi
  rc=$?
  printf '%s\n' "$status"
  cat "$body_file"
  rm -f "$body_file"
  return "$rc"
}
```

`rc` is `_rp_rest`'s own TRANSPORT verdict (curl's exit code); `status` is the HTTP status printed
on the first line, read separately by every caller (F14) — a caller must never conflate a
nonzero `rc` with "not found" or "refused"; those are statuses on a SUCCESSFUL transport. Every
`rp_cluster_*` function (`create`/`get`/`pods`/`list`/`delete`) follows the identical
capture-then-split-then-case-on-status shape, each documenting its own success status (`201`
create, `200` get/pods/list, `204` empty-body delete) and refusing any other status by name with
the response body's own text, truncated.

`_rp_entrypoint_setup` (`ci/scripts/runpod_lib.sh::_rp_entrypoint_setup`) is the ONE watchdog+sshd
string shared by BOTH the pod payload builder (`_rp_deploy_payload`) and the cluster payload
builder (`_rp_cluster_payload`) — verified by direct read that both callers pass the SAME
function with only `RP_TTL_HOURS` differing:

```
$ grep -n '_rp_entrypoint_setup "\$RP_TTL_HOURS"' ci/scripts/runpod_lib.sh
1393:  setup="$(_rp_entrypoint_setup "$RP_TTL_HOURS")" || return 1
1481:  setup="$(_rp_entrypoint_setup "$RP_TTL_HOURS")" || return 1
```

the first call site (1393) is inside `_rp_deploy_payload`, the second (1481) inside
`_rp_cluster_payload` — the exact "two subtly different kill-this-thing mechanisms drifting
apart unnoticed" class this factoring closes, per the function's own module-header comment
(`ci/scripts/runpod_lib.sh`, the "Cluster primitives" section header immediately above
`_rp_cluster_payload`). `test_runpod_cluster_lib.sh`'s own module doc names this as covered
property (2): "`_rp_entrypoint_setup`'s output is byte-identical whether reached via
`_rp_deploy_payload` ... or `_rp_cluster_payload` ... for the same RP_TTL_HOURS".

**`_rp_cluster_payload`** (`ci/scripts/runpod_lib.sh::_rp_cluster_payload`) builds
`CreateClusterRequest`'s exact documented key set — `unevaluatedProperties: false` per RunPod's
own schema (read 2026-09-14) means any extra key is an outright API rejection:

```
$ sed -n '1483,1498p' ci/scripts/runpod_lib.sh
```
```
import json, sys
gpu, image, pub, ttl_h, prefix, disk_gb, setup, dcs = sys.argv[1:9]
body = {
    "name": "%s-ttl%s" % (prefix, ttl_h),
    "type": "TRAINING",
    "compute": {"gpuTypeId": gpu, "gpuCountPerPod": 1, "podCount": 2},
    "image": image,
    # REST v2's env shape is an OBJECT (key -> value), unlike the GraphQL pod
    # payload's array-of-{key,value} — see BaseContainerConfig.
    "env": {"PUBLIC_KEY": pub},
    "ports": ["22/tcp"],
    "args": "bash -c '%s'" % setup,
    "disk": int(disk_gb),
}
if dcs:
    body["dataCenterIds"] = dcs.split()
```

— exactly the keys the session's own live schema read named, no more; `dataCenterIds` is omitted
entirely (never sent as an empty list) when the driver has no qualifying data center, matching
the schema's own documented "let the scheduler choose" default (never exercised in this unit's
own driver, which always supplies at least one — see M3 below — but preserved here as the
primitive's own honest behavior for any OTHER future caller).

**F7 — one name.** `RP_CLUSTER_PREFIX="jammi-cluster"` is a SEPARATE name space from
`RP_POD_PREFIX="jammi-gpu"` (verified: `grep -c cluster ci/scripts/runpod_lib.sh` at the file's
own header comment states the earlier `${RP_POD_PREFIX}-cluster-...` wording is deleted):

```
$ grep -n 'RP_POD_PREFIX=\|RP_CLUSTER_PREFIX=' ci/scripts/runpod_lib.sh
115:RP_POD_PREFIX="jammi-gpu"
126:RP_CLUSTER_PREFIX="jammi-cluster"
```

so the pod sweep's prefix match (`name.startswith(RP_POD_PREFIX)`) can never accidentally match a
cluster's own name, and vice versa — the two object types' own name spaces cannot collide by
construction, never by convention alone.

**`rp_cluster_sweep`** (`ci/scripts/runpod_lib.sh::rp_cluster_sweep`) is fail-closed on every
enumeration it needs: a failed `GET /v2/clusters` is `return 1` naming "sweep could NOT enumerate
clusters; orphans may exist unseen" (never "nothing to reap"); an unparseable body is the same;
an `UNAGEABLE` cluster (no usable `createdAt`) is reported and skipped, never force-terminated
blind; the post-delete re-enumeration is fail-closed the identical way — a failed re-GET, or a
deleted cluster still present in the second listing, is `return 1` by name, never silently
trusted as "the delete must have worked":

```
$ grep -n 'could NOT enumerate clusters\|still present after delete\|could NOT confirm cluster' ci/scripts/runpod_lib.sh
1738:    || { echo "::error::sweep could NOT enumerate clusters; orphans may exist unseen: RunPod REST request failed"; return 1; }
1742:    echo "::error::sweep could NOT enumerate clusters; orphans may exist unseen: status ${status}: $(printf '%s' "$body" | head -c 300)"
1783:    echo "::error::sweep could NOT enumerate clusters; orphans may exist unseen: ${out}"
1807:      || { echo "::error::sweep could NOT re-enumerate clusters after deleting; cannot confirm ${n} deletion(s) took"; return 1; }
1827:        echo "::error::cluster ${id} still present after delete"
1830:        echo "::error::sweep could NOT confirm cluster ${id} is gone: malformed re-enumeration body"
```

**F8 — the reap subject model, restated for clusters and shared with the pod sweep.**
`_rp_cluster_member_ids` (`ci/scripts/runpod_lib.sh::_rp_cluster_member_ids`) enumerates every
`jammi-cluster`-prefixed cluster and every one's member pod ids; a failure at EITHER level (the
cluster list, or any one cluster's own pod list) is `return 1` with NOTHING printed — an
INCOMPLETE exclusion set is worse than none. `rp_sweep` (`ci/scripts/runpod_lib.sh::rp_sweep`)
consults this BEFORE it enumerates pods at all, and a failure there skips the ENTIRE pod sweep —
not even an ordinary, genuinely orphaned pod is touched that run:

```
$ grep -n 'sweep could NOT enumerate cluster members; pod sweep skipped' ci/scripts/runpod_lib.sh
2556:    || { echo "::error::sweep could NOT enumerate cluster members; pod sweep skipped: could not create a capture file"; return 1; }
2558:    || { rm -f "$member_ids_file"; echo "::error::sweep could NOT enumerate cluster members; pod sweep skipped: could not create a capture file"; return 1; }
2565:    echo "::error::sweep could NOT enumerate cluster members; pod sweep skipped: ${member_err:-unknown reason}"
```

A live member that DOES pass through as a pod-sweep candidate (an enumeration race, or a bug) is
skipped by name, never terminated:

```
$ sed -n '2639,2642p' ci/scripts/runpod_lib.sh
```
```
    if [ -n "$member_ids" ] && printf '%s\n' "$member_ids" | grep -qx -- "$id"; then
      echo "cluster member, skipped: ${id}"
      continue
    fi
```

and a `podTerminate` whose GraphQL body carries `errors` is now a NAMED, non-fatal, COUNTED
outcome (`"terminate refused: <reason>"`) rather than silently folded into "swept" — the sweep's
own final line separately tallies terminated vs. refused (`ci/scripts/runpod_lib.sh::rp_sweep`,
the `echo "sweep: terminated ${n} orphaned pod(s) (${refused} terminate(s) refused)"` line). This
matters specifically BECAUSE member self-removal on a cluster pod's own accounting is UNMEASURED
(see §1): the exclusion set is belt-and-suspenders on TOP of an assumed API refusal, never the
only thing standing between the sweep and a live member.

`gpu-dev.sh reap` (`ci/scripts/gpu-dev.sh`, the `reap)` case arm) runs BOTH sweeps
unconditionally and fails non-zero if EITHER could not enumerate its own object type, even when
the other succeeded — verified by direct read:

```
$ sed -n '265,282p' ci/scripts/gpu-dev.sh
```
```
  reap)
    # shellcheck source=ci/scripts/runpod_lib.sh
    source "$DIR/runpod_lib.sh"
    ...
    pod_rc=0 cluster_rc=0
    rp_sweep "${1:-}" || pod_rc=$?
    rp_cluster_sweep "${1:-}" || cluster_rc=$?
    [ "$pod_rc" -eq 0 ] && [ "$cluster_rc" -eq 0 ] && exit 0
    exit $(( pod_rc != 0 ? pod_rc : cluster_rc ))
    ;;
```

`.github/workflows/gpu-reap.yml` invokes this arm every 6 hours (`cron: "23 */6 * * *"`, offset
off the hour) plus `workflow_dispatch` with an optional `force_hours` input — this is the PRIMARY
backstop for clusters (never merely secondary the way it is for pods), since member self-removal
is unmeasured for a cluster object.

**F10 (the shared `-ttl<H>` name parser and the shared `force_hours` validator).**
`_rp_ttl_parser_pysrc` (`ci/scripts/runpod_lib.sh::_rp_ttl_parser_pysrc`) and
`_rp_validate_force_hours` (`ci/scripts/runpod_lib.sh::_rp_validate_force_hours`) are each ONE
definition consumed by both `rp_sweep` and `rp_cluster_sweep` — "0"/"00" are refused (rc 2) rather
than accepted as a vacuous force-reap for EITHER object type, closing the same class for clusters
from day one that pods needed an incident to close (the function's own doc comment names the
round-4 pod audit finding it generalizes).

**Oracle:** `ci/scripts/test_runpod_cluster_lib.sh` (mocks-only, `_rp_rest` shimmed via a stub
`curl`) drives: the create body's exact key set against a reviewed schema fixture; the
entrypoint-text byte-identity between the pod and cluster payload builders; `rp_cluster_sweep`'s
over-TTL-deleted / under-TTL-kept / non-jammi-name-untouched / failed-GET-rc1 /
unparseable-body-rc1 / failed-DELETE-rc1-by-id / 204-success arms, including the fail-closed
post-delete re-enumeration; `_rp_validate_force_hours`'s "0"/"00"-refuse, leading-zero-accept,
non-digit-refuse, and overflow-refuse arms; and `rp_sweep`'s cluster-member-exclusion skip plus
the counted terminate-refused arm. `test_gpu_dev_lifecycle.sh` covers `gpu-dev.sh reap`'s own
wiring — an empty account (no clusters, no pods) sweeps clean on both arms, and `reap 0`/`reap
00` refuse (rc 2) rather than mass-terminating (its own module doc's covered-property list,
items 1-7, cites this explicitly for the reap command).

## 3. M2 — the two-host NCCL test body (`gang_nccl.rs`)

`gang_nccl_two_hosts_reduce_a_known_vector`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs::gang_nccl_two_hosts_reduce_a_known_vector`)
is feature-gated and reads its own, SEPARATE env contract from the pod-tier single-process test:
`JAMMI_GANG_TWO_HOSTS_RANK`, `JAMMI_GANG_TWO_HOSTS_WORLD` (must be exactly `2`; any other value is
a named panic, never a silent truncation), `JAMMI_GANG_TWO_HOSTS_ID_FILE`, and its OWN require
flag `JAMMI_REQUIRE_CUDA_TWO_HOSTS` (distinct from the pod leg's `JAMMI_REQUIRE_CUDA_GANG` —
verified: `grep -n JAMMI_REQUIRE_CUDA_TWO_HOSTS crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`
shows this is consulted BEFORE `skip_without_gpu!`-style skipping, so a device-less member on
this leg hard-fails rather than silently skipping — the F12 fold). Rank 0 mints
`Nccl::new_id()` and writes it ATOMICALLY (`write_id_file_atomically`: write to `<file>.tmp`,
`rename`) so no reader can ever observe a partial write; rank 1 refuses any file whose size is not
exactly 128 bytes (`read_id_file_exactly_128_bytes`) — the F11 fold, mirrored on the driver's own
shipping side by `_rpc_id_file_ready`/`stat`. Both ranks then run the SAME three checks the
single-process pod-leg test runs (`assert_gang_checks`, shared between both tests) over a REAL
cross-host `Nccl::from_rank` communicator, and each writes its own `rank-<r>.json` report via
`write_rank_report`.

**P-M2.** Off a cluster the test skips loudly with the reason (never `#[ignore]`, never a vacuous
pass — the same require-vs-skip doctrine every gated GPU test in this repo follows); a report is
written on both the pass and the fail arm. The id bytes themselves are never written anywhere but
the id file this driver ships out of band — the test process itself has no code path that copies
them into `JAMMI_GANG_ARTIFACT_DIR` or any report field.

**Oracle:** this construct compiles under `cargo clippy -p jammi-ai --features live-gpu-tests
--test gpu_capability` (the gated-surface clippy step; not itself proof the test PASSES, only
that it compiles under the feature gate this repo's CI matrix exercises). The real proof is the
executed run in §1/§8 below — NOT yet performed.

## 4. M3 — the cluster driver `ci/scripts/runpod_gpu_cluster.sh`

Never `runpod_gpu_gang.sh` — a fully separate driver, workflow, and RunPod object type (verified:
`grep -n runpod_gpu_gang.sh ci/scripts/runpod_gpu_cluster.sh` finds no match; the driver's own
module doc states this explicitly at its top). Sequence: read per-data-center availability
(`GET /v2/catalog/gpus?include=AVAILABILITY&product=CLUSTER&count=1&cloud=SECURE`) and pass only
data centers at `RP_CLUSTER_MIN_AVAILABILITY` (`MEDIUM`) or better as `dataCenterIds` — never left
to the scheduler, since co-placement needs exactly ONE data center and the account-wide figure
alone cannot establish that any single one qualifies (the A1 fold); create ONE 2x1 cluster
(`MAX_ATTEMPTS=1` inside the driver's own retry loop lives in the WORKFLOW, `.github/workflows/
gpu-cluster.yml`, never doubled here); poll both members RUNNING with a usable ssh path via
`_rpc_check_readback` (§1 above); build both members in parallel via ONE shared per-rank heredoc
function, `_rpc_remote_script` (`ci/scripts/runpod_gpu_cluster.sh::_rpc_remote_script`) — `world`,
`pod_count`, and `gpu_count_per_pod` all derive from `RP_CLUSTER_POD_COUNT`/
`RP_CLUSTER_GPU_COUNT_PER_POD`, never a second, independently duplicated literal per rank; start
rank 0, poll for the 128-byte id file over ssh (`stat`), `scp` it to a local staging copy
(`$RP_WORK/nccl.id`, mode 0600), gate the ship on `_rpc_id_file_ready` reporting exactly 128
bytes, `scp` it up to the member, THEN start rank 1; watch both ranks together (inactivity,
wrong-tree via `PROVE_EXPECT_SHA`, and the T-10m budget cut, F16); pull both `rank-<r>.json`
reports (a failed `rsync` JOINS the leg's own `rc`, mirroring PR-B1's own P1 property for the pod
leg — verified: `grep -n 'pull_rc' ci/scripts/runpod_gpu_cluster.sh` shows the identical
`[ "$rc" -eq 0 ] && rc="$pull_rc"` join, never a silent warning); assemble ONE `gang` artifact
(`_rpc_assemble_gang_artifact`, the SOLE writer — both ranks only report); run the id-secrecy scan
(M4); and delete the cluster on EVERY exit arm via an EXIT trap
(`_rpc_cleanup_cluster`/`trap _rpc_cleanup_cluster EXIT`), which ALSO records whether member
self-removal took (see §1).

**F3 — the no-public-port fallback.** Every member's direct ssh endpoint is read from
`Pod.ssh.direct` (`GET /v2/clusters/{id}/pods`, which returns full `Pod` objects). When a
member's own `ssh.direct` is null but its overlay `ip` is known, the driver proxies through the
primary (`ssh -J`/`ProxyJump` via `member_extra_sshopts`) — the ONLY alternative path, selected by
what the create response actually yields, never `proxy` for a data transfer:

```
$ grep -n 'ProxyJump' ci/scripts/runpod_gpu_cluster.sh
651:  member_extra_sshopts=(-o "ProxyJump=root@${primary_host}:${primary_port}")
```

The primary itself MUST carry a direct endpoint (checked explicitly — there is no second-order
proxy path if the primary itself has none); this is a real code path in this driver, but its
correctness on real RunPod infrastructure is one of the facts the un-executed pre-flight (§1)
would have settled and has not.

**Exit contract**, verified against the driver's own exit sites: `0` pass; `75` no cluster
capacity (the availability read failed, or no data center cleared the floor); `76` inactivity
kill OR the id never crossed within `RP_SSH_WAIT_SECS`; `77` wrong tree (`PROVE_EXPECT_SHA`
disagreement on either rank's log); `97` wrong shape (device-count/compute-cap mismatch inside the
remote heredoc, OR a member's launch-time read-back failure); `124` budget cut at T-10m; else the
driver's own post-run refusal (a failed pull, a failed assembly, a failed id-secrecy scan, a
missing `ens1` line), by name — never a bare nonzero with no message.

**Cost derivation** (committed, never re-derived per run): `2 x $1.908/GPU/h = $3.816/h`;
terminate-succeeds `1h x $3.816/h = $3.82`/run; sweep-only (member self-removal fails AND the
EXIT trap's own delete fails) `(1 + 6)h x $3.816/h = $26.71` — the 6h term is `gpu-reap.yml`'s own
cron period. `≤ 1h billed, ≤ 2 runs` is the standing spend authorization dated 2026-09-13, per the
plan document and `CONTRACT-U7b.md`'s own header — a human authorization this contract cites,
never re-derives.

**Oracle:** `ci/scripts/test_gpu_cluster_lane.sh` (sources the driver, never executes it — the
sourced-execution guard skips the live network flow) drives, over the REAL functions: G0 (sourcing
invokes no curl/ssh/scp/rsync/runpodctl — measured through a PATH shim, not asserted); G1
(`rp_cluster_rank_verdict`/`rp_cluster_verdict` over every rc arm); G2 (the `CLUSTER_GROUPS`
closure matches every `::group::` name in the remote heredoc minus `device`); G3 (the shared F13
zero-test tripwire, spliced into `_rpc_remote_script`); F2/F3 (the launch-time read-back: args
mismatch, `ssh.direct` null with no overlay ip, `ssh.direct` null with a usable overlay ip — still
OK, the fallback path — and a parse failure); F11 (127/128/129-byte and missing-file id-ready
arms); A5 (the `ens1` log assertion, `_rpc_ens1_seen`); G5 (the failed-pull-joins-rc property); G4
(the cost bound re-derived from the mechanism's own constants, never a hand-typed duplicate); G7
(no `schedule:` key anywhere in the committed `gpu-cluster.yml`).

## 5. M4 — the id-secrecy scan (`ci/scripts/gang_id_secrecy_scan.py`)

**The id crosses hosts HEX-encoded on the wire** — the `scp` ship step moves the raw 128-byte
file itself (never a text-encoded form in transit), but this scan's own threat model (per its
module doc, quoting PR-B1's own precedent for the pod leg's hex-encoded crossing) is that a LEAK,
if one occurs, could show up in any of the encodings a human or a later ship-step revision might
plausibly emit into a text carrier: raw bytes, hex (lower AND upper, checked as two DISTINCT
literal needles — never one case-folded search), and base64. `id_needles`
(`ci/scripts/gang_id_secrecy_scan.py::id_needles`) computes exactly these four:

```
$ sed -n '99,107p' ci/scripts/gang_id_secrecy_scan.py
```
```
    hex_lower = id_bytes.hex().encode("ascii")
    hex_upper = id_bytes.hex().upper().encode("ascii")
    b64 = base64.b64encode(id_bytes)
    return [
        ("raw", id_bytes),
        ("hex-lower", hex_lower),
        ("hex-upper", hex_upper),
        ("base64", b64),
    ]
```
(the enclosing `def id_needles(id_bytes: bytes) -> list[tuple[str, bytes]]:` sits at line 91, its
own doc comment naming the hex-case-independence rationale above these lines.)

**Exit lattice**: `0` clean; `1` a carrier carries the id in some encoding (named by carrier and
encoding — the bytes themselves are NEVER printed); `2` UNEXAMINABLE — a carrier could not be
read at all: missing, unreadable (permission), a dangling symlink, an archive member under the
pulled artifact directory (refused by SUFFIX match, never opened — `.tar`/`.tar.gz`/`.tgz`/
`.tar.bz2`/`.tbz2`/`.tar.xz`/`.txz`/`.zip`/`.gz`), or a staging id file that is not exactly 128
bytes. `2` is never read as clean — an unexaminable carrier is treated exactly as seriously as a
confirmed hit for the purpose of deciding whether the run is trustworthy.

**Carrier set**: the pulled artifact directory (recursively, symlinks followed); the driver's own
`tee`'d run log (F6 — the SAME file the workflow step uploads, never a second unscanned copy —
verified: `RUN_LOG="$(mktemp)"` then `exec > >(tee -a "$RUN_LOG") 2>&1` at the top of the driver's
sourced-execution guard, and `_rpc_run_id_secrecy_scan` is later called with `"$RUN_LOG"` as its
`--log` argument); the assembled `gang` artifact JSON; and the staging copy's own directory
LISTING (a leak spelled into a FILENAME next to it, never its content a second time — the content
is the needle SOURCE, read exactly once). The staging copy is deleted ONLY after a CLEAN scan
(`--delete-staging`), never unconditionally — a dirty run's staging file survives for hand
inspection.

**Oracle:** `gang_id_secrecy_scan.py --self-test` (17 `unittest` cases, run against a real tempdir
fixture tree, no mock filesystem): clean scan passes and preserves staging without the flag;
clean scan deletes staging only with the flag; raw/hex-lower/hex-upper/base64 planted hits in
each of the four carrier kinds; a hit spelled into a directory entry NAME; an archive member
refused without being opened; a dangling symlink UNEXAMINABLE; a symlink to a real file followed
(clean and hit arms both); a missing log/assembled-artifact UNEXAMINABLE; an unreadable file
UNEXAMINABLE (skipped under root — the `chmod_bypassed` class); a missing pulled dir UNEXAMINABLE;
a staging file of the wrong byte count UNEXAMINABLE; hex-upper and hex-lower asserted as distinct
needles. `test_gpu_cluster_lane.sh`'s F5 group drives the SAME scan through
`_rpc_run_id_secrecy_scan` (the driver's own one-place wrapper) against equivalent fixtures.

## 6. M5 — P8 (schedule visibility) and the `RENTING_ROOTS` derivation

`check_gpu_prove_once.py` ALREADY carries P1-P7 at this unit's own base (PR-B1 landed P7: every
paid pod lane held to P1's three sub-rules — exactly one invoker, no `push:`/`workflow_call:` in
that invoker's own `on:` block, nothing else `uses:` it — over a REVIEWED `PAID_POD_LANE_TABLE`
registry). **This unit adds a NEW arm, P8** — never "a new P7 arm"; P7 already existed and this
unit does not touch its own three sub-rules.

**P8's property** (`check_gpu_prove_once.py::check_p8_schedule_visibility`): for every
`PAID_POD_LANE_TABLE` workflow AND every OTHER workflow that mentions a `RENTING_ROOTS`-derived
driver while its own comment-stripped text ALSO carries `RUNPOD_API_KEY` at any scope, a
`schedule:` key in its `on:` block (read via the shared `read_top_level_on_block`) is a FINDING
unless that workflow is a reviewed key of `PAID_LANE_CRON_ALLOWLIST`:

```
$ sed -n '1274,1285p' ci/scripts/check_gpu_prove_once.py
```
```
PAID_LANE_CRON_ALLOWLIST: dict[str, tuple[str, str]] = {
    "gpu-prove.yml": (
        "capability-surface-proof",
        "runpod_gpu_prove.sh's never-vacuous capability-surface-build/-proof groups refuse a 0-test "
        "run -- the nightly cron can never silently pass on an empty suite",
    ),
    "gpu-reap.yml": (
        "rp_cluster_sweep",
        "gpu-dev.sh's reap arm fails closed (non-zero) when it cannot enumerate pods OR clusters -- "
        "the 6-hourly cron never reports 'nothing to reap' from an enumeration it could not make",
    ),
}
```

**Why exactly these two, and why they are on the list at all**: `gpu-prove.yml` carries a
pre-existing nightly cron that PREDATES this rule — it is reviewed and never-vacuous because
`runpod_gpu_prove.sh`'s own `capability-surface-build`/`-proof` groups refuse a 0-test run (the
same never-vacuous doctrine P1 already states for the prove lane). `gpu-reap.yml` carries the
6-hourly reap cron THIS unit's own M1 depends on as the primary backstop for cluster orphans — it
is reviewed and never-vacuous because `gpu-dev.sh reap` fails closed (non-zero) whenever EITHER
sweep cannot enumerate, per M1 above. Neither entry is a rubber stamp: each TOKEN
(`capability-surface-proof`, `rp_cluster_sweep`) must occur verbatim in the workflow's own
comment-stripped text OR its `PAID_POD_LANE_TABLE` driver's comment-stripped text (F10) — an
unresolvable token is a FAIL, exactly like a listed workflow carrying no `schedule:` at all (a
dead waiver). `gpu-cluster.yml` and `gpu-gang.yml` are NOT on this list — neither carries a
`schedule:` trigger today, and P8 is precisely what makes adding one to either a reviewed act
requiring a human to add an allow-list row naming its own never-vacuous arm, rather than a silent
default.

**`RENTING_ROOTS`** (`check_gpu_prove_once.py::RENTING_ROOTS`) is what makes the cluster driver
visible to P7 (and, through P7's own `PAID_POD_LANE_TABLE` derivation, to P8's subject set) at
all — a REVIEWED LIST, not a single hard-coded pod-only seed:

```
$ grep -n 'RENTING_ROOTS: tuple' ci/scripts/check_gpu_prove_once.py
948:RENTING_ROOTS: tuple[str, ...] = ("_rp_deploy_payload", "rp_cluster_create")
```

`derive_deploy_closure` (`check_gpu_prove_once.py::derive_deploy_closure`) computes the RENTING
CLOSURE as each root's transitive callers inside `runpod_lib.sh` PLUS the roots themselves — a
root is included because an external driver may call it with NO wrapper in between
(`rp_cluster_create` has no `rp_deploy_live`-shaped wrapper the way `_rp_deploy_payload` does), so
excluding the bare roots from the matched set would make P7 blind to a driver calling one
directly. A missing root fails closed with its OWN named finding — verified BOTH roots
independently (never one message that only names one of the two):

```
$ grep -n 'cannot derive the renting closure' ci/scripts/check_gpu_prove_once.py
996:            f"P7: cannot derive the renting closure — no `{r}() {{` definition in {RUNPOD_LIB_REL} "
```

`PAID_POD_LANE_TABLE` gains the cluster row through this exact derivation (never a hand-added
special case):

```
$ sed -n '844,850p' ci/scripts/check_gpu_prove_once.py
```
```
    "ci/scripts/gpu-dev.sh": "gpu-reap.yml",
    # The distributed-training CLUSTER leg: 2 hosts x 1 GPU on one RunPod
    # CLUSTER (REST v2) -- a second, independent renting mechanism from the
    # pod leg's GraphQL `podFindAndDeployOnDemand`, derived into P7's
    # subject set via RENTING_ROOTS below (never a hard-coded pod-only seed).
    "ci/scripts/runpod_gpu_cluster.sh": "gpu-cluster.yml",
```

**Oracle:** `test_check_gpu_prove_once.py`'s `DerivedRentingDriverTest` (closure equals both roots
plus their own callers; a new deploy wrapper joins with no gate edit; a library missing EITHER
root fails closed, independently of the other) and `ScheduleVisibilityTest` (the real tree is
clean; RED-then-GREEN — P7 alone does not catch a planted cron on `gpu-gang.yml`, P8 does; the
allow-listed prove cron is clean; an unresolvable allow-list token fails; a listed workflow with
no cron is a dead waiver; an allow-list entry naming a nonexistent workflow fails; an unreadable
`on:` block fails, never a silent skip; a derived driver with a schedule AND the secret is caught
even OFF the table). `PreFixShapeFixtureTest` (unchanged by this unit) continues to reproduce the
esc-084 pre-fix shape P7 itself closes.

## 7. M6 — artifact registry leg discrimination (`check_cuda_run_artifacts.py` rule (k))

`gang.leg` is now a REQUIRED, closed-set field (`"pod"` | `"cluster"`), checked FIRST, before
either per-leg registry — the two legs owe DIFFERENT payloads and neither registry is optional
padding on top of the other:

```
$ grep -n 'GANG_LEG_POD\|GANG_LEG_CLUSTER\|GANG_CLUSTER_HOSTS' ci/scripts/check_cuda_run_artifacts.py
```
```
998:GANG_LEG_POD = "pod"
999:GANG_LEG_CLUSTER = "cluster"
1005:GANG_CLUSTER_HOSTS = 2
```

Pod-leg rows keep today's registry unchanged (`world`, `collective`, per-rank `device`, the
same-seed digest PAIR, the measured per-step loss delta, epsilon). Cluster-leg rows
(`GANG_CLUSTER_FIELD_REGISTRY`) instead require `hosts` (exactly `GANG_CLUSTER_HOSTS == 2` — this
leg proves a two-host bootstrap only, never more), `ranks[]` (`rank`/`host`/`device`/`iface` per
entry, count must equal `world`), `reduced_vector_digest` (a SINGLE bit-exact digest, required
equal across both ranks on a `pass` — asserted by the DRIVER before assembly, never re-derived by
the gate itself), `verdict`, `reason` on `fail`, and the rented shape as measured from the create
response (`pod_count`, `gpu_count_per_pod`, `ttl_hours`). It carries NO `digests`/
`per_step_loss_delta`/`epsilon` — those name a training-loss reproducibility regime this leg does
not run; the cluster leg's own `reduced_vector_digest` equality is explicitly NEVER conflated with
the pod leg's LoRA-shaped same-seed reproducibility pair (a different measurement this leg does
not attempt).

**Oracle:** `check_cuda_run_artifacts.py --self-test`'s rule (k) cases (verified present by direct
read): a complete pod-leg artifact and a complete cluster-leg artifact both pass; a non-gang
artifact is never gang-checked; a missing/out-of-set `gang.leg` fails by name; each pod-leg field
individually missing fails by name; each cluster-leg field individually missing fails by name
(the contract's own named example: "a cluster artifact lacking `hosts` fails by name" — verified
present); `hosts != 2` fails; a `ranks[]` entry count that disagrees with `world` fails; a rank
missing `iface` fails; two ranks repeating the same rank index fails; a `pass` verdict with no
`reduced_vector_digest` fails; a `fail` verdict with a stated `reason` is legal (a `fail` is
representable at the schema level, never refused at assembly time — matching `_rpc_
assemble_gang_artifact`'s own behavior).

## 8. The executed attempt — status

`CONTRACT-U7b.md § 2` schedules "one real cluster run at the end of c3", authorized, bounded by
M3's own cost figures, whose log would be the evidence for: the `args` entrypoint reaching bash
on `RP_IMAGE` (REST v2, never independently confirmed — see §1); member sshd reachability;
`ens1` as the overlay iface actually appearing in NCCL's own log; and whether member
self-removal works on a cluster object. **This run has NOT happened as of this contract.** No
`RUNPOD_API_KEY`-bearing session has executed `runpod_gpu_cluster.sh` against a real RunPod
account within this unit's own commit history; no `gpu-cluster.yml` workflow run exists; no gang
artifact under `crates/jammi-kernels/artifacts/cuda-runs/` for the cluster leg has been committed
by this unit (verified: `git log --oneline --all -- 'crates/jammi-kernels/artifacts/cuda-runs/*cluster*'`
finds nothing on this branch, and c4's own commit — this contract plus the plan-doc notes — adds
no artifact file). A failed run, when one is executed, is itself a FINDING to be recorded and
fixed (re-run once at most, within the 2-run authorization) — never silently re-tried past that
ceiling, and never presented as a pass if it never ran clean.

## 9. Known-unmeasured / uncovered (named, never claimed closed)

- **Member self-removal on a cluster pod.** S4 measured only that a cluster member exposes
  `actions: []` in the RunPod API's own listing — a LISTING attribute, never an OBSERVED
  termination attempt. Whether `runpodctl remove pod` inside a cluster member's own entrypoint
  actually succeeds (and whether a success there is even reflected in the cluster's own billing
  state) is unknown until §8's run executes and records `cluster-self-remove: ok|refused`
  (`_rpc_self_remove_status`/`_rpc_cleanup_cluster`, `ci/scripts/runpod_gpu_cluster.sh`).
- **The REST v2 `args` field reaching `bash -c` on `RP_IMAGE`.** Never independently confirmed —
  see §1. The launch-time read-back (`_rpc_check_readback`) is the guard against this being
  false, not a proof that it is true; it can only ever detect the mismatch AFTER a real cluster
  has already been created.
- **The `NCCL_SOCKET_IFNAME=ens1`/pin set at world >= 3.** README.md's own S5 spike result states
  this explicitly: at world size 2 the reduction is commutative, so the `NCCL_ALGO`/`PROTO`/
  `NCHANNELS` pin set is UNTESTED at any world size where it is not — this unit's own leg proves
  world 2 only (`CLUSTER_TEST_FILTER=gang_nccl_two_hosts`, `RP_CLUSTER_POD_COUNT=2`, hard-pinned)
  and cannot itself close this question. `docs/maintainer/dev-gpu.md`'s own "Known-unmeasured"
  section for the cluster leg already states this and points back here.
- **S5's cross-host byte identity.** S5 (README.md) measured byte-IDENTICAL forward/backward/SGD
  across TWO PROCESSES and across two A100s with NO env pins for candle 0.11's LoRA-shaped
  arithmetic — but that measurement did not exercise a cross-HOST NCCL collective the way this
  unit's own leg does; this leg's `reduced_vector_digest` equality is a DIFFERENT, narrower
  measurement (a bit-exact sum reduction, not a training step's forward/backward/SGD digest pair)
  and does not itself extend S5's own claim across hosts. Filed here as uncovered, not silently
  assumed to follow from S5.

## 10. Invariants crossed

B2 (every script and doc here names no consumer — verified by `check_no_consumer_names.py`
below); the paid-lane doctrine (P1/P7/P8: label/dispatch only, never merge-path — `gpu-cluster.yml`
carries neither `push:` nor `workflow_call:`, verified by direct read of its `on:` block); B6
(the ai-core test body, M2, and the docs-ci lane, M1/M3-M6, land as one committed unit — this
branch's own five commits, `f772e2e8` through `ed3612e6`); K2 (every parsed API body is validated
before use — `_rp_rest`'s callers each check the required key is present before printing a
success, never trusting a 2xx status alone).

## 11. Gate files a human must review at this unit's merge

`swarm.yml`'s human-amend-only glob (`SWARM_GATE_TOUCHED`) covers gate-script edits; the reviewer
checks:

- `ci/scripts/check_gpu_prove_once.py` — the new P8 arm (`check_p8_schedule_visibility`,
  `PAID_LANE_CRON_ALLOWLIST`), `RENTING_ROOTS` widened to two roots, the `PAID_POD_LANE_TABLE` row
  for `ci/scripts/runpod_gpu_cluster.sh`.
- `ci/scripts/check_cuda_run_artifacts.py` — rule (k)'s `gang.leg` discriminator and
  `GANG_CLUSTER_FIELD_REGISTRY`.
- `ci/scripts/runpod_lib.sh` — every `rp_cluster_*` primitive, the `RP_CLUSTER_PREFIX` name space,
  `rp_sweep`'s cluster-member exclusion and counted terminate-refusal.
- `.github/workflows/gpu-cluster.yml` — its own `on:` block (label + dispatch only), the
  concurrency group, the timeout budget.
- `.github/workflows/gpu-reap.yml` — unchanged trigger shape, now backstopping two object types.

## 12. Residuals recorded UNCOVERED

- The un-executed REST v2 pre-flight (§1) and the un-executed cluster run (§8) — both named, not
  silently deferred.
- Member self-removal on a cluster object (§9).
- The NCCL pin set at world >= 3 (§9) — filed on this contract per `docs/maintainer/dev-gpu.md`'s
  own "Known-unmeasured" cross-reference.
- S5's cross-host byte identity (§9) — this leg's own `reduced_vector_digest` measurement is
  narrower and does not extend S5's claim across hosts.
- **U7b-A3** (the `gpu-gang.yml` 6-hourly cron re-add) is explicitly NOT part of this unit — a
  future, separately authorized re-add, made a reviewed act by P8 rather than a silent default.
- Every residual PR-B1's own contract (`docs/rigor/contracts/feat_500-PR-B1.md`) already recorded
  UNCOVERED for P6/P7 (issues #561, #563, #564, #565) is UNCHANGED by this unit — this unit adds a
  new arm (P8) beside them, touching none of their own named gaps.

## 13. Citations verified against which head

Every construct cited above was read directly against this worktree's tree at commit `ed3612e6`
(branch `feat/500-C-U7b`) via `grep -n`/`sed -n -p`, not against any scratchpad working document's
own line numbers, and not against `CONTRACT-U7b.md`'s own §7/§8 fold text beyond citing its
decisions by name. No bare `path:line` form appears in this document outside a fenced block
quoting an executed `grep`/`sed` command and its real output.
