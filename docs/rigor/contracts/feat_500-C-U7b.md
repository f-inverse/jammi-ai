# CONTRACT — feat/500-C-U7b: the RunPod cluster leg (2 hosts x 1 A100) proves the two-host NCCL bootstrap; the reap treats a cluster as its own object type

**Contract of record.** slug: `feat_500-C-U7b` — the committed mechanism contract
`ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` before this unit's
rigor record at `docs/rigor/feat_500-C-U7b.jsonl` (the lead's own export, landed separately)
satisfies that checker's disclosure requirement. This unit is **U7b-A2** in the decomposition
`docs/plans/67-distributed-training/UNITS.md § U7b` states: **A1-pull** (the pod-tier smoke's CI
scaffolding — `gpu-gang.yml`, `runpod_gpu_gang.sh`, P7) merged already, as PR-B1 (contract
`docs/rigor/contracts/feat_500-PR-B1.md`); **A2** is this unit's SHIPPED scope (primitives, reap,
the two-host test body, the registry's leg discrimination, P8) — the driver and its workflow,
originally also part of A2, were EXCISED at round 3 (below) and refiled as **U7b-A2b**; **A3** —
a 6-hourly cron re-add on `gpu-gang.yml` — is a FUTURE, separately authorized unit neither this
contract nor U7b-A2b builds.

**Round 3 (closing, 2026-09-15): the driver, its workflow, the id-secrecy scan and the lane
fixture suite are EXCISED**, per the pre-committed §9 stop rule (a third BLOCK on the driver
mechanism). `ci/scripts/runpod_gpu_cluster.sh`, `.github/workflows/gpu-cluster.yml`,
`ci/scripts/gang_id_secrecy_scan.py`, and `ci/scripts/test_gpu_cluster_lane.sh` are DELETED at
`55276624` (commit 1 of this docs-ci round), along with every table/registry/allowlist/guard-
matrix row that named them; `check_cuda_run_artifacts.py`'s rule (k) keeps the cluster leg's field
registry and shape/rank checks but now REFUSES any artifact claiming `gang.leg == "cluster"` (no
producer is registered for that leg). §4 and §5 below record the excision with the round-3
findings verbatim; every §2/§2b/§2c citation into the four now-deleted files is pinned
`(at 23ef24a9's tree)` — the last commit before the excision — never `(at HEAD)` into a path that
no longer exists.

**This revision supersedes the original c4 contract.** The closing adversarial audit BLOCKed
that revision on six findings (F1-F6) plus a citation-form defect: every citation there was a
fenced `grep -n`/`sed -n` transcript, which `check_rigor_record.py`'s own cost-floor check
(`check_path_line_citations`, matching bare `` `path.ext:NNN` `` tokens) does not scan grep
OUTPUT lines against at all — that revision's citations passed the checker VACUOUSLY, never
actually exercised. Every citation below is instead the bare `path:line` (or `path:line-line`)
form that checker's own regex matches, tagged `(at <sha>)`, re-derived by direct read against
THIS tree AFTER every fix below landed — the last thing done before this file was written.

Owner: **docs-ci** (fix round 1, dispatched after the c4 revision's closing BLOCK; fix round 2,
dispatched after round 1's own closing adversarial audit BLOCKed on five findings on round 1's own
new surfaces; fix round 2c, a small lead-initiated fix on round 2's own new surface found by the
lead BEFORE the closing adversarial audit re-ran — never itself a third audit BLOCK, §2c; round 3,
this revision, records the closing adversarial audit's pre-committed §9 stop rule firing on a
THIRD BLOCK on the driver mechanism — the driver, its workflow, the id-secrecy scan, and the lane
fixture suite are EXCISED). Commit history on this branch, oldest first, below the `main` merge
(`git log --reverse --format='%h %s' main..HEAD`, re-derived AFTER round 3's own excision commit
landed):

```
467dd9c9 test(ai): #500 U7b — the two-host NCCL smoke: rank 0 mints the id to a file, ...
b984e24c ci(runpod): #500 U7b — cluster primitives on REST v2, one renting entrypoint text, ...
c88593d8 ci(cluster): #500 U7b c3 -- the NCCL id-secrecy scan (F5)
39eeee0f ci(cluster): #500 U7b c3 -- check_cuda_run_artifacts.py rule (k) leg discrimination (M6)
5ebe53ab ci(cluster): #500 U7b c3 -- the cluster leg driver, its workflow, F13's shared ...
dbadfdbf ci(cluster): #500 U7b c3 -- P8 schedule visibility, RENTING_ROOTS, and the cluster's ...
d2497c42 ci(cluster): #500 U7b c3 -- record cluster-self-remove: ok|refused on the EXIT trap
ed2e7c3c docs: #500 U7b -- the cluster leg's contract of record and the plan's current-state notes
1480cacb test(ai): #500 U7b — the two-host test's skips are dominated by registered ... (KO-7)
c23049f9 merge: main (#571 R12 lead gate) into feat/500-C-U7b
b480f2dc test(ai): #500 U7b — a missing hostname or NCCL iface is a fail verdict, never ...
31c8aa64 fix(ci): #500 U7b — the cluster driver initialises its SSH state and chains the ...
c4f0c36e fix(ci): #500 U7b — every exit scans before anything can be uploaded; the scan's ...
ebe79a0d docs(rigor): #500 U7b — the contract of record cites only shas on this branch
23ef24a9 fix(ci): #500 U7b — a refusal arm keeps its clean diagnostics at the upload path; ...
2ae704b8 docs(rigor): #500 U7b — contract §2c for fix 2b
1df31973 docs(rigor): #500 U7b — the contract's §3 (M2, the two-host NCCL test body) restored ...
55276624 refactor(ci): #500 U7b — the cluster driver, its workflow, the id-secrecy scan and the
          lane suite are excised (round-3 stop rule); the cluster leg fails closed without a
          producer
```

`55276624` is round 3's own excision commit — the LAST commit on the branch before this
revision's own commit lands. Every construct §4/§5 below describe is read against `55276624`'s
own tree; every construct §2/§2b/§2c below describe that lived ONLY in the four now-deleted
files is read against `23ef24a9`'s own tree (the last commit before those files stopped existing)
and tagged accordingly — never `(at HEAD)` into a path `git show HEAD:<path>` can no longer
resolve.

**Correction (round 2):** round 1's own contract text (below, §2-§13, unchanged in substance
except where explicitly marked "round 2") named the ninth commit `d12e1689` and placed the
`c23049f9` merge between `ed2e7c3c` and `1480cacb`. Neither is accurate against this branch's
real history: `d12e1689` does not exist on this branch (that commit's own content — "a missing
hostname or NCCL iface is a fail verdict, never ... " — landed as `b480f2dc` instead, a distinct
sha under a rebase this round did not itself perform but must cite correctly); every prior
citation tagged `d12e1689` is corrected to `b480f2dc` below (5 sites: §2 F4's closing parenthetical,
§3's own heading and body, §9's first bullet). `c23049f9` in fact sits between `1480cacb` and
`b480f2dc` in the list above, not between `ed2e7c3c` and `1480cacb`. `b480f2dc`'s own diff
(`git show --stat b480f2dc`) touches exactly one file, `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`
(189 insertions, 19 deletions) — §3's own "unchanged by this fix round" heading is corrected below
to name what actually changed, since a file round 1 did not itself touch can still have moved
between the commit round 1's own contract was written against and this branch's real tip.

Round 1's own fix (§2-§13 below) landed as `31c8aa64`, combining the code fix and this contract's
own prior revision in one commit — every citation there tagged `(at HEAD)` refers to `31c8aa64`'s
own tree. Round 2's own fix (§2b below) landed as `c4f0c36e`, atop `31c8aa64`. Round 2c's own fix
(§2c below) landed as `23ef24a9`, atop `c4f0c36e`; every citation in THIS revision tagged
`(at HEAD)` refers to `23ef24a9`'s own tree — the tree THIS contract revision is committed
against, and the LAST commit on the branch once this file's own commit lands. Every `(at HEAD)`
citation inherited from an earlier round, for a file round 2c also touched
(`ci/scripts/runpod_gpu_cluster.sh`, `ci/scripts/gang_id_secrecy_scan.py`,
`ci/scripts/test_gpu_cluster_lane.sh`), is RE-DERIVED below against `23ef24a9`'s tree (line
numbers moved, each with a "moved from round 2's `N`" note); §2b's own P-A subsection, which
narrates round 2's OWN fix as it stood before round 2c changed it, is pinned explicitly to
`c4f0c36e`'s tree instead, never re-tagged `(at HEAD)` (see that subsection's own note). A
citation to a file neither round 2 nor round 2c touched keeps its earlier `(at HEAD)` tag unless
flagged otherwise, since an untouched file's tree carries forward byte-for-byte across every
round. Every citation to a file no round touched is tagged at the sha that last touched it, per
`git log -1 --format=%h -- <path>` against this same tree.

**Round 3's own retagging.** The paragraph above narrates what `(at HEAD)` meant AT EACH EARLIER
ROUND's own time of writing — that narration is accurate as history and is left as written. But
`HEAD` has since moved past `23ef24a9` to `55276624` (round 3's excision commit) and beyond, and
`ci/scripts/runpod_gpu_cluster.sh`, `ci/scripts/gang_id_secrecy_scan.py`, and
`ci/scripts/test_gpu_cluster_lane.sh` no longer exist at `HEAD` at all — every citation in
§2/§2b/§2c below that a previous round tagged `(at HEAD)` is, in THIS revision, re-tagged
`(at 23ef24a9's tree)`, meaning literally: read `git show 23ef24a9:<path>` (or
`git checkout 23ef24a9 -- <path>` into a scratch copy), never `HEAD`. §2b's own P-A subsection
keeps its earlier `(at c4f0c36e's own tree)` pin, unaffected by this round. §4 and §5 are REWRITTEN
in full (not merely re-tagged) to record the excision itself, at `55276624`'s own tree.

## 0. What this unit is, in one paragraph

Two RunPod PODS on one RunPod CLUSTER (REST v2, `POST /v2/clusters`), one A100 each, joined over
the cluster's own private overlay network — the only place a real cross-HOST NCCL gang
(`ncclCommInitRank`, an out-of-band id file) is exercised before release; the pod-tier gang leg
(PR-B1) proves a two-DEVICE collective inside ONE pod (`ncclCommInitAll`) and cannot reach this
bootstrap at all. Six mechanisms: **M1** the REST v2 cluster primitives in `runpod_lib.sh` plus
the reap arm that now treats a cluster as its own object type; **M2** the two-host NCCL test body
(`gang_nccl.rs`); **M3** the driver, `ci/scripts/runpod_gpu_cluster.sh`; **M4** the id-secrecy
scan; **M5** `check_gpu_prove_once.py`'s P8 arm (schedule visibility) plus the `RENTING_ROOTS`
derivation that makes the cluster driver visible to the existing P7 arm at all; **M6** the
artifact registry's leg discrimination in `check_cuda_run_artifacts.py`'s rule (k).

## 1. The pre-flight gap — stated honestly before anything else

The session's own working contract (`CONTRACT-U7b.md` §8, F2) called for a ~$0.05 pod pre-flight
BEFORE any cluster run, to settle on real hardware whether RunPod REST v2's `args` field actually
reaches `bash -c` on `RP_IMAGE` the way the GraphQL `dockerArgs` field measurably does (S4's own
probe measured the GraphQL path only; REST v2's `args` field has never itself been executed
against a live pod by anyone on this unit). **This pre-flight has NOT been executed** — the
lead's own attempt was BLOCKED by the harness's permission classifier (a real-world transaction
touching billed infrastructure); it needs a human to run it or explicitly allow it. This is a gap
in what has been verified about RunPod's own API surface, not a gap in the driver's own design.

Until that pre-flight runs, this unit's actual guard is the driver's own LAUNCH-TIME READ-BACK
refusal, never an independent confirmation that the mechanism works: `_rpc_check_readback`
(`ci/scripts/runpod_gpu_cluster.sh, lines 283-316` at 23ef24a9's tree, moved from round 2's `276-309` by round 2c's
own header growth above this point, §2c below) reads `GET /v2/clusters/{id}/pods` after
create and refuses (exit 97) the moment either (a) a member's own `Pod.args` does not contain the
exact entrypoint text this driver sent, or (b) neither `ssh.direct` nor a usable overlay `ip` is
present for it. This is a REFUSAL mechanism, not a proof the API behaves as documented — it can
only catch the failure AFTER a real cluster has already been created and billed for the time it
took to observe the mismatch (bounded by `RP_SSH_WAIT_SECS=300`s).

**§8 below — "the one real cluster run" — has also NOT happened.** No real RunPod cluster has
been created, no real cost has been billed, and no real answer exists yet to whether member
self-removal works on a cluster object, whether the `ens1` NCCL transport line actually appears
in a real run's log, or whether the two-host id ship completes within the driver's own wait
budgets. Every cost figure this contract cites (`$3.816/h`, `$3.82`/run, `$26.71` worst-case) is
a COMMITTED, re-derived-by-test figure — never a measured bill from an actual run.

## 2. Fix round 1 — the six findings, each closed by name

### F1 — the driver never called `rp_init`

At the c4 revision, `runpod_gpu_cluster.sh` never called `rp_init` — zero calls, against seven
`RP_SSHO[`/`$RP_PUBKEY` uses that all depend on it. `rp_init` (`ci/scripts/runpod_lib.sh:606-632`
at `b984e24c`, unchanged by this fix) is what generates the SSH keypair (`RP_PUBKEY`, read by
`_rp_cluster_payload` into the create body's own `env.PUBLIC_KEY`) and populates `RP_SSHO`
(`-i`/`IdentitiesOnly=yes`/`StrictHostKeyChecking=no`, `ci/scripts/runpod_lib.sh:631` at
`b984e24c`). Without it, the create payload ships an EMPTY authorized-key and every later
ssh/scp/rsync call runs with an empty `RP_SSHO` array — a silent, wrong-key failure that reads
exactly like "not yet reachable".

**Fix**: `rp_init` is now called immediately before the create call, exactly as
`runpod_gpu_gang.sh`'s own pod leg does it:

```
$ grep -n '^rp_init$\|^cluster_id="\$(rp_cluster_create' ci/scripts/runpod_gpu_cluster.sh
839:rp_init
842:cluster_id="$(rp_cluster_create "$RP_CLUSTER_GPU_TYPE" "$dcs")" || { echo "::error::cluster create failed"; exit 75; }
```

`ci/scripts/runpod_gpu_cluster.sh, line 839` precedes `:842` (at 23ef24a9's tree; round 2c's own P-A2 module header
prose and `assembly_ok=0` declaration moved both lines down from round 2's `795`/`798` without
changing their relative order, §2c below).

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh, lines 881-896` (at 23ef24a9's tree, moved from round 2's `806-821`
by round 2c's own P-A2 fixture growth above this point, §2c below; round 2 fixed this block's own
comment-filter — see §2b F10) is a static guard — a line-number comparison over the committed
driver text (`grep -n '^rp_init$'` vs. the first `rp_cluster_create ` call) — asserting `rp_init`
precedes the create call, never a behavioral probe (no network). A SECOND, class-level guard
closes the general case this specific line-number check does not: `test_check_gpu_prove_once.py`'s
`RpSshoRequiresRpInitTest` (`ci/scripts/test_check_gpu_prove_once.py:587-627` at `31c8aa64`,
unchanged by round 2) statically scans EVERY real `PAID_POD_LANE_TABLE` driver on disk and asserts
that any driver referencing `RP_SSHO[` also calls `rp_init` (a bare-line regex,
`RP_INIT_CALL_RE`), with its own RED-then-GREEN self-test proving the regex actually distinguishes
a call from a mention in prose.

### F2 — the trap replaced, never chained, the library's own cleanup

`trap _rpc_cleanup_cluster EXIT` REPLACED the `trap rp_cleanup EXIT` that sourcing
`runpod_lib.sh` installs unconditionally at source time (`ci/scripts/runpod_lib.sh:598` at
`b984e24c`, unchanged) — losing `rp_cleanup`'s own `rm -rf "$RP_WORK"` (the staging id file
`$RP_WORK/nccl.id` and the generated ssh keypair `$RP_WORK/id_ed25519`, both left on disk after
every run).

**Fix**: `_rpc_cleanup_cluster` now calls `rp_cleanup` explicitly, on every arm, before its own
`exit`:

```
$ grep -n 'rp_cleanup  # F2' ci/scripts/runpod_gpu_cluster.sh
769:  rp_cleanup  # F2: chain the library's own EXIT cleanup (rm -rf "$RP_WORK" when RP_WORK_IS_TEMP=1 -- the ssh keypair).
```

`ci/scripts/runpod_gpu_cluster.sh, lines 738-771` (at 23ef24a9's tree, moved from round 2's `701-734` by round 2c's
own P-A2/P-A3 growth above this point, §2c below — round 2 itself grew this body from round 1's
`591-608` by inserting the P-A scan-join and the unconditional staging-file delete — see §2b P-A)
is `_rpc_cleanup_cluster`'s own body, moved OUT of the sourced-execution guard (it was previously
defined only when the file is EXECUTED, making it untestable by sourcing) into the pure-helpers
section above it — only the `trap _rpc_cleanup_cluster EXIT` registration itself
(`ci/scripts/runpod_gpu_cluster.sh, line 849` at 23ef24a9's tree, moved from round 2's `805`) remains inside the
guard.

**Oracle**: `test_gpu_cluster_lane.sh`'s F2/F3(c) block (`ci/scripts/test_gpu_cluster_lane.sh, lines 104-191`
at 23ef24a9's tree, unchanged in shape or line range by either round 2 or round 2c) drives the REAL
`_rpc_cleanup_cluster` in a real subprocess (`bash -c '... source "$CLUSTER_SH" ...'`, so its own
`exit "$rc"` terminates that subprocess exactly the way a real EXIT trap fires), mocking only
`_rp_rest`/`rp_cluster_delete`/`rp_cleanup`: self-remove-ok, self-remove-refused-delete-ok, and
no-cluster-id-at-all all confirm `rp_cleanup` is chained (a marker file it writes exists
afterward) in every arm. Round 2 added a SECOND block (`ci/scripts/test_gpu_cluster_lane.sh, lines 194-382`
at 23ef24a9's tree, widened from round 2's own `194-319` by round 2c's own three additional arms — see §2c
below) driving the SAME real `_rpc_cleanup_cluster` through the id-secrecy-scan join this round
adds.

### F3 — three retire-failure classes read green

**(a)** `rp_sweep`'s own `refused` counter collapsed an unexpected terminate refusal (auth/
rate-limit/API error — the cluster-member case is already excluded upstream, before
`rp_terminate` is ever called) into a silent `0` return.

**Fix**: a non-zero refused count is now `return 1`, naming every refused id and reason:

```
$ grep -n 'sweep: \${refused} terminate' ci/scripts/runpod_lib.sh
2668:    echo "::error::sweep: ${refused} terminate(s) refused: ${refused_reasons[*]}"
```

`ci/scripts/runpod_lib.sh:2660-2669` (at 23ef24a9's tree) is the full arm — `return 1` follows the `echo` on
the very next line.

**(b)** `rp_cluster_sweep`'s `UNAGEABLE` arm (a cluster with no usable `createdAt`) `continue`d
the loop, and the function returned `0` at the end regardless.

**Fix**: `return 1`, naming the cluster id, never a `continue` back to a green summary:

```
$ grep -n 'cannot judge its age' ci/scripts/runpod_lib.sh
1796:      echo "::error::cluster ${age} (${why}) has no usable createdAt — cannot judge its age; reap explicitly if it is an orphan"
```

the `return 1` follows immediately on the next line. `ci/scripts/runpod_lib.sh:1793-1797` (at
HEAD) is the `UNAGEABLE` branch in full.

**(c)** The driver's own EXIT trap: a failed `rp_cluster_delete` was logged (`::error::`) but
never joined into the trap's own exit status.

**Fix**: `_rpc_cleanup_cluster` now joins a failed delete into `rc` and names the cluster LEAKED:

```
$ grep -n 'LEAKED cluster' ci/scripts/runpod_gpu_cluster.sh
749:      echo "::error::LEAKED cluster ${cluster_id}: could not delete on exit -- gpu-reap.yml's 6-hourly sweep is the backstop"
```

`ci/scripts/runpod_gpu_cluster.sh, lines 738-771` (at 23ef24a9's tree, the same span F2 cites) is the full trap
body: `rc` is captured from `$?` FIRST (the pending exit status), joined to `1` only when it was
still `0`, and the function's own `exit "$rc"` at the end — never a bare `return` — is what makes
the join visible to the process's real exit status (a trap's own `return` would not override an
already-pending exit code). `gpu-dev.sh reap` (`ci/scripts/gpu-dev.sh:265-281` at `b984e24c`,
unchanged) already joins `rp_sweep`'s and `rp_cluster_sweep`'s own rcs, so both (a) and (b) above
already propagate to `reap`'s own exit without a further change there.

**Oracle**: `test_runpod_cluster_lib.sh` Group 3 gains an `UNAGEABLE`-cluster fixture
(`ci/scripts/test_runpod_cluster_lib.sh:378-401` at 23ef24a9's tree: `rc=1`, names `cl-unageable`, deletes
nothing); Group 5's existing refused-terminate fixture is updated to assert `rc=1` and the named
summary line (`ci/scripts/test_runpod_cluster_lib.sh:499-512` at 23ef24a9's tree) rather than the pre-fix
`rc=0` it asserted before this round.

### F4 — the cluster registry never established two real hosts

At the c4 revision, `_gang_check_cluster_ranks` required only that `host`/`device`/`iface` be
non-empty strings — never that `host` be DISTINCT across ranks, never that `iface` (or `host`)
not be the driver's own `unknown` placeholder, never that `hosts == pod_count` or `world ==
pod_count * gpu_count_per_pod`, and the reduced-vector digest was a single TOP-LEVEL field the
driver had already collapsed at assembly time — an artifact with two ranks on the SAME host, or
an unresolved host/iface, or a digest disagreement hidden by the collapse, would pass.

**Fix, on the checker side** (`ci/scripts/check_cuda_run_artifacts.py`, at 23ef24a9's tree):

```
$ grep -n 'def _gang_check_cluster_ranks\|def _gang_check_cluster_shape\|def _gang_check_leg_producer_binding\|^GANG_LEG_PRODUCER_PATH' ci/scripts/check_cuda_run_artifacts.py
1023:GANG_LEG_PRODUCER_PATH = {
1376:def _gang_check_cluster_ranks(gang: dict, _data: dict, _repo_root: Path) -> list[str]:
1463:def _gang_check_cluster_shape(gang: dict) -> list[str]:
1496:def _gang_check_leg_producer_binding(gang: dict, data: dict, _repo_root: Path) -> list[str]:
```

`_gang_check_cluster_ranks` (`:1376-1466`; round 2 grew this from round 1's `1376-1461` by six
lines) now asserts `host` distinct across ranks, compared CASE-INSENSITIVELY
(`.strip().casefold()`, `:1452-1455` — round 2's own F4 advisory, matching the assembler's own
`_norm_host`, §2b), refuses `host`/`iface` matching `GANG_UNKNOWN_SENTINELS = ("unknown", "")`
(`:1373`, `:1418-1423`, unchanged by round 2), requires a PER-RANK `reduced_vector_digest`
(hex-validated on `pass`, `:1428-1439`, unchanged) and asserts those per-rank digests equal across
ranks on `pass` (`:1461-1466`) — never trusting the already-collapsed top-level field alone.
`_gang_check_cluster_shape` (`:1469-1500`, moved from round 1's `1463-1494` by the same six-line
shift) asserts `hosts == pod_count` and `world == pod_count * gpu_count_per_pod` — a check round 2
makes non-vacuous for the first time, since the driver's own assembler previously hardcoded
`pod_count`/`gpu_count_per_pod` to the same literals this check compares against (§2b P-C).
`_gang_check_leg_producer_binding` (`:1502-1518`), wired into `check_gang_artifact` right after
the leg is resolved, binds `gang.leg == "cluster"` to `producer.path ==
"ci/scripts/runpod_gpu_cluster.sh"` (and `"pod"` to `"ci/scripts/runpod_gpu_gang.sh"`,
`GANG_LEG_PRODUCER_PATH:1023-1026` — round 2 corrects round 1's own `1023-1032`, which over-ran
into the unrelated `GANG_CLUSTER_HOSTS` comment below the dict, unchanged by round 2) — a
self-declared leg can no longer dodge the other leg's registry by pointing `producer.path` at a
different driver.

**Fix, on the driver side** (`ci/scripts/runpod_gpu_cluster.sh`, at 23ef24a9's tree):
`_rpc_assemble_gang_artifact` now refuses to WRITE an artifact at all when any rank's own
`hostname`/`nccl_socket_ifname` is empty or `unknown`, or when both ranks report the same
`hostname`:

```
$ grep -n 'refusing to assemble' ci/scripts/runpod_gpu_cluster.sh
548:        print("refusing to assemble: rank %r own hostname is unresolved (%r)" % (r.get("rank"), host), file=sys.stderr)
551:        print("refusing to assemble: rank %r own nccl_socket_ifname is unresolved (%r)" % (r.get("rank"), iface), file=sys.stderr)
554:    print("refusing to assemble: both ranks report the SAME host (%r) -- not the two-host bootstrap this leg proves" % reports[0].get("hostname"), file=sys.stderr)
```

(moved from round 2's `541`/`544`/`547` by round 2c's own module-header growth above this
function, §2c below — round 2 itself moved these three lines from round 1's `473`/`476`/`479` by
growing the module header and adding the P-C measured-shape parameters above this function — see
§2b P-C/P-D; the host-repeat line, now `:554`, compares through `_norm_host` case-insensitively,
§2b advisory) — a named refusal (exit 2 from the assembler; joined into the driver's own `rc` by
the existing `_rpc_assemble_gang_artifact ... || { ...; [ "$rc" -eq 0 ] && rc=1; }` call site,
itself rewritten this round to an `if`/`else` that also sets `assembly_ok` — §2c below, same
site), never an artifact synthesized from unresolved data. Each rank's report now carries its OWN
`reduced_vector_digest` (`ci/scripts/runpod_gpu_cluster.sh, lines 557-567` at 23ef24a9's tree, inside the `ranks`
list construction), and the assembled artifact's `producer` block is now bound to this driver's
own path (`ci/scripts/runpod_gpu_cluster.sh, lines 592-598` at 23ef24a9's tree: `"path":
"ci/scripts/runpod_gpu_cluster.sh"`, `"kind": "script"`), matching `GANG_LEG_PRODUCER_PATH` above.
(A complementary fix on the Rust side, `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` at
`b480f2dc` — round 2 corrects round 1's own citation of this commit as `d12e1689`, a sha that does
not exist on this branch; see this file's own header correction note — outside this contract's own
owned files, landed by an ai-core agent in the same worktree during round 1's own fix window —
makes `hostname()` return a named `Result` instead of masking a failed read behind `"unknown"`, so
the driver-side refusal above is checking a value that itself can no longer silently BE `"unknown"`
on a healthy run.)

**Oracle**: `check_cuda_run_artifacts.py --self-test`'s rule (k) self-test gains, named
(`ci/scripts/check_cuda_run_artifacts.py:3349-3448` at 23ef24a9's tree; round 2 grew this from round 1's
`3343-3429` by adding a case-insensitive repeated-host arm, §2b advisory): two ranks on one host
FAILS; two ranks on the same host differing only in CASE FAILS (round 2); an `unknown` host FAILS;
an `UNKNOWN` (any case) iface FAILS; a per-rank digest mismatch on `pass` FAILS; a missing per-rank
digest on `pass` FAILS; `hosts != pod_count` FAILS; `world != pod_count * gpu_count_per_pod`
FAILS; a cluster-leg artifact carrying the pod leg's own `producer.path` FAILS (and vice versa).
`gang_baseline()`/`gang_cluster_baseline()` (`ci/scripts/check_cuda_run_artifacts.py:3154-3232` at
23ef24a9's tree — round 3 moved and reworded this pair further, §7's own citations below are
current-`HEAD`; through round 2c, moved from round 1's `3148-3226`) stamped each leg's real
producer path (`ci/scripts/runpod_gpu_gang.sh` / `ci/scripts/runpod_gpu_cluster.sh`), and the
self-test's own fixture repo carried tracked stand-ins for both paths
(`ci/scripts/check_cuda_run_artifacts.py:2751-2757` at 23ef24a9's tree, moved from round 1's
`2745-2751`) so rule (b)'s own producer.path-exists-and-is-tracked check had something real to
bind against — round 3 drops the cluster-leg stand-in file (§7's own note on `gang_cluster_
baseline()`'s current producer path).

**Round 2 adds a SECOND, independent oracle** for this same F4 property: rather than only driving
the CHECKER against hand-built fixture JSON, `ci/scripts/test_gpu_cluster_lane.sh, lines 733-877` (at
23ef24a9's tree, moved from round 2's `658-802` by round 2c's own P-A2 fixture growth above this point, §2c
below; §2b P-D) sources the REAL driver and calls the REAL `_rpc_assemble_gang_artifact` on the
happy path and
every refusal/representable-fail arm (unknown host, unknown iface, a case-insensitive repeated
host, a digest mismatch), then feeds each resulting artifact through the REAL
`check_gang_artifact` (imported directly from `check_cuda_run_artifacts.py`, never paraphrased) —
closing the gap between "the checker's own fixtures are self-consistent" and "the driver's own
assembler actually produces what the checker accepts".

### F5 — the id-secrecy scan could hang, and could read a FIFO/socket forever

`scan_dir` walked the pulled artifact directory via `os.walk(root, followlinks=True)` — that
detects no cycles at all, so a cyclic DIRECTORY symlink (`rsync -a` preserves one exactly as
planted) recurses forever. `scan_file` called `real.read_bytes()` on ANY non-directory path
regardless of its `st_mode` class — a `read_bytes()` against a FIFO or a UNIX socket with nothing
on the other end blocks forever, never returning.

**Fix** (`ci/scripts/gang_id_secrecy_scan.py`, at 23ef24a9's tree):

```
$ grep -n 'def scan_dir\|def scan_file\|def run_scan\|def _run_scan_body\|def wall_clock_budget\|class ScanTimeout' ci/scripts/gang_id_secrecy_scan.py
# re-derived directly against 23ef24a9's tree (git show 23ef24a9:ci/scripts/gang_id_secrecy_scan.py)
# this revision — round 3's own citation-round review found this block six lines stale
191:def scan_file(path: Path, needles: list[tuple[str, bytes]]) -> tuple[int, str]:
229:def scan_dir(root: Path, needles: list[tuple[str, bytes]]) -> list[tuple[int, str]]:
295:class ScanTimeout(Exception):
300:def wall_clock_budget(seconds: int):
343:def run_scan(
367:def _run_scan_body(
```

`scan_dir` (`:229-294` at 23ef24a9's tree; round 1 fixed the cycle-hang with a RECURSIVE `walk()`
closure — line numbers `190-234` at round 1's own `31c8aa64` — round 2's own closing audit
BLOCKed on that closure still being recursive, itself a SECOND way the scan could fail,
`RecursionError`, escaping uncaught as a bare exit 1/traceback; round 2 replaces it with an
EXPLICIT STACK (a plain Python `list`) — see §2b P-B for the full fix) tracks the REAL path of
every directory it enters in a `visited_dirs` set; a directory whose real path repeats is one
`"cyclic carrier"` UNEXAMINABLE finding, never a re-descent. `scan_file` (`:191-228` at 23ef24a9's
tree, moved from round 1's `66-96`) checks `stat.S_ISREG` explicitly before ever calling
`read_bytes()` — any other mode class (FIFO, socket, device) is refused by name (UNEXAMINABLE),
never opened. `wall_clock_budget` (`:300-321` at 23ef24a9's tree, moved from round 1's `254-271`,
a `contextlib.contextmanager` over `signal.alarm`) wraps the WHOLE scan body
(`run_scan`/`_run_scan_body`, `:343-366`/`:367-451` at 23ef24a9's tree, moved from round 1's
`179-183`) — round 2 also widens `run_scan`'s own `except` from `ScanTimeout` alone to `except
Exception`, so ANY scanner-internal failure (a `RecursionError` included, though the explicit
stack above no longer produces one; any other bug) is UNEXAMINABLE, never a traceback (§2b P-B).
Default budget 120s (`DEFAULT_BUDGET_SECS`/`GANG_ID_SCAN_BUDGET_SECS` env override,
`ci/scripts/gang_id_secrecy_scan.py, line 109` at 23ef24a9's tree — verified unchanged from the
transcript above's own earlier claim of `:109`), overridable via `--budget-secs`.

**Advisory, also fixed**: `id_needles` (`ci/scripts/gang_id_secrecy_scan.py, lines 118-142` at `31c8aa64`,
unchanged by round 2) now computes FOUR base64 variants — standard padded, standard un-padded,
URL-safe padded, URL-safe un-padded — rather than one. Round 2 adds a SECOND base64 advisory: a
line-wrapped base64 encoding (coreutils `base64`'s 76-column default, `openssl base64`'s 64-column
default) is matched too, via a whitespace-stripped copy of the scanned bytes computed lazily for
base64-labeled needles only (`_strip_whitespace`/`_scan_bytes`,
`ci/scripts/gang_id_secrecy_scan.py, lines 163-185` at 23ef24a9's tree, moved from round 2's `145-167` by round 2c,
§2c below — §2b advisory).

**Oracle**: `gang_id_secrecy_scan.py --self-test` gains, named (round 1): a cyclic directory
symlink (returns promptly, UNEXAMINABLE, never hangs); a FIFO under the pulled dir (UNEXAMINABLE,
"not a regular file"); a UNIX socket under the pulled dir (UNEXAMINABLE, same message); a
wall-clock budget expiry (`scan_dir` mocked to sleep past a 1s budget, UNEXAMINABLE, "wall-clock
budget"); base64-urlsafe and base64-unpadded planted-id hits — 23 cases at round 1 (up from 17 at
c4). Round 2 adds four more (§2b P-B/advisory): a tree 200 levels deep, walked under an
artificially lowered `sys.recursionlimit(40)`, completes and finds a planted id at the bottom
(proving the walk is not Python-stack-recursive at all, portable across the OS `PATH_MAX`
differences a literal 1,500-level fixture would hit inconsistently); `scan_dir` mocked to raise an
arbitrary `RuntimeError`, caught as UNEXAMINABLE, never a traceback; coreutils-width and
openssl-width line-wrapped base64 hits. 27 `unittest` cases total, all passing (`python3
ci/scripts/gang_id_secrecy_scan.py --self-test`, verified this round).

### F6 — the contract of record itself: the RUN_LOG claim, and this citation form

**(a)** The c4 contract's own §5 claimed the tee'd run log "IS the SAME file the workflow step
uploads" — false at that revision: `RUN_LOG="$(mktemp)"` created a file OUTSIDE
`.gpu-pull/gpu-cluster/` (the directory `.github/workflows/gpu-cluster.yml`'s
`actions/upload-artifact` step actually uploads,
`.github/workflows/gpu-cluster.yml, lines 152-156` at `5ebe53ab`, unchanged by this round), so the
uploaded artifact never actually carried the run log, and the id-secrecy scan's own artifact-dir
walk never actually covered it as a carrier via that path either (it was scanned only through the
SEPARATE, explicit `--log` argument).

**Fix** (`ci/scripts/runpod_gpu_cluster.sh`, at 23ef24a9's tree):

```
$ grep -n '^mkdir -p "\$CLUSTER_ARTIFACT_DIR"$\|^RUN_LOG="\$CLUSTER_ARTIFACT_DIR/run.log"$' ci/scripts/runpod_gpu_cluster.sh
784:mkdir -p "$CLUSTER_ARTIFACT_DIR"
785:RUN_LOG="$CLUSTER_ARTIFACT_DIR/run.log"
1021:mkdir -p "$CLUSTER_ARTIFACT_DIR"
```

(moved from round 2's `747`/`748`/`977` by round 2c's own module-header/`assembly_ok` growth
above this point, §2c below — round 2 itself moved these from round 1's `621`/`622`/`811` by
growing the module header's own P-A/P-C prose above this point — see §2b — without changing their
relative order or their own text) the directory is created BEFORE the tee starts (`:807`,
`exec > >(tee -a "$RUN_LOG") 2>&1`; round 2 also assigns `ASSEMBLED`/`id_landed=0` here, between
the `mkdir` and the `exec` — round 2c adds `assembly_ok=0` in the same span, §2c below — so the
EXIT trap's own P-A/P-A2 scan-or-destroy knows every global it reads on every exit arm — §2b
P-A, §2c P-A2), so the run log lives inside the uploaded/scanned directory from its first byte;
the SECOND `mkdir -p` match (`:1021`) is a defensive, idempotent re-assertion immediately before
the rank-log copies below — never a second, independent creation site with its own drift risk.
Both ranks' own remote logs are now ALSO copied there, unconditionally, pass or fail:

```
$ grep -n 'cp -f "\$rank0_log"\|cp -f "\$rank1_log"' ci/scripts/runpod_gpu_cluster.sh
1022:cp -f "$rank0_log" "${CLUSTER_ARTIFACT_DIR}/rank0.log" 2>/dev/null || echo "::warning::could not copy rank 0's own log into ${CLUSTER_ARTIFACT_DIR}"
1023:cp -f "$rank1_log" "${CLUSTER_ARTIFACT_DIR}/rank1.log" 2>/dev/null || echo "::warning::could not copy rank 1's own log into ${CLUSTER_ARTIFACT_DIR}"
```

placed immediately after both `wait` calls (`ci/scripts/runpod_gpu_cluster.sh, lines 1014-1023` at 23ef24a9's tree,
moved from round 2's `970-979` by round 2c's own module-header/`assembly_ok` growth above this
point, §2c below), before any pass/fail branching. The workflow's own upload step needed no
change — `path: .gpu-pull/gpu-cluster/` already covers the directory the run log now lives
inside.

**Oracle**: `test_gpu_cluster_lane.sh`'s F6(a) block
(`ci/scripts/test_gpu_cluster_lane.sh, lines 667-685` at 23ef24a9's tree, moved from round 2's `592-610` by round
2c's own P-A2 fixture growth above this point, §2c below; unchanged in shape) statically asserts
the `mkdir` line precedes the `RUN_LOG=` assignment line, and that both `cp -f` lines exist.

**(b) This citation form itself.** Every citation in this revision is the bare `path:line` (or
`path:line-line`) form, tagged `(at <sha>)`, re-derived by direct read against this tree AFTER
every fix above landed — never a fenced `grep -n`/`sed -n` transcript (the c4 revision's own
form, which carries zero tokens `check_path_line_citations`'s regex matches, and — separately —
had gone stale in 3 of 16 transcripts by the time of the closing audit, since a transcript's own
output is never re-verified by that checker at all).

## 2b. Fix round 2 — the closing adversarial audit's five findings, each closed by name

Round 1's own closing adversarial audit (2026-09-15) BLOCKed on five findings on round 1's OWN new
surfaces (never on the base this unit builds from): (A) the carrier set was uploaded on
`if: always()` while the id-secrecy scan ran only inline near the main body's own tail, so an
`exit` call between "the id lands" and that tail point left the carrier directory unscanned; (B)
the cycle-safe walk (§2 F5) was still RECURSIVE, itself a second way the scan could fail
(`RecursionError`), escaping the documented 0/1/2 exit lattice as a bare, uncaught exit 1 or a
traceback; (C) `world`/`hosts`/`pod_count`/`gpu_count_per_pod` were FOUR LITERALS in the assembler,
so `_gang_check_cluster_shape`'s own cross-field check (§2 F4) compared the artifact against
itself — a tautology, never a real falsification; (D) the assembler had no oracle beyond
hand-built fixture JSON — nothing drove the REAL `_rpc_assemble_gang_artifact` function and fed
its REAL output through the REAL checker; (E) this contract's own citations named a sha
(`d12e1689`) not reachable from the branch, and one file (§3) was claimed "unchanged" when it had
in fact changed. This round closes each by name below, landing as `c4f0c36e`.

### P-A — no byte reaches the upload unscanned, on EVERY exit arm

**Note (round 2c):** this whole subsection narrates round 2's OWN fix as it stood at `c4f0c36e`
(landing sha `c4f0c36e`) — every citation below this line, through the end of this subsection, is
tagged to THAT tree, not this contract revision's own HEAD; the function `_rpc_scan_or_quarantine`
names is renamed `_rpc_scan_or_destroy` and re-cited at its own current line numbers in §2c above.

**Fix** (`ci/scripts/runpod_gpu_cluster.sh`, at `c4f0c36e`'s own tree): the id-secrecy scan
invocation MOVED out of the main body's own tail entirely, into a new function called from the
EXIT trap itself:

```
$ grep -n '^_rpc_scan_or_quarantine()\|^id_landed=0$\|^id_landed=1$\|^ASSEMBLED=' ci/scripts/runpod_gpu_cluster.sh
668:_rpc_scan_or_quarantine() {
755:ASSEMBLED="${CLUSTER_ARTIFACT_DIR}/gang-cluster-$(date -u +%Y%m%d%H%M%S).json"
762:id_landed=0
917:id_landed=1
```

(at `c4f0c36e`'s own tree) `_rpc_scan_or_quarantine` (`:668-687` at that tree) runs
`_rpc_run_id_secrecy_scan` (unchanged, §2 F5) whenever the GLOBAL `id_landed` is `"1"`, and on
anything other than a clean scan MOVES the whole
`$CLUSTER_ARTIFACT_DIR` to a quarantine path under `$RP_WORK` (outside the uploaded
`.gpu-pull/gpu-cluster/` tree entirely) — or, when the move itself fails (e.g. a cross-device
`mv`), empties the directory in place — before returning the scan's own non-zero status to its
caller. Because `mv` preserves the inode, the reason line echoed immediately after the move still
lands as the (now-relocated) run log's own LAST line via the still-open `tee` file descriptor
(`exec > >(tee -a "$RUN_LOG")`, unchanged, `:763` at that same tree — `:807` at this contract's
own HEAD), and also reaches the job's own console
output regardless. `ASSEMBLED` (`:755`) and `id_landed=0` (`:762`) are both set as STABLE globals
at the very top of the executed block — before the tee even starts — so the trap's own read of
them is never garbage on an early exit; `id_landed` flips to `1` (`:917`) the moment the download
from the primary is ATTEMPTED (not only once it succeeds), so a partially-written or wrong-size
staging file on a failed download is still scanned (and refused as UNEXAMINABLE by its own
128-byte check, `_run_scan_body`, never silently skipped).

`_rpc_cleanup_cluster` (`ci/scripts/runpod_gpu_cluster.sh, lines 701-734` at `c4f0c36e`'s own tree — this
function keeps its own name; it is `_rpc_scan_or_quarantine`, the function it CALLS below, that is
renamed `_rpc_scan_or_destroy` and re-cited at its own current line numbers in §2c above; this
function's own current span is `738-771`, §2c P-A2/P-A3) now calls `_rpc_scan_or_quarantine`
unconditionally (joining its non-zero return into `rc` only
when `rc` was still `0` — the same "never overwrite a more specific existing failure" doctrine
F3(c) already uses), then deletes `$STAGING_ID_FILE` unconditionally (`:730` at that same tree)
regardless of `RP_SESSION`/`RP_WORK_IS_TEMP` (an advisory this round also closes — `rp_cleanup`'s
own conditional `rm -rf "$RP_WORK"` never ran at all when `RP_SESSION` was set, which would have
left the staging
id file on disk past process exit), THEN chains `rp_cleanup` (F2, unchanged) and exits. The OLD
inline scan call at the main body's own tail is GONE entirely — every exit arm, including the
natural fall-through at the bottom, now reaches the scan exactly once, through the trap.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh, lines 194-319` (at `c4f0c36e`'s own tree — see the
round-2c correction immediately below for what changed here) drives the REAL `_rpc_cleanup_cluster`
in a real subprocess against a REAL fixture carrier directory and the REAL scanner (never mocked),
on the four arms the closing audit named by name: "assembly refused" (pending rc=1, clean carrier
— the scan runs clean, rc=1 survives, the carrier is left intact, never quarantined for no reason),
"pull failed" (pending rc=1, a DIRTY carrier — the scan HITs, the carrier holds nothing at the
upload path afterward), "budget cut" (rc=124, clean carrier — the named exit code survives
verbatim), "wrong tree" (rc=77, DIRTY carrier — the scan HITs, the carrier is quarantined, AND the
named exit code still survives verbatim, proving the join never clobbers a more specific existing
failure). A fifth control (`id_landed=0`, the id never reached this runner) asserts the trap
invokes the scanner NOT AT ALL — nothing to protect against yet.

**Correction (round 2c):** the fixture this Oracle paragraph describes
(`run_trap_scan_arm`, `c4f0c36e`'s own tree) wrote `$pa_sandbox/assembled.json` UNCONDITIONALLY,
on every arm regardless of `$1`/`$2` — including the "assembly refused" and "budget cut" arms,
neither of which the REAL driver's own assembly phase ever reaches or writes anything for. That
made the "clean carrier is left intact" claim true of the FIXTURE, never of the real driver: the
real `_rpc_scan_or_quarantine` (this round renamed `_rpc_scan_or_destroy`, §2c below) passed
`$ASSEMBLED` to the scan UNCONDITIONALLY too, and the scanner's own (correct-by-design)
`--assembled-artifact` strictness then read the genuinely-missing file on those two arms as
UNEXAMINABLE, destroying the very run.log this paragraph claims survives. Reproduced directly
against `c4f0c36e`'s own driver and scanner, under §2c's corrected fixture (which no longer
pre-creates the file unconditionally): `gpu-cluster-lane: 70 passed, 7 failed`, naming exactly the
"assembly refused", "pull failed" (both expected clean, got UNEXAMINABLE), "budget cut" (expected
clean, got UNEXAMINABLE), and "wrong tree"/planted-id (expected HIT, got UNEXAMINABLE — the missing
file's own UNEXAMINABLE status outranks a real HIT in the scan's own `worst = max(...)` lattice,
so a genuine leak on that tree was ALSO mislabeled) arms as failing. §2c below is the fix; this
paragraph's own claims hold only from `23ef24a9` forward.

### P-B — the scan's own exit lattice is total, never a traceback

See §2b's own gang_id_secrecy_scan.py citations already folded into §5's revision above (the
explicit-stack `scan_dir` rewrite and `run_scan`'s widened `except Exception`). Restated here by
name: `scan_dir` (`ci/scripts/gang_id_secrecy_scan.py, lines 229-292` at 23ef24a9's tree, moved from round 2's
`211-274` by round 2c's own docstring growth above this point, §2c below) walks via an EXPLICIT
STACK (a plain Python `list`), never a recursive closure — the round-1 shape that made a
`RecursionError` possible in the first place. `run_scan` (`:343-366` at 23ef24a9's tree, moved
from round 2's `325-345`; round 2c also widens its own `assembled_artifact` parameter type to
`Path | None`, §2c below) wraps `_run_scan_body` in `except Exception`, not only
`except ScanTimeout` — ANY scanner-internal failure is UNEXAMINABLE (2), never exit 1, never an
uncaught traceback.

**Oracle**: `gang_id_secrecy_scan.py --self-test` gains `test_deep_tree_well_beyond_the_recursion_limit_does_not_crash`
(`ci/scripts/gang_id_secrecy_scan.py, lines 762-785` at 23ef24a9's tree, moved from round 2's `691-714` by round 2c's
own docstring/self-test growth above this point, §2c below — a 200-level-deep tree walked under an
artificially lowered `sys.setrecursionlimit(40)`, portable across OS `PATH_MAX` differences a
literal 1,500-level fixture would hit inconsistently on different filesystems; the planted id at
the bottom is still found, proving the walk reaches full depth) and
`test_scan_dir_raising_an_unexpected_exception_is_unexaminable_not_a_traceback`
(`ci/scripts/gang_id_secrecy_scan.py, lines 788-803` at 23ef24a9's tree, moved from round 2's `717-732` by round 2c,
§2c below — `scan_dir` mocked to raise a bare `RuntimeError`, asserted UNEXAMINABLE, never
propagated).

### P-C — the artifact's shape is measured, never four literals

**Fix** (`ci/scripts/runpod_gpu_cluster.sh`, at 23ef24a9's tree): a new pure helper parses the cluster's own
`compute` block from the SAME `Cluster` object `rp_cluster_get` already returns (unmodified,
`ci/scripts/runpod_lib.sh:1537-1557`, untouched by either round):

```
$ grep -n '^_rpc_parse_cluster_shape()\|cluster_body="\$(rp_cluster_get' ci/scripts/runpod_gpu_cluster.sh
342:_rpc_parse_cluster_shape() {
813:cluster_body="$(rp_cluster_get "$cluster_id")" || { echo "::error::could not read back the created cluster's own shape"; exit 97; }
```

`_rpc_parse_cluster_shape` (`:342-362` at 23ef24a9's tree) prints `"podCount gpuCountPerPod"` on a successful
parse of `compute.podCount`/`compute.gpuCountPerPod` (both required positive integers), or exits 2
on anything else. The executed block (`:807-821` at 23ef24a9's tree) calls it right after cluster create,
BEFORE any member work starts, and REFUSES (exit 97, "wrong shape" — the exit code the module
doc's own EXIT CONTRACT already named for this case, never actually checked against a measurement
until now) when the measured shape disagrees with `RP_CLUSTER_POD_COUNT`/
`RP_CLUSTER_GPU_COUNT_PER_POD`. `MEASURED_POD_COUNT`/`MEASURED_GPU_COUNT_PER_POD` are then threaded
into `_rpc_assemble_gang_artifact`'s own argv (`:1010-1013` at 23ef24a9's tree) — the four literals
(`"world": 2`, `"hosts": 2`, `"pod_count": 2`, `"gpu_count_per_pod": 1`) are GONE from the
assembler's own python (`ci/scripts/runpod_gpu_cluster.sh, lines 508-620` at 23ef24a9's tree, moved from round 2's
`501-613` by round 2c's own module-header growth above this function, §2c below — is the whole
function, signature grown from `$1..$5` to `$1..$7`); `gang.world`/`gang.hosts` are now computed
as `pod_count * gpu_count_per_pod`/`pod_count` from whatever the driver threads in.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh, lines 733-877` (at 23ef24a9's tree, moved from round 2's `658-802`
by round 2c, §2c below; shared with P-D below) drives
the REAL `_rpc_assemble_gang_artifact` with a fixture 3x8 shape (simulating a create response
`rp_cluster_get` would have returned for a differently-shaped cluster) against the SAME two rank
reports the happy-path case uses, and asserts the resulting artifact carries `hosts=3 world=24
pod_count=3 gpu_count_per_pod=8` — then feeds that artifact through the REAL `check_gang_artifact`
and asserts it is REFUSED (the checker's own `_gang_check_hosts`/`_gang_check_cluster_ranks` both
catch it independently: `gang.hosts` fixed at `GANG_CLUSTER_HOSTS = 2` is never 3, and `gang.ranks`
carries only 2 entries against a `gang.world` of 24) — the cross-field check is now genuinely
falsifiable, never a tautology.

### P-D — the assembler is oracled through the sourced driver, on every arm

**Fix**: no new production code beyond P-C's own signature change above; this finding is closed
entirely by the oracle itself, since the round-1 gap was "no test drives the real function", not a
mechanism defect.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh, lines 733-877` (at 23ef24a9's tree, moved from round 2's `658-802`
by round 2c, §2c below) sources the driver (unchanged,
`test_gpu_cluster_lane.sh`'s own G0 block already proves sourcing makes no network call) and calls
the REAL `_rpc_assemble_gang_artifact` on: the happy path (2x1, two ranks, matching digests —
assembly succeeds, and the resulting artifact passes the REAL `check_gang_artifact`, imported
directly from `check_cuda_run_artifacts.py` via `importlib`, never paraphrased or reimplemented);
an unresolved ("unknown") hostname (REFUSES, no artifact written); an unresolved iface (REFUSES);
two ranks reporting the same host differing only in CASE (REFUSES — see the advisory below); and a
digest disagreement between two `pass` ranks (REPRESENTABLE — assembly SUCCEEDS with
`verdict: "fail"` and a named reason, never refused at assembly time, and the resulting artifact
still passes the checker as legitimately-recorded evidence of a failed run).

### Advisories folded

- **Host comparison is case-insensitive on BOTH sides of the producer/checker boundary.** The
  assembler's own `_norm_host` (`ci/scripts/runpod_gpu_cluster.sh, lines 534-538` at 23ef24a9's tree, moved from
  round 2's `527-531` by round 2c, §2c below) and the checker's own `_gang_check_cluster_ranks`
  (`ci/scripts/check_cuda_run_artifacts.py:1452-1455` at 23ef24a9's tree, plus a self-test arm at
  `ci/scripts/check_cuda_run_artifacts.py:3361-3373`) both compare `.strip().casefold()` rather
  than raw string equality — "Host-A" and "host-a" are the same host on both sides now, never a
  false "two hosts" on one side and a false negative on the other.
- **Base64 needle matching survives line-wrapping.** See §5's revision above
  (`ci/scripts/gang_id_secrecy_scan.py, lines 163-185`, moved from round 2's `145-167` by round 2c, §2c
  below, `_strip_whitespace`/`_scan_bytes`) — a whitespace-stripped copy of the scanned bytes is
  checked for base64-labeled needles, lazily, only when the raw check misses.
- **`_strip_trailing_comment` handles a backslash-escaped apostrophe.** See §6's revision above
  (`ci/scripts/check_gpu_prove_once.py:477-479`).
- **The no-op `grep -v '^\s*#'` at `test_gpu_cluster_lane.sh` (round 1's own `:547`) is replaced by
  an assertion that actually filters.** `grep -n`'s own output is `"N:text"` — a line ALWAYS
  starts with digits, so a filter anchored at `^\s*#` against THAT text can never match anything (a
  no-op that happened to be harmless only because the real file had nothing to filter). Fixed
  (`ci/scripts/test_gpu_cluster_lane.sh, line 891` at 23ef24a9's tree, moved from round 2's `816` by round 2c's own
  P-A2 fixture growth above this point, §2c below: `grep -vE '^[0-9]+:[[:space:]]*#'`, stripping
  `grep -n`'s own prefix before testing for a leading `#`) and PROVEN to actually filter against a
  synthetic two-line fixture (`ci/scripts/test_gpu_cluster_lane.sh, lines 899-909` at 23ef24a9's tree, moved from
  round 2's `824-834` by round 2c: a comment-only line naming the target string on line 1, the
  real call on line 2 — the filtered result must land on line 2, not line 1).
- **The staging id file is deleted on EVERY exit after the scan, regardless of
  `RP_SESSION`/`RP_WORK_IS_TEMP`.** Folded into P-A above (`ci/scripts/runpod_gpu_cluster.sh, line 767`
  at 23ef24a9's tree, moved from round 2's `730` by round 2c, §2c below).
- **The tee flushes before the trap exits, so the trap's own last line reaches `run.log`.** Proven
  by construction in P-A's own oracle: every `run_trap_scan_arm` case reads the SAME
  `$out`/`scan_out` the subprocess's stdout+stderr produced, which is the identical stream `tee`
  writes to disk — the DESTROY reason line (round 2c wording, §2c P-A3) is asserted present in
  that captured output on every dirty arm.
- **`ci/scripts/runpod_gpu_prove.sh:196`'s discarded `rp_sweep` rc is now logged loudly, named as
  pre-run hygiene, never a proof failure.** This is the FIRST time either docs-ci fix round has
  touched this file (round 1 did not); `:196` was the pre-fix line the brief itself pointed at,
  now moved to `:200` by four lines of new explanatory comment prepended above it:

```
$ grep -n 'rp_sweep ||' ci/scripts/runpod_gpu_prove.sh
200:rp_sweep || echo "::warning::pre-run rp_sweep (orphan hygiene) failed rc=$? -- not a proof failure, continuing"
```

  (`ci/scripts/runpod_gpu_prove.sh:191-200` is the full comment explaining why a sweep failure
  here is orthogonal to THIS run's own proof.) This file carries no dedicated fixture suite of its
  own beyond `test_gpu_prove_lane.sh`'s existing sourced-driver coverage (unaffected by this
  one-line change, since the sourced-execution guard skips this line entirely);
  `test_gpu_prove_lane.sh` (74 cases, verified this round, `bash
  ci/scripts/test_gpu_prove_lane.sh`, exit 0) stays green unchanged.

**Round-3 stop rule status**: not triggered — round 2 closed all five findings on the first
attempt, never blocking a second time on the same mechanism (`M3`/`M4`).

## 2c. Fix round 2c — a lead-found defect on round 2's own new surface, closed by name

**This is not a third audit BLOCK.** The lead found this defect on the tree at `ebe79a0d` (round
2's own closing state) while preparing the closing adversarial audit, and dispatched it as a
small, lead-initiated fix (docs-ci) BEFORE that audit re-ran — §6's round-3 stop rule (a third
BLOCK on the driver excises the whole M3/M4 surface) is NOT triggered by this round: it closes one
defect by name, the same discipline §2b's own five findings used, never a mechanism trade.

**The defect.** `_rpc_scan_or_quarantine` (round 2's own name, at
`c4f0c36e`'s tree) ran on EVERY exit arm once the id had landed (P-A, correct), but passed
`"$ASSEMBLED"` to `_rpc_run_id_secrecy_scan` UNCONDITIONALLY, and `gang_id_secrecy_scan.py`'s own
`--assembled-artifact` was a REQUIRED argument whose absence-at-that-path was always UNEXAMINABLE
— exactly the scanner's own intended happy-path strictness (`test_missing_assembled_artifact_is_
unexaminable_never_clean`, self-test, unchanged by this round). The gap was never in the scanner:
it was that the driver called it as if EVERY arm claims to have produced an assembled artifact,
when in fact assembly is only ever REACHED on the tail of a passing pull; a failed pull (`:999`
at that tree), a refused assembly (`:1013`/`:1015`), a budget cut, an inactivity kill, or a
wrong-tree cut all exit LONG before `$ASSEMBLED` is ever written — and on every one of those arms
the scan came back UNEXAMINABLE for a file nobody ever promised, and `_rpc_scan_or_quarantine`
then MOVED the entire carrier directory (the run log, both ranks' own logs — the ONLY evidence a
reviewer has on exactly those arms) to a path under `$RP_WORK`, which `_rpc_cleanup_cluster`'s
own chained `rp_cleanup` then unconditionally `rm -rf`'d before the process exited. The driver's
own header already states this exact principle at `:999` ("a leg with no retrievable evidence
proves nothing reviewable") — this fix makes the EXIT trap honor it instead of contradicting it.
Reproduced directly (never merely asserted) against `c4f0c36e`'s own driver and scanner under
this round's own corrected fixture (§2c's own oracle, below): `gpu-cluster-lane: 70 passed, 7
failed`, naming the "assembly refused", "pull failed" (both expected CLEAN, got UNEXAMINABLE),
"budget cut" (expected CLEAN, got UNEXAMINABLE), and "wrong tree"/planted-id (expected HIT, got
UNEXAMINABLE — the missing-file finding outranks a real HIT in the scan's own `worst = max(...)`
lattice, so a genuine leak on that tree would ALSO have been mislabeled, not just under-reported)
arms as failing; two further failures in that same run are an artifact of this reproduction's own
partial sibling-file reconstruction (a missing `check_gpu_parity_matrix`/`gpu_prove_verdict`
import, unrelated to this fix) and are not part of this defect.

**Fix — P-A2 (absent vs. unexaminable).** `gang_id_secrecy_scan.py`'s `--assembled-artifact`
becomes OPTIONAL (`main`, `ci/scripts/gang_id_secrecy_scan.py, lines 454-467,475-489` at 23ef24a9's tree: removed
from the required-argument check; `run_scan`/`_run_scan_body`
(`ci/scripts/gang_id_secrecy_scan.py, lines 343-367` at 23ef24a9's tree, their own signatures) widen the parameter's
type to `Path | None`, and the required-carrier loop
(`ci/scripts/gang_id_secrecy_scan.py, lines 403-421` at 23ef24a9's tree) is split into `log` — ALWAYS required —
and `assembled_artifact` —
required ONLY when not `None`). The driver threads a NEW global, `assembly_ok` (declared `0`
alongside `id_landed=0`, `ci/scripts/runpod_gpu_cluster.sh, lines 800-806` at 23ef24a9's tree), set to `1` ONLY
immediately after `_rpc_assemble_gang_artifact` itself returns `0`
(`ci/scripts/runpod_gpu_cluster.sh, lines 1055-1068` at 23ef24a9's tree — the call site, previously a bare `||` refusal, is now an
`if`/`else` so the success arm can set the flag; a `1`/`2` refusal, which never writes the file,
leaves `assembly_ok=0`). `_rpc_run_id_secrecy_scan`'s own 4th argument becomes OPTIONAL
(`ci/scripts/runpod_gpu_cluster.sh, lines 638-644` at 23ef24a9's tree: an empty string omits `--assembled-artifact`
from the invocation entirely, never passing an empty path), and `_rpc_scan_or_destroy` (renamed,
below) passes `"$ASSEMBLED"` ONLY when `assembly_ok=1` (`ci/scripts/runpod_gpu_cluster.sh, lines 704-725`
at 23ef24a9's tree, specifically `:707`). Because `$ASSEMBLED` always lives INSIDE `$CLUSTER_ARTIFACT_DIR` in
this driver's own usage, omitting the flag never widens what gets scanned: the directory walk
(`scan_dir`) still covers whatever bytes actually exist there, so a stray or leaked file at that
exact path is still caught (proven directly, §2c's own oracle below); only the "must-exist"
requirement is what becomes conditional. The scanner's own happy-path strictness is UNCHANGED:
when the caller DOES pass `--assembled-artifact` (i.e. `assembly_ok=1`) and the file is missing —
a corruption or race this driver's own author did not anticipate, never an ordinary refusal arm —
the scan is still UNEXAMINABLE, never clean (proven directly through the driver's own
`assembly_ok` wiring, not only the scanner in isolation — §2c's own oracle, arm (g) below).

**Fix — P-A3 (honest wording).** The carrier removal on a dirty scan IS destruction: the
relocation target lives under `$RP_WORK`, and `_rpc_cleanup_cluster`'s own chained `rp_cleanup`
call (`ci/scripts/runpod_gpu_cluster.sh, line 769` at 23ef24a9's tree) unconditionally `rm -rf`s `$RP_WORK` before
the process exits whenever `RP_WORK_IS_TEMP=1` — the default this driver's own workflow runs
under (`.github/workflows/gpu-cluster.yml` sets neither `RP_SESSION` nor `RP_WORK`, so
`runpod_lib.sh`'s own `rp_init` takes the `mktemp -d`/`RP_WORK_IS_TEMP=1` branch on every CI run —
`ci/scripts/runpod_lib.sh:315-317`, unchanged by this round). A CI runner torn down at process
exit is not somewhere a human can later inspect anything, so "quarantine" was never an accurate
word for what this trap does; every site is renamed: the function itself
(`_rpc_scan_or_quarantine` → `_rpc_scan_or_destroy`, `ci/scripts/runpod_gpu_cluster.sh, line 704` at
23ef24a9's tree), its own header (`:665-703` at 23ef24a9's tree), the `::error::` line
(`ci/scripts/runpod_gpu_cluster.sh, line 718` at 23ef24a9's tree — the relocation is stated to exist ONLY so
the upload step, which starts concurrently and could otherwise race an in-place deletion, can
never see a half-removed directory), the module-level header's own carrier-set paragraph
(`ci/scripts/runpod_gpu_cluster.sh, lines 65-77` at 23ef24a9's tree), the EXIT CONTRACT paragraph
(`ci/scripts/runpod_gpu_cluster.sh, lines 114-118` at 23ef24a9's tree), and this contract (§2b's own P-A paragraph,
corrected above; §4, §11).

**Oracle.** `gang_id_secrecy_scan.py --self-test`: **29 tests, up from 27** at round 2's own close.
Self-check, both counts, reproducible from any checkout of this branch: `git show
c4f0c36e:ci/scripts/gang_id_secrecy_scan.py` written to a scratch copy and run with
`--self-test` prints `Ran 27 tests ... OK` (round 2's own tree); `python3
ci/scripts/gang_id_secrecy_scan.py --self-test` at this round's own HEAD prints `Ran 29 tests ...
OK`, exit 0 — the two new cases are `test_assembled_artifact_omitted_entirely_is_not_required_
and_stays_clean` (`ci/scripts/gang_id_secrecy_scan.py, lines 632-643` at 23ef24a9's tree) and
`test_assembled_artifact_omitted_but_a_leak_at_that_path_is_still_a_hit`
(`ci/scripts/gang_id_secrecy_scan.py, lines 645-660` at 23ef24a9's tree). `test_gpu_cluster_lane.sh`'s P-A/P-A2
block (`ci/scripts/test_gpu_cluster_lane.sh, lines 194-382` at 23ef24a9's tree) is REWRITTEN, not merely extended:
the round-2 fixture (`run_trap_scan_arm`) pre-created `assembled.json` UNCONDITIONALLY regardless
of arm, which is exactly what masked this defect from round 2's own oracle (see the corrected P-A
paragraph, §2b, above) — the corrected fixture takes a THIRD parameter, `assembly_claimed`, and
only writes the file when the arm being simulated actually claims assembly succeeded, matching
the real driver's own `assembly_ok` semantics exactly. Eight arms now (up from round 2's five):
(a) "assembly refused" — never claimed, clean, expects CLEAN + the arm's own rc + an intact
carrier (the exact defect's own repro case); (b) "pull failed" — never claimed, clean, same
expectation (the SECOND arm named in the round-2 audit's own P-A correction); (c) a planted-id
refusal arm — never claimed, dirty, expects HIT + DESTROYED (the "planted-id refusal arm
asserting removal" oracle); (d) "budget cut" — never claimed, clean, expects CLEAN + rc=124
preserved; (e) "wrong tree" — never claimed, dirty, expects HIT + DESTROYED + rc=77 preserved;
(f) the happy-path tail — claimed, file present, clean, expects CLEAN (proving `assembly_ok=1`
does not itself break the ordinary pass arm); (g) "claimed but missing" — claimed, file absent,
expects UNEXAMINABLE + DESTROYED (P-A2's happy-path strictness, now proven through the REAL
driver's `assembly_ok` wiring, not only the scanner in isolation); (h) `id_landed=0` — the control,
unchanged, asserts the scanner never runs at all. `bash ci/scripts/test_gpu_cluster_lane.sh`:
`gpu-cluster-lane: 77 passed, 0 failed`, exit 0, at this round's own HEAD.

**Round-3 stop rule status**: not triggered (see this section's own opening note).

## 2d. Fix round M1 (this revision) — round-4's SECOND block on the excised tree, the four
findings closed by name

Round 4 audited the excised tree (round 3's own M1/M2/M5/M6 scope) and BLOCKed on four executed
findings plus advisories, per `CONTRACT-U7b.md` §11 (the round's own pre-committed stop rule:
**round 5 is the LAST closer round for this unit**). Every citation below is `(at HEAD)` into
this fix round's own commit (`git log --format=%h main..HEAD` at this contract's own commit lands
this file second, atop the fix commit — the `(at HEAD)` tree the two share).

**F1 — three read sites on the reap path aliased "could not parse" onto "the answer is no".**
`rp_cluster_sweep`'s post-delete confirmation (`ci/scripts/runpod_lib.sh, lines 1900-1948` at
HEAD) ran `json.load` with no `try`/`except` at all — an unparseable, empty, or bare-array
second-GET body threw an UNCAUGHT Python exception, and Python's own default exit code for an
uncaught exception (1) collided EXACTLY with the explicit `sys.exit(1)` "confirmed gone" arm,
so a malformed re-enumeration read as a clean success (`rc=0`, "terminated N orphaned
cluster(s)", a traceback on stderr) while the PRE-delete parse's own `try`/`except`
(`ci/scripts/runpod_lib.sh, lines 1842-1845` at that round's tree) was three-valued at the
`json.load` only — round 5 (§2e) found the code AFTER it was not, and made every parser total. **Fix**: the post-delete parse (`ci/scripts/runpod_lib.sh, lines 1913-1934` at
HEAD) wraps `json.load` in `try`/`except` and adds `isinstance` guards on the body and its
`clusters` list, naming every parse/shape failure `sys.exit(2)` — never falling through to
Python's own default exit code. `rp_cluster_create` (`ci/scripts/runpod_lib.sh, lines 1539-1571`
at HEAD), `rp_cluster_get` (`:1581-1609`), `rp_cluster_pods` (`:1621-1663`) and `rp_cluster_list`
(`:1673-1707`) each gain the identical three-valued shape: exit 0 (the key is present), exit 1 (a
well-formed object missing the required key), exit 2 (unparseable/wrong-shaped body,
`ci/scripts/runpod_lib.sh:1553-1555` being `rp_cluster_create`'s own pair of `sys.exit(2)` sites)
— the bash dispatch around each (e.g. `ci/scripts/runpod_lib.sh:1565`) names a 201/200
unparseable body "unparseable", never "missing the required key". `rp_terminate`
(`ci/scripts/runpod_lib.sh, lines 388-432` at HEAD) no longer reads an unparseable podTerminate
body as `sys.exit(0)` ("no errors" — success, `:415`/`:418` are the two new named-refusal prints
replacing that arm); the doctrine contradiction between the cluster-member exclusion set's own
doc (previously "belt-and-suspenders") and `rp_sweep`'s own doctrine comment ("FAIL-CLOSED, not
belt-and-suspenders") is reconciled to the FAIL-CLOSED reading at `ci/scripts/runpod_lib.sh,
lines 1767-1779` (at HEAD; `:1771` inside that same span is `rp_sweep`'s own doctrine phrase
being echoed back, never contradicted, by the rewritten comment). **CHANGE, not merely a fix**:
`rp_sweep`'s own UNAGEABLE-pod arm (`ci/scripts/runpod_lib.sh, lines 2749-2762` at HEAD, the
`return 1` on the span's own last line) now bails immediately and names the pod id, mirroring
`rp_cluster_sweep`'s pre-existing UNAGEABLE handling — pre-fix this was a silent `continue` that
let the sweep finish green (`rc=0`); post-fix, a genuinely-reapable orphan later in the SAME
`out` list was left unswept for this run — a trade-off round 5 refuted (§2e): a missing `createdAt` is a
STATIC property, so "the next scheduled sweep" would bail at the same entry forever; both arms now name
the unjudgeable resource and still sweep the rest of the list.

**Oracle**: `ci/scripts/test_runpod_cluster_lib.sh` Group 9 (post-delete confirmation:
unparseable/empty/array second-GET bodies, and a 429 on that same second GET via
`MOCK_CLUSTER_LIST_STATUS_2`, never previously set by any fixture on this tree — round-4's own
citation), Group 10 (`rp_terminate` HTML-body refusal via `rp_sweep`), Group 11 (the pod
UNAGEABLE fix); `ci/scripts/test_gpu_dev_lifecycle.sh` gains a matching Group 4c driving the same
fix through this suite's own `rp_gql`-override harness. `bash ci/scripts/test_runpod_cluster_lib.sh`:
`runpod-cluster-lib: 50 passed, 0 failed`, exit 0. `bash ci/scripts/test_gpu_dev_lifecycle.sh`:
`gpu-dev-lifecycle: 157 passed, 0 failed, 0 skipped`, exit 0.

**F2 — `rp_cluster_create`/`rp_cluster_get` had zero non-comment invocations anywhere on this
tree.** `test_runpod_cluster_lib.sh`'s own header (`:5-6`) named both functions since commit c1,
but the suite body never called either — only the header COMMENT mentioned them, which
`drop_comment_lines` (`ci/scripts/check_gpu_prove_once.py`) strips, so neither function was even
DERIVED as a caller of its own root by P7's own scan (§6). **Fix**:
`ci/scripts/test_runpod_cluster_lib.sh` Group 7 (`rp_cluster_create`: 201 with an id, 201 with no id — named
distinctly from a 201 unparseable body — a 201 unparseable body, and a non-201 refusal) and Group
8 (`rp_cluster_get`: 200 valid, 200 missing the `id` key, 200 unparseable, 404, 500) each drive
the real function through the suite's own mock `curl` stub. `bash
ci/scripts/test_runpod_cluster_lib.sh`: 8 new PASS lines under `G7`/`G8`, all passing (see the F1
oracle line above for the suite's own total).

**F3 — the P7 clearance reason was misstated, and the gate's own self-match was undisclosed.**
`ci/scripts/check_gpu_prove_once.py`'s `PAID_POD_LANE_TABLE` comment (`:888-895` at `9d356dff`'s
tree, this round's own parent commit, before this round) claimed "P7's completeness rule clears
it through `_check_derived_driver_cannot_rent`" — false: `rp_cluster_create` is a ROOT
(`RENTING_ROOTS`), never itself subject to that predicate (which runs only on DERIVED DRIVERS —
other tracked files whose text MENTIONS a closure member); `rp_cluster_create` has no CALLER on
this tree at all, so it contributes no derived driver for P7 to hold to a row or to that
predicate. Verified empirically against the real tree (`derive_deploy_closure`/
`derive_renting_drivers`, `ci/scripts/check_gpu_prove_once.py`) that this file's OWN source
self-matches its own `RENTING_ROOTS`/
`PAID_POD_LANE_TABLE` definition (`derived["ci/scripts/check_gpu_prove_once.py"] ==
['_rp_deploy_payload', 'rp_cluster_create', 'rp_deploy_arch']`) and clears
`_check_derived_driver_cannot_rent` with zero findings and zero notes — a real, PRE-EXISTING,
already-tested self-reference (`ci/scripts/test_check_gpu_prove_once.py`'s own
`test_the_real_tree_derives_the_set_this_suite_claims`, `:1297-1355` at HEAD, already asserted
this file is in the derived set before this round touched anything). **Fix**: the true reason is
now stated at `ci/scripts/check_gpu_prove_once.py, lines 888-919` (at HEAD, the `PAID_POD_LANE_
TABLE` comment's rewrite) and `:960-968` (at HEAD, the "WHAT THE DERIVATION DELIBERATELY DOES
NOT DO" residual bullet's own extension, naming THIS FILE alongside its pre-existing
`test_check_gpu_prove_once.py` disclosure at `:952-959`); `docs/maintainer/dev-gpu.md, lines
859-877` (at HEAD, the "Schedule visibility (P8)" paragraph) and UNITS.md's own U7b section
(`docs/plans/67-distributed-training/UNITS.md, lines 347-356` at HEAD) restate the same true
reason. This is
DISCLOSED, not "fixed" as a defect — the module's own pre-existing design philosophy ("nothing
exempted for being ours") already applies identically to `test_check_gpu_prove_once.py`'s own
self-match on `_rp_deploy_payload`; fixing the CLASSIFICATION (adding an exemption) would have
been the actual regression here. `ci/scripts/test_check_gpu_prove_once.py`'s own comment on
`test_runpod_cluster_lib.sh`'s self-match (`:1342-1351` at HEAD) is corrected to name Group 7
(F2, above) — `rp_cluster_create` is now genuinely CALLED there, not merely mentioned in a
stripped header comment.

**Oracle**: `python3 -m unittest ci.scripts.test_check_gpu_prove_once`: `Ran 207 tests ... OK`,
exit 0 — including the real-tree anti-vacuity assertion above, unperturbed by this round's
comment-only edits (no closure member is mentioned in a NEW non-comment line by this round's own
prose additions; verified: `derive_renting_drivers` against the post-fix tree still returns the
identical derived set, plus `ci/scripts/test_runpod_cluster_lib.sh` now ALSO matching
`rp_cluster_create` for a real reason, F2 above). `python3 ci/scripts/check_gpu_prove_once.py`:
`gpu-prove-once: OK ...`, exit 0.

**F4 — the fixed 2×1 shape was stated nowhere a caller looks.** `_rp_cluster_payload`
(`ci/scripts/runpod_lib.sh:1488` at 23ef24a9's tree, unchanged by round 3) hardcoded
`gpuCountPerPod: 1, podCount: 2` while its own doc, `rp_cluster_create`'s own doc, the
cluster-primitives section header (`"N member pods"`), the reviewed key-set fixture's own
comment, and `docs/maintainer/dev-gpu.md:802` (at `9d356dff`'s tree, before this round) never
stated that shape is FIXED, never parameterised. **Fix**: `ci/scripts/runpod_lib.sh, lines
1458-1466` (at HEAD, the section header) and `:1479-1499` (at HEAD, `_rp_cluster_payload`'s own
doc) and `:1524-1538` (at HEAD, `rp_cluster_create`'s own doc) each state the fixed 2×1 request
directly; `ci/scripts/fixtures/runpod_cluster_create_request_keys.json:3` (at HEAD, a new `_note`
field) states the same; `docs/maintainer/dev-gpu.md, lines 763-769` (at HEAD, the cluster-leg
section's own intro) replaces "N member pods" with the same fixed statement. "N member pods" no
longer appears anywhere on this tree (`git grep -n 'N member pods'` — empty).

**Advisories closed**: `docs/maintainer/dev-gpu.md`'s by-hand procedure (steps 4-5, `:812-826` at
HEAD) now exports `JAMMI_REQUIRE_CUDA_TWO_HOSTS=1` on BOTH hosts (rank 0 at `:815`, rank 1 at
`:825`) — previously implied nowhere in the procedure text, so a maintainer following it by hand
would get a silent SKIP on a misconfigured host rather than the hard FAIL this leg's own
require-gate flag exists to force (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:57-62` at
`b480f2dc`, unchanged, documents the flag itself).

**Advisory, recorded UNCOVERED, not closed**: the REST v2 Bearer-auth transport path itself
(`_rp_rest`, `ci/scripts/runpod_lib.sh, lines 369-386` at HEAD) has never executed against a live
RunPod endpoint anywhere on this tree — every test suite that reaches it (`test_runpod_cluster_
lib.sh`, `test_gpu_dev_lifecycle.sh`, `test_gpu_gang_lane.sh`, `test_gpu_prove_lane.sh`) stubs
`curl` on `PATH`; the live pre-flight that would exercise it for real is blocked on the user
(§1). This is not a defect this round closes — it is named here and carried into §9 below,
exactly as §1's own REST v2 `args`-field gap already was.

**Citation-round self-check, executed for THIS revision** (§13's own convention, re-run after
every edit in this file landed):

```
$ grep -rln 'runpod_gpu_cluster\|gpu-cluster.yml\|gang_id_secrecy_scan\|test_gpu_cluster_lane' ci .github crates docs
```

matches only `docs/rigor/contracts/feat_500-C-U7b.md` itself (104 total line matches, `grep -rn`)
— never `docs/plans/67-distributed-training/UNITS.md` or `README.md`, and never a live `ci/`,
`.github/`, or `crates/` path. Round 3's own self-check paragraph (§13, below) claimed the grep
ALSO matched `UNITS.md`'s own U7b-A2b filing; re-executed against this round's own tree, it does
not — the U7b-A2b filing text (`docs/plans/67-distributed-training/UNITS.md`, the "U7b-A2b" H3
section) describes the excised driver conceptually without repeating any of the four deleted
paths as a bare string. §13's own paragraph is corrected below to state this round's real output.

**Round-4/M1 stop rule status**: `CONTRACT-U7b.md` §11's own pre-committed rule — **round 5 is
the LAST closer round for this unit.** A PASS on round 5 ships M1+M2+M5+M6 whole; a BLOCK of any
kind withholds the whole unit from wave 3 (nothing merges; U7b is refiled whole, A2 + A2b, with
all five rounds — this one included — as its spec; the reap cron keeps main's pod-only sweep).

## 2e. Round 5 (final closers, 2026-09-15) — the reap path's judgement lattice; the user's takeover

Round 5's closers: discipline PASS; citation BLOCK on one range endpoint (`RankReport` cited `:425-458`, the
struct closes at `:461` — corrected above); audit BLOCK with three executed findings on M1 and two advisories.
At that point the user took the unit over from the swarm ("implement the fixes, test and create the PRs and close
wave 3"); the pre-committed round-5 withhold was therefore NOT applied, the lead applied the fixes below directly,
and the closers were not re-run.

**F-A — one unjudgeable resource must not suppress the sweep, and the remedy must be real.** Both sweeps bailed
(`return 1`) on the FIRST pod or cluster with no usable `createdAt`, before the operator override was even read,
so the override the error message named was inert and a real orphan behind the unjudgeable entry billed forever
(executed by the audit: three runs, zero terminates). Now (`rp_cluster_sweep` and `rp_sweep`, `ci/scripts/runpod_lib.sh`
at HEAD): an unjudgeable resource — no usable `createdAt`, OR a prefixed name with no parseable `-ttl<H>` — is
collected and NAMED with its by-id remedy (`rp_cluster_delete <id>` / `rp_terminate <id>`; the override cannot
help, there is no age to apply it to), the rest of the list is still judged and swept THIS run, and the sweep
exits 1 at the end so the reap cron reddens until the resource is retired by id. Oracles: Group 12
(`ci/scripts/test_runpod_cluster_lib.sh` — an unageable pod beside a 50 h orphan: the orphan is terminated,
exactly once; the unageable pod is named and never terminated; rc 1; the same under `rp_sweep 8`; the cluster
arm likewise with `rp_cluster_delete`).

**F-B — a prefixed cluster with no parseable deadline was DELETED at any age.** The Python printed an
`unparseable-deadline` row and the bash loop deleted every non-UNAGEABLE row (executed by the audit: a
60-second-old `jammi-cluster-experiment`, gone with a green summary). Now both sweeps print `UNPARSEABLE` for
that state and it joins the unjudgeable set above — named, never deleted. The pod mirror (pre-existing on
`main`) is corrected the same way. Oracles: Group 13 (cluster and pod).

**F-C — every parse on the reap path is total.** Round 4 wrapped `json.load` and added `isinstance` guards on
five sites but left the code AFTER the load non-total, so a row of the wrong shape raised an uncaught exception
whose exit 1 the callers reported as "200 but the body is missing the required 'pods'/'clusters' key" — a named,
wrong diagnosis — or as an empty reason with a traceback. Now `rp_cluster_pods`, `rp_cluster_list`,
`rp_cluster_sweep`'s and `rp_sweep`'s row loops run under one `try`/`except` each: a row that cannot be read is
"could not be read (a row of the wrong shape, or unparseable)" / "could NOT enumerate …: could not read the …
list: <reason>", exit 2/3, fail-closed. Oracles: Group 14 (a non-object cluster row, an `ssh.direct` block without
`host`, a 503 on the member listing via `MOCK_CLUSTER_PODS_STATUS`, `[…]`/`null`/`42`/`"s"` bodies into the
cluster sweep, an array body and a `null` pod row into the pod sweep — each named with its reason, zero
terminates/deletes).

**F-D (advisory, folded) — an id-less member row is refused, never dropped.** `rp_cluster_pods` refuses a member
row with no readable id (exit 2), so `_rp_cluster_member_ids` fails and `rp_sweep` suspends the whole pod sweep —
the exclusion set is complete or absent, never short (executed by the audit pre-fix: the id-less member's own pod
was terminated as an orphan). Oracle: Group 15.

**F-E (advisory, folded) — the P7 texts and the dead mock knobs.** The gate's own derivation matches the
`rp_cluster_create` literal in THREE tracked files (`check_gpu_prove_once.py`, `test_check_gpu_prove_once.py`,
`test_runpod_cluster_lib.sh`); the three texts (`ci/scripts/check_gpu_prove_once.py`, `docs/maintainer/dev-gpu.md`,
`docs/plans/67-distributed-training/UNITS.md`) say so. `MOCK_REQUEST_BODY_LOG` is now exported by Group 16, which
asserts the SENT create body carries the fixed 2×1 shape and only keys in the reviewed
`ci/scripts/fixtures/runpod_cluster_create_request_keys.json`; `MOCK_CLUSTER_PODS_STATUS` drives
`rp_cluster_pods`'s non-200 arm in Group 14.

`docs/maintainer/dev-gpu.md`'s `rp_cluster_sweep` sentence states the judgement lattice (deleted past its
deadline; named-and-left-alone when unjudgeable; `return 1` on a failed enumeration).

Verification run by the lead at this revision: `bash -n` + `shellcheck -S warning` (only `main`'s pre-existing
SC2034s), `test_runpod_cluster_lib.sh` 69 passed / 0 failed, `test_gpu_dev_lifecycle.sh`, `test_gpu_gang_lane.sh`,
`test_gpu_prove_lane.sh`, `check_gpu_prove_once.py` + its 207 unit tests, `check_ci_guard_wiring.py`,
`check_execution_surface_reachability.py`, `check_no_consumer_names.py`, `check_doc_parity.py`,
`perf/check_citations.py`. §2d's line citations into `runpod_lib.sh` were derived at the round-4 tree
(`b1697a1f`); this round moved lines in that file, so §2d's `at HEAD` tags are read as "at `b1697a1f`'s tree".
The unit's suites run again on the consolidated wave-3 branch before its single PR.

## 3. M2 — the two-host NCCL test body (`gang_nccl.rs`) — changed once, by `b480f2dc`, untouched by every fix round

**Restored in this revision.** This section's own heading existed in the c4 revision
(`docs/rigor/contracts/feat_500-C-U7b.md` at `31c8aa64`'s own tree, its own §3) and round 2
corrected its body in place (§0's own header correction note, above, names this the "§3's own
heading and body" site) — but the corrected body was left as unheaded trailing prose at the tail
of §2b instead of being given back its own `## 3.` heading, and round 2c's own insertion of
`## 2c` directly after that stranded text left nothing separating `2c` from `4`: the section
sequence jumped straight from `2c` to `4` at 23ef24a9's tree, with no `## 3.` anywhere in the file. Restored
here, with every citation re-derived directly against this tree — no sha below is cited from
anywhere but `git log --format=%h main..HEAD`.

`gang_nccl_two_hosts_reduce_a_known_vector`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:633-776` at `HEAD`, current tree — round 3
reworded its own preceding doc-comment to name U7b-A2b instead of the excised driver, re-derived
this revision) is feature-gated and reads its own env contract (`JAMMI_GANG_TWO_HOSTS_RANK`/
`_WORLD`/`_ID_FILE`, `JAMMI_REQUIRE_CUDA_TWO_HOSTS`) through `two_hosts_env_or_require`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:201` at `HEAD`, current tree), consulted
BEFORE `serial_cuda_device_or_require_two_hosts`'s own availability check
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:166` at `HEAD`, current tree) — both
registered against KO-7's ungated-skip scan in `ci/kernel-oracle-helpers.txt:91-92` (unchanged by
any round — a function-name index, not a line-number one, so it never drifts with either file),
landed by `1480cacb`. Rank 0 mints `Nccl::new_id()` and writes it atomically; rank 1 refuses any
id file whose size is not
exactly 128 bytes.

`b480f2dc` — landed by an ai-core agent in this same worktree, outside this contract's own owned
files, and itself BEFORE round 1's own fix commit (`31c8aa64`) — closes a sibling finding to
round 1's own F4: `hostname()` (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:475-507` at
`HEAD`, current tree — moved from round 3's own module-doc growth above this point, re-derived
this revision) now returns a named `Result<String, String>` instead of masking a failed read
behind `"unknown"`, and `missing_report_metadata_reason`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:509-544` at `HEAD`, current tree,
re-derived this revision) — a pure decision function,
factored out of `hostname`'s own `cuda`-gated shell-out so it stays hermetically testable without
a GPU — names which of hostname/iface is
missing (one or both, `"; "`-joined) BEFORE any NCCL work runs, so a bad metadata read panics
inside the `catch_unwind` closure's FIRST statement rather than letting an
indistinguishable-from-real `"unknown"` reach a `pass` report; the existing `Err(payload)` arm
then writes the fail report and `resume_unwind`s the same reason. `RankReport`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:425-461` at `HEAD`, current tree, re-derived
this revision) has no field that could hold the NCCL id — it travels only through
`$JAMMI_GANG_TWO_HOSTS_ID_FILE` — and is written on BOTH the pass and fail arm, never only on
success.

Six hermetic tests (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:781` at `HEAD`, current
tree — round 3's own module-doc growth shifts this whole trailing region down by a consistent 8
lines from `23ef24a9`'s tree, verified via three anchor points (`mod report_tests` itself,
its first test, and its last), re-derived this revision, `mod report_tests`) drive
`missing_report_metadata_reason` directly, without a GPU or the `cuda` feature: both good
(`:797`), an unset iface (`:807`), an empty iface (`:826`), a failed hostname read (`:843`), an
empty-but-`Ok` hostname (`:863`), and both bad (`:876`). Three further tests in the same module
pin `RankReport` itself: the JSON round trip with every documented field present under its
documented name (`:935`), a fail verdict carrying no digest and the reason (`:978`), and — behind
a non-vacuous negative control first proven to catch a genuine leak in each of three encodings —
that the id never appears in the report in any encoding (`:1007`).

Neither docs-ci fix round has touched this file: `git log --format=%h main..HEAD --
crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` names exactly `b480f2dc`, `1480cacb`,
`467dd9c9` — none of them `31c8aa64`/`c4f0c36e` (round 1), `ebe79a0d` (round 2), or `23ef24a9`
(round 2c); `git status` over it stays clean across every round.

**Oracle**: compiles under `cargo clippy -p jammi-ai --features live-gpu-tests --test
gpu_capability` (the gated-surface clippy step; not itself proof the test PASSES, only that it
compiles). The real proof is the executed run (§8) — not yet performed.

## 4. M3 — the cluster driver — EXCISED at round 3

**What it was, through round 2c** (history; every construct below is read against `23ef24a9`'s
tree, the tree that existed the moment before round 3 deleted it — `git show
23ef24a9:ci/scripts/runpod_gpu_cluster.sh`): never `runpod_gpu_gang.sh` — a fully separate driver,
workflow, and RunPod object type. Sequence, post round-2c: read per-data-center availability and
pass only qualifying `dataCenterIds` (A1) → `rp_init` (F1, §2) → create ONE 2x1 cluster → read
back the MEASURED shape and refuse (97) on a mismatch (P-C, §2b) → poll both members RUNNING with
a usable ssh path (`_rpc_check_readback`, §1) → build both members in parallel via one shared
per-rank heredoc (`_rpc_remote_script`) → start rank 0, poll the 128-byte id file, mark
`id_landed=1` (P-A, §2b), `scp` to a local staging copy, `scp` up to the member, THEN start rank 1
→ watch both ranks (inactivity/wrong-tree/budget) → copy both ranks' own logs into the artifact
dir (F6, §2) → pull both `rank-<r>.json` reports (a failed pull joins `rc`) → assemble ONE `gang`
artifact with the MEASURED `pod_count`/`gpu_count_per_pod` threaded in (P-C, §2b), refusing on an
unresolved host/iface or a repeated host compared case-insensitively (F4, §2 + advisory, §2b),
setting `assembly_ok=1` ONLY on a successful write (P-A2, §2c) → on EVERY exit arm, the EXIT trap
(`_rpc_cleanup_cluster`) records self-removal, deletes the cluster (joining a failed delete into
`rc`, F3(c)), runs the id-secrecy scan (requiring the assembled artifact only when
`assembly_ok=1`, P-A2) and DESTROYS a dirty or genuinely-unexaminable carrier directory
(`_rpc_scan_or_destroy`, P-A + P-A2/P-A3, §2c), deletes the staging id file unconditionally,
chains `rp_cleanup` (F2), and exits.

**Round 3 (closing adversarial audit, 2026-09-15): the pre-committed §9 stop rule fires — a THIRD
BLOCK on this mechanism.** Three findings, executed against `23ef24a9`'s tree:

- **F1 — the id-secrecy scan was reachable only from a skippable path.** The scan ran ONLY from
  the EXIT trap (`ci/scripts/runpod_gpu_cluster.sh, line 849` at `23ef24a9`'s tree), sequenced THIRD
  behind two REST calls (`_rpc_self_remove_status`/`rp_cluster_delete`) whose own `curl`
  (`ci/scripts/runpod_lib.sh:375`/`:378` at `23ef24a9`'s tree) carried no `--max-time`; an
  untrapped `SIGINT` (what the CI runner sends first on cancellation) skips `EXIT` entirely. The
  workflow's own `if: always()` upload step would then publish whatever the pulled carrier
  directory held, unscanned.
- **F2 — "WILL BE DESTROYED" was false under `RP_SESSION`.** `rp_cleanup`
  (`ci/scripts/runpod_lib.sh:596` at `23ef24a9`'s tree) deletes `$RP_WORK` only when
  `RP_WORK_IS_TEMP=1`, which `ci/scripts/runpod_lib.sh:294` (at `23ef24a9`'s tree) CLEARS under an
  exported `RP_SESSION` — executed: the id-bearing staging file survived a run under
  `RP_SESSION`. Every one of `test_gpu_cluster_lane.sh`'s own lane fixtures stubbed `rp_cleanup`
  (`:119`/`:244`/`:372` at `23ef24a9`'s tree), so this class was never exercised by the suite that
  was supposed to catch it.
- **F3 — the in-place fallback's globs missed `..`-prefixed names.** The scanner's own recovery
  path globbed `/*` and `/.[!.]*` (`ci/scripts/runpod_gpu_cluster.sh, line 720` at `23ef24a9`'s tree),
  which misses any name starting `..` that Python's `iterdir()` (the scanner's own walk) sees —
  executed: a planted `..leak` file carrying the id remained at the upload path after the scan
  claimed the directory was clean.

Advisories, also found this round: (A1) `rp_cluster_pods`'s member-count check used `-ge` rather
than an exact match; (A2) the scan's hex needles were not whitespace-stripped before matching;
(A3) the driver's `tee`'d run log was never explicitly flushed/waited before the process exited;
(A4) `_rp_cluster_payload`'s request shape (`podCount 2`, `gpuCountPerPod 1`) was hardcoded —
resolved in commit 1 of this docs-ci round: kept FIXED, stated in that commit's own body (no
caller needs any other shape once the driver that would have needed one is excised).

Citation round 3 additionally BLOCKed on this contract's own F5 block (six stale
`gang_id_secrecy_scan.py` line citations) and on §10's B6 commit count — both closed by this
revision (§13, §10 below).

**Disposition.** `ci/scripts/runpod_gpu_cluster.sh` and `.github/workflows/gpu-cluster.yml` are
DELETED at `55276624` (commit 1 of this docs-ci round), per the pre-committed stop rule
(`CONTRACT-U7b.md §9`): no fourth round on this mechanism. The driver is refiled as
**U7b-A2b** (`docs/plans/67-distributed-training/UNITS.md`), with this contract's §2/§2b/§2c as
its history, §9's P-A/P-B/P-C/P-D as its acceptance properties, and round 3's F1–F3/A1–A4 above as
its first pressure round. The one real cluster run (§8) moves with it.

## 5. M4 — the id-secrecy scan — EXCISED at round 3

**What it was, through round 2c** (history; every construct below is read against `23ef24a9`'s
tree — `git show 23ef24a9:ci/scripts/gang_id_secrecy_scan.py`): see §2 F5, §2b P-A/P-B/advisory,
and §2c P-A2/P-A3 in full. Carrier set: the pulled artifact directory (hang-proof, FIFO/
socket-refusing, and stack-safe rather than recursion-limited, P-B), the run log (genuinely
inside that directory, F6, ALWAYS required), the assembled `gang` artifact JSON (OPTIONAL as of
round 2c — required ONLY when the caller passes `--assembled-artifact` at all; §2c P-A2), the
staging copy's own directory listing. Exit lattice: `0` clean, `1` hit (named by carrier and
encoding, including a line-wrapped base64 encoding, §2b advisory), `2` UNEXAMINABLE (covering ANY
scanner-internal exception, not only its own documented `ScanTimeout`, §2b P-B).

**Round 3 disposition.** F1/F3 above are findings on how and when this scan was INVOKED by the
driver (M3), not on the scan's own internal correctness (its self-test, cited in prior rounds,
kept passing throughout); it is deleted alongside the driver anyway because a scan nothing calls
proves nothing. `ci/scripts/gang_id_secrecy_scan.py` is DELETED at `55276624`. It is refiled with
the driver as part of **U7b-A2b** — the mechanisms above (the cycle-safe walk, the stack-safe
budget, the total exit lattice, the byte-level needle set) are its starting design, not
discarded; F1–F3/A1–A4 are the pressure round that design must additionally survive when rebuilt.

## 6. M5 — P8 (schedule visibility) and the `RENTING_ROOTS` derivation, post round-2

Unchanged in its core property since c4 (`check_p8_schedule_visibility`,
`ci/scripts/check_gpu_prove_once.py:1354-1448` at 23ef24a9's tree, moved from round 1's `1341-1435` — round
2's own `_strip_trailing_comment` fix, below, inserted 13 lines ahead of this function without
touching its own body): a `schedule:` key on a paid-pod-lane workflow (or any workflow mentioning
a `RENTING_ROOTS`-derived driver while carrying the secret) is a FINDING unless the workflow is a
reviewed `PAID_LANE_CRON_ALLOWLIST` (`ci/scripts/check_gpu_prove_once.py:1317-1327` at 23ef24a9's tree, moved
from round 1's `1304-1315`) entry whose token resolves.

**Advisory fix (round 1, unchanged by round 2)**: the allow-list review covers exactly ONE
reviewed cadence per lane — a SECOND `- cron:` entry under the same `schedule:` key, which the
token-resolution check alone cannot see, is its own FINDING:

`_read_schedule_cron_entries` (`ci/scripts/check_gpu_prove_once.py:1331-1350` at 23ef24a9's tree, moved from
round 1's `1318-1338`) reads `on.schedule` as a real list through the SAME PyYAML-backed parse
`read_top_level_on_block` uses (`check_execution_surface_reachability.py`'s own loader, imported as
`exec_mod`), never a second, independently-drifting text scan; its own finding text, naming the
count, sits at `ci/scripts/check_gpu_prove_once.py:1438` (at 23ef24a9's tree, moved from round 1's `1425`).

**Advisory fix (round 1)**: `drop_comment_lines` (`ci/scripts/check_gpu_prove_once.py:491-500` at
HEAD, moved from round 1's `478-487`) strips a TRAILING `# ...` comment off an otherwise-code line
(`_strip_trailing_comment`, `ci/scripts/check_gpu_prove_once.py:456-488` at 23ef24a9's tree — round 2 grew
this from round 1's `456-476` by twelve lines, see below), tracking quoted spans so a `#` inside a
string literal is never mistaken for a comment start — a token that occurs only after a trailing
`#` is prose, never code evidence, and never resolves a P8 allow-list token (or any other "does
this text mention X" search this helper backs).

**Advisory fix (round 2)**: `_strip_trailing_comment` mistoggled its own quote-state on a
backslash-escaped apostrophe — bash's own `'\''` idiom for embedding a literal quote inside a
single-quoted string is three quote characters but only TWO real delimiters, and toggling on all
three left the parser believing it was still inside a string, so a genuine trailing comment
following one would never be stripped. Fixed by skipping the escaped character
(`ci/scripts/check_gpu_prove_once.py:477-479` at 23ef24a9's tree, inside the function's own `while` loop)
rather than toggling on it.

`RENTING_ROOTS` (`ci/scripts/check_gpu_prove_once.py:991` at 23ef24a9's tree, moved from round 1's `978`,
unchanged content) still carries both roots through round 3; `PAID_POD_LANE_TABLE`'s cluster row
(`ci/scripts/check_gpu_prove_once.py:892` at 23ef24a9's tree, moved from round 1's `879`, unchanged
through round 2) is REMOVED at round 3 (§4) — `rp_cluster_create` remains a `RENTING_ROOTS`
member with no table row; its only callers on this tree are `runpod_lib.sh`'s own mocks-only tests,
cleared through `_check_derived_driver_cannot_rent`'s existing "mentioned in no secret-carrying
workflow" predicate (no new exemption). The row returns when U7b-A2b ships a driver.

**Oracle**: `test_check_gpu_prove_once.py` (207 cases at 23ef24a9's tree, up from 206 at round 1's
own close, 201 pre-round-1; STILL 207 at `HEAD` after round 3 — no `def test_*` method was added
or removed, only fixture bodies inside existing ones; all passing at both trees,
`python3 -m unittest ci.scripts.test_check_gpu_prove_once`, exit 0). Round 1's own additions:
`RpSshoRequiresRpInitTest` (§2 F1), a second-cron-entry RED case plus a single-cron GREEN control
in `ScheduleVisibilityTest`, `DropCommentLinesTrailingCommentTest` (5 cases). Round 2 adds ONE case
to that same class, `test_a_backslash_escaped_apostrophe_inside_a_single_quoted_string_is_not_a_toggle`
(`ci/scripts/test_check_gpu_prove_once.py:661-672` at 23ef24a9's tree). The pre-existing
`DerivedRentingDriverTest`'s own hard-coded expected-derivation list
(`ci/scripts/test_check_gpu_prove_once.py:1309-1348` at 23ef24a9's tree, moved from round 1's
`1296-1335`) gained `ci/scripts/test_gpu_cluster_lane.sh` at round 1 — round 1's own F2/F3(c)
fixture text sourced `runpod_gpu_cluster.sh` in a real subshell and named
`rp_cluster_delete`/`rp_cleanup` in non-comment text, so the deliberately over-approximating
derivation scan (§0's own doctrine) derived it too; round 2's own new P-A/P-C/P-D fixture text
(§2b) sourced the same file the same way and added no NEW derived driver. **Round 3** deletes
`ci/scripts/test_gpu_cluster_lane.sh` (M3/M4's own oracle, §4/§5) — the real-tree derivation list
(`DerivedRentingDriverTest.test_the_real_tree_derives_the_set_this_suite_claims`,
`ci/scripts/test_check_gpu_prove_once.py`, current tree) drops both it and
`ci/scripts/runpod_gpu_cluster.sh` accordingly, and `test_runpod_cluster_lib.sh` — M1's own
mocks-only suite, unaffected by round 3 — stays derived (it calls `rp_cluster_create` directly)
and stays cleared the identical way every sibling test file already is, by `ci.yml`'s guard job
carrying no `RUNPOD_API_KEY`.

## 7. M6 — artifact registry leg discrimination — P-E2 fail-closed at round 3

See §2 F4 and §2b advisory (case-insensitive host compare) in full for the fix, through round 2c
(citations there `at 23ef24a9's tree`). `gang.leg` remains a required, closed-set field
(`"pod"`|`"cluster"`), checked first. Pod-leg rows unchanged. Cluster-leg rows
(`GANG_CLUSTER_FIELD_REGISTRY`) and `_gang_check_cluster_shape`'s own `pod_count`/
`gpu_count_per_pod` cross-field check stay on this tree, untouched by round 3.

**Round 3 (P-E2, §4's own disposition): the leg fails closed without a producer.**
`GANG_LEG_PRODUCER_PATH` (`ci/scripts/check_cuda_run_artifacts.py:1027-1029` at `HEAD`, current
tree) carries NO `"cluster"` entry (the `"pod"` entry is unchanged). `_gang_check_leg_producer_
binding` (`ci/scripts/check_cuda_run_artifacts.py:1505-1530` at `HEAD`) refuses any `gang.leg ==
"cluster"` artifact outright — a leg absent from that mapping is a NAMED refusal ("no registered
producer on this tree") BEFORE `producer.path` is ever read, never the prior `expected is None:
return []` silent pass. The row returns when U7b-A2b ships a driver (§4).

**Oracle**: `check_cuda_run_artifacts.py --self-test` — verified green on the current tree
(`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`, exit 0), all rule (k) cases
including the nine F4 arms named in §2/§2b (through round 2c) PLUS the new P-E2 self-test rows:
`gang_cluster_baseline()` — an otherwise-complete cluster-leg artifact — is now `expect_hit`
("no registered producer on this tree"), never `expect_clean`; a new `expect_hit_only` helper
isolates that one standing finding from each field-specific mutation's own finding (so the
shape/rank checks §2 F4 added are still independently exercised); the pod leg's own
`gang_baseline()` fixtures and their GREEN controls are byte-for-byte unchanged.

## 8. The executed attempt — status, moves with U7b-A2b at round 3

No real RunPod cluster run has happened, and none can while no driver exists on this tree.
`CONTRACT-U7b.md §2` scheduled one at the end of c3, authorized, bounded by §4's own committed
cost figures (`2 x $1.908/GPU/h = $3.816/h`; ≤ 1h billed, ≤ 2 runs) — its log would be the
evidence for the `args` entrypoint reaching bash on `RP_IMAGE` (§1), member sshd reachability,
`ens1` as the overlay iface, and whether member self-removal works. Every fix round through round
3 was entirely mocks-only (never called the RunPod API, no key available to any agent on this
unit) and changed nothing about this status. This run — and its standing authorization — MOVES
with the driver to **U7b-A2b** (§4); it is not this unit's to execute, since this unit ships no
driver to execute it with. A failed run, when U7b-A2b eventually executes one, is itself a
FINDING to be recorded and fixed (re-run once at most, within the 2-run authorization).

## 9. Known-unmeasured / uncovered (named, never claimed closed)

- **Member self-removal on a cluster pod.** Unknown until §8's run executes (now under U7b-A2b)
  and records `cluster-self-remove: ok|refused` (`_rpc_self_remove_status`/`_rpc_cleanup_cluster`,
  `ci/scripts/runpod_gpu_cluster.sh, lines 655-771` at 23ef24a9's tree — the last tree this construct
  existed on before round 3's excision, §4).
- **The REST v2 `args` field reaching `bash -c` on `RP_IMAGE`.** Never independently confirmed —
  §1. The launch-time read-back is the guard against this being false, not a proof it is true.
- **The REST v2 Bearer-auth transport path itself** (`_rp_rest`, `ci/scripts/runpod_lib.sh, lines
  369-386` at HEAD — round 4/M1's own advisory, §2d). Every `rp_cluster_*` primitive and
  `rp_cluster_sweep`/`rp_sweep`'s own cluster-member exclusion set go through this one function,
  and every test suite that reaches it on this tree (`test_runpod_cluster_lib.sh`,
  `test_gpu_dev_lifecycle.sh`, `test_gpu_gang_lane.sh`, `test_gpu_prove_lane.sh`) stubs `curl` on
  `PATH` — this function's own real HTTP request/response handling (status-code parsing, the
  `Authorization: Bearer` header, a genuine transport failure) has never executed against a live
  RunPod endpoint anywhere on this tree. Blocked on the same user pre-flight §1 names; distinct
  from the `args`-field bullet above (that is about whether a documented REQUEST FIELD reaches the
  container; this is about whether the TRANSPORT CALL that ships it has ever really run).
- **The `NCCL_SOCKET_IFNAME=ens1`/pin set at world >= 3.** This unit's own leg proves world 2
  only. `docs/maintainer/dev-gpu.md`'s own "Known-unmeasured" section (at `5ebe53ab`, unchanged)
  states this and points back here.
- **S5's cross-host byte identity.** `docs/plans/67-distributed-training/README.md`'s own **S5**
  entry (cited by construct — a live plan doc, no sha; round 2 corrects round 1's own citation of
  this as "README.md, at `9f69275b`", which pointed at the repository-root README rather than the
  plan doc the measurement actually lives in) states candle 0.11 LoRA-shaped forward/backward/SGD
  is byte-identical across processes and across two A100s with no env pins — this measured two
  PROCESSES, never a cross-HOST NCCL collective; this leg's `reduced_vector_digest` equality is a
  narrower, different measurement (a bit-exact sum, not a training step's digest pair) and does
  not itself extend S5's claim.

## 10. Invariants crossed

B2 (every script and doc here names no consumer — `python3 ci/scripts/check_no_consumer_names.py`
verified green this round); the paid-lane doctrine (P1/P7/P8: label/dispatch only, never
merge-path — P8 stays; the cluster driver's own P7 table row is gone with it, §6); B6 (the
ai-core test body and the docs-ci lane land together in this unit's own scope — M2's test body
(commit `467dd9c9`) and the docs-ci lane (M1/M5/M6, commits `b984e24c` through this revision's own)
are on the SAME branch's SAME PR; the exact commit count is `git rev-list --count main..HEAD` at
whatever commit reads this, never a number hand-maintained in this prose — it has already moved
twice since this contract was first written and will move again); K2 (every parsed API body is
validated before use — round 2 adds the cluster's own `compute` block, §2b P-C, now historical
per §4/§5's own disposition but still true of the primitives M1 ships).

## 11. Gate files a human must review at this unit's merge

`swarm.yml`'s human-amend-only glob (`SWARM_GATE_TOUCHED`) covers gate-script edits; the reviewer
checks, on the CURRENT tree (round 3 — the driver/scan review list below §4/§5 made moot is
dropped, not left stale):

- `ci/scripts/check_gpu_prove_once.py` — round 3 removes `PAID_POD_LANE_TABLE`'s cluster row
  (§4/§6); `RENTING_ROOTS` and P8's schedule-cron-count check are unchanged.
- `ci/scripts/check_cuda_run_artifacts.py` — round 3's P-E2 fix: `GANG_LEG_PRODUCER_PATH` drops
  its `"cluster"` entry and `_gang_check_leg_producer_binding` refuses any `gang.leg == "cluster"`
  artifact outright (§7); rule (k)'s pod-leg registry and the cluster leg's own field/shape checks
  (§2 F4, §2b advisory — historical, `at 23ef24a9's tree`) are untouched.
- `ci/scripts/runpod_lib.sh` — round 3 rewords ONE comment (the F13 shared-tripwire header, no
  longer naming the excised driver) as a line-count-neutral edit (2860 lines before and after);
  `rp_sweep`'s/`rp_cluster_sweep`'s own fixes (§2 F3) are otherwise untouched.
- `.github/workflows/ci.yml` — round 3 removes the two guard-matrix rows that ran the now-deleted
  `test_gpu_cluster_lane.sh` and `gang_id_secrecy_scan.py --self-test`; the pre-existing
  `runpod cluster primitives suite` row (`test_runpod_cluster_lib.sh`, M1) is untouched.
- `ci/scripts/execution_surface_reachability_allowlist.txt` — round 3 removes the cluster-leg
  tuple row (the driver that tuple belonged to is gone).
- `ci/scripts/test_check_gpu_prove_once.py` — round 3 renames the synthetic
  "driver-calling-the-cluster-root-directly" fixture path away from the deleted file's own name,
  drops `CLUSTER_YML_GOOD`, and corrects the real-tree derived-driver-set assertion.
- `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` — round 3 restates doc-comments and one
  runtime skip message to name U7b-A2b instead of the excised driver; no Rust logic changed.

## 12. Residuals recorded UNCOVERED

- **The cluster driver, its workflow, and the id-secrecy scan** (M3/M4, §4/§5) — EXCISED at round
  3 and refiled as **U7b-A2b**, not built by this unit or this contract.
- The un-executed REST v2 pre-flight (§1) and the un-executed cluster run (§8) — both named, not
  silently deferred; both move with U7b-A2b.
- Member self-removal on a cluster object (§9) — moves with U7b-A2b.
- The NCCL pin set at world >= 3 (§9).
- S5's cross-host byte identity (§9).
- **U7b-A3** (the `gpu-gang.yml` 6-hourly cron re-add) is explicitly NOT part of this unit or
  U7b-A2b.
- Every residual PR-B1's own contract already recorded UNCOVERED for P6/P7 is unchanged by this
  revision.

## 13. Citations verified against which head

**Through round 2c** (history, unchanged by this note): every construct §2/§2b/§2c cited was read
directly against this worktree's tree via `grep -n`/direct file reads, AFTER round 2c's own fix
commit (`23ef24a9`) landed — the last step before that revision was written — never against a
scratchpad working document's own line numbers, and never against `CONTRACT-U7b.md`'s own §7/§8
fold text beyond citing its decisions by name. Round 2c re-derived, against `23ef24a9`'s own tree,
every prior citation into `ci/scripts/runpod_gpu_cluster.sh`, `ci/scripts/gang_id_secrecy_scan.py`,
and `ci/scripts/test_gpu_cluster_lane.sh` — each updated to its new line number with a "moved from
round 2's `N`" note, or, where the underlying construct round 2's own prose narrates no longer
exists under that name (§2b's own P-A subsection, which narrates round 2's fix as it stood before
round 2c), pinned explicitly `(at c4f0c36e's own tree)`.

**Round 3 (this revision).** `HEAD` is now `55276624` (round 3's own excision commit) and moves to
this revision's own commit once it lands; the three files named above no longer exist at `HEAD` at
all. Every citation in §2/§2b/§2c that an earlier round tagged `(at HEAD)` is corrected in THIS
revision to `(at 23ef24a9's tree)` — literally: `git show 23ef24a9:<path>` (never `git show
HEAD:<path>`, which for those three files errors "does not exist"). §4 and §5 are REWRITTEN in
full, not re-tagged, to record the excision itself against `55276624`'s own tree. Every citation
into a file round 3 touched but did NOT delete (`ci/scripts/check_gpu_prove_once.py`,
`ci/scripts/check_cuda_run_artifacts.py`, `ci/scripts/runpod_lib.sh`,
`ci/scripts/test_check_gpu_prove_once.py`, `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`) is
RE-DERIVED against `HEAD` (the current tree) in §6/§7/§10/§11 rather than left `at 23ef24a9's tree`
— round 3's own edits to `ci/scripts/runpod_lib.sh` were made line-count-neutral (2860 lines
before and after, confirmed via `wc -l`) so every OTHER citation into that file, unmentioned by
this revision, still resolves at its original line number on the current tree without a
re-derivation pass.

Every citation is the bare `path:line` (or `path:line-line`) form `check_rigor_record.py`'s own
`check_path_line_citations` matches, tagged `(at <sha>)` or `(at HEAD)` per this file's own header
convention (§0); a citation this round corrected (never merely re-derived) says so explicitly at
its own site. Self-check command for every citation in this revision: `git show <sha>:<path> | sed
-n '<line>,<line>p'` for a `(at <sha>)` tag, or `sed -n '<line>,<line>p' <path>` against the
current worktree for an `(at HEAD)` tag (`git log --reverse --format=%h main..HEAD` lists every sha
above, oldest first).

**Self-check, as round 3 wrote it (historical text, corrected below — never re-executed as
written)**: round 3's own revision of this paragraph claimed the grep below ALSO matched
`docs/plans/67-distributed-training/UNITS.md`'s own U7b-A2b filing (and its README.md
cost-ceiling cross-reference). Round 4/M1's own citation-round review (§2d above) re-ran the
identical command and found that claim was FALSE — not merely stale, but never true even AT round
3's own commit (`git show 9d356dff:docs/plans/67-distributed-training/UNITS.md | grep` and the
same against `README.md` at that commit both return nothing): the U7b-A2b filing describes the
excised driver conceptually and never repeats any of the four deleted paths as a bare string. This
is the exact citation-form defect this contract's own header paragraph ("This revision supersedes
the original c4 contract," `docs/rigor/contracts/feat_500-C-U7b.md:27-34`) already named once for
the c4 revision's own citations — a self-check that was never actually executed against the text
it describes, passing vacuously.

```
$ grep -rn 'runpod_gpu_cluster\|gpu-cluster.yml\|gang_id_secrecy_scan\|test_gpu_cluster_lane' ci .github crates docs
```

**Re-executed for THIS revision** (§2d's own citation-round self-check, above, is the same
command; repeated here per this section's own convention): matches only this contract's own
history sections (§2/§2b/§2c/§4/§5/§9/§11/§13/§2d, all pinned `(at 23ef24a9's tree)` or narrating
what round 3 deleted) — 104 total line matches, one file (§2d's own count). Never a live `ci/`,
`.github/`, or `crates/` path, never `docs/plans/67-distributed-training/UNITS.md` or `README.md`,
and never an unqualified `(at HEAD)` into any of the four deleted files.
