# CONTRACT — feat/500-C-U7b: the RunPod cluster leg (2 hosts x 1 A100) proves the two-host NCCL bootstrap; the reap treats a cluster as its own object type

**Contract of record.** slug: `feat_500-C-U7b` — the committed mechanism contract
`ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` before this unit's
rigor record at `docs/rigor/feat_500-C-U7b.jsonl` (the lead's own export, landed separately)
satisfies that checker's disclosure requirement. This unit is **U7b-A2** in the decomposition
`docs/plans/67-distributed-training/UNITS.md § U7b` states: **A1-pull** (the pod-tier smoke's CI
scaffolding — `gpu-gang.yml`, `runpod_gpu_gang.sh`, P7) merged already, as PR-B1 (contract
`docs/rigor/contracts/feat_500-PR-B1.md`); **A2** is this unit — the cluster leg, its own reap
arm, and P8; **A3** — a 6-hourly cron re-add on `gpu-gang.yml` — is a FUTURE, separately
authorized unit this contract does not build.

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
new surfaces; fix round 2c, this revision, a small lead-initiated fix on round 2's own new surface
found by the lead BEFORE the closing adversarial audit re-ran — never itself a third audit BLOCK,
§2c). Commit history on this branch, oldest first, below the `main` merge (`git log --format='%h
%s' main..HEAD`, re-derived AFTER round 2c's own fix commit landed):

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
```

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
(`ci/scripts/runpod_gpu_cluster.sh:283-316` at HEAD, moved from round 2's `276-309` by round 2c's
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

`ci/scripts/runpod_gpu_cluster.sh:839` precedes `:842` (at HEAD; round 2c's own P-A2 module header
prose and `assembly_ok=0` declaration moved both lines down from round 2's `795`/`798` without
changing their relative order, §2c below).

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh:881-896` (at HEAD, moved from round 2's `806-821`
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

`ci/scripts/runpod_gpu_cluster.sh:738-771` (at HEAD, moved from round 2's `701-734` by round 2c's
own P-A2/P-A3 growth above this point, §2c below — round 2 itself grew this body from round 1's
`591-608` by inserting the P-A scan-join and the unconditional staging-file delete — see §2b P-A)
is `_rpc_cleanup_cluster`'s own body, moved OUT of the sourced-execution guard (it was previously
defined only when the file is EXECUTED, making it untestable by sourcing) into the pure-helpers
section above it — only the `trap _rpc_cleanup_cluster EXIT` registration itself
(`ci/scripts/runpod_gpu_cluster.sh:849` at HEAD, moved from round 2's `805`) remains inside the
guard.

**Oracle**: `test_gpu_cluster_lane.sh`'s F2/F3(c) block (`ci/scripts/test_gpu_cluster_lane.sh:104-191`
at HEAD, unchanged in shape or line range by either round 2 or round 2c) drives the REAL
`_rpc_cleanup_cluster` in a real subprocess (`bash -c '... source "$CLUSTER_SH" ...'`, so its own
`exit "$rc"` terminates that subprocess exactly the way a real EXIT trap fires), mocking only
`_rp_rest`/`rp_cluster_delete`/`rp_cleanup`: self-remove-ok, self-remove-refused-delete-ok, and
no-cluster-id-at-all all confirm `rp_cleanup` is chained (a marker file it writes exists
afterward) in every arm. Round 2 added a SECOND block (`ci/scripts/test_gpu_cluster_lane.sh:194-382`
at HEAD, widened from round 2's own `194-319` by round 2c's own three additional arms — see §2c
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

`ci/scripts/runpod_lib.sh:2660-2669` (at HEAD) is the full arm — `return 1` follows the `echo` on
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

`ci/scripts/runpod_gpu_cluster.sh:738-771` (at HEAD, the same span F2 cites) is the full trap
body: `rc` is captured from `$?` FIRST (the pending exit status), joined to `1` only when it was
still `0`, and the function's own `exit "$rc"` at the end — never a bare `return` — is what makes
the join visible to the process's real exit status (a trap's own `return` would not override an
already-pending exit code). `gpu-dev.sh reap` (`ci/scripts/gpu-dev.sh:265-281` at `b984e24c`,
unchanged) already joins `rp_sweep`'s and `rp_cluster_sweep`'s own rcs, so both (a) and (b) above
already propagate to `reap`'s own exit without a further change there.

**Oracle**: `test_runpod_cluster_lib.sh` Group 3 gains an `UNAGEABLE`-cluster fixture
(`ci/scripts/test_runpod_cluster_lib.sh:378-401` at HEAD: `rc=1`, names `cl-unageable`, deletes
nothing); Group 5's existing refused-terminate fixture is updated to assert `rc=1` and the named
summary line (`ci/scripts/test_runpod_cluster_lib.sh:499-512` at HEAD) rather than the pre-fix
`rc=0` it asserted before this round.

### F4 — the cluster registry never established two real hosts

At the c4 revision, `_gang_check_cluster_ranks` required only that `host`/`device`/`iface` be
non-empty strings — never that `host` be DISTINCT across ranks, never that `iface` (or `host`)
not be the driver's own `unknown` placeholder, never that `hosts == pod_count` or `world ==
pod_count * gpu_count_per_pod`, and the reduced-vector digest was a single TOP-LEVEL field the
driver had already collapsed at assembly time — an artifact with two ranks on the SAME host, or
an unresolved host/iface, or a digest disagreement hidden by the collapse, would pass.

**Fix, on the checker side** (`ci/scripts/check_cuda_run_artifacts.py`, at HEAD):

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

**Fix, on the driver side** (`ci/scripts/runpod_gpu_cluster.sh`, at HEAD):
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
`reduced_vector_digest` (`ci/scripts/runpod_gpu_cluster.sh:557-567` at HEAD, inside the `ranks`
list construction), and the assembled artifact's `producer` block is now bound to this driver's
own path (`ci/scripts/runpod_gpu_cluster.sh:592-598` at HEAD: `"path":
"ci/scripts/runpod_gpu_cluster.sh"`, `"kind": "script"`), matching `GANG_LEG_PRODUCER_PATH` above.
(A complementary fix on the Rust side, `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` at
`b480f2dc` — round 2 corrects round 1's own citation of this commit as `d12e1689`, a sha that does
not exist on this branch; see this file's own header correction note — outside this contract's own
owned files, landed by an ai-core agent in the same worktree during round 1's own fix window —
makes `hostname()` return a named `Result` instead of masking a failed read behind `"unknown"`, so
the driver-side refusal above is checking a value that itself can no longer silently BE `"unknown"`
on a healthy run.)

**Oracle**: `check_cuda_run_artifacts.py --self-test`'s rule (k) self-test gains, named
(`ci/scripts/check_cuda_run_artifacts.py:3349-3448` at HEAD; round 2 grew this from round 1's
`3343-3429` by adding a case-insensitive repeated-host arm, §2b advisory): two ranks on one host
FAILS; two ranks on the same host differing only in CASE FAILS (round 2); an `unknown` host FAILS;
an `UNKNOWN` (any case) iface FAILS; a per-rank digest mismatch on `pass` FAILS; a missing per-rank
digest on `pass` FAILS; `hosts != pod_count` FAILS; `world != pod_count * gpu_count_per_pod`
FAILS; a cluster-leg artifact carrying the pod leg's own `producer.path` FAILS (and vice versa).
`gang_baseline()`/`gang_cluster_baseline()` (`ci/scripts/check_cuda_run_artifacts.py:3154-3232` at
HEAD, moved from round 1's `3148-3226`) now stamp each leg's real producer path
(`ci/scripts/runpod_gpu_gang.sh` / `ci/scripts/runpod_gpu_cluster.sh`), and the self-test's own
fixture repo gains tracked stand-ins for both paths
(`ci/scripts/check_cuda_run_artifacts.py:2751-2757` at HEAD, moved from round 1's `2745-2751`) so
rule (b)'s own producer.path-exists-and-is-tracked check has something real to bind against.
**Round 2 adds a SECOND, independent oracle** for this same F4 property: rather than only driving
the CHECKER against hand-built fixture JSON, `ci/scripts/test_gpu_cluster_lane.sh:733-877` (at
HEAD, moved from round 2's `658-802` by round 2c's own P-A2 fixture growth above this point, §2c
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

**Fix** (`ci/scripts/gang_id_secrecy_scan.py`, at HEAD):

```
$ grep -n 'def scan_dir\|def scan_file\|def run_scan\|def _run_scan_body\|def wall_clock_budget\|class ScanTimeout' ci/scripts/gang_id_secrecy_scan.py
173:def scan_file(path: Path, needles: list[tuple[str, bytes]]) -> tuple[int, str]:
211:def scan_dir(root: Path, needles: list[tuple[str, bytes]]) -> list[tuple[int, str]]:
277:class ScanTimeout(Exception):
282:def wall_clock_budget(seconds: int):
325:def run_scan(
349:def _run_scan_body(
```

`scan_dir` (`:211-274`; round 1 fixed the cycle-hang with a RECURSIVE `walk()` closure — line
numbers `190-234` at round 1's own `31c8aa64` — round 2's own closing audit BLOCKed on that
closure still being recursive, itself a SECOND way the scan could fail, `RecursionError`, escaping
uncaught as a bare exit 1/traceback; round 2 replaces it with an EXPLICIT STACK (a plain Python
`list`), moved round 1's own citation from `190-234` to this round's `211-274` — see §2b P-B for
the full fix) tracks the REAL path of every directory it enters in a `visited_dirs` set; a
directory whose real path repeats is one `"cyclic carrier"` UNEXAMINABLE finding, never a
re-descent. `scan_file` (`:173-208`, moved from round 1's `66-96`) checks `stat.S_ISREG` explicitly
before ever calling `read_bytes()` — any other mode class (FIFO, socket, device) is refused by
name (UNEXAMINABLE), never opened. `wall_clock_budget` (`:282-299`, moved from round 1's
`254-271`, a `contextlib.contextmanager` over `signal.alarm`) wraps the WHOLE scan body
(`run_scan`/`_run_scan_body`, `:325-345`/`:349-421`, moved from round 1's `179-183`) — round 2
also widens `run_scan`'s own `except` from `ScanTimeout` alone to `except Exception`, so ANY
scanner-internal failure (a `RecursionError` included, though the explicit stack above no longer
produces one; any other bug) is UNEXAMINABLE, never a traceback (§2b P-B). Default budget 120s
(`DEFAULT_BUDGET_SECS`/`GANG_ID_SCAN_BUDGET_SECS` env override,
`ci/scripts/gang_id_secrecy_scan.py:109` at HEAD, moved from round 2's `91` by round 2c's own
docstring growth above this point, §2c below), overridable via `--budget-secs`.

**Advisory, also fixed**: `id_needles` (`ci/scripts/gang_id_secrecy_scan.py:118-142` at `31c8aa64`,
unchanged by round 2) now computes FOUR base64 variants — standard padded, standard un-padded,
URL-safe padded, URL-safe un-padded — rather than one. Round 2 adds a SECOND base64 advisory: a
line-wrapped base64 encoding (coreutils `base64`'s 76-column default, `openssl base64`'s 64-column
default) is matched too, via a whitespace-stripped copy of the scanned bytes computed lazily for
base64-labeled needles only (`_strip_whitespace`/`_scan_bytes`,
`ci/scripts/gang_id_secrecy_scan.py:163-185` at HEAD, moved from round 2's `145-167` by round 2c,
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
`.github/workflows/gpu-cluster.yml:152-156` at `5ebe53ab`, unchanged by this round), so the
uploaded artifact never actually carried the run log, and the id-secrecy scan's own artifact-dir
walk never actually covered it as a carrier via that path either (it was scanned only through the
SEPARATE, explicit `--log` argument).

**Fix** (`ci/scripts/runpod_gpu_cluster.sh`, at HEAD):

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

placed immediately after both `wait` calls (`ci/scripts/runpod_gpu_cluster.sh:1014-1023` at HEAD,
moved from round 2's `970-979` by round 2c's own module-header/`assembly_ok` growth above this
point, §2c below), before any pass/fail branching. The workflow's own upload step needed no
change — `path: .gpu-pull/gpu-cluster/` already covers the directory the run log now lives
inside.

**Oracle**: `test_gpu_cluster_lane.sh`'s F6(a) block
(`ci/scripts/test_gpu_cluster_lane.sh:667-685` at HEAD, moved from round 2's `592-610` by round
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

`_rpc_cleanup_cluster` (`ci/scripts/runpod_gpu_cluster.sh:701-734` at `c4f0c36e`'s own tree — this
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

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh:194-319` (at `c4f0c36e`'s own tree — see the
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
name: `scan_dir` (`ci/scripts/gang_id_secrecy_scan.py:229-292` at HEAD, moved from round 2's
`211-274` by round 2c's own docstring growth above this point, §2c below) walks via an EXPLICIT
STACK (a plain Python `list`), never a recursive closure — the round-1 shape that made a
`RecursionError` possible in the first place. `run_scan` (`:343-363`, moved from round 2's
`325-345`; round 2c also widens its own `assembled_artifact` parameter type to `Path | None`,
§2c below) wraps `_run_scan_body` in `except Exception`, not only `except ScanTimeout` — ANY
scanner-internal failure is UNEXAMINABLE (2), never exit 1, never an uncaught traceback.

**Oracle**: `gang_id_secrecy_scan.py --self-test` gains `test_deep_tree_well_beyond_the_recursion_limit_does_not_crash`
(`ci/scripts/gang_id_secrecy_scan.py:762-785` at HEAD, moved from round 2's `691-714` by round 2c's
own docstring/self-test growth above this point, §2c below — a 200-level-deep tree walked under an
artificially lowered `sys.setrecursionlimit(40)`, portable across OS `PATH_MAX` differences a
literal 1,500-level fixture would hit inconsistently on different filesystems; the planted id at
the bottom is still found, proving the walk reaches full depth) and
`test_scan_dir_raising_an_unexpected_exception_is_unexaminable_not_a_traceback`
(`ci/scripts/gang_id_secrecy_scan.py:788-803` at HEAD, moved from round 2's `717-732` by round 2c,
§2c below — `scan_dir` mocked to raise a bare `RuntimeError`, asserted UNEXAMINABLE, never
propagated).

### P-C — the artifact's shape is measured, never four literals

**Fix** (`ci/scripts/runpod_gpu_cluster.sh`, at HEAD): a new pure helper parses the cluster's own
`compute` block from the SAME `Cluster` object `rp_cluster_get` already returns (unmodified,
`ci/scripts/runpod_lib.sh:1537-1557`, untouched by either round):

```
$ grep -n '^_rpc_parse_cluster_shape()\|cluster_body="\$(rp_cluster_get' ci/scripts/runpod_gpu_cluster.sh
342:_rpc_parse_cluster_shape() {
813:cluster_body="$(rp_cluster_get "$cluster_id")" || { echo "::error::could not read back the created cluster's own shape"; exit 97; }
```

`_rpc_parse_cluster_shape` (`:342-362` at HEAD) prints `"podCount gpuCountPerPod"` on a successful
parse of `compute.podCount`/`compute.gpuCountPerPod` (both required positive integers), or exits 2
on anything else. The executed block (`:807-821` at HEAD) calls it right after cluster create,
BEFORE any member work starts, and REFUSES (exit 97, "wrong shape" — the exit code the module
doc's own EXIT CONTRACT already named for this case, never actually checked against a measurement
until now) when the measured shape disagrees with `RP_CLUSTER_POD_COUNT`/
`RP_CLUSTER_GPU_COUNT_PER_POD`. `MEASURED_POD_COUNT`/`MEASURED_GPU_COUNT_PER_POD` are then threaded
into `_rpc_assemble_gang_artifact`'s own argv (`:1010-1013` at HEAD) — the four literals
(`"world": 2`, `"hosts": 2`, `"pod_count": 2`, `"gpu_count_per_pod": 1`) are GONE from the
assembler's own python (`ci/scripts/runpod_gpu_cluster.sh:508-620` at HEAD, moved from round 2's
`501-613` by round 2c's own module-header growth above this function, §2c below — is the whole
function, signature grown from `$1..$5` to `$1..$7`); `gang.world`/`gang.hosts` are now computed
as `pod_count * gpu_count_per_pod`/`pod_count` from whatever the driver threads in.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh:733-877` (at HEAD, moved from round 2's `658-802`
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

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh:733-877` (at HEAD, moved from round 2's `658-802`
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
  assembler's own `_norm_host` (`ci/scripts/runpod_gpu_cluster.sh:534-538` at HEAD, moved from
  round 2's `527-531` by round 2c, §2c below) and the checker's own `_gang_check_cluster_ranks`
  (`ci/scripts/check_cuda_run_artifacts.py:1452-1455` at HEAD, plus a self-test arm at
  `ci/scripts/check_cuda_run_artifacts.py:3361-3373`) both compare `.strip().casefold()` rather
  than raw string equality — "Host-A" and "host-a" are the same host on both sides now, never a
  false "two hosts" on one side and a false negative on the other.
- **Base64 needle matching survives line-wrapping.** See §5's revision above
  (`ci/scripts/gang_id_secrecy_scan.py:163-185`, moved from round 2's `145-167` by round 2c, §2c
  below, `_strip_whitespace`/`_scan_bytes`) — a whitespace-stripped copy of the scanned bytes is
  checked for base64-labeled needles, lazily, only when the raw check misses.
- **`_strip_trailing_comment` handles a backslash-escaped apostrophe.** See §6's revision above
  (`ci/scripts/check_gpu_prove_once.py:477-479`).
- **The no-op `grep -v '^\s*#'` at `test_gpu_cluster_lane.sh` (round 1's own `:547`) is replaced by
  an assertion that actually filters.** `grep -n`'s own output is `"N:text"` — a line ALWAYS
  starts with digits, so a filter anchored at `^\s*#` against THAT text can never match anything (a
  no-op that happened to be harmless only because the real file had nothing to filter). Fixed
  (`ci/scripts/test_gpu_cluster_lane.sh:891` at HEAD, moved from round 2's `816` by round 2c's own
  P-A2 fixture growth above this point, §2c below: `grep -vE '^[0-9]+:[[:space:]]*#'`, stripping
  `grep -n`'s own prefix before testing for a leading `#`) and PROVEN to actually filter against a
  synthetic two-line fixture (`ci/scripts/test_gpu_cluster_lane.sh:899-909` at HEAD, moved from
  round 2's `824-834` by round 2c: a comment-only line naming the target string on line 1, the
  real call on line 2 — the filtered result must land on line 2, not line 1).
- **The staging id file is deleted on EVERY exit after the scan, regardless of
  `RP_SESSION`/`RP_WORK_IS_TEMP`.** Folded into P-A above (`ci/scripts/runpod_gpu_cluster.sh:767`
  at HEAD, moved from round 2's `730` by round 2c, §2c below).
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
becomes OPTIONAL (`main`, `ci/scripts/gang_id_secrecy_scan.py:454-467,475-489` at HEAD: removed
from the required-argument check; `run_scan`/`_run_scan_body`
(`ci/scripts/gang_id_secrecy_scan.py:343,367` at HEAD, their own signatures) widen the parameter's
type to `Path | None`, and the required-carrier loop
(`ci/scripts/gang_id_secrecy_scan.py:403-421` at HEAD) is split into `log` — ALWAYS required —
and `assembled_artifact` —
required ONLY when not `None`). The driver threads a NEW global, `assembly_ok` (declared `0`
alongside `id_landed=0`, `ci/scripts/runpod_gpu_cluster.sh:800-806` at HEAD), set to `1` ONLY
immediately after `_rpc_assemble_gang_artifact` itself returns `0`
(`ci/scripts/runpod_gpu_cluster.sh:1055-1068` at HEAD — the call site, previously a bare `||` refusal, is now an
`if`/`else` so the success arm can set the flag; a `1`/`2` refusal, which never writes the file,
leaves `assembly_ok=0`). `_rpc_run_id_secrecy_scan`'s own 4th argument becomes OPTIONAL
(`ci/scripts/runpod_gpu_cluster.sh:638-644` at HEAD: an empty string omits `--assembled-artifact`
from the invocation entirely, never passing an empty path), and `_rpc_scan_or_destroy` (renamed,
below) passes `"$ASSEMBLED"` ONLY when `assembly_ok=1` (`ci/scripts/runpod_gpu_cluster.sh:704-725`
at HEAD, specifically `:707`). Because `$ASSEMBLED` always lives INSIDE `$CLUSTER_ARTIFACT_DIR` in
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
call (`ci/scripts/runpod_gpu_cluster.sh:769` at HEAD) unconditionally `rm -rf`s `$RP_WORK` before
the process exits whenever `RP_WORK_IS_TEMP=1` — the default this driver's own workflow runs
under (`.github/workflows/gpu-cluster.yml` sets neither `RP_SESSION` nor `RP_WORK`, so
`runpod_lib.sh`'s own `rp_init` takes the `mktemp -d`/`RP_WORK_IS_TEMP=1` branch on every CI run —
`ci/scripts/runpod_lib.sh:315-317`, unchanged by this round). A CI runner torn down at process
exit is not somewhere a human can later inspect anything, so "quarantine" was never an accurate
word for what this trap does; every site is renamed: the function itself
(`_rpc_scan_or_quarantine` → `_rpc_scan_or_destroy`, `ci/scripts/runpod_gpu_cluster.sh:704` at
HEAD), its own header (`:665-703` at HEAD), the `::error::` line
(`ci/scripts/runpod_gpu_cluster.sh:718` at HEAD — the relocation is stated to exist ONLY so
the upload step, which starts concurrently and could otherwise race an in-place deletion, can
never see a half-removed directory), the module-level header's own carrier-set paragraph
(`ci/scripts/runpod_gpu_cluster.sh:65-77` at HEAD), the EXIT CONTRACT paragraph
(`ci/scripts/runpod_gpu_cluster.sh:114-118` at HEAD), and this contract (§2b's own P-A paragraph,
corrected above; §4, §11).

**Oracle.** `gang_id_secrecy_scan.py --self-test`: **29 tests, up from 27** at round 2's own close.
Self-check, both counts, reproducible from any checkout of this branch: `git show
c4f0c36e:ci/scripts/gang_id_secrecy_scan.py` written to a scratch copy and run with
`--self-test` prints `Ran 27 tests ... OK` (round 2's own tree); `python3
ci/scripts/gang_id_secrecy_scan.py --self-test` at this round's own HEAD prints `Ran 29 tests ...
OK`, exit 0 — the two new cases are `test_assembled_artifact_omitted_entirely_is_not_required_
and_stays_clean` (`ci/scripts/gang_id_secrecy_scan.py:632-643` at HEAD) and
`test_assembled_artifact_omitted_but_a_leak_at_that_path_is_still_a_hit`
(`ci/scripts/gang_id_secrecy_scan.py:645-660` at HEAD). `test_gpu_cluster_lane.sh`'s P-A/P-A2
block (`ci/scripts/test_gpu_cluster_lane.sh:194-382` at HEAD) is REWRITTEN, not merely extended:
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

## 3. M2 — the two-host NCCL test body (`gang_nccl.rs`) — changed once, by `b480f2dc`, untouched by every fix round

**Restored in this revision.** This section's own heading existed in the c4 revision
(`docs/rigor/contracts/feat_500-C-U7b.md` at `31c8aa64`'s own tree, its own §3) and round 2
corrected its body in place (§0's own header correction note, above, names this the "§3's own
heading and body" site) — but the corrected body was left as unheaded trailing prose at the tail
of §2b instead of being given back its own `## 3.` heading, and round 2c's own insertion of
`## 2c` directly after that stranded text left nothing separating `2c` from `4`: the section
sequence jumped straight from `2c` to `4` at HEAD, with no `## 3.` anywhere in the file. Restored
here, with every citation re-derived directly against this tree — no sha below is cited from
anywhere but `git log --format=%h main..HEAD`.

`gang_nccl_two_hosts_reduce_a_known_vector`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:637-768` at HEAD) is feature-gated and reads
its own env contract (`JAMMI_GANG_TWO_HOSTS_RANK`/`_WORLD`/`_ID_FILE`,
`JAMMI_REQUIRE_CUDA_TWO_HOSTS`) through `two_hosts_env_or_require`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:194` at HEAD), consulted BEFORE
`serial_cuda_device_or_require_two_hosts`'s own availability check
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:159` at HEAD) — both registered against KO-7's
ungated-skip scan in `ci/kernel-oracle-helpers.txt:91-92` (at HEAD), landed by `1480cacb`. Rank 0
mints `Nccl::new_id()` and writes it atomically; rank 1 refuses any id file whose size is not
exactly 128 bytes.

`b480f2dc` — landed by an ai-core agent in this same worktree, outside this contract's own owned
files, and itself BEFORE round 1's own fix commit (`31c8aa64`) — closes a sibling finding to
round 1's own F4: `hostname()` (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:483-500` at
HEAD) now returns a named `Result<String, String>` instead of masking a failed read behind
`"unknown"`, and `missing_report_metadata_reason`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:518-538` at HEAD) — a pure decision function,
factored out of `hostname`'s own `cuda`-gated shell-out so it stays hermetically testable without
a GPU — names which of hostname/iface is
missing (one or both, `"; "`-joined) BEFORE any NCCL work runs, so a bad metadata read panics
inside the `catch_unwind` closure's FIRST statement rather than letting an
indistinguishable-from-real `"unknown"` reach a `pass` report; the existing `Err(payload)` arm
then writes the fail report and `resume_unwind`s the same reason. `RankReport`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:431-454` at HEAD) has no field that could hold
the NCCL id — it travels only through `$JAMMI_GANG_TWO_HOSTS_ID_FILE` — and is written on BOTH the
pass and fail arm, never only on success.

Six hermetic tests (`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs:788-876` at HEAD, `mod
report_tests`) drive `missing_report_metadata_reason` directly, without a GPU or the `cuda`
feature: both good (`:789`), an unset iface (`:799`), an empty iface (`:818`), a failed hostname
read (`:835`), an empty-but-`Ok` hostname (`:855`), and both bad (`:868`). Three further tests in
the same module pin `RankReport` itself: the JSON round trip with every documented field present
under its documented name (`:927`), a fail verdict carrying no digest and the reason (`:970`), and
— behind a non-vacuous negative control first proven to catch a genuine leak in each of three
encodings — that the id never appears in the report in any encoding (`:999`).

Neither docs-ci fix round has touched this file: `git log --format=%h main..HEAD --
crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` names exactly `b480f2dc`, `1480cacb`,
`467dd9c9` — none of them `31c8aa64`/`c4f0c36e` (round 1), `ebe79a0d` (round 2), or `23ef24a9`
(round 2c); `git status` over it stays clean across every round.

**Oracle**: compiles under `cargo clippy -p jammi-ai --features live-gpu-tests --test
gpu_capability` (the gated-surface clippy step; not itself proof the test PASSES, only that it
compiles). The real proof is the executed run (§8) — not yet performed.

## 4. M3 — the cluster driver, post round-2c — sequence and exit contract

Never `runpod_gpu_gang.sh` — a fully separate driver, workflow, and RunPod object type. Sequence,
post round-2c: read per-data-center availability and pass only qualifying `dataCenterIds` (A1) →
`rp_init` (F1, §2) → create ONE 2x1 cluster → **read back the MEASURED shape and refuse (97) on a
mismatch (P-C, §2b)** → poll both members RUNNING with a usable ssh path (`_rpc_check_readback`,
§1) → build both members in parallel via one shared per-rank heredoc (`_rpc_remote_script`) →
start rank 0, poll the 128-byte id file, **mark `id_landed=1`** (P-A, §2b), `scp` to a local
staging copy, `scp` up to the member, THEN start rank 1 → watch both ranks
(inactivity/wrong-tree/budget) → copy both ranks' own logs into the artifact dir (F6, §2) → pull
both `rank-<r>.json` reports (a failed pull joins `rc`) → assemble ONE `gang` artifact with the
MEASURED `pod_count`/`gpu_count_per_pod` threaded in (P-C, §2b), refusing on an unresolved
host/iface or a repeated host compared case-insensitively (F4, §2 + advisory, §2b), **setting
`assembly_ok=1` ONLY on a successful write (P-A2, §2c, new this round)** → on EVERY exit arm, the
EXIT trap (`_rpc_cleanup_cluster`) records self-removal, deletes the cluster (joining a failed
delete into `rc`, F3(c)), **runs the id-secrecy scan (requiring the assembled artifact only when
`assembly_ok=1`, P-A2) and DESTROYS a dirty or genuinely-unexaminable carrier directory
(`_rpc_scan_or_destroy`, P-A + P-A2/P-A3, §2c, renamed from round 2's `_rpc_scan_or_quarantine` —
moved OUT of the main body's own tail in round 2; it no longer runs there at all)**, deletes the
staging id file unconditionally, chains `rp_cleanup` (F2), and exits
(`ci/scripts/runpod_gpu_cluster.sh:738-771` at HEAD).

**Exit contract**, verified against the driver's own exit sites at HEAD: `0` pass; `75` no cluster
capacity; `76` inactivity kill OR the id never crossed within `RP_SSH_WAIT_SECS`; `77` wrong tree;
`97` wrong shape (device-count/compute-cap mismatch, a member's launch-time read-back failure, OR
— new this round, P-C — the MEASURED cluster shape disagreeing with what this driver requested);
`124` budget cut at T-10m; else the driver's own post-run refusal (a failed pull, a failed
assembly/refusal, a failed id-secrecy scan — which now also DESTROYS the carrier directory before
the process exits, P-A/P-A3 — a missing `ens1` line, or a LEAKED cluster on a failed exit-time
delete), by name.

**Cost derivation** (committed, never re-derived per run, unchanged by round 2):
`2 x $1.908/GPU/h = $3.816/h`; terminate-succeeds `1h x $3.816/h = $3.82`/run; sweep-only `(1 +
6)h x $3.816/h = $26.71`. `≤ 1h billed, ≤ 2 runs` is the standing spend authorization dated
2026-09-13.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh` (74 cases, up from 58 at round 1's own close, up
from 50 at c4; verified this round, `bash ci/scripts/test_gpu_cluster_lane.sh`, exit 0) sources the
driver, never executes it. Round 2 adds: the P-A block (§2b, four named exit arms plus an
`id_landed=0` control), the P-C/P-D block (§2b, happy path + 3x8 shape + four refusal/representable
arms), and F10 (§2b, the comment-filter fix's own proving fixture).

## 5. M4 — the id-secrecy scan, post round-2c

See §2 F5, §2b P-A/P-B/advisory, and §2c P-A2/P-A3 in full. Carrier set, current: the pulled
artifact directory (hang-proof, FIFO/socket-refusing, and stack-safe rather than
recursion-limited, P-B), the run log (genuinely inside that directory, F6, ALWAYS required), the
assembled `gang` artifact JSON (OPTIONAL as of round 2c — required ONLY when the caller passes
`--assembled-artifact` at all, i.e. this run's own assembly step claims to have written it; §2c
P-A2), the staging copy's own directory listing. Exit lattice unchanged: `0` clean, `1` hit (named
by carrier and encoding, including a line-wrapped base64 encoding, §2b advisory), `2` UNEXAMINABLE
(covering ANY scanner-internal exception, not only its own documented `ScanTimeout`, §2b P-B).
Round 2 moved WHEN the scan runs: no longer inline near the main body's own tail (round 1's own
shape, which left every `exit` call between "the id lands" and that tail point unscanned) — the
EXIT trap runs it on every one of those arms (§2b P-A). Round 2c corrects WHAT the scan is told to
require on those same arms: `assembly_ok` (a new driver global) gates whether `$ASSEMBLED` is
passed to the scan at all, so a refusal arm that never reached assembly is never told to require a
file it never promised (§2c P-A2), and the carrier-removal wording throughout is corrected to
DESTROY, never "quarantine" (§2c P-A3).

**Oracle**: `gang_id_secrecy_scan.py --self-test` (29 cases, up from 27 at round 2's own close, up
from 23 at round 1's own close, up from 17 at c4; verified this round, exit 0). `test_gpu_cluster_
lane.sh`'s F5 group drives the same scan through `_rpc_run_id_secrecy_scan` against equivalent
fixtures, unchanged in shape since round 2; the P-A/P-A2 block (§2b, §2c, §4) drives the SAME scan
through the real EXIT trap instead, over eight arms (up from round 2's five — three new this
round: happy-path-with-assembly_ok=1, claimed-but-missing, and the corrected assembly-refused/
pull-failed arms that no longer pre-create the assembled artifact unconditionally).

## 6. M5 — P8 (schedule visibility) and the `RENTING_ROOTS` derivation, post round-2

Unchanged in its core property since c4 (`check_p8_schedule_visibility`,
`ci/scripts/check_gpu_prove_once.py:1354-1448` at HEAD, moved from round 1's `1341-1435` — round
2's own `_strip_trailing_comment` fix, below, inserted 13 lines ahead of this function without
touching its own body): a `schedule:` key on a paid-pod-lane workflow (or any workflow mentioning
a `RENTING_ROOTS`-derived driver while carrying the secret) is a FINDING unless the workflow is a
reviewed `PAID_LANE_CRON_ALLOWLIST` (`ci/scripts/check_gpu_prove_once.py:1317-1327` at HEAD, moved
from round 1's `1304-1315`) entry whose token resolves.

**Advisory fix (round 1, unchanged by round 2)**: the allow-list review covers exactly ONE
reviewed cadence per lane — a SECOND `- cron:` entry under the same `schedule:` key, which the
token-resolution check alone cannot see, is its own FINDING:

`_read_schedule_cron_entries` (`ci/scripts/check_gpu_prove_once.py:1331-1350` at HEAD, moved from
round 1's `1318-1338`) reads `on.schedule` as a real list through the SAME PyYAML-backed parse
`read_top_level_on_block` uses (`check_execution_surface_reachability.py`'s own loader, imported as
`exec_mod`), never a second, independently-drifting text scan; its own finding text, naming the
count, sits at `ci/scripts/check_gpu_prove_once.py:1438` (at HEAD, moved from round 1's `1425`).

**Advisory fix (round 1)**: `drop_comment_lines` (`ci/scripts/check_gpu_prove_once.py:491-500` at
HEAD, moved from round 1's `478-487`) strips a TRAILING `# ...` comment off an otherwise-code line
(`_strip_trailing_comment`, `ci/scripts/check_gpu_prove_once.py:456-488` at HEAD — round 2 grew
this from round 1's `456-476` by twelve lines, see below), tracking quoted spans so a `#` inside a
string literal is never mistaken for a comment start — a token that occurs only after a trailing
`#` is prose, never code evidence, and never resolves a P8 allow-list token (or any other "does
this text mention X" search this helper backs).

**Advisory fix (round 2)**: `_strip_trailing_comment` mistoggled its own quote-state on a
backslash-escaped apostrophe — bash's own `'\''` idiom for embedding a literal quote inside a
single-quoted string is three quote characters but only TWO real delimiters, and toggling on all
three left the parser believing it was still inside a string, so a genuine trailing comment
following one would never be stripped. Fixed by skipping the escaped character
(`ci/scripts/check_gpu_prove_once.py:477-479` at HEAD, inside the function's own `while` loop)
rather than toggling on it.

`RENTING_ROOTS` (`ci/scripts/check_gpu_prove_once.py:991` at HEAD, moved from round 1's `978`,
unchanged content) still carries both roots; `PAID_POD_LANE_TABLE`'s cluster row
(`ci/scripts/check_gpu_prove_once.py:892` at HEAD, moved from round 1's `879`, unchanged content)
is unaffected by round 2.

**Oracle**: `test_check_gpu_prove_once.py` (207 cases at HEAD, up from 206 at round 1's own close,
201 pre-round-1; all passing, `python3 -m unittest ci.scripts.test_check_gpu_prove_once`, exit 0).
Round 1's own additions: `RpSshoRequiresRpInitTest` (§2 F1), a second-cron-entry RED case plus a
single-cron GREEN control in `ScheduleVisibilityTest`, `DropCommentLinesTrailingCommentTest` (5
cases). Round 2 adds ONE case to that same class,
`test_a_backslash_escaped_apostrophe_inside_a_single_quoted_string_is_not_a_toggle`
(`ci/scripts/test_check_gpu_prove_once.py:661-672` at HEAD). The pre-existing
`DerivedRentingDriverTest`'s own hard-coded expected-derivation list
(`ci/scripts/test_check_gpu_prove_once.py:1309-1348` at HEAD, moved from round 1's `1296-1335`)
still gained `ci/scripts/test_gpu_cluster_lane.sh` at round 1 — round 1's own F2/F3(c) fixture text
sources `runpod_gpu_cluster.sh` in a real subshell and names `rp_cluster_delete`/`rp_cleanup` in
non-comment text, so the deliberately over-approximating derivation scan (§0's own doctrine)
derives it too; round 2's own new P-A/P-C/P-D fixture text (§2b) sources the same file the same
way and adds no NEW derived driver, only more non-comment mentions of the same one already listed;
cleared the identical way every sibling test file already is, by `ci.yml`'s guard job carrying no
`RUNPOD_API_KEY`.

## 7. M6 — artifact registry leg discrimination, post round-2

See §2 F4 and §2b advisory (case-insensitive host compare) in full for the fix itself. `gang.leg`
remains a required, closed-set field (`"pod"`|`"cluster"`), checked first. Pod-leg rows unchanged.
Cluster-leg rows (`GANG_CLUSTER_FIELD_REGISTRY`, `ci/scripts/check_cuda_run_artifacts.py:1624-1676`
at HEAD, moved from round 1's `1610-1668`) now carry the strengthened
`ranks[]`/cross-field/producer-binding properties §2 F4 describes, and `_gang_check_cluster_shape`'s
own `pod_count`/`gpu_count_per_pod` cross-field check is non-vacuous for the first time since round
2's own P-C threads the driver's MEASURED shape rather than a hardcoded literal into the field the
checker compares against (§2b).

**Oracle**: `check_cuda_run_artifacts.py --self-test` — verified green this round
(`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`, exit 0), all rule (k) cases
including the nine F4 arms named in §2/§2b (eight at round 1's own close, plus round 2's own
case-insensitive-host arm).

## 8. The executed attempt — status, unchanged by either fix round

No real RunPod cluster run has happened. `CONTRACT-U7b.md §2` schedules one at the end of c3,
authorized, bounded by §4's own cost figures — its log would be the evidence for the `args`
entrypoint reaching bash on `RP_IMAGE` (§1), member sshd reachability, `ens1` as the overlay
iface, and whether member self-removal works. Both fix rounds are entirely mocks-only (per their
own briefs: never call the RunPod API, no key available to this agent) and change nothing about
this status. A failed run, when one is executed, is itself a FINDING to be recorded and fixed
(re-run once at most, within the 2-run authorization).

## 9. Known-unmeasured / uncovered (named, never claimed closed)

- **Member self-removal on a cluster pod.** Unknown until §8's run executes and records
  `cluster-self-remove: ok|refused` (`_rpc_self_remove_status`/`_rpc_cleanup_cluster`,
  `ci/scripts/runpod_gpu_cluster.sh:655-771` at HEAD, moved from round 2's `644-734` by round 2c,
  §2c below; itself moved from round 1's `570-608`).
- **The REST v2 `args` field reaching `bash -c` on `RP_IMAGE`.** Never independently confirmed —
  §1. The launch-time read-back is the guard against this being false, not a proof it is true.
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
merge-path); B6 (the ai-core test body and the docs-ci lane land together — this branch's own
twelve commits below the merge, §0's header, plus round 1's fix, round 2's fix, round 2c's fix
(`23ef24a9`), and this contract revision's own commit — 16 total on `git log --format=%h
main..HEAD` once this file's own commit lands); K2 (every parsed API body is validated before use
— round 2 adds the cluster's own `compute` block, §2b P-C).

## 11. Gate files a human must review at this unit's merge

`swarm.yml`'s human-amend-only glob (`SWARM_GATE_TOUCHED`) covers gate-script edits; the reviewer
checks:

- `ci/scripts/check_gpu_prove_once.py` — P8's schedule-cron-count check, the trailing-comment
  stripper (§2 F6 advisory / §6), and round 2's own backslash-apostrophe fix to that same stripper
  (§2b advisory).
- `ci/scripts/check_cuda_run_artifacts.py` — rule (k)'s strengthened cluster-leg registry, the
  leg/producer binding (§2 F4), and round 2's own case-insensitive host compare (§2b advisory).
- `ci/scripts/runpod_lib.sh` — `rp_sweep`'s now-fatal refused-terminate arm and
  `rp_cluster_sweep`'s now-fatal `UNAGEABLE` arm (§2 F3) — UNTOUCHED by round 2 (read-only this
  round; verified via `git status` at commit time).
- `ci/scripts/runpod_gpu_cluster.sh` — `rp_init` (§2 F1), the relocated/chained EXIT trap (§2
  F2/F3(c)), the assembly-time refusal and RUN_LOG relocation (§2 F4/F6), round 2's own
  measured-shape refusal, EXIT-trap scan-or-destroy, and unconditional staging-file delete
  (§2b P-A/P-C), and round 2c's own `assembly_ok` threading (`_rpc_scan_or_destroy`, renamed from
  `_rpc_scan_or_quarantine`) and DESTROY wording (§2c P-A2/P-A3).
- `ci/scripts/gang_id_secrecy_scan.py` — the cycle-safe walk, the regular-file-only read, and the
  wall-clock budget (§2 F5), round 2's own explicit-stack rewrite, total exception lattice, and
  whitespace-stripped base64 matching (§2b P-B/advisory), and round 2c's own optional
  `--assembled-artifact` (§2c P-A2).
- `ci/scripts/runpod_gpu_prove.sh` — round 2's own logged (never silently discarded) pre-run
  `rp_sweep` rc (§2b advisory).

## 12. Residuals recorded UNCOVERED

- The un-executed REST v2 pre-flight (§1) and the un-executed cluster run (§8) — both named, not
  silently deferred.
- Member self-removal on a cluster object (§9).
- The NCCL pin set at world >= 3 (§9).
- S5's cross-host byte identity (§9).
- **U7b-A3** (the `gpu-gang.yml` 6-hourly cron re-add) is explicitly NOT part of this unit.
- Every residual PR-B1's own contract already recorded UNCOVERED for P6/P7 is unchanged by
  either fix round.

## 13. Citations verified against which head

Every construct cited above was read directly against this worktree's tree via `grep -n`/direct
file reads, AFTER round 2c's own fix commit (`23ef24a9`) landed — the last step before this file
was written — never against a scratchpad working document's own line numbers, and never against
`CONTRACT-U7b.md`'s own §7/§8 fold text beyond citing its decisions by name. Every citation is the
bare `path:line` (or `path:line-line`) form `check_rigor_record.py`'s own
`check_path_line_citations` matches, tagged `(at <sha>)` per this file's own header convention
(§0); a citation this round corrected (never merely re-derived) says so explicitly at its own
site, rather than silently overwriting an earlier round's own prose. Round 2c re-derived, against
`23ef24a9`'s own tree, every prior citation into `ci/scripts/runpod_gpu_cluster.sh`,
`ci/scripts/gang_id_secrecy_scan.py`, and `ci/scripts/test_gpu_cluster_lane.sh` tagged `(at HEAD)`
in §2/§2b/§4/§5/§9/§11 (every file round 2c also touched) — each is either updated to its new line
number with a "moved from round 2's `N`" note, or, where the underlying construct round 2's own
prose narrates no longer exists under that name (§2b's own P-A subsection, which narrates round
2's fix as it stood before round 2c), pinned explicitly `(at c4f0c36e's own tree)` instead of
`(at HEAD)`. Self-check command for every citation in this revision: `git checkout <sha> --
<path> 2>/dev/null; sed -n '<line>,<line>p' <path>` against this branch's own worktree (`git log
--format=%h main..HEAD` lists every sha above).
