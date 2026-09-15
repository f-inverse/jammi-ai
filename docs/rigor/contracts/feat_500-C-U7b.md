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

Owner: **docs-ci** (fix round 1, dispatched after the c4 revision's closing BLOCK). Commit
history on this branch, oldest first, below the `main` merge (`git log --format='%h %s'
main..HEAD`, ten commits, the c1-c4 owners' own units plus one ai-core fix landed after c4):

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
d12e1689 test(ai): #500 U7b — a missing hostname or NCCL iface is a fail verdict, never ...
```

(`c23049f9`, a `main` merge, sits between `ed2e7c3c` and `1480cacb` in the branch's actual
commit graph — omitted from the list above since it carries no U7b content of its own.) This fix
round lands as an eleventh commit, by pathspec, atop `d12e1689`; every citation to a file this
round touched is tagged `(at HEAD)` — the fix commit's own tree, which this contract is
committed inside — rather than a sha that does not exist until that commit lands. Every citation
to a file this round did NOT touch is tagged at the sha that last touched it, per `git log -1
--format=%h -- <path>` against this same tree.

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
(`ci/scripts/runpod_gpu_cluster.sh:258-296` at HEAD) reads `GET /v2/clusters/{id}/pods` after
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
655:rp_init
658:cluster_id="$(rp_cluster_create "$RP_CLUSTER_GPU_TYPE" "$dcs")" || { echo "::error::cluster create failed"; exit 75; }
```

`ci/scripts/runpod_gpu_cluster.sh:655` precedes `:658` (at HEAD).

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh:542-551` (at HEAD) is a new static guard — a
line-number comparison over the committed driver text (`grep -n '^rp_init$'` vs. the first
`rp_cluster_create ` call) — asserting `rp_init` precedes the create call, never a behavioral
probe (no network). A SECOND, class-level guard closes the general case this specific line-number
check does not: `test_check_gpu_prove_once.py`'s `RpSshoRequiresRpInitTest`
(`ci/scripts/test_check_gpu_prove_once.py:587-627` at HEAD) statically scans EVERY real
`PAID_POD_LANE_TABLE` driver on disk and asserts that any driver referencing `RP_SSHO[` also
calls `rp_init` (a bare-line regex, `RP_INIT_CALL_RE`), with its own RED-then-GREEN self-test
proving the regex actually distinguishes a call from a mention in prose.

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
606:  rp_cleanup  # F2: chain the library's own EXIT cleanup (rm -rf "$RP_WORK" -- the staging id file and the ssh keypair).
```

`ci/scripts/runpod_gpu_cluster.sh:591-608` (at HEAD) is `_rpc_cleanup_cluster`'s own body,
moved OUT of the sourced-execution guard (it was previously defined only when the file is
EXECUTED, making it untestable by sourcing) into the pure-helpers section above it — only the
`trap _rpc_cleanup_cluster EXIT` registration itself (`ci/scripts/runpod_gpu_cluster.sh:665` at
HEAD) remains inside the guard.

**Oracle**: `test_gpu_cluster_lane.sh`'s new F2/F3(c) block (`ci/scripts/test_gpu_cluster_lane.sh:104-191`
at HEAD) drives the REAL `_rpc_cleanup_cluster` in a real subprocess (`bash -c '... source
"$CLUSTER_SH" ...'`, so its own `exit "$rc"` terminates that subprocess exactly the way a real
EXIT trap fires), mocking only `_rp_rest`/`rp_cluster_delete`/`rp_cleanup`: self-remove-ok,
self-remove-refused-delete-ok, and no-cluster-id-at-all all confirm `rp_cleanup` is chained
(a marker file it writes exists afterward) in every arm.

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
602:      echo "::error::LEAKED cluster ${cluster_id}: could not delete on exit -- gpu-reap.yml's 6-hourly sweep is the backstop"
```

`ci/scripts/runpod_gpu_cluster.sh:591-608` (at HEAD, the same span F2 cites) is the full trap
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

`_gang_check_cluster_ranks` (`:1376-1461`) now asserts `host` distinct across ranks
(`:1450`), refuses `host`/`iface` matching `GANG_UNKNOWN_SENTINELS = ("unknown", "")`
(`:1373`, `:1418-1423`), requires a PER-RANK `reduced_vector_digest`
(hex-validated on `pass`, `:1428-1439`) and asserts those per-rank digests equal across ranks on
`pass` (`:1456-1461`) — never trusting the already-collapsed top-level field alone.
`_gang_check_cluster_shape` (`:1463-1494`) asserts `hosts == pod_count` and `world == pod_count *
gpu_count_per_pod`. `_gang_check_leg_producer_binding` (`:1496-1511`), wired into
`check_gang_artifact` right after the leg is resolved, binds `gang.leg == "cluster"` to
`producer.path == "ci/scripts/runpod_gpu_cluster.sh"` (and `"pod"` to
`"ci/scripts/runpod_gpu_gang.sh"`, `GANG_LEG_PRODUCER_PATH:1023-1032`) — a self-declared leg can
no longer dodge the other leg's registry by pointing `producer.path` at a different driver.

**Fix, on the driver side** (`ci/scripts/runpod_gpu_cluster.sh`, at HEAD):
`_rpc_assemble_gang_artifact` now refuses to WRITE an artifact at all when any rank's own
`hostname`/`nccl_socket_ifname` is empty or `unknown`, or when both ranks report the same
`hostname`:

```
$ grep -n 'refusing to assemble' ci/scripts/runpod_gpu_cluster.sh
473:        print("refusing to assemble: rank %r own hostname is unresolved (%r)" % (r.get("rank"), host), file=sys.stderr)
476:        print("refusing to assemble: rank %r own nccl_socket_ifname is unresolved (%r)" % (r.get("rank"), iface), file=sys.stderr)
479:    print("refusing to assemble: both ranks report the SAME host (%r) -- not the two-host bootstrap this leg proves" % reports[0].get("hostname"), file=sys.stderr)
```

— a named refusal (exit 2 from the assembler; joined into the driver's own `rc` by the existing
`_rpc_assemble_gang_artifact ... || { ...; [ "$rc" -eq 0 ] && rc=1; }` call site), never an
artifact synthesized from unresolved data. Each rank's report now carries its OWN
`reduced_vector_digest` (`ci/scripts/runpod_gpu_cluster.sh:482-493` at HEAD, inside the `ranks`
list construction), and the assembled artifact's `producer` block is now bound to this driver's
own path (`ci/scripts/runpod_gpu_cluster.sh:518-523` at HEAD: `"path":
"ci/scripts/runpod_gpu_cluster.sh"`, `"kind": "script"`), matching `GANG_LEG_PRODUCER_PATH` above.
(A complementary fix on the Rust side, `crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` at
`d12e1689` — outside this contract's own owned files, landed by an ai-core agent in the same
worktree during this fix round — makes `hostname()` return a named `Result` instead of masking a
failed read behind `"unknown"`, so the driver-side refusal above is checking a value that itself
can no longer silently BE `"unknown"` on a healthy run.)

**Oracle**: `check_cuda_run_artifacts.py --self-test`'s rule (k) self-test gains, named
(`ci/scripts/check_cuda_run_artifacts.py:3343-3429` at HEAD): two ranks on one host FAILS; an
`unknown` host FAILS; an `UNKNOWN` (any case) iface FAILS; a per-rank digest mismatch on `pass`
FAILS; a missing per-rank digest on `pass` FAILS; `hosts != pod_count` FAILS; `world != pod_count
* gpu_count_per_pod` FAILS; a cluster-leg artifact carrying the pod leg's own `producer.path`
FAILS (and vice versa). `gang_baseline()`/`gang_cluster_baseline()`
(`ci/scripts/check_cuda_run_artifacts.py:3148-3226` at HEAD) now stamp each leg's real producer
path (`ci/scripts/runpod_gpu_gang.sh` / `ci/scripts/runpod_gpu_cluster.sh`), and the self-test's
own fixture repo gains tracked stand-ins for both paths
(`ci/scripts/check_cuda_run_artifacts.py:2745-2751` at HEAD) so rule (b)'s own
producer.path-exists-and-is-tracked check has something real to bind against.

### F5 — the id-secrecy scan could hang, and could read a FIFO/socket forever

`scan_dir` walked the pulled artifact directory via `os.walk(root, followlinks=True)` — that
detects no cycles at all, so a cyclic DIRECTORY symlink (`rsync -a` preserves one exactly as
planted) recurses forever. `scan_file` called `real.read_bytes()` on ANY non-directory path
regardless of its `st_mode` class — a `read_bytes()` against a FIFO or a UNIX socket with nothing
on the other end blocks forever, never returning.

**Fix** (`ci/scripts/gang_id_secrecy_scan.py`, at HEAD):

```
$ grep -n 'def scan_dir\|def wall_clock_budget\|class ScanTimeout' ci/scripts/gang_id_secrecy_scan.py
190:def scan_dir(root: Path, needles: list[tuple[str, bytes]]) -> list[tuple[int, str]]:
249:class ScanTimeout(Exception):
254:def wall_clock_budget(seconds: int):
```

`scan_dir` (`:190-234`) no longer calls `os.walk` at all — a private recursive `walk()` closure
tracks the REAL path of every directory it enters in a `visited_dirs` set; a directory whose real
path repeats is one `"cyclic carrier"` UNEXAMINABLE finding, never a re-descent.
`scan_file` (`:66-96`) now checks `stat.S_ISREG` explicitly before ever calling `read_bytes()` —
any other mode class (FIFO, socket, device) is refused by name (UNEXAMINABLE), never opened.
`wall_clock_budget` (`:254-271`, a `contextlib.contextmanager` over `signal.alarm`) wraps the
WHOLE scan body (`run_scan`/`_run_scan_body`, `:179-183`); an expiry raises `ScanTimeout`, caught
by `run_scan` and reported UNEXAMINABLE — independent of any single carrier's own shape, the
scan's own last line of defense. Default budget 120s
(`DEFAULT_BUDGET_SECS`/`GANG_ID_SCAN_BUDGET_SECS` env override,
`ci/scripts/gang_id_secrecy_scan.py:91` at HEAD), overridable via `--budget-secs`.

**Advisory, also fixed**: `id_needles` (`ci/scripts/gang_id_secrecy_scan.py:118-142` at HEAD) now
computes FOUR base64 variants — standard padded, standard un-padded, URL-safe padded, URL-safe
un-padded — rather than one.

**Oracle**: `gang_id_secrecy_scan.py --self-test` gains, named: a cyclic directory symlink
(returns promptly, UNEXAMINABLE, never hangs); a FIFO under the pulled dir (UNEXAMINABLE, "not a
regular file"); a UNIX socket under the pulled dir (UNEXAMINABLE, same message); a wall-clock
budget expiry (`scan_dir` mocked to sleep past a 1s budget, UNEXAMINABLE, "wall-clock budget");
base64-urlsafe and base64-unpadded planted-id hits. 23 `unittest` cases total (up from 17 at c4),
all passing (`python3 ci/scripts/gang_id_secrecy_scan.py --self-test`, verified this round).

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
621:mkdir -p "$CLUSTER_ARTIFACT_DIR"
622:RUN_LOG="$CLUSTER_ARTIFACT_DIR/run.log"
811:mkdir -p "$CLUSTER_ARTIFACT_DIR"
```

the directory is created BEFORE the tee starts (`:623`, `exec > >(tee -a "$RUN_LOG") 2>&1`), so
the run log lives inside the uploaded/scanned directory from its first byte; the SECOND `mkdir -p`
match (`:811`) is a defensive, idempotent re-assertion immediately before the rank-log copies
below — never a second, independent creation site with its own drift risk. Both ranks' own
remote logs are now ALSO copied there, unconditionally, pass or fail:

```
$ grep -n 'cp -f "\$rank0_log"\|cp -f "\$rank1_log"' ci/scripts/runpod_gpu_cluster.sh
```

placed immediately after both `wait` calls (`ci/scripts/runpod_gpu_cluster.sh:807-813` at HEAD),
before any pass/fail branching. The workflow's own upload step needed no change — `path:
.gpu-pull/gpu-cluster/` already covers the directory the run log now lives inside.

**Oracle**: `test_gpu_cluster_lane.sh`'s new F6(a) block
(`ci/scripts/test_gpu_cluster_lane.sh:478-494` at HEAD) statically asserts the `mkdir` line
precedes the `RUN_LOG=` assignment line, and that both `cp -f` lines exist.

**(b) This citation form itself.** Every citation in this revision is the bare `path:line` (or
`path:line-line`) form, tagged `(at <sha>)`, re-derived by direct read against this tree AFTER
every fix above landed — never a fenced `grep -n`/`sed -n` transcript (the c4 revision's own
form, which carries zero tokens `check_path_line_citations`'s regex matches, and — separately —
had gone stale in 3 of 16 transcripts by the time of the closing audit, since a transcript's own
output is never re-verified by that checker at all).

## 3. M2 — the two-host NCCL test body (`gang_nccl.rs`) — unchanged by this fix round

`gang_nccl_two_hosts_reduce_a_known_vector`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` at `d12e1689`) is feature-gated and reads its
own env contract (`JAMMI_GANG_TWO_HOSTS_RANK`/`_WORLD`/`_ID_FILE`,
`JAMMI_REQUIRE_CUDA_TWO_HOSTS`). Rank 0 mints `Nccl::new_id()` and writes it atomically; rank 1
refuses any id file whose size is not exactly 128 bytes. `d12e1689` (landed by an ai-core agent
in this same worktree during this fix round, outside this contract's own owned files) closes a
sibling finding to this round's own F4: `hostname()` now returns a named `Result` instead of
masking a failed read behind `"unknown"`, and `missing_report_metadata_reason` names which of
hostname/iface is missing before any NCCL work runs, so a bad metadata read panics rather than
writing an indistinguishable-from-real `"unknown"` pass report.

**Oracle**: compiles under `cargo clippy -p jammi-ai --features live-gpu-tests --test
gpu_capability` (the gated-surface clippy step; not itself proof the test PASSES, only that it
compiles). The real proof is the executed run (§8) — not yet performed.

## 4. M3 — the cluster driver, post-fix — sequence and exit contract

Never `runpod_gpu_gang.sh` — a fully separate driver, workflow, and RunPod object type. Sequence,
post-fix: read per-data-center availability and pass only qualifying `dataCenterIds` (A1) → `rp_init`
(F1, §2) → create ONE 2x1 cluster → poll both members RUNNING with a usable ssh path
(`_rpc_check_readback`, §1) → build both members in parallel via one shared per-rank heredoc
(`_rpc_remote_script`) → start rank 0, poll the 128-byte id file, `scp` to a local staging copy,
`scp` up to the member, THEN start rank 1 → watch both ranks (inactivity/wrong-tree/budget) →
copy both ranks' own logs into the artifact dir (F6, §2) → pull both `rank-<r>.json` reports (a
failed pull joins `rc`) → assemble ONE `gang` artifact, refusing on an unresolved host/iface or a
repeated host (F4, §2) → run the id-secrecy scan, now hang-proof (F5, §2) → on EVERY exit arm,
the EXIT trap (`_rpc_cleanup_cluster`) records self-removal, deletes the cluster (joining a
failed delete into `rc`, F3(c)), chains `rp_cleanup` (F2), and exits (`ci/scripts/runpod_gpu_cluster.sh:591-608`
at HEAD).

**Exit contract**, unchanged in shape by this round, verified against the driver's own exit
sites at HEAD: `0` pass; `75` no cluster capacity; `76` inactivity kill OR the id never crossed
within `RP_SSH_WAIT_SECS`; `77` wrong tree; `97` wrong shape (device-count/compute-cap mismatch,
OR a member's launch-time read-back failure); `124` budget cut at T-10m; else the driver's own
post-run refusal (a failed pull, a failed assembly/refusal, a failed id-secrecy scan, a missing
`ens1` line, or — new this round, F2/F3(c) — a LEAKED cluster on a failed exit-time delete), by
name.

**Cost derivation** (committed, never re-derived per run, unchanged by this round):
`2 x $1.908/GPU/h = $3.816/h`; terminate-succeeds `1h x $3.816/h = $3.82`/run; sweep-only `(1 +
6)h x $3.816/h = $26.71`. `≤ 1h billed, ≤ 2 runs` is the standing spend authorization dated
2026-09-13.

**Oracle**: `ci/scripts/test_gpu_cluster_lane.sh` (58 cases, up from 50 at c4; verified this
round, `bash ci/scripts/test_gpu_cluster_lane.sh`, exit 0) sources the driver, never executes it.
Beyond the c4-era G0-G7/F2/F3/F5/F11/A5 cases (unchanged in shape), this round adds: F1 (§2), the
F2/F3(c) EXIT-trap block (§2), F6(a) (§2).

## 5. M4 — the id-secrecy scan, post-fix

See §2 F5 in full. Carrier set unchanged: the pulled artifact directory (now hang-proof and
FIFO/socket-refusing), the run log (now genuinely inside that directory, F6), the assembled `gang`
artifact JSON, the staging copy's own directory listing. Exit lattice unchanged: `0` clean, `1`
hit (named by carrier and encoding), `2` UNEXAMINABLE.

**Oracle**: `gang_id_secrecy_scan.py --self-test` (23 cases, up from 17 at c4; verified this
round, exit 0). `test_gpu_cluster_lane.sh`'s F5 group drives the same scan through
`_rpc_run_id_secrecy_scan` against equivalent fixtures, unchanged in shape by this round.

## 6. M5 — P8 (schedule visibility) and the `RENTING_ROOTS` derivation, post-fix

Unchanged in its core property from c4 (`check_p8_schedule_visibility`,
`ci/scripts/check_gpu_prove_once.py:1341-1435` at HEAD): a `schedule:` key on a paid-pod-lane
workflow (or any workflow mentioning a `RENTING_ROOTS`-derived driver while carrying the secret)
is a FINDING unless the workflow is a reviewed `PAID_LANE_CRON_ALLOWLIST`
(`ci/scripts/check_gpu_prove_once.py:1304-1315` at HEAD) entry whose token resolves.

**Advisory fix this round**: the allow-list review covers exactly ONE reviewed cadence per lane —
a SECOND `- cron:` entry under the same `schedule:` key, which the token-resolution check alone
cannot see, is now its own FINDING:

`_read_schedule_cron_entries` (`ci/scripts/check_gpu_prove_once.py:1318-1338` at HEAD) reads
`on.schedule` as a real list through the SAME PyYAML-backed parse `read_top_level_on_block` uses
(`check_execution_surface_reachability.py`'s own loader, imported as `exec_mod`), never a second,
independently-drifting text scan; its own finding text, naming the count, sits at
`ci/scripts/check_gpu_prove_once.py:1425` (at HEAD).

**Advisory fix, also this round**: `drop_comment_lines`
(`ci/scripts/check_gpu_prove_once.py:478-487` at HEAD) now strips a TRAILING `# ...` comment off
an otherwise-code line (`_strip_trailing_comment`, `ci/scripts/check_gpu_prove_once.py:456-476`
at HEAD), tracking quoted spans so a `#` inside a string literal is never mistaken for a comment
start — a token that occurs only after a trailing `#` is prose, never code evidence, and no
longer resolves a P8 allow-list token (or any other "does this text mention X" search this helper
backs).

`RENTING_ROOTS` (`ci/scripts/check_gpu_prove_once.py:978` at HEAD, unchanged) still carries both
roots; `PAID_POD_LANE_TABLE`'s cluster row (`ci/scripts/check_gpu_prove_once.py:879` at HEAD,
unchanged) is unaffected by this round.

**Oracle**: `test_check_gpu_prove_once.py` (206 cases at HEAD, up from 201 pre-round; all
passing, `python3 ci/scripts/test_check_gpu_prove_once.py`, exit 0). This round's additions:
`RpSshoRequiresRpInitTest` (§2 F1), a second-cron-entry RED case plus a single-cron GREEN control
in `ScheduleVisibilityTest`, `DropCommentLinesTrailingCommentTest` (5 cases). The pre-existing
`DerivedRentingDriverTest`'s own hard-coded expected-derivation list
(`ci/scripts/test_check_gpu_prove_once.py:1296-1335` at HEAD) gained
`ci/scripts/test_gpu_cluster_lane.sh` — this round's own F2/F3(c) fixture text sources
`runpod_gpu_cluster.sh` in a real subshell and names `rp_cluster_delete`/`rp_cleanup` in
non-comment text, so the deliberately over-approximating derivation scan (§0's own doctrine)
derives it too; cleared the identical way every sibling test file already is, by `ci.yml`'s guard
job carrying no `RUNPOD_API_KEY`.

## 7. M6 — artifact registry leg discrimination, post-fix

See §2 F4 in full for the fix itself. `gang.leg` remains a required, closed-set field
(`"pod"`|`"cluster"`), checked first. Pod-leg rows unchanged. Cluster-leg rows
(`GANG_CLUSTER_FIELD_REGISTRY`, `ci/scripts/check_cuda_run_artifacts.py:1610-1668` at HEAD) now
carry the strengthened `ranks[]`/cross-field/producer-binding properties §2 F4 describes.

**Oracle**: `check_cuda_run_artifacts.py --self-test` — verified green this round
(`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`, exit 0), all rule (k) cases
including the eight new F4 arms named in §2.

## 8. The executed attempt — status, unchanged by this fix round

No real RunPod cluster run has happened. `CONTRACT-U7b.md §2` schedules one at the end of c3,
authorized, bounded by §4's own cost figures — its log would be the evidence for the `args`
entrypoint reaching bash on `RP_IMAGE` (§1), member sshd reachability, `ens1` as the overlay
iface, and whether member self-removal works. This fix round is entirely mocks-only (per its own
brief: never call the RunPod API, no key available to this agent) and changes nothing about this
status. A failed run, when one is executed, is itself a FINDING to be recorded and fixed (re-run
once at most, within the 2-run authorization).

## 9. Known-unmeasured / uncovered (named, never claimed closed)

- **Member self-removal on a cluster pod.** Unknown until §8's run executes and records
  `cluster-self-remove: ok|refused` (`_rpc_self_remove_status`/`_rpc_cleanup_cluster`,
  `ci/scripts/runpod_gpu_cluster.sh:570-608` at HEAD).
- **The REST v2 `args` field reaching `bash -c` on `RP_IMAGE`.** Never independently confirmed —
  §1. The launch-time read-back is the guard against this being false, not a proof it is true.
- **The `NCCL_SOCKET_IFNAME=ens1`/pin set at world >= 3.** This unit's own leg proves world 2
  only. `docs/maintainer/dev-gpu.md`'s own "Known-unmeasured" section (at `5ebe53ab`, unchanged)
  states this and points back here.
- **S5's cross-host byte identity.** S5 (README.md, at `9f69275b`, unchanged) measured
  byte-identical forward/backward/SGD across two PROCESSES on two A100s, never a cross-HOST NCCL
  collective; this leg's `reduced_vector_digest` equality is a narrower, different measurement
  (a bit-exact sum, not a training step's digest pair) and does not itself extend S5's claim.

## 10. Invariants crossed

B2 (every script and doc here names no consumer — `python3 ci/scripts/check_no_consumer_names.py`
verified green this round); the paid-lane doctrine (P1/P7/P8: label/dispatch only, never
merge-path); B6 (the ai-core test body and the docs-ci lane land together — this branch's own ten
commits below the merge, §0's header, plus this fix round's own eleventh); K2 (every parsed API
body is validated before use).

## 11. Gate files a human must review at this unit's merge

`swarm.yml`'s human-amend-only glob (`SWARM_GATE_TOUCHED`) covers gate-script edits; the reviewer
checks:

- `ci/scripts/check_gpu_prove_once.py` — P8's schedule-cron-count check and the trailing-comment
  stripper (§2 F6 advisory / §6).
- `ci/scripts/check_cuda_run_artifacts.py` — rule (k)'s strengthened cluster-leg registry and the
  leg/producer binding (§2 F4).
- `ci/scripts/runpod_lib.sh` — `rp_sweep`'s now-fatal refused-terminate arm and
  `rp_cluster_sweep`'s now-fatal `UNAGEABLE` arm (§2 F3).
- `ci/scripts/runpod_gpu_cluster.sh` — `rp_init` (§2 F1), the relocated/chained EXIT trap (§2
  F2/F3(c)), the assembly-time refusal and RUN_LOG relocation (§2 F4/F6).
- `ci/scripts/gang_id_secrecy_scan.py` — the cycle-safe walk, the regular-file-only read, and the
  wall-clock budget (§2 F5).

## 12. Residuals recorded UNCOVERED

- The un-executed REST v2 pre-flight (§1) and the un-executed cluster run (§8) — both named, not
  silently deferred.
- Member self-removal on a cluster object (§9).
- The NCCL pin set at world >= 3 (§9).
- S5's cross-host byte identity (§9).
- **U7b-A3** (the `gpu-gang.yml` 6-hourly cron re-add) is explicitly NOT part of this unit.
- Every residual PR-B1's own contract already recorded UNCOVERED for P6/P7 is unchanged by this
  unit or this fix round.

## 13. Citations verified against which head

Every construct cited above was read directly against this worktree's tree via `grep -n`/direct
file reads, AFTER every fix in §2 landed — the last step before this file was written — never
against a scratchpad working document's own line numbers, and never against `CONTRACT-U7b.md`'s
own §7/§8 fold text beyond citing its decisions by name. Every citation is the bare `path:line`
(or `path:line-line`) form `check_rigor_record.py`'s own `check_path_line_citations` matches,
tagged `(at <sha>)` per this file's own header convention (§0).
