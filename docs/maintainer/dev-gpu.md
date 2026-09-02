# GPU development on RunPod

jammi's default `cargo test` is CPU-hermetic; the GPU path is exercised only by
the gated `live-gpu-tests` lane. You don't need a local GPU — rent one per task
on RunPod through `ci/scripts/gpu-dev.sh`.

**Looking for how to do a specific thing?** [dev-gpu-recipes.md](dev-gpu-recipes.md)
is the task-oriented walkthrough. This page is the design and setup reference.

A pod has two independent axes:

- **lifetime** — `shell` gives a throwaway pod that dies when you exit. `up`
  starts a named session whose pod *survives* disconnect, so a fine-tune, eval or
  bench keeps running after you close the terminal.
- **state** — the pod itself is always disposable. The only thing worth keeping
  lives in git (your working tree, via `push`/`--ref`). Build-time COMPILATION
  state — the expensive part — lives in a per-pod **seed/clone build
  substrate** instead (below): a full-registry `CARGO_TARGET_DIR` (the
  "seed") built once per pod and cheaply `cp -a`-cloned into one throwaway
  `CARGO_TARGET_DIR` per tree (`target`, below) — never shared across pods, and
  never backed by an external object store. Nothing durable is ever stored on
  the pod itself.

## Why no network volume is attached

RunPod network volumes are Secure-Cloud-only and locked to a single datacenter,
and can only be attached at deploy time. Attaching one would delete both failover
dimensions in `runpod_lib.sh` — cloud tier and PCIe/SXM variant — and those exist
precisely because A100 supply is intermittent. Pinning the pod to one datacenter
trades away availability for persistence we can get more cheaply.

So the pod stays free to land anywhere, and every mechanism below (the
seed/clone substrate, `push`) is per-pod and self-contained rather than
depending on shared, location-pinned storage.

## One-time setup

A RunPod API key (RunPod → Settings → API Keys):

```bash
mkdir -p ~/.config/runpod && printf '%s' 'YOUR_KEY' > ~/.config/runpod/key && chmod 600 ~/.config/runpod/key
```

(CI reads the same key from the `RUNPOD_API_KEY` GitHub Actions secret.)

Nothing else to configure — the build-substrate cache below is entirely
per-pod, no second credential needed.

## The build substrate — seed and clone

Compilation is the real cost on a fresh pod, and `gpu-dev.sh` no longer pays
it more than once per pod. Right after bootstrap, `up`/`shell` kick off a
**seed** build, detached (`tmux attach -t =jammi-seed` on the pod to watch
it): a `CARGO_TARGET_DIR` with every third-party dependency fully compiled,
then made **member-free** — `cargo clean --workspace` (both profiles used) plus
an explicit `rm -rf */incremental` (cargo's own cleaner does not remove
`incremental/build_script_build-*`) strip every `jammi-*` artifact back out,
so the seed is pure registry output with nothing of jammi's own code baked in.

`gpu-dev.sh target <session> <name>` then `cp -a` (reflink where the
filesystem supports it) clones that seed into a fresh `CARGO_TARGET_DIR` for a
**tree** (`--tree <name>`, default the bootstrap checkout at
`/root/jammi-ai`). Because the seed is member-free, a clone is a **pure
copy** — no deletion step, no drift window — and every `jammi-*` unit
genuinely recompiles on the clone's first build (`target --verify` proves it:
no `Fresh jammi-*` line). Every third-party dependency, meanwhile, is already
built: only jammi's own code and whatever the tree's own `Cargo.lock`/feature
set actually changed ever compiles again.

This replaced an S3-backed sccache cache tried earlier: measured live on a
real pod (fresh `CARGO_TARGET_DIR` each leg, `cargo build --release -p
jammi-bench --features cuda`; session ledger row 17, producer
`/root/sccache-remeasure.sh` — `.jammi/ledger/` is gitignored, so the row is
named here rather than linked), sccache gave **zero cross-target-dir cache
reuse** for rustc units — every populate-then-reuse pair against a fresh
target dir re-missed everything sccache had just written — while adding
**+33% to +37.5% wall clock** to every build that ran it (344s wrapper-off
vs 457-473s wrapper-on: low end 344→457s is (457-344)/344 = +32.8% ≈ +33%;
high end 344→473s is (473-344)/344 = +37.5% — a single "~+33%" figure
previously cited only the low end, and a later revision of this line
rounded the high end up to "+38%", which does not match the arithmetic
either; both ends are now stated to the precision the same row actually
supports).
The wrapper is now off pod-wide
(`CARGO_BUILD_RUSTC_WRAPPER=` in `/root/.jammi_env`, every shell sources it).

The cargo **registry** is deliberately not cached either. It looks like the
expensive part, since the CI image wipes `/usr/local/cargo/registry`, but a
cold `cargo fetch --locked` measures **9s for 868 crates** on a RunPod host
(same measurement session as the sccache figures above; the specific ledger
row for this particular number is not separately pinned in the audit trail
this doc's other citations draw from — a follow-up, not a claim this line
retracts) — datacenter bandwidth makes it free.

Disk sizing (`RP_DISK_GB`): `>= 25` (base) `+ S_src + S_seed + N*S_clone`
(one clone per tree the pod hosts). Measured by this formula's producer,
`ci/scripts/perf/pod_build_timings.sh` (committed JSON:
`ci/artifacts/pod-build-timings/20260827T183928Z-bc27e75.json`, an
A100-SXM4 secure-cloud pod at `bc27e75`; all values decimal GB from the
artifact's exact byte fields): `S_src` ≈ 3.6 GB (the checkout, `.git`
included), `S_seed` ≈ 7.8 GB, `S_clone` ≈ 8.1 GB. By the formula, the
default `RP_DISK_GB=60` covers base + src + seed + **two** clones
(≈ 52.7 GB); a third tree computes to ≈ 60.9 GB — over the default — so a
pod hosting 3+ trees sizes up (`RP_DISK_GB=70`+). `N*S_clone` is the
conservative bound on purpose: the copy runs `cp --reflink=auto`, and the
artifact records only that reflink was *attempted* (`"reflink":
"attempted (auto; may have fallen back …)"` — the producer greps the flag,
it does not verify the filesystem reflinked), so real usage may be lower
when reflink takes, but sizing must assume it did not. The same artifact
carries the substrate's core walls on that box: seed→clone copy 2 s,
member-only build in a fresh clone 69 s vs 243 s from a genuinely empty
target dir, and the FA2 (`cuda,jammi-kernels/flash-attn`) leg at 122 s
over the clone.

`gpu-dev.sh` generates its own SSH key — nothing to register. Every SSH
invocation pins the connection to that key alone (`IdentitiesOnly=yes`): a
macOS ssh-agent auto-adds each session key it uses, and once it holds more
than a handful, offering all of them before this one can exhaust the pod's
`MaxAuthTries` and read a perfectly reachable pod as unreachable.

Deploy fails over across candidates and, for each one, polls for SSH up to
`RP_SSH_WAIT_SECS` (default 600) — the wall-clock budget the reachability
poll runs against. A cold host still pulling the multi-GB CUDA image can
take minutes before sshd is even up; raise this rather than losing a
healthy pod to the poll's own timeout:

```bash
RP_SSH_WAIT_SECS=900 ci/scripts/gpu-dev.sh shell a100
```

### `up` records the session write-ahead, before the reachability wait

A pod bills from the instant the deploy mutation returns an id — not from
the instant it answers SSH, which can be minutes later. `up` records the
session (pod id, arch, a host-unknown placeholder for host/port) at that
first instant, *before* the SSH-reachability wait and the driver-floor check
below it; the same record is then updated in place with the real host/port
once the pod is confirmed reachable. A failure during that wait — an
external kill that bypasses the tooling's own EXIT-trap teardown, or a
trap-time terminate call that itself silently fails — therefore still
leaves a session `ls` shows and `down` can terminate, rather than a running,
billing pod recorded nowhere and caught only by `reap`'s own late sweep.
`bootstrap_or_die`'s own failure paths print the recorded pod id and the
exact `down <session>` command to run, never a swallowed exit.

## Interactive debugging

```bash
ci/scripts/gpu-dev.sh shell a100     # sm_80 — the #277 floor (default)
ci/scripts/gpu-dev.sh shell l40s     # sm_89 (Ada) — fp8 work (#308)
ci/scripts/gpu-dev.sh shell h100     # sm_90 (Hopper)
ci/scripts/gpu-dev.sh shell a40      # sm_86 (Ampere workstation)
```

The pod boots the CUDA CI image, clones the repo, restores the cache, and drops
you into a shell in the checkout with the container's build environment already
loaded. It is terminated when you exit.

The checkout is placed on `main` unless you name a ref:

```bash
ci/scripts/gpu-dev.sh shell a40 --ref my-branch
ci/scripts/gpu-dev.sh up a100 --ref v0.47.0
```

`--ref` takes a branch, a tag or a commit. A branch or tag is proved to exist
with `git ls-remote` *before* a pod is rented, so a typo costs a second rather
than a GPU-hour plus the minutes-long wait for SSH; a commit id is the one form
no remote query can resolve, and is checked on the pod. That precheck never
prompts for anything and is bounded in time, so an unreachable or private
`RP_REPO_URL` fails fast instead of stalling every `up` on a credential prompt.
The ref is recorded in the session and shown by `ls`, because otherwise nothing
says which code a pod is running. Bootstrap fails loudly — and the pod is
terminated — if the fetch, the checkout, or the fast-forward fails, since a pod
quietly sitting on an older commit produces real results for code nobody is
reading.

`up` never moves a live pod onto a different ref: `down` the session and start it
again. A `--ref` that names a ref the live pod is **not** on is an error rather
than a silently ignored flag; naming the ref it is already on is a no-op.

### `up` refuses a session alias that already has a recorded pod

If a session alias already has a recorded pod id — even one that failed to
answer SSH (reaped, mid-reboot, or a *different* process's `up` on the same
alias winning a race) — `up` refuses outright (exit 2) rather than silently
deploying a second pod and overwriting the local record of whichever pod is
real:

```
$ ci/scripts/gpu-dev.sh up a100
::error::session 'a100' already has a recorded pod (…) that did not answer SSH.
::error::refusing to silently replace it. Inspect it: gpu-dev.sh ls
::error::once you are sure it should be replaced: gpu-dev.sh up a100 --replace
```

`--replace` overwrites only the *local* record so a new pod can be deployed
under the alias; it never terminates the old one — run `down` first if it
should be. This, together with the check below, is what closed the
2026-08-25 incident where an agent's `up`/`down` under an existing alias
terminated an unrelated stale pod.

### `down` verifies before it terminates, and confirms after

`down` never trusts the locally-recorded pod id on its own. Before issuing a
terminate, it confirms the id is **both** still present in the account's own
live pod list **and** named like one of this tooling's own pods
(`<prefix>-ttl<digits>`) — a mismatch refuses rather than acts. On a refusal
the local session record is deliberately **kept**, not forgotten: this is
exactly the ambiguous case where a follow-up `up` on the same alias most
needs to still see a recorded pod and refuse (or ask for `--replace`) rather
than deploying a third pod on top of the confusion. The pod itself, if it
still exists under a different session's name, is left running for that
session to manage. The only manual path out of a refusal is the RunPod
console.

An id that is **absent from the account entirely** is a different case, not
a refusal: it is the ordinary shape of "this pod already ended on its own"
(its own in-pod deadline, or the sweep) — the single most common way a
session's pod goes away, since `RP_TTL_HOURS`/`RP_DEV_TTL_HOURS` ceilings are
wall-clock, not idle detection (see the cost-guard section above). There is
nothing left to release, so `down` says so plainly and forgets the record —
without this, a session whose pod already self-terminated would sit stuck
until an operator remembered `up --replace` to clear it.

The **id is authoritative**; the TTL never gates release, and the check does
not look at it at all. Two earlier attempts to make the TTL part of this
check were both removed rather than patched further: matching an *exact*
recorded TTL refused to release a real `jammi-gpu-ttl72` pod when the
session's meta predated TTL tracking, and the
`RP_TTL_HOURS=<H>` override this repo tried next as the recovery path was
found **inert on every input** — the session meta is always loaded *before*
any override is read, so it either got clobbered or forced empty regardless
of what was set on the command line. RunPod pod ids are
globally unique, so a name shaped like this tooling's own naming convention,
on the exact id this session recorded, is already sufficient; the specific
number never added a real safety margin.

`down` also confirms *after* terminating: `rp_terminate` itself throws its
response away (it doubles as `rp_cleanup`'s best-effort EXIT-trap teardown,
where a network hiccup must never turn a normal shell exit into a hard
failure), so `down` re-queries the account and only forgets the local record
once the id is confirmed **absent**. A pod still present after the
terminate call — a rejected mutation, most likely — keeps its local record
and exits with a message to retry `down`, rather than silently leaking the
pod while also destroying the only record that pointed at it.

This confirmation (`rp_pod_gone` in `runpod_lib.sh`) assumes a successful
`podTerminate` removes the pod's id from `myself.pods` promptly — the same
account-query shape `rp_pod_verify` and `rp_sweep` already read, but not yet
verified live for this specific before/after transition. Confirmed on first
live use; if RunPod instead retains a terminated pod in that list for some
period (e.g. under a different `desiredStatus`), `rp_pod_gone` would read a
just-succeeded terminate as unconfirmed and `down` would report "not
confirmed" for a pod that in fact already ended — a false alarm asking for
an unnecessary retry, not a leak.

## Reproducing the shipped runtime image

The **runtime** image (e.g. for the uid-65532 JIT-cache case in #305) is not the
toolchain image and carries none of its tools:

```bash
RP_IMAGE=nvidia/cuda:12.6.3-runtime-ubi8 ci/scripts/gpu-dev.sh shell a100
```

That image ships no toolchain and **no git**, so the pod gets no checkout at all
and you land in `/root`. This is the one case where a pod is deliberately on no
ref, and it is reported as such: the banner reads `<none>` rather than naming a
ref that is not there, and so does `ls` for a session started with `up`. Naming
a `--ref` with such an image is an error — the request cannot be honoured, and a
pod that quietly ignored it would be exactly the failure `--ref` exists to
remove. Every other bootstrap failure stays fatal and takes the pod with it.

## Long-running work

```bash
ci/scripts/gpu-dev.sh up a100                     # session survives disconnect
ci/scripts/gpu-dev.sh push a100                   # send uncommitted work
ci/scripts/gpu-dev.sh run a100 cargo test -p jammi-ai --features cuda,live-gpu-tests
ci/scripts/gpu-dev.sh logs a100                   # follow it (Ctrl-C is safe)
ci/scripts/gpu-dev.sh wait-job a100               # BLOCK until it ends (or --timeout expires)
ci/scripts/gpu-dev.sh attach a100                 # shell in, from any terminal
ci/scripts/gpu-dev.sh pull a100 target/nextest    # bring artifacts back
ci/scripts/gpu-dev.sh down a100                   # terminate
ci/scripts/gpu-dev.sh ls                          # what's still running
ci/scripts/gpu-dev.sh reap                        # kill anything orphaned
```

`run` launches under tmux and returns immediately, so the job outlives both the
command and your SSH connection. There is exactly one job per **tree** (see
below), not per pod: `run` kills that tree's own `jammi-<tree>` tmux session
before starting the next one, so a long-lived job — a server, say — occupies
that tree's slot until something displaces it. A different tree's job is
untouched.

Sessions are named after the arch. `RP_SESSION` names one explicitly, and is
needed only on `up` and `shell`, the two verbs that take an *arch* where the rest
take a *session*; on the rest, an exported `RP_SESSION` that disagrees with an
explicit positional session argument REFUSES (exit 2, naming both) rather than
silently picking one — worth knowing before you export it in a shell you keep
around, rather than inline on the one command that needs it. The worked form is
in [dev-gpu-recipes.md](dev-gpu-recipes.md).

### Trees — more than one checkout per pod

A pod can host more than one checkout: `--tree <name>` on
`attach`/`run`/`logs`/`push`/`pull`/`wait-job`/`target` selects a plain
directory (`/root/trees/<name>`; the default `--tree jammi-ai` is the
bootstrap checkout at `/root/jammi-ai`), never a git worktree — a worktree
add fails on the checked-out ref, and a shared `.git` couples trees that must
be able to diverge independently. **A tree is populated ONLY by `push
--tree <name>`** (rsync); `gpu-dev.sh target <session> <name>` does NOT
create or populate the tree — it clones the pod's own build-substrate seed
into a wholly disjoint `CARGO_TARGET_DIR` (`/root/target-<name>`, a build
OUTPUT directory), for a tree that must already exist. `--with-cutlass`
additionally provisions the CUTLASS submodule (a `cp -a` copy from
`/root/jammi-ai`'s own initialised submodule) into that tree, and REFUSES
against a tree that has never been pushed ("tree source dir does not exist —
push to it first"). Each tree gets its own job script/log
(`<tree>/.jammi-job.sh`, `<tree>/.jammi.log`) and its own tmux session
(`jammi-<tree>`), so two trees' `run` jobs never collide.

`push` deliberately excludes `target/` — your host build output is the wrong
architecture and would poison the pod's — along with `.git`, `.venv*`,
`.claude`, `.sccache`, `.gpu-pull`, `scratchpad`, and the CUTLASS submodule
(provisioned separately, never rsync'd; see `target --with-cutlass` above).

`push` provisions its own tree's PARENT directory (`/root/trees` for any
non-default tree) before it rsyncs — rsync itself creates only the LAST path
component of its own destination, so the very first push against a name no
session has ever pushed before would otherwise fail outright on a fresh pod
(`rsync: mkdir "/root/trees/<name>" failed: No such file or directory`).
This runs unconditionally, every push, and is a no-op once the parent
already exists.

**`push`/`run`/`target` act on YOUR OWN checkout, not `$PWD`.** `REPO_ROOT`
is resolved from the SCRIPT's own on-disk location, never from the caller's
current directory — on a laptop with more than one checkout of this repo (a
multi-worktree swarm), always invoke the copy INSIDE the tree you mean to
act on. These three verbs refuse (exit 2, naming both paths) when the
current directory's own git toplevel disagrees with that location;
`RP_ALLOW_ROOT_MISMATCH=1` overrides for deliberate cross-tree use (e.g. one
tree's helper acting on another tree's already-up pod session on purpose).

**`--ref` and `push` are alternatives, not partners.** `push` is
`rsync -azc --no-times --delete` (the excludes above), so it overwrites the
working tree while leaving the pod's git metadata pointing at whatever was
checked out: the pod then reports HEAD on one ref while holding the contents
of another, and every git command on the pod answers about the wrong thing.
Use `--ref` for the modes that do not push — a shell on a branch, an editor
session, a job run straight from a pushed branch — and leave the pod on
`main` when the push loop is what moves your code.

`push` also writes `<tree>/.jammi-push-stamp.json` — the laptop's HEAD, a
sha256 of `git status --porcelain`, a sha256 of `git diff HEAD`, and a
sha256 over the sorted (path, mode, content-sha256) manifest of exactly what
the SAME exclude set would push (computed locally against an empty temp
directory, so it is deterministic regardless of the pod's current state).
This is **iteration provenance only** — a human debugging a live session can
tell what a pod actually received. It changes nothing about
`check_cuda_run_artifacts.py`'s `git_sha` rule: a **committed** artifact
still requires a pushed sha (a commit reachable from a remote branch), never
a push stamp.

### The timing lock — one exclusive build slot per pod

`run --timing` and the automatic seed build both acquire a single
pod-wide `flock` (`/root/.jammi-timing.lock`) INSIDE their own detached tmux
pane — the flock's lifetime is then the job's lifetime, not the short-lived
SSH invocation that launched tmux and returned immediately. A conflicting
`run --timing` (or another timing-sensitive producer, e.g.
`ci/scripts/perf/pod_build_timings.sh`) refuses immediately with exit `75`
rather than queuing silently, naming the current holder from the lock's own
holder file. The lock is **kernel-owned**: it dies the instant its holding
process exits or is killed, so there is no stale-lock state to clean up and
nothing to "steal" — an earlier rename-based scheme this replaced could be
raced into a double-acquire under a scheduling gap; `flock` removes the gap
entirely. Note the flip side, verified directly: `flock file command` forks
to run `command`, and a POSIX `flock()` lock is bound to the OPEN FILE
DESCRIPTION, which `fork()` shares — so killing ONLY the `flock` process
itself, leaving its child running, does **not** free the lock (the child
still holds the inherited fd). The realistic "holder dies" shape this
tooling relies on is `tmux kill-session`, which SIGHUPs the whole pane
**process group** at once (`run` does exactly this before starting the next
job) — killing the full tree frees the lock; killing just the wrapper does
not. Do not `nohup`/`&` a daemon out from under a lock-guarded job expecting
the lock to release while that daemon keeps running.

## Cost guard

**The EXIT trap is best-effort and must never be the only thing stopping the
meter.** A SIGKILLed process never runs it — a cancelled GitHub run, a job
timeout, a dropped laptop. On 2026-07-24 someone toggled the `run-gpu` label off
and on to re-run the gate, 71 seconds after the first run started. The
concurrency group cancelled that run while it was still waiting for its pod's SSH
to come up. The trap never fired, and the A100 it had just rented ran for seven
days, consuming ~$187.

Three guards, in order of when they act:

1. **The workflows never cancel a run that rents hardware.** Concurrency on both
   `gpu-prove.yml` and `_gpu-prove-gate.yml` sits on the *job* (so a run whose
   job is skipped by the label gate never enters the group) and sets
   `cancel-in-progress: false`. Superseded runs queue instead of dying mid-rent.

2. **A deadline armed inside the pod's own entrypoint**, before the container
   does anything else — including its package install, which reaches the network
   and could hang. Every pod self-terminates after `RP_TTL_HOURS`. The default
   depends on the caller: `runpod_lib.sh` itself defaults to 8h (`shell`, and
   any other throwaway pod); the CI prove lane sets its own 3h; `gpu-dev.sh up`
   alone raises the default to `RP_DEV_TTL_HOURS` (72h) when `RP_TTL_HOURS` is
   not set explicitly, because a dev session someone is actively using is
   meant to survive a workday, not die at the throwaway-pod default (an 8h
   ceiling killed every dev pod overnight on 2026-08-25). An explicit
   `RP_TTL_HOURS` always wins over either default. It deliberately is *not*
   installed over SSH: the gap between "pod rented" and "pod reachable" is
   minutes long, and that gap is where the orphan above was created.

   It terminates via `runpodctl remove pod $RUNPOD_POD_ID`. RunPod special-cases
   self-removal, so this succeeds in our custom image with no config file and
   without us placing any key on the pod — verified on an A40, along with the
   deadline firing and the pod staying gone. Note RunPod injects its own
   `RUNPOD_API_KEY` into every pod's environment; it does not grant account-wide
   access (`runpodctl get pod` returns `Unauthorized`).

   There is **no `kill 1` fallback**, because it was measured to do nothing:
   PID 1 in a PID namespace ignores signals it has no handler for, including
   SIGKILL, and the pod carried on RUNNING and billing at full rate. Instead the
   watchdog retries the removal, since the only remaining failure is no network
   at deadline time.

3. **A sweep**, for a pod whose container never got far enough to arm itself:

   ```bash
   ci/scripts/gpu-dev.sh reap        # honour each pod's own deadline
   ci/scripts/gpu-dev.sh reap 2      # force-reap everything older than 2h
   ```

   Each pod's deadline travels in its name (`jammi-gpu-ttl<H>`), so a sweep
   honours *that pod's* limit rather than imposing its own — without this, a CI
   prove run's 3 h sweep would reap your 8 h dev session. Only pods named
   `jammi-gpu*` are ever touched. `.github/workflows/gpu-reap.yml` runs it every
   6 h independently of the prove lane, because a backstop that only works when
   the thing it backs up is healthy is not a backstop.

   Every ambiguity resolves toward terminating: unreadable telemetry, an
   unparseable deadline, or a stopped pod is swept. And a sweep that cannot
   *reach* RunPod fails loudly rather than reporting "nothing to clean up".

   **There is no way to pause the sweep for a single pod.** RunPod's pod-edit
   mutation has no `name` field (and no rename capability at all) — the
   sweep's only account-visible per-pod signal is the name it was deployed
   with, which is immutable after deploy, so a "hold this one pod" marker is
   not something this tooling can implement. A running measurement is
   protected only by its own TTL: rent with `RP_TTL_HOURS`/`RP_DEV_TTL_HOURS`
   set to at least the job's expected length up front, rather than relying on
   pausing the sweep partway through.

Guard 2 needs the network at deadline time; guard 3 needs this repo's CI to be
running. They fail for unrelated reasons, which is the point of having both.

Guards 2 and 3 are **wall-clock ceilings, not idle detection** — a pod you stop
using bills until its deadline. Run `down` when you're finished; the guards are
for the times something kills the process before you can.

Lower the ceiling when you know the work is short, or raise it past `up`'s own
72h dev default for something that genuinely needs longer:

```bash
RP_TTL_HOURS=2  ci/scripts/gpu-dev.sh up a100   # a quick check
RP_TTL_HOURS=96 ci/scripts/gpu-dev.sh up a100   # a longer measurement
```

### `RP_TIMEOUT` is not a fourth guard

`RP_TIMEOUT` (default 3000s) wraps the `timeout … bash -s` that `rp_run_remote`
sends over SSH, so it bounds *that SSH invocation*: bootstrap, and the CI prove
lane's build-and-test script — which is what lets `gpu-prove.yml` reason about
its own job timeout.

It does **not** bound a running job. `run` uses `rp_run_remote` only to launch a
detached tmux session; tmux daemonizes, the invocation returns in under a second,
and the job it started is already outside the timeout's reach. It then runs — and
bills — until it finishes or the pod's deadline fires. Lowering `RP_TIMEOUT`
protects nothing about a `run` job; `RP_TTL_HOURS` and the sweep are what stop
one.

`ci/scripts/runpod_gpu_prove.sh` exports its own `RP_TIMEOUT` (default 6000s)
rather than relying on `runpod_lib.sh`'s 3000s default — the prove lane's own
budget, never shared with `run`/`shell`/`gpu-perf-ab.sh`, which still see the
library default. `check_gpu_prove_timings.py`'s R3 re-derives the floor this
value must clear from COMMITTED evidence
(`ci/artifacts/gpu-prove-timings/*.json`) on every CI run: `RP_TIMEOUT >= 1.5 ×
the largest HEALTHY leg's wall` AND `RP_TIMEOUT >= that wall + 3 × RP_INACTIVITY`
— raising either the healthy walls or `RP_INACTIVITY` tightens this floor, and
the gate goes vacuously RED (never a silent pass) if any shipped arch has zero
healthy evidence at all.

### `RP_INACTIVITY` — the hang detector `RP_TIMEOUT` cannot be

`RP_TIMEOUT` bounds total wall time; it cannot tell "busy and slow" from "dead
and silent" — a genuinely hung leg pays the FULL budget before `timeout` ever
fires. `rp_run_remote_watched` (`runpod_lib.sh`, used by
`runpod_gpu_prove.sh` only) layers an inactivity watchdog on top: `RP_INACTIVITY`
seconds (default 900 — derived from D5 run 33674156137's largest healthy
in-window silence, 285.2s on sm_90 during the repository clone, × R2's own
3x margin, rounded up to the next 300s step; see `runpod_lib.sh`'s own
setter comment for the full derivation) of silent remote stdout+stderr kills
the ssh session and returns 76, well before `RP_TIMEOUT` would ever have
expired.
`check_gpu_prove_timings.py`'s R2 re-demands `RP_INACTIVITY >= 3 ×` the largest
silent gap any healthy (or `slow-host`-disposed) leg has shown, on every run —
the same "re-checked, not one-time-derived" discipline R3 applies to
`RP_TIMEOUT`.

## Verbs that deliberately do not exist

Two are absent that a reader will look for. Both were considered and refused;
neither is a gap waiting to be filled.

### No `stop` — halt the job, keep the pod

`tmux kill-session` is not a process-tree kill. It destroys the pane and SIGHUPs
its foreground process group, so anything that `setsid`'d away from that group,
or is wedged in a CUDA ioctl, survives — still holding the GPU, on a pod that now
looks idle. A verb cannot honestly be called `stop` when its failure mode is a
live process nobody is watching.

It would also **save nothing**. The pod bills at its full rate whether a job runs
or not; the guards above are wall-clock ceilings, not idle detection. A `stop`
sitting next to `down` in the help output is exactly what someone watching spend
reaches for — and it would take their money while looking like it saved it.

What exists instead: `run` the next thing (it replaces the job), `attach` and
Ctrl-C to end it from its own terminal, or `down` — the only one that stops the
meter.

### No `refresh` — move a live pod's checkout to another ref

Moving a live checkout means `git checkout <ref>` against a working tree that may
be dirty, and **a dirty `git checkout` does not reliably fail**. It refuses only
when a modified file differs between the two refs; when it does not, the checkout
succeeds and carries the modification across. The pod is then on a tree matching
*neither* ref, while the session record reports the new one with full confidence
and `ls` repeats it. That is precisely the silent-wrong-checkout failure `--ref`
exists to remove, so a verb that reintroduces it is not a convenience.

`up` therefore refuses a `--ref` against a live pod rather than acting on it. The
answer is `down` then `up <arch> --ref <ref>`: a boot, for a tree you can trust.

## Automated proof — CI

The `gpu-prove` workflow (`_gpu-prove-gate.yml`) runs `grpc_embedding_gpu` +
`gpu_capability` on a real A100 via the same shared primitive
(`ci/scripts/runpod_lib.sh`), gating every CUDA release (build → prove →
promote). Trigger it per-PR with the `run-gpu` label, or it runs nightly. CI pods
are always throwaway and always terminate.

**Standing operator cost (R5, `check_gpu_prove_timings.py`):** proof surface ==
shipped surface is enforced by fingerprinting the exact `(crate, kind) →
features` pairs `runpod_gpu_prove.sh` proves (`ci/scripts/prove_surface.py`'s
`expected_id`) and demanding a fresh, matching, healthy artifact per shipped
arch. That fingerprint — and therefore this gate — moves the moment: a lane
feature is added to or removed from `ci/release-feature-manifest.json`'s
`cu12-tarball` lane; a `prove_lane.crates.<c>.prove_only` entry changes; or a
crate starts or stops declaring a feature the lane already carries. None of
those are prove-lane edits — they can land in an ordinary PR — but each one
reds this gate until a fresh 4-pod `gpu-prove.yml` dispatch lands new
`ci/artifacts/gpu-prove-timings/*.json` evidence for every arch. A waiver row
in `ci/scripts/gpu_prove_timings_allowlist.txt` is for a genuinely reviewed,
time-boxed exception on ONE arch — never a standing substitute for that
dispatch.

### Cross-pod seed cache — not yet built

The seed/clone substrate above is per-pod: a second pod builds its own seed
from scratch. A cross-pod cache (a laptop-minted presigned PUT of a seed
tarball to a read-only object-store bucket, so a NEW pod can rehydrate rather
than rebuild) is **phase 2 of this unit, not in this PR**, and is blocked on
a user action: creating the bucket (`jammi-seed-cache`) and a read-only
access token. Nothing in this tooling reads or writes that bucket today.

## Notes

- **A100 capacity on RunPod is intermittent** — deployment fails over across
  cloud tiers and PCIe/SXM variants; if all are exhausted it exits `75`. Retry.
- Pods below NVIDIA driver r560 are rejected and skipped: they cannot JIT the
  image's CUDA 12.6 PTX, so every model load would fail the #304 startup floor.
- If a run is killed uncleanly, `gpu-dev.sh reap` terminates the straggler;
  `gpu-dev.sh ls` shows only sessions this machine started, so it will not see a
  pod orphaned by a CI run or another checkout.
