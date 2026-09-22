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

Compilation is the real cost on a fresh pod, and `gpu-dev.sh` pays it at most
once per pod. Right after bootstrap, `up`/`shell` kick off a
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

sccache does not do this job: measured live on a
real pod (fresh `CARGO_TARGET_DIR` each leg, `cargo build --release -p
jammi-bench --features cuda`), sccache gave **zero cross-target-dir cache
reuse** for rustc units — every populate-then-reuse pair against a fresh
target dir re-missed everything sccache had just written — while adding
**+33% to +37.5% wall clock** to every build that ran it (344s wrapper-off
vs 457-473s wrapper-on: low end 344→457s is (457-344)/344 = +32.8% ≈ +33%;
high end 344→473s is (473-344)/344 = +37.5%).
The wrapper is off pod-wide
(`CARGO_BUILD_RUSTC_WRAPPER=` in `/root/.jammi_env`, every shell sources it).

The cargo **registry** is deliberately not cached either. It looks like the
expensive part, since the CI image wipes `/usr/local/cargo/registry`, but a
cold `cargo fetch --locked` measures **9s for 868 crates** on a RunPod host
(same measurement session as the sccache figures above; no committed artifact
records this particular number) — datacenter bandwidth makes it free.

Disk sizing (`RP_DISK_GB`): `>= 25` (base) `+ S_src + S_seed + N*S_clone`
(one clone per tree the pod hosts). Measured by this formula's producer,
`ci/scripts/perf/pod_build_timings.sh`, on an A100-SXM4 secure-cloud pod
(decimal GB): `S_src` ≈ 3.6 GB (the checkout, `.git`
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
ci/scripts/gpu-dev.sh shell a100     # sm_80 — the supported floor (default)
ci/scripts/gpu-dev.sh shell l40s     # sm_89 (Ada) — fp8 work
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
should be. This, together with the check below, is what keeps an `up`/`down`
under an existing alias from terminating an unrelated stale pod.

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
not look at it at all. A TTL check cannot be made sound: matching an
*exact* recorded TTL refuses to release a real pod whose session meta
carries no TTL, and an `RP_TTL_HOURS=<H>` override cannot rescue it — the
session meta is always loaded *before* any override is read. RunPod pod ids are
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

The **runtime** image (e.g. for the uid-65532 JIT-cache case) is not the
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

## NCCL is part of the CUDA link set

`jammi-ai`'s `cuda` feature includes `candle-core/nccl`, and cudarc's build
script emits `cargo:rustc-link-lib=dylib=nccl` for it. Two consequences follow,
and they land on different machines.

**At build time**, every host that compiles anything with `--features cuda`
needs `nccl.h` and the `libnccl.so` development symlink on disk — a pod, a
laptop with a local toolkit, and CI alike. The CUDA toolkit package does not
carry NCCL; the CUDA CI image installs it separately
(`.docker/ci-cuda.Dockerfile`), and `ci.yml`'s `flash-attn-compile` job asserts
it before its first clippy step, so a run on an image without the package reds
on a one-line preflight instead of on a `cannot find -lnccl` deep in a link.

**At run time**, the binary carries `DT_NEEDED libnccl.so.2` whether or not a
job ever forms a communicator — the loader resolves it before `main`. Each
CUDA-shipping artifact answers for that soname its own way:

| artifact | how `libnccl.so.2` gets there |
| --- | --- |
| `jammi-server-cu12` tarball | staged into the tarball's `lib/` by `ci/scripts/bundle_cuda_libs.sh`'s derivation: `libnccl.so.2` is a member of the binary's own transitive `DT_NEEDED` closure (resolved under the CUDA 12.6 toolkit then `/usr/lib64`, `/usr/lib64` being where this image's `libnccl` RPM installs), so it is staged the same way every other closure member is — no name is listed by hand. `release-binaries.yml`'s `server-cu12-build` step also asserts the real loader resolves it (and every other bundled member) from `lib/`, not from a host copy |
| `jammi-server-cu12` wheel | the `nvidia-nccl-cu12` dependency; the console script puts `nvidia/nccl/lib/` on `LD_LIBRARY_PATH`, and `verify_link_set.py` fails the build if a needed library is unclassified, or if it extracts no `DT_NEEDED` entries at all |
| `jammi-ai-server` CUDA image | nothing to do: the `nvidia/cuda:12.6.3-runtime-ubi8` base installs `libnccl-2.23.4-1+cuda12.6` itself (its own image config's `NV_LIBNCCL_PACKAGE`) — the same build the CI image and the wheel pin |

The host NVIDIA driver's own libraries (`libcuda.so.1`, `libnvidia-*`) are the
opposite case: never bundled anywhere, because they must match the driver the
GPU is running.

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
nothing to "steal" — a rename-based lock can be raced into a double-acquire
under a scheduling gap; `flock` has no such gap. Note the flip side, verified directly: `flock file command` forks
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
timeout, a dropped laptop. A run cancelled while it is still waiting for its
pod's SSH to come up never fires the trap, and the GPU it has just rented keeps
billing until something else stops it — an A100 orphaned this way has run for
seven days, ~$187.

Three guards, in order of when they act:

1. **The workflow never cancels a run that rents hardware.** Concurrency on
   `gpu-prove.yml` sits on the *job* (so a run whose job is skipped by the
   label gate never enters the group) and sets `cancel-in-progress: false`.
   Superseded runs queue instead of dying mid-rent. `_gpu-proof-required.yml`
   — the reusable EVERY release-publishing workflow calls to read the
   commit's already-recorded verdict — never rents anything and carries no
   concurrency group at all; there is nothing there for a superseded run to
   orphan.

2. **A deadline armed inside the pod's own entrypoint**, before the container
   does anything else — including its package install, which reaches the network
   and could hang. Every pod self-terminates after `RP_TTL_HOURS`. The default
   depends on the caller: `runpod_lib.sh` itself defaults to 8h (`shell`, and
   any other throwaway pod); the CI prove lane sets its own 3h; `gpu-dev.sh up`
   alone raises the default to `RP_DEV_TTL_HOURS` (72h) when `RP_TTL_HOURS` is
   not set explicitly, because a dev session someone is actively using is
   meant to survive a workday, not die at the throwaway-pod default (an 8h
   ceiling kills a dev pod overnight). An explicit
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
budget, never shared with `run`/`shell`, which still see the
library default. The budget clears `1.5 × the slowest healthy leg's wall` and
`that wall + 3 × RP_INACTIVITY`; a leg that grows past either moves it.

### `RP_INACTIVITY` — the hang detector `RP_TIMEOUT` cannot be

`RP_TIMEOUT` bounds total wall time; it cannot tell "busy and slow" from "dead
and silent" — a genuinely hung leg pays the FULL budget before `timeout` ever
fires. `rp_run_remote_watched` (`runpod_lib.sh`, used by
`runpod_gpu_prove.sh` only) layers an inactivity watchdog on top: `RP_INACTIVITY`
seconds (default 900: three times the longest silence a healthy leg has shown,
the repository clone at under five minutes on sm_90, rounded up to the next
300s step) of silent remote stdout+stderr kills the ssh session and returns
76, well before `RP_TIMEOUT` would ever have expired.

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

## GPU proof and the release gate

`gpu-prove.yml` runs `grpc_embedding_gpu` + `gpu_capability` on a real device
(matrixed over every shipped CUDA arch) via the same shared primitive
(`ci/scripts/runpod_lib.sh`). It is **never in the critical path of an
automated workflow**: its only triggers are the `run-gpu` PR label, the
nightly cron, and manual dispatch — it never fires on a push or a tag, and no
other workflow may `uses:` it (`ci/scripts/check_gpu_prove_once.py` pins
both by name).

EVERY release-publishing workflow — CUDA and non-CUDA alike: the server
image, the release binaries, the cu12 wheel, AND crates.io, npm, the native
wheel, the pure-Python client, the CPU server wheel — gates its promotion on
the **summary** of a prove run already on record for the commit it is
promoting, not on a fresh rental of its own: `_gpu-proof-required.yml` calls
`ci/scripts/gpu_prove_verdict.py`, which asks the GitHub API whether every
shipped arch's job at that commit's `head_sha` most recently concluded
`success`. This is a CHECK-ONCE, FAIL-LOUD lookup — no poll, no deadline, no
grace window. An in-progress prove run at that commit is invisible to the
check; it is never waited on. Nothing here starts a prove run — a red or
missing verdict fails EVERY release workflow immediately, with the exact
remedy (`gh workflow run gpu-prove.yml --ref <ref>`, or `gh run rerun
<run_id> --failed` for one red leg) printed in the job log.

**Release order:** before tagging, dispatch `gpu-prove.yml` on the commit to
be released (`--ref main` at the tip, or on the pushed tag once it exists);
once every shipped arch is green, push `v*` and `py-v*` together, on the
same commit. A tag push on a commit whose prove is not ALREADY green
publishes NOTHING — every publisher fails immediately, by design: prove
first, then tag, because a tag push commits the version number. See
MAINTAINER-GUIDE.md's release runbook for the exact sequencing.

CI pods are always throwaway and always terminate.

### Cross-pod seed cache — not yet built

The seed/clone substrate above is per-pod: a second pod builds its own seed
from scratch. A cross-pod cache (a laptop-minted presigned PUT of a seed
tarball to a read-only object-store bucket, so a NEW pod can rehydrate rather
than rebuild) is not built: it needs a bucket (`jammi-seed-cache`) and a
read-only access token, and nothing in this tooling reads or writes that bucket.

## The gang leg — two GPUs in one pod

`gpu-gang.yml` rents ONE pod holding TWO A100s and runs the distributed
fine-tune (gang) tests on it, through `ci/scripts/runpod_gpu_gang.sh` and the
same shared primitive every other lane uses (`ci/scripts/runpod_lib.sh`). It is
the only lane where a multi-device collective runs on real hardware; every
other GPU lane rents a single device.

**What makes the pod different.** `RP_GPU_COUNT` (default 1 for every other
lane) is 2 here — the one caller that moves it. At a count above 1
`rp_deploy_arch` tries the SXM4 candidates before the PCIe ones: a 2-GPU pod
provisioned on `A100-SXM4-80GB` SECURE while the PCIe pool reported no 2-GPU
capacity at all. Both pairs stay in the list, so a multi-GPU rental only
reorders the capacity search, never narrows it.

**What it proves.** The gang tests in `jammi-ai`'s `gpu_capability` target,
selected by the `gang_` name filter the driver owns. A name filter that matches
zero tests exits 0 with "running 0 tests" — the driver reads that as a
FAILURE, by name, and equally refuses a run that wrote no artifact. A leg with
no test is a leg with no proof.

**Triggers.** The `run-gang` PR label and manual dispatch — never a push, never
`workflow_call`, and no workflow may `uses:` it. The never-in-an-automated-path
doctrine is `gpu-prove.yml`'s own, and
`ci/scripts/check_gpu_prove_once.py`'s P7 rule pins it for every renting lane —
over a driver set *derived* from `runpod_lib.sh`'s deploy closure, with
`PAID_POD_LANE_TABLE` (driver script -> its one workflow) as the completeness
assertion over that set. Nothing about a release depends on this lane; the
release verdict is the prove lane's.

**Cost bound (human-approved).** Two bounds, each with the mechanism that
enforces it. The rented pod is not the only thing that bills: `rp_deploy_live`
walks the `a100` candidate list (4 entries) and terminates a pod that is not
SSH-reachable within `RP_SSH_WAIT_SECS` before trying the next, so the *search*
bills too.

- **Terminate-succeeds** — the ordinary path, every `rp_terminate` takes. The
  driver pins `RP_SSH_WAIT_SECS=300` (half the library default) and the workflow
  pins `MAX_ATTEMPTS: "1"`, so there is exactly one search; the winning pod then
  bills to `RP_TTL_HOURS=1`, baked into the pod's own entrypoint so a SIGKILLed
  runner cannot outlive it. `4 x 300 s x $3.18/h + 1 h x $3.18/h = $4.24` a run.
- **Sweep-only** — every `rp_terminate` call fails, the case `runpod_lib.sh`'s
  own header opens with. Nothing is torn down early, each pod bills to its
  baked-in TTL, and the only remaining enforcers are that TTL and `gpu-reap.yml`'s
  `rp_sweep`: `(4 + 1) x 1 h x $3.18/h = $15.90`.

`$3.18/h` is the rate the SECURE 2-GPU `A100-SXM4-80GB` pod was rented at. The
COMMUNITY 2-GPU rate is unmeasured — nothing has priced one — so neither figure
covers a COMMUNITY landing. `ci/scripts/test_gpu_gang_lane.sh` re-derives the
first bound from the candidate list, the driver's own two values and the
workflow's `MAX_ATTEMPTS`, and fails if the printed figure and the mechanism
disagree.

Inside the TTL hour, the shared `RP_TIMEOUT` default (50m, owned by
`runpod_lib.sh` — this lane declares no second one) is what cuts first, with the
cut group named. Whether a cold `cuda,flash-attn` build plus the gang tests fits
inside that hour is NOT established — nothing has measured it. A budget cut is
therefore a cost decision for a human (raise the bound deliberately), never
something the script raises on its own.

**Exit codes** (the workflow annotates each one separately, so a capacity night
never reads as a code regression):

- `0` — every gating group passed.
- `75` — no 2-GPU capacity. RED, with no retry: `MAX_ATTEMPTS` is `1`, because a
  second attempt is a second walk of the candidate list and doubles the search
  term of the first bound above. A leg with no capacity proved nothing.
- `76` — the inactivity watchdog killed a hang with a gating group unresolved.
  A hung collective is exactly what this lane exists to surface.
- `77` — wrong tree: the pod's own `PROVE_SHA` disagreed with the commit the
  run expected.
- `97` — the rented pod is not the device the leg asked for (fewer GPUs than
  requested, or the wrong compute capability). Refused before anything is
  built.
- `124` — budget cut with a gating group unresolved.

**The artifact.** The gang tests write their evidence into
`JAMMI_GANG_ARTIFACT_DIR` on the pod; the driver pulls that directory back
before the EXIT trap tears the pod down (the pod is the only place it exists)
and the workflow uploads it. The pod-leg's own producer is
`gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1`
(`crates/jammi-ai/tests/gpu_capability/gang_pod_leg.rs`, selected by the
`gang_` filter like every other test in this suite): it trains a real
two-rank gang over `Nccl` twice from the same seed (a reproducible adapter
digest pair) against a W=1 reference at double the per-rank batch (the
per-epoch training-loss delta), then writes the ONE `gang`-kind artifact
into that directory itself — nothing else on this tree does. A human
reviews it and commits it under `crates/jammi-kernels/artifacts/cuda-runs/`,
where `ci/scripts/check_cuda_run_artifacts.py`'s `gang` kind is its schema
gate. That schema requires the topology the run actually had (`world`, the
collective, and one device per rank), the same-seed digest pair, the
measured per-step loss delta, and the epsilon it is read against — with
epsilon's own derivation and the commit it was registered at. An epsilon
chosen after seeing the delta it excuses is not a tolerance, and the gate
refuses it by name.

**A failing run is representable.** The artifact carries the leg's own
`verdict`, exactly `pass` or `fail`. A `fail` is *admitted* with its deltas and
digests as measured — that record is the whole value of a non-reproducible run —
and owes a `reason` naming what failed plus a top-level `status` that is not
`GREEN`. A `pass` is a claim, so on a `pass` the worst measured delta must be
within epsilon, and the same-seed digest pair must be *equal* at `world` 2, the
one regime a spike measured byte-identical (candle 0.11's LoRA-shaped
forward/backward/SGD across A100s, no env pins). Above `world` 2 the pair is
recorded and not asserted: nothing has established what byte-identity should
mean for a reduction whose NCCL pin set is untested there.

**Where epsilon has to sit in history.** The gate reads the registration commit
against the artifact's *evidence anchor*: `git_sha` when that is an ancestor of
`HEAD` — the tree the run actually measured — otherwise `merged_as` as the
rescue, when the artifact carries one that is an ancestor of `HEAD` (a measured
tip whose landing commit rewrote it). A `merged_as` stamped beside a `git_sha`
that is still in this history changes nothing: it names a later commit, and
ordering against it would admit an epsilon registered in the measured commit
itself. When `merged_as` IS the anchor, epsilon is ordered against that
**landing** commit, never against the (now-unreachable) commit where the
measurement itself ran — the measured tip's own commit has no content left in
this history to order anything against. The registration commit must be an
ancestor of `HEAD` and a **strict** ancestor of that anchor; an artifact with
neither anchor in this history fails, naming both. What that
asks of whoever runs the leg: commit epsilon on its own, **before** the commit
you measure with, on the same branch. Landing that branch by a merge commit —
this repository's own merge style — keeps epsilon a strict ancestor afterwards.
A squash, or a rebase performed *after* measuring, rewrites both commits and the
artifact fails from the merge onwards, so do not rebase a measured branch:
land it, or re-measure.

## The cluster leg — two hosts, one GPU each

**Two transports, one proof.** `RP_TWO_HOST_TRANSPORT` selects which RunPod
object type carries the two-HOST NCCL bootstrap: `pods` (default) or
`cluster`. `pods` rents TWO ORDINARY pods (`POST /v2/pods`,
`rp_two_host_pod_create` — `cloud: SECURE`, `globalNetworking: true`, one
GPU each), co-located in ONE data center chosen by intersecting the pod
catalog's own per-data-center availability (`GET /v2/catalog/gpus?
include=AVAILABILITY&product=POD&count=1&cloud=SECURE`) with the data
centers RunPod's own `GET /v2/catalog/datacenters` reports `globalNetwork: true`
for — read LIVE at run time, never a hard-coded list (a snapshot verified
2026-09-16: CA-MTL-1, CA-MTL-3, EU-CZ-1, EU-FR-1, EU-NL-1, EU-RO-1, EU-SE-1,
EUR-IS-2, EUR-IS-4, OC-AU-1, US-CA-2, US-GA-2, US-IL-1, US-KS-2, US-NC-1,
US-TX-3, US-TX-4, US-WA-1). Rank is assigned by CREATION ORDER (the first
pod created is rank 0, the second rank 1), and each member DERIVES its own
`NCCL_SOCKET_IFNAME` from its Global-Networking ip at run time
(the kernel route table, `/proc/net/route`: the interface whose route
covers that ip by longest prefix — the image ships no `ip` binary) rather
than the cluster path's `ens1` literal — a member whose route table covers
no such ip refuses (97) by name, echoing `DERIVED_NCCL_IFACE=` so
the driver's own post-run proof reads which interface it actually used.
`cluster` is kept, byte-for-byte what it always was (below) — the SAME
proof over a different, near-zero-capacity rental mechanism (measured: 8 of
9 cluster creates refused `Insufficient resources` in one session). Both
transports assemble the SAME `gang` artifact (`gang.leg` stays `"cluster"`
either way — the two-HOST leg is the fact that matters downstream);
`gang.transport` (`instant-cluster` | `global-networking`) is the sub-fact
naming which mechanism actually carried the run, closed-set and required by
`check_cuda_run_artifacts.py` rule (k). Cost bound (both parts at the
measured cluster and catalog rates, `RP_TTL_HOURS=1`): `cluster` bills
`2 x $1.908/GPU/h = $3.816/h` (`1 h x $3.816/h = $3.82` terminate-succeeds;
`(1 + 6) h x $3.816/h = $26.71` sweep-only); `pods` bills
`2 x $1.59/GPU/h = $3.18/h` (`1 h x $3.18/h = $3.18` terminate-succeeds;
`(1 + 6) h x $3.18/h = $22.26` sweep-only — both pods fall under the
ORDINARY pod sweep's own name-shape match, so `gpu-reap.yml`'s SAME
6-hourly cadence is the backstop, no separate sweep primitive for this
transport).

A CLUSTER is a SEPARATE RunPod object type from a pod: member pods on one
private overlay network, created and destroyed as a unit — it is retired by
deleting the CLUSTER, never by terminating one of its member pods. This
tooling's own `_rp_cluster_payload` requests a FIXED shape, never a
caller-chosen one: exactly 2 member pods, 1 GPU each (`compute.
gpuCountPerPod=1`, `compute.podCount=2` — this tooling's own choice, not
something RunPod's schema demands; there is no parameter for any other
shape). There is
no GraphQL surface for it at all; every `rp_cluster_*` primitive in
`ci/scripts/runpod_lib.sh` goes over RunPod's REST v2 (`_rp_rest`):
`rp_cluster_create`, `rp_cluster_get`, `rp_cluster_pods` (members with
`rank`/overlay `ip`/`ssh.direct`), `rp_cluster_delete`, `rp_cluster_list`,
and `rp_cluster_sweep` (a cluster whose name carries `-ttl<H>` and whose age
exceeds `H` hours is deleted; one this sweep cannot JUDGE — no usable
`createdAt`, or a prefixed name with no parseable `-ttl<H>` — is named with
its by-id remedy (`rp_cluster_delete <id>`) and left alone while the rest
of the list is still swept, and the sweep exits 1; a failed enumeration is
`return 1` "could NOT enumerate clusters", never "nothing to reap"). `gpu-dev.sh reap` runs both
the pod sweep and `rp_cluster_sweep`; the pod sweep excludes every live
cluster member by id rather than ever calling `podTerminate` on one.

**What it proves.** `gang_nccl_two_hosts_reduce_a_known_vector`
(`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs`) is the only test body
that exercises the two-HOST NCCL bootstrap (`ncclCommInitRank`, an
out-of-band id crossing between hosts); the gang leg above proves a
two-DEVICE collective inside one pod (`ncclCommInitAll`), which cannot
exercise this bootstrap at all. Rank 0 mints the 128-byte NCCL id and writes
it to `$JAMMI_GANG_TWO_HOSTS_ID_FILE`; rank 1 reads it once it is ready
(rank 0 writes a `.tmp` file then renames, and only after `stat` reports
exactly 128 bytes). Both ranks then run the SAME assertions the pod leg's
single-process test runs (rank-ordered sum, unequal-count gather, lockstep
flags, barrier), over a real cross-host communicator instead of
`ncclCommInitAll`'s single-process one, and each writes its own
`rank-<r>.json` report into `$JAMMI_GANG_ARTIFACT_DIR`.

**The driver: `ci/scripts/runpod_gpu_cluster.sh`.** It rents a 2×1 cluster
of the part `RP_CLUSTER_GPU_TYPE` names (the workflow's `gpu_type` input;
A100 SXM4 by default, or any sm_80/86/89/90 part the driver's
`_rpc_compute_cap_for_gpu_type` maps to the compute capability the members
build for — an unmapped id is refused before anything is rented, and the
availability floor is the `min_availability` input),
waits for both members reachable (tracked by DISTINCT rank, never a raw
count — two rows both reading back as rank 0 must never satisfy readiness),
ships the id between hosts, pulls both ranks' reports, assembles the one
committed `gang` artifact (shape MEASURED from the create/get response,
never a literal), scans every carrier for the id BEFORE anything is
uploaded, and tears the cluster down on every exit arm — including a
SIGINT/SIGTERM/SIGHUP cancellation, not only a normal `exit` — via its own
cleanup trap, which runs the scan FIRST, ahead of its own (bounded) REST
calls, and destroys a dirty carrier synchronously rather than deferring to
a session-conditional cleanup. Its own workflow, `.github/workflows/
gpu-cluster.yml`, is `run-cluster` PR-label or `workflow_dispatch` only —
never a `schedule:`, and nothing else `uses:` it. As a FALLBACK — before
the driver's first real run, or if it is ever unavailable — a maintainer
may still drive the two-host test BY HAND with the primitives above:

1. Read per-data-center availability (`GET /v2/catalog/gpus?include=
   AVAILABILITY&product=CLUSTER&count=1&cloud=SECURE`) and pick a data
   center at `MEDIUM` or better — co-placement needs exactly one.
2. `rp_cluster_create <gpuTypeId> [dataCenterIds]`; poll `rp_cluster_get`/
   `rp_cluster_pods` until both members are `RUNNING` with a reachable
   `ssh.direct` (or the overlay-ip fallback through the primary), then
   probe each member with `ssh … true` until it answers
   (`rp_wait_sshd`, bounded by `RP_SSH_WAIT_SECS`): a `RUNNING`
   pod's entrypoint installs sshd after boot, so the endpoint RunPod
   reports refuses connections for a while first. Both transports run
   this probe for both members before any remote command.
3. On BOTH members, concurrently: fetch this tree at the EXACT commit under
   test (`PROVE_EXPECT_SHA`, by hash — never a branch name, which can move
   between dispatch and clone) and build the `gpu_capability` test target (only the proof needs the id,
   so neither build waits on the other; the driver's one watch loop bounds
   both by log growth within `RP_INACTIVITY` and the T-10m budget).
4. On the member running rank 0: export `JAMMI_GANG_TWO_HOSTS_RANK=0`,
   `JAMMI_GANG_TWO_HOSTS_WORLD=2`, `JAMMI_GANG_TWO_HOSTS_ID_FILE=<path>`,
   `JAMMI_GANG_ARTIFACT_DIR=<path>`, `NCCL_SOCKET_IFNAME=ens1`, and run
   `cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-cluster-tests
   --test gpu_capability gang_nccl_two_hosts -- --nocapture
   --test-threads=1`.
5. Once rank 0's id file holds exactly 128 bytes, `scp` it to the member
   running rank 1 (mode 0600; delete the local copy once the id has
   crossed). Rank 1 runs with the SAME env, `JAMMI_GANG_TWO_HOSTS_RANK=1`, and its
   script blocks between its build and its proof until that file holds
   128 bytes — the proof, never the build, waits for the id.
6. Read both `rank-<r>.json` reports back; `rp_cluster_delete` the cluster
   when done — do not rely on member self-removal alone (below).
7. Treat the id as a secret throughout: it must never appear in a
   terminal scrollback, a committed log, or a comment on this repo.

**Member self-removal is honestly unmeasured.** RunPod's REST v2 surface
reports member pods with `actions: []`, so even a successful in-pod
`runpodctl remove pod` self-termination's effect on cluster accounting is
unconfirmed for a cluster member — the driver's own `_rpc_self_remove_status`
treats a 404 on the cluster's own GET as "ok" and otherwise falls back to
its own `rp_cluster_delete` call, from its cleanup trap, on every exit arm.
The enforcers, in order: (1) the driver's own trap, (2) the cluster's own
name TTL plus `gpu-reap.yml`'s 6-hourly `rp_cluster_sweep`, (3) a human, via
the RunPod console. Cost bound, at the MEASURED `$1.908/GPU/h` cluster rate
(the catalog's own `$1.59` is the POD price, a different rate) and the
driver's own `RP_TTL_HOURS=1`: the 2×1 shape bills `2 x $1.908/GPU/h =
$3.816/h`, so (i) terminate-succeeds (the ordinary path): `1 h x $3.816/h =
$3.82` per run; (ii) sweep-only (the worst path — the trap's own delete call
fails): `(1 + 6) h x $3.816/h = $26.71`. The lane runs only on the
`run-cluster` label, <= 1 h billed per run.

**Pre-flight: REST v2 `args` reaches `bash -c`.** The cluster driver relies on
REST v2's `args` field reaching `bash -c` on `RP_IMAGE` the way the pod
path's GraphQL `dockerArgs` field does. A real, single-pod REST v2 create
shows it does: a pod (RTX A4000, SECURE) with `args: "bash -c '...'"` on
`ghcr.io/f-inverse/jammi-ai-ci-cuda:latest` — the container log printed the
exact marker that `args` command echoed (`PREFLIGHT-ARGS-OK`), followed by
`nvidia-smi -L` (`GPU 0: NVIDIA RTX A4000`) and `/sys/class/net` (`bonding_masters
eth0 lo`); the read-back `Pod.args` on a subsequent GET returned the exact
text sent. REST v2's `args` reaches `bash -c` exactly as the pod path's
GraphQL `dockerArgs` does, and the launch-time read-back refusal
(`_rpc_check_readback`) reads a real field. Still unmeasured, honestly,
because this was a single ordinary POD, never a cluster: `ens1` as a
cluster member's own overlay iface (this probe's own pod showed only
`eth0`/`lo` — no cluster overlay network), member sshd reachability on a
real cluster, and member self-removal (above).

**The artifact registry.** `ci/scripts/check_cuda_run_artifacts.py`'s `gang`
kind (rule (k)) discriminates by `gang.leg`: the pod leg's registry is
unchanged (`world`, `collective`, per-rank `device`, the same-seed digest
pair, the measured delta, epsilon). The cluster leg's own registry —
`hosts` (exactly 2), `ranks[]` (`rank`/`host`/`device`/`iface` per entry), a
`reduced_vector_digest` (a bit-exact digest, equal across both ranks on a
`pass`; NEVER conflated with the pod leg's LoRA-shaped same-seed
reproducibility pair, a different regime this leg does not measure), and
the shape/deadline it was rented at (`pod_count`, `gpu_count_per_pod`,
`ttl_hours`), with no `digests`/`per_step_loss_delta`/`epsilon` row at all —
stays on this tree. `GANG_LEG_PRODUCER_PATH` binds `gang.leg ==
"cluster"` to `producer.path == "ci/scripts/runpod_gpu_cluster.sh"` — THIS
driver is the sole writer of that leg's artifact, exactly as the pod leg's
own driver is bound to its own registry, and a leg naming any other path
(or no registered leg at all) is still refused before any shape is even
read.

**Schedule visibility (P8).** `ci/scripts/check_gpu_prove_once.py`'s P7 rule
derives its renting-closure subject set from a REVIEWED ROOT LIST —
`_rp_deploy_payload` for the pod surface, `rp_cluster_create` for the
cluster surface — so a second renting mechanism gets a table row through
the same derivation the pod legs always have the moment a real DRIVER
calls it. `rp_cluster_create`'s own real caller is `ci/scripts/
runpod_gpu_cluster.sh`, carrying its own `PAID_POD_LANE_TABLE` row
(`"gpu-cluster.yml"`) — the root's FIRST real driver, judged by P7 like any
other. Three OTHER tracked files also word-match the `rp_cluster_create`
literal and are each independently derived and cleared through the same
predicate (none is a renting driver and none is exempted for being ours):
`ci/scripts/test_check_gpu_prove_once.py` (its fixtures spell the literal),
`ci/scripts/test_runpod_cluster_lib.sh` (the mocks-only primitives
suite, which genuinely calls it) and `ci/scripts/check_gpu_prove_once.py`
itself (this very rule's own source names `rp_cluster_create` as a
`RENTING_ROOTS` string literal, so the gate self-matches its own
definition — see that file's own disclosure of this, alongside its
pre-existing `test_check_gpu_prove_once.py` self-match). P8 additionally
demands that ANY paid pod lane's `schedule:` trigger, if one is ever added,
is a reviewed `PAID_LANE_CRON_ALLOWLIST` entry naming its own never-vacuous
arm — `gpu-cluster.yml` carries no `schedule:` at all today, so this leg
adds nothing to that allowlist. Nothing about a release depends on this
leg; the release verdict is the prove lane's.

**Known-unmeasured.** This leg proves world 2 only. Whether the NCCL pin set
(`NCCL_SOCKET_IFNAME=ens1` and friends) that works at world 2 still suffices
at world >= 3 — a multi-rail/multi-NIC topology a 2-host gang cannot
exercise — is uncovered here, and stated as such rather than silently assumed.

## Notes

- **A100 capacity on RunPod is intermittent** — deployment fails over across
  cloud tiers and PCIe/SXM variants; if all are exhausted it exits `75`. Retry.
- Pods below NVIDIA driver r560 are rejected and skipped: they cannot JIT the
  image's CUDA 12.6 PTX, so every model load would fail the startup driver floor.
- If a run is killed uncleanly, `gpu-dev.sh reap` terminates the straggler;
  `gpu-dev.sh ls` shows only sessions this machine started, so it will not see a
  pod orphaned by a CI run or another checkout.
