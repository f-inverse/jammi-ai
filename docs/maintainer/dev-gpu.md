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
- **state** — the pod itself is always disposable. Anything worth keeping lives
  either in git (your working tree) or in an S3-compatible object store (the
  build substrate). Nothing durable is ever stored on the pod.

## Why no network volume is attached

RunPod network volumes are Secure-Cloud-only and locked to a single datacenter,
and can only be attached at deploy time. Attaching one would delete both failover
dimensions in `runpod_lib.sh` — cloud tier and PCIe/SXM variant — and those exist
precisely because A100 supply is intermittent. Pinning the pod to one datacenter
trades away availability for persistence we can get more cheaply.

So the pod stays free to land anywhere, and durable state is made
location-independent instead. A network volume is still the natural *backing
store* for that object storage, reached over its S3 endpoint — it is simply never
mounted on a pod.

## One-time setup

A RunPod API key (RunPod → Settings → API Keys):

```bash
mkdir -p ~/.config/runpod && printf '%s' 'YOUR_KEY' > ~/.config/runpod/key && chmod 600 ~/.config/runpod/key
```

(CI reads the same key from the `RUNPOD_API_KEY` GitHub Actions secret.)

The shared **sccache** compile cache is optional — without it everything still
works, just cold. To enable it, create a network volume plus an S3 API key
(RunPod → Settings → S3 API Keys) and write:

```bash
cat > ~/.config/runpod/s3 <<'EOF'
RP_S3_ENDPOINT=https://s3api-us-ne-1.runpod.io
RP_S3_BUCKET=<network-volume-id>
RP_S3_ACCESS_KEY_ID=user_...
RP_S3_SECRET_ACCESS_KEY=rps_...
EOF
chmod 600 ~/.config/runpod/s3
```

The volume is never attached to a pod, so its datacenter only has to serve the
S3 API — and **RunPod's documented list of S3 datacenters is not accurate**.
Probe before choosing one; a live endpoint answers `401`, a dead one `530` or
nothing:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://s3api-us-ne-1.runpod.io/
```

`RP_S3_REGION` is derived from the endpoint. RunPod signs SigV4 against the
datacenter and rejects `auto` outright, which — because `rustc-wrapper` is
sccache repo-wide — would otherwise stop every cargo command on the pod dead.
Bootstrap proves the sccache server starts against the bucket and falls back to
a local disk cache if it cannot.

The cargo **registry** is deliberately not cached. It looks like the expensive
part, since the CI image wipes `/usr/local/cargo/registry`, but a cold
`cargo fetch --locked` measures **9s for 868 crates** on a RunPod host —
datacenter bandwidth makes it free. Compilation is the real cost, and that is
what sccache holds. Measured on two separate A40 pods, `cargo build -p jammi-db`:

| pod | wall | cache hits | misses |
|-----|------|-----------|--------|
| 1 — cold, populating | 188s | 0 | 504 |
| 2 — reading pod 1's cache | 47s | 504 | 0 |

A 4× cut with a 100% hit rate across two pods that never met, which is the
property the whole ephemeral-pod design rests on. Expect a smaller ratio on a
full CUDA build — nvcc output is not rustc output — but the mechanism is what
matters: the cache survives the machine.

`gpu-dev.sh` generates its own SSH key — nothing to register.

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

### `down` verifies before it terminates

`down` never trusts the locally-recorded pod id on its own. Before issuing a
terminate, it confirms the id is **both** still present in the account's own
live pod list **and** still carries a name this session could plausibly own
— a mismatch refuses rather than acts. On a refusal the local session record
is deliberately **kept**, not forgotten: this is exactly the ambiguous case
where a follow-up `up` on the same alias most needs to still see a recorded
pod and refuse (or ask for `--replace`) rather than deploying a third pod on
top of the confusion. The pod itself, if it still exists under a different
session's name, is left running for that session to manage.

A session recorded before this repo tracked `RP_TTL_HOURS` in the session
file has no TTL to check against — `down` does not guess (a guessed "8"
against a real `jammi-gpu-ttl72` pod would refuse to release a pod this
tooling itself rented, and it would then bill its full 72h). Absence is
checked on the meta *file*, not a fallback-defaulted variable: with no
recorded TTL, `down` verifies by the pod id plus the `<prefix>-ttl<digits>`
name *shape* alone, and any refusal names the recovery command
(`RP_TTL_HOURS=<H> gpu-dev.sh down <session>`) if you know the pod's actual
TTL and want to force an exact match.

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
ci/scripts/gpu-dev.sh attach a100                 # shell in, from any terminal
ci/scripts/gpu-dev.sh pull a100 target/nextest    # bring artifacts back
ci/scripts/gpu-dev.sh down a100                   # terminate
ci/scripts/gpu-dev.sh ls                          # what's still running
ci/scripts/gpu-dev.sh reap                        # kill anything orphaned
```

`run` launches under tmux and returns immediately, so the job outlives both the
command and your SSH connection. There is exactly one job per pod: `run` kills
the `jammi` tmux session before starting the next one, so a long-lived job — a
server, say — occupies the pod's only slot until something displaces it.

Sessions are named after the arch. `RP_SESSION` names one explicitly, and is
needed only on `up` and `shell`, the two verbs that take an *arch* where the rest
take a *session*; it overrides the positional argument for the rest, which is a
sharp edge worth knowing before you export it. The worked form is in
[dev-gpu-recipes.md](dev-gpu-recipes.md).

`push` deliberately excludes `target/` — your host build output is the wrong
architecture and would poison the pod's.

**`--ref` and `push` are alternatives, not partners.** `push` is
`rsync --delete` excluding `.git`, so it overwrites the working tree while
leaving the pod's git metadata pointing at whatever was checked out: the pod then
reports HEAD on one ref while holding the contents of another, and every git
command on the pod answers about the wrong thing. Use `--ref` for the modes that
do not push — a shell on a branch, an editor session, a job run straight from a
pushed branch — and leave the pod on `main` when the push loop is what moves your
code.

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

## Notes

- **A100 capacity on RunPod is intermittent** — deployment fails over across
  cloud tiers and PCIe/SXM variants; if all are exhausted it exits `75`. Retry.
- Pods below NVIDIA driver r560 are rejected and skipped: they cannot JIT the
  image's CUDA 12.6 PTX, so every model load would fail the #304 startup floor.
- If a run is killed uncleanly, `gpu-dev.sh reap` terminates the straggler;
  `gpu-dev.sh ls` shows only sessions this machine started, so it will not see a
  pod orphaned by a CI run or another checkout.
