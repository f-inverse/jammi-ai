# GPU development on RunPod

jammi's default `cargo test` is CPU-hermetic; the GPU path is exercised only by
the gated `live-gpu-tests` lane. You don't need a local GPU — rent one per task
on RunPod through `ci/scripts/gpu-dev.sh`.

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

The build-substrate cache is **optional** — without it everything still works,
just cold. To enable it, create a network volume plus an S3 API key (RunPod →
Settings → S3 API Keys) and write:

```bash
cat > ~/.config/runpod/s3 <<'EOF'
RP_S3_ENDPOINT=https://s3api-us-ks-2.runpod.io
RP_S3_BUCKET=<network-volume-id>
RP_S3_ACCESS_KEY_ID=user_...
RP_S3_SECRET_ACCESS_KEY=rps_...
EOF
chmod 600 ~/.config/runpod/s3
```

Then publish the cargo-registry prewarm once per `Cargo.lock` change:

```bash
ci/scripts/gpu-dev.sh prewarm a100
```

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

To reproduce the **shipped runtime image** (e.g. the uid-65532 JIT-cache case in
#305) instead of the toolchain image:

```bash
RP_IMAGE=nvidia/cuda:12.6.3-runtime-ubi8 ci/scripts/gpu-dev.sh shell a100
```

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
command and your SSH connection. Sessions are named after the arch; set
`RP_SESSION` for a second pod of the same one.

`push` deliberately excludes `target/` — your host build output is the wrong
architecture and would poison the pod's.

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
   and could hang. Every pod self-terminates after `RP_TTL_HOURS` (default 8; the
   CI prove lane uses 3). It deliberately is *not* installed over SSH: the gap
   between "pod rented" and "pod reachable" is minutes long, and that gap is
   where the orphan above was created.

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

Guard 2 needs the network at deadline time; guard 3 needs this repo's CI to be
running. They fail for unrelated reasons, which is the point of having both.

All three are **wall-clock ceilings, not idle detection** — a pod you stop using
bills until its deadline. Run `down` when you're finished.

Both are **wall-clock ceilings, not idle detection** — a pod you stop using bills
until its deadline. Run `down` when you're finished; the guards are for the times
something kills the process before you can.

```bash
RP_TTL_HOURS=2 ci/scripts/gpu-dev.sh up a100
```

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
