# GPU development recipes

Task-oriented walkthroughs for `ci/scripts/gpu-dev.sh`. For *why* the tooling is
shaped this way — the cost guards, why no network volume is attached — see
[dev-gpu.md](dev-gpu.md).

The one idea to hold on to: **a pod is where work executes, not where it lives.**
Every pod carries a deadline and self-terminates; a sweep collects anything that
outlives it. Nothing you care about should exist only on a pod.

---

## Before anything else — your RunPod key

Get a key from the RunPod console: **Settings → API Keys → + Create API Key**.
Copy it at creation; the console will not show it again.

Then either write it to the file `gpu-dev.sh` reads:

```bash
mkdir -p ~/.config/runpod
printf '%s' 'YOUR_KEY' > ~/.config/runpod/key
chmod 600 ~/.config/runpod/key
```

…or export `RUNPOD_API_KEY`, which takes precedence over the file if both exist.

Confirm it works without renting anything — this only reads your balance:

```bash
curl -s "https://api.runpod.io/graphql?api_key=$(cat ~/.config/runpod/key)" \
  -H 'Content-Type: application/json' \
  -d '{"query":"query{ myself{ clientBalance } }"}'
```

A number back means you are set. `{"errors":…}` means the key is wrong.

### Handle it like a payment method, because it is

- **The key spends real money.** Anything holding it can rent GPUs on the
  account until the balance is gone.
- **Use your own key**, not a shared one. Revoking one person's access should not
  mean rotating everyone's.
- **Never commit it.** It belongs in `~/.config/runpod/`, outside any repo. The
  repo contains no key and no account identifiers — keep it that way.
- CI does **not** use your key. It reads the `RUNPOD_API_KEY` GitHub Actions
  secret, which is separate and only reachable by workflows in this repo.
- The account balance is the real spend ceiling, so leave auto-recharge **off**.
  If several developers share one account, you also share that ceiling.

That is the whole required setup. `gpu-dev.sh` generates its own SSH key per
session — nothing to register with RunPod.

The shared compile cache is optional and worth it (a build measured 188s cold vs
47s warm); see the setup block in [dev-gpu.md](dev-gpu.md). Note it needs a
*second* credential — an S3 API key, which is not the same thing as the API key
above.

### Picking an arch

```
a100   sm_80   the #277 floor; what CI proves against   (default)
l40s   sm_89   Ada — fp8 work (#308)
a40    sm_86   Ampere workstation — cheapest, usually in stock
h100   sm_90   Hopper
l4     sm_89   Ada, small
```

Prices and stock move. Observed 2026-08-03: A40 ≈ $0.35/hr with high stock,
A100 PCIe ≈ $1.19, A100 SXM ≈ $1.39, L40S ≈ $0.79, L4 ≈ $0.44.

**If you are not testing arch-specific behaviour, use `a40`.** It is a third the
price of an A100 and far more likely to be available. Reach for `a100` when the
thing you are checking is actually sm_80-specific.

---

## Recipe 1 — Look at something on a real GPU

*A quick question: does this run, what does `nvidia-smi` say, why does this kernel
behave differently.*

```bash
ci/scripts/gpu-dev.sh shell a40
```

Lands you in `/root/jammi-ai` with cargo, nvcc and mold on `PATH`. **The pod is
terminated when you exit.** Nothing survives — this is for looking, not working.

---

## Recipe 2 — Iterate on code (the main loop)

*You are changing code and want to compile or test it on a GPU repeatedly.*

```bash
ci/scripts/gpu-dev.sh up a40                    # once
ci/scripts/gpu-dev.sh push a40                  # after each local edit
ci/scripts/gpu-dev.sh run a40 cargo test -p jammi-ai --features cuda,live-gpu-tests
ci/scripts/gpu-dev.sh logs a40                  # Ctrl-C only detaches you
# ... edit, push, run, logs ...
ci/scripts/gpu-dev.sh down a40                  # when finished
```

You edit locally in your normal environment. The pod only executes.

Two things that surprise people:

- **The pod starts on `main`.** Bootstrap clones and checks out `main`, not your
  branch. Your branch and your uncommitted work arrive with `push`.
- **`push` is `rsync --delete`**, excluding `.git`, `target/` and `.venv`. It
  makes the pod's tree match yours exactly, so anything you edited *on the pod*
  is destroyed. Never mix `push` with editing on the pod.

---

## Recipe 3 — A job that outlives your terminal

*A fine-tune, an eval sweep, a long bench — something you do not want tied to an
SSH connection or a laptop lid.*

```bash
ci/scripts/gpu-dev.sh up a100
ci/scripts/gpu-dev.sh push a100
ci/scripts/gpu-dev.sh run a100 cargo run -p jammi-bench --release --features cuda -- gpu-inference-scale
```

`run` starts the command under tmux and returns immediately. Close the terminal,
shut the laptop, walk away. Then from anywhere:

```bash
ci/scripts/gpu-dev.sh logs a100      # follow output; Ctrl-C detaches, job lives
ci/scripts/gpu-dev.sh attach a100    # take the job's terminal
```

**Raise the deadline for a long job.** The default is 8 hours and the pod
self-terminates at it, mid-job included:

```bash
RP_TTL_HOURS=24 ci/scripts/gpu-dev.sh up a100
```

Only one job at a time per session — `run` replaces any existing one (it kills
the `jammi` tmux session first).

---

## Recipe 4 — Get back on a session you left

```bash
ci/scripts/gpu-dev.sh ls                  # what is still running
ci/scripts/gpu-dev.sh attach a100         # join the running job's terminal
ci/scripts/gpu-dev.sh attach a100 --shell # a plain prompt instead
```

`attach` joins the live job when there is one. Inside that pane, **Ctrl-B then D
detaches; Ctrl-C kills the job** — the tool says so before handing over the
keyboard. Use `--shell` when you want to poke around *while* a job runs, and
`logs` when you only want to watch.

Sessions are named after the arch. `RP_SESSION=second ci/scripts/gpu-dev.sh up a100`
gives you a second A100 session.

---

## Recipe 5 — Bring results back

```bash
ci/scripts/gpu-dev.sh pull a100 target/nextest
ci/scripts/gpu-dev.sh pull a100 bench-results.json
```

Paths are relative to `/root/jammi-ai`. Everything lands in `.gpu-pull/`, which
is gitignored.

Do this **before** `down`. Terminating is immediate and unrecoverable.

---

## Recipe 6 — Reproduce the shipped runtime image

*A bug that only appears in the deployed image, not the toolchain image — for
example the uid-65532 JIT-cache case (#305).*

```bash
RP_IMAGE=nvidia/cuda:12.6.3-runtime-ubi8 ci/scripts/gpu-dev.sh shell a100
```

That image has no toolchain, so you cannot build in it. It is for reproducing
runtime behaviour against an artifact you already have.

---

## Recipe 7 — Run what CI runs

*Convince yourself the GPU gate will pass before spending a CI run on it.*

```bash
ci/scripts/gpu-dev.sh up a100
ci/scripts/gpu-dev.sh push a100
ci/scripts/gpu-dev.sh run a100 'cargo test -p jammi-server --features cuda,live-gpu-tests --test it grpc_embedding_gpu -- --nocapture --test-threads=1 && cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1'
ci/scripts/gpu-dev.sh logs a100
```

Those are the two suites `ci/scripts/runpod_gpu_prove.sh` runs. It must be an
A100 — proving the sm_80 floor is the point.

---

## Recipe 8 — Trigger the GPU gate on a PR

Add the `run-gpu` label. Needs write or triage on the repo.

**Do not toggle the label to re-run.** Removing and re-adding starts a second run
against the same ref. Runs now queue instead of cancelling, so this is no longer
destructive — but on 2026-07-24 that exact gesture killed a run mid-rent and
orphaned an A100 for seven days, costing ~$187. To re-run, use GitHub's
**Re-run jobs** button or `workflow_dispatch`.

The gate also runs nightly and on every CUDA release.

---

## Recipe 9 — Find and kill strays

*You lost a terminal, a CI run died oddly, or you just want to be sure nothing is
billing.*

```bash
ci/scripts/gpu-dev.sh ls        # sessions THIS machine started
ci/scripts/gpu-dev.sh reap      # terminate any pod past its own deadline
ci/scripts/gpu-dev.sh reap 2    # force: kill everything older than 2h
```

`ls` and `reap` answer different questions. `ls` reads local session files, so it
cannot see a pod orphaned by CI or by another checkout. `reap` queries the
account and sees everything this tooling created.

`reap` only ever touches pods named `jammi-gpu*`; anything else in your RunPod
account is left alone. If it cannot reach RunPod it fails loudly rather than
reporting "no orphans" — a sweep that cannot check must never look like a clean
one.

A cron (`.github/workflows/gpu-reap.yml`) runs the same sweep every 6 hours.

---

## Recipe 10 — Editor / IDE remote session

Possible, with caveats worth reading first.

A login shell on the pod has the full toolchain (cargo, rustc, nvcc, `CC`), so a
remote server and language server work. Connect with the session's own key:

```bash
cat ~/.config/runpod/sessions/a100/meta      # RP_HOST, RP_PORT
# key: ~/.config/runpod/sessions/a100/id_ed25519
```

Add that host/port/key to `~/.ssh/config` and connect to `/root/jammi-ai`.

Three things to know:

1. **Host and port change with every pod.** Nothing maintains your ssh config;
   you re-edit it each session.
2. **Your work then lives on a disposable pod.** It self-terminates at
   `RP_TTL_HOURS` and the sweep collects it. Work on a branch and `git push`
   often — a reaped pod takes uncommitted work with it.
3. **Never run `push` in this mode.** It is `rsync --delete` from your machine
   and will destroy your pod-side edits without asking.

If you mostly want an IDE, Recipe 2 keeps your tree on your own disk and is
harder to lose work with.

---

## What things cost, and what stops them

Every pod carries a deadline in its own entrypoint from the moment it starts —
default 8 hours, 3 in the CI prove lane — and self-terminates. That deadline
travels in the pod's name (`jammi-gpu-ttl<H>`), so a sweep honours *that pod's*
limit rather than imposing its own.

The guards are wall-clock ceilings, **not idle detection**. A pod you stopped
using an hour ago is still billing. `down` when you are done; the guards are for
when something kills your process before you can.

Your prepaid balance is the real ceiling, provided auto-recharge stays off.

---

## When it goes wrong

**`exit 75` / "no GPU capacity"** — a genuine provider condition. Retry, or pick
another arch. This is a neutral skip, not a failure.

**"deployed … never became reachable"** — a dead host: it rented and never
exposed SSH. The tooling terminates it and tries the next candidate. Common
enough not to worry about; just retry.

**Any other refusal fails loudly and stops**, e.g. `INSUFFICIENT_BALANCE`. Only a
capacity condition is treated as a skip — anything else is a real fault and says
so.

**`no live session '<name>'`** — the pod was reaped or the host died. Start a new
one with `up`.

**`::warning::sccache could not use the S3 backend`** — the cache is unreachable
and the pod fell back to a local disk cache. Builds still work, just cold. Check
`~/.config/runpod/s3`; note the SigV4 region must be the datacenter, and RunPod's
documented list of S3-capable datacenters is not accurate.

**A build fails with `-fuse-ld=mold unrecognized`** — you are in a shell that did
not pick up the container environment. `. /root/.jammi_env` fixes it; report it,
because login and interactive shells are supposed to load it automatically.
