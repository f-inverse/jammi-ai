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
ci/scripts/gpu-dev.sh shell a40 --ref my-branch   # look at a branch, not main
```

Lands you in `/root/jammi-ai` with cargo, nvcc and mold on `PATH`. **The pod is
terminated when you exit.** Nothing survives — this is for looking, not working.

`--ref` takes a branch, a tag or a commit, and defaults to `main`. A branch or
tag that does not exist is caught by `git ls-remote` *before* anything is rented;
a commit id cannot be resolved that way, so it is the one form whose failure you
pay a pod for. Either way the boot fails loudly rather than leaving you on `main`
looking at the wrong code.

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

- **The pod starts on `main` unless you say otherwise.** Bootstrap checks out
  `main`; `--ref` names a different branch, tag or commit. In *this* loop leave
  it on `main` — `push` is what carries your code, and see the note below on why
  the two do not mix.
- **`push` is `rsync --delete`**, excluding `.git`, `target/` and `.venv`. It
  makes the pod's tree match yours exactly, so anything you edited *on the pod*
  is destroyed. Never mix `push` with editing on the pod.

**Do not combine `--ref` with the push loop.** Because `push` excludes `.git`, a
pod booted with `--ref X` and then pushed to from branch `Y` reports HEAD on `X`
while holding `Y`'s files. Nothing is lost — the code that runs is the code you
pushed — but every git command on the pod, and anything reading HEAD to label a
result, now answers about a commit that is not there. `--ref` is for the modes
that never push: a shell on a branch (Recipe 1), an editor session (Recipe 10),
or a job run straight from a pushed branch.

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

**Stopping one.** There is no `stop` verb. `run` something else to displace the
job, `attach` and Ctrl-C to end it from its own terminal, or `down` the pod.
Only `down` stops the meter: a pod with no job running bills exactly the same as
one at full tilt. [dev-gpu.md](dev-gpu.md) records why the verb does not exist.

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

### A second pod of the same arch

A session is named after its arch, so a second A100 needs a name of its own:

```bash
RP_SESSION=bench ci/scripts/gpu-dev.sh up a100     # `up` takes an ARCH — name it here
ci/scripts/gpu-dev.sh run bench cargo run -p jammi-bench --release --features cuda -- gpu-inference-scale
ci/scripts/gpu-dev.sh logs bench
ci/scripts/gpu-dev.sh down bench
```

`RP_SESSION` is only needed on `up` and `shell`, which take an *arch* where every
other verb takes a *session*. Each session is its own pod with its own SSH key,
its own `jammi-<session>` block in the generated ssh config, and its own bill;
`ls` shows them all regardless of which one you are pointed at.

Set it inline on the one command that needs it, never in your shell profile: on
`down`/`attach`/`run`/`logs`/`push`/`pull`, an exported `RP_SESSION` that
**disagrees with an explicit positional session argument refuses** (exit 2,
naming both) rather than silently picking one — `RP_SESSION=bench … down a100`
refuses outright instead of guessing which one you meant. Either alone still
resolves exactly as you would expect: drop the positional to act on
`RP_SESSION`, or leave `RP_SESSION` unset to act on the positional.

A session name may contain letters, digits, `_`, `-`, and `.` anywhere but as
the *whole* name (so `bench.1` is a fine session name, but `.` and `..` are
refused, along with any name containing `/` or starting with `-`).

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

**It also ships no git, so this pod gets no checkout** — you land in `/root`,
not in `/root/jammi-ai`. The tool says so before handing you the shell and
reports the pod as being on no ref rather than naming one that is not there
(`<none>`, and that is what `ls` shows if you use `up` for this instead of the
throwaway `shell`). This is the only image where an absent checkout is expected;
anywhere else a failed bootstrap terminates the pod.

Do **not** add `--ref` here. There is nothing on the pod that can act on it, so
it is refused *before* the shell rather than ignored — `--ref` is for images that
can hold a checkout.

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

*You are living inside the cuda-gated code and want completion, go-to-definition
and inline errors over it. A local editor cannot give you that: it cannot resolve
`--features cuda` without a CUDA toolchain, so everything behind
`#[cfg(feature = "cuda")]` stays dim.*

`up` writes an ssh config for every live session:

```bash
ci/scripts/gpu-dev.sh up a100 --ref my-branch     # boot straight onto your branch
ssh -F ~/.config/runpod/ssh_config jammi-a100     # confirm it works
```

It lives at `~/.config/runpod/ssh_config` and holds one `Host jammi-<session>`
block per session, regenerated on every `up` and `down`. **`~/.ssh/config` is
never touched** — a bug rewriting that file would break every other host you
have. To make the hosts visible to tools that read the default config, opt in
once, either way:

```bash
# in ~/.ssh/config, at the top:
Include ~/.config/runpod/ssh_config
```

or point your editor's remote-SSH `configFile` setting at
`~/.config/runpod/ssh_config`.

Then connect to `jammi-a100` and open `/root/jammi-ai`. A login shell there has
cargo, rustc, nvcc and the right `CC`, which is what a remote server and language
server need.

### What this does not solve

1. **Your work now lives on a disposable pod.** It self-terminates at
   `RP_TTL_HOURS` and the sweep collects it. Work on a branch and `git push`
   often; a reaped pod takes uncommitted work with it. Consider a longer
   deadline for an editor session — an 8-hour default expiring mid-afternoon
   lands on unsaved work.
2. **Never run `push` in this mode.** It is `rsync --delete` from your machine
   and destroys pod-side edits without asking. The push loop and editing on the
   pod are mutually exclusive.
3. **Boot on the branch you intend to edit** — `up a100 --ref my-branch`. This is
   the mode `--ref` exists for: you are editing on the pod, so `push` is off the
   table and the checkout is the only thing that can put your code there.
4. **Host and port change with every new pod.** The config file regenerates, but
   an editor that has cached the old connection may need its window reloaded.
5. **The remote server is re-downloaded per pod** (~100 MB). It lives on the
   pod's disk, so a new pod means a new download.
6. **Untested: the editor server's glibc floor.** VS Code Server requires glibc
   ≥ 2.28 and the CI image is `manylinux_2_28` — exactly 2.28. It should work; no
   one has confirmed it. The plain `ssh` path above is verified.

If you are mostly in shared CPU code and touch GPU paths occasionally, Recipe 2
keeps your tree on your own disk and is much harder to lose work with.

---

## Recipe 11 — Serve from the pod, query from your laptop

*The client/server shape: `jammi-server` holds the GPU and the models, your
laptop runs the client and never loads the ML stack. It is the topology a client
offloads training and inference to a GPU host in — and it is reachable on a
rented pod, over an SSH tunnel rather than an exposed port.*

Boot a session and put your code on it as usual, then start the server as the
pod's job:

```bash
ci/scripts/gpu-dev.sh up a100
ci/scripts/gpu-dev.sh push a100
ci/scripts/gpu-dev.sh run a100 JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:8081 JAMMI_SERVER__HEALTH_LISTEN=127.0.0.1:8080 JAMMI_GPU__REQUIRE_GPU=true cargo run -p jammi-server --release --features cuda
```

**Bind loopback, not `0.0.0.0`.** Those two overrides are the whole reason this
is safe. The defaults are `0.0.0.0:8081` (Flight SQL + gRPC) and `0.0.0.0:8080`
(health), and the pod is a rented multi-tenant host: binding the wildcard offers
those ports to whatever else can reach it. It *looks* harmless only because the
pod declares `22/tcp` and nothing else, so the provider routes nothing else in —
that is somebody else's routing table, not a boundary we chose, and not one to
build on. On loopback, the session's SSH key is the only way in.

`JAMMI_GPU__REQUIRE_GPU=true` makes an unusable device fatal. Without it a GPU
build degrades to CPU with a warning, and you would be serving CPU inference off
an A100 without noticing. (The full config surface is
`docs/guide/src/deploy-server.md`.)

`run` returns as soon as the job is launched, and this job compiles first, so
wait for the server to announce itself:

```bash
ci/scripts/gpu-dev.sh logs a100
# jammi-server listening flight=127.0.0.1:8081 health=127.0.0.1:8080
```

Ctrl-C only detaches you. Then, from a second terminal, forward both ports over
the ssh config `up` already wrote:

```bash
ssh -F ~/.config/runpod/ssh_config -N -o ExitOnForwardFailure=yes \
  -L 8081:127.0.0.1:8081 -L 8080:127.0.0.1:8080 jammi-a100
```

**`-o ExitOnForwardFailure=yes` is not decoration.** Without it, a local 8081
that is already busy — a local server, a tunnel to another pod — leaves `ssh`
*connected with no forward at all*. Every query below then goes to whatever else
holds that port and comes back confident and wrong. With it, `ssh` refuses to
come up and you find out immediately.

Now drive it from your laptop, against `localhost`:

```bash
cargo run -p jammi-cli -- status     # --target already defaults to grpc://127.0.0.1:8081
curl -s localhost:8080/healthz
```

### One job per pod

`run` kills the tmux session named `jammi` and starts a new one, so a pod runs
exactly one job — and a server started with `run` **is** that job:

- **A second `run` on this session kills the server.** To pick up new code:
  `push`, then `run` the same server command again.
- For a shell beside the running server use `attach a100 --shell`, or a second
  `ssh -F ~/.config/runpod/ssh_config jammi-a100`. Plain `attach` joins the
  server's own terminal, where Ctrl-C stops it.
- Work that needs a job slot of its own needs a pod of its own —
  `RP_SESSION=bench ci/scripts/gpu-dev.sh up a100`, per Recipe 4. That is a
  second pod and a second bill.

`down a100` stops the server, the pod and the meter together.

---

## What things cost, and what stops them

Every pod carries a deadline in its own entrypoint from the moment it starts —
default 8 hours, 3 in the CI prove lane — and self-terminates. That deadline
travels in the pod's name (`jammi-gpu-ttl<H>`), so a sweep honours *that pod's*
limit rather than imposing its own.

The guards are wall-clock ceilings, **not idle detection**. A pod you stopped
using an hour ago is still billing. `down` when you are done; the guards are for
when something kills your process before you can.

`RP_TIMEOUT` is **not** one of the guards, whatever its name suggests: it caps
the SSH invocation that *launches* a job, not the job, which is detached and
outlives it. [dev-gpu.md](dev-gpu.md) has what it actually bounds.

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

**`'<ref>' is not a branch or tag`** — the ref does not exist on the remote.
Nothing was rented; fix the spelling, or `git push` the branch first. The same
refusal covers a remote that cannot be reached at all: a ref that cannot be
verified never gets a pod.

**`--ref <ref> was IGNORED`** — the session is already up on a different ref, and
`up` does not move a live pod. `down` it and `up` again; `ls` shows which ref
each session booted on. Passing the ref the pod is *already* on is a no-op, not
an error. There is no verb that moves a live checkout either, and that is
deliberate — [dev-gpu.md](dev-gpu.md) says why one cannot be trusted.

**`--ref <ref> cannot be honoured: <image> ships no git`** — you combined `--ref`
with an image that cannot hold a checkout, which in practice means Recipe 6.
Drop the flag, or use the toolchain image.

**`no checkout: this image ships no git`** — expected on the runtime image
(Recipe 6) and nowhere else. The pod is fine; it is simply on no ref, and `ls`
says `<none>`.

**`'<ref>' cannot fast-forward`** — the branch was force-pushed, so the pod's
copy is on a commit that no longer exists upstream. Bootstrap stops rather than
build it. `down` and `up` again for a clean clone.

**`::warning::sccache could not use the S3 backend`** — the cache is unreachable
and the pod fell back to a local disk cache. Builds still work, just cold. Check
`~/.config/runpod/s3`; note the SigV4 region must be the datacenter, and RunPod's
documented list of S3-capable datacenters is not accurate.

**A build fails with `-fuse-ld=mold unrecognized`** — you are in a shell that did
not pick up the container environment. `. /root/.jammi_env` fixes it; report it,
because login and interactive shells are supposed to load it automatically.
