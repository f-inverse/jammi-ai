# Pod build guide — an ops manual for the pod build substrate

Companion to [dev-gpu.md](dev-gpu.md) (design/setup reference) and
[dev-gpu-recipes.md](dev-gpu-recipes.md) (task-oriented walkthroughs). Those
two answer "how do I rent a pod and run a job." This document answers a
narrower question in more depth: **how does a pod actually get from a cold
container to a build that only recompiles jammi's own code**, procedurally,
with exact commands, expected output, time budgets, and what to do when a
step deviates. It is written for whoever runs the seed/clone/timing scripts
directly — a maintainer debugging a slow or failed build, or an agent
instructed to reproduce one.

Every claim below cites the script and line(s) that make it true. These
citations are hand-maintained: `ci/scripts/perf/check_citations.py`
mechanically re-resolves citations only for the four files its
`_KNOWN_FILES` map names (`finetune_step.rs`, `grad_oracle.rs`,
`torch_grad_oracle.py`, `torch_finetune_step.py`) and only within its
`_SEARCH_ROOTS` (`crates/jammi-bench`, `ci/scripts/perf`, the cuda-runs
artifacts) — it never reads this guide. Every number carries its
producer; where no committed producer exists, this document says so instead
of stating the number (`ci/scripts/check_doc_numbers_have_producers.py`'s
own discipline, adopted here even though that gate does not scan `docs/`).

---

## 1. What this is / when to use it

**The problem.** A fresh RunPod container has no build state at all: the
Rust registry must be fetched, the crate graph compiled, cutlass provisioned
if you need FlashAttention-2. Doing that cold on every single edit is the
default failure mode this substrate exists to remove — `docs/maintainer/dev-gpu.md`'s
"build substrate" section is the design rationale (S3-backed sccache was
tried and measured to add wall-clock for zero cache reuse; see that doc for
the cited numbers, not restated here per the DRY rule — a fact lives in
exactly one place).

**The model, one diagram:**

```
 pod boots ──▶ bootstrap (git clone/fetch/checkout)  [runpod_lib.sh:1394 rp_bootstrap]
      │
      ▼
 SEED build, once per pod, detached
   (pod_seed_target.sh)                              [ci/scripts/gpu-dev.sh:581 start_seed_build]
   → a CARGO_TARGET_DIR with every third-party dependency
     fully compiled, then swept member-free
      │
      ▼
 CLONE, once per tree/job
   (pod_target_clone.sh)                              [ci/scripts/gpu-dev.sh:856 target case]
   → cp -a (reflink where possible) the seed into a
     fresh CARGO_TARGET_DIR; a pure copy, no deletion step
      │
      ▼
 PUSH-STAMP the tree's contents (optional, for
   uncommitted-work iteration)                        [ci/scripts/pod_push_stamp.sh]
      │
      ▼
 BUILD (cargo build/test/clippy against the clone's
   CARGO_TARGET_DIR) — only jammi's own code (and
   whatever Cargo.lock/features actually changed)
   ever recompiles
      │
      ▼
 TIMINGS (optional, for the pod-build-substrate's
   own acceptance measurement)                        [ci/scripts/perf/pod_build_timings.sh]
```

A **seed** is a `CARGO_TARGET_DIR` for one pod, built once, member-free (no
`jammi-*` artifact survives in it). A **clone** is a per-tree, per-job
`CARGO_TARGET_DIR` that starts as a pure copy of the seed. A **tree** is a
plain directory (`/root/jammi-ai` for the bootstrap checkout, or
`/root/trees/<name>` for any other) — never a git worktree
(`ci/scripts/runpod_lib.sh:604-622`, `rp_tree_dir`'s own doc: a worktree add
fails on the checked-out ref, and a shared `.git` couples trees that must
diverge independently).

**Who may do what.** Agents build and test over SSH (`shell`/`attach`/`run`,
and calling the seed/clone/timing scripts directly once on a pod). Only the
lead or the user runs the verbs that rent or terminate hardware:
`ci/scripts/gpu-dev.sh up`/`down`/`reap` — renting spends real money and
`down`/`reap` are destructive (`ci/scripts/gpu-dev.sh:906-984`, the `down`
case's own verify-before-terminate machinery, and `runpod_lib.sh:1167-1182`,
`rp_sweep`'s own doc: "every ambiguity resolves toward terminating"). A
session or pod belonging to a process you did not start is never yours to
`down` — `runpod_lib.sh:171-183`'s session-name containment check and the
`up --replace` refusal path (`ci/scripts/gpu-dev.sh:653-676`) exist
specifically because two processes racing `up`/`down` on the same alias is
how the 2026-08-25 incident (`docs/maintainer/dev-gpu.md:154-173`)
terminated an unrelated pod.

---

## 2. Prerequisites

1. **A RunPod API key.** `~/.config/runpod/key` (mode 600) or
   `RUNPOD_API_KEY` — `ci/scripts/gpu-dev.sh:87-88` reads the file first,
   falling back to the env var, and refuses loudly (not silently) if
   neither is set.
2. **`RP_TTL_HOURS` / `RP_DEV_TTL_HOURS`.** Every pod self-terminates at a
   deadline baked into its own entrypoint at deploy
   (`runpod_lib.sh:834-890`, `_rp_deploy_payload`'s watchdog). `up` alone
   raises the default from `runpod_lib.sh`'s own 8h
   (`runpod_lib.sh:94`) to `RP_DEV_TTL_HOURS` (default 72h) when
   `RP_TTL_HOURS` is not set explicitly (`ci/scripts/gpu-dev.sh:453-479`) —
   a dev session someone is actively using is meant to survive a workday,
   not die at the throwaway-pod default. This is independent of the
   **account-level sweep**: `reap` (`runpod_lib.sh:1183-1296`, `rp_sweep`)
   judges each `jammi-gpu*` pod against the deadline carried in its own
   name (`<prefix>-ttl<H>`, `runpod_lib.sh:1266-1276`) — so rent a dev pod
   with `RP_TTL_HOURS`/`RP_DEV_TTL_HOURS` set to at least the job's
   expected length; there is no verb to pause the sweep for one pod
   (`runpod_lib.sh:95-97`, `.github/workflows` runs `reap` every 6h
   independent of anything else). `RP_DEV_TTL_HOURS`/`RP_TTL_HOURS` are
   validated to be positive integers before use
   (`ci/scripts/gpu-dev.sh:462-479`).
3. **Disk sizing.** `RP_DISK_GB` (default 60,
   `runpod_lib.sh:106`) must cover `>= 25 (base) + S_src + S_seed +
   N*S_clone` (one clone per tree the pod hosts) — the exact `S_src`/
   `S_seed`/`S_clone` byte counts are measured by
   `ci/scripts/perf/pod_build_timings.sh` and committed at
   `ci/artifacts/pod-build-timings/20260827T183928Z-bc27e75.json`
   (§4 below): ≈ 3.6 / 7.8 / 8.1 GB (decimal, from the artifact's exact
   byte fields) — the default 60 GB covers base + src + seed + two
   clones (≈ 52.7 GB); a third tree computes to ≈ 60.9 GB, over the
   default, so a pod hosting 3+ trees sizes up (`RP_DISK_GB=70`+). A mutation-testing session (copy-mode `cargo mutants`, one full
   workspace+target copy per job) wants `RP_DISK_GB >= 120`
   (`runpod_lib.sh:56-72`, `ci/scripts/gpu-dev.sh:187-194`).
4. **Tool preflight, on the pod.** Both `pod_seed_target.sh` and
   `pod_push_stamp.sh` assert every external tool they call exists
   *before* spending real time: `pod_seed_assert_required_tools`
   (`ci/scripts/pod_seed_target.sh:105-115`, requires `cargo`, `git`,
   `python3`, and `sha256sum` or `shasum`) and
   `pod_push_assert_required_tools` (`ci/scripts/pod_push_stamp.sh:98-108`,
   additionally requires `rsync`, `stat`, `awk`, `sort`). Both fail loudly,
   naming every missing tool at once — never a cryptic "command not found"
   discovered one tool at a time deep into a build.

---

## 3. Runbook A — rent and reach a pod

```bash
ci/scripts/gpu-dev.sh up a100                 # lead/user only
```

`up` provisions across a candidate list (SECURE then COMMUNITY cloud tier,
PCIe then SXM4 variant for `a100` —
`runpod_lib.sh:1006-1018`, `rp_deploy_arch`), polls for SSH up to
`RP_SSH_WAIT_SECS` (default 600s — `runpod_lib.sh:137`, raised for a cold
image pull that can take minutes before sshd is even up), and rejects any
candidate below NVIDIA driver r560 (`runpod_lib.sh:86-91`, `965-979`: the
CUDA 12.6 PTX floor). Expected output ends with:

```
=== session 'a100' up on <host>:<port> @ <ref> (pod <id>) ===
    deadline: self-terminates in <H>h unless you 'down' it first
    ssh:     ssh -F <config> jammi-a100
    attach:  ci/scripts/gpu-dev.sh attach a100
    run job: ci/scripts/gpu-dev.sh run a100 cargo test ...
    STOP:    ci/scripts/gpu-dev.sh down a100
```

(`ci/scripts/gpu-dev.sh:697-705`). No candidate reachable at all exits `75`
(a neutral capacity skip, `runpod_lib.sh:892-1000`, `rp_deploy_live`'s own
return value) — retry, this is not a code failure.

**Readiness is polled state, never a log-banner grep.** `up`/`shell` block
on SSH liveness inside `rp_deploy_live`'s own wait loop
(`runpod_lib.sh:958-990`), and every later verb (`attach`/`run`/`push`/…)
calls `require_pod` → `rp_session_alive`, an actual `ssh ... true`
(`ci/scripts/gpu-dev.sh:549-554`, `runpod_lib.sh:487-490`) — never a string
match against a boot log. Chain any further pod work on this liveness
check, not on a fixed sleep.

**`up` refuses over a live session** (exit 2) rather than silently deploying
a second pod under the same alias — `ci/scripts/gpu-dev.sh:653-676`. If the
alias has a recorded-but-unreachable pod (reaped, mid-reboot, or a race with
another process's `up`), inspect with `ls` first; `--replace` overwrites
only the *local* record, never terminates the old pod
(`ci/scripts/gpu-dev.sh:674-676`).

```bash
ci/scripts/gpu-dev.sh ls                      # every session this machine started
```

Prints session/pod/ref/arch@host, column widths derived from the actual
rows (`runpod_lib.sh:492-524`).

```bash
ci/scripts/gpu-dev.sh attach a100             # join the running job, or a plain shell
```

`attach` joins the tree's own tmux job pane if one is running, else a plain
login shell (`ci/scripts/gpu-dev.sh:521-547`, `707-715`); `--shell` forces
the plain prompt even with a job running.

---

## 4. Runbook B — one-time seed

**When it runs.** `up`/`shell` kick it off *detached* immediately after
bootstrap, unconditionally, so it never blocks your shell
(`ci/scripts/gpu-dev.sh:581-598`, `start_seed_build`; `611`, `695`). It runs
under the shared timing lock, acquired as the first thing inside the
detached tmux pane (`flock -n -E 75 /root/.jammi-timing.lock bash
pod_seed_target.sh --no-lock ...`, `ci/scripts/gpu-dev.sh:594`) — the lock's
lifetime is then the seed job's lifetime, not the short-lived SSH
invocation that started tmux.

**Watch it:**

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100
tmux attach -t =jammi-seed          # detach: Ctrl-B then D
```

or tail `/root/.jammi-seed.log` directly.

**The phases, in order** (`ci/scripts/pod_seed_target.sh:632-918`,
`pod_seed_target_main`):

1. **`--no-lock` re-exec.** Absent `--no-lock`, the script re-execs itself
   through `pod_timing_lock.sh acquire -w <JAMMI_SEED_LOCK_WAIT_SECS>`
   (default 1800s, `pod_seed_target.sh:58`) — `:642-661`. `start_seed_build`
   already runs `--no-lock` inside its own tmux+flock wrapper
   (`gpu-dev.sh:594`), so this arm is for a caller invoking the script
   directly over SSH.
2. **Marker gate.** If `.jammi-seed-complete` already exists and `--reseed`
   was not given, the whole run is a no-op (`:683-686`). If
   `.jammi-seed-failed` exists, it refuses to retry automatically and
   prints the failure tail (`:687-691`) — pass `--reseed` to force either
   case.
3. **Required-tools preflight** (`:673-676`, §2 item 4 above).
4. **`cargo metadata --locked` priming, network allowed, exactly once**
   (`:743-772`). This is load-bearing: every later `--frozen` metadata call
   in this script needs the *full* cross-platform dependency graph already
   fetched (`cargo metadata` without `--filter-platform` walks
   platform-conditional crates `cargo build` alone never fetches). Skipping
   this step is what produced the a100c incident in §8, row 8.
5. **T1** — `cargo build --release -p jammi-bench --features cuda`
   (`:774-775`).
6. **T1b** — `cargo build --release -p jammi-bench --features
   cuda,jammi-kernels/flash-attn`, **main-only**, gated on the checkout's
   *resolved sha* matching `origin/main` (or `JAMMI_SEED_IS_MAIN=1`), never
   on `abbrev-ref == "main"` — a checkout-by-sha always leaves a detached
   HEAD (`:708-736`, `776-812`; see §8 row 3). `pod_seed_pkg_has_feature`
   detects flash-attn live via `cargo metadata`, never hand-asserted
   (`:264-331`); its rc=2 ("could not determine") **aborts the whole seed**
   rather than silently skipping T1b (`:800-811`) — a broken metadata query
   read as "absent" is exactly how a seed once stamped complete without its
   FA2 artifacts.
7. **T2** — `cargo test --no-run` for the exact crates/features
   `runpod_gpu_prove.sh`'s own CI suites use, kept in lockstep by naming
   the same `-p`/`--features`/`--test` here (`:814-818`).
8. **T3** — `cargo clippy -p jammi-kernels --all-targets --features cuda -D
   warnings` (`:820-821`).
9. **Capture build-script stdout before cleaning** (`:823-825`,
   `pod_seed_capture_build_output`) — cargo's own cleaner removes
   `build/<pkg>-*/output`, so this is the only chance to read it.
10. **Member-free clean**: `cargo clean --workspace` (both profiles) plus
    an explicit `rm -rf */incremental` (cargo's own cleaner does not
    remove `incremental/build_script_build-*`) (`:827-836`).
11. **Non-member path/patch package check** via `cargo metadata` (no
    `source: null` package outside `workspace_members`) (`:838-847`).
12. **Filesystem-level member-free assertion**
    (`pod_seed_assert_member_free`, `:333-470`, `:849-850`) — the *only*
    mechanical, always-on check that no `jammi-*`-named entry
    (hyphenated, underscored, or `lib`+underscored — the compiled-library
    naming form) survives under `{debug,release}/{.fingerprint,deps,build,
    incremental}`.
13. **Env-surface cross-check** (`pod_seed_check_stdout_subset`,
    `:190-241`, `:852-856`): every `cargo:rerun-if-env-changed=<NAME>` the
    captured build-script output actually announced must be listed in
    `ci/scripts/pod_seed_key_inputs.toml` (§9 below). A captured file that
    is legitimately zero bytes (a build script with nothing `cargo:`-shaped
    to print) is **not** an error on its own — only zero *captured files
    total* is (`:204-241`, see §8 row 9).
14. **Completion marker** — `${JAMMI_SEED_DIR}.jammi-seed-complete`, JSON:
    `ref`, `sha`, `date`, `tuples` (`["T1","T2","T3"]` plus `"T1b"` iff it
    ran), `rustflags`, `size_bytes`, `manifest_sha256`, `t1b_flash_attn_ran`,
    `t1b_flash_attn_reason` (`:858-885`).

**Verifying the seed:**

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100 \
  "cat /root/.jammi-seed.jammi-seed-complete"
```

A present file with `"tuples"` including everything you expect (and, on
`main`, `"t1b_flash_attn_ran": true`) is a complete seed.

**Failure markers and recovery.** A failed run writes
`.jammi-seed-failed` — not a bare log tail, but a diagnostic-bearing report:
the phase in progress at failure (the last `=== ... ===` echo), `df -h /`
and a memory snapshot, every line matching an error/warning/`nvcc fatal`/
`Killed`/`::error::`/`::warning::` shape (adjacent duplicates collapsed)
with 20 lines of trailing context, and the last 40 lines
(`ci/scripts/pod_seed_target.sh:574-629`,
`pod_seed_write_failure_marker`). Read it:

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100 \
  "cat /root/.jammi-seed.jammi-seed-failed"
```

Re-running without `--reseed` refuses and reprints the tail
(`:687-691`) — this is deliberate, so `up`/`shell` never silently burn real
compile minutes retrying a known-broken seed on every invocation. Once the
real cause is fixed:

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100 \
  "bash /root/jammi-ai/ci/scripts/pod_seed_target.sh --reseed --no-lock"
```

(`--no-lock` here because you are already inside an interactive shell, not
a lock-held tmux pane — see §6). Common failure classes and their meaning
are in §8.

**Time budget.** The committed producer JSON is
`ci/artifacts/pod-build-timings/20260827T183928Z-bc27e75.json`
(`ci/scripts/perf/pod_build_timings.sh` run on a live A100-SXM4 pod at
`bc27e75`; the script never runs in CI — its module doc,
`ci/scripts/perf/pod_build_timings.sh:1-9`). It pins the per-job walls:
seed→clone copy 2 s, member-only clone build 69 s, cold build from an
empty target dir 243 s, FA2 leg 122 s, plus the `S_src`/`S_seed`/`S_clone`
byte counts the `RP_DISK_GB` formula cites (see dev-gpu.md). The **full
seed wall itself** (T1+T1b+T2+T3+clean+checks, from nothing) is not a
field in that artifact's schema — the run's seed pre-existed, so leg (i)
was a marker/member-free verification, not a timed build; budget tens of
minutes for a fresh Ampere pod compiling the full cuda+FA2 graph (the two
live seeds of 2026-08-27 landed in that band — session-log evidence,
recorded in esc-050/esc-051's ledger rows, not a committed producer) and
watch the log until a
producer run with a cold seed commits that number. The artifact also
records `byte_equal_clone_vs_cold: false` for the member rlib/rmeta set
across the two (deliberately different) target-dir paths, with the full
per-file diff — an open question (path-embedding vs real
nondeterminism), stated rather than dropped.

---

## 5. Runbook C — clone and build a job

**`target` clones the build-substrate SEED into a fresh `CARGO_TARGET_DIR`
(`/root/target-<name>`) — it never creates or populates the tree/checkout
itself.** A tree is populated ONLY by `push` (`ci/scripts/runpod_lib.sh:604-622`,
`rp_tree_dir`'s own doc). `--with-cutlass` provisions cutlass INTO an
already-pushed tree (`pod_provision_cutlass.sh`); against a tree that has
never been pushed it REFUSES with the tool's own error text: `tree source dir
'...' does not exist — push to it first` (`pod_provision_cutlass.sh:106`).
Push before `target --with-cutlass`:

```bash
ci/scripts/gpu-dev.sh push a100 --tree mywork         # populates /root/trees/mywork
ci/scripts/gpu-dev.sh target a100 mywork              # clones the seed into /root/target-mywork
ci/scripts/gpu-dev.sh target a100 mywork --with-cutlass   # ALSO requires the push above
```

(Bare `target` without `--with-cutlass` does not itself touch
`/root/trees/mywork` and so does not strictly require a prior `push` — but
every other verb that acts on tree `mywork` (`run`/`attach`/`push`/`pull`)
does, so pushing first is the ordinary flow in practice.)

**Version consistency, by construction.** `target` stages THIS checkout's
OWN `pod_target_clone.sh`/`pod_seed_target.sh`/`pod_provision_cutlass.sh`/
`pod_push_stamp.sh` to `/root/.jammi-caller-scripts/` and runs THOSE, never
the pod's bootstrapped `/root/jammi-ai` copies (baked at boot from whatever
`main` was then) — a pod booted before a marker/behavior change like
esc-077 landed would otherwise clone successfully with an OLDER
`pod_target_clone.sh` that stamps no marker, and the very next `run` (whose
own preflight is always THIS checkout's code) would refuse a perfectly
legitimate clone. The seed build itself (`gpu-dev.sh up`/`shell`) is the one
deliberate exception — it runs the pod's OWN bootstrapped
`pod_seed_target.sh` against that SAME checkout's commit, since seeding a
tree with a different commit's build script is a worse inconsistency than
the one being avoided.

Under the hood this is `pod_target_clone.sh <seed-dir> <dest-dir>
[tree-dir]` (`ci/scripts/gpu-dev.sh:1203-1213`), which:

1. **Refuses without the seed's completion marker** (exit 3) —
   `pod_target_clone.sh:130-140`. No seed, no clone: this is the "did the
   seed actually finish" gate a caller cannot bypass by accident.
2. **`cp -a --reflink=auto`** the seed into the destination
   (`:144-160`) — CoW where the filesystem supports it, a real copy
   otherwise; never a hardlink (a hardlink clone was reproduced to corrupt
   the seed itself — writing through a hardlinked path mutates the shared
   inode, `pod_target_clone.sh:1-10`).
3. **Unconditional member-freedom check on the clone**
   (`pod_seed_assert_member_free`, `:166-174`) — if the seed was *not*
   actually member-free, the clone is deleted and the call fails right
   here, rather than surfacing later as a mysterious stale-artifact bug.
4. **`--verify` (opt-in, after your own first build):** pipe a `cargo
   build -v` log on stdin; asserts no line matches `^\s*Fresh\s+jammi-`
   (`:68-77`) — a member-free seed means every `jammi-*` unit must actually
   *compile* (not report `Fresh`) on the clone's first build. This is
   additional to, never a substitute for, the unconditional check in step 3
   — `--verify` only ever runs when a human remembers to run it and only
   catches units Fresh on *this one* build.

5. **Stamps `<dest-dir>/.jammi-clone-of-seed`** (esc-077) with the seed dir,
   the seed's own completion-marker mtime/sha256, and the clone timestamp —
   `gpu-dev.sh run` REFUSES a job whose `CARGO_TARGET_DIR` is missing this
   marker, and the refusal it gives depends on which of the three unmarked
   states the dir is in:

   | preflight state | what it means | remedy the refusal names |
   |---|---|---|
   | `MISSING` | no such directory | `target <session> <tree> --with-cutlass` |
   | `UNMARKED_COLD` | exists, no workspace-member artifacts — a job would pay the full build | `target <session> <tree> --with-cutlass` |
   | `UNMARKED_WARM` | carries this workspace's own member fingerprints, but no marker: built before the marker scheme, or outside the `target` verb. **Warm — a job would not rebuild the workspace; only its provenance is unverified.** | `target <session> <tree> --adopt` |

   `--adopt` copies and deletes nothing: it re-runs the same content
   validation a clone runs (`pod_seed_assert_member_free`, demanding the
   opposite answer — member artifacts must be PRESENT, since they are the
   warmth being claimed) and stamps the marker in place, recording
   `adopted: true` and no seed provenance. A cold dir, a dir with neither
   `debug/` nor `release/`, and a missing dir are all refused, so `--adopt`
   can never launder a cold dir into a marked one. A plain clone cannot
   serve a warm dir at all — it refuses to write over an existing
   destination (`ci/scripts/pod_target_clone.sh:142`).

   `RP_ALLOW_COLD_TARGET=1` is the sole override for proceeding without a
   marker (`pod_build_timings.sh`'s own measurement legs bypass `run`
   entirely and are unaffected — see that script's own module doc).
   **Documented residual:** this gate covers only a job launched through
   `run` — a caller who `ssh`es onto the pod directly and invokes `cargo`
   by hand bypasses it entirely; it is not claimed closed, only that the
   sanctioned path fails closed.

**One-pod-per-wave, wave-scoped; sub-units share.** `gpu-dev.sh run` REFUSES
when a LIVE job on the pod belongs to a DIFFERENT wave — `--wave W` /
`RP_WAVE` names the wave (default: the tree name, exactly preserving the
tree-scoped behavior a caller who sets neither ever sees). `run` records the
claim (`/root/.jammi-active-wave.d/<tree>.claim`: wave + tree + timestamp —
one file per HOLDER, so N sanctioned same-wave co-tenants are all
represented and each clears only its own; every write and removal runs
inside a flock on `/root/.jammi-active-wave.lock`) at job launch and
clears it when the job's tmux session ends; `push` also accepts `--wave`/
`RP_WAVE` (writes no claim itself — only so an operator can use the SAME
`--wave` across a push-then-run sequence without either verb rejecting it).
A wave's own sub-units (e.g. a CPU-build tree and a GPU-test tree) sharing
ONE `--wave` id may run sequentially on the SAME pod across DIFFERENT trees
— per-tree tmux still serializes the actual jobs; only a genuinely
DIFFERENT wave's live job is refused, naming the owning wave and the remedy
(rent another pod: `RP_SESSION=<alias> gpu-dev.sh up <arch>`), with
`RP_ALLOW_CONCURRENT=1` as the override for deliberate cross-wave
co-tenancy. A stale claim (file present, no live tmux session for
THAT holder — e.g. the wrapper's own cleanup was interrupted by a
SIGKILL/pod death) is skipped: tmux liveness is the primary signal, never a
claim file alone.
Same documented-residual scope as above.

**Poisoned-clone detection**, concretely:

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100 <<'EOF'
export CARGO_TARGET_DIR=/root/target-mywork
cd /root/trees/mywork
cargo build -v --release -p jammi-bench --features cuda 2>&1 | tee /tmp/build.log
EOF
cat /tmp/build.log | ssh -F ~/.config/runpod/ssh_config jammi-a100 \
  "bash /root/jammi-ai/ci/scripts/pod_target_clone.sh '' /root/target-mywork --verify"
```

`clone verify OK` means no member unit was Fresh; a `FAILED` line names
every offending unit.

**`pod_push_stamp.sh` — what is stamped, when.** `gpu-dev.sh push` rsyncs
your working tree (uncommitted work included) to the tree's directory,
excluding `target/`, `.git`, `.venv*`, `.claude`, `.sccache`, `.gpu-pull`,
`scratchpad`, and the cutlass submodule
(`ci/scripts/pod_push_stamp.sh:163-174`, the *one* place this list lives —
the real rsync and the stamp's own manifest hash read it from the same
function so they can never drift apart, `ci/scripts/gpu-dev.sh:812-816`).
The rsync itself passes `--no-owner --no-group` (`gpu-dev.sh:817-834`) — see
§8 row 7 for why this is load-bearing, not cosmetic. Immediately after,
`pod_push_stamp.sh compute <repo-root> <session>` writes
`<tree>/.jammi-push-stamp.json`:

- `laptop_head` — `git rev-parse HEAD`, or the literal `"unknown"` if
  `repo-root` is not a real git checkout (`pod_push_stamp.sh:253-256`).
- `porcelain_sha256` / `diff_head_sha256` — hashes of `git status
  --porcelain` / `git diff HEAD`, **not** silently `sha256("")` when either
  git command itself fails in a real repo (§8 row 4;
  `pod_push_stamp.sh:257-289`).
- `manifest_sha256` — sha256 over the sorted `(path, mode, content-sha256)`
  triples of exactly what the same exclude set would push, computed
  entirely locally against a fresh empty temp directory via `rsync --dry-run
  --out-format='%n'` (`pod_push_stamp.sh:194-248`). **Stat-flavour
  detection** (`pod_push_stat_mode`, `:110-152`) is what makes this
  deterministic across GNU and BSD hosts — see §8 row 6.
- `cutlass_gitlink` — `git rev-parse --verify --quiet
  HEAD:crates/jammi-kernels/third_party/cutlass`, `null` if the path is not
  a gitlink at HEAD (`:302-318`, `--verify --quiet` specifically — see §8
  row 5 for why the bare form is unsafe here).

`check_cuda_run_artifacts.py`'s `git_sha` rule is **unchanged** by any of
this: the push stamp is iteration provenance only, never a substitute for a
pushed (reachable-from-a-remote-branch) sha on a committed artifact
(`pod_push_stamp.sh:1-9`).

**`--with-cutlass`**, concretely — `pod_provision_cutlass.sh
<tree-source-dir> [super-dir]` (defaults `/root/jammi-ai`), the *one*
provisioning surface for cutlass in any tree
(`pod_provision_cutlass.sh:1-17`):

1. `git -C <super-dir> submodule update --init --depth 1 <cutlass-path>`
   (`:108-117`) — network required.
2. **Content-floor validation**: the checkout must have `.git` (or a
   gitlink file), `include/cutlass/cutlass.h`, and an on-disk file count
   `>= git ls-tree -r HEAD | wc -l` for the same commit
   (`:119-143`) — catches a technically-valid `.git`/HEAD sitting over an
   empty/partial checkout (a real a100e incident: an unrelated push deleted
   the superproject's own submodule content out from under it).
3. **Self-target guard**: refuses (as a no-op, exit 0) if `super-dir` and
   `tree-source-dir` are the identical path (`:145-155`) — `rm -rf` would
   otherwise delete the source before `cp -a` could read it.
4. **Expected pin**, decided by the destination tree's own shape
   (`:157-185`): a tree that is itself a real git checkout (its own `.git`,
   `.gitmodules` declaring the path) reads its own live `HEAD:<path>`
   gitlink; a tree with no `.git` (the ordinary `push`-populated case)
   falls back to `.jammi-push-stamp.json`'s `cutlass_gitlink`.
5. On mismatch: fetch+checkout the pinned commit into the *superproject's*
   submodule, re-verify, and only then copy (`:187-208`) — a `set -e`-EXEMPT
   `if`/`else` around the check call, not a bare command, so the
   remediation arm is reachable (a prior round's bare-command form made it
   dead code; see the file's own module doc for the reproduction).
6. `rm -rf` the destination path, `cp -a` the superproject's cutlass into
   it, then strip the copied `.git` (a bare submodule gitlink pointer file
   that would otherwise create a nested-repository boundary inside a
   git-backed destination tree) and assert it is gone
   (`:210-238`).

**Expected "exactly one PTX rewritten" for a `.cu` edit.** Edit one `.cu`
file, rebuild against the clone, and diff `release/*.ptx` timestamps/hashes
before and after — only the PTX for the kernel whose `.cu` changed should
differ; every other PTX and every third-party `.rlib`/`.rmeta` must be
byte-identical (this is exactly what
`ci/scripts/perf/pod_build_timings.sh`'s `snapshot_hashes` compares, §6
below — `pod_build_timings.sh:345-354`). **No committed timing number for
this edit's wall-clock exists in this tree** (see §4's time-budget note);
budget for one `nvcc` invocation plus a relink, not a full rebuild.

**The FA2 leg's own separate clone dir.** `pod_build_timings.sh` clones the
FA2 measurement into `/root/.jammi-clone-fa2-a2`
(`ci/scripts/perf/pod_build_timings.sh:409`), deliberately **never**
`CLONE_DIR` (`/root/.jammi-clone-a2`, `:271`) — T1's own byte-hash snapshot
is taken immediately after T1, before the FA2 leg touches anything
(`:298-304`, item N3 in that script's own module doc): reusing one
clone dir for both would make T1's snapshot include FA2-only artifacts,
guaranteeing a spurious byte-equality mismatch against the cold leg.

---

## 6. Runbook D — measure

**`pod_timing_lock.sh` — acquire/release semantics.**

```
pod_timing_lock.sh acquire -n         -- <cmd...>   # refuse immediately if held
pod_timing_lock.sh acquire -w <SECS>  -- <cmd...>   # wait up to SECS
```

(`ci/scripts/pod_timing_lock.sh:13-31`). The lock is a single pod-wide
`flock` at `/root/.jammi-timing.lock`, kernel-owned: it dies the instant the
holding process exits or is killed, so there is nothing to "steal" and
nothing stale to clean up (`:1-11`). A refusal (held elsewhere, or the wait
timed out) exits **75** — the same "resource unavailable, try again" code
`rp_deploy_live` uses for a capacity skip, never a code meaning "this is
broken" (`:33-37`). The holder file (`<lock>.holder`) is written
tmp-then-rename *under* the lock (never truncate-in-place — a reader racing
the write can only ever see the old-complete or new-complete file,
`:39-44`), and removed on release via a trap set *inside* the flock-held
child (`:46-59`, `pod_timing_lock.sh:114`).

**The concurrent `-n` case.** The automatic seed build still builds `flock
-n -E 75 ...` directly into its own detached tmux pane's command line
(`ci/scripts/gpu-dev.sh:594`, `start_seed_build`) rather than going through
this script — acquiring the lock is the *first* thing that happens inside
the pane itself, so the lock's lifetime is the job's lifetime, not the
short-lived SSH invocation that started tmux and returned in under a
second. `run --timing` moved its OWN acquisition *inside* the generated
`.jammi-job.sh` wrapper instead (`rp_job_wrapper_with_marker_lines`,
`ci/scripts/runpod_lib.sh:705-737`) — fd-based (`flock -n 9`), essentially
the first real step of that script too, for the identical M6 reasoning —
rather than split across an outer `flock -n -E 75 ... bash job.sh` command
line (`ci/scripts/gpu-dev.sh:750-754`, `LAUNCH`'s own if/else), so a lock
refusal and the job's own real exit code can never collide on the literal
value 75 (round-N audit finding B3). Either shape, a second `run --timing`
(or `pod_build_timings.sh` itself, or a manual `pod_seed_target.sh`
invocation) against an already-held lock refuses at once with exit 75,
naming the current holder from the holder file — it never silently queues.

**`pod_build_timings.sh` legs**, in order
(`ci/scripts/perf/pod_build_timings.sh`):

- **(i) seed** (`:183-224`) — runs `pod_seed_target.sh --no-lock`, asserts
  the completion marker, re-runs `pod_seed_assert_member_free` as an
  independent witness, records `S_seed_bytes` and the seed's own
  `t1b_flash_attn_ran`/`reason`.
- **source-tree size** (`:226-227`) — `S_src_bytes`, `du -sk --exclude=.git`.
- **(ii) clone build at the FA2 tip** (`:229-297`) — `git fetch`+`checkout`
  the tip ref (requires a git-backed tree with a reachable `origin`; a
  purely `push`-populated tree cannot run this leg, see §8 row 1),
  provisions cutlass if `.gitmodules` declares it, clones the seed, times
  `cargo build --release -p jammi-bench --features cuda -v`, records
  `copy_wall_s`, `clone_build_wall_s`, whether the copy reflinked.
- **(iii) sccache** (`:460-461`) — records `sccache --show-stats`
  before/after; labelled "unchanged by construction" since the wrapper is
  off pod-wide.
- **(iv) byte-equality vs a cold build** (`:463-537`) — a *genuinely empty*
  second `CARGO_TARGET_DIR` (never a copy-then-clean of the clone, which
  would still share every third-party artifact — `:470-479`), scoped to
  `release/*.ptx` + `release/**/libjammi_*.{rlib,rmeta}` only (see §8 row
  10 for why the linked binary is excluded). **Four-state result**
  (`byte_equal_state`): `invalid` (empty match set on either side — the
  comparison never ran), `set_mismatch` (both sides non-empty but the paths
  present differ), `true`, `false` — never collapsed into a bare boolean.
- **(v) sizes and copy mechanics** (`S_src_bytes`/`S_seed_bytes`/
  `S_clone_bytes`/`copy_wall_s`/`reflink`) — the `RP_DISK_GB` formula's
  producer (§2 item 3).
- **(vi) clone vs cold wall-clock, plus the FA2 leg separately** — both
  real numbers land in the same-run JSON so a reader computes any delta
  they want; the FA2 leg (`:359-457`) is gated on the *resolved* HEAD sha
  matching `origin/main` (or `JAMMI_MAIN_SHA`), always records `fa2_ran`/
  `fa2_reason` even when skipped (see §8 row 3 — the class this fixes).

`JAMMI_BUILD_TIMINGS_OUT` (required env var) is where the JSON is written —
**never stdout**, since this script also prints progress markers there
(`pod_build_timings.sh:89-97`). Run it:

```bash
JAMMI_FA2_TIP_REF=<ref-or-sha> JAMMI_BOX_LABEL='a100d (A100 PCIe, driver 570)' \
JAMMI_BUILD_TIMINGS_OUT=/root/pod-build-timings.json \
  bash ci/scripts/perf/pod_build_timings.sh
```

then copy the JSON to `ci/artifacts/pod-build-timings/<ts>-<sha7>.json` and
commit it (`pod_build_timings.sh:95-97`) — the first such committed run is
`ci/artifacts/pod-build-timings/20260827T183928Z-bc27e75.json`, the
producer this document's §4 walls and `dev-gpu.md`'s `RP_DISK_GB` S values
cite.

**Contamination.** The lock serializes only jammi's own timing-sensitive
producers on the same pod; it does nothing about an unrelated foreign build
sharing the box's CPU/disk/nvcc. `docs/maintainer/dev-gpu.md`'s timing-lock
section records a real instance of a foreign build turning a 348s leg into
415s — the fix is procedural, not mechanical: **rent an exclusive box** for
any run whose numbers you intend to commit, never share a pod between a
timing run and other work.

**Reading the JSON.** Every field name above is the field name in the
committed artifact; `clone_features == cold_features` (both hardcoded
`"cuda"` at their own assignment sites) is recorded for a reader's
information but is **not** itself an assertion of anything — a prior
version of this leg tested two adjacent string literals against each other
and could never fail (`pod_build_timings.sh:526-536`). The real like-for-like
guarantee is `byte_equal_state`, described above.

---

## 7. Runbook E — tear down

```bash
ci/scripts/gpu-dev.sh down a100               # lead/user only
```

`down` never trusts the locally-recorded pod id on its own
(`ci/scripts/gpu-dev.sh:906-984`, `runpod_lib.sh:258-353`,
`rp_pod_verify`). Before terminating, it confirms the id is **both** still
present in the account's live pod list **and** named like this tooling's
own pods (`<prefix>-ttl<digits>`):

- **Match** → `rp_terminate`, then **confirm gone** by re-querying the
  account (`rp_pod_gone`, `runpod_lib.sh:378-412`) before forgetting the
  local record — a rejected `podTerminate` must never both leak the pod and
  destroy the only record pointing at it (`gpu-dev.sh:920-939`).
- **Absent from the account entirely** (return code 3) → this is *not* a
  refusal, it is the ordinary shape of "this pod already ended on its own"
  (its own TTL, or the sweep) — `down` forgets the record and says so
  (`gpu-dev.sh:939-958`).
- **Present under a name that is not this tooling's shape, or the query
  itself failed** → refuse (exit 1), *keep* the local record — this is
  exactly the ambiguous case a follow-up `up` most needs to still see and
  refuse on (`gpu-dev.sh:958-980`).

**The id, not the TTL, is authoritative** — two earlier attempts to make the
exact TTL part of this check were both removed as either a false-refusal
source or found inert on every input (`runpod_lib.sh:266-287`, full
history in that comment). RunPod pod ids are globally unique, so name-shape
plus a matched id is already sufficient.

**Interrupted / read-only paths that never terminate.** A `down` that
cannot confirm the account state (network failure mid-query, an
unrecognized name shape) exits with the local record intact rather than
guessing — there is no code path in `down` that both fails to confirm *and*
still terminates or forgets the record (`gpu-dev.sh:958-980`). The only
manual escape from that refusal is the RunPod console.

```bash
ci/scripts/gpu-dev.sh reap                    # honour each pod's own deadline
ci/scripts/gpu-dev.sh reap 2                  # force-reap everything older than 2h
```

`reap` is account-wide (never a per-session verb) and judges age from
`createdAt`, never `runtime.uptimeInSeconds` — uptime is null for the first
minutes of a healthy pod and can reset on a container restart, either of
which would misjudge age (`runpod_lib.sh:1248-1252`). Every ambiguity
(unreadable telemetry, an unparseable deadline, a stopped pod) resolves
toward terminating (`runpod_lib.sh:1259-1275`); a query that cannot reach
RunPod at all fails loudly rather than reporting "nothing to clean up"
(`runpod_lib.sh:1280-1283`).

**Reconciling account state when boot output was empty.** If `up`/`shell`
returned with no coordinates (a dropped connection mid-deploy, a killed
laptop), do not assume nothing was rented — `rp_deploy_live` marks
`RP_POD_CREATED=1` the *instant* a pod id comes back from the deploy
mutation, minutes before SSH ever comes up (`runpod_lib.sh:229-241`,
`938-948`), so a pod can exist and bill even when the invocation that
rented it never got to print anything. Reconcile via:

```bash
ci/scripts/gpu-dev.sh ls
```

and, if that shows nothing but you suspect a leak, check the RunPod console
directly for any `jammi-gpu-ttl<H>` pod — its own in-pod deadline will
still fire even with no local record of it, and the next `reap` sweep will
catch anything that outlives that.

---

## 8. Invariants & failure catalogue — the tree-state class

Every row below is one instance of the same underlying class: a script
assumed a git/filesystem state that a pushed, cloned, or copy-provisioned
tree does not actually have. Each is pinned by a real fix and (where one
exists) a `test_pod_substrate.sh` leg.

| # | State assumed | What the tree actually has | Symptom | Pinned by |
|---|---|---|---|---|
| 1 | A tree has `.git` | `push` excludes `.git` entirely (`pod_push_stamp.sh:163-174`) — a pushed tree is a plain rsync'd directory | `git submodule`/any git command on a pushed tree fails "not a git repository" | `pod_provision_cutlass.sh:1-17`'s own module doc; `test_pod_substrate.sh` `(m/A1 match)`/`(m/A1 drift)`/`(m/A1 deinit)`/`(m/A1 fetch-failure)`/`(m/A1 revert-RED)` (`:1171`) |
| 2 | A destination path is either untouched or a real git submodule | It can already hold a **copy-provisioned** cutlass (`cp -a` with `.git` stripped, `pod_provision_cutlass.sh:210-238`) | `git submodule update --init` on that path refuses (non-empty, non-submodule-shaped dir) — a live a100c run wasted 819s before failing | `pod_provision_cutlass.sh:39-61` module doc; provisioning now `rm -rf` + `cp -a` unconditionally, never asks git to touch the destination |
| 3 | `git rev-parse --abbrev-ref HEAD == "main"` on a checkout of main | A checkout-by-sha (the ordinary shape for an FA2 PR tip / a resolved seed sha) always leaves a **detached HEAD**, whose abbrev-ref reads the literal `"HEAD"` | T1b/the FA2 leg silently never ran, with no recorded reason — indistinguishable from "correctly determined not-main" | `pod_seed_target.sh:708-736`; `pod_build_timings.sh:371-388`; `test_pod_substrate.sh` `(n/addendum EXECUTABLE b)` (`:1621`) — gated on the *resolved sha*, `t1b_ran`/`t1b_reason` and `fa2_ran`/`fa2_reason` always recorded |
| 4 | `git status --porcelain`/`git diff HEAD` failing == "no output" == "clean" | A failing git command in a real repo (locked index, corrupted ref) also produces empty stdout, hashing to the identical `sha256("")` a genuinely clean tree would produce | `manifest_sha256`/`porcelain_sha256` reports a byte-identical "clean" hash for a broken repo-root | `pod_push_stamp.sh:254-289` (discriminated on whether `HEAD` itself resolved) |
| 5 | `git rev-parse HEAD:<path>` on a missing path fails with empty output | The **bare** form echoes its own argument text to stdout (rc=128) — `git rev-parse HEAD:no/such/path` literally prints `HEAD:no/such/path` | A bogus literal string becomes `cutlass_gitlink`/an expected pin, read as a real (but wrong) value | `pod_push_stamp.sh:302-318`; `pod_provision_cutlass.sh:63-78` — every call site now uses `--verify --quiet`, silent on a miss |
| 6 | `stat -f FORMAT` means "use this format string" | On GNU coreutils, `-f` means "display file **system** status" (the opposite of BSD) — the old fallthrough printed a multi-line, live `Free:`-block-bearing status report to stdout before failing | `manifest_sha256` diverged nondeterministically between two hosts building the identical bundle (a100c vs a100e) | `pod_push_stamp.sh:110-152`, `pod_push_stat_mode` — flavour detected once via `stat --version`, memoized |
| 7 | Pushing files as `root@pod` leaves them root-owned | `rsync -a` preserves owner/group from the **laptop's** own uid (e.g. 501) unless told not to | `git`, run as root inside the pushed tree, refuses "detected dubious ownership" | `ci/scripts/gpu-dev.sh:817-834` — `--no-owner --no-group` |
| 8 | `cargo metadata --frozen` "just works" once `Cargo.lock` exists | `cargo metadata` (unlike `cargo build`) resolves the **full cross-platform** graph by default, needing source for platform-conditional crates never otherwise fetched | Seed pipeline died on "failed to download android_system_properties ... --frozen was specified" *after* T1–T3 had already succeeded | `pod_seed_target.sh:743-772` (one-time network-allowed priming call before every `--frozen` call); `:305-316`, `pod_seed_cargo_metadata_frozen` (captures real stderr, never discards it) |
| 9 | A zero-byte captured `build/<pkg>-*/output` file means "captured at the wrong moment" | Cargo creates that file for **every** build script it runs, regardless of whether the script prints anything — a real no-op script legitimately produces zero bytes | An earlier fix flagged legitimate zero-byte captures (chrono-tz, esaxx-rs, pulldown-cmark, rustls, scratch, snap, stacker, prometheus) as errors, aborting every real seed build | `pod_seed_target.sh:190-241`, `pod_seed_check_stdout_subset`; `test_pod_substrate.sh` `(l/N4)` / `(l/N4 revert-RED)` (`:970`) |
| 10 | Two builds of the identical tree on the same box produce byte-identical linked binaries | mold 2.35.1 / clang 21's ThinLTO codegen embeds local-symbol suffixes (`anon.<h>.N.llvm.<hash>`) that differ between two builds of the **same** tree | `release/jammi-bench` (467 differing symbols) made the byte-equality leg read `false` even though every deterministic artifact (`*.ptx`, `.rlib`/`.rmeta`) matched | `pod_build_timings.sh:330-344`, `:644-654` (`byte_equal_scope` — the linked binary is explicitly excluded, never silently dropped from the claim) |
| 11 | `push --tree <name>`'s rsync destination is reachable | rsync creates only the LAST path component of its own destination — nothing in the pod bootstrap or the build-substrate seed provisions `/root/trees` itself | The very FIRST `push` for a name no session has ever pushed before fails outright on a fresh pod: `rsync: mkdir "/root/trees/<name>" failed: No such file or directory (2)` (observed live on pod u4hfsqyu0i2qwa) | `runpod_lib.sh:657-665`, `rp_push_ensure_parent` (a bounded, idempotent remote `mkdir -p` on the parent, called before every push); `test_pod_substrate.sh` `(y/esc-056)` |
| 12 | `gpu-dev.sh`'s `REPO_ROOT` names the checkout the caller means | It is derived from the SCRIPT's own on-disk location, never `$PWD` — a multi-worktree laptop keeps more than one copy simultaneously | Invoking one tree's script copy from inside a DIFFERENT tree silently `push`/`run`/`target`s the WRONG tree; the push-stamp's own `laptop_head` field was the only tell (M1b) | `gpu-dev.sh:267-279` (push/run/target refuse on a cwd/REPO_ROOT mismatch, `RP_ALLOW_ROOT_MISMATCH=1` overrides); `test_pod_substrate.sh` `(z/esc-056)` |

---

## 9. Reference

| Script | Synopsis | Key env vars | Exit codes |
|---|---|---|---|
| `ci/scripts/gpu-dev.sh` | The CLI: `shell`/`up`/`target`/`attach`/`run`/`logs`/`push`/`pull`/`wait-seed`/`wait-job`/`down`/`ls`/`reap` (`:24-45`, `:90-197`) | `RUNPOD_API_KEY`, `RP_IMAGE`, `RP_TTL_HOURS`, `RP_DEV_TTL_HOURS`, `RP_DISK_GB`, `RP_VOLUME_GB`, `RP_WAIT_TIMEOUT_SECS`, `RP_WAIT_INTERVAL_SECS`, `RP_WAIT_MAX_TRANSPORT_FAILS` (`:67-74`) | `0` ok; `2` usage/argument error; `1` a real failure (bootstrap, verify, named wait-seed/wait-job failure); `75` no-capacity neutral skip (propagated from `rp_deploy_live`); `wait-seed`/`wait-job` additionally: `2` transport failure, `3` timed out with no verdict (`rp_wait_poll`) |
| `ci/scripts/runpod_lib.sh` | Shared RunPod primitive — deploy/verify/terminate/sweep/wait-poll; sourced, not run directly (`:1-82`) | `RUNPOD_API_KEY`, `RP_SESSION`, `RP_KEEP`, `RP_TTL_HOURS` (default 8, `:94`), `RP_DISK_GB`, `RP_SSH_WAIT_SECS` (default 600, `:137`), `RP_WAIT_SSH_BOUND_SECS` (default 60, `_rp_bounded_capture`'s own local kill bound) | `rp_pod_verify`: `0` match, `1` present-wrong-shape, `2` query failed, `3` absent (`:258-353`); `rp_deploy_live`: `0` ok, `1` real deploy fault, `75` no reachable capacity (`:892-1000`); `rp_wait_poll`: `0` success, `1` named failure, `2` transport failure, `3` timed out (`:1101-1165`) |
| `ci/scripts/pod_seed_target.sh` | Builds/cleans the member-free seed (`:1-51`) | `JAMMI_SEED_DIR` (default `/root/.jammi-seed`), `JAMMI_TREE_DIR` (default `/root/jammi-ai`), `JAMMI_SEED_LOCK_WAIT_SECS` (default 1800), `--reseed`, `--no-lock`, `JAMMI_SEED_DRY_RUN` | `0` ok/no-op; non-zero on any build/check failure, writes `.jammi-seed-failed` |
| `ci/scripts/pod_target_clone.sh` | Clones the seed into a fresh `CARGO_TARGET_DIR`; `--verify` checks a piped `cargo build -v` log (`:1-27`) | positional `<seed-dir> <dest-dir> [tree-dir]`, `--verify` | `0` ok; `1` member-free check failed post-clone (clone removed) or verify found a `Fresh jammi-*` unit; `2` destination already exists; `3` no seed at `<seed-dir>` |
| `ci/scripts/pod_provision_cutlass.sh` | The one cutlass provisioning surface into an existing tree (`:1-17`) | positional `<source-tree-dir> [super-dir]` (default `/root/jammi-ai`) | `0` ok/no-op (self-target); `1` any validation, network, or mismatch failure |
| `ci/scripts/pod_push_stamp.sh` | Exclude-set source of truth + push provenance stamp; `excludes`/`compute`/`cutlass-check` subcommands (`:11-55`) | — | `compute`: `0` ok, `1` a hashing/tool/git failure; `cutlass-check`: `0` match, `1` genuine mismatch, `2` no usable stamp |
| `ci/scripts/pod_timing_lock.sh` | One flock seam for every pod-side timing-sensitive producer (`:1-59`) | `JAMMI_TIMING_LOCK` (default `/root/.jammi-timing.lock`), `JAMMI_TIMING_LABEL`, `JAMMI_TIMING_JOB` | `0` ok (wrapped command's own exit code); `2` usage error; `75` lock refused/timed out |
| `ci/scripts/perf/pod_build_timings.sh` | A2 pod-build-substrate acceptance producer; live-pod only, never CI (`:1-97`) | `JAMMI_TREE_DIR`, `JAMMI_SEED_DIR`, `JAMMI_FA2_TIP_REF` (required), `JAMMI_BOX_LABEL` (required), `JAMMI_BUILD_TIMINGS_OUT` (required), `JAMMI_MAIN_SHA` (optional FA2-gate override), `--no-lock` | `0` ok; `1` any leg's `fail` call (named, on stderr) |

**Test suite:**

```bash
bash ci/scripts/test_pod_substrate.sh          # every leg (a)-(z), hermetic, no pod
JAMMI_REQUIRE_LOCK_TEST=1 bash ci/scripts/test_pod_substrate.sh   # additionally requires flock present (Linux-only leg (d))
bash ci/scripts/test_gpu_dev_lifecycle.sh      # gpu-dev.sh's own dispatch/session lifecycle legs
```

`test_pod_substrate.sh`'s assertions **are** the documented invariants — the
table in §8 cites specific legs; the full leg index (each a lettered
section header in the file) covers: `(a)`/`(c)` env-preamble + no S3/sccache
(`:83`); `(b)` `pod_target_clone.sh` (`:156`); `(d)` `pod_timing_lock.sh`
flock exclusivity, incl. the `run --timing` fd-based flock's own shape
(`:242`); `(e)` key-manifest RED tests against real sources (`:426`); `(f)`
push excludes pinned/single-sourced (`:558`); `(g)` tmux window-name
anchoring (`:598`); `(h)` exactly two `/root/jammi-ai` literal sites
(`:647`); `(i)` clone/tree directory namespaces stay disjoint, plus
`rp_job_wrapper_with_marker_lines`'s own token/marker/flock-inside-the-
wrapper shape (round-N audit finding B3) (`:674`); `(j)`/`(k)` seed
failure-arm (incl. `--reseed` removing a stale COMPLETE marker too, finding
B2a) and `--no-lock` re-exec argv shape (`:821`, `:907`); `(l)` env-surface
cross-check incl. zero-byte captures (`:970`); `(m)`/`(m/A1)` cutlass pin
comparison and provisioning (`:1070`, `:1171`); `(n)` `pod_seed_pkg_has_feature`
live detection (`:1440`); `(o)` timing-lock live witness (`:1825`); `(p)` the
other unconditional member-freedom call sites (`:1918`); `(q/A2)` a real
two-member cargo workspace fixture for member-freedom (`:2131`); `(r/A4)`
byte-equality tri/four-state (`:2258`); `(s/manifest)` real a100c seed
failure replay (`:2381`); `(t)` class-shaped tripwire convention
(`:2506`); `(u)` claim-tripwire convention (`:2596`); `(v/push)`
`pod_push_stamp.sh` determinism + preflight (`:2664`); `(w/esc-050)`/
`(x/esc-051)` the seed-tuple-unguarded class closure (`:2825`, `:3149`);
`(y/esc-056)` `push` provisions its own tree's parent directory before
rsyncing, on a fresh pod (`:3319`); `(z/esc-056)` `push`/`run`/`target`
refuse a cwd/`REPO_ROOT` mismatch, `RP_ALLOW_ROOT_MISMATCH=1` overrides
(`:3447`).

**The key-inputs manifest** (`ci/scripts/pod_seed_key_inputs.toml`) —
every input that can change the bytes a build emits without cargo's own
fingerprint noticing (`:1-12`). Sections: `[cargo]` (Cargo.lock/toml,
RUSTFLAGS family, `:97-114`), `[jammi_kernels_build_rs]` /
`[jammi_wire_build_rs]` / `[jammi_bench_build_rs]` (the three in-workspace
build scripts, `:116-164`), `[bindgen_cuda_0_1_6]` / `[cudarc]` (vendored
CUDA-toolchain build scripts, `:133-248`), `[cc_1_2_57]` (hand-enumerated —
the one section the automated scan cannot verify mechanically,
`:166-218`), `[toolchain_identities]` (`rustc -vV`/`cargo -V`/`nvcc
--version`/…, `:220-230`), `[non_key]` (cargo-derived, not a human toggle,
`:250-280`), `[vendored_non_cuda]` (package-specific opt-in toggles
unrelated to the CUDA toolchain, `:282-368`). Extending it requires (a) a
citation to the reading source and (b) `test_pod_substrate.sh`'s RED tests
(i)/(ii)/(iii) to still pass against the real sources (`:14-40`).

---

## 10. Glossary

- **Seed** — a `CARGO_TARGET_DIR` for one pod, every third-party dependency
  fully compiled, then swept **member-free** (`pod_seed_target.sh`).
- **Clone** — a per-tree, per-job `CARGO_TARGET_DIR` produced by `cp -a`
  from the seed (`pod_target_clone.sh`); a pure copy, no deletion step.
- **Tree** — a plain source-checkout directory on the pod (`/root/jammi-ai`
  or `/root/trees/<name>`), never a git worktree.
- **Member-free** — a `CARGO_TARGET_DIR` with zero `jammi-*`-named artifact
  under `{debug,release}/{.fingerprint,deps,build,incremental}`, verified
  at the filesystem level by `pod_seed_assert_member_free`.
- **Push stamp** — `.jammi-push-stamp.json`, iteration provenance for what
  a `push` actually sent; never a substitute for a committed artifact's
  pushed-sha requirement.
- **Timing lock** — the single pod-wide `flock` at
  `/root/.jammi-timing.lock` serializing every timing-sensitive producer on
  one pod.
- **T1/T1b/T2/T3** — the seed's four build tuples: T1 release build, T1b
  the main-only flash-attn leg, T2 `cargo test --no-run` for the CI prove
  lane's own suites, T3 clippy.
- **A2** — the pod-build-substrate acceptance measurement produced by
  `ci/scripts/perf/pod_build_timings.sh`, committed under
  `ci/artifacts/pod-build-timings/` once run.
- **`.jammi-seed-complete` / `.jammi-seed-failed`** — the seed's completion
  and failure markers, sitting beside (not inside) the seed's own
  `CARGO_TARGET_DIR`.
