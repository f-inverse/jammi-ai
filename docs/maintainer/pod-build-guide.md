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

Every claim below cites the script and line that make it true, by NAME
first and line second — `` `identifier` (`path:line`) ``, where the
identifier is verbatim text from the line it names.
`ci/scripts/perf/check_citations.py` mechanically re-resolves every one of
them: this guide is one of that gate's citing roots, a repo-root-relative
path is resolved directly against the working tree, and the backtick-quoted
identifier immediately preceding a citation must be a substring of the line
it names. A line number alone rots the moment a neighbouring function is
added or deleted; a name does not, and a name plus a line is re-resolvable
by grep the next time the line drifts. Every number carries its producer;
where no committed producer exists, this document says so instead of
stating the number (`ci/scripts/check_doc_numbers_have_producers.py`'s own
discipline, adopted here even though that gate does not scan `docs/`).

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
 pod boots ──▶ bootstrap (git clone/fetch/checkout)
      │
      ▼
 SEED build, once per pod, detached (pod_seed_target.sh)
   → a CARGO_TARGET_DIR with every third-party dependency
     fully compiled, then swept member-free
      │
      ▼
 CLONE, once per tree/job (pod_target_clone.sh)
   → cp -a (reflink where possible) the seed into a
     fresh CARGO_TARGET_DIR; a pure copy, no deletion step
      │
      ▼
 PUSH-STAMP the tree's contents (optional, for
   uncommitted-work iteration — pod_push_stamp.sh)
      │
      ▼
 BUILD (cargo build/test/clippy against the clone's
   CARGO_TARGET_DIR) — only jammi's own code (and
   whatever Cargo.lock/features actually changed)
   ever recompiles
      │
      ▼
 TIMINGS (optional, for the pod-build-substrate's
   own acceptance measurement — pod_build_timings.sh)
```

Each box's own entry point: bootstrap is `rp_bootstrap` (`ci/scripts/runpod_lib.sh:1782`);
the detached seed is `start_seed_build` (`ci/scripts/gpu-dev.sh:762`), called
from `[ -n "$RP_REF" ] && start_seed_build` (`ci/scripts/gpu-dev.sh:785`) for
`shell` and again at `[ -n "$RP_REF" ] && start_seed_build` (`ci/scripts/gpu-dev.sh:869`)
for `up`; the clone is the `target)` (`ci/scripts/gpu-dev.sh:1139`) case, which
runs `bash ${STAGE_DIR}/pod_target_clone.sh` (`ci/scripts/gpu-dev.sh:1211`).

A **seed** is a `CARGO_TARGET_DIR` for one pod, built once, member-free (no
`jammi-*` artifact survives in it). A **clone** is a per-tree, per-job
`CARGO_TARGET_DIR` that starts as a pure copy of the seed. A **tree** is a
plain directory (`/root/jammi-ai` for the bootstrap checkout, or
`/root/trees/<name>` for any other) produced by `rp_tree_dir`
(`ci/scripts/runpod_lib.sh:644`) — never a git worktree, because
`a worktree add fails on the checked-out ref` (`ci/scripts/runpod_lib.sh:635`)
and a shared `.git` couples trees that must diverge independently.

**Who may do what.** Agents build and test over SSH (`shell`/`attach`/`run`,
and calling the seed/clone/timing scripts directly once on a pod). Only the
lead or the user runs the verbs that rent or terminate hardware:
`ci/scripts/gpu-dev.sh up`/`down`/`reap` — renting spends real money and
`down`/`reap` are destructive: `down` runs its own verify-before-terminate
machinery at `rp_pod_verify "$RP_POD_ID"` (`ci/scripts/gpu-dev.sh:1258`), and
`reap`'s own doc states `Every ambiguity resolves toward terminating`
(`ci/scripts/runpod_lib.sh:1568`). A session or pod belonging to a process
you did not start is never yours to `down` — the session-name containment
check `rp_session_name_check` (`ci/scripts/runpod_lib.sh:207`) and the
`up --replace` refusal path, `refusing to silently replace it`
(`ci/scripts/gpu-dev.sh:844`), exist specifically because two processes
racing `up`/`down` on the same alias is how the
`2026-08-25 incident where an agent's` (`docs/maintainer/dev-gpu.md:200`)
terminated an unrelated pod.

---

## 2. Prerequisites

1. **A RunPod API key.** `~/.config/runpod/key` (mode 600) or
   `RUNPOD_API_KEY` — `cat "${HOME}/.config/runpod/key"`
   (`ci/scripts/gpu-dev.sh:116`) reads the file first, falling back to the
   env var, and `:?set RUNPOD_API_KEY` (`ci/scripts/gpu-dev.sh:117`) refuses
   loudly (not silently) if neither is set.
2. **`RP_TTL_HOURS` / `RP_DEV_TTL_HOURS`.** Every pod self-terminates at a
   deadline baked into its own entrypoint at deploy — `_rp_deploy_payload`
   (`ci/scripts/runpod_lib.sh:1173`) builds a `watchdog = ("( sleep %d; "`
   (`ci/scripts/runpod_lib.sh:1199`) into it. `up` alone raises the default
   from `RP_TTL_HOURS="${RP_TTL_HOURS:-8}"` (`ci/scripts/runpod_lib.sh:94`)
   to `RP_DEV_TTL_HOURS="${RP_DEV_TTL_HOURS:-72}"`
   (`ci/scripts/gpu-dev.sh:575`), assigned by
   `RP_TTL_HOURS="$RP_DEV_TTL_HOURS"` (`ci/scripts/gpu-dev.sh:589`) when
   `RP_TTL_HOURS` is not set explicitly — a dev session someone is actively
   using is meant to survive a workday, not die at the throwaway-pod
   default. This is independent of the **account-level sweep**: `rp_sweep`
   (`ci/scripts/runpod_lib.sh:1571`) judges each `jammi-gpu*` pod against the
   deadline carried in its own name, `"name": "%s-ttl%s" % (prefix, ttl_h),`
   (`ci/scripts/runpod_lib.sh:1223`), read back by
   `if tail.startswith('-ttl') and tail[4:].isdigit():`
   (`ci/scripts/runpod_lib.sh:1658`) — so rent a dev pod with
   `RP_TTL_HOURS`/`RP_DEV_TTL_HOURS` set to at least the job's expected
   length; there is no verb to pause the sweep for one pod —
   `Every pod this tooling rents is named` (`ci/scripts/runpod_lib.sh:95`)
   — and the sweep runs on its own schedule,
   `cron: "23 */6 * * *"` (`.github/workflows/gpu-reap.yml:22`), independent
   of anything else. `RP_DEV_TTL_HOURS` is validated to be a positive
   integer before use — `RP_DEV_TTL_HOURS must be a positive integer`
   (`ci/scripts/gpu-dev.sh:586`) and `RP_DEV_TTL_HOURS must be > 0`
   (`ci/scripts/gpu-dev.sh:588`) — as is `RP_TTL_HOURS` itself, by
   `RP_TTL_HOURS must be a positive integer` (`ci/scripts/runpod_lib.sh:103`).
3. **Disk sizing.** `RP_DISK_GB` (default 60 —
   `RP_DISK_GB="${RP_DISK_GB:-60}"`, `ci/scripts/runpod_lib.sh:106`) must cover `>= 25 (base) + S_src + S_seed +
   N*S_clone` (one clone per tree the pod hosts) — the exact `S_src`/
   `S_seed`/`S_clone` byte counts are measured by
   `ci/scripts/perf/pod_build_timings.sh` and committed at
   `ci/artifacts/pod-build-timings/20260827T183928Z-bc27e75.json`
   (§4 below): ≈ 3.6 / 7.8 / 8.1 GB (decimal, from the artifact's exact
   byte fields) — the default 60 GB covers base + src + seed + two
   clones (≈ 52.7 GB); a third tree computes to ≈ 60.9 GB, over the
   default, so a pod hosting 3+ trees sizes up (`RP_DISK_GB=70`+). A mutation-testing session (copy-mode `cargo mutants`, one full
   workspace+target copy per job) wants `RP_DISK_GB >= 120` —
   `A mutation-testing session wants >= 120 GB.`
   (`ci/scripts/runpod_lib.sh:72`), repeated in the CLI's own env doc at
   `a mutation-testing session wants >= 120 GB.`
   (`ci/scripts/gpu-dev.sh:102`).
4. **Tool preflight, on the pod.** Both `pod_seed_target.sh` and
   `pod_push_stamp.sh` assert every external tool they call exists
   *before* spending real time: `pod_seed_assert_required_tools`
   (`ci/scripts/pod_seed_target.sh:105`) requires `cargo`, `git`,
   `python3`, and `sha256sum` or `shasum`, and
   `pod_push_assert_required_tools` (`ci/scripts/pod_push_stamp.sh:98`)
   additionally requires `rsync`, `stat`, `awk`, `sort`. Both fail loudly,
   naming every missing tool at once — never a cryptic "command not found"
   discovered one tool at a time deep into a build.

---

## 3. Runbook A — rent and reach a pod

```bash
ci/scripts/gpu-dev.sh up a100                 # lead/user only
```

`up` provisions across a candidate list built by `rp_deploy_arch`
(`ci/scripts/runpod_lib.sh:1376`) — SECURE then COMMUNITY cloud tier, PCIe
then SXM4 variant for `a100`, plus that arch's same-SASS capacity
fallbacks where it has any (`a100)    cand=`,
`ci/scripts/runpod_lib.sh:1382`). It polls for SSH up to
`RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-600}"`
(`ci/scripts/runpod_lib.sh:137`), raised for a cold image pull that can take
minutes before sshd is even up, and rejects any candidate below NVIDIA
driver r560 — the CUDA 12.6 PTX floor,
`RP_MIN_DRIVER_MAJOR="${RP_MIN_DRIVER_MAJOR:-560}"`
(`ci/scripts/runpod_lib.sh:91`), enforced at
`-ge "$RP_MIN_DRIVER_MAJOR"` (`ci/scripts/runpod_lib.sh:1330`). Expected
output ends with:

```
=== session 'a100' up on <host>:<port> @ <ref> (pod <id>) ===
    deadline: self-terminates in <H>h unless you 'down' it first
    ssh:     ssh -F <config> jammi-a100
    attach:  ci/scripts/gpu-dev.sh attach a100
    run job: ci/scripts/gpu-dev.sh run a100 cargo test ...
    STOP:    ci/scripts/gpu-dev.sh down a100
```

— printed by `=== session '${SESSION}' up on` (`ci/scripts/gpu-dev.sh:873`).
No candidate reachable at all exits `75`, a neutral capacity skip:
`rp_deploy_live` (`ci/scripts/runpod_lib.sh:1236`) ends with its own
`return 75` (`ci/scripts/runpod_lib.sh:1360`). Retry — this is not a code
failure.

**Readiness is polled state, never a log-banner grep.** `up`/`shell` block
on SSH liveness inside `rp_deploy_live`'s own wait loop —
`while [ "$SECONDS" -lt "$_rp_deploy_deadline" ]; do`
(`ci/scripts/runpod_lib.sh:1317`) — and every later verb
(`attach`/`run`/`push`/…) calls `require_pod`
(`ci/scripts/gpu-dev.sh:691`), which is
`rp_session_load && rp_session_alive` (`ci/scripts/gpu-dev.sh:692`), and
`rp_session_alive` (`ci/scripts/runpod_lib.sh:513`) is an actual
`ssh "${RP_SSHO[@]}" -p "$RP_PORT" "root@${RP_HOST}" true`
(`ci/scripts/runpod_lib.sh:515`) — never a string match against a boot log.
Chain any further pod work on this liveness check, not on a fixed sleep.

**`up` refuses over a live session** (exit 2) rather than silently deploying
a second pod under the same alias — `elif [ "$REPLACE" != "1" ]; then`
(`ci/scripts/gpu-dev.sh:827`), refusing with
`refusing to silently replace it.` (`ci/scripts/gpu-dev.sh:844`) and naming
the remedy at `once you are sure it should be replaced`
(`ci/scripts/gpu-dev.sh:846`). If the alias has a recorded-but-unreachable
pod (reaped, mid-reboot, or a race with another process's `up`), inspect
with `ls` first; `--replace` overwrites only the *local* record, never
terminates the old pod —
`WITHOUT terminating it` (`ci/scripts/gpu-dev.sh:849`).

**A branch that exists only on your laptop cannot be `--ref`'d.** `up`
resolves a branch or tag against the remote *before* renting anything —
`rp_ref_precheck` (`ci/scripts/runpod_lib.sh:1736`) — and a name the remote
does not carry fails closed with
`is not a branch or tag in` (`ci/scripts/runpod_lib.sh:1746`), naming that
`nothing was rented`; an unreachable remote is equally refused
(`refusing to rent a pod for a ref that cannot be verified`,
`ci/scripts/runpod_lib.sh:1748`). Only a 40-hex commit id skips the remote
check, and it is then verified on the pod instead. So an unpushed campaign
branch reaches the pod the other way: boot on `main` (or any pushed ref) and
`push --tree <name>` the working tree, which sends uncommitted work too. §5
covers that flow.

```bash
ci/scripts/gpu-dev.sh ls                      # every session this machine started
```

Prints session/pod/ref/arch@host via `rp_session_list`
(`ci/scripts/runpod_lib.sh:522`), with
`column widths are derived from the rows rather than fixed`
(`ci/scripts/runpod_lib.sh:519`) — a ref is a branch name or a 40-character
commit id, so every fixed width is eventually too narrow.

```bash
ci/scripts/gpu-dev.sh attach a100             # join the running job, or a plain shell
```

`attach` (`ci/scripts/gpu-dev.sh:881`) joins the tree's own tmux job pane if
one is running, else a plain login shell: `rp_login_cmd`
(`ci/scripts/gpu-dev.sh:663`) emits either
`exec tmux attach -t` (`ci/scripts/gpu-dev.sh:685`) or a plain `'exec bash -i'`
(`ci/scripts/gpu-dev.sh:687`). `--shell` forces the plain prompt even with a
job running — `MODE=job; [ "${1:-}" = "--shell" ] && MODE=shell`
(`ci/scripts/gpu-dev.sh:886`).

---

## 4. Runbook B — one-time seed

**When it runs.** `up`/`shell` kick it off *detached* immediately after
bootstrap, unconditionally, so it never blocks your shell —
`start_seed_build` (`ci/scripts/gpu-dev.sh:762`), called from
`[ -n "$RP_REF" ] && start_seed_build` (`ci/scripts/gpu-dev.sh:785`) on the
`shell` path and `[ -n "$RP_REF" ] && start_seed_build`
(`ci/scripts/gpu-dev.sh:869`) on the `up` path. It runs under the shared
timing lock, acquired as the first thing inside the detached tmux pane —
`flock -n -E 75 /root/.jammi-timing.lock`
(`ci/scripts/gpu-dev.sh:768`) — so the lock's lifetime is the seed job's
lifetime, not the short-lived SSH invocation that started tmux.

**Watch it:**

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100
tmux attach -t =jammi-seed          # detach: Ctrl-B then D
```

or tail `/root/.jammi-seed.log` directly.

**The phases, in order**, all inside `pod_seed_target_main`
(`ci/scripts/pod_seed_target.sh:634`):

1. **`--no-lock` re-exec.** Absent `--no-lock`, the script re-execs itself
   through `exec "$DIR/pod_timing_lock.sh" acquire -w`
   (`ci/scripts/pod_seed_target.sh:661`), waiting up to
   `JAMMI_SEED_LOCK_WAIT_SECS="${JAMMI_SEED_LOCK_WAIT_SECS:-1800}"`
   (`ci/scripts/pod_seed_target.sh:58`). `start_seed_build` already runs
   `--no-lock` inside its own tmux+flock wrapper
   (`flock -n -E 75 /root/.jammi-timing.lock`,
   `ci/scripts/gpu-dev.sh:768`), so this arm is for a caller invoking the
   script directly over SSH.
2. **Marker gate.** If `.jammi-seed-complete` already exists and `--reseed`
   was not given, the whole run is a no-op —
   `seed already complete ($COMPLETE_MARKER)`
   (`ci/scripts/pod_seed_target.sh:686`). If `.jammi-seed-failed` exists it
   refuses to retry automatically and prints the failure tail —
   `seed previously FAILED ($FAILED_MARKER)`
   (`ci/scripts/pod_seed_target.sh:690`). `--reseed` forces either case, and
   removes BOTH markers before the rebuild starts:
   `rm -f "$FAILED_MARKER" "$COMPLETE_MARKER"`
   (`ci/scripts/pod_seed_target.sh:703`).
3. **Required-tools preflight** — `pod_seed_assert_required_tools || return 1`
   (`ci/scripts/pod_seed_target.sh:678`), §2 item 4 above.
4. **`cargo metadata --locked` priming, network allowed, exactly once** —
   `cargo metadata --locked --format-version 1`
   (`ci/scripts/pod_seed_target.sh:777`). This is load-bearing: every later
   `--frozen` metadata call in this script needs the *full* cross-platform
   dependency graph already fetched (`cargo metadata` without
   `--filter-platform` walks platform-conditional crates `cargo build` alone
   never fetches). Skipping this step is what produced the a100c incident in
   §8, row 8.
5. **T1** — `cargo build --release -p jammi-bench --features cuda`
   (`ci/scripts/pod_seed_target.sh:786`).
6. **T1b** — `cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn`
   (`ci/scripts/pod_seed_target.sh:831`), **main-only**, gated on the
   checkout's *resolved sha* matching `origin/main` (or
   `JAMMI_SEED_IS_MAIN=1`), never on `abbrev-ref == "main"` — a
   checkout-by-sha always leaves a detached HEAD. The resolution is
   `_seed_main_sha="$(git rev-parse --verify --quiet origin/main`
   (`ci/scripts/pod_seed_target.sh:735`), compared at
   `_seed_main_reason="resolved sha matches origin/main"`
   (`ci/scripts/pod_seed_target.sh:744`); see §8 row 3.
   `pod_seed_pkg_has_feature`
   (`ci/scripts/pod_seed_target.sh:319`) detects flash-attn live via
   `cargo metadata`, never hand-asserted; its rc=2 ("could not determine")
   **aborts the whole seed** rather than silently skipping T1b —
   `refusing to guess 'absent'`
   (`ci/scripts/pod_seed_target.sh:847`). A broken metadata query read as
   "absent" is exactly how a seed once stamped complete without its FA2
   artifacts. When T1b's prerequisite checkout is missing, the seed
   provisions it first —
   `git submodule update --init --force --checkout --depth 1`
   (`ci/scripts/pod_seed_target.sh:825`).
7. **T2** — `cargo test --no-run` for the exact crates/features
   `runpod_gpu_prove.sh`'s own CI suites use, kept in lockstep by naming
   the same `-p`/`--features`/`--test` here:
   `cargo test -p jammi-server --features cuda,live-gpu-tests --test it --no-run`
   (`ci/scripts/pod_seed_target.sh:853`) through
   `cargo test -p jammi-kernels --features cuda --no-run`
   (`ci/scripts/pod_seed_target.sh:856`).
8. **T3** — `cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings`
   (`ci/scripts/pod_seed_target.sh:859`).
9. **Capture build-script stdout before cleaning** —
   `pod_seed_capture_build_output`
   (`ci/scripts/pod_seed_target.sh:255`), called once per profile at
   `pod_seed_capture_build_output "$JAMMI_SEED_DIR" "$capture" debug`
   (`ci/scripts/pod_seed_target.sh:862`) and
   `pod_seed_capture_build_output "$JAMMI_SEED_DIR" "$capture" release`
   (`ci/scripts/pod_seed_target.sh:863`). Cargo's own cleaner removes
   `build/<pkg>-*/output`, so this is the only chance to read it.
10. **Member-free clean**: `cargo clean --workspace --frozen`
    (`ci/scripts/pod_seed_target.sh:866`) and
    `cargo clean --workspace --release --frozen`
    (`ci/scripts/pod_seed_target.sh:867`), plus an explicit
    `rm -rf "${JAMMI_SEED_DIR}"/*/incremental`
    (`ci/scripts/pod_seed_target.sh:868`) — cargo's own cleaner does not
    remove `incremental/build_script_build-*`.
11. **Non-member path/patch package check** via `cargo metadata`: no
    `source: null` package outside `workspace_members` —
    `asserting no non-member path/patch package (cargo metadata)`
    (`ci/scripts/pod_seed_target.sh:876`), which fails at
    `non-member path/patch package(s) with source=null`
    (`ci/scripts/pod_seed_target.sh:883`).
12. **Filesystem-level member-free assertion** —
    `pod_seed_assert_member_free`
    (`ci/scripts/pod_seed_target.sh:393`), invoked at
    `pod_seed_assert_member_free "$JAMMI_SEED_DIR" "$JAMMI_TREE_DIR" || exit 1`
    (`ci/scripts/pod_seed_target.sh:888`) — the *only* mechanical,
    always-on check that no `jammi-*`-named entry (hyphenated,
    underscored, or `lib`+underscored — the compiled-library naming form)
    survives under `{debug,release}/{.fingerprint,deps,build,incremental}`.
13. **Env-surface cross-check** — `pod_seed_check_stdout_subset`
    (`ci/scripts/pod_seed_target.sh:223`), invoked at
    `pod_seed_check_stdout_subset "$capture" "$MANIFEST" || {`
    (`ci/scripts/pod_seed_target.sh:891`): every
    `cargo:rerun-if-env-changed=<NAME>` the captured build-script output
    actually announced must be listed in
    `ci/scripts/pod_seed_key_inputs.toml` (§9 below). A captured file that
    is legitimately zero bytes (a build script with nothing `cargo:`-shaped
    to print) is **not** an error on its own —
    `cargo creates a` (`ci/scripts/pod_seed_target.sh:209`) such a file for
    every build script it runs. Only zero *captured files total* is an
    error: `cross-check saw no build-script output at all (capture_count=0)`
    (`ci/scripts/pod_seed_target.sh:237`), see §8 row 9.
14. **Completion marker** — `${JAMMI_SEED_DIR}.jammi-seed-complete`, JSON:
    `ref`, `sha`, `date`, `tuples` (`["T1","T2","T3"]` plus `"T1b"` iff it
    ran), `rustflags`, `size_bytes`, `manifest_sha256`,
    `"t1b_flash_attn_ran": t1b_ran, "t1b_flash_attn_reason": sys.argv[8],`
    (`ci/scripts/pod_seed_target.sh:919`), written by
    `> "$COMPLETE_MARKER"` (`ci/scripts/pod_seed_target.sh:922`).

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
with 20 lines of trailing context, and the last 40 lines — all written by
`pod_seed_write_failure_marker` (`ci/scripts/pod_seed_target.sh:584`). Read
it:

```bash
ssh -F ~/.config/runpod/ssh_config jammi-a100 \
  "cat /root/.jammi-seed.jammi-seed-failed"
```

Re-running without `--reseed` refuses and reprints the tail —
`seed previously FAILED ($FAILED_MARKER)`
(`ci/scripts/pod_seed_target.sh:690`) — this is deliberate, so `up`/`shell`
never silently burn real
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
`bc27e75`; the script never runs in CI — `POD, never in CI`
(`ci/scripts/perf/pod_build_timings.sh:3`), in its own module doc). It pins
the per-job walls:
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
itself.** A tree is populated ONLY by `push`, as `rp_tree_dir`
(`ci/scripts/runpod_lib.sh:644`) states in its own doc:
`A tree is populated by` (`ci/scripts/runpod_lib.sh:637`) that verb's rsync
and nothing else. `--with-cutlass` provisions cutlass INTO an
already-pushed tree (`pod_provision_cutlass.sh`); against a tree that has
never been pushed it REFUSES with the tool's own error text —
`does not exist — push to it first`
(`ci/scripts/pod_provision_cutlass.sh:106`). Push before
`target --with-cutlass`:

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

Under the hood this is
`bash ${STAGE_DIR}/pod_target_clone.sh /root/.jammi-seed`
(`ci/scripts/gpu-dev.sh:1211`), which:

1. **Refuses without the seed's completion marker** (exit 3) —
   `refusing to clone: no seed at`
   (`ci/scripts/pod_target_clone.sh:132`). No seed, no clone: this is the
   "did the seed actually finish" gate a caller cannot bypass by accident.
2. **The copy**: `cp -a --reflink=auto "$SEED_DIR" "$DEST_DIR"`
   (`ci/scripts/pod_target_clone.sh:151`) — CoW where the filesystem
   supports it, a real copy otherwise; never a hardlink (a hardlink clone
   was reproduced to corrupt the seed itself,
   `writing through a hardlinked path mutates the seed's own copy`
   (`ci/scripts/pod_target_clone.sh:9`), so this script
   `never hardlinks.` (`ci/scripts/pod_target_clone.sh:10`)).
3. **Unconditional member-freedom check on the clone** —
   `if ! pod_seed_assert_member_free "$DEST_DIR" "$TREE_DIR_FOR_METADATA"; then`
   (`ci/scripts/pod_target_clone.sh:170`) — if the seed was *not* actually
   member-free, the clone is deleted and the call fails right here, rather
   than surfacing later as a mysterious stale-artifact bug.
4. **`--verify` (opt-in, after your own first build):** pipe a `cargo
   build -v` log on stdin; asserts no line matches
   `grep -Eq '^[[:space:]]*Fresh[[:space:]]+jammi-'`
   (`ci/scripts/pod_target_clone.sh:71`) — a member-free seed means every
   `jammi-*` unit must actually *compile* (not report `Fresh`) on the
   clone's first build. This is additional to, never a substitute for, the
   unconditional check in step 3 — `--verify` only ever runs when a human
   remembers to run it and only catches units Fresh on *this one* build.

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
   destination — `refusing to clone over it`
   (`ci/scripts/pod_target_clone.sh:142`).

   `if [ "${RP_ALLOW_COLD_TARGET:-0}" != "1" ]; then`
   (`ci/scripts/gpu-dev.sh:912`) is the sole override for proceeding without
   a marker (`pod_build_timings.sh`'s own measurement legs bypass `run`
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
`scratchpad`, and the cutlass submodule — `pod_push_excludes`
(`ci/scripts/pod_push_stamp.sh:163`) is the *one* place this list lives, and
`exclude set is defined ONCE, in pod_push_stamp.sh`
(`ci/scripts/gpu-dev.sh:1088`) says why: the real rsync and the stamp's own
manifest hash both read it from that function
(`done < <("$DIR/pod_push_stamp.sh" excludes)`,
`ci/scripts/gpu-dev.sh:1099`), so they can never drift apart. The rsync
itself passes
`rsync -azc --no-times --no-owner --no-group --delete`
(`ci/scripts/gpu-dev.sh:1114`) — see §8 row 7 for why `--no-owner
--no-group` is load-bearing, not cosmetic. Immediately after,
`pod_push_stamp.sh compute <repo-root> <session>` writes
`<tree>/.jammi-push-stamp.json`:

- `laptop_head` — `git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown`
  (`ci/scripts/pod_push_stamp.sh:255`): a real sha, or the literal
  `"unknown"` when `repo-root` is not a git checkout at all.
- `porcelain_sha256` / `diff_head_sha256` — hashes of `git status
  --porcelain` / `git diff HEAD`, **not** silently `sha256("")` when either
  git command itself fails in a real repo. The discriminator is whether
  `HEAD` itself resolved — `if [ "$head" = "unknown" ]; then`
  (`ci/scripts/pod_push_stamp.sh:275`) — and a real repo whose status
  command still fails raises
  `'git status --porcelain' failed in a real repo`
  (`ci/scripts/pod_push_stamp.sh:281`); see §8 row 4.
- `manifest_sha256` — sha256 over the sorted `(path, mode, content-sha256)`
  triples of exactly what the same exclude set would push, computed
  entirely locally against a fresh empty temp directory by
  `pod_push_manifest_sha256` (`ci/scripts/pod_push_stamp.sh:196`) via
  `rsync -a --dry-run --out-format='%n'`
  (`ci/scripts/pod_push_stamp.sh:217`). **Stat-flavour detection** —
  `pod_push_stat_mode` (`ci/scripts/pod_push_stamp.sh:134`), which probes
  once with `if stat --version >/dev/null 2>&1; then`
  (`ci/scripts/pod_push_stamp.sh:137`) — is what makes this deterministic
  across GNU and BSD hosts; see §8 row 6.
- `cutlass_gitlink` —
  `cutlass_gitlink="$(git -C "$repo" rev-parse --verify --quiet`
  (`ci/scripts/pod_push_stamp.sh:321`), `null` if the path is not a gitlink
  at HEAD. `--verify --quiet` specifically — see §8 row 5 for why the bare
  form is unsafe here.

`check_cuda_run_artifacts.py`'s `git_sha` rule is **unchanged** by any of
this: the push stamp is iteration provenance only, never a substitute for a
pushed (reachable-from-a-remote-branch) sha on a committed artifact —
`never a substitute for that rule.` (`ci/scripts/pod_push_stamp.sh:9`).

**`JAMMI_BUILD_SHA` — what a binary built on a pushed tree knows about
itself.** `push` excludes `.git`, so a `cargo build` in a pushed tree has no
repository for `crates/jammi-bench/build.rs`'s `git rev-parse` fallback to
read: it bakes `build_sha="unknown"`, and every producer that cross-checks a
binary's own reported provenance then refuses the run. The default closes
this without anyone typing a sha: the job wrapper's env preamble
`rp_job_env_lines` (`ci/scripts/runpod_lib.sh:871`) ends by calling
`rp_job_build_sha_lines "$tree_dir"` (`ci/scripts/runpod_lib.sh:876`), and
`rp_job_build_sha_lines` (`ci/scripts/runpod_lib.sh:915`) reads
`<tree>/.jammi-push-stamp.json` on the pod and runs
`export JAMMI_BUILD_SHA` (`ci/scripts/runpod_lib.sh:941`) — **but only when
that stamp says the pushed tree was CLEAN**, i.e.
`and stamp.get("porcelain_sha256") == clean`
(`ci/scripts/runpod_lib.sh:934`) and the same for `diff_head_sha256`, both
against the digest of the empty string. `push` sends uncommitted work too,
so on a dirty push `laptop_head` names the commit the tree was *based on*,
not the commit it *is*; exporting it there would be a fabricated
provenance, which is worse than `unknown` because a reader cannot detect
it. On a dirty (or absent, or unreadable) stamp the variable is left unset
and the job log carries `JAMMI_BUILD_SHA left UNSET`
(`ci/scripts/runpod_lib.sh:945`) as a `::warning::`.

Two consequences worth knowing:

- **The manual form is still the escape hatch, and still wins.** A
  caller-supplied `JAMMI_BUILD_SHA` is never overwritten. Use it when you
  must measure a dirty tree, and the value **must be the pushed tree's exact
  commit** — a sha the bytes are not is a fabricated provenance, the one
  failure mode this whole path exists to avoid. `build.rs` accepts it
  verbatim iff it is exactly 40 lowercase hex characters, and falls back to
  `unknown` otherwise, so a typo degrades to "unknown" rather than to a
  plausible lie — `is_40_lowercase_hex`
  (`crates/jammi-bench/build.rs:120`), applied at
  `if !is_40_lowercase_hex(&sha) {` (`crates/jammi-bench/build.rs:191`). A
  profiling producer that builds its own binary outside
  `run` carries the same requirement in its own interface, as a *required*
  sha argument rather than an optional one — it has no push stamp to fall
  back to.
- **The first job after a push at a new commit relinks `jammi-bench`.**
  `build.rs` emits
  `println!("cargo:rerun-if-env-changed=JAMMI_BUILD_SHA");`
  (`crates/jammi-bench/build.rs:265`), so changing the value re-runs that
  one build script and relinks that one binary. That
  is the cost of the binary knowing what it is; it is not a workspace
  rebuild.

**ModernBERT-large does not fit at `b=8`/`W=512` on an 80 GB A100.** A leg
table that pre-registers that shape cannot run: every configuration
(f32/f16/bf16 fused and the eager-AdamW control) fails with
`DriverError(CUDA_ERROR_OUT_OF_MEMORY)` out of the LoRA encoder forward, at a
sampled device high-water mark of 75757 MiB of 81920 MiB — on an otherwise
idle GPU (0 MiB used, no other compute apps resident), so this is the model's
own footprint and not contention. `b=4` is the largest shape that fits and is
shared by every leg (`b=2` also clears). Pin the batch **before** writing a
leg table, and pin ONE batch across all legs: a per-leg batch makes the
per-step denominators, and therefore any ratio computed against them,
incomparable across legs.

**`push` ships git-lfs POINTERS unless you materialize them first.**
`push` rsyncs the worktree exactly as it is on disk; it has no git of its
own to resolve anything with (it excludes `.git` outright). A worktree whose
lfs-tracked files are still pointer stubs — the ordinary state of a fresh
`git worktree add` off a shared checkout, since lfs objects are not
materialized per-worktree — therefore sends 130-byte pointer files to the
pod, and the fp16 flash parity suite fails closed on them rather than
computing against garbage:
`this is an UNFETCHED git-lfs pointer file, not the real .npy fixture`
(`crates/jammi-kernels/tests/flash_torch_parity_f16.rs:110`). The tracked
set is declared in the repo-root `.gitattributes`, by the line
`crates/jammi-kernels/tests/fixtures/flash_reference/*.npy  filter=lfs diff=lfs merge=lfs -text`
(cited by its own text, not a line number: `.gitattributes` sits outside
`check_citations.py`'s full-path prefix set, so a line number there would be
a claim nothing re-checks). Before pushing a tree any flash suite will run
in:

```bash
git lfs pull --include='crates/jammi-kernels/tests/fixtures/**'
git lfs ls-files    # the fixtures must be listed as materialized, not pointers
```

**Documented residual:** the pod's own bootstrap `git clone` has the same
gap — nothing in `ci/scripts/runpod_lib.sh` or `ci/scripts/gpu-dev.sh`
fetches lfs objects, so a suite run against the bootstrap checkout rather
than a pushed tree depends on the image's own git-lfs configuration. Only
the two cookbook workflows set `lfs: true` (`.github/workflows/cookbook-book.yml:78`)
on their checkout today. It is not claimed closed; the fail-closed panic
above is what keeps it from being silent.

**Getting a model checkpoint onto the pod.** The CUDA CI image ships
`python3` 3.12 with **no `pip` module and no `hf` on PATH**, so the usual
one-liner (`pip install -U huggingface_hub && hf download …`) does not run
there at all. Bootstrap a package manager first, either into the user site
directory:

```bash
python3 -m ensurepip --user
python3 -m pip install --user -U huggingface_hub
```

or into a throwaway virtualenv, which keeps the image's own interpreter
untouched:

```bash
python3 -m venv /root/hfvenv
/root/hfvenv/bin/pip install -U huggingface_hub
```

Then fetch the checkpoint (use `/root/hfvenv/bin/hf` for the venv form):

```bash
hf download answerdotai/ModernBERT-large --local-dir /root/checkpoints/ModernBERT-large
```

Verify before you spend GPU time on it: `config.json`,
`model.safetensors`, and `tokenizer.json` must all be present, and the
weights must hash to

```
44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f  model.safetensors
```

```bash
sha256sum /root/checkpoints/ModernBERT-large/model.safetensors
```

A partial or LFS-pointer download passes "the file exists" and fails only
much later, inside a run — check the digest, not the directory listing.

**`--with-cutlass`**, concretely — `pod_provision_cutlass.sh
<tree-source-dir> [super-dir]` (defaults `/root/jammi-ai`), the *one*
provisioning surface for cutlass in any tree:
`ONE provisioning surface for cutlass in ANY tree`
(`ci/scripts/pod_provision_cutlass.sh:16`), never a second, independent
`git submodule update --init` run in-tree.

1. `if ! git -C "$SUPER_DIR" submodule update --init --depth 1 "$CUTLASS_PATH"; then`
   (`ci/scripts/pod_provision_cutlass.sh:114`) — network required.
2. **Content-floor validation**: the checkout must have `.git` (or a
   gitlink file), `include/cutlass/cutlass.h`
   (`[ -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ] \`,
   `ci/scripts/pod_provision_cutlass.sh:138`), and an on-disk file count
   at least its pinned commit's own
   `CUTLASS_TREE_FILE_COUNT="$(git -C "$CUTLASS_DIR" ls-tree -r HEAD --name-only`
   (`ci/scripts/pod_provision_cutlass.sh:140`), compared against
   `CUTLASS_DISK_FILE_COUNT="$(find "$CUTLASS_DIR" -type f`
   (`ci/scripts/pod_provision_cutlass.sh:141`) — this catches a
   technically-valid `.git`/HEAD sitting over an empty/partial checkout (a
   real a100e incident: an unrelated push deleted the superproject's own
   submodule content out from under it).
3. **Self-target guard**: refuses as a no-op (exit 0) if `super-dir` and
   `tree-source-dir` are the identical path —
   `source-tree-dir and super-dir are the SAME path`
   (`ci/scripts/pod_provision_cutlass.sh:153`); `rm -rf` would otherwise
   delete the source before `cp -a` could read it.
4. **Expected pin**, decided by the destination tree's own shape: a tree
   that is itself a real git checkout (its own `.git`, `.gitmodules`
   declaring the path) reads its own live `HEAD:<path>` gitlink —
   `PIN_SOURCE="the tree's own git index (git-backed tree)"`
   (`ci/scripts/pod_provision_cutlass.sh:176`); a tree with no `.git` (the
   ordinary `push`-populated case) falls back to
   `.jammi-push-stamp.json`'s `cutlass_gitlink`,
   `PIN_SOURCE="the tree's push stamp"`
   (`ci/scripts/pod_provision_cutlass.sh:183`).
5. On mismatch: fetch+checkout the pinned commit into the *superproject's*
   submodule, re-verify, and only then copy —
   `git -C "$CUTLASS_DIR" fetch --depth 1 origin "$STAMP_SHA"`
   (`ci/scripts/pod_provision_cutlass.sh:199`), reached through a
   `set -e`-EXEMPT `if`/`else` around the check call
   (`if bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA"; then`,
   `ci/scripts/pod_provision_cutlass.sh:190`) rather than a bare command,
   so the remediation arm is reachable (a prior round's bare-command form
   made it dead code; see the file's own module doc for the reproduction).
6. `rm -rf "${TREE_SOURCE_DIR:?}/${CUTLASS_PATH:?}"`
   (`ci/scripts/pod_provision_cutlass.sh:213`), then
   `cp -a "$CUTLASS_DIR" "$TREE_SOURCE_DIR/$CUTLASS_PATH"`
   (`ci/scripts/pod_provision_cutlass.sh:214`), then strip the copied
   `.git` — `rm -rf "${TREE_SOURCE_DIR:?}/${CUTLASS_PATH:?}/.git"`
   (`ci/scripts/pod_provision_cutlass.sh:236`), a bare submodule gitlink
   pointer file that would otherwise create a nested-repository boundary
   inside a git-backed destination tree — and assert it is gone:
   `cutlass/.git still present in the destination tree after stripping`
   (`ci/scripts/pod_provision_cutlass.sh:238`).

**Expected "exactly one PTX rewritten" for a `.cu` edit.** Edit one `.cu`
file, rebuild against the clone, and diff `release/*.ptx` timestamps/hashes
before and after — only the PTX for the kernel whose `.cu` changed should
differ; every other PTX and every third-party `.rlib`/`.rmeta` must be
byte-identical — this is exactly the comparison set `snapshot_hashes`
(`ci/scripts/perf/pod_build_timings.sh:389`) builds, §6 below. **No
committed timing number for
this edit's wall-clock exists in this tree** (see §4's time-budget note);
budget for one `nvcc` invocation plus a relink, not a full rebuild.

**The FA2 leg's own separate clone dir.** `pod_build_timings.sh` clones the
FA2 measurement into `CLONE_FA2_DIR="/root/.jammi-clone-fa2-a2"`
(`ci/scripts/perf/pod_build_timings.sh:453`), deliberately **never**
`CLONE_DIR="/root/.jammi-clone-a2"`
(`ci/scripts/perf/pod_build_timings.sh:314`) — T1's own byte-hash snapshot
is taken immediately after T1, at
`clone_hashes="$(snapshot_hashes "$CLONE_DIR")"`
(`ci/scripts/perf/pod_build_timings.sh:400`), before the FA2 leg touches
anything (`this leg reads that snapshot, never re-takes`,
`ci/scripts/perf/pod_build_timings.sh:510`): reusing one clone dir for both
would make T1's snapshot include FA2-only artifacts, guaranteeing a
spurious byte-equality mismatch against the cold leg.

---

## 6. Runbook D — measure

**`pod_timing_lock.sh` — acquire/release semantics.**

```
pod_timing_lock.sh acquire -n         -- <cmd...>   # refuse immediately if held
pod_timing_lock.sh acquire -w <SECS>  -- <cmd...>   # wait up to SECS
```

— the two forms this script accepts, `pod_timing_lock.sh acquire -n`
(`ci/scripts/pod_timing_lock.sh:14`) and
`pod_timing_lock.sh acquire -w <SECS>`
(`ci/scripts/pod_timing_lock.sh:15`). The lock is a single pod-wide `flock`
at `LOCK="${JAMMI_TIMING_LOCK:-/root/.jammi-timing.lock}"`
(`ci/scripts/pod_timing_lock.sh:62`), kernel-owned: it dies the
`INSTANT the holding process exits or is killed`
(`ci/scripts/pod_timing_lock.sh:9`), so there is nothing to "steal" and
nothing stale to clean up. A refusal (held elsewhere, or the wait timed out)
exits **75** — `Exit 75 = the lock was refused`
(`ci/scripts/pod_timing_lock.sh:33`), the same "resource unavailable, try
again" code `rp_deploy_live` uses for a capacity skip, never a code meaning
"this is broken"; it is applied by
`exec flock "${flock_args[@]}" -E 75 "$LOCK" bash -c '`
(`ci/scripts/pod_timing_lock.sh:106`). The holder file (`<lock>.holder`) is
written tmp-then-rename *under* the lock, never truncate-in-place —
`} > "$tmp" && mv -f "$tmp" "$holder"`
(`ci/scripts/pod_timing_lock.sh:113`), so a reader racing the write can only
ever see the old-complete or new-complete file — and removed on release via
a trap set *inside* the flock-held child:
`trap "rm -f \"$holder\"" EXIT INT TERM`
(`ci/scripts/pod_timing_lock.sh:114`).

**The concurrent `-n` case.** The automatic seed build in `start_seed_build`
(`ci/scripts/gpu-dev.sh:762`) still builds
`flock -n -E 75 /root/.jammi-timing.lock`
(`ci/scripts/gpu-dev.sh:768`) directly into its own detached tmux pane's
command line rather than going through this script — acquiring the lock is
the *first* thing that happens inside the pane itself, so the lock's
lifetime is the job's lifetime, not the short-lived SSH invocation that
started tmux and returned in under a second. `run --timing` moved its OWN
acquisition *inside* the generated `.jammi-job.sh` wrapper instead:
`rp_job_wrapper_with_marker_lines`
(`ci/scripts/runpod_lib.sh:1026`) emits the fd-based
`if ! flock -n 9; then` (`ci/scripts/runpod_lib.sh:1051`), essentially the
first real step of that script too, for the identical M6 reasoning — rather
than split across an outer `flock -n -E 75 ... bash job.sh` command line,
which is why `LAUNCH`'s own if/else,
`if [ "$TIMING" = "1" ]; then` (`ci/scripts/gpu-dev.sh:1022`), now emits a
plain redirection on the timing path —
`LAUNCH="bash '${TREE_DIR}'/.jammi-job.sh > '${TREE_DIR}'/.jammi.log 2>&1"`
(`ci/scripts/gpu-dev.sh:1023`) — beside the untimed
`LAUNCH="bash '${TREE_DIR}'/.jammi-job.sh 2>&1 | tee '${TREE_DIR}'/.jammi.log"`
(`ci/scripts/gpu-dev.sh:1025`), so a lock refusal and the job's own real
exit code can never collide on the literal value 75
(round-N audit finding B3). Either shape, a second `run --timing`
(or `pod_build_timings.sh` itself, or a manual `pod_seed_target.sh`
invocation) against an already-held lock refuses at once with exit 75,
naming the current holder from the holder file — it never silently queues.

**`pod_build_timings.sh` legs**, in order
(`ci/scripts/perf/pod_build_timings.sh`):

- **(i) seed** — `echo "::group::(i) seed"`
  (`ci/scripts/perf/pod_build_timings.sh:227`) runs
  `"$CI_SCRIPTS/pod_seed_target.sh" --no-lock`
  (`ci/scripts/perf/pod_build_timings.sh:228`), asserts the completion
  marker, re-runs
  `pod_seed_assert_member_free "$JAMMI_SEED_DIR" "$JAMMI_TREE_DIR" || fail`
  (`ci/scripts/perf/pod_build_timings.sh:234`) as an independent witness,
  and records `S_seed_bytes="$(du -sk "$JAMMI_SEED_DIR"`
  (`ci/scripts/perf/pod_build_timings.sh:235`) plus the seed's own
  `t1b_flash_attn_ran`/`reason`.
- **source-tree size** — `S_src_bytes="$(du -sk --exclude=.git`
  (`ci/scripts/perf/pod_build_timings.sh:270`).
- **(ii) clone build at the FA2 tip** —
  `echo "::group::(ii) clone build @ ${JAMMI_FA2_TIP_REF}"`
  (`ci/scripts/perf/pod_build_timings.sh:286`) runs
  `git fetch --all --tags --prune --quiet`
  (`ci/scripts/perf/pod_build_timings.sh:287`) and
  `git checkout --quiet "$JAMMI_FA2_TIP_REF"`
  (`ci/scripts/perf/pod_build_timings.sh:289`) — which requires a git-backed
  tree with a reachable `origin`; a purely `push`-populated tree cannot run
  this leg, see §8 row 1 — provisions cutlass if `.gitmodules` declares it,
  clones the seed at
  `"$CI_SCRIPTS/pod_target_clone.sh" "$JAMMI_SEED_DIR" "$CLONE_DIR"`
  (`ci/scripts/perf/pod_build_timings.sh:317`), times
  `cargo build --release -p jammi-bench --features cuda -v`
  (`ci/scripts/perf/pod_build_timings.sh:333`), and records
  `copy_wall=$((copy_t1 - copy_t0))`
  (`ci/scripts/perf/pod_build_timings.sh:319`),
  `clone_wall=$((clone_t1 - clone_t0))`
  (`ci/scripts/perf/pod_build_timings.sh:336`), and whether the copy
  reflinked (`reflink_took="no"`,
  `ci/scripts/perf/pod_build_timings.sh:320`).
- **(iii) sccache** — `sccache_before="$(sccache --show-stats`
  (`ci/scripts/perf/pod_build_timings.sh:328`) and
  `sccache_after="$(sccache --show-stats`
  (`ci/scripts/perf/pod_build_timings.sh:339`), labelled
  `wrapper is off (CARGO_BUILD_RUSTC_WRAPPER=)`
  (`ci/scripts/perf/pod_build_timings.sh:506`) — unchanged by construction,
  since the wrapper is off pod-wide.
- **(iv) byte-equality vs a cold build** —
  `echo "::group::(iv) cold build @ a separate, genuinely-empty CARGO_TARGET_DIR"`
  (`ci/scripts/perf/pod_build_timings.sh:512`) builds into
  `COLD_DIR="/root/.jammi-cold-a2"`
  (`ci/scripts/perf/pod_build_timings.sh:514`), a *genuinely empty* second
  `CARGO_TARGET_DIR` — `A genuinely EMPTY directory`
  (`ci/scripts/perf/pod_build_timings.sh:515`), never a copy-then-clean of
  the clone, which would still share every third-party artifact. The
  comparison is scoped to `release/*.ptx` +
  `release/**/libjammi_*.{rlib,rmeta}` only, by
  `find "$1/release" -type f`
  (`ci/scripts/perf/pod_build_timings.sh:395`); see §8 row 10 for why the
  linked binary is excluded. **Four-state result** (`byte_equal_state`):
  `invalid` (empty match set on either side — the comparison never ran),
  `set_mismatch` (both sides non-empty but the paths present differ),
  `true`, `false` — never collapsed into a bare boolean.
- **(v) sizes and copy mechanics** (`S_src_bytes`/`S_seed_bytes`/
  `S_clone_bytes`/`copy_wall_s`/`reflink`) — the `RP_DISK_GB` formula's
  producer (§2 item 3).
- **(vi) clone vs cold wall-clock, plus the FA2 leg separately** — both
  real numbers land in the same-run JSON so a reader computes any delta
  they want; the FA2 leg builds
  `cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn`
  (`ci/scripts/perf/pod_build_timings.sh:459`) and is gated on the
  *resolved* HEAD sha, `_head_sha="$(git rev-parse HEAD)"`
  (`ci/scripts/perf/pod_build_timings.sh:433`), matching `origin/main` (or
  `JAMMI_MAIN_SHA`), always recording `fa2_ran="false"`
  (`ci/scripts/perf/pod_build_timings.sh:405`) or `fa2_ran="true"`
  (`ci/scripts/perf/pod_build_timings.sh:467`) with a reason even when
  skipped (see §8 row 3 — the class this fixes).

`JAMMI_BUILD_TIMINGS_OUT` (required env var) is where the JSON is written —
**never stdout**, since this script also prints progress markers there:
`never stdout, since this script also prints progress markers to stdout`
(`ci/scripts/perf/pod_build_timings.sh:133`). Run it:

```bash
JAMMI_FA2_TIP_REF=<ref-or-sha> JAMMI_BOX_LABEL='a100d (A100 PCIe, driver 570)' \
JAMMI_BUILD_TIMINGS_OUT=/root/pod-build-timings.json \
  bash ci/scripts/perf/pod_build_timings.sh
```

then copy the JSON to `ci/artifacts/pod-build-timings/<ts>-<sha7>.json` and
commit it — `copy it to ci/artifacts/pod-build-timings/<ts>-<sha7>.json and commit it`
(`ci/scripts/perf/pod_build_timings.sh:726`), the script's own closing
instruction. The first such committed run is
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
and could never fail — `a self-check between two constants`
(`ci/scripts/perf/pod_build_timings.sh:576`). The real like-for-like
guarantee is `byte_equal_state`, described above.

---

## 7. Runbook E — tear down

```bash
ci/scripts/gpu-dev.sh down a100               # lead/user only
```

`down` (`ci/scripts/gpu-dev.sh:1246`) never trusts the locally-recorded pod
id on its own: it calls `rp_pod_verify "$RP_POD_ID" >/dev/null`
(`ci/scripts/gpu-dev.sh:1258`), and `rp_pod_verify`
(`ci/scripts/runpod_lib.sh:349`) confirms the id is **both** still present
in the account's live pod list **and** named like this tooling's own pods —
`shape = re.compile(r"^%s-ttl[0-9]+$" % re.escape(prefix))`
(`ci/scripts/runpod_lib.sh:367`):

- **Match** → `rp_terminate "$RP_POD_ID"` (`ci/scripts/gpu-dev.sh:1261`),
  then **confirm gone** by re-querying the account —
  `if rp_pod_gone "$RP_POD_ID"; then` (`ci/scripts/gpu-dev.sh:1268`), whose
  implementation is `rp_pod_gone` (`ci/scripts/runpod_lib.sh:404`) — before
  forgetting the local record. A rejected `podTerminate` must never both
  leak the pod and destroy the only record pointing at it, so the
  unconfirmed arm keeps it: `the local session record is KEPT — retry`
  (`ci/scripts/gpu-dev.sh:1276`).
- **Absent from the account entirely** (return code 3) → this is *not* a
  refusal, it is the ordinary shape of "this pod already ended on its own"
  (its own TTL, or the sweep) — `elif [ "$verify_rc" -eq 3 ]; then`
  (`ci/scripts/gpu-dev.sh:1279`), and `down` forgets the record and says so:
  `is gone from the account (deadline/sweep) — forgetting the record`
  (`ci/scripts/gpu-dev.sh:1294`).
- **Present under a name that is not this tooling's shape, or the query
  itself failed** → refuse (exit 1), *keep* the local record —
  `refusing to terminate pod` (`ci/scripts/gpu-dev.sh:1307`). This is
  exactly the ambiguous case a follow-up `up` most needs to still see and
  refuse on.

**The id, not the TTL, is authoritative** — `Two earlier versions tried`
(`ci/scripts/runpod_lib.sh:293`) to make the exact TTL part of this check,
and both were removed: one as a false-refusal source, the other because it
was `found INERT on every input` (`ci/scripts/runpod_lib.sh:303`). The full
history is in that comment. RunPod pod ids are globally unique, so
name-shape plus a matched id is already sufficient.

**Interrupted / read-only paths that never terminate.** A `down` that
cannot confirm the account state (network failure mid-query, an
unrecognized name shape) exits with the local record intact rather than
guessing — there is no code path in `down` that both fails to confirm *and*
still terminates or forgets the record;
`the local session record is KEPT (not forgotten)`
(`ci/scripts/gpu-dev.sh:1308`) is the whole of that arm's effect. The only
manual escape from that refusal is the RunPod console.

```bash
ci/scripts/gpu-dev.sh reap                    # honour each pod's own deadline
ci/scripts/gpu-dev.sh reap 2                  # force-reap everything older than 2h
```

`reap` is account-wide (never a per-session verb) and judges age from
`Age comes from createdAt, never from runtime.uptimeInSeconds`
(`ci/scripts/runpod_lib.sh:1636`) — uptime is null for the first minutes of
a healthy pod and can reset on a container restart, either of which would
misjudge age. `Every ambiguity resolves toward terminating.`
(`ci/scripts/runpod_lib.sh:1568`): a stopped pod
(`print(p['id'], age if age is not None else -1, 'not-running'); continue`,
`ci/scripts/runpod_lib.sh:1649`) and an unparseable deadline
(`print(p['id'], age, 'unparseable-deadline'); continue`,
`ci/scripts/runpod_lib.sh:1661`) are both swept. The ONE deliberate
exception is a pod with no usable `createdAt` at all: it is reported for an
operator to force-reap, never terminated on a guess —
`has no usable createdAt` (`ci/scripts/runpod_lib.sh:1676`), selected by
`print('UNAGEABLE', p['id'], name); continue`
(`ci/scripts/runpod_lib.sh:1653`). A query that cannot reach RunPod at all
fails loudly rather than reporting "nothing to clean up":
`sweep could NOT enumerate pods` (`ci/scripts/runpod_lib.sh:1669`).

**Reconciling account state when boot output was empty.** If `up`/`shell`
returned with no coordinates (a dropped connection mid-deploy, a killed
laptop), do not assume nothing was rented — `rp_deploy_live` sets
`RP_POD_CREATED=1` (`ci/scripts/runpod_lib.sh:1287`) the *instant* a pod id
comes back from the deploy mutation, minutes before SSH ever comes up
(`THIS is the moment a pod exists and starts billing`,
`ci/scripts/runpod_lib.sh:1278`), so a pod can exist and bill even when the
invocation that rented it never got to print anything. Reconcile via:

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
| 1 | A tree has `.git` | `push` excludes `.git` entirely — `pod_push_excludes` (`ci/scripts/pod_push_stamp.sh:163`) — a pushed tree is a plain rsync'd directory | `git submodule`/any git command on a pushed tree fails "not a git repository" | `ONE provisioning surface for cutlass in ANY tree` (`ci/scripts/pod_provision_cutlass.sh:16`), that file's own module doc; `(m/A1 match) matching stamp/submodule` (`ci/scripts/test_pod_substrate.sh:1601`) and its `(m/A1 drift)`/`(m/A1 deinit)`/`(m/A1 fetch-failure)`/`(m/A1 revert-RED)` siblings |
| 2 | A destination path is either untouched or a real git submodule | It can already hold a **copy-provisioned** cutlass (`cp -a` with `.git` stripped — `rm -rf "${TREE_SOURCE_DIR:?}/${CUTLASS_PATH:?}/.git"`, `ci/scripts/pod_provision_cutlass.sh:236`) | `git submodule update --init` on that path refuses (non-empty, non-submodule-shaped dir) — a live a100c run wasted 819s before failing | `the class this fix closes` (`ci/scripts/pod_provision_cutlass.sh:39`), that file's module doc; provisioning now `rm -rf` + `cp -a` unconditionally, never asks git to touch the destination |
| 3 | `git rev-parse --abbrev-ref HEAD == "main"` on a checkout of main | A checkout-by-sha (the ordinary shape for an FA2 PR tip / a resolved seed sha) always leaves a **detached HEAD**, whose abbrev-ref reads the literal `"HEAD"` | T1b/the FA2 leg silently never ran, with no recorded reason — indistinguishable from "correctly determined not-main" | `_seed_is_main="false"` (`ci/scripts/pod_seed_target.sh:736`); `Gated on the RESOLVED` (`ci/scripts/perf/pod_build_timings.sh:427`); `(n/addendum EXECUTABLE b) DETACHED HEAD at the SAME sha as origin/main` (`ci/scripts/test_pod_substrate.sh:1982`) — gated on the *resolved sha*, `t1b_ran`/`t1b_reason` and `fa2_ran`/`fa2_reason` always recorded |
| 4 | `git status --porcelain`/`git diff HEAD` failing == "no output" == "clean" | A failing git command in a real repo (locked index, corrupted ref) also produces empty stdout, hashing to the identical `sha256("")` a genuinely clean tree would produce | `manifest_sha256`/`porcelain_sha256` reports a byte-identical "clean" hash for a broken repo-root | `'git status --porcelain' failed in a real repo` (`ci/scripts/pod_push_stamp.sh:281`), discriminated on whether `HEAD` itself resolved at `if [ "$head" = "unknown" ]; then` (`ci/scripts/pod_push_stamp.sh:275`) |
| 5 | `git rev-parse HEAD:<path>` on a missing path fails with empty output | The **bare** form echoes its own argument text to stdout (rc=128) — `git rev-parse HEAD:no/such/path` literally prints `HEAD:no/such/path` | A bogus literal string becomes `cutlass_gitlink`/an expected pin, read as a real (but wrong) value | `cutlass_gitlink="$(git -C "$repo" rev-parse --verify --quiet` (`ci/scripts/pod_push_stamp.sh:321`); `git rev-parse HEAD:<gitlink-path>` (`ci/scripts/pod_provision_cutlass.sh:63`) — every call site now uses `--verify --quiet`, silent on a miss |
| 6 | `stat -f FORMAT` means "use this format string" | On GNU coreutils, `-f` means "display file **system** status" (the opposite of BSD) — the old fallthrough printed a multi-line, live `Free:`-block-bearing status report to stdout before failing | `manifest_sha256` diverged nondeterministically between two hosts building the identical bundle (a100c vs a100e) | `pod_push_stat_mode` (`ci/scripts/pod_push_stamp.sh:134`) — flavour detected once via `if stat --version >/dev/null 2>&1; then` (`ci/scripts/pod_push_stamp.sh:137`), memoized |
| 7 | Pushing files as `root@pod` leaves them root-owned | `rsync -a` preserves owner/group from the **laptop's** own uid (e.g. 501) unless told not to | `git`, run as root inside the pushed tree, refuses "detected dubious ownership" | `rsync -azc --no-times --no-owner --no-group --delete` (`ci/scripts/gpu-dev.sh:1114`) |
| 8 | `cargo metadata --frozen` "just works" once `Cargo.lock` exists | `cargo metadata` (unlike `cargo build`) resolves the **full cross-platform** graph by default, needing source for platform-conditional crates never otherwise fetched | Seed pipeline died on "failed to download android_system_properties ... --frozen was specified" *after* T1–T3 had already succeeded | `cargo metadata --locked --format-version 1` (`ci/scripts/pod_seed_target.sh:777`), the one-time network-allowed priming call before every `--frozen` call; `pod_seed_cargo_metadata_frozen` (`ci/scripts/pod_seed_target.sh:306`) captures real stderr, never discards it |
| 9 | A zero-byte captured `build/<pkg>-*/output` file means "captured at the wrong moment" | Cargo creates that file for **every** build script it runs, regardless of whether the script prints anything — a real no-op script legitimately produces zero bytes | An earlier fix flagged legitimate zero-byte captures (chrono-tz, esaxx-rs, pulldown-cmark, rustls, scratch, snap, stacker, prometheus) as errors, aborting every real seed build | `pod_seed_check_stdout_subset` (`ci/scripts/pod_seed_target.sh:223`), whose own doc records that `cargo creates a` (`ci/scripts/pod_seed_target.sh:209`) zero-byte file legitimately; `(l/N4) an unlisted announced var reddens the cross-check` (`ci/scripts/test_pod_substrate.sh:1356`) and `(l/N4 revert-RED) the OLD per-file empty-is-an-error rule` (`ci/scripts/test_pod_substrate.sh:1424`) |
| 10 | Two builds of the identical tree on the same box produce byte-identical linked binaries | mold 2.35.1 / clang 21's ThinLTO codegen embeds local-symbol suffixes (`anon.<h>.N.llvm.<hash>`) that differ between two builds of the **same** tree | `release/jammi-bench` (467 differing symbols) made the byte-equality leg read `false` even though every deterministic artifact (`*.ptx`, `.rlib`/`.rmeta`) matched | `the FINAL LINKED BINARY` (`ci/scripts/perf/pod_build_timings.sh:375`) and `"byte_equal_scope": {` (`ci/scripts/perf/pod_build_timings.sh:700`) — the linked binary is explicitly excluded, never silently dropped from the claim |
| 11 | `push --tree <name>`'s rsync destination is reachable | rsync creates only the LAST path component of its own destination — nothing in the pod bootstrap or the build-substrate seed provisions `/root/trees` itself | The very FIRST `push` for a name no session has ever pushed before fails outright on a fresh pod: `rsync: mkdir "/root/trees/<name>" failed: No such file or directory (2)` (observed live on pod u4hfsqyu0i2qwa) | `rp_push_ensure_parent` (`ci/scripts/runpod_lib.sh:682`), a bounded, idempotent remote `mkdir -p` on the parent called before every push at `rp_push_ensure_parent "$TREE_DIR" \` (`ci/scripts/gpu-dev.sh:1084`); `(y/esc-056) gpu-dev.sh's push case calls rp_push_ensure_parent` (`ci/scripts/test_pod_substrate.sh:3699`) |
| 12 | `gpu-dev.sh`'s `REPO_ROOT` names the checkout the caller means | It is derived from the SCRIPT's own on-disk location, never `$PWD` — a multi-worktree laptop keeps more than one copy simultaneously | Invoking one tree's script copy from inside a DIFFERENT tree silently `push`/`run`/`target`s the WRONG tree; the push-stamp's own `laptop_head` field was the only tell (M1b) | `if [ "${RP_ALLOW_ROOT_MISMATCH:-0}" != "1" ]; then` (`ci/scripts/gpu-dev.sh:293`) — push/run/target refuse on a cwd/`REPO_ROOT` mismatch at `would silently act on ${REPO_ROOT}, NOT the tree you are standing in.` (`ci/scripts/gpu-dev.sh:297`), and `set RP_ALLOW_ROOT_MISMATCH=1 to override` (`ci/scripts/gpu-dev.sh:299`); `(z/esc-056) 'push' from a plain (non-git) mismatched cwd REFUSES` (`ci/scripts/test_pod_substrate.sh:3827`) |

---

## 9. Reference

| Script | Synopsis | Key env vars | Exit codes |
|---|---|---|---|
| `ci/scripts/gpu-dev.sh` | The CLI: `shell`/`up`/`target`/`attach`/`run`/`logs`/`push`/`pull`/`wait-seed`/`wait-job`/`down`/`ls`/`reap` — `# Usage:` (`ci/scripts/gpu-dev.sh:33`), restated for `--help` by `usage()` (`ci/scripts/gpu-dev.sh:119`) | `RUNPOD_API_KEY`, `RP_IMAGE`, `RP_TTL_HOURS`, `RP_DEV_TTL_HOURS`, `RP_DISK_GB`, `RP_VOLUME_GB`, `RP_WAIT_TIMEOUT_SECS`, `RP_WAIT_INTERVAL_SECS`, `RP_WAIT_MAX_TRANSPORT_FAILS` — `# Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,` (`ci/scripts/gpu-dev.sh:95`), restated in `usage()`'s own `Env: RUNPOD_API_KEY (or ~/.config/runpod/key), RP_IMAGE,` (`ci/scripts/gpu-dev.sh:229`) | `0` ok; `2` usage/argument error; `1` a real failure (bootstrap, verify, named wait-seed/wait-job failure); `75` no-capacity neutral skip (propagated from `rp_deploy_live`); `wait-seed`/`wait-job` additionally: `2` transport failure, `3` timed out with no verdict — `A timeout with no verdict either way exits 3.` (`ci/scripts/gpu-dev.sh:228`), implemented by `rp_wait_poll` (`ci/scripts/runpod_lib.sh:1489`) |
| `ci/scripts/runpod_lib.sh` | Shared RunPod primitive — deploy/verify/terminate/sweep/wait-poll; sourced, not run directly — `Shared RunPod GPU primitive — one seam for every caller.` (`ci/scripts/runpod_lib.sh:2`) | `RUNPOD_API_KEY`, `RP_SESSION`, `RP_KEEP`, `RP_TTL_HOURS` (default 8 — `RP_TTL_HOURS="${RP_TTL_HOURS:-8}"`, `ci/scripts/runpod_lib.sh:94`), `RP_DISK_GB` (default 60 — `RP_DISK_GB="${RP_DISK_GB:-60}"`, `ci/scripts/runpod_lib.sh:106`), `RP_SSH_WAIT_SECS` (default 600 — `RP_SSH_WAIT_SECS="${RP_SSH_WAIT_SECS:-600}"`, `ci/scripts/runpod_lib.sh:137`), `RP_WAIT_SSH_BOUND_SECS`, the local kill bound `_rp_bounded_capture` (`ci/scripts/runpod_lib.sh:1434`) applies | `rp_pod_verify`: `0` match, `1` present-wrong-shape, `2` query failed, `3` absent — `Returns 1 (present, but` (`ci/scripts/runpod_lib.sh:315`), implemented at `rp_pod_verify` (`ci/scripts/runpod_lib.sh:349`); `rp_deploy_live`: `0` ok, `1` real deploy fault, `75` no reachable capacity — `(neutral skip) when no candidate yields a reachable pod` (`ci/scripts/runpod_lib.sh:1234`), implemented at `rp_deploy_live` (`ci/scripts/runpod_lib.sh:1236`); `rp_wait_poll`: `0` success, `1` named failure, `2` transport failure, `3` timed out — `Returns 0 (success), 1 (named failure` (`ci/scripts/runpod_lib.sh:1486`) |
| `ci/scripts/pod_seed_target.sh` | Builds/cleans the member-free seed — `Builds and cleans a MEMBER-FREE pod build-substrate seed` (`ci/scripts/pod_seed_target.sh:2`) | `JAMMI_SEED_DIR` (default `/root/.jammi-seed`), `JAMMI_TREE_DIR` (default `/root/jammi-ai`), `JAMMI_SEED_LOCK_WAIT_SECS` (default 1800), `--reseed`, `--no-lock`, `JAMMI_SEED_DRY_RUN` | `0` ok/no-op; non-zero on any build/check failure, writes `.jammi-seed-failed` |
| `ci/scripts/pod_target_clone.sh` | Clones the seed into a fresh `CARGO_TARGET_DIR`; `--verify` checks a piped `cargo build -v` log — `Usage: pod_target_clone.sh <seed-dir> <dest-dir> [tree-dir] [--verify|--adopt]` (`ci/scripts/pod_target_clone.sh:12`) | positional `<seed-dir> <dest-dir> [tree-dir]`, `--verify` | `0` ok; `1` member-free check failed post-clone (clone removed) or verify found a `Fresh jammi-*` unit; `2` destination already exists; `3` no seed at `<seed-dir>` |
| `ci/scripts/pod_provision_cutlass.sh` | The one cutlass provisioning surface into an existing tree — `Provisions cutlass INTO an already-pushed tree` (`ci/scripts/pod_provision_cutlass.sh:2`) | positional `<source-tree-dir> [super-dir]` (default `/root/jammi-ai`) | `0` ok/no-op (self-target); `1` any validation, network, or mismatch failure |
| `ci/scripts/pod_push_stamp.sh` | Exclude-set source of truth + push provenance stamp; `excludes`/`compute`/`cutlass-check` subcommands — `pod_push_stamp.sh excludes` (`ci/scripts/pod_push_stamp.sh:12`), dispatched at `excludes) pod_push_excludes ;;` (`ci/scripts/pod_push_stamp.sh:371`) | — | `compute`: `0` ok, `1` a hashing/tool/git failure; `cutlass-check`: `0` match, `1` genuine mismatch, `2` no usable stamp |
| `ci/scripts/pod_timing_lock.sh` | One flock seam for every pod-side timing-sensitive producer — `One flock seam for every pod-side thing that must not race a` (`ci/scripts/pod_timing_lock.sh:2`) | `JAMMI_TIMING_LOCK` (default `/root/.jammi-timing.lock`), `JAMMI_TIMING_LABEL`, `JAMMI_TIMING_JOB` | `0` ok (wrapped command's own exit code); `2` usage error; `75` lock refused/timed out |
| `ci/scripts/perf/pod_build_timings.sh` | A2 pod-build-substrate acceptance producer; live-pod only, never CI — `A2 producer (pod-build-substrate acceptance, contract v6)` (`ci/scripts/perf/pod_build_timings.sh:2`) | `JAMMI_TREE_DIR`, `JAMMI_SEED_DIR`, `JAMMI_FA2_TIP_REF` (required), `JAMMI_BOX_LABEL` (required), `JAMMI_BUILD_TIMINGS_OUT` (required), `JAMMI_MAIN_SHA` (optional FA2-gate override), `--no-lock` | `0` ok; `1` any leg's `fail` call (named, on stderr) |

**Test suite:**

```bash
bash ci/scripts/test_pod_substrate.sh          # every leg (a)-(z), hermetic, no pod
JAMMI_REQUIRE_LOCK_TEST=1 bash ci/scripts/test_pod_substrate.sh   # additionally requires flock present (Linux-only leg (d))
bash ci/scripts/test_gpu_dev_lifecycle.sh      # gpu-dev.sh's own dispatch/session lifecycle legs
```

`test_pod_substrate.sh`'s assertions **are** the documented invariants — the
table in §8 cites specific legs; the full leg index (each a lettered section
header in the file) covers:

- `(a) + (c): .jammi_env writer + no S3/sccache anywhere in runpod_lib.sh`
  (`ci/scripts/test_pod_substrate.sh:83`).
- `(b) pod_target_clone.sh` (`ci/scripts/test_pod_substrate.sh:156`).
- `(d) pod_timing_lock.sh — flock exclusivity`
  (`ci/scripts/test_pod_substrate.sh:362`), incl. the `run --timing`
  fd-based flock's own shape.
- `(e) key-manifest RED tests (i)/(ii) — against the REAL sources`
  (`ci/scripts/test_pod_substrate.sh:546`).
- `(f) push excludes — pinned + single-sourced`
  (`ci/scripts/test_pod_substrate.sh:678`).
- `(g) no unanchored/unquoted` (`ci/scripts/test_pod_substrate.sh:718`)
  tmux target; window ops use `"=name:"`.
- `(h) exactly two "/root/jammi-ai" literal sites in runpod_lib.sh`
  (`ci/scripts/test_pod_substrate.sh:767`).
- `(i) round-2 audit finding 1: the build-substrate clone (a CARGO_TARGET_DIR)`
  (`ci/scripts/test_pod_substrate.sh:794`) — clone/tree directory namespaces
  stay disjoint, plus `rp_job_wrapper_with_marker_lines`'s own
  token/marker/flock-inside-the-wrapper shape (round-N audit finding B3).
- `(j) round-2 audit finding 2: pod_seed_target.sh's failure arm`
  (`ci/scripts/test_pod_substrate.sh:1182`) — incl. `--reseed` removing a
  stale COMPLETE marker too (finding B2a) — and
  `(k) round-2 audit finding 4: the --no-lock re-exec passes an ARRAY, never`
  (`ci/scripts/test_pod_substrate.sh:1268`).
- `(l) round-3 audit N4: pod_seed_check_stdout_subset's cross-check must not`
  (`ci/scripts/test_pod_substrate.sh:1331`) — env-surface cross-check incl.
  zero-byte captures.
- `(m) round-3 audit N1 / item 1b: pod_push_cutlass_matches`
  (`ci/scripts/test_pod_substrate.sh:1431`) and
  `(m/A1) round-5 audit A1 (the load-bearing finding): pod_provision_cutlass.sh`
  (`ci/scripts/test_pod_substrate.sh:1532`) — cutlass pin comparison and
  provisioning.
- `(n) round-2 audit finding 3 / item 3: pod_seed_pkg_has_feature`
  (`ci/scripts/test_pod_substrate.sh:1801`) — live detection.
- `(o) round-2 item 7 (timings under lock) live witness`
  (`ci/scripts/test_pod_substrate.sh:2186`).
- `(p) round-4 audit A5: the OTHER two unconditional member-freedom call`
  (`ci/scripts/test_pod_substrate.sh:2279`) sites.
- `(q/A2) round-5` (`ci/scripts/test_pod_substrate.sh:2492`) — a real
  two-member cargo workspace fixture for member-freedom.
- `(r/A4) round-4 audit finding (zero coverage for two rounds): byte_equal's`
  (`ci/scripts/test_pod_substrate.sh:2619`) tri/four-state.
- `(s/manifest) round-5 (a100c on-pod A2 run at 80c7f59, real seed FAILURE):`
  (`ci/scripts/test_pod_substrate.sh:2742`) — real a100c seed failure replay.
- `(t) round-5 standing rule — class-shaped tripwire`
  (`ci/scripts/test_pod_substrate.sh:2867`) and
  `(u) round-5 standing rule — claim-tripwire`
  (`ci/scripts/test_pod_substrate.sh:2957`).
- `(v/push) round-5 addendum (coordinator, post-63bf905): pod_push_stamp.sh`
  (`ci/scripts/test_pod_substrate.sh:3025`) — determinism + preflight.
- `(w/esc-050) escape esc-050-seed-t1b-fresh-main-clone-cutlass-unprovisioned`
  (`ci/scripts/test_pod_substrate.sh:3186`) and
  `(x/esc-051) escape esc-051-seed-t3-clippy-tuple-twin-off-merge-path`
  (`ci/scripts/test_pod_substrate.sh:3510`) — the seed-tuple-unguarded class
  closure.
- `(y/esc-056) escape esc-056-pod-substrate-assumes-single-fresh-state,`
  (`ci/scripts/test_pod_substrate.sh:3680`) — `push` provisions its own
  tree's parent directory before rsyncing, on a fresh pod.
- `(z/esc-056) escape esc-056-pod-substrate-assumes-single-fresh-state,`
  (`ci/scripts/test_pod_substrate.sh:3808`) — `push`/`run`/`target` refuse a
  cwd/`REPO_ROOT` mismatch, `RP_ALLOW_ROOT_MISMATCH=1` overrides.
- `(aa/esc-056) escape esc-056-pod-substrate-assumes-single-fresh-state,`
  (`ci/scripts/test_pod_substrate.sh:3908`) — `up` records the session the
  MOMENT the pod id comes back from the deploy mutation, never only after
  the reachability wait, so a failure in that window cannot leave a
  running, billing pod recorded nowhere.

**The key-inputs manifest** (`ci/scripts/pod_seed_key_inputs.toml`) —
`every input that can change the BYTES a pod's build`
(`ci/scripts/pod_seed_key_inputs.toml:3`) scripts emit
`without cargo's OWN fingerprint noticing`
(`ci/scripts/pod_seed_key_inputs.toml:5`). Sections, in file order:
`[cargo]` (`ci/scripts/pod_seed_key_inputs.toml:90`) for Cargo.lock/toml and
the RUSTFLAGS family; the three in-workspace build scripts,
`[jammi_kernels_build_rs]` (`ci/scripts/pod_seed_key_inputs.toml:116`),
`[jammi_wire_build_rs]` (`ci/scripts/pod_seed_key_inputs.toml:153`) and
`[jammi_bench_build_rs]` (`ci/scripts/pod_seed_key_inputs.toml:157`); the
vendored CUDA-toolchain build scripts,
`[bindgen_cuda_0_1_6]` (`ci/scripts/pod_seed_key_inputs.toml:139`) and
`[cudarc]` (`ci/scripts/pod_seed_key_inputs.toml:238`);
`[cc_1_2_57]` (`ci/scripts/pod_seed_key_inputs.toml:172`), hand-enumerated —
the one section the automated scan cannot verify mechanically;
`[toolchain_identities]` (`ci/scripts/pod_seed_key_inputs.toml:226`) for
`rustc -vV`/`cargo -V`/`nvcc --version`/…;
`[non_key]` (`ci/scripts/pod_seed_key_inputs.toml:256`), cargo-derived and
not a human toggle; and `[vendored_non_cuda]`
(`ci/scripts/pod_seed_key_inputs.toml:288`), package-specific opt-in toggles
unrelated to the CUDA toolchain. Extending it requires
`(a) a citation to the reading source (file:line, or a`
(`ci/scripts/pod_seed_key_inputs.toml:15`) vendored crate name+version, and
`(i)/(ii)/(iii) to still pass against the real sources.`
(`ci/scripts/pod_seed_key_inputs.toml:17`).

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
