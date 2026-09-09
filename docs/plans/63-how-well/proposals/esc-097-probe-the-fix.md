# PROPOSAL (human-merge): tighten(lead-gate) — relays must probe the fix, not just the class's neighbourhood (esc-097)

Status: PROPOSED — this unit's own commit ships the fixtures (`ci/scripts/check_lead_gate.py`,
agent-writable) RED against the current hook, plus the FULL patch for the five files under
`.claude/hooks/**` / `.claude/agents/*.md` (agent-write-DENIED — a human applies them) as real,
tracked patch files under `docs/plans/63-how-well/proposals/esc-097/` (never as fenced diffs in
this doc — a nested ` ```json ` fence inside the `lead.md` diff terminates a fenced-diff block
early, and whitespace-only context lines get stripped by prose rendering; both defects were found
by trying to apply the fenced form and are why this delivery shape changed). Gate-definition
change: `SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `ci/scripts/check_lead_gate.py`,
`.claude/agents/*.md`) → proposal only, human admin-merge. Nothing under `.claude/hooks/**` or
`.claude/agents/*.md` has been applied to this repo yet.

## Applying the patch (human step)

```
git apply --check docs/plans/63-how-well/proposals/esc-097/*.patch && \
  git apply docs/plans/63-how-well/proposals/esc-097/*.patch
python3 ci/scripts/check_lead_gate.py --self-test
```

The five patch files, git-format unified diffs against `main` `065b72fc`, each touching exactly
one of the agent-write-denied paths:

- `0001-lead-gate-lib.patch` → `.claude/hooks/lead-gate-lib.py`
- `0002-hooks-readme.patch` → `.claude/hooks/README.md`
- `0003-lead-agent.patch` → `.claude/agents/lead.md`
- `0004-adversarial-audit-agent.patch` → `.claude/agents/adversarial-audit.md`
- `0005-pressure-tester.patch` → `.claude/agents/pressure-tester.md`

Proven in a THROWAWAY copy of the tree (`cp -R`, outside any git worktree of the repo, deleted
after use — never `git worktree add`, so the copy carries no live link back to this repo's object
store or refs): `git apply --check` is silent for all five, `git apply` applies all five cleanly,
and `python3 ci/scripts/check_lead_gate.py --self-test` then runs the G20–G31 arm for real (not
SKIPPED) and passes all of it, plus every pre-existing fixture (G1–G19/T/L/V/E/S/D/N6/R10) and N7's
wall-time check. Deleting just the R3 mechanism from the applied copy's `lead-gate-lib.py` (keeping
the `RELAY_R3` marker so the arm still runs, never falls back to SKIPPED) turns G20, G21, G22b,
G23, G26, G27, and G28's round-3 assertion RED — proving the fixture suite is not vacuous. Both
proofs were re-run for this round of fixes; see "Proof, re-run" below for the exact counts.

## The escape and the one real round it would have caught

Esc-064 (`docs/plans/63-how-well/proposals/esc-064-relay-conjunction.md`, already merged)
closed "the relay restates the enumeration and nothing else" — its conjunction requires >=2
DISTINCT probe sites outside the verifier's own `class_enumeration`/`findings`. It never required
those sites bear ANY relationship to what the fix actually changed. On a CI-base-image unit the
adversarial-audit BLOCKed SIX consecutive times (2026-09-09 17:58:02, 20:25:31, 20:41:40, 20:58:26,
21:19:31, 21:57:08) — the first five each on a mechanism the PREVIOUS fix commit introduced, and
the sixth (head `d73ab690`) finding a genuinely NEW defect in the fix (a `cargo rustc` step running
outside the mold-installing CI container) rather than a symptom of the earlier pattern — and every
one of the five relayed rounds satisfied esc-064's conjunction (the sixth BLOCK has no relay
artifact yet — it is the newest, and unrelayed at the time this proposal was written). The round-3
relay (block `cc9a59a3`, relayed to fix `8c7dec92`) is the clearest instance: its `probe` array
names `ci/scripts/rust_pin.sh`, `.github/workflows/dep-dag.yml`, `.devcontainer/Dockerfile:1-52`,
`ci/scripts/check_merged_index_platforms.sh:135` — four real, adjacent, non-reactive sites (R2
passes) — but `git diff --name-only cc9a59a3 8c7dec92` shows the fix actually changed
`.cargo/config.toml`, `.github/actions/setup-rust-ci/action.yml`,
`.github/workflows/_ci-base-image.yml`, `ci/scripts/rust_target_features.sh`. None of the four
probed paths is a member. The relay probed the neighbourhood of the ORIGINAL finding, never once
looked at the fix it was about to re-audit. Control: a separate unit's adversarial-audit BLOCK (a
cloud-credentials/config-secrets mechanism) went BLOCK to PASS in exactly ONE round — its relay
(`.jammi/gate-state/feat_deploy-shapes-B-config.relay.adversarial-audit.2026-09-09T19_48_13.299541_00_00.json`)
already carried `fix_head` and a `probe` array explicitly naming the fix's OWN new surfaces
(`crates/jammi-db/src/config/layers.rs`, `crates/jammi-db/src/config/secret.rs`,
`crates/jammi-db/src/storage/config.rs`, each marked "FIX'S OWN SURFACE" in the lead's own probe
text), and the follow-up adversarial-audit PASSed — this is the real control this class is measured
against: a relay that DOES probe the fix's own surface converges; one that doesn't relays into
more rounds.

## Decisions (V2/V3/V5/V8 from `scratchpad/plans/PLAN-G.md`, further narrowed by this round's fix-review)

**R3 "probe the fix."** The relay gains a lead-written `fix_head` (the fix commit's full sha).
Armed ONLY on a REPEAT dispatch of the SAME verifier type after a BLOCK — never on a first
dispatch (structurally unreachable: no prior row exists for `_decide_verifier_dispatch` to match),
and never on the CROSS-type clearing path a `fix-verifier`/`acceptance-verifier` PASS takes to
clear an older `adversarial-audit` BLOCK (`_adversarial_audit_cleared_by_verifier_pass` →
`_relay_accepted`, which now passes `check_fix=False` and so never touches git at all — this
closes an adversarial finding against an earlier draft, which ran the same `_relay_rejection`
function from BOTH call sites unconditionally). Per decision, at most ONE targeted unit's R3 runs
even when a single prompt happens to whole-token-name more than one open BLOCK of the same type —
every other targeted unit still gets a full, in-memory R1/R2 check, it just never pays for a second
git round-trip in the same decision. In `$CLAUDE_PROJECT_DIR` ONLY (required explicitly — this is
the documented hook environment contract: the harness always sets this variable for every hook
invocation, so `repo_root()`'s cwd fallback is never needed by this arm and is deliberately not
reused here), with a 5s timeout:

1. the BLOCK row's own `head_sha` ("block_sha") resolves as a commit;
2. `fix_head` resolves, and differs from `block_sha` (else: "no fix commit since the BLOCK; a
   second dispatch without a fix is a re-roll" — a re-roll, not a fix);
3. `fix_head` is reachable from the RELAY's OWN `unit_branch` field (`git merge-base
   --is-ancestor fix_head <relay's unit_branch>`), falling back to the BLOCK row's own recorded
   `unit_branch` only when the relay omits the field — the two are NO LONGER required to be
   byte-equal (an earlier draft of this arm forced them equal, which meant a BLOCK row landed in
   the `UNBOUND` fallback bucket, or whose recorded branch was later renamed or deleted, could
   never be satisfied by any relay at all; the artifact's identity is already pinned by the
   filename — built from the row's own unit_slug/agent_type/block_ts — plus the still-enforced
   `agent_type`/`block_ts` content checks, so the `unit_branch` field's only remaining job is
   naming R3's reachability target). Deliberately NOT reachable-from-`block_sha`: an `--amend`
   sibling (this repo's fix discipline produces them) breaks ancestry from `block_sha` but not
   reachability from the named branch's own tip; if the lead fixed on another branch or rebased,
   it names that branch's own head;
4. `fix_changed = git diff --name-only -z block_sha fix_head` — TRUSTED, COMPUTED, never
   lead-supplied, NUL-delimited so a changed path carrying a space or non-ASCII byte is still
   named correctly (an earlier draft split on newlines, which is unsound in general even though no
   file in this corpus currently exercises it);
5. at least one `probe` entry's PATH — the first whitespace-delimited token, a surrounding
   backtick or parenthesis AND trailing punctuation stripped, an optional trailing
   `:<n>[-<n>][,<n>]*` line-spec stripped; an unparseable token counts for nothing (monotone toward
   DENY, never toward a false ALLOW) — is EXACTLY a member of `fix_changed`. A probe naming a
   fix-changed file that is ALSO a finding location DOES satisfy R3 — probing the fix's own surface
   is the point. R2's own >=2-distinct-non-reactive requirement is unchanged and stays conjunctive,
   so the worst case is **three** probe entries (two adjacent + one fix-changed), though one entry
   can double as both when it happens to qualify for each.

Any git failure (non-zero exit, timeout, an unresolvable sha) DENIES, naming the failing command,
and states that the escape hatch (`rm .jammi/gate-state/<slug>.*`) destroys the unit's evidence
rather than fixing the underlying git problem. **§C5** amends `lead-gate-pre.sh`'s prior "no git
subprocess anywhere" doctrine to this ONE narrow, explicit exception (repeat-dispatch branch only,
never the cross-type clearing path, project-dir only, 5s bound, at most one unit per decision).

**Migration.** A relay written to disk BEFORE this patch lands carries no `fix_head` and stops
being acceptable the moment the patch IS applied — there is no grandfathering. A unit with an
in-flight relay must either rewrite it with `fix_head` (and, if needed, a `unit_branch` naming the
real branch the fix landed on) or simply re-relay against the still-open BLOCK.

**R4 (new-mechanism-arm gating) stays DROPPED**, per the prior pressure-test round: an earlier
draft added a round that armed on any `ci/scripts/**`/`.github/**` file addition; it missed most of
the real mechanism rounds in this corpus, armed on nearly every round regardless, and a design-pass
row was itself form-satisfiable — no better than R3 alone, at real cost. Design-before-mechanism is
a REQUIRED, human-owned step in `.claude/agents/lead.md` instead (loaded every session, not
mechanized): before dispatching a fix that ADDS or REWRITES a script/action/workflow step/module (a
mechanism, not a local correction), the lead writes a one-paragraph mechanism contract and
dispatches ONE `pressure-tester` round against it, on the unit itself, before the implementer. The
`pressure-tester` card is explicit that the LEAD reads that row directly, as its own design-pass
evidence, before dispatching the implementer — no gate mechanizes or reads this row; binding
`unit_branch` correctly on that dispatch is what keeps the row out of the `UNBOUND` fallback
bucket where the lead would never find it. The stopping rule (one fix round per BLOCK) lives there
too. **No round counting, no cap is mechanized** — R3 only redirects WHERE a relay probes; it does
not count rounds.

**Fixture harness (`ci/scripts/check_lead_gate.py`, agent-writable, committed in THIS unit).**
`_temp_repo(unit_branch)` — `git init` in a tempdir (with a `.gitignore` for `.jammi/`, so the
hook's OWN untracked state files are never accidentally staged by a later `git add -A` and then
deleted by a branch checkout — found by execution, see "Bugs found and fixed" below), identity via
`GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_*` env, `commit.gpgsign=false`, on a branch
named after the unit. `_write_block_row` mints a REAL commit sha as `head_sha` whenever `root` is
such a repo (never a `cafef00d` placeholder) — every existing DENY fixture that never reaches the
git arm (schema/R1/R2/"no relay artifact") is UNCHANGED and still denies with ITS OWN reason on a
plain, non-git `_fresh_root()`. `_write_relay_exact` gains `fix_head`. The existing ALLOW-reaching
fixtures (G6, G8's ALLOW half, G13's ALLOW half, G17) gain a repo + `fix_head` so they stay ALLOWED
once this patch lands — every DENY-reaching existing fixture (G1-G5, G7, G9-G16, G18-G19, T1-T4,
L1-L3, V1-V10, E1-E4, S1-S3, D1, N6) is untouched, since a DENY short-circuits before ever reaching
R3. New fixtures **G20-G31** (below), gated behind a version-marker guard (`RELAY_R3`) so
`--self-test` exits 0 in THIS tree today, reporting that arm SKIPPED, and runs the fixtures for
real the moment a human applies this doc's patches.

`_run()`'s subprocess `cwd` is a shared, empty DECOY directory by default (`_DECOY_CWD`, minted
once, never any fixture's own `project_dir`) — an adversarial/oracle finding against the prior
draft's harness caught it setting `cwd=project_dir` UNCONDITIONALLY on every call, which meant a
mutant that reduces `repo_root()` to `Path.cwd()` (dropping the `CLAUDE_PROJECT_DIR` env read
entirely) passed all 45 pre-existing fixtures anyway, because `cwd` and the env var always pointed
at the same place — the harness never actually distinguished them. Only **G26** (the one fixture
that legitimately needs the cwd FALLBACK, because it deliberately unsets `CLAUDE_PROJECT_DIR` to
test the relay arm's own explicit requirement) now passes `cwd=root` explicitly. **G30** is new: it
sets `CLAUDE_PROJECT_DIR` to the real fixture root (carrying an open, un-relayed BLOCK) and points
`cwd` at an entirely different, empty-of-blocks directory, and asserts the dispatch still DENIES —
which only holds if the code reads the env var, not `cwd`.

  - **G20** no `fix_head` at all → DENY, reason names `fix_head`.
  - **G21** `fix_head == block_sha` → DENY, reason names the re-roll.
  - **G22** (a) a relay naming the REAL branch the fix landed on is ALLOWED even though that branch
    differs from the BLOCK row's own recorded `unit_branch`; (b) on the SAME `fix_head`, a relay
    naming a DIFFERENT branch that does NOT contain it is DENIED, reason says "not reachable" (not
    `block_sha` ancestry — G25 covers that distinct axis).
  - **G23** a real repo + fix commit, but no probe names a fix-changed file → DENY (R3), reason
    redirects to "probe the fix."
  - **G24** a probe names a fix-changed file that is ALSO a finding location → ALLOW (probing
    the fix's own surface satisfies R3 even when the same path is a finding; R2's own two OTHER
    non-reactive probe entries still supply its requirement).
  - **G25** an `--amend` sibling: `block_sha` is NOT an ancestor of `fix_head` (both descend from a
    common parent) → ALLOW, because reachability is checked from the named branch's own tip, never
    `block_sha` ancestry.
  - **G26** `CLAUDE_PROJECT_DIR` unset → DENY, reason names `CLAUDE_PROJECT_DIR` (the fixture's own
    subprocess `cwd` is explicitly set to the fixture root here, so `repo_root()`'s OTHER, unrelated
    cwd fallback still resolves `.jammi/gate-state` correctly for the non-R3 arms — isolating this
    assertion to the R3 arm's OWN explicit-env-var requirement).
  - **G27** a `git` on PATH that never returns (a PATH shim) → DENY within the hook's own 5s
    budget, reason names the timeout — never an unbounded hang (see "Bugs found and fixed").
  - **G28** the REAL corpus (five relayed rounds) — see below.
  - **G29** a genuine FIRST dispatch of a brand-new unit, in a state dir that ALSO carries three
    OTHER open, fully-relayed BLOCKs (none named by this dispatch's prompt) → ALLOWED, returns in
    well under 1s, and spawns ZERO git processes — proven by a PATH-shimmed `git` that logs its own
    invocation before hanging for 30s; the log file never gets created.
  - **G30** env precedence over a decoy `cwd` (above).
  - **G31** a BLOCK row whose OWN `unit_branch` is empty (the `UNBOUND` fallback bucket) is still
    satisfiable by a relay that names the real branch the fix landed on.

**Docs.** `.claude/hooks/README.md` — the relay schema (`fix_head`), R3's full arm order, §C5, and
the HONEST LIMIT paragraph extended to R3 (below). `.claude/agents/lead.md` — the relay template
gains `fix_head`; R3 prose; the REQUIRED design-before-mechanism paragraph (mechanism contract +
one pressure-test round + one-fix-round-per-BLOCK stopping rule); the residual list gains R3's own
qualifier. `.claude/agents/adversarial-audit.md` / `pressure-tester.md` — `head_sha` is explicitly
"the head you read, nothing else" (R3 uses a verifier's reported `head_sha` as the fix window's
`block_sha`); `pressure-tester.md` additionally states that a design-pass dispatch's `unit_branch`
must be the real unit branch, never the `UNBOUND` fallback bucket, because the LEAD reads that
row directly as its own design-pass evidence (no gate reads it) — landing it in `UNBOUND` means the
lead never finds it.

## HONEST LIMIT (README, extended)

The hook enforces that adjacent probing (R2) AND fix-probing (R3) are each **asserted** with a
named, citation-checkable site that is ACTUALLY a member of the relevant set — never that the
lead semantically examined it. A lead can satisfy R3 by pasting a path out of its own diff without
reading it. **R3 converts silent neighbourhood-probing into fix-window-probing, on the record; it
does not by itself close esc-097's class.** The substance — auditing the fix's own surfaces before
re-dispatching, and not introducing a new mechanism without a design pass — is a
`.claude/agents/lead.md` discipline, not a hook. citation-checker and the retrospective judge the
probes' substance, not this hook. A further, deliberate residual: R3's reachability check trusts the
relay's OWN `unit_branch` field the same way it already trusted `fix_head` (the relay artifact is
written by the LEAD directly — `Write` is not gated — and trusted for content; only GIT-VERIFIED
facts, never lead-asserted ones, gate the decision). A lead who would misname `unit_branch` to
game reachability could already misname `fix_head` today; this is the same trust boundary, not a
new one.

## G28: the real corpus, run once and reviewed — not asserted in advance

`_E1_ROUND_FIX_CHANGED` (the real `git diff --name-only <round-N head> <round-(N+1) head>` file
sets) and `_E1_ROUND_PROBE` (the real `probe` arrays) are copied from the actual gate-state rows and
relay artifacts for the FIVE fix transitions that carry a relay (the sixth BLOCK, on `d73ab690`, has
none yet), read-only against the project checkout's real object store, reproduced in a hermetic
synthetic repo (never the real repo) so the fixture needs no network and no dependency on that
branch continuing to exist. Running the fixture once and reviewing its output (the patched lib, in
a throwaway applied copy) gives the ACTUAL, not assumed, per-round result:

| round | block (real sha) | fix (real sha) | real probe names a real fix-changed file? | R3 outcome |
|---|---|---|---|---|
| 1 | `3f5b5bf0` | `cf0d37e7` | yes — `.cargo/config.toml`, `ci/scripts/check_merged_index_platforms.sh` | **ALLOW** |
| 2 | `cf0d37e7` | `cc9a59a3` | yes — `.github/workflows/image.yml` | **ALLOW** |
| 3 | `cc9a59a3` | `8c7dec92` | **no** — none of `rust_pin.sh`/`dep-dag.yml`/`.devcontainer/Dockerfile`/`check_merged_index_platforms.sh` is in the real fix-changed set | **DENY (R3 fires)** |
| 4 | `8c7dec92` | `38bf6b44` | yes — `.github/workflows/ci.yml` | **ALLOW** |
| 5 | `38bf6b44` | `d73ab690` | yes — `.github/actions/setup-rust-ci/action.yml`, `.github/workflows/ci.yml`, `.github/workflows/dep-dag.yml`, `.cargo/config.toml` (this relay ALREADY carries its own real `fix_head: d73ab690…` — the lead had started probing the fix's own surface informally by round 5, ahead of this gate existing) | **ALLOW** |

A sixth adversarial-audit BLOCK exists on this same unit (head `d73ab690`, ts `21:57:08`) but has no
relay artifact yet at the time this proposal was written — it is not replayed by G28, and is not
counted in the "1-of-5" figure below.

The exact reason string G28 asserts for round 3 (computed by running the patched hook against the
hermetic reproduction, its own synthetic shas substituted for the real ones since the fixture never
touches the real repo):

```
a second adversarial-audit dispatch naming …/adversarial-audit:
relay `probe` names none of the 4 files the fix changed (block <synthetic7>..fix <synthetic7>);
probe the fix, not the neighbourhood — e.g. .cargo/config.toml,
.github/actions/setup-rust-ci/action.yml, .github/workflows/_ci-base-image.yml is denied — the
relay artifact for that (unit, agent_type, block_ts) is missing or insufficient
```

**Stated honestly, not oversold:** R3 as specified denies exactly **1 of the 5** relayed historical
rounds. The other four rounds' real probes ALSO happen to name a real fix-changed file — not
because the lead was probing the fix (in rounds 1/2/4 there is no annotation suggesting that; only
round 5 explicitly says so), but because this is a CI-infra unit that repeatedly touches the same
small hot-file set (`.cargo/config.toml`, `setup-rust-ci/action.yml`, `_ci-base-image.yml`,
`ci.yml`, `dep-dag.yml`) across nearly every round, so "probe adjacent to the finding" and "probe
the fix" collide by coincidence most of the time in THIS corpus. R3 is a real, mechanical
tightening (it converts a 0-real-file-named relay into a hard DENY, closing the clearest instance)
but its measured hit rate against its own motivating corpus is 1-of-5, not 5-of-5 — the
honest-limit paragraph above, and `.claude/agents/lead.md`'s design-before-mechanism paragraph,
carry the rest of the weight.

## Bugs found and fixed while building this proposal (found by execution, not asserted)

Building and running G20-G31 against a THROWAWAY, `cp -R` copy of the tree (outside any git
worktree, deleted after — this is a change from an earlier draft, which used `git worktree add
--detach`; a plain `cp -R` copy needs no `.git` link back to this repo at all, and is what "outside
any git worktree of the repo" now means literally) surfaced real defects, both this round and in
the FIRST draft of the R3 patch, all fixed in the patch files, not merely in the fixture harness:

1. **A `.jammi/`-tracked-file trap.** `_temp_repo`'s seed commit, without a `.gitignore` for
   `.jammi/`, let a later `_commit_fix`'s `git add -A` accidentally stage the hook's own
   (untracked) state JSONL once it existed — and a subsequent `git checkout` back to the unit
   branch (an amend-sibling setup) then DELETED it, because the file was tracked on one branch
   and absent on the other. `all_open_blocks` found no row and treated the dispatch as a harmless
   first round — a false ALLOW that had nothing to do with R3's own logic (confirmed by calling
   `_relay_rejection` directly with the in-memory row, which still computed the correct DENY).
   Fixed by gitignoring `.jammi/` in every fixture repo, matching the real repo's own convention.
2. **A grandchild-holding-the-pipes trap in `_run_git`, and a corrected account of what actually
   bounds a hang.** The first draft used a bare `subprocess.run(cmd, timeout=5)`. On POSIX, that
   call's own post-timeout `wait()` already returns once the DIRECT child is reaped — a timeout on
   a childless `git` does not, in fact, hang unboundedly; measured, it returns at ~5s. What actually
   IS unbounded is a `git` that leaves a GRANDCHILD alive holding the stdout/stderr pipes open after
   the direct child exits (the realistic production analogue is a hung credential helper or network
   transport `git` spawned and detached from; the fixture reproduces it with a PATH shim shaped like
   `sh -c 'sleep 30 & exec sleep 7'`, which discriminates a `killpg`-based implementation from a
   plain `proc.kill()` one specifically because the backgrounded `sleep 30` survives the shim's own
   `exec`). `killpg`'s real, measured benefit is REAPING that orphaned grandchild — without it, the
   orphan keeps running (and, in the pipe-holding shape, keeps a read end open) well past the
   declared timeout. Fixed by running `git` in its own process group (`start_new_session=True`) and
   killing the WHOLE group (`os.killpg`) on timeout. A related, compounding issue from the first
   draft — `_relay_rejection` had two callers reachable from a single `pre` decision (the
   repeat-dispatch check, and the cross-type clearing path via `_adversarial_audit_cleared_by_verifier_pass`)
   with BOTH capable of running R3's git arm — is now closed structurally, not just cached: the
   clearing path calls with `check_fix=False` and never reaches the git arm at all, so there is
   nothing left to memoize away. A one-process-lifetime memo (keyed additionally on `check_fix`, so
   a `check_fix=False` lookup can never be served from a `check_fix=True` cache entry or vice versa)
   remains for the OTHER real multiplier: a single prompt naming more than one open BLOCK of the
   same type, bounded to at most one unit's R3 per decision (see the R3 decision above).
3. **The fixture harness's own `cwd`-vs-env vacuity.** `_run()` set the hook subprocess's `cwd` to
   `project_dir` unconditionally on every call — so a mutant that reduces `repo_root()` to
   `Path.cwd()` (never reading `CLAUDE_PROJECT_DIR` at all) passed every one of the 45 pre-existing
   fixtures, because `cwd` and the env var always agreed. Fixed by defaulting `cwd` to a shared,
   empty decoy directory and scoping `cwd=project_dir` to the ONE fixture (G26) that legitimately
   needs it; **G30** is the new, explicit env-precedence witness. Re-run against the same 2×2
   mutation this was found by: the mutant now fails 33 of 45 base fixtures under the corrected
   harness (up from 0 of 45); the control lib (which reads the env var, falling back to `cwd`
   only when it is unset) still passes all 45.
4. **A reachability target that could never resolve for an `UNBOUND` or renamed/deleted-branch
   row.** The first draft of R3 required the relay's `unit_branch` to byte-equal the BLOCK row's
   own recorded value and then used THAT (necessarily identical) value as the reachability target —
   which meant a row that landed in the `UNBOUND` fallback bucket (empty `unit_branch` at
   verdict-write time), or whose recorded branch was since renamed or deleted, could never be
   satisfied by ANY relay: the target itself never resolved. Fixed by dropping the byte-equality
   requirement (the artifact's identity is already pinned by the filename and the
   `agent_type`/`block_ts` content checks) and targeting the relay's OWN `unit_branch` field,
   falling back to the row's only when the relay omits it — see **G22**, **G31**.
5. **`_probe_path` did not strip a surrounding backtick or parenthesis**, so a probe entry written
   as `` `ci/scripts/foo.sh` `` or `(ci/scripts/foo.sh)` (both realistic Markdown-flavoured
   phrasings a lead might use) parsed to a token that never matched a bare `fix_changed` entry —
   monotone toward a false DENY, not a false ALLOW, but still a real defect; fixed by stripping
   both before the trailing-punctuation and line-spec passes.

## Ledger lifecycle

`esc-097-relay-form-satisfied-without-probing-the-fix` stays `open` (its `eval_ref` names
G20-G31, currently SKIPPED) until a human applies this doc's patches and the self-test's G20-G31
arm goes green on main — at that point the row moves to `eval_added`, matching the precedent
(`esc-064-relay-conjunction.md`'s own lifecycle note).

## Session-local evidence

Every `scratchpad/…` path cited above (throwaway applied copies, the oracle's 2×2 mutation trees,
the plan-of-record file) is session-local working state, not a repo artifact — per this repo's own
`.gitignore` (`scratchpad/` is ignored), none of it is tracked or citable after the session ends;
it is named here only to show how the numbers in this doc were produced, not as a durable
reference. The durable evidence is the tracked patch files under
`docs/plans/63-how-well/proposals/esc-097/`, the fixtures in `ci/scripts/check_lead_gate.py`, and
the real `.jammi/gate-state/*` rows and relay artifacts cited by sha and timestamp above.
