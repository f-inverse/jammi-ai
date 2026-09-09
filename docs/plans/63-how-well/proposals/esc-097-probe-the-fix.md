# PROPOSAL (human-merge): tighten(lead-gate) — relays must probe the fix, not just the class's neighbourhood (esc-097)

Status: PROPOSED — this unit's own commit ships the fixtures (`ci/scripts/check_lead_gate.py`,
agent-writable) RED against the current hook, plus this doc's fenced diffs for the four files
under `.claude/hooks/**` / `.claude/agents/*.md` (agent-write-DENIED — a human applies them).
Gate-definition change: `SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `ci/scripts/check_lead_gate.py`,
`.claude/agents/*.md`) → proposal only, human admin-merge. Nothing under `.claude/hooks/**` or
`.claude/agents/*.md` has been applied to this repo yet.

## The escape and the one real round it would have caught

Esc-064 (`docs/plans/63-how-well/proposals/esc-064-relay-conjunction.md`, already merged)
closed "the relay restates the enumeration and nothing else" — its conjunction requires >=2
DISTINCT probe sites outside the verifier's own `class_enumeration`/`findings`. It never required
those sites bear ANY relationship to what the fix actually changed. On
`feat/deploy-shapes-E1-arm64-ci-base` the adversarial-audit BLOCKed five consecutive times
(2026-09-09 17:58:02, 20:25:31, 20:41:40, 20:58:26, 21:19:31), each on a mechanism the
PREVIOUS fix commit introduced — and every one of the five relays satisfied esc-064's
conjunction. The round-3 relay
(`.jammi/gate-state/feat_deploy-shapes-E1-arm64-ci-base.relay.adversarial-audit.2026-09-09T20_41_40.063877_00_00.json`,
block `cc9a59a3`, relayed to fix `8c7dec92`) is the clearest instance: its `probe` array names
`ci/scripts/rust_pin.sh`, `.github/workflows/dep-dag.yml`, `.devcontainer/Dockerfile:1-52`,
`ci/scripts/check_merged_index_platforms.sh:135` — four real, adjacent, non-reactive sites (R2
passes) — but `git diff --name-only cc9a59a3 8c7dec92` shows the fix actually changed
`.cargo/config.toml`, `.github/actions/setup-rust-ci/action.yml`,
`.github/workflows/_ci-base-image.yml`, `ci/scripts/rust_target_features.sh`. None of the four
probed paths is a member. The relay probed the neighbourhood of the ORIGINAL finding, never once
looked at the fix it was about to re-audit. Control: `fix/476-seam-counter-test-isolation`'s fix
rounds were plain value/branch corrections inside an existing call — no script/action/workflow/
module added or rewritten — and converged in one round without this pattern.

## Decisions (V2/V3/V5/V8 from `scratchpad/plans/PLAN-G.md`; V wins over D)

**R3 "probe the fix."** The relay gains a lead-written `fix_head` (the fix commit's full sha).
In the relay-validation arm ONLY — a second dispatch after a BLOCK, never the hot first-dispatch
path — the hook resolves, in `$CLAUDE_PROJECT_DIR` ONLY (required explicitly; `repo_root()`'s cwd
fallback is not reused here — V8), with a 5s timeout:

1. the BLOCK row's own `head_sha` ("block_sha") resolves as a commit;
2. `fix_head` resolves, and differs from `block_sha` (else: "no fix commit since the BLOCK; a
   second dispatch without a fix is a re-roll" — a re-roll, not a fix);
3. `fix_head` is reachable from the UNIT's own tip (`git merge-base --is-ancestor fix_head
   unit_branch`) — deliberately NOT from `block_sha`: an `--amend` sibling (this repo's fix
   discipline produces them) breaks ancestry from `block_sha` but not reachability from the unit's
   tip; if the lead fixed on another branch or rebased, it names that branch's own head;
4. `fix_changed = git diff --name-only block_sha fix_head` — TRUSTED, COMPUTED, never
   lead-supplied;
5. at least one `probe` entry's PATH — the first whitespace-delimited token, trailing punctuation
   stripped, an optional trailing `:<n>[-<n>][,<n>]*` line-spec stripped; an unparseable token
   counts for nothing (monotone toward DENY, never toward a false ALLOW) — is EXACTLY a member of
   `fix_changed`. A probe naming a fix-changed file that is ALSO a finding location DOES satisfy
   R3 — probing the fix's own surface is the point. R2's own >=2-distinct-non-reactive requirement
   is unchanged and stays conjunctive, so the worst case is **three** probe entries (two adjacent
   + one fix-changed), though one entry can double as both when it happens to qualify for each.

Any git failure (non-zero exit, timeout, an unresolvable sha) DENIES, naming the failing command,
and states the escape hatch (`rm .jammi/gate-state/<slug>.*`) destroys the unit's evidence rather
than fixing the underlying git problem. **§C5** amends `lead-gate-pre.sh`'s prior "no git
subprocess anywhere" doctrine to this ONE narrow, explicit exception (relay arm only, project-dir
only, 5s bound).

**R4 (new-mechanism-arm gating) is DROPPED.** An earlier draft of this contract (D4/D5) added a
round that armed on any `ci/scripts/**`/`.github/**` file addition; the pressure-test (V1) found
it misses 2 of the 3 real mechanism rounds in this exact corpus, arms on all 4, and a design-pass
row is itself form-satisfiable — no better than R3 alone, at real cost. Design-before-mechanism
becomes a REQUIRED, human-owned step in `.claude/agents/lead.md` instead (loaded every session,
not mechanized): before dispatching a fix that ADDS or REWRITES a script/action/workflow
step/module (a mechanism, not a local correction), the lead writes a one-paragraph mechanism
contract and dispatches ONE `pressure-tester` round against it, on the unit itself. The stopping
rule (one fix round per BLOCK) lives there too. **No round counting, no cap is mechanized** — R3
only redirects WHERE a relay probes; it does not count rounds.

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
R3. New fixtures **G20-G28** (below), gated behind a version-marker guard (`RELAY_R3`) so
`--self-test` exits 0 in THIS tree today, reporting that arm SKIPPED, and runs the fixtures for
real the moment a human applies this doc's diffs.

  - **G20** no `fix_head` at all → DENY, reason names `fix_head`.
  - **G21** `fix_head == block_sha` → DENY, reason names the re-roll.
  - **G22** `fix_head` resolves and even descends from `block_sha`, but on a DIFFERENT branch never
    merged into the unit → DENY, reason says "not reachable" (the reachability check is against
    the UNIT's tip, never `block_sha` — this is the axis G25 tests the other side of).
  - **G23** a real repo + fix commit, but no probe names a fix-changed file → DENY (R3), reason
    redirects to "probe the fix."
  - **G24** a probe names a fix-changed file that is ALSO a finding location → ALLOW (D3: probing
    the fix's own surface satisfies R3 even when the same path is a finding; R2's own two OTHER
    non-reactive probe entries still supply its requirement).
  - **G25** an `--amend` sibling: `block_sha` is NOT an ancestor of `fix_head` (both descend from a
    common parent) → ALLOW, because V2 requires reachability from the unit's tip, never
    `block_sha` ancestry.
  - **G26** `CLAUDE_PROJECT_DIR` unset → DENY, reason names `CLAUDE_PROJECT_DIR` (the fixture's own
    subprocess `cwd` is still set to the fixture root, so `repo_root()`'s OTHER, unrelated cwd
    fallback still resolves `.jammi/gate-state` correctly — isolating this assertion to the R3
    arm's OWN explicit-env-var requirement, per V8).
  - **G27** a `git` on PATH that never returns (a PATH shim) → DENY within the hook's own 5s
    budget, reason names the timeout — never an unbounded hang (see "Bugs found and fixed").
  - **G28** the REAL `feat/deploy-shapes-E1-arm64-ci-base` corpus — see below.

**Docs.** `.claude/hooks/README.md` — the relay schema (`fix_head`), R3's full arm order, §C5, and
the HONEST LIMIT paragraph extended to R3 (below). `.claude/agents/lead.md` — the relay template
gains `fix_head`; R3 prose; the REQUIRED design-before-mechanism paragraph (mechanism contract +
one pressure-test round + one-fix-round-per-BLOCK stopping rule); the residual list gains R3's own
qualifier. `.claude/agents/adversarial-audit.md` / `pressure-tester.md` — `head_sha` is explicitly
"the head you read, nothing else" (R3 uses a verifier's reported `head_sha` as the fix window's
`block_sha`); `pressure-tester.md` additionally states that a design-pass dispatch's `unit_branch`
must be the real unit branch, never the `UNBOUND` fallback bucket, so the lead's own
design-before-mechanism step actually lands where R3-adjacent tooling can find it.

## HONEST LIMIT (README, extended)

The hook enforces that adjacent probing (R2) AND fix-probing (R3) are each **asserted** with a
named, citation-checkable site that is ACTUALLY a member of the relevant set — never that the
lead semantically examined it. A lead can satisfy R3 by pasting a path out of its own diff without
reading it. **R3 converts silent neighbourhood-probing into fix-window-probing, on the record; it
does not by itself close esc-097's class.** The substance — auditing the fix's own surfaces before
re-dispatching, and not introducing a new mechanism without a design pass — is a
`.claude/agents/lead.md` discipline, not a hook. citation-checker and the retrospective judge the
probes' substance, not this hook.

## G28: the real corpus, run once and reviewed (V3) — not asserted in advance

`_E1_ROUND_FIX_CHANGED` (the real `git diff --name-only <round-N head> <round-(N+1) head>` file
sets) and `_E1_ROUND_PROBE` (the real `probe` arrays) are copied from the actual
`.jammi/gate-state/feat_deploy-shapes-E1-arm64-ci-base.jsonl` rows and its five
`*.relay.adversarial-audit.*.json` artifacts (read-only, against the project checkout's real
object store: shas `3f5b5bf0`/`cf0d37e7`/`cc9a59a3`/`8c7dec92`/`38bf6b44`/`d73ab690`), reproduced
in a hermetic synthetic repo (never the real repo) so the fixture needs no network and no
dependency on that branch continuing to exist. Running the fixture once and reviewing its output
(the patched lib, in a throwaway `git worktree add --detach` copy) gives the ACTUAL, not assumed,
per-round result:

| round | block (real sha) | fix (real sha) | real probe names a real fix-changed file? | R3 outcome |
|---|---|---|---|---|
| 1 | `3f5b5bf0` | `cf0d37e7` | yes — `.cargo/config.toml`, `ci/scripts/check_merged_index_platforms.sh` | **ALLOW** |
| 2 | `cf0d37e7` | `cc9a59a3` | yes — `.github/workflows/image.yml` | **ALLOW** |
| 3 | `cc9a59a3` | `8c7dec92` | **no** — none of `rust_pin.sh`/`dep-dag.yml`/`.devcontainer/Dockerfile`/`check_merged_index_platforms.sh` is in the real fix-changed set | **DENY (R3 fires)** |
| 4 | `8c7dec92` | `38bf6b44` | yes — `.github/workflows/ci.yml` | **ALLOW** |
| 5 | `38bf6b44` | `d73ab690` | yes — `.github/actions/setup-rust-ci/action.yml`, `.github/workflows/ci.yml`, `.github/workflows/dep-dag.yml`, `.cargo/config.toml` (this relay ALREADY carries its own real `fix_head: d73ab690...` — the lead had started probing the fix's own surface informally by round 5, ahead of this gate existing) | **ALLOW** |

The exact reason string G28 asserts for round 3 (computed by running the patched hook against the
hermetic reproduction, its own synthetic shas substituted for the real ones since the fixture never
touches the real repo):

```
a second adversarial-audit dispatch naming feat_deploy-shapes-E1-arm64-ci-base/adversarial-audit:
relay `probe` names none of the 4 files the fix changed (block <synthetic7>..fix <synthetic7>);
probe the fix, not the neighbourhood — e.g. .cargo/config.toml,
.github/actions/setup-rust-ci/action.yml, .github/workflows/_ci-base-image.yml is denied — the
relay artifact for that (unit, agent_type, block_ts) is missing or insufficient
```

**Stated honestly, not oversold:** R3 as specified denies exactly **1 of the 5** real historical
rounds. The other four rounds' real probes ALSO happen to name a real fix-changed file — not
because the lead was probing the fix (in rounds 1/2/4 there is no annotation suggesting that; only
round 5 explicitly says so), but because this is a CI-infra unit that repeatedly touches the same
small hot-file set (`.cargo/config.toml`, `setup-rust-ci/action.yml`, `_ci-base-image.yml`,
`ci.yml`, `dep-dag.yml`) across nearly every round, so "probe adjacent to the finding" and "probe
the fix" collide by coincidence most of the time in THIS corpus. This is exactly the shape the plan
(`scratchpad/plans/PLAN-G.md` V3) anticipated: "the per-round outcomes are RECORDED BY THE CODE,
never asserted in advance." R3 is a real, mechanical tightening (it converts a 0-real-file-named
relay into a hard DENY, closing the clearest instance) but its measured hit rate against its own
motivating corpus is 1-of-5, not 5-of-5 — the honest-limit paragraph above, and
`.claude/agents/lead.md`'s design-before-mechanism paragraph, carry the rest of the weight.

## Bugs found and fixed while building this proposal (found by execution, not asserted)

Building G20-G28 and running them against a THROWAWAY `git worktree add --detach` copy with this
doc's diffs applied surfaced two real defects in the FIRST draft of the R3 patch, both fixed in the
diff below (not merely in the fixture harness):

1. **A `.jammi/`-tracked-file trap.** `_temp_repo`'s seed commit, without a `.gitignore` for
   `.jammi/`, let a later `_commit_fix`'s `git add -A` accidentally stage the hook's own
   (untracked) state JSONL once it existed — and a subsequent `git checkout` back to the unit
   branch (G22's amend-sibling setup) then DELETED it, because the file was tracked on one branch
   and absent on the other. `all_open_blocks` found no row and treated the dispatch as a harmless
   first round — a false ALLOW that had nothing to do with R3's own logic (confirmed by calling
   `_relay_rejection` directly with the in-memory row, which still computed the correct DENY).
   Fixed by gitignoring `.jammi/` in every fixture repo, matching the real repo's own convention.
2. **An unbounded-timeout trap in `_run_git` itself, not just the fixture.** The first draft used
   `subprocess.run(cmd, timeout=5)` directly. A `git` that spawns a still-alive child holding the
   stdout/stderr pipes open (a hung network helper is the realistic production analogue of the
   fixture's PATH-shimmed `sleep`) is not killed by `proc.kill()` — Python's own post-timeout
   pipe-drain then blocks until that orphan exits on its own, silently turning the declared "5s
   timeout" into an unbounded wait, exactly the failure mode the timeout exists to rule out. Fixed
   by running `git` in its own process group (`start_new_session=True`) and killing the WHOLE
   group (`os.killpg`) on timeout. A related, compounding issue — `_relay_rejection` has two
   callers reachable from one `pre` decision (the direct check, and
   `_adversarial_audit_cleared_by_verifier_pass` via `all_open_blocks`), so an uncached git call
   would fire twice per decision, doubling the worst-case fail-closed latency to 10s on exactly the
   paths that matter most — fixed with a one-process-lifetime memo keyed on `(sdir, unit_slug,
   agent_type, ts)`, which changes no decision, only how many times git is asked to make it.

## Ledger lifecycle

`esc-097-relay-form-satisfied-without-probing-the-fix` stays `open` (its `eval_ref` names G20-G28,
currently SKIPPED) until a human applies this doc's diffs and the self-test's G20-G28 arm goes
green on main — at that point the row moves to `eval_added`, matching the precedent
(`esc-064-relay-conjunction.md`'s own lifecycle note).

---

## The patch, as fenced unified diffs (produced from edited copies under `scratchpad/wt-G-proposal/`, never the real paths — `.claude/hooks/**` and `.claude/agents/*.md` are agent-write-DENIED)

### (a) `.claude/hooks/lead-gate-lib.py`

```diff
--- a/.claude/hooks/lead-gate-lib.py
+++ b/.claude/hooks/lead-gate-lib.py
@@ -27,6 +27,27 @@
   `enumeration_missing` flag — and adjacent probing (`probe`, >=2 distinct
   sites outside enumeration+findings) is armed ALWAYS.
 
+esc-097 (R3, "PROBE THE FIX"): the relay arm gains ONE further requirement,
+armed only in this second-dispatch path — never on the hot `pre` dispatch of
+a FIRST round, which still runs no git subprocess at all. The relay gains a
+lead-written `fix_head` (full sha). Arm order (V5): artifact-exists -> schema
+-> R1 (coverage) -> R2 (proactivity, always) -> CLAUDE_PROJECT_DIR is set ->
+`row.head_sha` (the BLOCK's own head, "block_sha") resolves -> `fix_head`
+resolves / != block_sha / is reachable from the UNIT's own tip (never from
+block_sha — an `--amend` sibling breaks that ancestry, and this repo's fix
+discipline produces amend siblings) -> R3 (>=1 probe path names a file the
+fix actually changed). `fix_changed` is `git diff --name-only <block_sha>
+<fix_head>` — TRUSTED and COMPUTED, never lead-supplied. git runs ONLY here,
+ONLY in `$CLAUDE_PROJECT_DIR` (required explicitly — `repo_root()`'s cwd
+fallback is not used by this arm), with a 5s timeout; any git failure DENIES,
+naming the failing command, and states that `rm .jammi/gate-state/<slug>.*`
+is the escape hatch but destroys the unit's evidence rather than fixing the
+underlying git problem. HONEST LIMIT: R3 cannot verify examination, only
+that a named path is a real member of the fix's own diff — a lead can still
+satisfy it by pasting a path out of its own change; the substantive rule
+(design-before-mechanism, one fix round per BLOCK) is a `.claude/agents/
+lead.md` discipline, not a hook.
+
 Explicitly OUT OF SCOPE by this cut (dropped entirely, not log-only):
 `SendMessage` gating and all message-prose parsing; implementer-dispatch
 binding; the Bash backstop (the mechanical control is `permissions.deny`
@@ -67,7 +88,10 @@
 allow; no `errors="replace"` fallback). `start`/`stop` are best-effort
 writers that never block a subagent lifecycle event and always exit 0.
 
-No git subprocess anywhere in this module (hot-path speed).
+No git subprocess ANYWHERE except the relay-validation arm added by esc-097
+(R3) — never on the hot first-dispatch path, and never outside
+`$CLAUDE_PROJECT_DIR` (see the esc-097 paragraph above; §C5 amends the prior
+"no git subprocess anywhere" doctrine to this narrower, explicit exception).
 """
 
 from __future__ import annotations
@@ -75,10 +99,18 @@
 import json
 import os
 import re
+import signal
+import subprocess
 import sys
 from datetime import datetime, timezone
 from pathlib import Path
 
+# esc-097 (R3, "probe the fix"): the version marker `check_lead_gate.py
+# --self-test`'s G20-G28 arm probes for via `hasattr` — its presence is what
+# tells the fixture harness the hook patch below is actually applied, so the
+# arm runs for real instead of reporting SKIPPED.
+RELAY_R3 = True
+
 # --------------------------------------------------------------------------
 # Agent-type lattice (closed world, deny-unknown).
 # --------------------------------------------------------------------------
@@ -509,9 +541,142 @@
     import unicodedata
     cleaned = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
     return cleaned.strip()
+
+
+# --------------------------------------------------------------------------
+# esc-097 (R3, "probe the fix"): the ONLY git subprocess in this module, run
+# ONLY from the relay-validation arm (a second dispatch after a BLOCK) —
+# never on a first dispatch's hot path. `$CLAUDE_PROJECT_DIR` is REQUIRED
+# explicitly here (V8: `repo_root()`'s cwd fallback is deliberately not
+# reused by this arm), with a 5s timeout; any failure DENIES, naming the
+# failing command.
+# --------------------------------------------------------------------------
+
+_GIT_TIMEOUT_S = 5.0
+_ESCAPE_HATCH_NOTE = (
+    "rm .jammi/gate-state/<slug>.* is the operator escape hatch, but it "
+    "destroys the unit's evidence rather than fixing the underlying git problem"
+)
 
 
+def _run_git(args: list[str], cwd: str) -> tuple[bool, str]:
+    """Runs `git -C <cwd> <args>` with a 5s timeout. `(True, stdout)` on
+    success; `(False, reason)` naming the failing command on any failure
+    (non-zero exit, timeout, or git/the cwd being unusable) — the caller
+    always turns a `False` into a DENY, never a silent allow.
+
+    Runs in its OWN process group and, on timeout, kills the WHOLE group —
+    not merely the direct child. `git` can itself spawn a helper (a remote
+    transport, a credential helper) that inherits the stdout/stderr pipes;
+    killing only the direct `git` process leaves that grandchild alive,
+    holding the pipes open, so the "5s timeout" would silently degrade into
+    an UNBOUNDED wait for the orphan to exit on its own — exactly the failure
+    mode this fail-closed boundary exists to rule out."""
+    cmd = ["git", "-C", cwd] + list(args)
+    printable = "git " + " ".join(args)
+    try:
+        proc = subprocess.Popen(
+            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
+            start_new_session=True,  # its own process group, for the kill below
+        )
+    except OSError as exc:
+        return False, f"`{printable}` failed to run ({exc}) — {_ESCAPE_HATCH_NOTE}"
+    try:
+        out, _err = proc.communicate(timeout=_GIT_TIMEOUT_S)
+    except subprocess.TimeoutExpired:
+        try:
+            os.killpg(proc.pid, signal.SIGKILL)
+        except ProcessLookupError:
+            pass
+        proc.communicate()  # reap; the group is dead, this returns immediately
+        return False, f"`{printable}` timed out after {_GIT_TIMEOUT_S:g}s — {_ESCAPE_HATCH_NOTE}"
+    if proc.returncode != 0:
+        return False, f"`{printable}` exited {proc.returncode} — {_ESCAPE_HATCH_NOTE}"
+    return True, out.strip()
+
+
+# A probe entry's PATH is the first whitespace-delimited token, trailing
+# punctuation stripped, then an optional trailing `:<n>[-<n>][,<n>]*` line
+# spec stripped (D3). Unparseable -> None, which counts for nothing toward
+# R3 (monotone toward DENY, never toward a false ALLOW).
+_PROBE_LINESPEC_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+)*$")
+
+
+def _probe_path(entry: str) -> str | None:
+    if not isinstance(entry, str):
+        return None
+    parts = entry.strip().split(None, 1)
+    if not parts:
+        return None
+    tok = parts[0].rstrip(",;:.)")
+    tok = _PROBE_LINESPEC_RE.sub("", tok)
+    return tok or None
+
+
+def _fix_window(row: dict, fix_head: object) -> tuple[list[str] | None, str | None]:
+    """Resolves the trusted, COMPUTED `fix_changed` set for the relay arm.
+    `(fix_changed, None)` on success; `(None, deny_reason)` otherwise. Never
+    called on a first dispatch — only once R1/R2 have already passed."""
+    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
+    if not project_dir:
+        return None, "hook needs CLAUDE_PROJECT_DIR for the relay arm"
+
+    block_sha = row.get("head_sha")
+    if not isinstance(block_sha, str) or not block_sha:
+        return None, "BLOCK row's head_sha does not resolve"
+    ok, why = _run_git(["cat-file", "-e", f"{block_sha}^{{commit}}"], project_dir)
+    if not ok:
+        return None, f"BLOCK row's head_sha does not resolve (an amend can orphan it) — {why}"
+
+    if not isinstance(fix_head, str) or not fix_head:
+        return None, "relay carries no `fix_head` — the lead must write the fix commit's full sha"
+    ok, why = _run_git(["cat-file", "-e", f"{fix_head}^{{commit}}"], project_dir)
+    if not ok:
+        return None, f"relay `fix_head` does not resolve — {why}"
+
+    if fix_head == block_sha:
+        return None, "no fix commit since the BLOCK; a second dispatch without a fix is a re-roll"
+
+    unit_branch = row.get("unit_branch")
+    if not isinstance(unit_branch, str) or not unit_branch:
+        return None, "BLOCK row carries no unit_branch to check fix_head's reachability against"
+    ok, _out = _run_git(["merge-base", "--is-ancestor", fix_head, unit_branch], project_dir)
+    if not ok:
+        return None, (
+            f"relay `fix_head` {fix_head[:7]} is not reachable from unit `{unit_branch}`'s tip — "
+            "name a fix actually committed to this unit (or that branch's own head, if you rebased "
+            "or fixed elsewhere)"
+        )
+
+    ok, out = _run_git(["diff", "--name-only", block_sha, fix_head], project_dir)
+    if not ok:
+        return None, f"could not compute the fix window: {out}"
+    return [line for line in out.splitlines() if line.strip()], None
+
+
+# esc-097: `_relay_rejection` has TWO callers reachable from a single `pre`
+# decision — `_decide_verifier_dispatch`'s own check, AND
+# `_adversarial_audit_cleared_by_verifier_pass` (via `all_open_blocks`,
+# called earlier to build the SAME dispatch's `targeted` list) — a
+# pre-existing shape (esc-064's proposal doc notes it), harmless while every
+# check was in-memory. R3 is not: a genuinely hung `git` now costs up to 5s
+# PER CALL, so an uncached double-call would silently double the fail-closed
+# latency budget to 10s on exactly the paths that matter most (a real git
+# problem). One process-lifetime memo, keyed on the identity a relay
+# artifact is looked up by, closes this without changing any decision.
+_relay_rejection_memo: dict[tuple[str, str, str, str], str | None] = {}
+
+
 def _relay_rejection(sdir: Path, unit_slug: str, row: dict) -> str | None:
+    key = (str(sdir), unit_slug, str(row.get("agent_type")), str(row.get("ts")))
+    if key in _relay_rejection_memo:
+        return _relay_rejection_memo[key]
+    result = _relay_rejection_uncached(sdir, unit_slug, row)
+    _relay_rejection_memo[key] = result
+    return result
+
+
+def _relay_rejection_uncached(sdir: Path, unit_slug: str, row: dict) -> str | None:
     """`None` == accepted; otherwise the operator-facing REASON the relay is
     missing or insufficient.
 
@@ -570,6 +735,24 @@
                 "own class_enumeration/findings (>=2 required) — restating the enumeration "
                 "is reactive acknowledgment, not adjacent probing; name >=2 sites you "
                 "EXAMINED and found clean (esc-064)")
+
+    # R3 PROBE-THE-FIX (esc-097) — armed ALWAYS, after R1/R2, on every gated
+    # verifier type's SECOND (or later) relay. The only git subprocess in
+    # this module; see the esc-097 module-doc paragraph for the arm order.
+    fix_head = data.get("fix_head")
+    fix_changed, why = _fix_window(row, fix_head)
+    if why is not None:
+        return why
+    probe_paths = {p for p in (_probe_path(e) for e in probe) if p}
+    if not any(p in fix_changed for p in probe_paths):
+        n = len(fix_changed)
+        sample = ", ".join(fix_changed[:3])
+        block7 = (row.get("head_sha") or "")[:7]
+        fix7 = fix_head[:7] if isinstance(fix_head, str) else "?"
+        return (
+            f"relay `probe` names none of the {n} files the fix changed "
+            f"(block {block7}..fix {fix7}); probe the fix, not the neighbourhood — e.g. {sample}"
+        )
     return None
```

### (b) `.claude/hooks/README.md`

```diff
--- a/.claude/hooks/README.md
+++ b/.claude/hooks/README.md
@@ -49,8 +49,9 @@
 
 **The relay artifact** (`.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json`)
 is written by the LEAD directly (`Write` is not gated) — never scanned from message
-prose. It names `unit_branch`/`agent_type`/`block_ts` (the verdict row's own `ts`) and must
-satisfy BOTH requirements — a CONJUNCTION, never a choice of arms (esc-064).
+prose. It names `unit_branch`/`agent_type`/`block_ts` (the verdict row's own `ts`) and,
+per esc-097 below, `fix_head` — and must satisfy R1, R2, and R3, a CONJUNCTION, never
+a choice of arms (esc-064, esc-097).
 **(1) Coverage** — whenever the BLOCK's `class_enumeration` is non-empty: a `sites`
 object whose keys are an EXACT-STRING SUPERSET of it (no path parsing, no
 normalization — the lead copies the verifier's own strings verbatim, so
@@ -63,11 +64,38 @@
 only shrink the adjacent set) and is therefore NOT the acceptance-easing
 normalization the `sites` rule bans. The `enumeration_missing` field on a verdict
 row is diagnostic only — no gate decision reads it; which requirement has content is
-derived from the enumeration itself. HONEST LIMIT: the hook enforces that adjacent
-probing is ASSERTED with named, citation-checkable sites — never that it occurred,
-nor that the sites are semantically adjacent; it converts silent omission into an
-explicit after-the-fact-checkable claim (citation-checker and the retrospective
-judge the probes, not this hook).
+derived from the enumeration itself.
+**(3) Probe-the-fix — ALWAYS (esc-097).** The relay names `fix_head` (the fix
+commit's full sha); the hook resolves `fix_changed = git diff --name-only
+<block_sha> <fix_head>` ITSELF (trusted, computed, never lead-supplied) and requires
+`fix_head` to resolve, differ from the BLOCK's own `head_sha` (else "no fix commit
+since the BLOCK; a second dispatch without a fix is a re-roll" — a re-roll, not a
+fix), and be reachable from the UNIT's own tip (`git merge-base --is-ancestor
+fix_head unit_branch` — NOT from `block_sha`, which an `--amend` sibling breaks; if
+the lead fixed on another branch or rebased, it names THAT branch's head as
+`fix_head`). At least one `probe` entry's PATH (the first whitespace-delimited
+token, trailing punctuation stripped, an optional trailing `:<n>[-<n>][,<n>]*` line
+spec stripped — an unparseable token counts for nothing, monotone toward DENY, never
+toward a false ALLOW) must be EXACTLY a member of `fix_changed` — probing the fix's
+own surface satisfies this even when that file is also a finding location; R2's
+>=2-distinct-non-reactive requirement is unchanged and stays conjunctive with R3
+(worst case, three probe entries: 2 adjacent + 1 fix-changed, though one entry can
+double as both when it qualifies for each). **§C5 — the ONE amendment to "no git
+subprocess anywhere":** git runs ONLY in this relay-validation arm (a second dispatch
+after a BLOCK), NEVER on a first dispatch's hot path, and ONLY in
+`$CLAUDE_PROJECT_DIR` (required explicitly — `repo_root()`'s cwd fallback is not
+reused here), with a 5s timeout; any git failure (non-zero exit, timeout, an
+unresolvable sha — an amend can orphan one) DENIES, naming the failing command, and
+states that `rm .jammi/gate-state/<slug>.*` is the escape hatch but destroys the
+unit's evidence rather than fixing the underlying git problem. HONEST LIMIT
+(extended to R3): the hook enforces that adjacent probing AND fix-probing are each
+ASSERTED with a named, citation-checkable, ACTUALLY-a-fix-changed-file site — never
+that either was semantically examined; a lead can satisfy R3 by pasting a path out
+of its own diff. It converts silent neighbourhood-probing into fix-window-probing,
+on the record; it does not by itself close the class of "form-satisfied without
+substance" (esc-097) — the substantive rule (design-before-mechanism, one fix round
+per BLOCK) is a `.claude/agents/lead.md` discipline, not a hook (citation-checker and
+the retrospective judge the probes' substance, not this hook).
 The hook only ever READS this file, fresh, on every gate call — it never writes an
 "accepted" row itself, so a DENY can never leave a phantom acceptance behind.
```

### (c) `.claude/agents/lead.md`

```diff
--- a/.claude/agents/lead.md
+++ b/.claude/agents/lead.md
@@ -25,22 +25,25 @@
 
 Every verifier card REQUIRES `class_enumeration` in its verdict (the union, over every BLOCK-severity finding, of every sibling site its own sweep found, plus `sweep_method` naming how it swept and `exhaustive` stating whether it is confident it found every member) — demand this in every phase-4/5/6 audit brief you write, and treat a verdict that omits it (or reports `sweep_method: "none"`) as a verifier that did not sweep, not as evidence the class is empty.
 
-`hooks/lead-gate-pre.sh` mechanizes ONE choke point of this rule — the expensive one (F10's incident was the audit-round loop): a **second dispatch of the SAME verifier type** (`adversarial-audit`/`fix-verifier`/`acceptance-verifier`), whose prompt names, as a whole token (never a raw substring — `ci/gpu` does not gate `ci/gpu-dev`), an open BLOCK's recorded `worktree` (or a path under it), `head_sha` (full or a >=7-char prefix), or exact `unit_branch`, is denied unless an **accepted relay artifact** exists for that `(unit, agent_type, block_ts)`. Two rounds (r1, r2) both found that trying to detect "did the lead relay the class" from FREE-TEXT (a site regex, a token scan, a write-verb walk) always squeezed between jamming legitimate traffic and being dodged by a rewording — v3 is a mechanism change, not a third patch: it stopped trying to read prose and made the artifact structured instead. The gate's mechanism, relay-artifact format, and design history are documented in `docs/plans/53-agentic-swarm/ARCHITECTURE.md` §7 ("Enforcement") and `LESSONS.md` (family F10, "Per-mechanism").
+`hooks/lead-gate-pre.sh` mechanizes ONE choke point of this rule — the expensive one (F10's incident was the audit-round loop): a **second dispatch of the SAME verifier type** (`adversarial-audit`/`fix-verifier`/`acceptance-verifier`), whose prompt names, as a whole token (never a raw substring — `ci/gpu` does not gate `ci/gpu-dev`), an open BLOCK's recorded `worktree` (or a path under it), `head_sha` (full or a >=7-char prefix), or exact `unit_branch`, is denied unless an **accepted relay artifact** exists for that `(unit, agent_type, block_ts)`. Two rounds (r1, r2) both found that trying to detect "did the lead relay the class" from FREE-TEXT (a site regex, a token scan, a write-verb walk) always squeezed between jamming legitimate traffic and being dodged by a rewording — v3 is a mechanism change, not a third patch: it stopped trying to read prose and made the artifact structured instead. The gate's mechanism, relay-artifact format, and design history are documented in `docs/plans/53-agentic-swarm/ARCHITECTURE.md` §7 ("Enforcement") and `LESSONS.md` (family F10, "Per-mechanism"). esc-097 (R3) adds one further requirement to that same relay, armed unconditionally: the relay must also probe the FIX's own diff, not just the class's neighbourhood — see below.
 
 **Write the relay artifact yourself, explicitly** — `.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json` (`Write` is not gated):
 ```json
 {"unit_branch": "<the verifier's own unit_branch>", "agent_type": "adversarial-audit",
  "block_ts": "<the BLOCK row's own ts — read .jammi/gate-state/<slug>.jsonl to get it>",
+ "fix_head": "<the fix commit's full sha — the commit you are re-dispatching ON TOP of>",
  "sites": {"<verbatim site string from class_enumeration>": "<what you did about it>", ...},
  "probe": ["<a site you EXAMINED and found clean — outside class_enumeration and findings>",
-           "<another such site>"]}
+           "<another such site, OR a path the fix itself changed>"]}
 ```
-The relay's requirements are a CONJUNCTION, never a choice of arms (esc-064). **(1) Coverage** — whenever the BLOCK carries a non-empty `class_enumeration`: `sites.keys()` must be an EXACT-STRING SUPERSET of it — copy its strings verbatim, do not reformat or normalize them (`Makefile:12`, `src/a.rs`, `a.rs:10-12` are all fine as-is). Every disposition value must be non-empty (a site you dispute must still say why: `"not a member because …"`). **(2) Proactivity — ALWAYS required**, whether or not the verifier enumerated: a `"probe"` array naming ≥2 DISTINCT sites you actually EXAMINED and found clean (or fixed preemptively), outside both the `class_enumeration` and every `findings[].location`. This is your adjacent sweep on the record — on a single-file exhaustive BLOCK, name the caller, the test file, or the sibling function you checked; ≥2 examined-clean sites always exist and are always meaningful. A relay that only restates the verifier's enumeration is reactive acknowledgment and is denied with a reason naming the missing probe evidence — the remedy is to run the sweep and name what you examined, never `rm` the state. The hook only ever READS this file, fresh, on every gate call — there is no separate "accepted" state to fall out of sync.
+The relay's requirements are a CONJUNCTION, never a choice of arms (esc-064, esc-097). **(1) Coverage** — whenever the BLOCK carries a non-empty `class_enumeration`: `sites.keys()` must be an EXACT-STRING SUPERSET of it — copy its strings verbatim, do not reformat or normalize them (`Makefile:12`, `src/a.rs`, `a.rs:10-12` are all fine as-is). Every disposition value must be non-empty (a site you dispute must still say why: `"not a member because …"`). **(2) Proactivity — ALWAYS required**, whether or not the verifier enumerated: a `"probe"` array naming ≥2 DISTINCT sites you actually EXAMINED and found clean (or fixed preemptively), outside both the `class_enumeration` and every `findings[].location`. This is your adjacent sweep on the record — on a single-file exhaustive BLOCK, name the caller, the test file, or the sibling function you checked; ≥2 examined-clean sites always exist and are always meaningful. **(3) Probe the fix — ALWAYS required (esc-097):** write `fix_head` (the fix commit's full sha, reachable from THIS unit's own tip — its own branch's head if you fixed elsewhere or rebased) and make at least one `probe` entry EXACTLY name a path the fix itself changed (`git diff --name-only <the BLOCK's head_sha> <fix_head>`, which the hook recomputes itself — never trust your own recollection of what the fix touched). Probing the fix's own surface satisfies R3 even when that path is also a finding location; R2's own >=2-distinct requirement is unchanged and conjunctive, so the worst case is **three** probe entries (2 adjacent-but-clean + 1 fix-changed), though one entry can double as both when it happens to qualify for each. A relay that only restates the verifier's enumeration, or that never names a file the fix actually changed, is denied with a reason naming the missing evidence — the remedy is to run the sweep / cite the fix's own diff, never `rm` the state.
 
-**Operator escape hatch:** `rm .jammi/gate-state/<slug>.*` clears every row and relay artifact for a unit — use it on a stale BLOCK (e.g. a reused branch name) or to force a reset by hand.
+**Design-before-mechanism — REQUIRED, every session, before a non-local fix (esc-097).** A fix that only edits values/branches WITHIN an existing call is a **local correction** — dispatch it directly. A fix that instead ADDS or REWRITES a script, CI action, workflow step, or module — a **mechanism** — is different in kind: five consecutive esc-097 BLOCKs on `feat/deploy-shapes-E1-arm64-ci-base` each landed on the new mechanism the PREVIOUS fix introduced, never a local correction (the control, `fix/476-seam-counter-test-isolation`, was pure local corrections and converged in one round). Before dispatching a non-local fix, write a **one-paragraph mechanism contract** — where its value lives, EVERY reader of that value, every failure mode and its behavior, and what actually executes the path in CI (not merely what you intend to execute) — and dispatch **ONE `pressure-tester` round** against that contract, on THIS unit (`unit_branch` in the dispatch, so its verdict row lands in the unit's own file, never the `UNBOUND` fallback bucket), before the implementer. **Budget one fix round per BLOCK** — a second BLOCK on the same mechanism is a signal to stop and redesign, not to relay again; that is the stopping rule, and it lives here, in your judgment, not in a gate (no round counter, no cap is mechanized — R3 above only redirects WHERE you probe, it does not count rounds).
 
-**Four documented residuals, not claimed closed** (each with a runtime tell): relaying to a running agent by `SendMessage` is entirely out of scope by design (round 1+2 proved free-text message-relay detection jams legitimate freeze/status/stand-down/hygiene/advisory-fold traffic — the loop is choked at the dispatch gate instead, not at the message); an "unlabeled" verifier re-dispatch naming none of the three anchors; `disableAllHooks` in local settings; a relay whose `probe` sites are asserted but never examined — mechanically indistinguishable at the hook, and the tell is a probe site the next round's citation-checker cannot corroborate. See `hooks/README.md`'s "mechanical vs. visible-only" paragraph for the exact statement.
+**Operator escape hatch:** `rm .jammi/gate-state/<slug>.*` clears every row and relay artifact for a unit — use it on a stale BLOCK (e.g. a reused branch name) or to force a reset by hand; it does NOT excuse writing `fix_head` or probing the fix — it destroys the unit's evidence, it does not supply it.
 
+**Four documented residuals, not claimed closed** (each with a runtime tell): relaying to a running agent by `SendMessage` is entirely out of scope by design (round 1+2 proved free-text message-relay detection jams legitimate freeze/status/stand-down/hygiene/advisory-fold traffic — the loop is choked at the dispatch gate instead, not at the message); an "unlabeled" verifier re-dispatch naming none of the three anchors; `disableAllHooks` in local settings; a relay whose `probe` sites (including a fix-changed one) are asserted but never examined — mechanically indistinguishable at the hook, and the tell is a probe site the next round's citation-checker cannot corroborate. R3 converts neighbourhood-probing into fix-probing; it does not by itself close the class (esc-097) — the substance is the design-before-mechanism paragraph above. See `hooks/README.md`'s "mechanical vs. visible-only" paragraph for the exact statement.
+
 ## The phase machine (ARCHITECTURE §4)
 
 Run the fixed pipeline; each phase names the agent(s) you dispatch and the gate it clears. Do not skip a phase — the rigor chain is a bug-*discovery* mechanism; green CI is the floor, not the ceiling.
```

### (d) `.claude/agents/adversarial-audit.md`

```diff
--- a/.claude/agents/adversarial-audit.md
+++ b/.claude/agents/adversarial-audit.md
@@ -36,7 +36,7 @@
 
 ## The class you enumerate, not just the instance
 
-Beyond each individual finding, **sweep for the class it belongs to** — the same shape of violation at other sites the diff (or the tree the diff touches) also carries. A BLOCK you hand the lead with only the one site you happened to spot invites round-by-round relaying of one instance at a time (SELF-FAILURE-MODES F10); your own `class_enumeration` is what lets the lead's dispatch gate require the whole class in one shot. Report `unit_branch` (from `git -C <worktree-or-cwd> rev-parse --abbrev-ref HEAD` if you can resolve it, else the `unit:` line the lead's brief carried — report which source you used) and `head_sha` (`git rev-parse HEAD` at the same location) so the lead does not have to re-derive them. `class_enumeration` is the union, over every BLOCK-severity finding, of every `path:line` your sweep found in that finding's class — empty only when `sweep_method: "none"` (you did not sweep). `exhaustive: true` only when you are confident the sweep found every member, not merely the ones near the diff.
+Beyond each individual finding, **sweep for the class it belongs to** — the same shape of violation at other sites the diff (or the tree the diff touches) also carries. A BLOCK you hand the lead with only the one site you happened to spot invites round-by-round relaying of one instance at a time (SELF-FAILURE-MODES F10); your own `class_enumeration` is what lets the lead's dispatch gate require the whole class in one shot. Report `unit_branch` (from `git -C <worktree-or-cwd> rev-parse --abbrev-ref HEAD` if you can resolve it, else the `unit:` line the lead's brief carried — report which source you used) and `head_sha` (`git rev-parse HEAD` at the same location) so the lead does not have to re-derive them — **`head_sha` is exactly the head you READ this diff at**, nothing else; the lead-proactivity gate's esc-097 (R3) arm uses your reported `head_sha` as the fix window's own starting point (`block_sha`) when it later validates a relay against the fix's real diff, so a stale or guessed value here is not a cosmetic error. `class_enumeration` is the union, over every BLOCK-severity finding, of every `path:line` your sweep found in that finding's class — empty only when `sweep_method: "none"` (you did not sweep). `exhaustive: true` only when you are confident the sweep found every member, not merely the ones near the diff.
 
 ## Verdict schema
```

### (e) `.claude/agents/pressure-tester.md`

```diff
--- a/.claude/agents/pressure-tester.md
+++ b/.claude/agents/pressure-tester.md
@@ -33,7 +33,7 @@
 
 ## Reporting the unit, not just the plan
 
-Report `unit_branch` (`git -C <worktree> rev-parse --abbrev-ref HEAD` if resolvable, else the `unit:` line the lead's brief carried — say which) and `head_sha` (`git rev-parse HEAD` at the same location). `pressure-tester` is never gated by `hooks/lead-gate-pre.sh` (a design attack precedes any diff, so there is nothing yet to relay), but the state carrier reads its verdict row like every other verifier's — report `class_enumeration` when a REFINE/KILL finding has siblings worth naming (other plan sections carrying the same wrong-abstraction/band-aid shape); `sweep_method: "none"` when you did not sweep.
+Report `unit_branch` (`git -C <worktree> rev-parse --abbrev-ref HEAD` if resolvable, else the `unit:` line the lead's brief carried — say which) and `head_sha` (`git rev-parse HEAD` at the same location — again, exactly the head you read, nothing else). `pressure-tester` is never gated by `hooks/lead-gate-pre.sh` (a design attack precedes any diff, so there is nothing yet to relay), but the state carrier reads its verdict row like every other verifier's — report `class_enumeration` when a REFINE/KILL finding has siblings worth naming (other plan sections carrying the same wrong-abstraction/band-aid shape); `sweep_method: "none"` when you did not sweep. **When the lead dispatches you for a design-before-mechanism pass on a fix (esc-097, `.claude/agents/lead.md`)**, your `unit_branch` MUST be that unit's own branch, never omitted and never a generic/placeholder value — the lead's relay validation reads your `PROCEED` row from THAT unit's own `.jammi/gate-state/<slug>.jsonl` file, and a row that lands in the `UNBOUND` bucket (the fallback for a dispatch that named no resolvable unit) cannot satisfy it.
 
 ## Verdict schema
```
