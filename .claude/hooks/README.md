# Swarm hooks — most are advisory/fail-open; one is fail-closed by design

Three hooks (`build-env-guard.sh`, `stop-gate.sh`, `agent-routing-gate.sh`) are
**discipline, not enforcement** — labeled as such in `ARCHITECTURE.md §7` (Soft:
advisory, fail-open). Nothing in that group blocks a correct action or replaces a gate.

**One hook family is different by design.** `lead-gate-{start,stop,pre}.sh` (the
lead-proactivity gate) is the swarm's first **fail-closed** hook: `lead-gate-pre.sh`
DENIES (exit 2 + a reason) on an internal error, a missing `python3`, or an
unreadable state directory — it does not silently allow. This is a deliberate,
named exception to the "hooks are advisory" default above, not a drift from it; see
`ARCHITECTURE.md §2.3` and `§7` for why one hard hook does not contradict "hooks are
discipline" as a *default* posture, and `docs/swarm/SELF-FAILURE-MODES.md` **F10** for
the incident that motivated it. `lead-gate-start.sh`/`lead-gate-stop.sh` are pure
state WRITERS (they record a verifier's own verdict and an agent's unit binding) and
never deny — blocking a subagent's own start/stop event would be actively harmful.

**Armed by default.** `settings.json` wires `agent-routing-gate.sh` on
`PreToolUse(Agent|Task)` and the `lead-gate-*` family (`SubagentStart`,
`SubagentStop`, and `PreToolUse(Agent|Task)`). `build-env-guard.sh` and
`stop-gate.sh` stay opt-in — wire them yourself if you want the extra nudge (see
"How to wire" below).

## The hooks

### `lead-gate-{start,stop,pre}.sh` — the lead-proactivity gate, FAIL-CLOSED (v3, core)
Rounds 1 and 2 both found the SAME shape of bug in this hook: a predicate over FREE
TEXT (site regexes, worktree/sha/token scans, write-verb walks, tag scans) whose
input domain is unbounded, so every patch moved the squeeze between "jams legitimate
traffic" and "dodged by a rewording." v3 is a mechanism change, not a third patch,
and — under a usage-limit scope cut mid-round — narrows the gate to **ONE choke
point**, dropping `SendMessage` gating, implementer-dispatch binding, and the Bash
backstop ENTIRELY (not log-only; see "What v3 dropped" below).

**The one gate.** A `subagent_type` in neither `GATED_TYPES` nor `NEVER_GATED_TYPES`
is DENIED (closed-world lattice, deny-unknown; `ci/scripts/check_lead_gate.py`'s `L2`
fixture cross-checks every `.claude/agents/*.md` card's `tools:` frontmatter against
it). Beyond that: a fresh `Agent`/`Task` dispatch whose `subagent_type` is one of the
THREE verifier-exit types (`adversarial-audit`/`fix-verifier`/`acceptance-verifier`)
is denied iff its prompt names, as a WHOLE TOKEN (never a raw substring — an open
BLOCK on `ci/gpu` does not gate `ci/gpu-dev`), an open BLOCK's recorded `worktree`
(or a path under it), its recorded `head_sha` (full, or any prefix of at least 7
characters — this repo's short-sha convention), or its exact `unit_branch`, of the
SAME `agent_type`, AND no
**accepted relay artifact** exists for that `(unit, agent_type, block_ts)`. A first
dispatch of any agent_type is structurally never gated (no prior row to match). No
other `Agent`/`Task` dispatch, and no `SendMessage`/`Bash` call, is decided by this
hook at all — they pass straight through.

**The relay artifact** (`.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json`)
is written by the LEAD directly (`Write` is not gated) — never scanned from message
prose. It names `unit_branch`/`agent_type`/`block_ts` (the verdict row's own `ts`) and,
per esc-097 below, `fix_head` — and must satisfy R1, R2, and R3, a CONJUNCTION, never
a choice of arms (esc-064, esc-097).
**(1) Coverage** — whenever the BLOCK's `class_enumeration` is non-empty: a `sites`
object whose keys are an EXACT-STRING SUPERSET of it (no path parsing, no
normalization — the lead copies the verifier's own strings verbatim, so
`Makefile:12`, `src/a.rs`, `a.rs:10-12` are all fine), every disposition non-empty.
**(2) Proactivity — ALWAYS**, on every BLOCK relay of every gated verifier type: a
`probe` array naming ≥ 2 DISTINCT sites outside BOTH the `class_enumeration` and
every `findings[].location` — the lead's adjacent sweep on the record. Probe entries
are compared with Unicode Cf/Cc (zero-width/control) characters dropped and
surrounding whitespace stripped, on BOTH sides: this is monotone-toward-DENY (it can
only shrink the adjacent set) and is therefore NOT the acceptance-easing
normalization the `sites` rule bans. The `enumeration_missing` field on a verdict
row is diagnostic only — no gate decision reads it; which requirement has content is
derived from the enumeration itself.
**(3) Probe-the-fix — required on a REPEAT dispatch (esc-097).** R3 runs ONLY from
`_decide_verifier_dispatch`'s own repeat-dispatch branch — never on a FIRST dispatch
(no prior row exists to reach this arm at all) — and, per decision, for AT MOST ONE
targeted unit even when a prompt names more than one open BLOCK of this type (naming
more than one is denied outright — see "One unit per dispatch" below). The relay
names `fix_head` (the fix commit's full sha, `re.fullmatch(r"[0-9a-f]{7,40}")`); the
hook resolves `fix_changed = git diff --name-only -z <block_sha> <fix_head>` ITSELF
(NUL-split, so a path with a space or non-ASCII byte is still named correctly —
trusted, computed, never lead-supplied) and requires `fix_head` to resolve, differ
from the BLOCK's own `head_sha` (else "no fix commit since the BLOCK; a second
dispatch without a fix is a re-roll"). Reachability binds TWO things (V18, round-3
closure — supersedes the earlier "git-free only" design): FIRST, git-free, the
relay's OWN `unit_branch` field must `slugify()` to EXACTLY this BLOCK's own
`unit_slug` — this binds the unit's NAME, never an arbitrary branch the relay merely
asserts; a BLOCK row filed under the `UNBOUND` fallback bucket can never be satisfied
this way (no real branch slugifies to the literal string `UNBOUND` — re-dispatch
naming the unit, then hand-remove the stale row for this block from `UNBOUND.jsonl`,
never `rm` the shared file, which holds every other unit's rows too). SECOND, that
same `unit_branch` is resolved UNDER `refs/heads/` ONLY (`git rev-parse --verify
--end-of-options refs/heads/<name>^{commit}`, never a bare `<name>^{commit}` — a TAG
(or any other `refs/<kind>/<name>`) literally named like the branch otherwise wins
gitrevisions(7)'s own refs/tags-before-refs/heads disambiguation and resolves to the
TAG's target instead of the real branch's own tip; a name that only slugifies right
but names no real branch fails here too, closing the slugify-collision advisory) and
`fix_head` must `git merge-base --is-ancestor` that resolved tip — this binds
`fix_head`'s POSITION on the unit's own history, which the NAME check alone does not:
an amended-away orphan sha, or a sha that is a real commit on some UNRELATED branch,
both slugify-match the right unit's name while never being reachable from its tip.
`block_sha` and `fix_head` are themselves resolved via `git rev-parse --verify
--end-of-options <hex>^{commit}` (never `cat-file -e`, so the RESOLVED value can be
checked), and the resolved, full 40-hex object must START WITH the caller-supplied
hex — a ref (branch or tag) literally NAMED like a sha, or a short prefix of one, can
shadow the object it abbreviates the same way a same-named tag can shadow a branch;
the resolved, full sha is what every later git argv (`merge-base`, `diff`) actually
uses. An amended commit IS on the tip and allows; only a STALE relay naming the
pre-amend sha is denied — "fix_head <sha> is not on <unit_branch>; if the fix was
amended, name the amended sha; if it was committed on a child branch, commit or merge
it onto <unit_branch>". A unit whose worktree is on a DETACHED HEAD has no
`refs/heads/` entry to bind to and can never be relayed this way — name the unit's own
branch, or commit/merge the fix onto one, before dispatching the second round. On the
ACCEPT path, any of these calls that still wrote to stderr despite succeeding (e.g.
git's own `warning: refname '...' is ambiguous.` when a shadow happened to resolve to
a prefix-matching, correct object anyway) has that text appended to the
operator-facing ALLOW reason and its `hook.log` row, so a shadow stays visible even
when it did not change the outcome. At least one `probe` entry's PATH (the
first whitespace-delimited token — never a quote- or backtick-span rule; a fix
touching only a space-containing path is a documented limit — a surrounding backtick
or parenthesis and trailing punctuation stripped, an optional trailing
`:<n>[-<n>][,<n>]*` line spec stripped) must be EXACTLY a member of `fix_changed` —
probing the fix's own surface satisfies this even when that file is also a finding
location; R2's ≥2-distinct-non-reactive requirement is unchanged and stays
conjunctive with R3 (worst case, three probe entries: 2 adjacent + 1 fix-changed,
though one entry can double as both when it qualifies for each). **One unit per
dispatch.** If the prompt whole-token-names MORE THAN ONE open BLOCK of the same
type, the dispatch is denied outright, naming every targeted unit — R3 never
silently skips the others; dispatch each named unit separately. **§C5 — the ONE
amendment to "no git subprocess anywhere":** git runs ONLY from the repeat-dispatch
branch, NEVER on a first dispatch's hot path, and ONLY in `$CLAUDE_PROJECT_DIR` — the
documented hook environment contract: the harness always sets this variable for
every hook invocation, so `repo_root()`'s cwd fallback is never needed by this arm
and is deliberately not reused here — with FIVE git calls per decision (`rev-parse
--verify` block_sha, `rev-parse --verify` fix_head, `rev-parse --verify
--end-of-options refs/heads/...` unit_branch, `merge-base --is-ancestor`, `diff
--name-only`), all sharing ONE per-decision monotonic deadline (`_GIT_BUDGET_S =
5.0`, an absolute `time.monotonic()` value threaded through every call, never a
fresh 5s per call — the whole arm is bounded by 5s total, not 5x5s) (git >= 2.24
required: every invocation carries `--end-of-options` immediately before its
revision arguments, so a value shaped like an option — e.g. `--output=/tmp/x` — can
never be read as one; a pre-2.24 git fails that unrecognized-option check and
DENIES). Any git failure (non-zero exit, timeout, budget exhaustion, an
unresolvable/malformed sha — an amend can orphan one — or a resolved sha that does
not start with the hex the relay gave, a ref shadowing it) DENIES, naming the failing
command AND its stderr (read back from the same `tempfile.TemporaryFile()` the
command's output was captured to, after `wait()` returns — never left unread), and
states that `rm .jammi/gate-state/<slug>.*` is the escape hatch but destroys the
unit's evidence rather than fixing the underlying git problem.
**Migration.** A relay written to disk before this patch lands (no `fix_head`) stops
being acceptable the moment the patch IS applied — an in-flight relay must be
rewritten with `fix_head`, or the unit must re-relay; there is no grandfathering.
HONEST LIMIT (extended to R3): the hook enforces that adjacent probing AND fix-probing
are each ASSERTED with a named, citation-checkable site that is ACTUALLY a member of
the relevant set — never that either was semantically examined; a lead can satisfy R3
by pasting a path out of its own diff without reading it. It converts silent
neighbourhood-probing into fix-window-probing, on the record; it does not by itself
close the class of "form-satisfied without substance" (esc-097) — the substantive rule
(design-before-mechanism, one fix round per BLOCK) is a `.claude/agents/lead.md`
discipline, not a hook (citation-checker and the retrospective judge the probes'
substance, not this hook). A further residual: `unit_branch` is CONSTRAINED (bound to
the unit's own NAME and, as of V18, `fix_head`'s POSITION on it) but the branch name
itself is still lead-asserted, the same trust boundary `fix_head` already carries — a
lead who would misname `unit_branch` to game reachability could already misname
`fix_head` today. A PRE-EXISTING wrinkle, not introduced by R3: `_PASS_LIKE` is
checked globally when classifying a raw verdict value, so an adversarial-audit row
whose OWN verdict text merely contains a pass-like token ("verified"/"PROCEED") also
closes a BLOCK, independent of R3 entirely — out of scope for this proposal. A
further, honest limit: a unit whose worktree is on a DETACHED HEAD cannot be relayed
— `unit_branch` resolution is scoped to `refs/heads/` only (round-4), and a detached
HEAD has no `refs/heads` entry to bind to.
The hook only ever READS this file, fresh, on every gate call — it never writes an
"accepted" row itself, so a DENY can never leave a phantom acceptance behind.

**Clearing.** A same-`agent_type` PASS clears its own BLOCK — this is the ONLY way an
adversarial-audit BLOCK closes (that same-type PASS itself requires a relay that
passed R1+R2+R3 on the direct repeat-dispatch path above). esc-097 (V10) DELETES the
earlier cross-type clearing arm (a `fix-verifier`/`acceptance-verifier` PASS clearing
an older `adversarial-audit` BLOCK on the same unit whenever that BLOCK's own relay
was "accepted") — that arm's own acceptance check never ran R3, so a relay lacking
`fix_head` entirely could still clear a BLOCK it never probed the fix for. There is
now exactly ONE predicate (`_relay_rejection`) and exactly ONE reachable caller (the
repeat-dispatch branch): fix, verify, THEN re-dispatch the SAME verifier type once
more — the closing audit always re-runs, on the record.
**Operator escape hatch:** `rm .jammi/gate-state/<slug>.*` clears ALL state (rows and
relay artifacts) for a unit — the recovery for a stale BLOCK on a reused branch name,
or any other state you need to force-reset by hand; it does NOT excuse writing
`fix_head` or probing the fix — it destroys the unit's evidence, it does not supply it.

**Verdict parsing.** `lead-gate-stop.sh` (`SubagentStop`) takes the LAST fenced
` ```json ` block of the verifier's final message whose object has `"kind": "verdict"`
(dropped `<verdict>…</verdict>` tags from the cards; kept as a ONE-RELEASE fallback
when no fenced block exists). Both paths hand off to the SAME JSON-string-aware
object extractor starting right AFTER the opening marker — neither ever searches for
a CLOSING marker, so a `</verdict>` or a stray `}` sitting inside a quoted `notes`
string can never truncate the region early (round-2 finding 6 — a real BLOCK no
longer fails open into an `UNPARSEABLE` row just because a verifier quoted its own
schema). An `UNPARSEABLE` row is filed under the agent's OWN `SubagentStart` binding,
never a shared `UNBOUND` bucket, unless no binding exists either. `lead-gate-start.sh`
is a pure, ADVISORY-ONLY writer — its binding is consulted ONLY to file an
`UNPARSEABLE` row correctly, never for a gate decision. `lead-gate-pre.sh`
(`PreToolUse`) is the sole decider: exit 0 = allow, exit 2 + stderr = deny (the `.sh`
wrapper maps EVERY non-zero python exit to 2 — never propagates a raw interpreter
exit code). Logic lives in `lead-gate-lib.py`; every invocation of all three is
logged to `.jammi/gate-state/hook.log` (event, tool_name, agent_type, decision) — the
firing proof `ci/scripts/check_lead_gate.py --self-test` and `ci/hook-acceptance/`
cite. See `.claude/agents/lead.md` "The class, not the instance" for the rule this
mechanizes.

**What v3 dropped (not log-only — deleted).** `SendMessage` gating and all
message-prose parsing: round 1 and 2 both proved free-text relay detection on message
prose is undecidable without jamming legitimate freeze/status/stand-down/hygiene/
advisory-fold traffic (round-2 finding 1) — the loop is choked at the verifier-
dispatch gate instead. Implementer-dispatch binding: the same free-text-domain
problem, one channel over. The Bash backstop (its shlex-token detectors scanned the
WHOLE command line past `&&`/`;`/`|`, round-2 finding 3, jamming legitimate compound
reads and missing real writes past a separator) — the mechanical control is
`permissions.deny` on `Edit`/`Write`/`MultiEdit` of the hook files in
`.claude/settings.json`, unchanged and still active.

**What is mechanical vs. visible-only (stated exactly, not overclaimed).** Mechanical:
a second dispatch of the SAME verifier type, whole-token-bound to an open BLOCK,
is denied without an accepted relay artifact; an unrecognized `subagent_type` is
denied; `permissions.deny` blocks an agent-initiated Edit/Write/MultiEdit on the hook
files. Visible-only, each with the SAME runtime tell (the next verdict row on that
`(unit, agent_type)` with no accepted relay artifact between it and the prior BLOCK,
visible by comparing `.jammi/gate-state/<unit>.jsonl` timestamps against the relay
artifact files present): (1) relaying to a running agent BY MESSAGE — `SendMessage`
is out of scope by design, not merely undetected; (2) an "unlabeled" verifier
re-dispatch naming neither the recorded worktree, head_sha, nor unit_branch (DODGE-5);
(3) `disableAllHooks` in local settings (a verifier verdict in the transcript with no
hook-written state row is the tell); (4) a relay whose `probe` sites are
asserted but never examined — mechanically indistinguishable here; the tell is a
probe site the next round's citation-checker cannot corroborate.

### `build-env-guard.sh` — `PreToolUse(Bash)`, opt-in, fail-open
Warns (stderr only, **always exit 0**) when a Bash command carries a build-env hazard
from family S (`LESSONS.md`), by generic pattern — no hardcoded incident:
- an `RUSTFLAGS` / `RUSTC_WRAPPER` override (changes the sccache key → full cache-miss
  rebuild);
- a `cargo build|test|check|clippy|…` with no unique `CARGO_TARGET_DIR` (inline env,
  `--target-dir`, or exported) → build-lock contention / stale-artifact test runs across
  worktrees;
- (best-effort) working-disk usage ≥ 90% before a `cargo`/`maturin` build (NVMe/target
  pressure);
- (best-effort) a `maturin` run whose `PYTHONPATH` does not include the current tree
  (cross-worktree extension shadowing).

It never blocks — a warning is a nudge, and a false positive costs nothing.

### `stop-gate.sh` — `Stop`, opt-in, loop-safe
On a **dirty tree only**, runs the P0 static checks that exist in the tree
(`check_swarm_bijection.py`, `check_constitution_anchors.py`, `check_doc_parity.py`) and
**blocks Stop (exit 2) only on a genuine P0 gate failure**. A clean tree, an all-green
run, a not-yet-wired gate script, or a missing interpreter → exit 0 (never block on
infrastructure, only on a real gate verdict). Honors `stop_hook_active` so a
blocked-then-resumed Stop does not recurse.

### `agent-routing-gate.sh` — `PreToolUse(Agent|Task)`, advisory
Nudges (stderr, **always exit 0**) when a dispatched `Agent`/`Task` reads as a
rigor-chain phase step (adversarial audit, pressure-test, fix-verify,
discipline/boundary check, cookbook re-emit, citation re-check, oracle hard-block) but
is not routed to the matching gate agent. Generic keyword heuristic; the real boundary
is the agent's `tools:`. Re-matched from `Task`-only to `Agent|Task` in the
lead-proactivity-gate PR — the pressure-test's census of this session's own
transcripts (2026-08-26) found the MODEL-side dispatch tool named `Agent` 475 times
and `Task` 0 times, so the original `Task`-only matcher had almost certainly been
silently dead the entire time it was wired. The hook PAYLOAD's own `tool_name` field
is a separate claim, confirmed only by the fresh-session log
(`ci/hook-acceptance/README.md`) — pending as of this PR.

## How to wire

`settings.json` (`.claude/settings.json`) wires the default-armed set:

```json
{
  "hooks": {
    "SubagentStart": [
      { "matcher": ".*",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/lead-gate-start.sh" } ] }
    ],
    "SubagentStop": [
      { "matcher": "adversarial-audit|fix-verifier|acceptance-verifier|pressure-tester|oracle|citation-checker|discipline-test-auditor",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/lead-gate-stop.sh" } ] }
    ],
    "PreToolUse": [
      { "matcher": "Agent|Task",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-routing-gate.sh" } ] },
      { "matcher": "Agent|Task",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/lead-gate-pre.sh" } ] }
    ]
  },
  "permissions": {
    "deny": [
      "Edit(.claude/hooks/**)", "Write(.claude/hooks/**)", "MultiEdit(.claude/hooks/**)",
      "Edit(.claude/settings.json)", "Write(.claude/settings.json)", "MultiEdit(.claude/settings.json)"
    ]
  }
}
```

To opt into the two still-advisory hooks, add matchers of your own (in
`.claude/settings.local.json` so the default shared posture stays minimal):

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/build-env-guard.sh" } ] }
    ],
    "Stop": [
      { "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/stop-gate.sh" } ] }
    ]
  }
}
```

Every hook here is POSIX `sh` (`lead-gate-*.sh` are thin wrappers over
`lead-gate-lib.py`) and reads the hook JSON payload on stdin. The three advisory hooks
degrade to a silent `exit 0` if a parser (`jq`/`python3`) or expected field is
unavailable. `lead-gate-pre.sh` does the opposite on the same condition — it fails
CLOSED (exit 2, never exit 1, which Claude Code treats as non-blocking; the wrapper
maps every non-zero python exit code onto this two-value lattice, never propagating a
raw interpreter exit code). It is the one hook in this directory built to be hard to
dodge BY REWORDING — see the "mechanical vs. visible-only" paragraph above for the
three residuals it does NOT close and their runtime tell; it is not claimed
un-dodgeable outright.
