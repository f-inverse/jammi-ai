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
is denied iff its prompt contains, as an EXACT SUBSTRING, an open BLOCK's recorded
`worktree`, its recorded `head_sha` (full or the first 7 characters — this repo's
short-sha convention), or its exact `unit_branch`, of the SAME `agent_type`, AND no
**accepted relay artifact** exists for that `(unit, agent_type, block_ts)`. A first
dispatch of any agent_type is structurally never gated (no prior row to match). No
other `Agent`/`Task` dispatch, and no `SendMessage`/`Bash` call, is decided by this
hook at all — they pass straight through.

**The relay artifact** (`.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json`)
is written by the LEAD directly (`Write` is not gated) — never scanned from message
prose. It names `unit_branch`/`agent_type`/`block_ts` (the verdict row's own `ts`) and
a `sites` object whose keys must be an EXACT-STRING SUPERSET of the verifier's
`class_enumeration` (no path parsing, no normalization — the lead copies the
verifier's own strings verbatim, so `Makefile:12`, `src/a.rs`, `a.rs:10-12` are all
fine). When `class_enumeration` is empty (`enumeration_missing`), a `probe` array of
≥ 2 entries disjoint from every `findings[].location` string is the weaker fallback.
The hook only ever READS this file, fresh, on every gate call — it never writes an
"accepted" row itself, so a DENY can never leave a phantom acceptance behind.

**Clearing.** A same-`agent_type` PASS clears its own BLOCK. A `fix-verifier`/
`acceptance-verifier` PASS ALSO clears an older `adversarial-audit` BLOCK on the same
unit, but ONLY when that BLOCK's own relay artifact was accepted — the normal
workflow's resolution path (fix, verify, done — the audit need not always re-run).
**Operator escape hatch:** `rm .jammi/gate-state/<slug>.*` clears ALL state (rows and
relay artifacts) for a unit — the recovery for a stale BLOCK on a reused branch name,
or any other state you need to force-reset by hand.

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
a second dispatch of the SAME verifier type, exact-substring-bound to an open BLOCK,
is denied without an accepted relay artifact; an unrecognized `subagent_type` is
denied; `permissions.deny` blocks an agent-initiated Edit/Write/MultiEdit on the hook
files. Visible-only, each with the SAME runtime tell (the next verdict row on that
`(unit, agent_type)` with no accepted relay artifact between it and the prior BLOCK,
visible by comparing `.jammi/gate-state/<unit>.jsonl` timestamps against the relay
artifact files present): (1) relaying to a running agent BY MESSAGE — `SendMessage`
is out of scope by design, not merely undetected; (2) an "unlabeled" verifier
re-dispatch naming neither the recorded worktree, head_sha, nor unit_branch (DODGE-5);
(3) `disableAllHooks` in local settings (a verifier verdict in the transcript with no
hook-written state row is the tell).

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
