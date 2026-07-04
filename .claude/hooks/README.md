# Swarm hooks — opt-in, advisory, fail-open

These three hooks are **discipline, not enforcement**. They are labeled as such in
`ARCHITECTURE.md §7` (Soft: advisory, fail-open). The only hard teeth in the swarm are
(a) each agent's native `tools:` capability set and (b) fail-closed CI gates once
required-in-branch-protection. Nothing here blocks a correct action or replaces a gate.

**Not armed by default.** `settings.json` wires only the advisory
`agent-routing-gate.sh` on `PreToolUse(Task)`. The other two are opt-in — wire them
yourself if you want the extra nudge (see "How to wire" below). This keeps the repo's
default posture honest: a hook is soft until you choose to run it.

## The hooks

### `build-env-guard.sh` — `PreToolUse(Bash)`, fail-open
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

### `agent-routing-gate.sh` — `PreToolUse(Task)`, advisory
Nudges (stderr, **always exit 0**) when a dispatched `Task` reads as a rigor-chain phase
step (adversarial audit, pressure-test, fix-verify, discipline/boundary check, cookbook
re-emit, citation re-check, oracle hard-block) but is not routed to the matching gate
agent. Generic keyword heuristic; the real boundary is the agent's `tools:`.

## How to wire

`settings.json` in this directory's parent (`.claude/settings.json`) wires only the
routing nudge:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Task",
        "hooks": [ { "type": "command",
                     "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-routing-gate.sh" } ] }
    ]
  }
}
```

To opt into the other two, add matchers of your own (in `.claude/settings.local.json` so
the default shared posture stays minimal):

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

All three are POSIX `sh`, read the hook JSON payload on stdin, and degrade to a silent
`exit 0` if a parser (`jq`/`python3`) or expected field is unavailable.
