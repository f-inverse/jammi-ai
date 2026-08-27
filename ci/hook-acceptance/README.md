# Hook acceptance — the fresh-session execution-provenance step

`ci/scripts/check_lead_gate.py --self-test` proves the lead-proactivity gate's hook
scripts (`lead-gate-{start,stop,pre}.sh`, `lead-gate-lib.py`) do exactly what their
code says on fixture payloads — including, as of v2 (`CONTRACT-v2-amendments.md`,
after a round-1 adversarial audit's 9-finding BLOCK), the audit's own end-to-end
dodge reproduction ported verbatim as fixtures D1-D5. It is **hermetic** and runs in
every CI job. It CANNOT prove two things, because hook configuration is snapshotted
at session start and the exact JSON payload shape Claude Code's harness actually
sends is not fully documented (`FACTS-hooks.md` — `SubagentStart`/`SubagentStop`
field names beyond `agent_type`/`agent_id`/`last_assistant_message` are unverified):

1. **That the harness honors a `PreToolUse` deny at all** — the self-test invokes
   `lead-gate-pre.sh` directly as a subprocess; it never goes through the actual
   Claude Code tool-dispatch path, so it cannot observe whether an exit-2 deny
   actually stops a real `Agent`/`Task`/`SendMessage` call.
2. **The exact payload field names** `SubagentStart`/`SubagentStop`/
   `PreToolUse(SendMessage)` carry — `lead-gate-lib.py` tries several plausible
   field names per value (`_first_str` in that file) and degrades gracefully if
   none match, but this is a documented BEST EFFORT, not a verified schema.

Both require a **fresh session** on this PR's branch (hook config loads once, at
session start) with the hooks actually wired via `.claude/settings.json`.

## What to run (once, manually, before this PR leaves draft)

1. Start a **fresh** Claude Code session with this branch checked out (hook config
   must be freshly loaded — a session that has been running since before this PR's
   `settings.json` change will not have the new hooks wired).
2. Confirm `.jammi/gate-state/hook.log` does not yet exist (or note its current
   line count) — `wc -l .jammi/gate-state/hook.log 2>/dev/null || echo 0`.
3. Perform, in order:
   - **One real subagent dispatch** (any `Agent` dispatch the lead would normally
     make — e.g. dispatch `citation-checker` on a trivial read-only check). This
     should produce a `SubagentStart` line and, at minimum, a `PreToolUse` line
     with `tool_name` recording whatever value the harness actually sends (settle
     P1: is it `Agent`, as the transcript census in `PROBE.md`/`DECISIONS.md`
     found, or something else?).
   - **One real `SendMessage`** to a running agent (any legitimate message — a
     status check is fine).
   - **One verifier completion** (let any dispatched verifier — `citation-checker`
     is the cheapest, always-allowed choice — run to its `Stop` event) to produce
     a `SubagentStop` line.
4. `cat .jammi/gate-state/hook.log` (or `tail` the new lines since step 2) and
   confirm:
   - at least one line has `"event": "SubagentStart"` with non-empty
     `payload_keys` — record what keys actually arrived;
   - at least one line has `"event": "SubagentStop"` with non-empty
     `payload_keys`;
   - at least one line has `"event": "PreToolUse"` whose `tool_name` matches what
     `.claude/settings.json`'s `PreToolUse` matcher expects (`Agent`/`Task` —
     `SendMessage` is deliberately NOT wired: message-relay gating is out of
     scope per the v3 core cut, `lead-gate-pre.sh:10` / `lead-gate-lib.py:27`,
     so a real `SendMessage` producing no `PreToolUse` line is correct, not a
     dead route) — if it does NOT (e.g. the harness sends some other value),
     that is itself the finding: the matcher needs re-homing in a follow-up PR,
     and this log is the evidence, not a guess.
5. Copy the relevant `hook.log` lines (not the whole operational log — trim to the
   lines the three events above produced) into
   `ci/hook-acceptance/2026-08-26-lead-gate.log` in this directory, committed
   alongside this PR, and cite it in the PR description.
6. If step 4's `tool_name` check fails (the dispatch tool is not what
   `.claude/settings.json` matches), re-home the matcher in the SAME PR before it
   leaves draft — a hook wired to a matcher the log shows never fires is the exact
   `agent-routing-gate.sh` dead-routing-gate class this PR already fixed once; do
   not ship a second instance of it knowingly.

## Why this PR is draft until the artifact exists

The acceptance-verifier BLOCKS this unit without `ci/hook-acceptance/
2026-08-26-lead-gate.log` present — "done" without the artifact is not done
(`lead.md`, the load-bearing stance). `check_lead_gate.py --self-test` proves the
mechanism is internally correct; only a fresh-session log proves the harness
actually calls it the way `.claude/settings.json` assumes.
