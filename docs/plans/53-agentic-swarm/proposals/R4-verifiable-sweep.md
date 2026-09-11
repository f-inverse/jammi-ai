# PROPOSAL (human-merge): tighten(lead-gate) — a relay's sweep must be re-executable, and a repeat BLOCK must show a lane sweep the hook can run itself (R4)

Status: **DESIGN UNDER PRESSURE-TEST.** A `pressure-tester` round is running against this design as of
2026-09-11 (dispatched by the lead the same session this doc's own motivating incidents happened in
— `.jammi/ledger/program-6768-20260910.jsonl:85`). Per the lead's sequencing correction, **the `.patch`
file against `.claude/hooks/**` is NOT authored or committed in this pass** — only the pieces that
survive any verdict: the motivating incidents (cited by ledger row, not narrated), the mechanism as
currently specified, real replay evidence against today's hook, the six self-test fixture scenarios as
data, and the design tensions the pressure round is attacking, stated honestly rather than glossed
over. This doc will be revised once the verdict lands, and the `.patch` authored against the binding
refinements at that point — following the precedent this proposal itself follows,
`docs/plans/63-how-well/proposals/esc-097-probe-the-fix.md` (three pressure/adversarial rounds before
the mechanism shipped).

Touches (once the patch is authored): `.claude/hooks/lead-gate-lib.py`, `.claude/hooks/README.md`,
`ci/scripts/check_lead_gate.py`'s self-test fixtures. `SWARM_GATE_TOUCHED` (`.claude/hooks/**`,
`ci/scripts/check_lead_gate.py`) → proposal only, human admin-merge, matching the esc-097 precedent.

## The failure R1–R3 do not catch — three real incidents, cited by ledger row, not narrated

**1. A probe entry can be copied, not opened (motivates R4a).**
`.jammi/ledger/program-6768-20260910.jsonl:49`, the lead's own self-correction, verbatim: *"of 3 relay
artifacts, the client-deps probes were examined before writing, but 2 of 3 PR-A probe entries
(audit/record.rs:187, store/manifest.rs:1018) were copied from the auditor's notes rather than opened
— the exact residual `.claude/hooks/README.md:227` documents."* R2 (`lead-gate-lib.py:1018-1030`)
requires `probe` to name ≥2 distinct sites outside the enumeration/findings; it has no way to tell
whether a site string was ever opened, only that it is not a byte-for-byte copy of an enumeration or
finding-location string. `lead-gate-lib.py:130-134` (the module's own HONEST LIMIT paragraph) says
this precisely: *"a lead can still satisfy it by pasting a path out of its own change."*

**2. A lane-shaped BLOCK's fix can cover part of the class and R1 still passes (motivates R4b).**
`.jammi/ledger/program-6768-20260910.jsonl:41`, PR-A's round-2 closing-audit BLOCK, verbatim: *"the
deny.toml bans mechanism is right but its SCOPE is wrong — ci.yml:959 runs cargo deny on a
DEFAULT-FEATURE graph that excludes datafusion-table-providers{,-common,-mysql,-postgres}, the very
subtree most likely to carry a second DataFusion line."* The relayed fix
(`.jammi/ledger/program-6768-20260910.jsonl:69`) added exactly two of the four eventual
`cargo deny --all-features check {bans,advisories,licenses,sources}` lanes — `ci.yml:975-978`
(`bans`, `advisories`). The remaining two lanes (`licenses`, `sources`) were added only in the NEXT
round (`.jammi/ledger/program-6768-20260910.jsonl:80`: *"ci.yml:980-981 runs all four deny lanes with
--all-features"*). A second, independent instance of the exact same shape, in a different unit:
`.jammi/gate-state/feat_482-dist-data-plane.jsonl`'s `pressure-tester` row at ts
`2026-09-11T15:47:47.984824+00:00` carries, as ONE `class_enumeration` string (not five), *"crates/
jammi-db/src/index/placed.rs:225, :297, :336, :375, :441 — coordinator kernel entries, unguarded"* —
five call sites in one file, bundled into a single enumeration entry. R1's coverage check
(`lead-gate-lib.py:1004-1016`) requires `sites` to be an exact-string superset of `class_enumeration`
— since the verifier bundled five line numbers into one string, a relay's `sites` need only carry ONE
key (the entire bundled string) to satisfy R1, whether or not the fix touched all five lines. R1
cannot see inside a compound enumeration entry; it was never designed to look.

**3. A verifier type can BLOCK repeatedly on the same mechanism with nothing that forces a design pass
(motivates R4c).** `.jammi/gate-state/feat_482-dist-data-plane.jsonl` carries three adversarial-audit
rows to date: BLOCK at `07:53:12.988665` (round 1), BLOCK at `15:33:15.777101` (round 2 — later relayed
and ALLOWED under today's R1-R3; see the replay table below), BLOCK at `16:46:19.669946` (round 3, no
relay artifact yet). `.jammi/ledger/program-6768-20260910.jsonl:85` — the lead's own directive
authorizing this proposal — states the intended effect directly: R4c *"would have forced the DIST
redesign at round 3, not 4."* Today, `_decide_verifier_dispatch` (`lead-gate-lib.py:1137-1187`) and
`_relay_rejection` (`lead-gate-lib.py:963-1056`) treat every repeat dispatch identically regardless of
how many consecutive times the SAME unit has BLOCKed on the SAME `agent_type` before — there is no
round count anywhere in the gate's own state model.

## What R4 is, as currently specified (subject to revision once the verdict lands)

**R4a — verifiable sweep.** The relay gains a required `sweep` object: `{cmd, cwd, matches,
output_sha256}`. The hook re-executes `cmd` in `cwd` (read-only shell, 30s timeout, output capped at
1 MiB) and compares `sha256(stdout)` against `output_sha256`. Mismatch, timeout, non-zero exit, or a
`cmd` containing a write verb (`>`, `>>`, `rm`, `mv`, `git commit`, `sed -i`, `tee`) → DENY, naming the
reason. `matches` is informational only. This converts "I swept" from prose into a reproducible
artifact — the lead must have actually run the sweep to know its hash, and anyone can re-run it.

**R4b — lane completeness for lane-shaped BLOCKs.** When the R4a sweep's own output contains ≥2
`<path>:<line>:` lines in the SAME file, the relay's `sites` must cover EVERY such line, as an exact
`path:line` key with a non-empty disposition — not merely whatever compound string the verifier's own
`class_enumeration` happened to bundle them into (incident 2, above, is exactly this gap). The
candidate set is DERIVED from the sweep's OWN output, never from `class_enumeration` — this is the
point: R1 trusts the verifier's phrasing; R4b trusts only what the lead's own re-executed command
actually printed.

**R4c — a third intersecting BLOCK forces a redesign dispatch, not another fix round.** A THIRD BLOCK
of the same `agent_type` on the same `unit_slug`, whose `class_enumeration` intersects the
IMMEDIATELY PRIOR BLOCK's `class_enumeration` (same mechanism, not a new one), denies the next
dispatch if it targets an implementer-shaped type (`general-purpose` or a domain type) naming that
unit; the only accepted next dispatch is `pressure-tester`, and only when its prompt names a contract
file that the relay's own new `redesign_contract` field also names.

R4a/R4b/R4c are CONJUNCTIVE with R1–R3, in the same sense R2/R3 already are with each other
(`lead-gate-lib.py:973` — *"a CONJUNCTION, never a two-arm disjunction (esc-064)"*).

## Design tensions the pressure round is attacking — stated honestly

**R4c reaches code today's `_decide_dispatch` deliberately routes around, but the reopening is
narrower than what v3 rejected.** `_decide_dispatch` (`lead-gate-lib.py:1200-1223`) allows every
GATED non-verifier dispatch unconditionally the moment `subtype not in VERIFIER_SECOND_ROUND_TYPES`
(`lead-gate-lib.py:1218-1221`): *"Implementer-dispatch binding is OUT OF SCOPE by the round-3 core
cut."* The module's own doc groups this with `SendMessage` gating and the Bash backstop under one
scope cut (`lead-gate-lib.py:141-145`, *"dropped entirely, not log-only"*) — but unlike `SendMessage`
gating and the Bash backstop, which each have a NAMED round-1/round-2 jamming reproduction cited right
there (free-text relay detection on message prose; a shlex-token detector scanning past `&&`/`;`/`|`),
implementer-dispatch binding is not shown broken by either round — it was dropped under the SAME
usage-limit scope cut, not because it was independently found jammable. What R4c actually needs is
narrower than the general "implementer-dispatch binding" v3 rejected: it does not need to match an
implementer's `worktree`/`head_sha` the way verifier binding does — it only needs the SAME whole-token
`unit_slug` binding (`_block_named_in_text`, `lead-gate-lib.py:1119-1134`) reapplied to a type-class
outside `VERIFIER_SECOND_ROUND_TYPES`, gated by a NEW, narrow predicate (three consecutive
same-agent_type BLOCKs, enumeration-intersecting) rather than "any open BLOCK of this type." Whether
that narrower scope is safe from the SAME free-text-domain failure mode the original binding was
dropped for is exactly what the pressure round is testing.

**R4c needs round-counting history the hook does not currently query.**
`_unit_rows_by_agent_type` (`lead-gate-lib.py:528-544`) keeps only the LATEST row per `agent_type`;
`open_blocks_for_unit` (`lead-gate-lib.py:547-566`) only asks whether that latest row is
BLOCK-equivalent. Nothing today counts consecutive BLOCKs or diffs one BLOCK's `class_enumeration`
against the PRIOR BLOCK's own. R4c's predicate needs a walk over the SAME `read_rows(unit_file(...))`
call `_unit_rows_by_agent_type` already makes (`lead-gate-lib.py:532`), filtered to consecutive
same-`agent_type` BLOCK rows, with a set-intersection check between the last two — a new query shape,
not a field addition.

**The ALLOW half of R4c is already true today, for free.** `pressure-tester` is in
`NEVER_GATED_TYPES` (`lead-gate-lib.py:219`), and `_decide_dispatch` checks `NEVER_GATED_TYPES` BEFORE
`GATED_TYPES` (`lead-gate-lib.py:1211-1212`) — a `pressure-tester` dispatch is unconditionally allowed
regardless of any open BLOCK, today, with zero new code. R4c's real surface area is entirely the DENY
half: intercepting the `general-purpose`/domain-type branch at `lead-gate-lib.py:1218-1221` for the
one narrow (third-intersecting-BLOCK) case, leaving every OTHER implementer dispatch on that branch
unaffected.

**R4a's threat model: a lead-supplied `cmd` is still lead-authored.** Re-executing `sweep.cmd` and
comparing the hash proves the command was RUN and its output is what's claimed — it does not prove the
command was the RIGHT one. A lead wanting to dodge can write a real, deterministic, adjacent-but-
irrelevant `cmd` and still pass. This is the same shape of residual `lead-gate-lib.py:130-134` already
states for R2/R3: *"R3 cannot verify examination... the substantive rule is a
`.claude/agents/lead.md` discipline, not a hook."* Two alternatives under live consideration by the
pressure round would each close a DIFFERENT slice of this gap, not the same one: (a) sweep commands
drawn from a human-maintained, reviewed list keyed by BLOCK class (removes the lead-authorship problem
entirely, at the cost of needing that list maintained and kept in sync with new classes); (b) hashing
over the MATCHED LINES' TEXT rather than the line-numbered `grep -n` output (survives the swept file
being edited — line numbers shifting — between the sweep and the fix landing, which the current
line-numbered-output hash does not). Both are live threads; this doc is updated once the verdict picks
between them (or neither).

## Replay evidence against TODAY's real relay artifacts (R1–R3 baseline, not yet R4)

`ci/hook-acceptance/replay_relays.py` (committed alongside this doc) loads every
`(unit, agent_type, block_ts)` row with a recorded verdict, looks up its relay artifact by the SAME
`relay_artifact_path()` the hook itself uses, and re-runs `_relay_rejection` — the real function, not a
reimplementation — against the REAL `.jammi/gate-state/` directory, read-only (the only git calls made
are the ones the loaded lib's own R3 arm already makes: `rev-parse --verify`, `merge-base
--is-ancestor`, `diff --name-only`). Run on 2026-09-11 against today's `lead-gate-lib.py` (which
already carries R1–R3 / esc-097 — `RELAY_R3 = True` at `lead-gate-lib.py:205`):

```
python3 ci/hook-acceptance/replay_relays.py \
  --state-dir .jammi/gate-state --project-dir . --lib .claude/hooks/lead-gate-lib.py
```

255 rows replayed; **5 ALLOW, 250 DENY**. Full table: `ci/hook-acceptance/2026-09-11-r4-replay-baseline.log`.
Almost every DENY is one of two shapes, neither a finding about R1-R3 itself: `no relay artifact
exists` (a BLOCK the lead never relayed, or a PASS/UNPARSEABLE row a relay was never expected for), or
`relay unit_branch ... does not resolve under refs/heads/` (the unit's branch has since been merged and
its worktree removed — replaying LIVE state weeks after the fact against a `refs/heads/`-only
resolution, `lead-gate-lib.py:936-944`, correctly reports it as unreachable now). The 5 ALLOW rows:

| unit | agent_type | block_ts | `sites` keys | `probe` entries | carries `sweep`? |
|---|---|---|---|---|---|
| feat_482-dist-data-plane | adversarial-audit | 15:33:15.777101 | 20 | 3 | **no** |
| feat_482-graph-job-deps | adversarial-audit | 15:06:10.780145 | 18 | 3 | **no** |
| feat_500-A-datafusion-54 | adversarial-audit | 12:23:34.922027 | 34 | 3 | **no** |
| feat_500-A-datafusion-54 | adversarial-audit | 12:24:57.426904 | 34 | 3 | **no** |
| feat_500-A-datafusion-54 | adversarial-audit | 16:50:29.317998 | 34 | 3 | **no** |

None of the five carries a `sweep` key (checked directly against each artifact file). R4a, applied
with NO grandfathering — the same stance esc-097 itself took (`.claude/hooks/README.md:144-146`: *"A
relay written to disk before this patch lands... stops being acceptable the moment the patch IS
applied — an in-flight relay must be rewritten... there is no grandfathering"*) — would newly DENY all
five until the lead rewrites each with a real `sweep` object. This is the EXPECTED shape of the
evidence, not a finding that any of these five was itself a dodge: the `feat_500-A-datafusion-54` rows
at `12:23:34`/`12:24:57` are the SAME relay the "probe entries copied" incident (above) corrected
BEFORE these files were written to their current, well-examined state — they would simply need a
`sweep` object added, same as the other three. The migration cost is real and stated honestly: every
one of today's 5 accepted relays goes dark the moment R4a lands, until re-relayed.

## Six self-test fixture scenarios — data, not yet implementation (field names provisional)

Written as concrete BLOCK-row / relay-artifact data so they survive a change in predicate shape;
`ci/scripts/check_lead_gate.py`'s own G20-40 precedent (the `RELAY_R3` version-marker guard,
`check_lead_gate.py:1985-2019`) is the harness pattern these will slot into once the patch exists —
reported SKIPPED until a human applies the patch, run for real the moment it is.

**1. Sweep hash mismatch → DENY.** BLOCK row: `class_enumeration=["a.py:1","b.py:2"]`. Relay:
`sites` covers both, `probe` names 2 adjacent sites, `fix_head` valid and reachable, `sweep={"cmd":
"grep -n 'pattern' a.py b.py", "cwd": "<worktree>", "output_sha256": "<a wrong hash — one byte
flipped from the command's real output>"}`. Expected: DENY, reason names the sha mismatch (never
silently re-hashes and "corrects" it).

**2. A write verb in `cmd` → DENY.** Same BLOCK/relay as (1), but `sweep.cmd = "sed -i
's/foo/bar/' a.py"`, with a `output_sha256` that DOES match that command's real (mutating) output.
Expected: DENY on the write-verb check alone — never even re-executed, since a `sed -i` command must
not run as part of a gate decision. Reason names the write verb.

**3. Lane-shaped BLOCK, 2 of 4 sweep-derived lines covered → DENY, naming the 2 missing.** BLOCK row:
`class_enumeration=["a.py:10, :20, :30, :40 — four call sites, unguarded"]` (ONE bundled string,
matching incident 2's real shape). `sweep.cmd` is a real `grep -n` whose real output contains exactly
`a.py:10:`, `a.py:20:`, `a.py:30:`, `a.py:40:`. Relay's `sites` carries the bundled `class_enumeration`
string as its one key (satisfying R1 alone) but no `a.py:10`/`a.py:20`/`a.py:30`/`a.py:40` keys.
Expected: DENY, reason names `a.py:30` and `a.py:40` (the two not independently covered) — proving the
candidate set is derived from the SWEEP OUTPUT, not from how the verifier phrased the enumeration.

**4. Same lane, all 4 sweep-derived lines covered → ALLOW.** Same as (3), but `sites` additionally
carries `a.py:10`/`a.py:20`/`a.py:30`/`a.py:40`, each with a non-empty disposition. Expected: ALLOW
(R1/R2/R3/R4a/R4b all satisfied).

**5. Third intersecting BLOCK, implementer re-dispatch → DENY.** Three consecutive `adversarial-audit`
BLOCK rows on the same `unit_slug`, each `class_enumeration` sharing ≥1 entry with the one immediately
before it (same mechanism). A `general-purpose` dispatch whose prompt whole-token-names the unit
(worktree or `unit_branch`). Expected: DENY — the FIRST case in this proposal where a NON-verifier
`subagent_type` is denied by this hook at all; reason names the three BLOCK timestamps and states that
only a `pressure-tester` naming a `redesign_contract`-matching file is accepted next.

**6. Same state, pressure-tester naming the matching contract → ALLOW.** Same three-BLOCK state as
(5), but the dispatch is `pressure-tester` (already unconditionally allowed today —
`lead-gate-lib.py:219`/`1211-1212` — so this fixture's job is to prove R4c does not accidentally start
denying it) AND the third BLOCK's relay carries `redesign_contract` naming a file the `pressure-tester`
prompt also names, verbatim. Expected: ALLOW.

## Ledger lifecycle

No escape id assigned yet — `.jammi/escapes.jsonl` carries no `lead-gate-R4`/`verifiable-sweep` row as
of this doc. One will be appended, following the `esc-097-relay-form-satisfied-without-probing-the-fix`
precedent (`docs/plans/63-how-well/proposals/esc-097-probe-the-fix.md`, "Ledger lifecycle" section),
once the pressure-tester's verdict lands and the `.patch` is authored against it.

## Session-local evidence

`scratchpad/…` paths referenced during this doc's authoring (the throwaway replay runs, the sweep-field
check script) are session-local working state, per this repo's own `.gitignore` (`scratchpad/` is
ignored) — never tracked or citable after the session ends. The durable evidence is
`ci/hook-acceptance/replay_relays.py`, `ci/hook-acceptance/2026-09-11-r4-replay-baseline.log`, and the
`.jammi/ledger/*`/`.jammi/gate-state/*` rows cited above by exact path, line, and timestamp.
