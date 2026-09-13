# PROPOSAL (human-merge): tighten(lead-gate) — anticipate before the fix, not after (esc-lead-gate-R12)

Status: PROPOSED. This branch (`proposal/lead-gate-R12-anticipation`) applies the change
directly to the tracked files it touches, rather than shipping as a standalone patch series
— `SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `ci/scripts/check_*.py`, `.claude/agents/*`,
`docs/swarm/CONSTITUTION.md`) fires on it and it is human admin-merged, exactly as every
other lead-gate change in this family (R7/R10/R11) is.

## The escape (session ad5aa3db, 2026-09-13, unit feat/500-B-U4a)

The tarball loader arm of `ci/scripts/bundle_cuda_libs.sh` BLOCKed in closing audits #2, #3
and #4 — each on the mechanism the PREVIOUS fix introduced (a builder-resolved soname passed
an `LD_LIBRARY_PATH` prepend; a vacuous pass on an `ldd` failure/empty report; a fix-3 set
cross-check that failed a CORRECT stage because `ldd` prints its own loader's `DT_NEEDED`
entry by absolute path). Every one of the three defects was discoverable by running the
mechanism once, on real input, before the re-audit. esc-lead-gate-R10's `open_question`
field, as designed, required only that the lead NAME an attack "for the next round to run" —
a schema requirement over lead-authored prose, never an executed check — which
institutionalised exactly this deferral; every relay between these rounds satisfied
`open_question` in full, and the round-4 relay's own text reads "the next round asks
whether …", naming the attack as a hand-off rather than running it. The user's own words,
verbatim: "it's not about the count … the lead proactively anticipates and acts even before
the next audit round's finding."

## v1 (KILLED by its own pressure round)

The first design attached every new obligation to the RELAY — the artifact written only
AFTER a fix commit already exists. A pressure-tester round on that design found it fillable
after the fact for the identical reason `open_question` was: nothing forced the lead to have
run anything before writing the relay, so a plausible-sounding, never-executed "attack" and a
genuinely-executed one were indistinguishable at the schema layer. v1 was killed outright,
not softened, and the load-bearing check was moved to the one moment that cannot be gamed
after the fact — an IMPLEMENTER dispatch, before any fix exists at all.

## v2 (the design this branch implements) — READER 1 / READER 2 / READER 3

**Where the value lives.** A NEW artifact under `.jammi/gate-state/`, distinct from the
relay (which is written only after the fix commit exists): `{unit_branch, pre_fix_sha,
covers, attacks: {"<file>": {command, hash}}, residual_risk}`.

**Reader 1 — implementer dispatch (the pre-fix moment).** When a dispatch of an
implementer-type agent (or, per fix round 1's M6' below, a generic/harness type that DOES
name a unit) names, via a `unit:` line, a unit branch carrying ANY open second-round BLOCK
(`adversarial-audit`/`fix-verifier`/`acceptance-verifier`), the dispatch is DENIED unless the
anticipation artifact exists at the branch's CURRENT TIP, covers every derived key the union
of every such open BLOCK's own `finding_locations`/`class_enumeration` names, and every
command the hook itself re-executes hash-matches, in a worktree whose `HEAD` equals that tip
with `git status --porcelain` empty. `residual_risk` is the one field where the lead admits
an unclosed site — the defect this proposal answers was that the next round RAN the
deferred attack, not that a deferral was written down.

**Reader 2 — the closing-verifier relay (post-fix).** The relay gains `attacks_post`, the
SAME keys re-run at `fix_head`, widened by every file the fix window changed or added a new
surface to. `outcome` is DERIVED from a measured hash inequality for every fix-changed file a
pre-fix key covers — never declared; a pre-fix reproduction is the REQUIRED state and never
itself denies.

**Reader 3 — CI (`ci/scripts/check_rigor_record.py`), the half the lead cannot forge.** The
exported record carries the anticipation artifacts (`lead-gate-lib.py --export-anticipation
<slug>`); CI independently re-derives the required key set from the committed record and
checks the record's own SHAPE, with re-execution against a real checkout as a second,
advisory layer (fix round 1 narrows what is a hard fail here — see below).

**Execution mechanics.** Commands run in the worktree `git worktree list --porcelain`
resolves for the unit branch — a COMPUTED cwd, never `$CLAUDE_PROJECT_DIR` (a different
checkout on a stale branch that may not contain the fix's own files). The denylist admits a
suite's own entry point (`sh|bash <existing repo-relative path>`); everything else in R11's
denylist stays denied. Attack execution has its OWN budget (`_ATTACK_BUDGET_S`), separate
from the git-metadata budget. The witness is `sha256(rc + "\n" + stdout + "\n" + first line
of stderr)`; `rc ∈ {126, 127}` or empty captured output DENIES as VACUOUS; the SAME
`(command, hash)` pair reused across two keys DENIES as templated.

## Honest limits (stated as R3/R11 state theirs)

The hook cannot judge that an attack is a GOOD attack. What the pre-fix tip check and the
pre/post differential add is that the attack ran, at that instant, against a worktree whose
`HEAD` really was the tip with nothing uncommitted, and that its output changed after the
fix — this does NOT establish that no fix exists off-tree (a stash, another branch or
worktree), nor that the attack necessarily preceded any later fix commit. The human at merge
remains the control for attack quality, reading the exported attack list. The hook trusts the
lead's own `unit:` line — a decoy branch name is not detected.

## Fix round 1 (CONTRACT-R12-fix1.md, M1'–M6') — what changed from the design above

A design-round pressure-tester audited the mechanism described above BEFORE an implementer
ever wrote it, and found a mechanism-level defect the design's own prose had not surfaced:
keying the pre-fix artifact by a SINGLE BLOCK's own `ts`/`head_sha` and requiring the
branch's tip to equal exactly that sha is a PERMANENT DENY the moment two open second-round
BLOCKs of different types (e.g. an older `adversarial-audit` and a newer
`acceptance-verifier`, provoked after a fix commit landed in between) sit at different shas —
the tip cannot equal two different values at once. Fix round 1 amends the design in place,
before any implementer built the permanently-broken shape:

- **M1' — the artifact binds the CURRENT TIP; keys are the UNION over every open block.** One
  artifact per `(unit, tip)`, not per `(unit, block_ts)`. `block_sha` is NOT required to equal
  any one BLOCK row's own `head_sha` — only that the artifact's `pre_fix_sha` equals the
  branch's current tip, that tip equals the resolved worktree's own `HEAD`, and that the tree
  is clean. Keys are reduced PER FILE (`_key_to_file`) — this unit's many raw `path:line` keys
  cover a handful of files, and the artifact records an attack per file, not per line.
- **M2' — no key set is empty, vacuous, or unreachable.** An empty derived set (an
  `uncertain` BLOCK with no findings) requires ≥2 lead-chosen keys, each naming a file the
  unit's own `merge-base(main, tip)..tip` diff actually touches, distinct commands, distinct
  hashes, at least one execution-class (a real script, `cargo`, `python3`, `pytest`, `make` —
  never only inspectors: `sed`/`grep`/`cat`/`head`/`awk`/`rg`/`wc`/`tail`/`ls`).
- **M3' — reader 3 is a committed-record SHAPE check; re-execution is advisory.** CI HARD
  FAILS only on shape (the record exists, every union key is covered, every command passes
  the hook's own denylist); re-execution against a real `pre_fix_sha` checkout is reported
  per row as reproduced/mismatched/not_re_executed and NEVER fails the check (measured:
  BSD/GNU userland and path divergence between the lead's machine and CI's changes a witness
  hash without the mechanism under attack having changed). A shrink-only, human-maintained
  grandfather list (mirroring the existing `rigor_record_allowlist.txt` ratchet) exempts
  in-flight units whose BLOCK predates this mechanism landing.
- **M4' — reader 2 widens by FILE.** Every changed file in `block_sha..fix_head` widens the
  required set by file; this implementation does not further refine to the per-rewritten-
  definition granularity M4' also describes for `.rs`/`.py` files — a documented, honest
  simplification, never a weaker per-file guarantee (every changed file still needs its own
  covering key).
- **M5' — the deny-coverage sweep is committed.** `# R12-BEGIN`/`# R12-END` sentinels bound
  the four core `str | None`-returning mechanism helpers in `lead-gate-lib.py`;
  `ci/scripts/check_lead_gate.py --r12-sweep` (its own `swarm.yml` step, separate from
  `--self-test` — mutating and re-running ~24 R12 fixtures per deny arm is too slow to fold
  into the per-invocation self-test) enumerates every deny-`Return` inside that region by
  AST, neuters each arm's nearest enclosing `if` test to `False` ALONE, and fails when a
  neutered arm kills no R12 fixture. Scope note: the dispatch-routing arms M6' added
  (`_decide_implementer_dispatch`) sit outside the sentinel region and are not swept by this
  mechanism — they are independently covered by named, RED-verified fixtures instead
  (R12P1/R12P1b/R12P1c/R12P2/R12P2b).
- **M6' — close the fail-open enumerations.** `general-purpose`/`claude`/`fork`/
  `doc-updater` dispatches are now gated identically to the nine domain-implementer types the
  moment their prompt names a unit with an open second-round BLOCK (previously unconditionally
  allowed); an `IMPLEMENTER_TYPES` dispatch with NO `unit:` line now DENIES, naming the
  required line (previously allowed); a `unit:` line naming a branch that does not resolve
  under `refs/heads/` DENIES (previously silently treated as "no unit named").

## Residuals carried forward from fix round 1 (stated, not claimed closed)

The decoy-unit-line risk (the hook trusts the lead's own `unit:` line); the in-function-
rewrite residual (M4'); the `rm .jammi/gate-state/<slug>.*` escape hatch, which removes the
antecedent for every check above it and is never named as a remedy in any deny message;
`check_rigor_record.py`'s own re-execution being advisory, not a hard guarantee; the
`_pre_fix_anticipation_rejection`'s internal tip-moved re-check (a defense against a race
WITHIN one decision) has no practical, non-mocked fixture — stated as `uncovered` rather than
asserted tested.

See `.claude/agents/lead.md`'s "Anticipate before the fix, not after" paragraph and
`.claude/hooks/README.md`'s "(2b)" paragraph for the operator-facing statement of the same
mechanism, and `.jammi/escapes.jsonl`'s `esc-lead-gate-R12-anticipate-before-the-fix` row for
the ledgered symptom/control pair this proposal answers.
