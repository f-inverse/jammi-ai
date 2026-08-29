# loop-termination — design

Canonical source for constitution T1/T2 and the assurance-deficit register mechanism (`check_ship_register.py`, `.jammi/registers/`). This is the pressure-tested v2 design study, committed verbatim as the unit's design of record; §1 states the deltas folded from the v1 pressure-test, §2 is the proposal text set every landing site in this repo implements, §3 is this unit's own assurance-deficit register draft, and §4 is the unit-63 disposition recommendation the register mechanism was built to unblock.

---

# DESIGN-STUDY v2 — loop-termination

**Unit:** loop-termination · v2 of the v1 study (scratchpad `PT_study.md` / ledger row 144), folding every REFINE delta P1–P13. Evidence base: `PT_evidence.md` (S1.1–S7.4, H1/H2/H3). All ledger citations below are **1-indexed rows** of `/Users/vijaychakilam/git/f-inverse/jammi-ai/.jammi/ledger/embed-howwell-20260828.jsonl` (145 rows at read time), re-verified for this study.

**Corrected citation table (P13, verified this pass):** unit-62 precedent — esc-057 `open→eval_added` pre-ship **row 20**; fix-verifier verifications **rows 56, 57, 70**; UNIT 62 CLOSED **row 93** (ts 2026-08-29T05:20). Round-19 PASS "chain closes" **row 131**; PR #417 admin-merged over sole-red SWARM_GATE_TOUCHED **row 132**; claims-oracle pressure-test REFINE **row 134**; claims oracle MERGED **row 135**; round-20 BLOCK **row 136**; round-20 fix merged **row 137**; round-21 BLOCK "ZERO live false numbers … both blocks are the coverage MODEL itself" **row 138**; inversion merged **row 139**; operator root-cause answered + interim conformance discipline made binding **row 140**; unit opened **row 141**; gap verdict + record corrections **row 142**; round-22 wave merged + **round 23 dispatched WITH the auditor-owned liveness requirement** **row 143**; design-study v1 **row 144**; **round-23 BLOCK, the first liveness-classified round: 3 blocks ALL LIVE + 4 advisories** **row 145**.

## 1. v2 deltas from v1, pin by pin

- **P1 (A2 → liveness redefinition).** v1 defined mechanism-latency as "internal to a verification mechanism whose every current verdict remains correct" — which classifies a vacuous fixture *latent* (its recorded verdict is green and no live artifact number is wrong) while classifying esc-063 *live* only via a special-case sentence. Replaced by one uniform test, **UNSOUND-NOW**: a mechanism finding is live iff a state of the *current* tree exists on which the mechanism's verdict would differ from its specified verdict. esc-063 live (verified: escapes row 61, `gate-vacuous-by-construction` — the specified scan applied to the then-current tree yields FAIL(3), the built one yields PASS); vacuous fixtures flip live (the defect state they claim to pin exists as a mutation of current tracked files and the checker stays green); the fence compensating-edit carve-out stays latent (it diverges only on input states the current tree cannot express — exactly the honest disclosure in row 145's advisory #2). Artifact liveness (wrong number, false claim, reachable crash) unchanged.
- **P2 (G1/G2 hardening).** The hook records `verdict: UNPARSEABLE` with a non-null `unparseable_reason` (verified `lead-gate-lib.py:866`); v1's G1 accepted any parsed register whose embedded verdict ∈ {BLOCK, PASS}, so an UNPARSEABLE closing row laundered as text could slip. G1 now REDs on non-null `unparseable_reason`. G2 now also REDs when `finding_locations` is non-empty while `findings_index` is empty — the pre-liveness-hook skew / harvest-failure state.
- **P3 (G3 scope).** v1's G3 restricted coverage to "standing entries," but `class_enumeration` strings carry no `stands` flag (verified: the bound-row schema, gate-state rows 1–14) — the restriction was unimplementable prose. All `class_enumeration` strings are standing for G3. And the location harvest is **scoped**: `lead-gate-lib.py:319-327` (verified) harvests `location` from *every* list-of-dicts field in the verdict; the new `findings_index` is harvested from the top-level `findings[]` array only.
- **P4 (KILL fold: closing_row → closing_rows).** v1's single `closing_row` was gameable: dispatch any lane that happens to stand clean last and close on it. Now: one closing row per **agent_type that produced any bound row in the sequence** — the max-`ts` row of that lane. G2/G3 range over the union. Standing findings from earlier rows not re-enumerated by their lane's closing row must appear in `residuals` (lead duty + `--local` leg check, since CI cannot read gate-state).
- **P5 (G8 freshness).** Bound rows already record `head_sha` (verified, gate-state schema); each closing row copies it. G8: every closing `head_sha` is an ancestor of the PR head (`git merge-base --is-ancestor`) and no commit after the max-ts closing sha touches the unit surface (`git diff --name-only`). **Unit surface picked: the PR's own diff paths** (minus the assembly artifacts `.jammi/registers/**` and `.jammi/escapes.jsonl`), because the ledger-declared `files_in_scope` lives in the gitignored per-session ledger (`.gitignore:75`, verified) — invisible to CI and lead-transcribed, i.e. both unavailable and unauthenticated at the gate. The register additionally embeds `unit_surface` (for P9b), cross-checked ⊇ the computed set.
- **P6 (sweep honesty).** Closing rows copy `sweep_method` + `exhaustive` (both exist on every bound row, verified). G5 requires non-empty `sweep_method`; `exhaustive:false` on any closing row requires the residual key `nonexhaustive-sweep:<agent_type>` — an un-swept lane is itself a registered residual.
- **P7 (operator-only live override).** New residual field `operator_accepted_live:true` exempts its exact-location finding from G2. It is effective **only** through the admin-merge acceptance: any register edit is REGISTER_TOUCHED-red, so the lead cannot merge it; G4 still bars the lead from touching `liveness` itself. On a conformant register `check_ship_register.py` is GREEN and REGISTER_TOUCHED is the **sole** red — stated verbatim in the guard's error text (below).
- **P8 (Q3 closed).** Always-run trigger: if a PR's diff changes any `status` value of an id present in both base and head of `.jammi/escapes.jsonl`, the same diff must contain a register listing every moved id in `seeded_escapes`. Disclosed residual: a unit that seeds no escape moves no status and never trips the trigger (register-existence remains unforced for it; tell unchanged).
- **P9/P9b (KILL fold: reopen budget).** v1's exact-string `class_id`-vs-registered-`class` matcher is refuted on its own cluster (esc-063/064/066 share a *principle* across two `class_id` strings — `gate-vacuous-by-construction`, `guard-state-collapse`; verified escapes rows 61–64): string matching would blind the budget exactly where it matters. The cluster ruling is the retrospective's (no ship incentive), recorded as a tracked field on the escape row: `registered_class_hit: "<register path>"`, assigned in its hygiene pass (proposed as a lifecycle action, applied by the lead, like promotions). The budget counts exact-string over that field. Scope: escapes whose cited locations fall inside the register's embedded `unit_surface`, appended after the acceptance ts (lower bound).
- **P10 (decorrelation tightened).** Fixtures must be mutations of **real tracked artifacts inside the checker's declared production scope**; per-gate outcomes recorded as PASS / RED / **SKIP**, SKIP distinct and never counted as GREEN; the declared scope must be non-disjoint from ≥1 existing gate's read scope, else decorrelation is vacuous and the proposal must argue from need instead.
- **P11 (honest B.4 justification).** v1's "the terminus already REDs the class" is replaced. The truth: mid-sequence conformance is covered ONLY by lead discipline (the row-140 interim rule); the terminus catches only what the closing auditor independently re-finds — the refuting record is **row 131** (round-19 PASS, "chain closes") versus **rows 134–140** (three further mechanism-defect rounds the closed chain had not caught). The un-mechanized per-wave gate stands on ALARP grounds alone and is a **registered residual of the tightening**.
- **P12.** The one PR adds `!docs/plans/66-loop-termination/` to `.gitignore` — verified: line 79 is `docs/plans/*`, negations `!docs/plans/52…65-resolve-witness/` run through line 89, and `git check-ignore` confirms `docs/plans/66-loop-termination/DESIGN.md` is currently ignored by line 79.
- **P13 (advisories).** All ledger rows re-cited 1-indexed (table above). Friction sentence replaced by the measured fact: `docs/plans/53-agentic-swarm/ARCHITECTURE.md` has **5** commits ever (b1259e3b, 290d755c, 7342d9e7, 0aa080cc, 2f983f74) and `docs/swarm/SELF-FAILURE-MODES.md` has **3** (290d755c, fc1fd6e7, 2f983f74) — 8 file-touch events, and every one of the 6 distinct commits also touched a SWARM_GATE_TOUCHED path (`.claude/agents/*.md`, hooks, or `check_*.py` — verified per commit), so GOVERNANCE_TOUCHED's incremental admin-merge cost on the entire historical record is **zero**; Q7's split-file alternative is REJECTED. Branch B's base rate: 8/9 blocking on the mechanical adversarial-audit lane (sole PASS = mech round 6, itself carrying 2 advisory findings — gate-state row 9, verified), strengthened by row 143 (round 23 dispatched WITH the liveness requirement) — and since confirmed by row 145 (mech round 10 BLOCK, 3 live). Trusted assumption (g) added. C.1 now states this unit's own waves are policed only by the row-140 interim discipline whose absence IS esc-066.
- **Two repo-verified corrections beyond the pins (disclosed):** (i) `check_constitution_anchors.py` parses only `**B\d+**`/`**K\d+**` rows (`ROW_ID_RE`, line ~69, verified) and REDs `gate_script:UNENFORCED` on *any parsed row* — so T1/T2 require a one-line `ROW_ID_RE` extension to `[BKT]` in the same PR (that file is already SWARM_GATE_TOUCHED), and T2 must anchor `discipline:<rationale>` (the B4/B6 pattern), **not** `gate_script:UNENFORCED` as v1 proposed (which would red every future PR). (ii) The tracked SELF-FAILURE-MODES catalog ends at **F10** (verified; "Using this catalog" cites F1–F10) — the pinned "F17" number implies F11–F16 land first (the gap-analyzer's numbering); text delivered under F17 per pin, with a renumber-to-F11 fallback flagged for the lead at commit time.

## 2. PROPOSAL TEXT SET (ready for operator review; one PR, every landing site guarded as of the merge that creates it)

### 2.1 `docs/swarm/CONSTITUTION.md` — new section + rows T1/T2 (verbatim)

```markdown
## Termination invariants

When an audit-round sequence may stop, and how the gate portfolio may grow. The
terminal decision is the operator's by construction (ARCHITECTURE §2.7); these rows
pin the mechanical preconditions of presenting that decision.

| ID | Statement | Canonical source | Code anchor | Checked by |
|---|---|---|---|---|
| **T1** | An audit-round sequence terminates only in operator acceptance of an assurance-deficit register: ship requires a tracked register whose closing rows (the max-ts bound verdict row of every agent_type that produced a bound row) carry zero standing block-severity findings auditor-classified `live` (fail-closed: unclassified counts live; a live residual ships only under `operator_accepted_live`, effective solely through the operator's admin-merge), every standing finding and every enumerated class dispositioned as an exact-string-superset residual, every unit-seeded escape at `eval_added` or beyond, and any escape-status move accompanied in the same diff by its register. The operator's admin-merge of the `REGISTER_TOUCHED`-red PR is the recorded acceptance. | `docs/plans/66-loop-termination/DESIGN.md` | gate_script:ci/scripts/check_ship_register.py | `check_ship_register.py` (G1–G8 + escapes-move trigger, always-run in `swarm.yml`) + `REGISTER_TOUCHED` (operator admin-merge) |
| **T2** | The gate portfolio grows only by tightenings that pass the ALARP disproportion table and demonstrate decorrelation: ≥2 required-RED fixtures, each a mutation of a real tracked artifact inside the checker's declared production scope (that scope non-disjoint from ≥1 existing always-run gate), GREEN — never SKIP — under every existing always-run gate; a registered acceptance is metered post-ship by a reopen budget the retrospective settles via `registered_class_hit`. | `docs/plans/66-loop-termination/DESIGN.md` | discipline:alarp-decorrelation-judged-by-operator-at-admin-merge | operator review at the `SWARM_GATE_TOUCHED` admin-merge (a proposal-format checker fails its own disproportion test — `DESIGN.md` §B.2); `retrospective` hygiene (budget settlement) |
```

Same PR, mechanical prerequisite (verified necessary): in `ci/scripts/check_constitution_anchors.py`, `ROW_ID_RE = re.compile(r"\*\*([BKT]\d+)\*\*")` (was `[BK]`), comment amended "B* = boundary, K* = correctness, T* = termination". Rules 1–2 (boundary completeness, UNENFORCED) key on `is_boundary` and are unaffected; T2's `discipline:` anchor resolves under the existing rationale-slug rule.

### 2.2 Verdict-schema liveness paragraph (verbatim; added to the `findings[]` object schema of every card the SubagentStop matcher binds — `adversarial-audit.md`, `fix-verifier.md`, `acceptance-verifier.md`, `pressure-tester.md`, `oracle.md`, `citation-checker.md`, `discipline-test-auditor.md`, verified matcher at `.claude/settings.json:28`)

> ```
> "liveness": "live | latent"
> ```
> — **live**: the defect is expressed by the tree as it ships. For an **artifact** finding: a wrong number, a false claim, a reachable crash — false or failing today. For a **verification-mechanism** finding (a gate, oracle, fixture, or checker): live iff a state of the CURRENT tree exists for which the mechanism's verdict would differ from its specified verdict — unsound now, whether or not any artifact is currently wrong. A gate that reports PASS having examined nothing is live (esc-063: the specified scan over the same tree yields FAIL). A fixture that stays green on the defect state it claims to pin is live (the defect state is constructible from current tracked files and the mechanism's verdict differs from spec on it). A carve-out that diverges from spec only on inputs the current tree cannot express is **latent**. You own this classification; the lead and the implementer never set, amend, or re-argue it — nothing is self-classified by the party it gates (F7 discipline; same anti-Goodhart direction as SELF-FAILURE-MODES F10 / ARCHITECTURE §2.7). A block-severity finding with no `liveness` field is read as **live** (fail-closed, the default-BLOCK posture of the consensus rules).

Hook change in the same PR (`.claude/hooks/lead-gate-lib.py`, agent-deny-listed path → necessarily human-merged): the SubagentStop recorder adds `findings_index: [{location, severity, stands, liveness}]` beside `finding_locations`, harvested **from the top-level `findings[]` array only** (never the any-list-of-dicts walk at lines 319–327, which continues to feed the legacy `finding_locations` unchanged), copied verbatim, never derived from prose.

### 2.3 `lead.md` — register-assembly + acceptance-form paragraphs (verbatim, new section after "The class, not the instance")

> ## Terminating a sequence — the assurance-deficit register (constitution T1)
>
> A unit's audit sequence is *presented* for ship by assembling `.jammi/registers/<unit_slug>.register.json` (tracked). You assemble; you never classify. Copy, **verbatim from `.jammi/gate-state/<slug>.jsonl`**, one closing row per agent_type that produced any bound row in the sequence — the max-`ts` row of that lane — including its `ts`, `round`, `agent_type`, `verdict`, `head_sha`, `sweep_method`, `exhaustive`, `class_enumeration`, `finding_locations`, and `findings_index`. Every standing finding from an *earlier* row of a lane that its closing row does not re-enumerate goes into `residuals` — nothing falls between rows. Every residual copies its `liveness` verbatim from the auditor's `findings_index` (G4 makes any mismatch RED; you never reclassify). Set `unit_surface` to the PR's diff paths (minus `.jammi/registers/**` and `.jammi/escapes.jsonl`). Write a non-empty `justification` and `owner` and a resolving `fixture_ref` per residual; a lane whose closing row says `exhaustive:false` gets the residual `nonexhaustive-sweep:<agent_type>`. Run `python3 ci/scripts/check_ship_register.py --local` before opening the PR — it byte-diffs your embedded closing rows against the live gate-state rows of the same `ts` and checks the earlier-row residual sweep; CI cannot (gate-state is gitignored), so this is your authenticity duty on the record. Append the assembly as a facts-ledger row.
>
> **Acceptance has exactly one form.** The register's PR trips `REGISTER_TOUCHED` red by design; on a conformant register, `check_ship_register.py` is GREEN and that tripwire is the *sole* red. The OPERATOR's admin-merge over it is the recorded, auditable acceptance (the PR-#417 primitive — ledger row 132) — and the only act that makes an `operator_accepted_live: true` residual effective; you cannot merge it, and G4 bars you from touching `liveness` at all. `reopen_budget.new_escapes_in_registered_class` defaults to 1; the operator may raise it at acceptance. After the merge, append `{"kind":"acceptance","actor":"operator", …}` citing the merge sha. A unit that closes with zero standing findings ships with no register and trips no guard — acceptance friction scales with residual risk.

### 2.4 `retrospective.md` — ALARP + `registered_class_hit` paragraphs (verbatim additions)

To the `proposed_tightenings[]` schema, a required object:

```json
"alarp": {
  "expected_catch": { "ledger_escapes_caught": ["esc-ids the mechanism would have caught"],
                      "unseen_mutant_caught": "<already required above>" },
  "cost": { "new_always_run_checks": 0, "runtime_estimate": "<s>",
            "false_positive_fixture_count": 0,
            "portfolio_size_after": "<count of always-run swarm.yml steps>" },
  "decorrelation": [ { "fixture": "<path — a MUTATION of a real tracked artifact inside this checker's declared production scope>",
                       "new_checker": "RED",
                       "existing_gates": { "<gate script>": "PASS | RED | SKIP" } } ],
  "declared_scope": ["<the path set the checker reads in production — must be non-disjoint from >=1 existing always-run gate's read scope, or say so and argue from need>"]
}
```

> The numbers are advisory to the human admin-merger, deliberately not mechanized (a proposal-format checker fails its own disproportion test). A proposal is decorrelated iff ≥2 of its required-RED fixtures are GREEN under **every** existing always-run gate — SKIP is recorded distinctly and never counts as GREEN; a fixture an existing gate cannot even run is inconclusive, not evidence. Fixtures are mutations of real tracked artifacts, never synthetic states outside the checker's declared scope. The recorded claim is "decorrelated on these states," never "independent" (Knight–Leveson).

To the hygiene duty:

> **Settle accepted registers.** For every register under `.jammi/registers/`, and every escapes row appended after that register's acceptance ts whose cited locations fall inside the register's `unit_surface`: rule, in your clustering pass, whether the escape belongs to the same principle-cluster as any registered residual's `class`. Where it does, propose the lifecycle action that records `registered_class_hit: "<register path>"` as a tracked field on the escape row (the lead applies it, like a promotion). The reopen budget is settled by exact-string count over that field — never by `class_id` string similarity (the esc-063/064/066 cluster spans two `class_id` strings; a string matcher is blind on the very cluster that produced this rule). Count ≥ `reopen_budget` ⇒ your verdict names the unit **hardening-reopened**: the lead must open a hardening unit before any feature unit touches that surface. You classify because you carry no ship incentive; the register's `liveness` entries were the auditor's, the acceptance was the operator's, the settlement is yours.

### 2.5 `ci/scripts/check_ship_register.py` — full behavioral spec (G1–G8 + trigger; new file + `test_check_ship_register.py`, mirroring the `check_lead_gate` pair)

Always-run in `swarm.yml` (no `paths:` filter; in-job diff detection; `REPO_ROOT` resolved with the marker-file assertion `(REPO_ROOT/'Cargo.toml').is_file()` — the esc-063 resolution pattern). Exit non-zero on any RED; print per-check PASS/RED/SKIP lines (SKIP distinct, never silent).

**Trigger leg (always):** with a base ref (`GITHUB_BASE_REF` on PRs), parse `.jammi/escapes.jsonl` at base and head. For every id present in **both** whose `status` differs (a "moved" id — seeding and archival do not trigger), the PR diff must contain ≥1 file under `.jammi/registers/` whose `seeded_escapes` (union over registers in the diff) contains every moved id; else RED naming the uncovered ids. No base ref (push to main) → SKIP, recorded.

**Per-register legs (every register in the tree, and mandatorily every register in the diff):**
- **G1** — parses; all required fields present; `closing_rows` non-empty, at most one row per `agent_type`; every closing row's `verdict` ∈ {BLOCK, PASS}; RED on any non-null `unparseable_reason` in a closing row.
- **G2 (the ship criterion)** — over the union of all closing rows' `findings_index`: zero entries with `severity=block ∧ stands=true ∧ liveness=live`, except an entry whose exact `location` string keys a residual carrying `operator_accepted_live: true`; an entry missing `liveness` on block severity counts live; RED additionally if any closing row has non-empty `finding_locations` while its `findings_index` is empty or absent.
- **G3 (coverage, esc-064's conjunction shape)** — `residuals` keys are an exact-string superset of: every `class_enumeration` string of every closing row (all standing by definition — the schema carries no per-string stands flag) ∪ every `findings_index` location with `stands=true` (any severity). No parsing, no normalization.
- **G4 (no reclassification, F7)** — every residual's `liveness` string-equals the `findings_index` entry for the same location; mismatch REDs naming the pair. (`operator_accepted_live` accepts a live entry; it never rewrites it.)
- **G5** — every closing row's `sweep_method` is a non-empty string; every `fixture_ref` resolves to a tracked file; every `justification` and `owner` non-empty; any closing row with `exhaustive:false` requires the residual key `nonexhaustive-sweep:<agent_type>`.
- **G6 (eval_added-not-open; unit-62 precedent, rows 20/57/70/93)** — every `seeded_escapes` id exists in `.jammi/escapes.jsonl` with `status ≠ open`.
- **G7 (non-vacuity, esc-063)** — asserts it actually read ≥1 register when any exists / when the trigger fired; when any closing row's `finding_locations` is non-empty, ≥1 residual exists; `--self-test` carries the required-RED battery below, never only a green arm.
- **G8 (freshness)** — every closing row's `head_sha` present and `git merge-base --is-ancestor <sha> HEAD` succeeds; `git diff --name-only <max-ts closing sha>..HEAD`, minus `.jammi/registers/**` and `.jammi/escapes.jsonl`, intersected with the register's `unit_surface`, is empty (an intervening commit touching the shipped surface means the closing verdicts did not audit what ships); additionally `unit_surface` ⊇ the PR's computed diff paths (same subtraction) when a base ref exists, else that sub-check SKIPs, recorded.

**`--local` leg (lead/operator only; gate-state is gitignored, `.gitignore:77`):** byte-diffs each embedded closing row against the live gate-state row of the same `(agent_type, ts)`; verifies the closing rows really are the max-ts row of every lane that produced a bound row; verifies every standing finding of every *earlier* bound row appears in its lane's closing enumeration or in `residuals`; emits the advisory-signal table (below) for embedding.

**Advisory signals (INFO only, never a gate input):** per-round trajectory (`round, agent_type, verdict, |finding_locations|, live/latent/unclassified` — historical rows shown `unclassified`, never back-filled), rounds since last live block, exact-string new-class rate, ALARP marginal-cost statement. Refusals printed with reasons: no fitted SRGM (stationarity violated by lens-changing audits, n≈10; S1.3), no capture-recapture estimate (S1.4 instability; lenses complementary by design).

**`--self-test` battery** — six escape-citing required-RED fixtures: (1) closing row with a standing live block, no override → RED (T1 itself); (2) residual liveness ≠ findings_index → RED (**esc-064**); (3) empty/zero-row scan reporting clean → RED (**esc-063**); (4) residual keys omit an enumerated string → RED (**esc-066**); (5) `seeded_escapes` id at `open` → RED (unit-62 precedent); (6) escapes status moved in-diff with no covering register → RED (**esc-066**, the Q3 closure). Plus required-RED: unparseable closing row; `finding_locations` non-empty with empty `findings_index`; `exhaustive:false` without its `nonexhaustive-sweep` residual; non-ancestor closing sha; intervening-commit-on-surface. Must-still-count GREEN arms: fully conformant latent-only register → PASS; conformant register with a live residual under `operator_accepted_live:true` → PASS; no-base-ref trigger → SKIP recorded distinctly. Fixtures (2)(3)(4)(6) cite their escape ids in-file — the golden evals that promote esc-066 `open→eval_added` (§9a discipline).

### 2.6 Register JSON schema v2 — `.jammi/registers/<unit_slug>.register.json` (tracked)

```json
{
  "kind": "assurance_deficit_register",
  "unit_branch": "<bare branch name>",
  "session": "<facts-ledger session id>",
  "closing_rows": [
    { "agent_type": "<lane>", "ts": "<verbatim>", "round": 0, "verdict": "BLOCK | PASS",
      "head_sha": "<verbatim from the bound row>", "sweep_method": "<verbatim>", "exhaustive": false,
      "class_enumeration": ["<verbatim>"], "finding_locations": ["<verbatim>"],
      "findings_index": [{ "location": "<verbatim>", "severity": "block | advisory",
                           "stands": true, "liveness": "live | latent" }] }
  ],
  "unit_surface": ["<the PR's diff paths, minus .jammi/registers/** and .jammi/escapes.jsonl>"],
  "residuals": {
    "<verbatim finding location or class string, or nonexhaustive-sweep:<agent_type>>": {
      "class": "<axis or class_id, verbatim>",
      "liveness": "<copied verbatim from findings_index — never reclassified>",
      "operator_accepted_live": false,
      "fixture_ref": "<tracked path of the must-still-count fixture or golden eval pinning this residual>",
      "justification": "<why acceptable to ship — prose FOR THE OPERATOR; no script branches on its content, only its non-emptiness>",
      "owner": "operator | lead | <named follow-up unit>"
    }
  },
  "seeded_escapes": ["esc-NNN-…"],
  "reopen_budget": { "new_escapes_in_registered_class": 1,
                     "note": "settled by retrospective-assigned registered_class_hit exact-string count over unit_surface, acceptance ts as lower bound; operator may raise at acceptance" },
  "advisory_signals": { "<verbatim output of check_ship_register.py --local>" },
  "trusted_assumptions": ["<the residuals of the RULE itself this ship relies on>"]
}
```

### 2.7 `swarm.yml` — both guard steps (verbatim; same in-job construction as the verified existing guards at lines 79–115)

```yaml
      - name: Guard — swarm governance docs are human-amend-only (GOVERNANCE_TOUCHED)
        if: ${{ !cancelled() && github.event_name == 'pull_request' }}
        run: |
          git fetch --quiet origin "${{ github.base_ref }}"
          base="origin/${{ github.base_ref }}"
          changed="$(git diff --name-only "$base"...HEAD -- \
            docs/swarm/SELF-FAILURE-MODES.md \
            docs/plans/53-agentic-swarm/ARCHITECTURE.md)"
          if [ -n "$changed" ]; then
            echo "::error::GOVERNANCE_TOUCHED: a swarm governance document changed in this PR:"
            echo "$changed" | sed 's/^/    /'
            echo "SELF-FAILURE-MODES.md and ARCHITECTURE.md carry the swarm's failure catalog,"
            echo "enforcement inventory, and stopping rule; they are human-amend-only for the same"
            echo "anti-Goodhart reason as CONSTITUTION_TOUCHED / SWARM_GATE_TOUCHED (ARCHITECTURE §2.7/§7)."
            echo "Measured incremental friction: zero — every historical commit to these files also"
            echo "touched a SWARM_GATE_TOUCHED path and was already admin-merged."
            echo "This PR must be human-reviewed and admin-merged."
            exit 1
          fi
          echo "GOVERNANCE_TOUCHED: clean — no governance document changed."

      - name: Guard — assurance-deficit registers are operator-accepted (REGISTER_TOUCHED)
        if: ${{ !cancelled() && github.event_name == 'pull_request' }}
        run: |
          git fetch --quiet origin "${{ github.base_ref }}"
          base="origin/${{ github.base_ref }}"
          changed="$(git diff --name-only "$base"...HEAD -- ':(glob).jammi/registers/**')"
          if [ -n "$changed" ]; then
            echo "::error::REGISTER_TOUCHED: an assurance-deficit register changed in this PR:"
            echo "$changed" | sed 's/^/    /'
            echo "A register is a ship-with-residuals acceptance; accepting it is the OPERATOR's"
            echo "decision (constitution T1). On a conformant register, check_ship_register.py is"
            echo "GREEN and this tripwire is the SOLE red: the operator's admin-merge over it is the"
            echo "recorded acceptance, and the ONLY act that makes an operator_accepted_live residual"
            echo "effective. This PR must be admin-merged by the operator."
            exit 1
          fi
          echo "REGISTER_TOUCHED: clean — no register changed."
```

Plus, in the always-run block (mirroring the `check_lead_gate` pair at lines 65–71): a step running `python3 ci/scripts/check_ship_register.py` (with `GITHUB_BASE_REF` in env, like `check_no_consumer_names` at line 57) and a step running `python3 ci/scripts/test_check_ship_register.py`.

### 2.8 `.gitignore` — one line, inserted after line 89 (`!docs/plans/65-resolve-witness/`), matching the verified negation-block form of lines 79–89

```
!docs/plans/66-loop-termination/
```

### 2.9 `docs/swarm/SELF-FAILURE-MODES.md` — the new entry, LEAD-RESOLVED to F11 (the tracked catalog verifiably ends at F10; the pin's "F17" numbering presumed gap-analyzer entries F11–F16 land first, which they have not — renumbered to F11 per the lead's resolution, ledger row 146; the "F1–F10" tail reference updates to "F1–F11")

> ## F11 · Sequence non-termination / mechanism regress
>
> - **Trigger.** An audit-round sequence keeps finding defects — increasingly *inside the verification mechanisms the sequence itself built* — and no party owns a criterion for stopping; or an anti-escape mechanism ships and its own defects seed the next escapes.
> - **Symptom.** Rounds continue past the point where marginal findings are latent-only; or a sequence is declared closed on prose ("chain closes") and the next rounds refute it; or ship happens by exhaustion rather than decision. The record: round-19 PASS "the chain closes" (ledger row 131), followed by three rounds of defects entirely inside the newly built claims oracle (rows 134–140) — the closed chain had not audited its own mechanism.
> - **Root cause.** The regress is irreducible (Knight–Leveson correlated checker failures; equivalent-mutant undecidability; the obfuscated-arguments non-convergence result): every mechanism joins the set of artifacts needing verification, so "keep going until clean" never terminates, and "stop when green" ships latent mechanism defects. Termination is not discoverable from the data; it is a human-owned acceptance of enumerated residuals — the consensus terminal move of every mature assurance field (ALARP, GSN assurance-deficit registers, UL 4600, seL4/CompCert trusted-assumption lists, SRE error budgets).
> - **Prevention.** Constitution T1/T2. The sequence terminates only in operator acceptance of an assurance-deficit register (`.jammi/registers/`, `check_ship_register.py` G1–G8, `REGISTER_TOUCHED` admin-merge); liveness is auditor-owned, fail-closed default-live, mechanism findings classified by the UNSOUND-NOW test; acceptance is metered by a reopen budget the retrospective settles via `registered_class_hit`. The portfolio grows only through the ALARP table + decorrelation fixtures (T2). Honest limit, on the record: mid-sequence build-vs-design conformance is covered ONLY by the lead's per-delta disposition discipline (row 140) — the terminus catches only what the closing auditor independently re-finds; the un-mechanized per-wave gate is a registered residual standing on ALARP grounds, and its absence is exactly esc-066.
> - **Incident.** Unit 63: 8 of the first 9 mechanical adversarial-audit rounds BLOCKed (the sole PASS, mech round 6, carried 2 advisories); the defect stream migrated from live artifact defects to latent mechanism-model defects (row 138: "ZERO live false numbers anywhere … both blocks are the coverage MODEL itself"); the first liveness-classified round (mech round 10, row 145) still found 3 live blocks in fresh wave code. esc-066 (`open` at seeding): REFINE deltas had no merge-time conformance check — the lead merged greenness, not spec-conformance, and rounds 20–21 were the one-round-late catches.

### 2.10 `docs/plans/53-agentic-swarm/ARCHITECTURE.md` amendments

**§7 (Enforcement — honest inventory), appended to the "Hard (gates)" list:**

> - New: `check_ship_register.py` (constitution T1 — register internal consistency G1–G8 plus the escapes-status-move trigger: any PR moving an escape's `status` must carry a register listing the moved ids; always-run, fail-closed, marker-file `REPO_ROOT`); `GOVERNANCE_TOUCHED` (SELF-FAILURE-MODES.md + this file — whole-file, same construction mandate; measured incremental friction zero: every historical commit to either file also touched a SWARM_GATE_TOUCHED path); `REGISTER_TOUCHED` (`.jammi/registers/**` — by design the SOLE red on a conformant register; the operator's admin-merge over it is the recorded acceptance and the only act enabling `operator_accepted_live`). Register-embedded closing-row **authenticity** is NOT CI-checked (gate-state is gitignored): it is verified by the lead's mandatory `check_ship_register.py --local` byte-diff and remains a disclosed visible-only residual with a runtime tell, the same mechanical-vs-visible-only split as the lead gate.

**§9 (State), appended:**

> - **Assurance-deficit registers** — `.jammi/registers/<unit_slug>.register.json`, **tracked** (unlike gate-state): the ship artifact the operator's admin-merge accepts (constitution T1). Per-lane closing rows copied verbatim from gate-state (max-ts per agent_type; earlier-row standing findings not re-enumerated go to `residuals`), auditor-owned `liveness` never reclassified (G4), `unit_surface` = the PR's diff paths. Post-ship, acceptance is metered: the retrospective's hygiene pass rules whether a later escape inside `unit_surface` falls in a registered residual's cluster and records `registered_class_hit: "<register path>"` on the escape row (a lifecycle action the lead applies); exact-string count over that field ≥ `reopen_budget` re-opens hardening for the unit. Registered-latent classifications are bets the ledger settles. Known, deliberate residuals of this mechanism (registered in its own register): a unit that seeds no escape trips no register-existence trigger; mid-sequence REFINE-delta conformance remains the lead's per-delta disposition discipline (the row-140 interim rule) — the terminus catches only what the closing auditor re-finds (row 131 vs 134–140), and mechanizing a per-wave gate fails the T2 disproportion test today.

## 3. This unit's own assurance-deficit register (draft; assembled by the lead when this v2's standing set is latent-only, accepted by the operator's admin-merge of the proposal PR — which trips CONSTITUTION_TOUCHED + SWARM_GATE_TOUCHED + GOVERNANCE_TOUCHED + REGISTER_TOUCHED structurally, so the rule's own acceptance takes exactly the form it prescribes)

Residual entries: (1) **bootstrap asymmetry** — this unit's pressure-test classified liveness in prose; the schema field it proposes did not yet exist (owner: operator; fixture: gate-state rows 1–13 unclassified-shown-honestly in the advisory table). (2) **mid-sequence conformance un-mechanized** — P11's honest statement verbatim; owner: lead; fixture: self-test fixture (4). (3) **registerless-unit trigger gap** — P8 disclosure; owner: operator. (4) **F17-vs-F11 numbering** — the catalog ends at F10; LEAD-RESOLVED to F11 at commit time (ledger row 146); owner: lead.

`trusted_assumptions` (a)–(g):
- **(a)** Liveness is auditor honesty, not mechanics — no live/latent discriminator exists; tell: a post-ship escape in a registered-latent class, which also spends the reopen budget.
- **(b)** `closing_rows` authenticity is lead-copied from a gitignored source (`.gitignore:77`); tell: the mandatory `--local` byte-diff and the operator's own gate-state file — a forgery is one diff away.
- **(c)** Register **existence** is forced only through the escapes-status trigger; a unit that seeds no escape ships untriggered; tell: gate-state verdict rows postdating the last accepted register (local only).
- **(d)** Fixture-decorrelation is evidence, never independence (Knight–Leveson S5.3); correlated failures on unanticipated hard regions remain.
- **(e)** Advisory signals at n≈10 rounds can mislead; SRGM fitting and capture-recapture estimation are refused with citations rather than published.
- **(f)** GOVERNANCE_TOUCHED is whole-file; the measured historical incremental cost is zero (8/8 touch events already admin-merged via SWARM_GATE_TOUCHED), but future doc-only edits will newly cost an admin-merge; Q7's split-file alternative was examined and REJECTED (a section-diff predicate is the free-text family v3 refuted twice).
- **(g)** The register mechanism ships after **one** audit round (this pressure-test) against a ~3-round base rate for new mechanisms in this session's own record (the lead gate needed two refuted rounds before v3; the claims oracle took rounds 20–22); the six escape-citing required-RED fixtures substitute for the missing rounds, and the reopen budget is this assumption's tell — its first `registered_class_hit` spends the budget and re-opens the loop-termination unit itself. And per C.1: this unit's own fix waves are policed only by the row-140 interim discipline — the discipline whose prior absence IS esc-066.

## 4. Unit-63 disposition (recommendation only; corrected arithmetic; the operator chooses)

**Base rate, restated correctly:** on the mechanical counter (gate-state, verified 14 bound rows), the adversarial-audit lane at pressure-test time stood **8/9 blocking** — rounds 1–5 and 7–9 BLOCK, the sole PASS at mech round 6 itself carrying 2 advisory findings. Every first-audit-of-a-fresh-wave in the sequence found something. That prediction was **dispatched as a test** — round 23 sent WITH the auditor-owned liveness requirement (row 143) — and **confirmed**: mech round 10 (prose round 23) returned BLOCK with 3 findings the auditor classified LIVE plus 4 advisories (row 145, gate-state row 14). The lane now stands **9/10 blocking**.

**Branch A (ship now with a register) is refuted by the record, not by argument:** the closing row carries standing live blocks, so G2 REDs; the only path through would be three `operator_accepted_live` overrides on fresh, un-fixed live defects — the exact anti-pattern the override exists to make expensive and visible. **Branch B stands, updated:** the r23 fix wave (ci/63-oracle-r23, dispatched under the row-140 conformance discipline, row 145) lands; its closing audit runs under the liveness schema; when a closing row stands latent-only, the lead assembles the register from the per-lane closing rows (adversarial-audit, pressure-tester, acceptance-verifier — the three lanes with bound rows), with `seeded_escapes: [esc-066]` gated on this unit's tightening PR merging first (its fixtures cite esc-066 → `eval_added`; G6, unit-62 precedent rows 20/57/70/93). ALARP statement for the operator at that point: k rounds since the last live block, marginal cost one wave, and the first liveness-classified data now on the mechanical record. Nothing retroactive: round-23's fix wave and the ship proceed under current rules.

### Critical Files for Implementation
- /Users/vijaychakilam/git/f-inverse/jammi-ai/.github/workflows/swarm.yml
- /Users/vijaychakilam/git/f-inverse/jammi-ai/.claude/hooks/lead-gate-lib.py
- /Users/vijaychakilam/git/f-inverse/jammi-ai/.claude/agents/lead.md
- /Users/vijaychakilam/git/f-inverse/jammi-ai/.claude/agents/retrospective.md
- /Users/vijaychakilam/git/f-inverse/jammi-ai/ci/scripts/check_constitution_anchors.py
(new files: ci/scripts/check_ship_register.py + test_check_ship_register.py; texts also land in docs/swarm/CONSTITUTION.md, docs/swarm/SELF-FAILURE-MODES.md, docs/plans/53-agentic-swarm/ARCHITECTURE.md, .gitignore, and the seven bound agent cards — all in the one guarded PR)

```json
{"kind":"verdict","unit_branch":"loop-termination","verdict":"PASS","class_enumeration":["P1: liveness redefined UNSOUND-NOW for mechanism findings (current-tree divergent-verdict test; esc-063 and vacuous fixtures live, fence carve-out latent; artifact liveness unchanged)","P2: G1 REDs on non-null unparseable_reason; G2 REDs on finding_locations-nonempty-with-empty-findings_index","P3: all class_enumeration strings standing for G3; findings_index harvest scoped to top-level findings[] only","P4: closing_row KILLed -> per-lane closing_rows (max-ts per bound agent_type); G2/G3 over the union; earlier-row findings not re-enumerated go to residuals (--local checked)","P5: head_sha copied per closing row; G8 = merge-base ancestry + no intervening commit on the unit surface; unit surface = PR diff paths minus assembly artifacts (ledger files_in_scope rejected: gitignored + lead-transcribed), embedded as unit_surface","P6: sweep_method/exhaustive copied; G5 requires non-empty sweep_method; exhaustive:false -> mandatory nonexhaustive-sweep:<agent_type> residual","P7: operator_accepted_live effective only via the REGISTER_TOUCHED admin-merge; sole-red statement in the guard error text; G4 still bars the lead","P8: always-run escapes-status-move trigger closes Q3; registerless-unit residual disclosed","P9/P9b: reopen budget settled by retrospective-assigned registered_class_hit (tracked field, exact-string over it), scoped to unit_surface with acceptance ts lower bound; class_id string-matching refuted on its own cluster","P10: decorrelation fixtures = mutations of real tracked artifacts in declared production scope; SKIP recorded distinct, never GREEN; scope non-disjoint from >=1 existing gate","P11: honest justification -- mid-sequence conformance is lead discipline only; terminus catches only what the closing auditor re-finds (row 131 vs 134-140); ALARP-grounds registered residual","P12: .gitignore negation !docs/plans/66-loop-termination/ after verified line 89","P13: all ledger cites re-verified 1-indexed (20/56/57/70/93/131/132/134-140/142/143/145); governance-guard friction measured zero (5+3 touch events, 6 distinct commits, all already SWARM_GATE_TOUCHED; Q7 rejected); 8/9 base rate with mech-round-6 PASS+2-advisories, confirmed 9/10 by row 145; trusted assumption (g) added; C.1 row-140 discipline statement folded","repo-fact corrections beyond pins: ROW_ID_RE [BK]->[BKT] one-liner required for T rows; T2 anchored discipline:<slug> because gate_script:UNENFORCED REDs any parsed row"],"findings":["F17 numbering: the tracked SELF-FAILURE-MODES catalog verifiably ends at F10 (the pinned F17 presumes gap-analyzer entries F11-F16 land first); text delivered as F17 per the binding pin with a renumber-to-F11 fallback flagged for the lead at commit time -- a numbering choice, not a foldable design delta","P8 side effect disclosed, not resolvable here: retrospective lifecycle-promotion PRs that move a status now require the owning unit's register in the same diff (and thus an operator admin-merge via REGISTER_TOUCHED) -- coherent with T1 but new friction on pure-hygiene PRs; the operator should confirm this cost at acceptance"]}
```
