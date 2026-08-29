# PROPOSAL (human-merge): tighten(lead-gate) — the relay's requirements are a conjunction, not a two-arm disjunction (esc-064)

Status: PROPOSED — retrospective-drafted 2026-08-29, unit-63 session. Gate-definition
change: `.claude/hooks/**` is `permissions.deny` for agents and `SWARM_GATE_TOUCHED`
means admin-merge — the retrospective proposes, the OPERATOR applies and merges.
Nothing here has been applied.

## The escape and its three unlogged siblings (all live-reproduced against the real hooks)

`_relay_accepted` (`.claude/hooks/lead-gate-lib.py:490-532`) is a two-arm disjunction
where each arm enforces only half the design:

```
A reactive-only relay, enum present                       -> exit 0  (ALLOWED = esc-064)
B enumeration_missing key absent + class_enum non-empty,
  relay carries NO sites at all                           -> exit 0  (sites requirement dropped)
C probe entry = finding location + one trailing space     -> exit 0  (whitespace defeats disjointness)
D probe = ["bar.py:5","bar.py:5"]  (count 2, 1 site)      -> exit 0  (count inflation accepted)
```

Class: **disjunctive-arm requirement dilution** (peers: esc-053, esc-054, esc-063).
The arm is selected at `:513` by `enumeration_missing` — a flag with a weak-arm
default — not by the data. `_relay_accepted` has TWO callers (`:631` the gate,
`:420` the verifier-PASS audit-clearing path), so the dilution also lets a
fix-verifier PASS clear an audit BLOCK on a reactive-only relay: fix the
PREDICATE, not a caller.

## Principle

Every relay requirement is armed **unconditionally**. A branch condition may only
*add* a requirement or supply its *content* — never remove one. Which arm applies is
derived from the **data** (is `class_enumeration` non-empty?), never from a
separately recorded flag.

- **R1 coverage** — *if* `class_enumeration` is non-empty: `sites` keys ⊇ it,
  values non-empty strings.
- **R2 proactivity** — *always*: `probe` is a list with ≥ 2 **distinct,
  whitespace-normalized** entries outside `class_enumeration ∪ findings[].location`.

## Changes

### (a) `.claude/hooks/lead-gate-lib.py`
Replace `_relay_accepted` with a reason-returning `_relay_rejection` + thin bool
wrapper. Keep identity checks (`unit_branch`/`agent_type`/`block_ts`) and
fresh-read-per-call unchanged. Requirement block becomes:

```python
    class_enum = [s for s in (row.get("class_enumeration") or []) if isinstance(s, str)]
    finding_locs = [s for s in (row.get("finding_locations") or []) if isinstance(s, str)]

    # R1 COVERAGE — content supplied by the enumeration; never gates R2.
    if class_enum:
        sites = data.get("sites")
        if not isinstance(sites, dict):
            return "relay has no `sites` object, but the BLOCK enumerated a class"
        if not all(isinstance(v, str) and v.strip() for v in sites.values()):
            return "relay `sites` has an empty/non-string disposition value"
        missing = [s for s in class_enum if s not in sites]
        if missing:
            return f"relay `sites` omits {len(missing)} enumerated site(s), e.g. {missing[:3]}"

    # R2 PROACTIVITY — ARMED UNCONDITIONALLY (the esc-064 fix).
    probe = data.get("probe")
    if not isinstance(probe, list):
        return ("relay carries no `probe` array — on a BLOCK the lead must probe ADJACENT "
                "to the class and name >=2 sites outside it before re-dispatching")
    reactive = {s.strip() for s in class_enum} | {s.strip() for s in finding_locs}
    adjacent = {p.strip() for p in probe if isinstance(p, str) and p.strip()} - reactive
    if len(adjacent) < 2:
        return (f"relay `probe` names {len(adjacent)} distinct site(s) outside the verifier's own "
                "class_enumeration/findings (>=2 required) — restating the enumeration is reactive "
                "acknowledgment, not adjacent probing (esc-064)")
    return None
```

Also: `_decide_verifier_dispatch` (`:631-637`) carries the rejection reason into the
deny's stderr; `_adversarial_audit_cleared_by_verifier_pass` (`:420`) is deliberately
UNCHANGED (inherits via the predicate — say so in its docstring);
`parse_verdict_fields` keeps computing `enumeration_missing` but marked
*diagnostic only — no gate decision reads this field*; module docstring gains one
line stating the conjunction.

### (b) `ci/scripts/check_lead_gate.py`
Amend `fixture_g6_` and `fixture_g13_` to add `probe=["c.py:9","d.py:4"]` (they go
RED under the tightening otherwise); `fixture_g8_` unchanged (non-vacuity: no
enumeration → no `sites` demanded). New fixtures, docstrings citing **esc-064**:

- **G16** RED: enumerated BLOCK, relay restates enumeration as `sites`, NO `probe`
  → deny(2), stderr names the missing probe evidence.
- **G17** GREEN: same relay + ≥2 disjoint probe sites → allowed.
- **G18** boundaries (5 roots, each isolated, all deny): probe colliding with an
  ENUMERATED-but-not-found site; count 1; empty/whitespace entries; duplicates
  (mutant D); whitespace-padded collision `"a.py:1 "` (mutant C).
- **G19** the coverage arm is selected by the DATA: row with non-empty
  class_enumeration but NO `enumeration_missing` key, relay has `probe` but no
  `sites` → deny. This is the mutation-adequacy fixture: a fix that merely bolts
  `and probe_ok` onto the old enum arm leaves G19 red.

### (c) `.claude/hooks/README.md` (relay paragraph, `:50-59`)
State the conjunction (R1 by data, R2 always) AND the honest limit: the hook
enforces that adjacent probing is **asserted with named, citation-checkable sites**
— never that it occurred, nor that the sites are semantically adjacent. It converts
silent omission into an explicit, after-the-fact-checkable claim (citation-checker /
retrospective judge the probes, not this hook). Add residual (4) to the
visible-only list: boilerplate probe sites never examined — the tell is a probe
site the next round's citation-checker cannot corroborate.

### (d) Latency vs N7
Measured pre-change: self-test 41/41 in 2.85s, N7 max 0.027s vs 1.0s budget.
Per-invocation delta: sub-millisecond (two set comprehensions, no new I/O).
Self-test wall clock +≈0.3-0.5s. No ci.yml change.

### (e) Migration
Any on-disk relay with `sites` and no `probe` stops being accepted at merge — an
in-flight unit's next re-dispatch denies until the lead adds a `probe` array (or
uses the documented `rm .jammi/gate-state/<slug>.*` reset). Fail-closed in the
intended direction; call it out in the merge commit.

## Anti-Goodhart (stated, not hidden)
Genuinely open at the hook: boilerplate probes never examined (partly catchable
downstream only if probe sites surface in transcript/artifacts — cheap follow-up,
deliberately unbundled: require the re-dispatch prompt to echo the probe sites);
semantically irrelevant probes (`LICENSE:1`); line-splitting one site into two.
Rejected mechanism: denying repeated probe sets across rounds (false-denies
legitimate persistence; retrospective hygiene covers it). Honest cost: raises
pressure on the pre-existing DODGE-5 unlabeled-re-dispatch residual (unchanged;
tell unchanged).

## Ledger lifecycle
esc-064 stays `open` until G16-G19 land citing the id (→ `eval_added`), `closed`
when the self-test is green on main with the lib change merged. Named-but-not-
proposed (own row later): `sweep_method`/`exhaustive` are parsed and recorded but
armed on no branch — the degenerate case of this class.
