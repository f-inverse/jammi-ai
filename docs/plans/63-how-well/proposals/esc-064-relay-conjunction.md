# PROPOSAL (human-merge): tighten(lead-gate) — the relay's requirements are a conjunction, not a two-arm disjunction (esc-064)

Status: PROPOSED v2 — retrospective-drafted 2026-08-29, then PRESSURE-TESTED the same
day (verdict REFINE: design upheld, change set incomplete — the v2 amendments at the
bottom of this file are MANDATORY parts of the change, found by execution). Gate-
definition change: `.claude/hooks/**` is `permissions.deny` for agents and
`SWARM_GATE_TOUCHED` means admin-merge — agents propose, the OPERATOR applies and
merges. Nothing here has been applied.

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

---

# v2 amendments — pressure-test REFINE (2026-08-29, executed against real hooks in scratch roots)

The pressure-test upheld the core design (arm-from-data, R2-always, predicate-not-
caller, reason-returning refactor; every error path traced exits 2 through the
fail-closed boundary; in-flight sites-only relays are recoverable IN PLACE by adding
a probe array — no re-audit, no state reset). The following are MANDATORY additions,
each found by execution, most blocking:

## B1 (BLOCKING) — `.claude/agents/lead.md:30-36` + `:40` join the change set, SAME commit
The lead's card carries the pre-change relay template (no `probe` key) and says probe
is written "instead" only when class_enumeration is absent. After the change, a lead
following its own card produces a DENIED relay on EVERY enumerated BLOCK — and the
realistic failure is not the junk probe but the honest lead facing an uninterpretable
deny and reaching for `rm .jammi/gate-state/<slug>.*`, which deletes the BLOCK row
outright (that is how R2-always becomes WEAKER than the status quo). Update the
template (add `probe`), the prose (R2 is always required; the degenerate-case
instruction: ">=2 sites you EXAMINED and found clean — not 2 more bugs"), and the
residual list at `:40` (add residual (4), matching README's). `.claude/agents/**` is
not deny-listed, but this edit must ride the SAME operator commit as the hook change
— splitting the behavior change by file leaves the workspace inconsistent between
merges. Also add the "examined and found clean" phrasing to the R2 rejection string.

## B2 (BLOCKING) — `docs/plans/53-agentic-swarm/ARCHITECTURE.md:236-237` joins the change set
It defines an accepted relay as sites-superset-only; stale on merge, and doc-parity
runs against exactly this text.

## B3 (BLOCKING) — G18 gains roots 6 and 7 (the specified matrix was NOT mutation-adequate)
Executed 5-mutant x 14-fixture matrix: a mutant that dedups by EXACT set while
testing collision by `.strip()` survives ALL of G16-G19 — killed only by
`probe=[" x.py:1", "x.py:1 "]` (strip-identical pair → 1 adjacent site): G18 root 6.
A mutant that strips only the PROBE side (exact on the reactive side) survives
everything including root 6 — killed only by a verifier-emitted PADDED finding
location (`finding_locations: ["foo.py:10 "]` is reachable; `parse_verdict_fields`
records locations verbatim) with a probe restating it unpadded: G18 root 7.

## B4 (BLOCKING) — G19 buildability caveat, stated in the fixture docstring
`handle_stop` ALWAYS writes `enumeration_missing` (lead-gate-lib.py:822), so
`_write_block_row` cannot produce the key-absent row — G19 must post-edit the state
JSONL to delete the key, and its docstring must say so. G19 is the only fixture that
kills the bolt-on mutant; silently weakening it to a hook-emittable row evaporates
the mutation-adequacy claim.

## A1 — honesty corrections to this proposal's own claims
- Mutant B is a DOCTRINE fix against legacy/hand-edited rows, not a live escape: the
  hook's own writer makes `enumeration_missing == (not class_enum)` (`:306`) and
  `:513` already ORs the data — B required hand-deleting the key.
- The `:420` "second caller" corollary is decorative: `any_unit_has_open_block:463`
  has no caller and `:622` filters same-type, so the `:420` call has no observable
  effect in any reachable state (matches the existing G13 note). Fixing the
  predicate is still right; it does not close a second live surface.
- Latency: measured NO lib delta (suite 2.75s→2.74s; N7 0.027s→0.028s); the
  "+0.3-0.5s" is fixture subprocess cost and lands nearer +0.8s. Immaterial; framed
  honestly now.
- R2-always applies to fix-verifier and acceptance-verifier relays too (reproduced
  both ways); the README and merge commit must state the full blast radius.

## A2 — the zero-width residual: choose one, explicitly
`.strip()` does not remove U+200B/U+FEFF: `probe=["foo.py:10​", "foo.py:10​​"]`
against finding `foo.py:10` is ACCEPTED under the v1 predicate — the mutant-C CLASS
survives its instance fix. Either (recommended) drop Unicode Cf/Cc-category
characters before stripping (one `unicodedata` line, monotone-toward-deny), adding a
G18 root for it, or list invisible-character collision in BOTH residual lists. The
proposal may not claim "closes all four by construction" without one of the two.

## A3 — document the normalization asymmetry (it is defensible, but only if stated)
Probe-side strip-both-sides is provably monotone-toward-DENY (it can only shrink the
adjacent set), which is why it is NOT the normalization the sites doctrine bans —
that ban exists because normalizing `sites` would make acceptance EASIER. Ruled
against pure exact-string for the probe side (it re-allows `"foo.py:10 "` outright).
State this in the README's relay paragraph or the next audit reads it as
normalization creep.

## A4 — ledger class_id reconciliation
esc-064's row carries `gate-vacuous-by-construction` (with esc-063) while
esc-053/054 carry `guard-state-collapse`; the retrospective asserted a third name
spanning both. The 053/054 kinship (an arm keyed on a proxy rather than the data) is
the stronger and correct framing — reconcile the `class_id` or record the
cross-cluster relation in the row when promoting esc-064.
