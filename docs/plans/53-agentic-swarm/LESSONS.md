# Lessons → swarm mechanisms — the crystallized memory

Every hard-won lesson from Jammi's agentic history (`docs/plans/51-marathon-learnings/`,
`docs/plans/52-maintainer-roadmap/`, `CLAUDE.md`, `docs/guide/src/philosophy.md`) and the
cross-session memory (`~/.claude/projects/.../memory/`), organized as **generalizable
principle families** and mapped to the swarm mechanism that encodes each.

**Why families, not a bug list.** A flat catalogue of past bugs teaches a verifier to
recognize *those* bugs and miss the novel-but-analogous one. So each family carries:

1. **The principle** — stated at an altitude that transfers to code the agent has never seen.
2. **Instances** — one or two historical cases as *illustration and evidence only* (with
   source). The specific repros live in `.jammi/escapes.jsonl` as regression evidence
   (`closes_escape`); they are not the test.
3. **The mechanism** — the verifier rubric line / gate / constitution invariant that
   encodes *the principle*, phrased at the level of the class, never the instance.

**Mechanism type** ∈ {`verifier-check`, `oracle-hard-block`, `self-failure-mode`,
`escape-ledger-seed`, `constitution-invariant`, `CI-gate`, `hook`, `phase-step`,
`domain-agent-invariant`}.

The **per-verifier rubrics** at the end are what a build worker lifts onto an agent's
card — each is a principle-level instruction to *generalize and default-BLOCK on a
novel-but-analogous smell*, not a lookup of the past bugs.

Cross-refs: swarm design = `ARCHITECTURE.md` (phases §4, roles §5, gates §7, ledgers §9);
regression evidence = `.jammi/escapes.jsonl`.

---

## Correctness principle families

### A · Resource-guard state-collapse
**Principle.** A resource guard (RAII/`Drop`/`__exit__`/context manager) that collapses
distinct entry/exit states into one cleanup arm can *release something it never held*.
Audit every guard for state-collapse: enumerate the states the happy path never
constructs — never-entered, entered-while-unset, re-entry, reuse, error-path exit — and
require the cleanup to distinguish "held" from "never held."
**Instances.** A tenant-scope `__exit__` used `.take().flatten()`, collapsing "never
entered" (`None`) and "entered-while-unscoped" (`Some(None)`) into one `unbind` arm, so a
stray exit cleared a live scope — a data-scope leak on green CI (`CASE-STUDIES.md §3b`;
`crates/jammi-db/src/tenant_scope.rs`; esc-001).
**Mechanism.** `adversarial-audit` rubric: for any guard/cleanup, model the full state
lattice and BLOCK if one arm cannot tell "acquired" from "never acquired." → **verifier-check**

### B · Make-invalid-states-unrepresentable; reshape, don't work around
**Principle.** A type that conflates "absent" with a meaningful value, or that cannot
express a state some transition needs, is the *wrong shape* — reshape it (and change every
call site atomically), never bolt on a companion flag or special-case the write. Greenfield
posture: no backwards-compat shims, no band-aids; if a fix collides with a wrong
abstraction, reshape the abstraction.
**Instances.** A partial-`UPDATE` builder whose field was `Option<T>` (`None` = "leave
unchanged") could never emit `SET col = NULL`, so a transition that had to *clear* a nullable
column silently left the stale value (`partial-update-none-overload` memory). A verb with an
implicit "the embeddings" selector became ambiguous once a source had two embedding tables —
the fix added an explicit `table=` argument rather than guessing a default (`FIELD-NOTES.md
R3`).
**Mechanism.** `adversarial-audit` rubric: for any type modeling presence / selection /
partial-update, ask *"what state can this NOT express?"* and BLOCK if a required behavior
lands in an unrepresentable state or is rescued by a band-aid (`#[allow]`, `let _ =`,
`// TODO: later`, `#[ignore]`, a `clear_x: bool` beside an `Option`). Constitution boundary
invariant "no band-aids / greenfield." → **verifier-check + constitution-invariant**

### C · Standardize in the space the optimizer moves through
**Principle.** Any trainable head on a high-offset / low-variance / large-magnitude target
needs a *data-space* standardization + a domain contract — standardize the representation the
head conditions on, not the loss. Under Adam the parameter step is ~`lr` regardless of loss
scale, so loss-rescaling cannot move a raw parameter to a distant target.
**Instances.** A design to fix a target-collapse by rescaling the loss was killed at
pressure-test; the served mean stayed near the zero-init (~2163 off) until the target was
z-scored with a persisted de-standardization affine — and the same root cause had a *second
home* in a separate subsystem (`CASE-STUDIES.md §1`; `fine_tune/{regression_loss,target}.rs`;
esc-004).
**Mechanism.** Constitution correctness-invariant "standardize-in-data-space"; `ai-model`
domain agent requires a high-offset oracle per trainable head; `adversarial-audit` BLOCKs any
scale-problem "fix" that acts on the loss. → **constitution-invariant + oracle-hard-block + domain-agent-invariant**

### D · Domain-validity at every numeric AND catalog input edge
**Principle.** Compute nothing past a valid input domain. A function evaluated outside the
domain where its output means anything returns a *confident wrong number*, not an error.
Validate / clamp / normalize at every numeric and catalog edge, pin the mathematical object
(set vs multiset, directed vs undirected, `[0,1]` vs unbounded), and add a boundary/degenerate
oracle per operation. This is the single highest-leverage review lens the marathon produced —
one root cause wore four disguises.
**Instances.** A learning-rate schedule went *negative* past its horizon (undercounted
realized steps; `FIELD-NOTES.md O4`, esc-007); a regression mean blew out unstandardized
(family C); a degree was *doubled* by counting an undirected edge as a multiset member
(`FIELD-NOTES.md G2`, esc-008); a "tenant-isolated" source was globally readable because no
row predicate was enforced (`FIELD-NOTES.md S2`, esc-013).
**Mechanism.** Constitution correctness-invariant "domain-validity"; `adversarial-audit` +
`ai-pipeline`/`db` domain agents apply the validate-the-edge lens and BLOCK any operator with
no guard at its input edge. → **constitution-invariant + verifier-check + domain-agent-invariant**

### E · Bound the term that grows, not the aggregate
**Principle.** When you add a limit, first identify *which* quantity is unbounded (often
resident copies, not compute) and bound *that term* — never the aggregate that contains a
caller-controlled term, or you silently cap the caller's input.
**Instances.** An over-fetch cap `min(k + excluded + 1, MAX)` silently capped the requested
`k`; the bound belonged on the excluded headroom alone. The associated OOM was 3–4 resident
corpus copies, not quadratic compute — the index should be the single owner of its vectors
(`FIELD-NOTES.md O3`; `hard_negative_miner.rs`; esc-003).
**Mechanism.** `adversarial-audit` rubric: for any new cap/limit, name the unbounded term and
confirm the bound lands on it; BLOCK a bound placed on a sum containing a caller-controlled
quantity. → **verifier-check**

### F · A number is measured-and-asserted, never transcribed; controls must be non-vacuous
**Principle.** A claimed number must be computed live from committed artifacts and asserted
against an independently-known value; verify the *mechanism* produces the number, not that the
number appears; pin the measurement/admission convention before comparing across
implementations; and a negative control must fail on *every* way the bad path can fail —
including non-finite (`NaN > c` is `false`).
**Instances.** A "coverage restored 0.867→0.895" was an APS admission-convention artifact that
dissolved when the rule was aligned (`FIELD-NOTES.md C3`, esc-010); a "hard-zero leak
prevented" was a loader pre-filter, not engine isolation (`CASE-STUDIES.md §3a`, esc-009); a
"the bug still reproduces" control flaked because the un-standardized path diverged to NaN and
the predicate read it as a fit (`CASE-STUDIES.md §4`, esc-005).
**Mechanism.** `fix-verifier` non-finite-control rule; `adversarial-audit` honesty lens
(trace the number to its cause; pin the convention); `cookbook` measured-goldens. BLOCK a
headline number whose mechanism isn't traced or whose control can pass on an unmodelled
failure. → **verifier-check + domain-agent-invariant**

### G · The fix's test must bite (red-green); green CI is the floor
**Principle.** A regression test that does not fail *before* the fix proves nothing — revert
prod, keep the test → RED → GREEN. Green CI checks what someone asserted, not the state the
assertion forgot. A new regression test cites the escape it retires (`closes_escape`).
**Instances.** The None-overload (family B) passed fmt+clippy+test on the buggy code because
no test asserted the *cleared* post-state; the tenant leak (family A) was green (esc-001).
**Mechanism.** `fix-verifier` exit gate (phase 6): red-green + verdict ∈
{verified, tautological, not-symptom-faithful}; BLOCK a tautological test. → **verifier-check + phase-step**

### H · Cross-surface parity is byte-for-byte, on the divergence-prone case
**Principle.** When one capability has two surfaces (embedded vs remote, local vs engine,
CPU vs GPU), they must agree byte-for-byte, not "both respond" — and the parity test must
exercise the divergence-prone input (multi-chunk, boundary, empty), never only the happy path.
**Instances.** A remote `publish_topic` silently diverged from the embedded path on
multi-chunk tables, invisible to the single-chunk test (`AGENTIC-PLAYBOOK.md §1`, esc-002);
`local == engine` APS coverage only after aligning conventions (family F, esc-010).
**Mechanism.** `oracle` hard-block on embedded⇄remote byte-parity; `wire-server` domain
invariant; `adversarial-audit` remote-vs-embedded lens on the boundary case. → **oracle-hard-block + domain-agent-invariant**

### I · Content-addressable identity must be complete per-variant
**Principle.** Any identity/hash that claims "an output-affecting change changes the hash"
is load-bearing on *completeness*: enumerate the producer's full output-affecting determinant
set and confirm each is captured, per variant. Omitting one yields silent false matches. A
default-params round-trip passes vacuously *exactly where* the identity is lossy — assert
that NON-DEFAULT values of every param change the hash.
**Instances.** A `ProducingDescriptor` `definition_hash` shipped lossy for 3/5 producers
(omitted params + input anchors); the shape-audit PASSED and only a replay pressure-test caught
it (`identity-hash-completeness` memory, esc-014).
**Mechanism.** `adversarial-audit` completeness rubric (enumerate the signature, check each
determinant); `db-materialization` non-default round-trip oracle. → **verifier-check + oracle-hard-block**

### J · Determinism is engineered
**Principle.** Reproducible numerics require an explicitly fixed reduction/fold order and a
stable tie-break key (float addition is non-associative; default sorts and float ties are
unstable); and never assume the concrete type a default hands you — cast.
**Instances.** `propagate_embeddings` is byte-identical only because the `f64` fold order is
fixed (`FIELD-NOTES.md G3`); exact search was nondeterministic until ties broke on `_row_id`,
which resolves as `Utf8View` not `Utf8`, so a naive downcast broke search for sidecar-less
tables (`FIELD-NOTES.md R4`, esc-016).
**Mechanism.** Determinism domain-agent invariant (`ai-pipeline`/`db`/`bench`): no RNG, `f64`,
`total_cmp`, fixed fold order, explicit Arrow cast; a seeded/bit-repro oracle. → **domain-agent-invariant**

### K · Diagnose the structure before reaching for a tool; prefer the honest negative
**Principle.** A method only works where its assumptions hold — diagnose the geometry/structure
of the problem first, measure a claimed gain against the *strongest* baseline, and if the
honest result is "it doesn't help," that is the finding: ship it. (A weight repairs a
score-moving shift, not a location one; fusion helps only diverse arms; supervision caps the
gain not the loss; a distributional forecast needs proper scores + PIT, not point error.)
**Instances.** Importance-weighted conformal was a no-op under a location shift (`FIELD-NOTES.md
C2`); RRF fusion sat below the best single arm (`R1`); the loss kept falling after held-out
recall saturated (`O2`).
**Mechanism.** `cookbook` honesty domain-agent invariants (diagnose-then-apply, assert against
the strongest baseline, teach the honest negative). Largely discipline — flagged in §Gaps as
not fully automatable. → **domain-agent-invariant**

---

## Boundary principle families

### L · Names-no-consumer; the discipline test; one-way references; shape-the-seam
**Principle.** The engine names no consumer in *any* public artifact — code, config, docs,
tests, fixtures, scripts, PR bodies, commit messages, issue comments. Gate every new engine
surface with the discipline test: *"would a user who never heard of any particular consumer
reach for this?"* If yes → engine (generic mechanism/seam). Governance
(promote/retire/register/transition/gate/approve) is a consumer concern; mechanism
(list/describe/delete) is open-core. References point one way only. Shape a generic seam now
for foreseeable consumers, but build the consuming layer only on real demand.
**Instances.** A public issue-close comment named a consumer and leaked its internals into an
open-core issue, genericized after the fact (`platform-strand-coupling`); a boundary fork
(should the engine enforce tenant security against a hostile principal?) was first derived as
engine-owned and corrected — the trusted-network model makes it the consumer's access control
(`own-derivable-forks`, esc-020); promote/retire removed from open-core (#203).
**Mechanism.** Constitution boundary invariants; `discipline-test-auditor` semantic lens over
the whole diff incl. prose; `oracle` hard-block on `check_dep_direction.py` /
`check_cookbook_one_way.sh`; `dep-dag` freshness CI. BLOCK a surface that fails the discipline
test or a consumer-pulled layer masquerading as a seam. → **constitution-invariant + verifier-check + oracle-hard-block**

### M · Atomic-across-the-workspace; append-only history; frozen public surface
**Principle.** A behavior change ships atomically across every affected crate in one unit
(split by capability, never by crate); versions move in lockstep; migrations are append-only
and monotonically numbered; enum `Display`/`FromStr` are inverse; the public API is frozen
(the generic platform-facing seams are part of the frozen surface).
**Instances.** Lockstep `workspace.package.version`; H4 API-freeze-guard; `db-catalog`
migration/enum invariants (`CLAUDE.md` atomic-across-the-workspace, `ARCHITECTURE §5–§6`).
**Mechanism.** `oracle` hard-blocks: lockstep version, append-only migration numbering,
API-freeze-guard; `db-catalog` enum-round-trip invariant; Contract phase requires the change
span every affected crate. → **oracle-hard-block + domain-agent-invariant + constitution-invariant**

### N · The consumer-loop is the harshest integration test
**Principle.** Authoring a faithful consumer that ends every step in an independently-known
number is the integration test unit tests are not — it exercises real call *sequences* with
real data against a golden. Corollaries: one root cause can have more than one home, so re-emit
against the *shipped* fix; run the full consumer path before tagging a new public surface
(in-crate tests don't reach the consumer on-ramp); chapters read a committed cache and assert
frozen goldens, never recompute upstream.
**Instances.** The cookbook loop drove five releases and found bugs unit tests passed over
(`METHODOLOGY.md`); the standardization bug's second home surfaced only on re-emit (family C);
v0.28.0 shipped with no public on-ramp for a new verb, caught by a pre-release chapter
(`engine-cookbook-loop`).
**Mechanism.** `cookbook-emit` workflow + `cookbook` domain agent (measured-not-asserted,
read-the-cache); Ship-phase pre-tag consumer acceptance gate for any new public surface. → **phase-step + domain-agent-invariant**

---

## Process principle families

### O · The rigor chain is a discovery mechanism, run by construction
**Principle.** `plan → pressure-test → implement → independent adversarial audit → CI → merge`
is a bug-*discovery* mechanism, not ceremony — skip a step and you ship the plausible-wrong
version on green. The two load-bearing steps: **pressure-test the design before code** (a wrong
design compiles and passes its own tests; reproduce a spec's stated "why" before building its
fix — the spec is not gospel), and **independent audit before the PR** (a fresh agent given
diff+contract only, prompted to refute, default BLOCK). Never merge on green CI alone. The
audit runs the repo's `CLAUDE.md` self-check + "Dodges That Don't Fly" as a named dimension,
not just the grep-able slice.
**Instances.** The Adam loss-rescale was killed at pressure-test before a line was written
(`CASE-STUDIES.md §1`); a spec's `transaction()`-hangs premise was reproduced false and a
doc-only change shipped (`AGENTIC-PLAYBOOK.md §4`); the tenant leak, multi-chunk divergence, and
fetch-cap were all caught by audit on *green* PRs (`§1–§2`).
**Mechanism.** The lead's phase machine runs phases 1 (pressure-tester) and 4 (adversarial-audit)
before phase 7 (Ship), by construction; constitution invariant "green CI is the floor." → **phase-step + constitution-invariant**

### P · Complementary lenses; a PASS on one is not a PASS on another
**Principle.** Different adversarial lenses catch different flaws — a shape-audit is not a
completeness-audit; a design/feasibility pressure-test is not a correctness/non-vacuity audit.
Run both as *distinct* gates; never let a PASS on one substitute for the other's blind spot.
**Instances.** The lossy descriptor (family I) passed the shape-audit and failed only the replay
pressure-test (`identity-hash-completeness`); the None-overload (family B) passed local green and
failed only the independent audit (`partial-update-none-overload`).
**Mechanism.** The phase machine keeps `pressure-tester` (phase 1) and `adversarial-audit`
(phase 4) as separate gates with separate rubrics; `citation-checker` and `discipline-test-auditor`
are further distinct lenses. → **phase-step + self-failure-mode**

### Q · A delegated "done" is a claim to verify against the artifact
**Principle.** A subagent's report is evidence to audit, not a fact to accept — verify the
*artifact* (pushed commit, open PR, the FULL gate actually run); a revived/duplicate agent
narrating another's verdict is untrusted noise; when no auditor materializes, the lead
substitutes *objective* verification (re-emit into a temp dir and diff the asserted goldens — a
doctored golden can't survive a re-emit); run CI's exact full gate locally, never a subset, and
check real exit codes (never a pipe-masked `| tail && echo PASS`).
**Instances.** An implementer ran the gate then ended without committing, narrating "I'll wait
for notifications" (esc-015); a reviving agent fabricated "the audit returned PASS" for an audit
that hadn't run (esc-018); a subset gate reported fmt green on a real diff via a masked exit
(`audit-harness-fallback`).
**Mechanism.** Ship-phase artifact verification; `SELF-FAILURE-MODES.md` (done-is-a-claim,
fabricated-certification); `hooks/stop-gate.sh` + the Contract phase embed CI's exact full gate
verbatim from `.github/workflows/*.yml`, capturing `$?` per step. → **phase-step + self-failure-mode + hook + CI-gate**

### R · Gate the action on the principle; own derivable forks; size by rigor-unit
**Principle.** The meta-failure is *recalling* a discipline and not *gating the action* on it —
before any meaningful action, check the relevant principle first. Own every decision derivable
from the standing context (philosophy, principles, threat model, boundary, learnings); escalate
only a genuine product/irreversible-preference fork — but derive it *correctly*. Rigor-depth is
gated by what a change TOUCHES (contract/serve-path/keep-set/public surface), not add-vs-remove —
removals get the full chain. A PR = one rigor-chain unit; commits = steps within; fan delegation
out to commits on one branch, not to more PRs.
**Instances.** A cross-tenant leak was first halted-on then mis-derived as engine-owned
(`own-derivable-forks`, esc-020); a promote/retire *revert* was right-sized past the
pressure-test though it touched the serve path (`rigor-chain-mandate`); workstreams were
over-split into multiple PRs (`pr-sizing-rule`).
**Mechanism.** The phase machine forces the principle check by construction (each phase clears a
named gate before the next); lead-role invariants; `SELF-FAILURE-MODES.md` (deletion≠low-risk,
parallelism≠PR-count, deferring-a-derivable-fork). → **phase-step + self-failure-mode**

### S · The build environment is load-bearing and adversarial (this host)
**Principle.** The build/host environment can silently invalidate correctness — treat it as a
first-class hazard. Never override `RUSTC_WRAPPER`/`RUSTFLAGS` (changes the sccache key → cache
miss); give every worktree a *unique* `CARGO_TARGET_DIR` (a shared one serves stale artifacts, so
an auditor tests the wrong code); never `git checkout -b` in a shared checkout (it switches `main`
behind the lead's back and can push WIP to `origin/main`); pin `PYTHONPATH` against maturin
cross-worktree shadowing; treat remote CI (full checkout) as authoritative; prove a resource with
its own signal (`nvidia-smi`, not a JSON log); re-verify a memory's cited artifact before acting.
**Instances.** An `RUSTFLAGS` override turned a gate into ~100 min of redundant compiles
(esc-021); a pre-warmed shared target dir ran `main`'s 192-test binary for a 199-test branch
(`orchestration-model`); a shared-checkout `git checkout -b` stranded `main` and pushed WIP to
`origin/main` (esc-017).
**Mechanism.** `hooks/build-env-guard.sh` (PreToolUse Bash, fail-open) warns on RUSTFLAGS/wrapper
override, non-unique target dir, and disk pressure; domain agents are worktree-isolated with a
unique dir; `SELF-FAILURE-MODES.md` host-traps. → **hook + domain-agent-invariant + self-failure-mode**
(this hook stays fail-open by design — build-env hazards are recoverable nudges, not a class the
gate needs to be un-dodgeable on; contrast `hooks/lead-gate-pre.sh` under family F10, which IS
fail-closed because a lead can reword its way past a norm but not past denied state, §Per-mechanism
below.)

### T · The swarm obeys its own rules; the consumer swarm mirrors its shape
**Principle.** The swarm may *propose* to tighten itself via a human-merged PR but never
*weakens* itself (constitution + every executable gate are human-amend-only, fail-closed —
anti-Goodhart via human-in-the-loop, not a tighten-vs-weaken classifier); a fact lives in exactly one
place, and the constitution is an *anchor index* whose cited code anchors must resolve, so it
can't become the next stale doc; ownership is a strict partition (an unowned source path is a P0).
The consumer swarm mirrors the engine swarm's *shape* re-rooted in the consumer constitution — a
governing consumer composes the engine on one listener (compose, don't wire-client a separate
engine); judge architecture by seamless-upgrade UX, not purity.
**Instances.** The doc-parity gate retired the exact staleness that bit (a 7th enum variant added,
guide kept listing 6; esc-012); an architecture call was first made backwards optimizing for
purity, then corrected to composition on one listener (`jammi-upgrade-ladder`).
**Mechanism.** `CONSTITUTION_TOUCHED` / `SWARM_GATE_TOUCHED` fail-closed; `check_doc_parity.py`,
`check_constitution_anchors.py`, `check_swarm_bijection.py`; two repo-local swarms sharing shape
not files; `SELF-FAILURE-MODES.md` (purity≠product). → **constitution-invariant + CI-gate + self-failure-mode**

---

## Per-verifier PRINCIPLE-LEVEL RUBRICS (lift onto the agent's card)

Each line is an instruction to **generalize to novel code and default-BLOCK on a
novel-but-analogous smell** — not a lookup of the past bugs. The parenthetical names one
historical instance only as calibration.

### `adversarial-audit` (phase 4, default BLOCK if uncertain)
- **Guard state-collapse (family A):** for every RAII/`Drop`/`__exit__`/context guard in the
  diff, model the full entry/exit state lattice; BLOCK if any cleanup arm cannot distinguish
  "held" from "never held," or if a state the happy-path test never constructs (re-entry, reuse,
  never-entered, error-exit) is unhandled. *(cal: tenant-scope `__exit__`.)*
- **Unrepresentable state / band-aid (family B):** for any type modeling presence, selection, or
  partial-update, ask "what state can this NOT express?"; BLOCK if a required behavior lands in an
  unrepresentable state, or is rescued by `#[allow]` / `let _ =` / `// TODO: later` / `#[ignore]` /
  a companion bool beside an `Option` instead of reshaping the type. *(cal: `Option<T>` partial-UPDATE.)*
- **Domain-validity at the edge (family D):** for every operator, identify its valid input domain
  (range, set-vs-multiset, directedness, identity assumptions) and BLOCK if there is no
  validate/clamp/normalize guard at the input edge or no boundary/degenerate oracle. *(cal: negative-LR past horizon.)*
- **Bound the growing term (family E):** for any new cap/limit, name the unbounded term and BLOCK
  if the bound is placed on an aggregate that contains a caller-controlled quantity. *(cal: fetch cap over `k`.)*
- **Identity completeness (family I):** for any content-addressable identity/hash, enumerate the
  producer's full output-affecting determinant set and BLOCK if any determinant is uncaptured or if
  the only round-trip test uses default params. *(cal: lossy `ProducingDescriptor`.)*
- **Cross-surface parity (family H):** if the diff has two surfaces for one capability, BLOCK unless
  a byte-parity oracle exercises the divergence-prone input, not the happy path. *(cal: multi-chunk publish.)*
- **Honesty of numbers (family F):** for every headline metric, BLOCK unless the mechanism producing
  it is traced (remove the claimed cause, confirm the number moves), the measurement/admission
  convention is pinned before any cross-implementation comparison, and no manufactured/overclaimed
  result stands. *(cal: APS-convention artifact.)*
- **Principle adherence (family O):** run the repo's `CLAUDE.md` self-check + "Dodges That Don't Fly"
  as a distinct verdict dimension over the diff — wrong abstraction, non-atomic crate split,
  stringly-typed-where-an-enum-belongs, consumer-name leak.

### `pressure-tester` (phase 1)
- Attack the *design* against first principles before any code; BLOCK a design that would compile
  and pass its own tests but is mathematically/architecturally wrong (family C, O).
- Treat every spec premise (especially a stated "why it's broken") as a claim to *reproduce*, not a
  fact — "the spec said so" is a dodge (family O).
- Feasibility / "can this be replayed/reconstructed/round-tripped?" is a *distinct* lens from the
  correctness audit — run it as its own gate (family P).

### `fix-verifier` (phase 6, exit gate)
- Red-green: revert prod, keep the test → require RED → GREEN; BLOCK any test that does not fail
  pre-fix (tautological) (family G).
- Non-finite control: a "the bad path fails" control that can pass on NaN/±inf or any unmodelled
  failure mode is vacuous → BLOCK; require the control to count every way the bad path can fail
  (family F).
- Verdict ∈ {verified, tautological, not-symptom-faithful}; the new regression test cites the
  `closes_escape` id it retires.

### `discipline-test-auditor` (semantic engine-not-platform lens)
- Apply the discipline test to any new engine surface — *"would a user who never heard of any
  consumer reach for this?"*; BLOCK a surface that fails it (family L).
- Governance stems (promote/retire/register/transition/gate/approve/stage) are consumer concerns;
  mechanism (list/describe/delete) is open-core — BLOCK a governance-shaped verb in the engine.
- Names-no-consumer across code, config, docs, tests, fixtures, scripts, AND public PR bodies /
  commit messages / issue comments — BLOCK a consumer name or leaked consumer-internal anywhere in
  the diff or its prose.
- A generic seam is allowed; a consumer-pulled *layer* masquerading as a seam is BLOCKed (shape the
  seam, build the layer on demand).

### `citation-checker`
- Re-read every cited `path:line` in a verdict or doc; BLOCK on a fabricated or stale citation; a
  memory's cited artifact is re-verified against current code before it is asserted as fact.

### `oracle` hard-blocks (phase 5, not consensus-overridable)
- `check_dep_direction.py` / `check_cookbook_one_way.sh` — references point one way (family L).
- Lockstep `workspace.package.version`; append-only monotonic migration numbering; H4 API-freeze of
  the public + generic-seam surface (family M).
- Tenant-isolation seam — every wire RPC is a tested cross-tenant-denial CASE (family D/L).
- Embedded⇄remote byte-parity (family H); per-head high-offset / LR-boundary / step-count oracles
  (family C/D); descriptor non-default round-trip (family I).

### CI gates (fail-closed)
- `check_doc_parity.py` — documented enum ⇄ code enum set-equality + exception arms (family T, SHIPPED #245).
- `check_swarm_bijection.py` — every source path owned by exactly one domain; unowned = P0 (family T).
- `check_constitution_anchors.py` — every constitution `code anchor` still resolves (family T).
- `CONSTITUTION_TOUCHED` / `SWARM_GATE_TOUCHED` — self-tightening only, human-amend-only, fail-closed (family T).

### `hooks/build-env-guard.sh` (PreToolUse Bash, advisory / fail-open)
- Warn on any `RUSTFLAGS` / `RUSTC_WRAPPER` override, any non-unique `CARGO_TARGET_DIR`, and NVMe
  disk pressure — the class of "the build env silently invalidated correctness" (family S).

### `hooks/lead-gate-{start,stop,pre}.sh` (SubagentStart/SubagentStop/PreToolUse, FAIL-CLOSED)
- The lead-proactivity gate (family F10, below): the ONE hook in this repo whose decider DENIES
  on internal error rather than allowing (contrast every hook above, which is advisory / fail-open
  by design — a build-env or routing NUDGE costs nothing to miss; a lead that dodges a probe
  requirement by rewording its prompt does not). State is written by the harness from a verifier's
  own `<verdict>` JSON, never the lead's prose; a relay is allowed only if it names every site the
  verifier's `class_enumeration` enumerated.

### `SELF-FAILURE-MODES.md` (trigger → prevention)
- Fabricated certification, "done" as a claim, subset/masked-exit local gate → verify the artifact
  objectively; run CI's exact full gate (family Q).
- Deletion ≠ low-risk; parallelism ≠ PR count; deferring a derivable fork; check-the-principle-before-
  the-action (family R).
- One root cause can have two homes → re-emit against the shipped fix (family N); prove-the-resource-
  not-the-log, pin PYTHONPATH, never `git checkout -b` in the shared checkout, re-verify a memory's
  artifact (family S).
- Purity ≠ the product: judge consumer-swarm architecture by seamless-upgrade UX (family T).

### Domain-agent invariants (by crate — see `ARCHITECTURE §6`)
- `ai-model`: standardize-in-data-space + high-offset oracle per head (C); LR clamp/floor/realized-
  steps + step-count oracle (D); gate on the downstream metric, diagnose shift geometry, proper-
  score+PIT, index-single-owner-of-vectors, keep `(n+1)` (K).
- `ai-pipeline`: domain-validity at every numeric edge, pin set-vs-multiset (D); fixed fold order (J).
- `db-materialization`: descriptor completeness round-trip (I); stable tie-break + explicit Arrow cast (J); shape-the-nucleus-now (L).
- `db-catalog`: append-only migrations, enum Display/FromStr inverse (M).
- `wire-server`: tenant scope = row predicate, every RPC a tenant-iso CASE (D/L); embedded⇄remote parity (H); H4 API-freeze (M); explicit selector on ambiguous input (B).
- `python`: pin PYTHONPATH, verify the built artifact (S).
- `cookbook`: measured-not-asserted, read-the-cache, honest-negative, pre-tag consumer acceptance (F/K/N).
- `bench`/`db`: numpy-first oracle suites for numeric guarantees (F); generic fixtures only (L).

---

## Gaps — principles with no crisp mechanism yet (candidate new gates)

- **G-a · "One root cause has two homes" (family N) has no *mechanical* sibling-finder.** Re-emit
  catches it only if a chapter exercises the sibling subsystem; nothing enforces sweeping every site
  in the same root-cause class. Candidate: a `retrospective`-loop rule that, on closing an escape,
  greps for structurally-similar call sites (same producer family, same numeric edge) and files a
  sibling-check task.
- **G-b · Honesty-of-numbers (family F) is audit judgment, not a check.** "Verify the mechanism
  produces the number" and "manufactured benchmark is a blocker" have no fail-closed teeth beyond a
  sufficiently adversarial reader. Candidate: for cookbook headline metrics, an oracle that recomputes
  the number two independent ways (engine vs numpy-first) and BLOCKs on divergence — mechanizing the
  `local == engine` reconciliation that dissolved the APS illusion. (Partially covered by measured
  goldens; the convention-artifact class is not.)
- **G-c · Diagnose-then-apply / honest-negative (family K) is a teaching invariant with no automatable
  oracle** — "gate on the downstream metric," "diagnose shift geometry before weighting" are judgment
  calls a domain agent applies, living only as card prose. Acceptable as discipline, flagged so no one
  mistakes it for enforced.
- **G-d · The `discipline-test-auditor` semantic lens (family L) has no ground-truth oracle** beyond
  `check_dep_direction.py`, which catches only *dependency-level* leaks — a governance-shaped surface
  that imports nothing new would pass the dep gate. Candidate: a curated allow/deny list of governance
  verb-name stems (promote/retire/register/transition/gate/approve) as a grep tripwire feeding the
  auditor.

## 2026-08-28 — the attention-backends train (M1b/M2/M3), operational lessons

Recorded by the lead for future sessions; each cost at least one real round-trip.

- **Pod-result provenance is a discipline, not a nicety.** `gpu-dev.sh` derives its tree from the SCRIPT's location; a lead invoking the wrong checkout's copy silently validated main instead of the branch — caught only by the push-stamp's `laptop_head`. Rule: read the stamp back after every push, before trusting any leg. (The substrate now refuses cwd/root mismatch — esc-056.)
- **Never filter inside a pod job.** In-job `| grep`/`| tail` destroyed failure diagnostics three separate times (compile errors, panic text, a 66-line log for a full suite). The `.jammi.log` must hold full output; filter over ssh at read time. Corollary: pipelines swallow exit codes — read the wait-verb's `rc=` verdict line, never a piped RC.
- **80GB-class oracle legs are a device-capability domain.** The encoder-level flash oracles (production-scale fwd+bwd, one arm ≈ 40-60GB) are structurally impossible on 48GB SKUs — proven by solo-run OOMs on an empty 46GB card after a serialization theory was refuted by its own test. The honest state is a named VRAM capability skip plus a per-arch/per-leg coverage table, not a fixture shrink (the fixture's scale IS what it proves).
- **CUDA is bitwise deterministic per (arch, build).** 40 repeats, identical worst element, both Ada pods. Cross-run "flicker" of a violating element means the BUILD changed between runs. And sm86 vs sm89 select different cuBLAS accumulation kernels despite the identical smem tier — per-arch bounds derive from per-arch measurements (the K_MAX lesson's real generalization).
- **Validated ⊆ compiled must be representable or it is not real.** Round-2 proved an unvalidated arch could join the compiled set with every test green; the fix (VALIDATED_SMS as a second single-source array every fence reads, divergence pinned by an env!-reading hermetic test that runs in the DEFAULT lane) is the pattern: an invariant asserted only under cfgs no lane runs is prose.
- **Auditors should measure, and fixers should re-run the auditor's mutants.** The strongest rounds used tracking allocators and scratch-copy mutants on both sides; "verify by re-running the auditor's exact mutant — RED required" is the acceptance bar that made convergence fast (rounds shrank 5→4→3→1→0 findings). Known gate gap: check_doc_numbers_have_producers' regexes do not match MB/GB tokens — memory-model prose needs inline derivations until a tightening lands.
- **Verdict hygiene**: the gate parses the LAST fenced json block of an agent's final message; several verdicts landed UNBOUND/UNPARSEABLE from formatting. Verifier dispatch prompts should demand the block last.
- **Fleet economics**: rent all arches at once (seed wall = max, not sum); keep pods warm across phases with queued work (warmup, not idle dollars, is the cost); the failure case delta-pushes to all pods and reruns affected legs — still parallel.
