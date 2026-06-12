# How a Cookbook Hardened an Engine — the method

A retrospective on the work that carried `jammi_ai` from v0.25.0 to v0.26.4. It is
written for a human engineer or researcher who wants the *method*, not the changelog —
what we did, why it worked, and what transfers to any engine of numerical/ML primitives.
The full bug list is in the `CHANGELOG` and git history; this is the distillation.

Companions: `AGENTIC-PLAYBOOK.md` (the same lessons as operating rules for AI agents)
and `CASE-STUDIES.md` (the deepest findings, worked end to end).

---

## The arc

We set out to write a *consumer cookbook*: a Quarto book that teaches the engine by
running real workflows — embed a citation graph, build a neighbour graph, propagate,
fine-tune, predict with calibrated uncertainty — against committed datasets, each
recipe ending in a measured number asserted against a golden value.

The intent was pedagogy. What we got was the most effective integration test we had ever
written. Authoring the keystone vertical immediately surfaced that the engine *silently
ignored the `epochs` argument* and that the Gaussian context-predictor *collapsed on a
high-offset target* — bugs that every existing unit test had passed over because no test
exercised the full call sequence with realistic data and an independently-known answer.
That set the pattern for the next several weeks: every chapter we authored found engine
bugs, and every bug we fixed became a chapter. Five releases (0.26.0 → 0.26.4) and dozens
of fixes later, the well-trodden paths were solid and the method had a name.

## The method

### 1. The bidirectional loop is the thesis

The relationship between a faithful consumer and an engine runs *both ways*, and that is
the entire point:

- **Consumer → engine.** Authoring a measured workflow is an integration test of the
  harshest kind. It exercises real call *sequences* (not isolated functions), with real
  data shapes, against a number you computed independently. That is exactly the regime
  where a primitive engine reveals it has been computing confidently past its valid input
  domain.
- **Engine → consumer.** Every verb the engine *soundly* supports earns a chapter. The
  cookbook grows as the engine's proven surface grows; the chapter is the acceptance
  test that the verb is real.

So the loop is self-propelling: authoring finds bugs, fixing bugs unlocks chapters,
chapters find more bugs. It does not converge to "done" — it converges to *trust*. The
hardening roadmap makes this a standing commitment: no capability is considered
mainstream-ready without a measured chapter that runs in CI.

### 2. Measured, not asserted

A recipe without a real number — computed from committed artifacts, checked against a
golden value — is not done. This is the discipline that gives the loop its teeth. It is
easy to write a chapter that *claims* propagation improves recall; it is the assertion
`recall_after > recall_before`, computed live from the cache, that catches the day
propagation silently regressed. The same rule killed several tempting fictions (§6).

A corollary: chapters **read a committed cache and assert against frozen goldens; they do
not recompute upstream.** Re-execution drifts, drift fails tolerances, and a tolerance
failure with no provenance is the worst kind of red. The keystone emits the cache once;
every later chapter loads it and asserts.

### 3. The rigor chain — and why pressure-test and audit are load-bearing

Every change went through `plan → pressure-test → implement → independent adversarial
audit → CI → merge`, and the two steps a hurried engineer skips are the two that earned
their keep:

- **Pressure-test the plan, before code.** The most expensive bugs live in the design,
  not the implementation — and a wrong design *compiles and passes its own tests*. The
  marquee example: a plan to fix a target-collapse bug by rescaling the training loss.
  Pressure-testing it against first principles caught that Adam normalizes each parameter
  by its own gradient RMS, so a raw parameter moves ~lr/step *regardless of loss scale* —
  loss-rescaling cannot fix it. The correct fix standardizes the data the head conditions
  on, not the loss. Caught before a line was written (see `CASE-STUDIES.md`).
- **Independent adversarial audit, before merge.** A fresh reviewer — invested in
  refuting, not defending — finds what the author and the test suite cannot, because the
  author is committed to their solution and the suite only checks what someone thought to
  assert. On *green* PRs, audit caught a tenant-scope-restore leak, a silent remote/embedded
  divergence on multi-chunk publish, and an abstraction that capped the wrong quantity.

The standing rule that falls out: **never merge on green CI alone.** CI is the floor.

### 4. Adversarial oracle sweeps

Before a patch release we ran a deliberate sweep of a subsystem: construct
independently-computed correct answers plus degenerate and boundary inputs, and treat any
failing oracle as a repro. One such sweep of the training/graph/search/catalog surfaces
produced a whole batch release of correctness fixes at once. The oracles double as
proto-chapters — a real workflow with a verified answer is already most of a cookbook
recipe — so the bug hunt and the teaching share their work.

### 5. Domain-validity — the one invariant under all of it

The single most valuable abstraction to come out of the marathon: nearly every serious
bug was the same bug wearing different clothes — **the engine computed confidently past
its valid input domain.** A learning-rate schedule that went negative past its horizon.
A regression mean that blew out because the target wasn't standardized. A degree count
inflated by double-counting an undirected edge. A "tenant-isolated" source that was in
fact globally readable. Different subsystems, one root cause: a function applied outside
the domain where its result means anything.

Name it once and it becomes a review lens and a class of property test, rather than a
sequence of one-off patches: validate, clamp, or normalize at every numeric and catalog
input edge, and add a boundary/degenerate oracle for each.

### 6. Honesty is a release gate

A manufactured benchmark, a mismeasured fix, or an overclaimed guarantee is a *blocker*,
not a nuance — and holding that line is the reason the result is trustworthy. The
marathon repeatedly chose the honest negative over the satisfying story:

- The keystone's uncertainty chapter was *expected* to show that importance-weighted
  conformal restores coverage under distribution shift. It does not, when the shift is a
  location/orthogonal one — the score distribution barely moves. Rather than stage the
  expected win, the chapter teaches the true result: weighted conformal repairs a
  score-moving shift, and a location shift needs a governed, time-aware cohort. The
  honest negative is a *better* lesson than the fake positive would have been.
- A "coverage restored from 0.867 to 0.895" turned out to be an artifact of two different
  APS conventions; aligning the local computation to the engine's admission rule made
  `local == engine` and the illusion vanished.
- A tenancy chapter's dramatic "hard-zero leak, prevented by per-tenant sources" was a
  loader pre-filter artifact, not engine isolation. The audit caught the overclaim; the
  chapter was reframed to the engine's true two-layer model (catalog-row + discriminator-
  column isolation, with the honest caveat that a discriminator-less federated source is
  globally readable).

If the honest answer is "it doesn't help," that is the finding, and you ship it.

## Why it worked, and what transfers

None of these mechanisms is exotic. What made them effective together is that they attack
the *failure modes of confident software* from complementary angles: the loop and
measured-numbers surface the anomaly; the sweep makes the search systematic; the rigor
chain converts the anomaly into a *correct* fix rather than a plausible-wrong one
(pressure-test kills the wrong design, audit kills the wrong fix); domain-validity gives
the fixes a common shape so they generalize; and honesty stops the whole apparatus from
quietly manufacturing the result it hoped for.

For anyone building an engine of generic numerical or ML primitives, the transferable
core is two sentences. **Write a faithful consumer that ends every step in a measured,
independently-known number — it is the integration test your unit tests are not.** And
**treat pressure-test and independent audit as where bugs are caught, not as ceremony —
because a wrong design and a plausible-wrong fix both sail through green CI.**
