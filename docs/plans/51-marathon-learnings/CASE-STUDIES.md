# Case Studies — anatomies of the marathon's deepest findings

The proof behind `METHODOLOGY.md` and `AGENTIC-PLAYBOOK.md`. Each case follows the same
shape: **how it surfaced → the wrong first hypothesis → what caught it → the fix → the
transferable lesson.** These are the findings worth understanding in full; the rest are
in the `CHANGELOG`.

---

## 1. The Adam / standardization lesson — a wrong *design* caught before code, then an
## incomplete *fix* caught by the loop

**How it surfaced.** The keystone cookbook vertical tried to predict a calendar year
(target ≈ 2017) with the Gaussian context predictor. The served distribution collapsed:
the mean stuck near the zero-initialised value (~2163 off, in one run) and the variance
floored. A regression head that cannot reach `years`, `prices`, or `counts` is unusable —
this was a real, severe bug, found only because a chapter asked for a real high-offset
number.

**The wrong first hypothesis.** The natural fix: the loss on a target of magnitude ~2000
is enormous, so *rescale the loss* to tame it.

**What caught it — pressure-test, before any code.** Reasoning from first principles
about the optimiser: Adam divides each parameter's update by a running RMS of *that
parameter's own gradient*, so the effective step is ~`lr` per step in parameter space,
**independent of the loss's scale**. Multiplying the loss by a constant scales every
gradient by that constant — and Adam divides it right back out. The parameter still has to
travel ~2000 units at ~`lr`/step, which it cannot do in the training budget. Loss-rescaling
is a no-op for *how far a raw parameter moves*. The design was mathematically wrong, and it
would have compiled and passed a "loss went down" test.

**The fix.** Standardize in *data/representation space*, not the loss: z-score the target
(and the in-context members' targets) with one train-derived scaler, train the head in
that space, and apply a persisted de-standardization affine in the forward pass so the
served distribution comes back in original units. The scaler is persisted with the model
and reloaded. (`crates/jammi-ai/src/fine_tune/{regression_loss,target}.rs`,
`pipeline/context_predictor.rs`.)

**Then the loop caught the fix was incomplete.** The first release standardized the
fine-tune projection head. The cookbook *re-emit* against that release showed the
amortized context predictor — a *separate* subsystem — still collapsed. The same root
cause had two homes; only authoring the chapter a second time, against the shipped fix,
exposed the second. It was fixed in the next patch.

**Transferable lesson.** Every trainable head on a high-offset / low-variance /
large-magnitude target needs a standardization-and-domain contract, and the standardization
must live in the *space the optimiser moves through*, not in the loss. Add a high-offset
oracle for *every* head — and re-run your consumer against the shipped fix, because one
root cause can have more than one home.

---

## 2. Domain-validity — one root cause wearing four disguises

**How it surfaced.** A deliberate adversarial sweep of the training, graph, search, and
catalog surfaces (independently-computed answers + degenerate/boundary inputs) reddened
several oracles at once.

**The findings, which looked unrelated:**
- The fine-tune **learning-rate schedule went negative** past its horizon — the horizon
  miscounted the realised optimizer steps (it ignored each epoch's trailing
  gradient-accumulation flush), so progress exceeded 1.0 and the linear decay ran below
  zero.
- The regression head's **mean blew out** on a low-variance target (the standardization
  story, case 1).
- **Undirected graph propagation double-counted**: an edge list that declared both
  directions inflated each node's degree, so the degree-normalised average was wrong.
- A **"tenant-isolated" source was globally readable** (case 3).

**What caught it — naming the common shape.** These are one bug: *the engine computed
confidently past its valid input domain.* A schedule evaluated past its horizon; a head
evaluated outside its trained magnitude range; a degree computed on a multiset that should
have been a set; an identity assumed where none was enforced. Each produced a confident,
wrong number rather than an error.

**The fix (uniform in shape).** Validate / clamp / normalize at the input edge:
`compute_lr` clamps progress to `[0,1]` and floors the rate at zero and the horizon counts
realised steps; the regression head learns in standardized space; undirected edges collapse
to the unordered-edge set the other graph operators already use; the tenant boundary is
made explicit and documented.

**Transferable lesson.** When several bugs across unrelated subsystems share a smell, look
for the *one invariant* they all violate — fixing the invariant (here: input-domain
validity, as a review lens and a class of boundary property test) generalises in a way that
four independent patches never would.

---

## 3. The multi-tenant model — an honesty catch *and* a green-CI-insufficient catch

Two tenant-related findings, each teaching a different discipline.

**3a — The measurement-honesty catch (consumer → engine).** A tenancy chapter dramatically
demonstrated a "hard-zero leak prevented by registering a separate source per tenant":
tenant A's query returned exactly zero of tenant B's rows. It read like strong isolation.
The audit traced the zero to its actual cause: the *loader's own Python pre-filter* had
already removed B's rows before the query ran. Removing `with_tenant` left the zero intact,
and tenant A could `SELECT` tenant B's source *by name* — so source-separation isolated the
catalog *listing*, not the *data*. The chapter was reframed to the engine's true two-layer
model: catalog-row isolation (list/get filtered by tenant) **and** row-level isolation via a
`tenant_id` discriminator column (`crates/jammi-db/src/tenant_scope.rs`), with the honest
caveat that a discriminator-*less* federated source is globally readable — the engine does
not authenticate; that boundary lives above it. **Lesson:** a measured result can be real
*and* mean something other than you claim; verify the *mechanism* produces the number, not
just that the number appears.

**3b — The green-CI-insufficient catch (the audit).** Fixing the `with_tenant` ergonomics
(it bound in place and returned `None`, a footgun) introduced a scope-safe context manager,
`with db.tenant_scope("t"): …`, that restores the prior scope on exit. The PR was green.
The independent audit found that `__exit__` used `.take().flatten()`, which collapsed *two
distinct states* — "never entered" (`None`) and "entered while unscoped" (`Some(None)`) —
into the same `unbind` arm. Consequences: reusing one scope object clobbered its captured
prior, and a stray `__exit__` on a never-entered scope *cleared a live tenant scope*. That
is a tenant data-scope leak, on green CI, on the very feature meant to make scoping safe.
The fix matches the full `Option<Option<TenantId>>` (`Some(Some(t))` restores the prior,
`Some(None)` unbinds, `None` is a no-op) and refuses re-entry. **Lesson:** green CI checks
what someone asserted; an adversarial reader checks the states the assertions forgot — and
the worst bugs hide in the state a happy-path test never constructs.

---

## 4. The flaky negative control — when "the bug still reproduces" passes vacuously

**How it surfaced.** A CI run for an *unrelated* fix reddened on a test that asserts the
*un-standardised* path (case 1's bug) still collapses, proving the oracle exercises the bug.

**The wrong reading.** "A real regression in the standardisation fix." But the failing test
was in a different crate from the change, and it had passed on a sibling PR off the same
base — it was *flaky*, not regressed.

**What caught it — reading the assertion, not just the red.** The control asserted
`mean far OR σ floored`. The un-standardised path is numerically unstable on a high offset
and sometimes **diverges to NaN**; then `NaN > 100 || NaN < 0.05` is `false || false`, so a
*divergence* — the strongest possible collapse — was read as a *successful fit* and failed
the control. The fix makes a non-finite served value count as collapse.

**Transferable lesson.** A negative control can pass *vacuously* when the bad path fails in
a way the predicate didn't model. When a "the bug reproduces" assertion flakes, suspect the
predicate before suspecting a regression — and make "failed to fit" include the failure
modes you didn't enumerate (here, non-finite), or seed the run deterministically.
