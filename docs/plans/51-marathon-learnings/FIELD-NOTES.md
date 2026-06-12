# Field Notes — the domain lessons, as anomaly cards

The ML, search/retrieval, graph, and conformal lessons the v0.25→0.26.4 marathon
*taught, validated, or corrected* — distilled for absorption, not reference. The
how-to lives in the cookbook chapters; the process lives in `METHODOLOGY.md` and
`AGENTIC-PLAYBOOK.md`. This is the **so-what**.

## How to read this

Each lesson is a **card** with a fixed shape. Lessons are remembered through *active
recall* and the *correction of a wrong belief* (we encode the fix to a mistaken
prediction far more strongly than a stated fact), so each card makes you **predict
before you read**, then contrasts the naive model with what actually happened, gives the
mechanism at two depths, and ends with the portable rule.

```
Claim      — the one line you'd put on a flashcard front
Predict    — a concrete prompt; guess the answer before reading on
Naive      — what most would assume, and why it's reasonable
Happens    — what the measurement actually showed (with the number)
Mechanism  — why: one line, then the deeper derivation
Rule       — the transferable takeaway, portable beyond this engine
Evidence   — the chapter / test / release that proves it
```

Read a card top-to-bottom to absorb; skim **Claim + Rule** to review; follow **Evidence**
into an executable chapter for proof. Every number here is asserted live against a
committed golden somewhere — nothing is staged.

## The spine

One invariant threads all of these, and it is the most valuable thing the marathon
produced: **the engine kept computing confidently past its valid input domain.** A
schedule evaluated past its horizon, a head evaluated outside its trained magnitude, a
degree computed on a multiset that should be a set, an "isolated" source with no isolation
predicate — each returned a *confident wrong number* rather than an error. Two cards (S1,
S2) make the spine explicit; the rest are it, wearing a domain's clothes.

---

# I · Optimization & training

### O1 · You can't fix a large-magnitude regression by scaling the loss
- **Claim.** Under Adam, rescaling the loss does not change how far a raw parameter travels.
- **Predict.** You fine-tune a head to regress publication year (≈2017). Training looks
  stuck near 0, the loss is enormous, so you multiply the loss by `1e-6` to tame it. After
  training, the predicted mean is ≈ ____ ?
- **Naive.** Big target → big loss → unstable updates; calm it by scaling the loss down.
- **Happens.** Identical. The mean stays near the zero-init (~2163 off, in one run); the
  variance floors. Loss-scaling changed *nothing*.
- **Mechanism (1 line).** Adam divides each parameter's step by the RMS of *its own*
  gradient — scale the loss by `c`, every gradient scales by `c`, Adam divides it right
  back out; the step is ~`lr` per step regardless of loss scale. **(deep)** The update is
  `-lr·m̂/(√v̂+ε)`; `m∝g`, `√v∝|g|`, so the ratio is scale-free. To reach 2017 from ~0 the
  parameter must travel ~2017 units at ~`lr`/step — impossible in budget. Standardizing the
  *target* into ~`[-3,3]` makes the destination reachable; a persisted de-standardization
  affine returns the served distribution to original units.
- **Rule.** Standardize in the space the optimizer moves through (data/representation),
  never in the loss. Add a high-offset oracle to *every* trainable head.
- **Evidence.** cookbook `04-predict`; engine 0.26.1 (fine-tune head) + 0.26.2 (context
  predictor — the same bug had two homes); `CASE-STUDIES.md` §1.

### O2 · Supervision caps the gain, not the loss
- **Claim.** A fine-tune's ceiling is the signal in the labels, not the optimizer.
- **Predict.** You fine-tune embeddings for retrieval and the contrastive loss keeps
  dropping. Held-out recall keeps climbing too — true or false?
- **Naive.** Lower loss → better representation → better retrieval, monotonically.
- **Happens.** The downstream metric saturates while the loss still falls (LoRA / MNRL /
  graph-FT all work, but the lift is bounded — tier-03 commits a model at recall ~0.548
  and adding optimization does not break the ceiling).
- **Mechanism.** You are minimizing the *objective*, not the *task*; once the model has
  extracted the mutual information the supervision carries about the target, further loss
  reduction is overfitting the objective, invisible to the held-out metric.
- **Rule.** Gate on the downstream metric, not the loss. When held-out saturates, stop —
  more epochs buy loss, not skill.
- **Evidence.** cookbook `08-finetune-methods`; tier-03.

### O3 · Hard-negative mining is O(N log N), not O(N²) — and the OOM is *copies*, not the algorithm
- **Claim.** With an ANN index, mining is sublinear; memory blows from materializing the
  corpus several times, and the right bound is on the term that's unbounded.
- **Predict.** Hard-negative mining OOMs at ~1500 pairs. Is the cost quadratic in N?
- **Naive.** Mining = compare every anchor to every candidate → O(N²) similarity matrix.
- **Happens.** No — the index is HNSW (USearch), search ~O(log N), mining ~O(N log N). The
  OOM came from holding **3–4 full copies** of the corpus embeddings at once (a redundant
  id→vector map plus whole anchor/positive arrays). Separately, the over-fetch cap
  `min(k+excluded+1, MAX)` silently capped *k* — the bound belonged on the *excluded
  headroom*, leaving `k+1` uncapped.
- **Mechanism.** ANN gives sublinear retrieval; the quadratic intuition is for brute force.
  The unbounded quantity was *resident copies* (and the k-hop excluded set), not compute.
- **Rule.** Profile *which* quantity is unbounded and bound *that term*, not the aggregate
  it sits inside. ANN ≠ O(N²). The index should be the single owner of its vectors.
- **Evidence.** engine 0.26.4 (`hard_negative_miner.rs`); roadmap 3.1/3.3.

### O4 · A learning-rate schedule can return a *negative* rate
- **Claim.** A schedule evaluated past its horizon does gradient *ascent*.
- **Predict.** Linear-decay LR with gradient accumulation. Past the intended horizon, the
  LR is ____ ?
- **Naive.** It decays to 0 and stays at 0.
- **Happens.** It went **negative** — the horizon undercounted realized optimizer steps (it
  ignored each epoch's trailing accumulation flush), so `progress > 1` and `lr₀·(1−progress)`
  dropped below zero.
- **Mechanism.** `progress = step/horizon`; an undercounted horizon pushes progress past 1,
  and linear decay has no floor — a textbook domain-validity violation (the schedule
  evaluated outside `[0,1]`).
- **Rule.** Count realized steps (including accumulation flushes); clamp progress to
  `[0,1]`; floor the rate at 0.
- **Evidence.** engine 0.26.1 (LR horizon fix).

### O5 · Warmup is a learning-rate ramp, not extra epochs
- **Claim.** `warmup_steps` and the epoch count are orthogonal; conflating them is a
  category error.
- **Predict.** You pass `epochs=2` to graph fine-tune. How many passes over the data run?
- **Naive.** Maybe 3 — a warmup/zeroth epoch on top.
- **Happens.** Exactly 2. There is one `for epoch in 0..epochs` loop, shared by tabular and
  graph; `warmup_steps` ramps the LR per *step*, inside the loop. (An earlier bug *did*
  silently ignore `epochs` and run 3 — fixed — which is why the exact count is now pinned by
  an oracle.)
- **Mechanism.** Warmup is an LR-schedule device acting on steps; the epoch loop is data
  iteration. Different axes.
- **Rule.** Keep schedule devices off the iteration count, and lock the step count with an
  oracle (`steps == epochs × ⌈batches/grad_accum⌉`).
- **Evidence.** engine `fine_tune` epochs fix + 0.26.4 step-count oracle
  (`ft_correctness_sweep.rs`).

---

# II · Search & retrieval

### R1 · Fusing rankers doesn't reliably help — it can dilute the best arm
- **Claim.** RRF rewards cross-arm agreement; if one arm dominates, fusion drags it down.
- **Predict.** Dense recall@10 = 0.538, propagated = 0.556. RRF-fuse them → recall@10 = ____ ?
- **Naive.** Fusion combines complementary signal → meets or beats the best arm.
- **Happens.** Every fusion arm (0.550 / 0.542 / 0.529) sits *below* the best single arm
  (0.556). `fusion_helps = False`, asserted live.
- **Mechanism.** Reciprocal-rank fusion sums `1/(k+rank)` across arms — it boosts items
  *multiple arms agree on*. When one arm strictly dominates and the others are
  correlated-but-worse, the dominant arm's top items lose rank to averaged-in weaker votes.
  Fusion pays when arms are **diverse and individually competitive**.
- **Rule.** Fuse for diversity, not by reflex; always measure fusion against the best single
  arm before claiming a gain.
- **Evidence.** cookbook `10-retrieval` (live RRF re-fold to golden).

### R2 · Propagation is retrieval-conditioned denoising — it lifts recall on homophilous graphs
- **Claim.** `propagate_embeddings` is a low-pass filter; on graphs where neighbors share
  the label, it sharpens retrieval.
- **Predict.** Propagate embeddings 2 hops over a citation graph. Recall@10 from 0.538 to ___?
- **Naive.** Smoothing might wash the signal out.
- **Happens.** Up to **0.556** (arxiv, 4k papers); **0.747 → 0.919** (air routes, 3.5k
  airports). It denoises.
- **Mechanism.** `ÂᵏX` averages each node with its neighborhood. On a homophilous graph the
  neighborhood shares the target signal, so averaging cancels idiosyncratic noise — until `k`
  is large enough to over-smooth (mitigated by an `α`-teleport restart, hops capped at 3).
- **Rule.** Propagation helps at small `k` on homophilous graphs; treat `k`/`α` as the
  filter cutoff, not free knobs.
- **Evidence.** cookbook `01-02`; tier-02; the bridge.

### R3 · `search(source)` is underdetermined once a source has multiple embedding tables
- **Claim.** "The embeddings" is not a function of the source alone.
- **Predict.** A source carries two embedding tables. You call `search(source, query)`. Result?
- **Naive.** It searches "the" embeddings.
- **Happens.** Ambiguous — `No ready embedding table for source`. The caller must pick; the
  cookbook folds explicitly and reports the missing `table=` selector as an engine finding.
- **Mechanism.** Implicit selection only works while the choice is unique; a second embedding
  set makes the verb's input domain ambiguous.
- **Rule.** When implicit selection becomes ambiguous, add an explicit argument — don't
  guess a default.
- **Evidence.** cookbook `10-retrieval`; roadmap 3.2.

### R4 · Tied distances and Arrow types are silent reproducibility traps
- **Claim.** Exact search must break ties on a stable key, and never assume an Arrow array's
  concrete type.
- **Predict.** Two candidates tie on distance in exact search. The returned order is ____ ?
- **Naive.** Stable / deterministic.
- **Happens.** Nondeterministic until ties were broken on `_row_id` — and `_row_id` resolves
  as `Utf8View` (not `Utf8`) under the engine's default schema, so a naive downcast to
  `StringArray` failed and exact search was *broken for tables without an ANN sidecar*.
- **Mechanism.** Float ties + an unstable sort = order-dependent output; and a DataFusion
  default changed the concrete array type — two representation assumptions that broke
  silently.
- **Rule.** Break ties explicitly on a stable key; cast Arrow arrays, never assume the
  concrete type a default hands you.
- **Evidence.** engine 0.26.1 (tie-break + `Utf8View` fix; `exact.rs`).

---

# III · Graph

### G1 · Propagation = low-pass filter = SGC/APPNP = neighbor mean — one object, four names
- **Claim.** The `weighting` argument *is* the choice of graph filter; the op is the GNN
  canon's forward pass.
- **Predict.** How does `propagate_embeddings(weighting=…)` relate to SGC and APPNP?
- **Naive.** A bespoke engine operation.
- **Happens.** It *is* `ÂᵏX`. `uniform` = the random-walk operator `D̃⁻¹Ã` (SGC-flavored);
  `degree_normalized` + `α` = symmetric-normalized propagation with teleport restart,
  `(1−α)Â + α·X⁽⁰⁾` (**APPNP**); `output="jumping_knowledge"` = stack the filtered signals.
  One equation, three names, one call.
- **Mechanism.** In graph signal processing `Â` is a low-pass filter on the graph spectrum.
  SGC drops the per-layer nonlinearity and just applies `Âᵏ`; APPNP adds
  personalized-PageRank teleport. The `weighting` knob selects which filter.
- **Rule.** See the operator, not the brand. "Propagation," "SGC," "APPNP," "neighbor mean"
  are the same object at different altitudes; the method choice is a filter choice.
- **Evidence.** cookbook bridge chapters; S12; `04-predict` bridge note.

### G2 · An undirected edge is a *set* member, not a multiset member
- **Claim.** Declaring both directions of an undirected edge must not double its degree.
- **Predict.** An undirected edge list contains both `(a,b)` and `(b,a)`. Each node's degree
  is computed as ____ ?
- **Naive.** Every declared edge counts.
- **Happens.** Double-counted — degree inflated 2×, so the degree-normalized average was
  wrong. Fix: collapse to the same unordered-edge *set* the engine's other graph operators
  use.
- **Mechanism.** Undirected adjacency is a set of unordered pairs; treating a directed
  declaration as a multiset violates the object's definition.
- **Rule.** Pin the mathematical object (set vs multiset, directed vs undirected) and enforce
  it once, centrally — don't let the input representation leak into the count.
- **Evidence.** engine 0.26.1 (`graph_propagation.rs`).

### G3 · Reproducible float numerics require a fixed fold order — it's a design choice
- **Claim.** Byte-identical propagation comes from fixing the reduction order, not from luck.
- **Predict.** Run `propagate_embeddings` twice on the same input. Identical bytes?
- **Naive.** Floats → expect tiny run-to-run drift.
- **Happens.** Byte-identical — fixed `f64` fold order, no seeding. The deterministic end of
  the propagate↔learn spectrum.
- **Mechanism.** Floating-point addition is non-associative, so the *order* of reduction
  determines the bits; reproducibility requires fixing it.
- **Rule.** For reproducible numerics, fix the fold order explicitly. Determinism is
  engineered, not default.
- **Evidence.** cookbook bridge; S12.

---

# IV · Conformal prediction & uncertainty

### C1 · Conformal's value is the *proof*, not the score — and the (n+1) correction is exact
- **Claim.** Split conformal gives finite-sample, distribution-free coverage ≥ 1−α.
- **Predict.** Split conformal at α=0.1 on `n` calibration points. The coverage guarantee is ____ ?
- **Naive.** "Approximately 90%, asymptotically."
- **Happens.** Provably ≥ 1−α at *finite* n, distribution-free, using the exact quantile
  `⌈(n+1)(1−α)⌉/n` — validated by 28 numpy-first oracle/Monte-Carlo tests and an independent
  `engine == manual` reproduction of coverage.
- **Mechanism.** Under exchangeability, the rank of the test score among the calibration
  scores is uniform on `{1,…,n+1}`; thresholding at the `⌈(n+1)(1−α)⌉`-th score gives an exact
  finite-sample bound — no asymptotics.
- **Rule.** Use conformal for the guarantee, not as a score transform; keep the `(n+1)`
  correction — rounding it away breaks the finite-sample bound.
- **Evidence.** cookbook `08-conformal`; `jammi-numerics`; the 28-oracle suite.

### C2 · Weighted conformal repairs a *score-moving* shift, not a *location* shift
- **Claim.** Importance weighting can only fix a shift it can re-weight along the score axis.
- **Predict.** Marginal sets under-cover after a temporal shift (0.830 vs nominal 0.90). You
  apply importance-weighted conformal. Coverage → ____ ?
- **Naive.** Weighted/Mondrian conformal is *the* covariate-shift fix → coverage restored.
- **Happens.** A **no-op** for the regression target; the kNN-weighted classification scheme
  closes ~⅔ of the gap but nothing reaches nominal. The shift was **location/orthogonal**,
  not score-reweighting.
- **Mechanism.** Weighting re-scales the *calibration scores* by a likelihood ratio — it can
  only repair a shift that changes *which scores are likely*. A shift that translates the
  target leaves the conditional score distribution roughly fixed and merely relocates it;
  there is no axis for the weight to pull on. That needs a governed, time-aware *cohort*.
- **Rule.** Diagnose the *geometry* of the shift first. Score-moving → weight it.
  Location/orthogonal → re-cohort it. A weight is not a universal shift fix.
- **Evidence.** cookbook `04-predict` tier-04 + `08-conformal`; the keystone's honest crux.

### C3 · A "coverage restored" number can be an admission-convention artifact
- **Claim.** Comparing coverage across implementations requires pinning the set-admission rule.
- **Predict.** Local APS coverage = 0.895, engine APS = 0.867. Which is correct?
- **Naive.** A real gap — or a bug in one of them.
- **Happens.** A *convention* difference: whether the class that *crosses* the cumulative-mass
  threshold is admitted. Aligning the local rule to the engine's exclude-crossing rule made
  `local == engine == 0.867` — the "0.867 → 0.895 restore" was the artifact.
- **Mechanism.** APS includes classes by descending softmax mass until the cumulative mass
  crosses `1−α`; including vs excluding the crossing class changes set size, hence coverage.
  Both are valid; they give different numbers.
- **Rule.** Pin the admission convention before comparing coverage across implementations; a
  "restore" can be an apples-to-oranges artifact.
- **Evidence.** cookbook keystone tier-04; `08-conformal`.

### C4 · Score the *distribution*, not the mean — proper scores + PIT
- **Claim.** Calibration is invisible to point-error metrics.
- **Predict.** How do you check that a *distributional* forecast is honest, not just accurate
  on average?
- **Naive.** MSE / accuracy of the predicted mean.
- **Happens.** Proper scoring rules (CRPS, NLL) plus a PIT histogram (flat ⇒ calibrated);
  the engine's values reproduced a manual computation exactly.
- **Mechanism.** A proper scoring rule is minimized *only* by the true predictive
  distribution; the PIT value `F(y)` is uniform iff the forecast is calibrated. Mean-error
  cannot see over/under-dispersion.
- **Rule.** Evaluate distributional forecasts with a proper rule + PIT, never with the point
  estimate alone.
- **Evidence.** cookbook `09-calibration`.

---

# V · The spine, made explicit

### S1 · One root cause wore four disguises — compute-past-the-valid-domain
- **Claim.** Negative LR, blown-up mean, doubled degree, and a globally-readable "isolated"
  source are the *same* bug.
- **Predict.** What do those four findings, across four subsystems, have in common?
- **Naive.** Four unrelated bugs.
- **Happens.** One: the engine evaluated a function *outside the domain where its output
  means anything*, and returned a confident wrong number instead of an error — a schedule
  past its horizon, a head outside its trained magnitude, a degree on a multiset, an identity
  never enforced.
- **Mechanism.** Confident software computes whatever its arithmetic permits; correctness
  needs an explicit domain and a guard at its edge.
- **Rule.** Validate / clamp / normalize at every numeric *and* catalog input edge, and add a
  boundary/degenerate oracle per operation. This single lens is the highest-leverage review
  habit the marathon produced.
- **Evidence.** the #44 adversarial sweep; `CASE-STUDIES.md` §2.

### S2 · Data isolation is a row predicate, not a registration boundary
- **Claim.** Hiding a source from a tenant's *listing* is not hiding its *rows*.
- **Predict.** You register a separate source per tenant. Tenant A queries tenant B's source
  *by name*. How many rows come back?
- **Naive.** Zero — per-tenant registration isolates the data.
- **Happens.** All of them. Source separation isolates the catalog *listing*; real row
  isolation needs a `tenant_id` discriminator column the analyzer filters on. A
  discriminator-*less* federated source is globally readable — the engine does not
  authenticate; that boundary lives above it.
- **Mechanism.** Catalog filtering controls *what you can see listed*; row visibility needs a
  predicate injected onto the scan. Different layers, different guarantees.
- **Rule.** Separate "is it listed for me" from "can I read its rows." Isolation is a row
  predicate; a "leak prevented by separate sources" claim is usually a pre-filter artifact
  (verify the *mechanism* produces the zero). Auth is above the engine.
- **Evidence.** cookbook `11-tenancy`; `tenant_scope.rs`; `CASE-STUDIES.md` §3.

---

## Closing

Read across the cards and the spine reappears every time: the bugs that mattered were not
crashes — they were *confident wrong answers* at the edge of a valid domain, and they were
only caught because a consumer asked for a real number and an independent check verified it.
The domains differ; the discipline is one. Validate the edge, measure the number, and trust
the proof over the intuition.
