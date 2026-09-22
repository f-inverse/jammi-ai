# 69 — The parity ladder

How the engine shows that a workload run through its whole stack — candle kernels, the trainer,
a DataFusion plan, a Ballista placement — is as fast, as small, and learns as well as the same
workload in PyTorch, and how much each layer of the stack costs.

## The problem with a single comparison

A single "jammi vs PyTorch" number answers one question at one depth of the stack. A step-level
kernel bench says nothing about the plan above it; an end-to-end number that regresses says
nothing about which layer regressed. And a speed number on its own can always be bought with a
different result: fewer rows served, a looser numeric, a slower convergence.

## One abstraction

A **workload** is a function from committed inputs to an artifact.

A **rung** is one implementation stack of a workload. Rungs are ordered so that adjacent rungs
differ by exactly one layer. A ladder has as many rungs as its workload has layers:

| workload | artifact | rungs, in order |
|---|---|---|
| `encode` | one vector per key | `torch` → `direct` (the loaded model called on the same texts, no plan) → `plan` (DataFusion, 1 partition) → `plan-partitioned` (N partitions) → `placed` (the same plan on a Ballista executor) → `shape-d` (the deployed topology: the serve through the query tier, on a compute process) |
| `train-step` | one optimizer step's cost over a synthetic batch, swept over shapes; on the `torch` edge, gradient agreement at shared weights | `torch` → `reference` (the engine with every fused-kernel family off) → `fused` |
| `train-run` | an adapter and a held-out loss trajectory, from a pair table | `torch` → `resident-reference` (the trainer over in-memory rows, reference kernels) → `resident` (the fused kernels) → `streamed` (the job path: training-set table, streaming loader) → `placed` (the same job as a gang on an executor) → `shape-d` (the deployed topology: the job through the query tier, on a compute process) |
| `graph-sample` | a pair table, from random walks over a graph | `torch` (PyTorch Geometric's node2vec walk sampler) → `sampler` (the engine's graph sampler) |
| `propagate` | one propagated vector per node | `torch` (exact propagation by sparse matrix product) → `torch-geometric` (PyG's propagation layer: the practical bar) → `plan` (the engine's propagation, 1 partition) → `plan-partitioned` → `placed` |
| `predictor-train-run` | a context predictor's weights and a held-out loss trajectory | `torch` → `in-process` → `placed` (the same training as a job, placed on an executor) → `shape-d` (the deployed topology: the job claimed by a compute process) |

A **composite workload is cut at its committed intermediate artifact**, so each ladder compares
one thing. Graph-supervised fine-tuning is `graph-sample` ∘ `train-run`: its training half is the
`train-run` ladder fed the pair table the sampler committed, in committed `_ordinal` order, on
both stacks. The pair table is one more input digest in that ladder's identity — no new
comparison. Sharing the intermediate artifact across stacks removes the sampler's randomness
from the training comparison instead of averaging over it.

A **leg** is one run of one rung: identity (what two legs must agree on to be comparable),
provenance (recorded, never compared), and measurements.

An **edge** is an adjacent pair of rungs. One operator, `compare(edge)`, is applied to every
edge and yields a verdict on three axes — **speed**, **space**, **outcome**. There is no
per-rung or per-producer comparison logic: a new layer in the engine is a new rung in a table,
not a new merger.

What differs between the two legs of an edge is one typed thing, a **`Difference`**: a
*framework* (the rung below runs in PyTorch), a *kernel arm* (the same engine with a set of
fused-kernel families off below and on above), a *layer* (one engine layer added above), or a
*revision* (the same rung built from another revision of the engine). The first three are
edges of a ladder; the fourth is the one edge made outside a ladder — `jammi-bench ladder
<workload> <legs> --revision <rung>` compares `<rung>@base` against `<rung>@revised`, two
builds of one rung. Its expected cost is 1, so it is judged against the rung's own noise band:
the wider of what one build's repeats measure against each other and what a *second build of
the base revision*, `<rung>@rebuilt`, measures against the base in the same session — the
edge's A/A null. Two builds of one sha differ by more than one build's repeats do (the
committed A/A runs show it), so the null is measured in-session, never carried over. A cost
outside the band in the slow direction fails; in the fast direction it is routed to
investigation, never assumed favourable. The same edge with the same revision on every side is
the instrument's own null.

A rung's **kernel arm** is data: the set of fused-kernel families it turns off (`KernelArm`
over `KernelFamily` — LayerNorm, the flash cascade, the memory-efficient cascade, the attention
block, RoPE, softmax, GeGLU, GELU-erf, the LoRA site, AdamW). The concrete
`JAMMI_KERNELS_DISABLE` value a producer passes is never typed into a script: `jammi-bench
kernel-arm --model-dir <checkpoint> --off <families>` (or `--all`) derives it as the arm's keys
∩ the keys one training step on that checkpoint consults — the checkpoint's admission
*census*, taken to a fixpoint over absorption (a step with nothing off, then a step with every
consulted absorbing family off, until no new key appears; each step its own process, because
the disable list is read once per process). A BERT-family checkpoint has no GeGLU or RoPE seam
to turn off and a ModernBERT one has no GELU-erf seam; the derived sets differ by exactly those
families. An arm that turns off a family whose absorber it leaves on (RoPE without the attention
block) is refused by name, since its key could never fire on the device. The how-well reference
arm is `{flash attention, AdamW}` off; the step-level reference arm is every family off; both go
through the one derivation, and each rung's premises then prove the arm from the leg's own
dispatch counters. The census is device-independent — every call site consults its key before
the device is looked at — and this was measured on an A100: the derived ModernBERT all-off set
(nine keys) and the how-well arm's two keys each fired exactly under a strict step, with no
CUDA-only key.

Because adjacent rungs differ by one layer, an edge's speed ratio *is* that layer's cost, and
the ratios telescope: the product of the edge ratios is the end-to-end ratio against PyTorch.
The ladder also measures the end-to-end pair directly, in a session of its own; a product that
disagrees with the direct ratio beyond their intervals is a measurement-bias finding (an
interaction between layers, or a leg measured under different conditions), and the comparator
refuses the verdict.

## Kinds of edge

**Exact edges — jammi to jammi.** The engine is deterministic: the same seed and the same
committed row order produce a byte-identical adapter; the same keyed input produces
byte-identical vectors at any partition count and on any executor. So on every edge inside the
engine, the outcome axis is *digest equality* — no statistic, no margin, no seeds. This is what
lets a result established once at the bottom of the ladder hold exactly at the top: equality
composes without accumulating tolerance. It is also a hermetic test, not a campaign: the exact
edges run on CPU over the committed tiny fixtures on every change.

**Cross-stack edges.** Two frameworks — or two kernel sets — cannot be byte-identical. A ladder
is its reference rung, then the rungs reached by cross-stack edges, then the rungs reached by
exact edges: no cross-stack edge ever sits above an exact one, so nothing above an equality
reintroduces a margin. What a cross-stack edge's outcome is depends on how much of the
randomness between the two stacks can be removed rather than averaged over:

1. **Paired by seed** (`train-run`, `predictor-train-run`). For seed `s`, both sides start from
   the *same* initial tensors (jammi writes its seed-`s` initial adapter; the twin loads it),
   train with dropout off, over the same committed row order and the same batch partition. What
   remains is numerics. Both rungs are read at a point fixed before any leg runs: per seed, the
   epoch at which the *lower* rung's held-out loss is lowest — where the reference learned the
   most, never the final epoch of a run that may already be overfitting. The outcome is the
   paired difference `d_s = upper_s − lower_s` of held-out loss there, and two questions are
   asked of it:

   - *Is there a directional difference?* The exact two-sided sign test from `jammi-numerics`.
     The rule is stated for a fixed seed count `n` and level `α` (12 and 0.0064); the count of
     seeds that must share a sign is the sign test's own critical count for `(n, α)` — 11 of 12
     — and is taken over all `n` seeds, so a seed that ties is a seed that did not concord. A
     direction is declared only when the mean difference agrees with it. Any premise-clean seed
     count other than `n` is refused, never rescaled.
   - *Is the upper rung no worse?* Absence of a detected difference is not evidence of
     anything. The claim the ladder makes is **non-inferiority** — "on par with, if not better
     than" — and it is one-sided: the `1 − 2α` bootstrap interval of the mean paired difference
     must have its *upper* bound below `+δ`. The margin is not chosen: it is a fraction of the
     learning effect the reference rung itself establishes in the same session. Every reference
     leg records its held-out loss at the untrained model (`held_out_at_init`); the reference's
     improvement from there to the judged point, averaged over the seeds and lower-bounded by
     its own `1 − 2α` interval, is the established effect `M1`, and `δ = M1 / 2` — the
     construction of a margin `M2` as a fraction of the active control's established effect
     `M1` in FDA, *Non-Inferiority Clinical Trials to Establish Effectiveness* (2016), §III,
     one half being its worked example. Assay sensitivity is the same guidance's demand that
     the control's effect be shown in the trial at hand: a reference whose lower bound is not
     positive establishes no effect, so there is no margin to derive and the edge is refused
     (`AssayInsensitive`), as is a reference that never recorded its untrained loss. A lower
     bound far below `−δ` — the upper rung *better* by more than the margin — is not evidence
     against the claim. Two-sided **equivalence** (the interval inside `±δ`, Lakens' two
     one-sided tests) is the same interval read at both ends; it is reported beside the claim as
     evidence, never in its place.

   A detected degradation fails the edge; a detected *improvement* fails it for investigation
   (an anomaly is investigated, not celebrated) — it is non-inferior by construction, and it is
   still a finding. Every one-sided rule on every axis reads the same way: a bound on the
   unfavourable side only, with a failure in the favourable direction routed to investigation
   rather than counted as a pass or a fail.

2. **Paired by row** (`encode`, `propagate`). Deterministic math on the same keyed input needs
   no seeds: one vector per row on each side, and one allowance for both workloads — a relative
   perturbation of `√ε`, where `ε` is the compute dtype's machine epsilon (a well-conditioned
   computation agrees with its twin to half its fraction bits). The allowance is read through a
   metric chosen per edge: cosine `≥ 1 − ε/2` where vectors are consumed by similarity
   (`encode`), relative error `≤ √ε` where they are consumed as values (`propagate`).

3. **Paired by tensor** (`train-step`, the `torch` edge). One forward and backward on each
   stack from the same loaded adapter over the same synthetic batch, filed as the edge's
   `grads` take (`jammi-bench grad-oracle` and its torch twin). Per trainable tensor, the two
   sides' weights must be the same bits — a premise, not a tolerance; both gradients zero is
   vacuous (`dL/dA` is structurally zero at a zero `B`, and says nothing either way); exactly
   one zero, a non-finite entry, or a tensor one side lacks breaks the structure and fails the
   edge; a real pair's cosine is held to the `gradient_cosine_floor` budget, evidence until an
   artifact measures one. A gradient leg shares the edge's identity fields with its timed
   repeats and is free to differ on `warmup`, `steps_measured` and `max_grad_norm` alone.

4. **Law** (`graph-sample`). Two samplers with different random streams cannot be paired or
   digest-compared at all. Each rung's output is instead tested against the workload's
   *analytic ground truth*: the empirical second-order walk transition counts against node2vec's
   exact `p`/`q` transition probabilities on the fixture graph, by a likelihood-ratio
   goodness-of-fit test with its exact degrees of freedom (`Σ (categories − 1)`: the law is
   given, no parameter is fitted) at a pre-fixed `α = 0.001`. Both rungs must fit. The law is a
   committed artifact beside the fixture — `<unit>.json`, `{"cells": [[probability, …], …]}` —
   whose sha256 is an identity field of every leg, so the ground truth is never a producer's
   claim.

**Premises.** A leg's file name claims a rung; the rung's premises check the counted fact behind
the claim before any number is read. For `train-run`: the schedule is constant; variable-length
rows took the padded transport; the train-side probe (anchored at the untrained model) moved by
more than its floor, in the declared direction; the held-out tie fraction is under its cap; the
leg states its arm; and the dispatch counters prove the arm — a `resident` leg ran the fused
kernels and the flash cascade itself, a `resident-reference` leg shows a counted fallback behind
each kernel it disabled. A unit whose legs fail a premise is measured and reported, never
counted. Identity must agree between the two legs of a unit *and* across every leg entering the
comparison (all but the swept field), or two halves of a sweep run under different premises
would be averaged as one experiment.

**Controls and mutants** prove the rules can fail. The kernel edge carries an `lr = 0` control
at two seeds on both rungs: a run that cannot learn must *fail* the learning premise, and a
control that learns, was never run at `lr = 0`, or is missing (unless the operator waives it,
which the verdict records) refuses the edge. Same-seed repeats measure the outcome's own noise:
a repeat further from its first run than the seeds' differences are from each other refuses
the edge. A **mutant column** is the engine with one deliberate defect patched in, its legs
standing in for the edge's upper rung and judged by the same operator under the same rules
against the same lower legs. The signed `eps` family (optimizer update scaled by `1 + eps`) is
read as a dose ladder — the adjacent deflating doses that straddle detection are the
instrument's sensitivity; a deflating dose that *improves* is an anomaly; a `redproof-` mutant,
built to degrade outright, must be detected as a degradation. A mutant column judges direction
alone: it has no controls of its own and no margin to keep.

## The three axes

**Speed.** Every leg carries its per-iteration time series, never only a summary. The
comparator:

- (a) refuses a leg whose series is not stationary after warmup: a Mann-Kendall trend test
  rejecting at 0.01 *and* a Theil-Sen drift over the series above 2% of its median.
  Significance alone would refuse any long series for a drift too small to matter; drift alone
  would refuse a short noisy one for a slope it cannot resolve;
- (b) estimates each leg's location by the **minimum** — timing noise is one-sided, so the
  minimum is the robust estimator of the undisturbed run — and reports the median beside it;
- (c) puts the interval on the ratio of **medians**. A minimum cannot carry a bootstrap
  interval: a resample's minimum is the sample's minimum with probability `→ 1 − 1/e` and is
  never below it, so its bootstrap distribution is a spike whose percentile interval's lower
  bound is the observed value whatever the true floor is. The median is a smooth functional the
  bootstrap is valid for, and because an edge's legs are interleaved in one session its
  sensitivity to machine noise is shared by both sides of the ratio. Series are resampled in
  circular blocks of length `⌈n^⅓⌉`, each leg on its own, so serial dependence between
  iterations widens the interval instead of being resampled away. Across the units of a sweep
  the summary is the geometric mean of per-unit ratios, recomputed whole in every resample, so
  the interval belongs to the number printed beside it;
- (d) judges the interval against a rule fixed in the ladder's definition:
  - a cross-stack edge is a **non-inferiority** claim: the lower bound of `lower ÷ upper` must
    exceed the bar;
  - an exact edge is an **overhead budget**: the upper bound of `upper ÷ lower` must be under
    the layer's budget;
  - a revision edge must lie **inside the rung's own noise band**.

Legs of an edge are interleaved (A, B, B, A) on one device in one session, and a rung's repeats
are compared with each other to measure the ratio's own noise band; a ratio inside that band is
**INDETERMINATE** — reported as indistinguishable from 1, whatever its point value — and on a
revision edge that is the pass.

A layer's cost has a shape, not just a size. With a size sweep (three sizes or more), each rung
is fitted as `time = fixed + per_work · work` — `work` is rows for `encode`, edges × hops for
`propagate`, edges for `graph-sample` — from each size's fastest iteration; a fit whose residual
exceeds 10% of the mean time is refused rather than read. The two coefficients are budgeted
separately, because a fixed millisecond matters to a 16-row serving call and not to a
million-row batch, and a per-row nanosecond is the reverse. Both budgets are dimensionless, so
neither depends on the box: `per_work` as the ratio `upper ÷ lower`, and `fixed` as the layer's
added fixed cost divided by the lower rung's per-work cost — how many units of work the layer's
constant overhead is worth.

The telescoping check multiplies the edge intervals' bounds — at least as wide as the product's
own interval, so two intervals it finds disjoint are disjoint — and compares with the interval of
the end-to-end pair measured in its own session.

For the training workloads, speed and outcome are also fused into one number that cannot be
gamed: **time-to-quality**, the training wall time at which the held-out loss first comes within
`δ` of the lower rung's own lowest loss. A stack that steps faster but converges slower loses on
it.

**Space.** One instrument per quantity, the same for every rung including PyTorch: peak host
memory is the kernel's high-water mark for the process; peak device memory is one external
whole-device sampler wrapped around every leg. A framework's own allocator counters are
provenance. Where memory should not grow with input — the streaming loader exists so that host
memory is flat in the number of training rows — the quantity is the fitted slope over a size
sweep, and the budget is on the slope (16 bytes per row for `streamed`: an offset, never the
row).

**Outcome.** Digest equality on exact edges; the paired tests, gradient agreement or the law on
cross-stack edges.

## Verdicts

Every rule is **hard** or **evidence**. A failed hard rule fails the run; a failed evidence rule
is reported beside it. A hard rule with nothing to measure is refused, so a producer that stops
emitting a series cannot turn a verdict green. Every judgement carries the *direction* the upper
rung moved when it failed: a hard failure in the favourable direction — a detected improvement
in held-out loss, a revised build faster than its own noise band — makes the run
`RED_FOR_INVESTIGATION` rather than `RED`; better is investigated, never counted as a pass or
a fail. Every reason not to give a verdict is one typed refusal — identity absent or
disagreeing, a missing or unreadable leg, a leg filed under a rung or a control nobody
declares, a violated premise, too few samples, a non-stationary series, a digest mismatch, a
wrong seed count, a reference that establishes no learning effect, a hard rule with no measured
budget, a control that did not behave as one, a repeat outside the seeds' spread, a fit that is
not a line, malformed vectors, an unusable law, an invalid mutant column, a telescoping
contradiction — and any refusal makes the run `INVALID`. Otherwise the status is `GREEN`, `RED`,
or `RED_FOR_INVESTIGATION`.

### Budgets are measured or absent

No bound in the definition is invented. Every budget — a speed bar, a layer overhead, a memory
ratio, a per-work ratio, a fixed-cost work equivalent, a bytes-per-row slope, a gradient cosine
floor — is a measured value a committed artifact supplies through `crates/jammi-bench/budgets.json`,
one entry per `(workload, edge, rule)` naming the artifact its bound was read from. A rule with
no entry is reported **unbudgeted**: its judgement carries neither pass nor fail and the bound
it prints is `UNBUDGETED`; a hard rule left unbudgeted is a refusal. Every judgement names where
its bound came from — a fixed significance level, a value derived in the run (the margin from
the reference's effect, a noise band from a rung's repeats, the `√ε` row allowance), a measured
budget with its artifact, or none. The table is empty today: what decides a verdict is what the
run itself establishes — the seeded outcome against its derived margin, digest equality, the
law, gradient structure, and a revision's own in-session noise band.

## What is judged where

| where | what runs | verdict |
|---|---|---|
| every change (hermetic, CPU) | exact edges over the tiny fixtures: outcome digests equal across `direct`/`resident` … `placed` (`--axes outcome`); the kernel-arm derivation on the tiny BERT and ModernBERT fixtures; the committed campaigns and sweeps as oracles | hard |
| nightly (hosted CPU) | exact-edge overhead ratios, fixed and per-work. Both legs of a ratio run interleaved in one process on one box, so the box's speed cancels and no absolute rate is committed | evidence until measured |
| on demand (one GPU, one session) | the full ladder including the `torch` rung, at the shapes the performance guide reports (`ci/scripts/perf/finetune_step_ab.sh` for `train-step`); the kernel edge of `train-run` with its control and mutant columns (the how-well decision, `finetune_run_ab.sh`); the `encode` revision edge of this checkout against its merge-base with a rebuilt base as the A/A null (`gpu_inference_ab.sh`) | the `torch` edges are evidence; the kernel edge's outcome rules and the revision edge's band are hard; committed as an artifact |

The `torch` rung never decides a merge: PyTorch is not on the CI image, and a reference that
moves with every wheel release cannot be a merge condition. It is rebuilt on the box it is
measured on, every time.

## Where it lives

- Rungs are producers: `jammi-bench` tiers for the jammi rungs, scripts under
  `crates/jammi-bench/reference/` for the `torch` rung. A producer emits legs and decides
  nothing.
- The ladder — workloads, rungs, each edge's kind, rules and budgets — is one typed definition,
  `crates/jammi-bench/src/ladder/definition.rs`. `compare` is one function over it
  (`ladder/compare.rs`), using the statistics in `jammi-numerics::stats` (exact sign test and
  its critical count, the paired margin tests, circular block bootstrap, Mann-Kendall / Theil-Sen,
  least-squares line, multinomial goodness of fit, geometric mean); every refusal is a variant
  of one typed error (`ladder/refusal.rs`).
- `jammi-bench ladder <workload> <legs-dir> [--from RUNG] [--to RUNG | --revision RUNG] [--axes
  outcome,speed,space,shape] [--mutant LABEL:PATCH_SHA256]… [--waive-control] [--law-dir DIR]
  [--out DIR]` emits one JSON verdict (`ladder_verdict.json`) and a table, and exits non-zero on
  a refusal or a failed hard rule. `jammi-bench kernel-arm --model-dir DIR [--target-modules
  SITES] (--off FAMILIES | --all) [--json]` derives an arm's `JAMMI_KERNELS_DISABLE` value from
  the checkpoint's census.
- Producers are shell scripts that run legs in a balanced order and call the ladder:
  `ci/scripts/perf/finetune_step_ab.sh` (`train-step`), `finetune_run_ab.sh` (the kernel edge
  of `train-run`), `gpu_inference_ab.sh` (the `encode` revision edge), `encode_ab.sh` (the
  `encode` rung's replicate check: the revision edge with one build on every side).

### The leg contract

A leg is one type, `crates/jammi-bench/src/leg.rs`'s `Leg<P>`, which every `jammi-bench`
producer fills and the comparator reads as `Leg<Fields>`. It is four parts, serialized flat
under `tiers.<key>` in a `jammi-bench` report or `<key>` at the top level of any other
producer's JSON (`<key>` one of `encode_step`, `finetune_step`, `finetune_run`, `graph_sample`,
`propagate`, `predictor_train_run`):

| part | what it holds |
|---|---|
| the **payload** `P` | the workload's own fields; its identity is declared once, in `Payload::IDENTITY_FIELDS` (the `Payload` impl of `TrainStepPayload`, `TrainRunPayload`, `EncodePayload`; the list in `ladder/definition.rs` for the workloads without a `jammi-bench` producer). Another framework's leg is held to the same declaration. Numbers agree as numbers; everything else agrees as written |
| **provenance** | recorded, never compared: `device_name`, `build_features`, `flash_compiled`, `kernels_disabled_requested`, `kernels_disabled_fired`, `arm`, `attention_arm`, `mutant_id`/`mutant_base_sha`/`mutant_patch_sha256`. Absent on a leg another framework produced |
| **measured** | what every axis reads: `iter_wall_s` (post-warmup wall seconds per timed iteration, in run order; a training run's iteration is its optimizer step, every epoch's `step_walls` in order), `work` (the size the cost scales with), `peak_rss_bytes` and `peak_vram_bytes` (a number, or `{value, unit}` with `value: null` for not measured; every rung uses the same two instruments — the kernel's high-water mark and one whole-device sampler), `outcome_digest`, `held_out_example_mean`, `trajectory[].{epoch, held_out_mean, train_wall_s}`, `vectors_file` + `vector_dim`, `law_observed` |
| **facts** | what the rung premises read: `train_probe_series`, `admission_is_dense`, `tie_fraction`, and the dispatch counters, `<base>_fused_dispatches` with `<base>_eager_dispatches` (`_declined_dispatches` for the flash cascade) for every counted family, read as a whole or not at all |

A leg file is named `<rung>__<unit>__<take>.json`; the legs of the directly measured
end-to-end pair live in the legs directory's `direct/` subdirectory; a revision edge's legs are
filed under `<rung>@base`, `<rung>@revised` and `<rung>@rebuilt`. `unit` is one point of the
sweep (`seed3`, `rows4096`, `b8s128d0`); `take` is `r1`, `r2`, … for measured repeats (`r1`
carries the outcome, every repeat carries time and memory) or a control's tag (`lr0`).

## Sources of the method

- T. Hoefler, R. Belli, *Scientific Benchmarking of Parallel Computing Systems* (SC '15):
  report distributions and variability, summarize ratios with care, model performance instead
  of tabulating it.
- T. Kalibera, R. Jones, *Rigorous Benchmarking in Reasonable Time* (ISMM '13): the result of
  a comparison is a ratio with an interval.
- J. Chen, J. Revels, *Robust Benchmarking in Noisy Environments* (2016): the minimum as the
  location estimator under one-sided timing noise.
- T. Mytkowicz et al., *Producing Wrong Data Without Doing Anything Obviously Wrong!*
  (ASPLOS '09): measurement bias from setup; interleaving and same-session legs.
- H. Künsch, *The Jackknife and the Bootstrap for General Stationary Observations* (1989);
  D. Politis, J. Romano, *A Circular Block-Resampling Procedure for Stationary Data* (1992):
  resampling blocks keeps serial dependence.
- X. Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*
  (MLSys '21): name every source of variation; here they are removed by pairing rather than
  averaged over.
- D. Lakens, *Equivalence Tests* (2017): a margin claim — non-inferiority or equivalence — has a
  pre-specified smallest effect of interest, and is never the failure of a difference test.
- A. Grover, J. Leskovec, *node2vec* (KDD '16): the second-order walk's transition
  probabilities, the law the samplers are tested against.
- MLCommons, *MLPerf Training Rules*: time-to-train to a quality target, and reference
  convergence points — speed never stands apart from convergence.
