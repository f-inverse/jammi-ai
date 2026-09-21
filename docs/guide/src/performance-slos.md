# Performance SLOs

Jammi's performance contract is **throughput and coverage, gated against
committed baselines** — not latency. Each scale-relevant engine verb commits a
measured rate on a named reference box (or, for the recall tier, a recall
fraction gated against a committed floor; or, for the serving verbs, the
serving plan's cost as a same-process ratio that needs no reference box), and
a regression gate fails when a fresh run falls more than a fixed fraction
below it. This page is the operator's reference for every gated
target: the verb, the named scale it is measured at, the committed baseline, the
relative-drop threshold, and the box the baseline was emitted on.

## How the gate works

A measured rate must not fall more than the **relative-drop threshold** below
its committed baseline. The threshold derives an absolute floor from the
baseline — `floor = baseline · (1 − threshold)` — and the gate is a `>=` against
that floor (`measured >= floor`), never an equality and never a bit-compare. The
single threshold is **30%** (`DEFAULT_REGRESSION_THRESHOLD`), defined once in the
harness. It is generous on purpose: the load-bearing failure this gate exists to
catch is a *structural* regression — an algorithm that went quadratic, a lock
that serialized a parallel path, a dropped fast path — which collapses
throughput by far more than a third. A tighter threshold would trade that real
signal for false alarms on runner noise.

A gated **cost** — lower is better, the committed number a budget — is the same
gate over the reciprocals: the cost `c` is the rate `1/c`, so it fails when
`measured > budget / (1 − threshold)`.

The gate **fails closed**: a non-finite or non-positive baseline cannot anchor a
relative gate, so it fails (it never vacuously passes against a meaningless
baseline). Each `*-scale` bench subcommand maps its verdict to its process exit
code — a regression exits non-zero — which is what the CI lanes assert.

## Where the gate runs

| Lane | Trigger | Blocking? | Purpose |
|------|---------|-----------|---------|
| `ci.yml` (workspace tests) | every PR | **yes** | Gates a *property* of the mechanism: `committed_baseline_gates_with_teeth` proves the committed baseline is a well-formed, generously-thresholded gate that can fail. It does **not** re-measure the rate on the contended PR runner. |
| `perf.yml` | nightly `schedule:` + `workflow_dispatch` | no (early-warning) | Runs every `*-scale` tier's measured-rate gate on a real box, so a structural regression surfaces between releases. Non-blocking because the 30% band was sized for a same-box manual emit, not a contended shared runner — a required per-PR rate gate would flap and rot. |
| `crates.yml` (`perf-gate`) | `v*` release tag | **yes** | The authoritative same-box-ish gate: `publish` depends on it, so a structural perf regression on the release tag blocks the crates.io publish and the GitHub release. |

## The gated targets

Each row is one gated verb at one named scale. The rates are **same-box
throughputs**; the recall row is a **fraction gated by an inequality** — the
`measured >= floor` check is meaningful on any box, but the fraction itself is
bit-for-bit only on the same box (see the same-box caveat). Every committed number
is a real, re-derivable fold — a `rebuild-*` bench subcommand reproduces it on
the emit box.

| Verb | Bench tier | Named scale | Committed baseline | Threshold | Gated quantity |
|------|-----------|-------------|--------------------|-----------|----------------|
| `fine_tune` | `train-scale` | 1 536 in-batch-negative pairs, one GradCache backward + AdamW step, `Device::Cpu` | 180.0 pairs/s | 30% rel. drop | throughput (pairs/s) |
| `fine_tune_graph` | `graph-train-scale` | 8 communities × 64 nodes, biased-walk sampler (walk length 4, 4 walks/node) | 6 418.1 pairs/s | 30% rel. drop | sampled-pairs/s throughput (+ a portable determinism digest) |
| `train_context_predictor` | `context-predictor-scale` | CNP over 8 tasks × 18 rows, 30 epochs | 21.29 episode-steps/s | 30% rel. drop | meta-training throughput (+ a same-box predict digest) |
| `generate_embeddings` | `model-inference-scale` | 16 / 128 / 1 024 rows over a tiny 32-dim 1-layer BERT bundle, `Device::Cpu`, plan and direct legs interleaved | budgets in `baselines/model_inference.json` (`embed_overhead`) | 30% rel. drop of the reciprocal | **serving overhead** — two same-process ratios of the serving plan to the bare model, plus a two-term-shape check (+ a same-box embed digest); see below |
| `infer` (classification) | `model-inference-scale` | the same sweep over a tiny 32-dim 1-layer ModernBERT classifier bundle | budgets in `baselines/model_inference.json` (`infer_overhead`) | 30% rel. drop of the reciprocal | **serving overhead**, as above (+ a same-box infer digest) |
| `search` + `build_neighbor_graph` | `arxiv` | 2 000-row corpus slice, 100 held-out 768-dim queries (frozen sidecar) | recall@{1,10,100} = {1.0, 1.0, 0.997} | floor = measured − 0.04 (absolute margin) | **recall fraction** (not a rate) — `measured >= floor`, an inequality gate whose absolute margin absorbs cross-box float drift; the fraction is bit-for-bit only on the same box |

### The serving-overhead rows

A rows/s rate through a tiny model gates nothing on a box faster than the one
that committed it: the floor sits below anything the code could regress to. So
the two serving verbs do not commit a rate. `model-inference-scale` serves each
verb two ways in ONE process — through the engine's serving plan, and by calling
the loaded model directly over the same rows in the same `batch_size` chunks —
interleaved over a row sweep, fits each leg's fastest serves to
`serve_ms = fixed_ms + per_row_ms · rows`, and gates two dimensionless costs
against the committed budgets:

* `per_row_ratio = plan.per_row_ms / direct.per_row_ms` — what a row costs
  through the plan, in units of what the bare model charges for it. Lost
  batching, a per-row reload or a per-row copy moves it.
* `fixed_rows = plan.fixed_ms / direct.per_row_ms` — what one call costs before
  it serves a row, in rows of bare-model work. A plan that grew a per-call cost
  moves it.

Both legs ran on the same box in the same seconds, so the box's speed is in the
numerator and the denominator alike. The gate also fails when the plan's fit
stops describing its points (relative residual over 0.10): the serve is no
longer two-term over the sweep, which is what a cost that went superlinear
looks like. Before failing, it folds each leg's fastest serves over up to three
fresh-session sweeps — box interference only adds to a serve, a regression is in
every sweep.

Two limits, stated: the ratio cannot see a regression INSIDE the forward (it
slows both legs), which is measured where the forward is real — the GPU tiers
and the PyTorch reference (`crates/jammi-bench/reference/README.md`); and it
holds across boxes, not across thread postures, so the spec records the
`RAYON_NUM_THREADS` its budgets were measured under and the tier refuses to
run under another.

### The reference box

The committed rate baselines were emitted on this box, in the **release**
profile, with `RAYON_NUM_THREADS=1`:

| Property | Value |
|----------|-------|
| Logical CPUs | 8 |
| Total RAM | 31 720 MiB (~31 GiB) |
| Profile | `release` |
| Engine version when committed | `0.30.0` |

A baseline is refreshed by hand (via the tier's `rebuild-*` subcommand) when the
emit box changes; the version-stamped report lets a downstream gate reject a
cross-version comparison.

## The same-box caveat

A committed rate is **not a portable floor**. Stated verbatim from the gate's
own definition:

> A *rate* (throughput, QPS, pairs/s) is not portable the way the recall
> fraction is — it is a property of the box that produced it, so a committed rate
> baseline is a *same-box* reference, refreshed by hand when the emit box
> changes, not a number a different machine can re-derive.

(The serving-overhead rows are the exception by construction: they commit a
ratio of two legs measured on the running box, not a rate.)

What stays portable is the *shape* of the gate (a measured rate must not fall
more than a fixed fraction below the committed baseline; a measured recall must
not fall below the committed floor) — that is the sense of "portable" in the
quote above: the floor travels to another box, not the bits. Of the digests
above, only the `fine_tune_graph` sampled-pair-set checksum is portable
bit-for-bit: the pair selection is a seeded integer stream (its scalar `f64`
roulette arithmetic is neither contracted nor reordered by Rust) and the
checksum is an FNV-1a fold over the selected node-text bytes, so any box
re-derives it exactly. The **recall fraction** is not in that class — it is
scoped like the float digests. Recall-set membership is decided by an `f32`
cosine reduction (the exact oracle's `cosine_distance`, a sequential `f32`
accumulation over the dot product and norms), so the fraction is bit-for-bit
only on the same box; across boxes or architectures a near-tie can move a
neighbour in or out of the top-k, and the recall SLO is an inequality gate
(`measured >= floor`) whose absolute margin (0.04) absorbs that small float
drift — never a bit-for-bit equality. The predict/embed/infer digests fold an
`f32` forward, and an `f32` reduction is NOT bit-identical across CPUs
(SIMD/FMA contraction and BLAS reduction order differ by machine), so those
three are a same-box property: each is re-derived on the box that ran it, not
asserted equal across boxes. So the
rate rows above (not the serving-overhead rows) are meaningful only against the reference box; do not read
them as a throughput your hardware must hit. The release-tag gate is the
authoritative reading because it runs on a same-box-ish runner; the nightly lane
is early-warning, not a portable promise.

## Why no latency SLOs

The contract is throughput and coverage, not latency. A latency SLO on a shared
CI runner flaps — tail latency on a contended box is dominated by co-tenant load,
not by the engine's code path — so a latency gate would either flap (set tight)
or never bite (set loose), exactly the failure mode the relative-drop *rate*
threshold is designed around. Latency is therefore **out of scope** here. The
representative full-scale serving numbers (the GPU-model rates that latency would
ride on) are captured off-box in the cookbook's A/B split, not gated in CI.
