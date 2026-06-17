# Performance SLOs

Jammi's performance contract is **throughput and coverage, gated against
committed baselines** — not latency. Each scale-relevant engine verb commits a
measured rate (or, for the recall tier, a portable recall fraction) on a named
reference box, and a regression gate fails when a fresh run falls more than a
fixed fraction below it. This page is the operator's reference for every gated
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
throughputs**; the recall row is a **portable fraction**. Every committed number
is a real, re-derivable fold — a `rebuild-*` bench subcommand reproduces it on
the emit box.

| Verb | Bench tier | Named scale | Committed baseline | Threshold | Gated quantity |
|------|-----------|-------------|--------------------|-----------|----------------|
| `fine_tune` | `train-scale` | 1 536 in-batch-negative pairs, one GradCache backward + AdamW step, `Device::Cpu` | 180.0 pairs/s | 30% rel. drop | throughput (pairs/s) |
| `fine_tune_graph` | `graph-train-scale` | 8 communities × 64 nodes, biased-walk sampler (walk length 4, 4 walks/node) | 6 418.1 pairs/s | 30% rel. drop | sampled-pairs/s throughput (+ a portable determinism digest) |
| `train_context_predictor` | `context-predictor-scale` | CNP over 8 tasks × 18 rows, 30 epochs | 21.29 episode-steps/s | 30% rel. drop | meta-training throughput (+ a portable predict digest) |
| `generate_embeddings` | `model-inference-scale` | 16 rows over a tiny 32-dim 1-layer BERT bundle, `Device::Cpu` | 333.6 rows/s | 30% rel. drop | coarse serving throughput (+ a portable embed digest) |
| `infer` (classification) | `model-inference-scale` | 16 rows over a tiny 32-dim 1-layer ModernBERT classifier bundle, `Device::Cpu` | 207.0 rows/s | 30% rel. drop | coarse serving throughput (+ a portable infer digest) |
| `search` + `build_neighbor_graph` | `arxiv` | 2 000-row corpus slice, 100 held-out 768-dim queries (frozen sidecar) | recall@{1,10,100} = {1.0, 1.0, 0.997} | floor = measured − 0.04 (absolute margin) | **portable recall fraction** (not a rate) — `measured >= floor` |

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

What stays portable is the *shape* of the gate (a measured rate must not fall
more than a fixed fraction below the committed baseline) and the **determinism
digests** and the **recall fraction**, which any box re-derives bit-for-bit. So
the rate rows above are meaningful only against the reference box; do not read
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
