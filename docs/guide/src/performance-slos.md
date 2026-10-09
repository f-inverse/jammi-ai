# Performance SLOs

Jammi's performance contract is **throughput and coverage, gated against
committed baselines** — not latency. Each scale-relevant engine verb commits a
measured rate (or, for the recall tier, a recall fraction gated against a
committed floor) on a named reference box, and a regression gate fails when a
fresh run falls more than a fixed fraction below it. This page is the operator's reference for every gated
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
| `ci.yml` (`perf-gate`) | every push to `main` | **yes** | Runs every `*-scale` tier's measured-rate gate, and proves the gate bites (`ci/scripts/perf/check_rate_gate_bites.sh`). A structural regression reds `main`'s run, and every release requires that run green on its tree (every release workflow's `proof` job), so it blocks every publisher. `main` only: a hosted runner's throughput jitters with its neighbours, and a per-PR rate gate would flap; a red on `main` is measured again by re-running it. |

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
| `search` + `build_neighbor_graph` | `arxiv` | 2 000-row corpus slice, 100 held-out 768-dim queries (sidecar built by the engine under test) | recall@{1,10,100} = {1.0, 1.0, 0.997} | floor = measured − 0.04 (absolute margin) | **recall fraction** (not a rate) — `measured >= floor`, an inequality gate whose absolute margin absorbs cross-box float drift; the fraction is bit-for-bit only on the same box |

### The serving path is not a row here

`generate_embeddings` and `infer` are the `encode` workload, measured as a
ladder of rungs (`jammi-bench encode-step`: the loaded model called directly,
the serving plan at one partition, the plan at N) rather than as a committed
rate: a rows/s through a tiny model gates nothing on a box faster than the one
that committed it. Each layer's cost is the ratio of two legs measured
interleaved in one process on one box, judged by `jammi-bench ladder encode`
against a dimensionless budget. See `crates/jammi-bench/src/encode_step.rs`
and `crates/jammi-bench/reference/README.md`.

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
more than a fixed fraction below the committed baseline; a measured recall must
not fall below the committed floor) — that is the sense of "portable" in the
quote above: the floor travels to another box, not the bits. The **recall
fraction** is scoped like the float digests. Recall-set membership is decided by an `f32`
cosine reduction (the exact oracle's `cosine_distance`, a sequential `f32`
accumulation over the dot product and norms), so the fraction is bit-for-bit
only on the same box; across boxes or architectures a near-tie can move a
neighbour in or out of the top-k, and the recall SLO is an inequality gate
(`measured >= floor`) whose absolute margin (0.04) absorbs that small float
drift — never a bit-for-bit equality. An `f32` forward's digest — the `encode`
workload's `outcome_digest`, held equal across its rungs on one box — is a
same-box property: re-derived on the box that ran it, never asserted equal
across boxes. So the
rate rows above are meaningful only against the reference box; do not read
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

## The graph-learning workloads are measured as legs, not rate tiers

`fine_tune_graph`'s sampler, `propagate_embeddings` and
`train_context_predictor` carry no committed rate. A committed absolute rate is
a property of one box; what these three are compared on is a *ratio* between two
implementations of the same workload measured together on one box, which needs
no committed number at all. Each is a **leg producer**: a `jammi-bench`
subcommand (`graph-sample`, `propagate`, `predictor-train-run`) that runs the
engine's own code path and files what it measured as the ladder's leg — the
warm per-iteration time series, the process's peak resident set, and the
outcome (a digest and the file it digests, the walks' counts against the law,
a held-out trajectory) — with a PyTorch counterpart under
`crates/jammi-bench/reference/` that reads the same input files and files the
same fields. A producer judges nothing; `jammi-bench ladder <workload>` does.

What is asserted on every change, hermetically, is the part that is a property
of the code and not of the box:

| Workload | Held on every change |
|----------|----------------------|
| `graph-sample` | The pair table the sampler draws over the committed synthetic graph reproduces the committed digest **on any machine** (a seeded integer stream and a scalar `f64` roulette; the digest folds node text only), and a different seed or walk length moves it. Over many seeded walks every `(previous, current)` state's next-node frequencies sit within sampling error of node2vec's analytic transition law `π(x | t, v) ∝ α_pq(t, x) · w(v, x)`. |
| `propagate` | Two folds of one fixture agree bit for bit on the running box; so do `target_partitions = 1` and `4`; one hop fewer, one more, or a different `α` moves the digest. The output is `f32`, so no digest is committed. |
| `predictor-train-run` | The predictor's initial weights are a pure function of the run seed; the served predictor over the committed trained weights predicts the committed targets to the same bits twice on the running box, and a wrong `context_k` moves them. |
| graph fine-tune, training half | A resident fine-tune over the pair table `graph-pairs` writes and a `fine_tune_graph` job over the same graph, configuration and seed train the byte-identical adapter (at `lora_dropout = 0`; with dropout the job's pre-training acceleration probe has already consumed one mask draw per LoRA layer). |

