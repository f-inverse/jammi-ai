# Train an In-Context Predictor (Amortized, Adapts Without Retraining)

An **in-context predictor** meta-learns to turn a *context set* — a target's
retrieved neighbours and their outcomes — into a predictive distribution, in one
forward pass with **no gradient update at inference**. Trained once over many
tasks, it adapts to a new target's neighbourhood the way a prior-fitted network
does: condition on the context, emit the posterior, move on. It is the
learned-aggregation point of the uncertainty spectrum, above the cheaper
distribution-free and parametric options.

Three curated architectures, selected by config (never authored as tensor ops):

- **CNP** — a DeepSets encoder that mean-pools the context, then a decoder MLP.
  The baseline; the learned twin of fixed pooling.
- **Attentive CNP** (`attncnp`) — attention pooling over the context, so the
  target query weights the neighbours that matter. The payoff over fixed
  pooling, and the member that **widens its uncertainty when its context is thin
  or unfamiliar** (epistemic uncertainty).
- **TNP** — a transformer over the `(context ∪ target)` token set; the strongest
  member, the prior-fitted-network point.

## When to reach for it — the spectrum

The substrate offers three honest-uncertainty tools. Pick the cheapest one that
covers your need:

| Tool | Mechanism | Training | Reach for it when |
|---|---|---|---|
| [Conformal](./conformal-prediction.md) | distribution-free coverage wrap | none (calibrate) | you need a guarantee over *any* model, audit-reproducible and deterministic |
| [Distributional head](./distributional-inference.md) | learned aleatoric distribution | fine-tune a head | continuous outcomes where a density or quantiles suffice |
| **In-context predictor** | meta-learned posterior over a context set | episodic meta-train | few-shot / adapt-per-target without retraining; you want **epistemic** uncertainty |

The in-context predictor is the heaviest and most expressive. It does **not**
replace the other two — an amortized posterior is sharp but **not automatically
calibrated**, so it is *wrapped by conformal* for a coverage guarantee (below).
Conformal remains the deterministic, audit-reproducible option; this predictor
is what a continual, adapt-per-target setting reaches for.

## Train

Training is **episodic**: each *task* (the distinct values of a task column — a
cohort, a time window, a source partition) is split into a context set and
held-out targets, and the target's outcome is scored under a proper objective.
Tasks — not points — are partitioned into train/test, so generalisation is
measured on **held-out tasks**. The target is never in its own context
(self-exclusion plus a same-task split), and a meta-dataset with too few tasks is
rejected rather than meta-trained into memorisation.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# extern crate jammi_encoders;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::context_predictor::{
#     ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
# };
# async fn ex(session: &Arc<InferenceSession>) -> jammi_db::error::Result<()> {
let spec = ContextPredictorTrainConfig {
    model_id: "patents-context-predictor".into(),
    architecture: ContextArchitecture::AttnCnp, // CNP | AttnCnp | TNP, by config
    key_column: "_row_id".into(),               // the per-row identity
    task_column: "cohort".into(),               // distinct values = the tasks
    value_column: "outcome".into(),             // the scalar y to regress
    context_k: 32,                              // retrieval / context size
    hidden_dim: 64,
    num_heads: 4,
    num_layers: 2,
    head: PredictiveHead::Gaussian {            // an S18 head + proper score
        objective: GaussianObjective::Crps,
    },
    epochs: 100,
    learning_rate: 0.005,
    grad_clip: 1.0,
    test_task_fraction: 0.2,                    // tasks held out for eval
    min_task_count: 4,                          // the meta-overfitting guard
    seed: 0,
};
// Training is a durable, lease-claimed job: `train_context_predictor` submits a
// queued job and returns a handle immediately; a worker claims it, re-samples
// the episodic meta-dataset from the spec, trains it, and registers the model.
let job = session.train_context_predictor("patents", &spec).await?;
job.wait().await?; // block until a worker drives the job to completion
let model_id = job.model_id(); // the spec's `model_id`, now registered
# let _ = model_id;
# Ok(()) }
```

The objective is one of the proper scores the
[distributional head](./distributional-inference.md) uses — no new loss code. A
`PredictiveHead::Gaussian` serves `(mean, std)`; a `PredictiveHead::Quantile`
serves a non-crossing set of quantile levels.

## Predict — adapt to a new target, no retraining

Predicting assembles the target's **live** context (the serving corpus, with the
target excluded) and runs one in-context forward. There is no optimizer and no
weight update — the adaptation lives entirely in the forward.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::context_predictor::{ContextServeOptions, PredictedDistribution};
# async fn ex(session: &Arc<InferenceSession>) -> jammi_db::error::Result<()> {
// Reload the trained predictor for inference, serving over a corpus.
// `ContextServeOptions::default()` is embedding-similarity (ANN) context with no
// serving split — pass an edge-bearing source to condition on declared edges.
let served = session
    .load_context_predictor(
        "patents-context-predictor",
        "patents",
        ContextServeOptions::default(),
    )
    .await?;

// One forward over the target's live context — no gradient update.
let dist = session
    .predict_with_context_predictor(&served, "US-7654321")
    .await?;
match dist {
    PredictedDistribution::Gaussian { mean, std } => {
        let _ = (mean, std);
    }
    PredictedDistribution::Quantile { levels } => {
        let _ = levels; // ascending (level, value) pairs
    }
}
# Ok(()) }
```

The serving source need not be the training source — a predictor meta-trained on
one corpus serves a target's neighbourhood in another of the same shape (the
inductive prior-fitted-network property). An optional split predicate scopes the
serving context.

## Wrap with conformal for a coverage guarantee

An amortized posterior is sharp but can be **overconfident off its training
tasks** — its raw interval under-covers. Calibrate a
[conformal](./conformal-prediction.md) wrap on a **held-out** calibration set
(tasks disjoint from training) and the served interval recovers its nominal
coverage. A Gaussian head wraps with absolute-residual conformal over its mean; a
quantile head wraps with CQR over its lower/upper quantiles.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::context_predictor::{ConformalLevers, ServedContextPredictor};
# async fn ex(
#     session: &Arc<InferenceSession>,
#     served: &ServedContextPredictor,
#     held_out: &[(String, f64)], // (target_key, observed y), tasks disjoint from training
# ) -> jammi_db::error::Result<()> {
// Calibrate at 90% nominal coverage on the held-out set. `Marginal` is plain
// split-conformal; governance may instead supply a Mondrian cohort or weights.
let wrap = session
    .calibrate_context_predictor_conformal(served, held_out, 0.1, ConformalLevers::Marginal)
    .await?;

// Serving: turn a prediction into a coverage-guaranteed interval. The optional
// group is the test point's Mondrian cohort (`None` for a marginal wrap).
let dist = session
    .predict_with_context_predictor(served, "US-7654321")
    .await?;
let (lower, upper) = wrap.interval(&dist, None)?;
# let _ = (lower, upper);
# Ok(()) }
```

## Epistemic uncertainty — and its honest caveat

The attentive members (`attncnp`, `tnp`) **widen** their predicted uncertainty
when a target's context is sparse or unfamiliar — the property a fixed
distributional head lacks. This is primarily an attention property: a plain CNP's
mean-pool barely widens its σ as the context thins (it conditions on the context
*size* rather than reweighting members), so reach for `attncnp` or `tnp` when
epistemic widening matters. If you only need aleatoric noise on a continuous
outcome, the cheaper [distributional head](./distributional-inference.md) is the
right tool.

## From Python

```python
# Submits a durable training job and returns its handle; an embedded worker
# runs it. Block on `.wait()`, then read `.model_id` for the registered model.
job = db.train_context_predictor(
    "patents",
    key_column="_row_id",
    task_column="cohort",
    value_column="outcome",
    architecture="attncnp",   # "cnp" | "attncnp" | "tnp"
    output="gaussian",        # or "quantile" with levels=[0.1, 0.5, 0.9]
    objective="crps",         # "crps" | "nll" | "betanll"
    context_k=32,
)
job.wait()
model_id = job.model_id

dist = db.predict_with_context_predictor(
    model_id, source="patents", target_key="US-7654321"
)
# {"kind": "gaussian", "mean": ..., "std": ...}
```
