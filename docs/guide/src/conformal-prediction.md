# Conformal Prediction: Distribution-Free Coverage

Conformal prediction wraps **any** existing predictor and turns a point output
into a **prediction set** (classification) or **interval** (regression) carrying
a finite-sample, distribution-free coverage guarantee. Given a held-out
calibration set, the marginal coverage of the emitted sets is at least
`1 − alpha` under exchangeability — for *any* underlying model, *any* data
distribution, and *any* sample size. No retraining: a calibration pass and an
empirical quantile. Deterministic given the calibration set, which is the audit
property.

The serving primitive lives in the open engine because a calibrated set is a
serving output — it must work with no license. The *operationalization* of the
guarantee — rolling coverage monitoring, coverage-SLA gating, and managed
recalibration under drift — is a governed concern provided by the
[Jammi platform](./introduction.md), built on this same primitive.

## The one assumption

The guarantee holds **if and only if** the calibration and serving data are
exchangeable. Under distribution drift it degrades silently. Two levers correct
for *known* structure:

- **Weighted conformal** applies importance weights for a known covariate shift.
- **Mondrian conformal** keeps a per-cohort quantile keyed on a group column,
  the principled approximation to conditional coverage (full per-input coverage
  is provably impossible distribution-free).

The primitive *applies* the weights and the grouping; *detecting* drift and
*choosing* the cohorts is governance, not a serving output.

## The three-way split

Reusing test points to calibrate inflates coverage. The calibration set must be
disjoint from **both** the training set and the test/serving data. The
calibration source is a distinct argument throughout the API.

## Classification: prediction sets

The classification scores read the per-class softmax mass the classifier already
emits.

- **LAC** — nonconformity `1 − p_y`; the smallest sets at the nominal level, but
  non-adaptive.
- **APS** (default) — the cumulative mass of classes ranked most- to
  least-probable up to the true class; set size *adapts* to input difficulty.
- **RAPS** — APS plus a tail-rank penalty that shrinks sets on easy inputs.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# fn ex() -> jammi_db::error::Result<()> {
use jammi_ai::predict::{ClassScore, ConformalModel};

// Held-out calibration: per-class probabilities + the realised class index.
let calibration: Vec<Vec<f64>> = vec![
    vec![0.7, 0.2, 0.1],
    vec![0.1, 0.8, 0.1],
    // ... one row per calibration example
];
let true_labels: Vec<usize> = vec![0, 1 /* ... */];

// Calibrate at 90% nominal coverage with the adaptive APS score.
let model = ConformalModel::classification(&calibration, &true_labels, ClassScore::Aps, 0.1)?;

// Serving: emit the prediction set for a new row of class probabilities.
let probabilities = vec![0.45, 0.4, 0.15];
let prediction_set = model.predict_set(&probabilities, None)?; // e.g. [0, 1]
# let _ = prediction_set;
# Ok(()) }
```

### Python

```python
sets = db.conformalize(
    calibration=[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],  # per-class probabilities
    true_labels=[0, 1],                               # realised class indices
    test=[[0.45, 0.4, 0.15]],                         # rows to predict
    alpha=0.1,
    score="aps",                                      # "lac" | "aps" | "raps"
)
# sets -> [[0, 1]]  (one list of admitted class indices per test row)
```

## Regression: prediction intervals

- **Absolute residual** — nonconformity `|y − ŷ|`; a constant-width interval
  `[ŷ − q̂, ŷ + q̂]`. Distribution-free but uninformative under
  heteroscedasticity.
- **CQR** (Conformalized Quantile Regression) — nonconformity
  `max(q_lo − y, y − q_hi)` over a predictor's lower/upper quantile estimates;
  an *adaptive*-width interval whose width tracks local uncertainty.

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# fn ex() -> jammi_db::error::Result<()> {
use jammi_ai::predict::{ConformalModel, IntervalScore};

// Absolute-residual conformal over held-out (prediction, observation) pairs.
let predictions = vec![1.0, 2.0, 3.0 /* ... */];
let observed = vec![1.2, 1.7, 3.1 /* ... */];
let model = ConformalModel::regression(
    &predictions,
    &[],   // lower quantiles (CQR only)
    &[],   // upper quantiles (CQR only)
    &observed,
    IntervalScore::AbsoluteResidual,
    0.1,
)?;

// Serving: a 90% interval around a new point estimate.
let (lower, upper) = model.predict_interval(2.5, 0.0, 0.0, None)?;
# let _ = (lower, upper);
# Ok(()) }
```

### Python

```python
# Constant-width absolute-residual intervals.
intervals = db.conformalize_interval(
    predictions=[1.0, 2.0, 3.0],   # calibration point estimates
    observed=[1.2, 1.7, 3.1],      # calibration targets
    test_predictions=[2.5],        # point estimates to bound
    alpha=0.1,
)
# intervals -> [(lower, upper)]

# Adaptive-width CQR intervals from quantile estimates.
intervals = db.conformalize_cqr(
    lower=[0.5, 1.5, 2.5],         # calibration lower-quantile estimates
    upper=[1.5, 2.5, 3.5],         # calibration upper-quantile estimates
    observed=[1.2, 1.7, 3.1],
    test_lower=[2.0],
    test_upper=[3.0],
    alpha=0.1,
)
```

## The finite-sample quantile

The conformal threshold is the `⌈(n+1)(1 − alpha)⌉`-th smallest calibration
score, **not** the naive `⌈n(1 − alpha)⌉` order statistic. The `(n + 1)`
correction is what makes the guarantee *exact* rather than merely asymptotic;
the naive quantile leaves a `~1/n` coverage gap and under-covers. When the
calibration set is too small for the requested level — fewer than
`⌈1/alpha⌉ − 1` points — the threshold is `+∞`: the honest, conservative answer
is "every label". A full set is a real signal that the input is hard or the base
model is miscalibrated, not a bug.

## The `conformal` evidence channel

Conformal outputs ride the evidence substrate exactly as `vector` and
`inference` do — one channel, four declared columns, no new provenance
machinery:

| Column | Type | Classification | Regression |
|--------|------|----------------|------------|
| `prediction_set` | Utf8 | JSON array of class ids | null |
| `lower` | Float64 | null | interval lower bound |
| `upper` | Float64 | null | interval upper bound |
| `alpha` | Float64 | nominal level | nominal level |

Register the channel once, then attach a contribution per result batch:

```rust,no_run
# extern crate jammi_ai;
# extern crate jammi_db;
# async fn ex(catalog: &jammi_db::catalog::Catalog) -> jammi_db::error::Result<()> {
use jammi_ai::evidence::conformal::{channel_spec, contribution, ConformalOutput};

catalog.channels().register(&channel_spec()?).await?;

let _contrib = contribution(&[
    ConformalOutput::Set { classes: vec![0, 2], alpha: 0.1 },
    ConformalOutput::Interval { lower: -1.0, upper: 1.0, alpha: 0.1 },
])?;
// `_contrib` merges into result batches via `merge_channels`.
# Ok(()) }
```

## Verifying coverage

The realised coverage and mean set size of a labelled batch are pure functions
in `jammi-numerics` — the same functions the platform's coverage monitor calls
on a rolling window:

```rust,no_run
# extern crate jammi_numerics;
# fn ex() -> Result<(), jammi_numerics::error::NumericsError> {
use jammi_numerics::calibration::{coverage, mean_set_size};

let hits = [true, true, false, true];     // did each set contain the true label?
let sizes = [2usize, 1, 3, 2];            // cardinality of each set
let realised = coverage(&hits)?;          // ~ 1 - alpha when calibrated
let efficiency = mean_set_size(&sizes)?;  // smaller is sharper
# let _ = (realised, efficiency);
# Ok(()) }
```

## What lives in the platform, not here

This is the serving primitive only. The governed layer — a rolling
realised-vs-nominal **coverage monitor** with drift detection and online
adaptation, a **coverage-SLA gate**, and **managed recalibration** under shift —
is provided by the Jammi platform. It consumes this primitive and the OSS
coverage function; it is not part of the open engine.
