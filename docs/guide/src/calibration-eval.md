# Evaluate Uncertainty and Calibration

`eval_embeddings` and `eval_inference` answer *"is the prediction accurate?"*.
`eval_calibration` answers the orthogonal question — *"does the prediction know
what it doesn't know?"*. The two are independent: a model can be accurate and
badly calibrated, or perfectly calibrated and useless. When a predictor emits a
distribution or an interval, a point-accuracy metric cannot tell you whether
that uncertainty is *honest*. This harness can.

## What it reports

Every calibration eval reports three things together — reporting any one alone
is a trap:

- **A proper score (the headline).** CRPS (continuous ranked probability score)
  and NLL (negative log-likelihood). Strictly proper scores are uniquely
  minimised by the true distribution, so they reward calibration and sharpness
  *jointly* — the only safe headline metric.
- **A calibration diagnostic.** The adaptive, debiased PIT-calibration error:
  under calibration the probability-integral-transform of the outcomes is
  uniform, and this scores its departure from uniform. It is a diagnostic, never
  the verdict — reporting it alone admits the marginal-predictor degenerate (a
  model that predicts the global average is perfectly calibrated and worthless).
- **Sharpness and coverage.** The mean width of the nominal 90% interval and how
  often it actually contains the outcome. Sharper is better only at fixed
  coverage.

## The held-out, three-way-split contract

Calibration is measured on a **held-out test set** that is disjoint from both
the training data and any calibration set used to fit the predictor. Re-using
calibration points to *also* test inflates coverage — it is the single most
common conformal/calibration bug. The harness measures exactly the predictions
you give it; the split discipline is yours.

## Prepare a calibration golden set

A calibration golden set pairs a held-out predictive distribution with its
realised `outcome`. Two predictor shapes are supported, each reading different
columns.

### Parametric (Gaussian) predictor

For a predictor that emits a predictive `Normal(mean, sd)` per record:

```csv
record_id,mean,sd,outcome
r1,4.2,0.5,4.0
r2,1.1,0.9,2.3
r3,7.8,0.3,7.7
```

| Column | Type | Required |
|--------|------|----------|
| `record_id` | Utf8 | yes |
| `mean` | Float / Int | yes |
| `sd` | Float / Int (positive) | yes |
| `outcome` | Float / Int | yes |

### Ensemble (Sample) predictor

For a predictor that emits an ensemble of predictive draws per record, store the
draws as a JSON array in a `draws` column:

```csv
record_id,draws,outcome
r1,"[3.9, 4.1, 4.3, 4.0]",4.0
r2,"[0.8, 1.4, 1.0, 2.1]",2.3
```

| Column | Type | Required |
|--------|------|----------|
| `record_id` | Utf8 | yes |
| `draws` | Utf8 (JSON array of numbers) | yes |
| `outcome` | Float / Int | yes |

Register either as a source:

```python
db.add_source("calib", path="/data/calibration_holdout.csv", format="csv")
```

## Run the eval

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::eval::EvalCalibrationShape;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let report = session.eval_calibration(
    "patents",                          // source under test
    "calib.public.calibration_holdout", // held-out predictions + outcomes
    EvalCalibrationShape::Gaussian,     // or ::Sample for an ensemble
    &std::collections::HashMap::new(),  // no cohort tags
).await?;

// The proper score is the headline; the diagnostics explain it.
println!("CRPS (headline):  {}", report.aggregate.crps);
println!("NLL:              {}", report.aggregate.nll);
println!("PIT-calibration:  {}", report.aggregate.adaptive_ece);
println!("sharpness (90%):  {}", report.aggregate.sharpness);
println!("coverage (90%):   {}", report.aggregate.coverage);
# Ok(())
# }
```

### Python

```python
report = db.eval_calibration(
    "patents",
    "calib.public.calibration_holdout",
    shape="gaussian",   # or "sample"
)
print("CRPS:", report["aggregate"]["crps"])
print("coverage:", report["aggregate"]["coverage"])
```

## Slice by cohort

Marginal coverage hides conditional miscoverage: a predictor can hit 90%
coverage globally while systematically under-covering a subgroup. Tag records
with opaque cohort segments — keyed by `record_id` — and the report slices
coverage and CRPS per cohort, each with its sample size `n` and a bootstrap
confidence interval on the proper score, so a small cohort is not over-read.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# use std::collections::{BTreeMap, HashMap};
# use jammi_ai::session::InferenceSession;
# use jammi_ai::eval::EvalCalibrationShape;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let mut cohorts: HashMap<String, BTreeMap<String, String>> = HashMap::new();
cohorts.insert(
    "r1".to_string(),
    BTreeMap::from([("region".to_string(), "emea".to_string())]),
);

let report = session
    .eval_calibration(
        "patents",
        "calib.public.calibration_holdout",
        EvalCalibrationShape::Gaussian,
        &cohorts,
    )
    .await?;

for cohort in &report.per_cohort {
    println!(
        "{}={}: n={} coverage={} crps={}",
        cohort.key, cohort.value, cohort.n, cohort.coverage, cohort.crps
    );
}
# Ok(())
# }
```

## Compare two predictors with a p-value

The per-record scores are persisted to `_jammi_eval_per_query` keyed by the run
id, exactly like the embedding eval. Pairing the per-record CRPS of two runs by
`record_id` and running the same distribution-free paired significance test the
retrieval comparison uses turns *"B is better-calibrated than A"* into a CRPS
delta with a confidence interval and a p-value — not a vibe.

## Determinism

Given the same inputs the report is bit-for-bit reproducible: every scoring
function is deterministic and the only randomness — the cohort confidence-interval
bootstrap — runs under a pinned seed.
