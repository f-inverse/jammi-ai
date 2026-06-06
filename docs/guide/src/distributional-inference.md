# Distributional Inference: Predict a Distribution, Not a Point

A `ModelTask::Regression` head returns a **predictive distribution** per row —
a Gaussian `(predicted_mean, predicted_std)` or a set of quantiles — instead of
a single number. Where [conformal prediction](./conformal-prediction.md) wraps
*any* predictor with a distribution-free interval, a regression head is *trained*
to emit calibrated uncertainty directly, with proper-scoring objectives that make
that uncertainty honest.

Two output forms, both standard:

- **Parametric Gaussian** — the head predicts `μ` and a raw scale; serving maps
  the scale to a positive `σ = floor + softplus(raw)`. Smooth, cheap, a closed-
  form density. The default.
- **Quantile** — the head predicts a set of levels (e.g. `0.05, 0.5, 0.95`)
  directly. Distribution-free in shape, robust to non-Gaussian outcomes, and the
  input to conformal CQR. The serving adapter sorts each row's quantiles so they
  never cross.

## Choose the objective by your label

| Your label                          | Task / objective                          |
|-------------------------------------|-------------------------------------------|
| a continuous outcome + you want a density | `Regression`, β-NLL or CRPS (Gaussian) |
| a continuous outcome + you want robust intervals | `Regression`, pinball (quantile) |
| graded similarity scores            | embedding fine-tune, cosine-MSE / CoSENT  |
| ordered pairs / rankings            | embedding fine-tune, MNRL / triplet       |

The four regression objectives are all **proper scores** — minimising them
rewards a calibrated *distribution*, not merely an accurate mean. (MSE on the
predicted mean is *not* proper for a distribution and is only a secondary point-
accuracy diagnostic.)

- **β-NLL** *(default)* — Seitzer's variance-weighted Gaussian NLL. The plain
  joint `μ,σ²` NLL has a well-documented pathology: it down-weights high-error
  points by inflating their variance, starving the mean's gradient and collapsing
  to overconfidence elsewhere. β-NLL re-weights each row's NLL by a detached
  `σ^{2β}`, restoring the mean's gradient and removing the collapse. `β = 0.5` is
  the recommended default.
- **CRPS** — the closed-form Gaussian continuous ranked probability score, the
  other collapse-resistant choice: strictly proper, in the outcome's units, and
  far more stable under joint `μ,σ²` training than NLL.
- **Gaussian NLL** — the classic mean-variance objective, provided for
  completeness and as the pathology baseline. Prefer β-NLL or CRPS.
- **Pinball** — the quantile objective; trains each quantile to its level, with a
  non-crossing penalty that discourages crossing during training.

The same CRPS / NLL math headlines the
[calibration eval](./calibration-eval.md) — one source of truth for the score,
used as both the training loss and the eval metric.

## Calibrated, not merely accurate

A regression head is **not done** until its coverage is verified. Two models can
share a mean (identical MSE) yet one be badly miscalibrated. The
[calibration eval](./calibration-eval.md) is the gate: the central interval
should cover at ≈ its nominal level, and the head's proper score (CRPS/NLL)
should beat a constant-variance baseline. Verify coverage; never ship on NLL
alone.

## Aleatoric, not epistemic

A parametric Gaussian head models **aleatoric** (irreducible data) noise only. It
does *not* know what it has not seen: off-distribution it can be confidently
wrong. For uncertainty about the unseen, reach for the rest of the spectrum:

- distribution-free coverage with no model assumption →
  [conformal prediction](./conformal-prediction.md);
- amortized epistemic posteriors → a future neural-process head.

Do not read this head's `σ` as epistemic.

## The uncertainty evidence channel

A served distribution rides the `uncertainty` evidence channel
(`predicted_mean`, `predicted_std`, `quantiles`, `context_ref`) — the same
additive substrate as `vector`, `inference`, and `conformal`. When the prediction
was conditioned on an [assembled context set](./assemble-context.md), the channel
records which rows informed it in `context_ref` — data-driven provenance applied
to prediction. Register it like any custom channel (see
[Declare a Custom Provenance Channel](./declare-provenance-channel.md)); the
distribution columns then merge into the result and are SQL-reachable.
