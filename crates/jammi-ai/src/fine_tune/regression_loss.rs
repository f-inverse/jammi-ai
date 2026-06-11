//! Distributional regression objectives (S18): the stateless, batched autodiff
//! peers of the R2 calibration metrics and the inference adapter's σ map.
//!
//! These are pure functions of a head output `(batch, k)` and a target — they
//! read no training state, and they score the head's **raw, outcome-unit**
//! output against the **raw** target. The trainer's `regression_loss` dispatches
//! on the configured [`super::RegressionLoss`] to them; the inference adapter
//! reads the single [`STD_FLOOR`] so the trained σ and the served σ are one
//! constant.
//!
//! Reachability of a high-offset target (calendar years, prices) is *not* a loss
//! concern — scoring in a standardised space does not change how far Adam moves
//! the raw head parameter, because Adam's per-step update is ≈`lr·sign(grad)`
//! regardless of the gradient's magnitude. Reachability is solved inside the head
//! by [`TargetScaler::destandardize_gaussian`] /
//! [`TargetScaler::destandardize_quantile`], which reparameterise the head so its
//! *optimised* parameter is O(1) from zero-init while its *output* is
//! raw-correct. The loss only ever sees that raw-correct output, so it stays a
//! pure function of `(head, target)`.

use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};
use serde::{Deserialize, Serialize};

/// Hard numerical floor on the predictive standard deviation, the autodiff peer
/// of the inference adapter's `SERVED_STD_FLOOR`. The *learnable* part of the
/// floor is the head's own trainable bias under `softplus`; this constant only
/// guards against an exact-zero variance (the `σ→0` overconfidence collapse),
/// keeping every NLL/CRPS term finite. The single source of truth for the
/// floor: the inference adapter's `SERVED_STD_FLOOR` references this constant,
/// so the trained `σ` and the served `σ` are one transform.
pub(crate) const STD_FLOOR: f64 = 1e-3;

/// Dataset-level target standardiser carried *by the regression head*: the
/// `mean` (μ_y) and `std` (σ_y) of all training targets, computed once before
/// the loop and held fixed for the whole run.
///
/// # Why the head, not the loss
///
/// A distribution head is zero-initialised, so its predictive `mean` starts at
/// 0. A high-offset target — calendar years, prices — sits thousands of units
/// from 0. Adam's per-parameter update is ≈`lr·sign(grad)`: it normalises by the
/// gradient's running RMS, so the *distance* the raw mean parameter travels is
/// ≈`lr` per step **regardless of the loss's scale or units**. Scoring the loss
/// in a standardised z-space therefore does *not* help — the raw mean still
/// crawls ≈`lr·steps` units and stalls thousands of units short of the target.
///
/// The fix is a reparameterisation, applied inside the head's forward: the head
/// learns a z-space parameter `z` (O(1) from zero-init, reachable by Adam) and
/// its output is the **de-standardised** raw value `μ_y + σ_y·z`. With `z`
/// zero-init the head already emits `μ_y`, and Adam only has to nudge `z` by
/// O(1) to track each row. The loss then scores this raw-correct output against
/// the raw target with no scaler in sight, and serving reads the same raw output
/// directly. The scaler is the one transform shared by training and serving, so
/// it is **persisted with the head** (in the adapter config) and reloaded on
/// serve.
///
/// # The σ column stays in the existing softplus path
///
/// Only the *mean* (Gaussian) / quantile (pinball) columns carry the target
/// offset and need de-standardisation. The Gaussian σ column is a *raw scale*
/// the existing `STD_FLOOR + softplus(raw)` map already turns into a positive σ;
/// its optimum (`σ_y`, an O(1) spread) is reachable from zero-init without any
/// reparameterisation, so that column is passed through untouched — which is
/// exactly what keeps `gaussian_params` and the serving adapter's σ map
/// unchanged, and what keeps the served σ ≥ [`STD_FLOOR`].
///
/// `std` is floored at [`STD_FLOOR`] at construction, so a degenerate
/// (single-valued or constant) target set cannot produce a zero-multiplier
/// scaler — the invalid `std = 0` state is unrepresentable, and a row's
/// de-standardised mean still moves with `z` rather than collapsing onto μ_y.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct TargetScaler {
    /// Mean of all training targets, μ_y. The zero-init head emits this.
    mean: f64,
    /// Standard deviation of all training targets, σ_y, floored at [`STD_FLOOR`].
    std: f64,
}

impl TargetScaler {
    /// Build the scaler from all training targets: the population mean and
    /// standard deviation of `targets`, with `std` floored at [`STD_FLOOR`].
    pub(crate) fn from_targets(targets: &Tensor) -> Result<Self> {
        let mean = targets
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("scaler mean: {e}")))?
            .to_dtype(candle_core::DType::F64)
            .map_err(|e| JammiError::FineTune(format!("scaler mean dtype: {e}")))?
            .to_scalar::<f64>()
            .map_err(|e| JammiError::FineTune(format!("scaler mean scalar: {e}")))?;
        let var = (targets - mean)
            .map_err(|e| JammiError::FineTune(format!("scaler centre: {e}")))?
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("scaler sqr: {e}")))?
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("scaler var: {e}")))?
            .to_dtype(candle_core::DType::F64)
            .map_err(|e| JammiError::FineTune(format!("scaler var dtype: {e}")))?
            .to_scalar::<f64>()
            .map_err(|e| JammiError::FineTune(format!("scaler var scalar: {e}")))?;
        Ok(Self {
            mean,
            std: var.sqrt().max(STD_FLOOR),
        })
    }

    /// De-standardise a Gaussian head's raw `(batch, 2)` output. The mean column
    /// (0) is mapped `μ_y + σ_y·z`; the raw-scale column (1) is passed through so
    /// the downstream `STD_FLOOR + softplus(raw)` σ map is unchanged. With `z`
    /// zero-init the emitted mean is exactly μ_y, so a high-offset target is
    /// reachable; the emitted column-1 is the raw scale the loss and serving
    /// adapter both already interpret.
    pub(crate) fn destandardize_gaussian(&self, raw_head: &Tensor) -> Result<Tensor> {
        let (_, k) = raw_head
            .dims2()
            .map_err(|e| JammiError::FineTune(format!("destandardize gaussian dims: {e}")))?;
        if k != 2 {
            return Err(JammiError::FineTune(format!(
                "Gaussian head de-standardisation expects width 2 (mean, raw_std), got {k}"
            )));
        }
        let z_mean = raw_head
            .narrow(1, 0, 1)
            .map_err(|e| JammiError::FineTune(format!("destd mean narrow: {e}")))?;
        let raw_scale = raw_head
            .narrow(1, 1, 1)
            .map_err(|e| JammiError::FineTune(format!("destd scale narrow: {e}")))?;
        let mean = self.affine(&z_mean)?;
        Tensor::cat(&[&mean, &raw_scale], 1)
            .map_err(|e| JammiError::FineTune(format!("destd gaussian cat: {e}")))
    }

    /// De-standardise a quantile head's raw `(batch, n_levels)` output. Every
    /// column is an outcome-unit quantile, so the affine `μ_y + σ_y·z` is applied
    /// to all of them. The map is monotone (σ_y > 0), so a non-crossing raw head
    /// stays non-crossing after de-standardisation and the served set is coherent.
    pub(crate) fn destandardize_quantile(&self, raw_head: &Tensor) -> Result<Tensor> {
        self.affine(raw_head)
    }

    /// The de-standardising affine `μ_y + σ_y·z`, applied elementwise.
    fn affine(&self, z: &Tensor) -> Result<Tensor> {
        ((z * self.std).map_err(|e| JammiError::FineTune(format!("destd scale: {e}")))? + self.mean)
            .map_err(|e| JammiError::FineTune(format!("destd shift: {e}")))
    }
}

/// Numerically-stable `softplus(x) = ln(1 + e^x)`, computed as
/// `relu(x) + ln(1 + e^{−|x|})` so a large positive `x` cannot overflow `exp`.
/// Smooth and positive everywhere; the autodiff peer of the inference adapter's
/// `softplus_std`.
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x
        .relu()
        .map_err(|e| JammiError::FineTune(format!("softplus relu: {e}")))?;
    let neg_abs = x
        .abs()
        .map_err(|e| JammiError::FineTune(format!("softplus abs: {e}")))?
        .neg()
        .map_err(|e| JammiError::FineTune(format!("softplus neg: {e}")))?;
    let log_term = ((neg_abs
        .exp()
        .map_err(|e| JammiError::FineTune(format!("softplus exp: {e}")))?
        + 1.0)
        .map_err(|e| JammiError::FineTune(format!("softplus add1: {e}")))?)
    .log()
    .map_err(|e| JammiError::FineTune(format!("softplus log: {e}")))?;
    (&relu + &log_term).map_err(|e| JammiError::FineTune(format!("softplus sum: {e}")))
}

/// Read `(mean, σ)` from a two-wide Gaussian head output `(batch, 2)`, mapping
/// the raw scale column to a positive `σ = STD_FLOOR + softplus(raw)`. `softplus`
/// (candle-native) is smooth and positive everywhere, so the head trains in an
/// unconstrained space while `σ` stays valid — the standard mean-variance head
/// parameterisation ([Nix & Weigend 1994]).
pub(crate) fn gaussian_params(input: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_, k) = input
        .dims2()
        .map_err(|e| JammiError::FineTune(format!("regression head dims: {e}")))?;
    if k != 2 {
        return Err(JammiError::FineTune(format!(
            "Gaussian regression objective expects a 2-wide head (mean, raw_std), got width {k}"
        )));
    }
    let mean = input
        .narrow(1, 0, 1)
        .map_err(|e| JammiError::FineTune(format!("gauss mean narrow: {e}")))?
        .squeeze(1)
        .map_err(|e| JammiError::FineTune(format!("gauss mean squeeze: {e}")))?;
    let raw = input
        .narrow(1, 1, 1)
        .map_err(|e| JammiError::FineTune(format!("gauss raw narrow: {e}")))?
        .squeeze(1)
        .map_err(|e| JammiError::FineTune(format!("gauss raw squeeze: {e}")))?;
    // σ = STD_FLOOR + softplus(raw). Numerically-stable softplus:
    // softplus(x) = relu(x) + ln(1 + exp(−|x|)), which avoids `exp` overflow for
    // large positive `x` while staying smooth and exactly equal to ln(1+e^x).
    let sigma = (softplus(&raw)? + STD_FLOOR)
        .map_err(|e| JammiError::FineTune(format!("gauss sigma floor: {e}")))?;
    Ok((mean, sigma))
}

/// (β-)Gaussian-NLL loss over a two-wide head, the mean of the per-row
/// `½(log σ² + (y−μ)²/σ²)` weighted by a stop-gradient `σ^{2β}` (β-NLL,
/// [Seitzer et al. 2022]). `beta = 0` is the plain heteroscedastic NLL; `beta`
/// in `(0, 1]` re-weights the per-row term by the detached predictive variance,
/// restoring the mean's gradient on high-variance rows and removing the
/// variance-collapse / mean-starvation pathology of joint `μ,σ²` NLL.
///
/// Shares the closed form with [`jammi_numerics::calibration::gaussian_nll`]
/// (the R2 eval metric); this is its differentiable, batched autodiff peer.
pub(crate) fn gaussian_nll_loss(input: &Tensor, target: &Tensor, beta: f64) -> Result<Tensor> {
    let (mean, sigma) = gaussian_params(input)?;
    let var = sigma
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("nll var: {e}")))?;
    let resid = (&mean - target).map_err(|e| JammiError::FineTune(format!("nll resid: {e}")))?;
    let resid_sq = resid
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("nll resid sqr: {e}")))?;
    // Per-row NLL up to the constant ½log(2π): ½(log σ² + (y−μ)²/σ²).
    let log_var = var
        .log()
        .map_err(|e| JammiError::FineTune(format!("nll log var: {e}")))?;
    let quad = (&resid_sq / &var).map_err(|e| JammiError::FineTune(format!("nll quad: {e}")))?;
    let per_row =
        ((&log_var + &quad).map_err(|e| JammiError::FineTune(format!("nll add: {e}")))? * 0.5)
            .map_err(|e| JammiError::FineTune(format!("nll half: {e}")))?;

    if beta == 0.0 {
        return per_row
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("nll mean: {e}")));
    }

    // β weighting: multiply each row's NLL by a STOP-GRADIENT σ^{2β}. Detaching
    // the weight is what makes β-NLL a re-weighting of the NLL rather than a new
    // objective — the gradient flows only through `per_row`.
    let weight = var
        .powf(beta)
        .map_err(|e| JammiError::FineTune(format!("beta-nll pow: {e}")))?
        .detach();
    let weighted =
        (&per_row * &weight).map_err(|e| JammiError::FineTune(format!("beta-nll weight: {e}")))?;
    weighted
        .mean_all()
        .map_err(|e| JammiError::FineTune(format!("beta-nll mean: {e}")))
}

/// Closed-form Gaussian CRPS loss over a two-wide head — the strictly-proper,
/// outcome-unit-scaled, collapse-resistant alternative to NLL.
///
/// Per row, with `z = (y − μ)/σ` and standard-normal `Φ`, `φ`:
/// `CRPS = σ ( z(2Φ(z) − 1) + 2φ(z) − 1/√π )`. This is exactly
/// [`jammi_numerics::calibration::crps_gaussian`]'s closed form; here it is the
/// differentiable, batched autodiff peer (`Φ` via `erf`, `φ` the Gaussian PDF),
/// so the training loss and the R2 metric are one formula.
pub(crate) fn crps_gaussian_loss(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (mean, sigma) = gaussian_params(input)?;
    let z = ((&mean - target).map_err(|e| JammiError::FineTune(format!("crps resid: {e}")))?
        / &sigma)
        .map_err(|e| JammiError::FineTune(format!("crps z: {e}")))?;
    // Φ(z) = ½(1 + erf(z/√2)).
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let erf_arg =
        (&z * inv_sqrt2).map_err(|e| JammiError::FineTune(format!("crps erf arg: {e}")))?;
    let erf = erf_arg
        .erf()
        .map_err(|e| JammiError::FineTune(format!("crps erf: {e}")))?;
    let cdf = ((&erf + 1.0).map_err(|e| JammiError::FineTune(format!("crps cdf add: {e}")))? * 0.5)
        .map_err(|e| JammiError::FineTune(format!("crps cdf half: {e}")))?;
    // φ(z) = exp(−z²/2)/√(2π).
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let neg_half_z2 = ((&z
        .sqr()
        .map_err(|e| JammiError::FineTune(format!("crps z2: {e}")))?
        * -0.5)
        .map_err(|e| JammiError::FineTune(format!("crps z2 half: {e}")))?)
    .exp()
    .map_err(|e| JammiError::FineTune(format!("crps exp: {e}")))?;
    let pdf = (&neg_half_z2 * inv_sqrt_2pi)
        .map_err(|e| JammiError::FineTune(format!("crps pdf: {e}")))?;

    // z(2Φ−1) + 2φ − 1/√π.
    let two_cdf_m1 = ((&cdf * 2.0).map_err(|e| JammiError::FineTune(format!("crps 2cdf: {e}")))?
        - 1.0)
        .map_err(|e| JammiError::FineTune(format!("crps 2cdf-1: {e}")))?;
    let term_z =
        (&z * &two_cdf_m1).map_err(|e| JammiError::FineTune(format!("crps term_z: {e}")))?;
    let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI / 2.0;
    let two_pdf = (&pdf * 2.0).map_err(|e| JammiError::FineTune(format!("crps 2pdf: {e}")))?;
    let bracket = ((&term_z + &two_pdf)
        .map_err(|e| JammiError::FineTune(format!("crps bracket add: {e}")))?
        - inv_sqrt_pi)
        .map_err(|e| JammiError::FineTune(format!("crps bracket sub: {e}")))?;
    let per_row =
        (&sigma * &bracket).map_err(|e| JammiError::FineTune(format!("crps scale: {e}")))?;
    per_row
        .mean_all()
        .map_err(|e| JammiError::FineTune(format!("crps mean: {e}")))
}

/// Pinball / quantile loss ([Koenker & Bassett 1978]) over a quantile head
/// `(batch, n_levels)`, summed over the declared `levels` plus a non-crossing
/// penalty.
///
/// For level `q` and prediction `ŷ`, the per-row pinball term is
/// `max(q·(y−ŷ), (q−1)·(y−ŷ))`, minimised when `ŷ` is the `q`-quantile of `y`.
/// The non-crossing penalty `Σ relu(ŷ_k − ŷ_{k+1})` discourages adjacent
/// quantiles from crossing *during training*; the serving adapter additionally
/// sorts post-hoc so the served set is always coherent.
pub(crate) fn pinball_loss(input: &Tensor, target: &Tensor, levels: &[f64]) -> Result<Tensor> {
    let (_, k) = input
        .dims2()
        .map_err(|e| JammiError::FineTune(format!("pinball head dims: {e}")))?;
    if k != levels.len() {
        return Err(JammiError::FineTune(format!(
            "pinball head width {k} does not match {} declared quantile levels",
            levels.len()
        )));
    }
    if levels.is_empty() {
        return Err(JammiError::FineTune(
            "pinball loss requires at least one quantile level".into(),
        ));
    }

    let mut total: Option<Tensor> = None;
    for (idx, &q) in levels.iter().enumerate() {
        let pred = input
            .narrow(1, idx, 1)
            .map_err(|e| JammiError::FineTune(format!("pinball narrow: {e}")))?
            .squeeze(1)
            .map_err(|e| JammiError::FineTune(format!("pinball squeeze: {e}")))?;
        let err =
            (target - &pred).map_err(|e| JammiError::FineTune(format!("pinball err: {e}")))?;
        // max(q·err, (q−1)·err) elementwise.
        let lo = (&err * q).map_err(|e| JammiError::FineTune(format!("pinball lo: {e}")))?;
        let hi =
            (&err * (q - 1.0)).map_err(|e| JammiError::FineTune(format!("pinball hi: {e}")))?;
        let term = lo
            .maximum(&hi)
            .map_err(|e| JammiError::FineTune(format!("pinball max: {e}")))?
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("pinball mean: {e}")))?;
        total = Some(match total {
            None => term,
            Some(acc) => {
                (&acc + &term).map_err(|e| JammiError::FineTune(format!("pinball sum: {e}")))?
            }
        });
    }
    let mut loss = total.ok_or_else(|| JammiError::FineTune("pinball: no levels".into()))?;

    // Non-crossing penalty: Σ_k relu(ŷ_k − ŷ_{k+1}). Levels are ascending, so a
    // coherent head has ŷ_k ≤ ŷ_{k+1} and the penalty is zero.
    if levels.len() > 1 {
        let left = input
            .narrow(1, 0, levels.len() - 1)
            .map_err(|e| JammiError::FineTune(format!("noncross left: {e}")))?;
        let right = input
            .narrow(1, 1, levels.len() - 1)
            .map_err(|e| JammiError::FineTune(format!("noncross right: {e}")))?;
        let gap =
            (&left - &right).map_err(|e| JammiError::FineTune(format!("noncross gap: {e}")))?;
        let zero = Tensor::zeros_like(&gap)
            .map_err(|e| JammiError::FineTune(format!("noncross zero: {e}")))?;
        let penalty = gap
            .maximum(&zero)
            .map_err(|e| JammiError::FineTune(format!("noncross relu: {e}")))?
            .mean_all()
            .map_err(|e| JammiError::FineTune(format!("noncross mean: {e}")))?;
        loss = (&loss + &penalty)
            .map_err(|e| JammiError::FineTune(format!("pinball + penalty: {e}")))?;
    }
    Ok(loss)
}

/// Map a raw head scale to the trained `σ = STD_FLOOR + softplus(raw)`, used by
/// the in-crate regression tests to convert a crafted `raw_std` into the `σ` the
/// loss and the inference adapter both see.
#[cfg(test)]
pub(crate) fn softplus_std_for_test(raw: f64) -> f64 {
    let sp = if raw > 20.0 {
        raw
    } else {
        (1.0 + raw.exp()).ln()
    };
    STD_FLOOR + sp
}
