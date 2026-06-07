//! Distributional regression objectives (S18): the stateless, batched autodiff
//! peers of the R2 calibration metrics and the inference adapter's σ map.
//!
//! These are pure functions of a head output `(batch, k)` and a target — they
//! read no training state. The trainer's `regression_loss` dispatches on the
//! configured [`super::RegressionLoss`] to them; the inference adapter reads the
//! single [`STD_FLOOR`] so the trained σ and the served σ are one constant.

use candle_core::Tensor;
use jammi_db::error::{JammiError, Result};

/// Hard numerical floor on the predictive standard deviation, the autodiff peer
/// of the inference adapter's `SERVED_STD_FLOOR`. The *learnable* part of the
/// floor is the head's own trainable bias under `softplus`; this constant only
/// guards against an exact-zero variance (the `σ→0` overconfidence collapse),
/// keeping every NLL/CRPS term finite. The single source of truth for the
/// floor: the inference adapter's `SERVED_STD_FLOOR` references this constant,
/// so the trained `σ` and the served `σ` are one transform.
pub(crate) const STD_FLOOR: f64 = 1e-3;

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
