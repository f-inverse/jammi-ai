//! The served regression head's σ arithmetic, shared by training and
//! serving so the trained σ and the served σ are one transform.

/// Hard numerical floor on the predictive standard deviation. The
/// *learnable* part of the floor is the head's own trainable bias under
/// `softplus`; this constant only guards against an exact-zero variance
/// (the `σ→0` overconfidence collapse), keeping every NLL/CRPS term finite.
/// The single source of truth for the floor: a serving adapter references
/// this constant, so the trained `σ` and the served `σ` are one transform.
pub const STD_FLOOR: f64 = 1e-3;

/// The σ-axis de-standardise — the **single** source of the `σ_y·σ_z` math
/// every serve path uses.
///
/// A z-space-trained Gaussian head emits a z-scale σ (`σ_z`, post-softplus).
/// To recover the raw σ the serve path multiplies by σ_y (the target
/// scaler's `std`) and re-floors at [`STD_FLOOR`] so the positivity
/// invariant survives the multiply. The multiply has to land here, on the
/// *post-softplus* σ, because `softplus` is non-linear
/// (`σ_y·softplus(raw) ≠ softplus(σ_y·raw)`).
pub fn destandardize_sigma(std_scale: f32, sigma_z: f32) -> f32 {
    (std_scale * sigma_z).max(STD_FLOOR as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destandardize_scales_then_refloors() {
        assert_eq!(destandardize_sigma(2.0, 0.5), 1.0);
        assert_eq!(destandardize_sigma(1.0, 0.0), STD_FLOOR as f32);
        assert_eq!(destandardize_sigma(0.0, 3.0), STD_FLOOR as f32);
    }
}
