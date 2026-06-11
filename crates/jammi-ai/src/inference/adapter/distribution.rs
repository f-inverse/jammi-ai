//! The distributional output adapter: turns a regression backend's raw floats
//! into a *predictive distribution* per row, in one of two standard forms.
//!
//! - **Parametric** ([`DistributionForm::Gaussian`]) — two columns,
//!   `predicted_mean` and `predicted_std`, the parameters of a per-row
//!   `Normal`. Smooth, cheap, and a closed-form density; trained by the
//!   Gaussian-NLL / β-NLL / closed-form-CRPS objectives.
//! - **Quantile** ([`DistributionForm::Quantile`]) — one `quantile_{level}`
//!   column per declared level (e.g. `quantile_0.05`, `quantile_0.5`,
//!   `quantile_0.95`). Distribution-free in shape, feeds S17's CQR, robust to
//!   non-Gaussian outcomes; trained by the pinball objective. The adapter
//!   enforces monotonicity across levels (a non-crossing guard) so the served
//!   quantiles are coherent even if the raw head emits a crossing.
//!
//! The backend emits a single float head of shape `(rows, k)`: `k = 2` for the
//! Gaussian form (mean, raw-scale), `k = levels.len()` for the quantile form.
//! The raw second Gaussian column is mapped to a positive standard deviation by
//! the same `softplus + floor` the training objective uses, so the served `std`
//! and the trained `σ` are one transform — never two spellings of the variance.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field};
use jammi_db::error::{JammiError, Result};

use super::{BackendOutput, OutputAdapter};

/// The minimum standard deviation served by the Gaussian head, the inference
/// peer of the trainer's learnable variance floor. A served `σ` is never below
/// this, so a row the model is (over)confident about still yields a finite,
/// scorable density rather than a zero-width spike.
///
/// This is *the same* floor the training objective uses, not a second literal
/// kept equal by prose: it references the trainer's crate-internal `STD_FLOOR`
/// (the single source of truth) so the trained `σ` and the served `σ` are one
/// transform.
pub const SERVED_STD_FLOOR: f32 = crate::fine_tune::regression_loss::STD_FLOOR as f32;

/// Map a raw real-valued head output to a positive standard deviation:
/// `floor + softplus(raw)`. `softplus(x) = ln(1 + e^x)` is smooth and positive
/// everywhere; the additive floor keeps the variance away from zero (the
/// overconfidence failure mode). Shared shape with the trainer's
/// `softplus_std` so serving and training agree on what `σ` means.
fn softplus_std(raw: f32, floor: f32) -> f32 {
    // Numerically-stable softplus: for large x, ln(1+e^x) ≈ x.
    let sp = if raw > 20.0 {
        raw
    } else {
        (1.0 + raw.exp()).ln()
    };
    floor + sp
}

/// Which predictive distribution shape a regression model emits.
///
/// This is the authoritative gaussian-vs-quantile signal: it is set from the
/// configured regression objective at fine-tune time, persisted with the head
/// in the adapter config, and reloaded on serve so the de-standardisation
/// dispatches on the *form* the head was trained for — never on a head-width
/// heuristic (a 2-level quantile head is also width 2).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionForm {
    /// Parametric Gaussian: the head emits `(mean, raw_std)`; the served
    /// columns are `predicted_mean` and `predicted_std = floor + softplus(raw_std)`.
    Gaussian,
    /// Quantile: the head emits one value per declared level. The served
    /// columns are `quantile_{level}` in ascending level order, sorted per row
    /// so they never cross.
    Quantile {
        /// Quantile levels in `(0, 1)`, e.g. `[0.05, 0.5, 0.95]`. Held in
        /// ascending order; the constructor rejects an out-of-range or
        /// non-ascending set so the column order is well-defined.
        levels: Vec<f64>,
    },
}

/// Adapt a regression backend's raw floats into a predictive-distribution
/// Arrow schema: `(predicted_mean, predicted_std)` for the Gaussian form, or
/// `quantile_{level}` columns for the quantile form.
#[derive(Debug, Clone)]
pub struct DistributionAdapter {
    form: DistributionForm,
}

impl DistributionAdapter {
    /// Parametric Gaussian head: serves `predicted_mean` + `predicted_std`.
    pub fn gaussian() -> Self {
        Self {
            form: DistributionForm::Gaussian,
        }
    }

    /// Quantile head over the given levels. Levels must be ascending and lie in
    /// `(0, 1)`; otherwise the column order would be ambiguous, so it is a typed
    /// error rather than a silent sort.
    pub fn quantile(levels: Vec<f64>) -> Result<Self> {
        if levels.is_empty() {
            return Err(JammiError::Inference(
                "quantile distribution head requires at least one level".into(),
            ));
        }
        if levels.iter().any(|&q| !(0.0..1.0).contains(&q) || q <= 0.0) {
            return Err(JammiError::Inference(
                "quantile levels must lie strictly in (0, 1)".into(),
            ));
        }
        if levels.windows(2).any(|w| w[1] <= w[0]) {
            return Err(JammiError::Inference(
                "quantile levels must be strictly ascending".into(),
            ));
        }
        Ok(Self {
            form: DistributionForm::Quantile { levels },
        })
    }

    /// The number of raw float columns the backend head must emit for this form:
    /// `2` for Gaussian, one per level for quantile.
    fn head_width(&self) -> usize {
        match &self.form {
            DistributionForm::Gaussian => 2,
            DistributionForm::Quantile { levels } => levels.len(),
        }
    }

    /// The output form this adapter serves.
    pub fn form(&self) -> &DistributionForm {
        &self.form
    }
}

/// Column name for a quantile level, e.g. `0.05 -> "quantile_0.05"`. Trailing
/// zeros are trimmed so `0.5 -> "quantile_0.5"`, matching how a user names the
/// level.
fn quantile_column_name(level: f64) -> String {
    // `{}` on f64 already trims trailing zeros (0.5 -> "0.5", 0.05 -> "0.05").
    format!("quantile_{level}")
}

impl OutputAdapter for DistributionAdapter {
    fn output_schema(&self) -> Vec<Field> {
        match &self.form {
            DistributionForm::Gaussian => vec![
                Field::new("predicted_mean", DataType::Float32, true),
                Field::new("predicted_std", DataType::Float32, true),
            ],
            DistributionForm::Quantile { levels } => levels
                .iter()
                .map(|&q| Field::new(quantile_column_name(q), DataType::Float32, true))
                .collect(),
        }
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        let width = self.head_width();
        if row_count == 0 {
            return Ok((0..width)
                .map(|_| Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef)
                .collect());
        }

        let flat = output.float_outputs.first().ok_or_else(|| {
            JammiError::Inference("distribution adapter: backend emitted no float head".into())
        })?;
        if flat.len() != row_count * width {
            return Err(JammiError::Inference(format!(
                "distribution adapter: head has {} floats, expected rows({row_count}) * width({width})",
                flat.len()
            )));
        }

        match &self.form {
            DistributionForm::Gaussian => {
                let mut means = Vec::with_capacity(row_count);
                let mut stds = Vec::with_capacity(row_count);
                for row in 0..row_count {
                    let ok = output.row_status.get(row).copied().unwrap_or(false);
                    if ok {
                        let mean = flat[row * 2];
                        let raw_std = flat[row * 2 + 1];
                        means.push(Some(mean));
                        stds.push(Some(softplus_std(raw_std, SERVED_STD_FLOOR)));
                    } else {
                        means.push(None);
                        stds.push(None);
                    }
                }
                Ok(vec![
                    Arc::new(Float32Array::from(means)),
                    Arc::new(Float32Array::from(stds)),
                ])
            }
            DistributionForm::Quantile { levels } => {
                let q = levels.len();
                // One column vector per level; each row's quantiles are sorted
                // ascending so the served set never crosses (the monotonicity
                // guard). A failed row is null across every level.
                let mut cols: Vec<Vec<Option<f32>>> = vec![Vec::with_capacity(row_count); q];
                for row in 0..row_count {
                    let ok = output.row_status.get(row).copied().unwrap_or(false);
                    if ok {
                        let mut row_vals: Vec<f32> = flat[row * q..row * q + q].to_vec();
                        // Sort ascending — the post-hoc non-crossing guard. NaNs
                        // would break the order; a non-finite head output is a
                        // backend bug, surfaced as a typed error.
                        if row_vals.iter().any(|v| !v.is_finite()) {
                            return Err(JammiError::Inference(format!(
                                "distribution adapter: row {row} quantile output is non-finite"
                            )));
                        }
                        row_vals.sort_by(f32::total_cmp);
                        for (level_idx, &v) in row_vals.iter().enumerate() {
                            cols[level_idx].push(Some(v));
                        }
                    } else {
                        for col in cols.iter_mut() {
                            col.push(None);
                        }
                    }
                }
                Ok(cols
                    .into_iter()
                    .map(|c| Arc::new(Float32Array::from(c)) as ArrayRef)
                    .collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    fn backend(floats: Vec<f32>, rows: usize, cols: usize, status: Vec<bool>) -> BackendOutput {
        BackendOutput {
            float_outputs: vec![floats],
            string_outputs: vec![],
            row_status: status,
            row_errors: vec![String::new(); rows],
            shapes: vec![(rows, cols)],
        }
    }

    #[test]
    fn gaussian_schema_is_mean_then_std() {
        let fields = DistributionAdapter::gaussian().output_schema();
        let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["predicted_mean", "predicted_std"]);
    }

    #[test]
    fn served_floor_is_the_single_trainer_floor() {
        // The served floor is not a second literal kept equal by prose — it IS
        // the trainer's STD_FLOOR, referenced. One constant, two views (f64 in
        // the autodiff objective, f32 at serve time).
        assert_eq!(
            SERVED_STD_FLOOR,
            crate::fine_tune::regression_loss::STD_FLOOR as f32
        );
    }

    #[test]
    fn served_sigma_matches_trained_sigma_across_raw_sweep() {
        // The σ map must agree between training and serving for every raw scale,
        // or a model would be scored under a different σ than it was trained on.
        // Drive the adapter's full serve path (which applies the adapter-side
        // `softplus_std`) and the trainer's `gaussian_params` (the autodiff σ
        // map) on the same raw values and require the served σ to equal the
        // trained σ. The two share the floor constant exactly (the single
        // source of truth) and the same `floor + softplus(raw)` formula; they
        // differ only by the last-bit rounding of two softplus spellings — the
        // candle-native numerically-stable form vs the scalar `ln(1+e^x)` — so
        // the agreement is to f32 round-off (≤ a few ULP), not a wider drift
        // that would mean two different transforms.
        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let mut raw = -5.0_f32;
        while raw <= 5.0 {
            // Adapter side: serve a one-row Gaussian head `(mean, raw)`.
            let out = backend(vec![0.0, raw], 1, 2, vec![true]);
            let cols = DistributionAdapter::gaussian().adapt(&out, 1).unwrap();
            let served = cols[1]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0);

            // Trainer side: the same raw through the autodiff σ map, read as f32.
            let input = Tensor::from_vec(vec![0.0_f32, raw], (1, 2), &dev).unwrap();
            let (_, sigma) = crate::fine_tune::regression_loss::gaussian_params(&input).unwrap();
            let trained: f32 = sigma.squeeze(0).unwrap().to_scalar().unwrap();

            // Tight relative tolerance: only f32 last-bit rounding may separate
            // them. A real divergence (a changed floor or a different formula)
            // is orders of magnitude larger and trips this guard.
            let tol = 4.0 * f32::EPSILON * served.abs().max(1.0);
            assert!(
                (served - trained).abs() <= tol,
                "served σ {served} and trained σ {trained} disagree for raw={raw} (tol {tol})"
            );
            raw += 0.5;
        }
    }

    #[test]
    fn gaussian_std_is_softplus_with_floor() {
        // raw_std = 0 -> softplus(0) = ln 2 ≈ 0.693, plus the floor.
        let out = backend(vec![1.5, 0.0], 1, 2, vec![true]);
        let cols = DistributionAdapter::gaussian().adapt(&out, 1).unwrap();
        let mean = cols[0].as_any().downcast_ref::<Float32Array>().unwrap();
        let std = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(mean.value(0), 1.5);
        let expected = SERVED_STD_FLOOR + 2.0_f32.ln();
        assert!(
            (std.value(0) - expected).abs() < 1e-5,
            "got {}",
            std.value(0)
        );
    }

    #[test]
    fn gaussian_std_never_below_floor_even_for_very_negative_raw() {
        // A strongly negative raw_std (the overconfidence direction) still
        // serves a std at or above the floor — softplus(-inf) -> 0, + floor.
        let out = backend(vec![0.0, -50.0], 1, 2, vec![true]);
        let cols = DistributionAdapter::gaussian().adapt(&out, 1).unwrap();
        let std = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
        assert!(std.value(0) >= SERVED_STD_FLOOR);
    }

    #[test]
    fn failed_row_is_null_across_columns() {
        let out = backend(vec![0.0, 0.0], 1, 2, vec![false]);
        let cols = DistributionAdapter::gaussian().adapt(&out, 1).unwrap();
        assert!(cols[0].is_null(0));
        assert!(cols[1].is_null(0));
    }

    #[test]
    fn quantile_columns_are_named_per_level() {
        let adapter = DistributionAdapter::quantile(vec![0.05, 0.5, 0.95]).unwrap();
        let names: Vec<String> = adapter
            .output_schema()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            vec!["quantile_0.05", "quantile_0.5", "quantile_0.95"]
        );
    }

    #[test]
    fn quantile_adapter_sorts_crossing_rows_monotone() {
        // Head emits a CROSSING row (q05 > q95): 2.0, 0.5, 1.0. The non-crossing
        // guard sorts it ascending into 0.5, 1.0, 2.0 before serving.
        let out = backend(vec![2.0, 0.5, 1.0], 1, 3, vec![true]);
        let adapter = DistributionAdapter::quantile(vec![0.05, 0.5, 0.95]).unwrap();
        let cols = adapter.adapt(&out, 1).unwrap();
        let q05 = cols[0].as_any().downcast_ref::<Float32Array>().unwrap();
        let q50 = cols[1].as_any().downcast_ref::<Float32Array>().unwrap();
        let q95 = cols[2].as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(q05.value(0), 0.5);
        assert_eq!(q50.value(0), 1.0);
        assert_eq!(q95.value(0), 2.0);
        // Coherent: ascending across the served levels.
        assert!(q05.value(0) <= q50.value(0) && q50.value(0) <= q95.value(0));
    }

    #[test]
    fn quantile_rejects_non_ascending_levels() {
        assert!(DistributionAdapter::quantile(vec![0.5, 0.5]).is_err());
        assert!(DistributionAdapter::quantile(vec![0.9, 0.1]).is_err());
        assert!(DistributionAdapter::quantile(vec![0.0, 0.5]).is_err());
        assert!(DistributionAdapter::quantile(vec![]).is_err());
    }

    #[test]
    fn head_width_mismatch_is_typed_error() {
        // Gaussian needs 2 floats per row; supply 3.
        let out = backend(vec![1.0, 2.0, 3.0], 1, 3, vec![true]);
        assert!(DistributionAdapter::gaussian().adapt(&out, 1).is_err());
    }
}
