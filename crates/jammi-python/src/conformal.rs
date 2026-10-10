//! The conformal verbs: pure functions of caller-supplied arrays, calibrated by
//! the engine's [`ConformalModel`]. They hold no session, so they are module
//! functions of `jammi_native`, not methods of a database.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use jammi_ai::predict::{Calibration, ClassScore, ConformalModel, Estimate, Estimates, TestPoint};
use jammi_db::error::JammiError;

use crate::error::to_pyerr;

/// Conformalize a classification predictor into prediction sets.
///
/// Split (inductive) conformal: `calibration` holds one row of per-class
/// probabilities per held-out example and `true_labels[i]` is the realised
/// class for row `i`; the calibration scores yield the finite-sample
/// `⌈(n+1)(1-alpha)⌉` quantile, applied to every row of `test` to emit a
/// prediction set with coverage `>= 1 - alpha`. `score` is `"lac"`, `"aps"` or
/// `"raps"` with `raps_params = (lambda, k_reg)`; `scheme` names the cohorts or
/// likelihood ratios a Mondrian or weighted calibration pools by.
///
/// Returns one list of admitted class indices per row of `test`.
#[pyfunction]
#[pyo3(signature = (calibration, true_labels, test, *, alpha, score, raps_params, scheme))]
pub(crate) fn conformalize(
    calibration: Vec<Vec<f64>>,
    true_labels: Vec<usize>,
    test: Vec<Vec<f64>>,
    alpha: f64,
    score: &str,
    raps_params: (f64, usize),
    scheme: SchemeArgs,
) -> PyResult<Vec<Vec<usize>>> {
    let scheme = Scheme::new(scheme, test.len())?;
    let model = ConformalModel::classification(
        &calibration,
        &true_labels,
        parse_class_score(score, raps_params)?,
        &scheme.calibration,
        alpha,
    )
    .map_err(to_pyerr)?;
    test.iter()
        .enumerate()
        .map(|(i, row)| model.predict_set(row, scheme.point(i)).map_err(to_pyerr))
        .collect()
}

/// Conformalize an absolute-residual regression predictor into constant-width
/// intervals `[ŷ - q̂, ŷ + q̂]` around each `test_predictions` point.
///
/// Returns one `(lower, upper)` tuple per test row.
#[pyfunction]
#[pyo3(signature = (predictions, observed, test_predictions, *, alpha, scheme))]
pub(crate) fn conformalize_interval(
    predictions: Vec<f64>,
    observed: Vec<f64>,
    test_predictions: Vec<f64>,
    alpha: f64,
    scheme: SchemeArgs,
) -> PyResult<Vec<(f64, f64)>> {
    let scheme = Scheme::new(scheme, test_predictions.len())?;
    let model = ConformalModel::regression(
        Estimates::Points(&predictions),
        &observed,
        &scheme.calibration,
        alpha,
    )
    .map_err(to_pyerr)?;
    test_predictions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            model
                .predict_interval(Estimate::Point(p), scheme.point(i))
                .map_err(to_pyerr)
        })
        .collect()
}

/// Conformalize a Conformalized Quantile Regression predictor into
/// adaptive-width intervals `[q_lo - q̂, q_hi + q̂]` around each test band.
///
/// Returns one `(lower, upper)` tuple per test row.
#[pyfunction]
#[pyo3(signature = (lower, upper, observed, test_lower, test_upper, *, alpha, scheme))]
pub(crate) fn conformalize_cqr(
    lower: Vec<f64>,
    upper: Vec<f64>,
    observed: Vec<f64>,
    test_lower: Vec<f64>,
    test_upper: Vec<f64>,
    alpha: f64,
    scheme: SchemeArgs,
) -> PyResult<Vec<(f64, f64)>> {
    if test_lower.len() != test_upper.len() {
        return Err(PyValueError::new_err(
            "cqr conformal: 'test_lower' and 'test_upper' must have equal length",
        ));
    }
    let scheme = Scheme::new(scheme, test_lower.len())?;
    let model = ConformalModel::regression(
        Estimates::Bands {
            lower: &lower,
            upper: &upper,
        },
        &observed,
        &scheme.calibration,
        alpha,
    )
    .map_err(to_pyerr)?;
    test_lower
        .iter()
        .zip(&test_upper)
        .enumerate()
        .map(|(i, (&lower, &upper))| {
            model
                .predict_interval(Estimate::Band { lower, upper }, scheme.point(i))
                .map_err(to_pyerr)
        })
        .collect()
}

/// A conformal verb's `(groups, test_groups, weights, test_weights)` arguments,
/// as the Python wrapper passes them.
#[derive(FromPyObject)]
pub(crate) struct SchemeArgs(
    Option<Vec<String>>,
    Option<Vec<String>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
);

/// The calibration scheme a verb's arguments name, and what each test row
/// brings to it.
struct Scheme {
    calibration: Calibration,
    test: SchemeRows,
}

enum SchemeRows {
    Exchangeable,
    Groups(Vec<String>),
    Weights(Vec<f64>),
}

impl Scheme {
    /// `groups` comes with `test_groups` (Mondrian) and `weights` with
    /// `test_weights` (weighted); the test-side list holds one entry per test
    /// row.
    fn new(
        SchemeArgs(groups, test_groups, weights, test_weights): SchemeArgs,
        test_rows: usize,
    ) -> PyResult<Self> {
        let per_test_row = |argument: &str, given: usize| {
            if given == test_rows {
                return Ok(());
            }
            Err(invalid(
                argument,
                format!("conformal needs one entry per test row: {given} for {test_rows} rows"),
            ))
        };
        match (groups, test_groups, weights, test_weights) {
            (None, None, None, None) => Ok(Self {
                calibration: Calibration::Marginal,
                test: SchemeRows::Exchangeable,
            }),
            (Some(groups), Some(test_groups), None, None) => {
                per_test_row("test_groups", test_groups.len())?;
                Ok(Self {
                    calibration: Calibration::Mondrian { groups },
                    test: SchemeRows::Groups(test_groups),
                })
            }
            (None, None, Some(weights), Some(test_weights)) => {
                per_test_row("test_weights", test_weights.len())?;
                Ok(Self {
                    calibration: Calibration::Weighted { weights },
                    test: SchemeRows::Weights(test_weights),
                })
            }
            _ => Err(invalid(
                "calibration",
                "conformal takes `groups` with `test_groups`, or `weights` with `test_weights`, \
                 and not both"
                    .into(),
            )),
        }
    }

    fn point(&self, row: usize) -> TestPoint<'_> {
        match &self.test {
            SchemeRows::Exchangeable => TestPoint::Exchangeable,
            SchemeRows::Groups(groups) => TestPoint::Group(&groups[row]),
            SchemeRows::Weights(weights) => TestPoint::Weight(weights[row]),
        }
    }
}

fn invalid(argument: &str, reason: String) -> PyErr {
    to_pyerr(JammiError::InvalidArgument {
        argument: argument.into(),
        reason,
    })
}

/// Parse a classification conformal score family from its snake-case name.
/// `"raps"` carries the regularization parameters; `"lac"` and `"aps"` ignore
/// them.
fn parse_class_score(score: &str, (lambda, k_reg): (f64, usize)) -> PyResult<ClassScore> {
    match score {
        "lac" => Ok(ClassScore::Lac),
        "aps" => Ok(ClassScore::Aps),
        "raps" => Ok(ClassScore::Raps { lambda, k_reg }),
        other => Err(PyValueError::new_err(format!(
            "unknown classification conformal score '{other}', expected 'lac', 'aps', or 'raps'"
        ))),
    }
}
