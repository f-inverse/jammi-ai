//! Serving-time prediction wrappers that turn a point predictor's output into
//! a *guaranteed* output.
//!
//! The first member is `conformal` — split conformal prediction, the
//! distribution-free coverage primitive. It wraps any predictor's per-class
//! probabilities or point/quantile estimates into a prediction set or interval
//! with a finite-sample marginal coverage guarantee, deterministic given a
//! calibration set. It is a *serving* output; monitoring coverage over time and
//! deciding when to recalibrate are policies a caller builds on it.

pub mod conformal;

pub use conformal::{
    finite_sample_quantile, finite_sample_quantile_weighted, ClassScore, ConformalModel,
    IntervalScore, Threshold,
};
