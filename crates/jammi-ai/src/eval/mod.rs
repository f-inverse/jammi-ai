//! Evaluation: task-specific metrics against golden datasets.
//!
//! This module computes retrieval (recall@k, MRR, nDCG) and classification
//! (accuracy, F1) metrics. Results are recorded in the catalog for comparison
//! and for Phase 13's ExperimentRunner.

pub mod golden;
// The eval *runner* drives the embedded engine (it loads models and runs
// inference/search), so it rides the `local` feature. The report shapes,
// `EvalTask`, and `EvalCalibrationShape` are transport-neutral and live on the
// `jammi-wire` substrate (so the gRPC converters satisfy the orphan rule); they
// are re-exported here at their original paths so the engine modules and SDK
// consumers reach them as `jammi_ai::eval::*`.
#[cfg(feature = "local")]
pub mod runner;

pub use jammi_wire::eval::{
    AggregateDelta, CalibrationAggregate, CalibrationEvalReport, CalibrationPrediction,
    CohortCalibration, CompareEvalReport, DeltaSignificance, EmbeddingEvalReport,
    EvalCalibrationShape, EvalTask, InferenceAggregate, InferenceEvalReport, MetricDelta,
    MetricSignificance, PerQueryRecord, PerRecordCalibration, PerRecordPrediction, TableEvalReport,
};
