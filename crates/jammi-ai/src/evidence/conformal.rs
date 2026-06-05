//! The `conformal` evidence channel: a calibrated prediction set or interval
//! and its nominal coverage level, carried alongside the point output.
//!
//! Conformal prediction ([`crate::predict::conformal`]) emits, per row, either
//! a classification **prediction set** or a regression **interval**, each at a
//! nominal miscoverage level `alpha`. Those serving outputs ride the evidence
//! substrate exactly as `vector` and `inference` do — one channel, declared
//! columns, no new provenance machinery. This module declares the channel and
//! builds its per-batch [`ChannelContribution`]; registration and merging reuse
//! the catalog and [`super::merge_channels`] unchanged.
//!
//! The four declared columns hold both output shapes additively:
//!
//! | column           | dtype   | classification          | regression          |
//! |------------------|---------|-------------------------|---------------------|
//! | `prediction_set` | Utf8    | JSON array of class ids | null                |
//! | `lower`          | Float64 | null                    | interval lower      |
//! | `upper`          | Float64 | null                    | interval upper      |
//! | `alpha`          | Float64 | nominal level           | nominal level       |

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::error::{JammiError, Result};
use jammi_db::ChannelId;

use super::channel::ChannelContribution;

/// The channel's identifier slug.
pub const CHANNEL_NAME: &str = "conformal";

/// Priority of the `conformal` channel in the merged output's column order.
///
/// Sits after `vector` (1) and `inference` (2): a conformal set wraps an
/// inference output, so it reads naturally to the right of it.
pub const CHANNEL_PRIORITY: i32 = 3;

/// The declared schema of the `conformal` channel: `prediction_set`, `lower`,
/// `upper`, `alpha`.
///
/// Build once and pass to `catalog.channels().register(..)` to make the channel
/// available; it carries no policy and no consumer, only the columns a conformal
/// serving output writes. Errors only if the slug ever fails validation (it
/// will not — [`CHANNEL_NAME`] is a constant valid slug).
pub fn channel_spec() -> Result<ChannelSpec> {
    Ok(ChannelSpec {
        id: ChannelId::new(CHANNEL_NAME)?,
        priority: CHANNEL_PRIORITY,
        columns: vec![
            ChannelColumn {
                name: "prediction_set".into(),
                data_type: ChannelColumnType::Utf8,
            },
            ChannelColumn {
                name: "lower".into(),
                data_type: ChannelColumnType::Float64,
            },
            ChannelColumn {
                name: "upper".into(),
                data_type: ChannelColumnType::Float64,
            },
            ChannelColumn {
                name: "alpha".into(),
                data_type: ChannelColumnType::Float64,
            },
        ],
    })
}

/// One row's conformal output: a classification set or a regression interval,
/// both at level `alpha`.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalOutput {
    /// A classification prediction set — the admitted class indices.
    Set {
        /// Admitted class indices (ascending), as produced by
        /// [`crate::predict::conformal::ConformalModel::predict_set`].
        classes: Vec<usize>,
        /// Nominal miscoverage level.
        alpha: f64,
    },
    /// A regression prediction interval `[lower, upper]`.
    Interval {
        /// Interval lower bound.
        lower: f64,
        /// Interval upper bound.
        upper: f64,
        /// Nominal miscoverage level.
        alpha: f64,
    },
}

/// Build the `conformal` channel's [`ChannelContribution`] for one batch from
/// its per-row outputs.
///
/// `outputs` aligns 1:1 with the batch's rows. A `Set` row writes its admitted
/// class indices to `prediction_set` (JSON array) and leaves `lower`/`upper`
/// null; an `Interval` row writes `lower`/`upper` and leaves `prediction_set`
/// null. Both write `alpha`. The merger validates the column count, row count,
/// and dtypes against the catalog.
pub fn contribution(outputs: &[ConformalOutput]) -> Result<ChannelContribution> {
    let mut prediction_set: Vec<Option<String>> = Vec::with_capacity(outputs.len());
    let mut lower: Vec<Option<f64>> = Vec::with_capacity(outputs.len());
    let mut upper: Vec<Option<f64>> = Vec::with_capacity(outputs.len());
    let mut alpha: Vec<f64> = Vec::with_capacity(outputs.len());

    for output in outputs {
        match output {
            ConformalOutput::Set { classes, alpha: a } => {
                let json = serde_json::to_string(classes).map_err(JammiError::Json)?;
                prediction_set.push(Some(json));
                lower.push(None);
                upper.push(None);
                alpha.push(*a);
            }
            ConformalOutput::Interval {
                lower: lo,
                upper: hi,
                alpha: a,
            } => {
                prediction_set.push(None);
                lower.push(Some(*lo));
                upper.push(Some(*hi));
                alpha.push(*a);
            }
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(prediction_set)),
        Arc::new(Float64Array::from(lower)),
        Arc::new(Float64Array::from(upper)),
        Arc::new(Float64Array::from(alpha)),
    ];
    Ok(ChannelContribution {
        channel: ChannelId::new(CHANNEL_NAME)?,
        columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    #[test]
    fn spec_declares_four_columns_in_order() {
        let spec = channel_spec().unwrap();
        assert_eq!(spec.id.as_str(), "conformal");
        assert_eq!(spec.priority, CHANNEL_PRIORITY);
        let names: Vec<&str> = spec.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["prediction_set", "lower", "upper", "alpha"]);
        assert_eq!(spec.columns[0].data_type, ChannelColumnType::Utf8);
        assert_eq!(spec.columns[1].data_type, ChannelColumnType::Float64);
    }

    #[test]
    fn classification_rows_write_set_and_null_bounds() {
        let contrib = contribution(&[ConformalOutput::Set {
            classes: vec![1, 3],
            alpha: 0.1,
        }])
        .unwrap();
        assert_eq!(contrib.columns.len(), 4);
        let set = contrib.columns[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(set.value(0), "[1,3]");
        let lower = contrib.columns[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(lower.null_count(), 1);
        let alpha = contrib.columns[3]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(alpha.value(0), 0.1);
    }

    #[test]
    fn regression_rows_write_bounds_and_null_set() {
        let contrib = contribution(&[ConformalOutput::Interval {
            lower: -1.5,
            upper: 2.5,
            alpha: 0.2,
        }])
        .unwrap();
        let set = contrib.columns[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(set.null_count(), 1);
        let lower = contrib.columns[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(lower.value(0), -1.5);
        let upper = contrib.columns[2]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(upper.value(0), 2.5);
    }

    #[test]
    fn mixed_batch_keeps_rows_aligned() {
        let contrib = contribution(&[
            ConformalOutput::Set {
                classes: vec![0],
                alpha: 0.1,
            },
            ConformalOutput::Interval {
                lower: 0.0,
                upper: 1.0,
                alpha: 0.1,
            },
        ])
        .unwrap();
        assert_eq!(contrib.columns[0].len(), 2);
        let set = contrib.columns[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(set.is_valid(0));
        assert!(set.is_null(1));
    }
}
