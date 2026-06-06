//! The `uncertainty` evidence channel (S18): a predictive distribution carried
//! alongside a point output.
//!
//! A distributional regression head ([`crate::inference::adapter::DistributionAdapter`])
//! emits, per row, either a parametric Gaussian `(predicted_mean, predicted_std)`
//! or a set of quantiles. Those serving outputs ride the evidence substrate
//! exactly as `vector`, `inference`, and `conformal` do — one channel, declared
//! columns, no new provenance machinery. This module declares the channel and
//! builds its per-batch [`ChannelContribution`]; registration (via
//! `catalog.channels().register(..)`, as the `conformal` sibling does — not a
//! catalog migration) and merging reuse the catalog and
//! [`super::merge_channels`] unchanged.
//!
//! The four declared columns hold both output forms additively:
//!
//! | column           | dtype   | Gaussian form          | quantile form        |
//! |------------------|---------|------------------------|----------------------|
//! | `predicted_mean` | Float64 | predictive mean        | null                 |
//! | `predicted_std`  | Float64 | predictive std (>0)    | null                 |
//! | `quantiles`      | Utf8    | null                   | JSON `[[level,value]]` |
//! | `context_ref`    | Utf8    | S16 context provenance | S16 context provenance |
//!
//! `context_ref` is the data-driven provenance hook: when the prediction was
//! conditioned on an S16 context set, it records which rows informed it (a JSON
//! id list). It is null for an unconditioned prediction — the channel never
//! fabricates provenance.
//!
//! A parametric Gaussian carries **aleatoric** uncertainty only; the channel
//! does not claim epistemic coverage (that is S17/NP4). The columns describe
//! exactly what the head emits.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::error::{JammiError, Result};
use jammi_db::ChannelId;

use super::channel::ChannelContribution;

/// The channel's identifier slug.
pub const CHANNEL_NAME: &str = "uncertainty";

/// Priority of the `uncertainty` channel in the merged output's column order.
///
/// Sits after `vector` (1), `inference` (2), and `conformal` (3): a predictive
/// distribution annotates an inference output, reading naturally to the right of
/// the point columns and the conformal wrapper.
pub const CHANNEL_PRIORITY: i32 = 4;

/// The declared schema of the `uncertainty` channel: `predicted_mean`,
/// `predicted_std`, `quantiles`, `context_ref`.
///
/// Build once and pass to `catalog.channels().register(..)` to make the channel
/// available; it carries no policy and no consumer, only the columns a
/// distributional serving output writes. Errors only if the slug ever fails
/// validation (it will not — [`CHANNEL_NAME`] is a constant valid slug).
pub fn channel_spec() -> Result<ChannelSpec> {
    Ok(ChannelSpec {
        id: ChannelId::new(CHANNEL_NAME)?,
        priority: CHANNEL_PRIORITY,
        columns: vec![
            ChannelColumn {
                name: "predicted_mean".into(),
                data_type: ChannelColumnType::Float64,
            },
            ChannelColumn {
                name: "predicted_std".into(),
                data_type: ChannelColumnType::Float64,
            },
            ChannelColumn {
                name: "quantiles".into(),
                data_type: ChannelColumnType::Utf8,
            },
            ChannelColumn {
                name: "context_ref".into(),
                data_type: ChannelColumnType::Utf8,
            },
        ],
    })
}

/// One row's predictive distribution: a parametric Gaussian or a set of
/// quantiles, optionally with the S16 context provenance that informed it.
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyOutput {
    /// A parametric Gaussian `Normal(mean, std)`.
    Gaussian {
        /// Predictive mean.
        mean: f64,
        /// Predictive standard deviation (strictly positive — softplus + floor).
        std: f64,
        /// Optional S16 context provenance (ids of the rows that informed the
        /// prediction). `None` for an unconditioned prediction.
        context_ref: Option<Vec<String>>,
    },
    /// A set of predictive quantiles as `(level, value)` pairs, ascending in
    /// level. Coherent (non-crossing) — the adapter sorts post-hoc.
    Quantiles {
        /// `(level, value)` pairs, ascending in level.
        levels: Vec<(f64, f64)>,
        /// Optional S16 context provenance. `None` for an unconditioned
        /// prediction.
        context_ref: Option<Vec<String>>,
    },
}

/// Build the `uncertainty` channel's [`ChannelContribution`] for one batch from
/// its per-row outputs.
///
/// `outputs` aligns 1:1 with the batch's rows. A `Gaussian` row writes
/// `predicted_mean`/`predicted_std` and leaves `quantiles` null; a `Quantiles`
/// row writes the `quantiles` JSON and leaves the mean/std null. Both write
/// `context_ref` (null when the prediction was unconditioned). The merger
/// validates the column count, row count, and dtypes against the catalog.
pub fn contribution(outputs: &[UncertaintyOutput]) -> Result<ChannelContribution> {
    let mut mean: Vec<Option<f64>> = Vec::with_capacity(outputs.len());
    let mut std: Vec<Option<f64>> = Vec::with_capacity(outputs.len());
    let mut quantiles: Vec<Option<String>> = Vec::with_capacity(outputs.len());
    let mut context_ref: Vec<Option<String>> = Vec::with_capacity(outputs.len());

    for output in outputs {
        match output {
            UncertaintyOutput::Gaussian {
                mean: m,
                std: s,
                context_ref: ctx,
            } => {
                mean.push(Some(*m));
                std.push(Some(*s));
                quantiles.push(None);
                context_ref.push(encode_context(ctx)?);
            }
            UncertaintyOutput::Quantiles {
                levels,
                context_ref: ctx,
            } => {
                let json = serde_json::to_string(levels).map_err(JammiError::Json)?;
                mean.push(None);
                std.push(None);
                quantiles.push(Some(json));
                context_ref.push(encode_context(ctx)?);
            }
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(mean)),
        Arc::new(Float64Array::from(std)),
        Arc::new(StringArray::from(quantiles)),
        Arc::new(StringArray::from(context_ref)),
    ];
    Ok(ChannelContribution {
        channel: ChannelId::new(CHANNEL_NAME)?,
        columns,
    })
}

/// Encode the optional S16 context provenance as a JSON id list, or `None`.
fn encode_context(ctx: &Option<Vec<String>>) -> Result<Option<String>> {
    match ctx {
        Some(ids) => Ok(Some(serde_json::to_string(ids).map_err(JammiError::Json)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    #[test]
    fn spec_declares_four_columns_in_order() {
        let spec = channel_spec().unwrap();
        assert_eq!(spec.id.as_str(), "uncertainty");
        assert_eq!(spec.priority, CHANNEL_PRIORITY);
        let names: Vec<&str> = spec.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "predicted_mean",
                "predicted_std",
                "quantiles",
                "context_ref"
            ]
        );
        assert_eq!(spec.columns[0].data_type, ChannelColumnType::Float64);
        assert_eq!(spec.columns[2].data_type, ChannelColumnType::Utf8);
    }

    #[test]
    fn gaussian_rows_write_mean_std_and_null_quantiles() {
        let contrib = contribution(&[UncertaintyOutput::Gaussian {
            mean: 0.7,
            std: 0.2,
            context_ref: None,
        }])
        .unwrap();
        assert_eq!(contrib.columns.len(), 4);
        let mean = contrib.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(mean.value(0), 0.7);
        let std = contrib.columns[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(std.value(0), 0.2);
        let quant = contrib.columns[2]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(quant.is_null(0));
        // No context: provenance is null, never fabricated.
        let ctx = contrib.columns[3]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(ctx.is_null(0));
    }

    #[test]
    fn quantile_rows_write_json_and_null_mean() {
        let contrib = contribution(&[UncertaintyOutput::Quantiles {
            levels: vec![(0.05, -1.0), (0.5, 0.0), (0.95, 1.0)],
            context_ref: None,
        }])
        .unwrap();
        let mean = contrib.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(mean.is_null(0));
        let quant = contrib.columns[2]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(quant.value(0), "[[0.05,-1.0],[0.5,0.0],[0.95,1.0]]");
    }

    #[test]
    fn context_provenance_records_informing_rows() {
        let contrib = contribution(&[UncertaintyOutput::Gaussian {
            mean: 0.0,
            std: 1.0,
            context_ref: Some(vec!["row-a".into(), "row-b".into()]),
        }])
        .unwrap();
        let ctx = contrib.columns[3]
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ctx.value(0), "[\"row-a\",\"row-b\"]");
    }
}
