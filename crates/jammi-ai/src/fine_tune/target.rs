//! Closed-enum dispatch over the two ways jammi-ai trains and persists a
//! LoRA adapter:
//!
//! - [`TrainingTarget::ProjectionHead`] — LoRA wraps a projection head added
//!   on top of a frozen base model's pooled embedding output. The base model
//!   produces embeddings via its own (frozen) forward; the head shifts them.
//!   Works with any [`crate::model::LoadedModel`] backbone.
//! - [`TrainingTarget::EncoderAdapters`] — LoRA is injected into the encoder's
//!   internal attention/FFN linears via [`jammi_encoders::AnyEncoder`].
//!   Requires a jammi-encoders-supported architecture (BERT family).
//!
//! The two variants share no on-disk discriminator at the safetensors level —
//! the persisted [`SavedAdapter`] enum (serde-tagged via `adapter_type`) is
//! the type-level switch.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use jammi_db::error::{JammiError, Result};
use jammi_encoders::AnyEncoder;
use jammi_lora::AdapterConfig;
use serde::{Deserialize, Serialize};

use super::lora::LoraModel;
use super::regression_loss::TargetScaler;

/// What a [`crate::fine_tune::trainer::TrainingLoop`] is training.
///
/// The `EncoderAdapters` variant payload is boxed because [`AnyEncoder`] is
/// substantially larger than a [`LoraModel`] — keeping the variants on a
/// similar size footprint is what `clippy::large_enum_variant` is asking
/// for and is the right thing here, not a band-aid.
pub enum TrainingTarget {
    /// LoRA wraps a projection head sitting on top of a frozen base model's
    /// pooled output. The head is one or two [`jammi_lora::LoraLinear`]
    /// layers: a single `projection` layer for embedding fine-tunes, or a
    /// `projection` + `classifier` pair for classification / NER.
    ProjectionHead { head: LoraModel },
    /// LoRA is injected into the encoder's internal attention/FFN linears.
    EncoderAdapters(Box<EncoderAdaptersTarget>),
}

/// Payload of [`TrainingTarget::EncoderAdapters`].
pub struct EncoderAdaptersTarget {
    /// The LoRA-injected encoder used both during training and as the
    /// blueprint inference must rebuild from the saved tensors.
    pub encoder: AnyEncoder,
    /// Metadata persisted alongside the trained A/B weights so inference
    /// can rebuild the encoder with the adapter installed.
    pub adapter_cfg: AdapterConfig,
}

impl TrainingTarget {
    /// References to every trainable tensor in this target.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::ProjectionHead { head } => head.trainable_params(),
            Self::EncoderAdapters(state) => state.encoder.trainable_params(),
        }
    }

    /// Toggle training mode. Used to disable LoRA dropout during validation
    /// passes; no-op for variants whose layers don't use dropout.
    pub fn set_training(&mut self, training: bool) {
        match self {
            Self::ProjectionHead { head } => {
                for (_, layer) in head.layers.iter_mut() {
                    layer.set_training(training);
                }
            }
            Self::EncoderAdapters(state) => state.encoder.set_training(training),
        }
    }

    /// Collect all trainable A/B tensors as a CPU-side `HashMap` ready for
    /// safetensors serialisation. Keys follow the convention each variant
    /// uses to label its trainable sites.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::ProjectionHead { head } => {
                let mut out = HashMap::new();
                for (name, layer) in &head.layers {
                    out.insert(
                        format!("{name}.lora_a"),
                        layer.lora_a.to_device(&Device::Cpu).map_err(|e| {
                            JammiError::FineTune(format!("Move lora_a to CPU: {e}"))
                        })?,
                    );
                    out.insert(
                        format!("{name}.lora_b"),
                        layer.lora_b.to_device(&Device::Cpu).map_err(|e| {
                            JammiError::FineTune(format!("Move lora_b to CPU: {e}"))
                        })?,
                    );
                }
                Ok(out)
            }
            Self::EncoderAdapters(state) => state
                .encoder
                .named_trainable_weights()
                .map_err(|e| JammiError::FineTune(format!("Encoder trainable weights: {e}"))),
        }
    }

    /// Apply a previously-saved `HashMap<key, tensor>` back into the target's
    /// trainable sites. Used to restore the best checkpoint at end of
    /// training.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        match self {
            Self::ProjectionHead { head } => {
                for (name, layer) in head.layers.iter_mut() {
                    if let Some(a) = weights.get(&format!("{name}.lora_a")) {
                        layer.lora_a = a.clone();
                    }
                    if let Some(b) = weights.get(&format!("{name}.lora_b")) {
                        layer.lora_b = b.clone();
                    }
                }
                Ok(())
            }
            Self::EncoderAdapters(state) => state
                .encoder
                .load_weights(weights)
                .map_err(|e| JammiError::FineTune(format!("Encoder load weights: {e}"))),
        }
    }

    /// Build the persisted metadata for this target.
    ///
    /// `target_scaler` and `regression_form` are the regression head's persisted
    /// de-standardisation state, present together only for a
    /// `ModelTask::Regression` projection head. The scaler is the affine the
    /// head's raw output carries; the form is the authoritative gaussian-vs-quantile
    /// signal the serving de-standardisation dispatches on (a 2-level quantile head
    /// is also width 2, so head width cannot stand in for it). Both are the one
    /// transform shared by training and serving, so they are persisted with the head.
    pub(crate) fn saved_adapter(
        &self,
        config: &super::FineTuneConfig,
        target_scaler: Option<TargetScaler>,
        regression_form: Option<crate::inference::adapter::DistributionForm>,
    ) -> SavedAdapter {
        match self {
            Self::ProjectionHead { head } => SavedAdapter::ProjectionHead(ProjectionHeadConfig {
                lora_rank: config.lora_rank,
                lora_alpha: config.lora_alpha,
                head_layers: head.layers.iter().map(|(name, _)| name.clone()).collect(),
                target_scaler,
                regression_form,
            }),
            Self::EncoderAdapters(state) => {
                SavedAdapter::EncoderAdapters(Box::new(state.adapter_cfg.clone()))
            }
        }
    }
}

/// Persisted metadata serialised into `adapter_config.json` for both adapter
/// flavours jammi-ai produces. The `adapter_type` JSON field tags which
/// variant a saved adapter is.
///
/// `EncoderAdapters` boxes its payload because [`AdapterConfig`] is
/// substantially larger than [`ProjectionHeadConfig`] (multiple `Vec`s,
/// a `HashMap`, and a string), and unboxed it would inflate every
/// `SavedAdapter` value to the larger size regardless of variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "adapter_type", rename_all = "snake_case")]
pub enum SavedAdapter {
    /// Adapter is a projection head on top of a frozen base model. The
    /// `adapter.safetensors` keys are `{head_layer}.lora_a` /
    /// `{head_layer}.lora_b` for every entry in `head_layers`.
    ProjectionHead(ProjectionHeadConfig),
    /// Adapter is a set of LoRA A/B tensors injected into an encoder's
    /// attention/FFN linears.
    EncoderAdapters(Box<AdapterConfig>),
}

/// Metadata for a projection-head adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionHeadConfig {
    /// LoRA rank used at training time.
    pub lora_rank: usize,
    /// LoRA scaling factor used at training time.
    pub lora_alpha: f64,
    /// Names of the head layers, in order. Each name keys two safetensors
    /// tensors: `{name}.lora_a` and `{name}.lora_b`. Typical shapes:
    /// - embeddings: `["projection"]`
    /// - classification: `["projection", "classifier"]`
    /// - NER: `["projection", "token_classifier"]`
    pub head_layers: Vec<String>,
    /// De-standardising affine for a regression head: `Some(μ_y, σ_y)` for a
    /// `ModelTask::Regression` head, `None` for every other projection head.
    /// Serving rebuilds it so the reloaded head emits the same raw, outcome-unit
    /// distribution parameters the trained head did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) target_scaler: Option<TargetScaler>,
    /// Predictive distribution form of a regression head: `Gaussian` or
    /// `Quantile { levels }`, set from the configured regression objective and
    /// present exactly when [`Self::target_scaler`] is. Serving dispatches the
    /// de-standardisation on this form (Gaussian de-standardises only the mean
    /// column; quantile de-standardises every column), so a 2-level quantile head
    /// — also width 2 — is de-standardised as a quantile head, not a Gaussian one.
    /// `None`/absent for every non-regression head; existing non-regression
    /// adapter configs round-trip unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) regression_form: Option<crate::inference::adapter::DistributionForm>,
}

/// The closed set of **offset-bearing** distribution heads — the heads whose
/// output column(s) carry the target's offset and therefore depend on the
/// [`TargetScaler`] reparameterisation to fit a high-offset / low-variance
/// target under AdamW (whose per-step move is ≈`lr·sign(grad)`, independent of
/// loss scale; see [`super::regression_loss::TargetScaler`]). Every variant here
/// is a head that, served zero-init, must already emit μ_y and reach its target
/// by an O(1) nudge of an O(1) z-space parameter.
///
/// The two *entry points* that produce such a head are the only two
/// offset-bearing dispatch surfaces in the engine (confirmed exhaustively — no
/// third surface): the fine-tune regression path, which dispatches on
/// [`jammi_wire::fine_tune::RegressionLoss`], and the amortized in-context
/// predictor, which dispatches on
/// [`crate::pipeline::context_predictor::PredictiveHead`]. This enum is the
/// type-level union of the heads those two enums can land on, and
/// [`StandardizableHead::for_regression_loss`] /
/// [`StandardizableHead::for_predictive_head`] are the **exhaustive, no-wildcard**
/// maps from each enum's arms onto it — so a new arm on either enum fails to
/// compile until it is enumerated here and given its standardisation contract.
///
/// The heads this enum deliberately *excludes* are offset-**invariant** by
/// construction: CoSENT / MNRL / Triplet score cosine angles, Classification-CE
/// / NER-CE score a softmax over class indices — none carry a continuous offset,
/// so none needs (or gets) a [`TargetScaler`]. Adding one of them to this enum
/// would be a category error; the union is exactly the four distributional
/// heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StandardizableHead {
    /// The fine-tune parametric Gaussian regression head `(mean, raw_std)`,
    /// trained by the `GaussianNll` / `BetaNll` / `Crps` objectives. Reached
    /// through [`jammi_wire::fine_tune::RegressionLoss`] via the
    /// `TrainingBatch::Regression` dispatch.
    RegressionGaussian,
    /// The fine-tune quantile regression head (one column per level), trained by
    /// the `Pinball` objective. Reached through
    /// [`jammi_wire::fine_tune::RegressionLoss::Pinball`].
    RegressionQuantile,
    /// The amortized in-context Gaussian head, trained by the NLL / CRPS
    /// objectives. Reached through
    /// [`crate::pipeline::context_predictor::PredictiveHead::Gaussian`].
    ContextGaussian,
    /// The amortized in-context quantile head, trained by the pinball objective.
    /// Reached through
    /// [`crate::pipeline::context_predictor::PredictiveHead::Quantile`].
    ContextQuantile,
}

impl StandardizableHead {
    /// Every offset-bearing head, in declaration order. The single source of
    /// truth for "which heads carry the standardisation contract" — the
    /// completeness test fans over this and over the two source enums, and the
    /// exhaustive `for_*` maps below keep it from drifting from the enum body.
    /// Mirrors the [`jammi_db::ModelTask::ALL`] idiom.
    pub(crate) const ALL: &'static [StandardizableHead] = &[
        StandardizableHead::RegressionGaussian,
        StandardizableHead::RegressionQuantile,
        StandardizableHead::ContextGaussian,
        StandardizableHead::ContextQuantile,
    ];

    /// Map a fine-tune [`jammi_wire::fine_tune::RegressionLoss`] arm onto the
    /// offset-bearing head it trains. **Exhaustive, no `_ =>` wildcard**: a new
    /// `RegressionLoss` arm fails to compile here until it is given its
    /// standardisation contract — the cross-crate type-level guarantee that
    /// every regression objective the wire vocabulary can carry maps to a
    /// reparameterised head. The three Gaussian objectives all train the
    /// parametric Gaussian head; the pinball objective trains the quantile head.
    pub(crate) fn for_regression_loss(loss: super::RegressionLoss) -> StandardizableHead {
        let head = match loss {
            super::RegressionLoss::GaussianNll
            | super::RegressionLoss::BetaNll { .. }
            | super::RegressionLoss::Crps => StandardizableHead::RegressionGaussian,
            super::RegressionLoss::Pinball => StandardizableHead::RegressionQuantile,
        };
        debug_assert!(
            Self::ALL.contains(&head),
            "for_regression_loss produced {head:?} absent from StandardizableHead::ALL"
        );
        head
    }

    /// Map an amortized in-context
    /// [`crate::pipeline::context_predictor::PredictiveHead`] arm onto the
    /// offset-bearing head it trains. **Exhaustive, no `_ =>` wildcard**: a new
    /// `PredictiveHead` arm fails to compile here until it is enumerated.
    pub(crate) fn for_predictive_head(
        head: &crate::pipeline::context_predictor::PredictiveHead,
    ) -> StandardizableHead {
        use crate::pipeline::context_predictor::PredictiveHead;
        match head {
            PredictiveHead::Gaussian { .. } => StandardizableHead::ContextGaussian,
            PredictiveHead::Quantile { .. } => StandardizableHead::ContextQuantile,
        }
    }

    /// Whether this head's served distribution is parametric Gaussian
    /// (`(mean, raw_std)`, only the mean column de-standardised) or quantile
    /// (every column de-standardised). This is the single gaussian-vs-quantile
    /// signal the [`TargetScaler::destandardize`] dispatch keys on, derived once
    /// from the offset-bearing head identity so a regression head and the
    /// matching context head can never disagree on how their offset is restored.
    pub(crate) fn is_gaussian(self) -> bool {
        match self {
            StandardizableHead::RegressionGaussian | StandardizableHead::ContextGaussian => true,
            StandardizableHead::RegressionQuantile | StandardizableHead::ContextQuantile => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use tempfile::tempdir;

    use super::super::regression_loss::TargetScaler;
    use super::{ProjectionHeadConfig, SavedAdapter, StandardizableHead};
    use crate::inference::adapter::DistributionForm;
    use crate::pipeline::context_predictor::{GaussianObjective, PredictiveHead};

    /// COMPLETENESS GUARD (b): every arm of the two offset-bearing source enums
    /// — [`jammi_wire::fine_tune::RegressionLoss`] and
    /// [`crate::pipeline::context_predictor::PredictiveHead`] — maps to a
    /// [`StandardizableHead`] that is listed in [`StandardizableHead::ALL`]. This
    /// is complementary to the exhaustive-match guard in the `for_*` maps (which
    /// catches a new *enum arm*): this test catches a new *loss landing on an
    /// existing head* (e.g. a fourth Gaussian objective that should map to
    /// `RegressionGaussian`) and an `ALL` slice that drifts from the enum body.
    ///
    /// The lists below are themselves exhaustive: `cargo` will not let them omit
    /// a `RegressionLoss` / `PredictiveHead` arm silently, because the `for_*`
    /// maps they feed are no-wildcard matches — so a new arm forces an edit here.
    #[test]
    fn every_offset_bearing_loss_maps_into_standardizable_head() {
        // Every RegressionLoss arm (one representative per shape; the `for_*`
        // map is exhaustive so the compiler enforces the arm set is complete).
        let regression_losses = [
            super::super::RegressionLoss::GaussianNll,
            super::super::RegressionLoss::BetaNll { beta: 0.5 },
            super::super::RegressionLoss::Crps,
            super::super::RegressionLoss::Pinball,
        ];
        for loss in regression_losses {
            let head = StandardizableHead::for_regression_loss(loss);
            assert!(
                StandardizableHead::ALL.contains(&head),
                "RegressionLoss {loss:?} mapped to {head:?}, which is missing from \
                 StandardizableHead::ALL"
            );
        }

        // Every PredictiveHead arm.
        let predictive_heads = [
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Nll { beta: 0.5 },
            },
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            },
            PredictiveHead::Quantile {
                levels: vec![0.1, 0.5, 0.9],
            },
        ];
        for head in &predictive_heads {
            let mapped = StandardizableHead::for_predictive_head(head);
            assert!(
                StandardizableHead::ALL.contains(&mapped),
                "PredictiveHead {head:?} mapped to {mapped:?}, which is missing from \
                 StandardizableHead::ALL"
            );
        }

        // The two regression objective families land on the two regression
        // heads; the two context families on the two context heads — the union
        // is exactly covered, with no offset-bearing head left unreached.
        assert_eq!(
            StandardizableHead::for_regression_loss(super::super::RegressionLoss::Crps),
            StandardizableHead::RegressionGaussian
        );
        assert_eq!(
            StandardizableHead::for_regression_loss(super::super::RegressionLoss::Pinball),
            StandardizableHead::RegressionQuantile
        );
        assert_eq!(
            StandardizableHead::for_predictive_head(&PredictiveHead::Quantile {
                levels: vec![0.5]
            }),
            StandardizableHead::ContextQuantile
        );

        // Both regression heads and both context heads are reached by some arm —
        // no variant of the union is orphaned from the dispatch.
        let reached: std::collections::HashSet<StandardizableHead> = regression_losses
            .into_iter()
            .map(StandardizableHead::for_regression_loss)
            .chain(
                predictive_heads
                    .iter()
                    .map(StandardizableHead::for_predictive_head),
            )
            .collect();
        for head in StandardizableHead::ALL {
            assert!(
                reached.contains(head),
                "StandardizableHead::{head:?} is in ALL but no offset-bearing loss arm \
                 reaches it — the union has an unreachable head"
            );
        }
    }

    /// Persist a regression `ProjectionHeadConfig` (scaler + form) through the
    /// real `save_adapter` writer, reload it through the real `load_adapter`
    /// reader, and return the reloaded `(scaler, form)`. This is the exact
    /// persistence path `forward_regression` reads on serve.
    fn round_trip_regression_head(
        scaler: TargetScaler,
        form: DistributionForm,
        head_width: usize,
        device: &Device,
    ) -> (TargetScaler, DistributionForm) {
        let saved = SavedAdapter::ProjectionHead(ProjectionHeadConfig {
            lora_rank: 4,
            lora_alpha: 8.0,
            head_layers: vec!["projection".into(), "distribution".into()],
            target_scaler: Some(scaler),
            regression_form: Some(form),
        });

        // A two-tensor weights map so `save_adapter` writes a valid safetensors;
        // the `lora_b` rows match the head width so the persisted head is coherent.
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "distribution.lora_a".to_string(),
            Tensor::zeros((4, 8), candle_core::DType::F32, device).unwrap(),
        );
        weights.insert(
            "distribution.lora_b".to_string(),
            Tensor::zeros((head_width, 4), candle_core::DType::F32, device).unwrap(),
        );

        let dir = tempdir().unwrap();
        jammi_lora::save_adapter(dir.path(), &weights, &saved).unwrap();
        let (reloaded, _tensors): (SavedAdapter, _) =
            jammi_lora::load_adapter(dir.path(), device).unwrap();

        match reloaded {
            SavedAdapter::ProjectionHead(cfg) => (
                cfg.target_scaler
                    .expect("the reloaded regression head must carry its persisted target scaler"),
                cfg.regression_form
                    .expect("the reloaded regression head must carry its persisted form"),
            ),
            SavedAdapter::EncoderAdapters(_) => panic!("round-trip changed the adapter variant"),
        }
    }

    /// The calendar-year scaler shape (μ_y ≈ 2017, σ_y ≈ 2): high offset, low
    /// variance — the case where a dropped or mis-dispatched de-standardisation is
    /// glaring (a raw column sits near 0, the de-standardised one near 2017).
    fn calendar_year_scaler(device: &Device) -> TargetScaler {
        let years = Tensor::from_vec(
            vec![2014.0f32, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0],
            (7,),
            device,
        )
        .unwrap();
        TargetScaler::from_targets(&years).unwrap()
    }

    /// A regression head's de-standardising affine must survive the
    /// save → reload round-trip through the adapter directory, so a served head
    /// emits the target offset rather than the zero-init z-space value.
    ///
    /// Build a `ProjectionHeadConfig` carrying a non-trivial `TargetScaler`
    /// (μ_y ≈ 2017 — the calendar-year shape) and the Gaussian form, persist and
    /// reload it through the real adapter writer/reader, then de-standardise a
    /// ZERO z-space head through the reloaded form exactly as `forward_regression`
    /// does. The served mean must come back at μ_y (≈2017), NOT 0.
    #[test]
    fn regression_target_scaler_survives_adapter_round_trip() {
        let device = Device::Cpu;
        let true_mean = 2017.0f32;
        let (scaler, form) = round_trip_regression_head(
            calendar_year_scaler(&device),
            DistributionForm::Gaussian,
            2,
            &device,
        );

        // The served head: a ZERO z-space Gaussian head (mean column 0, raw_std 0),
        // de-standardised exactly as `forward_regression` does on serve — through
        // the persisted form, not a head-width heuristic.
        let z_head = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
        let served = scaler.destandardize(&z_head, &form).unwrap();
        let served_mean = served.to_vec2::<f32>().unwrap()[0][0];

        assert!(
            (served_mean - true_mean).abs() < 1.0,
            "reloaded scaler must de-standardise the served mean to μ_y ≈ {true_mean}, \
             got {served_mean} (a dropped scaler would serve ≈0)"
        );
        assert!(
            served_mean.abs() > 100.0,
            "served mean {served_mean} collapsed toward the zero-init z-space value — \
             the persisted de-standardisation was lost"
        );
    }

    /// The defect: a 2-LEVEL quantile head (`quantile_levels = [0.25, 0.75]`) is
    /// width 2, exactly like a Gaussian head. The old serving dispatch keyed
    /// gaussian-vs-quantile on head WIDTH, so this head wrongly hit the Gaussian
    /// branch — which de-standardises only column 0 and leaves column 1 (the 0.75
    /// quantile) RAW (near 0), so the served upper quantile was wrong by ≈μ_y.
    ///
    /// With the dispatch on the persisted `DistributionForm`, a width-2 quantile
    /// head de-standardises EVERY column. This test fails before the fix (column 1
    /// served raw, ≈0) and passes after (both columns ≈μ_y).
    #[test]
    fn two_level_quantile_head_destandardises_both_columns_after_round_trip() {
        let device = Device::Cpu;
        let true_mean = 2017.0f32;
        let (scaler, form) = round_trip_regression_head(
            calendar_year_scaler(&device),
            DistributionForm::Quantile {
                levels: vec![0.25, 0.75],
            },
            2,
            &device,
        );
        // The reloaded form must be the quantile form — the signal width alone
        // could not recover.
        assert!(
            matches!(form, DistributionForm::Quantile { .. }),
            "a 2-level quantile head must reload as the quantile form, not Gaussian"
        );

        // A ZERO z-space quantile head: both columns are z = 0, so both must
        // de-standardise to μ_y. Dispatched through the persisted form exactly as
        // `forward_regression` does.
        let z_head = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
        let served = scaler.destandardize(&z_head, &form).unwrap();
        let row = &served.to_vec2::<f32>().unwrap()[0];

        for (col, &q) in row.iter().enumerate() {
            assert!(
                (q - true_mean).abs() < 1.0,
                "quantile column {col} must de-standardise to μ_y ≈ {true_mean}, got {q}; \
                 the 0.75 column left RAW (≈0) is the width-heuristic defect"
            );
        }
    }

    /// A 3-level quantile head round-trips and de-standardises every column —
    /// guards that the form-dispatch fix did not regress the wider quantile case.
    #[test]
    fn three_level_quantile_head_destandardises_all_columns_after_round_trip() {
        let device = Device::Cpu;
        let true_mean = 2017.0f32;
        let (scaler, form) = round_trip_regression_head(
            calendar_year_scaler(&device),
            DistributionForm::Quantile {
                levels: vec![0.1, 0.5, 0.9],
            },
            3,
            &device,
        );

        let z_head = Tensor::zeros((1, 3), candle_core::DType::F32, &device).unwrap();
        let served = scaler.destandardize(&z_head, &form).unwrap();
        let row = &served.to_vec2::<f32>().unwrap()[0];
        assert_eq!(
            row.len(),
            3,
            "the served quantile head keeps all three levels"
        );
        for (col, &q) in row.iter().enumerate() {
            assert!(
                (q - true_mean).abs() < 1.0,
                "quantile column {col} must de-standardise to μ_y ≈ {true_mean}, got {q}"
            );
        }
    }
}
