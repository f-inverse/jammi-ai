//! What a fine-tune trains, and what it persists: an encoder that trains
//! frozen or with LoRA inside it, and a head over its pooled output when the
//! task needs one.
//!
//! - [`TrainingTarget::FrozenEncoder`] — the base model's own forward produces
//!   the pooled output, unchanged, and a LoRA head over it carries the whole
//!   adaptation. Works with any [`crate::model::LoadedModel`] backbone.
//! - [`TrainingTarget::AdaptedEncoder`] — LoRA inside the encoder's linears
//!   ([`jammi_encoders::AnyEncoder`]), and a head over its pooled output for a
//!   classification or regression task.
//!
//! A head is an ordered list of [`jammi_lora::LoraLinear`] layers applied, in
//! order, to the pooled output — by training and by serving alike. The
//! persisted [`SavedAdapter`] records the encoder's adapters and the head
//! together with what the head's output means ([`HeadOutput`]).

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
/// The `AdaptedEncoder` payload is boxed because [`AnyEncoder`] is
/// substantially larger than a [`LoraModel`] — keeping the variants on a
/// similar size footprint is what `clippy::large_enum_variant` is asking
/// for.
pub enum TrainingTarget {
    /// The encoder trains frozen; `head` carries the adaptation: a
    /// `projection` layer for an embedding task, or a `projection` and a
    /// task layer (`classifier`, `distribution`) for a classification or
    /// regression task.
    FrozenEncoder { head: LoraModel },
    /// LoRA inside the encoder's linears, and the task's head when it has one.
    AdaptedEncoder(Box<AdaptedEncoderTarget>),
}

/// Payload of [`TrainingTarget::AdaptedEncoder`].
pub struct AdaptedEncoderTarget {
    /// The LoRA-injected encoder used both during training and as the
    /// blueprint inference must rebuild from the saved tensors.
    pub encoder: AnyEncoder,
    /// Metadata persisted alongside the trained A/B weights so inference
    /// can rebuild the encoder with the adapter installed.
    pub adapter_cfg: AdapterConfig,
    /// The task's head over the adapted encoder's pooled output — the same
    /// layers a frozen encoder's head of that task carries; `None` for an
    /// embedding task, whose output is the pooled encoder itself.
    pub head: Option<LoraModel>,
}

/// The A/B tensors of every head layer, keyed `{layer}.lora_a` /
/// `{layer}.lora_b`, on the CPU for serialisation.
fn head_weights(head: &LoraModel) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (name, layer) in &head.layers {
        for (suffix, tensor) in [("lora_a", &layer.lora_a), ("lora_b", &layer.lora_b)] {
            out.insert(
                format!("{name}.{suffix}"),
                tensor.to_device(&Device::Cpu).map_err(|e| {
                    JammiError::FineTune(format!("Move {name}.{suffix} to CPU: {e}"))
                })?,
            );
        }
    }
    Ok(out)
}

/// Each head layer's dropout-stream position, keyed `{layer}.dropout`; a
/// layer with no dropout contributes no entry.
fn head_dropout_positions(head: &LoraModel) -> Result<HashMap<String, u64>> {
    let mut out = HashMap::new();
    for (name, layer) in &head.layers {
        if let Some(pos) = layer
            .dropout_position()
            .map_err(|e| JammiError::FineTune(format!("dropout position: {e}")))?
        {
            out.insert(format!("{name}.dropout"), pos);
        }
    }
    Ok(out)
}

impl TrainingTarget {
    /// The head over the pooled output, when this target trains one.
    pub fn head(&self) -> Option<&LoraModel> {
        match self {
            Self::FrozenEncoder { head } => Some(head),
            Self::AdaptedEncoder(state) => state.head.as_ref(),
        }
    }

    fn head_mut(&mut self) -> Option<&mut LoraModel> {
        match self {
            Self::FrozenEncoder { head } => Some(head),
            Self::AdaptedEncoder(state) => state.head.as_mut(),
        }
    }

    /// References to every trainable tensor in this target: the encoder's
    /// adapters, then the head's layers.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let encoder = match self {
            Self::FrozenEncoder { .. } => Vec::new(),
            Self::AdaptedEncoder(state) => state.encoder.trainable_params(),
        };
        encoder
            .into_iter()
            .chain(
                self.head()
                    .into_iter()
                    .flat_map(LoraModel::trainable_params),
            )
            .collect()
    }

    /// Toggle training mode. Used to disable LoRA dropout during validation
    /// passes.
    pub fn set_training(&mut self, training: bool) {
        if let Self::AdaptedEncoder(state) = self {
            state.encoder.set_training(training);
        }
        if let Some(head) = self.head_mut() {
            for (_, layer) in head.layers.iter_mut() {
                layer.set_training(training);
            }
        }
    }

    /// Whether a training forward draws dropout at the trainable sites —
    /// `jammi_lora::LoraLinear::set_dropout`: `false` keeps the tape but draws
    /// no mask, the forward a gradient cache's two encodes of one row need.
    pub fn set_dropout(&mut self, enabled: bool) {
        if let Self::AdaptedEncoder(state) = self {
            state.encoder.set_dropout(enabled);
        }
        if let Some(head) = self.head_mut() {
            for (_, layer) in head.layers.iter_mut() {
                layer.set_dropout(enabled);
            }
        }
    }

    /// Collect all trainable A/B tensors as a CPU-side `HashMap` ready for
    /// safetensors serialisation: the encoder's adapters under their
    /// checkpoint paths (`layer.{n}.query.lora_a`, …), the head's under its
    /// layer names (`projection.lora_a`, …) — two disjoint key spaces.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = match self {
            Self::FrozenEncoder { .. } => HashMap::new(),
            Self::AdaptedEncoder(state) => state
                .encoder
                .named_trainable_weights()
                .map_err(|e| JammiError::FineTune(format!("Encoder trainable weights: {e}")))?,
        };
        if let Some(head) = self.head() {
            out.extend(head_weights(head)?);
        }
        Ok(out)
    }

    /// Apply a previously-saved `HashMap<key, tensor>` back into the target's
    /// trainable sites, in place: the trainable leaves keep their identity
    /// (see `jammi_lora::LoraLinear::load_weights`), so a loop that trains
    /// again after a restore still moves the weights the sites read. Used to
    /// restore the best checkpoint at end of training.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Self::AdaptedEncoder(state) = self {
            state
                .encoder
                .load_weights(weights)
                .map_err(|e| JammiError::FineTune(format!("Encoder load weights: {e}")))?;
        }
        if let Some(head) = self.head_mut() {
            for (name, layer) in head.layers.iter_mut() {
                layer
                    .load_weights(
                        weights.get(&format!("{name}.lora_a")),
                        weights.get(&format!("{name}.lora_b")),
                    )
                    .map_err(|e| JammiError::FineTune(format!("Head load weights: {e}")))?;
            }
        }
        Ok(())
    }

    /// Per-layer dropout-stream positions keyed `{name}.dropout`, the resume
    /// state for the adapter's dropout: a resumed run replays each stream to the
    /// position it held at the persisted epoch boundary, so its next training
    /// forwards draw the same masks the uninterrupted run drew. Layers without
    /// dropout (`lora_dropout == 0`) contribute no entry.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>> {
        let mut out = match self {
            Self::FrozenEncoder { .. } => HashMap::new(),
            Self::AdaptedEncoder(state) => state
                .encoder
                .dropout_positions()
                .map_err(|e| JammiError::FineTune(format!("Encoder dropout positions: {e}")))?,
        };
        if let Some(head) = self.head() {
            out.extend(head_dropout_positions(head)?);
        }
        Ok(out)
    }

    /// `layer_id` — the Philox
    /// counter slot carrying a layer's identity — is a 32-bit HASH of its
    /// fully-qualified name, the only place two distinct sites in this
    /// target CAN collide; a collision means the two sites would share
    /// every dropout mask silently (no error, correlated dropout,
    /// forever). This asserts the induced `layer_id` set has the same
    /// cardinality as this target's own dropout-mask-bearing layer names
    /// — i.e. the hash is injective over THIS run's actual name set — via
    /// [`jammi_lora::assert_no_layer_id_collisions`], refusing with a
    /// typed error naming both colliding sites otherwise.
    ///
    /// Reuses [`Self::dropout_positions`]'s own key set (stripped of its
    /// `.dropout` suffix) rather than re-deriving the name list: those
    /// keys are already exactly the fully-qualified names each layer's
    /// `DropoutMasks` was constructed with (the same string
    /// `restore_dropout_positions` matches back against), so this is the
    /// SAME "run's DropoutMasks set" resume already depends on being
    /// named consistently, not a new assumption.
    pub fn assert_dropout_layer_ids_are_collision_free(&self) -> Result<()> {
        let positions = self.dropout_positions()?;
        let names: Vec<&str> = positions
            .keys()
            .map(|k| k.strip_suffix(".dropout").unwrap_or(k.as_str()))
            .collect();
        jammi_lora::assert_no_layer_id_collisions(names)
            .map_err(|e| JammiError::FineTune(format!("layer_id collision guard: {e}")))
    }

    /// Restore each layer's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. A missing key leaves that layer's
    /// stream at the origin (the from-scratch position), which is correct for a
    /// layer that had no dropout when the checkpoint was taken.
    pub fn restore_dropout_positions(&self, positions: &HashMap<String, u64>) -> Result<()> {
        if let Self::AdaptedEncoder(state) = self {
            state
                .encoder
                .restore_dropout_positions(positions)
                .map_err(|e| {
                    JammiError::FineTune(format!("Encoder restore dropout positions: {e}"))
                })?;
        }
        if let Some(head) = self.head() {
            for (name, layer) in &head.layers {
                if let Some(pos) = positions.get(&format!("{name}.dropout")) {
                    layer.restore_dropout_position(*pos).map_err(|e| {
                        JammiError::FineTune(format!("restore dropout position: {e}"))
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Build the persisted metadata for this target. `output` is what the
    /// head's output means — the label vocabulary of a classifier, the
    /// de-standardising scaler and form of a distribution head — which the
    /// trainer knows once the data is read; it is recorded with the head and
    /// unused by a target that trains none. Refused when the trained head's
    /// layers are not the ones `output` names: serving rebuilds the head from
    /// `output` alone ([`HeadOutput::layers`]).
    pub(crate) fn saved_adapter(
        &self,
        config: &super::FineTuneConfig,
        output: HeadOutput,
    ) -> Result<SavedAdapter> {
        let head_config = |head: &LoraModel, output: HeadOutput| {
            let trained: Vec<&str> = head.layers.iter().map(|(name, _)| name.as_str()).collect();
            let named: Vec<&str> = output.layers().iter().map(|layer| layer.name()).collect();
            if trained != named {
                return Err(JammiError::FineTune(format!(
                    "the trained head's layers {trained:?} are not the {named:?} its output \
                     ({output:?}) is served through"
                )));
            }
            Ok(HeadConfig {
                lora_rank: config.lora_rank,
                lora_alpha: config.lora_alpha,
                use_rslora: config.use_rslora,
                output,
            })
        };
        Ok(match self {
            Self::FrozenEncoder { head } => SavedAdapter::Frozen {
                head: head_config(head, output)?,
            },
            Self::AdaptedEncoder(state) => SavedAdapter::Adapted {
                adapters: Box::new(state.adapter_cfg.clone()),
                head: state
                    .head
                    .as_ref()
                    .map(|head| head_config(head, output))
                    .transpose()?,
            },
        })
    }
}

/// Persisted metadata serialised into `adapter_config.json`: the encoder's
/// adapters when it trained them, and the head over its pooled output when
/// the task trained one. The `encoder` JSON field tags which.
///
/// `Adapted` boxes its adapters because [`AdapterConfig`] is substantially
/// larger than [`HeadConfig`], and unboxed it would inflate every
/// `SavedAdapter` value to the larger size regardless of variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "encoder", rename_all = "snake_case")]
pub enum SavedAdapter {
    /// The encoder trained frozen; the head carries the adaptation.
    Frozen {
        /// The head over the frozen encoder's pooled output.
        head: HeadConfig,
    },
    /// LoRA inside the encoder, and the task's head when it trained one.
    Adapted {
        /// The encoder's LoRA adapters.
        adapters: Box<AdapterConfig>,
        /// The task's head over the adapted encoder's pooled output; `None`
        /// for an embedding task.
        head: Option<HeadConfig>,
    },
}

impl SavedAdapter {
    /// The head over the pooled output, when the adapter carries one.
    pub fn head(&self) -> Option<&HeadConfig> {
        match self {
            Self::Frozen { head } => Some(head),
            Self::Adapted { head, .. } => head.as_ref(),
        }
    }

    /// The encoder's adapters, when it trained them.
    pub fn adapters(&self) -> Option<&AdapterConfig> {
        match self {
            Self::Frozen { .. } => None,
            Self::Adapted { adapters, .. } => Some(adapters),
        }
    }
}

/// A persisted head: the LoRA scaling its layers trained at, and what its
/// output means — which names its layers ([`HeadOutput::layers`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadConfig {
    /// LoRA rank used at training time.
    pub lora_rank: usize,
    /// LoRA scaling factor used at training time.
    pub lora_alpha: f64,
    /// Whether training scaled the adapter with rSLoRA (`alpha / sqrt(rank)`)
    /// instead of the vanilla `alpha / rank` — see `jammi_lora::lora_scaling`.
    pub use_rslora: bool,
    /// What the head's output means.
    pub output: HeadOutput,
}

/// What a head's output means, and so which layers it applies and how
/// serving turns its output into a result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadOutput {
    /// An embedding: the head's output is the served vector.
    Embedding,
    /// Class scores over `labels`, in class-index order — the vocabulary the
    /// training rows' labels sorted into.
    Classes {
        /// The class names, indexed as the head's output columns are.
        labels: Vec<String>,
    },
    /// A predictive distribution's raw parameters, in standardized space:
    /// serving de-standardises them with `scaler` under `form`.
    Distribution {
        /// The de-standardising affine of the training targets.
        scaler: TargetScaler,
        /// Gaussian or quantile: the authoritative signal the
        /// de-standardisation dispatches on (a 2-level quantile head is also
        /// width 2, so head width cannot stand in for it).
        form: jammi_datafusion::inference::adapter::DistributionForm,
    },
}

impl HeadOutput {
    /// The head's layers, in the order they apply to the pooled output: the
    /// `projection`, then the task layer an output other than the embedding
    /// itself needs.
    pub fn layers(&self) -> &'static [HeadLayer] {
        match self {
            Self::Embedding => &[HeadLayer::Projection],
            Self::Classes { .. } => &[HeadLayer::Projection, HeadLayer::Classifier],
            Self::Distribution { .. } => &[HeadLayer::Projection, HeadLayer::Distribution],
        }
    }
}

/// One layer of a head. Its [`Self::name`] keys its two safetensors tensors
/// (`{name}.lora_a`, `{name}.lora_b`) and its [`Self::base`] is the frozen
/// map its LoRA delta adds to — one definition the trainer builds and
/// serving rebuilds from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadLayer {
    /// `hidden → hidden` over the pooled output, starting as the identity.
    Projection,
    /// `hidden → classes`, starting at zero.
    Classifier,
    /// `hidden → distribution parameters`, starting at zero.
    Distribution,
}

impl HeadLayer {
    /// The layer's name — its safetensors key prefix.
    pub fn name(self) -> &'static str {
        match self {
            Self::Projection => "projection",
            Self::Classifier => "classifier",
            Self::Distribution => "distribution",
        }
    }

    /// The frozen `input → out` map the layer's LoRA delta adds to: the
    /// identity for the projection, so an untrained head serves the pooled
    /// output unchanged; zeros for a task layer, whose whole map is learned
    /// and starts at zero. The projection's `out` must equal its `input`.
    pub fn base(
        self,
        out: usize,
        input: usize,
        device: &Device,
    ) -> candle_core::Result<candle_nn::Linear> {
        let weight = match self {
            Self::Projection if out == input => {
                Tensor::eye(input, candle_core::DType::F32, device)?
            }
            Self::Projection => candle_core::bail!(
                "a projection maps the pooled output to itself; {input} → {out} is not one"
            ),
            Self::Classifier | Self::Distribution => {
                Tensor::zeros((out, input), candle_core::DType::F32, device)?
            }
        };
        Ok(candle_nn::Linear::new(weight, None))
    }
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
/// scores a softmax over class indices — none carry a continuous offset,
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
    /// Mirrors the [`jammi_datafusion::ModelTask::ALL`] idiom.
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
    use super::{HeadConfig, HeadLayer, HeadOutput, SavedAdapter, StandardizableHead};
    use crate::pipeline::context_predictor::{GaussianObjective, PredictiveHead};
    use jammi_datafusion::inference::adapter::DistributionForm;

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

    /// A head config with `output`, at the rank and alpha the tests persist.
    fn head(output: HeadOutput, use_rslora: bool) -> HeadConfig {
        HeadConfig {
            lora_rank: 4,
            lora_alpha: 8.0,
            use_rslora,
            output,
        }
    }

    /// Persist a regression head (scaler + form) through the real
    /// `save_adapter` writer, reload it through the real `load_adapter` reader,
    /// and return the reloaded `(scaler, form)`. This is the exact persistence
    /// path regression serving reads.
    fn round_trip_regression_head(
        scaler: TargetScaler,
        form: DistributionForm,
        head_width: usize,
        device: &Device,
    ) -> (TargetScaler, DistributionForm) {
        let saved = SavedAdapter::Frozen {
            head: head(HeadOutput::Distribution { scaler, form }, false),
        };

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

        match reloaded.head().map(|head| head.output.clone()) {
            Some(HeadOutput::Distribution { scaler, form }) => (scaler, form),
            other => panic!("the reloaded regression head lost its distribution output: {other:?}"),
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
    /// Build a head config carrying a non-trivial `TargetScaler`
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

    /// A 2-LEVEL quantile head (`quantile_levels = [0.25, 0.75]`) is width 2,
    /// exactly like a Gaussian head, so serving must dispatch on the persisted
    /// `DistributionForm`, never on head WIDTH. A width-keyed dispatch sends this
    /// head down the Gaussian branch, which de-standardises only column 0 and
    /// leaves column 1 (the 0.75 quantile) RAW (≈0), wrong by ≈μ_y; the correct
    /// dispatch de-standardises EVERY column (both ≈μ_y).
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

    // ── `use_rslora` is a persisted, required field ─────────────────────────────

    /// `use_rslora` decides the scale serving applies to every head layer, so a
    /// config that does not record it cannot be served faithfully: it is
    /// refused at load, never read as either value.
    #[test]
    fn a_head_config_without_use_rslora_is_refused() {
        let json = r#"{"encoder":"frozen","head":{"lora_rank":16,"lora_alpha":32.0,
            "output":{"kind":"embedding"}}}"#;
        let err = serde_json::from_str::<SavedAdapter>(json).unwrap_err();
        assert!(err.to_string().contains("use_rslora"), "{err}");
    }

    /// Every serialized head config carries `use_rslora` explicitly and
    /// round-trips `true` and `false` distinctly.
    #[test]
    fn use_rslora_is_always_serialized_and_round_trips_both_values() {
        for use_rslora in [true, false] {
            let cfg = head(HeadOutput::Embedding, use_rslora);
            let json = serde_json::to_string(&cfg).unwrap();
            assert!(json.contains("\"use_rslora\""), "{json}");
            let reloaded: HeadConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(reloaded.use_rslora, use_rslora);
        }
    }

    /// An adapted encoder's classification head persists beside its adapters,
    /// with the class names serving labels its outputs by — the shape a
    /// classification fine-tune is served from.
    #[test]
    fn an_adapted_encoder_persists_its_classification_head_and_labels() {
        let adapters: jammi_lora::AdapterConfig = serde_json::from_value(serde_json::json!({
            "model_type": "bert",
            "target_modules": ["query", "value"],
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "use_rslora": false
        }))
        .unwrap();
        let saved = SavedAdapter::Adapted {
            adapters: Box::new(adapters),
            head: Some(head(
                HeadOutput::Classes {
                    labels: vec!["cs.AI".into(), "cs.LG".into()],
                },
                false,
            )),
        };
        let reloaded: SavedAdapter =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert!(reloaded.adapters().is_some());
        let head = reloaded.head().expect("the classification head persists");
        assert_eq!(
            head.output.layers(),
            [HeadLayer::Projection, HeadLayer::Classifier]
        );
        assert!(
            matches!(&head.output, HeadOutput::Classes { labels } if labels == &["cs.AI", "cs.LG"]),
            "{:?}",
            head.output
        );
    }
}
