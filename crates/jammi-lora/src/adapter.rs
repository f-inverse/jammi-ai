//! Persisted adapter metadata (`adapter_config.json` contents).

use std::collections::HashMap;

use jammi_numerics::ComputePrecision;
use serde::{Deserialize, Serialize};

use crate::config::LoraBuildConfig;

/// Which tower of a MULTI-tower checkpoint an adapter installs on.
///
/// A single-tower family (BERT, DistilBERT, ModernBERT) has nothing to
/// discriminate, so its adapters carry `None` — see
/// [`AdapterConfig::tower`]. A dual/triple-tower checkpoint (an OpenCLIP
/// text+vision pair, a CLAP text+audio pair) has one base architecture id
/// but two independently adaptable towers, so `model_type` alone cannot say
/// where the weights belong. This field says it.
///
/// Serialised in `snake_case` (`"text"`, `"vision"`, `"audio"`), the same
/// casing convention every other string in `adapter_config.json` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tower {
    /// The text tower (an OpenCLIP text transformer, a CLAP text encoder).
    Text,
    /// The vision tower (an OpenCLIP vision transformer).
    Vision,
    /// The audio tower (a CLAP HTSAT-Swin audio encoder).
    Audio,
}

/// Metadata describing a LoRA adapter injected into an encoder's internal
/// attention/FFN linears.
///
/// Persisted as JSON alongside `adapter.safetensors`. Discrimination between
/// different *kinds* of adapters (e.g. an external projection head vs. these
/// internal adapters) is a concern for the caller; this struct describes the
/// internal-adapter case only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// The base ARCHITECTURE id, as the shared encoder-family predicate
    /// (`jammi_ai::model::arch::EncoderFamily`) names it: `"bert"`,
    /// `"distilbert"`, `"modernbert"`, `"open_clip"`, `"clap_audio_model"`.
    ///
    /// Three of those are HuggingFace's own `config.json` `model_type`
    /// values and `"clap_audio_model"` is HF's id for a CLAP audio config;
    /// `"open_clip"` is this workspace's canonical id for the OpenCLIP
    /// checkpoint family, which ships `open_clip_config.json` with no
    /// `model_type` field at all. This field names the architecture, never
    /// which of a multi-tower checkpoint's towers the weights belong to —
    /// that is [`Self::tower`].
    pub model_type: String,
    /// Default LoRA rank used at training time.
    pub lora_rank: usize,
    /// LoRA scaling factor used at training time.
    pub lora_alpha: f64,
    /// Whether RSLoRA-style `alpha / sqrt(rank)` scaling was used.
    pub use_rslora: bool,
    /// Module-name suffixes that received a LoRA adapter.
    pub target_modules: Vec<String>,
    /// Optional restriction of LoRA injection to specific layer indices.
    #[serde(default)]
    pub layers_to_transform: Option<Vec<usize>>,
    /// Per-module rank overrides keyed by module-name substring.
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,
    /// Dtype used for the frozen backbone at training time. Defaults to F32.
    /// `BF16` halves backbone memory with negligible impact on training
    /// dynamics because the backbone weights are frozen; the trainable LoRA A
    /// and B matrices always stay in F32. The `ComputePrecision -> candle
    /// DType` mapping this field eventually needs lives at the candle
    /// boundary (`jammi_encoders::compute_precision_to_dtype`), not on this
    /// candle-free config type.
    #[serde(default)]
    pub backbone_dtype: ComputePrecision,
    /// Which tower of a multi-tower checkpoint this adapter installs on —
    /// see [`Tower`]. `None` for a single-tower family, and `None` for every
    /// adapter written before this field existed: `#[serde(default)]` means
    /// a legacy `adapter_config.json` with no `tower` key deserialises
    /// unchanged (asserted by
    /// `tests::legacy_adapter_json_without_tower_round_trips`).
    ///
    /// `skip_serializing_if = "Option::is_none"` omits the key entirely for
    /// a single-tower family: without it, every already-shipped BERT-family
    /// `adapter_config.json` would gain a `"tower":null` key it never had,
    /// changing the persisted adapter bytes (folded into the content digest,
    /// see K7) for no semantic reason. A dual/triple-tower adapter still
    /// emits `"tower":"text"` / `"vision"` / `"audio"` because the value is
    /// `Some`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tower: Option<Tower>,
}

impl AdapterConfig {
    /// Snapshot an `AdapterConfig` from a build-time `LoraBuildConfig`.
    ///
    /// Run-time-only fields (`lora_dropout`, `init_mode`) are intentionally
    /// not persisted — they affect training behaviour but do not change the
    /// shape or semantics of the loaded adapter weights.
    pub fn from_build(
        model_type: &str,
        lora: &LoraBuildConfig<'_>,
        backbone_dtype: ComputePrecision,
    ) -> Self {
        Self {
            model_type: model_type.into(),
            lora_rank: lora.lora_rank,
            lora_alpha: lora.lora_alpha,
            use_rslora: lora.use_rslora,
            target_modules: lora.target_modules.to_vec(),
            layers_to_transform: lora.layers_to_transform.clone(),
            rank_pattern: lora.rank_pattern.clone(),
            backbone_dtype,
            // A build config says which MODULES are adapted, never which
            // tower of a multi-tower checkpoint they belong to (the same
            // `LoraBuildConfig` builds either tower). The caller that knows
            // adds it with [`Self::with_tower`].
            tower: None,
        }
    }

    /// Record which tower of a multi-tower checkpoint this adapter installs
    /// on. Consumed fluently right after [`Self::from_build`], which cannot
    /// know it — see that method's own doc.
    pub fn with_tower(mut self, tower: Tower) -> Self {
        self.tower = Some(tower);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Domain edge (family D) for the ONE field this commit adds: an
    /// `adapter_config.json` written before `tower` existed has no such key,
    /// and must still deserialise — `#[serde(default)]` makes the absent key
    /// mean `None` ("single-tower family / unspecified"), not a parse error
    /// that would strand every already-shipped adapter. The round trip back
    /// out is asserted too, so a `None` tower is not silently re-materialised
    /// as some default variant.
    #[test]
    fn legacy_adapter_json_without_tower_round_trips() {
        let legacy = r#"{
            "model_type": "bert",
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "use_rslora": false,
            "target_modules": ["query", "value"]
        }"#;
        let cfg: AdapterConfig = serde_json::from_str(legacy).expect("legacy JSON must parse");
        assert_eq!(cfg.model_type, "bert");
        assert_eq!(cfg.lora_rank, 8);
        assert!(
            cfg.tower.is_none(),
            "an absent `tower` key must mean None, never a defaulted variant"
        );

        let round_tripped: AdapterConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert!(round_tripped.tower.is_none());
        assert_eq!(round_tripped.target_modules, vec!["query", "value"]);
    }

    /// The new field's own wire form: `snake_case`, and each variant
    /// round-trips through JSON as itself (not merely "some variant").
    #[test]
    fn tower_serialises_snake_case_and_round_trips_every_variant() {
        for (tower, wire) in [
            (Tower::Text, "\"text\""),
            (Tower::Vision, "\"vision\""),
            (Tower::Audio, "\"audio\""),
        ] {
            assert_eq!(serde_json::to_string(&tower).unwrap(), wire);
            let back: Tower = serde_json::from_str(wire).unwrap();
            assert_eq!(back, tower);
        }
    }

    /// A minimal `LoraBuildConfig` for the `from_build` byte-shape tests
    /// below. Only the fields `AdapterConfig::from_build` reads matter; the
    /// borrowed fields point at locals owned by the caller.
    struct TextFamilyBuildFixture {
        target_modules: Vec<String>,
        layers_to_transform: Option<Vec<usize>>,
        rank_pattern: HashMap<String, usize>,
    }

    impl TextFamilyBuildFixture {
        fn new() -> Self {
            Self {
                target_modules: vec!["query".to_string(), "value".to_string()],
                layers_to_transform: None,
                rank_pattern: HashMap::new(),
            }
        }

        fn build_config(&self) -> LoraBuildConfig<'_> {
            LoraBuildConfig {
                target_modules: &self.target_modules,
                layers_to_transform: &self.layers_to_transform,
                lora_rank: 8,
                lora_alpha: 16.0,
                use_rslora: false,
                lora_dropout: None,
                rank_pattern: &self.rank_pattern,
                init_mode: crate::init::LoraInitMode::default(),
                seed: 7,
            }
        }
    }

    /// K7 (a): a single-tower family's `from_build(..)` output serialises
    /// WITHOUT a `tower` key at all — byte-equal to the pre-unit JSON shape,
    /// constructed here explicitly rather than by re-deriving it from the
    /// struct (which would not catch a regression back to emitting the key).
    #[test]
    fn from_build_for_text_family_omits_tower_key_and_matches_pre_unit_shape() {
        let fixture = TextFamilyBuildFixture::new();
        let cfg = AdapterConfig::from_build("bert", &fixture.build_config(), ComputePrecision::F32);
        assert_eq!(cfg.tower, None);

        let json = serde_json::to_string(&cfg).expect("serialize adapter config");
        assert!(
            !json.contains("tower"),
            "a single-tower family must not serialise a `tower` key at all: {json}"
        );

        let expected = serde_json::json!({
            "model_type": "bert",
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "use_rslora": false,
            "target_modules": ["query", "value"],
            "layers_to_transform": null,
            "rank_pattern": {},
            "backbone_dtype": "f32",
        });
        let actual: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            actual, expected,
            "pre-unit JSON shape must be preserved byte-for-byte (as JSON values)"
        );
    }

    /// K7 (b): `with_tower(Tower::Vision)` serialises `"tower":"vision"` —
    /// the `Some` branch is unaffected by `skip_serializing_if`.
    #[test]
    fn with_tower_vision_serialises_the_tower_key() {
        let fixture = TextFamilyBuildFixture::new();
        let cfg =
            AdapterConfig::from_build("open_clip", &fixture.build_config(), ComputePrecision::F32)
                .with_tower(Tower::Vision);
        let json = serde_json::to_string(&cfg).expect("serialize adapter config");
        assert!(
            json.contains("\"tower\":\"vision\""),
            "a stamped tower must serialise its value: {json}"
        );
    }

    /// K7 (c): both shapes round-trip through `AdapterConfig` deserialize —
    /// the key-absent (single-tower) and key-present (`Some`) forms.
    #[test]
    fn tower_key_absent_and_present_both_round_trip() {
        let fixture = TextFamilyBuildFixture::new();
        let no_tower =
            AdapterConfig::from_build("bert", &fixture.build_config(), ComputePrecision::F32);
        let json = serde_json::to_string(&no_tower).unwrap();
        let back: AdapterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tower, None);

        let with_tower = no_tower.with_tower(Tower::Audio);
        let json = serde_json::to_string(&with_tower).unwrap();
        let back: AdapterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tower, Some(Tower::Audio));
    }
}
