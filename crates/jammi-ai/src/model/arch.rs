//! The ONE encoder-architecture predicate, and the ONE frozen
//! config/weights candidate-name precedence, this crate detects models with.
//!
//! # Why this module exists
//!
//! "Which architecture is this checkpoint?" and "which file in this directory
//! is its config / its weights?" were previously answered independently at
//! eight sites: the serving loader's `is_clap`/`is_open_clip` pair, the
//! [`ModelDimensions`](super::ModelDimensions) geometry parser, the
//! resolver's catalog-lookup / local / hub chains, the esc-058 fingerprint's
//! tracked-candidate lists, and the fine-tune worker's own on-disk read. Eight
//! independent answers to one question is eight chances for two of them to
//! disagree — and a disagreement here is silent: a checkpoint routed to the
//! wrong family builds the wrong tower, and a weights chain that disagrees
//! with the identity chain makes a file whose appearance flips the loaded
//! bytes invisible to the staleness probe.
//!
//! Everything here is a MOVE, not a redesign. [`EncoderFamily::from_config`]'s
//! CLAP rules are the ones the serving loader has always applied; the
//! candidate-name lists are in the precedence the resolver froze (issue #351)
//! and the fingerprint tracks.
//!
//! # The two candidate lists are NOT one list
//!
//! [`WEIGHTS_CANDIDATE_NAMES`] is the IDENTITY list: every file name whose
//! appearance in a model directory could change which bytes a cold resolve
//! loads, so the esc-058 fingerprint must track all four. It is deliberately
//! **not** a resolution chain: `model.onnx` selects a different BACKEND (the
//! resolver's ORT arm), not a different weights file for the same backend, so
//! collapsing the four into one "first existing wins" chain would make an
//! ONNX file shadow a `model.gguf` for a Candle load — a behaviour change.
//! [`weights_candidates`] therefore walks [`CANDLE_WEIGHTS_CANDIDATE_NAMES`],
//! the Candle-backend chain the resolver's own local arm froze.

use std::path::{Path, PathBuf};

use jammi_lora::Tower;

/// Config file names in the FROZEN resolution order every config-reading site
/// in this crate applies: the standard HuggingFace `config.json` first, the
/// OpenCLIP `open_clip_config.json` second.
pub const CONFIG_CANDIDATE_NAMES: [&str; 2] = ["config.json", "open_clip_config.json"];

/// Every weights file name whose PRESENCE is identity-bearing for a resolved
/// model directory, in the frozen order the esc-058 fingerprint tracks them
/// (`compute_model_fingerprint`'s weights slot).
///
/// Read the module doc before using this as a resolution chain — it is not
/// one. Use [`weights_candidates`] / [`CANDLE_WEIGHTS_CANDIDATE_NAMES`] to
/// pick the file a Candle load actually reads.
pub const WEIGHTS_CANDIDATE_NAMES: [&str; 4] = [
    "model.safetensors",
    "open_clip_model.safetensors",
    "model.onnx",
    "model.gguf",
];

/// The Candle-backend weights chain, in the precedence issue #351 froze:
/// `model.safetensors` wins, then the OpenCLIP-named safetensors, and only
/// when NEITHER is present does `model.gguf` enter the picture at all.
pub const CANDLE_WEIGHTS_CANDIDATE_NAMES: [&str; 3] = [
    "model.safetensors",
    "open_clip_model.safetensors",
    "model.gguf",
];

/// The canonical GGUF weights file name. A directory carrying some other
/// `*.gguf` file is a typed refusal at the resolver, never a silent load.
pub const GGUF_WEIGHTS_FILENAME: &str = "model.gguf";

/// The ONNX weights file name — the resolver's ORT-arm selector. Named here
/// so the identity list and the ORT arm share one spelling.
pub const ONNX_WEIGHTS_FILENAME: &str = "model.onnx";

/// The encoder architecture families this crate can load, train and serve.
///
/// This is the single answer to "what is this checkpoint?" — derived from the
/// checkpoint's OWN config ([`Self::from_config`]) or from a saved adapter's
/// recorded architecture id ([`Self::from_adapter_model_type`]), and compared
/// FAMILY-to-FAMILY at the one seam where the two must agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderFamily {
    /// BERT and its config-compatible relatives (RoBERTa, CamemBERT,
    /// XLM-RoBERTa) — the aliases the serving text arm has always accepted.
    Bert,
    /// DistilBERT.
    DistilBert,
    /// ModernBERT.
    ModernBert,
    /// An OpenCLIP checkpoint: one file carrying BOTH a text tower and a
    /// vision tower, described by `open_clip_config.json`'s `model_cfg`.
    OpenClip,
    /// An HF-CLAP checkpoint's HTSAT-Swin audio tower
    /// (`ClapAudioModelWithProjection` lineage).
    ClapAudio,
}

impl EncoderFamily {
    /// Classify a checkpoint from its parsed config JSON, or `None` when the
    /// config names an architecture this crate does not implement.
    ///
    /// Order matters and is the serving loader's own: CLAP first (its
    /// structural signal is the most specific), then OpenCLIP, then the text
    /// families keyed on `model_type`. The three groups are disjoint in
    /// practice — an OpenCLIP config carries no `model_type` at all and a
    /// CLAP config carries no `model_cfg` — so the order is a tiebreak that
    /// never fires, stated explicitly rather than left to chance.
    ///
    /// `None` (rather than a defaulted family) is the whole point: a caller
    /// that coerces an unrecognised `model_type` into BERT trains and serves
    /// a confidently wrong architecture. Every caller must handle `None` as a
    /// typed refusal.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        if is_clap_audio_config(config) {
            return Some(Self::ClapAudio);
        }
        if config.get("model_cfg").is_some() {
            return Some(Self::OpenClip);
        }
        match config.get("model_type").and_then(|v| v.as_str())? {
            // The alias set the serving text arm accepts, verbatim: these
            // four all parse as `BertConfig` and load as `Bert`.
            "bert" | "roberta" | "camembert" | "xlm-roberta" => Some(Self::Bert),
            "distilbert" => Some(Self::DistilBert),
            "modernbert" => Some(Self::ModernBert),
            _ => None,
        }
    }

    /// Classify from a saved adapter's recorded
    /// [`AdapterConfig::model_type`](jammi_lora::AdapterConfig::model_type).
    ///
    /// TOTAL by design, and the ONE place in this crate where an unknown
    /// string still becomes `Bert`. That coercion is not a shortcut — it
    /// reproduces the historical fine-tune worker's `_ => BERT` arm, which
    /// means every adapter already written to disk by a released build
    /// carries whatever `model_type` string its base config happened to
    /// declare (including the BERT aliases `roberta` / `camembert` /
    /// `xlm-roberta`, and anything else that parsed as a `BertConfig`). Those
    /// adapters were trained on a BERT tower and must keep loading onto one;
    /// refusing them here would strand shipped artifacts.
    ///
    /// The refusal that DOES matter is a CROSS-family one — an `open_clip`
    /// adapter on a CLAP base, a `clap_audio_model` adapter on a BERT base —
    /// and that is exactly what comparing this against [`Self::from_config`]
    /// catches, because the three non-text ids map to their own families
    /// here.
    pub fn from_adapter_model_type(model_type: &str) -> Self {
        match model_type {
            "open_clip" => Self::OpenClip,
            "clap_audio_model" => Self::ClapAudio,
            "distilbert" => Self::DistilBert,
            "modernbert" => Self::ModernBert,
            _ => Self::Bert,
        }
    }

    /// The architecture id this family writes into a saved
    /// `adapter_config.json`. Three are HuggingFace's own `model_type`
    /// values, `clap_audio_model` is HF's id for a CLAP audio config, and
    /// `open_clip` is this workspace's canonical id for a checkpoint family
    /// that ships no `model_type` field at all.
    ///
    /// Round-trips through [`Self::from_adapter_model_type`] for every
    /// variant (asserted by `adapter_model_type_round_trips`).
    pub fn adapter_model_type(&self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::DistilBert => "distilbert",
            Self::ModernBert => "modernbert",
            Self::OpenClip => "open_clip",
            Self::ClapAudio => "clap_audio_model",
        }
    }

    /// Whether this family's checkpoint actually HAS the named tower.
    ///
    /// `None` means "unspecified" — the wire form every adapter written
    /// before `AdapterConfig::tower` existed carries — and is accepted for
    /// every family so legacy adapters keep loading. A single-tower text
    /// family additionally accepts an explicit [`Tower::Text`]; it has no
    /// vision or audio tower to install onto, and saying so is a typed
    /// refusal rather than a silently-ignored field.
    pub fn has_tower(&self, tower: Option<Tower>) -> bool {
        matches!(
            (self, tower),
            (_, None)
                | (
                    Self::Bert | Self::DistilBert | Self::ModernBert,
                    Some(Tower::Text)
                )
                | (Self::OpenClip, Some(Tower::Text | Tower::Vision))
                | (Self::ClapAudio, Some(Tower::Audio))
        )
    }

    /// Human-readable list of the towers this family has, for refusal
    /// messages.
    pub fn towers(&self) -> &'static str {
        match self {
            Self::Bert | Self::DistilBert | Self::ModernBert => "text",
            Self::OpenClip => "text, vision",
            Self::ClapAudio => "audio",
        }
    }
}

/// Detect an HF-CLAP audio checkpoint (`ClapAudioModelWithProjection`
/// lineage) from its config: `model_type == "clap_audio_model"` at the top
/// level (flat `ClapAudioConfig`) or under a nested `audio_config` (top-level
/// `ClapConfig`), or `architectures` listing `ClapModel` /
/// `ClapAudioModelWithProjection`.
///
/// Moved verbatim from the serving loader's own `is_hf_clap_config`, which is
/// now a one-line delegation to [`EncoderFamily::from_config`].
fn is_clap_audio_config(config: &serde_json::Value) -> bool {
    let model_type_is_clap = |v: &serde_json::Value| {
        v.get("model_type").and_then(|m| m.as_str()) == Some("clap_audio_model")
    };
    if model_type_is_clap(config) {
        return true;
    }
    if config.get("audio_config").is_some_and(model_type_is_clap) {
        return true;
    }
    config
        .get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|arch| {
            arch.iter().any(|a| {
                matches!(
                    a.as_str(),
                    Some("ClapModel") | Some("ClapAudioModelWithProjection")
                )
            })
        })
}

/// The first EXISTING config file under `dir`, walking
/// [`CONFIG_CANDIDATE_NAMES`] in its frozen order. `None` when the directory
/// carries neither — the caller owns the typed refusal, because the message
/// differs per caller (a resolver miss is `Ok(None)` on the catalog arm and a
/// hard error on the local arm).
pub fn config_candidates(dir: &Path) -> Option<PathBuf> {
    first_existing(dir, &CONFIG_CANDIDATE_NAMES)
}

/// The first EXISTING Candle-loadable weights file under `dir`, walking
/// [`CANDLE_WEIGHTS_CANDIDATE_NAMES`] in the frozen precedence. See the
/// module doc for why `model.onnx` is not in this chain.
pub fn weights_candidates(dir: &Path) -> Option<PathBuf> {
    first_existing(dir, &CANDLE_WEIGHTS_CANDIDATE_NAMES)
}

/// Every candidate path under `dir` for `names`, existing or not — the shape
/// the esc-058 fingerprint needs (it tracks ABSENT candidates too, so their
/// later appearance is detectable).
pub fn candidate_paths(dir: &Path, names: &[&str]) -> Vec<PathBuf> {
    names.iter().map(|n| dir.join(n)).collect()
}

fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every family's persisted adapter id maps back to that same family.
    /// Without this, a tower adapter could be written under an id the load
    /// seam classifies as a DIFFERENT family and be refused by its own
    /// writer.
    #[test]
    fn adapter_model_type_round_trips() {
        for family in [
            EncoderFamily::Bert,
            EncoderFamily::DistilBert,
            EncoderFamily::ModernBert,
            EncoderFamily::OpenClip,
            EncoderFamily::ClapAudio,
        ] {
            assert_eq!(
                EncoderFamily::from_adapter_model_type(family.adapter_model_type()),
                family,
                "{} must classify back to its own family",
                family.adapter_model_type()
            );
        }
    }

    /// The three BERT aliases the serving text arm accepts classify as
    /// `Bert` from a config, and an architecture this crate does not
    /// implement classifies as `None` — the arm that replaces the worker's
    /// old `_ => BERT` coercion.
    #[test]
    fn from_config_classifies_text_aliases_and_refuses_unknown() {
        for alias in ["bert", "roberta", "camembert", "xlm-roberta"] {
            assert_eq!(
                EncoderFamily::from_config(&serde_json::json!({ "model_type": alias })),
                Some(EncoderFamily::Bert),
                "{alias} is a BERT-config alias"
            );
        }
        assert_eq!(
            EncoderFamily::from_config(&serde_json::json!({ "model_type": "distilbert" })),
            Some(EncoderFamily::DistilBert)
        );
        assert_eq!(
            EncoderFamily::from_config(&serde_json::json!({ "model_type": "modernbert" })),
            Some(EncoderFamily::ModernBert)
        );
        // A config that PARSES as a `BertConfig` but names an architecture
        // this crate has no loader for: `None`, never a coerced `Bert`.
        assert_eq!(
            EncoderFamily::from_config(&serde_json::json!({
                "model_type": "gpt2",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
            })),
            None
        );
        // No `model_type` and no structural signal at all.
        assert_eq!(
            EncoderFamily::from_config(&serde_json::json!({ "hidden_size": 32 })),
            None
        );
    }

    /// The three CLAP structural signals and the OpenCLIP one, each on its
    /// own. `architectures`-only is included because the serving loader has
    /// always honoured it.
    #[test]
    fn from_config_classifies_media_families() {
        for clap in [
            serde_json::json!({ "model_type": "clap_audio_model" }),
            serde_json::json!({ "audio_config": { "model_type": "clap_audio_model" } }),
            serde_json::json!({ "architectures": ["ClapModel"] }),
            serde_json::json!({ "architectures": ["ClapAudioModelWithProjection"] }),
        ] {
            assert_eq!(
                EncoderFamily::from_config(&clap),
                Some(EncoderFamily::ClapAudio),
                "{clap} is a CLAP audio checkpoint"
            );
        }
        assert_eq!(
            EncoderFamily::from_config(&serde_json::json!({
                "model_cfg": { "embed_dim": 16, "vision_cfg": {}, "text_cfg": {} }
            })),
            Some(EncoderFamily::OpenClip)
        );
    }

    /// A cross-family adapter id never equals the base's family — the
    /// comparison the serving load seam refuses on — while every legacy /
    /// alias text id still lands on `Bert`, so a shipped BERT-family adapter
    /// keeps loading.
    #[test]
    fn adapter_family_mapping_separates_cross_family_from_legacy_text() {
        assert_eq!(
            EncoderFamily::from_adapter_model_type("roberta"),
            EncoderFamily::Bert
        );
        assert_eq!(
            EncoderFamily::from_adapter_model_type("something-nobody-shipped"),
            EncoderFamily::Bert
        );
        assert_ne!(
            EncoderFamily::from_adapter_model_type("clap_audio_model"),
            EncoderFamily::Bert
        );
        assert_ne!(
            EncoderFamily::from_adapter_model_type("open_clip"),
            EncoderFamily::ClapAudio
        );
    }

    /// Tower validity is per family, and an absent (legacy) `tower` is valid
    /// everywhere.
    #[test]
    fn has_tower_matches_the_checkpoint_structure() {
        assert!(EncoderFamily::Bert.has_tower(None));
        assert!(EncoderFamily::Bert.has_tower(Some(Tower::Text)));
        assert!(!EncoderFamily::Bert.has_tower(Some(Tower::Vision)));
        assert!(!EncoderFamily::Bert.has_tower(Some(Tower::Audio)));

        assert!(EncoderFamily::OpenClip.has_tower(Some(Tower::Text)));
        assert!(EncoderFamily::OpenClip.has_tower(Some(Tower::Vision)));
        assert!(!EncoderFamily::OpenClip.has_tower(Some(Tower::Audio)));

        assert!(EncoderFamily::ClapAudio.has_tower(Some(Tower::Audio)));
        assert!(!EncoderFamily::ClapAudio.has_tower(Some(Tower::Text)));
    }

    /// The identity list and the Candle chain are DIFFERENT lists, and the
    /// chain's precedence is the frozen one. A regression that folded
    /// `model.onnx` into the chain would make this fail.
    #[test]
    fn weights_chain_excludes_onnx_and_keeps_the_frozen_precedence() {
        assert!(WEIGHTS_CANDIDATE_NAMES.contains(&ONNX_WEIGHTS_FILENAME));
        assert!(!CANDLE_WEIGHTS_CANDIDATE_NAMES.contains(&ONNX_WEIGHTS_FILENAME));
        assert_eq!(CANDLE_WEIGHTS_CANDIDATE_NAMES[0], "model.safetensors");
        assert_eq!(
            CANDLE_WEIGHTS_CANDIDATE_NAMES[2], GGUF_WEIGHTS_FILENAME,
            "GGUF is last: it enters only when neither safetensors name exists"
        );

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.gguf"), b"g").unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"o").unwrap();
        assert_eq!(
            weights_candidates(dir.path()),
            Some(dir.path().join("model.gguf")),
            "an ONNX file must not shadow the GGUF a Candle load reads"
        );
        std::fs::write(dir.path().join("open_clip_model.safetensors"), b"s").unwrap();
        assert_eq!(
            weights_candidates(dir.path()),
            Some(dir.path().join("open_clip_model.safetensors"))
        );
        std::fs::write(dir.path().join("model.safetensors"), b"s").unwrap();
        assert_eq!(
            weights_candidates(dir.path()),
            Some(dir.path().join("model.safetensors"))
        );
    }

    /// The config chain prefers `config.json`, falls back to the OpenCLIP
    /// name, and reports absence rather than inventing a path.
    #[test]
    fn config_chain_prefers_standard_then_open_clip() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(config_candidates(dir.path()), None);
        std::fs::write(dir.path().join("open_clip_config.json"), b"{}").unwrap();
        assert_eq!(
            config_candidates(dir.path()),
            Some(dir.path().join("open_clip_config.json"))
        );
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert_eq!(
            config_candidates(dir.path()),
            Some(dir.path().join("config.json"))
        );
    }
}
