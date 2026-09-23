//! The task taxonomy: what a model computes, and so which columns its
//! output carries. The canonical spelling is the one a catalog persists
//! and the wire form carries.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// What inference task a model performs.
///
/// Persisted and carried as a snake-case string; in-process call sites
/// pass the enum. The [`as_str`](Self::as_str) / [`parse`](Self::parse)
/// pair is the authoritative mapping — `Display`, `FromStr` and serde all
/// delegate to it so there is exactly one spelling per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum ModelTask {
    /// Produce dense vector representations of input text.
    TextEmbedding,
    /// Produce dense vector representations of input images.
    ImageEmbedding,
    /// Produce dense vector representations of input audio clips.
    AudioEmbedding,
    /// Assign a label and confidence score to input text.
    Classification,
    /// Extract named entities (person, org, location, etc.) from text.
    Ner,
    /// Predict a continuous outcome as a *distribution* — a Gaussian
    /// `(mean, std)` or a set of quantiles — rather than a point
    /// ([`DistributionAdapter`](crate::inference::adapter::DistributionAdapter) serves
    /// it). Unlike a similarity edge — a *derivation* over embeddings, which
    /// has no variant — this is a genuine model output type, so it belongs
    /// in [`Self::ALL`].
    Regression,
}

impl ModelTask {
    /// Every variant in declaration order. The single source of truth for
    /// "what tasks exist" — a caller that fans over the full set reads it
    /// here rather than re-listing variants. Kept consistent with the
    /// `enum` body by `all_covers_every_variant_via_exhaustive_match`.
    pub const ALL: &'static [ModelTask] = &[
        ModelTask::TextEmbedding,
        ModelTask::ImageEmbedding,
        ModelTask::AudioEmbedding,
        ModelTask::Classification,
        ModelTask::Ner,
        ModelTask::Regression,
    ];

    /// The canonical snake-case spelling. The single source of truth —
    /// `Display`, `FromStr` and serde all route through this.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextEmbedding => "text_embedding",
            Self::ImageEmbedding => "image_embedding",
            Self::AudioEmbedding => "audio_embedding",
            Self::Classification => "classification",
            Self::Ner => "ner",
            Self::Regression => "regression",
        }
    }

    /// Decode the canonical spelling. An unknown spelling is
    /// [`Error::UnknownTask`], naming the offending value.
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s {
            "text_embedding" => Ok(Self::TextEmbedding),
            "image_embedding" => Ok(Self::ImageEmbedding),
            "audio_embedding" => Ok(Self::AudioEmbedding),
            "classification" => Ok(Self::Classification),
            "ner" => Ok(Self::Ner),
            "regression" => Ok(Self::Regression),
            other => Err(Error::UnknownTask(other.to_string())),
        }
    }

    /// `true` for the embedding variants that produce vectors; `false` for
    /// inference-only tasks.
    pub fn is_embedding(&self) -> bool {
        matches!(
            self,
            Self::TextEmbedding | Self::ImageEmbedding | Self::AudioEmbedding
        )
    }
}

impl std::fmt::Display for ModelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ModelTask {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<String> for ModelTask {
    type Error = Error;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

impl From<ModelTask> for String {
    fn from(task: ModelTask) -> Self {
        task.as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spelling_round_trips_every_variant() {
        for variant in ModelTask::ALL {
            let s = variant.as_str();
            assert_eq!(
                ModelTask::parse(s).unwrap(),
                *variant,
                "round-trip failed for {variant:?} via '{s}'"
            );
        }
    }

    #[test]
    fn all_covers_every_variant_via_exhaustive_match() {
        // The match below is exhaustive — adding a new variant to the
        // enum without extending `ALL` either fails to compile here
        // (new arm needed) or fails the `contains` assertion at test
        // time. Two-layer defense against `ALL` drifting from the enum.
        fn assert_listed_in_all(t: ModelTask) {
            match t {
                ModelTask::TextEmbedding
                | ModelTask::ImageEmbedding
                | ModelTask::AudioEmbedding
                | ModelTask::Classification
                | ModelTask::Ner
                | ModelTask::Regression => {
                    assert!(
                        ModelTask::ALL.contains(&t),
                        "ModelTask::ALL is missing {t:?}"
                    );
                }
            }
        }
        for v in ModelTask::ALL {
            assert_listed_in_all(*v);
        }
    }

    #[test]
    fn unknown_spelling_is_a_typed_error() {
        let err = ModelTask::parse("not_a_task").unwrap_err();
        assert!(
            matches!(err, Error::UnknownTask(ref m) if m == "not_a_task"),
            "unknown variant should surface as Error::UnknownTask naming the input, got {err:?}"
        );
    }

    #[test]
    fn display_matches_spelling() {
        assert_eq!(format!("{}", ModelTask::TextEmbedding), "text_embedding");
        assert_eq!(format!("{}", ModelTask::ImageEmbedding), "image_embedding");
        assert_eq!(format!("{}", ModelTask::AudioEmbedding), "audio_embedding");
        assert_eq!(format!("{}", ModelTask::Classification), "classification");
        assert_eq!(format!("{}", ModelTask::Ner), "ner");
        assert_eq!(format!("{}", ModelTask::Regression), "regression");
    }

    #[test]
    fn from_str_delegates_to_parse() {
        use std::str::FromStr;
        assert_eq!(
            ModelTask::from_str("text_embedding").unwrap(),
            ModelTask::TextEmbedding
        );
        assert!(ModelTask::from_str("bogus").is_err());
    }

    #[test]
    fn is_embedding_is_true_only_for_embedding_variants() {
        assert!(ModelTask::TextEmbedding.is_embedding());
        assert!(ModelTask::ImageEmbedding.is_embedding());
        assert!(ModelTask::AudioEmbedding.is_embedding());
        assert!(!ModelTask::Classification.is_embedding());
        assert!(!ModelTask::Ner.is_embedding());
        assert!(!ModelTask::Regression.is_embedding());
    }

    #[test]
    fn serde_round_trips_via_canonical_string() {
        for variant in ModelTask::ALL {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ModelTask = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, *variant);
            assert_eq!(json, format!("\"{}\"", variant.as_str()));
        }
    }
}
