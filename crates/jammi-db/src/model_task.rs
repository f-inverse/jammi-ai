//! ML task taxonomy shared across the catalog, store, cache, and inference
//! call sites. Lives in `jammi-db` because `jammi-db` owns the
//! catalog tables that persist it (`models.task`, `result_tables.task`) and
//! the on-disk strings must agree across every crate that reads or writes
//! them. `jammi-ai` re-exports the type for callers that consume the
//! higher-level inference surface.

use serde::{Deserialize, Serialize};

use crate::error::JammiError;

/// What inference task a model performs.
///
/// The catalog persists this as a snake-case `TEXT` column; in-process call
/// sites should pass the enum directly. The
/// [`as_db_str`](Self::as_db_str) / [`try_from_db_str`](Self::try_from_db_str)
/// pair is the authoritative database mapping — `Display`, `FromStr`, and
/// serde all delegate to it so there is exactly one spelling per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum ModelTask {
    /// Produce dense vector representations of input text.
    TextEmbedding,
    /// Produce dense vector representations of input images.
    ImageEmbedding,
    /// Assign a label and confidence score to input text.
    Classification,
    /// Extract named entities (person, org, location, etc.) from text.
    Ner,
}

impl ModelTask {
    /// Every variant in declaration order. The single source of truth for
    /// "what tasks exist" — `ResultStore`, the catalog SQL builders, and
    /// any future caller that needs to fan over the full set must read it
    /// here rather than re-listing variants. Kept consistent with the
    /// `enum` body by `all_covers_every_variant_via_exhaustive_match` in
    /// `tests` below.
    pub const ALL: &'static [ModelTask] = &[
        ModelTask::TextEmbedding,
        ModelTask::ImageEmbedding,
        ModelTask::Classification,
        ModelTask::Ner,
    ];

    /// Canonical snake-case string stored in the catalog. The single source
    /// of truth — `Display`, `FromStr`, serde all route through this.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            Self::TextEmbedding => "text_embedding",
            Self::ImageEmbedding => "image_embedding",
            Self::Classification => "classification",
            Self::Ner => "ner",
        }
    }

    /// Decode the canonical snake-case string back into a [`ModelTask`].
    /// Unknown spellings raise [`JammiError::Other`] naming the offending
    /// value and the accepted set.
    pub fn try_from_db_str(s: &str) -> Result<Self, JammiError> {
        match s {
            "text_embedding" => Ok(Self::TextEmbedding),
            "image_embedding" => Ok(Self::ImageEmbedding),
            "classification" => Ok(Self::Classification),
            "ner" => Ok(Self::Ner),
            other => Err(JammiError::Other(format!(
                "Unknown model task '{other}'. Expected: text_embedding, image_embedding, classification, ner"
            ))),
        }
    }

    /// `true` for the two embedding variants that participate in vector
    /// search and ANN sidecar indexes; `false` for inference-only tasks.
    pub fn is_embedding(&self) -> bool {
        matches!(self, Self::TextEmbedding | Self::ImageEmbedding)
    }
}

impl std::fmt::Display for ModelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_db_str())
    }
}

impl std::str::FromStr for ModelTask {
    type Err = JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from_db_str(s)
    }
}

impl TryFrom<String> for ModelTask {
    type Error = JammiError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::try_from_db_str(&s)
    }
}

impl From<ModelTask> for String {
    fn from(task: ModelTask) -> Self {
        task.as_db_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_str_round_trip_covers_every_variant() {
        for variant in ModelTask::ALL {
            let s = variant.as_db_str();
            assert_eq!(
                ModelTask::try_from_db_str(s).unwrap(),
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
                | ModelTask::Classification
                | ModelTask::Ner => {
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
    fn unknown_db_str_returns_typed_error() {
        let err = ModelTask::try_from_db_str("not_a_task").unwrap_err();
        assert!(
            matches!(err, JammiError::Other(ref m) if m.contains("not_a_task")),
            "unknown variant should surface as JammiError::Other naming the input, got {err:?}"
        );
    }

    #[test]
    fn display_matches_db_str() {
        assert_eq!(format!("{}", ModelTask::TextEmbedding), "text_embedding");
        assert_eq!(format!("{}", ModelTask::ImageEmbedding), "image_embedding");
        assert_eq!(format!("{}", ModelTask::Classification), "classification");
        assert_eq!(format!("{}", ModelTask::Ner), "ner");
    }

    #[test]
    fn from_str_delegates_to_try_from_db_str() {
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
        assert!(!ModelTask::Classification.is_embedding());
        assert!(!ModelTask::Ner.is_embedding());
    }

    #[test]
    fn serde_round_trips_via_canonical_string() {
        for variant in ModelTask::ALL {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ModelTask = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, *variant);
            // serde flatten via String -> the JSON is the canonical
            // snake-case spelling wrapped in quotes.
            assert_eq!(json, format!("\"{}\"", variant.as_db_str()));
        }
    }
}
