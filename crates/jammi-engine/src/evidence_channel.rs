//! Validated identifier for an evidence channel.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{JammiError, Result};

/// Validated identifier for an evidence channel.
///
/// Channel ids are ASCII slugs (`[a-z][a-z0-9_]{0,63}`). They appear in
/// `retrieved_by` / `annotated_by` list columns and as foreign-key values
/// in the `evidence_channel_columns` catalog table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ChannelId(String);

impl ChannelId {
    /// Construct a channel id, validating the slug rules.
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        validate(&s)?;
        Ok(Self(s))
    }

    /// Return the underlying slug.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn validate(s: &str) -> Result<()> {
    if s.is_empty() {
        return Err(JammiError::EvidenceChannel(format!(
            "invalid channel id '{s}': must not be empty"
        )));
    }
    if s.len() > 64 {
        return Err(JammiError::EvidenceChannel(format!(
            "invalid channel id '{s}': must be at most 64 characters"
        )));
    }
    let bytes = s.as_bytes();
    let first = bytes[0];
    if !(first.is_ascii_lowercase()) {
        return Err(JammiError::EvidenceChannel(format!(
            "invalid channel id '{s}': must start with a lowercase ASCII letter"
        )));
    }
    for &b in &bytes[1..] {
        if !(b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
            return Err(JammiError::EvidenceChannel(format!(
                "invalid channel id '{s}': must be [a-z0-9_]"
            )));
        }
    }
    Ok(())
}

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ChannelId {
    type Err = JammiError;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

impl TryFrom<String> for ChannelId {
    type Error = JammiError;

    fn try_from(s: String) -> Result<Self> {
        Self::new(s)
    }
}

impl From<ChannelId> for String {
    fn from(c: ChannelId) -> String {
        c.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_lowercase_slug() {
        let c = ChannelId::new("scored_by").unwrap();
        assert_eq!(c.as_str(), "scored_by");
    }

    #[test]
    fn accepts_digits_after_first_char() {
        let c = ChannelId::new("rank2_score").unwrap();
        assert_eq!(c.as_str(), "rank2_score");
    }

    #[test]
    fn rejects_empty_string() {
        let err = ChannelId::new("").unwrap_err();
        assert!(matches!(err, JammiError::EvidenceChannel(_)));
    }

    #[test]
    fn rejects_uppercase_letter() {
        let err = ChannelId::new("Vector").unwrap_err();
        assert!(matches!(err, JammiError::EvidenceChannel(_)));
    }

    #[test]
    fn rejects_leading_digit() {
        let err = ChannelId::new("1vector").unwrap_err();
        assert!(matches!(err, JammiError::EvidenceChannel(_)));
    }

    #[test]
    fn rejects_hyphen() {
        let err = ChannelId::new("vector-1").unwrap_err();
        assert!(matches!(err, JammiError::EvidenceChannel(_)));
    }

    #[test]
    fn rejects_too_long_slug() {
        let too_long = "a".repeat(65);
        assert!(ChannelId::new(too_long).is_err());
    }

    #[test]
    fn accepts_max_length_slug() {
        let max = format!("a{}", "b".repeat(63));
        assert!(ChannelId::new(max).is_ok());
    }

    #[test]
    fn from_str_round_trip() {
        let c = ChannelId::from_str("inference").unwrap();
        assert_eq!(c.to_string(), "inference");
    }

    #[test]
    fn serde_round_trip_through_string() {
        let c = ChannelId::new("vector").unwrap();
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "\"vector\"");
        let parsed: ChannelId = serde_json::from_str(&json).unwrap();
        assert_eq!(c, parsed);
    }

    #[test]
    fn serde_rejects_invalid_slug() {
        let r: std::result::Result<ChannelId, _> = serde_json::from_str("\"Bad-Slug\"");
        assert!(r.is_err());
    }
}
