//! Newtype identifiers for trigger-stream entities.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Catalog-assigned topic identifier. UUIDv7 — time-ordered for index
/// locality per ADR-00.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TopicId(Uuid);

impl TopicId {
    /// Mint a fresh UUIDv7-backed identifier.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    pub const fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    pub const fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TopicId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TopicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl FromStr for TopicId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::from_str(s).map(Self)
    }
}

/// Server-assigned identifier for one live subscription. UUIDv4 — never
/// persisted, never indexed; a subscription's lifetime is one Subscribe RPC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubscriptionId(Uuid);

impl SubscriptionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub const fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SubscriptionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SubscriptionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
