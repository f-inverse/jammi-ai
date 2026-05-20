//! Broker-assigned monotonic offset for delivered events.

use chrono::{DateTime, Utc};

/// Monotonically increasing position within a topic. The broker assigns
/// `value` at publish time; `committed_at` is the UTC instant the backing
/// transaction committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset {
    value: u64,
    committed_at: DateTime<Utc>,
}

impl Offset {
    pub const fn new(value: u64, committed_at: DateTime<Utc>) -> Self {
        Self {
            value,
            committed_at,
        }
    }

    pub const fn value(&self) -> u64 {
        self.value
    }

    pub const fn committed_at(&self) -> DateTime<Utc> {
        self.committed_at
    }
}
