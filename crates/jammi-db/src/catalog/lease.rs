//! The one lease primitive every leased catalog row shares.
//!
//! Two row families carry a lease today: a claimed `training_jobs` row (the
//! worker heartbeats it while the run lasts) and a `building` `result_tables`
//! row (the writer heartbeats it while the bytes are being produced). Both
//! write the same timestamp shape into a `lease_expires_at` column, both renew
//! by a compare-and-set that names the owner, and both are reclaimed by a
//! sweep that compares the column against the engine clock. This module owns
//! the timestamp format, the clock helpers, the validated
//! [`LeaseIntervals`] pair, and the SQL fragment the sweeps test with, so no
//! row family can drift into its own lease arithmetic.

use std::time::Duration;

/// Format leases write into `lease_expires_at`. Lexicographic ordering of two
/// timestamps in this fixed-width UTC form matches chronological ordering, so
/// the SQL `lease_expires_at < $now` comparison is correct on both backends
/// without dialect-specific interval arithmetic.
pub const LEASE_TS_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.6fZ";

/// `now`, formatted for an engine-clock lease comparison or stamp.
pub fn lease_now() -> String {
    chrono::Utc::now().format(LEASE_TS_FORMAT).to_string()
}

/// `now + lease`, formatted as a lease deadline.
pub fn lease_deadline(lease: Duration) -> String {
    let expiry =
        chrono::Utc::now() + chrono::Duration::from_std(lease).unwrap_or(chrono::Duration::MAX);
    expiry.format(LEASE_TS_FORMAT).to_string()
}

/// The validated lease window and renewal interval a lease holder drives its
/// heartbeat with. `lease` is the single source of truth for the window: the
/// holder passes the same value to its claim and to every renewal, so the
/// renew always targets the same deadline the reclaim path compares against.
///
/// Only [`crate::config::LeaseConfig::intervals`] (or [`Default`], which is the
/// engine's built-in 30 s / 10 s pair) builds one, so `heartbeat * 2 < lease`
/// and both-non-zero hold by construction: a live holder renews at least twice
/// per window, so a single missed beat still leaves one in-window renewal that
/// lands strictly before expiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseIntervals {
    lease: Duration,
    heartbeat: Duration,
}

impl LeaseIntervals {
    /// Crate-internal constructor for the validated pair; the public
    /// constructor is [`crate::config::LeaseConfig::intervals`].
    pub(crate) fn new_validated(lease: Duration, heartbeat: Duration) -> Self {
        debug_assert!(
            !lease.is_zero() && !heartbeat.is_zero() && heartbeat * 2 < lease,
            "LeaseIntervals invariant: heartbeat * 2 < lease, both non-zero"
        );
        Self { lease, heartbeat }
    }

    /// The lease window — how long a claim owns its row before it is
    /// reclaimable.
    pub fn lease(&self) -> Duration {
        self.lease
    }

    /// The renewal interval, strictly inside half the window.
    pub fn heartbeat(&self) -> Duration {
        self.heartbeat
    }
}

impl Default for LeaseIntervals {
    /// The engine's built-in timing: a 30 s lease renewed every 10 s.
    fn default() -> Self {
        Self::new_validated(Duration::from_secs(30), Duration::from_secs(10))
    }
}

/// The SQL fragment that is true for an absent or expired lease:
/// `(col IS NULL OR col < $bind)`, where `$bind` is the position the caller
/// binds [`lease_now`] at. The caller supplies the position so the fragment
/// composes into any statement's parameter numbering.
pub fn lease_expired_clause(col: &str, bind: usize) -> String {
    format!("({col} IS NULL OR {col} < ${bind})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_is_after_now_and_sorts_lexicographically() {
        let now = lease_now();
        let later = lease_deadline(Duration::from_secs(30));
        assert!(later > now, "{later} must sort after {now}");
        assert_eq!(now.len(), later.len(), "fixed-width form");
    }

    #[test]
    fn expired_clause_names_the_column_and_bind() {
        assert_eq!(
            lease_expired_clause("lease_expires_at", 3),
            "(lease_expires_at IS NULL OR lease_expires_at < $3)"
        );
    }

    #[test]
    fn default_intervals_are_the_engine_constants() {
        let d = LeaseIntervals::default();
        assert_eq!(d.lease(), Duration::from_secs(30));
        assert_eq!(d.heartbeat(), Duration::from_secs(10));
    }
}
