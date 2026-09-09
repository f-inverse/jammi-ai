//! The one lease primitive every leased catalog row shares.
//!
//! Two row families carry a lease today: a claimed `training_jobs` row (the
//! worker heartbeats it while the run lasts) and a `building` `result_tables`
//! row (the writer heartbeats it while the bytes are being produced). Both
//! write the same timestamp shape into a `lease_expires_at` column, both renew
//! by a compare-and-set that names the owner, and both are reclaimed by a
//! sweep that compares the column against a clock. This module owns the
//! timestamp format, the clock helpers, the validated [`LeaseIntervals`] pair,
//! and the SQL fragments the sweeps test with, so no row family can drift into
//! its own lease arithmetic.
//!
//! **Whose clock.** Two replicas' application clocks can be skewed against
//! each other even when both are correct against wall time; comparing one
//! replica's stamp against another replica's `now` can reap a writer that
//! is very much alive. On Postgres — the deployment shape where two replicas
//! share one catalog — [`lease_expired_clause`] and [`lease_deadline_expr`]
//! render SQL that reads and writes the CATALOG DATABASE's own clock
//! (`now()`), never a bound application timestamp, so replica clock skew
//! cannot affect a lease predicate at all: every replica's SQL text says
//! `now()`, evaluated once, by the one database, inside the one statement.
//! On SQLite (always a single embedded process; there is no peer replica to
//! skew against) the application clock stays the one source of truth,
//! through this module's [`lease_now`] / [`lease_deadline`] pair — the ONE
//! helper every SQLite lease stamp and comparison goes through.

use std::time::Duration;

use crate::catalog::backend::{BackendKind, SqlValue};

/// Format leases write into `lease_expires_at`. Lexicographic ordering of two
/// timestamps in this fixed-width UTC form matches chronological ordering, so
/// the SQL `lease_expires_at < $now` comparison is correct on both backends
/// without dialect-specific interval arithmetic. Also a valid ISO-8601
/// `timestamptz` literal, so Postgres can cast a stored value straight into
/// its own clock's domain (`col::timestamptz`) without reparsing it through
/// this format at all.
pub const LEASE_TS_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.6fZ";

/// `now`, formatted for a SQLite (single-process) lease comparison or stamp —
/// the application clock, through the one helper. Never used to build a
/// Postgres lease predicate; see the module docs.
pub fn lease_now() -> String {
    chrono::Utc::now().format(LEASE_TS_FORMAT).to_string()
}

/// `now + lease`, formatted as a lease deadline — the application-clock
/// stamp SQLite's lease writes bind directly. Never used to build a Postgres
/// lease predicate; see the module docs.
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

/// The SQL fragment that is true for an absent or expired lease, comparing
/// against the BACKEND's own clock — never a bound application timestamp
/// (see the module docs). No bind parameter: the expression names no `$n`,
/// so callers do not need to reserve a position for it.
///
/// - Postgres: `(col IS NULL OR col::timestamptz < now())` — the stored
///   [`LEASE_TS_FORMAT`] text is a valid `timestamptz` literal, so the cast
///   reads the same value the write in [`lease_deadline_expr`] produced,
///   compared against the database's own `now()`.
/// - SQLite (single process; the application clock IS the database's
///   clock): `(col IS NULL OR datetime(col) < datetime('now'))` —
///   `datetime()` normalizes both sides through SQLite's own clock function
///   rather than a value this process computed and bound in, so the SQL text
///   itself carries no timestamp literal either.
pub fn lease_expired_clause(col: &str, kind: BackendKind) -> String {
    match kind {
        BackendKind::Postgres => format!("({col} IS NULL OR {col}::timestamptz < now())"),
        BackendKind::Sqlite => format!("({col} IS NULL OR datetime({col}) < datetime('now'))"),
    }
}

/// The value-expression for stamping `lease_expires_at` to "`lease` from
/// now", using the backend's OWN clock on Postgres and the application clock
/// (through [`lease_deadline`], the one SQLite helper) on SQLite. Appends
/// whatever bind the expression needs to `params` and returns SQL text with
/// no leading `=`, ready to drop into `SET col = <expr>` or a `VALUES` list —
/// the counterpart [`lease_expired_clause`] later compares against.
///
/// - Postgres: `(now() + make_interval(secs => $n))::text` — the deadline is
///   computed entirely inside the database, from its own clock; `$n` binds
///   only the lease WINDOW (a duration), never a timestamp.
/// - SQLite: binds [`lease_deadline`]'s application-clock stamp directly.
pub fn lease_deadline_expr(
    kind: BackendKind,
    lease: Duration,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => {
            params.push(SqlValue::Float(lease.as_secs_f64()));
            format!("(now() + make_interval(secs => ${}))::text", params.len())
        }
        BackendKind::Sqlite => {
            params.push(SqlValue::TextOwned(lease_deadline(lease)));
            format!("${}", params.len())
        }
    }
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
    fn expired_clause_names_the_column_and_carries_no_bind() {
        assert_eq!(
            lease_expired_clause("lease_expires_at", BackendKind::Postgres),
            "(lease_expires_at IS NULL OR lease_expires_at::timestamptz < now())"
        );
        assert_eq!(
            lease_expired_clause("lease_expires_at", BackendKind::Sqlite),
            "(lease_expires_at IS NULL OR datetime(lease_expires_at) < datetime('now'))"
        );
    }

    #[test]
    fn deadline_expr_postgres_binds_only_the_duration_never_a_timestamp() {
        let mut params = Vec::new();
        let expr = lease_deadline_expr(BackendKind::Postgres, Duration::from_secs(30), &mut params);
        assert_eq!(expr, "(now() + make_interval(secs => $1))::text");
        assert_eq!(params.len(), 1, "one bind: the duration, not a timestamp");
        assert!(
            matches!(params[0], SqlValue::Float(secs) if secs == 30.0),
            "the bind must be the lease WINDOW, never an app-clock stamp: {:?}",
            params[0]
        );
    }

    #[test]
    fn deadline_expr_sqlite_binds_the_app_clock_deadline() {
        let mut params = Vec::new();
        let expr = lease_deadline_expr(BackendKind::Sqlite, Duration::from_secs(30), &mut params);
        assert_eq!(expr, "$1");
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn default_intervals_are_the_engine_constants() {
        let d = LeaseIntervals::default();
        assert_eq!(d.lease(), Duration::from_secs(30));
        assert_eq!(d.heartbeat(), Duration::from_secs(10));
    }
}
