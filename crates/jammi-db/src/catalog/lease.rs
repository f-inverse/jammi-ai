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

/// Format SQLite's app-clock lease stamps write into `lease_expires_at`
/// ([`lease_now`] / [`lease_deadline`], the SQLite arm's one helper).
/// Lexicographic ordering of two timestamps in this fixed-width UTC form
/// matches chronological ordering, so [`lease_expired_clause`]'s SQLite arm
/// — [`lease_now`] bound as a parameter, compared with a plain string `<` —
/// is exact at full microsecond precision. SQLite itself has no SQL-visible
/// clock finer than milliseconds (`datetime('now')` truncates to whole
/// seconds; `strftime('%f','now')` and `unixepoch('now','subsec')` cap out
/// at milliseconds; `julianday('now')`'s double-precision day count loses
/// sub-~100-microsecond resolution to floating-point rounding), so a
/// no-bind comparison against any SQLite clock function silently fails to
/// distinguish two stamps computed back-to-back with no real work between
/// them — exactly the shape a test forging a near-zero-duration lease
/// relies on, even though a real deployment's lease (tens of seconds,
/// `heartbeat * 2 < lease`) would have tolerated any of those truncations.
///
/// **Postgres stores something else entirely for this column.** A Postgres
/// lease stamp is `(now() + make_interval(secs => $n))::text`
/// ([`lease_deadline_expr`]) — Postgres's OWN default `timestamptz` text
/// rendering (space-separated, zone-suffixed), not this format, and every
/// comparison casts back through `col::timestamptz` rather than comparing
/// the stored strings lexicographically at all. `LEASE_TS_FORMAT` names the
/// SQLite shape only; do not assume a `result_tables.lease_expires_at` value
/// is in this format without checking which backend wrote it.
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
/// against the BACKEND's own clock — never a bound application timestamp on
/// Postgres (see the module docs).
///
/// - Postgres: `(col IS NULL OR col::timestamptz < now())` — the stored
///   [`LEASE_TS_FORMAT`] text is a valid `timestamptz` literal, so the cast
///   reads the same value the write in [`lease_deadline_expr`] produced,
///   compared against the database's own `now()`. Appends NO bind: the
///   expression names no `$n` at all on this backend.
/// - SQLite (single process; the application clock IS the "database"
///   clock — there is no peer replica to skew against): appends
///   [`lease_now`] as ONE bind and compares `(col IS NULL OR col < $n)` —
///   full [`LEASE_TS_FORMAT`] microsecond precision, lexicographically.
///   Deliberately NOT a no-bind SQLite clock function: `datetime('now')`
///   truncates to WHOLE SECONDS, `strftime('%f','now')` and
///   `unixepoch('now','subsec')` both cap out at 3-digit MILLISECOND
///   precision, and `julianday('now')`'s double-precision day count loses
///   sub-~100-microsecond resolution to floating-point rounding — every one
///   of those silently fails to distinguish two stamps computed
///   back-to-back with no real work between them, exactly the shape a test
///   forging a near-zero-duration lease relies on (and the shape
///   `heartbeat * 2 < lease` guarantees never occurs in a real deployment,
///   where truncation to whole seconds would be harmless). SQLite has no
///   SQL-visible clock finer than milliseconds, so matching this format's
///   microsecond precision requires evaluating [`lease_now`] in the
///   application and binding it — the "keep the app clock through ONE
///   helper" alternative for the single-process backend.
pub fn lease_expired_clause(
    col: &str,
    kind: BackendKind,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => format!("({col} IS NULL OR {col}::timestamptz < now())"),
        BackendKind::Sqlite => {
            params.push(SqlValue::TextOwned(lease_now()));
            format!("({col} IS NULL OR {col} < ${})", params.len())
        }
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

/// The SQL predicate for "`col` is at least `margin` in the past" —
/// `now() - margin > col`, on the backend's own clock on Postgres and the
/// application clock (one bound parameter, this module's [`LEASE_TS_FORMAT`])
/// on SQLite — the general form of [`lease_expired_clause`]'s fixed
/// `margin = 0` comparison, minus that function's `col IS NULL` arm: every
/// caller here (`instances.last_seen_at`, `jobs.updated_at`) requires the
/// column non-null by construction, so there is no absent-lease case to admit.
///
/// Used by the job-worker liveness reclaim (a `2 * lease.duration` margin
/// against `instances.last_seen_at`), the instance-staleness sweep, and the
/// job-retention sweep/predicate (a `retention_days` margin against
/// `jobs.updated_at`) — one shared clock-arithmetic primitive for every
/// "how long ago" comparison outside the lease-deadline family above.
pub fn stale_before_clause(
    col: &str,
    kind: BackendKind,
    margin: Duration,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => {
            params.push(SqlValue::Float(margin.as_secs_f64()));
            format!(
                "{col}::timestamptz < (now() - make_interval(secs => ${}))",
                params.len()
            )
        }
        BackendKind::Sqlite => {
            let cutoff = (chrono::Utc::now()
                - chrono::Duration::from_std(margin).unwrap_or(chrono::Duration::MAX))
            .format(LEASE_TS_FORMAT)
            .to_string();
            params.push(SqlValue::TextOwned(cutoff));
            format!("{col} < ${}", params.len())
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
    fn expired_clause_postgres_carries_no_bind_and_names_the_column() {
        let mut params = Vec::new();
        let clause = lease_expired_clause("lease_expires_at", BackendKind::Postgres, &mut params);
        assert_eq!(
            clause,
            "(lease_expires_at IS NULL OR lease_expires_at::timestamptz < now())"
        );
        assert!(
            params.is_empty(),
            "Postgres's clause must bind no timestamp: {params:?}"
        );
    }

    #[test]
    fn expired_clause_sqlite_binds_the_app_clock_now() {
        let mut params = Vec::new();
        let clause = lease_expired_clause("lease_expires_at", BackendKind::Sqlite, &mut params);
        assert_eq!(
            clause,
            "(lease_expires_at IS NULL OR lease_expires_at < $1)"
        );
        assert_eq!(params.len(), 1, "one bind: lease_now(), the app clock");
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

    #[test]
    fn stale_before_clause_postgres_binds_only_the_margin_seconds() {
        let mut params = Vec::new();
        let clause = stale_before_clause(
            "last_seen_at",
            BackendKind::Postgres,
            Duration::from_secs(60),
            &mut params,
        );
        assert_eq!(
            clause,
            "last_seen_at::timestamptz < (now() - make_interval(secs => $1))"
        );
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], SqlValue::Float(secs) if secs == 60.0));
    }

    #[test]
    fn stale_before_clause_sqlite_binds_the_app_clock_cutoff() {
        let mut params = Vec::new();
        let clause = stale_before_clause(
            "updated_at",
            BackendKind::Sqlite,
            Duration::from_secs(60),
            &mut params,
        );
        assert_eq!(clause, "updated_at < $1");
        assert_eq!(params.len(), 1);
        match &params[0] {
            SqlValue::TextOwned(s) => assert_eq!(s.len(), lease_now().len(), "fixed-width form"),
            other => panic!("expected a TextOwned cutoff bind, got {other:?}"),
        }
    }
}
