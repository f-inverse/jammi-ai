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
    app_clock_now().format(LEASE_TS_FORMAT).to_string()
}

/// The application clock, read in ONE place. Every app-clock stamp and every
/// client-side comparison against a stored stamp (`lease_now`,
/// `lease_deadline`, `decode_lease_expires_at`'s and `last_seen_at_is_fresh`'s
/// `now`) derives from this call, so `jobs_repo.rs` never reads a clock of its
/// own — the property `assembly_outcome::cooldown_sql_has_no_second_clock_source`
/// pins by source scan.
pub fn app_clock_now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

/// `now + lease`, formatted as a lease deadline — the application-clock
/// stamp SQLite's lease writes bind directly. Never used to build a Postgres
/// lease predicate; see the module docs.
pub fn lease_deadline(lease: Duration) -> String {
    let expiry =
        app_clock_now() + chrono::Duration::from_std(lease).unwrap_or(chrono::Duration::MAX);
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

/// The SQL scalar expression for how many seconds remain in `col`'s lease
/// window — negative once expired, `NULL` when `col` itself is `NULL` (no
/// lease) — evaluated against the SAME clock [`lease_expired_clause`]
/// compares against: the backend's OWN `now()` on Postgres (stable for the
/// whole enclosing transaction, so a sibling `lease_expired_clause` call in
/// the same statement agrees with this one even though each names `now()`
/// independently), or the bound [`lease_now`] app-clock value on SQLite —
/// bound HERE, once, since two independent [`lease_now`] reads do not carry
/// Postgres's same-transaction guarantee.
///
/// `Catalog::get_job_for_rank` (`docs/rigor/contracts/feat_500-C-U5a-1.md` §
/// A6) reads this alongside [`lease_expired_clause`]'s own negation in ONE
/// statement, so "how much of
/// the window remains" is never a caller-side subtraction against its OWN
/// clock (SQLite: a replica-clock read no different from any other app-side
/// timestamp; Postgres: outright wrong, since only the database's `now()`
/// avoids replica skew — see this module's own docs) once bound to a
/// remaining-window value read back from a row a caller then acts on.
///
/// Honesty about the two backends' agreement: on Postgres this expression's
/// sign and [`lease_expired_clause`]'s boolean are exactly consistent (both
/// compare the SAME `col::timestamptz` against the SAME `now()` call within
/// one statement). On SQLite, `julianday(...)`'s floating-point day count
/// carries roughly sub-100-microsecond rounding at typical lease-scale
/// magnitudes relative to [`lease_expired_clause`]'s exact string compare
/// (`col < $bound`, `LEASE_TS_FORMAT`'s fixed-width text, byte-for-byte) —
/// negligible next to any `[lease] heartbeat_secs`/`duration_secs` a
/// deployment runs, but not bit-exact the way the Postgres arm is.
pub fn lease_remaining_seconds_expr(
    col: &str,
    kind: BackendKind,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => {
            // Postgres's `EXTRACT(...)` returns `numeric`, never `float8` —
            // an explicit `::double precision` cast is required so every
            // reader that decodes this column as `Option<f64>`
            // (`Catalog::get_job_for_rank`'s row mapper) gets the SQL type
            // it asked for; without it sqlx's Postgres decoder refuses the
            // row with a `ColumnDecode` error (a genuine backend/driver
            // fault, never a content defect) on every call, not just a
            // malformed one.
            format!("EXTRACT(EPOCH FROM ({col}::timestamptz - now()))::double precision")
        }
        BackendKind::Sqlite => {
            params.push(SqlValue::TextOwned(lease_now()));
            format!(
                "((julianday({col}) - julianday(${})) * 86400.0)",
                params.len()
            )
        }
    }
}

/// The outcome of decoding a lease/freshness TEXT column in RUST, never a
/// row FAULT — the [`super::jobs_repo::WorldSizeFact`] pattern applied to
/// `jobs.lease_expires_at` (via [`decode_lease_expires_at`]). A CLAIM/RECLAIM
/// write predicate ([`lease_expired_clause`], [`lease_remaining_seconds_expr`])
/// still re-parses the column IN SQL, against the backend's own clock — that
/// split is deliberate (see [`decode_lease_expires_at`]'s docs) and is the
/// resolution of <https://github.com/f-inverse/jammi-ai/issues/574>: on
/// Postgres, `col::timestamptz` faults the WHOLE statement the instant one
/// row's text does not parse, turning a garbage row into a read FAULT
/// (`Err`) there while SQLite's `julianday(...)` silently returns `NULL` for
/// the same text, turning it into `Ok(Some(row))` with `lease_live = false`
/// — the identical malformed row refusing `Unavailable` on one backend and
/// `FailedPrecondition` on the other. Decoding client-side, from the raw
/// TEXT, is infallible by construction on both backends: a value that does
/// not parse is `Undecodable`, a ROW FACT the caller (the gang admission
/// handler) refuses the same fixed way it refuses any other undecodable
/// content — never a fault of the read that found it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseFact {
    /// The column parsed and names an instant strictly after `now`: this
    /// many seconds remain.
    Live { remaining: Duration },
    /// The column was `NULL` (no lease — matching [`lease_expired_clause`]'s
    /// own `col IS NULL` arm: `NULL` means "remaining zero", never
    /// live-by-default) or parsed to an instant at or before `now`.
    Dead,
    /// The column held non-`NULL` text that did not parse as a timestamp
    /// under THIS backend's own write format (see [`decode_lease_expires_at`]).
    /// A ROW FACT about this claimant's own row content, never a fault of
    /// the read that found it.
    Undecodable,
}

impl LeaseFact {
    /// `true` only for [`Self::Live`] — the same predicate
    /// [`RankAdmissionRow::lease_live`](super::jobs_repo::RankAdmissionRow)
    /// used to carry directly; `Dead` and `Undecodable` both refuse
    /// admission, distinguished only for the `test-hooks` non-disclosure
    /// seam (`GangRefusalReason::{LeaseDead, LeaseUndecodable}`).
    pub fn is_live(self) -> bool {
        matches!(self, Self::Live { .. })
    }

    /// The remaining lease window, floored at zero for `Dead` and
    /// `Undecodable` alike.
    pub fn remaining(self) -> Duration {
        match self {
            Self::Live { remaining } => remaining,
            Self::Dead | Self::Undecodable => Duration::ZERO,
        }
    }
}

/// Parse an APPLICATION-CLOCK ISO-8601-with-trailing-`Z` stamp — the shape
/// every `now_sortable()` write uses (`instances.last_seen_at`,
/// `instances.started_at`: see `Catalog::upsert_instance` — that column is
/// NEVER database-clock stamped, on EITHER backend, so decoding it needs no
/// [`BackendKind`] at all) and the shape SQLite's own [`lease_now`] /
/// [`lease_deadline`] write for `jobs.lease_expires_at`. Accepts any
/// fractional-second width (`now_sortable` writes 9 digits, [`lease_now`] /
/// [`lease_deadline`] write 6 — [`LEASE_TS_FORMAT`] is a fixed-width special
/// case of this same shape); `None` for text that is not this shape at all.
fn parse_app_clock_stamp(text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.fZ")
        .ok()
        .map(|naive| chrono::DateTime::from_naive_utc_and_offset(naive, chrono::Utc))
}

/// Parse `jobs.lease_expires_at`'s stored text into a UTC instant, per the
/// shape THIS BACKEND's own write path ([`lease_deadline_expr`]) actually
/// stamps it in: `parse_app_clock_stamp` (an application-clock stamp) on
/// SQLite, Postgres's own default `timestamptz`-cast-to-`text` rendering (a
/// DATABASE-clock stamp — `YYYY-MM-DD HH:MM:SS[.ffffff]±HH[:MM]`, the
/// fractional part omitted entirely when it is exactly zero, the offset
/// without a leading zero and without a colon at whole-hour magnitudes, e.g.
/// `2026-09-15 22:46:57.675567-04` or `2026-01-01 00:00:00-05`, verified
/// against a live Postgres 16) on Postgres. `None` for text that does not
/// parse under that backend's own shape — [`LeaseFact::Undecodable`].
fn parse_lease_expires_at(kind: BackendKind, text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    match kind {
        BackendKind::Sqlite => parse_app_clock_stamp(text),
        BackendKind::Postgres => chrono::DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f%#z")
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc)),
    }
}

/// Decode `jobs.lease_expires_at`'s raw stored text (never a SQL-side
/// `col::timestamptz` cast — see [`LeaseFact`]'s docs) into a [`LeaseFact`]
/// against `now`, on either backend, infallibly.
///
/// **The split, and why it is the same clock discipline as
/// [`lease_expired_clause`] / [`lease_remaining_seconds_expr`].** Those two
/// functions stay exactly as they are and remain the ONLY lease predicate the
/// CLAIM (`Catalog::claim_next`) and RECLAIM (`Catalog::reclaim_expired_jobs`)
/// paths use: those paths WRITE — a claim believes a lease dead and takes the
/// row, a reclaim believes a claimant dead and fails it — so they must
/// compare against the ONE clock every replica agrees on (Postgres's own
/// `now()`, per this module's top-level docs) or risk reaping a claimant that
/// is very much alive. This function backs a READ-ONLY admission decision
/// (`Catalog::get_job_for_rank`, consumed by the gang `RunRank` handler) that
/// never writes or reaps anything: the worst an app-clock-relative answer
/// here can do is admit (or refuse) a rank a few hundred milliseconds earlier
/// or later than a hypothetical DB-clock answer would have — bounded by
/// ordinary inter-host clock skew, self-correcting at the very next
/// re-verification tick (`heartbeat`-cadence), and never destructive the way
/// a wrongful reap is. Both disciplines share the same rule: a WRITE that can
/// reap a live claimant always compares against the shared DB clock; a READ
/// that only refuses never does.
pub fn decode_lease_expires_at(
    kind: BackendKind,
    text: Option<&str>,
    now: chrono::DateTime<chrono::Utc>,
) -> LeaseFact {
    let Some(text) = text else {
        return LeaseFact::Dead;
    };
    match parse_lease_expires_at(kind, text) {
        None => LeaseFact::Undecodable,
        // `lease_expired_clause`'s predicate is `col < now()` (expired);
        // live is its negation, `col >= now()` — matching the OLD SQL-side
        // `remaining_secs >= 0.0` boundary exactly (`remaining_secs` was
        // `deadline - now` in seconds).
        Some(deadline) if deadline >= now => LeaseFact::Live {
            remaining: (deadline - now).to_std().unwrap_or(Duration::ZERO),
        },
        Some(_) => LeaseFact::Dead,
    }
}

/// Decode `instances.last_seen_at`'s raw stored text (never a SQL-side
/// `col::timestamptz` cast) into "is this row fresh" — present, parseable,
/// and within `margin` of `now` — against `now`, infallibly, on EITHER
/// backend: this column is ALWAYS an application-clock
/// `parse_app_clock_stamp` stamp (`Catalog::upsert_instance` /
/// `Catalog::reregister_instance` / `Catalog::touch_instance` all write
/// `now_sortable()`, never the database clock, on either backend), so unlike
/// [`decode_lease_expires_at`] this needs no [`BackendKind`] at all. Text
/// that does not parse is treated exactly like text that parses but is stale
/// — NOT fresh, a ROW FACT (`Catalog::fresh_instance`'s own docs: "`false`
/// for an absent OR a stale row alike... disclosing nothing about which" —
/// undecodable joins that same class) — never a read fault (the resolution
/// of <https://github.com/f-inverse/jammi-ai/issues/574> for this column:
/// [`stale_before_clause`]'s Postgres arm casts `last_seen_at::timestamptz`
/// in SQL and would otherwise fault the whole statement on the same
/// malformed text SQLite's `julianday(...)`-based comparison merely
/// evaluates to a stale answer for).
pub fn last_seen_at_is_fresh(
    last_seen_at: &str,
    margin: Duration,
    now: chrono::DateTime<chrono::Utc>,
) -> bool {
    match parse_app_clock_stamp(last_seen_at) {
        None => false,
        Some(seen) => {
            let margin = chrono::Duration::from_std(margin).unwrap_or(chrono::Duration::MAX);
            // `stale_before_clause`'s predicate is `col < cutoff` (stale);
            // fresh is its negation, `col >= cutoff` — never the strict
            // `>`, to agree exactly at the boundary.
            seen >= now - margin
        }
    }
}

/// The instance-liveness margin: `2 * lease`, against `instances.last_seen_at`
/// (the DB clock via [`stale_before_clause`]) — the same tolerance
/// `Catalog::reclaim_expired_jobs`'s inline-execution arm already computes
/// inline for "owning instance dead". Named here so a second caller (a gang
/// coordinator's own freshness check, `fresh_instance`) shares the SAME
/// margin rather than re-deriving the `2 *` factor at its own call site.
pub fn instance_liveness_margin(lease: Duration) -> Duration {
    lease.saturating_mul(2)
}

/// The instance PRUNE window: [`instance_liveness_margin`] (`2 * lease`)
/// plus one more `lease`, i.e. `3 * lease` — STRICTLY BEYOND the liveness
/// margin, so a member judged merely stale (`last_seen_at` in `(margin,
/// window]`) still keeps its row through at least one more sweep, giving the
/// lease keeper's `reregister_instance` re-upsert a chance to land before
/// `Catalog::prune_instances` reaps it. Before this function existed, the
/// only caller (`InferenceSession::wrap_with`) pruned at exactly the
/// liveness margin (`lease.saturating_mul(2)`) — a merely-stale member was
/// therefore ALREADY prune-eligible the instant it read stale, racing the
/// keeper's own recovery window to zero.
pub fn instance_prune_window(lease: Duration) -> Duration {
    instance_liveness_margin(lease).saturating_add(lease)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(
        y: i32,
        mo: u32,
        d: u32,
        h: u32,
        mi: u32,
        s: u32,
        micro: u32,
    ) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .with_ymd_and_hms(y, mo, d, h, mi, s)
            .single()
            .unwrap()
            + chrono::Duration::microseconds(micro as i64)
    }

    #[test]
    fn parse_app_clock_stamp_accepts_now_sortable_and_lease_ts_widths() {
        // `now_sortable()`'s 9-digit width.
        assert_eq!(
            parse_app_clock_stamp("2026-01-01T00:00:00.123456789Z"),
            Some(utc(2026, 1, 1, 0, 0, 0, 123456) + chrono::Duration::nanoseconds(789))
        );
        // `LEASE_TS_FORMAT`'s 6-digit width.
        assert_eq!(
            parse_app_clock_stamp("2026-01-01T00:00:00.000000Z"),
            Some(utc(2026, 1, 1, 0, 0, 0, 0))
        );
        assert_eq!(parse_app_clock_stamp("not-a-timestamp"), None);
        assert_eq!(parse_app_clock_stamp(""), None);
    }

    #[test]
    fn parse_lease_expires_at_postgres_accepts_its_own_default_text_rendering() {
        // Captured verbatim from a live Postgres 16:
        // `select (now() + make_interval(secs => 30))::text;`
        assert_eq!(
            parse_lease_expires_at(BackendKind::Postgres, "2026-09-15 22:46:57.675567-04"),
            Some(utc(2026, 9, 16, 2, 46, 57, 675567)),
        );
        // The fractional part is OMITTED ENTIRELY when it is exactly zero
        // (captured: `select ('2026-01-01 00:00:00'::timestamptz)::text`).
        assert_eq!(
            parse_lease_expires_at(BackendKind::Postgres, "2026-01-01 00:00:00-05"),
            Some(utc(2026, 1, 1, 5, 0, 0, 0)),
        );
        // A positive, two-part offset.
        assert_eq!(
            parse_lease_expires_at(BackendKind::Postgres, "2026-01-01 00:00:00+05:30"),
            Some(utc(2025, 12, 31, 18, 30, 0, 0)),
        );
        assert_eq!(
            parse_lease_expires_at(BackendKind::Postgres, "not-a-timestamp"),
            None,
        );
    }

    #[test]
    fn parse_lease_expires_at_sqlite_uses_the_app_clock_shape_only() {
        assert_eq!(
            parse_lease_expires_at(BackendKind::Sqlite, "2026-01-01T00:00:00.000000Z"),
            Some(utc(2026, 1, 1, 0, 0, 0, 0)),
        );
        // Postgres's OWN rendering must NOT parse under the SQLite arm (a
        // mismatch here would silently cross-decode a value this backend
        // never wrote in this shape).
        assert_eq!(
            parse_lease_expires_at(BackendKind::Sqlite, "2026-09-15 22:46:57.675567-04"),
            None,
        );
        assert_eq!(parse_lease_expires_at(BackendKind::Sqlite, "garbage"), None);
    }

    #[test]
    fn decode_lease_expires_at_is_infallible_on_every_input() {
        let now = utc(2026, 1, 1, 0, 0, 30, 0);
        // NULL column: Dead, matching `lease_expired_clause`'s own `IS NULL`
        // arm — never live-by-default.
        assert_eq!(
            decode_lease_expires_at(BackendKind::Postgres, None, now),
            LeaseFact::Dead
        );
        // Malformed text on EITHER backend: `Undecodable`, `Ok(Some(row))`
        // territory — never propagated as a read fault (issue #574).
        assert_eq!(
            decode_lease_expires_at(BackendKind::Postgres, Some("not-a-timestamp"), now),
            LeaseFact::Undecodable
        );
        assert_eq!(
            decode_lease_expires_at(BackendKind::Sqlite, Some("not-a-timestamp"), now),
            LeaseFact::Undecodable
        );
        // A deadline strictly in the future: Live, with the exact remaining
        // duration.
        let future = "2026-01-01T00:01:00.000000Z";
        assert_eq!(
            decode_lease_expires_at(BackendKind::Sqlite, Some(future), now),
            LeaseFact::Live {
                remaining: Duration::from_secs(30)
            }
        );
        // Exactly `now`: still Live (the boundary `lease_expired_clause`
        // itself draws: expired is strict `<`).
        let exactly_now = "2026-01-01T00:00:30.000000Z";
        assert_eq!(
            decode_lease_expires_at(BackendKind::Sqlite, Some(exactly_now), now),
            LeaseFact::Live {
                remaining: Duration::ZERO
            }
        );
        // A deadline in the past: Dead.
        let past = "2026-01-01T00:00:00.000000Z";
        assert_eq!(
            decode_lease_expires_at(BackendKind::Sqlite, Some(past), now),
            LeaseFact::Dead
        );
    }

    #[test]
    fn last_seen_at_is_fresh_matches_stale_before_clauses_boundary() {
        let now = utc(2026, 1, 1, 1, 0, 0, 0);
        let margin = Duration::from_secs(60);
        // Exactly at the margin: fresh (`stale_before_clause` is a strict
        // `<`; fresh is its negation, `>=`).
        assert!(last_seen_at_is_fresh(
            "2026-01-01T00:59:00.000000000Z",
            margin,
            now
        ));
        // One microsecond stale.
        assert!(!last_seen_at_is_fresh(
            "2026-01-01T00:58:59.999999000Z",
            margin,
            now
        ));
        // Malformed text: not fresh, never a fault (issue #574).
        assert!(!last_seen_at_is_fresh("not-a-timestamp", margin, now));
    }

    #[test]
    fn lease_fact_is_live_and_remaining() {
        assert!(LeaseFact::Live {
            remaining: Duration::from_secs(5)
        }
        .is_live());
        assert!(!LeaseFact::Dead.is_live());
        assert!(!LeaseFact::Undecodable.is_live());
        assert_eq!(LeaseFact::Dead.remaining(), Duration::ZERO);
        assert_eq!(LeaseFact::Undecodable.remaining(), Duration::ZERO);
    }

    #[test]
    fn instance_liveness_margin_is_twice_the_lease() {
        assert_eq!(
            instance_liveness_margin(Duration::from_secs(5)),
            Duration::from_secs(10)
        );
        // `saturating_mul`, never a wrapping/panicking overflow, at the
        // `Duration` ceiling — the same overflow shape
        // `reclaim_expired_jobs`'s own inline `lease.saturating_mul(2)`
        // relied on before this extraction.
        assert_eq!(instance_liveness_margin(Duration::MAX), Duration::MAX);
    }

    #[test]
    fn prune_window_is_strictly_beyond_the_liveness_margin() {
        for secs in [1u64, 5, 30, 3600] {
            let lease = Duration::from_secs(secs);
            let margin = instance_liveness_margin(lease);
            let window = instance_prune_window(lease);
            assert_eq!(window, Duration::from_secs(secs * 3), "3x lease exactly");
            assert!(
                window > margin,
                "the prune window ({window:?}) must be strictly beyond the \
                 liveness margin ({margin:?}) for lease {lease:?}"
            );
        }
        // `saturating_add`, never a wrapping/panicking overflow, at the
        // `Duration` ceiling.
        assert_eq!(instance_prune_window(Duration::MAX), Duration::MAX);
    }

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
    fn remaining_seconds_expr_postgres_binds_no_timestamp() {
        let mut params = Vec::new();
        let expr =
            lease_remaining_seconds_expr("lease_expires_at", BackendKind::Postgres, &mut params);
        assert_eq!(
            expr,
            "EXTRACT(EPOCH FROM (lease_expires_at::timestamptz - now()))::double precision"
        );
        assert!(
            params.is_empty(),
            "Postgres's remaining-seconds expression must bind no timestamp: {params:?}"
        );
    }

    #[test]
    fn remaining_seconds_expr_sqlite_binds_the_app_clock_once() {
        let mut params = Vec::new();
        let expr =
            lease_remaining_seconds_expr("lease_expires_at", BackendKind::Sqlite, &mut params);
        assert_eq!(
            expr,
            "((julianday(lease_expires_at) - julianday($1)) * 86400.0)"
        );
        assert_eq!(params.len(), 1, "one bind: lease_now(), the app clock");
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
