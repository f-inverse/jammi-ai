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
//! through this module's [`canonical_stamp_now`] / [`lease_deadline`] pair — the ONE
//! helper every SQLite lease stamp and comparison goes through.

use std::time::Duration;

use crate::catalog::backend::{BackendKind, SqlValue};

/// Format SQLite's app-clock lease stamps write into `lease_expires_at`
/// ([`canonical_stamp_now`] / [`lease_deadline`], the SQLite arm's one helper).
/// Lexicographic ordering of two timestamps in this fixed-width UTC form
/// matches chronological ordering, so [`lease_expired_clause`]'s SQLite arm
/// — [`canonical_stamp_now`] bound as a parameter, compared with a plain string `<` —
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
/// **Postgres writes the SAME shape, through a different mechanism.** A
/// Postgres lease stamp is [`pg_canonical_stamp`]`("now() + make_interval(secs
/// => $n)")` ([`lease_deadline_expr`]) — computed from the database's own
/// clock, then rendered through the identical [`LEASE_TS_FORMAT`] picture via
/// `to_char`, so the stored TEXT is byte-for-byte the same shape this
/// function produces. Every comparison on Postgres still casts back through
/// `col::timestamptz` (this module's clock discipline — see "Whose clock"
/// above — never a lexical compare there), but the STORED shape itself is
/// now backend-independent: a `result_tables.lease_expires_at` value is in
/// this format regardless of which backend wrote it.
pub const LEASE_TS_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.6fZ";

/// `now`, rendered in [`LEASE_TS_FORMAT`] — the ONE canonical catalog stamp
/// (`CANONICAL_STAMP`), UTC, `T`-separated, exactly six fraction digits, a
/// literal `Z`. This is the ONE application-side formatter every TEXT
/// timestamp column this crate writes goes through, whether or not the
/// column happens to carry "lease" in its name: a SQLite lease deadline
/// ([`lease_deadline`]), an `instances.started_at`/`last_seen_at` stamp
/// (`Catalog::upsert_instance`), a `jobs.created_at`/`updated_at` stamp, and
/// every other app-clock `*_at` column bind through this function so the
/// catalog never grows a second timestamp shape by accident. The Postgres
/// SQL-side counterpart is [`pg_canonical_stamp`] — the two together are the
/// stamp's ONLY two producers, on either backend.
pub fn canonical_stamp_now() -> String {
    app_clock_now().format(LEASE_TS_FORMAT).to_string()
}

/// The Postgres SQL expression that renders `expr` (any SQL expression
/// evaluating to a `timestamp`/`timestamptz`) as [`LEASE_TS_FORMAT`]'s exact
/// text shape: UTC, `T`-separated, exactly six fraction digits, a literal
/// `Z` — `to_char`, with an explicit picture, so the rendering never depends
/// on the session's `DateStyle`/`TimeZone` GUCs the way `::text` does. This
/// is the ONE Postgres-side stamp renderer; every Postgres writer of an
/// in-class TEXT timestamp column composes its value through this function
/// (see [`lease_deadline_expr`]), never a bare `::text` cast.
pub fn pg_canonical_stamp(expr: &str) -> String {
    format!("to_char(({expr}) AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"')")
}

/// The application clock, read in ONE place. Every app-clock stamp and every
/// client-side comparison against a stored stamp (`canonical_stamp_now`,
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
///   [`canonical_stamp_now`] as ONE bind and compares `(col IS NULL OR col < $n)` —
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
///   microsecond precision requires evaluating [`canonical_stamp_now`] in the
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
            params.push(SqlValue::TextOwned(canonical_stamp_now()));
            format!("({col} IS NULL OR {col} < ${})", params.len())
        }
    }
}

/// The SQL fragment that is true for a LIVE (non-expired, non-absent)
/// lease — the exact complement of [`lease_expired_clause`], `col IS NOT
/// NULL AND col > now` rather than that function's `col IS NULL OR col <
/// now`, on the SAME backend clock. `Catalog::transfer_claim`'s hand-off
/// predicate needs THIS positive polarity, never
/// [`lease_expired_clause`]'s: a RELEASE ([`super::jobs_repo::Catalog::
/// release_job_lease`]) sets `lease_expires_at = NULL`, and a transfer of a
/// released claim must FAIL — `lease_expired_clause`'s own `col IS NULL OR
/// …` shape reads a NULL lease as "expired" (true), which is the right
/// answer for a reclaim sweep deciding whether to requeue a job but the
/// WRONG answer for a hand-off deciding whether a live claim exists to
/// transfer; negating `lease_expired_clause` as a whole would still leave
/// the boundary at exactly `now` ambiguous between the two functions'
/// independent per-backend expressions, so this is its own, explicitly
/// authored predicate rather than `format!("NOT {}", lease_expired_clause(..))`.
pub fn lease_live_clause(
    col: &str,
    kind: BackendKind,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => format!("({col} IS NOT NULL AND {col}::timestamptz > now())"),
        BackendKind::Sqlite => {
            params.push(SqlValue::TextOwned(canonical_stamp_now()));
            format!("({col} IS NOT NULL AND {col} > ${})", params.len())
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
/// - Postgres: [`pg_canonical_stamp`]`("now() + make_interval(secs => $n)")`
///   — the deadline is computed entirely inside the database, from its own
///   clock, then rendered in `CANONICAL_STAMP` shape so this column holds the
///   identical text shape on either backend; `$n` binds only the lease
///   WINDOW (a duration), never a timestamp.
/// - SQLite: binds [`lease_deadline`]'s application-clock stamp directly.
pub fn lease_deadline_expr(
    kind: BackendKind,
    lease: Duration,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    match kind {
        BackendKind::Postgres => {
            params.push(SqlValue::Float(lease.as_secs_f64()));
            pg_canonical_stamp(&format!("now() + make_interval(secs => ${})", params.len()))
        }
        BackendKind::Sqlite => {
            params.push(SqlValue::TextOwned(lease_deadline(lease)));
            format!("${}", params.len())
        }
    }
}

/// A timestamp column the schema holds to the canonical fixed-width UTC shape
/// on every row, existing rows included (the canonical-stamps migration rewrites
/// them, then installs the constraint). That shape is what makes a lexical
/// comparison chronological, so [`stale_before_clause`] — which compares
/// without a cast, to stay sargable — accepts only these. A caller with a new
/// column adds a variant here, and the migration suite refuses the variant
/// until the schema enforces that column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::VariantArray)]
pub enum CanonicalStampColumn {
    InstancesLastSeenAt,
    JobsUpdatedAt,
}

impl CanonicalStampColumn {
    /// The `(table, column)` this variant names.
    pub const fn table_and_column(self) -> (&'static str, &'static str) {
        match self {
            Self::InstancesLastSeenAt => ("instances", "last_seen_at"),
            Self::JobsUpdatedAt => ("jobs", "updated_at"),
        }
    }

    /// The column as SQL, qualified by the statement's alias for its table
    /// when it has one.
    fn qualified(self, alias: Option<&str>) -> String {
        let (_, column) = self.table_and_column();
        match alias {
            Some(alias) => format!("{alias}.{column}"),
            None => column.to_string(),
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
/// **Sargability.** The Postgres arm renders the cutoff DB-side, in
/// [`LEASE_TS_FORMAT`] TEXT, via [`pg_canonical_stamp`], and compares it
/// against `col` as TEXT — no cast on `col` at all. Every writer of an
/// in-class `*_at` column renders the SAME fixed-width UTC shape (this
/// module's docs; the schema-edge CHECK constraint enforces it), so
/// lexicographic TEXT order equals chronological order and the comparison is
/// exact. Measured at 200k `instances`-shaped rows on the scratch Postgres
/// host (`idx_instances_seen`, a plain btree on the TEXT column): the
/// PRIOR form — `col::timestamptz < (now() - make_interval(...))`, casting
/// `col` — forces a Seq Scan (no btree supports an index condition on a
/// CAST of the indexed column) at ~80 ms for a low-selectivity ("who is
/// still live") predicate; this form is an Index (Only) Scan under 0.1 ms,
/// same row set. [`crate::catalog::Catalog::list_gang_members`] and
/// `super::instance::live_with_root_clause` (the ring predicate this
/// module's docs describe) both wrap this in `NOT (...)` for their
/// low-selectivity "who is live" read, which is exactly the case the cast
/// form scanned the whole table for.
pub fn stale_before_clause(
    column: CanonicalStampColumn,
    alias: Option<&str>,
    kind: BackendKind,
    margin: Duration,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    let col = column.qualified(alias);
    match kind {
        BackendKind::Postgres => {
            params.push(SqlValue::Float(margin.as_secs_f64()));
            format!(
                "{col} < {}",
                pg_canonical_stamp(&format!("now() - make_interval(secs => ${})", params.len()))
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
/// independently), or the bound [`canonical_stamp_now`] app-clock value on SQLite —
/// bound HERE, once, since two independent [`canonical_stamp_now`] reads do not carry
/// Postgres's same-transaction guarantee.
///
/// `Catalog::get_job_for_rank` reads this alongside [`lease_expired_clause`]'s
/// own negation in ONE statement, so "how much of
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
            params.push(SqlValue::TextOwned(canonical_stamp_now()));
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

/// Parse a `CANONICAL_STAMP`-shaped ISO-8601-with-trailing-`Z`
/// stamp — the ONE shape every writer produces on EITHER backend since S1/S3
/// (`instances.last_seen_at`, `instances.started_at`: see
/// `Catalog::upsert_instance`; SQLite's [`canonical_stamp_now`] /
/// [`lease_deadline`] for `jobs.lease_expires_at`; and, after the schema-edge
/// domain (migration `039_canonical_stamps`) makes it the ONLY representable
/// shape, Postgres's own `lease_expires_at` text too — see
/// [`parse_lease_expires_at`]). Accepts any fractional-second width up to six
/// digits ([`LEASE_TS_FORMAT`] is the fixed six-digit case this crate always
/// writes); `None` for text that is not this shape at all, OR that names a
/// calendar instant chrono refuses — e.g. a month field of `13`,
/// `…T00:00:00.000000Z` with `…` = `2026-13-01`: the schema-edge CHECK's
/// regex (`S3`) is shape-only and admits it, and this writer never produces
/// one, but a stored value like it decodes as [`LeaseFact::Undecodable`], its
/// documented home. (A leap second, `…T23:59:60.000000Z`, was considered for
/// this role and executed against both chrono and this crate's decode path:
/// chrono's `%S`/`%.f` parser DOES accept `:60` and represents it as a valid
/// `NaiveTime` — `decode_lease_expires_at` reads it as an ordinary, slightly
/// later instant, never `Undecodable` — so it is not this arm's example.)
fn parse_app_clock_stamp(text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.fZ")
        .ok()
        .map(|naive| chrono::DateTime::from_naive_utc_and_offset(naive, chrono::Utc))
}

/// Parse `jobs.lease_expires_at`'s stored text into a UTC instant.
///
/// Before migration `039_canonical_stamps`, Postgres wrote this column in its
/// own default `timestamptz`-cast-to-`text` rendering (a DATABASE-clock
/// stamp, DateStyle/TimeZone-dependent) while SQLite wrote the application
/// clock through [`canonical_stamp_now`]/[`lease_deadline`] — two shapes, one
/// per backend. `039` fixes the WRITER ([`pg_canonical_stamp`] renders every
/// Postgres write in [`LEASE_TS_FORMAT`] too) and enforces the domain at the
/// schema edge on both backends, so after it there is exactly ONE shape this
/// column can hold regardless of which backend wrote it — `kind` is no longer
/// needed to choose a parser, and this function is [`parse_app_clock_stamp`]
/// under a name that documents which column it decodes. `None` for text that
/// does not parse — [`LeaseFact::Undecodable`].
fn parse_lease_expires_at(text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    parse_app_clock_stamp(text)
}

/// Decode `jobs.lease_expires_at`'s raw stored text (never a SQL-side
/// `col::timestamptz` cast — see [`LeaseFact`]'s docs) into a [`LeaseFact`]
/// against `now`, on either backend, infallibly. Takes no [`BackendKind`]:
/// since migration `039_canonical_stamps` both backends store the identical
/// `CANONICAL_STAMP` shape (see `parse_lease_expires_at`).
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
    text: Option<&str>,
    now: chrono::DateTime<chrono::Utc>,
) -> LeaseFact {
    let Some(text) = text else {
        return LeaseFact::Dead;
    };
    match parse_lease_expires_at(text) {
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
/// [`canonical_stamp_now`], never the database clock, on either backend), so unlike
/// [`decode_lease_expires_at`] this needs no [`BackendKind`] at all. Text
/// that does not parse is treated exactly like text that parses but is stale
/// — NOT fresh, a ROW FACT (`Catalog::fresh_instance`'s own docs: "`false`
/// for an absent OR a stale row alike... disclosing nothing about which" —
/// undecodable joins that same class) — never a read fault (the resolution
/// of <https://github.com/f-inverse/jammi-ai/issues/574>, still honoured
/// here even though [`stale_before_clause`]'s Postgres arm no longer casts
/// this column at all: the SQL-side predicate is now a lexical TEXT
/// comparison against a DB-side-rendered canonical cutoff, sargable and
/// cast-free — see that function's own doc for the measured Seq-Scan-to-
/// Index-Scan fix. The column's domain is guarded by the schema-edge CHECK
/// `sdchk__instances__last_seen_at` (`schema.rs`), which validates shape AND
/// calendar-validity (via its own internal cast) at WRITE time on Postgres,
/// so a non-`NULL` `last_seen_at` the SQL predicate reads is already
/// guaranteed well-formed there — this function's own INFALLIBLE, no-fault
/// decode exists for the read path regardless, since the CHECK is a
/// Postgres-only, write-time guarantee: SQLite's own write-time guard
/// (`trg_instances_last_seen_at_canonical_ins`/`_upd`, `schema.rs`) enforces
/// only the canonical SHAPE (a `GLOB` pattern; a mismatch raises
/// `BackendError::DomainViolation` — `backend.rs`'s
/// `SQLITE_CONSTRAINT_TRIGGER_CODE` arm), never calendar validity: it has no
/// cast, so a shape-valid, calendar-invalid stamp such as
/// `0000-00-00T00:00:00.000000Z` still passes it. Neither guard is
/// retroactive, and a value that predates either one (or a column this crate
/// does not itself write) can still reach this decoder malformed on either
/// backend.
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
    fn parse_app_clock_stamp_accepts_any_fractional_width_up_to_nine_digits() {
        // A nine-digit width (no writer in this crate produces one, but a
        // column this crate does not enforce the domain of — an out-of-class
        // `*_at` column, S3's "no opt-in" universe gate names the class —
        // could still hold a pre-existing value at this width).
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
    fn parse_lease_expires_at_is_the_one_canonical_shape_on_either_backend() {
        // Since migration `039_canonical_stamps` this is the ONLY shape the
        // column can hold, on EITHER backend — `parse_lease_expires_at` takes
        // no `BackendKind` (see its docs).
        assert_eq!(
            parse_lease_expires_at("2026-01-01T00:00:00.000000Z"),
            Some(utc(2026, 1, 1, 0, 0, 0, 0)),
        );
        // Postgres's PRE-039 default rendering (space-separated, zone offset)
        // must NOT parse: it is not a shape any writer produces once the
        // schema-edge domain is in force, and a decoder that silently
        // accepted it would defeat the point of collapsing the parser.
        assert_eq!(
            parse_lease_expires_at("2026-09-15 22:46:57.675567-04"),
            None,
        );
        assert_eq!(parse_lease_expires_at("not-a-timestamp"), None);
        assert_eq!(parse_lease_expires_at("garbage"), None);
        // Shape-valid, calendar-invalid: the schema-edge CHECK's regex
        // (`S3`, shape only) admits a month of `13`, but no calendar has one
        // — this writer never produces such a value, and a stored one
        // decodes as `Undecodable` (see
        // `decode_lease_expires_at_treats_a_calendar_invalid_stamp_as_undecodable`).
        // A leap second (`…T23:59:60.000000Z`) was tried for this role
        // first and executed against chrono directly: chrono's `%S`/`%.f`
        // parser ACCEPTS `:60` as a valid `NaiveTime`, so it is not
        // calendar-invalid from this decoder's point of view and is not
        // used here.
        assert_eq!(parse_lease_expires_at("2026-13-01T00:00:00.000000Z"), None);
    }

    #[test]
    fn decode_lease_expires_at_is_infallible_on_every_input() {
        let now = utc(2026, 1, 1, 0, 0, 30, 0);
        // NULL column: Dead, matching `lease_expired_clause`'s own `IS NULL`
        // arm — never live-by-default.
        assert_eq!(decode_lease_expires_at(None, now), LeaseFact::Dead);
        // Malformed text: `Undecodable`, `Ok(Some(row))` territory — never
        // propagated as a read fault (issue #574), identically on either
        // backend since both store the same shape.
        assert_eq!(
            decode_lease_expires_at(Some("not-a-timestamp"), now),
            LeaseFact::Undecodable
        );
        // A deadline strictly in the future: Live, with the exact remaining
        // duration.
        let future = "2026-01-01T00:01:00.000000Z";
        assert_eq!(
            decode_lease_expires_at(Some(future), now),
            LeaseFact::Live {
                remaining: Duration::from_secs(30)
            }
        );
        // Exactly `now`: still Live (the boundary `lease_expired_clause`
        // itself draws: expired is strict `<`).
        let exactly_now = "2026-01-01T00:00:30.000000Z";
        assert_eq!(
            decode_lease_expires_at(Some(exactly_now), now),
            LeaseFact::Live {
                remaining: Duration::ZERO
            }
        );
        // A deadline in the past: Dead.
        let past = "2026-01-01T00:00:00.000000Z";
        assert_eq!(decode_lease_expires_at(Some(past), now), LeaseFact::Dead);
    }

    /// The shape-valid/calendar-invalid determinant named in
    /// `parse_lease_expires_at`'s docs: the schema-edge CHECK's regex (`S3`)
    /// is shape-only and admits a month of `13`, but no writer this crate
    /// owns ever produces one, and no calendar has a thirteenth month, so it
    /// decodes as `Undecodable` rather than panicking or silently
    /// misreading a different month.
    #[test]
    fn decode_lease_expires_at_treats_a_calendar_invalid_stamp_as_undecodable() {
        let now = utc(2026, 6, 30, 23, 59, 0, 0);
        assert_eq!(
            decode_lease_expires_at(Some("2026-13-01T00:00:00.000000Z"), now),
            LeaseFact::Undecodable
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
        let now = canonical_stamp_now();
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
        assert_eq!(
            params.len(),
            1,
            "one bind: canonical_stamp_now(), the app clock"
        );
    }

    #[test]
    fn deadline_expr_postgres_binds_only_the_duration_never_a_timestamp() {
        let mut params = Vec::new();
        let expr = lease_deadline_expr(BackendKind::Postgres, Duration::from_secs(30), &mut params);
        assert_eq!(
            expr,
            "to_char((now() + make_interval(secs => $1)) AT TIME ZONE 'UTC', \
             'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"')"
        );
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
        assert_eq!(
            params.len(),
            1,
            "one bind: canonical_stamp_now(), the app clock"
        );
    }

    #[test]
    fn default_intervals_are_the_engine_constants() {
        let d = LeaseIntervals::default();
        assert_eq!(d.lease(), Duration::from_secs(30));
        assert_eq!(d.heartbeat(), Duration::from_secs(10));
    }

    /// The PROPERTY (rewritten from a pinned clause string, RENDEZVOUS RV3):
    /// exactly one bind, carrying the margin in seconds untouched by the
    /// rewrite; the rendered clause casts NEITHER side of the comparison —
    /// `col` appears bare (never `col::timestamptz`), so a btree on `col`
    /// (TEXT) can serve it — and the cutoff is expressed through
    /// [`pg_canonical_stamp`], the SAME renderer [`lease_deadline_expr`]
    /// uses, so every DB-side stamp in this crate is produced by exactly two
    /// functions (this one, and the app-side [`canonical_stamp_now`]). The
    /// live sargability + row-set oracle (`EXPLAIN`, and a differential
    /// against the old cast form) is
    /// `stale_before_clause_postgres_is_sargable_and_agrees_with_the_cast_form`
    /// below.
    #[test]
    fn stale_before_clause_postgres_binds_only_the_margin_seconds_and_casts_neither_side() {
        let mut params = Vec::new();
        let clause = stale_before_clause(
            CanonicalStampColumn::InstancesLastSeenAt,
            None,
            BackendKind::Postgres,
            Duration::from_secs(60),
            &mut params,
        );
        assert!(
            !clause.contains("::timestamptz") && !clause.contains("::text"),
            "the rendered clause must cast neither operand: {clause}"
        );
        assert!(
            clause.starts_with("last_seen_at < "),
            "col must appear bare, never wrapped, so a btree on col serves it: {clause}"
        );
        assert!(
            clause.contains("to_char("),
            "the cutoff must be rendered through pg_canonical_stamp: {clause}"
        );
        assert_eq!(params.len(), 1, "one bind: the margin in seconds");
        assert!(matches!(params[0], SqlValue::Float(secs) if secs == 60.0));
    }

    /// Live (requires `JAMMI_TEST_PG_URL`; skips, never fails, otherwise):
    /// (1) `EXPLAIN` over an `instances`-shaped fixture (the real
    /// `idx_instances_seen` btree on a TEXT `last_seen_at`) shows an Index
    /// (Only) Scan for the rewritten clause and a Seq Scan for the old cast
    /// form, for the SAME low-selectivity "who is still live" read every
    /// caller (`list_gang_members`, the RENDEZVOUS ring,
    /// `reclaim_expired_jobs`'s instance-liveness arm, `prune_instances`,
    /// `prune_jobs`'s retention arm) makes; (2) the two forms return the
    /// IDENTICAL row set at one frozen instant, including a row planted
    /// EXACTLY at the margin boundary (the boundary a lexical `<` and a
    /// `timestamptz <` must agree on bit-for-bit, since both compare the same
    /// canonical stamp text).
    #[tokio::test]
    async fn stale_before_clause_postgres_is_sargable_and_agrees_with_the_cast_form() {
        let Some(url) = jammi_test_utils::pg_url_for_tests() else {
            eprintln!(
                "skipping stale_before_clause_postgres_is_sargable_and_agrees_with_the_cast_form: \
                 JAMMI_TEST_PG_URL unset"
            );
            return;
        };
        // `max_connections(1)` PLUS one explicit transaction for the WHOLE
        // test: a `CREATE TEMP TABLE` is visible only on the connection that
        // created it, and Postgres's `now()` is stable for the lifetime of a
        // transaction — so the boundary stamp this test plants (captured via
        // one `now()` read) and every later `now()` the two clause forms
        // themselves evaluate resolve to the IDENTICAL instant, making the
        // exact-microsecond boundary assertion deterministic rather than a
        // race against wall-clock drift between statements.
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to the scratch Postgres");
        let mut tx = pool.begin().await.unwrap();
        let table = format!("lease_rs_stale_before_clause_{}", std::process::id());
        sqlx::query(&format!(
            "CREATE TEMP TABLE {table} (last_seen_at TEXT NOT NULL)"
        ))
        .execute(&mut *tx)
        .await
        .unwrap();
        sqlx::query(&format!("CREATE INDEX ON {table} (last_seen_at)"))
            .execute(&mut *tx)
            .await
            .unwrap();
        let margin = Duration::from_secs(60);
        let margin_secs = margin.as_secs_f64();
        // The exact-microsecond boundary, computed by THIS transaction's own
        // stable `now()` via the SAME renderer `stale_before_clause` uses
        // (`pg_canonical_stamp`) — never Rust's app clock, which would race
        // the DB's `now()` by however long the round trip takes.
        let boundary: String = sqlx::query_scalar(&format!(
            "SELECT {}",
            pg_canonical_stamp("now() - make_interval(secs => $1)")
        ))
        .bind(margin_secs)
        .fetch_one(&mut *tx)
        .await
        .unwrap();
        // One row per second over ~14 hours (relative to the SAME stable
        // `now()`), so a 60 s liveness margin selects a small, realistic
        // fraction — the exact shape a fleet's `instances` table has (mostly
        // stale rows from processes long gone, a handful still live).
        let now_text: String = sqlx::query_scalar(
            "SELECT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"')",
        )
        .fetch_one(&mut *tx)
        .await
        .unwrap();
        let now = chrono::DateTime::parse_from_rfc3339(&now_text)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| {
                chrono::NaiveDateTime::parse_from_str(&now_text, "%Y-%m-%dT%H:%M:%S%.fZ")
                    .map(|naive| chrono::DateTime::from_naive_utc_and_offset(naive, chrono::Utc))
                    .expect("the DB's own canonical-stamp rendering must parse back")
            });
        let mut stamps: Vec<String> = (0i64..50_000)
            .map(|s| {
                (now - chrono::Duration::seconds(s))
                    .format(LEASE_TS_FORMAT)
                    .to_string()
            })
            .collect();
        // The exact-microsecond boundary: a row stamped precisely at the
        // cutoff is LIVE (`stale_before_clause` is strict `<`; live is `>=`).
        stamps.push(boundary.clone());
        sqlx::query(&format!(
            "INSERT INTO {table} (last_seen_at) SELECT * FROM UNNEST($1::text[])"
        ))
        .bind(&stamps)
        .execute(&mut *tx)
        .await
        .unwrap();
        sqlx::query(&format!("ANALYZE {table}"))
            .execute(&mut *tx)
            .await
            .unwrap();

        let mut new_params = Vec::new();
        let new_clause = stale_before_clause(
            CanonicalStampColumn::InstancesLastSeenAt,
            None,
            BackendKind::Postgres,
            margin,
            &mut new_params,
        );
        // The OLD (pre-rewrite) form, reconstructed verbatim here (never
        // reachable from production code any more) purely as this oracle's
        // baseline.
        let old_clause = "last_seen_at::timestamptz < (now() - make_interval(secs => $1))";

        let plan_new: Vec<String> = sqlx::query_scalar(&format!(
            "EXPLAIN SELECT last_seen_at FROM {table} WHERE NOT ({new_clause})"
        ))
        .bind(margin_secs)
        .fetch_all(&mut *tx)
        .await
        .unwrap();
        let plan_old: Vec<String> = sqlx::query_scalar(&format!(
            "EXPLAIN SELECT last_seen_at FROM {table} WHERE NOT ({old_clause})"
        ))
        .bind(margin_secs)
        .fetch_all(&mut *tx)
        .await
        .unwrap();
        let plan_new_text = plan_new.join("\n");
        let plan_old_text = plan_old.join("\n");
        assert!(
            plan_new_text.contains("Index"),
            "the rewritten clause must be an Index (Only) Scan: {plan_new_text}"
        );
        assert!(
            plan_old_text.contains("Seq Scan"),
            "the OLD cast form must still force a Seq Scan (else this fixture no longer \
             demonstrates the fix): {plan_old_text}"
        );

        let mut live_new: Vec<String> = sqlx::query_scalar(&format!(
            "SELECT last_seen_at FROM {table} WHERE NOT ({new_clause}) ORDER BY last_seen_at"
        ))
        .bind(margin_secs)
        .fetch_all(&mut *tx)
        .await
        .unwrap();
        let mut live_old: Vec<String> = sqlx::query_scalar(&format!(
            "SELECT last_seen_at FROM {table} WHERE NOT ({old_clause}) ORDER BY last_seen_at"
        ))
        .bind(margin_secs)
        .fetch_all(&mut *tx)
        .await
        .unwrap();
        live_new.sort();
        live_old.sort();
        assert_eq!(
            live_new, live_old,
            "the rewritten clause must return the IDENTICAL row set as the cast form, \
             including the row planted exactly at the margin boundary"
        );
        assert!(
            live_new.iter().any(|s| s == &stamps[stamps.len() - 1]),
            "the exact-boundary row must be counted as live (not stale) on both forms"
        );
        tx.commit().await.unwrap();
    }

    #[test]
    fn stale_before_clause_sqlite_binds_the_app_clock_cutoff() {
        let mut params = Vec::new();
        let clause = stale_before_clause(
            CanonicalStampColumn::JobsUpdatedAt,
            None,
            BackendKind::Sqlite,
            Duration::from_secs(60),
            &mut params,
        );
        assert_eq!(clause, "updated_at < $1");
        assert_eq!(params.len(), 1);
        match &params[0] {
            SqlValue::TextOwned(s) => {
                assert_eq!(s.len(), canonical_stamp_now().len(), "fixed-width form")
            }
            other => panic!("expected a TextOwned cutoff bind, got {other:?}"),
        }
    }
}
