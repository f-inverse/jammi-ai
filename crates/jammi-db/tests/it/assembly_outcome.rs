//! The assembly cooldown/counter (migration 037):
//! `Catalog::claim_next`'s cooldown term in its CANDIDATE subselect,
//! `Catalog::record_assembly_outcome`'s exhaustive [`AssemblyOutcome`] rule,
//! and `Catalog::materialize_or_reuse_training_set` (the coordinator's
//! call-site wrapper of `Catalog::fill_training_set_identity`).
//!
//! Every behavioural test is parameterised over [`BackendKind`] via
//! `test_case` + `cfg_attr`, matching `migrations.rs`/`jobs_queue.rs`'s own
//! shape. `cooldown_sql_has_no_second_clock_source` is a hermetic source
//! scan, no backend needed.

use std::sync::Arc;
use std::time::Duration;

use crate::common::queue_session;
use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::{AssemblyOutcome, SubmitJobParams, TrainingSetAssembly};
use jammi_db::catalog::lease::{lease_remaining_seconds_expr, pg_canonical_stamp};
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;
use tempfile::tempdir;
use test_case::test_case;

const KIND: &str = "fine_tune";
const KINDS: &[&str] = &[KIND];

fn job_params(job_id: &str) -> SubmitJobParams<'_> {
    SubmitJobParams {
        job_id,
        kind: KIND,
        execution: JobExecution::Queued,
        spec: "{}",
        model_ref: Some("q-base::1"),
        output_model_id: None,
        model_source: None,
        priority: 0,
    }
}

/// Read `(assembly_failures, next_assembly_after)` back via raw SQL -- these
/// columns are not on `JobRecord`/`SELECT_COLS` (no production reader needs
/// them as typed fields; `claim_next`'s own cooldown term reads them purely
/// in SQL).
async fn read_assembly_state(catalog: &Catalog, job_id: &str) -> (i32, Option<String>) {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT assembly_failures, next_assembly_after FROM jobs \
                         WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                        |row| {
                            Ok((
                                row.get::<i32>("assembly_failures")?,
                                row.try_get::<String>("next_assembly_after")?,
                            ))
                        },
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the row must exist")
}

/// The remaining seconds on `next_assembly_after`, via
/// `lease_remaining_seconds_expr` (the SAME expression
/// `get_job_for_rank` uses for `lease_expires_at`) -- `None` when the
/// column is `NULL`, comparable across both dialects.
async fn remaining_cooldown_secs(
    catalog: &Catalog,
    job_id: &str,
    kind: BackendKind,
) -> Option<f64> {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let mut params: Vec<SqlValue<'static>> = Vec::new();
                    let expr =
                        lease_remaining_seconds_expr("next_assembly_after", kind, &mut params);
                    params.push(SqlValue::TextOwned(job_id));
                    let job_bind = params.len();
                    let sql =
                        format!("SELECT {expr} AS remaining FROM jobs WHERE job_id = ${job_bind}");
                    tx.query_opt(&sql, &params, |row| row.try_get::<f64>("remaining"))
                        .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the row must exist")
}

/// Set `next_assembly_after` to `offset_secs` from the BACKEND's own clock,
/// via raw SQL -- Postgres: `now() + make_interval(...)`, entirely computed
/// by the server; SQLite: the application clock (the "backend" clock for a
/// single embedded process, `catalog::lease`'s own module docs), the SAME
/// source `lease_deadline`/`canonical_stamp_now` use. Never a value this test reads
/// back and re-asserts equality against -- only ever a relative offset, so
/// this fixture cannot itself fabricate the property under test.
async fn set_next_assembly_after_offset(
    catalog: &Catalog,
    job_id: &str,
    kind: BackendKind,
    offset_secs: i64,
) {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                match kind {
                    BackendKind::Postgres => {
                        let sql = format!(
                            "UPDATE jobs SET next_assembly_after = {} \
                             WHERE job_id = $2",
                            pg_canonical_stamp("now() + make_interval(secs => $1)")
                        );
                        tx.execute(
                            &sql,
                            &[
                                SqlValue::Float(offset_secs as f64),
                                SqlValue::TextOwned(job_id),
                            ],
                        )
                        .await
                    }
                    BackendKind::Sqlite => {
                        let ts = (chrono::Utc::now() + chrono::Duration::seconds(offset_secs))
                            .format("%Y-%m-%dT%H:%M:%S%.6fZ")
                            .to_string();
                        tx.execute(
                            "UPDATE jobs SET next_assembly_after = $1 WHERE job_id = $2",
                            &[SqlValue::TextOwned(ts), SqlValue::TextOwned(job_id)],
                        )
                        .await
                    }
                }
            })
        })
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// (a) a higher-priority job inside its cooldown does not block a
// lower-priority ready job.
// ---------------------------------------------------------------------------

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cooldown_job_never_blocks_a_lower_priority_ready_job(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog
        .submit_job(job_params("cooling-high"))
        .await
        .unwrap();
    catalog.submit_job(job_params("ready-low")).await.unwrap();
    // The higher-priority job is inside a one-hour cooldown; the
    // lower-priority job has none (`next_assembly_after` defaults NULL).
    set_next_assembly_after_offset(&catalog, "cooling-high", backend, 3600).await;
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET priority = 10 WHERE job_id = 'cooling-high'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let claimed = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a claimable job exists");
    assert_eq!(
        claimed.job_id, "ready-low",
        "the higher-priority job must be skipped while cooling down, even though \
         `ORDER BY priority DESC` would otherwise pick it first"
    );

    let none = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap();
    assert!(
        none.is_none(),
        "the cooling-down job must stay invisible to the claim while its cooldown \
         has not passed"
    );
}

// ---------------------------------------------------------------------------
// (b) no second clock source; the Postgres-only backend-clock oracle.
// ---------------------------------------------------------------------------

/// A source scan, hermetic: `jobs_repo.rs` must render every clock-bearing
/// SQL fragment through `catalog::lease`'s helpers
/// (`lease_expired_clause`/`lease_deadline_expr`), never a hand-written
/// second clock source. Fails against a mutation that inlines
/// `CURRENT_TIMESTAMP`, `datetime('now'`, or `chrono::Utc::now()` directly
/// into the cooldown SQL instead of delegating.
#[test]
fn cooldown_sql_has_no_second_clock_source() {
    let src = include_str!("../../src/catalog/jobs_repo.rs");
    for forbidden in ["CURRENT_TIMESTAMP", "datetime('now'", "chrono::Utc::now()"] {
        assert!(
            !src.contains(forbidden),
            "jobs_repo.rs must never hand-write a clock-bearing SQL fragment -- every \
             one goes through catalog::lease's helpers; found the literal {forbidden:?}"
        );
    }
}

/// The cooldown predicate is governed ENTIRELY by the Postgres SERVER's own
/// clock: `next_assembly_after` is written and compared using ONLY
/// server-side `now()` arithmetic (`make_interval`) -- this test never reads
/// `chrono::Utc::now()` (or any other process clock) to compute the values
/// it asserts against, so nothing this PROCESS's clock reads could change
/// the outcome even if it were skewed against the database's.
#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn postgres_cooldown_is_governed_by_the_server_clock_alone() {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(BackendKind::Postgres, dir.path()).await;

    catalog
        .submit_job(job_params("pg-cooled-past"))
        .await
        .unwrap();
    catalog
        .submit_job(job_params("pg-cooled-future"))
        .await
        .unwrap();
    // Already past, from the SERVER's own clock -- admits at once.
    set_next_assembly_after_offset(&catalog, "pg-cooled-past", BackendKind::Postgres, -1).await;
    // An hour ahead, from the SAME server clock -- stays invisible.
    set_next_assembly_after_offset(&catalog, "pg-cooled-future", BackendKind::Postgres, 3600).await;

    let first = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the already-passed cooldown must admit the row");
    assert_eq!(first.job_id, "pg-cooled-past");

    let none = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap();
    assert!(
        none.is_none(),
        "the hour-ahead cooldown must still hold the second row back"
    );
}

// ---------------------------------------------------------------------------
// (c) the assembly outcome table: one row per reason, each rule fires once,
// exhaustive over every variant.
// ---------------------------------------------------------------------------

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembly_outcome_table_is_exhaustive_and_each_rule_fires_once(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    // (outcome, tag, counted?, cooled?)
    let table: &[(AssemblyOutcome, &str, bool, bool)] = &[
        (AssemblyOutcome::Refuted, "refuted", true, true),
        (
            AssemblyOutcome::AllRootDivergent,
            "all-root-divergent",
            true,
            true,
        ),
        (AssemblyOutcome::Unavailable, "unavailable", false, true),
        (
            AssemblyOutcome::StoreUnavailable,
            "store-unavailable",
            false,
            true,
        ),
        (AssemblyOutcome::ShortListed, "short-listed", false, true),
        (AssemblyOutcome::NoBody, "no-body", false, false),
        (AssemblyOutcome::Drain, "drain", false, false),
        (AssemblyOutcome::Cancelled, "cancelled", false, false),
    ];

    for (outcome, tag, counted, cooled) in table {
        let job_id = format!("assembly-{tag}");
        catalog.submit_job(job_params(&job_id)).await.unwrap();
        let claimed = catalog
            .claim_next(&format!("w-{tag}"), KINDS, Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("the job is queued");
        assert_eq!(claimed.job_id, job_id);
        assert_eq!(claimed.attempts, 1);

        let (before_failures, before_cooldown) = read_assembly_state(&catalog, &job_id).await;
        assert_eq!(before_failures, 0, "{tag}: starts uncounted");
        assert_eq!(before_cooldown, None, "{tag}: starts with no cooldown");

        let wrote = catalog
            .record_assembly_outcome(&job_id, claimed.attempts, *outcome)
            .await
            .unwrap();
        assert!(wrote, "{tag}: the guarded row must still match");

        let (after_failures, after_cooldown) = read_assembly_state(&catalog, &job_id).await;
        assert_eq!(
            after_failures,
            if *counted { 1 } else { 0 },
            "{tag}: counted={counted}"
        );
        assert_eq!(
            after_cooldown.is_some(),
            *cooled,
            "{tag}: cooled={cooled}, got {after_cooldown:?}"
        );
    }
}

/// Consecutive COUNTED failures escalate the backoff (bounded exponential,
/// `assembly_backoff`'s own ceiling) -- the second refusal's cooldown is
/// stamped strictly later than the first's, read back via
/// `lease_remaining_seconds_expr` so the comparison holds on both dialects.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consecutive_counted_failures_escalate_the_backoff(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog.submit_job(job_params("escalate")).await.unwrap();
    let claimed = catalog
        .claim_next("w", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();

    catalog
        .record_assembly_outcome(&claimed.job_id, claimed.attempts, AssemblyOutcome::Refuted)
        .await
        .unwrap();
    let (failures_1, _) = read_assembly_state(&catalog, "escalate").await;
    let remaining_1 = remaining_cooldown_secs(&catalog, "escalate", backend)
        .await
        .expect("the first refusal must cool down");
    assert_eq!(failures_1, 1);

    catalog
        .record_assembly_outcome(&claimed.job_id, claimed.attempts, AssemblyOutcome::Refuted)
        .await
        .unwrap();
    let (failures_2, _) = read_assembly_state(&catalog, "escalate").await;
    let remaining_2 = remaining_cooldown_secs(&catalog, "escalate", backend)
        .await
        .expect("the second refusal must cool down");
    assert_eq!(failures_2, 2);

    assert!(
        remaining_2 > remaining_1,
        "the second counted failure's backoff ({remaining_2}s remaining) must exceed the \
         first's ({remaining_1}s remaining) -- bounded exponential on the counter"
    );
}

/// A success resets BOTH columns -- even after a counted failure has bumped
/// the counter and armed a cooldown.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn success_resets_the_counter_and_the_cooldown(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog.submit_job(job_params("resets")).await.unwrap();
    let claimed = catalog
        .claim_next("w", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();
    catalog
        .record_assembly_outcome(&claimed.job_id, claimed.attempts, AssemblyOutcome::Refuted)
        .await
        .unwrap();
    let (failures_before, cooldown_before) = read_assembly_state(&catalog, "resets").await;
    assert_eq!(failures_before, 1);
    assert!(cooldown_before.is_some());

    catalog
        .record_assembly_outcome(&claimed.job_id, claimed.attempts, AssemblyOutcome::Success)
        .await
        .unwrap();
    let (failures_after, cooldown_after) = read_assembly_state(&catalog, "resets").await;
    assert_eq!(failures_after, 0, "success must reset the counter to 0");
    assert_eq!(cooldown_after, None, "success must clear the cooldown");
}

/// A moved claim (`attempt` no longer names the row's current attempt)
/// aborts `record_assembly_outcome` with NO write.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_assembly_outcome_moved_claim_aborts_without_a_write(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog.submit_job(job_params("moved")).await.unwrap();
    let claimed = catalog
        .claim_next("w", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(claimed.attempts, 1);

    let wrote = catalog
        .record_assembly_outcome(
            &claimed.job_id,
            claimed.attempts + 1, // the stale attempt number
            AssemblyOutcome::Refuted,
        )
        .await
        .unwrap();
    assert!(!wrote, "a stale attempt number must write nothing at all");

    let (failures, cooldown) = read_assembly_state(&catalog, "moved").await;
    assert_eq!(
        failures, 0,
        "a moved claim must leave the counter untouched"
    );
    assert_eq!(
        cooldown, None,
        "a moved claim must leave the cooldown untouched"
    );
}

// ---------------------------------------------------------------------------
// (f) `Catalog::materialize_or_reuse_training_set` -- the coordinator's own
// call-site wrapper of `fill_training_set_identity`.
// ---------------------------------------------------------------------------

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialize_or_reuse_training_set_first_call_wins(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog
        .submit_job(job_params("assemble-win"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();

    let outcome = catalog
        .materialize_or_reuse_training_set("assemble-win", "coord-1", 1, "digest-a", "table-a")
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetAssembly::Won);

    let row = catalog.get_job("assemble-win").await.unwrap();
    assert_eq!(row.training_set_ref.as_deref(), Some("digest-a"));
    assert_eq!(row.training_set_location.as_deref(), Some("table-a"));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialize_or_reuse_training_set_second_call_same_values_reuses(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog
        .submit_job(job_params("assemble-reuse"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();

    let first = catalog
        .materialize_or_reuse_training_set("assemble-reuse", "coord-1", 1, "digest-b", "table-b")
        .await
        .unwrap();
    assert_eq!(first, TrainingSetAssembly::Won);

    let second = catalog
        .materialize_or_reuse_training_set("assemble-reuse", "coord-1", 1, "digest-b", "table-b")
        .await
        .unwrap();
    assert_eq!(
        second,
        TrainingSetAssembly::Reused,
        "a repeat call under the same claim, same values, must REUSE, never overwrite"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialize_or_reuse_training_set_concurrent_racer_reuses_never_overwrites(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog
        .submit_job(job_params("assemble-race"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let h1 = tokio::spawn(async move {
        c1.materialize_or_reuse_training_set("assemble-race", "coord-1", 1, "digest-r", "table-r")
            .await
            .unwrap()
    });
    let h2 = tokio::spawn(async move {
        c2.materialize_or_reuse_training_set("assemble-race", "coord-1", 1, "digest-r", "table-r")
            .await
            .unwrap()
    });
    let outcomes = [h1.await.unwrap(), h2.await.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == TrainingSetAssembly::Won)
            .count(),
        1,
        "exactly one racer must win, got {outcomes:?}"
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == TrainingSetAssembly::Reused)
            .count(),
        1,
        "the loser must REUSE, never error or silently overwrite, got {outcomes:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialize_or_reuse_training_set_moved_claim_aborts_without_a_write(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;

    catalog
        .submit_job(job_params("assemble-moved"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();

    // A second claim moves the attempt on: the stale caller still believes
    // attempts == 1.
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET claimed_by = 'coord-2', attempts = 2 \
                     WHERE job_id = 'assemble-moved'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let outcome = catalog
        .materialize_or_reuse_training_set("assemble-moved", "coord-1", 1, "digest-z", "table-z")
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetAssembly::Moved);

    let row = catalog.get_job("assemble-moved").await.unwrap();
    assert_eq!(
        row.training_set_ref, None,
        "a MOVED claim must leave the pair untouched -- no write at all"
    );
    assert_eq!(row.training_set_location, None);
}
