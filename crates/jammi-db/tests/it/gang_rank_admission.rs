//! `Catalog::get_job_for_rank` (I-GANG's row predicate) and
//! `Catalog::fill_training_set_identity` (the training-set identity
//! write-once CAS). `Catalog::fresh_instance`'s own
//! tests live in `gang_instance_freshness.rs`. Every test not itself
//! exercising a Postgres-only expression runs on a fresh SQLite catalog (a
//! private tempdir per test); `world_size` decoding (a targeted JSON field
//! read with no backend-specific SQL of its own, but still worth the same
//! `test_case`-parameterized sqlite/postgres shape `migrations.rs` uses) and
//! `lease` also run a `::postgres` arm gated by `live-postgres-tests`,
//! skipping (never failing) when `JAMMI_TEST_PG_URL` is unset. Since
//! migration `039_canonical_stamps`, `jammi_db::catalog::lease::
//! decode_lease_expires_at` parses the IDENTICAL text shape on either
//! backend (the schema-edge domain makes it the only representable one) --
//! issue #574's parity oracle
//! (`get_job_for_rank_undecodable_lease_is_a_row_fact_on_sqlite_and_a_write_refusal_on_postgres`)
//! is the one test in this file that MUST run on both backends whenever
//! Postgres is available; it now pins a stated ASYMMETRY rather than an
//! identity -- a shape-valid/calendar-invalid `lease_expires_at` is a row
//! FACT (`Ok(Some(row))` with `LeaseFact::Undecodable`) on SQLite (whose
//! trigger checks shape only) and a typed WRITE REFUSAL on Postgres (whose
//! `CHECK` also validates the cast) -- see that test's own docs for why.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::catalog::jobs_repo::{SubmitJobParams, TrainingSetFillOutcome, WorldSizeFact};
use jammi_db::catalog::lease::LeaseFact;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

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

async fn base_catalog() -> (tempfile::TempDir, Arc<Catalog>) {
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path())
        .await
        .expect("sqlite session always available");
    let catalog = Arc::clone(session.catalog());
    catalog
        .register_model(RegisterModelParams {
            model_id: "q-base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    (dir, catalog)
}

/// Clear every row from `jobs`/`instances`/`workers` so the global claim
/// scan in [`Catalog::claim_next`] sees only the rows a test creates itself
/// — the same shape as `jobs_queue.rs`'s own `reset_queue`. Needed because
/// the Postgres lane shares ONE live database across the whole run
/// (`jammi_test_utils::make_test_session`'s own docs); the SQLite lane gets
/// a fresh tempdir per test regardless, so running the reset there too keeps
/// one path for both backends. CI runs the Postgres lane under
/// `--test-threads=1`, so the reset-then-populate sequence here is
/// serialised and cannot race a sibling test.
async fn reset_queue(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("DELETE FROM jobs", &[]).await?;
                tx.execute("DELETE FROM workers", &[]).await?;
                tx.execute("DELETE FROM instances", &[]).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

/// Parameterized counterpart of [`base_catalog`] for the `world_size`
/// decoding oracles, which also run a `::postgres` arm. Returns `None`
/// (never a panic) when `kind = Postgres` and `JAMMI_TEST_PG_URL` is unset,
/// so callers skip exactly like `migrations.rs`'s own parameterized tests.
async fn base_catalog_kind(kind: BackendKind) -> Option<(tempfile::TempDir, Arc<Catalog>)> {
    let dir = tempdir().unwrap();
    let session = make_test_session(kind, dir.path()).await?;
    let catalog = Arc::clone(session.catalog());
    reset_queue(&catalog).await;
    catalog
        .register_model(RegisterModelParams {
            model_id: "q-base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    Some((dir, catalog))
}

// ---------------------------------------------------------------------------
// `Catalog::get_job_for_rank`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_job_for_rank_returns_none_for_an_absent_job() {
    let (_dir, catalog) = base_catalog().await;
    let row = catalog.get_job_for_rank("no-such-job").await.unwrap();
    assert!(
        row.is_none(),
        "an absent job must resolve to None, got {row:?}"
    );
}

/// The row's `status`/`claimed_by`/`attempts`/`lease` mirror a genuine
/// claim exactly, and `lease` is `LeaseFact::Live` for a freshly-claimed
/// lease. Parameterized (sqlite/postgres, the `migrations.rs` shape) so the
/// property is pinned against a real Postgres write, not just SQLite's; the
/// postgres arm skips (never fails) when `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_reflects_a_live_claim(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog.submit_job(job_params("job-1")).await.unwrap();
    let claimed = catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("must claim the only queued job");
    assert_eq!(claimed.attempts, 1);

    let row = catalog
        .get_job_for_rank("job-1")
        .await
        .unwrap()
        .expect("must find the claimed job by primary key");
    assert_eq!(row.status, "running");
    assert_eq!(row.claimed_by.as_deref(), Some("coord-1"));
    assert_eq!(row.attempts, 1);
    assert!(
        row.lease.is_live(),
        "a freshly-claimed 30s lease must be live, got {:?}",
        row.lease
    );
    let remaining = row.lease.remaining();
    assert!(
        remaining > Duration::from_secs(20) && remaining <= Duration::from_secs(31),
        "remaining must be close to the freshly-stamped 30s window (the deadline stamp and the read's own clock are two application-clock reads, so the window may overshoot by the documented sub-millisecond rounding), got {remaining:?}"
    );
}

/// A lease forced to `NULL` (the state both reclaim arms leave behind, and
/// migration invariant `lease.rs` documents: `NULL` means remaining 0, never
/// live-by-default) reads back `lease == LeaseFact::Dead` — never a panic,
/// never a "live" default. Parameterized (sqlite/postgres): the postgres arm
/// skips (never fails) when `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_treats_a_null_lease_as_not_live(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog.submit_job(job_params("job-2")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET lease_expires_at = NULL WHERE job_id = 'job-2'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let row = catalog.get_job_for_rank("job-2").await.unwrap().unwrap();
    assert_eq!(
        row.lease,
        LeaseFact::Dead,
        "a NULL lease column must never read as live"
    );
}

/// An expired (past) lease reads back `lease == LeaseFact::Dead` — the
/// boundary [`jammi_db::catalog::lease::decode_lease_expires_at`]'s
/// `deadline >= now` test names. Parameterized (sqlite/postgres): the
/// postgres arm skips (never fails) when `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_treats_an_expired_lease_as_not_live(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog.submit_job(job_params("job-3")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_millis(1))
        .await
        .unwrap()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let row = catalog.get_job_for_rank("job-3").await.unwrap().unwrap();
    assert_eq!(
        row.lease,
        LeaseFact::Dead,
        "an expired lease must never read as live"
    );
}

/// #574's own parity oracle — REWRITTEN by `catalog::lease`'s S4/migration
/// `039_canonical_stamps`: before 039, a `lease_expires_at` that failed to
/// parse read as `Ok(Some(row))` with `lease_live = false` on SQLite
/// (`julianday(...)` silently returns `NULL` for unparseable text) but `Err`
/// on Postgres (`col::timestamptz` raised a genuine SQL error) — the
/// backend-dependent classification #574 reported. That specific asymmetry
/// is CLOSED: the schema-edge domain (`catalog::lease`'s S3) makes every
/// value a live Postgres write can ever leave in `lease_expires_at`
/// cast-valid, so the read side can no longer fault on row content there.
///
/// What remains, stated rather than papered over: SQLite's trigger checks
/// SHAPE only (a `GLOB` over a digit-class pattern — SQLite has no calendar
/// parser), so a shape-valid, CALENDAR-invalid value (a month of `13`) is
/// still representable on SQLite and still decodes as
/// [`LeaseFact::Undecodable`] there (nothing in this crate's own write path
/// ever produces one; this plants it with raw SQL). On Postgres the SAME
/// literal text is REFUSED at the write itself — the CHECK requires BOTH
/// shape AND `c::timestamptz IS NOT NULL`, and Postgres's own calendar
/// validation rejects a month of `13` outright (SQLSTATE `22008`) — so the
/// value this test plants on SQLite can never reach a Postgres row at all;
/// `get_job_for_rank` never gets the chance to disagree about it. Every
/// OTHER column stays populated on the SQLite arm, proving the malformed
/// lease is isolated to `RankAdmissionRow::lease` alone. Parameterized
/// (sqlite/postgres): the postgres arm skips (never fails) when
/// `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_undecodable_lease_is_a_row_fact_on_sqlite_and_a_write_refusal_on_postgres(
    kind: BackendKind,
) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    let job_id = format!(
        "job-lease-undecodable-{}",
        jammi_test_utils::unique_suffix()
    );
    catalog.submit_job(job_params(&job_id)).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    // Shape-valid (matches the schema-edge CHECK/trigger's digit-class
    // regex/GLOB), calendar-invalid (no month `13`) — see this test's docs
    // for why a leap second does NOT serve this role (chrono accepts it).
    let sql = format!(
        "UPDATE jobs SET lease_expires_at = '2026-13-01T00:00:00.000000Z' WHERE job_id = '{job_id}'"
    );
    let write_result = catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let sql = sql.clone();
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await;

    match kind {
        BackendKind::Sqlite => {
            write_result.expect(
                "SQLite's trigger checks shape only; a calendar-invalid but \
                 shape-valid value is admitted",
            );
            let row = catalog
                .get_job_for_rank(&job_id)
                .await
                .expect("a malformed lease_expires_at must be a row fact, never a read fault")
                .expect("row present");
            assert_eq!(
                row.lease,
                LeaseFact::Undecodable,
                "a lease that does not parse as a timestamp must be Undecodable"
            );
            assert_eq!(row.status, "running", "every other column stays populated");
            assert_eq!(row.claimed_by.as_deref(), Some("coord-1"));
            assert_eq!(row.attempts, 1);
        }
        BackendKind::Postgres => {
            let err = write_result.expect_err(
                "Postgres's CHECK requires calendar validity too; a month of 13 must be \
                 refused at the write, never silently stored for a later Undecodable read",
            );
            assert!(
                matches!(
                    err,
                    jammi_db::catalog::backend::BackendError::DomainViolation { .. }
                ),
                "the refusal must be the typed domain-violation class, got {err:?}"
            );
        }
    }
}

/// The row's OWN `world_size`, decoded from `spec`
/// JSON, never the caller's `Assign.world`. A spec whose `common.world_size`
/// names 2 must read back 2 through `get_job_for_rank`.
#[tokio::test]
async fn get_job_for_rank_reflects_the_row_world_size() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(SubmitJobParams {
            job_id: "job-world-2",
            kind: KIND,
            execution: JobExecution::Queued,
            spec: r#"{"common":{"world_size":2}}"#,
            model_ref: Some("q-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let row = catalog
        .get_job_for_rank("job-world-2")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.world_size,
        WorldSizeFact::Decoded(2),
        "the row's own world_size must round-trip"
    );
}

/// N4 oracle (iv) (#548): `world_size_from_spec_json` (this module,
/// module-private) reads `jammi_ai::jobs::JobSpec`'s new flat-derive shape
/// unchanged, for every training kind and for a compute kind — `jammi-db`
/// cannot import `jammi-ai` (the dependency runs the other way), so these
/// fixtures are hand-written JSON matching that type's own byte-pin tests
/// (`crates/jammi-ai/src/jobs.rs`'s `job_spec_byte_pins_every_compiled_kind_
/// against_its_own_type`/`job_spec_fine_tune_literal_byte_pin`) rather than
/// constructed through it. `fine_tune`/`graph_fine_tune` carry `common.
/// world_size` and decode it explicitly; `context_predictor` has no
/// `common` field at all (`TrainingSpec::ContextPredictor` never did, tag
/// merge or not) and `neighbor_graph` (a compute kind) has no `world_size`
/// field either — both default to `1` through the SAME absent-field path,
/// which is the property this test pins: a row's kind never changes how
/// `world_size` is read, only whether the field exists. Not parameterized
/// (sqlite only): the read is a plain JSON field lookup with no
/// backend-specific SQL, already covered on both backends by the
/// `world_size`-decoding tests around it.
#[tokio::test]
async fn get_job_for_rank_reads_world_size_the_same_way_for_every_compiled_job_spec_kind_shape() {
    let (_dir, catalog) = base_catalog().await;
    let fixtures: &[(&str, &str, WorldSizeFact)] = &[
        (
            "job-shape-fine-tune",
            r#"{"kind":"fine_tune","source":"s","columns":["text"],"method":"lora","task":"text_embedding","common":{"base_model":"b","world_size":2},"cache":"bypass"}"#,
            WorldSizeFact::Decoded(2),
        ),
        (
            "job-shape-graph-fine-tune",
            r#"{"kind":"graph_fine_tune","sources":{},"sample_config":{},"common":{"base_model":"b","world_size":3}}"#,
            WorldSizeFact::Decoded(3),
        ),
        (
            "job-shape-context-predictor",
            r#"{"kind":"context_predictor","source":"s","predictor_spec":{}}"#,
            WorldSizeFact::Decoded(1),
        ),
        (
            "job-shape-neighbor-graph",
            r#"{"kind":"neighbor_graph","source_id":"s","embedding_table":null,"params":{},"cache":"bypass"}"#,
            WorldSizeFact::Decoded(1),
        ),
    ];
    for (job_id, spec, expected) in fixtures {
        catalog
            .submit_job(SubmitJobParams {
                job_id,
                kind: KIND,
                execution: JobExecution::Queued,
                spec,
                model_ref: Some("q-base::1"),
                output_model_id: None,
                model_source: None,
                priority: 0,
            })
            .await
            .unwrap();
        let row = catalog
            .get_job_for_rank(job_id)
            .await
            .unwrap()
            .expect("the row was just submitted");
        assert_eq!(
            row.world_size, *expected,
            "job {job_id} (spec {spec}) must read world_size as {expected:?}"
        );
    }
}

/// A spec naming no `world_size` at all (`job_params`'s `"{}"`, every
/// existing fixture in this file) reads back `1` — the single-rank default,
/// never an error and never left undefined. Parameterized (sqlite/postgres,
/// the `migrations.rs` shape): the postgres arm skips (never fails) when
/// `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_defaults_world_size_when_absent_from_spec(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog
        .submit_job(job_params("job-world-absent"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let row = catalog
        .get_job_for_rank("job-world-absent")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.world_size,
        WorldSizeFact::Decoded(1),
        "a spec naming no world_size must default to 1, never fail and never default to 0"
    );
}

/// A `world_size` field present but not a valid rank count (a string, here)
/// is a ROW FACT (`WorldSizeFact::Undecodable`), never a fault: the read
/// still succeeds (`Ok(Some(row))`) and every OTHER column round-trips
/// exactly as it would for a decodable row. Parameterized (sqlite/postgres);
/// the postgres arm skips (never fails) when `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_malformed_world_size_is_undecodable_not_a_fault(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog
        .submit_job(SubmitJobParams {
            job_id: "job-world-malformed",
            kind: KIND,
            execution: JobExecution::Queued,
            spec: r#"{"common":{"world_size":"two"}}"#,
            model_ref: Some("q-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let row = catalog
        .get_job_for_rank("job-world-malformed")
        .await
        .expect("a non-numeric world_size must be a row fact, never an Err")
        .expect("the row exists");
    assert_eq!(
        row.world_size,
        WorldSizeFact::Undecodable,
        "a non-numeric world_size must decode to Undecodable, never a silent 1"
    );
    assert_eq!(row.status, "running", "every other column stays populated");
    assert_eq!(row.claimed_by.as_deref(), Some("coord-1"));
    assert_eq!(row.attempts, 1);
}

/// Spec text that is not valid JSON at all (so it is not even representable
/// as any JSON object) is the SAME row fact as a malformed field —
/// `Undecodable`, never a fault — with every other column still populated.
/// Parameterized (sqlite/postgres); the postgres arm skips (never fails)
/// when `JAMMI_TEST_PG_URL` is unset.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_spec_not_json_is_undecodable_not_a_fault(kind: BackendKind) {
    // The require-gate itself: a direct, crate-qualified call to the
    // registered `shared:` helper (`ci/kernel-oracle-helpers.txt`), textually
    // in THIS test fn's own body — `base_catalog_kind`'s internal `?` on
    // `make_test_session` is one function away and does not dominate this
    // skip for the KO-7 scanner, which is per-`#[test]`-fn textual, not
    // whole-file (`migrations.rs`'s own parameterized tests use this exact
    // shape).
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    catalog
        .submit_job(SubmitJobParams {
            job_id: "job-world-not-json",
            kind: KIND,
            execution: JobExecution::Queued,
            spec: "not-even-json",
            model_ref: Some("q-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let row = catalog
        .get_job_for_rank("job-world-not-json")
        .await
        .expect("spec text that fails to parse must be a row fact, never an Err")
        .expect("the row exists");
    assert_eq!(row.world_size, WorldSizeFact::Undecodable);
    assert_eq!(row.status, "running", "every other column stays populated");
    assert_eq!(row.claimed_by.as_deref(), Some("coord-1"));
    assert_eq!(row.attempts, 1);
}

/// A genuine driver-level fault (the table this read depends on is gone) is
/// still `Err` — distinguishing "the row's content did not decode" from
/// "the read itself faulted" is the whole point of `WorldSizeFact`.
#[tokio::test]
async fn get_job_for_rank_driver_fault_still_surfaces_as_err() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(job_params("job-driver-fault"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move { tx.execute("DROP TABLE jobs", &[]).await })
        })
        .await
        .unwrap();

    catalog
        .get_job_for_rank("job-driver-fault")
        .await
        .expect_err("a dropped table must still surface as a genuine Err, not a row fact");
}

// ---------------------------------------------------------------------------
// `Catalog::fill_training_set_identity` (the training-set identity write-once CAS)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fill_training_set_identity_first_call_fills() {
    let (_dir, catalog) = base_catalog().await;
    catalog.submit_job(job_params("job-fill-1")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let outcome = catalog
        .fill_training_set_identity("job-fill-1", "coord-1", 1, "digest-a", "table-a")
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetFillOutcome::Filled);

    let row = catalog.get_job("job-fill-1").await.unwrap();
    assert_eq!(row.training_set_ref.as_deref(), Some("digest-a"));
    assert_eq!(row.training_set_location.as_deref(), Some("table-a"));
}

/// A second CAS under the SAME claim, passing the SAME values already
/// landed, is REUSE — never a second write, never an error.
#[tokio::test]
async fn fill_training_set_identity_second_call_same_values_reuses() {
    let (_dir, catalog) = base_catalog().await;
    catalog.submit_job(job_params("job-fill-2")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let first = catalog
        .fill_training_set_identity("job-fill-2", "coord-1", 1, "digest-b", "table-b")
        .await
        .unwrap();
    assert_eq!(first, TrainingSetFillOutcome::Filled);

    let second = catalog
        .fill_training_set_identity("job-fill-2", "coord-1", 1, "digest-b", "table-b")
        .await
        .unwrap();
    assert_eq!(
        second,
        TrainingSetFillOutcome::Reused,
        "a repeat CAS under the same claim, same values, must REUSE, never overwrite"
    );
}

/// This is the a1' concurrency shape: two "coordinators" (in spirit — one
/// process, two calls) race the SAME claim; the SECOND to land sees zero
/// rows updated and, since its own values match what the first wrote,
/// REUSEs rather than erroring or overwriting.
#[tokio::test]
async fn fill_training_set_identity_concurrent_racer_reuses_never_overwrites() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(job_params("job-fill-race"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let h1 = tokio::spawn(async move {
        c1.fill_training_set_identity("job-fill-race", "coord-1", 1, "digest-r", "table-r")
            .await
            .unwrap()
    });
    let h2 = tokio::spawn(async move {
        c2.fill_training_set_identity("job-fill-race", "coord-1", 1, "digest-r", "table-r")
            .await
            .unwrap()
    });
    let (a, b) = (h1.await.unwrap(), h2.await.unwrap());
    let outcomes = [a, b];
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == TrainingSetFillOutcome::Filled)
            .count(),
        1,
        "exactly one racer must win the CAS, got {outcomes:?}"
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == TrainingSetFillOutcome::Reused)
            .count(),
        1,
        "the loser must REUSE (matching values), never error or silently vanish, got {outcomes:?}"
    );

    let row = catalog.get_job("job-fill-race").await.unwrap();
    assert_eq!(row.training_set_ref.as_deref(), Some("digest-r"));
    assert_eq!(row.training_set_location.as_deref(), Some("table-r"));
}

/// A claim that moved (a different `attempts`, the shape a reclaim-then-
/// reclaim leaves) makes the CAS ABORT — no terminal write, the row simply
/// keeps whatever it already had.
#[tokio::test]
async fn fill_training_set_identity_moved_claim_aborts_without_a_terminal_write() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(job_params("job-fill-abort"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();

    // The stale caller still believes attempts == 1, but the row has since
    // moved on (a second claim under a different attempt/claimant) — the
    // shape a requeue-then-reclaim leaves, manufactured directly here.
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET claimed_by = 'coord-2', attempts = 2 \
                     WHERE job_id = 'job-fill-abort'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let outcome = catalog
        .fill_training_set_identity("job-fill-abort", "coord-1", 1, "digest-z", "table-z")
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetFillOutcome::Aborted);

    let row = catalog.get_job("job-fill-abort").await.unwrap();
    assert_eq!(
        row.training_set_ref, None,
        "an ABORTed CAS must leave the pair untouched — no terminal write"
    );
    assert_eq!(row.training_set_location, None);
    assert_eq!(
        row.status, "running",
        "an ABORT must not touch status either — it is a normal event, not a failure"
    );
}

/// A claim under the right `claimed_by`/`attempts`, but where the row
/// already carries a DIFFERENT pair (a hypothetical this program's own CAS
/// predicate cannot construct via its own writes, since the predicate always
/// requires the pair NULL first — manufactured directly to pin the "or a
/// different pair value" arm of ABORT independently of "the claim moved").
#[tokio::test]
async fn fill_training_set_identity_matching_claim_different_pair_aborts() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(job_params("job-fill-different-pair"))
        .await
        .unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET training_set_ref = 'digest-other', \
                         training_set_location = 'table-other' \
                     WHERE job_id = 'job-fill-different-pair'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    let outcome = catalog
        .fill_training_set_identity(
            "job-fill-different-pair",
            "coord-1",
            1,
            "digest-mine",
            "table-mine",
        )
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetFillOutcome::Aborted);
}

/// The stop rule as a property test: a raw single-column write is refused by
/// the schema `CHECK` constraint (migration 033) — never by mere convention.
/// `Catalog::fill_training_set_identity` is the only writer this crate
/// exposes for the pair; this asserts the SCHEMA itself would refuse any
/// OTHER writer too.
#[tokio::test]
async fn a_raw_single_column_write_is_refused_by_the_schema_check() {
    let (_dir, catalog) = base_catalog().await;
    catalog
        .submit_job(job_params("job-raw-write"))
        .await
        .unwrap();

    let err = catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET training_set_ref = 'lone-digest' \
                     WHERE job_id = 'job-raw-write'",
                    &[],
                )
                .await
            })
        })
        .await;
    assert!(
        err.is_err(),
        "a raw write of training_set_ref alone must be refused by the CHECK constraint"
    );
}

// ---------------------------------------------------------------------------
// The `world_size > 1` conjunct's three row facts on the admission row:
// `tenant_id` (raw text), `training_set_ref`, `training_set_location`.
// ---------------------------------------------------------------------------

/// `get_job_for_rank` carries the row's OWN `tenant_id` as raw text and the
/// filled training-set pair — the three facts the gang handler's
/// `world_size > 1` conjunct reads (#566 R2) — on both backends. Before the
/// CAS the pair reads `None`/`None`; after it, exactly the filled values;
/// and a `tenant_id` value that is not a UUID (manufactured by raw SQL —
/// nothing in this crate writes one) still comes back `Ok(Some(row))` with
/// the text verbatim: the read is infallible on content, the caller decides
/// what an undecodable tenant means. Mutation proof: dropping any of the
/// three columns from the SELECT fails to compile the row mapper; parsing
/// `tenant_id` into a `TenantId` inside the mapper flips the garbage arm
/// from `Ok(Some)` to `Err`.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn get_job_for_rank_carries_the_tenant_text_and_the_filled_pair(kind: BackendKind) {
    use std::str::FromStr;
    if matches!(kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none() {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    }
    let (_dir, catalog) = base_catalog_kind(kind).await.expect(
        "base_catalog_kind only returns None for an unconfigured postgres arm, already skipped above",
    );
    let tenant = jammi_db::TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e77").unwrap();
    let job_id = format!("job-pair-{}", jammi_test_utils::unique_suffix());
    // `submit_job` stamps `tenant_id` from the catalog's binding in force —
    // a catalog pinned to `tenant` writes exactly that tenant, the same
    // way a tenant-bound submission does.
    catalog
        .pinned_to_tenant(Some(tenant))
        .submit_job(job_params(&job_id))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("coord-pair", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("must claim the only queued job");

    let before = catalog
        .get_job_for_rank(&job_id)
        .await
        .unwrap()
        .expect("row present");
    assert_eq!(
        before.tenant_id.as_deref(),
        Some(tenant.to_string().as_str())
    );
    assert_eq!(before.training_set_ref, None);
    assert_eq!(before.training_set_location, None);

    let outcome = catalog
        .fill_training_set_identity(
            &job_id,
            "coord-pair",
            claimed.attempts,
            "sha256:pair-digest",
            "pair_table",
        )
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetFillOutcome::Filled);
    let after = catalog
        .get_job_for_rank(&job_id)
        .await
        .unwrap()
        .expect("row present");
    assert_eq!(
        after.training_set_ref.as_deref(),
        Some("sha256:pair-digest")
    );
    assert_eq!(after.training_set_location.as_deref(), Some("pair_table"));
    assert_eq!(after.tenant_id, before.tenant_id);

    let sql = format!("UPDATE jobs SET tenant_id = 'not-a-uuid' WHERE job_id = '{job_id}'");
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let sql = sql.clone();
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await
        .unwrap();
    let poisoned = catalog
        .get_job_for_rank(&job_id)
        .await
        .expect("an undecodable tenant text is a row fact, never a fault of the read")
        .expect("row present");
    assert_eq!(poisoned.tenant_id.as_deref(), Some("not-a-uuid"));
    assert_eq!(
        poisoned.training_set_location.as_deref(),
        Some("pair_table"),
        "every other column is still populated"
    );
}
