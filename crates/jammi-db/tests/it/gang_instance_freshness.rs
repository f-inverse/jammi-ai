//! `Catalog::fresh_instance`: is a gang
//! coordinator's `instances` row FRESH within
//! `instance_liveness_margin(lease) == 2 * lease`? Decoded in RUST from
//! `last_seen_at`'s raw stored text (`jammi_db::catalog::lease::
//! last_seen_at_is_fresh`)
//! against THIS PROCESS's own clock — never a SQL-side `col::timestamptz`
//! cast — because `last_seen_at` is ALWAYS an application-clock stamp on
//! either backend (`Catalog::upsert_instance` never writes the database
//! clock here). Parameterized (sqlite/postgres, the `migrations.rs` shape):
//! every test here also runs a `::postgres` arm gated by
//! `live-postgres-tests`, skipping (never failing) when `JAMMI_TEST_PG_URL`
//! is unset — kept parameterized even though the decode itself is now
//! backend-independent, so the parity oracle
//! (`fresh_instance_malformed_last_seen_at_is_not_fresh_on_sqlite_and_a_write_refusal_on_postgres`)
//! actually pins the stated asymmetry between the two backends (a row fact
//! on SQLite, a write refusal on Postgres — see that test's own docs), not
//! merely that each compiles.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

/// Returns `None` (never a panic) when `kind = Postgres` and
/// `JAMMI_TEST_PG_URL` is unset, so callers skip exactly like
/// `migrations.rs`'s own parameterized tests. The Postgres lane shares ONE
/// live database across the whole run (`jammi_test_utils::
/// make_test_session`'s own docs); every test in this file upserts its own
/// distinct `instance_id`, so no cross-test reset is needed the way
/// `gang_rank_admission.rs`'s `reset_queue` is for `jobs`.
async fn base_catalog_kind(kind: BackendKind) -> Option<(tempfile::TempDir, Arc<Catalog>)> {
    let dir = tempdir().unwrap();
    let session = make_test_session(kind, dir.path()).await?;
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
    Some((dir, catalog))
}

/// Force `instances.last_seen_at` into the past via raw SQL — the state a
/// dead process leaves behind (nothing heartbeats it any more).
async fn force_stale_instance(catalog: &Catalog, instance_id: &str, ago: Duration) {
    let cutoff = (chrono::Utc::now() - chrono::Duration::from_std(ago).unwrap())
        .format("%Y-%m-%dT%H:%M:%S%.6fZ")
        .to_string();
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET last_seen_at = $1 WHERE instance_id = $2",
                    &[
                        SqlValue::TextOwned(cutoff),
                        SqlValue::TextOwned(instance_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_true_for_a_recently_seen_instance(kind: BackendKind) {
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "inst-fresh",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let fresh = catalog
        .fresh_instance("inst-fresh", Duration::from_secs(30))
        .await
        .unwrap();
    assert!(fresh, "a just-upserted instance must be fresh");
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_false_for_an_absent_instance(kind: BackendKind) {
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
    let fresh = catalog
        .fresh_instance("no-such-instance-gang-freshness", Duration::from_secs(30))
        .await
        .unwrap();
    assert!(!fresh, "an absent instance must never read as fresh");
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_false_for_a_stale_instance(kind: BackendKind) {
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "inst-stale",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let lease = Duration::from_secs(30);
    // instance_liveness_margin(lease) == 2 * lease == 60s; push last_seen_at
    // well past that.
    force_stale_instance(&catalog, "inst-stale", Duration::from_secs(300)).await;
    let fresh = catalog.fresh_instance("inst-stale", lease).await.unwrap();
    assert!(
        !fresh,
        "an instance stale beyond 2x the lease must never read as fresh"
    );
}

/// The boundary just inside the margin still reads fresh — this is not an
/// off-by-one on `>=`/`>`, it is the same `stale_before_clause` every other
/// caller in this crate already relies on; asserted here too so a future
/// change to `fresh_instance`'s own predicate (not just `stale_before_clause`
/// itself) is caught locally.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_true_just_inside_the_liveness_margin(kind: BackendKind) {
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "inst-borderline",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let lease = Duration::from_secs(30);
    force_stale_instance(&catalog, "inst-borderline", Duration::from_secs(50)).await;
    let fresh = catalog
        .fresh_instance("inst-borderline", lease)
        .await
        .unwrap();
    assert!(
        fresh,
        "50s ago is inside the 2x30s=60s margin and must still read as fresh"
    );
}

/// The boundary just OUTSIDE the margin reads stale — the tight twin of
/// [`fresh_instance_true_just_inside_the_liveness_margin`] above. Mutation
/// proof: widen `instance_liveness_margin` (or an equivalent inline
/// `lease.saturating_mul(2)`) by so much as 2 extra seconds and this test
/// dies while the just-inside case stays green, since 61s ago would then
/// fall back inside the (now 62s) margin.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_false_just_outside_the_liveness_margin(kind: BackendKind) {
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
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            "inst-just-outside",
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let lease = Duration::from_secs(30);
    // instance_liveness_margin(lease) == 2 * 30s == 60s; 61s ago is 1s past
    // that margin — the tightest stale case short of the exact boundary.
    force_stale_instance(&catalog, "inst-just-outside", Duration::from_secs(61)).await;
    let fresh = catalog
        .fresh_instance("inst-just-outside", lease)
        .await
        .unwrap();
    assert!(
        !fresh,
        "61s ago is 1s past the 2x30s=60s margin and must read as stale"
    );
}

/// The backend-parity oracle for `last_seen_at`, the same shape as
/// `gang_rank_admission.rs`'s sibling test for `jobs.lease_expires_at`. A
/// SQL-side `::timestamptz` cast of an unparseable `last_seen_at` would read
/// as `Ok(false)` on SQLite (a plain string compare, never erroring) but
/// `Err` on Postgres — a backend-dependent classification. Migration
/// `039_canonical_stamps` rules that out: every value a live Postgres write
/// can leave in this column is cast-valid. What remains: SQLite's trigger
/// checks SHAPE only, so a shape-valid, CALENDAR-invalid value (a month of
/// `13` — a leap second does NOT serve this role, `lease.rs`'s own docs
/// state why) is still representable there and still reads `false` (chrono
/// refuses to parse it, `last_seen_at_is_fresh`'s `None` arm); on Postgres
/// the SAME text is refused at the WRITE itself (the CHECK also validates
/// the cast), so `fresh_instance` never gets the chance to read it.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn fresh_instance_malformed_last_seen_at_is_not_fresh_on_sqlite_and_a_write_refusal_on_postgres(
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
    let instance_id = format!(
        "inst-lease-undecodable-{}",
        jammi_test_utils::unique_suffix()
    );
    catalog
        .upsert_instance(&jammi_db::catalog::instance::InstanceRegistration::new(
            &instance_id,
            Some("label"),
            Some("host"),
            None,
            None,
        ))
        .await
        .unwrap();
    let write_result = catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            let instance_id = instance_id.clone();
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET last_seen_at = '2026-13-01T00:00:00.000000Z' \
                     WHERE instance_id = $1",
                    &[SqlValue::TextOwned(instance_id)],
                )
                .await
            })
        })
        .await;

    match kind {
        BackendKind::Sqlite => {
            write_result.expect(
                "SQLite's trigger checks shape only; a calendar-invalid but shape-valid \
                 value is admitted",
            );
            let fresh = catalog
                .fresh_instance(&instance_id, Duration::from_secs(30))
                .await
                .expect(
                    "a calendar-invalid last_seen_at must be a row fact, never a read fault, \
                     on SQLite",
                );
            assert!(
                !fresh,
                "a last_seen_at that does not parse as a timestamp must never read as fresh"
            );
            catalog
                .backend_arc()
                .transaction(TxOptions::default(), |tx| {
                    let instance_id = instance_id.clone();
                    Box::pin(async move {
                        tx.execute(
                            "DELETE FROM instances WHERE instance_id = $1",
                            &[SqlValue::TextOwned(instance_id)],
                        )
                        .await
                    })
                })
                .await
                .unwrap();
        }
        BackendKind::Postgres => {
            let err = write_result.expect_err(
                "Postgres's CHECK requires calendar validity too; a month of 13 must be \
                 refused at the write",
            );
            assert!(
                matches!(
                    err,
                    jammi_db::catalog::backend::BackendError::DomainViolation { .. }
                ),
                "the refusal must be the typed domain-violation class, got {err:?}"
            );
            // Postgres never took the write; nothing to clean up or assert
            // through `fresh_instance` on this backend.
            catalog
                .backend_arc()
                .transaction(TxOptions::default(), |tx| {
                    let instance_id = instance_id.clone();
                    Box::pin(async move {
                        tx.execute(
                            "DELETE FROM instances WHERE instance_id = $1",
                            &[SqlValue::TextOwned(instance_id)],
                        )
                        .await
                    })
                })
                .await
                .unwrap();
        }
    }
    // Both arms above are terminal assertions of their backend's own
    // behaviour; no arm returns early, so nothing here is a runtime skip.
}
