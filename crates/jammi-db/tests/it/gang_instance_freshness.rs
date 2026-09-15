//! `Catalog::fresh_instance`: is a gang
//! coordinator's `instances` row FRESH within
//! `instance_liveness_margin(lease) == 2 * lease` on the DB clock?
//! Parameterized (sqlite/postgres, the `migrations.rs` shape):
//! `fresh_instance` resolves through `stale_before_clause`
//! (`super::lease`), which renders a DIFFERENT SQL expression per backend,
//! so this predicate has no oracle at all on the Postgres arm without it —
//! every test here also runs a `::postgres` arm gated by
//! `live-postgres-tests`, skipping (never failing) when `JAMMI_TEST_PG_URL`
//! is unset.

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
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
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
        .upsert_instance("inst-fresh", Some("label"), Some("host"))
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
        .upsert_instance("inst-stale", Some("label"), Some("host"))
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
        .upsert_instance("inst-borderline", Some("label"), Some("host"))
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
        .upsert_instance("inst-just-outside", Some("label"), Some("host"))
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
