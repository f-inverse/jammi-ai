//! `CONTRACT-U5a.md` §I1 — `Catalog::fresh_instance`: is a gang
//! coordinator's `instances` row FRESH within
//! `instance_liveness_margin(lease) == 2 * lease` on the DB clock? Every
//! test runs on a fresh SQLite catalog (a private tempdir per test).

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

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

#[tokio::test]
async fn fresh_instance_true_for_a_recently_seen_instance() {
    let (_dir, catalog) = base_catalog().await;
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

#[tokio::test]
async fn fresh_instance_false_for_an_absent_instance() {
    let (_dir, catalog) = base_catalog().await;
    let fresh = catalog
        .fresh_instance("no-such-instance", Duration::from_secs(30))
        .await
        .unwrap();
    assert!(!fresh, "an absent instance must never read as fresh");
}

#[tokio::test]
async fn fresh_instance_false_for_a_stale_instance() {
    let (_dir, catalog) = base_catalog().await;
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
#[tokio::test]
async fn fresh_instance_true_just_inside_the_liveness_margin() {
    let (_dir, catalog) = base_catalog().await;
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
