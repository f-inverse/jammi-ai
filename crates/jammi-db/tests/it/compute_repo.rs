//! `catalog::compute_repo` — generic CRUD over `compute_executors` /
//! `compute_jobs` (migration 038). Parameterized sqlite/postgres, the `migrations.rs` /
//! `gang_membership.rs` shape: every test also runs a `::postgres` arm
//! gated by `live-postgres-tests`, skipping (never failing) when
//! `JAMMI_TEST_PG_URL` is unset.
//!
//! The Postgres arm runs against ONE shared, persistent database
//! (`jammi_test_utils::unique_suffix`'s doc), so every row this file plants
//! carries a unique id and no assertion counts rows this file did not seed.

use std::cell::RefCell;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;

use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::compute_repo::{ComputeExecutorRecord, ComputeJobRecord};
use jammi_db::catalog::instance::DeviceFact;
use jammi_db::catalog::Catalog;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

async fn open_catalog(kind: BackendKind) -> Option<(tempfile::TempDir, Arc<Catalog>)> {
    let dir = tempdir().unwrap();
    let session = make_test_session(kind, dir.path()).await?;
    Some((dir, Arc::clone(session.catalog())))
}

/// The require-gate for the Postgres arm: a direct, crate-qualified call to
/// the registered `shared:` helper, textually in each `#[test]` fn's own body
/// (the KO-7 scanner is per-fn textual; `open_catalog`'s internal `?` on
/// `make_test_session` is one function away and does not dominate the skip —
/// `gang_membership.rs`'s own shape).
macro_rules! skip_unless_ready {
    ($kind:expr) => {
        if matches!($kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none()
        {
            eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
}

/// Tests own their rows: the Postgres arm shares one database with every
/// other lane on this host, so every executor row a test registers is
/// removed on the way out — on the passing arm and on a PANIC alike (the
/// body runs under `catch_unwind`, the rows are removed, then the panic
/// resumes). A `cuda` row a red test left behind would make a later lane's
/// device-kind refusal admit a plan no live executor can run.
async fn with_owned_rows<F: std::future::Future<Output = ()>>(
    catalog: &Arc<Catalog>,
    owned: &RefCell<Vec<String>>,
    body: F,
) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    let ids = owned.borrow().clone();
    for id in ids {
        catalog.remove_compute_executor(&id).await.ok();
    }
    if let Err(payload) = outcome {
        std::panic::resume_unwind(payload);
    }
}

fn executor(id: &str, devices: Vec<DeviceFact>) -> ComputeExecutorRecord {
    ComputeExecutorRecord {
        executor_id: id.to_string(),
        instance_id: format!("inst-{id}"),
        host: "127.0.0.1".to_string(),
        port: 50051,
        grpc_port: 50052,
        task_slots: 4,
        available_slots: 4,
        status: "live".to_string(),
        heartbeat_at: "2026-01-01T00:00:00.000000Z".to_string(),
        metadata: "{}".to_string(),
        devices,
    }
}

fn job(id: &str, owner: &str) -> ComputeJobRecord {
    ComputeJobRecord {
        job_id: id.to_string(),
        owner: owner.to_string(),
        status: "queued".to_string(),
        queued_at: "2026-01-01T00:00:00.000000Z".to_string(),
        updated_at: "2026-01-01T00:00:00.000000Z".to_string(),
    }
}

// ─── Executors: upsert / list / get / remove round trip ────────────────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn upsert_list_get_remove_round_trip(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let id = format!("exec-rt-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let rec = executor(
            &id,
            vec![DeviceFact {
                kind: "cuda".to_string(),
                ordinal: 0,
            }],
        );
        catalog.upsert_compute_executor(&rec).await.unwrap();

        let got = catalog.get_compute_executor(&id).await.unwrap();
        assert_eq!(got.as_ref(), Some(&rec), "get must round-trip every field");

        let listed = catalog.list_compute_executors().await.unwrap();
        assert!(
            listed.iter().any(|r| r == &rec),
            "list must include the upserted row: {listed:?}"
        );

        // Upsert again with a changed field: a re-upsert REPLACES, never merges.
        let mut replaced = rec.clone();
        replaced.status = "draining".to_string();
        replaced.available_slots = 1;
        catalog.upsert_compute_executor(&replaced).await.unwrap();
        let got = catalog.get_compute_executor(&id).await.unwrap().unwrap();
        assert_eq!(got, replaced);

        let removed = catalog.remove_compute_executor(&id).await.unwrap();
        assert!(removed, "remove must report the row existed");
        assert!(catalog.get_compute_executor(&id).await.unwrap().is_none());
        let removed_again = catalog.remove_compute_executor(&id).await.unwrap();
        assert!(!removed_again, "a second remove finds no row");
    })
    .await;
}

// ─── heartbeat updates only its own row ─────────────────────────────────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn heartbeat_updates_only_status_and_heartbeat_at(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let target = format!("exec-hb-target-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(target.clone());
        let other = format!("exec-hb-other-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(other.clone());
        catalog
            .upsert_compute_executor(&executor(&target, vec![]))
            .await
            .unwrap();
        catalog
            .upsert_compute_executor(&executor(&other, vec![]))
            .await
            .unwrap();

        let updated = catalog
            .record_compute_heartbeat(&target, "draining", "2026-06-01T00:00:00.000000Z")
            .await
            .unwrap();
        assert!(updated);

        let target_row = catalog
            .get_compute_executor(&target)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(target_row.status, "draining");
        assert_eq!(target_row.heartbeat_at, "2026-06-01T00:00:00.000000Z");
        // Every other field on the target row is untouched.
        assert_eq!(target_row.task_slots, 4);
        assert_eq!(target_row.available_slots, 4);
        assert_eq!(target_row.host, "127.0.0.1");

        // The sibling row is completely untouched.
        let other_row = catalog.get_compute_executor(&other).await.unwrap().unwrap();
        assert_eq!(other_row, executor(&other, vec![]));

        // A heartbeat naming no row reports `false`.
        let missing = format!("exec-hb-missing-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(missing.clone());
        let updated = catalog
            .record_compute_heartbeat(&missing, "live", "2026-06-01T00:00:00.000000Z")
            .await
            .unwrap();
        assert!(!updated);
    })
    .await;
}

// ─── adjust_compute_slots: atomicity ────────────────────────────────────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn adjust_compute_slots_is_atomic_across_the_whole_batch(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let a = format!("exec-adj-a-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(a.clone());
        let b = format!("exec-adj-b-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(b.clone());
        catalog
            .upsert_compute_executor(&executor(&a, vec![]))
            .await
            .unwrap();
        catalog
            .upsert_compute_executor(&executor(&b, vec![]))
            .await
            .unwrap();

        // A valid batch: both rows move.
        catalog
            .adjust_compute_slots(&[(a.as_str(), -1), (b.as_str(), -2)])
            .await
            .unwrap();
        assert_eq!(
            catalog
                .get_compute_executor(&a)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            3
        );
        assert_eq!(
            catalog
                .get_compute_executor(&b)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            2
        );

        // A batch where `a`'s delta is fine but `b`'s would push available_slots
        // (currently 2) below 0 by more than task_slots allows: the WHOLE batch must
        // refuse, and `a`'s row (already valid on its own) must be unchanged.
        let err = catalog
            .adjust_compute_slots(&[(a.as_str(), -1), (b.as_str(), -3)])
            .await
            .expect_err("a batch with one out-of-bounds delta must refuse entirely");
        let msg = err.to_string();
        assert!(
            msg.contains(&b) || msg.to_lowercase().contains("available_slots"),
            "got {msg}"
        );
        assert_eq!(
            catalog
                .get_compute_executor(&a)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            3,
            "a's row must be UNCHANGED: the batch touched nothing"
        );
        assert_eq!(
            catalog
                .get_compute_executor(&b)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            2,
            "b's row must be UNCHANGED: the batch touched nothing"
        );

        // A batch naming a missing executor_id also refuses the whole batch.
        let missing = format!("exec-adj-missing-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(missing.clone());
        let err = catalog
            .adjust_compute_slots(&[(a.as_str(), 1), (missing.as_str(), 1)])
            .await
            .expect_err("a batch naming a missing executor_id must refuse");
        assert!(err.to_string().contains(&missing));
        assert_eq!(
            catalog
                .get_compute_executor(&a)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            3,
            "a's row must be UNCHANGED when a sibling pair names no row"
        );
    })
    .await;
}

// ─── bind_compute_slots: CAS under concurrent binders ───────────────────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn bind_compute_slots_cas_admits_exactly_capacity_concurrent_binders(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let id = format!("exec-cas-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(id.clone());
        let mut rec = executor(&id, vec![]);
        rec.task_slots = 3;
        rec.available_slots = 3;
        catalog.upsert_compute_executor(&rec).await.unwrap();

        // 8 concurrent binders each ask for 1 slot against a 3-slot executor:
        // exactly 3 must win.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let catalog = Arc::clone(&catalog);
            let id = id.clone();
            handles.push(tokio::spawn(async move {
                catalog.bind_compute_slots(&id, 1).await.unwrap()
            }));
        }
        let mut won = 0;
        for h in handles {
            if h.await.unwrap() {
                won += 1;
            }
        }
        assert_eq!(won, 3, "exactly task_slots binders must win the CAS");
        assert_eq!(
            catalog
                .get_compute_executor(&id)
                .await
                .unwrap()
                .unwrap()
                .available_slots,
            0,
            "every slot is now bound"
        );

        // A ninth request against an exhausted executor loses too.
        assert!(!catalog.bind_compute_slots(&id, 1).await.unwrap());
    })
    .await;
}

// ─── compute_jobs: put / get / list / delete ────────────────────────────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn compute_jobs_put_get_list_delete(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("job-{}", jammi_test_utils::unique_suffix());
    let rec = job(&id, "owner-1");
    catalog.put_compute_job(&rec).await.unwrap();
    assert_eq!(
        catalog.get_compute_job(&id).await.unwrap(),
        Some(rec.clone())
    );

    let listed = catalog.list_compute_jobs().await.unwrap();
    assert!(listed.iter().any(|r| r == &rec));

    let mut updated = rec.clone();
    updated.status = "running".to_string();
    updated.updated_at = "2026-06-01T00:00:00.000000Z".to_string();
    catalog.put_compute_job(&updated).await.unwrap();
    assert_eq!(catalog.get_compute_job(&id).await.unwrap(), Some(updated));

    assert!(catalog.delete_compute_job(&id).await.unwrap());
    assert!(catalog.get_compute_job(&id).await.unwrap().is_none());
    assert!(!catalog.delete_compute_job(&id).await.unwrap());
}

// ─── list_compute_executor_devices: the placement policy's own read ────────

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_compute_executor_devices_reads_the_executors_own_column(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = open_catalog(kind)
        .await
        .expect("already skipped above when unconfigured");
    let owned = RefCell::new(Vec::<String>::new());
    with_owned_rows(&catalog, &owned, async {
        let no_devices = format!("exec-dev-none-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(no_devices.clone());
        let with_devices = format!("exec-dev-some-{}", jammi_test_utils::unique_suffix());
        owned.borrow_mut().push(with_devices.clone());
        catalog
            .upsert_compute_executor(&executor(&no_devices, vec![]))
            .await
            .unwrap();
        let devices = vec![
            DeviceFact {
                kind: "cuda".to_string(),
                ordinal: 0,
            },
            DeviceFact {
                kind: "cuda".to_string(),
                ordinal: 1,
            },
        ];
        catalog
            .upsert_compute_executor(&executor(&with_devices, devices.clone()))
            .await
            .unwrap();

        let all = catalog.list_compute_executor_devices().await.unwrap();
        let none_entry = all
            .iter()
            .find(|(id, _)| id == &no_devices)
            .expect("the no-devices executor must appear");
        assert_eq!(
            none_entry.1,
            Vec::new(),
            "no devices registered = empty list"
        );

        let some_entry = all
            .iter()
            .find(|(id, _)| id == &with_devices)
            .expect("the with-devices executor must appear");
        assert_eq!(
            some_entry.1, devices,
            "the JSON as written, decoded back exactly"
        );
    })
    .await;
}
