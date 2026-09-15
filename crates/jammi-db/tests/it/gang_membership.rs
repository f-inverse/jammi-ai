//! `Catalog::list_gang_members` / `Catalog::peer_addr_of` (DESIGN.md § 4,
//! contract `feat_500-C-U5b-1a` M3): the gang-membership listing and by-id
//! resolution verbs. Parameterized sqlite/postgres, the `migrations.rs` /
//! `gang_instance_freshness.rs` shape: every test also runs a `::postgres`
//! arm gated by `live-postgres-tests`, skipping (never failing) when
//! `JAMMI_TEST_PG_URL` is unset.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::instance::{
    CanonicalRoot, GangListing, InstanceRegistration, PeerAddr, WorkerFacts,
};
use jammi_db::catalog::jobs_repo::WorkerState;
use jammi_db::catalog::lease::instance_prune_window;
use jammi_db::catalog::lease_keeper::LeaseTarget;
use jammi_db::catalog::Catalog;
use jammi_db::config::LeaseConfig;
use jammi_db::tenant::TenantId;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

/// The canonical root every "matching" fixture in this file shares.
const ROOT: &str = "file:///shared/jammi_db";
const LEASE: Duration = Duration::from_secs(30);

async fn base_catalog_kind(kind: BackendKind) -> Option<(tempfile::TempDir, Arc<Catalog>)> {
    let dir = tempdir().unwrap();
    let session = make_test_session(kind, dir.path()).await?;
    Some((dir, Arc::clone(session.catalog())))
}

/// Force `instances.last_seen_at` into the past — mirrors
/// `gang_instance_freshness.rs`'s own helper.
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

/// Force-delete an `instances` row (`workers` cascades with it) — the state
/// a stale sweep, or a process that lost its row during a transient outage,
/// leaves behind.
async fn force_delete_instance(catalog: &Catalog, instance_id: &str) {
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
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

/// Seed a fresh `instances` + `workers` row: `peer_addr`/`result_root` set,
/// `kinds`/`state` as given.
async fn seed_member(
    catalog: &Catalog,
    instance_id: &str,
    peer_addr: &str,
    root: &str,
    kinds: &str,
    state: WorkerState,
) {
    let reg = InstanceRegistration::new(
        instance_id,
        Some("label"),
        Some("host"),
        Some(PeerAddr::parse(peer_addr).unwrap()),
        Some(CanonicalRoot::new(root)),
    );
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(instance_id, kinds, state)
        .await
        .unwrap();
}

fn listing<'a>(kind: &'a str, self_instance: &'a str, root: &'a CanonicalRoot) -> GangListing<'a> {
    GangListing {
        kind,
        self_instance,
        canonical_root: root,
        lease: LEASE,
    }
}

macro_rules! skip_unless_ready {
    ($kind:expr) => {
        // The require-gate itself: a direct, crate-qualified call to the
        // registered `shared:` helper, textually in THIS test fn's own body
        // — `base_catalog_kind`'s internal `?` on `make_test_session` is one
        // function away and does not dominate this skip for the KO-7
        // scanner, which is per-`#[test]`-fn textual, not whole-file
        // (`migrations.rs`/`gang_instance_freshness.rs`'s own shape).
        if matches!($kind, BackendKind::Postgres) && jammi_test_utils::pg_url_for_tests().is_none()
        {
            eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// The exclusion matrix (P-M3), each its own named case.
// ---------------------------------------------------------------------------

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_the_caller_itself(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    // Every OTHER predicate matches — fresh, claiming, matching kind and
    // root — proving the exclusion is the self check, not some other arm.
    seed_member(
        &catalog,
        &self_id,
        "10.0.0.1:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", &self_id, &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != self_id),
        "the caller's own row must never be returned: {members:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_stale_member(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("stale-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    // instance_liveness_margin(30s) == 60s; push well past it.
    force_stale_instance(&catalog, &id, Duration::from_secs(600)).await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a stale member must never be returned: {members:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_draining_worker(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("draining-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        ROOT,
        "fine_tune",
        WorkerState::Draining,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a draining worker must never be returned: {members:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_warming_worker(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("warming-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        ROOT,
        "fine_tune",
        WorkerState::Warming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a warming worker must never be returned: {members:?}"
    );
}

/// A `kinds` token that CONTAINS the requested kind as a substring (never a
/// whole token) must not match: `graph_fine_tune` vs `fine_tune`.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_kind_that_is_only_a_substring_token(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("substr-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        ROOT,
        "graph_fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a 'graph_fine_tune' worker must never match a 'fine_tune' listing: {members:?}"
    );
}

/// A root that differs only by CASE, or by a trailing `/`, is a byte-exact
/// mismatch at this layer (this verb never re-canonicalizes) — both must be
/// excluded.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_root_divergent_by_case_or_trailing_slash(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id_case = format!("root-case-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id_case,
        "10.0.0.1:9000",
        "FILE:///SHARED/jammi_db",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let id_slash = format!("root-slash-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id_slash,
        "10.0.0.2:9000",
        "file:///shared/jammi_db/",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id_case),
        "a case-divergent root must never match byte-for-byte: {members:?}"
    );
    assert!(
        members.iter().all(|m| m.instance_id != id_slash),
        "a trailing-slash-divergent root must never match byte-for-byte: {members:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_null_peer_addr(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("nulladdr-{}", jammi_test_utils::unique_suffix());
    // A non-member registration: no peer_addr, no canonical_root.
    let reg = InstanceRegistration::new(&id, None, None, None, None);
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&id, "fine_tune", WorkerState::Claiming)
        .await
        .unwrap();
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a NULL peer_addr row must never be returned: {members:?}"
    );
}

/// The distinct `peer_addr` SET / `result_root` NULL state (representable
/// by construction — no paired `CHECK`, schema.rs ~:1225) gets its OWN
/// oracle, separate from the both-NULL case above: excluded from
/// `list_gang_members` (both are required by the listing predicate), while
/// an otherwise-identical full member IS returned, and `peer_addr_of` still
/// resolves it — a NULL `result_root` does not hide the address, since
/// `peer_addr_of` has no root predicate at all.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_member_with_peer_addr_set_but_result_root_null(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("addr-no-root-{}", jammi_test_utils::unique_suffix());
    // peer_addr SET, canonical_root NULL — distinct from both-NULL above.
    let reg = InstanceRegistration::new(
        &id,
        Some("label"),
        Some("host"),
        Some(PeerAddr::parse("10.0.0.6:9000").unwrap()),
        None,
    );
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&id, "fine_tune", WorkerState::Claiming)
        .await
        .unwrap();

    // An otherwise-identical full member (peer_addr AND result_root both
    // set) — the control proving the exclusion above is the NULL
    // result_root, not some other divergence between the two rows.
    let full_id = format!("addr-full-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &full_id,
        "10.0.0.7:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;

    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a NULL result_root row must never be returned by list_gang_members: {members:?}"
    );
    assert!(
        members.iter().any(|m| m.instance_id == full_id),
        "the otherwise-identical full member must still be returned: {members:?}"
    );

    // peer_addr_of has no root predicate at all — a NULL result_root must
    // never hide the address.
    let resolved = catalog.peer_addr_of(&id, LEASE).await.unwrap();
    assert_eq!(
        resolved.map(|a| a.as_str().to_string()),
        Some("10.0.0.6:9000".to_string()),
        "peer_addr_of must still resolve a NULL-result_root instance"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_an_instance_with_no_workers_row(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("noworker-{}", jammi_test_utils::unique_suffix());
    let reg = InstanceRegistration::new(
        &id,
        None,
        None,
        Some(PeerAddr::parse("10.0.0.1:9000").unwrap()),
        Some(CanonicalRoot::new(ROOT)),
    );
    catalog.upsert_instance(&reg).await.unwrap();
    // Deliberately no `upsert_worker` call: an `instances` row with no
    // `workers` row is a live process that never runs the claim loop, not a
    // fleet member (the INNER join, DESIGN.md § 4).
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a process with no `workers` row must never be returned: {members:?}"
    );
}

// ---------------------------------------------------------------------------
// The positive case, byte-order, and tenant independence.
// ---------------------------------------------------------------------------

/// A fresh, `claiming`, multi-kind worker whose kinds contain the requested
/// token IS returned, with the right `peer_addr`.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_includes_a_fresh_multi_kind_claiming_worker(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("multi-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.5:9000",
        ROOT,
        "embedding, fine_tune ,other",
        WorkerState::Claiming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    let member = members
        .iter()
        .find(|m| m.instance_id == id)
        .unwrap_or_else(|| panic!("the matching multi-kind worker must be returned: {members:?}"));
    assert_eq!(member.peer_addr.as_str(), "10.0.0.5:9000");
}

/// P-M3's byte-order property, with the §8 control: ids inserted in
/// DESCENDING byte order so the DB's natural (no `ORDER BY`) return order is
/// provably NOT already sorted — the raw-order control fails BY NAME, never
/// skips, if this assumption ever stops holding.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let suffix = jammi_test_utils::unique_suffix();
    let c = format!("zz-byteorder-c-{suffix}");
    let b = format!("zz-byteorder-b-{suffix}");
    let a = format!("zz-byteorder-a-{suffix}");
    // Inserted in DESCENDING byte order: c, then b, then a.
    for id in [&c, &b, &a] {
        seed_member(
            &catalog,
            id,
            "10.0.0.1:9000",
            ROOT,
            "fine_tune",
            WorkerState::Claiming,
        )
        .await;
    }

    // The control: the RAW, un-ordered SELECT must NOT already be ascending
    // — proving the sort assertion below is not vacuous. Selecting ONLY
    // `instance_id` (measured) lets both backends' planners satisfy the
    // query from the primary-key index alone (a covering-index scan),
    // returning KEY order "for free" regardless of insertion order — so a
    // second, non-indexed column (`started_at`) is selected too, forcing a
    // real row/heap scan in INSERTION order, exactly `c, b, a`.
    let raw: Vec<String> = catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT instance_id, started_at FROM instances",
                        &[],
                        |row| row.get::<String>("instance_id"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    let mut sorted_raw = raw.clone();
    sorted_raw.sort();
    assert_ne!(
        raw, sorted_raw,
        "control failed: the query planner returned these ids in \
         already-ascending order, so the sort assertion below would pass \
         vacuously; raw={raw:?}"
    );

    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    let ours: Vec<&str> = members
        .iter()
        .filter(|m| m.instance_id == a || m.instance_id == b || m.instance_id == c)
        .map(|m| m.instance_id.as_str())
        .collect();
    assert_eq!(
        ours,
        vec![a.as_str(), b.as_str(), c.as_str()],
        "list_gang_members must return these three in ascending byte order: {members:?}"
    );
}

/// `instances` carries no tenant column: `list_gang_members` returns the
/// SAME answer under a scoped tenant binding and under none — mirroring
/// `Catalog::get_job_for_rank`'s own tenant-independence oracle.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_gang_members_is_identical_under_a_scoped_tenant_and_under_none(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("tenant-indep-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.7:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let root = CanonicalRoot::new(ROOT);
    let unscoped = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    let scoped_catalog =
        catalog.pinned_to_tenant(Some(TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap()));
    let scoped = scoped_catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    assert_eq!(
        unscoped, scoped,
        "list_gang_members must be identical under any tenant scope"
    );
}

// ---------------------------------------------------------------------------
// `peer_addr_of`.
// ---------------------------------------------------------------------------

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn peer_addr_of_resolves_a_busy_or_other_kind_fresh_member(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    // "Busy" here means: this verb has NO kind/state filter at all — a
    // draining worker of a totally different kind still resolves by id.
    let id = format!("busy-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.3:9000",
        ROOT,
        "embedding",
        WorkerState::Draining,
    )
    .await;
    let resolved = catalog.peer_addr_of(&id, LEASE).await.unwrap();
    assert_eq!(
        resolved.map(|a| a.as_str().to_string()),
        Some("10.0.0.3:9000".to_string())
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn peer_addr_of_is_none_for_a_stale_instance(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("staleaddr-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.4:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    force_stale_instance(&catalog, &id, Duration::from_secs(600)).await;
    let resolved = catalog.peer_addr_of(&id, LEASE).await.unwrap();
    assert!(resolved.is_none(), "a stale instance must never resolve");
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn peer_addr_of_is_none_for_a_null_peer_addr(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let id = format!("nulladdr2-{}", jammi_test_utils::unique_suffix());
    let reg = InstanceRegistration::new(&id, None, None, None, None);
    catalog.upsert_instance(&reg).await.unwrap();
    let resolved = catalog.peer_addr_of(&id, LEASE).await.unwrap();
    assert!(
        resolved.is_none(),
        "a NULL peer_addr row must never resolve"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn peer_addr_of_is_none_for_an_absent_instance(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let resolved = catalog
        .peer_addr_of("no-such-instance-gang-membership", LEASE)
        .await
        .unwrap();
    assert!(resolved.is_none());
}

// ---------------------------------------------------------------------------
// M4 / P-M4 (restated over membership, §8 B1): the keeper's whole-tuple
// re-registration, and the prune window.
// ---------------------------------------------------------------------------

/// A process whose `instances` row was force-deleted during a transient
/// outage (its `workers` row cascades with it) is a gang member again after
/// ONE real `LeaseKeeper` pass — `peer_addr`, `result_root`, `kinds`, and
/// `state` all byte-identical to before. RED at base: `touch_instance` is a
/// pure `UPDATE` that can never resurrect a deleted row. Uses the REAL
/// `LeaseKeeper`, never a direct re-insert.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let instance_id = format!("pm4-{}", jammi_test_utils::unique_suffix());
    let reg = Arc::new(InstanceRegistration::new(
        instance_id.clone(),
        Some("label"),
        Some("host"),
        Some(PeerAddr::parse("10.0.0.9:9000").unwrap()),
        Some(CanonicalRoot::new(ROOT)),
    ));
    reg.set_worker(Some(WorkerFacts {
        kinds: "fine_tune".into(),
        state: WorkerState::Claiming,
    }));
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&instance_id, "fine_tune", WorkerState::Claiming)
        .await
        .unwrap();

    let intervals = LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap();
    let keeper = crate::common::keeper_for_backend(kind, dir.path().to_path_buf(), intervals).await;
    let _hold = keeper.hold(LeaseTarget::Instance(Arc::clone(&reg)));

    // The state a stale sweep, or a transient outage, leaves behind.
    force_delete_instance(&catalog, &instance_id).await;

    // One real keeper pass: wait a bit over one heartbeat tick.
    tokio::time::sleep(intervals.heartbeat() + Duration::from_millis(500)).await;

    let root = CanonicalRoot::new(ROOT);
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &root))
        .await
        .unwrap();
    let member = members
        .iter()
        .find(|m| m.instance_id == instance_id)
        .unwrap_or_else(|| {
            panic!("the keeper must have reregistered the whole tuple: {members:?}")
        });
    assert_eq!(
        member.peer_addr.as_str(),
        "10.0.0.9:9000",
        "peer_addr must be byte-identical to before"
    );
    let resolved = catalog
        .peer_addr_of(&instance_id, intervals.lease())
        .await
        .unwrap();
    assert_eq!(
        resolved.map(|a| a.as_str().to_string()),
        Some("10.0.0.9:9000".to_string())
    );

    keeper.shutdown_and_join(Duration::from_secs(10)).await.ok();
}

/// The prune window (`3 * lease`) is STRICTLY beyond the liveness margin
/// (`2 * lease`): a member stale in `(margin, window]` is NOT pruned.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn prune_window_does_not_prune_a_member_merely_stale_within_the_window(kind: BackendKind) {
    skip_unless_ready!(kind);
    let (_dir, catalog) = base_catalog_kind(kind)
        .await
        .expect("already skipped above when unconfigured");
    let lease = Duration::from_secs(10);
    // margin = 20s, window = 30s. 25s ago is stale (past the margin) but
    // strictly inside the window.
    let id = format!("prunewin-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        ROOT,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    force_stale_instance(&catalog, &id, Duration::from_secs(25)).await;
    assert!(
        !catalog.fresh_instance(&id, lease).await.unwrap(),
        "25s ago must already read stale past the 20s margin"
    );
    let deleted = catalog
        .prune_instances(instance_prune_window(lease))
        .await
        .unwrap();
    assert_eq!(
        deleted, 0,
        "a member merely stale within the (margin, window] range must not be pruned"
    );

    // Symmetric case: past the window, the row IS pruned.
    force_stale_instance(&catalog, &id, Duration::from_secs(35)).await;
    let deleted = catalog
        .prune_instances(instance_prune_window(lease))
        .await
        .unwrap();
    assert_eq!(deleted, 1, "a member stale past the window must be pruned");
}
