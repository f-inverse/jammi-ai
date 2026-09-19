//! `Catalog::list_gang_members` / `Catalog::peer_addr_of` (DESIGN.md § 4,
//! contract `feat_500-C-U5b-1a` §12 and unit U5b-1a-A2): the gang-membership
//! listing and by-id resolution verbs. `result_root` is written to every
//! member row verbatim, and its IDENTITY across spellings
//! (`result_root_identity`, derived by `MemberRoot::resolved`) is what the admission
//! predicate compares: members whose roots spell the same location
//! differently (`gcs://` vs `gs://`, a symlink and its target, a trailing
//! slash) are gang members of each other; members rooted elsewhere, and rows
//! with no identity, are not. Parameterized sqlite/postgres, the
//! `migrations.rs` / `gang_instance_freshness.rs` shape: every test also
//! runs a `::postgres` arm gated by `live-postgres-tests`.
//!
//! The Postgres arm runs every test in the lane against ONE shared, persistent
//! database (`jammi_test_utils::unique_suffix`'s doc), so two disciplines hold
//! in this file: a test never asserts a count over rows it did not seed (it
//! asserts the presence or absence of ITS row), and a test that plants a row
//! the listing predicate cannot tolerate (a corrupted `peer_addr`) deletes
//! that row BEFORE its assertion, on every arm, so a failure never leaks the
//! poison into every later listing in the lane.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::instance::{
    GangListing, InstanceRegistration, MemberRoot, PeerAddr, WorkerFacts,
};
use jammi_db::catalog::jobs_repo::WorkerState;
use jammi_db::catalog::lease::instance_prune_window;
use jammi_db::catalog::lease_keeper::LeaseTarget;
use jammi_db::catalog::Catalog;
use jammi_db::config::LeaseConfig;
use jammi_db::error::JammiError;
use jammi_db::tenant::TenantId;

use crate::common::catalog_on;

/// The member root every "matching" fixture in this file shares: a local
/// directory this process owns (the derivation creates it; a fixed absolute
/// path under `/` would be unwritable), one per test process.
fn root() -> &'static str {
    static SHARED: std::sync::OnceLock<(tempfile::TempDir, String)> = std::sync::OnceLock::new();
    &SHARED
        .get_or_init(|| {
            let dir = tempfile::tempdir().unwrap();
            let root = format!("file://{}/jammi_db", dir.path().to_str().unwrap());
            (dir, root)
        })
        .1
}
const LEASE: Duration = Duration::from_secs(30);

/// Force `instances.last_seen_at` into the past — mirrors
/// `gang_instance_freshness.rs`'s own helper.
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

/// Whether an `instances` row exists at all, fresh or stale — the row-scoped
/// witness the prune oracle asserts on, instead of a `prune_instances` count
/// that also counts every stale row a sibling test left in the shared
/// Postgres database.
async fn instance_row_exists(catalog: &Catalog, instance_id: &str) -> bool {
    let instance_id = instance_id.to_string();
    let rows = catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT instance_id FROM instances WHERE instance_id = $1",
                        &[SqlValue::TextOwned(instance_id)],
                        |row| row.get::<String>("instance_id"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    !rows.is_empty()
}

/// Corrupt an already-seeded row's `peer_addr` column out-of-band (a direct
/// `UPDATE`, never through [`PeerAddr::parse`]) — the state a hand-edited
/// row, or a future writer that skips the sealed constructor, leaves
/// behind. Both read verbs must surface this as the typed
/// [`jammi_db::error::JammiError::Catalog`] their doc comments promise
/// (`jobs_repo.rs` at [`Catalog::peer_addr_of`] and
/// [`Catalog::list_gang_members`]'s `# Errors` sections), never a panic and
/// never a silently-dropped row.
async fn force_corrupt_peer_addr(catalog: &Catalog, instance_id: &str) {
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET peer_addr = $1 WHERE instance_id = $2",
                    &[
                        SqlValue::Text("not an addr"),
                        SqlValue::TextOwned(instance_id),
                    ],
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
        Some(MemberRoot::new(root)),
    );
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(instance_id, kinds, state, &[])
        .await
        .unwrap();
}

/// The root every `root()`-rooted fixture shares — what a caller rooted
/// there passes as its own.
fn shared_root() -> MemberRoot {
    MemberRoot::new(root())
}

fn listing<'a>(kind: &'a str, self_instance: &'a str, root: &'a MemberRoot) -> GangListing<'a> {
    GangListing {
        kind,
        self_instance,
        root,
        lease: LEASE,
    }
}

// ---------------------------------------------------------------------------
// The exclusion matrix (P-M3, narrowed by P-Y1 §12 — the root arm is gone,
// replaced below by the "root is not consulted" inclusion oracles), each
// its own named case.
// ---------------------------------------------------------------------------

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_the_caller_itself(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    // Every OTHER predicate matches — fresh, claiming, matching kind —
    // proving the exclusion is the self check, not some other arm.
    seed_member(
        &catalog,
        &self_id,
        "10.0.0.1:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", &self_id, &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("stale-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    // instance_liveness_margin(30s) == 60s; push well past it.
    force_stale_instance(&catalog, &id, Duration::from_secs(600)).await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("draining-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        root(),
        "fine_tune",
        WorkerState::Draining,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("warming-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        root(),
        "fine_tune",
        WorkerState::Warming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("substr-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        root(),
        "graph_fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a 'graph_fine_tune' worker must never match a 'fine_tune' listing: {members:?}"
    );
}

/// The identity rules on object-store roots, through the verb: a member
/// whose root differs from the caller's only by a trailing slash is
/// returned; one whose bucket or key differs by case is not (the store and
/// the driver dial both as spelled).
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_folds_a_trailing_slash_but_neither_bucket_nor_key_case(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let me = MemberRoot::new("s3://bucket/prefix");
    let id_case = format!("auth-case-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id_case,
        "10.0.0.1:9000",
        "s3://BUCKET/prefix",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let id_slash = format!("slash-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id_slash,
        "10.0.0.2:9000",
        "s3://bucket/prefix/",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let id_key = format!("key-case-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id_key,
        "10.0.0.3:9000",
        "s3://bucket/Prefix",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &me))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id_case),
        "a bucket-case-divergent root is a DIFFERENT root — the driver dials it as spelled: {members:?}"
    );
    assert!(
        members.iter().any(|m| m.instance_id == id_slash),
        "a trailing-slash-divergent root is the same root: {members:?}"
    );
    assert!(
        members.iter().all(|m| m.instance_id != id_key),
        "a key-case-divergent root is a DIFFERENT root: {members:?}"
    );
}

/// Scheme aliasing, through the verb: `gcs://b/p` and `gs://b/p` are ONE
/// root, so each member is returned to a caller rooted at the other
/// spelling — and the rows still carry the two spellings verbatim.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn gcs_and_gs_spelled_members_are_gang_members_of_each_other(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let gcs_id = format!("gcs-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &gcs_id,
        "10.0.0.3:9000",
        "gcs://bucket/prefix",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let gs_id = format!("gs-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &gs_id,
        "10.0.0.4:9000",
        "gs://bucket/prefix",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    for (caller_root, other) in [
        ("gcs://bucket/prefix", &gs_id),
        ("gs://bucket/prefix", &gcs_id),
    ] {
        let me = MemberRoot::new(caller_root);
        let members = catalog
            .list_gang_members(listing("fine_tune", "someone-else", &me))
            .await
            .unwrap();
        assert!(
            members.iter().any(|m| &m.instance_id == other),
            "a caller rooted at {caller_root} must see the other spelling: {members:?}"
        );
    }
    let roots: Vec<Option<String>> = catalog
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let ids = vec![gcs_id.clone(), gs_id.clone()];
                Box::pin(async move {
                    let mut out = Vec::new();
                    for id in ids {
                        out.push(
                            tx.query_opt(
                                "SELECT result_root FROM instances WHERE instance_id = $1",
                                &[SqlValue::TextOwned(id)],
                                |row| row.try_get::<String>("result_root"),
                            )
                            .await?
                            .flatten(),
                        );
                    }
                    Ok(out)
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        roots,
        vec![
            Some("gcs://bucket/prefix".to_string()),
            Some("gs://bucket/prefix".to_string())
        ],
        "the rows keep the two spellings verbatim; only the identity folds"
    );
}

/// Two members rooted at DIFFERENT locations are not gang members of each
/// other, in either direction: a caller rooted at the local root sees only
/// the local member; a caller rooted at the bucket sees only the bucket
/// member.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn file_and_s3_rooted_members_are_not_gang_members_of_each_other(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let file_id = format!("filesch-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &file_id,
        "10.0.0.5:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let s3_id = format!("s3sch-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &s3_id,
        "10.0.0.6:9000",
        "s3://bucket/prefix",
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let as_file = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        as_file.iter().any(|m| m.instance_id == file_id),
        "{as_file:?}"
    );
    assert!(
        as_file.iter().all(|m| m.instance_id != s3_id),
        "{as_file:?}"
    );
    let s3 = MemberRoot::new("s3://bucket/prefix");
    let as_s3 = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &s3))
        .await
        .unwrap();
    assert!(as_s3.iter().any(|m| m.instance_id == s3_id), "{as_s3:?}");
    assert!(as_s3.iter().all(|m| m.instance_id != file_id), "{as_s3:?}");
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_null_peer_addr(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("nulladdr-{}", jammi_test_utils::unique_suffix());
    // A non-member registration: no peer_addr, no member_root.
    let reg = InstanceRegistration::new(&id, None, None, None, None);
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&id, "fine_tune", WorkerState::Claiming, &[])
        .await
        .unwrap();
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a NULL peer_addr row must never be returned: {members:?}"
    );
}

/// The distinct `peer_addr` SET / `result_root` NULL state (representable
/// by construction — no paired `CHECK`) has no root identity, so it is NOT
/// a gang member of anyone; an otherwise-identical full member is the
/// control. `peer_addr_of` (which has no root predicate) still resolves it.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_a_member_with_peer_addr_set_and_no_root(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("addr-no-root-{}", jammi_test_utils::unique_suffix());
    let reg = InstanceRegistration::new(
        &id,
        Some("label"),
        Some("host"),
        Some(PeerAddr::parse("10.0.0.6:9000").unwrap()),
        None,
    );
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&id, "fine_tune", WorkerState::Claiming, &[])
        .await
        .unwrap();
    let full_id = format!("addr-full-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &full_id,
        "10.0.0.7:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        members.iter().all(|m| m.instance_id != id),
        "a row with no root identity is never a gang member: {members:?}"
    );
    assert!(
        members.iter().any(|m| m.instance_id == full_id),
        "the control member must be returned: {members:?}"
    );
    let resolved = catalog.peer_addr_of(&id, LEASE).await.unwrap();
    assert_eq!(
        resolved.map(|a| a.as_str().to_string()),
        Some("10.0.0.6:9000".to_string()),
        "peer_addr_of has no root predicate and still resolves the row"
    );
}

/// A symlinked local root and its target are ONE root: each member is
/// returned to a caller rooted at the other spelling.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_symlinked_local_root_and_its_target_are_the_same_gang(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let roots = tempfile::tempdir().unwrap();
    let real = roots.path().join("real");
    std::fs::create_dir_all(real.join("jammi_db")).unwrap();
    let link = roots.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let real_root = format!("{}/jammi_db", real.to_str().unwrap());
    let link_root = format!("file://{}/jammi_db/", link.to_str().unwrap());
    let real_id = format!("real-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &real_id,
        "10.0.0.8:9000",
        &real_root,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let link_id = format!("link-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &link_id,
        "10.0.0.9:9000",
        &link_root,
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    for (caller_root, other) in [(&real_root, &link_id), (&link_root, &real_id)] {
        let me = MemberRoot::new(caller_root);
        let members = catalog
            .list_gang_members(listing("fine_tune", "someone-else", &me))
            .await
            .unwrap();
        assert!(
            members.iter().any(|m| &m.instance_id == other),
            "a caller rooted at {caller_root} must see the other spelling: {members:?}"
        );
    }
}

/// A row written before migration 036, or by a writer that skipped the
/// identity (the state a hand-edited row leaves behind): `result_root`
/// present, `result_root_identity` NULL — never a gang member, never an
/// error.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_row_with_a_root_but_no_identity_is_never_a_member(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("legacy-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.10:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let before = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        before.iter().any(|m| m.instance_id == id),
        "control: {before:?}"
    );
    let legacy = id.clone();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET result_root_identity = NULL WHERE instance_id = $1",
                    &[SqlValue::TextOwned(legacy)],
                )
                .await
            })
        })
        .await
        .unwrap();
    let after = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    assert!(
        after.iter().all(|m| m.instance_id != id),
        "a NULL identity never matches: {after:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_excludes_an_instance_with_no_workers_row(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("noworker-{}", jammi_test_utils::unique_suffix());
    let reg = InstanceRegistration::new(
        &id,
        None,
        None,
        Some(PeerAddr::parse("10.0.0.1:9000").unwrap()),
        Some(MemberRoot::new(root())),
    );
    catalog.upsert_instance(&reg).await.unwrap();
    // Deliberately no `upsert_worker` call: an `instances` row with no
    // `workers` row is a live process that never runs the claim loop, not a
    // fleet member (the INNER join, DESIGN.md § 4).
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("multi-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.5:9000",
        root(),
        "embedding, fine_tune ,other",
        WorkerState::Claiming,
    )
    .await;
    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
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
            root(),
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

    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("tenant-indep-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.7:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    let unscoped = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await
        .unwrap();
    let scoped_catalog =
        catalog.pinned_to_tenant(Some(TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap()));
    let scoped = scoped_catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    // "Busy" here means: this verb has NO kind/state filter at all — a
    // draining worker of a totally different kind still resolves by id.
    let id = format!("busy-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.3:9000",
        root(),
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
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("staleaddr-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.4:9000",
        root(),
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
    let (_dir, catalog) = catalog_on(kind).await;
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
    let (_dir, catalog) = catalog_on(kind).await;
    let resolved = catalog
        .peer_addr_of("no-such-instance-gang-membership", LEASE)
        .await
        .unwrap();
    assert!(resolved.is_none());
}

/// A row whose `peer_addr` column was corrupted out-of-band (never through
/// [`PeerAddr::parse`]) surfaces the typed [`JammiError::Catalog`] naming
/// the instance — never a panic, never a silently `None` result.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn peer_addr_of_returns_the_typed_error_for_a_corrupted_peer_addr(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("corrupt-addr-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.9:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    force_corrupt_peer_addr(&catalog, &id).await;
    let result = catalog.peer_addr_of(&id, LEASE).await;
    // The poison row leaves the shared database before any assertion can
    // fail, so a red here never cascades into every later listing.
    force_delete_instance(&catalog, &id).await;
    let err = result.expect_err("a corrupted peer_addr must be a typed error, never a silent None");
    match err {
        JammiError::Catalog(msg) => {
            assert!(
                msg.contains(&id),
                "the error must name the corrupted instance: {msg}"
            );
        }
        other => panic!("expected JammiError::Catalog, got {other:?}"),
    }
}

/// The `list_gang_members` sibling of the case above: a matching, fresh,
/// claiming candidate whose `peer_addr` is corrupted out-of-band surfaces
/// the same typed error, never a silently-dropped candidate.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn list_gang_members_returns_the_typed_error_for_a_corrupted_peer_addr(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let id = format!("corrupt-list-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.10:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    force_corrupt_peer_addr(&catalog, &id).await;
    let result = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
        .await;
    // Same discipline as the `peer_addr_of` case: the poison row is gone
    // before the assertion, on every arm.
    force_delete_instance(&catalog, &id).await;
    let err =
        result.expect_err("a corrupted peer_addr candidate must be a typed error, never dropped");
    match err {
        JammiError::Catalog(msg) => {
            assert!(
                msg.contains(&id),
                "the error must name the corrupted instance: {msg}"
            );
        }
        other => panic!("expected JammiError::Catalog, got {other:?}"),
    }
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
    let (dir, catalog) = catalog_on(kind).await;
    let instance_id = format!("pm4-{}", jammi_test_utils::unique_suffix());
    let reg = Arc::new(InstanceRegistration::new(
        instance_id.clone(),
        Some("label"),
        Some("host"),
        Some(PeerAddr::parse("10.0.0.9:9000").unwrap()),
        Some(MemberRoot::new(root())),
    ));
    reg.set_worker(Some(WorkerFacts {
        kinds: "fine_tune".into(),
        state: WorkerState::Claiming,
        devices: vec![],
    }));
    catalog.upsert_instance(&reg).await.unwrap();
    catalog
        .upsert_worker(&instance_id, "fine_tune", WorkerState::Claiming, &[])
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

    let members = catalog
        .list_gang_members(listing("fine_tune", "someone-else", &shared_root()))
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
    let (_dir, catalog) = catalog_on(kind).await;
    let lease = Duration::from_secs(10);
    // margin = 20s, window = 30s. 25s ago is stale (past the margin) but
    // strictly inside the window.
    let id = format!("prunewin-{}", jammi_test_utils::unique_suffix());
    seed_member(
        &catalog,
        &id,
        "10.0.0.1:9000",
        root(),
        "fine_tune",
        WorkerState::Claiming,
    )
    .await;
    force_stale_instance(&catalog, &id, Duration::from_secs(25)).await;
    assert!(
        !catalog.fresh_instance(&id, lease).await.unwrap(),
        "25s ago must already read stale past the 20s margin"
    );
    // The oracle is row-scoped: `prune_instances` returns the count over the
    // WHOLE table, which on the shared Postgres database also counts every
    // stale row a sibling test left behind, so the count is never asserted.
    catalog
        .prune_instances(instance_prune_window(lease))
        .await
        .unwrap();
    assert!(
        instance_row_exists(&catalog, &id).await,
        "a member merely stale within the (margin, window] range must not be pruned"
    );

    // Symmetric case: past the window, the row IS pruned.
    force_stale_instance(&catalog, &id, Duration::from_secs(35)).await;
    let deleted = catalog
        .prune_instances(instance_prune_window(lease))
        .await
        .unwrap();
    assert!(
        deleted >= 1,
        "the prune past the window must report at least this row"
    );
    assert!(
        !instance_row_exists(&catalog, &id).await,
        "a member stale past the window must be pruned"
    );
}
