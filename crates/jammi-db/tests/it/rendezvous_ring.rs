//! `Catalog::list_ring_members` (RENDEZVOUS RV2): the RENDEZVOUS ring read —
//! live, root-sharing, `peer_addr`-set `instances` rows, SELF INCLUDED, no
//! `workers` join and no kind vocabulary. Mirrors `gang_membership.rs`'s
//! fixtures and discipline: parameterized sqlite/postgres, a test asserts the
//! presence or absence of ITS OWN row only (the Postgres arm shares one
//! persistent database across the lane).

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::instance::{InstanceRegistration, MemberRoot, PeerAddr};
use jammi_db::catalog::lease::instance_liveness_margin;
use jammi_db::catalog::Catalog;
use jammi_db::index::{RendezvousPlacement, SegmentId, SegmentPlacement};

use crate::common::catalog_on;

const LEASE: Duration = Duration::from_secs(30);

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

fn other_root() -> &'static str {
    static SHARED: std::sync::OnceLock<(tempfile::TempDir, String)> = std::sync::OnceLock::new();
    &SHARED
        .get_or_init(|| {
            let dir = tempfile::tempdir().unwrap();
            let root = format!("file://{}/jammi_db", dir.path().to_str().unwrap());
            (dir, root)
        })
        .1
}

/// A root NO OTHER test in this file (or this file's `postgres` arm's shared,
/// persistent lane database) ever shares — for a property that asserts
/// something about the WHOLE ring ("self is the only member"), which
/// `root()`'s shared identity would falsify the instant a sibling test's
/// still-fresh row (seeded moments earlier, well inside the liveness margin)
/// shares it. The backing directory is leaked (never cleaned up) so it
/// outlives the process — this root is used exactly once, by exactly one
/// test, for a process that exits shortly after.
fn fresh_root() -> String {
    let dir = tempfile::tempdir().unwrap();
    let root = format!("file://{}/jammi_db", dir.path().to_str().unwrap());
    std::mem::forget(dir);
    root
}

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

/// Seed an `instances` row directly (no `workers` row at all — the
/// `[worker] enabled = false` shape: a process that advertises without ever
/// running a claim loop). `peer_addr`/`root` are `None` for a non-advertising
/// process.
async fn seed_instance(
    catalog: &Catalog,
    instance_id: &str,
    peer_addr: Option<&str>,
    root: Option<&str>,
) {
    let reg = InstanceRegistration::new(
        instance_id,
        Some("label"),
        Some("host"),
        peer_addr.map(|a| PeerAddr::parse(a).unwrap()),
        root.map(MemberRoot::new),
    );
    catalog.upsert_instance(&reg).await.unwrap();
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn self_appears_as_a_candidate(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    let ring = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap();
    assert!(
        ring.iter().any(|m| m.instance_id == self_id),
        "the RENDEZVOUS ring must include the caller's own row (unlike list_gang_members): {ring:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_stale_row_is_excluded(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    let stale_id = format!("stale-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    seed_instance(&catalog, &stale_id, Some("10.0.0.2:9000"), Some(root())).await;
    // instance_liveness_margin(30s) == 60s; push well past it.
    force_stale_instance(&catalog, &stale_id, Duration::from_secs(600)).await;
    let ring = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap();
    assert!(
        ring.iter().all(|m| m.instance_id != stale_id),
        "a stale row must never be a ring candidate: {ring:?}"
    );
    assert!(ring.iter().any(|m| m.instance_id == self_id));
}

/// The exclusion is proven here; it is NOT separately counted anywhere (no
/// metric distinguishes "excluded for a foreign root" from "excluded for
/// staleness" or "no peer_addr") — a second statement to tag the reason
/// would violate RV6's single-statement budget for a distinction only
/// human debugging would use, never placement correctness. See
/// `RendezvousPlacement`'s own rustdoc for the full reasoning; this test is
/// that claim's executed oracle, alongside `gang_membership`'s own
/// root-mismatch suite proving the SAME shared `live_with_root_clause`
/// fragment excludes correctly from `Catalog::list_gang_members`'s side too.
#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_root_identity_mismatched_row_is_excluded(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    let elsewhere_id = format!("elsewhere-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    seed_instance(
        &catalog,
        &elsewhere_id,
        Some("10.0.0.2:9000"),
        Some(other_root()),
    )
    .await;
    let ring = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap();
    assert!(
        ring.iter().all(|m| m.instance_id != elsewhere_id),
        "a member rooted elsewhere must never be a ring candidate: {ring:?}"
    );
    assert!(ring.iter().any(|m| m.instance_id == self_id));
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_non_advertising_process_is_not_a_member(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    let library_id = format!("library-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    // A library/CLI process: no `peer_advertise`, so `peer_addr`/`root` are
    // both `None` — exactly `InstanceRegistration::from_config`'s non-member
    // shape.
    seed_instance(&catalog, &library_id, None, None).await;
    let ring = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap();
    assert!(
        ring.iter().all(|m| m.instance_id != library_id),
        "a process with no peer_addr must never be a ring candidate: {ring:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn an_advertising_replica_with_no_workers_row_is_still_a_member(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    // `[worker] enabled = false`: advertises (`peer_advertise` set) but never
    // runs a claim loop, so it never upserts a `workers` row at all. RV2:
    // "a `[worker] enabled=false` advertising replica IS a member" — the
    // ring predicate joins no `workers` table, unlike `list_gang_members`.
    let quiet_id = format!("quiet-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    seed_instance(&catalog, &quiet_id, Some("10.0.0.2:9000"), Some(root())).await;
    let ring = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap();
    assert!(
        ring.iter().any(|m| m.instance_id == quiet_id),
        "an advertising replica with no workers row must still be a ring member: {ring:?}"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn plan_falls_back_to_all_local_and_counts_when_self_is_absent_from_the_ring(
    kind: BackendKind,
) {
    let (_dir, catalog) = catalog_on(kind).await;
    // `self_instance_id` names a row that was NEVER registered: the
    // self-referencing subquery in `list_ring_members` returns no rows, so
    // `result_root_identity = (subquery)` is `NULL = anything` (never true)
    // for every candidate — the ring reads empty even if OTHER members exist.
    let never_registered = format!("ghost-{}", jammi_test_utils::unique_suffix());
    let other_id = format!("other-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &other_id, Some("10.0.0.9:9000"), Some(root())).await;

    let placement = RendezvousPlacement::new(
        Arc::clone(&catalog),
        never_registered,
        instance_liveness_margin(LEASE),
    );
    let plan = placement
        .plan("t", &[SegmentId(0), SegmentId(1)])
        .await
        .unwrap();
    assert_eq!(
        plan,
        vec![Vec::new(), Vec::new()],
        "an empty (or self-absent) ring must fall back to all-local for every segment"
    );
    assert_eq!(
        placement.metrics().ring_empty_total(),
        1,
        "the ring-empty fallback must be counted, never silent"
    );
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn plan_arms_local_when_self_is_the_only_ring_member(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    seed_instance(
        &catalog,
        &self_id,
        Some("10.0.0.1:9000"),
        Some(&fresh_root()),
    )
    .await;
    let placement = RendezvousPlacement::new(
        Arc::clone(&catalog),
        self_id,
        instance_liveness_margin(LEASE),
    );
    let plan = placement
        .plan("t", &[SegmentId(0), SegmentId(1), SegmentId(2)])
        .await
        .unwrap();
    assert_eq!(
        plan,
        vec![Vec::new(), Vec::new(), Vec::new()],
        "with self the ONLY ring member, every segment must arm local (first == self)"
    );
    assert_eq!(
        placement.metrics().ring_empty_total(),
        0,
        "a real, non-empty ring (of one) must never count the ring-empty fallback"
    );
}

/// RENDEZVOUS RV6: the per-search ring-read cost, measured (never assumed)
/// at 100 and 10k `instances` rows on the scratch Postgres, over
/// `BackendImpl::query_untransacted` (no `BEGIN`/`SET TRANSACTION
/// .../COMMIT` — see `RendezvousPlacement`'s own rustdoc for the wrapper's
/// measured cost at each scale). Live only (requires `JAMMI_TEST_PG_URL`;
/// skips, never fails, otherwise) — this is a COST measurement, not a
/// correctness oracle (those are the tests above), so it prints the
/// measured milliseconds and asserts a HOST-RELATIVE bound: at each scale
/// the ring read may cost at most four times a plain transfer of the same
/// number of `(instance_id, peer_addr)` rows through this test's own pool
/// (plus a 2 ms floor for sub-millisecond noise), measured in the same
/// process seconds apart. That isolates what this measurement is about —
/// the predicate and plan cost RV3/RV6 changed — from wire transfer and
/// per-row decode, which scale with the ring size AND the host (an absolute
/// budget calibrated on one host tripped on a shared CI runner at 24.9 ms
/// for a read this host does in ~10 ms). Measured here over 3 repeated runs
/// (a warmed connection, isolating the query's own cost from a process's
/// first-connection handshake): 2.1-2.8 ms at 101 rows against a 0.40-0.59 ms
/// transfer of the same 51 rows — the ring read is ~5x the transfer at that
/// size because the fixed per-statement cost (planning plus the root
/// InitPlan) dominates, which is what the 2 ms floor is for; 8.6-8.9 ms at
/// 10,101 rows against a 6.8-7.1 ms transfer of the same 5,051 rows (1.2-1.3x:
/// past the fixed cost, the read IS the transfer). Earlier captures on this
/// host measured ~10.2-10.9 ms at 10,101 rows (`EXPLAIN` shows a Seq Scan — see `RendezvousPlacement`'s
/// doc for why, and why the remaining cost past the scan itself is the
/// 5,051-row result transfer, not the scan or the transaction wrapper this
/// read no longer pays at all). The 50 %-root-sharing 10k-row shape is a
/// deliberately ADVERSARIAL stress fixture (a fleet that let `instances`
/// bloat with thousands of unpruned rows sharing one root), never the
/// realistic ring size this placement is sized for, so the bound is stated
/// on that pathological shape, not on the realistic one (which the 101-row
/// number already covers with room to spare). What this bound catches is a
/// regression that makes the predicate expensive per candidate row on the
/// order of an index probe per row (a correlated subplan, ~1-10 µs/row
/// against a detection threshold of ~2 µs/row here and ~3.5 µs/row on the
/// CI image): it moves the ring read but not the baseline transfer, and
/// fails here by name. It does NOT catch a lost sargable rewrite — the cast
/// form costs ~0.4 µs/row (measured in `lease.rs`'s own oracle), ~4 ms at
/// this scale, inside the headroom; sargability is guarded by
/// `stale_before_clause_postgres_is_sargable_and_agrees_with_the_cast_form`
/// (`crates/jammi-db/src/catalog/lease.rs`), which asserts the plan shape.
#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn ring_read_cost_is_measured_at_100_and_10k_instance_rows() {
    let url = jammi_test_utils::postgres_url();
    let (_dir, catalog) = catalog_on(BackendKind::Postgres).await;
    let self_id = format!("cost-self-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;

    // Bulk-seed noise rows directly (never through the Rust API — 10k
    // individual round trips would dominate the measurement itself): half
    // share this run's root, half are rooted elsewhere; all fresh. This
    // 50%-root-sharing split is a deliberately ADVERSARIAL stress shape
    // (maximum candidate count at this table size), not a realistic ring —
    // a real ring is bounded by one deployment's own live replica count.
    // `a_stale_row_is_excluded` already proves the margin predicate on its
    // own, realistic-sized fixture.
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(2)
        .connect(&url)
        .await
        .expect("connect to the scratch Postgres");
    // The IDENTITY `seed_instance`/`MemberRoot::resolved` derived for self's
    // own row — NEVER `root()`'s raw string: `RootIdentity::of` canonicalises
    // a local root (symlinks, `.`/`..`), which can differ byte-for-byte from
    // the verbatim spelling on a host where the tempdir root is itself a
    // symlink (e.g. macOS's `/tmp` -> `/private/tmp`). The bulk rows below
    // must match what self's row ACTUALLY carries, or none of them will ever
    // satisfy the ring predicate's identity equality.
    let self_root_identity: String =
        sqlx::query_scalar("SELECT result_root_identity FROM instances WHERE instance_id = $1")
            .bind(&self_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let run_suffix = jammi_test_utils::unique_suffix();
    let bulk_seed = |n: usize, tag: &str| {
        let root_val = self_root_identity.clone();
        let other_root_val = other_root().to_string();
        let tag = tag.to_string();
        let pool = pool.clone();
        let run_suffix = run_suffix.clone();
        async move {
            let now = chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.6fZ")
                .to_string();
            let ids: Vec<String> = (0..n)
                .map(|i| format!("bulk-{tag}-{run_suffix}-{i}"))
                .collect();
            let peer_addrs: Vec<String> = (0..n)
                .map(|i| format!("10.1.{}.{}:9000", (i / 250) % 250, i % 250))
                .collect();
            let roots: Vec<String> = (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        root_val.clone()
                    } else {
                        other_root_val.clone()
                    }
                })
                .collect();
            sqlx::query(
                "INSERT INTO instances \
                     (instance_id, label, host, peer_addr, result_root, result_root_identity, \
                      started_at, last_seen_at) \
                 SELECT id, NULL, NULL, addr, r, r, $4, $4 \
                 FROM UNNEST($1::text[], $2::text[], $3::text[]) AS t(id, addr, r)",
            )
            .bind(&ids)
            .bind(&peer_addrs)
            .bind(&roots)
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();
        }
    };

    let margin = instance_liveness_margin(LEASE);
    // Warm the catalog's own connection pool before timing: the FIRST query
    // any process issues pays a cold TCP+auth handshake this measurement is
    // not about (measured: ~3-4 ms of one-time connection setup, otherwise
    // indistinguishable in the 101-row number from genuine query cost).
    let _ = catalog.list_ring_members(&self_id, margin).await.unwrap();
    let mut cumulative_instances = 1usize; // self's own row, already seeded.
    for &n in &[100usize, 10_000usize] {
        bulk_seed(n, &format!("n{n}")).await;
        cumulative_instances += n;
        let start = std::time::Instant::now();
        let ring = catalog.list_ring_members(&self_id, margin).await.unwrap();
        let elapsed = start.elapsed();
        eprintln!(
            "RENDEZVOUS RV6: list_ring_members over {cumulative_instances} cumulative instances \
             rows (+{n} this batch) took {elapsed:?} ({} ring members)",
            ring.len()
        );
        assert!(
            ring.iter().any(|m| m.instance_id == self_id),
            "self must still be in the ring at {cumulative_instances} cumulative rows"
        );
        // Host-relative bound: a plain transfer of the same number of rows,
        // same two columns, through this test's own (warm) pool. Everything
        // the ring read pays beyond this is the predicate and the plan.
        let baseline_start = std::time::Instant::now();
        let baseline: Vec<(String, Option<String>)> = sqlx::query_as(
            "SELECT instance_id, peer_addr FROM instances WHERE peer_addr IS NOT NULL LIMIT $1",
        )
        .bind(ring.len() as i64)
        .fetch_all(&pool)
        .await
        .unwrap();
        let baseline_elapsed = baseline_start.elapsed();
        assert_eq!(
            baseline.len(),
            ring.len(),
            "the baseline transfers exactly the ring's row count"
        );
        let bound = baseline_elapsed * 4 + Duration::from_millis(2);
        eprintln!(
            "RENDEZVOUS RV6: plain transfer of {} rows took {baseline_elapsed:?}; bound {bound:?}",
            baseline.len()
        );
        assert!(
            elapsed <= bound,
            "RV6 bound: the ring read at {cumulative_instances} cumulative instances rows took \
             {elapsed:?}, more than four times (+2 ms) a plain transfer of the same {} rows on this \
             host ({baseline_elapsed:?}) — the predicate or the plan regressed, not the host",
            baseline.len()
        );
    }

    // A shared, persistent lane database (`gang_membership.rs`'s own
    // discipline) must not accumulate this test's 10k+ bulk rows forever:
    // clean up everything this run created, self included.
    sqlx::query("DELETE FROM instances WHERE instance_id LIKE $1")
        .bind(format!("bulk-%-{run_suffix}-%"))
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DELETE FROM instances WHERE instance_id = $1")
        .bind(&self_id)
        .execute(&pool)
        .await
        .unwrap();
}

#[test_case::test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_corrupted_peer_addr_is_a_typed_catalog_error(kind: BackendKind) {
    let (_dir, catalog) = catalog_on(kind).await;
    let self_id = format!("self-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &self_id, Some("10.0.0.1:9000"), Some(root())).await;
    let corrupt_id = format!("corrupt-{}", jammi_test_utils::unique_suffix());
    seed_instance(&catalog, &corrupt_id, Some("10.0.0.2:9000"), Some(root())).await;
    let corrupt_id_owned = corrupt_id.clone();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET peer_addr = $1 WHERE instance_id = $2",
                    &[
                        SqlValue::Text("not an addr"),
                        SqlValue::TextOwned(corrupt_id_owned),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
    let err = catalog
        .list_ring_members(&self_id, instance_liveness_margin(LEASE))
        .await
        .unwrap_err();
    assert!(
        matches!(err, jammi_db::error::JammiError::Catalog(_)),
        "a corrupted peer_addr row must be a typed Catalog error, never a panic: {err:?}"
    );
    // Clean up before any later test in a shared Postgres database reads it.
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM instances WHERE instance_id = $1",
                    &[SqlValue::TextOwned(corrupt_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}
