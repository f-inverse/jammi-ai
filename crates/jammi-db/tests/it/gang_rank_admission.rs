//! `Catalog::get_job_for_rank` (I-GANG's row predicate,
//! `docs/rigor/contracts/feat_500-C-U5a-1.md` § A1) and
//! `Catalog::fill_training_set_identity` (the training-set identity
//! write-once CAS). `Catalog::fresh_instance`'s own
//! tests live in `gang_instance_freshness.rs`. Most tests run on a fresh
//! SQLite catalog (a private tempdir per test; the gang admission surface has
//! no Postgres-only behaviour these primitives need to exercise beyond what
//! `jobs_queue.rs` already covers for the shared lease/reclaim machinery),
//! except `world_size` decoding (a targeted JSON field read with no
//! backend-specific SQL of its own, but still worth the same
//! `test_case`-parameterized sqlite/postgres shape `migrations.rs` uses, per
//! the round-2 fold) — those cases also run a `::postgres` arm gated by
//! `live-postgres-tests`, skipping (never failing) when `JAMMI_TEST_PG_URL`
//! is unset.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::catalog::jobs_repo::{SubmitJobParams, TrainingSetFillOutcome, WorldSizeFact};
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

/// The row's `status`/`claimed_by`/`attempts`/`lease_live` mirror a genuine
/// claim exactly, and `lease_live` is `true` for a freshly-claimed lease.
#[tokio::test]
async fn get_job_for_rank_reflects_a_live_claim() {
    let (_dir, catalog) = base_catalog().await;
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
    assert!(row.lease_live, "a freshly-claimed 30s lease must be live");
    assert!(
        row.remaining > Duration::from_secs(20) && row.remaining <= Duration::from_secs(30),
        "remaining must be close to the freshly-stamped 30s window, got {:?}",
        row.remaining
    );
    assert_eq!(row.training_set_ref, None);
    assert_eq!(row.training_set_location, None);
}

/// A lease forced to `NULL` (the state both reclaim arms leave behind, and
/// migration invariant `lease.rs` documents: `NULL` means remaining 0, never
/// live-by-default) reads back `lease_live == false`, `remaining ==
/// Duration::ZERO` — never a panic, never a "live" default.
#[tokio::test]
async fn get_job_for_rank_treats_a_null_lease_as_not_live() {
    let (_dir, catalog) = base_catalog().await;
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
    assert!(
        !row.lease_live,
        "a NULL lease column must never read as live"
    );
    assert_eq!(row.remaining, Duration::ZERO);
}

/// An expired (past) lease reads back `lease_live == false` with zero
/// remaining — the boundary the `< now()` / `< $now` clause names.
#[tokio::test]
async fn get_job_for_rank_treats_an_expired_lease_as_not_live() {
    let (_dir, catalog) = base_catalog().await;
    catalog.submit_job(job_params("job-3")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_millis(1))
        .await
        .unwrap()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let row = catalog.get_job_for_rank("job-3").await.unwrap().unwrap();
    assert!(!row.lease_live, "an expired lease must never read as live");
    assert_eq!(row.remaining, Duration::ZERO);
}

/// The pair round-trips exactly once filled.
#[tokio::test]
async fn get_job_for_rank_returns_the_filled_pair() {
    let (_dir, catalog) = base_catalog().await;
    catalog.submit_job(job_params("job-4")).await.unwrap();
    catalog
        .claim_next("coord-1", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .unwrap();
    let outcome = catalog
        .fill_training_set_identity("job-4", "coord-1", 1, "digest-x", "table-y")
        .await
        .unwrap();
    assert_eq!(outcome, TrainingSetFillOutcome::Filled);

    let row = catalog.get_job_for_rank("job-4").await.unwrap().unwrap();
    assert_eq!(row.training_set_ref.as_deref(), Some("digest-x"));
    assert_eq!(row.training_set_location.as_deref(), Some("table-y"));
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
    let Some((_dir, catalog)) = base_catalog_kind(kind).await else {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    };
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
    let Some((_dir, catalog)) = base_catalog_kind(kind).await else {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    };
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
    assert_eq!(row.training_set_ref, None);
    assert_eq!(row.training_set_location, None);
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
    let Some((_dir, catalog)) = base_catalog_kind(kind).await else {
        eprintln!("skipping postgres: JAMMI_TEST_PG_URL unset");
        return;
    };
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
