//! esc-071 RED oracle (`closes_escape: esc-071`) — cross-session catalog read
//! visibility on ONE SQLite catalog file.
//!
//! Contract under test (`docs/guide/src/catalog-and-broker.md:90` "many readers
//! alongside one writer"; two-sessions-per-process supported per
//! `docs/guide/src/multi-tenant.md:17,228`): on one SQLite catalog file in one
//! process, a catalog read through ANY engine session must observe every write
//! another session has already committed — on that pooled connection's Nth
//! read, not only its first.
//!
//! The gate this closes: the only pre-existing two-sessions-one-file catalog
//! test (`tenant_scope.rs:65-125`) is tenant-DISJOINT, so a stale read passes it
//! vacuously; the single-pool read-after-write coverage
//! (`crates/jammi-ai/tests/it/fine_tune.rs:1078-1165`) never opens a second
//! pool at all (`backend_sqlite.rs:24-41` builds one pool per session).
//!
//! Oracle shape (verbatim from the row's `symptom_spec.observable`): two
//! `JammiSession`s built by `make_test_session(Sqlite, dir)` on ONE tempdir =>
//! two `SqliteBackend` pools on `<dir>/catalog.db`, BOTH live for the whole
//! test. Session B creates training job `j`, then over N = 4 rounds stamps a
//! round-distinct value (`metrics = {"round": i}`) through the catalog's own
//! write path; after each write session A calls `get_training_job("j")` and must
//! observe `round == i`.
//!
//! Failure per the row's `control`: any of (a) `round != i` — including the
//! exact-one-round lag `i - 1` the python wave reproduced, which the pairwise
//! distinct round values make impossible to match coincidentally; (b)
//! absent/not-found; (c) any `Err` at all (never swallowed, retried, or called
//! inconclusive); (d) panic/timeout. There is no sleep and no retry anywhere in
//! the assertion path — a value that is only correct after a delay is a
//! FAILURE, not a pass.
//!
//! Two non-vacuity arms run alongside the assertion, per the row: a SAME-POOL
//! control (B reads its own write each round — must be `round == i`, proving the
//! write itself landed) and a FRESH-POOL baseline (a brand-new session's FIRST
//! read after the same write — must be `round == i`, proving the value is
//! visible to a cold pool, i.e. the harness is not broken).
//!
//! Both sessions are unscoped (tenant `NULL`), never a tenant-disjoint pair:
//! disjoint tenants would make the assertion vacuous, which is exactly the hole
//! `tenant_scope.rs` leaves.
//!
//! Companion coverage: the reported lag's ACTUAL topology is not this one. The
//! python wave's writer was a raw CPython `sqlite3` connection — a second SQLite
//! *library instance* in the process, not a second engine pool — and the
//! sequential two-library equivalent of the loop below lives in
//! `esc_073_foreign_sqlite_library.rs`'s stale-read arm. Read the two together:
//! this file pins the SUPPORTED two-pool topology, that one pins the
//! out-of-contract two-library one.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::session::JammiSession;
use jammi_db::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

/// Number of round-distinct writes. Values are pairwise distinct (`1..=4`) so a
/// lag-by-one read can never coincidentally equal the expected round.
const ROUNDS: i64 = 4;

/// Hard deadline for any single catalog call. A hang (deadlock / starved write
/// lock) must surface as a FAILURE, never as a test-runner hang — this is a
/// failure detector, not a retry: nothing is re-attempted after it fires.
const OP_TIMEOUT: Duration = Duration::from_secs(30);

/// Await `fut`, failing the test on timeout with `what` in the message.
macro_rules! must {
    ($what:expr, $fut:expr) => {
        match tokio::time::timeout(OP_TIMEOUT, $fut).await {
            Ok(v) => v,
            Err(_) => panic!("esc-071: {} timed out after {OP_TIMEOUT:?}", $what),
        }
    };
}

/// Open a SQLite-backed session on `dir`, or skip (the Postgres arm of
/// `make_test_session` is the only `None` case, and this file is SQLite-only).
async fn session_on(dir: &Path) -> JammiSession {
    make_test_session(BackendKind::Sqlite, dir)
        .await
        .expect("sqlite session on the shared catalog dir")
}

/// Read job `job_id` through `catalog` and return the `round` field of its
/// metrics blob. Every non-observation is a panic, per the row's control: an
/// `Err`, a missing row, a `NULL`/unparseable metrics blob, or a blob with no
/// integer `round` — none of them are ever folded into "inconclusive".
async fn observed_round(catalog: &Catalog, job_id: &str, who: &str, round: i64) -> i64 {
    let record = must!(
        format!("{who} read of round {round}"),
        catalog.get_training_job(job_id)
    )
    .unwrap_or_else(|err| {
        panic!("esc-071: {who} read at round {round} failed: {err}");
    });

    let raw = record.metrics.unwrap_or_else(|| {
        panic!("esc-071: {who} read at round {round} saw metrics ABSENT (expected round {round})");
    });
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!("esc-071: {who} read at round {round} saw unparseable metrics {raw:?}: {err}");
    });
    parsed
        .get("round")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or_else(|| {
            panic!("esc-071: {who} read at round {round} saw metrics without an integer `round`: {raw:?}");
        })
}

/// A read through a second live pool on one catalog file must observe every
/// write the first pool has already committed — on the second read and after,
/// not only on the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn second_session_observes_every_committed_round() {
    let dir = tempdir().unwrap();

    // Two sessions on ONE tempdir => two SqliteBackend pools on
    // <dir>/catalog.db, both live for the whole test.
    let session_b = session_on(dir.path()).await;
    let session_a = session_on(dir.path()).await;
    let catalog_b = session_b.catalog();
    let catalog_a = session_a.catalog();

    // Fixture: a base model (FK target) and the job whose metrics carry the
    // round marker, both written through session B.
    must!(
        "register base model",
        catalog_b.register_model(RegisterModelParams {
            model_id: "base-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
    )
    .expect("register base model");

    must!(
        "create training job",
        catalog_b.create_training_job(CreateTrainingJobParams {
            job_id: "j",
            base_model_id: "base-model::1",
            training_source: "training_source",
            loss_type: "cosent",
            hyperparams: r#"{"lora_rank": 8}"#,
            kind: "fine_tune",
            training_spec: "{}",
        })
    )
    .expect("create training job");

    // The metrics write path is lease-guarded, so B claims the job first.
    let claimed = must!(
        "claim job",
        catalog_b.claim_next_training_job("worker-b", Duration::from_secs(600))
    )
    .expect("claim transaction")
    .expect("the queued job is claimable");
    assert_eq!(claimed.job_id, "j");

    // Warm session A's pool with a read BEFORE the first round, so the rounds
    // below are A's 2nd..5th reads — the row's defect is a stale value on the
    // 2nd+ read of an already-warm pooled connection, which a cold-pool-only
    // probe cannot see.
    let warm = must!("warm read", catalog_a.get_training_job("j")).expect("session A warm read");
    assert_eq!(
        warm.status, "running",
        "esc-071: session A's FIRST read must already observe B's committed claim"
    );

    for i in 1..=ROUNDS {
        let payload = format!(r#"{{"round": {i}}}"#);
        let landed = must!(
            format!("write round {i}"),
            catalog_b.mark_training_running("j", "worker-b", Some(&payload))
        )
        .unwrap_or_else(|err| panic!("esc-071: write of round {i} failed: {err}"));
        assert!(landed, "esc-071: write of round {i} matched no row");

        // ── Assertion: session A, second live pool, no sleep, no retry. ──
        let seen_a = observed_round(catalog_a, "j", "session A (2nd pool)", i).await;
        assert_eq!(
            seen_a,
            i,
            "esc-071: session A observed round {seen_a} on a read taken AFTER round {i} \
             committed through session B (a lag of {} round(s))",
            i - seen_a
        );

        // ── Same-pool control: the writer's own pool must see its write. ──
        let seen_b = observed_round(catalog_b, "j", "session B (writer's own pool)", i).await;
        assert_eq!(
            seen_b, i,
            "esc-071 CONTROL BROKEN: the writing session's own pool did not observe its \
             own committed round {i} (saw {seen_b}) — the write path, not cross-session \
             visibility, is at fault"
        );

        // ── Fresh-pool baseline: a cold pool's FIRST read must see it. ──
        let session_c = session_on(dir.path()).await;
        let seen_c =
            observed_round(session_c.catalog(), "j", "fresh session C (cold pool)", i).await;
        assert_eq!(
            seen_c, i,
            "esc-071 HARNESS BROKEN: a brand-new session's FIRST read did not observe \
             committed round {i} (saw {seen_c}) — the value is not on disk at all"
        );
        drop(session_c);
    }
}

// ── Mechanism probes ────────────────────────────────────────────────────────
//
// The oracle above is the row's `observable` verbatim. These probes widen it
// along the axes the python r3-repair wave actually touched, so that a GREEN
// oracle is a MEASURED absence of the defect on those axes rather than an
// unprobed one. Each probe carries the SAME control as the oracle: any `Err`,
// absence, or wrong round is a FAILURE, and there is no sleep and no retry.

/// Probe A — a stale value on a *warm pooled connection*, spread across the
/// whole pool. `SqlitePoolOptions::max_connections(8)` (`backend_sqlite.rs:35`)
/// means a session's reads can land on any of eight physical SQLite
/// connections; the oracle's sequential reads re-use one idle connection, so a
/// per-connection staleness could hide behind it. Here every one of A's
/// connections is warmed (a concurrent fan-out wider than the pool, which
/// forces them all open) BEFORE the write and read again after — so each round
/// is a 2nd+ read on every warm connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn probe_every_warm_pooled_connection_observes_the_write() {
    const FANOUT: usize = 16; // > max_connections(8): forces all eight open.

    let dir = tempdir().unwrap();
    let session_b = session_on(dir.path()).await;
    let session_a = session_on(dir.path()).await;
    let catalog_b = session_b.catalog();
    let catalog_a = session_a.catalog();

    seed_claimed_job(catalog_b).await;

    // Warm the pool: concurrent reads force sqlx to open every connection.
    fanout_rounds(catalog_a, FANOUT, 0).await;

    for i in 1..=ROUNDS {
        let payload = format!(r#"{{"round": {i}}}"#);
        let landed = must!(
            format!("write round {i}"),
            catalog_b.mark_training_running("j", "worker-b", Some(&payload))
        )
        .unwrap_or_else(|err| panic!("esc-071 probe A: write of round {i} failed: {err}"));
        assert!(landed, "esc-071 probe A: write of round {i} matched no row");

        let seen = fanout_rounds(catalog_a, FANOUT, i).await;
        for (slot, round) in seen.iter().enumerate() {
            assert_eq!(
                *round, i,
                "esc-071 probe A: warm pooled read #{slot} observed round {round} after \
                 round {i} committed through the other session's pool"
            );
        }
    }
}

/// Probe B — a read taken while the SAME session holds a long-lived read
/// transaction open on another of its pooled connections. A held `BEGIN
/// DEFERRED` pins a WAL snapshot on ITS connection (correct SQLite semantics);
/// what this probe answers is whether that snapshot leaks to the session's
/// OTHER pooled connections, which would be exactly the reported lag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn probe_read_beside_a_long_lived_read_transaction_observes_the_write() {
    use jammi_db::catalog::backend::TxOptions;

    let dir = tempdir().unwrap();
    let session_b = session_on(dir.path()).await;
    let session_a = session_on(dir.path()).await;
    let catalog_b = session_b.catalog();
    let catalog_a = session_a.catalog();

    seed_claimed_job(catalog_b).await;
    let warm = must!("warm read", catalog_a.get_training_job("j")).expect("session A warm read");
    assert_eq!(warm.status, "running");

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    let backend_a = catalog_a.backend_arc();
    let held = tokio::spawn(async move {
        backend_a
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    Box::pin(async move {
                        // Take the snapshot, then hold this transaction open
                        // across every write below.
                        let _ = tx
                            .query_opt(
                                "SELECT job_id FROM training_jobs WHERE job_id = 'j'",
                                &[],
                                |r| r.get::<String>("job_id"),
                            )
                            .await?;
                        let _ = ready_tx.send(());
                        let _ = done_rx.await;
                        Ok(())
                    })
                },
            )
            .await
    });
    ready_rx
        .await
        .expect("held read transaction reached its snapshot");

    for i in 1..=ROUNDS {
        let payload = format!(r#"{{"round": {i}}}"#);
        let landed = must!(
            format!("write round {i}"),
            catalog_b.mark_training_running("j", "worker-b", Some(&payload))
        )
        .unwrap_or_else(|err| panic!("esc-071 probe B: write of round {i} failed: {err}"));
        assert!(landed, "esc-071 probe B: write of round {i} matched no row");

        // A's read on a DIFFERENT pooled connection, while A's own long-lived
        // read transaction pins an older snapshot elsewhere in the same pool.
        let seen = observed_round(catalog_a, "j", "session A beside a held read tx", i).await;
        assert_eq!(
            seen, i,
            "esc-071 probe B: a read beside this session's own long-lived read transaction \
             observed round {seen} after round {i} committed"
        );
    }

    let _ = done_tx.send(());
    held.await
        .expect("held transaction task")
        .expect("held read transaction");
}

/// Probe C — the python wave's actual write shape: the value is committed by a
/// connection OUTSIDE either session's pool (`test_conformance.py:968-978`
/// commits it through a raw `sqlite3` connection it then closes) while the
/// reading session's pool stays warm. This is the closest in-Rust analogue: a
/// standalone `sqlx` connection opened directly on `<dir>/catalog.db` for one
/// `UPDATE`, then closed — never a member of either pool. (It is still the same
/// SQLite *library instance*; the two-library case is esc-073's harness.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn probe_foreign_connection_write_is_observed_by_a_warm_pool() {
    use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
    use sqlx::{ConnectOptions, Connection};

    let dir = tempdir().unwrap();
    let session_b = session_on(dir.path()).await;
    let session_a = session_on(dir.path()).await;
    let catalog_b = session_b.catalog();
    let catalog_a = session_a.catalog();

    seed_claimed_job(catalog_b).await;
    let warm = must!("warm read", catalog_a.get_training_job("j")).expect("session A warm read");
    assert_eq!(warm.status, "running");

    let db_path = dir.path().join("catalog.db");
    for i in 1..=ROUNDS {
        // Foreign writer: opened, committed, closed — once per round.
        let mut foreign = SqliteConnectOptions::new()
            .filename(&db_path)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal)
            .connect()
            .await
            .expect("foreign connection on the shared catalog file");
        let payload = format!(r#"{{"round": {i}}}"#);
        sqlx::query("UPDATE training_jobs SET metrics = ?1 WHERE job_id = 'j'")
            .bind(&payload)
            .execute(&mut foreign)
            .await
            .unwrap_or_else(|err| panic!("esc-071 probe C: foreign write of round {i}: {err}"));
        foreign.close().await.expect("close foreign connection");

        let seen = observed_round(catalog_a, "j", "session A (warm pool)", i).await;
        assert_eq!(
            seen, i,
            "esc-071 probe C: warm pool observed round {seen} after a foreign connection \
             committed round {i} to the same catalog file"
        );
        let seen_b = observed_round(catalog_b, "j", "session B (warm pool)", i).await;
        assert_eq!(
            seen_b, i,
            "esc-071 probe C: the other warm pool observed round {seen_b} after a foreign \
             connection committed round {i}"
        );
    }
}

/// Register the base model, create job `j`, and claim it — the fixture every
/// probe shares, written entirely through `catalog`.
async fn seed_claimed_job(catalog: &Catalog) {
    must!(
        "register base model",
        catalog.register_model(RegisterModelParams {
            model_id: "base-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
    )
    .expect("register base model");
    must!(
        "create training job",
        catalog.create_training_job(CreateTrainingJobParams {
            job_id: "j",
            base_model_id: "base-model::1",
            training_source: "training_source",
            loss_type: "cosent",
            hyperparams: r#"{"lora_rank": 8}"#,
            kind: "fine_tune",
            training_spec: "{}",
        })
    )
    .expect("create training job");
    let claimed = must!(
        "claim job",
        catalog.claim_next_training_job("worker-b", Duration::from_secs(600))
    )
    .expect("claim transaction")
    .expect("the queued job is claimable");
    assert_eq!(claimed.job_id, "j");
}

/// Issue `n` concurrent reads through `catalog` and return each one's observed
/// round. `expect_round == 0` means "warm-up only": the metrics column is still
/// absent, so the reads are made for their pool-warming effect and every
/// returned round is zero (asserted, so a warm-up that accidentally races a
/// write cannot pass silently).
async fn fanout_rounds(catalog: &Arc<Catalog>, n: usize, expect_round: i64) -> Vec<i64> {
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let catalog = Arc::clone(catalog);
        handles.push(tokio::spawn(async move {
            let record = tokio::time::timeout(OP_TIMEOUT, catalog.get_training_job("j"))
                .await
                .expect("fan-out read timed out")
                .expect("fan-out read failed");
            match record.metrics {
                None => 0,
                Some(raw) => serde_json::from_str::<serde_json::Value>(&raw)
                    .expect("metrics parse")
                    .get("round")
                    .and_then(serde_json::Value::as_i64)
                    .expect("integer round"),
            }
        }));
    }
    let mut out = Vec::with_capacity(n);
    for h in handles {
        out.push(h.await.expect("fan-out task panicked"));
    }
    if expect_round == 0 {
        assert!(
            out.iter().all(|r| *r == 0),
            "esc-071 probe: warm-up fan-out saw metrics before the first round was written"
        );
    }
    out
}
