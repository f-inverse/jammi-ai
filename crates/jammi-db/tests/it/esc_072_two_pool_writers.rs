//! esc-072 RED oracle — transaction integrity with TWO live pools on one
//! SQLite catalog file.
//!
//! Contract under test: two live sessions in one process on one `catalog.db`,
//! each running concurrent catalog write loops of the training worker's
//! claim/heartbeat/reclaim shape, complete every transaction cleanly — WAL's
//! "many readers alongside one writer" with `sqlx`'s `busy_timeout`
//! (`backend_sqlite.rs:29`) absorbing the contention.
//!
//! The gate this closes: the leaked-`BEGIN` defense
//! (`backend_sqlite.rs:96-110`, the detached-task begin) is regression-tested
//! for a SINGLE pool only (`concurrent_writers.rs:249-259`). Two pools on one
//! file is a topology `docs/guide/src/multi-tenant.md` permits and nothing
//! covers, so a recurrence of `InvalidSavePointStatement` there would mean the
//! defense is unproven, not proven, for the shipped topology.
//!
//! Oracle shape (from the row's `symptom_spec.observable`): two `JammiSession`s
//! on ONE tempdir's `catalog.db`, each driving its own embedded
//! training-worker-shaped write loop (claim → heartbeat → stamp metrics, plus a
//! reclaim sweeper) concurrently, for both a wall-clock floor and an op-count
//! floor at least as large as the single-pool workload
//! `concurrent_writers.rs` drives.
//!
//! Failure per the row's `control`: ANY surfaced `sqlx`
//! `InvalidSavePointStatement` / "non-zero transaction depth", any "disk I/O
//! error", any unabsorbed "database is locked", and any deadlock/timeout in
//! either loop. Errors are collected with their exact strings and reported
//! together rather than swallowed, so a RED run names the failure class instead
//! of merely failing.
//!
//! Non-vacuity: the SAME workload is also run against a SINGLE pool
//! ([`single_pool_worker_loops_stay_clean`]) as the workload-validity control —
//! if the one-pool arm is dirty, the workload is at fault, not the topology.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::session::JammiSession;
use jammi_db::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

/// Queued jobs seeded before the loops start. Enough that claimants contend on
/// the claim `UPDATE … WHERE status = 'queued'` without every worker starving.
const JOBS: usize = 16;

/// Worker tasks per session. Both arms run the same TOTAL number of workers
/// (`WORKERS_PER_SESSION * 2`), so the two-pool arm and the single-pool control
/// put the same concurrency on the file and differ only in pool count.
const WORKERS_PER_SESSION: usize = 4;

/// Wall-clock floor, per the row ("≥10s").
const MIN_WALL: Duration = Duration::from_secs(10);

/// Op-count floor. `concurrent_writers.rs` drives 8 × 60 = 480 write
/// transactions; every catalog call below is at least one transaction, so this
/// floor is comfortably above the single-pool workload the row calls the
/// baseline.
const MIN_OPS: u64 = 2_000;

/// Lease length. Short enough that the reclaim sweeper actually finds expired
/// leases and recycles jobs back to `queued`, keeping the claim path hot for
/// the whole run instead of draining after one pass.
const LEASE: Duration = Duration::from_millis(80);

/// Attempts ceiling for the reclaim sweep. Deliberately enormous: a job must
/// never exhaust its attempts and settle into `failed`, or the workload drains
/// itself and the op floor is met by idle claims.
const MAX_ATTEMPTS: u32 = 1_000_000;

/// Hard ceiling on the whole run. Exceeding it is a deadlock FAILURE, not a
/// retry — nothing is re-attempted after it fires.
const RUN_CEILING: Duration = Duration::from_secs(180);

/// One observed error, kept verbatim so a RED run reports the exact string the
/// row's control enumerates rather than a paraphrase.
#[derive(Debug)]
struct Observed {
    who: String,
    op: &'static str,
    message: String,
}

impl Observed {
    /// The row's failure classes, matched case-insensitively on the rendered
    /// error. Any error at all is a failure; this only names which class.
    fn class(&self) -> &'static str {
        let m = self.message.to_ascii_lowercase();
        if m.contains("invalidsavepointstatement") || m.contains("non-zero transaction depth") {
            "InvalidSavePointStatement / non-zero transaction depth"
        } else if m.contains("disk i/o error") {
            "disk I/O error"
        } else if m.contains("database is locked") {
            "unabsorbed 'database is locked'"
        } else {
            "other (still a failure)"
        }
    }
}

/// Shared run state: the collected errors, the op counter, and the stop flag.
#[derive(Default)]
struct Run {
    errors: Mutex<Vec<Observed>>,
    ops: AtomicU64,
    stop: AtomicBool,
}

impl Run {
    fn record(&self, who: &str, op: &'static str, message: String) {
        self.errors.lock().unwrap().push(Observed {
            who: who.to_string(),
            op,
            message,
        });
        // Stop early on the first error: the RED evidence is the error, and
        // continuing only floods the log.
        self.stop.store(true, Ordering::SeqCst);
    }

    fn tick(&self) -> u64 {
        self.ops.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn done(&self, started: Instant) -> bool {
        self.stop.load(Ordering::SeqCst)
            || (started.elapsed() >= MIN_WALL && self.ops.load(Ordering::Relaxed) >= MIN_OPS)
    }
}

async fn session_on(dir: &Path) -> JammiSession {
    make_test_session(BackendKind::Sqlite, dir)
        .await
        .expect("sqlite session on the shared catalog dir")
}

/// Seed the base model and `JOBS` queued jobs through `catalog`.
async fn seed(catalog: &Catalog) {
    catalog
        .register_model(RegisterModelParams {
            model_id: "base-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .expect("register base model");
    for n in 0..JOBS {
        catalog
            .create_training_job(CreateTrainingJobParams {
                job_id: &format!("job-{n}"),
                base_model_id: "base-model::1",
                training_source: "training_source",
                loss_type: "cosent",
                hyperparams: r#"{"lora_rank": 8}"#,
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .expect("create training job");
    }
}

/// The training-worker-shaped write loop: claim → heartbeat → stamp metrics,
/// falling back to a reclaim sweep when nothing is claimable. Every catalog
/// call's `Err` is recorded verbatim (never swallowed, never retried).
async fn worker_loop(catalog: Arc<Catalog>, run: Arc<Run>, who: String, started: Instant) {
    let mut spin: u64 = 0;
    while !run.done(started) {
        run.tick();
        let claimed = match catalog.claim_next_training_job(&who, LEASE).await {
            Ok(c) => c,
            Err(err) => {
                run.record(&who, "claim_next_training_job", err.to_string());
                return;
            }
        };

        match claimed {
            Some(job) => {
                run.tick();
                if let Err(err) = catalog
                    .heartbeat_training_job(&job.job_id, &who, LEASE)
                    .await
                {
                    run.record(&who, "heartbeat_training_job", err.to_string());
                    return;
                }
                run.tick();
                let metrics = format!(r#"{{"who": "{who}", "spin": {spin}}}"#);
                if let Err(err) = catalog
                    .mark_training_running(&job.job_id, &who, Some(&metrics))
                    .await
                {
                    run.record(&who, "mark_training_running", err.to_string());
                    return;
                }
            }
            None => {
                // Nothing claimable — sweep expired leases back to `queued`,
                // which is exactly what the worker's poll loop does.
                run.tick();
                if let Err(err) = catalog.reclaim_expired_training_jobs(MAX_ATTEMPTS).await {
                    run.record(&who, "reclaim_expired_training_jobs", err.to_string());
                    return;
                }
                tokio::task::yield_now().await;
            }
        }
        spin += 1;
    }
}

/// Drive `workers` against each catalog in `catalogs` concurrently until both
/// floors are met, then assert no error of any class was observed.
async fn drive(catalogs: Vec<Arc<Catalog>>, arm: &str) {
    let run = Arc::new(Run::default());
    let started = Instant::now();

    let mut handles = Vec::new();
    for (pool_ix, catalog) in catalogs.iter().enumerate() {
        for w in 0..WORKERS_PER_SESSION {
            let catalog = Arc::clone(catalog);
            let run = Arc::clone(&run);
            let who = format!("pool{pool_ix}-worker{w}");
            handles.push(tokio::spawn(worker_loop(catalog, run, who, started)));
        }
    }

    for (ix, handle) in handles.into_iter().enumerate() {
        match tokio::time::timeout(RUN_CEILING.saturating_sub(started.elapsed()), handle).await {
            Ok(joined) => joined.unwrap_or_else(|join| {
                panic!("esc-072 [{arm}]: worker task {ix} panicked: {join}");
            }),
            Err(_) => panic!(
                "esc-072 [{arm}]: worker task {ix} did not finish within {RUN_CEILING:?} — \
                 deadlocked or starved on the catalog write lock"
            ),
        }
    }

    let errors = run.errors.lock().unwrap();
    let ops = run.ops.load(Ordering::Relaxed);
    let wall = started.elapsed();

    assert!(
        errors.is_empty(),
        "esc-072 [{arm}]: {} catalog transaction failure(s) across {ops} ops in {wall:?}:\n{}",
        errors.len(),
        errors
            .iter()
            .map(|e| format!("  [{}] {} in {}: {}", e.class(), e.who, e.op, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Floors are asserted only on a clean run: an early stop is legitimate
    // exactly when an error was recorded, and that case already failed above.
    assert!(
        ops >= MIN_OPS,
        "esc-072 [{arm}]: only {ops} ops completed (floor {MIN_OPS}) — the workload did not \
         reach the single-pool baseline's volume"
    );
    assert!(
        wall >= MIN_WALL,
        "esc-072 [{arm}]: run lasted only {wall:?} (floor {MIN_WALL:?})"
    );
}

/// TWO live pools on one catalog file, each driving its own worker-shaped write
/// loop concurrently. Every transaction must complete cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_pool_worker_loops_stay_clean() {
    let dir = tempdir().unwrap();
    let session_a = session_on(dir.path()).await;
    let session_b = session_on(dir.path()).await;
    seed(session_a.catalog()).await;

    drive(
        vec![
            Arc::clone(session_a.catalog()),
            Arc::clone(session_b.catalog()),
        ],
        "two pools",
    )
    .await;
}

/// Workload-validity control: the SAME loops, the same total worker count, one
/// pool. If this arm is dirty the workload itself is at fault and the two-pool
/// arm proves nothing about the topology.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_pool_worker_loops_stay_clean() {
    let dir = tempdir().unwrap();
    let session = session_on(dir.path()).await;
    seed(session.catalog()).await;

    let catalog = Arc::clone(session.catalog());
    drive(vec![Arc::clone(&catalog), catalog], "single pool").await;
}
