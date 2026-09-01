//! CLI integration tests for `jammi train`.
//!
//! `jammi train` is read-only (submission is SDK-only), so these tests can't
//! drive a job into existence through the CLI itself. They seed a
//! `training_jobs` row directly with a [`jammi_db::catalog::Catalog`] — and
//! they do it strictly BEFORE the server exists.
//!
//! The catalog is single-process (`docs/guide/src/catalog-and-broker.md`): the
//! server owns `catalog.db` for its whole lifetime, and a second process
//! opening that file while the server is live is out of contract — the backend
//! refuses it with an `SQLITE_BUSY`-class error, and before that refusal
//! existed the sharing silently corrupted the WAL. So each test builds its own
//! scratch directory, seeds it, waits for its own connections to actually
//! close, and only then hands the directory to
//! [`TestServer::spawn_with_scratch`]. Every assertion after that reads back
//! over the real wire through the CLI binary — the test process never reopens
//! the catalog while the server is up.
//!
//! Rows are seeded in a terminal (`completed`) status, never `queued`, so the
//! server's own background `TrainingWorker` — which claims exclusively
//! `WHERE status = 'queued'` — never mutates a fixture out from under a test.

use jammi_db::catalog::backend::{SqlNullType, SqlValue};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::{ModelTask, TxOptions};
use tempfile::TempDir;

use crate::server_harness::{await_catalog_released, TestServer};

/// One training-job fixture to seed before the server starts.
struct JobFixture<'a> {
    job_id: &'a str,
    metrics: Option<&'a str>,
    acceleration_report: Option<&'a str>,
}

/// Build a scratch directory carrying `fixtures` (all pointed at one freshly
/// registered base model), with every catalog connection of this process
/// closed before the directory is returned.
///
/// The catalog handle is confined to an inner scope and dropped there, then
/// [`await_catalog_released`] settles the close before the directory is
/// returned — dropping the handle alone does not close the file. The returned
/// `TempDir` is therefore safe to hand to
/// [`TestServer::spawn_with_scratch`], which re-checks the same invariant.
async fn seeded_scratch(base_model_id: &str, fixtures: &[JobFixture<'_>]) -> TempDir {
    let scratch = TempDir::new().expect("tempdir for server scratch");
    {
        let catalog = Catalog::open(scratch.path())
            .await
            .expect("open catalog on the not-yet-served scratch dir");
        register_model(&catalog, base_model_id).await;
        let base_ref = format!("{base_model_id}::1");
        for f in fixtures {
            seed_completed_job(&catalog, f, &base_ref).await;
        }
    }
    await_catalog_released(scratch.path()).await;
    scratch
}

/// Register the FK target model every seeded row's `base_model_id` points at.
async fn register_model(catalog: &Catalog, model_id: &str) {
    catalog
        .register_model(RegisterModelParams {
            model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .expect("register test model");
}

/// Directly seed a `training_jobs` row in a terminal (`completed`) status.
/// Mirrors `jammi-server`'s `grpc_training.rs::seed_training_job_row`.
async fn seed_completed_job(catalog: &Catalog, fixture: &JobFixture<'_>, base_model_id: &str) {
    let job_id = fixture.job_id.to_string();
    let base_model_id = base_model_id.to_string();
    let metrics = fixture.metrics.map(str::to_string);
    let acceleration_report = fixture.acceleration_report.map(str::to_string);

    catalog
        .backend_arc()
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO training_jobs \
                     (job_id, base_model_id, training_source, loss_type, hyperparams, status, \
                      kind, training_spec, tenant_id, metrics, acceleration_report, claimed_by, \
                      attempts, lease_expires_at) \
                     VALUES ($1, $2, 'seed.csv', 'contrastive', '{}', 'completed', \
                             'fine_tune', $3, $4, $5, $6, $7, 0, $8)",
                    &[
                        SqlValue::TextOwned(job_id),
                        SqlValue::TextOwned(base_model_id),
                        SqlValue::Null(SqlNullType::Text),
                        SqlValue::Null(SqlNullType::Text),
                        SqlValue::from(metrics),
                        SqlValue::from(acceleration_report),
                        SqlValue::Null(SqlNullType::Text),
                        SqlValue::Null(SqlNullType::Text),
                    ],
                )
                .await
            })
        })
        .await
        .expect("seed training_jobs row");
}

/// Control for the seed-before-spawn ordering the other tests in this module
/// depend on: a server spawned on a pre-seeded directory must actually serve
/// those rows.
///
/// Asserted through `train list`, the CLI's own catalog-backed listing, so the
/// proof runs over the wire rather than by reopening the catalog. Both seeded
/// ids appear (the server adopted the pre-seeded database, it did not create a
/// fresh empty one), and a never-seeded id is refused (the listing is a real
/// read of that database, not an unconditional echo).
#[tokio::test]
async fn cli_train_sees_rows_seeded_before_the_server_spawned() {
    let scratch = seeded_scratch(
        "cli-seed-order-base",
        &[
            JobFixture {
                job_id: "cli-seed-order-first",
                metrics: None,
                acceleration_report: None,
            },
            JobFixture {
                job_id: "cli-seed-order-second",
                metrics: None,
                acceleration_report: None,
            },
        ],
    )
    .await;
    let server = TestServer::spawn_with_scratch(scratch);

    let out = server
        .cli()
        .args(["train", "list"])
        .output()
        .expect("run train list");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for job_id in ["cli-seed-order-first", "cli-seed-order-second"] {
        assert!(
            stdout.contains(job_id),
            "a row seeded before spawn is missing from `train list` — the server did not \
             adopt the pre-seeded catalog:\n{stdout}"
        );
    }

    let missing = server
        .cli()
        .args(["train", "status", "cli-seed-order-never-seeded"])
        .output()
        .expect("run train status for an unseeded id");
    assert!(
        !missing.status.success(),
        "an id that was never seeded must not resolve — `train list` passing would then \
         prove nothing about the pre-seeded catalog"
    );
}

/// `jammi train status` renders the `acceleration_report_json` field under
/// its own clearly labeled section, printed verbatim like `metrics`, when the
/// catalog row carries a determined report.
#[tokio::test]
async fn cli_train_status_shows_acceleration_report_when_present() {
    let report = r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#;
    let scratch = seeded_scratch(
        "cli-acc-base",
        &[JobFixture {
            job_id: "cli-acc-present",
            metrics: Some(r#"{"final_loss":0.1}"#),
            acceleration_report: Some(report),
        }],
    )
    .await;
    let server = TestServer::spawn_with_scratch(scratch);

    let out = server
        .cli()
        .args(["train", "status", "cli-acc-present"])
        .output()
        .expect("run train status");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(&format!("acceleration_report: {report}")),
        "status output missing the acceleration_report section:\n{stdout}"
    );
    // The metrics section keeps rendering unaffected, same styling.
    assert!(
        stdout.contains(r#"metrics:  {"final_loss":0.1}"#),
        "status output missing the metrics section:\n{stdout}"
    );
}

/// A legacy row predating the `acceleration_report` column (SQL `NULL`) omits
/// the section entirely — no fabricated placeholder such as `none`.
#[tokio::test]
async fn cli_train_status_omits_acceleration_report_when_absent() {
    let scratch = seeded_scratch(
        "cli-acc-legacy-base",
        &[JobFixture {
            job_id: "cli-acc-legacy",
            metrics: None,
            acceleration_report: None,
        }],
    )
    .await;
    let server = TestServer::spawn_with_scratch(scratch);

    let out = server
        .cli()
        .args(["train", "status", "cli-acc-legacy"])
        .output()
        .expect("run train status");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("acceleration_report"),
        "a legacy row with no acceleration_report column value must not print \
         a fabricated section:\n{stdout}"
    );
    assert!(
        !stdout.contains("metrics:"),
        "a row with no metrics must not print a fabricated metrics section either:\n{stdout}"
    );
}
