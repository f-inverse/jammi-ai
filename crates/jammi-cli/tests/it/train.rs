//! CLI integration tests for `jammi train`.
//!
//! `jammi train` is read-only (submission is SDK-only), so these tests can't
//! drive a job into existence through the CLI itself. Instead they open a
//! second [`jammi_db::catalog::Catalog`] against the hermetic server's own
//! `JAMMI_ARTIFACT_DIR` (same technique
//! `jammi-server`'s `grpc_training.rs` esc-075 wire tests use) and seed a
//! `training_jobs` row directly, landing it in a terminal state so the
//! server's own background `TrainingWorker` never claims and mutates it out
//! from under the test. The CLI then reads that row back over the real wire.

use jammi_db::catalog::backend::{SqlNullType, SqlValue};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::{ModelTask, TxOptions};

use crate::server_harness::TestServer;

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

/// Directly seed a `training_jobs` row in a terminal (`completed`) status —
/// never `queued` — so the live server's background `TrainingWorker` (which
/// claims exclusively `WHERE status = 'queued'`) never races the test for it.
/// Mirrors `jammi-server`'s `grpc_training.rs::seed_training_job_row`.
#[allow(clippy::too_many_arguments)]
async fn seed_completed_job(
    catalog: &Catalog,
    job_id: &str,
    base_model_id: &str,
    metrics: Option<&str>,
    acceleration_report: Option<&str>,
) {
    let job_id = job_id.to_string();
    let base_model_id = base_model_id.to_string();
    let metrics = metrics.map(str::to_string);
    let acceleration_report = acceleration_report.map(str::to_string);

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

/// `jammi train status` renders the `acceleration_report_json` field under
/// its own clearly labeled section, printed verbatim like `metrics`, when the
/// catalog row carries a determined report.
#[tokio::test]
async fn cli_train_status_shows_acceleration_report_when_present() {
    let server = TestServer::spawn();
    let catalog = Catalog::open(server.artifact_dir())
        .await
        .expect("open catalog against the server's own artifact dir");
    register_model(&catalog, "cli-acc-base").await;

    let report = r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#;
    seed_completed_job(
        &catalog,
        "cli-acc-present",
        "cli-acc-base::1",
        Some(r#"{"final_loss":0.1}"#),
        Some(report),
    )
    .await;

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
    let server = TestServer::spawn();
    let catalog = Catalog::open(server.artifact_dir())
        .await
        .expect("open catalog against the server's own artifact dir");
    register_model(&catalog, "cli-acc-legacy-base").await;

    seed_completed_job(
        &catalog,
        "cli-acc-legacy",
        "cli-acc-legacy-base::1",
        None,
        None,
    )
    .await;

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
