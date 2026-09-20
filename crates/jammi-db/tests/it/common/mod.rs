pub use jammi_test_utils::*;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use jammi_db::catalog::artifact_repo::{ArtifactRef, StagedArtifact};
use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::{ModelRow, ProducedModel, SubmitJobParams};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::ResultStore;

/// A migrated catalog on `kind`: the SQLite file under `dir`, or the shared
/// live Postgres.
pub async fn fresh_catalog(kind: BackendKind, dir: &std::path::Path) -> Arc<Catalog> {
    let backend = open_backend(kind, dir).await;
    backend.migrate().await.unwrap();
    Arc::new(Catalog::from_backend(backend))
}

/// The catalog of a fresh test session on `kind`, with the tempdir that
/// holds its artifacts (and, on SQLite, the catalog file) — the caller keeps
/// the dir alive for as long as it uses the catalog.
pub async fn catalog_on(kind: BackendKind) -> (tempfile::TempDir, Arc<Catalog>) {
    let dir = tempfile::tempdir().unwrap();
    let session = make_test_session(kind, dir.path()).await;
    (dir, Arc::clone(session.catalog()))
}

/// Empty the job queue, the worker/instance registries and the model-artifact
/// state. The Postgres lane runs every test against one shared database, so a
/// test that counts or claims jobs must start from an empty queue, and a test
/// that probes for a reusable artifact must start with none published — a
/// definition another test published under the same hash is a legitimate hit.
/// On SQLite (a fresh catalog per test) it is a no-op kept for one code path.
///
/// A `models` row that references an artifact goes before the artifact it
/// holds (`artifact_prefix` is `ON DELETE RESTRICT`); base models stay.
pub async fn reset_shared_catalog(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("DELETE FROM jobs", &[]).await?;
                tx.execute("DELETE FROM models WHERE artifact_prefix IS NOT NULL", &[])
                    .await?;
                tx.execute("DELETE FROM model_artifacts", &[]).await?;
                tx.execute("DELETE FROM workers", &[]).await?;
                tx.execute("DELETE FROM instances", &[]).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

/// The model id [`register_base_model`] registers: the base every queued
/// fine-tune job in these tests names.
pub const BASE_MODEL_ID: &str = "q-base";

/// Register [`BASE_MODEL_ID`] as a text-embedding model, so a job naming it as
/// its base satisfies the catalog's foreign key.
pub async fn register_base_model(catalog: &Catalog) {
    catalog
        .register_model(RegisterModelParams {
            model_id: BASE_MODEL_ID,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            external_location: None,
            config_json: None,
        })
        .await
        .unwrap();
}

/// The job kind every model-artifact test queues.
pub const FINE_TUNE_KIND: &str = "fine_tune";

/// A LoRA adapter bundle's two files, with bytes distinguishable by `tag`.
pub fn adapter_files(tag: &str) -> Vec<(String, Bytes)> {
    vec![
        (
            "adapter.safetensors".to_string(),
            Bytes::from(format!("weights:{tag}")),
        ),
        (
            "adapter_config.json".to_string(),
            Bytes::from(format!("{{\"r\":8,\"tag\":\"{tag}\"}}")),
        ),
    ]
}

/// A result store on a `file://` root under `dir`, over `catalog`.
pub fn store_over(dir: &Path, catalog: &Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, Arc::clone(catalog), AnnIndexConfig::default()).unwrap()
}

/// The local directory a `file://` artifact's bundle lives in.
pub fn bundle_dir(artifact: &ArtifactRef) -> PathBuf {
    PathBuf::from(artifact.url().as_str().strip_prefix("file://").unwrap())
}

/// The regular files directly inside `dir`, sorted; empty when `dir` is gone.
pub fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .map(|e| e.unwrap())
            .filter(|e| e.file_type().unwrap().is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
}

/// Submit a fresh fine-tune job over [`BASE_MODEL_ID`] and claim it as
/// `worker`: a job that is genuinely `running` the returned attempt. The id
/// is a v4 UUID, the shape a reconcile pass attributes under `models/`; the
/// output model is `output_model_id`, or the fine-tune default
/// `jammi:fine-tuned:{job_id}`.
pub async fn running_fine_tune_job(
    catalog: &Catalog,
    worker: &str,
    output_model_id: Option<&str>,
) -> (String, u32) {
    let job_id = uuid::Uuid::new_v4().to_string();
    let default_output = format!("jammi:fine-tuned:{job_id}");
    let output_model_id = Some(output_model_id.unwrap_or(&default_output));
    catalog
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: FINE_TUNE_KIND,
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some("q-base::1"),
            output_model_id,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next(worker, &[FINE_TUNE_KIND], Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("a queued job is claimable");
    assert_eq!(claimed.job_id, job_id, "the queue held only this job");
    (claimed.job_id, claimed.attempts)
}

/// The `models` row a fine-tune's finalize writes for `name`, attached to
/// `artifact`.
pub fn fine_tuned_model(name: &str, artifact: StagedArtifact) -> ProducedModel<'_> {
    ProducedModel {
        row: ModelRow {
            model_id: name,
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some(BASE_MODEL_ID),
            config_json: None,
        },
        artifact,
        materialization: None,
    }
}

/// Backdate the mtime of every regular file directly under `dir` by `by` —
/// lets a test manufacture an "already past grace" object deterministically,
/// with no real-time sleep.
pub fn backdate_dir(dir: &Path, by: Duration) {
    let target = std::time::SystemTime::now() - by;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::File::options()
                .write(true)
                .open(entry.path())
                .unwrap()
                .set_modified(target)
                .unwrap();
        }
    }
}

/// Move `artifact`'s row `by` into the past on the catalog's grace clock, so
/// a reconcile pass sees it as aged without a real-time wait.
pub async fn backdate_artifact(catalog: &Catalog, artifact: &ArtifactRef, by: Duration) {
    let stamp = (chrono::Utc::now() - chrono::Duration::from_std(by).unwrap())
        .format("%Y-%m-%dT%H:%M:%S%.6fZ")
        .to_string();
    let prefix = artifact.url().as_str().to_string();
    let updated = catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE model_artifacts SET created_at = $1 WHERE prefix = $2",
                    &[SqlValue::TextOwned(stamp), SqlValue::TextOwned(prefix)],
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(updated, 1, "the artifact row exists");
}

/// A test session on `kind` whose artifact dir is never deleted: the SQLite
/// arm's catalog file must outlive every handle the test gives away, and the
/// process exits shortly after the test.
pub async fn kept_dir_session(kind: BackendKind) -> jammi_db::session::JammiSession {
    let dir = tempfile::tempdir().unwrap().keep();
    make_test_session(kind, &dir).await
}

/// A fresh test session on `kind` (artifacts under `dir`) and its catalog,
/// with an empty queue ([`reset_shared_catalog`]) and the base model registered
/// ([`register_base_model`]) — the starting state of every job-queue test.
pub async fn queue_session(
    kind: BackendKind,
    dir: &std::path::Path,
) -> (jammi_db::session::JammiSession, Arc<Catalog>) {
    let session = make_test_session(kind, dir).await;
    let catalog = Arc::clone(session.catalog());
    reset_shared_catalog(&catalog).await;
    register_base_model(&catalog).await;
    (session, catalog)
}

/// Start a [`jammi_db::catalog::lease_keeper::LeaseKeeper`] whose connect
/// factory reopens a FRESH `Catalog` on `backend` every time it is invoked
/// (from inside the keeper thread's own runtime): the on-disk SQLite catalog
/// under `dir`, or the shared Postgres test database. Called only after the
/// test's own session opened on the same backend.
pub async fn keeper_for_backend(
    backend: jammi_db::catalog::backend::BackendKind,
    dir: std::path::PathBuf,
    intervals: jammi_db::catalog::lease::LeaseIntervals,
) -> std::sync::Arc<jammi_db::catalog::lease_keeper::LeaseKeeper> {
    use jammi_db::catalog::backend::{BackendImpl, BackendKind};
    use jammi_db::catalog::backend_postgres::PostgresBackend;
    use jammi_db::catalog::lease_keeper::LeaseKeeper;
    use jammi_db::catalog::Catalog;

    let url = match backend {
        BackendKind::Sqlite => None,
        BackendKind::Postgres => Some(postgres_url()),
    };
    LeaseKeeper::start(
        move || {
            let dir = dir.clone();
            let url = url.clone();
            Box::pin(async move {
                match url {
                    None => Catalog::open(&dir).await,
                    Some(url) => {
                        let pg = PostgresBackend::open_with_options(&url, 8, None).await?;
                        Ok(Catalog::from_backend(BackendImpl::Postgres(pg)))
                    }
                }
            })
        },
        intervals,
    )
    .await
    .expect("the keeper connects within the lease window")
}

/// The environment variable a crash-recovery child test reads its artifact
/// directory from.
#[cfg(feature = "test-hooks")]
pub const ARTIFACT_DIR_ENV: &str = "JAMMI_TEST_ARTIFACT_DIR";

/// Runs child test `path` on artifact directory `dir` with the test-hook
/// checkpoint variable `checkpoint.0` set to `checkpoint.1`, waits until the
/// hook parks it, and `SIGKILL`s it.
///
/// The hook writes the ready file only from inside the checkpoint, so reaching
/// the kill proves the child stopped exactly there, mid-transaction.
///
/// # Panics
/// When the child exits, or does not park within 30 s.
#[cfg(feature = "test-hooks")]
pub async fn kill_child_at_checkpoint(path: &str, dir: &std::path::Path, checkpoint: (&str, &str)) {
    use jammi_db::store::mutable::test_hook::READY_FILE_ENV;
    use std::time::{Duration, Instant};

    let ready_file = dir.join("ready");
    let mut child = tokio::process::Command::from(jammi_test_resources::child_test(path))
        .env(ARTIFACT_DIR_ENV, dir)
        .env(READY_FILE_ENV, &ready_file)
        .env(checkpoint.0, checkpoint.1)
        .spawn()
        .expect("spawn the child test process");

    // 30 s covers a cold runner's first spawn; a warm one parks in well under a second.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !tokio::fs::try_exists(&ready_file)
        .await
        .expect("check for the ready file")
    {
        if let Some(status) = child.try_wait().expect("poll the child") {
            panic!(
                "{path} exited before its {} checkpoint: {status}",
                checkpoint.1
            );
        }
        if Instant::now() > deadline {
            child.kill().await.expect("SIGKILL the stalled child");
            panic!(
                "{path} never reached its {} checkpoint within 30 s",
                checkpoint.1
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    child.kill().await.expect("SIGKILL the parked child");
}
