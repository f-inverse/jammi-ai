pub use jammi_test_utils::*;

use std::sync::Arc;

use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;

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

/// Empty the job queue and the worker/instance registries. The Postgres lane
/// runs every test against one shared database, so a queue test that counts or
/// claims jobs must start from an empty queue; on SQLite (a fresh catalog per
/// test) it is a no-op kept for one code path.
pub async fn reset_queue(catalog: &Catalog) {
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
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
}

/// A test session on `kind` whose artifact dir is never deleted: the SQLite
/// arm's catalog file must outlive every handle the test gives away, and the
/// process exits shortly after the test.
pub async fn kept_dir_session(kind: BackendKind) -> jammi_db::session::JammiSession {
    let dir = tempfile::tempdir().unwrap().keep();
    make_test_session(kind, &dir).await
}

/// A fresh test session on `kind` (artifacts under `dir`) and its catalog,
/// with an empty queue ([`reset_queue`]) and the base model registered
/// ([`register_base_model`]) — the starting state of every job-queue test.
pub async fn queue_session(
    kind: BackendKind,
    dir: &std::path::Path,
) -> (jammi_db::session::JammiSession, Arc<Catalog>) {
    let session = make_test_session(kind, dir).await;
    let catalog = Arc::clone(session.catalog());
    reset_queue(&catalog).await;
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
