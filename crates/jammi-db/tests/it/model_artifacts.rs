//! The `model_artifacts` entity: staging before the first byte, the reclaim
//! compare-and-set that mints a [`ReclaimLicence`], and the one `reclaim`
//! primitive that deletes a bundle's bytes under it.
//!
//! Every test runs real bundles through a real [`ResultStore`] on a `file://`
//! root, so "the bytes are gone" and "the bytes are intact" are read off the
//! filesystem, never inferred from a return value.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use jammi_db::catalog::artifact_repo::{ReclaimDecision, ReclaimLicence, StagedArtifact};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::status::{ArtifactState, JobExecution};
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::store::ResultStore;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::queue_session;

const KIND: &str = "fine_tune";
const WORKER: &str = "artifact-worker";

fn tenant_a() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f2a".parse().unwrap()
}

fn tenant_b() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f2b".parse().unwrap()
}

/// A LoRA adapter bundle's two files, with distinguishable bytes.
fn adapter_files(tag: &str) -> Vec<(String, Bytes)> {
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

fn store_over(dir: &Path, catalog: &Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, Arc::clone(catalog), AnnIndexConfig::default()).unwrap()
}

/// The local directory a `file://` artifact's bundle lives in.
fn bundle_dir(staged_url: &str) -> PathBuf {
    PathBuf::from(staged_url.strip_prefix("file://").unwrap())
}

fn files_in(dir: &Path) -> Vec<String> {
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

/// Submit and claim a fresh queued job, returning its id and the attempt the
/// claim stamped — a job that is genuinely `running` that attempt.
async fn running_job(catalog: &Catalog) -> (String, u32) {
    let job_id = uuid::Uuid::new_v4().to_string();
    catalog
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: KIND,
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some("q-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next(WORKER, &[KIND], Duration::from_secs(60))
        .await
        .unwrap()
        .expect("a queued job is claimable");
    (claimed.job_id, claimed.attempts)
}

async fn licensed(decision: ReclaimDecision) -> ReclaimLicence {
    match decision {
        ReclaimDecision::Licensed(licence) => licence,
        other => panic!("expected a licence, got {other:?}"),
    }
}

async fn stage(
    store: &ResultStore,
    catalog: &Catalog,
    job_id: &str,
    attempt: u32,
) -> StagedArtifact {
    store
        .artifact_store()
        .stage_attempt_artifact(catalog, job_id, WORKER, attempt, &adapter_files(job_id))
        .await
        .unwrap()
}

/// Staging writes the row before the bytes, names the writer, and is
/// idempotent for that writer only.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn staging_records_the_writer_and_refuses_another(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_job(&catalog).await;

    let staged = stage(&store, &catalog, &job_id, attempt).await;
    let record = catalog
        .get_model_artifact(staged.artifact())
        .await
        .unwrap()
        .expect("the staged row exists");
    assert_eq!(record.state, ArtifactState::Staged);
    assert_eq!(record.staging.as_ref(), Some(staged.scope()));
    assert_eq!(record.tenant_id, None);
    assert_eq!(
        files_in(&bundle_dir(staged.artifact().url().as_str())),
        vec![
            "adapter.safetensors",
            "adapter_config.json",
            "manifest.json"
        ]
    );

    // The same writer re-stages the same prefix (a resume checkpoint does,
    // every epoch); a different attempt of the same job may not.
    let again = stage(&store, &catalog, &job_id, attempt).await;
    assert_eq!(again, staged);
    let resume_first = store
        .artifact_store()
        .stage_resume_checkpoint(&catalog, &job_id, &adapter_files("epoch-0"))
        .await
        .unwrap();
    let resume_second = store
        .artifact_store()
        .stage_resume_checkpoint(&catalog, &job_id, &adapter_files("epoch-1"))
        .await
        .unwrap();
    assert_eq!(resume_first, resume_second);
    assert_eq!(
        std::fs::read(
            bundle_dir(resume_second.artifact().url().as_str()).join("adapter.safetensors")
        )
        .unwrap(),
        b"weights:epoch-1"
    );
}

/// The compare-and-set matrix over a `staged` artifact: a live stager's
/// bundle is closed to everyone but the stager; once the stager is no longer
/// live anyone in tenant scope may reclaim it.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_live_stagers_bundle_is_reclaimable_only_by_the_stager(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let artifacts = store.artifact_store();

    // Live attempt, not the stager: refused, bytes intact.
    let (job_id, attempt) = running_job(&catalog).await;
    let staged = stage(&store, &catalog, &job_id, attempt).await;
    let artifact = staged.artifact().clone();
    assert!(matches!(
        catalog.begin_artifact_reclaim(&artifact).await.unwrap(),
        ReclaimDecision::Live
    ));
    assert_eq!(files_in(&bundle_dir(artifact.url().as_str())).len(), 3);

    // The stager itself, while still live: licensed.
    let licence = licensed(catalog.reclaim_own_staged_artifact(staged).await.unwrap()).await;
    assert_eq!(
        catalog
            .get_model_artifact(&artifact)
            .await
            .unwrap()
            .unwrap()
            .state,
        ArtifactState::Reclaiming
    );
    artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert!(files_in(&bundle_dir(artifact.url().as_str())).is_empty());
    assert!(catalog
        .get_model_artifact(&artifact)
        .await
        .unwrap()
        .is_none());

    // A stager that is no longer live — its job failed — protects nothing.
    let (job_id, attempt) = running_job(&catalog).await;
    let staged = stage(&store, &catalog, &job_id, attempt).await;
    let artifact = staged.artifact().clone();
    assert!(catalog
        .fail_job(&job_id, WORKER, attempt, "out of memory")
        .await
        .unwrap());
    let licence = licensed(catalog.begin_artifact_reclaim(&artifact).await.unwrap()).await;
    artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert!(files_in(&bundle_dir(artifact.url().as_str())).is_empty());

    // Nothing left to reclaim.
    assert!(matches!(
        catalog.begin_artifact_reclaim(&artifact).await.unwrap(),
        ReclaimDecision::Absent
    ));
}

/// A job-scoped bundle (the resume checkpoint) is protected for as long as
/// its job is non-terminal — across attempts, queued or running — and is
/// reclaimable the moment the job ends.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_resume_checkpoint_is_protected_until_its_job_ends(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let artifacts = store.artifact_store();
    let (job_id, attempt) = running_job(&catalog).await;

    let resume = artifacts
        .stage_resume_checkpoint(&catalog, &job_id, &adapter_files("epoch-0"))
        .await
        .unwrap();
    let artifact = artifacts.resume_checkpoint_ref(None, &job_id).unwrap();
    assert_eq!(&artifact, resume.artifact());
    assert!(matches!(
        catalog.begin_artifact_reclaim(&artifact).await.unwrap(),
        ReclaimDecision::Live
    ));

    assert!(catalog
        .fail_job(&job_id, WORKER, attempt, "diverged")
        .await
        .unwrap());
    let licence = licensed(catalog.begin_artifact_reclaim(&artifact).await.unwrap()).await;
    let deleted = artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert_eq!(deleted.len(), 3, "two files and the manifest: {deleted:?}");
    assert!(files_in(&bundle_dir(artifact.url().as_str())).is_empty());
}

/// An epoch checkpoint nests beneath its attempt's served prefix in the
/// physical layout only: each is its own artifact, and reclaiming one never
/// touches the other's bytes — in either direction.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_licence_covers_its_own_flat_bundle_and_nothing_nested(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let artifacts = store.artifact_store();
    let (job_id, attempt) = running_job(&catalog).await;

    let served = stage(&store, &catalog, &job_id, attempt).await;
    let checkpoint = artifacts
        .stage_epoch_checkpoint(&catalog, &job_id, WORKER, attempt, 0, &adapter_files("e0"))
        .await
        .unwrap();
    let served_dir = bundle_dir(served.artifact().url().as_str());
    let checkpoint_dir = bundle_dir(checkpoint.artifact().url().as_str());
    assert!(checkpoint_dir.starts_with(&served_dir));

    let recovered = catalog
        .staged_artifacts_of_attempt(&job_id, attempt)
        .await
        .unwrap();
    assert_eq!(recovered.len(), 2, "both staged bundles are recoverable");

    // Reclaim the ENCLOSING served bundle, handing the reclaim the nested
    // checkpoint's keys as if a listing had found them: the licence refuses
    // them at the raw delete, and the checkpoint survives whole.
    let nested_key = object_store::path::Path::parse(
        checkpoint_dir
            .join("adapter.safetensors")
            .to_string_lossy()
            .trim_start_matches('/'),
    )
    .unwrap();
    let licence = licensed(catalog.reclaim_own_staged_artifact(served).await.unwrap()).await;
    assert!(!licence.covers(&nested_key));
    let refused = artifacts
        .reclaim(&catalog, licence, std::slice::from_ref(&nested_key))
        .await
        .unwrap_err();
    assert!(
        matches!(
            refused,
            jammi_db::error::JammiError::Storage(
                jammi_db::storage::StorageError::NotLicensed { .. }
            )
        ),
        "{refused}"
    );
    assert_eq!(files_in(&checkpoint_dir).len(), 3);

    // The refused reclaim left the served artifact `reclaiming`; licensing it
    // again resumes it, and the checkpoint is still untouched afterwards.
    let artifact = recovered
        .iter()
        .map(|s| s.artifact())
        .find(|a| bundle_dir(a.url().as_str()) == served_dir)
        .unwrap()
        .clone();
    let licence = licensed(catalog.begin_artifact_reclaim(&artifact).await.unwrap()).await;
    artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert!(files_in(&served_dir).is_empty());
    assert_eq!(files_in(&checkpoint_dir).len(), 3);

    let licence = licensed(
        catalog
            .reclaim_own_staged_artifact(checkpoint)
            .await
            .unwrap(),
    )
    .await;
    artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert!(files_in(&checkpoint_dir).is_empty());
}

/// Reclaim is a tenant-strict write: a tenant-bound caller sees neither a
/// peer's nor a global artifact, and learns nothing but `Absent`. Admin scope
/// spans every tenant.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn reclaim_is_tenant_strict_and_admin_scope_spans_tenants(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let artifacts = store.artifact_store();
    let cat_a = catalog.pinned_to_tenant(Some(tenant_a()));
    let cat_b = catalog.pinned_to_tenant(Some(tenant_b()));

    // No `jobs` row names these stagers, so neither bundle is live.
    let job_a = uuid::Uuid::new_v4().to_string();
    let job_global = uuid::Uuid::new_v4().to_string();
    let owned_by_a = stage(&store, &cat_a, &job_a, 1).await.artifact().clone();
    let global = stage(&store, &catalog, &job_global, 1)
        .await
        .artifact()
        .clone();
    assert_eq!(
        cat_a
            .get_model_artifact(&owned_by_a)
            .await
            .unwrap()
            .unwrap()
            .tenant_id,
        Some(tenant_a())
    );

    for foreign in [&owned_by_a, &global] {
        assert!(matches!(
            cat_b.begin_artifact_reclaim(foreign).await.unwrap(),
            ReclaimDecision::Absent
        ));
    }
    assert!(matches!(
        cat_a.begin_artifact_reclaim(&global).await.unwrap(),
        ReclaimDecision::Absent
    ));

    let licence = licensed(cat_a.begin_artifact_reclaim(&owned_by_a).await.unwrap()).await;
    artifacts.reclaim(&cat_a, licence, &[]).await.unwrap();

    let licence = licensed(
        TenantBinding::admin_scope(cat_b.begin_artifact_reclaim(&global))
            .await
            .unwrap(),
    )
    .await;
    artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    assert!(files_in(&bundle_dir(global.url().as_str())).is_empty());
}

/// The reconcile listing classifies on facts the catalog evaluates itself:
/// liveness against `jobs`, and age against the backend's own clock.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn the_reconcile_listing_reports_liveness_and_age(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let cat = catalog.pinned_to_tenant(Some(tenant_a()));
    let (job_id, attempt) = running_job(&catalog).await;
    let live = stage(&store, &cat, &job_id, attempt).await;
    let abandoned = stage(&store, &cat, &uuid::Uuid::new_v4().to_string(), 1).await;

    let fresh = cat
        .list_model_artifacts_for_reconcile(Duration::from_secs(3600))
        .await
        .unwrap();
    let aged = cat
        .list_model_artifacts_for_reconcile(Duration::ZERO)
        .await
        .unwrap();
    for (listing, expect_aged) in [(&fresh, false), (&aged, true)] {
        let find = |staged: &StagedArtifact| {
            listing
                .iter()
                .find(|a| &a.record.artifact == staged.artifact())
                .unwrap_or_else(|| panic!("{} is listed", staged.artifact()))
        };
        assert!(find(&live).live);
        assert!(!find(&abandoned).live);
        assert!(!find(&live).referenced && !find(&abandoned).referenced);
        assert_eq!(find(&abandoned).aged, expect_aged);
    }
}
