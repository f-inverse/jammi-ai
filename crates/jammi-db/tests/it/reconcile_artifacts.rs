//! `ResultStore::reconcile` / `reconcile_all` over `models/`: the pass is
//! row-driven. An artifact's row — its reference, its stager's liveness, its
//! age on the catalog's clock — decides its keys; the listing contributes
//! only strays no row names.
//!
//! Real adapter bundles on a `file://` root, real jobs through the queue.

use std::path::{Path, PathBuf};
use std::time::Duration;

use jammi_db::catalog::artifact_repo::{ArtifactRef, ReclaimDecision, StagedArtifact};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::jobs_repo::FinishJobWithModelParams;
use jammi_db::catalog::status::ArtifactState;
use jammi_db::catalog::Catalog;
use jammi_db::store::{ReconcileOptions, ReconcileReport, ResultStore};
use jammi_db::TenantId;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{
    adapter_files, backdate_artifact, backdate_dir, bundle_dir, files_in, fine_tuned_model,
    queue_session, running_fine_tune_job, store_over,
};

const WORKER: &str = "reconcile-worker";
const WEEK: Duration = Duration::from_secs(7 * 86_400);

const BUNDLE: [&str; 3] = [
    "adapter.safetensors",
    "adapter_config.json",
    "manifest.json",
];

fn apply() -> ReconcileOptions {
    ReconcileOptions {
        apply: true,
        grace: Duration::from_secs(3600),
    }
}

fn dry_run() -> ReconcileOptions {
    ReconcileOptions {
        apply: false,
        ..apply()
    }
}

fn peer_tenant() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f4b".parse().unwrap()
}

/// Age an artifact past any grace on BOTH clocks a pass reads: its row's
/// `created_at` and its objects' mtimes.
async fn age(catalog: &Catalog, artifact: &ArtifactRef) {
    backdate_artifact(catalog, artifact, WEEK).await;
    backdate_dir(&bundle_dir(artifact), WEEK);
}

async fn stage_served(
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

async fn stage_epoch(
    store: &ResultStore,
    catalog: &Catalog,
    job_id: &str,
    attempt: u32,
    epoch: usize,
) -> StagedArtifact {
    store
        .artifact_store()
        .stage_checkpoint(
            catalog,
            job_id,
            attempt,
            epoch,
            &adapter_files(&format!("epoch_{epoch}")),
        )
        .await
        .unwrap()
}

/// The root-relative keys of `artifact`'s bundle files, sorted — the shape a
/// report lists them in.
fn keys_of(root: &Path, artifact: &ArtifactRef, names: &[&str]) -> Vec<String> {
    let dir = bundle_dir(artifact);
    let relative = dir.strip_prefix(root.join("jammi_db")).unwrap();
    let mut keys: Vec<String> = names
        .iter()
        .map(|name| format!("{}/{name}", relative.display()))
        .collect();
    keys.sort();
    keys
}

fn bytes_of(artifact: &ArtifactRef) -> u64 {
    files_in(&bundle_dir(artifact))
        .iter()
        .map(|name| {
            std::fs::metadata(bundle_dir(artifact).join(name))
                .unwrap()
                .len()
        })
        .sum()
}

fn assert_parity(preview: &ReconcileReport, applied: &ReconcileReport) {
    assert!(!preview.applied && applied.applied);
    assert_eq!(preview.orphans, applied.orphans);
    assert_eq!(preview.orphan_count, applied.orphan_count);
    assert_eq!(preview.pending, applied.pending);
    assert_eq!(preview.damaged, applied.damaged);
    assert_eq!(preview.referenced, applied.referenced);
    assert_eq!(preview.unattributed, applied.unattributed);
    assert_eq!(preview.bytes_reclaimed, applied.bytes_reclaimed);
}

/// A live writer's bundles — the attempt's served bundle while the job runs
/// that attempt, the job-scoped resume checkpoint while the job is
/// non-terminal — survive a pass at any age and are never `unattributed`.
/// The moment the job ends they are reclaimable, and the pass reaps them.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_live_writers_bundles_survive_and_are_reaped_once_the_job_ends(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;

    let served = stage_served(&store, &catalog, &job_id, attempt).await;
    let resume = store
        .artifact_store()
        .stage_checkpoint(&catalog, &job_id, attempt, 0, &adapter_files("resume"))
        .await
        .unwrap();
    let artifacts = [served.artifact().clone(), resume.artifact().clone()];
    for artifact in &artifacts {
        age(&catalog, artifact).await;
    }

    let report = store.reconcile(apply()).await.unwrap();
    assert!(report.orphans.is_empty(), "{report:?}");
    assert!(report.pending.is_empty(), "{report:?}");
    assert!(report.unattributed.is_empty(), "{report:?}");
    for artifact in &artifacts {
        assert_eq!(files_in(&bundle_dir(artifact)), BUNDLE);
    }

    assert!(catalog
        .fail_job(&job_id, WORKER, attempt, "diverged")
        .await
        .unwrap());
    let preview = store.reconcile(dry_run()).await.unwrap();
    let expected_bytes: u64 = artifacts.iter().map(bytes_of).sum();
    let reaped = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &reaped);
    let mut expected: Vec<String> = artifacts
        .iter()
        .flat_map(|a| keys_of(dir.path(), a, &BUNDLE))
        .collect();
    expected.sort();
    assert_eq!(reaped.orphans, expected);
    assert_eq!(reaped.bytes_reclaimed, expected_bytes);
    for artifact in &artifacts {
        assert!(files_in(&bundle_dir(artifact)).is_empty());
        assert!(catalog
            .get_model_artifact(artifact)
            .await
            .unwrap()
            .is_none());
    }
}

/// A checkpoint write torn before its manifest — every data file present, no
/// manifest — is nothing a resume read uses (it reads the complete epoch
/// before it) and nothing the store's own retirement can inventory (the
/// store never lists): the pass, which does list, reaps it with its listed
/// keys once the job ends, exactly as it reaps any torn bundle.
#[cfg(feature = "test-hooks")]
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_torn_checkpoint_write_is_reaped_by_its_listing_once_the_job_ends(backend: BackendKind) {
    use jammi_db::store::artifact::artifact_test_hooks;
    use std::sync::Arc;

    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = Arc::new(store_over(dir.path(), &catalog));
    let artifacts = store.artifact_store();
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;

    let complete = artifacts
        .stage_checkpoint(&catalog, &job_id, attempt, 0, &adapter_files("epoch-0"))
        .await
        .unwrap()
        .artifact()
        .clone();
    let torn = ArtifactRef::parse(
        artifacts
            .checkpoint_prefix(None, &job_id, attempt, 1)
            .unwrap()
            .as_str(),
    )
    .unwrap();
    let park = artifact_test_hooks::arm_park_before_manifest(torn.url());
    let writer = tokio::spawn({
        let artifacts = Arc::clone(&artifacts);
        let catalog = Arc::clone(&catalog);
        let job_id = job_id.clone();
        async move {
            artifacts
                .stage_checkpoint(&catalog, &job_id, attempt, 1, &adapter_files("epoch-1"))
                .await
        }
    });
    tokio::time::timeout(Duration::from_secs(5), park.wait_parked())
        .await
        .expect("the writer reaches the seam");
    writer.abort();
    assert!(writer.await.unwrap_err().is_cancelled());
    const TORN: [&str; 2] = ["adapter.safetensors", "adapter_config.json"];
    assert_eq!(files_in(&bundle_dir(&torn)), TORN);

    // The read resumes from epoch 0; the torn prefix is never corruption.
    let read = artifacts
        .fetch_newest_checkpoint(&catalog, &job_id)
        .await
        .unwrap()
        .expect("epoch 0 is complete");
    assert_eq!(read.dir(), bundle_dir(&complete));

    for artifact in [&complete, &torn] {
        age(&catalog, artifact).await;
    }
    let live = store.reconcile(apply()).await.unwrap();
    assert!(live.orphans.is_empty(), "{live:?}");
    assert_eq!(files_in(&bundle_dir(&torn)), TORN);

    assert!(catalog
        .fail_job(&job_id, WORKER, attempt, "diverged")
        .await
        .unwrap());
    let preview = store.reconcile(dry_run()).await.unwrap();
    let reaped = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &reaped);
    let mut expected = keys_of(dir.path(), &complete, &BUNDLE);
    expected.extend(keys_of(dir.path(), &torn, &TORN));
    expected.sort();
    assert_eq!(reaped.orphans, expected);
    for artifact in [&complete, &torn] {
        assert!(files_in(&bundle_dir(artifact)).is_empty());
        assert!(catalog
            .get_model_artifact(artifact)
            .await
            .unwrap()
            .is_none());
    }
}

/// An abandoned bundle — staged by an attempt that is over — is `pending`
/// until its ROW ages past grace on the catalog's clock, whatever its
/// objects' own mtimes say; then it is reaped. A `reclaiming` artifact, by
/// contrast, is an interrupted reclaim and resumes at any age.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn an_abandoned_bundle_ages_on_its_row_and_an_interrupted_reclaim_resumes(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);

    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let abandoned = stage_served(&store, &catalog, &job_id, attempt)
        .await
        .artifact()
        .clone();
    assert!(catalog
        .fail_job(&job_id, WORKER, attempt, "out of memory")
        .await
        .unwrap());
    // Old bytes, young row: the row's clock is the one that counts.
    backdate_dir(&bundle_dir(&abandoned), WEEK);

    let young = store.reconcile(apply()).await.unwrap();
    assert!(young.orphans.is_empty(), "{young:?}");
    assert_eq!(young.pending, keys_of(dir.path(), &abandoned, &BUNDLE));
    assert_eq!(files_in(&bundle_dir(&abandoned)), BUNDLE);

    // A second abandoned bundle whose reclaim was licensed and then
    // interrupted before a single byte was deleted: `reclaiming`, and young.
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let interrupted = stage_served(&store, &catalog, &job_id, attempt).await;
    let interrupted_ref = interrupted.artifact().clone();
    assert!(matches!(
        catalog
            .reclaim_own_staged_artifact(interrupted)
            .await
            .unwrap(),
        ReclaimDecision::Licensed(_)
    ));
    assert_eq!(
        catalog
            .get_model_artifact(&interrupted_ref)
            .await
            .unwrap()
            .unwrap()
            .state,
        ArtifactState::Reclaiming
    );

    let resumed = store.reconcile(apply()).await.unwrap();
    assert_eq!(
        resumed.orphans,
        keys_of(dir.path(), &interrupted_ref, &BUNDLE)
    );
    assert_eq!(resumed.pending, keys_of(dir.path(), &abandoned, &BUNDLE));
    assert!(files_in(&bundle_dir(&interrupted_ref)).is_empty());
    assert!(catalog
        .get_model_artifact(&interrupted_ref)
        .await
        .unwrap()
        .is_none());

    backdate_artifact(&catalog, &abandoned, WEEK).await;
    let preview = store.reconcile(dry_run()).await.unwrap();
    let aged = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &aged);
    assert_eq!(aged.orphans, keys_of(dir.path(), &abandoned, &BUNDLE));
    assert!(files_in(&bundle_dir(&abandoned)).is_empty());
    assert!(catalog
        .get_model_artifact(&abandoned)
        .await
        .unwrap()
        .is_none());
}

/// A served, referenced attempt artifact beside the job's epoch checkpoints:
/// the pass reaps the UNRETAINED checkpoint — an artifact of its own,
/// unreferenced, its job over — and leaves the served bundle and the
/// retained, published checkpoint byte-intact and loadable.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn an_unretained_checkpoint_beneath_a_served_bundle_is_reaped_alone(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let name = format!("jammi:fine-tuned:{job_id}");
    let retained_name = format!("{name}:epoch_1");

    let served = stage_served(&store, &catalog, &job_id, attempt).await;
    let unretained = stage_epoch(&store, &catalog, &job_id, attempt, 0).await;
    let retained = stage_epoch(&store, &catalog, &job_id, attempt, 1).await;
    let (served_ref, unretained_ref, retained_ref) = (
        served.artifact().clone(),
        unretained.artifact().clone(),
        retained.artifact().clone(),
    );
    assert!(!bundle_dir(&unretained_ref).starts_with(bundle_dir(&served_ref)));

    // The finalize retains epoch 1 only; the finisher never gets to sweep
    // epoch 0 (its process ends right after the commit).
    assert!(catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: fine_tuned_model(&name, served),
            epoch_checkpoints: vec![fine_tuned_model(&retained_name, retained)],
        })
        .await
        .unwrap());
    drop(unretained);
    let before: Vec<(PathBuf, Vec<u8>)> = [&served_ref, &retained_ref]
        .into_iter()
        .flat_map(|artifact| {
            let bundle = bundle_dir(artifact);
            files_in(&bundle)
                .into_iter()
                .map(move |name| bundle.join(name))
        })
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    assert_eq!(before.len(), 2 * BUNDLE.len());
    for artifact in [&served_ref, &unretained_ref, &retained_ref] {
        age(&catalog, artifact).await;
    }

    let preview = store.reconcile(dry_run()).await.unwrap();
    let reaped = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &reaped);
    assert_eq!(
        reaped.orphans,
        keys_of(dir.path(), &unretained_ref, &BUNDLE)
    );
    assert!(reaped.damaged.is_empty(), "{reaped:?}");
    assert!(files_in(&bundle_dir(&unretained_ref)).is_empty());
    assert!(catalog
        .get_model_artifact(&unretained_ref)
        .await
        .unwrap()
        .is_none());

    for (path, bytes) in &before {
        assert_eq!(&std::fs::read(path).unwrap(), bytes, "{}", path.display());
    }
    for artifact in [&served_ref, &retained_ref] {
        store
            .artifact_store()
            .fetch_artifact(artifact.url())
            .await
            .unwrap();
    }
    assert!(catalog.get_model(&retained_name).await.unwrap().is_some());
}

/// A referenced artifact is only ever INSPECTED: a bundle that lost its
/// manifest, or an object its manifest lists, is reported `damaged` and never
/// deleted; a foreign file sitting inside it is protected with the rest.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_referenced_bundle_is_inspected_never_deleted(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);

    let mut served = Vec::new();
    for _ in 0..2 {
        let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
        let name = format!("jammi:fine-tuned:{job_id}");
        let staged = stage_served(&store, &catalog, &job_id, attempt).await;
        let artifact = staged.artifact().clone();
        assert!(catalog
            .finish_job_with_model(FinishJobWithModelParams {
                job_id: &job_id,
                instance_id: WORKER,
                attempts: attempt,
                result: "{}",
                output: fine_tuned_model(&name, staged),
                epoch_checkpoints: Vec::new(),
            })
            .await
            .unwrap());
        served.push(artifact);
    }
    let (no_manifest, missing_weights) = (&served[0], &served[1]);
    std::fs::remove_file(bundle_dir(no_manifest).join("manifest.json")).unwrap();
    std::fs::write(bundle_dir(no_manifest).join("debug_dump.tmp"), b"leftover").unwrap();
    std::fs::remove_file(bundle_dir(missing_weights).join("adapter.safetensors")).unwrap();
    for artifact in &served {
        age(&catalog, artifact).await;
    }

    let preview = store.reconcile(dry_run()).await.unwrap();
    let report = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &report);
    assert!(report.orphans.is_empty(), "{report:?}");
    assert!(report.pending.is_empty(), "{report:?}");
    let mut expected = keys_of(
        dir.path(),
        no_manifest,
        &[
            "adapter.safetensors",
            "adapter_config.json",
            "debug_dump.tmp",
            "manifest.json",
        ],
    );
    expected.extend(keys_of(
        dir.path(),
        missing_weights,
        &["adapter.safetensors"],
    ));
    expected.sort();
    assert_eq!(report.damaged, expected);
    assert_eq!(
        files_in(&bundle_dir(no_manifest)),
        [
            "adapter.safetensors",
            "adapter_config.json",
            "debug_dump.tmp"
        ]
    );
    assert_eq!(
        files_in(&bundle_dir(missing_weights)),
        ["adapter_config.json", "manifest.json"]
    );
}

/// Bytes under `models/` that no row names are strays: `pending` while
/// young, adopted and reclaimed once every key in the directory has aged —
/// dry-run and apply agreeing — with no row left behind. A scoped pass
/// touches only its own tenant segment; a key whose path is not an artifact's
/// shape at all stays `unattributed`, reported by the admin pass and never
/// deleted.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn an_aged_stray_is_adopted_and_reclaimed(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let models = dir.path().join("jammi_db").join("models");

    // A bundle directory restored from a backup the catalog never heard of.
    let write_stray = |segment: String| {
        let job = uuid::Uuid::new_v4().to_string();
        let stray = models.join(segment).join(job).join("restored").join("1");
        std::fs::create_dir_all(&stray).unwrap();
        for (name, bytes) in adapter_files("stray") {
            std::fs::write(stray.join(name), bytes).unwrap();
        }
        stray
    };
    let relative = |stray: &Path| -> Vec<String> {
        let rel = stray.strip_prefix(dir.path().join("jammi_db")).unwrap();
        files_in(stray)
            .iter()
            .map(|name| format!("{}/{name}", rel.display()))
            .collect()
    };
    let global_stray = write_stray("_global".to_string());
    let peer_stray = write_stray(peer_tenant().to_string());
    let unattributed = models.join("_global").join("not-a-job-id");
    std::fs::create_dir_all(&unattributed).unwrap();
    std::fs::write(unattributed.join("notes.txt"), b"keep").unwrap();
    backdate_dir(&unattributed, WEEK);
    let global_keys = relative(&global_stray);
    let peer_keys = relative(&peer_stray);

    let young = store.reconcile(apply()).await.unwrap();
    assert_eq!(young.pending, global_keys);
    assert!(young.orphans.is_empty(), "{young:?}");
    assert_eq!(files_in(&global_stray).len(), 2);

    backdate_dir(&global_stray, WEEK);
    backdate_dir(&peer_stray, WEEK);
    let preview = store.reconcile(dry_run()).await.unwrap();
    assert_eq!(
        files_in(&global_stray).len(),
        2,
        "a dry-run deletes nothing"
    );
    let reaped = store.reconcile(apply()).await.unwrap();
    assert_parity(&preview, &reaped);
    assert_eq!(reaped.orphans, global_keys);
    assert_eq!(
        reaped.bytes_reclaimed,
        adapter_files("stray")
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>()
    );
    assert!(reaped.unattributed.is_empty(), "a scoped pass: {reaped:?}");
    assert!(files_in(&global_stray).is_empty());
    assert_eq!(
        files_in(&peer_stray).len(),
        2,
        "an unbound pass never touches a tenant's segment"
    );

    let admin = store.reconcile_all(apply()).await.unwrap();
    assert_eq!(admin.orphans, peer_keys);
    assert!(files_in(&peer_stray).is_empty());
    assert_eq!(
        admin.unattributed,
        ["models/_global/not-a-job-id/notes.txt"]
    );
    assert!(unattributed.join("notes.txt").exists());

    // Adoption leaves no row behind once the bytes are gone.
    let settled = store.reconcile_all(apply()).await.unwrap();
    assert!(settled.orphans.is_empty() && settled.pending.is_empty());
    assert!(catalog
        .list_model_artifacts_for_reconcile(Duration::ZERO)
        .await
        .unwrap()
        .iter()
        .all(|a| !a.record.artifact.url().as_str().contains("/restored/")));
}
