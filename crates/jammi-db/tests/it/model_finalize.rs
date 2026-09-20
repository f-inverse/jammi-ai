//! The finalize transaction ([`Catalog::finish_job_with_model`]): the job's
//! terminal flip, the artifact's `staged → published` flip, and the `models`
//! row referencing it are one write or none — and what that reference then
//! protects.
//!
//! Every test runs real adapter bundles through a real [`ResultStore`] on a
//! `file://` root and real jobs through the queue, so "the bytes are gone"
//! and "the bytes are intact" are read off the filesystem.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::artifact_repo::{
    ArtifactRef, MaterializationSummary, ReclaimDecision, StagedArtifact,
};
use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::{FinishJobWithModelParams, ModelRow, ProducedModel};
use jammi_db::catalog::model_repo::{ModelLocation, RegisterModelParams};
use jammi_db::catalog::status::{ArtifactState, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ArtifactDigest, ComputeDevice, ComputePrecision, InputAnchor, Materialization,
    MaterializationEnv, ModelContentDigest, ModelIdentity, ProducingDescriptor,
};
use jammi_db::store::{ReconcileOptions, ResultStore};
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{
    adapter_files, backdate_artifact, bundle_dir, files_in, fine_tuned_model, queue_session,
    running_fine_tune_job, store_over, FINE_TUNE_KIND,
};

const WORKER: &str = "finalize-worker";

/// The three files a staged adapter bundle holds on disk.
const BUNDLE: [&str; 3] = [
    "adapter.safetensors",
    "adapter_config.json",
    "manifest.json",
];

fn tenant_a() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f3a".parse().unwrap()
}

fn output_name(job_id: &str) -> String {
    format!("jammi:fine-tuned:{job_id}")
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
        .stage_epoch_checkpoint(
            catalog,
            job_id,
            WORKER,
            attempt,
            epoch,
            &adapter_files(&format!("{job_id}:epoch_{epoch}")),
        )
        .await
        .unwrap()
}

async fn state_of(catalog: &Catalog, artifact: &ArtifactRef) -> Option<ArtifactState> {
    catalog
        .get_model_artifact(artifact)
        .await
        .unwrap()
        .map(|record| record.state)
}

/// Run one attempt end to end: stage the served bundle, finalize it as
/// `name`, and return the published artifact. The attempt must win.
async fn finalize_served(
    store: &ResultStore,
    catalog: &Catalog,
    job_id: &str,
    attempt: u32,
    name: &str,
) -> ArtifactRef {
    let served = stage_served(store, catalog, job_id, attempt).await;
    let artifact = served.artifact().clone();
    let won = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: fine_tuned_model(name, served),
            epoch_checkpoints: Vec::new(),
        })
        .await
        .unwrap();
    assert!(won, "the lease holder finalizes");
    artifact
}

/// The attestation a fine-tune writes beside its bundle, reduced to the
/// summary the finalize records on the artifact row.
async fn attest(store: &ResultStore, staged: &StagedArtifact) -> MaterializationSummary {
    let descriptor = ProducingDescriptor::FineTune {
        training_set_definition_hash: "a".repeat(64),
        training_set_artifact_digest: "b".repeat(64),
        training_set_row_count: 128,
        spec_canonical: r#"{"base_model":"q-base"}"#.into(),
        spec_schema_version: 1,
        base_model_id: "q-base".into(),
        world_size: 1,
        collective: "noop".into(),
        local_ranks: 1,
    };
    let env = MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "q-base".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("fixture-digest".into()),
            quantization: None,
        }],
    );
    let anchors: Vec<InputAnchor> = vec![InputAnchor::result_digest(
        "training-set",
        &ArtifactDigest("c".repeat(64)),
    )];
    let manifest = store
        .artifact_store()
        .write_model_materialization(staged, Materialization::new(&descriptor, &env, anchors))
        .await
        .unwrap();
    MaterializationSummary {
        definition_hash: manifest.definition_hash.as_str().to_string(),
        input_anchors_json: serde_json::to_string(&manifest.input_anchors).unwrap(),
    }
}

/// A won finalize is one write: the job completes, the served artifact and
/// every retained checkpoint publish, each `models` row references its
/// artifact, and the materialization summary lands on the ARTIFACT row —
/// none of which existed a moment before.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_won_finalize_publishes_and_attaches_in_one_transaction(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let name = output_name(&job_id);

    let served = stage_served(&store, &catalog, &job_id, attempt).await;
    let checkpoint = stage_epoch(&store, &catalog, &job_id, attempt, 1).await;
    let summary = attest(&store, &served).await;
    let (served_ref, checkpoint_ref) = (served.artifact().clone(), checkpoint.artifact().clone());
    let checkpoint_name = format!("{name}:epoch_1");

    // Nothing is registered ahead of the finalize.
    assert!(catalog.get_model(&name).await.unwrap().is_none());
    assert_eq!(
        state_of(&catalog, &served_ref).await,
        Some(ArtifactState::Staged)
    );

    let won = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: r#"{"kind":"model"}"#,
            output: {
                let model = fine_tuned_model(&name, served);
                ProducedModel {
                    row: ModelRow {
                        config_json: Some(r#"{"r":8}"#),
                        ..model.row
                    },
                    materialization: Some(summary.clone()),
                    ..model
                }
            },
            epoch_checkpoints: vec![fine_tuned_model(&checkpoint_name, checkpoint)],
        })
        .await
        .unwrap();
    assert!(won);

    assert_eq!(
        catalog.get_job(&job_id).await.unwrap().status,
        JobStatus::Completed.to_string()
    );
    let model = catalog.get_model(&name).await.unwrap().unwrap();
    assert_eq!(
        model.location,
        Some(ModelLocation::Artifact(served_ref.clone()))
    );
    assert_eq!(model.model_type, "fine-tuned");
    assert_eq!(model.base_model_id.as_deref(), Some("q-base"));
    assert_eq!(model.config_json.as_deref(), Some(r#"{"r":8}"#));
    assert_eq!(model.status, "registered");
    let checkpoint_model = catalog.get_model(&checkpoint_name).await.unwrap().unwrap();
    assert_eq!(
        checkpoint_model.location,
        Some(ModelLocation::Artifact(checkpoint_ref.clone()))
    );
    assert_eq!(checkpoint_model.status, "checkpoint");

    let served_row = catalog
        .get_model_artifact(&served_ref)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(served_row.state, ArtifactState::Published);
    assert_eq!(
        served_row.definition_hash.as_deref(),
        Some(summary.definition_hash.as_str())
    );
    assert_eq!(
        served_row.input_anchors_json.as_deref(),
        Some(summary.input_anchors_json.as_str())
    );
    let checkpoint_row = catalog
        .get_model_artifact(&checkpoint_ref)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(checkpoint_row.state, ArtifactState::Published);
    assert_eq!(checkpoint_row.definition_hash, None);

    // Both bundles load.
    for artifact in [&served_ref, &checkpoint_ref] {
        store
            .artifact_store()
            .fetch_artifact(artifact.url())
            .await
            .unwrap();
    }
}

/// An attempt that lost its lease finalizes nothing — no job status, no
/// `models` row, no published artifact — and its own sweep then reclaims
/// every byte it staged, leaving the successor's finalize a clean slate.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lost_lease_finalize_writes_nothing_and_its_sweep_reclaims_its_bytes(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let artifacts = store.artifact_store();

    // The first claimant's lease expires at once; the job is requeued and a
    // successor claims attempt 2.
    let job_id = uuid::Uuid::new_v4().to_string();
    let name = output_name(&job_id);
    catalog
        .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
            job_id: &job_id,
            kind: FINE_TUNE_KIND,
            execution: jammi_db::catalog::status::JobExecution::Queued,
            spec: "{}",
            model_ref: Some("q-base::1"),
            output_model_id: Some(&name),
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let stale = catalog
        .claim_next(WORKER, &[FINE_TUNE_KIND], Duration::ZERO)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        catalog
            .reclaim_expired_jobs(Duration::ZERO, 5)
            .await
            .unwrap(),
        1
    );
    let live = catalog
        .claim_next("successor", &[FINE_TUNE_KIND], Duration::from_secs(3600))
        .await
        .unwrap()
        .unwrap();
    assert_eq!((stale.attempts, live.attempts), (1, 2));

    let served = stage_served(&store, &catalog, &job_id, stale.attempts).await;
    let checkpoint = stage_epoch(&store, &catalog, &job_id, stale.attempts, 0).await;
    let staged_refs = [served.artifact().clone(), checkpoint.artifact().clone()];
    let checkpoint_name = format!("{name}:epoch_0");

    let won = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: stale.attempts,
            result: "{}",
            output: fine_tuned_model(&name, served),
            epoch_checkpoints: vec![fine_tuned_model(&checkpoint_name, checkpoint)],
        })
        .await
        .unwrap();
    assert!(!won, "a stale attempt does not finalize");
    let job = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(job.status, JobStatus::Running.to_string());
    assert_eq!(job.claimed_by.as_deref(), Some("successor"));
    assert!(catalog.get_model(&name).await.unwrap().is_none());
    assert!(catalog.get_model(&checkpoint_name).await.unwrap().is_none());
    for artifact in &staged_refs {
        assert_eq!(
            state_of(&catalog, artifact).await,
            Some(ArtifactState::Staged)
        );
    }

    // The loser's sweep: every artifact the attempt staged and did not
    // publish, recovered from the catalog, reclaimed as its own.
    let held = catalog
        .staged_artifacts_of_attempt(&job_id, stale.attempts)
        .await
        .unwrap();
    assert_eq!(held.len(), 2);
    for staged in held {
        let ReclaimDecision::Licensed(licence) =
            catalog.reclaim_own_staged_artifact(staged).await.unwrap()
        else {
            panic!("an attempt reclaims its own unpublished bundle");
        };
        artifacts.reclaim(&catalog, licence, &[]).await.unwrap();
    }
    for artifact in &staged_refs {
        assert!(files_in(&bundle_dir(artifact)).is_empty());
        assert_eq!(state_of(&catalog, artifact).await, None);
    }

    // The successor's own attempt finalizes normally.
    let served = store
        .artifact_store()
        .stage_attempt_artifact(
            &catalog,
            &job_id,
            "successor",
            live.attempts,
            &adapter_files("successor"),
        )
        .await
        .unwrap();
    let successor_ref = served.artifact().clone();
    assert!(catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: "successor",
            attempts: live.attempts,
            result: "{}",
            output: fine_tuned_model(&name, served),
            epoch_checkpoints: Vec::new(),
        })
        .await
        .unwrap());
    assert_eq!(
        catalog.get_model(&name).await.unwrap().unwrap().location,
        Some(ModelLocation::Artifact(successor_ref))
    );
}

/// A finalize that names an artifact which is no longer the caller's staged
/// bundle fails WHOLE: the job stays `running`, no `models` row exists, and
/// the output artifact the same transaction had already flipped is `staged`
/// again.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_finalize_over_an_artifact_no_longer_staged_rolls_back_whole(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let name = output_name(&job_id);
    let checkpoint_name = format!("{name}:epoch_0");

    let served = stage_served(&store, &catalog, &job_id, attempt).await;
    let checkpoint = stage_epoch(&store, &catalog, &job_id, attempt, 0).await;
    let (served_ref, checkpoint_ref) = (served.artifact().clone(), checkpoint.artifact().clone());

    // The retention prune began reclaiming the checkpoint: a second claim on
    // it, recovered from the catalog, wins the reclaim compare-and-set.
    let pruned = catalog
        .staged_artifacts_of_attempt(&job_id, attempt)
        .await
        .unwrap()
        .into_iter()
        .find(|s| s.artifact() == &checkpoint_ref)
        .unwrap();
    assert!(matches!(
        catalog.reclaim_own_staged_artifact(pruned).await.unwrap(),
        ReclaimDecision::Licensed(_)
    ));

    let refused = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: fine_tuned_model(&name, served),
            epoch_checkpoints: vec![fine_tuned_model(&checkpoint_name, checkpoint)],
        })
        .await
        .unwrap_err();
    assert!(
        refused.to_string().contains(checkpoint_ref.url().as_str()),
        "{refused}"
    );

    assert_eq!(
        catalog.get_job(&job_id).await.unwrap().status,
        JobStatus::Running.to_string()
    );
    assert!(catalog.get_model(&name).await.unwrap().is_none());
    assert!(catalog.get_model(&checkpoint_name).await.unwrap().is_none());
    assert_eq!(
        state_of(&catalog, &served_ref).await,
        Some(ArtifactState::Staged),
        "the output's publish rolled back with the rest"
    );
    assert_eq!(
        state_of(&catalog, &checkpoint_ref).await,
        Some(ArtifactState::Reclaiming)
    );
    assert_eq!(files_in(&bundle_dir(&served_ref)), BUNDLE);
}

/// A retained checkpoint whose catalog name another row already occupies is
/// skipped without failing the job — and is NOT published: it stays the
/// attempt's own staged bundle, which the finisher's sweep reclaims, while
/// every other checkpoint registers normally.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_name_occupied_checkpoint_stays_staged_for_the_sweep(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let name = output_name(&job_id);
    let (occupied_name, free_name) = (format!("{name}:epoch_0"), format!("{name}:epoch_1"));

    // An unrelated row already holds epoch_0's name, at another version.
    catalog
        .register_model(RegisterModelParams {
            model_id: &occupied_name,
            version: 7,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            external_location: Some("/weights/unrelated"),
            config_json: None,
        })
        .await
        .unwrap();

    let served = stage_served(&store, &catalog, &job_id, attempt).await;
    let skipped = stage_epoch(&store, &catalog, &job_id, attempt, 0).await;
    let registered = stage_epoch(&store, &catalog, &job_id, attempt, 1).await;
    let (skipped_ref, registered_ref) = (skipped.artifact().clone(), registered.artifact().clone());

    assert!(catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: fine_tuned_model(&name, served),
            epoch_checkpoints: vec![
                fine_tuned_model(&occupied_name, skipped),
                fine_tuned_model(&free_name, registered),
            ],
        })
        .await
        .unwrap());

    let occupant = catalog.get_model(&occupied_name).await.unwrap().unwrap();
    assert_eq!(occupant.version, 7, "the occupant is untouched");
    assert_eq!(
        occupant.location,
        Some(ModelLocation::External("/weights/unrelated".to_string()))
    );
    assert_eq!(
        catalog
            .get_model(&free_name)
            .await
            .unwrap()
            .unwrap()
            .location,
        Some(ModelLocation::Artifact(registered_ref))
    );
    assert_eq!(
        state_of(&catalog, &skipped_ref).await,
        Some(ArtifactState::Staged)
    );

    // What the winner's sweep finds is exactly the skipped checkpoint.
    let held = catalog
        .staged_artifacts_of_attempt(&job_id, attempt)
        .await
        .unwrap();
    assert_eq!(
        held.iter().map(|s| s.artifact()).collect::<Vec<_>>(),
        vec![&skipped_ref]
    );
}

/// The output row is written for exactly `(tenant, name, version)`: a peer
/// tenant's row and another version of the same name are untouched, and a
/// re-registration of the produced row is refused.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn the_output_row_is_scoped_by_tenant_and_version(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let cat_a = Arc::new(catalog.pinned_to_tenant(Some(tenant_a())));
    // Unique per run: the Postgres lane reuses one database across runs.
    let tuned = format!("acme/tuned-{}", uuid::Uuid::new_v4().simple());
    let base = |version: i32| RegisterModelParams {
        model_id: &tuned,
        version,
        model_type: "embedding",
        backend: "candle",
        task: ModelTask::TextEmbedding,
        base_model_id: None,
        external_location: Some("/weights/acme"),
        config_json: None,
    };
    cat_a.register_model(base(2)).await.unwrap();
    catalog.register_model(base(1)).await.unwrap();

    let (job_id, attempt) = running_fine_tune_job(&cat_a, WORKER, Some(&tuned)).await;
    let published = finalize_served(&store, &cat_a, &job_id, attempt, &tuned).await;
    assert!(published
        .url()
        .as_str()
        .contains(&format!("/models/{}/", tenant_a())));

    let produced = cat_a.get_model_version(&tuned, 1).await.unwrap().unwrap();
    assert_eq!(produced.location, Some(ModelLocation::Artifact(published)));
    assert_eq!(produced.catalog_pk, format!("{}::{tuned}::1", tenant_a()));
    let external = Some(ModelLocation::External("/weights/acme".to_string()));
    assert_eq!(
        cat_a
            .get_model_version(&tuned, 2)
            .await
            .unwrap()
            .unwrap()
            .location,
        external
    );
    assert_eq!(
        catalog
            .get_model_version(&tuned, 1)
            .await
            .unwrap()
            .unwrap()
            .location,
        external
    );

    // The produced row is the finalize's to write, not a registration's.
    let refused = cat_a.register_model(base(1)).await.unwrap_err();
    assert!(
        matches!(refused, jammi_db::error::JammiError::Model { .. }),
        "{refused}"
    );
    assert_eq!(
        cat_a
            .get_model_version(&tuned, 1)
            .await
            .unwrap()
            .unwrap()
            .model_type,
        "fine-tuned"
    );
}

/// The database itself refuses to drop an artifact row a `models` row
/// references — the foreign key is `ON DELETE RESTRICT` on both backends.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn the_foreign_key_refuses_to_drop_a_referenced_artifact_row(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let name = output_name(&job_id);
    let artifact = finalize_served(&store, &catalog, &job_id, attempt, &name).await;

    let prefix = artifact.url().as_str().to_string();
    let refused = catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM model_artifacts WHERE prefix = $1",
                    &[SqlValue::TextOwned(prefix)],
                )
                .await
            })
        })
        .await;
    assert!(
        refused.is_err(),
        "the delete must be refused, got {refused:?}"
    );
    assert_eq!(
        state_of(&catalog, &artifact).await,
        Some(ArtifactState::Published)
    );
    assert!(catalog
        .model_artifact_is_referenced(&artifact)
        .await
        .unwrap());
}

/// While the served row references the artifact nothing reclaims it — not
/// the compare-and-set, not a reconcile pass under any binding, at any age.
/// Once the model is deleted a real reconcile pass reaps the bundle and
/// retires its row.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_referenced_artifact_is_unreclaimable_until_its_model_is_deleted(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let cat_a = Arc::new(catalog.pinned_to_tenant(Some(tenant_a())));
    let store_a = store_over(dir.path(), &cat_a);
    let store_unbound = store_over(dir.path(), &catalog);
    let (job_id, attempt) = running_fine_tune_job(&cat_a, WORKER, None).await;
    let name = output_name(&job_id);
    let artifact = finalize_served(&store_a, &cat_a, &job_id, attempt, &name).await;
    backdate_artifact(&catalog, &artifact, Duration::from_secs(7 * 86_400)).await;

    assert!(matches!(
        cat_a.begin_artifact_reclaim(&artifact).await.unwrap(),
        ReclaimDecision::Referenced
    ));
    assert!(matches!(
        TenantBinding::admin_scope(catalog.begin_artifact_reclaim(&artifact))
            .await
            .unwrap(),
        ReclaimDecision::Referenced
    ));

    let apply = ReconcileOptions {
        apply: true,
        grace: Duration::from_secs(3600),
    };
    let passes = [
        store_a.reconcile(apply).await.unwrap(),
        store_unbound.reconcile(apply).await.unwrap(),
        store_unbound.reconcile_all(apply).await.unwrap(),
    ];
    for report in &passes {
        assert!(report.orphans.is_empty(), "{report:?}");
        assert!(report.damaged.is_empty(), "{report:?}");
        assert_eq!(report.bytes_reclaimed, 0, "{report:?}");
    }
    assert_eq!(files_in(&bundle_dir(&artifact)), BUNDLE);
    assert_eq!(
        state_of(&catalog, &artifact).await,
        Some(ArtifactState::Published)
    );
    store_a
        .artifact_store()
        .fetch_artifact(artifact.url())
        .await
        .unwrap();

    // Delete the model: the artifact is unreferenced, and the owning tenant's
    // pass reclaims it — dry-run and apply agreeing on what.
    cat_a.delete_model(&name, Some(1), false, 0).await.unwrap();
    let unbound = store_unbound.reconcile(apply).await.unwrap();
    assert!(
        unbound.orphans.is_empty(),
        "an unbound pass never reclaims a tenant's artifact: {unbound:?}"
    );
    let preview = store_a
        .reconcile(ReconcileOptions {
            apply: false,
            ..apply
        })
        .await
        .unwrap();
    assert_eq!(files_in(&bundle_dir(&artifact)), BUNDLE);
    let reaped = store_a.reconcile(apply).await.unwrap();
    assert_eq!(reaped.orphans, preview.orphans);
    assert_eq!(reaped.bytes_reclaimed, preview.bytes_reclaimed);
    assert_eq!(reaped.orphans.len(), BUNDLE.len(), "{reaped:?}");
    assert!(reaped.bytes_reclaimed > 0);
    assert!(files_in(&bundle_dir(&artifact)).is_empty());
    assert_eq!(state_of(&catalog, &artifact).await, None);
}
