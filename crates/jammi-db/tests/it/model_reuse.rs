//! The reusing finalize ([`Catalog::finish_job_reusing_artifact`]): a job
//! completes by attaching its output model to an artifact another job
//! published under the same definition, in one transaction with the reuse
//! probe and the attempt guard — and what that second reference then means
//! for the bytes.
//!
//! Every test runs real adapter bundles through a real [`ResultStore`] on a
//! `file://` root and real jobs through the queue, so "the bytes are gone"
//! and "the bytes are intact" are read off the filesystem.

use std::sync::Arc;
use std::time::Duration;

use jammi_datafusion::ModelTask;
use jammi_db::catalog::artifact_repo::{ArtifactRef, ReclaimDecision, StagedArtifact};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::jobs_repo::{
    FinishJobReusingArtifactParams, FinishJobWithModelParams, ModelRow, ProducedModel, ReuseFinish,
};
use jammi_db::catalog::model_repo::ModelLocation;
use jammi_db::catalog::status::{ArtifactState, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::store::manifest::{
    ArtifactDigest, ComputeDevice, ComputePrecision, ContentDigest, DefinitionHash, InputAnchor,
    LocalRun, Materialization, MaterializationEnv, ModelIdentity, ModelRun, ProducingDescriptor,
};
use jammi_db::store::{ReconcileOptions, ResultStore};
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::TenantId;
use tempfile::tempdir;
use test_case::test_case;

use crate::common::{
    adapter_files, backdate_artifact, bundle_dir, files_in, queue_session, running_fine_tune_job,
    store_over, BASE_MODEL_ID,
};

const WORKER: &str = "reuse-worker";

fn tenant_a() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f3a".parse().unwrap()
}

fn tenant_b() -> TenantId {
    "01906c83-d4c8-7e10-9c4f-3b6f7c5a8f3b".parse().unwrap()
}

fn output_name(job_id: &str) -> String {
    format!("jammi:fine-tuned:{job_id}")
}

/// The identity of one fine-tune definition: a descriptor and environment
/// that hash to one [`DefinitionHash`], and the anchor set it is produced
/// over. `row_count` varies the definition; `anchors` the inputs.
struct Definition {
    descriptor: ProducingDescriptor,
    env: MaterializationEnv,
    anchors: Vec<InputAnchor>,
}

impl Definition {
    fn new(row_count: u64, anchors: Vec<InputAnchor>) -> Self {
        Self {
            descriptor: ProducingDescriptor::FineTune {
                training_set_definition_hash: "a".repeat(64),
                training_set_artifact_digest: "b".repeat(64),
                training_set_row_count: row_count,
                spec_canonical: r#"{"base_model":"q-base"}"#.into(),
                spec_schema_version: 1,
                base_model_id: "q-base".into(),
                world_size: 1,
                collective: "noop".into(),
                local_ranks: 1,
            },
            env: MaterializationEnv::of_models(
                ComputeDevice::Cpu,
                vec![ModelIdentity {
                    model_id: "q-base".into(),
                    run: ModelRun::Local(LocalRun {
                        backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
                        compute_precision: ComputePrecision::F32,
                        content_digest: ContentDigest("fixture-digest".into()),
                        quantization: None,
                    }),
                }],
            ),
            anchors,
        }
    }

    /// A definition whose one input is named by content — the shape a
    /// fine-tune records: no anchor at all, the training set's digest folded
    /// into the descriptor.
    fn pinned(row_count: u64) -> Self {
        Self::new(row_count, Vec::new())
    }

    fn hash(&self) -> DefinitionHash {
        jammi_db::store::manifest::MaterializationManifest::definition_of(
            &self.descriptor,
            &self.env,
        )
        .unwrap()
    }
}

fn model_row(name: &str) -> ModelRow<'_> {
    ModelRow {
        model_id: name,
        version: 1,
        model_type: "fine-tuned",
        backend: jammi_db::catalog::model_repo::ModelBackendKind::Candle,
        task: ModelTask::TextEmbedding,
        base_model_id: Some(BASE_MODEL_ID),
        config_json: None,
    }
}

/// Stage a served bundle for `job_id`'s attempt, attest it under
/// `definition`, and finalize it as the job's output: the trained model
/// whose artifact later jobs reuse.
async fn train_and_publish(
    store: &ResultStore,
    catalog: &Catalog,
    job_id: &str,
    attempt: u32,
    definition: &Definition,
) -> ArtifactRef {
    let staged: StagedArtifact = store
        .artifact_store()
        .stage_attempt_artifact(catalog, job_id, WORKER, attempt, &adapter_files(job_id))
        .await
        .unwrap();
    let manifest = store
        .artifact_store()
        .write_model_materialization(
            &staged,
            Materialization::new(
                &definition.descriptor,
                &definition.env,
                definition.anchors.clone(),
            ),
        )
        .await
        .unwrap();
    let artifact = staged.artifact().clone();
    let name = output_name(job_id);
    let won = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: ProducedModel {
                row: model_row(&name),
                artifact: staged,
                materialization: Some(jammi_db::catalog::artifact_repo::MaterializationSummary {
                    definition_hash: manifest.definition_hash.as_str().to_string(),
                    input_anchors_json: serde_json::to_string(&manifest.input_anchors).unwrap(),
                }),
            },
            epoch_checkpoints: Vec::new(),
        })
        .await
        .unwrap();
    assert!(won.is_some());
    artifact
}

/// The reusing finalize for `job_id`'s attempt under `definition`, with the
/// payload naming the reused artifact the way a worker's would.
async fn reuse(
    catalog: &Catalog,
    job_id: &str,
    attempt: u32,
    definition: &Definition,
) -> ReuseFinish {
    let name = output_name(job_id);
    catalog
        .finish_job_reusing_artifact(FinishJobReusingArtifactParams {
            job_id,
            instance_id: WORKER,
            attempts: attempt,
            definition_hash: &definition.hash(),
            inputs: &definition.anchors,
            output: model_row(&name),
            result: Arc::new(|artifact| Ok(format!(r#"{{"reused":"{artifact}"}}"#))),
        })
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

async fn location_of(catalog: &Catalog, name: &str) -> Option<ModelLocation> {
    catalog
        .get_model(name)
        .await
        .unwrap()
        .and_then(|m| m.location)
}

/// A reaping pass; the artifacts it must reap are backdated past this grace.
fn reap() -> ReconcileOptions {
    ReconcileOptions {
        apply: true,
        grace: Duration::from_secs(3600),
    }
}

/// A hit is one write: the job completes with the payload naming the reused
/// artifact, and its output row references that artifact — the newest of
/// the published matches. Both rows then load the same bundle, and the
/// artifact carries no trace of which job produced it.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_hit_completes_the_job_against_the_newest_published_match(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let definition = Definition::pinned(128);

    let (older_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let older = train_and_publish(&store, &catalog, &older_job, attempt, &definition).await;
    backdate_artifact(&catalog, &older, Duration::from_secs(60)).await;
    let (newer_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let newer = train_and_publish(&store, &catalog, &newer_job, attempt, &definition).await;

    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    assert!(matches!(
        reuse(&catalog, &job_id, attempt, &definition).await,
        ReuseFinish::Reused { artifact, .. } if artifact == newer
    ));

    let job = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(job.status, JobStatus::Completed.to_string());
    assert_eq!(
        job.result.as_deref(),
        Some(format!(r#"{{"reused":"{newer}"}}"#).as_str())
    );
    assert_eq!(
        location_of(&catalog, &output_name(&job_id)).await,
        Some(ModelLocation::Artifact(newer.clone()))
    );
    assert_eq!(
        location_of(&catalog, &output_name(&newer_job)).await,
        Some(ModelLocation::Artifact(newer.clone()))
    );
    assert_ne!(older, newer);
    for artifact in [&older, &newer] {
        store
            .artifact_store()
            .fetch_artifact(artifact.url())
            .await
            .unwrap();
    }
}

/// Every miss writes nothing: a definition nobody published, a request with
/// an unpinned anchor (never a match, whatever is recorded), a matching
/// artifact that is still `staged`, and one already `reclaiming`. The job
/// stays `running` for the caller to train.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_miss_writes_nothing(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let published = Definition::pinned(128);
    let (trained_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let artifact = train_and_publish(&store, &catalog, &trained_job, attempt, &published).await;

    let (job_id, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let assert_untouched = |label: &'static str| {
        let catalog = Arc::clone(&catalog);
        let job_id = job_id.clone();
        async move {
            let job = catalog.get_job(&job_id).await.unwrap();
            assert_eq!(job.status, JobStatus::Running.to_string(), "{label}");
            assert!(job.result.is_none(), "{label}");
            assert!(
                catalog
                    .get_model(&output_name(&job_id))
                    .await
                    .unwrap()
                    .is_none(),
                "{label}"
            );
        }
    };

    let other_definition = Definition::pinned(129);
    assert_eq!(
        reuse(&catalog, &job_id, attempt, &other_definition).await,
        ReuseFinish::Miss
    );
    assert_untouched("another definition").await;

    let unpinned = Definition::new(
        128,
        vec![InputAnchor::unpinned_at_instant(
            "training",
            "2026-09-19T00:00:00Z",
        )],
    );
    assert_eq!(unpinned.hash(), published.hash());
    assert_eq!(
        reuse(&catalog, &job_id, attempt, &unpinned).await,
        ReuseFinish::Miss
    );
    assert_untouched("an unpinned request").await;

    // A recorded anchor set that differs from the request is a miss too.
    let anchored = Definition::new(
        128,
        vec![InputAnchor::result_digest(
            "training-set",
            &ArtifactDigest("c".repeat(64)),
        )],
    );
    assert_eq!(
        reuse(&catalog, &job_id, attempt, &anchored).await,
        ReuseFinish::Miss
    );
    assert_untouched("a different anchor set").await;

    // The matching artifact, no longer published: reclaiming.
    catalog
        .delete_model(&output_name(&trained_job), None, false, 0)
        .await
        .unwrap();
    assert!(matches!(
        catalog.begin_artifact_reclaim(&artifact).await.unwrap(),
        ReclaimDecision::Licensed(_)
    ));
    assert_eq!(
        state_of(&catalog, &artifact).await,
        Some(ArtifactState::Reclaiming)
    );
    assert_eq!(
        reuse(&catalog, &job_id, attempt, &published).await,
        ReuseFinish::Miss
    );
    assert_untouched("a reclaiming artifact").await;

    // A matching bundle still being written: staged, by a live job.
    let (staging_job, staging_attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let staged = store
        .artifact_store()
        .stage_attempt_artifact(
            &catalog,
            &staging_job,
            WORKER,
            staging_attempt,
            &adapter_files(&staging_job),
        )
        .await
        .unwrap();
    store
        .artifact_store()
        .write_model_materialization(
            &staged,
            Materialization::new(&published.descriptor, &published.env, Vec::new()),
        )
        .await
        .unwrap();
    assert_eq!(
        reuse(&catalog, &job_id, attempt, &published).await,
        ReuseFinish::Miss
    );
    assert_untouched("a staged artifact").await;
}

/// A hit whose attempt guard misses — the lease moved to a successor —
/// writes nothing: no job status, no `models` row, no second reference. The
/// published artifact and its bytes are untouched, and the successor's own
/// reuse then attaches.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_lost_lease_hit_writes_nothing_and_leaves_the_bytes_intact(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let definition = Definition::pinned(128);
    let (trained_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let artifact = train_and_publish(&store, &catalog, &trained_job, attempt, &definition).await;
    let bundle_before = files_in(&bundle_dir(&artifact));

    let (job_id, stale_attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let moved = catalog
        .transfer_claim(
            &job_id,
            WORKER,
            "successor",
            stale_attempt,
            Duration::from_secs(3600),
        )
        .await
        .unwrap();
    assert!(moved, "the lease moves to the successor");

    assert_eq!(
        reuse(&catalog, &job_id, stale_attempt, &definition).await,
        ReuseFinish::LostLease
    );
    let job = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(job.status, JobStatus::Running.to_string());
    assert_eq!(job.claimed_by.as_deref(), Some("successor"));
    assert!(catalog
        .get_model(&output_name(&job_id))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        state_of(&catalog, &artifact).await,
        Some(ArtifactState::Published)
    );
    assert_eq!(files_in(&bundle_dir(&artifact)), bundle_before);

    let name = output_name(&job_id);
    let successor = catalog
        .finish_job_reusing_artifact(FinishJobReusingArtifactParams {
            job_id: &job_id,
            instance_id: "successor",
            attempts: stale_attempt,
            definition_hash: &definition.hash(),
            inputs: &definition.anchors,
            output: model_row(&name),
            result: Arc::new(|artifact| Ok(format!(r#"{{"reused":"{artifact}"}}"#))),
        })
        .await
        .unwrap();
    assert!(
        matches!(successor, ReuseFinish::Reused { artifact: reused, .. } if reused == artifact)
    );
    assert_eq!(
        location_of(&catalog, &name).await,
        Some(ModelLocation::Artifact(artifact))
    );
}

/// The probe reads the caller's own tenant and the global scope: a peer
/// tenant's artifact is never a hit, a global one is. And a reference from
/// the reusing tenant keeps a global artifact's bytes exactly as the
/// producer's own reference would — under the owning scope, under admin
/// scope, and against a reconcile pass — until that reference is gone.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn a_hit_is_scoped_to_the_own_tenant_and_global_and_its_reference_guards_the_reap(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let cat_a = Arc::new(catalog.pinned_to_tenant(Some(tenant_a())));
    let cat_b = Arc::new(catalog.pinned_to_tenant(Some(tenant_b())));
    let store_global = store_over(dir.path(), &catalog);
    let store_b = store_over(dir.path(), &cat_b);
    let definition = Definition::pinned(128);

    // Tenant B trains one; tenant A sees nothing to reuse.
    let (b_job, attempt) = running_fine_tune_job(&cat_b, WORKER, None).await;
    let b_artifact = train_and_publish(&store_b, &cat_b, &b_job, attempt, &definition).await;
    let (a_job, a_attempt) = running_fine_tune_job(&cat_a, WORKER, None).await;
    assert_eq!(
        reuse(&cat_a, &a_job, a_attempt, &definition).await,
        ReuseFinish::Miss
    );

    // A global one is published; tenant A's job reuses it.
    let (global_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let global =
        train_and_publish(&store_global, &catalog, &global_job, attempt, &definition).await;
    assert!(matches!(
        reuse(&cat_a, &a_job, a_attempt, &definition).await,
        ReuseFinish::Reused { artifact, .. } if artifact == global
    ));
    assert_ne!(global, b_artifact);
    let a_name = output_name(&a_job);
    assert_eq!(
        location_of(&cat_a, &a_name).await,
        Some(ModelLocation::Artifact(global.clone()))
    );

    // The producer's row goes; tenant A's reference alone keeps the bytes.
    catalog
        .delete_model(&output_name(&global_job), None, false, 0)
        .await
        .unwrap();
    backdate_artifact(&catalog, &global, Duration::from_secs(7 * 86_400)).await;
    assert!(matches!(
        catalog.begin_artifact_reclaim(&global).await.unwrap(),
        ReclaimDecision::Referenced
    ));
    assert!(matches!(
        TenantBinding::admin_scope(catalog.begin_artifact_reclaim(&global))
            .await
            .unwrap(),
        ReclaimDecision::Referenced
    ));
    let held = store_global.reconcile(reap()).await.unwrap();
    assert!(held.orphans.is_empty(), "{held:?}");
    assert_eq!(held.bytes_reclaimed, 0, "{held:?}");
    assert_eq!(
        state_of(&catalog, &global).await,
        Some(ArtifactState::Published)
    );
    store_global
        .artifact_store()
        .fetch_artifact(global.url())
        .await
        .unwrap();

    // Once tenant A's row is gone too, the owning (global) pass reaps.
    cat_a.delete_model(&a_name, None, false, 0).await.unwrap();
    let reaped = store_global.reconcile(reap()).await.unwrap();
    assert!(reaped.bytes_reclaimed > 0, "{reaped:?}");
    assert!(files_in(&bundle_dir(&global)).is_empty());
    assert_eq!(state_of(&catalog, &global).await, None);
    // Tenant B's own artifact, still referenced, was never touched.
    assert_eq!(
        state_of(&catalog, &b_artifact).await,
        Some(ArtifactState::Published)
    );
}

/// Two rows, one artifact: whichever row is deleted first, the survivor
/// still loads the bundle, and the bytes are reaped only once both are
/// gone.
#[test_case(BackendKind::Sqlite, true ; "sqlite, the producer's row first")]
#[test_case(BackendKind::Sqlite, false ; "sqlite, the reuser's row first")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres, true ; "postgres, the producer's row first")
)]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres, false ; "postgres, the reuser's row first")
)]
#[tokio::test]
async fn deleting_either_row_leaves_the_other_loadable_and_the_reap_waits_for_both(
    backend: BackendKind,
    producer_first: bool,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);
    let definition = Definition::pinned(128);

    let (producer_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    let artifact = train_and_publish(&store, &catalog, &producer_job, attempt, &definition).await;
    let (reuser_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
    assert!(matches!(
        reuse(&catalog, &reuser_job, attempt, &definition).await,
        ReuseFinish::Reused { artifact: reused, .. } if reused == artifact
    ));
    backdate_artifact(&catalog, &artifact, Duration::from_secs(7 * 86_400)).await;
    let producer = output_name(&producer_job);
    let reuser = output_name(&reuser_job);
    let (first, survivor) = if producer_first {
        (producer, reuser)
    } else {
        (reuser, producer)
    };

    catalog.delete_model(&first, None, false, 0).await.unwrap();
    assert_eq!(
        location_of(&catalog, &survivor).await,
        Some(ModelLocation::Artifact(artifact.clone()))
    );
    let held = store.reconcile(reap()).await.unwrap();
    assert_eq!(held.bytes_reclaimed, 0, "{held:?}");
    store
        .artifact_store()
        .fetch_artifact(artifact.url())
        .await
        .unwrap();

    catalog
        .delete_model(&survivor, None, false, 0)
        .await
        .unwrap();
    let reaped = store.reconcile(reap()).await.unwrap();
    assert!(reaped.bytes_reclaimed > 0, "{reaped:?}");
    assert!(files_in(&bundle_dir(&artifact)).is_empty());
    assert_eq!(state_of(&catalog, &artifact).await, None);
}

/// The attach and the reclaim compare-and-set conflict: raced from two
/// tasks over an artifact whose last reference was just deleted, exactly one
/// of them wins each round. A hit that attaches leaves the reclaim
/// `Referenced`; a reclaim that licenses leaves the probe a miss (the
/// artifact is `reclaiming`). Never both, in any of the rounds — a reference
/// to licensed bytes would be a row pointing at a bundle about to vanish.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_attach_and_a_reclaim_over_the_same_artifact_never_both_win(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_session(backend, dir.path()).await;
    let store = store_over(dir.path(), &catalog);

    let mut attached = 0usize;
    let mut licensed = 0usize;
    for round in 0..24 {
        // One definition per round: an earlier round's artifact that a hit
        // attached to stays published, and must not be the match this
        // round's probe finds.
        let definition = Arc::new(Definition::pinned(1_000 + round));
        let (producer_job, attempt) = running_fine_tune_job(&catalog, WORKER, None).await;
        let artifact =
            train_and_publish(&store, &catalog, &producer_job, attempt, &definition).await;
        catalog
            .delete_model(&output_name(&producer_job), None, false, 0)
            .await
            .unwrap();
        let (reuser_job, reuser_attempt) = running_fine_tune_job(&catalog, WORKER, None).await;

        let attach = {
            let catalog = Arc::clone(&catalog);
            let definition = Arc::clone(&definition);
            let reuser_job = reuser_job.clone();
            tokio::spawn(
                async move { reuse(&catalog, &reuser_job, reuser_attempt, &definition).await },
            )
        };
        let reclaim = {
            let catalog = Arc::clone(&catalog);
            let artifact = artifact.clone();
            tokio::spawn(async move { catalog.begin_artifact_reclaim(&artifact).await.unwrap() })
        };
        let (attach, reclaim) = (attach.await.unwrap(), reclaim.await.unwrap());

        match (attach, reclaim) {
            (
                ReuseFinish::Reused {
                    artifact: reused, ..
                },
                ReclaimDecision::Referenced,
            ) => {
                assert_eq!(reused, artifact, "round {round}");
                assert_eq!(
                    location_of(&catalog, &output_name(&reuser_job)).await,
                    Some(ModelLocation::Artifact(artifact.clone())),
                    "round {round}"
                );
                assert_eq!(
                    state_of(&catalog, &artifact).await,
                    Some(ArtifactState::Published),
                    "round {round}"
                );
                attached += 1;
            }
            (ReuseFinish::Miss, ReclaimDecision::Licensed(licence)) => {
                assert!(
                    catalog
                        .get_model(&output_name(&reuser_job))
                        .await
                        .unwrap()
                        .is_none(),
                    "round {round}"
                );
                assert!(
                    !catalog
                        .model_artifact_is_referenced(&artifact)
                        .await
                        .unwrap(),
                    "round {round}"
                );
                store
                    .artifact_store()
                    .reclaim(&catalog, licence, &[])
                    .await
                    .unwrap();
                licensed += 1;
            }
            other => panic!("round {round}: attach and reclaim both won, or neither: {other:?}"),
        }
    }
    assert_eq!(attached + licensed, 24);
}
