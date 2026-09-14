//! U3 (#500): the `FineTune` producer's model-level cache reuse —
//! acceptance (a). Two submissions of the SAME `TrainingSpec::FineTune`
//! (same source, columns, method, task, base model, config, world size) with
//! `CachePolicy::Use` train ONCE: the second submission's `train_fine_tune`
//! probes `probe_model_by_definition` BEFORE spawning the blocking trainer,
//! hits, and completes by registering its OWN model name pointing at the
//! FIRST submission's already-published prefix — two model rows sharing one
//! prefix.
//!
//! # "Trains once" is asserted structurally, not by wall-clock — and not by
//! the global thread-finished counter either
//!
//! [`jammi_ai::fine_tune::worker::training_test_hooks::training_threads_finished`]
//! is a single PROCESS-WIDE atomic shared by every test in this binary; under
//! `cargo test`'s default parallel runner it is polluted by every OTHER
//! concurrently-running fine-tune test that also trains (confirmed empirically:
//! an exact before/after delta around this test's own first submission was
//! observed to move by more than one). The existing call sites of this hook
//! (`jobs_shutdown.rs`) only ever poll it as an inequality ("has AT LEAST one
//! more thread finished than before"), never an exact delta, for exactly this
//! reason. This suite instead asserts "trains once" via two structural
//! consequences a cache HIT (and only a cache hit) produces, neither racy:
//!
//! - **The published prefix.** `ArtifactStore::put_artifact`'s prefix is
//!   `{tenant}/{job_id}/{worker_id}/{attempt}` — unique by `job_id` (a fresh
//!   UUID per submission) BY CONSTRUCTION. Two independently-trained runs can
//!   therefore never coincidentally publish under the same prefix; observing
//!   the second job's `artifact_path` equal to the first's is possible ONLY
//!   via `FineTuneMaterializationOutcome::Reused` — i.e., only if training
//!   was skipped.
//! - **Recorded metrics.** `Reused` hands `publish_and_finalize` a
//!   `TrainedArtifact` with `metrics: None` (no training loop ever ran to
//!   produce any); a FRESH run always records `Some(..)` run-metrics JSON.
//!   The second job's own `jobs.result` therefore carries `metrics: None`
//!   while the first's carries `Some`.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::JobWorker;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::JobResult;
use jammi_ai::model::ModelTask;
use jammi_db::store::CachePolicy;

use crate::fine_tune::{session_with_training_data, tiny_bert_model};

/// A `TrainingSpec::FineTune` over the shared `training` fixture, small
/// enough to train in milliseconds on CPU, carrying `cache`.
fn spec_with_cache(cache: CachePolicy) -> TrainingSpec {
    TrainingSpec::FineTune {
        source: "training".into(),
        columns: vec![
            "text_a".to_string(),
            "text_b".to_string(),
            "score".to_string(),
        ],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            },
            world_size: jammi_ai::fine_tune::spec::DEFAULT_WORLD_SIZE,
            cache,
        },
    }
}

/// Submit `spec`, claim it with a fresh [`JobWorker`], and drive it to
/// completion — returning the completed job's own model id, whether ITS OWN
/// `jobs.result` recorded run-metrics (`Some` only for a run that actually
/// trained; `None` for a `Reused` cache hit — see this module's doc), and
/// the result's own `cache_outcome` (P6: asserted directly by the callers
/// below rather than inferred from the metrics flag alone).
async fn submit_and_run(
    session: &Arc<jammi_ai::session::InferenceSession>,
    spec: TrainingSpec,
) -> (String, bool, String) {
    let job = session.run_training_spec(spec).await.unwrap();
    let worker = JobWorker::new(session).expect("default worker intervals are valid");
    let claimed = session
        .catalog()
        .claim_next(
            worker.worker_id(),
            &["fine_tune"],
            std::time::Duration::from_secs(3600),
        )
        .await
        .unwrap()
        .expect("the queued job is claimable");
    worker.run_claimed_job(session, claimed).await;
    let after = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(
        after.status, "completed",
        "the job must complete for either outcome (fresh train or cache hit): {after:?}"
    );
    let result: JobResult = serde_json::from_str(
        after
            .result
            .as_deref()
            .expect("a completed job has a result"),
    )
    .expect("a fine-tune job's result is a JobResult::Model");
    let JobResult::Model {
        metrics,
        cache_outcome,
        ..
    } = result
    else {
        panic!("a training kind's result must be JobResult::Model, got {result:?}");
    };
    (job.model_id.clone(), metrics.is_some(), cache_outcome)
}

#[tokio::test(flavor = "multi_thread")]
async fn cache_use_trains_once_and_shares_one_prefix_across_two_model_rows() {
    let (session, _dir) = session_with_training_data().await;

    // The FIRST submission is honestly always a miss (nothing to reuse yet):
    // it must train for real — its own result carries metrics and reports
    // `cache_outcome: "computed"` (P6: asserted directly, not inferred).
    let (first_model_id, first_trained, first_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        first_trained,
        "the first submission has nothing to reuse and must train for real"
    );
    assert_eq!(first_cache_outcome, "computed");

    // The SECOND submission (same spec, same training-set digest) must be a
    // cache HIT: its own result carries no metrics, and its `cache_outcome`
    // names the FIRST submission's own model id (P6) — the trainer never ran.
    let (second_model_id, second_trained, second_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        !second_trained,
        "the second submission (an exact definition-hash + anchors match) must be a cache HIT: \
         the trainer must not run a second time"
    );
    assert_eq!(
        second_cache_outcome,
        format!("reused:{first_model_id}"),
        "a cache hit's own result must OBSERVABLY name the reused model, never merely be \
         inferred from the absent metrics field"
    );

    assert_ne!(
        first_model_id, second_model_id,
        "each submission completes under its OWN model name (two rows), never re-using the \
         first job's name"
    );

    let catalog = session.catalog();
    let first = catalog
        .get_model(&first_model_id)
        .await
        .unwrap()
        .expect("the first job's model row must exist");
    let second = catalog
        .get_model(&second_model_id)
        .await
        .unwrap()
        .expect("the second (cache-hit) job's model row must exist");

    // Two model rows, ONE prefix — possible ONLY via `Reused` (see module
    // doc: a fresh run's prefix is unique by `job_id` by construction).
    assert!(first.artifact_path.is_some());
    assert_eq!(
        first.artifact_path, second.artifact_path,
        "a cache hit registers its own name pointing at the SAME prefix, never a new one"
    );
    // Both rows carry the SAME materialization identity — the reuse key the
    // probe matched on.
    assert!(first.definition_hash.is_some());
    assert_eq!(first.definition_hash, second.definition_hash);
    assert_eq!(first.input_anchors_json, second.input_anchors_json);
    // P5 (fix round 1): the `FineTune` materialization records NO input
    // anchor — the training-set digest it would otherwise have carried is
    // already inside `definition_hash` (`ProducingDescriptor::FineTune::
    // training_set_artifact_digest`), so a separate anchor was redundant,
    // and pairing it with the fine-tune's own registered SOURCE name (a
    // long-lived, mutable relation, never the ephemeral training-set table
    // the digest actually names) was a false attestation. Pinned directly,
    // not merely "equal to each other": both rows must carry the empty set.
    assert_eq!(first.input_anchors_json.as_deref(), Some("[]"));
    // `manifest_path` is not a `models` column (P7, migration 033 rewritten
    // before merge): the sidecar path is always DERIVED from `artifact_path`
    // (`ArtifactStore::read_model_materialization`'s own doc), so the two
    // rows agreeing on `artifact_path` above already implies they agree on
    // the derived sidecar path — there is no separate column left to compare.

    // Deleting one row leaves the prefix and the other model still resolves.
    catalog
        .delete_model(&first_model_id, None, false, 0)
        .await
        .unwrap();
    let second_after_delete = catalog
        .get_model(&second_model_id)
        .await
        .unwrap()
        .expect("deleting the FIRST row must not affect the second");
    assert_eq!(second_after_delete.artifact_path, second.artifact_path);
}

/// `CachePolicy::Bypass` (the default) never probes: two submissions of the
/// identical spec both train for real, each under its own name, and (being
/// real, independent trainer runs) do NOT share a prefix.
#[tokio::test(flavor = "multi_thread")]
async fn cache_bypass_never_reuses() {
    let (session, _dir) = session_with_training_data().await;

    let (first_model_id, first_trained, first_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Bypass)).await;
    let (second_model_id, second_trained, second_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Bypass)).await;

    assert!(
        first_trained,
        "Bypass never probes: the first run must train"
    );
    assert!(
        second_trained,
        "Bypass never probes: the second run must train too, never short-circuiting"
    );
    assert_eq!(first_cache_outcome, "computed");
    assert_eq!(second_cache_outcome, "computed");

    let catalog = session.catalog();
    let first = catalog.get_model(&first_model_id).await.unwrap().unwrap();
    let second = catalog.get_model(&second_model_id).await.unwrap().unwrap();
    assert_ne!(
        first.artifact_path, second.artifact_path,
        "two independent Bypass runs must never share a prefix"
    );
}

/// P1' (fix round 1, BLOCK #1 finding F1): a cache HIT whose finalize CAS
/// loses the lease race must never delete the REUSED prefix — that prefix is
/// a DIFFERENT, already-servable model's committed artifact, not this
/// attempt's own bytes. Drives the real worker exactly like
/// `fine_tune::worker_that_lost_lease_does_not_finalize`: the SECOND
/// (cache-hit) submission's claim is stolen by a re-claiming worker before
/// the stale claim's own `run_claimed_job` reaches finalize, so its CAS
/// necessarily loses — `PublishedPrefix::delete_if_owned`'s own guard is the
/// only thing standing between that loss and the FIRST job's prefix being
/// deleted out from under it.
#[tokio::test(flavor = "multi_thread")]
async fn a_lost_lease_on_a_cache_hit_never_deletes_the_reused_prefix() {
    use jammi_ai::fine_tune::worker::JobWorker;
    use std::time::Duration;

    let (session, _dir) = session_with_training_data().await;

    // The FIRST submission trains for real; the SECOND will reuse its prefix.
    let (first_model_id, first_trained, _) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(first_trained);
    let first_before = session
        .catalog()
        .get_model(&first_model_id)
        .await
        .unwrap()
        .expect("the first job's model row must exist");
    let reused_prefix = first_before
        .artifact_path
        .clone()
        .expect("the first job's model must be servable");

    // The SECOND submission (same spec) is queued but not yet claimed.
    let job = session
        .run_training_spec(spec_with_cache(CachePolicy::Use))
        .await
        .unwrap();

    let worker_a = JobWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = JobWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with a zero (already-expired) lease.
    let stale_claim = session
        .catalog()
        .claim_next(worker_a.worker_id(), &["fine_tune"], Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued cache=Use job");

    // worker-b reclaims the expired lease and re-claims under a long one:
    // worker-b now owns the job.
    let actioned = session
        .catalog()
        .reclaim_expired_jobs(Duration::from_secs(60), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned = session
        .catalog()
        .claim_next(
            worker_b.worker_id(),
            &["fine_tune"],
            Duration::from_secs(3600),
        )
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");

    // worker-a runs its STALE claim to completion: this is a `Reused`
    // outcome (a cache HIT, no training), so it reaches
    // `publish_and_finalize` almost immediately — its finalize CAS then
    // loses to worker-b.
    worker_a.run_claimed_job(&session, stale_claim).await;

    let after_a = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(
        after_a.status, "running",
        "a worker that lost its lease must not finalize, even on a cache hit"
    );

    // THE PROPERTY: the FIRST job's model — the one the lost attempt's
    // `Reused` prefix pointed at — is STILL SERVABLE. Pre-fix, worker-a's
    // abort path unconditionally deleted `prefix` (the SAME reused prefix,
    // since `dir` is `None` for a `Reused` outcome), destroying the first
    // model's own committed bytes out from under it.
    let first_after = session
        .catalog()
        .get_model(&first_model_id)
        .await
        .unwrap()
        .expect("the reused model's row must still exist");
    assert_eq!(
        first_after.artifact_path.as_deref(),
        Some(reused_prefix.as_str()),
        "a lost-lease abort on a cache hit must never delete or unpoint the REUSED model's own \
         prefix"
    );
    // The DB column alone is not load-bearing (a mutated `delete_if_owned`
    // that deletes the OBJECT STORE bytes unconditionally would leave this
    // column untouched — it is the artifact store's own prefix, not the
    // catalog row, `delete_artifact_prefix` reaches). Fetch the actual
    // bytes: `ArtifactStore::fetch_artifact` fails with `StorageError::
    // NotPublished` when the manifest is gone (module doc), so a
    // successful fetch here is the one assertion this test cannot pass
    // vacuously.
    let reused_prefix_url = jammi_db::storage::StorageUrl::parse(&reused_prefix)
        .expect("the recorded artifact_path must be a valid StorageUrl");
    session
        .artifact_store()
        .fetch_artifact(&reused_prefix_url)
        .await
        .expect(
            "the reused prefix's bytes must still be fetchable after a lost-lease abort on the \
             cache-hit attempt that pointed at them",
        );

    // The legitimate owner (worker-b) still finalizes correctly against the
    // SAME reused prefix.
    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();
    let after_b = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(after_b.status, "completed");
    let second = session
        .catalog()
        .get_model(&job.model_id)
        .await
        .unwrap()
        .expect("the legitimate owner's model row must exist");
    assert_eq!(
        second.artifact_path.as_deref(),
        Some(reused_prefix.as_str())
    );
}
