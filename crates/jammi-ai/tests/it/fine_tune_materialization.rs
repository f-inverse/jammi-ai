//! The `FineTune` producer's materialization identity and its publish path.
//!
//! Model-level cache reuse (#562, wave-5 F5): `CachePolicy::Use` probes a
//! prior model row by materialization definition and finalizes a SECOND row
//! against its already-published prefix, without training —
//! `JobWorker::train_fine_tune`'s own doc has the probe mechanism;
//! `FineTuneMaterializationOutcome::Reused`'s doc has the N:1 byte-safety
//! argument (I1/I2's guard makes a prefix reachable for as long as EITHER
//! row exists, in any tenant). `CachePolicy::Bypass` (the default) never
//! probes: every `FineTune` run under it computes and owns its own
//! attempt-unique prefix, exactly as before this unit.

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
        },
        cache,
    }
}

/// Submit `spec`, claim it with a fresh [`JobWorker`], and drive it to
/// completion — returning the completed job's own model id, whether ITS OWN
/// `jobs.result` recorded run-metrics, and the result's own `cache_outcome`.
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

/// HEADLINE (#562 item 3, wave-5 F5): `cache = Use` reuses an exact
/// definition match through the REAL submit→claim→run path (never a
/// hand-built fixture). The FIRST submission has no candidate to match, so
/// it MISSES and trains for real (`cache_outcome = "computed"`); the
/// SECOND, textually identical submission HITS — no training happens (no
/// run-metrics recorded), `cache_outcome = "reused:{first_model_id}"`
/// (the wire/server parity claim: `jammi-server`'s
/// `job_status_response_from_record` copies `EngineJobResult::Model`'s
/// `cache_outcome` field into `pb::ModelResult` verbatim, with no
/// transformation — read directly, `crates/jammi-server/src/grpc/job.rs`
/// — so this string IS the wire value, not a proxy for it), and both
/// catalog rows serve the EXACT SAME `artifact_path`.
#[tokio::test(flavor = "multi_thread")]
async fn cache_use_reuses_an_exact_definition_match() {
    let (session, _dir) = session_with_training_data().await;

    let (first_model_id, first_trained, first_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        first_trained,
        "no candidate exists yet: the first Use submission must train for real"
    );
    assert_eq!(first_cache_outcome, "computed");

    let (second_model_id, second_trained, second_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        !second_trained,
        "an exact-definition hit must never train — no run-metrics recorded"
    );
    assert_eq!(second_cache_outcome, format!("reused:{first_model_id}"));
    assert_ne!(
        first_model_id, second_model_id,
        "each submission still gets its own output model_id — only the PREFIX is shared"
    );

    let catalog = session.catalog();
    let first = catalog.get_model(&first_model_id).await.unwrap().unwrap();
    let second = catalog.get_model(&second_model_id).await.unwrap().unwrap();
    assert!(first.artifact_path.is_some());
    assert_eq!(
        first.artifact_path, second.artifact_path,
        "a cache-hit row must name the EXACT SAME prefix the reused row serves"
    );
}

/// #562 item 3's N:1 byte-safety oracle, exercised through the SAME real
/// submit→claim→run path as the headline test above (never a hand-built
/// fixture): after a cache hit, deleting EITHER row leaves the OTHER one
/// loadable and the shared prefix reachable; deleting BOTH makes it
/// reclaimable. Complements (never duplicates) the catalog-level I5 oracle
/// in `crates/jammi-db/tests/it/model_prefix_ownership.rs`, which proves the
/// SAME property against hand-registered rows including the cross-tenant
/// shape — this test's own value is proving the REAL job path produces
/// exactly that shape, not a synthetic stand-in for it.
#[tokio::test(flavor = "multi_thread")]
async fn deleting_one_of_two_reused_rows_leaves_the_other_loadable_and_the_prefix_referenced() {
    let (session, _dir) = session_with_training_data().await;
    let (first_model_id, _, _) = submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    let (second_model_id, _, _) = submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;

    let catalog = session.catalog();
    let prefix_str = catalog
        .get_model(&first_model_id)
        .await
        .unwrap()
        .unwrap()
        .artifact_path
        .unwrap();
    let prefix = jammi_db::storage::StorageUrl::parse(&prefix_str).unwrap();
    let store = session.result_store();

    assert_eq!(
        store.prefix_is_referenced(&prefix).await.unwrap(),
        2,
        "both rows naming the prefix must count"
    );

    // Delete the OWNER (first-registered) row — never refused
    // (`Catalog::delete_model` scans no `models` edge at all): the reuser
    // still loads, and the prefix stays referenced by it alone.
    catalog
        .delete_model(&first_model_id, None, false, 0)
        .await
        .unwrap();
    assert!(
        catalog.get_model(&second_model_id).await.unwrap().is_some(),
        "the surviving reuser row must still load after the owner is gone"
    );
    assert_eq!(store.prefix_is_referenced(&prefix).await.unwrap(), 1);
    assert!(matches!(
        store.delete_unreferenced_prefix(&prefix).await,
        Err(jammi_db::error::JammiError::Storage(
            jammi_db::storage::StorageError::Referenced { count: 1, .. }
        ))
    ));

    // Delete the reuser too — both rows are gone, the prefix is unreferenced.
    catalog
        .delete_model(&second_model_id, None, false, 0)
        .await
        .unwrap();
    assert_eq!(store.prefix_is_referenced(&prefix).await.unwrap(), 0);
    store.delete_unreferenced_prefix(&prefix).await.unwrap();
}

/// Bundle flatness is an ORACLE, not an assumption: the containment-aware
/// predicate `ResultStore::prefix_is_referenced` (a row's `artifact_path` is
/// a FLAT directory of files — checked one level deep, never an arbitrary
/// ancestor) is sound only if every object this worker ever publishes under
/// a `models/**` attempt prefix actually sits either directly IN the
/// attempt directory or directly inside an epoch-checkpoint directory that
/// is its OWN `models` row. Drives a REAL fine-tune run with epoch
/// checkpointing enabled through the worker's own publish path (the same
/// `session_with_training_data`/`tiny_bert_model` fixture every other test
/// in this module uses), then walks the PHYSICAL directory tree on disk
/// under this job's own root (a local file-scheme store, the only scheme
/// these tests run against — [`jammi_test_utils::url_to_path`] is the same
/// helper other integration suites use for exactly this) and asserts every
/// regular file's immediate parent directory is EXACTLY the served model's
/// own `artifact_path` or one of the registered epoch-checkpoint rows'
/// `artifact_path`s — never a directory nested any deeper.
///
/// Mutation (executed and reverted against a live worktree, never shipped):
/// writing one extra object nested a level deeper than the attempt
/// directory (`{attempt}/extra/dir/file`) makes this assertion fail — the
/// oracle is not vacuous.
#[tokio::test(flavor = "multi_thread")]
async fn every_published_object_sits_flat_under_its_own_row() {
    let (session, _dir) = session_with_training_data().await;
    let worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 2,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                // n == epochs: every epoch is retained and gets its OWN row
                // (the shape this oracle must account for).
                keep_last_n_checkpoints: Some(2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let output_name = job.model_id().to_string();
    job.wait().await.unwrap();
    worker
        .stop_and_join()
        .await
        .expect("the embedded worker must join cleanly after the job completes");

    let catalog = session.catalog();
    let served = catalog
        .get_model(&output_name)
        .await
        .unwrap()
        .expect("the served model row must exist");
    let served_prefix = served
        .artifact_path
        .clone()
        .expect("the served model must carry an artifact_path");

    // Every prefix a `models` row is allowed to own bytes under, as a set
    // of PHYSICAL directory paths — the served attempt plus every retained
    // epoch-checkpoint row.
    let mut allowed_parents: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();
    allowed_parents.insert(jammi_test_utils::url_to_path(&served_prefix));
    for epoch in 0..2 {
        let epoch_name = format!("{output_name}:epoch_{epoch}");
        let row = catalog
            .get_model(&epoch_name)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("epoch {epoch} checkpoint row must be registered"));
        let epoch_prefix = row
            .artifact_path
            .clone()
            .unwrap_or_else(|| panic!("epoch {epoch} checkpoint row must carry an artifact_path"));
        allowed_parents.insert(jammi_test_utils::url_to_path(&epoch_prefix));
    }

    // Walk the physical directory tree under this job's own root
    // (`{tenant}/{job_id}/{worker_id}/`) — the served attempt's own parent
    // directory names it exactly.
    let served_dir = jammi_test_utils::url_to_path(&served_prefix);
    let job_root = served_dir
        .parent()
        .expect("the served prefix has a worker-id parent directory")
        .to_path_buf();

    let mut objects_seen = 0usize;
    let mut stack = vec![job_root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                stack.push(path);
                continue;
            }
            objects_seen += 1;
            let parent = path
                .parent()
                .expect("every regular file has a parent directory")
                .to_path_buf();
            assert!(
                allowed_parents.contains(&parent),
                "object {path:?} is nested deeper than any known row's artifact_path \
                 ({allowed_parents:?}); bundle flatness is violated"
            );
        }
    }
    assert!(
        objects_seen > 0,
        "the walk must actually find published objects, or this assertion is vacuous"
    );
}
