//! The `FineTune` producer's materialization identity and its publish path.
//!
//! Model-level cache reuse (`CachePolicy::Use` probing a prior model row by
//! materialization definition and finalizing a second row against its
//! already-published prefix) is not yet supported: `Use` is refused, typed,
//! at submit (`InferenceSession::submit_fine_tune_spec_deduped`), for both
//! the in-process spec-construction path and a spec decoded off the wire —
//! see <https://github.com/f-inverse/jammi-ai/issues/562>. Every `FineTune`
//! run therefore computes and owns its own attempt-unique prefix; no two
//! model rows this suite produces ever share one.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::JobWorker;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::JobResult;
use jammi_ai::model::ModelTask;
use jammi_db::error::JammiError;
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

/// `cache = Use` on `TrainingSpec::FineTune` is refused, typed, at the ONE
/// point every submission path — in-process or decoded off the wire —
/// passes through before any row is written
/// (`InferenceSession::submit_fine_tune_spec_deduped`). Exercises the
/// in-process construction path: [`InferenceSession::submit_fine_tune`]
/// builds the spec directly from a [`jammi_wire::request::FineTuneRequest`],
/// never touching the wire decode.
#[tokio::test(flavor = "multi_thread")]
async fn cache_use_is_refused_at_submit_on_the_embedded_path() {
    let (session, _dir) = session_with_training_data().await;

    let request = jammi_wire::request::FineTuneRequest {
        source: "training".into(),
        base_model: tiny_bert_model(),
        columns: vec![
            "text_a".to_string(),
            "text_b".to_string(),
            "score".to_string(),
        ],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        config: None,
        world_size: None,
        cache: CachePolicy::Use,
    };
    let err = session
        .submit_fine_tune(request)
        .await
        .expect_err("cache = Use must be refused before any row is written");
    assert!(
        matches!(&err, JammiError::Config(msg) if msg.contains("model-level cache reuse is not yet supported")),
        "got {err:?}"
    );

    // The refusal leaves no row behind.
    assert!(
        session.catalog().list_jobs().await.unwrap().is_empty(),
        "a refused submit must never write a `jobs` row"
    );
}

/// [`cache_use_is_refused_at_submit_on_the_embedded_path`]'s peer for the
/// WIRE decode path: a spec decoded off the wire
/// (`jammi_ai::wire::training_spec_from_bytes`, the same seam the gRPC
/// handler and the Python binding both drive) reaches the SAME refusal
/// through [`InferenceSession::run_training_spec`], never a separate
/// wire-only check.
#[tokio::test(flavor = "multi_thread")]
async fn cache_use_is_refused_at_submit_on_a_spec_decoded_off_the_wire() {
    let (session, _dir) = session_with_training_data().await;

    let spec = spec_with_cache(CachePolicy::Use);
    let proto = jammi_ai::wire::training_spec_to_proto(&spec);
    let bytes = prost::Message::encode_to_vec(&proto);
    let decoded = jammi_ai::wire::training_spec_from_bytes(&bytes)
        .expect("a well-formed request decodes: the refusal is not a decode-time one");

    let err = session
        .run_training_spec(decoded)
        .await
        .expect_err("cache = Use must be refused before any row is written");
    assert!(
        matches!(&err, JammiError::Config(msg) if msg.contains("model-level cache reuse is not yet supported")),
        "got {err:?}"
    );
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
