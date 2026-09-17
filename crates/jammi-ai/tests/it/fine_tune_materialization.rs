//! The `FineTune` producer's materialization identity and its publish path.
//!
//! `CachePolicy::Use` is refused, typed, by `admit_training_spec` — model-level
//! cache reuse is not yet supported (see
//! <https://github.com/f-inverse/jammi-ai/issues/562>). Every `FineTune` run
//! computes and owns its own attempt-unique prefix.

use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;

use crate::fine_tune::{session_with_training_data, tiny_bert_model};

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
