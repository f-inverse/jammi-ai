//! The `FineTune` producer's materialization identity, its publish path, and
//! the model-level reuse that identity enables: under `CachePolicy::Use` a
//! second submission of the same definition completes against the first
//! run's published artifact instead of training — two model rows, one
//! artifact — and the bytes outlive either row while the other references
//! them.
//!
//! # "Trains once" is asserted structurally, never by wall-clock
//!
//! A worker's own artifact prefix is `{tenant}/{job_id}/{worker_id}/{attempt}`,
//! unique per submission by construction, so two rows referencing ONE
//! artifact is possible only through the reusing finalize. A reused run
//! also records no run-metrics (no training loop ran to produce any) and a
//! `cache_outcome` naming the artifact it reused.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::JobWorker;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::JobResult;
use jammi_ai::model::ModelTask;
use jammi_db::catalog::model_repo::ModelLocation;
use jammi_db::catalog::status::ArtifactState;
use jammi_db::store::{CacheOutcome, CachePolicy, ReconcileOptions, ReusedArtifact};
use std::time::Duration;

use crate::fine_tune::{session_with_training_data, tiny_bert_model};

/// A `TrainingSpec::FineTune` over the shared `training` fixture, small
/// enough to train in milliseconds on CPU, at a given `backbone_dtype` and
/// `cache` policy.
///
/// **Deliberately stays on the PLAIN projection-head arm (`target_modules`
/// empty) for every `backbone_dtype`, including non-`F32`.** `backbone_dtype`
/// only takes effect (`validate_backbone_precision`/`compute_precision_to_dtype`)
/// on the ENCODER-ADAPTERS arm (`target_modules` non-empty — `worker.rs`'s
/// own module doc), so an `F16` leg through THAT arm would genuinely
/// exercise the cast — but it is not the shape this fn uses, for an
/// executed reason: reusing `acceleration_report.rs`'s own
/// `encoder_adapters_config`/`tiny_modernbert` shape for an F16 CPU job
/// that must actually COMPLETE (publish a `materialization.json` this
/// suite reads back) reproducibly diverges here — "Training diverged: loss
/// was NaN or >100 for 3 consecutive batches" — over one epoch on this
/// tiny fixture, with both `CosineDecay` and `Constant` schedules tried.
/// `acceleration_report.rs`'s OWN f16 tests are explicitly "tolerant of
/// EITHER terminal outcome" for exactly this reason (their oracle is the
/// PRE-training acceleration report, written before divergence could ever
/// happen) — a test that needs a PUBLISHED artifact cannot adopt that
/// tolerance, since a failed job publishes nothing to read back at all.
///
/// This is still the right input for THIS suite's own oracle: the kernel-admission-profile fold
/// under test (`render_kernel_admission_profile`'s `dtype` argument) reads
/// `common.config.backbone_dtype` UNCONDITIONALLY of which arm trains — so
/// `backbone_dtype: F16` here is exactly the value production's own fold
/// reads, even on an arm that ignores it for the forward math. What this
/// fn does NOT prove — and does not claim to — is that the encoder-adapters
/// arm's own F16-on-CPU numerics are stable; that is `acceleration_report.rs`'s
/// concern, with its own, correctly-tolerant oracle.
fn spec_with_backbone_dtype(
    cache: CachePolicy,
    backbone_dtype: jammi_numerics::ComputePrecision,
) -> TrainingSpec {
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
                backbone_dtype,
                ..Default::default()
            },
            world_size: jammi_ai::fine_tune::spec::DEFAULT_WORLD_SIZE,
            cache,
        },
    }
}

/// [`spec_with_backbone_dtype`] at `F32` — the spec for every test in this
/// file that does not vary the backbone dtype.
fn spec_with_cache(cache: CachePolicy) -> TrainingSpec {
    spec_with_backbone_dtype(cache, jammi_numerics::ComputePrecision::F32)
}

/// Submit `spec`, claim it with a fresh [`JobWorker`], and drive it to
/// completion — returning the completed job's own model id, whether ITS OWN
/// `jobs.result` recorded run-metrics, and the result's own `cache_outcome`.
async fn submit_and_run(
    session: &Arc<jammi_ai::session::InferenceSession>,
    spec: TrainingSpec,
) -> (String, bool, CacheOutcome) {
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

/// Two submissions of the SAME `TrainingSpec::FineTune` under `Use` train
/// once: the first has nothing to reuse and trains for real; the second's
/// definition hash — the same training-set content, spec, base model and
/// topology — matches the first's published artifact, so it completes with
/// its OWN model name referencing the FIRST run's artifact, no metrics, and
/// a `cache_outcome` naming that artifact. Then the bytes' lifetime: with
/// the producer's row deleted the reuser still loads; with both gone the
/// artifact is unreferenced and a reconcile pass may reap it once aged.
#[tokio::test(flavor = "multi_thread")]
async fn cache_use_trains_once_and_two_rows_share_one_artifact_until_both_are_gone() {
    let (session, _dir) = session_with_training_data().await;

    let (first_model_id, first_trained, first_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        first_trained,
        "the first submission has nothing to reuse and must train for real"
    );
    assert_eq!(first_cache_outcome, CacheOutcome::Computed);

    let (second_model_id, second_trained, second_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(
        !second_trained,
        "the second submission (an exact definition match) must reuse: no training loop runs"
    );
    assert_ne!(
        first_model_id, second_model_id,
        "each submission completes under its OWN model name"
    );

    let catalog = session.catalog();
    let first = catalog.get_model(&first_model_id).await.unwrap().unwrap();
    let second = catalog.get_model(&second_model_id).await.unwrap().unwrap();
    let Some(ModelLocation::Artifact(artifact)) = first.location.clone() else {
        panic!("a trained model references its artifact: {first:?}");
    };
    assert_eq!(
        second.location,
        Some(ModelLocation::Artifact(artifact.clone())),
        "the reuser's row references the SAME artifact the producer's does"
    );
    assert_eq!(
        second_cache_outcome,
        CacheOutcome::Reused(ReusedArtifact::Model(artifact.clone())),
        "a reuse names the artifact it reused on the job's own result"
    );
    let published = catalog
        .get_model_artifact(&artifact)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(published.state, ArtifactState::Published);
    assert_eq!(
        published.input_anchors_json.as_deref(),
        Some("[]"),
        "a fine-tune records the empty anchor set: its one input is named by content"
    );
    // Both rows serve the same bundle.
    let store = session.artifact_store();
    for model in [&first, &second] {
        store
            .fetch_artifact(&crate::common::served_bundle_url(model))
            .await
            .unwrap();
    }

    // The producer's row goes; the reuser's reference keeps the bytes.
    catalog
        .delete_model(&first_model_id, None, false, 0)
        .await
        .unwrap();
    assert_eq!(
        catalog
            .get_model(&second_model_id)
            .await
            .unwrap()
            .unwrap()
            .location,
        Some(ModelLocation::Artifact(artifact.clone()))
    );
    let reap = ReconcileOptions {
        apply: true,
        grace: Duration::from_secs(3600),
    };
    let held = session.result_store().reconcile(reap).await.unwrap();
    assert_eq!(held.bytes_reclaimed, 0, "{held:?}");
    store.fetch_artifact(artifact.url()).await.unwrap();

    // Both gone: the artifact is unreferenced. A reconcile pass still holds
    // it while it is younger than the grace (the only clock the pass reads
    // is the artifact row's own); the reap of an aged, unreferenced artifact
    // is `jammi-db`'s `model_reuse` suite's, over a backdated row.
    catalog
        .delete_model(&second_model_id, None, false, 0)
        .await
        .unwrap();
    assert!(!catalog
        .model_artifact_is_referenced(&artifact)
        .await
        .unwrap());
    let young = session.result_store().reconcile(reap).await.unwrap();
    assert_eq!(
        young.bytes_reclaimed, 0,
        "an unreferenced artifact younger than the grace is left alone: {young:?}"
    );
    assert_eq!(
        catalog
            .get_model_artifact(&artifact)
            .await
            .unwrap()
            .unwrap()
            .state,
        ArtifactState::Published
    );
}

/// A reuse hit whose attempt lost its lease before the reusing finalize
/// writes nothing — no reference, no job status — and the reused model's
/// bytes are intact. Drives the real worker: the second (reusing)
/// submission's claim is stolen by a re-claiming worker before the stale
/// claim reaches its finalize, so its attempt guard necessarily misses. The
/// legitimate owner then reuses the same artifact.
#[tokio::test(flavor = "multi_thread")]
async fn a_lost_lease_on_a_cache_hit_leaves_no_reference_and_the_bytes_intact() {
    use std::time::Duration;

    let (session, _dir) = session_with_training_data().await;

    let (first_model_id, first_trained, _) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    assert!(first_trained);
    let first_before = session
        .catalog()
        .get_model(&first_model_id)
        .await
        .unwrap()
        .expect("the first job's model row must exist");
    let Some(ModelLocation::Artifact(artifact)) = first_before.location.clone() else {
        panic!("a trained model references its artifact: {first_before:?}");
    };

    let job = session
        .run_training_spec(spec_with_cache(CachePolicy::Use))
        .await
        .unwrap();
    let worker_a = JobWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = JobWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with a zero (already-expired) lease; worker-b reclaims
    // the expired lease and re-claims under a long one.
    let stale_claim = session
        .catalog()
        .claim_next(worker_a.worker_id(), &["fine_tune"], Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued cache=Use job");
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

    // worker-a runs its STALE claim: the probe hits, but the attempt guard
    // inside the reusing finalize misses, so nothing is written.
    worker_a.run_claimed_job(&session, stale_claim).await;
    let after_a = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(
        after_a.status, "running",
        "a worker that lost its lease must not finalize, even on a cache hit"
    );
    assert_eq!(after_a.claimed_by.as_deref(), Some(worker_b.worker_id()));
    assert!(
        session
            .catalog()
            .get_model(&job.model_id)
            .await
            .unwrap()
            .is_none(),
        "the lost attempt attached no row"
    );
    let first_after = session
        .catalog()
        .get_model(&first_model_id)
        .await
        .unwrap()
        .expect("the reused model's row must still exist");
    assert_eq!(
        first_after.location,
        Some(ModelLocation::Artifact(artifact.clone()))
    );
    assert_eq!(
        session
            .catalog()
            .get_model_artifact(&artifact)
            .await
            .unwrap()
            .unwrap()
            .state,
        ArtifactState::Published
    );
    session
        .artifact_store()
        .fetch_artifact(artifact.url())
        .await
        .expect("the reused artifact's bytes are intact after the lost-lease attempt");

    // The legitimate owner reuses the same artifact.
    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();
    let second = session
        .catalog()
        .get_model(&job.model_id)
        .await
        .unwrap()
        .expect("the legitimate owner's model row must exist");
    assert_eq!(second.location, Some(ModelLocation::Artifact(artifact)));
}

/// `CachePolicy::Bypass` (the default) never probes: with a published
/// artifact of the identical definition already in the catalog (the first
/// run, under `Use`), a `Bypass` submission still trains for real, under
/// its own name, and does NOT share the artifact.
#[tokio::test(flavor = "multi_thread")]
async fn cache_bypass_never_reuses() {
    let (session, _dir) = session_with_training_data().await;

    let (first_model_id, first_trained, first_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Use)).await;
    let (second_model_id, second_trained, second_cache_outcome) =
        submit_and_run(&session, spec_with_cache(CachePolicy::Bypass)).await;

    assert!(
        first_trained,
        "the first run has nothing to reuse and must train"
    );
    assert!(
        second_trained,
        "Bypass never probes: the second run must train too, never short-circuiting"
    );
    assert_eq!(first_cache_outcome, CacheOutcome::Computed);
    assert_eq!(second_cache_outcome, CacheOutcome::Computed);

    let catalog = session.catalog();
    let first = catalog.get_model(&first_model_id).await.unwrap().unwrap();
    let second = catalog.get_model(&second_model_id).await.unwrap().unwrap();
    assert_ne!(
        crate::common::served_bundle_url(&first),
        crate::common::served_bundle_url(&second),
        "a Bypass run never shares an artifact with an earlier run"
    );
}

/// Bundle flatness is an ORACLE, not an assumption: a reclaim licence covers
/// exactly the keys DIRECTLY inside its artifact's prefix
/// (`ReclaimLicence::covers`), which is sound only if every object this
/// worker ever writes under a `models/**` attempt prefix actually sits
/// either directly IN the attempt directory or directly inside an
/// epoch-checkpoint directory that is its OWN artifact. Drives a REAL fine-tune run with epoch
/// checkpointing enabled through the worker's own publish path (the same
/// `session_with_training_data`/`tiny_bert_model` fixture every other test
/// in this module uses), then walks the PHYSICAL directory tree on disk
/// under this job's own root (a local file-scheme store, the only scheme
/// these tests run against — [`jammi_test_utils::url_to_path`] is the same
/// helper other integration suites use for exactly this) and asserts every
/// regular file's immediate parent directory is EXACTLY the served model's
/// own artifact or one of the registered epoch-checkpoint rows' artifacts —
/// never a directory nested any deeper.
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
    let served_prefix = crate::common::served_bundle_url(&served).to_string();

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
        let epoch_prefix = crate::common::served_bundle_url(&row).to_string();
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
                "object {path:?} is nested deeper than any known row's artifact \
                 ({allowed_parents:?}); bundle flatness is violated"
            );
        }
    }
    assert!(
        objects_seen > 0,
        "the walk must actually find published objects, or this assertion is vacuous"
    );
}

// ─── The ex-ante kernel-admission profile, end to end ─────────────────────────
//
// Oracle (b): a fine-tune under `JAMMI_KERNELS_DISABLE` naming one op vs a
// run with nothing disabled must produce DIFFERENT `DefinitionHash`es, and
// the recorded profile must name the flip. `jammi_kernels::admission`'s
// `disabled_ops()`/`admission_mode()` are process-wide `OnceLock`s over env
// vars (read once per process, memoized) — exactly the hazard
// `admission_mode_defaults_to_fallback_without_the_env_var`'s own doc names
// in `jammi-kernels` — so the two legs cannot both run as ordinary
// `#[tokio::test]` functions inside this ONE shared `it` binary: whichever
// leg's assertion runs first would freeze the `OnceLock` for every test
// that follows in the same process, including the other leg. Each leg
// therefore runs as its own freshly-spawned CHILD process of the exact
// same compiled `it` binary, the same technique
// `crates/jammi-kernels/src/admission.rs`'s own
// `admission_mode_reads_strict_from_the_real_env_var_in_a_fresh_process`
// uses for the identical hazard.

/// Runs `spec` to completion and reads back the REAL, durably-published
/// `materialization.json` this attempt wrote — never a hand-rolled
/// stand-in: the same `MaterializationManifest` a cold-restarted reader
/// would see.
async fn submit_and_read_manifest(
    session: &Arc<jammi_ai::session::InferenceSession>,
    spec: TrainingSpec,
) -> jammi_db::store::manifest::MaterializationManifest {
    let (model_id, _trained, _cache_outcome) = submit_and_run(session, spec).await;
    let row = session
        .catalog()
        .get_model(&model_id)
        .await
        .unwrap()
        .expect("the served model row must exist");
    let url = crate::common::served_bundle_url(&row);
    session
        .artifact_store()
        .read_model_materialization(&url)
        .await
        .unwrap()
        .expect("a fresh FineTune materialization always writes materialization.json")
}

/// The env var [`spawn_and_capture_profile`] sets to tell the child which
/// `backbone_dtype` to submit its spec at — read back by
/// [`kernel_admission_profile_child_process_body`]. A short tag, not
/// `serde_json`: this crosses a process boundary via `std::env`, not an
/// in-memory value.
const BACKBONE_DTYPE_ENV: &str = "KERNEL_ADMISSION_PROFILE_CHILD_BACKBONE_DTYPE";

fn backbone_dtype_tag(p: jammi_numerics::ComputePrecision) -> &'static str {
    match p {
        jammi_numerics::ComputePrecision::F32 => "f32",
        jammi_numerics::ComputePrecision::F16 => "f16",
        jammi_numerics::ComputePrecision::BF16 => "bf16",
    }
}

fn backbone_dtype_from_tag(tag: &str) -> jammi_numerics::ComputePrecision {
    match tag {
        "f32" => jammi_numerics::ComputePrecision::F32,
        "f16" => jammi_numerics::ComputePrecision::F16,
        "bf16" => jammi_numerics::ComputePrecision::BF16,
        other => panic!("{BACKBONE_DTYPE_ENV}: unrecognised tag {other:?}"),
    }
}

/// This job's backbone dtype as a [`jammi_kernels::admission::DtypeClass`]
/// — the SAME mapping `worker.rs`'s own (private) `dtype_class_of` applies,
/// duplicated here (three match arms, not worth exposing a new `pub` item
/// over).
fn dtype_class_for(p: jammi_numerics::ComputePrecision) -> jammi_kernels::admission::DtypeClass {
    match p {
        jammi_numerics::ComputePrecision::F32 => jammi_kernels::admission::DtypeClass::F32,
        jammi_numerics::ComputePrecision::BF16 => jammi_kernels::admission::DtypeClass::Bf16,
        jammi_numerics::ComputePrecision::F16 => jammi_kernels::admission::DtypeClass::F16,
    }
}

/// Runs [`kernel_admission_profile_child_process_body`] in a fresh process,
/// submitting a job at `backbone_dtype` (the job's own declared dtype, never
/// derived from a loaded model — see [`spec_with_backbone_dtype`]), with
/// `JAMMI_KERNELS_DISABLE` set to `disable`. `disable = None` removes the
/// variable rather than leaving it unset, so a value in this process's own
/// environment cannot leak into the "nothing disabled" leg. Returns the
/// child's printed `(kernel_admission_profile, definition_hash)` pair.
fn spawn_and_capture_profile(
    disable: Option<&str>,
    backbone_dtype: jammi_numerics::ComputePrecision,
) -> (String, String) {
    let mut cmd = jammi_test_resources::child_test(
        "fine_tune_materialization::kernel_admission_profile_child_process_body",
    );
    cmd.env(BACKBONE_DTYPE_ENV, backbone_dtype_tag(backbone_dtype))
        .env_remove("JAMMI_KERNELS_DISABLE");
    if let Some(op) = disable {
        cmd.env("JAMMI_KERNELS_DISABLE", op);
    }
    let stdout = jammi_test_resources::child_test_stdout(&mut cmd);
    // `profile` is itself multi-line (one line per `ProbedOpId` variant) —
    // `stdout.lines()` would otherwise split it apart and this find_map
    // would silently keep only its FIRST line (a real bug this file shipped
    // until an F16 leg finally exercised a
    // non-first row and caught it: every EARLIER assertion here only ever
    // checked `layer_norm`, `ProbedOpId::ALL`'s first variant, so the
    // truncation was invisible). The child encodes embedded newlines as
    // `\x1e` (ASCII record separator) on ONE printed line; decoded back
    // here.
    let profile = stdout
        .lines()
        .find_map(|l| l.strip_prefix("KERNEL_ADMISSION_PROFILE="))
        .unwrap_or_else(|| panic!("child did not print a profile line: {stdout}"))
        .replace('\x1e', "\n");
    let hash = stdout
        .lines()
        .find_map(|l| l.strip_prefix("DEFINITION_HASH="))
        .unwrap_or_else(|| panic!("child did not print a hash line: {stdout}"))
        .to_string();
    (profile, hash)
}

/// The child process [`spawn_and_capture_profile`] runs: it needs a fresh
/// process because the admission mode and the disabled-op set are read once
/// per process.
///
/// `MaterializationManifest` folds the environment (including
/// `kernel_admission_profile`) away into the opaque `definition_hash` and
/// keeps only `descriptor` in the clear (`MaterializationManifest`'s own
/// doc) — it is not retained verbatim on disk for this test to read back.
/// The profile string is instead COMPUTED here, in this same process,
/// through the exact call the worker itself made when it wrote this
/// attempt's manifest (`render_kernel_admission_profile`, `admission_mode()`,
/// `disabled_ops_requested()`, and [`dtype_class_for`] over the SAME
/// `backbone_dtype` this child submitted its own spec at — NEVER derived by
/// loading the model and reading its own
/// `compute_precision()`, which is a different axis and agreed with the
/// bug this fix corrects by construction): all facts are either
/// process-wide (fixed for this process's whole lifetime) or, for dtype, a
/// fact about the SAME `backbone_dtype` this process's own spec declared —
/// so calling them again AFTER training completes reads the identical
/// values the worker's own call site read before training started.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "child process of spawn_and_capture_profile"]
async fn kernel_admission_profile_child_process_body() {
    let backbone_dtype = backbone_dtype_from_tag(
        &std::env::var(BACKBONE_DTYPE_ENV)
            .unwrap_or_else(|_| panic!("{BACKBONE_DTYPE_ENV} must be set by the parent test")),
    );
    let (session, _dir) = session_with_training_data().await;
    let manifest = submit_and_read_manifest(
        &session,
        spec_with_backbone_dtype(CachePolicy::Bypass, backbone_dtype),
    )
    .await;
    let profile = jammi_kernels::admission::render_kernel_admission_profile(
        dtype_class_for(backbone_dtype),
        jammi_kernels::admission::admission_mode(),
        &jammi_kernels::admission::disabled_ops_requested(),
    );
    // Encoded onto ONE line — see `spawn_and_capture_profile`'s own doc for
    // why a bare multi-line `println!` here would silently truncate.
    println!("KERNEL_ADMISSION_PROFILE={}", profile.replace('\n', "\x1e"));
    println!("DEFINITION_HASH={}", manifest.definition_hash);
}

/// The end-to-end oracle: a real, `F32`-backbone fine-tune with
/// `layer_norm_fused` named in `JAMMI_KERNELS_DISABLE` vs the identical
/// spec with nothing disabled produce DIFFERENT `DefinitionHash`es, and the
/// recorded profile names the flip on exactly the disabled row's own line.
#[tokio::test(flavor = "multi_thread")]
async fn kernel_admission_profile_names_a_real_disabled_op_and_moves_the_definition_hash() {
    let f32 = jammi_numerics::ComputePrecision::F32;
    let (enabled_profile, enabled_hash) = spawn_and_capture_profile(None, f32);
    let (disabled_profile, disabled_hash) =
        spawn_and_capture_profile(Some("layer_norm_fused"), f32);

    assert!(
        enabled_profile.contains("layer_norm=enabled"),
        "with nothing disabled, layer_norm's own line must read enabled: {enabled_profile}"
    );
    assert!(
        disabled_profile.contains("layer_norm=disabled"),
        "with layer_norm_fused named in JAMMI_KERNELS_DISABLE, layer_norm's own line must \
         read disabled: {disabled_profile}"
    );
    assert_ne!(
        enabled_profile, disabled_profile,
        "the recorded profile must differ between the two legs"
    );
    assert_ne!(
        enabled_hash, disabled_hash,
        "a real, published DefinitionHash must differ between a run with layer_norm_fused \
         disabled and one without — enabled_profile={enabled_profile:?} \
         disabled_profile={disabled_profile:?}"
    );
}

/// A REAL `F16`-backbone fine-tune
/// (`spec_with_backbone_dtype` — see its own doc for why this stays on the
/// plain arm rather than the encoder-adapters arm `acceleration_report.rs`
/// uses) has its `DefinitionHash` MOVE under
/// `JAMMI_KERNELS_DISABLE=cast_scale_f16_f32` — the key its OWN dtype
/// resolves, per `cast_scale`'s two dtype-branching entries
/// (`Bf16`→`cast_scale_bf16_f32`, `F16`→`cast_scale_f16_f32`). The SAME
/// disabled entry on an `F32` job (which resolves NO key under
/// `cast_scale` at all — `n/a`) must NOT move that job's hash: the
/// over-discrimination half. There is no `Bf16` leg: `validate_backbone_precision`
/// is not reached on the plain arm this suite trains on, so a `Bf16` leg here
/// would prove nothing about that refusal either way.
#[tokio::test(flavor = "multi_thread")]
async fn kernel_admission_profile_f16_backbone_moves_the_hash_under_its_own_cast_key() {
    let f16 = jammi_numerics::ComputePrecision::F16;
    let (f16_baseline_profile, f16_baseline_hash) = spawn_and_capture_profile(None, f16);
    let (f16_disabled_profile, f16_disabled_hash) =
        spawn_and_capture_profile(Some("cast_scale_f16_f32"), f16);
    assert!(
        f16_baseline_profile.contains("cast_scale=enabled"),
        "an f16 job's cast_scale line must read enabled with nothing disabled: \
         {f16_baseline_profile}"
    );
    assert!(
        f16_disabled_profile.contains("cast_scale=disabled"),
        "an f16 job's cast_scale line must read disabled once cast_scale_f16_f32 is named: \
         {f16_disabled_profile}"
    );
    assert_ne!(
        f16_baseline_hash, f16_disabled_hash,
        "an f16-backbone job's DefinitionHash must move when its own cast_scale key \
         (cast_scale_f16_f32) is disabled — f16_baseline_profile={f16_baseline_profile:?} \
         f16_disabled_profile={f16_disabled_profile:?}"
    );

    // Over-discrimination control: the SAME disabled entry on an F32 job
    // (which can never resolve a cast_scale key at all) must be inert.
    let f32 = jammi_numerics::ComputePrecision::F32;
    let (f32_baseline_profile, f32_baseline_hash) = spawn_and_capture_profile(None, f32);
    let (f32_disabled_profile, f32_disabled_hash) =
        spawn_and_capture_profile(Some("cast_scale_f16_f32"), f32);
    assert!(
        f32_baseline_profile.contains("cast_scale=n/a"),
        "{f32_baseline_profile}"
    );
    assert!(
        f32_disabled_profile.contains("cast_scale=n/a"),
        "{f32_disabled_profile}"
    );
    assert_eq!(
        f32_baseline_hash, f32_disabled_hash,
        "an F32 job's DefinitionHash must NOT move when cast_scale_f16_f32 (a key it can \
         never resolve) is named in JAMMI_KERNELS_DISABLE"
    );
}

/// CONTROL: two children with the IDENTICAL environment (nothing disabled,
/// both legs) produce the IDENTICAL profile and the IDENTICAL published
/// `DefinitionHash`. Without this, the previous tests' `assert_ne!` pairs
/// would be equally consistent with "these two child processes just never
/// agree on anything" — this rules that out.
#[tokio::test(flavor = "multi_thread")]
async fn kernel_admission_profile_two_identical_env_children_match() {
    let f32 = jammi_numerics::ComputePrecision::F32;
    let (profile_a, hash_a) = spawn_and_capture_profile(None, f32);
    let (profile_b, hash_b) = spawn_and_capture_profile(None, f32);
    assert_eq!(
        profile_a, profile_b,
        "two children with an identical environment must render an identical profile"
    );
    assert_eq!(
        hash_a, hash_b,
        "two children with an identical environment must publish an identical DefinitionHash"
    );
}

/// An INERT `JAMMI_KERNELS_DISABLE` entry — a key that names no real
/// [`jammi_kernels::admission::ProbedOpId`] row's resolved key at all —
/// must render the SAME profile (and therefore the same `DefinitionHash`)
/// as nothing disabled. This is the over-discrimination control: a
/// disabled-set entry must only
/// ever move the line(s) it actually resolves against, never every line
/// unconditionally.
#[tokio::test(flavor = "multi_thread")]
async fn kernel_admission_profile_an_inert_disabled_key_does_not_move_the_hash() {
    let f32 = jammi_numerics::ComputePrecision::F32;
    let (baseline_profile, baseline_hash) = spawn_and_capture_profile(None, f32);
    let (inert_profile, inert_hash) = spawn_and_capture_profile(Some("nonexistent_key_xyz"), f32);
    assert_eq!(
        baseline_profile, inert_profile,
        "a JAMMI_KERNELS_DISABLE entry naming no real registry key must not change the profile"
    );
    assert_eq!(
        baseline_hash, inert_hash,
        "a JAMMI_KERNELS_DISABLE entry naming no real registry key must not move DefinitionHash"
    );
}
