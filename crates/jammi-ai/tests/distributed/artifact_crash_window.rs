//! Property 3 — the artifact crash-window on REAL MinIO (#22 cross-host
//! validation).
//!
//! Two facets, both over a real S3-compatible object store (MinIO), not a local
//! `file://` root:
//!
//! (a) **write-on-A / read-on-B**: an artifact a worker process publishes to the
//!     shared bucket is loadable by a *different* client (the harness session,
//!     itself a separate process+driver) — a genuine cross-host round-trip.
//!
//! (b) **crash between publish and finalize**: a worker is SIGKILL'd while it
//!     holds a claim; whatever it may have written to its per-attempt prefix is
//!     orphaned because its finalize CAS never ran. A surviving worker reclaims,
//!     writes its OWN unique per-attempt prefix, and its CAS commits *that*
//!     prefix as the served `artifact_path`. The committed pointer therefore
//!     roots under the WINNER's prefix (`…/models/{job}/{winner}/{attempt}`),
//!     never the loser's, and a reload of the completed model returns the
//!     winner's bytes — no cross-worker clobber on the shared bucket.
//!
//! Because every attempt writes a unique `{job}/{worker}/{attempt}` prefix and
//! the served pointer is written solely by the finalize CAS, the loser's prefix
//! is never pointed-to; the only key the committed model resolves is the
//! winner's. This is exactly the #22 content-addressed, commit-by-pointer
//! contract, validated across process + host boundaries.

use jammi_db::storage::StorageUrl;

use crate::harness::{self, Backends, Fleet, JobSize};

const TEST: &str = "artifact_crash_window";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crash_between_publish_and_finalize_commits_only_the_winner() {
    let Some(backends) = Backends::from_env_or_skip(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    harness::register_training_source(&session, "training").await;
    // Crashable: the claimer must still be running when we crash it mid-publish.
    let (job_id, expected_model) =
        harness::submit_fine_tune(&session, "training", JobSize::Crashable).await;

    let mut fleet = Fleet::spawn(&backends, &result_root, 2);

    // Detect and crash the first claimer mid-run — the publish/finalize window.
    let first_claimer =
        harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
            let session = &session;
            let job_id = &job_id;
            async move {
                let r = session.catalog().get_training_job(job_id).await.ok()?;
                (r.status == "running")
                    .then(|| r.claimed_by.clone())
                    .flatten()
            }
        })
        .await
        .expect("a worker claims and starts running before its lease window closes");
    assert!(fleet.kill9(&first_claimer), "claimer is a spawned worker");

    // The survivor reclaims and completes; the committed pointer is the winner's.
    let record = harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
        let session = &session;
        let job_id = &job_id;
        async move {
            let r = session.catalog().get_training_job(job_id).await.ok()?;
            (r.status == "completed").then_some(r)
        }
    })
    .await
    .expect("the survivor reclaims the crashed job and completes it");

    let winner = record
        .claimed_by
        .as_deref()
        .expect("completed job records its (reclaiming) claimer");
    assert_ne!(
        winner, first_claimer,
        "the winner is not the crashed worker"
    );

    // (b) The committed served pointer roots under the WINNER's per-attempt
    // prefix on the shared bucket — never the crashed loser's. The prefix layout
    // is `{result_root}/models/{job_id}/{worker_id}/{attempt}`.
    let model = session
        .catalog()
        .get_model(&expected_model)
        .await
        .unwrap()
        .expect("the completed job registered its output model");
    let artifact_path = model
        .artifact_path
        .as_deref()
        .expect("the finalize CAS commits the served artifact_path");
    let winner_prefix = format!("{result_root}/models/{job_id}/{winner}/");
    assert!(
        artifact_path.starts_with(&winner_prefix),
        "committed artifact_path {artifact_path:?} must root under the WINNER's prefix \
         {winner_prefix:?}, never the crashed loser {first_claimer:?}"
    );
    assert!(
        !artifact_path.contains(&format!("/{first_claimer}/")),
        "the crashed loser's prefix is never the committed pointer (no clobber): {artifact_path:?}"
    );

    // (a) write-on-worker / read-on-harness: the harness — a separate process
    // and S3 driver — fetches the committed artifact from MinIO, verifies its
    // manifest (sha256), and finds the non-empty LoRA adapter. This is the real
    // cross-host reload the local-FS `it` tests cannot exercise.
    let prefix =
        StorageUrl::parse(artifact_path).expect("committed artifact_path is a storage URL");
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("the winner's artifact fetches from MinIO and verifies its manifest");
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.is_file() && std::fs::metadata(&adapter).unwrap().len() > 0,
        "the reloaded artifact carries a non-empty adapter ({adapter:?}) — the winner's bytes"
    );

    drop(fleet);
}

/// The simpler facet of (a) on its own, with no crash: a single worker completes
/// a job, publishes to MinIO, and the harness (a different process) reloads the
/// bytes. Isolating this from the crash path proves the cross-host round-trip
/// independently of reclaim, so a failure here points squarely at the
/// object-store path rather than the lease machinery.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn artifact_written_on_worker_is_readable_by_a_different_client() {
    let Some(backends) = Backends::from_env_or_skip(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(&format!("{TEST}-roundtrip"));
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    harness::register_training_source(&session, "training").await;
    // Quick: no crash here — this facet only needs a clean completion to reload.
    let (job_id, expected_model) =
        harness::submit_fine_tune(&session, "training", JobSize::Quick).await;

    let fleet = Fleet::spawn(&backends, &result_root, 1);

    let record = harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
        let session = &session;
        let job_id = &job_id;
        async move {
            let r = session.catalog().get_training_job(job_id).await.ok()?;
            (r.status == "completed").then_some(r)
        }
    })
    .await
    .expect("the single worker completes the job");
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(expected_model.as_str())
    );

    let model = session
        .catalog()
        .get_model(&expected_model)
        .await
        .unwrap()
        .expect("output model registered");
    let prefix = StorageUrl::parse(model.artifact_path.as_deref().unwrap()).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("worker-written artifact is readable by the harness over MinIO");
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.is_file() && std::fs::metadata(&adapter).unwrap().len() > 0,
        "the cross-host reload finds the non-empty adapter at {adapter:?}"
    );

    drop(fleet);
}
