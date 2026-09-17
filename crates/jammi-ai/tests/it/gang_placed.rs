//! Plan 67 wave 4, GANG unit — the `Placed` arm of a claim, `run_placed_gang`
//! and the two seams (contract `feat_500-wave4` §2.3/§9), driven through the
//! REAL claim → `run_claimed_job` → `run_claimed_job_under` path over two
//! HERMETIC sessions sharing one catalog (the shape two `jammi-server`
//! replicas take — `gang_chaos.rs`'s own pattern, jammi-ai's own version:
//! the executor's `Peer`-shaped spec is placed with its OWN `[worker]
//! local_ranks = 2`, so it completes as a `Local` gang, entirely in-process
//! — no real gang listener / `MemberDialer` is needed to prove K4, exactly
//! the same substitution `gang_coordinator.rs`'s own "local fan-out" oracle
//! makes for the identical reason).
//!
//! Every stub below installs a real [`PlacedGangSubmitter`] on
//! `HostAdmission` — the production seam — never a test-hooks bypass of
//! `run_spec`/`run_claimed_job_under` themselves.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::DataFusionError;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use jammi_ai::fine_tune::worker::{
    loop_test_hooks, training_test_hooks, JobWorker, PlacedGangSubmitter,
};
use jammi_ai::operator::gang_exec::{GangDescriptor, PlacedOutcome};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::Catalog;
use jammi_db::config::JammiConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::manifest::ComputeDeviceKind;
use tempfile::TempDir;

use crate::common;
use crate::gang_coordinator::{
    fan_out_config, graph_edges, graph_loader, graph_nodes, published_adapter_bytes,
    reference_rank0_adapter_bytes, row, submit_and_claim, two_rank_graph_spec, write_csv,
};

fn dummy_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))]).unwrap()
}

fn ok_stream() -> BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>> {
    Box::pin(stream::once(async { Ok(dummy_batch()) }))
}

fn err_stream(
    e: JammiError,
) -> BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>> {
    Box::pin(stream::once(async move {
        Err(DataFusionError::External(Box::new(e)))
    }))
}

/// A session over a shared `dir` (the multi-process shape: two sessions over
/// ONE catalog file — `<dir>/catalog.db`). Mirrors
/// `gang_coordinator::coordinating_session`'s tuning exactly, minus the
/// fresh-`TempDir` allocation that function makes for a single-host test.
async fn shared_dir_session(
    dir: &std::path::Path,
    tune: impl FnOnce(&mut JammiConfig),
) -> Arc<InferenceSession> {
    let mut config = common::test_config(dir);
    config.distributed.max_world_size = 2;
    config.lease.duration_secs = 3;
    config.lease.heartbeat_secs = 1;
    config.worker.idle_poll_secs = 1;
    config.server.peer_bind = Some("127.0.0.1:0".into());
    config.server.peer_advertise = Some("127.0.0.1:1".into());
    tune(&mut config);
    Arc::new(InferenceSession::new(config).await.unwrap())
}

/// The two-host loopback fleet: a SUBMITTER (`[worker] local_ranks = 1`, so
/// its own topology decision for `world_size = 2` would be `Peer` — the
/// placement trigger) and an EXECUTOR (`local_ranks = 2`, so its OWN
/// `run_claimed_job_under` decides `Local{2}` and completes entirely
/// in-process, needing no real gang listener — pressure-round delta 1:
/// "the executor decides topology from its OWN `[worker] local_ranks`")
/// sharing one catalog and one result root (`dir`).
async fn fleet() -> (Arc<InferenceSession>, Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let submitter = shared_dir_session(dir.path(), |_| {}).await;
    for (name, url) in [
        (
            "nodes",
            write_csv(dir.path(), "nodes.csv", "id,text", &graph_nodes()),
        ),
        (
            "edges",
            write_csv(dir.path(), "edges.csv", "src,dst", &graph_edges()),
        ),
    ] {
        submitter
            .add_source(
                name,
                SourceType::File,
                SourceConnection {
                    url: Some(url),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }
    let executor = shared_dir_session(dir.path(), |c| {
        c.gpu.device = 0;
        c.gpu.devices = Some(vec![0, 1]);
        c.worker.local_ranks = 2;
        c.worker.rank_timeout_secs = 10;
    })
    .await;
    (submitter, executor, dir)
}

/// Drives a REAL [`JobWorker::run_placed_gang`] on the executor session —
/// the production seam, never a bypass.
struct DrivingSubmitter {
    executor: Arc<InferenceSession>,
}

impl PlacedGangSubmitter for DrivingSubmitter {
    fn submit(
        &self,
        descriptor: GangDescriptor,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>>>,
    > {
        let executor = Arc::clone(&self.executor);
        Box::pin(async move {
            let stream = match JobWorker::run_placed_gang(&executor, descriptor).await {
                Ok(_outcome) => ok_stream(),
                Err(e) => err_stream(e),
            };
            Ok(stream)
        })
    }

    fn placement_available(&self) -> bool {
        true
    }
}

/// Never reaches an executor at all: the submission itself faults before any
/// transfer (p1's minimal trigger, p4's "fault BEFORE transfer" shape).
struct RefusingSubmitter;

impl PlacedGangSubmitter for RefusingSubmitter {
    fn submit(
        &self,
        _descriptor: GangDescriptor,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>>>,
    > {
        Box::pin(async move { Ok(err_stream(JammiError::FineTune("stub: no executor".into()))) })
    }

    fn placement_available(&self) -> bool {
        true
    }
}

/// Transfers the claim to `executor_id` FIRST (mimicking the first half of
/// `run_placed_gang`'s own CAS, without running its body) then ends in error
/// with no batch — the "fault AFTER transfer" shape (p5).
struct TransferThenFailSubmitter {
    catalog: Catalog,
    executor_id: String,
    lease: Duration,
}

impl PlacedGangSubmitter for TransferThenFailSubmitter {
    fn submit(
        &self,
        descriptor: GangDescriptor,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>>>,
    > {
        let catalog = self.catalog.pinned_to_tenant(None);
        let executor_id = self.executor_id.clone();
        let lease = self.lease;
        Box::pin(async move {
            let transferred = catalog
                .transfer_claim(
                    &descriptor.job_id,
                    &descriptor.submitter,
                    &executor_id,
                    descriptor.attempt,
                    lease,
                )
                .await
                .unwrap();
            assert!(transferred, "the fixture's own transfer must land");
            Ok(err_stream(JammiError::FineTune(
                "stub: transferred then faulted".into(),
            )))
        })
    }

    fn placement_available(&self) -> bool {
        true
    }
}

/// p1 (RED at base: the `Placed` arm/`note_placed` hook does not exist):
/// with a submitter installed, a `world_size = 2` job takes the `Placed`
/// arm and NEVER reaches [`JobWorker::coordinate`] — `note_placed` fires,
/// `coordinator_ends_for` stays empty.
#[tokio::test(flavor = "multi_thread")]
async fn p1_a_world_size_two_job_takes_the_placed_arm_and_not_coordinate() {
    let (submitter, _executor, _dir) = fleet().await;
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(RefusingSubmitter));
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    worker.run_claimed_job(&submitter, record).await;

    assert_eq!(
        training_test_hooks::placed_attempts_for(&job_id),
        vec![1],
        "the Placed arm fired exactly once"
    );
    assert!(
        training_test_hooks::coordinator_ends_for(&job_id).is_empty(),
        "coordinate must never run once a submitter is installed"
    );
}

/// p2 (RED at base): the stub submitter drives a REAL `run_placed_gang` on a
/// second session; the transfer moves `claimed_by`, `attempts`/`releases`
/// unchanged; the executor session's body runs the coordinator through its
/// OWN `Local{2}` gang; the job completes with the SAME artifact bytes (K4)
/// as the wave-3 `Local` fan-out reference.
#[tokio::test(flavor = "multi_thread")]
async fn p2_the_stub_submitter_drives_a_real_run_placed_gang_to_the_same_bytes() {
    let (submitter, executor, _dir) = fleet().await;
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(DrivingSubmitter {
            executor: Arc::clone(&executor),
        }));
    let worker = JobWorker::new(&submitter).unwrap();
    let before = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = before.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();
    let executor_id = executor.instance_id().to_string();

    worker.run_claimed_job(&submitter, before).await;

    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
    assert_eq!(after.claimed_by.as_deref(), Some(executor_id.as_str()));
    assert_eq!(after.attempts, 1, "the transfer spends no attempt");
    assert_eq!(after.releases, 0);

    let published = published_adapter_bytes(&submitter, &job_id).await;
    let reference =
        reference_rank0_adapter_bytes(&submitter, "placed-k4", graph_loader, fan_out_config())
            .await;
    assert!(!published.is_empty());
    assert_eq!(
        published, reference,
        "a placed gang's published bytes equal the Local reference (K4)"
    );

    // p3: the submitter wrote NOTHING (no terminal write; `HandedOff`); its
    // slot released — a second claim on the submitter proceeds.
    let second = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    assert_ne!(second.job_id, job_id, "the submitter's slot is free again");
    let _ = submitter_id;
}

/// p3, the negative half: `probe_claim` on the submitter never observed a
/// stuck `JobRun`/`Awaiting` after the placed attempt ended — a fresh loop
/// iteration's slot is `Free` immediately (asserted directly, hermetically,
/// without needing the loop's own timing).
#[tokio::test(flavor = "multi_thread")]
async fn p3_the_submitters_slot_is_free_after_a_placed_attempt_ends() {
    let (submitter, executor, _dir) = fleet().await;
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(DrivingSubmitter {
            executor: Arc::clone(&executor),
        }));
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    worker.run_claimed_job(&submitter, record).await;

    let claim = submitter
        .host_admission()
        .probe_claim()
        .expect("the submitter's slot is Free once the placed attempt has ended");
    drop(claim);
}

/// p4: a stream fault BEFORE any transfer (the row still `claimed_by` the
/// submitter) → `Abandoned` — the row is `running`, no terminal write, no
/// error recorded; the next reclaim/claim spends the attempt (wave 3 §8's
/// shape) — asserted here on the row facts a fresh reclaim would act on.
#[tokio::test(flavor = "multi_thread")]
async fn p4_a_stream_fault_before_transfer_leaves_the_row_running_for_the_submitter() {
    let (submitter, _executor, _dir) = fleet().await;
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(RefusingSubmitter));
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();

    worker.run_claimed_job(&submitter, record).await;

    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(after.status, "running", "{after:?}");
    assert_eq!(after.claimed_by.as_deref(), Some(submitter_id.as_str()));
    assert_eq!(after.error, None, "no terminal write on Abandoned");
    assert_eq!(
        after.attempts, 1,
        "the attempt is spent only at reclaim/the successor's claim"
    );
    assert_eq!(
        training_test_hooks::placed_submit_ends_for(&job_id),
        vec![true],
        "the row is still this instance's at the re-read: Abandoned, never HandedOff"
    );
}

/// p5: a stream fault AFTER the transfer landed (the row's `claimed_by` is
/// the executor) → `HandedOff` — the submitter still writes nothing; the
/// row is the executor's now, its own (short) lease is what a reclaim sweep
/// would act on next.
#[tokio::test(flavor = "multi_thread")]
async fn p5_a_stream_fault_after_transfer_hands_off_and_the_submitter_writes_nothing() {
    let (submitter, executor, _dir) = fleet().await;
    let executor_id = executor.instance_id().to_string();
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(TransferThenFailSubmitter {
            catalog: submitter.catalog().pinned_to_tenant(None),
            executor_id: executor_id.clone(),
            lease: Duration::from_millis(200),
        }));
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();

    worker.run_claimed_job(&submitter, record).await;

    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(after.status, "running", "{after:?}");
    assert_eq!(
        after.claimed_by.as_deref(),
        Some(executor_id.as_str()),
        "the row is the executor's now"
    );
    assert_eq!(after.error, None, "no terminal write by the submitter");
    assert_eq!(after.attempts, 1, "the transfer itself spends no attempt");
    assert_eq!(
        training_test_hooks::placed_submit_ends_for(&job_id),
        vec![false],
        "the row had already moved at the re-read: HandedOff, never Abandoned"
    );

    // The submitter's own slot is free (never blocked on the executor's
    // lease) — a second claim proceeds at once.
    let second = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    assert_ne!(second.job_id, job_id);
}

/// p6: `run_placed_gang` refuses typed, before any row write, on a stale
/// `attempt`, and a SECOND launch of an already-transferred descriptor is
/// refused too (the `claimed_by = $from` conjunct no longer matches).
#[tokio::test(flavor = "multi_thread")]
async fn p6_run_placed_gang_refuses_a_stale_attempt_and_a_second_launch() {
    let (submitter, executor, _dir) = fleet().await;
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();

    // A stale `attempt` (the row's real attempts is 1).
    let stale = GangDescriptor {
        job_id: job_id.clone(),
        attempt: 99,
        world: 2,
        submitter: submitter_id.clone(),
        device_kind: ComputeDeviceKind::Cpu,
    };
    let err = JobWorker::run_placed_gang(&executor, stale)
        .await
        .expect_err("a stale attempt must refuse");
    assert!(err.to_string().contains("did not land"), "{err}");
    let unchanged = row(submitter.catalog(), &job_id).await;
    assert_eq!(unchanged.status, "running");
    assert_eq!(unchanged.claimed_by.as_deref(), Some(submitter_id.as_str()));

    // The real descriptor: the first launch succeeds and completes.
    let real = GangDescriptor {
        job_id: job_id.clone(),
        attempt: 1,
        world: 2,
        submitter: submitter_id.clone(),
        device_kind: ComputeDeviceKind::Cpu,
    };
    let outcome = JobWorker::run_placed_gang(&executor, real.clone())
        .await
        .expect("the first launch succeeds");
    assert!(
        matches!(outcome, PlacedOutcome::Trained { .. }),
        "{outcome:?}"
    );
    let completed = row(submitter.catalog(), &job_id).await;
    assert_eq!(completed.status, "completed", "{completed:?}");

    // A second launch of the SAME (now-transferred, now-completed)
    // descriptor is refused: `claimed_by = $from` no longer matches (the
    // row is the executor's), and it is no longer `running` either.
    let err = JobWorker::run_placed_gang(&executor, real)
        .await
        .expect_err("a second launch of a transferred attempt must refuse");
    assert!(err.to_string().contains("did not land"), "{err}");
    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(
        after, completed,
        "no further row write from the refused second launch"
    );
}

/// p7: `run_placed_gang` on a host already holding a rank refuses typed
/// BEFORE `transfer_claim` — the row's `claimed_by` is unchanged (wave 3c
/// §4's fixture: a rank taken directly on the admission cell).
#[tokio::test(flavor = "multi_thread")]
async fn p7_run_placed_gang_refuses_a_host_already_holding_a_rank_before_any_transfer() {
    let (submitter, executor, _dir) = fleet().await;
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();

    let busy_rank = executor
        .host_admission()
        .try_hold_rank("some-other-job", 1)
        .expect("Free admits a rank");

    let descriptor = GangDescriptor {
        job_id: job_id.clone(),
        attempt: 1,
        world: 2,
        submitter: submitter_id.clone(),
        device_kind: ComputeDeviceKind::Cpu,
    };
    let err = JobWorker::run_placed_gang(&executor, descriptor)
        .await
        .expect_err("a host already holding a rank must refuse");
    assert!(err.to_string().contains("busy"), "{err}");

    let unchanged = row(submitter.catalog(), &job_id).await;
    assert_eq!(unchanged.status, "running");
    assert_eq!(
        unchanged.claimed_by.as_deref(),
        Some(submitter_id.as_str()),
        "no transfer happened before the slot refusal"
    );
    drop(busy_rank);
}

/// p9 (contract §9 B6, "refuse what's new" for EVERY entry): a host that
/// has begun a DRAIN refuses a placed gang BEFORE any transfer — the row
/// stays the submitter's, so a successor (never this terminating process)
/// runs it. Mutation: drop `probe_claim`'s phase check and this reds (the
/// holder is `Free`, so the CAS would admit and `transfer_claim` would move
/// the row onto a process inside its termination grace).
#[tokio::test(flavor = "multi_thread")]
async fn p9_run_placed_gang_refuses_a_draining_host_before_any_transfer() {
    let (submitter, executor, _dir) = fleet().await;
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();

    assert!(
        executor.host_admission().begin_drain(),
        "Running -> Draining"
    );

    let descriptor = GangDescriptor {
        job_id: job_id.clone(),
        attempt: 1,
        world: 2,
        submitter: submitter_id.clone(),
        device_kind: ComputeDeviceKind::Cpu,
    };
    let err = JobWorker::run_placed_gang(&executor, descriptor)
        .await
        .expect_err("a draining host must refuse a new gang");
    assert!(
        err.to_string().contains("has begun a Draining"),
        "the refusal names the phase, not a busy slot: {err}"
    );

    let unchanged = row(submitter.catalog(), &job_id).await;
    assert_eq!(unchanged.status, "running");
    assert_eq!(
        unchanged.claimed_by.as_deref(),
        Some(submitter_id.as_str()),
        "no transfer happened before the drain refusal"
    );
}

/// p8: a placed run whose OWN process also has a submitter installed does
/// NOT re-submit — `note_placed` fires exactly once, on the original
/// submitter's own claim, never from inside the executor's
/// `run_claimed_job_under(.., placed = true)` call.
#[tokio::test(flavor = "multi_thread")]
async fn p8_a_placed_run_never_re_submits_even_with_a_submitter_installed_on_its_own_process() {
    let (submitter, executor, _dir) = fleet().await;
    // The executor ALSO hosts a scheduler (a single-node cluster's shape):
    // installed, but must never be consulted from inside a placed run.
    executor
        .host_admission()
        .install_placed_gang_submitter(Arc::new(RefusingSubmitter));
    submitter
        .host_admission()
        .install_placed_gang_submitter(Arc::new(DrivingSubmitter {
            executor: Arc::clone(&executor),
        }));
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();

    worker.run_claimed_job(&submitter, record).await;

    assert_eq!(
        training_test_hooks::placed_attempts_for(&job_id),
        vec![1],
        "note_placed fires exactly once, on the original submitter only"
    );
    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
}

/// The heartbeat-keying claim `WorkerJobError::HandedOff`'s own doc names
/// (item 4): `Catalog::heartbeat_job` is keyed on `claimed_by`, so a stale
/// holder's heartbeat after a hand-off can never resurrect the new holder's
/// lease.
#[tokio::test(flavor = "multi_thread")]
async fn the_submitters_heartbeat_after_hand_off_never_resurrects_the_executors_lease() {
    let (submitter, executor, _dir) = fleet().await;
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();
    let executor_id = executor.instance_id().to_string();

    let transferred = submitter
        .catalog()
        .transfer_claim(
            &job_id,
            &submitter_id,
            &executor_id,
            1,
            Duration::from_secs(3),
        )
        .await
        .unwrap();
    assert!(transferred);

    let before = row(submitter.catalog(), &job_id).await;
    let resurrected = submitter
        .catalog()
        .heartbeat_job(&job_id, &submitter_id, 1, Duration::from_secs(30))
        .await
        .unwrap();
    assert!(
        !resurrected,
        "a stale holder's heartbeat must not renew the new holder's lease"
    );
    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(
        before.lease_expires_at, after.lease_expires_at,
        "the lease is untouched by the stale holder's heartbeat"
    );
}

/// #500 wave 5 group E1, P7 pressure-round fix (adversarial-audit finding on
/// `register_job_hold_or_release`): `run_placed_gang` takes its claim
/// (`probe_claim`), snapshots its `WorkerShared` birth release-epoch in the
/// SAME synchronous step, then makes two catalog round trips
/// (`Catalog::transfer_claim`, `Catalog::get_job`) before the hold is
/// registered. A RELEASE landing anywhere in that window — parked here
/// immediately after the birth snapshot, before `transfer_claim` even runs —
/// must still be caught by `register_job_hold_or_release`'s
/// `released_since_birth` check and self-release the claim: never dispatch,
/// never register a lease hold, `coordinate` never reached. The sweeps run
/// before the transfer lands (the row is still the SUBMITTER's at that
/// instant) and find nothing of this job's; the epoch comparison alone is
/// what catches it.
///
/// Mutation (executed): in `JobWorker::run_placed_gang`, changing the final
/// `WorkerShared::for_single_run(admission, worker.worker_id.clone(),
/// claim_epoch)` call to ignore `claim_epoch` and read
/// `admission.release_epoch()` live at that call site instead (the pre-fix
/// shape, which reads the epoch only after both catalog round trips) reds
/// this test: `run_placed_gang` returns `Ok(PlacedOutcome::Trained { .. })`
/// instead of the expected `Err`, because by the time of that live read the
/// RELEASE has already landed and is folded into the very value compared
/// against itself, so `released_since_birth` reads `false` and the claim
/// dispatches straight through the RELEASE. First line of the red output:
/// `a claim that raced RELEASE must self-release, never dispatch: called
/// \`Result::expect_err\` on an \`Ok\` value: Trained { artifact_digest: ...
/// }`.
#[tokio::test(flavor = "multi_thread")]
async fn release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang() {
    let (submitter, executor, _dir) = fleet().await;
    let worker = JobWorker::new(&submitter).unwrap();
    let record = submit_and_claim(&submitter, &worker, two_rank_graph_spec()).await;
    let job_id = record.job_id.clone();
    let submitter_id = submitter.instance_id().to_string();
    let executor_id = executor.instance_id().to_string();

    let descriptor = GangDescriptor {
        job_id: job_id.clone(),
        attempt: 1,
        world: 2,
        submitter: submitter_id.clone(),
        device_kind: ComputeDeviceKind::Cpu,
    };

    let park = loop_test_hooks::arm(
        &job_id,
        loop_test_hooks::ParkPoint::PlacedGangBeforeTransfer,
    );
    let running = {
        let executor = Arc::clone(&executor);
        tokio::spawn(async move { JobWorker::run_placed_gang(&executor, descriptor).await })
    };
    park.wait_parked().await;

    // The claim's own `WorkerShared` birth epoch is already snapshotted;
    // land a RELEASE on the SAME session's admission now, before
    // `transfer_claim` has even run — exactly the window a live
    // `EmbeddedWorker` sharing this process's `HostAdmission` could deliver
    // one in.
    let (_holds, sweep) = executor.release_job_leases().await.unwrap();
    assert_eq!(
        sweep.jobs,
        Some(0),
        "the row is still the submitter's at this instant (no transfer yet); \
         the sweep must not be what releases it here"
    );
    park.release();

    let err = tokio::time::timeout(Duration::from_secs(30), running)
        .await
        .expect("run_placed_gang must return once the prologue self-releases")
        .unwrap()
        .expect_err("a claim that raced RELEASE must self-release, never dispatch");
    assert!(
        err.to_string().contains("left running for reclaim"),
        "{err}"
    );
    assert!(
        training_test_hooks::coordinator_ends_for(&job_id).is_empty(),
        "coordinate must never run: the claim self-released before dispatch"
    );

    let after = row(submitter.catalog(), &job_id).await;
    assert_eq!(after.status, "running", "{after:?}");
    assert_eq!(
        after.claimed_by.as_deref(),
        Some(executor_id.as_str()),
        "transfer_claim still lands (it runs after the park release); the \
         self-release hands the lease back under the EXECUTOR's own \
         claimed_by, never the submitter's"
    );
    assert!(after.lease_expires_at.is_none(), "{after:?}");
    assert_eq!(after.releases, 1, "{after:?}");
}
