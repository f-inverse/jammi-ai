//! The resume-parity row: a claimed `fine_tune` job that
//! DIES after epoch 1's durable resume checkpoint is written resumes
//! byte-for-byte the same way whether the host that died was the `Peer`
//! coordinator (a fleet member dialed over the real `GangServiceServer`
//! loopback, the same shape `gang_chaos.rs` runs) or the `Local` in-process
//! claimant (`[worker] local_ranks = 2`) — both routes converge on the SAME
//! shared entry (`worker.rs::run_fine_tune_blocking` → `discover_resume`).
//!
//! Each row's "SIGKILL" is the real thing a crashed process is, not a
//! parked/cancelled attempt (an attempt is never cancelled mid-run): the
//! claimed run is spawned onto the host's OWN dedicated tokio runtime (never
//! the test's). The kill point is the trainer's own discrete, test-observable
//! event — `jammi_ai::fine_tune::worker::loop_test_hooks::Event::
//! ResumeCheckpointWritten`, fired inside `TrainingLoop::save_resume_checkpoint`
//! the instant its `stage_resume_checkpoint` write lands (armed BEFORE the host
//! claims, as `jobs_shutdown.rs` does, so the fire can never
//! race ahead of the arm) — never a wall-clock poll racing the training
//! loop's own write cadence. Once observed, that host's lease keeper thread
//! is killed (`LeaseKeeper::kill_thread_for_test` — `gang_chaos.rs`'s
//! split-brain row's own mechanism, so the row's lease genuinely stops
//! renewing) and its runtime is dropped (`shutdown_background`,
//! `gang_chaos::Member::kill`'s exact shape): the blocking training thread
//! runs on to its next runtime-dependent op — epoch 2's own checkpoint write,
//! or the final publish — and fails there, so nothing terminal ever reaches
//! the row. A fresh host (a second `KillableHost`, simulating the process
//! restarting) then reclaims and completes the job.
//!
//! Two rows, EACH run under both topologies (`Peer` and `Local`), so
//! neither row's proof rests on the other topology's execution:
//!
//! - the parity property itself: attempt 2's published adapter, rank 0, is
//!   byte-identical whether the crash-and-resume ran under `Peer` or under
//!   `Local`;
//! - the mutation the parity property above is blind to: on `Device::Cpu`
//!   the trainer's whole trajectory is a pure function of `(seed, source
//!   rows, config)` (`resume.rs`'s own doc), so a resumed attempt 2 and a
//!   silently-skipped-resume attempt 2 that just retrains both epochs from
//!   scratch reach the IDENTICAL final bytes for this fixture — final-byte
//!   equality alone cannot prove `discover_resume` ran, under either
//!   topology. This row corrupts the ONLY durable evidence `discover_resume`
//!   reads (`optimizer.safetensors` under the shared `_resume/` prefix)
//!   between the kill and the successor's claim: `ArtifactStore::
//!   fetch_resume_checkpoint`'s own contract (`artifact.rs`) makes a
//!   present-but-corrupt bundle a HARD ERROR, never a silent from-scratch
//!   restart — so attempt 2 failing here, naming the digest mismatch, is the
//!   executed proof that SOME rank's body genuinely called back into
//!   `discover_resume` rather than skipping it. Run once under `Local`
//!   (`..._under_local`, `[worker] local_ranks = 2`, no fleet dial) and once
//!   under `Peer` (`..._under_peer`, a real loopback member over the same
//!   `GangServiceServer` `gang_chaos.rs` uses, unmodified, serving both
//!   attempts) — the two rows share one driver
//!   (`run_corrupted_epoch_1_checkpoint`), never duplicated per topology.
//!
//!   The job-terminal `failed`/`sha256` assertion alone cannot attribute
//!   WHICH rank's read produced it: the `_resume/` bundle is job-scoped, so
//!   on `Peer` the in-process rank-0 coordinator and the dialed member's
//!   rank 1 read the identical corrupted bundle and fail identically — a
//!   member that never even attempted resume would leave this assertion
//!   passing vacuously off rank 0's own failure alone. The `Peer` row
//!   closes that gap with a SECOND, rank-attributed assertion: it arms
//!   `loop_test_hooks::Event::ResumeAttempted(1)` (fired unconditionally,
//!   immediately before a rank body's own `discover_resume` call, by rank
//!   number) for attempt 2 and asserts it fired before the row went
//!   terminal — the executed proof that the MEMBER's own body, specifically,
//!   reached the resume seam.

#![cfg(feature = "test-hooks")]

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::worker::{loop_test_hooks, JobWorker};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::gang_rounds::GangDialer;
use tempfile::TempDir;

use crate::gang_chaos::{Fleet, Member};
use crate::gang_coordinator::{published_adapter_bytes, row, two_rank_spec, write_pairs_csv, Row};

/// The killed host's own lease: short, so attempt 2's reclaim is observed in
/// seconds.
const LEASE: Duration = Duration::from_secs(6);
const HEARTBEAT: Duration = Duration::from_secs(1);
const RANK_TIMEOUT_SECS: u64 = 5;
/// The claim loop's reclaim cap (`worker.rs`'s `MAX_ATTEMPTS`); never
/// reached in these rows (attempt 2 always completes or fails outright).
const MAX_ATTEMPTS: u32 = 3;

/// Register the shared CSV source once, before any host claims — the
/// catalog and result root are the fleet's, shared by every host that
/// claims against it. `gang_coordinator.rs`'s own `pairs()`/`gang_config(2)`
/// fixture (`two_rank_spec()`): the property needs one boundary of real,
/// checkpointable training (epoch 1 must complete so a `_resume/` bundle
/// exists) and a second epoch to resume into — nothing about the row count
/// or step count matters beyond that now that the kill point is the
/// trainer's own write event, not a wall-clock race against it.
async fn add_pairs_source(session: &Arc<InferenceSession>, fleet: &Fleet) {
    let url = write_pairs_csv(fleet.dir());
    session
        .add_source(
            "pairs",
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .expect("source");
}

/// A host whose claimed run can be SIGKILLed: the run is spawned onto this
/// host's own dedicated tokio runtime (never the test's own), so killing it
/// severs the run exactly as `gang_chaos::Member::kill` severs a member's
/// process — the blocking training thread runs on to its next
/// runtime-dependent op and fails there.
struct KillableHost {
    session: Arc<InferenceSession>,
    worker_id: String,
    runtime: Option<tokio::runtime::Runtime>,
    _dir: TempDir,
}

impl KillableHost {
    /// `install_dialer`: `true` for a `Peer` coordinator (the one statement
    /// `OssServer::bind` performs beside the gang listener); `false` for a
    /// `Local` in-process claimant, which never dials.
    async fn start(
        fleet: &Fleet,
        install_dialer: bool,
        configure: impl FnOnce(&mut JammiConfig),
    ) -> Self {
        let dir = TempDir::new().expect("host dir");
        let mut cfg = fleet.host_config(dir.path(), LEASE, HEARTBEAT, RANK_TIMEOUT_SECS);
        configure(&mut cfg);
        let session = Arc::new(InferenceSession::new(cfg).await.expect("host session"));
        if install_dialer {
            assert!(
                session
                    .host_admission()
                    .install_member_dialer(Arc::new(GangDialer)),
                "a fresh session has no dialer"
            );
        }
        let worker_id = JobWorker::new(&session)
            .expect("worker")
            .worker_id()
            .to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("host runtime");
        Self {
            session,
            worker_id,
            runtime: Some(runtime),
            _dir: dir,
        }
    }

    /// The claim loop's own two steps (`reclaim_expired_jobs`, then
    /// `claim_next`), polled on a lease-expiry cadence — the SAME shape
    /// `gang_chaos::Coordinator::claim` already uses (reclaim-then-claim,
    /// gated on the row's own lease, never a fixed wall-clock guess at an
    /// unrelated event); reproduced here rather than shared because
    /// `Coordinator` and `KillableHost` close over different lease/attempt
    /// constants.
    async fn claim(&self, within: Duration) -> JobRecord {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            self.session
                .catalog()
                .reclaim_expired_jobs(LEASE, MAX_ATTEMPTS)
                .await
                .expect("reclaim");
            if let Some(record) = self
                .session
                .catalog()
                .claim_next(&self.worker_id, &["fine_tune"], LEASE)
                .await
                .expect("claim")
            {
                return record;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the job was not claimable within {within:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Spawn `run_claimed_job` ON this host's own runtime — killable via
    /// [`Self::kill`]. Fire-and-forget: the caller never awaits this
    /// attempt's outcome (a killed attempt never reaches one).
    fn spawn_claimed_run(&self, record: JobRecord) {
        let session = Arc::clone(&self.session);
        self.runtime
            .as_ref()
            .expect("live host runtime")
            .spawn(async move {
                let worker = JobWorker::new(&session).expect("worker");
                worker.run_claimed_job(&session, record).await;
            });
    }

    /// Run the claimed job to completion directly, awaited on the caller's
    /// own runtime — the successor attempt, never killed.
    async fn run(&self, record: JobRecord) {
        let worker = JobWorker::new(&self.session).expect("worker");
        worker.run_claimed_job(&self.session, record).await;
    }

    /// SIGKILL: the lease keeper's OS thread is killed first (the row's
    /// lease genuinely stops renewing — `gang_chaos.rs`'s split-brain row's
    /// own mechanism), then this host's dedicated runtime is dropped with
    /// everything still running on it (`gang_chaos::Member::kill`'s exact
    /// shape): the blocking training thread runs on to its next
    /// runtime-dependent op — the next epoch's own checkpoint write, or the
    /// final publish — and fails there.
    fn kill(&mut self) {
        self.session.lease_keeper().kill_thread_for_test();
        let runtime = self.runtime.take().expect("a live host runtime");
        runtime.shutdown_background();
    }
}

impl Drop for KillableHost {
    /// A never-killed host's runtime still needs `shutdown_background` at
    /// drop time: tokio's own `Runtime::drop` blocks, which panics when it
    /// runs inside another (the test's own) async context — the exact
    /// panic a bare `self.runtime.take();` at the end of an async fn hits.
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Arm [`loop_test_hooks::Event::ResumeCheckpointWritten`] for `job_id`
/// BEFORE the host claims (as `jobs_shutdown.rs` does: armed
/// before the claim/spawn that starts the run, so the trainer's fire —
/// inside `save_resume_checkpoint`, the instant its `stage_resume_checkpoint`
/// write lands — can never race ahead of the arm), then wait on the REAL
/// event with a generous backstop against a wedged or starved machine —
/// never a wall-clock guess at when one epoch's write might complete.
async fn wait_for_resume_checkpoint_written(observed: loop_test_hooks::Observed) {
    tokio::time::timeout(Duration::from_secs(60), observed.wait_fired())
        .await
        .expect(
            "a generous backstop against a wedged or starved machine: no resume bundle write \
             was ever observed",
        );
}

/// A `Peer` W=2 gang: a coordinator plus one fleet member dialed over the
/// real `GangServiceServer` loopback (`gang_chaos::Member`, unmodified —
/// the SAME member serves both attempts). Attempt 1 is SIGKILLed once
/// epoch 1's checkpoint write is observed; attempt 2, over a fresh
/// coordinator (simulating the process restarting) and the SAME member,
/// resumes and completes. Returns attempt 2's published rank-0 adapter
/// bytes.
async fn run_peer_w2_resumed(fleet: &Fleet) -> Vec<u8> {
    let configure = |cfg: &mut JammiConfig| {
        cfg.server.peer_advertise = Some("127.0.0.1:1".into());
        cfg.distributed.max_world_size = 2;
    };
    let mut coordinator = KillableHost::start(fleet, true, configure).await;
    add_pairs_source(&coordinator.session, fleet).await;
    let member = Member::start(fleet).await;

    let job = coordinator
        .session
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job submits");
    let job_id = job.job_id.clone();

    // ── attempt 1: armed BEFORE the claim, killed once observed ──────────
    let bundle_written =
        loop_test_hooks::arm_observed(&job_id, loop_test_hooks::Event::ResumeCheckpointWritten);
    let record = coordinator.claim(Duration::from_secs(10)).await;
    assert_eq!(record.attempts, 1);
    coordinator.spawn_claimed_run(record);
    wait_for_resume_checkpoint_written(bundle_written).await;
    coordinator.kill();
    member.wait_slot_free(Duration::from_secs(15)).await;

    // ── attempt 2: a fresh coordinator, the same member, resumes ────────
    let coordinator2 = KillableHost::start(fleet, true, configure).await;
    let record2 = coordinator2.claim(LEASE + Duration::from_secs(20)).await;
    assert_eq!(
        record2.attempts, 2,
        "the successor's claim spends the attempt after the killed coordinator's lease expired"
    );
    assert_eq!(
        record2.releases, 0,
        "the coordinator died without releasing: the lease was left to expire"
    );
    coordinator2.run(record2).await;

    let after = row(&coordinator2.session, &job_id).await;
    assert_eq!(after.status, "completed", "Peer attempt 2: {after:?}");
    assert_eq!(after.error, None);
    let bytes = published_adapter_bytes(&coordinator2.session, &job_id).await;
    assert!(!bytes.is_empty());
    bytes
}

/// A `Local` W=2 gang: a `local_ranks = 2` host runs BOTH ranks in-process
/// (`TopologyDecision::Local`), no member. Same shape, no fleet dial.
/// Returns attempt 2's published rank-0 adapter bytes.
async fn run_local_w2_resumed(fleet: &Fleet) -> Vec<u8> {
    let configure = |cfg: &mut JammiConfig| {
        cfg.gpu.device = 0;
        cfg.gpu.devices = Some(vec![0, 1]);
        cfg.worker.local_ranks = 2;
    };
    let mut host = KillableHost::start(fleet, false, configure).await;
    add_pairs_source(&host.session, fleet).await;

    let job = host
        .session
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job submits on a local_ranks = 2 host");
    let job_id = job.job_id.clone();

    // ── attempt 1: armed BEFORE the claim, killed once observed ──────────
    let bundle_written =
        loop_test_hooks::arm_observed(&job_id, loop_test_hooks::Event::ResumeCheckpointWritten);
    let record = host.claim(Duration::from_secs(10)).await;
    assert_eq!(record.attempts, 1);
    host.spawn_claimed_run(record);
    wait_for_resume_checkpoint_written(bundle_written).await;
    host.kill();

    // ── attempt 2: a fresh in-process host resumes ───────────────────────
    let host2 = KillableHost::start(fleet, false, configure).await;
    let record2 = host2.claim(LEASE + Duration::from_secs(20)).await;
    assert_eq!(
        record2.attempts, 2,
        "the successor's claim spends the attempt after the killed host's lease expired"
    );
    assert_eq!(record2.releases, 0);
    host2.run(record2).await;

    let after = row(&host2.session, &job_id).await;
    assert_eq!(after.status, "completed", "Local attempt 2: {after:?}");
    assert_eq!(after.error, None);
    let bytes = published_adapter_bytes(&host2.session, &job_id).await;
    assert!(!bytes.is_empty());
    bytes
}

/// The parity property: attempt 2's published adapter is byte-identical
/// whether the crash-and-resume cycle ran under `Peer` or `Local` topology
/// — the two routes share exactly one entry
/// (`worker.rs::run_fine_tune_blocking` → `discover_resume`).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn peer_and_local_w2_gangs_resume_from_epoch_1s_checkpoint_and_publish_byte_identical_adapters(
) {
    let peer_fleet = Fleet::new();
    let peer_bytes = run_peer_w2_resumed(&peer_fleet).await;

    let local_fleet = Fleet::new();
    let local_bytes = run_local_w2_resumed(&local_fleet).await;

    assert_eq!(
        peer_bytes, local_bytes,
        "attempt 2's published adapter must be identical whether the coordinator that died was \
         a Peer or the whole job ran Local — one shared discover_resume entry"
    );
}

/// The mutation the byte-parity property above is blind to (this module's
/// doc), the shared driver: kill attempt 1 once epoch 1's resume checkpoint
/// is durably written, corrupt the ONE file `discover_resume` reads back
/// (`optimizer.safetensors`), then claim and run attempt 2 to its terminal
/// state. `member`, when `Some`, is the `Peer` topology's own dialed
/// loopback fleet member (unmodified, serving both attempts, exactly
/// `run_peer_w2_resumed`'s own shape) — its slot is awaited free between
/// the kill and the second host's claim, the same wait that row performs.
///
/// `member_rank`, when `Some(rank)`, arms
/// [`loop_test_hooks::Event::ResumeAttempted`] for `rank` immediately BEFORE
/// attempt 2 runs (never before attempt 1 — that attempt legitimately calls
/// `discover_resume` too, and the event is one-shot per (job, rank), so
/// arming any earlier would let attempt 1's own occurrence consume it) and,
/// once attempt 2 is terminal, asserts that rank's body reached the resume
/// seam. This is the ONLY rank-attributed evidence in this driver: the
/// job-terminal `failed`/`sha256` assertion below is job-scoped, not
/// rank-scoped — the `_resume/` bundle is shared and read independently by
/// EVERY rank, so on a `Peer` gang a rank-0 (coordinator) failure and a rank
/// 1 (member) failure produce an IDENTICAL terminal row; only the armed
/// event distinguishes "rank `member_rank`'s own body reached
/// `discover_resume`" from "some other rank's read failed the job first".
/// `None` on `Local` (there is no separate member to attribute to — the one
/// host runs every rank in-process).
///
/// Shared by both topology tests below so the corrupted-bundle assertion is
/// written once, never duplicated per topology.
async fn run_corrupted_epoch_1_checkpoint(
    fleet: &Fleet,
    install_dialer: bool,
    member: Option<&Member>,
    member_rank: Option<u32>,
    configure: impl Fn(&mut JammiConfig) + Copy,
) -> Row {
    let mut host = KillableHost::start(fleet, install_dialer, configure).await;
    add_pairs_source(&host.session, fleet).await;

    let job = host
        .session
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job submits");
    let job_id = job.job_id.clone();

    let bundle_written =
        loop_test_hooks::arm_observed(&job_id, loop_test_hooks::Event::ResumeCheckpointWritten);
    let record = host.claim(Duration::from_secs(10)).await;
    assert_eq!(record.attempts, 1);
    host.spawn_claimed_run(record);
    wait_for_resume_checkpoint_written(bundle_written).await;
    host.kill();
    if let Some(member) = member {
        member.wait_slot_free(Duration::from_secs(15)).await;
    }

    // Corrupt the ONE durable file `discover_resume` reads back: a fresh
    // sha256 mismatch against the untouched manifest, `artifact.rs`'s own
    // hard-error contract. The checkpoint write is already durably
    // complete (the event fired only after `stage_resume_checkpoint`
    // returned `Ok`) and the host is already dead (nothing else can write
    // to it), so this read is not racing anything.
    let checkpoint = host
        .session
        .artifact_store()
        .fetch_resume_checkpoint(host.session.catalog(), &job_id)
        .await
        .expect("resume read")
        .expect("epoch 1's checkpoint exists");
    std::fs::write(
        checkpoint.dir().join("optimizer.safetensors"),
        b"corrupted-by-the-test",
    )
    .expect("corrupt the checkpoint");

    let host2 = KillableHost::start(fleet, install_dialer, configure).await;
    let record2 = host2.claim(LEASE + Duration::from_secs(20)).await;
    assert_eq!(record2.attempts, 2);

    // Armed BEFORE attempt 2 runs (see this fn's own doc on why not earlier).
    let resume_attempted = member_rank.map(|rank| {
        loop_test_hooks::arm_observed(&job_id, loop_test_hooks::Event::ResumeAttempted(rank))
    });

    host2.run(record2).await;

    if let Some(observed) = resume_attempted {
        tokio::time::timeout(Duration::from_secs(10), observed.wait_fired())
            .await
            .expect(
                "rank-attributed proof: the member's own rank body must have reached \
                 discover_resume for this job during attempt 2 -- a job-terminal failure \
                 alone does not attribute which rank's read produced it, since every rank \
                 reads the identical job-scoped resume bundle",
            );
    }

    row(&host2.session, &job_id).await
}

/// The corrupted-bundle assertion both topology rows below share: a hard
/// error, never a silent from-scratch restart, naming the digest mismatch.
fn assert_corrupted_resume_failed_loudly(after: &Row) {
    assert_eq!(
        after.status, "failed",
        "a corrupted resume bundle is a hard error, never a silent from-scratch restart: \
         {after:?}"
    );
    assert!(
        after.error.as_deref().is_some_and(|e| e.contains("sha256")),
        "the digest mismatch names itself in the row's error: {after:?}"
    );
}

/// The `Local` arm: a `local_ranks = 2` host runs both ranks in-process, no
/// fleet dial — `run_local_w2_resumed`'s own topology. No separate member to
/// attribute to (`member_rank: None`): the job-terminal
/// `failed`/`sha256` assertion is the whole proof here, same as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_local()
{
    let fleet = Fleet::new();
    let configure = |cfg: &mut JammiConfig| {
        cfg.gpu.device = 0;
        cfg.gpu.devices = Some(vec![0, 1]);
        cfg.worker.local_ranks = 2;
    };
    let after = run_corrupted_epoch_1_checkpoint(&fleet, false, None, None, configure).await;
    assert_corrupted_resume_failed_loudly(&after);
}

/// The `Peer` arm: a coordinator plus one real fleet member dialed over the
/// loopback `GangServiceServer` (`gang_chaos::Member`, unmodified — the SAME
/// member serves both attempts, `run_peer_w2_resumed`'s own topology).
///
/// The job-terminal `failed`/`sha256` assertion this row shares with `Local`
/// does NOT, by itself, prove the MEMBER's rank body reached
/// `discover_resume`: the `_resume/` bundle is job-scoped, and rank 0 (the
/// in-process coordinator) reads the identical corrupted bundle and would
/// fail the SAME way even if the member's own rank-1 body never ran its
/// resume path at all. `member_rank: Some(1)` (the member always runs rank
/// 1 — `gang_coordinator.rs`'s own doc) closes that gap: it arms
/// [`loop_test_hooks::Event::ResumeAttempted(1)`] for attempt 2 and asserts
/// it fired — the executed, rank-attributed proof that the member's own
/// body called back into `discover_resume`, not merely that the job (via
/// SOME rank) ended `failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_peer() {
    let fleet = Fleet::new();
    let configure = |cfg: &mut JammiConfig| {
        cfg.server.peer_advertise = Some("127.0.0.1:1".into());
        cfg.distributed.max_world_size = 2;
    };
    let member = Member::start(&fleet).await;
    let after =
        run_corrupted_epoch_1_checkpoint(&fleet, true, Some(&member), Some(1), configure).await;
    assert_corrupted_resume_failed_loudly(&after);
}
