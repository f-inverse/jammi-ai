//! Plan 67 U5b-2 — the coordinator's failure path, hermetically, over a
//! two-HOST loopback fleet: a coordinator engine (its own `InferenceSession`
//! with the production `GangDialer` installed) and a member engine (its own
//! `InferenceSession`, mounting the REAL `GangServer::run_rank` on its own
//! tokio runtime — so "the member process died" is one runtime dropped),
//! both over ONE shared catalog and ONE shared result root, exactly the
//! shape two `jammi-server` replicas take in a deployment. The member's rank
//! body is the REAL one (`run_member_rank`, spawned by the member's handler
//! at admission), its `Peer` wrapped — through the `test-hooks` seam
//! `training_test_hooks::wrap_member_collective`, armed per job before the
//! coordinator dials — in a chaos collective that injects the failure at a
//! chosen step of the real training loop — the first optimizer step of the
//! SECOND epoch (the fifth `all_reduce_sum`), so epoch 1's resume checkpoint
//! exists when the attempt is retired and the successor attempt's bodies
//! (rank 0 and the member's) resume from it out of the shared store. Each
//! body's end is read back through `training_test_hooks::rank_outcomes_for`
//! (recorded whether or not its session lived to emit it).
//!
//! Every row pins DESIGN.md §4 "Failure and release" over the real
//! machinery: the coordinator's round deadline and its links' typed record
//! are the watchdog; a retired attempt writes NOTHING terminal (the row
//! stays `running`); the released-vs-failed split settles the lease —
//! `Aborted{Drain}` releases (`releases + 1`, OPS D10), every other gang
//! fault leaves it to expire so the successor's claim spends the attempt
//! (`attempts + 1`, `releases` unchanged); the successor attempt over a
//! fresh gang completes, publishing bytes EQUAL to an uninterrupted
//! `LocalGang` run of the same fixture — one model, from the checkpoint.
//!
//! The rows, each with its executed mutation in the unit's contract:
//! - the member's stream drops mid-round (its gang runtime is dropped) →
//!   `LinkFault`, attempt spent, reclaimed by a new gang after the lease;
//! - the member goes silent past `[worker] rank_timeout_secs` (its rank
//!   thread stalls inside the round) → `LinkFault` naming the deadline,
//!   attempt spent;
//! - the member's host DRAINs mid-round → `Aborted{Drain}` → `Released`:
//!   `releases + 1`, the row claimable at once (never a lease window), the
//!   next attempt over a fresh member completes — zero net attempts;
//! - split brain: attempt 1's coordinator loses its lease (its keeper dies)
//!   while its rank 1 is held; the successor's `RunRank` at attempt 2
//!   supersedes the elder hold (U5a-2's fence), the elder attempt is
//!   refuted and writes nothing, attempt 2 completes with `attempts == 2`.

#![cfg(feature = "test-hooks")]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::Tensor;
use jammi_ai::fine_tune::collective::{BlockingCall, Collective};
use jammi_ai::fine_tune::worker::{training_test_hooks, Holder, JobWorker};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::instance::{InstanceRegistration, MemberRoot, PeerAddr};
use jammi_db::catalog::jobs_repo::{JobRecord, WorkerState};
use jammi_db::config::{CatalogConfig, JammiConfig};
use jammi_db::error::{JammiError, Result};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::gang::GangServer;
use jammi_server::grpc::gang_rounds::GangDialer;
use jammi_wire::proto::gang::gang_service_server::GangServiceServer;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tonic::transport::server::TcpIncoming;

use crate::gang_coordinator::{
    pairs_loader, published_adapter_bytes, reference_rank0_adapter_bytes, row, two_rank_spec,
    write_pairs_csv, Row,
};

/// The coordinator hosts' lease: a retired attempt whose lease is left to
/// expire is reclaimable this long after the coordinator's hold dropped.
/// Short, so an attempt-spent row is observed in seconds.
const COORDINATOR_LEASE: Duration = Duration::from_secs(6);
/// The member host's lease — its held session's PARK bound and its
/// freshness margin for the coordinator's row. Longer than the
/// coordinator's, so an elder session outlives the coordinator's lease
/// expiry and is ended by the successor's fence (split brain), never by its
/// own park.
const MEMBER_LEASE: Duration = Duration::from_secs(20);
/// The coordinator hosts' heartbeat: the lease renewal and the instance
/// touch.
const HEARTBEAT: Duration = Duration::from_secs(1);
/// The member host's heartbeat — its lease renewal, its instance touch and
/// its held sessions' re-verification tick. Slower than a reclaim-and-claim
/// pair, so an elder session is ended by the successor's fence at its dial
/// (split brain), not refuted first by a tick that happened to land in the
/// requeue window.
const MEMBER_HEARTBEAT: Duration = Duration::from_secs(3);
/// The round deadline on a coordinator that must give a silent member up
/// within seconds.
const RANK_TIMEOUT_SECS: u64 = 3;
/// The claim loop's reclaim cap (`worker.rs`'s `MAX_ATTEMPTS`).
const MAX_ATTEMPTS: u32 = 3;
/// The `all_reduce_sum` call the chaos lands on. Every optimizer step is
/// TWO reduces (`optimizer::canonical_reduce`: the gradients, then the
/// presence vector) and the fixture has two optimizer steps per epoch
/// (eight rows, a global batch of four), so the fifth reduce is epoch 2's
/// first step — past epoch 1's resume checkpoint.
const CHAOS_AT_REDUCE: usize = 5;

/// The deployment: one catalog, one result root, the training source.
pub(crate) struct Fleet {
    dir: TempDir,
}

impl Fleet {
    pub(crate) fn new() -> Self {
        let dir = TempDir::new().expect("fleet dir");
        std::fs::create_dir_all(dir.path().join("jammi_db")).expect("result root");
        Self { dir }
    }

    fn catalog_path(&self) -> PathBuf {
        self.dir.path().join("catalog.db")
    }

    fn result_root(&self) -> String {
        self.dir.path().join("jammi_db").display().to_string()
    }

    /// The fleet's own temp dir — a host's private subdirectory (its
    /// `artifact_dir`) is created under this, and the shared training
    /// source's CSV file lives here too (`gang_resume_parity.rs` reuses
    /// this to register the source once, before any host claims).
    pub(crate) fn dir(&self) -> &Path {
        self.dir.path()
    }

    /// A host's config over the shared catalog and result root: its own
    /// `artifact_dir`, `[worker] enabled = false` (every claim below is the
    /// test's own), a serveable world of two, the fleet timing.
    pub(crate) fn host_config(
        &self,
        own: &Path,
        lease: Duration,
        heartbeat: Duration,
        rank_timeout_secs: u64,
    ) -> JammiConfig {
        let mut cfg = jammi_test_utils::test_config(own);
        cfg.catalog = CatalogConfig::Sqlite {
            path: Some(self.catalog_path()),
        };
        cfg.storage.result_root = Some(self.result_root());
        cfg.worker.enabled = false;
        cfg.worker.rank_timeout_secs = rank_timeout_secs;
        cfg.lease.duration_secs = lease.as_secs();
        cfg.lease.heartbeat_secs = heartbeat.as_secs();
        cfg.distributed.max_world_size = 2;
        cfg.server.peer_bind = Some("127.0.0.1:0".into());
        cfg
    }
}

/// A coordinator host: an engine with the production member dialer
/// installed (the one statement `OssServer::bind` performs beside the gang
/// listener) and a `MemberRoot` (`peer_advertise` set; the address is never
/// dialed — a coordinator lists no `workers` row, so no other coordinator
/// ever assigns it a rank).
struct Coordinator {
    session: Arc<InferenceSession>,
    worker: JobWorker,
    _dir: TempDir,
}

impl Coordinator {
    async fn start(fleet: &Fleet, rank_timeout_secs: u64) -> Self {
        let dir = TempDir::new().expect("coordinator dir");
        let mut cfg =
            fleet.host_config(dir.path(), COORDINATOR_LEASE, HEARTBEAT, rank_timeout_secs);
        cfg.server.peer_advertise = Some("127.0.0.1:1".into());
        let session = Arc::new(InferenceSession::new(cfg).await.expect("coordinator"));
        assert!(
            session
                .host_admission()
                .install_member_dialer(Arc::new(GangDialer)),
            "a fresh session has no dialer"
        );
        let worker = JobWorker::new(&session).expect("worker");
        Self {
            session,
            worker,
            _dir: dir,
        }
    }

    /// Register the CSV source on the fleet's catalog (a later host over
    /// the same catalog restores it at start).
    async fn add_pairs_source(&self, fleet: &Fleet) {
        let url = write_pairs_csv(fleet.dir.path());
        self.session
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

    /// The claim loop's own two steps (`reclaim_expired_jobs`, then
    /// `claim_next`), polled: a released row is claimable at once, an
    /// expired one after its lease window, and a cooling row after its
    /// `next_assembly_after`.
    async fn claim(&self, within: Duration) -> JobRecord {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            self.session
                .catalog()
                .reclaim_expired_jobs(COORDINATOR_LEASE, MAX_ATTEMPTS)
                .await
                .expect("reclaim");
            if let Some(record) = self
                .session
                .catalog()
                .claim_next(self.worker.worker_id(), &["fine_tune"], COORDINATOR_LEASE)
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

    async fn run(&self, record: JobRecord) {
        self.worker.run_claimed_job(&self.session, record).await;
    }
}

/// A member host: an engine whose `instances` row advertises its gang
/// listener and whose `workers` row is `claiming` over `fine_tune` (what
/// the coordinator's listing selects), and the REAL `GangServer::run_rank`
/// mounted on a tokio runtime of its own — the member "process" whose death
/// is that runtime dropped (its listener, every accepted connection, every
/// spawned hold loop and every rank body's task go with it; a body's
/// blocking thread runs on to its next collective, which fails).
pub(crate) struct Member {
    pub(crate) session: Arc<InferenceSession>,
    addr: SocketAddr,
    runtime: Option<tokio::runtime::Runtime>,
    _dir: TempDir,
}

impl Member {
    pub(crate) async fn start(fleet: &Fleet) -> Self {
        // The port is reserved BEFORE the session exists so the
        // registration can advertise it; the listener moves onto the gang
        // runtime below.
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = std_listener.local_addr().expect("addr");
        let dir = TempDir::new().expect("member dir");
        let mut cfg = fleet.host_config(
            dir.path(),
            MEMBER_LEASE,
            MEMBER_HEARTBEAT,
            RANK_TIMEOUT_SECS,
        );
        cfg.server.peer_advertise = Some(addr.to_string());
        let session = Arc::new(InferenceSession::new(cfg).await.expect("member"));
        session
            .catalog()
            .upsert_worker(
                session.instance_id(),
                "fine_tune",
                WorkerState::Claiming,
                &[],
            )
            .await
            .expect("workers row");
        let runtime = Self::serve(&session, std_listener);
        Self {
            session,
            addr,
            runtime: Some(runtime),
            _dir: dir,
        }
    }

    fn serve(
        session: &Arc<InferenceSession>,
        std_listener: std::net::TcpListener,
    ) -> tokio::runtime::Runtime {
        std_listener.set_nonblocking(true).expect("nonblocking");
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("gang runtime");
        let gang = GangServer::new(Arc::clone(session), MEMBER_LEASE, MEMBER_HEARTBEAT);
        let cap = usize::try_from(session.inner_config().server.limits.max_message_bytes).unwrap();
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(std_listener).expect("listener");
            tonic::transport::Server::builder()
                .add_service(GangServiceServer::new(gang).max_decoding_message_size(cap))
                .serve_with_incoming(TcpIncoming::from(listener))
                .await
                .expect("serve");
        });
        runtime
    }

    /// The member process dies mid-run: the gang runtime is dropped with
    /// everything on it. The engine (its keeper, its `instances` row) is
    /// left — the row stays fresh, as a process restarted at the same
    /// address within the liveness margin would be.
    fn kill(&mut self) {
        let runtime = self.runtime.take().expect("a live gang runtime");
        runtime.shutdown_background();
    }

    /// The member process is back: a fresh gang runtime and handler, on the
    /// same advertised address when the OS hands it back (the row then
    /// needs no change), otherwise on a fresh port re-registered on the
    /// row exactly as a restarted process re-registers itself.
    async fn restart(&mut self) {
        assert!(self.runtime.is_none(), "kill first");
        let mut rebound = None;
        for _ in 0..40 {
            match std::net::TcpListener::bind(self.addr) {
                Ok(listener) => {
                    rebound = Some(listener);
                    break;
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
            }
        }
        let std_listener = match rebound {
            Some(listener) => listener,
            None => {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
                self.addr = listener.local_addr().expect("addr");
                let root = MemberRoot::resolved(self.session.inner_config()).expect("root");
                self.session
                    .catalog()
                    .upsert_instance(&InstanceRegistration::new(
                        self.session.instance_id(),
                        None,
                        None,
                        Some(PeerAddr::parse(&self.addr.to_string()).unwrap()),
                        Some(root),
                    ))
                    .await
                    .expect("re-register");
                listener
            }
        };
        self.runtime = Some(Self::serve(&self.session, std_listener));
    }

    /// The member's host DRAINs (68 OPS): the phase leaves `Running`, so
    /// every held rank ends `Aborted{Drain}` and the claim loop's `workers`
    /// row leaves `claiming` (the loop would delete it once it exited; no
    /// coordinator lists a draining host).
    async fn drain(&self) {
        self.session
            .catalog()
            .set_worker_state(self.session.instance_id(), WorkerState::Draining)
            .await
            .expect("workers row");
        assert!(self.session.host_admission().begin_drain());
    }

    pub(crate) fn holder(&self) -> Holder {
        self.session.host_admission().holder()
    }

    pub(crate) async fn wait_slot_free(&self, within: Duration) {
        let deadline = tokio::time::Instant::now() + within;
        while self.holder() != Holder::Free {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the member's slot must free once the session ended, holder: {:?}",
                self.holder()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

impl Drop for Member {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// What the stalled rank thread does once the test resumes it.
#[derive(Debug, Clone, Copy)]
enum Resume {
    /// Enter the real round (the session has been ended under it: the
    /// round fails on the closed link).
    Proceed,
    /// Return an error without touching the link again (the link's runtime
    /// is gone, or nothing is left to rendezvous with).
    Fail,
}

/// The chaos seam: at the [`CHAOS_AT_REDUCE`]-th `all_reduce_sum` the rank
/// body tells the test (`tell`), blocks until the test resumes it, and
/// then proceeds or fails. Every other verb is the real `Peer`'s — the
/// wrapper the real body applies through
/// `training_test_hooks::wrap_member_collective`.
struct ChaosRank {
    inner: Arc<dyn Collective>,
    reduces: AtomicUsize,
    tell: Mutex<Option<oneshot::Sender<()>>>,
    resume: Mutex<Option<std::sync::mpsc::Receiver<Resume>>>,
}

/// The rank body's ends of the chaos channels — built with the
/// [`ChaosControl`] before the body exists, wrapped around its `Peer`
/// once the session is admitted and the body reaches the seam.
struct ChaosSeam {
    tell: oneshot::Sender<()>,
    resume: std::sync::mpsc::Receiver<Resume>,
}

/// The test's handles onto one [`ChaosRank`].
struct ChaosControl {
    told: oneshot::Receiver<()>,
    resume: std::sync::mpsc::Sender<Resume>,
}

fn chaos_channels() -> (ChaosSeam, ChaosControl) {
    let (tell, told) = oneshot::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel();
    (
        ChaosSeam {
            tell,
            resume: resume_rx,
        },
        ChaosControl {
            told,
            resume: resume_tx,
        },
    )
}

impl ChaosRank {
    fn new(inner: Arc<dyn Collective>, seam: ChaosSeam) -> Self {
        Self {
            inner,
            reduces: AtomicUsize::new(0),
            tell: Mutex::new(Some(seam.tell)),
            resume: Mutex::new(Some(seam.resume)),
        }
    }

    fn chaos_point(&self) -> Result<()> {
        let n = self.reduces.fetch_add(1, Ordering::SeqCst) + 1;
        if n != CHAOS_AT_REDUCE {
            return Ok(());
        }
        if let Some(tell) = self.tell.lock().unwrap().take() {
            let _ = tell.send(());
        }
        let resume = self
            .resume
            .lock()
            .unwrap()
            .take()
            .expect("the chaos point is reached once");
        match resume.recv().unwrap_or(Resume::Fail) {
            Resume::Proceed => Ok(()),
            Resume::Fail => Err(JammiError::FineTune(
                "chaos: rank 1 retired by the test".into(),
            )),
        }
    }
}

impl Collective for ChaosRank {
    fn all_gather(&self, call: &BlockingCall, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        self.inner.all_gather(call, local, counts)
    }

    fn all_reduce_sum(&self, call: &BlockingCall, tensors: &mut [Tensor]) -> Result<()> {
        self.chaos_point()?;
        self.inner.all_reduce_sum(call, tensors)
    }

    fn all_reduce_max_flags(&self, call: &BlockingCall, flags: u32) -> Result<u32> {
        self.inner.all_reduce_max_flags(call, flags)
    }

    fn broadcast(&self, call: &BlockingCall, t: &mut Tensor, root: u32) -> Result<()> {
        self.inner.broadcast(call, t, root)
    }

    fn barrier(&self, call: &BlockingCall) -> Result<()> {
        self.inner.barrier(call)
    }

    fn rank(&self) -> u32 {
        self.inner.rank()
    }

    fn world(&self) -> u32 {
        self.inner.world()
    }

    fn bind_agreement(&self, digest: String) -> Result<()> {
        self.inner.bind_agreement(digest)
    }
}

/// Arm the member's NEXT rank body for `job_id`: with `chaos`, its `Peer`
/// is wrapped in a [`ChaosRank`] (the fault at [`CHAOS_AT_REDUCE`]) and
/// the test holds the [`ChaosControl`]; without, the body runs the real
/// `Peer` bare. The body itself is the REAL one the member's handler
/// spawns at admission — it reconstructs the job from the row, resumes
/// from the job's checkpoint in the shared store if one exists, and trains
/// rank 1 over the session's own link.
fn arm_rank_1(job_id: &str, chaos: bool) -> Option<ChaosControl> {
    if !chaos {
        return None;
    }
    let (seam, control) = chaos_channels();
    training_test_hooks::wrap_member_collective(
        job_id,
        Box::new(move |inner| Arc::new(ChaosRank::new(inner, seam))),
    );
    Some(control)
}

/// Every rank body's end recorded for `job_id` once at least `count` have
/// been (`training_test_hooks::rank_outcomes_for`: `(rank, the
/// `RankOutcome`'s Debug)`), within a bound. A body whose task died with
/// its member's runtime never records one.
async fn wait_rank_ends(job_id: &str, count: usize) -> Vec<(u32, String)> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let ends = training_test_hooks::rank_outcomes_for(job_id);
        if ends.len() >= count {
            return ends;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{count} rank-body end(s) within the bound; recorded: {ends:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn assert_trained(end: &(u32, String)) {
    assert_eq!(end.0, 1);
    assert!(
        end.1.starts_with("Trained { artifact_digest: \""),
        "rank 1's body completed: {}",
        end.1
    );
}

fn assert_failed_naming(end: &(u32, String), what: &str) {
    assert_eq!(end.0, 1);
    assert!(
        end.1.starts_with("Failed { reason: ") && end.1.contains(what),
        "rank 1's body ended Failed naming {what:?}: {}",
        end.1
    );
}

async fn submit(coordinator: &Coordinator) -> String {
    coordinator
        .session
        .run_training_spec(two_rank_spec())
        .await
        .expect("a two-rank job submits")
        .job_id
        .clone()
}

/// The end the coordinator recorded for `attempt`: `(description,
/// ordinal)`.
fn end_of(job_id: &str, attempt: u32) -> (String, usize) {
    let ends = training_test_hooks::coordinator_ends_for(job_id);
    let mut hits = ends.iter().filter(|(a, _, _)| *a == attempt);
    let (_, end, ordinal) = hits
        .next()
        .unwrap_or_else(|| panic!("attempt {attempt} recorded no end: {ends:?}"));
    assert!(
        hits.next().is_none(),
        "attempt {attempt} recorded more than one end: {ends:?}"
    );
    (end.clone(), *ordinal)
}

/// The row a RETIRED attempt leaves behind: `running`, nothing terminal,
/// `attempts` unchanged, the lease exactly as the split says.
fn assert_retired(after: &Row, attempt: i32, released: bool) {
    assert_eq!(after.status, "running", "no terminal write: {after:?}");
    assert_eq!(after.error, None, "no terminal write: {after:?}");
    assert_eq!(after.attempts, attempt, "attempts move only at a claim");
    if released {
        assert_eq!(
            after.releases, 1,
            "Released: the lease is handed back (OPS D10)"
        );
        assert_eq!(after.lease_expires_at, None, "Released: lease NULL");
    } else {
        assert_eq!(
            after.releases, 0,
            "a rank failure spends the attempt: the lease is never released"
        );
        assert!(
            after.lease_expires_at.is_some(),
            "the lease is left to expire, never released: {after:?}"
        );
    }
}

/// The successor attempt completes over a fresh gang: one model whose
/// bytes equal an uninterrupted `LocalGang` run of the same fixture.
async fn assert_completed_like_the_reference(coordinator: &Coordinator, job_id: &str, tag: &str) {
    let after = row(&coordinator.session, job_id).await;
    assert_eq!(after.status, "completed", "{after:?}");
    assert_eq!(after.error, None);
    assert_eq!(
        after.next_assembly_after, None,
        "Success resets the cooldown"
    );
    let published = published_adapter_bytes(&coordinator.session, job_id).await;
    let reference = reference_rank0_adapter_bytes(&coordinator.session, tag, pairs_loader).await;
    assert!(!published.is_empty());
    assert_eq!(
        published, reference,
        "the successor's published adapter must equal an uninterrupted LocalGang run"
    );
    let models = coordinator.session.catalog().list_models().await.unwrap();
    assert_eq!(
        models
            .iter()
            .filter(|m| m
                .model_id
                .starts_with(&format!("jammi:fine-tuned:{job_id}")))
            .count(),
        1,
        "exactly one model row after the retry"
    );
}

/// The member's stream drops mid-round (its process dies): the coordinator's
/// round fails on the transport, the attempt ends `LinkFault` — recorded
/// `Unavailable` (cooled, not counted) — and the lease is LEFT TO EXPIRE:
/// `releases` stays 0, the row is `running` with no terminal write until
/// reclaim requeues it, and the successor's claim is attempt 2 (an attempt
/// spent). The restarted member is listed again; attempt 2 resumes from
/// epoch 1's checkpoint and publishes the reference bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_stream_dropped_mid_round_retires_the_attempt_spent_and_a_new_gang_completes_it() {
    let fleet = Fleet::new();
    let coordinator = Coordinator::start(&fleet, RANK_TIMEOUT_SECS).await;
    coordinator.add_pairs_source(&fleet).await;
    let mut member = Member::start(&fleet).await;
    let job_id = submit(&coordinator).await;

    // ── attempt 1: rank 1 dies at epoch 2's first optimizer step ──────────────
    let control = arm_rank_1(&job_id, true).expect("chaos");
    let record = coordinator.claim(Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    let injector = tokio::spawn(async move {
        control.told.await.expect("the chaos point is reached");
        member.kill();
        control
            .resume
            .send(Resume::Fail)
            .expect("the rank thread waits");
        member
    });
    coordinator.run(record).await;
    let mut member = injector.await.expect("injector");

    let (end, ordinal) = end_of(&job_id, 1);
    assert_eq!(ordinal, 10, "LinkFault: {end}");
    assert!(
        end.contains("all_reduce_sum: round")
            && (end.contains("the stream failed") || end.contains("the stream ended")),
        "the round the member died in fails on the transport, mid-run: {end}"
    );
    let after_one = row(&coordinator.session, &job_id).await;
    assert_retired(&after_one, 1, false);
    assert_eq!(after_one.assembly_failures, 0, "Unavailable is not counted");
    assert!(
        after_one.next_assembly_after.is_some(),
        "Unavailable is cooled: {after_one:?}"
    );
    assert!(
        coordinator
            .session
            .artifact_store()
            .fetch_resume_checkpoint(None, &job_id)
            .await
            .expect("resume read")
            .is_some(),
        "epoch 1's resume checkpoint exists when the attempt is retired in epoch 2"
    );
    // The body's task died with the member's runtime: its end is never
    // recorded (its blocking thread returned the chaos error into a dropped
    // handle).
    assert!(
        training_test_hooks::rank_outcomes_for(&job_id).is_empty(),
        "a body killed with its process records no end"
    );
    member.wait_slot_free(Duration::from_secs(5)).await;

    // ── attempt 2: the member is back; the lease expired; a new gang ────
    member.restart().await;
    let started = tokio::time::Instant::now();
    let record = coordinator
        .claim(COORDINATOR_LEASE + Duration::from_secs(10))
        .await;
    assert_eq!(
        record.attempts, 2,
        "the successor's claim spends the attempt"
    );
    assert_eq!(record.releases, 0);
    coordinator.run(record).await;
    let (end, ordinal) = end_of(&job_id, 2);
    assert_eq!(ordinal, 12, "attempt 2 publishes: {end}");
    assert!(
        started.elapsed() >= Duration::from_millis(500),
        "the lease was left to expire, never handed back"
    );
    let ends = wait_rank_ends(&job_id, 1).await;
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_trained(&ends[0]);
    let after_two = row(&coordinator.session, &job_id).await;
    assert_eq!(after_two.attempts, 2);
    assert_eq!(after_two.releases, 0);
    assert_completed_like_the_reference(&coordinator, &job_id, "drop").await;
}

/// The member goes silent past `[worker] rank_timeout_secs` mid-round: the
/// coordinator's round deadline retires the attempt (`LinkFault` naming the
/// timeout), every member is ended (the slot frees on the coordinator's
/// `Cancel` while the rank thread is still stalled), the attempt is spent,
/// and the next gang completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_silent_past_the_rank_timeout_retires_the_attempt_spent_and_a_new_gang_completes_it(
) {
    let fleet = Fleet::new();
    let coordinator = Coordinator::start(&fleet, RANK_TIMEOUT_SECS).await;
    coordinator.add_pairs_source(&fleet).await;
    let member = Member::start(&fleet).await;
    let job_id = submit(&coordinator).await;

    // ── attempt 1: rank 1 stalls at epoch 2's first optimizer step ────────────
    let control = arm_rank_1(&job_id, true).expect("chaos");
    let record = coordinator.claim(Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    let started = tokio::time::Instant::now();
    coordinator.run(record).await;
    let waited = started.elapsed();
    control.told.await.expect("the chaos point was reached");

    let (end, ordinal) = end_of(&job_id, 1);
    assert_eq!(ordinal, 10, "LinkFault: {end}");
    assert!(
        end.contains("all_reduce_sum: round")
            && end.contains(&format!(
                "timed out after {RANK_TIMEOUT_SECS}s waiting for rank 1"
            )),
        "the deadline names the silent rank and the round: {end}"
    );
    assert!(
        waited >= Duration::from_secs(RANK_TIMEOUT_SECS),
        "the attempt waited out the deadline: {waited:?}"
    );
    let after_one = row(&coordinator.session, &job_id).await;
    assert_retired(&after_one, 1, false);
    assert_eq!(after_one.assembly_failures, 0);
    assert!(after_one.next_assembly_after.is_some());
    // The coordinator ended the member's session (the stream close) while
    // its rank thread is still stalled: the slot frees regardless.
    member.wait_slot_free(Duration::from_secs(5)).await;
    control
        .resume
        .send(Resume::Fail)
        .expect("the rank body waits");
    let ends = wait_rank_ends(&job_id, 1).await;
    assert_failed_naming(&ends[0], "chaos");

    // ── attempt 2 ───────────────────────────────────────────────────────
    let record = coordinator
        .claim(COORDINATOR_LEASE + Duration::from_secs(10))
        .await;
    assert_eq!(record.attempts, 2);
    assert_eq!(record.releases, 0);
    coordinator.run(record).await;
    assert_eq!(end_of(&job_id, 2).1, 12, "attempt 2 publishes");
    let ends = wait_rank_ends(&job_id, 2).await;
    assert_trained(&ends[1]);
    let after_two = row(&coordinator.session, &job_id).await;
    assert_eq!((after_two.attempts, after_two.releases), (2, 0));
    assert_completed_like_the_reference(&coordinator, &job_id, "silent").await;
}

/// The member's host DRAINs mid-round (a rolling restart of the peer tier):
/// its held session ends `Aborted{Drain}`, the coordinator reads it typed
/// (`MemberAborted{Drain}`), records `Drain` (neither counted nor cooled)
/// and RELEASES the lease first (`releases + 1`, lease NULL): the row is
/// claimable at once — the next attempt lands within one poll, never a
/// lease window — over a fresh member, and completes. Net attempts: zero
/// (`attempts - releases == 1` at completion) — OPS D10.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_aborted_drain_mid_round_releases_the_lease_and_the_next_attempt_completes_at_once(
) {
    let fleet = Fleet::new();
    let coordinator = Coordinator::start(&fleet, RANK_TIMEOUT_SECS).await;
    coordinator.add_pairs_source(&fleet).await;
    let draining = Member::start(&fleet).await;
    let job_id = submit(&coordinator).await;

    // ── attempt 1: the member host drains at rank 1's first step of epoch 2 ────────
    let control = arm_rank_1(&job_id, true).expect("chaos");
    let record = coordinator.claim(Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    let injector = tokio::spawn(async move {
        control.told.await.expect("the chaos point is reached");
        draining.drain().await;
        // The rank body enters its round on a session that has ended:
        // the severed link ends it.
        control
            .resume
            .send(Resume::Proceed)
            .expect("the rank body waits");
        draining
    });
    coordinator.run(record).await;
    let draining = injector.await.expect("injector");

    let (end, ordinal) = end_of(&job_id, 1);
    assert_eq!(ordinal, 9, "MemberAborted: {end}");
    assert_eq!(end, "rank 1 ended its session: Aborted(Drain)");
    let after_one = row(&coordinator.session, &job_id).await;
    assert_retired(&after_one, 1, true);
    assert_eq!(after_one.assembly_failures, 0, "Drain is not counted");
    assert_eq!(
        after_one.next_assembly_after, None,
        "Drain is not cooled: the next attempt is admissible at once"
    );
    let ends = wait_rank_ends(&job_id, 1).await;
    assert_failed_naming(&ends[0], "nothing applied");
    draining.wait_slot_free(Duration::from_secs(5)).await;

    // ── attempt 2: a fresh member (the restarted host); claimable now ───
    let restarted = Member::start(&fleet).await;
    let started = tokio::time::Instant::now();
    let record = coordinator.claim(Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < COORDINATOR_LEASE,
        "a released row is claimable within one poll, never a lease window"
    );
    assert_eq!(record.attempts, 2);
    assert_eq!(record.releases, 1);
    coordinator.run(record).await;
    assert_eq!(end_of(&job_id, 2).1, 12, "attempt 2 publishes");
    let ends = wait_rank_ends(&job_id, 2).await;
    assert_trained(&ends[1]);
    let listings = training_test_hooks::assembly_listings_for(&job_id);
    assert_eq!(listings.len(), 2);
    assert_eq!(
        listings[0].1,
        vec![(1, draining.session.instance_id().to_string())]
    );
    assert_eq!(
        listings[1].1,
        vec![(1, restarted.session.instance_id().to_string())],
        "the draining host is never listed again; the restarted one is"
    );
    let after_two = row(&coordinator.session, &job_id).await;
    assert_eq!(
        (after_two.attempts, after_two.releases),
        (2, 1),
        "zero net attempts: attempts - releases == 1"
    );
    assert_completed_like_the_reference(&coordinator, &job_id, "drain").await;
    drop(draining);
}

/// Split brain — an older attempt's stale runner: attempt 1's coordinator
/// loses its lease (its keeper dies) while its rank 1 is held and stalled;
/// the lease expires, a SECOND coordinator reclaims and claims attempt 2 and
/// dials the same member, whose `RunRank` at the greater attempt takes the
/// slot from the elder hold (U5a-2's fence); the elder session is refuted
/// at its next tick, so the stale coordinator's round ends and its attempt
/// writes NOTHING (its outcome is a moved claim; its terminal arm is the
/// lease-lost one); attempt 2 completes with `attempts == 2` — the fence
/// took the slot on the first dial, never a refused dial and a third
/// attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_older_attempts_stale_runner_is_fenced_by_the_successor_and_writes_nothing() {
    let fleet = Fleet::new();
    // The elder waits on its stalled rank far beyond the successor's fence.
    let elder = Coordinator::start(&fleet, 60).await;
    elder.add_pairs_source(&fleet).await;
    // The member host both attempts dial: its handler spawns the elder's
    // body and, later, the successor's.
    let _member = Member::start(&fleet).await;
    let job_id = submit(&elder).await;

    // ── attempt 1 on the elder: rank 1 stalls; the elder's keeper dies ──
    let control = arm_rank_1(&job_id, true).expect("chaos");
    let record = elder.claim(Duration::from_secs(5)).await;
    assert_eq!(record.attempts, 1);
    let elder_session = Arc::clone(&elder.session);
    let keeper_killer = tokio::spawn(async move {
        let ChaosControl { told, resume } = control;
        told.await.expect("the chaos point is reached");
        elder_session.lease_keeper().kill_thread_for_test();
        resume
    });
    let elder_run = {
        let elder = &elder;
        async move { elder.run(record).await }
    };

    // ── the successor: reclaims once the elder's lease expired ──────────
    let successor = Coordinator::start(&fleet, RANK_TIMEOUT_SECS).await;
    let successor_run = async {
        let resume = keeper_killer.await.expect("killer");
        let record = successor
            .claim(COORDINATOR_LEASE + Duration::from_secs(10))
            .await;
        assert_eq!(
            record.attempts, 2,
            "reclaimed after the elder's lease expired"
        );
        assert_eq!(record.releases, 0);
        successor.run(record).await;
        resume
    };
    let ((), resume) = tokio::join!(elder_run, successor_run);

    // The elder's attempt: refuted through the member (its session was
    // superseded and refuted at the next tick) or cancelled through its
    // own lost lease — either way no write reached the row.
    let (end, ordinal) = end_of(&job_id, 1);
    assert!(
        ordinal == 7 || end == "rank 1 ended its session: Aborted(Refuted)",
        "the stale runner ends on the fence or on its lost lease: {end} ({ordinal})"
    );
    assert_eq!(
        end_of(&job_id, 2).1,
        12,
        "the successor's attempt publishes"
    );
    resume
        .send(Resume::Fail)
        .expect("the elder's rank body waits");
    // Two bodies ran on the member for this job: the elder attempt's,
    // retired by the test, and the successor's, which completed — in
    // either recording order.
    let ends = wait_rank_ends(&job_id, 2).await;
    assert_eq!(ends.len(), 2, "{ends:?}");
    let failed = ends
        .iter()
        .find(|e| e.1.starts_with("Failed"))
        .unwrap_or_else(|| panic!("the elder's body ended Failed: {ends:?}"));
    assert_failed_naming(failed, "chaos");
    let trained = ends
        .iter()
        .find(|e| e.1.starts_with("Trained"))
        .unwrap_or_else(|| panic!("the successor's body completed: {ends:?}"));
    assert_trained(trained);

    let after = row(&successor.session, &job_id).await;
    assert_eq!(
        (after.attempts, after.releases),
        (2, 0),
        "the fence took the slot on the successor's first dial: no third attempt"
    );
    assert_completed_like_the_reference(&successor, &job_id, "split").await;
    let done = successor
        .session
        .catalog()
        .get_job(&job_id)
        .await
        .expect("read");
    assert_eq!(
        done.claimed_by.as_deref(),
        Some(successor.worker.worker_id()),
        "the successor, never the stale runner, completed the row"
    );
}
