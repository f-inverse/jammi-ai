//! Hermetic oracles for the `Peer` collective: a whole gang in one process,
//! every rank on its own OS thread, the links over plain channels with a
//! TAP on either direction of any member's link so a fault can be injected
//! at an exact frame (a cut, a swallowed frame, a rewritten descriptor).
//! The round code under test is the same code a tonic stream drives; the
//! wire itself (prost encode/decode, the listener cap, the client cap) is
//! `tests/it/peer_gang.rs`'s.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use jammi_db::store::manifest::{parquet_leaves, LeafKey};
use jammi_wire::proto::gang::{
    rank_control, rank_event, AbortReason, Aborted, RankControl, RankEvent,
};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::mpsc;

use super::peer::{verify_leaves, CoordinatorLink, LinkFault, MemberLink, Peer, RankReadFault};
use super::{BlockingCall, Collective, Local, LocalGang};

// ── Wiring ──────────────────────────────────────────────────────────────────

/// What a tap does with one frame: send these frames on (the frame itself
/// to forward, none to swallow, several to inject); cut the link now; or
/// close the link's INBOUND side first (every later send by the far rank
/// fails from this instant) and then forward these frames — the shape that
/// makes "the stream is gone by the time the coordinator sends the commit"
/// deterministic rather than a race with the forwarder's exit.
enum Action<F> {
    Send(Vec<F>),
    Cut,
    CloseThenSend(Vec<F>),
}

type Tap<F> = Box<dyn FnMut(F) -> Action<F> + Send>;

/// Move frames from `rx` to `tx` through an optional tap; ending (a cut, or
/// either side gone) drops both ends, which is what a stream's end looks
/// like to the ranks.
fn forward<F: Send + 'static>(
    handle: &Handle,
    mut rx: mpsc::Receiver<F>,
    tx: mpsc::Sender<std::result::Result<F, LinkFault>>,
    mut tap: Option<Tap<F>>,
) {
    handle.spawn(async move {
        while let Some(frame) = rx.recv().await {
            let frames = match tap.as_mut() {
                None => vec![frame],
                Some(tap) => match tap(frame) {
                    Action::Send(frames) => frames,
                    Action::Cut => return,
                    Action::CloseThenSend(frames) => {
                        rx.close();
                        frames
                    }
                },
            };
            for frame in frames {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
            }
        }
    });
}

#[derive(Default)]
struct Taps {
    /// Coordinator → member, keyed by member rank.
    to_member: Vec<(u32, Tap<RankControl>)>,
    /// Member → coordinator, keyed by member rank.
    to_coordinator: Vec<(u32, Tap<RankEvent>)>,
}

/// A `world`-rank gang over channels: rank 0 the coordinator, ranks
/// `1..world` members, every link built inside `rt`'s context.
fn build_gang(
    rt: &Runtime,
    world: u32,
    cap: usize,
    timeout: Duration,
    mut taps: Taps,
) -> (Arc<Peer>, Vec<Arc<Peer>>) {
    let _guard = rt.enter();
    let handle = rt.handle();
    let mut member_links = Vec::new();
    let mut members = Vec::new();
    for rank in 1..world {
        let (c_out_tx, c_out_rx) = mpsc::channel::<RankControl>(64);
        let (m_in_tx, m_in_rx) = mpsc::channel(64);
        let tap = taps
            .to_member
            .iter()
            .position(|(r, _)| *r == rank)
            .map(|i| taps.to_member.remove(i).1);
        forward(handle, c_out_rx, m_in_tx, tap);

        let (m_out_tx, m_out_rx) = mpsc::channel::<RankEvent>(64);
        let (c_in_tx, c_in_rx) = mpsc::channel(64);
        let tap = taps
            .to_coordinator
            .iter()
            .position(|(r, _)| *r == rank)
            .map(|i| taps.to_coordinator.remove(i).1);
        forward(handle, m_out_rx, c_in_tx, tap);

        member_links.push(CoordinatorLink::from_channels(rank, c_in_rx, c_out_tx).expect("link"));
        let link = MemberLink::from_channels(m_in_rx, m_out_tx).expect("link");
        members.push(Arc::new(
            Peer::member(rank, world, link, Device::Cpu, cap)
                .expect("member")
                .with_timeout(timeout)
                .expect("timeout"),
        ));
    }
    let coordinator = Arc::new(
        Peer::coordinator(member_links, Device::Cpu, cap)
            .expect("coordinator")
            .with_timeout(timeout)
            .expect("timeout"),
    );
    (coordinator, members)
}

/// Run `body` on every rank of a gang, each on its own witness thread, and
/// return the results in rank order.
fn run_ranks<T, F>(coordinator: &Arc<Peer>, members: &[Arc<Peer>], body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(&Peer, BlockingCall) -> T + Send + Sync + 'static,
{
    let body = Arc::new(body);
    let handles: Vec<_> = std::iter::once(coordinator)
        .chain(members.iter())
        .map(|peer| {
            let peer = Arc::clone(peer);
            let body = Arc::clone(&body);
            BlockingCall::spawn_thread(move |call| body(&peer, call))
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("a rank thread panicked"))
        .collect()
}

/// The same `body` over a `Local` gang of the same world, for the twin.
fn run_local<T, F>(world: u32, body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(&Local, BlockingCall) -> T + Send + Sync + 'static,
{
    let gang = LocalGang::with_timeout(vec![Device::Cpu; world as usize], Duration::from_secs(30))
        .expect("gang");
    let body = Arc::new(body);
    let handles: Vec<_> = (0..world)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            let body = Arc::clone(&body);
            BlockingCall::spawn_thread(move |call| body(&local, call))
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("a rank thread panicked"))
        .collect()
}

/// The exact bits of a tensor's elements, any of the three dtypes.
fn raw_bits(t: &Tensor) -> Vec<u32> {
    let flat = t.flatten_all().expect("flatten");
    match t.dtype() {
        DType::F32 => flat
            .to_vec1::<f32>()
            .expect("f32")
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        DType::F16 => flat
            .to_vec1::<half::f16>()
            .expect("f16")
            .into_iter()
            .map(|v| u32::from(v.to_bits()))
            .collect(),
        DType::BF16 => flat
            .to_vec1::<half::bf16>()
            .expect("bf16")
            .into_iter()
            .map(|v| u32::from(v.to_bits()))
            .collect(),
        other => panic!("no bit view for {other:?}"),
    }
}

fn matrix(rows: usize, cols: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32 * 0.37).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).expect("tensor")
}

fn typed(t: &Tensor, dtype: DType) -> Tensor {
    t.to_dtype(dtype).expect("dtype")
}

fn runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime")
}

// ── (a) The fold over the wire equals Local's fold, byte for byte ───────────

/// One step's worth of every verb at every dtype, on a rank of `world`:
/// the per-rank result bits, so the Peer gang and the Local gang can be
/// compared rank by rank.
fn every_verb_at_every_dtype(c: &dyn Collective, call: &BlockingCall) -> Vec<Vec<u32>> {
    let rank = c.rank();
    let world = c.world() as usize;
    let mut out = Vec::new();
    // Unequal counts including a zero-row rank.
    let counts: Vec<usize> = (0..world).map(|r| if r == 1 { 0 } else { r + 1 }).collect();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mine = typed(
            &matrix(counts[rank as usize], 3, rank as f32 * 10.0 + 1.0),
            dtype,
        );
        let gathered = c.all_gather(call, &mine, &counts).expect("all_gather");
        out.push(raw_bits(&gathered));
    }
    let mut tensors = vec![
        typed(&matrix(2, 3, rank as f32 + 0.5), DType::F32),
        typed(&matrix(4, 1, rank as f32 * 3.0 + 0.1), DType::F16),
        typed(&matrix(3, 2, rank as f32 - 1.25), DType::BF16),
    ];
    c.all_reduce_sum(call, &mut tensors)
        .expect("all_reduce_sum");
    for t in &tensors {
        out.push(raw_bits(t));
    }
    let flags = c.all_reduce_max_flags(call, 1u32 << rank).expect("flags");
    out.push(vec![flags]);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mut t = typed(&matrix(2, 2, rank as f32 * 7.0), dtype);
        c.broadcast(call, &mut t, 1).expect("broadcast");
        out.push(raw_bits(&t));
    }
    c.barrier(call).expect("barrier");
    out
}

#[test]
fn peer_fold_over_the_wire_equals_local_fold_byte_for_byte_at_f32_f16_bf16() {
    let rt = runtime();
    let (coordinator, members) =
        build_gang(&rt, 3, 1 << 20, Duration::from_secs(20), Taps::default());
    let over_peer = run_ranks(&coordinator, &members, |peer, call| {
        every_verb_at_every_dtype(peer, &call)
    });
    let over_local = run_local(3, |local, call| every_verb_at_every_dtype(local, &call));
    assert_eq!(
        over_peer, over_local,
        "every rank's result bits over the wire must equal the in-process fold's"
    );
    // The control: the fixture is not degenerate (a rank's sum differs from
    // its own input, a gather is longer than a slice).
    assert_ne!(over_peer[0][3], raw_bits(&matrix(2, 3, 0.5)));
    assert_eq!(over_peer[0][0].len(), 4 * 3, "counts [1, 0, 3] × 3 columns");
}

/// Two runs of the same Peer gang over the same inputs produce the same
/// bytes — the determinism the equal-topology oracle rests on.
#[test]
fn peer_collectives_are_deterministic_across_two_gangs() {
    let rt = runtime();
    let mut runs = Vec::new();
    for _ in 0..2 {
        let (coordinator, members) =
            build_gang(&rt, 3, 1 << 20, Duration::from_secs(20), Taps::default());
        runs.push(run_ranks(&coordinator, &members, |peer, call| {
            every_verb_at_every_dtype(peer, &call)
        }));
    }
    assert_eq!(runs[0], runs[1]);
}

// ── (b) Trainer-shaped streams at W=2: Peer equals Local ────────────────────

/// The tensor stream one trainer step emits on `rank` of a two-rank gang:
/// the gathered encoder outputs (contrastive: `[rows, 8]` post-projection
/// embeddings and the `[rows]` pair scores; regression: the `[rows, 1]`
/// head output and the `[rows]` targets), the per-variable gradients in
/// canonical order (a LoRA `A` in f32 and a `B` in bf16), the lockstep
/// control word, and the rank-0 scaler broadcast.
fn trainer_step(
    c: &dyn Collective,
    call: &BlockingCall,
    step: usize,
    contrastive: bool,
) -> Vec<Vec<u32>> {
    let rank = c.rank() as usize;
    let mut out = Vec::new();
    let counts = if step.is_multiple_of(2) {
        vec![2, 2]
    } else {
        vec![2, 1]
    };
    let rows = counts[rank];
    let base = (step * 10 + rank) as f32;
    let (features, per_row) = if contrastive {
        (
            matrix(rows, 8, base),
            matrix(rows, 1, base + 0.5).squeeze(1).expect("scores"),
        )
    } else {
        (
            matrix(rows, 1, base),
            matrix(rows, 1, base + 0.5).squeeze(1).expect("targets"),
        )
    };
    let gathered = c.all_gather(call, &features, &counts).expect("gather");
    let gathered_rows = c.all_gather(call, &per_row, &counts).expect("gather");
    out.push(raw_bits(&gathered));
    out.push(raw_bits(&gathered_rows));
    let mut grads = vec![
        matrix(4, 8, base * 0.01),
        typed(&matrix(8, 4, base * 0.02), DType::BF16),
    ];
    c.all_reduce_sum(call, &mut grads).expect("reduce");
    for g in &grads {
        out.push(raw_bits(g));
    }
    out.push(vec![c
        .all_reduce_max_flags(call, if step == 1 && rank == 1 { 0b10 } else { 0 })
        .expect("flags")]);
    let mut scaler = matrix(1, 2, base);
    c.broadcast(call, &mut scaler, 0).expect("broadcast");
    out.push(raw_bits(&scaler));
    out
}

#[test]
fn peer_w2_fold_matches_local_w2_on_regression_and_contrastive_streams() {
    let rt = runtime();
    for contrastive in [true, false] {
        let (coordinator, members) =
            build_gang(&rt, 2, 1 << 20, Duration::from_secs(20), Taps::default());
        let over_peer = run_ranks(&coordinator, &members, move |peer, call| {
            (0..2)
                .map(|step| trainer_step(peer, &call, step, contrastive))
                .collect::<Vec<_>>()
        });
        let over_local = run_local(2, move |local, call| {
            (0..2)
                .map(|step| trainer_step(local, &call, step, contrastive))
                .collect::<Vec<_>>()
        });
        assert_eq!(
            over_peer,
            over_local,
            "{} fixture: Peer-W2 bytes must equal Local-W2 on every rank at every step",
            if contrastive {
                "contrastive"
            } else {
                "regression"
            }
        );
    }
}

// ── (c) Deadlines and the two-phase round ───────────────────────────────────

/// Every wait — the coordinator's for a contribution, a member's for the
/// result, a member's for the commit — expires at the gang deadline with an
/// error naming the round.
#[test]
fn every_round_wait_on_every_rank_expires_at_the_gang_deadline_naming_the_round() {
    let rt = runtime();
    let timeout = Duration::from_millis(400);

    // (i) The coordinator waits for a silent member; the other member, which
    // did contribute, is told by the coordinator's fault.
    let (coordinator, members) = build_gang(&rt, 3, 1 << 20, timeout, Taps::default());
    let started = Instant::now();
    let results = run_ranks(&coordinator, &members, |peer, call| {
        if peer.rank() == 2 {
            return "silent".to_string();
        }
        peer.barrier(&call)
            .expect_err("round 0 never completes")
            .to_string()
    });
    assert!(
        started.elapsed() < timeout * 4,
        "the waits must end at the deadline"
    );
    assert!(
        results[0].contains("timed out") && results[0].contains("round 0"),
        "the coordinator's wait must expire naming the round: {}",
        results[0]
    );
    // The contributing member shares the gang deadline: it is told by the
    // coordinator's fault, or its own wait for the result expires at the
    // same instant — either way the round is named and nothing applied.
    assert!(
        results[1].contains("round 0")
            && (results[1].contains("faulted by a peer") || results[1].contains("timed out")),
        "the contributing member's wait must end naming the round: {}",
        results[1]
    );

    // (ii) A member waits for a coordinator that never enters the round.
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, timeout, Taps::default());
    let results = run_ranks(&coordinator, &members, |peer, call| {
        if peer.rank() == 0 {
            return String::new();
        }
        peer.barrier(&call)
            .expect_err("round 0 never completes")
            .to_string()
    });
    assert!(
        results[1].contains("timed out") && results[1].contains("round 0"),
        "the member's wait for the result must expire naming the round: {}",
        results[1]
    );

    // (iii) A member waits for a commit that never arrives (swallowed in
    // transit); the round must not apply on it.
    let taps = Taps {
        to_member: vec![(
            1,
            Box::new(|frame: RankControl| match frame.control {
                Some(rank_control::Control::RoundCommit(_)) => Action::Send(Vec::new()),
                other => Action::Send(vec![RankControl { control: other }]),
            }),
        )],
        to_coordinator: Vec::new(),
    };
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, timeout, taps);
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let mut t = matrix(1, 2, 1.0);
        let before = raw_bits(&t);
        let outcome = peer.broadcast(&call, &mut t, 0);
        (outcome.err().map(|e| e.to_string()), raw_bits(&t) == before)
    });
    let member = results[1]
        .0
        .as_ref()
        .expect("the member never sees the commit");
    assert!(
        member.contains("timed out")
            && member.contains("round 0")
            && member.contains("the coordinator's commit"),
        "unexpected message: {member}"
    );
    assert!(
        results[1].1,
        "the member applied a round it was never told to commit"
    );
}

/// A member's stream ends between the coordinator's publish and its ACK:
/// no rank applies the round, and every rank's fault names it.
#[test]
fn a_disconnect_between_publish_and_the_last_ack_leaves_no_rank_applied_and_names_the_round() {
    let rt = runtime();
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            2,
            Box::new(|frame: RankEvent| match frame.event {
                Some(rank_event::Event::RoundAck(_)) => Action::Cut,
                other => Action::Send(vec![RankEvent { event: other }]),
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 3, 1 << 20, Duration::from_secs(5), taps);
    let started = Instant::now();
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let mut tensors = vec![matrix(2, 2, peer.rank() as f32 + 1.0)];
        let before = raw_bits(&tensors[0]);
        let outcome = peer.all_reduce_sum(&call, &mut tensors);
        (
            outcome
                .expect_err("round 0 must not apply on any rank")
                .to_string(),
            raw_bits(&tensors[0]) == before,
        )
    });
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the ranks must not wait out the deadline"
    );
    for (rank, (message, unchanged)) in results.iter().enumerate() {
        assert!(
            *unchanged,
            "rank {rank} applied round 0 after a fault before the last ACK"
        );
        assert!(
            message.contains("round 0"),
            "rank {rank}'s fault must name the round: {message}"
        );
    }
    assert!(
        results[0].0.contains("rank 2") && results[0].0.contains("no rank applies round 0"),
        "the coordinator names the lost rank: {}",
        results[0].0
    );
}

/// The commit-phase oracle: rank 2's stream is gone between the commit
/// fan-out to rank 1 and the fan-out to rank 2. Fatal on every rank — the
/// coordinator applies nothing, rank 2 applies nothing, rank 1 (which the
/// commit reached) applied round 0 but its next contribution is refused,
/// and no rank's next round is accepted.
#[test]
fn a_fault_during_the_commit_fan_out_is_fatal_on_every_rank_and_no_next_contribution_is_accepted() {
    let rt = runtime();
    // The result payload still reaches rank 2 (so it ACKs, so the
    // coordinator reaches the fan-out), but rank 2's inbound is closed from
    // that instant: the coordinator's commit to rank 1 lands, its commit to
    // rank 2 fails.
    let taps = Taps {
        to_member: vec![(
            2,
            Box::new(|frame: RankControl| match frame.control {
                Some(rank_control::Control::RoundResult(payload)) => {
                    Action::CloseThenSend(vec![RankControl {
                        control: Some(rank_control::Control::RoundResult(payload)),
                    }])
                }
                other => Action::Send(vec![RankControl { control: other }]),
            }),
        )],
        to_coordinator: Vec::new(),
    };
    let deadline = Duration::from_secs(5);
    let (coordinator, members) = build_gang(&rt, 3, 1 << 20, deadline, taps);
    let started = Instant::now();
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let flags = peer.all_reduce_max_flags(&call, 1 << peer.rank());
        let next = peer.barrier(&call);
        (
            flags.map_err(|e| e.to_string()),
            next.expect_err("no rank's next round is accepted")
                .to_string(),
        )
    });
    assert!(
        started.elapsed() < deadline,
        "no rank may wait out the deadline"
    );
    let coordinator_error = results[0]
        .0
        .as_ref()
        .expect_err("the coordinator applies nothing");
    assert!(
        coordinator_error.contains("round 0") && coordinator_error.contains("commit fan-out"),
        "unexpected: {coordinator_error}"
    );
    let rank2_error = results[2]
        .0
        .as_ref()
        .expect_err("rank 2 never saw the commit");
    assert!(rank2_error.contains("round 0"), "unexpected: {rank2_error}");
    // Rank 1 is the one rank the commit reached: it applied round 0. That is
    // the one residual state the protocol admits — and its next contribution
    // is refused by the coordinator's fault, promptly.
    assert_eq!(
        results[1].0,
        Ok(0b100),
        "rank 1 took the commit for round 0 (max of 1, 2, 4)"
    );
    for (rank, (_, next)) in results.iter().enumerate() {
        assert!(
            next.contains("round 0") || next.contains("the gang has already failed"),
            "rank {rank}'s next round must be refused on the round-0 fault: {next}"
        );
    }
    assert!(
        results[1].1.contains("faulted by a peer"),
        "rank 1's next contribution must be answered by the coordinator's fault: {}",
        results[1].1
    );
}

// ── (e) Descriptor disagreement: symmetric, naming both sides ───────────────

fn assert_symmetric_disagreement(results: &[String], what: &str) {
    for (rank, message) in results.iter().enumerate() {
        assert!(
            message.contains("disagree about what this round computes")
                || message.contains("faulted by a peer"),
            "{what}: rank {rank} was not refused symmetrically: {message}"
        );
        assert!(
            message.matches("Descriptor {").count() >= 2,
            "{what}: rank {rank}'s refusal must name BOTH descriptors: {message}"
        );
    }
}

#[test]
fn a_root_disagreement_is_a_typed_refusal_naming_both_sides_on_every_rank() {
    let rt = runtime();
    let (coordinator, members) =
        build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), Taps::default());
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let mut t = matrix(1, 1, 0.0);
        peer.broadcast(&call, &mut t, peer.rank())
            .expect_err("each rank names itself root")
            .to_string()
    });
    assert_symmetric_disagreement(&results, "root");
}

#[test]
fn a_counts_disagreement_is_a_typed_refusal_naming_both_sides_on_every_rank() {
    let rt = runtime();
    let (coordinator, members) =
        build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), Taps::default());
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let counts = if peer.rank() == 0 {
            vec![1, 1]
        } else {
            vec![1, 2]
        };
        let mine = matrix(counts[peer.rank() as usize], 2, 0.0);
        peer.all_gather(&call, &mine, &counts)
            .expect_err("the ranks disagree about the partition")
            .to_string()
    });
    assert_symmetric_disagreement(&results, "counts");
}

#[test]
fn an_agreement_slot_disagreement_is_a_typed_refusal_naming_both_sides_on_every_rank() {
    let rt = runtime();
    // Bound vs bound-differently, and bound vs unbound.
    for (a, b) in [
        (Some("sha256:a"), Some("sha256:b")),
        (Some("sha256:a"), None),
    ] {
        let (coordinator, members) =
            build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), Taps::default());
        let coordinator = Arc::new(
            Arc::try_unwrap(coordinator)
                .expect("sole handle")
                .with_agreement(a.expect("rank 0 binds")),
        );
        let members: Vec<Arc<Peer>> = members
            .into_iter()
            .map(|m| {
                let m = Arc::try_unwrap(m).expect("sole handle");
                Arc::new(match b {
                    Some(b) => m.with_agreement(b),
                    None => m,
                })
            })
            .collect();
        let results = run_ranks(&coordinator, &members, |peer, call| {
            peer.barrier(&call)
                .expect_err("the ranks bound different agreements")
                .to_string()
        });
        assert_symmetric_disagreement(&results, "agreement");
        assert!(
            results[0].contains("sha256:a"),
            "the refusal quotes the bound digest: {}",
            results[0]
        );
    }
}

/// A member whose wire descriptor names an unknown verb (a value outside
/// the closed enum) is refused naming both sides — on both ranks.
#[test]
fn an_unknown_wire_verb_is_a_typed_refusal_naming_both_sides_on_every_rank() {
    let rt = runtime();
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            1,
            Box::new(|frame: RankEvent| match frame.event {
                Some(rank_event::Event::RoundContribution(mut payload)) => {
                    payload.descriptor.as_mut().expect("descriptor").verb = 99;
                    Action::Send(vec![RankEvent {
                        event: Some(rank_event::Event::RoundContribution(payload)),
                    }])
                }
                other => Action::Send(vec![RankEvent { event: other }]),
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), taps);
    let results = run_ranks(&coordinator, &members, |peer, call| {
        peer.barrier(&call)
            .expect_err("an unknown wire verb is refused")
            .to_string()
    });
    for (rank, message) in results.iter().enumerate() {
        assert!(
            message.contains("unknown wire verb 99") && message.contains("Descriptor {"),
            "rank {rank}: the refusal must name the unknown verb and this rank's descriptor: \
             {message}"
        );
    }
}

/// A contribution stamped with a stale round index disagrees on the round
/// field of the descriptor — a typed refusal naming both, never a fold.
#[test]
fn a_contribution_from_a_stale_round_is_a_typed_disagreement_naming_both_sides() {
    let rt = runtime();
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            1,
            Box::new(|frame: RankEvent| match frame.event {
                Some(rank_event::Event::RoundContribution(mut payload)) => {
                    payload.descriptor.as_mut().expect("descriptor").round = 7;
                    Action::Send(vec![RankEvent {
                        event: Some(rank_event::Event::RoundContribution(payload)),
                    }])
                }
                other => Action::Send(vec![RankEvent { event: other }]),
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), taps);
    let results = run_ranks(&coordinator, &members, |peer, call| {
        peer.barrier(&call)
            .expect_err("a stale round never folds")
            .to_string()
    });
    assert_symmetric_disagreement(&results, "round");
    assert!(
        results[0].contains("round: 7") && results[0].contains("round: 0"),
        "both round indices are named: {}",
        results[0]
    );
}

/// A rank's own argument refusal (a counts vector that cannot describe the
/// gang) faults the peers promptly, never leaving them to the deadline.
#[test]
fn a_domain_refusal_on_one_rank_faults_its_peers_before_the_deadline() {
    let rt = runtime();
    let deadline = Duration::from_secs(4);
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, deadline, Taps::default());
    let started = Instant::now();
    let results = run_ranks(&coordinator, &members, |peer, call| {
        if peer.rank() == 1 {
            return peer
                .all_gather(&call, &matrix(1, 2, 0.0), &[1])
                .expect_err("one entry cannot describe a gang of two")
                .to_string();
        }
        peer.barrier(&call)
            .expect_err("the member's refusal faults the round")
            .to_string()
    });
    assert!(
        started.elapsed() < deadline / 4,
        "the coordinator waited {:?} for a fault the member had already raised",
        started.elapsed()
    );
    assert!(
        results[0].contains("counts has 1 entries"),
        "unexpected: {}",
        results[0]
    );
}

/// A dtype outside f32/f16/bf16 is refused at the seam on the calling rank,
/// before any descriptor exists, and its peers are faulted.
#[test]
fn a_dtype_outside_f32_f16_bf16_is_refused_at_the_seam_and_faults_the_peers() {
    let rt = runtime();
    let (coordinator, members) =
        build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), Taps::default());
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let mut tensors = vec![typed(&matrix(2, 2, 1.0), DType::F64)];
        peer.all_reduce_sum(&call, &mut tensors)
            .expect_err("f64 is not carried")
            .to_string()
    });
    for (rank, message) in results.iter().enumerate() {
        assert!(
            message.contains("f32, f16 and bf16 tensors only") && message.contains("F64"),
            "rank {rank}: unexpected: {message}"
        );
    }
}

/// Once faulted, every verb on every rank refuses promptly quoting the fault.
#[test]
fn every_verb_after_a_fault_refuses_promptly_on_every_rank_quoting_the_fault() {
    let rt = runtime();
    let deadline = Duration::from_secs(3);
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, deadline, Taps::default());
    let results = run_ranks(&coordinator, &members, move |peer, call| {
        let mut t = matrix(1, 1, 0.0);
        peer.broadcast(&call, &mut t, peer.rank())
            .expect_err("the control: a root disagreement faults the gang");
        let started = Instant::now();
        let mut sum = vec![matrix(1, 1, 0.0)];
        let attempts: Vec<String> = vec![
            peer.all_gather(&call, &matrix(1, 2, 0.0), &[1, 1])
                .expect_err("gather")
                .to_string(),
            peer.all_reduce_sum(&call, &mut sum)
                .expect_err("sum")
                .to_string(),
            peer.all_reduce_max_flags(&call, 0)
                .expect_err("flags")
                .to_string(),
            peer.broadcast(&call, &mut t, 0)
                .expect_err("broadcast")
                .to_string(),
            peer.barrier(&call).expect_err("barrier").to_string(),
        ];
        (attempts, started.elapsed())
    });
    for (rank, (attempts, elapsed)) in results.iter().enumerate() {
        assert!(
            *elapsed < deadline / 2,
            "rank {rank} parked instead of refusing"
        );
        for message in attempts {
            assert!(
                message.contains("the gang has already failed")
                    && message.contains("disagree about what this round computes"),
                "rank {rank}: unexpected: {message}"
            );
        }
    }
}

// ── Chunking, bounds, residency, gradients ──────────────────────────────────

/// A tensor larger than the message cap travels as chunks no larger than
/// the cap and reassembles byte-exact.
#[test]
fn a_tensor_larger_than_the_message_cap_travels_in_chunks_under_the_cap_and_reassembles_exact() {
    let rt = runtime();
    let cap = 1024;
    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            1,
            Box::new(move |frame: RankEvent| {
                if let Some(rank_event::Event::RoundChunk(chunk)) = &frame.event {
                    record.lock().unwrap().push(chunk.ipc.len());
                }
                Action::Send(vec![frame])
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 2, cap, Duration::from_secs(10), taps);
    let over_peer = run_ranks(&coordinator, &members, |peer, call| {
        let mut tensors = vec![matrix(64, 64, peer.rank() as f32)];
        peer.all_reduce_sum(&call, &mut tensors).expect("reduce");
        raw_bits(&tensors[0])
    });
    let over_local = run_local(2, |local, call| {
        let mut tensors = vec![matrix(64, 64, local.rank() as f32)];
        local.all_reduce_sum(&call, &mut tensors).expect("reduce");
        raw_bits(&tensors[0])
    });
    assert_eq!(over_peer, over_local);
    let seen = seen.lock().unwrap();
    assert!(
        seen.len() >= 17,
        "a 16 KiB tensor under a 1 KiB cap needs many chunks: {seen:?}"
    );
    assert!(
        seen.iter().all(|&len| len <= cap - 64),
        "every chunk must fit under the cap: {seen:?}"
    );
}

/// A peer announcing more bytes than its descriptor implies is refused
/// before they are buffered.
#[test]
fn a_payload_past_the_bound_its_descriptor_implies_is_refused_before_buffering() {
    let rt = runtime();
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            1,
            Box::new(|frame: RankEvent| match frame.event {
                Some(rank_event::Event::RoundContribution(mut payload)) => {
                    payload.chunk_count = 1_000_000;
                    Action::Send(vec![RankEvent {
                        event: Some(rank_event::Event::RoundContribution(payload)),
                    }])
                }
                Some(rank_event::Event::RoundChunk(chunk)) => {
                    // Flood: the same chunk many times over, re-indexed.
                    Action::Send(
                        (0..64)
                            .map(|i| RankEvent {
                                event: Some(rank_event::Event::RoundChunk(
                                    jammi_wire::proto::gang::RoundChunk {
                                        round: chunk.round,
                                        index: i,
                                        ipc: chunk.ipc.clone(),
                                    },
                                )),
                            })
                            .collect(),
                    )
                }
                other => Action::Send(vec![RankEvent { event: other }]),
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 2, 1024, Duration::from_secs(5), taps);
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let mut tensors = vec![matrix(4, 4, 1.0)];
        peer.all_reduce_sum(&call, &mut tensors)
            .expect_err("the flood is refused")
            .to_string()
    });
    assert!(
        results[0].contains("exceeds the") && results[0].contains("bound its descriptor implies"),
        "unexpected: {}",
        results[0]
    );
}

/// Results land on the rank's device, and only the rank's own gather slot
/// carries a gradient.
#[test]
fn results_land_on_the_ranks_device_and_only_the_own_gather_slot_is_attached() {
    let rt = runtime();
    let (coordinator, members) =
        build_gang(&rt, 3, 1 << 20, Duration::from_secs(10), Taps::default());
    let results = run_ranks(&coordinator, &members, |peer, call| {
        let rank = peer.rank();
        let var = candle_core::Var::from_tensor(&matrix(1, 2, rank as f32 + 1.0)).expect("var");
        let mine = var.as_tensor().affine(2.0, 0.0).expect("affine");
        let gathered = peer
            .all_gather(&call, &mine, &[1, 1, 1])
            .expect("all_gather");
        let on_device = gathered.device().same_device(peer.device());
        let loss = gathered.sum_all().expect("sum");
        let grads = loss.backward().expect("backward");
        (
            on_device,
            grads.get(&var).map(raw_bits),
            gathered.dims().to_vec(),
        )
    });
    for (rank, (on_device, grad, dims)) in results.iter().enumerate() {
        assert!(*on_device, "rank {rank}'s result is not on its device");
        assert_eq!(dims, &vec![3, 2]);
        assert_eq!(
            grad.as_deref(),
            Some(&[2.0f32.to_bits(), 2.0f32.to_bits()][..]),
            "rank {rank}'s gradient must be its OWN slot's, never scaled by the world size"
        );
    }
}

/// A member whose session ends (`Aborted`) mid-round faults the coordinator
/// naming the reason — and the coordinator records the reason TYPED, per
/// member link (`Peer::member_aborts`, the fact the coordinator body maps
/// through the assembly reason table), beside its permanent fault
/// (`Peer::fault`); a member records no aborts of its own.
#[test]
fn a_member_that_aborts_its_session_faults_the_coordinator_naming_the_reason() {
    let rt = runtime();
    let taps = Taps {
        to_member: Vec::new(),
        to_coordinator: vec![(
            1,
            Box::new(|frame: RankEvent| match frame.event {
                Some(rank_event::Event::RoundContribution(_)) => Action::Send(vec![RankEvent {
                    event: Some(rank_event::Event::Aborted(Aborted {
                        reason: AbortReason::StoreUnavailable as i32,
                    })),
                }]),
                other => Action::Send(vec![RankEvent { event: other }]),
            }),
        )],
    };
    let (coordinator, members) = build_gang(&rt, 2, 1 << 20, Duration::from_secs(5), taps);
    let results = run_ranks(&coordinator, &members, |peer, call| {
        peer.barrier(&call)
            .expect_err("the member aborted")
            .to_string()
    });
    assert!(
        results[0].contains("Aborted(StoreUnavailable)") && results[0].contains("round 0"),
        "unexpected: {}",
        results[0]
    );
    assert_eq!(
        coordinator.member_aborts(),
        vec![(1, AbortReason::StoreUnavailable as i32)],
        "the coordinator records the member's abort reason typed, on rank 1's link"
    );
    assert!(
        coordinator.fault().is_some(),
        "the round's fault is the coordinator's permanent state"
    );
    assert!(
        members
            .iter()
            .all(|member| member.member_aborts().is_empty()),
        "a member records no aborts: the record is the coordinator's"
    );
}

// ── Construction ────────────────────────────────────────────────────────────

#[test]
fn peer_construction_refuses_a_gang_of_one_a_rank_outside_the_gang_a_tiny_cap_and_a_zero_deadline()
{
    let rt = runtime();
    let _guard = rt.enter();
    let (_, in_rx) = mpsc::channel(1);
    let (out_tx, _) = mpsc::channel(1);
    let link = MemberLink::from_channels(in_rx, out_tx).expect("link");
    Peer::member(1, 1, link, Device::Cpu, 1 << 20).expect_err("world 1 never crosses the wire");
    let (_, in_rx) = mpsc::channel(1);
    let (out_tx, _) = mpsc::channel(1);
    let link = MemberLink::from_channels(in_rx, out_tx).expect("link");
    Peer::member(0, 2, link, Device::Cpu, 1 << 20).expect_err("rank 0 is the coordinator");
    let (_, in_rx) = mpsc::channel(1);
    let (out_tx, _) = mpsc::channel(1);
    let link = MemberLink::from_channels(in_rx, out_tx).expect("link");
    Peer::member(1, 2, link, Device::Cpu, 512).expect_err("a 512-byte cap fits no chunk");
    let (_, in_rx) = mpsc::channel(1);
    let (out_tx, _) = mpsc::channel(1);
    let link = MemberLink::from_channels(in_rx, out_tx).expect("link");
    Peer::member(1, 2, link, Device::Cpu, 1 << 20)
        .expect("member")
        .with_timeout(Duration::ZERO)
        .expect_err("a zero deadline");
    Peer::coordinator(Vec::new(), Device::Cpu, 1 << 20).expect_err("no members");
    let (_, in_rx) = mpsc::channel(1);
    let (out_tx, _) = mpsc::channel(1);
    let link = CoordinatorLink::from_channels(2, in_rx, out_tx).expect("link");
    Peer::coordinator(vec![link], Device::Cpu, 1 << 20).expect_err("a lone member must be rank 1");
}

#[test]
fn a_link_built_outside_a_runtime_context_is_refused() {
    let (_, in_rx) = mpsc::channel::<std::result::Result<RankControl, LinkFault>>(1);
    let (out_tx, _) = mpsc::channel::<RankEvent>(1);
    let error = MemberLink::from_channels(in_rx, out_tx).expect_err("no runtime here");
    assert!(error.to_string().contains("tokio runtime context"));
}

// ── (f) The rank's read path: one leaf at a time, member-scoped ─────────────

fn three_row_group_parquet() -> Vec<u8> {
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from((0..6).collect::<Vec<i64>>()))],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

#[test]
fn verify_leaves_reads_exactly_one_leaf_range_at_a_time_and_names_a_corrupted_leaf_member_scoped() {
    let bytes = three_row_group_parquet();
    let leaves = parquet_leaves(&bytes).expect("leaves");
    assert_eq!(leaves.len(), 3, "the fixture spans row groups");

    // Intact: every read is exactly one leaf's range, in order, and no read
    // is wider than the widest leaf.
    let reads = Mutex::new(Vec::new());
    verify_leaves(&leaves, |_, range| {
        reads.lock().unwrap().push(range.clone());
        Ok(bytes::Bytes::copy_from_slice(
            &bytes[range.start as usize..range.end as usize],
        ))
    })
    .expect("an intact object verifies");
    let reads = reads.into_inner().unwrap();
    let ranges: Vec<_> = leaves
        .iter()
        .map(|leaf| match &leaf.key {
            LeafKey::RowGroup { offset, length, .. } => *offset..*offset + *length,
            LeafKey::File { .. } => unreachable!(),
        })
        .collect();
    assert_eq!(
        reads, ranges,
        "one ranged read per leaf, never the whole object"
    );
    assert!(reads.iter().all(|r| (r.end - r.start) < bytes.len() as u64));

    // Row group 1 corrupted: the fault is member-scoped, names leaf 1, maps
    // to the wire's STORE_UNAVAILABLE, and no leaf after it is read.
    let LeafKey::RowGroup { offset, length, .. } = leaves[1].key.clone() else {
        unreachable!()
    };
    let mut tampered = bytes.clone();
    tampered[(offset + length / 2) as usize] ^= 0xff;
    let reads = Mutex::new(0usize);
    let fault = verify_leaves(&leaves, |_, range| {
        *reads.lock().unwrap() += 1;
        Ok(bytes::Bytes::copy_from_slice(
            &tampered[range.start as usize..range.end as usize],
        ))
    })
    .expect_err("a corrupted leaf is caught");
    match &fault {
        RankReadFault::StoreUnavailable { key, .. } => {
            assert!(
                matches!(key, Some(LeafKey::RowGroup { index: 1, .. })),
                "the fault names the corrupted leaf: {fault}"
            );
        }
    }
    assert_eq!(fault.abort_reason(), AbortReason::StoreUnavailable);
    assert_eq!(
        *reads.lock().unwrap(),
        2,
        "the verify stops at the first bad leaf"
    );
    assert!(fault.to_string().contains("member-scoped"));

    // A read error and a short read are the same member-scoped class.
    let fault = verify_leaves(&leaves, |_, _| Err("connection reset".into())).expect_err("io");
    assert!(matches!(fault, RankReadFault::StoreUnavailable { .. }));
    let fault =
        verify_leaves(&leaves, |_, _| Ok(bytes::Bytes::from_static(b"x"))).expect_err("short read");
    assert!(fault.to_string().contains("short read"));
}

/// An object-store driver that records the range of every read it serves
/// and delegates — the observation the bounded-memory claim needs: the
/// verify must read each leaf's range and nothing wider.
#[derive(Debug)]
struct RecordingStore {
    inner: Arc<dyn object_store::ObjectStore>,
    ranges: Arc<Mutex<Vec<Option<object_store::GetRange>>>>,
}

impl std::fmt::Display for RecordingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecordingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for RecordingStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.ranges.lock().unwrap().push(options.range.clone());
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[test]
fn verify_partition_leaves_over_a_file_store_reads_one_bounded_range_per_leaf_and_names_a_corrupted_leaf(
) {
    use jammi_db::storage::{JammiObjectStore, StorageUrl};
    let rt = runtime();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.parquet");
    let bytes = three_row_group_parquet();
    std::fs::write(&path, &bytes).expect("write");
    // The test builds its OWN driver directly (route 4 — no crate boundary
    // can seal direct `object_store` construction against paths this
    // process can already reach): a `LocalFileSystem` rooted at the tempdir
    // itself, never a bare `LocalFileSystem::new()` rooted at `/` — jammi-db
    // never hands this test a raw driver to wrap (`JammiObjectStore::open`
    // has no such seam since #588's closing audit removed `open_with`).
    let url = StorageUrl::parse("file:///data.parquet").expect("url");
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let driver = object_store::local::LocalFileSystem::new_with_prefix(dir.path())
        .expect("local driver rooted at the tempdir");
    let handle = JammiObjectStore::new(
        Arc::new(RecordingStore {
            inner: Arc::new(driver),
            ranges: Arc::clone(&ranges),
        }),
        url,
    );
    let leaves = parquet_leaves(&bytes).expect("leaves");

    rt.block_on(super::peer::verify_partition_leaves(&handle, &leaves))
        .expect("an intact object verifies through the store");
    // One BOUNDED read per leaf, exactly the leaf's range — never an
    // unbounded (whole-object) read, so memory is bounded by a row group.
    let recorded = ranges.lock().unwrap().clone();
    let expected: Vec<Option<object_store::GetRange>> = leaves
        .iter()
        .map(|leaf| match &leaf.key {
            LeafKey::RowGroup { offset, length, .. } => {
                Some(object_store::GetRange::Bounded(*offset..*offset + *length))
            }
            LeafKey::File { .. } => unreachable!(),
        })
        .collect();
    assert_eq!(recorded, expected, "every read is one leaf's bounded range");

    let LeafKey::RowGroup { offset, length, .. } = leaves[2].key.clone() else {
        unreachable!()
    };
    let mut tampered = bytes.clone();
    tampered[(offset + length / 2) as usize] ^= 0xff;
    std::fs::write(&path, &tampered).expect("write");
    let fault = rt
        .block_on(super::peer::verify_partition_leaves(&handle, &leaves))
        .expect_err("the corrupted leaf is caught");
    match &fault {
        RankReadFault::StoreUnavailable { key, .. } => assert!(
            matches!(key, Some(LeafKey::RowGroup { index: 2, .. })),
            "names leaf 2: {fault}"
        ),
    }
}
