//! The round protocol's server-side edges: the `[server.limits]
//! max_message_bytes` decode cap on EVERY listener (and every service of the
//! `peer_bind` listener), stated in `encoded_len()` terms and pinned at
//! exactly `n`, `n − 1` and `n + 1`; and `gang_rounds`'s seam — a round
//! frame delivered through `RoundInbox::deliver` reaches the member's
//! `Peer`, and `dial_member` builds the coordinator's capped link — driven
//! through one real round over a loopback listener whose handler mirrors
//! the hold loop's shape.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::{Device, Tensor};
use futures::Stream;
use jammi_ai::fine_tune::collective::{BlockingCall, Collective, LocalGang, MemberLink, Peer};
use jammi_db::catalog::instance::PeerAddr;
use jammi_db::config::LimitsConfig;
use jammi_server::grpc::gang_rounds::{dial_member, member_link, RoundInbox};
use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
use jammi_wire::proto::gang::gang_service_server::{GangService, GangServiceServer};
use jammi_wire::proto::gang::{rank_control, rank_event, Admitted, Assign, RankControl, RankEvent};
use jammi_wire::proto::job::job_service_client::JobServiceClient;
use jammi_wire::proto::job::CancelJobRequest;
use jammi_wire::proto::peer::peer_service_client::PeerServiceClient;
use jammi_wire::proto::peer::SegmentSearchRequest;
use prost::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt;
use tonic::transport::server::TcpIncoming;
use tonic::{Code, Request, Response, Status};

use crate::common::grpc::{channel, peer_bind_config, start_engine_server_from_config};

/// Grow-and-trim a string field until the enclosing message's
/// `encoded_len()` is exactly `target` — the length the gRPC frame header
/// carries and the cap compares.
fn sized<M: Message>(target: usize, build: impl Fn(String) -> M) -> M {
    let mut field = "x".repeat(target);
    loop {
        let message = build(field.clone());
        let len = message.encoded_len();
        if len == target {
            return message;
        }
        assert!(len > target, "the message overhead cannot be negative");
        field.pop();
    }
}

fn assign_sized(target: usize) -> RankControl {
    sized(target, |job_id| RankControl {
        control: Some(rank_control::Control::Assign(Assign {
            job_id,
            attempt: 1,
            rank: 0,
            world: 1,
            coordinator_instance_id: "c".into(),
        })),
    })
}

/// (d) The SAME configured cap, on both listeners and on both services of
/// the `peer_bind` listener: a frame of exactly `n` encoded bytes decodes
/// (the handler answers with its own status — never `OutOfRange`), so does
/// one of `n − 1`, and one of `n + 1` is refused `OutOfRange` naming the
/// CONFIGURED cap. RED at base for the peer listener: its `Routes` carried
/// no setter, so `n + 1` decoded there under tonic's 4 MiB default.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_of_exactly_max_message_bytes_decodes_on_every_listener_and_one_more_byte_is_refused_naming_the_configured_cap(
) {
    let cap = 2048usize;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cfg = peer_bind_config(dir.path());
    cfg.server.limits = LimitsConfig {
        max_message_bytes: cap as u64,
        ..LimitsConfig::default()
    };
    let server = start_engine_server_from_config(cfg, Some(dir)).await;
    let refusal = format!("the limit is: {cap} bytes");

    for (delta, decodes) in [(-1i64, true), (0, true), (1, false)] {
        let target = (cap as i64 + delta) as usize;

        // The peer listener, `GangService`: the handler decodes the Assign
        // and refuses admission (no such job) — a status that is NOT the
        // codec's.
        let mut gang = GangServiceClient::new(channel(server.peer_addr).await);
        let frame = assign_sized(target);
        assert_eq!(frame.encoded_len(), target);
        let status = gang
            .run_rank(tokio_stream::iter(vec![frame]))
            .await
            .expect_err("every arm ends in a status here");
        if decodes {
            assert_eq!(
                status.code(),
                Code::FailedPrecondition,
                "GangService: a {target}-byte frame under a {cap}-byte cap must decode: {status}"
            );
        } else {
            assert_eq!(status.code(), Code::OutOfRange, "GangService: {status}");
            assert!(
                status.message().contains(&refusal),
                "GangService: the refusal must name the CONFIGURED cap: {status}"
            );
        }

        // The peer listener, `PeerService`: the handler decodes and answers
        // for the unknown table — again not the codec's status.
        let mut peer = PeerServiceClient::new(channel(server.peer_addr).await);
        let request = sized(target, |table_name| SegmentSearchRequest {
            table_name,
            ..Default::default()
        });
        let status = peer
            .segment_search(request)
            .await
            .expect_err("an unknown table is refused");
        if decodes {
            assert_ne!(status.code(), Code::OutOfRange, "PeerService: {status}");
        } else {
            assert_eq!(status.code(), Code::OutOfRange, "PeerService: {status}");
            assert!(status.message().contains(&refusal), "PeerService: {status}");
        }

        // The public listener: the same value, the same edge.
        let mut jobs = JobServiceClient::new(channel(server.public_addr).await);
        let request = sized(target, |job_id| CancelJobRequest { job_id });
        let outcome = jobs.cancel_job(request).await;
        if decodes {
            assert!(
                !matches!(&outcome, Err(status) if status.code() == Code::OutOfRange),
                "public: a {target}-byte message under a {cap}-byte cap must decode: {outcome:?}"
            );
        } else {
            let status = outcome.expect_err("public: refused");
            assert_eq!(status.code(), Code::OutOfRange, "public: {status}");
            assert!(status.message().contains(&refusal), "public: {status}");
        }
    }
}

// ── The seam: a hold-loop-shaped handler, RoundInbox, dial_member ───────────

struct HoldLoopShaped {
    hand_off: Mutex<Option<oneshot::Sender<MemberLink>>>,
}

#[tonic::async_trait]
impl GangService for HoldLoopShaped {
    type RunRankStream = Pin<Box<dyn Stream<Item = Result<RankEvent, Status>> + Send + 'static>>;

    async fn run_rank(
        &self,
        request: Request<tonic::Streaming<RankControl>>,
    ) -> Result<Response<Self::RunRankStream>, Status> {
        let mut inbound = request.into_inner();
        let Some(Ok(RankControl {
            control: Some(rank_control::Control::Assign(_)),
        })) = inbound.next().await
        else {
            return Err(Status::invalid_argument("Assign first"));
        };
        let (events, stream) = mpsc::channel::<Result<RankEvent, Status>>(64);
        events
            .send(Ok(RankEvent {
                event: Some(rank_event::Event::Admitted(Admitted {})),
            }))
            .await
            .map_err(|_| Status::internal("stream gone"))?;
        let (inbox, link) = member_link(events).map_err(|e| Status::internal(e.to_string()))?;
        // The hold loop's inbound arm, reduced to its round-frame branch:
        // every round frame goes to the inbox; a transport error is
        // reported to it; anything else (a second Assign, a Cancel) is the
        // session's business and is not delivered.
        tokio::spawn(async move {
            while let Some(item) = inbound.next().await {
                match item {
                    Ok(frame) if RoundInbox::is_round_frame(&frame) => {
                        if !inbox.deliver(frame).await {
                            break;
                        }
                    }
                    Ok(_session_frame) => {}
                    Err(status) => {
                        inbox.fail(&status).await;
                        break;
                    }
                }
            }
        });
        let slot = self.hand_off.lock().unwrap().take().expect("one admission");
        let _ = slot.send(link);
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(stream),
        )))
    }
}

fn matrix(rows: usize, cols: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32 * 0.29).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).expect("tensor")
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn step(c: &dyn Collective, call: &BlockingCall) -> (Vec<u32>, Vec<u32>, u32) {
    let rank = c.rank();
    let gathered = c
        .all_gather(call, &matrix(2, 4, rank as f32), &[2, 2])
        .expect("gather");
    let mut sum = vec![matrix(16, 16, rank as f32 + 1.0)];
    c.all_reduce_sum(call, &mut sum).expect("reduce");
    let flags = c.all_reduce_max_flags(call, 1 << rank).expect("flags");
    (bits(&gathered), bits(&sum[0]), flags)
}

/// Round frames delivered through `RoundInbox::deliver` reach the member's
/// `Peer`, and `dial_member` builds the coordinator's capped link: one real
/// round of every tensor verb over the loopback stream equals `Local`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_round_delivered_through_the_inbox_and_dialed_through_dial_member_equals_local() {
    let (tx, rx) = oneshot::channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GangServiceServer::new(HoldLoopShaped {
                hand_off: Mutex::new(Some(tx)),
            }))
            .serve_with_incoming(TcpIncoming::from(listener))
            .await
            .expect("serve");
    });
    let cap = 1024usize;
    let member = BlockingCall::spawn_blocking(move |call| {
        let link = rx.blocking_recv().expect("admitted");
        let peer = Peer::member(1, 2, link, Device::Cpu, cap).expect("member");
        step(&peer, &call)
    });
    let peer_addr = PeerAddr::parse(&addr.to_string()).expect("peer addr");
    let link = dial_member(
        &peer_addr,
        Assign {
            job_id: "seam".into(),
            attempt: 1,
            rank: 1,
            world: 2,
            coordinator_instance_id: "coordinator".into(),
        },
        cap,
    )
    .await
    .expect("dialed and admitted");
    let coordinator = Peer::coordinator(vec![link], Device::Cpu, cap)
        .expect("coordinator")
        .with_timeout(Duration::from_secs(20))
        .expect("timeout");
    let coordinator = BlockingCall::spawn_blocking(move |call| step(&coordinator, &call));
    let (member, coordinator) = tokio::join!(member, coordinator);
    let (member, coordinator) = (member.expect("member"), coordinator.expect("coordinator"));

    let gang = Arc::new(LocalGang::new(vec![Device::Cpu; 2]).expect("gang"));
    let local: Vec<_> = (0..2)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            BlockingCall::spawn_thread(move |call| step(&local, &call))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().expect("local"))
        .collect();
    assert_eq!(coordinator, local[0]);
    assert_eq!(member, local[1]);
    assert_eq!(coordinator.2, 0b10, "the control word crossed the seam");
}

// ── The REAL handler: an admitted session's inbox and link ─────────────────

/// The REAL `GangServer` over `engine`, mounted on its own loopback
/// listener with the `test-hooks` link tap registered BEFORE the instance
/// moves into tonic's wrapper — the same `max_decoding_message_size` the
/// `peer_bind` listener applies. Returns the address and the tap's
/// receiver. A long lease so no park bound or freshness margin cuts the
/// session under the round; the re-verification tick stays at `HEARTBEAT`.
#[cfg(feature = "test-hooks")]
async fn mount_real_gang_server(
    engine: Arc<jammi_ai::session::InferenceSession>,
    cap: usize,
) -> (std::net::SocketAddr, mpsc::UnboundedReceiver<MemberLink>) {
    use jammi_server::grpc::gang::GangServer;

    let gang = GangServer::new(
        engine,
        Duration::from_secs(60),
        crate::gang_service::HEARTBEAT,
    );
    let links = gang.take_member_links();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GangServiceServer::new(gang).max_decoding_message_size(cap))
            .serve_with_incoming(TcpIncoming::from(listener))
            .await
            .expect("serve");
    });
    (addr, links)
}

/// Through the REAL `GangServer::run_rank` on a loopback listener: a
/// `world_size == 2` job whose every I-GANG determinant holds admits the
/// coordinator's `dial_member` call; the admitted session's hold loop
/// delivers one round's frames (`RoundResult`/`RoundChunk`/`RoundCommit`,
/// chunked under a 1 KiB cap) to its `RoundInbox`, and a `Peer` member
/// built over the session's OWN `MemberLink` — taken through the
/// `test-hooks` tap, the rank body's future seat — folds every tensor verb
/// byte-for-byte to what `Local` folds. Mutation proof: a hold loop whose
/// `dispatch_round_frame` refuses round frames as protocol violations (the
/// base tree) ends the session with a trailer at the first `RoundResult`,
/// and the coordinator's round faults instead of folding.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_round_through_the_real_run_rank_handler_reaches_the_member_link_and_equals_local() {
    let server = crate::gang_service::start_no_worker_server().await;
    let (attempt, before, _ready) = crate::gang_service::world_two_ready(
        &server,
        crate::gang_service::tenant(0xb1),
        "job-w2-real-round",
        "coord-w2-real-round",
    )
    .await;
    let cap = 1024usize;
    let (addr, mut links) = mount_real_gang_server(Arc::clone(&server.engine), cap).await;

    let member = BlockingCall::spawn_blocking(move |call| {
        let link = links
            .blocking_recv()
            .expect("the admitted session offered its MemberLink to the tap");
        let peer = Peer::member(1, 2, link, Device::Cpu, cap).expect("member");
        step(&peer, &call)
    });
    let peer_addr = PeerAddr::parse(&addr.to_string()).expect("peer addr");
    let link = dial_member(
        &peer_addr,
        Assign {
            job_id: "job-w2-real-round".into(),
            attempt,
            rank: 1,
            world: 2,
            coordinator_instance_id: "coord-w2-real-round".into(),
        },
        cap,
    )
    .await
    .expect("every I-GANG determinant holds: dialed and admitted through the real handler");
    let coordinator = Peer::coordinator(vec![link], Device::Cpu, cap)
        .expect("coordinator")
        .with_timeout(Duration::from_secs(20))
        .expect("timeout");
    let coordinator = BlockingCall::spawn_blocking(move |call| step(&coordinator, &call));
    let (member, coordinator) = tokio::join!(member, coordinator);
    let (member, coordinator) = (member.expect("member"), coordinator.expect("coordinator"));

    let gang = Arc::new(LocalGang::new(vec![Device::Cpu; 2]).expect("gang"));
    let local: Vec<_> = (0..2)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            BlockingCall::spawn_thread(move |call| step(&local, &call))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().expect("local"))
        .collect();
    assert_eq!(
        coordinator, local[0],
        "the coordinator's fold equals Local's rank 0"
    );
    assert_eq!(
        member, local[1],
        "the member's fold, over the session's own link, equals Local's rank 1"
    );
    assert_eq!(
        coordinator.2, 0b10,
        "the control word crossed the real handler"
    );
    // The peer wrote nothing terminal on the job row under the round.
    assert_eq!(
        crate::gang_service::row_facts(&server, "job-w2-real-round").await,
        before
    );
}

/// The delivery arm's refusal: a round frame on an admitted stream whose
/// `MemberLink` is gone — its owner (here the tap) dropped it, so
/// `RoundInbox::deliver` reports no member — ends the session with a
/// `FailedPrecondition` trailer naming the closed link, never buffering a
/// round nobody will read and never a bare close. A `world_size == 1`
/// session suffices: the hold loop is the same on every world size.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_round_frame_after_the_member_link_closed_ends_the_session_with_a_trailer() {
    use jammi_wire::proto::gang::RoundFault;

    let server = crate::gang_service::start_no_worker_server().await;
    crate::gang_service::fresh_coordinator(&server, "coord-link-closed").await;
    let attempt = crate::gang_service::submit_and_claim(
        &server,
        "job-link-closed",
        "coord-link-closed",
        Duration::from_secs(300),
        crate::gang_service::WORLD1_SPEC,
    )
    .await;
    let before = crate::gang_service::row_facts(&server, "job-link-closed").await;
    let (addr, mut links) = mount_real_gang_server(Arc::clone(&server.engine), 1024).await;
    let mut rank = crate::gang_service::open_rank_at(
        addr,
        crate::gang_service::assign_frame_full(
            "job-link-closed",
            attempt,
            0,
            1,
            "coord-link-closed",
        ),
    )
    .await
    .expect("every determinant holds: admitted");
    crate::gang_service::expect_admitted(&mut rank).await;
    let link = links.recv().await.expect("the session offered its link");
    drop(link);
    rank.outbound
        .send(RankControl {
            control: Some(rank_control::Control::RoundFault(RoundFault {
                round: 0,
                detail: "the coordinator gave up".into(),
            })),
        })
        .await
        .expect("the admitted stream is open");
    let err = crate::gang_service::next_event(&mut rank.events, Duration::from_secs(5))
        .await
        .expect_err("a round frame with no member to read it is a status, never an event");
    assert_eq!(err.code(), Code::FailedPrecondition, "{err}");
    assert!(
        err.message().contains("member link has closed"),
        "the trailer names the closed link: {err}"
    );
    assert_eq!(
        crate::gang_service::row_facts(&server, "job-link-closed").await,
        before
    );
}
