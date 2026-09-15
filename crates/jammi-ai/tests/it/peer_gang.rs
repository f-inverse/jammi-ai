//! The `Peer` collective over a REAL `RunRank` stream: a loopback tonic
//! listener in this process serves a member whose handler mirrors the hold
//! loop's shape (admit, then hand every round frame to the member's link),
//! and a real `GangServiceClient` on the coordinator side dials it. What
//! this adds over the in-process oracles (`fine_tune::collective::peer_tests`)
//! is the wire itself: prost encode/decode of every round frame, chunks
//! that must each fit the listener's cap, and the client's own inbound cap.
//! No catalog, no object-store service: a tempdir `file://` root.
//!
//! The fixture gang is `world = 2` — rank 0 the coordinator (a
//! `spawn_blocking` thread in this test), rank 1 the member (a
//! `spawn_blocking` thread behind the served handler).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::{DType, Device, Tensor};
use futures::Stream;
use jammi_ai::fine_tune::collective::peer::{verify_partition_leaves, LinkFault};
use jammi_ai::fine_tune::collective::{
    BlockingCall, Collective, CoordinatorLink, LocalGang, MemberLink, Peer,
};
use jammi_db::store::manifest::{parquet_leaves, LeafKey};
use jammi_wire::proto::gang::gang_service_server::{GangService, GangServiceServer};
use jammi_wire::proto::gang::{
    rank_control, rank_event, Aborted, Admitted, Assign, RankControl, RankEvent, RoundFault,
};
use prost::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

/// What the served handler hands the member's thread once it admitted the
/// stream: the member's link, and the raw event sender for a scenario that
/// speaks frames the `Peer` would never send.
struct HandOff {
    link: MemberLink,
    raw: mpsc::Sender<Result<RankEvent, Status>>,
}

/// A `GangService` shaped like the admitted session's hold loop: the first
/// frame is the `Assign`, the answer is `Admitted`, and from then on every
/// inbound frame is delivered to the member's link while the link's
/// outbound events are the response stream.
struct MemberService {
    hand_off: Mutex<Option<oneshot::Sender<HandOff>>>,
}

#[tonic::async_trait]
impl GangService for MemberService {
    type RunRankStream = Pin<Box<dyn Stream<Item = Result<RankEvent, Status>> + Send + 'static>>;

    async fn run_rank(
        &self,
        request: Request<tonic::Streaming<RankControl>>,
    ) -> Result<Response<Self::RunRankStream>, Status> {
        let mut inbound = request.into_inner();
        match inbound.next().await {
            Some(Ok(RankControl {
                control: Some(rank_control::Control::Assign(_)),
            })) => {}
            other => {
                return Err(Status::invalid_argument(format!(
                    "the fixture expects an Assign first, got {other:?}"
                )))
            }
        }
        let (events_tx, events_rx) = mpsc::channel::<Result<RankEvent, Status>>(64);
        events_tx
            .send(Ok(RankEvent {
                event: Some(rank_event::Event::Admitted(Admitted {})),
            }))
            .await
            .map_err(|_| Status::internal("response stream gone"))?;

        // The hold loop's inbound arm: every later frame goes to the link.
        let (in_tx, in_rx) = mpsc::channel(64);
        tokio::spawn(async move {
            while let Some(item) = inbound.next().await {
                let item = item.map_err(|status| LinkFault(status.to_string()));
                let ended = item.is_err();
                if in_tx.send(item).await.is_err() || ended {
                    break;
                }
            }
        });
        // The link's outbound events ride the response stream.
        let (out_tx, mut out_rx) = mpsc::channel::<RankEvent>(64);
        let events = events_tx.clone();
        tokio::spawn(async move {
            while let Some(event) = out_rx.recv().await {
                if events.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        });
        let link = MemberLink::from_channels(in_rx, out_tx)
            .map_err(|e| Status::internal(e.to_string()))?;
        let slot = self
            .hand_off
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| Status::failed_precondition("one admission per fixture"))?;
        let _ = slot.send(HandOff {
            link,
            raw: events_tx,
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(events_rx),
        )))
    }
}

/// Serve one member on a loopback ephemeral port.
async fn serve_member() -> (SocketAddr, oneshot::Receiver<HandOff>) {
    let (tx, rx) = oneshot::channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let service = MemberService {
        hand_off: Mutex::new(Some(tx)),
    };
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GangServiceServer::new(service))
            .serve_with_incoming(TcpIncoming::from(listener))
            .await
            .expect("serve");
    });
    (addr, rx)
}

async fn dial(addr: SocketAddr, cap: usize) -> CoordinatorLink {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    CoordinatorLink::over_client(
        channel,
        Assign {
            job_id: "peer-gang".into(),
            attempt: 1,
            rank: 1,
            world: 2,
            coordinator_instance_id: "coordinator".into(),
        },
        cap,
    )
    .await
    .expect("admitted")
}

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
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32 * 0.41).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).expect("tensor")
}

/// One step of every verb at every dtype, with a gather and a reduce whose
/// tensors exceed the message cap (so they travel chunked).
fn every_verb(c: &dyn Collective, call: &BlockingCall) -> Vec<Vec<u32>> {
    let rank = c.rank();
    let mut out = Vec::new();
    let counts = [3usize, 2];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mine = matrix(counts[rank as usize], 512, rank as f32 * 10.0 + 1.0)
            .to_dtype(dtype)
            .expect("dtype");
        out.push(raw_bits(
            &c.all_gather(call, &mine, &counts).expect("gather"),
        ));
    }
    let mut tensors = vec![
        matrix(64, 64, rank as f32 + 0.5),
        matrix(64, 32, rank as f32 * 3.0 + 0.1)
            .to_dtype(DType::F16)
            .expect("f16"),
        matrix(32, 32, rank as f32 - 1.25)
            .to_dtype(DType::BF16)
            .expect("bf16"),
    ];
    c.all_reduce_sum(call, &mut tensors).expect("reduce");
    for t in &tensors {
        out.push(raw_bits(t));
    }
    out.push(vec![c
        .all_reduce_max_flags(call, 1 << rank)
        .expect("flags")]);
    let mut t = matrix(2, 2, rank as f32 * 7.0)
        .to_dtype(DType::BF16)
        .expect("bf16");
    c.broadcast(call, &mut t, 1).expect("broadcast");
    out.push(raw_bits(&t));
    c.barrier(call).expect("barrier");
    out
}

/// One gang over the loopback wire: the member runs `member_body` on its
/// own blocking thread once admitted, the coordinator runs
/// `coordinator_body` on another; both results come back.
async fn run_gang<M, C, T, U>(
    cap: usize,
    timeout: Duration,
    member_body: M,
    coordinator_body: C,
) -> (T, U)
where
    M: FnOnce(HandOff, BlockingCall) -> T + Send + 'static,
    C: FnOnce(&Peer, BlockingCall) -> U + Send + 'static,
    T: Send + 'static,
    U: Send + 'static,
{
    let (addr, hand_off) = serve_member().await;
    let member = BlockingCall::spawn_blocking(move |call| {
        let hand_off = hand_off.blocking_recv().expect("the member is admitted");
        member_body(hand_off, call)
    });
    let link = dial(addr, cap).await;
    let coordinator = Peer::coordinator(vec![link], Device::Cpu, cap)
        .expect("coordinator")
        .with_timeout(timeout)
        .expect("timeout");
    let coordinator =
        BlockingCall::spawn_blocking(move |call| coordinator_body(&coordinator, call));
    let (member, coordinator) = tokio::join!(member, coordinator);
    (
        member.expect("member thread"),
        coordinator.expect("coordinator thread"),
    )
}

fn member_peer(hand_off: HandOff, cap: usize, timeout: Duration) -> Peer {
    Peer::member(1, 2, hand_off.link, Device::Cpu, cap)
        .expect("member")
        .with_timeout(timeout)
        .expect("timeout")
}

/// (a) over the wire: every rank's bits equal the in-process fold's, with
/// every tensor above the cap travelling chunked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peer_fold_over_a_loopback_run_rank_stream_equals_local_at_f32_f16_bf16_with_chunking() {
    let cap = 4096;
    let timeout = Duration::from_secs(30);
    let (member, coordinator) = run_gang(
        cap,
        timeout,
        move |hand_off, call| every_verb(&member_peer(hand_off, cap, timeout), &call),
        |peer, call| every_verb(peer, &call),
    )
    .await;

    let gang = LocalGang::new(vec![Device::Cpu; 2]).expect("gang");
    let local: Vec<_> = (0..2)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            BlockingCall::spawn_thread(move |call| every_verb(&local, &call))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().expect("local rank"))
        .collect();
    assert_eq!(
        coordinator, local[0],
        "rank 0 over the wire equals Local rank 0"
    );
    assert_eq!(member, local[1], "rank 1 over the wire equals Local rank 1");
    // The control: the gather really crossed the cap (5 × 512 f32 = 10 KiB
    // against a 4 KiB cap) and is the full concatenation.
    assert_eq!(coordinator[0].len(), 5 * 512);
}

/// (d), the client's side: a member frame of exactly `max_message_bytes`
/// encoded bytes decodes on the coordinator's client, as does one of `n − 1`;
/// one of `n + 1` is refused naming the CONFIGURED cap — tonic's default
/// (4 MiB) is never what bounds this side.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_recv_cap_is_the_configured_cap_at_n_minus_1_n_and_n_plus_1_encoded_bytes() {
    let cap = 2048usize;
    for (delta, decodes) in [(-1i64, true), (0, true), (1, false)] {
        let target = (cap as i64 + delta) as usize;
        // A `RoundFault` frame sized to exactly `target` encoded bytes: the
        // detail string is what grows, and `encoded_len` is the length the
        // gRPC frame header carries and the cap compares.
        let mut detail = "x".repeat(target);
        let frame = loop {
            let frame = RankEvent {
                event: Some(rank_event::Event::RoundFault(RoundFault {
                    round: 0,
                    detail: detail.clone(),
                })),
            };
            let len = frame.encoded_len();
            if len == target {
                break frame;
            }
            assert!(len > target, "the frame overhead cannot be negative");
            detail.pop();
        };
        assert_eq!(frame.encoded_len(), target);

        let (_, coordinator) = run_gang(
            cap,
            Duration::from_secs(10),
            move |hand_off, _call| {
                let raw = hand_off.raw;
                let _link = hand_off.link;
                tokio::runtime::Handle::current()
                    .block_on(raw.send(Ok(frame)))
                    .expect("stream");
                // Keep the stream open until the coordinator has read it.
                std::thread::sleep(Duration::from_millis(500));
            },
            |peer, call| {
                peer.barrier(&call)
                    .expect_err("the frame is a fault either way")
                    .to_string()
            },
        )
        .await;
        if decodes {
            assert!(
                coordinator.contains("faulted by a peer") && coordinator.contains("xxxx"),
                "a {target}-byte frame under a {cap}-byte cap must decode: {coordinator}"
            );
        } else {
            assert!(
                coordinator.contains(&format!("the limit is: {cap} bytes"))
                    && coordinator.contains("the stream failed"),
                "a {target}-byte frame over a {cap}-byte cap must be refused naming the \
                 configured cap: {coordinator}"
            );
        }
    }
}

/// (c) over the wire: a member that never contributes expires the
/// coordinator's wait at the gang deadline, naming the round.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_silent_member_over_the_wire_expires_the_coordinators_wait_at_the_deadline_naming_the_round(
) {
    let timeout = Duration::from_millis(400);
    let started = std::time::Instant::now();
    let (_, coordinator) = run_gang(
        1 << 20,
        timeout,
        |hand_off, _call| {
            // Admitted, then silent; the stream stays open.
            std::thread::sleep(Duration::from_secs(2));
            drop(hand_off);
        },
        |peer, call| {
            peer.barrier(&call)
                .expect_err("round 0 never completes")
                .to_string()
        },
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(
        coordinator.contains("timed out")
            && coordinator.contains("round 0")
            && coordinator.contains("rank 1's contribution"),
        "unexpected: {coordinator}"
    );
}

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

/// (f) over the wire: the member verifies its partition's leaves one row
/// group at a time before its first collective; a corrupted row group is
/// caught there, MEMBER-scoped — the session ends `Aborted(StoreUnavailable)`
/// and the coordinator's first round faults naming it, having folded
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_corrupted_leaf_is_caught_on_the_member_before_any_collective_and_ends_the_session_member_scoped(
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path: PathBuf = dir.path().join("data.parquet");
    let bytes = three_row_group_parquet();
    let leaves = parquet_leaves(&bytes).expect("leaves");
    let LeafKey::RowGroup { offset, length, .. } = leaves[1].key.clone() else {
        unreachable!()
    };
    let mut tampered = bytes.clone();
    tampered[(offset + length / 2) as usize] ^= 0xff;
    std::fs::write(&path, &tampered).expect("write");
    let url =
        jammi_db::storage::StorageUrl::parse(&format!("file://{}", path.display())).expect("url");
    let handle = jammi_db::storage::JammiObjectStore::new(
        jammi_db::storage::build_object_store(&url, None).expect("driver"),
        url,
    );

    let cap = 1 << 20;
    let timeout = Duration::from_secs(10);
    let (member, coordinator) = run_gang(
        cap,
        timeout,
        move |hand_off, call| {
            // The rank's read path, before its first collective.
            let verdict = tokio::runtime::Handle::current()
                .block_on(verify_partition_leaves(&handle, &leaves));
            match verdict {
                Ok(()) => {
                    // Not reached with the tampered object; a clean object
                    // would go on to the collective here.
                    member_peer(hand_off, cap, timeout)
                        .barrier(&call)
                        .map(|()| "ran".to_string())
                        .unwrap_or_else(|e| e.to_string())
                }
                Err(fault) => {
                    tokio::runtime::Handle::current()
                        .block_on(hand_off.raw.send(Ok(RankEvent {
                            event: Some(rank_event::Event::Aborted(Aborted {
                                reason: fault.abort_reason() as i32,
                            })),
                        })))
                        .expect("stream");
                    drop(hand_off);
                    fault.to_string()
                }
            }
        },
        |peer, call| {
            peer.barrier(&call)
                .expect_err("the member aborted before contributing")
                .to_string()
        },
    )
    .await;
    assert!(
        member.contains("member-scoped") && member.contains("index: 1"),
        "the member's fault names the leaf and its scope: {member}"
    );
    assert!(
        coordinator.contains("Aborted(StoreUnavailable)")
            && coordinator.contains("waiting for rank 1's contribution"),
        "the coordinator faults on the session end before any fold: {coordinator}"
    );
}
